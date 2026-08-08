#!/usr/bin/env python3
"""Unified fair-comparison finetune entrypoint (both arms, all task families).

    torchrun --nproc_per_node=6 finetune/common/train.py \
        --dataset casia --model ghost --data_path data/casia \
        --output_dir finetune/ghost/casia/outputs --use_dali

Smoke (CPU, no DALI, synthetic data):
    PYTHONPATH=./.pylibs python3 finetune/common/train.py --dataset <d> --model ghost \
        --data_path finetune/_smoke_data/<d> --smoke

The ONLY model-dependent code is the model name + checkpoint, both chosen via config from
--model {baseline,ghost}. Everything else (data, aug, loss, engine, metrics) is shared —
that is the fairness invariant (EXPERIMENT_DESIGN.md §7).
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

# repo root on path (so `util`, `dali_data`, `finetune.*` all import)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from finetune.common.config import resolve
from finetune.common.models_finetune import build_model
from finetune.common.checkpoint import load_pretrained_encoder
from finetune.common.optim import build_optimizer
from finetune.common import distributed as D
from finetune.common import engine, heads, profile_cost as prof
from finetune.common.data import identity as ds_identity
from finetune.common.data import celeba_attr as ds_celeba
from finetune.common.data import lfw_verify as ds_lfw
from finetune.common.data import m3dmad_liveness as ds_liveness
from finetune.common.data import dali_loader

from timm.data import Mixup


# "higher is better" for each task's primary metric (for best-checkpoint tracking).
PRIMARY = {"identity": ("acc1", True), "attribute": ("mAP", True),
           "verification": ("auc", True), "liveness": ("acer", False)}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_task_data(cfg):
    """Return dict with train_ds/val_ds (and pair_eval for verification) + resolved nb_classes."""
    if cfg.task == "identity":
        train_ds, val_ds, n = ds_identity.build(cfg)
        return dict(train_ds=train_ds, val_ds=val_ds, nb_classes=n)
    if cfg.task == "liveness":
        train_ds, val_ds = ds_liveness.build(cfg)
        return dict(train_ds=train_ds, val_ds=val_ds, nb_classes=2)
    if cfg.task == "attribute":
        train_ds, val_ds = ds_celeba.build(cfg)
        return dict(train_ds=train_ds, val_ds=val_ds, nb_classes=40)
    if cfg.task == "verification":
        train_ds, n, pair_ds = ds_lfw.build(cfg)
        return dict(train_ds=train_ds, val_ds=pair_ds, nb_classes=n, is_pairs=True)
    raise KeyError(cfg.task)


def make_loader(dataset, batch_size, cfg, dist_info, shuffle, seed):
    sampler = None
    # Shard only the TRAIN set (shuffle=True). Eval runs the FULL val set on every rank so the
    # reported metric is computed over all of val, not a 1/world_size shard (only rank 0 prints).
    # Face val sets are small, so the redundant eval compute is negligible.
    if dist_info.distributed and shuffle:
        sampler = DistributedSampler(dataset, num_replicas=dist_info.world_size,
                                     rank=dist_info.rank, shuffle=shuffle, seed=seed)
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=(0 if cfg.smoke else cfg.num_workers),
        pin_memory=not cfg.smoke, drop_last=shuffle,
    ), sampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    # bimamba/forwardmamba are CUDA-only (mamba_ssm + causal_conv1d); baseline/ghost run anywhere.
    p.add_argument("--model", required=True, choices=["baseline", "ghost", "bimamba", "forwardmamba"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data_path", default="")
    p.add_argument("--output_dir", default="")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--accum_iter", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--use_dali", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dev_steps", type=int, default=0,
                   help="cap train/eval to N steps but keep GPU/bf16/DALI/checkpoint (real quick check)")
    args, unknown = p.parse_known_args()

    # passthrough overrides: --key value pairs matching FairConfig fields
    overrides = {}
    for k in ("data_path", "output_dir"):
        if getattr(args, k):
            overrides[k] = getattr(args, k)
    for k in ("batch_size", "accum_iter", "epochs"):
        if getattr(args, k) is not None:
            overrides[k] = getattr(args, k)
    if args.use_dali:
        overrides["use_dali"] = True
    it = iter(unknown)
    for tok in it:
        if tok.startswith("--"):
            key = tok[2:]
            val = next(it, "true")
            overrides[key] = _coerce(val)

    if args.smoke:
        overrides.update(dict(smoke=True, use_dali=False, epochs=1, batch_size=4,
                              accum_iter=1, amp_dtype="fp32", num_workers=0))

    cfg = resolve(args.dataset, args.model, seed=args.seed, **overrides)

    dist_info = D.init_distributed(force_cpu=cfg.smoke)
    device = dist_info.device
    set_seed(cfg.seed + dist_info.rank)
    torch.backends.cudnn.benchmark = True

    if D.is_main_process():
        print(f"[cfg] dataset={cfg.dataset} arm={args.model} model={cfg.model} task={cfg.task} "
              f"seed={cfg.seed} epochs={cfg.epochs} bs={cfg.batch_size} device={device}")

    # -------------------- data --------------------
    data = build_task_data(cfg)
    nb_classes = data["nb_classes"]
    is_pairs = data.get("is_pairs", False)

    use_dali = dali_loader.is_dali_eligible(cfg)
    dali_train_iter = dali_val_iter = None
    n_iter_per_epoch = None
    if use_dali:
        dali_train_iter, dali_val_iter, n_train = dali_loader.build_dali(
            cfg, cfg.batch_size, dist_info.local_rank, dist_info.world_size, cfg.seed)
        n_iter_per_epoch = max(1, n_train // dist_info.world_size // cfg.batch_size)
        train_iterable, val_iterable = dali_train_iter, dali_val_iter
        train_sampler = None
    else:
        train_loader, train_sampler = make_loader(
            data["train_ds"], cfg.batch_size, cfg, dist_info, shuffle=True, seed=cfg.seed)
        val_loader, _ = make_loader(
            data["val_ds"], cfg.batch_size, cfg, dist_info, shuffle=False, seed=cfg.seed)
        train_iterable, val_iterable = train_loader, val_loader

    # -------------------- model --------------------
    model = build_model(cfg.model, num_classes=nb_classes,
                        drop_path_rate=cfg.drop_path, global_pool=cfg.global_pool)
    ckpt = cfg.finetune if os.path.isabs(cfg.finetune) else os.path.join(_REPO, cfg.finetune)
    if not cfg.smoke and os.path.exists(ckpt):
        load_pretrained_encoder(model, ckpt, global_pool=cfg.global_pool, strict_check=True)
    elif not cfg.smoke:
        print(f"[warn] pretrain checkpoint not found: {ckpt} (training from scratch)")
    model.to(device)

    model_without_ddp = model
    if dist_info.distributed and device.type == "cuda":
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dist_info.local_rank])
        model_without_ddp = model.module

    # -------------------- optim / criterion --------------------
    optimizer, lr = build_optimizer(model_without_ddp, cfg, dist_info.world_size)
    cfg.lr = lr  # freeze resolved lr so adjust_learning_rate reads it
    scaler = engine.make_scaler(cfg, device)

    mixup_active = heads.mixup_allowed(cfg) and (cfg.mixup > 0 or cfg.cutmix > 0)
    mixup_fn = None
    if mixup_active:
        mixup_fn = Mixup(mixup_alpha=cfg.mixup, cutmix_alpha=cfg.cutmix,
                         prob=cfg.mixup_prob, switch_prob=cfg.mixup_switch_prob,
                         mode=cfg.mixup_mode, label_smoothing=cfg.smoothing,
                         num_classes=nb_classes)
    criterion = heads.build_criterion(cfg, mixup_active=mixup_active)

    if D.is_main_process() and cfg.output_dir:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(cfg.output_dir, "log.txt") if cfg.output_dir else None

    metric_key, higher_better = PRIMARY[cfg.task]
    best_metric = -float("inf") if higher_better else float("inf")

    if args.dev_steps > 0:          # real quick check: cap steps, keep GPU/bf16/DALI/checkpoint
        max_train_steps = max_eval_steps = args.dev_steps
    elif cfg.smoke:
        max_train_steps = max_eval_steps = 3
    else:
        max_train_steps = max_eval_steps = None

    # -------------------- train loop --------------------
    for epoch in range(cfg.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        engine.train_one_epoch(
            model, criterion, train_iterable, optimizer, device, epoch, cfg, scaler,
            mixup_fn=mixup_fn, n_iter_per_epoch=n_iter_per_epoch, max_steps=max_train_steps)

        stats = engine.evaluate(val_iterable, model, device, cfg, max_steps=max_eval_steps)
        cur = stats.get(metric_key, float("nan"))
        is_best = (cur > best_metric) if higher_better else (cur < best_metric)
        if not (cur != cur):  # not NaN
            if is_best:
                best_metric = cur

        if D.is_main_process():
            log = {"epoch": epoch, **{f"eval_{k}": v for k, v in stats.items()},
                   "best": best_metric}
            print(f"[eval] epoch {epoch}: {stats}  best_{metric_key}={best_metric:.4f}")
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(log) + "\n")
            if cfg.output_dir and not cfg.smoke:
                _save(model_without_ddp, optimizer, scaler, epoch, cfg, "checkpoint.pth")
                if is_best:
                    _save(model_without_ddp, optimizer, scaler, epoch, cfg, "best_checkpoint.pth")

    # -------------------- final metrics + cost profile --------------------
    if D.is_main_process():
        profile = prof.profile_model(model_without_ddp, cfg.input_size, device)
        final = {"dataset": cfg.dataset, "arm": args.model, "model": cfg.model,
                 "task": cfg.task, "seed": cfg.seed, "primary_metric": metric_key,
                 f"best_{metric_key}": best_metric, "last_eval": stats,
                 "cost": profile}
        print("[final]", json.dumps(final, indent=2))
        if cfg.output_dir:
            with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
                json.dump(final, f, indent=2)

    D.cleanup()
    if cfg.smoke:
        print("SMOKE PASS")


def _save(model_without_ddp, optimizer, scaler, epoch, cfg, name):
    torch.save({"model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "epoch": epoch,
                "cfg": vars(cfg)}, os.path.join(cfg.output_dir, name))


def _coerce(v):
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


if __name__ == "__main__":
    main()
