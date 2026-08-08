"""train_one_epoch / evaluate — mirrors engine_finetune.py but task-dispatched and AMP-aware.

Key differences from the repo's engine_finetune.py:
  - AMP dtype from cfg.amp_dtype: bf16 on cuda (no GradScaler needed), fp16 uses a GradScaler,
    fp32 (cpu/smoke) runs eager. `torch.autocast(device_type, dtype=...)`.
  - Per-iteration cosine LR w/ warmup via util.lr_sched.adjust_learning_rate (FairConfig already
    exposes .lr/.min_lr/.warmup_epochs/.epochs).
  - gradient accumulation via cfg.accum_iter; mixup/cutmix (identity only) via timm Mixup.
  - evaluate() DISPATCHES on cfg.task -> topk / multilabel / verification / liveness metrics.

Model-agnostic: no model-name branch anywhere here (fairness invariant).
"""
import math
import sys

import torch

import util.lr_sched as lr_sched
from . import metrics as M
from .heads import eval_mode


# ----------------------------------------------------------------------- AMP helpers
def _amp_dtype(cfg, device):
    """Resolve the autocast dtype. fp32 on cpu/smoke; bf16/fp16 on cuda per cfg."""
    if device.type != "cuda":
        return torch.float32
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        cfg.amp_dtype, torch.bfloat16)


class Scaler:
    """Grad-accum-aware backward+step. GradScaler ONLY for fp16; bf16/fp32 need none."""

    def __init__(self, dtype):
        self.use_scaler = dtype == torch.float16
        self._scaler = torch.cuda.amp.GradScaler() if self.use_scaler else None

    def __call__(self, loss, optimizer, parameters, clip_grad=None, update_grad=True):
        if self._scaler is not None:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
        if not update_grad:
            return
        if self._scaler is not None:
            if clip_grad is not None:
                self._scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            optimizer.step()

    def state_dict(self):
        return self._scaler.state_dict() if self._scaler is not None else {}

    def load_state_dict(self, sd):
        if self._scaler is not None and sd:
            self._scaler.load_state_dict(sd)


def make_scaler(cfg, device):
    return Scaler(_amp_dtype(cfg, device))


# ----------------------------------------------------------------------- train
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, cfg, scaler,
                    mixup_fn=None, n_iter_per_epoch=None, max_steps=None):
    """One epoch. `n_iter_per_epoch` needed when data_loader has no __len__ (DALI)."""
    model.train(True)
    optimizer.zero_grad()
    accum_iter = cfg.accum_iter
    dtype = _amp_dtype(cfg, device)
    total_iters = n_iter_per_epoch if n_iter_per_epoch is not None else len(data_loader)

    running_loss, n_seen = 0.0, 0
    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break
        samples, targets = _unpack_batch(batch, device)

        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / total_iters + epoch, cfg)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss = loss / accum_iter
        update = (step + 1) % accum_iter == 0
        scaler(loss, optimizer, parameters=model.parameters(),
               clip_grad=cfg.clip_grad, update_grad=update)
        if update:
            optimizer.zero_grad()

        running_loss += loss_value
        n_seen += 1
        if step % 20 == 0:
            lr = max(g["lr"] for g in optimizer.param_groups)
            print(f"  epoch {epoch} step {step}/{total_iters} loss {loss_value:.4f} lr {lr:.2e}")

    return {"loss": running_loss / max(n_seen, 1)}


# ----------------------------------------------------------------------- evaluate
@torch.no_grad()
def evaluate(data_loader, model, device, cfg, max_steps=None):
    """Dispatch on cfg.task. Returns a stats dict whose primary key is task-obvious:
    identity->acc1, attribute->mAP, verification->auc, liveness->acer."""
    model.eval()
    mode = eval_mode(cfg)
    dtype = _amp_dtype(cfg, device)

    if mode == "embedding":
        return _evaluate_verification(data_loader, model, device, dtype, cfg, max_steps)

    all_out, all_tgt = [], []
    for step, batch in enumerate(data_loader):
        if max_steps is not None and step >= max_steps:
            break
        samples, targets = _unpack_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            outputs = model(samples)
        all_out.append(outputs.float().cpu())
        all_tgt.append(targets.cpu())

    out = torch.cat(all_out)
    tgt = torch.cat(all_tgt)

    if mode == "logits_topk":
        acc1, acc5 = M.accuracy_topk(out, tgt, ks=(1, 5))
        return {"acc1": acc1, "acc5": acc5}
    if mode == "multilabel":
        res = M.multilabel_metrics(out, tgt)
        return {"mAP": res["mAP"], "attr_acc": res["acc"]}
    if mode == "liveness":
        scores = torch.softmax(out, dim=1)[:, 1]  # P(attack)
        res = M.liveness_metrics(scores, tgt)
        return {"acer": res["acer"], "hter": res["hter"], "apcer": res["apcer"],
                "bpcer": res["bpcer"], "auc": res["auc"]}
    raise KeyError(f"unknown eval mode '{mode}'")


@torch.no_grad()
def _evaluate_verification(pair_loader, model, device, dtype, cfg, max_steps):
    """Embed each side of a pair, compute cosine-sim verification metrics."""
    base = model.module if hasattr(model, "module") else model
    embA, embB, labels = [], [], []
    for step, (ia, ib, lbl) in enumerate(pair_loader):
        if max_steps is not None and step >= max_steps:
            break
        ia, ib = ia.to(device, non_blocking=True), ib.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(dtype != torch.float32)):
            fa = base.forward_features(ia)
            fb = base.forward_features(ib)
        embA.append(fa.float().cpu())
        embB.append(fb.float().cpu())
        labels.append(lbl)
    A, B = torch.cat(embA), torch.cat(embB)
    lab = torch.cat(labels)
    res = M.verification_metrics(A, B, lab, far_target=1e-3, n_folds=1)
    return {"auc": res["auc"], "tar_at_far": res["tar_at_far"], "acc": res["acc"]}


# ----------------------------------------------------------------------- utils
def _unpack_batch(batch, device):
    """Normalise (samples, targets) from torch DataLoader or DALI iterator output.

    DALIGenericIterator yields a *list of dicts* (one dict per pipeline), e.g.
    ``[{"images": T, "labels": T}]`` — handle that before the plain (samples, targets) case.
    """
    if isinstance(batch, (list, tuple)) and len(batch) >= 1 and isinstance(batch[0], dict):
        d = batch[0]  # DALI: single-pipeline list of {"images","labels"}
        samples, targets = d["images"], d["labels"]
    elif isinstance(batch, dict):
        samples, targets = batch["images"], batch["labels"]
    elif isinstance(batch, (list, tuple)):
        samples, targets = batch[0], batch[-1]
    else:
        raise TypeError(f"unexpected batch type {type(batch)}")
    samples = samples.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    if targets.ndim > 1 and targets.shape[-1] == 1:
        targets = targets.squeeze(-1)
    return samples, targets
