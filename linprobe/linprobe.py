"""ImageNet-1K linear probing for the four arms (reproduces Fig VI-2 / the linear-probe Top-1).

Freezes the pretrained backbone and trains only a non-affine-BatchNorm + Linear head, following the
MAE/MoCo-v3 linear-probe recipe: LARS, zero weight decay, weak augmentation. Report recipe (§V):
effective batch 4,096, blr 0.1 (→ lr = 0.1 × 4096/256 = 1.6), 90 epochs, 10-epoch warm-up.

Run (one arm; each starts from that arm's 300-epoch pretrain checkpoint):
    python -m torch.distributed.launch --nproc_per_node=8 linprobe/linprobe.py \
        --model ghost --finetune models/allghost_epoch_300.pt \
        --data_path /path/to/imagenet --output_dir outputs/ghost_linprobe \
        --batch_size 512 --epochs 90 --blr 0.1 --weight_decay 0.0 --warmup_epochs 10 --dist_eval

Standard torchvision ImageFolder loading (no DALI). The Mamba arms need the CUDA-only mamba_ssm.
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # tensorboard is an optional logging dependency
    SummaryWriter = None
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# The model factories and util/ live at the repo root; make them importable when this launcher is
# run as `python linprobe/linprobe.py` from the repo root (engine_finetune.py sits alongside it).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timm

try:
    from timm.layers import trunc_normal_
except Exception:  # timm older layouts
    from timm.models.layers import trunc_normal_

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

from engine_finetune import train_one_epoch, evaluate

from model_convmae_cls_baseline import convmae_baseline_cls
from model_convmae_cls_ghost import convmae_ghost_cls
from model_convmae_cls_bimamba import convmae_bimamba_cls
from model_convmae_cls_forwardmamba import convmae_forwardmamba_cls


def get_model(model_name: str, num_classes: int = 1000, **s1_s2_kwargs):
    if model_name == "baseline":
        return convmae_baseline_cls(num_classes=num_classes)
    if model_name == "ghost":
        return convmae_ghost_cls(num_classes=num_classes)
    if model_name == "bimamba":
        return convmae_bimamba_cls(num_classes=num_classes, **s1_s2_kwargs)
    if model_name == "forwardmamba":
        return convmae_forwardmamba_cls(num_classes=num_classes, **s1_s2_kwargs)
    raise ValueError(f"Unknown model: {model_name}")


def _extract_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model", "module", "net", "teacher"):
            sd = checkpoint.get(key)
            if isinstance(sd, dict):
                return sd
        # Some checkpoints are a raw state_dict at the top level.
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint
        raise KeyError(
            "Could not find a model state_dict in checkpoint; "
            f"top-level keys: {list(checkpoint.keys())[:20]}"
        )
    # Some checkpoints are saved directly as a state_dict (rare).
    return checkpoint


def _load_pretrain_checkpoint(model, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    print("Load pre-trained checkpoint from: %s" % ckpt_path, flush=True)
    checkpoint_model = _extract_checkpoint_state_dict(checkpoint)

    # Strip DDP prefix if present.
    if any(k.startswith("module.") for k in checkpoint_model.keys()):
        checkpoint_model = {k.replace("module.", "", 1): v for k, v in checkpoint_model.items()}

    # If checkpoint contains a head with mismatched shape, drop it.
    state_dict = model.state_dict()
    for k in ("head.weight", "head.bias"):
        if k in checkpoint_model and k in state_dict and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint", flush=True)
            del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg, flush=True)


def validate_dataset_layout(train_dir: str, val_dir: str):
    train_classes = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    val_classes = sorted([
        d for d in os.listdir(val_dir)
        if os.path.isdir(os.path.join(val_dir, d))
    ])

    # Leftover train class tar files indicate extraction is incomplete.
    train_tar_leftovers = [
        f for f in os.listdir(train_dir)
        if f.endswith(".tar") and os.path.isfile(os.path.join(train_dir, f))
    ]
    if train_tar_leftovers:
        raise RuntimeError(
            f"Detected {len(train_tar_leftovers)} .tar files still inside train/ (extraction incomplete). "
            "Please finish extracting all class tar files before training."
        )

    if len(train_classes) != len(val_classes):
        raise RuntimeError(
            f"Class folder count mismatch: train={len(train_classes)}, val={len(val_classes)}. "
            "ImageFolder labels may be misaligned."
        )

    if train_classes != val_classes:
        raise RuntimeError(
            "Class folder names differ between train/ and val/. "
            "ImageFolder labels may be inconsistent."
        )

    print(f"[INFO] Dataset class folders are consistent: {len(train_classes)} classes", flush=True)


def get_args_parser():
    parser = argparse.ArgumentParser(
        "Linear probing (MAE recipe) for baseline / ghost / bimamba / forwardmamba",
        add_help=True,
    )
    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus; report uses 4,096)",
    )
    parser.add_argument("--epochs", default=90, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (effective batch size increases)",
    )

    # Model parameters
    parser.add_argument("--model", required=True, type=str,
                        choices=["baseline", "ghost", "bimamba", "forwardmamba"])

    # S1/S2 architecture flags (must match the pretrained model; the 300-ep Mamba arms use neither)
    parser.add_argument("--use_local_scan", action="store_true",
                        help="Enable S1 Windowed Local Scan (must match pretrained model).")
    parser.add_argument("--local_scan_window_size", type=int, default=4,
                        help="Window size for S1 local scan.")
    parser.add_argument("--scan_direction", choices=["horizontal", "vertical"], default="horizontal",
                        help="Token order inside each local scan window.")
    parser.add_argument("--use_convffn", action="store_true",
                        help="Enable S2 ConvFFN (must match pretrained model).")
    parser.add_argument("--convffn_expand_ratio", type=float, default=4.0,
                        help="Hidden expansion ratio for S2 ConvFFN.")
    parser.add_argument("--convffn_dw_kernel", type=int, default=3,
                        help="Depthwise conv kernel size for S2 ConvFFN.")

    # Optimizer parameters (MAE linear-probe recipe: LARS, zero weight decay)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="learning rate (absolute lr)")
    parser.add_argument(
        "--blr",
        type=float,
        default=0.1,
        metavar="LR",
        help="base lr: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=10, metavar="N")
    parser.add_argument("--optimizer", type=str, default="lars", choices=["lars"], help="kept for compat")
    parser.add_argument("--log_every", type=int, default=20, help="kept for compat (upstream uses 20)")
    parser.add_argument("--amp", action="store_true", help="kept for compat (upstream uses AMP)")

    # Finetuning params
    parser.add_argument("--finetune", default="", help="the 300-epoch pretrain checkpoint to probe")

    # Dataset parameters
    parser.add_argument("--data_path", required=True, type=str, help="ImageNet root containing train/ and val/")
    parser.add_argument("--num_classes", default=1000, type=int, help="number of classes")

    parser.add_argument("--output_dir", required=True, help="path where to save")
    parser.add_argument("--log_dir", default=None, help="path where to tensorboard log (default: output_dir)")
    parser.add_argument("--device", default="cuda", help="device to use")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--dist_eval", action="store_true", default=False)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters (upstream util.misc.init_distributed_mode)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))), flush=True)
    print("{}".format(args).replace(', ', ',\n'), flush=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available but --device cuda was requested.")

    # Reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Val directory not found: {val_dir}")
    validate_dataset_layout(train_dir, val_dir)

    # linear probe: weak augmentation (upstream)
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset_train = datasets.ImageFolder(train_dir, transform=transform_train)
    dataset_val = datasets.ImageFolder(val_dir, transform=transform_val)
    print(dataset_train, flush=True)
    print(dataset_val, flush=True)

    # Keep upstream behavior: always use DistributedSampler with world_size=1 when not distributed.
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train), flush=True)
    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added.',
                flush=True,
            )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_dir = args.log_dir if args.log_dir is not None else args.output_dir
    if global_rank == 0 and log_dir is not None and not args.eval and SummaryWriter is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    s1_s2_kwargs = {
        "use_local_scan": args.use_local_scan,
        "local_scan_window_size": args.local_scan_window_size,
        "scan_direction": args.scan_direction,
        "use_convffn": args.use_convffn,
        "convffn_expand_ratio": args.convffn_expand_ratio,
        "convffn_dw_kernel": args.convffn_dw_kernel,
    }
    model = get_model(args.model, num_classes=args.num_classes, **s1_s2_kwargs)

    if args.finetune and not args.eval:
        _load_pretrain_checkpoint(model, args.finetune)
        # manually initialize fc layer: following MoCo v3
        trunc_normal_(model.head.weight, std=0.01)

    # linear probe only: hack head with BN and freeze all but head
    model.head = torch.nn.Sequential(
        torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
        model.head,
    )
    for _, p in model.named_parameters():
        p.requires_grad = False
    for _, p in model.head.named_parameters():
        p.requires_grad = True

    model.to(device)
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp), flush=True)
    print('number of params (M): %.2f' % (n_parameters / 1.e6), flush=True)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size), flush=True)
    print("actual lr: %.2e" % args.lr, flush=True)
    print("accumulate grad iterations: %d" % args.accum_iter, flush=True)
    print("effective batch size: %d" % eff_batch_size, flush=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    optimizer = LARS(model_without_ddp.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer, flush=True)
    loss_scaler = NativeScaler()
    criterion = torch.nn.CrossEntropyLoss()
    print("criterion = %s" % str(criterion), flush=True)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%", flush=True)
        return

    print(f"Start training for {args.epochs} epochs", flush=True)
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir:
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%", flush=True)
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]
            misc.save_best_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )
        print(f"Max accuracy: {max_accuracy:.2f}%", flush=True)

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str), flush=True)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
