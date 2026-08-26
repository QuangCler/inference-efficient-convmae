"""ConvMAE-family masked-autoencoder pre-training for the four arms.

Standard MAE objective (normalized-pixel reconstruction loss, ConvMAE block-wise masking handled
inside the model), one run per arm. Report recipe (§V): ImageNet-1K 224², mask ratio 0.75,
effective batch 4,096, blr 1.5e-4 (→ lr = 1.5e-4 × 4096/256 = 2.4e-3), 300 epochs, 40-epoch warm-up.

Run (one arm; effective batch = batch_size × accum_iter × #gpus, target 4,096):
    python -m torch.distributed.launch --nproc_per_node=8 pretrain/pretrain.py \
        --model ghost --data_path /path/to/imagenet \
        --output_dir outputs/ghost_pretrain --batch_size 64 --accum_iter 8 \
        --epochs 300 --blr 1.5e-4 --warmup_epochs 40 --mask_ratio 0.75 --norm_pix_loss

Standard torchvision ImageFolder loading (no DALI). The Mamba arms need the CUDA-only mamba_ssm; the
300-epoch Mamba arms use neither S1 nor S2 (pass --use_local_scan / --use_convffn only for the
S1/S2 ablations).
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
# run as `python pretrain/pretrain.py` from the repo root (engine_pretrain.py sits alongside it).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from engine_pretrain import train_one_epoch

from models.model_convmae_baseline import convmae_baseline
from models.model_convmae_allghost import convmae_allghost


def build_model(arm, norm_pix_loss, s1_s2_kwargs):
    """One MAE model per arm, from the same factories the fine-tune / probe use."""
    if arm == "baseline":
        return convmae_baseline(norm_pix_loss=norm_pix_loss)
    if arm == "ghost":
        return convmae_allghost(norm_pix_loss=norm_pix_loss)
    if arm == "bimamba":
        from models.model_convmae_bimamba import convmae_bimamba  # CUDA-only (mamba_ssm), import lazily
        return convmae_bimamba(norm_pix_loss=norm_pix_loss, **s1_s2_kwargs)
    if arm == "forwardmamba":
        from models.model_convmae_forwardmamba import convmae_forwardmamba
        return convmae_forwardmamba(norm_pix_loss=norm_pix_loss, **s1_s2_kwargs)
    raise ValueError(f"Unknown model: {arm}")


def add_weight_decay(model, weight_decay=0.05, skip_list=()):
    """No weight decay on biases and 1-D (norm) parameters — the standard timm/MAE grouping."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay}]


def get_args_parser():
    parser = argparse.ArgumentParser("ConvMAE-family MAE pre-training", add_help=True)
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size per GPU (effective batch = batch_size * accum_iter * #gpus; report uses 4,096)")
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--accum_iter", default=1, type=int,
                        help="Accumulate gradient iterations to raise the effective batch size")

    # Model parameters
    parser.add_argument("--model", required=True, type=str,
                        choices=["baseline", "ghost", "bimamba", "forwardmamba"])
    parser.add_argument("--input_size", default=224, type=int)
    parser.add_argument("--mask_ratio", default=0.75, type=float,
                        help="Masking ratio (fraction of removed patches).")
    parser.add_argument("--norm_pix_loss", action="store_true", default=True,
                        help="Use per-patch normalized pixels as the reconstruction target (report default).")
    parser.add_argument("--no_norm_pix_loss", action="store_false", dest="norm_pix_loss")

    # S1/S2 flags for the Mamba arms (300-ep arms use neither; set only for the S1/S2 ablations)
    parser.add_argument("--use_local_scan", action="store_true", help="S1 Windowed Local Scan.")
    parser.add_argument("--local_scan_window_size", type=int, default=4)
    parser.add_argument("--scan_direction", choices=["horizontal", "vertical"], default="horizontal")
    parser.add_argument("--use_convffn", action="store_true", help="S2 ConvFFN.")
    parser.add_argument("--convffn_expand_ratio", type=float, default=4.0)
    parser.add_argument("--convffn_dw_kernel", type=int, default=3)

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=None, metavar="LR", help="absolute lr (overrides blr)")
    parser.add_argument("--blr", type=float, default=1.5e-4, metavar="LR",
                        help="base lr: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=0.0, metavar="LR")
    parser.add_argument("--warmup_epochs", type=int, default=40, metavar="N")

    # Dataset parameters
    parser.add_argument("--data_path", required=True, type=str, help="ImageNet root containing train/")
    parser.add_argument("--output_dir", default="./output_dir", help="path where to save (empty = no saving)")
    parser.add_argument("--log_dir", default=None, help="tensorboard log dir (default: output_dir)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N")
    parser.add_argument("--save_freq", default=50, type=int, help="save a checkpoint every N epochs")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument("--pin_mem", action="store_true")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))), flush=True)
    print("{}".format(args).replace(", ", ",\n"), flush=True)

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # simple augmentation (ImageNet normalization; report: random-resized-crop + hflip)
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 = bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=transform_train)
    print(dataset_train, flush=True)

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train), flush=True)

    log_dir = args.log_dir if args.log_dir is not None else args.output_dir
    if global_rank == 0 and log_dir is not None and SummaryWriter is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True)

    # define the model (no CLIP guidance — plain MAE reconstruction)
    s1_s2_kwargs = {
        "use_local_scan": args.use_local_scan,
        "local_scan_window_size": args.local_scan_window_size,
        "scan_direction": args.scan_direction,
        "use_convffn": args.use_convffn,
        "convffn_expand_ratio": args.convffn_expand_ratio,
        "convffn_dw_kernel": args.convffn_dw_kernel,
    }
    model = build_model(args.model, args.norm_pix_loss, s1_s2_kwargs)
    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp), flush=True)

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size), flush=True)
    print("actual lr: %.2e" % args.lr, flush=True)
    print("accumulate grad iterations: %d" % args.accum_iter, flush=True)
    print("effective batch size: %d" % eff_batch_size, flush=True)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer, flush=True)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs", flush=True)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args)
        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print("Training time {}".format(str(datetime.timedelta(seconds=int(total_time)))), flush=True)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
