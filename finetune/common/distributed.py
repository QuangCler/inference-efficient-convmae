"""Thin DDP init for torchrun; degrades to a single-process CPU path for smoke.

Reads RANK/WORLD_SIZE/LOCAL_RANK from the env (set by torchrun). When not launched
distributed (env vars absent), returns a world_size=1 / rank=0 / cpu context so the smoke
test runs unchanged. Query helpers delegate to torch.distributed but are safe when the
process group was never initialised.
"""
import os

import torch
import torch.distributed as dist


class DistInfo:
    def __init__(self, distributed, rank, world_size, local_rank, device):
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.device = device


def init_distributed(force_cpu=False):
    """Initialise the process group if launched via torchrun; else single-process.

    force_cpu (smoke) skips CUDA and any process-group init.
    """
    if force_cpu or "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        # Not launched via torchrun (single process). Still use the GPU if one is available —
        # otherwise a plain `python train.py` on a GPU box would silently run on CPU.
        if not force_cpu and torch.cuda.is_available():
            torch.cuda.set_device(0)
            return DistInfo(False, 0, 1, 0, torch.device("cuda", 0))
        return DistInfo(False, 0, 1, 0, torch.device("cpu"))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if not torch.cuda.is_available():
        # Distributed launch requested without CUDA (unusual): use gloo on cpu.
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
        return DistInfo(True, rank, world_size, local_rank, torch.device("cpu"))

    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)
    # Pass device_id so NCCL knows the rank->GPU mapping (avoids the "devices unknown" barrier
    # warning and the hang it can cause on multi-GPU nodes).
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=dev)
    dist.barrier(device_ids=[local_rank])
    return DistInfo(True, rank, world_size, local_rank, dev)


def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist() else 1


def get_rank():
    return dist.get_rank() if is_dist() else 0


def is_main_process():
    return get_rank() == 0


def barrier():
    if is_dist():
        dist.barrier()


def cleanup():
    if is_dist():
        dist.destroy_process_group()
