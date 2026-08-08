"""Cost profiling: #params, GFLOPs, throughput (img/s), peak VRAM.

For the paper's accuracy-vs-cost axis (EXPERIMENT_DESIGN.md §2.3). fvcore is used for FLOPs
when importable (guarded); otherwise GFLOPs is None with a note. Throughput and peak-VRAM are
guarded for CPU (few iters, no CUDA memory query). Never crashes on the CPU box.
"""
import time

import torch


def _count_params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def _gflops(model, input_size, device):
    """Return (gflops, note). Uses fvcore.FlopCountAnalysis if available."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None, "fvcore not installed; GFLOPs skipped"
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    try:
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        g = flops.total() / 1e9
        note = "per-image GFLOPs (fvcore)"
    except Exception as e:  # pragma: no cover
        g, note = None, f"fvcore failed: {type(e).__name__}"
    if was_training:
        model.train()
    return g, note


@torch.no_grad()
def _throughput(model, input_size, device, batch_size=16, iters=10, warmup=3):
    """Timed forward throughput (img/s). On CPU uses few iters/small batch to stay fast."""
    on_cuda = device.type == "cuda"
    if not on_cuda:
        batch_size, iters, warmup = 4, 3, 1
    model.eval()
    x = torch.randn(batch_size, 3, input_size, input_size, device=device)
    for _ in range(warmup):
        model(x)
    if on_cuda:
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        model(x)
    if on_cuda:
        torch.cuda.synchronize()
    dt = time.time() - t0
    return (batch_size * iters) / dt if dt > 0 else float("nan")


def profile_model(model, input_size, device):
    """Return a dict of cost numbers; safe on CPU."""
    device = torch.device(device) if isinstance(device, str) else device
    result = {
        "params_M": round(_count_params_m(model), 3),
        "gflops": None,
        "gflops_note": None,
        "throughput_img_s": None,
        "peak_vram_MB": None,
        "device": str(device),
    }
    g, note = _gflops(model, input_size, device)
    result["gflops"] = round(g, 3) if g is not None else None
    result["gflops_note"] = note

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        result["throughput_img_s"] = round(_throughput(model, input_size, device), 2)
    except Exception as e:  # pragma: no cover
        result["throughput_img_s"] = None
        result["throughput_note"] = f"{type(e).__name__}: {e}"
    if device.type == "cuda":
        result["peak_vram_MB"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
    return result
