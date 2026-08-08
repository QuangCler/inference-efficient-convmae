"""Minimal TensorRT runtime for the demo (TensorRT 10 API).

Loads a serialized engine built by build_trt.py and runs one inference, using PyTorch CUDA
tensors as the I/O buffers (no pycuda needed). Everything is best-effort: if TensorRT is not
installed or an engine is missing/unusable, the caller falls back to PyTorch.
"""
import os

import torch

_ENGINES = {}   # path -> (engine, context)


def available():
    try:
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def engine_path(engines_dir, job, arm, prec, feats=False):
    return os.path.join(engines_dir, f"{job}_{arm}_{prec}.engine")


def _load(path):
    if path in _ENGINES:
        return _ENGINES[path]
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    with open(path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    _ENGINES[path] = (engine, ctx)
    return engine, ctx


def _io_names(engine):
    import tensorrt as trt
    in_name = out_name = None
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_name = n
        else:
            out_name = n
    return in_name, out_name


@torch.no_grad()
def infer(path, x):
    """Run the engine on a single [1,3,224,224] input; return the output CUDA tensor."""
    engine, ctx = _load(path)
    in_name, out_name = _io_names(engine)
    x = x.contiguous().to("cuda", dtype=torch.float32)
    ctx.set_input_shape(in_name, tuple(x.shape))
    out_shape = tuple(ctx.get_tensor_shape(out_name))
    out = torch.empty(out_shape, device="cuda", dtype=torch.float32)
    ctx.set_tensor_address(in_name, x.data_ptr())
    ctx.set_tensor_address(out_name, out.data_ptr())
    ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    return out
