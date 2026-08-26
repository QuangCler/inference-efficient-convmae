#!/usr/bin/env python3
"""Deployment demo: run a fine-tuned/linprobe ConvMAE-family classifier through the
report's inference pipelines (Table V-1 decision tree) and classify real images.

- baseline / ghost  -> native PyTorch AND full TensorRT (ONNX export -> engine -> infer)
- bimamba / forwardmamba -> native PyTorch only (selective scan has no clean ONNX/TRT
  path in the tested toolchain; mirrors the report's mixed-pipeline rationale)

Usage (on gpu128, inside /root/capstone/linprobe_cls, venv /root/mamba_venv2):
  python demo_deploy_infer.py --arm baseline \
      --checkpoint outputs/baseline_ep50_lin10/best_checkpoint.pth \
      --images /opt/imagenet_stream/ILSVRC/Data/CLS-LOC/val/n01440764 --limit 4 \
      --pipeline auto --precision fp16
"""
import argparse, json, os, statistics, sys, time

# The classifier factories live at the repo root; make them importable when this
# script is run as `python scripts/demo_deploy_infer.py` from the repo root.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
from PIL import Image
from torchvision import transforms

from models.convmae_cls import convmae_cls

MAMBA_ARMS = ("bimamba", "forwardmamba")
# ImageNet class-index JSON (maps class idx -> [wnid, name]) for readable top-k labels.
# Optional: point IMAGENET_CLASS_INDEX at your own file; if absent, raw indices are shown.
CLASS_INDEX = os.environ.get(
    "IMAGENET_CLASS_INDEX", os.path.join(REPO, "scripts", "imagenet_class_index.json"))

PREPROC = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def build_model(arm):
    # convmae_cls imports the Mamba arms lazily, so baseline/ghost never pull mamba_ssm.
    return convmae_cls(arm, num_classes=1000)


def load_checkpoint(model, path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    # linprobe checkpoints wrap the head as Sequential(BN, Linear)
    if any(k.startswith("head.1.") for k in sd):
        model.head = torch.nn.Sequential(
            torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    msg = model.load_state_dict(sd, strict=False)
    missing = [k for k in msg.missing_keys]
    assert not missing, f"missing keys on load: {missing[:8]}"
    print(f"[load] {path}: OK (unexpected={len(msg.unexpected_keys)} decoder/MAE-only keys)")
    return model


def load_images(root, limit):
    if os.path.isfile(root):
        paths = [root]
    else:
        paths = sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith((".jpeg", ".jpg", ".png")))[:limit]
    batch = torch.stack([PREPROC(Image.open(p).convert("RGB")) for p in paths])
    return paths, batch


def class_names():
    if os.path.exists(CLASS_INDEX):
        idx = json.load(open(CLASS_INDEX))
        return {int(k): v[1] for k, v in idx.items()}
    return {}


def topk_report(logits, paths, names, k=5):
    probs = logits.softmax(-1)
    top = probs.topk(k, dim=-1)
    for i, p in enumerate(paths):
        row = ", ".join(f"{names.get(c.item(), c.item())}:{v.item():.3f}"
                        for v, c in zip(top.values[i], top.indices[i]))
        print(f"  {os.path.basename(p)}: {row}")


def run_pytorch(model, batch, paths, names, precision, iters=50):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    m = model.to(dtype).cuda().eval()
    x = batch.to("cuda", dtype)
    with torch.no_grad():
        logits = m(x).float().cpu()
        torch.cuda.synchronize()
        t = []
        for _ in range(iters):
            t0 = time.perf_counter(); m(x); torch.cuda.synchronize()
            t.append(time.perf_counter() - t0)
    lat = statistics.median(t) * 1000
    print(f"[pytorch/{precision}] median latency {lat:.1f} ms/batch "
          f"({len(paths)} imgs, {len(paths)/statistics.median(t):.0f} img/s)")
    topk_report(logits, paths, names)
    return logits


def run_tensorrt(model, batch, paths, names, precision, workdir="trt_demo"):
    try:
        import tensorrt as trt
    except ImportError:
        print("[tensorrt] tensorrt not installed here -> skipped (native PyTorch result above is the reference)")
        return None
    os.makedirs(workdir, exist_ok=True)
    bs = batch.shape[0]
    onnx_path = os.path.join(workdir, f"model_bs{bs}.onnx")
    if not os.path.exists(onnx_path):
        torch.onnx.export(model.float().cuda().eval(), batch.cuda().float(), onnx_path,
                          input_names=["input"], output_names=["logits"], opset_version=17)
        print(f"[tensorrt] exported {onnx_path}")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        for i in range(parser.num_errors):
            print("[tensorrt] parse error:", parser.get_error(i))
        print("[tensorrt] ONNX parse failed -> pipeline unavailable for this arm (as the report documents for state-space ops)")
        return None
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    engine_bytes = builder.build_serialized_network(network, config)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    ctx = engine.create_execution_context()
    x = batch.cuda().float().contiguous()
    out = torch.empty(bs, 1000, device="cuda", dtype=torch.float32)
    ctx.set_tensor_address("input", x.data_ptr())
    ctx.set_tensor_address("logits", out.data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    ctx.execute_async_v3(stream); torch.cuda.synchronize()
    t = []
    for _ in range(50):
        t0 = time.perf_counter(); ctx.execute_async_v3(stream); torch.cuda.synchronize()
        t.append(time.perf_counter() - t0)
    lat = statistics.median(t) * 1000
    print(f"[tensorrt/{precision}] median latency {lat:.1f} ms/batch "
          f"({bs} imgs, {bs/statistics.median(t):.0f} img/s)")
    logits = out.cpu()
    topk_report(logits, paths, names)
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "ghost", "bimamba", "forwardmamba"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="image file or directory")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--pipeline", default="auto", choices=["auto", "pytorch", "tensorrt"])
    ap.add_argument("--precision", default="fp16", choices=["fp32", "fp16"])
    args = ap.parse_args()

    names = class_names()
    model = load_checkpoint(build_model(args.arm), args.checkpoint)
    paths, batch = load_images(args.images, args.limit)
    print(f"[demo] arm={args.arm} pipeline={args.pipeline} precision={args.precision} imgs={len(paths)}")

    ref = run_pytorch(model, batch, paths, names, args.precision)

    want_trt = args.pipeline == "tensorrt" or (args.pipeline == "auto" and args.arm not in MAMBA_ARMS)
    if args.arm in MAMBA_ARMS and args.pipeline == "tensorrt":
        print("[tensorrt] state-space arm: full TensorRT is not supported by the tested "
              "export path (no standard ONNX selective-scan op) -> use native PyTorch / mixed pipeline.")
    elif want_trt:
        trt_logits = run_tensorrt(model, batch, paths, names, args.precision)
        if trt_logits is not None:
            diff = (trt_logits.softmax(-1) - ref.softmax(-1)).abs().max().item()
            agree = (trt_logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
            print(f"[gate] TRT vs PyTorch: max prob diff {diff:.4f}, top-1 agreement {agree*100:.0f}% "
                  f"({'PASS' if agree == 1.0 else 'CHECK'})")


if __name__ == "__main__":
    main()
