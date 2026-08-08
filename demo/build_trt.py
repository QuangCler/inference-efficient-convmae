#!/usr/bin/env python3
"""Build TensorRT engines for the fine-tuned face models (run once, on your NVIDIA GPU).

For each downloaded checkpoint it exports the model to ONNX and builds a TensorRT engine at
FP32 and FP16 into ./engines/, which app.py can then select as the inference backend.

IMPORTANT — build on the SAME GPU you will run on. A TensorRT engine is compiled for one
specific GPU architecture + TensorRT/driver version and is NOT portable: an engine built on,
say, an A5000 will fail to load on a GTX 1650. So run this script on your own machine (e.g.
the GTX 1650) — do not ship or copy .engine files between different GPUs.

Works on a GTX 1650 (Turing, sm_75): FP16 engines build and run (no Tensor Cores, so FP16 is
not accelerated, but it is smaller/valid). Classification tasks export encoder+head
(-> logits); LFW exports the encoder (-> 768-d embedding) for cosine verification.

Requirements: an NVIDIA GPU, a CUDA build of torch, `pip install onnx tensorrt`, and the
checkpoints fetched (python fetch_checkpoints.py). Each engine takes ~1-3 min to build.

    python build_trt.py                 # all jobs/arms, FP32+FP16
    python build_trt.py --jobs CelebA LFW --precisions fp16
"""
import argparse
import os

import torch
import torch.nn as nn

from face_models import build_and_load

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
ENGINES = os.path.join(HERE, "engines")
ONNX_DIR = os.path.join(HERE, "engines", "onnx")

JOBS = {
    "CelebA": {"task": "attribute", "nc": 40, "ck": {"baseline": "celeba_baseline.pth", "ghost": "celeba_ghost.pth"}},
    "CASIA":  {"task": "identity",  "nc": None, "ck": {"baseline": "casia_baseline.pth", "ghost": "casia_ghost.pth"}},
    "SCface": {"task": "identity",  "nc": None, "ck": {"baseline": "scface_baseline.pth", "ghost": "scface_ghost.pth"}},
    "LFW":    {"task": "verification", "nc": 1680, "ck": {"baseline": "lfw_baseline.pth", "ghost": "lfw_ghost.pth"}},
}


class EmbedOnly(nn.Module):
    """Wrap a classifier so ONNX export emits the 768-d embedding (for LFW verification)."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m.forward_features(x)


def _head_size(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    return sd["head.weight"].shape[0]


def export_onnx(job, arm):
    spec = JOBS[job]
    path = os.path.join(CKPT, spec["ck"][arm])
    if not os.path.exists(path):
        print(f"[skip] {job}/{arm}: checkpoint missing ({path})"); return None
    nc = spec["nc"] or _head_size(path)
    model = build_and_load(arm, nc, path, map_location="cpu").float().cuda().eval()
    if spec["task"] == "verification":
        model = EmbedOnly(model).cuda().eval()
    os.makedirs(ONNX_DIR, exist_ok=True)
    onnx_path = os.path.join(ONNX_DIR, f"{job}_{arm}.onnx")
    dummy = torch.randn(1, 3, 224, 224, device="cuda")
    torch.onnx.export(model, dummy, onnx_path, input_names=["input"], output_names=["output"],
                      opset_version=17)
    print(f"[onnx] {onnx_path}")
    return onnx_path


def build_engine(onnx_path, engine_path, fp16):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(open(onnx_path, "rb").read()):
        for i in range(parser.num_errors):
            print("   parse error:", parser.get_error(i))
        return False
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB (safe on 4 GB cards)
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    data = builder.build_serialized_network(network, config)
    if data is None:
        print("   build failed"); return False
    with open(engine_path, "wb") as f:
        f.write(data)
    print(f"[engine] {engine_path}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", default=list(JOBS))
    ap.add_argument("--arms", nargs="+", default=["baseline", "ghost"])
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU visible to torch — install a CUDA torch build first.")
    print(f"Building TensorRT engines on: {torch.cuda.get_device_name(0)}")
    print("NOTE: engines are specific to THIS GPU + TensorRT version — do not copy them to a "
          "different GPU; rebuild there instead.\n")
    os.makedirs(ENGINES, exist_ok=True)
    ok = 0
    for job in args.jobs:
        for arm in args.arms:
            onnx_path = export_onnx(job, arm)
            if onnx_path is None:
                continue
            for prec in args.precisions:
                epath = os.path.join(ENGINES, f"{job}_{arm}_{prec}.engine")
                print(f"building {job}/{arm}/{prec} ...")
                if build_engine(onnx_path, epath, prec == "fp16"):
                    ok += 1
    print(f"\nDone. {ok} engine(s) in {ENGINES}/  — select them in the app's backend dropdown.")


if __name__ == "__main__":
    main()
