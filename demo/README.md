# GhostConvMAE — Fine-tuned Face Model Demo

Interactive demo comparing the two deployed backbones — **ConvMAE-Base** vs **Ghost+ConvMAE** —
on four face tasks, side by side. Each run shows the prediction plus a resource table with the
**peak VRAM measured live on your GPU** *and* the paper's A5000 reference, and you can switch the
inference backend between **native PyTorch** and a **TensorRT engine** in the UI. Runs on a
laptop: a small CUDA GPU (e.g. **RTX 1650, 4 GB**) or plain **CPU** (auto-detected).

Only the two fine-tuned convolution/Transformer arms are included; the Mamba arms are not part of
the fine-tuning demo (their CUDA-only kernels don't ship to a laptop).

## Tasks
- **CelebA — attributes (top-5)**: drop a face → the 5 highest-confidence attributes, base vs ghost.
- **CASIA — identity (top-5)**: drop a face → top-5 identity confidence bars.
- **SCface — cross-resolution identity (top-5)**: the low-resolution stressor, where Ghost's
  compression costs most (report Top-1 Base 45.1% vs Ghost 31.4%).
- **LFW — verification**: drop **two** faces → cosine similarity + same/different verdict per model.

Each result also shows a resource table: backend, params, latency, **measured peak VRAM**, and the
**report peak VRAM (A5000, FP16)** for reference.

## 1. Install PyTorch for your GPU (do this first)

Install the torch build that matches your NVIDIA GPU / driver, then the rest. If you skip this and
`pip install torch` gives a CPU-only build, the app runs on CPU and cannot measure VRAM or use TensorRT.

| GPU (arch) | Recommended install | Notes |
|---|---|---|
| **GTX 16xx / RTX 20xx** (Turing, e.g. **RTX 1650**) | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` | No Tensor Cores → FP16 not accelerated, but valid; ~0.3–1.4 GB/model |
| **RTX 30xx** (Ampere) | `... /whl/cu121` | Tensor Cores → FP16/TF32 fast |
| **RTX 40xx** (Ada) | `... /whl/cu124` (or cu121) | |
| **A5000 / A100** (data-center) | `... /whl/cu124` | Matches the paper's benchmark GPU |
| **No NVIDIA GPU** | `pip install torch torchvision` | CPU only — VRAM/TensorRT disabled |

Verify: `python -c "import torch; print(torch.cuda.is_available())"` must print **True**.

```bash
python -m venv venv && source venv/bin/activate        # Windows: venv\Scripts\activate
# 1) install the torch build from the table above, then:
pip install -r requirements.txt
pip install gdown
python fetch_checkpoints.py         # face checkpoints (~5.5 GB) -> ./checkpoints/
```

## 2. (Optional) download the original datasets

```bash
pip install kaggle                  # then put your token at ~/.kaggle/kaggle.json
python fetch_datasets.py            # LFW + CelebA + CASIA -> ./datasets/  (large!)
python fetch_datasets.py --only lfw # just one
```
SCface is access-controlled and is not downloaded (request a licence, then place under
`datasets/scface/`). Any face image works as input — you don't need the datasets to try the demo.

## 3. (Optional) build TensorRT engines — **on your own GTX 1650**

> ⚠️ **Engines are not portable.** A TensorRT engine is compiled for one specific GPU
> architecture + TensorRT/driver version. An engine built on any other machine (an A5000, a
> friend's RTX 30xx, this repo's server, …) **will fail to load on your GTX 1650**. So the demo
> ships **no** pre-built engines — you must run `build_trt.py` **on the GTX 1650 itself**. Never
> copy `.engine` files between different GPUs; rebuild instead.

**3a. Install TensorRT** (matching the CUDA your torch uses — e.g. cu121 from step 1):

```bash
pip install onnx
pip install tensorrt                 # pulls the TensorRT 10.x wheels (cu12)
# Windows: same pip command works in the venv. If `import tensorrt` fails, install the
# TensorRT zip from developer.nvidia.com/tensorrt matching your CUDA, and add its lib/ to PATH.
python -c "import tensorrt, torch; print(tensorrt.__version__, torch.cuda.get_device_name(0))"
```

**3b. Build the engines on the GTX 1650** (takes ~1–3 min each; writes to `./engines/`):

```bash
python build_trt.py                                  # everything: 4 tasks x 2 arms x {fp32,fp16}
python build_trt.py --jobs CelebA LFW --precisions fp16   # or just what you need (faster)
```

It prints the GPU it is building on — make sure that is your **GTX 1650**. Classification tasks
export encoder+head (→ logits); LFW exports the encoder (→ 768-d embedding). Workspace is capped
at 1 GB so it fits the 4 GB card.

**3c.** Launch the app (step 4) and pick **TensorRT (FP16)** / **(FP32)** in the backend dropdown.
A model with no engine, or if TensorRT isn't importable, silently falls back to PyTorch and the
resource table shows the actual backend used.

## 4. Run

```bash
python app.py                       # then open http://127.0.0.1:7860
```
- **Backend dropdown** (top of the page): `PyTorch`, `TensorRT (FP16)`, or `TensorRT (FP32)`.
  TensorRT options use the engines from step 3; if an engine is missing or TensorRT isn't
  installed, that model transparently falls back to PyTorch and the table says so.
- Force CPU: `DEMO_DEVICE=cpu python app.py` (PowerShell: `$env:DEMO_DEVICE="cpu"; python app.py`).

## Peak VRAM
- On a **CUDA GPU** the table shows the **real peak VRAM measured live** per model
  (`torch.cuda.max_memory_allocated`, reset before each inference) next to the **paper A5000
  reference**; the header shows your GPU name / total / free memory.
- On **CPU** the measured column is `— (CPU)` (nothing to measure); only the reference is shown.
  If your RTX 1650 shows `— (CPU)`, your PyTorch is a CPU-only build — reinstall per the table above.

## Notes
- `casia_baseline.pth` is a freshly retrained demo checkpoint; the report's CASIA numbers are unchanged.
- `scface_{baseline,ghost}.pth` are seed-0 checkpoints (base 46.7% / ghost 31.7% Top-1).

## Layout
```
demo_app/
  app.py                 Gradio UI + inference (PyTorch / TensorRT backend selector)
  trt_backend.py         TensorRT engine loader + runtime
  build_trt.py           export ONNX + build TensorRT engines into engines/
  face_models.py         base/ghost model builders + checkpoint loader
  models/                bundled architecture code (ConvViT, Ghost blocks)
  fetch_checkpoints.py   downloads checkpoints into checkpoints/
  fetch_datasets.py      downloads the original datasets into datasets/ (Kaggle)
  resource_meta.json     reference A5000 resource numbers
  requirements.txt
  checkpoints/  engines/  datasets/   (populated by the scripts above)
```
