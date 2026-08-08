"""Face-model inference demo (Gradio).

Compares the two fine-tuned backbones — ConvMAE-Base vs Ghost+ConvMAE — on four face
tasks, side by side. Per model it reports the prediction and the resource cost, showing
BOTH the peak VRAM measured live on your GPU and the paper's A5000 reference. Inference
can run through native PyTorch or a TensorRT engine (built with build_trt.py), selectable
in the UI.

Launch:  python app.py        (then open the printed http://127.0.0.1:7860)
"""
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from face_models import build_and_load, embed
import trt_backend

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "checkpoints")
ENGINES = os.path.join(HERE, "engines")

_forced = os.environ.get("DEMO_DEVICE")
DEVICE = _forced or ("cuda" if torch.cuda.is_available() else "cpu")
CUDA = DEVICE == "cuda" and torch.cuda.is_available()

CELEBA_ATTRS = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", "Bald",
    "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", "Brown_Hair",
    "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", "Gray_Hair",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Mustache",
    "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline",
    "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"]

TOP_K = 5

JOBS = {
    "CelebA": {"task": "attribute", "num_classes": 40, "title": "CelebA — facial attributes",
               "ckpts": {"baseline": "celeba_baseline.pth", "ghost": "celeba_ghost.pth"},
               "ref": "Reference (paper, 3 seeds): mAP 0.789 (Base) vs 0.778 (Ghost)."},
    "CASIA": {"task": "identity", "num_classes": None, "title": "CASIA-WebFace — identity",
              "ckpts": {"baseline": "casia_baseline.pth", "ghost": "casia_ghost.pth"},
              "ref": "Reference (paper, 3 seeds): Top-1 91.49% (Base) vs 91.32% (Ghost)."},
    "SCface": {"task": "identity", "num_classes": None, "title": "SCface — cross-resolution identity",
               "ckpts": {"baseline": "scface_baseline.pth", "ghost": "scface_ghost.pth"},
               "ref": "Reference (paper, 3 seeds): Top-1 45.13% (Base) vs 31.36% (Ghost) — low-resolution stressor."},
    "LFW": {"task": "verification", "num_classes": 1680, "title": "LFW — face verification (two images)",
            "ckpts": {"baseline": "lfw_baseline.pth", "ghost": "lfw_ghost.pth"},
            "ref": "Reference (paper, 3 seeds): ROC-AUC 0.9921 (Base) vs 0.9833 (Ghost)."},
}

BACKENDS = {
    "PyTorch": ("pytorch", None),
    "TensorRT (FP16)": ("trt", "fp16"),
    "TensorRT (FP32)": ("trt", "fp32"),
}

PREPROC = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

_CACHE = {}
_META = json.load(open(os.path.join(HERE, "resource_meta.json"))) if os.path.exists(
    os.path.join(HERE, "resource_meta.json")) else {}
ARM_NAME = {"baseline": "ConvMAE-Base", "ghost": "Ghost+ConvMAE"}


def gpu_status():
    if not CUDA:
        return ("Running on **CPU**. Peak VRAM cannot be measured and TensorRT is unavailable — install a "
                "CUDA build of PyTorch (matching your driver) to run on your GPU. See the README's per-GPU guide.")
    name = torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    trt = "TensorRT available" if trt_backend.available() else "TensorRT not installed (PyTorch only)"
    return f"Running on **CUDA — {name}** · {total/1e9:.1f} GB total, {free/1e9:.1f} GB free · {trt}."


def _head_size(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    return sd["head.weight"].shape[0]


def get_model(job, arm):
    key = (job, arm)
    if key in _CACHE:
        return _CACHE[key]
    spec = JOBS[job]
    path = os.path.join(CKPT, spec["ckpts"][arm])
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    nc = spec["num_classes"] or _head_size(path)
    t0 = time.time()
    model = build_and_load(arm, nc, path, map_location=DEVICE).to(DEVICE)
    meta = {"load_s": time.time() - t0,
            "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
            "ckpt_MB": os.path.getsize(path) / 1e6}
    _CACHE[key] = (model, meta)
    return _CACHE[key]


@torch.no_grad()
def infer_one(job, arm, x, backend, feats=False):
    """Run one forward and return (output, latency_ms, peak_vram_MB|None, backend_label, meta)."""
    model, meta = get_model(job, arm)
    kind, prec = BACKENDS.get(backend, ("pytorch", None))

    if CUDA:
        torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    used = "PyTorch"
    if kind == "trt" and CUDA and trt_backend.available():
        epath = trt_backend.engine_path(ENGINES, job, arm, prec, feats)
        if os.path.exists(epath):
            try:
                t0 = time.time()
                out = trt_backend.infer(epath, x)
                torch.cuda.synchronize()
                lat = (time.time() - t0) * 1000
                vram = torch.cuda.max_memory_allocated() / 1e6
                return out, lat, vram, f"TensorRT-{prec.upper()}", meta
            except Exception as e:  # noqa
                used = f"PyTorch (TRT failed: {type(e).__name__})"
        else:
            used = "PyTorch (no engine — run build_trt.py)"

    t0 = time.time()
    out = embed(model, x) if feats else model(x)
    if CUDA:
        torch.cuda.synchronize()
    lat = (time.time() - t0) * 1000
    vram = (torch.cuda.max_memory_allocated() / 1e6) if CUDA else None
    return out, lat, vram, used, meta


def _resource_table(rows):
    """rows: (arm, meta, lat, vram_measured|None, backend_label). Shows measured + report VRAM."""
    out = ["| Model | Backend | Params (M) | Latency (ms) | Peak VRAM measured (MB) | Peak VRAM report (A5000, FP16) |",
           "|---|---|---|---|---|---|"]
    for arm, meta, lat, vram, blabel in rows:
        if meta is None:
            out.append(f"| {ARM_NAME.get(arm, arm)} | — | — | — | — | — |")
            continue
        meas = f"{vram:.0f}" if vram is not None else "— (CPU)"
        ref = _META.get(arm, {}).get("vram_fp16_MB", "—")
        out.append(f"| {ARM_NAME.get(arm, arm)} | {blabel} | {meta['params_M']:.1f} | "
                   f"{lat:.1f} | {meas} | {ref} |")
    return "\n".join(out)


def _predict_dict(spec, out):
    if spec["task"] == "attribute":
        probs = torch.sigmoid(out)[0].float().cpu().numpy()
        top = np.argsort(-probs)[:TOP_K]
        return {CELEBA_ATTRS[i]: float(probs[i]) for i in top}
    probs = out.float().softmax(-1)[0].cpu()
    v, idx = probs.topk(TOP_K)
    return {f"identity #{int(c)}": float(p) for p, c in zip(v, idx)}


def run_classify(job, image, backend):
    if image is None:
        return {}, {}, "Please provide an image."
    spec = JOBS[job]
    x = PREPROC(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    labels, res_rows = {"baseline": {}, "ghost": {}}, []
    for arm in ("baseline", "ghost"):
        try:
            out, lat, vram, blabel, meta = infer_one(job, arm, x, backend)
        except FileNotFoundError:
            res_rows.append((arm, None, None, None, None)); continue
        res_rows.append((arm, meta, lat, vram, blabel))
        labels[arm] = _predict_dict(spec, out)
    return labels["baseline"], labels["ghost"], "#### Resource usage\n" + _resource_table(res_rows)


def run_verify(image_a, image_b, backend):
    job = "LFW"
    if image_a is None or image_b is None:
        return "Please provide two images.", ""
    xa = PREPROC(image_a.convert("RGB")).unsqueeze(0).to(DEVICE)
    xb = PREPROC(image_b.convert("RGB")).unsqueeze(0).to(DEVICE)
    res_rows, cards = [], []
    for arm in ("baseline", "ghost"):
        try:
            ea, lat_a, vram, blabel, meta = infer_one(job, arm, xa, backend, feats=True)
            eb, lat_b, _, _, _ = infer_one(job, arm, xb, backend, feats=True)
        except FileNotFoundError:
            res_rows.append((arm, None, None, None, None)); continue
        ea = torch.nn.functional.normalize(ea.float(), dim=-1)
        eb = torch.nn.functional.normalize(eb.float(), dim=-1)
        cos = float((ea * eb).sum().cpu())
        verdict = "SAME person" if cos > 0.5 else "DIFFERENT person"
        cards.append(f"**{ARM_NAME[arm]}** — cosine similarity `{cos:+.3f}` → **{verdict}**")
        res_rows.append((arm, meta, (lat_a + lat_b) / 2, vram, blabel))
    body = "\n\n".join(cards) if cards else "_No checkpoints bundled._"
    body += "\n\n_Threshold 0.5 on L2-normalised embeddings._"
    return body, "#### Resource usage\n" + _resource_table(res_rows)


CSS = """
.gradio-container {background:#f4f6fb !important; font-family:Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;}
#hero {background:#ffffff; border:1px solid #e6eaf2; border-radius:14px; padding:20px 24px; margin-bottom:8px;}
#hero h1 {font-size:24px; font-weight:700; color:#132845; margin:0 0 4px;}
#hero p {color:#5b6b84; margin:0; font-size:15px;}
#devbar {background:#eef3ff; border:1px solid #d6e2ff; border-radius:10px; padding:10px 14px; color:#243b63; font-size:14px;}
button.primary, .gr-button-primary {background:#2563eb !important; border:none !important; color:#fff !important; font-weight:600 !important;}
table {font-size:14px; border-collapse:collapse;} table th {background:#f0f3fa; color:#243b63;}
footer {display:none !important;}
"""


def build_ui():
    import gradio as gr
    theme = gr.themes.Default(primary_hue=gr.themes.colors.blue, neutral_hue=gr.themes.colors.slate,
                              font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
    with gr.Blocks(title="Face Model Inference — ConvMAE-Base vs Ghost", theme=theme, css=CSS) as demo:
        gr.HTML("<div id='hero'><h1>Face Model Inference</h1>"
                "<p>ConvMAE-Base vs Ghost+ConvMAE, compared side by side on four face tasks, "
                "with a live peak-VRAM measurement and a selectable inference backend.</p></div>")
        gr.Markdown(gpu_status(), elem_id="devbar")
        backend = gr.Radio(list(BACKENDS.keys()), value="PyTorch", label="Inference backend",
                           info="TensorRT needs engines built on THIS GPU via build_trt.py (engines are not "
                                "portable across GPUs). Missing engine / no TensorRT → falls back to PyTorch.")

        def classify_tab(job_key):
            spec = JOBS[job_key]
            gr.Markdown(f"**{spec['title']}** — {spec['ref']}")
            with gr.Row():
                img = gr.Image(type="pil", label="Input face", height=280)
                with gr.Column():
                    lb = gr.Label(num_top_classes=TOP_K, label=ARM_NAME["baseline"])
                    lg = gr.Label(num_top_classes=TOP_K, label=ARM_NAME["ghost"])
            res = gr.Markdown()
            gr.Button("Run both models", variant="primary").click(
                lambda im, bk, jk=job_key: run_classify(jk, im, bk), inputs=[img, backend], outputs=[lb, lg, res])

        with gr.Tab("CelebA — Attributes"):
            classify_tab("CelebA")
        with gr.Tab("CASIA — Identity"):
            classify_tab("CASIA")
        with gr.Tab("SCface — Low-res Identity"):
            classify_tab("SCface")
        with gr.Tab("LFW — Verification"):
            spec = JOBS["LFW"]
            gr.Markdown(f"**{spec['title']}** — {spec['ref']}")
            with gr.Row():
                ia = gr.Image(type="pil", label="Face A", height=280)
                ib = gr.Image(type="pil", label="Face B", height=280)
            verdict = gr.Markdown(); res2 = gr.Markdown()
            gr.Button("Verify (both models)", variant="primary").click(
                run_verify, inputs=[ia, ib, backend], outputs=[verdict, res2])

        gr.Markdown("Reference metrics are the paper's measured multi-seed A5000 results; this demo runs a single image live.")
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
