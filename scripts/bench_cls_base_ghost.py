#!/usr/bin/env python3
"""Inference benchmark of the ConvMAE-Base and Ghost+ConvMAE CLS models — PyTorch + TensorRT.

Mirrors the linear-probe bench_cls flow but for the two selected convolutional arms. Measures,
at BS=32, throughput (img/s, median of 3×100 iters) and peak VRAM for:
  * PyTorch  : FP32 (TF32 on/off), FP16, BF16   (cuDNN autotuning on, 30-iter warm-up)
  * TensorRT : FP32, FP16   (via ONNX export)   -- SKIPPED gracefully if tensorrt/onnx absent

Run on a GPU box (from repo root):
    python scripts/bench_cls_base_ghost.py --bs 32 --iters 100
Output: scripts/bench_cls_base_ghost_result.json  (+ printed table).

Note: gpu128 (A5000) lacks tensorrt/onnx, so it produces the PyTorch rows only; run this on a
TensorRT-capable box (e.g. the RTX 5090 research server) to obtain the TensorRT rows.
"""
import argparse, json, os, sys, statistics as st
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,REPO); sys.path.insert(0,os.path.join(REPO,"finetune","common"))
import torch

BUILD={"baseline":("convvit_base_patch16","models/convmae_base_pretrain_epoch300.pt"),
       "ghost":("convvit_ghost_base_patch16","models/allghost_epoch_300.pt")}

def build(arm):
    import models_finetune
    name,ckpt=BUILD[arm]; m=models_finetune.build_model(name,num_classes=1000)
    p=os.path.join(REPO,ckpt)
    if os.path.exists(p):
        sd=torch.load(p,map_location="cpu",weights_only=False); sd=sd.get("model",sd.get("state_dict",sd))
        miss,unexp=m.load_state_dict(sd,strict=False); print(f"  [{arm}] loaded ({len(miss)} missing/{len(unexp)} unexpected)")
    return m.eval()

@torch.no_grad()
def bench_pt(model,dev,bs,dtype,warmup=30,iters=100,reps=3):
    model=model.to(device=dev,dtype=dtype); x=torch.randn(bs,3,224,224,device=dev,dtype=dtype)
    for _ in range(warmup): model(x)
    torch.cuda.synchronize(); thrs=[]
    for _ in range(reps):
        torch.cuda.reset_peak_memory_stats(); s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record()
        for _ in range(iters): model(x)
        e.record(); torch.cuda.synchronize(); thrs.append(bs*iters/(s.elapsed_time(e)/1000.0))
    return round(st.median(thrs),1), round(torch.cuda.max_memory_allocated()/1e6,1)

def try_trt(model,arm,bs,iters):
    """Export to ONNX, build FP32/FP16 TRT engines, benchmark. Returns dict or None if unavailable."""
    try:
        import tensorrt as trt, onnx  # noqa
        import numpy as np, pycuda.driver as cuda, pycuda.autoinit  # noqa
    except Exception as e:
        print(f"  [{arm}] TensorRT/ONNX not available ({type(e).__name__}); skipping TRT rows.")
        return None
    onnx_path=os.path.join(REPO,"scripts",f"{arm}_cls.onnx")
    dummy=torch.randn(bs,3,224,224)
    torch.onnx.export(model.cpu().float(),dummy,onnx_path,input_names=["x"],output_names=["logits"],
                      opset_version=17,do_constant_folding=True,dynamic_axes=None)
    out={}
    LOG=trt.Logger(trt.Logger.WARNING)
    for prec in ("fp32","fp16"):
        builder=trt.Builder(LOG); net=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(net,LOG); parser.parse(open(onnx_path,"rb").read())
        cfg=builder.create_builder_config(); cfg.max_workspace_size=4<<30
        if prec=="fp16" and builder.platform_has_fast_fp16: cfg.set_flag(trt.BuilderFlag.FP16)
        eng=builder.build_engine(net,cfg); ctx=eng.create_execution_context()
        # allocate + time
        inp=np.random.randn(bs,3,224,224).astype(np.float32); d_in=cuda.mem_alloc(inp.nbytes)
        outshape=(bs,1000); d_out=cuda.mem_alloc(int(np.prod(outshape))*4)
        cuda.memcpy_htod(d_in,inp); import time
        for _ in range(20): ctx.execute_v2([int(d_in),int(d_out)])
        cuda.Context.synchronize(); t0=time.perf_counter()
        for _ in range(iters): ctx.execute_v2([int(d_in),int(d_out)])
        cuda.Context.synchronize(); dt=time.perf_counter()-t0
        out[f"trt_{prec}"]={"throughput_img_s":round(bs*iters/dt,1)}
        print(f"  [{arm}] TRT {prec}: {out[f'trt_{prec}']['throughput_img_s']} img/s")
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--bs",type=int,default=32); ap.add_argument("--iters",type=int,default=100)
    a=ap.parse_args(); assert torch.cuda.is_available(),"need CUDA"
    torch.backends.cudnn.benchmark=True; dev=torch.device("cuda")
    print("GPU:",torch.cuda.get_device_name(0),"| torch",torch.__version__)
    DT={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}; out={}
    for arm in ("baseline","ghost"):
        print(f"\n=== {arm} ==="); m=build(arm)
        out[arm]={"params_M":round(sum(p.numel() for p in m.parameters())/1e6,3)}
        for tf32 in (False,True):
            torch.backends.cuda.matmul.allow_tf32=tf32; torch.backends.cudnn.allow_tf32=tf32
            thr,vram=bench_pt(m,dev,a.bs,DT["fp32"],iters=a.iters); out[arm][f"pt_fp32{'+tf32' if tf32 else ''}"]={"throughput_img_s":thr,"peak_vram_MB":vram}
            print(f"  PyTorch fp32{'+tf32' if tf32 else ''}: {thr} img/s | {vram} MB")
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        for pk in ("fp16","bf16"):
            thr,vram=bench_pt(m,dev,a.bs,DT[pk],iters=a.iters); out[arm][f"pt_{pk}"]={"throughput_img_s":thr,"peak_vram_MB":vram}
            print(f"  PyTorch {pk}: {thr} img/s | {vram} MB")
        trt=try_trt(m,arm,a.bs,a.iters)
        if trt: out[arm].update(trt)
        del m; torch.cuda.empty_cache()
    js=os.path.join(REPO,"scripts","bench_cls_base_ghost_result.json"); json.dump(out,open(js,"w"),indent=2)
    print("\nsaved",js); print(json.dumps(out,indent=2))

if __name__=="__main__": main()
