#!/usr/bin/env python3
"""Rigorous inference benchmark (bias-controlled). RUN ON THE GPU BOX.

Fixes vs v1 (which the user rightly flagged for possible bias):
  * torch.backends.cudnn.benchmark = True  -> autotunes conv algorithms (v1 left it off,
    which penalizes conv/depthwise models like Ghost).
  * warmup 30 + 100 timed iters, 3 repeats, report MEDIAN (v1 used warmup 10 / 50 iters).
  * explicit TF32 control; FP32 measured with TF32 OFF (true FP32) and separately with TF32 ON.
  * CUDA events for timing + synchronize; fixed input reused (weights fixed).
Only base/ghost (conv arms) run here (gpu128 has no mamba-ssm).
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
        miss,unexp=m.load_state_dict(sd,strict=False)
        print(f"  [{arm}] loaded (missing {len(miss)} unexpected {len(unexp)})")
    return m.eval()

@torch.no_grad()
def bench(model,dev,bs,dtype,warmup=30,iters=100,repeats=3):
    model=model.to(device=dev,dtype=dtype)
    x=torch.randn(bs,3,224,224,device=dev,dtype=dtype)
    for _ in range(warmup): model(x)
    torch.cuda.synchronize()
    thrs=[]
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        s=torch.cuda.Event(True); e=torch.cuda.Event(True)
        s.record()
        for _ in range(iters): model(x)
        e.record(); torch.cuda.synchronize()
        dt=s.elapsed_time(e)/1000.0
        thrs.append(bs*iters/dt)
    vram=torch.cuda.max_memory_allocated()/1e6
    return round(st.median(thrs),1), round(min(thrs),1), round(max(thrs),1), round(vram,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--models",nargs="+",default=["baseline","ghost"])
    ap.add_argument("--bs",type=int,default=32); ap.add_argument("--iters",type=int,default=100)
    a=ap.parse_args()
    assert torch.cuda.is_available(),"need CUDA"
    torch.backends.cudnn.benchmark=True
    dev=torch.device("cuda")
    print("GPU:",torch.cuda.get_device_name(0),"| torch",torch.__version__,"| cudnn.benchmark=True")
    DT={"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}
    out={}
    for arm in a.models:
        print(f"\n=== {arm} ===")
        m=build(arm); out[arm]={"params_M":round(sum(p.numel() for p in m.parameters())/1e6,3)}
        for tf32 in (False,True):
            torch.backends.cuda.matmul.allow_tf32=tf32; torch.backends.cudnn.allow_tf32=tf32
            med,lo,hi,vram=bench(m,dev,a.bs,DT["fp32"],iters=a.iters)
            tag=f"fp32{'+tf32' if tf32 else ''}"
            out[arm][tag]={"throughput_med":med,"min":lo,"max":hi,"peak_vram_MB":vram}
            print(f"  {tag}: {med} img/s (min {lo} max {hi}) | VRAM {vram} MB")
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        for pk in ("fp16","bf16"):
            med,lo,hi,vram=bench(m,dev,a.bs,DT[pk],iters=a.iters)
            out[arm][pk]={"throughput_med":med,"min":lo,"max":hi,"peak_vram_MB":vram}
            print(f"  {pk}: {med} img/s (min {lo} max {hi}) | VRAM {vram} MB")
        del m; torch.cuda.empty_cache()
    json.dump(out,open(os.path.join(REPO,"scripts","gpu_bench_v2_result.json"),"w"),indent=2)
    print("\n"+json.dumps(out,indent=2))

if __name__=="__main__": main()
