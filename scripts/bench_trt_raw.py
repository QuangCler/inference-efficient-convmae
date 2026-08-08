#!/usr/bin/env python3
"""Real TensorRT benchmark via the raw TensorRT 10 Python API + torch CUDA buffers (no pycuda,
no ONNX-Runtime graph partitioning). Builds ONE engine per (model, precision) from the exported
ONNX, then times BS=32 inference. Run with python3.10 on the A5000."""
import os, sys, time, json
REPO=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import torch, tensorrt as trt
BS=32; ITERS=1000; WARM=30
LOG=trt.Logger(trt.Logger.ERROR)

def build_engine(onnx_path, fp16):
    b=trt.Builder(LOG)
    net=b.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    p=trt.OnnxParser(net,LOG)
    with open(onnx_path,"rb") as f:
        if not p.parse(f.read()):
            for i in range(p.num_errors): print("  parse err:",p.get_error(i))
            return None
    cfg=b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4<<30)
    if fp16: cfg.set_flag(trt.BuilderFlag.FP16)
    ser=b.build_serialized_network(net,cfg)
    rt=trt.Runtime(LOG)
    return rt.deserialize_cuda_engine(ser)

def bench(engine):
    ctx=engine.create_execution_context()
    # tensor names
    names=[engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inp_name=names[0]; out_name=names[-1]
    inp=torch.randn(BS,3,224,224,device="cuda",dtype=torch.float32).contiguous()
    oshape=tuple(engine.get_tensor_shape(out_name))
    out=torch.empty(oshape,device="cuda",dtype=torch.float32).contiguous()
    ctx.set_tensor_address(inp_name,inp.data_ptr())
    ctx.set_tensor_address(out_name,out.data_ptr())
    s=torch.cuda.Stream()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARM): ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()
    t0=time.perf_counter()
    for _ in range(ITERS): ctx.execute_async_v3(s.cuda_stream)
    torch.cuda.synchronize()
    dt=time.perf_counter()-t0
    free,total=torch.cuda.mem_get_info(); vram=round((total-free)/1e6,1)
    return round(BS*ITERS/dt,1), vram

out={}
for arm in ("baseline","ghost"):
    onnx_path=os.path.join(REPO,"scripts",f"{arm}_cls_bs{BS}.onnx")
    out[arm]={}
    for prec,fp16 in (("trt_fp32",False),("trt_fp16",True)):
        print(f"building {arm} {prec} ...", flush=True)
        eng=build_engine(onnx_path,fp16)
        if eng is None: out[arm][prec]="build_failed"; continue
        thr,vram=bench(eng); out[arm][prec]={"throughput":thr,"vram_MB":vram}
        print(f"  {arm} {prec}: {thr} img/s | VRAM {vram} MB", flush=True)
        del eng
json.dump(out,open(os.path.join(REPO,"scripts","bench_trt_raw_result.json"),"w"),indent=2)
print("DONE", json.dumps(out))
