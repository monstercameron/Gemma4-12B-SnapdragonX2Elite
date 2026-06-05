"""Find the remaining overflow op inside safe2 shard 3 (layers 12-15). Feed the REAL input
(CPU trace[2]) and expose per-layer matmuls + residual boundaries; first inf/huge on HTP is
the culprit. Tries single-token AND a filled-KV stress to trigger the multi-token NaN."""
import os, numpy as np, onnx, onnxruntime as ort
from engine_gemma import ShardEngine
SH = "out/gemma4_fp16_safe2/fp16shard_3.onnx"; WIN, H = 64, 3840

import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu = [d for d in ort.get_ep_devices() if d.ep_name=="QNNExecutionProvider"
       and d.device.type==ort.OrtHardwareDeviceType.NPU][0]

# real input entering shard 3 (= CPU output of shard 2)
eng = ShardEngine(backend="cpu", max_live=12); eng.trace={}; eng.reset(); eng.forward(818,0)
hin = eng.trace[2].astype(np.float16)
print(f"real shard-3 input |x|={np.abs(eng.trace[2]).max():.1f}", flush=True)

TAPS = []
for L in range(4):
    for nm in ("o_proj","mlp/gate_proj","mlp/up_proj","mlp/down_proj"):
        TAPS.append(f"/layers.{L}/self_attn/{nm}/MatMul_output_0" if nm=="o_proj"
                    else f"/layers.{L}/{nm}/MatMul_output_0")
    TAPS.append(f"/layers.{L}/Add_1_output_0")   # layer residual output

m = onnx.load(SH)
inf = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
vi = {v.name: v for v in list(inf.graph.value_info)+list(inf.graph.output)}
have = {o.name for o in m.graph.output}
for t in TAPS:
    if t in vi and t not in have: m.graph.output.append(vi[t])
tp = SH.replace(".onnx","_taps.onnx"); onnx.save(m, tp)

mask = np.full((1,1,1,WIN+1),np.float16(-30000.0),np.float16); mask[...,-1]=0
def feeds(s):
    f={"hidden_in":hin,"position_ids":np.array([[0]],np.int64),"attention_mask":mask}
    for i in s.get_inputs():
        if i.name.startswith("past_key_values."):
            f[i.name]=np.zeros([d if isinstance(d,int) else 1 for d in i.shape],np.float16)
    return f
cse=ort.InferenceSession(tp,providers=["CPUExecutionProvider"])
so=ort.SessionOptions(); so.add_provider_for_devices([npu],{"backend_path":q.get_qnn_htp_path(),"htp_performance_mode":"burst"})
hse=ort.InferenceSession(tp,sess_options=so)
names=[o.name for o in cse.get_outputs()]
cpu=dict(zip(names,cse.run(None,feeds(cse)))); htp=dict(zip(names,hse.run(None,feeds(hse))))
print(f"\n{'tap':>40} {'cos':>8} {'cpu|x|':>8} {'htp|x|':>10} {'inf':>4}")
for t in [x for _,x in [(0,t) for t in TAPS]]:
    if t not in cpu: continue
    c=cpu[t].astype(np.float32).ravel(); h=htp[t].astype(np.float32).ravel()
    cos=float(np.dot(c,h)/(np.linalg.norm(c)*np.linalg.norm(h)+1e-9))
    print(f"{t.replace('/layers.',''):>40} {cos:>8.3f} {np.abs(c).max():>8.1f} {np.abs(h).max():>10.1f} {str(np.isinf(h).any()):>4}", flush=True)
