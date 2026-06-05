import numpy as np, onnxruntime as ort
from engine_gemma import ShardEngine
DEC="out/gemma4_decomp"; WIN,H=64,3840
import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu=[d for d in ort.get_ep_devices() if d.ep_name=="QNNExecutionProvider" and d.device.type==ort.OrtHardwareDeviceType.NPU][0]

eng=ShardEngine(backend="cpu",max_live=12); eng.trace={}; eng.reset(); eng.forward(818,0)
h=eng.trace[2].astype(np.float16)
mask=np.full((1,1,1,WIN+1),np.float16(-30000.0),np.float16); mask[...,-1]=0
def fd(s,hid):
    f={"hidden_in":hid,"position_ids":np.array([[0]],np.int64),"attention_mask":mask}
    for i in s.get_inputs():
        if i.name.startswith("past_key_values."):
            f[i.name]=np.zeros([d if isinstance(d,int) else 1 for d in i.shape],np.float16)
    return f
for g in (12,13,14):
    s=ort.InferenceSession(f"{DEC}/layer_{g}.onnx",providers=["CPUExecutionProvider"])
    h=s.run(None,fd(s,h))[0].astype(np.float16)
print(f"real layer-15 input |x|={np.abs(h.astype(np.float32)).max():.1f}")
def run(path,htp):
    so=ort.SessionOptions()
    if htp: so.add_provider_for_devices([npu],{"backend_path":q.get_qnn_htp_path(),"htp_performance_mode":"burst"})
    s=ort.InferenceSession(path,sess_options=so)
    return s.run(None,fd(s,h))[0].astype(np.float32)
for tag,path in (("orig",f"{DEC}/layer_15.onnx"),("SAFE",f"{DEC}/layer_15_safe.onnx")):
    cpu=run(path,False); htp=run(path,True)
    cos=float(np.dot(cpu.ravel(),htp.ravel())/(np.linalg.norm(cpu)*np.linalg.norm(htp)+1e-9))
    print(f"{tag:>5}: cos(htp,cpu)={cos:.5f}  cpu|x|={np.abs(cpu).max():.1f}  htp|x|={np.abs(htp).max():.1f}")
