"""Multi-token validation of patched shard 0: run all 20 chat tokens through shard 0
(shard-major, rolling KV) on CPU and HTP, compare last-token hidden. Tests whether the
fp16-safe attention fixes the multi-token divergence (was cos 0.80 norm-only)."""
import os, json, numpy as np, onnxruntime as ort

SHARD = "out/_vtest/fp16shard_0.onnx"
EMB = np.load("out/gemma4_fp16_safe/embed_tokens.npy", mmap_mode="r")
scale = np.float32(json.load(open("out/gemma4_fp16_safe/shards.json"))["embed_scale"])
WIN, H = 64, 3840
IDS = [2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,100,45518,107,101]

import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu = [d for d in ort.get_ep_devices() if d.ep_name=="QNNExecutionProvider"
       and d.device.type==ort.OrtHardwareDeviceType.NPU][0]

def make(htp):
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1,(os.cpu_count() or 2)-1)
    if htp: so.add_provider_for_devices([npu],{"backend_path":q.get_qnn_htp_path(),"htp_performance_mode":"burst"})
    return ort.InferenceSession(SHARD, sess_options=so)

def prefill(sess):
    meta = [(i.name,i.type,[d if isinstance(d,int) else 1 for d in i.shape]) for i in sess.get_inputs()]
    cache = {n: np.zeros(shp,np.float16) for n,t,shp in meta if n.startswith("past_key_values.")}
    last = None
    for pos,tid in enumerate(IDS):
        hidden = (np.asarray(EMB[tid],np.float32)*scale).reshape(1,1,H).astype(np.float16)
        mask = np.full((1,1,1,WIN+1),np.float16(-30000.0),np.float16)
        mask[..., WIN+1-min(pos+1,WIN+1):] = 0
        feeds = {"hidden_in":hidden,"position_ids":np.array([[pos]],np.int64),"attention_mask":mask}
        for n in cache: feeds[n]=cache[n]
        outs = sess.run(None,feeds); od = dict(zip([o.name for o in sess.get_outputs()],outs))
        for k,v in od.items():
            if k.startswith("present."):
                if v.shape[2]==WIN+1: v=v[:,:,1:,:]
                cache["past_key_values."+k[len("present."):]] = v.astype(np.float16)
        last = od["hidden_out"].astype(np.float32)
    return last

cpu = prefill(make(False)); htp = prefill(make(True))
cos = float(np.dot(cpu.ravel(),htp.ravel())/(np.linalg.norm(cpu)*np.linalg.norm(htp)+1e-9))
print(f"shard0 multi-token (20 tok): cos(htp,cpu)={cos:.5f}  cpu|x|={np.abs(cpu).max():.1f}  "
      f"htp|x|={np.abs(htp).max():.1f}  nan={np.isnan(htp).any()}")
