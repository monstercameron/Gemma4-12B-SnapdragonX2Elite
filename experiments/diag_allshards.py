"""Localize HTP divergence: single token (pos=0, empty KV) through all 12 shards. For each
shard, feed the SAME CPU-correct input to CPU and HTP and compare — isolates per-shard error
(no compounding). A uniform ~0.999 cos => fp16 accumulation; a sudden crash => a bad op/shard
(suspect: the full-attention layers, head_dim 512 / MQA / k=v).

Run under QNN venv: <pipeline>/.venv/Scripts/python.exe scripts/diag_allshards.py
"""
import os, json, numpy as np, onnxruntime as ort

DIR = "out/gemma4_fp16"
m = json.load(open(f"{DIR}/shards.json"))
WIN, H = m["win"], m["hidden"]
EMB = np.load(f"{DIR}/embed_tokens.npy", mmap_mode="r")
scale = np.float32(m["embed_scale"])
LT = m["layer_types"]; SH = m["shards"]

mask = np.full((1, 1, 1, WIN + 1), np.float16(-30000.0), np.float16); mask[..., -1] = 0
pos = np.array([[0]], np.int64)

import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu = [d for d in ort.get_ep_devices()
       if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU][0]


def sess(s, htp):
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
    path = f"{DIR}/ctx_fp16shard_{s}.onnx" if htp else f"{DIR}/fp16shard_{s}.onnx"
    if htp:
        so.add_provider_for_devices([npu], {"backend_path": q.get_qnn_htp_path(),
                                            "htp_performance_mode": "burst"})
    return ort.InferenceSession(path, sess_options=so)


def feeds(se, hidden):
    f = {"hidden_in": hidden, "position_ids": pos, "attention_mask": mask}
    for i in se.get_inputs():
        if i.name.startswith("past_key_values."):
            shp = [d if isinstance(d, int) else 1 for d in i.shape]
            f[i.name] = np.zeros(shp, np.float16)
    return f


hidden = (np.asarray(EMB[818], np.float32) * scale).reshape(1, 1, H).astype(np.float16)
print(f"{'shard':>5} {'layers':>10} {'types':>16} {'cos':>9} {'max_abs':>9} {'cpu|x|':>8} {'htp|x|':>8}")
for s in range(len(SH)):
    cse, hse = sess(s, False), sess(s, True)
    cpu = cse.run(None, feeds(cse, hidden))[0].astype(np.float32)
    htp = hse.run(None, feeds(hse, hidden))[0].astype(np.float32)
    cos = float(np.dot(cpu.ravel(), htp.ravel()) / (np.linalg.norm(cpu) * np.linalg.norm(htp)))
    tset = "+".join(sorted({("F" if LT[g] == "full_attention" else "s") for g in SH[s]}))
    print(f"{s:>5} {str(SH[s][0])+'-'+str(SH[s][-1]):>10} {tset:>16} {cos:>9.5f} "
          f"{np.abs(cpu-htp).max():>9.4f} {np.abs(cpu).max():>8.1f} {np.abs(htp).max():>8.1f}", flush=True)
    hidden = cpu.astype(np.float16)   # feed CPU-correct forward so errors don't compound
