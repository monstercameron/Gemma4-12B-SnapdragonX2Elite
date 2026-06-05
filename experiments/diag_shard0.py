"""Honest per-shard diagnosis on REAL input (not random): run the real first prompt token
through shard 0 on CPU and HTP, compare, and look at activation magnitudes / fp16 range.

Questions:
  - Does any activation approach fp16's 65504 limit (=> HTP native-fp16 overflow)?
  - How big is HTP-vs-CPU divergence on REAL input with a CORRECT pos=0 mask?

Run under the QNN venv: <pipeline>/.venv/Scripts/python.exe scripts/diag_shard0.py
"""
import os, json, numpy as np, onnxruntime as ort

DIR = "out/gemma4_fp16"
m = json.load(open(f"{DIR}/shards.json"))
WIN, H = m["win"], m["hidden"]
EMB = np.load(f"{DIR}/embed_tokens.npy", mmap_mode="r")
embed_scale = np.float32(m["embed_scale"])

# real first prompt token id (from the tokenizer run earlier): "The" -> 818
TOK0 = 818
hidden0 = (np.asarray(EMB[TOK0], np.float32) * embed_scale).reshape(1, 1, H).astype(np.float16)
print(f"input embed: max|x|={np.abs(hidden0.astype(np.float32)).max():.2f} "
      f"(fp16 max=65504); any inf={np.isinf(hidden0).any()}")

# pos=0 mask: only the current token is valid (last slot), rest -inf
mask = np.full((1, 1, 1, WIN + 1), np.float16(-30000.0), np.float16)
mask[..., -1] = np.float16(0.0)
pos = np.array([[0]], np.int64)

shard = f"{DIR}/fp16shard_0.onnx"


def feeds_for(sess):
    f = {"hidden_in": hidden0, "position_ids": pos, "attention_mask": mask}
    for i in sess.get_inputs():
        if i.name.startswith("past_key_values."):
            shp = [d if isinstance(d, int) else 1 for d in i.shape]
            f[i.name] = np.zeros(shp, np.float16)
    return f


def run(setup, label):
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
    setup(so)
    sess = ort.InferenceSession(shard, sess_options=so)
    out = sess.run(None, feeds_for(sess))[0].astype(np.float32)
    print(f"[{label}] hidden_out: max|x|={np.abs(out).max():.2f} mean={out.mean():.4f} "
          f"std={out.std():.4f} any_inf={np.isinf(out).any()} any_nan={np.isnan(out).any()}")
    return out


cpu = run(lambda so: None, "CPU(fp32-accum)")

import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu = [d for d in ort.get_ep_devices()
       if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU][0]
for fp16p in ("1", "0"):
    htp = run(lambda so: so.add_provider_for_devices(
        [npu], {"backend_path": q.get_qnn_htp_path(), "htp_performance_mode": "burst",
                "enable_htp_fp16_precision": fp16p}), f"HTP(fp16p={fp16p})")
    err = np.abs(htp - cpu)
    rel = err / (np.abs(cpu) + 1e-3)
    print(f"   -> vs CPU: max_abs={err.max():.4f} mean_abs={err.mean():.5f} "
          f"max_rel={rel.max():.3f} cos={np.dot(htp.ravel(),cpu.ravel())/(np.linalg.norm(htp)*np.linalg.norm(cpu)):.5f}")
