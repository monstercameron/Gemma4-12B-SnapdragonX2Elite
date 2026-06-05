"""Per-LAYER HTP-vs-CPU isolation for the decomposed bad-shard layers. Two stress modes:
  A) pos=0, empty KV (single token) — finds finite blow-ups.
  B) pos=40, FULL valid KV window of realistic-magnitude keys — stresses the attention/KV
     path (suspected source of the multi-token NaN).

Run (QNN venv): <pipeline>/.venv/Scripts/python.exe scripts/diag_layers.py
"""
import os, json, numpy as np, onnxruntime as ort

DIR = "out/gemma4_decomp"
WIN, H = 64, 3840
LAYERS = [12, 13, 14, 15, 20, 21, 22, 23]
rng = np.random.default_rng(0)

import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu = [d for d in ort.get_ep_devices()
       if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU][0]


FP16P = os.environ.get("HTP_FP16_PRECISION", "1")
def make_sess(path, htp):
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
    if htp:
        so.add_provider_for_devices([npu], {"backend_path": q.get_qnn_htp_path(),
                                            "htp_performance_mode": "burst",
                                            "enable_htp_fp16_precision": FP16P})
    return ort.InferenceSession(path, sess_options=so)


def feeds(se, hidden, pos, valid):
    mask = np.full((1, 1, 1, WIN + 1), np.float16(-30000.0), np.float16)
    mask[..., WIN + 1 - valid:] = 0
    f = {"hidden_in": hidden, "position_ids": np.array([[pos]], np.int64), "attention_mask": mask}
    for i in se.get_inputs():
        if i.name.startswith("past_key_values."):
            shp = [d if isinstance(d, int) else 1 for d in i.shape]
            # mode B: fill the valid KV slots with realistic-magnitude keys/values
            arr = np.zeros(shp, np.float16)
            if valid > 1:
                arr[:, :, WIN - (valid - 1):, :] = (rng.standard_normal(
                    (shp[0], shp[1], valid - 1, shp[3])) * 0.5).astype(np.float16)
            f[i.name] = arr
    return f


def test(layer, pos, valid, label):
    path = f"{DIR}/layer_{layer}.onnx"
    hidden = (rng.standard_normal((1, 1, H)) * 5).astype(np.float16)   # realistic mid-net magnitude
    cse, hse = make_sess(path, False), make_sess(path, True)
    cpu = cse.run(None, feeds(cse, hidden, pos, valid))[0].astype(np.float32)
    htp = hse.run(None, feeds(hse, hidden, pos, valid))[0].astype(np.float32)
    cos = float(np.dot(cpu.ravel(), htp.ravel()) / (np.linalg.norm(cpu) * np.linalg.norm(htp) + 1e-9))
    print(f"  L{layer:>2} [{label}] cos={cos:>8.4f} cpu|x|={np.abs(cpu).max():>7.1f} "
          f"htp|x|={np.abs(htp).max():>9.1f} htp_nan={np.isnan(htp).any()} htp_inf={np.isinf(htp).any()}",
          flush=True)


print(f"=== mode A: pos=0, empty KV (fp16p={FP16P}) ===")
for L in [22, 23]:
    test(L, 0, 1, "A")
