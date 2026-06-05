"""Reliable per-layer HTP divergence with REAL deterministic inputs (no random).
Builds the true hidden state via the CPU engine (token 818, pos=0), then runs layers
12-15 and 20-23 individually on CPU vs HTP, feeding the CPU-correct forward so each layer's
error is isolated. Reports which layer diverges. For the worst layer, also exposes internal
tensors to find the exact op.

Run (QNN venv): PYTHONPATH=scripts <pipeline>/.venv/Scripts/python.exe scripts/diag_layers_real.py
"""
import os, json, numpy as np, onnx, onnxruntime as ort
from engine_gemma import ShardEngine

DEC = "out/gemma4_decomp"
WIN, H = 64, 3840

import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu = [d for d in ort.get_ep_devices()
       if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU][0]

# 1) build the REAL hidden state entering shards via a CPU single-token forward
print("building real hidden via CPU forward...", flush=True)
eng = ShardEngine(backend="cpu", max_live=12)
eng.trace = {}; eng.reset()
eng.forward(818, 0)            # token "The", pos 0
real_in = {3: eng.trace[2].astype(np.float16),   # shard-2 out = layer-12 in
           5: eng.trace[4].astype(np.float16)}   # shard-4 out = layer-20 in
print(f"real input mags: layer12 |x|={np.abs(eng.trace[2]).max():.1f}  "
      f"layer20 |x|={np.abs(eng.trace[4]).max():.1f}", flush=True)

mask = np.full((1, 1, 1, WIN + 1), np.float16(-30000.0), np.float16); mask[..., -1] = 0
pos = np.array([[0]], np.int64)


def sess(path, htp, extra_out=None):
    if extra_out:
        m = onnx.load(path)  # add internal tensors as outputs
        existing = {o.name for o in m.graph.output}
        for t in extra_out:
            if t not in existing:
                m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.UNDEFINED, None))
        path = path.replace(".onnx", "_taps.onnx")
        onnx.save(m, path)
    so = ort.SessionOptions(); so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
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


def run_layers(globs, hidden):
    print(f"\n--- layers {globs[0]}-{globs[-1]} (real forward) ---", flush=True)
    for g in globs:
        p = f"{DEC}/layer_{g}.onnx"
        cse, hse = sess(p, False), sess(p, True)
        cpu = cse.run(None, feeds(cse, hidden))[0].astype(np.float32)
        htp = hse.run(None, feeds(hse, hidden))[0].astype(np.float32)
        cos = float(np.dot(cpu.ravel(), htp.ravel()) / (np.linalg.norm(cpu)*np.linalg.norm(htp)+1e-9))
        print(f"  L{g:>2}: cos={cos:>8.4f} cpu|x|={np.abs(cpu).max():>7.1f} "
              f"htp|x|={np.abs(htp).max():>9.1f} nan={np.isnan(htp).any()}", flush=True)
        hidden = cpu.astype(np.float16)   # feed CPU-correct forward


run_layers([12, 13, 14, 15], real_in[3])
run_layers([20, 21, 22, 23], real_in[5])
