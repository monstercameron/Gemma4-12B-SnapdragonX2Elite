"""Op-level pinpoint inside layer 15 (the worst per-layer HTP divergence) with REAL input.
Builds the true hidden entering layer 15 (CPU forward through 12,13,14), then exposes the
internal op outputs and compares CPU vs HTP at each — the FIRST divergent op is the culprit.

Run (QNN venv): PYTHONPATH=scripts <pipeline>/.venv/Scripts/python.exe scripts/diag_ops.py
"""
import numpy as np, onnx, onnxruntime as ort
from engine_gemma import ShardEngine

DEC = "out/gemma4_decomp"; WIN, H = 64, 3840
TAPS = [
    ("q_proj",      "/layers.0/self_attn/q_proj/MatMul_output_0"),
    ("qk_scores",   "/layers.0/self_attn/MatMul_output_0"),
    ("softmax",     "/layers.0/self_attn/Softmax_output_0"),
    ("attn@v",      "/layers.0/self_attn/MatMul_1_output_0"),
    ("o_proj",      "/layers.0/self_attn/o_proj/MatMul_output_0"),
    ("resid_attn",  "/layers.0/Add_output_0"),
    ("mlp_gate",    "/layers.0/mlp/gate_proj/MatMul_output_0"),
    ("mlp_up",      "/layers.0/mlp/up_proj/MatMul_output_0"),
    ("mlp_gate*up", "/layers.0/mlp/Mul_output_0"),
    ("mlp_down",    "/layers.0/mlp/down_proj/MatMul_output_0"),
    ("resid_final", "/layers.0/Add_1_output_0"),
    ("hidden_out",  "hidden_out"),
]

import onnxruntime_qnn as q
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
npu = [d for d in ort.get_ep_devices()
       if d.ep_name == "QNNExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.NPU][0]

# real hidden entering layer 15
eng = ShardEngine(backend="cpu", max_live=12); eng.trace = {}; eng.reset(); eng.forward(818, 0)
h = eng.trace[2].astype(np.float16)               # layer-12 input
for g in (12, 13, 14):                              # CPU-forward to get real layer-15 input
    s = ort.InferenceSession(f"{DEC}/layer_{g}.onnx", providers=["CPUExecutionProvider"])
    mask = np.full((1,1,1,WIN+1), np.float16(-30000.0), np.float16); mask[...,-1]=0
    feeds = {"hidden_in": h, "position_ids": np.array([[0]],np.int64), "attention_mask": mask}
    for i in s.get_inputs():
        if i.name.startswith("past_key_values."):
            feeds[i.name] = np.zeros([d if isinstance(d,int) else 1 for d in i.shape], np.float16)
    h = s.run(None, feeds)[0].astype(np.float16)
print(f"real layer-15 input |x|={np.abs(h.astype(np.float32)).max():.1f}", flush=True)

# build a tapped layer_15 (use shape inference to get correct tensor types)
m = onnx.load(f"{DEC}/layer_15.onnx")
inf = onnx.shape_inference.infer_shapes(m, strict_mode=False, data_prop=True)
vinfo = {vi.name: vi for vi in list(inf.graph.value_info) + list(inf.graph.output) + list(inf.graph.input)}
have = {o.name for o in m.graph.output}
for _, t in TAPS:
    if t not in have:
        if t in vinfo:
            m.graph.output.append(vinfo[t])
        else:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT16, None))
tp = f"{DEC}/layer_15_taps.onnx"; onnx.save(m, tp)

mask = np.full((1,1,1,WIN+1), np.float16(-30000.0), np.float16); mask[...,-1]=0
def feeds_for(s):
    f = {"hidden_in": h, "position_ids": np.array([[0]],np.int64), "attention_mask": mask}
    for i in s.get_inputs():
        if i.name.startswith("past_key_values."):
            f[i.name] = np.zeros([d if isinstance(d,int) else 1 for d in i.shape], np.float16)
    return f

cse = ort.InferenceSession(tp, providers=["CPUExecutionProvider"])
so = ort.SessionOptions(); so.add_provider_for_devices([npu], {"backend_path": q.get_qnn_htp_path(),
                                                               "htp_performance_mode":"burst"})
hse = ort.InferenceSession(tp, sess_options=so)
names = [o.name for o in cse.get_outputs()]
cpu = dict(zip(names, cse.run(None, feeds_for(cse))))
htp = dict(zip(names, hse.run(None, feeds_for(hse))))

print(f"\n{'op':>14} {'cos':>9} {'cpu|x|':>9} {'htp|x|':>10}")
for label, t in TAPS:
    if t not in cpu: continue
    c = cpu[t].astype(np.float32).ravel(); hh = htp[t].astype(np.float32).ravel()
    cos = float(np.dot(c,hh)/(np.linalg.norm(c)*np.linalg.norm(hh)+1e-9))
    print(f"{label:>14} {cos:>9.4f} {np.abs(c).max():>9.1f} {np.abs(hh).max():>10.1f}", flush=True)
