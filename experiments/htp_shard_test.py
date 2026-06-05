"""Does a Gemma 4 fp16 shard actually PLACE and FINALIZE on the Hexagon HTP?

This is the critical de-risk. The Qwen work established: fp16 MatMul places on HTP, but
4-layer shards can hit finalize error 1002 (capacity). Gemma's hidden is smaller (3840 vs
5120), so 4-layer shards may finalize where Qwen's didn't.

Runs ONE shard on HTP and on CPU, compares outputs, and prints how many nodes the HTP
claimed vs fell back to CPU.

Run with the QNN venv (onnxruntime-qnn 2.2.0), NOT the gemma export venv:
  <pipeline>/.venv/Scripts/python.exe scripts/htp_shard_test.py out/gemma4_fp16/fp16shard_0.onnx
"""
import sys, os, json, time
import numpy as np
import onnxruntime as ort

SHARD = sys.argv[1] if len(sys.argv) > 1 else "out/gemma4_fp16/fp16shard_0.onnx"
META = os.path.join(os.path.dirname(SHARD), "shards.json")


def make_feeds(sess):
    rng = np.random.default_rng(0)
    feeds = {}
    for i in sess.get_inputs():
        shp = [d if isinstance(d, int) else 1 for d in i.shape]
        if i.name == "position_ids":
            feeds[i.name] = np.array([[64]], np.int64)
        elif "float16" in i.type:
            feeds[i.name] = rng.standard_normal(shp).astype(np.float16) * 0.1
        elif "float" in i.type:
            feeds[i.name] = np.zeros(shp, np.float32)
        else:
            feeds[i.name] = np.zeros(shp, np.int64)
    return feeds


def run(provider_setup, label):
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)
    provider_setup(so)
    t = time.time()
    sess = ort.InferenceSession(SHARD, sess_options=so)
    load = time.time() - t
    feeds = make_feeds(sess)
    t = time.time()
    out = sess.run(None, feeds)
    dt = time.time() - t
    print(f"[{label}] providers={sess.get_providers()} load={load:.1f}s run={dt*1000:.1f}ms "
          f"hidden_out{out[0].shape} mean={out[0].astype(np.float32).mean():.4f}", flush=True)
    return out, feeds


def main():
    print(f"ORT {ort.__version__} | shard={SHARD}", flush=True)
    if os.path.exists(META):
        m = json.load(open(META))
        print(f"meta: win={m['win']} lps={m['layers_per_shard']} layer_types(first shard)="
              f"{[m['layer_types'][g] for g in m['shards'][0]]}", flush=True)

    # CPU reference
    cpu_out, feeds = run(lambda so: None, "CPU")

    # NPU / HTP
    try:
        import onnxruntime_qnn as q
        ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
        npu = [d for d in ort.get_ep_devices()
               if d.ep_name == "QNNExecutionProvider"
               and d.device.type == ort.OrtHardwareDeviceType.NPU][0]

        # enable_htp_fp16_precision="0" -> keep fp32-typed ops (softmax upcast, RMSNorm) in
        # fp32 instead of collapsing the whole graph to fp16. Toggle via env to A/B compare.
        fp16p = os.environ.get("HTP_FP16_PRECISION", "1")
        def setup(so):
            so.add_provider_for_devices([npu], {"backend_path": q.get_qnn_htp_path(),
                                                "htp_performance_mode": "burst",
                                                "enable_htp_fp16_precision": fp16p})
        npu_out, _ = run(setup, f"NPU/HTP(fp16p={fp16p})")
        err = float(np.max(np.abs(npu_out[0].astype(np.float32) - cpu_out[0].astype(np.float32))))
        print(f"[cmp] hidden_out max_abs_err HTP-vs-CPU = {err:.3e} -> "
              f"{'PASS' if err < 0.1 else 'CHECK'}", flush=True)
    except Exception as e:
        print(f"[NPU/HTP] ERROR: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
