"""Compile each fp16 Gemma shard ONCE to a QNN context binary (HTP or GPU) so the engine
reloads in ~seconds instead of recompiling each time.

  QNN_BACKEND=npu (default) -> ctx_fp16shard_{s}.onnx     (Hexagon HTP)
  QNN_BACKEND=gpu           -> ctxgpu_fp16shard_{s}.onnx  (Adreno GPU)

Run under the QNN venv:
  GEMMA_DIR=out/gemma4_fp16 QNN_BACKEND=gpu <pipeline>/.venv/Scripts/python.exe scripts/compile_contexts.py
"""
import os, glob, time
import onnxruntime as ort
import onnxruntime_qnn as q

DIR = os.environ.get("GEMMA_DIR", "out/gemma4_fp16")
BACKEND = os.environ.get("QNN_BACKEND", "npu")
NT = max(1, (os.cpu_count() or 2) - 1)
ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())

want = ort.OrtHardwareDeviceType.NPU if BACKEND == "npu" else ort.OrtHardwareDeviceType.GPU
dev = [d for d in ort.get_ep_devices()
       if d.ep_name == "QNNExecutionProvider" and d.device.type == want][0]
backend_path = q.get_qnn_htp_path() if BACKEND == "npu" else q.get_qnn_gpu_path()
prefix = "ctx_" if BACKEND == "npu" else "ctxgpu_"

NSH = len(glob.glob(f"{DIR}/fp16shard_*.onnx"))
print(f"compiling {NSH} shards -> {BACKEND.upper()} context binaries ({prefix}*)", flush=True)
for s in range(NSH):
    ctx = f"{DIR}/{prefix}fp16shard_{s}.onnx"
    if os.path.exists(ctx):
        print(f"shard {s}: ctx exists, skip", flush=True); continue
    so = ort.SessionOptions(); so.intra_op_num_threads = NT
    so.add_session_config_entry("ep.context_enable", "1")
    so.add_session_config_entry("ep.context_file_path", ctx)
    so.add_session_config_entry("ep.context_embed_mode", "0")   # external .bin (avoids 2GB protobuf)
    opts = {"backend_path": backend_path}
    if BACKEND == "npu":
        opts["htp_performance_mode"] = "burst"
    so.add_provider_for_devices([dev], opts)
    t = time.time()
    sess = ort.InferenceSession(f"{DIR}/fp16shard_{s}.onnx", sess_options=so)
    del sess
    print(f"shard {s}: compiled in {time.time()-t:.0f}s -> {os.path.basename(ctx)} "
          f"(exists={os.path.exists(ctx)})", flush=True)
print("ALL CONTEXTS BUILT", flush=True)
