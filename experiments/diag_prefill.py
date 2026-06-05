"""Real-context per-shard divergence: run the FULL 20-token chat prefill on CPU and on HTP,
tracing each shard's last-token hidden output, and compare. This captures KV/multi-token
effects the single-token probe missed.

Run under QNN venv: <pipeline>/.venv/Scripts/python.exe scripts/diag_prefill.py
"""
import numpy as np
from engine_gemma import ShardEngine

IDS = [2,105,2364,107,3689,563,506,5279,529,7001,236881,106,107,105,4368,107,100,45518,107,101]


def trace_backend(backend, cpu_shards=None):
    e = ShardEngine(backend=backend, max_live=(12 if backend == "cpu" else 4), cpu_shards=cpu_shards)
    e.trace = {}
    e.reset()
    logits = e.prefill(IDS)
    return {s: e.trace[s] for s in sorted(e.trace)}, int(np.argmax(logits))


import sys, os
sys.path.insert(0, os.path.dirname(__file__))

cpu_tr, cpu_arg = trace_backend("cpu")
htp_tr, htp_arg = trace_backend("npu")   # all shards on HTP

print(f"\n{'shard':>5} {'cos(htp,cpu)':>13} {'cpu|x|':>9} {'htp|x|':>9}")
for s in sorted(cpu_tr):
    c, h = cpu_tr[s].ravel(), htp_tr[s].ravel()
    cos = float(np.dot(c, h) / (np.linalg.norm(c) * np.linalg.norm(h) + 1e-9))
    print(f"{s:>5} {cos:>13.5f} {np.abs(c).max():>9.1f} {np.abs(h).max():>9.1f}", flush=True)
print(f"\nCPU argmax={cpu_arg}  HTP(all)argmax={htp_arg}")
