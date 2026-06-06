"""Reproducible engine benchmark for the modular optimization loop. Measures, via the engine directly
(no HTTP), the metrics that matter for int4-preserving changes:
  - decode tok/s at SHORT context (weight-bandwidth bound; baseline decode)
  - decode tok/s at LONG context (KV-read traffic now matters -> where KV-quant pays off)
  - prefill prompt-tok/s
Greedy/fixed token feed so timing is pure forward() cost (degeneration guard off via env)."""
import sys, os, time, numpy as np
sys.argv = [sys.argv[0]]; os.environ.setdefault("GEMMA4_REPEAT_LIMIT", "0")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import vk_engine as E
rng = np.random.default_rng(0)
MC = E.MC; V = E.V

def _prefill(ids):
    n = len(ids); nfull = (n - 1) // MC if n > MC else 0
    t0 = time.time()
    for c in range(nfull): E.forward_prefill(ids[c * MC:(c + 1) * MC], c * MC)
    logits = None
    for pos in range(nfull * MC, n): logits = E.forward(ids[pos], pos)
    return time.time() - t0, n, logits

def bench(prefill_len, ngen=64, warm=6):
    ids = [int(x) for x in rng.integers(100, V - 100, prefill_len)]
    t_pre, n, _ = _prefill(ids)
    pos = n
    for _ in range(warm): E.forward(100, pos); pos += 1     # warm steady-state clocks
    t0 = time.time()
    for _ in range(ngen): E.forward(100, pos); pos += 1
    dt = time.time() - t0
    return ngen / dt, prefill_len / t_pre

def run(tag):
    print(f"=== BENCH [{tag}] ===", flush=True)
    # decode at short / mid / long context (KV grows -> KV traffic share grows)
    for plen in (16, 1024, 4096):
        # best of 2 to dodge thermal noise
        d = max(bench(plen)[0] for _ in range(2))
        print(f"  decode @ctx~{plen:5}: {d:6.2f} tok/s", flush=True)
    # prefill rate on a 2048-token prompt
    _, pr = bench(2048, ngen=1)
    print(f"  prefill           : {pr:6.1f} prompt-tok/s", flush=True)

run(os.environ.get("BENCH_TAG", "baseline"))
