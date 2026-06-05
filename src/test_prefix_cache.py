"""Validate prefix KV caching: a request that REUSES a snapshotted prefix must produce byte-identical
output to a cold full-prefill of the same prompt. Two prompts share a 3-chunk (384-token) prefix but
differ in the suffix -- exactly the system-prompt + varying-user case."""
import sys, numpy as np
sys.argv = [sys.argv[0]]   # vk_engine reads argv for NGEN; keep default
import vk_engine as E

MC = E.MC
tok = E.tok
# build a >=384-token shared prefix (3 MC chunks) + two different suffixes
base = ("You are a meticulous assistant. Follow the rules exactly and answer concisely. " * 60)
pref = tok(base, return_tensors=None)["input_ids"][:3 * MC]      # exactly 384 shared tokens
assert len(pref) == 3 * MC, f"prefix is {len(pref)} tokens, need {3*MC}"
sufA = tok(" Question: what is two plus two? Answer:", return_tensors=None)["input_ids"]
sufB = tok(" Question: name a primary color. Answer:", return_tensors=None)["input_ids"]
P1 = [int(x) for x in pref + sufA]
P2 = [int(x) for x in pref + sufB]

def run(ids, reuse=0, snap=0, k=6):
    return list(E.generate(ids, max_new=k, temperature=0.0, reuse_chunks=reuse, snap_chunks=snap))

print("[test] cold reference for P2 ...", flush=True)
ref = run(P2)                                  # cold full prefill, no cache
print("  ref:", ref, repr(tok.decode(ref)), flush=True)
print("[test] snapshot P1's 3-chunk prefix ...", flush=True)
_ = run(P1, snap=3)                            # snapshot the shared prefix (P1 and P2 share it)
print("[test] P2 reusing the cached prefix ...", flush=True)
cached = run(P2, reuse=3)                      # restore prefix KV, prefill only the suffix
print("  cached:", cached, repr(tok.decode(cached)), flush=True)

ok = cached == ref
print(f"\nPREFIX-CACHE {'PASS' if ok else 'FAIL'}: cached {'==' if ok else '!='} cold", flush=True)
sys.exit(0 if ok else 1)
