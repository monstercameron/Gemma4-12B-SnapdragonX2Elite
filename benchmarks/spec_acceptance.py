"""Measure prompt-lookup speculative-decoding ECONOMICS before building it. Generates greedily, then
post-hoc simulates prompt-lookup drafting at each step and records how many tokens it WOULD accept.
Spec decoding wins only if avg-accepted > verify-cost (~8 decode-tokens at int8, ~16 at f16)."""
import sys, os, numpy as np
sys.argv = [sys.argv[0]]; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import vk_engine as E
tok = E.tok

def lookup_draft(seq, t, ng_max=3, K=16):
    """Draft up to K tokens to follow seq[:t] by matching the trailing ng-gram earlier in seq[:t]."""
    for ng in range(ng_max, 0, -1):
        if t < ng: continue
        pat = seq[t - ng:t]
        for i in range(t - ng - 1, ng - 2, -1):          # most-recent earlier occurrence first
            if seq[i:i + ng] == pat:
                return seq[i + ng:i + ng + K]
    return []

def analyse(name, prompt, n_new=120):
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = [int(x) for x in np.array(ids).ravel()]
    plen = len(ids)
    gen = list(E.generate(ids, max_new=n_new, temperature=0.0))
    seq = ids + gen
    accs = []
    t = plen
    while t < len(seq):
        draft = lookup_draft(seq, t)
        a = 0
        for k, d in enumerate(draft):
            if t + k < len(seq) and seq[t + k] == d: a += 1
            else: break
        accs.append(a)
        t += max(1, a + 1)                                # one verify step commits a accepted + 1 bonus/correction
    accs = np.array(accs)
    steps = len(accs); toks = int(accs.sum()) + steps     # accepted + 1 bonus per step
    # speedup vs decode = tokens produced / verify-steps, capped by verify cost in decode-token units
    print(f"\n[{name}] generated {len(gen)} tok over {steps} verify-steps", flush=True)
    print(f"  avg accepted/step = {accs.mean():.2f}   (>=8 needed to beat int8 verify, >=16 for f16)", flush=True)
    print(f"  steps with accept>=8: {100*np.mean(accs>=8):.0f}%   accept>=16: {100*np.mean(accs>=16):.0f}%", flush=True)
    print(f"  tokens/verify-step = {toks/steps:.2f}  -> spec is {'a WIN' if toks/steps>8 else 'a LOSS'} vs ~8-tok int8 verify", flush=True)

analyse("echo", "Repeat the following text exactly, word for word:\n" + ("The quick brown fox jumps over the lazy dog near the riverbank. " * 8))
analyse("code", "Add a one-line docstring to each function, keep everything else identical:\n\ndef add(a,b):\n    return a+b\n\ndef mul(a,b):\n    return a*b\n\ndef sub(a,b):\n    return a-b\n")
analyse("chat", "Tell me about the history of the Roman Empire in a few sentences.")
