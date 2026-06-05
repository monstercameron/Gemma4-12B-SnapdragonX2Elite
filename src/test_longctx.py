"""Long-context smoke + needle test for the 64k engine. Builds a ~1.4k-token prompt (crosses the
256-key flash-tile boundary AND the 1024 sliding-window ring wrap), hides a fact early, asks at the
end. Validates: no crash/NaN, coherent decode, and ideally retrieval via the global layers."""
import time, numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vk_engine as E
NEEDLE = "The secret passcode for the vault is CRIMSON-7492."
filler = ("The history of cartography spans many centuries, with mapmakers refining techniques "
          "for representing the curved earth on flat surfaces using various projections. ")
msg = NEEDLE + " " + filler * 60 + " Question: What is the secret passcode for the vault? Answer:"
ids = E.tok.apply_chat_template([{"role":"user","content":msg}],
                                add_generation_prompt=True, tokenize=True, return_dict=False)
ids = [int(i) for i in np.array(ids).ravel()]
print(f"prompt tokens = {len(ids)} (crosses 256-tile x{len(ids)//256} and 1024-ring={'YES' if len(ids)>1024 else 'no'})", flush=True)
t0 = time.time()
out = list(E.generate(ids, max_new=24, temperature=0.0, stop_ids=set()))
dt = time.time() - t0
txt = E.tok.decode(out)
finite = all(np.isfinite(E.forward(ids[-1], len(ids)-1)))  # sanity: logits finite
print("DECODE:", repr(txt), flush=True)
print(f"retrieved needle (CRIMSON-7492): {'YES' if 'CRIMSON' in txt or '7492' in txt else 'no'}", flush=True)
print(f"prefill+gen of {len(ids)}+24 tok in {dt:.1f}s; logits finite: {finite}", flush=True)
