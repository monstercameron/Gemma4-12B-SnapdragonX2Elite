"""Gemma 4 12B-it — transformers reference run on CPU (ARM64).

Goals:
  1. Prove the model loads & generates in the dedicated export venv (transformers >=5.10).
  2. Capture a GOLDEN continuation for "The capital of France is" (greedy) so the
     later ONNX/NPU path can be validated against it.
  3. Dump the text backbone's tied embed_tokens / lm_head table to .npy for the
     NPU shard engine (embeddings + final projection run on CPU there).

Run:
  .venv-gemma4/Scripts/python.exe scripts/ref_run.py
"""
import os, time, json
import numpy as np

MODEL = "models/gemma-4-12B-it"
PROMPT = "The capital of France is"
NGEN = 8
OUT = "out"

os.makedirs(OUT, exist_ok=True)
NT = max(1, (os.cpu_count() or 2) - 1)
os.environ.setdefault("OMP_NUM_THREADS", str(NT))

import torch
torch.set_num_threads(NT)
from transformers import AutoConfig, AutoTokenizer

print(f"[ref] torch {torch.__version__}  threads={NT}", flush=True)
import transformers
print(f"[ref] transformers {transformers.__version__}", flush=True)

cfg = AutoConfig.from_pretrained(MODEL)
print(f"[ref] model_type={cfg.model_type} arch={cfg.architectures}", flush=True)
tcfg = cfg.get_text_config()
print(f"[ref] text: layers={tcfg.num_hidden_layers} hidden={tcfg.hidden_size} "
      f"vocab={tcfg.vocab_size} heads={tcfg.num_attention_heads}/{tcfg.num_key_value_heads} "
      f"head_dim={tcfg.head_dim} global_head_dim={getattr(tcfg,'global_head_dim',None)} "
      f"ple={getattr(tcfg,'hidden_size_per_layer_input',0)}", flush=True)

# Pick the right model class. Prefer the explicit unified class; fall back to AutoModel.
ModelClass = None
for name in ("Gemma4UnifiedForConditionalGeneration", "AutoModelForMultimodalLM",
             "AutoModelForImageTextToText", "AutoModelForCausalLM"):
    try:
        ModelClass = getattr(transformers, name)
        print(f"[ref] using {name}", flush=True)
        break
    except AttributeError:
        continue
assert ModelClass is not None, "no suitable model class found in transformers"

tok = AutoTokenizer.from_pretrained(MODEL)
t0 = time.time()
model = ModelClass.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True)
model.eval()
print(f"[ref] loaded in {time.time()-t0:.0f}s", flush=True)

# Locate the text language model + its tied embedding table for the .npy dump.
def find_text_model(m):
    for path in ("model.language_model", "language_model.model", "language_model",
                 "model.text_model", "model"):
        obj = m
        ok = True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False; break
        if ok and hasattr(obj, "embed_tokens"):
            print(f"[ref] text model at: {path}", flush=True)
            return obj
    return None

ids = tok(PROMPT, return_tensors="pt")["input_ids"]
print(f"[ref] prompt={PROMPT!r} -> {ids.shape[1]} tokens: {ids[0].tolist()}", flush=True)

t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=NGEN, do_sample=False)
dt = time.time() - t0
gen = out[0, ids.shape[1]:].tolist()
cont = tok.decode(gen)
print(f"\n[ref] GOLDEN continuation ({NGEN} tok, {NGEN/dt:.3f} tok/s):", flush=True)
print(f"  tokens: {gen}", flush=True)
print(f"  text  : {cont!r}", flush=True)

# Per-token golden first-step logits argmax (what the NPU must reproduce on step 0).
with torch.no_grad():
    step0 = model(ids).logits[0, -1].float()
print(f"[ref] step-0 argmax token = {int(step0.argmax())} -> {tok.decode([int(step0.argmax())])!r}", flush=True)

tm = find_text_model(model)
if tm is not None:
    emb = tm.embed_tokens.weight.detach().float().cpu().numpy().astype(np.float16)
    np.save(f"{OUT}/embed_tokens.npy", emb)
    print(f"[ref] dumped embed_tokens.npy {emb.shape} (tied lm_head={tcfg.tie_word_embeddings})", flush=True)
    # scaled-embedding scale factor matters for Gemma — record it
    scale = getattr(tm.embed_tokens, "scalar_embed_scale", None)
    meta = {"golden_tokens": gen, "golden_text": cont, "step0_argmax": int(step0.argmax()),
            "embed_scale": float(scale) if scale is not None else (tcfg.hidden_size ** 0.5),
            "tie_word_embeddings": bool(tcfg.tie_word_embeddings)}
    json.dump(meta, open(f"{OUT}/golden.json", "w"), indent=2)
    print(f"[ref] wrote out/golden.json", flush=True)
else:
    print("[ref] WARN: could not locate text embed_tokens for .npy dump", flush=True)
