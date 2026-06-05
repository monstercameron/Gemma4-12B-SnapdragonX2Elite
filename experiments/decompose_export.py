"""Export the layers of the two bad shards (3: 12-15, 5: 20-23) as INDIVIDUAL 1-layer ONNX
shards, so we can test each layer on the HTP in isolation and find the exact offending op.

Run (gemma venv): .venv-gemma4/Scripts/python.exe scripts/decompose_export.py
"""
import os, sys, time, torch, transformers
sys.path.insert(0, os.path.dirname(__file__))
from export_shards import export_one_shard

OUT = "out/gemma4_decomp"
WIN = 64
TARGET = [12, 13, 14, 15, 20, 21, 22, 23]
os.makedirs(OUT, exist_ok=True)

t0 = time.time()
full = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
    "models/gemma-4-12B-it", dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation="eager").eval()
tm = full.model.language_model
tm.config._attn_implementation = "eager"
cfg = tm.config
rotary = tm.rotary_emb
layers = list(tm.layers)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

for g in TARGET:
    path = f"{OUT}/layer_{g}.onnx"
    if os.path.exists(path):
        print(f"layer {g}: exists, skip", flush=True); continue
    t = time.time()
    export_one_shard([layers[g]], [g], rotary, cfg, WIN, path)
    print(f"layer {g} ({cfg.layer_types[g]}) -> {path} "
          f"({os.path.getsize(path)/1e6:.0f} MB, {time.time()-t:.0f}s)", flush=True)
print("DONE", flush=True)
