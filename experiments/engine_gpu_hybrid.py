"""Gemma 4 12B GPU inference: every nn.Linear runs as a custom int4 GEMV compute shader (wgpu),
while HF handles the exact attention/RoPE/RMSNorm. Decode is seq=1 -> GPU GEMV (bandwidth path);
prefill (seq>1) falls back to torch. Measures real decode tok/s. Target: beat 6.

Run: .venv-gemma4/Scripts/python.exe scripts/engine_gpu_hybrid.py [ngen]
"""
import os, sys, time, numpy as np, torch
from torch import nn
sys.path.insert(0, os.path.dirname(__file__))
from gpu_int4 import WgpuEngine, Int4Linear

NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 12
BLK = 32
MODEL = "models/gemma-4-12B-it"
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

ENG = WgpuEngine()


class GpuLinear(nn.Module):
    """Replaces an nn.Linear with a GPU int4 GEMV. Frees the fp16 weight (saves 23GB on the
    shared 48GB RAM). Decode (1 row) = 1 GEMV; prefill (S rows) loops rows through the GPU."""
    def __init__(self, lin):
        super().__init__()
        W = lin.weight.detach().float().cpu().numpy()        # [N,K]
        self.N, self.K = W.shape
        self.out_features, self.in_features = self.N, self.K
        self.gpu = Int4Linear(ENG, W, self.K, self.N, blk=BLK)
        lin.weight = None                                    # free the fp16 weight

    def forward(self, x):
        rows = x.reshape(-1, self.K)
        out = np.empty((rows.shape[0], self.N), np.float32)
        rn = rows.float().cpu().numpy()
        for i in range(rows.shape[0]):
            out[i] = self.gpu.forward(rn[i])
        return torch.from_numpy(out).to(x.dtype).reshape(*x.shape[:-1], self.N)


def main():
    import transformers
    from transformers import AutoTokenizer
    print("[load] gemma 4 12B (fp16)...", flush=True)
    t0 = time.time()
    model = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager").eval()
    print(f"[load] {time.time()-t0:.0f}s; quantizing linears -> int4 GPU buffers...", flush=True)

    # wrap every decoder Linear + lm_head with the GPU int4 GEMV
    TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
               "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    t0 = time.time(); n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, nn.Linear) and (any(t in name for t in TARGETS) or name.endswith("lm_head")):
            if "." in name:
                parent = model.get_submodule(name.rsplit(".", 1)[0]); attr = name.rsplit(".", 1)[1]
            else:
                parent = model; attr = name
            setattr(parent, attr, GpuLinear(mod))
            n += 1
            if n % 50 == 0:
                print(f"  quantized {n} linears ({time.time()-t0:.0f}s)", flush=True)
    print(f"[load] wrapped {n} linears in {time.time()-t0:.0f}s", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? One word."}],
                                  add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = torch.tensor(np.array(ids).reshape(1, -1), dtype=torch.long)
    print(f"[gen] prompt {ids.shape[1]} tokens; prefill (torch)...", flush=True)

    with torch.no_grad():
        t0 = time.time()
        out = model(ids, use_cache=True)
        pkv = out.past_key_values
        logits = out.logits[:, -1]
        print(f"[gen] prefill {time.time()-t0:.1f}s; decoding {NGEN} tok on GPU...", flush=True)
        gen, tdec = [], time.time()
        for step in range(NGEN):
            nxt = int(logits.argmax())
            gen.append(nxt)
            o = model(torch.tensor([[nxt]]), past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values; logits = o.logits[:, -1]
        dt = time.time() - tdec
    print(f"\n[gen] CONTINUATION: {tok.decode(gen)!r}")
    print(f"[gen] decode {NGEN} tok in {dt:.1f}s = {NGEN/dt:.3f} tok/s", flush=True)


if __name__ == "__main__":
    main()
