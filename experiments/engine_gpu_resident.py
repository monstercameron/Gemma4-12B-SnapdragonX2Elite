"""Gemma 4 12B GPU-resident inference: hidden state lives on the GPU across all 48 layers.
Per layer the only CPU<->GPU round-trip is attention (read q/k/v, write attn_out) -> ~1 sync/
layer instead of ~7, so decode is bandwidth-bound, not sync-bound. Norms/MLP/residual run as
f32 GPU shaders; int4 GEMV shaders do the matmuls; attention reuses HF's exact q/k/v-norm + RoPE.

Run: .venv-gemma4/Scripts/python.exe scripts/engine_gpu_resident.py [ngen]
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from gpu_int4 import WgpuEngine, Int4Linear
from gpu_ops import Ops
import wgpu

NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BLK = 32
MODEL = "models/gemma-4-12B-it"
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
U = wgpu.BufferUsage
ENG = WgpuEngine(); OPS = Ops(ENG); DEV = ENG.dev


def f32buf(n, extra=0):
    return DEV.create_buffer(size=n * 4, usage=U.STORAGE | U.COPY_DST | U.COPY_SRC | extra)

def wbuf(arr):
    return DEV.create_buffer_with_data(data=np.ascontiguousarray(arr, np.float32).tobytes(), usage=U.STORAGE)

def uD(D, second=0):
    return DEV.create_buffer_with_data(data=np.array([D, second, 0, 0], np.uint32).tobytes(), usage=U.UNIFORM)


def main():
    import transformers
    from transformers import AutoTokenizer
    from transformers.models.gemma4_unified.modeling_gemma4_unified import apply_rotary_pos_emb, repeat_kv
    print("[load] gemma 4 12B...", flush=True); t0 = time.time()
    model = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager").eval()
    tm = model.model.language_model; cfg = tm.config
    H = cfg.hidden_size; eps = cfg.rms_norm_eps
    embed_scale = float(getattr(tm.embed_tokens, "scalar_embed_scale", H ** 0.5))
    EMB = tm.embed_tokens.weight.detach().float().cpu().numpy()
    softcap = cfg.final_logit_softcapping
    print(f"[load] {time.time()-t0:.0f}s; quantizing linears...", flush=True); t0 = time.time()

    def i4(lin):
        W = lin.weight.detach().float().cpu().numpy(); n = Int4Linear(ENG, W, W.shape[1], W.shape[0], blk=BLK)
        lin.weight = None; return n

    # shared GPU activation buffers (max sized)
    b = {k: f32buf(sz) for k, sz in {
        "h": H, "normed": H, "normed2": H, "o": H, "t": H, "t2": H, "down": H, "normed_f": H,
        "qkv": 16 * 512 + 8 * 256 + 8 * 256, "attn": 16 * 512,
        "gate": cfg.intermediate_size, "up": cfg.intermediate_size, "act": cfg.intermediate_size,
        "logits": cfg.vocab_size}.items()}

    layers = []
    for li, L in enumerate(tm.layers):
        sa = L.self_attn; lt = cfg.layer_types[li]
        is_sliding = lt == "sliding_attention"
        hd = cfg.head_dim if is_sliding else (cfg.global_head_dim or cfg.head_dim)
        k_eq_v = cfg.attention_k_eq_v and not is_sliding
        nkv = (cfg.num_global_key_value_heads if k_eq_v else cfg.num_key_value_heads)
        d = dict(idx=li, lt=lt, hd=hd, nkv=nkv, k_eq_v=k_eq_v,
                 nq=cfg.num_attention_heads * hd, nk=nkv * hd,
                 qp=i4(sa.q_proj), kp=i4(sa.k_proj), vp=(None if k_eq_v else i4(sa.v_proj)), op=i4(sa.o_proj),
                 gp=i4(L.mlp.gate_proj), up=i4(L.mlp.up_proj), dp=i4(L.mlp.down_proj),
                 q_norm=sa.q_norm, k_norm=sa.k_norm, v_norm=sa.v_norm,
                 in_w=wbuf(L.input_layernorm.weight.detach().float().numpy()),
                 pa_w=wbuf(L.post_attention_layernorm.weight.detach().float().numpy()),
                 pf_w=wbuf(L.pre_feedforward_layernorm.weight.detach().float().numpy()),
                 of_w=wbuf(L.post_feedforward_layernorm.weight.detach().float().numpy()),
                 scl=float(L.layer_scalar.item()) if hasattr(L, "layer_scalar") else 1.0,
                 kc=[], vc=[])
        layers.append(d)
        if (li + 1) % 12 == 0: print(f"  quantized layer {li+1}/48 ({time.time()-t0:.0f}s)", flush=True)
    lmh = i4(model.lm_head); fin_w = wbuf(tm.norm.weight.detach().float().numpy())
    print(f"[load] done in {time.time()-t0:.0f}s", flush=True)

    # ---- precompute bind groups (per layer) ----
    Du = {D: uD(D) for D in {H, cfg.intermediate_size, cfg.vocab_size}}
    nrm = lambda x, w, y: OPS.bg(OPS.p_norm, [x, w, y, Du[H]])
    for L in layers:
        L["bg_inorm"] = nrm(b["h"], L["in_w"], b["normed"])
        L["bg_q"] = L["qp"].make_chain_bg(b["normed"], b["qkv"], out_off=0)
        L["bg_k"] = L["kp"].make_chain_bg(b["normed"], b["qkv"], out_off=L["nq"] * 4)
        if L["vp"]: L["bg_v"] = L["vp"].make_chain_bg(b["normed"], b["qkv"], out_off=(L["nq"] + L["nk"]) * 4)
        L["bg_o"] = L["op"].make_chain_bg(b["attn"], b["o"])
        L["bg_panorm"] = OPS.bg(OPS.p_norm, [b["o"], L["pa_w"], b["t"], Du[H]])
        L["bg_add1"] = OPS.bg(OPS.p_add, [b["h"], b["t"], Du[H]])
        L["bg_pfnorm"] = OPS.bg(OPS.p_norm, [b["h"], L["pf_w"], b["normed2"], Du[H]])
        L["bg_g"] = L["gp"].make_chain_bg(b["normed2"], b["gate"])
        L["bg_u"] = L["up"].make_chain_bg(b["normed2"], b["up"])
        L["bg_gm"] = OPS.bg(OPS.p_gm, [b["gate"], b["up"], b["act"], Du[cfg.intermediate_size]])
        L["bg_d"] = L["dp"].make_chain_bg(b["act"], b["down"])
        L["bg_ofnorm"] = OPS.bg(OPS.p_norm, [b["down"], L["of_w"], b["t2"], Du[H]])
        L["bg_add2"] = OPS.bg(OPS.p_add, [b["h"], b["t2"], Du[H]])
        sb = uD(H, np.array([L["scl"]], np.float32).view(np.uint32)[0]); L["bg_scale"] = OPS.bg(OPS.p_scale, [b["h"], sb])
    bg_fnorm = OPS.bg(OPS.p_norm, [b["h"], fin_w, b["normed_f"], Du[H]])
    bg_lm = lmh.make_chain_bg(b["normed_f"], b["logits"])
    capb = uD(cfg.vocab_size, np.array([softcap], np.float32).view(np.uint32)[0]) if softcap else None
    bg_cap = OPS.bg(OPS.p_softcap, [b["logits"], capb]) if softcap else None

    def gemv(enc, lin, bg): lin.chain(enc, bg[0], bg[1])

    rope_cache = {}
    def get_rope(pos, lt, x):
        key = (pos, lt)
        if key not in rope_cache:
            rope_cache[key] = tm.rotary_emb(x, torch.tensor([[pos]]), layer_type=lt)
        return rope_cache[key]

    def forward(tid, pos):
        DEV.queue.write_buffer(b["h"], 0, (EMB[tid] * embed_scale).astype(np.float32).tobytes())
        for L in layers:
            # submit A: norm + q/k/v
            enc = DEV.create_command_encoder()
            OPS.norm(enc, L["bg_inorm"]); gemv(enc, L["qp"], L["bg_q"]); gemv(enc, L["kp"], L["bg_k"])
            if L["vp"]: gemv(enc, L["vp"], L["bg_v"])
            DEV.queue.submit([enc.finish()])
            hd, nkv = L["hd"], L["nkv"]
            tot = L["nq"] + L["nk"] + (0 if L["k_eq_v"] else L["nk"])
            qkv = np.frombuffer(DEV.queue.read_buffer(b["qkv"], 0, tot * 4), np.float32)   # ONE sync
            qn = qkv[:L["nq"]]; kn = qkv[L["nq"]:L["nq"] + L["nk"]]
            vn = kn if L["k_eq_v"] else qkv[L["nq"] + L["nk"]:]
            # CPU attention (HF q/k/v-norm + rope) -- f32
            q = torch.from_numpy(qn.copy()).view(1, 1, -1, hd)
            k = torch.from_numpy(kn.copy()).view(1, 1, nkv, hd)
            v = torch.from_numpy(vn.copy()).view(1, 1, nkv, hd)
            q = L["q_norm"](q); k = L["k_norm"](k); v = L["v_norm"](v)
            cos, sin = get_rope(pos, L["lt"], q)
            q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)   # [1,Hh,1,hd]
            k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)   # [1,nkv,1,hd]
            v = v.transpose(1, 2)
            L["kc"].append(k); L["vc"].append(v)
            K = torch.cat(L["kc"], dim=2); V = torch.cat(L["vc"], dim=2)              # [1,nkv,T,hd]
            grp = q.shape[1] // nkv
            Kr = repeat_kv(K, grp); Vr = repeat_kv(V, grp)
            scores = torch.matmul(q.float(), Kr.float().transpose(2, 3))              # scaling=1.0
            scores = scores - scores.amax(-1, keepdim=True)
            w = torch.softmax(scores, dim=-1)
            ao = torch.matmul(w, Vr.float()).transpose(1, 2).reshape(-1).numpy()       # [nq]
            DEV.queue.write_buffer(b["attn"], 0, np.ascontiguousarray(ao, np.float32).tobytes())
            # submit B: o + residual + mlp + residual + scale  (h stays resident)
            enc = DEV.create_command_encoder()
            gemv(enc, L["op"], L["bg_o"]); OPS.norm(enc, L["bg_panorm"]); OPS.ew(enc, OPS.p_add, L["bg_add1"], H)
            OPS.norm(enc, L["bg_pfnorm"]); gemv(enc, L["gp"], L["bg_g"]); gemv(enc, L["up"], L["bg_u"])
            OPS.ew(enc, OPS.p_gm, L["bg_gm"], cfg.intermediate_size); gemv(enc, L["dp"], L["bg_d"])
            OPS.norm(enc, L["bg_ofnorm"]); OPS.ew(enc, OPS.p_add, L["bg_add2"], H)
            OPS.ew(enc, OPS.p_scale, L["bg_scale"], H)
            DEV.queue.submit([enc.finish()])
        enc = DEV.create_command_encoder()
        OPS.norm(enc, bg_fnorm); lmh.chain(enc, bg_lm[0], bg_lm[1])
        if bg_cap: OPS.ew(enc, OPS.p_softcap, bg_cap, cfg.vocab_size)
        DEV.queue.submit([enc.finish()])
        return np.frombuffer(DEV.queue.read_buffer(b["logits"], 0, cfg.vocab_size * 4), np.float32)

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? One word."}],
                                  add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = list(np.array(ids).ravel())
    print(f"[gen] prompt {len(ids)} tok; prefill (resident)...", flush=True)
    with torch.no_grad():
        t0 = time.time(); logits = None
        for pos, tid in enumerate(ids):
            logits = forward(tid, pos)
        print(f"[gen] prefill {time.time()-t0:.1f}s ({len(ids)/(time.time()-t0):.2f} tok/s); decoding {NGEN}...", flush=True)
        gen, td = [], time.time()
        for _ in range(NGEN):
            nxt = int(np.argmax(logits)); gen.append(nxt)
            logits = forward(nxt, len(ids) + len(gen) - 1)
        dt = time.time() - td
    print(f"\n[gen] CONTINUATION: {tok.decode(gen)!r}")
    print(f"[gen] decode {NGEN} tok in {dt:.2f}s = {NGEN/dt:.3f} tok/s", flush=True)


if __name__ == "__main__":
    main()
