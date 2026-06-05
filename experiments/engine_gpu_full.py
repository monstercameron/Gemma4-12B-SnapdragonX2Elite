"""Gemma 4 12B FULLY on-GPU: every op (int4 GEMV, RMSNorm, RoPE, attention, MLP, residual)
runs as a compute shader; the whole token = ONE command encoder, ONE submit, ONE logits read.
Decode is bandwidth-bound (no per-layer sync). Target: max tok/s.

Run: .venv-gemma4/Scripts/python.exe scripts/engine_gpu_full.py [ngen]
"""
import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from gpu_int4 import WgpuEngine, Int4Linear
from gpu_ops import Ops
from gpu_attn import Attn, MAXT
import wgpu

NGEN = int(sys.argv[1]) if len(sys.argv) > 1 else 8
BLK = 32; MODEL = "models/gemma-4-12B-it"
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
U = wgpu.BufferUsage
ENG = WgpuEngine(); OPS = Ops(ENG); ATT = Attn(ENG); DEV = ENG.dev


def fbuf(n, src=False):
    return DEV.create_buffer(size=n * 4, usage=U.STORAGE | U.COPY_DST | (U.COPY_SRC if src else 0))
def wbuf(a):
    return DEV.create_buffer_with_data(data=np.ascontiguousarray(a, np.float32).tobytes(), usage=U.STORAGE)
def ubuf(vals):
    return DEV.create_buffer_with_data(data=np.array(vals, np.uint32).tobytes(), usage=U.UNIFORM | U.COPY_DST)


def main():
    import transformers
    from transformers import AutoTokenizer
    print("[load] gemma 4 12B...", flush=True); t0 = time.time()
    model = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
        MODEL, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager").eval()
    tm = model.model.language_model; cfg = tm.config
    H = cfg.hidden_size; nH = cfg.num_attention_heads; I = cfg.intermediate_size
    embed_scale = float(getattr(tm.embed_tokens, "scalar_embed_scale", H ** 0.5))
    EMB = tm.embed_tokens.weight.detach().float().cpu().numpy()
    softcap = cfg.final_logit_softcapping
    print(f"[load] {time.time()-t0:.0f}s; quantizing...", flush=True); t0 = time.time()

    def i4(lin):
        W = lin.weight.detach().float().cpu().numpy(); n = Int4Linear(ENG, W, W.shape[1], W.shape[0], blk=BLK)
        lin.weight = None; return n

    b = {k: fbuf(sz) for k, sz in {"h": H, "normed": H, "normed2": H, "o": H, "t": H, "t2": H,
         "down": H, "normed_f": H, "qkv": nH * 512 + 2 * 8 * 256, "attn": nH * 512,
         "gate": I, "up": I, "act": I}.items()}
    b["logits"] = fbuf(cfg.vocab_size, src=True)
    # per-type rope buffers + shared dynamic uniform (slot, T)
    cos = {"sliding_attention": fbuf(256), "full_attention": fbuf(512)}
    sin = {"sliding_attention": fbuf(256), "full_attention": fbuf(512)}
    DYN = ubuf([0, 0, 0, 0])

    Du = {D: ubuf([D, 0, 0, 0]) for D in {H, I, cfg.vocab_size}}
    layers = []
    for li, L in enumerate(tm.layers):
        sa = L.self_attn; lt = cfg.layer_types[li]; sliding = lt == "sliding_attention"
        hd = cfg.head_dim if sliding else (cfg.global_head_dim or cfg.head_dim)
        k_eq_v = cfg.attention_k_eq_v and not sliding
        nkv = cfg.num_global_key_value_heads if k_eq_v else cfg.num_key_value_heads
        nq, nk = nH * hd, nkv * hd
        d = dict(idx=li, lt=lt, hd=hd, nkv=nkv, nq=nq, nk=nk, grp=nH // nkv, k_eq_v=k_eq_v,
                 qp=i4(sa.q_proj), kp=i4(sa.k_proj), vp=None if k_eq_v else i4(sa.v_proj), op=i4(sa.o_proj),
                 gp=i4(L.mlp.gate_proj), up=i4(L.mlp.up_proj), dp=i4(L.mlp.down_proj),
                 qnw=wbuf(sa.q_norm.weight.detach().float().numpy()),
                 knw=wbuf(sa.k_norm.weight.detach().float().numpy()),
                 in_w=wbuf(L.input_layernorm.weight.detach().float().numpy()),
                 pa_w=wbuf(L.post_attention_layernorm.weight.detach().float().numpy()),
                 pf_w=wbuf(L.pre_feedforward_layernorm.weight.detach().float().numpy()),
                 of_w=wbuf(L.post_feedforward_layernorm.weight.detach().float().numpy()),
                 scl=float(L.layer_scalar.item()) if hasattr(L, "layer_scalar") else 1.0,
                 kc=fbuf(nkv * MAXT * hd), vc=fbuf(nkv * MAXT * hd))
        layers.append(d)
        if (li + 1) % 16 == 0: print(f"  layer {li+1}/48 ({time.time()-t0:.0f}s)", flush=True)
    lmh = i4(model.lm_head); fin_w = wbuf(tm.norm.weight.detach().float().numpy())
    print(f"[load] done {time.time()-t0:.0f}s", flush=True)

    # ---- precompute bind groups ----
    nbg = lambda x, w, y: OPS.bg(OPS.p_norm, [x, w, y, Du[H]])
    for L in layers:
        ct, st = cos[L["lt"]], sin[L["lt"]]
        L["bg_in"] = nbg(b["h"], L["in_w"], b["normed"])
        L["bg_q"] = L["qp"].make_chain_bg(b["normed"], b["qkv"], out_off=0)
        L["bg_k"] = L["kp"].make_chain_bg(b["normed"], b["qkv"], out_off=L["nq"] * 4)
        if L["vp"]: L["bg_v"] = L["vp"].make_chain_bg(b["normed"], b["qkv"], out_off=(L["nq"] + L["nk"]) * 4)
        L["P"] = ubuf([L["nkv"], L["hd"], 0, (L["nq"] + L["nk"]), 1 if L["k_eq_v"] else 0, MAXT, 0, 0])
        L["A"] = ubuf([nH, L["nkv"], L["hd"], MAXT, 0, L["grp"], 0, 0])
        # offsets in P/A are in FLOATS: koff=nq, voff=nq+nk, qoff=0
        L["P"] = ubuf([L["nkv"], L["hd"], L["nq"], L["nq"] + L["nk"], 1 if L["k_eq_v"] else 0, MAXT, 0, 0])
        L["bg_prep"] = OPS.bg(ATT.p_prep, [b["qkv"], L["knw"], ct, st, L["kc"], L["vc"], L["P"], DYN])
        L["bg_attn"] = OPS.bg(ATT.p_attn, [b["qkv"], L["qnw"], ct, st, L["kc"], L["vc"], b["attn"], L["A"], DYN])
        L["bg_o"] = L["op"].make_chain_bg(b["attn"], b["o"])
        L["bg_na"] = OPS.bg(OPS.p_na, [b["o"], L["pa_w"], b["h"], Du[H]])
        L["bg_pfn"] = nbg(b["h"], L["pf_w"], b["normed2"])
        L["bg_g"] = L["gp"].make_chain_bg(b["normed2"], b["gate"]); L["bg_u"] = L["up"].make_chain_bg(b["normed2"], b["up"])
        L["bg_gm"] = OPS.bg(OPS.p_gm, [b["gate"], b["up"], b["act"], Du[I]])
        L["bg_d"] = L["dp"].make_chain_bg(b["act"], b["down"])
        nasu = ubuf([H, int(np.array([L["scl"]], np.float32).view(np.uint32)[0]), 0, 0])
        L["bg_nas"] = OPS.bg(OPS.p_na, [b["down"], L["of_w"], b["h"], nasu])
    bg_fn = nbg(b["h"], fin_w, b["normed_f"]); bg_lm = lmh.make_chain_bg(b["normed_f"], b["logits"])
    capb = ubuf([cfg.vocab_size, np.array([softcap], np.float32).view(np.uint32)[0], 0, 0]) if softcap else None
    bg_cap = OPS.bg(OPS.p_softcap, [b["logits"], capb]) if softcap else None

    def disp(c, pipe, bg, nwg):
        cp = c.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0, bg); cp.dispatch_workgroups(nwg); cp.end()

    rope_cache = {}
    def set_rope(pos):
        for lt in ("sliding_attention", "full_attention"):
            if (pos, lt) not in rope_cache:
                cs, sn = tm.rotary_emb(torch.zeros(1, 1, cfg.head_dim if lt == "sliding_attention" else cfg.global_head_dim),
                                       torch.tensor([[pos]]), layer_type=lt)
                rope_cache[(pos, lt)] = (cs.reshape(-1).float().numpy(), sn.reshape(-1).float().numpy())
            cs, sn = rope_cache[(pos, lt)]
            DEV.queue.write_buffer(cos[lt], 0, np.ascontiguousarray(cs, np.float32).tobytes())
            DEV.queue.write_buffer(sin[lt], 0, np.ascontiguousarray(sn, np.float32).tobytes())

    TIMES = {k: 0.0 for k in ("write", "rope", "encode", "submit", "read")}
    NT = [0]
    def forward(tid, pos):
        prof = NT[0] > 0  # skip first (warmup) for timing
        t = time.perf_counter()
        DEV.queue.write_buffer(b["h"], 0, (EMB[tid] * embed_scale).astype(np.float32).tobytes())
        DEV.queue.write_buffer(DYN, 0, np.array([pos, pos + 1, 0, 0], np.uint32).tobytes())
        if prof: TIMES["write"] += time.perf_counter() - t; t = time.perf_counter()
        set_rope(pos)
        if prof: TIMES["rope"] += time.perf_counter() - t; t = time.perf_counter()
        enc = DEV.create_command_encoder()
        for L in layers:
            OPS.norm(enc, L["bg_in"])
            L["qp"].chain(enc, *L["bg_q"]); L["kp"].chain(enc, *L["bg_k"])
            if L["vp"]: L["vp"].chain(enc, *L["bg_v"])
            disp(enc, ATT.p_prep, L["bg_prep"], L["nkv"])
            disp(enc, ATT.p_attn, L["bg_attn"], nH)
            L["op"].chain(enc, *L["bg_o"]); OPS.norm(enc, L["bg_na"], OPS.p_na)     # fused norm+add
            OPS.norm(enc, L["bg_pfn"]); L["gp"].chain(enc, *L["bg_g"]); L["up"].chain(enc, *L["bg_u"])
            OPS.ew(enc, OPS.p_gm, L["bg_gm"], I); L["dp"].chain(enc, *L["bg_d"])
            OPS.norm(enc, L["bg_nas"], OPS.p_na)                                    # fused norm+add+scale
        OPS.norm(enc, bg_fn); lmh.chain(enc, *bg_lm)
        if bg_cap: OPS.ew(enc, OPS.p_softcap, bg_cap, cfg.vocab_size)
        if prof: TIMES["encode"] += time.perf_counter() - t; t = time.perf_counter()
        cb = enc.finish()
        if prof: TIMES["submit"] += time.perf_counter() - t; t = time.perf_counter()
        DEV.queue.submit([cb])
        r = np.frombuffer(DEV.queue.read_buffer(b["logits"], 0, cfg.vocab_size * 4), np.float32)
        if prof: TIMES["read"] += time.perf_counter() - t
        NT[0] += 1
        return r

    tok = AutoTokenizer.from_pretrained(MODEL)
    ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? One word."}],
                                  add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = list(np.array(ids).ravel())
    print(f"[gen] prompt {len(ids)} tok; prefill...", flush=True)
    with torch.no_grad():
        t0 = time.time(); logits = None
        for pos, tid in enumerate(ids):
            logits = forward(tid, pos)
        print(f"[gen] prefill {time.time()-t0:.1f}s; decoding {NGEN}...", flush=True)
        gen, td = [], time.time()
        for _ in range(NGEN):
            nxt = int(np.argmax(logits)); gen.append(nxt)
            logits = forward(nxt, len(ids) + len(gen) - 1)
        dt = time.time() - td
    print(f"\n[gen] CONTINUATION: {tok.decode(gen)!r}")
    print(f"[gen] decode {NGEN} tok in {dt:.2f}s = {NGEN/dt:.3f} tok/s", flush=True)

    # ---- per-token phase breakdown (TIMES accumulated over timed forwards) ----
    nt = NT[0] - 1  # forwards counted (minus the warmup one skipped)
    print(f"\n[profile] per-token phase breakdown (avg over {nt} timed forwards):", flush=True)
    for k in ("write", "rope", "encode", "submit", "read"):
        print(f"    {k:8} {TIMES[k]/nt*1e3:7.1f} ms   ({TIMES[k]/sum(TIMES.values())*100:4.1f}%)", flush=True)
    print(f"    {'TOTAL':8} {sum(TIMES.values())/nt*1e3:7.1f} ms/token", flush=True)
    print(f"    NOTE: 'read' blocks on GPU execution, so it = GPU compute + readback;\n"
          f"          'encode' = Python building the ~1200 compute passes.", flush=True)

    # ---- FULL call-stack profile of a SINGLE token forward ----
    import cProfile, pstats, io
    forward(gen[-1], len(ids) + len(gen))   # warm
    pr = cProfile.Profile(); pr.enable()
    forward(gen[-1], len(ids) + len(gen) + 1)   # the ONE token we profile
    pr.disable()
    for order, label in (("cumulative", "by CUMULATIVE time (call-stack rollup)"),
                         ("tottime", "by TOTAL self time")):
        s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats(order).print_stats()
        print(f"\n========== single-token cProfile {label} ==========", flush=True)
        for line in s.getvalue().splitlines():
            ln = line.rstrip()
            if ln.strip() and ("ncalls" in ln or "function calls" in ln or "{" in ln or ".py:" in ln or "/" in ln):
                print(ln, flush=True)
    # per-pass-type counts in one token
    print("\n========== compute-pass inventory (1 token) ==========", flush=True)
    ng = sum(1 for L in layers for _ in (L["qp"], L["kp"], L["op"], L["gp"], L["up"], L["dp"])) + sum(1 for L in layers if L["vp"]) + 1
    print(f"  GEMV (int4 matmul): {ng} GEMVs x2 passes = {ng*2}", flush=True)
    print(f"  RMSNorm passes:     {len(layers)*2 + 1} (in/pre-ff per layer + final)", flush=True)
    print(f"  fused norm+add:     {len(layers)*2} (post-attn, post-ff)", flush=True)
    print(f"  attention (prep+attn): {len(layers)*2}", flush=True)
    print(f"  gelu*up + softcap:  {len(layers) + 1}", flush=True)


if __name__ == "__main__":
    main()
