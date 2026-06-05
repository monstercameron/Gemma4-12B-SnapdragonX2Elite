"""Export the Gemma 4 12B-it text backbone to fp16 ONNX *shards* for the Hexagon NPU.

Why a custom exporter (not onnxruntime-genai): the genai 0.14 builder does not support
`gemma4_unified` (see ORT-GenAI #2062). But the 12B text decoder is all primitive ops
(RMSNorm, MatMul, Softmax, GeLU, RoPE) with NO PLE / NO MoE / NO shared-KV — so we can
torch.onnx.export it directly, one K-layer shard at a time (sidesteps the 2 GB protobuf
limit and produces HTP-ready shards in one step).

Correctness strategy: each shard wrapper REUSES the real `Gemma4UnifiedTextDecoderLayer.forward`
(hence the real eager attention, scaling, q/k/v-norm, partial-rotary on full layers) and only
supplies a tiny duck-typed KV-cache shim that does explicit concat. We never re-derive the math.

Decode graph (static, seq=1), per shard:
  inputs : hidden_in[1,1,H] fp16, position_ids[1,1] i64,
           attn_mask[1,1,1,WIN+1] fp16 (additive),
           past_key_values.{g}.key/value  for each global layer index g in the shard
  outputs: hidden_out[1,1,H], present.{g}.key/value (length WIN+1; engine trims to WIN)

Run self-test (no weights needed):
  .venv-gemma4/Scripts/python.exe scripts/export_shards.py --selftest
Full export (after weights download):
  .venv-gemma4/Scripts/python.exe scripts/export_shards.py --model models/gemma-4-12B-it \
      --out out/gemma4_fp16 --layers-per-shard 4 --win 64
"""
import os, sys, argparse, json, time
import numpy as np
import torch
from torch import nn

torch.manual_seed(0)


# --------------------------- KV-cache shim ---------------------------
class ShardCache:
    """Duck-typed Cache: layer.forward calls .update(k, v, layer_idx). We concat the
    provided past (fixed WIN length) with the current step and record the present."""
    def __init__(self, past_k, past_v, idx_map):
        # past_k/past_v: dict global_idx -> tensor [1, n_kv, WIN, hd]
        self.past_k, self.past_v = past_k, past_v
        self.idx_map = idx_map            # local layer_idx -> global_idx
        self.present_k, self.present_v = {}, {}

    def update(self, k, v, layer_idx, *a, **kw):
        g = self.idx_map[layer_idx]
        kf = torch.cat([self.past_k[g], k], dim=2)
        vf = torch.cat([self.past_v[g], v], dim=2)
        self.present_k[g] = kf
        self.present_v[g] = vf
        return kf, vf


# --------------------------- shard wrapper ---------------------------
class ShardWrapper(nn.Module):
    def __init__(self, layers, layer_globals, rotary, config, win):
        super().__init__()
        self.layers = nn.ModuleList(layers)              # the real decoder layers
        self.layer_globals = layer_globals               # [g0, g1, ...]
        self.rotary = rotary                             # real rotary module (both types)
        self.config = config
        self.win = win
        self.layer_types = [config.layer_types[g] for g in layer_globals]

    def forward(self, hidden_in, position_ids, attn_mask, *kv):
        # kv is flattened [past_k_0, past_v_0, past_k_1, past_v_1, ...] aligned to layer order
        past_k = {g: kv[2 * i] for i, g in enumerate(self.layer_globals)}
        past_v = {g: kv[2 * i + 1] for i, g in enumerate(self.layer_globals)}
        idx_map = {li: g for li, g in enumerate(self.layer_globals)}
        cache = ShardCache(past_k, past_v, idx_map)

        # real rotary tables, one per attention type present in this shard
        pe = {}
        for t in set(self.layer_types):
            pe[t] = self.rotary(hidden_in, position_ids, layer_type=t)

        h = hidden_in
        for li, layer in enumerate(self.layers):
            t = self.layer_types[li]
            # local layer_idx must match what the layer's attention uses for cache.update
            layer.self_attn.layer_idx = li
            h = layer(
                hidden_states=h,
                shared_kv_states={},
                position_embeddings=pe[t],
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=cache,
            )
        outs = [h]
        for g in self.layer_globals:
            outs.append(cache.present_k[g])
            outs.append(cache.present_v[g])
        return tuple(outs)


def _kv_heads(config, layer_type):
    is_sliding = layer_type == "sliding_attention"
    if config.attention_k_eq_v and not is_sliding:
        nkv = config.num_global_key_value_heads
    else:
        nkv = config.num_key_value_heads
    hd = config.head_dim if is_sliding else (config.global_head_dim or config.head_dim)
    return nkv, hd


def export_one_shard(shard_layers, shard_globals, rotary, config, win, path):
    H = config.hidden_size
    wrap = ShardWrapper(shard_layers, shard_globals, rotary, config, win).eval()

    hidden_in = torch.randn(1, 1, H, dtype=torch.float16)
    position_ids = torch.tensor([[win]], dtype=torch.int64)
    attn_mask = torch.zeros(1, 1, 1, win + 1, dtype=torch.float16)

    kv_inputs, in_names = [], ["hidden_in", "position_ids", "attention_mask"]
    out_names = ["hidden_out"]
    for g in shard_globals:
        nkv, hd = _kv_heads(config, config.layer_types[g])
        kv_inputs.append(torch.zeros(1, nkv, win, hd, dtype=torch.float16))   # past key
        kv_inputs.append(torch.zeros(1, nkv, win, hd, dtype=torch.float16))   # past value
        in_names += [f"past_key_values.{g}.key", f"past_key_values.{g}.value"]
        out_names += [f"present.{g}.key", f"present.{g}.value"]

    args = (hidden_in, position_ids, attn_mask, *kv_inputs)
    # export into a fresh temp dir so torch's scattered external-data files (for >2GB shards)
    # can't collide between shards; then consolidate to one named .data file.
    import shutil, onnx
    tmp = path + ".tmpdir"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    os.makedirs(tmp)
    tmp_path = os.path.join(tmp, "m.onnx")
    with torch.no_grad():
        torch.onnx.export(
            wrap, args, tmp_path,
            input_names=in_names, output_names=out_names,
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
    m = onnx.load(tmp_path)   # loads however torch wrote it (inline or scattered external)
    if os.path.exists(path):
        os.remove(path)
    data_name = os.path.basename(path) + ".data"
    for f in (path, path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    onnx.save_model(m, path, save_as_external_data=True, all_tensors_to_one_file=True,
                    location=data_name, size_threshold=1024)
    shutil.rmtree(tmp)
    return in_names, out_names


# --------------------------- self-test ---------------------------
def selftest():
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextConfig, Gemma4UnifiedTextDecoderLayer, Gemma4UnifiedTextRotaryEmbedding,
    )
    # tiny config exercising BOTH a sliding and a full layer (variable head dims)
    cfg = Gemma4UnifiedTextConfig(
        vocab_size=320, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, global_head_dim=32,
        num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        max_position_embeddings=64, final_logit_softcapping=30.0,
    )
    cfg._attn_implementation = "eager"
    torch.set_default_dtype(torch.float32)
    layers = [Gemma4UnifiedTextDecoderLayer(cfg, i).eval() for i in range(cfg.num_hidden_layers)]
    rotary = Gemma4UnifiedTextRotaryEmbedding(cfg)
    for l in layers:
        l.half()
    rotary.half()
    win = 8
    os.makedirs("out/_selftest", exist_ok=True)

    # shard = first 2 layers (sliding + full) — covers variable head dims in one graph
    shard_globals = [0, 1]
    shard_layers = [layers[g] for g in shard_globals]

    # golden via direct wrapper call (eager) BEFORE export
    H = cfg.hidden_size
    hidden_in = torch.randn(1, 1, H, dtype=torch.float16)
    pos = torch.tensor([[win]], dtype=torch.int64)
    mask = torch.zeros(1, 1, 1, win + 1, dtype=torch.float16)
    kv = []
    for g in shard_globals:
        nkv, hd = _kv_heads(cfg, cfg.layer_types[g])
        kv += [torch.zeros(1, nkv, win, hd, dtype=torch.float16),
               torch.zeros(1, nkv, win, hd, dtype=torch.float16)]
    wrap = ShardWrapper(shard_layers, shard_globals, rotary, cfg, win).eval()
    with torch.no_grad():
        gold = wrap(hidden_in, pos, mask, *kv)

    path = "out/_selftest/shard0.onnx"
    in_names, out_names = export_one_shard(shard_layers, shard_globals, rotary, cfg, win, path)
    print(f"[selftest] exported {path}: ins={len(in_names)} outs={len(out_names)}")

    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    feeds = {"hidden_in": hidden_in.numpy(), "position_ids": pos.numpy(),
             "attention_mask": mask.numpy()}
    for i, g in enumerate(shard_globals):
        feeds[f"past_key_values.{g}.key"] = kv[2 * i].numpy()
        feeds[f"past_key_values.{g}.value"] = kv[2 * i + 1].numpy()
    got = sess.run(None, feeds)
    err = float(np.max(np.abs(got[0].astype(np.float32) - gold[0].numpy().astype(np.float32))))
    print(f"[selftest] hidden_out max_abs_err onnx-vs-torch = {err:.3e} -> "
          f"{'PASS' if err < 1e-2 else 'FAIL'}")
    print(f"[selftest] present shapes: " +
          ", ".join(f"{out_names[1+i]}={got[1+i].shape}" for i in range(len(shard_globals) * 2)))
    return err < 1e-2


# --------------------------- full export ---------------------------
def full_export(model_dir, out_dir, lps, win):
    import transformers
    from transformers import AutoConfig
    from transformers.models.gemma4_unified import modeling_gemma4_unified as _M
    from transformers.models.gemma4_unified.modeling_gemma4_unified import (
        Gemma4UnifiedTextRotaryEmbedding,
    )
    if os.environ.get("SAFE_NORM") == "1":
        import patch_rmsnorm
        patch_rmsnorm.apply(_M)
        print("[export] applied overflow-safe RMSNorm (pre-scale 256)", flush=True)
    if os.environ.get("SAFE_ATTN") == "1":
        import patch_attention
        patch_attention.apply(_M)
        print("[export] applied fp16-safe attention (scaled matmul + stable softmax)", flush=True)
    os.makedirs(out_dir, exist_ok=True)
    cfg = AutoConfig.from_pretrained(model_dir).get_text_config()
    cfg._attn_implementation = "eager"
    print(f"[export] loading full unified checkpoint, then taking text backbone "
          f"({cfg.num_hidden_layers} layers)...", flush=True)
    t0 = time.time()
    # Load the full model the same way the (validated) reference did, then navigate to the
    # text backbone — guarantees the real weights (avoids prefix-mismatch random-init).
    full = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.float16, low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    tm = full.model.language_model
    tm.config._attn_implementation = "eager"
    cfg = tm.config
    print(f"[export] loaded in {time.time()-t0:.0f}s; text backbone has "
          f"{len(tm.layers)} layers, embed{tuple(tm.embed_tokens.weight.shape)}", flush=True)
    rotary = tm.rotary_emb if hasattr(tm, "rotary_emb") else Gemma4UnifiedTextRotaryEmbedding(cfg).half()
    layers = list(tm.layers)

    # Dump the CPU-side pieces the shards DON'T contain: the scaled input embedding
    # (also the tied lm_head), the final RMSNorm, and the logit softcap.
    emb = tm.embed_tokens.weight.detach().float().cpu().numpy().astype(np.float16)
    np.save(f"{out_dir}/embed_tokens.npy", emb)
    np.save(f"{out_dir}/norm.npy", tm.norm.weight.detach().float().cpu().numpy().astype(np.float16))
    embed_scale = float(getattr(tm.embed_tokens, "scalar_embed_scale", cfg.hidden_size ** 0.5))
    print(f"[export] dumped embed_tokens{emb.shape} + norm; embed_scale={embed_scale:.3f}", flush=True)

    n = cfg.num_hidden_layers
    shards = [list(range(i, min(i + lps, n))) for i in range(0, n, lps)]
    json.dump({"win": win, "layers_per_shard": lps, "num_layers": n,
               "shards": shards, "hidden": cfg.hidden_size,
               "layer_types": list(cfg.layer_types),
               "embed_scale": embed_scale,
               "final_logit_softcapping": cfg.final_logit_softcapping,
               "rms_norm_eps": cfg.rms_norm_eps},
              open(f"{out_dir}/shards.json", "w"), indent=2)
    for si, sg in enumerate(shards):
        path = f"{out_dir}/fp16shard_{si}.onnx"
        if os.path.exists(path):
            print(f"[export] shard {si} exists, skip", flush=True); continue
        t = time.time()
        export_one_shard([layers[g] for g in sg], sg, rotary, cfg, win, path)
        sz = os.path.getsize(path) / 1e6
        print(f"[export] shard {si} layers={sg} -> {path} ({sz:.0f} MB, {time.time()-t:.0f}s)", flush=True)
    print(f"[export] DONE: {len(shards)} shards in {out_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="models/gemma-4-12B-it")
    ap.add_argument("--out", default="out/gemma4_fp16")
    ap.add_argument("--layers-per-shard", type=int, default=4)
    ap.add_argument("--win", type=int, default=64)
    args = ap.parse_args()
    if args.selftest:
        ok = selftest()
        sys.exit(0 if ok else 1)
    full_export(args.model, args.out, args.layers_per_shard, args.win)
