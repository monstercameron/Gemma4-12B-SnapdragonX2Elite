# Running Gemma 4 12B on a Snapdragon X2 — the full journey

Getting Google's **Gemma 4 12B** to run *fast* on a **Snapdragon X2 Elite Extreme** laptop
(Windows 11 ARM64, Adreno X2‑90 GPU, Hexagon NPU, Oryon CPU, 48 GB unified LPDDR5x), starting
from Google's own tooling and ending with a hand‑written **raw‑Vulkan compute engine**.

## TL;DR

```
Gemma 4 12B decode speed, this laptop:
  Google litert-lm            could not load on this chip
  custom ONNX/QNN on NPU      runs but wrong token (fp16 overflow + QNN error 1003)
  custom ONNX/QNN on GPU      correct, ~0.024 tok/s (swap-bound)
  llama.cpp Vulkan            6.06 tok/s
  custom wgpu shader engine   7.07 tok/s
  llama.cpp CPU               7.9  tok/s
  ► custom RAW VULKAN engine  13.84 tok/s   ← final result, beats everything
```

A **~577× speedup** from the first working number, ending **+75% faster than llama.cpp CPU** —
every kernel measured and rebuilt from microbenchmark data, then driven by a command buffer
recorded once and resubmitted per token. Beyond decode: **prefill mega-optimized to ~210 prompt-tok/s
(2.2×)** via fp16 KV + an int8 cooperative-matrix GEMM, a **64k context**, an **OpenAI-compatible
server** with **transparent prefix KV caching**, and opt-in fp8 decode scales — each landed (or
declined) on measured evidence.

---

## The model

`google/gemma-4-12B-it` — architecture `Gemma4UnifiedForConditionalGeneration`, `model_type
gemma4_unified`. Apache‑2.0, **not gated**. Text backbone:

| | |
|---|---|
| layers | 48 (dense; **no PLE, no MoE** for the 12B) |
| hidden | 3840 · intermediate 15360 · vocab 262144 |
| attention | 16 heads; **sliding** layers = 8 KV heads, head_dim 256; **full** layers (every 6th) = 1 KV head (MQA, `k_eq_v`), head_dim **512**, partial RoPE 0.25 |
| other | scaling = 1.0, fp32 softmax, `final_logit_softcapping = 30`, scaled embedding ×√3840 |

`transformers 5.10.1` supports it; **`onnxruntime-genai` 0.14 does not** (the encoder‑free design
breaks the Gemma‑3 builder — variable head dims, per‑layer inputs). So everything custom had to
hand‑roll the export.

---

## Phase 1 — Google's tooling (`litert-lm`)

Installed `litert-lm`, imported the `litert-community/gemma-4-12B-it-litert-lm` GGUF‑equivalent.

- **CPU backend:** flatly rejected — the build is compiled GPU‑only.
- **GPU backend:** died on a **256 MB max‑buffer cap** (WebGPU/Dawn default) vs the model's ~1 GB
  logits buffer (32003 vocab × 2048 ctx × fp16).

**Verdict:** Google's own path can't even load Gemma 4 on this chip. Motivation to go custom.

## Phase 2 — Custom `gemma4_unified` ONNX exporter

Since the genai builder is out, wrote a `torch.onnx` exporter that **reuses the real HF
`layer.forward`** (so the attention/RoPE/scaling are exactly correct) and emits 12× 4‑layer fp16
shards. Validated: **CPU reproduces the HF golden token exactly** (236770 raw / 818 chat).

## Phase 3 — The NPU (Hexagon HTP) — fully characterized dead end

The fp16 shards **place and finalize on the HTP** (better than the prior Qwen work — no error
1002), but produce the **wrong token**. Op‑level HTP‑vs‑CPU diffing found two real bugs:

1. **fp16 overflow** (Gemma activations are bf16‑range; CPU silently uses fp32 accumulation, the
   HTP uses native fp16):
   - `RMSNorm.pow(2)` overflows when any activation > 256 (256² > 65504). **Fix:** divide by the
     per‑row max before squaring (exact, no over/under‑flow). A fixed `/256` was tried first and
     *underflowed* small activations — the dynamic per‑row‑max is the right version.
   - attention `q·k` / softmax overflow → **scaled matmul + explicit stable softmax**.
   - These took single‑token per‑shard cosine from **0.16 → 0.99**.
2. **The real wall — QNN HTP execute error 1003.** Even with overflow fixed: compiled HTP context
   binaries miscompute to NaN; direct‑ONNX throws error 1003 and silently CPU‑falls‑back (which
   made early "cos 1.0" diagnostics *misleadingly* look like HTP success). **ORT‑QNN 2.2 cannot
   faithfully execute these graphs on the HTP** — needs native QAIRT, not pursued.

**Lesson logged:** the fp16‑overflow root‑causes are new and reusable; the error‑1003 wall is the
same class the prior Qwen NPU work hit.

## Phase 4 — The GPU via ONNX/QNN, and a memory‑budget correction

The exact same fp16 shards **run correctly on the Adreno via QnnGpu** (`'Paris'`) — fp32
accumulation, no overflow, no 1003. But decode was **swap‑bound at ~0.024 tok/s** because the
fp16 model is too big to stay resident through ORT's execution providers.

I initially claimed a hard ~16 GB GPU memory ceiling (QnnGpu/OpenCL) / ~10–12 GB (DirectML/D3D12).
**That was overstated** — the driver actually reports ~24 GB (OpenCL) / ~28 GB (Vulkan) available
(34 GB system RAM was free when it failed). The ceilings I hit were **ORT‑EP allocation limits**,
not the hardware. (Caught and corrected mid‑investigation.) Either way: **int4/GGUF (~7 GB) is the
fit‑resident path.**

## Phase 5 — llama.cpp (the deployable baseline)

Gemma 4 is supported in llama.cpp since launch. On the existing Adreno build with the unsloth
`UD‑Q4_K_XL` GGUF (6.85 GB):

```
CPU (14 threads):  ~7.9 tok/s decode   (the deployable max)
Vulkan (Adreno):    6.06 tok/s decode / 39.5 tok/s prefill
```

`llama-server` gives the OpenAI‑compatible endpoint — the original `litert-lm serve` goal, the
proper way. Cleaned up ~213 GB of dead ORT/NPU artifacts here.

## Phase 6 — A from‑scratch wgpu compute‑shader engine

Built a complete custom GPU engine (`engine_gpu_full.py`) in **wgpu/WGSL** — int4 GEMV, RMSNorm,
RoPE, attention (q/k/v‑norm, GQA, scores/softmax/×V), MLP, residuals — **everything on GPU, one
submit + one logits read per token**, resident KV cache. The climb:

```
hybrid (per-GEMV readback)   2.0
resident + combined-qkv      4.6
+ fused norm+residual        6.28
+ microbench-tuned kernel    7.07 tok/s
```

### Microbenchmark‑driven kernel rebuild (the real win)

Decomposed the GEMV (`load → +unpack → +scale → +full`) on synthetic data:

- small‑N matmuls are **occupancy‑bound, not ALU‑bound** (full GEMV ≈ load‑only ceiling).
- the fixed `split=8` starved them: **`down` (N=3840) went 43 → 92 GB/s at `split=24`** (2.1×);
  large‑N (`gate`) was already saturated (flat ~98).

Governing quantity: **total workgroups = `(N/8/WGS) × split`**. Rebuilt the kernel with
**`workgroup_size=64`** (one Adreno wave) **+ adaptive split targeting ~192 workgroups** → GPU
`read` 126 → 88 ms, decode **6.28 → 7.07 tok/s**, beating llama.cpp Vulkan.

### Dead ends (measured, documented)

| Idea | Result | Why |
|---|---|---|
| Single‑pass GEMV (workgroup K‑reduce) | **12× slower** | breaks weight coalescing (strided reads) |
| Fold GEMV reduce into consumers | neutral/worse | reduce passes are cheap; folding double‑reads partials |
| `blk` 32→128 (less scale traffic) | token corruption | int4 accuracy loss over 48 layers |
| `x` activation in shared memory | **−50%** | x already L2‑resident; the barrier kills latency‑hiding |

## Phase 7 — The wall is WebGPU; drop to raw Vulkan

Profiling one token: **996 compute passes → ~6,000 wgpu FFI calls**, split roughly
`submit/encode ~100 ms` (CPU, command‑buffer serialization) + `read ~88 ms` (GPU execution).
The CPU half is a **WebGPU limitation** — command buffers are single‑use and there's **no compute
equivalent of render bundles**, so the *byte‑identical* 996‑pass command buffer is rebuilt every
token.

The stack was always native WebGPU: `wgpu-py → wgpu-native → Vulkan (Adreno)`. The limit is the
*abstraction*, not the silicon — plain Vulkan supports **record‑once / resubmit**. So I dropped a
layer down.

### Raw Vulkan engine — `11.04 tok/s`

Verified the pieces were all present (Vulkan loader, SDK 1.4.350 + `glslangValidator`, the `vulkan`
cffi binding on ARM64 — `kompute` wouldn't build), then:

1. Ported all 8 kernels **WGSL → GLSL → SPIR‑V** (`vk/*.comp` → `vk/*.spv`).
2. Built the engine in raw Vulkan (`vk_engine.py`): instance/device/queue, int4 weights in device
   buffers, descriptor sets, pipelines.
3. **Recorded the entire ~600‑dispatch token graph into one command buffer**, resubmitted per token
   — only the embedding / position / cos‑sin written into mapped UMA buffers.

Result: correct (`'Paris'`), **11.04 tok/s** — the WebGPU ~100 ms/token encode+submit *eliminated*,
exactly as the profile predicted. **+40% over llama.cpp CPU.**

---

## Phase 8 — Decompose → microbench → reassemble (the bandwidth pass)

After the Vulkan engine, the task became: *maximize bandwidth utilization*. The first thing was to
measure the ceiling honestly.

### The GPU's actual bandwidth

Decode is a memory‑bandwidth‑bound GEMV (M=1, ~6.5 GB of weights read per token). So the only
ceiling that matters is memory bandwidth, and on this UMA SoC the GPU shares the LPDDR5x bus:

| | GB/s | note |
|---|---|---|
| SoC theoretical (LPDDR5x) | **~230** | hard wall — shared across CPU/GPU/NPU; no kernel exceeds it |
| GPU achievable, int4 streaming | ~125 | **measured** (`microbench_gemv.py` split sweep) — ~54% of the SoC bus; the realistic GPU ceiling |
| GPU achievable, generic vec4 read | ~106 | measured directly (`microbench_bw.py`, 256 MB) |
| my real GEMV at the right split | ~92–106 | **~85% of the achievable GPU ceiling** |
| system‑level useful (6.5 GB ÷ token) | ~70–90 | the rest is inter‑dispatch overhead |

The takeaway that reframed everything: **the individual GEMVs are already ~85% of the achievable GPU
bandwidth.** `~125 GB/s ÷ 6.5 GB/token ≈ 19 tok/s` is the realistic decode ceiling (the SoC's ~230 GB/s
÷ 6.5 ≈ 35 tok/s is a wall no GPU kernel reaches — a UMA GPU macro only gets ~half the bus); we're at
~11→13.84. The gap to ~19 is the strided‑read penalty plus per‑token fixed overhead
(attention, norms, reduce passes, ~600 barriers) — **not** anything tweakable in the GEMV math.

### The split heuristic was measured wrong

The old GEMV split tuner used a *pass‑1‑only* microbench, blind to the second pass's partial‑buffer
round‑trip. A proper **two‑pass total‑time** microbench (`microbench_total.py`) showed the real
optimum is **~384 total workgroups** (down→48, gate→12, qkv→24), and — critically — that the
giant lm_head (N=262144) **must stay at split=1** (8.8 ms): splits 2–30 are a *catastrophic ~27 ms
zone*. An earlier "optimization" had been shoving lm_head straight into that zone. Fixed in
`vk_engine.py`; the heuristic is now derived from measured ground truth.

### Attention — the last un‑tuned block

Every GEMV had been microbenched; attention never had. Reading it back, `ATTN` had a **serial
single‑thread softmax** (thread 0 doing all max/exp/sum while 255 threads idled) and only launched
16 workgroups. A standalone microbench (`microbench_attn.py`, current vs. optimized, cos‑validated)
quantified the fix — replacing the serial softmax with a **parallel tree reduction**:

| case | current | parallel softmax | speedup |
|---|---|---|---|
| sliding hd=256, T=16 | 72 µs | 57 µs | 1.26× |
| sliding hd=256, T=256 | 210 µs | 131 µs | **1.60×** |
| full hd=512, T=256 | 317 µs | 200 µs | **1.58×** |

All `cos=1.0000` (numerically identical). Ported into `vk/attn.comp`, recompiled, validated
end‑to‑end. **Short‑context tok/s is unchanged** (~10.7) because at T≈10 attention is only ~3.7% of
the token and a 10‑element softmax was already cheap — this is a *scaling* win that grows with
context, exactly the regime a real deployment hits.

### Wide loads — the big one (`+29%`)

A load‑width microbench (`microbench_width.py`) asked: does Adreno give more effective bandwidth
from a *wider* weight load? Yes, decisively — reinterpreting the weight buffer as 64‑bit (`uvec2`,
16 outputs/thread) instead of 32‑bit (`u32`, 8/thread) lifts the GEMV load path **~82 → ~109 GB/s**
(and lm_head 27 → 63). Fewer, wider memory transactions = less load/store‑unit pressure. Rewrote
`vk/gemv.comp` to `uvec2` (16 outputs/thread, 4 vec4 accumulators + 4 vec4 scales — the register
sweet spot; `uvec4`/32‑wide spills and regresses). Same weight bytes, reinterpreted view; cos
0.99514 unchanged. **End‑to‑end: ~10.7 → `13.84 tok/s`, correct ('Paris') — +29%.**

### Honest steady state

The custom raw‑Vulkan engine is at **`13.84 tok/s`** (the earlier "11.04" was an optimistic 4‑token
run; ~10.7 was the honest pre‑uvec2 baseline). Every compute block — all GEMVs *and* attention — is
microbench‑tuned, and the core GEMV now uses wide loads. Remaining levers are **structural and
measured‑marginal or large‑build**: single‑pass **subgroup‑reduce GEMV** (kills the partial
round‑trip + reduce dispatch — composes with uvec2, the best remaining bet), **dropping barriers**
between independent dispatches (Q/K/V, gate/up), **texture‑cached weights** (big build, lm_head dim
limit), **W4A8 integer‑dot** (a *prefill* win — neutral for bandwidth‑bound decode), and **fp16 x
storage** (minor; x is L2‑resident).

### Shader‑optimization scorecard

The brief was "do all the GPU shader optimizations and benchmark." Rather than build changes the
microbenchmarks already predicted to be neutral, each was taken to a decision with evidence:

| # | Optimization | Verdict | Evidence / reason |
|---|---|---|---|
| 1 | **uvec2 wide weight loads** | ✅ **shipped, +29%** | `microbench_width.py`: 82→109 GB/s load path; end‑to‑end 10.7→**13.84 tok/s**, cos 0.99514 |
| 2 | single‑pass subgroup‑reduce GEMV | ⏳ deferred — best next bet | composes with uvec2; kills partial round‑trip + reduce dispatch. Large correctness‑critical rewrite |
| 3 | texture/image2D weights | ⏳ deferred — blocked | Adreno texture cache could help, but lm_head N=262144 > image dim limit; big build |
| 4 | W4A8 + integer dot‑product | ❌ not worth it (decode) | decode is bandwidth‑bound; W4A8 doesn't cut weight bytes → ~neutral. A *prefill* win |
| 5 | fp16 x (activation) storage | ❌ marginal | x is only K floats, already L2‑resident; needs `shaderFloat16` plumbing for ~0 gain |

System‑level useful bandwidth went ~70 → **~90 GB/s** (13.6 × 6.5 GB), i.e. **~73% of the ~123 GB/s
achievable GPU ceiling** (was ~58%). The hard physical wall is ~123 GB/s ÷ 6.5 GB/token ≈ **19 tok/s**
for this model at int4; #2 (subgroup‑reduce) is the realistic path to close more of the remaining gap.

### Whole‑engine audit (everything else)

After uvec2, audited every non‑GEMV kernel + the dispatch/barrier graph. Findings & outcomes:

- **Softcap skipped for greedy decode** — `cap·tanh(x/cap)` is monotonic, so `argmax(softcap)==argmax`.
  Removed a 262144‑wide dispatch + barrier + 2 MB/token at zero correctness cost. ✅ kept.
- **lm_head reduce eliminated** — lm_head is split=1, so its reduce was a pure 262144‑wide copy. The
  GEMV now writes straight to the logits buffer (`split==1` path), dropping the dispatch+barrier. ✅
- **vec4 reduce** — vectorized the reduce kernel (runs ~288×/token). ✅ kept (cos unchanged).
- **Split re‑tune to isolated optima — REVERTED.** Re‑microbenched the uvec2 GEMV split
  (`microbench_uvec2_split.py`): small‑K matmuls peak at split≈60 *in isolation*, but forcing that in
  the full engine **regressed −8%** — extra partial/reduce traffic across 5 matmuls × 48 layers beats
  the per‑matmul gain. The workgroup heuristic is the real full‑pipeline optimum. ❌ reverted.

Net: the three kept fixes are correct and remove genuine waste, but land **within noise** at this
decode profile — because the GEMV weight read (6.5 GB) dwarfs everything else (the removed dispatches
were ~3 MB = 0.05%). **The engine is GEMV‑bandwidth‑bound; only the weight‑read path moves tok/s.**
They still matter for longer context / prefill, where the reduces and softcap aren't negligible.
Honest stable decode: **~13.6 tok/s**.

### The full progression

| Engine | Decode tok/s | vs. start |
|---|---:|---|
| ONNX/QNN → GPU (where this began) | 0.024 | 1× |
| llama.cpp Vulkan | 6.06 | 250× |
| llama.cpp CPU | 7.9 | 330× |
| custom wgpu (WebGPU‑bound) | 7.07 | 295× |
| custom raw‑Vulkan (scalar GEMV) | ~10.7 | 445× |
| **custom raw‑Vulkan + uvec2 wide loads** | **13.84** | **577×** |

---

## Phase 9 — OpenAI‑compatible serving (FastAPI)

Wrapped the engine in a drop‑in OpenAI API. `vk_engine.py` was refactored to be importable (demo
guarded under `__main__`; exposes `forward(tid,pos)`, a `generate()` token generator, and `sample()`
with greedy / temperature / nucleus top‑p). `scripts/serve.py` is the FastAPI app:

- **Endpoints:** `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`
  — streaming (SSE: role chunk → content deltas → final `finish_reason` + `usage` → `data: [DONE]`)
  and non‑streaming, with exact `usage` token counts.
- **Single global GPU command buffer + KV cache → one `threading.Lock`** serializes requests (correct
  for a local single‑GPU server). Each request starts at `pos=0` so the KV cache (slots `0..pos`)
  overwrites cleanly — no cross‑request state. Prompt is left‑truncated to `MAXT=256`.
- **Stop tokens from the model's own `generation_config.eos_token_id` = `[1,50,106]`** (the turn
  delimiter is `<turn|>`=106, *not* `<end_of_turn>` — found by reading the config, not guessing).
- **Validated against the official `openai` Python client:** `models.list()`, non‑stream chat
  (`"4"`, `finish_reason="stop"`, usage), and streaming (`"Bonjour !"`) all correct.

```powershell
.venv-gemma4\Scripts\python.exe scripts\serve.py --host 127.0.0.1 --port 8000
# then: OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x").chat.completions.create(...)
```

---

## Phase 10 — 64k context (flash-decode + sliding-window ring buffer)

The 256-token cap was a hard limit of the original attention kernel (one thread per KV position,
256-wide shared softmax). Lifting it to 64k took a focused re-architecture — but two insights
collapsed the four "subsystems" into a tractable integration:

- **The flash kernel loops tiles internally on a `T` uniform → the command buffer stays fixed.** No
  indirect dispatch needed: attention dispatches `nH` workgroups once, and each kernel loops
  `ceil(T/256)` tiles based on `pos` written per token. Variable work, fixed record-once graph.
- **Softmax is permutation-invariant + RoPE is baked in at write time → sliding layers can use a
  plain ring buffer.** Slot = `pos % 1024`; attention reads all valid ring slots in whatever order.
  No wrap-handling in the hot loop.

Built and validated:
1. **Flash-decode attention** (`vk/attn.comp`, microbench `microbench_flash.py`): online-softmax tiled
   over the KV sequence, q-norm+RoPE preserved. Numerically exact — `cos=1.0` at T = 16 … 8192.
2. **Sliding-window ring buffer** (`vk/prepkv.comp`): the 40 sliding layers keep a `WIN=1024` ring;
   the 8 global layers keep a full `CTX=65536` cache. Per-layer `stride`/`window` in the uniforms.
3. **Memory fits:** `0.67 GB (sliding) + 2.1 GB (global) + 6.5 GB weights = ~9.3 GB` in 48 GB.

**Validation:** correct `'Paris'` at short context with **zero speed regression (13.6 tok/s)**, and a
**needle-in-haystack at 1602 tokens** — secret passcode hidden at the start, retrieved correctly at
the end (`DECODE: 'CRIMSON-7492…'`, logits finite). Crossing the 1024 window means that retrieval
*proves the global long-context attention path*, not just the sliding window.

**Honest remaining work for *practical* 64k** (correctness is done; these are performance):
- **Batched prefill.** Prefill is still one token per pass, so filling 64k ≈ 75 min. Real use needs an
  M>1 GEMM prefill path. (1602-token prefill took ~149 s.)
- **Multi-query global attention.** At full 64k the MQA global layers re-read the shared KV once per
  query head (16×) — measured 663 ms/token of attention at 64k. Multi-query/split-KV flash-decoding
  fixes it to ~tens of ms. Negligible below a few-k context.

The OpenAI server (`serve.py`) inherits `MAXT=65536` automatically — it now accepts up to 64k-token
prompts (subject to the prefill-time caveat above).

---

## Phase 11 — Batched prefill (the painful-prompt fix)

Per-token prefill re-reads all 6.5 GB of weights for *every* prompt token, so a 1.6k-token prompt took
~147 s. Fix: process the prompt in **chunks of MC=128 tokens through an int4 GEMM** so each weight is
read once and applied to all 128 tokens.

- Validated the core first (`microbench_gemm.py`): batched int4 GEMM amortizes the weight read
  **8–26×** on Adreno, `cos=1.0`. Then built the batched path: `vk/gemm.comp` +
  `vk/pre_norm/pre_na/pre_prepkv/pre_attn.comp` (gelumul is elementwise → reused).
- **Additive, non-breaking:** batched prefill only *builds the KV cache* for tokens `0..n-2` (no
  lm_head); the validated decode path handles the tail + last token + generation. Short prompts
  (< MC) skip it entirely.
- **Causal mask for the sliding ring:** writing a whole chunk's KV before its attention lets an early
  token see later same-chunk tokens once the 1024-ring wraps — so `pre_attn` masks keys whose absolute
  position is in the future (`posn>pos`). Global layers are clipped by `T=pos+1` and need no mask.
- **TDR:** the full 48-layer chunk in one submit (~5 s) tripped the Windows GPU watchdog
  (`VK_ERROR_INITIALIZATION_FAILED`). Split into **per-layer submits** (~100 ms each); buffers persist
  across submits so the chunk flows.

**Result: 1602+24 tokens 148.6 s → 59.6 s (~2.5×), needle still retrieved, logits finite.** The
batched matmuls dropped to ~34 ms/token (from ~92), which **exposed MQA-redundant prefill attention as
the new bottleneck** — the 16 query heads of each global layer re-read the single shared KV head, and
that isn't amortized by batching. Multi-query prefill attention (read KV once for all heads) is the
next lever, the same Phase-10 item now dominating.

---

## Phase 12 — Faster prefill: multi-query (dead end) → coopmat matrix cores

Chasing faster prompt processing past the 2.5× of batched prefill:

- **Multi-query attention — measured dead end for prefill.** Built it (process the query heads sharing
  a KV head in one workgroup, KV read once), validated `cos=1.0`, but **0.95× — no speedup**
  (`microbench_mqa.py`). The reason: prefill is **compute-bound** (batched GEMM is ~260 FMA/byte), so
  cutting redundant KV *memory* traffic does nothing; the Adreno's L2 amortizes it anyway. (Multi-query
  *would* help 64k *decode*, which is memory-bound — but not prefill.)
- **Coopmat matrix cores — the right lever, proven.** The Adreno X2-90 exposes `VK_KHR_cooperative_matrix`
  with **f16 64×64×16** and **s8×s8→s32 64×64×32** tiles (subgroup scope). De-risked the whole stack
  (`vk/coopgemm.comp`, `scripts/coopgemm.py`): device-feature enablement through the cffi binding +
  glslang coopmat SPIR-V + Adreno execution all work, `cos=0.99998`. Tiled across the GPU it hits
  **3169 GFLOP/s vs ~428 for my register-tiled int4 GEMM — ~7.4×.** Since prefill is GEMM-compute-bound,
  this is the fix.

**int4 coopmat GEMM — built & validated** (`vk/coopgemm_i4.comp`, `scripts/coopgemm_i4.py`): unpacks the
int4 weight tile + bakes the per-block scale into fp16 in shared, then `coopMatMulAdd`. Measured at the
real prefill dims (M=128): **down 35→5.87 ms (6.0×), gate 28.5→3.81 ms (7.5×)** vs the register GEMM,
**cos 0.9999–1.0** — fp16 accumulation holds even at K=15360. All prefill dims are clean multiples of
the 64/64/16 tile (no padding). 2,600–4,000 GFLOP/s.

**INTEGRATED (measured end-to-end).** Used the **fp32** coopmat path (`f32 64×64×8`, `vk/coopgemm_i4f.comp`)
instead of fp16 — a clean drop-in for the existing fp32 prefill buffers with *zero* conversion plumbing
(cos=1.0000, exact). Enabled `cooperativeMatrix` + `vulkanMemoryModel` in `vkCreateDevice` (Vulkan 1.1,
additive — decode unaffected) and swapped `gemm.comp`→`coopgemm_i4f` in the prefill command buffer.

**Needle test 1602+24 tok: 148.6 s (per-token) → 59.6 s (batched) → 36.3 s (fp32 coopmat) → `31.8 s`
(fp16 coopmat)** — needle retrieved, logits finite throughout. The GEMM does the bulk; attention/norms
don't speed up, so end-to-end gains dilute (coopmat 1.64× on batched; fp16 a further 1.14×). **Net ~4.7×
faster prefill (~61 prompt-tok/s).**

**fp16 upgrade:** the Adreno coopmat fp16 tile is K=16 vs fp32's K=8 → ~1.2–1.5× more MACs/instruction
(measured: gate 2572→3963 GFLOP/s). Switched the prefill GEMM to fp16: each matmul is now
`f32→f16 cast → fp16 coopMatMulAdd → f16→f32 cast` via shared fp16 staging buffers (`vk/coopgemm_i4h`,
`cast_dn`, `cast_up`); added `shaderFloat16`+16-bit-storage to `vkCreateDevice`. The surrounding kernels
stay fp32 (residual accumulation unchanged), fp16 confined to the matmul (cos 0.9999).
**Decode unchanged at ~13 tok/s** (coopmat is M=1-useless; decode is bandwidth-bound).

---

## Phase 13 — GPU timestamp profiling (where the token actually goes)

Added a real per-kernel GPU profiler to `vk_engine.py` (`PROFILE=1`): `vkCmdWriteTimestamp` after each
op in the decode command buffer, read back via `vkGetQueryPoolResults`, aggregated by kernel type
(`print_profile()`). It's a permanent tool — re-profile any kernel on demand. (The profiled run reads
~13→11 tok/s because the per-token timestamp readback adds CPU overhead; the *percentages* are exact
GPU-side timing.)

**Decode, per token:**

| kernel | % | | kernel | % |
|---|---:|---|---|---:|
| gate_up | **42.3** | | o_proj | 7.2 |
| down | 21.1 | | norm | 1.5 |
| qkv | 15.1 | | normadd | 1.4 |
| lm_head | 9.3 | | **attn** | **1.3** |
| | | | prepkv / gelu | 0.7 |

**Findings:**
- **94.5% of decode is GEMV weight reads** (gate_up+down+qkv+lm_head+o_proj). The time split matches
  the *weight-byte* split almost exactly (gate_up is ~43% of bytes → 42% of time) — hard proof that
  decode is purely **bandwidth-bound**, ~106 of the ~123 GB/s ceiling.
- **Everything non-GEMV is ~5% combined.** Optimizing norms/attention/gelu is wasted effort —
  attention is **1.3%**, which retroactively confirms coopmat-attention wouldn't help *decode*.
- **No free decode micro-opts exist.** The only lever the profile surfaces: per-block fp16 **scales**
  are ~12.5% of the GEMV bytes (`nblk·N·2` vs `K·N/2`) → fp8 scales could save ~6% (precision risk).
  The reduce pass is only ~2–3% (partials are tiny vs weights).

The profile settles the optimization question for decode: it's at the bandwidth wall, and remaining
wins are quantization-level (fewer/smaller bytes), not kernel micro-opts.

---

## Phase 14 — Decode at the bandwidth wall: fp8 scales (opt-in)

The Phase-13 profile settled that decode is 94.5% GEMV weight reads at the bandwidth wall, with one
sub-weight lever: the per-block fp16 **scales** (~11% of GEMV bytes). Shipped it opt-in:

- **`GEMV_FP8=1`** stores the int4 per-block scales as fp8 e4m3 (1 byte vs fp16's 2). Naive e4m3 lands
  the ~0.01-magnitude scales in the *subnormal* range (precision collapse); fixed with a **per-tensor
  power-of-2 normalization** into e4m3's normal band [2⁻⁶,448], the inverse folded into the GEMV output.
  Shader decode verified bit-exact vs torch's encoder (≤2.34%/scale over an 80× spread).
- A/B (interleaved best-of-2, 128-tok bursts): **+4.5–8% decode**, fp8 won both rounds; needle retrieved,
  decoded text identical. Flag-gated, default fp16 (zero quality risk).

A full GLSL Adreno-fit pass alongside it: **gemv `vec4` partial stores** (16 scalar → 4 wide, kept), and
**subgroup-arithmetic reductions** in attention — compiled clean but produced **non-finite logits** on
this Adreno, so reverted (<1% decode impact anyway). Measure, don't assume.

## Phase 15 — Prefill, mega-optimized: fp16 KV + W8A8 int8 coopmat (~2×)

A `PREPROF=1` per-kernel prefill profiler (timestamps in the batched-prefill graph) gave ground truth:
**attn 30%, GEMMs 66%, casts 2%**. Two real wins; the GEMM micro-opt that *looked* obvious was a trap.

- **dequant-hoist — reverted.** The coopmat GEMM re-dequantizes each weight tile once per M-row-tile;
  hoisting it (2 accumulators sharing one dequant) *regressed* prefill 95→84 prompt-tok/s — two 64×64 f16
  accumulators spill registers and collapse occupancy. The GEMMs are **occupancy/matrix-core-bound, not
  dequant-bound.**
- **fp16 KV cache — +10.5%, default on.** Prefill attention re-streams the whole K/V cache once per query
  token; the reads are L2-absorbed but still cost transaction bandwidth. fp16 storage (was f32) halves
  them: prefill 95→105 prompt-tok/s, **attention 30%→18% (−45%)**, and **2× the reachable 64k context**
  (KV memory halved). K (post-norm/RoPE) and V are O(1) → fp16 lossless; dots still accumulate f32.
- **W8A8 int8 coopmat GEMM — ~2×, `PREFILL_I8=1`.** A no-model-load microbench (`benchmarks/coopgemm_i8.py`)
  found the Adreno's **int8 coopmat is 3.7× the f16 path** (s8×s8→s32, K=32 tile vs f16 K=16), and the
  prefill GEMMs (77%) used f16. Two routes, both de-risked in `benchmarks/coopgemm_w4a8.py`:
  - **W4A8** (int4 weight + on-the-fly int8 requant): no extra RAM, but the per-element float-mul+round
    requant dominates → **0.27× (4× slower)**. Documented dead end.
  - **W8A8** (precomputed int8 weights, per-column scale — the int4 per-block scale folded in at load):
    pure int8 matmul + one final rescale → **10273 GFLOP/s, 2.78× the f16 path, cos 0.99993** (int8
    activations essentially free). Costs 2× weight RAM (int8 alongside int4; decode keeps int4).
  - **Integrated** (`cast_i8` per-row int8 activation quant → `coopgemm_w8a8` → rescale): measured
    **prefill 105 → 209.8 prompt-tok/s (2.0×)**; GEMMs qkv 4.4×, down 3.6×, o 3.3×, gateup 2.2×. Needle
    retrieved, text correct. ~12 GB extra RAM (fits 48 GB), flag-gated, decode untouched.

**Net prefill: 95 → 210 prompt-tok/s (2.2×) at identical quality.**

## Phase 16 — Transparent prefix KV cache (server)

A shared prompt prefix (system prompt, or a growing multi-turn conversation) produces identical KV every
request, but the engine re-prefilled it each time. Added **prefix KV caching** — the mechanism behind
"OpenAI-style prompt caching":

- The engine snapshots a prefix's KV (global layers: first `CACHE_MAX=2048` slots/kv-head; sliding layers:
  the whole ring) into side buffers, and **restores it with a ~ms GPU copy** instead of recomputing (vs
  *seconds* to re-prefill). `generate(reuse_chunks, snap_chunks)` skips/snapshots the first N prefill chunks.
- `serve.py` auto-detects the prefix as the chunk-aligned longest-common-prefix of consecutive requests;
  reuses **only at the exact cached length** so the sliding-window ring stays clean. A generation lock
  serializes requests (single-GPU engine isn't reentrant — also fixed a latent concurrency bug).
- **Correctness:** `test_prefix_cache.py` — a reused-prefix request yields **byte-identical** tokens to a
  cold full-prefill. **Live:** **−26–29% latency** on a 505-token shared system prompt; longer prefixes
  save proportionally more. Transparent (no API change), warms up after 2 requests.

## Phase 17 — Speculative decoding & MTP: measured dead ends

Two decode accelerators evaluated and **correctly declined**, with data not vibes:

- **MTP (multi-token prediction):** the Gemma-4 config has no MTP/nextn heads (standard NTP, tied lm_head)
  — self-speculation needs trained draft heads the model doesn't have.
- **Prompt-lookup speculative decoding:** decode is bandwidth-bound and the only M>1 verify path is the
  prefill chunk, costing ~8 (int8) / ~16 (f16) decode-tokens. Measured acceptance on real generations
  (`benchmarks/spec_acceptance.py`): **echo 9.3 tok/step (marginal), code 3.1 (loss), chat 1.2 (big loss)**.
  Only contrived verbatim echo clears the bar. A draft model would raise acceptance but needs a second
  model. **Not built** — the measurement says it'd be net-negative for real use.

## Phase 18 — Snapdragon X2 prefill tail buckets: measured, then rolled back

After the broader Snapdragon X2 optimization review, I tested a tempting tail optimization: record an
extra `M=64` coopmat prefill graph so the `0..127` prompt tokens left after full `MC=128` chunks could
avoid decode-style one-token prefill.

The narrow benchmark looked promising. In one process, with the same model load and greedy `max_new=4`,
whole 16..64-token tails were byte-identical to the old decode-tail path and faster:

| tail tokens | old decode-tail | 64-bucket tail | speedup | same IDs |
|---:|---:|---:|---:|---|
| 16 | 2.96 s | 2.49 s | 1.19× | yes |
| 32 | 4.84 s | 2.54 s | 1.91× | yes |
| 64 | 9.27 s | 2.56 s | 3.62× | yes |

The more aggressive variants failed correctness: padding a 96-token tail into the 128-row graph gave a
large **4.19×** latency win, but changed greedy output IDs; even "first 64 bucketed, rest decode" still
changed IDs.

Then the default engine showed a decode regression in interactive testing (`vk_engine.py 8` measured
around **9.6-9.8 tok/s** in that session, below the expected ~13 tok/s class). The runtime tail-bucket
refactor was rolled back, and the follow-up audit restored `src/vk_engine.py` to a clean no-diff
baseline. No runtime code from the tail-bucket pass was kept.

The remaining slowdown was system state, not engine code: a detached `src/serve.py` process from the
venv/native-ARM64 launcher was still resident, and Windows was on the default Balanced power scheme.
After stopping the stale server and activating a High performance scheme, decode returned above target:

| test state | command | result |
|---|---|---:|
| clean runtime, Balanced | `src\vk_engine.py 8` | 8 tok in 0.85 s = **9.459 tok/s** |
| clean runtime, High performance | `src\vk_engine.py 8` | 8 tok in 0.54 s = **14.746 tok/s** |
| clean runtime, High performance | `src\vk_engine.py 32` | 32 tok in 2.14 s = **14.977 tok/s** |

## Phase 19 — Server response latency audit

After the engine rollback and power-plan fix, the raw decode benchmark was healthy, but the API server
still had a bad user-visible latency path. I had only probed `/v1/models` after restart, which missed
the full chat path. A bounded first chat request reproduced the complaint:

| server state | request | result |
|---|---|---:|
| pre-fix, first chat after ready | chat, `max_tokens=8`, 22 prompt tokens | **56.224 s**, 2 completion tokens |
| pre-fix, second identical chat | chat, `max_tokens=8`, 22 prompt tokens | **1.580 s**, 2 completion tokens |

Fixes in `src/serve.py`:

- Warm the inference path before printing `ready`, so the first user request does not pay cold-path
  latency.
- Add per-request timing logs (`prompt`, `max`, `completion`, `finish`, elapsed seconds).
- Change the default max generation budget from 16k to `GEMMA4_DEFAULT_MAX_TOKENS` (default **512**) so
  clients that omit `max_tokens` do not accidentally run for minutes.

Post-fix, same running server:

| request | result |
|---|---:|
| first chat after ready, `max_tokens=8`, 22 prompt tokens | **1.625 s**, 2 completion tokens |
| chat with omitted `max_tokens`, 19 prompt tokens | **1.441 s**, 3 completion tokens, max capped at 512 |

---

## Phase 20 — Agent-grade serving: Responses API, sessions, tool calling

The Chat Completions server (Phase 9) was extended toward what real clients/agents need — all text-only,
all on the same `generate()`:

- **Responses API** (`/v1/responses`): OpenAI's `input`/`output` shape, typed-SSE streaming
  (`response.created → output_text.delta → completed`), and the spec's stateful path (`store` +
  `previous_response_id`) via an in-memory conversation store. No native tools/multimodal/structured
  output (text engine).
- **Stateful session WebSocket** (`/v1/sessions`): the server keeps the conversation; the client sends
  only the next message. Multi-turn rides the prefix KV cache. Blocking GPU generation runs in a
  threadpool with deltas streamed over an asyncio queue so the event loop never blocks.
- **OpenAI tool / function calling** (`/v1/chat/completions`) — the lever that makes agent clients
  (opencode) usable. Uses Gemma 4's **native** tool format: the chat template renders
  `<|tool>declaration…`, the model emits `<|tool_call>call:NAME{gemma-args}<tool_call|>` (tokens 48..49),
  and the server parses that span back into OpenAI `tool_calls`, converting Gemma's `key:<|"|>val<|"|>`
  arg syntax to JSON. The round-trip (assistant `tool_calls` + `role:tool` results) renders back through
  the template. `<tool_call|>`(49) had to be added to the stop set or the model rambled past the call.
  Validated end-to-end (`get_weather` → result → final answer) and confirmed live: **opencode**
  (configured `@ai-sdk/openai-compatible`) ran a 6-request tool loop against it.
- **Reasoning-channel strip**: Gemma 4 emits `<|channel>thought<channel|>` blocks; the label text leaked
  into `content` until those spans (tokens 100..101) were dropped in both the tool and normal paths.

opencode uses Chat Completions (not Responses or the WebSocket); the real blocker was tool calling, now
present. Its only remaining cost is model speed on large agent prompts — addressed next.

## Phase 21 — Multi-turn latency, loop escape, and an optional service

- **Prefix cache, take two — the multi-turn fix.** Phase 16's cache had a *2-turn warmup* (it
  snapshotted only `LCP(current, previous)`, so turn 1 cached nothing and turn 2 paid full cost), and
  `CACHE_MAX=2048` was too small for agent prompts. Measured: a same-prefix turn-2 was as slow as cold
  (14.8 s). Fixes: snapshot the **full** MC-aligned prefix every turn (1-turn warmup + a growing
  conversation cache that re-prefills only each turn's *new* tokens), and raise `CACHE_MAX` to 16384
  (`GEMMA4_CACHE_MAX`). Result: same-prefix turn-2 **14.8 → 3.1 s**; a 4181-token prompt's turn-2
  **~57 → 7.2 s (8×)**. For multi-turn/opencode: a cold first turn (~1 min for a big prompt), then
  ~3–7 s/turn — the 3–5 min → ~1 min fix.
- **Degeneration guard.** The int4-12B model can fall into a period-≤4 token loop (runaway `\t\t\t…`,
  repeated `<|image>`/zeros) and spew it to `max_tokens`, wasting tens of seconds for garbage.
  `generate()` now escapes when a short cycle repeats ≥ `GEMMA4_REPEAT_LIMIT` (32) tokens. Validated: a
  FizzBuzz-in-assembly prompt that looped stopped at 112 tokens / 9.9 s instead of running to 400;
  normal prompts unaffected.
- **`PREFILL_I8` under load** crashed natively (no traceback → OOM / GPU-TDR) after a couple of large
  agent requests, where the f16 path was stable through every test + opencode's traffic. f16 kept as the
  default server config; int8 prefill stays an opt-in flag pending a stability fix.
- **Optional service** (`service/`): a Windows PowerShell lifecycle script (start/stop/restart/status/
  logs + opt-in Task-Scheduler auto-start *at logon* — chosen over a session-0 service so the Adreno GPU
  has an interactive session) and a Linux systemd unit. Strictly opt-in; the manual `python src/serve.py`
  run is unchanged. (Restart re-pays the ~4-min model load — caching quantized buffers to disk is the
  open follow-up for fast restarts.)

## Phase 22 — int4-preserving optimization sweep (benchmark-gated, keep-only-if-better)

Surveyed what other inference frameworks add (vLLM, SGLang, TRT-LLM, llama.cpp), filtered to
**single-user + stays-at-int4**, then ran each candidate through a modular loop: establish a baseline,
apply one change, re-measure, **keep only if faster, revert if not**. A reproducible GPU-direct
benchmark (`benchmarks/bench_engine.py`) measures decode tok/s at short/mid/long context (KV-read share
grows with context) + prefill rate.

**Baseline (int4):** decode 14.7 / 12.1 / 10.9 tok/s @ctx 16 / 1024 / 4096; prefill 55.4 prompt-tok/s.
The 14.7→10.9 decode drop with context is the long-context attention cost — the regime KV-side
optimizations target.

- **Wide `f16vec4` K-loads — shipped, the win.** The attention K-dot streams each key's row
  *per-thread* (adjacent threads read different keys → non-coalesced), so it was bandwidth-starved on
  scalar fp16 loads — the same shape as the GEMV the `uvec2` trick fixed (Phase 8). Loading K as
  `f16vec4` (4 fp16/transaction) in `attn.comp` + `pre_attn.comp`: **prefill 55.4 → 62.3 prompt-tok/s
  (+12.5%)**, decode @ctx1024 +4.5%, needle retrieved / logits finite. Same fp16 data, identical math,
  **zero quality risk, no engine change**. The V-accumulate was left scalar — it's already coalesced
  across threads (per key), so wide loads wouldn't help it.
- **Richer samplers — shipped (perf-neutral).** `top_k` / `min_p` / `repetition_penalty` added to
  `sample()` (penalty applies to greedy too) + the OpenAI fields on the server. CPU-side, off the
  `forward()` hot path → benchmark unchanged; `repetition_penalty` also pre-empts the loops the Phase-21
  degeneration guard otherwise hard-escapes.

**Assessed and *not* pursued** (the discipline argues against heavy/marginal changes):

| Candidate | Verdict | Why |
|---|---|---|
| **int8 KV quant** | skip | wide-K already took the cheap K-read win; int8's remaining byte-halving is ~3–5% at long ctx (near thermal noise) for a heavy, bug-prone build (4 shaders + 2 new descriptor layouts + per-slot scale buffers + extend the prefix-cache snapshot to 4 buffers + quality risk) |
| **speculative / lookahead decode** | skip (for now) | the only lever that could give a *big* decode win, but needs a cheap small-M (M≈8) batched-GEMV forward the engine lacks (coopmat pads M<64 → too costly; M=1 GEMV reads weights per token). Major new kernel; verify-cost economics (Phase 17) make payoff uncertain |
| **grammar-constrained decoding** | defer | reliability (valid-JSON tool args), not decode speed — substantial grammar+logit-mask build |
| **multi-entry / radix prefix cache** | defer | helps a multi-prefix workload, not single-stream decode |
| **AWQ / sub-4-bit** | out of scope | AWQ is quality-at-int4 (no speed); sub-4-bit changes the bitrate (not int4-preserving) |

**Takeaway:** at int4, single-user, decode is at the bandwidth wall — the only *free* win left was the
attention K-load coalescing fix (which also lifted prefill). Everything else is either marginal,
non-speed, or needs a major kernel; the benchmark-gated loop kept exactly the change that earned it.

## Phase 23 — Vulkan vs D3D12 on the Adreno: would DirectX be faster? (measured)

"Would DirectX kernels be faster?" The bottleneck is the *silicon* (bandwidth for decode, matrix cores
for prefill), not the API — but instead of asserting, measured it. `benchmarks/bench_backend.py` runs the
**same WGSL kernel through wgpu's Vulkan vs D3D12 backends** (`WGPU_BACKEND_TYPE`), which isolates the
*driver* on the Adreno X2-90 / Windows-ARM64.

| Kernel | Vulkan | D3D12 | read |
|---|---|---|---|
| pure streaming read | ~99 GB/s | ~99 GB/s | **equal at full clocks** (thermal-noisy run-to-run) |
| int4 GEMV (decode pattern) | **~22.4 GB/s** | ~18.6 GB/s | **Vulkan +~20%, stable across runs** |

**Findings:**
- **D3D12 is stable** on this Windows-ARM64 Adreno — it initializes and runs cleanly every time (the
  "is the driver even usable" question: yes).
- **Pure memory bandwidth is a wash** — at full clocks both backends hit the same ~99 GB/s ceiling. That
  *confirms* the core claim: the bandwidth wall is the silicon; the API can't beat it. (An early run
  showed a false 55-vs-45 gap — pure thermal ramp / contention, gone on re-run.)
- **But the real int4-GEMV kernel runs ~20% faster on Vulkan** (22.4 vs 18.6 GB/s, consistent). Decode
  isn't *pure* bandwidth — it interleaves int4-unpack + scale + accumulate, and **Qualcomm's Adreno
  driver generates better code for that on the Vulkan path** (it's their Vulkan-native lineage from
  Android). So a D3D12 port would *regress* decode, not help.

**Caveats (honest):** measured through wgpu (fair — same wrapper both sides — but absolute GB/s sit below
our raw-Vulkan engine's tuned `uvec2` GEMV); the streaming-read number is thermal-noisy so the GEMV is the
reliable signal; and wgpu/WebGPU can't expose cooperative matrix, so the **prefill / matrix-core path is
untested** here — but a Vulkan-native driver winning the GEMV makes it very unlikely D3D12 would *gain* on
prefill (where it would more likely lose the coopmat 2× entirely).

**Verdict:** Vulkan is the right call on this hardware — stable parity on raw bandwidth, **~20% ahead on
the kernel that counts**, untestable-but-likely-better on the matrix cores, and portable to Linux on top.
DirectX would be a rewrite for a regression.

---

## Final standings

| Path | Correct? | Decode tok/s |
|---|---|---|
| Google `litert-lm` | — | won't load |
| ONNX/QNN → NPU (HTP) | ❌ (fp16 overflow + error 1003) | — |
| ONNX/QNN → GPU (QnnGpu) | ✅ | ~0.024 (swap‑bound) |
| llama.cpp Vulkan | ✅ | 6.06 |
| custom wgpu shader engine | ✅ | 7.07 |
| llama.cpp CPU | ✅ | 7.9 |
| custom raw‑Vulkan engine (scalar GEMV) | ✅ | ~10.7 (steady 8‑tok) |
| **custom raw‑Vulkan engine (uvec2 wide‑load GEMV)** | ✅ | **13.84** |

## Key technical findings

- **Gemma fp16 ≠ HTP‑safe:** bf16‑range activations overflow native fp16 (RMSNorm square, attention
  scores). Fixable with dynamic per‑row‑max norm + stable/scaled attention — but ORT‑QNN's HTP
  execution (error 1003) is the real blocker; needs native QAIRT.
- **GEMV is occupancy‑bound, not ALU‑bound** on the Adreno for small‑N matmuls; adaptive K‑split is
  the single biggest kernel lever (~2× on small N).
- **Caching doesn't help a streaming GEMV** — no weight reuse to exploit; the one reused operand
  (x) is already L2‑resident, and forcing it to shared memory *loses* (−50%, barrier kills overlap).
- **The dispatch/submit overhead is a WebGPU limit, not the GPU's.** Command‑buffer record‑once via
  raw Vulkan removed ~100 ms/token and was worth more than every kernel micro‑opt combined.
- **Unified memory matters:** host‑visible coherent buffers (no staging) make the per‑token data
  updates free; the model lives in the shared 48 GB.

## Artifacts (`Desktop/gemma4-litert`)

| Path | What |
|---|---|
| `vk/*.comp`, `vk/*.spv` | 8 Adreno‑tuned compute kernels (GLSL → SPIR‑V) |
| `scripts/vk_engine.py` | **the ~13.6 tok/s raw‑Vulkan engine** (record‑once command buffer); importable: exposes `forward`/`generate`/`sample`/`tok` |
| `scripts/serve.py` | **OpenAI‑compatible FastAPI server** wrapping the engine (chat/completions, streaming, `/v1/models`, usage) |
| `scripts/vk_gemv.py` | validated raw‑Vulkan foundation harness |
| `scripts/engine_gpu_full.py` | the 7.07 tok/s wgpu engine (WebGPU‑bound) |
| `scripts/gpu_int4.py`, `gpu_ops.py`, `gpu_attn.py` | wgpu kernels (Adreno‑tuned GEMV + ops) |
| `scripts/microbench_gemv.py`, `microbench_xcache.py` | the kernel decomposition + cache microbenchmarks |
| `scripts/microbench_bw.py` | pure GPU streaming‑read bandwidth (the ~101–123 GB/s ceiling) |
| `scripts/microbench_total.py` | two‑pass total‑time GEMV split sweep (corrected split heuristic) |
| `scripts/microbench_attn.py` | attention current‑vs‑parallel‑softmax microbench (1.6× at long ctx) |
| `scripts/export_shards.py`, `engine_gemma.py` | the custom `gemma4_unified` ONNX exporter + NPU/GPU shard engine |
| `scripts/patch_rmsnorm.py`, `patch_attention.py` | the fp16‑overflow‑safe reformulations |
| `gguf/` | unsloth GGUF for the llama.cpp baseline |

## Run it

```powershell
# custom raw-Vulkan engine (~13.8 tok/s decode):
.venv-gemma4\Scripts\python.exe src\vk_engine.py 8

# OpenAI-compatible server (transparent prefix KV cache; auto on repeated prompts):
.venv-gemma4\Scripts\python.exe src\serve.py --host 127.0.0.1 --port 8000

# opt-in: fp8 decode scales (+4.5-8% decode) / int8 prefill GEMMs (~2x prefill, +12GB RAM):
$env:GEMV_FP8=1;   .venv-gemma4\Scripts\python.exe src\vk_engine.py 8
$env:PREFILL_I8=1; .venv-gemma4\Scripts\python.exe src\serve.py

# per-kernel GPU profilers: PROFILE=1 (decode), PREPROF=1 (batched prefill)
```
