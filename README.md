# gemma4-adreno

**Gemma 4 12B running on a Snapdragon X2 Elite's Adreno X2-90 GPU — a from-scratch Vulkan compute engine, no vendor ML runtime.**

Every matmul, attention kernel, RMSNorm, and RoPE is a hand-written GLSL compute shader. The model
runs entirely on the integrated GPU at int4, serves an OpenAI-compatible API, and supports up to 64k
context — built without ONNX Runtime, llama.cpp, or Qualcomm's QNN/litert tooling.

> The full build narrative — every dead end, microbenchmark, and reversal — is in **[JOURNEY.md](JOURNEY.md)**.

## Results (Snapdragon X2 Elite Extreme, Adreno X2-90, Windows 11 ARM64)

| | throughput | bound by |
|---|---|---|
| **Decode** (generation) | **~13 tok/s** | memory bandwidth (~106 of ~123 GB/s achievable) |
| **Prefill** (prompt processing) | **~61 prompt-tok/s** (f16 coopmat) · **~2× with `PREFILL_I8=1`** (int8 coopmat) | compute (on the GPU's cooperative-matrix cores) |
| Context | up to **64k** tokens | sliding-window ring buffer + flash attention |

How it compares on the same machine (decode tok/s): this engine **~13** · llama.cpp CPU 7.9 ·
custom wgpu 7.07 · llama.cpp Vulkan 6.06 · the ONNX/QNN starting point 0.024.

## What's interesting here

- **int4 GEMV tuned to the Adreno** — `uvec2` (64-bit) coalesced weight loads + adaptive K-split for
  occupancy + a record-once Vulkan command buffer (one submit per token, no per-token re-encode).
- **64k context** via flash-decode attention (online softmax, unbounded T) and a sliding-window ring
  buffer for the local-attention layers — fits in 48 GB unified memory.
- **Batched prefill on the matrix cores** — `VK_KHR_cooperative_matrix` fp16 GEMM with an in-kernel
  int4→fp16 dequant, ~4.7× faster prompt processing than per-token prefill.
- **OpenAI-compatible server** (`/v1/chat/completions` streaming + non-streaming, validated against the
  official `openai` Python client).
- **A real GPU profiler** built in (`PROFILE=1`, Vulkan timestamp queries) — per-kernel timing that
  shows decode is 94.5% GEMV weight reads, i.e. bandwidth-bound at the hardware ceiling.

## Repository layout

```
src/            the engine + server
  vk_engine.py    the raw-Vulkan int4 engine (decode + batched prefill + 64k + profiler)
  serve.py        OpenAI-compatible FastAPI server
  test_longctx.py long-context needle-in-haystack test
vk/             GLSL compute shaders (.comp) + compiled SPIR-V (.spv) + build.sh
benchmarks/     standalone microbenchmarks (kernel decomposition, coopmat, bandwidth, ...)
experiments/    research path: the earlier ONNX/QNN shard pipeline + the WebGPU (wgpu) engine
JOURNEY.md      the full development log
```

## Setup

Requires **native ARM64 Python 3.12** (the x64-emulated default Python cannot reach the NPU/GPU),
the **Vulkan SDK** (for `glslangValidator`), and Gemma 4 12B weights placed in `models/gemma-4-12B-it`.

```powershell
# native ARM64 python
python -m venv .venv-gemma4
.venv-gemma4\Scripts\python.exe -m pip install -r requirements.txt

# compile the shaders (needs the Vulkan SDK on PATH)
bash vk/build.sh
```

## Usage

```powershell
# one-shot decode demo
.venv-gemma4\Scripts\python.exe src\vk_engine.py 8

# OpenAI-compatible server -> http://127.0.0.1:8000/v1
.venv-gemma4\Scripts\python.exe src\serve.py --host 127.0.0.1 --port 8000

# GPU per-kernel profile of a decode token graph
$env:PROFILE=1; .venv-gemma4\Scripts\python.exe src\vk_engine.py 8

# opt-in fp8 (e4m3) decode scales: +4.5-8% decode for a ~2%/scale quantization cost
$env:GEMV_FP8=1; .venv-gemma4\Scripts\python.exe src\vk_engine.py 8

# opt-in W8A8 int8 prefill GEMMs: ~2x prefill, at the cost of ~12GB extra weight RAM
$env:PREFILL_I8=1; .venv-gemma4\Scripts\python.exe src\serve.py

# a microbenchmark
.venv-gemma4\Scripts\python.exe benchmarks\microbench_gemv.py
```

`GEMV_FP8=1` stores the int4 per-block scales as fp8 e4m3 (1 byte vs fp16's 2),
halving the ~11% of decode bandwidth they cost. Scales are normalized per-tensor
into e4m3's normal range so quality holds (needle retrieval intact, decoded text
unchanged); default is fp16 for zero quality risk. Prefill always keeps fp16
scales — it's compute-bound, so fp8 would not help it.

`PREFILL_I8=1` runs the prefill GEMMs through the Adreno's int8 cooperative-matrix
engine (3.7× the f16 path here) instead of fp16: precomputed int8 weights (per-
column scale, the int4 per-block scale folded in at load) × per-row int8-quantized
activations → s8×s8→s32 → rescale. Measured **~2.0× prefill** (105 → 210 prompt-tok/s
GPU-only) at identical quality (microbench cos 0.99993 vs the f16 path; needle intact).
Cost: ~12GB extra weight RAM (int8 stored alongside int4 — decode keeps int4 for its
bandwidth). Default off; decode is unaffected. Validate the speed/quality yourself with
`benchmarks\coopgemm_w4a8.py` (no model load).

Point any OpenAI client at the server:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
print(c.chat.completions.create(model="gemma-4-12b-it",
    messages=[{"role": "user", "content": "What is the capital of France?"}]).choices[0].message.content)
```

## License

Engine code: **MIT** (see [LICENSE](LICENSE)). The Gemma 4 weights are **not** included and are
governed by [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).
