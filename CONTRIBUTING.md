# Contributing

Thanks for your interest. This is a from-scratch raw-Vulkan inference engine for Gemma 4 12B on the
Snapdragon X2 Adreno GPU — so most work needs that hardware to test for real.

## Setup

```powershell
.\scripts\setup.ps1 -Model        # Windows ARM64;  Linux: ./scripts/setup.sh --model
.venv-gemma4\Scripts\python.exe scripts\check_env.py
```

## How this codebase works (the philosophy)

**Measure before optimizing, and measure after.** Almost every change in this engine was decided by a
microbenchmark or a profile, and several "obvious" optimizations were *measured and rejected* (see
`JOURNEY.md`). Please follow the same bar:

- Add or reuse a microbenchmark in `benchmarks/` to justify a kernel change (most run with **no model
  load**, on synthetic data — e.g. `benchmarks/coopgemm_w4a8.py`).
- Validate correctness end-to-end with `src/test_longctx.py` (needle-in-haystack) and, for the prefix
  cache, `src/test_prefix_cache.py`.
- Profile with `PROFILE=1` (decode) or `PREPROF=1` (prefill) to confirm where time actually goes.
- Report the *real* numbers, including regressions. An honest "this didn't help" is valuable.

## Shaders

GLSL compute shaders live in `vk/*.comp`; recompile to SPIR-V after editing:

```bash
bash vk/build.sh          # needs the Vulkan SDK's glslangValidator
```

The `.spv` are **build artifacts** (gitignored) — commit only the `.comp`; everyone rebuilds via
`scripts/setup` or `vk/build.sh`.

## Before opening a PR

- `python -m compileall src scripts service benchmarks` passes (CI runs this).
- New kernels are correctness-validated (cos vs reference, or the needle test).
- Keep changes additive where possible — the decode path is bandwidth-tuned and fragile.
