#!/usr/bin/env bash
# setup.sh -- set up the gemma4-litert environment on Linux/macOS.
#
#   ./scripts/setup.sh                # venv + deps + (re)build shaders
#   ./scripts/setup.sh --model        # ... and download the weights (~24 GB)
#   ./scripts/setup.sh --no-shaders   # skip shader rebuild (committed .spv are used)
#
# NOTE: the reference target is Windows ARM64 + Adreno. On Linux you need a working Vulkan driver
# for your GPU; glslangValidator (Vulkan SDK) is only needed to rebuild shaders.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL=0; SHADERS=1
for a in "$@"; do case "$a" in --model) MODEL=1;; --no-shaders) SHADERS=0;; esac; done

echo "=== gemma4-litert setup ($ROOT) ==="
PY="${PYTHON:-python3}"
"$PY" --version

# venv
if [ ! -x ".venv-gemma4/bin/python" ]; then echo "creating venv .venv-gemma4..."; "$PY" -m venv .venv-gemma4; fi
VPY=".venv-gemma4/bin/python"

# deps
echo "installing dependencies..."
"$VPY" -m pip install --upgrade pip >/dev/null
"$VPY" -m pip install -r requirements.txt

# shaders (optional; .spv are committed)
if [ "$SHADERS" = 1 ]; then
  if command -v glslangValidator >/dev/null 2>&1; then
    echo "compiling shaders..."; bash vk/build.sh
  else
    echo "WARN: glslangValidator not found (Vulkan SDK). Using committed vk/*.spv."
  fi
fi

# model (optional)
if [ "$MODEL" = 1 ]; then echo "downloading weights (~24 GB, resumable)..."; "$VPY" scripts/download_model.py; fi

# verify
echo; echo "=== verifying ==="
"$VPY" scripts/check_env.py || true

echo; echo "Next:"
[ "$MODEL" = 0 ] && echo "  ./scripts/setup.sh --model      # download the weights"
echo "  $VPY src/serve.py     # start the OpenAI-compatible server"
