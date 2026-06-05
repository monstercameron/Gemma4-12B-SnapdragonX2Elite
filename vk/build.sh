#!/usr/bin/env bash
# Compile all GLSL compute shaders -> SPIR-V. Requires the Vulkan SDK's glslangValidator.
# Override the binary with GLSLANG=/path/to/glslangValidator if it's not on PATH.
set -e
GLSL="${GLSLANG:-glslangValidator}"
cd "$(dirname "$0")"
for f in *.comp; do
  "$GLSL" -V --target-env spirv1.6 "$f" -o "${f%.comp}.spv" >/dev/null
  echo "  $f -> ${f%.comp}.spv"
done
echo "done ($(ls *.spv | wc -l) shaders)."
