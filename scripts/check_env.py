#!/usr/bin/env python
"""Diagnose the gemma4-litert environment. Run after setup to confirm everything's in place:

  python scripts/check_env.py

Checks: native ARM64 Python, core deps, a usable Vulkan device, compiled shaders, and the weights.
Exits non-zero if any hard requirement fails (warnings don't fail).
"""
import os
import sys
import platform
import importlib.util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OK, WARN, FAIL = "  [ ok ]", "  [warn]", "  [FAIL]"
fails = 0


def check(label, cond, detail="", warn_only=False):
    global fails
    tag = OK if cond else (WARN if warn_only else FAIL)
    if not cond and not warn_only:
        fails += 1
    print(f"{tag} {label}" + (f"  -- {detail}" if (detail and not cond) else ""))   # hint only on miss
    return cond


print("=== gemma4-litert environment check ===\n")

# 1. Python: native ARM64 (the x64-emulated default cannot reach the Adreno GPU)
mach = platform.machine()
check(f"Python {platform.python_version()} ({mach})", mach.upper() in ("ARM64", "AARCH64"),
      "need NATIVE ARM64 Python 3.12; x64-emulated cannot reach the GPU", warn_only=True)

# 2. core dependencies
for mod in ("numpy", "vulkan", "torch", "transformers", "fastapi", "uvicorn"):
    check(f"import {mod}", importlib.util.find_spec(mod) is not None,
          "pip install -r requirements.txt" if importlib.util.find_spec(mod) is None else "")
for mod in ("huggingface_hub", "websockets"):
    check(f"import {mod}", importlib.util.find_spec(mod) is not None,
          "needed for model download / websocket API", warn_only=True)

# 3. a usable Vulkan compute device
try:
    import vulkan as vk
    inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=vk.VkApplicationInfo(
        apiVersion=(1 << 22) | (1 << 12))), None)
    devs = vk.vkEnumeratePhysicalDevices(inst)
    names = [vk.vkGetPhysicalDeviceProperties(d).deviceName for d in devs]
    check(f"Vulkan device(s): {', '.join(names) or 'none'}", len(devs) > 0,
          "no Vulkan device -- check the GPU driver / Vulkan loader")
except Exception as e:
    check("Vulkan device enumeration", False, str(e)[:120])

# 4. compiled shaders
spv = [f for f in os.listdir(os.path.join(ROOT, "vk")) if f.endswith(".spv")] if os.path.isdir(os.path.join(ROOT, "vk")) else []
check(f"compiled shaders (vk/*.spv): {len(spv)}", len(spv) >= 15, "run scripts/setup or bash vk/build.sh")

# 5. model weights
mdir = os.path.join(ROOT, "models", "gemma-4-12B-it")
has_cfg = os.path.isfile(os.path.join(mdir, "config.json"))
has_w = os.path.isdir(mdir) and any(f.endswith(".safetensors") for f in (os.listdir(mdir) if os.path.isdir(mdir) else []))
check("model weights (models/gemma-4-12B-it)", has_cfg and has_w,
      "run: python scripts/download_model.py")

print()
if fails:
    print(f"{fails} hard requirement(s) failed -- see [FAIL] above.")
    sys.exit(1)
print("All hard requirements met. Start the server:  .venv-gemma4/Scripts/python.exe src/serve.py")
