<#
  setup.ps1 -- set up the gemma4-litert environment on Windows (ARM64).

  .\scripts\setup.ps1                 # venv + deps + (re)build shaders
  .\scripts\setup.ps1 -Model          # ... and download the weights (~24 GB)
  .\scripts\setup.ps1 -NoShaders       # skip shader rebuild (committed .spv are used)

  Requires: native ARM64 Python 3.12 on PATH, and (to rebuild shaders) the Vulkan SDK's
  glslangValidator. The x64-emulated default Python CANNOT reach the Adreno GPU.
#>
param([switch]$Model, [switch]$NoShaders)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host "=== gemma4-litert setup ($Root) ===" -ForegroundColor Cyan

# 1. Python arch check
$arch = & python -c "import platform;print(platform.machine())"
if ($arch.ToUpper() -notin @("ARM64","AARCH64")) {
  Write-Warning "Python arch is '$arch'. You need NATIVE ARM64 Python 3.12 to reach the Adreno GPU; the x64-emulated default will not work. Continuing anyway."
} else { Write-Host "[ok] native ARM64 Python ($arch)" }

# 2. venv
$Venv = Join-Path $Root ".venv-gemma4"
$VPy  = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VPy)) { Write-Host "creating venv .venv-gemma4..."; & python -m venv $Venv }
else { Write-Host "[ok] venv exists" }

# 3. dependencies
Write-Host "installing dependencies..."
& $VPy -m pip install --upgrade pip | Out-Null
& $VPy -m pip install -r (Join-Path $Root "requirements.txt")

# 4. shaders (build artifact: vk/*.spv is gitignored, compiled from vk/*.comp)
if (-not $NoShaders) {
  $glslang = (Get-Command glslangValidator -ErrorAction SilentlyContinue).Source
  if (-not $glslang) {
    $haveSpv = @(Get-ChildItem (Join-Path $Root "vk\*.spv") -ErrorAction SilentlyContinue).Count
    if ($haveSpv -gt 0) { Write-Warning "glslangValidator not found; using the $haveSpv existing vk/*.spv." }
    else { throw "glslangValidator not found and no vk/*.spv present. Install the Vulkan SDK (provides glslangValidator) and re-run -- the .spv are build artifacts, not committed." }
  } else {
    Write-Host "compiling shaders with $glslang..."
    Get-ChildItem (Join-Path $Root "vk\*.comp") | ForEach-Object {
      $out = $_.FullName -replace '\.comp$', '.spv'
      & $glslang -V --target-env spirv1.6 $_.FullName -o $out | Out-Null
    }
    Write-Host "[ok] shaders compiled"
  }
}

# 5. model weights (optional)
if ($Model) {
  Write-Host "downloading weights (~24 GB, resumable)..."
  & $VPy (Join-Path $Root "scripts\download_model.py")
}

# 6. verify
Write-Host "`n=== verifying ===" -ForegroundColor Cyan
& $VPy (Join-Path $Root "scripts\check_env.py")

Write-Host "`nNext:" -ForegroundColor Cyan
if (-not $Model) { Write-Host "  .\scripts\setup.ps1 -Model            # download the weights" }
Write-Host "  $VPy src\serve.py     # start the OpenAI-compatible server"
Write-Host "  .\service\gemma4-service.ps1 start    # or run it as a background service"
