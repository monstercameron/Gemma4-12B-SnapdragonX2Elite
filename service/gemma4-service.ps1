<#
  gemma4-service.ps1 -- manage the Gemma 4 Vulkan server as a background service on Windows (ARM64).

  Usage:
    .\service\gemma4-service.ps1 <command>

  Commands:
    start            start the server in the background (no-op if already running)
    stop             stop the server
    restart          stop then start (model reload takes ~4 min -- this is one command, not instant)
    status           show running state + /health
    logs             tail the server log (Ctrl+C to exit)
    install          register a Scheduled Task to auto-start at logon (GPU needs an interactive session)
    uninstall        remove the auto-start task
    help             this text

  Config (environment, read at start): GEMMA4_HOST, GEMMA4_PORT, PREFILL_I8, GEMV_FP8,
    GEMMA4_DEFAULT_MAX_TOKENS, GEMMA4_CACHE_MAX, GEMMA4_REPEAT_LIMIT.
  Or drop a `service\gemma4.env` file with KEY=VALUE lines.
#>
param([Parameter(Position=0)][string]$Command = "help")

$ErrorActionPreference = "Stop"
$Root    = Split-Path -Parent $PSScriptRoot                  # repo root (parent of service\)
$Py      = Join-Path $Root ".venv-gemma4\Scripts\python.exe"
$Serve   = Join-Path $Root "src\serve.py"
$OutDir  = Join-Path $Root "out"
$Log     = Join-Path $OutDir "service.log"
$PidFile = Join-Path $OutDir "service.pid"
$TaskName = "Gemma4Server"

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

function Load-EnvFile {
  $envFile = Join-Path $PSScriptRoot "gemma4.env"
  if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
      if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$' -and $_ -notmatch '^\s*#') {
        Set-Item -Path ("env:" + $Matches[1]) -Value $Matches[2]
      }
    }
  }
}

function Get-ServerPid {
  if (Test-Path $PidFile) {
    $procId = (Get-Content $PidFile | Select-Object -First 1).Trim()
    if ($procId -and (Get-Process -Id $procId -ErrorAction SilentlyContinue)) { return [int]$procId }
  }
  return $null
}

function Server-Host { if ($env:GEMMA4_HOST) { $env:GEMMA4_HOST } else { "127.0.0.1" } }
function Server-Port { if ($env:GEMMA4_PORT) { $env:GEMMA4_PORT } else { "8000" } }

function Do-Start {
  Load-EnvFile
  $existing = Get-ServerPid
  if ($existing) { Write-Host "already running (pid $existing)"; return }
  if (-not (Test-Path $Py))    { throw "python not found: $Py (create the venv first)" }
  if (-not (Test-Path $Serve)) { throw "serve.py not found: $Serve" }
  $h = Server-Host; $p = Server-Port
  Write-Host "starting Gemma 4 server on http://${h}:${p} (model load ~4 min)..."
  # cmd wrapper so stdout+stderr land in one combined log; -PassThru gives us the pid
  $args = "/c `"`"$Py`" `"$Serve`" --host $h --port $p > `"$Log`" 2>&1`""
  $proc = Start-Process -FilePath "cmd.exe" -ArgumentList $args -WindowStyle Hidden -PassThru
  # the pid we want is the python child, not cmd -- find it after a moment
  Start-Sleep -Seconds 2
  $child = Get-CimInstance Win32_Process -Filter "ParentProcessId = $($proc.Id)" -ErrorAction SilentlyContinue | Select-Object -First 1
  $serverPid = if ($child) { $child.ProcessId } else { $proc.Id }
  Set-Content -Path $PidFile -Value $serverPid
  Write-Host "started (pid $serverPid). Log: $Log"
  Write-Host "watch readiness:  .\service\gemma4-service.ps1 status"
}

function Do-Stop {
  $procId = Get-ServerPid
  if (-not $procId) {
    # fallback: kill any python running our serve.py
    $procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -and $_.CommandLine -match 'serve\.py' }
    if ($procs) { $procs | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } ; Write-Host "stopped (by command-line match)" }
    else { Write-Host "not running" }
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
    return
  }
  Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
  if (Test-Path $PidFile) { Remove-Item $PidFile -Force }
  Write-Host "stopped (pid $procId)"
}

function Do-Status {
  $procId = Get-ServerPid
  if ($procId) { Write-Host "process: RUNNING (pid $procId)" } else { Write-Host "process: not running" }
  $h = Server-Host; $p = Server-Port
  try {
    $r = Invoke-WebRequest -UseBasicParsing -Uri "http://${h}:${p}/health" -TimeoutSec 5
    Write-Host "health : $($r.Content)"
  } catch {
    Write-Host "health : not responding yet (still loading, or stopped)"
  }
}

function Do-Logs {
  if (-not (Test-Path $Log)) { Write-Host "no log yet: $Log"; return }
  Get-Content -Path $Log -Tail 40 -Wait
}

function Do-Install {
  $psExe = (Get-Command powershell.exe).Source
  $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" start"
  $action  = New-ScheduledTaskAction -Execute $psExe -Argument $arg -WorkingDirectory $Root
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  # interactive session so the Adreno GPU / Vulkan is reachable; highest privileges
  $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Highest
  $settings  = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
  Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
  Write-Host "installed scheduled task '$TaskName' (auto-start at logon for $env:USERNAME)."
  Write-Host "start it now with:  .\service\gemma4-service.ps1 start"
}

function Do-Uninstall {
  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "removed scheduled task '$TaskName'."
  } else { Write-Host "no scheduled task '$TaskName'." }
}

switch ($Command.ToLower()) {
  "start"     { Do-Start }
  "stop"      { Do-Stop }
  "restart"   { Do-Stop; Start-Sleep -Seconds 2; Do-Start }
  "status"    { Do-Status }
  "logs"      { Do-Logs }
  "install"   { Do-Install }
  "uninstall" { Do-Uninstall }
  default     { Get-Content $PSCommandPath | Select-Object -First 30 | ForEach-Object { $_ -replace '^<#|#>','' } }
}
