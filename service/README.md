# Running the server as a service (optional)

These are **opt-in** helpers — they don't change anything about the engine or the normal way of
running the server (`python src\serve.py` still works exactly as before). Use them only if you want
auto-start on boot/login and one-command stop/start/restart.

> The model load is ~4 minutes (weights + int4 quantization + Vulkan buffer upload), so "restart" is
> a single command, not an instant reload. There's no way around the load cost short of caching the
> quantized buffers to disk (a possible future feature).

## Windows (the reference machine — ARM64 + Adreno GPU)

A PowerShell management script + Task Scheduler auto-start. Task Scheduler (run **at logon**, in the
interactive session) is used deliberately instead of a true session-0 Windows service, because the
Adreno GPU / Vulkan compute needs an interactive desktop session to initialize.

```powershell
# one-command lifecycle (no auto-start needed):
.\service\gemma4-service.ps1 start      # launch in background
.\service\gemma4-service.ps1 status     # process state + /health
.\service\gemma4-service.ps1 logs       # tail the log
.\service\gemma4-service.ps1 restart    # stop + start
.\service\gemma4-service.ps1 stop

# optional: auto-start at logon (and remove it)
.\service\gemma4-service.ps1 install
.\service\gemma4-service.ps1 uninstall
```

**Config** is read from the environment at start (or a `service\gemma4.env` file with `KEY=VALUE`
lines): `GEMMA4_HOST`, `GEMMA4_PORT`, `PREFILL_I8`, `GEMV_FP8`, `GEMMA4_DEFAULT_MAX_TOKENS`,
`GEMMA4_CACHE_MAX`, `GEMMA4_REPEAT_LIMIT`. Example `service\gemma4.env`:

```
GEMMA4_PORT=8000
GEMMA4_DEFAULT_MAX_TOKENS=1024
# PREFILL_I8=1     # ~2x prefill, +12GB RAM (crashed under heavy load in testing -- enable with care)
```

## Linux (generic deployment)

A systemd unit — see `service/gemma4.service` (it documents user-service install, recommended so the
GPU has a logged-in seat). Edit the paths/env, then:

```bash
cp service/gemma4.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gemma4     # auto-start at login + start
systemctl --user restart gemma4          # one-command restart
systemctl --user status gemma4 ; journalctl --user -u gemma4 -f
```
