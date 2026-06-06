"""OpenAI-compatible API server for the raw-Vulkan Gemma 4 12B engine (Adreno X2 GPU).

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions
(streaming + non-streaming). The engine holds one global GPU command buffer + KV cache, so all
generation is serialized behind a single lock -- correct for a local single-GPU server.

Run:  .venv-gemma4/Scripts/python.exe src/serve.py [--host H] [--port P]
Import of vk_engine loads the model + records the command buffer (~minutes) at startup.
"""
import os, sys, time, json, uuid, threading, argparse, asyncio, re, platform
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Resilience: bound the engine's GPU fence wait (default infinite). A real Adreno wedge then RAISES with
# the operation label instead of freezing the server forever. Must be set BEFORE importing vk_engine
# (the engine reads it at import). 60s is far above any legit single op (<~1s), so no false positives.
os.environ.setdefault("GEMMA4_FENCE_TIMEOUT_MS", "60000")
import numpy as np
from typing import Optional, Union, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import vk_engine as E   # heavy: loads weights + records the Vulkan command buffer on import

MODEL_ID = "gemma-4-12b-it"
DEFAULT_MAX_TOKENS = int(os.environ.get("GEMMA4_DEFAULT_MAX_TOKENS", "512"))
# Reasoning channel (enable_thinking). ON, the model reasons before responding -> better tool selection
# (e.g. realizing `bash` can mkdir). It spends tokens thinking first, so the engine force-closes the
# thought channel after THINK_BUDGET tokens -- caps the runaway case (model rambling toward max_tokens,
# e.g. minutes when a client sends a big max_tokens). NOTE: typical thinking is < budget, so the budget
# bounds the worst case but not the per-turn overhead (~seconds at ~13 tok/s). GEMMA4_THINK=0 to disable.
THINK = os.environ.get("GEMMA4_THINK", "1") != "0"
THINK_BUDGET = int(os.environ.get("GEMMA4_THINK_BUDGET", "1000"))   # max reasoning tokens before forced answer
# Tool calling is structured output: at high temperature the model samples AWAY from the native
# <|tool_call> DSL into a markdown code block (`tool_calls` empties out -> opencode can't act, "I can't
# create a folder"). Cap the temperature for tool requests to near-greedy. 0.0 = force greedy (most
# reliable for agents); raise GEMMA4_TOOL_TEMP if you want sampled tool args.
TOOL_TEMP = float(os.environ.get("GEMMA4_TOOL_TEMP", "0.0"))
# Coerce shell tools to PowerShell on Windows. opencode names its shell tool `bash` with a bash
# description even here, so the model emits bash-isms (ls, rm -rf, mkdir -p, &&). We rewrite the shell
# tool's description (NOT its name -- the name must round-trip back to opencode) to force PowerShell.
COERCE_PS = os.environ.get("GEMMA4_COERCE_PS", "1") != "0" and platform.system() == "Windows"
OUTPUT_CAP = int(os.environ.get("GEMMA4_MAX_OUTPUT", "4096"))   # hard ceiling on tokens per single response

# GPU wedge latch. When the engine raises a fence timeout (a real Adreno wedge), we set this: the
# in-flight request fails cleanly (the _gen_lock/GPU context auto-releases on the exception), and every
# later request gets an immediate 503 instead of each blocking 60s behind a dead device. A wedged Vulkan
# device cannot be reset in-process, so recovery = restart; the 503 says so.
_GPU_WEDGED = threading.Event()
def _is_fence_timeout(exc): return isinstance(exc, RuntimeError) and "fence TIMEOUT" in str(exc)
def _mark_wedged(exc): _GPU_WEDGED.set(); print(f"[serve] GPU WEDGED: {exc} -- 503 until restart", flush=True)
def _guard_wedged():
    if _GPU_WEDGED.is_set():
        raise HTTPException(status_code=503, detail="GPU wedged (fence timeout); server must be restarted")

# ---- DEBUG LOGGING (GEMMA4_DEBUG=1) -- remove this block + its call sites to revert ----
# Per inference, append one JSON line to out/debug.jsonl with the request, the rendered prompt tail,
# the RAW generation decoded WITH special tokens (so the <|channel>thought<channel|> reasoning span and
# <|tool_call> DSL are visible), the thinking/visible split, and the parsed tool calls. This is the
# artifact for "responses aren't coming through" (visible content empty -> see if it's trapped in the
# reasoning channel) and "thinking is bugged" (see exactly what/how much the model thought).
DEBUG = os.environ.get("GEMMA4_DEBUG", "0") == "1"
DEBUG_FILE = os.environ.get("GEMMA4_DEBUG_FILE",
                            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "debug.jsonl"))

# stop tokens: the model's own generation_config.eos_token_id is authoritative.
# For this Gemma 4 build that's [1=<eos>, 106=<turn|>, 50] -- the turn delimiter is <turn|>, NOT
# <end_of_turn>, so we must read it from the config rather than guess token names.
from transformers import GenerationConfig
STOP_IDS = set()
try:
    _e = GenerationConfig.from_pretrained(E.MODEL).eos_token_id
    if _e is not None:
        STOP_IDS.update(int(x) for x in (_e if isinstance(_e, (list, tuple)) else [_e]))
except Exception as _ex:
    print("[serve] gen-config eos lookup failed:", _ex, flush=True)
if E.tok.eos_token_id is not None:
    STOP_IDS.add(int(E.tok.eos_token_id))

GPU = threading.Lock()
app = FastAPI(title="gemma4-litert", description="Gemma 4 12B on Snapdragon X2 Adreno (raw Vulkan)")


# ---------- schemas ----------
class Msg(BaseModel):
    role: str
    content: Optional[Union[str, list]] = None
    tool_calls: Optional[list] = None      # assistant tool calls (round-trip back into the prompt)
    tool_call_id: Optional[str] = None     # for role="tool" results
    name: Optional[str] = None

class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: List[Msg]
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0                         # 0 = off
    min_p: float = 0.0                     # 0 = off
    repetition_penalty: float = 1.0        # 1.0 = off
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[dict]] = None     # OpenAI function tools -> Gemma native tool declarations
    tool_choice: Optional[Union[str, dict]] = None

class CompReq(BaseModel):
    model: str = MODEL_ID
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


# ---------- helpers ----------
def _now() -> int: return int(time.time())
def _uid(p: str) -> str: return f"{p}-{uuid.uuid4().hex[:24]}"

def _stops(stop) -> List[str]:
    if stop is None: return []
    return [stop] if isinstance(stop, str) else list(stop)

def _fit(prompt_len: int, req_max: Optional[int]) -> int:
    # Hard cap on output tokens: opencode sends max_tokens=32000, so one big turn (e.g. "generate full
    # CRUD") can try to emit thousands of tokens in a single ~5 tok/s stream -- minutes of held GPU lock
    # and elevated risk of an Adreno driver hang. OUTPUT_CAP bounds any single response. Raise via
    # GEMMA4_MAX_OUTPUT if you need longer single completions.
    return max(1, min(req_max or DEFAULT_MAX_TOKENS, OUTPUT_CAP, E.MAXT - prompt_len))

def _log_request(kind: str, prompt_n: int, max_new: int, comp_n: int, finish: str, t0: float):
    dt = max(time.time() - t0, 1e-9)
    rate = comp_n / dt
    print(f"[serve] {kind} prompt={prompt_n} max={max_new} completion={comp_n} "
          f"finish={finish} elapsed={dt:.3f}s out_tok_s={rate:.2f}", flush=True)

# Prefix KV cache: the engine snapshots a shared prompt prefix's KV so repeated prompts skip
# re-prefilling it. We auto-detect the prefix as the chunk-aligned longest-common-prefix of consecutive
# requests (typically the system prompt). Reuse is allowed only at the EXACT cached length so the
# sliding-window ring is clean. A lock serializes generations (the single-GPU engine is not reentrant).
_gen_lock = threading.Lock()
_pref_cache = {"ids": [], "chunks": 0}   # snapshot buffers currently hold this prefix's KV
_prev_ids: list = []

def _cache_plan(ids):
    """Decide prefix-cache reuse/snapshot. Call under _gen_lock.
    Reuse the cached prefix if this prompt starts with it (exact chunk-aligned length, so the sliding
    ring stays clean), then (re)snapshot this prompt's FULL MC-aligned prefix -- so the very next
    request that shares it reuses immediately (1-turn warmup) and a growing conversation keeps extending
    the cache, re-prefilling only each turn's new tokens."""
    MC = E.MC; CMAX = E.CACHE_MAX // MC
    nfull = (len(ids) - 1) // MC if len(ids) > MC else 0
    ck = _pref_cache["chunks"]
    reuse = ck if (0 < ck <= nfull and ids[:ck * MC] == _pref_cache["ids"]) else 0
    snap = min(nfull, CMAX)
    snap_arg = snap if snap > reuse else 0    # re-snapshot only when extending past what we reused
    return reuse, snap_arg

def _cache_commit(ids, snap_arg):
    if snap_arg: _pref_cache.update(ids=ids[:snap_arg * E.MC], chunks=snap_arg)

# ---- tool calling: Gemma emits <|tool_call>(48) call:NAME{gemma-args} <tool_call|>(49). Parse that
# span and convert Gemma's arg syntax (key:<|"|>val<|"|>) to JSON for OpenAI `tool_calls`. ----
_TC_OPEN, _TC_CLOSE = 48, 49
_CH_OPEN, _CH_CLOSE = 100, 101   # <|channel> .. <channel|> : Gemma's reasoning channel (keep out of content)

def _visible_ids(ids):
    """Drop reasoning-channel spans so the channel label/thought text never leaks into content."""
    out, inch = [], False
    for t in ids:
        if t == _CH_OPEN: inch = True
        elif t == _CH_CLOSE: inch = False
        elif not inch: out.append(t)
    return out

_QUOTE = '<|"|>'   # Gemma's string-delimiter token in tool-call args

# Per-OS filesystem handling: tools run on THIS host, so normalize path separators in path-like tool
# args to the host OS -- the model's slash direction (often Unix `/` even on Windows) then doesn't matter.
# Only path-keys are touched; bash `command`, glob/grep `pattern`, and `url` keep their slashes.
_HOST_OS = platform.system()          # 'Windows' | 'Linux' | 'Darwin'
_HOST_SEP, _FOREIGN_SEP = ("\\", "/") if _HOST_OS == "Windows" else ("/", "\\")
_PATH_KEYS = {"filepath", "path", "file", "filename", "dir", "directory", "folder",
              "cwd", "workdir", "source", "destination", "target", "output", "input"}

def _normalize_paths(args):
    if not isinstance(args, dict) or "_raw" in args:
        return args
    for k, v in list(args.items()):
        if isinstance(v, str) and v and k.lower() in _PATH_KEYS and (_FOREIGN_SEP in v):
            args[k] = v.replace(_FOREIGN_SEP, _HOST_SEP)
    return args

def _strip_val(v):
    """Unwrap a value that may be delimited by <|"|>, '...', or "..." (else return trimmed)."""
    v = v.strip()
    if v.startswith(_QUOTE) and v.endswith(_QUOTE) and len(v) >= 2 * len(_QUOTE):
        return v[len(_QUOTE):-len(_QUOTE)]
    if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
        return v[1:-1]
    return v

# Shell tools that opencode/agents declare under a bash-ish name but which actually run PowerShell here.
_SHELL_TOOL_NAMES = {"bash", "sh", "shell", "terminal", "run", "exec", "command",
                     "powershell", "cmd", "run_command", "runcommand", "execute", "process"}
_PS_DESC = ("Executes a command in Windows PowerShell (powershell.exe), NON-INTERACTIVE -- despite any "
            "name like 'bash', this is NOT bash. The command MUST be valid PowerShell: use New-Item / "
            "Remove-Item -Recurse -Force / Get-ChildItem / Get-Content / Set-Content, Start-Process, and "
            "Invoke-WebRequest -UseBasicParsing (or curl.exe). Windows paths use backslashes. Do NOT use "
            "bash syntax (ls, cat, rm -rf, mkdir -p, touch, &&, ||) or interactive prompts (Read-Host).")

def _coerce_shell_tools(tools):
    """Rewrite shell-tool DESCRIPTIONS (not names) so the model targets PowerShell, not bash. No-op unless
    GEMMA4_COERCE_PS (Windows). The name is preserved so the emitted tool_call still matches opencode's tool."""
    if not COERCE_PS or not tools:
        return tools
    out = []
    for t in tools:
        fn = t.get("function") or {}
        name = (fn.get("name") or "").lower()
        desc = (fn.get("description") or "").lower()
        if name in _SHELL_TOOL_NAMES or "bash" in desc or "shell command" in desc or "terminal" in desc:
            t = json.loads(json.dumps(t))                         # deep copy; don't mutate the request
            fn = t["function"]; fn["description"] = _PS_DESC
            props = ((fn.get("parameters") or {}).get("properties") or {})
            for ck in ("command", "cmd", "script"):
                if isinstance(props.get(ck), dict):
                    props[ck]["description"] = "PowerShell command (powershell.exe, non-interactive). Use cmdlets, not bash."
        out.append(t)
    return out

def _gemma_args_to_json(argstr):
    """Parse Gemma's tool-arg syntax {k:<|"|>str<|"|>, k2:num, ...} -> dict. Strings are <|"|>-delimited
    LITERALS, extracted verbatim (backslashes/quotes -- e.g. Windows paths C:\\Users\\... -- stay intact).

    Real opencode output is often malformed on complex values (a `write` whose content is JS full of
    backticks/quotes/colons): the model closes the content string with a stray ` instead of <|"|> and
    wraps filePath in '...' rather than <|"|>. We recover both: an unterminated <|"|> string splits off
    a trailing ,<key>:<value> from its tail, and values may be '..'/".."-quoted. Without this, filePath
    is lost -> opencode reports SchemaError(Missing filePath) and the model retries identically forever."""
    s = argstr.strip()
    if s.startswith('{'): s = s[1:]
    if s.endswith('}'): s = s[:-1]
    out, i, n, Q = {}, 0, len(s), len(_QUOTE)
    while i < n:
        while i < n and s[i] in ' ,\t\r\n': i += 1
        c = s.find(':', i)
        if c < 0: break
        key = s[i:c].strip().strip('"\'').strip()
        i = c + 1
        while i < n and s[i] in ' \t': i += 1
        if s[i:i + Q] == _QUOTE:                                  # <|"|> string literal
            e = s.find(_QUOTE, i + Q)
            if e >= 0:
                out[key] = s[i + Q:e]; i = e + Q
            else:                                                # unterminated: recover trailing ,key:val
                rest = s[i + Q:]
                ms = list(re.finditer(r"[`'\"]?\s*,\s*([A-Za-z_]\w*)\s*:", rest))
                if ms:
                    m = ms[-1]
                    out[key] = rest[:m.start()].rstrip("`'\" \t\r\n")
                    out[m.group(1)] = _strip_val(rest[m.end():])
                else:
                    out[key] = rest
                i = n
        elif s[i] in "'\"":                                       # ' or " quoted string literal
            q = s[i]; e = s.find(q, i + 1)
            if e < 0: e = n
            out[key] = s[i + 1:e]; i = e + 1
        else:                                                     # bare value, depth-aware up to a comma
            d, e = 0, i
            while e < n and (s[e] != ',' or d > 0):
                if s[e] in '[{': d += 1
                elif s[e] in ']}': d -= 1
                e += 1
            raw = s[i:e].strip(); i = e
            try: out[key] = json.loads(raw.replace(_QUOTE, '"'))
            except Exception: out[key] = raw.strip('"\'')
    return out

def _parse_tool_output(gen_ids):
    """gen_ids -> (content_text_or_None, [openai tool_call dicts])."""
    calls, i, first = [], 0, None
    while i < len(gen_ids):
        if gen_ids[i] == _TC_OPEN:
            if first is None: first = i
            j = i + 1
            while j < len(gen_ids) and gen_ids[j] != _TC_CLOSE: j += 1
            span = E.tok.decode(gen_ids[i + 1:j], skip_special_tokens=False)   # keep <|"|> markers intact
            m = re.match(r'\s*call:([A-Za-z_]\w*)\s*(\{.*\})?\s*$', span, re.S)
            name = m.group(1) if m else span.replace(_QUOTE, '"').strip()
            args = _normalize_paths(_gemma_args_to_json(m.group(2))) if (m and m.group(2)) else {}
            calls.append({"id": _uid("call"), "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args)}})
            i = j + 1
        else:
            i += 1
    seg = _visible_ids(gen_ids if first is None else gen_ids[:first])
    content = (E.tok.decode(seg, skip_special_tokens=True).strip() or None) if seg else None
    return content, calls

# ---- DEBUG LOGGING helpers (remove with the DEBUG block above) ----
_dbg_lock = threading.Lock()

def _split_think(gen_ids):
    """-> (decoded thought text, thinking_token_count) for the <|channel>..<channel|> reasoning spans."""
    th, inch = [], False
    for t in gen_ids:
        if t == _CH_OPEN: inch = True
        elif t == _CH_CLOSE: inch = False
        elif inch: th.append(t)
    return (E.tok.decode(th, skip_special_tokens=True) if th else ""), len(th)

def _dbg_msgs(messages):
    out = []
    for m in messages:
        d = {"role": m.role}
        if m.content is not None:
            c = m.content if isinstance(m.content, str) else json.dumps(m.content, ensure_ascii=False)
            d["content"] = c[:800]
        if m.tool_calls: d["tool_calls"] = m.tool_calls
        if m.tool_call_id: d["tool_call_id"] = m.tool_call_id
        if m.name: d["name"] = m.name
        out.append(d)
    return out

def _dbg_inference(kind, ids, gen_ids, content, calls, finish, comp_n, t0, req=None):
    """Append one structured record of a full inference to out/debug.jsonl (no-op unless GEMMA4_DEBUG=1)."""
    if not DEBUG: return
    try:
        thought, think_n = _split_think(gen_ids)
        rec = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "kind": kind,
            "prompt_tokens": len(ids),
            "completion_tokens": comp_n,
            "finish": finish,
            "elapsed_s": round(time.time() - t0, 3),
            "thinking_tokens": think_n,
            "thinking": thought,                                                   # decoded reasoning text
            "visible": content,                                                    # what the client actually gets
            "tool_calls": calls or [],
            "gen_raw": E.tok.decode(gen_ids, skip_special_tokens=False),           # RAW: special tokens kept
            "gen_token_ids": gen_ids,
            "prompt_tail": E.tok.decode(ids[-240:], skip_special_tokens=False),    # how the template rendered
        }
        if req is not None:
            rec["req_temperature"] = getattr(req, "temperature", None)   # to tell greedy from sampled failures
            rec["enable_thinking"] = THINK
            if getattr(req, "messages", None): rec["request_messages"] = _dbg_msgs(req.messages)
            if getattr(req, "tools", None):
                rec["request_tools"] = [t.get("function", {}).get("name") for t in req.tools]
        line = json.dumps(rec, ensure_ascii=False)
        with _dbg_lock:
            os.makedirs(os.path.dirname(DEBUG_FILE), exist_ok=True)
            with open(DEBUG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        # concise stdout breadcrumb -- the empty-response smell is visible_chars=0 with content generated
        print(f"[debug] {kind} prompt={len(ids)} completion={comp_n} finish={finish} "
              f"think_tok={think_n} visible_chars={len(content or '')} tool_calls={len(calls or [])}", flush=True)
    except Exception as ex:
        print("[debug] log failed:", ex, flush=True)


def _run(ids, max_new, temperature, top_p, stop_strs, top_k=0, min_p=0.0, rep_penalty=1.0, dbg_ids=None):
    """Yield (delta_text, finish_reason, n_tokens). finish_reason is None until the final yield.
    Decodes the whole running id list each step so sentencepiece spacing is correct; trims at stop
    strings. n_tokens is the real count of tokens the engine produced (for accurate usage)."""
    with _gen_lock:
        reuse, snap_arg = _cache_plan(ids)
        gen = E.generate(ids, max_new, temperature, top_p, STOP_IDS, reuse_chunks=reuse, snap_chunks=snap_arg,
                         top_k=top_k, min_p=min_p, rep_penalty=rep_penalty,
                         think_budget=(THINK_BUDGET if THINK else 0))
        _done = object(); first = next(gen, _done)   # runs prefill (incl. snapshot) + first decode token
        _cache_commit(ids, snap_arg)                  # snapshot now committed

        def _toks():
            if first is not _done:
                yield first
                yield from gen
        vis, prev, n, inch = [], "", 0, False
        for tid in _toks():
            n += 1
            if dbg_ids is not None: dbg_ids.append(tid)        # raw capture for GEMMA4_DEBUG
            if tid == _CH_OPEN: inch = True; continue          # suppress reasoning-channel content
            if tid == _CH_CLOSE: inch = False; continue
            if inch: continue
            vis.append(tid)
            text = E.tok.decode(vis)
            cut = min([text.find(s) for s in stop_strs if s and text.find(s) != -1], default=-1)
            if cut != -1:
                d = text[:cut][len(prev):]
                if d: yield d, None, n
                yield "", "stop", n; return
            d = text[len(prev):]; prev = text
            if d: yield d, None, n
        # OpenAI counts the stopping EOS token in completion_tokens; the engine consumes it without
        # yielding, so +1 when we stopped on a stop token (n < max_new) rather than hitting the length cap.
        yield "", ("length" if n >= max_new else "stop"), (n if n >= max_new else n + 1)


def _run_tool(ids, max_new, temperature, top_p, top_k=0, min_p=0.0, rep_penalty=1.0):
    """Token-level generation for tool mode. Also stops at <tool_call|> so we halt right after a call
    (49 is NOT in the model's default stops, so without this it rambles past the call). Yields token ids."""
    with _gen_lock:
        reuse, snap_arg = _cache_plan(ids)
        gen = E.generate(ids, max_new, temperature, top_p, set(STOP_IDS) | {_TC_CLOSE},
                         reuse_chunks=reuse, snap_chunks=snap_arg,
                         top_k=top_k, min_p=min_p, rep_penalty=rep_penalty,
                         think_budget=(THINK_BUDGET if THINK else 0))
        _done = object(); first = next(gen, _done)
        _cache_commit(ids, snap_arg)
        if first is not _done:
            yield first
            yield from gen

def _deser_tool_args(tool_calls):
    """--- gist-derived fix (remove this fn + its call to revert) ---
    OpenAI/Vercel-AI-SDK send tool_calls[].function.arguments as a JSON *string*; the Gemma template
    then wraps it as invalid DSL `call:fn{{"k":"v"}}` (quoted keys/colons the model never saw) -> tool-arg
    'collapse' on later turns. Deserialize string args -> dict so the template renders valid `{k:<|"|>v<|"|>}`."""
    out = []
    for tc in tool_calls or []:
        tc = dict(tc); fn = dict(tc.get("function") or {})
        a = fn.get("arguments")
        if isinstance(a, str):
            try: fn["arguments"] = json.loads(a)
            except Exception: fn["arguments"] = {}
        tc["function"] = fn; out.append(tc)
    return out

def _msg_dicts(messages):
    """Msg objects -> chat-template dicts, preserving tool_calls / tool results for the round-trip."""
    out = []
    for m in messages:
        d = {"role": m.role, "content": m.content if m.content is not None else ""}
        if m.tool_calls: d["tool_calls"] = _deser_tool_args(m.tool_calls)
        if m.tool_call_id: d["tool_call_id"] = m.tool_call_id
        if m.name: d["name"] = m.name
        out.append(d)
    return out


# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "wedged" if _GPU_WEDGED.is_set() else "ok", "model": MODEL_ID,
            "gpu_wedged": _GPU_WEDGED.is_set()}

@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "created": _now(), "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    _guard_wedged()
    kw = {"tools": _coerce_shell_tools(req.tools)} if req.tools else {}   # PowerShell-coerced tool decls
    ids = E.tok.apply_chat_template(_msg_dicts(req.messages), add_generation_prompt=True,
                                    tokenize=True, return_dict=False, enable_thinking=THINK, **kw)
    ids = [int(i) for i in np.array(ids).ravel()]
    if req.tools:
        return _complete_tools(req, ids)
    return _complete(req, ids, chat=True)


@app.post("/v1/completions")
def completions(req: CompReq):
    _guard_wedged()
    prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
    ids = [int(i) for i in E.tok(prompt, return_tensors=None)["input_ids"]]
    return _complete(req, ids, chat=False)


def _complete_tools(req, ids):
    """Chat completion with tool calling: stream text until a tool call starts, then emit OpenAI
    `tool_calls`. Non-stream returns content and/or tool_calls with finish_reason 'tool_calls'."""
    cid = _uid("chatcmpl"); created = _now(); prompt_n = len(ids)
    max_new = _fit(prompt_n, req.max_tokens); model = req.model
    ttemp = min(req.temperature, TOOL_TEMP)   # clamp: high temp samples away from the <|tool_call> DSL
    def chunk(delta, fin):
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}

    if req.stream:
        def sse():
            t0 = time.time()
            yield f"data: {json.dumps(chunk({'role': 'assistant'}, None))}\n\n"
            gen_ids, vis, prev, comp_n, toolmode, inch = [], [], "", 0, False, False
            try:
                for tid in _run_tool(ids, max_new, ttemp, req.top_p, req.top_k, req.min_p, req.repetition_penalty):
                    gen_ids.append(tid); comp_n = len(gen_ids)
                    if tid == _TC_OPEN: toolmode = True
                    if not toolmode:
                        if tid == _CH_OPEN: inch = True; continue
                        if tid == _CH_CLOSE: inch = False; continue
                        if inch: continue
                        vis.append(tid)
                        text = E.tok.decode(vis, skip_special_tokens=True)
                        d = text[len(prev):]; prev = text
                        if d: yield f"data: {json.dumps(chunk({'content': d}, None))}\n\n"
            except RuntimeError as ex:
                if not _is_fence_timeout(ex): raise
                _mark_wedged(ex)
                yield f"data: {json.dumps(chunk({'content': ' [error: GPU wedged, restart required]'}, 'stop'))}\n\n"
                yield "data: [DONE]\n\n"; return
            dbg_content, calls = _parse_tool_output(gen_ids)
            for idx, tc in enumerate(calls):
                yield f"data: {json.dumps(chunk({'tool_calls': [{'index': idx, **tc}]}, None))}\n\n"
            fin = "tool_calls" if calls else ("length" if comp_n >= max_new else "stop")
            final = chunk({}, fin)
            final["usage"] = {"prompt_tokens": prompt_n, "completion_tokens": comp_n, "total_tokens": prompt_n + comp_n}
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"
            _log_request("chat-tools-stream", prompt_n, max_new, comp_n, fin, t0)
            _dbg_inference("chat-tools-stream", ids, gen_ids, dbg_content, calls, fin, comp_n, t0, req)
        return StreamingResponse(sse(), media_type="text/event-stream")

    t0 = time.time()
    try:
        gen_ids = list(_run_tool(ids, max_new, ttemp, req.top_p, req.top_k, req.min_p, req.repetition_penalty))
    except RuntimeError as ex:
        if _is_fence_timeout(ex): _mark_wedged(ex); raise HTTPException(503, "GPU wedged (fence timeout); restart required")
        raise
    comp_n = len(gen_ids)
    content, calls = _parse_tool_output(gen_ids)
    fin = "tool_calls" if calls else ("length" if comp_n >= max_new else "stop")
    _log_request("chat-tools", prompt_n, max_new, comp_n, fin, t0)
    _dbg_inference("chat-tools", ids, gen_ids, content, calls, fin, comp_n, t0, req)
    msg = {"role": "assistant", "content": content}
    if calls: msg["tool_calls"] = calls
    return {"id": cid, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
            "usage": {"prompt_tokens": prompt_n, "completion_tokens": comp_n, "total_tokens": prompt_n + comp_n}}


def _complete(req, ids, chat: bool):
    max_new = _fit(len(ids), req.max_tokens)
    stop_strs = _stops(req.stop)
    cid = _uid("chatcmpl" if chat else "cmpl")
    created = _now(); prompt_n = len(ids)
    obj = "chat.completion" if chat else "text_completion"

    if req.stream:
        def sse():
            t0 = time.time()
            dbg = [] if DEBUG else None
            vparts = []
            with GPU:
                comp_n = 0
                finish = "length"
                if chat:
                    first = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": req.model, "choices": [{"index": 0,
                             "delta": {"role": "assistant"}, "finish_reason": None}]}
                    yield f"data: {json.dumps(first)}\n\n"
                try:
                    for delta, fin, ntok in _run(ids, max_new, req.temperature, req.top_p, stop_strs, req.top_k, req.min_p, req.repetition_penalty, dbg_ids=dbg):
                        comp_n = ntok
                        if delta: vparts.append(delta)
                        if fin is not None: finish = fin
                        if chat:
                            ch = {"delta": {"content": delta} if delta else {}, "index": 0, "finish_reason": fin}
                        else:
                            ch = {"text": delta, "index": 0, "finish_reason": fin, "logprobs": None}
                        chunk = {"id": cid, "object": "chat.completion.chunk" if chat else "text_completion",
                                 "created": created, "model": req.model, "choices": [ch]}
                        if fin is not None:
                            chunk["usage"] = {"prompt_tokens": prompt_n, "completion_tokens": comp_n,
                                              "total_tokens": prompt_n + comp_n}
                        yield f"data: {json.dumps(chunk)}\n\n"
                except RuntimeError as ex:
                    if not _is_fence_timeout(ex): raise
                    _mark_wedged(ex)
                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': created, 'model': req.model, 'choices': [{'index': 0, 'delta': {'content': ' [error: GPU wedged, restart required]'}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"
                _log_request("chat-stream" if chat else "completion-stream", prompt_n, max_new, comp_n, finish, t0)
                _dbg_inference("chat-stream" if chat else "completion-stream", ids, dbg or [],
                               "".join(vparts), [], finish, comp_n, t0, req)
        return StreamingResponse(sse(), media_type="text/event-stream")

    t0 = time.time()
    dbg = [] if DEBUG else None
    try:
        with GPU:
            parts, finish, comp_n = [], "length", 0
            for delta, fin, ntok in _run(ids, max_new, req.temperature, req.top_p, stop_strs, req.top_k, req.min_p, req.repetition_penalty, dbg_ids=dbg):
                if delta: parts.append(delta)
                comp_n = ntok
                if fin is not None: finish = fin
            text = "".join(parts)
    except RuntimeError as ex:
        if _is_fence_timeout(ex): _mark_wedged(ex); raise HTTPException(503, "GPU wedged (fence timeout); restart required")
        raise
    _log_request("chat" if chat else "completion", prompt_n, max_new, comp_n, finish, t0)
    _dbg_inference("chat" if chat else "completion", ids, dbg or [], text, [], finish, comp_n, t0, req)
    usage = {"prompt_tokens": prompt_n, "completion_tokens": comp_n, "total_tokens": prompt_n + comp_n}
    if chat:
        choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}
    else:
        choice = {"index": 0, "text": text, "finish_reason": finish, "logprobs": None}
    return {"id": cid, "object": obj, "created": created, "model": req.model,
            "choices": [choice], "usage": usage}


# ======================= Responses API (/v1/responses, OpenAI-spec, text) =======================
# Maps OpenAI's `input`/`output` shape onto the engine. Supports streaming (typed SSE events) and the
# spec's stateful path (`store` + `previous_response_id`) via a small in-memory conversation store.
# Not supported (text-only engine): tool/function calling, multimodal input, structured-output schemas.
_resp_store: dict = {}   # response_id -> full message list (incl. the assistant reply) for continuation

class RespReq(BaseModel):
    model: str = MODEL_ID
    input: Union[str, List[dict]]
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    store: bool = False
    previous_response_id: Optional[str] = None
    stop: Optional[Union[str, List[str]]] = None

def _resp_messages(req: RespReq):
    """Build the chat-template message list from a Responses request (+ prior turns if continuing)."""
    msgs = list(_resp_store.get(req.previous_response_id, [])) if req.previous_response_id else []
    if req.instructions and not msgs:
        msgs.append({"role": "system", "content": req.instructions})
    items = req.input if isinstance(req.input, list) else [{"role": "user", "content": req.input}]
    for it in items:
        if isinstance(it, str):
            msgs.append({"role": "user", "content": it}); continue
        if it.get("type", "message") != "message":   # skip tool calls / non-message items (unsupported)
            continue
        content = it.get("content", "")
        text = ("".join(c.get("text", "") for c in content if isinstance(c, dict) and "text" in c)
                if isinstance(content, list) else str(content))
        role = it.get("role", "user")
        msgs.append({"role": role if role in ("user", "assistant", "system") else "user", "content": text})
    return msgs

def _resp_obj(rid, model, status, output, pn, cn, req, incomplete=None):
    return {"id": rid, "object": "response", "created_at": _now(), "status": status, "model": model,
            "output": output, "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "temperature": req.temperature, "top_p": req.top_p, "max_output_tokens": req.max_output_tokens,
            "instructions": req.instructions, "incomplete_details": incomplete, "error": None, "metadata": {},
            "usage": {"input_tokens": pn, "output_tokens": cn, "total_tokens": pn + cn}}

def _msg_item(item_id, text, status="completed"):
    return {"id": item_id, "type": "message", "status": status, "role": "assistant",
            "content": ([{"type": "output_text", "text": text, "annotations": []}] if text or status == "completed" else [])}

@app.post("/v1/responses")
def responses(req: RespReq):
    _guard_wedged()
    msgs = _resp_messages(req)
    ids = [int(x) for x in np.array(E.tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=THINK)).ravel()]
    max_new = _fit(len(ids), req.max_output_tokens); pn = len(ids)
    rid = _uid("resp"); item_id = _uid("msg"); stop_strs = _stops(req.stop)

    def _finalize(text, cn, fin):
        if req.store:
            _resp_store[rid] = msgs + [{"role": "assistant", "content": text}]

    if req.stream:
        def sse():
            seq = [0]
            def ev(typ, payload):
                p = {"type": typ, "sequence_number": seq[0], **payload}; seq[0] += 1
                return f"event: {typ}\ndata: {json.dumps(p)}\n\n"
            r0 = _resp_obj(rid, req.model, "in_progress", [], pn, 0, req)
            yield ev("response.created", {"response": r0})
            yield ev("response.in_progress", {"response": r0})
            yield ev("response.output_item.added", {"output_index": 0, "item": _msg_item(item_id, "", "in_progress")})
            yield ev("response.content_part.added", {"item_id": item_id, "output_index": 0, "content_index": 0,
                     "part": {"type": "output_text", "text": "", "annotations": []}})
            parts, cn, fin = [], 0, "completed"
            for delta, f, ntok in _run(ids, max_new, req.temperature, req.top_p, stop_strs):
                cn = ntok
                if delta:
                    parts.append(delta)
                    yield ev("response.output_text.delta", {"item_id": item_id, "output_index": 0,
                             "content_index": 0, "delta": delta})
                if f is not None: fin = f
            text = "".join(parts); done = fin == "stop"
            yield ev("response.output_text.done", {"item_id": item_id, "output_index": 0, "content_index": 0, "text": text})
            yield ev("response.content_part.done", {"item_id": item_id, "output_index": 0, "content_index": 0,
                     "part": {"type": "output_text", "text": text, "annotations": []}})
            yield ev("response.output_item.done", {"output_index": 0, "item": _msg_item(item_id, text)})
            inc = None if done else {"reason": "max_output_tokens"}
            final = _resp_obj(rid, req.model, "completed" if done else "incomplete", [_msg_item(item_id, text)], pn, cn, req, inc)
            _finalize(text, cn, fin)
            yield ev("response.completed" if done else "response.incomplete", {"response": final})
        return StreamingResponse(sse(), media_type="text/event-stream")

    parts, cn, fin = [], 0, "completed"
    for delta, f, ntok in _run(ids, max_new, req.temperature, req.top_p, stop_strs):
        if delta: parts.append(delta)
        cn = ntok
        if f is not None: fin = f
    text = "".join(parts); done = fin == "stop"
    _finalize(text, cn, fin)
    inc = None if done else {"reason": "max_output_tokens"}
    return _resp_obj(rid, req.model, "completed" if done else "incomplete", [_msg_item(item_id, text)], pn, cn, req, inc)


# ======================= Session WebSocket (/v1/sessions) =======================
# A STATEFUL chat session over one connection: the server keeps the conversation, the client sends only
# the next message. Multi-turn rides the prefix KV cache (each turn re-prefills only the new message).
# Protocol (JSON per frame):
#   server->client: {"type":"session.created","session_id":...}
#   client->server: {"type":"configure","system":?,"temperature":?,"top_p":?,"max_tokens":?,"reset":?}
#                    {"type":"message","content":"..."}
#   server->client: {"type":"session.updated","config":{...}}
#                    {"type":"response.start"} {"type":"response.delta","content":...}
#                    {"type":"response.done","content":...,"finish_reason":...,"usage":{...}}
#                    {"type":"error","message":...}
@app.websocket("/v1/sessions")
async def sessions(ws: WebSocket):
    await ws.accept()
    st = {"system": None, "temperature": 1.0, "top_p": 1.0, "max_tokens": DEFAULT_MAX_TOKENS, "history": []}
    sid = _uid("sess")
    await ws.send_json({"type": "session.created", "session_id": sid})
    loop = asyncio.get_running_loop()
    try:
        while True:
            ev = await ws.receive_json()
            t = ev.get("type")
            if t == "configure":
                for k in ("system", "temperature", "top_p", "max_tokens"):
                    if k in ev: st[k] = ev[k]
                if ev.get("reset"): st["history"] = []
                await ws.send_json({"type": "session.updated",
                                    "config": {k: st[k] for k in ("system", "temperature", "top_p", "max_tokens")}})
            elif t == "message":
                st["history"].append({"role": "user", "content": ev.get("content", "")})
                msgs = ([{"role": "system", "content": st["system"]}] if st["system"] else []) + st["history"]
                ids = [int(x) for x in np.array(E.tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=THINK)).ravel()]
                pn = len(ids); max_new = _fit(pn, st["max_tokens"])
                await ws.send_json({"type": "response.start"})
                q: asyncio.Queue = asyncio.Queue()

                def worker():
                    parts, last_n, fin = [], 0, "stop"
                    try:
                        for delta, f, ntok in _run(ids, max_new, st["temperature"], st["top_p"], []):
                            last_n = ntok
                            if delta: parts.append(delta); loop.call_soon_threadsafe(q.put_nowait, ("delta", delta))
                            if f is not None: fin = f
                    finally:
                        loop.call_soon_threadsafe(q.put_nowait, ("done", "".join(parts), last_n, fin))
                fut = loop.run_in_executor(None, worker)
                while True:
                    m = await q.get()
                    if m[0] == "delta":
                        await ws.send_json({"type": "response.delta", "content": m[1]})
                    else:
                        _, text, n, fin = m
                        st["history"].append({"role": "assistant", "content": text})
                        await ws.send_json({"type": "response.done", "content": text, "finish_reason": fin,
                                            "usage": {"prompt_tokens": pn, "completion_tokens": n, "total_tokens": pn + n}})
                        break
                await fut
            else:
                await ws.send_json({"type": "error", "message": f"unsupported event type: {t!r}"})
    except WebSocketDisconnect:
        pass
    except Exception as ex:   # don't kill the worker thread silently; report and close
        try: await ws.send_json({"type": "error", "message": str(ex)})
        except Exception: pass


def _warmup():
    if os.environ.get("GEMMA4_SKIP_WARMUP") == "1":
        print("[serve] warmup skipped (GEMMA4_SKIP_WARMUP=1)", flush=True)
        return
    prompt = [{"role": "user", "content": "warmup"}]
    ids = E.tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, return_dict=False, enable_thinking=THINK)
    ids = [int(i) for i in np.array(ids).ravel()]
    t0 = time.time()
    gen = E.generate(ids, 1, 0.0, 1.0, STOP_IDS, reuse_chunks=0, snap_chunks=0)
    next(gen, None)
    print(f"[serve] warmup done prompt={len(ids)} elapsed={time.time()-t0:.3f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    _warmup()
    if DEBUG:
        print(f"[serve] DEBUG ON -> per-inference records appended to {os.path.abspath(DEBUG_FILE)}", flush=True)
    print(f"[serve] ready on http://{a.host}:{a.port}  (model={MODEL_ID}, stop_ids={sorted(STOP_IDS)}, "
          f"default_max_tokens={DEFAULT_MAX_TOKENS}, think={THINK})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
