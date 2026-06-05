"""OpenAI-compatible API server for the raw-Vulkan Gemma 4 12B engine (Adreno X2 GPU).

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions
(streaming + non-streaming). The engine holds one global GPU command buffer + KV cache, so all
generation is serialized behind a single lock -- correct for a local single-GPU server.

Run:  .venv-gemma4/Scripts/python.exe src/serve.py [--host H] [--port P]
Import of vk_engine loads the model + records the command buffer (~minutes) at startup.
"""
import os, sys, time, json, uuid, threading, argparse, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from typing import Optional, Union, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import vk_engine as E   # heavy: loads weights + records the Vulkan command buffer on import

MODEL_ID = "gemma-4-12b-it"

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
    content: str

class ChatReq(BaseModel):
    model: str = MODEL_ID
    messages: List[Msg]
    max_tokens: Optional[int] = 16384
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

class CompReq(BaseModel):
    model: str = MODEL_ID
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 16384
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
    return max(1, min(req_max or 16384, E.MAXT - prompt_len))

# Prefix KV cache: the engine snapshots a shared prompt prefix's KV so repeated prompts skip
# re-prefilling it. We auto-detect the prefix as the chunk-aligned longest-common-prefix of consecutive
# requests (typically the system prompt). Reuse is allowed only at the EXACT cached length so the
# sliding-window ring is clean. A lock serializes generations (the single-GPU engine is not reentrant).
_gen_lock = threading.Lock()
_pref_cache = {"ids": [], "chunks": 0}   # snapshot buffers currently hold this prefix's KV
_prev_ids: list = []

def _run(ids, max_new, temperature, top_p, stop_strs):
    """Yield (delta_text, finish_reason, n_tokens). finish_reason is None until the final yield.
    Decodes the whole running id list each step so sentencepiece spacing is correct; trims at stop
    strings. n_tokens is the real count of tokens the engine produced (for accurate usage)."""
    global _prev_ids
    with _gen_lock:
        MC = E.MC; CMAX = E.CACHE_MAX // MC
        n_ids = len(ids); nfull = (n_ids - 1) // MC if n_ids > MC else 0
        ck = _pref_cache["chunks"]
        reuse = ck if (0 < ck <= nfull and ids[:ck * MC] == _pref_cache["ids"]) else 0
        lcp = 0
        for a, b in zip(ids, _prev_ids):
            if a != b: break
            lcp += 1
        snap = min(lcp // MC, nfull, CMAX)
        snap_arg = snap if snap > reuse else 0   # (re)snapshot only when extending past the reused prefix
        _prev_ids = list(ids)
        gen = E.generate(ids, max_new, temperature, top_p, STOP_IDS, reuse_chunks=reuse, snap_chunks=snap_arg)
        _done = object(); first = next(gen, _done)   # runs prefill (incl. snapshot) + first decode token
        if snap_arg: _pref_cache.update(ids=ids[:snap_arg * MC], chunks=snap_arg)   # snapshot now committed

        def _toks():
            if first is not _done:
                yield first
                yield from gen
        gen_ids, prev, n = [], "", 0
        for tid in _toks():
            n += 1; gen_ids.append(tid)
            text = E.tok.decode(gen_ids)
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


# ---------- routes ----------
@app.get("/health")
def health(): return {"status": "ok", "model": MODEL_ID}

@app.get("/v1/models")
def models():
    return {"object": "list",
            "data": [{"id": MODEL_ID, "object": "model", "created": _now(), "owned_by": "local"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatReq):
    ids = E.tok.apply_chat_template([m.model_dump() for m in req.messages],
                                    add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = [int(i) for i in np.array(ids).ravel()]
    return _complete(req, ids, chat=True)


@app.post("/v1/completions")
def completions(req: CompReq):
    prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
    ids = [int(i) for i in E.tok(prompt, return_tensors=None)["input_ids"]]
    return _complete(req, ids, chat=False)


def _complete(req, ids, chat: bool):
    max_new = _fit(len(ids), req.max_tokens)
    stop_strs = _stops(req.stop)
    cid = _uid("chatcmpl" if chat else "cmpl")
    created = _now(); prompt_n = len(ids)
    obj = "chat.completion" if chat else "text_completion"

    if req.stream:
        def sse():
            with GPU:
                comp_n = 0
                if chat:
                    first = {"id": cid, "object": "chat.completion.chunk", "created": created,
                             "model": req.model, "choices": [{"index": 0,
                             "delta": {"role": "assistant"}, "finish_reason": None}]}
                    yield f"data: {json.dumps(first)}\n\n"
                for delta, fin, ntok in _run(ids, max_new, req.temperature, req.top_p, stop_strs):
                    comp_n = ntok
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
                yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")

    with GPU:
        parts, finish, comp_n = [], "length", 0
        for delta, fin, ntok in _run(ids, max_new, req.temperature, req.top_p, stop_strs):
            if delta: parts.append(delta)
            comp_n = ntok
            if fin is not None: finish = fin
        text = "".join(parts)
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
    max_output_tokens: Optional[int] = 16384
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
    msgs = _resp_messages(req)
    ids = [int(x) for x in np.array(E.tok.apply_chat_template(
        msgs, add_generation_prompt=True, tokenize=True, return_dict=False)).ravel()]
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
    st = {"system": None, "temperature": 1.0, "top_p": 1.0, "max_tokens": 16384, "history": []}
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
                    msgs, add_generation_prompt=True, tokenize=True, return_dict=False)).ravel()]
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"[serve] ready on http://{a.host}:{a.port}  (model={MODEL_ID}, stop_ids={sorted(STOP_IDS)})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
