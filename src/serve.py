"""OpenAI-compatible API server for the raw-Vulkan Gemma 4 12B engine (Adreno X2 GPU).

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions
(streaming + non-streaming). The engine holds one global GPU command buffer + KV cache, so all
generation is serialized behind a single lock -- correct for a local single-GPU server.

Run:  .venv-gemma4/Scripts/python.exe scripts/serve.py [--host H] [--port P]
Import of vk_engine loads the model + records the command buffer (~minutes) at startup.
"""
import os, sys, time, json, uuid, threading, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from typing import Optional, Union, List
from fastapi import FastAPI, HTTPException
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
    max_tokens: Optional[int] = 256
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None

class CompReq(BaseModel):
    model: str = MODEL_ID
    prompt: Union[str, List[str]]
    max_tokens: Optional[int] = 256
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
    return max(1, min(req_max or 256, E.MAXT - prompt_len))

def _run(ids, max_new, temperature, top_p, stop_strs):
    """Yield (delta_text, finish_reason, n_tokens). finish_reason is None until the final yield.
    Decodes the whole running id list each step so sentencepiece spacing is correct; trims at stop
    strings. n_tokens is the real count of tokens the engine produced (for accurate usage)."""
    gen_ids, prev, n = [], "", 0
    for tid in E.generate(ids, max_new, temperature, top_p, STOP_IDS):
        n += 1; gen_ids.append(tid)
        text = E.tok.decode(gen_ids)
        cut = min([text.find(s) for s in stop_strs if s and text.find(s) != -1], default=-1)
        if cut != -1:
            d = text[:cut][len(prev):]
            if d: yield d, None, n
            yield "", "stop", n; return
        d = text[len(prev):]; prev = text
        if d: yield d, None, n
    yield "", ("length" if n >= max_new else "stop"), n


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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    print(f"[serve] ready on http://{a.host}:{a.port}  (model={MODEL_ID}, stop_ids={sorted(STOP_IDS)})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")
