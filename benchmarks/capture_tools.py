"""Tiny capture endpoint: logs the `tools` array opencode sends, returns a minimal valid response."""
import json, uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
app = FastAPI()

@app.get("/v1/models")
def models(): return {"object": "list", "data": [{"id": "gemma-4-12b-it", "object": "model"}]}

@app.post("/v1/chat/completions")
async def cc(req: Request):
    body = await req.json()
    tools = body.get("tools") or []
    names = [t.get("function", {}).get("name") for t in tools]
    print("\n==== CAPTURED TOOLS:", len(names), "====", flush=True)
    for t in tools:
        fn = t.get("function", {})
        params = fn.get("parameters", {}).get("properties", {})
        print(f"  - {fn.get('name')}: {fn.get('description','')[:90]}  params={list(params.keys())}", flush=True)
    print("==== END TOOLS ====\n", flush=True)
    if body.get("stream"):
        def s():
            yield "data: " + json.dumps({"id": "x", "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}) + "\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(s(), media_type="text/event-stream")
    return JSONResponse({"id": "x", "object": "chat.completion", "choices": [{"index": 0,
        "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
