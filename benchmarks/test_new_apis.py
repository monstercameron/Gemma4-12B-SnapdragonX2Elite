"""Smoke-test the new server surfaces: Responses API (non-stream / stream / stateful continuation)
and the stateful session WebSocket (multi-turn, server-kept history)."""
import json, asyncio, urllib.request
import websockets

BASE = "http://127.0.0.1:8000"

def post(path, body):
    r = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())

print("=== /v1/responses (non-stream) ===")
d = post("/v1/responses", {"input": "Capital of France? One word.", "max_output_tokens": 16})
txt = d["output"][0]["content"][0]["text"]
print(f"  status={d['status']} object={d['object']} text={txt!r} usage={d['usage']}")
rid = d["id"]

print("\n=== /v1/responses (stateful: store + previous_response_id) ===")
d1 = post("/v1/responses", {"input": "My name is Ada. Remember it.", "max_output_tokens": 24, "store": True})
print(f"  turn1 id={d1['id']} text={d1['output'][0]['content'][0]['text']!r}")
d2 = post("/v1/responses", {"input": "What is my name?", "max_output_tokens": 16,
                            "previous_response_id": d1["id"], "store": True})
print(f"  turn2 (continues) text={d2['output'][0]['content'][0]['text']!r}")

print("\n=== /v1/responses (stream, typed SSE events) ===")
r = urllib.request.Request(BASE + "/v1/responses",
    data=json.dumps({"input": "Say hi in 3 words.", "max_output_tokens": 16, "stream": True}).encode(),
    headers={"Content-Type": "application/json"})
events, deltas = [], []
for raw in urllib.request.urlopen(r, timeout=120):
    line = raw.decode().strip()
    if line.startswith("event:"): events.append(line.split(":", 1)[1].strip())
    elif line.startswith("data:"):
        try:
            o = json.loads(line[5:].strip())
            if o.get("type") == "response.output_text.delta": deltas.append(o["delta"])
        except Exception: pass
print(f"  event sequence: {events}")
print(f"  streamed text: {''.join(deltas)!r}")

print("\n=== /v1/sessions (WebSocket, stateful multi-turn) ===")
async def ws_test():
    async with websockets.connect("ws://127.0.0.1:8000/v1/sessions") as ws:
        print("  ", json.loads(await ws.recv()))                      # session.created
        await ws.send(json.dumps({"type": "configure", "system": "You are concise.", "max_tokens": 24}))
        print("  ", json.loads(await ws.recv()))                      # session.updated
        async def turn(text):
            await ws.send(json.dumps({"type": "message", "content": text}))
            full = ""
            while True:
                m = json.loads(await ws.recv())
                if m["type"] == "response.delta": full += m["content"]
                elif m["type"] == "response.done":
                    print(f"   '{text}' -> {full!r}  usage={m['usage']}"); return
                elif m["type"] == "error": print("   ERROR", m); return
        await turn("My favorite color is teal.")
        await turn("What did I say my favorite color was?")          # tests server-kept history
asyncio.run(ws_test())
print("\nALL NEW-API SMOKE TESTS DONE")
