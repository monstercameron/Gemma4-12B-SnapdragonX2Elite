"""Test OpenAI tool calling end-to-end: (1) model emits a tool_call, (2) we send the tool result back,
(3) model gives a final answer using it. Also checks a no-tool prompt still returns plain content."""
import json, urllib.request
BASE = "http://127.0.0.1:8000"
def post(body):
    r = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())

TOOLS = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

print("=== turn 1: model should CALL the tool ===")
r1 = post({"model": "gemma-4-12b-it", "tools": TOOLS, "max_tokens": 128,
           "messages": [{"role": "user", "content": "What's the weather in Tokyo? Use the tool."}]})
ch = r1["choices"][0]; msg = ch["message"]
print(f"  finish_reason={ch['finish_reason']}")
print(f"  content={msg.get('content')!r}")
print(f"  tool_calls={json.dumps(msg.get('tool_calls'), indent=0)}")

if msg.get("tool_calls"):
    tc = msg["tool_calls"][0]
    print("\n=== turn 2: send tool result back, expect final answer ===")
    r2 = post({"model": "gemma-4-12b-it", "tools": TOOLS, "max_tokens": 128, "messages": [
        {"role": "user", "content": "What's the weather in Tokyo? Use the tool."},
        {"role": "assistant", "content": None, "tool_calls": [tc]},
        {"role": "tool", "tool_call_id": tc["id"], "content": "22C, partly cloudy"},
    ]})
    print(f"  finish_reason={r2['choices'][0]['finish_reason']}")
    print(f"  final answer={r2['choices'][0]['message']['content']!r}")

print("\n=== no-tool prompt with tools available (should still answer in text) ===")
r3 = post({"model": "gemma-4-12b-it", "tools": TOOLS, "max_tokens": 32,
           "messages": [{"role": "user", "content": "Say hi in one word."}]})
print(f"  finish_reason={r3['choices'][0]['finish_reason']}  content={r3['choices'][0]['message']['content']!r}")
print("\nTOOL-CALLING TEST DONE")
