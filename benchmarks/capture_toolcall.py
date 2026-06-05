"""Capture the EXACT tool-call format Gemma 4 emits, so the server parser matches reality."""
import sys, os; sys.argv = [sys.argv[0]]; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np, vk_engine as E
tok = E.tok
tools = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}},
                         "required": ["city"]}}}]

def run(label, msgs):
    ids = [int(x) for x in np.array(tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                                            tokenize=True, return_dict=False)).ravel()]
    gen = list(E.generate(ids, max_new=80, temperature=0.0))
    print(f"\n=== {label} (prompt {len(ids)} tok) ===")
    print("RAW  :", repr(tok.decode(gen, skip_special_tokens=False)))
    print("CLEAN:", repr(tok.decode(gen, skip_special_tokens=True)))

run("tool-likely prompt", [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}])
run("no-tool prompt", [{"role": "user", "content": "Say hello in one word."}])

print("\n=== special-token ids ===")
vocab = tok.get_vocab()
for s in ["<|tool_call>", "<tool_call|>", "<|tool_response>", "<tool_response|>",
          "<|channel>", "<channel|>", "<|tool>", "<tool|>"]:
    print(f"  {s!r:20} -> {vocab.get(s, '(multi-token)')}")
