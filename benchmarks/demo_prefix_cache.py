"""Live demo of the server's transparent prefix KV cache: same long system prompt, different user
turns. By request 3 the shared prefix is reused -> lower latency (prefill of the system prompt skipped)."""
import time, json, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
SYS = "You are an expert assistant with deep knowledge across many domains. " * 40   # long, multi-chunk
QS = ["What is 2+2?", "Name a primary color.", "What is the capital of Japan?", "Say hello in French."]
TAGS = ["cold (warms prev)", "snapshots prefix", "REUSES prefix", "reuses prefix"]

def ask(q):
    body = json.dumps({"model": "gemma-4-12b-it", "temperature": 0, "max_tokens": 8,
                       "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": q}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=120).read())
    dt = time.time() - t0
    return dt, d["choices"][0]["message"]["content"], d["usage"]["prompt_tokens"]

for i, (q, tag) in enumerate(zip(QS, TAGS), 1):
    dt, txt, pt = ask(q)
    print(f"  req{i} [{tag:18}] {dt:6.2f}s  prompt_toks={pt}  -> {txt!r}", flush=True)
