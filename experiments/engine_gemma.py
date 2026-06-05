"""Gemma 4 12B-it NPU/CPU shard engine — runs the fp16 ONNX shards produced by
export_shards.py through the Hexagon HTP (or CPU), with embeddings + final norm + tied
lm_head + logit-softcap on CPU.

Mirrors the Qwen NpuShardBackend (LRU pool capped at the HTP 4-context limit, shard-major
prefill, token-major decode, rolling-WIN KV) but:
  * shard I/O names: hidden_in/hidden_out, past_key_values.{g}.key/value, present.{g}.*
  * variable head dims per layer handled generically from input metadata
  * a PROPER additive validity mask (-inf on unfilled KV slots) instead of a constant mask,
    so short sequences (<= WIN) reproduce HF's exact causal attention.

Run under the QNN venv for NPU, or any onnxruntime for CPU:
  <pipeline>/.venv/Scripts/python.exe scripts/engine_gemma.py --backend npu "The capital of France is" --ngen 8
  .venv-gemma4/Scripts/python.exe   scripts/engine_gemma.py --backend cpu "..." --ngen 8
"""
import os, time, gc, json, argparse
import numpy as np
import onnxruntime as ort

DIR = os.environ.get("GEMMA_DIR", "out/gemma4_fp16")
MODEL = "models/gemma-4-12B-it"
NEG = np.float16(-30000.0)   # ~ -inf in fp16 for masking


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(f"{MODEL}/tokenizer.json")


class ShardEngine:
    def __init__(self, backend="npu", max_live=4, cpu_tail=0, cpu_shards=None):
        self.meta = json.load(open(f"{DIR}/shards.json"))
        self.win = self.meta["win"]
        self.shards = self.meta["shards"]
        self.H = self.meta["hidden"]
        self.embed_scale = np.float32(self.meta["embed_scale"])
        self.softcap = self.meta.get("final_logit_softcapping")
        self.eps = self.meta.get("rms_norm_eps", 1e-6)
        self.nsh = len(self.shards)
        self.backend = backend
        # Hybrid: run the last `cpu_tail` shards on CPU (fp32-accumulation) where the logit
        # decision is most precision-sensitive; the rest run on the NPU.
        if backend == "cpu":
            self.cpu_set = set(range(self.nsh))
        else:
            self.cpu_set = set(range(self.nsh - cpu_tail, self.nsh))
            if cpu_shards:
                self.cpu_set |= set(cpu_shards)
        self.max_live = min(max_live, self.nsh)
        self.NT = max(1, (os.cpu_count() or 2) - 1)
        self.EMB = np.load(f"{DIR}/embed_tokens.npy", mmap_mode="r")   # [V, H] fp16
        self.NORM = np.asarray(np.load(f"{DIR}/norm.npy"), np.float32)  # [H]
        self.live, self.lru, self.caches, self._meta = {}, [], {}, {}
        if backend in ("npu", "gpu"):
            import onnxruntime_qnn as q
            self.q = q
            ort.register_execution_provider_library("QNNExecutionProvider", q.get_library_path())
            want = ort.OrtHardwareDeviceType.NPU if backend == "npu" else ort.OrtHardwareDeviceType.GPU
            self.qnn_dev = [d for d in ort.get_ep_devices()
                            if d.ep_name == "QNNExecutionProvider" and d.device.type == want][0]
            self.qnn_backend_path = (q.get_qnn_htp_path() if backend == "npu"
                                     else q.get_qnn_gpu_path())
        print(f"[engine] backend={backend} shards={self.nsh} win={self.win} "
              f"max_live={self.max_live} softcap={self.softcap} "
              f"cpu_shards={sorted(self.cpu_set)}", flush=True)

    def _path(self, s):
        # backend-specific cached context binary (HTP ctx_*, GPU ctxgpu_*); CPU uses plain onnx.
        prefix = {"npu": "ctx_", "gpu": "ctxgpu_"}.get(self.backend)
        if prefix and s not in self.cpu_set and os.environ.get("GEMMA_NOCTX") != "1":
            c = f"{DIR}/{prefix}fp16shard_{s}.onnx"
            if os.path.exists(c):
                return c
        return f"{DIR}/fp16shard_{s}.onnx"

    def _session(self, s):
        if s in self.live:
            self.lru.remove(s); self.lru.append(s); return self.live[s]
        while len(self.live) >= self.max_live:
            old = self.lru.pop(0); del self.live[old]; gc.collect()
        so = ort.SessionOptions(); so.intra_op_num_threads = self.NT
        providers = None
        if s not in self.cpu_set:
            if self.backend in ("npu", "gpu"):
                opts = {"backend_path": self.qnn_backend_path}
                if self.backend == "npu":
                    opts["htp_performance_mode"] = "burst"
                so.add_provider_for_devices([self.qnn_dev], opts)
            elif self.backend == "dml":
                providers = ["DmlExecutionProvider"]
        t = time.time()
        sess = (ort.InferenceSession(self._path(s), sess_options=so, providers=providers)
                if providers else ort.InferenceSession(self._path(s), sess_options=so))
        self._loadt = time.time() - t
        self.live[s] = sess; self.lru.append(s)
        if s not in self._meta:
            self._meta[s] = [(i.name, i.type, [d if isinstance(d, int) else 1 for d in i.shape])
                             for i in sess.get_inputs()]
        return sess

    def reset(self):
        self.caches = {}

    def _zero_caches(self, s):
        c = {}
        for n, t, shp in self._meta[s]:
            if n.startswith("past_key_values."):
                c[n] = np.zeros(shp, np.float16)
        return c

    def _mask(self, pos):
        # additive [1,1,1,WIN+1]; for absolute 0-based position `pos` the number of valid
        # slots (real past tokens + current) is min(pos+1, WIN+1), right-aligned in the buffer.
        m = np.full((1, 1, 1, self.win + 1), NEG, np.float16)
        valid = min(pos + 1, self.win + 1)
        m[..., self.win + 1 - valid:] = np.float16(0.0)
        return m

    def _run_shard(self, sess, s, hidden, pos):
        if s not in self.caches:
            self.caches[s] = self._zero_caches(s)
        cache = self.caches[s]
        feeds = {"hidden_in": hidden.astype(np.float16),
                 "position_ids": np.array([[pos]], np.int64),
                 "attention_mask": self._mask(pos)}
        for n, t, shp in self._meta[s]:
            if n.startswith("past_key_values."):
                feeds[n] = cache[n]
        outs = sess.run(None, feeds)
        od = dict(zip([o.name for o in sess.get_outputs()], outs))
        if getattr(self, "trace", None) is not None:
            self.trace[s] = od["hidden_out"].astype(np.float32).copy()   # last write = last token
        for k, v in od.items():
            if k.startswith("present."):
                past = "past_key_values." + k[len("present."):]
                # trim WIN+1 -> WIN along the sequence axis (axis 2), dropping the oldest
                if v.shape[2] == self.win + 1:
                    v = v[:, :, 1:, :]
                cache[past] = v.astype(np.float16)
        return od["hidden_out"]

    def _logits(self, hidden):
        h = hidden.reshape(self.H).astype(np.float32)
        # final RMSNorm (Gemma: (1+weight) is already folded into saved weight? no -> weight as-is)
        rms = h / np.sqrt(np.mean(h * h) + self.eps)
        h = rms * self.NORM
        logits = h @ np.asarray(self.EMB, np.float32).T
        if self.softcap:
            logits = self.softcap * np.tanh(logits / self.softcap)
        return logits

    def _embed(self, token_id):
        return (np.asarray(self.EMB[token_id], np.float32) * self.embed_scale
                ).reshape(1, 1, self.H).astype(np.float16)

    def forward(self, token_id, pos):
        """Token-major: one token through all shards (decode). With the pool capped at 4 < 12,
        this reloads contexts as it cycles — the swap-bound decode path."""
        hidden = self._embed(token_id)
        for s in range(self.nsh):
            hidden = self._run_shard(self._session(s), s, hidden, pos)
        return self._logits(hidden)

    def prefill(self, token_ids):
        """Shard-major: load each shard ONCE, sweep ALL prompt tokens through it (its KV cache
        evolves token-by-token), then move on. 12 context loads total instead of 12xN — the
        right way to amortize the 4-concurrent-context HTP limit. Leaves caches in the
        post-prompt state so decode continues token-major."""
        carries = [self._embed(t) for t in token_ids]
        for s in range(self.nsh):
            sess = self._session(s)
            for pos in range(len(carries)):
                carries[pos] = self._run_shard(sess, s, carries[pos], pos)
        return self._logits(carries[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="The capital of France is")
    ap.add_argument("--backend", choices=["npu", "gpu", "dml", "cpu"], default="npu")
    ap.add_argument("--ngen", type=int, default=8)
    ap.add_argument("--max-live", type=int, default=4)
    ap.add_argument("--cpu-tail", type=int, default=0,
                    help="run the last N shards on CPU (fp32-accum) for precision")
    ap.add_argument("--chat", action="store_true", help="wrap prompt in the Gemma chat template")
    ap.add_argument("--ids", default=None, help="explicit comma-separated input token ids")
    ap.add_argument("--cpu-shards", default=None, help="comma-separated shard indices to force on CPU")
    args = ap.parse_args()
    cpu_shards = [int(x) for x in args.cpu_shards.split(",")] if args.cpu_shards else None

    tok = load_tokenizer()
    if args.ids:
        ids = [int(x) for x in args.ids.split(",")]
    elif args.chat:
        text = f"<start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n"
        ids = tok.encode(text).ids
        if not ids or ids[0] != 2:   # ensure <bos>
            ids = [2] + ids
    else:
        ids = tok.encode(args.prompt).ids
    eng = ShardEngine(backend=args.backend, max_live=args.max_live, cpu_tail=args.cpu_tail,
                      cpu_shards=cpu_shards)
    print(f"prompt={args.prompt!r} -> {len(ids)} tokens: {ids}", flush=True)

    eng.reset()
    t0 = time.time()
    logits = eng.prefill(ids)
    pos = len(ids)
    print(f"prefill {len(ids)} tok in {time.time()-t0:.1f}s; "
          f"step-0 argmax={int(np.argmax(logits))} -> {tok.decode([int(np.argmax(logits))])!r}", flush=True)

    gen, td = [], time.time()
    for step in range(args.ngen):
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        print(f"  tok {step}: {nxt} -> {tok.decode([nxt])!r}", flush=True)
        logits = eng.forward(nxt, pos); pos += 1
    print(f"\nCONTINUATION: {tok.decode(gen)!r}")
    print(f"decode {args.ngen} tok in {time.time()-td:.1f}s = {args.ngen/(time.time()-td):.3f} tok/s", flush=True)


if __name__ == "__main__":
    main()
