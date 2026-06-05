"""int4 GPU GEMV (wgpu/WGSL), 2-pass (partial + reduce) — coalesced weight reads (consecutive
threads -> consecutive memory) give ~112 GB/s on the Adreno X2. Shared scratch buffers (bx/by/bp)
avoid thousands of allocations. Coalesced int4 layout: wpackT[K,N/8] u32 + fp16 scales[nblk,N]."""
import numpy as np, wgpu

GEMV = """
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> wpackT: array<u32>;
@group(0) @binding(2) var<storage, read> scalesT: array<u32>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;   // N, NB8, BPC, BLK
@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let Nn=d.x; let nb8=d.y; let bpc=d.z; let bk=d.w;
  let t=gid.x; let c=gid.y;
  if (t*8u>=Nn) { return; }
  var a0=vec4<f32>(0.0); var a1=vec4<f32>(0.0);
  let b0=c*bpc;
  for (var bi=0u; bi<bpc; bi=bi+1u) {
    let b=b0+bi;
    let su=(b*Nn + t*8u)/2u;
    let s0=vec4<f32>(unpack2x16float(scalesT[su]), unpack2x16float(scalesT[su+1u]));
    let s1=vec4<f32>(unpack2x16float(scalesT[su+2u]), unpack2x16float(scalesT[su+3u]));
    let kb=b*bk;
    for (var kk=0u; kk<bk; kk=kk+1u) {
      let k=kb+kk; let p=wpackT[k*nb8+t]; let xv=x[k];
      a0=a0+(vec4<f32>(f32(p&15u),f32((p>>4u)&15u),f32((p>>8u)&15u),f32((p>>12u)&15u))-vec4<f32>(8.0))*s0*xv;
      a1=a1+(vec4<f32>(f32((p>>16u)&15u),f32((p>>20u)&15u),f32((p>>24u)&15u),f32((p>>28u)&15u))-vec4<f32>(8.0))*s1*xv;
    }
  }
  let o=c*Nn+t*8u;
  partial[o]=a0.x;partial[o+1u]=a0.y;partial[o+2u]=a0.z;partial[o+3u]=a0.w;
  partial[o+4u]=a1.x;partial[o+5u]=a1.y;partial[o+6u]=a1.z;partial[o+7u]=a1.w;
}
""".replace("@workgroup_size(128)", "@workgroup_size(64)")   # Adreno: 64 = 1 wave -> more workgroup-rows
REDUCE = """
@group(0) @binding(0) var<storage,read> partial: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n=gid.x; if(n>=d.x){return;}
  var s=0.0; for(var c=0u;c<d.y;c=c+1u){s=s+partial[c*d.x+n];} y[n]=s;
}
"""


def quant_coalesced(W, blk=128):
    W = np.ascontiguousarray(W, np.float32); N, K = W.shape
    assert K % blk == 0 and N % 8 == 0, f"K={K} N={N} not aligned"
    nblk = K // blk
    Wb = W.reshape(N, nblk, blk)
    scale = np.maximum(np.abs(Wb).max(axis=2) / 7.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(Wb / scale[:, :, None]) + 8, 0, 15).astype(np.uint8).reshape(N, K)
    qT = np.ascontiguousarray(q.T)
    wpackT = np.zeros((K, N // 8), np.uint32)
    for j in range(8):
        wpackT |= (qT[:, j::8].astype(np.uint32) & 15) << (j * 4)
    scalesT = np.ascontiguousarray(scale.T).astype(np.float16)
    return wpackT.reshape(-1), scalesT.reshape(-1).view(np.uint32), nblk


class WgpuEngine:
    def __init__(self, maxK=16384, maxN=262144, maxsplit=8):
        self.dev = wgpu.gpu.request_adapter_sync(power_preference="high-performance").request_device_sync()
        self.pp = self._pipe(GEMV); self.rp = self._pipe(REDUCE)
        U = wgpu.BufferUsage; dev = self.dev
        self.bx = dev.create_buffer(size=maxK * 4, usage=U.STORAGE | U.COPY_DST)
        self.by = dev.create_buffer(size=maxN * 4, usage=U.STORAGE | U.COPY_SRC)
        self.bp = dev.create_buffer(size=maxsplit * maxN * 4, usage=U.STORAGE)

    def _pipe(self, code):
        m = self.dev.create_shader_module(code=code)
        return self.dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,
                                                compute={"module": m, "entry_point": "main"})


class Int4Linear:
    def __init__(self, eng, W, K, N, blk=32, split=8):
        self.e = eng; self.K = K; self.N = N
        nblk = K // blk
        # Adreno-tuned adaptive K-split: target ~192 total workgroups for occupancy.
        # Small-N matmuls (few output-group rows) need a big split; large-N are already saturated.
        rows = (N // 8 + 63) // 64
        want = max(1, -(-192 // max(rows, 1)))
        divs = [s for s in range(1, nblk + 1) if nblk % s == 0]
        self.split = min(divs, key=lambda s: (abs(s - want), s))
        self.bpc = nblk // self.split
        wp, sp, _ = quant_coalesced(W, blk)
        U = wgpu.BufferUsage; dev = eng.dev
        mk = lambda a, u: dev.create_buffer_with_data(data=a.tobytes(), usage=u)
        self.bw = mk(wp, U.STORAGE); self.bs = mk(sp, U.STORAGE)
        self.d1 = mk(np.array([N, N // 8, self.bpc, blk], np.uint32), U.UNIFORM)
        self.d2 = mk(np.array([N, self.split, 0, 0], np.uint32), U.UNIFORM)
        self.WGx = (N // 8 + 63) // 64

    def make_chain_bg(self, in_buf, out_buf, in_off=0, out_off=0):
        dev = self.e.dev
        bg1 = dev.create_bind_group(layout=self.e.pp.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": in_buf, "offset": in_off, "size": self.K * 4}},
            {"binding": 1, "resource": {"buffer": self.bw, "offset": 0, "size": self.bw.size}},
            {"binding": 2, "resource": {"buffer": self.bs, "offset": 0, "size": self.bs.size}},
            {"binding": 3, "resource": {"buffer": self.e.bp, "offset": 0, "size": self.split * self.N * 4}},
            {"binding": 4, "resource": {"buffer": self.d1, "offset": 0, "size": self.d1.size}}])
        bg2 = dev.create_bind_group(layout=self.e.rp.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": self.e.bp, "offset": 0, "size": self.split * self.N * 4}},
            {"binding": 1, "resource": {"buffer": out_buf, "offset": out_off, "size": self.N * 4}},
            {"binding": 2, "resource": {"buffer": self.d2, "offset": 0, "size": self.d2.size}}])
        return bg1, bg2

    def chain(self, enc, bg1, bg2):
        c1 = enc.begin_compute_pass(); c1.set_pipeline(self.e.pp); c1.set_bind_group(0, bg1)
        c1.dispatch_workgroups(self.WGx, self.split); c1.end()
        c2 = enc.begin_compute_pass(); c2.set_pipeline(self.e.rp); c2.set_bind_group(0, bg2)
        c2.dispatch_workgroups((self.N + 63) // 64); c2.end()

    def make_partial_bg(self, in_buf, bp_buf, in_off=0, bp_off=0):
        """Partial-only GEMV -> bp_buf[bp_off] holds [split, N] partials (no reduce pass).
        The consumer reduces inline. Lets us fuse the reduce into norm_add / gelu*up."""
        dev = self.e.dev
        return dev.create_bind_group(layout=self.e.pp.get_bind_group_layout(0), entries=[
            {"binding": 0, "resource": {"buffer": in_buf, "offset": in_off, "size": self.K * 4}},
            {"binding": 1, "resource": {"buffer": self.bw, "offset": 0, "size": self.bw.size}},
            {"binding": 2, "resource": {"buffer": self.bs, "offset": 0, "size": self.bs.size}},
            {"binding": 3, "resource": {"buffer": bp_buf, "offset": bp_off, "size": self.split * self.N * 4}},
            {"binding": 4, "resource": {"buffer": self.d1, "offset": 0, "size": self.d1.size}}])

    def chain_partial(self, enc, bg):
        c = enc.begin_compute_pass(); c.set_pipeline(self.e.pp); c.set_bind_group(0, bg)
        c.dispatch_workgroups(self.WGx, self.split); c.end()

    def forward(self, x):
        dev = self.e.dev
        dev.queue.write_buffer(self.e.bx, 0, np.ascontiguousarray(x, np.float32).tobytes())
        enc = dev.create_command_encoder(); self.chain(enc, *self.make_chain_bg(self.e.bx, self.e.by))
        dev.queue.submit([enc.finish()])
        return np.frombuffer(dev.queue.read_buffer(self.e.by, 0, self.N * 4), np.float32).copy()
