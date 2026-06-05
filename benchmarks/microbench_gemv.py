"""Decompose the int4 GEMV kernel into atomic components and micro-benchmark each on the
Adreno X2 (wgpu/Vulkan), synthetic data — no model load. Reveals where bandwidth goes and the
best (WGS, split) so we can rebuild the kernel from measured ground truth.

Components (each strips one layer of work):
  L0 load_only   : sum raw weight u32                      -> pure read bandwidth CEILING
  L1 +unpack     : + extract 8 int4 nibbles                -> ALU unpack cost
  L2 +scale      : + unpack2x16float scales                -> scale-load cost
  L3 full GEMV   : + (q-8)*scale*x accumulate (real kernel)
Then sweep WGS in {64,128,256} and split in {4,8,16} on the full kernel.
"""
import time, numpy as np, wgpu

# representative dims: down_proj (big K) and gate_proj (big N)
DIMS = {"down K15360 N3840": (15360, 3840), "gate K3840 N15360": (3840, 15360),
        "lmhead K3840 N262144": (3840, 262144)}
BLK = 32
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
dev = adapter.request_device_sync()
U = wgpu.BufferUsage
print("adapter:", adapter.info.get("device"), "| wave", adapter.limits.get("min-subgroup-size", "?"), flush=True)


def body(level):
    # the inner per-(k) work, stripped to `level`
    if level == 0:  # load only
        return "acc = acc + f32(wpackT[k*nb8+t]);"
    if level == 1:  # + unpack nibbles
        return "let p=wpackT[k*nb8+t]; acc = acc + f32((p&15u)+((p>>4u)&15u)+((p>>8u)&15u)+((p>>12u)&15u));"
    if level == 2:  # + scale unpack (load scales each block-start handled outside)
        return "let p=wpackT[k*nb8+t]; acc = acc + s0.x*f32(p&15u);"
    return ("let p=wpackT[k*nb8+t]; let xv=x[k];"
            "acc = acc + (f32(p&15u)-8.0)*s0.x*xv + (f32((p>>4u)&15u)-8.0)*s0.y*xv;")


def shader(level, wgs, split):
    return f"""
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> wpackT: array<u32>;
@group(0) @binding(2) var<storage,read> scalesT: array<u32>;
@group(0) @binding(3) var<storage,read_write> partial: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;
@compute @workgroup_size({wgs})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let Nn=d.x; let nb8=d.y; let bpc=d.z; let bk=d.w;
  let t=gid.x; let c=gid.y;
  if (t*8u>=Nn) {{ return; }}
  var acc=0.0; let b0=c*bpc;
  if (t==4294967295u) {{ acc=acc+x[0]+f32(scalesT[0]); }}
  for (var bi=0u; bi<bpc; bi=bi+1u) {{
    let b=b0+bi; let su=(b*Nn + t*8u)/2u;
    let s0=vec4<f32>(unpack2x16float(scalesT[su]), unpack2x16float(scalesT[su+1u]));
    let kb=b*bk;
    for (var kk=0u; kk<bk; kk=kk+1u) {{ let k=kb+kk; {body(level)} }}
  }}
  partial[c*((Nn+7u)/8u)+t]=acc;
}}
"""


def bench(K, N, level, wgs, split):
    nblk = K // BLK; bpc = nblk // split
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(K) * 0.1).astype(np.float32)
    wpack = rng.integers(0, 2**32, size=K * (N // 8), dtype=np.uint32)
    scales = (np.ones(nblk * N) * 0.01).astype(np.float16).view(np.uint32)
    mk = lambda a, u: dev.create_buffer_with_data(data=a.tobytes(), usage=u)
    bx = mk(x, U.STORAGE); bw = mk(wpack, U.STORAGE); bs = mk(scales, U.STORAGE)
    bp = dev.create_buffer(size=split * (N // 8 + 1) * 4, usage=U.STORAGE)
    d1 = mk(np.array([N, N // 8, bpc, BLK], np.uint32), U.UNIFORM)
    m = dev.create_shader_module(code=shader(level, wgs, split))
    p = dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto, compute={"module": m, "entry_point": "main"})
    bg = dev.create_bind_group(layout=p.get_bind_group_layout(0), entries=[
        {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate([bx, bw, bs, bp, d1])])
    WGx = (N // 8 + wgs - 1) // wgs

    def run():
        e = dev.create_command_encoder(); c = e.begin_compute_pass(); c.set_pipeline(p); c.set_bind_group(0, bg)
        c.dispatch_workgroups(WGx, split); c.end(); dev.queue.submit([e.finish()])
    for _ in range(10): run()
    dev.queue.on_submitted_work_done_sync()
    R = 200; t = time.time()
    for _ in range(R): run()
    dev.queue.on_submitted_work_done_sync()
    ms = (time.time() - t) / R * 1e3
    wbytes = K * N // 2   # int4 weight bytes
    return ms, wbytes / (ms / 1e3) / 1e9


for name, (K, N) in DIMS.items():
    print(f"\n=== {name} ===", flush=True)
    # NB: L3 sums only 2 of the 8 packed nibbles (a bandwidth probe -- the full u32 is still read, so
    # GB/s is valid, but it is NOT the full kernel's ALU). microbench_total.py does the full 8-lane GEMV.
    for lvl, lab in [(0, "L0 load-only (ceiling)"), (1, "L1 +unpack"), (2, "L2 +scale"), (3, "L3 GEMV (2/8 lanes)")]:
        ms, gb = bench(K, N, lvl, 128, 8)
        print(f"  {lab:24} {ms:7.3f} ms  {gb:6.1f} GB/s", flush=True)
    print("  -- full-GEMV split sweep (WGS=64) --", flush=True)
    nblk = K // BLK
    best = (0, 0)
    for sp in (1, 2, 4, 8, 12, 16, 20, 24, 30, 40, 48, 60, 80, 120):
        if nblk % sp: continue
        _, gb = bench(K, N, 3, 64, sp)
        print(f"    split={sp:3} -> {gb:6.1f} GB/s", flush=True)
        if gb > best[0]: best = (gb, sp)
    print(f"  >>> best: split={best[1]} @ {best[0]:.1f} GB/s", flush=True)
