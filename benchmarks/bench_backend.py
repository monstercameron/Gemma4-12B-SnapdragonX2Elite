"""Vulkan vs D3D12 driver comparison on the Adreno X2-90 (Windows-on-ARM64), via wgpu's backend
selection (same WGSL kernel, two drivers -> isolates the driver). Tests: (1) does the backend init /
is it stable, (2) pure streaming-read bandwidth GB/s, (3) an int4-GEMV-style kernel GB/s (decode-shaped).
Run: bench_backend.py Vulkan   |   bench_backend.py D3D12   (WGPU_BACKEND_TYPE forced before import)."""
import os, sys, time
BACKEND = (sys.argv[1] if len(sys.argv) > 1 else "Vulkan")
os.environ["WGPU_BACKEND_TYPE"] = BACKEND          # MUST be set before importing wgpu
import numpy as np, wgpu

try:
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = dict(ad.info)
    dev = ad.request_device_sync()
except Exception as e:
    print(f"[{BACKEND}] ADAPTER/DEVICE INIT FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
    sys.exit(2)
print(f"[{BACKEND}] adapter={info.get('device','?')!r} backend={info.get('backend_type','?')} "
      f"type={info.get('adapter_type','?')}", flush=True)
U = wgpu.BufferUsage

def run(shader, bind_bufs, dispatch, bytes_moved, label, R=200):
    mod = dev.create_shader_module(code=shader)
    pipe = dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto, compute={"module": mod, "entry_point": "main"})
    bg = dev.create_bind_group(layout=pipe.get_bind_group_layout(0),
        entries=[{"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}} for i, b in enumerate(bind_bufs)])
    def go():
        e = dev.create_command_encoder(); c = e.begin_compute_pass(); c.set_pipeline(pipe); c.set_bind_group(0, bg)
        c.dispatch_workgroups(dispatch, 1, 1); c.end(); dev.queue.submit([e.finish()])
    for _ in range(20): go()
    dev.queue.on_submitted_work_done_sync()
    t = time.time()
    for _ in range(R): go()
    dev.queue.on_submitted_work_done_sync()
    ms = (time.time() - t) / R * 1e3
    print(f"  {label:22} {ms:7.3f} ms   {bytes_moved/(ms/1e3)/1e9:6.1f} GB/s", flush=True)

# ---- 1) pure streaming-read bandwidth: coalesced read of a big buffer ----
MB = 256; M = MB * 1024 * 1024 // 16          # vec4<u32> count
TG = 1024                                       # workgroups * 64 threads
READ = f"""
@group(0) @binding(0) var<storage,read> data: array<vec4<u32>>;
@group(0) @binding(1) var<storage,read_write> outb: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  let n = arrayLength(&data); var acc = vec4<u32>(0u); var i = g.x;
  loop {{ if (i >= n) {{ break; }} acc = acc + data[i]; i = i + {TG*64}u; }}
  outb[g.x] = acc.x + acc.y + acc.z + acc.w;
}}"""
data = dev.create_buffer(size=M*16, usage=U.STORAGE)
outb = dev.create_buffer(size=TG*64*4, usage=U.STORAGE)
run(READ, [data, outb], TG, M*16, "streaming read")

# ---- 2) int4 GEMV-shaped kernel (decode pattern: unpack int4, scale, accumulate) ----
K, N, BLK = 3840, 15360, 32
GEMV = """
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> wp: array<u32>;
@group(0) @binding(2) var<storage,read> sc: array<u32>;
@group(0) @binding(3) var<storage,read_write> partial: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let Nn=d.x; let nb8=d.y; let bk=d.w; let t=g.x; if (t*8u>=Nn) { return; }
  var acc=0.0;
  for (var b=0u; b<d.z; b=b+1u) { let su=(b*Nn+t*8u)/2u;
    let s0=unpack2x16float(sc[su]);
    let kb=b*bk;
    for (var kk=0u; kk<bk; kk=kk+1u) { let k=kb+kk; let p=wp[k*nb8+t]; let xv=x[k];
      acc=acc+(f32(p&15u)-8.0)*s0.x*xv+(f32((p>>4u)&15u)-8.0)*s0.y*xv; } }
  partial[t]=acc;
}"""
rng = np.random.default_rng(0); nblk = K//BLK
mk = lambda a, u: dev.create_buffer_with_data(data=a.tobytes(), usage=u)
bx = mk((rng.standard_normal(K)*0.1).astype(np.float32), U.STORAGE)
bw = mk(rng.integers(0, 2**32, size=K*(N//8), dtype=np.uint32), U.STORAGE)
bs = mk((np.ones(nblk*N)*0.01).astype(np.float16).view(np.uint32), U.STORAGE)
bp = dev.create_buffer(size=N*4, usage=U.STORAGE)
bd = mk(np.array([N, N//8, nblk, BLK], np.uint32), U.UNIFORM)
run(GEMV, [bx, bw, bs, bp, bd], (N//8+63)//64, K*N//2, "int4 GEMV (weight read)")
print(f"[{BACKEND}] OK", flush=True)
