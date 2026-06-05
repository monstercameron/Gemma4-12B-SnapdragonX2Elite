"""Re-derive optimal split for the SHIPPED uvec2 GEMV (16 outputs/thread, N/16 dispatch) + vec4
reduce, total two-pass time. Checks whether the current heuristic (rows=N//8) is mis-tuned now that
the kernel dispatches N//16 threads. Reports best split vs what the heuristic currently picks."""
import time, numpy as np, wgpu
DIMS={"down N3840":(15360,3840),"gate N15360":(3840,15360),"qkv N5120":(3840,5120)}
BLK=32; WGS=64
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage
GEMV="""
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> wp: array<vec2<u32>>;
@group(0) @binding(2) var<storage,read_write> partial: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let N=d.x; let nb16=d.y/2u; let bpc=d.z; let bk=d.w; let t=g.x; let c=g.y;
  if(t*16u>=N){return;}
  var a:array<f32,16>; for(var i=0u;i<16u;i++){a[i]=0.0;}
  let b0=c*bpc;
  for(var bi=0u;bi<bpc;bi++){ let b=b0+bi; let su=(b*N+t*16u)/2u;
    let kb=b*bk;
    for(var kk=0u;kk<bk;kk++){ let k=kb+kk; let p=wp[k*nb16+t]; let xv=x[k];
      for(var j=0u;j<8u;j++){ a[j]+=(f32((p.x>>(j*4u))&15u)-8.0)*xv; a[j+8u]+=(f32((p.y>>(j*4u))&15u)-8.0)*xv; } } }
  let o=c*N+t*16u; for(var i=0u;i<16u;i++){ partial[o+i]=a[i]; }
}"""
RED="""
@group(0) @binding(0) var<storage,read> partial: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> y: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let N4=d.x/4u; let i=g.x; if(i>=N4){return;} var s=vec4<f32>(0.0);
  for(var c=0u;c<d.y;c++){s=s+partial[c*N4+i];} y[i]=s; }"""
def bench(K,N,split):
  nblk=K//BLK; bpc=nblk//split; rng=np.random.default_rng(0)
  x=(rng.standard_normal(K)*0.1).astype(np.float32)
  wp=rng.integers(0,2**32,size=K*(N//8),dtype=np.uint32)
  scb=(np.ones(nblk*N)*0.01).astype(np.float16).view(np.uint32)
  mk=lambda a,u:dev.create_buffer_with_data(data=a.tobytes(),usage=u)
  bx=mk(x,U.STORAGE);bw=mk(wp,U.STORAGE);bs=mk(scb,U.STORAGE)
  bp=dev.create_buffer(size=split*N*4,usage=U.STORAGE);by=dev.create_buffer(size=N*4,usage=U.STORAGE)
  d1=mk(np.array([N,N//8,bpc,BLK],np.uint32),U.UNIFORM);d2=mk(np.array([N,split,0,0],np.uint32),U.UNIFORM)
  p1=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=GEMV),"entry_point":"main"})
  p2=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=RED),"entry_point":"main"})
  bg1=dev.create_bind_group(layout=p1.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bx,bw,bp,d1])])
  bg2=dev.create_bind_group(layout=p2.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bp,by,d2])])
  WGx=(N//16+WGS-1)//WGS
  def run():
    e=dev.create_command_encoder()
    c=e.begin_compute_pass();c.set_pipeline(p1);c.set_bind_group(0,bg1);c.dispatch_workgroups(WGx,split);c.end()
    c=e.begin_compute_pass();c.set_pipeline(p2);c.set_bind_group(0,bg2);c.dispatch_workgroups((N//4+63)//64,1);c.end()
    dev.queue.submit([e.finish()])
  for _ in range(10):run()
  dev.queue.on_submitted_work_done_sync();R=200;t=time.time()
  for _ in range(R):run()
  dev.queue.on_submitted_work_done_sync();return (time.time()-t)/R*1e3
def heur(N):
  rows=(N//8+63)//64; want=max(1,-(-384//max(rows,1))); nblk=3840//BLK if N>4096 else (15360//BLK if False else None)
  return want
for name,(K,N) in DIMS.items():
  nblk=K//BLK; best=(1e9,0)
  print(f"\n=== {name} (K={K}) ===",flush=True)
  for sp in (4,6,8,12,16,24,32,48,60,96,120):
    if nblk%sp:continue
    ms=bench(K,N,sp); mark=""
    if ms<best[0]:best=(ms,sp);mark=" <"
    print(f"  split={sp:3} {ms:7.3f} ms{mark}",flush=True)
  rows=(N//8+63)//64; want=max(1,-(-384//max(rows,1)))
  pick=min([s for s in range(1,nblk+1) if nblk%s==0],key=lambda s:(abs(s-want),s))
  print(f"  >>> best={best[1]} ({best[0]:.3f}ms)  heuristic_picks={pick}",flush=True)
