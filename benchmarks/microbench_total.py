"""TRUE BW utilization: time BOTH passes (gemv writing partials + reduce reading them back) and
report useful-BW (weight bytes / total time) vs total-BW (all bytes / total time). Sweep split to
find the split that MINIMIZES total time -- the real optimum once partial round-trip is counted."""
import time, numpy as np, wgpu
DIMS = {"down K15360 N3840":(15360,3840), "gate K3840 N15360":(3840,15360),
        "qkv  K3840 N5120":(3840,5120), "lmhead K3840 N262144":(3840,262144)}
BLK=32; WGS=64
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage
GEMV="""
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> wp: array<u32>;
@group(0) @binding(2) var<storage,read> sc: array<u32>;
@group(0) @binding(3) var<storage,read_write> partial: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;
@compute @workgroup_size(%d)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let N=d.x; let nb8=d.y; let bpc=d.z; let bk=d.w;
  let t=g.x; let c=g.y; if(t*8u>=N){return;}
  var a0=0.0;var a1=0.0;var a2=0.0;var a3=0.0;var a4=0.0;var a5=0.0;var a6=0.0;var a7=0.0;
  let b0=c*bpc;
  for(var bi=0u;bi<bpc;bi=bi+1u){
    let b=b0+bi; let su=(b*N+t*8u)/2u;
    let q0=unpack2x16float(sc[su]);let q1=unpack2x16float(sc[su+1u]);
    let q2=unpack2x16float(sc[su+2u]);let q3=unpack2x16float(sc[su+3u]);
    let kb=b*bk;
    for(var kk=0u;kk<bk;kk=kk+1u){let k=kb+kk; let p=wp[k*nb8+t]; let xv=x[k];
      a0=a0+(f32(p&15u)-8.0)*q0.x*xv; a1=a1+(f32((p>>4u)&15u)-8.0)*q0.y*xv;
      a2=a2+(f32((p>>8u)&15u)-8.0)*q1.x*xv; a3=a3+(f32((p>>12u)&15u)-8.0)*q1.y*xv;
      a4=a4+(f32((p>>16u)&15u)-8.0)*q2.x*xv; a5=a5+(f32((p>>20u)&15u)-8.0)*q2.y*xv;
      a6=a6+(f32((p>>24u)&15u)-8.0)*q3.x*xv; a7=a7+(f32((p>>28u)&15u)-8.0)*q3.y*xv;}}
  let row=(N+7u)/8u; let o=c*N+t*8u;
  partial[c*N+t*8u]=a0;partial[o+1u]=a1;partial[o+2u]=a2;partial[o+3u]=a3;
  partial[o+4u]=a4;partial[o+5u]=a5;partial[o+6u]=a6;partial[o+7u]=a7;
}"""
RED="""
@group(0) @binding(0) var<storage,read> partial: array<f32>;
@group(0) @binding(1) var<storage,read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let N=d.x; let sp=d.y; let i=g.x; if(i>=N){return;}
  var a=0.0; for(var c=0u;c<sp;c=c+1u){a=a+partial[c*N+i];} y[i]=a;
}"""
def bench(K,N,split):
  nblk=K//BLK; bpc=nblk//split
  rng=np.random.default_rng(0)
  x=(rng.standard_normal(K)*0.1).astype(np.float32)
  wp=rng.integers(0,2**32,size=K*(N//8),dtype=np.uint32)
  sc=(np.ones(nblk*N)*0.01).astype(np.float16).view(np.uint32)
  mk=lambda a,u:dev.create_buffer_with_data(data=a.tobytes(),usage=u)
  bx=mk(x,U.STORAGE);bw=mk(wp,U.STORAGE);bs=mk(sc,U.STORAGE)
  bp=dev.create_buffer(size=split*N*4,usage=U.STORAGE)
  by=dev.create_buffer(size=N*4,usage=U.STORAGE)
  d1=mk(np.array([N,N//8,bpc,BLK],np.uint32),U.UNIFORM)
  d2=mk(np.array([N,split,0,0],np.uint32),U.UNIFORM)
  m1=dev.create_shader_module(code=GEMV%WGS)
  p1=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":m1,"entry_point":"main"})
  m2=dev.create_shader_module(code=RED)
  p2=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":m2,"entry_point":"main"})
  bg1=dev.create_bind_group(layout=p1.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bx,bw,bs,bp,d1])])
  bg2=dev.create_bind_group(layout=p2.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bp,by,d2])])
  WGx=(N//8+WGS-1)//WGS
  def run():
    e=dev.create_command_encoder()
    c=e.begin_compute_pass();c.set_pipeline(p1);c.set_bind_group(0,bg1);c.dispatch_workgroups(WGx,split);c.end()
    c=e.begin_compute_pass();c.set_pipeline(p2);c.set_bind_group(0,bg2);c.dispatch_workgroups((N+63)//64,1);c.end()
    dev.queue.submit([e.finish()])
  for _ in range(10):run()
  dev.queue.on_submitted_work_done_sync()
  R=200;t=time.time()
  for _ in range(R):run()
  dev.queue.on_submitted_work_done_sync()
  ms=(time.time()-t)/R*1e3
  wb=K*N//2; tot=wb+2*split*N*4
  return ms, wb/(ms/1e3)/1e9, tot/(ms/1e3)/1e9
for name,(K,N) in DIMS.items():
  print(f"\n=== {name} ===  (useful=weights/time, total=all-bytes/time)",flush=True)
  nblk=K//BLK; best=(1e9,0)
  for sp in (1,2,4,6,8,12,16,24,32,48,64,96,120):
    if nblk%sp:continue
    ms,uf,to=bench(K,N,sp)
    mark=""
    if ms<best[0]:best=(ms,sp);mark=" <"
    print(f"  split={sp:3}  {ms:7.3f} ms  useful={uf:6.1f}  total={to:6.1f} GB/s{mark}",flush=True)
  print(f"  >>> FASTEST total time: split={best[1]} @ {best[0]:.3f} ms",flush=True)
