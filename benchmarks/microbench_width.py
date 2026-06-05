"""Does a WIDER weight load instruction give more effective bandwidth on Adreno? Isolate load width:
each thread reads + unpacks N int4 weights via a 32-bit (u32, 8 outs), 64-bit (uvec2, 16 outs) or
128-bit (uvec4, 32 outs) load. Pure load+unpack (no scale/x) -> measures the memory path only.
Same total bytes; only the per-thread load width + accumulator count (register pressure) change."""
import time, numpy as np, wgpu
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage; BLK=32
DIMS={"down K15360 N3840":(15360,3840),"gate K3840 N15360":(3840,15360),"lmhead K3840 N262144":(3840,262144)}
# OUT = ints per thread; LW = u32 words per load
SH={
 "u32  (32b,8out)":  ("u32","let p=wp[k*nb+t]; var a=0u; for(var b=0u;b<8u;b++){a+=(p>>(b*4u))&15u;} acc+=f32(a);",8,1),
 "uvec2(64b,16out)": ("vec2<u32>","let p=wp[k*nb+t]; var a=0u; for(var b=0u;b<8u;b++){a+=(p.x>>(b*4u))&15u;a+=(p.y>>(b*4u))&15u;} acc+=f32(a);",16,2),
 "uvec4(128b,32out)":("vec4<u32>","let p=wp[k*nb+t]; var a=0u; for(var b=0u;b<8u;b++){a+=(p.x>>(b*4u))&15u;a+=(p.y>>(b*4u))&15u;a+=(p.z>>(b*4u))&15u;a+=(p.w>>(b*4u))&15u;} acc+=f32(a);",32,4),
}
def shader(ty,body,wgs):
  return f"""
@group(0) @binding(0) var<storage,read> wp: array<{ty}>;
@group(0) @binding(1) var<storage,read_write> partial: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
@compute @workgroup_size({wgs})
fn main(@builtin(global_invocation_id) g: vec3<u32>){{
  let cols=d.x; let nb=d.y; let bpc=d.z; let bk=d.w; let t=g.x; let c=g.y;
  if(t>=cols){{return;}} var acc=0.0; let b0=c*bpc;
  for(var bi=0u;bi<bpc;bi++){{ let kb=(b0+bi)*bk; for(var kk=0u;kk<bk;kk++){{let k=kb+kk; {body} }} }}
  partial[c*cols+t]=acc;
}}"""
def bench(K,N,ty,body,OUT,LW,split,wgs=64):
  nblk=K//BLK; bpc=nblk//split; cols=N//OUT; nb=N//(8*LW)  # words-per-krow for this width
  rng=np.random.default_rng(0)
  wp=rng.integers(0,2**32,size=K*(N//8),dtype=np.uint32)
  mk=lambda a,u:dev.create_buffer_with_data(data=a.tobytes(),usage=u)
  bw=mk(wp,U.STORAGE); bp=dev.create_buffer(size=split*cols*4,usage=U.STORAGE)
  d1=mk(np.array([cols,nb,bpc,BLK],np.uint32),U.UNIFORM)
  m=dev.create_shader_module(code=shader(ty,body,wgs))
  p=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":m,"entry_point":"main"})
  bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bw,bp,d1])])
  WGx=(cols+wgs-1)//wgs
  def run():
    e=dev.create_command_encoder();c=e.begin_compute_pass();c.set_pipeline(p);c.set_bind_group(0,bg)
    c.dispatch_workgroups(WGx,split);c.end();dev.queue.submit([e.finish()])
  for _ in range(10):run()
  dev.queue.on_submitted_work_done_sync(); R=200; t=time.time()
  for _ in range(R):run()
  dev.queue.on_submitted_work_done_sync(); ms=(time.time()-t)/R*1e3
  return ms, (K*N//2)/(ms/1e3)/1e9
for name,(K,N) in DIMS.items():
  print(f"\n=== {name} ===",flush=True)
  for lab,(ty,body,OUT,LW) in SH.items():
    best=(0,0)
    for sp in (8,16,24,48,96):
      if (K//BLK)%sp:continue
      try: ms,gb=bench(K,N,ty,body,OUT,LW,sp)
      except Exception as ex: print(f"  {lab}: ERR {ex}");break
      if gb>best[0]:best=(gb,sp)
    print(f"  {lab:18} best {best[0]:6.1f} GB/s @ split={best[1]}",flush=True)
