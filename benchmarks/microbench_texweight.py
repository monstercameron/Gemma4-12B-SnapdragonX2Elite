"""Test the texture-cache weight path (the UBWC / 'texture weights' decode lever from the guide):
read the int4 weights via a 2D texture (textureLoad on r32uint, Adreno texture cache) vs a storage
buffer, same GEMV access pattern. If the texture path streams faster, it's a real decode lever."""
import time, numpy as np, wgpu
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage; TU=wgpu.TextureUsage; BLK=32; WGS=64
K,N=3840,15360; nb8=N//8; SPLIT=24; bpc=(K//BLK)//SPLIT
wp=np.random.default_rng(0).integers(0,2**32,size=K*nb8,dtype=np.uint32)
def common_body(read):  # read: expression reading weight at (k,t)
  return f"""
@group(0) @binding(1) var<storage,read_write> partial: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
@compute @workgroup_size({WGS})
fn main(@builtin(global_invocation_id) g: vec3<u32>){{
  let cols=d.x; let nb=d.y; let bk=d.w; let t=g.x; let c=g.y;
  if(t>=cols){{return;}} var acc=0u; let kb=c*{bpc}u*bk;
  for(var i=0u;i<{bpc}u*bk;i++){{ let k=kb+i; {read} acc=acc+(p&255u); }}
  partial[c*cols+t]=f32(acc);
}}"""
BUF="@group(0) @binding(0) var<storage,read> wp: array<u32>;\n"+common_body("let p=wp[k*nb+t];")
TEX="@group(0) @binding(0) var wp: texture_2d<u32>;\n"+common_body("let p=textureLoad(wp, vec2<i32>(i32(t),i32(k)), 0).x;")
def bench(code, tex):
  bp=dev.create_buffer(size=SPLIT*nb8*4,usage=U.STORAGE)
  d1=dev.create_buffer_with_data(data=np.array([nb8,nb8,bpc,BLK],np.uint32).tobytes(),usage=U.UNIFORM)
  if tex:
    t=dev.create_texture(size=(nb8,K,1),format="r32uint",usage=TU.TEXTURE_BINDING|TU.COPY_DST,dimension="2d")
    dev.queue.write_texture({"texture":t},wp.tobytes(),{"bytes_per_row":nb8*4,"rows_per_image":K},(nb8,K,1))
    res0={"binding":0,"resource":t.create_view()}
  else:
    b=dev.create_buffer_with_data(data=wp.tobytes(),usage=U.STORAGE); res0={"binding":0,"resource":{"buffer":b,"offset":0,"size":b.size}}
  m=dev.create_shader_module(code=code)
  p=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":m,"entry_point":"main"})
  bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[res0,
      {"binding":1,"resource":{"buffer":bp,"offset":0,"size":bp.size}},{"binding":2,"resource":{"buffer":d1,"offset":0,"size":d1.size}}])
  WGx=(nb8+WGS-1)//WGS
  def go():
    e=dev.create_command_encoder();c=e.begin_compute_pass();c.set_pipeline(p);c.set_bind_group(0,bg)
    c.dispatch_workgroups(WGx,SPLIT);c.end();dev.queue.submit([e.finish()])
  for _ in range(10):go()
  dev.queue.on_submitted_work_done_sync();R=200;t0=time.time()
  for _ in range(R):go()
  dev.queue.on_submitted_work_done_sync();ms=(time.time()-t0)/R*1e3
  return ms, (K*nb8*4)/(ms/1e3)/1e9
mb,gb=bench(BUF,False); print(f"buffer  weights: {mb:.3f} ms  {gb:.1f} GB/s",flush=True)
mt,gt=bench(TEX,True);  print(f"texture weights: {mt:.3f} ms  {gt:.1f} GB/s  ({gb/gt:.2f}x {'FASTER buffer' if gb>gt else 'FASTER texture'})",flush=True)
