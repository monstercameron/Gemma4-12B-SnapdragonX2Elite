"""Pure GPU streaming-read bandwidth (wgpu/Vulkan, Adreno X2). vec4 grid-stride read of a big
buffer -> the GPU macro's achievable peak, to compare against the SoC theoretical."""
import time, numpy as np, wgpu
ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev = ad.request_device_sync()
U = wgpu.BufferUsage
MB = 256; NV = MB*1024*1024//16          # vec4 count
NT = 65536
CODE = f"""
@group(0) @binding(0) var<storage,read> data: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read_write> outb: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
  var acc=vec4<f32>(0.0);
  for (var i=g.x; i<d.x; i=i+{NT}u) {{ acc=acc+data[i]; }}
  outb[g.x]=acc.x+acc.y+acc.z+acc.w;
}}
"""
data = dev.create_buffer(size=NV*16, usage=U.STORAGE)
outb = dev.create_buffer(size=NT*4, usage=U.STORAGE)
du = dev.create_buffer_with_data(data=np.array([NV,0,0,0],np.uint32).tobytes(), usage=U.UNIFORM)
m = dev.create_shader_module(code=CODE)
p = dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto, compute={"module":m,"entry_point":"main"})
bg = dev.create_bind_group(layout=p.get_bind_group_layout(0), entries=[
  {"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([data,outb,du])])
def run():
  e=dev.create_command_encoder(); c=e.begin_compute_pass(); c.set_pipeline(p); c.set_bind_group(0,bg)
  c.dispatch_workgroups(NT//256,1); c.end(); dev.queue.submit([e.finish()])
for _ in range(20): run()
dev.queue.on_submitted_work_done_sync()
R=300; t=time.time()
for _ in range(R): run()
dev.queue.on_submitted_work_done_sync()
ms=(time.time()-t)/R*1e3
print(f"streaming read {MB} MB: {ms:.3f} ms  -> {MB/1024/(ms/1e3):.1f} GB/s achievable", flush=True)
