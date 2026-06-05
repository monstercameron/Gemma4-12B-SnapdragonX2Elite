"""Cache microbench: does staging x (activation) into workgroup shared memory beat re-reading
it from global/L2? Within a workgroup the 64 threads share the same K-slice -> x is read 64x
redundantly. Compare current (x from global) vs x-shared, at each dim's good split."""
import time, numpy as np, wgpu
BLK = 32
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
dev = adapter.request_device_sync(); U = wgpu.BufferUsage

CUR = """
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> wpackT: array<u32>;
@group(0) @binding(2) var<storage,read> scalesT: array<u32>;
@group(0) @binding(3) var<storage,read_write> partial: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let Nn=d.x; let nb8=d.y; let bpc=d.z; let bk=d.w; let t=gid.x; let c=gid.y;
  if(t*8u>=Nn){return;}
  var a0=vec4<f32>(0.0); var a1=vec4<f32>(0.0); let b0=c*bpc;
  for(var bi=0u;bi<bpc;bi=bi+1u){ let b=b0+bi; let su=(b*Nn+t*8u)/2u;
    let s0=vec4<f32>(unpack2x16float(scalesT[su]),unpack2x16float(scalesT[su+1u]));
    let s1=vec4<f32>(unpack2x16float(scalesT[su+2u]),unpack2x16float(scalesT[su+3u])); let kb=b*bk;
    for(var kk=0u;kk<bk;kk=kk+1u){ let k=kb+kk; let p=wpackT[k*nb8+t]; let xv=x[k];
      a0=a0+(vec4<f32>(f32(p&15u),f32((p>>4u)&15u),f32((p>>8u)&15u),f32((p>>12u)&15u))-vec4<f32>(8.0))*s0*xv;
      a1=a1+(vec4<f32>(f32((p>>16u)&15u),f32((p>>20u)&15u),f32((p>>24u)&15u),f32((p>>28u)&15u))-vec4<f32>(8.0))*s1*xv; } }
  let o=c*Nn+t*8u; partial[o]=a0.x;partial[o+1u]=a0.y;partial[o+2u]=a0.z;partial[o+3u]=a0.w;
  partial[o+4u]=a1.x;partial[o+5u]=a1.y;partial[o+6u]=a1.z;partial[o+7u]=a1.w;
}
"""
XSH = """
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> wpackT: array<u32>;
@group(0) @binding(2) var<storage,read> scalesT: array<u32>;
@group(0) @binding(3) var<storage,read_write> partial: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;
var<workgroup> xs: array<f32, 4096>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Nn=d.x; let nb8=d.y; let bpc=d.z; let bk=d.w; let t=gid.x; let c=gid.y; let tid=lid.x;
  let slice=bpc*bk; let base=c*slice;
  for(var i=tid;i<slice;i=i+64u){ xs[i]=x[base+i]; }
  workgroupBarrier();
  if(t*8u>=Nn){return;}
  var a0=vec4<f32>(0.0); var a1=vec4<f32>(0.0); let b0=c*bpc;
  for(var bi=0u;bi<bpc;bi=bi+1u){ let b=b0+bi; let su=(b*Nn+t*8u)/2u;
    let s0=vec4<f32>(unpack2x16float(scalesT[su]),unpack2x16float(scalesT[su+1u]));
    let s1=vec4<f32>(unpack2x16float(scalesT[su+2u]),unpack2x16float(scalesT[su+3u]));
    for(var kk=0u;kk<bk;kk=kk+1u){ let li=bi*bk+kk; let k=b*bk+kk; let p=wpackT[k*nb8+t]; let xv=xs[li];
      a0=a0+(vec4<f32>(f32(p&15u),f32((p>>4u)&15u),f32((p>>8u)&15u),f32((p>>12u)&15u))-vec4<f32>(8.0))*s0*xv;
      a1=a1+(vec4<f32>(f32((p>>16u)&15u),f32((p>>20u)&15u),f32((p>>24u)&15u),f32((p>>28u)&15u))-vec4<f32>(8.0))*s1*xv; } }
  let o=c*Nn+t*8u; partial[o]=a0.x;partial[o+1u]=a0.y;partial[o+2u]=a0.z;partial[o+3u]=a0.w;
  partial[o+4u]=a1.x;partial[o+5u]=a1.y;partial[o+6u]=a1.z;partial[o+7u]=a1.w;
}
"""

def bench(code, K, N, split):
    nblk=K//BLK; bpc=nblk//split; rng=np.random.default_rng(0)
    x=(rng.standard_normal(K)*0.1).astype(np.float32); wp=rng.integers(0,2**32,size=K*(N//8),dtype=np.uint32)
    sc=(np.ones(nblk*N)*0.01).astype(np.float16).view(np.uint32)
    mk=lambda a,u:dev.create_buffer_with_data(data=a.tobytes(),usage=u)
    bx=mk(x,U.STORAGE);bw=mk(wp,U.STORAGE);bs=mk(sc,U.STORAGE)
    bp=dev.create_buffer(size=split*N*4,usage=U.STORAGE); d1=mk(np.array([N,N//8,bpc,BLK],np.uint32),U.UNIFORM)
    m=dev.create_shader_module(code=code); p=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":m,"entry_point":"main"})
    bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bx,bw,bs,bp,d1])])
    WGx=(N//8+63)//64
    def run(): e=dev.create_command_encoder();c=e.begin_compute_pass();c.set_pipeline(p);c.set_bind_group(0,bg);c.dispatch_workgroups(WGx,split);c.end();dev.queue.submit([e.finish()])
    for _ in range(10):run()
    dev.queue.on_submitted_work_done_sync(); R=200; t=time.time()
    for _ in range(R):run()
    dev.queue.on_submitted_work_done_sync(); ms=(time.time()-t)/R*1e3
    return K*N//2/(ms/1e3)/1e9

for name,(K,N,sp) in {"down K15360 N3840 split24":(15360,3840,24),"gate K3840 N15360 split8":(3840,15360,8),
                       "q K3840 N4096 split24":(3840,4096,24)}.items():
    if (K//BLK)*sp*4 > 4096*4 and (K//BLK//sp)*BLK > 4096:
        print(f"{name}: slice too big for 4096 shared, skip"); continue
    g0=bench(CUR,K,N,sp); g1=bench(XSH,K,N,sp)
    print(f"{name:28} global={g0:6.1f}  x-shared={g1:6.1f} GB/s  ({(g1/g0-1)*100:+.0f}%)", flush=True)
