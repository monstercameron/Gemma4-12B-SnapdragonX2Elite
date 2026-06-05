"""Prefill fix core: does a batched int4 GEMM amortize the weight read on Adreno? Y[M,N]=X[M,K]@deq(W).
Register-tiled: each thread does M_T tokens x 8 outputs, loading each weight u32 ONCE and applying to
all M_T tokens. Compare time-per-token at M=1 (=current GEMV prefill) vs M=8/32/64. If per-token time
collapses as M grows, batching works -> prefill goes from O(weights*N) to ~O(weights). Validate vs numpy."""
import time, numpy as np, wgpu
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage; BLK=32; WGS=64
DIMS={"down K15360 N3840":(15360,3840),"gate K3840 N15360":(3840,15360)}
def shader(MT):
  return f"""
@group(0) @binding(0) var<storage,read> x: array<f32>;       // [M,K]
@group(0) @binding(1) var<storage,read> wp: array<u32>;      // [K,N/8]
@group(0) @binding(2) var<storage,read> sc: array<u32>;      // [nblk,N] fp16
@group(0) @binding(3) var<storage,read_write> y: array<f32>; // [M,N]
@group(0) @binding(4) var<uniform> d: vec4<u32>;             // N, K, M, nb8
@compute @workgroup_size({WGS})
fn main(@builtin(global_invocation_id) g: vec3<u32>){{
  let N=d.x; let K=d.y; let M=d.z; let nb8=d.w; let t=g.x; let m0=g.y*{MT}u;
  if(t*8u>=N){{return;}}
  var acc: array<vec4<f32>,{2*MT}>;
  for(var i=0u;i<{2*MT}u;i++){{ acc[i]=vec4<f32>(0.0); }}
  for(var b=0u;b<K/{BLK}u;b++){{ let su=(b*N+t*8u)/2u;
    let s0=vec4<f32>(unpack2x16float(sc[su]),unpack2x16float(sc[su+1u]));
    let s1=vec4<f32>(unpack2x16float(sc[su+2u]),unpack2x16float(sc[su+3u]));
    for(var kk=0u;kk<{BLK}u;kk++){{ let k=b*{BLK}u+kk; let p=wp[k*nb8+t];
      let lo=vec4<f32>(f32(p&15u),f32((p>>4u)&15u),f32((p>>8u)&15u),f32((p>>12u)&15u))-vec4<f32>(8.0);
      let hi=vec4<f32>(f32((p>>16u)&15u),f32((p>>20u)&15u),f32((p>>24u)&15u),f32((p>>28u)&15u))-vec4<f32>(8.0);
      for(var mt=0u;mt<{MT}u;mt++){{ let m=m0+mt; if(m<M){{ let xv=x[m*K+k];
        acc[mt*2u]=acc[mt*2u]+lo*s0*xv; acc[mt*2u+1u]=acc[mt*2u+1u]+hi*s1*xv; }} }}
    }} }}
  for(var mt=0u;mt<{MT}u;mt++){{ let m=m0+mt; if(m<M){{ let o=m*N+t*8u;
    let a0=acc[mt*2u]; let a1=acc[mt*2u+1u];
    y[o]=a0.x;y[o+1u]=a0.y;y[o+2u]=a0.z;y[o+3u]=a0.w;y[o+4u]=a1.x;y[o+5u]=a1.y;y[o+6u]=a1.z;y[o+7u]=a1.w; }} }}
}}"""
def bench(K,N,M,MT,validate=False):
  nblk=K//BLK; rng=np.random.default_rng(0)
  X=(rng.standard_normal((M,K))*0.1).astype(np.float32)
  Wf=(rng.standard_normal((N,K))*0.05).astype(np.float32)
  Wb=Wf.reshape(N,nblk,BLK); scale=np.maximum(np.abs(Wb).max(2)/7.0,1e-8).astype(np.float32)
  q=np.clip(np.round(Wb/scale[:,:,None])+8,0,15).astype(np.uint32).reshape(N,K)
  qT=np.ascontiguousarray(q.T); wp=np.zeros((K,N//8),np.uint32)
  for j in range(8): wp|=(qT[:,j::8]&15)<<(j*4)
  scb=np.ascontiguousarray(scale.T).astype(np.float16).reshape(-1).view(np.uint32)
  mk=lambda a,u:dev.create_buffer_with_data(data=np.ascontiguousarray(a).tobytes(),usage=u)
  bx=mk(X,U.STORAGE);bw=mk(wp.reshape(-1),U.STORAGE);bs=mk(scb,U.STORAGE)
  by=dev.create_buffer(size=M*N*4,usage=U.STORAGE|U.COPY_SRC)
  bd=mk(np.array([N,K,M,N//8],np.uint32),U.UNIFORM)
  p=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=shader(MT)),"entry_point":"main"})
  bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bx,bw,bs,by,bd])])
  def go():
    e=dev.create_command_encoder();c=e.begin_compute_pass();c.set_pipeline(p);c.set_bind_group(0,bg)
    c.dispatch_workgroups((N//8+WGS-1)//WGS,(M+MT-1)//MT);c.end();dev.queue.submit([e.finish()])
  if validate:
    go(); Y=np.frombuffer(dev.queue.read_buffer(by),np.float32).reshape(M,N)
    Wdq=((q.astype(np.float32)-8).reshape(N,nblk,BLK)*scale[:,:,None]).reshape(N,K)
    ref=X@Wdq.T; cos=float((Y.ravel()@ref.ravel())/(np.linalg.norm(Y)*np.linalg.norm(ref)+1e-9))
    return cos
  for _ in range(5):go()
  dev.queue.on_submitted_work_done_sync();R=30;t=time.time()
  for _ in range(R):go()
  dev.queue.on_submitted_work_done_sync();return (time.time()-t)/R*1e3
for name,(K,N) in DIMS.items():
  print(f"\n=== {name} ===",flush=True)
  print(f"  validate M=8 MT=4: cos={bench(K,N,8,4,validate=True):.5f}",flush=True)
  t1=bench(K,N,1,1)
  print(f"  M=  1 (GEMV): {t1:7.3f} ms = {t1:.3f} ms/token",flush=True)
  for M,MT in [(8,4),(32,4),(64,8),(128,8)]:
    tm=bench(K,N,M,MT)
    print(f"  M={M:4} MT={MT}: {tm:7.3f} ms = {tm/M:.4f} ms/token  ({t1/(tm/M):.1f}x vs GEMV)",flush=True)
