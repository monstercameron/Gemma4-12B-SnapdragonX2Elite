"""Decode-attention microbench (wgpu/Adreno): isolate score->softmax->output (the part prep/attn
differ on) for one layer's worth = nH=16 workgroups. Compare CURRENT (serial thread-0 softmax,
scalar loads) vs OPTIMIZED (parallel tree-reduce softmax, vec4 score+output). Validate vs numpy,
sweep T in {16,256} x hd in {256,512}."""
import time, numpy as np, wgpu
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage; nH=16
CUR="""
@group(0) @binding(0) var<storage,read> qr: array<f32>;
@group(0) @binding(1) var<storage,read> kc: array<f32>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> outb: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;
var<workgroup> sc: array<f32,256>; var<workgroup> sm: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>){
  let h=wg.x; let t=l.x; let hd=d.x; let T=d.y; let maxt=d.z; let grp=d.w; let kvh=h/grp;
  if(t<T){ var a=0.0; let qb=h*hd; let kb=(kvh*maxt+t)*hd;
    for(var j=0u;j<hd;j=j+1u){a=a+qr[qb+j]*kc[kb+j];} sc[t]=a; }
  workgroupBarrier();
  if(t==0u){ var mx=-1e30; for(var j=0u;j<T;j=j+1u){mx=max(mx,sc[j]);}
    var s=0.0; for(var j=0u;j<T;j=j+1u){let e=exp(sc[j]-mx); sc[j]=e; s=s+e;} sm=s; }
  workgroupBarrier(); let s=sm;
  for(var i=t;i<hd;i=i+256u){ var a=0.0; for(var j=0u;j<T;j=j+1u){a=a+sc[j]*vc[(kvh*maxt+j)*hd+i];}
    outb[h*hd+i]=a/s; }
}"""
OPT="""
@group(0) @binding(0) var<storage,read> qr: array<vec4<f32>>;
@group(0) @binding(1) var<storage,read> kc: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> outb: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;
var<workgroup> sc: array<f32,256>; var<workgroup> red: array<f32,256>; var<workgroup> sm: f32;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>){
  let h=wg.x; let t=l.x; let hd=d.x; let T=d.y; let maxt=d.z; let grp=d.w; let kvh=h/grp; let h4=hd/4u;
  if(t<T){ var a=vec4<f32>(0.0); let qb=h*h4; let kb=(kvh*maxt+t)*h4;
    for(var j=0u;j<h4;j=j+1u){a=a+qr[qb+j]*kc[kb+j];} sc[t]=a.x+a.y+a.z+a.w; }
  workgroupBarrier();
  red[t]=select(-1e30,sc[t],t<T); workgroupBarrier();
  for(var st=128u;st>0u;st=st>>1u){ if(t<st){red[t]=max(red[t],red[t+st]);} workgroupBarrier(); }
  let mx=red[0]; workgroupBarrier();
  var e=0.0; if(t<T){e=exp(sc[t]-mx); sc[t]=e;} red[t]=e; workgroupBarrier();
  for(var st=128u;st>0u;st=st>>1u){ if(t<st){red[t]=red[t]+red[t+st];} workgroupBarrier(); }
  if(t==0u){sm=red[0];} workgroupBarrier(); let s=sm;
  for(var i=t;i<hd;i=i+256u){ var a=0.0; for(var j=0u;j<T;j=j+1u){a=a+sc[j]*vc[(kvh*maxt+j)*hd+i];}
    outb[h*hd+i]=a/s; }
}"""
def refout(qr,kc,vc,hd,T,maxt,grp):
  o=np.zeros(nH*hd,np.float32)
  for h in range(nH):
    kvh=h//grp; q=qr[h*hd:(h+1)*hd]
    s=np.array([q@kc[(kvh*maxt+t)*hd:(kvh*maxt+t)*hd+hd] for t in range(T)])
    s=np.exp(s-s.max()); s/=s.sum()
    for i in range(hd): o[h*hd+i]=sum(s[t]*vc[(kvh*maxt+t)*hd+i] for t in range(T))
  return o
def run(code,hd,T,maxt,grp,label):
  rng=np.random.default_rng(0); nkv=nH//grp
  qr=(rng.standard_normal(nH*hd)*0.1).astype(np.float32)
  kc=(rng.standard_normal(nkv*maxt*hd)*0.1).astype(np.float32)
  vc=(rng.standard_normal(nkv*maxt*hd)*0.1).astype(np.float32)
  mk=lambda a,u:dev.create_buffer_with_data(data=a.tobytes(),usage=u)
  bq=mk(qr,U.STORAGE);bk=mk(kc,U.STORAGE);bv=mk(vc,U.STORAGE)
  bo=dev.create_buffer(size=nH*hd*4,usage=U.STORAGE|U.COPY_SRC)
  bd=mk(np.array([hd,T,maxt,grp],np.uint32),U.UNIFORM)
  m=dev.create_shader_module(code=code)
  p=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":m,"entry_point":"main"})
  bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bq,bk,bv,bo,bd])])
  def go():
    e=dev.create_command_encoder();c=e.begin_compute_pass();c.set_pipeline(p);c.set_bind_group(0,bg)
    c.dispatch_workgroups(nH,1);c.end();dev.queue.submit([e.finish()])
  go(); y=np.frombuffer(dev.queue.read_buffer(bo),np.float32).copy()
  ref=refout(qr,kc,vc,hd,T,maxt,grp); cos=float(y@ref/(np.linalg.norm(y)*np.linalg.norm(ref)+1e-9))
  for _ in range(20):go()
  dev.queue.on_submitted_work_done_sync(); R=500; t0=time.time()
  for _ in range(R):go()
  dev.queue.on_submitted_work_done_sync(); ms=(time.time()-t0)/R*1e3
  print(f"  {label:9} hd={hd} T={T:3}: {ms*1000:7.1f} us  cos={cos:.4f}{'' if cos>0.99 else '  FAIL'}",flush=True)
  return ms
for hd,grp in [(256,2),(512,16)]:
  for T in (16,256):
    print(f"--- hd={hd} grp={grp} T={T} (x48 layers) ---",flush=True)
    c=run(CUR,hd,T,256,grp,"current"); o=run(OPT,hd,T,256,grp,"optimized")
    print(f"  speedup {c/o:.2f}x   per-token attn est: cur {c*48*1000:.0f}us -> opt {o*48*1000:.0f}us",flush=True)
