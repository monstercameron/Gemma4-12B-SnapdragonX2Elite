"""Validate multi-query flash attention: HG query heads sharing one KV head processed in ONE workgroup,
so each K/V element is read once and reused across HG heads (vs once-per-head today). Config fits the
16KB WebGPU shared limit (HG=2,hd=512); the GLSL port uses 32KB -> HG=4. Validate cos vs numpy + speed
vs HG=1 baseline. grp=16 (global MQA): all 16 heads share kvh=0 -> nH/HG workgroups, KV read nH/HG x."""
import time, numpy as np, wgpu
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage; nH=16; hd=512; grp=16; HG=2; maxt=4096
def shader(HG):
  return f"""
@group(0) @binding(0) var<storage,read> q: array<f32>;     // [nH,hd] normed+roped
@group(0) @binding(1) var<storage,read> kc: array<f32>;    // [nkv,maxt,hd]
@group(0) @binding(2) var<storage,read> vc: array<f32>;
@group(0) @binding(3) var<storage,read_write> outb: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;           // hd,T,maxt,grp
var<workgroup> qs: array<f32,{HG*512}>; var<workgroup> ac: array<f32,{HG*512}>;
var<workgroup> sc: array<f32,{HG*256}>; var<workgroup> red: array<f32,256>;
var<workgroup> mm: array<f32,{HG}>; var<workgroup> ll: array<f32,{HG}>; var<workgroup> cc: array<f32,{HG}>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>){{
  let g=wg.x; let h0=g*{HG}u; let t=li.x; let hd=d.x; let T=d.y; let maxt=d.z; let kvh=h0/d.w;
  for(var hh=0u;hh<{HG}u;hh++){{ for(var i=t;i<hd;i+=256u){{ qs[hh*hd+i]=q[(h0+hh)*hd+i]; ac[hh*hd+i]=0.0; }} }}
  if(t<{HG}u){{ mm[t]=-1e30; ll[t]=0.0; }}
  workgroupBarrier();
  var tile=0u;
  loop {{ if(tile>=T){{break;}} let tlen=min(256u,T-tile);
    // scores: thread-per-key reads k_j once, HG dots
    for(var j=t;j<tlen;j+=256u){{ let kb=(kvh*maxt+tile+j)*hd; var dd=array<f32,{HG}>();
      for(var i=0u;i<hd;i++){{ let kv=kc[kb+i]; for(var hh=0u;hh<{HG}u;hh++){{ dd[hh]+=qs[hh*hd+i]*kv; }} }}
      for(var hh=0u;hh<{HG}u;hh++){{ sc[hh*256u+j]=dd[hh]; }} }}
    workgroupBarrier();
    for(var hh=0u;hh<{HG}u;hh++){{
      var mx=-1e30; for(var j=t;j<tlen;j+=256u){{ mx=max(mx,sc[hh*256u+j]); }} red[t]=mx; workgroupBarrier();
      for(var st=128u;st>0u;st>>=1u){{ if(t<st){{red[t]=max(red[t],red[t+st]);}} workgroupBarrier(); }}
      if(t==0u){{ let mn=max(mm[hh],red[0]); cc[hh]=exp(mm[hh]-mn); mm[hh]=mn; }} workgroupBarrier();
      var ds=0.0; for(var j=t;j<tlen;j+=256u){{ let p=exp(sc[hh*256u+j]-mm[hh]); sc[hh*256u+j]=p; ds+=p; }} red[t]=ds; workgroupBarrier();
      for(var st=128u;st>0u;st>>=1u){{ if(t<st){{red[t]+=red[t+st];}} workgroupBarrier(); }}
      if(t==0u){{ ll[hh]=ll[hh]*cc[hh]+red[0]; }} workgroupBarrier();
    }}
    // acc: thread-per-i reads v_j once, HG heads
    for(var i=t;i<hd;i+=256u){{ var a=array<f32,{HG}>(); for(var hh=0u;hh<{HG}u;hh++){{ a[hh]=ac[hh*hd+i]*cc[hh]; }}
      for(var j=0u;j<tlen;j++){{ let vv=vc[(kvh*maxt+tile+j)*hd+i]; for(var hh=0u;hh<{HG}u;hh++){{ a[hh]+=sc[hh*256u+j]*vv; }} }}
      for(var hh=0u;hh<{HG}u;hh++){{ ac[hh*hd+i]=a[hh]; }} }}
    workgroupBarrier(); tile+=256u;
  }}
  for(var hh=0u;hh<{HG}u;hh++){{ for(var i=t;i<hd;i+=256u){{ outb[(h0+hh)*hd+i]=ac[hh*hd+i]/ll[hh]; }} }}
}}"""
def run(HG,T,timing=False):
  rng=np.random.default_rng(2); nkv=nH//grp
  q=(rng.standard_normal(nH*hd)*0.1).astype(np.float32)
  kc=(rng.standard_normal(nkv*maxt*hd)*0.1).astype(np.float32)
  vc=(rng.standard_normal(nkv*maxt*hd)*0.1).astype(np.float32)
  mk=lambda a,u:dev.create_buffer_with_data(data=a.tobytes(),usage=u)
  bq=mk(q,U.STORAGE);bk=mk(kc,U.STORAGE);bv=mk(vc,U.STORAGE)
  bo=dev.create_buffer(size=nH*hd*4,usage=U.STORAGE|U.COPY_SRC); bd=mk(np.array([hd,T,maxt,grp],np.uint32),U.UNIFORM)
  p=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=shader(HG)),"entry_point":"main"})
  bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bq,bk,bv,bo,bd])])
  def go():
    e=dev.create_command_encoder();c=e.begin_compute_pass();c.set_pipeline(p);c.set_bind_group(0,bg)
    c.dispatch_workgroups(nH//HG,1);c.end();dev.queue.submit([e.finish()])
  go(); y=np.frombuffer(dev.queue.read_buffer(bo),np.float32).copy()
  if not timing:
    nkv=nH//grp; o=np.zeros(nH*hd,np.float32)
    for h in range(nH):
      kvh=h//grp; qq=q[h*hd:(h+1)*hd]
      s=np.array([qq@kc[(kvh*maxt+k)*hd:(kvh*maxt+k)*hd+hd] for k in range(T)]); s=np.exp(s-s.max()); s/=s.sum()
      for i in range(hd): o[h*hd+i]=float(np.dot(s,vc[(kvh*maxt+np.arange(T))*hd+i]))
    return float(y@o/(np.linalg.norm(y)*np.linalg.norm(o)+1e-9))
  for _ in range(5):go()
  dev.queue.on_submitted_work_done_sync();R=30;t0=time.time()
  for _ in range(R):go()
  dev.queue.on_submitted_work_done_sync();return (time.time()-t0)/R*1e3
print(f"correctness HG=2 T=2000: cos={run(2,2000):.5f}",flush=True)
print(f"correctness HG=1 T=2000: cos={run(1,2000):.5f}  (baseline, 1 head/wg)",flush=True)
t1=run(1,4000,True); t2=run(2,4000,True)
print(f"timing T=4000 (global MQA): HG=1 {t1:.2f}ms  HG=2 {t2:.2f}ms  -> {t1/t2:.2f}x",flush=True)
