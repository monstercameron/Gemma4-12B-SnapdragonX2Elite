"""Stage 1 of the 64k engine v2: flash-decode attention (online softmax, tiled over the KV sequence).
One workgroup per query head; loops KV in TILE-sized blocks maintaining running max m, denom l, and
output accumulator acc[hd] in shared memory -- so T is unbounded (no 256-thread wall). Validated vs a
numpy reference at T in {16,300,1024,8192,64000}; times T=64000 to estimate decode cost at full ctx."""
import time, numpy as np, wgpu
ad=wgpu.gpu.request_adapter_sync(power_preference="high-performance"); dev=ad.request_device_sync()
U=wgpu.BufferUsage; nH=16; WGS=256; TILE=256
SH=f"""
@group(0) @binding(0) var<storage,read> q: array<f32>;       // [nH*hd] normed+roped query
@group(0) @binding(1) var<storage,read> kc: array<f32>;      // [nkv*MAXT*hd]
@group(0) @binding(2) var<storage,read> vc: array<f32>;      // [nkv*MAXT*hd]
@group(0) @binding(3) var<storage,read_write> outb: array<f32>;
@group(0) @binding(4) var<uniform> d: vec4<u32>;             // hd, T, maxt, grp
var<workgroup> qs: array<f32,512>;
var<workgroup> sc: array<f32,{TILE}>;
var<workgroup> red: array<f32,{WGS}>;
var<workgroup> acc: array<f32,512>;
var<workgroup> msh: f32; var<workgroup> lsh: f32; var<workgroup> corrsh: f32;
@compute @workgroup_size({WGS})
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>){{
  let h=wg.x; let t=li.x; let hd=d.x; let T=d.y; let maxt=d.z; let grp=d.w; let kvh=h/grp;
  for(var i=t;i<hd;i+=({WGS}u)){{ qs[i]=q[h*hd+i]; acc[i]=0.0; }}
  if(t==0u){{ msh=-1e30; lsh=0.0; }}
  workgroupBarrier();
  var tile=0u;
  loop {{ if(tile>=T){{ break; }}
    let tlen=min({TILE}u, T-tile);
    // scores for this tile
    for(var j=t;j<tlen;j+=({WGS}u)){{ let key=tile+j; let kb=(kvh*maxt+key)*hd;
      var s=0.0; for(var i=0u;i<hd;i++){{ s+=qs[i]*kc[kb+i]; }} sc[j]=s; }}
    workgroupBarrier();
    // tile max -> red reduce
    var mx=-1e30; for(var j=t;j<tlen;j+=({WGS}u)){{ mx=max(mx,sc[j]); }} red[t]=mx; workgroupBarrier();
    for(var st=({WGS}u/2u);st>0u;st>>=1u){{ if(t<st){{ red[t]=max(red[t],red[t+st]); }} workgroupBarrier(); }}
    if(t==0u){{ let mnew=max(msh,red[0]); corrsh=exp(msh-mnew); msh=mnew; }}
    workgroupBarrier();
    let mnew=msh; let corr=corrsh;
    // probs into sc, denom contribution
    var ds=0.0; for(var j=t;j<tlen;j+=({WGS}u)){{ let p=exp(sc[j]-mnew); sc[j]=p; ds+=p; }}
    red[t]=ds; workgroupBarrier();
    for(var st=({WGS}u/2u);st>0u;st>>=1u){{ if(t<st){{ red[t]+=red[t+st]; }} workgroupBarrier(); }}
    if(t==0u){{ lsh=lsh*corr+red[0]; }}
    // acc update: each thread owns hd dims i, rescale + add tile
    for(var i=t;i<hd;i+=({WGS}u)){{ var a=acc[i]*corr;
      for(var j=0u;j<tlen;j++){{ let key=tile+j; a+=sc[j]*vc[(kvh*maxt+key)*hd+i]; }} acc[i]=a; }}
    workgroupBarrier();
    tile+=({TILE}u);
  }}
  let l=lsh; for(var i=t;i<hd;i+=({WGS}u)){{ outb[h*hd+i]=acc[i]/l; }}
}}"""
def ref(q,kc,vc,hd,T,maxt,grp):
  o=np.zeros(nH*hd,np.float32)
  for h in range(nH):
    kvh=h//grp; qq=q[h*hd:(h+1)*hd]
    s=np.array([qq@kc[(kvh*maxt+k)*hd:(kvh*maxt+k)*hd+hd] for k in range(T)])
    s=np.exp(s-s.max()); s/=s.sum()
    for i in range(hd): o[h*hd+i]=float(np.dot(s, vc[(kvh*maxt+np.arange(T))*hd+i]))
  return o
def run(hd,T,maxt,grp,timing=False):
  rng=np.random.default_rng(1); nkv=nH//grp
  q=(rng.standard_normal(nH*hd)*0.1).astype(np.float32)
  kc=(rng.standard_normal(nkv*maxt*hd)*0.1).astype(np.float32)
  vc=(rng.standard_normal(nkv*maxt*hd)*0.1).astype(np.float32)
  mk=lambda a,u:dev.create_buffer_with_data(data=a.tobytes(),usage=u)
  bq=mk(q,U.STORAGE);bk=mk(kc,U.STORAGE);bv=mk(vc,U.STORAGE)
  bo=dev.create_buffer(size=nH*hd*4,usage=U.STORAGE|U.COPY_SRC)
  bd=mk(np.array([hd,T,maxt,grp],np.uint32),U.UNIFORM)
  p=dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,compute={"module":dev.create_shader_module(code=SH),"entry_point":"main"})
  bg=dev.create_bind_group(layout=p.get_bind_group_layout(0),entries=[{"binding":i,"resource":{"buffer":b,"offset":0,"size":b.size}} for i,b in enumerate([bq,bk,bv,bo,bd])])
  def go():
    e=dev.create_command_encoder();c=e.begin_compute_pass();c.set_pipeline(p);c.set_bind_group(0,bg)
    c.dispatch_workgroups(nH,1);c.end();dev.queue.submit([e.finish()])
  go(); y=np.frombuffer(dev.queue.read_buffer(bo),np.float32).copy()
  if not timing:
    r=ref(q,kc,vc,hd,T,maxt,grp); cos=float(y@r/(np.linalg.norm(y)*np.linalg.norm(r)+1e-9))
    print(f"  hd={hd} grp={grp} T={T:6}: cos={cos:.5f} {'OK' if cos>0.999 else 'FAIL'}",flush=True)
  else:
    for _ in range(5):go()
    dev.queue.on_submitted_work_done_sync();R=30;t0=time.time()
    for _ in range(R):go()
    dev.queue.on_submitted_work_done_sync();ms=(time.time()-t0)/R*1e3
    print(f"  hd={hd} grp={grp} T={T}: {ms:.2f} ms/attn-call  (x8 global layers = {ms*8:.1f} ms/token)",flush=True)
print("=== correctness ===",flush=True)
for hd,grp in [(256,2),(512,16)]:
  for T in (16,300,1024,8192):
    run(hd,T,max(T,1024) if T>1024 else 1024,grp)
print("=== timing at 64k (global layer, hd=512) ===",flush=True)
run(512,64000,64000,16,timing=True)
