"""On-GPU attention shaders for the fully-resident Gemma engine (eliminates per-layer readback).
PREP_KV: per kv-head, RMSNorm+RoPE k -> KV cache slot; RMSNorm(no-scale) v -> cache (k_eq_v aware).
ATTN: per query-head, RMSNorm+RoPE q; scores over T (scaling=1.0); stable softmax; weighted V -> out.
f32 throughout. MAXT=256 (no rolling; correct for sequences <=256)."""
import wgpu

MAXT = 256

PREP_KV = """
struct P { nkv:u32, hd:u32, koff:u32, voff:u32, keqv:u32, maxt:u32, a:u32, b:u32 };
struct D { slot:u32, T:u32, a:u32, b:u32 };
@group(0) @binding(0) var<storage,read> qkv: array<f32>;
@group(0) @binding(1) var<storage,read> knw: array<f32>;
@group(0) @binding(2) var<storage,read> cosb: array<f32>;
@group(0) @binding(3) var<storage,read> sinb: array<f32>;
@group(0) @binding(4) var<storage,read_write> kc: array<f32>;
@group(0) @binding(5) var<storage,read_write> vc: array<f32>;
@group(0) @binding(6) var<uniform> p: P;
@group(0) @binding(7) var<uniform> dy: D;
var<workgroup> hv: array<f32,512>;
var<workgroup> sh: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let kvh=wid.x; let hd=p.hd; let t=lid.x; let half=hd/2u;
  // K: norm + rope -> kc[kvh,slot]
  for(var i=t;i<hd;i=i+256u){ hv[i]=qkv[p.koff+kvh*hd+i]; } workgroupBarrier();
  var s=0.0; for(var i=t;i<hd;i=i+256u){ s=s+hv[i]*hv[i]; } sh[t]=s; workgroupBarrier();
  for(var st=128u;st>0u;st=st>>1u){ if(t<st){sh[t]=sh[t]+sh[t+st];} workgroupBarrier(); }
  let ik=inverseSqrt(sh[0]/f32(hd)+1e-6);
  for(var i=t;i<hd;i=i+256u){ hv[i]=hv[i]*ik*knw[i]; } workgroupBarrier();
  for(var i=t;i<hd;i=i+256u){ var rh:f32; if(i<half){rh=-hv[i+half];}else{rh=hv[i-half];}
    kc[(kvh*p.maxt+dy.slot)*hd+i]=hv[i]*cosb[i]+rh*sinb[i]; }
  workgroupBarrier();
  // V: norm (no weight, no rope). k_eq_v: read k region.
  var vo=p.voff; if(p.keqv==1u){ vo=p.koff; }
  for(var i=t;i<hd;i=i+256u){ hv[i]=qkv[vo+kvh*hd+i]; } workgroupBarrier();
  s=0.0; for(var i=t;i<hd;i=i+256u){ s=s+hv[i]*hv[i]; } sh[t]=s; workgroupBarrier();
  for(var st=128u;st>0u;st=st>>1u){ if(t<st){sh[t]=sh[t]+sh[t+st];} workgroupBarrier(); }
  let iv=inverseSqrt(sh[0]/f32(hd)+1e-6);
  for(var i=t;i<hd;i=i+256u){ vc[(kvh*p.maxt+dy.slot)*hd+i]=hv[i]*iv; }
}
"""

ATTN = """
struct A { nheads:u32, nkv:u32, hd:u32, maxt:u32, qoff:u32, grp:u32, a:u32, b:u32 };
struct D { slot:u32, T:u32, a:u32, b:u32 };
@group(0) @binding(0) var<storage,read> qkv: array<f32>;
@group(0) @binding(1) var<storage,read> qnw: array<f32>;
@group(0) @binding(2) var<storage,read> cosb: array<f32>;
@group(0) @binding(3) var<storage,read> sinb: array<f32>;
@group(0) @binding(4) var<storage,read> kc: array<f32>;
@group(0) @binding(5) var<storage,read> vc: array<f32>;
@group(0) @binding(6) var<storage,read_write> outb: array<f32>;
@group(0) @binding(7) var<uniform> a: A;
@group(0) @binding(8) var<uniform> dy: D;
var<workgroup> qv: array<f32,512>;
var<workgroup> qr: array<f32,512>;
var<workgroup> sh: array<f32,256>;
var<workgroup> sc: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h=wid.x; let hd=a.hd; let t=lid.x; let kvh=h/a.grp; let half=hd/2u; let T=dy.T;
  for(var i=t;i<hd;i=i+256u){ qv[i]=qkv[a.qoff+h*hd+i]; } workgroupBarrier();
  var s=0.0; for(var i=t;i<hd;i=i+256u){ s=s+qv[i]*qv[i]; } sh[t]=s; workgroupBarrier();
  for(var st=128u;st>0u;st=st>>1u){ if(t<st){sh[t]=sh[t]+sh[t+st];} workgroupBarrier(); }
  let iq=inverseSqrt(sh[0]/f32(hd)+1e-6);
  for(var i=t;i<hd;i=i+256u){ qv[i]=qv[i]*iq*qnw[i]; } workgroupBarrier();
  for(var i=t;i<hd;i=i+256u){ var rh:f32; if(i<half){rh=-qv[i+half];}else{rh=qv[i-half];}
    qr[i]=qv[i]*cosb[i]+rh*sinb[i]; } workgroupBarrier();
  if(t<T){ var d=0.0; let base=(kvh*a.maxt+t)*hd; for(var j=0u;j<hd;j=j+1u){ d=d+qr[j]*kc[base+j]; } sc[t]=d; }
  workgroupBarrier();
  if(t==0u){ var mx=-1e30; for(var j=0u;j<T;j=j+1u){ mx=max(mx,sc[j]); }
    var sm=0.0; for(var j=0u;j<T;j=j+1u){ let e=exp(sc[j]-mx); sc[j]=e; sm=sm+e; } sh[0]=sm; }
  workgroupBarrier();
  let sm=sh[0];
  for(var i=t;i<hd;i=i+256u){ var acc=0.0; for(var j=0u;j<T;j=j+1u){ acc=acc+sc[j]*vc[(kvh*a.maxt+j)*hd+i]; }
    outb[h*hd+i]=acc/sm; }
}
"""


class Attn:
    def __init__(self, eng):
        self.e = eng; dev = eng.dev
        mk = lambda c: dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,
                       compute={"module": dev.create_shader_module(code=c), "entry_point": "main"})
        self.p_prep = mk(PREP_KV); self.p_attn = mk(ATTN)
