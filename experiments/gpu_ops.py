"""Small f32 GPU compute shaders for the resident engine: rmsnorm, add(in-place), gelu*up,
scale, softcap. Each is a thin pipeline + a helper that encodes into a shared command encoder
(no readback) so the whole layer chains on-GPU."""
import numpy as np, wgpu

RMSNORM = """
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<f32>;
@group(0) @binding(2) var<storage,read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;     // D
var<workgroup> sh: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let D=d.x; let t=lid.x;
  var s=0.0;
  for (var i=t; i<D; i=i+256u) { let v=x[i]; s=s+v*v; }
  sh[t]=s; workgroupBarrier();
  for (var st=128u; st>0u; st=st>>1u){ if(t<st){sh[t]=sh[t]+sh[t+st];} workgroupBarrier(); }
  let inv=inverseSqrt(sh[0]/f32(D)+1e-6);
  for (var i=t; i<D; i=i+256u) { y[i]=x[i]*inv*w[i]; }
}
"""
ADD = """
@group(0) @binding(0) var<storage,read_write> a: array<f32>;
@group(0) @binding(1) var<storage,read> b: array<f32>;
@group(0) @binding(2) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x; if(i<d.x){a[i]=a[i]+b[i];} }
"""
GELUMUL = """
@group(0) @binding(0) var<storage,read> gate: array<f32>;
@group(0) @binding(1) var<storage,read> up: array<f32>;
@group(0) @binding(2) var<storage,read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let i=g.x; if(i>=d.x){return;}
  let x=gate[i];
  let gl=0.5*x*(1.0+tanh(0.7978845608028654*(x+0.044715*x*x*x)));
  out[i]=gl*up[i];
}
"""
SCALE = """
@group(0) @binding(0) var<storage,read_write> a: array<f32>;
@group(0) @binding(1) var<uniform> d: vec4<u32>;     // D, scale_bits
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){ let i=g.x; if(i<d.x){a[i]=a[i]*bitcast<f32>(d.y);} }
"""
NORM_ADD_REDUCE = """
@group(0) @binding(0) var<storage,read> bp: array<f32>;   // [split, D] GEMV partials
@group(0) @binding(1) var<storage,read> w: array<f32>;
@group(0) @binding(2) var<storage,read_write> h: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;          // D, split, scale_bits, 0
var<workgroup> sh: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let D=d.x; let sp=d.y; let t=lid.x;
  var s=0.0;
  for(var i=t;i<D;i=i+256u){ var xi=0.0; for(var c=0u;c<sp;c=c+1u){xi=xi+bp[c*D+i];} s=s+xi*xi; }
  sh[t]=s; workgroupBarrier();
  for(var st=128u;st>0u;st=st>>1u){ if(t<st){sh[t]=sh[t]+sh[t+st];} workgroupBarrier(); }
  let inv=inverseSqrt(sh[0]/f32(D)+1e-6);
  var sc=1.0; if(d.z!=0u){ sc=bitcast<f32>(d.z); }
  for(var i=t;i<D;i=i+256u){ var xi=0.0; for(var c=0u;c<sp;c=c+1u){xi=xi+bp[c*D+i];} h[i]=(h[i]+xi*inv*w[i])*sc; }
}
"""
GELUMUL_REDUCE = """
@group(0) @binding(0) var<storage,read> gbp: array<f32>;  // [split, I]
@group(0) @binding(1) var<storage,read> ubp: array<f32>;  // [split, I]
@group(0) @binding(2) var<storage,read_write> outb: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;          // I, split, 0, 0
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i=g.x; if(i>=d.x){return;} let sp=d.y;
  var ga=0.0; var ua=0.0;
  for(var c=0u;c<sp;c=c+1u){ ga=ga+gbp[c*d.x+i]; ua=ua+ubp[c*d.x+i]; }
  let gl=0.5*ga*(1.0+tanh(0.7978845608028654*(ga+0.044715*ga*ga*ga)));
  outb[i]=gl*ua;
}
"""
NORM_ADD = """
@group(0) @binding(0) var<storage,read> x: array<f32>;
@group(0) @binding(1) var<storage,read> w: array<f32>;
@group(0) @binding(2) var<storage,read_write> h: array<f32>;
@group(0) @binding(3) var<uniform> d: vec4<u32>;     // D, scale_bits (0 => no scale)
var<workgroup> sh: array<f32,256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let D=d.x; let t=lid.x;
  var s=0.0; for(var i=t;i<D;i=i+256u){ let v=x[i]; s=s+v*v; } sh[t]=s; workgroupBarrier();
  for(var st=128u;st>0u;st=st>>1u){ if(t<st){sh[t]=sh[t]+sh[t+st];} workgroupBarrier(); }
  let inv=inverseSqrt(sh[0]/f32(D)+1e-6);
  var sc=1.0; if(d.y!=0u){ sc=bitcast<f32>(d.y); }
  for(var i=t;i<D;i=i+256u){ h[i]=(h[i]+x[i]*inv*w[i])*sc; }
}
"""
SOFTCAP = """
@group(0) @binding(0) var<storage,read_write> a: array<f32>;
@group(0) @binding(1) var<uniform> d: vec4<u32>;     // V, cap_bits
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>){
  let i=g.x; if(i>=d.x){return;} let cap=bitcast<f32>(d.y); a[i]=cap*tanh(a[i]/cap);
}
"""


class Ops:
    def __init__(self, eng):
        self.e = eng; dev = eng.dev
        mk = lambda code: dev.create_compute_pipeline(layout=wgpu.enums.AutoLayoutMode.auto,
                          compute={"module": dev.create_shader_module(code=code), "entry_point": "main"})
        self.p_norm = mk(RMSNORM); self.p_add = mk(ADD); self.p_gm = mk(GELUMUL)
        self.p_scale = mk(SCALE); self.p_softcap = mk(SOFTCAP); self.p_na = mk(NORM_ADD)
        self.p_nar = mk(NORM_ADD_REDUCE); self.p_gmr = mk(GELUMUL_REDUCE)
        U = wgpu.BufferUsage
        self.uD = {}  # dim -> uniform buffer
        self._U = U

    def _u(self, vals):
        import numpy as np
        return self.e.dev.create_buffer_with_data(
            data=np.array(vals, np.uint32).tobytes(), usage=self._U.UNIFORM)

    def bg(self, pipe, bufs):
        return self.e.dev.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
            {"binding": i, "resource": {"buffer": b, "offset": 0, "size": b.size}}
            for i, b in enumerate(bufs)])

    # encode helpers (call .bg() once at setup and pass the cached bind group)
    def norm(self, enc, bg, pipe=None):
        c = enc.begin_compute_pass(); c.set_pipeline(pipe or self.p_norm); c.set_bind_group(0, bg)
        c.dispatch_workgroups(1); c.end()

    def ew(self, enc, pipe, bg, D):
        c = enc.begin_compute_pass(); c.set_pipeline(pipe); c.set_bind_group(0, bg)
        c.dispatch_workgroups((D + 63) // 64); c.end()
