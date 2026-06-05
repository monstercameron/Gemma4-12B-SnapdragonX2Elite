"""Gemma 4 12B on RAW VULKAN — the whole token graph (~all dispatches) recorded into ONE command
buffer, resubmitted per token (only embed/pos/cos-sin updated in mapped UMA buffers). Eliminates
the WebGPU per-token encode+submit. Adreno-tuned kernels (WGS=64 GEMV, adaptive split).
Run: .venv-gemma4/Scripts/python.exe scripts/vk_engine.py [ngen]
"""
import os, sys, time, numpy as np, vulkan as vk, torch
ffi = vk.ffi
FP8 = os.environ.get("GEMV_FP8") == "1"  # A/B: fp8 e4m3 per-block scales (half scale traffic) vs fp16
NGEN = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 6  # robust to importers
BLK = 32; WIN = 1024; CTX = 65536; MAXT = CTX; MODEL = "models/gemma-4-12B-it"
# 64k context: global (full) layers keep a CTX-deep KV; sliding layers keep only a WIN-deep ring
# buffer (softmax is permutation-invariant + RoPE is baked in at write time, so ring order is fine).
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

# ---------- Vulkan init ----------
app = vk.VkApplicationInfo(pApplicationName="g", apiVersion=(1 << 22) | (1 << 12))  # Vulkan 1.1 (coopmat)
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
print("device:", vk.vkGetPhysicalDeviceProperties(pdev).deviceName, flush=True)
qfi = next(i for i, q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev))
           if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
# enable cooperative-matrix + memory-model features (for the fp32 coopmat prefill GEMM); additive,
# decode path is unaffected. Structs kept module-level so the pNext chain stays alive.
_F16 = vk.VkPhysicalDeviceShaderFloat16Int8Features(shaderFloat16=vk.VK_TRUE)
_ST16 = vk.VkPhysicalDevice16BitStorageFeatures(storageBuffer16BitAccess=vk.VK_TRUE, pNext=ffi.addressof(_F16))
_COOPF = vk.VkPhysicalDeviceVulkanMemoryModelFeatures(vulkanMemoryModel=vk.VK_TRUE, pNext=ffi.addressof(_ST16))
_COOPC = vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(cooperativeMatrix=vk.VK_TRUE, pNext=ffi.addressof(_COOPF))
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(pNext=ffi.addressof(_COOPC), pQueueCreateInfos=[
    vk.VkDeviceQueueCreateInfo(queueFamilyIndex=qfi, pQueuePriorities=[1.0])],
    ppEnabledExtensionNames=["VK_KHR_cooperative_matrix", "VK_KHR_vulkan_memory_model",
                             "VK_KHR_16bit_storage", "VK_KHR_shader_float16_int8"]), None)
queue = vk.vkGetDeviceQueue(dev, qfi, 0)
PROFILE = os.environ.get("PROFILE") == "1"   # GPU timestamp profiling of the decode token graph
PREPROF = os.environ.get("PREPROF") == "1"   # GPU timestamp profiling of the batched-prefill graph
if PROFILE:
    _tsp = vk.vkGetPhysicalDeviceProperties(pdev).limits.timestampPeriod
    _qpool = vk.vkCreateQueryPool(dev, vk.VkQueryPoolCreateInfo(queryType=vk.VK_QUERY_TYPE_TIMESTAMP, queryCount=2048), None)
    _tslab = []; _agg = {}; _runs = [0]
if PREPROF:
    _tsp_pre = vk.vkGetPhysicalDeviceProperties(pdev).limits.timestampPeriod
    _qpre = vk.vkCreateQueryPool(dev, vk.VkQueryPoolCreateInfo(queryType=vk.VK_QUERY_TYPE_TIMESTAMP, queryCount=256), None)
    _preagg = {}
memp = vk.vkGetPhysicalDeviceMemoryProperties(pdev)
HV = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
def memtype(bits):
    for i in range(memp.memoryTypeCount):
        if (bits & (1 << i)) and (memp.memoryTypes[i].propertyFlags & HV) == HV: return i
    raise RuntimeError("no HV mem")
USAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT

class Buf:
    __slots__ = ("buf", "mem", "size")
    def __init__(self, nbytes):
        self.buf = vk.vkCreateBuffer(dev, vk.VkBufferCreateInfo(size=nbytes, usage=USAGE,
                   sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
        req = vk.vkGetBufferMemoryRequirements(dev, self.buf)
        self.mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(allocationSize=req.size,
                   memoryTypeIndex=memtype(req.memoryTypeBits)), None)
        vk.vkBindBufferMemory(dev, self.buf, self.mem, 0); self.size = nbytes
    def write(self, arr):
        a = np.ascontiguousarray(arr); p = vk.vkMapMemory(dev, self.mem, 0, a.nbytes, 0)
        ffi.memmove(p, ffi.from_buffer(a), a.nbytes); vk.vkUnmapMemory(dev, self.mem)  # no tobytes() copy
    def read(self, nbytes):
        p = vk.vkMapMemory(dev, self.mem, 0, nbytes, 0); out = bytes(p); vk.vkUnmapMemory(dev, self.mem); return out

KEEP = []
def buf_u32(vals):
    b = Buf(max(16, len(vals) * 4)); b.write(np.array(vals, np.uint32)); KEEP.append(b); return b

# ---------- pipelines ----------
def spv(name):
    data = open(os.path.join(HERE, "vk", name), "rb").read()
    return vk.vkCreateShaderModule(dev, vk.VkShaderModuleCreateInfo(codeSize=len(data), pCode=data), None)
def layout(ns, nu):
    binds = [vk.VkDescriptorSetLayoutBinding(binding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(ns)]
    binds += [vk.VkDescriptorSetLayoutBinding(binding=ns + j, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for j in range(nu)]
    return vk.vkCreateDescriptorSetLayout(dev, vk.VkDescriptorSetLayoutCreateInfo(pBindings=binds), None)
def pipe(mod, lay):
    pl = vk.vkCreatePipelineLayout(dev, vk.VkPipelineLayoutCreateInfo(pSetLayouts=[lay]), None)
    p = vk.vkCreateComputePipelines(dev, vk.VK_NULL_HANDLE, 1, [vk.VkComputePipelineCreateInfo(
        stage=vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main"),
        layout=pl)], None)[0]
    return p, pl
LG, LR, LN, LP, LA = layout(4,1), layout(2,1), layout(3,1), layout(6,2), layout(7,2)
P_gemv, PLg = pipe(spv("gemv_fp8.spv" if FP8 else "gemv.spv"), LG); P_red, PLr = pipe(spv("reduce.spv"), LR)
P_norm, PLn = pipe(spv("rmsnorm.spv"), LN); P_na, _ = pipe(spv("normadd.spv"), LN)
P_gm, _ = pipe(spv("gelumul.spv"), LN)   # softcap is applied CPU-side in sample(); no GPU softcap kernel
P_prep, PLp = pipe(spv("prepkv.spv"), LP); P_attn, PLa = pipe(spv("attn.spv"), LA)
# batched-prefill pipelines (reuse layouts: gemm=LG, pre_norm/pre_na=LN, pre_prepkv/pre_attn=LA)
P_pn, PLpn = pipe(spv("pre_norm.spv"), LN)
P_pna, _ = pipe(spv("pre_na.spv"), LN); P_ppk, PLap = pipe(spv("pre_prepkv.spv"), LA)
P_pat, _ = pipe(spv("pre_attn.spv"), LA)
P_cgemm16, PLcg = pipe(spv("coopgemm_i4h.spv"), LG)  # fp16 int4 coopmat GEMM (matrix cores) for prefill
P_castdn, PLca = pipe(spv("cast_dn.spv"), LR)       # f32->f16 (LR = 2 storage + 1 uniform)
P_castup, _ = pipe(spv("cast_up.spv"), LR)          # f16->f32

# descriptor pool (generous)
pool = vk.vkCreateDescriptorPool(dev, vk.VkDescriptorPoolCreateInfo(maxSets=4000, pPoolSizes=[
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=20000),
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=8000)]), None)
def ds(lay, stor, unif):  # entries: (buf,size) or (buf,size,offset)
    s = vk.vkAllocateDescriptorSets(dev, vk.VkDescriptorSetAllocateInfo(descriptorPool=pool, pSetLayouts=[lay]))[0]
    w = []
    for i, e in enumerate(stor):
        off = e[2] if len(e) > 2 else 0
        w.append(vk.VkWriteDescriptorSet(dstSet=s, dstBinding=i, descriptorCount=1,
                 descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=e[0], offset=off, range=e[1])]))
    for j, e in enumerate(unif):
        w.append(vk.VkWriteDescriptorSet(dstSet=s, dstBinding=len(stor) + j, descriptorCount=1,
                 descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=e[0], offset=0, range=e[1])]))
    vk.vkUpdateDescriptorSets(dev, len(w), w, 0, None); return s

# ---------- int4 quant ----------
def quant(W):  # [N,K] -> wpack buf, scales buf, split
    N, K = W.shape; nblk = K // BLK
    Wb = W.reshape(N, nblk, BLK); sc = np.maximum(np.abs(Wb).max(2) / 7.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(Wb / sc[:, :, None]) + 8, 0, 15).astype(np.uint8).reshape(N, K)
    qT = np.ascontiguousarray(q.T); wp = np.zeros((K, N // 8), np.uint32)
    for j in range(8): wp |= (qT[:, j::8].astype(np.uint32) & 15) << (j * 4)
    scb16 = np.ascontiguousarray(sc.T).astype(np.float16).reshape(-1).view(np.uint32)  # fp16 (prefill coopmat)
    inv = 1.0
    if FP8:  # e4m3 decode scales: per-tensor power-of-2 lift into the normal range [2^-6,448], inverse folded in shader
        mx = float(sc.max()); shift = int(np.floor(np.log2(256.0 / mx))) if mx > 0 else 0
        sc_s = np.clip(sc * (2.0 ** shift), 0.0, 448.0)
        e8 = torch.from_numpy(np.ascontiguousarray(sc_s.T)).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
        scb = e8.reshape(-1).view(np.uint32); inv = float(2.0 ** -shift)
    else:
        scb = scb16
    # measured (microbench_total.py, both passes): total time minimizes at ~384 total workgroups.
    # down(rows8)->48, gate(rows30)->12, qkv(rows10)->~24 all match. Huge-N lm_head (rows>=384)
    # naturally falls to split=1 (8.8ms) -- and MUST: split 2..30 is a catastrophic zone (~27ms).
    rows = (N // 8 + 63) // 64; want = max(1, -(-384 // max(rows, 1)))
    # NB: raising small-K splits to their isolated-microbench optimum (nblk/2=60) REGRESSED the full
    # engine (-8%) -- extra partial/reduce traffic across 5 matmuls x 48 layers outweighs the per-
    # matmul gain. The workgroup heuristic below is the measured full-pipeline optimum; keep it.
    divs = [s for s in range(1, nblk + 1) if nblk % s == 0]
    split = min(divs, key=lambda s: (abs(s - want), s))
    bw = Buf(wp.nbytes); bw.write(wp.reshape(-1)); bs = Buf(scb.nbytes); bs.write(scb)
    bs16 = bs if not FP8 else (lambda b: (b.write(scb16), b)[1])(Buf(scb16.nbytes))  # prefill keeps fp16 scales
    return bw, bs, split, nblk, inv, bs16

# ---------- load model ----------
import transformers
from transformers.models.gemma4_unified.modeling_gemma4_unified import apply_rotary_pos_emb  # noqa
print("[load] gemma 4 12B...", flush=True); t0 = time.time()
model = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
    MODEL, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation="eager").eval()
tm = model.model.language_model; cfg = tm.config
H = cfg.hidden_size; nH = cfg.num_attention_heads; I = cfg.intermediate_size; V = cfg.vocab_size
escale = float(getattr(tm.embed_tokens, "scalar_embed_scale", H ** 0.5))
EMB = tm.embed_tokens.weight.detach().float().cpu().numpy(); EMB *= escale  # pre-scale once (in-place, stays f32)
softcap = cfg.final_logit_softcapping
print(f"[load] {time.time()-t0:.0f}s; quantizing -> vulkan buffers...", flush=True); t0 = time.time()

def wbuf(a): b = Buf(np.asarray(a, np.float32).nbytes); b.write(np.asarray(a, np.float32)); return b

# shared buffers
def fb(n): return Buf(n * 4)
B = {k: fb(s) for k, s in {"h": H, "normed": H, "normed2": H, "o": H, "t": H, "t2": H, "down": H,
     "normed_f": H, "qkv": nH * 512 + 2 * 8 * 256, "attn": nH * 512, "gate": I, "up": I, "act": I}.items()}
B["logits"] = fb(V); B["bp"] = fb(32 * V)   # partial buffer sized for lm_head high-split
_logits = np.frombuffer(vk.vkMapMemory(dev, B["logits"].mem, 0, V * 4, 0), np.float32, count=V)  # persistent
_dyn = np.zeros(4, np.uint32)                # reused per-token DYN scratch (no 4-elem alloc per token)
cosb = {"sliding_attention": fb(256), "full_attention": fb(512)}
sinb = {"sliding_attention": fb(256), "full_attention": fb(512)}
DYN = buf_u32([0, 0, 0, 0])
def i4(lin): W = lin.weight.detach().float().cpu().numpy(); lin.weight = None; return quant(W), W.shape

layers = []
for li, L in enumerate(tm.layers):
    sa = L.self_attn; lt = cfg.layer_types[li]; slid = lt == "sliding_attention"
    hd = cfg.head_dim if slid else (cfg.global_head_dim or cfg.head_dim)
    keqv = cfg.attention_k_eq_v and not slid
    nkv = cfg.num_global_key_value_heads if keqv else cfg.num_key_value_heads
    nq, nk = nH * hd, nkv * hd
    stride = WIN if slid else CTX           # KV buffer depth: ring(WIN) for sliding, full(CTX) global
    win = WIN if slid else 0                # 0 -> full attention (T=pos+1); WIN -> windowed ring
    d = dict(li=li, lt=lt, hd=hd, nkv=nkv, nq=nq, nk=nk, grp=nH // nkv, keqv=keqv, stride=stride, win=win,
             qp=i4(sa.q_proj), kp=i4(sa.k_proj), vp=(None if keqv else i4(sa.v_proj)), op=i4(sa.o_proj),
             gp=i4(L.mlp.gate_proj), up=i4(L.mlp.up_proj), dp=i4(L.mlp.down_proj),
             qnw=wbuf(sa.q_norm.weight.detach().float().numpy()), knw=wbuf(sa.k_norm.weight.detach().float().numpy()),
             inw=wbuf(L.input_layernorm.weight.detach().float().numpy()),
             paw=wbuf(L.post_attention_layernorm.weight.detach().float().numpy()),
             pfw=wbuf(L.pre_feedforward_layernorm.weight.detach().float().numpy()),
             ofw=wbuf(L.post_feedforward_layernorm.weight.detach().float().numpy()),
             scl=float(L.layer_scalar.item()) if hasattr(L, "layer_scalar") else 1.0,
             kc=Buf(nkv * stride * hd * 2), vc=Buf(nkv * stride * hd * 2))   # fp16 KV cache (half traffic + 2x ctx capacity)
    layers.append(d)
    if (li + 1) % 16 == 0: print(f"  layer {li+1}/48 ({time.time()-t0:.0f}s)", flush=True)
lmh = i4(model.lm_head); finw = wbuf(tm.norm.weight.detach().float().numpy())
print(f"[load] done {time.time()-t0:.0f}s; building command buffer...", flush=True)

# ---------- descriptor sets + command recording ----------
DUH = buf_u32([H, 0, 0, 0]); DUI = buf_u32([I, 0, 0, 0])
def gemv_ds(qw, inbuf, outbuf, out_off=0):
    (bw, bs, split, nblk, inv, _bs16), (N, K) = qw
    # uniform: d=(N,nb8,bpc,blk); for fp8 also e.x=floatBits(inv) -> 8 u32, else 4. Same LG layout (range differs).
    du = [N, N // 8, nblk // split, BLK] + ([int(np.float32(inv).view(np.uint32)), 0, 0, 0] if FP8 else [])
    d1 = buf_u32(du); usz = 32 if FP8 else 16
    WGx = (N // 16 + 63) // 64   # uvec2 GEMV: 16 outputs/thread -> N/16 threads
    if split == 1:               # no reduction: GEMV writes straight to the output buffer, skip reduce
        dp = ds(LG, [(inbuf.buf, K * 4), (bw.buf, bw.size), (bs.buf, bs.size), (outbuf.buf, N * 4, out_off)], [(d1.buf, usz)])
        return (dp, None, N, split, WGx)
    d2 = buf_u32([N, split, 0, 0])
    dp = ds(LG, [(inbuf.buf, K * 4), (bw.buf, bw.size), (bs.buf, bs.size), (B["bp"].buf, split * N * 4)], [(d1.buf, usz)])
    dr = ds(LR, [(B["bp"].buf, split * N * 4), (outbuf.buf, N * 4, out_off)], [(d2.buf, 16)])
    return (dp, dr, N, split, WGx)
def normbg(inbuf, wb, outbuf, du):
    return ds(LN, [(inbuf.buf, H * 4), (wb.buf, H * 4), (outbuf.buf, H * 4)], [(du.buf, 16)])

for L in layers:
    ct, st = cosb[L["lt"]], sinb[L["lt"]]; hd = L["hd"]; nkv = L["nkv"]; nq = L["nq"]; nk = L["nk"]
    L["d_in"] = normbg(B["h"], L["inw"], B["normed"], DUH)
    L["d_q"] = gemv_ds(L["qp"], B["normed"], B["qkv"], 0)
    L["d_k"] = gemv_ds(L["kp"], B["normed"], B["qkv"], nq * 4)
    if L["vp"]: L["d_v"] = gemv_ds(L["vp"], B["normed"], B["qkv"], (nq + nk) * 4)
    P = buf_u32([nkv, hd, nq, nq + nk, 1 if L["keqv"] else 0, L["stride"], L["win"], 0])
    A = buf_u32([nH, nkv, hd, L["stride"], L["win"], 0, L["grp"], 0])
    L["d_prep"] = ds(LP, [(B["qkv"].buf, B["qkv"].size), (L["knw"].buf, hd * 4), (ct.buf, ct.size), (st.buf, st.size),
                          (L["kc"].buf, L["kc"].size), (L["vc"].buf, L["vc"].size)], [(P.buf, 32), (DYN.buf, 16)])
    L["d_attn"] = ds(LA, [(B["qkv"].buf, B["qkv"].size), (L["qnw"].buf, hd * 4), (ct.buf, ct.size), (st.buf, st.size),
                          (L["kc"].buf, L["kc"].size), (L["vc"].buf, L["vc"].size), (B["attn"].buf, B["attn"].size)], [(A.buf, 32), (DYN.buf, 16)])
    L["d_o"] = gemv_ds(L["op"], B["attn"], B["o"], 0)
    L["d_na"] = ds(LN, [(B["o"].buf, H * 4), (L["paw"].buf, H * 4), (B["h"].buf, H * 4)], [(DUH.buf, 16)])
    L["d_pfn"] = normbg(B["h"], L["pfw"], B["normed2"], DUH)
    L["d_g"] = gemv_ds(L["gp"], B["normed2"], B["gate"], 0)
    L["d_u"] = gemv_ds(L["up"], B["normed2"], B["up"], 0)
    L["d_gm"] = ds(LN, [(B["gate"].buf, I * 4), (B["up"].buf, I * 4), (B["act"].buf, I * 4)], [(DUI.buf, 16)])
    L["d_d"] = gemv_ds(L["dp"], B["act"], B["down"], 0)
    nasu = buf_u32([H, int(np.array([L["scl"]], np.float32).view(np.uint32)[0]), 0, 0])
    L["d_nas"] = ds(LN, [(B["down"].buf, H * 4), (L["ofw"].buf, H * 4), (B["h"].buf, H * 4)], [(nasu.buf, 16)])
d_fin = normbg(B["h"], finw, B["normed_f"], DUH)
d_lm = gemv_ds(lmh, B["normed_f"], B["logits"], 0)

# ---------- record ONE command buffer ----------
cpool = vk.vkCreateCommandPool(dev, vk.VkCommandPoolCreateInfo(queueFamilyIndex=qfi), None)
cb = vk.vkAllocateCommandBuffers(dev, vk.VkCommandBufferAllocateInfo(commandPool=cpool,
     level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
MB = vk.VkMemoryBarrier(srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT)
CS = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
def bar(): vk.vkCmdPipelineBarrier(cb, CS, CS, 0, 1, [MB], 0, None, 0, None)
_BOP = vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
def ts(label):   # write a GPU timestamp after the current op (no-op unless PROFILE)
    if PROFILE:
        vk.vkCmdWriteTimestamp(cb, _BOP, _qpool, len(_tslab)); _tslab.append(label)
def disp(pp, pl, dset, gx, gy=1):
    vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pp)
    vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [dset], 0, None)
    vk.vkCmdDispatch(cb, gx, gy, 1); bar()
def gemv(g):
    dp, dr, N, split, WGx = g; disp(P_gemv, PLg, dp, WGx, split)
    if dr is not None: disp(P_red, PLr, dr, (N // 4 + 63) // 64)   # vec4 reduce: N/4 threads

vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
if PROFILE: vk.vkCmdResetQueryPool(cb, _qpool, 0, 2048)
ts("start")
for L in layers:
    disp(P_norm, PLn, L["d_in"], 1); ts("norm")
    gemv(L["d_q"]); gemv(L["d_k"])
    if L["vp"]: gemv(L["d_v"])
    ts("qkv")
    disp(P_prep, PLp, L["d_prep"], L["nkv"]); ts("prepkv"); disp(P_attn, PLa, L["d_attn"], nH); ts("attn")
    gemv(L["d_o"]); ts("o_proj"); disp(P_na, PLn, L["d_na"], 1); ts("normadd")
    disp(P_norm, PLn, L["d_pfn"], 1); ts("norm")
    gemv(L["d_g"]); gemv(L["d_u"]); ts("gate_up"); disp(P_gm, PLn, L["d_gm"], (I + 63) // 64); ts("gelu")
    gemv(L["d_d"]); ts("down"); disp(P_na, PLn, L["d_nas"], 1); ts("normadd")
disp(P_norm, PLn, d_fin, 1); ts("norm"); gemv(d_lm); ts("lm_head")
# no GPU softcap: cap*tanh(x/cap) is monotonic so argmax(softcap)==argmax for greedy decode, and
# sample() applies final-logit softcapping CPU-side for temperature/top-p. So logits stay raw here.
vk.vkEndCommandBuffer(cb)
print("[rec] command buffer recorded; running inference...", flush=True)

fence = vk.vkCreateFence(dev, vk.VkFenceCreateInfo(), None)
rope_cache = {}
def set_rope(pos):
    for lt in ("sliding_attention", "full_attention"):
        if (pos, lt) not in rope_cache:
            cs, sn = tm.rotary_emb(torch.zeros(1, 1, cfg.head_dim if lt == "sliding_attention" else cfg.global_head_dim),
                                   torch.tensor([[pos]]), layer_type=lt)
            rope_cache[(pos, lt)] = (cs.reshape(-1).float().numpy(), sn.reshape(-1).float().numpy())  # .float() is already f32
        cs, sn = rope_cache[(pos, lt)]; cosb[lt].write(cs); sinb[lt].write(sn)

def forward(tid, pos):
    B["h"].write(EMB[tid])                                  # EMB pre-scaled at load
    _dyn[0] = pos; _dyn[1] = pos + 1; DYN.write(_dyn); set_rope(pos)
    vk.vkResetFences(dev, 1, [fence])
    vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cb])], fence)
    vk.vkWaitForFences(dev, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
    if PROFILE: _read_profile()
    return _logits                                          # persistent coherent view -- no 1MB copy/token

def _read_profile():
    n = len(_tslab); buf = ffi.new(f"uint64_t[{n}]")
    vk.vkGetQueryPoolResults(dev, _qpool, 0, n, n * 8, buf, 8, vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
    for i in range(1, n):
        d = (int(buf[i]) - int(buf[i - 1])) * _tsp / 1000.0   # us
        _agg[_tslab[i]] = _agg.get(_tslab[i], 0.0) + d
    _runs[0] += 1

def print_profile():
    if not _runs[0]: return
    tot = sum(_agg.values()) / _runs[0]
    print(f"\n=== GPU profile (per token, avg of {_runs[0]} tokens) ===", flush=True)
    for k in sorted(_agg, key=lambda x: -_agg[x]):
        us = _agg[k] / _runs[0]; print(f"  {k:10} {us:8.1f} us  {100*us/tot:5.1f}%", flush=True)
    print(f"  {'TOTAL':10} {tot:8.1f} us  -> {1e6/tot:.1f} tok/s (GPU-only)", flush=True)

# ================= BATCHED PREFILL (additive: builds KV for a chunk of MC tokens at once so each
# weight is read once instead of MC times -> ~10x faster prompt processing. Decode path untouched). =
MC = 128                                # prefill chunk size (tokens processed per coopmat GEMM pass)
nqM = nH * 512; nkM = 8 * 256
Xp = fb(MC * H); Np = fb(MC * H); Tp = fb(MC * H)
Qp = fb(MC * nqM); Ap = fb(MC * nqM); Kp = fb(MC * nkM); Vp = fb(MC * nkM)
Gp = fb(MC * I); Upp = fb(MC * I); ACp = fb(MC * I)
Xf16 = Buf(MC * I * 2); Yf16 = Buf(MC * I * 2)   # fp16 GEMM staging (max dim = I); reused per GEMM
cosMp = {"sliding_attention": fb(MC * 256), "full_attention": fb(MC * 512)}
sinMp = {"sliding_attention": fb(MC * 256), "full_attention": fb(MC * 512)}
DYNP = buf_u32([0, MC, 0, 0]); DUIp = buf_u32([MC * I, 0, 0, 0])

def gemm_ds(qw, inbuf, outbuf):
    # fp16 coopmat GEMM: cast inbuf f32->f16 (Xf16), fp16 GEMM -> Yf16, cast Yf16 f16->f32 (outbuf).
    (bw, _bs, _s, _n, _inv, bs16), (N, K) = qw  # prefill coopmat reads fp16 scales (bs16)
    u = buf_u32([N, K, MC, N // 8]); uin = buf_u32([MC * K, 0, 0, 0]); uout = buf_u32([MC * N, 0, 0, 0])
    cin = ds(LR, [(inbuf.buf, MC * K * 4), (Xf16.buf, MC * K * 2)], [(uin.buf, 16)])
    gem = ds(LG, [(Xf16.buf, MC * K * 2), (bw.buf, bw.size), (bs16.buf, bs16.size), (Yf16.buf, MC * N * 2)], [(u.buf, 16)])
    cout = ds(LR, [(Yf16.buf, MC * N * 2), (outbuf.buf, MC * N * 4)], [(uout.buf, 16)])
    return (cin, gem, cout, N, K)

for L in layers:
    ct, sn = cosMp[L["lt"]], sinMp[L["lt"]]; hd = L["hd"]; nkv = L["nkv"]; nq = L["nq"]; nk = L["nk"]
    L["p_in"] = ds(LN, [(Xp.buf, Xp.size), (L["inw"].buf, H * 4), (Np.buf, Np.size)], [(DUH.buf, 16)])
    L["p_q"] = gemm_ds(L["qp"], Np, Qp); L["p_k"] = gemm_ds(L["kp"], Np, Kp)
    if L["vp"]: L["p_v"] = gemm_ds(L["vp"], Np, Vp)
    Pp = buf_u32([nkv, hd, nk, 0, L["stride"], L["win"], 0, 0]); vsrc = Vp if L["vp"] else Kp
    L["p_prep"] = ds(LA, [(Kp.buf, Kp.size), (vsrc.buf, vsrc.size), (L["knw"].buf, hd * 4), (ct.buf, ct.size), (sn.buf, sn.size),
                          (L["kc"].buf, L["kc"].size), (L["vc"].buf, L["vc"].size)], [(Pp.buf, 32), (DYNP.buf, 16)])
    Aa = buf_u32([nH, nq, hd, L["stride"], L["win"], L["grp"], 0, 0])
    L["p_attn"] = ds(LA, [(Qp.buf, Qp.size), (L["qnw"].buf, hd * 4), (ct.buf, ct.size), (sn.buf, sn.size),
                          (L["kc"].buf, L["kc"].size), (L["vc"].buf, L["vc"].size), (Ap.buf, Ap.size)], [(Aa.buf, 32), (DYNP.buf, 16)])
    L["p_o"] = gemm_ds(L["op"], Ap, Tp)
    L["p_na"] = ds(LN, [(Tp.buf, Tp.size), (L["paw"].buf, H * 4), (Xp.buf, Xp.size)], [(DUH.buf, 16)])
    L["p_pfn"] = ds(LN, [(Xp.buf, Xp.size), (L["pfw"].buf, H * 4), (Np.buf, Np.size)], [(DUH.buf, 16)])
    L["p_g"] = gemm_ds(L["gp"], Np, Gp); L["p_u"] = gemm_ds(L["up"], Np, Upp)
    L["p_gm"] = ds(LN, [(Gp.buf, Gp.size), (Upp.buf, Upp.size), (ACp.buf, ACp.size)], [(DUIp.buf, 16)])
    L["p_d"] = gemm_ds(L["dp"], ACp, Tp)
    naspu = buf_u32([H, int(np.array([L["scl"]], np.float32).view(np.uint32)[0]), 0, 0])
    L["p_nas"] = ds(LN, [(Tp.buf, Tp.size), (L["ofw"].buf, H * 4), (Xp.buf, Xp.size)], [(naspu.buf, 16)])

def gd(C, pp, pl, dset, gx, gy=1):
    vk.vkCmdBindPipeline(C, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pp)
    vk.vkCmdBindDescriptorSets(C, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [dset], 0, None)
    vk.vkCmdDispatch(C, gx, gy, 1); vk.vkCmdPipelineBarrier(C, CS, CS, 0, 1, [MB], 0, None, 0, None)
def gmm(C, t, L=None, tag=""):   # fp16 coopmat: cast-down -> GEMM (grid N/64 x M/64) -> cast-up
    cin, gem, cout, N, K = t
    gd(C, P_castdn, PLca, cin, (MC * K + 63) // 64);   tsp(C, L, "cast")
    gd(C, P_cgemm16, PLcg, gem, N // 64, MC // 64);    tsp(C, L, tag + "_mm")
    gd(C, P_castup, PLca, cout, (MC * N + 63) // 64);  tsp(C, L, "cast")
# GROUP GLAY layers per submit: a full 48-layer chunk in one submit (~5s) trips the Windows TDR
# watchdog (~2s), but 1-per-layer wastes ~48 CPU<->GPU round-trips/chunk. Grouping cuts that overhead
# while staying well under TDR. Buffers persist across submits so the chunk flows.
# NB: tried GLAY=8 (6 submits/chunk vs 48) -> no measurable gain (34.6 vs 31.8s, within thermal noise);
# the per-layer submit/fence overhead is ~1.5% of prefill, below the noise floor. Kept GLAY=1.
GLAY = 1
def tsp(C, L, label):   # prefill timestamp (no-op unless PREPROF); indices are per-layer (pool reset each layer)
    if PREPROF:
        vk.vkCmdWriteTimestamp(C, _BOP, _qpre, len(L["_lbl"])); L["_lbl"].append(label)
def rec_layer(C, L):
    if PREPROF: vk.vkCmdResetQueryPool(C, _qpre, 0, 256); L["_lbl"] = []
    tsp(C, L, "start")
    gd(C, P_pn, PLpn, L["p_in"], MC); tsp(C, L, "norm")
    gmm(C, L["p_q"], L, "qkv"); gmm(C, L["p_k"], L, "qkv")
    if L["vp"]: gmm(C, L["p_v"], L, "qkv")
    gd(C, P_ppk, PLap, L["p_prep"], L["nkv"], MC); tsp(C, L, "prepkv")
    gd(C, P_pat, PLap, L["p_attn"], nH, MC); tsp(C, L, "attn")
    gmm(C, L["p_o"], L, "o")
    gd(C, P_pna, PLpn, L["p_na"], MC); tsp(C, L, "normadd")
    gd(C, P_pn, PLpn, L["p_pfn"], MC); tsp(C, L, "norm")
    gmm(C, L["p_g"], L, "gateup"); gmm(C, L["p_u"], L, "gateup")
    gd(C, P_gm, PLpn, L["p_gm"], (MC * I + 63) // 64); tsp(C, L, "gelu")
    gmm(C, L["p_d"], L, "down")
    gd(C, P_pna, PLpn, L["p_nas"], MC); tsp(C, L, "normadd")
ngrp = (len(layers) + GLAY - 1) // GLAY
cb_pre_L = vk.vkAllocateCommandBuffers(dev, vk.VkCommandBufferAllocateInfo(commandPool=cpool,
           level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=ngrp))
for gi in range(ngrp):
    C = cb_pre_L[gi]; vk.vkBeginCommandBuffer(C, vk.VkCommandBufferBeginInfo())
    for li in range(gi * GLAY, min((gi + 1) * GLAY, len(layers))):
        rec_layer(C, layers[li])
    vk.vkEndCommandBuffer(C)
print(f"[rec] prefill command buffers recorded ({ngrp} groups of <={GLAY} layers)", flush=True)

def set_rope_batch(p0, M):
    posids = torch.arange(p0, p0 + M).reshape(1, M)
    for lt in ("sliding_attention", "full_attention"):
        hdl = cfg.head_dim if lt == "sliding_attention" else cfg.global_head_dim
        cs, sn = tm.rotary_emb(torch.zeros(1, M, hdl), posids, layer_type=lt)
        cosMp[lt].write(cs.reshape(-1).float().numpy())   # .float() is already f32
        sinMp[lt].write(sn.reshape(-1).float().numpy())

def forward_prefill(chunk_ids, p0):
    Xp.write(EMB[np.array(chunk_ids)].reshape(-1))   # EMB is pre-scaled at load
    DYNP.write(np.array([p0, MC, 0, 0], np.uint32)); set_rope_batch(p0, MC)
    for li in range(len(cb_pre_L)):   # grouped submits (TDR-safe); buffers persist between submits
        vk.vkResetFences(dev, 1, [fence])
        vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cb_pre_L[li]])], fence)
        vk.vkWaitForFences(dev, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        if PREPROF: _read_preprofile(layers[li])

def _read_preprofile(L):
    n = len(L["_lbl"]); buf = ffi.new(f"uint64_t[{n}]")
    vk.vkGetQueryPoolResults(dev, _qpre, 0, n, n * 8, buf, 8, vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT)
    for i in range(1, n):
        d = (int(buf[i]) - int(buf[i - 1])) * _tsp_pre / 1000.0   # us
        _preagg[L["_lbl"][i]] = _preagg.get(L["_lbl"][i], 0.0) + d

def print_preprofile(nchunks):
    if not _preagg: return
    tot = sum(_preagg.values()) / nchunks
    print(f"\n=== prefill GPU profile (per {MC}-token chunk, avg of {nchunks} chunks) ===", flush=True)
    for k in sorted(_preagg, key=lambda x: -_preagg[x]):
        us = _preagg[k] / nchunks; print(f"  {k:14} {us:9.1f} us  {100*us/tot:5.1f}%", flush=True)
    print(f"  {'TOTAL':14} {tot:9.1f} us  -> {1e6*MC/tot:.1f} prompt-tok/s (GPU-only)", flush=True)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)


def sample(logits, temperature=0.0, top_p=1.0):
    """CPU-side next-token pick. Greedy when temperature<=0 (softcap is monotonic -> argmax invariant,
    so we skip it). Otherwise apply final-logit softcapping, temperature, nucleus top-p, then sample."""
    if temperature <= 0.0:
        return int(np.argmax(logits))
    z = (softcap * np.tanh(logits.astype(np.float64) / softcap)) / max(temperature, 1e-6)
    z -= z.max(); p = np.exp(z); p /= p.sum()
    if 0.0 < top_p < 1.0:
        order = np.argsort(p)[::-1]; cum = np.cumsum(p[order])
        cut = int(np.searchsorted(cum, top_p)) + 1
        mask = np.zeros_like(p); keep = order[:cut]; mask[keep] = p[keep]
        p = mask / mask.sum()
    return int(np.random.choice(p.shape[0], p=p))


def generate(ids, max_new=256, temperature=0.0, top_p=1.0, stop_ids=()):
    """Prefill `ids` then yield generated token ids one at a time (greedy or sampled). Each call
    starts at pos=0 so the KV cache (slots 0..pos) is overwritten cleanly -- no cross-request state.
    The fixed command buffer caps total context at MAXT; the prompt is left-truncated to fit."""
    ids = [int(i) for i in ids]
    if not ids: return   # empty prompt -> nothing to generate (avoids sample(None) crash)
    if len(ids) >= MAXT: ids = ids[-(MAXT - 1):]
    n = len(ids); nfull = (n - 1) // MC if n > MC else 0   # full MC-chunks among tokens 0..n-2
    for c in range(nfull): forward_prefill(ids[c * MC:(c + 1) * MC], c * MC)   # batched prefill
    logits = None
    for pos in range(nfull * MC, n): logits = forward(ids[pos], pos)           # tail + last via decode
    pos = n
    for _ in range(max_new):
        nxt = sample(logits, temperature, top_p)
        if nxt in stop_ids: return
        yield nxt
        if pos >= MAXT - 1: return
        logits = forward(nxt, pos); pos += 1


if __name__ == "__main__" and PREPROF:
    nch = 6                                   # warm + timed chunks of MC tokens (synthetic ids)
    rng = np.random.default_rng(0)
    for c in range(nch):
        forward_prefill([int(x) for x in rng.integers(0, V, MC)], c * MC)
    print_preprofile(nch)
    sys.exit(0)

if __name__ == "__main__":
    ids = tok.apply_chat_template([{"role": "user", "content": "What is the capital of France? One word."}],
                                  add_generation_prompt=True, tokenize=True, return_dict=False)
    ids = list(np.array(ids).ravel())
    logits = None
    for pos, tid in enumerate(ids): logits = forward(tid, pos)
    gen = []; t0 = time.time()
    for _ in range(NGEN):
        nxt = int(np.argmax(logits)); gen.append(nxt); logits = forward(nxt, len(ids) + len(gen) - 1)
    dt = time.time() - t0
    print("CONTINUATION:", repr(tok.decode(gen)))
    print(f"[VULKAN] decode {NGEN} tok in {dt:.2f}s = {NGEN/dt:.3f} tok/s", flush=True)
    if PROFILE: print_profile()
