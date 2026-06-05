"""De-risk W4A8 prefill GEMM: int8-activation x int4-weight coopmat (vk/coopgemm_w4a8.spv) vs the
shipped f16 path (vk/coopgemm_i4h.spv), raw Vulkan, no model load. Measures BOTH (a) real speedup at a
prefill shape and (b) quality: cos of W4A8 output vs the f16-path output (isolating the extra error from
int8 activations + on-the-fly int8 requant, beyond the int4 weight error both share)."""
import os, time, numpy as np, vulkan as vk
ffi = vk.ffi
M, K, N = 128, 3840, 15360          # gate_proj prefill shape (M=chunk, K=hidden, N=intermediate)
BLK = 32; nblk = K // BLK
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = vk.VkApplicationInfo(apiVersion=(1 << 22) | (1 << 12))
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
qfi = next(i for i, q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev)) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
f16 = vk.VkPhysicalDeviceShaderFloat16Int8Features(shaderFloat16=vk.VK_TRUE, shaderInt8=vk.VK_TRUE)
st8 = vk.VkPhysicalDevice8BitStorageFeatures(storageBuffer8BitAccess=vk.VK_TRUE, pNext=ffi.addressof(f16))
st16 = vk.VkPhysicalDevice16BitStorageFeatures(storageBuffer16BitAccess=vk.VK_TRUE, pNext=ffi.addressof(st8))
mm = vk.VkPhysicalDeviceVulkanMemoryModelFeatures(vulkanMemoryModel=vk.VK_TRUE, pNext=ffi.addressof(st16))
coop = vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(cooperativeMatrix=vk.VK_TRUE, pNext=ffi.addressof(mm))
exts = ["VK_KHR_cooperative_matrix", "VK_KHR_vulkan_memory_model", "VK_KHR_16bit_storage", "VK_KHR_8bit_storage", "VK_KHR_shader_float16_int8"]
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(pNext=ffi.addressof(coop), pQueueCreateInfos=[
    vk.VkDeviceQueueCreateInfo(queueFamilyIndex=qfi, pQueuePriorities=[1.0])], ppEnabledExtensionNames=exts), None)
queue = vk.vkGetDeviceQueue(dev, qfi, 0)
memp = vk.vkGetPhysicalDeviceMemoryProperties(pdev)
HV = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
def memtype(bits):
    for i in range(memp.memoryTypeCount):
        if (bits & (1 << i)) and (memp.memoryTypes[i].propertyFlags & HV) == HV: return i
USAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
def mkbuf(nbytes):
    b = vk.vkCreateBuffer(dev, vk.VkBufferCreateInfo(size=nbytes, usage=USAGE, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
    req = vk.vkGetBufferMemoryRequirements(dev, b)
    mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(allocationSize=req.size, memoryTypeIndex=memtype(req.memoryTypeBits)), None)
    vk.vkBindBufferMemory(dev, b, mem, 0); return b, mem
def up(mem, arr):
    p = vk.vkMapMemory(dev, mem, 0, arr.nbytes, 0); ffi.memmove(p, arr.tobytes(), arr.nbytes); vk.vkUnmapMemory(dev, mem)
def dn(mem, n):
    p = vk.vkMapMemory(dev, mem, 0, n, 0); o = bytes(p); vk.vkUnmapMemory(dev, mem); return o
def mkpipe(spv, nbind):
    data = open(os.path.join(HERE, spv), "rb").read()
    mod = vk.vkCreateShaderModule(dev, vk.VkShaderModuleCreateInfo(codeSize=len(data), pCode=data), None)
    binds = [vk.VkDescriptorSetLayoutBinding(binding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(nbind)]
    binds.append(vk.VkDescriptorSetLayoutBinding(binding=nbind, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT))
    dsl = vk.vkCreateDescriptorSetLayout(dev, vk.VkDescriptorSetLayoutCreateInfo(pBindings=binds), None)
    pl = vk.vkCreatePipelineLayout(dev, vk.VkPipelineLayoutCreateInfo(pSetLayouts=[dsl]), None)
    stage = vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main")
    p = vk.vkCreateComputePipelines(dev, vk.VK_NULL_HANDLE, 1, [vk.VkComputePipelineCreateInfo(stage=stage, layout=pl)], None)[0]
    return p, pl, dsl
pool = vk.vkCreateDescriptorPool(dev, vk.VkDescriptorPoolCreateInfo(maxSets=4, pPoolSizes=[
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=40),
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=8)]), None)
def dset(dsl, bufs, unif):  # bufs=[(buf,size)], unif=(buf,size)
    s = vk.vkAllocateDescriptorSets(dev, vk.VkDescriptorSetAllocateInfo(descriptorPool=pool, pSetLayouts=[dsl]))[0]
    w = [vk.VkWriteDescriptorSet(dstSet=s, dstBinding=i, descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=b, offset=0, range=sz)]) for i, (b, sz) in enumerate(bufs)]
    w.append(vk.VkWriteDescriptorSet(dstSet=s, dstBinding=len(bufs), descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
             pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=unif[0], offset=0, range=unif[1])]))
    vk.vkUpdateDescriptorSets(dev, len(w), w, 0, None); return s
cpool = vk.vkCreateCommandPool(dev, vk.VkCommandPoolCreateInfo(queueFamilyIndex=qfi), None)
fence = vk.vkCreateFence(dev, vk.VkFenceCreateInfo(), None)
def timeit(pipe, pl, s, gx, gy):
    cb = vk.vkAllocateCommandBuffers(dev, vk.VkCommandBufferAllocateInfo(commandPool=cpool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
    vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
    vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe)
    vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [s], 0, None)
    vk.vkCmdDispatch(cb, gx, gy, 1); vk.vkEndCommandBuffer(cb)
    def run():
        vk.vkResetFences(dev, 1, [fence]); vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cb])], fence)
        vk.vkWaitForFences(dev, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
    for _ in range(20): run()
    t = time.time(); R = 200
    for _ in range(R): run()
    return (time.time() - t) / R * 1e3

rng = np.random.default_rng(0)
# ---- weights: f32 [N,K] -> int4 per-block (replicate engine quant) ----
W = (rng.standard_normal((N, K)) * 0.08).astype(np.float32)
Wb = W.reshape(N, nblk, BLK); sc = np.maximum(np.abs(Wb).max(2) / 7.0, 1e-8).astype(np.float32)   # [N,nblk]
q = np.clip(np.round(Wb / sc[:, :, None]) + 8, 0, 15).astype(np.uint8).reshape(N, K)               # [N,K]
deqW = ((q.astype(np.float32) - 8.0) * np.repeat(sc, BLK, axis=1))                                  # [N,K] int4-dequant weight
qT = np.ascontiguousarray(q.T); wp = np.zeros((K, N // 8), np.uint32)
for j in range(8): wp |= (qT[:, j::8].astype(np.uint32) & 15) << (j * 4)
# per-column int8 weight scale + ratio (fold per-block scale into requant). column n = output row of W[N,K]
s_w = np.maximum(np.abs(deqW).max(1) / 127.0, 1e-12).astype(np.float32)                             # [N]
ratio = (sc / s_w[:, None]).T.astype(np.float16)                                                    # [nblk,N] = wscale[n,b]/s_w[n]
# ---- activations: f32 [M,K] -> int8 per-row ----
X = (rng.standard_normal((M, K)) * 0.3).astype(np.float32)
s_a = np.maximum(np.abs(X).max(1) / 127.0, 1e-12).astype(np.float32)                                # [M]
A_i8 = np.clip(np.round(X / s_a[:, None]), -127, 127).astype(np.int8)                               # [M,K]

# ---- W4A8 run ----
bA, mA = mkbuf(A_i8.nbytes); up(mA, A_i8.reshape(-1))
bW, mW = mkbuf(wp.nbytes); up(mW, wp.reshape(-1))
bR, mR = mkbuf(ratio.nbytes); up(mR, np.ascontiguousarray(ratio).reshape(-1))   # raw fp16 bytes; shader unpacks
bSA, mSA = mkbuf(s_a.nbytes); up(mSA, s_a)
sw16 = s_w.astype(np.float16); bSW, mSW = mkbuf(sw16.nbytes); up(mSW, sw16)
bY, mY = mkbuf(M * N * 2)
bU, mU = mkbuf(16); up(mU, np.array([M, N, K, N // 8], np.uint32))
p1, pl1, dsl1 = mkpipe("vk/coopgemm_w4a8.spv", 6)
s1 = dset(dsl1, [(bA, A_i8.nbytes), (bW, wp.nbytes), (bR, ratio.nbytes), (bSA, s_a.nbytes), (bSW, sw16.nbytes), (bY, M * N * 2)], (bU, 16))
ms1 = timeit(p1, pl1, s1, N // 64, M // 64)
Yw = np.frombuffer(dn(mY, M * N * 2), np.float16).astype(np.float32).reshape(M, N)

# ---- references ----
Y_full = X @ W.T                       # full precision (original weights)
Y_f16path = X @ deqW.T                 # what the f16 engine path computes (int4 weights, ~f16 acts)
def cos(a, b): return float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
print(f"shape M={M} K={K} N={N}", flush=True)
print(f"  W4A8 cos vs f16-path  = {cos(Yw, Y_f16path):.5f}  (isolates int8-act + requant error)", flush=True)
print(f"  W4A8 cos vs full-prec = {cos(Yw, Y_full):.5f}", flush=True)
print(f"  f16path cos vs full   = {cos(Y_f16path, Y_full):.5f}  (int4 weight error baseline)", flush=True)
gf = 2 * M * N * K / 1e9
print(f"  W4A8: {ms1:.3f} ms  ({gf/(ms1/1e3):.0f} GFLOP/s)", flush=True)

# ---- f16 path timing at same shape (coopgemm_i4h) ----
Xf16 = X.astype(np.float16); scb = np.ascontiguousarray(sc.T).astype(np.float16).reshape(-1).view(np.uint32)
bXf, mXf = mkbuf(Xf16.nbytes); up(mXf, Xf16.reshape(-1).view(np.uint16))
bScb, mScb = mkbuf(scb.nbytes); up(mScb, scb)
bYf, mYf = mkbuf(M * N * 2)
bU2, mU2 = mkbuf(16); up(mU2, np.array([N, K, M, N // 8], np.uint32))
p2, pl2, dsl2 = mkpipe("vk/coopgemm_i4h.spv", 4)
s2 = dset(dsl2, [(bXf, Xf16.nbytes), (bW, wp.nbytes), (bScb, scb.nbytes), (bYf, M * N * 2)], (bU2, 16))
ms2 = timeit(p2, pl2, s2, N // 64, M // 64)
print(f"  f16 (i4h): {ms2:.3f} ms  ({gf/(ms2/1e3):.0f} GFLOP/s)", flush=True)
print(f"  >>> W4A8 (on-the-fly requant) speedup = {ms2/ms1:.2f}x", flush=True)

# ---- W8A8: precomputed int8 weights (no requant) -> realistic int8 ceiling ----
Wi8 = np.clip(np.round(deqW / s_w[:, None]), -127, 127).astype(np.int8)   # [N,K]
Wi8T = np.ascontiguousarray(Wi8.T)                                        # [K,N] row-major for coopMatLoad B
bWi8, mWi8 = mkbuf(Wi8T.nbytes); up(mWi8, Wi8T.reshape(-1))
bY3, mY3 = mkbuf(M * N * 2)
bU3, mU3 = mkbuf(16); up(mU3, np.array([M, N, K, 0], np.uint32))
p3, pl3, dsl3 = mkpipe("vk/coopgemm_w8a8.spv", 5)
s3 = dset(dsl3, [(bA, A_i8.nbytes), (bWi8, Wi8T.nbytes), (bSA, s_a.nbytes), (bSW, sw16.nbytes), (bY3, M * N * 2)], (bU3, 16))
ms3 = timeit(p3, pl3, s3, N // 64, M // 64)
Yw8 = np.frombuffer(dn(mY3, M * N * 2), np.float16).astype(np.float32).reshape(M, N)
print(f"  W8A8 cos vs f16-path  = {cos(Yw8, Y_f16path):.5f}", flush=True)
print(f"  W8A8 (precomputed int8): {ms3:.3f} ms  ({gf/(ms3/1e3):.0f} GFLOP/s)", flush=True)
print(f"  >>> W8A8 speedup vs f16 = {ms2/ms3:.2f}x  (weight storage 2x)", flush=True)
