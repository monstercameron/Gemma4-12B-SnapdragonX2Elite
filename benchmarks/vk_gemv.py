"""Minimal raw-Vulkan compute harness (vulkan cffi binding) — foundation for the command-buffer-
reuse engine. Runs the int4 GEMV+reduce (SPIR-V from vk/*.spv) on synthetic data, validates vs
numpy. Proves: Python -> raw Vulkan -> Adreno works, with host-visible (UMA) buffers.
"""
import os, numpy as np, vulkan as vk
ffi = vk.ffi

K, N, BLK, SPLIT = 3840, 4096, 32, 8
nblk = K // BLK; bpc = nblk // SPLIT
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- instance / device / compute queue ----
app = vk.VkApplicationInfo(pApplicationName="g", applicationVersion=0, pEngineName="g",
                           engineVersion=0, apiVersion=vk.VK_API_VERSION_1_0)
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
props = vk.vkGetPhysicalDeviceProperties(pdev)
print("device:", props.deviceName, flush=True)
qfams = vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev)
qfi = next(i for i, q in enumerate(qfams) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(pQueueCreateInfos=[
    vk.VkDeviceQueueCreateInfo(queueFamilyIndex=qfi, pQueuePriorities=[1.0])]), None)
queue = vk.vkGetDeviceQueue(dev, qfi, 0)

memp = vk.vkGetPhysicalDeviceMemoryProperties(pdev)
HV = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
def memtype(bits):
    for i in range(memp.memoryTypeCount):
        if (bits & (1 << i)) and (memp.memoryTypes[i].propertyFlags & HV) == HV:
            return i
    raise RuntimeError("no host-visible coherent memory")

USAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
def mkbuf(nbytes):
    buf = vk.vkCreateBuffer(dev, vk.VkBufferCreateInfo(size=nbytes, usage=USAGE,
                            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
    req = vk.vkGetBufferMemoryRequirements(dev, buf)
    mem = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(allocationSize=req.size,
                              memoryTypeIndex=memtype(req.memoryTypeBits)), None)
    vk.vkBindBufferMemory(dev, buf, mem, 0)
    return buf, mem, nbytes

def upload(mem, arr):
    n = arr.nbytes
    p = vk.vkMapMemory(dev, mem, 0, n, 0)
    ffi.memmove(p, arr.tobytes(), n)
    vk.vkUnmapMemory(dev, mem)

def download(mem, n):
    p = vk.vkMapMemory(dev, mem, 0, n, 0)
    out = bytes(p)
    vk.vkUnmapMemory(dev, mem)
    return out

# ---- data ----
rng = np.random.default_rng(0)
xa = (rng.standard_normal(K) * 0.5).astype(np.float32)
# build real int4 weights from a random fp32 W so we can validate
W = (rng.standard_normal((N, K)) * 0.1).astype(np.float32)
Wb = W.reshape(N, nblk, BLK); scale = np.maximum(np.abs(Wb).max(2) / 7.0, 1e-8).astype(np.float32)
q = np.clip(np.round(Wb / scale[:, :, None]) + 8, 0, 15).astype(np.uint8).reshape(N, K)
qT = np.ascontiguousarray(q.T); wpack = np.zeros((K, N // 8), np.uint32)
for j in range(8): wpack |= (qT[:, j::8].astype(np.uint32) & 15) << (j * 4)
scales = np.ascontiguousarray(scale.T).astype(np.float16).reshape(-1).view(np.uint32)
ref = (W @ xa).astype(np.float32)

bx, mx, _ = mkbuf(xa.nbytes); upload(mx, xa)
bw, mw, _ = mkbuf(wpack.nbytes); upload(mw, wpack.reshape(-1))
bs, ms, _ = mkbuf(scales.nbytes); upload(ms, scales)
bp, mp, _ = mkbuf(SPLIT * N * 4)
by, my, _ = mkbuf(N * 4)
d1 = np.array([N, N // 8, bpc, BLK], np.uint32); bd1, md1, _ = mkbuf(16); upload(md1, d1)
d2 = np.array([N, SPLIT, 0, 0], np.uint32); bd2, md2, _ = mkbuf(16); upload(md2, d2)

# ---- descriptor set layouts + pipelines ----
def spv(path):
    data = open(os.path.join(HERE, path), "rb").read()
    return vk.vkCreateShaderModule(dev, vk.VkShaderModuleCreateInfo(codeSize=len(data), pCode=data), None)

def dsl(n_storage, has_uniform):
    binds = [vk.VkDescriptorSetLayoutBinding(binding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
             descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(n_storage)]
    if has_uniform:
        binds.append(vk.VkDescriptorSetLayoutBinding(binding=n_storage,
                     descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1,
                     stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT))
    return vk.vkCreateDescriptorSetLayout(dev, vk.VkDescriptorSetLayoutCreateInfo(pBindings=binds), None)

def pipeline(spv_mod, layout):
    pl = vk.vkCreatePipelineLayout(dev, vk.VkPipelineLayoutCreateInfo(pSetLayouts=[layout]), None)
    stage = vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=spv_mod, pName="main")
    p = vk.vkCreateComputePipelines(dev, vk.VK_NULL_HANDLE, 1,
        [vk.VkComputePipelineCreateInfo(stage=stage, layout=pl)], None)[0]
    return p, pl

L1 = dsl(4, True); L2 = dsl(2, True)
P1, PL1 = pipeline(spv("vk/gemv.spv"), L1)
P2, PL2 = pipeline(spv("vk/reduce.spv"), L2)

pool = vk.vkCreateDescriptorPool(dev, vk.VkDescriptorPoolCreateInfo(maxSets=2, pPoolSizes=[
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=8),
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=2)], flags=0), None)
sets = vk.vkAllocateDescriptorSets(dev, vk.VkDescriptorSetAllocateInfo(descriptorPool=pool, pSetLayouts=[L1, L2]))
DS1, DS2 = sets[0], sets[1]

def binfo(buf, sz): return vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=sz)
def write_ds(ds, items):  # items: list of (binding, buf, size, is_uniform)
    writes = [vk.VkWriteDescriptorSet(dstSet=ds, dstBinding=b, descriptorCount=1,
              descriptorType=(vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER if u else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
              pBufferInfo=[binfo(buf, sz)]) for (b, buf, sz, u) in items]
    vk.vkUpdateDescriptorSets(dev, len(writes), writes, 0, None)

write_ds(DS1, [(0, bx, xa.nbytes, 0), (1, bw, wpack.nbytes, 0), (2, bs, scales.nbytes, 0),
               (3, bp, SPLIT * N * 4, 0), (4, bd1, 16, 1)])
write_ds(DS2, [(0, bp, SPLIT * N * 4, 0), (1, by, N * 4, 0), (2, bd2, 16, 1)])

# ---- record command buffer (ONCE) ----
cpool = vk.vkCreateCommandPool(dev, vk.VkCommandPoolCreateInfo(queueFamilyIndex=qfi), None)
cb = vk.vkAllocateCommandBuffers(dev, vk.VkCommandBufferAllocateInfo(commandPool=cpool,
     level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, P1)
vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, PL1, 0, 1, [DS1], 0, None)
vk.vkCmdDispatch(cb, (N // 16 + 63) // 64, SPLIT, 1)  # uvec2 GEMV: 16 outputs/thread
bar = vk.VkMemoryBarrier(srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT, dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT)
vk.vkCmdPipelineBarrier(cb, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 1, [bar], 0, None, 0, None)
vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, P2)
vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, PL2, 0, 1, [DS2], 0, None)
vk.vkCmdDispatch(cb, (N // 4 + 63) // 64, 1, 1)  # vec4 reduce
vk.vkEndCommandBuffer(cb)

fence = vk.vkCreateFence(dev, vk.VkFenceCreateInfo(), None)
import time
def run():
    vk.vkResetFences(dev, 1, [fence])
    vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cb])], fence)
    vk.vkWaitForFences(dev, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)

run()
y = np.frombuffer(download(my, N * 4), np.float32)
cos = float(np.dot(y, ref) / (np.linalg.norm(y) * np.linalg.norm(ref) + 1e-9))
print(f"GEMV cos vs fp32 ref = {cos:.5f}  {'OK' if cos > 0.98 else 'FAIL'}", flush=True)
# replay timing (the whole point: command buffer recorded once, resubmitted)
for _ in range(20): run()
t = time.time()
for _ in range(200): run()
ms = (time.time() - t) / 200 * 1e3
print(f"replay (resubmit same cmd buffer): {ms:.3f} ms/dispatch  {N*K//2/(ms/1e3)/1e9:.1f} GB/s", flush=True)
