"""Validate + benchmark the int4-weight coopmat GEMM (vk/coopgemm_i4.spv). int4 weights dequantized to
fp16 in shared, coopMatMulAdd. cos vs the true dequant-GEMM; GFLOP/s at real prefill dims (M=128)."""
import os, time, numpy as np, vulkan as vk
ffi = vk.ffi; BLK = 32
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = vk.VkApplicationInfo(apiVersion=(1 << 22) | (1 << 12))
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
qfi = next(i for i, q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev)) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
f16 = vk.VkPhysicalDeviceShaderFloat16Int8Features(shaderFloat16=vk.VK_TRUE)
st16 = vk.VkPhysicalDevice16BitStorageFeatures(storageBuffer16BitAccess=vk.VK_TRUE, pNext=ffi.addressof(f16))
mm = vk.VkPhysicalDeviceVulkanMemoryModelFeatures(vulkanMemoryModel=vk.VK_TRUE, pNext=ffi.addressof(st16))
coop = vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(cooperativeMatrix=vk.VK_TRUE, pNext=ffi.addressof(mm))
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(pNext=ffi.addressof(coop), pQueueCreateInfos=[
    vk.VkDeviceQueueCreateInfo(queueFamilyIndex=qfi, pQueuePriorities=[1.0])],
    ppEnabledExtensionNames=["VK_KHR_cooperative_matrix", "VK_KHR_vulkan_memory_model", "VK_KHR_16bit_storage", "VK_KHR_shader_float16_int8"]), None)
queue = vk.vkGetDeviceQueue(dev, qfi, 0)
memp = vk.vkGetPhysicalDeviceMemoryProperties(pdev); HV = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
def memtype(b):
    for i in range(memp.memoryTypeCount):
        if (b & (1 << i)) and (memp.memoryTypes[i].propertyFlags & HV) == HV: return i
USAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
def mkbuf(n):
    b = vk.vkCreateBuffer(dev, vk.VkBufferCreateInfo(size=n, usage=USAGE, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE), None)
    r = vk.vkGetBufferMemoryRequirements(dev, b); m = vk.vkAllocateMemory(dev, vk.VkMemoryAllocateInfo(allocationSize=r.size, memoryTypeIndex=memtype(r.memoryTypeBits)), None)
    vk.vkBindBufferMemory(dev, b, m, 0); return b, m
def upb(m, a): p = vk.vkMapMemory(dev, m, 0, a.nbytes, 0); ffi.memmove(p, a.tobytes(), a.nbytes); vk.vkUnmapMemory(dev, m)
def dnb(m, n): p = vk.vkMapMemory(dev, m, 0, n, 0); o = bytes(p); vk.vkUnmapMemory(dev, m); return o
data = open(os.path.join(HERE, "vk/coopgemm_i4.spv"), "rb").read()
mod = vk.vkCreateShaderModule(dev, vk.VkShaderModuleCreateInfo(codeSize=len(data), pCode=data), None)
binds = [vk.VkDescriptorSetLayoutBinding(binding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(4)]
binds.append(vk.VkDescriptorSetLayoutBinding(binding=4, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT))
dsl = vk.vkCreateDescriptorSetLayout(dev, vk.VkDescriptorSetLayoutCreateInfo(pBindings=binds), None)
pl = vk.vkCreatePipelineLayout(dev, vk.VkPipelineLayoutCreateInfo(pSetLayouts=[dsl]), None)
pipe = vk.vkCreateComputePipelines(dev, vk.VK_NULL_HANDLE, 1, [vk.VkComputePipelineCreateInfo(
    stage=vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main"), layout=pl)], None)[0]
pool = vk.vkCreateDescriptorPool(dev, vk.VkDescriptorPoolCreateInfo(maxSets=4, pPoolSizes=[
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=16),
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=4)]), None)
cpool = vk.vkCreateCommandPool(dev, vk.VkCommandPoolCreateInfo(queueFamilyIndex=qfi), None)
fence = vk.vkCreateFence(dev, vk.VkFenceCreateInfo(), None)
def bench(M, N, K):
    nblk = K // BLK; rng = np.random.default_rng(0)
    Xf = (rng.standard_normal((M, K)) * 0.1).astype(np.float32); X16 = Xf.astype(np.float16)
    Wf = (rng.standard_normal((N, K)) * 0.05).astype(np.float32)
    Wb = Wf.reshape(N, nblk, BLK); scale = np.maximum(np.abs(Wb).max(2) / 7.0, 1e-8).astype(np.float32)
    q = np.clip(np.round(Wb / scale[:, :, None]) + 8, 0, 15).astype(np.uint32).reshape(N, K)
    qT = np.ascontiguousarray(q.T); wp = np.zeros((K, N // 8), np.uint32)
    for j in range(8): wp |= (qT[:, j::8] & 15) << (j * 4)
    scb = np.ascontiguousarray(scale.T).astype(np.float16).reshape(-1).view(np.uint32)
    Wdq = ((q.astype(np.float32) - 8).reshape(N, nblk, BLK) * scale[:, :, None]).reshape(N, K)
    ref = (X16.astype(np.float32) @ Wdq.T)
    bx, mx = mkbuf(X16.nbytes); upb(mx, X16.reshape(-1))
    bw, mw = mkbuf(wp.nbytes); upb(mw, wp.reshape(-1))
    bs, ms = mkbuf(scb.nbytes); upb(ms, scb)
    by, my = mkbuf(M * N * 2); bd, md = mkbuf(16); upb(md, np.array([M, N, K, N // 8], np.uint32))
    dset = vk.vkAllocateDescriptorSets(dev, vk.VkDescriptorSetAllocateInfo(descriptorPool=pool, pSetLayouts=[dsl]))[0]
    def wr(b, i, sz, u): return vk.VkWriteDescriptorSet(dstSet=dset, dstBinding=i, descriptorCount=1, descriptorType=(vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER if u else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=b, offset=0, range=sz)])
    vk.vkUpdateDescriptorSets(dev, 5, [wr(bx, 0, X16.nbytes, 0), wr(bw, 1, wp.nbytes, 0), wr(bs, 2, scb.nbytes, 0), wr(by, 3, M * N * 2, 0), wr(bd, 4, 16, 1)], 0, None)
    cb = vk.vkAllocateCommandBuffers(dev, vk.VkCommandBufferAllocateInfo(commandPool=cpool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
    vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
    vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe)
    vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [dset], 0, None)
    vk.vkCmdDispatch(cb, N // 64, M // 64, 1); vk.vkEndCommandBuffer(cb)
    def run(): vk.vkResetFences(dev, 1, [fence]); vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cb])], fence); vk.vkWaitForFences(dev, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
    run(); C = np.frombuffer(dnb(my, M * N * 2), np.float16).astype(np.float32).reshape(M, N)
    cos = float((C.ravel() @ ref.ravel()) / (np.linalg.norm(C) * np.linalg.norm(ref) + 1e-9))
    for _ in range(20): run()
    t = time.time()
    for _ in range(100): run()
    msr = (time.time() - t) / 100 * 1e3
    print(f"  M{M} N{N} K{K}: cos={cos:.4f}  {msr:.3f} ms  {2*M*N*K/(msr/1e3)/1e9:.0f} GFLOP/s", flush=True)
print("int4 coopmat GEMM (M=128 prefill chunk):", flush=True)
bench(128, 3840, 15360)   # down
bench(128, 15360, 3840)   # gate
