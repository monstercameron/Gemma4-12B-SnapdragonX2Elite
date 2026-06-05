"""De-risk coopmat on Adreno X2: run the fp16 64x64xK coopmat GEMM (vk/coopgemm_i8.spv) via raw Vulkan,
validate vs numpy, time it. Needs the cooperativeMatrix + memory-model + fp16 device features enabled."""
import os, time, numpy as np, vulkan as vk
ffi = vk.ffi
M, N, K = 1024, 1024, 2048
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = vk.VkApplicationInfo(apiVersion=(1 << 22) | (1 << 12))   # Vulkan 1.1
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
# subgroup size
sgp = vk.VkPhysicalDeviceSubgroupProperties()
p2 = vk.VkPhysicalDeviceProperties2(pNext=ffi.addressof(sgp))
vk.vkGetPhysicalDeviceProperties2(pdev, p2)
SG = sgp.subgroupSize; print("subgroup size:", SG, flush=True)
qfi = next(i for i, q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev)) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
# feature chain: coop -> memmodel -> 16bitstorage -> f16int8
f16 = vk.VkPhysicalDeviceShaderFloat16Int8Features(shaderFloat16=vk.VK_TRUE, shaderInt8=vk.VK_TRUE)
st8 = vk.VkPhysicalDevice8BitStorageFeatures(storageBuffer8BitAccess=vk.VK_TRUE, pNext=ffi.addressof(f16)); st16 = vk.VkPhysicalDevice16BitStorageFeatures(storageBuffer16BitAccess=vk.VK_TRUE, pNext=ffi.addressof(st8))
mm = vk.VkPhysicalDeviceVulkanMemoryModelFeatures(vulkanMemoryModel=vk.VK_TRUE, pNext=ffi.addressof(st16))
coop = vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(cooperativeMatrix=vk.VK_TRUE, pNext=ffi.addressof(mm))
exts = ["VK_KHR_cooperative_matrix", "VK_KHR_vulkan_memory_model", "VK_KHR_16bit_storage", "VK_KHR_8bit_storage", "VK_KHR_shader_float16_int8"]
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(pNext=ffi.addressof(coop), pQueueCreateInfos=[
    vk.VkDeviceQueueCreateInfo(queueFamilyIndex=qfi, pQueuePriorities=[1.0])], ppEnabledExtensionNames=exts), None)
print("device created with coopmat feature", flush=True)
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
rng = np.random.default_rng(0)
A = rng.integers(-8,8,size=(M,K),dtype=np.int8)
B = rng.integers(-8,8,size=(K,N),dtype=np.int8)
ref = (A.astype(np.int32) @ B.astype(np.int32))
ba, ma = mkbuf(A.nbytes); up(ma, A.reshape(-1))
bb, mb = mkbuf(B.nbytes); up(mb, B.reshape(-1))
bc, mc = mkbuf(M * N * 4)
bd, md = mkbuf(16); up(md, np.array([M, N, K], np.uint32))
data = open(os.path.join(HERE, "vk/coopgemm_i8.spv"), "rb").read()
mod = vk.vkCreateShaderModule(dev, vk.VkShaderModuleCreateInfo(codeSize=len(data), pCode=data), None)
binds = [vk.VkDescriptorSetLayoutBinding(binding=i, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(3)]
binds.append(vk.VkDescriptorSetLayoutBinding(binding=3, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT))
dsl = vk.vkCreateDescriptorSetLayout(dev, vk.VkDescriptorSetLayoutCreateInfo(pBindings=binds), None)
pl = vk.vkCreatePipelineLayout(dev, vk.VkPipelineLayoutCreateInfo(pSetLayouts=[dsl]), None)
# spec constant 0 = subgroup size (local_size_x)
stage = vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=mod, pName="main")
pipe = vk.vkCreateComputePipelines(dev, vk.VK_NULL_HANDLE, 1, [vk.VkComputePipelineCreateInfo(stage=stage, layout=pl)], None)[0]
print("pipeline created", flush=True)
pool = vk.vkCreateDescriptorPool(dev, vk.VkDescriptorPoolCreateInfo(maxSets=1, pPoolSizes=[
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=3),
    vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, descriptorCount=1)]), None)
dset = vk.vkAllocateDescriptorSets(dev, vk.VkDescriptorSetAllocateInfo(descriptorPool=pool, pSetLayouts=[dsl]))[0]
def wr(b, i, sz, u): return vk.VkWriteDescriptorSet(dstSet=dset, dstBinding=i, descriptorCount=1,
    descriptorType=(vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER if u else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER), pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=b, offset=0, range=sz)])
vk.vkUpdateDescriptorSets(dev, 4, [wr(ba, 0, A.nbytes, 0), wr(bb, 1, B.nbytes, 0), wr(bc, 2, M*N*4, 0), wr(bd, 3, 16, 1)], 0, None)
cpool = vk.vkCreateCommandPool(dev, vk.VkCommandPoolCreateInfo(queueFamilyIndex=qfi), None)
cb = vk.vkAllocateCommandBuffers(dev, vk.VkCommandBufferAllocateInfo(commandPool=cpool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1))[0]
vk.vkBeginCommandBuffer(cb, vk.VkCommandBufferBeginInfo())
vk.vkCmdBindPipeline(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipe)
vk.vkCmdBindDescriptorSets(cb, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, [dset], 0, None)
vk.vkCmdDispatch(cb, N//64, M//64, 1)
vk.vkEndCommandBuffer(cb)
fence = vk.vkCreateFence(dev, vk.VkFenceCreateInfo(), None)
def run():
    vk.vkResetFences(dev, 1, [fence]); vk.vkQueueSubmit(queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[cb])], fence)
    vk.vkWaitForFences(dev, 1, [fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
run()
C = np.frombuffer(dn(mc, M*N*4), np.int32).astype(np.float32).reshape(M, N)
cos = float((C.ravel() @ ref.ravel()) / (np.linalg.norm(C) * np.linalg.norm(ref) + 1e-9))
print(f"COOPMAT GEMM cos vs fp32 ref = {cos:.5f}  {'OK' if cos > 0.99 else 'FAIL'}", flush=True)
for _ in range(50): run()
t = time.time()
for _ in range(500): run()
ms = (time.time() - t) / 500 * 1e3
print(f"{M}x{N}x{K} coopmat tiled ({(M//64)*(N//64)} wg): {ms:.2f} ms  ({2*M*N*K/(ms/1e3)/1e9:.0f} GFLOP/s aggregate)", flush=True)
