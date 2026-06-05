"""De-risk coopmat attention: validate the transposed-K score matmul S=Q@K^T + measure throughput."""
import os, time, numpy as np, vulkan as vk
ffi = vk.ffi; M, T, hd = 128, 768, 256
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = vk.VkApplicationInfo(apiVersion=(1 << 22) | (1 << 12))
inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(pApplicationInfo=app), None)
pdev = vk.vkEnumeratePhysicalDevices(inst)[0]
qfi = next(i for i,q in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(pdev)) if q.queueFlags & vk.VK_QUEUE_COMPUTE_BIT)
f16 = vk.VkPhysicalDeviceShaderFloat16Int8Features(shaderFloat16=vk.VK_TRUE)
st16 = vk.VkPhysicalDevice16BitStorageFeatures(storageBuffer16BitAccess=vk.VK_TRUE, pNext=ffi.addressof(f16))
mm = vk.VkPhysicalDeviceVulkanMemoryModelFeatures(vulkanMemoryModel=vk.VK_TRUE, pNext=ffi.addressof(st16))
coop = vk.VkPhysicalDeviceCooperativeMatrixFeaturesKHR(cooperativeMatrix=vk.VK_TRUE, pNext=ffi.addressof(mm))
dev = vk.vkCreateDevice(pdev, vk.VkDeviceCreateInfo(pNext=ffi.addressof(coop), pQueueCreateInfos=[vk.VkDeviceQueueCreateInfo(queueFamilyIndex=qfi, pQueuePriorities=[1.0])], ppEnabledExtensionNames=["VK_KHR_cooperative_matrix","VK_KHR_vulkan_memory_model","VK_KHR_16bit_storage","VK_KHR_shader_float16_int8"]), None)
queue = vk.vkGetDeviceQueue(dev, qfi, 0); memp = vk.vkGetPhysicalDeviceMemoryProperties(pdev)
HV = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
def mt(b):
    for i in range(memp.memoryTypeCount):
        if (b & (1<<i)) and (memp.memoryTypes[i].propertyFlags & HV)==HV: return i
USAGE = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
def mk(n):
    b=vk.vkCreateBuffer(dev,vk.VkBufferCreateInfo(size=n,usage=USAGE,sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE),None)
    r=vk.vkGetBufferMemoryRequirements(dev,b); m=vk.vkAllocateMemory(dev,vk.VkMemoryAllocateInfo(allocationSize=r.size,memoryTypeIndex=mt(r.memoryTypeBits)),None); vk.vkBindBufferMemory(dev,b,m,0); return b,m
def up(m,a): p=vk.vkMapMemory(dev,m,0,a.nbytes,0); ffi.memmove(p,a.tobytes(),a.nbytes); vk.vkUnmapMemory(dev,m)
def dn(m,n): p=vk.vkMapMemory(dev,m,0,n,0); o=bytes(p); vk.vkUnmapMemory(dev,m); return o
rng=np.random.default_rng(0)
Q=(rng.standard_normal((M,hd))*0.1).astype(np.float16); K=(rng.standard_normal((T,hd))*0.1).astype(np.float16)
ref=(Q.astype(np.float32)@K.astype(np.float32).T)
bq,mq=mk(Q.nbytes);up(mq,Q.reshape(-1)); bk,mkk=mk(K.nbytes);up(mkk,K.reshape(-1))
bs,ms=mk(M*T*2); bd,md=mk(16);up(md,np.array([M,T,hd],np.uint32))
data=open(os.path.join(HERE,"vk/coopqk.spv"),"rb").read(); mod=vk.vkCreateShaderModule(dev,vk.VkShaderModuleCreateInfo(codeSize=len(data),pCode=data),None)
bn=[vk.VkDescriptorSetLayoutBinding(binding=i,descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,descriptorCount=1,stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT) for i in range(3)]
bn.append(vk.VkDescriptorSetLayoutBinding(binding=3,descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,descriptorCount=1,stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT))
dsl=vk.vkCreateDescriptorSetLayout(dev,vk.VkDescriptorSetLayoutCreateInfo(pBindings=bn),None)
pl=vk.vkCreatePipelineLayout(dev,vk.VkPipelineLayoutCreateInfo(pSetLayouts=[dsl]),None)
pipe=vk.vkCreateComputePipelines(dev,vk.VK_NULL_HANDLE,1,[vk.VkComputePipelineCreateInfo(stage=vk.VkPipelineShaderStageCreateInfo(stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,module=mod,pName="main"),layout=pl)],None)[0]
pool=vk.vkCreateDescriptorPool(dev,vk.VkDescriptorPoolCreateInfo(maxSets=1,pPoolSizes=[vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,descriptorCount=3),vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,descriptorCount=1)]),None)
ds=vk.vkAllocateDescriptorSets(dev,vk.VkDescriptorSetAllocateInfo(descriptorPool=pool,pSetLayouts=[dsl]))[0]
def wr(b,i,sz,u): return vk.VkWriteDescriptorSet(dstSet=ds,dstBinding=i,descriptorCount=1,descriptorType=(vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER if u else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),pBufferInfo=[vk.VkDescriptorBufferInfo(buffer=b,offset=0,range=sz)])
vk.vkUpdateDescriptorSets(dev,4,[wr(bq,0,Q.nbytes,0),wr(bk,1,K.nbytes,0),wr(bs,2,M*T*2,0),wr(bd,3,16,1)],0,None)
cp=vk.vkCreateCommandPool(dev,vk.VkCommandPoolCreateInfo(queueFamilyIndex=qfi),None)
cb=vk.vkAllocateCommandBuffers(dev,vk.VkCommandBufferAllocateInfo(commandPool=cp,level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,commandBufferCount=1))[0]
vk.vkBeginCommandBuffer(cb,vk.VkCommandBufferBeginInfo()); vk.vkCmdBindPipeline(cb,vk.VK_PIPELINE_BIND_POINT_COMPUTE,pipe)
vk.vkCmdBindDescriptorSets(cb,vk.VK_PIPELINE_BIND_POINT_COMPUTE,pl,0,1,[ds],0,None); vk.vkCmdDispatch(cb,T//64,M//64,1); vk.vkEndCommandBuffer(cb)
fence=vk.vkCreateFence(dev,vk.VkFenceCreateInfo(),None)
def run(): vk.vkResetFences(dev,1,[fence]); vk.vkQueueSubmit(queue,1,[vk.VkSubmitInfo(pCommandBuffers=[cb])],fence); vk.vkWaitForFences(dev,1,[fence],vk.VK_TRUE,0xFFFFFFFFFFFFFFFF)
run(); C=np.frombuffer(dn(ms,M*T*2),np.float16).astype(np.float32).reshape(M,T)
cos=float((C.ravel()@ref.ravel())/(np.linalg.norm(C)*np.linalg.norm(ref)+1e-9))
print(f"coopmat Q@K^T cos={cos:.5f} {'OK' if cos>0.99 else 'FAIL'}",flush=True)
for _ in range(50): run()
t=time.time()
for _ in range(500): run()
ms_=(time.time()-t)/500*1e3
print(f"S=Q@K^T [{M}x{T}x{hd}]: {ms_*1000:.1f} us  ({2*M*T*hd/(ms_/1e3)/1e9:.0f} GFLOP/s)",flush=True)
