/**
 * @file ComputeBackend.cpp
 * @brief Device manager and backend implementation
 *
 * Supports:
 * - CPU (OpenBLAS/MKL)
 * - NVIDIA CUDA
 * - AMD ROCm
 * - Vulkan (cross-vendor)
 *
 * @author David Sanftenberg
 */

#include "ComputeBackend.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "../utils/CPUFeatures.h"
#include "../kernels/cpu/CPURoPEKernel.h"
#include "../kernels/cpu/CPUSoftmaxKernel.h"
#include "../kernels/cpu/CPURMSNormKernel.h"
#include "../kernels/cpu/CPUSwiGLUKernel.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

// Conditional includes based on backend availability
#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#endif

#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#endif

namespace llaminar2
{

    // ============================================================================
    // Helper: Get backend type name
    // ============================================================================

    static const char *backend_type_name(ComputeBackendType type)
    {
        switch (type)
        {
        case ComputeBackendType::CPU_OPENBLAS:
            return "CPU (OpenBLAS)";
        case ComputeBackendType::CPU_MKL:
            return "CPU (Intel MKL)";
        case ComputeBackendType::GPU_CUDA:
            return "NVIDIA CUDA";
        case ComputeBackendType::GPU_ROCM:
            return "AMD ROCm";
        case ComputeBackendType::GPU_VULKAN:
            return "Vulkan";
        default:
            return "Unknown";
        }
    }

    // ============================================================================
    // CPU Device Enumeration
    // ============================================================================

    static ComputeDevice enumerate_cpu_device()
    {
        ComputeDevice dev;

        // Backend is selected at compile time based on CPU vendor
        // (see CMakeLists.txt BLAS_BACKEND selection)
#ifdef HAVE_MKL
        dev.type = ComputeBackendType::CPU_MKL;
        dev.name = "Intel MKL (CPU)";
#elif defined(HAVE_OPENBLAS)
        dev.type = ComputeBackendType::CPU_OPENBLAS;
        dev.name = "OpenBLAS (CPU)";
#else
#error "No BLAS backend configured - set BLAS_BACKEND in CMake"
#endif

        dev.device_id = 0;
        dev.compute_capability = 0;

        // Get system memory info
#ifdef __linux__
        FILE *meminfo = fopen("/proc/meminfo", "r");
        if (meminfo)
        {
            char line[256];
            while (fgets(line, sizeof(line), meminfo))
            {
                if (strncmp(line, "MemAvailable:", 13) == 0)
                {
                    unsigned long kb = 0;
                    if (sscanf(line + 13, "%lu", &kb) == 1)
                    {
                        dev.total_memory_bytes = static_cast<size_t>(kb) * 1024;
                        dev.free_memory_bytes = dev.total_memory_bytes; // Approximate
                    }
                    break;
                }
            }
            fclose(meminfo);
        }
#else
        // Fallback: assume 16 GB
        dev.total_memory_bytes = 16ULL * 1024 * 1024 * 1024;
        dev.free_memory_bytes = dev.total_memory_bytes;
#endif

        dev.supports_fp16 = false; // Depends on CPU features (AVX512-FP16)
        dev.supports_bf16 = true;  // Software emulation always available
        dev.supports_int8 = true;  // VNNI/DP4A support (detect at runtime)

        return dev;
    }

    // ============================================================================
    // CUDA Device Enumeration
    // ============================================================================

#ifdef HAVE_CUDA
    static std::vector<ComputeDevice> enumerate_cuda_devices()
    {
        std::vector<ComputeDevice> devices;

        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);

        if (err != cudaSuccess || device_count == 0)
        {
            return devices; // No CUDA devices
        }

        for (int i = 0; i < device_count; ++i)
        {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, i) != cudaSuccess)
            {
                continue; // Skip failed device
            }

            ComputeDevice dev;
            dev.type = ComputeBackendType::GPU_CUDA;
            dev.name = std::string(prop.name);
            dev.device_id = i;
            dev.compute_capability = prop.major * 10 + prop.minor; // e.g., 80 for SM 8.0
            dev.total_memory_bytes = prop.totalGlobalMem;

            // Get free memory
            size_t free_bytes = 0, total_bytes = 0;
            if (cudaSetDevice(i) == cudaSuccess)
            {
                cudaMemGetInfo(&free_bytes, &total_bytes);
                dev.free_memory_bytes = free_bytes;
            }
            else
            {
                dev.free_memory_bytes = dev.total_memory_bytes; // Assume free
            }

            // Feature detection based on compute capability
            dev.supports_fp16 = (prop.major >= 6); // Pascal (SM 6.0+)
            dev.supports_bf16 = (prop.major >= 8); // Ampere (SM 8.0+)
            dev.supports_int8 = (prop.major >= 6); // DP4A on Pascal+, Tensor Cores on Volta+

            devices.push_back(dev);
        }

        return devices;
    }
#else
    static std::vector<ComputeDevice> enumerate_cuda_devices()
    {
        return {}; // CUDA not available
    }
#endif

    // ============================================================================
    // ROCm Device Enumeration
    // ============================================================================

#ifdef HAVE_ROCM
    static std::vector<ComputeDevice> enumerate_rocm_devices()
    {
        std::vector<ComputeDevice> devices;

        int device_count = 0;
        hipError_t err = hipGetDeviceCount(&device_count);

        if (err != hipSuccess || device_count == 0)
        {
            return devices; // No ROCm devices
        }

        for (int i = 0; i < device_count; ++i)
        {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, i) != hipSuccess)
            {
                continue; // Skip failed device
            }

            ComputeDevice dev;
            dev.type = ComputeBackendType::GPU_ROCM;
            dev.name = std::string(prop.name);
            dev.device_id = i;
            dev.compute_capability = prop.gcnArch; // GCN architecture version
            dev.total_memory_bytes = prop.totalGlobalMem;

            // Get free memory
            size_t free_bytes = 0, total_bytes = 0;
            if (hipSetDevice(i) == hipSuccess)
            {
                hipMemGetInfo(&free_bytes, &total_bytes);
                dev.free_memory_bytes = free_bytes;
            }
            else
            {
                dev.free_memory_bytes = dev.total_memory_bytes;
            }

            // AMD feature support
            dev.supports_fp16 = true;                  // All modern AMD GPUs support FP16
            dev.supports_bf16 = (prop.gcnArch >= 908); // MI100+ (gfx908+)
            dev.supports_int8 = true;                  // CDNA/RDNA support

            devices.push_back(dev);
        }

        return devices;
    }
#else
    static std::vector<ComputeDevice> enumerate_rocm_devices()
    {
        return {}; // ROCm not available
    }
#endif

    // ============================================================================
    // Vulkan Device Enumeration
    // ============================================================================

#ifdef HAVE_VULKAN
    static std::vector<ComputeDevice> enumerate_vulkan_devices()
    {
        std::vector<ComputeDevice> devices;

        // Create Vulkan instance
        VkApplicationInfo app_info = {};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "Llaminar";
        app_info.applicationVersion = VK_MAKE_VERSION(2, 0, 0);
        app_info.pEngineName = "Llaminar";
        app_info.engineVersion = VK_MAKE_VERSION(2, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_2;

        VkInstanceCreateInfo create_info = {};
        create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        create_info.pApplicationInfo = &app_info;

        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS)
        {
            return devices; // Vulkan not available
        }

        // Enumerate physical devices
        uint32_t device_count = 0;
        vkEnumeratePhysicalDevices(instance, &device_count, nullptr);

        if (device_count == 0)
        {
            vkDestroyInstance(instance, nullptr);
            return devices;
        }

        std::vector<VkPhysicalDevice> physical_devices(device_count);
        vkEnumeratePhysicalDevices(instance, &device_count, physical_devices.data());

        for (uint32_t i = 0; i < device_count; ++i)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(physical_devices[i], &props);

            VkPhysicalDeviceMemoryProperties mem_props;
            vkGetPhysicalDeviceMemoryProperties(physical_devices[i], &mem_props);

            ComputeDevice dev;
            dev.type = ComputeBackendType::GPU_VULKAN;
            dev.name = std::string(props.deviceName);
            dev.device_id = i;
            dev.compute_capability = VK_VERSION_MAJOR(props.apiVersion) * 10 +
                                     VK_VERSION_MINOR(props.apiVersion);

            // Sum up device-local memory heaps
            dev.total_memory_bytes = 0;
            for (uint32_t j = 0; j < mem_props.memoryHeapCount; ++j)
            {
                if (mem_props.memoryHeaps[j].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                {
                    dev.total_memory_bytes += mem_props.memoryHeaps[j].size;
                }
            }
            dev.free_memory_bytes = dev.total_memory_bytes; // Approximate

            // Vulkan feature support (query via extensions)
            dev.supports_fp16 = true;  // VK_KHR_shader_float16_int8
            dev.supports_bf16 = false; // Limited BF16 support in Vulkan
            dev.supports_int8 = true;  // VK_KHR_shader_float16_int8

            devices.push_back(dev);
        }

        vkDestroyInstance(instance, nullptr);
        return devices;
    }
#else
    static std::vector<ComputeDevice> enumerate_vulkan_devices()
    {
        return {}; // Vulkan not available
    }
#endif

    // ============================================================================
    // DeviceManager Implementation
    // ============================================================================

    void DeviceManager::initialize()
    {
        devices_.clear();
        contexts_.clear();

        // Always enumerate CPU first (device index 0)
        devices_.push_back(enumerate_cpu_device());

        // Enumerate GPUs
        auto cuda_devices = enumerate_cuda_devices();
        auto rocm_devices = enumerate_rocm_devices();
        auto vulkan_devices = enumerate_vulkan_devices();

        devices_.insert(devices_.end(), cuda_devices.begin(), cuda_devices.end());
        devices_.insert(devices_.end(), rocm_devices.begin(), rocm_devices.end());
        devices_.insert(devices_.end(), vulkan_devices.begin(), vulkan_devices.end());

        // Resize contexts_ vector to match devices
        contexts_.resize(devices_.size(), nullptr);

        // Log discovered devices
        LOG_INFO("[DeviceManager] Enumerated " << devices_.size() << " device(s):");
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            const auto &dev = devices_[i];
            std::string device_info = "  [" + std::to_string(i) + "] " + backend_type_name(dev.type) + " - " + dev.name + " (" + std::to_string(dev.total_memory_bytes / (1024 * 1024 * 1024)) + " GB";
            if (dev.type != ComputeBackendType::CPU_OPENBLAS &&
                dev.type != ComputeBackendType::CPU_MKL)
            {
                device_info += ", SM " + std::to_string(dev.compute_capability / 10) + "." + std::to_string(dev.compute_capability % 10);
            }
            device_info += ")";
            LOG_INFO(device_info);
        }
    }

    std::shared_ptr<ComputeContext> DeviceManager::create_context(size_t device_index)
    {
        if (device_index >= devices_.size())
        {
            LOG_ERROR("[DeviceManager] Invalid device index: " << device_index << "");
            return nullptr;
        }

        // Check if context already exists
        if (device_index < contexts_.size() && contexts_[device_index])
        {
            return contexts_[device_index]; // Reuse existing context
        }

        // Ensure contexts_ vector is large enough
        if (device_index >= contexts_.size())
        {
            contexts_.resize(device_index + 1, nullptr);
        }

        // Create concrete context based on backend type
        std::shared_ptr<ComputeContext> ctx;
        const auto &device = devices_[device_index];

        switch (device.type)
        {
        case ComputeBackendType::CPU_OPENBLAS:
        case ComputeBackendType::CPU_MKL:
            ctx = std::make_shared<CPUComputeContext>();
            break;

#ifdef HAVE_CUDA
        case ComputeBackendType::GPU_CUDA:
        {
            auto cuda_ctx = std::make_shared<CUDAComputeContext>();
            if (cudaSetDevice(device.device_id) != cudaSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to set CUDA device "
                          << device.device_id << "");
                return nullptr;
            }

            cudaStream_t stream;
            if (cudaStreamCreate(&stream) != cudaSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to create CUDA stream");
                return nullptr;
            }
            cuda_ctx->stream = stream;
            cuda_ctx->device_id = device.device_id;

            cublasHandle_t cublas_handle;
            if (cublasCreate(&cublas_handle) != CUBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[DeviceManager] Failed to create cuBLAS handle");
                cudaStreamDestroy(stream);
                return nullptr;
            }
            cublasSetStream(cublas_handle, stream);
            cuda_ctx->cublas_handle = cublas_handle;
            ctx = cuda_ctx;
            break;
        }
#endif

#ifdef HAVE_ROCM
        case ComputeBackendType::GPU_ROCM:
        {
            auto rocm_ctx = std::make_shared<ROCmComputeContext>();
            if (hipSetDevice(device.device_id) != hipSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to set ROCm device "
                          << device.device_id << "");
                return nullptr;
            }

            hipStream_t stream;
            if (hipStreamCreate(&stream) != hipSuccess)
            {
                LOG_ERROR("[DeviceManager] Failed to create HIP stream");
                return nullptr;
            }
            rocm_ctx->stream = stream;
            rocm_ctx->device_id = device.device_id;

            hipblasHandle_t hipblas_handle;
            if (hipblasCreate(&hipblas_handle) != HIPBLAS_STATUS_SUCCESS)
            {
                LOG_ERROR("[DeviceManager] Failed to create hipBLAS handle");
                hipStreamDestroy(stream);
                return nullptr;
            }
            hipblasSetStream(hipblas_handle, stream);
            rocm_ctx->hipblas_handle = hipblas_handle;
            ctx = rocm_ctx;
            break;
        }
#endif

#ifdef HAVE_VULKAN
        case ComputeBackendType::GPU_VULKAN:
            // TODO: Vulkan context initialization
            ctx = std::make_shared<VulkanComputeContext>();
            LOG_ERROR("[DeviceManager] Vulkan context creation not fully implemented");
            break;
#else
        case ComputeBackendType::GPU_VULKAN:
            LOG_ERROR("[DeviceManager] Vulkan not available in this build");
            return nullptr;
#endif

        default:
            LOG_ERROR("[DeviceManager] Unknown backend type");
            return nullptr;
        }

        // Cache context
        contexts_[device_index] = ctx;

        LOG_INFO("[DeviceManager] Created context for device " << device_index
                                                               << " (" << backend_type_name(device.type) << ")");

        return ctx;
    }

    int DeviceManager::find_device(ComputeBackendType type, int device_id) const
    {
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            if (devices_[i].type == type && devices_[i].device_id == device_id)
            {
                return static_cast<int>(i);
            }
        }
        return -1; // Not found
    }

    size_t DeviceManager::select_device(size_t estimated_memory_bytes)
    {
        if (devices_.empty())
        {
            LOG_ERROR("[DeviceManager] No devices available");
            return 0;
        }

        // Find GPUs with sufficient memory
        struct DeviceScore
        {
            size_t index;
            size_t free_memory;
            int priority; // Higher = better
        };

        std::vector<DeviceScore> candidates;

        for (size_t i = 0; i < devices_.size(); ++i)
        {
            const auto &dev = devices_[i];

            // Skip if insufficient memory
            if (estimated_memory_bytes > 0 && dev.free_memory_bytes < estimated_memory_bytes)
            {
                continue;
            }

            // Priority: CUDA > ROCm > Vulkan > CPU
            int priority = 0;
            switch (dev.type)
            {
            case ComputeBackendType::GPU_CUDA:
                priority = 300;
                break;
            case ComputeBackendType::GPU_ROCM:
                priority = 200;
                break;
            case ComputeBackendType::GPU_VULKAN:
                priority = 100;
                break;
            case ComputeBackendType::CPU_MKL:
                priority = 50;
                break;
            case ComputeBackendType::CPU_OPENBLAS:
                priority = 10;
                break;
            }

            candidates.push_back({i, dev.free_memory_bytes, priority});
        }

        if (candidates.empty())
        {
            // Fall back to CPU
            LOG_INFO("[DeviceManager] No suitable GPU found, using CPU");
            return 0;
        }

        // Sort by priority (descending), then by free memory (descending)
        std::sort(candidates.begin(), candidates.end(),
                  [](const DeviceScore &a, const DeviceScore &b)
                  {
                      if (a.priority != b.priority)
                          return a.priority > b.priority;
                      return a.free_memory > b.free_memory;
                  });

        size_t selected = candidates[0].index;
        LOG_INFO("[DeviceManager] Auto-selected device " << selected
                                                         << " (" << backend_type_name(devices_[selected].type)
                                                         << "): " << devices_[selected].name << "");

        return selected;
    }

    bool DeviceManager::has_gpu() const
    {
        for (const auto &dev : devices_)
        {
            if (dev.type == ComputeBackendType::GPU_CUDA ||
                dev.type == ComputeBackendType::GPU_ROCM ||
                dev.type == ComputeBackendType::GPU_VULKAN)
            {
                return true;
            }
        }
        return false;
    }

    // ============================================================================
    // CPUComputeContext Implementation
    // ============================================================================

    struct CPUComputeContext::Impl
    {
        std::unique_ptr<CPURoPEKernel> rope_kernel;
        std::unique_ptr<CPUSoftmaxKernel> softmax_kernel;
        std::unique_ptr<CPURMSNormKernel> rmsnorm_kernel;
        std::unique_ptr<CPUSwiGLUKernel> swiglu_kernel;
    };

    CPUComputeContext::CPUComputeContext()
        : pimpl_(std::make_unique<Impl>())
    {
    }

    CPUComputeContext::~CPUComputeContext() = default;

    void *CPUComputeContext::allocate(size_t bytes)
    {
        return std::malloc(bytes);
    }

    void CPUComputeContext::free(void *ptr)
    {
        std::free(ptr);
    }

    void CPUComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        std::memcpy(dst, src, bytes); // CPU-to-CPU copy
    }

    void CPUComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        std::memcpy(dst, src, bytes); // CPU-to-CPU copy
    }

    ITensorRoPE *CPUComputeContext::get_rope_kernel()
    {
        if (!pimpl_->rope_kernel)
        {
            pimpl_->rope_kernel = std::make_unique<CPURoPEKernel>();
        }
        return pimpl_->rope_kernel.get();
    }

    ITensorSoftmax *CPUComputeContext::get_softmax_kernel()
    {
        if (!pimpl_->softmax_kernel)
        {
            pimpl_->softmax_kernel = std::make_unique<CPUSoftmaxKernel>();
        }
        return pimpl_->softmax_kernel.get();
    }

    ITensorRMSNorm *CPUComputeContext::get_rmsnorm_kernel()
    {
        if (!pimpl_->rmsnorm_kernel)
        {
            pimpl_->rmsnorm_kernel = std::make_unique<CPURMSNormKernel>();
        }
        return pimpl_->rmsnorm_kernel.get();
    }

    ITensorSwiGLU *CPUComputeContext::get_swiglu_kernel()
    {
        if (!pimpl_->swiglu_kernel)
        {
            pimpl_->swiglu_kernel = std::make_unique<CPUSwiGLUKernel>();
        }
        return pimpl_->swiglu_kernel.get();
    }

    // ============================================================================
    // CUDAComputeContext Implementation
    // ============================================================================

#ifdef HAVE_CUDA
    void *CUDAComputeContext::allocate(size_t bytes)
    {
        void *ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDA] Failed to allocate " << bytes << " bytes: "
                                                   << cudaGetErrorString(err) << "");
            return nullptr;
        }
        return ptr;
    }

    void CUDAComputeContext::free(void *ptr)
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    void CUDAComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDA] copy_to_device failed: " << cudaGetErrorString(err) << "");
        }
    }

    void CUDAComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDA] copy_from_device failed: " << cudaGetErrorString(err) << "");
        }
    }

    void CUDAComputeContext::synchronize()
    {
        if (stream)
        {
            cudaStreamSynchronize(stream);
        }
        else
        {
            cudaDeviceSynchronize();
        }
    }
#endif

    // ============================================================================
    // ROCmComputeContext Implementation
    // ============================================================================

#ifdef HAVE_ROCM
    void *ROCmComputeContext::allocate(size_t bytes)
    {
        void *ptr = nullptr;
        hipError_t err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCm] Failed to allocate " << bytes << " bytes");
            return nullptr;
        }
        return ptr;
    }

    void ROCmComputeContext::free(void *ptr)
    {
        if (ptr)
        {
            hipFree(ptr);
        }
    }

    void ROCmComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCm] copy_to_device failed");
        }
    }

    void ROCmComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCm] copy_from_device failed");
        }
    }

    void ROCmComputeContext::synchronize()
    {
        if (stream)
        {
            hipStreamSynchronize(stream);
        }
        else
        {
            hipDeviceSynchronize();
        }
    }
#endif

    // ============================================================================
    // VulkanComputeContext Implementation (Stub)
    // ============================================================================

#ifdef HAVE_VULKAN
    void *VulkanComputeContext::allocate(size_t bytes)
    {
        // TODO: Vulkan buffer allocation
        LOG_ERROR("[Vulkan] allocate() not yet implemented");
        return nullptr;
    }

    void VulkanComputeContext::free(void *ptr)
    {
        // TODO: Vulkan buffer deallocation
    }

    void VulkanComputeContext::copy_to_device(void *dst, const void *src, size_t bytes)
    {
        // TODO: Vulkan staging buffer upload
        LOG_ERROR("[Vulkan] copy_to_device() not yet implemented");
    }

    void VulkanComputeContext::copy_from_device(void *dst, const void *src, size_t bytes)
    {
        // TODO: Vulkan staging buffer download
        LOG_ERROR("[Vulkan] copy_from_device() not yet implemented");
    }

    void VulkanComputeContext::synchronize()
    {
        // TODO: Vulkan queue submit + wait
    }
#endif

} // namespace llaminar2
