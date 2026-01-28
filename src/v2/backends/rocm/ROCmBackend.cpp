/**
 * @file ROCmBackend.cpp
 * @brief ROCm/HIP backend implementation with hip_runtime.h
 *
 * **Purpose**: Implements IBackend for AMD GPUs. This .cpp file is the ONLY
 * compilation unit that includes hip_runtime.h, preventing header conflicts.
 *
 * @author David Sanftenberg
 */

#include "ROCmBackend.h"
#include "../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <dlfcn.h> // For HSA runtime loading

namespace llaminar2
{

    // ====================================================================
    // Constructor / Destructor
    // ====================================================================

    ROCmBackend::ROCmBackend()
        : device_count_(0)
    {
        hipError_t err = hipGetDeviceCount(&device_count_);
        if (err != hipSuccess)
        {
            device_count_ = 0;
            // Log warning but don't throw - allow CPU-only execution
        }
    }

    ROCmBackend::~ROCmBackend()
    {
        // hipDeviceReset() intentionally omitted - managed by HIP runtime
    }

    // ====================================================================
    // Memory Transfer Operations
    // ====================================================================

    bool ROCmBackend::deviceToHost(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
        return (err == hipSuccess);
    }

    bool ROCmBackend::hostToDevice(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice);
        return (err == hipSuccess);
    }

    bool ROCmBackend::synchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipDeviceSynchronize();
        return (err == hipSuccess);
    }

    bool ROCmBackend::streamSynchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        // Synchronize only the default stream (nullptr), not all streams
        hipError_t err = hipStreamSynchronize(nullptr);
        return (err == hipSuccess);
    }

    // ====================================================================
    // Event Operations (Fine-grained Synchronization)
    // ====================================================================

    void *ROCmBackend::createEvent(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return nullptr;
        }

        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            return nullptr;
        }

        hipEvent_t event;
        err = hipEventCreate(&event);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::createEvent] hipEventCreate failed: " << hipGetErrorString(err));
            return nullptr;
        }

        return reinterpret_cast<void *>(event);
    }

    void ROCmBackend::destroyEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return;
        }

        hipSetDevice(device_id);
        hipEvent_t hip_event = reinterpret_cast<hipEvent_t>(event);
        hipEventDestroy(hip_event);
    }

    bool ROCmBackend::recordEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        hipEvent_t hip_event = reinterpret_cast<hipEvent_t>(event);
        // Record on default stream (0)
        err = hipEventRecord(hip_event, 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::recordEvent] hipEventRecord failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmBackend::waitForEvent(void *event, int device_id)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        double set_device_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Use event-based sync for fine-grained synchronization
        // This waits only for the specific kernel that recorded this event,
        // NOT for all work on the stream (which could include unrelated prior work)
        hipEvent_t hip_event = reinterpret_cast<hipEvent_t>(event);
        err = hipEventSynchronize(hip_event);

        auto t2 = std::chrono::high_resolution_clock::now();
        double event_sync_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();

        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::waitForEvent] hipEventSynchronize failed: " << hipGetErrorString(err));
            return false;
        }

        double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();
        if (total_ms > 1.0)
        {
            LOG_TRACE("[ROCmBackend::waitForEvent] setDevice=" << set_device_ms << "ms, eventSync=" << event_sync_ms << "ms, TOTAL=" << total_ms << "ms");
        }

        return true;
    }

    bool ROCmBackend::setDevice(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err = hipSetDevice(device_id);
        return (err == hipSuccess);
    }

    // ====================================================================
    // Memory Allocation Operations
    // ====================================================================

    void *ROCmBackend::allocate(size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend] Invalid device ID " << device_id << " (max: " << device_count_ - 1 << ")");
            return nullptr;
        }

        // Set device before allocation
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << ": " << hipGetErrorString(err));
            return nullptr;
        }

        void *ptr = nullptr;
        err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipMalloc failed for " << bytes << " bytes on device "
                                                            << device_id << ": " << hipGetErrorString(err));
            return nullptr;
        }

        return ptr;
    }

    void *ROCmBackend::allocateMapped(size_t bytes, int device_id, void **device_ptr)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend] Invalid device ID " << device_id << " for allocateMapped");
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Set device before allocation
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << ": " << hipGetErrorString(err));
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Allocate mapped host memory (GPU can write directly to this via PCIe)
        void *host_ptr = nullptr;
        err = hipHostMalloc(&host_ptr, bytes, hipHostMallocMapped | hipHostMallocWriteCombined);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipHostMalloc(Mapped) failed for " << bytes << " bytes on device "
                                                                        << device_id << ": " << hipGetErrorString(err));
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Get the device-visible pointer for this mapped host memory
        if (device_ptr)
        {
            err = hipHostGetDevicePointer(device_ptr, host_ptr, 0);
            if (err != hipSuccess)
            {
                LOG_ERROR("[ROCmBackend] hipHostGetDevicePointer failed: " << hipGetErrorString(err));
                hipHostFree(host_ptr);
                *device_ptr = nullptr;
                return nullptr;
            }
            LOG_TRACE("[ROCmBackend] allocateMapped: " << bytes << " bytes, host_ptr=" << host_ptr
                                                       << ", device_ptr=" << *device_ptr);
        }

        return host_ptr;
    }

    void ROCmBackend::freeMapped(void *host_ptr, int device_id)
    {
        if (host_ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        // hipHostFree doesn't require setting device, but we do it for consistency
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_WARN("[ROCmBackend] Invalid device ID " << device_id << " for freeMapped, attempting anyway");
        }
        else
        {
            hipSetDevice(device_id); // Best effort
        }

        hipError_t err = hipHostFree(host_ptr);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipHostFree failed: " << hipGetErrorString(err));
        }
    }

    void ROCmBackend::free(void *ptr, int device_id)
    {
        if (ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend] Invalid device ID " << device_id << " for hipFree");
            return;
        }

        // Set device before freeing
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            // During shutdown, hipSetDevice may fail - this is expected
            if (err == hipErrorDeinitialized || err == hipErrorContextIsDestroyed)
            {
                LOG_DEBUG("[ROCmBackend] hipSetDevice failed during shutdown (expected): "
                          << hipGetErrorString(err));
            }
            else
            {
                LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << " before hipFree: "
                                                                << hipGetErrorString(err));
            }
            return;
        }

        err = hipFree(ptr);
        if (err != hipSuccess)
        {
            // During shutdown, hipFree may fail with "invalid argument" if the memory
            // was already cleaned up by the HIP runtime or the pointer is stale.
            // Also handle explicit deinitialization errors.
            if (err == hipErrorDeinitialized || err == hipErrorContextIsDestroyed ||
                err == hipErrorInvalidValue)
            {
                LOG_TRACE("[ROCmBackend] hipFree skipped (driver shutting down or memory already freed)");
            }
            else
            {
                LOG_ERROR("[ROCmBackend] hipFree failed: " << hipGetErrorString(err));
            }
        }
    }

    bool ROCmBackend::memset(void *ptr, int value, size_t bytes, int device_id)
    {
        if (ptr == nullptr || bytes == 0)
        {
            return true; // No-op for null pointer or zero bytes
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend] Invalid device ID " << device_id << " for hipMemset");
            return false;
        }

        // Set device before memset
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << " before hipMemset: "
                                                            << hipGetErrorString(err));
            return false;
        }

        err = hipMemset(ptr, value, bytes);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipMemset failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    // ====================================================================
    // Device Query Operations
    // ====================================================================

    int ROCmBackend::deviceCount() const
    {
        return device_count_;
    }

    std::string ROCmBackend::backendName() const
    {
        return "ROCm";
    }

    std::string ROCmBackend::deviceName(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return "Invalid Device";
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return "Unknown Device";
        }

        return std::string(prop.name);
    }

    size_t ROCmBackend::deviceMemoryTotal(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return 0;
        }

        return prop.totalGlobalMem;
    }

    size_t ROCmBackend::deviceMemoryFree(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return 0;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        hipError_t err = hipMemGetInfo(&free_bytes, &total_bytes);
        if (err != hipSuccess)
        {
            return 0;
        }

        return free_bytes;
    }

    // ====================================================================
    // Capability Queries
    // ====================================================================

    bool ROCmBackend::supportsBF16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        // BF16 support on AMD GPUs:
        // - MI200 series (gfx90a): Full BF16 support
        // - MI100 (gfx908): Limited BF16 support
        // GCN architecture ID (gcnArch) is deprecated in newer ROCm versions
        // Use compute capability via prop.major/minor or architecture string

        // Conservative check: Assume MI200+ (gfx90a and later) for full BF16
        // This is a heuristic - may need refinement based on actual hardware
        std::string arch_name(prop.gcnArchName);
        return (arch_name.find("gfx90a") != std::string::npos ||
                arch_name.find("gfx940") != std::string::npos ||
                arch_name.find("gfx941") != std::string::npos ||
                arch_name.find("gfx942") != std::string::npos);
    }

    bool ROCmBackend::supportsFP16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        // FP16 support widely available on AMD GPUs (Vega and later)
        // gfx900 (Vega 10) and later all support FP16
        std::string arch_name(prop.gcnArchName);
        return (arch_name.find("gfx9") != std::string::npos ||
                arch_name.find("gfx10") != std::string::npos ||
                arch_name.find("gfx11") != std::string::npos);
    }

    bool ROCmBackend::supportsINT8(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        // INT8 support widely available on modern AMD GPUs
        // Conservatively assume gfx9 and later (Vega+)
        std::string arch_name(prop.gcnArchName);
        return (arch_name.find("gfx9") != std::string::npos ||
                arch_name.find("gfx10") != std::string::npos ||
                arch_name.find("gfx11") != std::string::npos);
    }

    // ====================================================================
    // Compute Operations
    // ====================================================================

    bool ROCmBackend::gemmIQ4NL(
        const void *A_device,
        const void *B_device,
        void *C_device,
        int m,
        int n,
        int k,
        int device_id)
    {
        // TODO: Implement ROCm/HIP version of IQ4_NL GEMM kernel
        // For now, return false to indicate not implemented
        (void)A_device;
        (void)B_device;
        (void)C_device;
        (void)m;
        (void)n;
        (void)k;
        (void)device_id;

        return false;
    }

    // ====================================================================
    // Extended Operations
    // ====================================================================

    bool ROCmBackend::queryPointerAttributes(const void *ptr, bool &is_device_ptr, bool &is_host_ptr,
                                             bool &is_managed, int &device_id) const
    {
        hipPointerAttribute_t attr;
        hipError_t err = hipPointerGetAttributes(&attr, ptr);

        if (err != hipSuccess)
        {
            // Reset outputs
            is_device_ptr = false;
            is_host_ptr = false;
            is_managed = false;
            device_id = -1;
            return false;
        }

        // Interpret the memory type
        // hipMemoryType: hipMemoryTypeHost, hipMemoryTypeDevice, hipMemoryTypeUnified, hipMemoryTypeManaged
        is_host_ptr = (attr.type == hipMemoryTypeHost);
        is_device_ptr = (attr.type == hipMemoryTypeDevice);
        is_managed = (attr.type == hipMemoryTypeManaged || attr.type == hipMemoryTypeUnified);
        device_id = attr.device;

        return true;
    }

    bool ROCmBackend::deviceToDevice(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice);
        return (err == hipSuccess);
    }

    bool ROCmBackend::registerIoMemory(void *ptr, size_t size, void **device_ptr)
    {
        if (!ptr || size == 0 || !device_ptr)
        {
            return false;
        }

        *device_ptr = nullptr;

        // Try hipHostRegister with different flag combinations
        // hipHostRegisterIoMemory = 0x4 (maps IO memory to device address space)
        // hipHostRegisterMapped = 0x2 (maps host memory to device address space)
        // hipHostRegisterPortable = 0x1 (memory can be accessed from any context)

        hipError_t err;

        // Attempt 1: IoMemory flag (most promising for BAR memory)
        LOG_DEBUG("[ROCmBackend::registerIoMemory] Trying hipHostRegisterIoMemory flag");
        err = hipHostRegister(ptr, size, hipHostRegisterIoMemory);

        if (err == hipSuccess)
        {
            LOG_INFO("[ROCmBackend::registerIoMemory] hipHostRegisterIoMemory succeeded!");

            err = hipHostGetDevicePointer(device_ptr, ptr, 0);
            if (err == hipSuccess && *device_ptr != nullptr)
            {
                LOG_INFO("[ROCmBackend::registerIoMemory] Got device pointer: " << *device_ptr);
                return true;
            }
            else
            {
                LOG_WARN("[ROCmBackend::registerIoMemory] hipHostGetDevicePointer failed: "
                         << hipGetErrorString(err));
                hipHostUnregister(ptr);
            }
        }
        else
        {
            LOG_DEBUG("[ROCmBackend::registerIoMemory] hipHostRegisterIoMemory failed: "
                      << hipGetErrorString(err));
        }

        // Attempt 2: Mapped + Portable flags
        LOG_DEBUG("[ROCmBackend::registerIoMemory] Trying hipHostRegisterMapped | hipHostRegisterPortable");
        err = hipHostRegister(ptr, size, hipHostRegisterMapped | hipHostRegisterPortable);

        if (err == hipSuccess)
        {
            LOG_INFO("[ROCmBackend::registerIoMemory] hipHostRegisterMapped succeeded!");

            err = hipHostGetDevicePointer(device_ptr, ptr, 0);
            if (err == hipSuccess && *device_ptr != nullptr)
            {
                LOG_INFO("[ROCmBackend::registerIoMemory] Got device pointer: " << *device_ptr);
                return true;
            }
            else
            {
                LOG_WARN("[ROCmBackend::registerIoMemory] hipHostGetDevicePointer failed: "
                         << hipGetErrorString(err));
                hipHostUnregister(ptr);
            }
        }
        else
        {
            LOG_DEBUG("[ROCmBackend::registerIoMemory] hipHostRegisterMapped failed: "
                      << hipGetErrorString(err));
        }

        // Attempt 3: Default flags
        LOG_DEBUG("[ROCmBackend::registerIoMemory] Trying hipHostRegisterDefault");
        err = hipHostRegister(ptr, size, hipHostRegisterDefault);

        if (err == hipSuccess)
        {
            LOG_INFO("[ROCmBackend::registerIoMemory] hipHostRegisterDefault succeeded!");

            err = hipHostGetDevicePointer(device_ptr, ptr, 0);
            if (err == hipSuccess && *device_ptr != nullptr)
            {
                LOG_INFO("[ROCmBackend::registerIoMemory] Got device pointer: " << *device_ptr);
                return true;
            }
            else
            {
                hipHostUnregister(ptr);
            }
        }

        LOG_WARN("[ROCmBackend::registerIoMemory] All registration attempts failed for ptr=" << ptr);
        return false;
    }

    void ROCmBackend::unregisterIoMemory(void *ptr)
    {
        if (ptr)
        {
            hipError_t err = hipHostUnregister(ptr);
            if (err != hipSuccess)
            {
                LOG_WARN("[ROCmBackend::unregisterIoMemory] hipHostUnregister failed: "
                         << hipGetErrorString(err));
            }
        }
    }

    bool ROCmBackend::getPointerInfo(const void *ptr, void **device_ptr, void **host_ptr,
                                     std::string &mem_type) const
    {
        if (!ptr)
        {
            return false;
        }

        hipPointerAttribute_t attr;
        std::memset(&attr, 0, sizeof(attr));

        hipError_t err = hipPointerGetAttributes(&attr, ptr);

        if (err != hipSuccess)
        {
            mem_type = "unknown (query failed: " + std::string(hipGetErrorString(err)) + ")";
            if (device_ptr)
                *device_ptr = nullptr;
            if (host_ptr)
                *host_ptr = nullptr;
            return false;
        }

        if (device_ptr)
            *device_ptr = attr.devicePointer;
        if (host_ptr)
            *host_ptr = attr.hostPointer;

        // Decode memory type
        switch (attr.type)
        {
        case hipMemoryTypeHost:
            mem_type = "host";
            break;
        case hipMemoryTypeDevice:
            mem_type = "device";
            break;
        case hipMemoryTypeManaged:
            mem_type = "managed";
            break;
        case hipMemoryTypeUnified:
            mem_type = "unified";
            break;
        default:
            mem_type = "unknown(" + std::to_string(static_cast<int>(attr.type)) + ")";
            break;
        }

        return true;
    }

    // ====================================================================
    // HSA-Level Memory Operations
    // ====================================================================

    bool ROCmBackend::hsaMemoryLock(void *host_ptr, size_t size, void **agent_ptr)
    {
        if (!host_ptr || !agent_ptr || size == 0)
        {
            return false;
        }

        LOG_INFO("[ROCmBackend::hsaMemoryLock] Attempting to lock " << size
                                                                    << " bytes at " << std::hex << host_ptr << std::dec);

        // Use hipExtMallocWithFlags or try hipHostRegister with HSA underneath
        // First, let's try a simple approach: use HIP's internal HSA handle

        // Get the HSA agent for the current GPU device
        // HIP wraps HSA, so we can access HSA functions through the hip runtime

        // Try hipHostRegister with hipHostRegisterDefault first, then query device pointer
        // The key insight: hipMemcpy(D2D) works, so HIP internally knows how to access BAR
        // Maybe we can get that internal knowledge exposed via hipPointerGetAttributes

        // Alternative approach: Use hipExtMallocWithFlags to create a "view" of existing memory
        // But this doesn't exist either...

        // Let's try to use HSA directly via dlsym
        void *hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hsa_handle)
        {
            LOG_ERROR("[ROCmBackend::hsaMemoryLock] Failed to load HSA runtime: " << dlerror());
            return false;
        }

        // Type for hsa_amd_memory_lock
        typedef int (*hsa_amd_memory_lock_fn)(void *host_ptr, size_t size,
                                              void *agents, int num_agent,
                                              void **agent_ptr);

        auto memory_lock = (hsa_amd_memory_lock_fn)dlsym(hsa_handle, "hsa_amd_memory_lock");
        if (!memory_lock)
        {
            LOG_ERROR("[ROCmBackend::hsaMemoryLock] hsa_amd_memory_lock not found: " << dlerror());
            dlclose(hsa_handle);
            return false;
        }

        LOG_INFO("[ROCmBackend::hsaMemoryLock] Found hsa_amd_memory_lock, calling...");

        // Call hsa_amd_memory_lock with NULL agents (all agents)
        // This pins the memory and returns a device-accessible pointer
        int status = memory_lock(host_ptr, size, nullptr, 0, agent_ptr);

        dlclose(hsa_handle);

        if (status == 0) // HSA_STATUS_SUCCESS = 0
        {
            LOG_INFO("[ROCmBackend::hsaMemoryLock] SUCCESS! agent_ptr = "
                     << std::hex << *agent_ptr << std::dec);
            return true;
        }
        else
        {
            LOG_ERROR("[ROCmBackend::hsaMemoryLock] hsa_amd_memory_lock failed with status " << status);
            *agent_ptr = nullptr;
            return false;
        }
    }

    void ROCmBackend::hsaMemoryUnlock(void *host_ptr)
    {
        if (!host_ptr)
        {
            return;
        }

        void *hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hsa_handle)
        {
            LOG_WARN("[ROCmBackend::hsaMemoryUnlock] Failed to load HSA runtime");
            return;
        }

        typedef int (*hsa_amd_memory_unlock_fn)(void *host_ptr);
        auto memory_unlock = (hsa_amd_memory_unlock_fn)dlsym(hsa_handle, "hsa_amd_memory_unlock");

        if (memory_unlock)
        {
            int status = memory_unlock(host_ptr);
            if (status != 0)
            {
                LOG_WARN("[ROCmBackend::hsaMemoryUnlock] hsa_amd_memory_unlock failed: " << status);
            }
        }

        dlclose(hsa_handle);
    }

    // ====================================================================
    // HSA Interop and External Memory Operations
    // ====================================================================

    bool ROCmBackend::hsaInteropMapBuffer(int dmabuf_fd, size_t *size, void **device_ptr)
    {
        if (dmabuf_fd < 0 || !device_ptr)
        {
            return false;
        }

        LOG_INFO("[ROCmBackend::hsaInteropMapBuffer] Attempting to map dmabuf fd=" << dmabuf_fd);

        void *hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hsa_handle)
        {
            LOG_ERROR("[ROCmBackend::hsaInteropMapBuffer] Failed to load HSA runtime: " << dlerror());
            return false;
        }

        // hsa_amd_interop_map_buffer(num_agents, agents, interop_handle, flags, size, ptr, metadata_size, metadata)
        typedef int (*hsa_amd_interop_map_buffer_fn)(
            uint32_t num_agents,
            void *agents, // hsa_agent_t*
            int interop_handle,
            uint32_t flags,
            size_t *size,
            void **ptr,
            size_t *metadata_size,
            const void **metadata);

        auto interop_map = (hsa_amd_interop_map_buffer_fn)dlsym(hsa_handle, "hsa_amd_interop_map_buffer");
        if (!interop_map)
        {
            LOG_ERROR("[ROCmBackend::hsaInteropMapBuffer] hsa_amd_interop_map_buffer not found: " << dlerror());
            dlclose(hsa_handle);
            return false;
        }

        LOG_INFO("[ROCmBackend::hsaInteropMapBuffer] Found hsa_amd_interop_map_buffer, calling...");

        // Call with NULL agents to allow access from all agents
        size_t mapped_size = 0;
        void *mapped_ptr = nullptr;

        int status = interop_map(
            0,         // num_agents (0 = all agents)
            nullptr,   // agents
            dmabuf_fd, // interop_handle (dmabuf fd)
            0,         // flags (reserved, must be 0)
            &mapped_size,
            &mapped_ptr,
            nullptr,  // metadata_size (optional)
            nullptr); // metadata (optional)

        dlclose(hsa_handle);

        if (status == 0) // HSA_STATUS_SUCCESS
        {
            LOG_INFO("[ROCmBackend::hsaInteropMapBuffer] SUCCESS! mapped_ptr="
                     << std::hex << mapped_ptr << std::dec << ", size=" << mapped_size);
            if (size)
                *size = mapped_size;
            *device_ptr = mapped_ptr;
            return true;
        }
        else
        {
            LOG_ERROR("[ROCmBackend::hsaInteropMapBuffer] hsa_amd_interop_map_buffer failed with status " << status);
            // Decode common HSA errors
            if (status == 0x1008)
            {
                LOG_ERROR("  -> HSA_STATUS_ERROR_OUT_OF_RESOURCES");
            }
            else if (status == 0x1001)
            {
                LOG_ERROR("  -> HSA_STATUS_ERROR_INVALID_ARGUMENT");
            }
            *device_ptr = nullptr;
            return false;
        }
    }

    void ROCmBackend::hsaInteropUnmapBuffer(void *device_ptr)
    {
        if (!device_ptr)
        {
            return;
        }

        void *hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hsa_handle)
        {
            LOG_WARN("[ROCmBackend::hsaInteropUnmapBuffer] Failed to load HSA runtime");
            return;
        }

        typedef int (*hsa_amd_interop_unmap_buffer_fn)(void *ptr);
        auto interop_unmap = (hsa_amd_interop_unmap_buffer_fn)dlsym(hsa_handle, "hsa_amd_interop_unmap_buffer");

        if (interop_unmap)
        {
            int status = interop_unmap(device_ptr);
            if (status != 0)
            {
                LOG_WARN("[ROCmBackend::hsaInteropUnmapBuffer] hsa_amd_interop_unmap_buffer failed: " << status);
            }
        }

        dlclose(hsa_handle);
    }

    bool ROCmBackend::importExternalMemory(int fd, size_t size, void **device_ptr)
    {
        if (fd < 0 || !device_ptr || size == 0)
        {
            return false;
        }

        LOG_INFO("[ROCmBackend::importExternalMemory] Attempting to import fd=" << fd << ", size=" << size);

        // Use hipImportExternalMemory API
        hipExternalMemoryHandleDesc extMemHandleDesc = {};
        extMemHandleDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
        extMemHandleDesc.handle.fd = fd;
        extMemHandleDesc.size = size;
        extMemHandleDesc.flags = 0;

        hipExternalMemory_t extMem = nullptr;
        hipError_t err = hipImportExternalMemory(&extMem, &extMemHandleDesc);

        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::importExternalMemory] hipImportExternalMemory failed: "
                      << hipGetErrorString(err));
            *device_ptr = nullptr;
            return false;
        }

        LOG_INFO("[ROCmBackend::importExternalMemory] hipImportExternalMemory succeeded, getting mapped buffer...");

        // Map the external memory to a device pointer
        hipExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = 0;
        bufferDesc.size = size;
        bufferDesc.flags = 0;

        void *mappedPtr = nullptr;
        err = hipExternalMemoryGetMappedBuffer(&mappedPtr, extMem, &bufferDesc);

        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::importExternalMemory] hipExternalMemoryGetMappedBuffer failed: "
                      << hipGetErrorString(err));
            hipDestroyExternalMemory(extMem);
            *device_ptr = nullptr;
            return false;
        }

        LOG_INFO("[ROCmBackend::importExternalMemory] SUCCESS! mapped_ptr="
                 << std::hex << mappedPtr << std::dec);

        *device_ptr = mappedPtr;
        // Note: We should store extMem for later cleanup, but for now we're exploring
        return true;
    }

    bool ROCmBackend::getHsaAgent(int device_id, uint64_t *agent)
    {
        if (!agent || device_id < 0 || device_id >= device_count_)
        {
            return false;
        }

        // HIP exposes hipDeviceGetAttribute for getting the HSA agent
        // But we need to use HSA directly for this

        void *hsa_handle = dlopen("libhsa-runtime64.so", RTLD_NOW | RTLD_GLOBAL);
        if (!hsa_handle)
        {
            LOG_ERROR("[ROCmBackend::getHsaAgent] Failed to load HSA runtime");
            return false;
        }

        // We need to iterate agents to find GPU agents
        // This is complex - for now, we'll use hipGetDeviceProperties to get the agent

        hipDeviceProp_t prop;
        hipError_t err = hipGetDeviceProperties(&prop, device_id);
        if (err != hipSuccess)
        {
            dlclose(hsa_handle);
            return false;
        }

        // The gcnArchName contains arch info but not the HSA agent handle directly
        // For proper implementation, we'd need to iterate HSA agents

        LOG_INFO("[ROCmBackend::getHsaAgent] Device " << device_id << ": " << prop.name
                                                      << ", arch=" << prop.gcnArchName);

        dlclose(hsa_handle);

        // Return placeholder - proper implementation would need HSA agent iteration
        *agent = 0;
        return false; // Not fully implemented yet
    }

} // namespace llaminar2
