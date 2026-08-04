/**
 * @file CUDABackend.cu
 * @brief CUDA backend implementation with cuda_runtime.h
 *
 * **Purpose**: Implements IBackend for NVIDIA GPUs. This .cu file is the ONLY
 * compilation unit that includes cuda_runtime.h, preventing header conflicts.
 *
 * @author David Sanftenberg
 */

#include "CUDABackend.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"
#include "../../execution/mtp/MTPVerifierOutcomeGraph.h"
#include "../../kernels/common/SamplingMath.h"
#include "../../kernels/cuda/ops/CUDAVectorAddKernels.h"
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <cstdint>

namespace llaminar2
{
    namespace
    {
        constexpr std::uintptr_t kDeviceAllocationAlignment = 256;
    }

    // ====================================================================
    // Helper Macros for CUDA Error Checking
    // ====================================================================
    //
    // CUDA_CHECK_OR_THROW: Use for hot-path control flow (success-path API
    // calls in compute kernels) where silent failure causes silent miscompute
    // or delayed segfault. Logs at ERROR and throws std::runtime_error with
    // file/line context.
    //
    // CUDA_WARN_IF_FAIL: Use for cleanup/destructor/rollback paths (free,
    // destroy, error-recovery after upstream failure, resource-clear-before-
    // reuse). Logs at WARN and continues. Throwing here would call
    // std::terminate from destructors on stack unwind, or mask the real
    // upstream failure when called during error rollback.
    //
    // cudaErrorCudartUnloading is silenced because it is expected during
    // process exit when the CUDA runtime tears down before our cleanup runs.
#define CUDA_CHECK_OR_THROW(call)                                                          \
    do                                                                                     \
    {                                                                                      \
        cudaError_t _err = (call);                                                         \
        if (_err != cudaSuccess)                                                           \
        {                                                                                  \
            std::ostringstream _oss;                                                       \
            _oss << "[CUDABackend] " << #call << " failed: "                               \
                 << cudaGetErrorString(_err) << " (" << __FILE__ << ":" << __LINE__ << ")";\
            LOG_ERROR(_oss.str());                                                         \
            throw std::runtime_error(_oss.str());                                          \
        }                                                                                  \
    } while (0)

#define CUDA_WARN_IF_FAIL(call)                                                            \
    do                                                                                     \
    {                                                                                      \
        cudaError_t _err = (call);                                                         \
        if (_err != cudaSuccess)                                                           \
        {                                                                                  \
            if (_err == cudaErrorCudartUnloading)                                          \
            {                                                                              \
                LOG_TRACE("[CUDABackend] " << #call                                        \
                                           << " skipped: CUDA runtime shutting down");    \
            }                                                                              \
            else                                                                           \
            {                                                                              \
                LOG_WARN("[CUDABackend] " << #call << " failed: "                          \
                                          << cudaGetErrorString(_err) << " ("             \
                                          << __FILE__ << ":" << __LINE__ << ")");         \
            }                                                                              \
        }                                                                                  \
    } while (0)

    // ====================================================================
    // Constructor / Destructor
    // ====================================================================

    CUDABackend::CUDABackend()
        : device_count_(0)
    {
        cudaError_t err = cudaGetDeviceCount(&device_count_);
        if (err != cudaSuccess)
        {
            device_count_ = 0;
            // Log warning but don't throw - allow CPU-only execution
        }
        penalty_buffers_.resize(
            static_cast<size_t>(std::max(device_count_, 0)));
    }

    CUDABackend::~CUDABackend()
    {
        // cudaDeviceReset() intentionally omitted - managed by CUDA runtime
    }

    // ====================================================================
    // Stream Resolution Helper
    // ====================================================================

    /**
     * @brief Convert an opaque execution stream after enforcing explicit ownership.
     *
     * CUDA's null stream is process-global scheduling state, not a harmless
     * default. Accepting it here would erase the producer/consumer ordering
     * expressed by the graph. Every executable backend API therefore fails at
     * this common boundary before it can enqueue work ambiguously.
     */
    static cudaStream_t requireExplicitStream(void *stream, const char *operation)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                std::string(operation ? operation : "CUDABackend operation") +
                " requires an explicit non-null CUDA stream");
        }
        return static_cast<cudaStream_t>(stream);
    }

    // ====================================================================
    // Memory Transfer Operations
    // ====================================================================

    bool CUDABackend::deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() to establish the CUDA runtime context for this thread.
        if (!setDevice(device_id))
        {
            return false;
        }

        cudaStream_t s = requireExplicitStream(stream, "CUDABackend::deviceToHost");
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s);
        if (err != cudaSuccess)
            return false;
        err = cudaStreamSynchronize(s);
        return (err == cudaSuccess);
    }

    bool CUDABackend::hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        cudaStream_t s = requireExplicitStream(stream, "CUDABackend::hostToDevice");
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
        if (err != cudaSuccess)
            return false;
        err = cudaStreamSynchronize(s);
        return (err == cudaSuccess);
    }

    bool CUDABackend::deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        // Same-GPU VRAM copy: both src and dst are device pointers on device_id.
        cudaStream_t s = requireExplicitStream(stream, "CUDABackend::deviceToDevice");
        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::deviceToDevice] cudaMemcpyAsync failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        err = cudaStreamSynchronize(s);
        return (err == cudaSuccess);
    }

    bool CUDABackend::synchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        cudaError_t err = cudaDeviceSynchronize();
        return (err == cudaSuccess);
    }

    bool CUDABackend::streamSynchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        // Synchronize only the default stream (nullptr), not all streams
        cudaError_t err = cudaStreamSynchronize(nullptr);
        return (err == cudaSuccess);
    }

    // ====================================================================
    // Event Operations (Fine-grained Synchronization)
    // ====================================================================

    void *CUDABackend::createEvent(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return nullptr;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return nullptr;
        }

        cudaEvent_t event;
        err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
        // Note: cudaEventDisableTiming avoids GPU pipeline flush from timing events
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::createEvent] cudaEventCreate failed: " << cudaGetErrorString(err));
            return nullptr;
        }

        return reinterpret_cast<void *>(event);
    }

    void *CUDABackend::createTimingEvent(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return nullptr;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return nullptr;
        }

        cudaEvent_t event;
        /*
         * Unlike createEvent(), this intentionally keeps timing enabled.  It
         * is used only under perfstats instrumentation so normal inference
         * synchronization events remain cheap.
         */
        err = cudaEventCreate(&event);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::createTimingEvent] cudaEventCreate failed: " << cudaGetErrorString(err));
            return nullptr;
        }

        return reinterpret_cast<void *>(event);
    }

    void CUDABackend::destroyEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return;
        }

        CUDA_WARN_IF_FAIL(cudaSetDevice(device_id)); // cleanup path
        cudaEvent_t cuda_event = reinterpret_cast<cudaEvent_t>(event);
        CUDA_WARN_IF_FAIL(cudaEventDestroy(cuda_event));
    }

    bool CUDABackend::recordEvent(void *event, int device_id, void *stream)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        cudaEvent_t cuda_event = reinterpret_cast<cudaEvent_t>(event);
        cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::recordEvent");

        /*
         * IBackend events publish completed graph work to consumers outside the
         * captured DAG. A cudaEventRecord made during capture becomes an
         * internal graph node and cannot serve that external publication
         * contract. Internal graph fork/join edges belong to
         * IWorkerGPUContext; reject accidental capture-time publication here
         * instead of silently claiming that an event was recorded.
         */
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        err = cudaStreamIsCapturing(cuda_stream, &capture_status);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::recordEvent] cudaStreamIsCapturing failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        if (capture_status != cudaStreamCaptureStatusNone)
        {
            LOG_ERROR("[CUDABackend::recordEvent] External event publication is forbidden during graph capture; record it after graph launch");
            return false;
        }
        err = cudaEventRecord(cuda_event, cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::recordEvent] cudaEventRecord failed: " << cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    bool CUDABackend::eventElapsedTimeMs(
        void *start_event,
        void *stop_event,
        int device_id,
        float *out_ms)
    {
        if (!start_event || !stop_event || !out_ms ||
            device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        err = cudaEventElapsedTime(
            out_ms,
            reinterpret_cast<cudaEvent_t>(start_event),
            reinterpret_cast<cudaEvent_t>(stop_event));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::eventElapsedTimeMs] cudaEventElapsedTime failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDABackend::waitForEvent(void *event, int device_id)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            LOG_ERROR("[CUDABackend::waitForEvent] setDevice(" << device_id << ") failed");
            return false;
        }

        cudaEvent_t cuda_event = reinterpret_cast<cudaEvent_t>(event);
        cudaError_t err = cudaEventSynchronize(cuda_event);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::waitForEvent] cudaEventSynchronize failed: " << cudaGetErrorString(err)
                                                                                  << " (device=" << device_id << ", event=" << event << ")");
            return false;
        }

        return true;
    }

    bool CUDABackend::setDevice(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return false;
        }
        LOG_TRACE("[CUDABackend::setDevice] Set CUDA runtime device " << device_id);
        return true;
    }

    // ====================================================================
    // Memory Allocation Operations
    // ====================================================================

    void *CUDABackend::allocate(size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " (max: " << device_count_ - 1 << ")");
            return nullptr;
        }

        // Set device before allocation
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] Failed to set device " << device_id << ": " << cudaGetErrorString(err));
            return nullptr;
        }

        // Pre-allocation memory check: verify sufficient free VRAM before attempting cudaMalloc.
        // This provides a graceful error with actionable diagnostics instead of a raw OOM crash.
        {
            size_t free_bytes = 0, total_bytes = 0;
            cudaError_t mem_err = cudaMemGetInfo(&free_bytes, &total_bytes);
            if (mem_err == cudaSuccess)
            {
                // Require at least 64MB headroom beyond the allocation itself
                constexpr size_t HEADROOM = 64ULL * 1024 * 1024;
                if (bytes + HEADROOM > free_bytes)
                {
                    double req_mb = bytes / (1024.0 * 1024.0);
                    double free_mb = free_bytes / (1024.0 * 1024.0);
                    double total_mb = total_bytes / (1024.0 * 1024.0);
                    double used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0);
                    LOG_ERROR("[CUDABackend] Insufficient GPU memory on device " << device_id
                                                                                 << ": requested " << std::fixed << std::setprecision(1) << req_mb
                                                                                 << " MB but only " << free_mb << " MB free ("
                                                                                 << used_mb << " / " << total_mb << " MB used). "
                                                                                 << "Try reducing context length (-c), using a smaller model, "
                                                                                 << "or adding more GPUs for tensor parallelism.");
                    return nullptr;
                }
            }
        }

        void *ptr = nullptr;
        err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess)
        {
            // Include memory diagnostics in the error message
            size_t free_bytes = 0, total_bytes = 0;
            (void)cudaMemGetInfo(&free_bytes, &total_bytes); // best-effort enrichment for the LOG_ERROR below
            LOG_ERROR("[CUDABackend] cudaMalloc failed for " << bytes << " bytes on device "
                                                             << device_id << ": " << cudaGetErrorString(err)
                                                             << " (free: " << (free_bytes / (1024 * 1024))
                                                             << " MB, total: " << (total_bytes / (1024 * 1024)) << " MB)");
            return nullptr;
        }

        if ((reinterpret_cast<std::uintptr_t>(ptr) & (kDeviceAllocationAlignment - 1)) != 0)
        {
            LOG_ERROR("[CUDABackend] cudaMalloc returned unaligned pointer " << ptr
                                                                             << " for " << bytes << " bytes on device "
                                                                             << device_id << " (required "
                                                                             << kDeviceAllocationAlignment << "-byte alignment)");
            cudaFree(ptr);
            return nullptr;
        }

        LOG_TRACE("[CUDABackend::allocate] ALLOC ptr=" << ptr << " bytes=" << bytes << " device_id=" << device_id);
        if (vramBomEnabled())
        {
            size_t free_after = 0;
            size_t total_bytes = 0;
            (void)cudaMemGetInfo(&free_after, &total_bytes);
            logVramBomLine(
                "backend_allocation",
                "backend=cuda action=allocate device=" + std::to_string(device_id) +
                    " ptr=" + vramBomPointer(ptr) +
                    " " + vramBomBytes(bytes) +
                    " free_after_bytes=" + std::to_string(free_after) +
                    " total_bytes=" + std::to_string(total_bytes));
        }
        return ptr;
    }

    void CUDABackend::free(void *ptr, int device_id)
    {
        if (ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " for cudaFree");
            return;
        }

        // Set device before freeing
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            // Benign during process exit — driver is already shutting down
            LOG_DEBUG("[CUDABackend] Failed to set device " << device_id << " before cudaFree: "
                                                            << cudaGetErrorString(err));
            return;
        }

        const std::string bom_ptr =
            vramBomEnabled() ? vramBomPointer(ptr) : std::string{};
        err = cudaFree(ptr);
        if (err != cudaSuccess)
        {
            LOG_DEBUG("[CUDABackend] cudaFree failed for ptr=" << std::hex << ptr << std::dec
                                                               << " on device " << device_id << ": " << cudaGetErrorString(err));
        }
        else if (vramBomEnabled())
        {
            size_t free_after = 0;
            size_t total_bytes = 0;
            (void)cudaMemGetInfo(&free_after, &total_bytes);
            logVramBomLine(
                "backend_allocation",
                "backend=cuda action=free device=" + std::to_string(device_id) +
                    " ptr=" + bom_ptr +
                    " free_after_bytes=" + std::to_string(free_after) +
                    " total_bytes=" + std::to_string(total_bytes));
        }
    }

    bool CUDABackend::memset(void *ptr, int value, size_t bytes, int device_id, void *stream)
    {
        if (ptr == nullptr || bytes == 0)
        {
            return true; // No-op for null pointer or zero bytes
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " for cudaMemset");
            return false;
        }

        // Set device before memset
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] Failed to set device " << device_id << " before cudaMemset: "
                                                            << cudaGetErrorString(err));
            return false;
        }

        err = cudaMemsetAsync(
            ptr,
            value,
            bytes,
            requireExplicitStream(stream, "CUDABackend::memset"));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaMemsetAsync failed: " << cudaGetErrorString(err));
            return false;
        }

        return true;
    }

    void *CUDABackend::allocateMapped(size_t bytes, int device_id, void **device_ptr)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[CUDABackend] Invalid device ID " << device_id << " for allocateMapped");
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Set device before allocation
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] Failed to set device " << device_id << ": " << cudaGetErrorString(err));
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Allocate mapped host memory (GPU can write directly to this via PCIe)
        // NOTE: Do NOT use cudaHostAllocWriteCombined here. WC memory makes CPU
        // reads ~1000x slower (each load bypasses all CPU caches). Logits are
        // GPU-written then CPU-read (argmax in sampler), so WC provides no
        // benefit and causes a ~13ms penalty per token for 152K-vocab models.
        void *host_ptr = nullptr;
        err = cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaHostAlloc(Mapped) failed for " << bytes << " bytes on device "
                                                                        << device_id << ": " << cudaGetErrorString(err));
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Get the device-visible pointer for this mapped host memory
        if (device_ptr)
        {
            err = cudaHostGetDevicePointer(device_ptr, host_ptr, 0);
            if (err != cudaSuccess)
            {
                LOG_ERROR("[CUDABackend] cudaHostGetDevicePointer failed: " << cudaGetErrorString(err));
                CUDA_WARN_IF_FAIL(cudaFreeHost(host_ptr)); // rollback after cudaHostGetDevicePointer fail
                *device_ptr = nullptr;
                return nullptr;
            }
            LOG_TRACE("[CUDABackend] allocateMapped: " << bytes << " bytes, host_ptr=" << host_ptr
                                                       << ", device_ptr=" << *device_ptr);
        }

        return host_ptr;
    }

    void CUDABackend::freeMapped(void *host_ptr, int device_id)
    {
        if (host_ptr == nullptr)
        {
            return; // Freeing nullptr is a no-op
        }

        // cudaFreeHost doesn't require setting device, but we do it for consistency
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_WARN("[CUDABackend] Invalid device ID " << device_id << " for freeMapped, attempting anyway");
        }
        else
        {
            CUDA_WARN_IF_FAIL(cudaSetDevice(device_id)); // best-effort in cleanup path
        }

        cudaError_t err = cudaFreeHost(host_ptr);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend] cudaFreeHost failed: " << cudaGetErrorString(err));
        }
    }

    // ====================================================================
    // Device Query Operations
    // ====================================================================

    int CUDABackend::deviceCount() const
    {
        return device_count_;
    }

    std::string CUDABackend::backendName() const
    {
        return "CUDA";
    }

    std::string CUDABackend::deviceName(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return "Invalid Device";
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return "Unknown Device";
        }

        return std::string(prop.name);
    }

    size_t CUDABackend::deviceMemoryTotal(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return 0;
        }

        return prop.totalGlobalMem;
    }

    size_t CUDABackend::deviceMemoryFree(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return 0;
        }

        cudaError_t err_set = cudaSetDevice(device_id);
        if (err_set != cudaSuccess)
        {
            return 0;
        }

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
        if (err != cudaSuccess)
        {
            return 0;
        }

        return free_bytes;
    }

    // ====================================================================
    // Capability Queries
    // ====================================================================

    bool CUDABackend::supportsBF16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // BF16 support requires compute capability >= 8.0 (Ampere and later)
        int compute_capability = prop.major * 10 + prop.minor;
        return compute_capability >= 80;
    }

    bool CUDABackend::supportsFP16(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // FP16 support requires compute capability >= 5.3 (Maxwell and later)
        int compute_capability = prop.major * 10 + prop.minor;
        return compute_capability >= 53;
    }

    bool CUDABackend::supportsINT8(int device_id) const
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        cudaDeviceProp prop;
        cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // INT8 support requires compute capability >= 6.1 (Pascal and later)
        int compute_capability = prop.major * 10 + prop.minor;
        return compute_capability >= 61;
    }

    // ====================================================================
    // Compute Operations
    // ====================================================================

    bool CUDABackend::gemmIQ4NL(
        const void * /*A_device*/,
        const void * /*B_device*/,
        void * /*C_device*/,
        int /*m*/,
        int /*n*/,
        int /*k*/,
        int /*device_id*/)
    {
        // IQ4_NL GEMM via backend is deprecated - use kernel interface directly
        LOG_ERROR("CUDABackend::gemmIQ4NL is deprecated and no longer implemented");
        return false;
    }

    // ====================================================================
    // GPU-side Sampling Operations
    // ====================================================================

    // ── pinHostMemory / unpinHostMemory ────────────────────────────────────

    bool CUDABackend::pinHostMemory(void *ptr, size_t bytes)
    {
        cudaError_t err = cudaHostRegister(ptr, bytes, cudaHostRegisterDefault);
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDABackend::pinHostMemory] cudaHostRegister failed for "
                     << bytes << " bytes: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDABackend::unpinHostMemory(void *ptr)
    {
        cudaError_t err = cudaHostUnregister(ptr);
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDABackend::unpinHostMemory] cudaHostUnregister failed: "
                     << cudaGetErrorString(err));
            // Clear the sticky CUDA error so it doesn't contaminate subsequent
            // CUDA operations (kernel launches, memcpy, etc.).  This commonly
            // happens during teardown when mmap pages are already unmapped.
            (void)cudaGetLastError();
            return false;
        }
        return true;
    }

    // Forward declarations for CUDA sampling kernels (CUDASamplingKernels.cu)
    extern "C" bool cudaOps_argmax_f32(
        const float *data, int n, float *out_value, int *out_index,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream);
    extern "C" bool cudaOps_argmax_f32_batched_rows(
        const float *data, int rows, int cols, int row_stride,
        float *out_values, int *out_indices,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool cudaOps_argmax_f32_batched_rows_publish_mtp_chain(
        const float *data, int rows, int cols, int row_stride,
        float *out_values, int *out_indices,
        int *chain_condition_tokens, int *chain_position_ids,
        int chain_position_increment,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool cudaOps_retain_mtp_first_transaction_draft_boundary(
        const uint32_t *data_words,
        int word_count,
        int boundary,
        int draft_slot,
        const int *condition_token,
        const int *position_id,
        const int *generation_control,
        int generation_control_stride,
        void *diagnostic_record,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_configure_mtp_greedy_penalty_policy(
        MTPGreedyPenaltyPolicy *controls,
        float presence_penalty,
        float frequency_penalty,
        bool first_token_already_in_history,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_argmax_f32_batched_rows_mtp_penalties(
        const float *data, int rows, int cols, int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        float *out_values, int *out_indices,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool cudaOps_apply_mtp_penalties_f32_rows(
        float *data, int rows, int cols, int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        int device_idx, void *stream);
    extern "C" bool cudaOps_apply_mtp_branch_penalties_f32_row(
        float *data, int cols,
        const int *first_condition_token,
        const int *prior_draft_tokens,
        int prior_draft_count,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        int device_idx, void *stream);
    extern "C" bool cudaOps_commit_mtp_greedy_penalty_history(
        const int *output_tokens,
        const int *output_meta,
        MTPGreedyPenaltyPolicy *policy,
        const int *accepted_state_counts,
        const int *stopped_flags,
        int output_token_capacity,
        int vocab_size,
        int *generated_token_counts,
        int device_idx,
        void *stream);

    extern "C" bool cudaOps_topk_f32(
        const float *data, int n, int k, float *out_values, int *out_indices,
        int device_idx, void *stream);
    extern "C" bool cudaOps_sample_topk_topp_f32(
        const float *data, int n, int k, float top_p, float temperature,
        unsigned long long rng_seed, unsigned long long rng_offset,
        int *out_token, int device_idx, void *stream);
    extern "C" bool cudaOps_publish_int32_control_scalar(
        int32_t value,
        int32_t *out_value,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_topk_topp_distribution_f32(
        const float *data, int n, int k, float top_p, float temperature,
        int *out_token_ids, float *out_probs,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        int device_idx, void *stream);
    extern "C" bool cudaOps_topk_topp_distributions_f32(
        const float *data, int row_count, int n, int row_stride, int k,
        float top_p, float temperature,
        int *out_token_ids, int out_stride, float *out_probs,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        const int *active_rows,
        int device_idx, void *stream);
    extern "C" bool cudaOps_topk_topp_processed_logits_f32(
        const float *data, int row_count, int n, int row_stride, int k,
        float top_p, float temperature,
        float *out_logits, int out_row_stride,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int draft_token,
        unsigned long long accept_seed, unsigned long long accept_offset,
        unsigned long long residual_seed, unsigned long long residual_offset,
        int *out_token, int *out_accepted,
        float *out_accept_probability, float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool cudaOps_sample_distribution_f32(
        const int *token_ids, const float *probs,
        int k, float threshold,
        int *out_token, float *out_probability,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        int device_idx, void *stream);
    extern "C" bool cudaOps_sample_processed_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float threshold,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float threshold,
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        int first_token,
        const int *first_token_device,
        int stop_token0,
        int stop_token1,
        int stop_token2,
        int stop_token3,
        int stop_token4,
        int stop_token5,
        int stop_token6,
        int stop_token7,
        int stop_token_count,
        int *out_token,
        float *out_probability,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_softmax_processed_logits_f32(
        const float *logits,
        int row_count,
        int vocab_size,
        int row_stride,
        float *out_probabilities,
        int out_row_stride,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_softmax_sample_temperature_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        float *out_probabilities,
        int out_row_stride,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_scale_sample_temperature_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        float *out_logits,
        int out_row_stride,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_fill_inverse_exponential_samples_f32(
        float *out_samples,
        int row_count,
        int vocab_size,
        int row_stride,
        unsigned long long seed,
        int first_logical_position,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_threshold_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int draft_token,
        float accept_threshold, float residual_threshold,
        int *out_token, int *out_accepted,
        float *out_accept_probability, float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_thresholds_batch_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int distribution_stride,
        const int *draft_tokens_host,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int row_count,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int distribution_stride,
        const int *sampled_draft_tokens,
        const float *sampled_draft_probabilities,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int row_count,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int inverse_sample_vocab_size,
        unsigned long long threshold_seed,
        int threshold_first_logical_position,
        int thresholds_from_seed,
        const int *threshold_base_position,
        int threshold_position_offset,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_logits,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        const float *draft_token_probabilities,
        int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_probabilities,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        const float *accept_thresholds_host,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int thresholds_from_seed,
        const int *threshold_base_position,
        int threshold_position_offset,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int no_draft_probabilities,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
        const float *target_logits,
        const float *draft_logits,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const int *sampled_draft_tokens,
        const float *sampled_draft_probabilities,
        const float *accept_thresholds_host,
        unsigned long long inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
        const float *target_probabilities,
        const float *draft_probabilities,
        const float *inverse_rejection_samples,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        int inverse_sample_row_stride,
        const int *sampled_draft_tokens,
        const float *accept_thresholds_host,
        int no_draft_probabilities,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_summarize_speculative_verify_batch(
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        int first_token,
        int stop_token0, int stop_token1, int stop_token2, int stop_token3,
        int stop_token4, int stop_token5, int stop_token6, int stop_token7,
        int stop_token_count,
        const int *bonus_token,
        int has_bonus_token,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_summarize_speculative_verify_batch_device_first_token(
        const int *verify_tokens,
        const int *verify_accepted,
        int row_count,
        const int *first_token,
        int stop_token0, int stop_token1, int stop_token2, int stop_token3,
        int stop_token4, int stop_token5, int stop_token6, int stop_token7,
        int stop_token_count,
        const int *bonus_token,
        int has_bonus_token,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_summarize_speculative_verify_batch_device_generation_controls(
        const int *verify_tokens,
        const int *verify_accepted,
        const int *greedy_draft_tokens,
        int row_count,
        const int *first_token,
        const int *stop_tokens,
        const int *bonus_token,
        int has_bonus_token,
        const int *generation_control,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
        const int *target_token_ids,
        const float *target_probs,
        int target_row_stride,
        int top_k,
        int row_count,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        const int *verifier_input_tokens,
        const int *stop_tokens,
        const int *generation_control,
        int *sampled_target_tokens,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        void *first_transaction_diagnostic,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_summarize_greedy_speculative_verify_batch(
        const int *verify_tokens,
        const int *draft_tokens,
        int compare_row_count,
        int first_token,
        int stop_token0, int stop_token1, int stop_token2, int stop_token3,
        int stop_token4, int stop_token5, int stop_token6, int stop_token7,
        int stop_token_count,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        int leading_committed_output_count,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_summarize_greedy_speculative_verify_batch_device_controls(
        const int *verify_tokens,
        const int *draft_tokens,
        int compare_row_count,
        const int *active_verifier_row_count,
        const int *stop_tokens,
        int *out_tokens,
        int out_token_capacity,
        int *out_meta,
        const uint32_t *max_state_commit_rows,
        const MTPGreedyPenaltyPolicy *penalty_policy,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_advance_speculative_commit_boundary(
        const int *meta,
        int request_count,
        int meta_stride,
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_initialize_device_generation(
        int request_count,
        int max_new_tokens,
        const sampling_math::DeviceGenerationDepthPolicy &depth_policy,
        int response_token_stride,
        int control_stride,
        int *control,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_device_generation_transaction_budget(
        int *control,
        int control_stride,
        int request_count,
        int verifier_row_capacity,
        const uint32_t *maintenance_rows_remaining,
        const uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_commit_device_generation_and_derive_speculative_publication_metadata(
        const int32_t *compact_tokens,
        int output_token_stride,
        int *compact_meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int32_t *response_tokens,
        int response_token_stride,
        int *control,
        int control_stride,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_sidecar_condition_tokens,
        int *out_next_sidecar_position_ids,
        int *out_next_verifier_condition_tokens,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_derive_speculative_publication_metadata(
        const int *meta,
        int meta_stride,
        const int *base_cached_tokens,
        int request_count,
        int padded_state_rows_per_request,
        int max_state_commit_rows,
        int *out_restore_rows,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int *out_next_condition_tokens,
        const int32_t *output_tokens,
        int output_token_stride,
        int *out_all_drafts_accepted_flags,
        int *out_stopped_flags,
        int *out_next_verifier_condition_tokens,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaOps_derive_shifted_speculative_publication_metadata_from_primary(
        const int *base_cached_tokens,
        const int *main_target_cached_tokens,
        const int *main_publication_ok,
        int request_count,
        int mtp_depth,
        int *out_target_cached_tokens,
        int *out_accepted_state_counts,
        int *out_ok,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_speculative_shifted_kv_tokens(
        const int *meta,
        int meta_stride,
        const int32_t *output_tokens,
        int output_token_stride,
        int request_index,
        int first_output_token_index,
        int row_count,
        int32_t filler_token,
        int32_t *out_tokens,
        const int32_t *base_positions,
        int position_offset,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_batched_sidecar_inputs(
        const int32_t *condition_tokens,
        int condition_token_stride,
        const int32_t *base_positions,
        int position_offset,
        int request_count,
        int32_t *out_condition_tokens,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_verifier_position_ids(
        const int32_t *base_positions,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_verifier_geometry(
        const int32_t *base_positions,
        const int32_t *valid_graph_rows,
        int valid_graph_row_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int32_t *out_request_lengths,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_prepare_mtp_verifier_controlled_row(
        const int32_t *first_token,
        const int32_t *draft_tokens,
        const int32_t *base_position,
        int *generation_control_row,
        int generation_control_stride,
        int padded_seq_len,
        int32_t *out_tokens,
        int32_t *out_position_ids,
        int32_t *out_request_length,
        int32_t *out_base_position_snapshot,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_initialize_mtp_device_logical_state(
        const int32_t *sampled_tokens,
        const int32_t *target_positions_device,
        int request_count,
        int32_t *out_base_cached_tokens,
        int32_t *out_target_positions,
        int32_t *out_accepted_state_counts,
        int32_t *out_next_condition_tokens,
        int32_t *out_all_drafts_accepted_flags,
        int32_t *out_stopped_flags,
        int32_t *out_publication_ok_flags,
        int device_idx,
        void *stream);

    bool CUDABackend::argmaxF32(const void *data_device, int n, int device_id,
                                float *out_value, int *out_index, void *stream,
                                void *partial_vals, void *partial_idxs, int partial_capacity)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device || n <= 0)
            return false;

        // Lazily allocate the tiny per-device D2H result staging buffers (8 bytes
        // total). The larger partial-reduction scratch is NOT allocated here — it
        // is owned by the orchestrator's BufferArena and supplied by the caller.
        if (argmax_buffers_.empty())
            argmax_buffers_.resize(device_count_);

        auto &bufs = argmax_buffers_[device_id];
        if (!bufs.value_ptr)
        {
            cudaError_t err = cudaSetDevice(device_id);
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(&bufs.value_ptr, sizeof(float));
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(&bufs.index_ptr, sizeof(int));
            if (err != cudaSuccess)
            {
                CUDA_WARN_IF_FAIL(cudaFree(bufs.value_ptr)); // rollback after cudaMalloc fail
                bufs.value_ptr = nullptr;
                return false;
            }
            bufs.allocated_count = 1;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        cudaStream_t s = requireExplicitStream(stream, "CUDABackend::argmaxF32");
        // Pass the caller-supplied partial scratch through to the kernel wrapper.
        // The scratch is mandatory (arena-owned); the wrapper fails loud if it is
        // missing or undersized — there is no single-block fallback.
        if (!cudaOps_argmax_f32(
                static_cast<const float *>(data_device), n,
                static_cast<float *>(bufs.value_ptr),
                static_cast<int *>(bufs.index_ptr),
                static_cast<float *>(partial_vals),
                static_cast<int *>(partial_idxs),
                partial_capacity,
                device_id, s))
        {
            return false;
        }

        // The argmax kernel and the two D2H copies are all enqueued on stream `s`,
        // so stream ordering already guarantees the copies observe the kernel's
        // results. A single synchronize after the copies is sufficient — an
        // intermediate sync between the kernel and the copies would add a
        // redundant host<->GPU round-trip on the per-decode-step hot path.
        CUDA_CHECK_OR_THROW(cudaMemcpyAsync(out_value, bufs.value_ptr, sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK_OR_THROW(cudaMemcpyAsync(out_index, bufs.index_ptr, sizeof(int), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK_OR_THROW(cudaStreamSynchronize(s));

        return true;
    }

    bool CUDABackend::argmaxF32BatchedRows(const void *data_device, int rows, int cols, int device_id,
                                           float *out_values, int *out_indices, void *stream,
                                           void *partial_vals, void *partial_idxs, int partial_capacity)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device ||
            rows <= 0 || cols <= 0 || !out_values || !out_indices)
        {
            return false;
        }

        if (!partial_vals || !partial_idxs || partial_capacity < rows)
        {
            LOG_ERROR("[CUDABackend::argmaxF32BatchedRows] missing arena-owned partial scratch "
                      << "(rows=" << rows << " capacity=" << partial_capacity << ")");
            return false;
        }

        if (argmax_buffers_.empty())
            argmax_buffers_.resize(device_count_);

        auto &bufs = argmax_buffers_[device_id];
        if (!bufs.value_ptr || bufs.allocated_count < rows)
        {
            CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
            if (bufs.value_ptr)
                CUDA_WARN_IF_FAIL(cudaFree(bufs.value_ptr));
            if (bufs.index_ptr)
                CUDA_WARN_IF_FAIL(cudaFree(bufs.index_ptr));
            bufs.value_ptr = nullptr;
            bufs.index_ptr = nullptr;
            bufs.allocated_count = 0;

            cudaError_t err = cudaMalloc(&bufs.value_ptr, static_cast<size_t>(rows) * sizeof(float));
            if (err != cudaSuccess)
                return false;
            err = cudaMalloc(&bufs.index_ptr, static_cast<size_t>(rows) * sizeof(int));
            if (err != cudaSuccess)
            {
                CUDA_WARN_IF_FAIL(cudaFree(bufs.value_ptr));
                bufs.value_ptr = nullptr;
                return false;
            }
            bufs.allocated_count = rows;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        cudaStream_t s =
            requireExplicitStream(stream, "CUDABackend::argmaxF32BatchedRows");
        {
            PerfStatsCollector::ScopedTimer timer(
                "backend", "cuda_argmax_f32_batched_rows_launch", "decode");
            if (!cudaOps_argmax_f32_batched_rows(
                    static_cast<const float *>(data_device),
                    rows,
                    cols,
                    cols,
                    static_cast<float *>(bufs.value_ptr),
                    static_cast<int *>(bufs.index_ptr),
                    static_cast<float *>(partial_vals),
                    static_cast<int *>(partial_idxs),
                    partial_capacity,
                    device_id,
                    s,
                    /*output_stride=*/1))
            {
                return false;
            }
        }

        {
            PerfStatsCollector::ScopedTimer timer(
                "backend", "cuda_argmax_f32_batched_rows_d2h_enqueue", "decode");
            CUDA_CHECK_OR_THROW(cudaMemcpyAsync(out_values,
                                                bufs.value_ptr,
                                                static_cast<size_t>(rows) * sizeof(float),
                                                cudaMemcpyDeviceToHost,
                                                s));
            CUDA_CHECK_OR_THROW(cudaMemcpyAsync(out_indices,
                                                bufs.index_ptr,
                                                static_cast<size_t>(rows) * sizeof(int),
                                                cudaMemcpyDeviceToHost,
                                                s));
        }
        {
            PerfStatsCollector::ScopedTimer timer(
                "backend", "cuda_argmax_f32_batched_rows_sync", "decode");
            CUDA_CHECK_OR_THROW(cudaStreamSynchronize(s));
        }
        return true;
    }

    bool CUDABackend::enqueueArgmaxF32BatchedRowsDevice(
        const void *data_device,
        int rows,
        int cols,
        int device_id,
        void *stream,
        void *out_values_device,
        void *out_indices_device,
        void *partial_vals,
        void *partial_idxs,
        int partial_capacity,
        int output_stride)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device ||
            rows <= 0 || cols <= 0 || !stream ||
            !out_values_device || !out_indices_device ||
            !partial_vals || !partial_idxs || partial_capacity < rows ||
            output_stride <= 0)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend", "cuda_argmax_f32_batched_rows_device_launch", "decode");
        return cudaOps_argmax_f32_batched_rows(
            static_cast<const float *>(data_device),
            rows,
            cols,
            cols,
            static_cast<float *>(out_values_device),
            static_cast<int *>(out_indices_device),
            static_cast<float *>(partial_vals),
            static_cast<int *>(partial_idxs),
            partial_capacity,
            device_id,
            static_cast<cudaStream_t>(stream),
            output_stride);
    }

    bool CUDABackend::enqueueArgmaxF32BatchedRowsAndPublishMTPChainDevice(
        const void *data_device,
        int rows,
        int cols,
        int device_id,
        void *stream,
        void *out_values_device,
        void *out_indices_device,
        void *chain_condition_tokens_device,
        void *chain_position_ids_device,
        int chain_position_increment,
        void *partial_vals,
        void *partial_idxs,
        int partial_capacity,
        int output_stride)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device ||
            rows <= 0 || cols <= 0 || !stream || !out_values_device ||
            !out_indices_device || !chain_condition_tokens_device ||
            !chain_position_ids_device || chain_position_increment <= 0 ||
            !partial_vals || !partial_idxs || partial_capacity < rows ||
            output_stride <= 0)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "cuda_argmax_f32_mtp_chain_publication_launch",
            "decode");
        return cudaOps_argmax_f32_batched_rows_publish_mtp_chain(
            static_cast<const float *>(data_device),
            rows,
            cols,
            cols,
            static_cast<float *>(out_values_device),
            static_cast<int *>(out_indices_device),
            static_cast<int *>(chain_condition_tokens_device),
            static_cast<int *>(chain_position_ids_device),
            chain_position_increment,
            static_cast<float *>(partial_vals),
            static_cast<int *>(partial_idxs),
            partial_capacity,
            device_id,
            stream,
            output_stride);
    }

    bool CUDABackend::enqueueRetainMTPFirstTransactionDraftBoundaryDevice(
        const void *data_words_device,
        int word_count,
        int boundary,
        int draft_slot,
        const void *condition_token_device,
        const void *position_id_device,
        const void *generation_control_device,
        int generation_control_stride,
        void *diagnostic_record_device,
        int device_id,
        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            word_count < 0 || (word_count > 0 && !data_words_device) ||
            !condition_token_device || !position_id_device ||
            !generation_control_device || generation_control_stride <= 0 ||
            !diagnostic_record_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "cuda_mtp_first_transaction_draft_boundary_diagnostic_launch",
            "decode");
        return cudaOps_retain_mtp_first_transaction_draft_boundary(
            static_cast<const uint32_t *>(data_words_device),
            word_count,
            boundary,
            draft_slot,
            static_cast<const int *>(condition_token_device),
            static_cast<const int *>(position_id_device),
            static_cast<const int *>(generation_control_device),
            generation_control_stride,
            diagnostic_record_device,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueConfigureMTPGreedyPenaltyPolicyDevice(
        void *controls_device,
        float presence_penalty,
        float frequency_penalty,
        bool first_token_already_in_history,
        int device_id,
        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !controls_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_configure_mtp_greedy_penalty_policy(
            static_cast<MTPGreedyPenaltyPolicy *>(controls_device),
            presence_penalty,
            frequency_penalty,
            first_token_already_in_history,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueArgmaxF32BatchedRowsWithMTPPenaltiesDevice(
        const void *data_device,
        int rows,
        int cols,
        const void *verifier_input_tokens_device,
        const void *generated_token_counts_device,
        const void *penalty_policy_device,
        const void *active_rows_device,
        int device_id,
        void *stream,
        void *out_values_device,
        void *out_indices_device,
        void *partial_vals,
        void *partial_idxs,
        int partial_capacity,
        int output_stride)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !data_device || rows <= 0 || cols <= 0 ||
            !verifier_input_tokens_device ||
            !generated_token_counts_device || !penalty_policy_device ||
            !active_rows_device ||
            !stream || !out_values_device || !out_indices_device ||
            !partial_vals || !partial_idxs || partial_capacity < rows ||
            output_stride <= 0)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "cuda_mtp_penalty_argmax_batched_rows_device_launch",
            "decode");
        return cudaOps_argmax_f32_batched_rows_mtp_penalties(
            static_cast<const float *>(data_device),
            rows,
            cols,
            cols,
            static_cast<const int *>(verifier_input_tokens_device),
            static_cast<const int *>(generated_token_counts_device),
            static_cast<const MTPGreedyPenaltyPolicy *>(
                penalty_policy_device),
            static_cast<const int *>(active_rows_device),
            static_cast<float *>(out_values_device),
            static_cast<int *>(out_indices_device),
            static_cast<float *>(partial_vals),
            static_cast<int *>(partial_idxs),
            partial_capacity,
            device_id,
            stream,
            output_stride);
    }

    bool CUDABackend::enqueueApplyMTPPenaltiesToF32RowsDevice(
        void *data_device,
        int rows,
        int cols,
        int row_stride,
        const void *verifier_input_tokens_device,
        const void *generated_token_counts_device,
        const void *penalty_policy_device,
        int device_id,
        void *stream,
        const void *active_rows_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !data_device || rows <= 0 || cols <= 0 || row_stride < cols ||
            !generated_token_counts_device || !penalty_policy_device ||
            !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "cuda_mtp_penalty_logit_rows_device_launch",
            "decode");
        return cudaOps_apply_mtp_penalties_f32_rows(
            static_cast<float *>(data_device),
            rows,
            cols,
            row_stride,
            static_cast<const int *>(verifier_input_tokens_device),
            static_cast<const int *>(generated_token_counts_device),
            static_cast<const MTPGreedyPenaltyPolicy *>(
                penalty_policy_device),
            static_cast<const int *>(active_rows_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueApplyMTPBranchPenaltiesToF32RowDevice(
        void *data_device,
        int cols,
        const void *first_condition_token_device,
        const void *prior_draft_tokens_device,
        int prior_draft_count,
        const void *generated_token_counts_device,
        const void *penalty_policy_device,
        int device_id,
        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !data_device || cols <= 0 || !first_condition_token_device ||
            prior_draft_count < 0 ||
            (prior_draft_count > 0 && !prior_draft_tokens_device) ||
            !generated_token_counts_device || !penalty_policy_device ||
            !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "cuda_mtp_branch_penalty_logit_row_device_launch",
            "decode");
        return cudaOps_apply_mtp_branch_penalties_f32_row(
            static_cast<float *>(data_device),
            cols,
            static_cast<const int *>(first_condition_token_device),
            static_cast<const int *>(prior_draft_tokens_device),
            prior_draft_count,
            static_cast<const int *>(generated_token_counts_device),
            static_cast<const MTPGreedyPenaltyPolicy *>(
                penalty_policy_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueCommitMTPGreedyPenaltyHistoryDevice(
        const void *output_tokens_device,
        const void *output_meta_device,
        void *penalty_policy_device,
        const void *accepted_state_counts_device,
        const void *stopped_flags_device,
        int output_token_capacity,
        int vocab_size,
        void *generated_token_counts_device,
        int device_id,
        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !output_tokens_device || !output_meta_device ||
            !penalty_policy_device || !accepted_state_counts_device ||
            !stopped_flags_device || output_token_capacity <= 0 ||
            vocab_size <= 0 || !generated_token_counts_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_commit_mtp_greedy_penalty_history(
            static_cast<const int *>(output_tokens_device),
            static_cast<const int *>(output_meta_device),
            static_cast<MTPGreedyPenaltyPolicy *>(
                penalty_policy_device),
            static_cast<const int *>(accepted_state_counts_device),
            static_cast<const int *>(stopped_flags_device),
            output_token_capacity,
            vocab_size,
            static_cast<int *>(generated_token_counts_device),
            device_id,
            stream);
    }

    bool CUDABackend::topKF32(const void *data_device, int n, int k, int device_id,
                              float *out_values, int *out_indices, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device || n <= 0 || k <= 0)
            return false;

        if (k > n)
            k = n;

        // Lazily allocate per-device result buffers
        if (topk_buffers_.empty())
            topk_buffers_.resize(device_count_);

        auto &bufs = topk_buffers_[device_id];

        if (bufs.allocated_k < k)
        {
            CUDA_WARN_IF_FAIL(cudaSetDevice(device_id)); // realloc path; subsequent cudaMalloc will surface real errors
            if (bufs.values_ptr)
                CUDA_WARN_IF_FAIL(cudaFree(bufs.values_ptr)); // clearing old buffer before realloc
            if (bufs.indices_ptr)
                CUDA_WARN_IF_FAIL(cudaFree(bufs.indices_ptr)); // clearing old buffer before realloc

            cudaError_t err = cudaMalloc(&bufs.values_ptr, k * sizeof(float));
            if (err != cudaSuccess)
            {
                bufs.values_ptr = nullptr;
                bufs.allocated_k = 0;
                return false;
            }
            err = cudaMalloc(&bufs.indices_ptr, k * sizeof(int));
            if (err != cudaSuccess)
            {
                CUDA_WARN_IF_FAIL(cudaFree(bufs.values_ptr)); // rollback after cudaMalloc fail
                bufs.values_ptr = nullptr;
                bufs.allocated_k = 0;
                return false;
            }
            bufs.allocated_k = k;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        cudaStream_t s = requireExplicitStream(stream, "CUDABackend::topKF32");
        if (!cudaOps_topk_f32(
                static_cast<const float *>(data_device), n, k,
                static_cast<float *>(bufs.values_ptr),
                static_cast<int *>(bufs.indices_ptr),
                device_id, s))
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaStreamSynchronize(s));
        CUDA_CHECK_OR_THROW(cudaMemcpyAsync(out_values, bufs.values_ptr, k * sizeof(float), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK_OR_THROW(cudaMemcpyAsync(out_indices, bufs.indices_ptr, k * sizeof(int), cudaMemcpyDeviceToHost, s));
        CUDA_CHECK_OR_THROW(cudaStreamSynchronize(s));

        return true;
    }

    bool CUDABackend::enqueueSampleTopKTopPF32Device(const void *data_device, int n,
                                                     int top_k, float top_p, float temperature,
                                                     uint64_t rng_seed, uint64_t rng_offset,
                                                     int device_id, void *stream,
                                                     void *out_token_device)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device ||
            n <= 0 || top_k <= 0 || !stream || !out_token_device)
        {
            return false;
        }

        if (top_k > 256)
            top_k = 256;
        if (top_k > n)
            top_k = n;

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_sample_topk_topp_f32(
            static_cast<const float *>(data_device),
            n,
            top_k,
            top_p,
            temperature,
            static_cast<unsigned long long>(rng_seed),
            static_cast<unsigned long long>(rng_offset),
            static_cast<int *>(out_token_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePublishInt32ControlScalarDevice(
        int32_t value,
        void *out_value_device,
        int device_id,
        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            value < 0 || !out_value_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_publish_int32_control_scalar(
            value,
            static_cast<int32_t *>(out_value_device),
            device_id,
            stream);
    }

    bool CUDABackend::sampleTopKTopPF32(const void *data_device, int n,
                                        int top_k, float top_p, float temperature,
                                        uint64_t rng_seed, uint64_t rng_offset,
                                        int device_id, int *out_token,
                                        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device ||
            n <= 0 || top_k <= 0 || !out_token || !stream)
        {
            return false;
        }

        if (sample_token_buffers_.empty())
            sample_token_buffers_.resize(device_count_);

        auto &bufs = sample_token_buffers_[device_id];
        if (!bufs.token_ptr)
        {
            CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
            cudaError_t err = cudaMalloc(&bufs.token_ptr, sizeof(int));
            if (err != cudaSuccess)
            {
                bufs.token_ptr = nullptr;
                return false;
            }
        }

        if (!enqueueSampleTopKTopPF32Device(data_device,
                                            n,
                                            top_k,
                                            top_p,
                                            temperature,
                                            rng_seed,
                                            rng_offset,
                                            device_id,
                                            stream,
                                            bufs.token_ptr))
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaMemcpyAsync(out_token,
                                            bufs.token_ptr,
                                            sizeof(int),
                                            cudaMemcpyDeviceToHost,
                                            static_cast<cudaStream_t>(stream)));
        CUDA_CHECK_OR_THROW(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        return true;
    }

    bool CUDABackend::enqueueBuildTopKTopPDistributionF32Device(
        const void *data_device,
        int n,
        int top_k,
        float top_p,
        float temperature,
        int device_id,
        void *stream,
        void *out_token_ids_device,
        void *out_probs_device,
        void *scratch_values_device,
        void *scratch_indices_device,
        int scratch_capacity)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device ||
            n <= 0 || top_k <= 0 || !stream || !out_token_ids_device || !out_probs_device)
        {
            return false;
        }

        if (top_k > 256)
            top_k = 256;
        if (top_k > n)
            top_k = n;

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_topk_topp_distribution_f32(
            static_cast<const float *>(data_device),
            n,
            top_k,
            top_p,
            temperature,
            static_cast<int *>(out_token_ids_device),
            static_cast<float *>(out_probs_device),
            static_cast<float *>(scratch_values_device),
            static_cast<int *>(scratch_indices_device),
            scratch_capacity,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueBuildTopKTopPDistributionsF32Device(
        const void *data_device,
        int row_count,
        int n,
        int row_stride,
        int top_k,
        float top_p,
        float temperature,
        int device_id,
        void *stream,
        void *out_token_ids_device,
        int out_stride,
        void *out_probs_device,
        void *scratch_values_device,
        void *scratch_indices_device,
        int scratch_capacity,
        const void *active_rows_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !data_device || row_count <= 0 || n <= 0 || row_stride < n ||
            top_k <= 0 || out_stride <= 0 || out_stride < top_k ||
            !stream || !out_token_ids_device || !out_probs_device)
        {
            return false;
        }

        if (top_k > 256)
            top_k = 256;
        if (top_k > n)
            top_k = n;

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_topk_topp_distributions_f32(
            static_cast<const float *>(data_device),
            row_count,
            n,
            row_stride,
            top_k,
            top_p,
            temperature,
            static_cast<int *>(out_token_ids_device),
            out_stride,
            static_cast<float *>(out_probs_device),
            static_cast<float *>(scratch_values_device),
            static_cast<int *>(scratch_indices_device),
            scratch_capacity,
            static_cast<const int *>(active_rows_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueBuildTopKTopPProcessedLogitsF32Device(
        const void *data_device,
        int row_count,
        int n,
        int row_stride,
        int top_k,
        float top_p,
        float temperature,
        int device_id,
        void *stream,
        void *out_logits_device,
        int out_row_stride,
        void *scratch_values_device,
        void *scratch_indices_device,
        int scratch_capacity)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !data_device || row_count <= 0 || n <= 0 ||
            row_stride < n || out_row_stride < n ||
            top_k <= 0 || !stream || !out_logits_device)
        {
            return false;
        }

        if (top_k > 256)
            top_k = 256;
        if (top_k > n)
            top_k = n;

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_topk_topp_processed_logits_f32(
            static_cast<const float *>(data_device),
            row_count,
            n,
            row_stride,
            top_k,
            top_p,
            temperature,
            static_cast<float *>(out_logits_device),
            out_row_stride,
            static_cast<float *>(scratch_values_device),
            static_cast<int *>(scratch_indices_device),
            scratch_capacity,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyDistributionsF32Device(
        const void *target_token_ids_device,
        const void *target_probs_device,
        const void *draft_token_ids_device,
        const void *draft_probs_device,
        int top_k,
        int draft_token,
        uint64_t accept_seed,
        uint64_t accept_offset,
        uint64_t residual_seed,
        uint64_t residual_offset,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            !draft_token_ids_device || !draft_probs_device ||
            top_k <= 0 || top_k > 256 || !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_distribution_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            draft_token,
            static_cast<unsigned long long>(accept_seed),
            static_cast<unsigned long long>(accept_offset),
            static_cast<unsigned long long>(residual_seed),
            static_cast<unsigned long long>(residual_offset),
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSampleDistributionF32Device(
        const void *token_ids_device,
        const void *probs_device,
        int top_k,
        float threshold,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_probability_device,
        uint64_t threshold_seed,
        const void *threshold_position_device,
        int threshold_position_offset)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !token_ids_device || !probs_device ||
            top_k <= 0 || top_k > 256 || !stream || !out_token_device ||
            (threshold_position_device && threshold_seed == 0))
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_sample_distribution_f32(
            static_cast<const int *>(token_ids_device),
            static_cast<const float *>(probs_device),
            top_k,
            threshold,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            static_cast<unsigned long long>(threshold_seed),
            static_cast<const int *>(threshold_position_device),
            threshold_position_offset,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSampleProcessedLogitsF32Device(
        const void *logits_device,
        int vocab_size,
        int row_stride,
        float threshold,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_probability_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            !stream || !out_token_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_sample_processed_logits_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            threshold,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
        const void *logits_device,
        int vocab_size,
        int row_stride,
        float threshold,
        const void *verify_tokens_device,
        const void *verify_accepted_device,
        int row_count,
        int first_token,
        const void *first_token_device,
        const int *stop_tokens_host,
        int stop_token_count,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_probability_device,
        uint64_t threshold_seed,
        const void *threshold_position_device,
        int threshold_position_offset)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            !verify_tokens_device || !verify_accepted_device ||
            row_count < 0 ||
            (first_token < 0 && !first_token_device) ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            !stream || !out_token_device ||
            (threshold_position_device && threshold_seed == 0))
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            threshold,
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            row_count,
            first_token,
            static_cast<const int *>(first_token_device),
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            static_cast<unsigned long long>(threshold_seed),
            static_cast<const int *>(threshold_position_device),
            threshold_position_offset,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
        const void *logits_device,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        int device_id,
        void *stream,
        void *out_probabilities_device,
        int out_row_stride,
        void *out_token_device,
        void *out_probability_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            out_row_stride < vocab_size || !stream ||
            !out_probabilities_device || !out_token_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_softmax_sample_temperature_logits_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            temperature,
            threshold,
            static_cast<float *>(out_probabilities_device),
            out_row_stride,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueScaleAndSampleTemperatureLogitsF32Device(
        const void *logits_device,
        int vocab_size,
        int row_stride,
        float temperature,
        float threshold,
        int device_id,
        void *stream,
        void *out_logits_device,
        int out_row_stride,
        void *out_token_device,
        void *out_probability_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || vocab_size <= 0 || row_stride < vocab_size ||
            out_row_stride < vocab_size || !stream ||
            !out_logits_device || !out_token_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_scale_sample_temperature_logits_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            temperature,
            threshold,
            static_cast<float *>(out_logits_device),
            out_row_stride,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSoftmaxProcessedLogitsF32Device(
        const void *logits_device,
        int row_count,
        int vocab_size,
        int row_stride,
        int device_id,
        void *stream,
        void *out_probabilities_device,
        int out_row_stride)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !logits_device || row_count <= 0 || vocab_size <= 0 ||
            row_stride < vocab_size || out_row_stride < vocab_size ||
            !stream || !out_probabilities_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_softmax_processed_logits_f32(
            static_cast<const float *>(logits_device),
            row_count,
            vocab_size,
            row_stride,
            static_cast<float *>(out_probabilities_device),
            out_row_stride,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueFillInverseExponentialSamplesF32Device(
        void *out_samples_device,
        int row_count,
        int vocab_size,
        int row_stride,
        uint64_t seed,
        int first_logical_position,
        int device_id,
        void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !out_samples_device || row_count <= 0 ||
            vocab_size <= 0 || row_stride < vocab_size || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_fill_inverse_exponential_samples_f32(
            static_cast<float *>(out_samples_device),
            row_count,
            vocab_size,
            row_stride,
            static_cast<unsigned long long>(seed),
            first_logical_position,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
        const void *target_token_ids_device,
        const void *target_probs_device,
        const void *draft_token_ids_device,
        const void *draft_probs_device,
        int top_k,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            !draft_token_ids_device || !draft_probs_device ||
            top_k <= 0 || top_k > 256 || !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_distribution_threshold_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            draft_token,
            accept_threshold,
            residual_threshold,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
        const void *target_token_ids_device,
        const void *target_probs_device,
        const void *draft_token_ids_device,
        const void *draft_probs_device,
        int top_k,
        int distribution_stride,
        const int *draft_tokens_host,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int row_count,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            !draft_token_ids_device || !draft_probs_device ||
            top_k <= 0 || top_k > 256 ||
            distribution_stride < top_k ||
            row_count <= 0 ||
            !draft_tokens_host || !accept_thresholds_host ||
            !residual_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_distribution_thresholds_batch_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            distribution_stride,
            draft_tokens_host,
            accept_thresholds_host,
            residual_thresholds_host,
            row_count,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
        const void *target_token_ids_device,
        const void *target_probs_device,
        const void *draft_token_ids_device,
        const void *draft_probs_device,
        int top_k,
        int distribution_stride,
        const void *draft_tokens_device,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int row_count,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        const void *draft_token_probabilities_device,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int inverse_sample_vocab_size,
        const void *threshold_base_position_device,
        int threshold_position_offset)
    {
        const bool has_draft_distribution =
            draft_token_ids_device != nullptr && draft_probs_device != nullptr;
        const bool has_one_hot_draft_distribution =
            draft_token_ids_device == nullptr && draft_probs_device == nullptr;
        const bool has_host_thresholds =
            accept_thresholds_host != nullptr &&
            residual_thresholds_host != nullptr;
        const int threshold_position_source_count =
            (inverse_sample_first_logical_position >= 0 ? 1 : 0) +
            (threshold_base_position_device != nullptr ? 1 : 0);
        const bool uses_seeded_device_thresholds =
            accept_thresholds_host == nullptr &&
            residual_thresholds_host == nullptr &&
            has_one_hot_draft_distribution &&
            inverse_sample_seed != 0 &&
            threshold_position_source_count == 1;
        if (device_id >= device_count_ || device_id < 0 ||
            !target_token_ids_device || !target_probs_device ||
            (!has_draft_distribution && !has_one_hot_draft_distribution) ||
            !draft_tokens_device ||
            top_k <= 0 || top_k > 256 ||
            distribution_stride < top_k ||
            row_count <= 0 ||
            (!has_host_thresholds && !uses_seeded_device_thresholds) ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            static_cast<const int *>(draft_token_ids_device),
            static_cast<const float *>(draft_probs_device),
            top_k,
            distribution_stride,
            static_cast<const int *>(draft_tokens_device),
            static_cast<const float *>(draft_token_probabilities_device),
            accept_thresholds_host,
            residual_thresholds_host,
            row_count,
            inverse_sample_seed,
            inverse_sample_first_logical_position,
            inverse_sample_vocab_size,
            uses_seeded_device_thresholds ? inverse_sample_seed : 0ull,
            inverse_sample_first_logical_position,
            uses_seeded_device_thresholds ? 1 : 0,
            static_cast<const int *>(threshold_base_position_device),
            threshold_position_offset,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
        const void *target_logits_device,
        const void *draft_logits_device,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const void *draft_tokens_device,
        const float *accept_thresholds_host,
        const float *residual_thresholds_host,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        const void *draft_token_probabilities_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_logits_device || !draft_logits_device ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size ||
            !accept_thresholds_host || !residual_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_logits_device),
            static_cast<const float *>(draft_logits_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            static_cast<const int *>(draft_tokens_device),
            accept_thresholds_host,
            residual_thresholds_host,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            static_cast<const float *>(draft_token_probabilities_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
        const void *target_logits_device,
        const void *draft_probabilities_device,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const void *draft_tokens_device,
        const float *accept_thresholds_host,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        bool no_draft_probabilities,
        const void *threshold_base_position_device,
        int threshold_position_offset)
    {
        const bool has_host_thresholds = accept_thresholds_host != nullptr;
        const int threshold_position_source_count =
            (inverse_sample_first_logical_position >= 0 ? 1 : 0) +
            (threshold_base_position_device != nullptr ? 1 : 0);
        const bool uses_seeded_device_thresholds =
            !has_host_thresholds &&
            inverse_sample_seed != 0 &&
            threshold_position_source_count == 1;
        if (device_id >= device_count_ || device_id < 0 ||
            !target_logits_device ||
            (!no_draft_probabilities && !draft_probabilities_device) ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size) ||
            (!has_host_thresholds && !uses_seeded_device_thresholds) ||
            (has_host_thresholds && threshold_base_position_device) ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_logits_device),
            static_cast<const float *>(draft_probabilities_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            static_cast<const int *>(draft_tokens_device),
            accept_thresholds_host,
            static_cast<unsigned long long>(inverse_sample_seed),
            inverse_sample_first_logical_position,
            uses_seeded_device_thresholds ? 1 : 0,
            static_cast<const int *>(threshold_base_position_device),
            threshold_position_offset,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            no_draft_probabilities ? 1 : 0,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
        const void *target_logits_device,
        const void *draft_logits_device,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const void *draft_tokens_device,
        const float *accept_thresholds_host,
        uint64_t inverse_sample_seed,
        int inverse_sample_first_logical_position,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        const void *draft_token_probabilities_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_logits_device || !draft_logits_device ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size ||
            !accept_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_logits_device),
            static_cast<const float *>(draft_logits_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            static_cast<const int *>(draft_tokens_device),
            static_cast<const float *>(draft_token_probabilities_device),
            accept_thresholds_host,
            static_cast<unsigned long long>(inverse_sample_seed),
            inverse_sample_first_logical_position,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
        const void *target_probabilities_device,
        const void *draft_probabilities_device,
        const void *inverse_rejection_samples_device,
        int row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        int inverse_sample_row_stride,
        const void *draft_tokens_device,
        const float *accept_thresholds_host,
        int device_id,
        void *stream,
        void *out_token_device,
        void *out_accepted_device,
        void *out_accept_probability_device,
        void *out_accept_threshold_device,
        bool no_draft_probabilities)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !target_probabilities_device || !inverse_rejection_samples_device ||
            (!no_draft_probabilities && !draft_probabilities_device) ||
            !draft_tokens_device ||
            row_count <= 0 ||
            vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            (!no_draft_probabilities && draft_row_stride < vocab_size) ||
            inverse_sample_row_stride < vocab_size ||
            !accept_thresholds_host ||
            !stream || !out_token_device || !out_accepted_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
            static_cast<const float *>(target_probabilities_device),
            static_cast<const float *>(draft_probabilities_device),
            static_cast<const float *>(inverse_rejection_samples_device),
            row_count,
            vocab_size,
            target_row_stride,
            draft_row_stride,
            inverse_sample_row_stride,
            static_cast<const int *>(draft_tokens_device),
            accept_thresholds_host,
            no_draft_probabilities ? 1 : 0,
            static_cast<int *>(out_token_device),
            static_cast<int *>(out_accepted_device),
            static_cast<float *>(out_accept_probability_device),
            static_cast<float *>(out_accept_threshold_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSummarizeSpeculativeVerifyBatch(
        const void *verify_tokens_device,
        const void *verify_accepted_device,
        int row_count,
        int first_token,
        const int *stop_tokens_host,
        int stop_token_count,
        const void *bonus_token_device,
        bool has_bonus_token,
        int device_id,
        void *stream,
        int out_token_capacity,
        void *out_tokens_device,
        void *out_meta_device,
        const void *max_state_commit_rows_device,
        int leading_committed_output_count)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !verify_accepted_device ||
            row_count < 0 || out_token_capacity < row_count + 1 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            (has_bonus_token && !bonus_token_device) ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_summarize_speculative_verify_batch(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            row_count,
            first_token,
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<const int *>(bonus_token_device),
            has_bonus_token ? 1 : 0,
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            leading_committed_output_count,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
        const void *verify_tokens_device,
        const void *verify_accepted_device,
        int row_count,
        const void *first_token_device,
        const int *stop_tokens_host,
        int stop_token_count,
        const void *bonus_token_device,
        bool has_bonus_token,
        int device_id,
        void *stream,
        int out_token_capacity,
        void *out_tokens_device,
        void *out_meta_device,
        const void *max_state_commit_rows_device,
        int leading_committed_output_count)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !verify_accepted_device ||
            !first_token_device ||
            row_count < 0 || out_token_capacity < row_count + 1 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            (has_bonus_token && !bonus_token_device) ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_summarize_speculative_verify_batch_device_first_token(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            row_count,
            static_cast<const int *>(first_token_device),
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<const int *>(bonus_token_device),
            has_bonus_token ? 1 : 0,
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            leading_committed_output_count,
            device_id,
            stream);
    }

    bool CUDABackend::
        enqueueSummarizeSpeculativeVerifyBatchDeviceGenerationControls(
            const void *verify_tokens_device,
            const void *verify_accepted_device,
            const void *greedy_draft_tokens_device,
            int row_count,
            const void *first_token_device,
            const void *stop_tokens_device,
            const void *bonus_token_device,
            bool has_bonus_token,
            const void *generation_control_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device)
    {
        const bool has_acceptance_rows = verify_accepted_device != nullptr;
        const bool has_greedy_rows = greedy_draft_tokens_device != nullptr;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || has_acceptance_rows == has_greedy_rows ||
            row_count < 0 || !first_token_device || !stop_tokens_device ||
            (has_bonus_token && !bonus_token_device) ||
            !generation_control_device ||
            out_token_capacity < row_count + 1 || !stream ||
            !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_summarize_speculative_verify_batch_device_generation_controls(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(verify_accepted_device),
            static_cast<const int *>(greedy_draft_tokens_device),
            row_count,
            static_cast<const int *>(first_token_device),
            static_cast<const int *>(stop_tokens_device),
            static_cast<const int *>(bonus_token_device),
            has_bonus_token ? 1 : 0,
            static_cast<const int *>(generation_control_device),
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            device_id,
            stream);
    }

    bool CUDABackend::
        enqueueSampleAndSummarizeSerialEquivalentSpeculativeBatchDeviceGenerationControls(
            const void *target_token_ids_device,
            const void *target_probs_device,
            int target_row_stride,
            int top_k,
            int row_count,
            uint64_t threshold_seed,
            const void *threshold_position_device,
            int threshold_position_offset,
            const void *verifier_input_tokens_device,
            const void *stop_tokens_device,
            const void *generation_control_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *sampled_target_tokens_device,
            void *out_tokens_device,
            void *out_meta_device,
            void *first_transaction_diagnostic_device)
    {
        if (device_id >= device_count_ || device_id < 0)
            return false;

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
            static_cast<const int *>(target_token_ids_device),
            static_cast<const float *>(target_probs_device),
            target_row_stride,
            top_k,
            row_count,
            static_cast<unsigned long long>(threshold_seed),
            static_cast<const int *>(threshold_position_device),
            threshold_position_offset,
            static_cast<const int *>(verifier_input_tokens_device),
            static_cast<const int *>(stop_tokens_device),
            static_cast<const int *>(generation_control_device),
            static_cast<int *>(sampled_target_tokens_device),
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            first_transaction_diagnostic_device,
            device_id,
            stream);
    }

    bool CUDABackend::enqueueSummarizeGreedySpeculativeVerifyBatch(
        const void *verify_tokens_device,
        const void *draft_tokens_device,
        int compare_row_count,
        int first_token,
        const int *stop_tokens_host,
        int stop_token_count,
        int device_id,
        void *stream,
        int out_token_capacity,
        void *out_tokens_device,
        void *out_meta_device,
        const void *max_state_commit_rows_device,
        int leading_committed_output_count)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !draft_tokens_device ||
            compare_row_count < 0 ||
            out_token_capacity < compare_row_count + 1 ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens ||
            (stop_token_count > 0 && !stop_tokens_host) ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        int stop_tokens[kSpeculativeBatchMaxStopTokens] =
            {-1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < stop_token_count; ++i)
            stop_tokens[i] = stop_tokens_host[i];

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_summarize_greedy_speculative_verify_batch(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(draft_tokens_device),
            compare_row_count,
            first_token,
            stop_tokens[0],
            stop_tokens[1],
            stop_tokens[2],
            stop_tokens[3],
            stop_tokens[4],
            stop_tokens[5],
            stop_tokens[6],
            stop_tokens[7],
            stop_token_count,
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            leading_committed_output_count,
            device_id,
            stream);
    }

    bool CUDABackend::
        enqueueSummarizeGreedySpeculativeVerifyBatchDeviceControls(
            const void *verify_tokens_device,
            const void *draft_tokens_device,
            int compare_row_count,
            const void *active_verifier_row_count_device,
            const void *stop_tokens_device,
            int device_id,
            void *stream,
            int out_token_capacity,
            void *out_tokens_device,
            void *out_meta_device,
            const void *max_state_commit_rows_device,
            const void *penalty_policy_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !draft_tokens_device ||
            !active_verifier_row_count_device || !stop_tokens_device ||
            !penalty_policy_device || compare_row_count < 0 ||
            out_token_capacity < compare_row_count + 1 ||
            !stream || !out_tokens_device || !out_meta_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_summarize_greedy_speculative_verify_batch_device_controls(
            static_cast<const int *>(verify_tokens_device),
            static_cast<const int *>(draft_tokens_device),
            compare_row_count,
            static_cast<const int *>(active_verifier_row_count_device),
            static_cast<const int *>(stop_tokens_device),
            static_cast<int *>(out_tokens_device),
            out_token_capacity,
            static_cast<int *>(out_meta_device),
            static_cast<const uint32_t *>(max_state_commit_rows_device),
            static_cast<const MTPGreedyPenaltyPolicy *>(penalty_policy_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueAdvanceSpeculativeCommitBoundary(
        void *meta_device,
        int request_count,
        int meta_stride,
        void *decode_rounds_committed_device,
        void *decode_rounds_until_maintenance_device,
        void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ || !meta_device ||
            request_count <= 0 ||
            meta_stride < sampling_math::kSpeculativeBatchMetaCount ||
            !decode_rounds_committed_device ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device ||
            !decode_boundary_advanced_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_advance_speculative_commit_boundary(
            static_cast<int *>(meta_device),
            request_count,
            meta_stride,
            static_cast<uint32_t *>(decode_rounds_committed_device),
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueInitializeDeviceGeneration(
        int request_count,
        int max_new_tokens,
        const sampling_math::DeviceGenerationDepthPolicy &depth_policy,
        int response_token_stride,
        void *response_tokens_device,
        int control_stride,
        void *control_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            request_count <= 0 || max_new_tokens <= 0 ||
            !depth_policy.valid() ||
            response_token_stride < max_new_tokens ||
            !response_tokens_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            !control_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_initialize_device_generation(
            request_count,
            max_new_tokens,
            depth_policy,
            response_token_stride,
            control_stride,
            static_cast<int *>(control_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePrepareDeviceGenerationTransactionBudget(
        void *control_device,
        int control_stride,
        int request_count,
        int verifier_row_capacity,
        const void *maintenance_rows_remaining_device,
        const void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        const bool has_maintenance_boundary =
            maintenance_rows_remaining_device != nullptr ||
            maintenance_due_device != nullptr ||
            decode_boundary_advanced_device != nullptr;
        const bool has_complete_maintenance_boundary =
            maintenance_rows_remaining_device != nullptr &&
            maintenance_due_device != nullptr &&
            decode_boundary_advanced_device != nullptr;
        if (device_id < 0 || device_id >= device_count_ || !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || verifier_row_capacity <= 0 || !stream ||
            (has_maintenance_boundary &&
             !has_complete_maintenance_boundary))
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_prepare_device_generation_transaction_budget(
            static_cast<int *>(control_device),
            control_stride,
            request_count,
            verifier_row_capacity,
            static_cast<const uint32_t *>(maintenance_rows_remaining_device),
            static_cast<const uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool CUDABackend::
        enqueueCommitDeviceGenerationAndDeriveSpeculativePublicationMetadata(
        const void *output_tokens_device,
        int output_token_stride,
        void *meta_device,
        int meta_stride,
        const void *base_cached_tokens_device,
        int request_count,
        int padded_state_rows_per_request,
        void *response_tokens_device,
        int response_token_stride,
        void *control_device,
        int control_stride,
        int device_id,
        void *stream,
        void *out_restore_rows_device,
        void *out_target_cached_tokens_device,
        void *out_accepted_state_counts_device,
        void *out_ok_device,
        void *out_next_condition_tokens_device,
        void *out_all_drafts_accepted_flags_device,
        void *out_stopped_flags_device,
        void *out_next_sidecar_condition_tokens_device,
        void *out_next_sidecar_position_ids_device,
        void *out_next_verifier_condition_tokens_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !output_tokens_device || output_token_stride <= 0 ||
            !meta_device || !base_cached_tokens_device ||
            meta_stride < sampling_math::kSpeculativeBatchMetaCount ||
            request_count <= 0 || padded_state_rows_per_request <= 0 ||
            !response_tokens_device ||
            response_token_stride <= 0 || !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            !out_restore_rows_device || !out_target_cached_tokens_device ||
            !out_accepted_state_counts_device || !out_ok_device ||
            ((out_next_sidecar_condition_tokens_device ||
              out_next_sidecar_position_ids_device) &&
             (!out_next_sidecar_condition_tokens_device ||
              !out_next_sidecar_position_ids_device ||
              !out_next_condition_tokens_device)) ||
            !out_next_verifier_condition_tokens_device ||
            !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_commit_device_generation_and_derive_speculative_publication_metadata(
            static_cast<const int32_t *>(output_tokens_device),
            output_token_stride,
            static_cast<int *>(meta_device),
            meta_stride,
            static_cast<const int *>(base_cached_tokens_device),
            request_count,
            padded_state_rows_per_request,
            static_cast<int32_t *>(response_tokens_device),
            response_token_stride,
            static_cast<int *>(control_device),
            control_stride,
            static_cast<int *>(out_restore_rows_device),
            static_cast<int *>(out_target_cached_tokens_device),
            static_cast<int *>(out_accepted_state_counts_device),
            static_cast<int *>(out_ok_device),
            static_cast<int *>(out_next_condition_tokens_device),
            static_cast<int *>(out_all_drafts_accepted_flags_device),
            static_cast<int *>(out_stopped_flags_device),
            static_cast<int *>(out_next_sidecar_condition_tokens_device),
            static_cast<int *>(out_next_sidecar_position_ids_device),
            static_cast<int *>(out_next_verifier_condition_tokens_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueDeriveSpeculativePublicationMetadata(
        const void *meta_device,
        int meta_stride,
        const void *base_cached_tokens_device,
        int request_count,
        int padded_state_rows_per_request,
        int max_state_commit_rows,
        int device_id,
        void *stream,
        void *out_restore_rows_device,
        void *out_target_cached_tokens_device,
        void *out_accepted_state_counts_device,
        void *out_ok_device,
        void *out_next_condition_tokens_device,
        const void *output_tokens_device,
        int output_token_stride,
        void *out_all_drafts_accepted_flags_device,
        void *out_stopped_flags_device,
        void *out_next_verifier_condition_tokens_device)
    {
        using namespace sampling_math;
        if (device_id >= device_count_ || device_id < 0 ||
            !meta_device || !base_cached_tokens_device ||
            meta_stride < kSpeculativeBatchMetaCount ||
            request_count <= 0 ||
            padded_state_rows_per_request <= 0 ||
            max_state_commit_rows < 0 ||
            max_state_commit_rows > padded_state_rows_per_request ||
            !stream ||
            !out_restore_rows_device ||
            !out_target_cached_tokens_device ||
            !out_accepted_state_counts_device ||
            !out_ok_device ||
            ((out_next_condition_tokens_device || output_tokens_device) &&
             (!out_next_condition_tokens_device ||
              !output_tokens_device ||
              output_token_stride <= 0)))
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_derive_speculative_publication_metadata(
            static_cast<const int *>(meta_device),
            meta_stride,
            static_cast<const int *>(base_cached_tokens_device),
            request_count,
            padded_state_rows_per_request,
            max_state_commit_rows,
            static_cast<int *>(out_restore_rows_device),
            static_cast<int *>(out_target_cached_tokens_device),
            static_cast<int *>(out_accepted_state_counts_device),
            static_cast<int *>(out_ok_device),
            static_cast<int *>(out_next_condition_tokens_device),
            static_cast<const int32_t *>(output_tokens_device),
            output_token_stride,
            static_cast<int *>(out_all_drafts_accepted_flags_device),
            static_cast<int *>(out_stopped_flags_device),
            static_cast<int *>(out_next_verifier_condition_tokens_device),
            device_id,
            stream);
    }

    bool CUDABackend::
        enqueueDeriveShiftedSpeculativePublicationMetadataFromPrimary(
        const void *base_cached_tokens_device,
        const void *main_target_cached_tokens_device,
        const void *main_publication_ok_device,
        int request_count,
        int mtp_depth,
        int device_id,
        void *stream,
        void *out_target_cached_tokens_device,
        void *out_accepted_state_counts_device,
        void *out_ok_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !base_cached_tokens_device ||
            !main_target_cached_tokens_device ||
            !main_publication_ok_device ||
            request_count <= 0 ||
            mtp_depth < 0 ||
            !stream ||
            !out_target_cached_tokens_device ||
            !out_accepted_state_counts_device ||
            !out_ok_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_derive_shifted_speculative_publication_metadata_from_primary(
            static_cast<const int *>(base_cached_tokens_device),
            static_cast<const int *>(main_target_cached_tokens_device),
            static_cast<const int *>(main_publication_ok_device),
            request_count,
            mtp_depth,
            static_cast<int *>(out_target_cached_tokens_device),
            static_cast<int *>(out_accepted_state_counts_device),
            static_cast<int *>(out_ok_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePrepareSpeculativeShiftedKVTokens(
        const void *meta_device,
        int meta_stride,
        const void *output_tokens_device,
        int output_token_stride,
        int request_index,
        int first_output_token_index,
        int row_count,
        int32_t filler_token,
        int device_id,
        void *stream,
        void *out_tokens_device,
        const void *base_positions_device,
        int position_offset,
        void *out_position_ids_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !meta_device ||
            !output_tokens_device ||
            !out_tokens_device ||
            !base_positions_device ||
            !out_position_ids_device ||
            !stream ||
            meta_stride < sampling_math::kSpeculativeBatchMetaCount ||
            output_token_stride <= first_output_token_index ||
            request_index < 0 ||
            first_output_token_index < 0 ||
            row_count <= 0)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_prepare_speculative_shifted_kv_tokens(
            static_cast<const int *>(meta_device),
            meta_stride,
            static_cast<const int32_t *>(output_tokens_device),
            output_token_stride,
            request_index,
            first_output_token_index,
            row_count,
            filler_token,
            static_cast<int32_t *>(out_tokens_device),
            static_cast<const int32_t *>(base_positions_device),
            position_offset,
            static_cast<int32_t *>(out_position_ids_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePrepareMTPBatchedSidecarInputs(
        const void *condition_tokens_device,
        int condition_token_stride,
        const void *base_positions_device,
        int position_offset,
        int request_count,
        int device_id,
        void *stream,
        void *out_condition_tokens_device,
        void *out_position_ids_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !condition_tokens_device ||
            condition_token_stride <= 0 ||
            !base_positions_device ||
            request_count <= 0 ||
            !stream ||
            !out_condition_tokens_device ||
            !out_position_ids_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_prepare_mtp_batched_sidecar_inputs(
            static_cast<const int32_t *>(condition_tokens_device),
            condition_token_stride,
            static_cast<const int32_t *>(base_positions_device),
            position_offset,
            request_count,
            static_cast<int32_t *>(out_condition_tokens_device),
            static_cast<int32_t *>(out_position_ids_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePrepareMTPVerifierPositionIds(
        const void *base_positions_device,
        int request_count,
        int padded_seq_len,
        int device_id,
        void *stream,
        void *out_position_ids_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !base_positions_device ||
            request_count <= 0 ||
            padded_seq_len <= 0 ||
            !stream ||
            !out_position_ids_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_prepare_mtp_verifier_position_ids(
            static_cast<const int32_t *>(base_positions_device),
            request_count,
            padded_seq_len,
            static_cast<int32_t *>(out_position_ids_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePrepareMTPVerifierGeometry(
        const void *base_positions_device,
        const void *valid_graph_rows_device,
        int valid_graph_row_count,
        void *generation_control_device,
        int generation_control_stride,
        int request_count,
        int padded_seq_len,
        int device_id,
        void *stream,
        void *out_position_ids_device,
        void *out_request_lengths_device)
    {
        const bool has_generation_control =
            generation_control_device != nullptr;
        if (device_id < 0 || device_id >= device_count_ ||
            !base_positions_device || request_count <= 0 ||
            padded_seq_len <= 0 || !stream || !out_position_ids_device ||
            !out_request_lengths_device ||
            has_generation_control != (generation_control_stride > 0) ||
            (has_generation_control &&
             generation_control_stride <
                 sampling_math::kDeviceGenerationControlCount))
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_prepare_mtp_verifier_geometry(
            static_cast<const int32_t *>(base_positions_device),
            static_cast<const int32_t *>(valid_graph_rows_device),
            valid_graph_row_count,
            static_cast<int *>(generation_control_device),
            generation_control_stride,
            request_count,
            padded_seq_len,
            static_cast<int32_t *>(out_position_ids_device),
            static_cast<int32_t *>(out_request_lengths_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePrepareMTPVerifierControlledRow(
        const void *first_token_device,
        const void *draft_tokens_device,
        const void *base_position_device,
        void *generation_control_row_device,
        int generation_control_stride,
        int padded_seq_len,
        int device_id,
        void *stream,
        void *out_tokens_device,
        void *out_position_ids_device,
        void *out_request_length_device,
        void *out_base_position_snapshot_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !first_token_device || !draft_tokens_device ||
            !base_position_device || !generation_control_row_device ||
            generation_control_stride <
                sampling_math::kDeviceGenerationControlCount ||
            padded_seq_len <= 1 || !stream || !out_tokens_device ||
            !out_position_ids_device || !out_request_length_device ||
            !out_base_position_snapshot_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_prepare_mtp_verifier_controlled_row(
            static_cast<const int32_t *>(first_token_device),
            static_cast<const int32_t *>(draft_tokens_device),
            static_cast<const int32_t *>(base_position_device),
            static_cast<int *>(generation_control_row_device),
            generation_control_stride,
            padded_seq_len,
            static_cast<int32_t *>(out_tokens_device),
            static_cast<int32_t *>(out_position_ids_device),
            static_cast<int32_t *>(out_request_length_device),
            static_cast<int32_t *>(out_base_position_snapshot_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueInitializeMTPDeviceLogicalState(
        const void *sampled_tokens_device,
        const void *target_positions_device,
        int request_count,
        int device_id,
        void *stream,
        void *out_base_cached_tokens_device,
        void *out_target_positions_device,
        void *out_accepted_state_counts_device,
        void *out_next_condition_tokens_device,
        void *out_all_drafts_accepted_flags_device,
        void *out_stopped_flags_device,
        void *out_publication_ok_flags_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !sampled_tokens_device ||
            !target_positions_device ||
            request_count <= 0 ||
            !stream ||
            !out_base_cached_tokens_device ||
            !out_target_positions_device ||
            !out_accepted_state_counts_device ||
            !out_next_condition_tokens_device ||
            !out_all_drafts_accepted_flags_device ||
            !out_stopped_flags_device ||
            !out_publication_ok_flags_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_initialize_mtp_device_logical_state(
            static_cast<const int32_t *>(sampled_tokens_device),
            static_cast<const int32_t *>(target_positions_device),
            request_count,
            static_cast<int32_t *>(out_base_cached_tokens_device),
            static_cast<int32_t *>(out_target_positions_device),
            static_cast<int32_t *>(out_accepted_state_counts_device),
            static_cast<int32_t *>(out_next_condition_tokens_device),
            static_cast<int32_t *>(out_all_drafts_accepted_flags_device),
            static_cast<int32_t *>(out_stopped_flags_device),
            static_cast<int32_t *>(out_publication_ok_flags_device),
            device_id,
            stream);
    }

    // Forward declaration for CUDA penalty kernel
    extern "C" bool cudaOps_apply_logit_penalties_f32(
        float *logits, const int *token_ids, const float *penalties,
        int num_penalties, int vocab_size, int device_idx, void *stream);

    bool CUDABackend::prepareLogitPenaltyWorkspace(
        int vocab_size,
        int device_id)
    {
        if (device_id < 0 ||
            device_id >= device_count_ ||
            vocab_size <= 0 ||
            static_cast<size_t>(device_id) >= penalty_buffers_.size())
        {
            return false;
        }

        auto &bufs = penalty_buffers_[static_cast<size_t>(device_id)];
        if (bufs.allocated_count >= vocab_size &&
            bufs.token_ids_ptr &&
            bufs.penalties_ptr &&
            bufs.ready_event)
        {
            return true;
        }
        if (bufs.allocated_count != 0 ||
            bufs.token_ids_ptr ||
            bufs.penalties_ptr ||
            bufs.ready_event)
        {
            LOG_ERROR("[CUDABackend] Refusing to resize an active logit-penalty workspace");
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        cudaError_t err =
            cudaMalloc(&bufs.token_ids_ptr, vocab_size * sizeof(int));
        if (err != cudaSuccess)
            return false;
        err = cudaMalloc(
            &bufs.penalties_ptr,
            vocab_size * sizeof(float));
        if (err != cudaSuccess)
        {
            CUDA_WARN_IF_FAIL(cudaFree(bufs.token_ids_ptr));
            bufs.token_ids_ptr = nullptr;
            return false;
        }
        cudaEvent_t ready_event = nullptr;
        err = cudaEventCreateWithFlags(
            &ready_event,
            cudaEventDisableTiming);
        if (err != cudaSuccess)
        {
            CUDA_WARN_IF_FAIL(cudaFree(bufs.penalties_ptr));
            CUDA_WARN_IF_FAIL(cudaFree(bufs.token_ids_ptr));
            bufs.penalties_ptr = nullptr;
            bufs.token_ids_ptr = nullptr;
            return false;
        }
        bufs.ready_event = ready_event;
        bufs.allocated_count = vocab_size;
        return true;
    }

    bool CUDABackend::applyLogitPenaltiesF32(void *logits_device,
                                              const int *token_ids_host,
                                              const float *penalties_host,
                                              int num_penalties, int vocab_size,
                                              int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 || !logits_device ||
            !token_ids_host || !penalties_host || num_penalties <= 0)
            return false;

        auto &bufs = penalty_buffers_[static_cast<size_t>(device_id)];
        if (num_penalties > bufs.allocated_count ||
            !bufs.token_ids_ptr ||
            !bufs.penalties_ptr ||
            !bufs.ready_event)
            return false;

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        cudaStream_t s =
            requireExplicitStream(stream, "CUDABackend::applyLogitPenaltiesF32");

        /*
         * The penalty slot may follow main, sidecar, or verifier logits. Queue
         * cross-stream ownership on device before overwriting the shared slot.
         */
        if (bufs.publication_valid &&
            bufs.producer_stream != stream)
        {
            CUDA_CHECK_OR_THROW(cudaStreamWaitEvent(
                s,
                static_cast<cudaEvent_t>(bufs.ready_event),
                0));
        }

        // Upload penalty data to device
        CUDA_CHECK_OR_THROW(cudaMemcpyAsync(bufs.token_ids_ptr, token_ids_host,
                                             num_penalties * sizeof(int),
                                             cudaMemcpyHostToDevice, s));
        CUDA_CHECK_OR_THROW(cudaMemcpyAsync(bufs.penalties_ptr, penalties_host,
                                             num_penalties * sizeof(float),
                                             cudaMemcpyHostToDevice, s));

        // Apply penalties in-place on device
        if (!cudaOps_apply_logit_penalties_f32(
                static_cast<float *>(logits_device),
                static_cast<const int *>(bufs.token_ids_ptr),
                static_cast<const float *>(bufs.penalties_ptr),
                num_penalties, vocab_size, device_id, s))
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaEventRecord(
            static_cast<cudaEvent_t>(bufs.ready_event),
            s));
        bufs.producer_stream = stream;
        bufs.publication_valid = true;
        return true;
    }

    bool CUDABackend::enqueueLogitPenaltiesF32Device(void *logits_device,
                                                     const void *token_ids_device,
                                                     const void *penalties_device,
                                                     int num_penalties,
                                                     int vocab_size,
                                                     int device_id,
                                                     void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 || !logits_device ||
            !token_ids_device || !penalties_device || num_penalties <= 0 ||
            vocab_size <= 0 || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_apply_logit_penalties_f32(
            static_cast<float *>(logits_device),
            static_cast<const int *>(token_ids_device),
            static_cast<const float *>(penalties_device),
            num_penalties,
            vocab_size,
            device_id,
            stream);
    }

    // ====================================================================
    // Stream Management
    // ====================================================================

    void *CUDABackend::createStream(int device_id)
    {
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::createStream] cudaSetDevice(" << device_id
                                                                   << ") failed: " << cudaGetErrorString(err));
            return nullptr;
        }

        cudaStream_t stream;
        err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::createStream] cudaStreamCreateWithFlags failed: "
                      << cudaGetErrorString(err));
            return nullptr;
        }
        return stream;
    }

    void CUDABackend::destroyStream(void *stream, int device_id)
    {
        if (!stream)
            return;
        CUDA_WARN_IF_FAIL(cudaSetDevice(device_id)); // cleanup path
        CUDA_WARN_IF_FAIL(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
    }

    bool CUDABackend::synchronizeStream(void *stream, int device_id)
    {
        cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::synchronizeStream");
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
            return false;
        err = cudaStreamSynchronize(cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::synchronizeStream] failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDABackend::streamWaitEvent(void *stream, void *event, int device_id)
    {
        cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::streamWaitEvent");
        if (!event || device_id < 0 || device_id >= device_count_)
        {
            LOG_ERROR("[CUDABackend::streamWaitEvent] invalid event-wait ownership"
                      << " device=" << device_id
                      << " stream=" << stream
                      << " event=" << event);
            return false;
        }

        /*
         * LocalTP orchestrators submit child resets serially from one host
         * thread. The ambient CUDA device therefore belongs to whichever child
         * ran most recently, not necessarily to this stream/event pair. Select
         * the caller-declared owner before touching either resource so a
         * cross-child reset cannot become cudaErrorInvalidResourceHandle.
         */
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamWaitEvent] cudaSetDevice("
                      << device_id << ") failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        err = cudaStreamWaitEvent(
            cuda_stream,
            static_cast<cudaEvent_t>(event), 0);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamWaitEvent] failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // ====================================================================
    // Async H2D Without Sync (Pipeline Support)
    // ====================================================================

    bool CUDABackend::hostToDeviceOnStream(void *dst, const void *src, size_t bytes,
                                           int device_id, void *stream)
    {
        cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::hostToDeviceOnStream");
        if (device_id >= device_count_ || device_id < 0)
            return false;
        if (!setDevice(device_id))
            return false;

        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                          cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::hostToDeviceOnStream] failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDABackend::deviceToHostOnStream(void *dst, const void *src, size_t bytes,
                                           int device_id, void *stream)
    {
        cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::deviceToHostOnStream");
        if (device_id >= device_count_ || device_id < 0)
            return false;
        if (!setDevice(device_id))
            return false;

        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost,
                                          cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::deviceToHostOnStream] failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // ====================================================================
    // Pinned Host Memory
    // ====================================================================

    void *CUDABackend::allocatePinned(size_t bytes, int device_id)
    {
        (void)device_id;
        void *ptr = nullptr;
        cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::allocatePinned] cudaHostAlloc(" << bytes
                      << ") failed: " << cudaGetErrorString(err));
            return nullptr;
        }
        return ptr;
    }

    void CUDABackend::freePinned(void *ptr, int device_id)
    {
        (void)device_id;
        if (ptr)
            CUDA_WARN_IF_FAIL(cudaFreeHost(ptr));
    }

    // ====================================================================
    // Stream-Aware Memory Operations
    // ====================================================================

    bool CUDABackend::deviceCopyAsync(void *dst, const void *src, size_t bytes,
                                      int device_id, void *stream)
    {
        cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::deviceCopyAsync");
        if (bytes == 0)
            return true;
        if (!dst || !src)
        {
            LOG_ERROR("[CUDABackend::deviceCopyAsync] null pointer for non-empty copy"
                      << " dst=" << dst << " src=" << src << " bytes=" << bytes);
            return false;
        }
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::deviceCopyAsync] cudaSetDevice(" << device_id
                                                                      << ") failed: " << cudaGetErrorString(err));
            return false;
        }
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                              cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::deviceCopyAsync] failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    // ====================================================================
    // Collective Reduction Primitives
    // ====================================================================

    bool CUDABackend::vectorAddInplace(void *output, const void *input, size_t count,
                                       int element_size, int device_id, void *stream)
    {
        cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::vectorAddInplace");
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::vectorAddInplace] cudaSetDevice failed: "
                      << cudaGetErrorString(err));
            return false;
        }
        switch (element_size)
        {
        case 4: // FP32 or INT32
            return cuda::launchVectorAddInplace_f32(
                static_cast<float *>(output),
                static_cast<const float *>(input),
                count, cuda_stream);

        case 2: // FP16 or BF16 — defaults to FP16
            return cuda::launchVectorAddInplace_f16(
                output, input, count, cuda_stream);

        case 1: // INT8
            return cuda::launchVectorAddInplace_i8(
                static_cast<int8_t *>(output),
                static_cast<const int8_t *>(input),
                count, cuda_stream);

        default:
            LOG_ERROR("[CUDABackend::vectorAddInplace] unsupported element_size: "
                      << element_size);
            return false;
        }
    }

} // namespace llaminar2
