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
#include "../GPUDeviceContextPool.h"
#include "NvidiaDeviceContext.h"
#include "../../utils/Logger.h"
#include "../../kernels/cuda/ops/CUDAVectorAddKernels.h"
#include <cuda_runtime.h>
#include <cuda.h> // For cuCtxSetCurrent, cuDevicePrimaryCtxRetain
#include <future>
#include <memory>
#include <stdexcept>
#include <sstream>

namespace llaminar2
{

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
    }

    CUDABackend::~CUDABackend()
    {
        // cudaDeviceReset() intentionally omitted - managed by CUDA runtime
    }

    // ====================================================================
    // Stream Resolution Helper
    // ====================================================================

    /// Resolve a CUDA stream for the given device.
    ///
    /// Returns the caller-provided stream if non-null, otherwise nullptr
    /// (legacy default stream). The pool's non-blocking default stream is
    /// NOT used as a fallback because non-blocking streams have different
    /// synchronization semantics: cudaFree and cudaHostUnregister do NOT
    /// implicitly synchronize with non-blocking streams, causing silent
    /// data corruption when tensors are destroyed and GPU memory is reused
    /// by subsequent operations (e.g., weight repack pipelines).
    ///
    /// Callers that need the pool's stream (e.g., DeviceGraphExecutor)
    /// should pass it explicitly via the stream parameter.
    static cudaStream_t resolveStream(int device_id, void *stream)
    {
        if (stream)
            return static_cast<cudaStream_t>(stream);
        return nullptr; // Use legacy default stream
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

        // Use setDevice() which handles both runtime and driver API context
        if (!setDevice(device_id))
        {
            return false;
        }

        cudaStream_t s = resolveStream(device_id, stream);
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

        cudaStream_t s = resolveStream(device_id, stream);
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
        cudaStream_t s = resolveStream(device_id, stream);
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
        cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream); // nullptr = default stream
        if (cuda_stream)
        {
            cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(cuda_stream, &capture_status) == cudaSuccess &&
                capture_status != cudaStreamCaptureStatusNone)
            {
                return true;
            }
        }

        err = cudaEventRecord(cuda_event, cuda_stream);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::recordEvent] cudaEventRecord failed: " << cudaGetErrorString(err));
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

        // First, set the runtime API device
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            return false;
        }

        // Also set the driver API context for cross-thread compatibility.
        // This is crucial when CUDA is called from a thread that primarily
        // uses another GPU runtime (e.g., HIP/ROCm). The cudaSetDevice alone
        // may not properly establish the context for driver API operations
        // or for events created on other threads.
        CUdevice cu_device;
        CUresult cu_err = cuDeviceGet(&cu_device, device_id);
        if (cu_err != CUDA_SUCCESS)
        {
            // Fall through - runtime API is set, driver API failed but may work
            LOG_WARN("[CUDABackend::setDevice] cuDeviceGet failed for device " << device_id << " error=" << cu_err);
            return true; // cudaSetDevice succeeded, return true
        }

        CUcontext ctx;
        cu_err = cuDevicePrimaryCtxRetain(&ctx, cu_device);
        if (cu_err != CUDA_SUCCESS)
        {
            LOG_WARN("[CUDABackend::setDevice] cuDevicePrimaryCtxRetain failed for device " << device_id << " error=" << cu_err);
            return true; // cudaSetDevice succeeded
        }

        cu_err = cuCtxSetCurrent(ctx);
        if (cu_err != CUDA_SUCCESS)
        {
            LOG_WARN("[CUDABackend::setDevice] cuCtxSetCurrent failed for device " << device_id << " error=" << cu_err);
        }
        else
        {
            LOG_DEBUG("[CUDABackend::setDevice] Successfully set context for device " << device_id);
        }

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

        LOG_TRACE("[CUDABackend::allocate] ALLOC ptr=" << ptr << " bytes=" << bytes << " device_id=" << device_id);
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

        err = cudaFree(ptr);
        if (err != cudaSuccess)
        {
            LOG_DEBUG("[CUDABackend] cudaFree failed for ptr=" << std::hex << ptr << std::dec
                                                               << " on device " << device_id << ": " << cudaGetErrorString(err));
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

        err = cudaMemsetAsync(ptr, value, bytes, resolveStream(device_id, stream));
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
        int device_idx, void *stream);

    extern "C" bool cudaOps_topk_f32(
        const float *data, int n, int k, float *out_values, int *out_indices,
        int device_idx, void *stream);
    extern "C" bool cudaOps_sample_topk_topp_f32(
        const float *data, int n, int k, float top_p, float temperature,
        unsigned long long rng_seed, unsigned long long rng_offset,
        int *out_token, int device_idx, void *stream);
    extern "C" bool cudaOps_topk_topp_distribution_f32(
        const float *data, int n, int k, float top_p, float temperature,
        int *out_token_ids, float *out_probs, int device_idx, void *stream);
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
        int *out_token, int device_idx, void *stream);
    extern "C" bool cudaOps_speculative_verify_distribution_threshold_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int draft_token,
        float accept_threshold, float residual_threshold,
        int *out_token, int *out_accepted,
        float *out_accept_probability, float *out_accept_threshold,
        int device_idx, void *stream);

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
        cudaStream_t s = resolveStream(device_id, stream);
        // Pass the caller-supplied partial scratch through to the kernel wrapper.
        // When partial_vals/partial_idxs are null (or capacity < 1), the wrapper
        // transparently falls back to a single-block reduction.
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
        cudaStream_t s = resolveStream(device_id, stream);
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
                s))
        {
            return false;
        }

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
        CUDA_CHECK_OR_THROW(cudaStreamSynchronize(s));
        return true;
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
        cudaStream_t s = resolveStream(device_id, stream);
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
        void *out_probs_device)
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
        void *out_token_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !token_ids_device || !probs_device ||
            top_k <= 0 || top_k > 256 || !stream || !out_token_device)
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

    // Forward declaration for CUDA penalty kernel
    extern "C" bool cudaOps_apply_logit_penalties_f32(
        float *logits, const int *token_ids, const float *penalties,
        int num_penalties, int vocab_size, int device_idx, void *stream);

    bool CUDABackend::applyLogitPenaltiesF32(void *logits_device,
                                              const int *token_ids_host,
                                              const float *penalties_host,
                                              int num_penalties, int vocab_size,
                                              int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 || !logits_device ||
            !token_ids_host || !penalties_host || num_penalties <= 0)
            return false;

        // Lazily allocate per-device penalty upload buffers
        if (penalty_buffers_.empty())
            penalty_buffers_.resize(device_count_);

        auto &bufs = penalty_buffers_[device_id];

        // Reallocate if num_penalties exceeds current allocation
        if (bufs.allocated_count < num_penalties)
        {
            CUDA_WARN_IF_FAIL(cudaSetDevice(device_id));
            if (bufs.token_ids_ptr)
                CUDA_WARN_IF_FAIL(cudaFree(bufs.token_ids_ptr));
            if (bufs.penalties_ptr)
                CUDA_WARN_IF_FAIL(cudaFree(bufs.penalties_ptr));

            cudaError_t err = cudaMalloc(&bufs.token_ids_ptr, num_penalties * sizeof(int));
            if (err != cudaSuccess)
            {
                bufs.token_ids_ptr = nullptr;
                bufs.allocated_count = 0;
                return false;
            }
            err = cudaMalloc(&bufs.penalties_ptr, num_penalties * sizeof(float));
            if (err != cudaSuccess)
            {
                CUDA_WARN_IF_FAIL(cudaFree(bufs.token_ids_ptr));
                bufs.token_ids_ptr = nullptr;
                bufs.allocated_count = 0;
                return false;
            }
            bufs.allocated_count = num_penalties;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        cudaStream_t s = resolveStream(device_id, stream);

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

        CUDA_CHECK_OR_THROW(cudaStreamSynchronize(s));
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
    // Async Operations (Route through NvidiaDeviceContext worker thread)
    // ====================================================================

    std::future<bool> CUDABackend::deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            std::promise<bool> p;
            p.set_value(false);
            return p.get_future();
        }

        try
        {
            NvidiaDeviceContext &ctx = static_cast<NvidiaDeviceContext &>(
                GPUDeviceContextPool::instance().getNvidiaContext(device_id));
            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();
            ctx.submitAsync([this, dst, src, bytes, device_id, promise]()
                            {
                cudaStream_t ws = resolveStream(device_id, nullptr);
                cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, ws);
                if (err == cudaSuccess) err = cudaStreamSynchronize(ws);
                promise->set_value(err == cudaSuccess); });
            return future;
        }
        catch (...)
        {
            // Fallback: execute synchronously
            std::promise<bool> p;
            p.set_value(deviceToHost(dst, src, bytes, device_id));
            return p.get_future();
        }
    }

    std::future<bool> CUDABackend::hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            std::promise<bool> p;
            p.set_value(false);
            return p.get_future();
        }

        try
        {
            NvidiaDeviceContext &ctx = static_cast<NvidiaDeviceContext &>(
                GPUDeviceContextPool::instance().getNvidiaContext(device_id));
            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();
            ctx.submitAsync([this, dst, src, bytes, device_id, promise]()
                            {
                cudaStream_t ws = resolveStream(device_id, nullptr);
                cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, ws);
                if (err == cudaSuccess) err = cudaStreamSynchronize(ws);
                promise->set_value(err == cudaSuccess); });
            return future;
        }
        catch (...)
        {
            // Fallback: execute synchronously
            std::promise<bool> p;
            p.set_value(hostToDevice(dst, src, bytes, device_id));
            return p.get_future();
        }
    }

    std::future<bool> CUDABackend::synchronizeAsync(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            std::promise<bool> p;
            p.set_value(false);
            return p.get_future();
        }

        try
        {
            NvidiaDeviceContext &ctx = static_cast<NvidiaDeviceContext &>(
                GPUDeviceContextPool::instance().getNvidiaContext(device_id));
            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();
            ctx.submitAsync([promise]()
                            {
                cudaError_t err = cudaDeviceSynchronize();
                promise->set_value(err == cudaSuccess); });
            return future;
        }
        catch (...)
        {
            // Fallback: execute synchronously
            std::promise<bool> p;
            p.set_value(synchronize(device_id));
            return p.get_future();
        }
    }

    std::future<void *> CUDABackend::allocateAsync(size_t bytes, int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            std::promise<void *> p;
            p.set_value(nullptr);
            return p.get_future();
        }

        try
        {
            NvidiaDeviceContext &ctx = static_cast<NvidiaDeviceContext &>(
                GPUDeviceContextPool::instance().getNvidiaContext(device_id));
            auto promise = std::make_shared<std::promise<void *>>();
            auto future = promise->get_future();
            ctx.submitAsync([bytes, promise]()
                            {
                void *ptr = nullptr;
                cudaError_t err = cudaMalloc(&ptr, bytes);
                promise->set_value(err == cudaSuccess ? ptr : nullptr); });
            return future;
        }
        catch (...)
        {
            // Fallback: execute synchronously
            std::promise<void *> p;
            p.set_value(allocate(bytes, device_id));
            return p.get_future();
        }
    }

    std::future<void> CUDABackend::freeAsync(void *ptr, int device_id)
    {
        if (ptr == nullptr || device_id >= device_count_ || device_id < 0)
        {
            std::promise<void> p;
            p.set_value();
            return p.get_future();
        }

        try
        {
            NvidiaDeviceContext &ctx = static_cast<NvidiaDeviceContext &>(
                GPUDeviceContextPool::instance().getNvidiaContext(device_id));
            auto promise = std::make_shared<std::promise<void>>();
            auto future = promise->get_future();
            ctx.submitAsync([ptr, promise]()
                            {
                CUDA_WARN_IF_FAIL(cudaFree(ptr)); // async free; no return path to caller
                promise->set_value(); });
            return future;
        }
        catch (...)
        {
            // Fallback: execute synchronously
            free(ptr, device_id);
            std::promise<void> p;
            p.set_value();
            return p.get_future();
        }
    }

    std::future<bool> CUDABackend::memsetAsync(void *ptr, int value, size_t bytes, int device_id)
    {
        if (ptr == nullptr || bytes == 0)
        {
            std::promise<bool> p;
            p.set_value(true);
            return p.get_future();
        }

        if (device_id >= device_count_ || device_id < 0)
        {
            std::promise<bool> p;
            p.set_value(false);
            return p.get_future();
        }

        try
        {
            NvidiaDeviceContext &ctx = static_cast<NvidiaDeviceContext &>(
                GPUDeviceContextPool::instance().getNvidiaContext(device_id));
            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();
            ctx.submitAsync([this, ptr, value, bytes, device_id, promise]()
                            {
                cudaStream_t ws = resolveStream(device_id, nullptr);
                cudaError_t err = cudaMemsetAsync(ptr, value, bytes, ws);
                if (err == cudaSuccess) err = cudaStreamSynchronize(ws);
                promise->set_value(err == cudaSuccess); });
            return future;
        }
        catch (...)
        {
            // Fallback: execute synchronously
            std::promise<bool> p;
            p.set_value(memset(ptr, value, bytes, device_id));
            return p.get_future();
        }
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
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
            return false;
        err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::synchronizeStream] failed: " << cudaGetErrorString(err));
            return false;
        }
        return true;
    }

    bool CUDABackend::streamWaitEvent(void *stream, void *event, int device_id)
    {
        (void)device_id;
        cudaError_t err = cudaStreamWaitEvent(
            static_cast<cudaStream_t>(stream),
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
        if (device_id >= device_count_ || device_id < 0)
            return false;
        if (!setDevice(device_id))
            return false;

        cudaError_t err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice,
                                          static_cast<cudaStream_t>(stream));
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::hostToDeviceOnStream] failed: " << cudaGetErrorString(err));
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
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
            return false;
        err = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                              static_cast<cudaStream_t>(stream));
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
        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::vectorAddInplace] cudaSetDevice failed: "
                      << cudaGetErrorString(err));
            return false;
        }

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

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
