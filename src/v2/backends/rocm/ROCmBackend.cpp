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
#include "AMDDeviceContext.h"
#include "HipDeviceGuard.h"
#include "backends/GPUDeviceContextPool.h"
#include "../../utils/Logger.h"
#include <hip/hip_runtime.h>
#include <chrono>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <dlfcn.h> // For HSA runtime loading
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <limits>
#include <thread>

namespace llaminar2
{

    extern "C" bool rocmOps_vector_add_inplace_fp32(
        float *output,
        const float *input,
        size_t count,
        int device_idx,
        void *stream);

    namespace
    {
        // Immortal singletons: heap-allocated and never destroyed.
        // Prevents static destruction order fiasco when KernelFactory's static
        // caches (which hold GEMM kernels with shared_ptr<LoadOrchestrator>)
        // are destroyed during __cxa_finalize AFTER these singletons.
        std::mutex &rocmPinnedAllocationsMutex()
        {
            static auto *mutex = new std::mutex();
            return *mutex;
        }

        std::unordered_map<void *, int> &rocmPinnedAllocations()
        {
            static auto *allocations = new std::unordered_map<void *, int>();
            return *allocations;
        }
    }

    // Check a HIP API result and throw on failure with full diagnostic context.
    //
    // Use this for HIP calls in **normal control-flow** that have no meaningful
    // local recovery — e.g. a failed hipSetDevice or hipStreamSynchronize on a
    // hot path means subsequent calls hit the wrong device or run on stale data.
    // Continuing typically produces silent miscompute or a delayed segfault
    // inside the HIP runtime, which is much harder to debug than failing fast.
    //
    // For HIP calls in **cleanup paths** (destructors, freeXxx, destroyXxx,
    // resource clear-before-reuse, error-rollback after a prior failure), use
    // HIP_WARN_IF_FAIL instead so we don't throw during teardown / mask the
    // original failure.
#define HIP_CHECK_OR_THROW(call)                                                    \
    do                                                                              \
    {                                                                               \
        hipError_t _err = (call);                                                   \
        if (_err != hipSuccess)                                                     \
        {                                                                           \
            std::ostringstream _oss;                                                \
            _oss << "[ROCmBackend] " << #call << " failed: "                        \
                 << hipGetErrorString(_err) << " (" << __FILE__ << ":" << __LINE__  \
                 << ")";                                                            \
            LOG_ERROR(_oss.str());                                                  \
            throw std::runtime_error(_oss.str());                                   \
        }                                                                           \
    } while (0)

    // Best-effort logging for HIP calls in cleanup paths (destructors, freeXxx,
    // destroyXxx, error-rollback after a prior failure). Logs at WARN and
    // continues — we deliberately don't throw or log at ERROR here because the
    // caller is already tearing down or unwinding from a different failure.
    // Throwing during stack unwind would call std::terminate; logging at ERROR
    // would mask the real upstream failure.
#define HIP_WARN_IF_FAIL(call)                                                      \
    do                                                                              \
    {                                                                               \
        hipError_t _err = (call);                                                   \
        if (_err != hipSuccess)                                                     \
        {                                                                           \
            LOG_WARN("[ROCmBackend] " << #call << " failed: "                       \
                                      << hipGetErrorString(_err) << " ("           \
                                      << __FILE__ << ":" << __LINE__ << ")");      \
        }                                                                           \
    } while (0)

    namespace
    {
        /**
         * RAII guard that saves the current HIP device on construction and
         * restores it (including HipDeviceGuard tracking) on destruction.
         *
         * This ensures that ROCmBackend operations that internally call
         * hipSetDevice() do not corrupt the caller's device context.
         * Without this, TP threads running on device 0 can find themselves
         * silently switched to device 1 after a coherence operation,
         * causing "invalid resource handle" errors on kernel launch.
         */
        class HipDeviceSaveRestore
        {
        public:
            HipDeviceSaveRestore() : saved_device_(-1), valid_(false)
            {
                hipError_t err = hipGetDevice(&saved_device_);
                valid_ = (err == hipSuccess && saved_device_ >= 0);
            }

            ~HipDeviceSaveRestore()
            {
                if (valid_)
                {
                    // Restore the HIP device and synchronize HipDeviceGuard tracking
                    HipDeviceGuard::forceSetDevice(saved_device_);
                }
            }

            HipDeviceSaveRestore(const HipDeviceSaveRestore &) = delete;
            HipDeviceSaveRestore &operator=(const HipDeviceSaveRestore &) = delete;

        private:
            int saved_device_;
            bool valid_;
        };

        struct PointerEvent
        {
            const char *kind = "?";
            void *base_ptr = nullptr;
            size_t size_bytes = 0;
            int device_id = -1;
            uint64_t sequence = 0;
            uint64_t thread_hash = 0;
            bool active = false;
        };

        // Immortal: heap-allocated and never destroyed to avoid static
        // destruction order issues during process exit.
        std::mutex &g_ptr_registry_mutex = *new std::mutex();
        std::unordered_map<void *, ROCmPointerOwnerInfo> &g_active_ptrs =
            *new std::unordered_map<void *, ROCmPointerOwnerInfo>();
        std::deque<PointerEvent> &g_ptr_events = *new std::deque<PointerEvent>();
        uint64_t g_ptr_sequence = 0;
        constexpr size_t kMaxPointerEvents = 512;

        uint64_t currentThreadHash()
        {
            return static_cast<uint64_t>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
        }

        void recordPointerEvent(const char *kind, void *ptr, size_t bytes, int device_id, bool active)
        {
            PointerEvent event;
            event.kind = kind;
            event.base_ptr = ptr;
            event.size_bytes = bytes;
            event.device_id = device_id;
            event.active = active;
            event.thread_hash = currentThreadHash();
            event.sequence = ++g_ptr_sequence;
            g_ptr_events.push_back(event);
            if (g_ptr_events.size() > kMaxPointerEvents)
            {
                g_ptr_events.pop_front();
            }
        }
    }

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
    // Stream Resolution Helper
    // ====================================================================

    /// Resolve a HIP stream for the given device. When the caller passes
    /// nullptr we look up the device context's default (non-blocking) stream
    /// so that NO operation ever runs on the null HIP stream.
    static hipStream_t resolveStream(int device_id, void *stream)
    {
        if (stream)
            return static_cast<hipStream_t>(stream);

        try
        {
            auto &ctx = GPUDeviceContextPool::instance().getAMDContext(device_id);
            void *def = ctx.defaultStream();
            if (def)
                return static_cast<hipStream_t>(def);
        }
        catch (...)
        {
            // Context not yet initialised (early weight load, tests).
        }
        return nullptr; // absolute fallback
    }

    // ====================================================================
    // Memory Transfer Operations
    // ====================================================================

    bool ROCmBackend::deviceToHost(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        HipDeviceSaveRestore device_guard;
        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipPointerAttribute_t src_attrs{};
        hipError_t src_attr_err = hipPointerGetAttributes(&src_attrs, src);
        if (src_attr_err != hipSuccess)
        {
            (void)hipGetLastError();  // clear sticky error state
            LOG_ERROR("[ROCmBackend::deviceToHost] Invalid source device pointer: src=" << src
                                                                                        << " bytes=" << bytes
                                                                                        << " device_id=" << device_id
                                                                                        << " hip_error=" << hipGetErrorString(src_attr_err));
            return false;
        }

        if (src_attrs.device != device_id)
        {
            LOG_ERROR("[ROCmBackend::deviceToHost] Source pointer device mismatch: src=" << src
                                                                                         << " ptr_device=" << src_attrs.device
                                                                                         << " requested_device=" << device_id
                                                                                         << " bytes=" << bytes);
            return false;
        }

        hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
                                        resolveStream(device_id, stream));
        if (err != hipSuccess)
            return false;
        err = hipStreamSynchronize(resolveStream(device_id, stream));
        return (err == hipSuccess);
    }

    bool ROCmBackend::deviceToHostFast(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        // Fast path: skip pointer validation and device save/restore.
        // Caller guarantees src is valid device memory and GPU work is complete.
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }
        hipStream_t s = resolveStream(device_id, stream);
        hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, s);
        if (err != hipSuccess)
            return false;
        err = hipStreamSynchronize(s);
        return (err == hipSuccess);
    }

    bool ROCmBackend::pinHostMemory(void *ptr, size_t bytes)
    {
        hipError_t err = hipHostRegister(ptr, bytes, hipHostRegisterDefault);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmBackend::pinHostMemory] hipHostRegister failed for "
                     << bytes << " bytes: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool ROCmBackend::unpinHostMemory(void *ptr)
    {
        hipError_t err = hipHostUnregister(ptr);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmBackend::unpinHostMemory] hipHostUnregister failed: "
                     << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // Forward declaration for HIP argmax kernel (implemented in ROCmArgmaxKernels.hip)
    extern "C" bool rocmOps_argmax_f32(
        const float *data, int n, float *out_value, int *out_index,
        int device_idx, void *stream);

    bool ROCmBackend::argmaxF32(const void *data_device, int n, int device_id,
                                float *out_value, int *out_index, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device || n <= 0)
            return false;

        // Lazily allocate per-device result buffers
        if (argmax_buffers_.empty())
            argmax_buffers_.resize(device_count_);

        auto &bufs = argmax_buffers_[device_id];
        if (!bufs.value_ptr)
        {
            hipError_t err = hipSetDevice(device_id);
            if (err != hipSuccess)
                return false;
            err = hipMalloc(&bufs.value_ptr, sizeof(float));
            if (err != hipSuccess)
                return false;
            err = hipMalloc(&bufs.index_ptr, sizeof(int));
            if (err != hipSuccess)
            {
                HIP_WARN_IF_FAIL(hipFree(bufs.value_ptr));  // rollback after malloc fail
                bufs.value_ptr = nullptr;
                return false;
            }
        }

        // Launch kernel on device's managed stream
        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipStream_t s = resolveStream(device_id, stream);
        if (!rocmOps_argmax_f32(
                static_cast<const float *>(data_device), n,
                static_cast<float *>(bufs.value_ptr),
                static_cast<int *>(bufs.index_ptr),
                device_id, s))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipStreamSynchronize(s));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(out_value, bufs.value_ptr, sizeof(float), hipMemcpyDeviceToHost, s));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(out_index, bufs.index_ptr, sizeof(int), hipMemcpyDeviceToHost, s));
        HIP_CHECK_OR_THROW(hipStreamSynchronize(s));

        return true;
    }

    // Forward declaration for HIP top-k kernel (implemented in ROCmSamplingKernels.hip)
    extern "C" bool rocmOps_topk_f32(
        const float *data, int n, int k, float *out_values, int *out_indices,
        int device_idx, void *stream);

    bool ROCmBackend::topKF32(const void *data_device, int n, int k, int device_id,
                              float *out_values, int *out_indices, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0 || !data_device || n <= 0 || k <= 0)
            return false;

        // Clamp k to n
        if (k > n)
            k = n;

        // Lazily allocate per-device result buffers
        if (topk_buffers_.empty())
            topk_buffers_.resize(device_count_);

        auto &bufs = topk_buffers_[device_id];

        // Reallocate if k grew beyond previous allocation
        if (bufs.allocated_k < k)
        {
            HIP_CHECK_OR_THROW(hipSetDevice(device_id));
            if (bufs.values_ptr)
                HIP_WARN_IF_FAIL(hipFree(bufs.values_ptr));   // clearing old buffer before realloc
            if (bufs.indices_ptr)
                HIP_WARN_IF_FAIL(hipFree(bufs.indices_ptr));  // clearing old buffer before realloc

            hipError_t err = hipMalloc(&bufs.values_ptr, k * sizeof(float));
            if (err != hipSuccess)
            {
                bufs.values_ptr = nullptr;
                bufs.allocated_k = 0;
                return false;
            }
            err = hipMalloc(&bufs.indices_ptr, k * sizeof(int));
            if (err != hipSuccess)
            {
                HIP_WARN_IF_FAIL(hipFree(bufs.values_ptr));   // rollback after malloc fail
                bufs.values_ptr = nullptr;
                bufs.allocated_k = 0;
                return false;
            }
            bufs.allocated_k = k;
        }

        // Launch kernel
        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipStream_t s = resolveStream(device_id, stream);
        if (!rocmOps_topk_f32(
                static_cast<const float *>(data_device), n, k,
                static_cast<float *>(bufs.values_ptr),
                static_cast<int *>(bufs.indices_ptr),
                device_id, s))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipStreamSynchronize(s));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(out_values, bufs.values_ptr, k * sizeof(float), hipMemcpyDeviceToHost, s));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(out_indices, bufs.indices_ptr, k * sizeof(int), hipMemcpyDeviceToHost, s));
        HIP_CHECK_OR_THROW(hipStreamSynchronize(s));

        return true;
    }

    // Forward declaration for HIP penalty kernel (implemented in ROCmSamplingKernels.hip)
    extern "C" bool rocmOps_apply_logit_penalties_f32(
        float *logits, const int *token_ids, const float *penalties,
        int num_penalties, int vocab_size, int device_idx, void *stream);

    bool ROCmBackend::applyLogitPenaltiesF32(void *logits_device,
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
            HIP_CHECK_OR_THROW(hipSetDevice(device_id));
            if (bufs.token_ids_ptr)
                HIP_WARN_IF_FAIL(hipFree(bufs.token_ids_ptr));
            if (bufs.penalties_ptr)
                HIP_WARN_IF_FAIL(hipFree(bufs.penalties_ptr));

            hipError_t err = hipMalloc(&bufs.token_ids_ptr, num_penalties * sizeof(int));
            if (err != hipSuccess)
            {
                bufs.token_ids_ptr = nullptr;
                bufs.allocated_count = 0;
                return false;
            }
            err = hipMalloc(&bufs.penalties_ptr, num_penalties * sizeof(float));
            if (err != hipSuccess)
            {
                HIP_WARN_IF_FAIL(hipFree(bufs.token_ids_ptr));
                bufs.token_ids_ptr = nullptr;
                bufs.allocated_count = 0;
                return false;
            }
            bufs.allocated_count = num_penalties;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipStream_t s = resolveStream(device_id, stream);

        // Upload penalty data to device
        HIP_CHECK_OR_THROW(hipMemcpyAsync(bufs.token_ids_ptr, token_ids_host,
                                           num_penalties * sizeof(int),
                                           hipMemcpyHostToDevice, s));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(bufs.penalties_ptr, penalties_host,
                                           num_penalties * sizeof(float),
                                           hipMemcpyHostToDevice, s));

        // Apply penalties in-place on device
        if (!rocmOps_apply_logit_penalties_f32(
                static_cast<float *>(logits_device),
                static_cast<const int *>(bufs.token_ids_ptr),
                static_cast<const float *>(bufs.penalties_ptr),
                num_penalties, vocab_size, device_id, s))
        {
            return false;
        }

        HIP_CHECK_OR_THROW(hipStreamSynchronize(s));
        return true;
    }

    bool ROCmBackend::hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        HipDeviceSaveRestore device_guard;
        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipPointerAttribute_t dst_attrs{};
        hipError_t dst_attr_err = hipPointerGetAttributes(&dst_attrs, dst);
        if (dst_attr_err != hipSuccess)
        {
            (void)hipGetLastError();  // clear sticky error state
            LOG_ERROR("[ROCmBackend::hostToDevice] Invalid destination device pointer: dst=" << dst
                                                                                             << " bytes=" << bytes
                                                                                             << " device_id=" << device_id
                                                                                             << " hip_error=" << hipGetErrorString(dst_attr_err));
            return false;
        }

        if (dst_attrs.device != device_id)
        {
            LOG_ERROR("[ROCmBackend::hostToDevice] Destination pointer device mismatch: dst=" << dst
                                                                                              << " ptr_device=" << dst_attrs.device
                                                                                              << " requested_device=" << device_id
                                                                                              << " bytes=" << bytes);
            return false;
        }

        hipStream_t s = resolveStream(device_id, stream);
        hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, s);
        if (err != hipSuccess)
            return false;
        err = hipStreamSynchronize(s);
        return (err == hipSuccess);
    }

    bool ROCmBackend::synchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        HipDeviceSaveRestore device_guard;
        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipError_t err = hipDeviceSynchronize();
        if (err == hipErrorStreamCaptureUnsupported ||
            err == hipErrorStreamCaptureImplicit)
        {
            // Benign: graph capture is active on this device — skip sync.
            return true;
        }
        return (err == hipSuccess);
    }

    bool ROCmBackend::streamSynchronize(int device_id)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        HipDeviceSaveRestore device_guard;
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

        HipDeviceSaveRestore device_guard;
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

        HipDeviceSaveRestore device_guard;
        HIP_WARN_IF_FAIL(hipSetDevice(device_id));
        hipEvent_t hip_event = reinterpret_cast<hipEvent_t>(event);
        HIP_WARN_IF_FAIL(hipEventDestroy(hip_event));
    }

    bool ROCmBackend::recordEvent(void *event, int device_id, void *stream)
    {
        if (!event || device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        HipDeviceSaveRestore device_guard;
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        hipEvent_t hip_event = reinterpret_cast<hipEvent_t>(event);
        hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream); // nullptr = default stream
        err = hipEventRecord(hip_event, hip_stream);
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

        HipDeviceSaveRestore device_guard;
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

        // Intentional device change — update HipDeviceGuard tracking
        // (no save/restore here, caller explicitly wants to change device)
        int result = HipDeviceGuard::forceSetDevice(device_id);
        return (result == 0);
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
        HipDeviceSaveRestore device_guard;
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << ": " << hipGetErrorString(err));
            return nullptr;
        }

        // Pre-allocation memory check: verify sufficient free VRAM before attempting hipMalloc.
        // This provides a graceful error with actionable diagnostics instead of a raw OOM crash.
        {
            size_t free_bytes = 0, total_bytes = 0;
            hipError_t mem_err = hipMemGetInfo(&free_bytes, &total_bytes);
            if (mem_err == hipSuccess)
            {
                // Require at least 64MB headroom beyond the allocation itself
                constexpr size_t HEADROOM = 64ULL * 1024 * 1024;
                if (bytes + HEADROOM > free_bytes)
                {
                    double req_mb = bytes / (1024.0 * 1024.0);
                    double free_mb = free_bytes / (1024.0 * 1024.0);
                    double total_mb = total_bytes / (1024.0 * 1024.0);
                    double used_mb = (total_bytes - free_bytes) / (1024.0 * 1024.0);
                    LOG_ERROR("[ROCmBackend] Insufficient GPU memory on device " << device_id
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
        err = hipMalloc(&ptr, bytes);
        if (err != hipSuccess)
        {
            // Include memory diagnostics in the error message
            size_t free_bytes = 0, total_bytes = 0;
            (void)hipMemGetInfo(&free_bytes, &total_bytes);  // diagnostic-only; OK if it fails
            LOG_ERROR("[ROCmBackend] hipMalloc failed for " << bytes << " bytes on device "
                                                            << device_id << ": " << hipGetErrorString(err)
                                                            << " (free: " << (free_bytes / (1024 * 1024))
                                                            << " MB, total: " << (total_bytes / (1024 * 1024)) << " MB)");
            return nullptr;
        }

        // TRACE: Log allocation with device and pointer for debugging multi-GPU memory issues
        LOG_TRACE("[ROCmBackend::allocate] ALLOC ptr=" << ptr << " bytes=" << bytes
                                                       << " device_id=" << device_id << " (ROCm ordinal)");

        {
            std::lock_guard<std::mutex> lock(g_ptr_registry_mutex);
            ROCmPointerOwnerInfo info;
            info.base_ptr = ptr;
            info.size_bytes = bytes;
            info.device_id = device_id;
            info.active = true;
            info.thread_hash = currentThreadHash();
            info.sequence = g_ptr_sequence + 1;
            g_active_ptrs[ptr] = info;
            recordPointerEvent("alloc", ptr, bytes, device_id, true);
        }

        LOG_DEBUG("[ROCM_PTR_ALLOC] ptr=" << ptr
                                          << " bytes=" << bytes
                                          << " device=" << device_id);

        // DIAGNOSTIC: Verify allocation ended up on the correct device
        {
            hipPointerAttribute_t attr = {};
            hipError_t attr_err = hipPointerGetAttributes(&attr, ptr);
            if (attr_err == hipSuccess && attr.device != device_id)
            {
                LOG_ERROR("[ROCmBackend::allocate] WRONG DEVICE! Requested device_id=" << device_id
                                                                                       << " but hipPointerGetAttributes says device=" << attr.device
                                                                                       << " ptr=" << ptr << " bytes=" << bytes);
            }
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
        HipDeviceSaveRestore device_guard;
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << ": " << hipGetErrorString(err));
            if (device_ptr)
                *device_ptr = nullptr;
            return nullptr;
        }

        // Allocate mapped host memory (GPU can write directly to this via PCIe)
        // NOTE: Do NOT use hipHostMallocWriteCombined here. WC memory makes CPU
        // reads ~1000x slower (each load bypasses all CPU caches). Logits are
        // GPU-written then CPU-read (argmax in sampler), so WC provides no
        // benefit and causes a ~13ms penalty per token for 152K-vocab models.
        void *host_ptr = nullptr;
        err = hipHostMalloc(&host_ptr, bytes, hipHostMallocMapped);
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
                HIP_WARN_IF_FAIL(hipHostFree(host_ptr));  // rollback after getDevicePointer fail
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
        HipDeviceSaveRestore device_guard;
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_WARN("[ROCmBackend] Invalid device ID " << device_id << " for freeMapped, attempting anyway");
        }
        else
        {
            HIP_WARN_IF_FAIL(hipSetDevice(device_id));  // best-effort in cleanup path
        }

        hipError_t err = hipHostFree(host_ptr);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmBackend] hipHostFree failed: " << hipGetErrorString(err));
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
        HipDeviceSaveRestore device_guard;
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

        size_t recorded_size = 0;
        {
            std::lock_guard<std::mutex> lock(g_ptr_registry_mutex);
            auto it = g_active_ptrs.find(ptr);
            if (it != g_active_ptrs.end())
            {
                recorded_size = it->second.size_bytes;
                it->second.active = false;
                recordPointerEvent("free", ptr, it->second.size_bytes, device_id, false);
                g_active_ptrs.erase(it);
            }
            else
            {
                recordPointerEvent("free-unknown", ptr, 0, device_id, false);
            }
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
        else
        {
            LOG_DEBUG("[ROCM_PTR_FREE] ptr=" << ptr
                                             << " bytes=" << recorded_size
                                             << " device=" << device_id);
        }
    }

    bool ROCmBackend::queryPointerOwner(const void *ptr, ROCmPointerOwnerInfo &info)
    {
        if (!ptr)
        {
            return false;
        }

        std::lock_guard<std::mutex> lock(g_ptr_registry_mutex);
        uintptr_t target = reinterpret_cast<uintptr_t>(ptr);
        for (const auto &[base, meta] : g_active_ptrs)
        {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(base);
            const uintptr_t end = begin + meta.size_bytes;
            if (target >= begin && target < end)
            {
                info = meta;
                return true;
            }
        }
        return false;
    }

    void ROCmBackend::dumpRecentPointerEvents(size_t max_events)
    {
        std::lock_guard<std::mutex> lock(g_ptr_registry_mutex);
        if (g_ptr_events.empty())
        {
            LOG_WARN("[ROCM_PTR_EVENTS] no events recorded");
            return;
        }

        const size_t total = g_ptr_events.size();
        const size_t start = (total > max_events) ? (total - max_events) : 0;
        LOG_WARN("[ROCM_PTR_EVENTS] dumping " << (total - start) << " of " << total << " recent events");
        for (size_t i = start; i < total; ++i)
        {
            const auto &e = g_ptr_events[i];
            LOG_WARN("[ROCM_PTR_EVENTS] #" << e.sequence
                                           << " kind=" << e.kind
                                           << " ptr=" << e.base_ptr
                                           << " bytes=" << e.size_bytes
                                           << " dev=" << e.device_id
                                           << " active=" << (e.active ? 1 : 0)
                                           << " thread=" << e.thread_hash);
        }
    }

    bool ROCmBackend::memset(void *ptr, int value, size_t bytes, int device_id, void *stream)
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
        HipDeviceSaveRestore device_guard;
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] Failed to set device " << device_id << " before hipMemset: "
                                                            << hipGetErrorString(err));
            return false;
        }

        err = hipMemsetAsync(ptr, value, bytes, resolveStream(device_id, stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipMemsetAsync failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmBackend::vectorAddInplace(void *output, const void *input, size_t count,
                                       int element_size, int device_id, void *stream)
    {
        if (count == 0)
            return true;
        if (!output || !input)
        {
            LOG_ERROR("[ROCmBackend::vectorAddInplace] Null output or input pointer");
            return false;
        }
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend::vectorAddInplace] Invalid device ID " << device_id);
            return false;
        }
        if (element_size != static_cast<int>(sizeof(float)))
        {
            LOG_ERROR("[ROCmBackend::vectorAddInplace] unsupported element_size: " << element_size);
            return false;
        }

        return rocmOps_vector_add_inplace_fp32(
            static_cast<float *>(output),
            static_cast<const float *>(input),
            count,
            device_id,
            stream ? stream : resolveStream(device_id, nullptr));
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

        HipDeviceSaveRestore device_guard;
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
    // Stream Management
    // ====================================================================

    void *ROCmBackend::createStream(int device_id)
    {
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::createStream] hipSetDevice(" << device_id
                      << ") failed: " << hipGetErrorString(err));
            return nullptr;
        }

        hipStream_t stream;
        err = hipStreamCreateWithFlags(&stream, hipStreamNonBlocking);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::createStream] hipStreamCreateWithFlags failed: "
                      << hipGetErrorString(err));
            return nullptr;
        }
        return stream;
    }

    void ROCmBackend::destroyStream(void *stream, int device_id)
    {
        if (!stream)
            return;
        (void)hipSetDevice(device_id);
        (void)hipStreamDestroy(static_cast<hipStream_t>(stream));
    }

    bool ROCmBackend::synchronizeStream(void *stream, int device_id)
    {
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
            return false;
        err = hipStreamSynchronize(static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::synchronizeStream] failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool ROCmBackend::streamWaitEvent(void *stream, void *event, int device_id)
    {
        (void)device_id;
        hipError_t err = hipStreamWaitEvent(
            static_cast<hipStream_t>(stream),
            static_cast<hipEvent_t>(event), 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamWaitEvent] failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // ====================================================================
    // Async H2D Without Sync (Pipeline Support)
    // ====================================================================

    bool ROCmBackend::hostToDeviceOnStream(void *dst, const void *src, size_t bytes,
                                            int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0)
            return false;

        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
            return false;

        err = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice,
                             static_cast<hipStream_t>(stream));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::hostToDeviceOnStream] failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    // ====================================================================
    // Pinned Host Memory
    // ====================================================================

    void *ROCmBackend::allocatePinned(size_t bytes, int device_id)
    {
        hipError_t set_err = hipSetDevice(device_id);
        if (set_err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::allocatePinned] hipSetDevice(" << device_id
                      << ") failed: " << hipGetErrorString(set_err));
            return nullptr;
        }

        void *ptr = nullptr;
        hipError_t err = hipHostMalloc(&ptr, bytes, hipHostMallocDefault);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::allocatePinned] hipHostMalloc(" << bytes
                      << ") failed: " << hipGetErrorString(err));
            return nullptr;
        }
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations()[ptr] = device_id;
        }
        return ptr;
    }

    void ROCmBackend::freePinned(void *ptr, int device_id)
    {
        if (!ptr)
            return;

        int owner_device = device_id;
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            auto &allocations = rocmPinnedAllocations();
            auto it = allocations.find(ptr);
            if (it != allocations.end())
            {
                owner_device = it->second;
                allocations.erase(it);
            }
            else
            {
                // Pointer not found in tracking map. This can happen during static
                // destruction when the Meyers singleton (rocmPinnedAllocations) was
                // destroyed before KernelFactory's static caches, causing the map
                // to be re-created empty. Still proceed with hipHostFree using the
                // caller-provided device_id.
                LOG_DEBUG("[ROCmBackend::freePinned] untracked pinned pointer " << ptr
                          << " (normal during static destruction)");
            }
        }

        hipError_t set_err = hipSetDevice(owner_device);
        if (set_err != hipSuccess)
        {
            // During static destruction, hipSetDevice may fail if the HIP runtime
            // has already been torn down. This is expected — skip the free.
            LOG_DEBUG("[ROCmBackend::freePinned] hipSetDevice(" << owner_device
                     << ") failed before hipHostFree: " << hipGetErrorString(set_err)
                     << " (may be normal during shutdown)");
            return;
        }

        hipError_t err = hipHostFree(ptr);
        if (err != hipSuccess)
        {
            LOG_DEBUG("[ROCmBackend::freePinned] hipHostFree failed for " << ptr
                     << " on device " << owner_device << ": " << hipGetErrorString(err)
                     << " (may be normal during shutdown)");
        }
    }

    // ====================================================================
    // Async Operations (via AMDDeviceContext worker thread)
    // ====================================================================

    std::future<bool> ROCmBackend::deviceToHostAsync(void *dst, const void *src, size_t bytes, int device_id)
    {
        try
        {
            AMDDeviceContext &ctx = static_cast<AMDDeviceContext &>(
                GPUDeviceContextPool::instance().getAMDContext(device_id));

            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();

            ctx.submitAsync([this, dst, src, bytes, device_id, promise]()
                            {
                bool result = deviceToHost(dst, src, bytes, device_id);
                promise->set_value(result); });

            return future;
        }
        catch (...)
        {
            // Fall back to synchronous execution
            std::promise<bool> promise;
            promise.set_value(deviceToHost(dst, src, bytes, device_id));
            return promise.get_future();
        }
    }

    std::future<bool> ROCmBackend::hostToDeviceAsync(void *dst, const void *src, size_t bytes, int device_id)
    {
        try
        {
            AMDDeviceContext &ctx = static_cast<AMDDeviceContext &>(
                GPUDeviceContextPool::instance().getAMDContext(device_id));

            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();

            ctx.submitAsync([this, dst, src, bytes, device_id, promise]()
                            {
                bool result = hostToDevice(dst, src, bytes, device_id);
                promise->set_value(result); });

            return future;
        }
        catch (...)
        {
            // Fall back to synchronous execution
            std::promise<bool> promise;
            promise.set_value(hostToDevice(dst, src, bytes, device_id));
            return promise.get_future();
        }
    }

    std::future<bool> ROCmBackend::synchronizeAsync(int device_id)
    {
        try
        {
            AMDDeviceContext &ctx = static_cast<AMDDeviceContext &>(
                GPUDeviceContextPool::instance().getAMDContext(device_id));

            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();

            ctx.submitAsync([this, device_id, promise]()
                            {
                bool result = synchronize(device_id);
                promise->set_value(result); });

            return future;
        }
        catch (...)
        {
            // Fall back to synchronous execution
            std::promise<bool> promise;
            promise.set_value(synchronize(device_id));
            return promise.get_future();
        }
    }

    std::future<void *> ROCmBackend::allocateAsync(size_t bytes, int device_id)
    {
        try
        {
            AMDDeviceContext &ctx = static_cast<AMDDeviceContext &>(
                GPUDeviceContextPool::instance().getAMDContext(device_id));

            auto promise = std::make_shared<std::promise<void *>>();
            auto future = promise->get_future();

            ctx.submitAsync([this, bytes, device_id, promise]()
                            {
                void* result = allocate(bytes, device_id);
                promise->set_value(result); });

            return future;
        }
        catch (...)
        {
            // Fall back to synchronous execution
            std::promise<void *> promise;
            promise.set_value(allocate(bytes, device_id));
            return promise.get_future();
        }
    }

    std::future<void> ROCmBackend::freeAsync(void *ptr, int device_id)
    {
        try
        {
            AMDDeviceContext &ctx = static_cast<AMDDeviceContext &>(
                GPUDeviceContextPool::instance().getAMDContext(device_id));

            auto promise = std::make_shared<std::promise<void>>();
            auto future = promise->get_future();

            ctx.submitAsync([this, ptr, device_id, promise]()
                            {
                free(ptr, device_id);
                promise->set_value(); });

            return future;
        }
        catch (...)
        {
            // Fall back to synchronous execution
            free(ptr, device_id);
            std::promise<void> promise;
            promise.set_value();
            return promise.get_future();
        }
    }

    std::future<bool> ROCmBackend::memsetAsync(void *ptr, int value, size_t bytes, int device_id)
    {
        try
        {
            AMDDeviceContext &ctx = static_cast<AMDDeviceContext &>(
                GPUDeviceContextPool::instance().getAMDContext(device_id));

            auto promise = std::make_shared<std::promise<bool>>();
            auto future = promise->get_future();

            ctx.submitAsync([this, ptr, value, bytes, device_id, promise]()
                            {
                bool result = memset(ptr, value, bytes, device_id);
                promise->set_value(result); });

            return future;
        }
        catch (...)
        {
            // Fall back to synchronous execution
            std::promise<bool> promise;
            promise.set_value(memset(ptr, value, bytes, device_id));
            return promise.get_future();
        }
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

    bool ROCmBackend::deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }

        HipDeviceSaveRestore device_guard;
        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            return false;
        }

        hipStream_t s = resolveStream(device_id, stream);
        hipError_t err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, s);
        if (err != hipSuccess)
            return false;
        err = hipStreamSynchronize(s);
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

        // Attempt 1: IoMemory flag (for memory-mapped I/O regions)
        LOG_INFO("[ROCmBackend::registerIoMemory] Trying hipHostRegisterIoMemory flag (0x4)");
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
                HIP_WARN_IF_FAIL(hipHostUnregister(ptr));  // rollback after getDevicePointer fail
            }
        }
        else
        {
            LOG_WARN("[ROCmBackend::registerIoMemory] hipHostRegisterIoMemory failed: "
                     << hipGetErrorString(err) << " (code " << static_cast<int>(err) << ")");
        }

        // Attempt 2: Mapped + Portable flags
        LOG_INFO("[ROCmBackend::registerIoMemory] Trying hipHostRegisterMapped | hipHostRegisterPortable");
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
                HIP_WARN_IF_FAIL(hipHostUnregister(ptr));  // rollback after getDevicePointer fail
            }
        }
        else
        {
            LOG_WARN("[ROCmBackend::registerIoMemory] hipHostRegisterMapped failed: "
                     << hipGetErrorString(err) << " (code " << static_cast<int>(err) << ")");
        }

        // Attempt 3: Default flags
        LOG_INFO("[ROCmBackend::registerIoMemory] Trying hipHostRegisterDefault");
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
                HIP_WARN_IF_FAIL(hipHostUnregister(ptr));  // rollback after getDevicePointer fail
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
            HIP_WARN_IF_FAIL(hipDestroyExternalMemory(extMem));  // rollback after mapped-buffer fail
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
