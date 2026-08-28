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
#include "CUDAGraphCapture.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"
#include "../../execution/moe/DeviceMoERebalanceABI.h"
#include "../../execution/mtp/MTPVerifierOutcomeGraph.h"
#include "../../transfer/MappedTransferProgressABI.h"
#include "../../kernels/common/SamplingMath.h"
#include "../../kernels/cuda/ops/CUDARowSelectKernels.h"
#include "../../kernels/cuda/ops/CUDAVectorAddKernels.h"
#include <cuda/atomic>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <memory>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <exception>
#include <mutex>
#include <unordered_map>

namespace llaminar2
{
    extern "C" bool llaminar2_retireCUDATensorValidatorRuntimeGeneration(
        int device_id);

    namespace
    {
        constexpr std::uintptr_t kDeviceAllocationAlignment = 256;
        constexpr unsigned int kMappedHostCopyThreads = 256u;
        constexpr unsigned int kMappedHostCopyMaximumBlocks = 4096u;

        /** Logical CUDA allocation tracked by the canonical backend allocator. */
        struct CUDADeviceAllocationRecord
        {
            size_t bytes = 0u; ///< Exact cudaMalloc byte count.
            int device_id = -1; ///< Owning CUDA ordinal.
        };

        /** CUDA host allocation or registration whose aliases die on reset. */
        struct CUDAHostRegistrationRecord
        {
            size_t bytes = 0u; ///< Known byte count, or zero for free-only APIs.
            int device_id = -1; ///< Registration-context CUDA ordinal.
        };

        /**
         * @return Mutex serializing resource mutation against runtime reset.
         *
         * It is intentionally process-lifetime storage: BackendManager's CUDA
         * backend is also process-lifetime, and static destruction order must
         * never recreate an empty authority around still-live resources.
         */
        std::mutex &cudaRuntimeResourceLifecycleMutex()
        {
            static auto *mutex = new std::mutex();
            return *mutex;
        }

        /** @return Exact live allocations made through CUDABackend::allocate. */
        std::unordered_map<void *, CUDADeviceAllocationRecord> &
        cudaTrackedDeviceAllocations()
        {
            static auto *allocations =
                new std::unordered_map<void *, CUDADeviceAllocationRecord>();
            return *allocations;
        }

        /** @return Exact live CUDA host allocations and registrations. */
        std::unordered_map<void *, CUDAHostRegistrationRecord> &
        cudaTrackedHostRegistrations()
        {
            static auto *registrations =
                new std::unordered_map<void *, CUDAHostRegistrationRecord>();
            return *registrations;
        }

        /**
         * @brief Preserve the caller's exact CUDA device across backend work.
         *
         * Device-memory accounting is a lifecycle operation and may inspect a
         * device other than the caller's current one.  Failing to restore the
         * TLS current-device identity can make a later stream or library handle
         * appear to belong to the wrong device, so restoration failure is
         * terminal rather than a warning.
         */
        class CUDADeviceSaveRestore final
        {
        public:
            /** @brief Capture the current CUDA device without changing it. */
            CUDADeviceSaveRestore()
            {
                valid_ = cudaGetDevice(&saved_device_) == cudaSuccess &&
                         saved_device_ >= 0;
            }

            /** @brief Restore the captured device or terminate on lost identity. */
            ~CUDADeviceSaveRestore()
            {
                if (valid_ && cudaSetDevice(saved_device_) != cudaSuccess)
                {
                    (void)cudaGetLastError();
                    LOG_ERROR(
                        "[CUDABackend] Could not restore owning CUDA device "
                        << saved_device_);
                    std::terminate();
                }
            }

            CUDADeviceSaveRestore(const CUDADeviceSaveRestore &) = delete;
            CUDADeviceSaveRestore &operator=(const CUDADeviceSaveRestore &) = delete;

            /** @return Whether an exact caller device was captured. */
            [[nodiscard]] bool valid() const noexcept { return valid_; }

        private:
            int saved_device_ = -1; ///< Exact caller device restored at scope exit.
            bool valid_ = false; ///< Guards against restoring an unknown identity.
        };

        /** @return System-scope acquire load from a node-local mapped word. */
        __device__ __forceinline__ std::uint64_t mappedSystemAcquire64(
            const std::uint64_t *value)
        {
            ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> reference(
                *const_cast<std::uint64_t *>(value));
            return reference.load(::cuda::memory_order_acquire);
        }

        /** @brief System-scope release store into a node-local mapped word. */
        __device__ __forceinline__ void mappedSystemRelease64(
            std::uint64_t *value,
            std::uint64_t published)
        {
            ::cuda::atomic_ref<std::uint64_t, ::cuda::thread_scope_system> reference(
                *value);
            reference.store(published, ::cuda::memory_order_release);
        }

        /** @brief Vectorized VRAM-to-mapped-host progress copy. */
        __global__ void mappedHostCopyVectorKernel(
            uint4 *__restrict__ destination,
            const uint4 *__restrict__ source,
            std::size_t vector_count)
        {
            const std::size_t stride =
                static_cast<std::size_t>(gridDim.x) * blockDim.x;
            for (std::size_t index =
                     static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
                 index < vector_count;
                 index += stride)
            {
                destination[index] = source[index];
            }
        }

        /** @brief Byte-total tail path for an arbitrarily aligned region. */
        __global__ void mappedHostCopyByteKernel(
            std::uint8_t *__restrict__ destination,
            const std::uint8_t *__restrict__ source,
            std::size_t bytes)
        {
            const std::size_t stride =
                static_cast<std::size_t>(gridDim.x) * blockDim.x;
            for (std::size_t index =
                     static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
                 index < bytes;
                 index += stride)
            {
                destination[index] = source[index];
            }
        }

        /**
         * @brief Snapshot every host-published transfer slot into device memory.
         *
         * One block owns one permanent slot. Thread zero reads the mapped cache
         * line exactly once and publishes generation last into the claim; the
         * following graph node consequently cannot observe a mixed command.
         */
        __global__ void mappedTransferProgressClaimKernel(
            const MappedTransferProgressCommand *__restrict__ commands,
            MappedTransferProgressClaim *__restrict__ claims,
            std::size_t slot_capacity)
        {
            const std::size_t slot = blockIdx.x;
            if (slot >= slot_capacity || threadIdx.x != 0u)
                return;

            const MappedTransferProgressCommand &command = commands[slot];
            MappedTransferProgressClaim &claim = claims[slot];
            /* The host release-stores generation after every other field. This
             * system-scope acquire is the ABI edge; an ordinary volatile load
             * is insufficient for host-mapped PCIe memory. */
            const std::uint64_t generation =
                mappedSystemAcquire64(&command.generation);
            claim.generation_magic = command.generation_magic;
            claim.generation_version = command.generation_version;
            claim.source_address = command.source_address;
            claim.destination_address = command.destination_address;
            claim.bytes = command.bytes;
            claim.source_complement = command.source_complement;
            claim.destination_complement = command.destination_complement;
            claim.bytes_complement = command.bytes_complement;
            __threadfence();
            claim.generation = generation;
        }

        /**
         * @brief Copy every active claimed slot and publish its exact completion.
         *
         * A single block owns each command, eliminating any cross-block counter
         * or order-dependent reduction. Aligned commands use 16-byte lanes;
         * arbitrary tails retain byte totality. Thread zero performs the final
         * system-release publication only after the whole block has joined.
         */
        __global__ void mappedTransferProgressCopyKernel(
            const MappedTransferProgressClaim *__restrict__ claims,
            MappedTransferProgressCompletion *__restrict__ completions,
            std::size_t slot_capacity,
            std::size_t maximum_bytes)
        {
            const std::size_t slot = blockIdx.x;
            if (slot >= slot_capacity)
                return;

            const MappedTransferProgressClaim claim = claims[slot];
            MappedTransferProgressCompletion &completion = completions[slot];
            __shared__ std::uint32_t execute_claim;
            if (threadIdx.x == 0u)
            {
                /* Completion is host-mapped system memory, so independent
                 * per-lane reads need not observe the same coherence instant.
                 * A mixed early-return decision would strand the remaining
                 * lanes at the terminal __syncthreads(). Snapshot once and
                 * broadcast the branch before any thread may leave. */
                execute_claim =
                    claim.generation != 0u &&
                    claim.generation != mappedSystemAcquire64(
                                            &completion.completed_generation)
                        ? 1u
                        : 0u;
            }
            __syncthreads();
            if (execute_claim == 0u)
                return;

            MappedTransferProgressError error =
                MappedTransferProgressError::None;
            if (claim.generation_magic !=
                    (kMappedTransferProgressMagic ^
                     static_cast<std::uint32_t>(claim.generation)) ||
                claim.generation_version !=
                    (kMappedTransferProgressVersion ^
                     static_cast<std::uint32_t>(claim.generation >> 32u)) ||
                claim.source_complement != ~claim.source_address ||
                claim.destination_complement != ~claim.destination_address ||
                claim.bytes_complement != ~claim.bytes)
            {
                error = MappedTransferProgressError::InvalidIdentity;
            }
            else if (claim.source_address == 0u ||
                     claim.destination_address == 0u)
            {
                error = MappedTransferProgressError::InvalidAddress;
            }
            else if (claim.bytes == 0u || claim.bytes > maximum_bytes)
            {
                error = MappedTransferProgressError::InvalidByteCount;
            }

            if (error == MappedTransferProgressError::None)
            {
                const auto source_address = static_cast<std::uintptr_t>(
                    claim.source_address);
                const auto destination_address = static_cast<std::uintptr_t>(
                    claim.destination_address);
                const bool vector_aligned =
                    source_address % alignof(uint4) == 0u &&
                    destination_address % alignof(uint4) == 0u;
                if (vector_aligned)
                {
                    const auto *const source =
                        reinterpret_cast<const uint4 *>(source_address);
                    auto *const destination =
                        reinterpret_cast<uint4 *>(destination_address);
                    const std::size_t vector_count =
                        static_cast<std::size_t>(claim.bytes) / sizeof(uint4);
                    for (std::size_t index = threadIdx.x;
                         index < vector_count;
                         index += blockDim.x)
                    {
                        destination[index] = source[index];
                    }
                    const std::size_t vector_bytes =
                        vector_count * sizeof(uint4);
                    auto *const destination_tail =
                        reinterpret_cast<std::uint8_t *>(destination_address);
                    const auto *const source_tail =
                        reinterpret_cast<const std::uint8_t *>(source_address);
                    for (std::size_t index = vector_bytes + threadIdx.x;
                         index < claim.bytes;
                         index += blockDim.x)
                    {
                        destination_tail[index] = source_tail[index];
                    }
                }
                else
                {
                    auto *const destination =
                        reinterpret_cast<std::uint8_t *>(destination_address);
                    const auto *const source =
                        reinterpret_cast<const std::uint8_t *>(source_address);
                    for (std::size_t index = threadIdx.x;
                         index < claim.bytes;
                         index += blockDim.x)
                    {
                        destination[index] = source[index];
                    }
                }
            }

            __syncthreads();
            if (threadIdx.x == 0u)
            {
                completion.completed_bytes =
                    error == MappedTransferProgressError::None
                        ? claim.bytes
                        : 0u;
                completion.error = static_cast<std::uint32_t>(error);
                __threadfence_system();
                mappedSystemRelease64(
                    &completion.completed_generation, claim.generation);
            }
        }

        /** @return Bounded nonzero grid for one positive item count. */
        unsigned int mappedHostCopyBlocks(std::size_t items) noexcept
        {
            return static_cast<unsigned int>(std::min<std::size_t>(
                kMappedHostCopyMaximumBlocks,
                (items + kMappedHostCopyThreads - 1u) /
                    kMappedHostCopyThreads));
        }
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
        runtime_generations_.assign(
            static_cast<size_t>(std::max(device_count_, 0)), 1u);
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
        if (!deviceToHostOnStream(dst, src, bytes, device_id, stream))
            return false;

        /*
         * This compatibility API promises completed host bytes. Fence only the
         * submitted copy frontier; synchronizing the stream would also drain
         * unrelated work queued after it by another producer.
         */
        void *const completion = createEvent(device_id);
        if (!completion)
            return false;
        const bool recorded = recordEvent(completion, device_id, stream);
        const bool completed = recorded && waitForEvent(completion, device_id);
        destroyEvent(completion, device_id);
        return completed;
    }

    bool CUDABackend::hostToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (!hostToDeviceOnStream(dst, src, bytes, device_id, stream))
            return false;

        void *const completion = createEvent(device_id);
        if (!completion)
            return false;
        const bool recorded = recordEvent(completion, device_id, stream);
        const bool completed = recorded && waitForEvent(completion, device_id);
        destroyEvent(completion, device_id);
        return completed;
    }

    bool CUDABackend::deviceToDevice(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        if (!deviceCopyAsync(dst, src, bytes, device_id, stream))
            return false;

        void *const completion = createEvent(device_id);
        if (!completion)
            return false;
        const bool recorded = recordEvent(completion, device_id, stream);
        const bool completed = recorded && waitForEvent(completion, device_id);
        destroyEvent(completion, device_id);
        return completed;
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
            LOG_ERROR("[CUDABackend::recordEvent] Invalid event publication"
                      << " event=" << event
                      << " device_id=" << device_id
                      << " device_count=" << device_count_
                      << " stream=" << stream);
            return false;
        }

        cudaError_t err = cudaSetDevice(device_id);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::recordEvent] cudaSetDevice("
                      << device_id << ") failed: "
                      << cudaGetErrorString(err));
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

    bool CUDABackend::queryEvent(void *event, int device_id, bool *ready)
    {
        if (ready)
            *ready = false;
        if (!event || !ready || device_id < 0 || device_id >= device_count_)
            return false;
        if (!setDevice(device_id))
        {
            LOG_ERROR("[CUDABackend::queryEvent] setDevice(" << device_id
                                                              << ") failed");
            return false;
        }

        const cudaError_t err = cudaEventQuery(
            reinterpret_cast<cudaEvent_t>(event));
        if (err == cudaSuccess)
        {
            *ready = true;
            return true;
        }
        if (err == cudaErrorNotReady)
            return true;

        LOG_ERROR("[CUDABackend::queryEvent] cudaEventQuery failed: "
                  << cudaGetErrorString(err)
                  << " (device=" << device_id << ", event=" << event << ")");
        return false;
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
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
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

        void *ptr = nullptr;
        /*
         * cudaMemGetInfo describes driver-visible free memory, not every byte
         * the process allocator can reuse after retiring graph-bound storage.
         * Workload admission owns the complete BOM; cudaMalloc is the exact
         * authority for this concrete allocation. Rejecting it first through
         * a second free-byte heuristic can turn reusable allocator backing
         * into a false OOM and makes CUDA diverge from the ROCm contract.
         */
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
        cudaTrackedDeviceAllocations().emplace(
            ptr,
            CUDADeviceAllocationRecord{
                .bytes = bytes,
                .device_id = device_id,
            });
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
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
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
        else
        {
            cudaTrackedDeviceAllocations().erase(ptr);
            if (vramBomEnabled())
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

    bool CUDABackend::enqueuePreparePrefillChunkView(
        const void *request_token_ids_device,
        const void *request_position_ids_device,
        const void *request_total_rows_device,
        const void *cached_tokens_device,
        int request_row_capacity,
        int bucket_seq_len,
        int pad_token_id,
        int device_id,
        void *stream,
        void *out_token_ids_device,
        void *out_position_ids_device,
        void *out_real_rows_device,
        void *out_row_stride_device)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !request_token_ids_device || !request_position_ids_device ||
            !request_total_rows_device || !cached_tokens_device ||
            request_row_capacity <= 0 || bucket_seq_len <= 0 ||
            bucket_seq_len > request_row_capacity || !stream ||
            !out_token_ids_device || !out_position_ids_device ||
            !out_real_rows_device || !out_row_stride_device)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cuda::launchPreparePrefillChunkView(
            static_cast<const int32_t *>(request_token_ids_device),
            static_cast<const int32_t *>(request_position_ids_device),
            static_cast<const int32_t *>(request_total_rows_device),
            static_cast<const int32_t *>(cached_tokens_device),
            request_row_capacity,
            bucket_seq_len,
            static_cast<int32_t>(pad_token_id),
            static_cast<int32_t *>(out_token_ids_device),
            static_cast<int32_t *>(out_position_ids_device),
            static_cast<int32_t *>(out_real_rows_device),
            static_cast<int32_t *>(out_row_stride_device),
            stream);
    }

    void *CUDABackend::allocateMapped(size_t bytes, int device_id, void **device_ptr)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
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

        cudaTrackedHostRegistrations().emplace(
            host_ptr,
            CUDAHostRegistrationRecord{
                .bytes = bytes,
                .device_id = device_id,
            });
        return host_ptr;
    }

    void CUDABackend::freeMapped(void *host_ptr, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
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
        else
        {
            cudaTrackedHostRegistrations().erase(host_ptr);
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

        CUDADeviceSaveRestore device_guard;
        if (!device_guard.valid())
        {
            (void)cudaGetLastError();
            return 0;
        }

        cudaError_t err_set = cudaSetDevice(device_id);
        if (err_set != cudaSuccess)
        {
            (void)cudaGetLastError();
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

    DeviceAllocationAccounting
    CUDABackend::deviceAllocationAccounting(int device_id) const
    {
        DeviceAllocationAccounting accounting;
        if (device_id < 0 || device_id >= device_count_)
        {
            accounting.diagnostic = "invalid CUDA device ordinal " +
                                    std::to_string(device_id);
            return accounting;
        }

        /* Allocation/free and generation reset take this same lock. The
         * returned count and byte sum therefore describe one exact canonical
         * allocator state rather than two observations that could straddle a
         * concurrent ownership transition. */
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
        for (const auto &[pointer, allocation] :
             cudaTrackedDeviceAllocations())
        {
            (void)pointer;
            if (allocation.device_id != device_id)
                continue;
            if (allocation.bytes >
                std::numeric_limits<size_t>::max() - accounting.active_bytes)
            {
                accounting.diagnostic =
                    "CUDA canonical allocation byte accounting overflow";
                return accounting;
            }
            ++accounting.active_allocations;
            accounting.active_bytes += allocation.bytes;
        }
        accounting.supported = true;
        return accounting;
    }

    DeviceMemoryCacheReclamationResult
    CUDABackend::trimUnusedDeviceMemoryCaches(int device_id)
    {
        DeviceMemoryCacheReclamationResult result;
        if (device_id < 0 || device_id >= device_count_)
        {
            result.diagnostic = "invalid CUDA device ordinal " +
                                std::to_string(device_id);
            return result;
        }

        CUDADeviceSaveRestore device_guard;
        if (!device_guard.valid())
        {
            (void)cudaGetLastError();
            result.diagnostic =
                "cudaGetDevice could not preserve the caller device identity";
            return result;
        }
        const cudaError_t select_error = cudaSetDevice(device_id);
        if (select_error != cudaSuccess)
        {
            result.diagnostic =
                "cudaSetDevice failed for CUDA:" + std::to_string(device_id) +
                ": " + cudaGetErrorString(select_error);
            (void)cudaGetLastError();
            return result;
        }
        result.supported = true;

        cudaMemPool_t default_pool = nullptr;
        bool default_pool_available = false;
        const auto query_snapshot =
            [&](DeviceMemoryCacheSnapshot &snapshot,
                const char *phase) -> bool
        {
            size_t total_bytes = 0u;
            cudaError_t error = cudaMemGetInfo(
                &snapshot.driver_free_bytes, &total_bytes);
            if (error != cudaSuccess)
            {
                result.diagnostic = std::string("cudaMemGetInfo failed ") +
                                    phase + ": " + cudaGetErrorString(error);
                (void)cudaGetLastError();
                return false;
            }

            std::uint64_t graph_used = 0u;
            std::uint64_t graph_reserved = 0u;
            error = cudaDeviceGetGraphMemAttribute(
                device_id, cudaGraphMemAttrUsedMemCurrent, &graph_used);
            if (error == cudaSuccess)
            {
                error = cudaDeviceGetGraphMemAttribute(
                    device_id,
                    cudaGraphMemAttrReservedMemCurrent,
                    &graph_reserved);
            }
            if (error != cudaSuccess)
            {
                result.diagnostic =
                    std::string("CUDA graph-memory accounting failed ") +
                    phase + ": " + cudaGetErrorString(error);
                (void)cudaGetLastError();
                return false;
            }
            snapshot.graph_accounting_available = true;
            snapshot.graph_used_bytes = static_cast<size_t>(graph_used);
            snapshot.graph_reserved_bytes =
                static_cast<size_t>(graph_reserved);

            if (!default_pool_available)
            {
                error = cudaDeviceGetDefaultMemPool(
                    &default_pool, device_id);
                if (error == cudaErrorNotSupported)
                {
                    (void)cudaGetLastError();
                    return true;
                }
                if (error != cudaSuccess || default_pool == nullptr)
                {
                    result.diagnostic =
                        std::string("CUDA default memory-pool resolution failed ") +
                        phase + ": " + cudaGetErrorString(error);
                    (void)cudaGetLastError();
                    return false;
                }
                default_pool_available = true;
            }

            std::uint64_t pool_used = 0u;
            std::uint64_t pool_reserved = 0u;
            error = cudaMemPoolGetAttribute(
                default_pool, cudaMemPoolAttrUsedMemCurrent, &pool_used);
            if (error == cudaSuccess)
            {
                error = cudaMemPoolGetAttribute(
                    default_pool,
                    cudaMemPoolAttrReservedMemCurrent,
                    &pool_reserved);
            }
            if (error != cudaSuccess)
            {
                result.diagnostic =
                    std::string("CUDA default memory-pool accounting failed ") +
                    phase + ": " + cudaGetErrorString(error);
                (void)cudaGetLastError();
                return false;
            }
            snapshot.async_pool_accounting_available = true;
            snapshot.async_pool_used_bytes = static_cast<size_t>(pool_used);
            snapshot.async_pool_reserved_bytes =
                static_cast<size_t>(pool_reserved);
            return true;
        };

        if (!query_snapshot(result.before, "before trim"))
            return result;

        const cudaError_t graph_trim_error =
            cudaDeviceGraphMemTrim(device_id);
        result.graph_trim_invoked = true;
        if (graph_trim_error != cudaSuccess)
        {
            result.diagnostic =
                "cudaDeviceGraphMemTrim failed for CUDA:" +
                std::to_string(device_id) + ": " +
                cudaGetErrorString(graph_trim_error);
            (void)cudaGetLastError();
            return result;
        }

        if (default_pool_available)
        {
            const cudaError_t pool_trim_error =
                cudaMemPoolTrimTo(default_pool, 0u);
            result.async_pool_trim_invoked = true;
            if (pool_trim_error != cudaSuccess)
            {
                result.diagnostic =
                    "cudaMemPoolTrimTo failed for CUDA:" +
                    std::to_string(device_id) + ": " +
                    cudaGetErrorString(pool_trim_error);
                (void)cudaGetLastError();
                return result;
            }
        }

        if (!query_snapshot(result.after, "after trim"))
            return result;

        result.success = true;
        return result;
    }

    DeviceRuntimeGenerationRetirementResult
    CUDABackend::retireExclusiveDeviceRuntimeGeneration(
        const DeviceRuntimeGenerationRetirementRequest &request)
    {
        DeviceRuntimeGenerationRetirementResult result;
        result.supported = true;
        const int device_id = request.deviceOrdinal();
        if (device_id < 0 || device_id >= device_count_)
        {
            result.diagnostic = "invalid CUDA device ordinal " +
                                std::to_string(device_id);
            return result;
        }

        /*
         * The lifecycle mutex is the reset exclusion edge for canonical
         * backend allocations and registrations. Ordinary execution has
         * already ended by construction of request; holding this lock makes a
         * concurrent new owner fail to interleave allocation with preflight.
         */
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());

        for (const auto &[pointer, allocation] :
             cudaTrackedDeviceAllocations())
        {
            (void)pointer;
            if (allocation.device_id == device_id)
            {
                ++result.tracked_device_allocations;
                if (allocation.bytes >
                    std::numeric_limits<size_t>::max() -
                        result.tracked_device_allocation_bytes)
                {
                    result.diagnostic =
                        "CUDA runtime-generation allocation byte accounting overflow";
                    return result;
                }
                result.tracked_device_allocation_bytes += allocation.bytes;
            }
        }
        result.tracked_host_registrations =
            cudaTrackedHostRegistrations().size();
        if (result.tracked_device_allocations != 0u ||
            result.tracked_host_registrations != 0u)
        {
            std::ostringstream diagnostic;
            diagnostic
                << "CUDA runtime-generation retirement rejected: live "
                << "tracked_device_allocations="
                << result.tracked_device_allocations
                << " tracked_device_allocation_bytes="
                << result.tracked_device_allocation_bytes
                << " tracked_host_registrations="
                << result.tracked_host_registrations;
            result.diagnostic = diagnostic.str();
            return result;
        }

        {
            std::lock_guard<std::mutex> generation_lock(
                runtime_generation_mutex_);
            result.retired_generation =
                runtime_generations_[static_cast<size_t>(device_id)];
            if (result.retired_generation == 0u ||
                result.retired_generation ==
                    std::numeric_limits<std::uint64_t>::max())
            {
                result.diagnostic =
                    "CUDA runtime generation is invalid or exhausted";
                return result;
            }
        }

        int previous_device = -1;
        cudaError_t error = cudaGetDevice(&previous_device);
        if (error != cudaSuccess || previous_device < 0)
        {
            result.diagnostic =
                "cudaGetDevice failed before runtime reset: " +
                std::string(cudaGetErrorString(error));
            (void)cudaGetLastError();
            return result;
        }
        error = cudaSetDevice(device_id);
        if (error != cudaSuccess)
        {
            result.diagnostic =
                "cudaSetDevice failed before runtime reset: " +
                std::string(cudaGetErrorString(error));
            (void)cudaGetLastError();
            return result;
        }

        size_t total_bytes = 0u;
        error = cudaMemGetInfo(
            &result.driver_free_bytes_before, &total_bytes);
        if (error != cudaSuccess)
        {
            result.diagnostic =
                "cudaMemGetInfo failed before runtime reset: " +
                std::string(cudaGetErrorString(error));
            (void)cudaGetLastError();
            if (previous_device != device_id)
                (void)cudaSetDevice(previous_device);
            return result;
        }

        if (!llaminar2_retireCUDATensorValidatorRuntimeGeneration(device_id))
        {
            result.diagnostic =
                "CUDA tensor-validator generation could not retire";
            if (previous_device != device_id)
                (void)cudaSetDevice(previous_device);
            return result;
        }

        result.reset_invoked = true;
        error = cudaDeviceReset();
        if (error != cudaSuccess)
        {
            result.diagnostic = "cudaDeviceReset failed for CUDA:" +
                                std::to_string(device_id) + ": " +
                                cudaGetErrorString(error);
            (void)cudaGetLastError();
            if (previous_device != device_id)
                (void)cudaSetDevice(previous_device);
            return result;
        }

        /*
         * Every pointer/event below belonged to the retired primary context.
         * cudaDeviceReset destroyed the resources; clearing host identities is
         * mandatory so lazy setup cannot reuse stale addresses or handles.
         */
        if (static_cast<size_t>(device_id) < argmax_buffers_.size())
            argmax_buffers_[static_cast<size_t>(device_id)] = {};
        if (static_cast<size_t>(device_id) < topk_buffers_.size())
            topk_buffers_[static_cast<size_t>(device_id)] = {};
        if (static_cast<size_t>(device_id) < sample_token_buffers_.size())
            sample_token_buffers_[static_cast<size_t>(device_id)] = {};
        if (static_cast<size_t>(device_id) < penalty_buffers_.size())
            penalty_buffers_[static_cast<size_t>(device_id)] = {};

        /* The native reset is already irrevocable. Publish its new identity
         * before any diagnostic query so an error cannot leave host caches
         * keyed to the retired generation. */
        {
            std::lock_guard<std::mutex> generation_lock(
                runtime_generation_mutex_);
            result.active_generation = result.retired_generation + 1u;
            runtime_generations_[static_cast<size_t>(device_id)] =
                result.active_generation;
        }

        error = cudaSetDevice(device_id);
        if (error == cudaSuccess)
        {
            error = cudaMemGetInfo(
                &result.driver_free_bytes_after, &total_bytes);
        }
        if (error != cudaSuccess)
        {
            result.diagnostic =
                "CUDA runtime could not materialize the fresh generation: " +
                std::string(cudaGetErrorString(error));
            (void)cudaGetLastError();
            if (previous_device != device_id &&
                cudaSetDevice(previous_device) != cudaSuccess)
            {
                (void)cudaGetLastError();
                std::terminate();
            }
            return result;
        }

        if (previous_device != device_id &&
            cudaSetDevice(previous_device) != cudaSuccess)
        {
            (void)cudaGetLastError();
            LOG_ERROR(
                "[CUDABackend] Failed to restore CUDA device "
                << previous_device << " after retiring CUDA:" << device_id);
            std::terminate();
        }

        result.success = true;
        result.diagnostic = "CUDA runtime generation retired";
        return result;
    }

    std::uint64_t CUDABackend::deviceRuntimeGeneration(
        int device_id) const
    {
        if (device_id < 0 || device_id >= device_count_)
            return 0u;
        std::lock_guard<std::mutex> lock(runtime_generation_mutex_);
        return runtime_generations_[static_cast<size_t>(device_id)];
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

    bool CUDABackend::pinHostMemory(void *ptr, size_t bytes, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
        int previous_device = -1;
        cudaError_t err = cudaGetDevice(&previous_device);
        if (!ptr || bytes == 0 || device_id < 0 ||
            err != cudaSuccess || cudaSetDevice(device_id) != cudaSuccess)
        {
            (void)cudaGetLastError();
            LOG_WARN("[CUDABackend::pinHostMemory] invalid registration for CUDA:"
                     << device_id << " ptr=" << ptr << " bytes=" << bytes);
            return false;
        }

        err = cudaHostRegister(ptr, bytes, cudaHostRegisterPortable);
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDABackend::pinHostMemory] cudaHostRegister failed for "
                     << bytes << " bytes on CUDA:" << device_id << ": "
                     << cudaGetErrorString(err));
            (void)cudaGetLastError();
            if (previous_device != device_id)
                (void)cudaSetDevice(previous_device);
            return false;
        }
        if (previous_device != device_id &&
            cudaSetDevice(previous_device) != cudaSuccess)
        {
            (void)cudaGetLastError();
            LOG_ERROR("[CUDABackend::pinHostMemory] could not restore CUDA:"
                      << previous_device << " after registering on CUDA:"
                      << device_id);
            /* The failed restore leaves the registration context current, so
             * retire the mapping before terminating. Continuing would expose
             * both a leaked registration and an unknown current device. */
            if (cudaHostUnregister(ptr) != cudaSuccess)
            {
                (void)cudaGetLastError();
            }
            std::terminate();
        }
        cudaTrackedHostRegistrations().emplace(
            ptr,
            CUDAHostRegistrationRecord{
                .bytes = bytes,
                .device_id = device_id,
            });
        return true;
    }

    bool CUDABackend::unpinHostMemory(void *ptr, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
        int previous_device = -1;
        cudaError_t err = cudaGetDevice(&previous_device);
        if (!ptr || device_id < 0 || err != cudaSuccess ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            (void)cudaGetLastError();
            LOG_ERROR("[CUDABackend::unpinHostMemory] invalid retirement for CUDA:"
                      << device_id << " ptr=" << ptr);
            return false;
        }

        err = cudaHostUnregister(ptr);
        if (err != cudaSuccess)
        {
            LOG_WARN("[CUDABackend::unpinHostMemory] cudaHostUnregister failed: "
                     << cudaGetErrorString(err) << " owner=CUDA:" << device_id
                     << " ptr=" << ptr);
            // Clear the sticky CUDA error so it doesn't contaminate subsequent
            // CUDA operations (kernel launches, memcpy, etc.).  This commonly
            // happens during teardown when mmap pages are already unmapped.
            (void)cudaGetLastError();
            if (previous_device != device_id)
                (void)cudaSetDevice(previous_device);
            return false;
        }
        if (previous_device != device_id &&
            cudaSetDevice(previous_device) != cudaSuccess)
        {
            (void)cudaGetLastError();
            LOG_ERROR("[CUDABackend::unpinHostMemory] could not restore CUDA:"
                      << previous_device << " after retiring registration on CUDA:"
                      << device_id);
            std::terminate();
        }
        cudaTrackedHostRegistrations().erase(ptr);
        return true;
    }

    bool CUDABackend::registerExternalMappedHostMemory(
        void *ptr,
        size_t bytes,
        int registration_device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
        if (!ptr || bytes == 0u || registration_device_id < 0 ||
            registration_device_id >= device_count_ ||
            cudaSetDevice(registration_device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::registerExternalMappedHostMemory] invalid region or registration device="
                      << registration_device_id);
            return false;
        }
        const cudaError_t error = cudaHostRegister(
            ptr,
            bytes,
            cudaHostRegisterMapped | cudaHostRegisterPortable);
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::registerExternalMappedHostMemory] cudaHostRegister(mapped|portable) failed for "
                      << bytes << " bytes: " << cudaGetErrorString(error));
            (void)cudaGetLastError();
            return false;
        }
        cudaTrackedHostRegistrations().emplace(
            ptr,
            CUDAHostRegistrationRecord{
                .bytes = bytes,
                .device_id = registration_device_id,
            });
        return true;
    }

    bool CUDABackend::externalMappedHostDevicePointer(
        void *host_ptr,
        int device_id,
        void **device_ptr)
    {
        if (device_ptr)
            *device_ptr = nullptr;
        if (!host_ptr || !device_ptr || device_id < 0 ||
            device_id >= device_count_ || cudaSetDevice(device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::externalMappedHostDevicePointer] invalid mapped region or device="
                      << device_id);
            return false;
        }
        const cudaError_t error = cudaHostGetDevicePointer(
            device_ptr, host_ptr, 0u);
        if (error != cudaSuccess || !*device_ptr)
        {
            LOG_ERROR("[CUDABackend::externalMappedHostDevicePointer] cudaHostGetDevicePointer failed for device="
                      << device_id << ": " << cudaGetErrorString(error));
            (void)cudaGetLastError();
            *device_ptr = nullptr;
            return false;
        }
        return true;
    }

    bool CUDABackend::unregisterExternalMappedHostMemory(
        void *ptr,
        int registration_device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
        if (!ptr || registration_device_id < 0 ||
            registration_device_id >= device_count_ ||
            cudaSetDevice(registration_device_id) != cudaSuccess)
        {
            return false;
        }
        const cudaError_t error = cudaHostUnregister(ptr);
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::unregisterExternalMappedHostMemory] cudaHostUnregister failed: "
                      << cudaGetErrorString(error));
            (void)cudaGetLastError();
            return false;
        }
        cudaTrackedHostRegistrations().erase(ptr);
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
        const int *next_leading_committed_output_count,
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
    extern "C" bool cudaOps_publish_serial_decode_commit_boundary(
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_acknowledge_decode_commit_boundary(
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_initialize_device_moe_rebalance_dispatch_ticket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        uint32_t participant_id,
        uint32_t participant_count,
        DeviceMoERebalanceDispatchTicket *ticket,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_publish_device_moe_rebalance_dispatch_ticket(
        const uint32_t *controller_magic,
        const uint32_t *controller_version,
        const uint32_t *controller_error,
        const uint32_t *decode_rounds_committed,
        const uint32_t *decode_rounds_until_maintenance,
        const uint32_t *maintenance_due,
        const uint32_t *decode_boundary_advanced,
        DeviceMoERebalanceDispatchTicket *ticket,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_initialize_device_generation(
        int request_count,
        int max_new_tokens,
        const sampling_math::DeviceGenerationDepthPolicy &depth_policy,
        sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition,
        int response_token_stride,
        int control_stride,
        int *control,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_initialize_device_generation_dispatch_tickets(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        int *control,
        int control_stride,
        int request_count,
        sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream);
    extern "C" bool cudaOps_publish_device_generation_dispatch_tickets(
        int *control,
        int control_stride,
        int request_count,
        const uint32_t *maintenance_due,
        sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream);
    extern "C" bool
    cudaMoE_publish_current_batch_llep_evidence_to_generation_control(
        const void *runtime_layers,
        int layer_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int movement_layer_count_index,
        int non_owner_assignment_layer_count_index,
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
        const int32_t *verifier_input_tokens,
        int verifier_input_token_stride,
        sampling_math::MTPCommittedVerifierIdentityRecord *
            out_committed_verifier_identity,
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
            const void *next_leading_committed_output_count_device)
    {
        if (device_id >= device_count_ || device_id < 0 ||
            !verify_tokens_device || !draft_tokens_device ||
            !active_verifier_row_count_device || !stop_tokens_device ||
            !next_leading_committed_output_count_device ||
            compare_row_count < 0 ||
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
            static_cast<const int *>(
                next_leading_committed_output_count_device),
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

    bool CUDABackend::enqueuePublishSerialDecodeCommitBoundary(
        void *decode_rounds_committed_device,
        void *decode_rounds_until_maintenance_device,
        void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !decode_rounds_committed_device ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device ||
            !decode_boundary_advanced_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_publish_serial_decode_commit_boundary(
            static_cast<uint32_t *>(decode_rounds_committed_device),
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueAcknowledgeDecodeCommitBoundary(
        void *decode_rounds_until_maintenance_device,
        void *maintenance_due_device,
        void *decode_boundary_advanced_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device ||
            !decode_boundary_advanced_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_acknowledge_decode_commit_boundary(
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueInitializeDeviceMoERebalanceDispatchTicket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        uint32_t participant_id,
        uint32_t participant_count,
        void *ticket_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            session_epoch == 0u || workspace_generation == 0u ||
            participant_count == 0u || participant_id >= participant_count ||
            !ticket_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_initialize_device_moe_rebalance_dispatch_ticket(
            session_epoch,
            workspace_generation,
            participant_id,
            participant_count,
            static_cast<DeviceMoERebalanceDispatchTicket *>(ticket_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePublishDeviceMoERebalanceDispatchTicket(
        const void *controller_magic_device,
        const void *controller_version_device,
        const void *controller_error_device,
        const void *decode_rounds_committed_device,
        const void *decode_rounds_until_maintenance_device,
        const void *maintenance_due_device,
        const void *decode_boundary_advanced_device,
        void *ticket_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !controller_magic_device || !controller_version_device ||
            !controller_error_device || !decode_rounds_committed_device ||
            !decode_rounds_until_maintenance_device ||
            !maintenance_due_device || !decode_boundary_advanced_device ||
            !ticket_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_publish_device_moe_rebalance_dispatch_ticket(
            static_cast<const uint32_t *>(controller_magic_device),
            static_cast<const uint32_t *>(controller_version_device),
            static_cast<const uint32_t *>(controller_error_device),
            static_cast<const uint32_t *>(decode_rounds_committed_device),
            static_cast<const uint32_t *>(
                decode_rounds_until_maintenance_device),
            static_cast<const uint32_t *>(maintenance_due_device),
            static_cast<const uint32_t *>(decode_boundary_advanced_device),
            static_cast<DeviceMoERebalanceDispatchTicket *>(ticket_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueInitializeDeviceGeneration(
        int request_count,
        int max_new_tokens,
        const sampling_math::DeviceGenerationDepthPolicy &depth_policy,
        sampling_math::DeviceGenerationLeadingRowDisposition
            initial_leading_row_disposition,
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
            !sampling_math::valid_device_generation_leading_row_disposition(
                initial_leading_row_disposition) ||
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
            initial_leading_row_disposition,
            response_token_stride,
            control_stride,
            static_cast<int *>(control_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueueInitializeDeviceGenerationDispatchTicket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        void *control_device,
        int control_stride,
        int request_count,
        void *dispatch_tickets_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            session_epoch == 0 || workspace_generation == 0 ||
            !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !dispatch_tickets_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_initialize_device_generation_dispatch_tickets(
            session_epoch,
            workspace_generation,
            static_cast<int *>(control_device),
            control_stride,
            request_count,
            static_cast<sampling_math::DeviceGenerationDispatchTicket *>(
                dispatch_tickets_device),
            device_id,
            stream);
    }

    bool CUDABackend::enqueuePublishDeviceGenerationDispatchTickets(
        void *control_device,
        int control_stride,
        int request_count,
        const void *maintenance_due_device,
        void *dispatch_tickets_device,
        int device_id,
        void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ || !control_device ||
            control_stride < sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !dispatch_tickets_device || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaOps_publish_device_generation_dispatch_tickets(
            static_cast<int *>(control_device),
            control_stride,
            request_count,
            static_cast<const uint32_t *>(maintenance_due_device),
            static_cast<sampling_math::DeviceGenerationDispatchTicket *>(
                dispatch_tickets_device),
            device_id,
            stream);
    }

    bool CUDABackend::
        enqueuePublishMoECurrentBatchLLEPEvidenceToGenerationControl(
            const void *runtime_layers_device,
            int layer_count,
            void *generation_control_device,
            int generation_control_stride,
            int request_count,
            int device_id,
            void *stream)
    {
        if (device_id < 0 || device_id >= device_count_ ||
            !runtime_layers_device || layer_count <= 0 ||
            !generation_control_device ||
            generation_control_stride <
                sampling_math::kDeviceGenerationControlCount ||
            request_count <= 0 || !stream)
        {
            return false;
        }

        CUDA_CHECK_OR_THROW(cudaSetDevice(device_id));
        return cudaMoE_publish_current_batch_llep_evidence_to_generation_control(
            runtime_layers_device,
            layer_count,
            static_cast<int *>(generation_control_device),
            generation_control_stride,
            request_count,
            sampling_math::
                kDeviceGenerationControlCurrentBatchLLEPMovementLayerCount,
            sampling_math::
                kDeviceGenerationControlCurrentBatchLLEPNonOwnerAssignmentLayerCount,
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
        void *out_next_verifier_condition_tokens_device,
        const void *verifier_input_tokens_device,
        int verifier_input_token_stride,
        void *out_committed_verifier_identity_device)
    {
        const bool has_verifier_identity_binding =
            verifier_input_tokens_device != nullptr ||
            verifier_input_token_stride != 0 ||
            out_committed_verifier_identity_device != nullptr;
        const bool has_complete_verifier_identity_binding =
            verifier_input_tokens_device != nullptr &&
            verifier_input_token_stride > 0 &&
            out_committed_verifier_identity_device != nullptr;
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
            (has_verifier_identity_binding &&
             !has_complete_verifier_identity_binding) ||
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
            static_cast<const int32_t *>(verifier_input_tokens_device),
            verifier_input_token_stride,
            static_cast<
                sampling_math::MTPCommittedVerifierIdentityRecord *>(
                out_committed_verifier_identity_device),
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

    bool CUDABackend::supportsStreamTimelineSignal32(int device_id) const
    {
        if (device_id < 0 || device_id >= device_count_)
            return false;

        /*
         * CUDA's current 32-bit cuStreamWaitValue32/cuStreamWriteValue32
         * contract has no corresponding current capability attribute.  The
         * similarly named CAN_USE_STREAM_MEM_OPS_V1 attribute describes the
         * deprecated v1 batch-mem-op ABI and returns zero on current drivers;
         * using it would incorrectly reject devices which implement the
         * current pair (including Ampere).  Prove that the exact ordinal is
         * addressable here, then let allocation plus the real queued
         * wait/write calls fail closed if a driver cannot execute them.
         */
        CUdevice device = 0;
        return cuInit(0) == CUDA_SUCCESS &&
               cuDeviceGet(&device, device_id) == CUDA_SUCCESS;
    }

    void *CUDABackend::allocateStreamTimelineSignal32(int device_id)
    {
        if (!supportsStreamTimelineSignal32(device_id) ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::allocateStreamTimelineSignal32] unsupported or invalid device="
                      << device_id);
            return nullptr;
        }

        void *signal = nullptr;
        const cudaError_t error = cudaMalloc(&signal, sizeof(uint32_t));
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::allocateStreamTimelineSignal32] cudaMalloc failed: "
                      << cudaGetErrorString(error));
            return nullptr;
        }
        return signal;
    }

    void CUDABackend::freeStreamTimelineSignal32(void *signal, int device_id)
    {
        if (!signal)
            return;
        CUDA_WARN_IF_FAIL(cudaSetDevice(device_id));
        CUDA_WARN_IF_FAIL(cudaFree(signal));
    }

    bool CUDABackend::streamWaitTimelineSignal32(
        void *stream,
        void *signal,
        uint32_t value,
        int device_id)
    {
        const cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::streamWaitTimelineSignal32");
        if (!signal || device_id < 0 || device_id >= device_count_ ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamWaitTimelineSignal32] invalid ownership"
                      << " device=" << device_id << " signal=" << signal);
            return false;
        }

        const CUresult result = cuStreamWaitValue32(
            reinterpret_cast<CUstream>(cuda_stream),
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(signal)),
            value,
            CU_STREAM_WAIT_VALUE_GEQ);
        if (result != CUDA_SUCCESS)
        {
            const char *name = nullptr;
            const char *description = nullptr;
            (void)cuGetErrorName(result, &name);
            (void)cuGetErrorString(result, &description);
            LOG_ERROR("[CUDABackend::streamWaitTimelineSignal32] cuStreamWaitValue32 failed: "
                      << (name ? name : "unknown") << " ("
                      << (description ? description : "no description") << ")");
            return false;
        }
        return true;
    }

    bool CUDABackend::streamPublishTimelineSignal32(
        void *stream,
        void *signal,
        uint32_t value,
        int device_id)
    {
        const cudaStream_t cuda_stream =
            requireExplicitStream(stream, "CUDABackend::streamPublishTimelineSignal32");
        if (!signal || device_id < 0 || device_id >= device_count_ ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamPublishTimelineSignal32] invalid ownership"
                      << " device=" << device_id << " signal=" << signal);
            return false;
        }

        // The default write mode includes the device memory fence that makes
        // every earlier H2D publication visible before consumers are released.
        const CUresult result = cuStreamWriteValue32(
            reinterpret_cast<CUstream>(cuda_stream),
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(signal)),
            value,
            CU_STREAM_WRITE_VALUE_DEFAULT);
        if (result != CUDA_SUCCESS)
        {
            const char *name = nullptr;
            const char *description = nullptr;
            (void)cuGetErrorName(result, &name);
            (void)cuGetErrorString(result, &description);
            LOG_ERROR("[CUDABackend::streamPublishTimelineSignal32] cuStreamWriteValue32 failed: "
                      << (name ? name : "unknown") << " ("
                      << (description ? description : "no description") << ")");
            return false;
        }
        return true;
    }

    bool CUDABackend::supportsStreamTimelineSignal64(int device_id) const
    {
        if (device_id < 0 || device_id >= device_count_)
            return false;
        CUdevice device = 0;
        return cuInit(0) == CUDA_SUCCESS &&
               cuDeviceGet(&device, device_id) == CUDA_SUCCESS;
    }

    void *CUDABackend::allocateStreamTimelineSignal64(int device_id)
    {
        if (!supportsStreamTimelineSignal64(device_id) ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::allocateStreamTimelineSignal64] unsupported or invalid device="
                      << device_id);
            return nullptr;
        }
        void *signal = nullptr;
        const cudaError_t error = cudaMalloc(&signal, sizeof(uint64_t));
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::allocateStreamTimelineSignal64] cudaMalloc failed: "
                      << cudaGetErrorString(error));
            return nullptr;
        }
        return signal;
    }

    void CUDABackend::freeStreamTimelineSignal64(
        void *signal,
        int device_id)
    {
        if (!signal)
            return;
        CUDA_WARN_IF_FAIL(cudaSetDevice(device_id));
        CUDA_WARN_IF_FAIL(cudaFree(signal));
    }

    bool CUDABackend::streamWaitTimelineSignal64(
        void *stream,
        void *signal,
        uint64_t value,
        int device_id)
    {
        const cudaStream_t cuda_stream = requireExplicitStream(
            stream, "CUDABackend::streamWaitTimelineSignal64");
        if (!signal || device_id < 0 || device_id >= device_count_ ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamWaitTimelineSignal64] invalid ownership device="
                      << device_id << " signal=" << signal);
            return false;
        }
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaError_t capture_query =
            cudaStreamIsCapturing(cuda_stream, &capture_status);
        if (capture_query != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamWaitTimelineSignal64] cudaStreamIsCapturing failed: "
                      << cudaGetErrorString(capture_query));
            return false;
        }
        if (capture_status == cudaStreamCaptureStatusActive)
        {
            return appendCUDAActiveCaptureTimelineWait64(
                cuda_stream, signal, value);
        }
        if (capture_status == cudaStreamCaptureStatusInvalidated)
        {
            LOG_ERROR("[CUDABackend::streamWaitTimelineSignal64] capture stream is invalidated");
            return false;
        }
        const CUresult result = cuStreamWaitValue64(
            reinterpret_cast<CUstream>(cuda_stream),
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(signal)),
            value,
            CU_STREAM_WAIT_VALUE_GEQ);
        if (result != CUDA_SUCCESS)
        {
            const char *name = nullptr;
            const char *description = nullptr;
            (void)cuGetErrorName(result, &name);
            (void)cuGetErrorString(result, &description);
            LOG_ERROR("[CUDABackend::streamWaitTimelineSignal64] cuStreamWaitValue64 failed: "
                      << (name ? name : "unknown") << " ("
                      << (description ? description : "no description") << ")");
            return false;
        }
        return true;
    }

    bool CUDABackend::streamPublishTimelineSignal64(
        void *stream,
        void *signal,
        uint64_t value,
        int device_id)
    {
        const cudaStream_t cuda_stream = requireExplicitStream(
            stream, "CUDABackend::streamPublishTimelineSignal64");
        if (!signal || device_id < 0 || device_id >= device_count_ ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamPublishTimelineSignal64] invalid ownership device="
                      << device_id << " signal=" << signal);
            return false;
        }
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        const cudaError_t capture_query =
            cudaStreamIsCapturing(cuda_stream, &capture_status);
        if (capture_query != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::streamPublishTimelineSignal64] cudaStreamIsCapturing failed: "
                      << cudaGetErrorString(capture_query));
            return false;
        }
        if (capture_status == cudaStreamCaptureStatusActive)
        {
            return appendCUDAActiveCaptureTimelinePublish64(
                cuda_stream, signal, value);
        }
        if (capture_status == cudaStreamCaptureStatusInvalidated)
        {
            LOG_ERROR("[CUDABackend::streamPublishTimelineSignal64] capture stream is invalidated");
            return false;
        }
        const CUresult result = cuStreamWriteValue64(
            reinterpret_cast<CUstream>(cuda_stream),
            static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(signal)),
            value,
            CU_STREAM_WRITE_VALUE_DEFAULT);
        if (result != CUDA_SUCCESS)
        {
            const char *name = nullptr;
            const char *description = nullptr;
            (void)cuGetErrorName(result, &name);
            (void)cuGetErrorString(result, &description);
            LOG_ERROR("[CUDABackend::streamPublishTimelineSignal64] cuStreamWriteValue64 failed: "
                      << (name ? name : "unknown") << " ("
                      << (description ? description : "no description") << ")");
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
            cudaStreamCaptureStatus capture_status =
                cudaStreamCaptureStatusInvalidated;
            const cudaError_t capture_query =
                cudaStreamIsCapturing(cuda_stream, &capture_status);
            cudaPointerAttributes destination_attributes{};
            const cudaError_t destination_query =
                cudaPointerGetAttributes(&destination_attributes, dst);
            LOG_ERROR(
                "[CUDABackend::hostToDeviceOnStream] failed: "
                << cudaGetErrorString(err)
                << " device=" << device_id
                << " bytes=" << bytes
                << " dst=" << dst
                << " src=" << src
                << " stream=" << stream
                << " capture_query=" << cudaGetErrorString(capture_query)
                << " capture_status=" << static_cast<int>(capture_status)
                << " destination_query="
                << cudaGetErrorString(destination_query)
                << " destination_type="
                << (destination_query == cudaSuccess
                        ? static_cast<int>(destination_attributes.type)
                        : -1)
                << " destination_device="
                << (destination_query == cudaSuccess
                        ? destination_attributes.device
                        : -1));
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

    bool CUDABackend::deviceToMappedHostByKernelOnStream(
        void *dst,
        const void *src,
        size_t bytes,
        int device_id,
        void *stream)
    {
        cudaStream_t cuda_stream = requireExplicitStream(
            stream,
            "CUDABackend::deviceToMappedHostByKernelOnStream");
        if (!dst || !src || bytes == 0u ||
            device_id < 0 || device_id >= device_count_ ||
            !setDevice(device_id))
        {
            return false;
        }

        const bool vector_aligned =
            reinterpret_cast<std::uintptr_t>(dst) % alignof(uint4) == 0u &&
            reinterpret_cast<std::uintptr_t>(src) % alignof(uint4) == 0u &&
            bytes % sizeof(uint4) == 0u;
        if (vector_aligned)
        {
            const std::size_t vector_count = bytes / sizeof(uint4);
            mappedHostCopyVectorKernel<<<
                mappedHostCopyBlocks(vector_count),
                kMappedHostCopyThreads,
                0u,
                cuda_stream>>>(
                static_cast<uint4 *>(dst),
                static_cast<const uint4 *>(src),
                vector_count);
        }
        else
        {
            mappedHostCopyByteKernel<<<
                mappedHostCopyBlocks(bytes),
                kMappedHostCopyThreads,
                0u,
                cuda_stream>>>(
                static_cast<std::uint8_t *>(dst),
                static_cast<const std::uint8_t *>(src),
                bytes);
        }
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::deviceToMappedHostByKernelOnStream] failed: "
                      << cudaGetErrorString(error));
            return false;
        }
        return true;
    }

    bool CUDABackend::enqueueMappedTransferProgressClaims(
        const MappedTransferProgressCommand *commands,
        MappedTransferProgressClaim *claims,
        size_t slot_capacity,
        int device_id,
        void *stream)
    {
        const cudaStream_t cuda_stream = requireExplicitStream(
            stream,
            "CUDABackend::enqueueMappedTransferProgressClaims");
        if (!commands || !claims || slot_capacity == 0u ||
            slot_capacity > std::numeric_limits<unsigned int>::max() ||
            device_id < 0 || device_id >= device_count_ ||
            !setDevice(device_id))
        {
            return false;
        }

        mappedTransferProgressClaimKernel<<<
            static_cast<unsigned int>(slot_capacity),
            1u,
            0u,
            cuda_stream>>>(commands, claims, slot_capacity);
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::enqueueMappedTransferProgressClaims] "
                      "claim launch failed: " << cudaGetErrorString(error));
            return false;
        }
        return true;
    }

    bool CUDABackend::enqueueMappedTransferProgressCopies(
        const MappedTransferProgressClaim *claims,
        MappedTransferProgressCompletion *completions,
        size_t slot_capacity,
        size_t maximum_bytes,
        int device_id,
        void *stream)
    {
        const cudaStream_t cuda_stream = requireExplicitStream(
            stream,
            "CUDABackend::enqueueMappedTransferProgressCopies");
        if (!claims || !completions || slot_capacity == 0u ||
            maximum_bytes == 0u ||
            slot_capacity > std::numeric_limits<unsigned int>::max() ||
            device_id < 0 || device_id >= device_count_ ||
            !setDevice(device_id))
        {
            return false;
        }

        mappedTransferProgressCopyKernel<<<
            static_cast<unsigned int>(slot_capacity),
            kMappedHostCopyThreads,
            0u,
            cuda_stream>>>(
                claims,
                completions,
                slot_capacity,
                maximum_bytes);
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::enqueueMappedTransferProgressCopies] "
                      "copy launch failed: " << cudaGetErrorString(error));
            return false;
        }
        return true;
    }

    // ====================================================================
    // Pinned Host Memory
    // ====================================================================

    void *CUDABackend::allocatePinned(size_t bytes, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
        if (device_id < 0 || device_id >= device_count_ ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            (void)cudaGetLastError();
            LOG_ERROR("[CUDABackend::allocatePinned] invalid CUDA device "
                      << device_id);
            return nullptr;
        }
        void *ptr = nullptr;
        cudaError_t err = cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault);
        if (err != cudaSuccess)
        {
            LOG_ERROR("[CUDABackend::allocatePinned] cudaHostAlloc(" << bytes
                      << ") failed: " << cudaGetErrorString(err));
            return nullptr;
        }
        cudaTrackedHostRegistrations().emplace(
            ptr,
            CUDAHostRegistrationRecord{
                .bytes = bytes,
                .device_id = device_id,
            });
        return ptr;
    }

    void CUDABackend::freePinned(void *ptr, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            cudaRuntimeResourceLifecycleMutex());
        if (!ptr)
            return;
        if (device_id < 0 || device_id >= device_count_ ||
            cudaSetDevice(device_id) != cudaSuccess)
        {
            (void)cudaGetLastError();
            LOG_ERROR("[CUDABackend::freePinned] invalid CUDA device "
                      << device_id << " for ptr=" << ptr);
            return;
        }
        const cudaError_t error = cudaFreeHost(ptr);
        if (error != cudaSuccess)
        {
            LOG_WARN("[CUDABackend::freePinned] cudaFreeHost failed for ptr="
                     << ptr << ": " << cudaGetErrorString(error));
            (void)cudaGetLastError();
            return;
        }
        cudaTrackedHostRegistrations().erase(ptr);
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
