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
#include "HipDeviceGuard.h"
#include "HIPGraphTimelineKernels.h"
#include "../../utils/Logger.h"
#include "../../utils/PerfStatsCollector.h"
#include "../../utils/VramBillOfMaterials.h"
#include "../../execution/moe/DeviceMoERebalanceABI.h"
#include "../../execution/mtp/MTPVerifierOutcomeGraph.h"
#include "../../transfer/MappedTransferProgressABI.h"
#include "../../kernels/common/SamplingMath.h"
#include "../../kernels/rocm/ops/ROCmRowSelectKernels.h"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <dlfcn.h> // For HSA runtime loading
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <limits>
#include <thread>
#include <cstdint>

namespace llaminar2
{
    extern "C" bool llaminar2_retireROCmTensorValidatorRuntimeGeneration(
        int device_id);

    extern "C" bool rocmOps_vector_add_inplace_fp32(
        float *output,
        const float *input,
        size_t count,
        int device_idx,
        void *stream);

    namespace
    {
        constexpr std::uintptr_t kDeviceAllocationAlignment = 256;
        constexpr unsigned int kMappedHostCopyThreads = 256u;
        constexpr unsigned int kMappedHostCopyMaximumBlocks = 4096u;

        /**
         * @return Mutex serializing HIP resource mutation against runtime reset.
         *
         * All tracked device allocations and host registrations take this lock
         * before entering the HIP runtime. The exclusive generation reset holds
         * it across preflight, hipDeviceReset, and new-generation publication.
         */
        std::mutex &rocmRuntimeResourceLifecycleMutex()
        {
            static auto *mutex = new std::mutex();
            return *mutex;
        }

        /** @return System-scope acquire load from a node-local mapped word. */
        __device__ __forceinline__ std::uint64_t mappedSystemAcquire64(
            const std::uint64_t *value)
        {
            return __hip_atomic_load(
                value, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
        }

        /** @brief System-scope release store into a node-local mapped word. */
        __device__ __forceinline__ void mappedSystemRelease64(
            std::uint64_t *value,
            std::uint64_t published)
        {
            __hip_atomic_store(
                value,
                published,
                __ATOMIC_RELEASE,
                __HIP_MEMORY_SCOPE_SYSTEM);
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

        /** @brief Snapshot one mapped command per block into ordinary VRAM. */
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
         * @brief Copy one active command per block and system-publish completion.
         *
         * One workgroup per slot keeps independent ExpertOverlay movements
         * concurrent without a global counter. Vector lanes cover the aligned
         * body while a byte tail preserves arbitrary-length totality.
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
                /* Completion lives in host-mapped memory. Every lane reading it
                 * independently can observe a different coherence instant: a
                 * subset may return while its peers reach the terminal block
                 * barrier, deadlocking this retained replay forever. One
                 * system-acquire snapshot and a shared broadcast make the
                 * active/inactive branch uniform for the whole workgroup. */
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
                    if (static_cast<hipError_t>(
                            HipDeviceGuard::forceSetDevice(saved_device_)) !=
                        hipSuccess)
                    {
                        LOG_ERROR("[ROCmBackend] Could not restore owning HIP device "
                                  << saved_device_);
                        std::terminate();
                    }
                }
            }

            HipDeviceSaveRestore(const HipDeviceSaveRestore &) = delete;
            HipDeviceSaveRestore &operator=(const HipDeviceSaveRestore &) = delete;

            /** @return Whether the caller's exact HIP device was captured. */
            [[nodiscard]] bool valid() const noexcept { return valid_; }

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
        penalty_buffers_.resize(
            static_cast<size_t>(std::max(device_count_, 0)));
        runtime_generations_.assign(
            static_cast<size_t>(std::max(device_count_, 0)), 1u);
    }

    ROCmBackend::~ROCmBackend()
    {
        // hipDeviceReset() intentionally omitted - managed by HIP runtime
    }

    // ====================================================================
    // Stream Resolution Helper
    // ====================================================================

    /**
     * @brief Convert an opaque execution stream after enforcing explicit ownership.
     *
     * HIP's null stream discards the graph's producer/consumer relationship.
     * Every executable backend API therefore fails at this common boundary
     * before it can enqueue work with ambiguous ordering.
     */
    static hipStream_t requireExplicitStream(void *stream, const char *operation)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                std::string(operation ? operation : "ROCmBackend operation") +
                " requires an explicit non-null HIP stream");
        }
        return static_cast<hipStream_t>(stream);
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

        if (!deviceToHostOnStream(dst, src, bytes, device_id, stream))
            return false;

        /*
         * The compatibility method returns host-owned bytes. Wait only an
         * event recorded after this copy, never the entire producer stream.
         */
        void *const completion = createEvent(device_id);
        if (!completion)
            return false;
        const bool recorded = recordEvent(completion, device_id, stream);
        const bool completed = recorded && waitForEvent(completion, device_id);
        destroyEvent(completion, device_id);
        return completed;
    }

    bool ROCmBackend::deviceToHostFast(void *dst, const void *src, size_t bytes, int device_id, void *stream)
    {
        // Fast path: skip pointer validation and device save/restore.
        // Caller guarantees src is valid device memory and GPU work is complete.
        if (device_id >= device_count_ || device_id < 0)
        {
            return false;
        }
        if (!deviceToHostOnStream(dst, src, bytes, device_id, stream))
            return false;

        void *const completion = createEvent(device_id);
        if (!completion)
            return false;
        const bool recorded = recordEvent(completion, device_id, stream);
        const bool completed = recorded && waitForEvent(completion, device_id);
        destroyEvent(completion, device_id);
        return completed;
    }

    bool ROCmBackend::pinHostMemory(void *ptr, size_t bytes, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
        HipDeviceSaveRestore device_guard;
        if (!ptr || bytes == 0 || device_id < 0 || device_id >= device_count_ ||
            static_cast<hipError_t>(HipDeviceGuard::forceSetDevice(device_id)) != hipSuccess)
        {
            LOG_WARN("[ROCmBackend::pinHostMemory] invalid registration for ROCm:"
                     << device_id << " ptr=" << ptr << " bytes=" << bytes);
            return false;
        }
        hipError_t err = hipHostRegister(ptr, bytes, hipHostRegisterDefault);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmBackend::pinHostMemory] hipHostRegister failed for "
                     << bytes << " bytes on ROCm:" << device_id << ": "
                     << hipGetErrorString(err));
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations()[ptr] = device_id;
        }
        return true;
    }

    bool ROCmBackend::unpinHostMemory(void *ptr, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
        HipDeviceSaveRestore device_guard;
        if (!ptr || device_id < 0 || device_id >= device_count_ ||
            static_cast<hipError_t>(HipDeviceGuard::forceSetDevice(device_id)) != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::unpinHostMemory] invalid retirement for ROCm:"
                      << device_id << " ptr=" << ptr);
            return false;
        }
        hipError_t err = hipHostUnregister(ptr);
        if (err != hipSuccess)
        {
            LOG_WARN("[ROCmBackend::unpinHostMemory] hipHostUnregister failed: "
                     << hipGetErrorString(err) << " owner=ROCm:" << device_id
                     << " ptr=" << ptr);
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations().erase(ptr);
        }
        return true;
    }

    bool ROCmBackend::registerExternalMappedHostMemory(
        void *ptr,
        size_t bytes,
        int registration_device_id,
        MappedHostRegistrationScope scope)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
        if (!ptr || bytes == 0u || registration_device_id < 0 ||
            registration_device_id >= device_count_)
        {
            LOG_ERROR("[ROCmBackend::registerExternalMappedHostMemory] invalid region or registration device="
                      << registration_device_id);
            return false;
        }
        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(registration_device_id) != hipSuccess)
            return false;
        /* Portable registration mutates every ROCm page table and is
         * materially more expensive on multi-GPU IOMMU hosts. In particular,
         * thousands of one-device retained-graph tickets used to create an
         * iova_depot backlog and IH-ring overflow storm. Only a genuinely
         * shared same-family region is allowed to pay that cost. */
        const unsigned int flags = hipHostRegisterMapped |
                                   hipExtHostRegisterUncached |
                                   (scope == MappedHostRegistrationScope::BackendPortable
                                        ? hipHostRegisterPortable
                                        : 0u);
        const hipError_t error = hipHostRegister(ptr, bytes, flags);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::registerExternalMappedHostMemory] hipHostRegister failed for "
                      << bytes << " bytes scope=" << to_string(scope)
                      << ": " << hipGetErrorString(error));
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations()[ptr] = registration_device_id;
        }
        return true;
    }

    bool ROCmBackend::externalMappedHostDevicePointer(
        void *host_ptr,
        int device_id,
        void **device_ptr)
    {
        if (device_ptr)
            *device_ptr = nullptr;
        if (!host_ptr || !device_ptr || device_id < 0 ||
            device_id >= device_count_)
        {
            LOG_ERROR("[ROCmBackend::externalMappedHostDevicePointer] invalid mapped region or device="
                      << device_id);
            return false;
        }
        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(device_id) != hipSuccess)
            return false;
        const hipError_t error = hipHostGetDevicePointer(
            device_ptr, host_ptr, 0u);
        if (error != hipSuccess || !*device_ptr)
        {
            LOG_ERROR("[ROCmBackend::externalMappedHostDevicePointer] hipHostGetDevicePointer failed for device="
                      << device_id << ": " << hipGetErrorString(error));
            *device_ptr = nullptr;
            return false;
        }
        return true;
    }

    bool ROCmBackend::unregisterExternalMappedHostMemory(
        void *ptr,
        int registration_device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
        if (!ptr || registration_device_id < 0 ||
            registration_device_id >= device_count_)
        {
            return false;
        }
        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(registration_device_id) != hipSuccess)
            return false;
        const hipError_t error = hipHostUnregister(ptr);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::unregisterExternalMappedHostMemory] hipHostUnregister failed: "
                      << hipGetErrorString(error));
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations().erase(ptr);
        }
        return true;
    }

    // Forward declaration for HIP argmax kernel (implemented in ROCmArgmaxKernels.hip)
    extern "C" bool rocmOps_argmax_f32(
        const float *data, int n, float *out_value, int *out_index,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream);
    extern "C" bool rocmOps_argmax_f32_batched_rows(
        const float *data, int rows, int cols, int row_stride,
        float *out_values, int *out_indices,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool rocmOps_argmax_f32_batched_rows_publish_mtp_chain(
        const float *data, int rows, int cols, int row_stride,
        float *out_values, int *out_indices,
        int *chain_condition_tokens, int *chain_position_ids,
        int chain_position_increment,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool rocmOps_retain_mtp_first_transaction_draft_boundary(
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
    extern "C" bool rocmOps_configure_mtp_greedy_penalty_policy(
        MTPGreedyPenaltyPolicy *controls,
        float presence_penalty,
        float frequency_penalty,
        bool first_token_already_in_history,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_argmax_f32_batched_rows_mtp_penalties(
        const float *data, int rows, int cols, int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        float *out_values, int *out_indices,
        float *partial_vals, int *partial_idxs, int partial_capacity,
        int device_idx, void *stream, int output_stride);
    extern "C" bool rocmOps_apply_mtp_penalties_f32_rows(
        float *data, int rows, int cols, int row_stride,
        const int *verifier_input_tokens,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        const int *active_rows,
        int device_idx, void *stream);
    extern "C" bool rocmOps_apply_mtp_branch_penalties_f32_row(
        float *data, int cols,
        const int *first_condition_token,
        const int *prior_draft_tokens,
        int prior_draft_count,
        const int *generated_token_counts,
        const MTPGreedyPenaltyPolicy *policy,
        int device_idx, void *stream);
    extern "C" bool rocmOps_commit_mtp_greedy_penalty_history(
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

    bool ROCmBackend::argmaxF32(const void *data_device, int n, int device_id,
                                float *out_value, int *out_index, void *stream,
                                void *partial_vals, void *partial_idxs, int partial_capacity)
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
            bufs.allocated_count = 1;
        }

        // Launch kernel on device's managed stream
        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipStream_t s = requireExplicitStream(stream, "ROCmBackend::argmaxF32");
        // Pass the caller-supplied partial scratch through to the kernel wrapper.
        // The scratch is mandatory (arena-owned); the wrapper fails loud if it is
        // missing or undersized — there is no single-block fallback.
        if (!rocmOps_argmax_f32(
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

        // The argmax kernel and the two tiny D2H copies are enqueued on the
        // same stream. Stream ordering guarantees the copies observe the kernel
        // results, so only the final synchronize is needed.
        HIP_CHECK_OR_THROW(hipMemcpyAsync(out_value, bufs.value_ptr, sizeof(float), hipMemcpyDeviceToHost, s));
        HIP_CHECK_OR_THROW(hipMemcpyAsync(out_index, bufs.index_ptr, sizeof(int), hipMemcpyDeviceToHost, s));
        HIP_CHECK_OR_THROW(hipStreamSynchronize(s));

        return true;
    }

    bool ROCmBackend::argmaxF32BatchedRows(const void *data_device, int rows, int cols, int device_id,
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
            LOG_ERROR("[ROCmBackend::argmaxF32BatchedRows] missing arena-owned partial scratch "
                      << "(rows=" << rows << " capacity=" << partial_capacity << ")");
            return false;
        }

        if (argmax_buffers_.empty())
            argmax_buffers_.resize(device_count_);

        auto &bufs = argmax_buffers_[device_id];
        if (!bufs.value_ptr || bufs.allocated_count < rows)
        {
            HIP_CHECK_OR_THROW(hipSetDevice(device_id));
            if (bufs.value_ptr)
                HIP_WARN_IF_FAIL(hipFree(bufs.value_ptr));
            if (bufs.index_ptr)
                HIP_WARN_IF_FAIL(hipFree(bufs.index_ptr));
            bufs.value_ptr = nullptr;
            bufs.index_ptr = nullptr;
            bufs.allocated_count = 0;

            HIP_CHECK_OR_THROW(hipMalloc(&bufs.value_ptr, static_cast<size_t>(rows) * sizeof(float)));
            hipError_t err = hipMalloc(&bufs.index_ptr, static_cast<size_t>(rows) * sizeof(int));
            if (err != hipSuccess)
            {
                HIP_WARN_IF_FAIL(hipFree(bufs.value_ptr));
                bufs.value_ptr = nullptr;
                LOG_ERROR("[ROCmBackend::argmaxF32BatchedRows] hipMalloc indices failed: "
                          << hipGetErrorString(err));
                return false;
            }
            bufs.allocated_count = rows;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipStream_t s =
            requireExplicitStream(stream, "ROCmBackend::argmaxF32BatchedRows");
        {
            PerfStatsCollector::ScopedTimer timer(
                "backend", "rocm_argmax_f32_batched_rows_launch", "decode");
            if (!rocmOps_argmax_f32_batched_rows(
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
                "backend", "rocm_argmax_f32_batched_rows_d2h_enqueue", "decode");
            HIP_CHECK_OR_THROW(hipMemcpyAsync(out_values,
                                              bufs.value_ptr,
                                              static_cast<size_t>(rows) * sizeof(float),
                                              hipMemcpyDeviceToHost,
                                              s));
            HIP_CHECK_OR_THROW(hipMemcpyAsync(out_indices,
                                              bufs.index_ptr,
                                              static_cast<size_t>(rows) * sizeof(int),
                                              hipMemcpyDeviceToHost,
                                              s));
        }
        {
            PerfStatsCollector::ScopedTimer timer(
                "backend", "rocm_argmax_f32_batched_rows_sync", "decode");
            HIP_CHECK_OR_THROW(hipStreamSynchronize(s));
        }
        return true;
    }

    bool ROCmBackend::enqueueArgmaxF32BatchedRowsDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend", "rocm_argmax_f32_batched_rows_device_launch", "decode");
        return rocmOps_argmax_f32_batched_rows(
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
            static_cast<hipStream_t>(stream),
            output_stride);
    }

    bool ROCmBackend::enqueueArgmaxF32BatchedRowsAndPublishMTPChainDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "rocm_argmax_f32_mtp_chain_publication_launch",
            "decode");
        return rocmOps_argmax_f32_batched_rows_publish_mtp_chain(
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

    bool ROCmBackend::enqueueRetainMTPFirstTransactionDraftBoundaryDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "rocm_mtp_first_transaction_draft_boundary_diagnostic_launch",
            "decode");
        return rocmOps_retain_mtp_first_transaction_draft_boundary(
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

    bool ROCmBackend::enqueueConfigureMTPGreedyPenaltyPolicyDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_configure_mtp_greedy_penalty_policy(
            static_cast<MTPGreedyPenaltyPolicy *>(controls_device),
            presence_penalty,
            frequency_penalty,
            first_token_already_in_history,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueArgmaxF32BatchedRowsWithMTPPenaltiesDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "rocm_mtp_penalty_argmax_batched_rows_device_launch",
            "decode");
        return rocmOps_argmax_f32_batched_rows_mtp_penalties(
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

    bool ROCmBackend::enqueueApplyMTPPenaltiesToF32RowsDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "rocm_mtp_penalty_logit_rows_device_launch",
            "decode");
        return rocmOps_apply_mtp_penalties_f32_rows(
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

    bool ROCmBackend::enqueueApplyMTPBranchPenaltiesToF32RowDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        PerfStatsCollector::ScopedTimer timer(
            "backend",
            "rocm_mtp_branch_penalty_logit_row_device_launch",
            "decode");
        return rocmOps_apply_mtp_branch_penalties_f32_row(
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

    bool ROCmBackend::enqueueCommitMTPGreedyPenaltyHistoryDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_commit_mtp_greedy_penalty_history(
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

    // Forward declaration for HIP top-k kernel (implemented in ROCmSamplingKernels.hip)
    extern "C" bool rocmOps_topk_f32(
        const float *data, int n, int k, float *out_values, int *out_indices,
        int device_idx, void *stream);
    extern "C" bool rocmOps_sample_topk_topp_f32(
        const float *data, int n, int k, float top_p, float temperature,
        unsigned long long rng_seed, unsigned long long rng_offset,
        int *out_token, int device_idx, void *stream);
    extern "C" bool rocmOps_publish_int32_control_scalar(
        int32_t value,
        int32_t *out_value,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_topk_topp_distribution_f32(
        const float *data, int n, int k, float top_p, float temperature,
        int *out_token_ids, float *out_probs,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        int device_idx, void *stream);
    extern "C" bool rocmOps_topk_topp_distributions_f32(
        const float *data, int row_count, int n, int row_stride, int k,
        float top_p, float temperature,
        int *out_token_ids, int out_stride, float *out_probs,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        const int *active_rows,
        int device_idx, void *stream);
    extern "C" bool rocmOps_topk_topp_processed_logits_f32(
        const float *data, int row_count, int n, int row_stride, int k,
        float top_p, float temperature,
        float *out_logits, int out_row_stride,
        float *scratch_values, int *scratch_indices, int scratch_capacity,
        int device_idx, void *stream);
    extern "C" bool rocmOps_speculative_verify_distribution_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int draft_token,
        unsigned long long accept_seed, unsigned long long accept_offset,
        unsigned long long residual_seed, unsigned long long residual_offset,
        int *out_token, int *out_accepted,
        float *out_accept_probability, float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool rocmOps_sample_distribution_f32(
        const int *token_ids, const float *probs,
        int k, float threshold,
        int *out_token, float *out_probability,
        unsigned long long threshold_seed,
        const int *threshold_position,
        int threshold_position_offset,
        int device_idx, void *stream);
    extern "C" bool rocmOps_sample_processed_logits_f32(
        const float *logits,
        int vocab_size,
        int row_stride,
        float threshold,
        int *out_token,
        float *out_probability,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
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
    extern "C" bool rocmOps_softmax_processed_logits_f32(
        const float *logits,
        int row_count,
        int vocab_size,
        int row_stride,
        float *out_probabilities,
        int out_row_stride,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_softmax_sample_temperature_logits_f32(
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
    extern "C" bool rocmOps_scale_sample_temperature_logits_f32(
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
    extern "C" bool rocmOps_speculative_verify_distribution_threshold_f32(
        const int *target_token_ids, const float *target_probs,
        const int *draft_token_ids, const float *draft_probs,
        int k, int draft_token,
        float accept_threshold, float residual_threshold,
        int *out_token, int *out_accepted,
        float *out_accept_probability, float *out_accept_threshold,
        int device_idx, void *stream);
    extern "C" bool rocmOps_fill_inverse_exponential_samples_f32(
        float *out_samples,
        int row_count,
        int vocab_size,
        int row_stride,
        unsigned long long seed,
        int first_logical_position,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_speculative_verify_distribution_thresholds_batch_f32(
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
    extern "C" bool rocmOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
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
    extern "C" bool rocmOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
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
    extern "C" bool rocmOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
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
    extern "C" bool rocmOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
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
    extern "C" bool rocmOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
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
    extern "C" bool rocmOps_summarize_speculative_verify_batch(
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
    extern "C" bool rocmOps_summarize_speculative_verify_batch_device_first_token(
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
    rocmOps_summarize_speculative_verify_batch_device_generation_controls(
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
    rocmOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
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
    extern "C" bool rocmOps_summarize_greedy_speculative_verify_batch(
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
    rocmOps_summarize_greedy_speculative_verify_batch_device_controls(
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
    extern "C" bool rocmOps_advance_speculative_commit_boundary(
        const int *meta,
        int request_count,
        int meta_stride,
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_publish_serial_decode_commit_boundary(
        uint32_t *decode_rounds_committed,
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_acknowledge_decode_commit_boundary(
        uint32_t *decode_rounds_until_maintenance,
        uint32_t *maintenance_due,
        uint32_t *decode_boundary_advanced,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_initialize_device_moe_rebalance_dispatch_ticket(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        uint32_t participant_id,
        uint32_t participant_count,
        DeviceMoERebalanceDispatchTicket *ticket,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_publish_device_moe_rebalance_dispatch_ticket(
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
    extern "C" bool rocmOps_initialize_device_generation(
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
    extern "C" bool rocmOps_initialize_device_generation_dispatch_tickets(
        uint64_t session_epoch,
        uint64_t workspace_generation,
        int *control,
        int control_stride,
        int request_count,
        sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_publish_device_generation_dispatch_tickets(
        int *control,
        int control_stride,
        int request_count,
        const uint32_t *maintenance_due,
        sampling_math::DeviceGenerationDispatchTicket *tickets,
        int device_idx,
        void *stream);
    extern "C" bool
    hipMoE_publish_current_batch_llep_evidence_to_generation_control(
        const void *runtime_layers,
        int layer_count,
        int *generation_control,
        int generation_control_stride,
        int request_count,
        int movement_layer_count_index,
        int non_owner_assignment_layer_count_index,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_prepare_device_generation_transaction_budget(
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
    rocmOps_commit_device_generation_and_derive_speculative_publication_metadata(
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
    extern "C" bool rocmOps_derive_speculative_publication_metadata(
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
    rocmOps_derive_shifted_speculative_publication_metadata_from_primary(
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
    extern "C" bool rocmOps_prepare_speculative_shifted_kv_tokens(
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
    extern "C" bool rocmOps_prepare_mtp_batched_sidecar_inputs(
        const int32_t *condition_tokens,
        int condition_token_stride,
        const int32_t *base_positions,
        int position_offset,
        int request_count,
        int32_t *out_condition_tokens,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_prepare_mtp_verifier_position_ids(
        const int32_t *base_positions,
        int request_count,
        int padded_seq_len,
        int32_t *out_position_ids,
        int device_idx,
        void *stream);
    extern "C" bool rocmOps_prepare_mtp_verifier_geometry(
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
    extern "C" bool rocmOps_prepare_mtp_verifier_controlled_row(
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
    extern "C" bool rocmOps_initialize_mtp_device_logical_state(
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
        hipStream_t s = requireExplicitStream(stream, "ROCmBackend::topKF32");
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

    bool ROCmBackend::enqueueSampleTopKTopPF32Device(const void *data_device, int n,
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_sample_topk_topp_f32(
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

    bool ROCmBackend::enqueuePublishInt32ControlScalarDevice(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_publish_int32_control_scalar(
            value,
            static_cast<int32_t *>(out_value_device),
            device_id,
            stream);
    }

    bool ROCmBackend::sampleTopKTopPF32(const void *data_device, int n,
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
            HIP_CHECK_OR_THROW(hipSetDevice(device_id));
            hipError_t err = hipMalloc(&bufs.token_ptr, sizeof(int));
            if (err != hipSuccess)
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

        HIP_CHECK_OR_THROW(hipMemcpyAsync(out_token,
                                          bufs.token_ptr,
                                          sizeof(int),
                                          hipMemcpyDeviceToHost,
                                          static_cast<hipStream_t>(stream)));
        HIP_CHECK_OR_THROW(hipStreamSynchronize(static_cast<hipStream_t>(stream)));
        return true;
    }

    bool ROCmBackend::enqueueBuildTopKTopPDistributionF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_topk_topp_distribution_f32(
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

    bool ROCmBackend::enqueueBuildTopKTopPDistributionsF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_topk_topp_distributions_f32(
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

    bool ROCmBackend::enqueueBuildTopKTopPProcessedLogitsF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_topk_topp_processed_logits_f32(
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

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_f32(
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

    bool ROCmBackend::enqueueSampleDistributionF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_sample_distribution_f32(
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

    bool ROCmBackend::enqueueSampleProcessedLogitsF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_sample_processed_logits_f32(
            static_cast<const float *>(logits_device),
            vocab_size,
            row_stride,
            threshold,
            static_cast<int *>(out_token_device),
            static_cast<float *>(out_probability_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSampleProcessedLogitsF32DeviceIfSpeculativeBatchNeedsBonus(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_sample_processed_logits_if_speculative_batch_needs_bonus_f32(
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

    bool ROCmBackend::enqueueSoftmaxAndSampleTemperatureLogitsF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_softmax_sample_temperature_logits_f32(
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

    bool ROCmBackend::enqueueScaleAndSampleTemperatureLogitsF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_scale_sample_temperature_logits_f32(
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

    bool ROCmBackend::enqueueSoftmaxProcessedLogitsF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_softmax_processed_logits_f32(
            static_cast<const float *>(logits_device),
            row_count,
            vocab_size,
            row_stride,
            static_cast<float *>(out_probabilities_device),
            out_row_stride,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueFillInverseExponentialSamplesF32Device(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_fill_inverse_exponential_samples_f32(
            static_cast<float *>(out_samples_device),
            row_count,
            vocab_size,
            row_stride,
            static_cast<unsigned long long>(seed),
            first_logical_position,
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholds(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_threshold_f32(
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

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatch(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_thresholds_batch_f32(
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

    bool ROCmBackend::enqueueSpeculativeVerifyDistributionsF32DeviceThresholdsBatchDeviceTokens(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_distribution_thresholds_batch_device_tokens_f32(
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

    bool ROCmBackend::enqueueSpeculativeVerifyProcessedLogitsF32DeviceThresholdsBatchDeviceTokens(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_processed_logits_thresholds_batch_device_tokens_f32(
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

    bool ROCmBackend::enqueueSpeculativeVerifyProcessedTargetDraftProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_processed_target_draft_probabilities_thresholds_batch_device_tokens_f32(
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

    bool ROCmBackend::enqueueSpeculativeVerifyProcessedTargetDraftLogitsF32DeviceThresholdsBatchDeviceTokens(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_processed_target_draft_logits_thresholds_batch_device_tokens_f32(
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

    bool ROCmBackend::enqueueSpeculativeVerifyProbabilitiesF32DeviceThresholdsBatchDeviceTokens(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_speculative_verify_probabilities_thresholds_batch_device_tokens_f32(
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

    bool ROCmBackend::enqueueSummarizeSpeculativeVerifyBatch(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_speculative_verify_batch(
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

    bool ROCmBackend::enqueueSummarizeSpeculativeVerifyBatchDeviceFirstToken(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_speculative_verify_batch_device_first_token(
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

    bool ROCmBackend::
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_speculative_verify_batch_device_generation_controls(
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

    bool ROCmBackend::
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
        if (device_id < 0 || device_id >= device_count_)
            return false;

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_sample_and_summarize_serial_equivalent_speculative_batch_device_generation_controls(
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

    bool ROCmBackend::enqueueSummarizeGreedySpeculativeVerifyBatch(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_greedy_speculative_verify_batch(
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

    bool ROCmBackend::
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_summarize_greedy_speculative_verify_batch_device_controls(
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

    bool ROCmBackend::enqueueAdvanceSpeculativeCommitBoundary(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_advance_speculative_commit_boundary(
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

    bool ROCmBackend::enqueuePublishSerialDecodeCommitBoundary(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_publish_serial_decode_commit_boundary(
            static_cast<uint32_t *>(decode_rounds_committed_device),
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueAcknowledgeDecodeCommitBoundary(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_acknowledge_decode_commit_boundary(
            static_cast<uint32_t *>(decode_rounds_until_maintenance_device),
            static_cast<uint32_t *>(maintenance_due_device),
            static_cast<uint32_t *>(decode_boundary_advanced_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueueInitializeDeviceMoERebalanceDispatchTicket(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_initialize_device_moe_rebalance_dispatch_ticket(
            session_epoch,
            workspace_generation,
            participant_id,
            participant_count,
            static_cast<DeviceMoERebalanceDispatchTicket *>(ticket_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePublishDeviceMoERebalanceDispatchTicket(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_publish_device_moe_rebalance_dispatch_ticket(
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

    bool ROCmBackend::enqueueInitializeDeviceGeneration(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_initialize_device_generation(
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

    bool ROCmBackend::enqueueInitializeDeviceGenerationDispatchTicket(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_initialize_device_generation_dispatch_tickets(
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

    bool ROCmBackend::enqueuePublishDeviceGenerationDispatchTickets(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_publish_device_generation_dispatch_tickets(
            static_cast<int *>(control_device),
            control_stride,
            request_count,
            static_cast<const uint32_t *>(maintenance_due_device),
            static_cast<sampling_math::DeviceGenerationDispatchTicket *>(
                dispatch_tickets_device),
            device_id,
            stream);
    }

    bool ROCmBackend::
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

        HipDeviceGuard::setDevice(device_id);
        return hipMoE_publish_current_batch_llep_evidence_to_generation_control(
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

    bool ROCmBackend::enqueuePrepareDeviceGenerationTransactionBudget(
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_prepare_device_generation_transaction_budget(
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

    bool ROCmBackend::
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

        HipDeviceGuard::setDevice(device_id);
        return rocmOps_commit_device_generation_and_derive_speculative_publication_metadata(
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

    bool ROCmBackend::enqueueDeriveSpeculativePublicationMetadata(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_derive_speculative_publication_metadata(
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

    bool ROCmBackend::
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_derive_shifted_speculative_publication_metadata_from_primary(
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

    bool ROCmBackend::enqueuePrepareSpeculativeShiftedKVTokens(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_speculative_shifted_kv_tokens(
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

    bool ROCmBackend::enqueuePrepareMTPBatchedSidecarInputs(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_batched_sidecar_inputs(
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

    bool ROCmBackend::enqueuePrepareMTPVerifierPositionIds(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_verifier_position_ids(
            static_cast<const int32_t *>(base_positions_device),
            request_count,
            padded_seq_len,
            static_cast<int32_t *>(out_position_ids_device),
            device_id,
            stream);
    }

    bool ROCmBackend::enqueuePrepareMTPVerifierGeometry(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_verifier_geometry(
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

    bool ROCmBackend::enqueuePrepareMTPVerifierControlledRow(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_prepare_mtp_verifier_controlled_row(
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

    bool ROCmBackend::enqueueInitializeMTPDeviceLogicalState(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_initialize_mtp_device_logical_state(
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

    // Forward declaration for HIP penalty kernel (implemented in ROCmSamplingKernels.hip)
    extern "C" bool rocmOps_apply_logit_penalties_f32(
        float *logits, const int *token_ids, const float *penalties,
        int num_penalties, int vocab_size, int device_idx, void *stream);

    bool ROCmBackend::prepareLogitPenaltyWorkspace(
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
            LOG_ERROR("[ROCmBackend] Refusing to resize an active logit-penalty workspace");
            return false;
        }

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipError_t err =
            hipMalloc(&bufs.token_ids_ptr, vocab_size * sizeof(int));
        if (err != hipSuccess)
            return false;
        err = hipMalloc(
            &bufs.penalties_ptr,
            vocab_size * sizeof(float));
        if (err != hipSuccess)
        {
            HIP_WARN_IF_FAIL(hipFree(bufs.token_ids_ptr));
            bufs.token_ids_ptr = nullptr;
            return false;
        }
        hipEvent_t ready_event = nullptr;
        err = hipEventCreateWithFlags(
            &ready_event,
            hipEventDisableTiming);
        if (err != hipSuccess)
        {
            HIP_WARN_IF_FAIL(hipFree(bufs.penalties_ptr));
            HIP_WARN_IF_FAIL(hipFree(bufs.token_ids_ptr));
            bufs.penalties_ptr = nullptr;
            bufs.token_ids_ptr = nullptr;
            return false;
        }
        bufs.ready_event = ready_event;
        bufs.allocated_count = vocab_size;
        return true;
    }

    bool ROCmBackend::applyLogitPenaltiesF32(void *logits_device,
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        hipStream_t s =
            requireExplicitStream(stream, "ROCmBackend::applyLogitPenaltiesF32");

        if (bufs.publication_valid &&
            bufs.producer_stream != stream)
        {
            HIP_CHECK_OR_THROW(hipStreamWaitEvent(
                s,
                static_cast<hipEvent_t>(bufs.ready_event),
                0));
        }

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

        HIP_CHECK_OR_THROW(hipEventRecord(
            static_cast<hipEvent_t>(bufs.ready_event),
            s));
        bufs.producer_stream = stream;
        bufs.publication_valid = true;
        return true;
    }

    bool ROCmBackend::enqueueLogitPenaltiesF32Device(void *logits_device,
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocmOps_apply_logit_penalties_f32(
            static_cast<float *>(logits_device),
            static_cast<const int *>(token_ids_device),
            static_cast<const float *>(penalties_device),
            num_penalties,
            vocab_size,
            device_id,
            stream);
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
        if (err != hipSuccess)
            LOG_ERROR("[ROCmBackend::synchronize] hipDeviceSynchronize failed: "
                      << hipGetErrorString(err));
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
        /*
         * Match CUDA's event contract: ordinary events are used as cheap
         * stream-ordering tokens and must not collect elapsed-time data.  HIP
         * timing events can introduce extra dependency cost on hot paths such
         * as stochastic MTP response bridging, so callers that need elapsed
         * time must request createTimingEvent() explicitly.
         */
        err = hipEventCreateWithFlags(&event, hipEventDisableTiming);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::createEvent] hipEventCreate failed: " << hipGetErrorString(err));
            return nullptr;
        }

        return reinterpret_cast<void *>(event);
    }

    void *ROCmBackend::createTimingEvent(int device_id)
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
            LOG_ERROR("[ROCmBackend::createTimingEvent] hipEventCreate failed: " << hipGetErrorString(err));
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
        hipStream_t hip_stream =
            requireExplicitStream(stream, "ROCmBackend::recordEvent");

        /*
         * IBackend events publish graph completion to consumers outside the
         * captured DAG. A capture-time hipEventRecord is an internal graph node,
         * not that external handoff. Internal graph fork/join edges belong to
         * IWorkerGPUContext, so reject this misuse rather than reporting a
         * publication that no outside consumer can legally wait on.
         */
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        err = hipStreamIsCapturing(hip_stream, &capture_status);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::recordEvent] hipStreamIsCapturing failed: "
                      << hipGetErrorString(err));
            return false;
        }
        if (capture_status != hipStreamCaptureStatusNone)
        {
            LOG_ERROR("[ROCmBackend::recordEvent] External event publication is forbidden during graph capture; record it after graph launch");
            return false;
        }
        err = hipEventRecord(hip_event, hip_stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::recordEvent] hipEventRecord failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmBackend::eventElapsedTimeMs(
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

        HipDeviceSaveRestore device_guard;
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            return false;
        }

        err = hipEventElapsedTime(
            out_ms,
            reinterpret_cast<hipEvent_t>(start_event),
            reinterpret_cast<hipEvent_t>(stop_event));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::eventElapsedTimeMs] hipEventElapsedTime failed: " << hipGetErrorString(err));
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

    bool ROCmBackend::queryEvent(void *event, int device_id, bool *ready)
    {
        if (ready)
            *ready = false;
        if (!event || !ready || device_id < 0 || device_id >= device_count_)
            return false;

        HipDeviceSaveRestore device_guard;
        const hipError_t set_error = hipSetDevice(device_id);
        if (set_error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::queryEvent] hipSetDevice(" << device_id
                                                                 << ") failed: "
                      << hipGetErrorString(set_error));
            return false;
        }

        const hipError_t err = hipEventQuery(
            reinterpret_cast<hipEvent_t>(event));
        if (err == hipSuccess)
        {
            *ready = true;
            return true;
        }
        if (err == hipErrorNotReady)
            return true;

        LOG_ERROR("[ROCmBackend::queryEvent] hipEventQuery failed: "
                  << hipGetErrorString(err)
                  << " (device=" << device_id << ", event=" << event << ")");
        return false;
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
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
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

        void *ptr = nullptr;
        /*
         * hipMemGetInfo reports memory returned to the device driver, not the
         * complete allocation capacity reusable by this process. In
         * particular, HIP may retain successfully freed graph-bound slabs and
         * satisfy a later hipMalloc from that process cache without increasing
         * the reported free-byte scalar. Admission owns the complete workload
         * BOM; at this infrastructure boundary hipMalloc is therefore the only
         * valid authority for one concrete allocation. A speculative
         * `bytes + cushion <= free` test can reject an allocation the runtime
         * can satisfy and made prepared-model reuse topology-order dependent.
         */
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

        if ((reinterpret_cast<std::uintptr_t>(ptr) & (kDeviceAllocationAlignment - 1)) != 0)
        {
            LOG_ERROR("[ROCmBackend] hipMalloc returned unaligned pointer " << ptr
                                                                            << " for " << bytes << " bytes on device "
                                                                            << device_id << " (required "
                                                                            << kDeviceAllocationAlignment << "-byte alignment)");
            (void)hipFree(ptr);
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

        LOG_TRACE("[ROCM_PTR_ALLOC] ptr=" << ptr
                                          << " bytes=" << bytes
                                          << " device=" << device_id);
        if (vramBomEnabled())
        {
            size_t free_after = 0;
            size_t total_bytes = 0;
            (void)hipMemGetInfo(&free_after, &total_bytes);
            logVramBomLine(
                "backend_allocation",
                "backend=rocm action=allocate device=" + std::to_string(device_id) +
                    " ptr=" + vramBomPointer(ptr) +
                    " " + vramBomBytes(bytes) +
                    " free_after_bytes=" + std::to_string(free_after) +
                    " total_bytes=" + std::to_string(total_bytes));
        }

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
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
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

        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations()[host_ptr] = device_id;
        }
        return host_ptr;
    }

    void ROCmBackend::freeMapped(void *host_ptr, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
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
        else
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations().erase(host_ptr);
        }
    }

    void ROCmBackend::free(void *ptr, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
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
            {
                std::lock_guard<std::mutex> lock(g_ptr_registry_mutex);
                const auto it = g_active_ptrs.find(ptr);
                if (it != g_active_ptrs.end())
                {
                    recordPointerEvent(
                        "free", ptr, it->second.size_bytes, device_id, false);
                    g_active_ptrs.erase(it);
                }
                else
                {
                    recordPointerEvent(
                        "free-unknown", ptr, 0, device_id, false);
                }
            }
            LOG_TRACE("[ROCM_PTR_FREE] ptr=" << ptr
                                             << " bytes=" << recorded_size
                                             << " device=" << device_id);
            if (vramBomEnabled())
            {
                size_t free_after = 0;
                size_t total_bytes = 0;
                (void)hipMemGetInfo(&free_after, &total_bytes);
                logVramBomLine(
                    "backend_allocation",
                    "backend=rocm action=free device=" + std::to_string(device_id) +
                        " ptr=" + vramBomPointer(ptr) +
                        " " + vramBomBytes(recorded_size) +
                        " free_after_bytes=" + std::to_string(free_after) +
                        " total_bytes=" + std::to_string(total_bytes));
            }
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

        err = hipMemsetAsync(
            ptr,
            value,
            bytes,
            requireExplicitStream(stream, "ROCmBackend::memset"));
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend] hipMemsetAsync failed: " << hipGetErrorString(err));
            return false;
        }

        return true;
    }

    bool ROCmBackend::enqueuePreparePrefillChunkView(
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

        HIP_CHECK_OR_THROW(hipSetDevice(device_id));
        return rocm::launchPreparePrefillChunkView(
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
            requireExplicitStream(stream, "ROCmBackend::vectorAddInplace"));
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

    DeviceAllocationAccounting
    ROCmBackend::deviceAllocationAccounting(int device_id) const
    {
        DeviceAllocationAccounting accounting;
        if (device_id < 0 || device_id >= device_count_)
        {
            accounting.diagnostic = "invalid HIP device ordinal " +
                                    std::to_string(device_id);
            return accounting;
        }

        /* The lifecycle lock excludes allocation/free and native reset while
         * the pointer-registry lock makes the count and byte sum one atomic
         * observation of canonical ROCm ownership. */
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
        std::lock_guard<std::mutex> pointer_lock(g_ptr_registry_mutex);
        for (const auto &[pointer, allocation] : g_active_ptrs)
        {
            (void)pointer;
            if (!allocation.active || allocation.device_id != device_id)
                continue;
            if (allocation.size_bytes >
                std::numeric_limits<size_t>::max() - accounting.active_bytes)
            {
                accounting.diagnostic =
                    "HIP canonical allocation byte accounting overflow";
                return accounting;
            }
            ++accounting.active_allocations;
            accounting.active_bytes += allocation.size_bytes;
        }
        accounting.supported = true;
        return accounting;
    }

    DeviceMemoryCacheReclamationResult
    ROCmBackend::trimUnusedDeviceMemoryCaches(int device_id)
    {
        DeviceMemoryCacheReclamationResult result;
        if (device_id < 0 || device_id >= device_count_)
        {
            result.diagnostic = "invalid HIP device ordinal " +
                                std::to_string(device_id);
            return result;
        }

        HipDeviceSaveRestore device_guard;
        if (!device_guard.valid())
        {
            (void)hipGetLastError();
            result.diagnostic =
                "hipGetDevice could not preserve the caller device identity";
            return result;
        }
        const hipError_t select_error = static_cast<hipError_t>(
            HipDeviceGuard::forceSetDevice(device_id));
        if (select_error != hipSuccess)
        {
            result.diagnostic =
                "hipSetDevice failed for ROCm:" + std::to_string(device_id) +
                ": " + hipGetErrorString(select_error);
            (void)hipGetLastError();
            return result;
        }
        result.supported = true;

        hipMemPool_t default_pool = nullptr;
        bool default_pool_available = false;
        const auto query_snapshot =
            [&](DeviceMemoryCacheSnapshot &snapshot,
                const char *phase) -> bool
        {
            size_t total_bytes = 0u;
            hipError_t error = hipMemGetInfo(
                &snapshot.driver_free_bytes, &total_bytes);
            if (error != hipSuccess)
            {
                result.diagnostic = std::string("hipMemGetInfo failed ") +
                                    phase + ": " + hipGetErrorString(error);
                (void)hipGetLastError();
                return false;
            }

            std::uint64_t graph_used = 0u;
            std::uint64_t graph_reserved = 0u;
            error = hipDeviceGetGraphMemAttribute(
                device_id, hipGraphMemAttrUsedMemCurrent, &graph_used);
            if (error == hipSuccess)
            {
                error = hipDeviceGetGraphMemAttribute(
                    device_id,
                    hipGraphMemAttrReservedMemCurrent,
                    &graph_reserved);
            }
            if (error != hipSuccess)
            {
                result.diagnostic =
                    std::string("HIP graph-memory accounting failed ") +
                    phase + ": " + hipGetErrorString(error);
                (void)hipGetLastError();
                return false;
            }
            snapshot.graph_accounting_available = true;
            snapshot.graph_used_bytes = static_cast<size_t>(graph_used);
            snapshot.graph_reserved_bytes =
                static_cast<size_t>(graph_reserved);

            if (!default_pool_available)
            {
                error = hipDeviceGetDefaultMemPool(
                    &default_pool, device_id);
                if (error == hipErrorNotSupported)
                {
                    (void)hipGetLastError();
                    return true;
                }
                if (error != hipSuccess || default_pool == nullptr)
                {
                    result.diagnostic =
                        std::string("HIP default memory-pool resolution failed ") +
                        phase + ": " + hipGetErrorString(error);
                    (void)hipGetLastError();
                    return false;
                }
                default_pool_available = true;
            }

            std::uint64_t pool_used = 0u;
            std::uint64_t pool_reserved = 0u;
            error = hipMemPoolGetAttribute(
                default_pool, hipMemPoolAttrUsedMemCurrent, &pool_used);
            if (error == hipSuccess)
            {
                error = hipMemPoolGetAttribute(
                    default_pool,
                    hipMemPoolAttrReservedMemCurrent,
                    &pool_reserved);
            }
            if (error != hipSuccess)
            {
                result.diagnostic =
                    std::string("HIP default memory-pool accounting failed ") +
                    phase + ": " + hipGetErrorString(error);
                (void)hipGetLastError();
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

        const hipError_t graph_trim_error =
            hipDeviceGraphMemTrim(device_id);
        result.graph_trim_invoked = true;
        if (graph_trim_error != hipSuccess)
        {
            result.diagnostic =
                "hipDeviceGraphMemTrim failed for ROCm:" +
                std::to_string(device_id) + ": " +
                hipGetErrorString(graph_trim_error);
            (void)hipGetLastError();
            return result;
        }

        if (default_pool_available)
        {
            const hipError_t pool_trim_error =
                hipMemPoolTrimTo(default_pool, 0u);
            result.async_pool_trim_invoked = true;
            if (pool_trim_error != hipSuccess)
            {
                result.diagnostic =
                    "hipMemPoolTrimTo failed for ROCm:" +
                    std::to_string(device_id) + ": " +
                    hipGetErrorString(pool_trim_error);
                (void)hipGetLastError();
                return result;
            }
        }

        if (!query_snapshot(result.after, "after trim"))
            return result;

        result.success = true;
        return result;
    }

    DeviceRuntimeGenerationRetirementResult
    ROCmBackend::retireExclusiveDeviceRuntimeGeneration(
        const DeviceRuntimeGenerationRetirementRequest &request)
    {
        DeviceRuntimeGenerationRetirementResult result;
        result.supported = true;
        const int device_id = request.deviceOrdinal();
        if (device_id < 0 || device_id >= device_count_)
        {
            result.diagnostic = "invalid HIP device ordinal " +
                                std::to_string(device_id);
            return result;
        }

        /*
         * Exclude canonical allocation/registration mutation across preflight
         * and reset. The unforgeable request separately proves that ordinary
         * model execution has ended before this backend boundary is entered.
         */
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());

        {
            std::lock_guard<std::mutex> pointer_lock(g_ptr_registry_mutex);
            for (const auto &[pointer, allocation] : g_active_ptrs)
            {
                (void)pointer;
                if (allocation.active && allocation.device_id == device_id)
                {
                    ++result.tracked_device_allocations;
                    if (allocation.size_bytes >
                        std::numeric_limits<size_t>::max() -
                            result.tracked_device_allocation_bytes)
                    {
                        result.diagnostic =
                            "HIP runtime-generation allocation byte accounting overflow";
                        return result;
                    }
                    result.tracked_device_allocation_bytes +=
                        allocation.size_bytes;
                }
            }
        }
        {
            std::lock_guard<std::mutex> registration_lock(
                rocmPinnedAllocationsMutex());
            result.tracked_host_registrations =
                rocmPinnedAllocations().size();
        }
        if (result.tracked_device_allocations != 0u ||
            result.tracked_host_registrations != 0u)
        {
            std::ostringstream diagnostic;
            diagnostic
                << "HIP runtime-generation retirement rejected: live "
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
                    "HIP runtime generation is invalid or exhausted";
                return result;
            }
        }

        hipError_t error = static_cast<hipError_t>(
            HipDeviceGuard::forceSetDevice(device_id));
        if (error != hipSuccess)
        {
            result.diagnostic =
                "hipSetDevice failed before runtime reset: " +
                std::string(hipGetErrorString(error));
            (void)hipGetLastError();
            return result;
        }

        size_t total_bytes = 0u;
        error = hipMemGetInfo(
            &result.driver_free_bytes_before, &total_bytes);
        if (error != hipSuccess)
        {
            result.diagnostic =
                "hipMemGetInfo failed before runtime reset: " +
                std::string(hipGetErrorString(error));
            (void)hipGetLastError();
            return result;
        }

        if (!llaminar2_retireROCmTensorValidatorRuntimeGeneration(device_id))
        {
            result.diagnostic =
                "ROCm tensor-validator generation could not retire";
            return result;
        }

        result.reset_invoked = true;
        error = hipDeviceReset();
        if (error != hipSuccess)
        {
            result.diagnostic = "hipDeviceReset failed for ROCm:" +
                                std::to_string(device_id) + ": " +
                                hipGetErrorString(error);
            (void)hipGetLastError();
            HipDeviceGuard::resetTracking();
            return result;
        }

        /*
         * hipDeviceReset destroyed these backend-owned device pointers and
         * events. Their host identities must be cleared without calling HIP on
         * the now-stale handles; the next model lazily creates fresh storage.
         */
        if (static_cast<size_t>(device_id) < argmax_buffers_.size())
            argmax_buffers_[static_cast<size_t>(device_id)] = {};
        if (static_cast<size_t>(device_id) < topk_buffers_.size())
            topk_buffers_[static_cast<size_t>(device_id)] = {};
        if (static_cast<size_t>(device_id) < sample_token_buffers_.size())
            sample_token_buffers_[static_cast<size_t>(device_id)] = {};
        if (static_cast<size_t>(device_id) < penalty_buffers_.size())
            penalty_buffers_[static_cast<size_t>(device_id)] = {};

        /* hipDeviceReset is irrevocable. Advance the authority before the
         * diagnostic reinitialization below so no error path can advertise the
         * retired generation as live. */
        {
            std::lock_guard<std::mutex> generation_lock(
                runtime_generation_mutex_);
            result.successor_generation = result.retired_generation + 1u;
            runtime_generations_[static_cast<size_t>(device_id)] =
                result.successor_generation;
        }

        /*
         * The retired device must remain absent from HipDeviceGuard's
         * thread-local identity. Forcing a device or querying memory here
         * would initialize the successor context and consume the capacity
         * that retirement is meant to expose to the next model admission.
         */
        HipDeviceGuard::resetTracking();

        /*
         * AMD HIP's deprecated primary-context APIs intentionally cannot
         * certify inactivity: hipDevicePrimaryCtxRelease is documented as a
         * successful no-op on the AMD path. hipDeviceReset is the native
         * authority that discards execution state. Its success, the empty
         * canonical ledgers above, and making no subsequent materializing HIP
         * call are the ROCm proof that the successor remains quiescent.
         */
        result.post_reset_state = DeviceRuntimePostResetState::Quiescent;
        result.success = true;
        result.diagnostic =
            "HIP runtime generation retired with quiescent successor";
        return result;
    }

    std::uint64_t ROCmBackend::deviceRuntimeGeneration(
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
        hipStream_t hip_stream =
            requireExplicitStream(stream, "ROCmBackend::synchronizeStream");
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
            return false;
        err = hipStreamSynchronize(hip_stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::synchronizeStream] failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool ROCmBackend::streamWaitEvent(void *stream, void *event, int device_id)
    {
        hipStream_t hip_stream =
            requireExplicitStream(stream, "ROCmBackend::streamWaitEvent");
        if (!event || device_id < 0 || device_id >= device_count_)
        {
            LOG_ERROR("[ROCmBackend::streamWaitEvent] invalid event-wait ownership"
                      << " device=" << device_id
                      << " stream=" << stream
                      << " event=" << event);
            return false;
        }

        /*
         * A LocalTP control thread can alternate between several HIP children.
         * Event waits must select the stream/event owner's ordinal explicitly,
         * then restore the caller's ambient device just like recordEvent().
         */
        HipDeviceSaveRestore device_guard;
        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamWaitEvent] hipSetDevice("
                      << device_id << ") failed: "
                      << hipGetErrorString(err));
            return false;
        }
        err = hipStreamWaitEvent(
            hip_stream,
            static_cast<hipEvent_t>(event), 0);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamWaitEvent] failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool ROCmBackend::supportsStreamTimelineSignal32(int device_id) const
    {
        if (device_id < 0 || device_id >= device_count_)
            return false;

        HipDeviceSaveRestore device_guard;
        int supported = 0;
        return hipSetDevice(device_id) == hipSuccess &&
               hipDeviceGetAttribute(
                   &supported,
                   hipDeviceAttributeCanUseStreamWaitValue,
                   device_id) == hipSuccess &&
               supported != 0;
    }

    void *ROCmBackend::allocateStreamTimelineSignal32(int device_id)
    {
        if (!supportsStreamTimelineSignal32(device_id))
        {
            LOG_ERROR("[ROCmBackend::allocateStreamTimelineSignal32] unsupported device="
                      << device_id);
            return nullptr;
        }

        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(device_id) != hipSuccess)
            return nullptr;

        void *signal = nullptr;
        /*
         * hipMallocSignalMemory represents one native 64-bit HSA signal and
         * the ROCm allocator therefore requires an exact eight-byte request,
         * even when the queued protocol operates on its low 32 bits.  A
         * four-byte allocation is rejected with hipErrorInvalidValue on the
         * production gfx906 runtime.
         */
        const hipError_t error = hipExtMallocWithFlags(
            &signal,
            sizeof(uint64_t),
            hipMallocSignalMemory);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::allocateStreamTimelineSignal32] hipExtMallocWithFlags failed: "
                      << hipGetErrorString(error));
            return nullptr;
        }
        return signal;
    }

    void ROCmBackend::freeStreamTimelineSignal32(void *signal, int device_id)
    {
        if (!signal)
            return;
        HipDeviceSaveRestore device_guard;
        HIP_WARN_IF_FAIL(hipSetDevice(device_id));
        HIP_WARN_IF_FAIL(hipFree(signal));
    }

    bool ROCmBackend::streamWaitTimelineSignal32(
        void *stream,
        void *signal,
        uint32_t value,
        int device_id)
    {
        const hipStream_t hip_stream =
            requireExplicitStream(stream, "ROCmBackend::streamWaitTimelineSignal32");
        if (!signal || device_id < 0 || device_id >= device_count_)
        {
            LOG_ERROR("[ROCmBackend::streamWaitTimelineSignal32] invalid ownership"
                      << " device=" << device_id << " signal=" << signal);
            return false;
        }

        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(device_id) != hipSuccess)
            return false;
        const hipError_t error = hipStreamWaitValue32(
            hip_stream,
            signal,
            value,
            hipStreamWaitValueGte,
            UINT32_MAX);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamWaitTimelineSignal32] hipStreamWaitValue32 failed: "
                      << hipGetErrorString(error));
            return false;
        }
        return true;
    }

    bool ROCmBackend::streamPublishTimelineSignal32(
        void *stream,
        void *signal,
        uint32_t value,
        int device_id)
    {
        const hipStream_t hip_stream =
            requireExplicitStream(stream, "ROCmBackend::streamPublishTimelineSignal32");
        if (!signal || device_id < 0 || device_id >= device_count_)
        {
            LOG_ERROR("[ROCmBackend::streamPublishTimelineSignal32] invalid ownership"
                      << " device=" << device_id << " signal=" << signal);
            return false;
        }

        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(device_id) != hipSuccess)
            return false;
        const hipError_t error = hipStreamWriteValue32(
            hip_stream,
            signal,
            value,
            0);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamPublishTimelineSignal32] hipStreamWriteValue32 failed: "
                      << hipGetErrorString(error));
            return false;
        }
        return true;
    }

    bool ROCmBackend::supportsStreamTimelineSignal64(int device_id) const
    {
        return supportsStreamTimelineSignal32(device_id);
    }

    void *ROCmBackend::allocateStreamTimelineSignal64(int device_id)
    {
        if (!supportsStreamTimelineSignal64(device_id))
        {
            LOG_ERROR("[ROCmBackend::allocateStreamTimelineSignal64] unsupported device="
                      << device_id);
            return nullptr;
        }
        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(device_id) != hipSuccess)
            return nullptr;
        void *signal = nullptr;
        const hipError_t error = hipExtMallocWithFlags(
            &signal, sizeof(uint64_t), hipMallocSignalMemory);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::allocateStreamTimelineSignal64] hipExtMallocWithFlags failed: "
                      << hipGetErrorString(error));
            return nullptr;
        }
        return signal;
    }

    void ROCmBackend::freeStreamTimelineSignal64(
        void *signal,
        int device_id)
    {
        if (!signal)
            return;
        HipDeviceSaveRestore device_guard;
        HIP_WARN_IF_FAIL(hipSetDevice(device_id));
        HIP_WARN_IF_FAIL(hipFree(signal));
    }

    bool ROCmBackend::streamWaitTimelineSignal64(
        void *stream,
        void *signal,
        uint64_t value,
        int device_id)
    {
        const hipStream_t hip_stream = requireExplicitStream(
            stream, "ROCmBackend::streamWaitTimelineSignal64");
        if (!signal || device_id < 0 || device_id >= device_count_)
        {
            LOG_ERROR("[ROCmBackend::streamWaitTimelineSignal64] invalid ownership device="
                      << device_id << " signal=" << signal);
            return false;
        }
        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(device_id) != hipSuccess)
            return false;
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        const hipError_t capture_query =
            hipStreamIsCapturing(hip_stream, &capture_status);
        if (capture_query != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamWaitTimelineSignal64] hipStreamIsCapturing failed: "
                      << hipGetErrorString(capture_query));
            return false;
        }
        if (capture_status == hipStreamCaptureStatusActive)
        {
            return hip_graph_timeline::appendActiveCaptureSystemWaitValue64(
                hip_stream, signal, value);
        }
        if (capture_status == hipStreamCaptureStatusInvalidated)
        {
            LOG_ERROR("[ROCmBackend::streamWaitTimelineSignal64] capture stream is invalidated");
            return false;
        }
        const hipError_t error = hipStreamWaitValue64(
            hip_stream,
            signal,
            value,
            hipStreamWaitValueGte,
            UINT64_MAX);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamWaitTimelineSignal64] hipStreamWaitValue64 failed: "
                      << hipGetErrorString(error));
            return false;
        }
        return true;
    }

    bool ROCmBackend::streamPublishTimelineSignal64(
        void *stream,
        void *signal,
        uint64_t value,
        int device_id)
    {
        const hipStream_t hip_stream = requireExplicitStream(
            stream, "ROCmBackend::streamPublishTimelineSignal64");
        if (!signal || device_id < 0 || device_id >= device_count_)
        {
            LOG_ERROR("[ROCmBackend::streamPublishTimelineSignal64] invalid ownership device="
                      << device_id << " signal=" << signal);
            return false;
        }
        HipDeviceSaveRestore device_guard;
        if (hipSetDevice(device_id) != hipSuccess)
            return false;
        hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
        const hipError_t capture_query =
            hipStreamIsCapturing(hip_stream, &capture_status);
        if (capture_query != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamPublishTimelineSignal64] hipStreamIsCapturing failed: "
                      << hipGetErrorString(capture_query));
            return false;
        }
        if (capture_status == hipStreamCaptureStatusActive)
        {
            return hip_graph_timeline::appendActiveCaptureSystemReleaseValue64(
                hip_stream, signal, value);
        }
        if (capture_status == hipStreamCaptureStatusInvalidated)
        {
            LOG_ERROR("[ROCmBackend::streamPublishTimelineSignal64] capture stream is invalidated");
            return false;
        }
        const hipError_t error = hipStreamWriteValue64(
            hip_stream, signal, value, 0u);
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::streamPublishTimelineSignal64] hipStreamWriteValue64 failed: "
                      << hipGetErrorString(error));
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
        hipStream_t hip_stream =
            requireExplicitStream(stream, "ROCmBackend::hostToDeviceOnStream");
        if (device_id >= device_count_ || device_id < 0)
            return false;

        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
            return false;

        err = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice,
                             hip_stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::hostToDeviceOnStream] failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool ROCmBackend::deviceToHostOnStream(void *dst, const void *src, size_t bytes,
                                            int device_id, void *stream)
    {
        hipStream_t hip_stream =
            requireExplicitStream(stream, "ROCmBackend::deviceToHostOnStream");
        if (device_id >= device_count_ || device_id < 0)
            return false;

        hipError_t err = hipSetDevice(device_id);
        if (err != hipSuccess)
            return false;

        err = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost,
                             hip_stream);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::deviceToHostOnStream] failed: " << hipGetErrorString(err));
            return false;
        }
        return true;
    }

    bool ROCmBackend::deviceToMappedHostByKernelOnStream(
        void *dst,
        const void *src,
        size_t bytes,
        int device_id,
        void *stream)
    {
        hipStream_t hip_stream = requireExplicitStream(
            stream,
            "ROCmBackend::deviceToMappedHostByKernelOnStream");
        if (!dst || !src || bytes == 0u ||
            device_id < 0 || device_id >= device_count_ ||
            hipSetDevice(device_id) != hipSuccess)
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
            hipLaunchKernelGGL(
                mappedHostCopyVectorKernel,
                dim3(mappedHostCopyBlocks(vector_count)),
                dim3(kMappedHostCopyThreads),
                0u,
                hip_stream,
                static_cast<uint4 *>(dst),
                static_cast<const uint4 *>(src),
                vector_count);
        }
        else
        {
            hipLaunchKernelGGL(
                mappedHostCopyByteKernel,
                dim3(mappedHostCopyBlocks(bytes)),
                dim3(kMappedHostCopyThreads),
                0u,
                hip_stream,
                static_cast<std::uint8_t *>(dst),
                static_cast<const std::uint8_t *>(src),
                bytes);
        }
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::deviceToMappedHostByKernelOnStream] failed: "
                      << hipGetErrorString(error));
            return false;
        }
        return true;
    }

    bool ROCmBackend::enqueueMappedTransferProgressClaims(
        const MappedTransferProgressCommand *commands,
        MappedTransferProgressClaim *claims,
        size_t slot_capacity,
        int device_id,
        void *stream)
    {
        const hipStream_t hip_stream = requireExplicitStream(
            stream,
            "ROCmBackend::enqueueMappedTransferProgressClaims");
        if (!commands || !claims || slot_capacity == 0u ||
            slot_capacity > std::numeric_limits<unsigned int>::max() ||
            device_id < 0 || device_id >= device_count_ ||
            hipSetDevice(device_id) != hipSuccess)
        {
            return false;
        }

        hipLaunchKernelGGL(
            mappedTransferProgressClaimKernel,
            dim3(static_cast<unsigned int>(slot_capacity)),
            dim3(1u),
            0u,
            hip_stream,
            commands,
            claims,
            slot_capacity);
        hipError_t error = hipGetLastError();
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::enqueueMappedTransferProgressClaims] "
                      "claim launch failed: " << hipGetErrorString(error));
            return false;
        }
        return true;
    }

    bool ROCmBackend::enqueueMappedTransferProgressCopies(
        const MappedTransferProgressClaim *claims,
        MappedTransferProgressCompletion *completions,
        size_t slot_capacity,
        size_t maximum_bytes,
        int device_id,
        void *stream)
    {
        const hipStream_t hip_stream = requireExplicitStream(
            stream,
            "ROCmBackend::enqueueMappedTransferProgressCopies");
        if (!claims || !completions || slot_capacity == 0u ||
            maximum_bytes == 0u ||
            slot_capacity > std::numeric_limits<unsigned int>::max() ||
            device_id < 0 || device_id >= device_count_ ||
            hipSetDevice(device_id) != hipSuccess)
        {
            return false;
        }

        hipLaunchKernelGGL(
            mappedTransferProgressCopyKernel,
            dim3(static_cast<unsigned int>(slot_capacity)),
            dim3(kMappedHostCopyThreads),
            0u,
            hip_stream,
            claims,
            completions,
            slot_capacity,
            maximum_bytes);
        const hipError_t error = hipGetLastError();
        if (error != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::enqueueMappedTransferProgressCopies] "
                      "copy launch failed: " << hipGetErrorString(error));
            return false;
        }
        return true;
    }

    // ====================================================================
    // Pinned Host Memory
    // ====================================================================

    void *ROCmBackend::allocatePinned(size_t bytes, int device_id)
    {
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
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
        std::lock_guard<std::mutex> lifecycle_lock(
            rocmRuntimeResourceLifecycleMutex());
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
        else
        {
            std::lock_guard<std::mutex> lock(rocmPinnedAllocationsMutex());
            rocmPinnedAllocations().erase(ptr);
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

    bool ROCmBackend::deviceCopyAsync(void *dst, const void *src, size_t bytes,
                                      int device_id, void *stream)
    {
        if (bytes == 0)
            return true;
        if (!dst || !src)
        {
            LOG_ERROR("[ROCmBackend::deviceCopyAsync] null pointer: dst=" << dst
                                                                          << " src=" << src
                                                                          << " bytes=" << bytes);
            return false;
        }
        if (device_id >= device_count_ || device_id < 0)
        {
            LOG_ERROR("[ROCmBackend::deviceCopyAsync] invalid device_id=" << device_id);
            return false;
        }

        HipDeviceSaveRestore device_guard;
        hipError_t err_set = hipSetDevice(device_id);
        if (err_set != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::deviceCopyAsync] hipSetDevice(" << device_id
                                                                      << ") failed: "
                                                                      << hipGetErrorString(err_set));
            return false;
        }

        hipStream_t s =
            requireExplicitStream(stream, "ROCmBackend::deviceCopyAsync");

        /*
         * The MTP sidecar path copies tiny INT32 token slots between arena
         * buffers. Checking both endpoints here gives junior maintainers a
         * useful failure message if a future caller accidentally passes a host
         * shadow pointer or a pointer from a different GPU.
         */
        hipPointerAttribute_t dst_attrs{};
        hipPointerAttribute_t src_attrs{};
        hipError_t dst_attr_err = hipPointerGetAttributes(&dst_attrs, dst);
        hipError_t src_attr_err = hipPointerGetAttributes(&src_attrs, src);
        if (dst_attr_err != hipSuccess || src_attr_err != hipSuccess)
        {
            (void)hipGetLastError(); // clear any sticky pointer-query error
            LOG_ERROR("[ROCmBackend::deviceCopyAsync] pointer attribute query failed: dst_err="
                      << hipGetErrorString(dst_attr_err)
                      << " src_err=" << hipGetErrorString(src_attr_err));
            return false;
        }
        if (dst_attrs.device != device_id || src_attrs.device != device_id)
        {
            LOG_ERROR("[ROCmBackend::deviceCopyAsync] device mismatch: dst_device="
                      << dst_attrs.device << " src_device=" << src_attrs.device
                      << " requested_device=" << device_id);
            return false;
        }

        hipError_t err =
            hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, s);
        if (err != hipSuccess)
        {
            LOG_ERROR("[ROCmBackend::deviceCopyAsync] hipMemcpyAsync failed: "
                      << hipGetErrorString(err));
            return false;
        }
        return true;
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
