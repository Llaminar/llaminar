/**
 * @file LocalTPContext.cpp
 * @brief Implementation of LOCAL tensor parallelism context
 * @author David Sanftenberg
 * @date January 2026
 */

#include "LocalTPContext.h"
#include "backends/HostBackend.h"
#include "../tensors/TensorClasses.h"
#include "../backends/BackendManager.h" // For getCUDABackend, getROCmBackend
#include "../backends/ComputeBackend.h" // For DeviceManager (NUMA lookup)
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <stdexcept>

// Conditionally include GPU-specific backends
#ifdef HAVE_CUDA
#include "backends/NCCLBackend.h"
#include <cuda_runtime.h> // For cudaMemcpy in zero-copy allreduce
#endif

#ifdef HAVE_ROCM
#include "backends/RCCLBackend.h"
#include "../backends/rocm/ROCmBackend.h" // For ROCmBackend::deviceToDevice
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/PCIeBARBackend.h"
#include "backends/HeterogeneousBackend.h"
#endif

// ============================================================================
// Extern declarations for FP32 ↔ FP16 cast kernels (mixed-precision allreduce)
// ============================================================================
#ifdef HAVE_CUDA
extern "C"
{
    cudaError_t cudaCastFP32ToFP16(const float *fp32_input, void *fp16_output,
                                   size_t count, cudaStream_t stream);
    cudaError_t cudaCastFP16ToFP32(const void *fp16_input, float *fp32_output,
                                   size_t count, cudaStream_t stream);
    int cudaFP16ScratchAlloc(void **buf, size_t bytes, int ordinal);
    void cudaFP16ScratchFree(void *buf, int ordinal);
}
#endif

#ifdef HAVE_ROCM
extern "C"
{
    int rocmCastFP32ToFP16(const float *fp32_input, void *fp16_output,
                           size_t count, void *stream);
    int rocmCastFP16ToFP32(const void *fp16_input, float *fp32_output,
                           size_t count, void *stream);
    int rocmFP16ScratchAlloc(void **buf, size_t bytes, int ordinal);
    void rocmFP16ScratchFree(void *buf, int ordinal);
}
#endif

namespace llaminar2
{
    bool LocalTPContext::isLocalTPNCCLGraphPolicySupported(std::string *reason_out) const
    {
        const auto &exec = debugEnv().execution;

        // No graph capture in use: LocalTP NCCL is unaffected.
        if (!exec.gpu_graphs)
        {
            if (reason_out)
            {
                *reason_out = "gpu_graphs_off";
            }
            return true;
        }

        // Phase 3 support matrix: collectives under graph mode are only supported
        // when segmented collective replay is explicitly enabled.
        if (exec.gpu_graph_collective_segmented)
        {
            if (reason_out)
            {
                *reason_out = "gpu_graphs_segmented_collectives_enabled";
            }
            return true;
        }

        if (reason_out)
        {
            *reason_out = "gpu_graphs_on_without_segmented_collectives";
        }
        return false;
    }

    bool LocalTPContext::validateBarrierTensorSetForMultiGpuAllreduce(
        size_t effective_count,
        CollectiveDataType expected_dtype) const
    {
        // Junior-friendly note:
        // The barrier gathers one tensor per LOCAL TP device. Before launching a
        // grouped NCCL/RCCL collective, we validate that all participants describe
        // the same logical reduction problem.
        if (effective_count == 0)
        {
            LOG_ERROR("LocalTPContext::validateBarrierTensorSetForMultiGpuAllreduce: "
                      "effective_count must be > 0");
            return false;
        }

        for (int i = 0; i < degree(); ++i)
        {
            TensorBase *tensor = barrier_tensors_[i];
            if (!tensor)
            {
                LOG_ERROR("LocalTPContext::validateBarrierTensorSetForMultiGpuAllreduce: "
                          "null tensor at slot "
                          << i);
                return false;
            }

            auto device = tensor->current_device();
            if (!device.has_value() || !device->is_gpu())
            {
                LOG_ERROR("LocalTPContext::validateBarrierTensorSetForMultiGpuAllreduce: "
                          "tensor at slot "
                          << i << " is not resident on a GPU device");
                return false;
            }

            DeviceId expected_device = devices_[i].toLocalDeviceId();
            if (*device != expected_device)
            {
                LOG_ERROR("LocalTPContext::validateBarrierTensorSetForMultiGpuAllreduce: "
                          "device mismatch at slot "
                          << i
                          << " expected=" << expected_device.toString()
                          << " actual=" << device->toString());
                return false;
            }

            if (tensor->numel() < effective_count)
            {
                LOG_ERROR("LocalTPContext::validateBarrierTensorSetForMultiGpuAllreduce: "
                          "count exceeds tensor size at slot "
                          << i
                          << " count=" << effective_count
                          << " numel=" << tensor->numel());
                return false;
            }

            CollectiveDataType slot_dtype = tensorDTypeToCollective(tensor);
            if (slot_dtype != expected_dtype)
            {
                LOG_ERROR("LocalTPContext::validateBarrierTensorSetForMultiGpuAllreduce: "
                          "dtype mismatch at slot "
                          << i
                          << " expected=" << static_cast<int>(expected_dtype)
                          << " actual=" << static_cast<int>(slot_dtype));
                return false;
            }
        }

        return true;
    }

    // Helper function to get the appropriate backend for a device
    static IBackend *getBackendForDevice(DeviceId device)
    {
        if (device.is_cpu())
            return nullptr;

        if (device.is_cuda())
            return getCUDABackend();
        else if (device.is_rocm())
            return getROCmBackend();

        LOG_ERROR("[LocalTPContext] Unknown device type: " << device.toString());
        return nullptr;
    }

#ifdef HAVE_ROCM
    static uint64_t fnv1a64(const uint8_t *data, size_t length)
    {
        constexpr uint64_t FNV_OFFSET_BASIS = 1469598103934665603ull;
        constexpr uint64_t FNV_PRIME = 1099511628211ull;
        uint64_t hash = FNV_OFFSET_BASIS;
        for (size_t i = 0; i < length; ++i)
        {
            hash ^= static_cast<uint64_t>(data[i]);
            hash *= FNV_PRIME;
        }
        return hash;
    }

    static bool validateRocmAllreducePointerForSlot(const std::string &stage_name,
                                                    const char *phase,
                                                    int slot,
                                                    DeviceId expected_device,
                                                    TensorBase *tensor,
                                                    void *ptr,
                                                    uint64_t *watch_checksum_out = nullptr,
                                                    size_t *watch_sample_bytes_out = nullptr,
                                                    size_t *watch_sample_offset_out = nullptr)
    {
        if (watch_checksum_out)
            *watch_checksum_out = 0;
        if (watch_sample_bytes_out)
            *watch_sample_bytes_out = 0;
        if (watch_sample_offset_out)
            *watch_sample_offset_out = 0;

        if (!expected_device.is_rocm())
        {
            return true;
        }

        auto *backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(expected_device));
        if (!backend)
        {
            LOG_ERROR("[LOCALTP_ROCM_PTR_VALIDATE_FAIL] missing ROCm backend for slot=" << slot
                                                                                        << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                                                                                        << " expected_device=" << expected_device.toString());
            return false;
        }

        const int expected_ordinal = expected_device.rocm_ordinal();
        backend->setDevice(expected_ordinal);

        bool is_device_ptr = false;
        bool is_host_ptr = false;
        bool is_managed = false;
        int attr_device = -1;
        if (!backend->queryPointerAttributes(ptr, is_device_ptr, is_host_ptr, is_managed, attr_device))
        {
            LOG_ERROR("[LOCALTP_ROCM_PTR_VALIDATE_FAIL] hip attribute query failed"
                      << " slot=" << slot
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " ptr=" << ptr
                      << " expected_device=" << expected_ordinal
                      << " tensor=" << static_cast<void *>(tensor)
                      << " tensor_device="
                      << (tensor && tensor->current_device().has_value() ? tensor->current_device()->toString() : "none"));
            ROCmBackend::dumpRecentPointerEvents(128);
            return false;
        }

        if (!is_device_ptr || attr_device != expected_ordinal)
        {
            LOG_ERROR("[LOCALTP_ROCM_PTR_VALIDATE_FAIL] hip attribute mismatch"
                      << " slot=" << slot
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " ptr=" << ptr
                      << " expected_device=" << expected_ordinal
                      << " attr_device=" << attr_device
                      << " is_device_ptr=" << (is_device_ptr ? 1 : 0)
                      << " is_host_ptr=" << (is_host_ptr ? 1 : 0)
                      << " is_managed=" << (is_managed ? 1 : 0)
                      << " tensor=" << static_cast<void *>(tensor)
                      << " tensor_device="
                      << (tensor && tensor->current_device().has_value() ? tensor->current_device()->toString() : "none"));
            ROCmBackend::dumpRecentPointerEvents(128);
            return false;
        }

        ROCmPointerOwnerInfo owner;
        if (!ROCmBackend::queryPointerOwner(ptr, owner))
        {
            LOG_ERROR("[LOCALTP_ROCM_PTR_VALIDATE_FAIL] owner lookup failed"
                      << " slot=" << slot
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " ptr=" << ptr
                      << " expected_device=" << expected_ordinal
                      << " tensor=" << static_cast<void *>(tensor));
            ROCmBackend::dumpRecentPointerEvents(128);
            return false;
        }

        if (owner.device_id != expected_ordinal)
        {
            LOG_ERROR("[LOCALTP_ROCM_PTR_VALIDATE_FAIL] owner mismatch"
                      << " slot=" << slot
                      << " phase=" << (phase ? phase : "(unknown)")
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " ptr=" << ptr
                      << " expected_device=" << expected_ordinal
                      << " owner_device=" << owner.device_id
                      << " owner_base=" << owner.base_ptr
                      << " owner_bytes=" << owner.size_bytes
                      << " owner_seq=" << owner.sequence
                      << " owner_thread=" << owner.thread_hash
                      << " tensor=" << static_cast<void *>(tensor)
                      << " tensor_device="
                      << (tensor && tensor->current_device().has_value() ? tensor->current_device()->toString() : "none"));
            ROCmBackend::dumpRecentPointerEvents(128);
            return false;
        }

        const auto &validation = debugEnv().validation;
        if (validation.trace_local_tp_pointer)
        {
            const uintptr_t watch = static_cast<uintptr_t>(validation.trace_local_tp_pointer_address);
            const uintptr_t begin = reinterpret_cast<uintptr_t>(owner.base_ptr);
            const uintptr_t end = begin + owner.size_bytes;
            if (watch >= begin && watch < end)
            {
                const size_t offset = static_cast<size_t>(watch - begin);
                constexpr size_t WATCH_SAMPLE_MAX_BYTES = 256;
                const size_t available = owner.size_bytes > offset ? (owner.size_bytes - offset) : 0;
                const size_t sample_bytes = std::min(WATCH_SAMPLE_MAX_BYTES, available);

                uint64_t checksum = 0;
                bool checksum_ready = false;
                if (sample_bytes > 0)
                {
                    std::array<uint8_t, WATCH_SAMPLE_MAX_BYTES> sample{};
                    const uint8_t *sample_src = reinterpret_cast<const uint8_t *>(owner.base_ptr) + offset;
                    if (backend->deviceToHost(sample.data(), const_cast<uint8_t *>(sample_src), sample_bytes, expected_ordinal))
                    {
                        checksum = fnv1a64(sample.data(), sample_bytes);
                        checksum_ready = true;
                    }
                    else
                    {
                        LOG_WARN("[LOCALTP_PTR_WATCH_COPY_FAIL]"
                                 << " phase=" << (phase ? phase : "(unknown)")
                                 << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                                 << " slot=" << slot
                                 << " watch=" << reinterpret_cast<const void *>(watch)
                                 << " owner_base=" << owner.base_ptr
                                 << " sample_offset=" << offset
                                 << " sample_bytes=" << sample_bytes
                                 << " copy_error=deviceToHost_failed");
                    }
                }

                if (checksum_ready)
                {
                    if (watch_checksum_out)
                        *watch_checksum_out = checksum;
                    if (watch_sample_bytes_out)
                        *watch_sample_bytes_out = sample_bytes;
                    if (watch_sample_offset_out)
                        *watch_sample_offset_out = offset;
                }

                LOG_WARN("[LOCALTP_PTR_WATCH_HIT]"
                         << " phase=" << (phase ? phase : "(unknown)")
                         << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                         << " slot=" << slot
                         << " watch=" << reinterpret_cast<const void *>(watch)
                         << " buffer_ptr=" << ptr
                         << " expected_device=" << expected_device.toString()
                         << " owner_device=" << owner.device_id
                         << " owner_base=" << owner.base_ptr
                         << " owner_bytes=" << owner.size_bytes
                         << " owner_seq=" << owner.sequence
                         << " offset=" << offset
                         << " sample_bytes=" << sample_bytes
                         << " checksum=" << (checksum_ready ? std::to_string(checksum) : std::string("n/a"))
                         << " tensor=" << static_cast<void *>(tensor)
                         << " tensor_name=" << (tensor && !tensor->debugName().empty() ? tensor->debugName() : "(unnamed)"));
            }
        }

        return true;
    }
#endif

    // =========================================================================
    // Construction
    // =========================================================================

    LocalTPContext::LocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend)
        : devices_(std::move(devices))
    {
        // Validate devices
        if (devices_.empty())
        {
            throw std::invalid_argument("LocalTPContext: devices cannot be empty");
        }

        // Handle weights
        if (weights.empty())
        {
            // Equal distribution
            weights_.resize(devices_.size(), 1.0f / static_cast<float>(devices_.size()));
        }
        else if (weights.size() != devices_.size())
        {
            throw std::invalid_argument(
                "LocalTPContext: weights count (" + std::to_string(weights.size()) +
                ") must match device count (" + std::to_string(devices_.size()) + ")");
        }
        else
        {
            weights_ = normalizeWeights(weights);
        }

        // Handle backend
        if (backend == CollectiveBackendType::AUTO)
        {
            backend_ = autoDetectBackend(devices_);
        }
        else
        {
            backend_ = backend;
        }

        // Build lookup index
        buildDeviceIndex();

        LOG_DEBUG("LocalTPContext created: degree=" << degree()
                                                    << ", backend=" << collectiveBackendTypeToString(backend_));

        // Initialize backend for multi-device scenarios
        if (degree() > 1)
        {
            if (!initializeBackend())
            {
                LOG_WARN("LocalTPContext: Failed to initialize backend "
                         << collectiveBackendTypeToString(backend_)
                         << ", collectives will be no-ops");
            }
        }
    }

    LocalTPContext::~LocalTPContext()
    {
        // Free FP16 scratch buffers via helper functions (avoids HIP/CUDA header conflicts)
        for (size_t i = 0; i < fp16_scratch_buffers_.size(); ++i)
        {
            if (fp16_scratch_buffers_[i])
            {
                const int ordinal = devices_[i].device_ordinal;
#ifdef HAVE_CUDA
                if (device_group_.allCUDA())
                    cudaFP16ScratchFree(fp16_scratch_buffers_[i], ordinal);
#endif
#ifdef HAVE_ROCM
                if (device_group_.allROCm())
                    rocmFP16ScratchFree(fp16_scratch_buffers_[i], ordinal);
#endif
                fp16_scratch_buffers_[i] = nullptr;
            }
        }

        const uint64_t attempts = nccl_allreduce_attempts_.load();
        const uint64_t success = nccl_allreduce_success_.load();
        const uint64_t failures = nccl_allreduce_failures_.load();

        if (attempts == 0)
        {
            return;
        }

        LOG_INFO("[LocalTPContext][Telemetry] "
                 << "backend=" << collectiveBackendTypeToString(backend_)
                 << " nccl_allreduce_attempts=" << attempts
                 << " nccl_allreduce_success=" << success
                 << " nccl_allreduce_failures=" << failures);
    }

    // =========================================================================
    // Configuration
    // =========================================================================

    void LocalTPContext::requestAbort()
    {
        // Set the flag first so other threads see it immediately
        bool was_set = abort_requested_.exchange(true, std::memory_order_acq_rel);
        if (was_set)
        {
            LOG_WARN("[LocalTPContext] requestAbort() called but abort already in progress");
            return;
        }

        LOG_WARN("[LocalTPContext] Abort requested — aborting collective backend to unblock stuck devices");

        if (backend_impl_)
        {
            backend_impl_->abort();
        }

        // Wake any threads blocked on the barrier condition variable
        barrier_cv_.notify_all();
    }

    const std::vector<GlobalDeviceAddress> &LocalTPContext::devices() const
    {
        return devices_;
    }

    const std::vector<float> &LocalTPContext::weights() const
    {
        return weights_;
    }

    CollectiveBackendType LocalTPContext::backend() const
    {
        return backend_;
    }

    int LocalTPContext::degree() const
    {
        return static_cast<int>(devices_.size());
    }

    int LocalTPContext::myIndex() const
    {
        if (current_device_index_ < 0)
        {
            throw std::runtime_error(
                "LocalTPContext::myIndex() called before setCurrentDeviceIndex(). "
                "In orchestrator-driven LOCAL TP, the current device must be set explicitly.");
        }
        return current_device_index_;
    }

    void LocalTPContext::setCurrentDeviceIndex(int index)
    {
        if (index < 0 || index >= static_cast<int>(devices_.size()))
        {
            throw std::out_of_range(
                "LocalTPContext::setCurrentDeviceIndex(): index " + std::to_string(index) +
                " out of range [0, " + std::to_string(devices_.size()) + ")");
        }
        current_device_index_ = index;
    }

    // =========================================================================
    // Collective Operations
    // =========================================================================

    bool LocalTPContext::allreduce(TensorBase *tensor)
    {
        // Delegate to overload with empty stage name and default count (0 = use numel)
        return allreduce(tensor, "", 0);
    }

    bool LocalTPContext::allreduce(TensorBase *tensor, const std::string &stage_name, size_t count)
    {
        if (!tensor)
        {
            LOG_ERROR("LocalTPContext::allreduce: null tensor");
            return false;
        }

        // Resolve count: 0 means use tensor->numel()
        const size_t effective_count = (count > 0) ? count : tensor->numel();

        std::unique_lock<std::mutex> lock(mutex_);

        // Single device - no-op
        if (degree() == 1)
        {
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allreduce: Backend not initialized, skipping");
            return true; // Return true to allow pipeline to continue
        }

        // ================================================================
        // Multi-GPU Backends (NCCL/RCCL): Use barrier-synchronized allreduce
        // ================================================================
        // For LOCAL TP with multiple threads (one per device), each thread calls
        // allreduce() with its OWN tensor. We CANNOT use getDeviceBuffers() on a
        // single tensor because TensorBase can only be on ONE GPU at a time.
        // Instead, use the barrier-synchronized approach where all device threads
        // rendezvous, collect their buffers, then the last arrival executes
        // allreduceMulti with all buffers.
        // ================================================================
        // PCIeBAR Backend: Use barrier-synchronized allreduce
        // ================================================================
        // CRITICAL: This check MUST happen BEFORE ensureOnDevice() because:
        // - Multiple threads call allreduce() concurrently, each with their OWN tensor
        // - Each thread may be running on behalf of a different device (CUDA:0 or ROCm:0)
        // - We cannot know which device "this thread" is without the barrier rendezvous
        // - If we call ensureOnDevice(devices_[0]) here, ALL threads would upload to CUDA:0
        // - The allreduceWithBarrier path handles device placement correctly via stage registration
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (backend_ == CollectiveBackendType::PCIE_BAR && degree() > 1)
        {
            // Release the main mutex before entering barrier (to avoid deadlock)
            // The barrier has its own mutex for synchronization
            lock.unlock();
            return allreduceWithBarrier(tensor, stage_name, effective_count);
        }
#endif

        // ================================================================
        // Multi-GPU Backends (NCCL/RCCL): Prefer barrier-free per-device allreduce
        // ================================================================
        // Each device thread independently calls rcclAllReduce with its own
        // communicator. RCCL internally matches calls across devices.
        // Stream dependencies ensure GPU-side ordering — host never blocks.
        //
        // Falls back to barrier-synchronized path if the backend doesn't
        // support per-device async (e.g., PCIeBAR or older NCCL).
        if (backend_impl_->isMultiGpuSingleProcess() && degree() > 1)
        {
            // Release the main mutex before any blocking path
            lock.unlock();
            return allreducePerDeviceOrBarrier(tensor, stage_name, effective_count);
        }
        else
        {
            // Fallback: single-buffer API (tensor must be on local device)
            // Ensure tensor is on the local device
            DeviceId local_device = devices_[0].toLocalDeviceId();
            if (!tensor->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allreduce: Failed to ensure tensor on device");
                return false;
            }

            void *buffer = tensor->gpu_data_ptr();
            if (!buffer)
            {
                LOG_ERROR("LocalTPContext::allreduce: No device buffer available");
                return false;
            }

            size_t reduce_count = effective_count;
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            LOG_DEBUG("LocalTPContext::allreduce: Single-buffer allreduce with "
                      << reduce_count << " elements (tensor numel=" << tensor->numel() << ")");

            bool result = backend_impl_->allreduce(
                buffer, reduce_count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (result)
            {
                tensor->mark_device_dirty_with_event();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allreduce: Backend allreduce failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
    }

    bool LocalTPContext::allreduceOnStream(TensorBase *tensor, const std::string &stage_name,
                                           size_t count, void *stream,
                                           const std::string &precision)
    {
        // When no stream is provided, fall back to the normal allreduce path
        if (!stream)
        {
            return allreduce(tensor, stage_name, count);
        }

        if (!tensor)
        {
            LOG_ERROR("LocalTPContext::allreduceOnStream: null tensor");
            return false;
        }

        // Single-device context — no-op
        if (degree() == 1)
        {
            return true;
        }

        const size_t effective_count = (count > 0) ? count : tensor->numel();

        // Determine device index from tensor placement
        int device_index = -1;
        auto tensor_device = tensor->current_device();
        if (tensor_device.has_value())
        {
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i].toLocalDeviceId() == *tensor_device)
                {
                    device_index = static_cast<int>(i);
                    break;
                }
            }
        }

        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::allreduceOnStream: tensor device "
                      << (tensor_device.has_value() ? tensor_device->toString() : "none")
                      << " not found in devices list (degree=" << degree() << ")");
            return false;
        }

        // Get GPU pointer directly — do NOT call ensureOnDevice() here.
        // During HIP/CUDA graph capture, ensureOnDevice() can trigger
        // hipDeviceSynchronize() (via event waits, backend sync, or H2D copy),
        // which is illegal and poisons the capture state.
        // This is safe because:
        //   - Phase 1 (warmup) already executed all stages normally, uploading
        //     all tensors to their target devices
        //   - Phase 2 (capture) replays the same stages, so data is already on-device
        //   - If gpu_data_ptr() is null here, it's a bug in the warmup path
        void *buffer = tensor->gpu_data_ptr();
        if (!buffer)
        {
            LOG_ERROR("LocalTPContext::allreduceOnStream: null GPU buffer for slot "
                      << device_index << " — tensor was not uploaded during warmup. "
                      << "This is a bug: allreduceOnStream requires data already on-device.");
            return false;
        }

        // Issue allreduce directly on the caller's stream (graph-capturable)
        CollectiveDataType dtype = tensorDTypeToCollective(tensor);

        // =================================================================
        // FP16 mixed-precision allreduce path
        // =================================================================
        // Precision can be set per-layer via the schema precision policy,
        // or globally via LLAMINAR_ALLREDUCE_PRECISION environment variable.
        // Per-call precision (from schema) takes priority over the global env.
        const std::string &effective_precision =
            precision.empty() ? debugEnv().allreduce_precision : precision;
        const bool use_fp16_allreduce =
            effective_precision == "fp16" &&
            dtype == CollectiveDataType::FLOAT32;

        if (use_fp16_allreduce)
        {
            // Lazy-init scratch buffer vectors
            if (fp16_scratch_buffers_.empty())
            {
                fp16_scratch_buffers_.resize(degree(), nullptr);
                fp16_scratch_counts_.resize(degree(), 0);
            }

            // Ensure scratch buffer is large enough for this allreduce
            if (fp16_scratch_counts_[device_index] < effective_count)
            {
                const int ordinal = devices_[device_index].device_ordinal;
                const size_t alloc_bytes = effective_count * sizeof(uint16_t); // FP16 = 2 bytes

                // Free old buffer if resizing
                if (fp16_scratch_buffers_[device_index])
                {
#ifdef HAVE_CUDA
                    if (device_group_.allCUDA())
                        cudaFP16ScratchFree(fp16_scratch_buffers_[device_index], ordinal);
#endif
#ifdef HAVE_ROCM
                    if (device_group_.allROCm())
                        rocmFP16ScratchFree(fp16_scratch_buffers_[device_index], ordinal);
#endif
                    fp16_scratch_buffers_[device_index] = nullptr;
                }

                // Allocate new FP16 scratch buffer on the correct device
                bool alloc_ok = false;
#ifdef HAVE_CUDA
                if (device_group_.allCUDA())
                    alloc_ok = (cudaFP16ScratchAlloc(&fp16_scratch_buffers_[device_index], alloc_bytes, ordinal) == 0);
#endif
#ifdef HAVE_ROCM
                if (device_group_.allROCm())
                    alloc_ok = (rocmFP16ScratchAlloc(&fp16_scratch_buffers_[device_index], alloc_bytes, ordinal) == 0);
#endif
                if (!alloc_ok)
                {
                    LOG_WARN("LocalTPContext::allreduceOnStream: FP16 scratch alloc failed ("
                             << alloc_bytes << " bytes on device " << ordinal
                             << "), falling back to FP32 allreduce");
                    // Fall through to FP32 path below
                }
                else
                {
                    fp16_scratch_counts_[device_index] = effective_count;
                    LOG_INFO("LocalTPContext: Allocated FP16 scratch buffer: "
                             << (alloc_bytes / 1024) << " KB on device " << ordinal);
                }
            }

            // Execute FP16 allreduce if scratch is available
            if (fp16_scratch_buffers_[device_index] &&
                fp16_scratch_counts_[device_index] >= effective_count)
            {
                void *fp16_buf = fp16_scratch_buffers_[device_index];
                bool cast_ok = false;

                // Step 1: Cast FP32 → FP16 on caller's stream
#ifdef HAVE_CUDA
                if (device_group_.allCUDA())
                {
                    cast_ok = (cudaCastFP32ToFP16(
                                   static_cast<const float *>(buffer), fp16_buf,
                                   effective_count,
                                   static_cast<cudaStream_t>(stream)) == 0);
                }
#endif
#ifdef HAVE_ROCM
                if (device_group_.allROCm())
                {
                    cast_ok = (rocmCastFP32ToFP16(
                                   static_cast<const float *>(buffer), fp16_buf,
                                   effective_count, stream) == 0);
                }
#endif
                if (!cast_ok)
                {
                    LOG_WARN("LocalTPContext: FP32→FP16 cast failed, falling back to FP32");
                    // Fall through to FP32 path
                }
                else
                {
                    // Step 2: Allreduce in FP16 (half the bytes!)
                    bool ar_ok = backend_impl_->allreduceSingleDeviceOnStream(
                        fp16_buf, effective_count, CollectiveDataType::FLOAT16,
                        CollectiveOp::ALLREDUCE_SUM, device_index, stream);

                    if (!ar_ok)
                    {
                        LOG_WARN("LocalTPContext: FP16 allreduce failed: "
                                 << backend_impl_->lastError() << ", falling back to FP32");
                        // Fall through to FP32 path
                    }
                    else
                    {
                        // Step 3: Cast FP16 → FP32 back into the original buffer
                        bool back_ok = false;
#ifdef HAVE_CUDA
                        if (device_group_.allCUDA())
                        {
                            back_ok = (cudaCastFP16ToFP32(
                                           fp16_buf,
                                           static_cast<float *>(buffer),
                                           effective_count,
                                           static_cast<cudaStream_t>(stream)) == 0);
                        }
#endif
#ifdef HAVE_ROCM
                        if (device_group_.allROCm())
                        {
                            back_ok = (rocmCastFP16ToFP32(
                                           fp16_buf,
                                           static_cast<float *>(buffer),
                                           effective_count, stream) == 0);
                        }
#endif
                        if (back_ok)
                        {
                            tensor->mark_device_dirty_flags_only();
                            return true;
                        }
                        LOG_WARN("LocalTPContext: FP16→FP32 cast-back failed, data may be corrupt");
                        // Fall through but data integrity is questionable
                    }
                }
            }
        }

        // =================================================================
        // Standard FP32 allreduce path (also fallback from FP16 failures)
        // =================================================================
        bool success = backend_impl_->allreduceSingleDeviceOnStream(
            buffer, effective_count, dtype, CollectiveOp::ALLREDUCE_SUM,
            device_index, stream);

        if (success)
        {
            // Mark tensor dirty (the allreduce result is on-device)
            tensor->mark_device_dirty_flags_only();
            return true;
        }

        LOG_WARN("LocalTPContext::allreduceOnStream: on-stream allreduce not supported, "
                 "falling back to normal path for stage="
                 << stage_name);
        return allreduce(tensor, stage_name, effective_count);
    }

    bool LocalTPContext::allreduce(const TensorBase *input, TensorBase *output)
    {
        if (!input || !output)
        {
            LOG_ERROR("LocalTPContext::allreduce: null input or output tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy
        if (degree() == 1)
        {
            // Copy input to output
            const float *src = input->data();
            float *dst = output->mutable_data();
            size_t count = std::min(input->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allreduce (out-of-place): Backend not initialized, skipping");
            // Fall back to copy
            const float *src = input->data();
            float *dst = output->mutable_data();
            size_t count = std::min(input->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // For out-of-place allreduce:
        // 1. Copy input to output
        // 2. Perform in-place allreduce on output
        LOG_DEBUG("LocalTPContext::allreduce (out-of-place): copying input to output first");

        // Copy on host first
        const float *src = input->data();
        float *dst = output->mutable_data();
        size_t count = std::min(input->numel(), output->numel());
        std::memcpy(dst, src, count * sizeof(float));

        // Now delegate to in-place allreduce (need to cast away const for the API)
        // The mutex is already held, so we call directly without re-locking
        return allreduceImpl(output);
    }

    // Private implementation that assumes lock is already held
    bool LocalTPContext::allreduceImpl(TensorBase *tensor)
    {
        if (!tensor)
        {
            return false;
        }

        if (degree() == 1)
        {
            return true;
        }

        if (!backend_initialized_ || !backend_impl_)
        {
            return true;
        }

        if (backend_impl_->isMultiGpuSingleProcess())
        {
            auto buffers = getDeviceBuffers(tensor);
            if (buffers.size() != static_cast<size_t>(degree()))
            {
                LOG_ERROR("LocalTPContext::allreduceImpl: Failed to get device buffers");
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            return backend_impl_->allreduceMulti(
                buffers, count, dtype, CollectiveOp::ALLREDUCE_SUM);
        }
        else
        {
            DeviceId local_device = devices_[0].toLocalDeviceId();
            if (!tensor->ensureOnDevice(local_device))
            {
                return false;
            }

            void *buffer = tensor->gpu_data_ptr();
            if (!buffer)
            {
                return false;
            }

            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);

            // PCIeBAR Backend: Use registered allreduce path (same logic as main allreduce)
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            if (backend_ == CollectiveBackendType::PCIE_BAR)
            {
                if (!ensurePCIeBarBuffersRegistered(tensor))
                {
                    return false;
                }

                auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
                if (pcie_backend)
                {
                    // Copy tensor data to the registered BAR buffer on ROCm side
                    DeviceId rocm_device;
                    for (const auto &device : devices_)
                    {
                        DeviceId local_id = device.toLocalDeviceId();
                        if (local_id.is_rocm())
                        {
                            rocm_device = local_id;
                            break;
                        }
                    }

                    auto rocm_buf_opt = pcie_backend->getBuffer(pciebar_collective_id_, rocm_device);
                    if (rocm_buf_opt.has_value() && rocm_buf_opt->ptr)
                    {
                        const float *host_data = tensor->data();
                        std::memcpy(rocm_buf_opt->ptr, host_data, count * sizeof(float));
                    }
                }

                bool result = backend_impl_->allreduceRegistered(
                    pciebar_collective_id_, count, dtype, CollectiveOp::ALLREDUCE_SUM);

                if (result)
                {
                    tensor->mark_device_dirty_with_event();
                }
                return result;
            }
#endif

            bool result = backend_impl_->allreduce(
                buffer, count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (result)
            {
                tensor->mark_device_dirty_with_event();
            }
            return result;
        }
    }

    // =========================================================================
    // PCIeBAR Barrier-Synchronized Allreduce
    // =========================================================================

    bool LocalTPContext::allreduceWithBarrier(TensorBase *tensor, const std::string &stage_name, size_t count)
    {
        const int num_participants = degree();

        std::unique_lock<std::mutex> lock(barrier_mutex_);

        // Capture current generation to detect spurious wakeups
        uint64_t my_generation = barrier_generation_.load();

        // Increment arrival count
        int arrival_order = barrier_count_.fetch_add(1);

        if (arrival_order == 0)
        {
            // First arrival: initialize tensor collection vector and store stage name + count
            barrier_tensors_.clear();
            barrier_tensors_.resize(num_participants, nullptr);
            barrier_tensor_ = tensor;         // Keep for backward compatibility
            barrier_stage_name_ = stage_name; // Store stage name for BAR-backed tensor lookup
            barrier_element_count_ = count;   // Store element count (0 = use tensor->numel())
            LOG_DEBUG("LocalTPContext::allreduceWithBarrier: First arrival (device thread), "
                      << "stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << ", count=" << count << " (0=use numel)"
                      << ", waiting for " << (num_participants - 1) << " more devices");
        }

        // Store this device's tensor in the collection
        // Each device thread stores its own tensor at its arrival slot
        barrier_tensors_[arrival_order] = tensor;

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: Device arrival #" << (arrival_order + 1)
                                                                           << " of " << num_participants
                                                                           << " (tensor ptr=" << tensor << ")");

        if (arrival_order + 1 < num_participants)
        {
            // Not the last arrival: wait for completion with timeout
            // Use longer timeout for first barrier to accommodate GPU workspace allocation
            // (hipMalloc can take 30-60s per device and is serialized within a process)
            const auto BARRIER_TIMEOUT = first_barrier_completed_.load()
                                             ? std::chrono::seconds(30)
                                             : std::chrono::seconds(300);

            bool completed = barrier_cv_.wait_for(lock, BARRIER_TIMEOUT, [this, my_generation]()
                                                  { return barrier_generation_.load() > my_generation; });

            if (!completed)
            {
                // Timeout - likely a deadlock or missing participant
                int timeout_secs = first_barrier_completed_.load() ? 30 : 300;
                LOG_ERROR("LocalTPContext::allreduceWithBarrier: TIMEOUT after " << timeout_secs
                                                                                 << "s waiting for barrier! "
                                                                                 << "arrival_order=" << arrival_order << ", expected=" << num_participants
                                                                                 << " devices. Possible causes: missing device thread, kernel crash, or deadlock.");

                // Reset barrier state to allow recovery
                barrier_count_.store(0);
                barrier_generation_.fetch_add(1);
                barrier_tensors_.clear();
                barrier_tensor_ = nullptr;
                barrier_stage_name_.clear();
                barrier_element_count_ = 0;

                lock.unlock();
                barrier_cv_.notify_all(); // Wake any other waiters
                return false;
            }

            // Woke up - get the shared result
            bool result = barrier_result_;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrier: Waiter released with result=" << result);
            return result;
        }

        // =====================================================================
        // LAST ARRIVAL: Execute the actual allreduce
        // =====================================================================
        // All other threads are waiting, so we have exclusive access to barrier_tensors_

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: All " << num_participants
                                                               << " devices arrived, executing PCIeBAR allreduce"
                                                               << " (count=" << barrier_element_count_ << ")");

        // The actual PCIeBAR transfer using all collected tensors and stored count
        bool success = executePCIeBarAllreduce(nullptr, barrier_element_count_);

        // Store result and signal completion
        barrier_result_ = success;
        if (success)
            first_barrier_completed_.store(true);
        barrier_tensors_.clear();
        barrier_tensor_ = nullptr;
        barrier_stage_name_.clear();
        barrier_element_count_ = 0;
        barrier_count_.store(0);
        barrier_generation_.fetch_add(1);

        LOG_DEBUG("LocalTPContext::allreduceWithBarrier: PCIeBAR allreduce completed with result="
                  << success << ", releasing waiters (generation=" << barrier_generation_.load() << ")");

        lock.unlock();
        barrier_cv_.notify_all();

        return success;
    }

    // =========================================================================
    // Multi-GPU (NCCL/RCCL) Barrier-Synchronized Allreduce
    // =========================================================================
    //
    // For LOCAL TP with NCCL/RCCL, multiple device threads call allreduce()
    // concurrently, each with its OWN tensor. We need to:
    // 1. Collect all device buffers via barrier synchronization
    // 2. Have ONE thread execute allreduceMulti with all buffers
    // 3. All threads return after the collective completes
    //
    // This is necessary because TensorBase can only exist on ONE GPU at a time,
    // so we cannot use getDeviceBuffers() to gather buffers from a single tensor.
    // =========================================================================

    bool LocalTPContext::allreducePerDeviceOrBarrier(TensorBase *tensor,
                                                     const std::string &stage_name, size_t count)
    {
        // Fast path: per-device async allreduce — no barrier, no buffer collection.
        // Each device thread independently calls RCCL/NCCL AllReduce with its own
        // communicator. RCCL internally matches calls from different threads.
        // Host returns immediately; all sync is GPU-side via stream deps.

        // 1. Determine device index from tensor placement
        int device_index = -1;
        auto tensor_device = tensor->current_device();
        if (tensor_device.has_value())
        {
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i].toLocalDeviceId() == *tensor_device)
                {
                    device_index = static_cast<int>(i);
                    break;
                }
            }
        }

        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::allreducePerDeviceOrBarrier: tensor device "
                      << (tensor_device.has_value() ? tensor_device->toString() : "none")
                      << " not found in devices list (degree=" << degree() << ")");
            return false;
        }

        // 2. Ensure tensor is on device and get GPU pointer
        DeviceId expected_device = devices_[device_index].toLocalDeviceId();
        if (!tensor->ensureOnDevice(expected_device))
        {
            LOG_ERROR("LocalTPContext::allreducePerDeviceOrBarrier: ensureOnDevice failed for slot "
                      << device_index);
            return false;
        }

        void *buffer = tensor->gpu_data_ptr();
        if (!buffer)
        {
            LOG_ERROR("LocalTPContext::allreducePerDeviceOrBarrier: null GPU buffer for slot "
                      << device_index);
            return false;
        }

        // 3. Try per-device async allreduce (barrier-free)
        CollectiveDataType dtype = tensorDTypeToCollective(tensor);
        bool success = backend_impl_->allreduceSingleDeviceAsync(
            buffer, count, dtype, CollectiveOp::ALLREDUCE_SUM, device_index);

        if (success)
        {
            // Mark tensor dirty (flags only — RCCL may not have completed yet,
            // but compute stream has a WaitEvent dep on RCCL completion)
            tensor->mark_device_dirty_flags_only();
            return true;
        }

        // 4. Fallback: barrier-synchronized path (for backends that don't support per-device async)
        LOG_DEBUG("LocalTPContext::allreducePerDeviceOrBarrier: per-device async not supported, "
                  "falling back to barrier path for stage="
                  << stage_name);
        return allreduceWithBarrierMultiGpu(tensor, stage_name, count);
    }

    bool LocalTPContext::allreduceWithBarrierMultiGpu(TensorBase *tensor, const std::string &stage_name, size_t count)
    {
        const int num_participants = degree();
        const bool strict_stage_barrier = debugEnv().validation.strict_local_tp_stage_barrier;

        // Determine which device index this tensor belongs to BEFORE taking the lock
        // This is critical: buffers must be ordered by device index, NOT arrival order
        int device_index = -1;
        auto tensor_device = tensor->current_device();
        if (tensor_device.has_value())
        {
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i].toLocalDeviceId() == *tensor_device)
                {
                    device_index = static_cast<int>(i);
                    break;
                }
            }
        }

        if (device_index < 0 || device_index >= num_participants)
        {
            LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: tensor device "
                      << (tensor_device.has_value() ? tensor_device->toString() : "none")
                      << " not found in LocalTPContext devices list (degree=" << num_participants << ")");
            return false;
        }

        std::unique_lock<std::mutex> lock(barrier_mutex_);

        // Capture current generation to detect spurious wakeups
        uint64_t my_generation = barrier_generation_.load();

        // Increment arrival count
        int arrival_order = barrier_count_.fetch_add(1);

        if (arrival_order >= num_participants)
        {
            LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: barrier overflow detected "
                      << "arrival_order=" << arrival_order << " participants=" << num_participants
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " generation=" << my_generation << " - resetting barrier state");

            barrier_count_.store(0);
            barrier_generation_.fetch_add(1);
            barrier_tensors_.clear();
            barrier_stage_name_.clear();
            barrier_element_count_ = 0;
            barrier_result_ = false;

            lock.unlock();
            barrier_cv_.notify_all();
            return false;
        }

        if (arrival_order == 0)
        {
            // First arrival: initialize tensor collection vector and store count
            barrier_tensors_.clear();
            barrier_tensors_.resize(num_participants, nullptr);
            barrier_watch_checksums_.assign(num_participants, 0);
            barrier_watch_sample_bytes_.assign(num_participants, 0);
            barrier_watch_sample_offsets_.assign(num_participants, 0);
            barrier_watch_checksum_valid_.assign(num_participants, false);
            barrier_element_count_ = count;
            barrier_stage_name_ = stage_name;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: First arrival (device thread), "
                      << "stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << ", count=" << count << " (0=use numel)"
                      << ", waiting for " << (num_participants - 1) << " more devices");
        }
        else if (strict_stage_barrier && barrier_stage_name_ != stage_name)
        {
            LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: mixed stage names in the same barrier generation "
                      << "expected='" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                      << "' got='" << (stage_name.empty() ? "(none)" : stage_name)
                      << "' arrival_order=" << arrival_order
                      << " participants=" << num_participants
                      << " generation=" << my_generation);

            barrier_count_.store(0);
            barrier_generation_.fetch_add(1);
            barrier_tensors_.clear();
            barrier_stage_name_.clear();
            barrier_watch_checksums_.clear();
            barrier_watch_sample_bytes_.clear();
            barrier_watch_sample_offsets_.clear();
            barrier_watch_checksum_valid_.clear();
            barrier_element_count_ = 0;
            barrier_result_ = false;

            lock.unlock();
            barrier_cv_.notify_all();
            return false;
        }

        // Store this device's tensor at its DEVICE INDEX slot (not arrival order!)
        // This ensures buffers[i] corresponds to device_ordinals_[i] in RCCL
        barrier_tensors_[device_index] = tensor;

        DeviceId expected_device_for_slot = devices_[device_index].toLocalDeviceId();
        if (!tensor->ensureOnDevice(expected_device_for_slot))
        {
            LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: failed ensureOnDevice at arrival for slot "
                      << device_index << " expected_device=" << expected_device_for_slot.toString());
            barrier_count_.store(0);
            barrier_generation_.fetch_add(1);
            barrier_tensors_.clear();
            barrier_stage_name_.clear();
            barrier_watch_checksums_.clear();
            barrier_watch_sample_bytes_.clear();
            barrier_watch_sample_offsets_.clear();
            barrier_watch_checksum_valid_.clear();
            barrier_element_count_ = 0;
            barrier_result_ = false;

            lock.unlock();
            barrier_cv_.notify_all();
            return false;
        }

        void *arrival_ptr = tensor->gpu_data_ptr();
        if (!arrival_ptr)
        {
            LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: missing GPU buffer at arrival for slot "
                      << device_index << " expected_device=" << expected_device_for_slot.toString());
            barrier_count_.store(0);
            barrier_generation_.fetch_add(1);
            barrier_tensors_.clear();
            barrier_stage_name_.clear();
            barrier_element_count_ = 0;
            barrier_result_ = false;

            lock.unlock();
            barrier_cv_.notify_all();
            return false;
        }

        const auto arrival_current_device = tensor->current_device();
        LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: arrival tensor diagnostics"
                  << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                  << " slot=" << device_index
                  << " expected_device=" << expected_device_for_slot.toString()
                  << " tensor=" << static_cast<void *>(tensor)
                  << " tensor_name=" << (tensor->debugName().empty() ? "(unnamed)" : tensor->debugName())
                  << " home_device=" << tensor->home_device().toString()
                  << " current_device=" << (arrival_current_device.has_value() ? arrival_current_device->toString() : "none")
                  << " gpu_ptr=" << arrival_ptr
                  << " requested_count=" << count
                  << " tensor_numel=" << tensor->numel());

#ifdef HAVE_ROCM
        uint64_t arrival_watch_checksum = 0;
        size_t arrival_watch_sample_bytes = 0;
        size_t arrival_watch_sample_offset = 0;
        if (expected_device_for_slot.is_rocm() &&
            !validateRocmAllreducePointerForSlot(barrier_stage_name_, "arrival", device_index, expected_device_for_slot, tensor, arrival_ptr,
                                                 &arrival_watch_checksum, &arrival_watch_sample_bytes, &arrival_watch_sample_offset))
        {
            barrier_count_.store(0);
            barrier_generation_.fetch_add(1);
            barrier_tensors_.clear();
            barrier_stage_name_.clear();
            barrier_watch_checksums_.clear();
            barrier_watch_sample_bytes_.clear();
            barrier_watch_sample_offsets_.clear();
            barrier_watch_checksum_valid_.clear();
            barrier_element_count_ = 0;
            barrier_result_ = false;

            lock.unlock();
            barrier_cv_.notify_all();
            return false;
        }

        if (arrival_watch_sample_bytes > 0)
        {
            barrier_watch_checksums_[device_index] = arrival_watch_checksum;
            barrier_watch_sample_bytes_[device_index] = arrival_watch_sample_bytes;
            barrier_watch_sample_offsets_[device_index] = arrival_watch_sample_offset;
            barrier_watch_checksum_valid_[device_index] = true;
        }

        if (debugEnv().validation.validate_gpu_ptrs && expected_device_for_slot.is_rocm())
        {
            auto *rocm_backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(expected_device_for_slot));
            if (rocm_backend)
            {
                rocm_backend->setDevice(expected_device_for_slot.toKernelDeviceIndex());
                bool is_device_ptr = false;
                bool is_host_ptr = false;
                bool is_managed = false;
                int attr_device = -1;
                (void)rocm_backend->queryPointerAttributes(arrival_ptr, is_device_ptr, is_host_ptr, is_managed, attr_device);

                ROCmPointerOwnerInfo owner;
                if (ROCmBackend::queryPointerOwner(arrival_ptr, owner))
                {
                    LOG_DEBUG("[LOCALTP_PTR_OWNER] phase=arrival"
                              << " stage=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                              << " slot=" << device_index
                              << " expected_device=" << expected_device_for_slot.toString()
                              << " ptr=" << arrival_ptr
                              << " attr_device=" << attr_device
                              << " is_device_ptr=" << (is_device_ptr ? 1 : 0)
                              << " is_host_ptr=" << (is_host_ptr ? 1 : 0)
                              << " is_managed=" << (is_managed ? 1 : 0)
                              << " owner_device=" << owner.device_id
                              << " owner_base=" << owner.base_ptr
                              << " owner_bytes=" << owner.size_bytes
                              << " owner_seq=" << owner.sequence
                              << " owner_thread=" << owner.thread_hash
                              << " tensor=" << static_cast<void *>(tensor)
                              << " tensor_name=" << (tensor->debugName().empty() ? "(unnamed)" : tensor->debugName()));
                }
            }
        }
#endif

        LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Device arrival #" << (arrival_order + 1)
                                                                                   << " of " << num_participants
                                                                                   << " (tensor ptr=" << tensor
                                                                                   << ", device_index=" << device_index << ")");

        if (arrival_order + 1 < num_participants)
        {
            // Not the last arrival: wait for completion with timeout
            // Use longer timeout for first barrier to accommodate GPU workspace allocation
            // (hipMalloc can take 30-60s per device and is serialized within a process)
            const auto BARRIER_TIMEOUT = first_barrier_completed_.load()
                                             ? std::chrono::seconds(30)
                                             : std::chrono::seconds(300);

            bool completed = barrier_cv_.wait_for(lock, BARRIER_TIMEOUT, [this, my_generation]()
                                                  { return barrier_generation_.load() > my_generation; });

            if (!completed)
            {
                // Timeout - likely a deadlock or missing participant
                int timeout_secs = first_barrier_completed_.load() ? 30 : 300;
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: TIMEOUT after " << timeout_secs
                                                                                         << "s waiting for barrier! "
                                                                                         << "arrival_order=" << arrival_order << ", expected=" << num_participants
                                                                                         << " devices. Possible causes: missing device thread, kernel crash, or deadlock.");

                // Reset barrier state to allow recovery
                barrier_count_.store(0);
                barrier_generation_.fetch_add(1);
                barrier_tensors_.clear();
                barrier_element_count_ = 0;

                lock.unlock();
                barrier_cv_.notify_all(); // Wake any other waiters
                return false;
            }

            // Woke up - get the shared result
            bool result = barrier_result_;
            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Waiter released with result=" << result);

            return result;
        }

        // =====================================================================
        // LAST ARRIVAL: Execute the actual multi-GPU allreduce
        // =====================================================================
        // All other threads are waiting, so we have exclusive access to barrier_tensors_

        LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: All " << num_participants
                                                                       << " devices arrived, executing multi-GPU allreduce"
                                                                       << " (count=" << barrier_element_count_ << ")");

        // Collect device buffers from all tensors
        // CRITICAL: Each tensor must be on its own device, and we need to ensure that
        std::vector<void *> buffers;
        buffers.reserve(num_participants);

        // Determine effective count (use first tensor's numel if count is 0)
        size_t effective_count = barrier_element_count_;
        if (effective_count == 0 && barrier_tensors_[0] != nullptr)
        {
            effective_count = barrier_tensors_[0]->numel();
        }

        // Get buffer pointer from each tensor (they should already be on their respective devices)
        for (int i = 0; i < num_participants; ++i)
        {
            TensorBase *t = barrier_tensors_[i];
            if (!t)
            {
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: null tensor at slot " << i);
                barrier_result_ = false;
                goto cleanup;
            }

            DeviceId expected_device = devices_[i].toLocalDeviceId();
            if (!t->ensureOnDevice(expected_device))
            {
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: failed ensureOnDevice for slot "
                          << i << " expected_device=" << expected_device.toString());
                barrier_result_ = false;
                goto cleanup;
            }

            void *ptr = t->gpu_data_ptr();
            if (!ptr)
            {
                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: tensor at slot " << i
                                                                                          << " has no GPU buffer");
                barrier_result_ = false;
                goto cleanup;
            }

            const auto prelaunch_current_device = t->current_device();
            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: prelaunch slot diagnostics"
                      << " stage=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                      << " slot=" << i
                      << " expected_device=" << expected_device.toString()
                      << " tensor=" << static_cast<void *>(t)
                      << " tensor_name=" << (t->debugName().empty() ? "(unnamed)" : t->debugName())
                      << " home_device=" << t->home_device().toString()
                      << " current_device=" << (prelaunch_current_device.has_value() ? prelaunch_current_device->toString() : "none")
                      << " gpu_ptr=" << ptr
                      << " effective_count=" << effective_count
                      << " tensor_numel=" << t->numel());

#ifdef HAVE_ROCM
            uint64_t prelaunch_watch_checksum = 0;
            size_t prelaunch_watch_sample_bytes = 0;
            size_t prelaunch_watch_sample_offset = 0;
            if (expected_device.is_rocm() &&
                !validateRocmAllreducePointerForSlot(barrier_stage_name_, "prelaunch", i, expected_device, t, ptr,
                                                     &prelaunch_watch_checksum, &prelaunch_watch_sample_bytes, &prelaunch_watch_sample_offset))
            {
                barrier_result_ = false;
                goto cleanup;
            }

            if (prelaunch_watch_sample_bytes > 0)
            {
                const bool had_arrival_checksum = (i < static_cast<int>(barrier_watch_checksum_valid_.size()))
                                                      ? barrier_watch_checksum_valid_[i]
                                                      : false;
                if (had_arrival_checksum &&
                    barrier_watch_sample_bytes_[i] == prelaunch_watch_sample_bytes &&
                    barrier_watch_sample_offsets_[i] == prelaunch_watch_sample_offset &&
                    barrier_watch_checksums_[i] != prelaunch_watch_checksum)
                {
                    LOG_ERROR("[LOCALTP_PTR_WATCH_CHECKSUM_MISMATCH]"
                              << " stage=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                              << " slot=" << i
                              << " generation=" << my_generation
                              << " arrival_checksum=" << barrier_watch_checksums_[i]
                              << " prelaunch_checksum=" << prelaunch_watch_checksum
                              << " sample_bytes=" << prelaunch_watch_sample_bytes
                              << " sample_offset=" << prelaunch_watch_sample_offset
                              << " expected_device=" << expected_device.toString()
                              << " tensor=" << static_cast<void *>(t)
                              << " tensor_name=" << (t->debugName().empty() ? "(unnamed)" : t->debugName()));
                    ROCmBackend::dumpRecentPointerEvents(128);
                    barrier_result_ = false;
                    goto cleanup;
                }
            }

            if (debugEnv().validation.validate_gpu_ptrs && expected_device.is_rocm())
            {
                ROCmPointerOwnerInfo owner;
                if (ROCmBackend::queryPointerOwner(ptr, owner))
                {
                    auto *rocm_backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(expected_device));
                    bool is_device_ptr = false;
                    bool is_host_ptr = false;
                    bool is_managed = false;
                    int attr_device = -1;
                    if (rocm_backend)
                    {
                        rocm_backend->setDevice(expected_device.toKernelDeviceIndex());
                        (void)rocm_backend->queryPointerAttributes(ptr, is_device_ptr, is_host_ptr, is_managed, attr_device);
                    }

                    LOG_DEBUG("[LOCALTP_PTR_OWNER] phase=prelaunch"
                              << " stage=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                              << " slot=" << i
                              << " expected_device=" << expected_device.toString()
                              << " ptr=" << ptr
                              << " attr_device=" << attr_device
                              << " is_device_ptr=" << (is_device_ptr ? 1 : 0)
                              << " is_host_ptr=" << (is_host_ptr ? 1 : 0)
                              << " is_managed=" << (is_managed ? 1 : 0)
                              << " owner_device=" << owner.device_id
                              << " owner_base=" << owner.base_ptr
                              << " owner_bytes=" << owner.size_bytes
                              << " owner_seq=" << owner.sequence
                              << " owner_thread=" << owner.thread_hash
                              << " tensor=" << static_cast<void *>(t)
                              << " tensor_name=" << (t->debugName().empty() ? "(unnamed)" : t->debugName()));

                    const int expected_ordinal = expected_device.rocm_ordinal();
                    if (owner.device_id != expected_ordinal)
                    {
                        LOG_ERROR("[LOCALTP_GPU_PTR_MISMATCH] slot=" << i
                                                                     << " ptr=" << ptr
                                                                     << " owner.dev=" << owner.device_id
                                                                     << " expected.dev=" << expected_ordinal
                                                                     << " owner.base=" << owner.base_ptr
                                                                     << " owner.bytes=" << owner.size_bytes
                                                                     << " owner.seq=" << owner.sequence
                                                                     << " owner.thread=" << owner.thread_hash
                                                                     << " tensor=" << static_cast<void *>(t)
                                                                     << " tensor_device="
                                                                     << (t->current_device().has_value() ? t->current_device()->toString() : "none"));
                        ROCmBackend::dumpRecentPointerEvents(128);
                        barrier_result_ = false;
                        goto cleanup;
                    }
                }
            }
#endif

            // TRACE: Log detailed buffer info including device for debugging memory faults
            LOG_TRACE("LocalTPContext::allreduceWithBarrierMultiGpu: BUFFER[" << i << "] "
                                                                              << "ptr=" << ptr << " tensor=" << static_cast<void *>(t)
                                                                              << " device=" << (t->current_device().has_value() ? t->current_device()->toString() : "none")
                                                                              << " name=" << (t->debugName().empty() ? "(unnamed)" : t->debugName())
                                                                              << " numel=" << t->numel());

            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Buffer " << i
                                                                              << " ptr=" << ptr
                                                                              << " from tensor=" << t);
            buffers.push_back(ptr);
        }

        {
            // Execute the multi-GPU allreduce
            CollectiveDataType dtype = tensorDTypeToCollective(barrier_tensors_[0]);
            const bool serialize_local_tp_launch = debugEnv().validation.serialize_local_tp_allreduce_launch;
            static std::mutex local_tp_launch_mutex;
            std::unique_lock<std::mutex> launch_lock(local_tp_launch_mutex, std::defer_lock);
            if (serialize_local_tp_launch)
            {
                launch_lock.lock();
                LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: serialized allreduce launch lock acquired "
                          << "stage=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                          << " participants=" << num_participants
                          << " count=" << effective_count);
            }

            // Phase 3 runtime policy: make graph-capture support explicit.
            // If users enable global GPU graph mode without segmented-collective
            // support, we fail fast with a clear marker instead of attempting an
            // undefined collective scheduling path.
            if (backend_ == CollectiveBackendType::NCCL)
            {
                std::string graph_policy_reason;
                const bool graph_supported = isLocalTPNCCLGraphPolicySupported(&graph_policy_reason);
                if (!graph_supported)
                {
                    if (!logged_graph_policy_reject_marker_.exchange(true))
                    {
                        LOG_ERROR("LOCALTP_NCCL_GRAPH_POLICY=UNSUPPORTED reason=" << graph_policy_reason);
                    }

                    LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: Unsupported LocalTP NCCL graph mode. "
                              << "Enable LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED=1 when LLAMINAR_GPU_GRAPHS=1");
                    barrier_result_ = false;
                    goto cleanup;
                }

                if (!logged_graph_policy_allow_marker_.exchange(true))
                {
                    LOG_INFO("LOCALTP_NCCL_GRAPH_POLICY=SUPPORTED reason=" << graph_policy_reason);
                }
            }

            // Phase 2 guardrail: fail fast on shape/device/dtype mismatches before
            // entering backend collective code. This makes bugs deterministic and
            // easier to diagnose than backend-side "unhandled" failures.
            if (!validateBarrierTensorSetForMultiGpuAllreduce(effective_count, dtype))
            {
                barrier_result_ = false;
                goto cleanup;
            }

            // TRACE: Log all buffer pointers before allreduce
            LOG_TRACE("LocalTPContext::allreduceWithBarrierMultiGpu: ALLREDUCE START "
                      << "num_buffers=" << buffers.size() << " count=" << effective_count);
            for (size_t i = 0; i < buffers.size(); ++i)
            {
                LOG_TRACE("  allreduce buffer[" << i << "] = " << buffers[i]);
            }

            LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Calling allreduceMulti with "
                      << buffers.size() << " buffers, " << effective_count << " elements");

            if (debugEnv().validation.sync_local_tp_allreduce)
            {
#ifdef HAVE_ROCM
                for (int i = 0; i < num_participants; ++i)
                {
                    DeviceId sync_device = devices_[i].toLocalDeviceId();
                    if (!sync_device.is_rocm())
                    {
                        continue;
                    }

                    auto *rocm_backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(sync_device));
                    if (!rocm_backend)
                    {
                        LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: missing ROCm backend for pre-allreduce sync "
                                  << sync_device.toString());
                        barrier_result_ = false;
                        goto cleanup;
                    }

                    rocm_backend->setDevice(sync_device.toKernelDeviceIndex());
                    if (!rocm_backend->synchronize(sync_device.toKernelDeviceIndex()))
                    {
                        LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: pre-allreduce synchronize failed for "
                                  << sync_device.toString());
                        barrier_result_ = false;
                        goto cleanup;
                    }
                }
#endif
            }

            bool success = backend_impl_->allreduceMultiWithComputeDeps(
                buffers, effective_count, dtype, CollectiveOp::ALLREDUCE_SUM);

            if (backend_ == CollectiveBackendType::NCCL)
            {
                nccl_allreduce_attempts_.fetch_add(1);
            }

            if (!success)
            {
                const std::string backend_error = backend_impl_->lastError();

                if (backend_ == CollectiveBackendType::NCCL)
                {
                    nccl_allreduce_failures_.fetch_add(1);
                }

                LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: allreduceMulti FAILED: "
                          << backend_error);

                barrier_result_ = false;
                goto cleanup;
            }

            // TRACE: Log after allreduce+sync completes
            LOG_TRACE("LocalTPContext::allreduceWithBarrierMultiGpu: ALLREDUCE+SYNC COMPLETE");

            if (debugEnv().validation.sync_local_tp_allreduce)
            {
#ifdef HAVE_ROCM
                for (int i = 0; i < num_participants; ++i)
                {
                    DeviceId sync_device = devices_[i].toLocalDeviceId();
                    if (!sync_device.is_rocm())
                    {
                        continue;
                    }

                    auto *rocm_backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(sync_device));
                    if (!rocm_backend)
                    {
                        LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: missing ROCm backend for post-allreduce sync "
                                  << sync_device.toString());
                        barrier_result_ = false;
                        goto cleanup;
                    }

                    rocm_backend->setDevice(sync_device.toKernelDeviceIndex());
                    if (!rocm_backend->synchronize(sync_device.toKernelDeviceIndex()))
                    {
                        LOG_ERROR("LocalTPContext::allreduceWithBarrierMultiGpu: post-allreduce synchronize failed for "
                                  << sync_device.toString());
                        barrier_result_ = false;
                        goto cleanup;
                    }
                }
#endif
            }

#ifdef HAVE_ROCM
            if (debugEnv().validation.validate_gpu_ptrs)
            {
                for (int i = 0; i < num_participants; ++i)
                {
                    TensorBase *t = barrier_tensors_[i];
                    if (!t)
                    {
                        continue;
                    }

                    DeviceId expected_device = devices_[i].toLocalDeviceId();
                    if (!expected_device.is_rocm())
                    {
                        continue;
                    }

                    void *ptr = t->gpu_data_ptr();
                    if (!ptr)
                    {
                        LOG_ERROR("[LOCALTP_PTR_OWNER] phase=postallreduce slot=" << i
                                                                                  << " stage=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                                                                                  << " expected_device=" << expected_device.toString()
                                                                                  << " ptr=null");
                        continue;
                    }

                    auto *rocm_backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(expected_device));
                    bool is_device_ptr = false;
                    bool is_host_ptr = false;
                    bool is_managed = false;
                    int attr_device = -1;
                    if (rocm_backend)
                    {
                        rocm_backend->setDevice(expected_device.toKernelDeviceIndex());
                        (void)rocm_backend->queryPointerAttributes(ptr, is_device_ptr, is_host_ptr, is_managed, attr_device);
                    }

                    ROCmPointerOwnerInfo owner;
                    if (ROCmBackend::queryPointerOwner(ptr, owner))
                    {
                        LOG_DEBUG("[LOCALTP_PTR_OWNER] phase=postallreduce"
                                  << " stage=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_)
                                  << " slot=" << i
                                  << " expected_device=" << expected_device.toString()
                                  << " ptr=" << ptr
                                  << " attr_device=" << attr_device
                                  << " is_device_ptr=" << (is_device_ptr ? 1 : 0)
                                  << " is_host_ptr=" << (is_host_ptr ? 1 : 0)
                                  << " is_managed=" << (is_managed ? 1 : 0)
                                  << " owner_device=" << owner.device_id
                                  << " owner_base=" << owner.base_ptr
                                  << " owner_bytes=" << owner.size_bytes
                                  << " owner_seq=" << owner.sequence
                                  << " owner_thread=" << owner.thread_hash
                                  << " tensor=" << static_cast<void *>(t)
                                  << " tensor_name=" << (t->debugName().empty() ? "(unnamed)" : t->debugName()));
                    }
                }
            }
#endif

            // Mark all tensors as device-dirty (data was modified on GPU).
            // Use flags-only variant because with non-blocking allreduce
            // (allreduceMultiWithComputeDeps), the RCCL work may not have
            // completed on the GPU yet. Recording an event here would be
            // on the wrong stream. The compute stream has a WaitEvent dep
            // on the RCCL completion event, so subsequent GPU kernels are
            // safe. If data() is later called (e.g., for sampling), it
            // falls back to a full device sync which IS correct.
            for (int i = 0; i < num_participants; ++i)
            {
                if (barrier_tensors_[i])
                {
                    barrier_tensors_[i]->mark_device_dirty_flags_only();
                }
            }

            if (backend_ == CollectiveBackendType::NCCL)
            {
                nccl_allreduce_success_.fetch_add(1);
                if (!logged_real_path_marker_.exchange(true))
                {
                    LOG_INFO("LOCALTP_NCCL_PATH=REAL backend=NCCL collective=allreduce_multi count="
                             << effective_count << " participants=" << num_participants);
                }
            }

            barrier_result_ = true;
        }

    cleanup:
        // Clear barrier state and signal completion
        barrier_tensors_.clear();
        barrier_stage_name_.clear();
        barrier_watch_checksums_.clear();
        barrier_watch_sample_bytes_.clear();
        barrier_watch_sample_offsets_.clear();
        barrier_watch_checksum_valid_.clear();
        barrier_element_count_ = 0;
        barrier_count_.store(0);
        barrier_generation_.fetch_add(1);

        bool final_result = barrier_result_;
        if (final_result)
            first_barrier_completed_.store(true);

        LOG_DEBUG("LocalTPContext::allreduceWithBarrierMultiGpu: Multi-GPU allreduce completed with result="
                  << final_result << ", releasing waiters (generation=" << barrier_generation_.load() << ")");

        lock.unlock();
        barrier_cv_.notify_all();

        return final_result;
    }

    bool LocalTPContext::executePCIeBarAllreduce(TensorBase * /* unused_tensor */, size_t count_param)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // ===========================================================================
        // PHASE 4 IMPLEMENTATION: Zero-Copy Allreduce using BAR-Backed Tensors
        // ===========================================================================
        //
        // For LOCAL TP with PCIeBAR, the key insight is:
        // - Each device has its OWN output tensor for row-parallel operations
        // - These tensors were allocated as BAR-backed and registered via registerBARBackedOutput()
        // - The stage name (stored in barrier_stage_name_) identifies which tensors to use
        //
        // Zero-Copy Flow:
        // 1. Lookup BAR-backed tensors by stage name
        // 2. CUDA device reads from ROCm's BAR region directly (zero-copy)
        // 3. Sum: cuda_output = cuda_partial + rocm_partial
        // 4. Write result back to ROCm via BAR (so both have the reduced value)
        // ===========================================================================

        // Lock the main mutex for backend operations
        std::lock_guard<std::mutex> lock(mutex_);

        // Find which device is CUDA and which is ROCm
        int cuda_idx = -1;
        int rocm_idx = -1;
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            DeviceId local_id = devices_[i].toLocalDeviceId();
            if (local_id.is_cuda())
                cuda_idx = static_cast<int>(i);
            else if (local_id.is_rocm())
                rocm_idx = static_cast<int>(i);
        }

        if (cuda_idx < 0 || rocm_idx < 0)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Need exactly one CUDA and one ROCm device");
            return false;
        }

        // ===========================================================================
        // PRIMARY PATH: Use registered BAR-backed tensors for zero-copy operation
        // ===========================================================================
        if (!barrier_stage_name_.empty() && hasBARBackedOutputs(barrier_stage_name_))
        {
            auto bar_tensors = getBARBackedOutputs(barrier_stage_name_);

            FP32Tensor *cuda_tensor = bar_tensors.size() > static_cast<size_t>(cuda_idx) ? bar_tensors[cuda_idx] : nullptr;
            FP32Tensor *rocm_tensor = bar_tensors.size() > static_cast<size_t>(rocm_idx) ? bar_tensors[rocm_idx] : nullptr;

            if (cuda_tensor && rocm_tensor)
            {
                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Using BAR-backed tensors for stage '"
                          << barrier_stage_name_ << "' (CUDA tensor=" << cuda_tensor
                          << ", ROCm tensor=" << rocm_tensor << ")");

                // Validate tensors
                size_t tensor_numel = cuda_tensor->numel();
                if (rocm_tensor->numel() != tensor_numel)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Tensor size mismatch: CUDA="
                              << tensor_numel << " ROCm=" << rocm_tensor->numel());
                    return false;
                }

                // Use count_param if provided, otherwise use tensor->numel()
                // CRITICAL: For decode with dynamic seq_len, count_param MUST be set correctly
                size_t count = (count_param > 0) ? count_param : tensor_numel;

                if (count > tensor_numel)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: count (" << count
                                                                                 << ") exceeds tensor capacity (" << tensor_numel << ")");
                    return false;
                }

                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: count_param=" << count_param
                                                                                  << " tensor_numel=" << tensor_numel << " -> effective_count=" << count);

                // Ensure CUDA tensor is on the CUDA device
                DeviceId cuda_device = devices_[cuda_idx].toLocalDeviceId();
                if (!cuda_tensor->ensureOnDevice(cuda_device))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to ensure CUDA tensor on device");
                    return false;
                }

                // Ensure ROCm tensor is on the ROCm device (should already be BAR-backed)
                DeviceId rocm_device = devices_[rocm_idx].toLocalDeviceId();
                if (!rocm_tensor->ensureOnDevice(rocm_device))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to ensure ROCm tensor on device");
                    return false;
                }

                // Get pointers
                float *cuda_output = static_cast<float *>(cuda_tensor->gpu_data_ptr());

                // ===========================================================================
                // CRITICAL: ROCm kernels wrote to the HIP staging buffer, NOT directly to BAR!
                // We must copy from HIP staging → BAR so CUDA can read the ROCm partial result.
                //
                // Finding from Test__BARBackedHipAllocation exploration:
                //   - HIP kernels CANNOT dereference BAR mmap addresses (memory fault)
                //   - hipMemcpy(D2D) with BAR as destination WORKS at ~4+ GB/s (DMA engine)
                //   - This is the ONLY way for ROCm data to reach BAR
                // ===========================================================================
                const float *hip_staging_ptr = static_cast<const float *>(rocm_tensor->rocm_data_ptr());
                if (!hip_staging_ptr)
                {
                    hip_staging_ptr = static_cast<const float *>(rocm_tensor->gpu_data_ptr());
                    LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: rocm_data_ptr() is null; "
                              << "falling back to gpu_data_ptr()=" << hip_staging_ptr);
                }
                float *bar_ptr = static_cast<float *>(rocm_tensor->bar_address());

                if (!cuda_output || !hip_staging_ptr)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Invalid pointers "
                              << "(cuda=" << cuda_output << ", hip_staging=" << hip_staging_ptr << ")");
                    if (debugEnv().validation.validate_gpu_ptrs)
                    {
                        LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: FAIL-FAST due to invalid GPU pointers "
                                  << "with LLAMINAR_VALIDATE_GPU_PTRS=1");
#ifdef HAVE_ROCM
                        ROCmBackend::dumpRecentPointerEvents(128);
#endif
                        throw std::runtime_error("LocalTPContext PCIeBAR allreduce invalid pointers (hip_staging null)");
                    }
                    return false;
                }

                if (!bar_ptr)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ROCm tensor has no BAR address "
                              << "(isBARBacked=" << rocm_tensor->isBARBacked() << ")");
                    return false;
                }

                size_t bytes = count * sizeof(float);

                // Step 1: Copy ROCm partial result from HIP staging buffer → BAR via hipMemcpy D2D
                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Copying ROCm staging → BAR: "
                          << bytes << " bytes, src=" << hip_staging_ptr << " dst=" << bar_ptr);

                // We need to use hipMemcpy to copy from HIP device memory to BAR
                // Note: We use the ROCm backend's deviceToDevice which wraps hipMemcpy
                ROCmBackend *rocm_backend = dynamic_cast<ROCmBackend *>(getBackendForDevice(rocm_device));
                if (!rocm_backend)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ROCm backend not available");
                    return false;
                }

                rocm_backend->setDevice(rocm_device.toKernelDeviceIndex());

                // hipMemcpy(D2D) from HIP staging to BAR - this uses AMD DMA engine
                if (!rocm_backend->deviceToDevice(bar_ptr, hip_staging_ptr, bytes, rocm_device.toKernelDeviceIndex()))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to copy HIP staging → BAR");
                    return false;
                }

                // Synchronize HIP to ensure copy is complete before CUDA reads
                rocm_backend->synchronize(rocm_device.toKernelDeviceIndex());

                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Reducing " << count << " elements "
                                                                               << "(cuda_ptr=" << cuda_output << ", bar_ptr=" << bar_ptr
                                                                               << ", ROCm BAR-backed=" << rocm_tensor->isBARBacked() << ")");

                // Get PCIeBAR backend for reduction kernel
                auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
                if (!pcie_backend)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Backend is not PCIeBARBackend");
                    return false;
                }

                // Step 2: Perform reduction: cuda_output += bar_ptr (CUDA reads from BAR)
                CollectiveDataType dtype = tensorDTypeToCollective(cuda_tensor);
                if (!pcie_backend->reduceOnCUDA(cuda_output, cuda_output, bar_ptr,
                                                count, dtype, CollectiveOp::ALLREDUCE_SUM))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA reduction kernel failed");
                    return false;
                }

                // Synchronize CUDA
                if (!pcie_backend->synchronize())
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA synchronization failed");
                    return false;
                }

                // Step 3: Write result back to ROCm via BAR
                cudaError_t err = cudaMemcpy(bar_ptr, cuda_output,
                                             bytes, cudaMemcpyDeviceToDevice);
                if (err != cudaSuccess)
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to write result to BAR: "
                              << cudaGetErrorString(err));
                    return false;
                }

                // Step 4: Copy result from BAR back to HIP staging buffer
                // so ROCm tensor has the final reduced value
                if (!rocm_backend->deviceToDevice(const_cast<float *>(hip_staging_ptr), bar_ptr,
                                                  bytes, rocm_device.toKernelDeviceIndex()))
                {
                    LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to copy result BAR → HIP staging");
                    return false;
                }
                rocm_backend->synchronize(rocm_device.toKernelDeviceIndex());

                // Mark both tensors as having valid device data (with events for fine-grained sync)
                cuda_tensor->mark_device_dirty_with_event();
                rocm_tensor->mark_device_dirty_with_event();

                LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Zero-copy allreduce completed successfully "
                          << "for stage '" << barrier_stage_name_ << "'");
                return true;
            }
            else
            {
                LOG_WARN("LocalTPContext::executePCIeBarAllreduce: BAR-backed tensors incomplete for stage '"
                         << barrier_stage_name_ << "' (cuda=" << cuda_tensor << ", rocm=" << rocm_tensor << ")");
                // Fall through to barrier_tensors_ path
            }
        }

        // ===========================================================================
        // FALLBACK PATH: Use barrier_tensors_[] (legacy - may have issues)
        // ===========================================================================
        // This path is used when:
        // - No stage name provided
        // - BAR-backed tensors not registered for this stage
        //
        // WARNING: This path has known issues when both threads pass the same tensor
        // (both CUDA and ROCm threads have reference to same params_.tensor)

        LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Using barrier_tensors_ fallback path "
                  << "(stage_name=" << (barrier_stage_name_.empty() ? "(none)" : barrier_stage_name_) << ")");

        // Validate we have tensors from barrier
        if (barrier_tensors_.empty())
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: No tensors collected from barrier");
            return false;
        }

        // Identify CUDA and ROCm tensors from the barrier collection
        // NOTE: cuda_idx and rocm_idx were already determined above
        FP32Tensor *cuda_tensor = nullptr;
        FP32Tensor *rocm_tensor = nullptr;

        // The barrier_tensors_ stores by ARRIVAL order, not device order.
        // Try to identify by checking tensor properties.
        for (size_t i = 0; i < barrier_tensors_.size() && i < devices_.size(); ++i)
        {
            TensorBase *t = barrier_tensors_[i];
            if (!t)
            {
                LOG_WARN("LocalTPContext::executePCIeBarAllreduce: barrier_tensors_[" << i << "] is null");
                continue;
            }

            auto *fp32_t = dynamic_cast<FP32Tensor *>(t);
            if (!fp32_t)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Tensor is not FP32Tensor");
                return false;
            }

            // Check if this tensor is BAR-backed (indicates ROCm origin for zero-copy)
            if (fp32_t->isBARBacked() && fp32_t->rocm_data_ptr() != nullptr)
            {
                rocm_tensor = fp32_t;
            }
            else if (fp32_t->gpu_data_ptr() != nullptr)
            {
                cuda_tensor = fp32_t;
            }
            else
            {
                // Fallback: assign by index matching device order
                DeviceId local_id = devices_[i].toLocalDeviceId();
                if (local_id.is_cuda())
                    cuda_tensor = fp32_t;
                else if (local_id.is_rocm())
                    rocm_tensor = fp32_t;
            }
        }

        // If we couldn't identify by properties, fall back to device index order
        if (!cuda_tensor && cuda_idx >= 0 && cuda_idx < static_cast<int>(barrier_tensors_.size()))
            cuda_tensor = dynamic_cast<FP32Tensor *>(barrier_tensors_[cuda_idx]);
        if (!rocm_tensor && rocm_idx >= 0 && rocm_idx < static_cast<int>(barrier_tensors_.size()))
            rocm_tensor = dynamic_cast<FP32Tensor *>(barrier_tensors_[rocm_idx]);

        // Check for shared buffer scenario (both devices passed the same tensor)
        if (cuda_tensor == rocm_tensor)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ARCHITECTURAL BUG - both devices "
                      "passed the same tensor pointer ("
                      << cuda_tensor << "). "
                                        "PCIeBAR allreduce requires each device to have its own buffer: "
                                        "CUDA device needs a CUDA-memory tensor, ROCm device needs a BAR-backed tensor. "
                                        "Fix: Ensure DeviceGraphBufferManager allocates per-device buffers for row-parallel "
                                        "outputs when using LOCAL TP with PCIeBAR backend.");
            return false;
        }

        // Final validation
        if (!cuda_tensor)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Could not identify CUDA tensor");
            return false;
        }
        if (!rocm_tensor)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Could not identify ROCm tensor");
            return false;
        }

        size_t tensor_numel = cuda_tensor->numel();
        if (rocm_tensor->numel() != tensor_numel)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Tensor size mismatch: CUDA="
                      << tensor_numel << " ROCm=" << rocm_tensor->numel());
            return false;
        }

        // Use count_param if provided, otherwise use tensor->numel()
        // CRITICAL: For decode with dynamic seq_len, count_param MUST be set correctly
        size_t count = (count_param > 0) ? count_param : tensor_numel;

        if (count > tensor_numel)
        {
            LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: count (" << count
                                                                         << ") exceeds tensor capacity (" << tensor_numel << ")");
            return false;
        }

        LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Reducing " << count << " elements"
                                                                       << " (CUDA tensor=" << cuda_tensor
                                                                       << ", ROCm tensor=" << rocm_tensor
                                                                       << ", ROCm BAR-backed=" << rocm_tensor->isBARBacked()
                                                                       << ", count_param=" << count_param
                                                                       << ", tensor_numel=" << tensor_numel << ")");

        // ===========================================================================
        // ZERO-COPY PATH: If ROCm tensor is BAR-backed
        // ===========================================================================
        if (rocm_tensor->isBARBacked() && rocm_tensor->gpu_data_ptr() != nullptr)
        {
            // Zero-copy: CUDA reads directly from ROCm's BAR region
            //
            // The ROCm tensor's gpu_data_ptr() returns the CUDA-accessible pointer
            // to the BAR region (set up by initBARBackedDirect with bar_cuda_device_ptr_)
            //
            // Algorithm:
            // 1. Ensure CUDA tensor is on GPU
            // 2. CUDA kernel: cuda_output += rocm_bar_data
            // 3. Copy result back to ROCm (via BAR write)

            // Ensure CUDA tensor is on the CUDA device
            DeviceId cuda_device = devices_[cuda_idx].toLocalDeviceId();
            if (!cuda_tensor->ensureOnDevice(cuda_device))
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to ensure CUDA tensor on device");
                return false;
            }

            float *cuda_output = static_cast<float *>(cuda_tensor->gpu_data_ptr());
            const float *rocm_bar_ptr = static_cast<const float *>(rocm_tensor->gpu_data_ptr());

            if (!cuda_output || !rocm_bar_ptr)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Invalid GPU pointers "
                          << "(cuda=" << cuda_output << ", rocm_bar=" << rocm_bar_ptr << ")");
                return false;
            }

            LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Zero-copy path - "
                      << "CUDA reading from BAR at " << rocm_bar_ptr);

            // Get PCIeBAR backend for reduction kernel
            auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
            if (!pcie_backend)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Backend is not PCIeBARBackend");
                return false;
            }

            // Use the backend's reduceOnCUDA to perform: cuda_output += rocm_bar_ptr
            // This uses the CUDA vector add kernel
            CollectiveDataType dtype = tensorDTypeToCollective(cuda_tensor);
            if (!pcie_backend->reduceOnCUDA(cuda_output, cuda_output, rocm_bar_ptr,
                                            count, dtype, CollectiveOp::ALLREDUCE_SUM))
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA reduction kernel failed");
                return false;
            }

            // Synchronize CUDA
            if (!pcie_backend->synchronize())
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: CUDA synchronization failed");
                return false;
            }

            // Write result back to ROCm via BAR (so ROCm has the reduced result too)
            // For BAR-backed tensors, we can write directly through the BAR pointer
            // Use a CUDA memcpy from cuda_output to rocm_bar_ptr
            size_t bytes = count * sizeof(float);

            // For BAR-backed memory, the rocm_bar_ptr IS the CUDA-accessible pointer
            // We can use cudaMemcpy to write to it (CUDA writes through PCIe to AMD VRAM)
            cudaError_t err = cudaMemcpy(const_cast<float *>(rocm_bar_ptr), cuda_output,
                                         bytes, cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess)
            {
                LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: Failed to write result to ROCm: "
                          << cudaGetErrorString(err));
                return false;
            }

            // Mark both tensors as having valid device data (with events for fine-grained sync)
            cuda_tensor->mark_device_dirty_with_event();
            rocm_tensor->mark_device_dirty_with_event();

            LOG_DEBUG("LocalTPContext::executePCIeBarAllreduce: Zero-copy allreduce completed successfully");
            return true;
        }

        // ===========================================================================
        // ERROR: ROCm tensor is not BAR-backed
        // ===========================================================================
        // PCIeBAR allreduce requires BAR-backed tensors for zero-copy operation.
        // The tensor must be allocated in the BAR region using TensorFactory with
        // BAR-backed allocation enabled.
        //
        // To fix this:
        // 1. Call TensorFactory::setDirectP2P(p2p_instance) before tensor allocation
        // 2. Ensure tensors are created with createFP32BARBacked() or similar
        // 3. Verify LocalTPContext tracks BAR-backed tensors correctly

        LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: ROCm tensor is NOT BAR-backed! "
                  << "PCIeBAR allreduce requires zero-copy BAR-backed tensors. "
                  << "(rocm_tensor=" << rocm_tensor
                  << ", isBARBacked=" << rocm_tensor->isBARBacked()
                  << ", gpu_data_ptr=" << rocm_tensor->gpu_data_ptr() << ")");

        return false;
#else
        // Shouldn't be called without CUDA+ROCm
        (void)count_param;
        LOG_ERROR("LocalTPContext::executePCIeBarAllreduce: PCIeBAR requires both CUDA and ROCm");
        return false;
#endif
    }

    bool LocalTPContext::allgather(const TensorBase *local_shard, TensorBase *global_tensor)
    {
        if (!local_shard || !global_tensor)
        {
            LOG_ERROR("LocalTPContext::allgather: null tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy
        if (degree() == 1)
        {
            const float *src = local_shard->data();
            float *dst = global_tensor->mutable_data();
            size_t count = std::min(local_shard->numel(), global_tensor->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::allgather: Backend not initialized, skipping");
            // Fall back to copy of local shard
            const float *src = local_shard->data();
            float *dst = global_tensor->mutable_data();
            size_t count = std::min(local_shard->numel(), global_tensor->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // For LOCAL TP with Multi-GPU:
        // Each device sends its shard, receives all shards concatenated
        if (backend_impl_->isMultiGpuSingleProcess())
        {
            // For multi-GPU allgather, we need send buffers (one per device)
            // and recv buffers (one per device, each gets the full gathered result)

            // Note: In LOCAL TP, the local_shard and global_tensor parameters
            // represent the buffers for the "local" device. For true multi-GPU,
            // we would need separate buffers per device. For now, we use the
            // single-buffer API.
            DeviceId local_device = devices_[0].toLocalDeviceId();

            // Ensure local shard is on device (const-cast needed for ensureOnDevice)
            auto *mutable_shard = const_cast<TensorBase *>(local_shard);
            if (!mutable_shard->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to ensure local_shard on device");
                return false;
            }

            // Ensure global tensor is allocated on device
            if (!global_tensor->allocateOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to allocate global_tensor on device");
                return false;
            }

            const void *send_buf = mutable_shard->gpu_data_ptr();
            void *recv_buf = global_tensor->gpu_data_ptr();

            if (!send_buf || !recv_buf)
            {
                LOG_ERROR("LocalTPContext::allgather: No device buffers available");
                return false;
            }

            size_t send_count = local_shard->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(local_shard);

            LOG_DEBUG("LocalTPContext::allgather: allgather with "
                      << degree() << " devices, " << send_count << " elements per device");

            bool result = backend_impl_->allgather(
                send_buf, recv_buf, send_count, dtype);

            if (result)
            {
                global_tensor->mark_device_dirty_with_event();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allgather: Backend allgather failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
        else
        {
            // Fallback to single-buffer allgather
            DeviceId local_device = devices_[0].toLocalDeviceId();

            auto *mutable_shard = const_cast<TensorBase *>(local_shard);
            if (!mutable_shard->ensureOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to ensure local_shard on device");
                return false;
            }

            if (!global_tensor->allocateOnDevice(local_device))
            {
                LOG_ERROR("LocalTPContext::allgather: Failed to allocate global_tensor on device");
                return false;
            }

            const void *send_buf = mutable_shard->gpu_data_ptr();
            void *recv_buf = global_tensor->gpu_data_ptr();

            if (!send_buf || !recv_buf)
            {
                LOG_ERROR("LocalTPContext::allgather: No device buffers available");
                return false;
            }

            size_t send_count = local_shard->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(local_shard);

            bool result = backend_impl_->allgather(
                send_buf, recv_buf, send_count, dtype);

            if (result)
            {
                global_tensor->mark_device_dirty_with_event();
            }
            else
            {
                LOG_ERROR("LocalTPContext::allgather: Backend allgather failed: "
                          << backend_impl_->lastError());
            }
            return result;
        }
    }

    bool LocalTPContext::gatherFromDevices(
        const std::vector<const TensorBase *> &shards,
        TensorBase *output)
    {
        if (shards.empty() || !output)
        {
            LOG_ERROR("LocalTPContext::gatherFromDevices: empty shards or null output");
            return false;
        }

        // Validate shard count matches device count
        if (static_cast<int>(shards.size()) != degree())
        {
            LOG_ERROR("LocalTPContext::gatherFromDevices: shard count (" << shards.size()
                                                                         << ") doesn't match device count (" << degree() << ")");
            return false;
        }

        // Verify all shards are non-null
        for (size_t i = 0; i < shards.size(); ++i)
        {
            if (!shards[i])
            {
                LOG_ERROR("LocalTPContext::gatherFromDevices: shard[" << i << "] is null");
                return false;
            }
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy the shard to output
        if (degree() == 1)
        {
            const float *src = shards[0]->data();
            float *dst = output->mutable_data();
            size_t count = std::min(shards[0]->numel(), output->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Multi-device: concatenate all shards into output
        // For now, use CPU-side gather (works with any backend)
        // TODO: For GPU backends with allgatherMulti support, use device-side gather

        float *dst = output->mutable_data();
        size_t offset = 0;
        size_t output_capacity = output->numel();

        for (size_t i = 0; i < shards.size(); ++i)
        {
            const float *src = shards[i]->data();
            size_t shard_size = shards[i]->numel();

            // Check bounds
            if (offset + shard_size > output_capacity)
            {
                LOG_ERROR("LocalTPContext::gatherFromDevices: output buffer too small. "
                          << "Need " << (offset + shard_size) << ", have " << output_capacity);
                return false;
            }

            std::memcpy(dst + offset, src, shard_size * sizeof(float));
            offset += shard_size;
        }

        LOG_DEBUG("LocalTPContext::gatherFromDevices: gathered " << shards.size()
                                                                 << " shards, total " << offset << " elements");

        return true;
    }

    bool LocalTPContext::reduceScatter(const TensorBase *input, TensorBase *output_shard)
    {
        if (!input || !output_shard)
        {
            LOG_ERROR("LocalTPContext::reduceScatter: null tensor");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - just copy the appropriate shard
        if (degree() == 1)
        {
            const float *src = input->data();
            float *dst = output_shard->mutable_data();
            size_t count = std::min(input->numel(), output_shard->numel());
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_WARN("LocalTPContext::reduceScatter: Backend not initialized, skipping");
            // Fall back to copy of first shard
            const float *src = input->data();
            float *dst = output_shard->mutable_data();
            size_t count = output_shard->numel();
            std::memcpy(dst, src, count * sizeof(float));
            return true;
        }

        // ReduceScatter: reduce across devices, each device gets a slice
        DeviceId local_device = devices_[0].toLocalDeviceId();

        // Ensure input is on device
        auto *mutable_input = const_cast<TensorBase *>(input);
        if (!mutable_input->ensureOnDevice(local_device))
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Failed to ensure input on device");
            return false;
        }

        // Ensure output is allocated on device
        if (!output_shard->allocateOnDevice(local_device))
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Failed to allocate output_shard on device");
            return false;
        }

        const void *send_buf = mutable_input->gpu_data_ptr();
        void *recv_buf = output_shard->gpu_data_ptr();

        if (!send_buf || !recv_buf)
        {
            LOG_ERROR("LocalTPContext::reduceScatter: No device buffers available");
            return false;
        }

        size_t recv_count = output_shard->numel();
        CollectiveDataType dtype = tensorDTypeToCollective(input);

        LOG_DEBUG("LocalTPContext::reduceScatter: reduceScatter with "
                  << degree() << " devices, " << recv_count << " elements per device");

        bool result = backend_impl_->reduceScatter(
            send_buf, recv_buf, recv_count, dtype, CollectiveOp::ALLREDUCE_SUM);

        if (result)
        {
            output_shard->mark_device_dirty_with_event();
        }
        else
        {
            LOG_ERROR("LocalTPContext::reduceScatter: Backend reduceScatter failed: "
                      << backend_impl_->lastError());
        }
        return result;
    }

    bool LocalTPContext::broadcast(TensorBase *tensor, int source_device_index)
    {
        if (!tensor)
        {
            LOG_ERROR("LocalTPContext::broadcast: null tensor");
            return false;
        }

        if (source_device_index < 0 || source_device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::broadcast: invalid source_device_index "
                      << source_device_index << " (degree=" << degree() << ")");
            return false;
        }

        // Single device - no-op (already broadcast to the only device)
        if (degree() == 1)
        {
            LOG_DEBUG("LocalTPContext::broadcast: single device, no-op");
            return true;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // Ensure backend is initialized
        if (!backend_initialized_)
        {
            LOG_WARN("LocalTPContext::broadcast: Backend not initialized, skipping");
            return true; // Non-fatal: single device fallback
        }

        const GlobalDeviceAddress &source_device = devices_[source_device_index];
        DeviceId src_device_id = source_device.toLocalDeviceId();

        LOG_DEBUG("LocalTPContext::broadcast: Broadcasting from device "
                  << source_device_index << " (" << source_device.toString()
                  << ") to " << degree() << " devices");

        // Ensure tensor data is on the source device
        if (!tensor->ensureOnDevice(src_device_id))
        {
            LOG_ERROR("LocalTPContext::broadcast: Failed to ensure tensor on source device "
                      << src_device_id.toString());
            return false;
        }
        tensor->mark_device_dirty();

        // For homogeneous backends (NCCL/RCCL), use native broadcast if available
        // For now, we implement broadcast as point-to-point transfers from source to all others
        // TODO: Use backend_impl_->broadcast() when available in backend interface

        bool all_ok = true;
        for (int i = 0; i < degree(); ++i)
        {
            if (i == source_device_index)
            {
                continue; // Skip source device
            }

            DeviceId dst_device_id = devices_[i].toLocalDeviceId();

            LOG_DEBUG("LocalTPContext::broadcast: " << src_device_id.toString()
                                                    << " → " << dst_device_id.toString());

            // Use tensor's transferTo which uses GlobalBackendRouter
            // For same-vendor this will use NCCL/RCCL P2P or CUDA/HIP memcpy
            // For cross-vendor this will use PCIe BAR staging
            if (!tensor->transferTo(dst_device_id))
            {
                LOG_ERROR("LocalTPContext::broadcast: Transfer failed from "
                          << src_device_id.toString() << " to " << dst_device_id.toString());
                all_ok = false;
                // Continue trying other devices
            }
        }

        if (all_ok)
        {
            LOG_DEBUG("LocalTPContext::broadcast: Complete, tensor on all "
                      << degree() << " devices");
        }

        return all_ok;
    }

    // =========================================================================
    // Synchronization
    // =========================================================================

    void LocalTPContext::synchronize()
    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Single device - no-op
        if (degree() == 1)
        {
            return;
        }

        // Synchronize via backend
        if (backend_initialized_ && backend_impl_)
        {
            LOG_DEBUG("LocalTPContext::synchronize: Synchronizing backend "
                      << collectiveBackendTypeToString(backend_));
            if (!backend_impl_->synchronize())
            {
                LOG_WARN("LocalTPContext::synchronize: Backend synchronize failed: "
                         << backend_impl_->lastError());
            }
        }
    }

    // =========================================================================
    // Stream Configuration
    // =========================================================================

    void LocalTPContext::setComputeStreams(const std::vector<void *> &compute_streams)
    {
        if (backend_initialized_ && backend_impl_)
        {
            backend_impl_->setComputeStreams(compute_streams);
        }
    }

    // =========================================================================
    // Device Management
    // =========================================================================

    int LocalTPContext::indexForDevice(const GlobalDeviceAddress &device) const
    {
        auto it = device_to_index_.find(device);
        if (it != device_to_index_.end())
        {
            return it->second;
        }
        return -1;
    }

    const GlobalDeviceAddress &LocalTPContext::deviceAt(int index) const
    {
        if (index < 0 || index >= static_cast<int>(devices_.size()))
        {
            throw std::out_of_range(
                "LocalTPContext::deviceAt: index " + std::to_string(index) +
                " out of range [0, " + std::to_string(devices_.size()) + ")");
        }
        return devices_[index];
    }

    float LocalTPContext::weightForDevice(const GlobalDeviceAddress &device) const
    {
        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::weightForDevice: device not found");
            return 0.0f;
        }
        return weights_[idx];
    }

    // =========================================================================
    // Weight Sharding Utilities
    // =========================================================================

    int LocalTPContext::headsForDevice(const GlobalDeviceAddress &device, int total_heads) const
    {
        if (total_heads <= 0)
        {
            return 0;
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::headsForDevice: device not found");
            return 0;
        }

        // Use cumulative counts to ensure exact distribution
        auto cumulative = computeCumulativeCounts(total_heads, weights_);
        return cumulative[idx + 1] - cumulative[idx];
    }

    std::pair<int, int> LocalTPContext::rowRangeForDevice(
        const GlobalDeviceAddress &device, int total_rows) const
    {
        if (total_rows <= 0)
        {
            return {0, 0};
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::rowRangeForDevice: device not found");
            return {0, 0};
        }

        auto cumulative = computeCumulativeCounts(total_rows, weights_);
        return {cumulative[idx], cumulative[idx + 1]};
    }

    std::pair<int, int> LocalTPContext::colRangeForDevice(
        const GlobalDeviceAddress &device, int total_cols) const
    {
        if (total_cols <= 0)
        {
            return {0, 0};
        }

        int idx = indexForDevice(device);
        if (idx < 0)
        {
            LOG_WARN("LocalTPContext::colRangeForDevice: device not found");
            return {0, 0};
        }

        auto cumulative = computeCumulativeCounts(total_cols, weights_);
        return {cumulative[idx], cumulative[idx + 1]};
    }

    // =========================================================================
    // Private Helpers
    // =========================================================================

    std::vector<float> LocalTPContext::normalizeWeights(const std::vector<float> &weights)
    {
        if (weights.empty())
        {
            return {};
        }

        // Check for non-positive weights
        for (float w : weights)
        {
            if (w <= 0.0f)
            {
                throw std::invalid_argument("LocalTPContext: weights must be positive");
            }
        }

        float sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
        if (sum <= 0.0f)
        {
            throw std::invalid_argument("LocalTPContext: weight sum must be positive");
        }

        std::vector<float> normalized(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
        {
            normalized[i] = weights[i] / sum;
        }

        return normalized;
    }

    std::vector<int> LocalTPContext::computeCumulativeCounts(
        int total, const std::vector<float> &norm_weights)
    {
        std::vector<int> cumulative(norm_weights.size() + 1);
        cumulative[0] = 0;

        // Distribute proportionally with rounding to ensure exact total
        int remaining = total;
        float remaining_weight = 1.0f;

        for (size_t i = 0; i < norm_weights.size(); ++i)
        {
            if (i == norm_weights.size() - 1)
            {
                // Last device gets the remainder to ensure exact total
                cumulative[i + 1] = total;
            }
            else
            {
                // Proportional distribution with proper rounding
                float proportion = norm_weights[i] / remaining_weight;
                int count = static_cast<int>(std::round(proportion * remaining));

                // Ensure at least 1 if there's work remaining
                if (count == 0 && remaining > 0)
                {
                    count = 1;
                }
                // Don't exceed remaining
                count = std::min(count, remaining);

                cumulative[i + 1] = cumulative[i] + count;
                remaining -= count;
                remaining_weight -= norm_weights[i];
            }
        }

        return cumulative;
    }

    CollectiveBackendType LocalTPContext::autoDetectBackend(
        const std::vector<GlobalDeviceAddress> &devices)
    {
        bool has_cuda = false;
        bool has_rocm = false;
        bool has_cpu = false;

        for (const auto &dev : devices)
        {
            if (dev.isCUDA())
                has_cuda = true;
            else if (dev.isROCm())
                has_rocm = true;
            else if (dev.isCPU())
                has_cpu = true;
        }

        // If CPU is involved, use host-staged backend
        if (has_cpu)
        {
            return CollectiveBackendType::HOST;
        }

        // Mixed GPU types
        if (has_cuda && has_rocm)
        {
            // Count devices of each type to determine backend
            int num_cuda = 0;
            int num_rocm = 0;
            for (const auto &dev : devices)
            {
                if (dev.isCUDA())
                    num_cuda++;
                else if (dev.isROCm())
                    num_rocm++;
            }

            // Use simple PCIeBAR for exactly 1+1 case if on the same NUMA node
            if (num_cuda == 1 && num_rocm == 1)
            {
                // Check NUMA affinity via DeviceManager — PCIeBAR requires same-NUMA
                bool same_numa = true;
                int first_numa = -1;
                const auto &dm = DeviceManager::instance();
                for (const auto &dev : devices)
                {
                    if (!dev.isGPU())
                        continue;
                    // Look up real NUMA from hardware inventory
                    int dev_numa = -1;
                    for (const auto &cd : dm.devices())
                    {
                        bool type_match = (dev.isCUDA() && cd.type == ComputeBackendType::GPU_CUDA) ||
                                          (dev.isROCm() && cd.type == ComputeBackendType::GPU_ROCM);
                        if (type_match && cd.device_id == dev.device_ordinal)
                        {
                            dev_numa = cd.numa_node;
                            break;
                        }
                    }
                    if (dev_numa < 0)
                        continue; // Unknown — don't block
                    if (first_numa < 0)
                        first_numa = dev_numa;
                    else if (dev_numa != first_numa)
                        same_numa = false;
                }
                if (same_numa)
                {
                    return CollectiveBackendType::PCIE_BAR;
                }
                LOG_WARN("Cross-NUMA heterogeneous GPU collective: HOST backend selected "
                         "instead of PCIeBAR — this will be slow! "
                         "For best performance, use GPUs on the same NUMA node "
                         "so PCIeBAR peer-to-peer transfers can be used.");
                return CollectiveBackendType::HOST;
            }

            // Use hierarchical backend for N+M case (>2 devices)
            return CollectiveBackendType::HETEROGENEOUS;
        }

        // All CUDA - use NCCL
        if (has_cuda && !has_rocm)
        {
            return CollectiveBackendType::NCCL;
        }

        // All ROCm - use RCCL
        if (has_rocm && !has_cuda)
        {
            return CollectiveBackendType::RCCL;
        }

        // Default to HOST if nothing detected (shouldn't happen)
        return CollectiveBackendType::HOST;
    }

    void LocalTPContext::buildDeviceIndex()
    {
        device_to_index_.clear();
        for (size_t i = 0; i < devices_.size(); ++i)
        {
            device_to_index_[devices_[i]] = static_cast<int>(i);
        }
    }

    // =========================================================================
    // Backend Initialization and Helper Methods
    // =========================================================================

    bool LocalTPContext::initializeBackend()
    {
        // Build device group from devices_
        DeviceGroupBuilder builder;
        builder.setName("LocalTP_" + std::to_string(degree()) + "_devices");
        builder.setScope(CollectiveScope::LOCAL);
        builder.setLocalRank(0); // In LOCAL TP, we manage all devices from rank 0

        for (const auto &device : devices_)
        {
            builder.addDevice(device.toLocalDeviceId());
        }

        device_group_ = builder.build();

        // Create appropriate backend based on type
        switch (backend_)
        {
        case CollectiveBackendType::NCCL:
#ifdef HAVE_CUDA
            LOG_DEBUG("LocalTPContext: Creating NCCL backend");
            backend_impl_ = std::make_unique<NCCLBackend>();
#else
            LOG_WARN("LocalTPContext: NCCL requested but CUDA not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::RCCL:
#ifdef HAVE_ROCM
            LOG_DEBUG("LocalTPContext: Creating RCCL backend");
            backend_impl_ = std::make_unique<RCCLBackend>();
#else
            LOG_WARN("LocalTPContext: RCCL requested but ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::PCIE_BAR:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            LOG_DEBUG("LocalTPContext: Creating PCIe BAR backend");
            backend_impl_ = std::make_unique<PCIeBARBackend>();
#else
            LOG_WARN("LocalTPContext: PCIE_BAR requested but both CUDA and ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::HETEROGENEOUS:
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            LOG_DEBUG("LocalTPContext: Creating Heterogeneous backend");
            backend_impl_ = std::make_unique<HeterogeneousBackend>();
#else
            LOG_WARN("LocalTPContext: HETEROGENEOUS requested but both CUDA and ROCm not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
#endif
            break;

        case CollectiveBackendType::HOST:
        case CollectiveBackendType::AUTO:
        default:
            LOG_DEBUG("LocalTPContext: Creating HOST backend");
            backend_impl_ = std::make_unique<HostBackend>();
            break;
        }

        // Check if backend is available
        if (!backend_impl_->isAvailable())
        {
            LOG_WARN("LocalTPContext: Backend " << collectiveBackendTypeToString(backend_)
                                                << " not available, falling back to HOST");
            backend_impl_ = std::make_unique<HostBackend>();
            backend_ = CollectiveBackendType::HOST; // Update to reflect actual backend
        }

        // Initialize the backend with device group
        if (!backend_impl_->initialize(device_group_))
        {
            LOG_ERROR("LocalTPContext: Failed to initialize backend: "
                      << backend_impl_->lastError());
            backend_impl_.reset();
            backend_initialized_ = false;
            return false;
        }

        backend_initialized_ = true;
        LOG_INFO("LocalTPContext: Backend " << backend_impl_->name()
                                            << " initialized for " << degree() << " devices");

        if (backend_ == CollectiveBackendType::NCCL)
        {
            LOG_INFO("[LocalTPContext][NCCLReady] "
                     << "status=ready"
                     << " backend_impl=" << backend_impl_->name()
                     << " degree=" << degree()
                     << " multi_gpu_single_process=" << (backend_impl_->isMultiGpuSingleProcess() ? 1 : 0));
        }
        return true;
    }

    bool LocalTPContext::ensurePCIeBarBuffersRegistered(TensorBase *tensor)
    {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        if (!tensor)
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: null tensor");
            return false;
        }

        // Only needed for PCIeBAR backend
        if (backend_ != CollectiveBackendType::PCIE_BAR)
        {
            return true;
        }

        auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
        if (!pcie_backend)
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Backend is not PCIeBARBackend");
            return false;
        }

        size_t buffer_bytes = tensor->numel() * sizeof(float);

        // If buffers are already registered for this size, we're done
        if (pciebar_buffers_registered_ && buffer_bytes <= pciebar_buffer_size_)
        {
            LOG_DEBUG("ensurePCIeBarBuffersRegistered: Reusing registered buffers for collective "
                      << pciebar_collective_id_);
            return true;
        }

        // If size changed, we need to re-register (unregister old first)
        if (pciebar_buffers_registered_ && buffer_bytes > pciebar_buffer_size_)
        {
            LOG_WARN("ensurePCIeBarBuffersRegistered: Buffer size increased from "
                     << pciebar_buffer_size_ << " to " << buffer_bytes
                     << ", need to re-register. This may leak BAR memory.");
            // Note: We can't actually reclaim BAR allocations (bump allocator),
            // but we can unregister the old collective_id
            for (const auto &device : devices_)
            {
                pcie_backend->unregisterBuffer(pciebar_collective_id_, device.toLocalDeviceId());
            }
            pciebar_buffers_registered_ = false;
        }

        // Generate a unique collective_id for this LocalTPContext
        pciebar_collective_id_ = "LocalTP_allreduce_" + std::to_string(reinterpret_cast<uintptr_t>(this));
        pciebar_buffer_size_ = buffer_bytes;

        LOG_DEBUG("ensurePCIeBarBuffersRegistered: Registering buffers for " << pciebar_collective_id_
                                                                             << " (size=" << buffer_bytes << " bytes)");

        // For 2-device LOCAL TP (CUDA + ROCm):
        // 1. CUDA buffer: use tensor's existing GPU buffer (allocated by standard path)
        // 2. ROCm buffer: must be allocated in BAR region to get correct offset

        DeviceId cuda_device;
        DeviceId rocm_device;
        void *cuda_ptr = nullptr;

        // Identify CUDA and ROCm devices
        for (const auto &device : devices_)
        {
            DeviceId local_id = device.toLocalDeviceId();
            if (local_id.is_cuda())
            {
                cuda_device = local_id;
                // Get CUDA buffer from tensor
                if (!tensor->ensureOnDevice(cuda_device))
                {
                    LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to ensure tensor on CUDA device");
                    return false;
                }
                cuda_ptr = tensor->gpu_data_ptr();
                if (!cuda_ptr)
                {
                    LOG_ERROR("ensurePCIeBarBuffersRegistered: No CUDA buffer available");
                    return false;
                }
            }
            else if (local_id.is_rocm())
            {
                rocm_device = local_id;
            }
        }

        if (!cuda_device.is_valid() || !rocm_device.is_valid())
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Need exactly one CUDA and one ROCm device");
            return false;
        }

        // Allocate ROCm buffer in BAR region
        auto bar_alloc = pcie_backend->allocateInBarRegion(buffer_bytes);
        if (!bar_alloc.has_value())
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to allocate " << buffer_bytes
                                                                            << " bytes in BAR region");
            return false;
        }

        void *rocm_bar_ptr = bar_alloc->first;
        size_t rocm_bar_offset = bar_alloc->second;

        LOG_DEBUG("ensurePCIeBarBuffersRegistered: Allocated BAR buffer at offset " << rocm_bar_offset);

        // Register CUDA buffer
        if (!pcie_backend->registerBuffer(pciebar_collective_id_, cuda_device, cuda_ptr, buffer_bytes))
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to register CUDA buffer");
            return false;
        }

        // Register ROCm buffer (with BAR offset)
        if (!pcie_backend->registerBuffer(pciebar_collective_id_, rocm_device, rocm_bar_ptr, buffer_bytes))
        {
            LOG_ERROR("ensurePCIeBarBuffersRegistered: Failed to register ROCm buffer");
            pcie_backend->unregisterBuffer(pciebar_collective_id_, cuda_device);
            return false;
        }

        pciebar_buffers_registered_ = true;
        LOG_INFO("ensurePCIeBarBuffersRegistered: Successfully registered buffers for PCIeBAR allreduce"
                 << " (collective_id=" << pciebar_collective_id_
                 << ", size=" << buffer_bytes << " bytes"
                 << ", BAR offset=" << rocm_bar_offset << ")");

        return true;
#else
        (void)tensor;
        return true; // No-op when PCIeBAR not available
#endif
    }

    std::vector<void *> LocalTPContext::getDeviceBuffers(TensorBase *tensor)
    {
        std::vector<void *> buffers;
        buffers.reserve(devices_.size());

        // For LOCAL TP, we need to get the GPU buffer for each device
        // Current implementation assumes tensor has data on all devices
        // or we need to replicate it

        for (size_t i = 0; i < devices_.size(); ++i)
        {
            DeviceId device_id = devices_[i].toLocalDeviceId();

            // Ensure tensor data is on this device
            if (!tensor->ensureOnDevice(device_id))
            {
                LOG_ERROR("LocalTPContext::getDeviceBuffers: Failed to ensure tensor on device "
                          << device_id.toString());
                return {}; // Return empty vector to indicate failure
            }

            void *ptr = tensor->gpu_data_ptr();
            if (!ptr)
            {
                LOG_ERROR("LocalTPContext::getDeviceBuffers: No GPU buffer for device "
                          << device_id.toString());
                return {};
            }

            buffers.push_back(ptr);
        }

        return buffers;
    }

    CollectiveDataType LocalTPContext::tensorDTypeToCollective(const TensorBase *tensor) const
    {
        if (!tensor)
        {
            return CollectiveDataType::FLOAT32;
        }

        // Get tensor type and map to CollectiveDataType
        // Most common case is FP32 for activation tensors
        TensorType tt = tensor->native_type();
        switch (tt)
        {
        case TensorType::FP32:
            return CollectiveDataType::FLOAT32;
        case TensorType::FP16:
            return CollectiveDataType::FLOAT16;
        case TensorType::BF16:
            return CollectiveDataType::BFLOAT16;
        case TensorType::INT32:
            return CollectiveDataType::INT32;
        case TensorType::INT8:
        case TensorType::Q8_0:
        case TensorType::Q8_1:
            return CollectiveDataType::INT8;
        default:
            // For quantized types that don't have a direct mapping, use FLOAT32
            // (collectives are typically on dequantized activations)
            return CollectiveDataType::FLOAT32;
        }
    }

    // =========================================================================
    // BAR-Backed Tensor Registry
    // =========================================================================

    void LocalTPContext::registerBARBackedOutput(
        const std::string &stage_name,
        const GlobalDeviceAddress &device,
        TensorBase *tensor)
    {
        // Validate device is in our context
        int idx = indexForDevice(device);
        if (idx < 0)
        {
            throw std::invalid_argument(
                "Device " + device.toString() + " not in LocalTPContext");
        }

        // Validate tensor is non-null
        if (!tensor)
        {
            throw std::invalid_argument(
                "Tensor must be non-null");
        }

        // Cast to FP32Tensor (allreduce outputs are FP32)
        FP32Tensor *fp32_tensor = dynamic_cast<FP32Tensor *>(tensor);
        if (!fp32_tensor)
        {
            LOG_WARN("[LocalTPContext] registerBARBackedOutput: tensor is not FP32Tensor, skipping");
            return;
        }

        // For PCIeBAR backend:
        // - CUDA device: tensor does NOT need to be BAR-backed (regular CUDA memory)
        // - ROCm device: tensor MUST be BAR-backed for zero-copy P2P access
        //
        // We register ALL tensors here. The actual BAR-backed requirement only
        // applies to the ROCm device, which is validated during allreduce execution.
        //
        // For non-PCIeBAR backends: require BAR-backed for testing consistency
        if (backend_ != CollectiveBackendType::PCIE_BAR)
        {
            // Non-PCIeBAR backends: still require BAR-backed for consistency with existing tests
            // This maintains backward compatibility with unit tests using HOST backend
            if (!fp32_tensor->isBARBacked())
            {
                throw std::invalid_argument(
                    "Tensor must be BAR-backed");
            }
        }

        // Log whether tensor is BAR-backed for debugging
        LOG_TRACE("[LocalTPContext] registerBARBackedOutput: stage='" << stage_name
                                                                      << "' device=" << device.toString()
                                                                      << " is_bar_backed=" << fp32_tensor->isBARBacked());

        // Initialize vector for this stage if needed
        auto &tensors = bar_output_tensors_[stage_name];
        if (tensors.empty())
        {
            tensors.resize(degree(), nullptr);
        }

        // Register
        tensors[idx] = fp32_tensor;

        LOG_DEBUG("[LocalTPContext] Registered BAR-backed output for stage '"
                  << stage_name << "' device " << device.toString());
    }

    std::vector<FP32Tensor *> LocalTPContext::getBARBackedOutputs(
        const std::string &stage_name) const
    {

        auto it = bar_output_tensors_.find(stage_name);
        if (it == bar_output_tensors_.end())
        {
            return std::vector<FP32Tensor *>(degree(), nullptr);
        }
        return it->second;
    }

    bool LocalTPContext::hasBARBackedOutputs(const std::string &stage_name) const
    {
        auto it = bar_output_tensors_.find(stage_name);
        if (it == bar_output_tensors_.end())
        {
            return false;
        }
        // Check if at least one entry is non-null
        for (auto *t : it->second)
        {
            if (t != nullptr)
                return true;
        }
        return false;
    }

    void LocalTPContext::clearBARBackedOutputs()
    {
        bar_output_tensors_.clear();
        LOG_DEBUG("[LocalTPContext] Cleared all BAR-backed output registrations");
    }

    std::shared_ptr<DirectP2PEngine> LocalTPContext::getDirectP2PEngine() const
    {
        // Only available for PCIeBAR backend
        if (backend_ != CollectiveBackendType::PCIE_BAR)
        {
            return nullptr;
        }

        if (!backend_impl_)
        {
            return nullptr;
        }

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
        // Cast to PCIeBARBackend and get the engine
        auto *pcie_backend = dynamic_cast<PCIeBARBackend *>(backend_impl_.get());
        if (!pcie_backend)
        {
            return nullptr;
        }

        DirectP2PEngine *raw_ptr = pcie_backend->getDirectP2PEngine();
        if (!raw_ptr)
        {
            return nullptr;
        }

        // Return a shared_ptr that doesn't delete the engine (it's owned by PCIeBARBackend)
        // We use aliasing constructor with empty shared_ptr to create non-owning shared_ptr
        return std::shared_ptr<DirectP2PEngine>(std::shared_ptr<void>(), raw_ptr);
#else
        // PCIeBAR backend requires both CUDA and ROCm
        return nullptr;
#endif
    }

    bool LocalTPContext::reserveTempBufferBytes(size_t bytes)
    {
        if (!backend_impl_)
        {
            LOG_WARN("[LocalTPContext] Cannot reserve temp buffer: backend not initialized");
            return false;
        }

        LOG_DEBUG("[LocalTPContext] Reserving temp buffer: " << bytes << " bytes");
        return backend_impl_->reserveTempBufferBytes(bytes);
    }

    // =========================================================================
    // Factory Function
    // =========================================================================

    std::unique_ptr<ILocalTPContext> createLocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend)
    {
        return std::make_unique<LocalTPContext>(
            std::move(devices),
            std::move(weights),
            backend);
    }

} // namespace llaminar2
