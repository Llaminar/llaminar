/**
 * @file LocalTPContext.cpp
 * @brief Implementation of LOCAL tensor parallelism context
 * @author David Sanftenberg
 * @date January 2026
 */

#include "LocalTPContext.h"
#include "AllreducePrecisionPolicy.h"
#include "CollectiveTimeoutPolicy.h"
#include "backends/HostBackend.h"
#include "../tensors/TensorClasses.h"
#include "../backends/BackendManager.h" // For getCUDABackend, getROCmBackend
#include "../backends/ComputeBackend.h" // For DeviceManager (NUMA lookup)
#include "../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../planning/CollectiveMemoryEstimator.h"
#include "../transfer/TransferEngine.h"
#include "../utils/DebugEnv.h"
#include "../utils/Logger.h"
#include "../utils/PerfStatsCollector.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
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
#include "backends/HeterogeneousBackend.h"
#endif

// ============================================================================
// Extern declarations for FP32 ↔ FP16 cast kernels (mixed-precision allreduce)
// ============================================================================
#ifdef HAVE_CUDA
extern "C"
{
    cudaError_t cudaCastFP32ToFP16(const float *fp32_input, void *fp16_output,
                                   size_t count, int ordinal, cudaStream_t stream);
    cudaError_t cudaCastFP16ToFP32(const void *fp16_input, float *fp32_output,
                                   size_t count, int ordinal, cudaStream_t stream);
}
#endif

#ifdef HAVE_CUDA
namespace llaminar2
{
    namespace nccl_backend_detail
    {
        bool cudaSetDeviceOrdinal(int device_ordinal);
        bool cudaMemcpyAsyncSameDevice(void *dst, const void *src, size_t bytes, int device_ordinal, void *stream);
        bool cudaMemsetAsyncDevice(void *dst, int value, size_t bytes, int device_ordinal, void *stream);
    }
}
#endif

#ifdef HAVE_ROCM
namespace llaminar2
{
    namespace rccl_backend_detail
    {
        bool hipSetDeviceOrdinal(int device_ordinal);
        bool hipMemcpyAsyncSameDevice(void *dst, const void *src, size_t bytes, int device_ordinal, void *stream);
        bool hipMemsetAsyncDevice(void *dst, int value, size_t bytes, int device_ordinal, void *stream);
    }
}

extern "C"
{
    int rocmCastFP32ToFP16(const float *fp32_input, void *fp16_output,
                           size_t count, int ordinal, void *stream);
    int rocmCastFP16ToFP32(const void *fp16_input, float *fp32_output,
                           size_t count, int ordinal, void *stream);
}
#endif

namespace llaminar2
{
    namespace
    {
        constexpr const char *kDefaultAllreducePrecision = "fp32";
    }

    std::atomic<uint64_t> LocalTPContext::next_context_id_{1};

    namespace
    {
        const char *collectiveDataTypeName(CollectiveDataType dtype)
        {
            switch (dtype)
            {
            case CollectiveDataType::FLOAT32:
                return "fp32";
            case CollectiveDataType::FLOAT16:
                return "fp16";
            case CollectiveDataType::BFLOAT16:
                return "bf16";
            case CollectiveDataType::INT32:
                return "int32";
            case CollectiveDataType::INT8:
                return "int8";
            }
            return "unknown";
        }

        size_t collectiveDataTypeBytes(CollectiveDataType dtype)
        {
            switch (dtype)
            {
            case CollectiveDataType::FLOAT32:
            case CollectiveDataType::INT32:
                return sizeof(std::uint32_t);
            case CollectiveDataType::FLOAT16:
            case CollectiveDataType::BFLOAT16:
                return sizeof(std::uint16_t);
            case CollectiveDataType::INT8:
                return sizeof(std::uint8_t);
            }
            return 0;
        }

        void recordLocalTPRuntimeAllreduce(
            const DeviceGroup &device_group,
            CollectiveBackendType backend,
            const DeviceId &device,
            const std::string &stage_name,
            size_t degree,
            size_t elements,
            CollectiveDataType dtype,
            const std::string &path,
            const std::string &requested_precision)
        {
            if (!PerfStatsCollector::isDomainEnabled(
                    "tp_allreduce_runtime"))
                return;

            const size_t element_bytes = collectiveDataTypeBytes(dtype);
            PerfStatsCollector::Tags tags{
                {"stage", stage_name.empty() ? "unnamed" : stage_name},
                {"backend", collectiveBackendTypeToString(backend)},
                {"scope", "rank_local"},
                {"degree", std::to_string(degree)},
                {"dtype", collectiveDataTypeName(dtype)},
                {"element_bytes", std::to_string(element_bytes)},
                {"elements", std::to_string(elements)},
                {"path", path},
                {"requested_precision", requested_precision.empty() ? "default" : requested_precision},
                {"homogeneous", device_group.is_homogeneous ? "true" : "false"}};

            PerfStatsCollector::addCounter(
                "tp_allreduce_runtime",
                "calls",
                1.0,
                {},
                device.toString(),
                tags);
            PerfStatsCollector::addCounter(
                "tp_allreduce_runtime",
                "bytes",
                static_cast<double>(elements * element_bytes),
                {},
                device.toString(),
                std::move(tags));
        }

        /**
         * @brief Record the physical primitive used by a raw LocalTP allgather.
         *
         * Raw graph-state handoffs carry planner metadata, recurrent state, and
         * occasionally large expert payloads.  Their telemetry must therefore
         * distinguish a native NCCL/RCCL allgather from any emulation that
         * increases traffic or introduces a host rendezvous.
         */
        void recordLocalTPRuntimeRawAllgather(
            const DeviceGroup &device_group,
            CollectiveBackendType backend,
            const DeviceId &device,
            const std::string &stage_name,
            size_t degree,
            int device_index,
            size_t elements,
            CollectiveDataType dtype,
            bool graph_capture)
        {
            if (!PerfStatsCollector::isDomainEnabled(
                    "tp_raw_allgather_runtime"))
                return;

            const size_t element_bytes = collectiveDataTypeBytes(dtype);
            const size_t local_bytes = elements * element_bytes;
            const size_t result_elements = elements * degree;
            PerfStatsCollector::Tags tags{
                {"stage", stage_name.empty() ? "unnamed" : stage_name},
                {"backend", collectiveBackendTypeToString(backend)},
                {"scope", "rank_local"},
                {"degree", std::to_string(degree)},
                {"device_index", std::to_string(device_index)},
                {"dtype", collectiveDataTypeName(dtype)},
                {"element_bytes", std::to_string(element_bytes)},
                {"elements", std::to_string(elements)},
                {"result_elements", std::to_string(result_elements)},
                {"path", "native_single_device_on_stream"},
                {"backend_primitive",
                 backend == CollectiveBackendType::NCCL
                     ? "ncclAllGather"
                     : "rcclAllGather"},
                {"graph_capture", graph_capture ? "true" : "false"},
                {"host_rendezvous", "false"},
                {"homogeneous", device_group.is_homogeneous ? "true" : "false"}};

            PerfStatsCollector::addCounter(
                "tp_raw_allgather_runtime",
                "calls",
                1.0,
                {},
                device.toString(),
                tags);
            PerfStatsCollector::addCounter(
                "tp_raw_allgather_runtime",
                "local_bytes",
                static_cast<double>(local_bytes),
                {},
                device.toString(),
                tags);
            PerfStatsCollector::addCounter(
                "tp_raw_allgather_runtime",
                "result_bytes",
                static_cast<double>(result_elements * element_bytes),
                {},
                device.toString(),
                std::move(tags));
        }

        /**
         * @brief Record one native rooted LocalTP collective without inventing
         *        allreduce-equivalent traffic.
         */
        void recordLocalTPRuntimeRawRootedCollective(
            const DeviceGroup &device_group,
            CollectiveBackendType backend,
            const DeviceId &device,
            const std::string &stage_name,
            size_t degree,
            int device_index,
            int root_device_index,
            size_t elements,
            CollectiveDataType dtype,
            const char *operation,
            bool graph_capture)
        {
            if (!PerfStatsCollector::isDomainEnabled(
                    "tp_rooted_collective_runtime"))
                return;

            const size_t element_bytes = collectiveDataTypeBytes(dtype);
            const std::string primitive =
                std::string(backend == CollectiveBackendType::NCCL
                                ? "nccl"
                                : "rccl") +
                (std::string(operation) == "reduce" ? "Reduce" : "Broadcast");
            PerfStatsCollector::Tags tags{
                {"stage", stage_name.empty() ? "unnamed" : stage_name},
                {"operation", operation},
                {"backend", collectiveBackendTypeToString(backend)},
                {"scope", "rank_local"},
                {"degree", std::to_string(degree)},
                {"device_index", std::to_string(device_index)},
                {"root_device_index", std::to_string(root_device_index)},
                {"dtype", collectiveDataTypeName(dtype)},
                {"element_bytes", std::to_string(element_bytes)},
                {"elements", std::to_string(elements)},
                {"path", "native_single_device_on_stream"},
                {"backend_primitive", primitive},
                {"graph_capture", graph_capture ? "true" : "false"},
                {"host_rendezvous", "false"},
                {"homogeneous", device_group.is_homogeneous ? "true" : "false"}};

            PerfStatsCollector::addCounter(
                "tp_rooted_collective_runtime",
                "calls",
                1.0,
                {},
                device.toString(),
                tags);
            PerfStatsCollector::addCounter(
                "tp_rooted_collective_runtime",
                "bytes",
                static_cast<double>(elements * element_bytes),
                {},
                device.toString(),
                std::move(tags));
        }

        size_t sidebandResultElements(
            LocalTPCollectiveSidebandKind kind,
            size_t element_count,
            size_t degree)
        {
            if (kind == LocalTPCollectiveSidebandKind::Allgather)
                return element_count * degree;
            return element_count;
        }

        CollectiveSidebandOp toBackendSidebandOp(LocalTPCollectiveSidebandKind kind)
        {
            switch (kind)
            {
            case LocalTPCollectiveSidebandKind::AllreduceSum:
                return CollectiveSidebandOp::AllreduceSum;
            case LocalTPCollectiveSidebandKind::Allgather:
                return CollectiveSidebandOp::Allgather;
            case LocalTPCollectiveSidebandKind::Broadcast:
                return CollectiveSidebandOp::Broadcast;
            }
            return CollectiveSidebandOp::AllreduceSum;
        }

        const char *backendSidebandPrimitiveName(LocalTPCollectiveSidebandKind kind)
        {
            switch (kind)
            {
            case LocalTPCollectiveSidebandKind::AllreduceSum:
                return "allreduceWithSidebandsMultiOnStreams/allreduce";
            case LocalTPCollectiveSidebandKind::Allgather:
                return "allreduceWithSidebandsMultiOnStreams/allgather";
            case LocalTPCollectiveSidebandKind::Broadcast:
                return "allreduceWithSidebandsMultiOnStreams/broadcast";
            }
            return "allreduceWithSidebandsMultiOnStreams/unknown";
        }

        void recordLocalTPRuntimeGroupedSidebands(
            const DeviceGroup &device_group,
            CollectiveBackendType backend,
            const DeviceId &device,
            const std::string &stage_name,
            size_t degree,
            int device_index,
            const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands)
        {
            if (!PerfStatsCollector::isDomainEnabled(
                    "tp_allreduce_runtime"))
                return;

            for (const auto &sideband : sidebands)
            {
                const size_t element_bytes = collectiveDataTypeBytes(sideband.dtype);
                const size_t local_bytes = sideband.element_count * element_bytes;
                const size_t result_bytes =
                    sidebandResultElements(sideband.kind, sideband.element_count, degree) *
                    element_bytes;
                PerfStatsCollector::Tags tags{
                    {"stage", stage_name.empty() ? "unnamed" : stage_name},
                    {"anchor_stage", stage_name.empty() ? "unnamed" : stage_name},
                    {"anchor_collective", "allreduce"},
                    {"backend", collectiveBackendTypeToString(backend)},
                    {"scope", "rank_local"},
                    {"degree", std::to_string(degree)},
                    {"sideband", sideband.name.empty() ? "unnamed" : sideband.name},
                    {"kind", toString(sideband.kind)},
                    {"dtype", collectiveDataTypeName(sideband.dtype)},
                    {"element_bytes", std::to_string(element_bytes)},
                    {"elements", std::to_string(sideband.element_count)},
                    {"result_elements", std::to_string(sidebandResultElements(
                                            sideband.kind, sideband.element_count, degree))},
                    {"root_device_index", std::to_string(sideband.root_device_index)},
                    {"device_index", std::to_string(device_index)},
                    {"backend_primitive", backendSidebandPrimitiveName(sideband.kind)},
                    {"path", "allreduce_with_sidebands_grouped_on_streams"},
                    {"launch_relation", "same_group_as_anchor"},
                    {"fused_with_anchor", "true"},
                    {"physical_fusion", "grouped_with_anchor_collective"},
                    {"homogeneous", device_group.is_homogeneous ? "true" : "false"}};

                PerfStatsCollector::addCounter(
                    "tp_allreduce_runtime",
                    "sideband_backend_collective_calls",
                    1.0,
                    {},
                    device.toString(),
                    tags);
                PerfStatsCollector::addCounter(
                    "tp_allreduce_runtime",
                    "sideband_grouped_with_anchor_collective_calls",
                    1.0,
                    {},
                    device.toString(),
                    tags);

                PerfStatsCollector::Tags local_byte_tags = tags;
                PerfStatsCollector::addCounter(
                    "tp_allreduce_runtime",
                    "sideband_local_bytes",
                    static_cast<double>(local_bytes),
                    {},
                    device.toString(),
                    std::move(local_byte_tags));

                PerfStatsCollector::Tags result_byte_tags = tags;
                PerfStatsCollector::addCounter(
                    "tp_allreduce_runtime",
                    "sideband_result_bytes",
                    static_cast<double>(result_bytes),
                    {},
                    device.toString(),
                    std::move(result_byte_tags));
            }
        }
    } // namespace

    bool LocalTPContext::isLocalTPGpuGraphPolicySupported(
        CollectiveBackendType backend,
        std::string *reason_out)
    {
        const auto &exec = debugEnv().execution;

        // No graph capture in use: LocalTP GPU-native collectives are unaffected.
        if (!exec.gpu_graphs)
        {
            if (reason_out)
            {
                *reason_out = "gpu_graphs_off";
            }
            return true;
        }

        if (backend != CollectiveBackendType::NCCL &&
            backend != CollectiveBackendType::RCCL)
        {
            if (reason_out)
            {
                *reason_out = "unsupported_backend";
            }
            return false;
        }

        /*
         * A backend-only policy query describes homogeneous NCCL or RCCL
         * LocalTP. Its only graph-enabled architecture is therefore direct
         * collective capture inside one full graph. Heterogeneous segmentation
         * is admitted later by DeviceGraphOrchestrator, where the complete
         * device list and graph collective inventory are both available.
         */
        if (exec.gpu_graph_capture_collectives)
        {
            if (reason_out)
            {
                *reason_out = (backend == CollectiveBackendType::NCCL)
                                  ? "nccl_captured_collectives_enabled"
                                  : "rccl_captured_collectives_enabled";
            }
            return true;
        }

        if (reason_out)
        {
            *reason_out = "homogeneous_collectives_require_full_graph_capture";
        }
        return false;
    }

    bool LocalTPContext::isLocalTPGpuGraphPolicySupported(std::string *reason_out) const
    {
        return isLocalTPGpuGraphPolicySupported(backend_, reason_out);
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

    // =========================================================================
    // Construction
    // =========================================================================

    LocalTPContext::LocalTPContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights,
        CollectiveBackendType backend)
        : context_id_(next_context_id_.fetch_add(1, std::memory_order_relaxed)),
          devices_(std::move(devices))
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
        onstream_sequence_by_slot_.assign(devices_.size(), 0);
        fp16_scratch_buffers_.assign(devices_.size(), nullptr);
        fp16_scratch_counts_.assign(devices_.size(), 0);
        fp16_scratch_memory_leases_.resize(devices_.size());
        graph_capture_boundary_device_words_.assign(devices_.size(), nullptr);
        graph_capture_boundary_memory_leases_.resize(devices_.size());

        LOG_DEBUG("LocalTPContext created: degree=" << degree()
                                                    << ", backend=" << collectiveBackendTypeToString(backend_));
        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTEXT] event=localtp_create"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " degree=" << degree()
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " devices=" << [&]()
                      {
                            std::string out;
                            for (size_t i = 0; i < devices_.size(); ++i)
                            {
                                if (i > 0)
                                    out += ",";
                                out += devices_[i].toString();
                            }
                            return out; }());
        }

        // Initialize backend for multi-device scenarios
        if (degree() > 1)
        {
            if (!initializeBackend())
            {
                throw std::runtime_error(
                    std::string("LocalTPContext: Failed to initialize collective backend ") +
                    collectiveBackendTypeToString(backend_) +
                    ". Tensor parallelism requires a working collective backend. "
                    "Check GPU availability and driver status. "
                    "Run with NCCL_DEBUG=INFO for details.");
            }
        }
    }

    LocalTPContext::~LocalTPContext()
    {
        abortBackendAfterGraphOwnersReleased();
        releaseGraphCaptureBoundaryDeviceWords();
        releaseFp16ScratchBuffers();

        const uint64_t attempts = nccl_allreduce_attempts_.load();
        const uint64_t success = nccl_allreduce_success_.load();
        const uint64_t failures = nccl_allreduce_failures_.load();

        if (attempts == 0)
        {
            return;
        }

        LOG_DEBUG("[LocalTPContext][Telemetry] "
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
        const bool was_set = abort_requested_.exchange(true, std::memory_order_acq_rel);
        if (was_set)
        {
            LOG_TRACE("[LocalTPContext] Fatal cancellation was already published");
            return;
        }

        LOG_ERROR("[LocalTPContext] Fatal cancellation published; communicator abort is deferred "
                  "until native graph owners are released"
                  << " context_id=" << context_id_
                  << " context=" << static_cast<const void *>(this));

        if (debugEnv().tp_collective_contract_trace)
        {
            std::lock_guard<std::mutex> trace_lock(contract_trace_mutex_);
            for (const auto &entry_pair : onstream_contracts_)
            {
                const auto &entry = entry_pair.second;
                LOG_WARN("[TP_COLLECTIVE_CONTRACT] event=localtp_abort_pending"
                         << " context_id=" << context_id_
                         << " context=" << static_cast<const void *>(this)
                         << " sequence=" << entry_pair.first
                         << " stage=" << (entry.stage_name.empty() ? "(none)" : entry.stage_name)
                         << " arrivals=" << entry.arrivals << "/" << degree()
                         << " seen_slots_mask=0x" << std::hex << entry.seen_slots << std::dec
                         << " count=" << entry.count
                         << " dtype=" << entry.dtype
                         << " precision=" << (entry.precision.empty() ? "(default)" : entry.precision));
            }
        }

        /*
         * Wake every host rendezvous immediately. The owner now unwinds the
         * participant runners; LocalTPContext destruction performs the actual
         * communicator abort only after those graph owners have disappeared.
         */
        barrier_cv_.notify_all();
        grouped_onstream_allreduce_cv_.notify_all();
        graph_capture_boundary_cv_.notify_all();
    }

    void LocalTPContext::abortBackendAfterGraphOwnersReleased() noexcept
    {
        if (!abort_requested_.load(std::memory_order_acquire) ||
            !backend_initialized_ ||
            !backend_impl_)
        {
            return;
        }

        LOG_ERROR("[LocalTPContext] Native graph owners released; aborting collective backend"
                  << " context_id=" << context_id_
                  << " context=" << static_cast<const void *>(this)
                  << " backend=" << collectiveBackendTypeToString(backend_));
        backend_impl_->abort();
        backend_initialized_ = false;
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

    void LocalTPContext::setBackendForTesting(
        std::unique_ptr<ICollectiveBackend> backend,
        CollectiveBackendType backend_type,
        bool initialized)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        backend_impl_ = std::move(backend);
        backend_ = backend_type;
        backend_initialized_ = initialized && backend_impl_;
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
        // CPU-Only TP: Barrier-synchronized host-memory allreduce
        // ================================================================
        // For LOCAL TP with all-CPU devices (multi-socket NUMA), each worker
        // thread has its tensor in host memory. Use barrier synchronization
        // to collect host pointers and reduce in-place.
        if (backend_ == CollectiveBackendType::HOST && degree() > 1)
        {
            lock.unlock();
            return allreduceCpuBarrier(
                tensor, stage_name, effective_count, nullptr);
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
        // Multi-GPU Backends (NCCL/RCCL): Require barrier-free per-device allreduce
        // ================================================================
        // Each device thread independently calls rcclAllReduce with its own
        // communicator. RCCL internally matches calls across devices.
        // Stream dependencies ensure GPU-side ordering — host never blocks.
        //
        if (backend_impl_->isMultiGpuSingleProcess() && degree() > 1)
        {
            // The backend call is independently enqueued by each device
            // participant; no host barrier is part of this production path.
            lock.unlock();
            return allreducePerDeviceRequired(tensor, stage_name, effective_count);
        }
        else
        {
            /*
             * The legacy collective API owns an internal stream that is not
             * returned to this caller. Publishing its output would therefore
             * require guessing a completion stream after work was already
             * enqueued. GPU stages must enter through allreduceOnStream(), where
             * launch and publication share the same explicit stream.
             */
            throw std::runtime_error(
                "LocalTPContext::allreduce requires allreduceOnStream for GPU "
                "collectives");
        }
    }

    bool LocalTPContext::allreduceOnStream(TensorBase *tensor, const std::string &stage_name,
                                           size_t count, void *stream,
                                           const std::string &precision)
    {
        if (!stream)
        {
            throw std::invalid_argument("LocalTPContext::allreduceOnStream requires a non-null GPU stream");
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

        // HOST collectives are host-staged by definition, including the
        // deliberately heterogeneous CPU/GPU case. Carry the exact producer
        // stream into the barrier so staging and publication remain ordered.
        if (backend_ == CollectiveBackendType::HOST)
        {
            return allreduceCpuBarrier(
                tensor, stage_name, effective_count, stream);
        }

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

        // =================================================================
        // FP16 mixed-precision allreduce path
        // =================================================================
        // Precision can be set per-layer via the schema precision policy,
        // via a graph-level override, or globally via LLAMINAR_ALLREDUCE_PRECISION.
        // Per-call precision (from schema) takes priority over the global env.
        CollectiveDataType dtype = tensorDTypeToCollective(tensor);
        const std::string effective_precision =
            precision.empty() ? std::string(kDefaultAllreducePrecision) : precision;
        const bool grouped_explicit_streams =
            backend_ == CollectiveBackendType::NCCL ||
            backend_ == CollectiveBackendType::RCCL ||
            backend_ == CollectiveBackendType::HETEROGENEOUS;

        if (grouped_explicit_streams)
        {
            const bool supported =
                backend_impl_ &&
                (backend_ == CollectiveBackendType::HETEROGENEOUS
                     ? backend_impl_->supportsAllreduceMultiOnStreamsWithHostCompletionTicket()
                     : backend_impl_->supportsAllreduceMultiOnStreams());
            if (!supported)
            {
                LOG_ERROR("LocalTPContext::allreduceOnStream: backend "
                          << collectiveBackendTypeToString(backend_)
                          << " does not support the required grouped explicit-stream completion authority for stage="
                          << (stage_name.empty() ? "(none)" : stage_name));
                requestAbort();
                return false;
            }
        }
        else if (!rendezvousOnStreamCollective(device_index, tensor, stage_name,
                                               effective_count, dtype, stream, precision))
        {
            requestAbort();
            return false;
        }

        // Homogeneous GPU domains use one grouped launch over every participant's
        // explicit stream, including while those streams are being graph-captured.
        // Capturing independent per-device NCCL/RCCL calls is fragile because a
        // single asymmetric capture/replay decision poisons the communicator.
        const bool use_fp16_allreduce =
            effective_precision == "fp16" &&
            dtype == CollectiveDataType::FLOAT32 &&
            batchInvariantAllreduceDecisionElements(
                effective_count, tensor->cols()) >=
                debugEnv().allreduce_fp16_min_elements;

        if (use_fp16_allreduce)
        {
            // The graph planner owns capacity. Execution only validates and
            // borrows the persistent device-local scratch reservation.
            void *fp16_buf = requireReservedFp16Scratch(
                device_index,
                effective_count,
                stage_name,
                "LocalTPContext::allreduceOnStream");
            {
                const int ordinal = devices_[device_index].device_ordinal;
                bool cast_ok = false;

                // Step 1: Cast FP32 → FP16 on caller's stream
#ifdef HAVE_CUDA
                if (devices_[device_index].device_type == DeviceType::CUDA)
                {
                    cast_ok = (cudaCastFP32ToFP16(
                                   static_cast<const float *>(buffer), fp16_buf,
                                   effective_count,
                                   ordinal,
                                   static_cast<cudaStream_t>(stream)) == 0);
                }
#endif
#ifdef HAVE_ROCM
                if (devices_[device_index].device_type == DeviceType::ROCm)
                {
                    cast_ok = (rocmCastFP32ToFP16(
                                   static_cast<const float *>(buffer), fp16_buf,
                                   effective_count, ordinal, stream) == 0);
                }
#endif
                if (!cast_ok)
                {
                    LOG_ERROR("LocalTPContext: FP32->FP16 cast failed for stage="
                              << (stage_name.empty() ? "(none)" : stage_name)
                              << "; failing fast to avoid asymmetric transport");
                    requestAbort();
                    return false;
                }
                else
                {
                    // Step 2: Allreduce in FP16 (half the bytes!)
                    const bool ar_ok = grouped_explicit_streams
                                           ? allreduceGroupedOnExplicitStreams(
                                                 fp16_buf,
                                                 effective_count,
                                                 CollectiveDataType::FLOAT16,
                                                 device_index,
                                                 stream,
                                                 stage_name,
                                                 effective_precision)
                                           : backend_impl_->allreduceSingleDeviceOnStream(
                                                 fp16_buf, effective_count, CollectiveDataType::FLOAT16,
                                                 CollectiveOp::ALLREDUCE_SUM, device_index, stream);

                    if (!ar_ok)
                    {
                        LOG_ERROR("LocalTPContext: FP16 allreduce failed: "
                                  << (backend_impl_ ? backend_impl_->lastError() : "missing backend")
                                  << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                        if (grouped_explicit_streams)
                        {
                            requestAbort();
                            return false;
                        }
                        requestAbort();
                        return false;
                    }
                    else
                    {
                        // Step 3: Cast FP16 → FP32 back into the original buffer
                        bool back_ok = false;
#ifdef HAVE_CUDA
                        if (devices_[device_index].device_type == DeviceType::CUDA)
                        {
                            back_ok = (cudaCastFP16ToFP32(
                                           fp16_buf,
                                           static_cast<float *>(buffer),
                                           effective_count,
                                           ordinal,
                                           static_cast<cudaStream_t>(stream)) == 0);
                        }
#endif
#ifdef HAVE_ROCM
                        if (devices_[device_index].device_type == DeviceType::ROCm)
                        {
                            back_ok = (rocmCastFP16ToFP32(
                                           fp16_buf,
                                           static_cast<float *>(buffer),
                                           effective_count, ordinal, stream) == 0);
                        }
#endif
                        if (back_ok)
                        {
                            recordLocalTPRuntimeAllreduce(
                                device_group_, backend_, devices_[device_index].toLocalDeviceId(),
                                stage_name, static_cast<size_t>(degree()), effective_count,
                                CollectiveDataType::FLOAT16,
                                backend_ == CollectiveBackendType::HETEROGENEOUS
                                    ? "on_stream_grouped_host_ticket_fp16_scratch"
                                    : (grouped_explicit_streams
                                           ? "on_stream_grouped_fp16_scratch"
                                           : "on_stream_fp16_scratch"),
                                effective_precision);
                            TransferEngine::publishDeviceWrite(
                                tensor,
                                devices_[device_index].toLocalDeviceId(),
                                stream);
                            return true;
                        }
                        LOG_ERROR("LocalTPContext: FP16->FP32 cast-back failed for stage="
                                  << (stage_name.empty() ? "(none)" : stage_name)
                                  << "; failing fast to avoid asymmetric transport");
                        requestAbort();
                        return false;
                    }
                }
            }
        }

        // =================================================================
        // Standard FP32 allreduce path. FP16 transport failures fail fast above;
        // grouped collectives must never fall through asymmetrically.
        // =================================================================
        if (grouped_explicit_streams)
        {
            const bool success = allreduceGroupedOnExplicitStreams(
                buffer, effective_count, dtype, device_index, stream,
                stage_name, effective_precision);
            if (!success)
            {
                LOG_ERROR("LocalTPContext::allreduceOnStream: grouped explicit-stream allreduce failed"
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                          << " backend=" << collectiveBackendTypeToString(backend_)
                          << " error=" << (backend_impl_ ? backend_impl_->lastError() : "missing backend"));
                requestAbort();
                return false;
            }

            recordLocalTPRuntimeAllreduce(
                device_group_, backend_, devices_[device_index].toLocalDeviceId(),
                stage_name, static_cast<size_t>(degree()), effective_count,
                dtype,
                backend_ == CollectiveBackendType::HETEROGENEOUS
                    ? "on_stream_grouped_host_ticket"
                    : "on_stream_grouped",
                effective_precision);
            if (backend_ == CollectiveBackendType::HETEROGENEOUS)
            {
                // The authenticated ticket already observed every terminal H2D
                // event. Recording a compute-stream event here would invent a
                // producer that did not perform the write.
                TransferEngine::publishCompletedDeviceWrite(
                    tensor,
                    devices_[device_index].toLocalDeviceId());
            }
            else
            {
                TransferEngine::publishDeviceWrite(
                    tensor,
                    devices_[device_index].toLocalDeviceId(),
                    stream);
            }
            return true;
        }

        bool success = backend_impl_->allreduceSingleDeviceOnStream(
            buffer, effective_count, dtype, CollectiveOp::ALLREDUCE_SUM,
            device_index, stream);

        if (success)
        {
            recordLocalTPRuntimeAllreduce(
                device_group_, backend_, devices_[device_index].toLocalDeviceId(),
                stage_name, static_cast<size_t>(degree()), effective_count,
                dtype, "on_stream_native", effective_precision);
            // Mark tensor dirty and record completion event on the allreduce stream.
            // This ensures ensureOnHost() waits for the allreduce to finish before D2H.
            TransferEngine::publishDeviceWrite(
                tensor,
                devices_[device_index].toLocalDeviceId(),
                stream);
            return true;
        }

        LOG_ERROR("LocalTPContext::allreduceOnStream: the configured homogeneous collective backend "
                  "does not implement the required explicit-stream allreduce for stage="
                  << stage_name
                  << "; synchronous barrier fallback is forbidden");
        return false;
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

        if (!device_group_.allCPU())
        {
            throw std::runtime_error(
                "LocalTPContext::allreduce out-of-place GPU execution is forbidden; "
                "GPU collectives require allreduceOnStream() with a device-resident "
                "buffer and the exact producer stream");
        }

        // Check if backend is initialized
        if (!backend_initialized_ || !backend_impl_)
        {
            throw std::runtime_error(
                "LocalTPContext::allreduce requires an initialized CPU collective backend");
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

        // CPU-only TP: use host pointer with single-buffer allreduce
        if (backend_ == CollectiveBackendType::HOST)
        {
            float *buffer = tensor->mutable_data();
            size_t count = tensor->numel();
            CollectiveDataType dtype = tensorDTypeToCollective(tensor);
            return backend_impl_->allreduce(buffer, count, dtype, CollectiveOp::ALLREDUCE_SUM);
        }

        throw std::runtime_error(
            "LocalTPContext::allreduceImpl cannot execute a GPU collective "
            "without an explicit producer stream");
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

    bool LocalTPContext::allreducePerDeviceRequired(TensorBase *tensor,
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
            LOG_ERROR("LocalTPContext::allreducePerDeviceRequired: tensor device "
                      << (tensor_device.has_value() ? tensor_device->toString() : "none")
                      << " not found in devices list (degree=" << degree() << ")");
            return false;
        }

        /*
         * The backend places its completion wait on this registered compute
         * stream. Retaining the same handle here lets input preparation join the
         * preceding producer and output publication record after the collective.
         */
        if (compute_streams_.size() != devices_.size() ||
            !compute_streams_[device_index])
        {
            LOG_ERROR(
                "LocalTPContext::allreducePerDeviceRequired: missing registered "
                "compute stream for slot "
                << device_index);
            return false;
        }

        DeviceId expected_device = devices_[device_index].toLocalDeviceId();
        void *const compute_stream = compute_streams_[device_index];
        TransferEngine::prepareDeviceInput(
            tensor, expected_device, compute_stream);

        void *buffer = tensor->gpu_data_ptr();
        if (!buffer)
        {
            LOG_ERROR("LocalTPContext::allreducePerDeviceRequired: null GPU buffer for slot "
                      << device_index);
            return false;
        }

        // 3. Enqueue the required per-device allreduce (barrier-free).
        CollectiveDataType dtype = tensorDTypeToCollective(tensor);
        bool success = backend_impl_->allreduceSingleDeviceAsync(
            buffer, count, dtype, CollectiveOp::ALLREDUCE_SUM, device_index);

        if (success)
        {
            TransferEngine::publishDeviceWrite(
                tensor,
                expected_device,
                compute_streams_[device_index]);
            return true;
        }

        LOG_ERROR(
            "LocalTPContext::allreducePerDeviceRequired: backend rejected the "
            "required per-device asynchronous allreduce for stage="
            << stage_name << "; blocking barrier recovery is forbidden");
        return false;
    }

    // =========================================================================
    // CPU-Only Barrier-Synchronized Allreduce
    // =========================================================================
    //
    // For LOCAL TP where all devices are CPU (multi-socket NUMA), each worker
    // thread has its tensor in host memory. We use barrier synchronization to:
    // 1. Collect host pointers from all threads
    // 2. Have the last-arriving thread perform element-wise reduction
    // 3. Broadcast result to all participants
    // =========================================================================

    bool LocalTPContext::allreduceCpuBarrier(
        TensorBase *tensor,
        const std::string &stage_name,
        size_t count,
        void *producer_stream)
    {
        const auto arrival_device = tensor ? tensor->current_device() : std::nullopt;
        if (arrival_device && arrival_device->is_gpu() && !producer_stream)
        {
            throw std::invalid_argument(
                "LocalTPContext::allreduceCpuBarrier requires the exact "
                "producer stream for a GPU participant");
        }

        const int num_participants = degree();

        std::unique_lock<std::mutex> lock(barrier_mutex_);
        uint64_t my_generation = barrier_generation_.load();
        int arrival_order = barrier_count_.fetch_add(1);

        if (arrival_order == 0)
        {
            barrier_tensors_.clear();
            barrier_tensors_.resize(num_participants, nullptr);
            barrier_producer_streams_.clear();
            barrier_producer_streams_.resize(num_participants, nullptr);
            barrier_element_count_ = count;
            barrier_stage_name_ = stage_name;
            LOG_DEBUG("LocalTPContext::allreduceCpuBarrier: First arrival, "
                      << "stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << ", count=" << count
                      << ", waiting for " << (num_participants - 1) << " more CPU devices");
        }
        else if (!barrier_stage_name_.empty() && !stage_name.empty() && barrier_stage_name_ != stage_name)
        {
            LOG_ERROR("LocalTPContext::allreduceCpuBarrier: stage mismatch while waiting for HOST allreduce: first='"
                      << barrier_stage_name_ << "' arrival='" << stage_name
                      << "' arrival_order=" << arrival_order
                      << ", expected=" << num_participants);
            barrier_count_.store(0);
            barrier_generation_.fetch_add(1);
            barrier_tensors_.clear();
            barrier_producer_streams_.clear();
            barrier_stage_name_.clear();
            barrier_element_count_ = 0;

            lock.unlock();
            barrier_cv_.notify_all();
            return false;
        }

        // Use arrival_order for slot assignment (sum is commutative, order doesn't matter)
        barrier_tensors_[arrival_order] = tensor;
        barrier_producer_streams_[arrival_order] = producer_stream;

        if (arrival_order + 1 < num_participants)
        {
            // Wait for the last arrival to complete the reduction
            const int barrier_timeout_ms =
                collective_timeout_policy::effectiveCollectTimeoutMs(
                    debugEnv().tp_collect_timeout_ms);

            bool completed = barrier_cv_.wait_for(
                lock,
                std::chrono::milliseconds(barrier_timeout_ms),
                [this, my_generation]()
                                                  { return barrier_generation_.load() > my_generation; });

            if (!completed)
            {
                LOG_ERROR("LocalTPContext::allreduceCpuBarrier: TIMEOUT after " << barrier_timeout_ms
                                                                                << "ms waiting for barrier! "
                                                                                << "arrival_order=" << arrival_order
                                                                                << ", expected=" << num_participants);
                barrier_count_.store(0);
                barrier_generation_.fetch_add(1);
                barrier_tensors_.clear();
                barrier_producer_streams_.clear();
                barrier_stage_name_.clear();
                barrier_element_count_ = 0;

                lock.unlock();
                barrier_cv_.notify_all();
                return false;
            }

            return barrier_result_;
        }

        // =================================================================
        // LAST ARRIVAL: Perform host-memory allreduce
        // =================================================================
        size_t effective_count = barrier_element_count_;
        if (effective_count == 0 && barrier_tensors_[0])
            effective_count = barrier_tensors_[0]->numel();

        LOG_DEBUG("LocalTPContext::allreduceCpuBarrier: All " << num_participants
                                                              << " CPU devices arrived, reducing "
                                                              << effective_count << " elements");

        const size_t bytes = effective_count * sizeof(float);

        if (debugEnv().runtime_debug.tp_host_allreduce_trace)
        {
            for (int i = 0; i < num_participants; ++i)
            {
                TensorBase *tb = barrier_tensors_[i];
                LOG_DEBUG("[LocalTPContext][HostAllreduceTrace] slot=" << i
                                                                       << " tensor=" << static_cast<void *>(tb)
                                                                       << " current_device=" << (tb && tb->current_device() ? tb->current_device()->toString() : "none")
                                                                       << " host=" << (tb ? tb->raw_data() : nullptr)
                                                                       << " gpu=" << (tb ? tb->gpu_data_ptr() : nullptr)
                                                                       << " numel=" << (tb ? tb->numel() : 0)
                                                                       << " count=" << effective_count
                                                                       << " stage=" << barrier_stage_name_);
            }
        }

        auto copy_to_host = [&](int slot, TensorBase *tb, float *dst) -> bool
        {
            if (!tb || !dst)
                return false;

            auto dev = tb->current_device();
            if (dev && dev->is_gpu() && tb->gpu_data_ptr())
            {
                IBackend *backend = getBackendForDevice(*dev);
                if (!backend)
                    return false;
                void *const stream = barrier_producer_streams_.at(slot);
                if (!stream)
                {
                    throw std::runtime_error(
                        "LocalTPContext::allreduceCpuBarrier lost a GPU "
                        "participant's producer stream");
                }
                return backend->deviceToHost(
                    dst, tb->gpu_data_ptr(), bytes, dev->gpu_ordinal(), stream);
            }

            const float *src = tb->data();
            if (!src)
                return false;
            std::memcpy(dst, src, bytes);
            return true;
        };

        auto copy_from_host = [&](int slot, TensorBase *tb, const float *src) -> bool
        {
            if (!tb || !src)
                return false;

            auto dev = tb->current_device();
            if (dev && dev->is_gpu() && tb->gpu_data_ptr())
            {
                IBackend *backend = getBackendForDevice(*dev);
                if (!backend)
                    return false;
                void *const stream = barrier_producer_streams_.at(slot);
                if (!stream)
                {
                    throw std::runtime_error(
                        "LocalTPContext::allreduceCpuBarrier lost a GPU "
                        "participant's publication stream");
                }
                if (!backend->hostToDevice(
                        tb->gpu_data_ptr(), src, bytes, dev->gpu_ordinal(), stream))
                    return false;
                TransferEngine::publishDeviceWrite(tb, *dev, stream);
                return true;
            }

            float *dst = tb->mutable_data();
            if (!dst)
                return false;
            std::memcpy(dst, src, bytes);
            return true;
        };

        std::vector<float> accum(effective_count, 0.0f);
        std::vector<float> temp(effective_count, 0.0f);

        if (!copy_to_host(0, barrier_tensors_[0], accum.data()))
        {
            LOG_ERROR("LocalTPContext::allreduceCpuBarrier: failed to stage slot 0 to host");
            barrier_result_ = false;
            barrier_count_.store(0);
            barrier_producer_streams_.clear();
            barrier_generation_.fetch_add(1);
            lock.unlock();
            barrier_cv_.notify_all();
            return false;
        }

        // Sum contributions from all other tensors
        for (int i = 1; i < num_participants; ++i)
        {
            if (!copy_to_host(i, barrier_tensors_[i], temp.data()))
            {
                LOG_ERROR("LocalTPContext::allreduceCpuBarrier: failed to stage slot " << i << " to host");
                barrier_result_ = false;
                barrier_count_.store(0);
                barrier_producer_streams_.clear();
                barrier_generation_.fetch_add(1);
                lock.unlock();
                barrier_cv_.notify_all();
                return false;
            }
            for (size_t j = 0; j < effective_count; ++j)
                accum[j] += temp[j];
        }

        // Copy reduced result to all other tensors
        for (int i = 0; i < num_participants; ++i)
        {
            if (!copy_from_host(i, barrier_tensors_[i], accum.data()))
            {
                LOG_ERROR("LocalTPContext::allreduceCpuBarrier: failed to write reduced data to slot " << i);
                barrier_result_ = false;
                barrier_count_.store(0);
                barrier_producer_streams_.clear();
                barrier_generation_.fetch_add(1);
                lock.unlock();
                barrier_cv_.notify_all();
                return false;
            }
        }

        // Cleanup and release waiters
        barrier_result_ = true;
        barrier_tensors_.clear();
        barrier_producer_streams_.clear();
        barrier_stage_name_.clear();
        barrier_element_count_ = 0;
        barrier_count_.store(0);
        barrier_generation_.fetch_add(1);

        LOG_DEBUG("LocalTPContext::allreduceCpuBarrier: CPU allreduce completed successfully");

        lock.unlock();
        barrier_cv_.notify_all();
        return true;
    }

    // =========================================================================
    // CPU-Only Barrier-Synchronized Allgather
    // =========================================================================

    bool LocalTPContext::allgatherCpuBarrier(const TensorBase *local_shard, TensorBase *global_tensor)
    {
        const int num_participants = degree();

        // NOTE: For CPU TP, all tensors have the same generic home_device "CPU"
        // (no NUMA distinction), so we cannot determine device_index from the tensor.
        // We use arrival_order for slot assignment. This means shard ordering depends
        // on thread arrival order, which is non-deterministic.
        // In practice, this method is currently dead code for LOCAL CPU TP because
        // RankOrchestrator::gatherLogits() handles LM head vocab gathering
        // directly at the orchestrator level, bypassing LocalTPContext::allgather().
        // If this method is ever used in a context where shard ordering matters,
        // a thread-safe device identification mechanism will be needed.

        std::unique_lock<std::mutex> lock(barrier_mutex_);
        uint64_t my_generation = barrier_generation_.load();
        int arrival_order = barrier_count_.fetch_add(1);

        if (arrival_order == 0)
        {
            barrier_tensors_.clear();
            barrier_tensors_.resize(num_participants, nullptr);
            barrier_element_count_ = local_shard->numel(); // elements per shard
            barrier_stage_name_ = "allgather_cpu";
            LOG_DEBUG("LocalTPContext::allgatherCpuBarrier: First arrival, "
                      << "shard_elements=" << local_shard->numel()
                      << ", waiting for " << (num_participants - 1) << " more CPU devices");
        }

        // Store the shard using arrival_order (const_cast safe: we only read from it)
        barrier_tensors_[arrival_order] = const_cast<TensorBase *>(local_shard);

        if (arrival_order + 1 < num_participants)
        {
            const int barrier_timeout_ms =
                collective_timeout_policy::effectiveCollectTimeoutMs(
                    debugEnv().tp_collect_timeout_ms);

            bool completed = barrier_cv_.wait_for(
                lock,
                std::chrono::milliseconds(barrier_timeout_ms),
                [this, my_generation]()
                                                  { return barrier_generation_.load() > my_generation; });

            if (!completed)
            {
                LOG_ERROR("LocalTPContext::allgatherCpuBarrier: TIMEOUT after "
                          << barrier_timeout_ms << "ms");
                barrier_count_.store(0);
                barrier_generation_.fetch_add(1);
                barrier_tensors_.clear();
                barrier_stage_name_.clear();
                barrier_element_count_ = 0;
                lock.unlock();
                barrier_cv_.notify_all();
                return false;
            }

            // After barrier release, the last arrival has already written
            // the gathered result to its own global_tensor. In typical LOCAL TP
            // usage, all threads share the same combined_logits_ buffer through
            // the orchestrator, so no additional copy is needed.
            return barrier_result_;
        }

        // =================================================================
        // LAST ARRIVAL: Concatenate all shards
        // =================================================================
        size_t shard_elements = barrier_element_count_;

        LOG_DEBUG("LocalTPContext::allgatherCpuBarrier: All " << num_participants
                                                              << " CPU devices arrived, gathering "
                                                              << shard_elements << " elements each");

        // Concatenate all shards into global_tensor
        float *output = global_tensor->mutable_data();
        for (int i = 0; i < num_participants; ++i)
        {
            const float *shard_data = barrier_tensors_[i]->data();
            std::memcpy(output + (i * shard_elements), shard_data, shard_elements * sizeof(float));
        }

        barrier_result_ = true;
        barrier_tensors_.clear();
        barrier_stage_name_.clear();
        barrier_element_count_ = 0;
        barrier_count_.store(0);
        barrier_generation_.fetch_add(1);

        LOG_DEBUG("LocalTPContext::allgatherCpuBarrier: CPU allgather completed");

        lock.unlock();
        barrier_cv_.notify_all();
        return true;
    }

    bool LocalTPContext::allgather(const TensorBase *local_shard, TensorBase *global_tensor)
    {
        if (!local_shard || !global_tensor)
        {
            LOG_ERROR("LocalTPContext::allgather: null tensor");
            return false;
        }

        std::unique_lock<std::mutex> lock(mutex_);

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

        // CPU-only TP: Use barrier-synchronized host-memory allgather
        if (backend_ == CollectiveBackendType::HOST && degree() > 1)
        {
            // Release the main mutex before entering barrier (barrier has its own mutex)
            // to avoid deadlock: all threads must arrive at barrier, but mutex_ is exclusive.
            lock.unlock();
            return allgatherCpuBarrier(local_shard, global_tensor);
        }

        /*
         * The streamless allgather backend API cannot expose the producer
         * ordering needed by a device-resident consumer. Production GPU graphs
         * use allgatherRawOnStream() or collectiveSidebandOnStream(), both of
         * which make the launch stream part of the operation contract.
         */
        throw std::runtime_error(
            "LocalTPContext::allgather requires an explicit-stream GPU "
            "collective API");
    }

    bool LocalTPContext::supportsRawAllgatherOnStreamGraphCapture() const
    {
        if (degree() <= 1 || !backend_initialized_ || !backend_impl_)
            return false;
        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
            return false;
        if (!backend_impl_->isMultiGpuSingleProcess())
            return false;
        return backend_impl_->supportsAllgatherSingleDeviceOnStream();
    }

    bool LocalTPContext::supportsCollectiveSidebandOnStreamGraphCapture() const
    {
        if (degree() <= 1 || !backend_initialized_ || !backend_impl_)
            return false;
        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
            return false;
        if (!backend_impl_->isMultiGpuSingleProcess())
            return false;
        return backend_impl_->supportsAllreduceWithSidebandsMultiOnStreams();
    }

    /**
     * @brief Reusable LocalTP barrier for named GPU graph-capture lifecycle boundaries.
     *
     * The rendezvous protects multi-device graph capture from asymmetric phase
     * transitions. Without this barrier one participant can start HIP/CUDA stream
     * capture while another participant is still synchronizing or publishing the
     * previous eager prefill chunk; ROCm then reports capture-implicit stream
     * dependency errors and the following RCCL group launch fails. The barrier is
     * intentionally strict and aborts the LocalTP context on mismatched names,
     * duplicate arrivals, or timeout so later collectives do not limp onward in a
     * poisoned state.
     */
    bool LocalTPContext::graphCaptureBoundaryRendezvous(
        const std::string &boundary_name,
        int device_index,
        int timeout_ms)
    {
        if (degree() <= 1)
            return true;

        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::graphCaptureBoundaryRendezvous: invalid slot "
                      << device_index << " degree=" << degree()
                      << " boundary=" << boundary_name);
            requestAbort();
            return false;
        }

        std::unique_lock<std::mutex> lock(graph_capture_boundary_mutex_);
        if (abort_requested_.load(std::memory_order_acquire))
            return false;

        auto reset_generation_state = [&]()
        {
            graph_capture_boundary_arrivals_ = 0;
            graph_capture_boundary_departures_ = 0;
            graph_capture_boundary_ready_ = false;
            graph_capture_boundary_result_ = false;
            graph_capture_boundary_name_.clear();
            graph_capture_boundary_error_.clear();
            graph_capture_boundary_seen_.clear();
            ++graph_capture_boundary_generation_;
        };

        auto fail_generation = [&](const std::string &error)
        {
            graph_capture_boundary_result_ = false;
            graph_capture_boundary_error_ = error;
            reset_generation_state();
            abort_requested_.store(true, std::memory_order_release);
            LOG_ERROR("LocalTPContext::graphCaptureBoundaryRendezvous: " << error);
            lock.unlock();
            graph_capture_boundary_cv_.notify_all();
        };

        while (graph_capture_boundary_arrivals_ >= degree() &&
               graph_capture_boundary_departures_ > 0 &&
               !abort_requested_.load(std::memory_order_acquire))
        {
            graph_capture_boundary_cv_.wait(lock);
        }
        if (abort_requested_.load(std::memory_order_acquire))
            return false;

        const uint64_t my_generation = graph_capture_boundary_generation_;
        const int arrival_order = graph_capture_boundary_arrivals_++;

        auto depart_generation = [&]() -> bool
        {
            const bool result = graph_capture_boundary_ready_ &&
                                graph_capture_boundary_result_ &&
                                !abort_requested_.load(std::memory_order_acquire);
            ++graph_capture_boundary_departures_;
            if (graph_capture_boundary_departures_ >= degree())
            {
                reset_generation_state();
                lock.unlock();
                graph_capture_boundary_cv_.notify_all();
            }
            return result;
        };

        if (arrival_order >= degree())
        {
            fail_generation("arrival overflow boundary=" + boundary_name);
            return false;
        }

        if (arrival_order == 0)
        {
            graph_capture_boundary_ready_ = false;
            graph_capture_boundary_result_ = false;
            graph_capture_boundary_name_ = boundary_name;
            graph_capture_boundary_error_.clear();
            graph_capture_boundary_seen_.assign(static_cast<size_t>(degree()), false);
        }
        else if (graph_capture_boundary_name_ != boundary_name)
        {
            fail_generation("boundary mismatch expected=" +
                            (graph_capture_boundary_name_.empty() ? std::string("(none)") : graph_capture_boundary_name_) +
                            " actual=" + (boundary_name.empty() ? std::string("(none)") : boundary_name));
            return false;
        }

        if (graph_capture_boundary_seen_[static_cast<size_t>(device_index)])
        {
            fail_generation("duplicate slot arrival slot=" + std::to_string(device_index) +
                            " boundary=" + boundary_name);
            return false;
        }
        graph_capture_boundary_seen_[static_cast<size_t>(device_index)] = true;

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=localtp_graph_capture_boundary_arrival"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " generation=" << my_generation
                      << " slot=" << device_index
                      << " arrival_order=" << arrival_order
                      << " degree=" << degree()
                      << " boundary=" << boundary_name);
        }

        if (arrival_order + 1 < degree())
        {
            auto ready = [&]()
            {
                return abort_requested_.load(std::memory_order_acquire) ||
                       graph_capture_boundary_generation_ > my_generation ||
                       (graph_capture_boundary_generation_ == my_generation &&
                        graph_capture_boundary_ready_);
            };

            bool completed = true;
            if (timeout_ms > 0)
            {
                completed = graph_capture_boundary_cv_.wait_for(
                    lock,
                    std::chrono::milliseconds(timeout_ms),
                    ready);
            }
            else
            {
                graph_capture_boundary_cv_.wait(lock, ready);
            }

            if (!completed)
            {
                fail_generation("timeout waiting for graph-capture boundary peers boundary=" +
                                boundary_name +
                                " arrivals=" + std::to_string(graph_capture_boundary_arrivals_) +
                                " degree=" + std::to_string(degree()));
                return false;
            }

            if (graph_capture_boundary_generation_ != my_generation &&
                !graph_capture_boundary_ready_)
            {
                return false;
            }

            return depart_generation();
        }

        for (int i = 0; i < degree(); ++i)
        {
            if (i >= static_cast<int>(graph_capture_boundary_seen_.size()) ||
                !graph_capture_boundary_seen_[static_cast<size_t>(i)])
            {
                fail_generation("missing graph-capture boundary participant slot=" +
                                std::to_string(i) +
                                " boundary=" + boundary_name);
                return false;
            }
        }

        graph_capture_boundary_result_ = true;
        graph_capture_boundary_ready_ = true;
        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=localtp_graph_capture_boundary_released"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " generation=" << my_generation
                      << " boundary=" << boundary_name);
        }
        graph_capture_boundary_cv_.notify_all();
        return depart_generation();
    }

    bool LocalTPContext::graphCaptureBoundaryOnStream(
        const std::string &boundary_name,
        int device_index,
        void *stream,
        int timeout_ms)
    {
        if (!stream)
        {
            throw std::invalid_argument(
                "LocalTPContext::graphCaptureBoundaryOnStream requires a non-null GPU stream");
        }
        if (degree() <= 1)
            return true;
        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::graphCaptureBoundaryOnStream: invalid slot "
                      << device_index << " degree=" << degree()
                      << " boundary=" << boundary_name);
            requestAbort();
            return false;
        }
        const bool homogeneous_device_collective =
            backend_initialized_ && backend_impl_ &&
            (backend_ == CollectiveBackendType::NCCL ||
             backend_ == CollectiveBackendType::RCCL) &&
            backend_impl_->isMultiGpuSingleProcess() &&
            backend_impl_->supportsAllreduceSingleDeviceOnStream();
        const bool heterogeneous_stream_tickets =
            backend_initialized_ && backend_impl_ &&
            backend_ == CollectiveBackendType::HETEROGENEOUS &&
            backend_impl_->isMultiGpuSingleProcess() &&
            backend_impl_->supportsGraphCaptureLifecycleTickets();
        if (!homogeneous_device_collective && !heterogeneous_stream_tickets)
        {
            LOG_ERROR("LocalTPContext::graphCaptureBoundaryOnStream requires either a "
                      << "homogeneous explicit-stream collective fence or heterogeneous "
                      << "persistent lifecycle tickets"
                      << " boundary=" << boundary_name
                      << " backend=" << collectiveBackendTypeToString(backend_));
            requestAbort();
            return false;
        }
        if (homogeneous_device_collective &&
            (graph_capture_boundary_device_words_.size() != devices_.size() ||
             !graph_capture_boundary_device_words_[static_cast<size_t>(device_index)]))
        {
            LOG_ERROR("LocalTPContext::graphCaptureBoundaryOnStream: persistent device fence "
                      << "storage is unavailable for slot=" << device_index
                      << " boundary=" << boundary_name);
            requestAbort();
            return false;
        }

        /*
         * The first rendezvous prevents one participant from enqueueing this
         * lifecycle collective while another still believes it belongs to a
         * different boundary generation. The second proves every peer has
         * enqueued the matching collective before any thread begins capture or
         * launches the newly instantiated graph.
         */
        if (!graphCaptureBoundaryRendezvous(
                boundary_name +
                    (heterogeneous_stream_tickets
                         ? ":ticket_generation_ready"
                         : ":device_fence_ready"),
                device_index,
                timeout_ms))
        {
            return false;
        }

        if (heterogeneous_stream_tickets)
        {
            /*
             * CUDA and ROCm events cannot be consumed directly by the other
             * vendor's stream. Each participant therefore records its own
             * persistent event, then the host-side generation observes only
             * those exact events. The second rendezvous proves every event was
             * recorded before observation; the third prevents a fast device
             * from entering capture while a slower sibling is still draining.
             */
            if (!backend_impl_->recordGraphCaptureLifecycleTicket(
                    device_index,
                    stream))
            {
                LOG_ERROR("LocalTPContext::graphCaptureBoundaryOnStream: heterogeneous "
                          "ticket publication failed"
                          << " slot=" << device_index
                          << " boundary=" << boundary_name
                          << " backend_error=" << backend_impl_->lastError());
                requestAbort();
                return false;
            }

            if (!graphCaptureBoundaryRendezvous(
                    boundary_name + ":ticket_recorded",
                    device_index,
                    timeout_ms))
            {
                return false;
            }

            if (!backend_impl_->awaitGraphCaptureLifecycleTicket(
                    device_index,
                    timeout_ms))
            {
                LOG_ERROR("LocalTPContext::graphCaptureBoundaryOnStream: heterogeneous "
                          "ticket observation failed"
                          << " slot=" << device_index
                          << " boundary=" << boundary_name
                          << " backend_error=" << backend_impl_->lastError());
                requestAbort();
                return false;
            }

            if (!graphCaptureBoundaryRendezvous(
                    boundary_name + ":ticket_observed",
                    device_index,
                    timeout_ms))
            {
                return false;
            }

            const char *const phase =
                boundary_name.rfind("decode_graph:", 0) == 0
                    ? "decode"
                    : "prefill";
            const DeviceId device =
                devices_[static_cast<size_t>(device_index)].toLocalDeviceId();
            const PerfStatsCollector::Tags tags{
                {"boundary", boundary_name},
                {"backend", collectiveBackendTypeToString(backend_)},
                {"authority", "host_observed_stream_ticket"},
                {"steady_state_replay", "false"}};
            PerfStatsCollector::addCounter(
                "graph_capture",
                "localtp_device_boundary_fences",
                1.0,
                phase,
                device.toString(),
                tags);
            PerfStatsCollector::addCounter(
                "graph_capture",
                "localtp_heterogeneous_ticket_boundary_fences",
                1.0,
                phase,
                device.toString(),
                tags);
            return true;
        }

        void *const device_word =
            graph_capture_boundary_device_words_[static_cast<size_t>(device_index)];

        /*
         * Publish the ordering token on the same explicit stream that enters
         * the collective. This initialization is graph-capturable and makes the
         * token independent of allocator contents without a setup-time default
         * stream or host write.
         */
        bool token_ready = false;
#ifdef HAVE_CUDA
        if (device_group_.allCUDA())
        {
            token_ready = nccl_backend_detail::cudaMemsetAsyncDevice(
                device_word,
                0,
                sizeof(int32_t),
                devices_[static_cast<size_t>(device_index)].device_ordinal,
                stream);
        }
#endif
#ifdef HAVE_ROCM
        if (device_group_.allROCm())
        {
            token_ready = rccl_backend_detail::hipMemsetAsyncDevice(
                device_word,
                0,
                sizeof(int32_t),
                devices_[static_cast<size_t>(device_index)].device_ordinal,
                stream);
        }
#endif
        if (!token_ready)
        {
            LOG_ERROR("LocalTPContext::graphCaptureBoundaryOnStream: device fence "
                      "token publication failed"
                      << " slot=" << device_index
                      << " boundary=" << boundary_name);
            requestAbort();
            return false;
        }

        if (!backend_impl_->allreduceSingleDeviceOnStream(
                device_word,
                /*count=*/1,
                CollectiveDataType::INT32,
                CollectiveOp::ALLREDUCE_SUM,
                device_index,
                stream))
        {
            LOG_ERROR("LocalTPContext::graphCaptureBoundaryOnStream: device fence enqueue failed"
                      << " slot=" << device_index
                      << " boundary=" << boundary_name
                      << " backend_error=" << backend_impl_->lastError());
            requestAbort();
            return false;
        }

        if (!graphCaptureBoundaryRendezvous(
                boundary_name + ":device_fence_enqueued",
                device_index,
                timeout_ms))
        {
            return false;
        }

        const char *const phase =
            boundary_name.rfind("decode_graph:", 0) == 0
                ? "decode"
                : "prefill";
        PerfStatsCollector::addCounter(
            "graph_capture",
            "localtp_device_boundary_fences",
            1.0,
            phase,
            devices_[static_cast<size_t>(device_index)].toLocalDeviceId().toString(),
            {{"boundary", boundary_name},
             {"backend", collectiveBackendTypeToString(backend_)},
             {"authority", "native_device_collective"},
             {"steady_state_replay", "false"}});
        return true;
    }

    bool LocalTPContext::allreduceWithSidebandsOnStream(
        TensorBase *tensor,
        const std::string &stage_name,
        size_t count,
        void *producer_stream,
        const std::string &precision,
        const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
        int device_index)
    {
        if (!producer_stream)
            throw std::invalid_argument("LocalTPContext::allreduceWithSidebandsOnStream requires a non-null GPU stream");

        if (sidebands.empty())
            return allreduceOnStream(tensor, stage_name, count, producer_stream, precision);

        if (!tensor)
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: null tensor"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }

        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: invalid device_index="
                      << device_index << " degree=" << degree()
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }

        if (degree() <= 1)
            return allreduceOnStream(tensor, stage_name, count, producer_stream, precision);

        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: backend is not initialized"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }

        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: only homogeneous NCCL/RCCL domains are supported"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_));
            return false;
        }

        if (!backend_impl_->isMultiGpuSingleProcess())
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: backend is not in multi-GPU single-process mode"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }

        if (!backend_impl_->supportsAllreduceWithSidebandsMultiOnStreams())
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: backend does not support grouped allreduce sideband bundles"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_));
            requestAbort();
            return false;
        }

        auto tensor_device = tensor->current_device();
        if (!tensor_device.has_value() ||
            devices_[static_cast<size_t>(device_index)].toLocalDeviceId() != *tensor_device)
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: tensor device mismatch"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " slot=" << device_index
                      << " expected=" << devices_[static_cast<size_t>(device_index)].toLocalDeviceId().toString()
                      << " actual=" << (tensor_device.has_value() ? tensor_device->toString() : "none"));
            requestAbort();
            return false;
        }

        void *buffer = tensor->gpu_data_ptr();
        if (!buffer)
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: null GPU buffer for slot "
                      << device_index << " stage="
                      << (stage_name.empty() ? "(none)" : stage_name));
            requestAbort();
            return false;
        }

        const size_t effective_count = (count > 0) ? count : tensor->numel();
        if (effective_count == 0)
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: zero allreduce count"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            requestAbort();
            return false;
        }

        for (const auto &sideband : sidebands)
        {
            if (sideband.element_count == 0)
            {
                LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: zero sideband element_count"
                          << " sideband=" << (sideband.name.empty() ? "(unnamed)" : sideband.name)
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                requestAbort();
                return false;
            }
            if (sideband.root_device_index < 0 || sideband.root_device_index >= degree())
            {
                LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: invalid sideband root_device_index="
                          << sideband.root_device_index << " degree=" << degree()
                          << " sideband=" << (sideband.name.empty() ? "(unnamed)" : sideband.name)
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                requestAbort();
                return false;
            }
        }

        CollectiveDataType dtype = tensorDTypeToCollective(tensor);
        const std::string effective_precision =
            precision.empty() ? std::string(kDefaultAllreducePrecision) : precision;
        const bool use_fp16_allreduce =
            effective_precision == "fp16" &&
            dtype == CollectiveDataType::FLOAT32 &&
            batchInvariantAllreduceDecisionElements(
                effective_count, tensor->cols()) >=
                debugEnv().allreduce_fp16_min_elements;

        void *collective_buffer = buffer;
        CollectiveDataType collective_dtype = dtype;
        const int ordinal = devices_[static_cast<size_t>(device_index)].device_ordinal;

        if (use_fp16_allreduce)
        {
            collective_buffer = requireReservedFp16Scratch(
                device_index,
                effective_count,
                stage_name,
                "LocalTPContext::allreduceWithSidebandsOnStream");
            collective_dtype = CollectiveDataType::FLOAT16;
            bool cast_ok = false;
#ifdef HAVE_CUDA
            if (device_group_.allCUDA())
            {
                cast_ok = (cudaCastFP32ToFP16(
                               static_cast<const float *>(buffer),
                               collective_buffer,
                               effective_count,
                               ordinal,
                               static_cast<cudaStream_t>(producer_stream)) == 0);
            }
#endif
#ifdef HAVE_ROCM
            if (device_group_.allROCm())
            {
                cast_ok = (rocmCastFP32ToFP16(
                               static_cast<const float *>(buffer),
                               collective_buffer,
                               effective_count,
                               ordinal,
                               producer_stream) == 0);
            }
#endif
            if (!cast_ok)
            {
                LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: FP32->FP16 cast failed"
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                requestAbort();
                return false;
            }
        }

        const bool grouped_ok = allreduceGroupedOnExplicitStreams(
            collective_buffer,
            effective_count,
            collective_dtype,
            device_index,
            producer_stream,
            stage_name,
            effective_precision,
            &sidebands);
        if (!grouped_ok)
        {
            LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: grouped allreduce sideband bundle failed"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " error=" << (backend_impl_ ? backend_impl_->lastError() : "missing backend"));
            requestAbort();
            return false;
        }

        if (use_fp16_allreduce)
        {
            bool back_ok = false;
#ifdef HAVE_CUDA
            if (device_group_.allCUDA())
            {
                back_ok = (cudaCastFP16ToFP32(
                               collective_buffer,
                               static_cast<float *>(buffer),
                               effective_count,
                               ordinal,
                               static_cast<cudaStream_t>(producer_stream)) == 0);
            }
#endif
#ifdef HAVE_ROCM
            if (device_group_.allROCm())
            {
                back_ok = (rocmCastFP16ToFP32(
                               collective_buffer,
                               static_cast<float *>(buffer),
                               effective_count,
                               ordinal,
                               producer_stream) == 0);
            }
#endif
            if (!back_ok)
            {
                LOG_ERROR("LocalTPContext::allreduceWithSidebandsOnStream: FP16->FP32 cast-back failed"
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                requestAbort();
                return false;
            }
        }

        recordLocalTPRuntimeAllreduce(
            device_group_, backend_, devices_[static_cast<size_t>(device_index)].toLocalDeviceId(),
            stage_name, static_cast<size_t>(degree()), effective_count,
            collective_dtype, "on_stream_grouped_with_sidebands", effective_precision);
        recordLocalTPRuntimeGroupedSidebands(
            device_group_, backend_, devices_[static_cast<size_t>(device_index)].toLocalDeviceId(),
            stage_name, static_cast<size_t>(degree()), device_index, sidebands);

        TransferEngine::publishDeviceWrite(
            tensor,
            devices_[static_cast<size_t>(device_index)].toLocalDeviceId(),
            producer_stream);
        return true;
    }

    bool LocalTPContext::collectiveSidebandOnStream(
        const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
        int device_index,
        void *producer_stream,
        const std::string &anchor_stage_name)
    {
        return collectiveSidebandSpanOnStream(
            sidebands,
            device_index,
            producer_stream,
            anchor_stage_name);
    }

    bool LocalTPContext::collectiveSidebandSpanOnStream(
        std::span<const LocalTPCollectiveSidebandBuffer> sidebands,
        int device_index,
        void *producer_stream,
        const std::string &anchor_stage_name)
    {
        if (!producer_stream)
            throw std::invalid_argument("LocalTPContext::collectiveSidebandOnStream requires a non-null GPU stream");

        if (sidebands.empty())
            return true;

        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: invalid device_index="
                      << device_index << " degree=" << degree()
                      << " anchor=" << (anchor_stage_name.empty() ? "(none)" : anchor_stage_name));
            return false;
        }

        if (degree() <= 1)
            return true;

        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: backend is not initialized"
                      << " anchor=" << (anchor_stage_name.empty() ? "(none)" : anchor_stage_name));
            return false;
        }

        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
        {
            LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: only homogeneous NCCL/RCCL domains are supported"
                      << " anchor=" << (anchor_stage_name.empty() ? "(none)" : anchor_stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_));
            return false;
        }

        if (!backend_impl_->isMultiGpuSingleProcess())
        {
            LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: backend is not in multi-GPU single-process mode"
                      << " anchor=" << (anchor_stage_name.empty() ? "(none)" : anchor_stage_name));
            return false;
        }

        auto fail_backend = [&](const LocalTPCollectiveSidebandBuffer &sideband,
                                const std::string &op_name) -> bool
        {
            LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: " << op_name
                      << " failed"
                      << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name)
                      << " kind=" << toString(sideband.kind)
                      << " anchor=" << (anchor_stage_name.empty() ? "(none)" : anchor_stage_name)
                      << " backend_error=" << backend_impl_->lastError());
            requestAbort();
            return false;
        };

        auto record_sideband_runtime = [&](
                                           const LocalTPCollectiveSidebandBuffer &sideband,
                                           const char *backend_primitive)
        {
            if (!PerfStatsCollector::isDomainEnabled(
                    "tp_allreduce_runtime"))
                return;

            const size_t element_bytes = collectiveDataTypeBytes(sideband.dtype);
            const size_t local_bytes = sideband.element_count * element_bytes;
            const size_t result_bytes =
                sidebandResultElements(
                    sideband.kind,
                    sideband.element_count,
                    static_cast<size_t>(degree())) *
                element_bytes;
            const std::string device_label =
                (device_index >= 0 && device_index < static_cast<int>(devices_.size()))
                    ? devices_[static_cast<size_t>(device_index)].toLocalDeviceId().toString()
                    : std::string{};

            PerfStatsCollector::Tags tags{
                {"stage", anchor_stage_name.empty() ? "unnamed" : anchor_stage_name},
                {"anchor_stage", anchor_stage_name.empty() ? "unnamed" : anchor_stage_name},
                {"anchor_collective", "allreduce"},
                {"backend", collectiveBackendTypeToString(backend_)},
                {"scope", "rank_local"},
                {"degree", std::to_string(degree())},
                {"sideband", sideband.name.empty() ? "unnamed" : sideband.name},
                {"kind", toString(sideband.kind)},
                {"dtype", collectiveDataTypeName(sideband.dtype)},
                {"element_bytes", std::to_string(element_bytes)},
                {"elements", std::to_string(sideband.element_count)},
                {"result_elements", std::to_string(sidebandResultElements(
                                        sideband.kind,
                                        sideband.element_count,
                                        static_cast<size_t>(degree())))},
                {"root_device_index", std::to_string(sideband.root_device_index)},
                {"device_index", std::to_string(device_index)},
                {"backend_primitive", backend_primitive ? backend_primitive : "unknown"},
                {"path", "collective_sideband_on_stream"},
                {"launch_relation", "same_stream_after_anchor"},
                {"fused_with_anchor", "false"},
                {"physical_fusion", "separate_backend_collective"},
                {"homogeneous", device_group_.is_homogeneous ? "true" : "false"}};

            PerfStatsCollector::addCounter(
                "tp_allreduce_runtime",
                "sideband_backend_collective_calls",
                1.0,
                {},
                device_label,
                tags);
            PerfStatsCollector::addCounter(
                "tp_allreduce_runtime",
                "sideband_separate_backend_collective_calls",
                1.0,
                {},
                device_label,
                tags);

            PerfStatsCollector::Tags local_byte_tags = tags;
            PerfStatsCollector::addCounter(
                "tp_allreduce_runtime",
                "sideband_local_bytes",
                static_cast<double>(local_bytes),
                {},
                device_label,
                std::move(local_byte_tags));

            PerfStatsCollector::Tags result_byte_tags = tags;
            PerfStatsCollector::addCounter(
                "tp_allreduce_runtime",
                "sideband_result_bytes",
                static_cast<double>(result_bytes),
                {},
                device_label,
                std::move(result_byte_tags));
        };

        for (const auto &sideband : sidebands)
        {
            if (sideband.element_count == 0)
            {
                LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: zero element_count"
                          << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name)
                          << " anchor=" << (anchor_stage_name.empty() ? "(none)" : anchor_stage_name));
                return false;
            }
            if (sideband.root_device_index < 0 || sideband.root_device_index >= degree())
            {
                LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: invalid root_device_index="
                          << sideband.root_device_index << " degree=" << degree()
                          << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name)
                          << " anchor=" << (anchor_stage_name.empty() ? "(none)" : anchor_stage_name));
                return false;
            }

            switch (sideband.kind)
            {
            case LocalTPCollectiveSidebandKind::AllreduceSum:
                if (!sideband.recv_buffer)
                {
                    LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: AllreduceSum requires recv_buffer"
                              << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name));
                    return false;
                }
                if (sideband.send_buffer && sideband.send_buffer != sideband.recv_buffer)
                {
                    LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: AllreduceSum is currently in-place only"
                              << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name));
                    return false;
                }
                if (!backend_impl_->allreduceSingleDeviceOnStream(
                        sideband.recv_buffer,
                        sideband.element_count,
                        sideband.dtype,
                        CollectiveOp::ALLREDUCE_SUM,
                        device_index,
                        producer_stream))
                    return fail_backend(sideband, "allreduce sideband");
                record_sideband_runtime(sideband, "allreduceSingleDeviceOnStream");
                break;

            case LocalTPCollectiveSidebandKind::Allgather:
                if (!sideband.send_buffer || !sideband.recv_buffer)
                {
                    LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: Allgather requires send_buffer and recv_buffer"
                              << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name));
                    return false;
                }
                if (!backend_impl_->supportsAllgatherSingleDeviceOnStream())
                {
                    LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: backend does not support on-stream allgather sidebands"
                              << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name)
                              << " backend=" << collectiveBackendTypeToString(backend_));
                    return false;
                }
                if (!backend_impl_->allgatherSingleDeviceOnStream(
                        sideband.send_buffer,
                        sideband.recv_buffer,
                        sideband.element_count,
                        sideband.dtype,
                        device_index,
                        producer_stream))
                    return fail_backend(sideband, "allgather sideband");
                record_sideband_runtime(sideband, "allgatherSingleDeviceOnStream");
                break;

            case LocalTPCollectiveSidebandKind::Broadcast:
            {
                if (!sideband.recv_buffer)
                {
                    LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: Broadcast requires recv_buffer"
                              << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name));
                    return false;
                }
                if (!backend_impl_->supportsBroadcastSingleDeviceOnStream())
                {
                    LOG_ERROR("LocalTPContext::collectiveSidebandOnStream: backend does not support on-stream broadcast sidebands"
                              << " name=" << (sideband.name.empty() ? "(unnamed)" : sideband.name)
                              << " backend=" << collectiveBackendTypeToString(backend_));
                    return false;
                }
                const void *send_buffer = sideband.send_buffer ? sideband.send_buffer : sideband.recv_buffer;
                if (!backend_impl_->broadcastSingleDeviceOnStream(
                        send_buffer,
                        sideband.recv_buffer,
                        sideband.element_count,
                        sideband.dtype,
                        sideband.root_device_index,
                        device_index,
                        producer_stream))
                    return fail_backend(sideband, "broadcast sideband");
                record_sideband_runtime(sideband, "broadcastSingleDeviceOnStream");
                break;
            }
            }
        }

        return true;
    }

    bool LocalTPContext::collectiveSidebandsMultiOnStreams(
        const std::vector<std::vector<LocalTPCollectiveSidebandBuffer>>
            &participant_sidebands,
        const std::vector<void *> &producer_streams,
        const std::string &publication_name)
    {
        auto fail = [&](const std::string &reason) -> bool
        {
            LOG_ERROR("LocalTPContext::collectiveSidebandsMultiOnStreams: "
                      << reason
                      << " publication="
                      << (publication_name.empty() ? "(none)"
                                                   : publication_name));
            requestAbort();
            return false;
        };

        if (degree() <= 1)
            return participant_sidebands.empty() ||
                   participant_sidebands.front().empty();
        if (!backend_initialized_ || !backend_impl_)
            return fail("backend is not initialized");
        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
        {
            return fail(
                std::string("requires a homogeneous NCCL/RCCL backend, got ") +
                collectiveBackendTypeToString(backend_));
        }
        if (!backend_impl_->isMultiGpuSingleProcess() ||
            !backend_impl_->supportsCollectiveSidebandsMultiOnStreams())
        {
            return fail(
                "backend lacks required anchor-free grouped sideband support");
        }
        if (participant_sidebands.size() != static_cast<size_t>(degree()) ||
            producer_streams.size() != static_cast<size_t>(degree()))
        {
            return fail(
                "participant descriptor/stream count does not match LocalTP degree");
        }

        const size_t sideband_count = participant_sidebands.front().size();
        if (sideband_count == 0)
            return fail("publication bundle is empty");
        for (int participant_index = 0;
             participant_index < degree();
             ++participant_index)
        {
            const size_t slot = static_cast<size_t>(participant_index);
            if (!producer_streams[slot])
            {
                throw std::invalid_argument(
                    "LocalTPContext::collectiveSidebandsMultiOnStreams requires non-null GPU streams");
            }
            if (participant_sidebands[slot].size() != sideband_count)
            {
                return fail(
                    "sideband count mismatch for participant " +
                    std::to_string(participant_index));
            }
        }

        /*
         * Lower the participant-major semantic matrix into the sideband-major
         * backend matrix consumed by NCCL/RCCL. Every descriptor is checked
         * against participant zero before any backend operation is submitted.
         * This prevents a malformed rank transaction from entering only a
         * prefix of the collective sequence.
         */
        std::vector<CollectiveSidebandMultiOnStreamsOp> backend_sidebands;
        backend_sidebands.reserve(sideband_count);
        for (size_t sideband_index = 0;
             sideband_index < sideband_count;
             ++sideband_index)
        {
            const LocalTPCollectiveSidebandBuffer &reference =
                participant_sidebands.front()[sideband_index];
            if (reference.element_count == 0 ||
                reference.root_device_index < 0 ||
                reference.root_device_index >= degree())
            {
                return fail(
                    "invalid reference descriptor at sideband " +
                    std::to_string(sideband_index));
            }

            CollectiveSidebandMultiOnStreamsOp backend_sideband;
            backend_sideband.kind = toBackendSidebandOp(reference.kind);
            backend_sideband.count = reference.element_count;
            backend_sideband.dtype = reference.dtype;
            backend_sideband.root = reference.root_device_index;
            backend_sideband.recv_buffers.assign(
                static_cast<size_t>(degree()), nullptr);
            if (reference.kind == LocalTPCollectiveSidebandKind::Allgather ||
                reference.kind == LocalTPCollectiveSidebandKind::Broadcast)
            {
                backend_sideband.send_buffers.assign(
                    static_cast<size_t>(degree()), nullptr);
            }

            for (int participant_index = 0;
                 participant_index < degree();
                 ++participant_index)
            {
                const size_t slot = static_cast<size_t>(participant_index);
                const LocalTPCollectiveSidebandBuffer &participant =
                    participant_sidebands[slot][sideband_index];
                if (participant.kind != reference.kind ||
                    participant.element_count != reference.element_count ||
                    participant.dtype != reference.dtype ||
                    participant.root_device_index !=
                        reference.root_device_index ||
                    participant.name != reference.name)
                {
                    return fail(
                        "sideband descriptor mismatch at participant " +
                        std::to_string(participant_index) + " sideband " +
                        std::to_string(sideband_index));
                }
                if (!participant.recv_buffer)
                {
                    return fail(
                        "missing receive buffer at participant " +
                        std::to_string(participant_index) + " sideband " +
                        std::to_string(sideband_index));
                }

                backend_sideband.recv_buffers[slot] =
                    participant.recv_buffer;
                switch (participant.kind)
                {
                case LocalTPCollectiveSidebandKind::AllreduceSum:
                    if (participant.send_buffer &&
                        participant.send_buffer != participant.recv_buffer)
                    {
                        return fail(
                            "AllreduceSum sideband must be in-place");
                    }
                    break;
                case LocalTPCollectiveSidebandKind::Allgather:
                    if (!participant.send_buffer)
                    {
                        return fail(
                            "Allgather sideband is missing a send buffer");
                    }
                    backend_sideband.send_buffers[slot] =
                        participant.send_buffer;
                    break;
                case LocalTPCollectiveSidebandKind::Broadcast:
                    backend_sideband.send_buffers[slot] =
                        participant.send_buffer
                            ? participant.send_buffer
                            : participant.recv_buffer;
                    break;
                }
            }
            backend_sidebands.push_back(std::move(backend_sideband));
        }

        if (!backend_impl_->collectiveSidebandsMultiOnStreams(
                backend_sidebands, producer_streams))
        {
            return fail(
                std::string("grouped backend launch failed: ") +
                backend_impl_->lastError());
        }

        for (int participant_index = 0;
             participant_index < degree();
             ++participant_index)
        {
            recordLocalTPRuntimeGroupedSidebands(
                device_group_,
                backend_,
                devices_[static_cast<size_t>(participant_index)]
                    .toLocalDeviceId(),
                publication_name,
                static_cast<size_t>(degree()),
                participant_index,
                participant_sidebands[static_cast<size_t>(participant_index)]);
        }
        PerfStatsCollector::addCounter(
            "tp_collective_runtime",
            "sideband_only_multi_stream_groups",
            1.0,
            {},
            "localtp",
            {{"publication",
              publication_name.empty() ? "unnamed" : publication_name},
             {"backend", collectiveBackendTypeToString(backend_)},
             {"degree", std::to_string(degree())},
             {"sideband_count", std::to_string(sideband_count)},
             {"host_rendezvous", "false"},
             {"device_completion_wait", "false"}});
        return true;
    }

    bool LocalTPContext::reduceRawOnStream(
        const void *local_send,
        void *root_recv,
        size_t count,
        CollectiveDataType dtype,
        CollectiveOp op,
        int root_device_index,
        int device_index,
        void *producer_stream,
        const std::string &stage_name)
    {
        if (!local_send || !root_recv || count == 0)
        {
            LOG_ERROR("LocalTPContext::reduceRawOnStream: invalid buffer/count"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "LocalTPContext::reduceRawOnStream requires a non-null GPU stream");
        }
        if (device_index < 0 || device_index >= degree() ||
            root_device_index < 0 || root_device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::reduceRawOnStream: invalid participant/root"
                      << " device_index=" << device_index
                      << " root_device_index=" << root_device_index
                      << " degree=" << degree()
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (degree() <= 1 || !backend_initialized_ || !backend_impl_ ||
            (backend_ != CollectiveBackendType::NCCL &&
             backend_ != CollectiveBackendType::RCCL) ||
            !backend_impl_->isMultiGpuSingleProcess() ||
            !backend_impl_->supportsReduceSingleDeviceOnStream())
        {
            LOG_ERROR("LocalTPContext::reduceRawOnStream: homogeneous native rooted reduce is unavailable"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " degree=" << degree());
            return false;
        }

        if (!backend_impl_->reduceSingleDeviceOnStream(
                local_send,
                root_recv,
                count,
                dtype,
                op,
                root_device_index,
                device_index,
                producer_stream))
        {
            LOG_ERROR("LocalTPContext::reduceRawOnStream: native rooted reduce failed"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend_error=" << backend_impl_->lastError());
            requestAbort();
            return false;
        }

        recordLocalTPRuntimeRawRootedCollective(
            device_group_,
            backend_,
            devices_[static_cast<size_t>(device_index)].toLocalDeviceId(),
            stage_name,
            static_cast<size_t>(degree()),
            device_index,
            root_device_index,
            count,
            dtype,
            "reduce",
            isGraphCaptureActive());
        return true;
    }

    bool LocalTPContext::broadcastRawOnStream(
        const void *root_send,
        void *local_recv,
        size_t count,
        CollectiveDataType dtype,
        int root_device_index,
        int device_index,
        void *producer_stream,
        const std::string &stage_name)
    {
        if (!root_send || !local_recv || count == 0)
        {
            LOG_ERROR("LocalTPContext::broadcastRawOnStream: invalid buffer/count"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (!producer_stream)
        {
            throw std::invalid_argument(
                "LocalTPContext::broadcastRawOnStream requires a non-null GPU stream");
        }
        if (device_index < 0 || device_index >= degree() ||
            root_device_index < 0 || root_device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::broadcastRawOnStream: invalid participant/root"
                      << " device_index=" << device_index
                      << " root_device_index=" << root_device_index
                      << " degree=" << degree()
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (degree() <= 1 || !backend_initialized_ || !backend_impl_ ||
            (backend_ != CollectiveBackendType::NCCL &&
             backend_ != CollectiveBackendType::RCCL) ||
            !backend_impl_->isMultiGpuSingleProcess() ||
            !backend_impl_->supportsBroadcastSingleDeviceOnStream())
        {
            LOG_ERROR("LocalTPContext::broadcastRawOnStream: homogeneous native broadcast is unavailable"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " degree=" << degree());
            return false;
        }

        if (!backend_impl_->broadcastSingleDeviceOnStream(
                root_send,
                local_recv,
                count,
                dtype,
                root_device_index,
                device_index,
                producer_stream))
        {
            LOG_ERROR("LocalTPContext::broadcastRawOnStream: native rooted broadcast failed"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend_error=" << backend_impl_->lastError());
            requestAbort();
            return false;
        }

        recordLocalTPRuntimeRawRootedCollective(
            device_group_,
            backend_,
            devices_[static_cast<size_t>(device_index)].toLocalDeviceId(),
            stage_name,
            static_cast<size_t>(degree()),
            device_index,
            root_device_index,
            count,
            dtype,
            "broadcast",
            isGraphCaptureActive());
        return true;
    }

    bool LocalTPContext::allgatherRawOnStream(
        const void *local_send,
        void *full_recv,
        size_t send_count,
        CollectiveDataType dtype,
        int device_index,
        void *producer_stream,
        const std::string &stage_name)
    {
        if (!local_send || !full_recv)
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: null buffer"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (send_count == 0)
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: zero send_count"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: invalid device_index="
                      << device_index << " degree=" << degree()
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (!producer_stream)
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: explicit non-null producer stream is required"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " device_index=" << device_index);
            return false;
        }

        if (degree() == 1)
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: single-device raw allgather is not a valid handoff"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }

        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: backend is not initialized"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }

        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: only homogeneous NCCL/RCCL domains are supported"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_));
            return false;
        }

        if (!backend_impl_->isMultiGpuSingleProcess())
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: backend does not support multi-GPU single-process allgather"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }

        if (!backend_impl_->supportsAllgatherSingleDeviceOnStream())
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: backend does not support native allgather on the producer stream"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_));
            return false;
        }

        /*
         * There is deliberately one GPU raw-allgather path.  NCCL/RCCL enqueue
         * the native collective directly on each participant's producer stream,
         * which provides producer -> collective -> consumer ordering in eager
         * execution and records that same ordering into a captured graph.  A
         * host rendezvous would add contention and an allreduce-based emulation
         * would multiply traffic for the large LLEP expert payload.
         */
        if (!backend_impl_->allgatherSingleDeviceOnStream(
                local_send,
                full_recv,
                send_count,
                dtype,
                device_index,
                producer_stream))
        {
            LOG_ERROR("LocalTPContext::allgatherRawOnStream: native on-stream allgather failed"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " backend_error=" << backend_impl_->lastError());
            requestAbort();
            return false;
        }

        recordLocalTPRuntimeRawAllgather(
            device_group_,
            backend_,
            devices_[static_cast<size_t>(device_index)].toLocalDeviceId(),
            stage_name,
            static_cast<size_t>(degree()),
            device_index,
            send_count,
            dtype,
            isGraphCaptureActive());
        return true;
    }

    bool LocalTPContext::groupedP2PRawOnStream(
        const std::vector<CollectiveP2POp> &ops,
        int device_index,
        void *producer_stream,
        const std::string &stage_name)
    {
        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: invalid device_index="
                      << device_index << " degree=" << degree()
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (!producer_stream)
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: explicit non-null producer stream is required"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " device_index=" << device_index);
            return false;
        }
        if (ops.empty())
            return true;

        if (degree() == 1)
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: single-device raw P2P is not a valid handoff"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: backend is not initialized"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: only homogeneous NCCL/RCCL domains are supported"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_));
            return false;
        }
        if (!backend_impl_->isMultiGpuSingleProcess())
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: backend does not support multi-GPU single-process P2P"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            return false;
        }
        if (!backend_impl_->supportsGroupedP2PSingleDeviceOnStream())
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: backend cannot capture grouped raw P2P on explicit stream"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend=" << collectiveBackendTypeToString(backend_));
            return false;
        }

        for (const auto &op : ops)
        {
            if (op.count == 0)
            {
                LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: zero-count P2P op"
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                return false;
            }
            if (op.peer < 0 || op.peer >= degree() || op.peer == device_index)
            {
                LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: invalid peer="
                          << op.peer << " device_index=" << device_index
                          << " degree=" << degree()
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                return false;
            }
            if (op.kind == CollectiveP2POpKind::Send && !op.send_buffer)
            {
                LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: null send buffer"
                          << " peer=" << op.peer
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                return false;
            }
            if (op.kind == CollectiveP2POpKind::Recv && !op.recv_buffer)
            {
                LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: null recv buffer"
                          << " peer=" << op.peer
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                return false;
            }
        }

        if (!backend_impl_->groupedP2PSingleDeviceOnStream(
                ops,
                device_index,
                producer_stream))
        {
            LOG_ERROR("LocalTPContext::groupedP2PRawOnStream: grouped on-stream P2P failed"
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " backend_error=" << backend_impl_->lastError());
            return false;
        }
        return true;
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

        throw std::runtime_error(
            "LocalTPContext::reduceScatter has no streamless GPU contract; "
            "use an explicit-stream collective implementation");
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
        if (!backend_initialized_ || !backend_impl_)
        {
            LOG_ERROR(
                "LocalTPContext::broadcast: multi-device broadcast requires "
                "an initialized collective backend");
            return false;
        }

        const GlobalDeviceAddress &source_device = devices_[source_device_index];
        DeviceId src_device_id = source_device.toLocalDeviceId();

        LOG_DEBUG("LocalTPContext::broadcast: Broadcasting from device "
                  << source_device_index << " (" << source_device.toString()
                  << ") to " << degree() << " devices");

        if (compute_streams_.size() != devices_.size() ||
            !compute_streams_[source_device_index])
        {
            LOG_ERROR(
                "LocalTPContext::broadcast: missing source compute stream for slot "
                << source_device_index);
            return false;
        }
        TransferEngine::prepareDeviceInput(
            tensor,
            src_device_id,
            compute_streams_[source_device_index]);
        // For homogeneous backends (NCCL/RCCL), point-to-point copies use the
        // backend's native P2P transport. Cross-vendor domains are explicitly
        // heterogeneous and are host-staged by TransferEngine.
        for (int i = 0; i < degree(); ++i)
        {
            if (i == source_device_index)
            {
                continue; // Skip source device
            }

            DeviceId dst_device_id = devices_[i].toLocalDeviceId();

            LOG_DEBUG("LocalTPContext::broadcast: " << src_device_id.toString()
                                                    << " → " << dst_device_id.toString());

            auto result = TransferEngine::instance().transferActivation(
                tensor,
                dst_device_id);
            if (!result.success)
            {
                LOG_ERROR("LocalTPContext::broadcast: Transfer failed from "
                          << src_device_id.toString() << " to "
                          << dst_device_id.toString() << ": "
                          << result.error);
                return false;
            }
        }

        LOG_DEBUG("LocalTPContext::broadcast: Complete, tensor on all "
                  << degree() << " devices");
        return true;
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
        if (!compute_streams.empty() &&
            compute_streams.size() != devices_.size())
        {
            throw std::invalid_argument(
                "LocalTPContext::setComputeStreams requires one stream per device");
        }

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTEXT] event=localtp_set_compute_streams"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " backend_impl=" << static_cast<const void *>(backend_impl_.get())
                      << " stream_count=" << compute_streams.size());
            for (size_t i = 0; i < compute_streams.size(); ++i)
            {
                LOG_DEBUG("[TP_COLLECTIVE_CONTEXT] event=localtp_compute_stream"
                          << " context_id=" << context_id_
                          << " slot=" << i
                          << " stream=" << compute_streams[i]
                          << " device=" << (i < devices_.size() ? devices_[i].toString() : std::string("(unknown)")));
            }
        }

        if (backend_initialized_ && backend_impl_)
        {
            backend_impl_->setComputeStreams(compute_streams);
        }
        compute_streams_ = compute_streams;
    }

    bool LocalTPContext::allreduceGroupedOnExplicitStreams(void *buffer,
                                                           size_t effective_count,
                                                           CollectiveDataType dtype,
                                                           int device_index,
                                                           void *stream,
                                                           const std::string &stage_name,
                                                           const std::string &precision,
                                                           const std::vector<LocalTPCollectiveSidebandBuffer> *sidebands)
    {
        if (degree() <= 1)
            return true;

        const size_t sideband_count = sidebands ? sidebands->size() : 0;
        const bool graph_capture_active = isGraphCaptureActive();

        if (!buffer || !stream)
        {
            LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: null "
                      << (!buffer ? "buffer" : "stream")
                      << " slot=" << device_index
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
            requestAbort();
            return false;
        }

        if (device_index < 0 || device_index >= degree())
        {
            LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: invalid slot "
                      << device_index << " degree=" << degree());
            requestAbort();
            return false;
        }

        std::unique_lock<std::mutex> lock(grouped_onstream_allreduce_mutex_);
        if (abort_requested_.load(std::memory_order_acquire))
            return false;

        auto reset_generation_state = [&]()
        {
            grouped_onstream_allreduce_arrivals_ = 0;
            grouped_onstream_allreduce_departures_ = 0;
            grouped_onstream_allreduce_ready_ = false;
            grouped_onstream_allreduce_result_ = false;
            grouped_onstream_allreduce_count_ = 0;
            grouped_onstream_allreduce_dtype_ = -1;
            grouped_onstream_allreduce_stage_.clear();
            grouped_onstream_allreduce_precision_.clear();
            grouped_onstream_allreduce_buffers_.clear();
            grouped_onstream_allreduce_streams_.clear();
            grouped_onstream_allreduce_seen_.clear();
            grouped_onstream_allreduce_graph_capture_active_ = false;
            grouped_onstream_allreduce_sideband_count_ = 0;
            grouped_onstream_allreduce_reference_sidebands_.clear();
            grouped_onstream_allreduce_sidebands_.clear();
            grouped_onstream_allreduce_generation_++;
        };

        while (grouped_onstream_allreduce_arrivals_ >= degree() &&
               grouped_onstream_allreduce_departures_ > 0 &&
               !abort_requested_.load(std::memory_order_acquire))
        {
            grouped_onstream_allreduce_cv_.wait(lock);
        }
        if (abort_requested_.load(std::memory_order_acquire))
            return false;

        const uint64_t my_generation = grouped_onstream_allreduce_generation_;
        const int arrival_order = grouped_onstream_allreduce_arrivals_++;
        auto fail_generation = [&](const std::string &error)
        {
            grouped_onstream_allreduce_ok_ = false;
            grouped_onstream_allreduce_result_ = false;
            grouped_onstream_allreduce_error_ = error;
            reset_generation_state();
            abort_requested_.store(true, std::memory_order_release);
            LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: " << error);
            lock.unlock();
            grouped_onstream_allreduce_cv_.notify_all();
        };

        auto depart_generation = [&](bool participant_result = true) -> bool
        {
            if (!participant_result)
                grouped_onstream_allreduce_result_ = false;
            const bool result = grouped_onstream_allreduce_ready_ &&
                                grouped_onstream_allreduce_result_ &&
                                participant_result &&
                                !abort_requested_.load(std::memory_order_acquire);
            grouped_onstream_allreduce_departures_++;
            if (grouped_onstream_allreduce_departures_ >= degree())
            {
                reset_generation_state();
                lock.unlock();
                grouped_onstream_allreduce_cv_.notify_all();
            }
            return result;
        };

        if (arrival_order >= degree())
        {
            fail_generation("arrival overflow stage=" +
                            (stage_name.empty() ? std::string("(none)") : stage_name));
            return false;
        }

        if (arrival_order == 0)
        {
            grouped_onstream_allreduce_ok_ = true;
            grouped_onstream_allreduce_ready_ = false;
            grouped_onstream_allreduce_result_ = false;
            grouped_onstream_allreduce_count_ = effective_count;
            grouped_onstream_allreduce_dtype_ = static_cast<int>(dtype);
            grouped_onstream_allreduce_stage_ = stage_name;
            grouped_onstream_allreduce_precision_ = precision;
            grouped_onstream_allreduce_error_.clear();
            grouped_onstream_allreduce_buffers_.assign(static_cast<size_t>(degree()), nullptr);
            grouped_onstream_allreduce_streams_.assign(static_cast<size_t>(degree()), nullptr);
            grouped_onstream_allreduce_seen_.assign(static_cast<size_t>(degree()), false);
            grouped_onstream_allreduce_graph_capture_active_ = graph_capture_active;
            grouped_onstream_allreduce_sideband_count_ = sideband_count;
            grouped_onstream_allreduce_reference_sidebands_ =
                sidebands ? *sidebands : std::vector<LocalTPCollectiveSidebandBuffer>{};
            grouped_onstream_allreduce_sidebands_.assign(
                static_cast<size_t>(degree()),
                std::vector<LocalTPCollectiveSidebandBuffer>{});
        }
        else
        {
            if (grouped_onstream_allreduce_stage_ != stage_name)
            {
                fail_generation("stage mismatch expected=" +
                                (grouped_onstream_allreduce_stage_.empty() ? std::string("(none)") : grouped_onstream_allreduce_stage_) +
                                " actual=" + (stage_name.empty() ? std::string("(none)") : stage_name));
                return false;
            }
            if (grouped_onstream_allreduce_count_ != effective_count)
            {
                fail_generation("count mismatch expected=" +
                                std::to_string(grouped_onstream_allreduce_count_) +
                                " actual=" + std::to_string(effective_count));
                return false;
            }
            if (grouped_onstream_allreduce_dtype_ != static_cast<int>(dtype))
            {
                fail_generation("dtype mismatch expected=" +
                                std::to_string(grouped_onstream_allreduce_dtype_) +
                                " actual=" + std::to_string(static_cast<int>(dtype)));
                return false;
            }
            if (grouped_onstream_allreduce_precision_ != precision)
            {
                fail_generation("precision mismatch expected=" +
                                (grouped_onstream_allreduce_precision_.empty() ? std::string("(default)") : grouped_onstream_allreduce_precision_) +
                                " actual=" + (precision.empty() ? std::string("(default)") : precision));
                return false;
            }
            if (grouped_onstream_allreduce_sideband_count_ != sideband_count)
            {
                fail_generation("sideband count mismatch expected=" +
                                std::to_string(grouped_onstream_allreduce_sideband_count_) +
                                " actual=" + std::to_string(sideband_count));
                return false;
            }
            if (grouped_onstream_allreduce_graph_capture_active_ != graph_capture_active)
            {
                fail_generation("graph capture state mismatch expected=" +
                                std::string(grouped_onstream_allreduce_graph_capture_active_ ? "active" : "inactive") +
                                " actual=" + (graph_capture_active ? "active" : "inactive") +
                                " stage=" + (stage_name.empty() ? std::string("(none)") : stage_name));
                return false;
            }
            for (size_t sideband_index = 0; sideband_index < sideband_count; ++sideband_index)
            {
                const auto &expected =
                    grouped_onstream_allreduce_reference_sidebands_[sideband_index];
                const auto &actual = (*sidebands)[sideband_index];
                if (expected.kind != actual.kind ||
                    expected.element_count != actual.element_count ||
                    expected.dtype != actual.dtype ||
                    expected.root_device_index != actual.root_device_index)
                {
                    fail_generation("sideband descriptor mismatch index=" +
                                    std::to_string(sideband_index) +
                                    " stage=" + (stage_name.empty() ? std::string("(none)") : stage_name));
                    return false;
                }
            }
        }

        if (grouped_onstream_allreduce_seen_[static_cast<size_t>(device_index)])
        {
            fail_generation("duplicate slot arrival slot=" + std::to_string(device_index) +
                            " stage=" + (stage_name.empty() ? std::string("(none)") : stage_name));
            return false;
        }

        grouped_onstream_allreduce_seen_[static_cast<size_t>(device_index)] = true;
        grouped_onstream_allreduce_buffers_[static_cast<size_t>(device_index)] = buffer;
        grouped_onstream_allreduce_streams_[static_cast<size_t>(device_index)] = stream;
        if (sideband_count > 0)
            grouped_onstream_allreduce_sidebands_[static_cast<size_t>(device_index)] = *sidebands;

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=localtp_grouped_onstream_arrival"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " generation=" << my_generation
                      << " slot=" << device_index
                      << " arrival_order=" << arrival_order
                      << " degree=" << degree()
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " count=" << effective_count
                      << " dtype=" << static_cast<int>(dtype)
                      << " precision=" << (precision.empty() ? "(default)" : precision)
                      << " sidebands=" << sideband_count
                      << " stream=" << stream
                      << " buffer=" << buffer);
        }

        if (arrival_order + 1 < degree())
        {
            auto ready = [&]()
            {
                return abort_requested_.load(std::memory_order_acquire) ||
                       grouped_onstream_allreduce_generation_ > my_generation ||
                       (grouped_onstream_allreduce_generation_ == my_generation &&
                        grouped_onstream_allreduce_ready_);
            };

            const int timeout_ms = collective_timeout_policy::effectiveCollectTimeoutMs(
                debugEnv().tp_collect_timeout_ms);
            bool completed = true;
            if (timeout_ms > 0)
            {
                completed = grouped_onstream_allreduce_cv_.wait_for(
                    lock,
                    std::chrono::milliseconds(timeout_ms),
                    ready);
            }
            else
            {
                grouped_onstream_allreduce_cv_.wait(lock, ready);
            }

            if (!completed)
            {
                fail_generation("timeout waiting for grouped on-stream allreduce peers stage=" +
                                (stage_name.empty() ? std::string("(none)") : stage_name) +
                                " arrivals=" + std::to_string(grouped_onstream_allreduce_arrivals_) +
                                " degree=" + std::to_string(degree()));
                return false;
            }

            if (grouped_onstream_allreduce_generation_ != my_generation &&
                !grouped_onstream_allreduce_ready_)
            {
                return false;
            }

            if (grouped_onstream_allreduce_graph_capture_active_)
            {
                if (backend_ == CollectiveBackendType::RCCL ||
                    grouped_onstream_allreduce_sideband_count_ > 0)
                {
                    /*
                     * The final arrival records one backend group spanning all
                     * participant streams for RCCL and for any anchor carrying
                     * sidebands. Earlier arrivals must only observe that grouped
                     * launch outcome. Enqueuing a participant-local anchor here
                     * would either duplicate RCCL work or omit the sideband
                     * operations from CUDA participant graphs.
                     */
                    return depart_generation();
                }

                const bool ready_to_enqueue =
                    grouped_onstream_allreduce_ready_ &&
                    grouped_onstream_allreduce_result_ &&
                    !abort_requested_.load(std::memory_order_acquire);
                lock.unlock();

                bool enqueue_ok = false;
                if (ready_to_enqueue && backend_impl_)
                {
                    enqueue_ok = backend_impl_->allreduceSingleDeviceOnStream(
                        buffer, effective_count, dtype, CollectiveOp::ALLREDUCE_SUM,
                        device_index, stream);
                }

                lock.lock();
                if (!enqueue_ok)
                {
                    grouped_onstream_allreduce_result_ = false;
                    grouped_onstream_allreduce_error_ =
                        backend_impl_ ? backend_impl_->lastError() : std::string("missing backend");
                    abort_requested_.store(true, std::memory_order_release);
                    LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: graph-captured participant launch failed"
                              << " backend=" << collectiveBackendTypeToString(backend_)
                              << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                              << " slot=" << device_index
                              << " error=" << grouped_onstream_allreduce_error_);
                }
                return depart_generation(enqueue_ok);
            }

            return depart_generation();
        }

        for (int i = 0; i < degree(); ++i)
        {
            if (i >= static_cast<int>(grouped_onstream_allreduce_seen_.size()) ||
                !grouped_onstream_allreduce_seen_[static_cast<size_t>(i)] ||
                !grouped_onstream_allreduce_buffers_[static_cast<size_t>(i)] ||
                !grouped_onstream_allreduce_streams_[static_cast<size_t>(i)] ||
                (grouped_onstream_allreduce_sideband_count_ > 0 &&
                 grouped_onstream_allreduce_sidebands_[static_cast<size_t>(i)].size() !=
                     grouped_onstream_allreduce_sideband_count_))
            {
                fail_generation("missing grouped on-stream participant slot=" +
                                std::to_string(i) +
                                " stage=" + (stage_name.empty() ? std::string("(none)") : stage_name));
                return false;
            }
        }

        if (grouped_onstream_allreduce_sideband_count_ > 0)
        {
            if (!backend_impl_ || !backend_impl_->supportsAllreduceWithSidebandsMultiOnStreams())
            {
                fail_generation(std::string("backend does not support grouped allreduce sideband bundles backend=") +
                                collectiveBackendTypeToString(backend_));
                return false;
            }
        }
        else if (!backend_impl_ ||
                 (backend_ == CollectiveBackendType::HETEROGENEOUS
                      ? !backend_impl_->supportsAllreduceMultiOnStreamsWithHostCompletionTicket()
                      : !backend_impl_->supportsAllreduceMultiOnStreams()))
        {
            if (!grouped_onstream_allreduce_graph_capture_active_ ||
                !backend_impl_ ||
                !backend_impl_->supportsAllreduceSingleDeviceOnStream())
            {
                fail_generation(std::string("backend does not support grouped explicit-stream allreduce backend=") +
                                collectiveBackendTypeToString(backend_));
                return false;
            }
        }

        /*
         * Lower participant-local semantic descriptors into one backend bundle
         * only after every LocalTP slot has arrived. The resulting vectors hold
         * one buffer address per device and are valid for both eager execution
         * and graph capture. Keeping one lowering path prevents capture from
         * silently exercising a weaker collective contract than production.
         */
        std::vector<CollectiveSidebandMultiOnStreamsOp> backend_sidebands;
        if (grouped_onstream_allreduce_sideband_count_ > 0)
        {
            backend_sidebands.reserve(grouped_onstream_allreduce_sideband_count_);
            for (size_t sideband_index = 0;
                 sideband_index < grouped_onstream_allreduce_sideband_count_;
                 ++sideband_index)
            {
                const auto &reference =
                    grouped_onstream_allreduce_reference_sidebands_[sideband_index];
                CollectiveSidebandMultiOnStreamsOp backend_sideband;
                backend_sideband.kind = toBackendSidebandOp(reference.kind);
                backend_sideband.count = reference.element_count;
                backend_sideband.dtype = reference.dtype;
                backend_sideband.root = reference.root_device_index;
                backend_sideband.recv_buffers.assign(static_cast<size_t>(degree()), nullptr);
                if (reference.kind == LocalTPCollectiveSidebandKind::Allgather ||
                    reference.kind == LocalTPCollectiveSidebandKind::Broadcast)
                {
                    backend_sideband.send_buffers.assign(static_cast<size_t>(degree()), nullptr);
                }

                for (int i = 0; i < degree(); ++i)
                {
                    const auto &participant =
                        grouped_onstream_allreduce_sidebands_[static_cast<size_t>(i)][sideband_index];
                    switch (participant.kind)
                    {
                    case LocalTPCollectiveSidebandKind::AllreduceSum:
                        if (!participant.recv_buffer)
                        {
                            fail_generation("AllreduceSum sideband missing recv buffer slot=" +
                                            std::to_string(i) + " sideband=" +
                                            std::to_string(sideband_index));
                            return false;
                        }
                        if (participant.send_buffer &&
                            participant.send_buffer != participant.recv_buffer)
                        {
                            fail_generation("AllreduceSum sideband is currently in-place only slot=" +
                                            std::to_string(i) + " sideband=" +
                                            std::to_string(sideband_index));
                            return false;
                        }
                        backend_sideband.recv_buffers[static_cast<size_t>(i)] =
                            participant.recv_buffer;
                        break;
                    case LocalTPCollectiveSidebandKind::Allgather:
                        if (!participant.send_buffer || !participant.recv_buffer)
                        {
                            fail_generation("Allgather sideband missing buffer slot=" +
                                            std::to_string(i) + " sideband=" +
                                            std::to_string(sideband_index));
                            return false;
                        }
                        backend_sideband.send_buffers[static_cast<size_t>(i)] =
                            participant.send_buffer;
                        backend_sideband.recv_buffers[static_cast<size_t>(i)] =
                            participant.recv_buffer;
                        break;
                    case LocalTPCollectiveSidebandKind::Broadcast:
                        if (!participant.recv_buffer)
                        {
                            fail_generation("Broadcast sideband missing recv buffer slot=" +
                                            std::to_string(i) + " sideband=" +
                                            std::to_string(sideband_index));
                            return false;
                        }
                        backend_sideband.send_buffers[static_cast<size_t>(i)] =
                            participant.send_buffer ? participant.send_buffer : participant.recv_buffer;
                        backend_sideband.recv_buffers[static_cast<size_t>(i)] =
                            participant.recv_buffer;
                        break;
                    }
                }
                backend_sidebands.push_back(std::move(backend_sideband));
            }
        }

        if (grouped_onstream_allreduce_graph_capture_active_)
        {
            if (backend_ == CollectiveBackendType::HETEROGENEOUS)
            {
                fail_generation(
                    "heterogeneous allreduce reached an active native graph capture; "
                    "the graph planner must expose it as an explicit segmented collective boundary");
                return false;
            }

            if (grouped_onstream_allreduce_sideband_count_ > 0)
            {
                /*
                 * CUDA and ROCm relaxed capture both permit the rendezvous owner
                 * to submit one NCCL/RCCL group across streams whose captures
                 * were begun by their participant workers. The backend group is
                 * the indivisible production operation: anchor first, followed
                 * by every sideband, with one group end publishing all nodes.
                 */
                const bool enqueue_ok =
                    backend_impl_->allreduceWithSidebandsMultiOnStreams(
                        grouped_onstream_allreduce_buffers_,
                        effective_count,
                        dtype,
                        CollectiveOp::ALLREDUCE_SUM,
                        backend_sidebands,
                        grouped_onstream_allreduce_streams_);
                grouped_onstream_allreduce_result_ = enqueue_ok;
                grouped_onstream_allreduce_ready_ = true;
                if (!enqueue_ok)
                {
                    grouped_onstream_allreduce_error_ =
                        backend_impl_ ? backend_impl_->lastError() : std::string("missing backend");
                    abort_requested_.store(true, std::memory_order_release);
                    LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: graph-captured grouped sideband bundle failed"
                              << " backend=" << collectiveBackendTypeToString(backend_)
                              << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                              << " sidebands=" << grouped_onstream_allreduce_sideband_count_
                              << " error=" << grouped_onstream_allreduce_error_);
                }
                grouped_onstream_allreduce_cv_.notify_all();
                return depart_generation(enqueue_ok);
            }

            if (backend_ == CollectiveBackendType::RCCL)
            {
                /*
                 * RCCL single-process graph capture must enqueue the
                 * participant streams through one grouped launch.  The
                 * previous participant-local branch issued independent
                 * rcclAllReduce calls from each worker thread; that can
                 * fail during HIP capture with an "unhandled cuda error"
                 * even though the same streams are graph-capturable when
                 * wrapped in rcclGroupStart/rcclGroupEnd.  The final
                 * arrival owns the complete stream/buffer set, so it can
                 * record the exact grouped operation that ordinary
                 * explicit-stream RCCL execution uses without falling back
                 * to non-captured execution.
                 */
                if (!backend_impl_ || !backend_impl_->supportsAllreduceMultiOnStreams())
                {
                    fail_generation(std::string("RCCL graph-captured grouped allreduce requires grouped explicit-stream support backend=") +
                                    collectiveBackendTypeToString(backend_));
                    return false;
                }

                const bool enqueue_ok = backend_impl_->allreduceMultiOnStreams(
                    grouped_onstream_allreduce_buffers_,
                    effective_count,
                    dtype,
                    CollectiveOp::ALLREDUCE_SUM,
                    grouped_onstream_allreduce_streams_);
                grouped_onstream_allreduce_result_ = enqueue_ok;
                grouped_onstream_allreduce_ready_ = true;
                if (!enqueue_ok)
                {
                    grouped_onstream_allreduce_error_ =
                        backend_impl_ ? backend_impl_->lastError() : std::string("missing backend");
                    abort_requested_.store(true, std::memory_order_release);
                    LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: graph-captured grouped RCCL launch failed"
                              << " backend=" << collectiveBackendTypeToString(backend_)
                              << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                              << " error=" << grouped_onstream_allreduce_error_);
                }
                grouped_onstream_allreduce_cv_.notify_all();
                return depart_generation(enqueue_ok);
            }

            if (!backend_impl_ || !backend_impl_->supportsAllreduceSingleDeviceOnStream())
            {
                fail_generation(std::string("backend does not support graph-captured participant-local allreduce backend=") +
                                collectiveBackendTypeToString(backend_));
                return false;
            }

            grouped_onstream_allreduce_result_ = true;
            grouped_onstream_allreduce_ready_ = true;
            grouped_onstream_allreduce_cv_.notify_all();
            lock.unlock();

            const bool enqueue_ok = backend_impl_->allreduceSingleDeviceOnStream(
                buffer, effective_count, dtype, CollectiveOp::ALLREDUCE_SUM,
                device_index, stream);

            lock.lock();
            if (!enqueue_ok)
            {
                grouped_onstream_allreduce_result_ = false;
                grouped_onstream_allreduce_error_ =
                    backend_impl_ ? backend_impl_->lastError() : std::string("missing backend");
                abort_requested_.store(true, std::memory_order_release);
                LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: graph-captured participant launch failed"
                          << " backend=" << collectiveBackendTypeToString(backend_)
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                          << " slot=" << device_index
                          << " error=" << grouped_onstream_allreduce_error_);
            }
            return depart_generation(enqueue_ok);
        }

        else if (!backend_impl_ ||
                 (backend_ == CollectiveBackendType::HETEROGENEOUS
                      ? !backend_impl_->supportsAllreduceMultiOnStreamsWithHostCompletionTicket()
                      : !backend_impl_->supportsAllreduceMultiOnStreams()))
        {
            fail_generation(std::string("backend does not support the required grouped explicit-stream completion authority backend=") +
                            collectiveBackendTypeToString(backend_));
            return false;
        }

        bool success = false;
        if (backend_ == CollectiveBackendType::HETEROGENEOUS)
        {
            if (grouped_onstream_allreduce_sideband_count_ > 0)
            {
                fail_generation(
                    "heterogeneous grouped sidebands require their own typed host-ticket transport");
                return false;
            }

            const auto ticket =
                backend_impl_->allreduceMultiOnStreamsWithHostCompletionTicket(
                    grouped_onstream_allreduce_buffers_,
                    effective_count,
                    dtype,
                    CollectiveOp::ALLREDUCE_SUM,
                    grouped_onstream_allreduce_streams_);
            if (ticket.has_value())
            {
                // This is the explicit heterogeneous segment boundary. Release
                // the LocalTP state mutex while the background worker progresses
                // producer-event -> D2H -> fixed-order reduce -> H2D. Waiting
                // threads remain parked on the generation predicate and cannot
                // consume the output until readiness is published below.
                const int timeout_ms =
                    collective_timeout_policy::effectiveCollectTimeoutMs(
                        debugEnv().tp_collect_timeout_ms);
                lock.unlock();
                success = backend_impl_->awaitHostCompletionTicket(
                    *ticket,
                    timeout_ms);
                lock.lock();
            }
        }
        else
        {
            success =
                grouped_onstream_allreduce_sideband_count_ > 0
                    ? backend_impl_->allreduceWithSidebandsMultiOnStreams(
                          grouped_onstream_allreduce_buffers_,
                          effective_count,
                          dtype,
                          CollectiveOp::ALLREDUCE_SUM,
                          backend_sidebands,
                          grouped_onstream_allreduce_streams_)
                    : backend_impl_->allreduceMultiOnStreams(
                          grouped_onstream_allreduce_buffers_,
                          effective_count,
                          dtype,
                          CollectiveOp::ALLREDUCE_SUM,
                          grouped_onstream_allreduce_streams_);
        }

        grouped_onstream_allreduce_result_ = success;
        grouped_onstream_allreduce_ready_ = true;
        if (!success)
        {
            grouped_onstream_allreduce_error_ =
                backend_impl_ ? backend_impl_->lastError() : std::string("missing backend");
            abort_requested_.store(true, std::memory_order_release);
            LOG_ERROR("LocalTPContext::allreduceGroupedOnExplicitStreams: backend launch failed"
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " error=" << grouped_onstream_allreduce_error_);
        }
        if (success && debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=localtp_grouped_onstream_enqueued"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " generation=" << my_generation
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " count=" << effective_count
                      << " dtype=" << static_cast<int>(dtype)
                      << " completion="
                      << (backend_ == CollectiveBackendType::HETEROGENEOUS
                              ? "host_observed_ticket"
                              : "device_stream"));
        }

        grouped_onstream_allreduce_cv_.notify_all();
        return depart_generation();
    }

    bool LocalTPContext::rendezvousOnStreamCollective(int device_index,
                                                      TensorBase *tensor,
                                                      const std::string &stage_name,
                                                      size_t effective_count,
                                                      CollectiveDataType dtype,
                                                      void *stream,
                                                      const std::string &precision)
    {
        if (degree() <= 1)
            return true;

        uint64_t sequence = 0;
        bool ok = true;
        std::string error;
        {
            std::unique_lock<std::mutex> lock(contract_trace_mutex_);
            if (onstream_sequence_by_slot_.size() != devices_.size())
            {
                onstream_sequence_by_slot_.assign(devices_.size(), 0);
            }

            if (device_index < 0 ||
                device_index >= static_cast<int>(onstream_sequence_by_slot_.size()))
            {
                LOG_ERROR("[TP_COLLECTIVE_CONTRACT] event=localtp_onstream_invalid_slot"
                          << " context_id=" << context_id_
                          << " context=" << static_cast<const void *>(this)
                          << " slot=" << device_index
                          << " degree=" << degree()
                          << " stage=" << (stage_name.empty() ? "(none)" : stage_name));
                return false;
            }

            sequence = onstream_sequence_by_slot_[static_cast<size_t>(device_index)]++;
            auto &entry = onstream_contracts_[sequence];
            const int dtype_value = static_cast<int>(dtype);
            if (!entry.initialized)
            {
                entry.initialized = true;
                entry.stage_name = stage_name;
                entry.count = effective_count;
                entry.dtype = dtype_value;
                entry.precision = precision;
                entry.ok = true;
            }
            else
            {
                if (entry.stage_name != stage_name)
                {
                    entry.ok = false;
                    entry.error = "stage mismatch expected=" + (entry.stage_name.empty() ? std::string("(none)") : entry.stage_name) +
                                  " actual=" + (stage_name.empty() ? std::string("(none)") : stage_name);
                }
                else if (entry.count != effective_count)
                {
                    entry.ok = false;
                    entry.error = "count mismatch expected=" + std::to_string(entry.count) +
                                  " actual=" + std::to_string(effective_count);
                }
                else if (entry.dtype != dtype_value)
                {
                    entry.ok = false;
                    entry.error = "dtype mismatch expected=" + std::to_string(entry.dtype) +
                                  " actual=" + std::to_string(dtype_value);
                }
            }

            if (device_index < 64)
            {
                const uint64_t bit = 1ull << static_cast<uint64_t>(device_index);
                if ((entry.seen_slots & bit) != 0)
                {
                    entry.ok = false;
                    entry.error = "duplicate slot arrival for sequence=" + std::to_string(sequence) +
                                  " slot=" + std::to_string(device_index);
                }
                entry.seen_slots |= bit;
            }
            ++entry.arrivals;

            if (entry.arrivals >= degree() || !entry.ok)
                contract_trace_cv_.notify_all();

            auto ready = [&]()
            {
                return abort_requested_.load(std::memory_order_acquire) ||
                       !entry.ok ||
                       entry.arrivals >= degree();
            };

            if (!ready())
            {
                const int timeout_ms = collective_timeout_policy::effectiveCollectTimeoutMs(
                    debugEnv().tp_collect_timeout_ms);
                if (timeout_ms > 0)
                {
                    const bool completed = contract_trace_cv_.wait_for(
                        lock,
                        std::chrono::milliseconds(timeout_ms),
                        ready);
                    if (!completed)
                    {
                        entry.ok = false;
                        entry.error = "timeout waiting for on-stream collective peers"
                                      " sequence=" +
                                      std::to_string(sequence) +
                                      " arrivals=" + std::to_string(entry.arrivals) +
                                      " degree=" + std::to_string(degree()) +
                                      " stage=" + (stage_name.empty() ? std::string("(none)") : stage_name);
                        contract_trace_cv_.notify_all();
                    }
                }
                else
                {
                    contract_trace_cv_.wait(lock, ready);
                }
            }

            ok = entry.ok && !abort_requested_.load(std::memory_order_acquire);
            error = entry.error;
            ++entry.departures;
            if (entry.departures >= entry.arrivals)
            {
                onstream_contracts_.erase(sequence);
                contract_trace_cv_.notify_all();
            }
        }

        if (debugEnv().tp_collective_contract_trace)
        {
            LOG_DEBUG("[TP_COLLECTIVE_CONTRACT] event=localtp_onstream_rendezvous"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " backend_impl=" << static_cast<const void *>(backend_impl_.get())
                      << " sequence=" << sequence
                      << " slot=" << device_index
                      << " degree=" << degree()
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " count=" << effective_count
                      << " dtype=" << static_cast<int>(dtype)
                      << " precision=" << (precision.empty() ? "(default)" : precision)
                      << " stream=" << stream
                      << " tensor=" << static_cast<void *>(tensor)
                      << " tensor_name=" << (tensor && !tensor->debugName().empty() ? tensor->debugName() : "(unnamed)")
                      << " gpu_ptr=" << (tensor ? tensor->gpu_data_ptr() : nullptr)
                      << " ok=" << (ok ? 1 : 0));
        }

        if (!ok)
        {
            LOG_ERROR("[TP_COLLECTIVE_CONTRACT] event=localtp_onstream_rendezvous_failed"
                      << " context_id=" << context_id_
                      << " context=" << static_cast<const void *>(this)
                      << " sequence=" << sequence
                      << " slot=" << device_index
                      << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                      << " reason=" << (error.empty() ? "abort requested" : error));
        }

        return ok;
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

            // Use HOST backend for exactly 1+1 case (host-staged cross-vendor transfer)
            if (num_cuda == 1 && num_rocm == 1)
            {
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
        LOG_DEBUG("LocalTPContext: Backend " << backend_impl_->name()
                                             << " initialized for " << degree() << " devices");

        if (backend_ == CollectiveBackendType::NCCL)
        {
            LOG_DEBUG("[LocalTPContext][NCCLReady] "
                      << "status=ready"
                      << " backend_impl=" << backend_impl_->name()
                      << " degree=" << degree()
                      << " multi_gpu_single_process=" << (backend_impl_->isMultiGpuSingleProcess() ? 1 : 0));
        }
        return true;
    }

    bool LocalTPContext::initializeGraphCaptureBoundaryDeviceWords(
        const std::shared_ptr<PhysicalMemoryAuthority> &memory_authority)
    {
        if (degree() <= 1 ||
            (backend_ != CollectiveBackendType::NCCL &&
             backend_ != CollectiveBackendType::RCCL))
        {
            return true;
        }
        if (!device_group_.allCUDA() && !device_group_.allROCm())
            return false;
        if (!memory_authority)
        {
            LOG_ERROR("LocalTPContext: native graph-boundary storage has no "
                      "physical-memory authority");
            return false;
        }
        if (graph_capture_boundary_device_words_.size() != devices_.size() ||
            graph_capture_boundary_memory_leases_.size() != devices_.size())
        {
            LOG_ERROR("LocalTPContext: graph-boundary storage metadata has "
                      "invalid participant geometry");
            return false;
        }

        constexpr size_t kControlBytes = sizeof(int32_t);
        for (size_t slot = 0; slot < devices_.size(); ++slot)
        {
            if (graph_capture_boundary_device_words_[slot])
            {
                if (!graph_capture_boundary_memory_leases_[slot].has_value() ||
                    !graph_capture_boundary_memory_leases_[slot]->valid())
                {
                    LOG_ERROR("LocalTPContext: graph-boundary storage exists "
                              "without its physical-memory lease");
                    return false;
                }
                continue;
            }

            const DeviceId device = devices_[slot].toLocalDeviceId();
            IBackend *const backend = getBackendForDevice(device);
            std::optional<PhysicalMemoryAllocationLease> lease;
            void *buffer = nullptr;
            try
            {
                lease.emplace(memory_authority->claimNewAllocation(
                    device,
                    PhysicalMemoryOwner::LocalCollective,
                    kControlBytes));
                buffer = backend
                             ? backend->allocate(
                                   kControlBytes, device.ordinal)
                             : nullptr;
                if (!buffer)
                {
                    throw std::runtime_error(
                        "backend allocation returned a null graph-boundary pointer");
                }
            }
            catch (const std::exception &error)
            {
                // A backend may throw after the ledger claim. Roll back both
                // halves of this participant transaction before releasing any
                // already-published peers from the incomplete generation.
                if (buffer && backend)
                    backend->free(buffer, device.ordinal);
                LOG_ERROR("LocalTPContext: graph-boundary memory claim failed"
                          << " device=" << device.toString()
                          << " error=" << error.what());
                releaseGraphCaptureBoundaryDeviceWords();
                return false;
            }
            graph_capture_boundary_device_words_[slot] = buffer;
            graph_capture_boundary_memory_leases_[slot] = std::move(lease);
        }
        return true;
    }

    void LocalTPContext::releaseGraphCaptureBoundaryDeviceWords() noexcept
    {
        for (size_t slot = 0;
             slot < graph_capture_boundary_device_words_.size() &&
             slot < devices_.size();
             ++slot)
        {
            void *const buffer = graph_capture_boundary_device_words_[slot];
            if (buffer)
            {
                const DeviceId device = devices_[slot].toLocalDeviceId();
                if (IBackend *const backend = getBackendForDevice(device))
                    backend->free(buffer, device.ordinal);
            }
            graph_capture_boundary_device_words_[slot] = nullptr;
            if (slot < graph_capture_boundary_memory_leases_.size())
                graph_capture_boundary_memory_leases_[slot].reset();
        }
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
        // Thread-safe: multiple device threads call this concurrently during graph setup
        std::lock_guard<std::mutex> lock(mutex_);

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

        LOG_TRACE("[LocalTPContext] registerBARBackedOutput: stage='" << stage_name
                                                                      << "' device=" << device.toString());

        // Initialize vector for this stage if needed
        auto &tensors = bar_output_tensors_[stage_name];
        if (tensors.empty())
        {
            tensors.resize(degree(), nullptr);
        }

        // Register
        tensors[idx] = fp32_tensor;

        LOG_DEBUG("[LocalTPContext] Registered output for stage '"
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
        LOG_DEBUG("[LocalTPContext] Cleared all output registrations");
    }

    bool LocalTPContext::reserveFp16ScratchElements(
        size_t element_count,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory_authority)
    {
        if (backend_ != CollectiveBackendType::NCCL &&
            backend_ != CollectiveBackendType::RCCL)
        {
            return true;
        }
        if ((!device_group_.allCUDA() && !device_group_.allROCm()) ||
            element_count == 0)
        {
            LOG_ERROR("[LocalTPContext] Invalid FP16 collective scratch reservation"
                      << " elements=" << element_count
                      << " backend=" << collectiveBackendTypeToString(backend_));
            return false;
        }
        if (!memory_authority)
        {
            LOG_ERROR("[LocalTPContext] Native FP16 scratch reservation has no "
                      "physical-memory authority");
            return false;
        }

        size_t allocation_bytes = 0u;
        try
        {
            allocation_bytes =
                CollectiveMemoryEstimator::fp16ScratchAllocationBytes(
                    element_count);
        }
        catch (const std::overflow_error &error)
        {
            LOG_ERROR("[LocalTPContext] Invalid FP16 collective scratch reservation"
                      << " elements=" << element_count
                      << " backend=" << collectiveBackendTypeToString(backend_)
                      << " error=" << error.what());
            return false;
        }
        if (fp16_scratch_buffers_.size() != devices_.size() ||
            fp16_scratch_counts_.size() != devices_.size() ||
            fp16_scratch_memory_leases_.size() != devices_.size())
        {
            LOG_ERROR("[LocalTPContext] FP16 scratch metadata has invalid "
                      "participant geometry");
            return false;
        }
        for (size_t slot = 0; slot < devices_.size(); ++slot)
        {
            if (fp16_scratch_buffers_[slot] &&
                (!fp16_scratch_memory_leases_[slot].has_value() ||
                 !fp16_scratch_memory_leases_[slot]->valid()))
            {
                LOG_ERROR("[LocalTPContext] FP16 scratch exists without its "
                          "physical-memory lease"
                          << " slot=" << slot);
                return false;
            }
            if (fp16_scratch_buffers_[slot] &&
                fp16_scratch_counts_[slot] < element_count)
            {
                LOG_ERROR("[LocalTPContext] FP16 scratch capacity is immutable "
                          "after authoritative reservation"
                          << " slot=" << slot
                          << " existing_elements="
                          << fp16_scratch_counts_[slot]
                          << " requested_elements=" << element_count);
                return false;
            }
        }
        std::vector<void *> replacements(devices_.size(), nullptr);
        std::vector<std::optional<PhysicalMemoryAllocationLease>>
            replacement_leases(devices_.size());
        std::vector<bool> replace(devices_.size(), false);

        /*
         * Allocate every replacement before publishing any of them. A partial
         * reservation failure therefore leaves the prior complete generation
         * intact instead of exposing an asymmetric device set.
         */
        try
        {
            for (size_t slot = 0; slot < devices_.size(); ++slot)
            {
                if (fp16_scratch_buffers_[slot] &&
                    fp16_scratch_counts_[slot] >= element_count)
                {
                    continue;
                }

                const DeviceId device = devices_[slot].toLocalDeviceId();
                IBackend *const backend = getBackendForDevice(device);
                replacement_leases[slot].emplace(
                    memory_authority->claimNewAllocation(
                        device,
                        PhysicalMemoryOwner::LocalCollective,
                        allocation_bytes));
                replacements[slot] =
                    backend ? backend->allocate(allocation_bytes, device.ordinal) : nullptr;
                if (!replacements[slot])
                {
                    throw std::runtime_error(
                        "backend allocation returned a null FP16 scratch pointer");
                }
                replace[slot] = true;
            }
        }
        catch (const std::exception &error)
        {
            for (size_t slot = 0; slot < replacements.size(); ++slot)
            {
                if (!replacements[slot])
                    continue;
                const DeviceId device = devices_[slot].toLocalDeviceId();
                if (IBackend *const backend = getBackendForDevice(device))
                    backend->free(replacements[slot], device.ordinal);
            }
            LOG_ERROR("[LocalTPContext] FP16 scratch reservation failed"
                      << " elements=" << element_count
                      << " bytes_per_device=" << allocation_bytes
                      << " error=" << error.what());
            return false;
        }

        for (size_t slot = 0; slot < devices_.size(); ++slot)
        {
            if (!replace[slot])
                continue;
            const DeviceId device = devices_[slot].toLocalDeviceId();
            if (fp16_scratch_buffers_[slot])
            {
                IBackend *const backend = getBackendForDevice(device);
                if (!backend)
                    throw std::runtime_error(
                        "LocalTPContext lost the backend owning FP16 scratch");
                backend->free(fp16_scratch_buffers_[slot], device.ordinal);
            }
            fp16_scratch_buffers_[slot] = replacements[slot];
            fp16_scratch_counts_[slot] = element_count;
            fp16_scratch_memory_leases_[slot] =
                std::move(replacement_leases[slot]);
        }

        LOG_DEBUG("[LocalTPContext] Reserved persistent FP16 collective scratch"
                  << " elements=" << element_count
                  << " bytes_per_device=" << allocation_bytes
                  << " participants=" << devices_.size());
        return true;
    }

    void LocalTPContext::releaseFp16ScratchBuffers() noexcept
    {
        for (size_t slot = 0;
             slot < fp16_scratch_buffers_.size() && slot < devices_.size();
             ++slot)
        {
            void *const buffer = fp16_scratch_buffers_[slot];
            if (buffer)
            {
                const DeviceId device = devices_[slot].toLocalDeviceId();
                if (IBackend *const backend = getBackendForDevice(device))
                    backend->free(buffer, device.ordinal);
            }
            fp16_scratch_buffers_[slot] = nullptr;
            fp16_scratch_counts_[slot] = 0;
            if (slot < fp16_scratch_memory_leases_.size())
                fp16_scratch_memory_leases_[slot].reset();
        }
    }

    void *LocalTPContext::requireReservedFp16Scratch(
        int device_index,
        size_t element_count,
        const std::string &stage_name,
        const char *caller)
    {
        const bool valid_index =
            device_index >= 0 &&
            static_cast<size_t>(device_index) < devices_.size();
        const bool valid_metadata =
            fp16_scratch_buffers_.size() == devices_.size() &&
            fp16_scratch_counts_.size() == devices_.size();
        const bool sufficient =
            valid_index &&
            valid_metadata &&
            fp16_scratch_buffers_[static_cast<size_t>(device_index)] &&
            fp16_scratch_counts_[static_cast<size_t>(device_index)] >= element_count;
        if (sufficient)
            return fp16_scratch_buffers_[static_cast<size_t>(device_index)];

        requestAbort();
        std::ostringstream message;
        message << (caller ? caller : "LocalTPContext")
                << ": FP16 collective scratch reservation contract violated"
                << " stage=" << (stage_name.empty() ? "(none)" : stage_name)
                << " slot=" << device_index
                << " requested_elements=" << element_count
                << " reserved_elements="
                << ((valid_index && valid_metadata)
                        ? fp16_scratch_counts_[static_cast<size_t>(device_index)]
                        : 0)
                << ". Collective execution is allocation-free; fix graph setup capacity.";
        throw std::runtime_error(message.str());
    }

    bool LocalTPContext::reserveCollectiveResources(
        size_t backend_payload_capacity_bytes,
        size_t fp16_scratch_elements,
        const std::shared_ptr<PhysicalMemoryAuthority> &memory_authority)
    {
        if (!backend_impl_ || !backend_initialized_)
        {
            LOG_ERROR("[LocalTPContext] Cannot reserve collective resources: "
                      "backend is not initialized");
            return false;
        }
        if (backend_payload_capacity_bytes == 0 || fp16_scratch_elements == 0)
        {
            LOG_ERROR("[LocalTPContext] Collective resource reservation requires "
                      "non-zero byte and element capacities");
            return false;
        }

        std::lock_guard<std::mutex> lock(mutex_);
        LOG_DEBUG("[LocalTPContext] Reserving collective resources"
                  << " backend_payload_capacity_bytes="
                  << backend_payload_capacity_bytes
                  << " fp16_scratch_elements=" << fp16_scratch_elements);
        if (!initializeGraphCaptureBoundaryDeviceWords(memory_authority))
        {
            LOG_ERROR("[LocalTPContext] Graph-boundary resource reservation failed");
            return false;
        }
        if (!backend_impl_->reserveTempBufferBytes(
                backend_payload_capacity_bytes))
        {
            LOG_ERROR("[LocalTPContext] Backend payload-capacity reservation failed"
                      << " bytes=" << backend_payload_capacity_bytes);
            return false;
        }
        return reserveFp16ScratchElements(
            fp16_scratch_elements, memory_authority);
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
