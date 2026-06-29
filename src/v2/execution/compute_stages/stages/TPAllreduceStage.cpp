/**
 * @file TPAllreduceStage.cpp
 * @brief Implementation of all-reduce stage for tensor parallelism
 * @author David Sanftenberg
 * @date February 2026
 */

#include "TPAllreduceStage.h"
#include "../../../execution/local_execution/device/DeviceWorkspaceManager.h"
#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../memory/StageBufferContract.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/Logger.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/PerfStatsCollector.h"

#include <cstdint>
#include <utility>

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#endif

namespace llaminar2
{
    namespace
    {
        constexpr const char *kDefaultAllreducePrecision = "fp32";

        const char *allreduceRoleForStage(const std::string &stage_name)
        {
            if (stage_name.find("embedding") != std::string::npos)
                return "embedding";
            if (stage_name.find("shared_expert") != std::string::npos)
                return "moe_shared_expert";
            if (stage_name.find("moe_combined") != std::string::npos)
                return "moe_combined";
            if (stage_name.find("moe_expert") != std::string::npos ||
                stage_name.find("sparse_return") != std::string::npos)
                return "moe_routed_expert";
            if (stage_name.find("gdn_wo") != std::string::npos ||
                stage_name.find("wo_allreduce") != std::string::npos)
                return "attention_output";
            if (stage_name.find("down_allreduce") != std::string::npos)
                return "ffn_down";
            return "unknown";
        }

        std::string allreduceScopeString(const ITPContext *tp_ctx)
        {
            if (!tp_ctx)
                return "none";
            if (tp_ctx->isLocal())
                return "local";
            if (tp_ctx->isNodeLocal())
                return "node_local";
            return "global";
        }

        std::string requestedTransportPrecision(const TPAllreduceParams &params)
        {
            return params.precision.empty() ? std::string(kDefaultAllreducePrecision) : params.precision;
        }

        std::string effectiveTransportPrecision(
            const TPAllreduceParams &params,
            size_t effective_count)
        {
            const std::string requested_precision = requestedTransportPrecision(params);
            if (requested_precision == "fp16" &&
                params.tensor &&
                params.tensor->native_type() == TensorType::FP32 &&
                effective_count < debugEnv().allreduce_fp16_min_elements)
            {
                return "fp32";
            }
            return requested_precision;
        }

        size_t effectiveTransportElementBytes(
            const TPAllreduceParams &params,
            size_t effective_count,
            size_t tensor_element_bytes)
        {
            const std::string transport_precision =
                effectiveTransportPrecision(params, effective_count);
            if (transport_precision == "fp16" && params.tensor &&
                params.tensor->native_type() == TensorType::FP32)
            {
                return sizeof(uint16_t);
            }
            if (transport_precision == "bf16" && params.tensor &&
                params.tensor->native_type() == TensorType::FP32)
            {
                return sizeof(uint16_t);
            }
            return tensor_element_bytes;
        }

        const char *collectiveDataTypeNameForStats(CollectiveDataType dtype)
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

        size_t collectiveDataTypeBytesForStats(CollectiveDataType dtype)
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

        size_t sidebandResultElements(
            LocalTPCollectiveSidebandKind kind,
            size_t element_count,
            size_t degree)
        {
            if (kind == LocalTPCollectiveSidebandKind::Allgather)
                return element_count * degree;
            return element_count;
        }

        void recordAllreduceSidebandBillOfMaterialsEntry(
            const TPAllreduceParams &params,
            LocalTPCollectiveSidebandKind kind,
            size_t element_count,
            CollectiveDataType dtype,
            int root_device_index,
            const std::string &sideband_name,
            const std::string &source,
            bool grouped_with_anchor)
        {
            if (!PerfStatsCollector::isEnabled())
                return;

            const size_t degree = params.tp_ctx ? static_cast<size_t>(params.tp_ctx->degree()) : 0;
            const size_t element_bytes = collectiveDataTypeBytesForStats(dtype);
            const size_t local_bytes = element_count * element_bytes;
            const size_t result_bytes =
                sidebandResultElements(kind, element_count, degree) * element_bytes;

            PerfStatsCollector::Tags tags{
                {"stage", params.stage_name.empty() ? "unnamed" : params.stage_name},
                {"role", allreduceRoleForStage(params.stage_name)},
                {"anchor_stage", params.stage_name.empty() ? "unnamed" : params.stage_name},
                {"anchor_collective", "allreduce"},
                {"backend", params.tp_ctx ? collectiveBackendTypeToString(params.tp_ctx->backend()) : "none"},
                {"scope", allreduceScopeString(params.tp_ctx)},
                {"degree", std::to_string(degree)},
                {"sideband", sideband_name.empty() ? "unnamed" : sideband_name},
                {"kind", toString(kind)},
                {"dtype", collectiveDataTypeNameForStats(dtype)},
                {"element_bytes", std::to_string(element_bytes)},
                {"elements", std::to_string(element_count)},
                {"result_elements", std::to_string(sidebandResultElements(kind, element_count, degree))},
                {"root_device_index", std::to_string(root_device_index)},
                {"source", source},
                {"attachment", "tp_allreduce_stage_sideband"},
                {"launch_relation", grouped_with_anchor ? "same_group_as_anchor" : "same_stream_after_anchor"},
                {"fused_with_anchor", grouped_with_anchor ? "true" : "false"},
                {"physical_fusion", grouped_with_anchor ? "grouped_with_anchor_collective" : "separate_backend_collective"}};

            PerfStatsCollector::addCounter(
                "tp_allreduce_bom",
                "sideband_collective_calls",
                1.0,
                {},
                params.device_id.toString(),
                tags);
            PerfStatsCollector::addCounter(
                "tp_allreduce_bom",
                grouped_with_anchor
                    ? "sideband_grouped_with_anchor_collective_calls"
                    : "sideband_separate_backend_collective_calls",
                1.0,
                {},
                params.device_id.toString(),
                tags);

            PerfStatsCollector::Tags local_byte_tags = tags;
            PerfStatsCollector::addCounter(
                "tp_allreduce_bom",
                "sideband_local_bytes",
                static_cast<double>(local_bytes),
                {},
                params.device_id.toString(),
                std::move(local_byte_tags));

            PerfStatsCollector::Tags result_byte_tags = tags;
            PerfStatsCollector::addCounter(
                "tp_allreduce_bom",
                "sideband_result_bytes",
                static_cast<double>(result_bytes),
                {},
                params.device_id.toString(),
                std::move(result_byte_tags));
        }

        void recordAllreduceSidebandBillOfMaterials(
            const TPAllreduceParams &params,
            const std::vector<LocalTPCollectiveSidebandBuffer> &sidebands,
            bool grouped_with_anchor)
        {
            for (const auto &sideband : sidebands)
            {
                recordAllreduceSidebandBillOfMaterialsEntry(
                    params,
                    sideband.kind,
                    sideband.element_count,
                    sideband.dtype,
                    sideband.root_device_index,
                    sideband.name,
                    "raw_buffer",
                    grouped_with_anchor);
            }
        }

        void recordAllreduceWorkspaceSidebandBillOfMaterials(
            const TPAllreduceParams &params,
            bool grouped_with_anchor)
        {
            for (const auto &binding : params.sideband_workspace_bindings)
            {
                recordAllreduceSidebandBillOfMaterialsEntry(
                    params,
                    binding.kind,
                    binding.element_count,
                    binding.dtype,
                    binding.root_device_index,
                    binding.name.empty()
                        ? (binding.recv_buffer_name.empty()
                               ? binding.send_buffer_name
                                           : binding.recv_buffer_name)
                        : binding.name,
                    "workspace_binding",
                    grouped_with_anchor);
            }
        }

        void recordAllreduceBillOfMaterials(
            const TPAllreduceParams &params,
            size_t effective_count,
            bool no_op)
        {
            if (!PerfStatsCollector::isEnabled() || !params.tensor)
                return;

            const size_t tensor_numel = params.tensor->numel();
            const size_t tensor_element_bytes =
                tensor_numel > 0 ? (params.tensor->size_bytes() / tensor_numel) : 0;
            const size_t element_bytes =
                effectiveTransportElementBytes(params, effective_count, tensor_element_bytes);
            const size_t reduced_bytes = effective_count * element_bytes;
            PerfStatsCollector::Tags common_tags{
                {"stage", params.stage_name.empty() ? "unnamed" : params.stage_name},
                {"role", allreduceRoleForStage(params.stage_name)},
                {"backend", params.tp_ctx ? collectiveBackendTypeToString(params.tp_ctx->backend()) : "none"},
                {"scope", allreduceScopeString(params.tp_ctx)},
                {"degree", params.tp_ctx ? std::to_string(params.tp_ctx->degree()) : "0"},
                {"no_op", no_op ? "true" : "false"}};
            common_tags.emplace("elements", std::to_string(effective_count));
            common_tags.emplace("precision", params.precision.empty() ? "default" : params.precision);
            common_tags.emplace("requested_transport_precision", requestedTransportPrecision(params));
            common_tags.emplace("transport_precision", effectiveTransportPrecision(params, effective_count));
            PerfStatsCollector::Tags byte_tags = common_tags;
            byte_tags.emplace("element_bytes", std::to_string(element_bytes));
            byte_tags.emplace("tensor_element_bytes", std::to_string(tensor_element_bytes));
            byte_tags.emplace("tensor_numel", std::to_string(tensor_numel));
            byte_tags.emplace("tensor_type", params.tensor->dtype_name());

            PerfStatsCollector::addCounter(
                "tp_allreduce_bom",
                "bytes",
                static_cast<double>(reduced_bytes),
                {},
                params.device_id.toString(),
                std::move(byte_tags));

            PerfStatsCollector::addCounter(
                "tp_allreduce_bom",
                "stages",
                1.0,
                {},
                params.device_id.toString(),
                common_tags);
        }
    } // namespace

    // =========================================================================
    // Construction
    // =========================================================================

    TPAllreduceStage::TPAllreduceStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        // Validation is done at execute time to allow late binding
    }

    // =========================================================================
    // IComputeStage Implementation
    // =========================================================================

    bool TPAllreduceStage::execute(IDeviceContext *ctx)
    {
        KERNEL_PROFILE_SCOPE(KernelType::ALLREDUCE);

        (void)ctx; // Device context not directly used - TP context handles devices

        // Validate parameters
        if (!params_.tp_ctx)
        {
            LOG_ERROR("TPAllreduceStage: null tp_ctx");
            return false;
        }

        if (!params_.tensor)
        {
            LOG_ERROR("TPAllreduceStage: null tensor");
            return false;
        }

        // Resolve count: 0 means use tensor->numel()
        const size_t effective_count = (params_.count > 0) ? params_.count : params_.tensor->numel();
        const bool no_op_allreduce = params_.tp_ctx->degree() == 1 || debugEnv().skip_allreduce;
        recordAllreduceBillOfMaterials(params_, effective_count, no_op_allreduce);

        // Single-device context - no-op
        if (params_.tp_ctx->degree() == 1)
        {
            LOG_DEBUG("TPAllreduceStage: single device, no-op");
            return true;
        }

        // DIAGNOSTIC: Skip allreduce entirely for profiling (results will be wrong!)
        // Use LLAMINAR_SKIP_ALLREDUCE=1 to measure allreduce cost by elimination.
        if (debugEnv().skip_allreduce)
        {
            return true;
        }

        LOG_DEBUG("TPAllreduceStage: tensor diagnostics"
                      << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                      << " tensor=" << static_cast<void *>(params_.tensor)
                      << " tensor_name=" << (params_.tensor->debugName().empty() ? "(unnamed)" : params_.tensor->debugName())
                      << " home_device=" << params_.tensor->home_device().toString()
                      << " current_device=" << (params_.tensor->current_device().has_value() ? params_.tensor->current_device()->toString() : "none")
                      << " stage_stream=" << gpuStream()
                      << " hip_current_device=" << [&]() -> int
                                                                {
#ifdef HAVE_ROCM
                                                                    int dev = -1;
                                                                    if (hipGetDevice(&dev) != hipSuccess)
                                                                        dev = -1;
                                                                    return dev;
#else
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        return -1;
#endif
                                                                }()
                                                                << " gpu_ptr=" << params_.tensor->gpu_data_ptr()
                                                                << " count=" << effective_count
                                                                << " tensor_numel=" << params_.tensor->numel());

        // Log scope-aware message
        const char *scope_str = params_.tp_ctx->isLocal() ? "LOCAL" : (params_.tp_ctx->isNodeLocal() ? "NODE_LOCAL" : "GLOBAL");
        LOG_DEBUG("TPAllreduceStage (" << scope_str << "): all-reduce across " << params_.tp_ctx->degree()
                                       << " devices using " << collectiveBackendTypeToString(params_.tp_ctx->backend())
                                       << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                                       << " count=" << effective_count
                                       << " (params_.count=" << params_.count
                                       << ", tensor numel=" << params_.tensor->numel() << ")");

        void *stage_stream = gpuStream();
        const std::string transport_precision = requestedTransportPrecision(params_);
        const bool gpu_stage =
            params_.device_id.is_gpu() || (ctx && ctx->isGPU());

        std::vector<LocalTPCollectiveSidebandBuffer> sidebands = params_.sidebands;
        if (!params_.sideband_workspace_bindings.empty())
        {
            if (!bound_workspace_)
            {
                LOG_ERROR("TPAllreduceStage: workspace sidebands require a bound workspace"
                          << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                          << " binding_count=" << params_.sideband_workspace_bindings.size());
                return false;
            }

            for (const auto &binding : params_.sideband_workspace_bindings)
            {
                auto resolveBuffer = [&](const std::string &buffer_name) -> void *
                {
                    if (buffer_name.empty())
                        return nullptr;
                    return bound_workspace_->getBuffer(buffer_name);
                };

                void *send_buffer = resolveBuffer(binding.send_buffer_name);
                void *recv_buffer = resolveBuffer(binding.recv_buffer_name);
                if (!binding.send_buffer_name.empty() && !send_buffer)
                {
                    LOG_ERROR("TPAllreduceStage: missing sideband send workspace buffer"
                              << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                              << " sideband=" << (binding.name.empty() ? "(unnamed)" : binding.name)
                              << " buffer=" << binding.send_buffer_name);
                    return false;
                }
                if (!binding.recv_buffer_name.empty() && !recv_buffer)
                {
                    LOG_ERROR("TPAllreduceStage: missing sideband recv workspace buffer"
                              << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                              << " sideband=" << (binding.name.empty() ? "(unnamed)" : binding.name)
                              << " buffer=" << binding.recv_buffer_name);
                    return false;
                }

                LocalTPCollectiveSidebandBuffer sideband;
                sideband.kind = binding.kind;
                sideband.send_buffer = send_buffer;
                sideband.recv_buffer = recv_buffer;
                sideband.element_count = binding.element_count;
                sideband.dtype = binding.dtype;
                sideband.root_device_index = binding.root_device_index;
                sideband.name = binding.name.empty()
                                    ? (binding.recv_buffer_name.empty()
                                           ? binding.send_buffer_name
                                           : binding.recv_buffer_name)
                                    : binding.name;
                sidebands.push_back(std::move(sideband));
            }
        }

        // Use stage_name overload with count parameter.
        // CRITICAL: Pass actual count for decode (seq_len * hidden_dim, not buffer size).
        bool success;
        if (!sidebands.empty())
        {
            if (!stage_stream)
            {
                LOG_ERROR("TPAllreduceStage: sidebands require an explicit non-null stream"
                          << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name));
                return false;
            }
            auto *local_tp = dynamic_cast<ILocalTPContext *>(params_.tp_ctx);
            if (!local_tp)
            {
                LOG_ERROR("TPAllreduceStage: sidebands require a LocalTP context"
                          << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name));
                return false;
            }
            if (params_.sideband_device_index < 0)
            {
                LOG_ERROR("TPAllreduceStage: sidebands require a valid LocalTP participant index"
                          << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                          << " sideband_device_index=" << params_.sideband_device_index);
                return false;
            }
            recordAllreduceSidebandBillOfMaterials(params_, sidebands, true);
            success = local_tp->allreduceWithSidebandsOnStream(
                params_.tensor,
                params_.stage_name,
                effective_count,
                stage_stream,
                transport_precision,
                sidebands,
                params_.sideband_device_index);
            if (!success)
            {
                LOG_ERROR("TPAllreduceStage: LocalTP grouped allreduce sideband bundle failed"
                          << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                          << " sideband_count=" << sidebands.size());
                return false;
            }
        }
        else if (gpu_stage)
        {
            if (!stage_stream)
            {
                LOG_ERROR("TPAllreduceStage: GPU allreduce requires an explicit non-null stream"
                          << " stage_name=" << (params_.stage_name.empty() ? "(none)" : params_.stage_name)
                          << " device=" << params_.device_id.toString());
                return false;
            }
            success = params_.tp_ctx->allreduceOnStream(
                params_.tensor, params_.stage_name, effective_count, stage_stream,
                transport_precision);
        }
        else if (!params_.stage_name.empty())
        {
            // When a GPU stream is set (e.g., from graph capture), route through
            // the on-stream path so the allreduce kernel is recorded into the graph.
            if (stage_stream)
            {
                success = params_.tp_ctx->allreduceOnStream(
                    params_.tensor, params_.stage_name, effective_count, stage_stream,
                    transport_precision);
            }
            else
            {
                success = params_.tp_ctx->allreduce(params_.tensor, params_.stage_name, effective_count);
            }
        }
        else
        {
            // Fall back to no-stage-name overload (uses tensor->numel())
            success = params_.tp_ctx->allreduce(params_.tensor);
        }

        if (!success)
        {
            LOG_ERROR("TPAllreduceStage (" << scope_str << "): allreduce failed");
            return false;
        }

        // Dirty-marking is handled by LocalTPContext::allreduceOnStream() which
        // calls transitionToWithEvent(DEVICE_AUTHORITATIVE, ..., stream) to record a completion event.
        // This ensures ensureOnHost() waits for the allreduce to finish before D2H.

        return true;
    }

    WorkspaceRequirements TPAllreduceStage::getWorkspaceRequirements(
        int m, int n, int k) const
    {
        (void)m;
        (void)n;
        (void)k;
        return {};
    }

    void TPAllreduceStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
    }

    void TPAllreduceStage::unbindWorkspace()
    {
        bound_workspace_ = nullptr;
    }

    void TPAllreduceStage::onGraphReplayed()
    {
        if (!PerfStatsCollector::isEnabled() || !params_.tensor)
            return;

        const size_t effective_count =
            (params_.count > 0) ? params_.count : params_.tensor->numel();
        const bool no_op_allreduce =
            !params_.tp_ctx || params_.tp_ctx->degree() == 1 || debugEnv().skip_allreduce;
        recordAllreduceBillOfMaterials(params_, effective_count, no_op_allreduce);
        recordAllreduceSidebandBillOfMaterials(params_, params_.sidebands, true);
        recordAllreduceWorkspaceSidebandBillOfMaterials(params_, true);
    }

    bool TPAllreduceStage::needsOnGraphReplayed() const
    {
        return PerfStatsCollector::isEnabled();
    }

    bool TPAllreduceStage::supportsBackend(ComputeBackendType backend) const
    {
        // TP can work with any backend - the tp_ctx handles routing
        (void)backend;
        return true;
    }

    StageBufferRequirements TPAllreduceStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;

        if (params_.tensor)
        {
            // In-place all-reduce: buffer is both input and output
            BufferTensorType buf_type = toBufferTensorType(params_.tensor->native_type());
            reqs.addInout("tensor", params_.tensor->shape(), buf_type);
        }

        return reqs;
    }

    StageDumpInfo TPAllreduceStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;

        if (params_.tensor)
        {
            // All-reduce is in-place, so tensor is both input and output
            // Use actual count being reduced, not buffer size
            // For decode: count = seq_len * hidden_dim (not max_seq_len * hidden_dim)
            const size_t effective_count = (params_.count > 0) ? params_.count : params_.tensor->numel();
            const size_t cols = params_.tensor->cols();
            const size_t rows = (cols > 0) ? effective_count / cols : effective_count;

            info.addInput("tensor", params_.tensor, rows, cols);
            info.addOutput("tensor", params_.tensor, rows, cols);
        }

        if (params_.tp_ctx)
        {
            info.addScalarInt("tp_degree", params_.tp_ctx->degree());
            info.addScalarInt("backend", static_cast<int>(params_.tp_ctx->backend()));
            info.addScalarInt("tp_scope", static_cast<int>(params_.tp_ctx->scope()));
        }

        return info;
    }

    StageBufferContract TPAllreduceStage::bufferContract() const
    {
        if (!params_.tensor_buffer_id)
            return {};

        return StageBufferContract::build()
            .addPreallocatedInOut(*params_.tensor_buffer_id);
    }

    void TPAllreduceStage::setParams(const Params &params)
    {
        params_ = params;
        // Update base class device
        // Note: IComputeStage doesn't expose setDevice() publicly, so device
        // is fixed at construction. For reuse, create a new stage.
    }

} // namespace llaminar2
