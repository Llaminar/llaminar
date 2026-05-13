/**
 * @file MoEOverlayDomainRuntime.cpp
 * @brief Default MoE overlay domain runtime service implementation.
 */

#include "MoEOverlayDomainRuntime.h"

#include "utils/DebugEnv.h"
#include "utils/Logger.h"

#include <algorithm>
#include <mpi.h>
#include <mutex>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        bool isSparseTransferMode(MoEExpertTransferMode mode)
        {
            return mode == MoEExpertTransferMode::SparseTokenRows ||
                   mode == MoEExpertTransferMode::DecodeOneToken;
        }

        const MoEExpertTierDispatch *dispatchTierFor(
            const MoEOverlayDomainWorkRequest &request)
        {
            const auto *dispatch = request.dispatch_output
                                       ? request.dispatch_output
                                       : request.dispatch_output_lifetime.get();
            if (!dispatch)
                return nullptr;

            const int tier_index = request.dispatch_tier_index >= 0
                                       ? request.dispatch_tier_index
                                       : request.tier_index;
            if (tier_index < 0 || tier_index >= static_cast<int>(dispatch->tiers.size()))
                return nullptr;
            return &dispatch->tiers[static_cast<size_t>(tier_index)];
        }

        DeviceId sourceDeviceFor(const MoEOverlayDomainWorkRequest &request)
        {
            if (request.output_device.is_valid())
                return request.output_device;
            if (request.runtime_domain.primary_device.is_valid())
                return request.runtime_domain.primary_device;
            if (request.has_local_tp_params)
                return request.local_tp_params.device_id;
            if (request.has_cpu_fallback_params)
                return request.cpu_fallback_params.device_id;
            if (request.has_compute_params)
                return request.compute_params.device_id;
            return DeviceId::invalid();
        }

        MoEExpertParallelReducePartialInfo densePartialInfoFor(
            const MoEOverlayDomainWorkRequest &request)
        {
            MoEExpertParallelReducePartialInfo info;
            info.name = request.tier_name.empty()
                            ? "tier" + std::to_string(request.tier_index)
                            : request.tier_name;
            info.source_domain = request.runtime_domain.name.empty()
                                     ? request.source_domain.name
                                     : request.runtime_domain.name;
            info.source_device = sourceDeviceFor(request);
            return info;
        }

        TensorBase *dispatchInputFor(const MoEOverlayDomainWorkRequest &request)
        {
            if (request.has_local_tp_params)
                return request.local_tp_params.input;
            if (request.has_cpu_fallback_params)
                return request.cpu_fallback_params.input;
            if (request.has_compute_params)
                return request.compute_params.input;
            return nullptr;
        }

        MoEOverlayDomainWorkResult makeFailure(
            const MoEOverlayDomainWorkDescriptor &descriptor,
            const std::string &error)
        {
            MoEOverlayDomainWorkResult result;
            result.ok = false;
            result.error = error;
            result.descriptor = descriptor;
            result.dispatch_kind = descriptor.dispatch_kind;
            result.request_kind = descriptor.request_kind;
            result.dispatch_metrics = descriptor.dispatch_metrics;
            result.remote_placeholder = descriptor.remote_placeholder;
            result.partial_info = descriptor.partial_info;
            return result;
        }

        bool cpuFallbackHasActiveExperts(const MoEExpertOverlayCPUFallbackStage::Params &params)
        {
            return params.expert_mask.empty() ||
                   std::any_of(params.expert_mask.begin(), params.expert_mask.end(), [](bool active)
                               { return active; });
        }

        void clearOutput(TensorBase *output)
        {
            if (!output)
                return;
            std::fill_n(output->mutable_data(), output->numel(), 0.0f);
        }

        MoEOverlayDomainWorkResult makeCPUFallbackNoWorkResult(
            const MoEOverlayDomainWorkRequest &request,
            const MoEOverlayDomainWorkDescriptor &descriptor,
            bool compatibility_bridge_used)
        {
            clearOutput(request.output);

            MoEOverlayDomainWorkResult result;
            result.ok = true;
            result.descriptor = descriptor;
            result.dispatch_kind = descriptor.dispatch_kind;
            result.request_kind = MoEOverlayDispatchRequestKind::NoOp;
            result.dispatch_metrics = descriptor.dispatch_metrics;
            result.dispatch_metrics.no_op_count = 1;
            result.dispatch_metrics.routed_request_count = 0;
            result.dispatch_metrics.selected_row_count = 0;
            result.dispatch_metrics.routed_entry_count = 0;
            result.dispatch_metrics.transfer_bytes = 0;
            result.partial_tensor = request.output;
            result.partial_info = descriptor.partial_info;
            result.partial_info.selected_rows.clear();
            result.remote_placeholder = false;
            result.compatibility_fallback_used = compatibility_bridge_used;

            if (debugEnv().moe_expert_overlay.trace || debugEnv().tp_collective_contract_trace)
            {
                LOG_INFO("[MoEOverlayDomainRuntime] CPU fallback no-work fast path layer="
                         << request.layer_idx << " tier='" << request.tier_name
                         << "' domain='" << descriptor.domain_name
                         << "' skipped domain context creation");
            }
            return result;
        }

        MPI_Comm worldCommFor(const MoEExpertOverlayCPUFallbackStage::Params &params)
        {
            if (params.mpi_ctx && params.mpi_ctx->communicator() != MPI_COMM_NULL)
                return params.mpi_ctx->communicator();
            return MPI_COMM_WORLD;
        }

        std::string cpuFallbackContextKeyFor(const MoEExpertOverlayCPUFallbackStage::Params &params)
        {
            const auto world_ranks = MoEExpertOverlayCPUFallback::domainWorldRanks(params.domain);
            std::ostringstream key;
            key << MPI_Comm_c2f(worldCommFor(params))
                << ":" << params.domain.name
                << ":" << (params.domain_id >= 0 ? params.domain_id : MoEExpertOverlayCPUFallback::stableDomainId(params.domain.name))
                << ":" << params.root_world_rank
                << ":" << static_cast<int>(params.domain.compute_kind)
                << ":";
            for (size_t index = 0; index < world_ranks.size(); ++index)
            {
                if (index != 0)
                    key << ",";
                key << world_ranks[index];
            }
            return key.str();
        }

    } // namespace

    const char *toString(MoEOverlayDomainRuntimeDispatchKind kind)
    {
        switch (kind)
        {
        case MoEOverlayDomainRuntimeDispatchKind::ContinuationLocal:
            return "ContinuationLocal";
        case MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective:
            return "GraphDispatchCollective";
        case MoEOverlayDomainRuntimeDispatchKind::CompatibilitySingleDeviceStage:
            return "CompatibilitySingleDeviceStage";
        case MoEOverlayDomainRuntimeDispatchKind::CompatibilityLocalTPStage:
            return "CompatibilityLocalTPStage";
        case MoEOverlayDomainRuntimeDispatchKind::CompatibilityCPUFallbackStage:
            return "CompatibilityCPUFallbackStage";
        case MoEOverlayDomainRuntimeDispatchKind::Unsupported:
            return "Unsupported";
        }
        return "Unknown";
    }

    const char *toString(MoEOverlayDomainPartialLayout layout)
    {
        switch (layout)
        {
        case MoEOverlayDomainPartialLayout::DenseFullSequence:
            return "DenseFullSequence";
        case MoEOverlayDomainPartialLayout::SparseTokenRows:
            return "SparseTokenRows";
        }
        return "Unknown";
    }

    MoEOverlayDomainRuntime::MoEOverlayDomainRuntime(Config config)
        : config_(std::move(config))
    {
    }

    const MoEExpertOverlayExecutionPlan *MoEOverlayDomainRuntime::executionPlanFor(
        const MoEOverlayDomainWorkRequest &request) const
    {
        if (request.execution_plan)
            return request.execution_plan.get();
        return config_.execution_plan.get();
    }

    bool MoEOverlayDomainRuntime::hasRemoteParticipant(
        const MoEOverlayDomainWorkRequest &request) const
    {
        const auto *plan = executionPlanFor(request);
        const int current_rank = plan ? plan->currentRankPlan().world_rank : 0;

        for (const auto &participant : request.runtime_domain.participants)
        {
            if (participant.world_rank_known && participant.world_rank != current_rank)
                return true;
        }

        if (!request.source_domain.world_ranks.empty())
        {
            return std::any_of(request.source_domain.world_ranks.begin(),
                               request.source_domain.world_ranks.end(),
                               [&](int rank)
                               { return rank != current_rank; });
        }

        if (request.source_domain.owner_rank >= 0 && request.source_domain.owner_rank != current_rank)
            return true;

        return request.source_domain.kind == ExpertDomainKind::NodeLocalTP &&
               request.source_domain.participants.size() > 1;
    }

    bool MoEOverlayDomainRuntime::isContinuationLocal(
        const MoEOverlayDomainWorkRequest &request) const
    {
        const auto *plan = executionPlanFor(request);
        const std::string continuation_domain = plan
                                                    ? plan->continuation_domain
                                                    : request.continuation_domain;
        if (!continuation_domain.empty() && request.runtime_domain.name == continuation_domain)
            return true;
        return !hasRemoteParticipant(request) && request.runtime_domain.local_reachable_for_mvp;
    }

    MoEOverlayDomainWorkDescriptor MoEOverlayDomainRuntime::describeWork(
        const MoEOverlayDomainWorkRequest &request) const
    {
        MoEOverlayDomainWorkDescriptor descriptor;
        descriptor.layer_idx = request.layer_idx;
        descriptor.tier_index = request.tier_index;
        descriptor.tier_name = request.tier_name;
        descriptor.domain_name = request.runtime_domain.name.empty()
                                     ? request.source_domain.name
                                     : request.runtime_domain.name;
        descriptor.continuation_domain = request.continuation_domain;
        descriptor.partial_info = densePartialInfoFor(request);
        descriptor.dispatch_group = request.dispatch_group;

        if (const auto *tier = dispatchTierFor(request))
        {
            descriptor.transfer_mode = tier->transfer_mode;
            descriptor.selected_rows = tier->token_rows;
            descriptor.routed_entries = tier->entries.size();
            descriptor.transfer_volume = tier->transfer_volume;
            descriptor.request_kind = tier->entries.empty()
                                          ? MoEOverlayDispatchRequestKind::NoOp
                                          : MoEOverlayDispatchRequestKind::RoutedWork;
            descriptor.dispatch_metrics.no_op_count = tier->entries.empty() ? 1 : 0;
            descriptor.dispatch_metrics.routed_request_count = tier->entries.empty() ? 0 : 1;
            descriptor.dispatch_metrics.selected_row_count = tier->token_rows.size();
            descriptor.dispatch_metrics.routed_entry_count = tier->entries.size();
            descriptor.dispatch_metrics.transfer_bytes = tier->transfer_volume.totalBytes();
        }

        if (isContinuationLocal(request))
        {
            descriptor.dispatch_kind = MoEOverlayDomainRuntimeDispatchKind::ContinuationLocal;
        }
        else if (hasRemoteParticipant(request))
        {
            descriptor.dispatch_kind = MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective;
            descriptor.compatibility_fallback_available =
                config_.enable_compatibility_fallback &&
                (request.has_cpu_fallback_params || request.has_local_tp_params || request.has_compute_params);
            if (isSparseTransferMode(descriptor.transfer_mode) && !descriptor.selected_rows.empty())
            {
                descriptor.expected_partial_layout = MoEOverlayDomainPartialLayout::SparseTokenRows;
                descriptor.partial_info.selected_rows = descriptor.selected_rows;
            }
        }
        else if (request.has_local_tp_params)
        {
            descriptor.dispatch_kind = MoEOverlayDomainRuntimeDispatchKind::CompatibilityLocalTPStage;
        }
        else if (request.has_cpu_fallback_params)
        {
            descriptor.dispatch_kind = MoEOverlayDomainRuntimeDispatchKind::CompatibilityCPUFallbackStage;
        }
        else if (request.has_compute_params)
        {
            descriptor.dispatch_kind = MoEOverlayDomainRuntimeDispatchKind::CompatibilitySingleDeviceStage;
        }

        return descriptor;
    }

    MoEOverlayDispatchRequest MoEOverlayDomainRuntime::buildDispatchRequest(
        const MoEOverlayDomainWorkRequest &request,
        const MoEOverlayDomainWorkDescriptor &descriptor) const
    {
        if (descriptor.request_kind == MoEOverlayDispatchRequestKind::NoOp)
        {
            return MoEOverlayDispatchRequest::noOp(
                descriptor.dispatch_group,
                descriptor.layer_idx,
                descriptor.tier_index);
        }

        if (descriptor.request_kind == MoEOverlayDispatchRequestKind::Cancel)
        {
            return MoEOverlayDispatchRequest::cancel(
                descriptor.dispatch_group,
                descriptor.layer_idx,
                descriptor.tier_index,
                0);
        }

        return MoEOverlayDispatchRequest::routedWork(
            descriptor.dispatch_group,
            descriptor.layer_idx,
            descriptor.tier_index,
            descriptor.selected_rows.empty() ? nullptr : descriptor.selected_rows.data(),
            descriptor.selected_rows.size(),
            descriptor.routed_entries,
            descriptor.transfer_volume.totalBytes(),
            dispatchInputFor(request),
            request.output);
    }

    MoEOverlayDomainWorkResult MoEOverlayDomainRuntime::runDispatchBackend(
        const MoEOverlayDomainWorkRequest &request,
        const MoEOverlayDomainWorkDescriptor &descriptor,
        IDeviceContext *ctx)
    {
        MoEOverlayDomainWorkResult result;
        result.descriptor = descriptor;
        result.dispatch_kind = descriptor.dispatch_kind;
        result.request_kind = descriptor.request_kind;
        result.dispatch_metrics = descriptor.dispatch_metrics;
        result.partial_tensor = request.output;
        result.partial_info = descriptor.partial_info;

        if (!config_.dispatch_backend)
            return makeFailure(descriptor, "graph-native overlay dispatch backend is not configured");

        const auto dispatch_request = buildDispatchRequest(request, descriptor);
        auto dispatch_result = config_.dispatch_backend->dispatch(
            descriptor.dispatch_group,
            dispatch_request,
            ctx);

        result.ok = dispatch_result.ok;
        result.request_kind = dispatch_result.request_kind;
        result.dispatch_metrics.merge(dispatch_result.metrics);
        result.partial_tensor = dispatch_result.partial_output ? dispatch_result.partial_output : request.output;
        if (!dispatch_result.ok)
        {
            std::ostringstream error;
            error << "graph-native overlay dispatch backend failed for domain '"
                  << descriptor.domain_name << "'";
            if (!dispatch_result.error.empty())
                error << ": " << dispatch_result.error;
            else if (dispatch_result.error_code != 0)
                error << " with error_code=" << dispatch_result.error_code;
            result.error = error.str();
        }
        return result;
    }

    MoEOverlayDomainWorkResult MoEOverlayDomainRuntime::runCPUFallbackGraphDispatch(
        const MoEOverlayDomainWorkRequest &request,
        const MoEOverlayDomainWorkDescriptor &descriptor,
        IDeviceContext *ctx)
    {
        MoEOverlayDomainWorkResult result;
        result.descriptor = descriptor;
        result.dispatch_kind = descriptor.dispatch_kind;
        result.request_kind = descriptor.request_kind;
        result.dispatch_metrics = descriptor.dispatch_metrics;
        result.remote_placeholder = false;
        result.compatibility_fallback_used = false;
        result.partial_tensor = request.output;
        result.partial_info = descriptor.partial_info;

        if (!cpuFallbackHasActiveExperts(request.cpu_fallback_params))
        {
            return makeCPUFallbackNoWorkResult(request, descriptor, false);
        }

        try
        {
            MoECPUFallbackTransferStats transfer_stats;
            MoECPUFallbackTensorParallelStats tensor_parallel_stats;
            auto params = request.cpu_fallback_params;

            if (!params.domain_context)
            {
                params.domain_context = cpuFallbackDomainContextFor(params);
            }
            if (params.domain_context && params.domain_context->participates())
            {
                result.dispatch_metrics.remote_endpoint_work_count += 1;
            }
            if (!params.transfer_stats)
                params.transfer_stats = &transfer_stats;
            if (!params.tensor_parallel_stats)
                params.tensor_parallel_stats = &tensor_parallel_stats;

            MoEExpertOverlayCPUFallbackStage stage(std::move(params));
            result.ok = stage.execute(ctx);
            result.cpu_transfer_stats = transfer_stats;
            result.cpu_tensor_parallel_stats = tensor_parallel_stats;
            if (!result.ok)
                result.error = "CPU fallback graph dispatch stage failed";
        }
        catch (const std::exception &e)
        {
            result.ok = false;
            result.error = std::string("CPU fallback graph dispatch stage failed: ") + e.what();
        }
        catch (...)
        {
            result.ok = false;
            result.error = "CPU fallback graph dispatch stage failed: unknown exception";
        }
        return result;
    }

    MoEOverlayDomainWorkResult MoEOverlayDomainRuntime::runCompatibilityPath(
        const MoEOverlayDomainWorkRequest &request,
        const MoEOverlayDomainWorkDescriptor &descriptor,
        IDeviceContext *ctx,
        bool compatibility_bridge_used)
    {
        MoEOverlayDomainWorkResult result;
        result.descriptor = descriptor;
        result.dispatch_kind = descriptor.dispatch_kind;
        result.request_kind = descriptor.request_kind;
        result.dispatch_metrics = descriptor.dispatch_metrics;
        result.remote_placeholder = descriptor.remote_placeholder;
        result.compatibility_fallback_used = compatibility_bridge_used;
        result.partial_tensor = request.output;
        result.partial_info = densePartialInfoFor(request);

        if (request.has_local_tp_params)
        {
            auto params = request.local_tp_params;
            if (!params.diagnostics && params.diagnostics_lifetime)
                params.diagnostics = params.diagnostics_lifetime.get();
            MoEExpertOverlayLocalTPStage stage(params);
            result.ok = stage.execute(ctx);
            if (params.diagnostics)
                result.local_tp_diagnostics = *params.diagnostics;
            if (!result.ok)
                result.error = "LocalTP compatibility stage failed";
            return result;
        }

        if (request.has_cpu_fallback_params)
        {
            if (!cpuFallbackHasActiveExperts(request.cpu_fallback_params))
            {
                return makeCPUFallbackNoWorkResult(request, descriptor, compatibility_bridge_used);
            }

            MoECPUFallbackTransferStats transfer_stats;
            MoECPUFallbackTensorParallelStats tensor_parallel_stats;
            auto params = request.cpu_fallback_params;
            if (!params.domain_context)
                params.domain_context = cpuFallbackDomainContextFor(params);
            if (!params.transfer_stats)
                params.transfer_stats = &transfer_stats;
            if (!params.tensor_parallel_stats)
                params.tensor_parallel_stats = &tensor_parallel_stats;
            MoEExpertOverlayCPUFallbackStage stage(std::move(params));
            result.ok = stage.execute(ctx);
            result.cpu_transfer_stats = transfer_stats;
            result.cpu_tensor_parallel_stats = tensor_parallel_stats;
            if (!result.ok)
                result.error = "CPU fallback compatibility stage failed";
            return result;
        }

        if (request.has_compute_params)
        {
            MoEExpertComputeStage stage(request.compute_params);
            result.ok = stage.execute(ctx);
            if (!result.ok)
                result.error = "single-device compatibility stage failed";
            return result;
        }

        result.ok = false;
        result.error = "no local, CPU fallback, or compatibility execution path is available";
        return result;
    }

    std::shared_ptr<MoECPUFallbackDomainContext> MoEOverlayDomainRuntime::cpuFallbackDomainContextFor(
        const MoEExpertOverlayCPUFallbackStage::Params &params)
    {
        const std::string key = cpuFallbackContextKeyFor(params);
        std::lock_guard<std::mutex> lock(cpu_domain_context_mutex_);

        auto existing = cpu_domain_contexts_.find(key);
        if (existing != cpu_domain_contexts_.end())
        {
            if (debugEnv().moe_expert_overlay.trace)
            {
                LOG_INFO("[MoEOverlayDomainRuntime][trace] reuse CPU fallback domain context domain='"
                         << params.domain.name << "' root_rank=" << params.root_world_rank);
            }
            return existing->second;
        }

        MoECPUFallbackDomainConfig domain_config;
        domain_config.domain = params.domain;
        domain_config.world_comm = worldCommFor(params);
        domain_config.root_world_rank = params.root_world_rank;
        domain_config.domain_id = params.domain_id;
        domain_config.hostfile_path = params.hostfile_path;

        if (debugEnv().moe_expert_overlay.trace)
        {
            LOG_INFO("[MoEOverlayDomainRuntime][trace] create CPU fallback domain context domain='"
                     << params.domain.name << "' root_rank=" << params.root_world_rank);
        }
        auto context = std::make_shared<MoECPUFallbackDomainContext>(
            MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(domain_config));
        cpu_domain_contexts_.emplace(key, context);
        return context;
    }

    MoEOverlayDomainWorkResult MoEOverlayDomainRuntime::submit(
        const MoEOverlayDomainWorkRequest &request,
        IDeviceContext *ctx)
    {
        const auto descriptor = describeWork(request);
        const bool trace = debugEnv().moe_expert_overlay.trace;
        if (trace)
        {
            LOG_INFO("[MoEOverlayDomainRuntime][trace] layer=" << request.layer_idx
                                                               << " tier='" << request.tier_name
                                                               << "' domain='" << descriptor.domain_name
                                                               << "' kind=" << toString(descriptor.dispatch_kind)
                                                               << " selected_rows=" << descriptor.selected_rows.size());
        }

        if (descriptor.dispatch_kind == MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective &&
            config_.dispatch_backend)
        {
            return runDispatchBackend(request, descriptor, ctx);
        }

        if (descriptor.dispatch_kind == MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective &&
            request.has_cpu_fallback_params)
        {
            if (!cpuFallbackHasActiveExperts(request.cpu_fallback_params))
            {
                return makeCPUFallbackNoWorkResult(request, descriptor, false);
            }

            const auto *plan = executionPlanFor(request);
            if (plan && plan->world_size > 1 && hasRemoteParticipant(request))
            {
                std::ostringstream error;
                error << "graph-native CPU fallback dispatch for domain '" << descriptor.domain_name
                      << "' requires a dispatch backend; refusing native CPU fallback compatibility path "
                      << "because remote CPU fallback participants run endpoint services";
                return makeFailure(descriptor, error.str());
            }

            return runCPUFallbackGraphDispatch(request, descriptor, ctx);
        }

        if (descriptor.dispatch_kind == MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective &&
            !descriptor.compatibility_fallback_available)
        {
            std::ostringstream error;
            error << "graph-native overlay dispatch collective for domain '" << descriptor.domain_name
                  << "' has no backend or compatibility execution path";
            return makeFailure(descriptor, error.str());
        }

        return runCompatibilityPath(
            request,
            descriptor,
            ctx,
            descriptor.dispatch_kind == MoEOverlayDomainRuntimeDispatchKind::GraphDispatchCollective);
    }

    std::shared_ptr<IOverlayDomainRuntime> makeMoEOverlayDomainRuntime(
        MoEOverlayDomainRuntime::Config config)
    {
        return std::make_shared<MoEOverlayDomainRuntime>(std::move(config));
    }

} // namespace llaminar2
