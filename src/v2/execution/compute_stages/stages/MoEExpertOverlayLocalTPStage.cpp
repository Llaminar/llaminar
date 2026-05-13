/**
 * @file MoEExpertOverlayLocalTPStage.cpp
 * @brief Graph stage wrapper for accelerator LocalTP MoE expert-overlay tiers.
 */

#include "MoEExpertOverlayLocalTPStage.h"

#include "../../../execution/local_execution/device/WorkspaceDescriptor.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../interfaces/IWorkspaceConsumer.h"
#include "../../../tensors/TensorKernels.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <exception>
#include <functional>
#include <utility>

namespace llaminar2
{
    namespace
    {
        bool validateShape2D(const TensorBase *tensor, int rows, int cols, const char *name)
        {
            if (!tensor)
            {
                LOG_ERROR("[MoEExpertOverlayLocalTPStage] Null " << name << " tensor");
                return false;
            }
            const auto &shape = tensor->shape();
            if (shape.size() != 2 || shape[0] < static_cast<size_t>(rows) || shape[1] != static_cast<size_t>(cols))
            {
                LOG_ERROR("[MoEExpertOverlayLocalTPStage] " << name << " shape mismatch: got rank "
                                                            << shape.size() << ", expected at least [" << rows << ", " << cols << "]");
                return false;
            }
            return true;
        }

        const MoEExpertTierDispatch *resolveDispatchTier(
            const MoEExpertOverlayLocalTPStage::Params &params)
        {
            if (!params.dispatch_output)
                return nullptr;
            if (params.dispatch_tier_index < 0 ||
                params.dispatch_tier_index >= static_cast<int>(params.dispatch_output->tiers.size()))
            {
                LOG_ERROR("[MoEExpertOverlayLocalTPStage] Dispatch tier index "
                          << params.dispatch_tier_index << " is outside descriptor tier count "
                          << params.dispatch_output->tiers.size());
                return nullptr;
            }

            if (params.dispatch_output->seq_len != params.seq_len ||
                params.dispatch_output->top_k != params.top_k ||
                params.dispatch_output->d_model != params.d_model)
            {
                LOG_ERROR("[MoEExpertOverlayLocalTPStage] Dispatch descriptor dimension mismatch for domain '"
                          << params.domain.name << "': descriptor seq_len=" << params.dispatch_output->seq_len
                          << " top_k=" << params.dispatch_output->top_k
                          << " d_model=" << params.dispatch_output->d_model
                          << " stage seq_len=" << params.seq_len
                          << " top_k=" << params.top_k
                          << " d_model=" << params.d_model);
                return nullptr;
            }

            const auto &tier = params.dispatch_output->tiers[static_cast<size_t>(params.dispatch_tier_index)];
            if (tier.domain != params.domain.name)
            {
                LOG_ERROR("[MoEExpertOverlayLocalTPStage] Dispatch tier domain '" << tier.domain
                                                                                  << "' does not match LocalTP domain '" << params.domain.name << "'");
                return nullptr;
            }
            return &tier;
        }

        void clearOutput(TensorBase *output)
        {
            if (!output)
                return;
            std::fill_n(output->mutable_data(), output->numel(), 0.0f);
        }

        void recordTransferDiagnostics(
            MoEExpertOverlayLocalTPDiagnostics *diagnostics,
            const MoEExpertTierDispatch &tier)
        {
            if (!diagnostics)
                return;
            diagnostics->transfer_mode = tier.transfer_mode;
            diagnostics->transfer_volume = tier.transfer_volume;
            diagnostics->selected_token_rows = tier.token_rows;
        }

        void initializeEmptyDiagnostics(
            const MoEExpertOverlayLocalTPStage::Params &params,
            const ILocalTPContext &local_tp_context,
            const MoEExpertTierDispatch &tier,
            MoEExpertOverlayLocalTPDiagnostics *diagnostics)
        {
            if (!diagnostics)
                return;
            diagnostics->clear();
            diagnostics->domain_name = params.domain.name;
            diagnostics->backend = local_tp_context.backend();
            diagnostics->degree = local_tp_context.degree();
            recordTransferDiagnostics(diagnostics, tier);
        }

        void traceDescriptorConsumption(
            const MoEExpertOverlayLocalTPStage::Params &params,
            const MoEExpertTierDispatch &tier)
        {
            const auto &env = debugEnv();
            if (!env.moe_expert_overlay.transfer_trace && !env.moe_expert_overlay.trace && !env.profile.enabled)
                return;

            LOG_INFO("[MoEExpertOverlayLocalTPStage] layer=" << params.layer_idx
                                                             << " domain=" << params.domain.name
                                                             << " tier=" << tier.tier_index
                                                             << " selected_rows=" << tier.token_rows.size()
                                                             << " routed_entries=" << tier.entries.size()
                                                             << " mode=" << toString(tier.transfer_mode)
                                                             << " outbound_bytes=" << tier.transfer_volume.outbound_bytes
                                                             << " return_bytes=" << tier.transfer_volume.return_bytes);
        }

        bool isSparseTransferMode(MoEExpertTransferMode mode)
        {
            return mode == MoEExpertTransferMode::SparseTokenRows ||
                   mode == MoEExpertTransferMode::DecodeOneToken;
        }

        int activeExpertCount(const std::vector<bool> &mask, int num_experts)
        {
            if (mask.empty())
                return num_experts;
            return static_cast<int>(std::count(mask.begin(), mask.end(), true));
        }

        std::shared_ptr<FP32Tensor> makeScratchFP32(size_t rows, size_t cols, DeviceId device)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{rows, cols});
            if (device.is_gpu())
                tensor->allocateOnDevice(device);
            return tensor;
        }

        void ensureScratchDeviceAllocation(FP32Tensor *tensor, DeviceId device)
        {
            if (tensor && device.is_gpu())
                tensor->allocateOnDevice(device);
        }

        void initializePreparedParticipantScratch(MoEExpertOverlayLocalTPStage::Params &params)
        {
            if (params.prepared_participants.empty())
                return;

            const size_t rows = params.seq_len > 0 ? static_cast<size_t>(params.seq_len) : 1u;
            const size_t d_model = params.d_model > 0 ? static_cast<size_t>(params.d_model) : 1u;
            const size_t intermediate = params.expert_intermediate > 0 ? static_cast<size_t>(params.expert_intermediate) : 1u;
            const size_t max_routed_entries = rows * static_cast<size_t>(std::max(1, params.top_k));

            for (auto &participant : params.prepared_participants)
            {
                if (!participant.input_scratch)
                    participant.input_scratch = makeScratchFP32(rows, d_model, participant.device);
                if (!participant.batch_scratch)
                    participant.batch_scratch = makeScratchFP32(rows, d_model, participant.device);
                if (!participant.gate_scratch)
                    participant.gate_scratch = makeScratchFP32(rows, intermediate, participant.device);
                if (!participant.up_scratch)
                    participant.up_scratch = makeScratchFP32(rows, intermediate, participant.device);
                if (!participant.partial_scratch)
                    participant.partial_scratch = makeScratchFP32(rows, d_model, participant.device);
                ensureScratchDeviceAllocation(participant.input_scratch.get(), participant.device);
                ensureScratchDeviceAllocation(participant.batch_scratch.get(), participant.device);
                ensureScratchDeviceAllocation(participant.gate_scratch.get(), participant.device);
                ensureScratchDeviceAllocation(participant.up_scratch.get(), participant.device);
                ensureScratchDeviceAllocation(participant.partial_scratch.get(), participant.device);
                participant.token_indices.reserve(max_routed_entries);
                participant.token_weights.reserve(max_routed_entries);
            }

            params.prepared_partial_views.clear();
            params.prepared_partial_views.reserve(params.prepared_participants.size());
            for (auto &participant : params.prepared_participants)
                params.prepared_partial_views.push_back(participant.partial_scratch.get());

            (void)MoEExpertTokenRowTransfer::ensureBuffers(
                &params.sparse_transfer_buffers,
                rows,
                params.top_k,
                params.d_model);
        }

        void forEachPreparedGemm(
            const MoEExpertOverlayLocalTPStage::Params &params,
            const std::function<void(ITensorGemm *)> &visitor)
        {
            for (const auto &participant : params.prepared_participants)
            {
                for (auto *gemm : participant.gate_gemm)
                    visitor(gemm);
                for (auto *gemm : participant.up_gemm)
                    visitor(gemm);
                for (auto *gemm : participant.down_gemm)
                    visitor(gemm);
            }
        }

        void forEachParticipantPreparedGemm(
            const MoEExpertOverlayLocalTPPreparedParticipant &participant,
            const std::function<void(ITensorGemm *)> &visitor)
        {
            for (auto *gemm : participant.gate_gemm)
                visitor(gemm);
            for (auto *gemm : participant.up_gemm)
                visitor(gemm);
            for (auto *gemm : participant.down_gemm)
                visitor(gemm);
        }

        WorkspaceRequirements firstParticipantWorkspaceRequirements(
            const MoEExpertOverlayLocalTPPreparedParticipant &participant,
            int m,
            int n,
            int k)
        {
            WorkspaceRequirements requirements;
            forEachParticipantPreparedGemm(participant, [&](ITensorGemm *gemm)
                                           {
            if (!gemm || !requirements.buffers.empty())
                return;
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
            if (consumer)
                requirements = consumer->getWorkspaceRequirements(m, n, k); });
            return requirements;
        }

        void bindParticipantPreparedGemm(
            const MoEExpertOverlayLocalTPPreparedParticipant &participant,
            DeviceWorkspaceManager *workspace)
        {
            forEachParticipantPreparedGemm(participant, [workspace](ITensorGemm *gemm)
                                           {
            if (!gemm)
                return;
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
            if (consumer)
                consumer->bindWorkspace(workspace); });
        }
    } // namespace

    MoEExpertOverlayLocalTPStage::MoEExpertOverlayLocalTPStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.dispatch_output && params_.dispatch_output_lifetime)
            params_.dispatch_output = params_.dispatch_output_lifetime.get();
        if (!params_.diagnostics && params_.diagnostics_lifetime)
            params_.diagnostics = params_.diagnostics_lifetime.get();
        initializePreparedParticipantScratch(params_);
    }

    bool MoEExpertOverlayLocalTPStage::execute(IDeviceContext *ctx)
    {
        (void)ctx;

        const bool profiling_enabled = MoEExpertOverlayProfiler::isEnabled();
        const auto stage_start = profiling_enabled ? std::chrono::steady_clock::now()
                                                   : std::chrono::steady_clock::time_point{};
        MoEExpertOverlayLocalTPDiagnostics local_diagnostics;
        MoEExpertOverlayLocalTPDiagnostics *diagnostics = params_.diagnostics;
        if (!diagnostics && profiling_enabled)
            diagnostics = &local_diagnostics;

        if (!params_.local_tp_context)
        {
            LOG_ERROR("[MoEExpertOverlayLocalTPStage] Missing LocalTP context for domain '"
                      << params_.domain.name << "'");
            return false;
        }
        if (params_.seq_len <= 0 || params_.d_model <= 0 || params_.num_experts <= 0 ||
            params_.top_k <= 0 || params_.expert_intermediate <= 0)
        {
            LOG_ERROR("[MoEExpertOverlayLocalTPStage] Invalid dimensions for domain '"
                      << params_.domain.name << "': seq_len=" << params_.seq_len
                      << " d_model=" << params_.d_model
                      << " num_experts=" << params_.num_experts
                      << " top_k=" << params_.top_k
                      << " expert_intermediate=" << params_.expert_intermediate);
            return false;
        }
        if (!validateShape2D(params_.input, params_.seq_len, params_.d_model, "input") ||
            !validateShape2D(params_.routing_indices, params_.seq_len, params_.top_k, "routing_indices") ||
            !validateShape2D(params_.routing_weights, params_.seq_len, params_.top_k, "routing_weights") ||
            !validateShape2D(params_.output, params_.seq_len, params_.d_model, "output"))
        {
            return false;
        }

        std::string reason;
        if (!MoEExpertOverlayLocalTPExecutor::canExecute(params_.domain, *params_.local_tp_context, &reason))
        {
            LOG_ERROR("[MoEExpertOverlayLocalTPStage] Domain '" << params_.domain.name
                                                                << "' cannot execute with supplied LocalTP context: " << reason);
            return false;
        }

        const MoEExpertTierDispatch *dispatch_tier = nullptr;
        if (params_.dispatch_output)
        {
            dispatch_tier = resolveDispatchTier(params_);
            if (!dispatch_tier)
                return false;
            traceDescriptorConsumption(params_, *dispatch_tier);

            if (dispatch_tier->entries.empty())
            {
                clearOutput(params_.output);
                initializeEmptyDiagnostics(params_, *params_.local_tp_context, *dispatch_tier, diagnostics);
                if (profiling_enabled && diagnostics)
                {
                    diagnostics->compute_ms = std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() - stage_start)
                                                  .count();
                    MoEExpertOverlayProfiler::recordLocalTP(
                        params_.layer_idx,
                        params_.domain,
                        activeExpertCount(params_.expert_mask, params_.num_experts),
                        *diagnostics);
                }
                if (debugEnv().moe_expert_overlay.trace || debugEnv().tp_collective_contract_trace)
                {
                    LOG_INFO("[MoEExpertOverlayLocalTPStage] layer=" << params_.layer_idx
                                                                     << " domain='" << params_.domain.name
                                                                     << "' tier=" << dispatch_tier->tier_index
                                                                     << " no routed work; skipped LocalTP executor");
                }
                return true;
            }
        }

        MoEExpertOverlayLocalTPRunParams run;
        run.input = params_.input;
        run.routing_indices = params_.routing_indices;
        run.routing_weights = params_.routing_weights;
        run.gate_exps = params_.gate_exps;
        run.up_exps = params_.up_exps;
        run.down_exps = params_.down_exps;
        run.output = params_.output;
        run.seq_len = params_.seq_len;
        run.d_model = params_.d_model;
        run.num_experts = params_.num_experts;
        run.top_k = params_.top_k;
        run.expert_intermediate = params_.expert_intermediate;
        run.layer_idx = params_.layer_idx;
        run.output_device = params_.device_id;
        run.expert_mask = params_.expert_mask;
        run.stage_name_prefix = params_.stage_name_prefix;
        run.upload_partials_to_participant_devices = params_.upload_partials_to_participant_devices;
        run.prepared_participants = &params_.prepared_participants;
        run.prepared_partial_views = &params_.prepared_partial_views;
        if (dispatch_tier)
            run.dispatch_entries = &dispatch_tier->entries;

        try
        {
            if (dispatch_tier && isSparseTransferMode(dispatch_tier->transfer_mode))
            {
                clearOutput(params_.output);
                if (dispatch_tier->token_rows.empty())
                {
                    initializeEmptyDiagnostics(params_, *params_.local_tp_context, *dispatch_tier, diagnostics);
                    if (profiling_enabled && diagnostics)
                    {
                        diagnostics->compute_ms = std::chrono::duration<double, std::milli>(
                                                      std::chrono::steady_clock::now() - stage_start)
                                                      .count();
                        MoEExpertOverlayProfiler::recordLocalTP(
                            params_.layer_idx,
                            params_.domain,
                            activeExpertCount(params_.expert_mask, params_.num_experts),
                            *diagnostics);
                    }
                    return true;
                }

                MoEExpertTransferVolume volume;
                if (!MoEExpertTokenRowTransfer::gatherRows(
                        params_.input,
                        params_.routing_indices,
                        params_.routing_weights,
                        dispatch_tier->token_rows,
                        params_.seq_len,
                        params_.top_k,
                        params_.d_model,
                        &params_.sparse_transfer_buffers,
                        &volume,
                        dispatch_tier->transfer_mode))
                {
                    return false;
                }

                run.input = params_.sparse_transfer_buffers.hidden.get();
                run.routing_indices = params_.sparse_transfer_buffers.routing_indices.get();
                run.routing_weights = params_.sparse_transfer_buffers.routing_weights.get();
                run.output = params_.sparse_transfer_buffers.output.get();
                run.seq_len = static_cast<int>(dispatch_tier->token_rows.size());
                run.dispatch_token_rows = &dispatch_tier->token_rows;

                const bool ok = MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts(
                    params_.domain,
                    *params_.local_tp_context,
                    run,
                    diagnostics);
                if (!ok)
                    return false;

                recordTransferDiagnostics(diagnostics, *dispatch_tier);
                if (!MoEExpertTokenRowTransfer::scatterAddRows(
                        params_.sparse_transfer_buffers.output.get(),
                        dispatch_tier->token_rows,
                        params_.output,
                        params_.seq_len,
                        params_.d_model))
                {
                    return false;
                }

                if (diagnostics)
                {
                    if (profiling_enabled)
                        diagnostics->compute_ms = std::chrono::duration<double, std::milli>(
                                                      std::chrono::steady_clock::now() - stage_start)
                                                      .count();
                    LOG_DEBUG("[MoEExpertOverlayLocalTPStage] Domain '" << diagnostics->domain_name
                                                                        << "' executed sparse LocalTP degree=" << diagnostics->degree
                                                                        << " selected_rows=" << diagnostics->selected_token_rows.size()
                                                                        << " routed_entries=" << diagnostics->total_routed_entries);
                    MoEExpertOverlayProfiler::recordLocalTP(
                        params_.layer_idx,
                        params_.domain,
                        activeExpertCount(params_.expert_mask, params_.num_experts),
                        *diagnostics);
                }
                return true;
            }

            const bool ok = MoEExpertOverlayLocalTPExecutor::runTensorParallelExperts(
                params_.domain,
                *params_.local_tp_context,
                run,
                diagnostics);
            if (ok && dispatch_tier)
                recordTransferDiagnostics(diagnostics, *dispatch_tier);
            if (ok && diagnostics)
            {
                if (profiling_enabled)
                    diagnostics->compute_ms = std::chrono::duration<double, std::milli>(
                                                  std::chrono::steady_clock::now() - stage_start)
                                                  .count();
                LOG_DEBUG("[MoEExpertOverlayLocalTPStage] Domain '" << diagnostics->domain_name
                                                                    << "' executed LocalTP degree=" << diagnostics->degree
                                                                    << " routed_entries=" << diagnostics->total_routed_entries);
                MoEExpertOverlayProfiler::recordLocalTP(
                    params_.layer_idx,
                    params_.domain,
                    activeExpertCount(params_.expert_mask, params_.num_experts),
                    *diagnostics);
            }
            return ok;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[MoEExpertOverlayLocalTPStage] Domain '" << params_.domain.name
                                                                << "' failed: " << e.what());
            return false;
        }
    }

    size_t MoEExpertOverlayLocalTPStage::estimatedFlops() const
    {
        if (params_.seq_len <= 0 || params_.d_model <= 0 || params_.expert_intermediate <= 0)
            return 0;
        const size_t active_experts = params_.expert_mask.empty()
                                          ? static_cast<size_t>(params_.num_experts)
                                          : static_cast<size_t>(std::count(params_.expert_mask.begin(), params_.expert_mask.end(), true));
        return static_cast<size_t>(params_.seq_len) * active_experts *
               static_cast<size_t>(params_.d_model) * static_cast<size_t>(params_.expert_intermediate) * 4u;
    }

    size_t MoEExpertOverlayLocalTPStage::estimatedMemoryBytes() const
    {
        if (params_.seq_len <= 0 || params_.d_model <= 0)
            return 0;
        return static_cast<size_t>(params_.seq_len) * static_cast<size_t>(params_.d_model) * sizeof(float) * 4u;
    }

    bool MoEExpertOverlayLocalTPStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU ||
               backend == ComputeBackendType::GPU_CUDA ||
               backend == ComputeBackendType::GPU_ROCM;
    }

    StageBufferRequirements MoEExpertOverlayLocalTPStage::getBufferRequirements() const
    {
        StageBufferRequirements reqs;
        if (params_.input)
            reqs.addInput("input", params_.input->shape(), toBufferTensorType(params_.input->native_type()));
        if (params_.routing_indices)
            reqs.addInput("routing_indices", params_.routing_indices->shape(), toBufferTensorType(params_.routing_indices->native_type()));
        if (params_.routing_weights)
            reqs.addInput("routing_weights", params_.routing_weights->shape(), toBufferTensorType(params_.routing_weights->native_type()));
        if (params_.output)
            reqs.addOutput("output", params_.output->shape(), toBufferTensorType(params_.output->native_type()));
        if (params_.gate_exps)
            reqs.addWeight("gate_exps", params_.gate_exps->shape(), toBufferTensorType(params_.gate_exps->native_type()));
        if (params_.up_exps)
            reqs.addWeight("up_exps", params_.up_exps->shape(), toBufferTensorType(params_.up_exps->native_type()));
        if (params_.down_exps)
            reqs.addWeight("down_exps", params_.down_exps->shape(), toBufferTensorType(params_.down_exps->native_type()));
        return reqs;
    }

    StageDumpInfo MoEExpertOverlayLocalTPStage::buildDumpInfoImpl() const
    {
        StageDumpInfo info;
        if (params_.input)
            info.addInput("input", params_.input, params_.seq_len, params_.d_model);
        if (params_.routing_indices)
            info.addInput("routing_indices", params_.routing_indices, params_.seq_len, params_.top_k);
        if (params_.routing_weights)
            info.addInput("routing_weights", params_.routing_weights, params_.seq_len, params_.top_k);
        if (params_.gate_exps)
            info.addWeight("gate_exps", params_.gate_exps);
        if (params_.up_exps)
            info.addWeight("up_exps", params_.up_exps);
        if (params_.down_exps)
            info.addWeight("down_exps", params_.down_exps);
        if (params_.output)
            info.addOutput("output", params_.output, params_.seq_len, params_.d_model);
        info.addScalarInt("layer_idx", params_.layer_idx);
        info.addScalarInt("localtp_degree", params_.local_tp_context ? params_.local_tp_context->degree() : 0);
        info.addScalarInt("active_experts", static_cast<int>(std::count(params_.expert_mask.begin(), params_.expert_mask.end(), true)));
        info.addScalarInt("dispatch_tier_index", params_.dispatch_tier_index);
        info.addScalarBool("has_dispatch_descriptor", params_.dispatch_output != nullptr);
        if (params_.diagnostics)
        {
            info.addScalarInt("diagnostic_total_routed_entries", params_.diagnostics->total_routed_entries);
            info.addScalarInt("diagnostic_participants", static_cast<int>(params_.diagnostics->participants.size()));
            info.addScalarInt("diagnostic_selected_rows", static_cast<int>(params_.diagnostics->selected_token_rows.size()));
        }
        return info;
    }

    WorkspaceRequirements MoEExpertOverlayLocalTPStage::getWorkspaceRequirements(int m, int n, int k) const
    {
        for (const auto &participant : params_.prepared_participants)
        {
            auto requirements = firstParticipantWorkspaceRequirements(participant, m, n, k);
            if (!requirements.buffers.empty())
                return requirements;
        }
        return WorkspaceRequirements{};
    }

    void MoEExpertOverlayLocalTPStage::bindWorkspace(DeviceWorkspaceManager *workspace)
    {
        bound_workspace_ = workspace;
        owned_participant_workspaces_.clear();

        if (!workspace)
        {
            unbindWorkspace();
            return;
        }

        const size_t budget = workspace->budget();
        size_t participant_workspace_count = 0;
        for (const auto &participant : params_.prepared_participants)
        {
            DeviceWorkspaceManager *participant_workspace = workspace;
            if (participant.device != workspace->device())
            {
                auto requirements = firstParticipantWorkspaceRequirements(participant, 4096, 0, 0);
                auto owned_workspace = std::make_unique<DeviceWorkspaceManager>(participant.device, budget);
                if (!owned_workspace->allocate(requirements))
                {
                    LOG_ERROR("[MoEExpertOverlayLocalTPStage] Failed to allocate participant workspace for domain '"
                              << params_.domain.name << "' participant=" << participant.participant_index
                              << " device=" << participant.device.to_string());
                    continue;
                }
                participant_workspace = owned_workspace.get();
                owned_participant_workspaces_.push_back(std::move(owned_workspace));
                ++participant_workspace_count;
            }

            bindParticipantPreparedGemm(participant, participant_workspace);
        }

        LOG_DEBUG("[MoEExpertOverlayLocalTPStage] Bound workspaces for "
                  << params_.prepared_participants.size() << " prepared LocalTP participants in domain '"
                  << params_.domain.name << "' owned_extra_workspaces=" << participant_workspace_count);
    }

    void MoEExpertOverlayLocalTPStage::unbindWorkspace()
    {
        for (const auto &participant : params_.prepared_participants)
            bindParticipantPreparedGemm(participant, nullptr);
        owned_participant_workspaces_.clear();
        bound_workspace_ = nullptr;
    }

    bool MoEExpertOverlayLocalTPStage::hasWorkspace() const
    {
        if (!bound_workspace_)
            return false;

        bool any_bound = false;
        forEachPreparedGemm(params_, [&](ITensorGemm *gemm)
                            {
            if (any_bound || !gemm)
                return;
            auto *consumer = dynamic_cast<IWorkspaceConsumer *>(gemm);
            any_bound = consumer && consumer->hasWorkspace(); });
        return any_bound;
    }

} // namespace llaminar2