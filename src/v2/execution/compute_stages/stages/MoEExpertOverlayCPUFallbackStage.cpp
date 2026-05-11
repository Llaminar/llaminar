/**
 * @file MoEExpertOverlayCPUFallbackStage.cpp
 * @brief Graph-integrated NodeLocalTP CPU fallback stage for MoE expert overlays.
 */

#include "MoEExpertOverlayCPUFallbackStage.h"

#include "MoEExpertDispatchStage.h"
#include "../../../execution/moe/MoEExpertOverlayProfiler.h"
#include "../../../interfaces/IMPIContext.h"
#include "../../../tensors/Tensors.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/Logger.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MPI_Comm resolveWorldComm(const IMPIContext *mpi_ctx)
        {
            if (mpi_ctx && mpi_ctx->communicator() != MPI_COMM_NULL)
                return mpi_ctx->communicator();
            return MPI_COMM_WORLD;
        }

        bool validateShape2D(const TensorBase *tensor, int rows, int cols, const char *name)
        {
            if (!tensor)
            {
                LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Null " << name << " tensor");
                return false;
            }
            const auto &shape = tensor->shape();
            if (shape.size() != 2 || shape[0] < static_cast<size_t>(rows) || shape[1] != static_cast<size_t>(cols))
            {
                LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] " << name << " shape mismatch: got rank "
                                                                << shape.size() << ", expected at least [" << rows << ", " << cols << "]");
                return false;
            }
            return true;
        }

        const MoEExpertTierDispatch *resolveDispatchTier(
            const MoEExpertOverlayCPUFallbackStage::Params &params)
        {
            if (!params.dispatch_output)
                return nullptr;
            if (params.dispatch_tier_index < 0 ||
                params.dispatch_tier_index >= static_cast<int>(params.dispatch_output->tiers.size()))
            {
                LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Dispatch tier index "
                          << params.dispatch_tier_index << " is outside descriptor tier count "
                          << params.dispatch_output->tiers.size());
                return nullptr;
            }

            if (params.dispatch_output->seq_len != params.seq_len ||
                params.dispatch_output->top_k != params.top_k ||
                params.dispatch_output->d_model != params.d_model)
            {
                LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Dispatch descriptor dimension mismatch for domain '"
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
                LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Dispatch tier domain '" << tier.domain
                                                                                      << "' does not match CPU fallback domain '" << params.domain.name << "'");
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

        void traceDescriptorConsumption(
            const MoEExpertOverlayCPUFallbackStage::Params &params,
            const MoEExpertTierDispatch &tier)
        {
            const auto &env = debugEnv();
            if (!env.moe_expert_overlay.transfer_trace && !env.moe_expert_overlay.trace && !env.profile.enabled)
                return;

            LOG_INFO("[MoEExpertOverlayCPUFallbackStage] layer=" << params.layer_idx
                                                                 << " domain=" << params.domain.name
                                                                 << " tier=" << tier.tier_index
                                                                 << " selected_rows=" << tier.token_rows.size()
                                                                 << " routed_entries=" << tier.entries.size()
                                                                 << " mode=" << toString(tier.transfer_mode)
                                                                 << " outbound_bytes=" << tier.transfer_volume.outbound_bytes
                                                                 << " return_bytes=" << tier.transfer_volume.return_bytes);
        }

        int activeExpertCount(const std::vector<bool> &mask, int num_experts)
        {
            if (mask.empty())
                return num_experts;
            return static_cast<int>(std::count(mask.begin(), mask.end(), true));
        }
    } // namespace

    MoEExpertOverlayCPUFallbackStage::MoEExpertOverlayCPUFallbackStage(Params params)
        : IComputeStage(params.device_id), params_(std::move(params))
    {
        if (!params_.dispatch_output && params_.dispatch_output_lifetime)
            params_.dispatch_output = params_.dispatch_output_lifetime.get();
    }

    bool MoEExpertOverlayCPUFallbackStage::execute(IDeviceContext *ctx)
    {
        if (!ctx)
        {
            LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Null device context");
            return false;
        }
        if (params_.device_id != DeviceId::cpu())
        {
            LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] CPU fallback stage must execute on CPU, got "
                      << params_.device_id.to_string());
            return false;
        }
        if (params_.domain.kind != ExpertDomainKind::NodeLocalTP)
        {
            LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Domain '" << params_.domain.name
                                                                    << "' must be NodeLocalTP, got " << toString(params_.domain.kind));
            return false;
        }
        if (params_.seq_len <= 0 || params_.d_model <= 0 || params_.num_experts <= 0 ||
            params_.top_k <= 0 || params_.expert_intermediate <= 0)
        {
            LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Invalid dimensions for domain '"
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

        try
        {
            const bool profiling_enabled = MoEExpertOverlayProfiler::isEnabled();
            MoECPUFallbackTensorParallelStats local_tensor_parallel_stats;
            MoECPUFallbackTransferStats local_transfer_stats;
            const MoEExpertTierDispatch *dispatch_tier = nullptr;

            MoECPUFallbackDomainConfig domain_config;
            domain_config.domain = params_.domain;
            domain_config.world_comm = resolveWorldComm(params_.mpi_ctx);
            domain_config.root_world_rank = params_.root_world_rank;
            domain_config.domain_id = params_.domain_id;
            domain_config.hostfile_path = params_.hostfile_path;

            auto domain = MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(domain_config);

            MoECPUFallbackRunParams run;
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
            run.fallback_expert_mask = params_.expert_mask;
            run.transfer_mode = params_.transfer_mode;
            run.transfer_token_rows = params_.transfer_token_rows;
            if (params_.dispatch_output)
            {
                dispatch_tier = resolveDispatchTier(params_);
                if (!dispatch_tier)
                    return false;
                traceDescriptorConsumption(params_, *dispatch_tier);
                run.transfer_mode = dispatch_tier->transfer_mode;
                run.transfer_token_rows = dispatch_tier->token_rows;
                if (dispatch_tier->transfer_mode == MoEExpertTransferMode::SparseTokenRows ||
                    dispatch_tier->transfer_mode == MoEExpertTransferMode::DecodeOneToken)
                {
                    clearOutput(params_.output);
                }
            }
            run.prepared_store = params_.prepared_store;
            run.tensor_parallel_stats = params_.tensor_parallel_stats ? params_.tensor_parallel_stats : (profiling_enabled ? &local_tensor_parallel_stats : nullptr);
            run.transfer_stats = params_.transfer_stats ? params_.transfer_stats : (profiling_enabled ? &local_transfer_stats : nullptr);

            const auto start = profiling_enabled ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
            const bool ok = MoEExpertOverlayCPUFallback::runExpertFallback(domain, run, ctx);
            if (ok && profiling_enabled)
            {
                const double compute_ms = std::chrono::duration<double, std::milli>(
                                              std::chrono::steady_clock::now() - start)
                                              .count();
                MoEExpertOverlayProfiler::recordCPUFallback(
                    params_.layer_idx,
                    params_.domain,
                    activeExpertCount(params_.expert_mask, params_.num_experts),
                    dispatch_tier ? dispatch_tier->entries.size() : 0,
                    dispatch_tier ? dispatch_tier->token_rows.size() : params_.transfer_token_rows.size(),
                    run.transfer_stats,
                    run.tensor_parallel_stats,
                    compute_ms);
            }
            return ok;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("[MoEExpertOverlayCPUFallbackStage] Domain '" << params_.domain.name
                                                                    << "' failed: " << e.what());
            return false;
        }
    }

    size_t MoEExpertOverlayCPUFallbackStage::estimatedFlops() const
    {
        if (params_.seq_len <= 0 || params_.d_model <= 0 || params_.expert_intermediate <= 0)
            return 0;
        const size_t active_experts = params_.expert_mask.empty()
                                          ? static_cast<size_t>(params_.num_experts)
                                          : static_cast<size_t>(std::count(params_.expert_mask.begin(), params_.expert_mask.end(), true));
        return static_cast<size_t>(params_.seq_len) * active_experts *
               static_cast<size_t>(params_.d_model) * static_cast<size_t>(params_.expert_intermediate) * 4u;
    }

    size_t MoEExpertOverlayCPUFallbackStage::estimatedMemoryBytes() const
    {
        if (params_.seq_len <= 0 || params_.d_model <= 0)
            return 0;
        return static_cast<size_t>(params_.seq_len) * static_cast<size_t>(params_.d_model) * sizeof(float) * 4u;
    }

    bool MoEExpertOverlayCPUFallbackStage::supportsBackend(ComputeBackendType backend) const
    {
        return backend == ComputeBackendType::CPU;
    }

    StageBufferRequirements MoEExpertOverlayCPUFallbackStage::getBufferRequirements() const
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

    StageDumpInfo MoEExpertOverlayCPUFallbackStage::buildDumpInfoImpl() const
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
        info.addScalarInt("root_world_rank", params_.root_world_rank);
        info.addScalarInt("domain_id", params_.domain_id);
        info.addScalarInt("active_experts", static_cast<int>(std::count(params_.expert_mask.begin(), params_.expert_mask.end(), true)));
        info.addScalarInt("dispatch_tier_index", params_.dispatch_tier_index);
        info.addScalarBool("has_dispatch_descriptor", params_.dispatch_output != nullptr);
        return info;
    }

} // namespace llaminar2
