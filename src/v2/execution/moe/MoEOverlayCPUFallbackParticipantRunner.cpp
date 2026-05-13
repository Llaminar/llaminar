/**
 * @file MoEOverlayCPUFallbackParticipantRunner.cpp
 * @brief Non-root MoE overlay CPU fallback participant runner implementation.
 */

#include "MoEOverlayCPUFallbackParticipantRunner.h"

#include "execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "interfaces/IMPIContext.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        int metadataInt(const std::shared_ptr<ModelContext> &ctx, const std::string &suffix, int fallback = 0)
        {
            if (!ctx)
                return fallback;
            const auto &loader = ctx->concreteLoader();
            const std::string key = ctx->architecture() + "." + suffix;
            return loader.getInt(key, fallback);
        }

        bool fillTensor(TensorBase *tensor, float value)
        {
            if (!tensor)
                return false;
            std::fill_n(tensor->mutable_data(), tensor->numel(), value);
            return true;
        }
    } // namespace

    MoEOverlayCPUFallbackParticipantRunner::MoEOverlayCPUFallbackParticipantRunner(Config config)
        : config_(std::move(config))
    {
        if (config_.model_ctx)
            architecture_ = config_.model_ctx->architecture();
    }

    MoEOverlayCPUFallbackParticipantRunner::~MoEOverlayCPUFallbackParticipantRunner() = default;

    bool MoEOverlayCPUFallbackParticipantRunner::initialize()
    {
        if (initialized_)
            return true;

        if (!config_.model_ctx)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing model context");
            return false;
        }
        if (!config_.mpi_ctx)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing MPI context");
            return false;
        }
        if (!config_.overlay_plan || !config_.overlay_plan->isTieredOverlay())
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing tiered overlay plan");
            return false;
        }
        if (!config_.execution_plan)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing overlay execution plan");
            return false;
        }

        const auto &rank_plan = config_.execution_plan->currentRankPlan();
        if (!rank_plan.hasRole(OverlayRankRole::CpuFallbackParticipant) || rank_plan.builds_root_graph)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Rank " << rank_plan.world_rank
                                                                       << " is not a non-root CPU fallback participant");
            return false;
        }

        d_model_ = config_.model_ctx->embeddingLength();
        num_experts_ = metadataInt(config_.model_ctx, "expert_count", 0);
        top_k_ = metadataInt(config_.model_ctx, "expert_used_count", 0);
        expert_intermediate_ = metadataInt(config_.model_ctx, "expert_feed_forward_length", 0);
        if (expert_intermediate_ == 0)
            expert_intermediate_ = config_.model_ctx->feedForwardLength();

        if (d_model_ <= 0 || num_experts_ <= 0 || top_k_ <= 0)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Invalid MoE metadata: d_model=" << d_model_
                                                                                                << " num_experts=" << num_experts_
                                                                                                << " top_k=" << top_k_
                                                                                                << " expert_intermediate=" << expert_intermediate_);
            return false;
        }

        cpu_ctx_ = IDeviceContext::create(DeviceId::cpu());
        if (!cpu_ctx_)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Failed to create CPU device context");
            return false;
        }

        if (auto weight_manager = config_.model_ctx->concreteWeightManager())
            prepared_store_ = weight_manager->preparedWeightStore();

        initialized_ = true;
        LOG_INFO("[MoEOverlayCPUFallbackParticipantRunner] Rank " << config_.mpi_ctx->rank()
                                                                  << " initialized as graph-native CPU fallback participant"
                                                                  << " d_model=" << d_model_
                                                                  << " num_experts=" << num_experts_
                                                                  << " top_k=" << top_k_
                                                                  << " layers=" << config_.model_ctx->totalBlockCount());
        return true;
    }

    int MoEOverlayCPUFallbackParticipantRunner::vocab_size() const
    {
        return config_.model_ctx ? config_.model_ctx->vocabSize() : 0;
    }

    const char *MoEOverlayCPUFallbackParticipantRunner::architecture() const
    {
        return architecture_.c_str();
    }

    void MoEOverlayCPUFallbackParticipantRunner::clear_cache()
    {
        position_ = 0;
    }

    const ExpertLayerPlacement *MoEOverlayCPUFallbackParticipantRunner::placementForLayer(int layer_idx) const
    {
        auto it = std::find_if(config_.overlay_plan->placements.begin(),
                               config_.overlay_plan->placements.end(),
                               [layer_idx](const ExpertLayerPlacement &placement)
                               {
                                   return placement.layer == layer_idx;
                               });
        return it == config_.overlay_plan->placements.end() ? nullptr : &(*it);
    }

    const ExpertComputeDomain *MoEOverlayCPUFallbackParticipantRunner::domainForName(const std::string &domain_name) const
    {
        auto it = std::find_if(config_.overlay_plan->domains.begin(),
                               config_.overlay_plan->domains.end(),
                               [&](const ExpertComputeDomain &domain)
                               {
                                   return domain.name == domain_name;
                               });
        return it == config_.overlay_plan->domains.end() ? nullptr : &(*it);
    }

    int MoEOverlayCPUFallbackParticipantRunner::cpuFallbackTierIndex() const
    {
        for (size_t tier_index = 0; tier_index < config_.overlay_plan->routed_tiers.size(); ++tier_index)
        {
            const auto &tier = config_.overlay_plan->routed_tiers[tier_index];
            const auto *domain = domainForName(tier.domain);
            if (tier.fallback && domain &&
                domain->kind == ExpertDomainKind::NodeLocalTP &&
                domain->compute_kind == ExpertDomainComputeKind::TensorParallelExperts)
            {
                return static_cast<int>(tier_index);
            }
        }
        return -1;
    }

    std::vector<bool> MoEOverlayCPUFallbackParticipantRunner::expertMaskForTier(
        const ExpertLayerPlacement &placement,
        int tier_index) const
    {
        std::vector<bool> mask(static_cast<size_t>(num_experts_), false);
        if (static_cast<int>(placement.routed_expert_tier.size()) != num_experts_)
            return mask;

        for (int expert = 0; expert < num_experts_; ++expert)
            mask[static_cast<size_t>(expert)] = placement.routed_expert_tier[static_cast<size_t>(expert)] == tier_index;
        return mask;
    }

    bool MoEOverlayCPUFallbackParticipantRunner::hasActiveExpertMask(const std::vector<bool> &mask) const
    {
        return std::any_of(mask.begin(), mask.end(), [](bool enabled)
                           { return enabled; });
    }

    bool MoEOverlayCPUFallbackParticipantRunner::isMoELayer(int layer_idx) const
    {
        const std::string name = "blk." + std::to_string(layer_idx) + ".ffn_gate_inp.weight";
        return config_.model_ctx && config_.model_ctx->hasTensor(name);
    }

    bool MoEOverlayCPUFallbackParticipantRunner::loadLayerWeights(
        int layer_idx,
        LayerFallbackWeights &weights)
    {
        auto weight_manager = config_.model_ctx ? config_.model_ctx->concreteWeightManager() : nullptr;
        if (!weight_manager)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing concrete WeightManager");
            return false;
        }

        const std::string prefix = "blk." + std::to_string(layer_idx) + ".";
        weights.gate = weight_manager->getWeightForDevice(prefix + "ffn_gate_exps.weight", DeviceId::cpu(), layer_idx);
        weights.up = weight_manager->getWeightForDevice(prefix + "ffn_up_exps.weight", DeviceId::cpu(), layer_idx);
        weights.down = weight_manager->getWeightForDevice(prefix + "ffn_down_exps.weight", DeviceId::cpu(), layer_idx);

        if (!weights.gate || !weights.up || !weights.down)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing CPU fallback expert weights for layer "
                      << layer_idx);
            return false;
        }
        return true;
    }

    MoEOverlayCPUFallbackParticipantRunner::LayerFallbackWeights *
    MoEOverlayCPUFallbackParticipantRunner::cachedLayerWeights(int layer_idx)
    {
        auto existing = layer_weights_.find(layer_idx);
        if (existing != layer_weights_.end())
            return &existing->second;

        LayerFallbackWeights weights;
        if (!loadLayerWeights(layer_idx, weights))
            return nullptr;

        auto [inserted, _] = layer_weights_.emplace(layer_idx, std::move(weights));
        return &inserted->second;
    }

    bool MoEOverlayCPUFallbackParticipantRunner::ensureScratch(int seq_len)
    {
        if (seq_len <= 0)
            return false;

        if (scratch_.seq_len != seq_len)
        {
            scratch_.input = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
            scratch_.routing_indices = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(top_k_)});
            scratch_.routing_weights = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(top_k_)});
            scratch_.output = std::make_shared<FP32Tensor>(
                std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d_model_)});
            scratch_.seq_len = seq_len;
        }

        return fillTensor(scratch_.input.get(), 0.0f) &&
               fillTensor(scratch_.routing_indices.get(), -1.0f) &&
               fillTensor(scratch_.routing_weights.get(), 0.0f) &&
               fillTensor(scratch_.output.get(), 0.0f);
    }

    int MoEOverlayCPUFallbackParticipantRunner::expertIntermediate(const LayerFallbackWeights &weights) const
    {
        if (expert_intermediate_ > 0)
            return expert_intermediate_;
        if (!weights.gate || num_experts_ <= 0)
            return 0;
        return static_cast<int>(weights.gate->rows() / static_cast<size_t>(num_experts_));
    }

    bool MoEOverlayCPUFallbackParticipantRunner::executeLayerFallback(
        int layer_idx,
        int tier_index,
        const ExpertComputeDomain &domain,
        const std::vector<bool> &expert_mask,
        const LayerFallbackWeights &weights)
    {
        MoEExpertOverlayCPUFallbackStage::Params params;
        params.device_id = DeviceId::cpu();
        params.mpi_ctx = config_.mpi_ctx.get();
        params.domain = domain;
        params.root_world_rank = config_.execution_plan->continuation_root_rank;
        params.domain_id = MoEExpertOverlayCPUFallback::stableDomainId(domain.name);
        params.hostfile_path = config_.hostfile_path;
        params.input = scratch_.input.get();
        params.routing_indices = scratch_.routing_indices.get();
        params.routing_weights = scratch_.routing_weights.get();
        params.gate_exps = weights.gate.get();
        params.up_exps = weights.up.get();
        params.down_exps = weights.down.get();
        params.output = scratch_.output.get();
        params.seq_len = scratch_.seq_len;
        params.d_model = d_model_;
        params.num_experts = num_experts_;
        params.top_k = top_k_;
        params.expert_intermediate = expertIntermediate(weights);
        params.layer_idx = layer_idx;
        params.expert_mask = expert_mask;
        params.transfer_mode = MoEExpertTransferMode::Auto;
        params.dispatch_tier_index = tier_index;
        params.prepared_store = prepared_store_.get();

        MoEExpertOverlayCPUFallbackStage stage(std::move(params));
        return stage.execute(cpu_ctx_.get());
    }

    bool MoEOverlayCPUFallbackParticipantRunner::forward(const int *tokens, int seq_len)
    {
        (void)tokens;
        if (!initialized_ && !initialize())
            return false;

        if (!config_.enable_native_compatibility_fallback)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Rank " << config_.mpi_ctx->rank()
                                                                       << " cannot execute graph-native CPU fallback endpoint work without an overlay dispatch backend; "
                                                                       << "refusing native NodeLocalTP compatibility loop to avoid unmatched GlobalTP collectives");
            return false;
        }

        if (!ensureScratch(seq_len))
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Failed to allocate scratch for seq_len=" << seq_len);
            return false;
        }

        const int tier_index = cpuFallbackTierIndex();
        if (tier_index < 0)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] No NodeLocalTP CPU fallback tier found");
            return false;
        }
        const auto &tier = config_.overlay_plan->routed_tiers[static_cast<size_t>(tier_index)];
        const auto *domain = domainForName(tier.domain);
        if (!domain)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] CPU fallback tier references missing domain '"
                      << tier.domain << "'");
            return false;
        }

        for (int layer_idx = 0; layer_idx < config_.model_ctx->totalBlockCount(); ++layer_idx)
        {
            if (!isMoELayer(layer_idx))
                continue;

            const auto *placement = placementForLayer(layer_idx);
            if (!placement)
                continue;

            auto mask = expertMaskForTier(*placement, tier_index);
            if (!hasActiveExpertMask(mask))
                continue;

            auto *weights = cachedLayerWeights(layer_idx);
            if (!weights)
                return false;

            if (!executeLayerFallback(layer_idx, tier_index, *domain, mask, *weights))
            {
                LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] CPU fallback participant failed at layer "
                          << layer_idx << " tier " << tier_index << " domain '" << domain->name << "'");
                return false;
            }
        }

        position_ += seq_len;
        return true;
    }

    std::unique_ptr<IInferenceRunner> createMoEOverlayCPUFallbackParticipantRunner(
        MoEOverlayCPUFallbackParticipantRunner::Config config)
    {
        auto runner = std::make_unique<MoEOverlayCPUFallbackParticipantRunner>(std::move(config));
        if (!runner->initialize())
            return nullptr;
        return runner;
    }

} // namespace llaminar2