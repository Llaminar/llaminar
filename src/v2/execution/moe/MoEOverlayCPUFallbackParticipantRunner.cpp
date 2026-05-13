/**
 * @file MoEOverlayCPUFallbackParticipantRunner.cpp
 * @brief Non-root MoE overlay CPU fallback participant runner implementation.
 */

#include "MoEOverlayCPUFallbackParticipantRunner.h"

#include "execution/compute_stages/stages/MoEExpertOverlayCPUFallbackStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/moe/MoEOverlayMPIDispatchBackend.h"
#include "interfaces/IMPIContext.h"
#include "loaders/PreparedWeightStore.h"
#include "tensors/Tensors.h"
#include "utils/Logger.h"

#include <algorithm>
#include <sstream>
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

    std::shared_ptr<const MoECPUFallbackDomainContext>
    MoEOverlayCPUFallbackParticipantRunner::domainContextFor(const ExpertComputeDomain &domain)
    {
        const int root_world_rank = config_.execution_plan ? config_.execution_plan->continuation_root_rank : 0;
        const int domain_id = MoEExpertOverlayCPUFallback::stableDomainId(domain.name);

        std::ostringstream key;
        key << domain.name << ":" << domain_id << ":" << root_world_rank << ":";
        const auto world_ranks = MoEExpertOverlayCPUFallback::domainWorldRanks(domain);
        for (size_t index = 0; index < world_ranks.size(); ++index)
        {
            if (index != 0)
                key << ",";
            key << world_ranks[index];
        }

        const std::string cache_key = key.str();
        auto existing = domain_contexts_.find(cache_key);
        if (existing != domain_contexts_.end())
            return existing->second;

        MoECPUFallbackDomainConfig domain_config;
        domain_config.domain = domain;
        domain_config.world_comm = config_.mpi_ctx && config_.mpi_ctx->communicator() != MPI_COMM_NULL
                                       ? config_.mpi_ctx->communicator()
                                       : MPI_COMM_WORLD;
        domain_config.root_world_rank = root_world_rank;
        domain_config.domain_id = domain_id;
        domain_config.hostfile_path = config_.hostfile_path;

        auto context = std::make_shared<MoECPUFallbackDomainContext>(
            MoEExpertOverlayCPUFallback::createNodeLocalTPDomain(domain_config));
        domain_contexts_.emplace(cache_key, context);
        LOG_INFO("[MoEOverlayCPUFallbackParticipantRunner] Cached CPU fallback domain context domain='"
                 << domain.name << "' root_rank=" << root_world_rank);
        return context;
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

    bool MoEOverlayCPUFallbackParticipantRunner::runDispatchEndpointPump(int seq_len)
    {
        if (!config_.dispatch_backend)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing graph-native dispatch backend");
            return false;
        }

        while (true)
        {
            MoEOverlayMPIDispatchHeader header;
            std::string error;
            if (!config_.dispatch_backend->receiveHeader(header, &error))
            {
                LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Failed to receive dispatch message: " << error);
                return false;
            }

            switch (header.kind)
            {
            case MoEOverlayMPIMessageKind::NoOp:
                LOG_DEBUG("[MoEOverlayCPUFallbackParticipantRunner] No-op dispatch"
                          << " domain_id=" << header.domain_id
                          << " layer=" << header.layer_id
                          << " tier=" << header.tier_index
                          << " group=" << header.dispatch_group_id);
                continue;

            case MoEOverlayMPIMessageKind::RoutedWork:
            {
                if (header.layer_id < 0 || header.tier_index < 0 ||
                    header.tier_index >= static_cast<int>(config_.overlay_plan->routed_tiers.size()))
                {
                    LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Invalid routed dispatch envelope"
                              << " domain_id=" << header.domain_id
                              << " layer=" << header.layer_id
                              << " tier=" << header.tier_index
                              << " group=" << header.dispatch_group_id);
                    return false;
                }

                const auto *placement = placementForLayer(header.layer_id);
                if (!placement)
                {
                    LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Missing expert placement for routed dispatch"
                              << " layer=" << header.layer_id
                              << " tier=" << header.tier_index);
                    return false;
                }

                const auto &tier = config_.overlay_plan->routed_tiers[static_cast<size_t>(header.tier_index)];
                const auto *domain = domainForName(tier.domain);
                if (!domain)
                {
                    LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Routed dispatch references missing domain '"
                              << tier.domain << "' layer=" << header.layer_id
                              << " tier=" << header.tier_index);
                    return false;
                }
                if (domain->kind != ExpertDomainKind::NodeLocalTP)
                {
                    LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Routed dispatch domain '"
                              << domain->name << "' is not a NodeLocalTP CPU fallback domain");
                    return false;
                }

                auto mask = expertMaskForTier(*placement, header.tier_index);
                if (!hasActiveExpertMask(mask))
                {
                    LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Routed dispatch has no active CPU-cold experts"
                              << " layer=" << header.layer_id
                              << " tier=" << header.tier_index
                              << " domain='" << domain->name << "'");
                    return false;
                }

                auto *weights = cachedLayerWeights(header.layer_id);
                if (!weights)
                    return false;

                LOG_DEBUG("[MoEOverlayCPUFallbackParticipantRunner] Routed dispatch envelope"
                          << " domain_id=" << header.domain_id
                          << " layer=" << header.layer_id
                          << " tier=" << header.tier_index
                          << " group=" << header.dispatch_group_id
                          << " selected_rows=" << header.selected_row_count
                          << " routed_entries=" << header.routed_entry_count
                          << " transfer_bytes=" << header.transfer_bytes);

                if (!executeLayerFallback(header.layer_id, header.tier_index, *domain, mask, *weights))
                {
                    LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] CPU fallback dispatch envelope failed"
                              << " layer=" << header.layer_id
                              << " tier=" << header.tier_index
                              << " domain='" << domain->name << "'");
                    return false;
                }
                continue;
            }

            case MoEOverlayMPIMessageKind::Cancel:
                LOG_WARN("[MoEOverlayCPUFallbackParticipantRunner] Dispatch canceled"
                         << " domain_id=" << header.domain_id
                         << " layer=" << header.layer_id
                         << " tier=" << header.tier_index
                         << " reason=" << header.cancel_reason_code);
                return false;

            case MoEOverlayMPIMessageKind::ForwardDone:
                position_ += seq_len;
                return true;

            case MoEOverlayMPIMessageKind::Shutdown:
                LOG_INFO("[MoEOverlayCPUFallbackParticipantRunner] Received dispatch shutdown");
                return true;
            }
        }
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
        params.domain_context = domainContextFor(domain);
        if (!params.domain_context)
            return false;

        MoEExpertOverlayCPUFallbackStage stage(std::move(params));
        return stage.execute(cpu_ctx_.get());
    }

    bool MoEOverlayCPUFallbackParticipantRunner::forward(const int *tokens, int seq_len)
    {
        (void)tokens;
        if (!initialized_ && !initialize())
            return false;

        if (!config_.dispatch_backend)
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Rank " << config_.mpi_ctx->rank()
                                                                       << " requires a graph-native overlay dispatch backend; "
                                                                       << "endpoint ranks must consume dispatch envelopes and cannot run an independent all-layer fallback loop");
            return false;
        }

        if (!ensureScratch(seq_len))
        {
            LOG_ERROR("[MoEOverlayCPUFallbackParticipantRunner] Failed to allocate dispatch scratch for seq_len=" << seq_len);
            return false;
        }
        return runDispatchEndpointPump(seq_len);
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