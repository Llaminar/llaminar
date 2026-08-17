/**
 * @file MoERebalanceController.cpp
 * @brief Orchestrates MoE decode histogram tracking and socket-aware rebalancing
 */

#include "MoERebalanceController.h"
#include "../../utils/Logger.h"
#include "DeviceMoERebalancePolicyShared.h"
#include "fort.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <iomanip>

namespace llaminar2
{
    namespace
    {
        ExpertLoadImbalanceStats imbalanceStatsFromParticipantLoads(
            const std::vector<std::vector<uint64_t>> &layer_loads)
        {
            ExpertLoadImbalanceStats stats;
            stats.layer_count = static_cast<int>(layer_loads.size());

            double ratio_sum = 0.0;
            double spread_sum = 0.0;
            for (int layer_idx = 0; layer_idx < static_cast<int>(layer_loads.size()); ++layer_idx)
            {
                const auto &loads = layer_loads[static_cast<size_t>(layer_idx)];
                if (loads.empty())
                    continue;

                const uint64_t total =
                    std::accumulate(loads.begin(), loads.end(), uint64_t{0});
                if (total == 0)
                    continue;

                ++stats.active_layer_count;
                stats.total_activations += total;

                const auto [min_it, max_it] = std::minmax_element(loads.begin(), loads.end());
                const uint64_t min_load = *min_it;
                const uint64_t max_load = *max_it;
                const double spread =
                    max_load > 0
                        ? static_cast<double>(max_load - min_load) / static_cast<double>(max_load)
                        : 0.0;
                spread_sum += spread;

                double ratio = 1.0;
                if (min_load == 0 && max_load > 0)
                {
                    ratio = std::numeric_limits<double>::infinity();
                    ++stats.infinite_ratio_layers;
                }
                else if (min_load > 0)
                {
                    ratio = static_cast<double>(max_load) / static_cast<double>(min_load);
                    ratio_sum += ratio;
                }
                else
                {
                    ratio_sum += ratio;
                }

                if (stats.worst_layer < 0 || spread > stats.worst_spread)
                {
                    stats.worst_spread = spread;
                    stats.worst_ratio = ratio;
                    stats.worst_layer = layer_idx;
                }
            }

            stats.valid = stats.active_layer_count > 0;
            if (!stats.valid)
                return stats;

            stats.average_spread =
                spread_sum / static_cast<double>(stats.active_layer_count);
            if (stats.infinite_ratio_layers > 0)
            {
                stats.average_ratio = std::numeric_limits<double>::infinity();
                stats.worst_ratio = std::numeric_limits<double>::infinity();
            }
            else
            {
                stats.average_ratio =
                    ratio_sum / static_cast<double>(stats.active_layer_count);
            }

            return stats;
        }
    }

    const char *toString(MoERebalanceDecisionReason reason)
    {
        switch (reason)
        {
        case MoERebalanceDecisionReason::ModeOff:
            return "mode_off";
        case MoERebalanceDecisionReason::DynamicDisabledForDomain:
            return "dynamic_disabled_for_domain";
        case MoERebalanceDecisionReason::SingleParticipantObserveOnly:
            return "single_participant_observe_only";
        case MoERebalanceDecisionReason::WindowNotFull:
            return "window_not_full";
        case MoERebalanceDecisionReason::Ready:
            return "ready";
        }
        return "unknown";
    }

    MoERebalanceController *selectActiveMoERebalanceController(
        const std::vector<MoERebalanceController *> &controllers)
    {
        auto first_valid = std::find_if(
            controllers.begin(),
            controllers.end(),
            [](const MoERebalanceController *controller)
            {
                return controller != nullptr;
            });
        if (first_valid == controllers.end())
            return nullptr;

        auto routed_multi_participant = std::find_if(
            controllers.begin(),
            controllers.end(),
            [](const MoERebalanceController *controller)
            {
                return controller &&
                       controller->domainId() != "single" &&
                       controller->participantCount() > 1;
            });
        if (routed_multi_participant != controllers.end())
            return *routed_multi_participant;

        auto any_multi_participant = std::find_if(
            controllers.begin(),
            controllers.end(),
            [](const MoERebalanceController *controller)
            {
                return controller && controller->participantCount() > 1;
            });
        if (any_multi_participant != controllers.end())
            return *any_multi_participant;

        auto routed_hot_cache = std::find_if(
            controllers.begin(),
            controllers.end(),
            [](const MoERebalanceController *controller)
            {
                return controller &&
                       controller->domainId() != "single" &&
                       controller->maxReplicasPerSocket() > 0;
            });
        if (routed_hot_cache != controllers.end())
            return *routed_hot_cache;

        return *first_valid;
    }

    MoERebalanceController *selectActiveMoERebalanceController(
        const std::vector<std::unique_ptr<MoERebalanceController>> &controllers)
    {
        std::vector<MoERebalanceController *> pointers;
        pointers.reserve(controllers.size());
        for (const auto &controller : controllers)
            pointers.push_back(controller.get());
        return selectActiveMoERebalanceController(pointers);
    }

    // =========================================================================
    // ExpertReplicaSet — deterministic per-token dispatch
    // =========================================================================

    bool ExpertReplicaSet::hasLayerReplicaPlacement() const
    {
        return !base_ownership.empty() &&
               replica_participants_by_layer.size() ==
                   static_cast<size_t>(base_ownership.layerCount());
    }

    bool ExpertReplicaSet::hasReplicaOnParticipant(
        int layer_idx,
        int expert_id,
        int participant_id) const
    {
        if (!hasLayerReplicaPlacement() ||
            layer_idx < 0 || layer_idx >= base_ownership.layerCount() ||
            expert_id < 0 || expert_id >= base_ownership.expertCount() ||
            participant_id < 0 || participant_id >= participantCount())
        {
            throw std::out_of_range(
                "ExpertReplicaSet replica lookup is outside its layered ownership geometry");
        }

        return replica_participants_by_layer[static_cast<size_t>(layer_idx)]
                                            [static_cast<size_t>(expert_id)]
                                            [static_cast<size_t>(participant_id)];
    }

    bool ExpertReplicaSet::isReplicatedForLayer(int layer_idx, int expert_id) const
    {
        if (!hasLayerReplicaPlacement() ||
            layer_idx < 0 || layer_idx >= base_ownership.layerCount() ||
            expert_id < 0 || expert_id >= base_ownership.expertCount())
        {
            throw std::out_of_range(
                "ExpertReplicaSet replica query is outside its layered ownership geometry");
        }

        const auto &participants =
            replica_participants_by_layer[static_cast<size_t>(layer_idx)]
                                         [static_cast<size_t>(expert_id)];
        return std::any_of(
            participants.begin(), participants.end(), [](bool resident)
            { return resident; });
    }

    void ExpertReplicaSet::setReplicaOnParticipant(
        int layer_idx,
        int expert_id,
        int participant_id,
        bool enabled)
    {
        if (!hasLayerReplicaPlacement() ||
            layer_idx < 0 || layer_idx >= base_ownership.layerCount() ||
            expert_id < 0 || expert_id >= base_ownership.expertCount() ||
            participant_id < 0 || participant_id >= participantCount())
        {
            throw std::out_of_range(
                "ExpertReplicaSet mutation is outside its layered ownership geometry");
        }
        if (enabled &&
            base_ownership.owner(layer_idx, expert_id) == participant_id)
        {
            throw std::invalid_argument(
                "ExpertReplicaSet cannot advertise an owner as its own replica");
        }

        auto &participants =
            replica_participants_by_layer[static_cast<size_t>(layer_idx)]
                                         [static_cast<size_t>(expert_id)];
        participants[static_cast<size_t>(participant_id)] = enabled;
    }

    void ExpertReplicaSet::rebuildAggregateReplicaFlags()
    {
        if (base_ownership.empty() && replica_participants_by_layer.empty())
        {
            replicated_in_any_layer.clear();
            num_replicated = 0;
            return;
        }
        if (!hasLayerReplicaPlacement())
        {
            throw std::invalid_argument(
                "ExpertReplicaSet layer count does not match its base ownership");
        }

        replicated_in_any_layer.assign(
            static_cast<size_t>(base_ownership.expertCount()), false);
        num_replicated = 0;
        for (int layer_idx = 0; layer_idx < base_ownership.layerCount(); ++layer_idx)
        {
            const auto &layer =
                replica_participants_by_layer[static_cast<size_t>(layer_idx)];
            if (layer.size() != static_cast<size_t>(base_ownership.expertCount()))
            {
                throw std::invalid_argument(
                    "ExpertReplicaSet expert count does not match its base ownership");
            }

            for (int expert_id = 0;
                 expert_id < base_ownership.expertCount();
                 ++expert_id)
            {
                const auto &participants = layer[static_cast<size_t>(expert_id)];
                if (participants.size() != static_cast<size_t>(participantCount()))
                {
                    throw std::invalid_argument(
                        "ExpertReplicaSet participant count does not match its base ownership");
                }
                for (int participant = 0;
                     participant < participantCount();
                     ++participant)
                {
                    if (!participants[static_cast<size_t>(participant)])
                        continue;
                    if (base_ownership.owner(layer_idx, expert_id) == participant)
                    {
                        throw std::invalid_argument(
                            "ExpertReplicaSet contains an owner-as-replica entry");
                    }

                    replicated_in_any_layer[static_cast<size_t>(expert_id)] = true;
                    ++num_replicated;
                }
            }
        }
    }

    void ExpertReplicaSet::assignForToken(
        const int *expert_indices,
        const float *expert_weights,
        int top_k,
        int my_socket_id,
        const std::vector<bool> &expert_mask,
        bool *compute_here,
        int layer_idx) const
    {
        constexpr int kMaxRoutedExpertsPerToken = 16;
        if (!expert_indices || !expert_weights || !compute_here ||
            top_k < 0 || top_k > kMaxRoutedExpertsPerToken)
        {
            throw std::invalid_argument(
                "ExpertReplicaSet::assignForToken requires complete route arrays and top_k in [0, 16]");
        }

        std::array<bool, kMaxRoutedExpertsPerToken> active_route{};
        for (int k = 0; k < top_k; ++k)
        {
            compute_here[k] = false;
            if (!std::isfinite(expert_weights[k]))
            {
                throw std::invalid_argument(
                    "ExpertReplicaSet::assignForToken received a non-finite route weight");
            }
            active_route[static_cast<size_t>(k)] = expert_weights[k] != 0.0f;
        }

        if (num_replicated == 0)
        {
            // No replicas — simple mask check
            for (int k = 0; k < top_k; ++k)
            {
                if (!active_route[static_cast<size_t>(k)])
                    continue;
                const int expert = expert_indices[k];
                if (expert < 0 || expert >= static_cast<int>(expert_mask.size()))
                {
                    throw std::out_of_range(
                        "ExpertReplicaSet::assignForToken received an active expert outside its mask");
                }
                compute_here[k] = expert_mask[static_cast<size_t>(expert)];
            }
            return;
        }

        if (!hasLayerReplicaPlacement() ||
            layer_idx < 0 || layer_idx >= base_ownership.layerCount())
        {
            throw std::invalid_argument(
                "ExpertReplicaSet decode assignment requires complete layered ownership");
        }
        const int participant_count = participantCount();
        if (my_socket_id < 0 || my_socket_id >= participant_count)
        {
            throw std::out_of_range(
                "ExpertReplicaSet decode participant is outside its ownership geometry");
        }

        // Phase 1: Count fixed assignments (non-replicated experts go to owner).
        std::vector<int> load(static_cast<size_t>(participant_count), 0);
        std::array<bool, kMaxRoutedExpertsPerToken> is_fixed{};

        for (int k = 0; k < top_k; ++k)
        {
            if (!active_route[static_cast<size_t>(k)])
                continue;
            int e = expert_indices[k];
            const bool replicated = isReplicatedForLayer(layer_idx, e);
            if (!replicated)
            {
                is_fixed[k] = true;
                const int owner = base_ownership.owner(layer_idx, e);
                load[static_cast<size_t>(owner)]++;
            }
            else
            {
                is_fixed[k] = false;
            }
        }

        // Phase 2: Greedily assign replicated experts to the least-loaded
        // resident participant.
        // Process in index order for determinism (both ranks see same routing).
        for (int k = 0; k < top_k; ++k)
        {
            if (!active_route[static_cast<size_t>(k)] || is_fixed[k])
                continue;

            int e = expert_indices[k];
            const int owner = base_ownership.owner(layer_idx, e);

            std::vector<int> resident_participants;
            resident_participants.reserve(static_cast<size_t>(participant_count));
            resident_participants.push_back(owner);
            for (int participant = 0; participant < participant_count; ++participant)
            {
                if (participant == owner)
                    continue;
                if (hasReplicaOnParticipant(layer_idx, e, participant))
                    resident_participants.push_back(participant);
            }

            // Find the resident participant with the lowest current load.
            // Break ties: prefer the owner participant, then lower id for determinism.
            int best = owner;
            for (int participant : resident_participants)
            {
                if (participant < 0 || participant >= participant_count)
                    continue;
                if (load[static_cast<size_t>(participant)] < load[static_cast<size_t>(best)] ||
                    (load[static_cast<size_t>(participant)] == load[static_cast<size_t>(best)] &&
                     (participant == owner || (best != owner && participant < best))))
                    best = participant;
            }

            // Assign to the chosen participant.
            load[static_cast<size_t>(best)]++;
            is_fixed[k] = true; // Mark as assigned

            // Store assignment result directly
            compute_here[k] = (best == my_socket_id);
        }

        // Phase 3: Fill in non-replicated assignment results
        for (int k = 0; k < top_k; ++k)
        {
            if (!active_route[static_cast<size_t>(k)])
                continue;
            int e = expert_indices[k];
            if (!isReplicatedForLayer(layer_idx, e))
            {
                const int owner = base_ownership.owner(layer_idx, e);
                compute_here[k] = (owner == my_socket_id);
            }
        }
    }

    void ExpertReplicaSet::buildPrefillMask(
        int my_socket_id,
        const std::vector<bool> &expert_mask,
        int layer_idx)
    {
        prefill_mask.resize(expert_mask.size());
        for (size_t e = 0; e < expert_mask.size(); ++e)
        {
            // During prefill, only the owner participant processes replicated experts.
            // Non-replicated experts use the standard expert_mask.
            prefill_mask[e] =
                expert_mask[e] &&
                (!isReplicatedForLayer(layer_idx, static_cast<int>(e)) ||
                 base_ownership.owner(layer_idx, static_cast<int>(e)) == my_socket_id);
        }
    }

    bool ExpertReplicaSet::sameReplicaPlacement(const ExpertReplicaSet &other) const
    {
        return domain_id == other.domain_id &&
               num_replicated == other.num_replicated &&
               base_ownership == other.base_ownership &&
               replicated_in_any_layer == other.replicated_in_any_layer &&
               replica_participants_by_layer == other.replica_participants_by_layer;
    }

    ExpertReplicaSet ExpertReplicaSet::arrivalsSince(const ExpertReplicaSet &previous) const
    {
        ExpertReplicaSet arrivals;
        arrivals.domain_id = domain_id;
        arrivals.base_ownership = base_ownership;
        arrivals.replica_participants_by_layer.assign(
            static_cast<size_t>(base_ownership.layerCount()),
            std::vector<std::vector<bool>>(
                static_cast<size_t>(base_ownership.expertCount()),
                std::vector<bool>(static_cast<size_t>(participantCount()), false)));

        if (!hasLayerReplicaPlacement())
        {
            throw std::invalid_argument(
                "ExpertReplicaSet arrival diff requires complete layered replica placement");
        }

        for (int layer_idx = 0; layer_idx < base_ownership.layerCount(); ++layer_idx)
        {
            for (int expert_id = 0; expert_id < base_ownership.expertCount(); ++expert_id)
            {
                for (int participant = 0; participant < participantCount(); ++participant)
                {
                    if (!hasReplicaOnParticipant(layer_idx, expert_id, participant))
                        continue;

                    const bool same_geometry =
                        previous.base_ownership.layerCount() == base_ownership.layerCount() &&
                        previous.base_ownership.expertCount() == base_ownership.expertCount() &&
                        previous.base_ownership.participantCount() ==
                            base_ownership.participantCount();
                    const bool same_owner =
                        same_geometry &&
                        previous.base_ownership.owner(layer_idx, expert_id) ==
                            base_ownership.owner(layer_idx, expert_id);
                    const bool previously_resident =
                        same_owner && previous.hasLayerReplicaPlacement() &&
                        previous.hasReplicaOnParticipant(
                            layer_idx, expert_id, participant);

                    if (!previously_resident)
                    {
                        arrivals.replica_participants_by_layer
                            [static_cast<size_t>(layer_idx)]
                            [static_cast<size_t>(expert_id)]
                            [static_cast<size_t>(participant)] = true;
                    }
                }
            }
        }

        arrivals.rebuildAggregateReplicaFlags();
        return arrivals;
    }

    MoERebalanceController::MoERebalanceController(Config config)
        : requested_mode_(config.mode),
          config_(std::move(config)),
          current_ownership_(config_.initial_ownership),
          current_window_size_(config_.window_size)
    {
        if (current_ownership_.layerCount() != config_.num_layers ||
            current_ownership_.expertCount() != config_.num_experts ||
            current_ownership_.participantCount() !=
                static_cast<int>(config_.sockets.size()))
        {
            throw std::invalid_argument(
                "MoERebalanceController initial ownership geometry does not match its model and participants");
        }

        current_replicas_.domain_id = config_.domain_id;
        current_replicas_.base_ownership = current_ownership_;
        current_replicas_.replica_participants_by_layer.assign(
            static_cast<size_t>(config_.num_layers),
            std::vector<std::vector<bool>>(
                static_cast<size_t>(config_.num_experts),
                std::vector<bool>(config_.sockets.size(), false)));
        current_replicas_.rebuildAggregateReplicaFlags();

        if (config_.mode == MoERebalanceMode::DYNAMIC && config_.sockets.size() < 2)
        {
            config_.mode = MoERebalanceMode::OBSERVE;
            LOG_INFO("[MoERebalanceController] Dynamic rebalance downgraded to OBSERVE for single-participant domain");
        }

        if (config_.mode == MoERebalanceMode::OFF)
            return;

        // Create histogram for OBSERVE and DYNAMIC modes
        DecodeExpertHistogramConfig hcfg;
        hcfg.num_layers = config_.num_layers;
        hcfg.num_experts = config_.num_experts;
        hcfg.top_k = config_.top_k;
        hcfg.window_size = config_.window_size;
        hcfg.token_boundary_layer_idx = config_.token_boundary_layer_idx;
        hcfg.sockets = config_.sockets;
        hcfg.ownership = current_ownership_;
        histogram_ = std::make_unique<DecodeExpertHistogram>(std::move(hcfg));

        // Create rebalancer for DYNAMIC mode
        if (config_.mode == MoERebalanceMode::DYNAMIC)
        {
            rebalancer_ = std::make_unique<SocketAwareRebalancer>(config_.rebalance_config);
        }

        LOG_DEBUG("[MoERebalanceController] Initialized: mode="
                  << (config_.mode == MoERebalanceMode::OBSERVE ? "OBSERVE" : "DYNAMIC")
                  << " layers=" << config_.num_layers
                  << " experts=" << config_.num_experts
                  << " top_k=" << config_.top_k
                  << " window=" << config_.window_size
                  << " token_boundary_layer=" << config_.token_boundary_layer_idx
                  << " sockets=" << config_.sockets.size());
    }

    bool MoERebalanceController::shouldRebalance() const
    {
        return rebalanceDecision().ready;
    }

    MoERebalanceDecision MoERebalanceController::rebalanceDecision() const
    {
        if (requested_mode_ == MoERebalanceMode::OFF || config_.mode == MoERebalanceMode::OFF)
        {
            return {false, MoERebalanceDecisionReason::ModeOff};
        }

        if (requested_mode_ == MoERebalanceMode::DYNAMIC && config_.sockets.size() < 2)
        {
            return {false, MoERebalanceDecisionReason::SingleParticipantObserveOnly};
        }

        if (config_.mode != MoERebalanceMode::DYNAMIC)
        {
            return {false, MoERebalanceDecisionReason::DynamicDisabledForDomain};
        }

        if (!histogram_ || !histogram_->windowFull())
        {
            return {false, MoERebalanceDecisionReason::WindowNotFull};
        }

        return {true, MoERebalanceDecisionReason::Ready};
    }

    std::vector<MoELayeredExpertOwnershipChange> MoERebalanceController::rebalance()
    {
        if (!rebalancer_ || !histogram_)
            return {};

        last_imbalance_before_ = histogram_->placementImbalance(current_ownership_);
        last_imbalance_after_ = last_imbalance_before_;

        const auto proposal = rebalancer_->propose(*histogram_, current_ownership_);

        if (proposal.empty())
        {
            LOG_DEBUG("[MoERebalanceController] No beneficial swaps found, resetting window");
            histogram_->resetWindow();
            growWindowIfAdaptive();
            return {};
        }

        MoELayeredExpertOwnership candidate = current_ownership_;
        for (const auto &swap : proposal.swaps)
        {
            const int installed_owner =
                candidate.owner(swap.layer_idx, swap.expert_id);
            if (installed_owner != swap.from_socket)
            {
                throw std::logic_error(
                    "Dynamic MoE proposal was planned against a stale layered owner");
            }
            candidate.assignOwner(
                swap.layer_idx, swap.expert_id, swap.to_socket);
        }

        if (!candidate.hasSameLayerCapacitiesAs(current_ownership_))
        {
            throw std::logic_error(
                "Dynamic MoE proposal changed a layer's participant expert capacity");
        }

        last_imbalance_after_ = histogram_->placementImbalance(candidate);
        constexpr double kSpreadComparisonEpsilon = 1e-12;
        const bool aggregate_improved =
            last_imbalance_before_.valid &&
            last_imbalance_after_.valid &&
            last_imbalance_after_.average_spread + kSpreadComparisonEpsilon <
                last_imbalance_before_.average_spread;
        const bool worst_layer_did_not_regress =
            last_imbalance_before_.valid &&
            last_imbalance_after_.valid &&
            last_imbalance_after_.worst_spread <=
                last_imbalance_before_.worst_spread + kSpreadComparisonEpsilon;
        if (!aggregate_improved || !worst_layer_did_not_regress)
        {
            LOG_DEBUG("[MoERebalanceController] Rejected layered ownership proposal: "
                      << "average_spread " << last_imbalance_before_.average_spread
                      << " -> " << last_imbalance_after_.average_spread
                      << ", worst_spread " << last_imbalance_before_.worst_spread
                      << " -> " << last_imbalance_after_.worst_spread);
            histogram_->resetWindow();
            growWindowIfAdaptive();
            return {};
        }

        const auto changes = candidate.changesFrom(current_ownership_);
        if (changes.empty())
        {
            throw std::logic_error(
                "Dynamic MoE proposal contained entries but produced no ownership changes");
        }

        current_ownership_ = std::move(candidate);
        current_replicas_.base_ownership = current_ownership_;
        current_replicas_.rebuildAggregateReplicaFlags();
        ++placement_epoch_;
        rebalancer_->recordApplied(proposal);
        histogram_->updateOwnership(current_ownership_);
        histogram_->resetWindow();
        growWindowIfAdaptive();

        ++total_rebalances_;
        total_swap_pairs_ += proposal.numSwapPairs();
        total_ownership_changes_ += static_cast<int>(changes.size());

        LOG_DEBUG("[MoERebalanceController] Rebalance #" << total_rebalances_
                                                         << ": " << proposal.summary());

        return changes;
    }

    void MoERebalanceController::resetRebalanceWindow()
    {
        if (!histogram_)
            return;
        histogram_->resetWindow();
        growWindowIfAdaptive();
    }

    void MoERebalanceController::logHistogramSummary() const
    {
        if (!histogram_)
        {
            LOG_DEBUG("[MoERebalanceController] No histogram (mode=OFF)");
            return;
        }

        LOG_DEBUG("[MoERebalanceController] Histogram summary (window tokens="
                  << histogram_->windowTokenCount()
                  << " gen=" << histogram_->windowGeneration()
                  << " avg_imbalance=" << histogram_->averageSocketImbalance() << "):");

        for (int l = 0; l < config_.num_layers; ++l)
        {
            LOG_DEBUG("  " << histogram_->layerSummary(l));
        }
    }

    std::vector<std::vector<bool>> MoERebalanceController::computeExpertMasks(int socket_id) const
    {
        const int num_layers = config_.num_layers;
        const int num_experts = config_.num_experts;
        auto masks = current_ownership_.masksForParticipant(socket_id);

        // If replicas are active, expand masks to include replicated experts
        // that this socket should also have GEMM engines for.
        if (current_replicas_.num_replicated > 0)
        {
            for (int l = 0; l < num_layers; ++l)
            {
                for (int e = 0; e < num_experts; ++e)
                {
                    if (current_replicas_.hasReplicaOnParticipant(l, e, socket_id))
                        masks[l][e] = true;
                }
            }
        }

        return masks;
    }

    std::vector<std::vector<std::vector<bool>>> MoERebalanceController::computeGpuCacheExpertMasks(
        int gpu_cache_experts_per_layer) const
    {
        const int num_layers = config_.num_layers;
        const int num_experts = config_.num_experts;
        const int num_sockets = static_cast<int>(config_.sockets.size());

        std::vector<std::vector<std::vector<bool>>> masks_by_socket(num_sockets);
        for (int s = 0; s < num_sockets; ++s)
            masks_by_socket[s].assign(num_layers, std::vector<bool>(num_experts, false));

        auto installed_ownership_masks = [&]()
        {
            for (int s = 0; s < num_sockets; ++s)
                masks_by_socket[s] = computeExpertMasks(s);
            return masks_by_socket;
        };

        if (gpu_cache_experts_per_layer <= 0 || !histogram_ || num_layers <= 0 || num_experts <= 0)
            return installed_ownership_masks();

        std::vector<int> gpu_sockets;
        std::vector<int> cpu_sockets;
        for (int s = 0; s < num_sockets; ++s)
        {
            if (config_.sockets[s].is_gpu())
                gpu_sockets.push_back(s);
            else
                cpu_sockets.push_back(s);
        }

        if (gpu_sockets.empty() || cpu_sockets.empty())
            return installed_ownership_masks();

        const int cache_count = std::min(gpu_cache_experts_per_layer, num_experts);
        int total_gpu_assignments = 0;

        for (int l = 0; l < num_layers; ++l)
        {
            auto counts = histogram_->layerHistogram(l);
            std::vector<int> experts(num_experts);
            std::iota(experts.begin(), experts.end(), 0);
            std::sort(experts.begin(), experts.end(),
                      [&](int a, int b)
                      {
                          if (counts[a] != counts[b])
                              return counts[a] > counts[b];
                          return a < b;
                      });

            std::vector<bool> gpu_cached(num_experts, false);
            for (int i = 0; i < cache_count; ++i)
            {
                const int expert = experts[i];
                gpu_cached[expert] = true;
            }

            auto assign_lpt = [&](const std::vector<int> &socket_ids,
                                  const std::vector<int> &expert_ids)
            {
                std::vector<uint64_t> loads(socket_ids.size(), 0);
                std::vector<int> expert_counts(socket_ids.size(), 0);
                for (int expert : expert_ids)
                {
                    size_t best = 0;
                    for (size_t i = 1; i < socket_ids.size(); ++i)
                    {
                        if (loads[i] < loads[best] ||
                            (loads[i] == loads[best] && expert_counts[i] < expert_counts[best]))
                        {
                            best = i;
                        }
                    }
                    const int socket = socket_ids[best];
                    masks_by_socket[socket][l][expert] = true;
                    loads[best] += counts[expert];
                    expert_counts[best]++;
                }
            };

            std::vector<int> hot_experts;
            hot_experts.reserve(cache_count);
            for (int expert : experts)
            {
                if (gpu_cached[expert])
                    hot_experts.push_back(expert);
            }

            std::vector<int> cold_experts;
            cold_experts.reserve(num_experts - cache_count);
            for (int expert : experts)
            {
                if (!gpu_cached[expert])
                    cold_experts.push_back(expert);
            }

            assign_lpt(gpu_sockets, hot_experts);
            assign_lpt(cpu_sockets, cold_experts);
            total_gpu_assignments += static_cast<int>(hot_experts.size());
        }

        LOG_DEBUG("[MoERebalanceController] GPU expert cache masks: "
                  << cache_count << "/" << num_experts << " routed experts per layer on GPU domain"
                  << " (gpu_sockets=" << gpu_sockets.size()
                  << ", cpu_sockets=" << cpu_sockets.size()
                  << ", assignments=" << total_gpu_assignments << ")");

        return masks_by_socket;
    }

    void MoERebalanceController::growWindowIfAdaptive()
    {
        if (config_.max_window_size <= 0 || config_.window_growth_factor <= 1.0f)
            return;

        int new_size = static_cast<int>(current_window_size_ * config_.window_growth_factor);
        new_size = std::min(new_size, config_.max_window_size);
        if (new_size > current_window_size_)
        {
            LOG_DEBUG("[MoERebalanceController] Adaptive window: "
                      << current_window_size_ << " -> " << new_size);
            current_window_size_ = new_size;
            histogram_->setWindowSize(new_size);
        }
    }

    ExpertReplicaSet MoERebalanceController::proposeReplicas(int max_replicas_per_socket)
    {
        last_imbalance_before_ = {};
        last_imbalance_after_ = {};

        ExpertReplicaSet result;
        result.domain_id = config_.domain_id;
        result.base_ownership = current_ownership_;
        result.num_replicated = 0;
        const int num_sockets = result.participantCount();
        result.replica_participants_by_layer.assign(
            static_cast<size_t>(std::max(0, config_.num_layers)),
            std::vector<std::vector<bool>>(
                static_cast<size_t>(std::max(0, config_.num_experts)),
                std::vector<bool>(static_cast<size_t>(std::max(0, num_sockets)), false)));
        result.rebuildAggregateReplicaFlags();

        if (!histogram_ || max_replicas_per_socket <= 0 || config_.sockets.size() < 2)
            return result;

        const int num_experts = config_.num_experts;
        const int num_layers = config_.num_layers;

        std::vector<std::vector<uint64_t>> layer_counts(
            static_cast<size_t>(num_layers),
            std::vector<uint64_t>(static_cast<size_t>(num_experts), 0));

        uint64_t total_activations = 0;
        for (int l = 0; l < num_layers; ++l)
        {
            auto counts = histogram_->layerHistogram(l);
            for (int e = 0; e < num_experts; ++e)
            {
                layer_counts[static_cast<size_t>(l)][static_cast<size_t>(e)] = counts[e];
                total_activations += counts[e];
            }
        }

        if (total_activations == 0 && current_replicas_.num_replicated > 0)
        {
            last_imbalance_before_ = histogram_->placementImbalance(current_ownership_);
            last_imbalance_after_ = last_imbalance_before_;
            LOG_DEBUG("[MoERebalanceController] No activation signal; preserving "
                      << current_replicas_.num_replicated << " existing expert replica slots");
            return current_replicas_;
        }

        std::vector<std::vector<uint64_t>> projected_loads(
            static_cast<size_t>(num_layers),
            std::vector<uint64_t>(static_cast<size_t>(num_sockets), 0));
        for (int l = 0; l < num_layers; ++l)
        {
            for (int e = 0; e < num_experts; ++e)
            {
                const int owner = current_ownership_.owner(l, e);
                if (owner >= 0 && owner < num_sockets)
                {
                    projected_loads[static_cast<size_t>(l)][static_cast<size_t>(owner)] +=
                        layer_counts[static_cast<size_t>(l)][static_cast<size_t>(e)];
                }
            }
        }
        last_imbalance_before_ = imbalanceStatsFromParticipantLoads(projected_loads);
        last_imbalance_after_ = last_imbalance_before_;

        auto owner_for = [&](int layer, int expert)
        {
            if (layer < 0 || layer >= num_layers ||
                expert < 0 || expert >= num_experts)
                return -1;
            return current_ownership_.owner(layer, expert);
        };

        auto projected_shift = [&](int layer, int expert, int target_socket)
        {
            const int owner = owner_for(layer, expert);
            if (owner < 0 || owner >= num_sockets || owner == target_socket)
                return uint64_t{0};

            const uint64_t count =
                layer_counts[static_cast<size_t>(layer)][static_cast<size_t>(expert)];
            if (count == 0)
                return uint64_t{0};

            const uint32_t owner_mask =
                moe_rebalance_policy::participantBit(static_cast<uint32_t>(owner));
            const uint32_t target_mask =
                moe_rebalance_policy::participantBit(static_cast<uint32_t>(target_socket));
            const auto delta =
                moe_rebalance_policy::evaluateAddingResidentDynamicSpread(
                    projected_loads[static_cast<size_t>(layer)].data(),
                    count,
                    owner_mask,
                    owner_mask | target_mask,
                    static_cast<uint32_t>(num_sockets),
                    static_cast<uint32_t>(owner),
                    static_cast<uint32_t>(target_socket),
                    static_cast<uint32_t>(std::max(1, current_window_size_)));
            return delta.improvement;
        };

        const uint64_t minimum_replica_shift =
            moe_rebalance_policy::dynamicMinimumProjectedShift(
                static_cast<uint32_t>(std::max(1, current_window_size_)),
                total_activations);

        auto admissible_projected_shift = [&](int layer, int expert, int target_socket)
        {
            const uint64_t shift = projected_shift(layer, expert, target_socket);
            return shift >= minimum_replica_shift ? shift : uint64_t{0};
        };

        auto apply_projected_shift = [&](int layer, int expert, int target_socket)
        {
            const int owner = owner_for(layer, expert);
            const uint64_t shift = admissible_projected_shift(layer, expert, target_socket);
            if (shift == 0 || owner < 0 || owner >= num_sockets)
                return false;

            result.setReplicaOnParticipant(layer, expert, target_socket, true);
            auto &layer_loads = projected_loads[static_cast<size_t>(layer)];
            layer_loads[static_cast<size_t>(owner)] -=
                std::min(layer_loads[static_cast<size_t>(owner)], shift);
            layer_loads[static_cast<size_t>(target_socket)] += shift;
            return true;
        };

        std::vector<int> replicated_per_target(static_cast<size_t>(num_sockets), 0);

        struct Candidate
        {
            int layer = -1;
            int expert = -1;
            int target = -1;
            uint64_t count = 0;
            uint64_t shift = 0;
        };

        auto best_count_for_target = [&](int target_socket)
        {
            uint64_t best = 0;
            for (int l = 0; l < num_layers; ++l)
            {
                for (int e = 0; e < num_experts; ++e)
                {
                    if (owner_for(l, e) == target_socket)
                        continue;
                    if (admissible_projected_shift(l, e, target_socket) == 0)
                        continue;
                    best = std::max(
                        best,
                        layer_counts[static_cast<size_t>(l)][static_cast<size_t>(e)]);
                }
            }
            return best;
        };

        // Keep still-warm existing replicas before filling new slots so short
        // decode windows do not churn PCIe/NVLink transfers on tiny rank-order
        // changes. A previous replica must stay within 50% of the current top
        // load-reducing candidate for that target and still reduce projected
        // per-layer participant load.
        for (int target_socket = 0; target_socket < num_sockets; ++target_socket)
        {
            const uint64_t top_count = best_count_for_target(target_socket);
            if (top_count == 0)
                continue;

            std::vector<Candidate> previous_candidates;
            for (int l = 0; l < num_layers; ++l)
            {
                for (int e = 0; e < num_experts; ++e)
                {
                    const uint64_t count = layer_counts[static_cast<size_t>(l)][static_cast<size_t>(e)];
                    const uint64_t shift = admissible_projected_shift(l, e, target_socket);
                    if (count == 0 || shift == 0)
                        continue;
                    if (count * 2u < top_count)
                        continue;
                    const int owner = owner_for(l, e);
                    if (owner < 0 || owner >= num_sockets || owner == target_socket)
                        continue;
                    if (current_replicas_.base_ownership.empty() ||
                        current_replicas_.base_ownership.owner(l, e) != owner ||
                        !current_replicas_.hasReplicaOnParticipant(l, e, target_socket))
                    {
                        continue;
                    }
                    previous_candidates.push_back({l, e, target_socket, count, shift});
                }
            }

            std::sort(previous_candidates.begin(), previous_candidates.end(),
                      [](const Candidate &a, const Candidate &b)
                      {
                          if (a.shift != b.shift)
                              return a.shift > b.shift;
                          if (a.count != b.count)
                              return a.count > b.count;
                          if (a.layer != b.layer)
                              return a.layer < b.layer;
                          if (a.expert != b.expert)
                              return a.expert < b.expert;
                          return a.target < b.target;
                      });

            for (const Candidate &candidate : previous_candidates)
            {
                if (replicated_per_target[static_cast<size_t>(target_socket)] >= max_replicas_per_socket)
                    break;
                if (result.hasReplicaOnParticipant(candidate.layer, candidate.expert, target_socket))
                    continue;
                if (apply_projected_shift(candidate.layer, candidate.expert, target_socket))
                    ++replicated_per_target[static_cast<size_t>(target_socket)];
            }
        }

        while (true)
        {
            Candidate best;
            for (int target_socket = 0; target_socket < num_sockets; ++target_socket)
            {
                if (replicated_per_target[static_cast<size_t>(target_socket)] >= max_replicas_per_socket)
                    continue;
                for (int l = 0; l < num_layers; ++l)
                {
                    for (int e = 0; e < num_experts; ++e)
                    {
                        if (result.hasReplicaOnParticipant(l, e, target_socket))
                            continue;
                        const uint64_t shift = admissible_projected_shift(l, e, target_socket);
                        if (shift == 0)
                            continue;
                        const uint64_t count =
                            layer_counts[static_cast<size_t>(l)][static_cast<size_t>(e)];
                        Candidate candidate{l, e, target_socket, count, shift};
                        if (best.layer < 0 ||
                            candidate.shift > best.shift ||
                            (candidate.shift == best.shift && candidate.count > best.count) ||
                            (candidate.shift == best.shift && candidate.count == best.count &&
                             candidate.layer < best.layer) ||
                            (candidate.shift == best.shift && candidate.count == best.count &&
                             candidate.layer == best.layer && candidate.expert < best.expert) ||
                            (candidate.shift == best.shift && candidate.count == best.count &&
                             candidate.layer == best.layer && candidate.expert == best.expert &&
                             candidate.target < best.target))
                        {
                            best = candidate;
                        }
                    }
                }
            }

            if (best.layer < 0)
                break;
            if (!apply_projected_shift(best.layer, best.expert, best.target))
                break;
            ++replicated_per_target[static_cast<size_t>(best.target)];
        }

        result.rebuildAggregateReplicaFlags();
        last_imbalance_after_ = imbalanceStatsFromParticipantLoads(projected_loads);
        const bool replica_placement_changed = !current_replicas_.sameReplicaPlacement(result);
        current_replicas_ = result;
        if (replica_placement_changed && result.num_replicated > 0)
            ++placement_epoch_;

        LOG_DEBUG("[MoERebalanceController] Proposed " << result.num_replicated
                                                       << " expert replica slots (max " << max_replicas_per_socket << " per participant)");
        LOG_DEBUG("[MoERebalanceController] Replica admission minimum projected shift="
                  << minimum_replica_shift << " assignments/window");

        // Log the top replicas per participant.
        for (int s = 0; s < num_sockets; ++s)
        {
            std::ostringstream oss;
            oss << "  Participant " << s << " gets replicas: ";
            int count = 0;
            for (int l = 0; l < num_layers; ++l)
            {
                for (int e = 0; e < num_experts; ++e)
                {
                    if (!result.hasReplicaOnParticipant(l, e, s))
                        continue;
                    if (count > 0)
                        oss << ", ";
                    oss << "l" << l << ".e" << e
                        << "(" << layer_counts[static_cast<size_t>(l)][static_cast<size_t>(e)] << ")";
                    count++;
                }
            }
            LOG_DEBUG("[MoERebalanceController] " << oss.str());
        }

        return result;
    }

    std::string MoERebalanceController::getProfilingSummary() const
    {
        std::ostringstream oss;

        // ── Title ──────────────────────────────────────────────────
        {
            fort::utf8_table title;
            title.set_border_style(FT_DOUBLE2_STYLE);
            title << "MOE EXPERT REBALANCE PROFILING" << fort::endr;
            title[0][0].set_cell_text_align(fort::text_align::center);
            title.row(0).set_cell_row_type(fort::row_type::header);
            oss << "\n"
                << title.to_string();
        }

        // ── Config table ───────────────────────────────────────────
        {
            fort::utf8_table t;
            t.set_border_style(FT_DOUBLE2_STYLE);
            t << fort::header << "Parameter" << "Value" << fort::endr;
            t.column(0).set_cell_text_align(fort::text_align::left);
            t.column(1).set_cell_text_align(fort::text_align::right);

            t << "Mode" << (config_.mode == MoERebalanceMode::OBSERVE ? "OBSERVE" : config_.mode == MoERebalanceMode::DYNAMIC ? "DYNAMIC"
                                                                                                                              : "OFF")
              << fort::endr;
            if (requested_mode_ != config_.mode)
            {
                t << "Requested mode" << (requested_mode_ == MoERebalanceMode::OBSERVE ? "OBSERVE" : requested_mode_ == MoERebalanceMode::DYNAMIC ? "DYNAMIC"
                                                                                                                                                   : "OFF")
                  << fort::endr;
            }
            t << "Rebalance state" << toString(rebalanceDecision().reason) << fort::endr;
            t << "Experts" << config_.num_experts << fort::endr;
            t << "Participants" << config_.sockets.size() << fort::endr;
            t << "Top-K" << config_.top_k << fort::endr;
            t << "Window size" << config_.window_size << fort::endr;
            if (config_.max_window_size > 0 && config_.window_growth_factor > 1.0f)
            {
                std::ostringstream ws;
                ws << config_.window_size << " -> " << current_window_size_
                   << " (x" << std::fixed << std::setprecision(1) << config_.window_growth_factor
                   << ", max " << config_.max_window_size << ")";
                t << "Adaptive window" << ws.str() << fort::endr;
            }
            oss << t.to_string();
        }

        // ── Rebalance timing (if any rebalancing occurred) ─────────
        if (total_rebalances_ > 0)
        {
            fort::utf8_table t;
            t.set_border_style(FT_DOUBLE2_STYLE);
            t << fort::header << "Rebalance Metric" << "Value" << fort::endr;
            t.column(0).set_cell_text_align(fort::text_align::left);
            t.column(1).set_cell_text_align(fort::text_align::right);

            t << "Total rebalances" << total_rebalances_ << fort::endr;
            t << "Total swap pairs" << total_swap_pairs_ << fort::endr;
            t << "Total layer/expert moves" << total_ownership_changes_ << fort::endr;

            if (last_prep_duration_ms_ > 0)
            {
                std::ostringstream val;
                val << std::fixed << std::setprecision(2) << last_prep_duration_ms_ << " ms";
                t << "VNNI prep time" << val.str() << fort::endr;
            }

            if (last_imbalance_before_.valid)
            {
                std::ostringstream val;
                val << std::fixed << std::setprecision(3)
                    << last_imbalance_before_.average_spread
                    << " -> " << last_imbalance_after_.average_spread;
                t << "Avg load spread" << val.str() << fort::endr;
            }

            if (last_imbalance_before_.valid)
            {
                std::ostringstream val;
                val << std::fixed << std::setprecision(3)
                    << last_imbalance_before_.worst_spread
                    << " -> " << last_imbalance_after_.worst_spread
                    << " (layer " << last_imbalance_before_.worst_layer << ")";
                t << "Worst load spread" << val.str() << fort::endr;
            }

            oss << t.to_string();
        }

        // ── Histogram summary ──────────────────────────────────────
        if (histogram_ && histogram_->windowTokenCount() > 0)
        {
            int num_layers = config_.num_layers;
            int num_sockets = static_cast<int>(config_.sockets.size());

            // Collect per-layer imbalance data
            struct LayerImbalance
            {
                int layer;
                float imbalance;
                std::vector<uint64_t> socket_loads;
            };
            std::vector<LayerImbalance> layer_data;
            layer_data.reserve(num_layers);
            float worst_imbalance = 0.0f;
            int worst_layer = 0;
            float avg_imbalance = 0.0f;

            for (int l = 0; l < num_layers; ++l)
            {
                float imb = histogram_->socketImbalanceRatio(l);
                auto loads = histogram_->socketLoads(l);
                avg_imbalance += imb;
                if (imb > worst_imbalance)
                {
                    worst_imbalance = imb;
                    worst_layer = l;
                }
                layer_data.push_back({l, imb, std::move(loads)});
            }
            avg_imbalance /= std::max(1, num_layers);

            // Top-5 hottest experts globally
            std::vector<uint64_t> global_counts(config_.num_experts, 0);
            for (int l = 0; l < num_layers; ++l)
            {
                auto lc = histogram_->layerHistogram(l);
                for (int e = 0; e < config_.num_experts; ++e)
                    global_counts[e] += lc[e];
            }
            std::vector<int> sorted(config_.num_experts);
            std::iota(sorted.begin(), sorted.end(), 0);
            std::sort(sorted.begin(), sorted.end(),
                      [&](int a, int b)
                      { return global_counts[a] > global_counts[b]; });

            // Per-socket stats
            std::vector<int> socket_expert_counts(num_sockets, 0);
            for (int l = 0; l < num_layers; ++l)
            {
                for (int e = 0; e < config_.num_experts; ++e)
                {
                    ++socket_expert_counts[static_cast<size_t>(
                        current_ownership_.owner(l, e))];
                }
            }
            std::vector<uint64_t> socket_total_load(num_sockets, 0);
            for (int l = 0; l < num_layers; ++l)
            {
                for (int s = 0; s < num_sockets; ++s)
                {
                    if (s < static_cast<int>(layer_data[l].socket_loads.size()))
                        socket_total_load[s] += layer_data[l].socket_loads[s];
                }
            }
            uint64_t total_load = 0;
            for (int s = 0; s < num_sockets; ++s)
                total_load += socket_total_load[s];

            // Histogram overview table
            {
                fort::utf8_table t;
                t.set_border_style(FT_DOUBLE2_STYLE);
                t << fort::header << "Histogram Metric" << "Value" << fort::endr;
                t.column(0).set_cell_text_align(fort::text_align::left);
                t.column(1).set_cell_text_align(fort::text_align::right);

                t << "Window tokens" << histogram_->windowTokenCount() << fort::endr;

                {
                    std::ostringstream val;
                    val << std::fixed << std::setprecision(3) << avg_imbalance << "x";
                    t << "Avg imbalance" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    val << std::fixed << std::setprecision(3) << worst_imbalance << "x (layer " << worst_layer << ")";
                    t << "Worst imbalance" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    int show = std::min(5, config_.num_experts);
                    for (int i = 0; i < show; ++i)
                    {
                        if (i > 0)
                            val << ", ";
                        val << "e" << sorted[i] << "(" << global_counts[sorted[i]] << ")";
                    }
                    t << "Top-5 hottest" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    for (int s = 0; s < num_sockets; ++s)
                    {
                        if (s > 0)
                            val << ", ";
                        val << "s" << s << "=" << socket_expert_counts[s];
                    }
                    t << "Experts/socket" << val.str() << fort::endr;
                }
                {
                    std::ostringstream val;
                    for (int s = 0; s < num_sockets; ++s)
                    {
                        if (s > 0)
                            val << ", ";
                        double pct = total_load > 0 ? 100.0 * socket_total_load[s] / total_load : 0;
                        val << "s" << s << "=" << std::fixed << std::setprecision(1) << pct << "%";
                    }
                    t << "Load/socket" << val.str() << fort::endr;
                }
                oss << t.to_string();
            }

            // Top-5 most imbalanced layers table
            if (num_layers > 1 && num_sockets >= 2)
            {
                auto sorted_layers = layer_data;
                std::sort(sorted_layers.begin(), sorted_layers.end(),
                          [](const auto &a, const auto &b)
                          { return a.imbalance > b.imbalance; });

                int show_layers = std::min(5, num_layers);

                fort::utf8_table t;
                t.set_border_style(FT_DOUBLE2_STYLE);
                t << fort::header << "Layer" << "Imbalance";
                for (int s = 0; s < num_sockets; ++s)
                {
                    std::ostringstream hdr;
                    hdr << "Socket " << s;
                    t << hdr.str();
                }
                t << fort::endr;

                t.column(0).set_cell_text_align(fort::text_align::right);
                t.column(1).set_cell_text_align(fort::text_align::right);
                for (int s = 0; s < num_sockets; ++s)
                    t.column(2 + s).set_cell_text_align(fort::text_align::right);

                for (int i = 0; i < show_layers; ++i)
                {
                    const auto &ld = sorted_layers[i];
                    std::ostringstream imb_ss;
                    imb_ss << std::fixed << std::setprecision(2) << ld.imbalance << "x";
                    t << ld.layer << imb_ss.str();
                    for (int s = 0; s < num_sockets; ++s)
                    {
                        uint64_t load = (s < static_cast<int>(ld.socket_loads.size()))
                                            ? ld.socket_loads[s]
                                            : 0;
                        t << load;
                    }
                    t << fort::endr;
                }

                oss << t.to_string();
            }
        }

        return oss.str();
    }

} // namespace llaminar2
