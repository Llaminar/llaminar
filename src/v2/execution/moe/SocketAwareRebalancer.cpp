/**
 * @file SocketAwareRebalancer.cpp
 * @brief Socket-load-aware expert rebalancer implementation
 */

#include "SocketAwareRebalancer.h"
#include "../../utils/Logger.h"
#include "DeviceMoERebalancePolicyShared.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace llaminar2
{
    namespace
    {
        uint32_t ratioToPerMille(float ratio, uint32_t fallback)
        {
            if (!std::isfinite(ratio) || ratio < 0.0f)
                return fallback;
            const double value = static_cast<double>(ratio) * 1000.0;
            if (value <= 0.0)
                return 0u;
            if (value >= static_cast<double>(std::numeric_limits<uint32_t>::max()))
                return std::numeric_limits<uint32_t>::max();
            return static_cast<uint32_t>(std::llround(value));
        }

        float imbalanceRatio(uint64_t max_load, uint64_t min_load)
        {
            if (min_load > 0)
                return static_cast<float>(max_load) / static_cast<float>(min_load);
            return max_load > 0 ? std::numeric_limits<float>::infinity() : 1.0f;
        }

        float expectedRatioReduction(
            const moe_rebalance_policy::OwnershipSwapChoice &choice)
        {
            const float before = imbalanceRatio(choice.old_max_load, choice.old_min_load);
            const float after = imbalanceRatio(choice.new_max_load, choice.new_min_load);
            if (std::isfinite(before) && std::isfinite(after))
                return before - after;
            return static_cast<float>(choice.improvement);
        }
    }

    // ── SocketRebalanceProposal ───────────────────────

    std::string SocketRebalanceProposal::summary() const
    {
        if (empty())
            return "SocketRebalanceProposal: no swaps proposed";

        int num_layers = static_cast<int>(layer_metrics.size());
        std::ostringstream ss;
        ss << "SocketRebalanceProposal: " << numSwapPairs() << " swap pair"
           << (numSwapPairs() != 1 ? "s" : "") << " ("
           << numOwnershipChanges() << " ownership changes) across " << num_layers
           << " layer" << (num_layers != 1 ? "s" : "")
           << " (gen=" << window_generation << ")";

        for (const auto& lm : layer_metrics) {
            ss << "\n  Layer " << lm.layer_idx << ": "
               << std::fixed;
            ss.precision(2);
            ss << lm.imbalance_before << " -> ~" << lm.estimated_imbalance_after
               << " (" << lm.num_swaps << " swap"
               << (lm.num_swaps != 1 ? "s" : "") << ")";
        }

        return ss.str();
    }

    // ── SocketAwareRebalancer ─────────────────────────

    SocketAwareRebalancer::SocketAwareRebalancer(SocketRebalanceConfig config)
        : config_(config)
    {
    }

    SocketRebalanceProposal SocketAwareRebalancer::propose(
        const DecodeExpertHistogram &histogram,
        const MoELayeredExpertOwnership &ownership) const
    {
        SocketRebalanceProposal proposal;
        proposal.window_generation = histogram.windowGeneration();

        const auto& hcfg = histogram.config();
        const int num_layers = hcfg.num_layers;
        const int num_sockets = static_cast<int>(hcfg.sockets.size());

        if (num_layers <= 0 || num_sockets < 2) {
            return proposal;
        }
        if (ownership.layerCount() != num_layers ||
            ownership.expertCount() != hcfg.num_experts ||
            ownership.participantCount() != num_sockets)
        {
            throw std::invalid_argument(
                "SocketAwareRebalancer ownership geometry does not match the histogram");
        }

        // Ensure cooldown vector is sized (UINT64_MAX = never rebalanced)
        if (layer_last_rebalanced_.size() < static_cast<size_t>(num_layers))
            layer_last_rebalanced_.resize(num_layers, UINT64_MAX);

        const uint64_t current_gen = histogram.windowGeneration();
        int total_swaps = 0;

        for (int l = 0; l < num_layers; ++l) {
            // Check layer cooldown (UINT64_MAX = never rebalanced, always allow)
            if (layer_last_rebalanced_[l] != UINT64_MAX &&
                (current_gen - layer_last_rebalanced_[l]) < static_cast<uint64_t>(config_.layer_cooldown_generations)) {
                continue;
            }

            auto expert_counts = histogram.layerHistogram(l);

            // Check minimum activations
            uint64_t total_activations = std::accumulate(
                expert_counts.begin(), expert_counts.end(), uint64_t{0});
            if (total_activations < config_.min_window_activations)
                continue;

            // Every routed layer owns its own complete placement row.  Never
            // substitute a global expert map here: conflicting layer-local
            // swaps are both valid and must survive publication.
            const auto &expert_to_socket = ownership.ownersForLayer(l);

            auto layer_swaps = proposeForLayer(l, expert_counts, expert_to_socket, num_sockets);
            if (layer_swaps.empty())
                continue;

            // Enforce the total entry budget without ever splitting a paired
            // swap. A half-pair would change participant expert capacity.
            int remaining = config_.max_total_swaps - total_swaps;
            remaining -= remaining % 2;
            if (remaining < 2)
                break;
            if (static_cast<int>(layer_swaps.size()) > remaining)
                layer_swaps.resize(remaining);

            // Compute estimated imbalance after swaps
            std::vector<uint64_t> current_loads(static_cast<size_t>(num_sockets), 0);
            for (int e = 0; e < static_cast<int>(expert_counts.size()); ++e)
            {
                current_loads[static_cast<size_t>(expert_to_socket[static_cast<size_t>(e)])] +=
                    expert_counts[static_cast<size_t>(e)];
            }
            const auto [current_min_it, current_max_it] =
                std::minmax_element(current_loads.begin(), current_loads.end());
            float imbalance_before = imbalanceRatio(*current_max_it, *current_min_it);

            // Simulate the swaps to estimate new imbalance
            std::vector<int> simulated_placement = expert_to_socket;
            for (const auto& swap : layer_swaps) {
                // Find a partner: the expert from to_socket that we swap with
                // The swap already contains the from/to info
                simulated_placement[swap.expert_id] = swap.to_socket;
            }

            // Each accepted Dynamic ownership move is represented as paired entries:
            // heavy overloaded owner -> underloaded participant, light underloaded
            // owner -> overloaded participant.
            std::vector<uint64_t> sim_loads(num_sockets, 0);
            for (int e = 0; e < static_cast<int>(expert_counts.size()); ++e) {
                sim_loads[simulated_placement[e]] += expert_counts[e];
            }
            auto [sim_min_it, sim_max_it] = std::minmax_element(sim_loads.begin(), sim_loads.end());
            float estimated_after = 1.0f;
            if (*sim_min_it > 0)
                estimated_after = static_cast<float>(*sim_max_it) / static_cast<float>(*sim_min_it);
            else if (*sim_max_it > 0)
                estimated_after = std::numeric_limits<float>::infinity();

            // Add layer metrics
            SocketRebalanceProposal::LayerMetrics lm;
            lm.layer_idx = l;
            lm.imbalance_before = imbalance_before;
            lm.estimated_imbalance_after = estimated_after;
            lm.num_swaps = static_cast<int>(layer_swaps.size()) / 2; // Each swap is a pair of 2 entries
            if (lm.num_swaps == 0) lm.num_swaps = static_cast<int>(layer_swaps.size());
            proposal.layer_metrics.push_back(lm);

            total_swaps += static_cast<int>(layer_swaps.size());
            proposal.swaps.insert(proposal.swaps.end(), layer_swaps.begin(), layer_swaps.end());

        }

        if (!proposal.empty()) {
            LOG_DEBUG(proposal.summary());
        }

        return proposal;
    }

    void SocketAwareRebalancer::recordApplied(
        const SocketRebalanceProposal &proposal) const
    {
        if (proposal.empty())
            return;

        const int required_layers = proposal.layer_metrics.empty()
                                        ? 0
                                        : 1 + std::max_element(
                                                  proposal.layer_metrics.begin(),
                                                  proposal.layer_metrics.end(),
                                                  [](const auto &lhs, const auto &rhs)
                                                  {
                                                      return lhs.layer_idx < rhs.layer_idx;
                                                  })
                                                  ->layer_idx;
        if (required_layers > static_cast<int>(layer_last_rebalanced_.size()))
        {
            layer_last_rebalanced_.resize(
                static_cast<size_t>(required_layers), UINT64_MAX);
        }

        for (const auto &metrics : proposal.layer_metrics)
        {
            layer_last_rebalanced_.at(static_cast<size_t>(metrics.layer_idx)) =
                proposal.window_generation;
        }
    }

    std::vector<ExpertSwap> SocketAwareRebalancer::proposeForLayer(
        int layer_idx,
        const std::vector<uint64_t>& expert_counts,
        const std::vector<int>& expert_to_socket,
        int num_sockets) const
    {
        std::vector<ExpertSwap> swaps;
        const int num_experts = static_cast<int>(expert_counts.size());

        // Working copy of placement and loads. The selection step delegates to
        // the shared Dynamic policy used by the graph-captured GPU controller.
        std::vector<int32_t> placement(static_cast<size_t>(num_experts), -1);
        std::vector<uint64_t> socket_loads(num_sockets, 0);
        for (int e = 0; e < num_experts; ++e) {
            const int owner =
                e < static_cast<int>(expert_to_socket.size()) ? expert_to_socket[e] : -1;
            if (owner < 0 || owner >= num_sockets)
                continue;
            placement[static_cast<size_t>(e)] = owner;
            socket_loads[static_cast<size_t>(owner)] += expert_counts[e];
        }

        const uint32_t imbalance_threshold_per_mille =
            ratioToPerMille(config_.imbalance_threshold, 1300u);
        const uint32_t min_improvement_per_mille =
            ratioToPerMille(config_.min_improvement_ratio, 50u);

        for (int swap_iter = 0; swap_iter < config_.max_swaps_per_layer; ++swap_iter) {
            const auto choice = moe_rebalance_policy::bestDynamicOwnershipSwap(
                socket_loads.data(),
                expert_counts.data(),
                placement.data(),
                static_cast<uint32_t>(num_experts),
                static_cast<uint32_t>(num_sockets),
                imbalance_threshold_per_mille,
                min_improvement_per_mille,
                config_.min_window_activations);
            if (!choice.valid)
                break;

            const float load_reduction = expectedRatioReduction(choice);
            swaps.push_back({layer_idx,
                             static_cast<int>(choice.heavy_expert),
                             static_cast<int>(choice.overloaded_participant),
                             static_cast<int>(choice.underloaded_participant),
                             choice.heavy_count,
                             load_reduction});
            swaps.push_back({layer_idx,
                             static_cast<int>(choice.light_expert),
                             static_cast<int>(choice.underloaded_participant),
                             static_cast<int>(choice.overloaded_participant),
                             choice.light_count,
                             load_reduction});

            if (!moe_rebalance_policy::applyDynamicOwnershipSwap(
                    socket_loads.data(),
                    placement.data(),
                    choice))
            {
                break;
            }
        }

        return swaps;
    }

} // namespace llaminar2
