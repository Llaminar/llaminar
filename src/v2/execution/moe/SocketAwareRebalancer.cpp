/**
 * @file SocketAwareRebalancer.cpp
 * @brief Socket-load-aware expert rebalancer implementation
 */

#include "SocketAwareRebalancer.h"
#include "../../utils/Logger.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>

namespace llaminar2
{

    // ── SocketRebalanceProposal ───────────────────────

    std::string SocketRebalanceProposal::summary() const
    {
        if (empty())
            return "SocketRebalanceProposal: no swaps proposed";

        int num_layers = static_cast<int>(layer_metrics.size());
        std::ostringstream ss;
        ss << "SocketRebalanceProposal: " << numSwaps() << " swap"
           << (numSwaps() != 1 ? "s" : "") << " across " << num_layers
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
        const DecodeExpertHistogram& histogram) const
    {
        SocketRebalanceProposal proposal;
        proposal.window_generation = histogram.windowGeneration();

        const auto& hcfg = histogram.config();
        const int num_layers = hcfg.num_layers;
        const int num_sockets = static_cast<int>(hcfg.sockets.size());

        if (num_layers <= 0 || num_sockets < 2) {
            return proposal;
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

            // Get current placement
            const auto& expert_to_socket = hcfg.expert_to_socket;

            auto layer_swaps = proposeForLayer(l, expert_counts, expert_to_socket, num_sockets);
            if (layer_swaps.empty())
                continue;

            // Enforce max_total_swaps
            int remaining = config_.max_total_swaps - total_swaps;
            if (remaining <= 0)
                break;
            if (static_cast<int>(layer_swaps.size()) > remaining)
                layer_swaps.resize(remaining);

            // Compute estimated imbalance after swaps
            float imbalance_before = histogram.socketImbalanceRatio(l);

            // Simulate the swaps to estimate new imbalance
            std::vector<int> simulated_placement = expert_to_socket;
            for (const auto& swap : layer_swaps) {
                // Find a partner: the expert from to_socket that we swap with
                // The swap already contains the from/to info
                simulated_placement[swap.expert_id] = swap.to_socket;
            }

            // Actually, swaps come in pairs (heavy from overloaded ↔ light from underloaded)
            // The proposeForLayer already handles the pair logic; we need to apply
            // the full swap set. Each ExpertSwap represents one side of a swap.
            // Let's recompute loads with simulated placement.
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

            // Record cooldown
            layer_last_rebalanced_[l] = current_gen;
        }

        if (!proposal.empty()) {
            LOG_DEBUG(proposal.summary());
        }

        return proposal;
    }

    std::vector<int> SocketAwareRebalancer::apply(
        const std::vector<int>& current_placement,
        const SocketRebalanceProposal& proposal) const
    {
        std::vector<int> new_placement = current_placement;
        for (const auto& swap : proposal.swaps) {
            new_placement[swap.expert_id] = swap.to_socket;
        }
        return new_placement;
    }

    std::vector<ExpertSwap> SocketAwareRebalancer::proposeForLayer(
        int layer_idx,
        const std::vector<uint64_t>& expert_counts,
        const std::vector<int>& expert_to_socket,
        int num_sockets) const
    {
        std::vector<ExpertSwap> swaps;
        const int num_experts = static_cast<int>(expert_counts.size());

        // Working copy of placement and loads (we mutate as we accept swaps)
        std::vector<int> placement = expert_to_socket;
        std::vector<uint64_t> socket_loads(num_sockets, 0);
        for (int e = 0; e < num_experts; ++e) {
            socket_loads[placement[e]] += expert_counts[e];
        }

        for (int swap_iter = 0; swap_iter < config_.max_swaps_per_layer; ++swap_iter) {
            // Find overloaded (max) and underloaded (min) sockets
            auto max_it = std::max_element(socket_loads.begin(), socket_loads.end());
            auto min_it = std::min_element(socket_loads.begin(), socket_loads.end());
            int overloaded = static_cast<int>(std::distance(socket_loads.begin(), max_it));
            int underloaded = static_cast<int>(std::distance(socket_loads.begin(), min_it));

            if (overloaded == underloaded)
                break;

            // Check imbalance threshold
            uint64_t max_load = *max_it;
            uint64_t min_load = *min_it;
            if (min_load == 0) {
                if (max_load == 0) break; // no activations at all
                // Infinite imbalance — proceed
            } else {
                float imbalance = static_cast<float>(max_load) / static_cast<float>(min_load);
                if (imbalance < config_.imbalance_threshold)
                    break; // already balanced enough
            }

            // Gather experts on overloaded socket, sorted by count DESC
            std::vector<int> over_experts;
            for (int e = 0; e < num_experts; ++e) {
                if (placement[e] == overloaded)
                    over_experts.push_back(e);
            }
            std::sort(over_experts.begin(), over_experts.end(),
                      [&](int a, int b) { return expert_counts[a] > expert_counts[b]; });

            // Gather experts on underloaded socket, sorted by count ASC
            std::vector<int> under_experts;
            for (int e = 0; e < num_experts; ++e) {
                if (placement[e] == underloaded)
                    under_experts.push_back(e);
            }
            std::sort(under_experts.begin(), under_experts.end(),
                      [&](int a, int b) { return expert_counts[a] < expert_counts[b]; });

            if (over_experts.empty() || under_experts.empty())
                break;

            // Pick heaviest from overloaded, lightest from underloaded
            int heavy_expert = over_experts[0];
            int light_expert = under_experts[0];

            // Compute expected load after swap
            uint64_t heavy_count = expert_counts[heavy_expert];
            uint64_t light_count = expert_counts[light_expert];

            uint64_t new_over_load = max_load - heavy_count + light_count;
            uint64_t new_under_load = min_load - light_count + heavy_count;

            // Compute new imbalance across ALL sockets (not just these two)
            std::vector<uint64_t> new_loads = socket_loads;
            new_loads[overloaded] = new_over_load;
            new_loads[underloaded] = new_under_load;

            auto new_max_it = std::max_element(new_loads.begin(), new_loads.end());
            auto new_min_it = std::min_element(new_loads.begin(), new_loads.end());
            uint64_t new_max = *new_max_it;
            uint64_t new_min = *new_min_it;

            float old_imbalance = (min_load > 0)
                ? static_cast<float>(max_load) / static_cast<float>(min_load)
                : std::numeric_limits<float>::infinity();
            float new_imbalance = (new_min > 0)
                ? static_cast<float>(new_max) / static_cast<float>(new_min)
                : std::numeric_limits<float>::infinity();

            // Check if the swap improves things enough
            if (std::isfinite(old_imbalance) && std::isfinite(new_imbalance)) {
                float improvement = (old_imbalance - new_imbalance) / old_imbalance;
                if (improvement < config_.min_improvement_ratio)
                    break; // marginal improvement, stop

                float load_reduction = old_imbalance - new_imbalance;

                // Accept the swap — emit two ExpertSwap entries
                swaps.push_back({layer_idx, heavy_expert, overloaded, underloaded,
                                 heavy_count, load_reduction});
                swaps.push_back({layer_idx, light_expert, underloaded, overloaded,
                                 light_count, load_reduction});
            } else if (!std::isfinite(old_imbalance) && std::isfinite(new_imbalance)) {
                // Going from infinite to finite is always an improvement
                swaps.push_back({layer_idx, heavy_expert, overloaded, underloaded,
                                 heavy_count, new_imbalance});
                swaps.push_back({layer_idx, light_expert, underloaded, overloaded,
                                 light_count, new_imbalance});
            } else if (!std::isfinite(old_imbalance) && !std::isfinite(new_imbalance)) {
                // Both infinite (e.g., 3+ sockets where some have 0 load).
                // Accept if the max load decreased (spreading work toward balance).
                if (new_max < max_load) {
                    swaps.push_back({layer_idx, heavy_expert, overloaded, underloaded,
                                     heavy_count, 0.0f});
                    swaps.push_back({layer_idx, light_expert, underloaded, overloaded,
                                     light_count, 0.0f});
                } else {
                    break;
                }
            } else {
                break; // can't improve
            }

            // Update working state
            placement[heavy_expert] = underloaded;
            placement[light_expert] = overloaded;
            socket_loads[overloaded] = new_over_load;
            socket_loads[underloaded] = new_under_load;
        }

        return swaps;
    }

} // namespace llaminar2
