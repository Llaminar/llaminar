/**
 * @file MoERebalanceController.cpp
 * @brief Orchestrates MoE decode histogram tracking and socket-aware rebalancing
 */

#include "MoERebalanceController.h"
#include "../../utils/Logger.h"

#include <algorithm>
#include <numeric>

namespace llaminar2
{

    MoERebalanceController::MoERebalanceController(Config config)
        : config_(std::move(config)),
          current_placement_(config_.initial_expert_to_socket)
    {
        if (config_.mode == MoERebalanceMode::OFF)
            return;

        // Create histogram for OBSERVE and DYNAMIC modes
        DecodeExpertHistogramConfig hcfg;
        hcfg.num_layers = config_.num_layers;
        hcfg.num_experts = config_.num_experts;
        hcfg.top_k = config_.top_k;
        hcfg.window_size = config_.window_size;
        hcfg.sockets = config_.sockets;
        hcfg.expert_to_socket = config_.initial_expert_to_socket;
        histogram_ = std::make_unique<DecodeExpertHistogram>(std::move(hcfg));

        // Create rebalancer for DYNAMIC mode
        if (config_.mode == MoERebalanceMode::DYNAMIC)
        {
            rebalancer_ = std::make_unique<SocketAwareRebalancer>(config_.rebalance_config);
        }

        LOG_INFO("[MoERebalanceController] Initialized: mode="
                 << (config_.mode == MoERebalanceMode::OBSERVE ? "OBSERVE" : "DYNAMIC")
                 << " layers=" << config_.num_layers
                 << " experts=" << config_.num_experts
                 << " top_k=" << config_.top_k
                 << " window=" << config_.window_size
                 << " sockets=" << config_.sockets.size());
    }

    bool MoERebalanceController::shouldRebalance() const
    {
        return config_.mode == MoERebalanceMode::DYNAMIC
               && histogram_
               && histogram_->windowFull();
    }

    std::vector<int> MoERebalanceController::rebalance()
    {
        if (!rebalancer_ || !histogram_)
            return {};

        auto proposal = rebalancer_->propose(*histogram_);

        if (proposal.empty())
        {
            LOG_DEBUG("[MoERebalanceController] No beneficial swaps found, resetting window");
            histogram_->resetWindow();
            return {};
        }

        // Apply swaps to get new placement
        auto new_placement = rebalancer_->apply(current_placement_, proposal);
        current_placement_ = new_placement;

        // Update histogram's socket mapping for next window
        histogram_->updatePlacement(current_placement_);
        histogram_->resetWindow();

        total_rebalances_++;
        total_swaps_ += proposal.numSwaps();

        LOG_INFO("[MoERebalanceController] Rebalance #" << total_rebalances_
                 << ": " << proposal.summary());

        return new_placement;
    }

    void MoERebalanceController::logHistogramSummary() const
    {
        if (!histogram_)
        {
            LOG_INFO("[MoERebalanceController] No histogram (mode=OFF)");
            return;
        }

        LOG_INFO("[MoERebalanceController] Histogram summary (window tokens="
                 << histogram_->windowTokenCount()
                 << " gen=" << histogram_->windowGeneration()
                 << " avg_imbalance=" << histogram_->averageSocketImbalance() << "):");

        for (int l = 0; l < config_.num_layers; ++l)
        {
            LOG_INFO("  " << histogram_->layerSummary(l));
        }
    }

    std::vector<std::vector<bool>> MoERebalanceController::computeExpertMasks(int socket_id) const
    {
        const int num_layers = config_.num_layers;
        const int num_experts = config_.num_experts;

        std::vector<std::vector<bool>> masks(num_layers);

        if (use_per_layer_placement_ && !per_layer_placement_.empty())
        {
            // Per-layer placement (after rebalanceLPT)
            for (int l = 0; l < num_layers; ++l)
            {
                masks[l].resize(num_experts, false);
                for (int e = 0; e < num_experts; ++e)
                {
                    if (per_layer_placement_[l][e] == socket_id)
                        masks[l][e] = true;
                }
            }
        }
        else
        {
            // Global placement (initial or after swap-based rebalance)
            std::vector<bool> global_mask(num_experts, false);
            for (int e = 0; e < num_experts; ++e)
            {
                if (e < static_cast<int>(current_placement_.size()) && current_placement_[e] == socket_id)
                    global_mask[e] = true;
            }
            masks.assign(num_layers, global_mask);
        }

        return masks;
    }

    void MoERebalanceController::rebalanceLPT()
    {
        if (!histogram_)
            return;

        const int num_layers = config_.num_layers;
        const int num_experts = config_.num_experts;
        const int num_sockets = static_cast<int>(config_.sockets.size());

        if (num_sockets < 2)
            return;

        // Aggregate per-expert activation counts across ALL layers.
        // This produces a single global partition optimized for the average case,
        // ensuring each rank still has ~N/2 experts and GEMM engines (no 2x memory).
        std::vector<uint64_t> total_counts(num_experts, 0);
        for (int l = 0; l < num_layers; ++l)
        {
            auto layer_counts = histogram_->layerHistogram(l);
            for (int e = 0; e < num_experts; ++e)
                total_counts[e] += layer_counts[e];
        }

        // Compute imbalance with initial contiguous partition
        std::vector<uint64_t> old_loads(num_sockets, 0);
        for (int e = 0; e < num_experts; ++e)
            old_loads[current_placement_[e]] += total_counts[e];
        auto [old_min_it, old_max_it] = std::minmax_element(old_loads.begin(), old_loads.end());
        float imbalance_before = (*old_min_it > 0)
            ? static_cast<float>(*old_max_it) / static_cast<float>(*old_min_it)
            : 1.0f;

        // LPT: sort experts by total activation count descending
        std::vector<int> sorted(num_experts);
        std::iota(sorted.begin(), sorted.end(), 0);
        std::sort(sorted.begin(), sorted.end(),
                  [&](int a, int b) { return total_counts[a] > total_counts[b]; });

        // Greedy assign to least-loaded socket
        std::vector<uint64_t> loads(num_sockets, 0);
        std::vector<int> new_placement(num_experts);

        for (int idx = 0; idx < num_experts; ++idx)
        {
            int e = sorted[idx];
            int target = 0;
            for (int s = 1; s < num_sockets; ++s)
            {
                if (loads[s] < loads[target])
                    target = s;
            }
            new_placement[e] = target;
            loads[target] += total_counts[e];
        }

        // Compute imbalance after
        auto [new_min_it, new_max_it] = std::minmax_element(loads.begin(), loads.end());
        float imbalance_after = (*new_min_it > 0)
            ? static_cast<float>(*new_max_it) / static_cast<float>(*new_min_it)
            : 1.0f;

        // Count how many experts changed socket
        int experts_moved = 0;
        for (int e = 0; e < num_experts; ++e)
        {
            if (new_placement[e] != current_placement_[e])
                experts_moved++;
        }

        // Also compute per-layer imbalance improvement
        float per_layer_before = 0.0f, per_layer_after = 0.0f;
        for (int l = 0; l < num_layers; ++l)
        {
            auto lc = histogram_->layerHistogram(l);
            std::vector<uint64_t> lb(num_sockets, 0), la(num_sockets, 0);
            for (int e = 0; e < num_experts; ++e)
            {
                lb[current_placement_[e]] += lc[e];
                la[new_placement[e]] += lc[e];
            }
            auto [lb_min, lb_max] = std::minmax_element(lb.begin(), lb.end());
            auto [la_min, la_max] = std::minmax_element(la.begin(), la.end());
            per_layer_before += (*lb_min > 0) ? float(*lb_max) / float(*lb_min) : 1.0f;
            per_layer_after += (*la_min > 0) ? float(*la_max) / float(*la_min) : 1.0f;
        }
        per_layer_before /= num_layers;
        per_layer_after /= num_layers;

        current_placement_ = new_placement;
        use_per_layer_placement_ = false; // global partition, not per-layer
        total_rebalances_++;

        LOG_INFO("[MoERebalanceController] LPT global rebalance: "
                 << experts_moved << "/" << num_experts << " experts moved, "
                 << "aggregate imbalance " << imbalance_before << " -> " << imbalance_after
                 << ", per-layer avg " << per_layer_before << " -> " << per_layer_after);

        histogram_->resetWindow();
    }

} // namespace llaminar2
