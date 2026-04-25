/**
 * @file MoERebalanceController.h
 * @brief Orchestrates MoE decode histogram tracking and socket-aware rebalancing
 *
 * Lifecycle:
 * 1. Created at graph-build time with model config
 * 2. Histogram pointer passed to MoEFFNStage params for recording
 * 3. After each decode step, caller checks shouldRebalance()
 * 4. If true, caller calls rebalance() which proposes + applies swaps
 * 5. Updated placement is available for next decode step
 */

#pragma once

#include "DecodeExpertHistogram.h"
#include "SocketAwareRebalancer.h"

#include <memory>
#include <vector>

namespace llaminar2
{

    enum class MoERebalanceMode
    {
        OFF,     ///< No histogram tracking, no rebalancing
        OBSERVE, ///< Track histograms but don't rebalance (for profiling)
        DYNAMIC  ///< Track histograms and dynamically rebalance
    };

    class MoERebalanceController
    {
    public:
        struct Config
        {
            MoERebalanceMode mode = MoERebalanceMode::OFF;
            int num_layers = 0;
            int num_experts = 0;
            int top_k = 0;
            int window_size = 256;
            std::vector<DeviceId> sockets;             ///< e.g. {cpu:0, cpu:1}
            std::vector<int> initial_expert_to_socket;  ///< [num_experts]
            SocketRebalanceConfig rebalance_config;
        };

        explicit MoERebalanceController(Config config);

        /// Get histogram pointer for MoEFFNStage params (nullptr if OFF)
        DecodeExpertHistogram* histogram() { return histogram_.get(); }

        /// Check if rebalancing should be attempted (window full + mode == DYNAMIC)
        bool shouldRebalance() const;

        /// Propose and apply rebalance (swap-based, global placement).
        /// Returns the new expert_to_socket mapping.
        /// Returns empty vector if no rebalancing was done.
        /// The caller is responsible for updating MoEFFNStage local_expert_start/count.
        std::vector<int> rebalance();

        /// Compute optimal per-layer partition using LPT (Longest Processing Time First).
        /// Each layer gets its own independently-optimized expert assignment.
        /// This is more effective than swap-based rebalancing because layer routing
        /// patterns differ — an expert popular in layer 5 may be rare in layer 20.
        void rebalanceLPT();

        /// Get current global expert-to-socket mapping (used by swap-based rebalance)
        const std::vector<int>& currentPlacement() const { return current_placement_; }

        /// Get the rebalance mode
        MoERebalanceMode mode() const { return config_.mode; }

        /// Get total rebalances performed
        int totalRebalances() const { return total_rebalances_; }

        /// Get total swaps performed across all rebalances
        int totalSwaps() const { return total_swaps_; }

        /// Log current histogram summary (for OBSERVE mode)
        void logHistogramSummary() const;

        /// Compute per-layer expert masks for a given socket/rank.
        /// Returns a vector of num_layers expert masks (each size num_experts).
        /// expert_mask[layer][expert] == true means this rank computes that expert.
        /// After rebalanceLPT(), uses per-layer placement. Otherwise uses global placement.
        std::vector<std::vector<bool>> computeExpertMasks(int socket_id) const;

    private:
        Config config_;
        std::unique_ptr<DecodeExpertHistogram> histogram_;
        std::unique_ptr<SocketAwareRebalancer> rebalancer_;
        std::vector<int> current_placement_;                      ///< Global placement (swap-based)
        std::vector<std::vector<int>> per_layer_placement_;       ///< Per-layer placement (LPT-based)
        bool use_per_layer_placement_ = false;                     ///< True after rebalanceLPT()
        int total_rebalances_ = 0;
        int total_swaps_ = 0;
    };

} // namespace llaminar2
