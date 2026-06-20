/**
 * @file SocketAwareRebalancer.h
 * @brief Socket-load-aware expert rebalancer using decode histograms
 *
 * Analyzes per-layer expert activation histograms from DecodeExpertHistogram
 * and proposes expert-swap movements between CPU sockets to reduce
 * decode load imbalance. Uses hysteresis and cooldowns to prevent thrashing.
 */

#pragma once

#include "DecodeExpertHistogram.h"
#include <string>
#include <vector>

namespace llaminar2
{

    struct SocketRebalanceConfig
    {
        /// Minimum socket imbalance ratio to trigger rebalancing (e.g., 1.3 = 30% imbalance)
        float imbalance_threshold = 1.3f;

        /// Maximum expert swaps per rebalance cycle per layer
        int max_swaps_per_layer = 4;

        /// Maximum total expert swaps per cycle across all layers
        int max_total_swaps = 16;

        /// Minimum improvement required to accept a swap (prevent marginal moves)
        float min_improvement_ratio = 0.05f;

        /// Cooldown: minimum window generations between rebalancing the same layer
        int layer_cooldown_generations = 2;

        /// Minimum activations in a window before considering rebalancing
        uint64_t min_window_activations = 64;
    };

    /// A proposed swap: move expert from overloaded socket to underloaded socket
    struct ExpertSwap
    {
        int layer_idx;
        int expert_id;
        int from_socket;   ///< Socket index (into sockets vector)
        int to_socket;     ///< Socket index
        uint64_t activation_count; ///< Expert's activation count (for logging)
        float expected_load_reduction; ///< Expected reduction in imbalance ratio
    };

    struct SocketRebalanceProposal
    {
        std::vector<ExpertSwap> swaps;
        uint64_t window_generation; ///< Histogram window generation this proposal is based on

        /// Per-layer imbalance before and after (estimated)
        struct LayerMetrics
        {
            int layer_idx;
            float imbalance_before;
            float estimated_imbalance_after;
            int num_swaps;
        };
        std::vector<LayerMetrics> layer_metrics;

        bool empty() const { return swaps.empty(); }
        int numSwaps() const { return static_cast<int>(swaps.size()); }

        /// Summary string for logging
        std::string summary() const;
    };

    class SocketAwareRebalancer
    {
    public:
        explicit SocketAwareRebalancer(SocketRebalanceConfig config = {});

        /// Analyze histogram and propose expert swaps to reduce socket imbalance.
        /// Returns empty proposal if no beneficial swaps found.
        SocketRebalanceProposal propose(const DecodeExpertHistogram& histogram) const;

        /// Apply a proposal: returns the new expert_to_socket mapping.
        /// The caller is responsible for updating the histogram's placement.
        /// current_placement: [num_experts] current socket assignments
        /// Returns: updated [num_experts] socket assignments
        std::vector<int> apply(
            const std::vector<int>& current_placement,
            const SocketRebalanceProposal& proposal) const;

        const SocketRebalanceConfig& config() const { return config_; }

    private:
        SocketRebalanceConfig config_;

        /// Per-layer cooldown tracking: layer_idx -> last rebalanced generation
        /// UINT64_MAX means "never rebalanced"
        mutable std::vector<uint64_t> layer_last_rebalanced_;

        /// Propose swaps for a single layer.
        /// expert_counts: [num_experts] activation counts
        /// expert_to_socket: [num_experts] current socket assignments
        /// num_sockets: number of sockets
        /// Returns proposed swaps for this layer.
        std::vector<ExpertSwap> proposeForLayer(
            int layer_idx,
            const std::vector<uint64_t>& expert_counts,
            const std::vector<int>& expert_to_socket,
            int num_sockets) const;
    };

} // namespace llaminar2
