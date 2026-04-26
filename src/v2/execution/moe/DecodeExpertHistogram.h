/**
 * @file DecodeExpertHistogram.h
 * @brief Per-layer decode expert utilization tracker with sliding window
 *
 * Tracks expert activation patterns during MoE decode for socket-aware
 * dynamic rebalancing. Designed for zero allocation on the hot path
 * after initialization.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{

    struct DecodeExpertHistogramConfig
    {
        int num_layers = 0;
        int num_experts = 0;
        int top_k = 0;
        int window_size = 256; ///< Decode tokens per window epoch
        std::vector<DeviceId> sockets;
        /// Map expert_id -> socket index (index into sockets vector).
        /// Updated when placement changes.
        std::vector<int> expert_to_socket;
    };

    class DecodeExpertHistogram
    {
    public:
        static constexpr int MAX_TOP_K = 16;

        explicit DecodeExpertHistogram(DecodeExpertHistogramConfig config);

        // ── Hot path (allocation-free) ────────────────────

        /// Record decode routing result for one token at one layer.
        /// expert_indices: [top_k] selected expert IDs
        /// expert_weights: [top_k] corresponding routing weights
        /// Thread-safe via atomics (counts) and per-layer mutex (weighted sums).
        void record(int layer_idx,
                    const int* expert_indices,
                    const float* expert_weights,
                    int top_k);

        // ── Queries (read-only, lock-free for counts) ─────

        /// Get activation count for a specific expert at a specific layer
        uint64_t activationCount(int layer_idx, int expert_id) const;

        /// Get full per-expert activation counts for a layer [num_experts]
        std::vector<uint64_t> layerHistogram(int layer_idx) const;

        /// Get per-socket total activations for a layer [num_sockets]
        std::vector<uint64_t> socketLoads(int layer_idx) const;

        /// Get weighted activation sum for a specific expert at a layer
        float weightedActivation(int layer_idx, int expert_id) const;

        /// Socket imbalance ratio for a layer: max_socket_load / min_socket_load
        /// Returns 1.0 for perfect balance, >1.0 for imbalance.
        /// If min_load == 0: returns infinity when max_load > 0, else 1.0.
        float socketImbalanceRatio(int layer_idx) const;

        /// Average socket imbalance across all layers
        float averageSocketImbalance() const;

        /// Total tokens recorded in current window
        uint64_t windowTokenCount() const;

        /// Whether the window is full (ready for rebalance decision)
        bool windowFull() const;

        /// Current window generation (incremented each reset)
        uint64_t windowGeneration() const;

        // ── Window management ─────────────────────────────

        /// Reset all counters and advance window generation
        void resetWindow();

        /// Update the window size (for adaptive window growth)
        void setWindowSize(int new_size) { config_.window_size = new_size; }

        // ── Placement update ──────────────────────────────

        /// Update expert-to-socket mapping (called after rebalancing)
        void updatePlacement(const std::vector<int>& expert_to_socket);

        // ── Diagnostics ───────────────────────────────────

        /// Top-N hottest experts for a layer (sorted by activation count desc)
        std::vector<std::pair<int, uint64_t>> topExperts(int layer_idx, int n) const;

        /// Human-readable summary of a layer's histogram
        std::string layerSummary(int layer_idx) const;

        const DecodeExpertHistogramConfig& config() const { return config_; }

    private:
        DecodeExpertHistogramConfig config_;

        struct LayerData
        {
            /// Atomic counters for lock-free hot path [num_experts]
            std::vector<std::atomic<uint64_t>> expert_counts;

            /// Protected by mutex (less frequent access)
            mutable std::mutex weight_mutex;
            std::vector<float> weighted_sums;                      // [num_experts]
            std::vector<std::array<uint64_t, MAX_TOP_K>> slot_counts; // [num_experts][MAX_TOP_K]

            explicit LayerData(int num_experts);
            LayerData(LayerData&& other) noexcept;
            LayerData& operator=(LayerData&&) = delete;
            LayerData(const LayerData&) = delete;
            LayerData& operator=(const LayerData&) = delete;
            void reset();
        };

        std::vector<LayerData> layer_data_; // [num_layers]
        std::atomic<uint64_t> window_token_count_{0};
        std::atomic<uint64_t> window_generation_{0};

        // Placement mapping protected by mutex (updated infrequently)
        mutable std::mutex placement_mutex_;
        std::vector<int> expert_to_socket_; // [num_experts]
    };

} // namespace llaminar2
