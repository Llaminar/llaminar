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
#include <functional>
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

    enum class ExpertHistogramSource
    {
        DecodeToken,
        PrefillChunk,
        SyntheticTest,
    };

    struct RoutedExpertHistogramMerge
    {
        ExpertHistogramSource source = ExpertHistogramSource::SyntheticTest;
        int layer_idx = -1;
        int real_token_count = 0;
        int bucket_token_count = 0;
        int top_k = 0;
        int route_stride = 0;
        bool count_window_tokens = true;
    };

    struct ExpertHistogramMergeResult
    {
        bool ok = false;
        uint64_t tokens_counted = 0;
        uint64_t activations_merged = 0;
        std::string error;

        explicit operator bool() const { return ok; }
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
                    const int *expert_indices,
                    const float *expert_weights,
                    int top_k);

        /// Record only the decode-token boundary for a routed layer.
        /// This is used by graph-captured device routing paths where expert
        /// counts stay on device and are merged in batches. The token window is
        /// still advanced once per decode token by the final MoE layer.
        void recordTokenBoundary(int layer_idx, uint64_t token_count = 1);

        /// Merge per-expert activation counts that were accumulated outside the
        /// host hot path, for example in a runtime-table device histogram.
        /// Weighted activation sums are intentionally not reconstructed here.
        /// If count_window_tokens is true and layer_idx is the final MoE layer,
        /// windowTokenCount advances by total_count / configured top_k.
        void mergeLayerCounts(int layer_idx,
                              const uint64_t *expert_counts,
                              int num_experts,
                              bool count_window_tokens = false);

        /// Merge a routed-token slice into the histogram. Only the leading
        /// real_token_count rows are counted; padded bucket rows are ignored.
        ExpertHistogramMergeResult mergeRoutedExpertRows(
            const int *expert_indices,
            const RoutedExpertHistogramMerge &merge);

        using RuntimeHistogramSyncCallback = std::function<bool()>;

        /// Register a lazy sync source for device/runtime histograms.
        /// Callbacks should merge pending counts into this histogram and reset
        /// their source counters so repeated syncs are idempotent.
        void registerRuntimeHistogramSync(RuntimeHistogramSyncCallback callback);

        /// Merge all registered runtime histogram sources into this host view.
        bool syncRuntimeHistograms();

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
        void updatePlacement(const std::vector<int> &expert_to_socket);

        // ── Diagnostics ───────────────────────────────────

        /// Top-N hottest experts for a layer (sorted by activation count desc)
        std::vector<std::pair<int, uint64_t>> topExperts(int layer_idx, int n) const;

        /// Human-readable summary of a layer's histogram
        std::string layerSummary(int layer_idx) const;

        const DecodeExpertHistogramConfig &config() const { return config_; }

    private:
        DecodeExpertHistogramConfig config_;

        struct LayerData
        {
            /// Atomic counters for lock-free hot path [num_experts]
            std::vector<std::atomic<uint64_t>> expert_counts;

            /// Protected by mutex (less frequent access)
            mutable std::mutex weight_mutex;
            std::vector<float> weighted_sums;                         // [num_experts]
            std::vector<std::array<uint64_t, MAX_TOP_K>> slot_counts; // [num_experts][MAX_TOP_K]

            explicit LayerData(int num_experts);
            LayerData(LayerData &&other) noexcept;
            LayerData &operator=(LayerData &&) = delete;
            LayerData(const LayerData &) = delete;
            LayerData &operator=(const LayerData &) = delete;
            void reset();
        };

        std::vector<LayerData> layer_data_; // [num_layers]
        std::atomic<uint64_t> window_token_count_{0};
        std::atomic<uint64_t> window_generation_{0};

        // Placement mapping protected by mutex (updated infrequently)
        mutable std::mutex placement_mutex_;
        std::vector<int> expert_to_socket_; // [num_experts]

        mutable std::mutex runtime_sync_mutex_;
        std::vector<RuntimeHistogramSyncCallback> runtime_sync_callbacks_;
    };

    using DomainExpertHistogram = DecodeExpertHistogram;

} // namespace llaminar2
