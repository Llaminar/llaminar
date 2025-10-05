/**
 * @file prefill_diagnostics.h
 * @brief Prefill diagnostics helpers for baseline comparison and FFN shard tracing.
 * @author David Sanftenberg
 *
 * Extracted from distributed_transformer_pipeline.cpp to improve modularity.
 * Preserves all existing log tags and behavior.
 */
#pragma once

#include "tensors/tensor_base.h"
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <cstddef>

namespace llaminar
{
    /**
     * @brief Statistical summary of a float buffer.
     */
    struct BufferStats
    {
        float min = 0.0f;
        float max = 0.0f;
        double mean = 0.0;
        double rms = 0.0;
        double l2 = 0.0;
        double stddev = 0.0;
    };

    /**
     * @brief Difference statistics between two buffers.
     */
    struct DiffSummary
    {
        double max_abs = 0.0;
        double mean_abs = 0.0;
        double rel_l2 = 0.0;
        size_t worst_index = 0;
        float value_a = 0.0f;
        float value_b = 0.0f;
    };

    /**
     * @brief Compute statistical summary of a float buffer.
     *
     * @param data Buffer to analyze
     * @param size Number of elements
     * @return BufferStats with min, max, mean, rms, l2, stddev
     */
    BufferStats computeBufferStats(const float *data, size_t size);

    /**
     * @brief Compute difference statistics between two float buffers.
     *
     * Calculates max absolute error, mean absolute error, relative L2 norm,
     * and identifies the worst-case element for debugging.
     *
     * @param a First buffer (e.g., incremental result)
     * @param b Second buffer (e.g., reference/replay result)
     * @param size Number of elements in each buffer
     * @return DiffSummary containing error metrics and worst-case location
     */
    DiffSummary computeDiffSummary(const float *a, const float *b, size_t size);

    /**
     * @brief Registry for storing and comparing prefill baseline tensors.
     *
     * Thread-safe singleton for capture/compare diagnostics.
     * Uses "[PrefillBaseline]" log tag.
     */
    class PrefillBaselineRegistry
    {
    public:
        static PrefillBaselineRegistry &instance();

        void clear();
        bool ensure(const std::string &name, const float *data, size_t count);
        bool fetch(const std::string &name, std::vector<float> &out) const;

    private:
        PrefillBaselineRegistry() = default;
        PrefillBaselineRegistry(const PrefillBaselineRegistry &) = delete;
        PrefillBaselineRegistry &operator=(const PrefillBaselineRegistry &) = delete;

        mutable std::mutex mutex_;
        std::unordered_map<std::string, std::vector<float>> storage_;
    };

    /**
     * @brief Handle prefill stage snapshot for baseline capture/compare.
     *
     * Rank-0 only operation. Uses "[PrefillBaseline]" log tags.
     *
     * @param rank MPI rank (only rank 0 processes)
     * @param name Snapshot identifier
     * @param data Tensor data pointer
     * @param count Number of elements
     * @param cols Number of columns (for logging, unused)
     * @param warn_threshold Relative L2 threshold for warnings
     * @param capture_enabled Whether to capture baseline
     * @param compare_enabled Whether to compare against baseline
     */
    void handle_prefill_stage_snapshot(int rank, const std::string &name, const float *data, size_t count,
                                       int cols, double warn_threshold, bool capture_enabled, bool compare_enabled);

    /**
     * @brief Log tensor row preview if FFN shard tracing is enabled.
     *
     * Uses "[PrefillFFNPreview]" log tag.
     *
     * @param label Shard label to check against trace configuration
     * @param tensor Tensor to preview
     * @param default_preview_cols Default number of columns to show
     */
    void logFFNRowPreviewIfEnabled(const std::string &label,
                                   const std::shared_ptr<TensorBase> &tensor,
                                   size_t default_preview_cols = 8);

    /**
     * @brief Check if FFN shard tracing is enabled for given label.
     *
     * @param label Shard identifier
     * @return true if tracing enabled for this shard
     */
    bool isFFNShardTracingEnabledFor(const std::string &label);

} // namespace llaminar
