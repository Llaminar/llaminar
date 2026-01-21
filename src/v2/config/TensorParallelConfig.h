/**
 * @file TensorParallelConfig.h
 * @brief Configuration for tensor parallelism across heterogeneous devices
 *
 * Supports both equal splits (backward compatible) and proportional splits
 * for heterogeneous GPU configurations (e.g., NVIDIA 73% + AMD 27%).
 *
 * Design:
 * - DeviceShardingAssignment: Per-device assignment of heads, FFN slices, vocab
 * - TensorParallelConfig: Container with factory methods for common patterns
 *
 * Usage:
 *   // Equal split across 2 GPUs
 *   auto config = TensorParallelConfig::equalSplit(2, n_heads, n_kv_heads, d_ff, vocab);
 *
 *   // Proportional split: NVIDIA gets 73%, AMD gets 27%
 *   auto config = TensorParallelConfig::proportionalSplit(
 *       {DeviceId::cuda(0), DeviceId::rocm(0)},
 *       {0.73f, 0.27f},
 *       n_heads, n_kv_heads, d_ff, vocab);
 */

#pragma once

#include "backends/DeviceId.h"
#include <vector>
#include <optional>
#include <stdexcept>
#include <string>
#include <cmath>

namespace llaminar2
{

    /**
     * @brief Per-device assignment for tensor parallelism
     *
     * Contains the slice of work assigned to a single device, including:
     * - Attention head ranges (Q heads and KV heads for GQA)
     * - FFN dimension slice (gate/up/down projections)
     * - Vocabulary slice (LM head)
     */
    struct DeviceShardingAssignment
    {
        DeviceId device;

        // Attention sharding
        int head_start = 0;    // First Q head index for this device
        int head_count = 0;    // Number of Q heads
        int kv_head_start = 0; // First KV head index
        int kv_head_count = 0; // Number of KV heads (for GQA)

        // FFN sharding
        int d_ff_start = 0; // Start index in FFN dimension
        int d_ff_count = 0; // FFN slice size

        // LM Head sharding
        int vocab_start = 0; // Start index in vocabulary
        int vocab_count = 0; // Vocab slice size

        // Work fraction (for display/logging)
        float work_fraction = 0.0f;

        // Rank in the device group (0, 1, 2, ...)
        int local_rank = 0;

        /**
         * @brief Check if this assignment is valid (non-empty)
         */
        bool isValid() const
        {
            return device.is_valid() && head_count > 0;
        }

        /**
         * @brief Get head end index (exclusive)
         */
        int headEnd() const { return head_start + head_count; }

        /**
         * @brief Get KV head end index (exclusive)
         */
        int kvHeadEnd() const { return kv_head_start + kv_head_count; }

        /**
         * @brief Get d_ff end index (exclusive)
         */
        int dFFEnd() const { return d_ff_start + d_ff_count; }

        /**
         * @brief Get vocab end index (exclusive)
         */
        int vocabEnd() const { return vocab_start + vocab_count; }

        /**
         * @brief String representation for logging
         */
        std::string toString() const;
    };

    /**
     * @brief Configuration for tensor parallelism across devices
     *
     * Supports both equal splits (backward compatible) and proportional splits
     * for heterogeneous GPU configurations.
     *
     * Key design decisions:
     * - Head counts are always integers (no fractional heads)
     * - FFN slice sizes are aligned to 32 for quantization block boundaries
     * - KV heads maintain the Q-head to KV-head ratio for GQA correctness
     * - Work fractions are normalized to sum to 1.0
     */
    class TensorParallelConfig
    {
    public:
        // =========================================================================
        // Constructors
        // =========================================================================

        TensorParallelConfig() = default;
        explicit TensorParallelConfig(std::vector<DeviceShardingAssignment> assignments);

        // =========================================================================
        // Accessors
        // =========================================================================

        const std::vector<DeviceShardingAssignment> &assignments() const { return assignments_; }
        const DeviceShardingAssignment &forDevice(DeviceId device) const;
        const DeviceShardingAssignment &forRank(int rank) const;
        int worldSize() const { return static_cast<int>(assignments_.size()); }
        bool isProportional() const { return is_proportional_; }

        // =========================================================================
        // Totals (for validation)
        // =========================================================================

        int totalHeads() const { return total_heads_; }
        int totalKVHeads() const { return total_kv_heads_; }
        int totalDFF() const { return total_d_ff_; }
        int totalVocab() const { return total_vocab_; }

        // =========================================================================
        // Validation
        // =========================================================================

        bool validate() const;
        std::string validationError() const;

        // =========================================================================
        // Factory Methods
        // =========================================================================

        /**
         * @brief Create equal 1/N splits (backward compatible)
         *
         * Divides work equally among world_size ranks. If work doesn't divide
         * evenly, earlier ranks get the remainder.
         *
         * @param world_size Number of ranks/devices
         * @param n_heads Total number of Q attention heads
         * @param n_kv_heads Total number of KV heads (for GQA)
         * @param d_ff FFN intermediate dimension
         * @param vocab_size Vocabulary size
         * @param devices Optional device IDs (defaults to cuda(0), cuda(1), ...)
         * @return TensorParallelConfig with equal splits
         */
        static TensorParallelConfig equalSplit(
            int world_size,
            int n_heads,
            int n_kv_heads,
            int d_ff,
            int vocab_size,
            std::optional<std::vector<DeviceId>> devices = std::nullopt);

        /**
         * @brief Create proportional splits based on work fractions
         *
         * Assigns work to each device based on its work fraction. Useful for
         * heterogeneous configurations (e.g., fast GPU gets more work).
         *
         * Key behaviors:
         * - Work fractions are normalized to sum to 1.0
         * - Head counts are rounded to maintain integer heads
         * - KV heads maintain the Q-to-KV ratio for GQA
         * - FFN slices are aligned to 32-element boundaries
         *
         * @param devices Devices to distribute work across
         * @param work_fractions Relative work fraction per device (normalized)
         * @param n_heads Total number of Q attention heads
         * @param n_kv_heads Total number of KV heads (for GQA)
         * @param d_ff FFN intermediate dimension
         * @param vocab_size Vocabulary size
         * @return TensorParallelConfig with proportional splits
         *
         * @throws std::invalid_argument if devices.size() != work_fractions.size()
         * @throws std::invalid_argument if any work fraction is negative
         */
        static TensorParallelConfig proportionalSplit(
            const std::vector<DeviceId> &devices,
            const std::vector<float> &work_fractions,
            int n_heads,
            int n_kv_heads,
            int d_ff,
            int vocab_size);

        /**
         * @brief Create single-device configuration (no parallelism)
         *
         * @param device The single device to use
         * @param n_heads Total number of Q attention heads
         * @param n_kv_heads Total number of KV heads
         * @param d_ff FFN intermediate dimension
         * @param vocab_size Vocabulary size
         * @return TensorParallelConfig for single device
         */
        static TensorParallelConfig singleDevice(
            DeviceId device,
            int n_heads,
            int n_kv_heads,
            int d_ff,
            int vocab_size);

        /**
         * @brief String representation for logging
         */
        std::string toString() const;

    private:
        std::vector<DeviceShardingAssignment> assignments_;
        int total_heads_ = 0;
        int total_kv_heads_ = 0;
        int total_d_ff_ = 0;
        int total_vocab_ = 0;
        bool is_proportional_ = false;

        void computeTotals();

        /**
         * @brief Round value to alignment boundary (round down)
         *
         * Used for FFN slices to maintain quantization block alignment.
         */
        static int roundToAlignment(int value, int alignment);

        /**
         * @brief Distribute integer count among N buckets proportionally
         *
         * Ensures sum equals total, handles rounding without losing elements.
         *
         * @param total Total count to distribute
         * @param fractions Normalized fractions (sum to 1.0)
         * @param alignment Alignment constraint (1 for heads, 32 for FFN)
         * @return Vector of counts, one per bucket
         */
        static std::vector<int> distributeProportionally(
            int total,
            const std::vector<float> &fractions,
            int alignment = 1);
    };

} // namespace llaminar2
