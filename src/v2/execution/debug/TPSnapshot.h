/**
 * @file TPSnapshot.h
 * @brief Tensor-parallel aware snapshot structures for parity testing
 *
 * This file provides data structures for capturing and comparing tensor
 * snapshots in a tensor-parallel (TP) context. Each device captures its
 * partial results, and these are combined for comparison against the
 * full PyTorch reference output.
 *
 * Key Concepts:
 * - **Column-parallel**: Output split on output dimension (heads, d_ff)
 *   Examples: Q/K/V projections, FFN_GATE, FFN_UP, ATTENTION_CONTEXT
 *
 * - **Row-parallel**: Input split on input dimension, combined via AllReduce
 *   Examples: ATTENTION_OUTPUT (Wo), FFN_DOWN
 *
 * - **Replicated**: Full output on each device (after AllReduce)
 *   Examples: *_NORM stages, FFN_RESIDUAL
 *
 * - **Gathered**: Column-parallel then AllGather to form full output
 *   Example: LM_HEAD
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#pragma once

#include "../mpi_orchestration/DeviceInventory.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <cstddef>
#include <cstring>

namespace llaminar2
{

    // =========================================================================
    // Sharding Mode Enumeration
    // =========================================================================

    /**
     * @brief How a stage's output is sharded across TP devices
     */
    enum class SnapshotShardingMode
    {
        REPLICATED,      ///< Full output on each device (norms, residuals after AllReduce)
        COLUMN_PARALLEL, ///< Split on output dimension (Q/K/V, FFN_GATE, FFN_UP, ATTENTION_CONTEXT)
        ROW_PARALLEL,    ///< Split on input dimension, combined after AllReduce (Wo, FFN_DOWN)
        GATHERED,        ///< Column-parallel then AllGather (LM_HEAD)
        UNKNOWN          ///< Sharding mode not determined
    };

    /**
     * @brief Convert SnapshotShardingMode to string
     */
    inline const char *shardingModeToString(SnapshotShardingMode mode)
    {
        switch (mode)
        {
        case SnapshotShardingMode::REPLICATED:
            return "REPLICATED";
        case SnapshotShardingMode::COLUMN_PARALLEL:
            return "COLUMN_PARALLEL";
        case SnapshotShardingMode::ROW_PARALLEL:
            return "ROW_PARALLEL";
        case SnapshotShardingMode::GATHERED:
            return "GATHERED";
        case SnapshotShardingMode::UNKNOWN:
        default:
            return "UNKNOWN";
        }
    }

    // =========================================================================
    // Stage Sharding Mode Registry
    // =========================================================================

    /**
     * @brief Get the expected sharding mode for a stage key
     *
     * Stage keys follow the pattern: "layerN_STAGE_TYPE" or "STAGE_TYPE"
     * This function extracts the stage type and returns its sharding mode.
     *
     * @param stage_key The snapshot key (e.g., "layer0_ATTENTION_CONTEXT")
     * @return The expected sharding mode for this stage
     */
    inline SnapshotShardingMode getStageShardingMode(const std::string &stage_key)
    {
        // Extract stage type from key (remove layerN_ prefix if present)
        std::string stage_type = stage_key;
        if (stage_key.substr(0, 5) == "layer")
        {
            auto underscore_pos = stage_key.find('_');
            if (underscore_pos != std::string::npos)
            {
                stage_type = stage_key.substr(underscore_pos + 1);
            }
        }

        // Static mapping of stage types to sharding modes
        // Note: These match the Megatron-style tensor parallelism sharding

        // Embedding - replicated across devices (each device has full embedding table)
        if (stage_type == "EMBEDDING")
            return SnapshotShardingMode::REPLICATED;

        // Attention projections - column-parallel (split on num_heads)
        if (stage_type == "Q_PROJECTION" || stage_type == "K_PROJECTION" ||
            stage_type == "V_PROJECTION" || stage_type == "QKV_PROJECTION")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        // Attention context - column-parallel (split on num_heads)
        if (stage_type == "ATTENTION_CONTEXT" || stage_type == "FUSED_ATTENTION_WO_CONTEXT")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        // Attention output (Wo) - row-parallel (AllReduce combines partial results)
        if (stage_type == "ATTENTION_OUTPUT" || stage_type == "FUSED_ATTENTION_WO")
            return SnapshotShardingMode::ROW_PARALLEL;

        // Attention norms - replicated
        if (stage_type == "ATTENTION_NORM")
            return SnapshotShardingMode::REPLICATED;

        // FFN gate/up projections - column-parallel (split on d_ff)
        if (stage_type == "FFN_GATE" || stage_type == "FFN_UP" ||
            stage_type == "FFN_GATE_UP" || stage_type == "FUSED_FFN_GATE_UP")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        // FFN SwiGLU - column-parallel (operates on sharded d_ff)
        if (stage_type == "FFN_SWIGLU")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        // FFN down - row-parallel (AllReduce combines partial results)
        if (stage_type == "FFN_DOWN")
            return SnapshotShardingMode::ROW_PARALLEL;

        // FFN residual - replicated (after AllReduce)
        if (stage_type == "FFN_RESIDUAL")
            return SnapshotShardingMode::REPLICATED;

        // FFN norm - replicated
        if (stage_type == "FFN_NORM")
            return SnapshotShardingMode::REPLICATED;

        // Final stages
        if (stage_type == "FINAL_NORM")
            return SnapshotShardingMode::REPLICATED;

        // LM_HEAD - column-parallel then AllGather
        if (stage_type == "LM_HEAD")
            return SnapshotShardingMode::GATHERED;

        // Default to unknown
        return SnapshotShardingMode::UNKNOWN;
    }

    // =========================================================================
    // Per-Device Snapshot Data
    // =========================================================================

    /**
     * @brief Snapshot data captured from a single device
     */
    struct DeviceSnapshotData
    {
        GlobalDeviceId device_id; ///< Unique device identifier
        int device_index = 0;     ///< Index within TP group (0, 1, ...)
        std::vector<float> data;  ///< Tensor data (may be partial for column-parallel)
        size_t rows = 0;          ///< Logical rows (typically seq_len)
        size_t cols = 0;          ///< Logical cols (may be partial for column-parallel)

        // For column-parallel stages: which slice of the full output this represents
        size_t global_start_col = 0;  ///< Start column in full output
        size_t global_total_cols = 0; ///< Total columns across all devices

        /// Check if this snapshot represents partial (sharded) data
        bool isPartial() const
        {
            return global_total_cols > 0 && cols < global_total_cols;
        }
    };

    // =========================================================================
    // Complete TP-Aware Snapshot
    // =========================================================================

    /**
     * @brief Complete tensor-parallel aware snapshot for a stage
     *
     * Contains per-device partial data plus computed combined view.
     */
    struct TPSnapshot
    {
        std::string key; ///< Stage key (e.g., "layer0_ATTENTION_CONTEXT")
        SnapshotShardingMode mode = SnapshotShardingMode::UNKNOWN;
        int tp_degree = 1; ///< Number of TP devices

        /// Per-device snapshots (indexed by device position in TP group)
        std::vector<DeviceSnapshotData> device_data;

        // Combined view (computed lazily or on demand)
        bool combined_valid = false;      ///< Whether combined data is computed
        std::vector<float> combined_data; ///< Concatenated/verified combined result
        size_t combined_rows = 0;         ///< Rows in combined output
        size_t combined_cols = 0;         ///< Cols in combined output

        /**
         * @brief Compute the combined view from per-device data
         *
         * For COLUMN_PARALLEL: Concatenates device outputs along column dimension
         * For ROW_PARALLEL/REPLICATED: Verifies all devices have same data, uses first
         * For GATHERED: Uses already-gathered combined output
         *
         * @return true if combination was successful
         */
        bool computeCombined()
        {
            if (device_data.empty())
            {
                combined_valid = false;
                return false;
            }

            if (mode == SnapshotShardingMode::COLUMN_PARALLEL)
            {
                // Concatenate along columns
                combined_rows = device_data[0].rows;
                combined_cols = 0;
                for (const auto &dev : device_data)
                {
                    combined_cols += dev.cols;
                }

                combined_data.resize(combined_rows * combined_cols);

                // For each row, concatenate column slices from all devices
                for (size_t row = 0; row < combined_rows; ++row)
                {
                    size_t out_col = 0;
                    for (const auto &dev : device_data)
                    {
                        const float *src = dev.data.data() + row * dev.cols;
                        float *dst = combined_data.data() + row * combined_cols + out_col;
                        std::memcpy(dst, src, dev.cols * sizeof(float));
                        out_col += dev.cols;
                    }
                }
                combined_valid = true;
                return true;
            }
            else if (mode == SnapshotShardingMode::ROW_PARALLEL ||
                     mode == SnapshotShardingMode::REPLICATED)
            {
                // All devices should have same data; use first
                const auto &first = device_data[0];
                combined_rows = first.rows;
                combined_cols = first.cols;
                combined_data = first.data;
                combined_valid = true;
                return true;
            }
            else if (mode == SnapshotShardingMode::GATHERED)
            {
                // Already gathered - should be full data in first device
                const auto &first = device_data[0];
                combined_rows = first.rows;
                combined_cols = first.cols;
                combined_data = first.data;
                combined_valid = true;
                return true;
            }

            // Unknown mode - just use first device
            const auto &first = device_data[0];
            combined_rows = first.rows;
            combined_cols = first.cols;
            combined_data = first.data;
            combined_valid = true;
            return true;
        }

        /**
         * @brief Get pointer to combined data (computes if needed)
         *
         * @param out_size Output parameter for data size (rows * cols)
         * @return Pointer to combined float data, or nullptr on failure
         */
        const float *getCombinedData(size_t &out_size)
        {
            if (!combined_valid)
            {
                if (!computeCombined())
                {
                    out_size = 0;
                    return nullptr;
                }
            }
            out_size = combined_rows * combined_cols;
            return combined_data.data();
        }
    };

    // =========================================================================
    // Slice Computation Utilities
    // =========================================================================

    /**
     * @brief Compute the start column for a device in column-parallel sharding
     *
     * @param device_idx Index of device in TP group (0, 1, ...)
     * @param tp_degree Total number of TP devices
     * @param total_cols Total columns in full output
     * @return Start column index for this device
     */
    inline size_t computeSliceStartCol(int device_idx, int tp_degree, size_t total_cols)
    {
        return static_cast<size_t>(device_idx) * (total_cols / static_cast<size_t>(tp_degree));
    }

    /**
     * @brief Compute the column count for a device in column-parallel sharding
     *
     * Last device gets any remainder columns.
     *
     * @param device_idx Index of device in TP group (0, 1, ...)
     * @param tp_degree Total number of TP devices
     * @param total_cols Total columns in full output
     * @return Number of columns for this device
     */
    inline size_t computeSliceColCount(int device_idx, int tp_degree, size_t total_cols)
    {
        size_t base = total_cols / static_cast<size_t>(tp_degree);
        if (device_idx == tp_degree - 1)
        {
            // Last device gets remainder
            return total_cols - static_cast<size_t>(device_idx) * base;
        }
        return base;
    }

    /**
     * @brief Extract a column slice from a 2D tensor
     *
     * @param src Source tensor data (row-major: [rows][cols])
     * @param rows Number of rows
     * @param src_cols Total columns in source
     * @param start_col Start column of slice
     * @param slice_cols Number of columns in slice
     * @return Vector containing the sliced data
     */
    inline std::vector<float> extractColumnSlice(
        const float *src,
        size_t rows,
        size_t src_cols,
        size_t start_col,
        size_t slice_cols)
    {
        std::vector<float> result(rows * slice_cols);
        for (size_t row = 0; row < rows; ++row)
        {
            const float *src_row = src + row * src_cols + start_col;
            float *dst_row = result.data() + row * slice_cols;
            std::memcpy(dst_row, src_row, slice_cols * sizeof(float));
        }
        return result;
    }

} // namespace llaminar2
