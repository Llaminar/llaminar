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

#include "../StageShardingMode.h"
#include "../mpi_orchestration/DeviceInventory.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <cstddef>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace llaminar2
{

    // =========================================================================
    // Stage Sharding Mode Registry
    // =========================================================================

    /**
     * @brief Extract stage type suffix from a snapshot key
     *
     * Strips the "layerN_" prefix if present.
     * E.g., "layer0_ATTENTION_CONTEXT" → "ATTENTION_CONTEXT", "LM_HEAD" → "LM_HEAD"
     */
    inline std::string extractStageType(const std::string &stage_key)
    {
        std::string stage_type = stage_key;
        if (stage_key.substr(0, 5) == "layer")
        {
            auto underscore_pos = stage_key.find('_');
            if (underscore_pos != std::string::npos)
            {
                stage_type = stage_key.substr(underscore_pos + 1);
            }
        }
        return stage_type;
    }

    /**
     * @brief Get sharding mode for a stage key using a schema-provided config map
     *
     * Preferred overload: uses a model-specific StageShardingConfig returned
     * by ISchemaFactory::getStageShardingConfig(). Returns UNKNOWN for stage
     * types not present in the map.  A schema key ending in `*` declares a
     * parameterized snapshot family and is matched as a prefix after exact-key
     * lookup.  Callers that combine multi-device snapshots must treat UNKNOWN
     * as a contract error, not as permission to choose an arbitrary TP
     * participant.
     *
     * @param stage_key The snapshot key (e.g., "layer0_ATTENTION_CONTEXT")
     * @param config    Stage type → SnapshotShardingMode map from the schema factory
     * @return The expected sharding mode for this stage
     */
    inline SnapshotShardingMode getStageShardingMode(
        const std::string &stage_key,
        const StageShardingConfig &config)
    {
        std::string stage_type = extractStageType(stage_key);
        auto it = config.find(stage_type);
        if (it != config.end())
        {
            return it->second;
        }

        /*
         * Request-batched diagnostics and other cardinality-dependent
         * snapshots append an integer identity to a semantic family name.
         * Keep that variability declarative in the schema: a terminal '*'
         * means prefix match.  Longest-prefix selection is deterministic and
         * permits a model schema to refine a broader family if necessary.
         */
        size_t best_prefix_length = 0;
        SnapshotShardingMode family_mode = SnapshotShardingMode::UNKNOWN;
        for (const auto &[configured_key, configured_mode] : config)
        {
            if (configured_key.empty() || configured_key.back() != '*')
            {
                continue;
            }

            const std::string_view family_prefix(
                configured_key.data(),
                configured_key.size() - 1);
            if (stage_type.starts_with(family_prefix) &&
                family_prefix.size() > best_prefix_length)
            {
                best_prefix_length = family_prefix.size();
                family_mode = configured_mode;
            }
        }
        if (best_prefix_length != 0)
        {
            return family_mode;
        }

        return SnapshotShardingMode::UNKNOWN;
    }

    /**
     * @brief Get the expected sharding mode for a stage key (legacy hardcoded version)
     *
     * @deprecated Prefer the overload taking StageShardingConfig from ISchemaFactory.
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

        // Vocab-parallel embedding publishes row-parallel partial rows before
        // the explicit embedding allreduce and replicated rows after it.
        if (stage_type == "EMBEDDING")
            return SnapshotShardingMode::ROW_PARALLEL;
        if (stage_type == "EMBEDDING_ALLREDUCED")
            return SnapshotShardingMode::REPLICATED;

        // Attention projections - column-parallel (split on num_heads)
        if (stage_type == "Q_PROJECTION" || stage_type == "K_PROJECTION" ||
            stage_type == "V_PROJECTION" || stage_type == "QKV_PROJECTION" ||
            stage_type == "Q_NORM" || stage_type == "K_NORM" ||
            stage_type == "FA_GATE")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        // RoPE outputs - column-parallel (split on num_heads for Q, num_kv_heads for K)
        if (stage_type == "Q_ROPE" || stage_type == "K_ROPE" ||
            stage_type == "KV_APPEND_SOURCE_K" || stage_type == "KV_APPEND_SOURCE_V" ||
            stage_type == "KV_CACHE_K" || stage_type == "KV_CACHE_V" ||
            stage_type == "ATTENTION_EFFECTIVE_K" || stage_type == "ATTENTION_EFFECTIVE_V")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        // Attention context - column-parallel (split on num_heads)
        if (stage_type == "ATTENTION_CONTEXT" || stage_type == "ATTENTION_CONTEXT_GATED")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        // GDN projections and per-head state are column-parallel by local head/channel.
        if (stage_type == "GDN_PROJECTION" ||
            stage_type == "GDN_CONV1D" ||
            stage_type == "GDN_CONV1D_OUTPUT" ||
            stage_type == "GDN_RECURRENCE" ||
            stage_type == "GDN_DELTA_RULE_OUTPUT" ||
            stage_type == "GATED_RMSNORM" ||
            stage_type == "GDN_NORM_GATE_OUTPUT" ||
            stage_type == "GDN_Z_PROJECTION" ||
            stage_type == "GDN_ALPHA" ||
            stage_type == "GDN_BETA")
            return SnapshotShardingMode::COLUMN_PARALLEL;

        if (stage_type == "GDN_OUTPUT")
            return SnapshotShardingMode::ROW_PARALLEL;

        // Attention output (Wo) - row-parallel before AllReduce, replicated after it.
        if (stage_type == "ATTENTION_OUTPUT")
            return SnapshotShardingMode::ROW_PARALLEL;
        if (stage_type == "ATTENTION_OUTPUT_ALLREDUCED")
            return SnapshotShardingMode::REPLICATED;

        // Attention norms - replicated
        if (stage_type == "ATTENTION_NORM" ||
            stage_type == "ATTENTION_NORM_RESIDUAL_OUT" ||
            stage_type == "ATTENTION_RESIDUAL")
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
        if (stage_type == "FFN_DOWN_ALLREDUCED")
            return SnapshotShardingMode::REPLICATED;

        // FFN residual - replicated (after AllReduce)
        if (stage_type == "FFN_RESIDUAL")
            return SnapshotShardingMode::REPLICATED;

        // FFN norm - replicated
        if (stage_type == "FFN_NORM" || stage_type == "FFN_NORM_RESIDUAL_OUT")
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
         * For ROW_PARALLEL: Sums same-shaped per-device partials
         * For REPLICATED: Verifies every device published the same full output,
         * then uses the first full-device output as the combined view
         * For GATHERED: Uses already-gathered combined output
         * For UNKNOWN: Fails explicitly; callers must extend the schema/runtime
         *   sharding contract before comparing a multi-device semantic snapshot.
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
            else if (mode == SnapshotShardingMode::ROW_PARALLEL)
            {
                const auto &first = device_data[0];
                combined_rows = first.rows;
                combined_cols = first.cols;
                const size_t element_count = first.data.size();
                combined_data.assign(element_count, 0.0f);

                for (const auto &dev : device_data)
                {
                    if (dev.rows != combined_rows ||
                        dev.cols != combined_cols ||
                        dev.data.size() != element_count)
                    {
                        combined_valid = false;
                        combined_data.clear();
                        combined_rows = 0;
                        combined_cols = 0;
                        return false;
                    }

                    for (size_t i = 0; i < element_count; ++i)
                    {
                        combined_data[i] += dev.data[i];
                    }
                }

                combined_valid = true;
                return true;
            }
            else if (mode == SnapshotShardingMode::REPLICATED)
            {
                // Replicated stages already contain the full result on each device.
                // Treat disagreement as a snapshot contract failure: post-collective
                // diagnostics must not silently choose one participant and hide a
                // divergent allreduce output.
                const auto &first = device_data[0];
                combined_rows = first.rows;
                combined_cols = first.cols;
                const size_t element_count = first.data.size();
                constexpr float kReplicatedAbsTolerance = 1.0e-5f;
                constexpr float kReplicatedRelTolerance = 1.0e-6f;
                for (size_t device_index = 1; device_index < device_data.size(); ++device_index)
                {
                    const auto &dev = device_data[device_index];
                    if (dev.rows != combined_rows ||
                        dev.cols != combined_cols ||
                        dev.data.size() != element_count)
                    {
                        combined_valid = false;
                        combined_data.clear();
                        combined_rows = 0;
                        combined_cols = 0;
                        return false;
                    }
                    for (size_t i = 0; i < element_count; ++i)
                    {
                        const float a = first.data[i];
                        const float b = dev.data[i];
                        const float diff = std::fabs(a - b);
                        const float scale = std::max(std::fabs(a), std::fabs(b));
                        if (diff > kReplicatedAbsTolerance &&
                            diff > scale * kReplicatedRelTolerance)
                        {
                            combined_valid = false;
                            combined_data.clear();
                            combined_rows = 0;
                            combined_cols = 0;
                            return false;
                        }
                    }
                }
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

            combined_valid = false;
            combined_data.clear();
            combined_rows = 0;
            combined_cols = 0;
            return false;
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
