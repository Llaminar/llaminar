/**
 * @file StageShardingMode.h
 * @brief Shared enumeration for stage output sharding across TP devices
 *
 * Extracted from TPSnapshot.h so that both GraphSchema.h (ISchemaFactory)
 * and TPSnapshot.h can reference the same enum without circular includes.
 */

#pragma once

#include <string>
#include <unordered_map>

namespace llaminar2
{

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

    /**
     * @brief Map of stage type string to sharding mode
     *
     * Stage type strings are the canonical suffixes (e.g., "Q_PROJECTION",
     * "FFN_DOWN", "LM_HEAD") without layer prefixes.
     */
    using StageShardingConfig = std::unordered_map<std::string, SnapshotShardingMode>;

} // namespace llaminar2
