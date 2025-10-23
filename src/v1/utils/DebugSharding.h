/**
 * @file DebugSharding.h
 * @brief Centralized parsing of tensor sharding related debug environment flags.
 *
 * This consolidates scattered getenv() calls into a single lightweight snapshot
 * queried at first use. All flags are optional and default to disabled.
 *
 * Environment Variables:
 *  - LLAMINAR_DEBUG_MATERIALIZE_ATTENTION : If set non-empty, force an Allgather
 *      of the attention partial output (post W_O) for validation / inspection.
 *  - LLAMINAR_DUMP_SHARDS : If set, log first N (default 16) scalars of each
 *      sharded tensor as it is created / populated.
 *  - LLAMINAR_SHARD_PARITY_CHECK : If set, perform a reconstruction (sum/concat)
 *      after key kernels and diff against rank0 reference for small problem sizes.
 *  - LLAMINAR_ASSERT_REPLICATED_MISUSE : If set, embed a per-rank canary marker
 *      in partial outputs to help detect erroneous replicated assumptions.
 */
#pragma once

#include "DebugEnv.h" // Reuse unified snapshot but keep backwards compatibility for existing callers

namespace llaminar
{

    struct ShardingDebugConfig
    {
        bool debug_materialize_attention = false;
        bool dump_shards = false;
        bool shard_parity_check = false;
        bool assert_replicated_misuse = false;
        int dump_shards_n = 16; // future: allow override via LLAMINAR_DUMP_SHARDS_N
    };

    /**
     * @brief Return a const reference to the (lazily) initialized sharding debug config.
     * Thread-safe (idempotent) using function-local static initialization.
     */
    inline const ShardingDebugConfig &getShardingDebugConfig()
    {
        static ShardingDebugConfig shim = []()
        {
        ShardingDebugConfig s;
        const auto &d = debugEnv().sharding;
        s.debug_materialize_attention = d.debug_materialize_attention;
        s.dump_shards = d.dump_shards;
        s.dump_shards_n = d.dump_shards_n;
        s.shard_parity_check = d.shard_parity_check;
        s.assert_replicated_misuse = d.assert_replicated_misuse;
        return s; }();
        return shim;
    }

} // namespace llaminar
