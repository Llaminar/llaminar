/**
 * @file ROCmHybridRingKVCacheFactory.h
 * @brief Factory functions for creating ROCmHybridRingKVCache without HIP headers
 *
 * This header provides factory functions that can be included in non-HIP code.
 * The actual implementation is in ROCmHybridRingKVCacheFactory.cpp which is compiled
 * with hipcc.
 */

#pragma once

#include <memory>
#include "../../IKVCache.h"
#include "../../../execution/config/RuntimeConfig.h"

namespace llaminar2
{
    struct HybridKVCacheConfig;

    /**
     * @brief Create a ROCmHybridRingKVCache with the specified precision
     */
    std::unique_ptr<IKVCache> createROCmHybridRingKVCache(
        const HybridKVCacheConfig &hybrid_config,
        ActivationPrecision precision,
        int n_layers,
        int batch_size,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        int device_id);

    /**
     * @brief Create a sharded ROCmHybridRingKVCache for tensor parallelism
     */
    std::unique_ptr<IKVCache> createShardedROCmHybridRingKVCache(
        const HybridKVCacheConfig &hybrid_config,
        ActivationPrecision precision,
        int n_layers,
        int batch_size,
        int max_seq_len,
        int n_kv_heads,
        int local_n_kv_heads,
        int kv_head_start,
        int head_dim,
        int device_id);

} // namespace llaminar2
