/**
 * @file ROCmRingKVCacheFactory.h
 * @brief Factory functions for creating ROCmRingKVCache without HIP headers
 * @author Llaminar Team
 * @date January 2026
 *
 * This header provides factory functions that can be included in non-HIP code.
 * The actual implementation is in ROCmRingKVCacheFactory.cpp which is compiled
 * with hipcc.
 *
 * Returns IKVCache* to avoid unique_ptr issues with incomplete types.
 */

#pragma once

#include <memory>
#include "../../IKVCache.h"                          // For IKVCache base class
#include "../../../execution/config/RuntimeConfig.h" // For ActivationPrecision

namespace llaminar2
{

    /**
     * @brief Create a ROCmRingKVCache with the specified precision
     *
     * This function allows creating ROCm KV caches from non-HIP code.
     * The implementation is compiled separately with hipcc.
     *
     * @param precision FP32, FP16, or BF16
     * @param n_layers Number of transformer layers
     * @param batch_size Number of sequences
     * @param max_seq_len Maximum sequence length
     * @param n_kv_heads Number of KV heads
     * @param head_dim Dimension per head
     * @param device_id ROCm device ID
     * @return Unique pointer to IKVCache (underlying is ROCmRingKVCache)
     * @throws std::runtime_error if precision is not supported
     */
    std::unique_ptr<IKVCache> createROCmRingKVCache(
        ActivationPrecision precision,
        int n_layers,
        int batch_size,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        int device_id);

    /**
     * @brief Create a sharded ROCmRingKVCache for tensor parallelism
     *
     * @param precision FP32, FP16, or BF16
     * @param n_layers Number of transformer layers
     * @param batch_size Number of sequences
     * @param max_seq_len Maximum sequence length
     * @param n_kv_heads Total number of KV heads (across all ranks)
     * @param local_n_kv_heads Number of KV heads on this rank
     * @param kv_head_start Starting KV head index for this rank
     * @param head_dim Dimension per head
     * @param device_id ROCm device ID
     * @return Unique pointer to IKVCache (underlying is ROCmRingKVCache)
     * @throws std::runtime_error if precision is not supported
     */
    std::unique_ptr<IKVCache> createShardedROCmRingKVCache(
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
