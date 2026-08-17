/**
 * @file ROCmRingKVCacheTQFactory.h
 * @brief Factory function for ROCmRingKVCacheTQ without HIP headers
 * @author David Sanftenberg
 *
 * This header provides a factory function that can be included in non-HIP code
 * (e.g., KernelFactory.cpp which also includes CUDA headers).
 * The actual implementation is compiled separately with hipcc in the .hip file.
 */

#pragma once

#include <memory>
#include "../../IKVCache.h"
#include "../../kvcache/TurboQuantKVMode.h"

namespace llaminar2
{
    class TurboQuantContext;

    /**
     * @brief Create a ROCmRingKVCacheTQ (TQ8-K / TQ4-V asymmetric cache)
     *
     * @param n_layers     Number of transformer layers
     * @param batch_size   Number of sequences
     * @param max_seq_len  Ring buffer capacity
     * @param n_kv_heads   Number of KV heads
     * @param head_dim     Head dimension (64 or 128)
     * @param tq_ctx       TurboQuant context (null → use default seed)
     * @param device_id    ROCm device ordinal
     * @return Unique pointer to IKVCache
     */
    std::unique_ptr<IKVCache> createROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int head_dim,
        const TurboQuantContext *tq_ctx,
        int device_id,
        TurboQuantKVMode mode = TurboQuantKVMode::AQ8_K_TQ4_V);

    /**
     * @brief Create a LocalTP shard of the ROCm asymmetric TQ cache.
     *
     * @param n_kv_heads Global KV-head count.
     * @param local_n_kv_heads Heads resident on this device.
     * @param kv_head_start First global KV head represented by the shard.
     */
    std::unique_ptr<IKVCache> createShardedROCmRingKVCacheTQ(
        int n_layers, int batch_size, int max_seq_len,
        int n_kv_heads, int local_n_kv_heads, int kv_head_start,
        int head_dim, const TurboQuantContext *tq_ctx,
        int device_id,
        TurboQuantKVMode mode = TurboQuantKVMode::AQ8_K_TQ4_V);

} // namespace llaminar2
