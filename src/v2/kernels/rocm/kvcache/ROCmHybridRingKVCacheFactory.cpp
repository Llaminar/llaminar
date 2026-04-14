/**
 * @file ROCmHybridRingKVCacheFactory.cpp
 * @brief Factory implementation for ROCmHybridRingKVCache (compiled with hipcc)
 */

#include "ROCmHybridRingKVCacheFactory.h"
#include "ROCmHybridRingKVCache.h"
#include <stdexcept>

namespace llaminar2
{

    std::unique_ptr<IKVCache> createROCmHybridRingKVCache(
        const HybridKVCacheConfig &hybrid_config,
        ActivationPrecision precision,
        int n_layers,
        int batch_size,
        int max_seq_len,
        int n_kv_heads,
        int head_dim,
        int device_id)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<ROCmHybridRingKVCacheFP32>(
                hybrid_config, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);
        case ActivationPrecision::FP16:
            return std::make_unique<ROCmHybridRingKVCacheFP16>(
                hybrid_config, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);
        case ActivationPrecision::BF16:
            return std::make_unique<ROCmHybridRingKVCacheBF16>(
                hybrid_config, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);
        case ActivationPrecision::Q8_1:
            return std::make_unique<ROCmHybridRingKVCacheQ8_1>(
                hybrid_config, n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);
        default:
            throw std::runtime_error("createROCmHybridRingKVCache: Unsupported precision. Use FP32, FP16, BF16, or Q8_1.");
        }
    }

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
        int device_id)
    {
        switch (precision)
        {
        case ActivationPrecision::FP32:
            return std::make_unique<ROCmHybridRingKVCacheFP32>(
                hybrid_config, n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);
        case ActivationPrecision::FP16:
            return std::make_unique<ROCmHybridRingKVCacheFP16>(
                hybrid_config, n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);
        case ActivationPrecision::BF16:
            return std::make_unique<ROCmHybridRingKVCacheBF16>(
                hybrid_config, n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);
        case ActivationPrecision::Q8_1:
            return std::make_unique<ROCmHybridRingKVCacheQ8_1>(
                hybrid_config, n_layers, batch_size, max_seq_len,
                n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);
        default:
            throw std::runtime_error("createShardedROCmHybridRingKVCache: Unsupported precision. Use FP32, FP16, BF16, or Q8_1.");
        }
    }

} // namespace llaminar2
