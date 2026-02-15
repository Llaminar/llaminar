/**
 * @file ROCmRingKVCacheFactory.cpp
 * @brief Factory implementation for ROCmRingKVCache (compiled with hipcc)
 * @author Llaminar Team
 * @date January 2026
 */

#include "ROCmRingKVCacheFactory.h"
#include "ROCmRingKVCache.h"
#include <stdexcept>

namespace llaminar2
{

    std::unique_ptr<IKVCache> createROCmRingKVCache(
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
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::BF16:
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::BF16>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::FP16:
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::FP16>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        case ActivationPrecision::Q8_1:
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::Q8_1>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, head_dim, device_id);

        default:
            throw std::runtime_error("createROCmRingKVCache: Unsupported precision. Use FP32, FP16, BF16, or Q8_1.");
        }
    }

    std::unique_ptr<IKVCache> createShardedROCmRingKVCache(
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
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::FP32>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);

        case ActivationPrecision::BF16:
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::BF16>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);

        case ActivationPrecision::FP16:
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::FP16>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);

        case ActivationPrecision::Q8_1:
            return std::make_unique<ROCmRingKVCache<ActivationPrecision::Q8_1>>(
                n_layers, batch_size, max_seq_len, n_kv_heads, local_n_kv_heads, kv_head_start, head_dim, device_id);

        default:
            throw std::runtime_error("createShardedROCmRingKVCache: Unsupported precision. Use FP32, FP16, BF16, or Q8_1.");
        }
    }

} // namespace llaminar2
