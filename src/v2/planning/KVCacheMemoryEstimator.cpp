#include "planning/KVCacheMemoryEstimator.h"

namespace llaminar2
{

float KVCacheMemoryEstimator::getBytesPerElement(const std::string& kv_precision)
{
    if (kv_precision == "fp32") return 4.0f;
    if (kv_precision == "fp16" || kv_precision == "bf16") return 2.0f;
    if (kv_precision == "q8_1")
    {
        // Q8_1 block: 36 bytes per 32 elements = 1.125 bytes/element
        return 36.0f / 32.0f;
    }
    if (kv_precision == "tq")
    {
        // TurboQuant ring KV: 16-bit entries + quantization params + scratch
        // Approximate: ~2.5 bytes/element including scratch overhead
        return 2.5f;
    }
    // Default: FP16 (most common GPU KV precision)
    return 2.0f;
}

size_t KVCacheMemoryEstimator::estimate(
    int n_layers,
    int batch_size,
    int max_seq_len,
    int n_kv_heads,
    int head_dim,
    const std::string& kv_precision,
    DeviceId device)
{
    if (n_layers <= 0 || batch_size <= 0 || max_seq_len <= 0 ||
        n_kv_heads <= 0 || head_dim <= 0)
    {
        return 0;
    }

    float bytes_per_element = getBytesPerElement(kv_precision);

    // K + V caches, each: n_layers × batch × max_seq × n_kv_heads × head_dim
    size_t elements_per_cache = static_cast<size_t>(n_layers) *
                                static_cast<size_t>(batch_size) *
                                static_cast<size_t>(max_seq_len) *
                                static_cast<size_t>(n_kv_heads) *
                                static_cast<size_t>(head_dim);

    // Two caches: K and V
    size_t total_elements = elements_per_cache * 2;

    size_t base_bytes = static_cast<size_t>(
        static_cast<float>(total_elements) * bytes_per_element);

    // TQ on GPU has additional overhead for scratch buffers and dequant params
    if (kv_precision == "tq" && device.is_gpu())
    {
        // Per-layer scratch: 2 × max_seq × n_kv_heads × head_dim × sizeof(__half)
        size_t scratch_per_layer = 2ULL *
            static_cast<size_t>(max_seq_len) *
            static_cast<size_t>(n_kv_heads) *
            static_cast<size_t>(head_dim) * 2;  // FP16 = 2 bytes
        base_bytes += scratch_per_layer * static_cast<size_t>(n_layers);
    }

    return base_bytes;
}

} // namespace llaminar2
