#include "planning/ActivationMemoryEstimator.h"

#include <algorithm>

namespace llaminar2
{

size_t ActivationMemoryEstimator::estimate(
    int batch_size,
    int max_seq_len,
    int d_model,
    int d_ff,
    int n_heads,
    int n_kv_heads,
    int head_dim,
    int vocab_size,
    DeviceId device)
{
    if (batch_size <= 0 || max_seq_len <= 0 || d_model <= 0)
    {
        return 0;
    }

    size_t B = static_cast<size_t>(batch_size);
    size_t S = static_cast<size_t>(max_seq_len);
    size_t D = static_cast<size_t>(d_model);
    size_t F = static_cast<size_t>(d_ff);
    size_t H = static_cast<size_t>(n_heads);
    size_t HK = static_cast<size_t>(n_kv_heads);
    size_t HD = static_cast<size_t>(head_dim);
    size_t V = static_cast<size_t>(vocab_size);

    // All sizes in bytes (FP32 = 4 bytes per element unless noted)
    constexpr size_t FP32 = 4;

    // Hidden state buffer: B × S × D
    size_t hidden_state = B * S * D * FP32;

    // Residual buffer: B × S × D (for skip connections)
    size_t residual = B * S * D * FP32;

    // QKV projection outputs:
    // Q: B × S × H × HD, K: B × S × HK × HD, V: B × S × HK × HD
    size_t q_proj = B * S * H * HD * FP32;
    size_t k_proj = B * S * HK * HD * FP32;
    size_t v_proj = B * S * HK * HD * FP32;

    // Attention output: B × S × D
    size_t attn_output = B * S * D * FP32;

    // FFN intermediates: gate and up are B × S × F each
    size_t ffn_gate = B * S * F * FP32;
    size_t ffn_up = B * S * F * FP32;

    // FFN down output (reuses hidden state buffer in practice, but count it)
    size_t ffn_down = B * S * D * FP32;

    // Logits: B × S × V (only during prefill for full sequence, 1 token during decode)
    size_t logits = B * S * V * FP32;

    // Norm scratch: B × S × D (for RMS norm intermediates)
    size_t norm_scratch = B * S * D * FP32;

    // Total: many of these alias in the BufferArena, but we estimate worst-case
    // concurrent usage. In practice, aliasing reduces this by ~40-50%.
    // The BufferArena allocates based on alias groups, so concurrent buffers
    // that don't overlap in the compute graph share memory.
    //
    // Conservative estimate: count all non-aliasable concurrent buffers.
    // During attention: hidden_state + residual + Q + K + V + attn_output + norm
    // During FFN: hidden_state + residual + gate + up + ffn_down + norm
    // During LM head: hidden_state + logits
    //
    // Peak is max of these phases:
    size_t attn_phase = hidden_state + residual + q_proj + k_proj + v_proj + attn_output + norm_scratch;
    size_t ffn_phase = hidden_state + residual + ffn_gate + ffn_up + ffn_down + norm_scratch;
    size_t lm_head_phase = hidden_state + logits;

    size_t peak = std::max({attn_phase, ffn_phase, lm_head_phase});

    return peak;
}

} // namespace llaminar2
