/**
 * @file HybridPrecisionConfig.cpp
 * @brief Implementation of hybrid precision configuration
 */

#include "HybridPrecisionConfig.h"
#include "../utils/Logger.h"

namespace llaminar2
{

    ActivationPrecision HybridPrecisionConfig::getPrecision(HybridBufferType buffer_type) const
    {
        switch (buffer_type)
        {
        // Core buffers - always FP32
        case HybridBufferType::Residual:
        case HybridBufferType::Normalized:
        case HybridBufferType::Hidden:
        case HybridBufferType::Logits:
            return ActivationPrecision::FP32;

        // QKV path
        case HybridBufferType::QKV_GEMM_Output:
            return qkv_gemm_output;
        case HybridBufferType::Q_After_RoPE:
            return q_after_rope;
        case HybridBufferType::K_After_RoPE:
            return k_after_rope;
        case HybridBufferType::KV_Cache:
            return kv_cache;

        // Attention
        case HybridBufferType::Attention_Context:
            return attention_context;
        case HybridBufferType::Attention_Output:
            return attention_output;

        // FFN
        case HybridBufferType::FFN_Gate:
            return ffn_gate;
        case HybridBufferType::FFN_Up:
            return ffn_up;
        case HybridBufferType::FFN_Down:
            return ffn_down;

        default:
            LOG_WARN("Unknown HybridBufferType: " << static_cast<int>(buffer_type) << ", defaulting to FP32");
            return ActivationPrecision::FP32;
        }
    }

    HybridPrecisionConfig HybridPrecisionConfig::defaultConfig()
    {
        // Returns default configuration which eliminates unnecessary requantization:
        // - QKV GEMM outputs Q8_1 (acceptable)
        // - RoPE outputs FP32 (avoid requant)
        // - KV cache uses BF16 (better than Q8_1)
        // - Attention context stays FP32 (no context quantization for Wo)
        // - FFN intermediates use Q8_1, but output to residual is FP32
        return HybridPrecisionConfig{};
    }

    bool HybridPrecisionConfig::requiresSeparateBuffer(HybridBufferType buffer_type)
    {
        // In Hybrid mode, these buffers need separate FP32 allocation
        // because their preceding operation outputs a different precision
        switch (buffer_type)
        {
        case HybridBufferType::Q_After_RoPE:
        case HybridBufferType::K_After_RoPE:
            // QKV GEMM outputs Q8_1, but RoPE needs to output FP32
            // So we need separate FP32 buffers for post-RoPE Q/K
            return true;

        default:
            return false;
        }
    }

    ActivationPrecision resolveBufferPrecision(
        ActivationPrecision global_precision,
        HybridBufferType buffer_type,
        const HybridPrecisionConfig *hybrid_config)
    {
        // For non-Hybrid modes, return global precision for most buffers
        // (except core buffers which are always FP32)
        if (global_precision != ActivationPrecision::Hybrid)
        {
            // Core buffers are always FP32 regardless of global setting
            // FFN_Gate and FFN_Up are kept FP32 to avoid triple quantization in SwiGLU:
            //   - Without: gate→Q8_1, up→Q8_1, swiglu_result→Q8_1 (3 rounds)
            //   - With FP32: silu(gate)*up computed in FP32, only output quantized (1 round)
            // This significantly reduces error accumulation in the FFN path.
            switch (buffer_type)
            {
            case HybridBufferType::Residual:
            case HybridBufferType::Normalized:
            case HybridBufferType::Hidden:
            case HybridBufferType::Logits:
            case HybridBufferType::Attention_Output:
            case HybridBufferType::FFN_Down:
            case HybridBufferType::FFN_Gate:
            case HybridBufferType::FFN_Up:
                return ActivationPrecision::FP32;

            default:
                return global_precision;
            }
        }

        // For Hybrid mode, use per-buffer precision
        if (hybrid_config)
        {
            return hybrid_config->getPrecision(buffer_type);
        }

        // Default hybrid config if none provided
        static const HybridPrecisionConfig default_config = HybridPrecisionConfig::defaultConfig();
        return default_config.getPrecision(buffer_type);
    }

    const char *hybridBufferTypeToString(HybridBufferType buffer_type)
    {
        switch (buffer_type)
        {
        case HybridBufferType::Residual:
            return "Residual";
        case HybridBufferType::Normalized:
            return "Normalized";
        case HybridBufferType::Hidden:
            return "Hidden";
        case HybridBufferType::Logits:
            return "Logits";
        case HybridBufferType::QKV_GEMM_Output:
            return "QKV_GEMM_Output";
        case HybridBufferType::Q_After_RoPE:
            return "Q_After_RoPE";
        case HybridBufferType::K_After_RoPE:
            return "K_After_RoPE";
        case HybridBufferType::KV_Cache:
            return "KV_Cache";
        case HybridBufferType::Attention_Context:
            return "Attention_Context";
        case HybridBufferType::Attention_Output:
            return "Attention_Output";
        case HybridBufferType::FFN_Gate:
            return "FFN_Gate";
        case HybridBufferType::FFN_Up:
            return "FFN_Up";
        case HybridBufferType::FFN_Down:
            return "FFN_Down";
        default:
            return "Unknown";
        }
    }

} // namespace llaminar2
