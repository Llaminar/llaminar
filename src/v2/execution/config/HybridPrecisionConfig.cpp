/**
 * @file HybridPrecisionConfig.cpp
 * @brief Implementation of hybrid precision configuration
 */

#include "HybridPrecisionConfig.h"
#include "../../utils/Logger.h"

namespace llaminar2
{

    ActivationPrecision HybridPrecisionConfig::getPrecision(HybridBufferType buffer_type) const
    {
        switch (buffer_type)
        {
        // Core buffers - always FP32
        case HybridBufferType::Residual:
        case HybridBufferType::ResidualStream:
        case HybridBufferType::Normalized:
        case HybridBufferType::Hidden:
        case HybridBufferType::Logits:
            return ActivationPrecision::FP32;

        // QKV path - legacy uniform output
        case HybridBufferType::QKV_GEMM_Output:
            return qkv_gemm_output;
        // Individual Q/K/V GEMM outputs - default to qkv_gemm_output for Hybrid mode
        case HybridBufferType::Q_GEMM_Output:
        case HybridBufferType::K_GEMM_Output:
        case HybridBufferType::V_GEMM_Output:
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
        // For HybridQ16 mode, use the HybridQ16PrecisionConfig
        if (global_precision == ActivationPrecision::HybridQ16)
        {
            static const HybridQ16PrecisionConfig q16_config = HybridQ16PrecisionConfig::defaultConfig();
            return q16_config.getPrecision(buffer_type);
        }

        // For non-Hybrid modes, return global precision for most buffers
        // (except core buffers which are always FP32)
        if (global_precision != ActivationPrecision::Hybrid)
        {
            // Core buffers are always FP32 regardless of global setting
            // Note: FFN_Gate, FFN_Up follow global precision (not forced FP32)
            // to maintain existing behavior. Attention_Output and FFN_Down
            // are forced FP32 because they feed directly into the residual stream.
            switch (buffer_type)
            {
            case HybridBufferType::Residual:
            case HybridBufferType::Normalized:
            case HybridBufferType::Hidden:
            case HybridBufferType::Logits:
            case HybridBufferType::Attention_Output:
            case HybridBufferType::FFN_Down:
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
        case HybridBufferType::Q_GEMM_Output:
            return "Q_GEMM_Output";
        case HybridBufferType::K_GEMM_Output:
            return "K_GEMM_Output";
        case HybridBufferType::V_GEMM_Output:
            return "V_GEMM_Output";
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
        case HybridBufferType::ResidualStream:
            return "ResidualStream";
        default:
            return "Unknown";
        }
    }

    // =========================================================================
    // HybridQ16PrecisionConfig Implementation
    // =========================================================================

    ActivationPrecision HybridQ16PrecisionConfig::getPrecision(HybridBufferType buffer_type) const
    {
        switch (buffer_type)
        {
        // Residual stream - Q16_1 (the key change from Hybrid mode)
        case HybridBufferType::Residual:
        case HybridBufferType::ResidualStream:
            return residual_stream;

        // Normalized buffer - FP32 (output of RMSNorm for GEMM input)
        case HybridBufferType::Normalized:
        case HybridBufferType::Hidden:
        case HybridBufferType::Logits:
            return ActivationPrecision::FP32;

        // QKV path - mixed precision for HybridQ16
        // K is Q16_1 to preserve small values, Q and V are Q8_1
        case HybridBufferType::QKV_GEMM_Output:
            return qkv_gemm_output; // Legacy uniform (deprecated)
        case HybridBufferType::Q_GEMM_Output:
            return q_gemm_output; // Q8_1
        case HybridBufferType::K_GEMM_Output:
            return k_gemm_output; // Q16_1 (256× better precision!)
        case HybridBufferType::V_GEMM_Output:
            return v_gemm_output; // Q8_1
        case HybridBufferType::Q_After_RoPE:
            return q_after_rope;
        case HybridBufferType::K_After_RoPE:
            return k_after_rope;
        case HybridBufferType::KV_Cache:
            return kv_cache;

        // Attention - Q16 fused kernel uses INT32 internally, outputs Q16_1
        case HybridBufferType::Attention_Context:
            return attention_context; // Q16_1 for snapshots (fused kernel is INT32 internal)
        case HybridBufferType::Attention_Output:
            return attention_output; // Q16_1 in HybridQ16 (fused write to residual)

        // FFN - gate/up are Q8_1, down is Q8_1 (added to Q16_1 residual)
        case HybridBufferType::FFN_Gate:
            return ffn_gate;
        case HybridBufferType::FFN_Up:
            return ffn_up;
        case HybridBufferType::FFN_Down:
            return ffn_down; // Q8_1 in HybridQ16

        default:
            LOG_WARN("Unknown HybridBufferType in HybridQ16: " << static_cast<int>(buffer_type) << ", defaulting to FP32");
            return ActivationPrecision::FP32;
        }
    }

    HybridQ16PrecisionConfig HybridQ16PrecisionConfig::defaultConfig()
    {
        // Returns default HybridQ16 configuration for Q16 integer attention:
        // - Residual stream: Q16_1 (high-precision accumulator)
        // - Q/K after RoPE: Q16_1 (Q16 fused kernel expects Q16_1 inputs)
        // - KV Cache: Q16_1 (matches Q16 attention)
        // - Attention context: Q16_1 (for snapshots; fused kernel uses INT32 internally)
        // - Attention output: Q16_1 (fused kernel writes Q16_1 directly to residual)
        // - FFN down: Q8_1 (added to Q16_1 residual)
        return HybridQ16PrecisionConfig{};
    }

} // namespace llaminar2
