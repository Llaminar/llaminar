#pragma once
/**
 * @file TensorType.h
 * @brief Tensor data type enum - CUDA-safe, no SIMD dependencies
 *
 * This file is safe to include from CUDA code (.cu files) as it contains
 * no platform-specific code (no SIMD intrinsics, no CUDA kernels).
 */

namespace llaminar2
{

    /**
     * @brief Tensor data type
     */
    enum class TensorType
    {
        FP32,    // 32-bit float
        BF16,    // 16-bit bfloat
        FP16,    // 16-bit float
        INT8,    // 8-bit integer
        INT32,   // 32-bit integer accumulator
        IQ4_NL,  // 4-bit quantized (non-linear)
        IQ4_XS,  // 4-bit quantized (extra-small IQ)
        Q8_0,    // 8-bit quantized (weights)
        Q8_1,    // 8-bit quantized with pre-computed sum
        Q16_1,   // 16-bit quantized with pre-computed sum
        Q4_0,    // 4-bit quantized
        Q4_1,    // 4-bit quantized with min
        Q5_0,    // 5-bit quantized
        Q5_1,    // 5-bit quantized with min
        Q6_K,    // 6-bit K-quant
        Q2_K,    // 2-bit K-quant
        Q5_K,    // 5-bit K-quant
        Q3_K,    // 3-bit K-quant
        Q4_K,    // 4-bit K-quant
        Q8_K,    // 8-bit K-quant
        IQ2_XXS, // 2-bit extra-extra-small IQ
        IQ2_XS,  // 2-bit extra-small IQ
        IQ3_XXS, // 3-bit extra-extra-small IQ
        IQ2_S,   // 2-bit small IQ
        IQ3_S,   // 3-bit small IQ
        IQ1_S,   // 1-bit small IQ
        IQ1_M,   // 1-bit medium IQ
        TQ4,     // TurboQuant 4-bit (KV cache)
        TQ8      // TurboQuant 8-bit (KV cache K-projections)
    };

    /**
     * @brief Get human-readable name for TensorType
     * @param type The tensor type enum value
     * @return Static string like "FP32", "Q8_0", "IQ4_NL", etc.
     */
    inline const char *tensorTypeName(TensorType type)
    {
        switch (type)
        {
        case TensorType::FP32:
            return "FP32";
        case TensorType::BF16:
            return "BF16";
        case TensorType::FP16:
            return "FP16";
        case TensorType::INT8:
            return "INT8";
        case TensorType::INT32:
            return "INT32";
        case TensorType::IQ4_NL:
            return "IQ4_NL";
        case TensorType::IQ4_XS:
            return "IQ4_XS";
        case TensorType::Q8_0:
            return "Q8_0";
        case TensorType::Q8_1:
            return "Q8_1";
        case TensorType::Q16_1:
            return "Q16_1";
        case TensorType::Q4_0:
            return "Q4_0";
        case TensorType::Q4_1:
            return "Q4_1";
        case TensorType::Q5_0:
            return "Q5_0";
        case TensorType::Q5_1:
            return "Q5_1";
        case TensorType::Q6_K:
            return "Q6_K";
        case TensorType::Q2_K:
            return "Q2_K";
        case TensorType::Q5_K:
            return "Q5_K";
        case TensorType::Q3_K:
            return "Q3_K";
        case TensorType::Q4_K:
            return "Q4_K";
        case TensorType::Q8_K:
            return "Q8_K";
        case TensorType::IQ2_XXS:
            return "IQ2_XXS";
        case TensorType::IQ2_XS:
            return "IQ2_XS";
        case TensorType::IQ3_XXS:
            return "IQ3_XXS";
        case TensorType::IQ2_S:
            return "IQ2_S";
        case TensorType::IQ3_S:
            return "IQ3_S";
        case TensorType::IQ1_S:
            return "IQ1_S";
        case TensorType::IQ1_M:
            return "IQ1_M";
        case TensorType::TQ4:
            return "TQ4";
        case TensorType::TQ8:
            return "TQ8";
        default:
            return "UNKNOWN";
        }
    }

    /**
     * @brief Check if tensor type is a native-VNNI format (≤6-bit quantization)
     *
     * These formats use lossless native-VNNI decode with per-block FP16 scales.
     * Other quantized types (Q8_1, Q8_K) use INT8-VNNI instead.
     *
     * @param type The tensor type enum value
     * @return true if the format supports native-VNNI kernels
     */
    inline bool isNativeVnniFormat(TensorType type)
    {
        switch (type)
        {
        case TensorType::Q4_0:
        case TensorType::Q4_1:
        case TensorType::Q5_0:
        case TensorType::Q5_1:
        case TensorType::Q6_K:
        case TensorType::Q2_K:
        case TensorType::Q3_K:
        case TensorType::Q4_K:
        case TensorType::Q5_K:
        case TensorType::IQ4_NL:
        case TensorType::IQ4_XS:
        case TensorType::IQ3_S:
        case TensorType::IQ3_XXS:
        case TensorType::IQ2_S:
        case TensorType::IQ2_XS:
        case TensorType::IQ2_XXS:
        case TensorType::IQ1_S:
        case TensorType::IQ1_M:
            return true;
        default:
            return false;
        }
    }

    /**
     * @brief Check if tensor type is an INT8-VNNI format (legacy 8-bit path)
     *
     * Q8_0 and Q8_1 use INT8-VNNI (direct INT8 requantization).
     * Q8_K is legacy.
     *
     * @param type The tensor type enum value
     * @return true if the format uses the INT8-VNNI path
     */
    inline bool isInt8VnniFormat(TensorType type)
    {
        switch (type)
        {
        case TensorType::Q8_0:
        case TensorType::Q8_1:
        case TensorType::Q8_K:
            return true;
        default:
            return false;
        }
    }

} // namespace llaminar2
