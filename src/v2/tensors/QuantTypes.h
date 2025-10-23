#pragma once

/**
 * @file QuantTypes.h
 * @brief Quantization type enumeration for v2 architecture
 * @author David Sanftenberg
 */

namespace llaminar2 {

/**
 * @brief Quantization format types supported by Llaminar v2
 */
enum class QuantType {
    UNKNOWN = 0,
    
    // Floating point formats
    FP32,      ///< Standard 32-bit floating point
    FP16,      ///< IEEE 754 16-bit floating point
    BF16,      ///< Brain float 16-bit format
    
    // Integer quantization (uniform)
    Q8_0,      ///< 8-bit quantization (1 scale per 32 elements)
    Q6_K,      ///< 6-bit quantization with K-quants super-blocking
    Q4_0,      ///< 4-bit quantization (1 scale per 32 elements)
    
    // Non-linear quantization (lookup table)
    IQ4_NL,    ///< 4-bit non-linear quantization (4.5 bpw, 7.1× compression)
    
    // Future formats
    IQ3_XXS,   ///< 3-bit extra-extra-small quantization
    IQ2_XXS,   ///< 2-bit extra-extra-small quantization
};

/**
 * @brief Get human-readable name for quantization type
 */
inline const char* quant_type_name(QuantType type) {
    switch (type) {
        case QuantType::FP32:    return "FP32";
        case QuantType::FP16:    return "FP16";
        case QuantType::BF16:    return "BF16";
        case QuantType::Q8_0:    return "Q8_0";
        case QuantType::Q6_K:    return "Q6_K";
        case QuantType::Q4_0:    return "Q4_0";
        case QuantType::IQ4_NL:  return "IQ4_NL";
        case QuantType::IQ3_XXS: return "IQ3_XXS";
        case QuantType::IQ2_XXS: return "IQ2_XXS";
        default:                 return "UNKNOWN";
    }
}

/**
 * @brief Get bits per weight for quantization type
 */
inline float quant_type_bpw(QuantType type) {
    switch (type) {
        case QuantType::FP32:    return 32.0f;
        case QuantType::FP16:    return 16.0f;
        case QuantType::BF16:    return 16.0f;
        case QuantType::Q8_0:    return 8.5f;   // 8 bits + scale overhead
        case QuantType::Q6_K:    return 6.5f;   // 6 bits + scale overhead
        case QuantType::Q4_0:    return 4.5f;   // 4 bits + scale overhead
        case QuantType::IQ4_NL:  return 4.5f;   // 4 bits + scale overhead
        case QuantType::IQ3_XXS: return 3.3f;   // ~3 bits + scale overhead
        case QuantType::IQ2_XXS: return 2.3f;   // ~2 bits + scale overhead
        default:                 return 0.0f;
    }
}

} // namespace llaminar2
