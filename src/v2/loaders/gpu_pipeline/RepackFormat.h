#pragma once

/**
 * @file RepackFormat.h
 * @brief Shared repack format enum and kernel function pointer types
 *
 * Backend-agnostic definitions used by DeviceLoadPipeline, WeightVRAMPool,
 * and both CUDA/ROCm repack kernel implementations.
 */

#include <cstdint>
#include <optional>

namespace llaminar2 {

/// Format identifiers for GPU repack kernel dispatch.
/// Values are sequential; they do NOT correspond to codebook_ids
/// (some codebook_ids are shared between simple and superblock variants).
enum class RepackFormat : uint8_t {
    Q4_0    = 0,   ///< Symmetric 4-bit (32-element blocks, 18 bytes)
    IQ4_NL  = 1,   ///< Non-linear 4-bit (identical block layout to Q4_0)
    Q4_1    = 2,   ///< Asymmetric 4-bit (32-element blocks, 20 bytes)
    Q5_0    = 3,   ///< Symmetric 5-bit (32-element blocks, 22 bytes)
    Q5_1    = 4,   ///< Asymmetric 5-bit (32-element blocks, 24 bytes)
    Q4_K    = 5,   ///< K-quant 4-bit (256-element super-blocks, 144 bytes)
    Q5_K    = 6,   ///< K-quant 5-bit (256-element super-blocks, 176 bytes)
    Q6_K    = 7,   ///< K-quant 6-bit (256-element super-blocks, 210 bytes)
    Q3_K    = 8,   ///< K-quant 3-bit (256-element super-blocks, 110 bytes)
    Q2_K    = 9,   ///< K-quant 2-bit (256-element super-blocks, 84 bytes)
    IQ4_XS  = 10,  ///< IQ 4-bit (256-element super-blocks, 136 bytes)
    IQ3_S   = 11,  ///< IQ 3-bit (256-element super-blocks, 110 bytes)
    IQ3_XXS = 12,  ///< IQ 3-bit compact (256-element super-blocks, 98 bytes)
    IQ2_S   = 13,  ///< IQ 2-bit (256-element super-blocks, 82 bytes)
    IQ2_XS  = 14,  ///< IQ 2-bit compact (256-element super-blocks, 74 bytes)
    IQ2_XXS = 15,  ///< IQ 2-bit ultra-compact (256-element super-blocks, 66 bytes)
    IQ1_S   = 16,  ///< IQ 1-bit (256-element super-blocks, 50 bytes)
    IQ1_M   = 17,  ///< IQ 1-bit (256-element super-blocks, 56 bytes)
    Q8_0    = 18,  ///< Symmetric 8-bit (32-element blocks, 34 bytes)
    RAW_FP  = 255, ///< Floating-point passthrough (no repack, direct H2D copy)
};

/// Function pointer types for backend-agnostic repack kernel dispatch.
/// DeviceLoadPipeline stores these and calls through them without knowing
/// whether the underlying implementation is CUDA or ROCm.
struct RepackKernels {
    /// VNNI repack: raw GGUF blocks → interleaved payload + scales + mins + emins
    using VnniRepackFn = bool(*)(RepackFormat format,
                                  const void* d_raw_blocks,
                                  uint8_t* d_payload,
                                  uint16_t* d_scales,
                                  uint16_t* d_mins,
                                  uint32_t* d_emins,
                                  int N, int K,
                                  void* stream);

    VnniRepackFn vnniRepack = nullptr;

    bool isValid() const { return vnniRepack != nullptr; }
};

/// Map NativeVnniFormatInfo codebook_id + is_superblock to RepackFormat.
/// Returns std::nullopt for unknown or unsupported combinations.
inline std::optional<RepackFormat> codebookIdToRepackFormat(uint8_t codebook_id, bool is_superblock)
{
    if (!is_superblock) {
        switch (codebook_id) {
        case 0:  return RepackFormat::Q4_0;
        case 4:  return RepackFormat::IQ4_NL;
        case 5:  return RepackFormat::Q4_1;
        case 6:  return RepackFormat::Q5_0;
        case 7:  return RepackFormat::Q5_1;
        case 19: return RepackFormat::Q8_0;
        default: return std::nullopt;
        }
    } else {
        switch (codebook_id) {
        case 4:  return RepackFormat::IQ4_XS;
        case 5:  return RepackFormat::Q4_K;
        case 7:  return RepackFormat::Q5_K;
        case 8:  return RepackFormat::Q6_K;
        case 9:  return RepackFormat::Q3_K;
        case 10: return RepackFormat::Q2_K;
        case 11: return RepackFormat::IQ3_S;
        case 12: return RepackFormat::IQ3_XXS;
        case 13: return RepackFormat::IQ2_S;
        case 14: return RepackFormat::IQ2_XS;
        case 15: return RepackFormat::IQ2_XXS;
        case 16: return RepackFormat::IQ1_S;
        case 17: return RepackFormat::IQ1_M;
        default: return std::nullopt;
        }
    }
}

} // namespace llaminar2
