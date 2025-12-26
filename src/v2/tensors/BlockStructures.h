/**
 * @file BlockStructures.h
 * @brief Quantization block structure definitions
 * @author David Sanftenberg
 *
 * Defines packed block structures for all supported quantization formats.
 * These are POD types matching GGML/llama.cpp binary layout for compatibility.
 *
 * This header is intentionally standalone (no dependencies) to avoid circular
 * includes. It's included early in the compilation order by both SIMDHelpers.h
 * and Tensors.h.
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace llaminar2
{
    // ========================================================================
    // Simple Quantization Formats (32-element blocks)
    // ========================================================================

    /**
     * @brief IQ4_NL block structure (exactly 18 bytes) representing 32 quantized elements.
     *
     * Layout mirrors GGML's block_iq4_nl. Two 4-bit indices per byte in @p qs select entries
     * in kvalues_iq4nl, scaled by FP16 value @p d.
     */
    struct IQ4_NLBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< Packed 4-bit indices (2 per byte)

        static constexpr size_t BLOCK_SIZE = 32; ///< Elements per block
    };
    static_assert(sizeof(IQ4_NLBlock) == 18, "IQ4_NLBlock must be 18 bytes");

    /** @brief Q8_0 block: 8-bit quantization (32 elements per block, 34 bytes) */
    struct Q8_0Block
    {
        uint16_t d;    ///< FP16 scale factor
        int8_t qs[32]; ///< 32 quantized int8 values
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q8_0Block) == 34, "Q8_0Block must be 34 bytes");

    /**
     * @brief Q8_1 block: 8-bit quantization with pre-computed sum (36 bytes)
     *
     * Like Q8_0 but stores the pre-computed sum for asymmetric quantization compensation.
     * This format is used as an intermediate format for activations (matching CUDA pattern):
     * - FP32/FP16/BF16 activations → Q8_1 (quantize once per panel)
     * - Q8_1 activations × quantized weights (many dot products)
     * - Pre-computed sum eliminates expensive horizontal reductions in K-loop
     *
     * Formula: output = d * sum(qs[i])
     *
     * GEMM OPTIMIZATION (Nov 2024):
     * The 'sum_qs' field now stores the raw integer sum of qs[i] values (INT16),
     * NOT d × sum(qs[i]). This eliminates FP16→FP32 conversion and division from
     * the GEMM K-loop, allowing pure VNNI compute followed by scaling in post-processing.
     *
     * Compensation formula in GEMM:
     *   C[i,j] = Σ_kb ((accum[kb] - 128*sum_qs[kb]) * d_a[kb] * d_b[kb])
     */
    struct Q8_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        int16_t sum_qs; ///< INT16 pre-computed sum: Σ(qs[i]) - raw integer sum!
        int8_t qs[32];  ///< 32 quantized int8 values
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q8_1Block) == 36, "Q8_1Block must be 36 bytes");

    /**
     * @brief Q16_1 block: 16-bit quantization with pre-computed sum (72 bytes)
     *
     * Like Q8_1 but with 256× more precision (int16 vs int8).
     * Designed for residual stream where error accumulation is critical.
     *
     * Layout:
     * - uint16_t d: FP16 scale factor (same as Q8_1)
     * - int32_t sum_qs: INT32 pre-computed sum of qs[i] values (wider range needed)
     * - int16_t qs[32]: 32 quantized int16 values
     *
     * Memory: 2 + 4 + 2 (padding) + 64 = 72 bytes per block (2× Q8_1)
     *
     * Range: [-32767, 32767] per element vs [-127, 127] for Q8_1
     * This provides 256× finer granularity at 2× memory cost.
     */
    struct Q16_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        int32_t sum_qs; ///< INT32 pre-computed sum: Σ(qs[i]) - wider range for int16 values!
        int16_t qs[32]; ///< 32 quantized int16 values
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q16_1Block) == 72, "Q16_1Block must be 72 bytes");

    /** @brief Q4_0 block: 4-bit quantization (32 elements per block, 18 bytes) */
    struct Q4_0Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[16]; ///< 32 4-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q4_0Block) == 18, "Q4_0Block must be 18 bytes");

    /** @brief Q4_1 block: 4-bit quantization with min (32 elements per block, 20 bytes) */
    struct Q4_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint16_t m;     ///< FP16 minimum value
        uint8_t qs[16]; ///< 32 4-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q4_1Block) == 20, "Q4_1Block must be 20 bytes");

    /** @brief Q5_0 block: 5-bit quantization (32 elements per block, 22 bytes) */
    struct Q5_0Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qh[4];  ///< High bits (5th bit) for all 32 elements (32 bits = 4 bytes)
        uint8_t qs[16]; ///< Lower 4 bits of 32 5-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q5_0Block) == 22, "Q5_0Block must be 22 bytes");

    /** @brief Q5_1 block: 5-bit quantization with min (32 elements per block, 24 bytes) */
    struct Q5_1Block
    {
        uint16_t d;     ///< FP16 scale factor
        uint16_t m;     ///< FP16 minimum value
        uint8_t qh[4];  ///< High bits (5th bit) for all 32 elements (32 bits = 4 bytes)
        uint8_t qs[16]; ///< Lower 4 bits of 32 5-bit values packed (2 per byte)
        static constexpr size_t BLOCK_SIZE = 32;
    };
    static_assert(sizeof(Q5_1Block) == 24, "Q5_1Block must be 24 bytes");

    // ========================================================================
    // K-Quant Formats (256-element super-blocks)
    // ========================================================================

    /** @brief Q2_K block: 2-bit K-quant (256 elements per super-block, 84 bytes) */
    struct Q2_KBlock
    {
        uint8_t scales[16]; ///< Scales and mins (packed)
        uint8_t qs[64];     ///< 2-bit values packed (4 per byte)
        uint16_t d;         ///< FP16 super-block scale for scales
        uint16_t dmin;      ///< FP16 super-block scale for mins
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q2_KBlock) == 84, "Q2_KBlock must be 84 bytes");

    /** @brief Q3_K block: 3-bit K-quant (256 elements per super-block, 110 bytes) */
    struct Q3_KBlock
    {
        uint8_t hmask[32];  ///< High bit masks (1 bit per element)
        uint8_t qs[64];     ///< Lower 2 bits of 3-bit values
        uint8_t scales[12]; ///< 16 scales packed (6 bits each)
        uint16_t d;         ///< FP16 super-block scale
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q3_KBlock) == 110, "Q3_KBlock must be 110 bytes");

    /** @brief Q4_K block: 4-bit K-quant (256 elements per super-block, 144 bytes) */
    struct Q4_KBlock
    {
        uint16_t d;         ///< FP16 super-block scale
        uint16_t dmin;      ///< FP16 super-block min scale
        uint8_t scales[12]; ///< 12 6-bit scales packed
        uint8_t qs[128];    ///< Lower 4 bits of 4-bit values
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q4_KBlock) == 144, "Q4_KBlock must be 144 bytes");

    /** @brief Q5_K block: 5-bit K-quant (256 elements per super-block, 176 bytes) */
    struct Q5_KBlock
    {
        uint16_t d;         ///< FP16 super-block scale
        uint16_t dmin;      ///< FP16 super-block min scale
        uint8_t scales[12]; ///< 8 6-bit scales packed
        uint8_t qh[32];     ///< High bits (5th bit of 5-bit values)
        uint8_t qs[128];    ///< Lower 4 bits of 5-bit values
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q5_KBlock) == 176, "Q5_KBlock must be 176 bytes");

    /** @brief Q6_K block: 6-bit K-quant (256 elements per super-block, 210 bytes) */
    struct Q6_KBlock
    {
        uint8_t ql[128];   ///< Lower 4 bits of 6-bit values
        uint8_t qh[64];    ///< Upper 2 bits of 6-bit values (packed)
        int8_t scales[16]; ///< Per-block scales
        uint16_t d;        ///< FP16 super-block scale
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q6_KBlock) == 210, "Q6_KBlock must be 210 bytes");

    /** @brief Q8_K block: 8-bit K-quant super-block (256 elements, 288 bytes) */
    struct Q8_KBlock
    {
        int8_t qs[256];    ///< 8-bit quantized values
        int16_t bsums[16]; ///< Block sums for fast dot products
        static constexpr size_t BLOCK_SIZE = 256;
    };
    static_assert(sizeof(Q8_KBlock) == 288, "Q8_KBlock must be 288 bytes");

    // ========================================================================
    // IQ Formats (Importance Quantization, 256-element super-blocks)
    // ========================================================================

    /** @brief IQ2_XXS block: 2-bit extra-extra-small IQ (256 elements per super-block, 66 bytes) */
    struct IQ2_XXSBlock
    {
        uint16_t d;      ///< FP16 scale factor
        uint16_t qs[32]; ///< Grid indices packed (QK_K/8 = 256/8 = 32 uint16_t)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ2_XXSBlock) == 66, "IQ2_XXSBlock must be 66 bytes");

    /** @brief IQ2_XS block: 2-bit extra-small IQ (256 elements per super-block, 74 bytes) */
    struct IQ2_XSBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint16_t qs[32];   ///< Grid indices (QK_K/8 = 32 uint16_t)
        uint8_t scales[8]; ///< Per-block scales (QK_K/32 = 8)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ2_XSBlock) == 74, "IQ2_XSBlock must be 74 bytes");

    /** @brief IQ2_S block: 2-bit small IQ (256 elements per super-block, 82 bytes) */
    struct IQ2_SBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint8_t qs[64];    ///< Quantized values (QK_K/4 = 64)
        uint8_t qh[8];     ///< High bits (QK_K/32 = 8)
        uint8_t scales[8]; ///< Scales (QK_K/32 = 8)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ2_SBlock) == 82, "IQ2_SBlock must be 82 bytes");

    /** @brief IQ3_XXS block: 3-bit extra-extra-small IQ (256 elements per super-block, 98 bytes) */
    struct IQ3_XXSBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[96]; ///< Grid indices (3*QK_K/8 = 3*256/8 = 96)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ3_XXSBlock) == 98, "IQ3_XXSBlock must be 98 bytes");

    /** @brief IQ3_S block: 3-bit small IQ (256 elements per super-block, 110 bytes) */
    struct IQ3_SBlock
    {
        uint16_t d;        ///< FP16 scale factor
        uint8_t qs[64];    ///< Quantized values (QK_K/4 = 64)
        uint8_t qh[8];     ///< High bits (QK_K/32 = 8)
        uint8_t signs[32]; ///< Sign patterns (QK_K/8 = 32)
        uint8_t scales[4]; ///< Scales (IQ3S_N_SCALE = QK_K/64 = 4)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ3_SBlock) == 110, "IQ3_SBlock must be 110 bytes");

    /** @brief IQ4_NL block (already defined above as simple format) */
    // IQ4_NLBlock is defined in the simple quantization section

    /** @brief IQ4_XS block: 4-bit extra-small IQ (256 elements per super-block, 136 bytes) */
    struct IQ4_XSBlock
    {
        uint16_t d;          ///< FP16 scale factor
        uint16_t scales_h;   ///< High bits of scales
        uint8_t scales_l[4]; ///< Low bits of scales (QK_K/64 = 256/64 = 4)
        uint8_t qs[128];     ///< Grid indices (QK_K/2 = 256/2 = 128)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ4_XSBlock) == 136, "IQ4_XSBlock must be 136 bytes");

    /** @brief IQ1_S block: 1-bit small IQ (256 elements per super-block, 50 bytes) */
    struct IQ1_SBlock
    {
        uint16_t d;     ///< FP16 scale factor
        uint8_t qs[32]; ///< Grid indices (QK_K/8 = 256/8 = 32 bytes)
        uint16_t qh[8]; ///< High bits and scales (QK_K/32 = 256/32 = 8 uint16_t values)
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ1_SBlock) == 50, "IQ1_SBlock must be 50 bytes");

    /** @brief IQ1_M block: 1-bit medium IQ (256 elements per super-block, 56 bytes) */
    struct IQ1_MBlock
    {
        uint8_t qs[32];    ///< Grid indices (QK_K/8 = 256/8 = 32 bytes)
        uint8_t qh[16];    ///< High bits and scale info (packed)
        uint8_t scales[8]; ///< Per-block scale adjustments
        static constexpr size_t BLOCK_SIZE = 256;
    } __attribute__((packed));
    static_assert(sizeof(IQ1_MBlock) == 56, "IQ1_MBlock must be 56 bytes");

} // namespace llaminar2
