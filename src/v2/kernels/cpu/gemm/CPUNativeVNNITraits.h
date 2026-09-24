/**
 * @file CPUNativeVNNITraits.h
 * @brief Compile-time traits for CPU NativeVNNI format-specific decode and accumulation.
 *
 * Maps codebook_id (from NativeVnniFormatInfo) to compile-time properties
 * controlling how native-format weight blocks are decoded and accumulated
 * using AVX-512 VNNI instructions.
 *
 * Mirrors the GPU NativeVNNI traits hierarchy:
 * - CUDA: CodebookTraits<CODEBOOK_ID>
 * - ROCm: NVNNIGemmTraits<FMT> / NVNNITraits<FMT>
 * - CPU:  CPUNativeVNNITraits<FMT>
 */

#pragma once

#include <cstdint>

namespace llaminar2::cpu::native_vnni
{

    /**
     * @brief Format enumeration matching GPU codebook IDs.
     *
     * Only formats with IINT8Unpackable + vnniFormatInfo() support are included.
     */
    enum class NativeFormat : uint8_t
    {
        Q4_0 = 0,
        IQ4_NL = 4,
        Q4_1 = 5,
        Q5_0 = 6,
        Q5_1 = 7,
        Q6_K = 8,
        Q3_K = 9,
        Q2_K = 10,
        IQ3_S = 11,
        IQ3_XXS = 12,
        IQ2_S = 13,
        IQ2_XS = 14,
        IQ2_XXS = 15,
        IQ1_S = 16,
        IQ1_M = 17,
        IQ4_XS = 18,
    };

    /**
     * @brief Accumulation pattern for native VNNI decode.
     *
     * S  = Symmetric: single scale per block, values are signed integers
     * A  = Asymmetric: scale + min offset per block
     * D  = Dual-scale: superblock with separate lo/hi scales
     * DA = Dual-scale + Asymmetric: superblock with scales + embedded mins
     */
    enum class AccumPattern
    {
        S,  ///< Symmetric (Q4_0, IQ4_NL, Q5_0, IQ3_S, IQ3_XXS, IQ2_XXS)
        A,  ///< Asymmetric (Q4_1, Q5_1, IQ1_S)
        D,  ///< Dual-scale (Q6_K, Q3_K, IQ2_S, IQ2_XS, IQ1_M)
        DA, ///< Dual-scale asymmetric (Q2_K)
    };

    /**
     * @brief Compile-time traits per native format.
     *
     * Primary template is undefined — only specializations are valid.
     */
    template <NativeFormat FMT>
    struct CPUNativeVNNITraits;

    // =========================================================================
    // Tier 1: Simple 4-bit formats (16-byte payload per 32-element block)
    // =========================================================================

    template <>
    struct CPUNativeVNNITraits<NativeFormat::Q4_0>
    {
        static constexpr uint8_t codebook_id = 0;
        static constexpr int payload_bytes = 16;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::S;
        static constexpr bool is_superblock = false;
        static constexpr bool has_lut = false;
        static constexpr float max_abs_factor = 8.0f;
        /// Nibble bias: Q4_0 values are [0,15], subtract 8 for [-8,7]
        static constexpr int8_t nibble_bias = 8;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ4_NL>
    {
        static constexpr uint8_t codebook_id = 4;
        static constexpr int payload_bytes = 16;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::S;
        static constexpr bool is_superblock = false;
        static constexpr bool has_lut = true; ///< Non-linear lookup table decode
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0; // LUT provides signed values directly
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::Q4_1>
    {
        static constexpr uint8_t codebook_id = 5;
        static constexpr int payload_bytes = 16;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::A;
        static constexpr bool is_superblock = false;
        static constexpr bool has_lut = false;
        static constexpr float max_abs_factor = 15.0f;
        static constexpr int8_t nibble_bias = 0; // Q4_1 is unsigned [0,15]
    };

    // =========================================================================
    // Tier 1: 5-bit formats (20-byte payload per 32-element block)
    // =========================================================================

    template <>
    struct CPUNativeVNNITraits<NativeFormat::Q5_0>
    {
        static constexpr uint8_t codebook_id = 6;
        static constexpr int payload_bytes = 20;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::S;
        static constexpr bool is_superblock = false;
        static constexpr bool has_lut = false;
        static constexpr float max_abs_factor = 16.0f;
        static constexpr int8_t nibble_bias = 16;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::Q5_1>
    {
        static constexpr uint8_t codebook_id = 7;
        static constexpr int payload_bytes = 20;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::A;
        static constexpr bool is_superblock = false;
        static constexpr bool has_lut = false;
        static constexpr float max_abs_factor = 31.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    // =========================================================================
    // Tier 2: K-quant superblock formats
    // =========================================================================

    template <>
    struct CPUNativeVNNITraits<NativeFormat::Q6_K>
    {
        static constexpr uint8_t codebook_id = 8;
        static constexpr int payload_bytes = 24;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::D;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = false;
        static constexpr float max_abs_factor = 32.0f;
        static constexpr int8_t nibble_bias = 32;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::Q3_K>
    {
        static constexpr uint8_t codebook_id = 9;
        static constexpr int payload_bytes = 12;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::D;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = false;
        static constexpr float max_abs_factor = 4.0f;
        static constexpr int8_t nibble_bias = 4;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::Q2_K>
    {
        static constexpr uint8_t codebook_id = 10;
        static constexpr int payload_bytes = 8;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::DA;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = false;
        static constexpr float max_abs_factor = 3.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    // =========================================================================
    // Tier 3: IQ grid formats
    // =========================================================================

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ4_XS>
    {
        static constexpr uint8_t codebook_id = 4; // Same as IQ4_NL (kvalues_iq4nl LUT)
        static constexpr int payload_bytes = 16;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::S;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ3_S>
    {
        static constexpr uint8_t codebook_id = 11;
        static constexpr int payload_bytes = 13;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::S;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ3_XXS>
    {
        static constexpr uint8_t codebook_id = 12;
        static constexpr int payload_bytes = 12;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::S;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ2_S>
    {
        static constexpr uint8_t codebook_id = 13;
        static constexpr int payload_bytes = 9;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::D;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ2_XS>
    {
        static constexpr uint8_t codebook_id = 14;
        static constexpr int payload_bytes = 9;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::D;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ2_XXS>
    {
        static constexpr uint8_t codebook_id = 15;
        static constexpr int payload_bytes = 8;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::S;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ1_S>
    {
        static constexpr uint8_t codebook_id = 16;
        static constexpr int payload_bytes = 6;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::A;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    template <>
    struct CPUNativeVNNITraits<NativeFormat::IQ1_M>
    {
        static constexpr uint8_t codebook_id = 17;
        static constexpr int payload_bytes = 6;
        static constexpr int block_elements = 32;
        static constexpr AccumPattern pattern = AccumPattern::D;
        static constexpr bool is_superblock = true;
        static constexpr bool has_lut = true;
        static constexpr float max_abs_factor = 127.0f;
        static constexpr int8_t nibble_bias = 0;
    };

    /**
     * @brief Runtime format info for dispatch.
     *
     * Populated from NativeVnniFormatInfo at weight-packing time.
     */
    struct RuntimeFormatInfo
    {
        uint8_t codebook_id;
        int payload_bytes;
        bool is_asymmetric;
        bool is_superblock;
        bool has_lut;
    };

    /**
     * @brief Map codebook_id to RuntimeFormatInfo.
     */
    inline RuntimeFormatInfo getRuntimeFormatInfo(uint8_t codebook_id)
    {
        switch (codebook_id)
        {
        case 0:
            return {0, 16, false, false, false}; // Q4_0
        case 4:
            return {4, 16, false, false, true}; // IQ4_NL
        case 5:
            return {5, 16, true, false, false}; // Q4_1
        case 6:
            return {6, 20, false, false, false}; // Q5_0
        case 7:
            return {7, 20, true, false, false}; // Q5_1
        case 8:
            return {8, 24, false, true, false}; // Q6_K
        case 9:
            return {9, 12, false, true, false}; // Q3_K
        case 10:
            return {10, 8, true, true, false}; // Q2_K
        case 11:
            return {11, 13, false, true, true}; // IQ3_S
        case 12:
            return {12, 12, false, true, true}; // IQ3_XXS
        case 13:
            return {13, 9, false, true, true}; // IQ2_S
        case 14:
            return {14, 9, false, true, true}; // IQ2_XS
        case 15:
            return {15, 8, false, true, true}; // IQ2_XXS
        case 16:
            return {16, 6, true, true, true}; // IQ1_S
        case 17:
            return {17, 6, false, true, true}; // IQ1_M
        case 19:
            return {19, 32, false, false, false}; // Q8_0
        case 20:
            return {20, 32, false, false, false}; // Q8_1
        case 21:
            return {21, 32, false, true, false}; // Q8_K source normalized to INT8 blocks
        default:
            return {codebook_id, 0, false, false, false};
        }
    }

} // namespace llaminar2::cpu::native_vnni
