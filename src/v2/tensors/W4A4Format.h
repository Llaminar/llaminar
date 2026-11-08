/**
 * @file W4A4Format.h
 * @brief Native ruquant W4A4/H storage and exact integer arithmetic shared by backends.
 *
 * These are online-A4 operands, not a weight-only NativeVNNI codebook. A source
 * block contains 256 signed nibble codes plus an FP32 row unit and integer
 * policy metadata. Nibbles enumerate adjacent K elements, unlike MXFP4's two
 * half-block planes. Weight units must never be rounded to FP16 or multiplied
 * into the nibble codes. The group correction and arithmetic shift precede
 * the complete int32 sum; only the final epilogue is floating point.
 *
 * The normative source is skyne98/mx-llama.cpp commit
 * bfc4c5dc3c96df652f42924370ef49b728d585ad. Matrix admission is host-owned in
 * W4A4MatrixContract; this header contains only device-safe value contracts.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_W4A4_HD __host__ __device__
#else
#define LLAMINAR_W4A4_HD
#endif

namespace llaminar2::w4a4
{
    /** @brief Version of group-64 activation quantization and shift-before-sum arithmetic. */
    inline constexpr std::uint32_t kArithmeticVersion = 1;
    /** @brief Logical values per serialized source block. */
    inline constexpr int kBlockElements = 256;
    /** @brief Logical values in each independently corrected activation group. */
    inline constexpr int kGroupElements = 64;
    /** @brief Number of native arithmetic groups in one source block. */
    inline constexpr int kGroupsPerBlock = kBlockElements / kGroupElements;
    /** @brief Exact serialized bytes per source block. */
    inline constexpr std::size_t kSourceBlockBytes = 144;
    /** @brief Integer division applied to every signed group term before reduction. */
    inline constexpr int kTermShift = 4;
    /** @brief Largest admitted integer activation-group multiplier. */
    inline constexpr int kActivationMultiplierMax = 255;
    /** @brief Largest admitted integer weight-group multiplier. */
    inline constexpr int kWeightMultiplierMax = 15;
    /** @brief Worst absolute shifted term, from |w| <= 8 and |q-z| <= 15. */
    inline constexpr std::int32_t kMaximumGroupTerm =
        (kWeightMultiplierMax * kActivationMultiplierMax * 8 * 15 * kGroupElements) >> kTermShift;
    /** @brief Maximum complete group count whose sum cannot overflow signed int32. */
    inline constexpr int kMaximumGroups = std::numeric_limits<std::int32_t>::max() / kMaximumGroupTerm;

    /** @brief Online transform declared by the source tensor, never selected as a fallback. */
    enum class Transform : std::uint8_t { Identity, PaleySylvester };

    /**
     * @brief Exact little-endian GGUF block shared by types 44 and 45.
     *
     * The descriptor is either a repeated whole-row code sum, a tagged block
     * multiplier, or four tagged group-64 multipliers. The row unit repeats
     * bit-identically across its blocks. Rotation and clipping metadata repeat
     * across the whole matrix and are sealed before graph construction.
     */
    struct SourceBlock
    {
        float unit;                         ///< FP32 row unit, including its exact zero sign.
        std::uint32_t descriptor;           ///< Tagged integer weight multipliers or row code sum.
        std::int32_t rotation_length;       ///< Full row width for W4A4; transform-block width for H.
        std::uint32_t clip_bits;             ///< FP32 activation clipping bits; zero means unset/one.
        std::uint8_t codes[128];             ///< Adjacent signed low/high nibble codes along K.
    };
    static_assert(sizeof(SourceBlock) == kSourceBlockBytes);
    static_assert(offsetof(SourceBlock, codes) == 16);
    static_assert(std::is_standard_layout_v<SourceBlock> && std::is_trivially_copyable_v<SourceBlock>);

    /** @brief Fixed factorization; ascending Paley search is part of transform identity. */
    struct HadamardGeometry
    {
        int length;     ///< Number of values in one normalized transform block.
        int paley;      ///< Paley-I order, or one for a pure Sylvester transform.
        int sylvester;  ///< Power-of-two butterfly width within each Paley row.
    };

    /**
     * @brief Decode one adjacent signed int4 code without implementation-defined narrowing.
     * @param nibble Packed value; only its four low bits are significant.
     * @return The exact signed value in [-8, 7].
     */
    LLAMINAR_W4A4_HD constexpr std::int8_t signedNibble(std::uint8_t nibble) noexcept
    {
        const int value = nibble & 15;
        return static_cast<std::int8_t>(value < 8 ? value : value - 16);
    }

    /**
     * @brief Resolve the integer weight multiplier for one admitted group.
     * @param descriptor Native descriptor bits from that group's source block.
     * @param group Group index in [0, 4); callers use admitted geometry.
     * @return The encoded 1..15 multiplier, or one for the whole-row encoding.
     *
     * Group-64 tags also match the broader group-256 tag; precedence is part
     * of the storage contract. Admission rejects zero encoded multipliers.
     */
    LLAMINAR_W4A4_HD constexpr std::uint8_t weightMultiplier(
        std::uint32_t descriptor, unsigned group) noexcept
    {
        return static_cast<std::uint8_t>((descriptor >> 29) == 3
            ? (descriptor >> (4 * group)) & 15
            : (descriptor >> 30) == 1 ? descriptor & 15 : 1);
    }

    /**
     * @brief Correct and scale one integer dot, then shift before any K reduction.
     * @param dot Sum of 64 signed weight-code times signed activation-code products.
     * @param weight_sum Sum of the same 64 signed weight codes.
     * @param zero_point Admitted asymmetric activation zero point in [0, 15].
     * @param activation_multiplier Admitted activation multiplier in [1, 255].
     * @param weight_multiplier Admitted weight multiplier in [1, 15].
     * @return Exact shifted term; every intermediate fits int32 under these bounds.
     */
    LLAMINAR_W4A4_HD constexpr std::int32_t shiftedGroupTerm(
        std::int32_t dot, std::int32_t weight_sum, int zero_point,
        int activation_multiplier, int weight_multiplier) noexcept
    {
        const std::int32_t corrected = dot + (8 - zero_point) * weight_sum;
        return (corrected * weight_multiplier * activation_multiplier) >> kTermShift;
    }
    static_assert((-1 >> kTermShift) == -1, "W4A4 requires arithmetic signed right shift");
    static_assert(kMaximumGroups >= 17408 / kGroupElements);
} // namespace llaminar2::w4a4

#undef LLAMINAR_W4A4_HD
