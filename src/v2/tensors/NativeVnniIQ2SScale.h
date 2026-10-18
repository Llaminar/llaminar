/**
 * @file NativeVnniIQ2SScale.h
 * @brief Lossless IQ2_S raw scale-byte metadata for CPU and GPU preparation.
 *
 * A logical 32-value block retains its original half base in the primary word
 * and its original source scales byte in the low eight secondary bits. The
 * upper eight bits are zero. Consumers widen the base and reconstruct each
 * factor as RN32(RN32(base * (0.5f + nibble)) * 0.25f), preserving the source
 * expression without narrowing either derived factor. Payload, two integer
 * dots, activation quantization and reduction ownership remain unchanged.
 */
#pragma once
#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_IQ2S_SCALE_HD __host__ __device__
#else
#define LLAMINAR_IQ2S_SCALE_HD
#endif
namespace llaminar2::native_vnni_iq2s_scale
{
    /**
     * @brief Preserve the original unsigned source byte without conversion.
     * @param source_scales Original IQ2_SBlock::scales byte.
     * @return Opaque secondary word with its upper eight bits zero.
     */
    LLAMINAR_IQ2S_SCALE_HD constexpr uint16_t pack(uint8_t source_scales) noexcept
    { return static_cast<uint16_t>(source_scales); }
    /**
     * @brief Extract the source coefficient for the first sixteen values.
     * @param metadata Opaque secondary word; never an FP16 scale.
     * @return Unsigned low nibble in the range zero through fifteen.
     */
    LLAMINAR_IQ2S_SCALE_HD constexpr int low(uint16_t metadata) noexcept
    { return static_cast<int>(metadata & 15u); }
    /**
     * @brief Extract the source coefficient for the last sixteen values.
     * @param metadata Opaque secondary word; never an FP16 scale.
     * @return Unsigned high nibble in the range zero through fifteen.
     */
    LLAMINAR_IQ2S_SCALE_HD constexpr int high(uint16_t metadata) noexcept
    { return static_cast<int>((metadata >> 4) & 15u); }
    /**
     * @brief Check the canonical raw-word storage invariant without repairing it.
     * @param metadata Opaque secondary word.
     * @return Whether its reserved upper eight bits are zero.
     */
    LLAMINAR_IQ2S_SCALE_HD constexpr bool valid(uint16_t metadata) noexcept
    { return (metadata & 0xff00u) == 0; }
    static_assert(low(pack(0xf0)) == 0 && high(pack(0xf0)) == 15);
    static_assert(low(pack(0x0f)) == 15 && high(pack(0x0f)) == 0);
    static_assert(valid(pack(0xff)) && !valid(0x0100));
}
#undef LLAMINAR_IQ2S_SCALE_HD
