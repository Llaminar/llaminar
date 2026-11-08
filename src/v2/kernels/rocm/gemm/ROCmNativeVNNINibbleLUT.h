/**
 * @file ROCmNativeVNNINibbleLUT.h
 * @brief Packed signed-nibble lookup words shared by every ROCm NativeVNNI decoder.
 *
 * ROCm kernels expand 4-bit codes to INT8 with three `v_perm_b32` byte
 * selections: one each against the lookup halves for codes 0-7 and 8-15, then a
 * byte mux on code bit 3. Only the four packed lookup words depend on the
 * codebook. IQ4_NL (4) and the original IQ4_XS tuple (24) share
 * `kvalues_iq4nl`. MXFP4 (25) uses the doubled E2M1 values
 * (`mxfp4::kValues`). Keeping the words in one place means a new nibble
 * codebook cannot silently reuse the IQ4_NL table. The static assertions below
 * check every packed byte against the canonical source tables.
 */
#pragma once

#include "tensors/MXFP4Format.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <cstdint>

namespace llaminar2::rocm::nibble_lut
{
    /** @brief Four little-endian lookup words: codes 0-3, 4-7, 8-11 and 12-15. */
    struct Words
    {
        uint32_t lo_a; ///< Codes 0..3.
        uint32_t lo_b; ///< Codes 4..7.
        uint32_t hi_a; ///< Codes 8..11.
        uint32_t hi_b; ///< Codes 12..15.
    };

    /** @brief ggml kvalues_iq4nl, the IQ4_NL / IQ4_XS signed lookup. */
    inline constexpr int8_t kIQ4NLValues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

    /** @brief Whether @p codebook decodes through a 16-entry signed nibble lookup. */
    constexpr bool isLookupCodebook(int codebook) noexcept
    {
        return codebook == 4 || codebook == kNativeVnniIQ4XSSourceScaleCodebook ||
               codebook == kNativeVnniMXFP4Codebook;
    }

    /** @brief Pack four signed lookup entries into one little-endian byte word. */
    constexpr uint32_t pack(const int8_t *values) noexcept
    {
        return uint32_t(uint8_t(values[0])) | (uint32_t(uint8_t(values[1])) << 8) |
               (uint32_t(uint8_t(values[2])) << 16) | (uint32_t(uint8_t(values[3])) << 24);
    }

    /**
     * @brief Lookup words of a nibble codebook.
     * @tparam Codebook IQ4_NL (4), IQ4_XS source tuple (24) or MXFP4 (25).
     */
    template <int Codebook>
    constexpr Words words() noexcept
    {
        static_assert(isLookupCodebook(Codebook), "Codebook has no signed nibble lookup");
        const int8_t *values = Codebook == kNativeVnniMXFP4Codebook ? mxfp4::kValues : kIQ4NLValues;
        return {pack(values), pack(values + 4), pack(values + 8), pack(values + 12)};
    }

    static_assert(words<4>().lo_a == 0xBFAD9881u && words<4>().lo_b == 0xF6EADDCFu &&
                      words<4>().hi_a == 0x26190D01u && words<4>().hi_b == 0x71594535u,
                  "IQ4_NL lookup words must equal the historical ROCm constants");
    static_assert(words<kNativeVnniMXFP4Codebook>().lo_a == 0x03020100u &&
                      words<kNativeVnniMXFP4Codebook>().lo_b == 0x0C080604u &&
                      words<kNativeVnniMXFP4Codebook>().hi_a == 0xFDFEFF00u &&
                      words<kNativeVnniMXFP4Codebook>().hi_b == 0xF4F8FAFCu,
                  "MXFP4 lookup words must equal the doubled E2M1 table");

#if defined(__HIPCC__)
    /**
     * @brief Expand four byte-local nibbles (already masked to 0x0F0F0F0F) to INT8.
     * @tparam Codebook Lookup codebook; see words().
     * @param nibbles Four codes, one per byte.
     * @return Four signed INT8 lookup values in byte order.
     */
    template <int Codebook>
    __device__ __forceinline__ uint32_t decode4(uint32_t nibbles)
    {
        constexpr Words lut = words<Codebook>();
        // __builtin_amdgcn_perm(a, b, sel) maps sel 0-3 -> b and 4-7 -> a.
        const uint32_t sel = nibbles & 0x07070707u;
        const uint32_t low = __builtin_amdgcn_perm(lut.lo_b, lut.lo_a, sel);
        const uint32_t high = __builtin_amdgcn_perm(lut.hi_b, lut.hi_a, sel);
        const uint32_t mux = 0x03020100u | ((nibbles >> 1) & 0x04040404u);
        return __builtin_amdgcn_perm(high, low, mux);
    }
#endif
} // namespace llaminar2::rocm::nibble_lut
