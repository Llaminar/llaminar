/**
 * @file MXFP4Format.h
 * @brief OCP MXFP4 (GGUF type 39) block semantics shared by CPU and GPU code.
 *
 * A GGUF MXFP4 block is 17 bytes for 32 values: one E8M0 shared exponent
 * followed by sixteen bytes of FP4 E2M1 codes (low nibble = element j, high
 * nibble = element j + 16, the same nibble order as IQ4_NL). llama.cpp stores
 * the E2M1 magnitudes doubled as integers, kvalues_mxfp4 =
 * {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}, and halves the
 * exponent: value = kvalues_mxfp4[code] * 2^(e - 128).
 *
 * NativeVNNI execution keeps the sixteen code bytes unchanged and stores the
 * block scale as one FP16 word, exactly like IQ4_NL. Every scale 2^(e - 128)
 * with e - 128 in [-24, 15] is an exact FP16 power of two (subnormal below
 * 2^-14), so the prepared integer dot product times that scale reproduces the
 * source values bit for bit. Exponents outside that window are rejected rather
 * than rounded: a prepared MXFP4 weight is either exact or does not load.
 *
 * One exception keeps llama.cpp compatibility without losing exactness: ggml's
 * quantizer writes e = 0 (2^-128) for an all-zero block. Every value of such a
 * block is zero under any scale, so preparation stores FP16 +0 instead
 * (blockFitsFP16 / preparedScaleFP16Bits).
 */
#pragma once

#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MXFP4_HD __host__ __device__
#else
#define LLAMINAR_MXFP4_HD
#endif

namespace llaminar2::mxfp4
{
    /** @brief Elements per MXFP4 block (OCP MX block size). */
    inline constexpr int kBlockElements = 32;
    /** @brief Source bytes per block: one E8M0 exponent plus sixteen code bytes. */
    inline constexpr int kSourceBlockBytes = 17;

    /** @brief Doubled E2M1 code values (ggml kvalues_mxfp4), indexed by 4-bit code. */
    inline constexpr int8_t kValues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    /** @brief Smallest E8M0 byte whose block scale 2^(e-128) is an exact FP16 value. */
    inline constexpr uint8_t kMinExactFP16Exponent = 128 - 24;
    /** @brief Largest E8M0 byte whose block scale 2^(e-128) is an exact FP16 value. */
    inline constexpr uint8_t kMaxExactFP16Exponent = 128 + 15;

    /** @return Whether the block scale of exponent byte @p e is exactly representable in FP16. */
    LLAMINAR_MXFP4_HD constexpr bool exponentFitsFP16(uint8_t e) noexcept
    {
        return e >= kMinExactFP16Exponent && e <= kMaxExactFP16Exponent;
    }

    /**
     * @brief Whether every code of a block decodes to zero.
     * @param qs Sixteen code bytes (two 4-bit codes each).
     * @return True when each nibble is code 0 or code 8, the two encodings of zero.
     */
    LLAMINAR_MXFP4_HD constexpr bool codesAllZero(const uint8_t *qs) noexcept
    {
        for (int i = 0; i < kSourceBlockBytes - 1; ++i)
            if ((qs[i] & 0x77u) != 0)
                return false;
        return true;
    }

    /**
     * @brief Whether a block has an exact FP16 NativeVNNI preparation.
     * @param e E8M0 exponent byte.
     * @param qs Sixteen code bytes of the same block.
     * @return True for an exactly representable scale, or for an all-zero block
     *         whose scale is irrelevant (ggml writes e = 0 for those).
     */
    LLAMINAR_MXFP4_HD constexpr bool blockFitsFP16(uint8_t e, const uint8_t *qs) noexcept
    {
        return exponentFitsFP16(e) || codesAllZero(qs);
    }

    /**
     * @brief Exact FP16 bit pattern of the block scale 2^(e - 128).
     * @param e E8M0 exponent byte; must satisfy exponentFitsFP16().
     * @return Normal FP16 power of two for e - 128 >= -14, otherwise the exact subnormal.
     */
    LLAMINAR_MXFP4_HD constexpr uint16_t scaleFP16Bits(uint8_t e) noexcept
    {
        const int exponent = static_cast<int>(e) - 128;
        return exponent >= -14
                   ? static_cast<uint16_t>((exponent + 15) << 10)
                   : static_cast<uint16_t>(1u << (exponent + 24));
    }

    /**
     * @brief FP16 scale stored by NativeVNNI preparation for a block accepted by blockFitsFP16().
     * @param e E8M0 exponent byte.
     * @return scaleFP16Bits(e) when representable, otherwise +0 (only reachable for all-zero blocks).
     */
    LLAMINAR_MXFP4_HD constexpr uint16_t preparedScaleFP16Bits(uint8_t e) noexcept
    {
        return exponentFitsFP16(e) ? scaleFP16Bits(e) : uint16_t{0};
    }

    /**
     * @brief Smallest E8M0 byte whose scale 2^(e-128) is a normal FP32 value.
     *
     * Kernels that build the FP32 scale from the exponent bits (native E8M0
     * scale plane) must not produce FP32 subnormals: CUDA builds flush
     * subnormals to zero (--ftz=true), which would silently lose exactness.
     */
    inline constexpr uint8_t kMinNormalFP32Exponent = 128 - 126;
    /**
     * @brief Largest E8M0 byte accepted by the native scale plane.
     *
     * OCP MX defines 0xFF as NaN. ggml decodes it as the finite 2^127, but no
     * real checkpoint uses it; nonzero blocks with it are rejected, not guessed.
     */
    inline constexpr uint8_t kMaxNativeE8M0Exponent = 254;

    /** @return Whether exponent byte @p e has an exact normal FP32 scale for the native plane. */
    LLAMINAR_MXFP4_HD constexpr bool exponentFitsNativeE8M0(uint8_t e) noexcept
    {
        return e >= kMinNormalFP32Exponent && e <= kMaxNativeE8M0Exponent;
    }

    /**
     * @brief IEEE-754 binary32 bits of a normal scale 2^(e - 128).
     * @param e Exponent byte satisfying exponentFitsNativeE8M0().
     * @return Bits with biased exponent e - 1 and zero mantissa.
     *
     * One integer shift: the device builds the scale without libm, FP16
     * widening or a lookup table.
     */
    LLAMINAR_MXFP4_HD constexpr uint32_t normalScaleFP32Bits(uint8_t e) noexcept
    {
        return static_cast<uint32_t>(e - 1u) << 23;
    }

    /**
     * @brief CPU-unit word for a native E8M0 scale: a BF16 power of two.
     * @param e Stored plane byte: 0 marks an all-zero block, otherwise
     *        exponentFitsNativeE8M0(e).
     * @return 0 for the zero marker, otherwise (e - 1) << 7.
     *
     * CPU units keep a 16-bit scale slot. Zero-extending this word and shifting
     * it left by 16 gives exactly normalScaleFP32Bits(e), so SIMD widening needs
     * no BF16 arithmetic and the word never goes through F16C.
     */
    LLAMINAR_MXFP4_HD constexpr uint16_t powerOfTwoBF16Word(uint8_t e) noexcept
    {
        return e == 0 ? uint16_t{0} : static_cast<uint16_t>((e - 1u) << 7);
    }

    /**
     * @brief Recover the E8M0 plane byte from a CPU-unit BF16 power-of-two word.
     * @param word Candidate word produced by powerOfTwoBF16Word().
     * @param[out] e Plane byte (0 for the zero marker).
     * @return False for any word that is not +0 or a positive normal power of two
     *         with biased exponent 1..253 (e in [2, 254]).
     */
    LLAMINAR_MXFP4_HD constexpr bool e8m0FromPowerOfTwoBF16Word(uint16_t word, uint8_t &e) noexcept
    {
        if (word == 0)
        {
            e = 0;
            return true;
        }
        const unsigned biased = word >> 7;
        if ((word & 0x807fu) != 0 || biased < 1u || biased > 253u)
            return false;
        e = static_cast<uint8_t>(biased + 1u);
        return true;
    }

    /** @brief Exact FP32 block scale 2^(e - 128) for any exponent byte. */
    inline float scaleFP32(uint8_t e) noexcept
    {
        // 2^(e-128) spans 2^-128 .. 2^127: every value is an exact FP32 (normal
        // or subnormal) power of two. Build it from bits to avoid libm.
        const int exponent = static_cast<int>(e) - 128;
        uint32_t bits = exponent >= -126
                            ? static_cast<uint32_t>(exponent + 127) << 23
                            : (1u << (exponent + 149));
        float out;
        __builtin_memcpy(&out, &bits, sizeof(out));
        return out;
    }

    static_assert(scaleFP16Bits(128) == 0x3c00u, "2^0 is FP16 one");
    static_assert(scaleFP16Bits(128 - 14) == 0x0400u, "2^-14 is the smallest FP16 normal");
    static_assert(scaleFP16Bits(128 - 24) == 0x0001u, "2^-24 is the smallest FP16 subnormal");
    static_assert(scaleFP16Bits(128 + 15) == 0x7800u, "2^15 is the largest FP16 power of two");
    static_assert(!exponentFitsFP16(103) && !exponentFitsFP16(144));
    static_assert(preparedScaleFP16Bits(0) == 0u && preparedScaleFP16Bits(128) == 0x3c00u);
    static_assert(normalScaleFP32Bits(128) == 0x3f800000u, "2^0 is FP32 one");
    static_assert(normalScaleFP32Bits(kMinNormalFP32Exponent) == 0x00800000u, "2^-126 is the smallest FP32 normal");
    static_assert(normalScaleFP32Bits(kMaxNativeE8M0Exponent) == 0x7e800000u, "2^126");
    static_assert(!exponentFitsNativeE8M0(1) && !exponentFitsNativeE8M0(255) && exponentFitsNativeE8M0(104));
    static_assert((uint32_t{powerOfTwoBF16Word(128)} << 16) == normalScaleFP32Bits(128));
    static_assert((uint32_t{powerOfTwoBF16Word(2)} << 16) == normalScaleFP32Bits(2));
    static_assert((uint32_t{powerOfTwoBF16Word(254)} << 16) == normalScaleFP32Bits(254));
} // namespace llaminar2::mxfp4

#undef LLAMINAR_MXFP4_HD
