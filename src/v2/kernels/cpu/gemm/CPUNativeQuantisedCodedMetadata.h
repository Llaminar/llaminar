/**
 * @file CPUNativeQuantisedCodedMetadata.h
 * @brief Exact-width code planes, superblock words and their SIMD reconstruction for CPU units.
 *
 * Superblock-coded CPU units (preparedSuperblockCoded()) store, per 64-column
 * block, the source scale code at its exact bit width and, per 64-column
 * source superblock, one `d` (or `d | dmin << 16`) word. This header owns
 * that byte layout and its readers:
 *
 * | Code bits | Planes after the block's payload (64 columns) |
 * |---|---|
 * | 16 | `u16[64]` |
 * | 12 | `u8[64]` low byte, then a nibble plane (high 4 bits) |
 * | 8 | `u8[64]` |
 * | 6 | nibble plane (low 4 bits), then a 2-bit plane (high 2 bits) |
 * | 4 | nibble plane |
 *
 * Every plane is column-contiguous at its exact width: a nibble plane is 32
 * bytes with columns 2j and 2j + 1 in byte j (low, high nibble); a 2-bit
 * plane is 16 bytes with columns 4j..4j + 3 in byte j (bits 0-1 .. 6-7). L
 * live columns therefore occupy exactly ceil(L * bits / 8) bytes of each
 * plane, which is what a live-column wire unit carries. SIMD readers load the
 * contiguous bytes of an aligned column vector, duplicate each byte into its
 * lanes with one constant shuffle and apply a constant per-lane shift.
 *
 * Reconstruction mirrors decodeSuperblockMetadata() exactly: the same binary32
 * expression in the same order, rounded once to binary16 by the hardware
 * round-to-nearest-even conversion, the publication contract's signed-zero
 * canonicalization (-0 -> +0), then widened by the ISA's existing metadata
 * widening. The result equals widening the per-block word the codec
 * reconstructs, so arithmetic downstream is unchanged.
 *
 * AArch64 envelopes with FP16 vector arithmetic additionally own a fused
 * Q4_K reader (neon_half::q4KFactors): the binary16 product of the source
 * half and the integer code is that same single RNE rounding, so the scale and
 * minimum of four columns cost one table gather, one half multiply and two
 * widenings per block instead of the generic binary32 program. Like every
 * native NEON projection program it is exact under the admitted per-thread
 * FPCR profile (cpu::ARMFloatingPointProfile), which the projection entries
 * require; signed-zero canonicalization is an integer select, never an add.
 */
#pragma once

#include "CPUNativeQuantisedPreparedFootprint.h"
#include "tensors/NativeQuantisedSuperblockMetadata.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace llaminar2::cpu::native_quantised::coded_metadata
{
    namespace sb = ::llaminar2::native_quantised_superblock;

    /** @return Bytes of one 64-column code block for @p codebook (8 x code bits). */
    [[nodiscard]] constexpr std::uint32_t codeBlockBytes(std::uint8_t codebook) noexcept
    {
        return 8u * sb::superblockCodePlaneBits(codebook);
    }

    /** @return Bytes of one 64-column superblock header for @p codebook. */
    [[nodiscard]] constexpr std::uint32_t headerBytes(std::uint8_t codebook) noexcept
    {
        return 64u * sb::superblockFieldBytes(codebook);
    }

    /**
     * @brief Store one column's code into a block's code planes.
     * @param planes First code-plane byte of the block (zero-initialized by the producer).
     * @param codebook Superblock-coded codebook with a separate code plane.
     * @param column Column 0..63.
     * @param code Zero-extended source code.
     */
    inline void storeCode(std::uint8_t *planes, std::uint8_t codebook, int column, std::uint32_t code)
    {
        const auto nibble = [&](std::uint8_t *plane, std::uint32_t value) {
            plane[column >> 1] = static_cast<std::uint8_t>(plane[column >> 1] | ((value & 15u) << ((column & 1) * 4)));
        };
        switch (sb::superblockCodePlaneBits(codebook))
        {
        case 16:
            planes[2 * column] = static_cast<std::uint8_t>(code);
            planes[2 * column + 1] = static_cast<std::uint8_t>(code >> 8);
            return;
        case 12:
            planes[column] = static_cast<std::uint8_t>(code);
            nibble(planes + 64, code >> 8);
            return;
        case 8:
            planes[column] = static_cast<std::uint8_t>(code);
            return;
        case 6:
            nibble(planes, code);
            planes[32 + (column >> 2)] = static_cast<std::uint8_t>(
                planes[32 + (column >> 2)] | (((code >> 4) & 3u) << ((column & 3) * 2)));
            return;
        case 4:
            nibble(planes, code);
            return;
        default:
            throw std::invalid_argument("CPU coded unit codebook has no separate code plane");
        }
    }

    /** @brief Scalar reader of storeCode(). */
    [[nodiscard]] inline std::uint32_t loadCode(const std::uint8_t *planes, std::uint8_t codebook, int column)
    {
        const auto nibble = [&](const std::uint8_t *plane) {
            return static_cast<std::uint32_t>(plane[column >> 1] >> ((column & 1) * 4)) & 15u;
        };
        switch (sb::superblockCodePlaneBits(codebook))
        {
        case 16: return std::uint32_t{planes[2 * column]} | (std::uint32_t{planes[2 * column + 1]} << 8);
        case 12: return std::uint32_t{planes[column]} | (nibble(planes + 64) << 8);
        case 8: return planes[column];
        case 6: return nibble(planes) | (((std::uint32_t{planes[32 + (column >> 2)]} >> ((column & 3) * 2)) & 3u) << 4);
        case 4: return nibble(planes);
        default: throw std::invalid_argument("CPU coded unit codebook has no separate code plane");
        }
    }

    /** @brief Store one column's superblock word into a 64-column header. */
    inline void storeSuperblockWord(std::uint8_t *header, std::uint8_t codebook, int column, std::uint32_t word)
    {
        std::memcpy(header + static_cast<std::size_t>(column) * sb::superblockFieldBytes(codebook), &word,
                    sb::superblockFieldBytes(codebook));
    }

    /** @brief Scalar reader of storeSuperblockWord(). */
    [[nodiscard]] inline std::uint32_t loadSuperblockWord(const std::uint8_t *header, std::uint8_t codebook, int column)
    {
        std::uint32_t word = 0;
        std::memcpy(&word, header + static_cast<std::size_t>(column) * sb::superblockFieldBytes(codebook),
                    sb::superblockFieldBytes(codebook));
        return word;
    }

    /** @brief Write Q5_K high bits as eight 32-byte planes, one nibble per column in each group.
     * @param planes Zero-initialized high-bit region after the four low-nibble words.
     * @param column Physical output column in [0,63].
     * @param qh Four source high-bit bytes in increasing K order.
     */
    inline void storeQ5HighBits(std::uint8_t *planes, int column, const std::uint8_t *qh)
    {
        for (int group = 0; group < 8; ++group)
        {
            const auto bits = (qh[group / 2] >> (4 * (group % 2))) & 15;
            planes[32 * group + column / 2] |= static_cast<std::uint8_t>(bits << (4 * (column % 2)));
        }
    }

    /** @brief Reconstruct the four original Q5_K high-bit bytes for one column. */
    inline void loadQ5HighBits(const std::uint8_t *planes, int column, std::uint8_t *qh)
    {
        for (int byte = 0; byte < 4; ++byte)
        {
            const int shift = 4 * (column % 2);
            const auto *even = planes + 64 * byte + column / 2;
            const int low = (*even >> shift) & 15;
            const int high = (even[32] >> shift) & 15;
            qh[byte] = static_cast<std::uint8_t>(low | (high << 4));
        }
    }

    /**
     * @brief Code of one column of a block, including code-in-payload formats.
     * @param block First byte of the 64-column block (payload planes first).
     * @param data_stride Payload bytes of the block (code planes follow).
     * @param payload Column's payload bytes (needed for code-in-payload formats).
     */
    [[nodiscard]] inline std::uint32_t columnCode(const std::uint8_t *block, std::uint32_t data_stride,
                                                  std::uint8_t codebook, int column, const std::uint8_t *payload)
    {
        switch (codebook)
        {
        case 12: return sb::codeFromPayload<12>(payload);
        case 14: return sb::codeFromPayload<14>(payload);
        case 15: return sb::codeFromPayload<15>(payload);
        case 16: return sb::codeFromPayload<16>(payload);
        default: return loadCode(block + data_stride, codebook, column);
        }
    }

    /** @brief Narrow GCC vector types per lane count (attributes on dependent aliases are ignored). */
    template <int Lanes> struct NarrowVectors;
    template <> struct NarrowVectors<4>
    {
        typedef std::uint8_t U8 __attribute__((vector_size(4)));
        typedef std::uint16_t U16 __attribute__((vector_size(8)));
    };
    template <> struct NarrowVectors<8>
    {
        typedef std::uint8_t U8 __attribute__((vector_size(8)));
        typedef std::uint16_t U16 __attribute__((vector_size(16)));
    };
    template <> struct NarrowVectors<16>
    {
        typedef std::uint8_t U8 __attribute__((vector_size(16)));
        typedef std::uint16_t U16 __attribute__((vector_size(32)));
    };

    /** @brief SIMD lane readers; Ops supplies the vector types and half conversions. */
    template <typename Ops> struct Lanes
    {
        using I = typename Ops::I;
        using F = typename Ops::F;
        static constexpr int kLanes = static_cast<int>(sizeof(I) / sizeof(int));
        static_assert(kLanes == 4 || kLanes == 8 || kLanes == 16);

        /** @return kLanes contiguous unsigned bytes, zero-extended. */
        static I bytes(const std::uint8_t *source)
        {
            typename NarrowVectors<kLanes>::U8 value;
            std::memcpy(&value, source, sizeof(value));
            return __builtin_convertvector(value, I);
        }
        /** @return kLanes contiguous unsigned halfwords, zero-extended. */
        static I halfwords(const std::uint8_t *source)
        {
            typename NarrowVectors<kLanes>::U16 value;
            std::memcpy(&value, source, sizeof(value));
            return __builtin_convertvector(value, I);
        }
        /** @return kLanes contiguous 32-bit words. */
        static I words(const std::uint8_t *source)
        {
            I value;
            std::memcpy(&value, source, sizeof(value));
            return value;
        }
        /**
         * @brief Spread @p Width-bit fields of kLanes columns from their packed bytes.
         * @tparam Width 4 (two columns per byte) or 2 (four columns per byte).
         * @param plane Plane base; @p column is a multiple of kLanes.
         */
        template <int Width> static I fields(const std::uint8_t *plane, int column)
        {
            constexpr int per_byte = 8 / Width;
            const auto *source = plane + column / per_byte;
            if constexpr (Width == 4 && requires { Ops::unpackNibbles(source); })
                return Ops::unpackNibbles(source);
            else
            {
                typename NarrowVectors<kLanes>::U8 packed{};
                std::memcpy(&packed, source, kLanes / per_byte);
                return spread<Width>(packed, std::make_index_sequence<kLanes>{});
            }
        }
        /** @return Nibble-plane field of columns [column, column + kLanes). */
        static I nibbles(const std::uint8_t *plane, int column) { return fields<4>(plane, column); }
        /** @return 2-bit-plane field of columns [column, column + kLanes). */
        static I crumbs(const std::uint8_t *plane, int column) { return fields<2>(plane, column); }

        /** @return Codes of columns [column, column + kLanes) from a block's separate code planes. */
        template <std::uint8_t Codebook> static I code(const std::uint8_t *planes, int column)
        {
            constexpr std::uint32_t bits = sb::superblockCodePlaneBits(Codebook);
            static_assert(bits != 0, "code-in-payload codebooks read their code from the payload");
            if constexpr (bits == 16)
                return halfwords(planes + 2 * column);
            else if constexpr (bits == 12)
                return bytes(planes + column) | (nibbles(planes + 64, column) << 8);
            else if constexpr (bits == 8)
                return bytes(planes + column);
            else if constexpr (bits == 6)
                return nibbles(planes, column) | (crumbs(planes + 32, column) << 4);
            else
                return nibbles(planes, column);
        }

        /** @return Superblock words of columns [column, column + kLanes). */
        template <std::uint8_t Codebook> static I superblockWords(const std::uint8_t *header, int column)
        {
            if constexpr (sb::superblockHasDmin(Codebook))
                return words(header + 4 * column);
            else
                return halfwords(header + 2 * column);
        }

        /**
         * @brief Restore the even-parity top bit of four packed sign selectors per lane.
         *
         * Code-in-payload formats keep their code in the sign bytes' top bits;
         * the decoder needs ksigns_iq2xs masks, whose top bit is the parity of
         * the low seven bits (restoreSignParity, per byte).
         */
        static I restoreSignParity(I word)
        {
            const I low = word & 0x7F7F7F7F;
            I parity = low ^ (low >> 4);
            parity ^= parity >> 2;
            parity ^= parity >> 1;
            return low | ((parity & 0x01010101) << 7);
        }

        /** @brief Duplicate each packed byte into its lanes and shift each lane to its field. */
        template <int Width, typename U8, std::size_t... Lane>
        static I spread(U8 packed, std::index_sequence<Lane...>)
        {
            constexpr int per_byte = 8 / Width;
            const U8 duplicated = __builtin_shufflevector(packed, packed, static_cast<int>(Lane / per_byte)...);
            const I shift{static_cast<int>((Lane % per_byte) * Width)...};
            return (__builtin_convertvector(duplicated, I) >> shift) & ((1 << Width) - 1);
        }

        /** @return FP32 widening of round-to-nearest-even binary16 with -0 canonicalized to +0. */
        static F staged(F value)
        {
            if constexpr (requires { Ops::stagedHalf(value); })
                return Ops::stagedHalf(value);
            else
            {
                I bits = Ops::halfBits(value);
                bits &= ~(bits == 0x8000);
                return Ops::halfWords(bits);
            }
        }
        /** @return FP32 widening of the low binary16 of each lane. */
        static F half(I word) { return Ops::halfWords(word & 0xFFFF); }
        /** @return Lanes of @p value as FP32. */
        static F toFloat(I value) { return __builtin_convertvector(value, F); }
    };

    /** @brief Reconstructed per-column factors of one block, as the per-block words widened. */
    template <typename F> struct Factors
    {
        F primary{};   ///< Widened primary word (scale, or source `d` for source-scale formats).
        F secondary{}; ///< Widened secondary word (second scale or additive minimum); unused for source-scale formats.
        F emin_low{};  ///< Q2_K effective minimum of the low half.
        F emin_high{}; ///< Q2_K effective minimum of the high half.
    };

    /** @brief Widened Q4_K source header reused by every block of one superblock. */
    template <typename F> struct Q4KBaseFactors
    {
        F d;
        F dmin;
    };

    /** @return Q4_K's original d and dmin, widened exactly for one lane vector.
     * @details Always inlined on AArch64: the native NEON tile register witness
     * admits no call inside a tile program, and GCC 14's translation-unit
     * inlining budget otherwise left this helper out of line in the
     * backend-native Q4_K tiles once the GPU-aligned programs grew.
     */
    template <typename Ops>
#if defined(__aarch64__)
    [[gnu::always_inline]]
#endif
    inline Q4KBaseFactors<typename Ops::F> q4KBase(typename Ops::I word)
    {
        using L = Lanes<Ops>;
        if constexpr (requires { Ops::halfProductInput(word); })
            return {Ops::halfProductInput(word), Ops::halfProductInput(word >> 16)};
        else
            return {L::half(word), L::half(word >> 16)};
    }

    /** @return The staged Q4_K scale and minimum from one code and its reusable source factors. */
    template <std::uint8_t Codebook, typename Ops> requires (Codebook == 5)
    inline Factors<typename Ops::F> reconstruct(typename Ops::I code, const Q4KBaseFactors<typename Ops::F> &base)
    {
        using L = Lanes<Ops>;
        // Code = 6-bit scale | 6-bit min << 6. Keep both old binary16
        // rounding points and signed-zero normalization of the source codec.
        const auto primary = base.d * L::toFloat(code & 63);
        if constexpr (requires { Ops::stagedNegatedProduct(primary); })
        {
            // AArch64: the codec evaluates `-dmin * c` as one negated product
            // (FNMUL); staging `dmin * c` with an integer sign flip reproduces
            // it bit for bit, including the sign of an inf * 0 default NaN.
            return {.primary = Ops::stagedProduct(primary),
                    .secondary = Ops::stagedNegatedProduct(base.dmin * L::toFloat((code >> 6) & 63))};
        }
        else
        {
            const auto minimum = -base.dmin * L::toFloat((code >> 6) & 63);
            return {.primary = L::staged(primary), .secondary = L::staged(minimum)};
        }
    }

    /**
     * @brief Multiply the original base by the signed source coefficients without half rounding.
     * @tparam Codebook Physical source-scale codebook: Q6_K or IQ4_XS.
     * @tparam Ops SIMD lane and half-conversion implementation.
     * @param code Packed signed-byte pairs (Q6_K) or six-bit ls (IQ4_XS).
     * @param d Original FP16 bases widened to FP32.
     * @return Effective FP32 factors for the two Q6_K halves or the entire IQ4_XS block.
     */
    template <std::uint8_t Codebook, typename Ops>
    inline Factors<typename Ops::F> sourceScaleFactors(typename Ops::I code, typename Ops::F d)
    {
        using L = Lanes<Ops>;
        if constexpr (Codebook == 8)
            return {.primary = d * L::toFloat((code & 127) - (code & 128)),
                    .secondary = d * L::toFloat(((code >> 8) & 127) - ((code >> 8) & 128))};
        else
        {
            static_assert(Codebook == kNativeQuantisedIQ4XSSourceScaleCodebook);
            return {.primary = d * L::toFloat(code - 32)};
        }
    }

    /**
     * @brief Reconstruct one block's factors for kLanes columns.
     * @tparam Codebook Compact codable codebook other than the source-scale IQ2_S.
     * @param code Codes of the columns.
     * @param word Superblock words of the columns.
     * @return Exactly the widened staged words of decodeSuperblockMetadata().
     */
    template <std::uint8_t Codebook, typename Ops>
    inline Factors<typename Ops::F> reconstruct(typename Ops::I code, typename Ops::I word)
    {
        using L = Lanes<Ops>;
        using F = typename Ops::F;
        Factors<F> out;
        const F d = L::half(word);
        if constexpr (Codebook == 15)
            out.primary = L::staged((d * (0.5f + L::toFloat(code & 15))) * 0.25f);
        else if constexpr (Codebook == 14)
        {
            out.primary = L::staged((d * (0.5f + L::toFloat(code & 15))) * 0.25f);
            out.secondary = L::staged((d * (0.5f + L::toFloat((code >> 4) & 15))) * 0.25f);
        }
        else if constexpr (Codebook == 12)
            out.primary = L::staged((d * (0.5f + L::toFloat(code & 15))) * 0.5f);
        else if constexpr (Codebook == 16)
        {
            const F dl = d * (2.0f * L::toFloat(code & 7) + 1.0f);
            // Code bit 3 is the delta sign: +-0.125 (0x3e000000 with the sign bit).
            using I = typename Ops::I;
            const F delta = std::bit_cast<F>((I{} + 0x3e000000) ^ (((code >> 3) & 1) << 31));
            out.primary = L::staged(dl);
            out.secondary = L::staged(dl * delta);
        }
        else if constexpr (Codebook == 17)
        {
            out.primary = L::staged(d * (2.0f * L::toFloat(code & 7) + 1.0f));
            out.secondary = L::staged(d * (2.0f * L::toFloat((code >> 3) & 7) + 1.0f));
        }
        else if constexpr (Codebook == 9)
        {
            out.primary = L::staged(d * L::toFloat((code & 63) - 32));
            out.secondary = L::staged(d * L::toFloat(((code >> 6) & 63) - 32));
        }
        else if constexpr (Codebook == 10)
        {
            const F dmin = L::half(word >> 16);
            out.primary = L::staged(d * L::toFloat(code & 15));
            out.secondary = L::staged(d * L::toFloat((code >> 8) & 15));
            out.emin_low = L::staged(-dmin * L::toFloat((code >> 4) & 15));
            out.emin_high = L::staged(-dmin * L::toFloat((code >> 12) & 15));
        }
        else if constexpr (Codebook == 5)
            return reconstruct<5, Ops>(code, q4KBase<Ops>(word));
        else if constexpr (Codebook == 7)
        {
            // Q5_K: one 6-bit scale and one 6-bit additive minimum per block.
            const F dmin = L::half(word >> 16);
            out.primary = L::staged(d * L::toFloat(code & 63));
            out.secondary = L::staged(-dmin * L::toFloat((code >> 6) & 63));
        }
        else
            static_assert(Codebook == 15 && Codebook != 15, "reconstruct() covers the derived-scale compact codebooks");
        return out;
    }
} // namespace llaminar2::cpu::native_quantised::coded_metadata

namespace llaminar2::cpu::native_quantised
{
    /**
     * @brief Gather one column's payload bytes from a coded compact block.
     * @param block First byte of the 64-column block.
     * @param codebook Compact codebook.
     * @param column Column 0..63.
     * @param payload Receives preparedCompactPayloadBytes() bytes (code-in-payload layout).
     */
    inline void compactColumnPayload(const std::uint8_t *block, std::uint8_t codebook, int column, std::uint8_t *payload)
    {
        if (codebook == 7)
        {
            for (int word = 0; word < 4; ++word)
                std::memcpy(payload + 4 * word, block + 256 * word + 4 * column, 4);
            coded_metadata::loadQ5HighBits(block + 1024, column, payload + 16);
            return;
        }
        const int words = int(preparedCompactPayloadWords(codebook));
        const int tail = int(preparedCompactPayloadTailBytes(codebook));
        for (int byte = 0; byte < 4 * words; ++byte)
            payload[byte] = block[(byte / 4) * 256 + column * 4 + byte % 4];
        for (int byte = 0; byte < tail; ++byte)
            payload[4 * words + byte] = block[words * 256 + column * tail + byte];
    }

} // namespace llaminar2::cpu::native_quantised

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>

namespace llaminar2::cpu::native_quantised::coded_metadata::neon_half
{
    /**
     * @brief Q4_K superblock factors of four columns as one binary16 vector.
     * @details Lanes 0-3 hold the four columns' source `d`, lanes 4-7 their
     * source `dmin`. One vector serves every block of the superblock, so the
     * per-block work is a single binary16 multiply for the scale and the
     * minimum together; q4KFactors() negates the minimum product.
     */
    struct Q4KHalfBase
    {
        float16x8_t factors; ///< {d0..d3, dmin0..dmin3}.
    };

    /**
     * @brief Load the Q4_K header words (d | dmin << 16) of four columns.
     * @param header First byte of the 64-column superblock header (4-byte aligned).
     * @param column First of four columns, a multiple of four.
     * @return Binary16 base factors for q4KFactors().
     */
    [[gnu::always_inline]] inline Q4KHalfBase q4KHalfBase(const std::uint8_t *header, int column) noexcept
    {
        // De-interleave {d, dmin} pairs: val[0] = d, val[1] = dmin.
        const auto words = vld2_u16(reinterpret_cast<const std::uint16_t *>(header + 4 * column));
        return {vreinterpretq_f16_u16(vcombine_u16(words.val[0], words.val[1]))};
    }

    /**
     * @brief Reconstruct one block's staged Q4_K scale and minimum for four columns.
     * @param codes First byte of the block's 12-bit code planes (u8[64], then a nibble plane).
     * @param column First of four columns, a multiple of four.
     * @param base Superblock factors from q4KHalfBase().
     * @param scale Receives the widened staged scale words.
     * @param minimum Receives the widened staged (negative) minimum words.
     *
     * @details Bit-identical (raw bits, NaN sign and payload included) to
     * widening the codec's staged words under the admitted FPCR profile
     * (round-to-nearest-even, FZ16 = 0, AHP = 0; required by every projection
     * entry through cpu::ARMFloatingPointProfile): the codec forms the exact
     * binary32 product `d * c` (an 11-bit significand times a 6-bit integer
     * is exact in binary32) and rounds it once to binary16 with
     * round-to-nearest-even. A binary16 multiply returns that same correctly
     * rounded product, including overflow to infinity, subnormal results and
     * NaN propagation (a signalling `d` is quieted with its payload, as the
     * binary32 product quiets it). The AArch64 codec evaluates `-dmin * c` as
     * one negated product (FNMUL), so the minimum lanes' sign bit is flipped
     * after the multiply as an integer (also the sign of an inf * 0 default
     * NaN). The publication's -0 -> +0 canonicalization is an integer select
     * on the half bits, independent of the rounding mode. The 12-bit code is
     * gathered with one two-register table lookup: lanes 0-3 take the low
     * code byte (scale bits 0-5), lanes 4-7 the low byte below the column's
     * high nibble so one shift by six yields minimum bits 0-5.
     */
    [[gnu::always_inline]] inline void q4KFactors(const std::uint8_t *codes, int column, const Q4KHalfBase &base,
        float32x4_t &scale, float32x4_t &minimum) noexcept
    {
        // Bytes 0-3: low code bytes of the four columns; bytes 4-5: their two
        // high-nibble bytes (columns c, c+1 in byte 4; c+2, c+3 in byte 5).
        auto raw = vreinterpretq_u8_u32(vld1q_dup_u32(reinterpret_cast<const std::uint32_t *>(codes + column)));
        raw = vreinterpretq_u8_u16(vld1q_lane_u16(reinterpret_cast<const std::uint16_t *>(codes + 64 + column / 2),
                                                  vreinterpretq_u16_u8(raw), 2));
        const uint8x16x2_t table{{raw, vshrq_n_u8(raw, 4)}};
        // Halfword lanes: {b0, b1, b2, b3, b0 | n0 << 8, b1 | (n0 >> 4) << 8, b2 | n1 << 8, b3 | (n1 >> 4) << 8}.
        const uint8x16_t gather{0, 255, 1, 255, 2, 255, 3, 255, 0, 4, 1, 20, 2, 5, 3, 21};
        const auto words = vreinterpretq_u16_u8(vqtbl2q_u8(table, gather));
        const int16x8_t shift{0, 0, 0, 0, -6, -6, -6, -6};
        const auto code = vandq_u16(vshlq_u16(words, shift), vdupq_n_u16(63));
        const uint16x8_t minimum_sign{0, 0, 0, 0, 0x8000, 0x8000, 0x8000, 0x8000};
        auto staged = veorq_u16(vreinterpretq_u16_f16(vmulq_f16(vcvtq_f16_u16(code), base.factors)), minimum_sign);
        staged = vbicq_u16(staged, vceqq_u16(staged, vdupq_n_u16(0x8000)));
        scale = vcvt_f32_f16(vget_low_f16(vreinterpretq_f16_u16(staged)));
        minimum = vcvt_high_f32_f16(vreinterpretq_f16_u16(staged));
    }
} // namespace llaminar2::cpu::native_quantised::coded_metadata::neon_half
#endif
