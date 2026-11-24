/**
 * @file CPUNativeQuantisedMultiScaleKernels.h
 * @brief Lossless compact multi-scale CPU projections with shared serial/grouped arithmetic.
 *
 * Prepared blocks carry exact source codes and use one base/minimum word per
 * column and source superblock. Q5_K retains both its five-bit unsigned values
 * and additive minimum. IQ3_S decodes its signed grid in registers and
 * derives its coefficient from the source nibble and header d. SIMD
 * lanes represent output columns, with decoded integer values shared by rows.
 * Q5_K and IQ3_S use the same FP32 contribution policy as their former
 * expanded units; other compact codebooks keep the CUDA/HIP-ordered program.
 * The same primitive serves serial decode and grouped verification; no row
 * replay, temporary decoded matrix or inference-time allocation is used.
 * AArch64's four-column native operations use the same decoder and rounded
 * source-factor program as x86. Borrowed read views keep the owning storage
 * dependency out of native leaves, and bounded stores support logical N tails.
 */
#pragma once

#include <array>
#include <algorithm>
#include <bit>
#include <cstring>
#include <stdexcept>

#include "CPUNativeQuantisedDecode.h"
#include "CPUNativeQuantisedFP16.h"
#include "CPUNativeQuantisedActivationSums.h"
#include "CPUNativeQuantisedContributionContract.h"
#include "CPUNativeQuantisedCodedMetadata.h"
#include "CPUNativeQuantisedPreparedFootprint.h"
#if defined(__aarch64__) && defined(__ARM_NEON)
#include "CPUNativeNEONMultiScaleOps.h"
#else
#include <immintrin.h>
#include "VNNIEmulation.h"
#endif

namespace llaminar2::cpu::native_quantised::multi_scale
{
    /** @brief Arithmetic ownership of a complete column tile's K traversal. */
    enum class Reduction
    {
        Contiguous,        ///< One accumulator visits the requested K interval.
        OrderedPartitions, ///< Positive-zero partials folded in ascending order.
    };

    /** @return Whether an immutable signed grid fits the AVX2 sign/maddubs domain. */
    template <size_t N> constexpr bool gridFitsSignedDot(const uint64_t (&table)[N])
    {
        for (uint64_t word : table)
            for (int byte = 0; byte < 8; ++byte)
                if (static_cast<uint8_t>(word >> (byte * 8)) == 128)
                    return false;
        return true;
    }
    static_assert(gridFitsSignedDot(iq2s_grid) && gridFitsSignedDot(iq2xs_grid) &&
                  gridFitsSignedDot(iq1s_grid) && gridFitsSignedDot(iq3xxs_grid64),
                  "Native grids must exclude the non-negatable INT8 minimum");

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_MULTISCALE_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_MULTISCALE_EXACT
#endif

    /** @brief ISA operations; the decoded arithmetic is shared by all three widths. */
    template <int Lanes> struct SIMD;

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    /**
     * @brief Interleave each packed byte's low and high nibble as adjacent bytes.
     * @param packed Low 4 or 8 bytes hold the source nibbles, in column order.
     * @return The low 8 or 16 bytes hold the unpacked nibbles, in column order.
     *
     * Unpacking before widening avoids scalar byte assembly on GCC x86.
     * Padding in the unused upper bytes is never read by a caller.
     */
    static inline __m128i x86UnpackNibbles(__m128i packed)
    {
        return _mm_and_si128(_mm_unpacklo_epi8(packed, _mm_srli_epi16(packed, 4)), _mm_set1_epi8(15));
    }

    /** @brief Eight-column AVX2 implementation, including saturation-safe VNNI. */
    template <> struct SIMD<8>
    {
        using I = int __attribute__((vector_size(32)));
        using F = float __attribute__((vector_size(32)));
        /**
         * @return Exact signed native-grid dot accumulation.
         *
         * Every grid in this family lies in [-127,127]. Moving the activation
         * sign onto the weight lets maddubs consume abs(activation), including
         * -128 as unsigned 128. A pair is bounded by 2*128*127 = 32512, so its
         * saturating INT16 intermediate cannot saturate. This avoids generic
         * full-uint8 VNNI emulation and its extra masks/register pressure.
         */
        static I dot(I acc, I weights, I activation)
        {
            const auto a = std::bit_cast<__m256i>(activation);
            const auto w = _mm256_sign_epi8(std::bit_cast<__m256i>(weights), a);
            const auto pairs = _mm256_maddubs_epi16(_mm256_abs_epi8(a), w);
            return std::bit_cast<I>(_mm256_add_epi32(
                std::bit_cast<__m256i>(acc),
                _mm256_madd_epi16(pairs, _mm256_set1_epi16(1))));
        }
        /** @brief Spread eight columns' packed Q5_K high bits directly into the dot byte lanes. */
        template <bool ScalarSpread> static I q5High(const std::uint8_t *source)
        {
            std::uint32_t packed;
            std::memcpy(&packed, source, sizeof(packed));
            if constexpr (ScalarSpread)
            {
                const auto spread = [](std::uint64_t bits) {
                    bits = (bits | (bits << 28)) & 0x0000000f0000000full;
                    bits = (bits | (bits << 14)) & 0x0003000300030003ull;
                    bits = (bits | (bits << 7)) & 0x0101010101010101ull;
                    return static_cast<long long>(bits << 4);
                };
                return std::bit_cast<I>(_mm256_setr_epi64x(spread(packed & 255), spread((packed >> 8) & 255),
                    spread((packed >> 16) & 255), spread(packed >> 24)));
            }
            else
            {
                const auto qh = _mm256_set1_epi32(static_cast<int>(packed));
                const auto shuffle = _mm256_setr_epi32(0, 0, 0x01010101, 0x01010101,
                    0x02020202, 0x02020202, 0x03030303, 0x03030303);
                const auto masks = _mm256_setr_epi32(0x08040201, 0x80402010, 0x08040201, 0x80402010,
                    0x08040201, 0x80402010, 0x08040201, 0x80402010);
                const auto set = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(qh, shuffle), masks), masks);
                return std::bit_cast<I>(_mm256_and_si256(set, _mm256_set1_epi8(16)));
            }
        }
        /** @brief Dot unsigned five-bit weights against signed Q8 without saturation. */
        static I dotUnsignedQ5(I acc, I weights, I activation)
        {
            const auto pairs = _mm256_maddubs_epi16(std::bit_cast<__m256i>(weights), std::bit_cast<__m256i>(activation));
            return std::bit_cast<I>(_mm256_add_epi32(std::bit_cast<__m256i>(acc),
                _mm256_madd_epi16(pairs, _mm256_set1_epi16(1))));
        }
        /** @return Bytewise difference without cross-byte borrow. */
        static I subtractBytes(I a, I b)
        {
            return std::bit_cast<I>(_mm256_sub_epi8(std::bit_cast<__m256i>(a), std::bit_cast<__m256i>(b)));
        }
        /** @return Selected four-byte halves of immutable eight-byte grid entries. */
        static inline __attribute__((always_inline)) I grid(const uint64_t *table, I index, int half)
        {
            return std::bit_cast<I>(_mm256_i32gather_epi32(
                reinterpret_cast<const int *>(table), std::bit_cast<__m256i>(index * 2 + half), 4));
        }
        /** @return One selected metadata byte repeated within each column's word. */
        template <int Byte> static I repeatByte(I metadata)
        {
            const __m256i indices = _mm256_setr_epi32(
                0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c,
                0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c);
            return std::bit_cast<I>(_mm256_shuffle_epi8(
                _mm256_srli_epi32(std::bit_cast<__m256i>(metadata), Byte * 8), indices));
        }
        /** @return Signed grid bytes selected by one half of the replicated sign byte. */
        template <int Half> static I signGrid(I values, I signs)
        {
            const auto bits = _mm256_set1_epi32(0x08040201 << (Half * 4));
            const auto positive = _mm256_cmpeq_epi8(
                _mm256_and_si256(std::bit_cast<__m256i>(signs), bits), _mm256_setzero_si256());
            const auto negative = _mm256_xor_si256(positive, _mm256_set1_epi8(-1));
            return std::bit_cast<I>(_mm256_sub_epi8(
                _mm256_xor_si256(std::bit_cast<__m256i>(values), negative), negative));
        }
        /** @brief Load both four-byte halves with the AVX2 gather geometry. */
        static inline __attribute__((always_inline)) void gridPair(const uint64_t *table, I index, I &low, I &high)
        {
            low = grid(table, index, 0);
            high = grid(table, index, 1);
        }
        /** @brief Gather four source grid values for each output column. */
        static I grid4(const std::uint32_t *table, I index)
        {
            return std::bit_cast<I>(_mm256_i32gather_epi32(
                reinterpret_cast<const int *>(table), std::bit_cast<__m256i>(index), 4));
        }
        /** @param source Four readable bytes of column-ordered nibbles.
         * @return Eight adjacent column nibbles widened to INT32.
         */
        static I unpackNibbles(const uint8_t *source)
        {
            return std::bit_cast<I>(_mm256_cvtepu8_epi32(x86UnpackNibbles(_mm_loadu_si32(source))));
        }
        /** @param value Eight FP32 products before binary16 staging.
         * @return Eight staged values; canonicalize negative zero in the narrow FP16 register.
         */
        static F stagedHalf(F value)
        {
            const auto words = _mm256_cvtps_ph(std::bit_cast<__m256>(value),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const auto nonzero = _mm_andnot_si128(_mm_cmpeq_epi16(words, _mm_set1_epi16(-32768)), words);
            return std::bit_cast<F>(_mm256_cvtph_ps(nonzero));
        }
        /** @return Eight exact FP16-to-FP32 conversions. */
        static F half(const uint16_t *source)
        {
            return std::bit_cast<F>(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source))));
        }
        /**
         * @brief Widen opaque metadata without FP16 interpretation.
         * @param source Eight readable unsigned metadata words.
         * @return One zero-extended word per SIMD column.
         */
        static I words(const uint16_t *source)
        { return std::bit_cast<I>(_mm256_cvtepu16_epi32(_mm_loadu_si128(
            reinterpret_cast<const __m128i *>(source)))); }
        /** @brief Reject reserved source coefficient bits across eight columns.
         * @param metadata Zero-extended original source words.
         * @param permitted_bits Source-owned coefficient bit positions.
         * @return Whether all words obey the source metadata ABI.
         */
        static bool metadataFits(I metadata, uint32_t permitted_bits)
        {
            return _mm256_testz_si256(std::bit_cast<__m256i>(metadata),
                _mm256_set1_epi32(static_cast<int>(~permitted_bits))) != 0;
        }
        /** @return FP32 values from the low FP16 half of each packed word. */
        static F halfWords(I words)
        {
            const __m256i bits = std::bit_cast<__m256i>(words & 65535);
            return std::bit_cast<F>(_mm256_cvtph_ps(_mm_packus_epi32(
                _mm256_castsi256_si128(bits), _mm256_extracti128_si256(bits, 1))));
        }
    };

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /** @brief Sixteen-column native AVX-512 VNNI implementation. */
    template <> struct SIMD<16>
    {
        using I = int __attribute__((vector_size(64)));
        using F = float __attribute__((vector_size(64)));
        /** @return Exact unsigned-byte times signed-byte dot accumulation. */
        static I dot(I acc, I weights, I activation)
        {
            return std::bit_cast<I>(_mm512_dpbusd_epi32(
                std::bit_cast<__m512i>(acc), std::bit_cast<__m512i>(weights),
                std::bit_cast<__m512i>(activation)));
        }
        /** @return Bytewise difference without cross-byte borrow. */
        static I subtractBytes(I a, I b)
        {
            return std::bit_cast<I>(_mm512_sub_epi8(std::bit_cast<__m512i>(a), std::bit_cast<__m512i>(b)));
        }
        /** @return One metadata byte per column, replicated for its four dot bytes. */
        template <int Byte> static I repeatByte(I metadata)
        {
            const __m512i indices = _mm512_setr_epi32(
                0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c,
                0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c,
                0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c,
                0x00000000, 0x04040404, 0x08080808, 0x0c0c0c0c);
            return std::bit_cast<I>(_mm512_shuffle_epi8(
                _mm512_srli_epi32(std::bit_cast<__m512i>(metadata), Byte * 8), indices));
        }
        /**
         * @return Grid bytes negated exactly where their packed sign bit is set.
         *
         * Byte tests directly produce the AVX-512 predicate. This replaces
         * multiplication-based bit spreading and its long shift/add chain.
         * These codebook grids exclude -128, so negation preserves every byte.
         */
        template <int Half> static I signGrid(I values, I signs)
        {
            const auto bits = _mm512_set1_epi32(0x08040201 << (Half * 4));
            const auto negative = _mm512_test_epi8_mask(std::bit_cast<__m512i>(signs), bits);
            return std::bit_cast<I>(_mm512_mask_sub_epi8(
                std::bit_cast<__m512i>(values), negative,
                _mm512_setzero_si512(), std::bit_cast<__m512i>(values)));
        }
        /**
         * @brief Fetch each complete eight-byte grid entry once for sixteen columns.
         * @param table Immutable codebook's eight-byte entries.
         * @param index One entry index per output column.
         * @param low Receives values K0..3 for every column.
         * @param high Receives values K4..7 for every column.
         *
         * Two eight-lane QWORD gathers replace two sixteen-lane DWORD gathers.
         * This halves the number of indexed loads, while the permutations only
         * rearrange integer bytes. No decoded-weight cache or wider prepared
         * representation is created.
         */
        static inline __attribute__((always_inline)) void gridPair(const uint64_t *table, I index, I &low, I &high)
        {
            const __m512i indices = std::bit_cast<__m512i>(index);
            const __m512i first = _mm512_i32gather_epi64(
                _mm512_castsi512_si256(indices), table, 8);
            const __m512i second = _mm512_i32gather_epi64(
                _mm512_extracti64x4_epi64(indices, 1), table, 8);
            const __m512i even = _mm512_setr_epi32(
                0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
            const __m512i odd = _mm512_setr_epi32(
                1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
            low = std::bit_cast<I>(_mm512_permutex2var_epi32(first, even, second));
            high = std::bit_cast<I>(_mm512_permutex2var_epi32(first, odd, second));
        }
        /** @brief Gather four source grid values for each output column. */
        static I grid4(const std::uint32_t *table, I index)
        {
            return std::bit_cast<I>(_mm512_i32gather_epi32(
                std::bit_cast<__m512i>(index), table, 4));
        }
        /** @param source Eight readable bytes of column-ordered nibbles.
         * @return Sixteen adjacent column nibbles widened to INT32.
         */
        static I unpackNibbles(const uint8_t *source)
        {
            return std::bit_cast<I>(_mm512_cvtepu8_epi32(x86UnpackNibbles(_mm_loadu_si64(source))));
        }
        /**
         * @brief Insert sixteen columns' fifth bits directly into their VNNI bytes.
         * @param low Four low-nibble weights per INT32 lane.
         * @param source Eight readable bytes, one bit per weight byte in lane order.
         * @return Exact unsigned five-bit weights in VNNI byte order.
         */
        static I insertQ5High(I low, const uint8_t *source)
        {
            uint64_t bits;
            std::memcpy(&bits, source, sizeof(bits));
            const auto values = std::bit_cast<__m512i>(low);
            return std::bit_cast<I>(_mm512_mask_add_epi8(values, static_cast<__mmask64>(bits),
                values, _mm512_set1_epi8(16)));
        }
        /** @param value Sixteen FP32 products before binary16 staging.
         * @return Sixteen staged values; canonicalize negative zero in the narrow FP16 register.
         */
        static F stagedHalf(F value)
        {
            const auto words = _mm512_cvtps_ph(std::bit_cast<__m512>(value),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            const auto keep = _mm256_cmpneq_epi16_mask(words, _mm256_set1_epi16(-32768));
            return std::bit_cast<F>(_mm512_maskz_cvtph_ps(keep, words));
        }
        /** @return Sixteen exact FP16-to-FP32 conversions. */
        static F half(const uint16_t *source)
        {
            return std::bit_cast<F>(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(source))));
        }
        /**
         * @brief Widen opaque metadata without FP16 interpretation.
         * @param source Sixteen readable unsigned metadata words.
         * @return One zero-extended word per SIMD column.
         */
        static I words(const uint16_t *source)
        { return std::bit_cast<I>(_mm512_cvtepu16_epi32(_mm256_loadu_si256(
            reinterpret_cast<const __m256i *>(source)))); }
        /** @brief Reject reserved source coefficient bits across sixteen columns.
         * @param metadata Zero-extended original source words.
         * @param permitted_bits Source-owned coefficient bit positions.
         * @return Whether all words obey the source metadata ABI.
         */
        static bool metadataFits(I metadata, uint32_t permitted_bits)
        {
            return _mm512_test_epi32_mask(std::bit_cast<__m512i>(metadata),
                _mm512_set1_epi32(static_cast<int>(~permitted_bits))) == 0;
        }
        /** @return FP32 values from the low FP16 half of each packed word. */
        static F halfWords(I words)
        {
            return std::bit_cast<F>(_mm512_cvtph_ps(
                _mm512_cvtepi32_epi16(std::bit_cast<__m512i>(words))));
        }
    };
#endif

#endif // native x86 SIMD operations

    /**
     * @brief Whether an ISA operation set consumes shared-row tiles in row pairs.
     * @tparam Ops Native operation set; only the AArch64 I8MM family declares row pairs.
     */
    template <typename Ops> inline constexpr bool usesRowPairs = requires { requires Ops::row_pairs; };

    /** @brief Load a SIMD value without imposing aliasing or alignment on the caller. */
    template <typename V> inline V load(const void *source)
    {
        V value;
        std::memcpy(&value, source, sizeof(value));
        return value;
    }

    /**
     * @brief Load payload word @p offset of a coded compact block for SIMD columns.
     * @tparam Raw True to return the stored bytes; false restores the sign-selector
     *         parity bits that code-in-payload formats repurpose for their code.
     * @param unit First byte of the 64-column block.
     * @param offset Payload word index (bytes 4*offset .. 4*offset+3).
     * @param column First SIMD column.
     *
     * Whole words live at `offset * 256 + column * 4`; the 1-2 byte tail of
     * IQ2_S/IQ2_XS/IQ1_S/IQ1_M is a `[column][tail]` plane loaded
     * zero-extended, which is exactly the word the per-block layout held.
     */
    template <int Codebook, typename Ops, bool Raw = false>
    [[gnu::always_inline]] inline typename Ops::I payloadWord(const uint8_t *unit, int offset, int column)
    {
        using I = typename Ops::I;
        using L = coded_metadata::Lanes<Ops>;
        constexpr int words = static_cast<int>(preparedCompactPayloadWords(Codebook));
        constexpr int tail = static_cast<int>(preparedCompactPayloadTailBytes(Codebook));
        I word{};
        if (offset < words)
            word = load<I>(unit + offset * 256 + column * 4);
        else if constexpr (tail == 1)
            word = L::bytes(unit + words * 256 + column);
        else if constexpr (tail == 2)
            word = L::halfwords(unit + words * 256 + column * 2);
        if constexpr (!Raw)
        {
            // Sign selectors carry the code in their parity bits (IQ3_XXS word
            // 2, IQ2_XXS word 1, IQ2_XS words 1 and 2). Restoring parity on the
            // IQ2_XS qh byte only sets its unused bit 7.
            const bool signs = (Codebook == 12 && offset == 2) || (Codebook == 15 && offset == 1) ||
                               (Codebook == 14 && offset >= 1);
            if (signs)
                word = L::restoreSignParity(word);
        }
        return word;
    }

    /**
     * @brief Codes of SIMD columns of one coded compact block.
     * @param unit First byte of the 64-column block.
     * @param data_stride Payload bytes of the block (code planes follow).
     * @param column First SIMD column.
     */
    template <int Codebook, typename Ops>
    [[gnu::always_inline]] inline typename Ops::I compactCode(const uint8_t *unit, int data_stride, int column)
    {
        using I = typename Ops::I;
        using L = coded_metadata::Lanes<Ops>;
        if constexpr (Codebook == 16)
            return (payloadWord<Codebook, Ops, true>(unit, 1, column) >> 12) & 15;
        else if constexpr (Codebook == 12 || Codebook == 15)
        {
            const I word = payloadWord<Codebook, Ops, true>(unit, Codebook == 12 ? 2 : 1, column);
            return ((word >> 7) & 1) | ((word >> 14) & 2) | ((word >> 21) & 4) | ((word >> 28) & 8);
        }
        else if constexpr (Codebook == 14)
        {
            // Sign bytes 5..8: word 1 bytes 1..3 and the tail byte; qh (word 1 byte 0) high nibble.
            const I word = payloadWord<Codebook, Ops, true>(unit, 1, column);
            const I tail = payloadWord<Codebook, Ops, true>(unit, 2, column);
            return ((word >> 15) & 1) | ((word >> 22) & 2) | ((word >> 29) & 4) | ((tail >> 4) & 8) |
                   (word & 0xF0);
        }
        else
            return L::template code<static_cast<std::uint8_t>(Codebook)>(unit + data_stride, column);
    }

    /** @brief Reconstruct IQ3_S's FP32 factor without a half round trip. */
    template <typename Ops>
    [[gnu::always_inline]] inline typename Ops::F iq3sSourceScale(typename Ops::I code, typename Ops::I word)
    {
        using L = coded_metadata::Lanes<Ops>;
        return L::half(word) * L::toFloat(2 * (code & 15) + 1);
    }

    /**
     * @brief Whether a compact codebook carries one weight scale per 32-value block.
     *
     * Q5_K, IQ3_S, IQ3_XXS, IQ2_XXS and IQ1_S apply one scale to the whole
     * 32-value dot. Q5_K and IQ1_S also apply a whole-block minimum.
     */
    constexpr bool isSingleScaleCompactCodebook(int codebook) noexcept
    {
        return codebook == 7 || codebook == 11 || codebook == 12 || codebook == 15 || codebook == 16;
    }

    /**
     * @brief Scalar diagnostic oracle for one multi-scale block, never a production dispatch.
     * @param packed Native immutable representation.
     * @param a Quantized activation block.
     * @param chunk Output-column chunk.
     * @param kb Logical K block.
     * @param column Local output column.
     * @return GPU-ordered FP32 block contribution, including required corrections.
     */
    template <typename Packed>
    LLAMINAR_MULTISCALE_EXACT
    inline float scalarReferenceBlock(const Packed &packed,
                                     const Q8_1Block &a, int chunk, int kb, int column)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        if (!packed.usesCompactMultiScale() || packed.payload_bytes <= 0 || packed.payload_bytes > 20)
            throw std::invalid_argument("Multi-scale diagnostic requires its bounded compact source payload");
        uint8_t payload[20]{};
        int8_t values[32];
        compactColumnPayload(packed.blockUnit(chunk, kb), packed.codebook_id, column, payload);
        native_quantised_superblock::restoreCodedPayload(packed.codebook_id, payload);
        decode_native_block(packed.codebook_id, payload, values);
        int dots[2]{}, sums[4]{};
        for (int k = 0; k < 32; ++k)
        {
            dots[k / 16] += static_cast<int>(values[k]) * a.qs[k];
            sums[k / 8] += a.qs[k];
        }
        const float scale0 = packed.blockScale(chunk, kb, column);
        const float scale1 = packed.blockMin(chunk, kb, column);
        const float activation_scale = nativeQuantisedFP16ScaleToFP32(a.d);
        if (isSingleScaleCompactCodebook(packed.codebook_id))
        {
            // GPU singleScaleBlock: (activation * weight scale) * dot, plus the
            // IQ1_S/Q5_K additive minimum through singleScaleCorrection.
            float contribution = (activation_scale * scale0) * static_cast<float>(dots[0] + dots[1]);
            if (packed.codebook_id == 16 || packed.codebook_id == 7)
                contribution += (activation_scale * scale1) *
                                static_cast<float>(sums[0] + sums[1] + sums[2] + sums[3]);
            return contribution;
        }
        float contribution = activation_scale * (scale0 * dots[0] + scale1 * dots[1]);
        if (packed.codebook_id == 10)
        {
            const uint32_t bits = packed.codedBlockMetadata(chunk, kb, column).emin;
            const float min0 = fp16_to_fp32(static_cast<uint16_t>(bits));
            const float min1 = fp16_to_fp32(static_cast<uint16_t>(bits >> 16));
            contribution += activation_scale * (min0 * (sums[0] + sums[1]) + min1 * (sums[2] + sums[3]));
        }
        if (packed.codebook_id == 17)
        {
            const float delta0 = (payload[4] & 8) ? -0.125f : 0.125f;
            const float delta1 = (payload[4] & 128) ? -0.125f : 0.125f;
            const float delta2 = (payload[5] & 8) ? -0.125f : 0.125f;
            const float delta3 = (payload[5] & 128) ? -0.125f : 0.125f;
            contribution += activation_scale * (
                (delta0 * sums[0] + delta1 * sums[1]) * scale0 +
                (delta2 * sums[2] + delta3 * sums[3]) * scale1);
        }
        return contribution;
    }

    /** @brief Decode one IQ3_S four-value group per SIMD column from the source grid and sign bits.
     * @param unit Resident block's three word planes followed by one byte-tail plane.
     * @param column First SIMD column in the 64-column block.
     * @param group Original four-value group in [0,7].
     * @return Signed dot operands, biased for unsigned AVX-512 VNNI.
     */
    template <int Lanes, typename Ops>
    [[gnu::always_inline]] inline typename Ops::I decodeIQ3SGroup(const std::uint8_t *unit, int column, int group)
    {
        const auto metadata = payloadWord<11, Ops>(unit, 2, column);
        const auto qs = payloadWord<11, Ops>(unit, group / 4, column);
        const auto index = ((qs >> (8 * (group % 4))) & 255) | (((metadata >> group) & 1) << 8);
        const auto signs = group >= 6 ? payloadWord<11, Ops>(unit, 3, column)
                                      : metadata >> (8 * (1 + group / 2));
        const auto masks = Ops::template repeatByte<0>(signs >> (4 * (group % 2)));
        auto weights = Ops::template signGrid<0>(Ops::grid4(iq3s_grid, index), masks);
        if constexpr (Lanes == 16)
            weights ^= static_cast<int>(0x80808080u);
        return weights;
    }

    /**
     * @brief Decode one linear two/three-bit four-K group for SIMD columns.
     * @tparam Codebook Q3_K (9) or Q2_K (10) execution format.
     * @tparam Group Four-value group in [0,7].
     * @tparam Lanes Four AArch64, eight AVX2 or sixteen AVX-512 columns.
     * @param unit Native CPU unit's first payload byte.
     * @param column First SIMD column within the 64-column unit.
     * @return Signed native grids on AVX2; unsigned-biased grids for AVX-512 VNNI.
     */
    template <int Codebook, int Group, int Lanes, typename Ops = SIMD<Lanes>>
    inline typename Ops::I decodeLinear(const uint8_t *unit, int column)
    {
        static_assert(Codebook == 9 || Codebook == 10);
        using I = typename Ops::I;
        const auto word = [&](int index) { return payloadWord<Codebook, Ops>(unit, index, column); };
        I values = (word(Group / 4) >> ((Group % 4) * 2)) & 0x03030303;
        if constexpr (Codebook == 9)
        {
            // Four high bits become bit two of four separate byte lanes.
            const I high = (word(2) >> (Group * 4)) & 15;
            values |= ((high * 0x02040810) & 0x10101010) >> 2;
            if constexpr (Lanes != 16)
                values = Ops::subtractBytes(values, I{} + 0x04040404);
        }
        return values;
    }

    /**
     * @brief Apply one half's four groups to row pairs with two-row matrix dots.
     * @tparam Codebook Native source execution codebook.
     * @tparam Half Zero for K0..15 or one for K16..31.
     * @tparam Rows Shared rows in [2,4]; an odd last row keeps the single-row leaf.
     * @param unit Native unit, shared by every row.
     * @param column First SIMD output column.
     * @param activation Persistent quantized rows.
     * @param kb Source block index.
     * @param dots One independent exact integer accumulator per row.
     *
     * Each group pair is decoded exactly as dotHalf decodes it and consumed in
     * one pairDot per row pair. The integer sums cover the same products of
     * the same half, so every row's INT32 result equals the single-row walk.
     */
    template <int Codebook, int Half, int Lanes, size_t Rows, typename Ops>
    inline void dotHalfRowPairs(const uint8_t *unit, int column,
                                const std::array<const Q8_1Block *, Rows> &activation,
                                int kb, typename Ops::I (&dots)[Rows])
    {
        using I = typename Ops::I;
        static_assert(Lanes == 4 && Rows >= 2 && Rows <= 4);
        constexpr size_t pairs = Rows / 2;
        I columns01[pairs]{}, columns23[pairs]{};
        // Consume two adjacent four-value groups starting at activation byte
        // `offset`. The odd last row retains the SDOT leaf in group order.
        const auto consume = [&](I first, I second, int offset) {
#pragma GCC unroll 2
            for (size_t pair = 0; pair < pairs; ++pair)
                Ops::pairDot(columns01[pair], columns23[pair],
                    Ops::activationPair(activation[2 * pair][kb].qs + offset,
                                        activation[2 * pair + 1][kb].qs + offset),
                    first, second);
            if constexpr (Rows % 2 != 0)
            {
                int word0 = 0, word1 = 0;
                std::memcpy(&word0, activation[Rows - 1][kb].qs + offset, 4);
                std::memcpy(&word1, activation[Rows - 1][kb].qs + offset + 4, 4);
                dots[Rows - 1] = Ops::dot(dots[Rows - 1], first, I{} + word0);
                dots[Rows - 1] = Ops::dot(dots[Rows - 1], second, I{} + word1);
            }
        };
        if constexpr (Codebook == 12)
        {
            // IQ3_XXS: eight four-value grid entries per block.  Two adjacent
            // groups share one sign byte (bits 0..3 even, 4..7 odd) and are
            // decoded as one eight-value consume against the mirrored 64-bit
            // table whose upper half is zero.
            const auto quarter = [&]<int Q>() {
                const auto word = [&](int offset) {
                    return payloadWord<Codebook, Ops>(unit, offset, column);
                };
                const I index_even = (word((2 * Q) / 4) >> (8 * ((2 * Q) % 4))) & 255;
                const I index_odd = (word((2 * Q + 1) / 4) >> (8 * ((2 * Q + 1) % 4))) & 255;
                // signGrid tests one bit per value byte, so the shared sign
                // byte must be replicated into every byte of each lane.
                const I signs = Ops::template repeatByte<Q>(word(2));
                I low, high, unused;
                Ops::gridPair(iq3xxs_grid64, index_even, low, unused);
                Ops::gridPair(iq3xxs_grid64, index_odd, high, unused);
                low = Ops::template signGrid<0>(low, signs);
                high = Ops::template signGrid<1>(high, signs);
                consume(low, high, Q * 8);
            };
            quarter.template operator()<Half * 2>();
            quarter.template operator()<Half * 2 + 1>();
        }
        else if constexpr (Codebook == 13 || Codebook == 14 || Codebook == 15 ||
                      Codebook == 16 || Codebook == 17)
        {
            const auto quarter = [&]<int Q>() {
                const auto word = [&](int offset) {
                    return payloadWord<Codebook, Ops>(unit, offset, column);
                };
                I index = (word(0) >> (Q * 8)) & 255;
                if constexpr (Codebook == 16)
                    index |= ((word(1) & 0xFFFF) >> (3 * Q) & 7) << 8;
                else if constexpr (Codebook == 17)
                    index |= ((word(1) >> ((Q / 2) * 8 + (Q % 2) * 4)) & 7) << 8;
                else if constexpr (Codebook != 15)
                {
                    constexpr int bits = Codebook == 13 ? 2 : 1;
                    index |= ((word(1) >> (Q * bits)) & ((1 << bits) - 1)) << 8;
                }
                const uint64_t *table = (Codebook == 17 || Codebook == 16) ? iq1s_grid
                    : Codebook == 13 ? iq2s_grid
                    : Codebook == 15 ? iq2xxs_grid : iq2xs_grid;
                I low, high;
                Ops::gridPair(table, index, low, high);
                if constexpr (Codebook == 13 || Codebook == 14 || Codebook == 15)
                {
                    constexpr int sign_byte = Codebook == 15 ? 4 + Q : 5 + Q;
                    const I signs = Ops::template repeatByte<sign_byte % 4>(word(sign_byte / 4));
                    low = Ops::template signGrid<0>(low, signs);
                    high = Ops::template signGrid<1>(high, signs);
                }
                // One grid entry is eight consecutive K values per column:
                // its low/high halves are exactly one adjacent group pair.
                consume(low, high, Q * 8);
            };
            quarter.template operator()<Half * 2>();
            quarter.template operator()<Half * 2 + 1>();
        }
        else
        {
            const auto group_pair = [&]<int G>() {
                consume(decodeLinear<Codebook, G, Lanes, Ops>(unit, column),
                        decodeLinear<Codebook, G + 1, Lanes, Ops>(unit, column), G * 4);
            };
            group_pair.template operator()<Half * 4>();
            group_pair.template operator()<Half * 4 + 2>();
        }
#pragma GCC unroll 2
        for (size_t pair = 0; pair < pairs; ++pair)
        {
            I row0, row1;
            Ops::splitPair(columns01[pair], columns23[pair], row0, row1);
            dots[2 * pair] += row0;
            dots[2 * pair + 1] += row1;
        }
    }

    /** @brief Accumulate whole-block Q5_K/IQ3_S products while keeping only one decoded group live. */
    template <int Codebook, int Lanes, size_t Rows, Reduction Tree, typename Ops>
    [[gnu::always_inline]] inline void dotSingleScale(const uint8_t *unit, int column,
        const std::array<const Q8_1Block *, Rows> &activation, int kb, typename Ops::I (&dots)[Rows])
    {
        using I = typename Ops::I;
#pragma GCC unroll 1
        for (int group = 0; group < 8; ++group)
        {
            I weights;
            if constexpr (Codebook == 7)
            {
                const I raw = payloadWord<Codebook, Ops>(unit, group % 4, column);
                const I low = (group < 4 ? raw : raw >> 4) & 0x0F0F0F0F;
                if constexpr (Lanes == 8)
                    // Ordered partials keep an additional output vector per row
                    // live. Move the bit spread to integer registers for the
                    // widest tiles so the dot and FP32 state stay in registers.
                    weights = low | Ops::template q5High<Rows == 4 || (Rows == 3 && Tree == Reduction::OrderedPartitions)>(
                        unit + 1024 + 32 * group + column / 2);
                else if constexpr (Lanes == 16)
                    weights = Ops::insertQ5High(low, unit + 1024 + 32 * group + column / 2);
                else
                {
                    const I high = coded_metadata::Lanes<Ops>::nibbles(unit + 1024 + 32 * group, column);
                    weights = low | ((high * 0x02040810) & 0x10101010);
                }
            }
            else
            {
                static_assert(Codebook == 11);
                weights = decodeIQ3SGroup<Lanes, Ops>(unit, column, group);
            }
#pragma GCC unroll 4
            for (size_t row = 0; row < Rows; ++row)
            {
                int activation_word;
                std::memcpy(&activation_word, activation[row][kb].qs + group * 4, 4);
                if constexpr (Codebook == 7 && Lanes == 8)
                    dots[row] = Ops::dotUnsignedQ5(dots[row], weights, I{} + activation_word);
                else
                    dots[row] = Ops::dot(dots[row], weights, I{} + activation_word);
            }
        }
    }

    /**
     * @brief Apply all four groups of a half to every independently accumulated row.
     * @tparam Half Zero for K0..15 or one for K16..31.
     * @param unit Native unit, shared by every row.
     * @param column First SIMD output column.
     * @param activation Persistent quantized rows.
     * @param kb Source block index.
     * @param dots One independent exact integer accumulator per row.
     *
     * The decoded vector dies after reuse by all rows. It is never stored as a
     * temporary matrix; this bounds register pressure and preserves grouped reuse.
     */
    template <int Codebook, int Half, int Lanes, size_t Rows, typename Ops = SIMD<Lanes>>
    inline void dotHalf(const uint8_t *unit, int column,
                        const std::array<const Q8_1Block *, Rows> &activation,
                        int kb, typename Ops::I (&dots)[Rows])
    {
        using I = typename Ops::I;
        if constexpr (usesRowPairs<Ops> && Rows >= 2)
            return dotHalfRowPairs<Codebook, Half, Lanes, Rows, Ops>(unit, column, activation, kb, dots);
        else if constexpr (Codebook == 12)
        {
            // IQ3_XXS: pair two adjacent four-value groups per consume; the
            // per-half consume applies the matching sign half and the offset
            // encoding owned by the arithmetic family.
            const auto quarter = [&]<int Q>() {
                const auto word = [&](int offset) {
                    return payloadWord<Codebook, Ops>(unit, offset, column);
                };
                const I index_even = (word((2 * Q) / 4) >> (8 * ((2 * Q) % 4))) & 255;
                const I index_odd = (word((2 * Q + 1) / 4) >> (8 * ((2 * Q + 1) % 4))) & 255;
                // signGrid tests one bit per value byte, so the shared sign
                // byte must be replicated into every byte of each lane.
                const I signs = Ops::template repeatByte<Q>(word(2));
                I low, high, unused;
                Ops::gridPair(iq3xxs_grid64, index_even, low, unused);
                Ops::gridPair(iq3xxs_grid64, index_odd, high, unused);
                const auto consume = [&]<int H>(I weights) {
                    weights = Ops::template signGrid<H>(weights, signs);
                    if constexpr (Lanes == 16)
                        weights ^= static_cast<int>(0x80808080u);
#pragma GCC unroll 4
                    for (size_t row = 0; row < Rows; ++row)
                    {
                        int activation_word = 0;
                        std::memcpy(&activation_word,
                            activation[row][kb].qs + Q * 8 + H * 4, 4);
                        dots[row] = Ops::dot(dots[row], weights, I{} + activation_word);
                    }
                };
                consume.template operator()<0>(low);
                consume.template operator()<1>(high);
            };
            quarter.template operator()<Half * 2>();
            quarter.template operator()<Half * 2 + 1>();
        }
        else if constexpr (Codebook == 13 || Codebook == 14 || Codebook == 15 ||
                           Codebook == 16 || Codebook == 17)
        {
            const auto quarter = [&]<int Q>() {
                const auto word = [&](int offset) {
                    return payloadWord<Codebook, Ops>(unit, offset, column);
                };
                I index = (word(0) >> (Q * 8)) & 255;
                if constexpr (Codebook == 16)
                    index |= ((word(1) & 0xFFFF) >> (3 * Q) & 7) << 8;
                else if constexpr (Codebook == 17)
                    index |= ((word(1) >> ((Q / 2) * 8 + (Q % 2) * 4)) & 7) << 8;
                else if constexpr (Codebook != 15)
                {
                    constexpr int bits = Codebook == 13 ? 2 : 1;
                    index |= ((word(1) >> (Q * bits)) & ((1 << bits) - 1)) << 8;
                }
                const uint64_t *table = (Codebook == 17 || Codebook == 16) ? iq1s_grid
                    : Codebook == 13 ? iq2s_grid
                    : Codebook == 15 ? iq2xxs_grid : iq2xs_grid;
                I low, high;
                Ops::gridPair(table, index, low, high);
                I signs{};
                if constexpr (Codebook == 13 || Codebook == 14 || Codebook == 15)
                {
                    constexpr int sign_byte = Codebook == 15 ? 4 + Q : 5 + Q;
                    signs = Ops::template repeatByte<sign_byte % 4>(word(sign_byte / 4));
                }
                const auto consume = [&]<int H>(I weights) {
                    if constexpr (Codebook != 17)
                        weights = Ops::template signGrid<H>(weights, signs);
                    if constexpr (Lanes == 16)
                        weights ^= static_cast<int>(0x80808080u);
#pragma GCC unroll 4
                    for (size_t row = 0; row < Rows; ++row)
                    {
                        int activation_word = 0;
                        std::memcpy(&activation_word,
                            activation[row][kb].qs + Q * 8 + H * 4, 4);
                        dots[row] = Ops::dot(dots[row], weights, I{} + activation_word);
                    }
                };
                // Consume one signed half at a time. Both ISA implementations
                // share the decoded index; AVX-512 also halves indexed loads.
                consume.template operator()<0>(low);
                consume.template operator()<1>(high);
            };
            quarter.template operator()<Half * 2>();
            quarter.template operator()<Half * 2 + 1>();
        }
        else
        {
            const auto group = [&]<int G>() {
                const I weights = decodeLinear<Codebook, G, Lanes, Ops>(unit, column);
#pragma GCC unroll 4
                for (size_t row = 0; row < Rows; ++row)
                {
                    int word = 0;
                    std::memcpy(&word, activation[row][kb].qs + G * 4, 4);
                    dots[row] = Ops::dot(dots[row], weights, I{} + word);
                }
            };
            group.template operator()<Half * 4>();
            group.template operator()<Half * 4 + 1>();
            group.template operator()<Half * 4 + 2>();
            group.template operator()<Half * 4 + 3>();
        }
    }

    /**
     * @brief Execute a shared-weight tile with the GPU's exact multi-scale expression.
     * @tparam Codebook Native source execution codebook.
     * @tparam Lanes Physical SIMD column width.
     * @tparam Rows Independent rows sharing each payload decode.
     * @tparam Tree Contiguous dot or the admitted ordered expert partition tree.
     * @param packed Prepared immutable native matrix.
     * @param activation Persistent Q8_1 rows.
     * @param output Padded output rows for this 64-column chunk.
     * @param chunk Output-column chunk index.
     * @param begin First logical K block.
     * @param end One-past-last logical K block.
     * @param accumulate Whether the output owns a preceding K partition.
     * @param partitions Exact number of ordered partitions; unused for Contiguous.
     * @param blocks_per_partition Exact K-block span; unused for Contiguous.
     * @param columns Live output span in this chunk; padded x86 callers retain 64.
     * @throws std::invalid_argument For invalid chunk, K, output or partition geometry.
     */
    template <int Codebook, int Lanes, size_t Rows, Reduction Tree = Reduction::Contiguous,
              typename Packed, typename Ops = SIMD<Lanes>>
    LLAMINAR_MULTISCALE_EXACT inline void rows(const Packed &packed,
                                               const std::array<const Q8_1Block *, Rows> &activation,
                                               const std::array<float *, Rows> &output, int chunk, int begin,
                                               int end, bool accumulate, int partitions = 1,
                                               int blocks_per_partition = 0, int columns = 64)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        using I = typename Ops::I;
        using F = typename Ops::F;
        static_assert(Rows >= 1 && Rows <= 4);
        // Ordered K tiles have a fixed count, so the trailing tiles of a short
        // K start at or beyond blocks_per_row with a clamped end. Such an empty
        // span reads no weight block and still publishes zero or the retained
        // accumulator; only a nonempty span must lie inside the matrix.
        if (chunk < 0 || chunk >= packed.N_padded / 64 || columns <= 0 || columns > 64 ||
            (end > begin && (begin < 0 || end > packed.blocks_per_row)))
            throw std::invalid_argument("Multi-scale CPU tile is outside its admitted geometry");
        if constexpr (Tree == Reduction::OrderedPartitions)
            if (partitions <= 0 || blocks_per_partition <= 0 ||
                static_cast<int64_t>(partitions) * blocks_per_partition < end - begin ||
                static_cast<int64_t>(begin) + static_cast<int64_t>(partitions) * blocks_per_partition > INT32_MAX)
                throw std::invalid_argument("Multi-scale CPU tile requires an exact bounded ordered K tree");
        // Q5_K (7) decodes to unsigned five-bit values (0..31), so its AVX-512
        // bytes need no signed->unsigned bias and the recovered dot needs no
        // center correction.
        constexpr int center = Lanes != 16 || Codebook == 10 || Codebook == 7
            ? 0 : Codebook == 9 ? 4 : 128;
#if defined(__AVX2__)
        // Bind the already-admitted layout from the shared geometry authority.
        // Keep the traversal's additional address state within the budget
        // of each emitted tile. In particular the AVX2 grouped Q5_K tile
        // would save live vectors; the AVX-512 grouped contiguously folded
        // tile does not have the extra ordered partition accumulators.
        constexpr bool fixed_layout = (Lanes == 8 && Codebook == 7 && Rows == 1) ||
            (Lanes == 16 && Codebook == 11 && (Tree == Reduction::Contiguous || Rows <= 2)) ||
            (Lanes == 16 && Codebook == 7 && Rows > 1 && Tree == Reduction::Contiguous);
        constexpr auto encoding = CPUNativeQuantisedEncoding::CompactMultiScale;
        constexpr auto data_stride = preparedDataStride(encoding, Codebook);
        const auto geometry = preparedBlockGeometry(encoding, Codebook == 7, Codebook, packed.blocks_per_row);
        const auto *chunk_base = [&] {
            if constexpr (fixed_layout)
                return packed.interleavedBase() + static_cast<std::size_t>(chunk) * geometry.rowBytes();
            else
                return static_cast<const uint8_t *>(nullptr);
        }();
#endif
        for (int column = 0; column < columns; column += Lanes)
        {
            F reduced[Rows]{};
            const int partition_count = Tree == Reduction::OrderedPartitions ? partitions : 1;
            for (int partition = 0; partition < partition_count; ++partition)
            {
                F acc[Rows];
#pragma GCC unroll 4
                for (size_t row = 0; row < Rows; ++row)
                {
                    acc[row] = F{};
                    if constexpr (Tree == Reduction::Contiguous)
                        if (accumulate)
                        {
                            if (column + Lanes <= columns)
                                acc[row] = load<F>(output[row] + column);
                            else
                                std::memcpy(&acc[row], output[row] + column, (columns - column) * sizeof(float));
                        }
                }
                const int first =
                    Tree == Reduction::OrderedPartitions ? begin + partition * blocks_per_partition : begin;
                const int last =
                    Tree == Reduction::OrderedPartitions ? std::min(first + blocks_per_partition, end) : end;
                for (int kb = first; kb < last; ++kb)
                {
#if defined(__AVX2__)
                    const auto *unit = [&] {
                        if constexpr (fixed_layout)
                            return chunk_base + geometry.blockOffset(0, kb);
                        else
                            return packed.admittedBlockUnit(chunk, kb);
                    }();
#else
                    const uint8_t *const unit = packed.admittedBlockUnit(chunk, kb);
#endif
                    I low[Rows]{}, high[Rows]{};
                    if constexpr (Codebook == 7 || Codebook == 11)
                        dotSingleScale<Codebook, Lanes, Rows, Tree, Ops>(unit, column, activation, kb, low);
                    else
                    {
                        dotHalf<Codebook, 0, Lanes, Rows, Ops>(unit, column, activation, kb, low);
                        dotHalf<Codebook, 1, Lanes, Rows, Ops>(unit, column, activation, kb, high);
                    }
                    // Coded metadata: the block's code and its superblock word
                    // reconstruct exactly the factors the per-block words held.
                    using L = coded_metadata::Lanes<Ops>;
#if defined(__AVX2__)
                    const auto *header = [&] {
                        if constexpr (fixed_layout)
                            return chunk_base + geometry.headerOffset(0, kb);
                        else
                            return packed.admittedSuperblockHeader(chunk, kb);
                    }();
                    const I superblock_word = L::template superblockWords<static_cast<std::uint8_t>(Codebook)>(header, column);
                    const I code = compactCode<Codebook, Ops>(unit, fixed_layout ? data_stride : packed.data_stride, column);
#else
                    const I superblock_word = L::template superblockWords<static_cast<std::uint8_t>(Codebook)>(
                        packed.admittedSuperblockHeader(chunk, kb), column);
                    const I code = compactCode<Codebook, Ops>(unit, packed.data_stride, column);
#endif
                    F scale0, scale1;
                    [[maybe_unused]] coded_metadata::Factors<F> factors;
                    if constexpr (Codebook == 13)
                    {
                        // IQ2_S keeps its source factors: base d and the coefficient byte.
                        // Function contraction/reassociation is disabled above;
                        // each source multiply remains a separate FP32 edge.
                        const F low_coefficient = __builtin_convertvector(code & 15, F);
                        const F high_coefficient = __builtin_convertvector((code >> 4) & 15, F);
                        const F base = L::half(superblock_word);
                        scale0 = (base * (low_coefficient + 0.5f)) * 0.25f;
                        scale1 = (base * (high_coefficient + 0.5f)) * 0.25f;
                    }
                    else if constexpr (Codebook == 11)
                    {
                        // IQ3_S keeps its source factors: base d and the scale nibble,
                        // factor = d * (2*nibble+1) in FP32 with no half re-rounding,
                        // exactly nativeQuantisedPrimaryScale's IQ3_S expression.
                        scale0 = iq3sSourceScale<Ops>(code, superblock_word);
                        scale1 = F{};  // single-source-scale, no additive minimum.
                    }
                    else
                    {
                        factors = coded_metadata::reconstruct<static_cast<std::uint8_t>(Codebook), Ops>(code, superblock_word);
                        scale0 = factors.primary;
                        scale1 = factors.secondary;
                    }
                    // Rows is at most four. Expanding this tiny loop keeps every
                    // accumulator/dot in a distinct register: a runtime-indexed
                    // array otherwise spills the entire tile on every K block.
#pragma GCC unroll 4
                    for (size_t row = 0; row < Rows; ++row)
                    {
                        const auto &a = activation[row][kb];
                        // IQ1's signed quarter corrections need all four sums.
                        // Other formats need only one half: the quantizer already
                        // publishes the exact whole-block sum in Q8_1Block. Avoid
                        // expanding/reducing the same 32 bytes for every column.
                        std::array<int, 4> sums{};
                        int sum0 = 0, sum1 = 0;
                        if constexpr (Codebook == 17)
                        {
                            sums = activationQuarterSums(a.qs);
                            sum0 = sums[0] + sums[1];
                            sum1 = sums[2] + sums[3];
                        }
                        else if constexpr (center != 0 || Codebook == 10)
                        {
                            sum0 = activationHalfSum(a.qs);
                            sum1 = static_cast<int>(a.sum_qs) - sum0;
                        }
                        const F activation_scale = F{} + nativeQuantisedFP16ScaleToFP32(a.d);
                        if constexpr (Codebook == 7 || Codebook == 11)
                        {
                            const auto dot = low[row] - center * static_cast<int>(a.sum_qs);
                            const auto scale_a = nativeQuantisedFP16ScaleToFP32(a.d);
#if defined(__aarch64__) && defined(__ARM_NEON)
                            if constexpr (Codebook == 7)
                                acc[row] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockNEON(
                                    std::bit_cast<float32x4_t>(acc[row]), std::bit_cast<int32x4_t>(dot),
                                    std::bit_cast<float32x4_t>(scale0), a.sum_qs,
                                    std::bit_cast<float32x4_t>(scale1), scale_a, packed.numerical_policy));
                            else
                                acc[row] = std::bit_cast<F>(accumulateSingleScaleBlockNEON(
                                    std::bit_cast<float32x4_t>(acc[row]), std::bit_cast<int32x4_t>(dot),
                                    std::bit_cast<float32x4_t>(scale0), scale_a, packed.numerical_policy));
#else
                            if constexpr (Lanes == 8)
                            {
                                if constexpr (Codebook == 7)
                                    acc[row] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX2(
                                        std::bit_cast<__m256>(acc[row]), std::bit_cast<__m256i>(dot),
                                        std::bit_cast<__m256>(scale0), a.sum_qs,
                                        std::bit_cast<__m256>(scale1), scale_a, packed.numerical_policy));
                                else
                                    acc[row] = std::bit_cast<F>(accumulateSingleScaleBlockAVX2(
                                        std::bit_cast<__m256>(acc[row]), std::bit_cast<__m256i>(dot),
                                        std::bit_cast<__m256>(scale0), scale_a, packed.numerical_policy));
                            }
#if defined(__AVX512F__)
                            else
                            {
                                if constexpr (Codebook == 7)
                                    acc[row] = std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(
                                        std::bit_cast<__m512>(acc[row]), std::bit_cast<__m512i>(dot),
                                        std::bit_cast<__m512>(scale0), a.sum_qs,
                                        std::bit_cast<__m512>(scale1), scale_a, packed.numerical_policy));
                                else
                                    acc[row] = std::bit_cast<F>(accumulateSingleScaleBlockAVX512(
                                        std::bit_cast<__m512>(acc[row]), std::bit_cast<__m512i>(dot),
                                        std::bit_cast<__m512>(scale0), scale_a, packed.numerical_policy));
                            }
#endif
#endif
                            continue;
                        }
                        if constexpr (isSingleScaleCompactCodebook(Codebook))
                        {
                            // GPU singleScaleBlock over the exact whole-block
                            // integer dot; IQ1_S adds its scaled additive
                            // minimum over the whole block sum.
                            const int whole_sum = static_cast<int>(a.sum_qs);
                            const F dot = __builtin_convertvector(low[row] + high[row] - center * whole_sum, F);
                            F contribution = (activation_scale * scale0) * dot;
                            if constexpr (Codebook == 16)
                                contribution += (activation_scale * scale1) * static_cast<float>(whole_sum);
                            acc[row] += contribution;
                            continue;
                        }
                        const F dot0 = __builtin_convertvector(low[row] - center * sum0, F);
                        const F dot1 = __builtin_convertvector(high[row] - center * sum1, F);
                        const F term0 = scale0 * dot0;
                        const F term1 = scale1 * dot1;
                        F contribution = activation_scale * (term0 + term1);
                        if constexpr (Codebook == 10)
                        {
                            // Minima retain their original half bits and the GPU
                            // rounds this correction independently of the dot term.
                            const F correction0 = factors.emin_low * static_cast<float>(sum0);
                            const F correction1 = factors.emin_high * static_cast<float>(sum1);
                            contribution += activation_scale * (correction0 + correction1);
                        }
                        if constexpr (Codebook == 17)
                        {
                            const I signs = payloadWord<Codebook, Ops>(unit, 1, column);
                            const auto delta = [&](int bit, int sum) {
                                // Materialize the sign on the FP32 delta itself so
                                // zero sums preserve exactly the GPU's signed zero.
                                const I delta_bits = (I{} + 0x3e000000) ^ (((signs >> bit) & 1) << 31);
                                return std::bit_cast<F>(delta_bits) * static_cast<float>(sum);
                            };
                            const F correction0 = (delta(3, sums[0]) + delta(7, sums[1])) * scale0;
                            const F correction1 = (delta(11, sums[2]) + delta(15, sums[3])) * scale1;
                            contribution += activation_scale * (correction0 + correction1);
                        }
                        acc[row] += contribution;
                    }
                }
#pragma GCC unroll 4
                for (size_t row = 0; row < Rows; ++row)
                {
                    // The first partial is copied, not added to an extra zero.
                    // Later partials retain the exact device tree's left fold.
                    // Keeping this inside the column loop removes repeated
                    // dispatch/setup and temporary 64-column partial planes.
                    if constexpr (Tree == Reduction::OrderedPartitions)
                        reduced[row] = partition == 0 ? acc[row] : reduced[row] + acc[row];
                    else
                        reduced[row] = acc[row];
                }
            }
#pragma GCC unroll 4
            for (size_t row = 0; row < Rows; ++row)
            {
                if (column + Lanes <= columns)
                    std::memcpy(output[row] + column, &reduced[row], sizeof(F));
                else
                    std::memcpy(output[row] + column, &reduced[row], (columns - column) * sizeof(float));
            }
        }
    }

    /**
     * @brief Dispatch native format once outside the entire K traversal.
     * @param packed Compact multi-scale representation; other encodings fail.
     * @param activation Quantized rows sharing this tile.
     * @param output One 64-column output span per row.
     * @param chunk Output-column chunk.
     * @param begin First K block.
     * @param end Exclusive K block bound.
     * @param accumulate Preserve prior K partition output.
     * @param columns Live output columns; the padded x86 contract defaults to 64.
     */
    template <int Lanes, size_t Rows, typename Ops = SIMD<Lanes>, typename Packed>
    inline void dispatch(const Packed &packed,
                         const std::array<const Q8_1Block *, Rows> &activation,
                         const std::array<float *, Rows> &output, int chunk, int begin, int end,
                         bool accumulate, int columns = 64)
    {
        if (!packed.usesCompactMultiScale())
            throw std::invalid_argument("Compact multi-scale kernel requires its native encoding");
        switch (packed.codebook_id)
        {
#define LLAMINAR_MULTISCALE_CASE(C) case C: return rows<C, Lanes, Rows, Reduction::Contiguous, Packed, Ops>(packed, activation, output, chunk, begin, end, accumulate, 1, 0, columns)
        LLAMINAR_MULTISCALE_CASE(7);
        LLAMINAR_MULTISCALE_CASE(9);
        LLAMINAR_MULTISCALE_CASE(10);
        LLAMINAR_MULTISCALE_CASE(11);
        LLAMINAR_MULTISCALE_CASE(12);
        LLAMINAR_MULTISCALE_CASE(13);
        LLAMINAR_MULTISCALE_CASE(14);
        LLAMINAR_MULTISCALE_CASE(15);
        LLAMINAR_MULTISCALE_CASE(16);
        LLAMINAR_MULTISCALE_CASE(17);
#undef LLAMINAR_MULTISCALE_CASE
        }
        throw std::invalid_argument("Unsupported native multi-scale codebook");
    }

    /**
     * @brief Execute the admitted ordered expert tree in one compact-format call.
     * @param packed Immutable compact multi-scale weights.
     * @param activation One persistent Q8_1 activation row per tile row.
     * @param output Sixty-four writable FP32 columns per tile row.
     * @param chunk Output column chunk.
     * @param blocks Complete logical K extent in 32-value blocks.
     * @param partitions Exact arithmetic partition count from the shared contract.
     * @param blocks_per_partition Consecutive K blocks per partial.
     * @param columns Live output columns; the padded x86 contract defaults to 64.
     * @throws std::invalid_argument For inconsistent encoding or partition geometry.
     *
     * This changes physical traversal only. Each partial starts at positive
     * zero, visits the same K interval and is folded at the same rounding point
     * as the independent producer/reducer path. No quantized bytes, dispatch
     * choice, output precision, or scratch capacity are changed.
     */
    template <int Lanes, size_t Rows, typename Ops = SIMD<Lanes>, typename Packed>
    inline void dispatchOrdered(const Packed &packed,
                                const std::array<const Q8_1Block *, Rows> &activation,
                                const std::array<float *, Rows> &output,
                                int chunk, int blocks, int partitions,
                                int blocks_per_partition, int columns = 64)
    {
        if (!packed.usesCompactMultiScale() || blocks <= 0 ||
            blocks != packed.blocks_per_row || partitions <= 0 ||
            blocks_per_partition <= 0 ||
            static_cast<int64_t>(partitions) * blocks_per_partition < blocks ||
            static_cast<int64_t>(partitions) * blocks_per_partition > INT32_MAX)
            throw std::invalid_argument("Compact ordered dot requires its exact admitted K tree");
        switch (packed.codebook_id)
        {
#define LLAMINAR_MULTISCALE_ORDERED_CASE(C) case C: return rows<C, Lanes, Rows, Reduction::OrderedPartitions, Packed, Ops>(packed, activation, output, chunk, 0, blocks, false, partitions, blocks_per_partition, columns)
        LLAMINAR_MULTISCALE_ORDERED_CASE(7);
        LLAMINAR_MULTISCALE_ORDERED_CASE(9);
        LLAMINAR_MULTISCALE_ORDERED_CASE(10);
        LLAMINAR_MULTISCALE_ORDERED_CASE(11);
        LLAMINAR_MULTISCALE_ORDERED_CASE(12);
        LLAMINAR_MULTISCALE_ORDERED_CASE(13);
        LLAMINAR_MULTISCALE_ORDERED_CASE(14);
        LLAMINAR_MULTISCALE_ORDERED_CASE(15);
        LLAMINAR_MULTISCALE_ORDERED_CASE(16);
        LLAMINAR_MULTISCALE_ORDERED_CASE(17);
#undef LLAMINAR_MULTISCALE_ORDERED_CASE
        }
        throw std::invalid_argument("Unsupported native multi-scale codebook");
    }
#undef LLAMINAR_MULTISCALE_EXACT
}
