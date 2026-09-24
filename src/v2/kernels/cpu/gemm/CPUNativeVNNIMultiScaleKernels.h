/**
 * @file CPUNativeVNNIMultiScaleKernels.h
 * @brief Lossless compact multi-scale CPU projections with shared serial/grouped arithmetic.
 *
 * Prepared blocks retain the GPU-native payload, two FP16 scales and optional
 * Q2_K minima. SIMD lanes represent output columns, so each activation word is
 * broadcast once and reused across columns and verifier rows. Integer lookup
 * and dot work never changes source quantization. Each row visits increasing
 * K blocks and publishes the same separately rounded contribution as CUDA/HIP.
 * The same primitive serves serial decode and grouped verification; no row
 * replay, temporary decoded matrix or inference-time allocation is used.
 */
#pragma once

#include <array>
#include <bit>
#include <cstring>
#include <immintrin.h>
#include <stdexcept>

#include "CPUNativeVNNIWeightPacker.h"
#include "CPUNativeVNNIFP16.h"
#include "VNNIEmulation.h"

namespace llaminar2::cpu::native_vnni::multi_scale
{
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
                  gridFitsSignedDot(iq1s_grid), "Native grids must exclude the non-negatable INT8 minimum");

#if defined(__GNUC__) && !defined(__clang__)
#define LLAMINAR_MULTISCALE_EXACT __attribute__((optimize("fp-contract=off", "no-associative-math")))
#else
#define LLAMINAR_MULTISCALE_EXACT
#endif

    /** @brief ISA operations; vector arithmetic below is identical for both widths. */
    template <int Lanes> struct SIMD;

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
        /** @return Bytewise difference without cross-byte borrow. */
        static I subtractBytes(I a, I b)
        {
            return std::bit_cast<I>(_mm256_sub_epi8(std::bit_cast<__m256i>(a), std::bit_cast<__m256i>(b)));
        }
        /** @return Selected four-byte halves of immutable eight-byte grid entries. */
        static I grid(const uint64_t *table, I index, int half)
        {
            return std::bit_cast<I>(_mm256_i32gather_epi32(
                reinterpret_cast<const int *>(table), std::bit_cast<__m256i>(index * 2 + half), 4));
        }
        /** @return Eight exact FP16-to-FP32 conversions. */
        static F half(const uint16_t *source)
        {
            return std::bit_cast<F>(_mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(source))));
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
        /** @return Selected four-byte halves of immutable eight-byte grid entries. */
        static I grid(const uint64_t *table, I index, int half)
        {
            return std::bit_cast<I>(_mm512_i32gather_epi32(
                std::bit_cast<__m512i>(index * 2 + half), table, 4));
        }
        /** @return Sixteen exact FP16-to-FP32 conversions. */
        static F half(const uint16_t *source)
        {
            return std::bit_cast<F>(_mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(source))));
        }
        /** @return FP32 values from the low FP16 half of each packed word. */
        static F halfWords(I words)
        {
            return std::bit_cast<F>(_mm512_cvtph_ps(
                _mm512_cvtepi32_epi16(std::bit_cast<__m512i>(words))));
        }
    };
#endif

    /** @brief Load a SIMD value without imposing aliasing or alignment on the caller. */
    template <typename V> inline V load(const void *source)
    {
        V value;
        std::memcpy(&value, source, sizeof(value));
        return value;
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
    LLAMINAR_MULTISCALE_EXACT
    inline float scalarReferenceBlock(const CPUNativeVNNIPackedWeights &packed,
                                     const Q8_1Block &a, int chunk, int kb, int column)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        uint8_t payload[16]{};
        int8_t values[32];
        for (int byte = 0; byte < packed.payload_bytes; ++byte)
            payload[byte] = packed.interleavedB(chunk, kb, byte / 4, column / 16)[(column % 16) * 4 + byte % 4];
        decode_native_block(packed.codebook_id, payload, values);
        int dots[2]{}, sums[4]{};
        for (int k = 0; k < 32; ++k)
        {
            dots[k / 16] += static_cast<int>(values[k]) * a.qs[k];
            sums[k / 8] += a.qs[k];
        }
        const float scale0 = fp16_to_fp32(packed.chunkScales(chunk, kb)[column]);
        const float scale1 = fp16_to_fp32(packed.chunkMins(chunk, kb)[column]);
        const float activation_scale = nativeVNNIFP16ScaleToFP32(a.d);
        float contribution = activation_scale * (scale0 * dots[0] + scale1 * dots[1]);
        if (packed.codebook_id == 10)
        {
            const uint32_t bits = packed.chunkEffectiveMins(chunk, kb)[column];
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

    /**
     * @brief Decode one four-K group for a SIMD vector of output columns.
     * @tparam Codebook Source execution format (9, 10, 13, 14 or 17).
     * @tparam Group Four-value group in [0,7].
     * @tparam Lanes Eight AVX2 or sixteen AVX-512 columns.
     * @param unit Native CPU unit's first payload byte.
     * @param column First SIMD column within the 64-column unit.
     * @return Signed native grids on AVX2; unsigned-biased grids for AVX-512 VNNI.
     */
    template <int Codebook, int Group, int Lanes>
    inline typename SIMD<Lanes>::I decode(const uint8_t *unit, int column)
    {
        using Ops = SIMD<Lanes>;
        using I = typename Ops::I;
        const auto word = [&](int index) { return load<I>(unit + index * 256 + column * 4); };
        if constexpr (Codebook == 9 || Codebook == 10)
        {
            I values = (word(Group / 4) >> ((Group % 4) * 2)) & 0x03030303;
            if constexpr (Codebook == 9)
            {
                // Four high bits become bit two of four separate byte lanes.
                const I high = (word(2) >> (Group * 4)) & 15;
                values |= ((high * 0x02040810) & 0x10101010) >> 2;
                if constexpr (Lanes == 8)
                    values = Ops::subtractBytes(values, I{} + 0x04040404);
            }
            return values;
        }
        else
        {
            constexpr int grid_group = Group / 2;
            I index = (word(0) >> (grid_group * 8)) & 255;
            I values;
            if constexpr (Codebook == 17)
            {
                index |= ((word(1) >> ((grid_group / 2) * 8 + (grid_group % 2) * 4)) & 7) << 8;
                values = Ops::grid(iq1s_grid, index, Group % 2);
            }
            else
            {
                constexpr int bits = Codebook == 13 ? 2 : 1;
                index |= ((word(1) >> (grid_group * bits)) & ((1 << bits) - 1)) << 8;
                values = Ops::grid(Codebook == 13 ? iq2s_grid : iq2xs_grid, index, Group % 2);
                constexpr int sign_byte = 5 + grid_group;
                const I signs = (word(sign_byte / 4) >> ((sign_byte % 4) * 8 + (Group % 2) * 4)) & 15;
                const I spread = ((signs * 0x02040810) & 0x10101010) >> 4;
                const I mask = Ops::subtractBytes(I{}, spread);
                values = Ops::subtractBytes(values ^ mask, mask);
            }
            if constexpr (Lanes == 8)
                return values;
            // Native AVX-512 VNNI takes unsigned weights without a sign shuffle.
            return values ^ static_cast<int>(0x80808080u);
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
    template <int Codebook, int Half, int Lanes, size_t Rows>
    inline void dotHalf(const uint8_t *unit, int column,
                        const std::array<const Q8_1Block *, Rows> &activation,
                        int kb, typename SIMD<Lanes>::I (&dots)[Rows])
    {
        using Ops = SIMD<Lanes>;
        using I = typename Ops::I;
        const auto group = [&]<int G>() {
            const I weights = decode<Codebook, G, Lanes>(unit, column);
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

    /**
     * @brief Execute a shared-weight tile with the GPU's exact multi-scale expression.
     * @tparam Codebook Native source execution codebook.
     * @tparam Lanes Physical SIMD column width.
     * @tparam Rows Independent rows sharing each payload decode.
     * @param packed Prepared immutable native matrix.
     * @param activation Persistent Q8_1 rows.
     * @param output Padded output rows for this 64-column chunk.
     * @param chunk Output-column chunk index.
     * @param begin First logical K block.
     * @param end One-past-last logical K block.
     * @param accumulate Whether the output owns a preceding K partition.
     */
    template <int Codebook, int Lanes, size_t Rows>
    LLAMINAR_MULTISCALE_EXACT
    inline void rows(const CPUNativeVNNIPackedWeights &packed,
                     const std::array<const Q8_1Block *, Rows> &activation,
                     const std::array<float *, Rows> &output,
                     int chunk, int begin, int end, bool accumulate)
    {
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#endif
        using Ops = SIMD<Lanes>;
        using I = typename Ops::I;
        using F = typename Ops::F;
        static_assert(Rows >= 1 && Rows <= 4);
        constexpr int center = Lanes == 8 || Codebook == 10 ? 0 : Codebook == 9 ? 4 : 128;
        for (int column = 0; column < 64; column += Lanes)
        {
            F acc[Rows];
#pragma GCC unroll 4
            for (size_t row = 0; row < Rows; ++row)
                acc[row] = accumulate ? load<F>(output[row] + column) : F{};
            for (int kb = begin; kb < end; ++kb)
            {
                const uint8_t *const unit = packed.interleavedBase() +
                    (static_cast<size_t>(chunk) * packed.blocks_per_row + kb) * packed.interleaved_block_stride;
                I low[Rows]{}, high[Rows]{};
                dotHalf<Codebook, 0, Lanes>(unit, column, activation, kb, low);
                dotHalf<Codebook, 1, Lanes>(unit, column, activation, kb, high);
                const F scale0 = Ops::half(packed.chunkScales(chunk, kb) + column);
                const F scale1 = Ops::half(packed.chunkMins(chunk, kb) + column);
                // Rows is at most four. Expanding this tiny loop keeps every
                // accumulator/dot in a distinct register: a runtime-indexed
                // array otherwise spills the entire tile on every K block.
#pragma GCC unroll 4
                for (size_t row = 0; row < Rows; ++row)
                {
                    const auto &a = activation[row][kb];
                    int sums[4]{};
                    for (int group = 0; group < 4; ++group)
                        for (int k = 0; k < 8; ++k)
                            sums[group] += a.qs[group * 8 + k];
                    const int sum0 = sums[0] + sums[1];
                    const int sum1 = sums[2] + sums[3];
                    const F dot0 = __builtin_convertvector(low[row] - center * sum0, F);
                    const F dot1 = __builtin_convertvector(high[row] - center * sum1, F);
                    const F activation_scale = F{} + nativeVNNIFP16ScaleToFP32(a.d);
                    const F term0 = scale0 * dot0;
                    const F term1 = scale1 * dot1;
                    F contribution = activation_scale * (term0 + term1);
                    if constexpr (Codebook == 10)
                    {
                        // Minima retain their original half bits and the GPU
                        // rounds this correction independently of the dot term.
                        const I bits = load<I>(packed.chunkEffectiveMins(chunk, kb) + column);
                        const F correction0 = Ops::halfWords(bits) * static_cast<float>(sum0);
                        const F correction1 = Ops::halfWords(bits >> 16) * static_cast<float>(sum1);
                        contribution += activation_scale * (correction0 + correction1);
                    }
                    if constexpr (Codebook == 17)
                    {
                        const I signs = load<I>(unit + 256 + column * 4);
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
                std::memcpy(output[row] + column, &acc[row], sizeof(F));
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
     */
    template <int Lanes, size_t Rows>
    inline void dispatch(const CPUNativeVNNIPackedWeights &packed,
                         const std::array<const Q8_1Block *, Rows> &activation,
                         const std::array<float *, Rows> &output,
                         int chunk, int begin, int end, bool accumulate)
    {
        if (!packed.usesCompactMultiScale())
            throw std::invalid_argument("Compact multi-scale kernel requires its native encoding");
        switch (packed.codebook_id)
        {
#define LLAMINAR_MULTISCALE_CASE(C) case C: return rows<C, Lanes>(packed, activation, output, chunk, begin, end, accumulate)
        LLAMINAR_MULTISCALE_CASE(9);
        LLAMINAR_MULTISCALE_CASE(10);
        LLAMINAR_MULTISCALE_CASE(13);
        LLAMINAR_MULTISCALE_CASE(14);
        LLAMINAR_MULTISCALE_CASE(17);
#undef LLAMINAR_MULTISCALE_CASE
        }
        throw std::invalid_argument("Unsupported native multi-scale codebook");
    }
#undef LLAMINAR_MULTISCALE_EXACT
}
