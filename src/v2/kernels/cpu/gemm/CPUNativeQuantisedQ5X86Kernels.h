/**
 * @file CPUNativeQuantisedQ5X86Kernels.h
 * @brief AVX2 and AVX-512 projection of 20-byte Q5_0/Q5_1 payloads.
 *
 * Both ISAs decode the same four paired K groups into unsigned [0,31]
 * bytes and dot against signed Q8 bytes. Q5_0 then subtracts sixteen times
 * the actual activation sum in INT32; neither format stores weight sums.
 * The surrounding FP32 block program is the existing per-policy contract.
 */
#pragma once

#include "CPUNativeQuantisedContributionContract.h"
#include "CPUNativeQuantisedMultiScaleKernels.h"
#include "CPUNativeQuantisedPackedWeights.h"

namespace llaminar2::cpu::native_quantised::q5
{
    /** @brief Load low nibbles and expand the one-byte fifth-bit pairs for eight columns. */
    [[nodiscard]] inline auto decode8(const std::uint8_t *unit, int column, int group) noexcept
    {
        using Ops = multi_scale::SIMD<8>;
        using I = Ops::I;
        const auto nibbles = multi_scale::load<__m256i>(unit + Q5NativePayload::nibbleOffset(group, column));
        const auto fifth = _mm256_srl_epi32(
            multi_scale::load<__m256i>(unit + Q5NativePayload::fifthOffset(0, column)), _mm_cvtsi32_si128(group * 8));
        const auto bits = std::bit_cast<__m256i>(Ops::repeatByte<0>(std::bit_cast<I>(fifth)));
        const auto unpack = [](__m256i raw, __m256i fifth_bits) {
            const auto mask = _mm256_set1_epi32(0x08040201);
            const auto missing = _mm256_cmpeq_epi8(_mm256_and_si256(fifth_bits, mask), _mm256_setzero_si256());
            return _mm256_or_si256(_mm256_and_si256(raw, _mm256_set1_epi8(15)),
                                   _mm256_andnot_si256(missing, _mm256_set1_epi8(16)));
        };
        // Shift the second set of fifth bits into the same mask positions,
        // reducing the constants competing with the live row accumulators.
        return std::pair{std::bit_cast<I>(unpack(nibbles, bits)),
                         std::bit_cast<I>(unpack(_mm256_srli_epi16(nibbles, 4), _mm256_srli_epi16(bits, 4)))};
    }

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    /** @brief Load low nibbles and expand the one-byte fifth-bit pairs for sixteen columns. */
    [[nodiscard]] inline auto decode16(const std::uint8_t *unit, int column, int group) noexcept
    {
        using Ops = multi_scale::SIMD<16>;
        using I = Ops::I;
        const auto nibbles = multi_scale::load<__m512i>(unit + Q5NativePayload::nibbleOffset(group, column));
        const auto fifth = _mm512_srl_epi32(
            multi_scale::load<__m512i>(unit + Q5NativePayload::fifthOffset(0, column)), _mm_cvtsi32_si128(group * 8));
        const auto bits = std::bit_cast<__m512i>(Ops::repeatByte<0>(std::bit_cast<I>(fifth)));
        const auto unpack = [&]<int Shift>(__m512i raw) {
            const auto mask = _mm512_set1_epi32(static_cast<int>(0x08040201u << Shift));
            const auto low = _mm512_and_si512(raw, _mm512_set1_epi8(15));
            return _mm512_mask_add_epi8(low, _mm512_test_epi8_mask(bits, mask), low, _mm512_set1_epi8(16));
        };
        return std::pair{std::bit_cast<I>(unpack.template operator()<0>(nibbles)),
                         std::bit_cast<I>(unpack.template operator()<4>(_mm512_srli_epi16(nibbles, 4)))};
    }
#endif

    /** @brief Broadcast one signed activation four-byte group and append its exact dot. */
    template <int Lanes>
    [[nodiscard]] inline typename multi_scale::SIMD<Lanes>::I dot(
        typename multi_scale::SIMD<Lanes>::I acc, typename multi_scale::SIMD<Lanes>::I weights,
        const Q8_1Block &activation, int group) noexcept
    {
        const auto word = multi_scale::load<int>(activation.qs + group * 4);
        using I = typename multi_scale::SIMD<Lanes>::I;
        if constexpr (Lanes == 8)
        {
            // 2*31*128 = 7936, so VPMADDUBSW never saturates, including for -128 activations.
            const auto pair = _mm256_maddubs_epi16(std::bit_cast<__m256i>(weights), _mm256_set1_epi32(word));
            return acc + std::bit_cast<I>(_mm256_madd_epi16(pair, _mm256_set1_epi16(1)));
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
            return multi_scale::SIMD<16>::dot(acc, weights, std::bit_cast<I>(_mm512_set1_epi32(word)));
#endif
    }

    /** @brief Sum all original Q8 bytes for the Q5_0 zero-point correction.
     * @param qs Full 32-byte signed activation payload (including -128).
     * @return The exact signed sum, independently of any staged FP32 metadata.
     */
    [[nodiscard]] inline int sumQ8(const std::int8_t *qs) noexcept
    {
        const auto input = multi_scale::load<__m256i>(qs);
        const auto biased = _mm256_xor_si256(input, _mm256_set1_epi8(-128));
        const auto sums = _mm256_sad_epu8(biased, _mm256_setzero_si256());
        const auto pairs = _mm_add_epi64(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
        return _mm_cvtsi128_si32(_mm_add_epi64(pairs, _mm_srli_si128(pairs, 8))) - 32 * 128;
    }

    /** @brief Apply the unchanged FP32 contribution program on either x86 width. */
    template <int Lanes, bool Asymmetric>
    [[nodiscard]] inline typename multi_scale::SIMD<Lanes>::F contribute(
        typename multi_scale::SIMD<Lanes>::F acc, typename multi_scale::SIMD<Lanes>::I dot,
        const std::uint16_t *d, const std::uint16_t *m, const Q8_1Block &a,
        CPUProjectionNumericalPolicy policy) noexcept
    {
        using F = typename multi_scale::SIMD<Lanes>::F;
        const float a_scale = nativeQuantisedFP16ScaleToFP32(a.d);
        if constexpr (!Asymmetric)
            dot -= 16 * sumQ8(a.qs);
        if constexpr (Lanes == 8)
        {
            const auto scale = _mm256_cvtph_ps(multi_scale::load<__m128i>(d));
            if constexpr (Asymmetric)
                return std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX2(std::bit_cast<__m256>(acc),
                    std::bit_cast<__m256i>(dot), scale, a.sum_qs, _mm256_cvtph_ps(multi_scale::load<__m128i>(m)),
                    a_scale, policy));
            return std::bit_cast<F>(accumulateSingleScaleBlockAVX2(std::bit_cast<__m256>(acc),
                std::bit_cast<__m256i>(dot), scale, a_scale, policy));
        }
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
        else
        {
            const auto scale = _mm512_cvtph_ps(multi_scale::load<__m256i>(d));
            if constexpr (Asymmetric)
                return std::bit_cast<F>(accumulateCorrectedSingleScaleBlockAVX512(std::bit_cast<__m512>(acc),
                    std::bit_cast<__m512i>(dot), scale, a.sum_qs, _mm512_cvtph_ps(multi_scale::load<__m256i>(m)),
                    a_scale, policy));
            return std::bit_cast<F>(accumulateSingleScaleBlockAVX512(std::bit_cast<__m512>(acc),
                std::bit_cast<__m512i>(dot), scale, a_scale, policy));
        }
#endif
    }

    /** @brief Accumulate a bounded K range while sharing one weight decode across live rows.
     * @tparam Lanes Physical AVX2 or AVX-512 output width.
     * @tparam Asymmetric Q5_1 minimum term, instead of Q5_0 centering.
     * @tparam SumPairs Publish the existing two-row fused epilogue sum in the corresponding even row.
     * @tparam Rows Number of live activation and output rows.
     * @param packed Admitted native Q5 publication.
     * @param input Activation rows with complete Q8 blocks.
     * @param output Writable physical chunk per row; callers own any N-tail handling.
     * @param chunk Physical 64-column output chunk.
     * @param first First logical K block in this range.
     * @param last One past the last block in this range.
     * @param accumulate Read existing output accumulators instead of starting from positive zero.
     */
    template <int Lanes, bool Asymmetric, bool SumPairs, std::size_t Rows>
    inline void project(const CPUNativeQuantisedPackedWeights &packed,
        const std::array<const Q8_1Block *, Rows> &input, const std::array<float *, Rows> &output,
        int chunk, int first, int last, bool accumulate)
    {
        static_assert(!SumPairs || Rows % 2 == 0);
        using Ops = multi_scale::SIMD<Lanes>;
        using I = typename Ops::I;
        using F = typename Ops::F;
        for (int column = 0; column < 64; column += Lanes)
        {
            F partial[Rows]{};
            if (accumulate)
                for (std::size_t row = 0; row < Rows; ++row)
                    partial[row] = multi_scale::load<F>(output[row] + column);
            for (int kb = first; kb < last; ++kb)
            {
                const auto *unit = packed.admittedBlockUnit(chunk, kb);
                I dots[Rows]{};
#if defined(__clang__)
#pragma clang loop unroll(disable)
#else
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 4; ++group)
                {
                    const auto [low, high] = [&] {
                        if constexpr (Lanes == 8) return decode8(unit, column, group);
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
                        else return decode16(unit, column, group);
#endif
                    }();
                    for (std::size_t row = 0; row < Rows; ++row)
                    {
                        dots[row] = dot<Lanes>(dots[row], low, input[row][kb], group);
                        dots[row] = dot<Lanes>(dots[row], high, input[row][kb], group + 4);
                    }
                }
                // This leaf is bound to Q5NativeSingleScale. Its inline half
                // planes immediately follow the payload; reuse the hot unit
                // pointer for both rather than resolving the layout again.
                const auto *d = reinterpret_cast<const std::uint16_t *>(unit + Q5NativePayload::bytes) + column;
                const auto *m = Asymmetric ? d + 64 : nullptr;
                for (std::size_t row = 0; row < Rows; ++row)
                    partial[row] = contribute<Lanes, Asymmetric>(partial[row], dots[row], d, m,
                        input[row][kb], packed.numerical_policy);
            }
            if constexpr (SumPairs)
            {
                for (std::size_t row = 0; row < Rows; row += 2)
                {
                    const F sum = partial[row] + partial[row + 1];
                    std::memcpy(output[row] + column, &sum, sizeof(F));
                }
            }
            else
                for (std::size_t row = 0; row < Rows; ++row)
                    std::memcpy(output[row] + column, &partial[row], sizeof(F));
        }
    }

    /** @brief Select symmetric or asymmetric Q5 once before the K traversal. */
    template <int Lanes, bool SumPairs = false, std::size_t Rows>
    inline void dispatch(const CPUNativeQuantisedPackedWeights &packed,
        const std::array<const Q8_1Block *, Rows> &input, const std::array<float *, Rows> &output,
        int chunk, int first, int last, bool accumulate)
    {
        if (packed.is_asymmetric)
            project<Lanes, true, SumPairs>(packed, input, output, chunk, first, last, accumulate);
        else
            project<Lanes, false, SumPairs>(packed, input, output, chunk, first, last, accumulate);
    }
}
