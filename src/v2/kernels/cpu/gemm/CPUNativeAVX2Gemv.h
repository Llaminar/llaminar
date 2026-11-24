/**
 * @file CPUNativeAVX2Gemv.h
 * @brief AVX2 GEMV/GEMM kernels using saturation-safe emulated VNNI.
 *
 * These kernels read the SAME packed weight format as the AVX512 VNNI kernels
 * (CPUNativeQuantisedGemv.h). Each 64-byte ZMM slot is processed as two 32-byte
 * YMM halves, and 8 YMM accumulators cover the same 64 columns that 4 ZMM
 * accumulators cover in AVX512.
 * GPU-aligned K folds retain the original serial NaN operand precedence as
 * well as the fixed finite arithmetic tree; native FMA paths remain unchanged.
 *
 * Both AVX512 and AVX2 paths are always compiled (no #ifdef gates) so they
 * can be tested against each other on AVX512 hardware via runtime dispatch.
 *
 * Performance note: AVX2 processes half the data per instruction vs AVX512,
 * so expect ~1.5-2× slower throughput. The packed weight layout is unchanged.
 */

#pragma once

#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

#include "CPUNativeQuantisedWeightPacker.h"
#include "CPUNativeQuantisedContributionContract.h"
#include "CPUNativeQuantisedFP16.h"
#include "CPUNativeQuantisedQ6Kernels.h"
#include "CPUNativeQuantisedQ5X86Kernels.h"
#include "CPUNativeQuantisedMultiScaleKernels.h"
#include "VNNIEmulation.h"
#include "tensors/BlockStructures.h"
#include "tensors/SIMDHelpers.h"

namespace llaminar2::cpu::native_quantised
{

    // =========================================================================
    // Decode LUT tables for nibble → signed int8 mapping
    // =========================================================================
    // Only nonlinear codebooks consume a lookup. Q4_0 and Q4_1/Q4_K
    // decode directly to their unsigned nibble indices.
    // =========================================================================

    namespace avx2_luts
    {
        // IQ4_NL: nibble → non-linear codebook
        alignas(16) static constexpr int8_t IQ4_NL_DECODE_LUT[16] = {
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
        // MXFP4: nibble → doubled E2M1 value (mxfp4::kValues)
        alignas(16) static constexpr int8_t MXFP4_DECODE_LUT[16] = {
            0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    } // namespace avx2_luts

    /** @brief Choose the established unsigned zero point of the AVX-512 nibble decoder.
     * @param codebook Original nibble codec (5 includes both Q4_1 and Q4_K).
     * @return z such that the unsigned decoder publishes w + z.
     * @details The previous sum((a+128)*w) - 128*sum(w) and the new
     * sum(a*(w+z)) - z*sum(a) are both exactly sum(a*w) in INT32. At
     * 32 terms, the biased dot has magnitude at most 32*128*255.
     */
    inline constexpr int nibbleWeightBias(uint8_t codebook) noexcept
    {
        return codebook == 5 ? 0 : codebook == 0 ? 8 : 128;
    }

    /** @return Four signed activation bytes, in native dot order. */
    inline int32_t nativeActivationWord(const Q8_1Block &block, int first) noexcept
    {
        int32_t word;
        std::memcpy(&word, block.qs + first, sizeof(word));
        return word;
    }

    /** @brief Exact dot of unsigned weights and signed activations without pair saturation.
     * @details Splitting off bit 7 bounds both maddubs pair sums to int16, even
     * for a=-128 and w=255. Widen each sum before recombining the two parts.
     */
    inline __m256i dotUnsignedWeightsAVX2(__m256i dot, __m256i weight, __m256i activation) noexcept
    {
        const auto high_bit = _mm256_set1_epi8(-128);
        const auto ones = _mm256_set1_epi16(1);
        const auto low = _mm256_maddubs_epi16(_mm256_andnot_si256(high_bit, weight), activation);
        const auto high = _mm256_maddubs_epi16(_mm256_and_si256(high_bit, weight), activation);
        return _mm256_add_epi32(dot, _mm256_add_epi32(_mm256_madd_epi16(low, ones), _mm256_madd_epi16(high, ones)));
    }

    /** @brief Integer domain of an admitted AVX2 nibble codebook. */
    enum class NibbleDotAVX2 { Linear, SmallLUT, SignedLUT };

    /** @brief Select the AVX2 dot from the immutable source codebook.
     * @param codebook Original codebook retained in admitted nibble weights.
     * @return Domain of the decoded bytes feeding the exact integer dot.
     * @throws std::invalid_argument If a non-nibble codebook reached this leaf.
     */
    inline NibbleDotAVX2 nibbleDotAVX2(uint8_t codebook)
    {
        switch (codebook)
        {
        case native_quantised_formats::Q4_0.codebook_id:
        case native_quantised_formats::Q4_1.codebook_id:
            return NibbleDotAVX2::Linear;
        case kNativeQuantisedMXFP4Codebook:
            return NibbleDotAVX2::SmallLUT;
        case native_quantised_formats::IQ4_NL.codebook_id:
            return NibbleDotAVX2::SignedLUT;
        }
        throw std::invalid_argument("NativeQuantised AVX2 nibble dot requires a nibble codebook");
    }

    /**
     * @brief Append a nibble dot without splitting even the nonlinear IQ4 weights.
     * @tparam Domain Whether decoded bytes are small unsigned or native signed IQ4.
     * @param dot Register chain for at most four groups of one nibble plane.
     * @param weight Decoded nibble values in VNNI lane order.
     * @param activation Four signed activation bytes broadcast to each column.
     * @return Two INT16 pair sums per column, or a single INT32 sum for signed IQ4.
     *
     * Linear and offset MXFP4 weights are bounded by 15 and 24. Four groups
     * contribute eight products per INT16 lane, bounded by 8*24*128=24576.
     * IQ4's signed range [-127,113] excludes -128, so SIMD<8>::dot transfers
     * the activation sign exactly but widens each group before accumulating.
     */
    template <NibbleDotAVX2 Domain>
    inline __m256i dotNibbleAVX2(__m256i dot, __m256i weight, __m256i activation) noexcept
    {
        if constexpr (Domain == NibbleDotAVX2::SignedLUT)
        {
            using Ops = multi_scale::SIMD<8>;
            using I = Ops::I;
            return std::bit_cast<__m256i>(Ops::dot(std::bit_cast<I>(dot), std::bit_cast<I>(weight),
                std::bit_cast<I>(activation)));
        }
        else
            return _mm256_add_epi16(dot, _mm256_maddubs_epi16(weight, activation));
    }

    /** @brief Finish the exact 32-term dot without narrowing a nonlinear codebook's sum.
     * Linear nibbles can merge their two chains before widening (16*15*128 < 32768).
     * Offset MXFP4 instead widens each chain independently before the merge.
     */
    template <NibbleDotAVX2 Domain>
    inline __m256i finishNibbleDotAVX2(__m256i low, __m256i high) noexcept
    {
        if constexpr (Domain == NibbleDotAVX2::Linear)
            return _mm256_madd_epi16(_mm256_add_epi16(low, high), _mm256_set1_epi16(1));
        if constexpr (Domain == NibbleDotAVX2::SmallLUT)
        {
            const auto ones = _mm256_set1_epi16(1);
            return _mm256_add_epi32(_mm256_madd_epi16(low, ones), _mm256_madd_epi16(high, ones));
        }
        return _mm256_add_epi32(low, high);
    }

    /** @brief Decode one nibble per byte after low/high extraction.
     * @tparam Domain Linear codebook or a nonlinear lookup program.
     * @param indices Values in [0,15] in prepared column order.
     * @param lut Immutable byte lookup table for nonlinear codebooks.
     * @return The original indices for linear codebooks or decoded LUT bytes.
     */
    template <NibbleDotAVX2 Domain>
    inline __m256i decodeNibbleAVX2(__m256i indices, __m256i lut) noexcept
    {
        if constexpr (Domain == NibbleDotAVX2::Linear)
            return indices;
        else
            return _mm256_shuffle_epi8(lut, indices);
    }

    /** @return Activation-sum multiplier needed by the AVX2 integer operand domain. */
    template <NibbleDotAVX2 Domain>
    inline int nibbleWeightBiasAVX2(uint8_t codebook) noexcept
    {
        if constexpr (Domain == NibbleDotAVX2::Linear)
            return codebook == native_quantised_formats::Q4_0.codebook_id ? 8 : 0;
        if constexpr (Domain == NibbleDotAVX2::SmallLUT)
            return 12;
        return 0;
    }

    /** @brief Dot one native signed Q8 group using a recentered unsigned weight operand. */
    inline __m256i dotSignedQ8WeightsAVX2(__m256i dot, __m256i weights, const Q8_1Block &activation,
                                          int first) noexcept
    {
        auto a = nativeActivationWord(activation, first);
        return dotUnsignedWeightsAVX2(dot, _mm256_xor_si256(weights, _mm256_set1_epi8(-128)), _mm256_set1_epi32(a));
    }

    /** @brief Exact signed Q8 dot, with two bounded INT16 planes per row.
     * Split the unsigned form of each signed weight into two nibbles. Each
     * plane accumulates 16 products per pair lane in [-30720,30480] before
     * widening once. Multiplying the upper dot by 16 and subtracting the
     * usual activation-side bias recovers the signed result in INT32.
     * @param weights First group of the selected eight-column strip in an admitted block.
     * @param activation Independently owned activation blocks for the live rows.
     * @return One signed INT32 dot vector per row, in the original column order.
     */
    template <std::size_t Rows>
    inline auto dotSignedQ8BlockAVX2(const uint8_t *weights,
        const std::array<const Q8_1Block *, Rows> &activation) noexcept
    {
        static_assert(Rows == 1 || Rows == 2);
        struct Dots { __m256i values[Rows]; };
        Dots low{}, high{};
        const auto mask = _mm256_set1_epi8(15);
        const auto bias = _mm256_set1_epi8(8);
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
        for (int g = 0; g < 8; ++g)
        {
            const auto raw = _mm256_load_si256(reinterpret_cast<const __m256i *>(weights + g * 256));
            const auto lo = _mm256_and_si256(raw, mask);
            const auto hi = _mm256_xor_si256(_mm256_and_si256(_mm256_srli_epi16(raw, 4), mask), bias);
#pragma GCC unroll 2
            for (std::size_t row = 0; row < Rows; ++row)
            {
                const auto a = _mm256_set1_epi32(nativeActivationWord(*activation[row], g * 4));
                low.values[row] = _mm256_add_epi16(low.values[row], _mm256_maddubs_epi16(lo, a));
                high.values[row] = _mm256_add_epi16(high.values[row], _mm256_maddubs_epi16(hi, a));
            }
        }
        Dots result;
        const auto ones = _mm256_set1_epi16(1);
#pragma GCC unroll 2
        for (std::size_t row = 0; row < Rows; ++row)
        {
            const auto lo = _mm256_madd_epi16(low.values[row], ones);
            const auto hi = _mm256_slli_epi32(_mm256_madd_epi16(high.values[row], ones), 4);
            result.values[row] = _mm256_sub_epi32(_mm256_add_epi32(lo, hi),
                _mm256_set1_epi32(128 * activation[row]->sum_qs));
        }
        return result;
    }

    /** @brief Project one symmetric Q8 chunk with a shared block dot.
     * The output accumulator for each row visits K in the identical serial
     * order and uses the original prepared FP scale and policy.
     * @tparam Rows Number of independently accumulated output rows.
     * @tparam SumPairs Publish only the sum of two final output rows.
     * @param packed Admitted signed Q8 stream.
     * @param a,c Activation and output pointers in original row order.
     * @param chunk,kb_start,kb_end Original chunk and half-open block interval.
     * @param accumulate Whether the destinations own a previous FP partial.
     */
    template <std::size_t Rows, bool SumPairs = false>
    inline void projectSignedQ8AVX2(const CPUNativeQuantisedPackedWeights &packed,
        const std::array<const Q8_1Block *, Rows> &a, const std::array<float *, Rows> &c,
        int chunk, int kb_start, int kb_end, bool accumulate)
    {
        static_assert(!SumPairs || Rows == 2);
        for (int column = 0; column < 64; column += 8)
        {
            __m256 fp[Rows]{};
#pragma GCC unroll 2
            for (std::size_t row = 0; accumulate && row < Rows; ++row)
                fp[row] = _mm256_loadu_ps(c[row] + column);
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                std::array<const Q8_1Block *, Rows> block;
#pragma GCC unroll 2
                for (std::size_t row = 0; row < Rows; ++row)
                    block[row] = a[row] + kb;
                const auto dots = dotSignedQ8BlockAVX2(packed.admittedBlockUnit(chunk, kb) + 4 * column, block);
                const auto scale = packed.loadScales256(packed.chunkScales(chunk, kb) + column);
#pragma GCC unroll 2
                for (std::size_t row = 0; row < Rows; ++row)
                    fp[row] = accumulateSingleScaleBlockAVX2(fp[row], dots.values[row], scale,
                        nativeQuantisedFP16ScaleToFP32(block[row]->d), packed.numerical_policy);
            }
            if constexpr (SumPairs)
                _mm256_storeu_ps(c[0] + column, _mm256_add_ps(fp[0], fp[1]));
            else
            {
#pragma GCC unroll 2
                for (std::size_t row = 0; row < Rows; ++row)
                    _mm256_storeu_ps(c[row] + column, fp[row]);
            }
        }
    }

    /** @return Recenter the unsigned Q8 weight operand using only the activation sum. */
    inline __m256i q8WeightBiasAVX2(const Q8_1Block &activation) noexcept
    { return _mm256_set1_epi32(128 * activation.sum_qs); }

    // =========================================================================
    // AVX2 decode LUT builder (selects correct LUT by codebook_id)
    // =========================================================================

    /** @brief Publish the AVX2 operand table associated with a nibble domain.
     * @param codebook_id Original nibble codebook, including its source aliases.
     * @return Native signed IQ4 table, offset MXFP4 table, or unused zero for linear decode.
     */
    inline __m256i build_decode_lut_avx2_for_codebook(uint8_t codebook_id)
    {
        switch (nibbleDotAVX2(codebook_id))
        {
        case NibbleDotAVX2::Linear:
            return _mm256_setzero_si256();
        case NibbleDotAVX2::SmallLUT:
            return _mm256_add_epi8(isa::build_decode_lut_avx2(avx2_luts::MXFP4_DECODE_LUT),
                _mm256_set1_epi8(nibbleWeightBiasAVX2<NibbleDotAVX2::SmallLUT>(codebook_id)));
        case NibbleDotAVX2::SignedLUT:
            return isa::build_decode_lut_avx2(avx2_luts::IQ4_NL_DECODE_LUT);
        }
        throw std::logic_error("Invalid NativeQuantised AVX2 nibble dot domain");
    }

    /** @brief Decode nibble factors in the caller's registers, without a hidden result buffer.
     * @tparam Domain Admitted nibble family which determines the possible factor layouts.
     * @param packed Prepared nibble matrix owning the source factor representation.
     * @param chunk Physical 64-column chunk.
     * @param block Logical 32-value reduction block.
     * @param column First of eight SIMD columns.
     * @return Exact primary and secondary factor lanes from the admitted representation.
     */
    template <NibbleDotAVX2 Domain>
    [[gnu::always_inline]] inline coded_metadata::Factors<__m256> loadNibbleFactors256(
        const CPUNativeQuantisedPackedWeights &packed, int chunk, int block, int column)
    {
        if constexpr (Domain == NibbleDotAVX2::Linear)
        {
            if (packed.encoding == CPUNativeQuantisedEncoding::Q4KSuperblockCoded)
            {
                using Ops = multi_scale::SIMD<8>;
                using L = coded_metadata::Lanes<Ops>;
                return coded_metadata::reconstruct<5, Ops>(
                    L::code<5>(packed.admittedBlockUnit(chunk, block) + packed.data_stride, column),
                    L::superblockWords<5>(packed.admittedSuperblockHeader(chunk, block), column));
            }
        }
        if constexpr (Domain == NibbleDotAVX2::SmallLUT)
        {
            const auto e = _mm_loadu_si64(packed.admittedBlockUnit(chunk, block) + packed.data_stride + column);
            const auto biased = _mm_subs_epu8(e, _mm_set1_epi8(1));
            return {.primary = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu8_epi32(biased), 23))};
        }
        else
        {
            if constexpr (Domain == NibbleDotAVX2::SignedLUT)
            {
                if (packed.usesNibbleLUTSourceScale())
                    return {.primary = packed.loadIQ4XSCodedScales<8>(chunk, block, column)};
            }

            // The remaining ordinary nibble encodings have adjacent half
            // planes. The dispatch has already excluded every coded layout.
            const auto *scale = reinterpret_cast<const uint16_t *>(
                packed.admittedBlockUnit(chunk, block) + packed.data_stride) + column;
            return {.primary = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(scale))),
                    .secondary = packed.is_asymmetric ? _mm256_cvtph_ps(_mm_loadu_si128(
                        reinterpret_cast<const __m128i *>(scale + 64))) : _mm256_setzero_ps()};
        }
    }

    static_assert(native_quantised_formats::Q4_K.codebook_id == native_quantised_formats::Q4_1.codebook_id);
    static_assert(native_quantised_formats::IQ4_XS.codebook_id == native_quantised_formats::IQ4_NL.codebook_id);
    static_assert([] {
        for (int value : avx2_luts::IQ4_NL_DECODE_LUT)
            if (value == -128) return false;
        for (int value : avx2_luts::MXFP4_DECODE_LUT)
            if (value < -12 || value > 12) return false;
        return true;
    }(), "Native nibble tables must fit their exact AVX2 dot domains");

    // =========================================================================
    // AVX2 GEMV: nibble path (Q4_0, IQ4_NL, Q4_1/Q4_K, IQ4_XS)
    // =========================================================================
    //
    // Processes one 64-column N-chunk using 8 YMM accumulators (8 cols each).
    // Each AVX512 ZMM load (64 bytes) is split into two YMM loads (32 bytes).
    // The nibble decode and VNNI accumulation use AVX2 equivalents.
    // =========================================================================

    /**
     * @brief Accumulate one eight-column AVX2 nibble segment over a K range.
     *
     * This is the single nibble dot/scale implementation shared by ordinary
     * GEMV and the device-ordered ExpertOverlay tree. Keeping the segment in a
     * register lets the latter reset and fold exact K partitions without a
     * scratch plane or an alternate arithmetic body.
     *
     * @param packed Prepared nibble matrix.
     * @param A_q8 Q8_1 activation row.
     * @param chunk Sixty-four-column output chunk.
     * @param kb_start First K block, inclusive.
     * @param kb_end Final K block, exclusive.
     * @param decode_lut Broadcast nibble codebook.
     * @param segment Eight-column segment within the chunk.
     * @param accumulator Initial FP32 value for the segment.
     * @return Updated FP32 segment.
     * @tparam Domain Exact integer operand domain of the prepared nibble weights.
     */
    template <NibbleDotAVX2 Domain>
    inline __m256 accumulateNibbleSegmentAVX2Impl(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        int segment,
        __m256 accumulator)
    {
        const int weight_bias = nibbleWeightBiasAVX2<Domain>(packed.codebook_id);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
        const int z = segment / 2;
        const int byte_offset = (segment % 2) * 32;

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
            const auto *unit = packed.admittedBlockUnit(chunk, kb);
            const Q8_1Block &activation = A_q8[kb];
            __m256i dot_low = _mm256_setzero_si256();
            __m256i dot_high = _mm256_setzero_si256();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
            for (int group = 0; group < 4; ++group)
            {
                const uint8_t *base =
                    unit + group * 256 + z * 64 + byte_offset;
                const __m256i raw = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i *>(base));
                const __m256i low = decodeNibbleAVX2<Domain>(
                    _mm256_and_si256(raw, mask_0F), decode_lut);
                const __m256i high = decodeNibbleAVX2<Domain>(
                    _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_0F), decode_lut);
                dot_low = dotNibbleAVX2<Domain>(dot_low, low, _mm256_set1_epi32(nativeActivationWord(activation, group * 4)));
                dot_high = dotNibbleAVX2<Domain>(dot_high, high, _mm256_set1_epi32(nativeActivationWord(activation, group * 4 + 16)));
            }

            const __m256i corrected = _mm256_sub_epi32(
                finishNibbleDotAVX2<Domain>(dot_low, dot_high),
                _mm256_set1_epi32(weight_bias * activation.sum_qs));
            const auto factors = loadNibbleFactors256<Domain>(packed, chunk, kb, segment * 8);
            const float activation_scale =
                nativeQuantisedFP16ScaleToFP32(activation.d);
            if (packed.is_asymmetric)
            {
                accumulator = accumulateCorrectedSingleScaleBlockAVX2(
                    accumulator,
                    corrected,
                    factors.primary,
                    activation.sum_qs,
                    factors.secondary,
                    activation_scale,
                    packed.numerical_policy);
            }
            else
            {
                accumulator = accumulateSingleScaleBlockAVX2(
                    accumulator,
                    corrected,
                    factors.primary,
                    activation_scale,
                    packed.numerical_policy);
            }
        }
        return accumulator;
    }

    /** @brief Select the immutable nibble domain once before the K traversal. */
    inline __m256 accumulateNibbleSegmentAVX2(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        int segment,
        __m256 accumulator)
    {
        switch (nibbleDotAVX2(packed.codebook_id))
        {
        case NibbleDotAVX2::Linear:
            return accumulateNibbleSegmentAVX2Impl<NibbleDotAVX2::Linear>(packed, A_q8, chunk, kb_start, kb_end, decode_lut, segment, accumulator);
        case NibbleDotAVX2::SmallLUT:
            return accumulateNibbleSegmentAVX2Impl<NibbleDotAVX2::SmallLUT>(packed, A_q8, chunk, kb_start, kb_end, decode_lut, segment, accumulator);
        case NibbleDotAVX2::SignedLUT:
            return accumulateNibbleSegmentAVX2Impl<NibbleDotAVX2::SignedLUT>(packed, A_q8, chunk, kb_start, kb_end, decode_lut, segment, accumulator);
        }
        throw std::logic_error("Invalid NativeQuantised AVX2 nibble dot domain");
    }

    /**
     * @brief Execute one serial-shaped AVX2 nibble chunk over a K range.
     *
     * @param accumulate Load `C` as the initial FP32 accumulator. Tile callers
     *        use this mode to materialize cache boundaries without changing
     *        the serial per-K-block reduction tree.
     */
    inline void gemv_avx2_chunk_native(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        bool accumulate = false)
    {
        for (int segment = 0; segment < 8; ++segment)
        {
            const __m256 initial = accumulate
                ? _mm256_loadu_ps(C + segment * 8)
                : _mm256_setzero_ps();
            _mm256_storeu_ps(
                C + segment * 8,
                accumulateNibbleSegmentAVX2(
                    packed,
                    A_q8,
                    chunk,
                    kb_start,
                    kb_end,
                    decode_lut,
                    segment,
                    initial));
        }
    }

    // =========================================================================
    // AVX2 GEMV: non-nibble prepared encodings
    // =========================================================================

    /**
     * @brief Dispatch one non-nibble AVX2 chunk by its typed prepared encoding.
     *
     * Q6_K consumes its native dual-scale payload. Remaining codebooks use the
     * signed Q8 encoding below. Keeping the decision here makes
     * every serial and K-partition caller exhaustive over the packed ABI.
     *
     * @param accumulate Load `C` as the initial FP32 accumulator. The flag is
     *        forwarded to native Q6 as well as the signed Q8 path so every
     *        prepared encoding has identical tile-boundary semantics.
     */
    inline void gemv_avx2_chunk_non_nibble(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate = false)
    {
        if (packed.encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale)
            return q5::dispatch<8>(packed, std::array{A_q8}, std::array{C},
                chunk, kb_start, kb_end, accumulate);
        if (packed.usesCompactMultiScale())
        {
            multi_scale::dispatch<8>(packed, std::array{A_q8}, std::array{C},
                                     chunk, kb_start, kb_end, accumulate);
            return;
        }
        if (packed.usesQ6KNativeDualScale())
        {
            gemvQ6KNativeAVX2Chunk(
                packed, A_q8, C, chunk, kb_start, kb_end, accumulate);
            return;
        }
        if (!packed.usesSignedInt8SingleScale())
            throw std::invalid_argument(
                "AVX2 non-nibble GEMV received an unsupported packed encoding");
        if (packed.codebook_id == native_quantised_formats::Q8_0.codebook_id)
            return projectSignedQ8AVX2(packed, std::array{A_q8}, std::array{C}, chunk, kb_start, kb_end, accumulate);
        for (int segment = 0; segment < 8; ++segment)
        {
            const int z = segment / 2;
            const int byte_offset = (segment % 2) * 32;
            __m256 fp_accumulator = accumulate
                ? _mm256_loadu_ps(C + segment * 8)
                : _mm256_setzero_ps();
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &activation = A_q8[kb];
                __m256i dot_even = _mm256_setzero_si256();
                __m256i dot_odd = _mm256_setzero_si256();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 8; group += 2)
                {
                    const __m256i weight_even = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group, z) + byte_offset));
                    const __m256i weight_odd = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group + 1, z) + byte_offset));
                    dot_even = dotSignedQ8WeightsAVX2(dot_even, weight_even, activation, group * 4);
                    dot_odd = dotSignedQ8WeightsAVX2(dot_odd, weight_odd, activation, (group + 1) * 4);
                }

                const __m256i corrected = _mm256_sub_epi32(
                    _mm256_add_epi32(dot_even, dot_odd),
                    q8WeightBiasAVX2(activation));
                const __m256 weight_scale = packed.loadScales256(packed.chunkScales(chunk, kb) + segment * 8);
                const float activation_scale =
                    nativeQuantisedFP16ScaleToFP32(activation.d);
                fp_accumulator = accumulateSingleScaleBlockAVX2(
                    fp_accumulator,
                    corrected,
                    weight_scale,
                    activation_scale,
                    packed.numerical_policy);
            }
            _mm256_storeu_ps(C + segment * 8, fp_accumulator);
        }
    }

    // =========================================================================
    // AVX2 multi-chunk GEMV block (processes consecutive 64-column chunks)
    // =========================================================================

    inline void gemv_avx2_block(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8,
        float *C,
        int chunk_start,
        int chunk_count,
        int K_blocks,
        int N,
        const __m256i decode_lut)
    {
        const bool use_nibble_lut = packed.usesNibbleLUT();

        for (int ci = 0; ci < chunk_count; ++ci)
        {
            int chunk = chunk_start + ci;
            int n_start = chunk * 64;
            int n_cols = std::min(64, N - n_start);

            if (n_cols < 64)
            {
                alignas(64) float tmp[64];
                if (use_nibble_lut)
                    gemv_avx2_chunk_native(packed, A_q8, tmp, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_avx2_chunk_non_nibble(packed, A_q8, tmp, chunk, 0, K_blocks);
                std::memcpy(C + n_start, tmp, n_cols * sizeof(float));
            }
            else
            {
                if (use_nibble_lut)
                    gemv_avx2_chunk_native(packed, A_q8, C + n_start, chunk, 0, K_blocks, decode_lut);
                else
                    gemv_avx2_chunk_non_nibble(packed, A_q8, C + n_start, chunk, 0, K_blocks);
            }
        }
    }

    // =========================================================================
    // AVX2 2-Row GEMM Microkernels (share B loads across 2 M rows)
    // =========================================================================

    /**
     * @brief Accumulate one shared-decode AVX2 segment for two rows.
     *
     * This register-returning primitive owns the two-row nibble dot and scale
     * program for both ordinary grouped execution and the exact device K tree.
     * The caller supplies each row's initial value, allowing a partition to
     * begin at positive zero without materializing a scratch plane.
     *
     * @param packed Prepared nibble matrix.
     * @param A_q8_row0 First Q8_1 activation row.
     * @param A_q8_row1 Second Q8_1 activation row.
     * @param chunk Sixty-four-column output chunk.
     * @param kb_start First K block, inclusive.
     * @param kb_end Final K block, exclusive.
     * @param decode_lut Broadcast nibble codebook.
     * @param segment Eight-column segment within the chunk.
     * @param fp0 Running first-row FP32 value, updated in place.
     * @param fp1 Running second-row FP32 value, updated in place.
     * @tparam Domain Exact integer operand domain of the prepared nibble weights.
     */
    template <NibbleDotAVX2 Domain>
    inline void accumulateTwoRowsNibbleSegmentAVX2Impl(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        int segment,
        __m256 &fp0,
        __m256 &fp1)
    {
        const int weight_bias = nibbleWeightBiasAVX2<Domain>(packed.codebook_id);
        const __m256i mask_0F = _mm256_set1_epi8(0x0F);
        const int z = segment / 2;
        const int byte_offset = (segment % 2) * 32;

        for (int kb = kb_start; kb < kb_end; ++kb)
        {
                const auto *unit = packed.admittedBlockUnit(chunk, kb);
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                __m256i dot0_low = _mm256_setzero_si256();
                __m256i dot0_high = _mm256_setzero_si256();
                __m256i dot1_low = _mm256_setzero_si256();
                __m256i dot1_high = _mm256_setzero_si256();

#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 4; ++group)
                {
                    const uint8_t *base =
                        unit + group * 256 + z * 64 + byte_offset;
                    const __m256i raw = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i *>(base));
                    const __m256i low = decodeNibbleAVX2<Domain>(
                        _mm256_and_si256(raw, mask_0F), decode_lut);
                    const __m256i high = decodeNibbleAVX2<Domain>(
                        _mm256_and_si256(_mm256_srli_epi16(raw, 4), mask_0F), decode_lut);
                    dot0_low = dotNibbleAVX2<Domain>(dot0_low, low, _mm256_set1_epi32(nativeActivationWord(a0, group * 4)));
                    dot1_low = dotNibbleAVX2<Domain>(dot1_low, low, _mm256_set1_epi32(nativeActivationWord(a1, group * 4)));
                    dot0_high = dotNibbleAVX2<Domain>(dot0_high, high, _mm256_set1_epi32(nativeActivationWord(a0, group * 4 + 16)));
                    dot1_high = dotNibbleAVX2<Domain>(dot1_high, high, _mm256_set1_epi32(nativeActivationWord(a1, group * 4 + 16)));
                }

                const __m256i corrected0 = _mm256_sub_epi32(
                    finishNibbleDotAVX2<Domain>(dot0_low, dot0_high), _mm256_set1_epi32(weight_bias * a0.sum_qs));
                const __m256i corrected1 = _mm256_sub_epi32(
                    finishNibbleDotAVX2<Domain>(dot1_low, dot1_high), _mm256_set1_epi32(weight_bias * a1.sum_qs));
                const auto factors = loadNibbleFactors256<Domain>(packed, chunk, kb, segment * 8);
                const float a0_scale = nativeQuantisedFP16ScaleToFP32(a0.d);
                const float a1_scale = nativeQuantisedFP16ScaleToFP32(a1.d);
                if (packed.is_asymmetric)
                {
                    fp0 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp0, corrected0, factors.primary, a0.sum_qs,
                        factors.secondary, a0_scale, packed.numerical_policy);
                    fp1 = accumulateCorrectedSingleScaleBlockAVX2(
                        fp1, corrected1, factors.primary, a1.sum_qs,
                        factors.secondary, a1_scale, packed.numerical_policy);
                }
                else
                {
                    fp0 = accumulateSingleScaleBlockAVX2(
                        fp0, corrected0, factors.primary, a0_scale,
                        packed.numerical_policy);
                    fp1 = accumulateSingleScaleBlockAVX2(
                        fp1, corrected1, factors.primary, a1_scale,
                        packed.numerical_policy);
                }
        }
    }

    /** @brief Select the immutable nibble domain once before the K traversal. */
    inline void accumulateTwoRowsNibbleSegmentAVX2(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        int segment,
        __m256 &fp0,
        __m256 &fp1)
    {
        switch (nibbleDotAVX2(packed.codebook_id))
        {
        case NibbleDotAVX2::Linear:
            return accumulateTwoRowsNibbleSegmentAVX2Impl<NibbleDotAVX2::Linear>(packed, A_q8_row0, A_q8_row1, chunk, kb_start, kb_end, decode_lut, segment, fp0, fp1);
        case NibbleDotAVX2::SmallLUT:
            return accumulateTwoRowsNibbleSegmentAVX2Impl<NibbleDotAVX2::SmallLUT>(packed, A_q8_row0, A_q8_row1, chunk, kb_start, kb_end, decode_lut, segment, fp0, fp1);
        case NibbleDotAVX2::SignedLUT:
            return accumulateTwoRowsNibbleSegmentAVX2Impl<NibbleDotAVX2::SignedLUT>(packed, A_q8_row0, A_q8_row1, chunk, kb_start, kb_end, decode_lut, segment, fp0, fp1);
        }
        throw std::logic_error("Invalid NativeQuantised AVX2 nibble dot domain");
    }

    /**
     * @brief Register-resident AVX2 two-row nibble-LUT verifier kernel.
     *
     * One eight-column segment is carried through the complete increasing-K
     * loop. Independent row chains provide two-way ILP while each decoded
     * weight vector is shared across both rows. Low and high nibbles are folded
     * into those row chains so the saturation-safe emulation fits the sixteen-
     * register AVX2 ABI without hot-loop spills.
     */
    inline void gemm_2row_native_chunk_avx2(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        float *C_row0,
        float *C_row1,
        int chunk,
        int kb_start,
        int kb_end,
        const __m256i decode_lut,
        bool accumulate)
    {
        for (int segment = 0; segment < 8; ++segment)
        {
            __m256 fp0 = accumulate
                ? _mm256_loadu_ps(C_row0 + segment * 8)
                : _mm256_setzero_ps();
            __m256 fp1 = accumulate
                ? _mm256_loadu_ps(C_row1 + segment * 8)
                : _mm256_setzero_ps();
            accumulateTwoRowsNibbleSegmentAVX2(
                packed,
                A_q8_row0,
                A_q8_row1,
                chunk,
                kb_start,
                kb_end,
                decode_lut,
                segment,
                fp0,
                fp1);
            _mm256_storeu_ps(C_row0 + segment * 8, fp0);
            _mm256_storeu_ps(C_row1 + segment * 8, fp1);
        }
    }

    /**
     * @brief AVX2 2-row INT8 pre-decoded GEMM microkernel for one 64-col chunk.
     */
    /** @brief Dispatch a two-row non-nibble AVX2 verifier chunk. */
    template<bool SumPairs=false>
    inline void gemm_2row_non_nibble_chunk_avx2(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *A_q8_row0,
        const Q8_1Block *A_q8_row1,
        float *C_row0,
        float *C_row1,
        int chunk,
        int kb_start,
        int kb_end,
        bool accumulate)
    {
        if (packed.encoding == CPUNativeQuantisedEncoding::Q5NativeSingleScale)
            return q5::dispatch<8, SumPairs>(packed, std::array{A_q8_row0, A_q8_row1}, std::array{C_row0, C_row1},
                chunk, kb_start, kb_end, accumulate);
        if (packed.usesCompactMultiScale())
        {
            multi_scale::dispatch<8>(packed, std::array{A_q8_row0, A_q8_row1},
                                     std::array{C_row0, C_row1}, chunk, kb_start, kb_end, accumulate);
            return;
        }
        if (packed.usesQ6KNativeDualScale())
        {
            gemmQ6KNativeTwoRowsAVX2Chunk(
                packed,
                A_q8_row0,
                A_q8_row1,
                C_row0,
                C_row1,
                chunk,
                kb_start,
                kb_end,
                accumulate);
            return;
        }
        if (!packed.usesSignedInt8SingleScale())
            throw std::invalid_argument(
                "AVX2 non-nibble two-row kernel received an unsupported packed encoding");
        if (packed.codebook_id == native_quantised_formats::Q8_0.codebook_id)
            return projectSignedQ8AVX2<2, SumPairs>(packed, std::array{A_q8_row0, A_q8_row1},
                std::array{C_row0, C_row1}, chunk, kb_start, kb_end, accumulate);
        for (int segment = 0; segment < 8; ++segment)
        {
            const int z = segment / 2;
            const int byte_offset = (segment % 2) * 32;
            __m256 fp0 = accumulate
                ? _mm256_loadu_ps(C_row0 + segment * 8)
                : _mm256_setzero_ps();
            __m256 fp1 = accumulate
                ? _mm256_loadu_ps(C_row1 + segment * 8)
                : _mm256_setzero_ps();
            for (int kb = kb_start; kb < kb_end; ++kb)
            {
                const Q8_1Block &a0 = A_q8_row0[kb];
                const Q8_1Block &a1 = A_q8_row1[kb];
                __m256i dot0_even = _mm256_setzero_si256();
                __m256i dot0_odd = _mm256_setzero_si256();
                __m256i dot1_even = _mm256_setzero_si256();
                __m256i dot1_odd = _mm256_setzero_si256();
#if defined(__clang__)
#pragma clang loop unroll(disable)
#elif defined(__GNUC__)
#pragma GCC unroll 1
#endif
                for (int group = 0; group < 8; group += 2)
                {
                    const __m256i weight_even = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group, z) + byte_offset));
                    const __m256i weight_odd = _mm256_load_si256(
                        reinterpret_cast<const __m256i *>(
                            packed.interleavedB(
                                chunk, kb, group + 1, z) + byte_offset));
                    dot0_even = dotSignedQ8WeightsAVX2(dot0_even, weight_even, a0, group * 4);
                    dot1_even = dotSignedQ8WeightsAVX2(dot1_even, weight_even, a1, group * 4);
                    dot0_odd = dotSignedQ8WeightsAVX2(dot0_odd, weight_odd, a0, (group + 1) * 4);
                    dot1_odd = dotSignedQ8WeightsAVX2(dot1_odd, weight_odd, a1, (group + 1) * 4);
                }

                const __m256i corrected0 = _mm256_sub_epi32(
                    _mm256_add_epi32(dot0_even, dot0_odd), q8WeightBiasAVX2(a0));
                const __m256i corrected1 = _mm256_sub_epi32(
                    _mm256_add_epi32(dot1_even, dot1_odd), q8WeightBiasAVX2(a1));
                const __m256 weight_scale = packed.loadScales256(packed.chunkScales(chunk, kb) + segment * 8);
                const float a0_scale = nativeQuantisedFP16ScaleToFP32(a0.d);
                const float a1_scale = nativeQuantisedFP16ScaleToFP32(a1.d);
                fp0 = accumulateSingleScaleBlockAVX2(
                    fp0, corrected0, weight_scale, a0_scale,
                    packed.numerical_policy);
                fp1 = accumulateSingleScaleBlockAVX2(
                    fp1, corrected1, weight_scale, a1_scale,
                    packed.numerical_policy);
            }
            if constexpr(SumPairs) _mm256_storeu_ps(C_row0 + segment * 8,_mm256_add_ps(fp0,fp1));
            else { _mm256_storeu_ps(C_row0 + segment * 8, fp0);
            _mm256_storeu_ps(C_row1 + segment * 8, fp1); }
        }
    }

    /**
     * @brief Fold one 64-column AVX2 K partition in device order.
     *
     * The destination is initialized by partition zero and subsequently read,
     * added, and written exactly once for every increasing partition. This is
     * the AVX2 expression of the same placement-invariant tree used by AVX-512,
     * CUDA, and ROCm.
     *
     * @param destination Running 64-column reduction.
     * @param partial Newly completed 64-column partition.
     * @param first_partition Whether @p partial initializes the reduction.
     */
    inline void foldOrderedAVX2Partition(
        float *destination,
        const float *partial,
        bool first_partition)
    {
        for (int segment = 0; segment < 8; ++segment)
        {
            const __m256 value = _mm256_load_ps(partial + segment * 8);
            _mm256_storeu_ps(
                destination + segment * 8,
                first_partition
                    ? value
                    : addOrderedAVX2(
                          _mm256_loadu_ps(destination + segment * 8),
                          value));
        }
    }

    /**
     * @brief Execute one AVX2 row with the exact device K-partition tree.
     *
     * The encoding's established AVX2 microkernel produces one partition from
     * positive zero. Immediate task-local folding avoids the complete partial
     * plane and applies equally to nibble, signed Q8, and native Q6 weights.
     *
     * @param packed Prepared matrix carrying its typed physical encoding.
     * @param row Q8_1 activation row.
     * @param output FP32 destination row.
     * @param chunk Sixty-four-column output chunk.
     * @param K_blocks Number of source Q8_1 blocks.
     * @param k_tiles Ordered K-partition count.
     * @param k_blocks_per_tile Consecutive blocks in one partition.
     * @param decode_lut Broadcast decoder for nibble formats.
     */
    inline void gemvAVX2OrderedPartitions(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *row,
        float *output,
        int chunk,
        int K_blocks,
        int k_tiles,
        int k_blocks_per_tile,
        const __m256i decode_lut)
    {
        if (MoEProjectionNumericalContract::
                oneNativeQuantisedBlockPerOrderedPartition(
                    K_blocks *
                    MoEProjectionNumericalContract::
                        native_quantised_values_per_block))
        {
            if (packed.usesNibbleLUT())
            {
                gemv_avx2_chunk_native(
                    packed,
                    row,
                    output,
                    chunk,
                    0,
                    K_blocks,
                    decode_lut,
                    /*accumulate=*/false);
            }
            else
            {
                gemv_avx2_chunk_non_nibble(
                    packed,
                    row,
                    output,
                    chunk,
                    0,
                    K_blocks,
                    /*accumulate=*/false);
            }
            return;
        }

        if (packed.usesNibbleLUT())
        {
            for (int segment = 0; segment < 8; ++segment)
            {
                __m256 reduced = _mm256_setzero_ps();
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
                {
                    const int kb_start = k_tile * k_blocks_per_tile;
                    const int kb_end = std::min(
                        kb_start + k_blocks_per_tile, K_blocks);
                    const __m256 partial = accumulateNibbleSegmentAVX2(
                        packed,
                        row,
                        chunk,
                        kb_start,
                        kb_end,
                        decode_lut,
                        segment,
                        _mm256_setzero_ps());
                    reduced = k_tile == 0
                        ? partial
                        : addOrderedAVX2(reduced, partial);
                }
                _mm256_storeu_ps(output + segment * 8, reduced);
            }
            return;
        }

        alignas(32) float partial[64];
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int kb_start = k_tile * k_blocks_per_tile;
            const int kb_end = std::min(
                kb_start + k_blocks_per_tile, K_blocks);
            gemv_avx2_chunk_non_nibble(
                packed,
                row,
                partial,
                chunk,
                kb_start,
                kb_end,
                /*accumulate=*/false);
            foldOrderedAVX2Partition(
                output, partial, k_tile == 0);
        }
    }

    /**
     * @brief Execute two AVX2 rows with shared decode and exact K folding.
     *
     * Each physical encoding keeps its canonical two-row microkernel. Only one
     * 512-byte task-local partition is materialized before an immediate ordered
     * fold, retaining shared weight decode without a layer-sized scratch plane.
     *
     * @param packed Prepared matrix carrying its typed physical encoding.
     * @param row0 First Q8_1 activation row.
     * @param row1 Second Q8_1 activation row.
     * @param output0 First FP32 destination row.
     * @param output1 Second FP32 destination row.
     * @param chunk Sixty-four-column output chunk.
     * @param K_blocks Number of source Q8_1 blocks.
     * @param k_tiles Ordered K-partition count.
     * @param k_blocks_per_tile Consecutive blocks in one partition.
     * @param decode_lut Broadcast decoder for nibble formats.
     */
    inline void gemmTwoRowsAVX2OrderedPartitions(
        const CPUNativeQuantisedPackedWeights &packed,
        const Q8_1Block *row0,
        const Q8_1Block *row1,
        float *output0,
        float *output1,
        int chunk,
        int K_blocks,
        int k_tiles,
        int k_blocks_per_tile,
        const __m256i decode_lut)
    {
        if (MoEProjectionNumericalContract::
                oneNativeQuantisedBlockPerOrderedPartition(
                    K_blocks *
                    MoEProjectionNumericalContract::
                        native_quantised_values_per_block))
        {
            if (packed.usesNibbleLUT())
            {
                gemm_2row_native_chunk_avx2(
                    packed,
                    row0,
                    row1,
                    output0,
                    output1,
                    chunk,
                    0,
                    K_blocks,
                    decode_lut,
                    /*accumulate=*/false);
            }
            else
            {
                gemm_2row_non_nibble_chunk_avx2(
                    packed,
                    row0,
                    row1,
                    output0,
                    output1,
                    chunk,
                    0,
                    K_blocks,
                    /*accumulate=*/false);
            }
            return;
        }

        if (packed.usesNibbleLUT())
        {
            for (int segment = 0; segment < 8; ++segment)
            {
                __m256 reduced0 = _mm256_setzero_ps();
                __m256 reduced1 = _mm256_setzero_ps();
                for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
                {
                    const int kb_start = k_tile * k_blocks_per_tile;
                    const int kb_end = std::min(
                        kb_start + k_blocks_per_tile, K_blocks);
                    __m256 partial0 = _mm256_setzero_ps();
                    __m256 partial1 = _mm256_setzero_ps();
                    accumulateTwoRowsNibbleSegmentAVX2(
                        packed,
                        row0,
                        row1,
                        chunk,
                        kb_start,
                        kb_end,
                        decode_lut,
                        segment,
                        partial0,
                        partial1);
                    if (k_tile == 0)
                    {
                        reduced0 = partial0;
                        reduced1 = partial1;
                    }
                    else
                    {
                        reduced0 = addOrderedAVX2(reduced0, partial0);
                        reduced1 = addOrderedAVX2(reduced1, partial1);
                    }
                }
                _mm256_storeu_ps(output0 + segment * 8, reduced0);
                _mm256_storeu_ps(output1 + segment * 8, reduced1);
            }
            return;
        }

        alignas(32) float partial0[64];
        alignas(32) float partial1[64];
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile)
        {
            const int kb_start = k_tile * k_blocks_per_tile;
            const int kb_end = std::min(
                kb_start + k_blocks_per_tile, K_blocks);
            gemm_2row_non_nibble_chunk_avx2(
                packed,
                row0,
                row1,
                partial0,
                partial1,
                chunk,
                kb_start,
                kb_end,
                /*accumulate=*/false);
            foldOrderedAVX2Partition(
                output0, partial0, k_tile == 0);
            foldOrderedAVX2Partition(
                output1, partial1, k_tile == 0);
        }
    }

} // namespace llaminar2::cpu::native_quantised
