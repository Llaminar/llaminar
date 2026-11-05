/**
 * @file CPUNativeNEONResidualCorrection.h
 * @brief Native ARM paired-Q8 residual publication and shared-weight term folding.
 *
 * Two real rows become four independent primary/residual dot chains in one
 * native tile. A bounded four-by-64 private fragment, not a matrix mirror,
 * retains the separately completed terms before the single canonical pair add.
 * All weight decoding and source arithmetic are the ordinary prepared tile's
 * program. No serial-row inference replay or secondary weight pack is used.
 */
#pragma once
#include "CPUNativeNEONPreparedProjection.h"
#include "CPUProjectionQ8Publication.h"
#include <array>
#include <span>
#include <stdexcept>

namespace llaminar2::cpu::native_vnni
{
    /** @brief Quantize the original FP32 residual against the stored primary half scale.
     * @param input Thirty-two original finite FP32 values.
     * @param primary Complete published original primary Q8 block.
     * @param residual Exclusive secondary Q8 destination.
     * @param policy Existing prepared source activation/arithmetic policy.
     */
    inline void quantizeResidualBlockNEON(const float *input, const Q8_1Block &primary,
        Q8_1Block &residual, CPUProjectionNumericalPolicy policy)
    {
        alignas(64) float values[32];
        const auto scale = nativeVNNIFP16ScaleToFP32(primary.d);
        for (int i = 0; i < 32; i += 4)
        {
            const auto decoded = persistRoundedNEON(vmulq_n_f32(simd::neon_typed::load4(primary.qs + i), scale));
            vst1q_f32(values + i, vsubq_f32(vld1q_f32(input + i), decoded));
        }
        quantizeActivationBlock(values, residual, 32, policy);
    }

    /** @brief Compute two real rows' four source-correct dot chains and fold each pair once.
     * @tparam Instruction Authenticated native signed-byte instruction family.
     * @param weights Retained canonical prepared publication.
     * @param primary0 First real row's original complete primary bank.
     * @param residual0 First real row's separately quantized residual bank.
     * @param primary1 Second real row's primary bank, or null for an odd tail.
     * @param residual1 Second real row's paired residual bank, or null.
     * @param destination0 Exclusive live first-row output interval.
     * @param destination1 Exclusive live second-row interval, or null.
     * @param chunk Original 64-column source chunk.
     * @param valid Exact logical live column count in that source chunk.
     * @throws std::invalid_argument For an incomplete real-row pair or bad live geometry.
     * @pre The enclosing invocation validates final destinations against every
     * retained input/source/publication bank; they remain exclusive through completion.
     */
    template <multi_scale::NEONByteDot Instruction>
    inline void nativeNEONResidualChunk(CPUNativeVNNIWeightView weights,
        const Q8_1Block *primary0, const Q8_1Block *residual0,
        const Q8_1Block *primary1, const Q8_1Block *residual1,
        float *destination0, float *destination1, int chunk, int valid)
    {
        if (!primary0 || !residual0 || !destination0 || chunk < 0 || chunk >= weights.N_padded / 64 ||
            valid != std::min(64, weights.N - chunk * 64) ||
            (bool(primary1) != bool(residual1)) || (bool(primary1) != bool(destination1)))
            throw std::invalid_argument("Native ARM residual projection requires complete original rows and exact live columns");
        alignas(64) std::array<std::array<float, 64>, 4> terms;
        const auto block_count = std::size_t(weights.blocks_per_row);
        const auto live = std::size_t(valid);
        if (primary1)
            CPUNativeNEONProjectionTile<4>::bind(weights,
                {std::span(primary0, block_count), std::span(residual0, block_count),
                 std::span(primary1, block_count), std::span(residual1, block_count)},
                {std::span(terms[0]).first(live), std::span(terms[1]).first(live),
                 std::span(terms[2]).first(live), std::span(terms[3]).first(live)}, chunk).template execute<Instruction>();
        else
            CPUNativeNEONProjectionTile<2>::bind(weights,
                {std::span(primary0, block_count), std::span(residual0, block_count)},
                {std::span(terms[0]).first(live), std::span(terms[1]).first(live)}, chunk).template execute<Instruction>();
        int column = 0;
        for (; column + 4 <= valid; column += 4)
        {
            vst1q_f32(destination0 + column, vaddq_f32(vld1q_f32(terms[0].data() + column), vld1q_f32(terms[1].data() + column)));
            if (primary1)
                vst1q_f32(destination1 + column, vaddq_f32(vld1q_f32(terms[2].data() + column), vld1q_f32(terms[3].data() + column)));
        }
        for (; column < valid; ++column)
        {
            destination0[column] = terms[0][column] + terms[1][column];
            if (primary1) destination1[column] = terms[2][column] + terms[3][column];
        }
    }
} // namespace llaminar2::cpu::native_vnni
