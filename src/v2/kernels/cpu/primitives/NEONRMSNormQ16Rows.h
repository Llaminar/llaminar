/**
 * @file NEONRMSNormQ16Rows.h
 * @brief Native original-block Q16 RMSNorm tiles with distinct FP32-square arithmetic.
 *
 * Q16 normalization first rounds scale*code and its square to FP32, then widens
 * the square into eight fixed FP64 chains. This is not floating RMSNorm's
 * widen-before-square program. Every M retains the same increasing-K chains and
 * fold;1..4 rows share gamma traversal without replaying serial-row calls.
 * Source blocks/scales remain authoritative, sum_qs is not a squared-summary
 * authority, and no decoded head/row bank, allocation or workspace is created.
 */
#pragma once
#include "CPURMSNormBorrow.h"
#include "NEONRMSNormReduction.h"
#include "../../../tensors/NEONFP32.h"
#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>

namespace llaminar2::primitives::neon_rmsnorm
{
namespace detail
{
/** @brief Decode four original Q16 codes with a retained FP32 product boundary.
 * @param block Original compressed block and original FP32 scale.
 * @param offset Four-aligned element offset0..28 inside that original block.
 * @return Four separately rounded original FP32 values, in native registers only.
 */
[[gnu::always_inline]] inline float32x4_t q16Fragment(const Q16_1Block &block,std::size_t offset)
{
    const auto integer=vmovl_s16(vld1_s16(block.qs+offset));
    return neon_fp32::multiply(vcvtq_f32_s32(integer),vdupq_n_f32(block.d));
}

/** @brief Consume one four-value fragment using FP32 squares before FP64 addition.
 * @param block Original Q16 block; @param offset Original aligned in-block offset.
 * @param low Fixed summary chains for the first two lanes.
 * @param high Fixed summary chains for the last two lanes.
 */
[[gnu::always_inline]] inline void q16SquareFragment(const Q16_1Block &block,std::size_t offset,
                                                    float64x2_t &low,float64x2_t &high)
{
    const auto value=q16Fragment(block,offset);
    const auto square=neon_fp32::multiply(value,value);
    low=vaddq_f64(low,vcvt_f64_f32(vget_low_f32(square)));
    high=vaddq_f64(high,vcvt_f64_f32(vget_high_f32(square)));
}
} // namespace detail

/** @brief Normalize one shared-gamma tile directly from checked original Q16 storage.
 * @tparam Count Physical rows1..4; arithmetic identity never depends on this count.
 * @param bank Complete original-bank/policy borrow, protecting all source/gamma bytes.
 * @param first Original first row; never a worker-local rebased coordinate.
 * @throws std::out_of_range For a tile outside the checked original row bank.
 * @throws std::domain_error For nonfinite summaries or an unrepresentable FP32 inverse.
 * @details All tile summaries/inverses complete before publication. Previously
 * retired tiles are not rolled back after later corrupt input is rejected.
 */
template<std::size_t Count>
inline void normalizeQ16(const rmsnorm_operands::Q16ExpansionBorrow &bank,std::size_t first)
{
    static_assert(Count>=1&&Count<=4);
    if(first>bank.rows()||Count>bank.rows()-first)
        throw std::out_of_range("Native Q16 RMS tile outside original row bank");
    if(bank.empty())return;
    // Complete-bank admission and the checked tile interval prove these
    // contiguous original strides. Do not materialize span/pointer arrays:
    // they force repeated hot stack loads on the baseline compiler.
    const auto *input=bank.sourceRow(first).data();
    auto *output=bank.outputRow(first).data();
    const auto columns=bank.columns();
    const auto blocks=columns/32;
    std::array<std::array<float64x2_t,4>,Count> accumulators{};
    for(std::size_t block=0;block<blocks;++block)
        for(std::size_t offset=0;offset<32;offset+=8)
            detail::visit<Count>([&]<std::size_t Row>{
                auto &a=accumulators[Row];
                detail::q16SquareFragment(input[Row*blocks+block],offset,a[0],a[1]);
                detail::q16SquareFragment(input[Row*blocks+block],offset+4,a[2],a[3]);
            });
    auto denominator=vdupq_n_f32(1.0f);
    detail::visit<Count>([&]<std::size_t Row>{
        const auto &a=accumulators[Row];
        const auto folded=vaddq_f64(vaddq_f64(a[0],a[1]),vaddq_f64(a[2],a[3]));
        const double sum=vgetq_lane_f64(folded,0)+vgetq_lane_f64(folded,1);
        if(!std::isfinite(sum))throw std::domain_error("Native Q16 RMS observed a nonfinite FP32-square summary");
        const float mean=float(sum/double(columns));
        denominator=vsetq_lane_f32(mean+bank.epsilon(),denominator,Row);
    });
    // SIMD across rows preserves each FP32 sqrt/divide program, avoids libc
    // errno branches, and keeps inactive lanes exactly1 rather than0/0.
    const auto inverse=vdivq_f32(vdupq_n_f32(1.0f),vsqrtq_f32(denominator));
    detail::visit<Count>([&]<std::size_t Row>{
        if(!std::isfinite(vgetq_lane_f32(inverse,Row)))
            throw std::domain_error("Native Q16 RMS inverse is not representable");
    });
    const auto gamma=bank.gamma();
    for(std::size_t block=0;block<blocks;++block)
        for(std::size_t offset=0;offset<32;offset+=4)
        {
            const auto column=block*32+offset;
            const auto weights=vld1q_f32(gamma.data()+column);
            detail::visit<Count>([&]<std::size_t Row>{
                const auto value=detail::q16Fragment(input[Row*blocks+block],offset);
                const auto normalized=neon_fp32::multiply(value,vdupq_laneq_f32(inverse,Row));
                vst1q_f32(output+Row*columns+column,neon_fp32::multiply(normalized,weights));
            });
        }
}
} // namespace llaminar2::primitives::neon_rmsnorm
