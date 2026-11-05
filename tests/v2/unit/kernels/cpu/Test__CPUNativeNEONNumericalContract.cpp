/**
 * @file Test__CPUNativeNEONNumericalContract.cpp
 * @brief Independent signed-dot and rounded contribution proof for native ARM leaves.
 *
 * The tests enter the production contribution header and compare against exact
 * scalar integer products or the existing memory-rounded numerical oracle.
 * They cover byte extrema, column/K lane ordering, source-scale precision,
 * both CPU projection policies and invalid policy rejection. This is a block
 * arithmetic proof, not a certificate for the complete NativeVNNI launcher.
 */
#include "kernels/cpu/gemm/CPUNativeVNNIContributionContract.h"
#include "kernels/cpu/gemm/CPUNativeVNNIFP16.h"
#include "utils/CPUFeatures.h"
#include "utils/CPUExplicitRoundingOracle.h"

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace llaminar2::test
{
namespace
{
using namespace cpu::native_vnni;

/** @brief Compare complete FP32 words, including signed zero. */
void expectWord(float actual, float expected, int lane)
{
    EXPECT_EQ(std::bit_cast<std::uint32_t>(actual), std::bit_cast<std::uint32_t>(expected))
        << "lane=" << lane;
}

/** @brief Return deterministic local fixture bits without mutable process state. */
std::uint32_t nextWord(std::uint32_t &state)
{
    state = state * 1664525u + 1013904223u;
    return state;
}

/** @brief Observe four independent column dots from the compiled production leaf. */
int32x4_t dotColumns(int32x4_t sums, int8x16_t activation, int8x16_t weights)
{
#if defined(__ARM_FEATURE_DOTPROD)
    static const bool admitted = []
    {
        cpu::ARMCPUCapabilities::native().require(cpu::ARMCPUFeature::DotProd);
        return true;
    }();
    (void)admitted;
    const int32x4_t actual = dotFourColumnsDotProd(sums, activation, weights);
    std::array<std::int32_t, 4> widening{}, dot{};
    vst1q_s32(widening.data(), dotFourColumnsASIMD(sums, activation, weights));
    vst1q_s32(dot.data(), actual);
    EXPECT_EQ(dot, widening);
    return actual;
#else
    return dotFourColumnsASIMD(sums, activation, weights);
#endif
}
} // namespace

/** @test Every signed-byte product avoids a saturating INT16 pair sum. */
TEST(CPUNativeNEONNumericalContract, AllSignedBytePairsAccumulateWithoutSaturation)
{
    const std::array<std::int32_t, 4> initial{7, -17, 911, -43};
    for (int activation = -128; activation <= 127; ++activation)
        for (int weight = -128; weight <= 127; ++weight)
        {
            std::array<std::int32_t, 4> actual;
            vst1q_s32(actual.data(), dotColumns(vld1q_s32(initial.data()),
                vdupq_n_s8(static_cast<std::int8_t>(activation)),
                vdupq_n_s8(static_cast<std::int8_t>(weight))));
            for (int column = 0; column < 4; ++column)
                ASSERT_EQ(actual[column], initial[column] + 4 * activation * weight)
                    << "a=" << activation << " w=" << weight << " column=" << column;
        }
}

/** @test Eight K-word groups retain every original activation/column coordinate. */
TEST(CPUNativeNEONNumericalContract, FullBlockDotsRetainFourIndependentColumns)
{
    std::uint32_t state = 42;
    for (int fixture = 0; fixture < 1024; ++fixture)
    {
        std::array<std::int8_t, 32> activations;
        std::array<std::int8_t, 4 * 32> weights;
        for (auto &value : activations)
            value = std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(nextWord(state) >> 24));
        for (auto &value : weights)
            value = std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(nextWord(state) >> 24));
        std::array<std::int32_t, 4> expected{};
        int32x4_t sums = vdupq_n_s32(0);
        for (int group = 0; group < 8; ++group)
        {
            std::array<std::int8_t, 16> activation_word{}, weight_word{};
            for (int column = 0; column < 4; ++column)
                for (int element = 0; element < 4; ++element)
                {
                    const int k = group * 4 + element;
                    activation_word[column * 4 + element] = activations[k];
                    weight_word[column * 4 + element] = weights[column * 32 + k];
                    expected[column] += int(activations[k]) * int(weights[column * 32 + k]);
                }
            sums = dotColumns(sums, vld1q_s8(activation_word.data()), vld1q_s8(weight_word.data()));
        }
        std::array<std::int32_t, 4> actual;
        vst1q_s32(actual.data(), sums);
        ASSERT_EQ(actual, expected) << "fixture=" << fixture;
    }
}

/** @test Both numerical programs match an independent memory-rounded block oracle. */
TEST(CPUNativeNEONNumericalContract, SymmetricAndAsymmetricContributionsKeepEveryRoundingEdge)
{
    std::uint32_t state = 917;
    for (int fixture = 0; fixture < 4096; ++fixture)
    {
        std::array<float, 4> accumulator, scales, minima;
        std::array<std::int32_t, 4> dots;
        const auto half = static_cast<std::uint16_t>(1 + nextWord(state) % 0x7bffu);
        const float activation = nativeVNNIFP16ScaleToFP32(half);
        const auto sum = static_cast<std::int32_t>(nextWord(state) % 8129u) - 4064;
        for (int lane = 0; lane < 4; ++lane)
        {
            accumulator[lane] = float(static_cast<int>(nextWord(state) % 20001u) - 10000) * 0.03125f;
            const auto scale_half = static_cast<std::uint16_t>(nextWord(state) % 0x7c00u);
            const auto min_half = static_cast<std::uint16_t>(nextWord(state) % 0x7c00u);
            // Effective native source scales may be a half base multiplied by
            // an integer coefficient. Never narrow that FP32 product back to half.
            scales[lane] = memoryRoundedMultiply(nativeVNNIFP16ScaleToFP32(scale_half),
                float(static_cast<int>(nextWord(state) % 255u) - 127));
            minima[lane] = nativeVNNIFP16ScaleToFP32(min_half);
            if (nextWord(state) & 1u) minima[lane] = -minima[lane];
            dots[lane] = static_cast<std::int32_t>(nextWord(state) % 1040385u) - 520192;
        }
        for (const auto policy : {CPUProjectionNumericalPolicy::BackendNative,
                                  CPUProjectionNumericalPolicy::GPUAlignedExpert})
        {
            std::array<float, 4> symmetric, asymmetric;
            vst1q_f32(symmetric.data(), accumulateSingleScaleBlockNEON(
                vld1q_f32(accumulator.data()), vld1q_s32(dots.data()),
                vld1q_f32(scales.data()), activation, policy));
            vst1q_f32(asymmetric.data(), accumulateCorrectedSingleScaleBlockNEON(
                vld1q_f32(accumulator.data()), vld1q_s32(dots.data()),
                vld1q_f32(scales.data()), sum, vld1q_f32(minima.data()), activation, policy));
            for (int lane = 0; lane < 4; ++lane)
            {
                const float scale = memoryRoundedMultiply(activation, scales[lane]);
                const float dot = static_cast<float>(dots[lane]);
                if (policy == CPUProjectionNumericalPolicy::GPUAlignedExpert)
                {
                    const float product = memoryRoundedMultiply(scale, dot);
                    const float correction = memoryRoundedMultiply(
                        memoryRoundedMultiply(activation, minima[lane]), static_cast<float>(sum));
                    expectWord(symmetric[lane], memoryRoundedAdd(accumulator[lane], product), lane);
                    expectWord(asymmetric[lane], memoryRoundedAdd(accumulator[lane],
                        memoryRoundedAdd(product, correction)), lane);
                }
                else
                {
                    const float product = std::fma(dot, scale, accumulator[lane]);
                    expectWord(symmetric[lane], product, lane);
                    expectWord(asymmetric[lane], std::fma(
                        memoryRoundedMultiply(static_cast<float>(sum), activation), minima[lane], product), lane);
                }
            }
        }
    }
}

/** @test A cancellation fixture distinguishes expert rounding from native FMA. */
TEST(CPUNativeNEONNumericalContract, ExpertRoundingCannotCollapseIntoNativeFMA)
{
    const float scale = std::bit_cast<float>(0x3f800001u);
    const float accumulator = -memoryRoundedMultiply(3.0f, scale);
    std::array<float, 4> native, expert;
    vst1q_f32(native.data(), accumulateSingleScaleBlockNEON(vdupq_n_f32(accumulator),
        vdupq_n_s32(3), vdupq_n_f32(scale), 1.0f, CPUProjectionNumericalPolicy::BackendNative));
    vst1q_f32(expert.data(), accumulateSingleScaleBlockNEON(vdupq_n_f32(accumulator),
        vdupq_n_s32(3), vdupq_n_f32(scale), 1.0f, CPUProjectionNumericalPolicy::GPUAlignedExpert));
    for (int lane = 0; lane < 4; ++lane)
    {
        expectWord(native[lane], std::fma(3.0f, scale, accumulator), lane);
        expectWord(expert[lane], 0.0f, lane);
        EXPECT_NE(std::bit_cast<std::uint32_t>(native[lane]), std::bit_cast<std::uint32_t>(expert[lane]));
    }
}

/** @test Unknown policy values fail before any contribution can be published. */
TEST(CPUNativeNEONNumericalContract, RejectsUnknownNumericalPolicy)
{
    const auto invalid = static_cast<CPUProjectionNumericalPolicy>(255);
    EXPECT_THROW((accumulateSingleScaleBlockNEON(vdupq_n_f32(0), vdupq_n_s32(0),
        vdupq_n_f32(1), 1, invalid)), std::invalid_argument);
    EXPECT_THROW((accumulateCorrectedSingleScaleBlockNEON(vdupq_n_f32(0), vdupq_n_s32(0),
        vdupq_n_f32(1), 0, vdupq_n_f32(0), 1, invalid)), std::invalid_argument);
}
} // namespace llaminar2::test
