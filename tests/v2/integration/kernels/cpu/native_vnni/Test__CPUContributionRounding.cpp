/**
 * @file Test__CPUContributionRounding.cpp
 * @brief Independent byte oracle for inlined CPU quantized-block arithmetic.
 *
 * Grouped-versus-serial tests can miss a defect shared by both paths. These
 * tests compare both numerical policies with explicitly rounded scalar steps,
 * including asymmetric correction and accumulated K blocks. They exercise the
 * common contribution primitive used by every single-scale source codebook;
 * the existing all-format projection gate covers decoding and dispatch.
 */

#include <gtest/gtest.h>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <random>

#include "kernels/cpu/gemm/CPUNativeVNNIContributionContract.h"
#include "utils/CPUFeatures.h"

namespace
{
    using llaminar2::CPUProjectionNumericalPolicy;
    using namespace llaminar2::cpu::native_vnni;

    /** Materialize an independent binary32 rounding boundary for the oracle. */
    float rounded(float value)
    {
        volatile float result = value;
        return result;
    }

    /** Evaluate one block without reusing any production SIMD arithmetic. */
    float reference(float accumulator, std::int32_t dot, float scale,
                    std::int32_t sum, float minimum, float activation,
                    CPUProjectionNumericalPolicy policy, bool asymmetric)
    {
        const float scaled_weight = rounded(activation * scale);
        if (policy == CPUProjectionNumericalPolicy::BackendNative)
        {
            const float result = std::fma(static_cast<float>(dot), scaled_weight, accumulator);
            return asymmetric
                ? std::fma(rounded(static_cast<float>(sum) * activation), minimum, result)
                : result;
        }
        const float contribution = rounded(scaled_weight * static_cast<float>(dot));
        const float correction = asymmetric
            ? rounded(rounded(activation * minimum) * static_cast<float>(sum))
            : 0.0f;
        const float block = asymmetric ? rounded(contribution + correction) : contribution;
        return rounded(accumulator + block);
    }

    /** Sweep nontrivial rounded contributions through one explicit SIMD ISA. */
    template <bool AVX512>
    void verifyContributionRounding()
    {
        constexpr int lanes = AVX512 ? 16 : 8;
        std::mt19937 random(0xD15F00Du);
        std::uniform_int_distribution<int> integer(-400000, 400000);
        const auto half_scale = [&]()
        {
            // Binary16-representable scales, including both signs. Randomize
            // exponents to expose rounding at every accumulation boundary.
            const float significand = static_cast<float>(1024 + random() % 1024) / 1024.0f;
            return std::ldexp((random() & 1) ? significand : -significand,
                              -static_cast<int>(random() % 12));
        };
        for (auto policy : {CPUProjectionNumericalPolicy::BackendNative,
                            CPUProjectionNumericalPolicy::GPUAlignedExpert})
        {
            for (bool asymmetric : {false, true})
            {
                SCOPED_TRACE(static_cast<int>(policy));
                SCOPED_TRACE(asymmetric);
                std::array<float, lanes> actual{}, expected{}, scales{}, minima{};
                std::array<std::int32_t, lanes> dots{};
                for (int block = 0; block < 1024; ++block)
                {
                    const float activation = half_scale();
                    const int sum = integer(random) % 4065;
                    for (int lane = 0; lane < lanes; ++lane)
                    {
                        dots[lane] = integer(random);
                        scales[lane] = half_scale();
                        minima[lane] = half_scale();
                        expected[lane] = reference(expected[lane], dots[lane], scales[lane],
                                                   sum, minima[lane], activation, policy, asymmetric);
                    }
#if defined(__AVX512F__)
                    if constexpr (AVX512)
                    {
                        const auto acc = _mm512_loadu_ps(actual.data());
                        const auto dot = _mm512_loadu_si512(dots.data());
                        const auto scale = _mm512_loadu_ps(scales.data());
                        const auto result = asymmetric
                            ? accumulateCorrectedSingleScaleBlockAVX512(acc, dot, scale, sum,
                                  _mm512_loadu_ps(minima.data()), activation, policy)
                            : accumulateSingleScaleBlockAVX512(acc, dot, scale, activation, policy);
                        _mm512_storeu_ps(actual.data(), result);
                    }
                    else
#endif
                    {
                        const auto acc = _mm256_loadu_ps(actual.data());
                        const auto dot = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(dots.data()));
                        const auto scale = _mm256_loadu_ps(scales.data());
                        const auto result = asymmetric
                            ? accumulateCorrectedSingleScaleBlockAVX2(acc, dot, scale, sum,
                                  _mm256_loadu_ps(minima.data()), activation, policy)
                            : accumulateSingleScaleBlockAVX2(acc, dot, scale, activation, policy);
                        _mm256_storeu_ps(actual.data(), result);
                    }
                    for (int lane = 0; lane < lanes; ++lane)
                        ASSERT_EQ(std::bit_cast<std::uint32_t>(actual[lane]),
                                  std::bit_cast<std::uint32_t>(expected[lane]))
                            << "block=" << block << " lane=" << lane;
                }
            }
        }
    }

    /** The AVX2 image and AVX2 runtime must retain identical rounding edges. */
    TEST(CPUContributionRounding, AVX2)
    {
        ASSERT_GE(llaminar2::activeISALevel(), llaminar2::ISALevel::AVX2);
        verifyContributionRounding<false>();
    }

#if defined(__AVX512F__)
    /** Inlining wide leaves must preserve both native FMA and expert semantics. */
    TEST(CPUContributionRounding, AVX512)
    {
        ASSERT_GE(llaminar2::activeISALevel(), llaminar2::ISALevel::AVX512);
        verifyContributionRounding<true>();
    }
#endif
}
