/**
 * @file Test__ParityTensorFiniteWitness.cpp
 * @brief The actual parity comparator rejects nonfinite operands before histogram indexing.
 *
 * These device-free tests call the canonical raw comparison used by ordinary
 * prefill/decode and production MTP. No fixture instance, SetUp, runner, model,
 * backend, alternate comparator or tolerance policy is introduced.
 */
#include "integration/parity/ParityTestBase.h"
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <limits>

namespace llaminar2::test::parity
{
/** @test NaN and signed infinities in either operand remain explicit failed raw witnesses. */
TEST(ParityTensorFiniteWitness,EitherOperandNaNOrInfinityHasUndefinedEntropyAndFails)
{
    for (const bool corruptReference : {false, true})
    for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity()})
    {
        SCOPED_TRACE(::testing::Message() << "reference=" << corruptReference << " invalid=" << invalid);
        std::array actual{1.0f, 2.0f, 3.0f}, reference{1.0f, 2.0f, 3.0f};
        (corruptReference ? reference : actual)[0] = invalid;
        // Keep a finite nonzero error too: the old max reduction could ignore
        // the NaN while still entering the unsafe histogram loop.
        actual[1] = 4.0f;
        const auto result = compareParityTensorData(
            actual.data(), reference.data(), actual.size(), "MOE_ROUTE_CONTRIBUTIONS", 0.99f);
        EXPECT_FALSE(result.passed);
        EXPECT_TRUE(std::isnan(result.error_entropy));
        const auto &damaged = corruptReference ? result.pytorch_stats : result.llaminar_stats;
        const auto &healthy = corruptReference ? result.llaminar_stats : result.pytorch_stats;
        EXPECT_EQ(damaged.nan_count + damaged.inf_count, 1u);
        EXPECT_EQ(healthy.nan_count + healthy.inf_count, 0u);
        EXPECT_EQ(result.total_elements, actual.size());
        EXPECT_EQ(result.stage_name, "MOE_ROUTE_CONTRIBUTIONS");
    }
}

/** @test Invalid reference data cannot pass merely because the cosine default meets its supplied gate. */
TEST(ParityTensorFiniteWitness,ReferenceNaNIsAnIndependentFailureWitness)
{
    const std::array actual{1.0f, 4.0f};
    const std::array reference{std::numeric_limits<float>::quiet_NaN(), 2.0f};
    const auto result = compareParityTensorData(
        actual.data(), reference.data(), actual.size(), "FFN_RESIDUAL", 0.0f);
    EXPECT_FLOAT_EQ(result.cosine_similarity, 0.0f);
    EXPECT_EQ(result.llaminar_stats.nan_count, 0u);
    EXPECT_EQ(result.pytorch_stats.nan_count, 1u);
    EXPECT_TRUE(std::isnan(result.error_entropy));
    EXPECT_FALSE(result.passed);
}

/** @test Finite operands can overflow the FP32 error; no Inf*0 bin conversion may follow. */
TEST(ParityTensorFiniteWitness,FiniteSubtractionOverflowHasUndefinedEntropyAndFails)
{
    const auto maximum = std::numeric_limits<float>::max();
    const std::array actual{maximum, maximum}, reference{-maximum, maximum};
    const auto result = compareParityTensorData(
        actual.data(), reference.data(), actual.size(), "MOE_EXPERT_OUTPUT", 0.0f);
    EXPECT_EQ(result.llaminar_stats.nan_count + result.llaminar_stats.inf_count, 0u);
    EXPECT_EQ(result.pytorch_stats.nan_count + result.pytorch_stats.inf_count, 0u);
    EXPECT_TRUE(std::isinf(result.max_abs_diff));
    EXPECT_TRUE(std::isnan(result.error_entropy));
    EXPECT_FALSE(result.passed);
}

/** @test A finite ordinary histogram retains the original64-bin arithmetic and numerical verdict. */
TEST(ParityTensorFiniteWitness,FiniteHistogramAndExactZeroErrorAreUnchanged)
{
    const std::array actual{0.0f, 1.0f, 2.0f, 3.0f}, reference{0.0f, 1.0f, 2.0f, 2.0f};
    const auto result = compareParityTensorData(
        actual.data(), reference.data(), actual.size(), "ATTENTION_OUTPUT", 0.9f);
    const float expectedEntropy = static_cast<float>(-0.75 * std::log2(0.75) - 0.25 * std::log2(0.25));
    EXPECT_FLOAT_EQ(result.error_entropy, expectedEntropy);
    EXPECT_FLOAT_EQ(result.max_abs_diff, 1.0f);
    EXPECT_TRUE(result.passed);
    const auto exact = compareParityTensorData(
        reference.data(), reference.data(), reference.size(), "ATTENTION_OUTPUT", 0.99f);
    EXPECT_FLOAT_EQ(exact.error_entropy, 0.0f);
    EXPECT_FLOAT_EQ(exact.max_abs_diff, 0.0f);
    EXPECT_TRUE(exact.passed);
}
} // namespace llaminar2::test::parity
