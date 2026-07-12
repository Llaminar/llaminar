/**
 * @file Test__NativeVNNITrainerEvidence.cpp
 * @brief CPU-only contract tests for cross-backend trainer evidence math.
 */

#include "utils/NativeVNNITrainerEvidence.h"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace
{
    using llaminar2::test::trainer::compareFP32;
    using llaminar2::test::trainer::nativeByteDigest;
    using llaminar2::test::trainer::nativeByteMismatchCount;
    using llaminar2::test::trainer::rowSoftmaxSymmetricKLDivergence;
    using llaminar2::test::trainer::summarizeSortedTimingSamples;

    TEST(Test__NativeVNNITrainerEvidence, NativeBytesDistinguishSignedZero)
    {
        const std::vector<float> positive{0.0f, 1.0f};
        const std::vector<float> negative{-0.0f, 1.0f};

        EXPECT_NE(nativeByteDigest(positive), nativeByteDigest(negative));
        EXPECT_EQ(nativeByteMismatchCount(positive, negative), 1u);

        const auto evidence = compareFP32(positive, negative, positive.size());
        EXPECT_EQ(evidence.mismatch_count, 1u);
        EXPECT_EQ(evidence.first_mismatch_index, 0u);
        EXPECT_DOUBLE_EQ(evidence.max_abs, 0.0);
        EXPECT_FALSE(evidence.bitwiseEqual());
    }

    TEST(Test__NativeVNNITrainerEvidence, TimingSummaryUsesTrainerRankPolicy)
    {
        const std::vector<double> samples{1.0, 2.0, 3.0, 4.0, 100.0};
        const auto summary = summarizeSortedTimingSamples(samples);

        EXPECT_DOUBLE_EQ(summary.min, 1.0);
        EXPECT_DOUBLE_EQ(summary.median, 3.0);
        EXPECT_DOUBLE_EQ(summary.p95, 100.0);
        EXPECT_DOUBLE_EQ(summary.mad, 1.0);
        EXPECT_GT(summary.cv, 1.0);
        EXPECT_EQ(summary.digest, nativeByteDigest(samples));
    }

    TEST(Test__NativeVNNITrainerEvidence, SymmetricKLIsActuallySymmetric)
    {
        const std::vector<float> lhs{3.0f, 1.0f, -2.0f};
        const std::vector<float> rhs{2.5f, 1.5f, -1.0f};

        const double forward = rowSoftmaxSymmetricKLDivergence(
            lhs.data(), rhs.data(), lhs.size());
        const double reverse = rowSoftmaxSymmetricKLDivergence(
            rhs.data(), lhs.data(), lhs.size());
        EXPECT_GT(forward, 0.0);
        EXPECT_NEAR(forward, reverse, 1.0e-15);
        EXPECT_DOUBLE_EQ(
            rowSoftmaxSymmetricKLDivergence(
                lhs.data(), lhs.data(), lhs.size()),
            0.0);
    }

    TEST(Test__NativeVNNITrainerEvidence, ExactRowsPassEveryEvidenceGate)
    {
        const std::vector<float> values{0.25f, -1.5f, 7.0f, 0.0f};
        const auto evidence = compareFP32(values, values, 2);

        EXPECT_TRUE(evidence.bitwiseEqual());
        EXPECT_EQ(evidence.mismatch_count, 0u);
        EXPECT_DOUBLE_EQ(evidence.cosine, 1.0);
        EXPECT_DOUBLE_EQ(evidence.relative_l2, 0.0);
        EXPECT_DOUBLE_EQ(evidence.max_abs, 0.0);
        EXPECT_DOUBLE_EQ(evidence.symmetric_kld, 0.0);
        EXPECT_EQ(evidence.actual_digest, evidence.expected_digest);
    }
} // namespace
