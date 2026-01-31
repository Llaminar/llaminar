#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <iostream>

#include "v2/kernels/cpu/primitives/SoftmaxPrimitives_New.h"

using namespace llaminar2::primitives;

namespace
{
    // Helper to check if AVX512 is available at runtime
    bool has_avx512()
    {
#if defined(__AVX512F__)
        return true;
#else
        return false;
#endif
    }

    TEST(Test__SoftmaxInf, PaddingRow_AllInf)
    {
        if (!has_avx512())
        {
            GTEST_SKIP() << "AVX512 not available";
        }

        const int cols = 128;
        std::vector<float> row(cols, -std::numeric_limits<float>::infinity());

        // Call AVX512 implementation directly
        softmax_row_fp32_avx512(row.data(), cols, false, 1.0f, 0);

        // Expect all zeros (or soft zeros)
        // If sum is 0, implementation sets sum=1.0, inv=1.0.
        // exp(-inf - 0) * 1.0 = 0.
        for (int i = 0; i < cols; ++i)
        {
            EXPECT_EQ(row[i], 0.0f) << "Index " << i;
        }
    }

    TEST(Test__SoftmaxInf, MixedValues_WithInf)
    {
        if (!has_avx512())
        {
            GTEST_SKIP() << "AVX512 not available";
        }

        const int cols = 16;
        std::vector<float> row(cols);
        // First half 1.0, second half -inf
        for (int i = 0; i < 8; ++i)
            row[i] = 1.0f;
        for (int i = 8; i < 16; ++i)
            row[i] = -std::numeric_limits<float>::infinity();

        // Call AVX512 implementation directly
        softmax_row_fp32_avx512(row.data(), cols, false, 1.0f, 0);

        // Expected: softmax of [1, 1, ..., 1] (8 times) is 1/8 = 0.125
        // -inf should be 0.
        for (int i = 0; i < 8; ++i)
        {
            EXPECT_NEAR(row[i], 0.125f, 1e-6f) << "Index " << i;
        }
        for (int i = 8; i < 16; ++i)
        {
            EXPECT_EQ(row[i], 0.0f) << "Index " << i;
        }
    }

    TEST(Test__SoftmaxInf, CausalMasking_WithInfInput)
    {
        if (!has_avx512())
        {
            GTEST_SKIP() << "AVX512 not available";
        }

        const int cols = 16;
        std::vector<float> row(cols, 1.0f);

        // Apply causal mask for row_idx = 7 (first 8 elements valid)
        // But input also has -inf at index 0 (just to test mixing)
        row[0] = -std::numeric_limits<float>::infinity();

        // Call AVX512 implementation directly
        softmax_row_fp32_avx512(row.data(), cols, true, 1.0f, 7);

        // Valid indices: 0..7.
        // Index 0 is -inf -> exp(-inf) = 0.
        // Indices 1..7 are 1.0 -> exp(1.0).
        // Indices 8..15 are masked -> 0.

        // Max should be 1.0.
        // Sum = exp(-inf - 1) + 7 * exp(1 - 1) = 0 + 7 * 1 = 7.
        // Prob(0) = 0.
        // Prob(1..7) = 1/7 = 0.142857.
        // Prob(8..15) = 0.

        EXPECT_EQ(row[0], 0.0f);
        for (int i = 1; i <= 7; ++i)
        {
            EXPECT_NEAR(row[i], 1.0f / 7.0f, 1e-6f) << "Index " << i;
        }
        for (int i = 8; i < 16; ++i)
        {
            EXPECT_EQ(row[i], 0.0f) << "Index " << i;
        }
    }

    TEST(Test__SoftmaxInf, NaN_Propagation)
    {
        if (!has_avx512())
        {
            GTEST_SKIP() << "AVX512 not available";
        }

        const int cols = 16;
        std::vector<float> row(cols, 1.0f);
        row[0] = std::numeric_limits<float>::quiet_NaN();

        // Call AVX512 implementation directly
        softmax_row_fp32_avx512(row.data(), cols, false, 1.0f, 0);

        // If input has NaN, output should be NaN (or at least defined behavior)
        // We just want to ensure it doesn't crash or hang.
        // And ideally it propagates NaN so we can detect it.

        bool has_nan = false;
        for (int i = 0; i < cols; ++i)
        {
            if (std::isnan(row[i]))
                has_nan = true;
        }
        EXPECT_TRUE(has_nan) << "NaN should propagate";
    }
}
