
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "v2/kernels/cpu/fused/FusedRMSNormQuantize.h"

using namespace llaminar2;

TEST(Test__FusedRMSNormQuantize_Padding, ZeroInput)
{
    const int rows = 2;
    const int cols = 128;
    const float epsilon = 1e-6f;

    std::vector<float> input(rows * cols, 0.0f);
    std::vector<float> weight(cols, 1.0f);
    std::vector<int8_t> output(rows * cols);
    std::vector<float> scales(rows);

    FusedRMSNormQuantize kernel;
    kernel.execute(input.data(), weight.data(), output.data(), scales.data(), rows, cols, epsilon, nullptr, -1);

    for (int i = 0; i < rows * cols; ++i)
    {
        EXPECT_EQ(output[i], 0);
    }
    for (int i = 0; i < rows; ++i)
    {
        // Scale should be valid (e.g. 1.0 or something finite)
        EXPECT_TRUE(std::isfinite(scales[i]));
        EXPECT_GT(scales[i], 0.0f);
    }
}
