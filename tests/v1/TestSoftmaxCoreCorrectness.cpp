/**
 * @file test_softmax_core_parity.cpp
 * @brief Parity regression tests for unified softmax core.
 */
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include <cmath>
#include "operators/common/SoftmaxCore.h"

using namespace llaminar::kernels;

static void reference_softmax(std::vector<float> &data, int rows, int cols, bool causal, float scale)
{
    for (int r = 0; r < rows; ++r)
    {
        float *row = data.data() + size_t(r) * cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c)
        {
            bool masked = causal && c > r;
            if (masked)
                continue;
            float v = row[c];
            if (scale != 1.f)
                v *= scale;
            m = std::max(m, v);
        }
        if (!std::isfinite(m))
            m = 0.f;
        double s = 0.0;
        for (int c = 0; c < cols; ++c)
        {
            bool masked = causal && c > r;
            if (masked)
            {
                row[c] = 0.f;
                continue;
            }
            float v = row[c];
            if (scale != 1.f)
                v *= scale;
            float e = std::exp(v - m);
            row[c] = e;
            s += e;
        }
        if (s <= 0.0)
            s = 1.0;
        float inv = float(1.0 / s);
        for (int c = 0; c < cols; ++c)
            row[c] *= inv;
    }
}

static double rel_l2(const std::vector<float> &a, const std::vector<float> &b)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double d = double(a[i]) - double(b[i]);
        num += d * d;
        den += double(a[i]) * double(a[i]);
    }
    if (den == 0.0)
        den = 1.0;
    return std::sqrt(num / den);
}

class SoftmaxCoreTest : public ::testing::TestWithParam<std::tuple<int, int, bool, float>>
{
};

TEST_P(SoftmaxCoreTest, RowMajorParity)
{
    auto [rows, cols, causal, scale] = GetParam();
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-3.f, 3.f);
    std::vector<float> baseline(size_t(rows) * cols);
    for (auto &v : baseline)
        v = dist(rng);
    auto test = baseline;

    // Reference
    reference_softmax(baseline, rows, cols, causal, scale);

    // Core
    SoftmaxRowArgs args;
    args.scores = test.data();
    args.rows = rows;
    args.cols = cols;
    args.causal = causal;
    args.scale = scale;
    softmax_row_major(args);

    double rl2 = rel_l2(baseline, test);
    ASSERT_LT(rl2, 1e-7) << "Relative L2 too large rl2=" << rl2;
}

INSTANTIATE_TEST_SUITE_P(SoftmaxShapes, SoftmaxCoreTest,
                         ::testing::Values(
                             std::make_tuple(1, 16, false, 1.0f),
                             std::make_tuple(4, 32, true, 1.0f),
                             std::make_tuple(8, 64, false, 0.5f),
                             std::make_tuple(16, 16, true, 1.0f)));
