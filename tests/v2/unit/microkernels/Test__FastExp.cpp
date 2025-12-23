/**
 * @file Test__FastExp.cpp
 * @brief Unit tests for fast exponential microkernel
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "kernels/cpu/attention/q8_1/microkernels/FastExp.h"

#include <cmath>
#include <random>
#include <vector>

using namespace llaminar::v2::kernels::microkernels;

class Test__FastExp : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};
};

TEST_F(Test__FastExp, ExpZero)
{
    // exp(0) = 1
    float result = fast_exp_ref(0.0f);
    EXPECT_FLOAT_EQ(result, 1.0f);

    float poly_result = fast_exp_poly(0.0f);
    EXPECT_NEAR(poly_result, 1.0f, 1e-5f);

    float rr_result = fast_exp_range_reduced(0.0f);
    EXPECT_NEAR(rr_result, 1.0f, 1e-5f);
}

TEST_F(Test__FastExp, ExpOne)
{
    // exp(1) ≈ 2.71828...
    float expected = std::exp(1.0f);

    float ref_result = fast_exp_ref(1.0f);
    EXPECT_FLOAT_EQ(ref_result, expected);

    float poly_result = fast_exp_poly(1.0f);
    float poly_error = std::abs(poly_result - expected) / expected;
    EXPECT_LT(poly_error, 0.01f); // < 1% relative error

    float rr_result = fast_exp_range_reduced(1.0f);
    float rr_error = std::abs(rr_result - expected) / expected;
    EXPECT_LT(rr_error, 0.001f); // < 0.1% relative error
}

TEST_F(Test__FastExp, NegativeValues_SoftmaxRange)
{
    // Test typical softmax range: exp(score - max) where score - max < 0
    std::vector<float> test_values = {-0.5f, -1.0f, -2.0f, -3.0f, -5.0f, -10.0f};

    for (float x : test_values)
    {
        float expected = std::exp(x);
        float rr_result = fast_exp_range_reduced(x);

        float rel_error = std::abs(rr_result - expected) / expected;
        EXPECT_LT(rel_error, 0.001f) << "Failed for x=" << x;
    }
}

TEST_F(Test__FastExp, LargeNegative)
{
    // Very negative values should approach 0
    float result = fast_exp_range_reduced(-50.0f);
    EXPECT_GT(result, 0.0f);
    EXPECT_LT(result, 1e-20f);

    // Extreme negative (clamped)
    result = fast_exp_range_reduced(-100.0f);
    EXPECT_GE(result, 0.0f);
}

TEST_F(Test__FastExp, PositiveValues)
{
    // Test positive values
    std::vector<float> test_values = {0.5f, 1.0f, 2.0f, 3.0f, 5.0f};

    for (float x : test_values)
    {
        float expected = std::exp(x);
        float rr_result = fast_exp_range_reduced(x);

        float rel_error = std::abs(rr_result - expected) / expected;
        EXPECT_LT(rel_error, 0.001f) << "Failed for x=" << x;
    }
}

TEST_F(Test__FastExp, LargePositive)
{
    // Large positive values
    float result = fast_exp_range_reduced(50.0f);
    float expected = std::exp(50.0f);

    float rel_error = std::abs(result - expected) / expected;
    EXPECT_LT(rel_error, 0.01f);
}

TEST_F(Test__FastExp, ClampFunction)
{
    // Test clamping
    float clamped = clamp_for_exp(-200.0f);
    EXPECT_FLOAT_EQ(clamped, -88.0f);

    clamped = clamp_for_exp(200.0f);
    EXPECT_FLOAT_EQ(clamped, 88.0f);

    clamped = clamp_for_exp(5.0f);
    EXPECT_FLOAT_EQ(clamped, 5.0f);
}

TEST_F(Test__FastExp, PolynomialAccuracy_SmallX)
{
    // Polynomial is most accurate for small |x|
    for (float x = -1.0f; x <= 1.0f; x += 0.1f)
    {
        float expected = std::exp(x);
        float poly_result = fast_exp_poly(x);

        float rel_error = std::abs(poly_result - expected) / expected;
        EXPECT_LT(rel_error, 0.01f) << "Failed for x=" << x;
    }
}

TEST_F(Test__FastExp, RandomValues)
{
    // Random values in softmax-relevant range
    std::uniform_real_distribution<float> dist(-10.0f, 0.0f);

    for (int i = 0; i < 100; ++i)
    {
        float x = dist(rng_);
        float expected = std::exp(x);
        float result = fast_exp_range_reduced(x);

        float rel_error = (expected > 1e-10f)
                              ? std::abs(result - expected) / expected
                              : std::abs(result - expected);

        EXPECT_LT(rel_error, 0.001f) << "Failed for x=" << x;
    }
}

#if defined(__AVX512F__)

TEST_F(Test__FastExp, AVX512_Poly_MatchesScalar)
{
    // Test AVX-512 polynomial against scalar
    alignas(64) float inputs[16];
    alignas(64) float outputs[16];

    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (int i = 0; i < 16; ++i)
    {
        inputs[i] = dist(rng_);
    }

    __m512 x = _mm512_load_ps(inputs);
    __m512 result = fast_exp_poly_avx512(x);
    _mm512_store_ps(outputs, result);

    for (int i = 0; i < 16; ++i)
    {
        float expected = fast_exp_poly(inputs[i]);
        EXPECT_NEAR(outputs[i], expected, 1e-5f) << "Failed at index " << i;
    }
}

TEST_F(Test__FastExp, AVX512_MatchesReference)
{
    // Test AVX-512 implementation against std::exp
    alignas(64) float inputs[16];
    alignas(64) float outputs[16];

    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (int i = 0; i < 16; ++i)
    {
        inputs[i] = dist(rng_);
    }

    __m512 x = _mm512_load_ps(inputs);
    __m512 result = fast_exp_avx512(x);
    _mm512_store_ps(outputs, result);

    for (int i = 0; i < 16; ++i)
    {
        float expected = std::exp(inputs[i]);
        float rel_error = (expected > 1e-10f)
                              ? std::abs(outputs[i] - expected) / expected
                              : std::abs(outputs[i] - expected);

        EXPECT_LT(rel_error, 0.001f) << "Failed at index " << i
                                     << " (input=" << inputs[i] << ", got=" << outputs[i]
                                     << ", expected=" << expected << ")";
    }
}

TEST_F(Test__FastExp, AVX512_SoftmaxRange)
{
    // Test with values typical for softmax (score - max_score)
    alignas(64) float inputs[16] = {
        -0.1f, -0.5f, -1.0f, -1.5f, -2.0f, -2.5f, -3.0f, -4.0f,
        -5.0f, -6.0f, -7.0f, -8.0f, -9.0f, -10.0f, -15.0f, -20.0f};
    alignas(64) float outputs[16];

    __m512 x = _mm512_load_ps(inputs);
    __m512 result = fast_exp_avx512(x);
    _mm512_store_ps(outputs, result);

    for (int i = 0; i < 16; ++i)
    {
        float expected = std::exp(inputs[i]);
        float rel_error = (expected > 1e-10f)
                              ? std::abs(outputs[i] - expected) / expected
                              : std::abs(outputs[i] - expected);

        EXPECT_LT(rel_error, 0.001f) << "Failed at index " << i;
    }
}

#endif // __AVX512F__

TEST_F(Test__FastExp, SoftmaxApplication)
{
    // End-to-end test: compute softmax using fast_exp
    std::vector<float> scores = {2.0f, 1.0f, 0.1f, -1.0f, -2.0f};
    float max_score = *std::max_element(scores.begin(), scores.end());

    // Compute softmax with fast_exp
    float sum_exp = 0.0f;
    std::vector<float> exp_scores(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
    {
        exp_scores[i] = fast_exp_range_reduced(scores[i] - max_score);
        sum_exp += exp_scores[i];
    }

    std::vector<float> softmax_fast(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
    {
        softmax_fast[i] = exp_scores[i] / sum_exp;
    }

    // Compute reference softmax with std::exp
    float sum_exp_ref = 0.0f;
    std::vector<float> softmax_ref(scores.size());
    for (size_t i = 0; i < scores.size(); ++i)
    {
        softmax_ref[i] = std::exp(scores[i] - max_score);
        sum_exp_ref += softmax_ref[i];
    }
    for (auto &s : softmax_ref)
        s /= sum_exp_ref;

    // Compare
    for (size_t i = 0; i < scores.size(); ++i)
    {
        EXPECT_NEAR(softmax_fast[i], softmax_ref[i], 1e-5f)
            << "Softmax mismatch at index " << i;
    }

    // Verify sum is 1
    float sum = 0.0f;
    for (float s : softmax_fast)
        sum += s;
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}
