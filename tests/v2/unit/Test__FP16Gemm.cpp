/**
 * @file Test__FP16Gemm.cpp
 * @brief Unit tests for FP16×FP16→FP32 GEMM implementations
 * @author David Sanftenberg
 */

#include "../../../src/v2/kernels/cpu/gemm/fp16/FP16GemmImpl.h"
#include "../../../src/v2/tensors/FP16Utils.h"
#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <cmath>

using namespace llaminar2::kernels::gemm::fp16_impl;

namespace
{

    /**
     * @brief Generate random FP16 values in range [-1, 1]
     */
    void fill_random_fp16(uint16_t *data, size_t count, int seed = 42)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (size_t i = 0; i < count; ++i)
        {
            float val = dist(rng);
            data[i] = llaminar2::fp32_to_fp16(val);
        }
    }

    /**
     * @brief Compare two float arrays with relative tolerance
     */
    void compare_arrays(const float *expected, const float *actual, size_t count,
                        float rel_tol = 1e-3f, float abs_tol = 1e-4f)
    {
        size_t mismatches = 0;
        float max_rel_error = 0.0f;

        for (size_t i = 0; i < count; ++i)
        {
            float exp_val = expected[i];
            float act_val = actual[i];

            float abs_error = std::abs(exp_val - act_val);
            float rel_error = abs_error / (std::abs(exp_val) + 1e-8f);

            max_rel_error = std::max(max_rel_error, rel_error);

            if (abs_error > abs_tol && rel_error > rel_tol)
            {
                mismatches++;
                if (mismatches <= 5)
                {
                    std::cerr << "Mismatch at index " << i << ": "
                              << "expected=" << exp_val << ", "
                              << "actual=" << act_val << ", "
                              << "rel_error=" << rel_error << std::endl;
                }
            }
        }

        EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches, "
                                 << "max_rel_error=" << max_rel_error;
    }

} // anonymous namespace

/**
 * @brief Test suite for FP16 GEMM kernels
 */
class FP16GemmTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Seed for reproducible tests
        seed_ = 42;
    }

    int seed_;
};

/**
 * @brief Test basic correctness of scalar FP16 GEMM
 */
TEST_F(FP16GemmTest, ScalarBasicCorrectness)
{
    const int m = 4, n = 4, k = 8;

    std::vector<uint16_t> A(m * k);
    std::vector<uint16_t> B(n * k);
    std::vector<float> C_scalar(m * n);

    fill_random_fp16(A.data(), m * k, seed_);
    fill_random_fp16(B.data(), n * k, seed_ + 1);

    // Compute with scalar implementation
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n, 1.0f, 0.0f);

    // Verify against manual computation (first element)
    float expected_c00 = 0.0f;
    for (int p = 0; p < k; ++p)
    {
        float a_val = llaminar2::fp16_to_fp32(A[p]);
        float b_val = llaminar2::fp16_to_fp32(B[p]);
        expected_c00 += a_val * b_val;
    }

    EXPECT_NEAR(C_scalar[0], expected_c00, 1e-3f);
}

/**
 * @brief Test alpha/beta scaling
 */
TEST_F(FP16GemmTest, AlphaBetaScaling)
{
    const int m = 4, n = 4, k = 8;
    const float alpha = 2.0f, beta = 0.5f;

    std::vector<uint16_t> A(m * k);
    std::vector<uint16_t> B(n * k);
    std::vector<float> C(m * n, 1.0f); // Initialize to 1.0 for beta test
    std::vector<float> C_expected(m * n);

    fill_random_fp16(A.data(), m * k, seed_);
    fill_random_fp16(B.data(), n * k, seed_ + 1);

    // Compute expected: C = alpha * A * B^T + beta * C_old
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p)
            {
                float a_val = llaminar2::fp16_to_fp32(A[i * k + p]);
                float b_val = llaminar2::fp16_to_fp32(B[j * k + p]);
                sum += a_val * b_val;
            }
            C_expected[i * n + j] = alpha * sum + beta * 1.0f;
        }
    }

    // Compute with kernel
    gemm_scalar_impl(A.data(), B.data(), C.data(), m, n, k, n, alpha, beta);

    compare_arrays(C_expected.data(), C.data(), m * n, 1e-3f, 1e-4f);
}

#if defined(__AVX2__) && defined(__F16C__)
/**
 * @brief Test AVX2+F16C implementation against scalar
 */
TEST_F(FP16GemmTest, AVX2_F16C_Correctness)
{
    const int m = 16, n = 16, k = 64;

    std::vector<uint16_t> A(m * k);
    std::vector<uint16_t> B(n * k);
    std::vector<float> C_scalar(m * n);
    std::vector<float> C_avx2(m * n);

    fill_random_fp16(A.data(), m * k, seed_);
    fill_random_fp16(B.data(), n * k, seed_ + 1);

    // Compute with both implementations
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n, 1.0f, 0.0f);
    gemm_avx2_f16c_impl(A.data(), B.data(), C_avx2.data(), m, n, k, n, 1.0f, 0.0f);

    // Compare results (FP16 has limited precision)
    compare_arrays(C_scalar.data(), C_avx2.data(), m * n, 1e-5f, 1e-6f);
}
#endif

#if defined(__AVX512F__)
/**
 * @brief Test AVX512F implementation against scalar
 */
TEST_F(FP16GemmTest, AVX512F_Correctness)
{
    const int m = 32, n = 32, k = 128;

    std::vector<uint16_t> A(m * k);
    std::vector<uint16_t> B(n * k);
    std::vector<float> C_scalar(m * n);
    std::vector<float> C_avx512(m * n);

    fill_random_fp16(A.data(), m * k, seed_);
    fill_random_fp16(B.data(), n * k, seed_ + 1);

    // Compute with both implementations
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n, 1.0f, 0.0f);
    gemm_avx512f_impl(A.data(), B.data(), C_avx512.data(), m, n, k, n, 1.0f, 0.0f);

    // Compare results (FP16 has limited precision)
    compare_arrays(C_scalar.data(), C_avx512.data(), m * n, 1e-5f, 1e-6f);
}
#endif

/**
 * @brief Test auto-dispatch selects correct implementation
 */
TEST_F(FP16GemmTest, AutoDispatch)
{
    const int m = 8, n = 8, k = 32;

    std::vector<uint16_t> A(m * k);
    std::vector<uint16_t> B(n * k);
    std::vector<float> C_auto(m * n);
    std::vector<float> C_scalar(m * n);

    fill_random_fp16(A.data(), m * k, seed_);
    fill_random_fp16(B.data(), n * k, seed_ + 1);

    // Compute with auto-dispatch
    const char *isa = gemm_fp16_auto(A.data(), B.data(), C_auto.data(), m, n, k, n, 1.0f, 0.0f);

    // Compute with scalar for reference
    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n, 1.0f, 0.0f);

    std::cout << "Auto-dispatch selected: " << isa << std::endl;

    // Results should match regardless of ISA
    compare_arrays(C_scalar.data(), C_auto.data(), m * n, 1e-5f, 1e-6f);
}

/**
 * @brief Test edge case: single element
 */
TEST_F(FP16GemmTest, SingleElement)
{
    const int m = 1, n = 1, k = 1;

    uint16_t A[1] = {llaminar2::fp32_to_fp16(2.0f)};
    uint16_t B[1] = {llaminar2::fp32_to_fp16(3.0f)};
    float C[1] = {0.0f};

    gemm_scalar_impl(A, B, C, m, n, k, n, 1.0f, 0.0f);

    EXPECT_NEAR(C[0], 6.0f, 1e-3f);
}

/**
 * @brief Test edge case: K not multiple of vector width
 */
TEST_F(FP16GemmTest, NonAlignedK)
{
    // K = 13 (not divisible by 8 or 16)
    const int m = 4, n = 4, k = 13;

    std::vector<uint16_t> A(m * k);
    std::vector<uint16_t> B(n * k);
    std::vector<float> C_scalar(m * n);
    std::vector<float> C_auto(m * n);

    fill_random_fp16(A.data(), m * k, seed_);
    fill_random_fp16(B.data(), n * k, seed_ + 1);

    gemm_scalar_impl(A.data(), B.data(), C_scalar.data(), m, n, k, n, 1.0f, 0.0f);
    gemm_fp16_auto(A.data(), B.data(), C_auto.data(), m, n, k, n, 1.0f, 0.0f);

    compare_arrays(C_scalar.data(), C_auto.data(), m * n, 1e-5f, 1e-6f);
}

/**
 * @brief Test large matrix
 */
TEST_F(FP16GemmTest, LargeMatrix)
{
    const int m = 128, n = 128, k = 256;

    std::vector<uint16_t> A(m * k);
    std::vector<uint16_t> B(n * k);
    std::vector<float> C(m * n);

    fill_random_fp16(A.data(), m * k, seed_);
    fill_random_fp16(B.data(), n * k, seed_ + 1);

    // Just verify it doesn't crash
    const char *isa = gemm_fp16_auto(A.data(), B.data(), C.data(), m, n, k, n, 1.0f, 0.0f);

    std::cout << "Large matrix test completed with ISA: " << isa << std::endl;

    // Sanity check: verify first element is reasonable
    EXPECT_FALSE(std::isnan(C[0]));
    EXPECT_FALSE(std::isinf(C[0]));
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
