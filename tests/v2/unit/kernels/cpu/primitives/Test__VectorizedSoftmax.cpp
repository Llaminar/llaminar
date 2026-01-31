/**
 * @file Test__VectorizedSoftmax.cpp
 * @brief Correctness and performance tests for vectorized softmax implementations
 *
 * Tests:
 * 1. ISA Parity: AVX512 == AVX2 == Scalar for same inputs
 * 2. Numerical Stability: Large/small values, edge cases
 * 3. Performance: Measure speedups vs scalar baseline
 *
 * @author David Sanftenberg
 * @date November 2025
 */

#include <gtest/gtest.h>
#include "kernels/cpu/gemm/int8/VectorizedSoftmax.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>

using namespace llaminar2::kernels::gemm;
using namespace llaminar2::kernels::simd;

namespace
{
    /**
     * @brief Helper to compare two softmax outputs
     *
     * Note: Tolerance relaxed to 1e-3 for fast exp approximation
     * (polynomial approximation has ~1e-5 relative error, which compounds in softmax)
     */
    bool softmax_outputs_match(const std::vector<float> &a, const std::vector<float> &b, float tolerance = 1e-3f)
    {
        if (a.size() != b.size())
            return false;

        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            float diff = std::abs(a[i] - b[i]);
            max_abs_diff = std::max(max_abs_diff, diff);
        }

        return max_abs_diff <= tolerance;
    }

    /**
     * @brief Helper to verify softmax properties
     */
    void verify_softmax_properties(const std::vector<float> &output, const char *test_name)
    {
        // All values should be in [0, 1]
        for (float val : output)
        {
            EXPECT_GE(val, 0.0f) << test_name << ": Negative probability";
            EXPECT_LE(val, 1.0f) << test_name << ": Probability > 1";
        }

        // Sum should be ~1.0
        float sum = 0.0f;
        for (float val : output)
        {
            sum += val;
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << test_name << ": Sum != 1.0";
    }
}

// ============================================================================
// ISA Parity Tests: Verify all implementations produce identical results
// ============================================================================

class Test__VectorizedSoftmax : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(Test__VectorizedSoftmax, ISA_Parity_UniformInput)
{
    const int n = 32;
    std::vector<float> input(n, 1.0f);

    std::vector<float> output_scalar(n);
    std::vector<float> output_avx2(n);
    std::vector<float> output_avx512(n);

    VectorizedSoftmax<ScalarTag>::apply(input.data(), output_scalar.data(), n);

#if defined(__AVX2__)
    VectorizedSoftmax<AVX2Tag>::apply(input.data(), output_avx2.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx2))
        << "AVX2 != Scalar for uniform input";
    verify_softmax_properties(output_avx2, "AVX2 Uniform");
#endif

#if defined(__AVX512F__)
    VectorizedSoftmax<AVX512Tag>::apply(input.data(), output_avx512.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx512))
        << "AVX512 != Scalar for uniform input";
    verify_softmax_properties(output_avx512, "AVX512 Uniform");
#endif

    verify_softmax_properties(output_scalar, "Scalar Uniform");

    // Uniform input should produce uniform output (1/n for each element)
    float expected = 1.0f / n;
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NEAR(output_scalar[i], expected, 1e-5f);
    }
}

TEST_F(Test__VectorizedSoftmax, ISA_Parity_RandomInput)
{
    const int n = 32;
    std::vector<float> input(n);

    // Generate random inputs in range [-10, 10]
    for (int i = 0; i < n; ++i)
    {
        input[i] = (static_cast<float>(rand()) / RAND_MAX) * 20.0f - 10.0f;
    }

    std::vector<float> output_scalar(n);
    std::vector<float> output_avx2(n);
    std::vector<float> output_avx512(n);

    VectorizedSoftmax<ScalarTag>::apply(input.data(), output_scalar.data(), n);

#if defined(__AVX2__)
    VectorizedSoftmax<AVX2Tag>::apply(input.data(), output_avx2.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx2, 1e-3f))
        << "AVX2 != Scalar for random input";
    verify_softmax_properties(output_avx2, "AVX2 Random");
#endif

#if defined(__AVX512F__)
    VectorizedSoftmax<AVX512Tag>::apply(input.data(), output_avx512.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx512, 1e-3f))
        << "AVX512 != Scalar for random input";
    verify_softmax_properties(output_avx512, "AVX512 Random");
#endif

    verify_softmax_properties(output_scalar, "Scalar Random");
}

TEST_F(Test__VectorizedSoftmax, ISA_Parity_LargeValues)
{
    const int n = 32;
    std::vector<float> input(n);

    // Test with large values (potential overflow without max subtraction)
    for (int i = 0; i < n; ++i)
    {
        input[i] = 50.0f + i;
    }

    std::vector<float> output_scalar(n);
    std::vector<float> output_avx2(n);
    std::vector<float> output_avx512(n);

    VectorizedSoftmax<ScalarTag>::apply(input.data(), output_scalar.data(), n);

#if defined(__AVX2__)
    VectorizedSoftmax<AVX2Tag>::apply(input.data(), output_avx2.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx2, 1e-3f))
        << "AVX2 != Scalar for large values";
    verify_softmax_properties(output_avx2, "AVX2 Large");
#endif

#if defined(__AVX512F__)
    VectorizedSoftmax<AVX512Tag>::apply(input.data(), output_avx512.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx512, 1e-3f))
        << "AVX512 != Scalar for large values";
    verify_softmax_properties(output_avx512, "AVX512 Large");
#endif

    verify_softmax_properties(output_scalar, "Scalar Large");
}

TEST_F(Test__VectorizedSoftmax, ISA_Parity_NegativeValues)
{
    const int n = 32;
    std::vector<float> input(n);

    // Test with negative values
    for (int i = 0; i < n; ++i)
    {
        input[i] = -20.0f + i * 0.5f;
    }

    std::vector<float> output_scalar(n);
    std::vector<float> output_avx2(n);
    std::vector<float> output_avx512(n);

    VectorizedSoftmax<ScalarTag>::apply(input.data(), output_scalar.data(), n);

#if defined(__AVX2__)
    VectorizedSoftmax<AVX2Tag>::apply(input.data(), output_avx2.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx2, 1e-3f))
        << "AVX2 != Scalar for negative values";
    verify_softmax_properties(output_avx2, "AVX2 Negative");
#endif

#if defined(__AVX512F__)
    VectorizedSoftmax<AVX512Tag>::apply(input.data(), output_avx512.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx512, 1e-3f))
        << "AVX512 != Scalar for negative values";
    verify_softmax_properties(output_avx512, "AVX512 Negative");
#endif

    verify_softmax_properties(output_scalar, "Scalar Negative");
}

TEST_F(Test__VectorizedSoftmax, ISA_Parity_NonMultipleOf16)
{
    // Test n=33 (not multiple of 16, tests remainder handling)
    const int n = 33;
    std::vector<float> input(n);

    for (int i = 0; i < n; ++i)
    {
        input[i] = static_cast<float>(i) * 0.3f;
    }

    std::vector<float> output_scalar(n);
    std::vector<float> output_avx2(n);
    std::vector<float> output_avx512(n);

    VectorizedSoftmax<ScalarTag>::apply(input.data(), output_scalar.data(), n);

#if defined(__AVX2__)
    VectorizedSoftmax<AVX2Tag>::apply(input.data(), output_avx2.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx2, 1e-3f))
        << "AVX2 != Scalar for n=33 (remainder handling)";
    verify_softmax_properties(output_avx2, "AVX2 n=33");
#endif

#if defined(__AVX512F__)
    VectorizedSoftmax<AVX512Tag>::apply(input.data(), output_avx512.data(), n);
    EXPECT_TRUE(softmax_outputs_match(output_scalar, output_avx512, 1e-3f))
        << "AVX512 != Scalar for n=33 (remainder handling)";
    verify_softmax_properties(output_avx512, "AVX512 n=33");
#endif

    verify_softmax_properties(output_scalar, "Scalar n=33");
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST_F(Test__VectorizedSoftmax, Performance_N32)
{
    const int n = 32;
    const int iterations = 100000;

    std::vector<float> input(n);
    std::vector<float> output(n);

    // Random input
    for (int i = 0; i < n; ++i)
    {
        input[i] = (static_cast<float>(rand()) / RAND_MAX) * 20.0f - 10.0f;
    }

    // Benchmark Scalar
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        VectorizedSoftmax<ScalarTag>::apply(input.data(), output.data(), n);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto scalar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_scalar - start_scalar).count();
    double scalar_per_call = static_cast<double>(scalar_ns) / iterations;

    std::cout << "\n=== Vectorized Softmax Performance (n=" << n << ") ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Scalar:  " << scalar_per_call << " ns/call\n";

#if defined(__AVX2__)
    // Benchmark AVX2
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        VectorizedSoftmax<AVX2Tag>::apply(input.data(), output.data(), n);
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    auto avx2_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_avx2 - start_avx2).count();
    double avx2_per_call = static_cast<double>(avx2_ns) / iterations;
    double avx2_speedup = scalar_per_call / avx2_per_call;

    std::cout << "AVX2:    " << avx2_per_call << " ns/call (" << avx2_speedup << "× speedup)\n";
    EXPECT_GT(avx2_speedup, 1.5) << "AVX2 should be at least 1.5× faster than scalar";
#endif

#if defined(__AVX512F__)
    // Benchmark AVX512
    auto start_avx512 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        VectorizedSoftmax<AVX512Tag>::apply(input.data(), output.data(), n);
    }
    auto end_avx512 = std::chrono::high_resolution_clock::now();
    auto avx512_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_avx512 - start_avx512).count();
    double avx512_per_call = static_cast<double>(avx512_ns) / iterations;
    double avx512_speedup = scalar_per_call / avx512_per_call;

    std::cout << "AVX512:  " << avx512_per_call << " ns/call (" << avx512_speedup << "× speedup)\n";
    EXPECT_GT(avx512_speedup, 2.0) << "AVX512 should be at least 2× faster than scalar";
#endif

    std::cout << "===========================================\n\n";
}

TEST_F(Test__VectorizedSoftmax, Performance_N128)
{
    const int n = 128;
    const int iterations = 50000;

    std::vector<float> input(n);
    std::vector<float> output(n);

    // Random input
    for (int i = 0; i < n; ++i)
    {
        input[i] = (static_cast<float>(rand()) / RAND_MAX) * 20.0f - 10.0f;
    }

    // Benchmark Scalar
    auto start_scalar = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        VectorizedSoftmax<ScalarTag>::apply(input.data(), output.data(), n);
    }
    auto end_scalar = std::chrono::high_resolution_clock::now();
    auto scalar_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_scalar - start_scalar).count();
    double scalar_per_call = static_cast<double>(scalar_ns) / iterations;

    std::cout << "\n=== Vectorized Softmax Performance (n=" << n << ") ===\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Scalar:  " << scalar_per_call << " ns/call\n";

#if defined(__AVX2__)
    // Benchmark AVX2
    auto start_avx2 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        VectorizedSoftmax<AVX2Tag>::apply(input.data(), output.data(), n);
    }
    auto end_avx2 = std::chrono::high_resolution_clock::now();
    auto avx2_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_avx2 - start_avx2).count();
    double avx2_per_call = static_cast<double>(avx2_ns) / iterations;
    double avx2_speedup = scalar_per_call / avx2_per_call;

    std::cout << "AVX2:    " << avx2_per_call << " ns/call (" << avx2_speedup << "× speedup)\n";
    EXPECT_GT(avx2_speedup, 1.5) << "AVX2 should be at least 1.5× faster than scalar";
#endif

#if defined(__AVX512F__)
    // Benchmark AVX512
    auto start_avx512 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter)
    {
        VectorizedSoftmax<AVX512Tag>::apply(input.data(), output.data(), n);
    }
    auto end_avx512 = std::chrono::high_resolution_clock::now();
    auto avx512_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_avx512 - start_avx512).count();
    double avx512_per_call = static_cast<double>(avx512_ns) / iterations;
    double avx512_speedup = scalar_per_call / avx512_per_call;

    std::cout << "AVX512:  " << avx512_per_call << " ns/call (" << avx512_speedup << "× speedup)\n";
    EXPECT_GT(avx512_speedup, 2.0) << "AVX512 should be at least 2× faster than scalar";
#endif

    std::cout << "===========================================\n\n";
}
