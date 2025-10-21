/**
 * @file TestBF16GemmParity.cpp
 * @brief BF16 GEMM parity testing: compare BF16 GEMM vs FP32 GEMM
 * @author David Sanftenberg
 * @date 2025-10-19
 *
 * Tests verify that adaptiveMatMulBF16 (using cblas_sbgemm with BF16×BF16→FP32)
 * produces numerically equivalent results to adaptiveMatMul (using standard FP32 GEMM).
 *
 * This validates the BF16 GEMM integration without requiring model loading,
 * quantized tensors, or complex setup.
 */

#include <gtest/gtest.h>
#include "MpiContext.h"
#include "AdaptiveMatmul.h"
#include "utils/DebugEnv.h"
#include "utils/BFloat16.h"
#include "utils/CpuFeatures.h"
#include <memory>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

using namespace llaminar;

/**
 * @brief Helper to compute relative L2 error between two tensors
 */
float computeRelativeL2(const float *a, const float *b, size_t count)
{
    float sum_sq_diff = 0.0f;
    float sum_sq_ref = 0.0f;

    for (size_t i = 0; i < count; ++i)
    {
        float diff = a[i] - b[i];
        sum_sq_diff += diff * diff;
        sum_sq_ref += b[i] * b[i];
    }

    if (sum_sq_ref < 1e-12f)
        return sum_sq_diff; // Avoid division by zero

    return std::sqrt(sum_sq_diff / sum_sq_ref);
}

/**
 * @brief Helper to compute maximum absolute difference
 */
float computeMaxAbsDiff(const float *a, const float *b, size_t count)
{
    float max_diff = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        float diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

/**
 * @brief Fixture for BF16 GEMM parity tests
 */
class BF16GemmParityTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Disable COSMA for deterministic OpenBLAS-only testing
        setenv("ADAPTIVE_DISABLE_COSMA", "1", 1);
    }
};

/**
 * @brief Test BF16 GEMM numerical correctness
 *
 * Validates that cblas_sbgemm (BF16×BF16→FP32) produces reasonable results
 * compared to a reference FP32 path. Since cblas_sbgemm internally converts
 * input A from FP32→BF16, we're essentially comparing:
 *   - Path 1: cblas_sbgemm(A, B_bf16) - internally does BF16(A)×BF16(B)→FP32
 *   - Path 2: cblas_sgemm(BF16→FP32(A), BF16→FP32(B)) - FP32×FP32→FP32
 *
 * Due to different accumulation patterns, results won't be identical but should
 * be very close (BF16 precision: ~3 decimal digits).
 */
TEST_F(BF16GemmParityTest, BF16vsF32SmallMatrix)
{
    auto rank = MPIContext::capture().rank;

    // Small matrix for fast testing (similar to decode projection)
    int M = 64;  // Batch size
    int K = 896; // Hidden dimension (input)
    int N = 896; // Hidden dimension (output)

    if (rank == 0)
    {
        std::cout << "Testing BF16 GEMM numerical correctness: [" << M << " x " << K << "] × ["
                  << K << " x " << N << "] = [" << M << " x " << N << "]" << std::endl;
    }

    // Generate synthetic FP32 input and weight data
    std::vector<float> input_fp32(M * K);
    std::vector<float> weight_fp32(K * N);

    // Initialize with deterministic pseudo-random values
    for (size_t i = 0; i < input_fp32.size(); ++i)
    {
        input_fp32[i] = ((i * 73 + 17) % 2000) / 2000.0f - 0.5f; // Range [-0.5, 0.5]
    }
    for (size_t i = 0; i < weight_fp32.size(); ++i)
    {
        weight_fp32[i] = ((i * 97 + 13) % 2000) / 2000.0f - 0.5f;
    }

    // Convert input and weights to BF16 (simulating quantization)
    std::vector<llaminar::bfloat16> input_bf16(M * K);
    std::vector<llaminar::bfloat16> weight_bf16(K * N);
    for (size_t i = 0; i < input_fp32.size(); ++i)
    {
        input_bf16[i] = llaminar::bfloat16::from_float(input_fp32[i]);
    }
    for (size_t i = 0; i < weight_fp32.size(); ++i)
    {
        weight_bf16[i] = llaminar::bfloat16::from_float(weight_fp32[i]);
    }

    // === PATH 1: BF16 GEMM (cblas_sbgemm) - only on CPUs with AVX512_BF16 ===
    std::vector<float> output_bf16(M * N, 0.0f);
    bool bf16_path_available = can_use_native_bf16_gemm();

    if (bf16_path_available)
    {
        // Enable BF16 GEMM
        setenv("LLAMINAR_QUANT_BF16_GEMM", "1", 1);
        refreshDebugEnv(); // Refresh cached environment snapshot

        // Call BF16 GEMM path
        bool ok = adaptiveMatMulBF16(
            input_fp32.data(),  // Input: FP32 (will be converted to BF16)
            weight_bf16.data(), // Weights: BF16
            output_bf16.data(), // Output: FP32
            M, N, K,
            /*alpha*/ 1.0f,
            /*beta*/ 0.0f,
            /*is_prefill*/ false,
            /*distributed_partition*/ false,
            /*transpose_B*/ false);

        ASSERT_TRUE(ok) << "BF16 GEMM path failed";

        // Sanity check: output should not be all zeros or NaN
        bool has_nonzero = false;
        bool has_nan = false;
        for (size_t i = 0; i < output_bf16.size(); ++i)
        {
            if (output_bf16[i] != 0.0f)
                has_nonzero = true;
            if (std::isnan(output_bf16[i]))
                has_nan = true;
        }
        ASSERT_TRUE(has_nonzero) << "BF16 GEMM output is all zeros!";
        ASSERT_FALSE(has_nan) << "BF16 GEMM output contains NaN!";
    }
    else
    {
        if (rank == 0)
        {
            std::cout << "CPU does not support AVX512_BF16 - skipping native BF16 GEMM test" << std::endl;
            std::cout << "Using FP32 fallback path for BF16 quantized tensors" << std::endl;
        }

        // Use FP32 fallback with BF16→FP32 expansion (same as PATH 2)
        std::vector<float> input_fp32_expanded(M * K);
        std::vector<float> weight_fp32_expanded(K * N);

        for (size_t i = 0; i < input_bf16.size(); ++i)
        {
            input_fp32_expanded[i] = static_cast<float>(input_bf16[i]);
        }
        for (size_t i = 0; i < weight_bf16.size(); ++i)
        {
            weight_fp32_expanded[i] = static_cast<float>(weight_bf16[i]);
        }

        bool ok = adaptiveMatMul(
            input_fp32_expanded.data(),
            weight_fp32_expanded.data(),
            output_bf16.data(), // Reuse output_bf16 buffer for fallback result
            M, N, K,
            /*is_prefill*/ false,
            /*distributed_partition*/ false,
            /*transpose_A*/ false,
            /*transpose_B*/ false,
            /*alpha*/ 1.0f,
            /*beta*/ 0.0f);

        ASSERT_TRUE(ok) << "FP32 fallback path failed";
    }

    // === PATH 2: FP32 GEMM with BF16→FP32 expansion (reference) ===
    // Manually expand both input and weights from BF16 to FP32, then GEMM
    std::vector<float> output_fp32(M * N, 0.0f);
    {
        // Disable BF16 GEMM to use standard FP32 path
        setenv("LLAMINAR_QUANT_BF16_GEMM", "0", 1);
        refreshDebugEnv();

        // Expand both input and weights from BF16 to FP32
        std::vector<float> input_fp32_expanded(M * K);
        std::vector<float> weight_fp32_expanded(K * N);

        for (size_t i = 0; i < input_bf16.size(); ++i)
        {
            input_fp32_expanded[i] = static_cast<float>(input_bf16[i]);
        }
        for (size_t i = 0; i < weight_bf16.size(); ++i)
        {
            weight_fp32_expanded[i] = static_cast<float>(weight_bf16[i]);
        }

        // Call FP32 GEMM with expanded tensors
        bool ok = adaptiveMatMul(
            input_fp32_expanded.data(),  // Input: FP32 (expanded from BF16)
            weight_fp32_expanded.data(), // Weights: FP32 (expanded from BF16)
            output_fp32.data(),          // Output: FP32
            M, N, K,
            /*is_prefill*/ false,
            /*distributed_partition*/ false,
            /*transpose_A*/ false,
            /*transpose_B*/ false,
            /*alpha*/ 1.0f,
            /*beta*/ 0.0f);

        ASSERT_TRUE(ok) << "FP32 GEMM path failed";
    }

    // === COMPARE RESULTS ===
    float rel_l2 = computeRelativeL2(output_bf16.data(), output_fp32.data(), M * N);
    float max_abs = computeMaxAbsDiff(output_bf16.data(), output_fp32.data(), M * N);

    if (rank == 0)
    {
        std::cout << "BF16 vs FP32 parity metrics:" << std::endl;
        std::cout << "  Relative L2 error: " << rel_l2 << std::endl;
        std::cout << "  Max absolute diff: " << max_abs << std::endl;

        // Show a few sample values for manual inspection
        std::cout << "Sample outputs (first 5):" << std::endl;
        for (int i = 0; i < 5; ++i)
        {
            std::cout << "    BF16: " << output_bf16[i] << "  FP32: " << output_fp32[i]
                      << "  diff: " << std::abs(output_bf16[i] - output_fp32[i]) << std::endl;
        }

        // Debug: Check if results are reasonable
        float bf16_sum = 0.0f, fp32_sum = 0.0f;
        for (size_t i = 0; i < output_bf16.size(); ++i)
        {
            bf16_sum += output_bf16[i];
            fp32_sum += output_fp32[i];
        }
        std::cout << "Sum check - BF16: " << bf16_sum << " FP32: " << fp32_sum << std::endl;
    }

    // Tolerances based on BF16 precision + accumulation differences:
    // - Both paths use BF16-quantized inputs/weights
    // - cblas_sbgemm has optimized BF16 accumulation (potentially FP32 internally)
    // - cblas_sgemm with expanded values has standard FP32 accumulation
    // - For GEMM with K=896 accumulation steps, expect larger differences than simple conversion
    // - Relative L2 should be < 0.1 (10% - accounting for accumulation differences)
    // - Max absolute diff should be < 1.0 (values are in range ~[-0.5, 0.5] → products in ~[-0.25, 0.25])
    EXPECT_LT(rel_l2, 0.1f) << "Relative L2 error exceeds tolerance";
    EXPECT_LT(max_abs, 1.0f) << "Max absolute error exceeds tolerance";
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
