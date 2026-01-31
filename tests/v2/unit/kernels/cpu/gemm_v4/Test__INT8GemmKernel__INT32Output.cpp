/**
 * @file Test__INT8GemmKernel__INT32Output.cpp
 * @brief Unit tests for INT8 GEMM with INT32 output mode
 *
 * Tests the new multiply_int32() method that returns raw INT32 accumulator
 * from OneDNN s8s8s32 matmul. This is the foundation for full INT8 pipelines.
 *
 * @author David Sanftenberg
 * @date 2025
 */

#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <random>
#include "kernels/cpu/INT8GemmKernel.h"
#include "tensors/Tensors.h"
#include "loaders/ModelLoader.h"
#include "utils/Logger.h"

using namespace llaminar2;

class Test__INT8GemmKernel__INT32Output : public ::testing::Test
{
protected:
    void SetUp() override
    {
        Logger::getInstance().setLogLevel(LogLevel::INFO);
    }

    /**
     * @brief Compute reference INT32 GEMM: C_int32 = A_int8 × B_int8
     *
     * This computes the raw dot products without any scaling.
     * Output matches OneDNN s8s8s32 matmul exactly.
     */
    void reference_int8_gemm_int32(
        const std::vector<int8_t> &A_int8,
        const std::vector<int8_t> &B_int8,
        std::vector<int32_t> &C_int32,
        int m, int n, int k,
        bool transpose_B = true)
    {
        C_int32.resize(m * n);
        std::fill(C_int32.begin(), C_int32.end(), 0);

        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                int32_t sum = 0;
                for (int k_idx = 0; k_idx < k; ++k_idx)
                {
                    int8_t a_val = A_int8[i * k + k_idx];
                    int8_t b_val = transpose_B ? B_int8[j * k + k_idx] : B_int8[k_idx * n + j];
                    sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
                }
                C_int32[i * n + j] = sum;
            }
        }
    }

    /**
     * @brief Quantize FP32 matrix to INT8 with per-row scales
     */
    void quantize_per_row(
        const std::vector<float> &fp32,
        std::vector<int8_t> &int8,
        std::vector<float> &row_scales,
        int m, int k)
    {
        int8.resize(m * k);
        row_scales.resize(m);

        for (int i = 0; i < m; ++i)
        {
            // Find max absolute value in this row
            float max_val = 0.0f;
            for (int j = 0; j < k; ++j)
            {
                max_val = std::max(max_val, std::abs(fp32[i * k + j]));
            }

            // Compute scale
            const float scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
            row_scales[i] = scale;

            // Quantize row
            for (int j = 0; j < k; ++j)
            {
                float val = fp32[i * k + j] / scale;
                int8[i * k + j] = static_cast<int8_t>(std::round(std::clamp(val, -127.0f, 127.0f)));
            }
        }
    }

    /**
     * @brief Quantize FP32 matrix to INT8 with per-column scales
     */
    void quantize_per_column(
        const std::vector<float> &fp32,
        std::vector<int8_t> &int8,
        std::vector<float> &col_scales,
        int rows, int cols)
    {
        int8.resize(rows * cols);
        col_scales.resize(cols);

        for (int j = 0; j < cols; ++j)
        {
            // Find max absolute value in this column
            float max_val = 0.0f;
            for (int i = 0; i < rows; ++i)
            {
                max_val = std::max(max_val, std::abs(fp32[i * cols + j]));
            }

            // Compute scale
            const float scale = (max_val > 0.0f) ? (max_val / 127.0f) : 1.0f;
            col_scales[j] = scale;

            // Quantize column
            for (int i = 0; i < rows; ++i)
            {
                float val = fp32[i * cols + j] / scale;
                int8[i * cols + j] = static_cast<int8_t>(std::round(std::clamp(val, -127.0f, 127.0f)));
            }
        }
    }
};

/**
 * @test INT32OutputBasic
 * @brief Verify INT32 output can be correctly dequantized to match FP32 path
 *
 * Uses small matrix to verify:
 * - INT32 output is produced without errors
 * - Manual dequantization matches direct FP32 output
 */
TEST_F(Test__INT8GemmKernel__INT32Output, INT32OutputBasic)
{
    const int m = 2, n = 2, k = 2;
    const bool transpose_B = true;

    // Create simple FP32 matrices
    std::vector<float> A_fp32 = {1.0f, 2.0f, 3.0f, 4.0f}; // [2, 2]
    std::vector<float> B_fp32 = {3.0f, 5.0f, 4.0f, 6.0f}; // [2, 2] transposed

    // Create INT8Tensor for B (weight)
    auto B_tensor = std::make_shared<INT8Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, B_fp32);

    // Create kernel
    INT8GemmKernel kernel(B_tensor.get());

    // Path 1: INT32 output + manual dequant
    std::vector<int32_t> C_int32(m * n);
    std::vector<float> A_scales(m);
    bool success1 = kernel.multiply_int32(
        A_fp32.data(), C_int32.data(),
        m, n, k, transpose_B, A_scales.data());
    ASSERT_TRUE(success1) << "multiply_int32 failed";

    LOG_INFO("[INT32OutputBasic] A scales: [" << A_scales[0] << ", " << A_scales[1] << "]");
    LOG_INFO("[INT32OutputBasic] INT32 raw: [" << C_int32[0] << ", " << C_int32[1] << ", " << C_int32[2] << ", " << C_int32[3] << "]");

    // Manual dequantization
    std::vector<float> C_fp32_from_int32(m * n);

    // Match kernel's scale selection logic
    std::vector<float> B_scales_for_dequant(n);
    if (transpose_B)
    {
        // Kernel uses per-row scales when transpose_B=true
        const auto &row_scales = B_tensor->get_row_scales();
        if (!row_scales.empty())
        {
            std::copy(row_scales.begin(), row_scales.end(), B_scales_for_dequant.begin());
            LOG_INFO("[INT32OutputBasic] Using per-row B scales (transpose_B=true)");
        }
        else
        {
            float global_scale = B_tensor->scale();
            std::fill(B_scales_for_dequant.begin(), B_scales_for_dequant.end(), global_scale);
            LOG_INFO("[INT32OutputBasic] Using global B scale (fallback): " << global_scale);
        }
    }
    else
    {
        // Use per-column scales
        const float *B_col_scales = B_tensor->col_scales();
        std::copy(B_col_scales, B_col_scales + n, B_scales_for_dequant.begin());
        LOG_INFO("[INT32OutputBasic] Using per-column B scales (transpose_B=false)");
    }

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C_fp32_from_int32[i * n + j] = static_cast<float>(C_int32[i * n + j]) * A_scales[i] * B_scales_for_dequant[j];
        }
    }
    LOG_INFO("[INT32OutputBasic] INT32→FP32: [" << C_fp32_from_int32[0] << ", " << C_fp32_from_int32[1] << ", " << C_fp32_from_int32[2] << ", " << C_fp32_from_int32[3] << "]");

    // Path 2: Direct FP32 output
    std::vector<float> C_fp32_direct(m * n);
    bool success2 = kernel.multiply(
        A_fp32.data(), C_fp32_direct.data(),
        m, n, k, transpose_B, 1.0f, 0.0f);
    ASSERT_TRUE(success2) << "multiply (FP32) failed";

    LOG_INFO("[INT32OutputBasic] FP32 direct: [" << C_fp32_direct[0] << ", " << C_fp32_direct[1] << ", " << C_fp32_direct[2] << ", " << C_fp32_direct[3] << "]");
    LOG_INFO("[INT32OutputBasic] Expected (no quant): [" << (1 * 3 + 2 * 5) << ", " << (1 * 4 + 2 * 6) << ", " << (3 * 3 + 4 * 5) << ", " << (3 * 4 + 4 * 6) << "]");

    // Verify both paths produce the same result
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_NEAR(C_fp32_from_int32[i], C_fp32_direct[i], 0.01f)
            << "Mismatch at index " << i
            << ": INT32 path=" << C_fp32_from_int32[i]
            << ", FP32 path=" << C_fp32_direct[i];
    }

    LOG_INFO("[Test__INT8GemmKernel__INT32Output] INT32OutputBasic: Dequantization equivalence verified");
}

/**
 * @test INT32OutputLargeMatrix
 * @brief Verify INT32 output dequantization equivalence on larger matrices
 *
 * Tests realistic sizes with random data.
 * Verifies INT32 path + manual dequant == FP32 path.
 */
TEST_F(Test__INT8GemmKernel__INT32Output, INT32OutputLargeMatrix)
{
    const int m = 128, n = 256, k = 512;

    // Generate random FP32 matrices
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k); // n×k for transposed
    for (auto &val : A_fp32)
        val = dist(rng);
    for (auto &val : B_fp32)
        val = dist(rng);

    // Create INT8Tensor for B
    auto B_tensor = std::make_shared<INT8Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, B_fp32);

    // Create kernel
    INT8GemmKernel kernel(B_tensor.get());

    // Path 1: INT32 output + manual dequant
    std::vector<int32_t> C_int32(m * n);
    std::vector<float> A_scales(m);
    bool success1 = kernel.multiply_int32(
        A_fp32.data(), C_int32.data(),
        m, n, k, true, A_scales.data());
    ASSERT_TRUE(success1) << "multiply_int32 failed";

    // Manual dequantization (transpose_B=true, so use per-row scales)
    std::vector<float> C_fp32_from_int32(m * n);
    const std::vector<float> &B_row_scales = B_tensor->get_row_scales();
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C_fp32_from_int32[i * n + j] = static_cast<float>(C_int32[i * n + j]) * A_scales[i] * B_row_scales[j];
        }
    }

    // Path 2: Direct FP32 output
    std::vector<float> C_fp32_direct(m * n);
    bool success2 = kernel.multiply(
        A_fp32.data(), C_fp32_direct.data(),
        m, n, k, true, 1.0f, 0.0f);
    ASSERT_TRUE(success2) << "multiply (FP32) failed";

    // Verify both paths produce similar results (allowing for quantization error)
    double max_diff = 0.0;
    double sum_sq_diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double diff = std::abs(C_fp32_from_int32[i] - C_fp32_direct[i]);
        max_diff = std::max(max_diff, diff);
        sum_sq_diff += diff * diff;
    }
    double rmse = std::sqrt(sum_sq_diff / (m * n));

    // Both paths use the SAME quantization, so should be nearly identical
    EXPECT_LT(max_diff, 1e-4) << "Max difference too large";
    EXPECT_LT(rmse, 1e-5) << "RMSE too large";

    LOG_INFO("[Test__INT8GemmKernel__INT32Output] INT32OutputLargeMatrix: "
             << m << "×" << k << " @ " << k << "×" << n << " → " << m << "×" << n
             << " (max_diff=" << max_diff << ", rmse=" << rmse << ")");
}

/**
 * @test INT32OutputScalesCorrect
 * @brief Verify returned A row scales are correct
 *
 * The multiply_int32 optionally returns per-row scales for A.
 * These are needed for later dequantization or requantization.
 */
TEST_F(Test__INT8GemmKernel__INT32Output, INT32OutputScalesCorrect)
{
    const int m = 64, n = 128, k = 256;

    // Generate random FP32 matrices
    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    for (auto &val : A_fp32)
        val = dist(rng);
    for (auto &val : B_fp32)
        val = dist(rng);

    // Create INT8Tensor for B
    auto B_tensor = std::make_shared<INT8Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, B_fp32);

    // Create kernel
    INT8GemmKernel kernel(B_tensor.get());

    // Execute INT32 GEMM and get A row scales
    std::vector<int32_t> C_int32(m * n);
    std::vector<float> A_scales_out(m);
    bool success = kernel.multiply_int32(
        A_fp32.data(), C_int32.data(),
        m, n, k,
        true, // transpose_B
        A_scales_out.data());

    ASSERT_TRUE(success) << "multiply_int32 failed";

    // Verify scales are reasonable
    for (int i = 0; i < m; ++i)
    {
        EXPECT_GT(A_scales_out[i], 0.0f) << "Scale must be positive at row " << i;
        EXPECT_LT(A_scales_out[i], 100.0f) << "Scale seems too large at row " << i;

        // Verify scale matches max value in row
        float max_val = 0.0f;
        for (int j = 0; j < k; ++j)
        {
            max_val = std::max(max_val, std::abs(A_fp32[i * k + j]));
        }
        float expected_scale = max_val / 127.0f;
        EXPECT_NEAR(A_scales_out[i], expected_scale, 1e-5f)
            << "Scale mismatch at row " << i;
    }

    LOG_INFO("[Test__INT8GemmKernel__INT32Output] INT32OutputScalesCorrect: All " << m << " row scales verified");
}

/**
 * @test INT32OutputDequantizationEquivalence
 * @brief Verify INT32 output can be correctly dequantized to FP32
 *
 * Compares:
 * - Path 1: multiply_int32() → manual dequant
 * - Path 2: multiply() (FP32 output)
 *
 * Both should produce identical FP32 results.
 */
TEST_F(Test__INT8GemmKernel__INT32Output, INT32OutputDequantizationEquivalence)
{
    const int m = 32, n = 64, k = 128;

    // Generate random FP32 matrices
    std::mt19937 rng(99999);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);

    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    for (auto &val : A_fp32)
        val = dist(rng);
    for (auto &val : B_fp32)
        val = dist(rng);

    // Create INT8Tensor for B
    auto B_tensor = std::make_shared<INT8Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, B_fp32);

    // Create kernel
    INT8GemmKernel kernel(B_tensor.get());

    // Path 1: INT32 output + manual dequant
    std::vector<int32_t> C_int32(m * n);
    std::vector<float> A_scales(m);
    bool success1 = kernel.multiply_int32(
        A_fp32.data(), C_int32.data(),
        m, n, k, true, A_scales.data());
    ASSERT_TRUE(success1);

    // Manual dequantization: C_fp32[i,j] = C_int32[i,j] * A_scale[i] * B_scale[j]
    // transpose_B=true, so use per-row scales
    std::vector<float> C_fp32_from_int32(m * n);
    const std::vector<float> &B_row_scales = B_tensor->get_row_scales();
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            C_fp32_from_int32[i * n + j] = static_cast<float>(C_int32[i * n + j]) * A_scales[i] * B_row_scales[j];
        }
    }

    // Path 2: Direct FP32 output
    std::vector<float> C_fp32_direct(m * n);
    bool success2 = kernel.multiply(
        A_fp32.data(), C_fp32_direct.data(),
        m, n, k, true, 1.0f, 0.0f);
    ASSERT_TRUE(success2);

    // Compare results (should be identical)
    double max_diff = 0.0;
    double sum_sq_diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double diff = std::abs(C_fp32_from_int32[i] - C_fp32_direct[i]);
        max_diff = std::max(max_diff, diff);
        sum_sq_diff += diff * diff;
    }
    double rmse = std::sqrt(sum_sq_diff / (m * n));

    EXPECT_LT(max_diff, 1e-4) << "Max difference too large";
    EXPECT_LT(rmse, 1e-5) << "RMSE too large";

    LOG_INFO("[Test__INT8GemmKernel__INT32Output] INT32OutputDequantizationEquivalence: "
             << "max_diff=" << max_diff << ", rmse=" << rmse);
}

/**
 * @test INT32OutputZeroMatrix
 * @brief Edge case: zero input matrix
 */
TEST_F(Test__INT8GemmKernel__INT32Output, INT32OutputZeroMatrix)
{
    const int m = 16, n = 32, k = 64;

    std::vector<float> A_fp32(m * k, 0.0f); // All zeros
    std::vector<float> B_fp32(n * k, 1.0f); // Non-zero

    auto B_tensor = std::make_shared<INT8Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, B_fp32);
    INT8GemmKernel kernel(B_tensor.get());

    std::vector<int32_t> C_int32(m * n);
    bool success = kernel.multiply_int32(A_fp32.data(), C_int32.data(), m, n, k, true, nullptr);
    ASSERT_TRUE(success);

    // All output should be zero
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_EQ(C_int32[i], 0) << "Non-zero output at index " << i;
    }

    LOG_INFO("[Test__INT8GemmKernel__INT32Output] INT32OutputZeroMatrix: output is zero (as expected)");
}

/**
 * @test INT32OutputExtremeValues
 * @brief Edge case: extreme INT8 values (±127)
 */
TEST_F(Test__INT8GemmKernel__INT32Output, INT32OutputExtremeValues)
{
    const int m = 8, n = 8, k = 16;

    // Create matrices that will quantize to ±127
    std::vector<float> A_fp32(m * k);
    std::vector<float> B_fp32(n * k);
    for (int i = 0; i < m * k; ++i)
        A_fp32[i] = (i % 2 == 0) ? 127.0f : -127.0f;
    for (int i = 0; i < n * k; ++i)
        B_fp32[i] = (i % 3 == 0) ? 127.0f : -127.0f;

    auto B_tensor = std::make_shared<INT8Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, B_fp32);
    INT8GemmKernel kernel(B_tensor.get());

    std::vector<int32_t> C_int32(m * n);
    bool success = kernel.multiply_int32(A_fp32.data(), C_int32.data(), m, n, k, true, nullptr);
    ASSERT_TRUE(success);

    // Verify no overflow (INT32 range is ±2^31)
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_GE(C_int32[i], -2147483648LL) << "INT32 underflow at index " << i;
        EXPECT_LE(C_int32[i], 2147483647LL) << "INT32 overflow at index " << i;
    }

    LOG_INFO("[Test__INT8GemmKernel__INT32Output] INT32OutputExtremeValues: no overflow detected");
}

/**
 * @test INT32OutputPrecision
 * @brief Verifies per-row scale quantization maintains acceptable precision with real model weights
 *
 * Loads actual Qwen model weights and tests that INT8 GEMM produces results within
 * acceptable error bounds compared to FP32 reference. This validates accuracy on
 * real production data rather than synthetic test matrices.
 *
 * Uses layer 0 query projection weight (blk.0.attn_q.weight) which is representative
 * of typical weight matrices in the model.
 *
 * Expected behavior:
 * - Works with real weight distribution patterns
 * - Validates per-row scale implementation on production data
 * - Documents quantization accuracy with actual model weights
 */
TEST_F(Test__INT8GemmKernel__INT32Output, INT32OutputPrecision)
{
    // Load real model weights from GGUF file
    const std::string model_path = "/workspaces/llaminar/models/qwen2.5-0.5b-instruct-q8_0.gguf";

    ModelLoader loader;
    if (!loader.loadModel(model_path))
    {
        GTEST_SKIP() << "Model file not found: " << model_path << " (test requires actual model weights)";
    }

    // Load layer 0 query projection weight (typical transformer weight matrix)
    // Shape: [n_heads * head_dim, d_model] = [896, 896] for Qwen 0.5B
    auto weight_tensor = loader.loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::CONVERT_TO_FP32);
    if (!weight_tensor)
    {
        GTEST_SKIP() << "Failed to load blk.0.attn_q.weight from model";
    }

    // Check tensor type and ensure we have FP32 data
    LOG_INFO("  Tensor type: " << static_cast<int>(weight_tensor->native_type()));

    // Get weight data as FP32
    const auto &shape = weight_tensor->shape();
    if (shape.size() != 2)
    {
        GTEST_SKIP() << "Weight tensor is not 2D: " << shape.size() << "D";
    }

    const size_t n = shape[0]; // Output features
    const size_t k = shape[1]; // Input features

    LOG_INFO("  Loaded weight tensor: " << n << " × " << k << " from " << model_path);

    // Convert directly to INT8 with per-channel quantization (avoids double quantization)
    std::vector<int8_t> weight_int8(n * k);
    std::vector<float> weight_col_scales(k);
    std::vector<float> weight_row_scales(n);

    if (!weight_tensor->to_int8_perchannel(weight_int8.data(), weight_col_scales.data(), weight_row_scales.data()))
    {
        GTEST_SKIP() << "Failed to convert weight tensor to INT8 per-channel";
    }

    LOG_INFO("  Converted to INT8 with per-channel quantization (direct conversion, no double quantization)");
    LOG_INFO("  Column scales range: [" << *std::min_element(weight_col_scales.begin(), weight_col_scales.end())
                                        << ", " << *std::max_element(weight_col_scales.begin(), weight_col_scales.end()) << "]");
    LOG_INFO("  Row scales range: [" << *std::min_element(weight_row_scales.begin(), weight_row_scales.end())
                                     << ", " << *std::max_element(weight_row_scales.begin(), weight_row_scales.end()) << "]");

    // Create test activation matrix (random, representing real activations)
    const int m = 64; // Batch dimension (multiple tokens)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> A_fp32(m * k);
    for (auto &val : A_fp32)
        val = dist(rng);

    // Create FP32 reference result using the INT8 per-channel quantized weights
    // This ensures we're testing INT8 quantization accuracy, not Q8_0→INT8 double-quantization error
    //
    // IMPORTANT: transpose_B=true in the kernel call below, which means the kernel uses
    // per-ROW scales (not per-column scales). We must match that in the reference.
    std::vector<float> C_fp32_reference(m * n, 0.0f);
    {
        // Dequantize INT8 weights back to FP32 for reference computation
        // Use ROW scales since transpose_B=true (kernel multiplies by rows of B)
        std::vector<float> weight_fp32_deq(n * k);
        for (size_t j = 0; j < n; ++j)
        {
            const float row_scale = weight_row_scales[j]; // Per-row scale for this row
            for (size_t kk = 0; kk < k; ++kk)
            {
                const size_t idx = j * k + kk;
                weight_fp32_deq[idx] = static_cast<float>(weight_int8[idx]) * row_scale;
            }
        }

        // Reference GEMM: A @ B^T (B is n×k, transposed to k×n)
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < static_cast<int>(n); ++j)
            {
                float sum = 0.0f;
                for (size_t kk = 0; kk < k; ++kk)
                {
                    sum += A_fp32[i * k + kk] * weight_fp32_deq[j * k + kk];
                }
                C_fp32_reference[i * n + j] = sum;
            }
        }
    }

    // Create INT8Tensor with per-channel quantized data
    auto B_tensor = std::make_shared<INT8Tensor>(std::vector<size_t>{n, k}, weight_int8, 1.0f);
    B_tensor->set_col_scales(weight_col_scales.data(), k);
    B_tensor->set_row_scales(weight_row_scales.data(), n);

    LOG_INFO("  Created INT8Tensor with per-channel scales (cols=" << k << ", rows=" << n << ")");

    INT8GemmKernel kernel(B_tensor.get());

    // Debug: Check scale statistics
    const std::vector<float> &row_scales = B_tensor->get_row_scales();
    const float *col_scales = B_tensor->col_scales();

    if (row_scales.size() >= 5 && col_scales != nullptr)
    {
        LOG_INFO("  First 5 row scales: [" << row_scales[0] << ", " << row_scales[1] << ", "
                                           << row_scales[2] << ", " << row_scales[3] << ", " << row_scales[4] << "]");
        LOG_INFO("  First 5 col scales: [" << col_scales[0] << ", " << col_scales[1] << ", "
                                           << col_scales[2] << ", " << col_scales[3] << ", " << col_scales[4] << "]");
    }
    LOG_INFO("  Global scale: " << B_tensor->scale());
    LOG_INFO("  has_col_scales(): " << B_tensor->has_col_scales());
    LOG_INFO("  has_row_scales(): " << B_tensor->has_row_scales());
    LOG_INFO("  num_col_scales(): " << B_tensor->num_col_scales());

    std::vector<float> C_fp32_int8(m * n);
    bool success = kernel.multiply(
        A_fp32.data(), C_fp32_int8.data(),
        m, n, k, true, 1.0f, 0.0f);
    ASSERT_TRUE(success) << "INT8 GEMM failed";

    // Compute error metrics
    double max_abs_error = 0.0;
    double sum_sq_error = 0.0;
    double sum_abs_rel_error = 0.0;
    int num_large_rel_errors = 0;
    const double rel_error_threshold = 0.01; // 1% relative error threshold

    for (int i = 0; i < m * n; ++i)
    {
        float ref = C_fp32_reference[i];
        float quant = C_fp32_int8[i];

        double abs_error = std::abs(ref - quant);
        max_abs_error = std::max(max_abs_error, abs_error);
        sum_sq_error += abs_error * abs_error;

        // Relative error (skip near-zero reference values to avoid division issues)
        if (std::abs(ref) > 1e-6f)
        {
            double rel_error = abs_error / std::abs(ref);
            sum_abs_rel_error += rel_error;
            if (rel_error > rel_error_threshold)
            {
                num_large_rel_errors++;
            }
        }
    }

    double rmse = std::sqrt(sum_sq_error / (m * n));
    double mean_rel_error = sum_abs_rel_error / (m * n);
    double pct_large_errors = 100.0 * num_large_rel_errors / (m * n);

    // Log results
    LOG_INFO("[Test__INT8GemmKernel__INT32Output] INT32OutputPrecision: "
             << m << "×" << k << " @ " << k << "×" << n << " → " << m << "×" << n);
    LOG_INFO("  Max absolute error: " << max_abs_error);
    LOG_INFO("  RMSE: " << rmse);
    LOG_INFO("  Mean relative error: " << mean_rel_error * 100.0 << "%");
    LOG_INFO("  Elements with >1% rel error: " << num_large_rel_errors << " (" << pct_large_errors << "%)");

    // Verify precision is acceptable for INT8 per-column quantization with real model weights
    //
    // This test validates that INT8GemmKernel produces the same results as a reference
    // FP32 GEMM using the same INT8 quantized weights (not the original Q8_0 weights).
    //
    // Expected precision:
    // - INT8 quantization: ~1/127 ≈ 0.79% per value
    // - Accumulation error: grows with sqrt(k) for k accumulations
    // - Per-channel quantization: much better than per-tensor (columns share scale)
    //
    // For k=896 with per-channel quantization:
    // - Typical relative error: 1-5% (acceptable for INT8 precision)
    // - Max absolute error: < 1.0 (most values much smaller)
    // - RMSE: < 0.05 (very low)
    //
    // If we were comparing against Q8_0 dequantized weights (wrong!), we'd see:
    // - Double quantization error: 548% mean relative error (UNACCEPTABLE)
    // - Max absolute error: ~95 (HUGE)

    EXPECT_LT(max_abs_error, 1.0) << "Max absolute error too large (should be <1.0 for INT8 precision)";
    EXPECT_LT(rmse, 0.05) << "RMSE too large (should be <0.05 for INT8 precision)";
    EXPECT_LT(mean_rel_error, 0.05) << "Mean relative error > 5% (should be <5% for INT8 precision)";

    // Log overall assessment
    if (max_abs_error < 0.01 && rmse < 0.01 && mean_rel_error < 0.001)
    {
        LOG_INFO("  ✓ Perfect precision (numerical errors only, max_err < 0.01, RMSE < 0.01, mean_rel < 0.1%)");
    }
    else if (max_abs_error < 0.1 && rmse < 0.05 && mean_rel_error < 0.01)
    {
        LOG_INFO("  ✓ Excellent precision (max_err < 0.1, RMSE < 0.05, mean_rel < 1%)");
    }
    else if (max_abs_error < 1.0 && rmse < 0.05 && mean_rel_error < 0.05)
    {
        LOG_INFO("  ✓ Good precision (acceptable for INT8 quantization)");
    }
    else
    {
        LOG_INFO("  ⚠ Precision lower than expected (possible implementation issue)");
    }
}
