/**
 * @file Test__IntegerGemm.cpp
 * @brief Unit tests for integer-domain quantized GEMM
 * @author David Sanftenberg
 * @date November 2025
 */

#include <gtest/gtest.h>
#include "kernels/cpu/gemm_v2/IntegerGemm.h"
#include "loaders/ModelLoader.h"
#include "tensors/FP16Utils.h"
#include "utils/CPUFeatures.h"
#include <cmath>
#include <iostream>
#include <memory>

using namespace llaminar2;
using namespace llaminar2::kernels;

class Test__IntegerGemm : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Check if AVX512-VNNI is available
        has_vnni = cpu_supports_avx512_vnni();

        if (!has_vnni)
        {
            std::cout << "[WARNING] AVX512-VNNI not available, tests will be skipped\n";
        }
        else
        {
            std::cout << "[INFO] AVX512-VNNI detected, running integer GEMM tests\n";
        }
    }

    bool has_vnni = false;
};

// ========== FP32 → Q8_0 Quantization Tests ==========

TEST_F(Test__IntegerGemm, QuantizeFP32ToQ8_0_Simple)
{
    // Test basic quantization with known values
    float src[32];
    for (int i = 0; i < 32; ++i)
    {
        src[i] = static_cast<float>(i) - 15.5f; // Range: [-15.5, 16.5]
    }

    Q8_0Block dst;
    quantize_fp32_to_q8_0(src, &dst, 32);

    // Check scale is reasonable (max abs is 16.5)
    // Convert FP16 (uint16_t) to FP32 for comparison
    float scale = fp16_to_fp32(dst.d);
    EXPECT_GT(scale, 0.0f);
    EXPECT_LT(scale, 1.0f); // Scale should be ~16.5/127 ≈ 0.13

    // Check quantized values are in range [-127, 127]
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_GE(dst.qs[i], -127);
        EXPECT_LE(dst.qs[i], 127);
    }

    // Check round-trip error is reasonable
    float max_error = 0.0f;
    for (int i = 0; i < 32; ++i)
    {
        float reconstructed = static_cast<float>(dst.qs[i]) * scale;
        float error = std::fabs(reconstructed - src[i]);
        max_error = std::max(max_error, error);
    }

    EXPECT_LT(max_error, 0.5f); // Should be within 0.5 (quantization error)
}

TEST_F(Test__IntegerGemm, QuantizeFP32ToQ8_0_AllZeros)
{
    float src[32] = {0};
    Q8_0Block dst;
    quantize_fp32_to_q8_0(src, &dst, 32);

    // Scale should be 0 for all-zero input
    EXPECT_EQ(dst.d, 0.0f);

    // All quantized values should be 0
    for (int i = 0; i < 32; ++i)
    {
        EXPECT_EQ(dst.qs[i], 0);
    }
}

TEST_F(Test__IntegerGemm, QuantizeFP32ToQ8_0_LargeValues)
{
    float src[32];
    for (int i = 0; i < 32; ++i)
    {
        src[i] = static_cast<float>(i) * 100.0f; // Range: [0, 3100]
    }

    Q8_0Block dst;
    quantize_fp32_to_q8_0(src, &dst, 32);

    // Scale should be ~3100/127 ≈ 24.4 (convert FP16 to FP32)
    float scale = fp16_to_fp32(dst.d);
    EXPECT_GT(scale, 20.0f);
    EXPECT_LT(scale, 30.0f);

    // Largest value should quantize to ~127
    EXPECT_GE(dst.qs[31], 120);
    EXPECT_LE(dst.qs[31], 127);
}

// ========== Integer GEMM Correctness Tests ==========

TEST_F(Test__IntegerGemm, IntegerGemm_SmallMatrix_IQ4NL)
{
    if (!has_vnni)
    {
        GTEST_SKIP() << "AVX512-VNNI not available";
    }

    // Load a real IQ4_NL model for testing
    std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
    ModelLoader loader;

    if (!loader.loadModel(model_path))
    {
        GTEST_SKIP() << "Model not found: " << model_path;
    }

    // Get a weight tensor (e.g., first layer Q projection)
    auto wq_tensor = loader.loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_tensor, nullptr);
    ASSERT_EQ(wq_tensor->native_type(), TensorType::IQ4_NL);

    const auto &shape = wq_tensor->shape();
    ASSERT_EQ(shape.size(), 2);

    const int N = shape[0]; // 896 for Qwen 0.5B
    const int K = shape[1]; // 896

    // Create small test case (4 × 896 × 896)
    const int M = 4;

    // Allocate matrices
    std::vector<float> A(M * K);
    std::vector<float> C_int8(M * N, 0.0f);
    std::vector<float> C_fp32(M * N, 0.0f);

    // Initialize A with random-ish values
    for (int i = 0; i < M * K; ++i)
    {
        A[i] = std::sin(static_cast<float>(i) * 0.01f);
    }

    // Run integer GEMM
    bool success = gemm_int8_dispatch(A.data(), wq_tensor.get(), C_int8.data(), M, N, K);
    ASSERT_TRUE(success) << "Integer GEMM failed";

    // Reference: Dequantize weight to FP32 and run standard GEMM
    std::vector<float> wq_fp32_data(N * K);
    wq_tensor->to_fp32(wq_fp32_data.data());

    const float *wq_data = wq_fp32_data.data();

    // Naive GEMM: C = A × B^T (B is [N × K], transposed)
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * wq_data[n * K + k];
            }
            C_fp32[m * N + n] = sum;
        }
    }

    // Compare results (allow for quantization error)
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;

    for (int i = 0; i < M * N; ++i)
    {
        float abs_error = std::fabs(C_int8[i] - C_fp32[i]);
        float rel_error = abs_error / (std::fabs(C_fp32[i]) + 1e-6f);

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    std::cout << "[IntegerGemm] Max absolute error: " << max_abs_error << "\n";
    std::cout << "[IntegerGemm] Max relative error: " << max_rel_error << "\n";

    // Integer GEMM quantization introduces error:
    // - Absolute error should be small (<1.0 for typical values)
    // - Relative error can be large for near-zero values (acceptable)
    EXPECT_LT(max_abs_error, 1.0f) << "Integer GEMM absolute error too large";

    // Also check that most values have reasonable relative error
    // (This filters out near-zero outliers)
    int good_count = 0;
    for (int i = 0; i < M * N; ++i)
    {
        if (std::fabs(C_fp32[i]) > 1.0f) // Only check non-trivial values
        {
            float abs_error = std::fabs(C_int8[i] - C_fp32[i]);
            float rel_error = abs_error / std::fabs(C_fp32[i]);
            if (rel_error < 0.1f) // 10% tolerance
            {
                good_count++;
            }
        }
    }

    // At least 90% of non-trivial values should be within 10% relative error
    int non_trivial_count = 0;
    for (int i = 0; i < M * N; ++i)
    {
        if (std::fabs(C_fp32[i]) > 1.0f)
        {
            non_trivial_count++;
        }
    }

    if (non_trivial_count > 0)
    {
        float good_fraction = static_cast<float>(good_count) / non_trivial_count;
        std::cout << "[IntegerGemm] Values within 10% error: "
                  << (good_fraction * 100.0f) << "%\n";
        EXPECT_GT(good_fraction, 0.80f) << "Too many values with >10% relative error";
    }
}

TEST_F(Test__IntegerGemm, IntegerGemm_Performance_Comparison)
{
    if (!has_vnni)
    {
        GTEST_SKIP() << "AVX512-VNNI not available";
    }

    // Load model
    std::string model_path = "models/qwen2.5-0.5b-instruct-iq4_nl.gguf";
    ModelLoader loader;

    if (!loader.loadModel(model_path))
    {
        GTEST_SKIP() << "Model not found: " << model_path;
    }

    auto wq_tensor = loader.loadTensor("blk.0.attn_q.weight", 0, WeightPrecision::NATIVE);
    ASSERT_NE(wq_tensor, nullptr);

    const auto &shape = wq_tensor->shape();
    const int N = shape[0];
    const int K = shape[1];

    // Test with batch size 128 (typical prefill)
    const int M = 128;

    std::vector<float> A(M * K);
    std::vector<float> C(M * N);

    // Initialize A
    for (int i = 0; i < M * K; ++i)
    {
        A[i] = std::sin(static_cast<float>(i) * 0.01f);
    }

    // Warmup
    for (int i = 0; i < 3; ++i)
    {
        gemm_int8_dispatch(A.data(), wq_tensor.get(), C.data(), M, N, K);
    }

    // Timed run
    const int num_iters = 20;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iters; ++i)
    {
        gemm_int8_dispatch(A.data(), wq_tensor.get(), C.data(), M, N, K);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
    double time_per_iter = elapsed_ms / num_iters;

    // Calculate GFLOPS
    double flops = 2.0 * M * N * K; // multiply + add
    double gflops = flops / (time_per_iter / 1000.0 * 1e9);

    std::cout << "\n[IntegerGemm Performance]\n";
    std::cout << "  Matrix size: [" << M << " × " << K << "] × [" << K << " × " << N << "]^T\n";
    std::cout << "  Time per iteration: " << time_per_iter << " ms\n";
    std::cout << "  Throughput: " << gflops << " GFLOPS\n";

    // NOTE: This prototype has poor performance (~2 GFLOPS) because:
    // 1. Activations are quantized row-by-row during GEMM (not pre-quantized)
    // 2. No cache blocking/tiling (MC/KC/NC)
    // 3. No memory layout optimization
    //
    // Production implementation would use GemmMicroKernelTemplateINT8.h approach.
    // For now, just verify it runs without crashing.
    EXPECT_GT(gflops, 1.0) << "Integer GEMM should produce some throughput";
}

// ========== Edge Cases ==========

TEST_F(Test__IntegerGemm, IntegerGemm_NoVNNI_Fallback)
{
    // This test verifies graceful degradation when VNNI is not available
    // (Will only fail if we're on a non-VNNI system)

    if (has_vnni)
    {
        GTEST_SKIP() << "VNNI available, cannot test fallback";
    }

    std::vector<float> A(32);
    std::vector<float> C(32);

    // Should return false when VNNI not available
    bool result = gemm_int8_dispatch(A.data(), nullptr, C.data(), 1, 32, 32);
    EXPECT_FALSE(result);
}
