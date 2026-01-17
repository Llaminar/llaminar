/**
 * @file Test__ROCmFlashAttentionParity.cpp
 * @brief Parity tests for ROCm Flash Attention kernel vs CPU reference
 *
 * **Purpose**: Validate that ROCm Flash Attention kernels produce numerically
 * equivalent results to CPU attention kernels with high cosine similarity.
 *
 * **Tests**:
 * - Flash Attention 2 (prefill) vs CPU attention
 * - Flash Decoding (decode) vs CPU attention
 * - Various head dimensions (64, 128)
 * - GQA configurations (n_heads != n_kv_heads)
 *
 * **Pass Criteria**:
 * - Cosine similarity >= 0.99 (attention is numerically sensitive)
 * - No NaN/Inf in outputs
 * - Relative error < 5% for FP32
 *
 * Target Hardware: AMD MI50 (gfx906 / Vega 20)
 *
 * @author Llaminar Team
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "tensors/Tensors.h"
#include "execution/RuntimeConfig.h"
#include "utils/MPIContext.h"
#include "utils/Logger.h"

#ifdef HAVE_ROCM
#include <hip/hip_runtime.h>
#include "kernels/rocm/attention/ROCmFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUAttentionKernelT.h"
#endif

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <limits>

using namespace llaminar2;

namespace
{

    // ============================================================================
    // ROCm Availability Check
    // ============================================================================

    bool hasROCm()
    {
#ifdef HAVE_ROCM
        int count = 0;
        hipError_t err = hipGetDeviceCount(&count);
        return (err == hipSuccess && count > 0);
#else
        return false;
#endif
    }

    /**
     * @brief Check if Flash Attention 2 will fit in LDS for the given head_dim
     *
     * The FA2 kernel uses:
     * - Q_tile:    [64, head_dim + 8] in FP16
     * - K/V double buffer: 2 * 2 * [64, head_dim + 8] in FP16
     * - Scores:    [64, 68] in FP32
     *
     * MI50 has 65KB LDS per workgroup.
     *
     * @param head_dim The head dimension to check
     * @return true if the kernel will fit, false otherwise
     */
    bool flashAttn2FitsInLDS(int head_dim)
    {
#ifdef HAVE_ROCM
        constexpr int tile_q = 64;
        constexpr int tile_kv = 64;
        constexpr int lds_pad_fp16 = 8;
        constexpr int lds_pad_fp32 = 4;
        constexpr int num_stages = 2;
        constexpr int max_lds = 65536; // MI50 LDS size

        int q_stride = head_dim + lds_pad_fp16;
        int kv_stride = head_dim + lds_pad_fp16;
        int scores_stride = tile_kv + lds_pad_fp32;

        int q_tile_size = tile_q * q_stride * sizeof(uint16_t);    // FP16
        int kv_tile_size = tile_kv * kv_stride * sizeof(uint16_t); // FP16
        int scores_size = tile_q * scores_stride * sizeof(float);

        // Total: Q + 2*(K+V) for double buffer + scores
        int total_lds = q_tile_size + num_stages * 2 * kv_tile_size + scores_size;

        return total_lds <= max_lds;
#else
        return false;
#endif
    }

    // ============================================================================
    // Similarity Utilities
    // ============================================================================

    double cosineSimilarity(const float *a, const float *b, size_t count)
    {
        double dot = 0.0, norm_a = 0.0, norm_b = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            dot += static_cast<double>(a[i]) * b[i];
            norm_a += static_cast<double>(a[i]) * a[i];
            norm_b += static_cast<double>(b[i]) * b[i];
        }
        double denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        if (denom < 1e-12)
            return 0.0;
        return dot / denom;
    }

    double relativeL2Error(const float *actual, const float *expected, size_t count)
    {
        double diff_norm = 0.0, expected_norm = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double diff = actual[i] - expected[i];
            diff_norm += diff * diff;
            expected_norm += static_cast<double>(expected[i]) * expected[i];
        }
        if (expected_norm < 1e-12)
            return diff_norm > 1e-12 ? 1e9 : 0.0;
        return std::sqrt(diff_norm / expected_norm);
    }

    double maxAbsError(const float *actual, const float *expected, size_t count)
    {
        double max_err = 0.0;
        for (size_t i = 0; i < count; ++i)
        {
            double err = std::abs(static_cast<double>(actual[i]) - expected[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    bool hasNaNOrInf(const float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            if (std::isnan(data[i]) || std::isinf(data[i]))
            {
                return true;
            }
        }
        return false;
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__ROCmFlashAttentionParity : public ::testing::Test
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-0.5f, 0.5f};
    MPIContext mpi_ctx_{0, 1, MPI_COMM_WORLD};

    std::vector<float> randomFP32(size_t count)
    {
        std::vector<float> data(count);
        for (auto &val : data)
        {
            val = dist_(rng_);
        }
        return data;
    }

    void printComparisonStats(
        const char *test_name,
        double cosine, double l2_error, double max_error,
        size_t count)
    {
        std::cout << "  " << test_name << ": "
                  << "cosine=" << std::fixed << std::setprecision(6) << cosine
                  << ", L2_error=" << std::scientific << std::setprecision(3) << l2_error
                  << ", max_error=" << max_error
                  << ", count=" << count
                  << std::endl;
    }
};

#ifdef HAVE_ROCM

// ============================================================================
// Flash Attention 2 (Prefill) Parity Tests
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_Small)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Small test case for basic correctness
    constexpr int seq_len = 8;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4; // MHA (not GQA)
    constexpr int head_dim = 32;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    // Create test data
    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true,    // causal
        -1,      // window_size
        nullptr, // workspace_scores
        nullptr, // workspace_buffer
        nullptr, // workspace_context
        nullptr, // workspace_mask
        false,   // use_bf16
        &mpi_ctx_,
        -1 // device_idx (CPU)
    );
    ASSERT_TRUE(cpu_success) << "CPU attention failed";

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    // Copy inputs to device
    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    // Execute ROCm kernel
    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true,    // causal
        -1,      // window_size
        nullptr, // workspace_scores
        nullptr, // workspace_buffer
        nullptr, // workspace_context
        nullptr, // workspace_mask
        false,   // use_bf16
        &mpi_ctx_,
        0 // device_idx
    );
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success) << "ROCm attention failed";

    // Copy output back
    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    // Cleanup
    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size)) << "ROCm output contains NaN/Inf";
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size)) << "CPU output contains NaN/Inf";

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Small", cosine, l2_error, max_error, out_size);

    // Attention has some numerical sensitivity, so we use looser thresholds
    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_Small] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_Medium_GQA)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Medium test case with GQA (typical Qwen2 configuration)
    constexpr int seq_len = 64;
    constexpr int n_heads = 14;   // Qwen2-0.5B
    constexpr int n_kv_heads = 2; // GQA with ratio 7
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Medium GQA", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_Medium_GQA] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_LargeHeadDim)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Test with larger head_dim (Llama-3 style)
    // Kernel now automatically selects smaller tiles (32x32) to fit in LDS
    constexpr int seq_len = 32;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8; // MHA
    constexpr int head_dim = 128;

    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Large HeadDim", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_LargeHeadDim] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_LargeHeadDim_LargerSeq)
{
    // Tests adaptive tile sizing with larger sequence lengths
    // This ensures the 32x32 tile path works correctly across more Q tiles
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 128; // More Q tiles with 32x32 tiling
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128; // Triggers adaptive tiling

    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 LargeHeadDim LargerSeq", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_LargeHeadDim_LargerSeq] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_LargeHeadDim_GQA)
{
    // Tests adaptive tile sizing with GQA (grouped query attention)
    // Llama-3 uses head_dim=128 with GQA
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    constexpr int seq_len = 64;
    constexpr int n_heads = 32;   // 32 query heads
    constexpr int n_kv_heads = 8; // 8 KV heads (GQA ratio = 4)
    constexpr int head_dim = 128; // Triggers adaptive tiling

    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 LargeHeadDim GQA", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_LargeHeadDim_GQA] PASSED");
}

// ============================================================================
// Flash Decoding (Decode) Parity Tests
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_SmallKVLen)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Decode: single query attending to small KV cache
    constexpr int kv_len = 16;
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1; // Decode = single query
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference - for decode with seq_len=1, use compute_decode
    // The CPU kernel expects Q[seq_len, n_heads, head_dim], K[kv_len, n_kv_heads, head_dim]
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false); // Non-causal for decode (attend to full KV cache)
    ASSERT_TRUE(cpu_success);

    // ROCm kernel (decode path)
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    // Use compute_decode for single-query path
    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // Non-causal
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success) << "ROCm decode attention failed";

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Small KVLen", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashDecode_FP32_SmallKVLen] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_LargeKVLen_GQA)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Decode with larger KV cache and GQA
    constexpr int kv_len = 512;
    constexpr int n_heads = 14;   // Qwen2
    constexpr int n_kv_heads = 2; // GQA
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false); // Non-causal
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // Non-causal
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Large KVLen GQA", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashDecode_FP32_LargeKVLen_GQA] PASSED");
}

// ============================================================================
// Non-Causal Attention (for KV cache scenarios)
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_NonCausal)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Non-causal attention (e.g., for cross-attention or encoder)
    constexpr int seq_len = 32;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // NON-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // ROCm kernel
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // NON-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Non-Causal", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashAttn2_FP32_NonCausal] PASSED");
}

// ============================================================================
// Additional Parity Tests (matching CUDA coverage)
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_FP32_Large)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Large test case (longer sequence)
    constexpr int seq_len = 256;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Large", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashAttn2_FP32_Large] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_VeryLong_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Very long KV cache - stress test for split-K with many splits
    constexpr int kv_len = 1024;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, true);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // ROCm decode
    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        0);   // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 VeryLong Parity (kv=1024, split-K)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";

    LOG_INFO("[FlashDecode_FP32_VeryLong_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_MHA_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Multi-head attention (not GQA) - n_heads == n_kv_heads
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8; // MHA
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, true);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        0);   // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 MHA Parity (n_heads=n_kv_heads)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashDecode_FP32_MHA_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_HeadDim128_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Llama-style head_dim=128
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, true);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        0);   // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 HeadDim128 Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashDecode_FP32_HeadDim128_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_FP32_NonCausal_Parity)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Non-causal decode (bidirectional attention)
    constexpr int kv_len = 128;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    constexpr int seq_len = 1;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, kv_len, n_heads, n_kv_heads, head_dim, false); // non-causal
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, kv_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        0);    // position_offset
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 NonCausal Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    LOG_INFO("[FlashDecode_FP32_NonCausal_Parity] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashDecode_BatchDecoding)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Test batch decoding: multiple independent sequences decoded in parallel
    // Each batch element has seq_len=1 (decode) with different KV cache lengths
    constexpr int batch_size = 4;
    constexpr int kv_len = 256; // Context length
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2; // GQA
    constexpr int head_dim = 64;

    // For batch decoding, we process each batch sequentially using compute_decode
    // Q: [1, n_heads, head_dim] per batch
    // K/V: [kv_len, n_kv_heads, head_dim] per batch
    const size_t q_per_batch = 1 * n_heads * head_dim;
    const size_t kv_per_batch = kv_len * n_kv_heads * head_dim;
    const size_t out_per_batch = 1 * n_heads * head_dim;

    std::vector<float> Q_data = randomFP32(batch_size * q_per_batch);
    std::vector<float> K_data = randomFP32(batch_size * kv_per_batch);
    std::vector<float> V_data = randomFP32(batch_size * kv_per_batch);
    std::vector<float> rocm_output(batch_size * out_per_batch, 0.0f);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    // Allocate device memory for largest batch element
    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_per_batch * sizeof(float));
    hipMalloc(&d_K, kv_per_batch * sizeof(float));
    hipMalloc(&d_V, kv_per_batch * sizeof(float));
    hipMalloc(&d_output, out_per_batch * sizeof(float));

    bool all_success = true;

    // Process each batch element
    for (int b = 0; b < batch_size; b++)
    {
        hipMemcpy(d_Q, Q_data.data() + b * q_per_batch,
                  q_per_batch * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_K, K_data.data() + b * kv_per_batch,
                  kv_per_batch * sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_V, V_data.data() + b * kv_per_batch,
                  kv_per_batch * sizeof(float), hipMemcpyHostToDevice);
        hipMemset(d_output, 0, out_per_batch * sizeof(float));

        // Use compute_decode for single-token decode
        bool success = rocm_kernel.compute_decode(
            d_Q, d_K, d_V, d_output,
            1,      // seq_len = 1 for decode
            kv_len, // kv_len from cache
            n_heads, n_kv_heads, head_dim,
            true, // causal
            0);   // position_offset
        hipDeviceSynchronize();

        if (!success)
        {
            std::cerr << "Batch " << b << " decode failed" << std::endl;
            all_success = false;
            continue;
        }

        hipMemcpy(rocm_output.data() + b * out_per_batch, d_output,
                  out_per_batch * sizeof(float), hipMemcpyDeviceToHost);
    }

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_TRUE(all_success) << "All batch decode operations should succeed";
    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), batch_size * out_per_batch));

    // Verify each batch element has valid output
    bool all_batches_valid = true;
    for (int b = 0; b < batch_size; b++)
    {
        float batch_sum = 0.0f;
        float batch_max = -std::numeric_limits<float>::infinity();
        float batch_min = std::numeric_limits<float>::infinity();

        for (int h = 0; h < n_heads; h++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                float val = rocm_output[b * out_per_batch + h * head_dim + d];
                batch_sum += val;
                batch_max = std::max(batch_max, val);
                batch_min = std::min(batch_min, val);
            }
        }

        // Each batch should have non-trivial output
        bool batch_valid = (batch_sum != 0.0f) &&
                           (batch_max != batch_min) &&
                           std::isfinite(batch_sum);

        if (!batch_valid)
        {
            std::cerr << "Batch " << b << " invalid: sum=" << batch_sum
                      << ", min=" << batch_min << ", max=" << batch_max << std::endl;
            all_batches_valid = false;
        }
    }

    EXPECT_TRUE(all_batches_valid) << "All batch elements should have valid, non-trivial output";

    // Verify batches are independent (different inputs should give different outputs)
    // Compare batch 0 and batch 1 outputs
    float diff_sum = 0.0f;
    for (size_t i = 0; i < out_per_batch; i++)
    {
        float diff = rocm_output[i] - rocm_output[out_per_batch + i];
        diff_sum += diff * diff;
    }
    EXPECT_GT(diff_sum, 0.0f) << "Different batch inputs should produce different outputs";

    std::cout << "  FlashDecode BatchDecoding: batch_size=" << batch_size
              << ", kv_len=" << kv_len << ", n_heads=" << n_heads
              << ", n_kv_heads=" << n_kv_heads << " - PASSED" << std::endl;

    LOG_INFO("[FlashDecode_BatchDecoding] PASSED");
}

TEST_F(Test__ROCmFlashAttentionParity, FlashAttn2_CausalMasking)
{
    if (!hasROCm())
    {
        GTEST_SKIP() << "ROCm not available";
    }

    // Test that causal masking is correctly applied:
    // Position i should only attend to positions j where j <= i
    constexpr int seq_len = 64;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    // Use structured data to verify masking behavior
    // Q[i] = i+1 (so position 0 has Q=1, position 1 has Q=2, etc.)
    // K[j] = 1 for all j
    // V[j] = j+1 for all j
    // With causal masking, output[i] should be weighted average of V[0..i]
    std::vector<float> Q_data(q_size);
    std::vector<float> K_data(kv_size, 1.0f);
    std::vector<float> V_data(kv_size);

    for (int pos = 0; pos < seq_len; pos++)
    {
        for (int h = 0; h < n_heads; h++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                Q_data[pos * n_heads * head_dim + h * head_dim + d] = static_cast<float>(pos + 1);
            }
        }
    }
    for (int pos = 0; pos < seq_len; pos++)
    {
        for (int h = 0; h < n_kv_heads; h++)
        {
            for (int d = 0; d < head_dim; d++)
            {
                V_data[pos * n_kv_heads * head_dim + h * head_dim + d] = static_cast<float>(pos + 1);
            }
        }
    }

    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> rocm_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::rocm::ROCmFlashAttentionKernelT<ActivationPrecision::FP32> rocm_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    hipMalloc(&d_Q, q_size * sizeof(float));
    hipMalloc(&d_K, kv_size * sizeof(float));
    hipMalloc(&d_V, kv_size * sizeof(float));
    hipMalloc(&d_output, out_size * sizeof(float));

    hipMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_K, K_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_V, V_data.data(), kv_size * sizeof(float), hipMemcpyHostToDevice);
    hipMemset(d_output, 0, out_size * sizeof(float));

    bool rocm_success = rocm_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    hipDeviceSynchronize();

    ASSERT_TRUE(rocm_success);

    hipMemcpy(rocm_output.data(), d_output, out_size * sizeof(float), hipMemcpyDeviceToHost);

    hipFree(d_Q);
    hipFree(d_K);
    hipFree(d_V);
    hipFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(rocm_output.data(), out_size));

    double cosine = cosineSimilarity(rocm_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(rocm_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(rocm_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 CausalMasking", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    // Additional verification: first position should only see V[0]
    // and last position should see weighted average of all V
    float first_pos_val = rocm_output[0];                                 // First element of first position
    float last_pos_val = rocm_output[(seq_len - 1) * n_heads * head_dim]; // First element of last position

    // First position with uniform K should output V[0] = 1.0
    EXPECT_NEAR(first_pos_val, 1.0f, 0.01f) << "First position should only attend to position 0";

    // Last position should have higher value (attending to all positions)
    EXPECT_GT(last_pos_val, first_pos_val) << "Last position should attend to more context";

    LOG_INFO("[FlashAttn2_CausalMasking] PASSED");
}

#endif // HAVE_ROCM

// ============================================================================
// No ROCm Fallback
// ============================================================================

TEST_F(Test__ROCmFlashAttentionParity, NoROCm_SkipsGracefully)
{
    if (!hasROCm())
    {
        SUCCEED() << "ROCm not available - tests skip gracefully";
    }
    else
    {
        SUCCEED() << "ROCm available - tests will run";
    }
}
