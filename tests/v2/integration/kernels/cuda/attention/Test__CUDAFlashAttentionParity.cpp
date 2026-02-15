/**
 * @file Test__CUDAFlashAttentionParity.cpp
 * @brief Parity tests for CUDA Flash Attention kernel vs CPU reference
 *
 * **Purpose**: Validate that CUDA Flash Attention kernels produce numerically
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
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "execution/config/RuntimeConfig.h"
#include "utils/MPIContext.h"
#include "kernels/cpu/CPUKVCache.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUAttentionKernelT.h"
#include <cuda_runtime.h>
#endif

// Now include test utils
#include "../../../../utils/CUDATestUtils.h"
#include "../../../../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test;

namespace
{

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

    // ============================================================================
    // CPU Reference for Decode Attention (single query attending to KV cache)
    // ============================================================================

    /**
     * @brief CPU reference implementation for decode attention
     *
     * Computes attention for a single query token (seq_len=1) attending to
     * a KV cache of length kv_len. This is the ground truth for Flash Decoding.
     *
     * Layout:
     *   Q: [n_heads, head_dim]
     *   K: [kv_len, n_kv_heads, head_dim]
     *   V: [kv_len, n_kv_heads, head_dim]
     *   O: [n_heads, head_dim]
     *
     * @param causal If true, Q at position (kv_len-1+position_offset) attends to K[0..kv_len-1].
     *               For standard decode, position_offset=0 means Q can see all of KV cache.
     */
    void cpuDecodeAttentionReference(
        const float *Q, // [n_heads, head_dim]
        const float *K, // [kv_len, n_kv_heads, head_dim]
        const float *V, // [kv_len, n_kv_heads, head_dim]
        float *O,       // [n_heads, head_dim]
        int kv_len,
        int n_heads,
        int n_kv_heads,
        int head_dim,
        bool causal,
        int position_offset) // Q's logical position = kv_len - 1 + position_offset
    {
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const int gqa_ratio = n_heads / n_kv_heads;

        // For each query head
        for (int h = 0; h < n_heads; h++)
        {
            const int kv_h = h / gqa_ratio; // GQA: which KV head to use

            const float *Q_head = Q + h * head_dim;
            float *O_head = O + h * head_dim;

            // Q's logical position for causal masking
            // In decode, Q is the "next" token, so it's at position (kv_len - 1 + position_offset)
            // If position_offset = 0, Q is at position kv_len-1 and can attend to all KV[0..kv_len-1]
            const int q_pos = kv_len - 1 + position_offset;

            // Step 1: Compute attention scores and find max for numerical stability
            std::vector<float> scores(kv_len);
            float max_score = -std::numeric_limits<float>::infinity();

            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++)
            {
                // Causal mask: Q at q_pos can only attend to K at kv_pos if kv_pos <= q_pos
                bool masked = causal && (kv_pos > q_pos);

                if (masked)
                {
                    scores[kv_pos] = -std::numeric_limits<float>::infinity();
                }
                else
                {
                    const float *K_vec = K + kv_pos * n_kv_heads * head_dim + kv_h * head_dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                    {
                        dot += Q_head[d] * K_vec[d];
                    }
                    scores[kv_pos] = dot * scale;
                    max_score = std::max(max_score, scores[kv_pos]);
                }
            }

            // Step 2: Compute softmax(scores) and weighted sum of V
            float sum_exp = 0.0f;
            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++)
            {
                if (scores[kv_pos] > -std::numeric_limits<float>::infinity() / 2)
                {
                    scores[kv_pos] = std::exp(scores[kv_pos] - max_score);
                    sum_exp += scores[kv_pos];
                }
                else
                {
                    scores[kv_pos] = 0.0f;
                }
            }

            // Step 3: Compute output = sum(softmax_scores * V)
            for (int d = 0; d < head_dim; d++)
            {
                O_head[d] = 0.0f;
            }

            for (int kv_pos = 0; kv_pos < kv_len; kv_pos++)
            {
                if (scores[kv_pos] > 0.0f)
                {
                    float weight = scores[kv_pos] / sum_exp;
                    const float *V_vec = V + kv_pos * n_kv_heads * head_dim + kv_h * head_dim;
                    for (int d = 0; d < head_dim; d++)
                    {
                        O_head[d] += weight * V_vec[d];
                    }
                }
            }
        }
    }

} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAFlashAttentionParity : public CUDATestBase
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

#ifdef HAVE_CUDA

// ============================================================================
// Flash Attention 2 (Prefill) Parity Tests
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Small)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(out_size, 0.0f);

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

    // CUDA kernel
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    // Execute CUDA kernel
    bool cuda_success = cuda_kernel.compute(
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
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success) << "CUDA attention failed";

    // Copy output back
    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size)) << "CUDA output contains NaN/Inf";
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size)) << "CPU output contains NaN/Inf";

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Small", cosine, l2_error, max_error, out_size);

    // Attention has some numerical sensitivity, so we use looser thresholds
    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Medium)
{
    SKIP_IF_NO_CUDA();

    // Medium test case (typical prefill scenario)
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
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    // CUDA kernel
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Medium (GQA)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_FP32_Large)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 FP32 Large", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

// ============================================================================
// Flash Decoding Parity Tests (with CPU reference)
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_Short_Parity)
{
    SKIP_IF_NO_CUDA();

    // Short KV cache - tests the fallback path (may use prefill kernel)
    constexpr int kv_len = 32;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2; // GQA
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim; // [n_heads, head_dim]
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using CPUAttentionKernelT::compute_decode() - apples-to-apples comparison
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, // seq_len = 1 (single query token)
        kv_len,
        n_heads, n_kv_heads, head_dim,
        true); // causal
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // CUDA decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, // seq_len = 1 (single query token)
        kv_len,
        n_heads, n_kv_heads, head_dim,
        true, // causal
        0);   // position_offset
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success) << "CUDA Flash Decoding failed";

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    // Validate
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size)) << "CUDA output contains NaN/Inf";
    ASSERT_FALSE(hasNaNOrInf(cpu_output.data(), out_size)) << "CPU output contains NaN/Inf";

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Short Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low - decode kernel may be incorrect";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_Long_Parity)
{
    SKIP_IF_NO_CUDA();

    // Longer KV cache - this exercises the split-K Flash Decoding path
    constexpr int kv_len = 512;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using CPUAttentionKernelT::compute_decode() - apples-to-apples comparison
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true); // causal
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // CUDA decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true); // causal
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 Long Parity (split-K) vs CPUAttentionKernelT", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low - split-K reduction may be incorrect";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_Q81KVCacheConsumption_Parity)
{
    SKIP_IF_NO_CUDA();

    constexpr int kv_len = 128;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;

    const size_t q_size = static_cast<size_t>(n_heads) * head_dim;
    const size_t kv_size = static_cast<size_t>(kv_len) * n_kv_heads * head_dim;
    const size_t out_size = static_cast<size_t>(n_heads) * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data_fp32 = randomFP32(kv_size);
    auto V_data_fp32 = randomFP32(kv_size);

    std::vector<float> cpu_baseline_output(out_size, 0.0f);
    std::vector<float> cpu_q81_output(out_size, 0.0f);
    std::vector<float> cuda_q81_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;

    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_data_fp32.data(), V_data_fp32.data(), cpu_baseline_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true));

    MPIContext local_mpi_ctx(0, 1, MPI_COMM_WORLD);
    auto kv_cache = std::make_unique<CPUKVCache<ActivationPrecision::Q8_1>>(
        local_mpi_ctx,
        1,
        1,
        kv_len,
        n_kv_heads,
        head_dim,
        DeviceId::cpu());

    auto k_q81 = Q8_1Tensor::quantize_from_fp32(
        K_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto v_q81 = Q8_1Tensor::quantize_from_fp32(
        V_data_fp32.data(), {static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});

    ASSERT_NE(k_q81, nullptr);
    ASSERT_NE(v_q81, nullptr);
    ASSERT_TRUE(kv_cache->append_kv(0, 0, k_q81.get(), v_q81.get(), kv_len));

    auto gathered_K_q81 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto gathered_V_q81 = std::make_unique<Q8_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    std::vector<int> kv_lens;
    int gathered_max = kv_cache->gather_kv_batched(0, 1, gathered_K_q81.get(), gathered_V_q81.get(), kv_lens);
    ASSERT_EQ(gathered_max, kv_len);
    ASSERT_EQ(kv_lens.size(), 1u);
    ASSERT_EQ(kv_lens[0], kv_len);

    const float *K_from_q81 = gathered_K_q81->fp32_data();
    const float *V_from_q81 = gathered_V_q81->fp32_data();
    ASSERT_NE(K_from_q81, nullptr);
    ASSERT_NE(V_from_q81, nullptr);

    ASSERT_TRUE(cpu_kernel.compute_decode(
        Q_data.data(), K_from_q81, V_from_q81, cpu_q81_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true));

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q = nullptr;
    float *d_K = nullptr;
    float *d_V = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_from_q81, kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_from_q81, kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true,
        0);
    cudaDeviceSynchronize();
    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_q81_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cpu_q81_output.data(), out_size));
    ASSERT_FALSE(hasNaNOrInf(cuda_q81_output.data(), out_size));

    const double q81_cuda_cpu_cos = cosineSimilarity(cuda_q81_output.data(), cpu_q81_output.data(), out_size);
    const double q81_cuda_cpu_l2 = relativeL2Error(cuda_q81_output.data(), cpu_q81_output.data(), out_size);

    const double q81_vs_fp32_cos = cosineSimilarity(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);
    const double q81_vs_fp32_l2 = relativeL2Error(cpu_q81_output.data(), cpu_baseline_output.data(), out_size);

    printComparisonStats("FlashDecode Q8_1-consumed CUDA vs CPU", q81_cuda_cpu_cos, q81_cuda_cpu_l2,
                         maxAbsError(cuda_q81_output.data(), cpu_q81_output.data(), out_size), out_size);
    printComparisonStats("FlashDecode Q8_1-consumed CPU vs FP32 baseline", q81_vs_fp32_cos, q81_vs_fp32_l2,
                         maxAbsError(cpu_q81_output.data(), cpu_baseline_output.data(), out_size), out_size);

    EXPECT_GE(q81_cuda_cpu_cos, 0.99) << "CUDA vs CPU parity too low for Q8_1-consumed path";
    EXPECT_LE(q81_cuda_cpu_l2, 0.05) << "CUDA vs CPU L2 too high for Q8_1-consumed path";
    EXPECT_GE(q81_vs_fp32_cos, 0.95) << "Q8_1-consumed drift vs FP32 baseline too high";
    EXPECT_LE(q81_vs_fp32_l2, 0.15) << "Q8_1-consumed L2 drift vs FP32 baseline too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_VeryLong_Parity)
{
    SKIP_IF_NO_CUDA();

    // Very long KV cache - stress test for split-K with many splits
    constexpr int kv_len = 2048;
    constexpr int n_heads = 14;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, true);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    // CUDA decode
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 VeryLong Parity (kv=2048)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99) << "Cosine similarity too low";
    EXPECT_LE(l2_error, 0.05) << "L2 error too high";
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_MHA_Parity)
{
    SKIP_IF_NO_CUDA();

    // Multi-head attention (not GQA) - n_heads == n_kv_heads
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8; // MHA
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, true);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 MHA Parity (CPUAttentionKernelT vs CUDA)", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_HeadDim128_Parity)
{
    SKIP_IF_NO_CUDA();

    // Llama-style head_dim=128
    constexpr int kv_len = 256;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, true);
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        true, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 HeadDim128 Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_FP32_NonCausal_Parity)
{
    SKIP_IF_NO_CUDA();

    // Non-causal decode (bidirectional attention)
    constexpr int kv_len = 128;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 64;
    const size_t q_size = n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference using production CPUAttentionKernelT::compute_decode()
    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute_decode(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        1, kv_len, n_heads, n_kv_heads, head_dim, false); // non-causal
    ASSERT_TRUE(cpu_success) << "CPU compute_decode failed";

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    // Note: compute_decode may not support non-causal, but let's test it
    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        1, kv_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashDecode FP32 NonCausal Parity", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

// ============================================================================
// Head Dimension Tests
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_HeadDim128)
{
    SKIP_IF_NO_CUDA();

    // Test with head_dim=128 (Llama-style)
    constexpr int seq_len = 32;
    constexpr int n_heads = 8;
    constexpr int n_kv_heads = 8;
    constexpr int head_dim = 128;
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = seq_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 HeadDim128", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

// ============================================================================
// Non-Causal Attention Test
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_NonCausal)
{
    SKIP_IF_NO_CUDA();

    // Non-causal (bidirectional) attention
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
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // non-causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 NonCausal", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);
}

// ============================================================================
// Causal Masking Verification Test
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashAttn2_CausalMasking)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(out_size, 0.0f);

    CPUAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;
    bool cpu_success = cpu_kernel.compute(
        Q_data.data(), K_data.data(), V_data.data(), cpu_output.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, -1);
    ASSERT_TRUE(cpu_success);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    bool cuda_success = cuda_kernel.compute(
        d_Q, d_K, d_V, d_output,
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1, nullptr, nullptr, nullptr, nullptr, false, &mpi_ctx_, 0);
    cudaDeviceSynchronize();

    ASSERT_TRUE(cuda_success);

    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), out_size));

    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double l2_error = relativeL2Error(cuda_output.data(), cpu_output.data(), out_size);
    double max_error = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    printComparisonStats("FlashAttn2 CausalMasking", cosine, l2_error, max_error, out_size);

    EXPECT_GE(cosine, 0.99);
    EXPECT_LE(l2_error, 0.05);

    // Additional verification: first position should only see V[0]
    // and last position should see weighted average of all V
    float first_pos_val = cuda_output[0];                                 // First element of first position
    float last_pos_val = cuda_output[(seq_len - 1) * n_heads * head_dim]; // First element of last position

    // First position with uniform K should output V[0] = 1.0
    EXPECT_NEAR(first_pos_val, 1.0f, 0.01f) << "First position should only attend to position 0";

    // Last position should have higher value (attending to all positions)
    EXPECT_GT(last_pos_val, first_pos_val) << "Last position should attend to more context";
}

// ============================================================================
// Batch Decoding Test
// ============================================================================

TEST_F(Test__CUDAFlashAttentionParity, FlashDecode_BatchDecoding)
{
    SKIP_IF_NO_CUDA();

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
    std::vector<float> cuda_output(batch_size * out_per_batch, 0.0f);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(0);

    // Allocate device memory for largest batch element
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_per_batch * sizeof(float));
    cudaMalloc(&d_K, kv_per_batch * sizeof(float));
    cudaMalloc(&d_V, kv_per_batch * sizeof(float));
    cudaMalloc(&d_output, out_per_batch * sizeof(float));

    bool all_success = true;

    // Process each batch element
    for (int b = 0; b < batch_size; b++)
    {
        cudaMemcpy(d_Q, Q_data.data() + b * q_per_batch,
                   q_per_batch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, K_data.data() + b * kv_per_batch,
                   kv_per_batch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, V_data.data() + b * kv_per_batch,
                   kv_per_batch * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, out_per_batch * sizeof(float));

        // Use compute_decode for single-token decode
        bool success = cuda_kernel.compute_decode(
            d_Q, d_K, d_V, d_output,
            1,      // seq_len = 1 for decode
            kv_len, // kv_len from cache
            n_heads, n_kv_heads, head_dim,
            true, // causal
            0);   // position_offset
        cudaDeviceSynchronize();

        if (!success)
        {
            std::cerr << "Batch " << b << " decode failed" << std::endl;
            all_success = false;
            continue;
        }

        cudaMemcpy(cuda_output.data() + b * out_per_batch, d_output,
                   out_per_batch * sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);

    ASSERT_TRUE(all_success) << "All batch decode operations should succeed";
    ASSERT_FALSE(hasNaNOrInf(cuda_output.data(), batch_size * out_per_batch));

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
                float val = cuda_output[b * out_per_batch + h * head_dim + d];
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
        float diff = cuda_output[i] - cuda_output[out_per_batch + i];
        diff_sum += diff * diff;
    }
    EXPECT_GT(diff_sum, 0.0f) << "Different batch inputs should produce different outputs";

    std::cout << "  FlashDecode BatchDecoding: batch_size=" << batch_size
              << ", kv_len=" << kv_len << ", n_heads=" << n_heads
              << ", n_kv_heads=" << n_kv_heads << " - PASSED" << std::endl;
}

#else // !HAVE_CUDA

TEST_F(Test__CUDAFlashAttentionParity, SkipWithoutCUDA)
{
    GTEST_SKIP() << "CUDA not available";
}

#endif
