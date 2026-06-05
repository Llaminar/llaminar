/**
 * @file Test__CUDAFlashAttention_DecodeKVLen.cpp
 * @brief Unit tests for CUDA Flash Attention decode mode with proper kv_len handling
 *
 * **Background**: Bug discovered 2026-01-07 where GPU attention in decode mode
 * was using kv_len = seq_len instead of the actual KV cache length. For decode
 * with seq_len=1 but kv_len=7, GPU only attended to 1 token instead of 7.
 *
 * **Root Cause**: CUDAFlashAttentionKernelT::compute() hardcoded:
 *   apply_typed(Q, K, V, output, 1, seq_len, seq_len, ...)
 *   should use kv_len parameter, not seq_len for decode mode.
 *
 * **Fix**: Use compute_decode() which takes separate seq_len and kv_len parameters.
 *
 * **Tests**:
 * - compute_decode() with seq_len=1, kv_len=7 produces correct attention weights
 * - compute_decode() attends to ALL kv_len tokens, not just seq_len tokens
 * - Parity with CPU reference implementation
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "utils/MPIContext.h"

#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "kernels/cuda/attention/CUDAFlashAttentionKernelT.h"
#include "kernels/cpu/attention/CPUFlashAttentionKernelT.h"
#include <cuda_runtime.h>
#endif

// Now include test utils
#include "../../../utils/CUDATestUtils.h"
#include "../../../utils/TestTensorFactory.h"

#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <memory>

using namespace llaminar2;
using namespace llaminar2::test::cuda;
using namespace llaminar2::test;

namespace
{
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
} // namespace

// ============================================================================
// Test Fixture
// ============================================================================

class Test__CUDAFlashAttention_DecodeKVLen : public CUDATestBase
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

#ifdef HAVE_CUDA
    std::unique_ptr<DeviceWorkspaceManager> bindAttentionWorkspace(
        llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> &kernel,
        int n_heads,
        int head_dim)
    {
        auto requirements = kernel.getWorkspaceRequirements(1, n_heads, head_dim);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(
            gpu_device_, requirements.total_bytes_with_alignment() + 4096);
        if (!workspace->allocate(requirements))
        {
            ADD_FAILURE() << "Failed to allocate CUDA attention workspace";
            return nullptr;
        }
        kernel.bindWorkspace(workspace.get());
        return workspace;
    }
#endif
};

#ifdef HAVE_CUDA

/**
 * @brief Test that compute_decode() properly uses kv_len, not seq_len
 *
 * This test reproduces the exact bug scenario:
 * - seq_len = 1 (decode mode, single new token)
 * - kv_len = 7 (KV cache has 7 tokens from previous prefill/decode)
 * - GPU should attend to ALL 7 tokens, not just 1
 *
 * If the bug exists (kv_len treated as seq_len=1), the attention output
 * would only consider one token and produce incorrect results.
 */
TEST_F(Test__CUDAFlashAttention_DecodeKVLen, DecodeMode_SeqLen1_KVLen7)
{
    SKIP_IF_NO_CUDA();

    // This is the exact scenario where the bug manifested
    constexpr int seq_len = 1; // Decode mode: generating one token
    constexpr int kv_len = 7;  // KV cache has 7 tokens
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 64;

    // Q has shape [seq_len * n_heads * head_dim] = [1 * 4 * 64] = 256
    // K, V have shape [kv_len * n_kv_heads * head_dim] = [7 * 4 * 64] = 1792
    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    // Create test data
    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> cpu_output(out_size, 0.0f);
    std::vector<float> cuda_output(out_size, 0.0f);

    // CPU reference - uses seq_len=1, kv_len=7 correctly
    CPUFlashAttentionKernelT<ActivationPrecision::FP32> cpu_kernel;

    // For CPU decode, we need to set up the proper attention computation
    // The CPU kernel in decode mode should attend from Q[1 token] to K/V[7 tokens]
    // We'll compute reference using manual attention calculation

    // Manual reference: softmax(Q @ K^T / sqrt(head_dim)) @ V
    // For each head: Q[1 x head_dim] @ K[kv_len x head_dim]^T = scores[1 x kv_len]
    // scores = softmax(scores / sqrt(head_dim))
    // output = scores @ V[kv_len x head_dim] = [1 x head_dim]

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int h = 0; h < n_heads; ++h)
    {
        // Q for this head: offset h * head_dim (since seq_len=1)
        const float *Q_h = Q_data.data() + h * head_dim;

        // K, V for this head (using n_kv_heads, handle GQA)
        int kv_h = h / (n_heads / n_kv_heads);
        const float *K_h = K_data.data() + kv_h * head_dim; // K is [kv_len, n_kv_heads, head_dim]
        const float *V_h = V_data.data() + kv_h * head_dim;

        // Compute attention scores: Q @ K^T
        std::vector<float> scores(kv_len);
        for (int t = 0; t < kv_len; ++t)
        {
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d)
            {
                // K is stored as [kv_len, n_kv_heads, head_dim]
                dot += Q_h[d] * K_data[t * n_kv_heads * head_dim + kv_h * head_dim + d];
            }
            scores[t] = dot * scale;
        }

        // Softmax
        float max_score = *std::max_element(scores.begin(), scores.end());
        float sum_exp = 0.0f;
        for (auto &s : scores)
        {
            s = std::exp(s - max_score);
            sum_exp += s;
        }
        for (auto &s : scores)
        {
            s /= sum_exp;
        }

        // Weighted sum of V
        float *out_h = cpu_output.data() + h * head_dim;
        for (int d = 0; d < head_dim; ++d)
        {
            float val = 0.0f;
            for (int t = 0; t < kv_len; ++t)
            {
                val += scores[t] * V_data[t * n_kv_heads * head_dim + kv_h * head_dim + d];
            }
            out_h[d] = val;
        }
    }

    // CUDA kernel using compute_decode()
    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, out_size * sizeof(float));

    // Call compute_decode() which should properly use kv_len=7
    bool cuda_success = cuda_kernel.compute_decode(
        d_Q, d_K, d_V, d_output,
        seq_len, // 1
        kv_len,  // 7 - THIS IS THE CRITICAL PARAMETER
        n_heads, n_kv_heads, head_dim,
        false, // causal=false for decode (already causal by construction)
        0      // position_offset
    );

    ASSERT_TRUE(cuda_success) << "CUDA compute_decode failed";

    // Copy result back
    cudaMemcpy(cuda_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Compare
    double cosine = cosineSimilarity(cuda_output.data(), cpu_output.data(), out_size);
    double max_err = maxAbsError(cuda_output.data(), cpu_output.data(), out_size);

    std::cout << "  DecodeMode_SeqLen1_KVLen7: cosine=" << cosine
              << ", max_err=" << max_err << std::endl;

    // Should have high similarity - if the bug existed (kv_len=1),
    // cosine would be very low
    EXPECT_GE(cosine, 0.99) << "GPU decode should attend to all kv_len tokens";
    EXPECT_LT(max_err, 0.1) << "Max error too high";

    // Cleanup
    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}

/**
 * @brief Test that using wrong kv_len produces different (wrong) results
 *
 * This demonstrates the bug behavior: if we accidentally used seq_len
 * where kv_len was needed, we'd get wrong output.
 */
TEST_F(Test__CUDAFlashAttention_DecodeKVLen, WrongKVLen_ProducesWrongOutput)
{
    SKIP_IF_NO_CUDA();

    constexpr int seq_len = 1;
    constexpr int kv_len = 16; // Larger KV cache
    constexpr int n_heads = 2;
    constexpr int n_kv_heads = 2;
    constexpr int head_dim = 32;

    const size_t q_size = seq_len * n_heads * head_dim;
    const size_t kv_size = kv_len * n_kv_heads * head_dim;
    const size_t out_size = seq_len * n_heads * head_dim;

    auto Q_data = randomFP32(q_size);
    auto K_data = randomFP32(kv_size);
    auto V_data = randomFP32(kv_size);
    std::vector<float> correct_output(out_size, 0.0f);
    std::vector<float> wrong_output(out_size, 0.0f);

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    float *d_Q, *d_K, *d_V, *d_output;
    cudaMalloc(&d_Q, q_size * sizeof(float));
    cudaMalloc(&d_K, kv_size * sizeof(float));
    cudaMalloc(&d_V, kv_size * sizeof(float));
    cudaMalloc(&d_output, out_size * sizeof(float));

    cudaMemcpy(d_Q, Q_data.data(), q_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_data.data(), kv_size * sizeof(float), cudaMemcpyHostToDevice);

    // CORRECT: Use proper kv_len
    cudaMemset(d_output, 0, out_size * sizeof(float));
    cuda_kernel.compute_decode(d_Q, d_K, d_V, d_output,
                               seq_len, kv_len, // kv_len = 16
                               n_heads, n_kv_heads, head_dim, false, 0);
    cudaMemcpy(correct_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    // WRONG: Use seq_len as kv_len (the bug behavior)
    cudaMemset(d_output, 0, out_size * sizeof(float));
    cuda_kernel.compute_decode(d_Q, d_K, d_V, d_output,
                               seq_len, seq_len, // kv_len = 1 (WRONG!)
                               n_heads, n_kv_heads, head_dim, false, 0);
    cudaMemcpy(wrong_output.data(), d_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);

    // The outputs should be DIFFERENT
    double cosine = cosineSimilarity(correct_output.data(), wrong_output.data(), out_size);

    std::cout << "  WrongKVLen_ProducesWrongOutput: cosine(correct,wrong)=" << cosine << std::endl;

    // If kv_len matters (which it should), using wrong kv_len should produce
    // significantly different output
    EXPECT_LT(cosine, 0.9) << "Using wrong kv_len should produce different output";

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}

/**
 * @brief Test multiple decode steps to simulate autoregressive generation
 *
 * Simulates: prefill(6 tokens) -> decode(1) -> decode(1) -> decode(1)
 * kv_len grows: 6 -> 7 -> 8 -> 9
 */
TEST_F(Test__CUDAFlashAttention_DecodeKVLen, MultipleDecodeSteps_GrowingKVCache)
{
    SKIP_IF_NO_CUDA();

    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 4;
    constexpr int head_dim = 64;
    constexpr int initial_kv_len = 6;
    constexpr int num_decode_steps = 3;

    llaminar2::cuda::CUDAFlashAttentionKernelT<ActivationPrecision::FP32> cuda_kernel(cuda_ordinal_);
    auto attention_workspace = bindAttentionWorkspace(cuda_kernel, n_heads, head_dim);
    ASSERT_NE(attention_workspace, nullptr);

    // Pre-allocate K/V buffer for maximum size
    const int max_kv_len = initial_kv_len + num_decode_steps;
    const size_t max_kv_size = max_kv_len * n_kv_heads * head_dim;

    // Generate full KV data upfront (simulates cache)
    auto K_full = randomFP32(max_kv_size);
    auto V_full = randomFP32(max_kv_size);

    float *d_K, *d_V, *d_Q, *d_output;
    cudaMalloc(&d_K, max_kv_size * sizeof(float));
    cudaMalloc(&d_V, max_kv_size * sizeof(float));
    cudaMalloc(&d_Q, n_heads * head_dim * sizeof(float)); // seq_len=1
    cudaMalloc(&d_output, n_heads * head_dim * sizeof(float));

    // Copy full K/V to device
    cudaMemcpy(d_K, K_full.data(), max_kv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V_full.data(), max_kv_size * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<std::vector<float>> decode_outputs;

    for (int step = 0; step < num_decode_steps; ++step)
    {
        int current_kv_len = initial_kv_len + step + 1;

        // Generate new Q for this decode step
        auto Q_data = randomFP32(n_heads * head_dim);
        cudaMemcpy(d_Q, Q_data.data(), n_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_output, 0, n_heads * head_dim * sizeof(float));

        // compute_decode with growing kv_len
        bool success = cuda_kernel.compute_decode(
            d_Q, d_K, d_V, d_output,
            1,              // seq_len = 1 (decode)
            current_kv_len, // Growing KV cache
            n_heads, n_kv_heads, head_dim,
            false, 0);

        ASSERT_TRUE(success) << "Decode step " << step << " failed";

        // Save output
        std::vector<float> output(n_heads * head_dim);
        cudaMemcpy(output.data(), d_output, n_heads * head_dim * sizeof(float), cudaMemcpyDeviceToHost);
        decode_outputs.push_back(output);
    }

    // Verify outputs are not all the same (they should differ as KV cache grows)
    for (int i = 1; i < num_decode_steps; ++i)
    {
        double cosine = cosineSimilarity(
            decode_outputs[i].data(),
            decode_outputs[i - 1].data(),
            decode_outputs[i].size());

        std::cout << "  Decode step " << i << " vs " << (i - 1) << ": cosine=" << cosine << std::endl;

        // Outputs should be different (not identical) since KV cache is different
        // But not completely uncorrelated either
        EXPECT_GT(cosine, 0.5) << "Consecutive decode outputs should be somewhat correlated";
        EXPECT_LT(cosine, 0.999) << "Consecutive decode outputs should not be identical";
    }

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}

#endif // HAVE_CUDA
