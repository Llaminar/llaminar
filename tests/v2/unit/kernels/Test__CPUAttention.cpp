/**
 * @file Test__CPUAttention.cpp
 * @brief Unit tests for CPUAttention kernel via ITensorAttention interface
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/kernels/cpu/attention/CpuAttentionKernelT.h"
#include "../../../../src/v2/tensors/Tensors.h"
#include <memory>
#include <vector>
#include <cmath>

using namespace llaminar2;

/**
 * @brief Test that CPUAttention can be created via FP32Tensor factory
 */
TEST(CPUAttentionInterface, CreateViaFactory)
{
    // Create an FP32 tensor
    FP32Tensor tensor({10, 128});

    // Create attention kernel via factory method
    auto attention = tensor.createAttention();

    ASSERT_NE(attention, nullptr);
    EXPECT_TRUE(attention->supports_device(-1)); // CPU
    EXPECT_FALSE(attention->supports_device(0)); // GPU
}

/**
 * @brief Test basic attention computation via interface
 */
TEST(CPUAttentionInterface, BasicComputation)
{
    const int seq_len = 4;
    const int n_heads = 2;
    const int n_kv_heads = 2; // MHA
    const int head_dim = 8;

    // Create input data (small values for numerical stability)
    std::vector<float> Q_data(seq_len * n_heads * head_dim, 0.1f);
    std::vector<float> K_data(seq_len * n_kv_heads * head_dim, 0.1f);
    std::vector<float> V_data(seq_len * n_kv_heads * head_dim, 0.1f);
    std::vector<float> output_data(seq_len * n_heads * head_dim, 0.0f);

    // Create attention kernel
    CpuAttentionKernelT<FP32Tensor> attention;

    // Compute attention
    bool success = attention.compute(
        Q_data.data(), K_data.data(), V_data.data(),
        output_data.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false,                              // causal
        -1,                                 // window_size
        nullptr, nullptr, nullptr, nullptr, // workspaces (auto-allocated)
        false,                              // use_bf16
        nullptr,                            // mpi_ctx
        -1);                                // device_idx (CPU)

    ASSERT_TRUE(success);

    // Verify output is not all zeros
    float sum = 0.0f;
    for (float v : output_data)
    {
        sum += std::abs(v);
    }
    EXPECT_GT(sum, 0.0f);
}

/**
 * @brief Test causal masking via interface
 */
TEST(CPUAttentionInterface, CausalMasking)
{
    const int seq_len = 4;
    const int n_heads = 1;
    const int n_kv_heads = 1;
    const int head_dim = 4;

    // Create input data
    std::vector<float> Q_data(seq_len * n_heads * head_dim, 1.0f);
    std::vector<float> K_data(seq_len * n_kv_heads * head_dim, 1.0f);
    std::vector<float> V_data(seq_len * n_kv_heads * head_dim);

    // Set V to have distinguishable values per position
    for (int i = 0; i < seq_len; ++i)
    {
        for (int d = 0; d < head_dim; ++d)
        {
            V_data[i * head_dim + d] = static_cast<float>(i + 1);
        }
    }

    std::vector<float> output_data(seq_len * n_heads * head_dim, 0.0f);

    // Create attention kernel
    CpuAttentionKernelT<FP32Tensor> attention;

    // Compute with causal masking
    bool success = attention.compute(
        Q_data.data(), K_data.data(), V_data.data(),
        output_data.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        true, // causal
        -1,   // window_size
        nullptr, nullptr, nullptr, nullptr,
        false, nullptr, -1);

    ASSERT_TRUE(success);

    // First position should only attend to itself (V[0] = 1.0)
    for (int d = 0; d < head_dim; ++d)
    {
        EXPECT_NEAR(output_data[d], 1.0f, 0.01f) << "First position should only see V[0]=1.0";
    }
}

/**
 * @brief Test error handling: null pointers
 */
TEST(CPUAttentionInterface, NullPointers)
{
    CpuAttentionKernelT<FP32Tensor> attention;
    std::vector<float> dummy(16);

    // Null Q
    EXPECT_FALSE(attention.compute(
        nullptr, dummy.data(), dummy.data(), dummy.data(),
        2, 1, 1, 4, false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Null K
    EXPECT_FALSE(attention.compute(
        dummy.data(), nullptr, dummy.data(), dummy.data(),
        2, 1, 1, 4, false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Null V
    EXPECT_FALSE(attention.compute(
        dummy.data(), dummy.data(), nullptr, dummy.data(),
        2, 1, 1, 4, false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));

    // Null output
    EXPECT_FALSE(attention.compute(
        dummy.data(), dummy.data(), dummy.data(), nullptr,
        2, 1, 1, 4, false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, -1));
}

/**
 * @brief Test error handling: wrong device
 */
TEST(CPUAttentionInterface, WrongDevice)
{
    CpuAttentionKernelT<FP32Tensor> attention;
    std::vector<float> dummy(16);

    // CPU kernel shouldn't accept GPU device index
    EXPECT_FALSE(attention.compute(
        dummy.data(), dummy.data(), dummy.data(), dummy.data(),
        2, 1, 1, 4, false, -1, nullptr, nullptr, nullptr, nullptr, false, nullptr, 0)); // device_idx=0 (GPU)
}

/**
 * @brief Test BF16 mode for reduced memory bandwidth
 */
TEST(CPUAttentionInterface, BF16Mode)
{
    const int seq_len = 8;
    const int n_heads = 4;
    const int n_kv_heads = 4;
    const int head_dim = 16;

    // Create input data
    std::vector<float> Q_data(seq_len * n_heads * head_dim);
    std::vector<float> K_data(seq_len * n_kv_heads * head_dim);
    std::vector<float> V_data(seq_len * n_kv_heads * head_dim);

    // Initialize with small random-ish values
    for (size_t i = 0; i < Q_data.size(); ++i)
    {
        Q_data[i] = 0.1f * (i % 10);
        K_data[i] = 0.1f * ((i + 3) % 10);
        V_data[i] = 0.1f * ((i + 7) % 10);
    }

    std::vector<float> output_fp32(seq_len * n_heads * head_dim, 0.0f);
    std::vector<float> output_bf16(seq_len * n_heads * head_dim, 0.0f);

    // Create attention kernel
    CpuAttentionKernelT<FP32Tensor> attention;

    // Compute with FP32
    bool success_fp32 = attention.compute(
        Q_data.data(), K_data.data(), V_data.data(),
        output_fp32.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // causal
        -1,    // window_size
        nullptr, nullptr, nullptr, nullptr,
        false, // use_bf16=false (FP32 path)
        nullptr, -1);

    ASSERT_TRUE(success_fp32);

    // Compute with BF16
    bool success_bf16 = attention.compute(
        Q_data.data(), K_data.data(), V_data.data(),
        output_bf16.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        false, // causal
        -1,    // window_size
        nullptr, nullptr, nullptr, nullptr,
        true, // use_bf16=true (BF16 path)
        nullptr, -1);

    ASSERT_TRUE(success_bf16);

    // BF16 and FP32 should produce similar results
    // BF16 has ~3 decimal digits precision, so expect ~1-3% relative error
    float max_rel_error = 0.0f;
    float max_abs_error = 0.0f;

    for (size_t i = 0; i < output_fp32.size(); ++i)
    {
        float abs_error = std::abs(output_fp32[i] - output_bf16[i]);
        float rel_error = abs_error / (std::abs(output_fp32[i]) + 1e-6f);

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
    }

    // BF16 should match FP32 within reasonable tolerance
    // Allow up to 3% relative error (BF16 has 7 mantissa bits)
    EXPECT_LT(max_rel_error, 0.03f) << "BF16 relative error: " << max_rel_error;

    // Both outputs should be non-zero
    float sum_fp32 = 0.0f;
    float sum_bf16 = 0.0f;
    for (size_t i = 0; i < output_fp32.size(); ++i)
    {
        sum_fp32 += std::abs(output_fp32[i]);
        sum_bf16 += std::abs(output_bf16[i]);
    }

    EXPECT_GT(sum_fp32, 0.0f) << "FP32 output should be non-zero";
    EXPECT_GT(sum_bf16, 0.0f) << "BF16 output should be non-zero";
}
