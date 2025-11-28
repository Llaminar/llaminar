/**
 * @file Test__CPURoPEKernelT.cpp
 * @brief Unit tests for CPURoPEKernelT class
 */

#include "kernels/cpu/ops/CPURoPEKernelT.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace llaminar2;

class CPURoPEKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }
};

TEST_F(CPURoPEKernelTest, FP32_Apply_Basic)
{
    CPURoPEKernelT<FP32Tensor> kernel;
    int seq_len = 1;
    int n_heads = 2;
    int n_kv_heads = 2;
    int head_dim = 64;
    float rope_theta = 10000.0f;

    std::vector<float> Q(seq_len * n_heads * head_dim, 1.0f);
    std::vector<float> K(seq_len * n_kv_heads * head_dim, 1.0f);
    std::vector<int> pos_ids = {0};

    bool result = kernel.apply(
        Q.data(), K.data(),
        pos_ids.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        rope_theta,
        false,   // use_bf16
        nullptr, // mpi_ctx
        -1       // device_idx
    );

    EXPECT_TRUE(result);
}

TEST_F(CPURoPEKernelTest, FP32_Apply_Padding)
{
    CPURoPEKernelT<FP32Tensor> kernel;
    int seq_len = 2;
    int n_heads = 1;
    int n_kv_heads = 1;
    int head_dim = 64;
    float rope_theta = 10000.0f;

    std::vector<float> Q(seq_len * n_heads * head_dim, 1.0f);
    std::vector<float> K(seq_len * n_kv_heads * head_dim, 1.0f);
    std::vector<int> pos_ids = {0, -1}; // Second token is padding

    bool result = kernel.apply(
        Q.data(), K.data(),
        pos_ids.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        rope_theta,
        false,   // use_bf16
        nullptr, // mpi_ctx
        -1       // device_idx
    );

    EXPECT_TRUE(result);

    // Check that padding token was not modified (simplified check)
    // In reality, RoPE modifies in place, so if it ran, values would change.
    // Since we initialized to 1.0f, if it didn't run, they should stay 1.0f.
    // But wait, 1.0f rotated might still be close to 1.0f depending on position 0?
    // Position is -1, so it should skip.

    // Let's check the second token values.
    for (int i = 0; i < head_dim; ++i)
    {
        EXPECT_EQ(Q[head_dim + i], 1.0f);
    }
}

TEST_F(CPURoPEKernelTest, Q8_1_Apply_Basic)
{
    CPURoPEKernelT<Q8_1Tensor> kernel;
    int seq_len = 1;
    int n_heads = 2;
    int n_kv_heads = 2;
    int head_dim = 64; // Must be multiple of 32 for Q8_1
    float rope_theta = 10000.0f;

    int blocks_per_head = head_dim / 32;
    std::vector<Q8_1Block> Q(seq_len * n_heads * blocks_per_head);
    std::vector<Q8_1Block> K(seq_len * n_kv_heads * blocks_per_head);
    std::vector<int> pos_ids = {0};

    // Initialize blocks
    for (auto &b : Q)
    {
        b.d = 1.0f;
        b.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
            b.qs[i] = 1;
    }
    for (auto &b : K)
    {
        b.d = 1.0f;
        b.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
            b.qs[i] = 1;
    }

    bool result = kernel.apply_q8_1(
        Q.data(), K.data(),
        pos_ids.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        rope_theta,
        -1 // device_idx
    );

    EXPECT_TRUE(result);
}

TEST_F(CPURoPEKernelTest, Q8_1_Apply_InvalidHeadDim)
{
    CPURoPEKernelT<Q8_1Tensor> kernel;
    int seq_len = 1;
    int n_heads = 1;
    int n_kv_heads = 1;
    int head_dim = 48; // Not multiple of 32
    float rope_theta = 10000.0f;

    std::vector<Q8_1Block> Q(10); // Dummy size
    std::vector<Q8_1Block> K(10);
    std::vector<int> pos_ids = {0};

    bool result = kernel.apply_q8_1(
        Q.data(), K.data(),
        pos_ids.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        rope_theta,
        -1 // device_idx
    );

    EXPECT_FALSE(result);
}
