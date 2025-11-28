/**
 * @file Test__CPUSwiGLUKernelT.cpp
 * @brief Unit tests for CPUSwiGLUKernelT class
 */

#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "tensors/Tensors.h"
#include "tensors/FP16Utils.h"
#include "tensors/SIMDHelpers.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>

using namespace llaminar2;

class CPUSwiGLUKernelTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
    }

    // Helper to compute expected SwiGLU
    float swiglu_ref(float g, float u)
    {
        float silu = u / (1.0f + std::exp(-u));
        return g * silu;
    }
};

TEST_F(CPUSwiGLUKernelTest, FP32_Apply_Basic)
{
    CPUSwiGLUKernelT<FP32Tensor> kernel;
    int rows = 1;
    int cols = 10;
    int size = rows * cols;

    std::vector<float> gate(size, 1.0f);
    std::vector<float> up(size, 2.0f);
    std::vector<float> output(size);

    bool result = kernel.apply(
        gate.data(), up.data(), output.data(),
        rows, cols,
        false,   // add_residual
        nullptr, // mpi_ctx
        -1       // device_idx
    );

    EXPECT_TRUE(result);

    for (int i = 0; i < size; ++i)
    {
        float expected = swiglu_ref(gate[i], up[i]);
        EXPECT_NEAR(output[i], expected, 1e-5f);
    }
}

TEST_F(CPUSwiGLUKernelTest, BF16_Apply_Basic)
{
    CPUSwiGLUKernelT<BF16Tensor> kernel;
    int rows = 1;
    int cols = 10;
    int size = rows * cols;

    std::vector<uint16_t> gate(size);
    std::vector<uint16_t> up(size);
    std::vector<uint16_t> output(size);

    for (int i = 0; i < size; ++i)
    {
        gate[i] = simd::fp32_to_bf16(1.0f);
        up[i] = simd::fp32_to_bf16(2.0f);
    }

    bool result = kernel.apply_bf16(
        gate.data(), up.data(), output.data(),
        rows, cols,
        false,
        nullptr,
        -1);

    EXPECT_TRUE(result);

    for (int i = 0; i < size; ++i)
    {
        float g = simd::bf16_to_fp32(gate[i]);
        float u = simd::bf16_to_fp32(up[i]);
        float out = simd::bf16_to_fp32(output[i]);
        float expected = swiglu_ref(g, u);
        EXPECT_NEAR(out, expected, 1e-2f); // BF16 has lower precision
    }
}

TEST_F(CPUSwiGLUKernelTest, FP16_Apply_Basic)
{
    CPUSwiGLUKernelT<FP16Tensor> kernel;
    int rows = 1;
    int cols = 10;
    int size = rows * cols;

    std::vector<uint16_t> gate(size);
    std::vector<uint16_t> up(size);
    std::vector<uint16_t> output(size);

    for (int i = 0; i < size; ++i)
    {
        gate[i] = simd::fp32_to_fp16(1.0f);
        up[i] = simd::fp32_to_fp16(2.0f);
    }

    bool result = kernel.apply_fp16(
        gate.data(), up.data(), output.data(),
        rows, cols,
        false,
        nullptr,
        -1);

    EXPECT_TRUE(result);

    for (int i = 0; i < size; ++i)
    {
        float g = simd::fp16_to_fp32(gate[i]);
        float u = simd::fp16_to_fp32(up[i]);
        float out = simd::fp16_to_fp32(output[i]);
        float expected = swiglu_ref(g, u);
        EXPECT_NEAR(out, expected, 1e-3f);
    }
}

TEST_F(CPUSwiGLUKernelTest, Q8_1_Apply_Basic)
{
    CPUSwiGLUKernelT<Q8_1Tensor> kernel;
    int rows = 1;
    int cols = 32; // Must be multiple of 32
    int size = rows * cols;
    int num_blocks = size / 32;

    std::vector<Q8_1Block> gate(num_blocks);
    std::vector<Q8_1Block> up(num_blocks);
    std::vector<Q8_1Block> output(num_blocks);

    // Initialize blocks
    for (auto &b : gate)
    {
        b.d = simd::fp32_to_fp16(1.0f);
        b.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
            b.qs[i] = 1; // Value = 1.0
    }
    for (auto &b : up)
    {
        b.d = simd::fp32_to_fp16(1.0f);
        b.sum_qs = 0;
        for (int i = 0; i < 32; ++i)
            b.qs[i] = 2; // Value = 2.0
    }

    bool result = kernel.apply_q8_1(
        gate.data(), up.data(), output.data(),
        rows, cols,
        false,
        nullptr,
        -1);

    EXPECT_TRUE(result);

    // Check output
    for (int b = 0; b < num_blocks; ++b)
    {
        float d = simd::fp16_to_fp32(output[b].d);
        for (int i = 0; i < 32; ++i)
        {
            float val = d * output[b].qs[i];
            float expected = swiglu_ref(1.0f, 2.0f);
            EXPECT_NEAR(val, expected, 0.1f); // Quantization error
        }
    }
}
