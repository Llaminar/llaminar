#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <random>
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/FP16Utils.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"

using namespace llaminar2;

class Test__Q3_KTensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Mock MPI context (rank 0, size 1)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;

    // Helper to pack scales (copied from unit test)
    void pack_scales(Q3_KBlock &block, const uint8_t *unpacked_scales)
    {
        for (int i = 0; i < 8; ++i)
        {
            block.scales[i] = (unpacked_scales[i] & 0xF) | ((unpacked_scales[i + 8] & 0xF) << 4);
        }
        for (int i = 0; i < 4; ++i)
        {
            block.scales[8] |= ((unpacked_scales[i] >> 4) & 0x3) << (i * 2);
            block.scales[9] |= ((unpacked_scales[i + 4] >> 4) & 0x3) << (i * 2);
            block.scales[10] |= ((unpacked_scales[i + 8] >> 4) & 0x3) << (i * 2);
            block.scales[11] |= ((unpacked_scales[i + 12] >> 4) & 0x3) << (i * 2);
        }
    }

    /**
     * @brief Compute relative L2 error between two output vectors.
     */
    float compute_relative_l2_error(const float *a, const float *b, size_t n)
    {
        float sum_sq_diff = 0.0f;
        float sum_sq_ref = 0.0f;
        for (size_t i = 0; i < n; ++i)
        {
            float diff = a[i] - b[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += b[i] * b[i];
        }
        if (sum_sq_ref < 1e-10f)
            return 0.0f;
        return std::sqrt(sum_sq_diff / sum_sq_ref);
    }
};

TEST_F(Test__Q3_KTensor, GemmCorrectness_SingleBlock_Zero)
{
    // M=1, N=1, K=256 (1 super-block)
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q3_KBlock));
    Q3_KBlock *block = reinterpret_cast<Q3_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q3_KBlock));

    // Set d=1.0. Scales=0 -> dl = 1.0 * (0-32) = -32.0.
    // qs=0, hmask=0 -> q=0, h=0 -> val = 0-4 = -4.
    // output = -32.0 * -4 = 128.0.
    // Wait, if we want zero output, we need dl=0 or val=0.
    // To get dl=0, we need scale=32.

    block->d = fp32_to_fp16(1.0f);
    uint8_t scales[16];
    for (int i = 0; i < 16; ++i)
        scales[i] = 32;
    pack_scales(*block, scales);

    // qs=0, hmask=0 -> val=-4.
    // output = 0.0 * -4 = 0.0.

    auto weights = std::make_unique<Q3_KTensor>(std::vector<size_t>{1, 256}, raw_data);

    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({1, 256});
    float *input_data = input->mutable_data();
    for (int i = 0; i < 256; ++i)
        input_data[i] = 1.0f;

    auto output = factory.createFP32({1, 1});
    float *output_data = output->mutable_data();
    output_data[0] = 123.0f; // Garbage

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply(input_data, output_data, m, n, k));

    EXPECT_NEAR(output_data[0], 0.0f, 1e-4f);
}

TEST_F(Test__Q3_KTensor, GemmCorrectness_SingleBlock_Ones)
{
    // M=1, N=1, K=256
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q3_KBlock));
    Q3_KBlock *block = reinterpret_cast<Q3_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q3_KBlock));

    block->d = fp32_to_fp16(1.0f);

    // Set scales to 33 (dl = 1.0 * (33-32) = 1.0)
    uint8_t scales[16];
    for (int i = 0; i < 16; ++i)
        scales[i] = 33;
    pack_scales(*block, scales);

    // We want val=1.
    // val = q - 4 (if h=0) or q (if h=1).
    // If h=1, q=1 -> val=1.
    // q=1 -> bits 01. h=1 -> bit 1.

    // Set qs to 0x55 (01 01 01 01) -> q=1 for all.
    std::memset(block->qs, 0x55, 64);
    // Set hmask to 0xFF -> h=1 for all.
    std::memset(block->hmask, 0xFF, 32);

    auto weights = std::make_unique<Q3_KTensor>(std::vector<size_t>{1, 256}, raw_data);

    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({1, 256});
    float *input_data = input->mutable_data();
    for (int i = 0; i < 256; ++i)
        input_data[i] = 1.0f;

    auto output = factory.createFP32({1, 1});
    float *output_data = output->mutable_data();
    output_data[0] = 0.0f;

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply(input_data, output_data, m, n, k));

    // Expected: 256 elements * 1.0 * 1.0 = 256.0
    float expected = 256.0f;
    float diff = output_data[0] - expected;
    float l2_error = std::sqrt(diff * diff);

    // Allow some error due to INT8 re-quantization
    EXPECT_LT(l2_error, 5.0f);
}

/**
 * @brief Compare CPUNativeVNNIGemmKernel (INT8) vs FloatingPointGemmKernel (FP32) for Q3_K.
 *
 * This test verifies that the quantized GEMM kernel produces results close to
 * the FP32 reference implementation using OneDNN.
 */
TEST_F(Test__Q3_KTensor, QuantizedVsFP32Parity)
{
    // Realistic dimensions: 64 tokens, 512 hidden dim (2 super-blocks)
    const int m = 64;
    const int n = 512;
    const int k = 512;

    // Q3_K: 256 elements per super-block
    const size_t num_blocks = (static_cast<size_t>(n) * k) / 256;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q3_KBlock));
    Q3_KBlock *blocks = reinterpret_cast<Q3_KBlock *>(raw_data.data());

    // Initialize with random but valid data
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    for (size_t b = 0; b < num_blocks; ++b)
    {
        // Random hmask (high bit for 3-bit values)
        for (int i = 0; i < 32; ++i)
        {
            blocks[b].hmask[i] = byte_dist(rng);
        }
        // Random qs (low 2 bits of 3-bit values, packed)
        for (int i = 0; i < 64; ++i)
        {
            blocks[b].qs[i] = byte_dist(rng);
        }
        // Random scales (12 bytes packed format)
        for (int i = 0; i < 12; ++i)
        {
            blocks[b].scales[i] = byte_dist(rng);
        }
        // Valid FP16 scale factor
        blocks[b].d = fp32_to_fp16(scale_dist(rng));
    }

    // Create quantized tensor
    auto q3k_tensor = std::make_unique<Q3_KTensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
        raw_data);

    // Dequantize to FP32 for reference
    TensorFactory factory(*mpi_ctx_);
    auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    q3k_tensor->to_fp32(fp32_weights->mutable_data());

    // Create random input activations
    auto input = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    float *input_data = input->mutable_data();
    std::uniform_real_distribution<float> input_dist(-1.0f, 1.0f);
    for (int i = 0; i < m * k; ++i)
    {
        input_data[i] = input_dist(rng);
    }

    // Allocate outputs
    auto output_quantized = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    auto output_fp32 = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::memset(output_quantized->mutable_data(), 0, m * n * sizeof(float));
    std::memset(output_fp32->mutable_data(), 0, m * n * sizeof(float));

    // Run quantized GEMM (INT8 path)
    auto quantized_gemm = q3k_tensor->createGemm();
    ASSERT_TRUE(quantized_gemm->multiply(
        input_data,
        output_quantized->mutable_data(),
        m, n, k));

    // Run FP32 GEMM (OneDNN reference)
    gemm::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());
    ASSERT_TRUE(fp32_gemm.multiply(
        input_data,
        output_fp32->mutable_data(),
        m, n, k));

    // Compare results
    float rel_l2_error = compute_relative_l2_error(
        output_quantized->data(),
        output_fp32->data(),
        m * n);

    std::cout << "[Q3_K Parity] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

    // 1% tolerance for quantization error
    EXPECT_LT(rel_l2_error, 0.01f)
        << "Q3_K quantized GEMM diverged from FP32 reference by "
        << (rel_l2_error * 100.0f) << "%";
}
