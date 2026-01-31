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
#include "v2/kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h"

using namespace llaminar2;

class Test__Q4_1Tensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Mock MPI context (rank 0, size 1)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;

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

TEST_F(Test__Q4_1Tensor, GemmCorrectness_SingleBlock)
{
    // M=1, N=1, K=32
    int m = 1;
    int n = 1;
    int k = 32;

    // Prepare raw data for 1 block
    std::vector<uint8_t> raw_data(sizeof(Q4_1Block));
    Q4_1Block *block = reinterpret_cast<Q4_1Block *>(raw_data.data());

    // Set up block 0
    // 1.0f in FP16 is 0x3c00
    // 0.5f in FP16 is 0x3800
    block->d = 0x3c00; // scale = 1.0
    block->m = 0x3800; // min = 0.5

    // Fill nibbles
    // qs[i] contains two nibbles: low 4 bits (even index), high 4 bits (odd index)
    // q[0] = 0, q[1] = 1, ...
    for (int i = 0; i < 16; ++i)
    {
        uint8_t low = (2 * i) % 16;
        uint8_t high = (2 * i + 1) % 16;
        block->qs[i] = (high << 4) | low;
    }

    // Create Q4_1 tensor (weights) - 1x32
    auto weights = std::make_unique<Q4_1Tensor>(std::vector<size_t>{1, 32}, raw_data);

    // Create Input tensor (A) - 1x32
    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({1, 32});
    float *input_data = input->mutable_data();
    for (int i = 0; i < 32; ++i)
    {
        input_data[i] = 1.0f;
    }

    // Create Output tensor (C) - 1x1
    auto output = factory.createFP32({1, 1});
    float *output_data = output->mutable_data();
    output_data[0] = 0.0f;

    // Create GEMM kernel
    auto gemm = weights->createGemm();
    ASSERT_NE(gemm, nullptr);

    // Run GEMM
    // multiply(A, C, m, n, k)
    bool success = gemm->multiply(input_data, output_data, m, n, k);
    ASSERT_TRUE(success);

    // Calculate expected result
    float expected_sum = 0.0f;
    for (int i = 0; i < 32; ++i)
    {
        uint8_t nibble = i % 16;
        float val = nibble * 1.0f + 0.5f; // d * q + m
        expected_sum += val * 1.0f;       // input is 1.0
    }

    // Note: QuantisedGemmKernel quantizes the input (A) to Q8_1 (int8 + fp16 scale) on the fly.
    // This introduces a small precision loss due to FP16 scale round-tripping.
    // For 1.0 input, scale is 1/127 stored as FP16. 127 * fp16(1/127) ~= 0.9999...
    // Accumulated over 32 elements, this results in ~0.016 error (255.984 vs 256.0).
    // We use 0.5f tolerance to account for this inherent quantization noise.
    float diff = output_data[0] - expected_sum;
    float l2_error = std::sqrt(diff * diff);
    EXPECT_LT(l2_error, 0.5f) << "L2 error too high: " << l2_error << ", expected: " << expected_sum << ", actual: " << output_data[0];
}

TEST_F(Test__Q4_1Tensor, GemmCorrectness_MultipleBlocks)
{
    // M=1, N=1, K=64 (2 blocks)
    int m = 1;
    int n = 1;
    int k = 64;

    // Prepare raw data for 2 blocks
    std::vector<uint8_t> raw_data(2 * sizeof(Q4_1Block));
    Q4_1Block *blocks = reinterpret_cast<Q4_1Block *>(raw_data.data());

    // Block 0: scale=1.0, min=0.0
    blocks[0].d = 0x3c00; // 1.0
    blocks[0].m = 0x0000; // 0.0
    for (int i = 0; i < 16; ++i)
        blocks[0].qs[i] = 0x11 * i; // Pattern

    // Block 1: scale=2.0, min=1.0
    // 2.0f in FP16 is 0x4000
    // 1.0f in FP16 is 0x3c00
    blocks[1].d = 0x4000;
    blocks[1].m = 0x3c00;
    for (int i = 0; i < 16; ++i)
        blocks[1].qs[i] = 0xFF; // All 15s

    auto weights = std::make_unique<Q4_1Tensor>(std::vector<size_t>{1, 64}, raw_data);

    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({1, 64});
    float *input_data = input->mutable_data();
    for (int i = 0; i < 64; ++i)
        input_data[i] = 0.5f;

    auto output = factory.createFP32({1, 1});
    float *output_data = output->mutable_data();
    output_data[0] = 0.0f;

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply(input_data, output_data, m, n, k));

    float expected_sum = 0.0f;
    // Block 0
    for (int i = 0; i < 32; ++i)
    {
        uint8_t nibble = (i % 2 == 0) ? (blocks[0].qs[i / 2] & 0x0F) : (blocks[0].qs[i / 2] >> 4);
        float val = nibble * 1.0f + 0.0f;
        expected_sum += val * 0.5f;
    }
    // Block 1
    for (int i = 0; i < 32; ++i)
    {
        uint8_t nibble = 15;
        float val = nibble * 2.0f + 1.0f;
        expected_sum += val * 0.5f;
    }

    // Note: Same quantization noise applies here (input is 0.5f).
    float diff = output_data[0] - expected_sum;
    float l2_error = std::sqrt(diff * diff);
    EXPECT_LT(l2_error, 0.5f) << "L2 error too high: " << l2_error << ", expected: " << expected_sum << ", actual: " << output_data[0];
}

/**
 * @brief Compare QuantisedGemmKernel (INT8) vs FloatingPointGemmKernel (FP32) for Q4_1.
 *
 * This test verifies that the quantized GEMM kernel produces results close to
 * the FP32 reference implementation using OneDNN.
 */
TEST_F(Test__Q4_1Tensor, QuantizedVsFP32Parity)
{
    // Realistic dimensions: 64 tokens, 512 hidden dim
    const int m = 64;
    const int n = 512;
    const int k = 512;

    // Q4_1: 32 elements per block
    const size_t num_blocks = (static_cast<size_t>(n) * k) / 32;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q4_1Block));
    Q4_1Block *blocks = reinterpret_cast<Q4_1Block *>(raw_data.data());

    // Initialize with random but valid data
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    for (size_t b = 0; b < num_blocks; ++b)
    {
        // Valid FP16 scale and min factors
        blocks[b].d = fp32_to_fp16(scale_dist(rng));
        blocks[b].m = fp32_to_fp16(scale_dist(rng));
        // Random qs (4-bit values packed into 16 bytes)
        for (int i = 0; i < 16; ++i)
        {
            blocks[b].qs[i] = byte_dist(rng);
        }
    }

    // Create quantized tensor
    auto q4_1_tensor = std::make_unique<Q4_1Tensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
        raw_data);

    // Dequantize to FP32 for reference
    TensorFactory factory(*mpi_ctx_);
    auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    q4_1_tensor->to_fp32(fp32_weights->mutable_data());

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
    auto quantized_gemm = q4_1_tensor->createGemm();
    ASSERT_TRUE(quantized_gemm->multiply(
        input_data,
        output_quantized->mutable_data(),
        m, n, k));

    // Run FP32 GEMM (OneDNN reference)
    gemm_v4::FloatingPointGemmKernel fp32_gemm(fp32_weights.get());
    ASSERT_TRUE(fp32_gemm.multiply(
        input_data,
        output_fp32->mutable_data(),
        m, n, k));

    // Compare results
    float rel_l2_error = compute_relative_l2_error(
        output_quantized->data(),
        output_fp32->data(),
        m * n);

    std::cout << "[Q4_1 Parity] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

    // 1% tolerance for quantization error
    EXPECT_LT(rel_l2_error, 0.01f)
        << "Q4_1 quantized GEMM diverged from FP32 reference by "
        << (rel_l2_error * 100.0f) << "%";
}
