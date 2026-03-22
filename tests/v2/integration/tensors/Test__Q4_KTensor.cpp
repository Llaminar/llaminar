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

class Test__Q4_KTensor : public ::testing::Test
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

TEST_F(Test__Q4_KTensor, GemmCorrectness_SingleBlock_Zero)
{
    // M=1, N=1, K=256 (1 super-block)
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q4_KBlock));
    Q4_KBlock *block = reinterpret_cast<Q4_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q4_KBlock));

    // Set scales
    block->d = fp32_to_fp16(1.0f);
    block->dmin = fp32_to_fp16(0.0f);
    // scales[i] = 0 -> d=0, m=0.
    // So values will be 0.

    auto weights = std::make_unique<Q4_KTensor>(std::vector<size_t>{1, 256}, raw_data);

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

    EXPECT_NEAR(output_data[0], 0.0f, 1e-5f);
}

TEST_F(Test__Q4_KTensor, GemmCorrectness_SingleBlock_Ones)
{
    // M=1, N=1, K=256
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q4_KBlock));
    Q4_KBlock *block = reinterpret_cast<Q4_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q4_KBlock));

    block->d = fp32_to_fp16(1.0f);
    block->dmin = fp32_to_fp16(0.0f);

    // Set scales to 1.0 (d=1, m=0)
    // Packing logic from SIMDHelpers.h get_scale_min_k4:
    // j < 4: d = q[j] & 63, m = q[j+4] & 63
    // j >= 4: d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
    //         m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4)

    // We want d=1, m=0 for all 8 sub-blocks.
    // q[0..3] = 1 (d for j=0..3)
    // q[4..7] = 0 (m for j=0..3)
    // q[8..11] = 1 (d for j=4..7, low nibble 1, high nibble 0)
    // Note: q[0..3] >> 6 is 0. q[4..7] >> 6 is 0.

    for (int i = 0; i < 4; ++i)
        block->scales[i] = 1;
    for (int i = 4; i < 8; ++i)
        block->scales[i] = 0;
    for (int i = 8; i < 12; ++i)
        block->scales[i] = 1;

    // Set qs to 1.0
    // val = d * scale * (q - zero) - dmin * min_scale
    // val = 1.0 * 1.0 * (q & 0xF) - 0.0
    // We want val = 1.0, so q & 0xF = 1.
    // Also for upper nibble: q >> 4 = 1.
    // So q = 0x11.
    for (int i = 0; i < 128; ++i)
        block->qs[i] = 0x11;

    auto weights = std::make_unique<Q4_KTensor>(std::vector<size_t>{1, 256}, raw_data);

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
    EXPECT_NEAR(output_data[0], expected, 5.0f) << "Output: " << output_data[0];
}

/**
 * @brief Compare CPUNativeVNNIGemmKernel (INT8) vs FloatingPointGemmKernel (FP32) for Q4_K.
 *
 * This test verifies that the quantized GEMM kernel produces results close to
 * the FP32 reference implementation using OneDNN.
 */
TEST_F(Test__Q4_KTensor, QuantizedVsFP32Parity)
{
    // Realistic dimensions: 64 tokens, 512 hidden dim (2 super-blocks)
    const int m = 64;
    const int n = 512;
    const int k = 512;

    // Q4_K: 256 elements per super-block
    const size_t num_blocks = (static_cast<size_t>(n) * k) / 256;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q4_KBlock));
    Q4_KBlock *blocks = reinterpret_cast<Q4_KBlock *>(raw_data.data());

    // Initialize with random but valid data
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);

    for (size_t b = 0; b < num_blocks; ++b)
    {
        // Valid FP16 scale factors
        blocks[b].d = fp32_to_fp16(scale_dist(rng));
        blocks[b].dmin = fp32_to_fp16(scale_dist(rng));
        // Random scales (12 bytes packed)
        for (int i = 0; i < 12; ++i)
        {
            blocks[b].scales[i] = byte_dist(rng);
        }
        // Random qs (4-bit values packed into 128 bytes)
        for (int i = 0; i < 128; ++i)
        {
            blocks[b].qs[i] = byte_dist(rng);
        }
    }

    // Create quantized tensor
    auto q4k_tensor = std::make_unique<Q4_KTensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)},
        raw_data);

    // Dequantize to FP32 for reference
    TensorFactory factory(*mpi_ctx_);
    auto fp32_weights = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    q4k_tensor->to_fp32(fp32_weights->mutable_data());

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
    auto quantized_gemm = q4k_tensor->createGemm();
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

    std::cout << "[Q4_K Parity] Relative L2 error: " << (rel_l2_error * 100.0f) << "%" << std::endl;

    // 1% tolerance for quantization error
    EXPECT_LT(rel_l2_error, 0.01f)
        << "Q4_K quantized GEMM diverged from FP32 reference by "
        << (rel_l2_error * 100.0f) << "%";
}
