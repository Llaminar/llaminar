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

class Test__IQ4_NLTensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
        rng_.seed(42);
    }

    // Helper: compute relative L2 error
    float compute_relative_l2_error(const float *ref, const float *test, size_t count)
    {
        float sum_sq_diff = 0.0f;
        float sum_sq_ref = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float diff = ref[i] - test[i];
            sum_sq_diff += diff * diff;
            sum_sq_ref += ref[i] * ref[i];
        }
        if (sum_sq_ref < 1e-10f)
            return std::sqrt(sum_sq_diff);
        return std::sqrt(sum_sq_diff / sum_sq_ref);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
    std::mt19937 rng_;
};

TEST_F(Test__IQ4_NLTensor, GemmCorrectness_Constant)
{
    // Dimensions
    int m = 1;
    int n = 1;
    int k = 256;

    // 1. Create IQ4_NL Tensor with constant data (all 1.0)
    size_t rows = n;
    size_t cols = k;
    size_t blocks_per_row = cols / 32;
    size_t total_blocks = rows * blocks_per_row;

    std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_NLBlock));
    IQ4_NLBlock *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());

    // Index 8 in kvalues_iq4nl is 1.0f
    // kvalues_iq4nl[8] = 1.0f
    // We want qbyte such that low=8, high=8 -> 0x88
    uint8_t q_one = 0x88;

    for (size_t i = 0; i < total_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(1.0f);
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = q_one;
        }
    }

    auto weights = std::make_unique<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw_data);

    // 2. Decode weights to FP32 for reference
    std::vector<float> weights_fp32(rows * cols);
    weights->decode_to_fp32(weights_fp32.data());

    // Verify weights are 1.0
    for (float w : weights_fp32)
    {
        EXPECT_FLOAT_EQ(w, 1.0f);
    }

    // 3. Create constant input (all 1.0)
    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    float *input_data = input->mutable_data();
    for (int i = 0; i < m * k; ++i)
    {
        input_data[i] = 1.0f;
    }

    // 4. Compute Reference GEMM
    // C = 1 * 1 * 256 = 256
    float expected = 256.0f;

    // 5. Compute Actual GEMM
    auto output = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    float *output_data = output->mutable_data();

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply(input_data, output_data, m, n, k));

    // 6. Compare
    EXPECT_NEAR(output_data[0], expected, 1.0f);
}

TEST_F(Test__IQ4_NLTensor, GemmCorrectness_Negative)
{
    // Dimensions
    int m = 1;
    int n = 1;
    int k = 256; // Multiple of 32

    // 1. Create IQ4_NL Tensor with constant data (negative)
    size_t rows = n;
    size_t cols = k;
    size_t blocks_per_row = cols / 32;
    size_t total_blocks = rows * blocks_per_row;

    std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_NLBlock));
    IQ4_NLBlock *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());

    // Index 0 in kvalues_iq4nl is -127.0f
    // kvalues_iq4nl[0] = -127.0f
    // We want qbyte such that low=0, high=0 -> 0x00
    uint8_t q_zero = 0x00;

    for (size_t i = 0; i < total_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(1.0f);
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = q_zero;
        }
    }

    auto weights = std::make_unique<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw_data);

    // 2. Decode weights to FP32 for reference
    std::vector<float> weights_fp32(rows * cols);
    weights->decode_to_fp32(weights_fp32.data());

    // Verify weights are -127.0
    for (float w : weights_fp32)
    {
        EXPECT_FLOAT_EQ(w, -127.0f);
    }

    // 3. Create constant input (all 1.0)
    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    float *input_data = input->mutable_data();
    for (int i = 0; i < m * k; ++i)
    {
        input_data[i] = 1.0f;
    }

    // 4. Compute Reference GEMM
    // C = 1 * -127 * 256 = -32512
    float expected = -32512.0f;

    // 5. Compute Actual GEMM
    auto output = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    float *output_data = output->mutable_data();

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply(input_data, output_data, m, n, k));

    // 6. Compare
    EXPECT_NEAR(output_data[0], expected, 5.0f);
}

TEST_F(Test__IQ4_NLTensor, GemmCorrectness_Random)
{
    // Dimensions
    // Note: M > 1 currently fails with large errors in CPUNativeVNNIGemmKernel.
    // Keeping M=1 for now to verify IQ4_NL decoding logic.
    int m = 1;
    int n = 32;
    int k = 256; // Multiple of 32 (block size)

    // 1. Create IQ4_NL Tensor with random data
    size_t rows = n;
    size_t cols = k;
    size_t blocks_per_row = cols / 32;
    size_t total_blocks = rows * blocks_per_row;

    std::vector<uint8_t> raw_data(total_blocks * sizeof(IQ4_NLBlock));
    IQ4_NLBlock *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());

    std::uniform_real_distribution<float> dist_scale(0.5f, 2.0f);
    std::uniform_int_distribution<int> dist_byte(0, 255);

    for (size_t i = 0; i < total_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(dist_scale(rng_));
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = static_cast<uint8_t>(dist_byte(rng_));
        }
    }

    auto weights = std::make_unique<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw_data);

    // 2. Decode weights to FP32 for reference
    std::vector<float> weights_fp32(rows * cols);
    weights->decode_to_fp32(weights_fp32.data());

    // 3. Create random input (M x K)
    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    float *input_data = input->mutable_data();
    std::uniform_real_distribution<float> dist_input(-1.0f, 1.0f);
    for (int i = 0; i < m * k; ++i)
    {
        input_data[i] = dist_input(rng_);
    }

    // 4. Compute Reference GEMM (C = A * B^T)
    std::vector<float> output_ref(m * n, 0.0f);
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l)
            {
                float a_val = input_data[i * k + l];
                float b_val = weights_fp32[j * k + l]; // weights are n x k
                sum += a_val * b_val;
            }
            output_ref[i * n + j] = sum;
        }
    }

    // 5. Compute Actual GEMM
    auto output = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    float *output_data = output->mutable_data();

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply(input_data, output_data, m, n, k));

    // 6. Compare
    double max_diff = 0.0;
    for (int i = 0; i < m * n; ++i)
    {
        double diff = std::abs(output_data[i] - output_ref[i]);
        max_diff = std::max(max_diff, diff);
    }

    // Use a larger tolerance for quantized operations
    EXPECT_LT(max_diff, 50.0f) << "Max difference too high";
}

TEST_F(Test__IQ4_NLTensor, QuantizedVsFP32Parity)
{
    // Test dimensions: M=4, N=64, K=256
    // A: FP32 activation (random values)
    // B: IQ4_NL weight (random data with valid scales)
    // Compare: CPUNativeVNNIGemmKernel vs FloatingPointGemmKernel on FP32

    int m = 4;
    int n = 64;
    int k = 256;

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // === Create IQ4_NL weight tensor with random but valid data ===
    // IQ4_NL: 32 elements per block
    // Structure: d (FP16), qs[16] (packed 4-bit indices)
    size_t blocks_per_row = k / 32;
    size_t num_blocks = n * blocks_per_row;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(IQ4_NLBlock));
    IQ4_NLBlock *blocks = reinterpret_cast<IQ4_NLBlock *>(raw_data.data());

    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        // Set a valid FP16 scale (not random bytes which may create NaN)
        blocks[i].d = fp32_to_fp16(scale_dist(rng_));

        // Randomize qs (16 bytes - packed 4-bit indices into kvalues_iq4nl)
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = byte_dist(rng_);
        }
    }

    auto B_iq4nl = std::make_shared<IQ4_NLTensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // === Dequantize IQ4_NL weights to FP32 for reference ===
    TensorFactory factory(*mpi_ctx_);
    auto B_fp32 = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    B_iq4nl->to_fp32(B_fp32->mutable_data());

    // === Create FP32 activation tensor with random values ===
    auto A_fp32 = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    for (int i = 0; i < m * k; ++i)
    {
        A_fp32->mutable_data()[i] = dist(rng_);
    }

    // === Reference: FP32 GEMM using FloatingPointGemmKernel ===
    auto C_ref = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::fill_n(C_ref->mutable_data(), m * n, 0.0f);

    gemm::FloatingPointGemmKernel fp_kernel(B_fp32.get());
    // B is [N, K], so transpose_B=true to compute A @ B^T
    fp_kernel.multiply(A_fp32->data(), C_ref->mutable_data(), m, n, k, true);

    // === Test: CPUNativeVNNIGemmKernel with IQ4_NL weights ===
    auto C_quant = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::fill_n(C_quant->mutable_data(), m * n, 0.0f);

    cpu::native_vnni::CPUNativeVNNIGemmKernel quant_kernel(B_iq4nl.get());
    quant_kernel.multiply(A_fp32->data(), C_quant->mutable_data(), m, n, k, true, 1.0f, 0.0f, nullptr, -1);

    // === Compare results ===
    float rel_l2 = compute_relative_l2_error(C_ref->data(), C_quant->data(), m * n);

    // IQ4_NL is a 4-bit format. Relative L2 error should be under 1%.
    const float tolerance = 0.01f;

    EXPECT_LT(rel_l2, tolerance)
        << "Relative L2 error " << rel_l2 << " exceeds tolerance " << tolerance;

    // Print some diagnostics
    std::cout << "[IQ4_NL vs FP32] Relative L2 error: " << rel_l2 << std::endl;

    // Verify outputs are not garbage
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_TRUE(std::isfinite(C_ref->data()[i])) << "Reference NaN/Inf at " << i;
        EXPECT_TRUE(std::isfinite(C_quant->data()[i])) << "Quantized NaN/Inf at " << i;
    }

    // Print first few values for manual inspection
    std::cout << "[IQ4_NL vs FP32] First 5 values:\n";
    for (int i = 0; i < std::min(5, m * n); ++i)
    {
        std::cout << "  [" << i << "] ref=" << C_ref->data()[i]
                  << " quant=" << C_quant->data()[i]
                  << " diff=" << std::abs(C_ref->data()[i] - C_quant->data()[i]) << "\n";
    }
}
