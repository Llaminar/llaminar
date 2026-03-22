#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/FP16Utils.h"
#include "v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"

using namespace llaminar2;

class Test__Q2_KTensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Mock MPI context (rank 0, size 1)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
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
};

TEST_F(Test__Q2_KTensor, GemmCorrectness_SingleBlock_Zero)
{
    // M=1, N=1, K=256 (1 super-block)
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    Q2_KBlock *block = reinterpret_cast<Q2_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q2_KBlock));

    // Set scales
    block->d = fp32_to_fp16(1.0f);
    block->dmin = fp32_to_fp16(0.0f);
    // scales[i] = 0 -> dl=0, ml=0.
    // So values will be 0.

    auto weights = std::make_unique<Q2_KTensor>(std::vector<size_t>{1, 256}, raw_data);

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

TEST_F(Test__Q2_KTensor, GemmCorrectness_SingleBlock_Ones)
{
    // M=1, N=1, K=256
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q2_KBlock));
    Q2_KBlock *block = reinterpret_cast<Q2_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q2_KBlock));

    block->d = fp32_to_fp16(1.0f);
    block->dmin = fp32_to_fp16(0.0f);

    // Set scales to 1.0 (dl=1, ml=0)
    // sc = (ml << 4) | dl = 0x01
    for (int i = 0; i < 16; ++i)
        block->scales[i] = 0x01;

    // Set qs to 1 (01 01 01 01 = 0x55)
    for (int i = 0; i < 64; ++i)
        block->qs[i] = 0x55;

    auto weights = std::make_unique<Q2_KTensor>(std::vector<size_t>{1, 256}, raw_data);

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
    // Tolerance: Re-quantization to INT8 introduces error.
    // 256 elements * error per element.
    // Error per element ~ 1/127 * range. Range is small here (0..3).
    // Should be reasonably accurate.
    // L2 error check is better.

    float expected = 256.0f;
    float diff = output_data[0] - expected;
    float l2_error = std::sqrt(diff * diff);

    // With 256 elements, error can accumulate.
    // Let's use a generous tolerance for now, or check L2.
    EXPECT_NEAR(output_data[0], expected, 5.0f) << "Output: " << output_data[0];
}

TEST_F(Test__Q2_KTensor, QuantizedVsFP32Parity)
{
    // Test dimensions: M=4, N=64, K=256
    // A: FP32 activation (random values)
    // B: Q2_K weight (random data with valid scales)
    // Compare: CPUNativeVNNIGemmKernel vs FloatingPointGemmKernel on FP32

    int m = 4;
    int n = 64;
    int k = 256;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // === Create Q2_K weight tensor with random but valid data ===
    // Q2_K: 256 elements per super-block, 84 bytes
    // Structure: scales[16], qs[64], d (FP16), dmin (FP16)
    size_t blocks_per_row = (k + 255) / 256;
    size_t num_blocks = n * blocks_per_row;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q2_KBlock));
    Q2_KBlock *blocks = reinterpret_cast<Q2_KBlock *>(raw_data.data());

    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        // Set valid FP16 scales (not random bytes which may create NaN)
        blocks[i].d = fp32_to_fp16(scale_dist(rng));
        blocks[i].dmin = fp32_to_fp16(scale_dist(rng) * 0.5f);

        // Randomize scales (16 bytes - packed scale and min values)
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].scales[j] = byte_dist(rng);
        }

        // Randomize qs (64 bytes - packed 2-bit values)
        for (int j = 0; j < 64; ++j)
        {
            blocks[i].qs[j] = byte_dist(rng);
        }
    }

    auto B_q2k = std::make_shared<Q2_KTensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // === Dequantize Q2_K weights to FP32 for reference ===
    TensorFactory factory(*mpi_ctx_);
    auto B_fp32 = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    B_q2k->to_fp32(B_fp32->mutable_data());

    // === Create FP32 activation tensor with random values ===
    auto A_fp32 = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    for (int i = 0; i < m * k; ++i)
    {
        A_fp32->mutable_data()[i] = dist(rng);
    }

    // === Reference: FP32 GEMM using FloatingPointGemmKernel ===
    auto C_ref = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::fill_n(C_ref->mutable_data(), m * n, 0.0f);

    gemm::FloatingPointGemmKernel fp_kernel(B_fp32.get());
    // B is [N, K], so transpose_B=true to compute A @ B^T
    fp_kernel.multiply(A_fp32->data(), C_ref->mutable_data(), m, n, k, true);

    // === Test: CPUNativeVNNIGemmKernel with Q2_K weights ===
    auto C_quant = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::fill_n(C_quant->mutable_data(), m * n, 0.0f);

    cpu::native_vnni::CPUNativeVNNIGemmKernel quant_kernel(B_q2k.get());
    quant_kernel.multiply(A_fp32->data(), C_quant->mutable_data(), m, n, k, true, 1.0f, 0.0f, nullptr, -1);

    // === Compare results ===
    float rel_l2 = compute_relative_l2_error(C_ref->data(), C_quant->data(), m * n);

    // Q2_K is a 2-bit format. Relative L2 error should be under 1%.
    const float tolerance = 0.01f;

    EXPECT_LT(rel_l2, tolerance)
        << "Relative L2 error " << rel_l2 << " exceeds tolerance " << tolerance;

    // Print some diagnostics
    std::cout << "[Q2_K vs FP32] Relative L2 error: " << rel_l2 << std::endl;

    // Verify outputs are not garbage
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_TRUE(std::isfinite(C_ref->data()[i])) << "Reference NaN/Inf at " << i;
        EXPECT_TRUE(std::isfinite(C_quant->data()[i])) << "Quantized NaN/Inf at " << i;
    }

    // Print first few values for manual inspection
    std::cout << "[Q2_K vs FP32] First 5 values:\n";
    for (int i = 0; i < std::min(5, m * n); ++i)
    {
        std::cout << "  [" << i << "] ref=" << C_ref->data()[i]
                  << " quant=" << C_quant->data()[i]
                  << " diff=" << std::abs(C_ref->data()[i] - C_quant->data()[i]) << "\n";
    }
}
