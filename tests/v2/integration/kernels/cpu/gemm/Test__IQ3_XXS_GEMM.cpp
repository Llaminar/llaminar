/**
 * @file Test__IQ3_XXS_GEMM.cpp
 * @brief Integration tests for IQ3_XXS GEMM
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/tensors/TensorFactory.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include "../../../../src/v2/tensors/FP16Utils.h"
#include "../../../../src/v2/kernels/cpu/gemm/FloatingPointGemmKernel.h"
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
#include <vector>
#include <cstring>
#include <cmath>
#include <random>

using namespace llaminar2;

// Custom main to initialize MPI
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

class IQ3_XXS_GEMM : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;

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
};

TEST_F(IQ3_XXS_GEMM, BasicMultiplication)
{
    // M=1, N=32, K=256
    // A: 1x256 (FP32) = 1.0
    // B: 32x256 (IQ3_XXS) = 2.0 (constructed)
    // C: 1x32 (FP32)

    int m = 1;
    int n = 32;
    int k = 256;

    // Create A (activation)
    TensorFactory factory(*mpi_ctx_);
    auto A = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    std::fill_n(A->mutable_data(), m * k, 1.0f);

    // Create B (weight) - IQ3_XXS
    size_t num_blocks = 32; // 32 rows * 1 block/row
    std::vector<uint8_t> raw_data(num_blocks * sizeof(IQ3_XXSBlock));
    IQ3_XXSBlock *blocks = reinterpret_cast<IQ3_XXSBlock *>(raw_data.data());

    for (size_t i = 0; i < num_blocks; ++i)
    {
        std::memset(&blocks[i], 0, sizeof(IQ3_XXSBlock));
        blocks[i].d = fp32_to_fp16(2.0f);
        // qs=0 -> grid=4. aux32=0 -> scale=0.5. signs=0 -> pos.
        // val = 2.0 * 0.5 * 4 * 1 = 2.0.
    }

    auto B = std::make_shared<IQ3_XXSTensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // Create C (output)
    auto C = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});

    // Create GEMM
    auto gemm = B->createGemm();

    // Execute
    gemm->multiply(A->data(), C->mutable_data(), m, n, k);

    // Verify
    // A = 1.0
    // B = 2.0
    // C = 1.0 * 2.0 * 256 = 512.0

    const float *c_data = C->data();
    for (int i = 0; i < n; ++i)
    {
        // Allow for quantization error (Q8_0 quantization of 2.0 is exact)
        EXPECT_NEAR(c_data[i], 512.0f, 1.0f) << "Mismatch at index " << i;
    }
}

TEST_F(IQ3_XXS_GEMM, QuantizedVsFP32Parity)
{
    // Test dimensions: M=4, N=64, K=256
    // A: FP32 activation (random values)
    // B: IQ3_XXS weight (random data with valid scales)
    // Compare: CPUNativeVNNIGemmKernel vs FloatingPointGemmKernel on FP32

    int m = 4;
    int n = 64;
    int k = 256;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // === Create IQ3_XXS weight tensor with random but valid data ===
    // IQ3_XXS: 256 elements per super-block, 98 bytes
    // Structure: d (FP16), qs[96] (grid indices, with scales/signs packed in high bits)
    size_t blocks_per_row = (k + 255) / 256;
    size_t num_blocks = n * blocks_per_row;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(IQ3_XXSBlock));
    IQ3_XXSBlock *blocks = reinterpret_cast<IQ3_XXSBlock *>(raw_data.data());

    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        // Set a valid FP16 scale (not random bytes which may create NaN)
        blocks[i].d = fp32_to_fp16(scale_dist(rng));

        // Randomize qs (96 bytes - contains grid indices with packed scales/signs)
        for (int j = 0; j < 96; ++j)
        {
            blocks[i].qs[j] = byte_dist(rng);
        }
    }

    auto B_iq3xxs = std::make_shared<IQ3_XXSTensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // === Dequantize IQ3_XXS weights to FP32 for reference ===
    TensorFactory factory(*mpi_ctx_);
    auto B_fp32 = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    B_iq3xxs->to_fp32(B_fp32->mutable_data());

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

    // === Test: CPUNativeVNNIGemmKernel with IQ3_XXS weights ===
    auto C_quant = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::fill_n(C_quant->mutable_data(), m * n, 0.0f);

    cpu::native_vnni::CPUNativeVNNIGemmKernel quant_kernel(B_iq3xxs.get());
    quant_kernel.multiply(A_fp32->data(), C_quant->mutable_data(), m, n, k, true, 1.0f, 0.0f, nullptr, -1);

    // === Compare results ===
    float rel_l2 = compute_relative_l2_error(C_ref->data(), C_quant->data(), m * n);

    // IQ3_XXS is a 3-bit format. Relative L2 error should be under 1%.
    const float tolerance = 0.01f;

    EXPECT_LT(rel_l2, tolerance)
        << "Relative L2 error " << rel_l2 << " exceeds tolerance " << tolerance;

    // Print some diagnostics
    std::cout << "[IQ3_XXS vs FP32] Relative L2 error: " << rel_l2 << std::endl;

    // Verify outputs are not garbage
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_TRUE(std::isfinite(C_ref->data()[i])) << "Reference NaN/Inf at " << i;
        EXPECT_TRUE(std::isfinite(C_quant->data()[i])) << "Quantized NaN/Inf at " << i;
    }

    // Print first few values for manual inspection
    std::cout << "[IQ3_XXS vs FP32] First 5 values:\n";
    for (int i = 0; i < std::min(5, m * n); ++i)
    {
        std::cout << "  [" << i << "] ref=" << C_ref->data()[i]
                  << " quant=" << C_quant->data()[i]
                  << " diff=" << std::abs(C_ref->data()[i] - C_quant->data()[i]) << "\n";
    }
}
