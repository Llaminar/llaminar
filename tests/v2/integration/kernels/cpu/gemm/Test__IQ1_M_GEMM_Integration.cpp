/**
 * @file Test__IQ1_M_GEMM_Integration.cpp
 * @brief Integration tests for IQ1_M GEMM
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/tensors/TensorFactory.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include "../../../../src/v2/tensors/FP16Utils.h"
#include "../../../../src/v2/kernels/cpu/gemm_v4/FloatingPointGemmKernel.h"
#include "../../../../src/v2/kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
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

class IQ1_M_GEMM_Integration : public ::testing::Test
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

TEST_F(IQ1_M_GEMM_Integration, BasicMultiplication)
{
    // M=1, N=32, K=256
    // A: 1x256 (FP32) = 1.0
    // B: 32x256 (IQ1_M) = 1.0 (constructed)
    // C: 1x32 (FP32)

    int m = 1;
    int n = 32;
    int k = 256;

    // Create A (activation)
    TensorFactory factory(*mpi_ctx_);
    auto A = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    std::fill_n(A->mutable_data(), m * k, 1.0f);

    // Create B (weight) - IQ1_M
    // IQ1_M block size is 256. So 1 block per row for K=256.
    size_t num_blocks = n * (k / 256); // 32 * 1 = 32 blocks
    std::vector<uint8_t> raw_data(num_blocks * sizeof(IQ1_MBlock));
    IQ1_MBlock *blocks = reinterpret_cast<IQ1_MBlock *>(raw_data.data());

    for (size_t i = 0; i < num_blocks; ++i)
    {
        // Fill with non-zero pattern to ensure non-zero output
        std::memset(&blocks[i], 0x11, sizeof(IQ1_MBlock));
    }

    auto B = std::make_shared<IQ1_MTensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // Create C (output)
    auto C = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});

    // Create GEMM
    auto gemm = B->createGemm();

    // Execute
    gemm->multiply(A->data(), C->mutable_data(), m, n, k);

    // Verify
    const float *c_data = C->data();
    for (int i = 0; i < n; ++i)
    {
        EXPECT_NE(c_data[i], 0.0f) << "Output should not be zero at index " << i;
        EXPECT_TRUE(std::isfinite(c_data[i]));
    }
}

TEST_F(IQ1_M_GEMM_Integration, QuantizedVsFP32Parity)
{
    // Test dimensions: M=4, N=64, K=256
    // A: Q8_1 activation (will be quantized from FP32)
    // B: IQ1_M weight
    // Compare: QuantisedGemmKernel vs FloatingPointGemmKernel on FP32

    int m = 4;
    int n = 64;
    int k = 256;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // === Create IQ1_M weight tensor with random but valid data ===
    size_t blocks_per_row = (k + 255) / 256;
    size_t num_blocks = n * blocks_per_row;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(IQ1_MBlock));
    IQ1_MBlock *blocks = reinterpret_cast<IQ1_MBlock *>(raw_data.data());

    std::uniform_int_distribution<uint8_t> byte_dist(0, 255);
    std::uniform_real_distribution<float> scale_dist(0.01f, 1.0f);

    for (size_t i = 0; i < num_blocks; ++i)
    {
        // Randomize qs (grid indices low 8 bits)
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qs[j] = byte_dist(rng);
        }

        // Randomize qh (grid index high bits + delta signs)
        for (int j = 0; j < 16; ++j)
        {
            blocks[i].qh[j] = byte_dist(rng);
        }

        // Set valid packed scales (8 bytes encode the global FP16 scale + sub-scales)
        // To avoid NaN, we ensure the packed scale bits form a valid FP16
        uint16_t valid_fp16_scale = fp32_to_fp16(scale_dist(rng));
        // Embed the global scale in the lower bits of scales, following IQ1_M format
        // The global scale is extracted via extract_iq1m_global_scale which reads
        // bits from sc[0]>>12 | (sc[1]>>8 & 0xf0) | (sc[2]>>4 & 0x0f00) | (sc[3] & 0xf000)
        // For simplicity, just set random sub-scales but ensure we have a reasonable global
        uint16_t *sc = reinterpret_cast<uint16_t *>(blocks[i].scales);
        for (int j = 0; j < 4; ++j)
        {
            sc[j] = byte_dist(rng) | (byte_dist(rng) << 8);
        }
        // Patch in a valid global scale in the expected bit positions
        sc[0] = (sc[0] & 0x0FFF) | ((valid_fp16_scale & 0x000F) << 12);
        sc[1] = (sc[1] & 0x00FF) | ((valid_fp16_scale & 0x00F0) << 4);
        sc[2] = (sc[2] & 0x000F) | ((valid_fp16_scale & 0x0F00) >> 4);
        sc[3] = (sc[3] & 0x0FFF) | (valid_fp16_scale & 0xF000);
    }

    auto B_iq1m = std::make_shared<IQ1_MTensor>(
        std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // === Dequantize IQ1_M weights to FP32 for reference ===
    TensorFactory factory(*mpi_ctx_);
    auto B_fp32 = factory.createFP32({static_cast<size_t>(n), static_cast<size_t>(k)});
    B_iq1m->to_fp32(B_fp32->mutable_data());

    // === Create FP32 activation tensor with random values ===
    auto A_fp32 = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    for (int i = 0; i < m * k; ++i)
    {
        A_fp32->mutable_data()[i] = dist(rng);
    }

    // === Reference: FP32 GEMM using FloatingPointGemmKernel ===
    auto C_ref = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::fill_n(C_ref->mutable_data(), m * n, 0.0f);

    gemm_v4::FloatingPointGemmKernel fp_kernel(B_fp32.get());
    // B is [N, K], so transpose_B=true to compute A @ B^T
    fp_kernel.multiply(A_fp32->data(), C_ref->mutable_data(), m, n, k, true);

    // === Test: QuantisedGemmKernel with IQ1_M weights ===
    auto C_quant = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});
    std::fill_n(C_quant->mutable_data(), m * n, 0.0f);

    gemm_v4::QuantisedGemmKernel quant_kernel(B_iq1m.get());
    quant_kernel.multiply(A_fp32->data(), C_quant->mutable_data(), m, n, k, true, 1.0f, 0.0f, nullptr, -1);

    // === Compare results ===
    float rel_l2 = compute_relative_l2_error(C_ref->data(), C_quant->data(), m * n);

    // IQ1_M is a 1-bit format. Relative L2 error should be under 1%.
    const float tolerance = 0.01f;

    EXPECT_LT(rel_l2, tolerance)
        << "Relative L2 error " << rel_l2 << " exceeds tolerance " << tolerance;

    // Print some diagnostics
    std::cout << "[IQ1_M vs FP32] Relative L2 error: " << rel_l2 << std::endl;

    // Verify outputs are not garbage
    for (int i = 0; i < m * n; ++i)
    {
        EXPECT_TRUE(std::isfinite(C_ref->data()[i])) << "Reference NaN/Inf at " << i;
        EXPECT_TRUE(std::isfinite(C_quant->data()[i])) << "Quantized NaN/Inf at " << i;
    }

    // Print first few values for manual inspection
    std::cout << "[IQ1_M vs FP32] First 5 values:\n";
    for (int i = 0; i < std::min(5, m * n); ++i)
    {
        std::cout << "  [" << i << "] ref=" << C_ref->data()[i]
                  << " quant=" << C_quant->data()[i]
                  << " diff=" << std::abs(C_ref->data()[i] - C_quant->data()[i]) << "\n";
    }
}
