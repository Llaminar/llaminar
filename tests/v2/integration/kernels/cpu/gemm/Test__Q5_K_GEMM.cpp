/**
 * @file Test__Q5_K_GEMM.cpp
 * @brief Integration tests for Q5_K GEMM
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "../../../../src/v2/tensors/Tensors.h"
#include "../../../../src/v2/tensors/TensorFactory.h"
#include "../../../../src/v2/utils/MPIContext.h"
#include "../../../../src/v2/tensors/FP16Utils.h"
#include <vector>
#include <random>
#include <cstring>

using namespace llaminar2;

class Q5_K_GEMM : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
};

TEST_F(Q5_K_GEMM, BasicMultiplication)
{
    // M=1, N=32, K=256
    // A: 1x256 (FP32)
    // B: 32x256 (Q5_K) [N, K]
    // C: 1x32 (FP32)

    int m = 1;
    int n = 32;
    int k = 256;

    // Create A (activation)
    TensorFactory factory(*mpi_ctx_);
    auto A = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    std::fill_n(A->mutable_data(), m * k, 1.0f);

    // Create B (weight) - Q5_K
    // Shape [N, K] = [32, 256]
    // Blocks per row = 256/256 = 1
    // Total blocks = 32 rows * 1 block/row = 32 blocks

    size_t num_blocks = 32;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q5_KBlock));
    Q5_KBlock *blocks = reinterpret_cast<Q5_KBlock *>(raw_data.data());

    for (size_t i = 0; i < num_blocks; ++i)
    {
        std::memset(&blocks[i], 0, sizeof(Q5_KBlock));
        blocks[i].d = fp32_to_fp16(1.0f);
        blocks[i].dmin = fp32_to_fp16(0.0f);

        // Set scales to 1.0 (sc=1, d=1.0 -> scale=1.0)
        // Set mins to 0.0 (m=0, dmin=0.0 -> min=0.0)
        // scales[j] & 63 = sc. scales[j+4] & 63 = m.
        // We want sc=1, m=0.
        // scales[0..3] = 1. scales[4..7] = 0.
        // scales[8..11] = 1 (for sub-blocks 4-7, low 4 bits of scales[8..11] contribute to d)

        for (int j = 0; j < 4; ++j)
            blocks[i].scales[j] = 1;
        for (int j = 4; j < 8; ++j)
            blocks[i].scales[j] = 0;
        for (int j = 8; j < 12; ++j)
            blocks[i].scales[j] = 1;

        // Set qs to 1 (low nibble=1, high nibble=1)
        // 0x11 = 17.
        // qh = 0.
        // val = 1.0 * 1 - 0.0 = 1.0.
        std::memset(blocks[i].qs, 0x11, 128);
        std::memset(blocks[i].qh, 0, 32);
    }

    auto B = std::make_shared<Q5_KTensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // Create C (output)
    auto C = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});

    // Create GEMM
    auto gemm = B->createGemm();

    // Execute
    // multiply_tensor(A, C, m, n, k)
    gemm->multiply_tensor(A.get(), C.get(), m, n, k);

    // Verify
    // A = 1.0
    // B = 1.0
    // C = 1.0 * 1.0 * 256 (K) = 256.0

    const float *c_data = C->data();
    for (int i = 0; i < n; ++i)
    {
        // Allow for quantization error
        EXPECT_NEAR(c_data[i], 256.0f, 1.0f) << "Mismatch at index " << i;
    }
}

int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
