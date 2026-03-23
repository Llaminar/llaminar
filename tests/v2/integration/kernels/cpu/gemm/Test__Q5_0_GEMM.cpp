/**
 * @file Test__Q5_0_GEMM.cpp
 * @brief Integration tests for Q5_0 GEMM
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

class Q5_0_GEMM : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
};

TEST_F(Q5_0_GEMM, BasicMultiplication)
{
    // M=1, N=32, K=32
    // A: 1x32 (FP32)
    // B: 32x32 (Q5_0) [N, K]
    // C: 1x32 (FP32)

    int m = 1;
    int n = 32;
    int k = 32;

    // Create A (activation)
    TensorFactory factory(*mpi_ctx_);
    auto A = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(k)});
    std::fill_n(A->mutable_data(), m * k, 1.0f);

    // Create B (weight) - Q5_0
    // We need to create raw Q5_0 data
    // Shape [N, K] = [32, 32]
    // Blocks per row = 32/32 = 1
    // Total blocks = 32 rows * 1 block/row = 32 blocks

    size_t num_blocks = 32;
    std::vector<uint8_t> raw_data(num_blocks * sizeof(Q5_0Block));
    Q5_0Block *blocks = reinterpret_cast<Q5_0Block *>(raw_data.data());

    for (size_t i = 0; i < num_blocks; ++i)
    {
        blocks[i].d = fp32_to_fp16(1.0f);
        std::memset(blocks[i].qs, 0, 16); // All 0 -> -16
        std::memset(blocks[i].qh, 0, 4);  // All 0
        // Value = -16.0f
    }

    auto B = std::make_shared<Q5_0Tensor>(std::vector<size_t>{static_cast<size_t>(n), static_cast<size_t>(k)}, raw_data);

    // Create C (output)
    auto C = factory.createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});

    // Create GEMM
    auto gemm = B->createGemm();

    // Execute
    // multiply_tensor(A, C, m, n, k)
    gemm->multiply_tensor(A.get(), C.get(), m, n, k);

    // Verify
    // A = 1.0
    // B = -16.0
    // C = 1.0 * -16.0 * 32 (K) = -512.0

    const float *c_data = C->data();
    for (int i = 0; i < n; ++i)
    {
        // Allow for small quantization error due to Q8_1 activation quantization (fp16 scale)
        // 0.03125 is typical error for 1.0 input with Q8_1
        EXPECT_NEAR(c_data[i], -512.0f, 0.1f) << "Mismatch at index " << i;
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
