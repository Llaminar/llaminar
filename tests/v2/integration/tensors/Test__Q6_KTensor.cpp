#include <gtest/gtest.h>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include "v2/tensors/Tensors.h"
#include "v2/utils/MPIContext.h"
#include "v2/tensors/TensorFactory.h"
#include "v2/tensors/BlockStructures.h"
#include "v2/tensors/FP16Utils.h"

using namespace llaminar2;

class Test__Q6_KTensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Mock MPI context (rank 0, size 1)
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
};

TEST_F(Test__Q6_KTensor, GemmCorrectness_SingleBlock_Zero)
{
    // M=1, N=1, K=256 (1 super-block)
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    Q6_KBlock *block = reinterpret_cast<Q6_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q6_KBlock));

    // Set scales
    block->d = fp32_to_fp16(1.0f);
    // scales[i] = 0.
    // ql, qh = 0 -> q = -32.
    // val = d * scale * q = 1.0 * 0 * -32 = 0.

    auto weights = std::make_unique<Q6_KTensor>(std::vector<size_t>{1, 256}, raw_data);

    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({1, 256});
    float *input_data = input->mutable_data();
    for (int i = 0; i < 256; ++i)
        input_data[i] = 1.0f;

    auto output = factory.createFP32({1, 1});
    float *output_data = output->mutable_data();
    output_data[0] = 123.0f; // Garbage

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply_tensor(input.get(), output.get(), m, n, k));

    EXPECT_NEAR(output_data[0], 0.0f, 1e-5f);
}

TEST_F(Test__Q6_KTensor, GemmCorrectness_SingleBlock_Ones)
{
    // M=1, N=1, K=256
    int m = 1;
    int n = 1;
    int k = 256;

    std::vector<uint8_t> raw_data(sizeof(Q6_KBlock));
    Q6_KBlock *block = reinterpret_cast<Q6_KBlock *>(raw_data.data());
    std::memset(block, 0, sizeof(Q6_KBlock));

    block->d = fp32_to_fp16(1.0f);

    // Set scales to 1
    for (int i = 0; i < 16; ++i)
        block->scales[i] = 1;

    // Set qs to 1.0
    // val = d * scale * q
    // val = 1.0 * 1.0 * q
    // We want val = 1.0, so q = 1.
    // q = raw_q - 32.
    // So raw_q = 33 = 0x21.

    // Packing logic:
    // q1 = (ql[l] & 0xF) | ((qh[l] >> 0) & 3) << 4
    // q2 = (ql[l+32] & 0xF) | ((qh[l] >> 2) & 3) << 4
    // q3 = (ql[l] >> 4) | ((qh[l] >> 4) & 3) << 4
    // q4 = (ql[l+32] >> 4) | ((qh[l] >> 6) & 3) << 4

    // We want q1=q2=q3=q4 = 33 (0x21).
    // Lower 4 bits of 33 is 1 (0001).
    // Upper 2 bits of 33 is 2 (10).

    // ql[l] & 0xF = 1
    // ql[l] >> 4 = 1
    // So ql[l] = 0x11.

    // ql[l+32] & 0xF = 1
    // ql[l+32] >> 4 = 1
    // So ql[l+32] = 0x11.

    // (qh[l] >> 0) & 3 = 2 (10)
    // (qh[l] >> 2) & 3 = 2 (10)
    // (qh[l] >> 4) & 3 = 2 (10)
    // (qh[l] >> 6) & 3 = 2 (10)
    // So qh[l] = 0b10101010 = 0xAA.

    for (int i = 0; i < 128; ++i)
        block->ql[i] = 0x11;
    for (int i = 0; i < 64; ++i)
        block->qh[i] = 0xAA;

    auto weights = std::make_unique<Q6_KTensor>(std::vector<size_t>{1, 256}, raw_data);

    TensorFactory factory(*mpi_ctx_);
    auto input = factory.createFP32({1, 256});
    float *input_data = input->mutable_data();
    for (int i = 0; i < 256; ++i)
        input_data[i] = 1.0f;

    auto output = factory.createFP32({1, 1});
    float *output_data = output->mutable_data();
    output_data[0] = 0.0f;

    auto gemm = weights->createGemm();
    ASSERT_TRUE(gemm->multiply_tensor(input.get(), output.get(), m, n, k));

    // Expected: 256 elements * 1.0 * 1.0 = 256.0
    float expected = 256.0f;
    EXPECT_NEAR(output_data[0], expected, 5.0f) << "Output: " << output_data[0];
}
