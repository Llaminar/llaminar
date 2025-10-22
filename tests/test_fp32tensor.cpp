/**
 * @file test_fp32tensor.cpp
 * @brief Unit test for FP32Tensor and FP32Gemm
 *
 * Verifies that FP32Tensor:
 * 1. Has same interface as SimpleTensor
 * 2. Implements ITensorGemm correctly
 * 3. FP32Gemm produces correct GEMM results
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "tensors/FP32Tensor.h"
#include "Logger.h"
#include <cmath>

using namespace llaminar;

TEST(FP32TensorTest, BasicConstruction)
{
    // Test basic construction
    FP32Tensor tensor({4, 4});

    EXPECT_EQ(tensor.shape().size(), 2);
    EXPECT_EQ(tensor.shape()[0], 4);
    EXPECT_EQ(tensor.shape()[1], 4);
    EXPECT_EQ(tensor.size(), 16);
    EXPECT_EQ(tensor.ndim(), 2);
    EXPECT_EQ(tensor.element_count(), 16);
    EXPECT_FALSE(tensor.is_distributed());
    EXPECT_EQ(tensor.type_name(), "FP32Tensor");
}

TEST(FP32TensorTest, DataAccess)
{
    FP32Tensor tensor({2, 3});

    // Fill with test data
    float *data = tensor.data();
    for (int i = 0; i < 6; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Verify const access
    const FP32Tensor &const_tensor = tensor;
    const float *const_data = const_tensor.data();

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(const_data[i], static_cast<float>(i));
    }

    // Verify native type
    EXPECT_EQ(tensor.native_type(), TensorDataType::FP32);

    // Verify data_fp32() fast path
    EXPECT_EQ(tensor.data_fp32(), data);
}

TEST(FP32TensorTest, FillAndZero)
{
    FP32Tensor tensor({3, 3});

    // Test fill
    tensor.fill(42.0f);
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_FLOAT_EQ(tensor.data()[i], 42.0f);
    }

    // Test zero
    tensor.zero();
    for (int i = 0; i < 9; ++i)
    {
        EXPECT_FLOAT_EQ(tensor.data()[i], 0.0f);
    }
}

TEST(FP32TensorTest, ElementAccess)
{
    FP32Tensor tensor({3, 4});

    // Set via operator()
    tensor(0, 0) = 1.0f;
    tensor(1, 2) = 2.5f;
    tensor(2, 3) = 3.7f;

    // Verify via const operator()
    const FP32Tensor &const_tensor = tensor;
    EXPECT_FLOAT_EQ(const_tensor(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(const_tensor(1, 2), 2.5f);
    EXPECT_FLOAT_EQ(const_tensor(2, 3), 3.7f);
}

TEST(FP32TensorTest, Resize)
{
    FP32Tensor tensor({2, 2});
    EXPECT_EQ(tensor.size(), 4);

    tensor.resize({3, 5});
    EXPECT_EQ(tensor.shape()[0], 3);
    EXPECT_EQ(tensor.shape()[1], 5);
    EXPECT_EQ(tensor.size(), 15);
}

TEST(FP32TensorTest, ITensorGemmInterface)
{
    FP32Tensor weight({4, 3}); // [out_features=4, in_features=3]

    // Fill weight matrix: W^T (we'll use transpose_B=true)
    // W^T = [[1, 0, 0],
    //        [0, 1, 0],
    //        [0, 0, 1],
    //        [1, 1, 1]]
    weight.zero();
    weight(0, 0) = 1.0f;
    weight(1, 1) = 1.0f;
    weight(2, 2) = 1.0f;
    weight(3, 0) = 1.0f;
    weight(3, 1) = 1.0f;
    weight(3, 2) = 1.0f;

    // Create GEMM implementation
    ITensorGemm *gemm = weight.createGemmRaw();
    ASSERT_NE(gemm, nullptr);

    EXPECT_STREQ(gemm->name(), "FP32_BLAS_Gemm");
    EXPECT_TRUE(gemm->supports(2, 4, 3));
    EXPECT_FALSE(gemm->supports_bf16()); // FP32Gemm doesn't support BF16

    // Test GEMM: C = A @ W^T
    // A = [[1, 2, 3],
    //      [4, 5, 6]]  (2x3)
    // W^T as above (4x3)
    // C = A @ W^T (2x4)

    float A[6] = {1, 2, 3, 4, 5, 6};
    float C[8] = {0}; // 2x4 output

    bool success = gemm->multiply(A, C, 2, 4, 3, /*transpose_B=*/true);
    EXPECT_TRUE(success);

    // Expected results:
    // C[0,0] = 1*1 + 2*0 + 3*0 = 1
    // C[0,1] = 1*0 + 2*1 + 3*0 = 2
    // C[0,2] = 1*0 + 2*0 + 3*1 = 3
    // C[0,3] = 1*1 + 2*1 + 3*1 = 6
    // C[1,0] = 4*1 + 5*0 + 6*0 = 4
    // C[1,1] = 4*0 + 5*1 + 6*0 = 5
    // C[1,2] = 4*0 + 5*0 + 6*1 = 6
    // C[1,3] = 4*1 + 5*1 + 6*1 = 15

    EXPECT_FLOAT_EQ(C[0], 1.0f);
    EXPECT_FLOAT_EQ(C[1], 2.0f);
    EXPECT_FLOAT_EQ(C[2], 3.0f);
    EXPECT_FLOAT_EQ(C[3], 6.0f);
    EXPECT_FLOAT_EQ(C[4], 4.0f);
    EXPECT_FLOAT_EQ(C[5], 5.0f);
    EXPECT_FLOAT_EQ(C[6], 6.0f);
    EXPECT_FLOAT_EQ(C[7], 15.0f);

    delete gemm;
}

TEST(FP32TensorTest, GemmWithAlphaBeta)
{
    FP32Tensor weight({2, 2});
    // W^T = [[2, 0],
    //        [0, 3]]
    weight.zero();
    weight(0, 0) = 2.0f;
    weight(1, 1) = 3.0f;

    ITensorGemm *gemm = weight.createGemmRaw();
    ASSERT_NE(gemm, nullptr);

    // A = [[1, 0], [0, 1]]  (identity)
    float A[4] = {1, 0, 0, 1};
    float C[4] = {10, 20, 30, 40}; // Pre-existing values

    // C = 0.5 * A @ W^T + 0.5 * C
    bool success = gemm->multiply(A, C, 2, 2, 2,
                                  /*transpose_B=*/true,
                                  /*alpha=*/0.5f,
                                  /*beta=*/0.5f);
    EXPECT_TRUE(success);

    // Expected:
    // A @ W^T = [[2, 0], [0, 3]]
    // C = 0.5 * [[2, 0], [0, 3]] + 0.5 * [[10, 20], [30, 40]]
    //   = [[1, 0], [0, 1.5]] + [[5, 10], [15, 20]]
    //   = [[6, 10], [15, 21.5]]

    EXPECT_FLOAT_EQ(C[0], 6.0f);
    EXPECT_FLOAT_EQ(C[1], 10.0f);
    EXPECT_FLOAT_EQ(C[2], 15.0f);
    EXPECT_FLOAT_EQ(C[3], 21.5f);

    delete gemm;
}

int main(int argc, char **argv)
{
    Logger::getInstance().setLogLevel(LogLevel::ERROR); // Suppress verbose logs
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
