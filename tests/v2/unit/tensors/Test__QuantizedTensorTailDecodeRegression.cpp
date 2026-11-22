/**
 * @file Test__QuantizedTensorTailDecodeRegression.cpp
 * @brief Regression tests for partial-tail dequantization in quantized tensor data() paths
 *
 * Covers quantized tensor families where data() previously decoded full 256-element blocks
 * into partial row tails, causing out-of-bounds writes/corruption for K not divisible by 256.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    template <typename TensorT>
    void verifyTailSafeDequantization(const std::string &tensor_name, std::unique_ptr<TensorT> tensor)
    {
        ASSERT_NE(tensor, nullptr);

        const auto &shape = tensor->shape();
        ASSERT_EQ(shape.size(), 2u);
        const size_t rows = shape[0];
        const size_t cols = shape[1];
        ASSERT_GT(rows, 0u);
        ASSERT_GT(cols, 0u);

        const float *dense = tensor->data();
        ASSERT_NE(dense, nullptr) << tensor_name;

        std::vector<float> row_buffer(cols, 0.0f);
        for (size_t row = 0; row < rows; ++row)
        {
            tensor->to_fp32_row(row, row_buffer.data());
            for (size_t col = 0; col < cols; ++col)
            {
                const float from_data = dense[row * cols + col];
                const float from_row_decode = row_buffer[col];
                EXPECT_NEAR(from_data, from_row_decode, 1e-5f)
                    << "Tensor=" << tensor_name << " row=" << row << " col=" << col;
                EXPECT_TRUE(std::isfinite(from_data))
                    << "Tensor=" << tensor_name << " row=" << row << " col=" << col;
            }
        }
    }
}

class Test__QuantizedTensorTailDecodeRegression : public ::testing::Test
{
protected:
    static constexpr size_t kRows = 13;
    static constexpr size_t kPartialCols = 17;
};

TEST_F(Test__QuantizedTensorTailDecodeRegression, Q6_K_K17Tail)
{
    auto tensor = TestTensorFactory::createQ6_KRandom({kRows, kPartialCols}, 1001);
    verifyTailSafeDequantization("Q6_K", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, Q4_K_K17Tail)
{
    auto tensor = TestTensorFactory::createQ4_KRandom({kRows, kPartialCols}, 1011);
    verifyTailSafeDequantization("Q4_K", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, Q5_K_K17Tail)
{
    auto tensor = TestTensorFactory::createQ5_KRandom({kRows, kPartialCols}, 1012);
    verifyTailSafeDequantization("Q5_K", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ4_XS_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ4_XSRandom({kRows, kPartialCols}, 1013);
    verifyTailSafeDequantization("IQ4_XS", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, Q3_K_K17Tail)
{
    auto tensor = TestTensorFactory::createQ3_KRandom({kRows, kPartialCols}, 1002);
    verifyTailSafeDequantization("Q3_K", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, Q2_K_K17Tail)
{
    auto tensor = TestTensorFactory::createQ2_KRandom({kRows, kPartialCols}, 1003);
    verifyTailSafeDequantization("Q2_K", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ3_XXS_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ3_XXSRandom({kRows, kPartialCols}, 1004);
    verifyTailSafeDequantization("IQ3_XXS", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ3_S_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ3_SRandom({kRows, kPartialCols}, 1005);
    verifyTailSafeDequantization("IQ3_S", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ2_XXS_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ2_XXSRandom({kRows, kPartialCols}, 1006);
    verifyTailSafeDequantization("IQ2_XXS", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ2_XS_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ2_XSRandom({kRows, kPartialCols}, 1007);
    verifyTailSafeDequantization("IQ2_XS", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ2_S_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ2_SRandom({kRows, kPartialCols}, 1008);
    verifyTailSafeDequantization("IQ2_S", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ1_S_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ1_SRandom({kRows, kPartialCols}, 1009);
    verifyTailSafeDequantization("IQ1_S", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, IQ1_M_K17Tail)
{
    auto tensor = TestTensorFactory::createIQ1_MRandom({kRows, kPartialCols}, 1010);
    verifyTailSafeDequantization("IQ1_M", std::move(tensor));
}

TEST_F(Test__QuantizedTensorTailDecodeRegression, MXFP4_K17Tail)
{
    auto tensor = TestTensorFactory::createMXFP4Random({kRows, kPartialCols}, 1011);
    verifyTailSafeDequantization("MXFP4", std::move(tensor));
}
