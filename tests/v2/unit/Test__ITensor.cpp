/**
 * @file Test__ITensor.cpp
 * @brief Unit tests for ITensor runtime interface
 * @author David Sanftenberg
 *
 * Tests the ITensor interface including:
 * - native_type_id() correctness for all tensor types
 * - is<T>() type checking
 * - typed_as<T>() safe downcasting
 * - try_as<T>() optional downcasting
 * - static_type_id() consistency with TensorTypeId constants
 */

#include <gtest/gtest.h>
#include <memory>
#include "v2/tensors/Tensors.h"
#include "v2/tensors/ITensor.h"

using namespace llaminar2;

class Test__ITensor : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test tensors of various types
        fp32_tensor_ = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 4});
        bf16_tensor_ = std::make_shared<BF16Tensor>(std::vector<size_t>{4, 4});
        fp16_tensor_ = std::make_shared<FP16Tensor>(std::vector<size_t>{4, 4});
        int8_tensor_ = std::make_shared<INT8Tensor>(std::vector<size_t>{4, 4});
        int32_tensor_ = std::make_shared<INT32Tensor>(std::vector<size_t>{4, 4});

        // Create quantized tensors with raw data
        std::vector<uint8_t> iq4_nl_data(sizeof(IQ4_NLBlock) * 4); // 4 blocks for 4 rows
        iq4_nl_tensor_ = std::make_shared<IQ4_NLTensor>(
            std::vector<size_t>{4, 32}, iq4_nl_data);

        std::vector<uint8_t> q8_0_data(sizeof(Q8_0Block) * 4);
        q8_0_tensor_ = std::make_shared<Q8_0Tensor>(
            std::vector<size_t>{4, 32}, q8_0_data);

        std::vector<uint8_t> q4_0_data(sizeof(Q4_0Block) * 4);
        q4_0_tensor_ = std::make_shared<Q4_0Tensor>(
            std::vector<size_t>{4, 32}, q4_0_data);
    }

    std::shared_ptr<FP32Tensor> fp32_tensor_;
    std::shared_ptr<BF16Tensor> bf16_tensor_;
    std::shared_ptr<FP16Tensor> fp16_tensor_;
    std::shared_ptr<INT8Tensor> int8_tensor_;
    std::shared_ptr<INT32Tensor> int32_tensor_;
    std::shared_ptr<IQ4_NLTensor> iq4_nl_tensor_;
    std::shared_ptr<Q8_0Tensor> q8_0_tensor_;
    std::shared_ptr<Q4_0Tensor> q4_0_tensor_;
};

// ============================================================================
// Test native_type_id() matches TensorTypeId constants
// ============================================================================

TEST_F(Test__ITensor, NativeTypeId_FP32)
{
    EXPECT_EQ(fp32_tensor_->native_type_id(), TensorTypeId::FP32);
    EXPECT_EQ(FP32Tensor::static_type_id(), TensorTypeId::FP32);
}

TEST_F(Test__ITensor, NativeTypeId_BF16)
{
    EXPECT_EQ(bf16_tensor_->native_type_id(), TensorTypeId::BF16);
    EXPECT_EQ(BF16Tensor::static_type_id(), TensorTypeId::BF16);
}

TEST_F(Test__ITensor, NativeTypeId_FP16)
{
    EXPECT_EQ(fp16_tensor_->native_type_id(), TensorTypeId::FP16);
    EXPECT_EQ(FP16Tensor::static_type_id(), TensorTypeId::FP16);
}

TEST_F(Test__ITensor, NativeTypeId_INT8)
{
    EXPECT_EQ(int8_tensor_->native_type_id(), TensorTypeId::INT8);
    EXPECT_EQ(INT8Tensor::static_type_id(), TensorTypeId::INT8);
}

TEST_F(Test__ITensor, NativeTypeId_INT32)
{
    EXPECT_EQ(int32_tensor_->native_type_id(), TensorTypeId::INT32);
    EXPECT_EQ(INT32Tensor::static_type_id(), TensorTypeId::INT32);
}

TEST_F(Test__ITensor, NativeTypeId_IQ4_NL)
{
    EXPECT_EQ(iq4_nl_tensor_->native_type_id(), TensorTypeId::IQ4_NL);
    EXPECT_EQ(IQ4_NLTensor::static_type_id(), TensorTypeId::IQ4_NL);
}

TEST_F(Test__ITensor, NativeTypeId_Q8_0)
{
    EXPECT_EQ(q8_0_tensor_->native_type_id(), TensorTypeId::Q8_0);
    EXPECT_EQ(Q8_0Tensor::static_type_id(), TensorTypeId::Q8_0);
}

TEST_F(Test__ITensor, NativeTypeId_Q4_0)
{
    EXPECT_EQ(q4_0_tensor_->native_type_id(), TensorTypeId::Q4_0);
    EXPECT_EQ(Q4_0Tensor::static_type_id(), TensorTypeId::Q4_0);
}

// ============================================================================
// Test is<T>() type checking
// ============================================================================

TEST_F(Test__ITensor, Is_CorrectType)
{
    const ITensor *fp32_itensor = fp32_tensor_.get();
    const ITensor *bf16_itensor = bf16_tensor_.get();
    const ITensor *iq4_nl_itensor = iq4_nl_tensor_.get();

    EXPECT_TRUE(fp32_itensor->is<FP32Tensor>());
    EXPECT_TRUE(bf16_itensor->is<BF16Tensor>());
    EXPECT_TRUE(iq4_nl_itensor->is<IQ4_NLTensor>());
}

TEST_F(Test__ITensor, Is_WrongType)
{
    const ITensor *fp32_itensor = fp32_tensor_.get();

    EXPECT_FALSE(fp32_itensor->is<BF16Tensor>());
    EXPECT_FALSE(fp32_itensor->is<IQ4_NLTensor>());
    EXPECT_FALSE(fp32_itensor->is<Q8_0Tensor>());
}

// ============================================================================
// Test typed_as<T>() safe downcasting
// ============================================================================

TEST_F(Test__ITensor, TypedAs_CorrectType)
{
    ITensor *fp32_itensor = fp32_tensor_.get();
    ITensor *iq4_nl_itensor = iq4_nl_tensor_.get();

    // Should succeed and return valid reference
    FP32Tensor &fp32_result = fp32_itensor->typed_as<FP32Tensor>();
    EXPECT_EQ(&fp32_result, fp32_tensor_.get());

    IQ4_NLTensor &iq4_nl_result = iq4_nl_itensor->typed_as<IQ4_NLTensor>();
    EXPECT_EQ(&iq4_nl_result, iq4_nl_tensor_.get());
}

TEST_F(Test__ITensor, TypedAs_Const_CorrectType)
{
    const ITensor *fp32_itensor = fp32_tensor_.get();

    // Const version should also work
    const FP32Tensor &fp32_result = fp32_itensor->typed_as<FP32Tensor>();
    EXPECT_EQ(&fp32_result, fp32_tensor_.get());
}

// ============================================================================
// Test try_as<T>() safe downcasting (returns pointer)
// ============================================================================

TEST_F(Test__ITensor, TryAs_CorrectType)
{
    ITensor *fp32_itensor = fp32_tensor_.get();

    FP32Tensor *result = fp32_itensor->try_as<FP32Tensor>();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result, fp32_tensor_.get());
}

TEST_F(Test__ITensor, TryAs_WrongType)
{
    ITensor *fp32_itensor = fp32_tensor_.get();

    BF16Tensor *result = fp32_itensor->try_as<BF16Tensor>();
    EXPECT_EQ(result, nullptr);
}

TEST_F(Test__ITensor, TryAs_Const_CorrectType)
{
    const ITensor *fp32_itensor = fp32_tensor_.get();

    const FP32Tensor *result = fp32_itensor->try_as<FP32Tensor>();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result, fp32_tensor_.get());
}

// ============================================================================
// Test raw_data() / raw_mutable_data()
// ============================================================================

TEST_F(Test__ITensor, RawData_NotNull)
{
    EXPECT_NE(fp32_tensor_->raw_data(), nullptr);
    EXPECT_NE(bf16_tensor_->raw_data(), nullptr);
    EXPECT_NE(int8_tensor_->raw_data(), nullptr);
}

TEST_F(Test__ITensor, RawMutableData_NotNull)
{
    EXPECT_NE(fp32_tensor_->raw_mutable_data(), nullptr);
    EXPECT_NE(bf16_tensor_->raw_mutable_data(), nullptr);
    EXPECT_NE(int8_tensor_->raw_mutable_data(), nullptr);
}

// ============================================================================
// Test numel() and size_bytes()
// ============================================================================

TEST_F(Test__ITensor, Numel_Correct)
{
    EXPECT_EQ(fp32_tensor_->numel(), 16); // 4 * 4
    EXPECT_EQ(bf16_tensor_->numel(), 16);
    EXPECT_EQ(int8_tensor_->numel(), 16);
}

TEST_F(Test__ITensor, SizeBytes_Correct)
{
    // FP32: 16 elements * 4 bytes = 64 bytes
    EXPECT_EQ(fp32_tensor_->size_bytes(), 64);

    // BF16: 16 elements * 2 bytes = 32 bytes
    EXPECT_EQ(bf16_tensor_->size_bytes(), 32);

    // INT8: 16 elements * 1 byte = 16 bytes
    EXPECT_EQ(int8_tensor_->size_bytes(), 16);

    // INT32: 16 elements * 4 bytes = 64 bytes
    EXPECT_EQ(int32_tensor_->size_bytes(), 64);
}

// ============================================================================
// Test value_type typedef
// ============================================================================

TEST_F(Test__ITensor, ValueType_Float)
{
    static_assert(std::is_same_v<FP32Tensor::value_type, float>,
                  "FP32Tensor::value_type should be float");
}

TEST_F(Test__ITensor, ValueType_BF16)
{
    static_assert(std::is_same_v<BF16Tensor::value_type, uint16_t>,
                  "BF16Tensor::value_type should be uint16_t");
}

TEST_F(Test__ITensor, ValueType_FP16)
{
    static_assert(std::is_same_v<FP16Tensor::value_type, uint16_t>,
                  "FP16Tensor::value_type should be uint16_t");
}

TEST_F(Test__ITensor, ValueType_INT8)
{
    static_assert(std::is_same_v<INT8Tensor::value_type, int8_t>,
                  "INT8Tensor::value_type should be int8_t");
}

TEST_F(Test__ITensor, ValueType_INT32)
{
    static_assert(std::is_same_v<INT32Tensor::value_type, int32_t>,
                  "INT32Tensor::value_type should be int32_t");
}

TEST_F(Test__ITensor, ValueType_IQ4_NL)
{
    static_assert(std::is_same_v<IQ4_NLTensor::value_type, IQ4_NLBlock>,
                  "IQ4_NLTensor::value_type should be IQ4_NLBlock");
}

TEST_F(Test__ITensor, ValueType_Q8_0)
{
    static_assert(std::is_same_v<Q8_0Tensor::value_type, Q8_0Block>,
                  "Q8_0Tensor::value_type should be Q8_0Block");
}

TEST_F(Test__ITensor, ValueType_Q4_0)
{
    static_assert(std::is_same_v<Q4_0Tensor::value_type, Q4_0Block>,
                  "Q4_0Tensor::value_type should be Q4_0Block");
}

// ============================================================================
// Test static_type_id() constexpr
// ============================================================================

TEST_F(Test__ITensor, StaticTypeId_Constexpr)
{
    // Verify static_type_id() can be used at compile time
    constexpr int fp32_id = FP32Tensor::static_type_id();
    constexpr int bf16_id = BF16Tensor::static_type_id();
    constexpr int iq4_nl_id = IQ4_NLTensor::static_type_id();

    EXPECT_EQ(fp32_id, TensorTypeId::FP32);
    EXPECT_EQ(bf16_id, TensorTypeId::BF16);
    EXPECT_EQ(iq4_nl_id, TensorTypeId::IQ4_NL);
}

// ============================================================================
// Test polymorphic usage through ITensor pointer
// ============================================================================

TEST_F(Test__ITensor, PolymorphicUsage)
{
    // Store different tensor types as ITensor pointers
    std::vector<ITensor *> tensors = {
        fp32_tensor_.get(),
        bf16_tensor_.get(),
        iq4_nl_tensor_.get(),
        q8_0_tensor_.get()};

    // Verify we can identify types at runtime
    EXPECT_EQ(tensors[0]->native_type_id(), TensorTypeId::FP32);
    EXPECT_EQ(tensors[1]->native_type_id(), TensorTypeId::BF16);
    EXPECT_EQ(tensors[2]->native_type_id(), TensorTypeId::IQ4_NL);
    EXPECT_EQ(tensors[3]->native_type_id(), TensorTypeId::Q8_0);

    // Verify is<T>() works through base pointer
    EXPECT_TRUE(tensors[0]->is<FP32Tensor>());
    EXPECT_FALSE(tensors[0]->is<BF16Tensor>());
}
