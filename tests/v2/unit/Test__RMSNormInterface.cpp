/**
 * @file Test__RMSNormInterface.cpp
 * @brief Unit tests demonstrating clean ITensorRMSNorm interface
 *
 * This test demonstrates the improved RMSNorm interface design where:
 * - Each precision (FP32/BF16/FP16/INT32→INT8) has its own method
 * - No need to cast to CPURMSNormKernel - use the interface directly
 * - Type-safe: Wrong pointer types won't compile
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include "v2/kernels/cpu/ops/CPURMSNormKernelT.h"
#include "v2/tensors/Tensors.h"
#include <vector>
#include <cmath>
#include <random>

using namespace llaminar2;

class Test__RMSNormInterface : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create kernel through interface
        kernel_ = std::make_unique<CPURMSNormKernelT<FP32Tensor>>();
    }

    std::unique_ptr<ITensorRMSNorm> kernel_;
};

/**
 * @brief Test FP32 RMSNorm through interface
 */
TEST_F(Test__RMSNormInterface, FP32_ThroughInterface)
{
    const int seq_len = 4;
    const int d_model = 8;

    // Allocate buffers
    std::vector<float> input(seq_len * d_model, 1.0f);
    std::vector<float> gamma(d_model, 1.0f);
    std::vector<float> output(seq_len * d_model, 0.0f);

    // Call through interface - clean and type-safe
    bool success = kernel_->apply(
        input.data(),
        gamma.data(),
        output.data(),
        seq_len, d_model);

    EXPECT_TRUE(success);
    // Output should be normalized (approximately 1.0 since input is all 1s)
    for (size_t i = 0; i < output.size(); ++i)
    {
        EXPECT_NEAR(output[i], 1.0f, 1e-5f);
    }
}

/**
 * @brief Test BF16 RMSNorm through interface
 */
TEST_F(Test__RMSNormInterface, BF16_ThroughInterface)
{
    // Use BF16 kernel
    kernel_ = std::make_unique<CPURMSNormKernelT<BF16Tensor>>();

    const int seq_len = 4;
    const int d_model = 8;

    // Allocate BF16 buffers (stored as uint16_t)
    std::vector<uint16_t> input_bf16(seq_len * d_model);
    std::vector<float> gamma(d_model, 1.0f);
    std::vector<uint16_t> output_bf16(seq_len * d_model, 0);

    // Initialize input to BF16(1.0)
    uint16_t bf16_one = 0x3F80; // BF16 representation of 1.0
    std::fill(input_bf16.begin(), input_bf16.end(), bf16_one);

    // Call through interface - NO CAST NEEDED!
    bool success = kernel_->apply_bf16(
        input_bf16.data(),
        gamma.data(),
        output_bf16.data(),
        seq_len, d_model);

    EXPECT_TRUE(success);
    // Verify output is approximately BF16(1.0)
    for (size_t i = 0; i < output_bf16.size(); ++i)
    {
        // Allow some tolerance for BF16 rounding
        EXPECT_NEAR(output_bf16[i], bf16_one, 2); // Within 2 BF16 ULPs
    }
}

/**
 * @brief Test FP16 RMSNorm through interface
 */
TEST_F(Test__RMSNormInterface, FP16_ThroughInterface)
{
    // Use FP16 kernel
    kernel_ = std::make_unique<CPURMSNormKernelT<FP16Tensor>>();

    const int seq_len = 4;
    const int d_model = 8;

    // Allocate FP16 buffers (stored as uint16_t)
    std::vector<uint16_t> input_fp16(seq_len * d_model);
    std::vector<float> gamma(d_model, 1.0f);
    std::vector<uint16_t> output_fp16(seq_len * d_model, 0);

    // Initialize input to FP16(1.0)
    uint16_t fp16_one = 0x3C00; // FP16 representation of 1.0
    std::fill(input_fp16.begin(), input_fp16.end(), fp16_one);

    // Call through interface - NO CAST NEEDED!
    bool success = kernel_->apply_fp16(
        input_fp16.data(),
        gamma.data(),
        output_fp16.data(),
        seq_len, d_model);

    EXPECT_TRUE(success);
    // Verify output is approximately FP16(1.0)
    for (size_t i = 0; i < output_fp16.size(); ++i)
    {
        EXPECT_NEAR(output_fp16[i], fp16_one, 2); // Within 2 FP16 ULPs
    }
}

/**
 * @brief Test INT32→INT8 RMSNorm through interface
 */
TEST_F(Test__RMSNormInterface, INT32ToINT8_ThroughInterface)
{
    // Use INT32 kernel
    kernel_ = std::make_unique<CPURMSNormKernelT<INT32Tensor>>();

    const int seq_len = 4;
    const int d_model = 8;

    // Allocate INT32 input (typical accumulator values)
    std::vector<int32_t> input_int32(seq_len * d_model);
    std::vector<float> gamma(d_model, 1.0f);
    std::vector<int8_t> output_int8(seq_len * d_model, 0);
    std::vector<float> row_scales(seq_len, 0.0f);

    // Initialize with random INT32 values
    std::mt19937 gen(42);
    std::uniform_int_distribution<int32_t> dist(-10000, 10000);
    for (auto &val : input_int32)
    {
        val = dist(gen);
    }

    // Call through interface - NO CAST NEEDED!
    bool success = kernel_->apply_int32_to_int8(
        input_int32.data(),
        gamma.data(),
        output_int8.data(),
        row_scales.data(),
        seq_len, d_model);

    EXPECT_TRUE(success);

    // Verify scales are positive
    for (size_t r = 0; r < seq_len; ++r)
    {
        EXPECT_GT(row_scales[r], 0.0f) << "Row " << r << " scale should be positive";
    }

    // Verify INT8 values are in valid range
    for (size_t i = 0; i < output_int8.size(); ++i)
    {
        EXPECT_GE(output_int8[i], -127);
        EXPECT_LE(output_int8[i], 127);
    }
}

/**
 * @brief Test that default implementations return false
 *
 * This demonstrates that if a kernel doesn't implement a precision,
 * the default implementation safely returns false.
 */
TEST_F(Test__RMSNormInterface, DefaultImplementationsReturnFalse)
{
    // Create a minimal kernel that only implements FP32
    class MinimalRMSNormKernel : public ITensorRMSNorm
    {
    public:
        bool supports_device(int device_idx) const override { return device_idx == -1; }

        bool apply(const float *, const float *, float *,
                   int, int, float, bool, const MPIContext *, int) override
        {
            return true; // FP32 supported
        }
        // BF16, FP16, INT32→INT8 use default implementations (return false)
    };

    MinimalRMSNormKernel minimal_kernel;
    std::vector<uint16_t> bf16_dummy(8);
    std::vector<float> gamma_dummy(8, 1.0f);

    // BF16 should fail (not implemented)
    bool result = minimal_kernel.apply_bf16(
        bf16_dummy.data(),
        gamma_dummy.data(),
        bf16_dummy.data(),
        1, 8);

    EXPECT_FALSE(result) << "Default BF16 implementation should return false";
}

/**
 * @brief Demonstrate clean usage in pipeline-like code
 */
TEST_F(Test__RMSNormInterface, PipelineLikeUsage)
{
    // Create activation tensors of different types
    auto fp32_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 8});
    auto bf16_tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{4, 8});

    // Get kernels from activation tensors (factory pattern)
    auto fp32_kernel = fp32_tensor->createRMSNorm();
    auto bf16_kernel = bf16_tensor->createRMSNorm();

    EXPECT_NE(fp32_kernel, nullptr);
    EXPECT_NE(bf16_kernel, nullptr);

    // Both return the same kernel type, but the tensor knows how to use it
    // This is the V2 pattern: Tensors create kernels, pipelines dispatch based on tensor type

    std::vector<float> gamma(8, 1.0f);

    // FP32 path
    bool fp32_success = fp32_kernel->apply(
        fp32_tensor->data(),
        gamma.data(),
        fp32_tensor->mutable_data(),
        4, 8);
    EXPECT_TRUE(fp32_success);

    // BF16 path - uses native precision method
    bool bf16_success = bf16_kernel->apply_bf16(
        bf16_tensor->bf16_data(),
        gamma.data(),
        bf16_tensor->mutable_bf16_data(),
        4, 8);
    EXPECT_TRUE(bf16_success);
}

/**
 * @brief NEW: Test FP32Tensor::applyRMSNorm() - clean interface method
 */
TEST_F(Test__RMSNormInterface, FP32_TensorMethod)
{
    const int seq_len = 4;
    const int d_model = 8;

    // Create FP32 tensor and initialize to 1.0
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{seq_len, d_model});
    std::fill_n(tensor->mutable_data(), seq_len * d_model, 1.0f);

    // Create gamma weights
    std::vector<float> gamma(d_model, 1.0f);

    // Call applyRMSNorm directly on tensor - clean!
    bool success = tensor->applyRMSNorm(gamma.data(), seq_len, d_model);

    EXPECT_TRUE(success);

    // Verify normalization (input all 1s → output all 1s)
    const float *output = tensor->data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        EXPECT_NEAR(output[i], 1.0f, 1e-5f);
    }
}

/**
 * @brief NEW: Test BF16Tensor::applyRMSNorm() - clean interface method
 */
TEST_F(Test__RMSNormInterface, BF16_TensorMethod)
{
    const int seq_len = 4;
    const int d_model = 8;

    // Create BF16 tensor
    auto tensor = std::make_shared<BF16Tensor>(std::vector<size_t>{seq_len, d_model});

    // Initialize to BF16(1.0)
    uint16_t bf16_one = 0x3F80;
    std::fill_n(tensor->mutable_bf16_data(), seq_len * d_model, bf16_one);

    // Create gamma weights
    std::vector<float> gamma(d_model, 1.0f);

    // Call applyRMSNorm directly on tensor - no type dispatch in pipeline!
    bool success = tensor->applyRMSNorm(gamma.data(), seq_len, d_model);

    EXPECT_TRUE(success);

    // Verify output is approximately BF16(1.0)
    const uint16_t *output = tensor->bf16_data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        EXPECT_NEAR(output[i], bf16_one, 2); // Within 2 ULPs
    }
}

/**
 * @brief NEW: Test FP16Tensor::applyRMSNorm() - clean interface method
 */
TEST_F(Test__RMSNormInterface, FP16_TensorMethod)
{
    const int seq_len = 4;
    const int d_model = 8;

    // Create FP16 tensor
    auto tensor = std::make_shared<FP16Tensor>(std::vector<size_t>{seq_len, d_model});

    // Initialize to FP16(1.0)
    uint16_t fp16_one = 0x3C00;
    std::fill_n(tensor->mutable_fp16_data(), seq_len * d_model, fp16_one);

    // Create gamma weights
    std::vector<float> gamma(d_model, 1.0f);

    // Call applyRMSNorm directly on tensor
    bool success = tensor->applyRMSNorm(gamma.data(), seq_len, d_model);

    EXPECT_TRUE(success);

    // Verify output is approximately FP16(1.0)
    const uint16_t *output = tensor->fp16_data();
    for (size_t i = 0; i < seq_len * d_model; ++i)
    {
        EXPECT_NEAR(output[i], fp16_one, 2); // Within 2 ULPs
    }
}

/**
 * @brief NEW: Polymorphic dispatch through IActivationTensor interface
 */
TEST_F(Test__RMSNormInterface, PolymorphicDispatch)
{
    const int seq_len = 4;
    const int d_model = 8;

    // Create tensors of different types
    std::vector<IActivationTensor *> tensors;
    tensors.push_back(new FP32Tensor({seq_len, d_model}));
    tensors.push_back(new BF16Tensor({seq_len, d_model}));
    tensors.push_back(new FP16Tensor({seq_len, d_model}));

    std::vector<float> gamma(d_model, 1.0f);

    // Polymorphic dispatch - each tensor knows how to normalize itself!
    for (auto *tensor : tensors)
    {
        bool success = tensor->applyRMSNorm(gamma.data(), seq_len, d_model);
        EXPECT_TRUE(success) << "Failed for tensor type";
        delete tensor;
    }
}
