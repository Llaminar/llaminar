/**
 * @file Test__ResidualAddStage_Precision.cpp
 * @brief Unit tests for precision-aware ResidualAddStage (TensorBase* API)
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests verify that ResidualAddStage correctly handles different tensor types:
 * - FP32Tensor: Standard float addition
 * - BF16Tensor: BF16 addition with conversion
 * - FP16Tensor: FP16 addition with conversion
 *
 * NOTE: Q8_1 tests require Q8_1Tensor (not raw Q8_1Block buffers).
 * Q8_1Tensor tests are deferred until Q8_1Tensor supports construction from raw data.
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/ComputeStages.h"
#include "execution/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/SIMDHelpers.h"
#include <vector>
#include <cmath>
#include <random>
#include <memory>

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class ResidualAddStagePrecisionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<CPUDeviceContext>(0, 4);
        ASSERT_NE(ctx_, nullptr);

        // Initialize RNG for test data
        rng_.seed(42);
    }

    // Helper: Create FP32Tensor with given dimensions
    std::unique_ptr<FP32Tensor> makeFP32Tensor(size_t rows, size_t cols)
    {
        return std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols}, 0);
    }

    // Helper: Create and fill FP32Tensor with values
    std::unique_ptr<FP32Tensor> makeFP32Tensor(size_t rows, size_t cols, const std::vector<float> &data)
    {
        auto tensor = makeFP32Tensor(rows, cols);
        if (!data.empty())
        {
            std::copy(data.begin(), data.end(), tensor->mutable_data());
        }
        return tensor;
    }

    // Helper: Create BF16Tensor with given dimensions (no device_idx)
    std::unique_ptr<BF16Tensor> makeBF16Tensor(size_t rows, size_t cols)
    {
        return std::make_unique<BF16Tensor>(std::vector<size_t>{rows, cols});
    }

    // Helper: Create FP16Tensor with given dimensions (no device_idx)
    std::unique_ptr<FP16Tensor> makeFP16Tensor(size_t rows, size_t cols)
    {
        return std::make_unique<FP16Tensor>(std::vector<size_t>{rows, cols});
    }

    // Helper: Create Q8_1Tensor with given dimensions and FP32 data (quantized to Q8_1)
    // Uses quantize_from_fp32() for proper block construction
    std::unique_ptr<Q8_1Tensor> makeQ8_1Tensor(size_t rows, size_t cols, const std::vector<float> &fp32_data)
    {
        auto tensor = Q8_1Tensor::quantize_from_fp32(fp32_data.data(), {rows, cols});
        // Use copy constructor to convert shared_ptr to unique_ptr
        return std::make_unique<Q8_1Tensor>(*tensor);
    }

    // Helper: Create empty Q8_1Tensor (for output buffers)
    std::unique_ptr<Q8_1Tensor> makeQ8_1Tensor(size_t rows, size_t cols)
    {
        return std::make_unique<Q8_1Tensor>(std::vector<size_t>{rows, cols}, -1);
    }

    // Fill FP32 data with random values in [-1, 1]
    void fillRandomFP32(float *data, size_t n)
    {
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (size_t i = 0; i < n; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    std::unique_ptr<CPUDeviceContext> ctx_;
    std::mt19937 rng_;
};

// =============================================================================
// FP32 Tests (using FP32Tensor)
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, FP32_BasicAddition)
{
    const size_t n = 128;
    std::vector<float> input_data(n), residual_data(n);

    // Fill with simple test values
    for (size_t i = 0; i < n; ++i)
    {
        input_data[i] = static_cast<float>(i);
        residual_data[i] = static_cast<float>(i * 2);
    }

    auto input = makeFP32Tensor(1, n, input_data);
    auto residual = makeFP32Tensor(1, n, residual_data);
    auto output = makeFP32Tensor(1, n);

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Verify: output[i] = input[i] + residual[i] = i + 2i = 3i
    const float *out_data = output->data();
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(out_data[i], static_cast<float>(3 * i))
            << "Mismatch at index " << i;
    }
}

TEST_F(ResidualAddStagePrecisionTest, FP32_InPlace)
{
    const size_t n = 64;
    std::vector<float> input_data(n), residual_data(n);
    fillRandomFP32(input_data.data(), n);
    fillRandomFP32(residual_data.data(), n);

    // Keep copies for verification
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i)
    {
        expected[i] = input_data[i] + residual_data[i];
    }

    auto input = makeFP32Tensor(1, n, input_data);
    auto residual = makeFP32Tensor(1, n, residual_data);

    // In-place: output = residual (same buffer)
    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = residual.get(); // In-place into residual

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    const float *out_data = residual->data();
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_NEAR(out_data[i], expected[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

TEST_F(ResidualAddStagePrecisionTest, FP32_QwenDimensions)
{
    // Test with Qwen2 0.5B dimensions: seq_len=32, d_model=896
    const size_t rows = 32;
    const size_t cols = 896;

    auto input = makeFP32Tensor(rows, cols);
    auto residual = makeFP32Tensor(rows, cols);
    auto output = makeFP32Tensor(rows, cols);

    fillRandomFP32(input->mutable_data(), input->numel());
    fillRandomFP32(residual->mutable_data(), residual->numel());

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Spot check
    const float *in_data = input->data();
    const float *res_data = residual->data();
    const float *out_data = output->data();
    for (size_t i = 0; i < input->numel(); i += 100)
    {
        EXPECT_NEAR(out_data[i], in_data[i] + res_data[i], 1e-6f)
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// Q8_1 Tests
// Uses Q8_1Tensor with quantize_from_fp32() for proper block construction.
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, Q8_1_BasicAddition)
{
    // Q8_1 blocks are 32 elements each, test with multiple blocks
    const size_t n = 128; // 4 blocks

    // Create FP32 source data
    std::vector<float> in_fp32(n), res_fp32(n);
    for (size_t i = 0; i < n; ++i)
    {
        in_fp32[i] = static_cast<float>(i) * 0.1f;
        res_fp32[i] = static_cast<float>(i) * 0.05f;
    }

    // Create Q8_1 tensors via quantization
    auto input = makeQ8_1Tensor(1, n, in_fp32);
    auto residual = makeQ8_1Tensor(1, n, res_fp32);
    auto output = makeQ8_1Tensor(1, n, std::vector<float>(n, 0.0f)); // Zero-initialized

    ASSERT_NE(input, nullptr);
    ASSERT_NE(residual, nullptr);
    ASSERT_NE(output, nullptr);

    // Create the stage
    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    auto stage = ComputeStageFactory::createResidualAdd(params);
    ASSERT_NE(stage, nullptr);

    // Execute
    bool result = stage->execute(ctx_.get());
    EXPECT_TRUE(result);

    // Verify output by dequantizing and comparing
    // Q8_1 has some quantization error, so we use a reasonable tolerance
    // Use fp32_data() to explicitly get dequantized values
    const float *out_dequant = output->fp32_data();
    for (size_t i = 0; i < n; ++i)
    {
        float expected = in_fp32[i] + res_fp32[i];
        // Q8_1 quantization can introduce error proportional to the magnitude
        float tolerance = std::max(0.1f, std::abs(expected) * 0.05f);
        EXPECT_NEAR(out_dequant[i], expected, tolerance)
            << "Mismatch at index " << i << ": expected " << expected
            << " got " << out_dequant[i];
    }
}

TEST_F(ResidualAddStagePrecisionTest, Q8_1_QwenDimensions)
{
    // Test with Qwen 2.5 0.5B dimensions: d_model = 896
    const size_t d_model = 896;

    // Create FP32 source data with typical activation magnitudes
    std::vector<float> in_fp32(d_model), res_fp32(d_model);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < d_model; ++i)
    {
        in_fp32[i] = dist(rng_);
        res_fp32[i] = dist(rng_);
    }

    // Create Q8_1 tensors
    auto input = makeQ8_1Tensor(1, d_model, in_fp32);
    auto residual = makeQ8_1Tensor(1, d_model, res_fp32);
    auto output = makeQ8_1Tensor(1, d_model, std::vector<float>(d_model, 0.0f));

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    auto stage = ComputeStageFactory::createResidualAdd(params);
    ASSERT_NE(stage, nullptr);

    bool result = stage->execute(ctx_.get());
    EXPECT_TRUE(result);

    // Verify output - use fp32_data() to explicitly get dequantized values
    const float *out_dequant = output->fp32_data();
    float max_error = 0.0f;
    for (size_t i = 0; i < d_model; ++i)
    {
        float expected = in_fp32[i] + res_fp32[i];
        float error = std::abs(out_dequant[i] - expected);
        max_error = std::max(max_error, error);
        // Q8_1 has ~0.5% relative error at typical magnitudes
        float tolerance = std::max(0.05f, std::abs(expected) * 0.02f);
        EXPECT_NEAR(out_dequant[i], expected, tolerance)
            << "Mismatch at index " << i;
    }

    LOG_INFO("[Q8_1_QwenDimensions] Max error: " << max_error);
}

TEST_F(ResidualAddStagePrecisionTest, Q8_1_MatchesDirectSIMD)
{
    // Verify that ResidualAddStage produces the same result as direct simd::q8_1_add_q8_1 call
    const size_t n = 256; // 8 blocks

    // Create FP32 source data
    std::vector<float> in_fp32(n), res_fp32(n);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    for (size_t i = 0; i < n; ++i)
    {
        in_fp32[i] = dist(rng_);
        res_fp32[i] = dist(rng_);
    }

    // Create Q8_1 tensors for ResidualAddStage
    auto input = makeQ8_1Tensor(1, n, in_fp32);
    auto residual = makeQ8_1Tensor(1, n, res_fp32);
    auto output_stage = makeQ8_1Tensor(1, n, std::vector<float>(n, 0.0f));

    // Create separate Q8_1 tensors for direct SIMD call
    auto output_simd = makeQ8_1Tensor(1, n, std::vector<float>(n, 0.0f));

    // Execute via ResidualAddStage
    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output_stage.get();

    auto stage = ComputeStageFactory::createResidualAdd(params);
    ASSERT_TRUE(stage->execute(ctx_.get()));

    // Execute via direct SIMD call
    simd::q8_1_add_q8_1(
        input->q8_1_blocks(),
        residual->q8_1_blocks(),
        output_simd->mutable_q8_1_blocks(),
        n);

    // Compare outputs - they should be bit-identical
    const Q8_1Block *stage_blocks = output_stage->q8_1_blocks();
    const Q8_1Block *simd_blocks = output_simd->q8_1_blocks();
    const size_t num_blocks = n / 32;

    for (size_t b = 0; b < num_blocks; ++b)
    {
        EXPECT_EQ(stage_blocks[b].d, simd_blocks[b].d)
            << "Scale mismatch at block " << b;
        EXPECT_EQ(stage_blocks[b].sum_qs, simd_blocks[b].sum_qs)
            << "Sum mismatch at block " << b;
        for (int i = 0; i < 32; ++i)
        {
            EXPECT_EQ(stage_blocks[b].qs[i], simd_blocks[b].qs[i])
                << "Quantized value mismatch at block " << b << " element " << i;
        }
    }
}

// =============================================================================
// BF16 Tests (using BF16Tensor)
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, BF16_BasicAddition)
{
    const size_t n = 64;

    auto input = makeBF16Tensor(1, n);
    auto residual = makeBF16Tensor(1, n);
    auto output = makeBF16Tensor(1, n);

    // Fill with BF16 values via mutable_bf16_data()
    uint16_t *in_data = input->mutable_bf16_data();
    uint16_t *res_data = residual->mutable_bf16_data();
    for (size_t i = 0; i < n; ++i)
    {
        in_data[i] = simd::fp32_to_bf16(static_cast<float>(i) * 0.1f);
        res_data[i] = simd::fp32_to_bf16(static_cast<float>(i) * 0.2f);
    }

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Verify (with BF16 precision tolerance)
    const uint16_t *out_data = output->bf16_data();
    for (size_t i = 0; i < n; ++i)
    {
        float expected = static_cast<float>(i) * 0.3f; // 0.1i + 0.2i
        float actual = simd::bf16_to_fp32(out_data[i]);
        // BF16 has limited precision
        EXPECT_NEAR(actual, expected, 0.01f * std::abs(expected) + 0.01f)
            << "BF16 mismatch at index " << i;
    }
}

// =============================================================================
// FP16 Tests (using FP16Tensor)
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, FP16_BasicAddition)
{
    const size_t n = 64;

    auto input = makeFP16Tensor(1, n);
    auto residual = makeFP16Tensor(1, n);
    auto output = makeFP16Tensor(1, n);

    // Fill with FP16 values via mutable_fp16_data()
    uint16_t *in_data = input->mutable_fp16_data();
    uint16_t *res_data = residual->mutable_fp16_data();
    for (size_t i = 0; i < n; ++i)
    {
        in_data[i] = simd::fp32_to_fp16(static_cast<float>(i) * 0.1f);
        res_data[i] = simd::fp32_to_fp16(static_cast<float>(i) * 0.2f);
    }

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    ASSERT_TRUE(stage.execute(ctx_.get()));

    // Verify (with FP16 precision tolerance)
    const uint16_t *out_data = output->fp16_data();
    for (size_t i = 0; i < n; ++i)
    {
        float expected = static_cast<float>(i) * 0.3f;
        float actual = simd::fp16_to_fp32(out_data[i]);
        EXPECT_NEAR(actual, expected, 0.01f * std::abs(expected) + 0.01f)
            << "FP16 mismatch at index " << i;
    }
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, NullContext_Fails)
{
    auto input = makeFP32Tensor(1, 32);
    auto residual = makeFP32Tensor(1, 32);
    auto output = makeFP32Tensor(1, 32);

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    EXPECT_FALSE(stage.execute(nullptr));
}

TEST_F(ResidualAddStagePrecisionTest, EstimatedFlops_TensorBased)
{
    // Test with FP32Tensor - 1 flop per element
    const size_t n = 1000;
    auto input = makeFP32Tensor(1, n);
    auto residual = makeFP32Tensor(1, n);
    auto output = makeFP32Tensor(1, n);

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    EXPECT_EQ(stage.estimatedFlops(), 1000);
}

TEST_F(ResidualAddStagePrecisionTest, EstimatedMemory_TensorBased)
{
    // FP32: 4 bytes per element, 3 buffers
    const size_t n = 1000;
    auto fp32_input = makeFP32Tensor(1, n);
    auto fp32_residual = makeFP32Tensor(1, n);
    auto fp32_output = makeFP32Tensor(1, n);

    ResidualAddStage::Params fp32_params;
    fp32_params.input = fp32_input.get();
    fp32_params.residual = fp32_residual.get();
    fp32_params.output = fp32_output.get();

    ResidualAddStage fp32_stage(fp32_params);
    EXPECT_EQ(fp32_stage.estimatedMemoryBytes(), 1000 * 4 * 3);

    // BF16: 2 bytes per element, 3 buffers
    auto bf16_input = makeBF16Tensor(1, n);
    auto bf16_residual = makeBF16Tensor(1, n);
    auto bf16_output = makeBF16Tensor(1, n);

    ResidualAddStage::Params bf16_params;
    bf16_params.input = bf16_input.get();
    bf16_params.residual = bf16_residual.get();
    bf16_params.output = bf16_output.get();

    ResidualAddStage bf16_stage(bf16_params);
    EXPECT_EQ(bf16_stage.estimatedMemoryBytes(), 1000 * 2 * 3);
}

// =============================================================================
// Stage Introspection Tests
// =============================================================================

TEST_F(ResidualAddStagePrecisionTest, StageType)
{
    auto input = makeFP32Tensor(1, 10);
    auto residual = makeFP32Tensor(1, 10);
    auto output = makeFP32Tensor(1, 10);

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);
    EXPECT_STREQ(stage.name().c_str(), "ADD_RESIDUAL");
}

TEST_F(ResidualAddStagePrecisionTest, SnapshotInfo_FP32)
{
    auto input = makeFP32Tensor(10, 10);
    auto residual = makeFP32Tensor(10, 10);
    auto output = makeFP32Tensor(10, 10);

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    auto dump_info = stage.getDumpInfo();

    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output->data());
    EXPECT_EQ(dump_info.outputs[0].rows, 10);
    EXPECT_EQ(dump_info.outputs[0].cols, 10);
}

TEST_F(ResidualAddStagePrecisionTest, DumpInfo_BF16_ReturnsWithDtype)
{
    // BF16 dump info returns the FP32 shadow data pointer with dtype="BF16"
    // (getDumpInfo uses tensor->data() which returns FP32 accessor for all tensor types)
    auto input = makeBF16Tensor(1, 128);
    auto residual = makeBF16Tensor(1, 128);
    auto output = makeBF16Tensor(1, 128);

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();

    ResidualAddStage stage(params);
    auto dump_info = stage.getDumpInfo();

    // BF16 outputs are present with correct dtype annotation
    ASSERT_EQ(dump_info.outputs.size(), 1);
    // getDumpInfo returns FP32 data() pointer, not raw bf16_data()
    EXPECT_EQ(dump_info.outputs[0].data, output->data());
    EXPECT_STREQ(dump_info.outputs[0].dtype, "BF16");
    // element_size reflects the BF16 size (2 bytes)
    EXPECT_EQ(dump_info.outputs[0].element_size, sizeof(uint16_t));
}
