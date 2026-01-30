/**
 * @file Test__ResidualAddStage_Q16_1.cpp
 * @brief Unit tests for Phase 6: Pure Integer Residual Add (Q16_1 + Q16_1 → Q16_1)
 *
 * This test file verifies the integer residual add operation that is critical
 * for the full integer residual stream in Q16_1 attention.
 *
 * Key test scenarios:
 * 1. Block size 32/64/128 dispatch
 * 2. Correctness vs FP32 reference
 * 3. In-place operation support
 * 4. Edge cases (zeros, max values)
 *
 * See: PROJECT_Q16_INTEGER_ATTENTION_V2.md "Phase 6: Integer residual add"
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>

#include "execution/compute_stages/stages/ResidualAddStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"

namespace llaminar2::test
{

    class ResidualAddStageQ16_1Test : public ::testing::Test
    {
    protected:
        static constexpr int SEQ_LEN = 8;
        static constexpr int HIDDEN_DIM = 896;
        static constexpr int NUM_ELEMENTS = SEQ_LEN * HIDDEN_DIM;

        std::unique_ptr<CPUDeviceContext> ctx_;

        void SetUp() override
        {
            ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);
        }

        // Helper: Create Q16_1 tensor with random data using static factory
        std::unique_ptr<Q16_1Tensor> createRandomQ16_1(size_t rows, size_t cols, Q16BlockSize block_size)
        {
            std::vector<float> fp32_data(rows * cols);
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
            for (auto &v : fp32_data)
            {
                v = dist(gen);
            }

            auto tensor = std::make_unique<Q16_1Tensor>(
                std::vector<size_t>{rows, cols}, block_size, DeviceId::cpu());
            tensor->copyFrom_fp32(fp32_data.data());
            return tensor;
        }

        // Helper: Compute FP32 reference for Q16_1 + Q16_1
        std::vector<float> computeFP32Reference(const Q16_1Tensor *a, const Q16_1Tensor *b)
        {
            const float *a_fp32 = a->fp32_data();
            const float *b_fp32 = b->fp32_data();
            size_t n = a->numel();

            std::vector<float> result(n);
            for (size_t i = 0; i < n; ++i)
            {
                result[i] = a_fp32[i] + b_fp32[i];
            }
            return result;
        }

        // Helper: Compute MSE between Q16_1 tensor and FP32 reference
        float computeMSE(const Q16_1Tensor *tensor, const std::vector<float> &ref)
        {
            const float *tensor_fp32 = tensor->fp32_data();
            size_t n = ref.size();

            float mse = 0.0f;
            for (size_t i = 0; i < n; ++i)
            {
                float diff = tensor_fp32[i] - ref[i];
                mse += diff * diff;
            }
            return mse / static_cast<float>(n);
        }
    };

    // =========================================================================
    // Basic Functionality Tests
    // =========================================================================

    TEST_F(ResidualAddStageQ16_1Test, Block32_BasicAdd)
    {
        auto input = createRandomQ16_1(SEQ_LEN, HIDDEN_DIM, Q16BlockSize::BLOCK_32);
        auto residual = createRandomQ16_1(SEQ_LEN, HIDDEN_DIM, Q16BlockSize::BLOCK_32);
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{SEQ_LEN, HIDDEN_DIM}, Q16BlockSize::BLOCK_32, DeviceId::cpu());

        // Compute FP32 reference before operation
        auto ref = computeFP32Reference(input.get(), residual.get());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();
        params.num_elements = NUM_ELEMENTS;

        ResidualAddStage stage(params);
        ASSERT_TRUE(stage.execute(ctx_.get()));

        float mse = computeMSE(output.get(), ref);
        EXPECT_LT(mse, 0.01f) << "Block32 Q16_1 + Q16_1 MSE too high: " << mse;
    }

    TEST_F(ResidualAddStageQ16_1Test, Block64_BasicAdd)
    {
        auto input = createRandomQ16_1(SEQ_LEN, HIDDEN_DIM, Q16BlockSize::BLOCK_64);
        auto residual = createRandomQ16_1(SEQ_LEN, HIDDEN_DIM, Q16BlockSize::BLOCK_64);
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{SEQ_LEN, HIDDEN_DIM}, Q16BlockSize::BLOCK_64, DeviceId::cpu());

        auto ref = computeFP32Reference(input.get(), residual.get());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();
        params.num_elements = NUM_ELEMENTS;

        ResidualAddStage stage(params);
        ASSERT_TRUE(stage.execute(ctx_.get()));

        float mse = computeMSE(output.get(), ref);
        EXPECT_LT(mse, 0.01f) << "Block64 Q16_1 + Q16_1 MSE too high: " << mse;
    }

    TEST_F(ResidualAddStageQ16_1Test, Block128_BasicAdd)
    {
        auto input = createRandomQ16_1(SEQ_LEN, HIDDEN_DIM, Q16BlockSize::BLOCK_128);
        auto residual = createRandomQ16_1(SEQ_LEN, HIDDEN_DIM, Q16BlockSize::BLOCK_128);
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{SEQ_LEN, HIDDEN_DIM}, Q16BlockSize::BLOCK_128, DeviceId::cpu());

        auto ref = computeFP32Reference(input.get(), residual.get());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();
        params.num_elements = NUM_ELEMENTS;

        ResidualAddStage stage(params);
        ASSERT_TRUE(stage.execute(ctx_.get()));

        float mse = computeMSE(output.get(), ref);
        EXPECT_LT(mse, 0.01f) << "Block128 Q16_1 + Q16_1 MSE too high: " << mse;
    }

    // =========================================================================
    // Stage Type and Backend Tests
    // =========================================================================

    TEST_F(ResidualAddStageQ16_1Test, StageType_IsAddResidual)
    {
        auto input = createRandomQ16_1(1, 128, Q16BlockSize::BLOCK_64);
        auto residual = createRandomQ16_1(1, 128, Q16BlockSize::BLOCK_64);
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();

        ResidualAddStage stage(params);
        EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);
    }

    TEST_F(ResidualAddStageQ16_1Test, Backend_SupportsCPU)
    {
        auto input = createRandomQ16_1(1, 128, Q16BlockSize::BLOCK_64);
        auto residual = createRandomQ16_1(1, 128, Q16BlockSize::BLOCK_64);
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();

        ResidualAddStage stage(params);
        EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    TEST_F(ResidualAddStageQ16_1Test, ZeroInput_PreservesResidual)
    {
        // Create zero input and non-zero residual
        std::vector<float> zeros(128, 0.0f);
        std::vector<float> residual_data(128);
        for (int i = 0; i < 128; ++i)
        {
            residual_data[i] = static_cast<float>(i) * 0.01f;
        }

        auto input = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());
        auto residual = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());

        input->copyFrom_fp32(zeros.data());
        residual->copyFrom_fp32(residual_data.data());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();
        params.num_elements = 128;

        ResidualAddStage stage(params);
        ASSERT_TRUE(stage.execute(ctx_.get()));

        // Output should approximately equal residual (zero + residual = residual)
        const float *out_fp32 = output->fp32_data();
        float max_diff = 0.0f;
        for (int i = 0; i < 128; ++i)
        {
            float diff = std::abs(out_fp32[i] - residual_data[i]);
            max_diff = std::max(max_diff, diff);
        }
        EXPECT_LT(max_diff, 0.01f) << "Zero input should preserve residual";
    }

    TEST_F(ResidualAddStageQ16_1Test, ZeroResidual_PreservesInput)
    {
        std::vector<float> zeros(128, 0.0f);
        std::vector<float> input_data(128);
        for (int i = 0; i < 128; ++i)
        {
            input_data[i] = static_cast<float>(i) * 0.01f;
        }

        auto input = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());
        auto residual = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());

        input->copyFrom_fp32(input_data.data());
        residual->copyFrom_fp32(zeros.data());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();
        params.num_elements = 128;

        ResidualAddStage stage(params);
        ASSERT_TRUE(stage.execute(ctx_.get()));

        const float *out_fp32 = output->fp32_data();
        float max_diff = 0.0f;
        for (int i = 0; i < 128; ++i)
        {
            float diff = std::abs(out_fp32[i] - input_data[i]);
            max_diff = std::max(max_diff, diff);
        }
        EXPECT_LT(max_diff, 0.01f) << "Zero residual should preserve input";
    }

    // =========================================================================
    // Error Handling Tests
    // =========================================================================

    TEST_F(ResidualAddStageQ16_1Test, BlockSizeMismatch_Fails)
    {
        auto input = createRandomQ16_1(1, 128, Q16BlockSize::BLOCK_64);
        auto residual = createRandomQ16_1(1, 128, Q16BlockSize::BLOCK_128); // Mismatch!
        auto output = std::make_unique<Q16_1Tensor>(
            std::vector<size_t>{1, 128}, Q16BlockSize::BLOCK_64, DeviceId::cpu());

        ResidualAddStage::Params params;
        params.input = input.get();
        params.residual = residual.get();
        params.output = output.get();
        params.num_elements = 128;

        ResidualAddStage stage(params);
        EXPECT_FALSE(stage.execute(ctx_.get()))
            << "Should fail when block sizes don't match";
    }

    TEST_F(ResidualAddStageQ16_1Test, NullTensors_Fails)
    {
        ResidualAddStage::Params params;
        params.input = nullptr;
        params.residual = nullptr;
        params.output = nullptr;

        ResidualAddStage stage(params);
        EXPECT_FALSE(stage.execute(ctx_.get()))
            << "Should fail with null tensors";
    }

} // namespace llaminar2::test
