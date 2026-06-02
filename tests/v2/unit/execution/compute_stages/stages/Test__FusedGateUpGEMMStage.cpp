/**
 * @file Test__FusedGateUpGEMMStage.cpp
 * @brief Unit tests for FusedGateUpGEMMStage compute stage
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests the FusedGateUpGEMMStage which performs fused gate/up projections
 * for FFN with shared activation quantization.
 *
 * Key test focus:
 * - Bias is correctly applied when set
 * - Bias is not applied when nullptr
 * - Basic execution without bias
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>
#include <memory>
#include <numeric>
#include <optional>

#include "execution/compute_stages/ComputeStages.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/IQQuantTables.h"
#include "tensors/FP16Utils.h"
#include "kernels/KernelFactory.h"
#include "utils/Logger.h"
#include "../../../../utils/PreparedWeightTestHarness.h"

namespace llaminar2
{
    namespace
    {
        // Helper: Generate random FP32 data
        void fill_random(float *data, size_t count, float bound = 1.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Helper: Fill with constant value
        void fill_constant(float *data, size_t count, float value)
        {
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = value;
            }
        }

        // Helper: Create mock IQ4_NL weight tensor for testing
        std::unique_ptr<TensorBase> create_mock_weights(int rows, int cols, unsigned seed = 123)
        {
            std::vector<size_t> shape = {static_cast<size_t>(rows), static_cast<size_t>(cols)};
            size_t blocks_per_row = (cols + 31) / 32;
            size_t total_blocks = rows * blocks_per_row;
            std::vector<uint8_t> raw_data(total_blocks * 18); // 18 bytes per IQ4_NL block

            std::mt19937 rng(seed);
            std::uniform_real_distribution<float> scale_dist(0.1f, 2.0f);
            std::uniform_int_distribution<uint8_t> index_dist(0, 15);

            for (size_t b = 0; b < total_blocks; ++b)
            {
                IQ4_NLBlock *block = reinterpret_cast<IQ4_NLBlock *>(raw_data.data() + b * 18);
                block->d = fp32_to_fp16(scale_dist(rng));
                for (int i = 0; i < 16; ++i)
                {
                    uint8_t low = index_dist(rng);
                    uint8_t high = index_dist(rng);
                    block->qs[i] = (high << 4) | low;
                }
            }

            return std::make_unique<IQ4_NLTensor>(shape, raw_data);
        }

        // Helper: Check if two arrays are approximately equal
        bool arrays_approx_equal(const float *a, const float *b, size_t count, float tol = 1e-5f)
        {
            for (size_t i = 0; i < count; ++i)
            {
                if (std::abs(a[i] - b[i]) > tol)
                    return false;
            }
            return true;
        }
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__FusedGateUpGEMMStage : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Clear kernel cache to prevent stale pointers from previous tests
            llaminar::v2::kernels::KernelFactory::clearCache();

            // Default dimensions (small for fast tests)
            m_ = 4;        // sequence length
            k_ = 64;       // d_model (input features)
            n_gate_ = 128; // intermediate dimension
            n_up_ = 128;   // intermediate dimension (same as gate for most models)

            // Create CPU device context
            ctx_ = std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 4);
            ASSERT_NE(ctx_, nullptr);

            // Create random input activations
            input_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(k_)}, DeviceId::cpu());
            fill_random(input_->mutable_data(), m_ * k_, 1.0f, 42);

            // Create weight tensors
            w_gate_ = create_mock_weights(n_gate_, k_, 100);
            w_up_ = create_mock_weights(n_up_, k_, 200);
            prepared_gate_up_.emplace(test::makePreparedGateUpFixture(
                w_gate_.get(), w_up_.get(), DeviceId::cpu(), 0));

            // Create output tensors
            output_gate_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_gate_)}, DeviceId::cpu());
            output_up_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(m_), static_cast<size_t>(n_up_)}, DeviceId::cpu());

            // Create bias vectors (filled with known values for easy verification)
            bias_gate_data_.resize(n_gate_);
            bias_up_data_.resize(n_up_);

            // Use distinct constant values for easy checking
            fill_constant(bias_gate_data_.data(), n_gate_, 2.0f);
            fill_constant(bias_up_data_.data(), n_up_, 3.0f);

            // Wrap in FP32Tensor for API
            bias_gate_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_gate_)}, DeviceId::cpu());
            bias_up_ = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(n_up_)}, DeviceId::cpu());
            std::copy(bias_gate_data_.begin(), bias_gate_data_.end(), bias_gate_->mutable_data());
            std::copy(bias_up_data_.begin(), bias_up_data_.end(), bias_up_->mutable_data());
        }

        void attachPreparedRefs(FusedGateUpGEMMStage::Params &params)
        {
            params.prepared_ref_gate = prepared_gate_up_->gate_ref;
            params.prepared_ref_up = prepared_gate_up_->up_ref;
            params.prepared_store = prepared_gate_up_->store.get();
        }

        void TearDown() override
        {
            // Clear kernel cache to prevent stale pointers
            llaminar::v2::kernels::KernelFactory::clearCache();
        }

        // Dimensions
        int m_, k_, n_gate_, n_up_;

        // Device context
        std::unique_ptr<CPUDeviceContext> ctx_;

        // Tensors
        std::unique_ptr<FP32Tensor> input_;
        std::unique_ptr<TensorBase> w_gate_, w_up_;
        std::optional<test::PreparedGateUpFixture> prepared_gate_up_;
        std::unique_ptr<FP32Tensor> output_gate_, output_up_;

        // Bias tensors (FP32Tensor for API compatibility)
        std::vector<float> bias_gate_data_, bias_up_data_;
        std::unique_ptr<FP32Tensor> bias_gate_, bias_up_;
    };

    // =============================================================================
    // Basic Functionality Tests
    // =============================================================================

    TEST_F(Test__FusedGateUpGEMMStage, ExecuteWithoutBias)
    {
        // Test basic execution without bias
        FusedGateUpGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .w_gate = w_gate_.get(),
            .output_gate = output_gate_.get(),
            .n_gate = n_gate_,
            .bias_gate = nullptr,
            .w_up = w_up_.get(),
            .output_up = output_up_.get(),
            .n_up = n_up_,
            .bias_up = nullptr};

        attachPreparedRefs(params);

        FusedGateUpGEMMStage stage(params);

        // Execute
        ASSERT_TRUE(stage.execute(ctx_.get()));

        // Verify outputs are non-zero (GEMM was actually computed)
        bool gate_nonzero = false, up_nonzero = false;
        for (int i = 0; i < m_ * n_gate_; ++i)
        {
            if (std::abs(output_gate_->data()[i]) > 1e-10f)
            {
                gate_nonzero = true;
                break;
            }
        }
        for (int i = 0; i < m_ * n_up_; ++i)
        {
            if (std::abs(output_up_->data()[i]) > 1e-10f)
            {
                up_nonzero = true;
                break;
            }
        }

        EXPECT_TRUE(gate_nonzero) << "Gate output should have non-zero values";
        EXPECT_TRUE(up_nonzero) << "Up output should have non-zero values";
    }

    // =============================================================================
    // Bias Tests
    // =============================================================================

    TEST_F(Test__FusedGateUpGEMMStage, BiasIsAppliedWhenSet)
    {
        // Run the same GEMM with and without bias, and verify the difference.

        // --- Step 1: Run without bias ---
        std::vector<float> output_gate_no_bias(m_ * n_gate_);
        std::vector<float> output_up_no_bias(m_ * n_up_);

        {
            // Zero output buffers
            std::fill(output_gate_->mutable_data(), output_gate_->mutable_data() + m_ * n_gate_, 0.0f);
            std::fill(output_up_->mutable_data(), output_up_->mutable_data() + m_ * n_up_, 0.0f);

            FusedGateUpGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .w_gate = w_gate_.get(),
                .output_gate = output_gate_.get(),
                .n_gate = n_gate_,
                .bias_gate = nullptr,
                .w_up = w_up_.get(),
                .output_up = output_up_.get(),
                .n_up = n_up_,
                .bias_up = nullptr};

            attachPreparedRefs(params);

            FusedGateUpGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            // Save results
            std::copy(output_gate_->data(), output_gate_->data() + m_ * n_gate_, output_gate_no_bias.begin());
            std::copy(output_up_->data(), output_up_->data() + m_ * n_up_, output_up_no_bias.begin());
        }

        // --- Step 2: Run WITH bias ---
        std::vector<float> output_gate_with_bias(m_ * n_gate_);
        std::vector<float> output_up_with_bias(m_ * n_up_);

        {
            // Zero output buffers
            std::fill(output_gate_->mutable_data(), output_gate_->mutable_data() + m_ * n_gate_, 0.0f);
            std::fill(output_up_->mutable_data(), output_up_->mutable_data() + m_ * n_up_, 0.0f);

            FusedGateUpGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .w_gate = w_gate_.get(),
                .output_gate = output_gate_.get(),
                .n_gate = n_gate_,
                .bias_gate = bias_gate_.get(),
                .w_up = w_up_.get(),
                .output_up = output_up_.get(),
                .n_up = n_up_,
                .bias_up = bias_up_.get()};

            attachPreparedRefs(params);

            FusedGateUpGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            // Save results
            std::copy(output_gate_->data(), output_gate_->data() + m_ * n_gate_, output_gate_with_bias.begin());
            std::copy(output_up_->data(), output_up_->data() + m_ * n_up_, output_up_with_bias.begin());
        }

        // --- Step 3: Verify that output_with_bias - output_no_bias == bias ---
        float tol = 1e-4f;

        // Check Gate projection bias
        for (int row = 0; row < m_; ++row)
        {
            for (int col = 0; col < n_gate_; ++col)
            {
                int idx = row * n_gate_ + col;
                float diff = output_gate_with_bias[idx] - output_gate_no_bias[idx];
                float expected = bias_gate_data_[col];
                EXPECT_NEAR(diff, expected, tol)
                    << "Gate bias mismatch at row=" << row << " col=" << col
                    << " diff=" << diff << " expected=" << expected;
            }
        }

        // Check Up projection bias
        for (int row = 0; row < m_; ++row)
        {
            for (int col = 0; col < n_up_; ++col)
            {
                int idx = row * n_up_ + col;
                float diff = output_up_with_bias[idx] - output_up_no_bias[idx];
                float expected = bias_up_data_[col];
                EXPECT_NEAR(diff, expected, tol)
                    << "Up bias mismatch at row=" << row << " col=" << col
                    << " diff=" << diff << " expected=" << expected;
            }
        }
    }

    TEST_F(Test__FusedGateUpGEMMStage, PartialBiasOnly_Gate)
    {
        // Test that only Gate bias is applied when Up bias is nullptr
        std::vector<float> output_gate_no_bias(m_ * n_gate_);
        std::vector<float> output_up_no_bias(m_ * n_up_);

        // Run without any bias first
        {
            std::fill(output_gate_->mutable_data(), output_gate_->mutable_data() + m_ * n_gate_, 0.0f);
            std::fill(output_up_->mutable_data(), output_up_->mutable_data() + m_ * n_up_, 0.0f);

            FusedGateUpGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .w_gate = w_gate_.get(),
                .output_gate = output_gate_.get(),
                .n_gate = n_gate_,
                .bias_gate = nullptr,
                .w_up = w_up_.get(),
                .output_up = output_up_.get(),
                .n_up = n_up_,
                .bias_up = nullptr};

            attachPreparedRefs(params);

            FusedGateUpGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            std::copy(output_gate_->data(), output_gate_->data() + m_ * n_gate_, output_gate_no_bias.begin());
            std::copy(output_up_->data(), output_up_->data() + m_ * n_up_, output_up_no_bias.begin());
        }

        // Run with only Gate bias
        std::vector<float> output_gate_with_bias(m_ * n_gate_);
        std::vector<float> output_up_partial(m_ * n_up_);

        {
            std::fill(output_gate_->mutable_data(), output_gate_->mutable_data() + m_ * n_gate_, 0.0f);
            std::fill(output_up_->mutable_data(), output_up_->mutable_data() + m_ * n_up_, 0.0f);

            FusedGateUpGEMMStage::Params params{
                .input = input_.get(),
                .m = m_,
                .k = k_,
                .w_gate = w_gate_.get(),
                .output_gate = output_gate_.get(),
                .n_gate = n_gate_,
                .bias_gate = bias_gate_.get(), // Only Gate has bias
                .w_up = w_up_.get(),
                .output_up = output_up_.get(),
                .n_up = n_up_,
                .bias_up = nullptr};

            attachPreparedRefs(params);

            FusedGateUpGEMMStage stage(params);
            ASSERT_TRUE(stage.execute(ctx_.get()));

            std::copy(output_gate_->data(), output_gate_->data() + m_ * n_gate_, output_gate_with_bias.begin());
            std::copy(output_up_->data(), output_up_->data() + m_ * n_up_, output_up_partial.begin());
        }

        float tol = 1e-4f;

        // Gate should have bias applied
        for (int row = 0; row < m_; ++row)
        {
            for (int col = 0; col < n_gate_; ++col)
            {
                int idx = row * n_gate_ + col;
                float diff = output_gate_with_bias[idx] - output_gate_no_bias[idx];
                EXPECT_NEAR(diff, bias_gate_data_[col], tol)
                    << "Gate bias should be applied at row=" << row << " col=" << col;
            }
        }

        // Up should be unchanged (no bias)
        EXPECT_TRUE(arrays_approx_equal(output_up_partial.data(), output_up_no_bias.data(), m_ * n_up_, tol))
            << "Up output should be unchanged when Up bias is nullptr";
    }

    // =============================================================================
    // Stage Metadata Tests
    // =============================================================================

    TEST_F(Test__FusedGateUpGEMMStage, StageType)
    {
        FusedGateUpGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .w_gate = w_gate_.get(),
            .output_gate = output_gate_.get(),
            .n_gate = n_gate_,
            .bias_gate = nullptr,
            .w_up = w_up_.get(),
            .output_up = output_up_.get(),
            .n_up = n_up_,
            .bias_up = nullptr};

        FusedGateUpGEMMStage stage(params);

        EXPECT_EQ(stage.type(), ComputeStageType::GEMM_FUSED_GATE_UP);
    }

    TEST_F(Test__FusedGateUpGEMMStage, EstimatedFlops)
    {
        FusedGateUpGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .w_gate = w_gate_.get(),
            .output_gate = output_gate_.get(),
            .n_gate = n_gate_,
            .bias_gate = nullptr,
            .w_up = w_up_.get(),
            .output_up = output_up_.get(),
            .n_up = n_up_,
            .bias_up = nullptr};

        FusedGateUpGEMMStage stage(params);

        // Expected: 2 * M * N * K for each of gate and up
        size_t expected_flops =
            2 * static_cast<size_t>(m_) * n_gate_ * k_ +
            2 * static_cast<size_t>(m_) * n_up_ * k_;

        EXPECT_EQ(stage.estimatedFlops(), expected_flops);
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    TEST_F(Test__FusedGateUpGEMMStage, NullContextFails)
    {
        FusedGateUpGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .w_gate = w_gate_.get(),
            .output_gate = output_gate_.get(),
            .n_gate = n_gate_,
            .bias_gate = nullptr,
            .w_up = w_up_.get(),
            .output_up = output_up_.get(),
            .n_up = n_up_,
            .bias_up = nullptr};

        FusedGateUpGEMMStage stage(params);

        EXPECT_FALSE(stage.execute(nullptr));
    }

    TEST_F(Test__FusedGateUpGEMMStage, NullInputFails)
    {
        FusedGateUpGEMMStage::Params params{
            .input = nullptr, // Null input
            .m = m_,
            .k = k_,
            .w_gate = w_gate_.get(),
            .output_gate = output_gate_.get(),
            .n_gate = n_gate_,
            .bias_gate = nullptr,
            .w_up = w_up_.get(),
            .output_up = output_up_.get(),
            .n_up = n_up_,
            .bias_up = nullptr};

        FusedGateUpGEMMStage stage(params);

        EXPECT_FALSE(stage.execute(ctx_.get()));
    }

    TEST_F(Test__FusedGateUpGEMMStage, InvalidDimensionsFails)
    {
        FusedGateUpGEMMStage::Params params{
            .input = input_.get(),
            .m = 0, // Invalid dimension
            .k = k_,
            .w_gate = w_gate_.get(),
            .output_gate = output_gate_.get(),
            .n_gate = n_gate_,
            .bias_gate = nullptr,
            .w_up = w_up_.get(),
            .output_up = output_up_.get(),
            .n_up = n_up_,
            .bias_up = nullptr};

        FusedGateUpGEMMStage stage(params);

        EXPECT_FALSE(stage.execute(ctx_.get()));
    }

    TEST_F(Test__FusedGateUpGEMMStage, OutputCapacityTooSmallFailsBeforeKernelLookup)
    {
        auto undersized_gate = std::make_unique<FP32Tensor>(
            std::vector<size_t>{static_cast<size_t>(m_ - 1), static_cast<size_t>(n_gate_)},
            DeviceId::cpu());

        FusedGateUpGEMMStage::Params params{
            .input = input_.get(),
            .m = m_,
            .k = k_,
            .w_gate = w_gate_.get(),
            .output_gate = undersized_gate.get(),
            .n_gate = n_gate_,
            .bias_gate = nullptr,
            .w_up = w_up_.get(),
            .output_up = output_up_.get(),
            .n_up = n_up_,
            .bias_up = nullptr};

        FusedGateUpGEMMStage stage(params);

        EXPECT_FALSE(stage.execute(ctx_.get()));
    }

} // namespace llaminar2
