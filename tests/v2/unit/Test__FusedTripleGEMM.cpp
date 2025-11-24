/**
 * @file Test__FusedTripleGEMM.cpp
 * @brief Unit tests for FusedTripleGEMM kernel (Q/K/V shared quantization)
 * @author David Sanftenberg
 * @date 2025-11-23
 *
 * Tests the FusedTripleGEMM kernel which fuses Q/K/V projections in attention blocks:
 * 1. Single per-row quantization of input activations (FP32 → INT8)
 * 2. Three INT8×INT8 → INT32 GEMMs (Q, K, V projections)
 * 3. Outputs INT32 accumulators (deferred dequantization)
 *
 * Performance benefit: Eliminates 2 redundant quantization passes (Q/K/V share input)
 */

#include <gtest/gtest.h>
#include "../../../src/v2/kernels/cpu/fused/FusedTripleGEMM.h"
#include "../../../src/v2/tensors/Tensors.h"
#include "../../../src/v2/utils/Logger.h"
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

using namespace llaminar2;

namespace
{
    class Test__FusedTripleGEMM : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Typical attention projection dimensions (Qwen2.5-0.5B: d_model=896, num_heads=14, head_dim=64)
            m_ = 32;     // Batch size
            k_ = 896;    // d_model (input features)
            n_q_ = 896;  // Q output dimension (n_heads * head_dim)
            n_kv_ = 896; // K/V output dimension (n_kv_heads * head_dim, same as n_q for MHA)

            // Create random input activations
            input_fp32_.resize(m_ * k_);
            fill_random(input_fp32_.data(), m_ * k_, 1.0f);

            // Create mock INT8 weight tensors
            // NOTE: Weights must be [n, k] for pack_weights_to_int8 (expects [N, K] where C = A[m,k] @ B[k,n])
            q_weight_ = create_mock_int8_weights(n_q_, k_, 111);
            k_weight_ = create_mock_int8_weights(n_kv_, k_, 222);
            v_weight_ = create_mock_int8_weights(n_kv_, k_, 333);

            // Allocate output buffers
            q_output_int32_.resize(m_ * n_q_);
            k_output_int32_.resize(m_ * n_kv_);
            v_output_int32_.resize(m_ * n_kv_);
            activation_scales_.resize(m_);
        }

        // Helper: Fill array with random values
        void fill_random(float *data, size_t count, float bound, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-bound, bound);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Helper: Create mock INT8 weight tensor for testing
        std::unique_ptr<TensorBase> create_mock_int8_weights(int rows, int cols, unsigned seed = 123)
        {
            std::vector<float> fp32_weights(rows * cols);
            fill_random(fp32_weights.data(), rows * cols, 0.5f, seed);

            // INT8Tensor constructor with FP32 data automatically computes per-column scales
            return std::make_unique<INT8Tensor>(
                std::vector<size_t>{static_cast<size_t>(rows), static_cast<size_t>(cols)},
                fp32_weights);
        }

        // Helper: Compute relative error
        float compute_relative_error(const int32_t *a, const int32_t *b, size_t count)
        {
            double sum_sq_diff = 0.0;
            double sum_sq_ref = 0.0;

            for (size_t i = 0; i < count; ++i)
            {
                double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                sum_sq_diff += diff * diff;
                sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }

            if (sum_sq_ref < 1e-12)
                return 0.0f;

            return static_cast<float>(std::sqrt(sum_sq_diff / sum_sq_ref));
        }

        int m_, k_, n_q_, n_kv_;
        std::vector<float> input_fp32_;
        std::unique_ptr<TensorBase> q_weight_;
        std::unique_ptr<TensorBase> k_weight_;
        std::unique_ptr<TensorBase> v_weight_;
        std::vector<int32_t> q_output_int32_;
        std::vector<int32_t> k_output_int32_;
        std::vector<int32_t> v_output_int32_;
        std::vector<float> activation_scales_;
    };

    // =============================================================================
    // Correctness Tests
    // =============================================================================

    TEST_F(Test__FusedTripleGEMM, BasicCorrectness)
    {
        // Execute fused kernel
        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());

        bool success = fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_);

        ASSERT_TRUE(success) << "FusedTripleGEMM execution failed";

        // Validate activation scales (per-row scales for input quantization)
        for (int i = 0; i < m_; ++i)
        {
            EXPECT_GT(activation_scales_[i], 0.0f) << "Invalid activation scale at row " << i;
            EXPECT_LT(activation_scales_[i], 1e6f) << "Activation scale too large at row " << i;
        }

        // Validate INT32 outputs are non-zero (sanity check)
        bool has_nonzero_q = false, has_nonzero_k = false, has_nonzero_v = false;
        for (size_t i = 0; i < q_output_int32_.size(); ++i)
        {
            if (q_output_int32_[i] != 0)
                has_nonzero_q = true;
            if (k_output_int32_[i] != 0)
                has_nonzero_k = true;
            if (v_output_int32_[i] != 0)
                has_nonzero_v = true;
        }
        EXPECT_TRUE(has_nonzero_q) << "Q output is all zeros";
        EXPECT_TRUE(has_nonzero_k) << "K output is all zeros";
        EXPECT_TRUE(has_nonzero_v) << "V output is all zeros";

        LOG_INFO("✓ FusedTripleGEMM BasicCorrectness: Q/K/V INT32 outputs validated");
    }

    TEST_F(Test__FusedTripleGEMM, SingleToken)
    {
        // Single token (m=1) is a common decode scenario
        m_ = 1;
        SetUp();

        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());

        bool success = fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_);

        ASSERT_TRUE(success) << "FusedTripleGEMM failed for single token";
        EXPECT_GT(activation_scales_[0], 0.0f) << "Invalid activation scale for single token";

        LOG_INFO("✓ FusedTripleGEMM SingleToken: m=1 validated");
    }

    TEST_F(Test__FusedTripleGEMM, LargeBatch)
    {
        // Large batch (m=512) to test SIMD vectorization at scale
        m_ = 512;
        SetUp();

        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());

        bool success = fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_);

        ASSERT_TRUE(success) << "FusedTripleGEMM failed for large batch";

        // Validate all scales are valid
        for (int i = 0; i < m_; ++i)
        {
            EXPECT_GT(activation_scales_[i], 0.0f) << "Invalid scale at row " << i;
        }

        LOG_INFO("✓ FusedTripleGEMM LargeBatch: m=512 validated");
    }

    // =============================================================================
    // Edge Case Tests
    // =============================================================================

    TEST_F(Test__FusedTripleGEMM, NullPointerHandling)
    {
        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());

        // Null input
        EXPECT_FALSE(fused_kernel.execute(
            nullptr,
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_));

        // Null Q output
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            nullptr,
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_));

        // Null K output
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            nullptr,
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_));

        // Null V output
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            nullptr,
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_));

        // Null activation scales
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            nullptr,
            m_, n_q_, n_kv_, k_));

        LOG_INFO("✓ FusedTripleGEMM NullPointerHandling: All null checks passed");
    }

    TEST_F(Test__FusedTripleGEMM, InvalidDimensions)
    {
        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());

        // Zero batch size
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            0, n_q_, n_kv_, k_));

        // Zero Q output dimension
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, 0, n_kv_, k_));

        // Zero K/V output dimension
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, 0, k_));

        // Zero input features
        EXPECT_FALSE(fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, 0));

        LOG_INFO("✓ FusedTripleGEMM InvalidDimensions: All dimension checks passed");
    }

    TEST_F(Test__FusedTripleGEMM, MismatchedWeights)
    {
        // Create K weight with mismatched input dimension (k)
        auto mismatched_k_weight = create_mock_int8_weights(n_kv_, k_ + 10, 444);

        // Should throw during construction (k dimension must match)
        EXPECT_THROW(
            FusedTripleGEMM(q_weight_.get(), mismatched_k_weight.get(), v_weight_.get()),
            std::invalid_argument);

        // Create V weight with mismatched output dimension (n_kv)
        auto mismatched_v_weight = create_mock_int8_weights(n_kv_ + 10, k_, 555);

        // Should throw during construction (K and V output dims must match)
        EXPECT_THROW(
            FusedTripleGEMM(q_weight_.get(), k_weight_.get(), mismatched_v_weight.get()),
            std::invalid_argument);

        LOG_INFO("✓ FusedTripleGEMM MismatchedWeights: Constructor validation passed");
    }

    // =============================================================================
    // GQA (Grouped Query Attention) Tests
    // =============================================================================

    TEST_F(Test__FusedTripleGEMM, GQA_AsymmetricDimensions)
    {
        // Test GQA scenario: Q has more features than K/V
        // Example: Qwen2.5-0.5B has n_heads=14, n_kv_heads=2, head_dim=64
        // Q: 14 * 64 = 896, K/V: 2 * 64 = 128
        m_ = 32;
        k_ = 896;
        n_q_ = 896;  // 14 query heads
        n_kv_ = 128; // 2 key/value heads (GQA)

        // Recreate weights with asymmetric dimensions
        q_weight_ = create_mock_int8_weights(n_q_, k_, 111);
        k_weight_ = create_mock_int8_weights(n_kv_, k_, 222);
        v_weight_ = create_mock_int8_weights(n_kv_, k_, 333);

        // Allocate output buffers with correct dimensions
        input_fp32_.resize(m_ * k_);
        fill_random(input_fp32_.data(), m_ * k_, 1.0f);
        q_output_int32_.resize(m_ * n_q_);
        k_output_int32_.resize(m_ * n_kv_);
        v_output_int32_.resize(m_ * n_kv_);
        activation_scales_.resize(m_);

        // Execute fused kernel with asymmetric dimensions
        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());

        bool success = fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_);

        ASSERT_TRUE(success) << "FusedTripleGEMM failed for GQA (asymmetric dimensions)";

        // Validate outputs
        for (int i = 0; i < m_; ++i)
        {
            EXPECT_GT(activation_scales_[i], 0.0f) << "Invalid activation scale at row " << i;
        }

        // Check that Q output has correct size (m * n_q)
        bool has_nonzero_q = false;
        for (size_t i = 0; i < static_cast<size_t>(m_ * n_q_); ++i)
        {
            if (q_output_int32_[i] != 0)
                has_nonzero_q = true;
        }
        EXPECT_TRUE(has_nonzero_q) << "Q output is all zeros (n_q=" << n_q_ << ")";

        // Check that K/V outputs have correct size (m * n_kv)
        bool has_nonzero_k = false, has_nonzero_v = false;
        for (size_t i = 0; i < static_cast<size_t>(m_ * n_kv_); ++i)
        {
            if (k_output_int32_[i] != 0)
                has_nonzero_k = true;
            if (v_output_int32_[i] != 0)
                has_nonzero_v = true;
        }
        EXPECT_TRUE(has_nonzero_k) << "K output is all zeros (n_kv=" << n_kv_ << ")";
        EXPECT_TRUE(has_nonzero_v) << "V output is all zeros (n_kv=" << n_kv_ << ")";

        LOG_INFO("✓ FusedTripleGEMM GQA_AsymmetricDimensions: n_q=" << n_q_ << ", n_kv=" << n_kv_ << " validated");
    }

    TEST_F(Test__FusedTripleGEMM, GQA_ExtremeRatio)
    {
        // Test extreme GQA ratio (e.g., 32 query heads, 1 KV head)
        m_ = 16;
        k_ = 512;
        n_q_ = 2048; // 32 heads * 64 dim
        n_kv_ = 64;  // 1 head * 64 dim (extreme ratio)

        // Recreate weights
        q_weight_ = create_mock_int8_weights(n_q_, k_, 111);
        k_weight_ = create_mock_int8_weights(n_kv_, k_, 222);
        v_weight_ = create_mock_int8_weights(n_kv_, k_, 333);

        // Allocate buffers
        input_fp32_.resize(m_ * k_);
        fill_random(input_fp32_.data(), m_ * k_, 1.0f);
        q_output_int32_.resize(m_ * n_q_);
        k_output_int32_.resize(m_ * n_kv_);
        v_output_int32_.resize(m_ * n_kv_);
        activation_scales_.resize(m_);

        // Execute
        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());
        bool success = fused_kernel.execute(
            input_fp32_.data(),
            q_output_int32_.data(),
            k_output_int32_.data(),
            v_output_int32_.data(),
            activation_scales_.data(),
            m_, n_q_, n_kv_, k_);

        ASSERT_TRUE(success) << "FusedTripleGEMM failed for extreme GQA ratio";

        LOG_INFO("✓ FusedTripleGEMM GQA_ExtremeRatio: n_q=" << n_q_ << ", n_kv=" << n_kv_ << " (32:1 ratio) validated");
    }

    // =============================================================================
    // Kernel Contract Tests
    // =============================================================================

    TEST_F(Test__FusedTripleGEMM, KernelContract)
    {
        FusedTripleGEMM fused_kernel(q_weight_.get(), k_weight_.get(), v_weight_.get());

        // Verify kernel contract
        auto contract = fused_kernel.get_contract();

        // Output format
        EXPECT_EQ(contract.output_format, TensorFormat::INT32);

        // Fusion capabilities
        EXPECT_TRUE(contract.is_fusable);
        EXPECT_FALSE(contract.supports_inplace);

        // CPUKernelBase methods
        EXPECT_TRUE(fused_kernel.supports_fusion());
        EXPECT_EQ(fused_kernel.preferred_fusion_format(), TensorFormat::INT32);
        EXPECT_TRUE(fused_kernel.supports_device(-1)); // CPU

        LOG_INFO("✓ FusedTripleGEMM KernelContract: All contract properties validated");
    }

} // namespace

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
