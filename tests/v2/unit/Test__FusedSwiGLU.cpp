/**
 * @file Test__FusedSwiGLU.cpp
 * @brief Unit tests for FusedSwiGLU kernel (SwiGLU activation for FP32 inputs)
 * @author David Sanftenberg
 * @date 2025-11-23
 * @updated 2025-11-26 - Updated for FP32-only API (removed INT32 dequantization)
 * @updated 2025-12-01 - Fixed SwiGLU formula: silu(gate) * up (matches HuggingFace)
 *
 * Tests the FusedSwiGLU kernel against a reference implementation
 * to validate correctness. The kernel applies SwiGLU activation:
 *   output = silu(gate) * up
 *   where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * NOTE: This is the correct formula matching HuggingFace:
 *   down_proj(act_fn(gate_proj(x)) * up_proj(x))
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <random>

#include "kernels/cpu/fused/FusedDequantSwiGLU.h"
#include "utils/Logger.h"

namespace llaminar2
{
    namespace
    {
        // Helper: Generate random FP32 data
        void fill_random_fp32(float *data, size_t count, float range = 2.0f, unsigned seed = 42)
        {
            std::mt19937 gen(seed);
            std::uniform_real_distribution<float> dist(-range, range);
            for (size_t i = 0; i < count; ++i)
            {
                data[i] = dist(gen);
            }
        }

        // Reference SiLU implementation
        float silu_ref(float x)
        {
            return x / (1.0f + std::exp(-x));
        }

        // Reference implementation: SwiGLU on FP32 inputs
        // Correct formula: silu(gate) * up (matches HuggingFace)
        void reference_swiglu(
            const float *gate,
            const float *up,
            float *output,
            int m, int n)
        {
            for (int i = 0; i < m; ++i)
            {
                for (int j = 0; j < n; ++j)
                {
                    int idx = i * n + j;
                    output[idx] = silu_ref(gate[idx]) * up[idx];
                }
            }
        }

        // Compute relative error
        float compute_relative_error(const float *a, const float *b, size_t count)
        {
            double sum_sq_diff = 0.0;
            double sum_sq_ref = 0.0;

            for (size_t i = 0; i < count; ++i)
            {
                double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
                sum_sq_diff += diff * diff;
                sum_sq_ref += static_cast<double>(b[i]) * static_cast<double>(b[i]);
            }

            return static_cast<float>(std::sqrt(sum_sq_diff / (sum_sq_ref + 1e-12)));
        }
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__FusedSwiGLU : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Test dimensions
            m_ = 32;   // sequence length
            n_ = 4864; // intermediate_size

            // Allocate buffers
            gate_fp32_.resize(m_ * n_);
            up_fp32_.resize(m_ * n_);
            output_fp32_.resize(m_ * n_);
            output_ref_.resize(m_ * n_);

            // Fill with random data
            fill_random_fp32(gate_fp32_.data(), m_ * n_, 2.0f, 42);
            fill_random_fp32(up_fp32_.data(), m_ * n_, 2.0f, 123);
        }

        int m_, n_;
        std::vector<float> gate_fp32_;
        std::vector<float> up_fp32_;
        std::vector<float> output_fp32_;
        std::vector<float> output_ref_;
    };

    // =============================================================================
    // Correctness Tests
    // =============================================================================

    TEST_F(Test__FusedSwiGLU, BasicCorrectness)
    {
        // Execute fused kernel
        FusedSwiGLU kernel;

        bool success = kernel.execute(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_fp32_.data(),
            m_, n_);

        ASSERT_TRUE(success) << "FusedSwiGLU execution failed";

        // Compute reference
        reference_swiglu(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_ref_.data(),
            m_, n_);

        // Compare outputs
        float rel_error = compute_relative_error(
            output_fp32_.data(), output_ref_.data(), m_ * n_);

        EXPECT_LT(rel_error, 1e-5f) << "Relative error too large: " << rel_error;
    }

    TEST_F(Test__FusedSwiGLU, SingleToken)
    {
        m_ = 1;
        gate_fp32_.resize(m_ * n_);
        up_fp32_.resize(m_ * n_);
        output_fp32_.resize(m_ * n_);
        output_ref_.resize(m_ * n_);

        fill_random_fp32(gate_fp32_.data(), m_ * n_);
        fill_random_fp32(up_fp32_.data(), m_ * n_);

        FusedSwiGLU kernel;

        bool success = kernel.execute(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_fp32_.data(),
            m_, n_);

        ASSERT_TRUE(success);

        // Compute reference and compare
        reference_swiglu(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_ref_.data(),
            m_, n_);

        float rel_error = compute_relative_error(
            output_fp32_.data(), output_ref_.data(), m_ * n_);

        EXPECT_LT(rel_error, 1e-5f);
    }

    TEST_F(Test__FusedSwiGLU, LargeBatch)
    {
        m_ = 128;
        gate_fp32_.resize(m_ * n_);
        up_fp32_.resize(m_ * n_);
        output_fp32_.resize(m_ * n_);
        output_ref_.resize(m_ * n_);

        fill_random_fp32(gate_fp32_.data(), m_ * n_);
        fill_random_fp32(up_fp32_.data(), m_ * n_);

        FusedSwiGLU kernel;

        bool success = kernel.execute(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_fp32_.data(),
            m_, n_);

        ASSERT_TRUE(success);

        reference_swiglu(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_ref_.data(),
            m_, n_);

        float rel_error = compute_relative_error(
            output_fp32_.data(), output_ref_.data(), m_ * n_);

        EXPECT_LT(rel_error, 1e-5f);
    }

    TEST_F(Test__FusedSwiGLU, InPlace)
    {
        // Test in-place operation (output aliases gate)
        FusedSwiGLU kernel;

        // Make a copy of gate for reference computation
        std::vector<float> gate_copy = gate_fp32_;

        // Compute reference first
        reference_swiglu(
            gate_copy.data(),
            up_fp32_.data(),
            output_ref_.data(),
            m_, n_);

        // Execute kernel with output aliasing gate
        bool success = kernel.execute(
            gate_fp32_.data(),
            up_fp32_.data(),
            gate_fp32_.data(), // output aliases gate
            m_, n_);

        ASSERT_TRUE(success);

        float rel_error = compute_relative_error(
            gate_fp32_.data(), output_ref_.data(), m_ * n_);

        EXPECT_LT(rel_error, 1e-5f);
    }

    // =============================================================================
    // Error Handling Tests
    // =============================================================================

    TEST_F(Test__FusedSwiGLU, NullPointerHandling)
    {
        FusedSwiGLU kernel;

        // Null gate input
        EXPECT_FALSE(kernel.execute(
            nullptr,
            up_fp32_.data(),
            output_fp32_.data(),
            m_, n_));

        // Null up input
        EXPECT_FALSE(kernel.execute(
            gate_fp32_.data(),
            nullptr,
            output_fp32_.data(),
            m_, n_));

        // Null output
        EXPECT_FALSE(kernel.execute(
            gate_fp32_.data(),
            up_fp32_.data(),
            nullptr,
            m_, n_));
    }

    TEST_F(Test__FusedSwiGLU, InvalidDimensions)
    {
        FusedSwiGLU kernel;

        // Zero dimensions
        EXPECT_FALSE(kernel.execute(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_fp32_.data(),
            0, n_));

        EXPECT_FALSE(kernel.execute(
            gate_fp32_.data(),
            up_fp32_.data(),
            output_fp32_.data(),
            m_, 0));
    }

    // =============================================================================
    // Device Support Tests
    // =============================================================================

    TEST_F(Test__FusedSwiGLU, DeviceSupport)
    {
        FusedSwiGLU kernel;

        EXPECT_TRUE(kernel.supports_device(-1)); // CPU
        EXPECT_FALSE(kernel.supports_device(0)); // GPU not supported yet
    }

} // namespace llaminar2

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
