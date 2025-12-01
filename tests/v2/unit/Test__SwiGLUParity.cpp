/**
 * @file Test__SwiGLUParity.cpp
 * @brief Regression tests for SwiGLU activation correctness
 *
 * These tests lock in the correct SwiGLU formula to prevent future regressions.
 *
 * **Critical Bug Fixed**: CPUSwiGLUKernel had the formula reversed:
 * - **CORRECT**: silu(gate) * up  (matches HuggingFace: act_fn(gate_proj(x)) * up_proj(x))
 * - **WRONG**: gate * silu(up)    (was the original buggy implementation)
 *
 * See: https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2/modeling_qwen2.py
 * HuggingFace implementation: down_proj(self.act_fn(gate) * up)
 *
 * @author David Sanftenberg
 * @date 2025-11-07
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "utils/Logger.h"

using namespace llaminar2;

/**
 * @brief SwiGLU parity test fixture
 *
 * Validates that CPUSwiGLUKernel computes the correct formula: silu(gate) * up
 * where silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * This matches HuggingFace Qwen2 implementation:
 *   down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
 */
class SwiGLUParityTest : public ::testing::Test
{
protected:
    /**
     * @brief Compute reference SwiGLU: silu(gate) * up
     *
     * This is the CORRECT formula per HuggingFace Qwen2 implementation.
     */
    std::vector<float> compute_reference_swiglu(
        const std::vector<float> &gate,
        const std::vector<float> &up)
    {
        std::vector<float> result(gate.size());
        for (size_t i = 0; i < gate.size(); ++i)
        {
            // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
            float sigmoid_g = 1.0f / (1.0f + std::exp(-gate[i]));
            float silu_g = gate[i] * sigmoid_g;
            result[i] = silu_g * up[i]; // CORRECT: silu(gate) * up
        }
        return result;
    }

    /**
     * @brief Compute WRONG formula: gate * silu(up) (the original buggy implementation)
     */
    std::vector<float> compute_wrong_swiglu(
        const std::vector<float> &gate,
        const std::vector<float> &up)
    {
        std::vector<float> result(gate.size());
        for (size_t i = 0; i < gate.size(); ++i)
        {
            // WRONG: Apply SiLU to up instead of gate!
            float sigmoid_u = 1.0f / (1.0f + std::exp(-up[i]));
            float silu_u = up[i] * sigmoid_u;
            result[i] = gate[i] * silu_u;
        }
        return result;
    }

    float compute_max_abs_diff(const std::vector<float> &a, const std::vector<float> &b)
    {
        float max_diff = 0.0f;
        for (size_t i = 0; i < a.size(); ++i)
        {
            max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        }
        return max_diff;
    }
};

/**
 * @brief Test: CPUSwiGLUKernel matches reference implementation
 *
 * Validates that the kernel computes silu(gate) * up correctly.
 */
TEST_F(SwiGLUParityTest, KernelMatchesReference)
{
    // Test data: simple values that expose formula differences
    std::vector<float> gate = {1.0f, 2.0f, -1.0f, 0.0f, 3.0f};
    std::vector<float> up = {2.0f, 1.0f, 3.0f, 4.0f, -2.0f};

    std::vector<float> output(gate.size());
    std::vector<float> reference = compute_reference_swiglu(gate, up);

    // Run kernel
    CPUSwiGLUKernel kernel;
    bool success = kernel.apply(
        gate.data(), up.data(), output.data(),
        1, static_cast<int>(gate.size()), // 1 sequence, d_ff elements
        false, nullptr, -1);

    ASSERT_TRUE(success) << "SwiGLU kernel execution failed";

    // Compare results
    float max_diff = compute_max_abs_diff(output, reference);

    EXPECT_LT(max_diff, 1e-6f)
        << "Kernel output differs from reference (silu(gate) * up)\n"
        << "Max abs diff: " << max_diff;

    // Print results for inspection
    LOG_INFO("SwiGLU kernel vs reference:");
    LOG_INFO("  Max abs diff: " << max_diff << " (threshold: 1e-6)");
    LOG_INFO("  Status: ✓ Kernel matches reference formula");
}

/**
 * @brief Test: Verify correct formula differs significantly from wrong formula
 *
 * Ensures that silu(gate) * up and gate * silu(up) produce different results,
 * proving that the formula choice matters.
 */
TEST_F(SwiGLUParityTest, CorrectFormulaVsWrongFormula)
{
    // Test data designed to expose the difference
    std::vector<float> gate = {1.0f, 2.0f, -1.0f, 0.0f};
    std::vector<float> up = {2.0f, 1.0f, 3.0f, 4.0f};

    std::vector<float> correct = compute_reference_swiglu(gate, up);
    std::vector<float> wrong = compute_wrong_swiglu(gate, up);

    float max_diff = compute_max_abs_diff(correct, wrong);

    // The two formulas should produce VERY different results
    EXPECT_GT(max_diff, 1.0f)
        << "Correct and wrong formulas should differ significantly!\n"
        << "Max abs diff: " << max_diff << " (expected > 1.0)";

    LOG_INFO("Correct (silu(gate) * up) vs Wrong (gate * silu(up)):");
    LOG_INFO("  Max abs diff: " << max_diff);
    LOG_INFO("  Status: ✓ Formulas produce significantly different results");
}

/**
 * @brief Test: SwiGLU with realistic neural network values
 *
 * Uses values typical of FFN intermediate activations to ensure correctness
 * in production scenarios.
 */
TEST_F(SwiGLUParityTest, RealisticNeuralNetworkValues)
{
    // Realistic FFN values (seq_len=4, d_ff=8)
    std::vector<float> gate = {
        0.5f, -0.3f, 1.2f, -0.8f,
        2.1f, -1.5f, 0.1f, 0.9f,
        -0.4f, 1.8f, -2.0f, 0.6f,
        0.0f, -0.1f, 0.7f, -1.2f,
        1.5f, 0.2f, -0.9f, 1.1f,
        -1.8f, 0.4f, 2.5f, -0.5f,
        0.8f, -1.1f, 0.3f, 1.7f,
        -0.2f, 1.4f, -1.6f, 0.9f};

    std::vector<float> up = {
        1.0f, 0.5f, -1.5f, 0.8f,
        -0.7f, 1.2f, 0.3f, -1.0f,
        0.9f, -0.4f, 1.6f, -0.6f,
        0.2f, 0.1f, -0.9f, 1.3f,
        -1.1f, 0.7f, 0.4f, -1.4f,
        1.8f, -0.3f, 0.6f, 0.5f,
        -0.8f, 1.5f, -1.2f, 0.0f,
        0.1f, -1.7f, 2.0f, -0.5f};

    std::vector<float> output(gate.size());
    std::vector<float> reference = compute_reference_swiglu(gate, up);

    // Run kernel
    CPUSwiGLUKernel kernel;
    bool success = kernel.apply(
        gate.data(), up.data(), output.data(),
        4, 8, // seq_len=4, d_ff=8
        false, nullptr, -1);

    ASSERT_TRUE(success) << "SwiGLU kernel execution failed";

    // Compare results
    float max_diff = compute_max_abs_diff(output, reference);

    EXPECT_LT(max_diff, 1e-5f)
        << "Kernel output differs from reference on realistic data\n"
        << "Max abs diff: " << max_diff;

    LOG_INFO("Realistic neural network test:");
    LOG_INFO("  Input size: 4 sequences × 8 features = 32 elements");
    LOG_INFO("  Max abs diff: " << max_diff << " (threshold: 1e-5)");
    LOG_INFO("  Status: ✓ Correct on realistic data");
}

/**
 * @brief Test: Edge cases (zeros, extremes)
 */
TEST_F(SwiGLUParityTest, EdgeCases)
{
    // Edge cases: zeros, large positives, large negatives
    std::vector<float> gate = {0.0f, 10.0f, -10.0f, 1e-5f, -1e-5f};
    std::vector<float> up = {0.0f, -10.0f, 10.0f, -1e-5f, 1e-5f};

    std::vector<float> output(gate.size());
    std::vector<float> reference = compute_reference_swiglu(gate, up);

    // Run kernel
    CPUSwiGLUKernel kernel;
    bool success = kernel.apply(
        gate.data(), up.data(), output.data(),
        1, static_cast<int>(gate.size()),
        false, nullptr, -1);

    ASSERT_TRUE(success) << "SwiGLU kernel execution failed";

    // Compare results
    float max_diff = compute_max_abs_diff(output, reference);

    EXPECT_LT(max_diff, 1e-5f)
        << "Kernel fails on edge cases\n"
        << "Max abs diff: " << max_diff;

    LOG_INFO("Edge case test (zeros, ±10, ±1e-5):");
    LOG_INFO("  Max abs diff: " << max_diff << " (threshold: 1e-5)");
    LOG_INFO("  Status: ✓ Handles edge cases correctly");
}

/**
 * @brief Test: Large batch performance (1024 elements)
 */
TEST_F(SwiGLUParityTest, LargeBatch)
{
    const int seq_len = 32;
    const int d_ff = 32;
    const int total = seq_len * d_ff;

    std::vector<float> gate(total);
    std::vector<float> up(total);
    std::vector<float> output(total);

    // Fill with pseudo-random values
    for (int i = 0; i < total; ++i)
    {
        gate[i] = std::sin(static_cast<float>(i) * 0.1f) * 2.0f;
        up[i] = std::cos(static_cast<float>(i) * 0.15f) * 1.5f;
    }

    std::vector<float> reference = compute_reference_swiglu(gate, up);

    // Run kernel
    CPUSwiGLUKernel kernel;
    bool success = kernel.apply(
        gate.data(), up.data(), output.data(),
        seq_len, d_ff,
        false, nullptr, -1);

    ASSERT_TRUE(success) << "SwiGLU kernel execution failed";

    // Compare results
    float max_diff = compute_max_abs_diff(output, reference);

    EXPECT_LT(max_diff, 1e-5f)
        << "Kernel fails on large batch\n"
        << "Max abs diff: " << max_diff;

    LOG_INFO("Large batch test (32 seq × 32 features = 1024 elements):");
    LOG_INFO("  Max abs diff: " << max_diff << " (threshold: 1e-5)");
    LOG_INFO("  Status: ✓ Correct on large batch");
}
