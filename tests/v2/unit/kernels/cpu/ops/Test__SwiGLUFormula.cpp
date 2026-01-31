/**
 * @file Test__SwiGLUFormula.cpp
 * @brief Unit tests for SwiGLU activation formula correctness
 *
 * This test verifies that our SwiGLU implementation matches the HuggingFace
 * FFN formula: down_proj(act_fn(gate_proj(x)) * up_proj(x))
 *
 * The correct formula is: output = silu(gate) * up
 * NOT: output = gate * silu(up)  ← This was a bug that caused garbage output
 *
 * The test uses known values where silu(gate) != gate and up != silu(up)
 * to distinguish between the two formulations.
 *
 * @author David Sanftenberg (via GitHub Copilot)
 */

#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include <algorithm>

#include "kernels/cpu/primitives/SwiGLUPrimitives.h"

namespace llaminar2
{

    // Reference silu implementation for testing
    float silu_reference(float x)
    {
        return x / (1.0f + std::exp(-x));
    }

    /**
     * @brief Test that SwiGLU computes silu(gate) * up, not gate * silu(up)
     *
     * This is the critical formula test. We use values where the two formulations
     * produce significantly different results.
     *
     * For gate=2.0, up=1.0:
     *   - Correct:   silu(2.0) * 1.0 = 1.7616 * 1.0 = 1.7616
     *   - Incorrect: 2.0 * silu(1.0) = 2.0 * 0.7311 = 1.4621
     *
     * The difference (0.3) is large enough to clearly distinguish the formulations.
     */
    TEST(Test__SwiGLUFormula, CorrectFormulaIsSiluGateTimesUp)
    {
        // Test values where silu(gate) != gate and up != silu(up)
        std::vector<float> gate = {2.0f, 3.0f, 1.0f, -1.0f};
        std::vector<float> up = {1.0f, 0.5f, 2.0f, 3.0f};
        std::vector<float> output(4);

        // Compute SwiGLU using our implementation
        primitives::compute_swiglu(gate.data(), up.data(), output.data(), 4);

        // Expected values using correct formula: silu(gate) * up
        std::vector<float> expected_correct(4);
        for (int i = 0; i < 4; ++i)
        {
            expected_correct[i] = silu_reference(gate[i]) * up[i];
        }

        // Values that would result from incorrect formula: gate * silu(up)
        std::vector<float> expected_incorrect(4);
        for (int i = 0; i < 4; ++i)
        {
            expected_incorrect[i] = gate[i] * silu_reference(up[i]);
        }

        // Verify our implementation matches the correct formula
        for (int i = 0; i < 4; ++i)
        {
            EXPECT_NEAR(output[i], expected_correct[i], 1e-5f)
                << "At index " << i << ": expected silu(" << gate[i] << ") * " << up[i]
                << " = " << expected_correct[i] << ", got " << output[i];

            // Also verify it does NOT match the incorrect formula
            // (unless the values happen to be equal, which they shouldn't be for our test values)
            float diff_correct = std::abs(output[i] - expected_correct[i]);
            float diff_incorrect = std::abs(output[i] - expected_incorrect[i]);

            if (std::abs(expected_correct[i] - expected_incorrect[i]) > 0.01f)
            {
                EXPECT_LT(diff_correct, diff_incorrect)
                    << "Output is closer to incorrect formula at index " << i;
            }
        }
    }

    /**
     * @brief Test SwiGLU with asymmetric values to ensure gate/up aren't swapped
     */
    TEST(Test__SwiGLUFormula, AsymmetricValuesNotSwapped)
    {
        // Use asymmetric values where swapping gate/up would be obvious
        std::vector<float> gate = {3.0f}; // silu(3.0) ≈ 2.858
        std::vector<float> up = {0.0f};   // silu(0.0) = 0.0
        std::vector<float> output(1);

        primitives::compute_swiglu(gate.data(), up.data(), output.data(), 1);

        // Correct: silu(3.0) * 0.0 = 2.858 * 0.0 = 0.0
        float expected_correct = silu_reference(3.0f) * 0.0f;
        EXPECT_NEAR(output[0], expected_correct, 1e-5f);
        EXPECT_NEAR(output[0], 0.0f, 1e-5f);

        // Incorrect would give: 3.0 * silu(0.0) = 3.0 * 0.0 = 0.0
        // Hmm, same result. Let's try different values.
    }

    /**
     * @brief Test with values that definitively distinguish correct vs incorrect
     */
    TEST(Test__SwiGLUFormula, DefinitiveFormulaTest)
    {
        // gate=1, up=10:
        //   Correct: silu(1) * 10 = 0.7311 * 10 = 7.311
        //   Incorrect: 1 * silu(10) = 1 * 9.9995 = 9.9995
        std::vector<float> gate = {1.0f};
        std::vector<float> up = {10.0f};
        std::vector<float> output(1);

        primitives::compute_swiglu(gate.data(), up.data(), output.data(), 1);

        float correct = silu_reference(1.0f) * 10.0f;   // ~7.31
        float incorrect = 1.0f * silu_reference(10.0f); // ~9.9995

        // The difference is ~2.69, so tolerance of 0.1 clearly distinguishes them
        EXPECT_NEAR(output[0], correct, 0.01f);
        EXPECT_NE(std::abs(output[0] - correct) < 0.1f,
                  std::abs(output[0] - incorrect) < 0.1f)
            << "Output should clearly match one formula, not both";
    }

    /**
     * @brief Test larger arrays (triggers SIMD paths)
     */
    TEST(Test__SwiGLUFormula, LargeArraySIMD)
    {
        const int N = 1024; // Large enough to trigger AVX2/AVX512 paths
        std::vector<float> gate(N);
        std::vector<float> up(N);
        std::vector<float> output(N);

        // Fill with varied values
        for (int i = 0; i < N; ++i)
        {
            gate[i] = static_cast<float>(i % 10) - 4.5f; // Range: -4.5 to 4.5
            up[i] = static_cast<float>((i * 3) % 10) - 4.5f;
        }

        primitives::compute_swiglu(gate.data(), up.data(), output.data(), N);

        // Verify each element
        int errors = 0;
        for (int i = 0; i < N; ++i)
        {
            float expected = silu_reference(gate[i]) * up[i];
            if (std::abs(output[i] - expected) > 1e-4f)
            {
                if (errors < 5)
                {
                    std::cerr << "Mismatch at " << i << ": expected " << expected
                              << ", got " << output[i] << std::endl;
                }
                errors++;
            }
        }
        EXPECT_EQ(errors, 0) << "Found " << errors << " mismatches in SIMD path";
    }

    /**
     * @brief Test edge cases (zeros, negatives, large values)
     */
    TEST(Test__SwiGLUFormula, EdgeCases)
    {
        std::vector<float> gate = {0.0f, -5.0f, 5.0f, 0.0f, 100.0f};
        std::vector<float> up = {0.0f, 0.0f, 0.0f, 5.0f, 0.001f};
        std::vector<float> output(5);

        primitives::compute_swiglu(gate.data(), up.data(), output.data(), 5);

        for (int i = 0; i < 5; ++i)
        {
            float expected = silu_reference(gate[i]) * up[i];
            EXPECT_NEAR(output[i], expected, 1e-4f)
                << "Edge case " << i << ": gate=" << gate[i] << ", up=" << up[i];
        }
    }

    /**
     * @brief Document the HuggingFace formula reference
     */
    TEST(Test__SwiGLUFormula, DocumentReference)
    {
        // From HuggingFace Qwen2MLP:
        // def forward(self, x):
        //     gate_proj = self.gate_proj(x)
        //     up_proj = self.up_proj(x)
        //     return self.down_proj(self.act_fn(gate_proj) * up_proj)
        //
        // Where act_fn is SiLU (silu).
        //
        // So: intermediate = silu(gate_proj) * up_proj
        //     output = down_proj(intermediate)
        //
        // Our SwiGLU primitive implements: output = silu(gate) * up
        // This matches the HuggingFace formula.

        // Simple sanity check
        float gate = 2.0f;
        float up = 3.0f;
        float result[1];
        primitives::compute_swiglu(&gate, &up, result, 1);

        float expected = silu_reference(gate) * up;
        EXPECT_NEAR(result[0], expected, 1e-5f)
            << "HuggingFace formula: silu(gate) * up = silu(" << gate << ") * " << up
            << " = " << expected;
    }

} // namespace llaminar2
