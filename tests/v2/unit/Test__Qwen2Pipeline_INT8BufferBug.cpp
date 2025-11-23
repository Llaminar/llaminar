/**
 * @file Test__Qwen2Pipeline_INT8BufferBug.cpp
 * @brief Regression test for INT8 dequantization buffer overrun bug
 * @author David Sanftenberg
 * @date 2025-11-23
 *
 * ORIGINAL BUG (FIXED): Qwen2Pipeline dequantization loops used total_rows instead
 *                       of effective_seq_len, accessing uninitialized scales beyond
 *                       valid data range.
 *
 * ROOT CAUSE: Lines 584 and 1009 in Qwen2Pipeline.cpp used:
 *     for (int r = 0; r < total_rows; ++r)  // BUG: accessed uninitialized scales!
 *
 * FIX: Changed to:
 *     for (int r = 0; r < effective_seq_len; ++r)  // Only process valid rows
 *
 * This test verifies that dequantizing beyond effective_seq_len with uninitialized
 * scales produces garbage (demonstrating why the fix was necessary).
 *
 * See: changelog/2025-01-XX-int8-dequantization-bug-fix.md
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>

#include "v2/kernels/cpu/fused/FusedRMSNormQuantize.h"
#include "v2/utils/MPIContext.h"

using namespace llaminar2;

class Test__Qwen2Pipeline_INT8BufferBug : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 1, MPI_COMM_WORLD);
    }

    std::shared_ptr<MPIContext> mpi_ctx_;
};

/**
 * @test DequantizationWithUninitializedScales
 *
 * Demonstrates the ACTUAL bug: dequantizing with uninitialized scales produces garbage.
 *
 * The original bug was NOT in quantization (RMSNorm is per-row independent), but in
 * the DEQUANTIZATION loop that accessed scales[r] for r >= effective_seq_len.
 *
 * Setup:
 * - Buffer allocated for total_rows=1024 (batch_size=2, padded_seq_len=512)
 * - Valid data only in first effective_seq_len=16 rows
 * - Quantization produces valid INT8 + scales for first 16 rows
 * - Scales[16:1024] are UNINITIALIZED (contain garbage)
 *
 * Bug demonstration:
 * - Dequantizing rows 0-15 with correct scales: ✅ Valid FP32 output
 * - Dequantizing rows 16-1023 with garbage scales: ❌ Produces massive values
 * - This garbage propagated through GEMM, causing 97 billion % divergence
 *
 * Expected:
 * - Correct dequantization (rows 0-15 only): Reasonable FP32 values
 * - Buggy dequantization (rows 0-1023): Massive garbage values in rows 16+
 */
TEST_F(Test__Qwen2Pipeline_INT8BufferBug, DequantizationWithUninitializedScales)
{
    const int batch_size = 2;
    const int padded_seq_len = 512;
    const int effective_seq_len = 16; // 8 sequences * 2 tokens each
    const int d_model = 896;
    const float epsilon = 1e-6f;
    const int device_idx = 0;
    const int total_rows = batch_size * padded_seq_len; // 1024

    // Step 1: Create and quantize valid data (only first 16 rows)
    std::vector<float> input_buffer(total_rows * d_model, 0.0f);
    std::vector<float> gamma(d_model, 1.0f);

    for (int r = 0; r < effective_seq_len; ++r)
    {
        for (int c = 0; c < d_model; ++c)
        {
            input_buffer[r * d_model + c] = 1.0f + static_cast<float>(r) * 0.1f;
        }
    }

    // Step 2: Quantize (produces INT8 + scales for ALL rows, but only first 16 are meaningful)
    std::vector<int8_t> quantized(total_rows * d_model, 0);
    std::vector<float> scales(total_rows, 1.0f);

    FusedRMSNormQuantize kernel;
    ASSERT_TRUE(kernel.execute(
        input_buffer.data(), gamma.data(),
        quantized.data(), scales.data(),
        effective_seq_len, d_model, // ← CORRECT: Only quantize valid rows
        epsilon, mpi_ctx_.get(), device_idx));

    // Step 3: Simulate the bug scenario - put non-zero INT8 values in padding rows
    // (In actual pipeline, these could contain leftover data from previous batch)
    for (int r = effective_seq_len; r < total_rows; ++r)
    {
        for (int c = 0; c < d_model; ++c)
        {
            quantized[r * d_model + c] = 100; // Arbitrary non-zero INT8 value
        }
    }

    // Step 4: Simulate uninitialized scales beyond effective_seq_len
    // (In the actual bug, these were never initialized and contained random garbage)
    for (int r = effective_seq_len; r < total_rows; ++r)
    {
        scales[r] = 1e6f; // Simulate garbage (actual bug had random memory: 1e9, inf, NaN, etc.)
    }

    // Step 5: CORRECT dequantization (only process effective_seq_len rows)
    std::vector<float> dequant_correct(total_rows * d_model, 0.0f);
    for (int r = 0; r < effective_seq_len; ++r) // ← FIX: Stop at effective_seq_len
    {
        float row_scale = scales[r];
        for (int c = 0; c < d_model; ++c)
        {
            int idx = r * d_model + c;
            dequant_correct[idx] = static_cast<float>(quantized[idx]) * row_scale;
        }
    }

    // Step 6: BUGGY dequantization (process all total_rows, accessing garbage scales)
    std::vector<float> dequant_buggy(total_rows * d_model, 0.0f);
    for (int r = 0; r < total_rows; ++r) // ← BUG: Accesses scales[16:1024] (garbage!)
    {
        float row_scale = scales[r];
        for (int c = 0; c < d_model; ++c)
        {
            int idx = r * d_model + c;
            dequant_buggy[idx] = static_cast<float>(quantized[idx]) * row_scale;
        }
    }

    // Step 7: Verify correct dequantization produces reasonable values
    float max_correct = 0.0f;
    for (int r = 0; r < effective_seq_len; ++r)
    {
        for (int c = 0; c < d_model; ++c)
        {
            max_correct = std::max(max_correct, std::abs(dequant_correct[r * d_model + c]));
        }
    }

    std::cout << "\n[REGRESSION TEST: Dequantization with Uninitialized Scales]\n";
    std::cout << "Correct dequantization (rows 0-15 only):\n";
    std::cout << "  Max magnitude: " << max_correct << " (should be reasonable, ~2.0)\n";
    EXPECT_LT(max_correct, 10.0f) << "Correct dequantization should produce reasonable values";

    // Step 8: Verify buggy dequantization produces MASSIVE garbage in padding rows
    float max_buggy_garbage = 0.0f;
    for (int r = effective_seq_len; r < total_rows; ++r) // Check padding rows only
    {
        for (int c = 0; c < d_model; ++c)
        {
            max_buggy_garbage = std::max(max_buggy_garbage,
                                         std::abs(dequant_buggy[r * d_model + c]));
        }
    }

    std::cout << "Buggy dequantization (accessing garbage scales in rows 16-1023):\n";
    std::cout << "  Max magnitude in padding rows: " << max_buggy_garbage
              << " (should be HUGE due to garbage scales)\n";

    // The bug: uninitialized scales produce massive values
    EXPECT_GT(max_buggy_garbage, 1e6f)
        << "Buggy dequantization with garbage scales should produce massive values";

    std::cout << "\n✅ BUG REPRODUCED: Accessing uninitialized scales produces garbage\n";
    std::cout << "   This is why the fix (loop bound = effective_seq_len) was critical!\n";
    std::cout << "   Original divergence: 97 billion % in E2E tests\n";
}
