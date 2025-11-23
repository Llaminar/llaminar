/**
 * @file Test__Qwen2Pipeline_INT8BufferBug.cpp
 * @brief Unit test demonstrating the INT8 buffer bug in Qwen2Pipeline
 * @author David Sanftenberg
 * @date 2025-11-23
 *
 * BUG: Qwen2Pipeline::attention_block processes total_rows (batch_size * padded_seq_len)
 *      instead of effective_seq_len when calling fused RMSNorm + INT8 quantization.
 *
 * IMPACT: This causes the kernel to process garbage/uninitialized data beyond
 *         effective_seq_len, leading to divergence in batch padding tests.
 *
 * ROOT CAUSE: Lines 547-555 in Qwen2Pipeline.cpp:
 *     const int total_rows = batch_size_ * padded_seq_len_;  // ← BUG!
 *     rmsnorm_kernel->execute(..., total_rows, ...)
 *
 * FIX: Change to:
 *     rmsnorm_kernel->execute(..., effective_seq_len, ...)
 *
 * This test demonstrates that processing total_rows produces different results
 * than processing only effective_seq_len.
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
 * @test ProcessingTotalRowsVsEffectiveSeqLen
 *
 * Demonstrates that processing total_rows (batch_size * padded_seq_len)
 * produces different results than processing effective_seq_len.
 *
 * Setup:
 * - Buffer allocated for batch_size=2, padded_seq_len=512 (1024 rows)
 * - Valid data only in first effective_seq_len=16 rows (8 sequences * 2 tokens)
 * - Rows 16-1023 contain GARBAGE (uninitialized or zeros)
 *
 * Bug manifestation:
 * - Fused RMSNorm processes ALL 1024 rows (including garbage)
 * - This corrupts the per-row scales buffer
 * - Dequantization then uses corrupted scales
 * - Result: Divergence in E2E tests
 *
 * Expected:
 * - Processing only effective_seq_len=16 rows produces correct results
 * - Processing all 1024 rows produces incorrect/divergent results
 */
TEST_F(Test__Qwen2Pipeline_INT8BufferBug, ProcessingTotalRowsVsEffectiveSeqLen)
{
    // Simulate batched execution parameters
    const int batch_size = 2;
    const int padded_seq_len = 512;   // max_seq_len (buffer capacity)
    const int effective_seq_len = 16; // batch_size * actual_tokens (8 seqs * 2 tokens)
    const int d_model = 896;          // Qwen 0.5B
    const float epsilon = 1e-6f;
    const int device_idx = 0;

    // Total buffer capacity
    const int total_rows = batch_size * padded_seq_len; // 1024

    // Allocate buffer with FULL capacity (mimics activation_buffers_)
    std::vector<float> input_buffer(total_rows * d_model, 0.0f);
    std::vector<float> gamma(d_model, 1.0f);

    // Initialize ONLY effective_seq_len rows with valid data
    for (int r = 0; r < effective_seq_len; ++r)
    {
        for (int c = 0; c < d_model; ++c)
        {
            input_buffer[r * d_model + c] = 1.0f + static_cast<float>(r * d_model + c) * 0.001f;
        }
    }
    // Rows [16:1024] remain ZERO (garbage simulation)

    // Scenario 1: Process ONLY effective_seq_len (CORRECT behavior)
    std::vector<int8_t> output_correct(total_rows * d_model, 0);
    std::vector<float> scales_correct(total_rows, 1.0f);

    FusedRMSNormQuantize kernel_correct;
    bool success_correct = kernel_correct.execute(
        input_buffer.data(), gamma.data(),
        output_correct.data(), scales_correct.data(),
        effective_seq_len, d_model, // ← Process only effective_seq_len
        epsilon, mpi_ctx_.get(), device_idx);
    ASSERT_TRUE(success_correct) << "Correct kernel execution failed";

    // Scenario 2: Process total_rows (BUG - current Qwen2Pipeline behavior)
    std::vector<int8_t> output_buggy(total_rows * d_model, 0);
    std::vector<float> scales_buggy(total_rows, 1.0f);

    FusedRMSNormQuantize kernel_buggy;
    bool success_buggy = kernel_buggy.execute(
        input_buffer.data(), gamma.data(),
        output_buggy.data(), scales_buggy.data(),
        total_rows, d_model, // ← BUG: Process all total_rows (including garbage)
        epsilon, mpi_ctx_.get(), device_idx);
    ASSERT_TRUE(success_buggy) << "Buggy kernel execution failed";

    // CRITICAL COMPARISON: Scales for valid rows should differ
    // because buggy version processes garbage rows, affecting scale normalization

    std::cout << "\n[BUG DEMONSTRATION: SCALE COMPARISON]\n";
    std::cout << "Comparing scales for first 16 (valid) rows:\n";

    bool scales_differ = false;
    float max_scale_diff = 0.0f;

    for (int r = 0; r < effective_seq_len; ++r)
    {
        float diff = std::abs(scales_correct[r] - scales_buggy[r]);
        max_scale_diff = std::max(max_scale_diff, diff);

        if (diff > 1e-6f)
        {
            scales_differ = true;
        }

        if (r < 5)
        {
            std::cout << "  Row " << r << ": correct=" << scales_correct[r]
                      << ", buggy=" << scales_buggy[r]
                      << ", diff=" << diff << "\n";
        }
    }

    std::cout << "Max scale difference: " << max_scale_diff << "\n";

    // CRITICAL ASSERTIONS:
    // 1. Scales for padding rows in buggy version should be very small (near epsilon)
    //    because those rows are all zeros
    std::cout << "\n[PADDING ROW SCALES (rows 16-20)]:\n";
    for (int r = effective_seq_len; r < effective_seq_len + 5; ++r)
    {
        std::cout << "  Row " << r << " (padding): scale_buggy=" << scales_buggy[r] << "\n";
        EXPECT_LT(scales_buggy[r], 1e-4f)
            << "Padding row " << r << " should have near-zero scale in buggy version";
    }

    // 2. If scales differ, dequantization will produce different results
    if (scales_differ)
    {
        std::cout << "\n✅ BUG CONFIRMED: Processing total_rows produces different scales\n";
        std::cout << "   This explains the E2E batch padding divergence!\n";
    }
    else
    {
        std::cout << "\n❌ BUG NOT REPRODUCED: Scales are identical (unexpected)\n";
        FAIL() << "Expected scales to differ between correct and buggy execution";
    }

    // 3. Verify INT8 outputs differ for valid rows
    bool outputs_differ = false;
    int diff_count = 0;

    for (int r = 0; r < effective_seq_len && diff_count < 10; ++r)
    {
        for (int c = 0; c < d_model && diff_count < 10; ++c)
        {
            int idx = r * d_model + c;
            if (output_correct[idx] != output_buggy[idx])
            {
                if (diff_count < 5)
                {
                    std::cout << "INT8 diff at [" << r << "," << c << "]: "
                              << "correct=" << static_cast<int>(output_correct[idx])
                              << ", buggy=" << static_cast<int>(output_buggy[idx]) << "\n";
                }
                outputs_differ = true;
                diff_count++;
            }
        }
    }

    if (outputs_differ)
    {
        std::cout << "✅ INT8 outputs also differ (found " << diff_count << " differences)\n";
    }

    // EXPECTED RESULT: This test should PASS, demonstrating the bug exists
    EXPECT_TRUE(scales_differ) << "Scales should differ when processing total_rows vs effective_seq_len";
}

/**
 * @test DequantizationDivergence
 *
 * Demonstrates that using corrupted scales from total_rows processing
 * causes dequantization divergence.
 */
TEST_F(Test__Qwen2Pipeline_INT8BufferBug, DequantizationDivergence)
{
    const int batch_size = 2;
    const int padded_seq_len = 512;
    const int effective_seq_len = 16;
    const int d_model = 896;
    const float epsilon = 1e-6f;
    const int device_idx = 0;
    const int total_rows = batch_size * padded_seq_len;

    // Create input
    std::vector<float> input_buffer(total_rows * d_model, 0.0f);
    std::vector<float> gamma(d_model, 1.0f);

    for (int r = 0; r < effective_seq_len; ++r)
    {
        for (int c = 0; c < d_model; ++c)
        {
            input_buffer[r * d_model + c] = 1.0f + static_cast<float>(r) * 0.1f;
        }
    }

    // Process with correct row count
    std::vector<int8_t> int8_correct(total_rows * d_model, 0);
    std::vector<float> scales_correct(total_rows, 1.0f);

    FusedRMSNormQuantize kernel1;
    ASSERT_TRUE(kernel1.execute(
        input_buffer.data(), gamma.data(),
        int8_correct.data(), scales_correct.data(),
        effective_seq_len, d_model, epsilon,
        mpi_ctx_.get(), device_idx));

    // Process with buggy row count
    std::vector<int8_t> int8_buggy(total_rows * d_model, 0);
    std::vector<float> scales_buggy(total_rows, 1.0f);

    FusedRMSNormQuantize kernel2;
    ASSERT_TRUE(kernel2.execute(
        input_buffer.data(), gamma.data(),
        int8_buggy.data(), scales_buggy.data(),
        total_rows, d_model, epsilon,
        mpi_ctx_.get(), device_idx));

    // Dequantize both
    std::vector<float> dequant_correct(total_rows * d_model);
    std::vector<float> dequant_buggy(total_rows * d_model);

    for (int r = 0; r < effective_seq_len; ++r)
    {
        for (int c = 0; c < d_model; ++c)
        {
            int idx = r * d_model + c;
            dequant_correct[idx] = static_cast<float>(int8_correct[idx]) * scales_correct[r];
            dequant_buggy[idx] = static_cast<float>(int8_buggy[idx]) * scales_buggy[r];
        }
    }

    // Compare dequantized outputs
    float max_diff = 0.0f;
    float sum_diff = 0.0f;
    int diff_count = 0;

    for (int r = 0; r < effective_seq_len; ++r)
    {
        for (int c = 0; c < d_model; ++c)
        {
            int idx = r * d_model + c;
            float diff = std::abs(dequant_correct[idx] - dequant_buggy[idx]);
            max_diff = std::max(max_diff, diff);
            sum_diff += diff;

            if (diff > 0.01f)
            {
                diff_count++;
            }
        }
    }

    float mean_diff = sum_diff / (effective_seq_len * d_model);

    std::cout << "\n[DEQUANTIZATION DIVERGENCE]\n";
    std::cout << "Max diff: " << max_diff << "\n";
    std::cout << "Mean diff: " << mean_diff << "\n";
    std::cout << "Values with diff > 0.01: " << diff_count << " / " << (effective_seq_len * d_model) << "\n";

    // This demonstrates the bug causes dequantization divergence
    EXPECT_GT(max_diff, 0.001f) << "Expected dequantization divergence due to corrupted scales";
    std::cout << "✅ CONFIRMED: Processing total_rows corrupts scales, causing dequantization divergence\n";
}
