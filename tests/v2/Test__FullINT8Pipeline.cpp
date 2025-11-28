/**
 * @file Test__FullINT8Pipeline.cpp
 * @brief Test full INT8 pipeline: FP32 → INT8 → [INT8 layers with RMSNorm] → FP32
 * @author David Sanftenberg
 * @date 2025-11-05 (updated 2025-11-06)
 *
 * Demonstrates the full INT8 inference pipeline where:
 * 1. Input: FP32 → INT8 (quantize once)
 * 2. Layer 0: INT8×INT8 → INT32 → RMSNorm → INT8
 * 3. Layer 1: INT8×INT8 → INT32 → RMSNorm → INT8
 * 4. Output: INT32 → FP32 (dequantize once)
 *
 * Benefits:
 * - 4× memory reduction (INT8 vs FP32 activations)
 * - No per-layer quant/dequant overhead
 * - INT32 accumulator prevents overflow
 * - RMSNorm operates on INT32, requantizes to INT8
 */

#include <gtest/gtest.h>
#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/kernels/cpu/INT8GemmKernel.h"
#include "../../src/v2/kernels/cpu/ops/CPURMSNormKernelT.h"
#include "../../src/v2/utils/Logger.h"
#include <vector>
#include <cmath>
#include <algorithm>

using namespace llaminar2;

class FullINT8Pipeline : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Dimensions for simple 2-layer network
        m_ = 4; // batch size
        k_ = 8; // hidden dim (input)
        n_ = 6; // hidden dim (output)

        // Create FP32 input activations
        std::vector<float> fp32_input(m_ * k_, 1.5f);
        for (size_t i = 0; i < fp32_input.size(); ++i)
        {
            fp32_input[i] += 0.1f * static_cast<float>(i % k_); // Add variation
        }

        input_fp32_ = fp32_input;

        // Create INT8 weights for Layer 0 (simulate quantized weights)
        // When transpose_B=true, OneDNN expects weights in [n, k] layout
        std::vector<int8_t> layer0_weights_int8(n_ * k_);
        std::vector<float> layer0_col_scales(n_);
        for (size_t j = 0; j < n_; ++j)
        {
            float scale = 0.02f + 0.001f * j; // Different scale per column
            layer0_col_scales[j] = scale;
            for (size_t i = 0; i < k_; ++i)
            {
                // Quantized weights: small values around 0
                // Store in [n, k] layout for transpose_B=true
                layer0_weights_int8[j * k_ + i] = static_cast<int8_t>(10 - (i + j) % 20);
            }
        }

        layer0_weight_ = std::make_shared<INT8Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(k_)}, // [n, k] for transpose_B=true
            layer0_weights_int8,
            1.0f);
        layer0_weight_->set_col_scales(layer0_col_scales);

        // Create INT8 weights for Layer 1 (n × n)
        std::vector<int8_t> layer1_weights_int8(n_ * n_);
        std::vector<float> layer1_col_scales(n_);
        for (size_t j = 0; j < n_; ++j)
        {
            float scale = 0.015f + 0.002f * j;
            layer1_col_scales[j] = scale;
            for (size_t i = 0; i < n_; ++i)
            {
                layer1_weights_int8[i * n_ + j] = static_cast<int8_t>(5 - (i * j) % 15);
            }
        }

        layer1_weight_ = std::make_shared<INT8Tensor>(
            std::vector<size_t>{static_cast<size_t>(n_), static_cast<size_t>(n_)},
            layer1_weights_int8,
            1.0f);
        layer1_weight_->set_col_scales(layer1_col_scales);
    }

    int m_, k_, n_;
    std::vector<float> input_fp32_;
    std::shared_ptr<INT8Tensor> layer0_weight_;
    std::shared_ptr<INT8Tensor> layer1_weight_;
};

/**
 * @brief Demonstrate full INT8 pipeline with 2 layers
 *
 * Flow:
 *   FP32 input → quantize → INT8
 *   ↓
 *   Layer 0: INT8 × INT8 → INT32 → requantize → INT8
 *   ↓
 *   Layer 1: INT8 × INT8 → INT32 → dequantize → FP32
 *
 * This shows the key operations:
 * 1. Initial quantization (FP32 → INT8)
 * 2. INT8×INT8 GEMM → INT32
 * 3. Requantization (INT32 → INT8) between layers
 * 4. Final dequantization (INT32 → FP32)
 */
TEST_F(FullINT8Pipeline, TwoLayerFlow)
{
    LOG_INFO("=== Full INT8 Pipeline Demo ===");
    LOG_INFO("Input: FP32[" << m_ << "," << k_ << "]");

    // =========================================================================
    // LAYER 0: FP32 → INT8 → INT32
    // =========================================================================

    // Step 1: Quantize input to INT8 (per-row)
    std::vector<int8_t> input_int8(m_ * k_);
    std::vector<float> input_row_scales(m_);

    for (int i = 0; i < m_; ++i)
    {
        // Find max abs in row
        float max_abs = 0.0f;
        for (int j = 0; j < k_; ++j)
        {
            max_abs = std::max(max_abs, std::abs(input_fp32_[i * k_ + j]));
        }

        // Compute scale
        input_row_scales[i] = max_abs / 127.0f;
        float inv_scale = 1.0f / input_row_scales[i];

        // Quantize row
        for (int j = 0; j < k_; ++j)
        {
            float scaled = input_fp32_[i * k_ + j] * inv_scale;
            input_int8[i * k_ + j] = static_cast<int8_t>(std::round(
                std::clamp(scaled, -127.0f, 127.0f)));
        }
    }

    LOG_INFO("✓ Quantized input: INT8[" << m_ << "," << k_ << "] (scales: " << input_row_scales[0] << " ...)");

    // Step 2: Layer 0 GEMM: INT8×INT8 → INT32
    std::vector<int32_t> layer0_output_int32(m_ * n_);
    INT8GemmKernel layer0_gemm(layer0_weight_.get());

    bool success = layer0_gemm.multiply_int8_activations_int32(
        input_int8.data(),
        input_row_scales.data(),
        layer0_output_int32.data(),
        m_, n_, k_,
        true); // transpose_B=true

    ASSERT_TRUE(success) << "Layer 0 GEMM failed";
    LOG_INFO("✓ Layer 0 GEMM: INT8×INT8 → INT32[" << m_ << "," << n_ << "]");
    LOG_INFO("  Sample INT32 values: " << layer0_output_int32[0] << ", "
                                       << layer0_output_int32[1] << ", "
                                       << layer0_output_int32[2]);

    // =========================================================================
    // LAYER 0 RMSNORM: INT32 → INT8 (with normalization)
    // =========================================================================

    // Step 3: RMSNorm on INT32, output INT8 (replaces simple requantization)
    std::vector<int8_t> layer0_output_int8(m_ * n_);
    std::vector<float> layer0_output_row_scales(m_);
    std::vector<float> gamma0(n_, 1.0f); // Unity gamma for simplicity

    CPURMSNormKernelT<INT32Tensor> rmsnorm_kernel;
    success = rmsnorm_kernel.apply_int32_to_int8(
        layer0_output_int32.data(),
        gamma0.data(),
        layer0_output_int8.data(),
        layer0_output_row_scales.data(),
        m_, n_,
        1e-6f); // eps

    ASSERT_TRUE(success) << "Layer 0 RMSNorm failed";
    LOG_INFO("✓ Layer 0 RMSNorm: INT32 → INT8[" << m_ << "," << n_ << "] (scales: " << layer0_output_row_scales[0] << " ...)");
    LOG_INFO("  Sample INT8 values: " << static_cast<int>(layer0_output_int8[0]) << ", "
                                      << static_cast<int>(layer0_output_int8[1]) << ", "
                                      << static_cast<int>(layer0_output_int8[2]));

    // =========================================================================
    // LAYER 1: INT8 → INT32 → RMSNORM → INT8
    // =========================================================================

    // Step 4: Layer 1 GEMM: INT8×INT8 → INT32
    std::vector<int32_t> layer1_output_int32(m_ * n_);
    INT8GemmKernel layer1_gemm(layer1_weight_.get());

    success = layer1_gemm.multiply_int8_activations_int32(
        layer0_output_int8.data(),
        layer0_output_row_scales.data(),
        layer1_output_int32.data(),
        m_, n_, n_,
        true); // transpose_B=true

    ASSERT_TRUE(success) << "Layer 1 GEMM failed";
    LOG_INFO("✓ Layer 1 GEMM: INT8×INT8 → INT32[" << m_ << "," << n_ << "]");
    LOG_INFO("  Sample INT32 values: " << layer1_output_int32[0] << ", "
                                       << layer1_output_int32[1] << ", "
                                       << layer1_output_int32[2]);

    // Step 5: RMSNorm on Layer 1 output (INT32 → INT8)
    std::vector<int8_t> layer1_output_int8(m_ * n_);
    std::vector<float> layer1_output_row_scales(m_);
    std::vector<float> gamma1(n_, 1.0f);

    success = rmsnorm_kernel.apply_int32_to_int8(
        layer1_output_int32.data(),
        gamma1.data(),
        layer1_output_int8.data(),
        layer1_output_row_scales.data(),
        m_, n_,
        1e-6f);

    ASSERT_TRUE(success) << "Layer 1 RMSNorm failed";
    LOG_INFO("✓ Layer 1 RMSNorm: INT32 → INT8[" << m_ << "," << n_ << "]");

    // Step 6: Dequantize final INT8 output to FP32 for interpretation
    std::vector<float> final_output_fp32(m_ * n_);
    for (int i = 0; i < m_; ++i)
    {
        float scale = layer1_output_row_scales[i];
        for (int j = 0; j < n_; ++j)
        {
            final_output_fp32[i * n_ + j] = static_cast<float>(layer1_output_int8[i * n_ + j]) * scale;
        }
    }

    LOG_INFO("✓ Dequantized: INT8 → FP32[" << m_ << "," << n_ << "]");
    LOG_INFO("  Sample FP32 values: " << final_output_fp32[0] << ", "
                                      << final_output_fp32[1] << ", "
                                      << final_output_fp32[2]);

    // =========================================================================
    // SUMMARY
    // =========================================================================

    LOG_INFO("");
    LOG_INFO("=== Pipeline Summary ===");
    LOG_INFO("Input:       FP32 → INT8 (quantized ONCE)");
    LOG_INFO("Layer 0:     INT8×INT8 → INT32 → RMSNorm → INT8");
    LOG_INFO("Layer 1:     INT8×INT8 → INT32 → RMSNorm → INT8");
    LOG_INFO("Output:      INT8 → FP32 (dequantized ONCE)");
    LOG_INFO("");
    LOG_INFO("✓ Activations stayed in INT8/INT32 between layers");
    LOG_INFO("✓ RMSNorm operates on INT32, requantizes to INT8");
    LOG_INFO("✓ No per-layer FP32 conversion overhead");
    LOG_INFO("✓ 4× memory reduction vs FP32 activations");

    // Sanity check: output should be non-zero
    float sum = 0.0f;
    for (auto val : final_output_fp32)
    {
        sum += std::abs(val);
    }
    EXPECT_GT(sum, 0.0f) << "Output should be non-zero";
}

/**
 * @brief Test INT32Tensor requantization correctness
 *
 * Validates that requantize_to_int8() correctly:
 * 1. Computes per-row scales to fit INT32 range into INT8 [-127, 127]
 * 2. Quantizes INT32 values accurately
 * 3. Allows reconstruction via dequantization
 */
TEST_F(FullINT8Pipeline, INT32ToINT8Requantization)
{
    LOG_INFO("=== Testing INT32 → INT8 Requantization ===");

    // Create test INT32 data with known range
    std::vector<int32_t> int32_data = {
        1000, 2000, 3000, 4000, 5000, 6000,      // Row 0: max=6000
        -500, -1000, -1500, -2000, -2500, -3000, // Row 1: max=3000
        100, 200, 300, 400, 500, 600,            // Row 2: max=600
        10000, 20000, 30000, 40000, 50000, 60000 // Row 3: max=60000
    };

    auto int32_tensor = std::make_shared<INT32Tensor>(
        std::vector<size_t>{4, 6},
        int32_data);

    // Requantize to INT8
    std::vector<int8_t> int8_data(24);
    std::vector<float> row_scales(4);

    bool success = int32_tensor->requantize_to_int8(int8_data.data(), row_scales.data());
    ASSERT_TRUE(success);

    // Check scales are computed correctly
    EXPECT_NEAR(row_scales[0], 6000.0f / 127.0f, 1.0f);  // Row 0: max=6000
    EXPECT_NEAR(row_scales[1], 3000.0f / 127.0f, 1.0f);  // Row 1: max=3000
    EXPECT_NEAR(row_scales[2], 600.0f / 127.0f, 1.0f);   // Row 2: max=600
    EXPECT_NEAR(row_scales[3], 60000.0f / 127.0f, 1.0f); // Row 3: max=60000

    // Check INT8 values are in range [-127, 127]
    for (auto val : int8_data)
    {
        EXPECT_GE(val, -127);
        EXPECT_LE(val, 127);
    }

    // Check reconstruction accuracy (INT8 × scale should ≈ INT32)
    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t j = 0; j < 6; ++j)
        {
            size_t idx = i * 6 + j;
            float reconstructed = static_cast<float>(int8_data[idx]) * row_scales[i];
            float original = static_cast<float>(int32_data[idx]);
            float rel_error = std::abs(reconstructed - original) / (std::abs(original) + 1e-6f);
            EXPECT_LT(rel_error, 0.02f) << "Row " << i << " col " << j << ": "
                                        << "original=" << original << ", "
                                        << "reconstructed=" << reconstructed;
        }
    }

    LOG_INFO("✓ Requantization passed: scales computed correctly, values in range, reconstruction accurate");
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
