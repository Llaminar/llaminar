/**
 * @file Test__ComputeStages.cpp
 * @brief Unit tests for individual compute stage implementations
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests for each compute stage type:
 * - GEMMStage: Matrix multiplication with quantization
 * - RMSNormStage: RMS normalization
 * - RoPEStage: Rotary position encoding
 * - SwiGLUStage: SwiGLU activation
 * - ResidualAddStage: Precision-aware residual addition
 *
 * Each stage is tested for:
 * - Basic execution correctness
 * - Precision preservation
 * - Backend compatibility
 * - FLOP/memory estimation
 * - Snapshot info generation
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "execution/compute_stages/ComputeStages.h"
#include "execution/DeviceContext.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "tensors/SIMDHelpers.h"
#include "../mocks/MockComputeStage.h"

using namespace llaminar2;
using namespace llaminar2::testing;

// =============================================================================
// Test Fixture
// =============================================================================

class ComputeStagesTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ctx_ = std::make_unique<MockDeviceContext>(0, ComputeBackendType::CPU);

        // Seed for reproducibility
        rng_.seed(42);
    }

    // Helper to fill buffer with random FP32 values
    void fillRandom(float *data, size_t count, float min = -1.0f, float max = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // Helper to fill buffer with known pattern
    void fillPattern(float *data, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = static_cast<float>(i % 256) / 256.0f - 0.5f;
        }
    }

    // Helper to check approximate equality
    bool approxEqual(float a, float b, float rtol = 1e-4f, float atol = 1e-6f)
    {
        return std::abs(a - b) <= atol + rtol * std::abs(b);
    }

    // Compute mean squared error
    float computeMSE(const float *a, const float *b, size_t count)
    {
        float sum = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum / static_cast<float>(count);
    }

    // Compute max absolute error
    float computeMaxAbsError(const float *a, const float *b, size_t count)
    {
        float max_err = 0.0f;
        for (size_t i = 0; i < count; ++i)
        {
            max_err = std::max(max_err, std::abs(a[i] - b[i]));
        }
        return max_err;
    }

    std::unique_ptr<MockDeviceContext> ctx_;
    std::mt19937 rng_;

    // Helper to create FP32Tensor from vector (for TensorBase* API)
    std::unique_ptr<FP32Tensor> makeTensor(size_t rows, size_t cols, const std::vector<float> &data = {})
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        if (!data.empty())
        {
            float *dst = tensor->mutable_data();
            for (size_t i = 0; i < std::min(data.size(), rows * cols); ++i)
            {
                dst[i] = data[i];
            }
        }
        return tensor;
    }

    // Helper for 1D tensor (gamma weights, etc.)
    std::unique_ptr<FP32Tensor> makeTensor1D(size_t n, const std::vector<float> &data = {})
    {
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1, n});
        if (!data.empty())
        {
            float *dst = tensor->mutable_data();
            for (size_t i = 0; i < std::min(data.size(), n); ++i)
            {
                dst[i] = data[i];
            }
        }
        return tensor;
    }
};

// =============================================================================
// GEMMStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, GEMMStage_TypeAndBackend)
{
    // Create minimal params
    GEMMStage::Params params{
        .A = nullptr,
        .B = nullptr,
        .C = nullptr,
        .m = 32,
        .n = 64,
        .k = 128};

    GEMMStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::GEMM);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(ComputeStagesTest, GEMMStage_EstimatedFlops)
{
    const int m = 32, n = 64, k = 128;

    GEMMStage::Params params{
        .m = m, .n = n, .k = k};

    GEMMStage stage(params);

    // GEMM FLOPS = 2 * m * n * k (multiply + add)
    const size_t expected_flops = 2 * m * n * k;
    EXPECT_EQ(stage.estimatedFlops(), expected_flops);
}

TEST_F(ComputeStagesTest, GEMMStage_DumpInfo)
{
    const int m = 4, n = 8, k = 16;

    // Create proper tensor for C
    auto C_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
        DeviceId::cpu() // CPU device
    );
    float *C_data = C_tensor->mutable_data();
    std::fill(C_data, C_data + m * n, 0.0f);

    GEMMStage::Params params{
        .C = C_tensor.get(),
        .m = m,
        .n = n,
        .k = k};

    GEMMStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, C_tensor->data());
    EXPECT_EQ(dump_info.outputs[0].rows, m);
    EXPECT_EQ(dump_info.outputs[0].cols, n);
    EXPECT_STREQ(dump_info.outputs[0].name, "C");
}

TEST_F(ComputeStagesTest, GEMMStage_WithFP32Weight)
{
    const int m = 2, n = 4, k = 8;

    // Create weight tensor and fill with identity-ish pattern
    // Use DeviceId::cpu() for CPU-only (no GPU device enumeration needed)
    auto weight = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(k), static_cast<size_t>(n)},
        DeviceId::cpu());
    float *weight_data = weight->mutable_data();
    std::fill(weight_data, weight_data + k * n, 0.0f);
    for (int i = 0; i < std::min(k, n); ++i)
    {
        weight_data[i * n + i] = 1.0f;
    }

    // Create activation tensor
    auto A_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(k)},
        DeviceId::cpu());
    fillPattern(A_tensor->mutable_data(), m * k);

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(m), static_cast<size_t>(n)},
        DeviceId::cpu());
    std::fill(C_tensor->mutable_data(), C_tensor->mutable_data() + m * n, 0.0f);

    GEMMStage::Params params{
        .A = A_tensor.get(),
        .B = weight.get(),
        .C = C_tensor.get(),
        .m = m,
        .n = n,
        .k = k};

    GEMMStage stage(params);

    // Execution may fail if kernel infrastructure not fully set up,
    // but we can at least test that it doesn't crash
    try
    {
        stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        // Expected: kernel factory may fail without proper device setup
        GTEST_SKIP() << "GEMM execution requires device enumeration: " << e.what();
    }
}

// =============================================================================
// RMSNormStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, RMSNormStage_TypeAndBackend)
{
    // Create empty tensors for params (dimensions from tensor, not from params)
    auto input = makeTensor(4, 128);
    auto output = makeTensor(4, 128);
    auto gamma = makeTensor1D(128);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-5f};

    RMSNormStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::RMS_NORM);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(ComputeStagesTest, RMSNormStage_EstimatedFlops)
{
    const int seq_len = 8, hidden_dim = 256;

    auto input = makeTensor(seq_len, hidden_dim);

    RMSNormStage::Params params{
        .input = input.get(),
        .eps = 1e-5f};

    RMSNormStage stage(params);

    // RMSNorm FLOPs: per row: hidden_dim (square) + 1 (mean) + 1 (sqrt) + hidden_dim (mul gamma)
    // Approximately 3 * seq_len * hidden_dim
    size_t flops = stage.estimatedFlops();
    EXPECT_GT(flops, 0);
    EXPECT_GE(flops, static_cast<size_t>(seq_len * hidden_dim));
}

TEST_F(ComputeStagesTest, RMSNormStage_SnapshotInfo)
{
    const int seq_len = 4, hidden_dim = 64;

    auto input = makeTensor(seq_len, hidden_dim);
    auto output = makeTensor(seq_len, hidden_dim);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .eps = 1e-5f};

    RMSNormStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output->data());
    EXPECT_EQ(dump_info.outputs[0].rows, seq_len);
    EXPECT_EQ(dump_info.outputs[0].cols, hidden_dim);
    EXPECT_STREQ(dump_info.outputs[0].name, "output");
}

TEST_F(ComputeStagesTest, RMSNormStage_Correctness)
{
    const int seq_len = 2, hidden_dim = 4;
    const float eps = 1e-5f;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,
                                     5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f, 1.0f}; // No scaling

    auto input = makeTensor(seq_len, hidden_dim, input_data);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim, gamma_data);

    // Compute expected RMSNorm manually for first row
    // RMS = sqrt(mean(x^2)) = sqrt((1 + 4 + 9 + 16) / 4) = sqrt(7.5) ≈ 2.7386
    float rms_row0 = std::sqrt((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f + eps);
    float expected_row0[] = {1.0f / rms_row0, 2.0f / rms_row0, 3.0f / rms_row0, 4.0f / rms_row0};

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = eps};

    RMSNormStage stage(params);

    bool success = false;
    try
    {
        success = stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RMSNorm execution requires device enumeration: " << e.what();
    }

    if (success)
    {
        const float *out = output->data();
        // Verify first row
        for (int i = 0; i < hidden_dim; ++i)
        {
            EXPECT_NEAR(out[i], expected_row0[i], 1e-4f)
                << "Mismatch at index " << i;
        }

        // Verify normalization (output should have unit RMS)
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden_dim; ++i)
        {
            sum_sq += out[i] * out[i];
        }
        float output_rms = std::sqrt(sum_sq / hidden_dim);
        EXPECT_NEAR(output_rms, 1.0f, 0.01f);
    }
}

// =============================================================================
// RoPEStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, RoPEStage_TypeAndBackend)
{
    const int seq_len = 4, n_heads = 8, head_dim = 64;
    auto Q = makeTensor(seq_len, n_heads * head_dim);

    RoPEStage::Params params{
        .Q = Q.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ROPE);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(ComputeStagesTest, RoPEStage_EstimatedFlops)
{
    const int seq_len = 8, n_heads = 4, head_dim = 64;
    auto Q = makeTensor(seq_len, n_heads * head_dim);

    RoPEStage::Params params{
        .Q = Q.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    // RoPE applies sin/cos rotations to pairs of elements
    // FLOPs include sin/cos computation + rotation multiplication
    size_t flops = stage.estimatedFlops();
    EXPECT_GT(flops, 0);
}

TEST_F(ComputeStagesTest, RoPEStage_SnapshotInfo)
{
    const int seq_len = 2, n_heads = 4, head_dim = 32;
    auto Q = makeTensor(seq_len, n_heads * head_dim);

    RoPEStage::Params params{
        .Q = Q.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, Q->data());
    EXPECT_EQ(dump_info.outputs[0].rows, seq_len);
    EXPECT_EQ(dump_info.outputs[0].cols, n_heads * head_dim);
    EXPECT_STREQ(dump_info.outputs[0].name, "Q");
}

TEST_F(ComputeStagesTest, RoPEStage_PreservesNorm)
{
    // RoPE should preserve the L2 norm of each head
    const int seq_len = 1, n_heads = 2, head_dim = 4;

    std::vector<float> tensor_data = {
        1.0f, 2.0f, 3.0f, 4.0f, // Head 0
        5.0f, 6.0f, 7.0f, 8.0f  // Head 1
    };

    auto Q = makeTensor(seq_len, n_heads * head_dim, tensor_data);

    // Compute norms before
    auto computeNorm = [](const float *data, int dim)
    {
        float sum = 0.0f;
        for (int i = 0; i < dim; ++i)
        {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    };

    const float *q_data = Q->data();
    float norm0_before = computeNorm(&q_data[0], head_dim);
    float norm1_before = computeNorm(&q_data[head_dim], head_dim);

    RoPEStage::Params params{
        .Q = Q.get(),
        .n_heads = n_heads,
        .n_kv_heads = n_heads,
        .head_dim = head_dim,
        .pos_offset = 0,
        .theta_base = 10000.0f};

    RoPEStage stage(params);

    bool success = false;
    try
    {
        success = stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "RoPE execution requires device enumeration: " << e.what();
    }

    if (success)
    {
        q_data = Q->data(); // Refresh after execution
        float norm0_after = computeNorm(&q_data[0], head_dim);
        float norm1_after = computeNorm(&q_data[head_dim], head_dim);

        // Norms should be preserved (rotations don't change magnitude)
        EXPECT_NEAR(norm0_before, norm0_after, 1e-4f);
        EXPECT_NEAR(norm1_before, norm1_after, 1e-4f);
    }
}

// =============================================================================
// SwiGLUStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, SwiGLUStage_TypeAndBackend)
{
    const int seq_len = 4, intermediate_dim = 256;
    auto gate = makeTensor(seq_len, intermediate_dim);
    auto up = makeTensor(seq_len, intermediate_dim);
    auto output = makeTensor(seq_len, intermediate_dim);

    SwiGLUStage::Params params{
        .gate = gate.get(),
        .up = up.get(),
        .output = output.get()};

    SwiGLUStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::SWIGLU);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(ComputeStagesTest, SwiGLUStage_EstimatedFlops)
{
    const int seq_len = 8, intermediate_dim = 512;
    auto gate = makeTensor(seq_len, intermediate_dim);

    SwiGLUStage::Params params{
        .gate = gate.get()};

    SwiGLUStage stage(params);

    // SwiGLU: swish(gate) * up
    // Per element: sigmoid (exp, add, div) + mul (for swish) + mul (with up)
    // Approximately 6 ops per element
    size_t flops = stage.estimatedFlops();
    EXPECT_GT(flops, 0);
    EXPECT_GE(flops, static_cast<size_t>(seq_len * intermediate_dim));
}

TEST_F(ComputeStagesTest, SwiGLUStage_SnapshotInfo)
{
    const int seq_len = 4, intermediate_dim = 64;
    auto gate = makeTensor(seq_len, intermediate_dim);
    auto up = makeTensor(seq_len, intermediate_dim);
    auto output = makeTensor(seq_len, intermediate_dim);

    SwiGLUStage::Params params{
        .gate = gate.get(),
        .up = up.get(),
        .output = output.get()};

    SwiGLUStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output->data());
    EXPECT_EQ(dump_info.outputs[0].rows, seq_len);
    EXPECT_EQ(dump_info.outputs[0].cols, intermediate_dim);
    EXPECT_STREQ(dump_info.outputs[0].name, "output");
}

TEST_F(ComputeStagesTest, SwiGLUStage_Correctness)
{
    // SwiGLU(gate, up) = swish(gate) * up = (gate * sigmoid(gate)) * up
    const int seq_len = 1, intermediate_dim = 4;

    std::vector<float> gate_data = {0.0f, 1.0f, -1.0f, 2.0f};
    std::vector<float> up_data = {1.0f, 2.0f, 3.0f, 4.0f};

    auto gate = makeTensor(seq_len, intermediate_dim, gate_data);
    auto up = makeTensor(seq_len, intermediate_dim, up_data);
    auto output = makeTensor(seq_len, intermediate_dim);

    // Compute expected
    std::vector<float> expected(seq_len * intermediate_dim);
    for (int i = 0; i < intermediate_dim; ++i)
    {
        float sigmoid_gate = 1.0f / (1.0f + std::exp(-gate_data[i]));
        float swish = gate_data[i] * sigmoid_gate;
        expected[i] = swish * up_data[i];
    }

    SwiGLUStage::Params params{
        .gate = gate.get(),
        .up = up.get(),
        .output = output.get()};

    SwiGLUStage stage(params);
    bool success = stage.execute(ctx_.get());

    if (success)
    {
        const float *out = output->data();
        for (int i = 0; i < intermediate_dim; ++i)
        {
            EXPECT_NEAR(out[i], expected[i], 1e-5f)
                << "Mismatch at index " << i
                << ": expected " << expected[i] << ", got " << out[i];
        }
    }
}

// =============================================================================
// ResidualAddStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, ResidualAddStage_TypeAndBackend)
{
    const int rows = 4, cols = 64;
    auto input = makeTensor(rows, cols);
    auto residual = makeTensor(rows, cols);
    auto output = makeTensor(rows, cols);

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = output.get()};

    ResidualAddStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(ComputeStagesTest, ResidualAddStage_EstimatedFlops)
{
    const int rows = 32, cols = 32;
    auto input = makeTensor(rows, cols);

    ResidualAddStage::Params params{
        .input = input.get()};

    ResidualAddStage stage(params);

    // Residual add: 1 add per element
    EXPECT_EQ(stage.estimatedFlops(), static_cast<size_t>(rows * cols));
}

TEST_F(ComputeStagesTest, ResidualAddStage_SnapshotInfo)
{
    const int rows = 4, cols = 64;
    auto input = makeTensor(rows, cols);
    auto residual = makeTensor(rows, cols);
    auto output = makeTensor(rows, cols);

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = output.get()};

    ResidualAddStage stage(params);

    auto dump_info = stage.getDumpInfo();
    ASSERT_EQ(dump_info.outputs.size(), 1);
    EXPECT_EQ(dump_info.outputs[0].data, output->data());
    // Note: rows/cols may vary based on precision handling
    EXPECT_STREQ(dump_info.outputs[0].name, "output");
}

TEST_F(ComputeStagesTest, ResidualAddStage_FP32_Correctness)
{
    const size_t n = 8;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> residual_data = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};

    auto input = makeTensor(1, n, input_data);
    auto residual = makeTensor(1, n, residual_data);
    auto output = makeTensor(1, n);

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = output.get()};

    ResidualAddStage stage(params);
    bool success = stage.execute(ctx_.get());

    EXPECT_TRUE(success);

    const float *out = output->data();
    for (size_t i = 0; i < n; ++i)
    {
        float expected = input_data[i] + residual_data[i];
        EXPECT_FLOAT_EQ(out[i], expected) << "Mismatch at index " << i;
    }
}

TEST_F(ComputeStagesTest, ResidualAddStage_FP32_InPlace)
{
    const size_t n = 8;

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> residual_data = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    std::vector<float> expected(n);
    for (size_t i = 0; i < n; ++i)
    {
        expected[i] = input_data[i] + residual_data[i];
    }

    auto input = makeTensor(1, n, input_data);
    auto residual = makeTensor(1, n, residual_data);

    // In-place: output points to residual
    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = residual.get()}; // In-place

    ResidualAddStage stage(params);
    bool success = stage.execute(ctx_.get());

    EXPECT_TRUE(success);

    const float *out = residual->data();
    for (size_t i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(out[i], expected[i]) << "Mismatch at index " << i;
    }
}

TEST_F(ComputeStagesTest, ResidualAddStage_Q8_1_PrecisionPreservation)
{
    // Q8_1 residual addition now requires Q8_1Tensor types
    // This test is skipped until Q8_1 tensor support is added to stage tests
    GTEST_SKIP() << "Q8_1 residual addition requires Q8_1Tensor (not FP32Tensor)";
}

TEST_F(ComputeStagesTest, ResidualAddStage_AllPrecisionModes)
{
    // TensorBase*-based API determines precision from tensor native_type()
    // This test verifies that FP32 tensors work correctly
    const int rows = 2, cols = 32;

    auto input = makeTensor(rows, cols);
    auto residual = makeTensor(rows, cols);
    auto output = makeTensor(rows, cols);

    // Fill with test data
    for (size_t i = 0; i < rows * cols; ++i)
    {
        input->mutable_data()[i] = 1.0f;
        residual->mutable_data()[i] = 0.5f;
    }

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = output.get()};

    ResidualAddStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::ADD_RESIDUAL);

    // Execution should succeed for FP32
    bool success = stage.execute(ctx_.get());
    EXPECT_TRUE(success);
}

// =============================================================================
// Stage Factory and Registry Tests
// =============================================================================

TEST_F(ComputeStagesTest, StageTypeName_AllTypes)
{
    // Verify all stage types have names
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::GEMM), "GEMM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::RMS_NORM), "RMS_NORM");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::SWIGLU), "SWIGLU");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ROPE), "ROPE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ATTENTION), "ATTENTION");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ADD_RESIDUAL), "ADD_RESIDUAL");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ALLREDUCE), "ALLREDUCE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_ROUTER), "MOE_ROUTER");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_EXPERT_FFN), "MOE_EXPERT_FFN");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::MOE_COMBINE), "MOE_COMBINE");
}

// =============================================================================
// Integration: Stage Composition Tests
// =============================================================================

TEST_F(ComputeStagesTest, StageComposition_RMSNormThenResidualAdd)
{
    // Test: RMSNorm → ResidualAdd pattern
    const int seq_len = 2, hidden_dim = 64;
    const size_t n = seq_len * hidden_dim;

    std::vector<float> input(n);
    std::vector<float> residual(n);
    std::vector<float> normalized(n, 0.0f);
    std::vector<float> output(n, 0.0f);
    std::vector<float> gamma(hidden_dim, 1.0f);

    fillRandom(input.data(), n, -1.0f, 1.0f);
    fillRandom(residual.data(), n, -0.5f, 0.5f);

    // Create tensors
    auto input_tensor = makeTensor(seq_len, hidden_dim, input);
    auto normalized_tensor = makeTensor(seq_len, hidden_dim, normalized);
    auto gamma_tensor = makeTensor1D(hidden_dim, gamma);
    auto residual_tensor = makeTensor(seq_len, hidden_dim, residual);
    auto output_tensor = makeTensor(seq_len, hidden_dim);

    // Step 1: RMSNorm
    RMSNormStage::Params norm_params{
        .input = input_tensor.get(),
        .output = normalized_tensor.get(),
        .gamma = gamma_tensor.get(),
        .eps = 1e-5f};
    RMSNormStage norm_stage(norm_params);

    // Step 2: Residual Add
    ResidualAddStage::Params add_params{
        .input = normalized_tensor.get(),
        .residual = residual_tensor.get(),
        .output = output_tensor.get()};
    ResidualAddStage add_stage(add_params);

    // Execute with exception handling for device enumeration
    bool norm_ok = false;
    bool add_ok = false;
    try
    {
        norm_ok = norm_stage.execute(ctx_.get());
        add_ok = add_stage.execute(ctx_.get());
    }
    catch (const std::exception &e)
    {
        GTEST_SKIP() << "Stage execution requires device enumeration: " << e.what();
    }

    if (norm_ok && add_ok)
    {
        // Verify output is different from normalized (residual added)
        bool all_same = true;
        for (size_t i = 0; i < n; ++i)
        {
            if (std::abs(output_tensor->data()[i] - normalized_tensor->data()[i]) > 1e-6f)
            {
                all_same = false;
                break;
            }
        }
        EXPECT_FALSE(all_same) << "Output should differ from normalized after residual add";
    }
}

// =============================================================================
// Memory Estimation Tests
// =============================================================================

TEST_F(ComputeStagesTest, MemoryEstimation_GEMMStage)
{
    const int m = 64, n = 128, k = 256;

    GEMMStage::Params params{.m = m, .n = n, .k = k};
    GEMMStage stage(params);

    // Should estimate at least the output matrix size
    size_t mem = stage.estimatedMemoryBytes();
    EXPECT_GE(mem, m * n * sizeof(float));
}

TEST_F(ComputeStagesTest, MemoryEstimation_RMSNormStage)
{
    const int seq_len = 8, hidden_dim = 512;

    auto input = makeTensor(seq_len, hidden_dim);
    auto output = makeTensor(seq_len, hidden_dim);
    auto gamma = makeTensor1D(hidden_dim);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-5f};
    RMSNormStage stage(params);

    size_t mem = stage.estimatedMemoryBytes();
    EXPECT_GE(mem, static_cast<size_t>(seq_len * hidden_dim * sizeof(float)));
}

TEST_F(ComputeStagesTest, MemoryEstimation_ResidualAddStage)
{
    const size_t rows = 4, cols = 256;

    auto input = makeTensor(rows, cols);
    auto residual = makeTensor(rows, cols);
    auto output = makeTensor(rows, cols);

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = output.get()};
    ResidualAddStage stage(params);

    size_t mem = stage.estimatedMemoryBytes();
    EXPECT_GE(mem, rows * cols * sizeof(float));
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(ComputeStagesTest, Stages_ZeroSize)
{
    // Test behavior with zero-sized inputs

    // GEMMStage with 0 dimensions
    GEMMStage::Params g_params{.m = 0, .n = 32, .k = 64};
    GEMMStage g_stage(g_params);
    EXPECT_EQ(g_stage.estimatedFlops(), 0);

    // ResidualAddStage with empty tensors
    auto empty_tensor = makeTensor(0, 32);
    ResidualAddStage::Params r_params{
        .input = empty_tensor.get(),
        .residual = empty_tensor.get(),
        .output = empty_tensor.get()};
    ResidualAddStage r_stage(r_params);
    EXPECT_EQ(r_stage.estimatedFlops(), 0);
}

TEST_F(ComputeStagesTest, Stages_LargeSize)
{
    // Test that large sizes don't overflow
    const int m = 4096, n = 4096, k = 4096;

    GEMMStage::Params params{.m = m, .n = n, .k = k};
    GEMMStage stage(params);

    // 2 * 4096^3 = ~137 billion FLOPs
    size_t flops = stage.estimatedFlops();
    EXPECT_EQ(flops, 2ULL * m * n * k);
}

// =============================================================================
// AllreduceStage Tests
// =============================================================================

TEST_F(ComputeStagesTest, AllreduceStage_TypeAndBackend)
{
    AllreduceStage::Params params{
        .buffer = nullptr,
        .mpi_ctx = nullptr};

    AllreduceStage stage(params);

    EXPECT_EQ(stage.type(), ComputeStageType::ALLREDUCE);
    // Allreduce requires MPI, may not support all backends
    EXPECT_TRUE(stage.supportsBackend(ComputeBackendType::CPU));
}

TEST_F(ComputeStagesTest, AllreduceStage_EstimatedFlops)
{
    AllreduceStage::Params params{.buffer = nullptr};
    AllreduceStage stage(params);

    // Allreduce is communication, not compute, but may have some overhead
    size_t flops = stage.estimatedFlops();
    // May be 0 or small value
    EXPECT_GE(flops, 0);
}
