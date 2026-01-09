/**
 * @file Test__DeviceIndexCPUKernels.cpp
 * @brief Unit tests for CPU kernel device_idx handling
 * @author David Sanftenberg
 *
 * Regression tests for device_idx bug where CPU kernels incorrectly rejected
 * device_idx = 0 (CPU device) thinking it was GPU.
 *
 * Bug History:
 * - DeviceManager assigns device 0 to CPU
 * - CPU kernels checked `if (device_idx != -1)` and failed
 * - Correct behavior: CPU kernels should accept any device_idx since they
 *   only operate on CPU tensor buffers (enforced by type system)
 */

#include <gtest/gtest.h>
#include "../../src/v2/kernels/cpu/ops/CPURMSNormKernelT.h"
#include "../../src/v2/kernels/cpu/ops/CPURoPEKernelT.h"
#include "../../src/v2/kernels/cpu/ops/CPUSwiGLUKernelT.h"
#include "../../src/v2/kernels/cpu/ops/CPUSoftmaxKernelT.h"

#include "../../src/v2/tensors/Tensors.h"
#include "../../src/v2/utils/MPIContext.h"
#include "../../src/v2/utils/MPIStager.h"
#include <vector>
#include <cmath>

using namespace llaminar2;

/**
 * Test fixture for device_idx validation
 */
class Test__DeviceIndexCPUKernels : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mpi_ctx = std::make_shared<MPIContext>(0, 1, MPI_COMM_SELF);
    }

    std::shared_ptr<MPIContext> mpi_ctx;
};

// ============================================================================
// CPURMSNormKernel Tests (using CPURMSNormKernelT<FP32>)
// ============================================================================

TEST_F(Test__DeviceIndexCPUKernels, RMSNorm_AcceptsDeviceZero)
{
    CPURMSNormKernelT<ActivationPrecision::FP32> kernel;

    const int seq_len = 4;
    const int d_model = 8;
    std::vector<float> input(seq_len * d_model, 1.0f);
    std::vector<float> weight(d_model, 1.0f);
    std::vector<float> output(seq_len * d_model, 0.0f);

    // device_idx = 0 (CPU device) should work
    bool result = kernel.apply_typed(
        input.data(), weight.data(), output.data(),
        seq_len, d_model, 1e-5f,
        0 // device_idx = 0 (CPU)
    );

    EXPECT_TRUE(result) << "RMSNorm should accept device_idx = 0 (CPU device)";

    // Verify output is normalized (not all zeros)
    float sum = 0.0f;
    for (float val : output)
    {
        sum += val;
    }
    EXPECT_GT(sum, 0.0f) << "Output should be normalized, not zeros";
}

TEST_F(Test__DeviceIndexCPUKernels, RMSNorm_AcceptsDeviceMinusOne)
{
    CPURMSNormKernelT<ActivationPrecision::FP32> kernel;

    const int seq_len = 4;
    const int d_model = 8;
    std::vector<float> input(seq_len * d_model, 1.0f);
    std::vector<float> weight(d_model, 1.0f);
    std::vector<float> output(seq_len * d_model, 0.0f);

    // device_idx = -1 (unspecified) should also work
    bool result = kernel.apply_typed(
        input.data(), weight.data(), output.data(),
        seq_len, d_model, 1e-5f,
        -1 // device_idx = -1 (unspecified)
    );

    EXPECT_TRUE(result) << "RMSNorm should accept device_idx = -1 (unspecified)";
}

TEST_F(Test__DeviceIndexCPUKernels, RMSNorm_IgnoresDeviceIndex)
{
    CPURMSNormKernelT<ActivationPrecision::FP32> kernel;

    const int seq_len = 4;
    const int d_model = 8;
    std::vector<float> input(seq_len * d_model, 2.0f);
    std::vector<float> weight(d_model, 1.0f);
    std::vector<float> output0(seq_len * d_model, 0.0f);
    std::vector<float> output1(seq_len * d_model, 0.0f);

    // Run with device_idx = 0
    kernel.apply_typed(input.data(), weight.data(), output0.data(),
                       seq_len, d_model, 1e-5f, 0);

    // Run with device_idx = -1
    kernel.apply_typed(input.data(), weight.data(), output1.data(),
                       seq_len, d_model, 1e-5f, -1);

    // Results should be identical (device_idx is ignored)
    for (size_t i = 0; i < output0.size(); ++i)
    {
        EXPECT_FLOAT_EQ(output0[i], output1[i])
            << "RMSNorm output should be identical regardless of device_idx";
    }
}

// ============================================================================
// CPURoPEKernel Tests (using CPURoPEKernelT<FP32>)
// ============================================================================

TEST_F(Test__DeviceIndexCPUKernels, RoPE_AcceptsDeviceZero)
{
    CPURoPEKernelT<ActivationPrecision::FP32> kernel;

    const int seq_len = 2;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 8;

    std::vector<float> Q(seq_len * n_heads * head_dim, 1.0f);
    std::vector<float> K(seq_len * n_kv_heads * head_dim, 1.0f);
    std::vector<int> position_ids = {0, 1};

    // device_idx = 0 (CPU device) should work
    bool result = kernel.apply_typed(
        Q.data(), K.data(),
        position_ids.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        10000.0f, // rope_theta
        0         // device_idx = 0 (CPU)
    );

    EXPECT_TRUE(result) << "RoPE should accept device_idx = 0 (CPU device)";
}

TEST_F(Test__DeviceIndexCPUKernels, RoPE_AcceptsDeviceMinusOne)
{
    CPURoPEKernelT<ActivationPrecision::FP32> kernel;

    const int seq_len = 2;
    const int n_heads = 4;
    const int n_kv_heads = 2;
    const int head_dim = 8;

    std::vector<float> Q(seq_len * n_heads * head_dim, 1.0f);
    std::vector<float> K(seq_len * n_kv_heads * head_dim, 1.0f);
    std::vector<int> position_ids = {0, 1};

    // device_idx = -1 (unspecified) should work
    bool result = kernel.apply_typed(
        Q.data(), K.data(),
        position_ids.data(),
        seq_len, n_heads, n_kv_heads, head_dim,
        10000.0f,
        -1 // device_idx = -1 (unspecified)
    );

    EXPECT_TRUE(result) << "RoPE should accept device_idx = -1 (unspecified)";
}

// ============================================================================
// CPUSwiGLUKernel Tests (using CPUSwiGLUKernelT<FP32>)
// ============================================================================

TEST_F(Test__DeviceIndexCPUKernels, SwiGLU_AcceptsDeviceZero)
{
    CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;

    const int seq_len = 4;
    const int d_ff = 8;
    const int size = seq_len * d_ff;
    std::vector<float> gate(size, 1.0f);
    std::vector<float> up(size, 1.0f);
    std::vector<float> output(size, 0.0f);

    // device_idx = 0 (CPU device) should work
    bool result = kernel.apply_typed(
        gate.data(), up.data(), output.data(),
        size,
        0 // device_idx = 0 (CPU)
    );

    EXPECT_TRUE(result) << "SwiGLU should accept device_idx = 0 (CPU device)";

    // Verify output is computed (not all zeros)
    float sum = 0.0f;
    for (float val : output)
    {
        sum += std::abs(val);
    }
    EXPECT_GT(sum, 0.0f) << "SwiGLU output should be non-zero";
}

TEST_F(Test__DeviceIndexCPUKernels, SwiGLU_AcceptsDeviceMinusOne)
{
    CPUSwiGLUKernelT<ActivationPrecision::FP32> kernel;

    const int seq_len = 4;
    const int d_ff = 8;
    const int size = seq_len * d_ff;
    std::vector<float> gate(size, 1.0f);
    std::vector<float> up(size, 1.0f);
    std::vector<float> output(size, 0.0f);

    // device_idx = -1 (unspecified) should work
    bool result = kernel.apply_typed(
        gate.data(), up.data(), output.data(),
        size,
        -1 // device_idx = -1 (unspecified)
    );

    EXPECT_TRUE(result) << "SwiGLU should accept device_idx = -1 (unspecified)";
}

// ============================================================================
// CPUSoftmaxKernel Tests (using CPUSoftmaxKernelT<FP32>)
// ============================================================================

TEST_F(Test__DeviceIndexCPUKernels, Softmax_AcceptsDeviceZero)
{
    CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

    const int rows = 4;
    const int cols = 8;
    std::vector<float> data(rows * cols);

    // Fill with simple values
    for (size_t i = 0; i < data.size(); ++i)
    {
        data[i] = static_cast<float>(i % cols);
    }

    // device_idx = 0 (CPU device) should work
    // Note: Typed kernel applies softmax in-place
    bool result = kernel.apply_typed(
        data.data(),
        rows, cols,
        false, // use_causal_mask
        1.0f,  // scale
        0      // device_idx = 0 (CPU)
    );

    EXPECT_TRUE(result) << "Softmax should accept device_idx = 0 (CPU device)";

    // Verify each row sums to ~1.0
    for (int r = 0; r < rows; ++r)
    {
        float row_sum = 0.0f;
        for (int c = 0; c < cols; ++c)
        {
            row_sum += data[r * cols + c];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f) << "Softmax row " << r << " should sum to 1.0";
    }
}

TEST_F(Test__DeviceIndexCPUKernels, Softmax_AcceptsDeviceMinusOne)
{
    CPUSoftmaxKernelT<ActivationPrecision::FP32> kernel;

    const int rows = 4;
    const int cols = 8;
    std::vector<float> data(rows * cols, 1.0f);

    // device_idx = -1 (unspecified) should work
    // Note: Typed kernel applies softmax in-place
    bool result = kernel.apply_typed(
        data.data(),
        rows, cols,
        false, // use_causal_mask
        1.0f,  // scale
        -1     // device_idx = -1 (unspecified)
    );

    EXPECT_TRUE(result) << "Softmax should accept device_idx = -1 (unspecified)";
}

// ============================================================================
// MPIStager Tests
// ============================================================================

TEST_F(Test__DeviceIndexCPUKernels, MPIStager_CPUTensorNoStaging)
{
    // Create real CPU tensor (device_idx = 0)
    auto cpu_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 2}, DeviceId::cpu());

    // CPU tensor (device_idx = 0) should NOT require staging
    bool requires_staging = MPIStager::requiresStaging(cpu_tensor.get());
    EXPECT_FALSE(requires_staging)
        << "CPU tensor (device_idx=0) should not require GPU staging";
}

TEST_F(Test__DeviceIndexCPUKernels, MPIStager_UnspecifiedDeviceNoStaging)
{
    // Create tensor with CPU device
    auto unspecified_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{2, 2}, DeviceId::cpu());

    // CPU device should not require staging
    bool requires_staging = MPIStager::requiresStaging(unspecified_tensor.get());
    EXPECT_FALSE(requires_staging)
        << "CPU device should not require staging";
}

// Note: Cannot test GPU staging without actual GPU backend, but the logic is:
// device_idx > 0 should require staging (GPU devices)
