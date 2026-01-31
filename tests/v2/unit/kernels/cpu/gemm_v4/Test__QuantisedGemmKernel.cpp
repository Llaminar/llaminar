#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmKernel.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "backends/DeviceId.h"
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace llaminar2;
using namespace llaminar2::gemm_v4;

TEST(Test__QuantisedGemmKernel, BasicMatMul)
{
    // Dimensions
    int M = 1;
    int N = 64;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    // Q8_1Tensor expects [N, K] shape
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

    // Create kernel
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random input A (M x K)
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Compute reference C (M x N)
    std::vector<float> C_ref(M * N, 0.0f);
    // A is M x K, Weights is N x K.
    // C = A * Weights^T
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A[m * K + k] * weights_fp32[n * K + k];
            }
            C_ref[m * N + n] = sum;
        }
    }

    // Compute actual C
    std::vector<float> C_act(M * N, 0.0f);
    kernel->multiply(A.data(), C_act.data(), M, N, K);

    // Compare
    // Q8_1 quantization introduces error.
    // Tolerance depends on range.
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C_act[i], C_ref[i], 1.0f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test the new multiply_tensor interface with FP32 activations
 *
 * This tests the fallback path: FP32 activations go through online quantization
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensorFP32Activations)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random FP32 input tensor (M x K)
    std::vector<float> A_data(M * K);
    for (auto &x : A_data)
        x = dist(gen);
    auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Compute reference C (M x N)
    std::vector<float> C_ref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += A_data[m * K + k] * weights_fp32[n * K + k];
            }
            C_ref[m * N + n] = sum;
        }
    }

    // Test multiply_tensor interface
    bool ok = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compare
    const float *C_act = C_tensor->data();
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C_act[i], C_ref[i], 1.0f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test multiply_tensor with Q8_1 activations (zero-copy direct path)
 *
 * This tests the optimized path: Q8_1 activations bypass float conversion
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensorQ8_1Activations)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create FP32 activations first (for reference)
    std::vector<float> A_fp32(M * K);
    for (auto &x : A_fp32)
        x = dist(gen);

    // Quantize activations to Q8_1
    auto A_q8_1 = Q8_1Tensor::quantize_from_fp32(A_fp32.data(), {static_cast<size_t>(M), static_cast<size_t>(K)});
    ASSERT_NE(A_q8_1, nullptr);

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Compute reference using FP32 path
    std::vector<float> C_ref(M * N, 0.0f);
    kernel->multiply(A_fp32.data(), C_ref.data(), M, N, K);

    // Test multiply_tensor with Q8_1 input (should use direct path)
    bool ok = kernel->multiply_tensor(A_q8_1.get(), C_tensor.get(), true, 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compare - both use quantized GEMM, so tolerance depends on quantization error
    const float *C_act = C_tensor->data();
    for (int i = 0; i < M * N; ++i)
    {
        // Same quantization, so results should be close (but not identical
        // because the FP32 path does its own online quantization)
        EXPECT_NEAR(C_act[i], C_ref[i], 2.0f) << "Mismatch at index " << i;
    }
}

/**
 * @brief Test activation sharing interface (quantize_activations + multiply_with_precomputed_q8_1)
 *
 * This tests the fused multi-GEMM workflow:
 * 1. quantize_activations() once
 * 2. multiply_with_precomputed_q8_1() multiple times with different weights
 */
TEST(Test__QuantisedGemmKernel, ActivationSharingInterface)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights for two projections (like gate/up in FFN)
    std::vector<float> weights1_fp32(N * K);
    std::vector<float> weights2_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights1_fp32)
        x = dist(gen);
    for (auto &x : weights2_fp32)
        x = dist(gen);

    // Quantize weights
    auto weights1 = Q8_1Tensor::quantize_from_fp32(weights1_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto weights2 = Q8_1Tensor::quantize_from_fp32(weights2_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});

    // Create kernels
    auto kernel1 = weights1->createGemm();
    auto kernel2 = weights2->createGemm();
    ASSERT_NE(kernel1, nullptr);
    ASSERT_NE(kernel2, nullptr);

    // Verify activation sharing is supported
    EXPECT_TRUE(kernel1->supports_activation_sharing());
    EXPECT_TRUE(kernel2->supports_activation_sharing());

    // Create random input
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Compute reference using standard path
    std::vector<float> C1_ref(M * N, 0.0f);
    std::vector<float> C2_ref(M * N, 0.0f);
    ASSERT_TRUE(kernel1->multiply(A.data(), C1_ref.data(), M, N, K));
    ASSERT_TRUE(kernel2->multiply(A.data(), C2_ref.data(), M, N, K));

    // Test shared activation path
    // Step 1: Allocate buffer
    size_t buffer_size = kernel1->get_quantized_activation_buffer_size(M, K);
    EXPECT_GT(buffer_size, 0u);
    std::vector<uint8_t> q8_1_buffer(buffer_size);

    // Step 2: Quantize activations once
    ASSERT_TRUE(kernel1->quantize_activations(A.data(), q8_1_buffer.data(), M, K));

    // Step 3: Execute both GEMMs with shared activations
    std::vector<float> C1_act(M * N, 0.0f);
    std::vector<float> C2_act(M * N, 0.0f);
    ASSERT_TRUE(kernel1->multiply_with_precomputed_q8_1(q8_1_buffer.data(), C1_act.data(), M, N, K));
    ASSERT_TRUE(kernel2->multiply_with_precomputed_q8_1(q8_1_buffer.data(), C2_act.data(), M, N, K));

    // Compare - shared activation path should produce identical results to standard path
    // (both use the same quantization algorithm)
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(C1_act[i], C1_ref[i], 0.01f) << "Kernel1 mismatch at index " << i;
        EXPECT_NEAR(C2_act[i], C2_ref[i], 0.01f) << "Kernel2 mismatch at index " << i;
    }
}

/**
 * @brief Test that non-quantized kernels correctly report no activation sharing support
 */
TEST(Test__QuantisedGemmKernel, ActivationSharingNotSupportedByDefault)
{
    // The base ITensorGemm returns false for supports_activation_sharing()
    // We can't easily test this without a non-quantized kernel, but we can at least
    // verify the interface methods exist and return sensible values
    int M = 2;
    int K = 32;

    // Create a simple Q8_1 tensor just to get a kernel
    std::vector<float> weights(32 * 32);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights)
        x = dist(gen);
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights.data(), {32, 32});
    auto kernel = weights_tensor->createGemm();

    // Test buffer size calculation
    size_t buffer_size = kernel->get_quantized_activation_buffer_size(M, K);
    EXPECT_EQ(buffer_size, M * (K / 32) * sizeof(Q8_1Block));
}

// =============================================================================
// SwiGLU Fused Operation Tests
// =============================================================================

namespace
{
    /**
     * @brief Reference implementation of sigmoid function
     */
    inline float sigmoid_ref(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }

    /**
     * @brief Reference implementation of swish/silu activation: x * sigmoid(x)
     */
    inline float swish_ref(float x)
    {
        return x * sigmoid_ref(x);
    }

    /**
     * @brief Reference implementation of SwiGLU: up * swish(gate)
     *
     * SwiGLU(gate, up) = up * swish(gate) = up * gate * sigmoid(gate)
     */
    inline float swiglu_ref(float gate, float up)
    {
        return up * swish_ref(gate);
    }
}

/**
 * @brief Test SwiGLU fused operation mathematical correctness (M=1, single row)
 *
 * This tests the complete fused workflow:
 * 1. Compute gate_output = input @ gate_weights.T
 * 2. Compute up_output = input @ up_weights.T * swish(gate_output)
 *
 * The kernel should produce: up_output[i] = (input @ up_weights.T)[i] * swish((input @ gate_weights.T)[i])
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_SingleRow_Correctness)
{
    int M = 1;   // Single token (uses M1 kernel)
    int N = 128; // Output dimension
    int K = 64;  // Input dimension

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random weights for gate and up projections
    std::vector<float> gate_weights_fp32(N * K);
    std::vector<float> up_weights_fp32(N * K);
    for (auto &x : gate_weights_fp32)
        x = dist(gen);
    for (auto &x : up_weights_fp32)
        x = dist(gen);

    // Quantize weights
    auto gate_weights = Q8_1Tensor::quantize_from_fp32(gate_weights_fp32.data(),
                                                       {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto up_weights = Q8_1Tensor::quantize_from_fp32(up_weights_fp32.data(),
                                                     {static_cast<size_t>(N), static_cast<size_t>(K)});

    // Create kernels
    auto gate_kernel = gate_weights->createGemm();
    auto up_kernel = up_weights->createGemm();
    ASSERT_NE(gate_kernel, nullptr);
    ASSERT_NE(up_kernel, nullptr);

    // Create random input
    std::vector<float> input(M * K);
    for (auto &x : input)
        x = dist(gen);

    // Step 1: Compute gate output (non-fused, for reference and as input to SwiGLU)
    std::vector<float> gate_output(M * N, 0.0f);
    ASSERT_TRUE(gate_kernel->multiply(input.data(), gate_output.data(), M, N, K));

    // Step 2: Compute up output WITHOUT SwiGLU (for reference)
    std::vector<float> up_output_no_swiglu(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply(input.data(), up_output_no_swiglu.data(), M, N, K));

    // Step 3: Compute reference SwiGLU output manually
    std::vector<float> swiglu_ref_output(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        swiglu_ref_output[i] = swiglu_ref(gate_output[i], up_output_no_swiglu[i]);
    }

    // Step 4: Compute fused SwiGLU output using the kernel
    // First quantize activations
    size_t buffer_size = up_kernel->get_quantized_activation_buffer_size(M, K);
    std::vector<uint8_t> q8_1_buffer(buffer_size);
    ASSERT_TRUE(up_kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K));

    // Execute with SwiGLU enabled
    std::vector<float> swiglu_actual_output(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply_with_precomputed_q8_1(
        q8_1_buffer.data(),
        swiglu_actual_output.data(),
        M, N, K,
        nullptr, // no bias
        false,   // no accumulate
        1.0f, 0.0f,
        nullptr, -1,
        GemmFusedOps::swiglu(gate_output.data())));

    // Compare fused output vs reference
    // Tolerance accounts for:
    // - Quantization error in GEMM
    // - Polynomial approximation of exp() in JIT kernel
    float max_diff = 0.0f;
    float mean_diff = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float diff = std::abs(swiglu_actual_output[i] - swiglu_ref_output[i]);
        max_diff = std::max(max_diff, diff);
        mean_diff += diff;
        EXPECT_NEAR(swiglu_actual_output[i], swiglu_ref_output[i], 0.5f)
            << "Mismatch at index " << i
            << ", gate=" << gate_output[i]
            << ", up=" << up_output_no_swiglu[i];
    }
    mean_diff /= (M * N);

    // Also verify that the output is meaningfully different from un-swiglu'd output
    // (i.e., the SwiGLU was actually applied)
    float diff_from_unswiglu = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        diff_from_unswiglu += std::abs(swiglu_actual_output[i] - up_output_no_swiglu[i]);
    }
    diff_from_unswiglu /= (M * N);
    EXPECT_GT(diff_from_unswiglu, 0.01f) << "SwiGLU should modify the output";
}

/**
 * @brief Test SwiGLU fused operation with multiple rows (M=2, uses M2 kernel)
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_MultiRow_Correctness)
{
    int M = 2;   // Two rows (uses M2 kernel)
    int N = 128; // Output dimension
    int K = 64;  // Input dimension

    std::mt19937 gen(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create random weights
    std::vector<float> gate_weights_fp32(N * K);
    std::vector<float> up_weights_fp32(N * K);
    for (auto &x : gate_weights_fp32)
        x = dist(gen);
    for (auto &x : up_weights_fp32)
        x = dist(gen);

    // Quantize weights
    auto gate_weights = Q8_1Tensor::quantize_from_fp32(gate_weights_fp32.data(),
                                                       {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto up_weights = Q8_1Tensor::quantize_from_fp32(up_weights_fp32.data(),
                                                     {static_cast<size_t>(N), static_cast<size_t>(K)});

    auto gate_kernel = gate_weights->createGemm();
    auto up_kernel = up_weights->createGemm();

    // Create random input
    std::vector<float> input(M * K);
    for (auto &x : input)
        x = dist(gen);

    // Compute gate output
    std::vector<float> gate_output(M * N, 0.0f);
    ASSERT_TRUE(gate_kernel->multiply(input.data(), gate_output.data(), M, N, K));

    // Compute up output without SwiGLU
    std::vector<float> up_output_no_swiglu(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply(input.data(), up_output_no_swiglu.data(), M, N, K));

    // Compute reference SwiGLU
    std::vector<float> swiglu_ref_output(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        swiglu_ref_output[i] = swiglu_ref(gate_output[i], up_output_no_swiglu[i]);
    }

    // Compute fused SwiGLU
    size_t buffer_size = up_kernel->get_quantized_activation_buffer_size(M, K);
    std::vector<uint8_t> q8_1_buffer(buffer_size);
    ASSERT_TRUE(up_kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K));

    std::vector<float> swiglu_actual_output(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply_with_precomputed_q8_1(
        q8_1_buffer.data(),
        swiglu_actual_output.data(),
        M, N, K,
        nullptr, false, 1.0f, 0.0f,
        nullptr, -1,
        GemmFusedOps::swiglu(gate_output.data())));

    // Compare per-row
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            int idx = m * N + n;
            EXPECT_NEAR(swiglu_actual_output[idx], swiglu_ref_output[idx], 0.5f)
                << "Row " << m << ", col " << n
                << ", gate=" << gate_output[idx]
                << ", up=" << up_output_no_swiglu[idx];
        }
    }
}

/**
 * @brief Test SwiGLU with larger batch (M=8, tests OpenMP parallelization)
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_LargeBatch_Correctness)
{
    int M = 8;
    int N = 256;
    int K = 128;

    std::mt19937 gen(456);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> gate_weights_fp32(N * K);
    std::vector<float> up_weights_fp32(N * K);
    for (auto &x : gate_weights_fp32)
        x = dist(gen);
    for (auto &x : up_weights_fp32)
        x = dist(gen);

    auto gate_weights = Q8_1Tensor::quantize_from_fp32(gate_weights_fp32.data(),
                                                       {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto up_weights = Q8_1Tensor::quantize_from_fp32(up_weights_fp32.data(),
                                                     {static_cast<size_t>(N), static_cast<size_t>(K)});

    auto gate_kernel = gate_weights->createGemm();
    auto up_kernel = up_weights->createGemm();

    std::vector<float> input(M * K);
    for (auto &x : input)
        x = dist(gen);

    // Compute gate and up outputs
    std::vector<float> gate_output(M * N, 0.0f);
    std::vector<float> up_output_no_swiglu(M * N, 0.0f);
    ASSERT_TRUE(gate_kernel->multiply(input.data(), gate_output.data(), M, N, K));
    ASSERT_TRUE(up_kernel->multiply(input.data(), up_output_no_swiglu.data(), M, N, K));

    // Reference SwiGLU
    std::vector<float> swiglu_ref_output(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        swiglu_ref_output[i] = swiglu_ref(gate_output[i], up_output_no_swiglu[i]);
    }

    // Fused SwiGLU
    size_t buffer_size = up_kernel->get_quantized_activation_buffer_size(M, K);
    std::vector<uint8_t> q8_1_buffer(buffer_size);
    ASSERT_TRUE(up_kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K));

    std::vector<float> swiglu_actual_output(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply_with_precomputed_q8_1(
        q8_1_buffer.data(),
        swiglu_actual_output.data(),
        M, N, K,
        nullptr, false, 1.0f, 0.0f,
        nullptr, -1,
        GemmFusedOps::swiglu(gate_output.data())));

    // Verify correctness
    float max_diff = 0.0f;
    int mismatch_count = 0;
    for (int i = 0; i < M * N; ++i)
    {
        float diff = std::abs(swiglu_actual_output[i] - swiglu_ref_output[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 0.5f)
        {
            mismatch_count++;
        }
    }
    EXPECT_LT(max_diff, 1.0f) << "Max difference too large";
    EXPECT_LT(mismatch_count, M * N * 0.01) << "Too many mismatches";
}

/**
 * @brief Test SwiGLU with extreme gate values (tests exp() approximation stability)
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_ExtremeGateValues)
{
    int M = 1;
    int N = 64;
    int K = 32;

    std::mt19937 gen(789);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Create simple identity-like weights so we can control the outputs
    std::vector<float> weights_fp32(N * K, 0.0f);
    // Make a diagonal-ish pattern
    for (int n = 0; n < N && n < K; ++n)
    {
        weights_fp32[n * K + n] = 1.0f;
    }
    for (int n = K; n < N; ++n)
    {
        weights_fp32[n * K + (n % K)] = 0.5f;
    }

    auto weights = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(),
                                                  {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights->createGemm();

    // Create input that will produce extreme gate values
    std::vector<float> input(M * K);
    for (auto &x : input)
        x = dist(gen);

    // Compute a non-fused output for the "up" path
    std::vector<float> up_output(M * N, 0.0f);
    ASSERT_TRUE(kernel->multiply(input.data(), up_output.data(), M, N, K));

    // Create extreme gate values: very negative, zero, very positive
    std::vector<float> gate_values(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        if (i % 4 == 0)
            gate_values[i] = -50.0f; // Very negative -> sigmoid ~ 0
        else if (i % 4 == 1)
            gate_values[i] = 0.0f; // Zero -> swish = 0
        else if (i % 4 == 2)
            gate_values[i] = 50.0f; // Very positive -> sigmoid ~ 1
        else
            gate_values[i] = dist(gen) * 2.0f; // Normal range
    }

    // Reference SwiGLU
    std::vector<float> swiglu_ref_output(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        swiglu_ref_output[i] = swiglu_ref(gate_values[i], up_output[i]);
    }

    // Fused SwiGLU
    size_t buffer_size = kernel->get_quantized_activation_buffer_size(M, K);
    std::vector<uint8_t> q8_1_buffer(buffer_size);
    ASSERT_TRUE(kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K));

    std::vector<float> swiglu_actual_output(M * N, 0.0f);
    ASSERT_TRUE(kernel->multiply_with_precomputed_q8_1(
        q8_1_buffer.data(),
        swiglu_actual_output.data(),
        M, N, K,
        nullptr, false, 1.0f, 0.0f,
        nullptr, -1,
        GemmFusedOps::swiglu(gate_values.data())));

    // Check specific cases
    for (int i = 0; i < M * N; ++i)
    {
        if (i % 4 == 0)
        {
            // Very negative gate -> output should be ~0
            EXPECT_NEAR(swiglu_actual_output[i], 0.0f, 0.1f)
                << "Very negative gate should produce ~0 output at idx " << i;
        }
        else if (i % 4 == 1)
        {
            // Zero gate -> swish(0) = 0 -> output should be 0
            EXPECT_NEAR(swiglu_actual_output[i], 0.0f, 0.1f)
                << "Zero gate should produce 0 output at idx " << i;
        }
        else if (i % 4 == 2)
        {
            // Very positive gate -> swish(gate) ~ gate -> output ~ up * gate
            EXPECT_NEAR(swiglu_actual_output[i], swiglu_ref_output[i], 1.0f)
                << "Very positive gate mismatch at idx " << i;
        }
        else
        {
            // Normal range
            EXPECT_NEAR(swiglu_actual_output[i], swiglu_ref_output[i], 0.5f)
                << "Normal gate mismatch at idx " << i;
        }
    }
}

/**
 * @brief Test that SwiGLU disabled (do_swiglu=false) produces same output as non-fused
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_Disabled_MatchesNonFused)
{
    int M = 4;
    int N = 128;
    int K = 64;

    std::mt19937 gen(999);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> weights_fp32(N * K);
    for (auto &x : weights_fp32)
        x = dist(gen);

    auto weights = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(),
                                                  {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights->createGemm();

    std::vector<float> input(M * K);
    for (auto &x : input)
        x = dist(gen);

    // Reference: standard multiply path
    std::vector<float> ref_output(M * N, 0.0f);
    ASSERT_TRUE(kernel->multiply(input.data(), ref_output.data(), M, N, K));

    // Test: multiply_with_precomputed_q8_1 with do_swiglu=false
    size_t buffer_size = kernel->get_quantized_activation_buffer_size(M, K);
    std::vector<uint8_t> q8_1_buffer(buffer_size);
    ASSERT_TRUE(kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K));

    std::vector<float> actual_output(M * N, 0.0f);
    // Provide gate_input but with do_swiglu = false via empty GemmFusedOps
    std::vector<float> dummy_gate(M * N, 1.0f);
    ASSERT_TRUE(kernel->multiply_with_precomputed_q8_1(
        q8_1_buffer.data(),
        actual_output.data(),
        M, N, K,
        nullptr, false, 1.0f, 0.0f,
        nullptr, -1,
        GemmFusedOps{} // No fused ops - do_swiglu defaults to false
        ));

    // Should match exactly (within floating-point tolerance)
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(actual_output[i], ref_output[i], 0.01f)
            << "do_swiglu=false should match standard multiply at idx " << i;
    }
}

/**
 * @brief Test SwiGLU with bias (bias is applied before SwiGLU in the JIT kernel)
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_WithBias)
{
    int M = 2;
    int N = 64;
    int K = 32;

    std::mt19937 gen(111);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> gate_weights_fp32(N * K);
    std::vector<float> up_weights_fp32(N * K);
    std::vector<float> up_bias_data(N);
    for (auto &x : gate_weights_fp32)
        x = dist(gen);
    for (auto &x : up_weights_fp32)
        x = dist(gen);
    for (auto &x : up_bias_data)
        x = dist(gen) * 0.1f;

    auto up_bias = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(N)}, DeviceId::cpu());
    std::copy(up_bias_data.begin(), up_bias_data.end(), up_bias->mutable_data());

    auto gate_weights = Q8_1Tensor::quantize_from_fp32(gate_weights_fp32.data(),
                                                       {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto up_weights = Q8_1Tensor::quantize_from_fp32(up_weights_fp32.data(),
                                                     {static_cast<size_t>(N), static_cast<size_t>(K)});

    auto gate_kernel = gate_weights->createGemm();
    auto up_kernel = up_weights->createGemm();

    std::vector<float> input(M * K);
    for (auto &x : input)
        x = dist(gen);

    // Compute gate output
    std::vector<float> gate_output(M * N, 0.0f);
    ASSERT_TRUE(gate_kernel->multiply(input.data(), gate_output.data(), M, N, K));

    // Compute up output without SwiGLU, with bias
    std::vector<float> up_output_with_bias(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply(input.data(), up_output_with_bias.data(), M, N, K));
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            up_output_with_bias[m * N + n] += up_bias_data[n];
        }
    }

    // Reference SwiGLU: bias is applied to up BEFORE SwiGLU
    std::vector<float> swiglu_ref_output(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        swiglu_ref_output[i] = swiglu_ref(gate_output[i], up_output_with_bias[i]);
    }

    // Fused SwiGLU with bias
    size_t buffer_size = up_kernel->get_quantized_activation_buffer_size(M, K);
    std::vector<uint8_t> q8_1_buffer(buffer_size);
    ASSERT_TRUE(up_kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K));

    std::vector<float> swiglu_actual_output(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply_with_precomputed_q8_1(
        q8_1_buffer.data(),
        swiglu_actual_output.data(),
        M, N, K,
        up_bias.get(), // bias
        false, 1.0f, 0.0f,
        nullptr, -1,
        GemmFusedOps::swiglu(gate_output.data())));

    // Compare
    for (int i = 0; i < M * N; ++i)
    {
        EXPECT_NEAR(swiglu_actual_output[i], swiglu_ref_output[i], 0.5f)
            << "SwiGLU with bias mismatch at idx " << i;
    }
}

/**
 * @brief Compare L2 drift between fused SwiGLU (in-kernel) vs non-fused (separate kernel)
 *
 * This test verifies that the fused SwiGLU path produces equal or better numerical
 * accuracy compared to the non-fused path (GEMM + separate SwiGLU application).
 *
 * The hypothesis is that fused SwiGLU should have less L2 drift because:
 * 1. One fewer memory round-trip (GEMM result stays in registers)
 * 2. Fewer floating-point rounding operations
 *
 * Both paths use the same polynomial exp() approximation, so the improvement
 * should be minimal but measurable.
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_FusedVsNonFused_L2Drift)
{
    // Use realistic FFN dimensions for Qwen2-0.5B
    // hidden_size=896, intermediate_size=4864
    int M = 8;    // Batch of tokens
    int N = 4864; // Intermediate size (d_ff)
    int K = 896;  // Hidden size (d_model)

    std::mt19937 gen(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f); // Realistic activation range

    // Create weights for gate and up projections
    std::vector<float> gate_weights_fp32(N * K);
    std::vector<float> up_weights_fp32(N * K);
    for (auto &x : gate_weights_fp32)
        x = dist(gen);
    for (auto &x : up_weights_fp32)
        x = dist(gen);

    // Quantize weights
    auto gate_weights = Q8_1Tensor::quantize_from_fp32(gate_weights_fp32.data(),
                                                       {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto up_weights = Q8_1Tensor::quantize_from_fp32(up_weights_fp32.data(),
                                                     {static_cast<size_t>(N), static_cast<size_t>(K)});

    auto gate_kernel = gate_weights->createGemm();
    auto up_kernel = up_weights->createGemm();

    // Create random input
    std::vector<float> input(M * K);
    for (auto &x : input)
        x = dist(gen);

    // =========================================================================
    // PATH 1: Non-fused (GEMM + separate SwiGLU)
    // =========================================================================

    // Compute gate output
    std::vector<float> gate_output(M * N, 0.0f);
    ASSERT_TRUE(gate_kernel->multiply(input.data(), gate_output.data(), M, N, K));

    // Compute up output (stored to memory)
    std::vector<float> up_output_nonfused(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply(input.data(), up_output_nonfused.data(), M, N, K));

    // Apply SwiGLU separately (simulates loading from memory and applying)
    std::vector<float> swiglu_nonfused(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        swiglu_nonfused[i] = swiglu_ref(gate_output[i], up_output_nonfused[i]);
    }

    // =========================================================================
    // PATH 2: Fused SwiGLU (GEMM with in-kernel SwiGLU)
    // =========================================================================

    // Quantize activations once for both paths
    size_t buffer_size = up_kernel->get_quantized_activation_buffer_size(M, K);
    std::vector<uint8_t> q8_1_buffer(buffer_size);
    ASSERT_TRUE(up_kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K));

    // Execute fused GEMM + SwiGLU
    std::vector<float> swiglu_fused(M * N, 0.0f);
    ASSERT_TRUE(up_kernel->multiply_with_precomputed_q8_1(
        q8_1_buffer.data(),
        swiglu_fused.data(),
        M, N, K,
        nullptr, false, 1.0f, 0.0f,
        nullptr, -1,
        GemmFusedOps::swiglu(gate_output.data())));

    // =========================================================================
    // GROUND TRUTH: FP64 reference (highest precision)
    // =========================================================================

    // Compute FP64 ground truth for gate GEMM
    std::vector<double> gate_output_fp64(M * N, 0.0);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            double sum = 0.0;
            for (int kk = 0; kk < K; ++kk)
            {
                sum += static_cast<double>(input[m * K + kk]) *
                       static_cast<double>(gate_weights_fp32[n * K + kk]);
            }
            gate_output_fp64[m * N + n] = sum;
        }
    }

    // Compute FP64 ground truth for up GEMM
    std::vector<double> up_output_fp64(M * N, 0.0);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            double sum = 0.0;
            for (int kk = 0; kk < K; ++kk)
            {
                sum += static_cast<double>(input[m * K + kk]) *
                       static_cast<double>(up_weights_fp32[n * K + kk]);
            }
            up_output_fp64[m * N + n] = sum;
        }
    }

    // Compute FP64 ground truth for SwiGLU: up * silu(gate)
    std::vector<double> swiglu_fp64(M * N);
    for (int i = 0; i < M * N; ++i)
    {
        double gate = gate_output_fp64[i];
        double up = up_output_fp64[i];
        double sigmoid_gate = 1.0 / (1.0 + std::exp(-gate));
        double silu_gate = gate * sigmoid_gate;
        swiglu_fp64[i] = up * silu_gate;
    }

    // =========================================================================
    // COMPUTE L2 DRIFT METRICS
    // =========================================================================

    double l2_nonfused = 0.0;
    double l2_fused = 0.0;
    double linf_nonfused = 0.0;
    double linf_fused = 0.0;
    double sum_sq_ref = 0.0;

    for (int i = 0; i < M * N; ++i)
    {
        double ref = swiglu_fp64[i];
        double err_nonfused = static_cast<double>(swiglu_nonfused[i]) - ref;
        double err_fused = static_cast<double>(swiglu_fused[i]) - ref;

        l2_nonfused += err_nonfused * err_nonfused;
        l2_fused += err_fused * err_fused;

        linf_nonfused = std::max(linf_nonfused, std::abs(err_nonfused));
        linf_fused = std::max(linf_fused, std::abs(err_fused));

        sum_sq_ref += ref * ref;
    }

    double rel_l2_nonfused = std::sqrt(l2_nonfused / sum_sq_ref);
    double rel_l2_fused = std::sqrt(l2_fused / sum_sq_ref);
    double rms_nonfused = std::sqrt(l2_nonfused / (M * N));
    double rms_fused = std::sqrt(l2_fused / (M * N));

    // Print comparison
    std::cout << "\n========== SwiGLU Fused vs Non-Fused L2 Drift Comparison ==========\n";
    std::cout << "Dimensions: M=" << M << " N=" << N << " K=" << K << "\n";
    std::cout << "Total elements: " << (M * N) << "\n\n";

    std::cout << "NON-FUSED (GEMM + separate SwiGLU):\n";
    std::cout << "  Relative L2 error: " << std::scientific << rel_l2_nonfused << "\n";
    std::cout << "  RMS error:         " << std::scientific << rms_nonfused << "\n";
    std::cout << "  L-infinity error:  " << std::scientific << linf_nonfused << "\n\n";

    std::cout << "FUSED (GEMM with in-kernel SwiGLU):\n";
    std::cout << "  Relative L2 error: " << std::scientific << rel_l2_fused << "\n";
    std::cout << "  RMS error:         " << std::scientific << rms_fused << "\n";
    std::cout << "  L-infinity error:  " << std::scientific << linf_fused << "\n\n";

    double improvement = (rel_l2_nonfused - rel_l2_fused) / rel_l2_nonfused * 100.0;
    std::cout << "COMPARISON:\n";
    std::cout << "  Fused L2 improvement: " << std::fixed << std::setprecision(2) << improvement << "%\n";
    std::cout << "  Fused is " << (rel_l2_fused < rel_l2_nonfused ? "BETTER" : "WORSE") << " than non-fused\n";
    std::cout << "===================================================================\n\n";

    // The fused path should be at least as good as non-fused
    // Allow 1% tolerance for measurement noise
    EXPECT_LE(rel_l2_fused, rel_l2_nonfused * 1.01)
        << "Fused SwiGLU should not be significantly worse than non-fused";

    // Both should be within reasonable accuracy bounds
    EXPECT_LT(rel_l2_fused, 1e-2) << "Fused L2 error too high";
    EXPECT_LT(rel_l2_nonfused, 1e-2) << "Non-fused L2 error too high";
}

/**
 * @brief Test L2 drift across multiple batch sizes to verify consistency
 */
TEST(Test__QuantisedGemmKernel, SwiGLU_FusedVsNonFused_BatchSizeScaling)
{
    const std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32};
    int N = 1536; // Intermediate size
    int K = 512;  // Hidden size

    std::cout << "\n========== SwiGLU L2 Drift vs Batch Size ==========\n";
    std::cout << std::setw(8) << "Batch"
              << std::setw(16) << "NonFused L2"
              << std::setw(16) << "Fused L2"
              << std::setw(16) << "Improvement"
              << "\n";
    std::cout << std::string(56, '-') << "\n";

    for (int M : batch_sizes)
    {
        std::mt19937 gen(42 + M); // Different seed per batch size
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

        // Create weights
        std::vector<float> gate_weights_fp32(N * K);
        std::vector<float> up_weights_fp32(N * K);
        for (auto &x : gate_weights_fp32)
            x = dist(gen);
        for (auto &x : up_weights_fp32)
            x = dist(gen);

        auto gate_weights = Q8_1Tensor::quantize_from_fp32(gate_weights_fp32.data(),
                                                           {static_cast<size_t>(N), static_cast<size_t>(K)});
        auto up_weights = Q8_1Tensor::quantize_from_fp32(up_weights_fp32.data(),
                                                         {static_cast<size_t>(N), static_cast<size_t>(K)});

        auto gate_kernel = gate_weights->createGemm();
        auto up_kernel = up_weights->createGemm();

        std::vector<float> input(M * K);
        for (auto &x : input)
            x = dist(gen);

        // Non-fused path
        std::vector<float> gate_output(M * N, 0.0f);
        std::vector<float> up_output(M * N, 0.0f);
        gate_kernel->multiply(input.data(), gate_output.data(), M, N, K);
        up_kernel->multiply(input.data(), up_output.data(), M, N, K);

        std::vector<float> swiglu_nonfused(M * N);
        for (int i = 0; i < M * N; ++i)
        {
            swiglu_nonfused[i] = swiglu_ref(gate_output[i], up_output[i]);
        }

        // Fused path
        size_t buffer_size = up_kernel->get_quantized_activation_buffer_size(M, K);
        std::vector<uint8_t> q8_1_buffer(buffer_size);
        up_kernel->quantize_activations(input.data(), q8_1_buffer.data(), M, K);

        std::vector<float> swiglu_fused(M * N, 0.0f);
        up_kernel->multiply_with_precomputed_q8_1(
            q8_1_buffer.data(), swiglu_fused.data(), M, N, K,
            nullptr, false, 1.0f, 0.0f, nullptr, -1,
            GemmFusedOps::swiglu(gate_output.data()));

        // FP64 reference
        std::vector<double> swiglu_fp64(M * N);
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                int idx = m * N + n;
                double gate_sum = 0.0, up_sum = 0.0;
                for (int kk = 0; kk < K; ++kk)
                {
                    gate_sum += static_cast<double>(input[m * K + kk]) * gate_weights_fp32[n * K + kk];
                    up_sum += static_cast<double>(input[m * K + kk]) * up_weights_fp32[n * K + kk];
                }
                double silu_gate = gate_sum / (1.0 + std::exp(-gate_sum));
                swiglu_fp64[idx] = up_sum * silu_gate;
            }
        }

        // Compute relative L2
        double l2_nonfused = 0.0, l2_fused = 0.0, sum_sq_ref = 0.0;
        for (int i = 0; i < M * N; ++i)
        {
            double ref = swiglu_fp64[i];
            double err_nonfused = swiglu_nonfused[i] - ref;
            double err_fused = swiglu_fused[i] - ref;
            l2_nonfused += err_nonfused * err_nonfused;
            l2_fused += err_fused * err_fused;
            sum_sq_ref += ref * ref;
        }
        double rel_l2_nonfused = std::sqrt(l2_nonfused / sum_sq_ref);
        double rel_l2_fused = std::sqrt(l2_fused / sum_sq_ref);
        double improvement = (rel_l2_nonfused - rel_l2_fused) / rel_l2_nonfused * 100.0;

        std::cout << std::setw(8) << M
                  << std::setw(16) << std::scientific << rel_l2_nonfused
                  << std::setw(16) << std::scientific << rel_l2_fused
                  << std::setw(15) << std::fixed << std::setprecision(2) << improvement << "%"
                  << "\n";

        // Fused should not be significantly worse
        EXPECT_LE(rel_l2_fused, rel_l2_nonfused * 1.05);
    }
    std::cout << "====================================================\n\n";
}

/**
 * @brief Test multiply_tensor with BF16 activations
 *
 * This tests the new IActivationTensor::quantize_to_q8_1() path for BF16.
 * BF16 → FP32 → Q8_1 quantization happens inside the tensor.
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensorBF16Activations)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create FP32 activations first
    std::vector<float> A_fp32(M * K);
    for (auto &x : A_fp32)
        x = dist(gen);

    // Convert to BF16
    std::vector<uint16_t> A_bf16(M * K);
    for (int i = 0; i < M * K; ++i)
    {
        // BF16 = upper 16 bits of FP32
        uint32_t fp32_bits;
        std::memcpy(&fp32_bits, &A_fp32[i], sizeof(float));
        A_bf16[i] = static_cast<uint16_t>((fp32_bits + 0x8000) >> 16); // Round to nearest
    }

    // Create BF16Tensor
    auto A_bf16_tensor = std::make_unique<BF16Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)},
        A_bf16);

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Compute reference using FP32 path
    std::vector<float> C_ref(M * N, 0.0f);
    kernel->multiply(A_fp32.data(), C_ref.data(), M, N, K);

    // Test multiply_tensor with BF16 input
    bool ok = kernel->multiply_tensor(A_bf16_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compare - BF16 has limited precision, so tolerance is higher
    const float *C_act = C_tensor->data();
    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i)
    {
        double err = std::abs(C_act[i] - C_ref[i]);
        if (err > max_err)
            max_err = err;
        // BF16 has ~3 decimal digits of precision, accumulated over K elements
        EXPECT_NEAR(C_act[i], C_ref[i], 2.0f) << "Mismatch at index " << i;
    }
    std::cout << "[BF16 GEMM] Max error: " << max_err << std::endl;
}

/**
 * @brief Test multiply_tensor with INT8 activations (transcoding path)
 *
 * This tests the INT8 → FP32 → Q8_1 transcoding path.
 * INT8 activations are dequantized using their per-tensor scale,
 * then re-quantized to Q8_1 block format.
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensorINT8Activations)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create FP32 activations first
    std::vector<float> A_fp32(M * K);
    for (auto &x : A_fp32)
        x = dist(gen);

    // Create INT8Tensor by quantizing FP32 (uses constructor that takes fp32_data)
    auto A_int8_tensor = std::make_unique<INT8Tensor>(
        std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)},
        A_fp32);
    ASSERT_NE(A_int8_tensor, nullptr);

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Compute reference using FP32 path
    std::vector<float> C_ref(M * N, 0.0f);
    kernel->multiply(A_fp32.data(), C_ref.data(), M, N, K);

    // Test multiply_tensor with INT8 input (should use transcoding path)
    bool ok = kernel->multiply_tensor(A_int8_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f);
    ASSERT_TRUE(ok);

    // Compare - INT8 has ~7 bits of precision, plus Q8_1 quantization
    const float *C_act = C_tensor->data();
    double max_err = 0.0;
    for (int i = 0; i < M * N; ++i)
    {
        double err = std::abs(C_act[i] - C_ref[i]);
        if (err > max_err)
            max_err = err;
        // Double quantization (INT8→Q8_1) accumulates error
        EXPECT_NEAR(C_act[i], C_ref[i], 3.0f) << "Mismatch at index " << i;
    }
    std::cout << "[INT8 GEMM] Max error: " << max_err << std::endl;
}

// =============================================================================
// Workspace Parameter Tests
// =============================================================================

/**
 * @brief Test that multiply() works with workspace=nullptr (existing behavior)
 *
 * CPU kernels don't use workspace, so nullptr should work fine.
 */
TEST(Test__QuantisedGemmKernel, Multiply_WithNullWorkspace_Succeeds)
{
    int M = 4;
    int N = 64;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random input A (M x K)
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Create output buffer
    std::vector<float> C(M * N, 0.0f);

    // Call multiply with explicit workspace=nullptr
    bool success = kernel->multiply(A.data(), C.data(), M, N, K, true, 1.0f, 0.0f, nullptr, -1, nullptr);
    EXPECT_TRUE(success);

    // Verify output is non-zero (kernel actually computed something)
    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
        sum += std::abs(C[i]);
    EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
}

/**
 * @brief Test that multiply() works with a non-null workspace
 *
 * CPU kernels accept but ignore the workspace parameter.
 * This tests that the parameter is accepted without error.
 */
TEST(Test__QuantisedGemmKernel, Multiply_WithWorkspace_Succeeds)
{
    int M = 4;
    int N = 64;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random input A (M x K)
    std::vector<float> A(M * K);
    for (auto &x : A)
        x = dist(gen);

    // Create output buffer
    std::vector<float> C(M * N, 0.0f);

    // Create a workspace manager for CPU
    auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), 1024 * 1024);
    ASSERT_NE(workspace, nullptr);

    // Call multiply with non-null workspace (CPU kernel should accept but ignore it)
    bool success = kernel->multiply(A.data(), C.data(), M, N, K, true, 1.0f, 0.0f, nullptr, -1, workspace.get());
    EXPECT_TRUE(success);

    // Verify output is non-zero (kernel actually computed something)
    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
        sum += std::abs(C[i]);
    EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
}

/**
 * @brief Test that multiply_tensor() works with workspace=nullptr (existing behavior)
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensor_WithNullWorkspace_Succeeds)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random FP32 input tensor (M x K)
    std::vector<float> A_data(M * K);
    for (auto &x : A_data)
        x = dist(gen);
    auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Call multiply_tensor with explicit workspace=nullptr
    bool success = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f, nullptr, nullptr, -1, nullptr);
    EXPECT_TRUE(success);

    // Verify output is non-zero
    const float *C_data = C_tensor->data();
    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
        sum += std::abs(C_data[i]);
    EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
}

/**
 * @brief Test that multiply_tensor() works with a non-null workspace
 *
 * CPU kernels accept but ignore the workspace parameter.
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensor_WithWorkspace_Succeeds)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random FP32 input tensor (M x K)
    std::vector<float> A_data(M * K);
    for (auto &x : A_data)
        x = dist(gen);
    auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Create a workspace manager for CPU
    auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), 1024 * 1024);
    ASSERT_NE(workspace, nullptr);

    // Call multiply_tensor with non-null workspace (CPU kernel should accept but ignore it)
    bool success = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), true, 1.0f, 0.0f, nullptr, nullptr, -1, workspace.get());
    EXPECT_TRUE(success);

    // Verify output is non-zero
    const float *C_data = C_tensor->data();
    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
        sum += std::abs(C_data[i]);
    EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
}

/**
 * @brief Test that multiply_tensor() with explicit dims works with workspace
 */
TEST(Test__QuantisedGemmKernel, MultiplyTensorExplicitDims_WithWorkspace_Succeeds)
{
    int M = 4;
    int N = 128;
    int K = 64;

    // Create random weights (N x K)
    std::vector<float> weights_fp32(N * K);
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : weights_fp32)
        x = dist(gen);

    // Quantize weights to Q8_1Tensor
    auto weights_tensor = Q8_1Tensor::quantize_from_fp32(weights_fp32.data(), {static_cast<size_t>(N), static_cast<size_t>(K)});
    auto kernel = weights_tensor->createGemm();
    ASSERT_NE(kernel, nullptr);

    // Create random FP32 input tensor (M x K)
    std::vector<float> A_data(M * K);
    for (auto &x : A_data)
        x = dist(gen);
    auto A_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(K)});
    std::memcpy(A_tensor->mutable_data(), A_data.data(), A_data.size() * sizeof(float));

    // Create output tensor
    auto C_tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(M), static_cast<size_t>(N)});

    // Create a workspace manager for CPU
    auto workspace = std::make_unique<DeviceWorkspaceManager>(DeviceId::cpu(), 1024 * 1024);
    ASSERT_NE(workspace, nullptr);

    // Call multiply_tensor with explicit dimensions and workspace
    bool success = kernel->multiply_tensor(A_tensor.get(), C_tensor.get(), M, N, K, true, 1.0f, 0.0f, nullptr, nullptr, -1, workspace.get());
    EXPECT_TRUE(success);

    // Verify output is non-zero
    const float *C_data = C_tensor->data();
    float sum = 0.0f;
    for (int i = 0; i < M * N; ++i)
        sum += std::abs(C_data[i]);
    EXPECT_GT(sum, 0.0f) << "Output should contain non-zero values";
}
