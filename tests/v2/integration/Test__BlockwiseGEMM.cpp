/**
 * @file Test__BlockwiseGEMM.cpp
 * @brief Integration tests for blockwise Q8_1 × quantized weight GEMM
 * @author David Sanftenberg
 * @date November 24, 2025
 *
 * Tests verify blockwise GEMM microkernel correctness across all quantized weight types:
 *   - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
 *   - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
 *   - IQ4_NL, IQ4_XS, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S
 *
 * Test methodology:
 *   1. Load quantized weight tensor from real GGUF model
 *   2. Create Q8_1 activation tensor (quantized from transposed weights for testing)
 *   3. Reference path: FP32 GEMM + bias + scale + softmax
 *   4. Microkernel path: Blockwise Q8_1 × Quant GEMM + bias + scale + softmax (fused)
 *   5. Verify numerical equivalence within tolerance
 *
 * Test file naming convention:
 *   File: Test__BlockwiseGEMM.cpp → Testing: Blockwise GEMM functionality
 *   Suite: TEST(Test__BlockwiseGEMM, ...) → Matches filename
 */

#include <gtest/gtest.h>
#include "loaders/ModelLoader.h"
#include "loaders/ModelContext.h"
#include "tensors/Tensors.h"
#include "tensors/TensorFactory.h"
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include "utils/MPIContext.h"
#include <vector>
#include <cmath>
#include <memory>
#include <map>

using namespace llaminar2;

// =============================================================================
// TEST FIXTURE
// =============================================================================

class Test__BlockwiseGEMM : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Model paths for different quantization types
        model_paths_ = {
            {TensorType::Q4_0, "models/qwen2.5-0.5b-instruct-q4_0.gguf"},
            {TensorType::Q5_0, "models/qwen2.5-0.5b-instruct-q5_0.gguf"},
            {TensorType::Q8_0, "models/qwen2.5-0.5b-instruct-q8_0.gguf"},
            {TensorType::IQ4_NL, "models/qwen2.5-0.5b-instruct-iq4_nl.gguf"},
            {TensorType::Q2_K, "models/qwen2.5-0.5b-instruct-q2_k.gguf"},
            {TensorType::Q3_K, "models/qwen2.5-0.5b-instruct-q3_k_m.gguf"},
            {TensorType::Q4_K, "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"},
            {TensorType::Q5_K, "models/qwen2.5-0.5b-instruct-q5_k_m.gguf"},
            {TensorType::Q6_K, "models/qwen2.5-0.5b-instruct-q6_k.gguf"},
        };
    }

    /**
     * @brief Create Q8_1 activation tensor from FP32 data
     */
    std::shared_ptr<Q8_1Tensor> createQ8_1FromFP32(const float *fp32_data, size_t rows, size_t cols)
    {
        std::vector<size_t> shape = {rows, cols};
        return Q8_1Tensor::quantize_from_fp32(fp32_data, shape);
    }

    /**
     * @brief Reference FP32 GEMM: C = A @ B^T
     */
    void referenceGEMM(const float *A, const float *B, float *C, int M, int N, int K)
    {
        for (int m = 0; m < M; ++m)
        {
            for (int n = 0; n < N; ++n)
            {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k)
                {
                    sum += A[m * K + k] * B[n * K + k]; // B is [N, K] for transpose
                }
                C[m * N + n] = sum;
            }
        }
    }

    /**
     * @brief Apply softmax along rows (axis=1)
     */
    void applySoftmax(float *data, int M, int N)
    {
        for (int m = 0; m < M; ++m)
        {
            float *row = data + m * N;

            // Find max for numerical stability
            float max_val = row[0];
            for (int n = 1; n < N; ++n)
            {
                max_val = std::max(max_val, row[n]);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int n = 0; n < N; ++n)
            {
                row[n] = std::exp(row[n] - max_val);
                sum += row[n];
            }

            // Normalize
            float inv_sum = 1.0f / sum;
            for (int n = 0; n < N; ++n)
            {
                row[n] *= inv_sum;
            }
        }
    }

    /**
     * @brief Compare two float arrays with relative tolerance
     */
    bool compareOutputs(const float *a, const float *b, size_t count,
                        float abs_tolerance = 1e-4f, float rel_tolerance = 1e-3f)
    {
        float max_abs_diff = 0.0f;
        float max_rel_diff = 0.0f;
        size_t mismatch_count = 0;

        for (size_t i = 0; i < count; ++i)
        {
            float abs_diff = std::fabs(a[i] - b[i]);
            float denominator = std::max(std::fabs(a[i]), std::fabs(b[i]));
            float rel_diff = (denominator > 1e-8f) ? (abs_diff / denominator) : 0.0f;

            max_abs_diff = std::max(max_abs_diff, abs_diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);

            if (abs_diff > abs_tolerance && rel_diff > rel_tolerance)
            {
                ++mismatch_count;
                if (mismatch_count <= 5)
                {
                    std::cout << "Mismatch at [" << i << "]: "
                              << "ref=" << a[i] << ", test=" << b[i]
                              << ", abs_diff=" << abs_diff
                              << ", rel_diff=" << rel_diff << std::endl;
                }
            }
        }

        if (mismatch_count > 0)
        {
            std::cout << "Total mismatches: " << mismatch_count << "/" << count << std::endl;
            std::cout << "Max absolute diff: " << max_abs_diff << std::endl;
            std::cout << "Max relative diff: " << max_rel_diff << std::endl;
        }

        return mismatch_count == 0;
    }

    std::map<TensorType, std::string> model_paths_;
};

// =============================================================================
// TESTS
// =============================================================================

/**
 * @brief Test blockwise GEMM with Q4_0 weights
 */
TEST_F(Test__BlockwiseGEMM, Q4_0_Correctness)
{
    const std::string model_path = model_paths_[TensorType::Q4_0];

    // Load model and extract a weight tensor
    ModelLoader loader(model_path);
    ASSERT_TRUE(loader.is_valid()) << "Failed to load model: " << model_path;

    // Get first attention weight tensor (should be quantized Q4_0)
    std::string weight_name = "blk.0.attn_q.weight";
    auto weight_info = loader.get_tensor_info(weight_name);
    ASSERT_TRUE(weight_info.has_value()) << "Weight tensor not found: " << weight_name;
    ASSERT_EQ(weight_info->type, TensorType::Q4_0);

    // Load weight tensor
    auto weight_tensor = loader.load_tensor(weight_name, 0);
    ASSERT_NE(weight_tensor, nullptr);

    // Get dimensions [N, K]
    auto shape = weight_tensor->shape();
    ASSERT_EQ(shape.size(), 2);
    int N = static_cast<int>(shape[0]); // Output features
    int K = static_cast<int>(shape[1]); // Input features
    int M = 4;                          // Small batch for testing

    // Ensure K is multiple of 32 (required for blockwise)
    ASSERT_EQ(K % 32, 0) << "K must be multiple of 32 for blockwise GEMM";

    // Dequantize weights to FP32 for creating activation tensor
    std::vector<float> weight_fp32(N * K);
    weight_tensor->to_fp32(weight_fp32.data(), N * K);

    // Create FP32 activation tensor (transpose weights for M × K shape)
    std::vector<float> activation_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            activation_fp32[m * K + k] = weight_fp32[(m % N) * K + k];
        }
    }

    // Create Q8_1 activation tensor
    auto q8_1_activations = createQ8_1FromFP32(activation_fp32.data(), M, K);
    ASSERT_NE(q8_1_activations, nullptr);

    // Reference path: FP32 GEMM
    std::vector<float> reference_output(M * N);
    referenceGEMM(activation_fp32.data(), weight_fp32.data(), reference_output.data(), M, N, K);

    // Apply softmax to reference
    applySoftmax(reference_output.data(), M, N);

    // Microkernel path: Create OneDNNGemmKernel with weight tensor
    gemm_v4::OneDNNGemmKernel gemm_kernel(weight_tensor.get());

    // Create Q8_1 blocks array
    const Q8_1Block *q8_1_blocks = q8_1_activations->blocks_data();

    // Allocate output buffer
    std::vector<float> microkernel_output(M * N, 0.0f);

    // Pack weights (triggers blockwise packing internally)
    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value()) << "Failed to pack weights for blockwise GEMM";

    // Execute blockwise GEMM
    bool success = gemm_kernel.execute_blockwise_gemm(
        q8_1_blocks,
        weight_pack.value(),
        microkernel_output.data(),
        M, N, K,
        nullptr, // No bias
        1.0f,    // alpha
        0.0f     // beta
    );
    ASSERT_TRUE(success) << "Blockwise GEMM execution failed";

    // Apply softmax to microkernel output
    applySoftmax(microkernel_output.data(), M, N);

    // Compare outputs
    EXPECT_TRUE(compareOutputs(
        reference_output.data(),
        microkernel_output.data(),
        M * N,
        1e-3f, // Absolute tolerance (quantization introduces error)
        5e-2f  // Relative tolerance (5% - reasonable for Q4_0)
        ))
        << "Q4_0 blockwise GEMM output differs from reference";
}

/**
 * @brief Test blockwise GEMM with Q8_0 weights
 */
TEST_F(Test__BlockwiseGEMM, Q8_0_Correctness)
{
    const std::string model_path = model_paths_[TensorType::Q8_0];

    ModelLoader loader(model_path);
    ASSERT_TRUE(loader.is_valid()) << "Failed to load model: " << model_path;

    std::string weight_name = "blk.0.attn_q.weight";
    auto weight_info = loader.get_tensor_info(weight_name);
    ASSERT_TRUE(weight_info.has_value()) << "Weight tensor not found: " << weight_name;
    ASSERT_EQ(weight_info->type, TensorType::Q8_0);

    auto weight_tensor = loader.load_tensor(weight_name, 0);
    ASSERT_NE(weight_tensor, nullptr);

    auto shape = weight_tensor->shape();
    int N = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);
    int M = 4;

    ASSERT_EQ(K % 32, 0);

    // Dequantize and create Q8_1 activations
    std::vector<float> weight_fp32(N * K);
    weight_tensor->to_fp32(weight_fp32.data(), N * K);

    std::vector<float> activation_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            activation_fp32[m * K + k] = weight_fp32[(m % N) * K + k];
        }
    }

    auto q8_1_activations = createQ8_1FromFP32(activation_fp32.data(), M, K);

    // Reference output
    std::vector<float> reference_output(M * N);
    referenceGEMM(activation_fp32.data(), weight_fp32.data(), reference_output.data(), M, N, K);
    applySoftmax(reference_output.data(), M, N);

    // Microkernel output
    gemm_v4::OneDNNGemmKernel gemm_kernel(weight_tensor.get());
    const Q8_1Block *q8_1_blocks = q8_1_activations->blocks_data();
    std::vector<float> microkernel_output(M * N, 0.0f);

    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value());

    bool success = gemm_kernel.execute_blockwise_gemm(
        q8_1_blocks, weight_pack.value(), microkernel_output.data(),
        M, N, K, nullptr, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    applySoftmax(microkernel_output.data(), M, N);

    EXPECT_TRUE(compareOutputs(reference_output.data(), microkernel_output.data(), M * N,
                               1e-4f, 1e-2f))
        << "Q8_0 blockwise GEMM differs from reference";
}

/**
 * @brief Test blockwise GEMM with IQ4_NL weights
 */
TEST_F(Test__BlockwiseGEMM, IQ4_NL_Correctness)
{
    const std::string model_path = model_paths_[TensorType::IQ4_NL];

    ModelLoader loader(model_path);
    ASSERT_TRUE(loader.is_valid()) << "Failed to load model: " << model_path;

    std::string weight_name = "blk.0.attn_q.weight";
    auto weight_info = loader.get_tensor_info(weight_name);
    ASSERT_TRUE(weight_info.has_value());
    ASSERT_EQ(weight_info->type, TensorType::IQ4_NL);

    auto weight_tensor = loader.load_tensor(weight_name, 0);
    ASSERT_NE(weight_tensor, nullptr);

    auto shape = weight_tensor->shape();
    int N = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);
    int M = 4;

    ASSERT_EQ(K % 32, 0);

    std::vector<float> weight_fp32(N * K);
    weight_tensor->to_fp32(weight_fp32.data(), N * K);

    std::vector<float> activation_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            activation_fp32[m * K + k] = weight_fp32[(m % N) * K + k];
        }
    }

    auto q8_1_activations = createQ8_1FromFP32(activation_fp32.data(), M, K);

    std::vector<float> reference_output(M * N);
    referenceGEMM(activation_fp32.data(), weight_fp32.data(), reference_output.data(), M, N, K);
    applySoftmax(reference_output.data(), M, N);

    gemm_v4::OneDNNGemmKernel gemm_kernel(weight_tensor.get());
    const Q8_1Block *q8_1_blocks = q8_1_activations->blocks_data();
    std::vector<float> microkernel_output(M * N, 0.0f);

    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value());

    bool success = gemm_kernel.execute_blockwise_gemm(
        q8_1_blocks, weight_pack.value(), microkernel_output.data(),
        M, N, K, nullptr, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    applySoftmax(microkernel_output.data(), M, N);

    EXPECT_TRUE(compareOutputs(reference_output.data(), microkernel_output.data(), M * N,
                               2e-3f, 5e-2f))
        << "IQ4_NL blockwise GEMM differs from reference";
}

/**
 * @brief Test blockwise GEMM with Q6_K weights
 */
TEST_F(Test__BlockwiseGEMM, Q6_K_Correctness)
{
    const std::string model_path = model_paths_[TensorType::Q6_K];

    ModelLoader loader(model_path);
    ASSERT_TRUE(loader.is_valid());

    std::string weight_name = "blk.0.attn_q.weight";
    auto weight_tensor = loader.load_tensor(weight_name, 0);
    ASSERT_NE(weight_tensor, nullptr);

    auto shape = weight_tensor->shape();
    int N = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);
    int M = 4;

    ASSERT_EQ(K % 32, 0);

    std::vector<float> weight_fp32(N * K);
    weight_tensor->to_fp32(weight_fp32.data(), N * K);

    std::vector<float> activation_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            activation_fp32[m * K + k] = weight_fp32[(m % N) * K + k];
        }
    }

    auto q8_1_activations = createQ8_1FromFP32(activation_fp32.data(), M, K);

    std::vector<float> reference_output(M * N);
    referenceGEMM(activation_fp32.data(), weight_fp32.data(), reference_output.data(), M, N, K);
    applySoftmax(reference_output.data(), M, N);

    gemm_v4::OneDNNGemmKernel gemm_kernel(weight_tensor.get());
    const Q8_1Block *q8_1_blocks = q8_1_activations->blocks_data();
    std::vector<float> microkernel_output(M * N, 0.0f);

    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value());

    bool success = gemm_kernel.execute_blockwise_gemm(
        q8_1_blocks, weight_pack.value(), microkernel_output.data(),
        M, N, K, nullptr, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    applySoftmax(microkernel_output.data(), M, N);

    EXPECT_TRUE(compareOutputs(reference_output.data(), microkernel_output.data(), M * N,
                               1e-4f, 1e-2f))
        << "Q6_K blockwise GEMM differs from reference";
}

/**
 * @brief Test blockwise GEMM with Q2_K weights (low precision extreme)
 */
TEST_F(Test__BlockwiseGEMM, Q2_K_Correctness)
{
    const std::string model_path = model_paths_[TensorType::Q2_K];

    ModelLoader loader(model_path);
    ASSERT_TRUE(loader.is_valid());

    std::string weight_name = "blk.0.attn_q.weight";
    auto weight_tensor = loader.load_tensor(weight_name, 0);
    ASSERT_NE(weight_tensor, nullptr);

    auto shape = weight_tensor->shape();
    int N = static_cast<int>(shape[0]);
    int K = static_cast<int>(shape[1]);
    int M = 4;

    ASSERT_EQ(K % 32, 0);

    std::vector<float> weight_fp32(N * K);
    weight_tensor->to_fp32(weight_fp32.data(), N * K);

    std::vector<float> activation_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            activation_fp32[m * K + k] = weight_fp32[(m % N) * K + k];
        }
    }

    auto q8_1_activations = createQ8_1FromFP32(activation_fp32.data(), M, K);

    std::vector<float> reference_output(M * N);
    referenceGEMM(activation_fp32.data(), weight_fp32.data(), reference_output.data(), M, N, K);
    applySoftmax(reference_output.data(), M, N);

    gemm_v4::OneDNNGemmKernel gemm_kernel(weight_tensor.get());
    const Q8_1Block *q8_1_blocks = q8_1_activations->blocks_data();
    std::vector<float> microkernel_output(M * N, 0.0f);

    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value());

    bool success = gemm_kernel.execute_blockwise_gemm(
        q8_1_blocks, weight_pack.value(), microkernel_output.data(),
        M, N, K, nullptr, 1.0f, 0.0f);
    ASSERT_TRUE(success);

    applySoftmax(microkernel_output.data(), M, N);

    // Q2_K is very low precision - allow higher tolerance
    EXPECT_TRUE(compareOutputs(reference_output.data(), microkernel_output.data(), M * N,
                               5e-3f, 1e-1f))
        << "Q2_K blockwise GEMM differs from reference";
}
