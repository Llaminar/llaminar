/**
 * @file Test__BlockwiseGEMM_Simple.cpp
 * @brief Simple integration test for blockwise Q8_1 × quantized weight GEMM
 * @author David Sanftenberg
 * @date November 24, 2025
 *
 * Test verifies blockwise GEMM microkernel correctness for Q4_0 quantization.
 */

#include <gtest/gtest.h>
#include "loaders/ModelContext.h"
#include "tensors/Tensors.h"
#include "kernels/cpu/gemm_v4/OneDNNGemmKernel.h"
#include <vector>
#include <cmath>

using namespace llaminar2;

TEST(Test__BlockwiseGEMM_Simple, Q4_0_BasicCorrectness)
{
    const std::string model_path = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

    // Load model using ModelContext
    auto model_ctx = ModelContext::create(model_path, nullptr);
    ASSERT_NE(model_ctx, nullptr) << "Failed to load model: " << model_path;

    // Get loader from context
    auto &loader = model_ctx->loader();

    // Get first attention weight tensor (Q4_0)
    std::string weight_name = "blk.0.attn_q.weight";
    const auto *weight_info = loader.getModel().findTensor(weight_name);
    ASSERT_NE(weight_info, nullptr) << "Weight tensor not found: " << weight_name;

    // Load weight tensor
    auto weight_tensor = loader.loadTensor(weight_name, 0);
    ASSERT_NE(weight_tensor, nullptr);
    ASSERT_EQ(weight_tensor->native_type(), TensorType::Q4_0) << "Expected Q4_0 tensor";

    // Get dimensions [N, K]
    auto shape = weight_tensor->shape();
    ASSERT_EQ(shape.size(), 2);
    int N = static_cast<int>(shape[0]); // Output features
    int K = static_cast<int>(shape[1]); // Input features
    int M = 4;                          // Small batch for testing

    // Ensure K is multiple of 32 (required for blockwise)
    ASSERT_EQ(K % 32, 0) << "K must be multiple of 32 for blockwise GEMM";

    // Dequantize weights to FP32
    std::vector<float> weight_fp32(N * K);
    weight_tensor->to_fp32(weight_fp32.data());

    // Create FP32 activation tensor (reuse first N rows of weights)
    std::vector<float> activation_fp32(M * K);
    for (int m = 0; m < M; ++m)
    {
        for (int k = 0; k < K; ++k)
        {
            activation_fp32[m * K + k] = weight_fp32[(m % N) * K + k];
        }
    }

    // Create Q8_1 activation tensor
    std::vector<size_t> act_shape = {static_cast<size_t>(M), static_cast<size_t>(K)};
    auto q8_1_activations = Q8_1Tensor::quantize_from_fp32(activation_fp32.data(), act_shape);
    ASSERT_NE(q8_1_activations, nullptr);

    // Reference path: FP32 GEMM (C = A @ B^T)
    std::vector<float> reference_output(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
    {
        for (int n = 0; n < N; ++n)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
            {
                sum += activation_fp32[m * K + k] * weight_fp32[n * K + k];
            }
            reference_output[m * N + n] = sum;
        }
    }

    // Microkernel path: Create OneDNNGemmKernel with weight tensor
    gemm_v4::OneDNNGemmKernel gemm_kernel(weight_tensor.get());

    // Get Q8_1 blocks via IQ8_1Decodable interface
    auto *decodable = dynamic_cast<IQ8_1Decodable *>(q8_1_activations.get());
    ASSERT_NE(decodable, nullptr);

    // Collect Q8_1 blocks for all rows
    size_t k_blocks = (K + 31) / 32;
    std::vector<Q8_1Block> blocks(M * k_blocks);
    for (int m = 0; m < M; ++m)
    {
        for (size_t kb = 0; kb < k_blocks; ++kb)
        {
            const Q8_1Block *block_ptr = decodable->decode_to_q8_1(m, kb);
            blocks[m * k_blocks + kb] = *block_ptr;
        }
    }

    // Allocate output buffer
    std::vector<float> microkernel_output(M * N, 0.0f);

    // Pack weights
    auto weight_pack = gemm_kernel.get_blockwise_weight_pack(K, N);
    ASSERT_TRUE(weight_pack.has_value()) << "Failed to pack weights for blockwise GEMM";

    // Execute blockwise GEMM
    bool success = gemm_kernel.execute_blockwise_gemm_test(
        blocks.data(), // Pointer to Q8_1Block array
        weight_pack.value(),
        microkernel_output.data(),
        M, N, K,
        nullptr, // No bias
        1.0f,    // alpha
        0.0f     // beta
    );
    ASSERT_TRUE(success) << "Blockwise GEMM execution failed";

    // Compare outputs (element-wise)
    float max_abs_diff = 0.0f;
    float max_rel_diff = 0.0f;
    for (int i = 0; i < M * N; ++i)
    {
        float abs_diff = std::abs(reference_output[i] - microkernel_output[i]);
        float rel_diff = abs_diff / (std::abs(reference_output[i]) + 1e-6f);
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);
    }

    // Q4_0 quantization introduces error, so use relaxed tolerance
    EXPECT_LT(max_abs_diff, 0.1f) << "Max absolute difference too large: " << max_abs_diff;
    EXPECT_LT(max_rel_diff, 0.1f) << "Max relative difference too large (10%): " << max_rel_diff;
}
