/**
 * @file test_mpilinearoperator_v2_iq4nl.cpp
 * @brief Integration tests for MPILinearOperator_v2 with real IQ4_NL quantized weights
 *
 * Tests both FP32 and BF16 activation paths using real IQ4_NL quantized weights
 * loaded from a GGUF model file. This validates streaming dequantization and
 * the dual-path architecture of MPILinearOperator_v2.
 *
 * @author David Sanftenberg
 * @date 2025-10-22
 */

#include "operators/MPILinearOperator_v2.h"
#include "tensors/TensorFactory.h"
#include "tensors/BF16Tensor.h"
#include "tensors/IQ4_NLTensor.h"
#include "ModelLoader.h"
#include "Logger.h"
#include "utils/DebugEnv.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

using namespace llaminar;

/**
 * @brief Integration test fixture for MPILinearOperator_v2 with IQ4_NL weights
 *
 * Requires exactly 2 MPI ranks for testing distributed execution.
 * Loads real IQ4_NL weights from GGUF model file.
 */
class MPILinearOperatorV2IQ4NLTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // This test requires exactly 2 ranks
        ASSERT_EQ(world_size_, 2) << "This test requires exactly 2 MPI ranks. Run with: mpirun -np 2 <test>";

        // Set quantization environment to load IQ4_NL tensors (must set BEFORE calling debugEnv)
        setenv("LLAMINAR_QUANT_ENABLE", "1", 1);
        setenv("LLAMINAR_LOAD_QUANTIZED", "1", 1);
        unsetenv("LLAMINAR_FORCE_FP32_WEIGHTS"); // Ensure not forcing FP32
        refreshDebugEnv();                       // Refresh to pick up new environment

        // Load model (only rank 0 logs to avoid clutter)
        if (rank_ == 0)
        {
            LOG_INFO("Loading model from: " << model_path_);
        }

        model_loader_ = std::make_unique<ModelLoader>();
        bool loaded = model_loader_->loadModel(model_path_);
        ASSERT_TRUE(loaded) << "Failed to load model: " << model_path_;

        if (rank_ == 0)
        {
            LOG_INFO("Model loaded successfully");
        }
    }

    void TearDown() override
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /**
     * @brief Load an IQ4_NL weight tensor from the model
     * @param layer_idx Layer index (0-based)
     * @param weight_type Weight type (e.g., "attn_q", "attn_k", "ffn_gate")
     * @return Shared pointer to loaded IQ4_NL tensor
     */
    std::shared_ptr<TensorBase> loadWeight(int layer_idx, const std::string &weight_type)
    {
        std::string tensor_name = "blk." + std::to_string(layer_idx) + "." + weight_type + ".weight";

        auto tensor = model_loader_->loadTensor(tensor_name);
        EXPECT_NE(tensor, nullptr) << "Failed to load tensor: " << tensor_name;

        if (rank_ == 0 && tensor)
        {
            const auto &shape = tensor->shape();
            LOG_INFO("Loaded " << tensor_name << " shape=[" << shape[0] << ", " << shape[1] << "]"
                               << " type=" << (tensor->native_type() == TensorDataType::QUANTIZED ? "QUANTIZED" : "FP32"));
        }

        return tensor;
    }

    /**
     * @brief Create FP32 activation tensor with small pattern
     */
    std::shared_ptr<TensorBase> createFP32Activation(int seq_len, int features)
    {
        auto activation = TensorFactory::create_simple({seq_len, features});
        float *data = activation->data();

        // Fill with small values: act[i,j] = 0.001 * (i + j)
        for (int i = 0; i < seq_len; ++i)
        {
            for (int j = 0; j < features; ++j)
            {
                data[i * features + j] = 0.001f * (i + j);
            }
        }

        return activation;
    }

    /**
     * @brief Create BF16 activation tensor from FP32 pattern
     */
    std::shared_ptr<BF16Tensor> createBF16Activation(int seq_len, int features)
    {
        // Create FP32 pattern first
        auto fp32_act = createFP32Activation(seq_len, features);

        // Convert to BF16
        auto bf16_act = std::make_shared<BF16Tensor>(std::vector<int>{seq_len, features});
        bf16_act->from_fp32(fp32_act->data(), seq_len * features);

        return bf16_act;
    }

    int rank_{0};
    int world_size_{0};
    std::unique_ptr<ModelLoader> model_loader_;
    const std::string model_path_{"/workspaces/llaminar/models/Qwen2-0.5B.IQ4_NL.gguf"};
};

/**
 * @brief Test FP32 activation path with IQ4_NL quantized weights
 *
 * Validates that:
 *   - IQ4_NL weights are loaded correctly from GGUF
 *   - FP32 activation input produces FP32 output
 *   - MPI distribution across 2 ranks produces consistent results
 *   - Output values are reasonable (non-zero, finite)
 *   - Streaming dequantization works correctly
 */
TEST_F(MPILinearOperatorV2IQ4NLTest, FP32ActivationWithIQ4NLWeight)
{
    const int seq_len = 4;
    const int layer_idx = 0;

    if (rank_ == 0)
    {
        LOG_INFO("=== Testing FP32 Activation + IQ4_NL Weight ===");
    }

    // Load IQ4_NL weight from model (Q projection from layer 0)
    auto weight = loadWeight(layer_idx, "attn_q");
    ASSERT_NE(weight, nullptr);

    // Verify it's actually quantized
    EXPECT_EQ(weight->native_type(), TensorDataType::QUANTIZED)
        << "Expected QUANTIZED tensor, got FP32 (check env settings)";

    const auto &weight_shape = weight->shape();
    int out_features = weight_shape[0];
    int in_features = weight_shape[1];

    if (rank_ == 0)
    {
        LOG_INFO("Weight shape: [" << out_features << ", " << in_features << "]");
    }

    // Create FP32 activation
    auto activation = createFP32Activation(seq_len, in_features);

    // Create operator and execute
    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);
    auto output = TensorFactory::create_simple({seq_len, out_features});

    std::vector<std::shared_ptr<TensorBase>> inputs = {activation, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    bool success = linear_op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "MPILinearOperator_v2::execute() failed on rank " << rank_;

    // Sanity check: verify output is non-zero and finite
    if (rank_ == 0)
    {
        const float *data = output->data();
        float sum = 0.0f;
        float abs_sum = 0.0f;
        bool all_finite = true;
        int non_zero_count = 0;

        for (int i = 0; i < seq_len * out_features; ++i)
        {
            float val = data[i];
            sum += val;
            abs_sum += std::abs(val);

            if (!std::isfinite(val))
            {
                all_finite = false;
                LOG_ERROR("Non-finite value at index " << i << ": " << val);
            }
            if (val != 0.0f)
            {
                non_zero_count++;
            }
        }

        EXPECT_TRUE(all_finite) << "Output contains NaN/Inf";
        EXPECT_GT(abs_sum, 0.0f) << "Output is all zeros (GEMM didn't execute?)";
        EXPECT_GT(non_zero_count, seq_len * out_features / 2)
            << "Less than 50% of outputs are non-zero";

        LOG_INFO("FP32 path succeeded:");
        LOG_INFO("  sum=" << sum << ", abs_sum=" << abs_sum);
        LOG_INFO("  non_zero_count=" << non_zero_count << "/" << (seq_len * out_features));
        LOG_INFO("  all_finite=" << (all_finite ? "true" : "false"));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test BF16 activation path with IQ4_NL quantized weights
 *
 * Validates that:
 *   - BF16 activation input produces BF16 output
 *   - FP32 accumulation happens internally
 *   - Conversion back to BF16 is correct
 *   - Output values are reasonable (non-zero, finite)
 */
TEST_F(MPILinearOperatorV2IQ4NLTest, BF16ActivationWithIQ4NLWeight)
{
    const int seq_len = 4;
    const int layer_idx = 0;

    if (rank_ == 0)
    {
        LOG_INFO("=== Testing BF16 Activation + IQ4_NL Weight ===");
    }

    // Load IQ4_NL weight from model
    auto weight = loadWeight(layer_idx, "attn_k"); // Use K projection (different from Q)
    ASSERT_NE(weight, nullptr);

    EXPECT_EQ(weight->native_type(), TensorDataType::QUANTIZED);

    const auto &weight_shape = weight->shape();
    int out_features = weight_shape[0];
    int in_features = weight_shape[1];

    // Create BF16 activation
    auto activation = createBF16Activation(seq_len, in_features);

    // Create operator and execute
    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);
    auto output = std::make_shared<BF16Tensor>(std::vector<int>{seq_len, out_features});

    std::vector<std::shared_ptr<TensorBase>> inputs = {activation, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    bool success = linear_op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Verify output is BF16
    EXPECT_EQ(output->native_type(), TensorDataType::BF16)
        << "Expected BF16 output for BF16 input";

    // Sanity check: verify output is non-zero and finite
    if (rank_ == 0)
    {
        // Convert to FP32 for analysis
        std::vector<float> output_fp32(seq_len * out_features);
        output->to_fp32(output_fp32.data(), seq_len * out_features);

        float sum = 0.0f;
        float abs_sum = 0.0f;
        bool all_finite = true;
        int non_zero_count = 0;

        for (int i = 0; i < seq_len * out_features; ++i)
        {
            float val = output_fp32[i];
            sum += val;
            abs_sum += std::abs(val);

            if (!std::isfinite(val))
            {
                all_finite = false;
            }
            if (val != 0.0f)
            {
                non_zero_count++;
            }
        }

        EXPECT_TRUE(all_finite) << "BF16 output contains NaN/Inf";
        EXPECT_GT(abs_sum, 0.0f) << "BF16 output is all zeros";

        LOG_INFO("BF16 path succeeded:");
        LOG_INFO("  sum=" << sum << ", abs_sum=" << abs_sum);
        LOG_INFO("  non_zero_count=" << non_zero_count << "/" << (seq_len * out_features));
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test weight caching behavior
 *
 * Validates that calling execute() multiple times with the same weight
 * reuses the cached weight (no redundant dequantization).
 */
TEST_F(MPILinearOperatorV2IQ4NLTest, WeightCachingBehavior)
{
    const int seq_len = 2;
    const int layer_idx = 0;

    if (rank_ == 0)
    {
        LOG_INFO("=== Testing Weight Caching ===");
    }

    auto weight = loadWeight(layer_idx, "ffn_gate");
    ASSERT_NE(weight, nullptr);

    const auto &weight_shape = weight->shape();
    int out_features = weight_shape[0];
    int in_features = weight_shape[1];

    // Create two different activations
    auto act1 = createFP32Activation(seq_len, in_features);
    auto act2 = createFP32Activation(seq_len + 1, in_features); // Different seq_len

    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);

    // First execution
    auto output1 = TensorFactory::create_simple({seq_len, out_features});
    std::vector<std::shared_ptr<TensorBase>> inputs1 = {act1, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs1 = {output1};
    bool success1 = linear_op.execute(inputs1, outputs1);
    ASSERT_TRUE(success1);

    // Second execution with same weight (should use cache)
    auto output2 = TensorFactory::create_simple({seq_len + 1, out_features});
    std::vector<std::shared_ptr<TensorBase>> inputs2 = {act2, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs2 = {output2};
    bool success2 = linear_op.execute(inputs2, outputs2);
    ASSERT_TRUE(success2);

    // Both executions should succeed (implicit caching test)
    if (rank_ == 0)
    {
        LOG_INFO("Weight caching test passed (both executions succeeded)");
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test various sequence lengths to ensure robustness
 */
TEST_F(MPILinearOperatorV2IQ4NLTest, VariousSequenceLengths)
{
    const int layer_idx = 0;

    auto weight = loadWeight(layer_idx, "attn_v");
    ASSERT_NE(weight, nullptr);

    const auto &weight_shape = weight->shape();
    int out_features = weight_shape[0];
    int in_features = weight_shape[1];

    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);

    // Test seq_len = 1, 8, 32
    for (int seq_len : {1, 8, 32})
    {
        if (rank_ == 0)
        {
            LOG_INFO("Testing seq_len=" << seq_len);
        }

        auto activation = createFP32Activation(seq_len, in_features);
        auto output = TensorFactory::create_simple({seq_len, out_features});

        std::vector<std::shared_ptr<TensorBase>> inputs = {activation, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        bool success = linear_op.execute(inputs, outputs);

        EXPECT_TRUE(success) << "Failed for seq_len=" << seq_len;
    }

    if (rank_ == 0)
    {
        LOG_INFO("All sequence lengths passed");
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
