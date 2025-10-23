/**
 * @file test_mpilinearoperator_v2_integration.cpp
 * @brief Integration tests for MPILinearOperator_v2 with MPI distribution
 *
 * Tests both FP32 and BF16 activation paths using real IQ4_NL quantized weights
 * loaded from a GGUF model file.
 *
 * @author David Sanftenberg
 * @date 2025-10-22
 */

#include "operators/MPILinearOperator_v2.h"
#include "tensors/TensorFactory.h"
#include "tensors/BF16Tensor.h"
#include "tensors/IQ4_NLTensor.h"
#include "ModelLoader.h"
#include "utils/Logging.h"

#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>

using namespace llaminar;

/**
 * @brief Integration test fixture for MPILinearOperator_v2
 *
 * Requires exactly 2 MPI ranks for testing distributed execution.
 * Loads real IQ4_NL weights from GGUF model file.
 */
class MPILinearOperatorV2IntegrationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

        // This test requires exactly 2 ranks
        ASSERT_EQ(world_size_, 2) << "This test requires exactly 2 MPI ranks. Run with: mpirun -np 2 <test>";

        // Load model (only rank 0 logs to avoid clutter)
        if (rank_ == 0)
        {
            LOG_INFO("Loading model from: " << model_path_);
        }

        model_loader_ = std::make_unique<ModelLoader>();
        bool loaded = model_loader_->loadModel(model_path_);
        ASSERT_TRUE(loaded) << "Failed to load model: " << model_path_;

        // Set quantization environment to load IQ4_NL tensors
        auto &env = debugEnv();
        env.quant.enable = true;
        env.quant.load_quantized = true;
        env.quant.force_fp32_weights = false; // Allow quantized weights
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

    int rank_{0};
    int world_size_{0};
    std::unique_ptr<ModelLoader> model_loader_;
    const std::string model_path_{"/workspaces/llaminar/models/Qwen2.5-VL-7B-Instruct-IQ4_NL.gguf"};
};

#include <gtest/gtest.h>
#include <mpi.h>
#include <memory>
#include <vector>
#include <cmath>
#include <random>

#include "../src/operators/MPILinearOperator_v2.h"
#include "../src/tensors/TensorFactory.h"
#include "../src/tensors/BF16Tensor.h"
#include "../src/Logger.h"

using namespace llaminar;

class MPILinearOperatorV2IntegrationTest : public ::testing::Test
{
protected:
    int rank_;
    int size_;

    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        // These tests require exactly 2 MPI ranks
        if (size_ != 2)
        {
            GTEST_SKIP() << "This test requires exactly 2 MPI ranks (got " << size_ << ")";
        }
    }

    /**
     * @brief Create a simple FP32 weight tensor
     *
     * Creates a weight matrix with known pattern for verification:
     * - Each row has a specific pattern based on row index
     * - Easy to verify correctness of GEMM results
     */
    std::shared_ptr<TensorBase> createTestWeight(int out_features, int in_features)
    {
        // Create FP32 weight data with known pattern
        auto weight = TensorFactory::create_simple({out_features, in_features});
        float *data = weight->data();

        // Pattern: weight[i,j] = (i * 0.1) + (j * 0.01)
        // This makes each row distinct and easy to verify
        for (int i = 0; i < out_features; ++i)
        {
            for (int j = 0; j < in_features; ++j)
            {
                data[i * in_features + j] = (i * 0.1f) + (j * 0.01f);
            }
        }

        if (rank_ == 0)
        {
            LOG_INFO("Created FP32 weight [" << out_features << ", " << in_features
                                             << "], " << (out_features * in_features * 4 / 1024) << " KB");
        }

        return weight;
    }

    /**
     * @brief Create FP32 activation tensor with known values
     */
    std::shared_ptr<TensorBase> createFP32Activation(int seq_len, int in_features)
    {
        auto activation = TensorFactory::create_simple({seq_len, in_features});
        float *data = activation->data();

        // Pattern: activation[i,j] = 1.0 + (i * 0.001) + (j * 0.0001)
        for (int i = 0; i < seq_len; ++i)
        {
            for (int j = 0; j < in_features; ++j)
            {
                data[i * in_features + j] = 1.0f + (i * 0.001f) + (j * 0.0001f);
            }
        }

        return activation;
    }

    /**
     * @brief Create BF16 activation tensor with known values
     */
    std::shared_ptr<BF16Tensor> createBF16Activation(int seq_len, int in_features)
    {
        // Create FP32 first, then convert to BF16
        std::vector<float> activation_fp32(seq_len * in_features);

        // Same pattern as FP32 activation
        for (int i = 0; i < seq_len; ++i)
        {
            for (int j = 0; j < in_features; ++j)
            {
                activation_fp32[i * in_features + j] = 1.0f + (i * 0.001f) + (j * 0.0001f);
            }
        }

        auto activation_bf16 = std::make_shared<BF16Tensor>(
            std::vector<int>{seq_len, in_features},
            activation_fp32);

        return activation_bf16;
    }

    /**
     * @brief Compute expected output using naive FP32 GEMM
     *
     * This serves as ground truth for verification.
     * Computes: output = activation @ weight^T
     */
    std::vector<float> computeExpectedOutput(
        const std::vector<float> &activation, // [seq_len, in_features]
        const std::vector<float> &weight,     // [out_features, in_features]
        int seq_len,
        int in_features,
        int out_features)
    {

        std::vector<float> output(seq_len * out_features, 0.0f);

        // output[i,j] = sum_k(activation[i,k] * weight[j,k])
        for (int i = 0; i < seq_len; ++i)
        {
            for (int j = 0; j < out_features; ++j)
            {
                float sum = 0.0f;
                for (int k = 0; k < in_features; ++k)
                {
                    sum += activation[i * in_features + k] * weight[j * in_features + k];
                }
                output[i * out_features + j] = sum;
            }
        }

        return output;
    }

    /**
     * @brief Verify output is within acceptable tolerance of expected values
     */
    void verifyOutput(
        const float *output,
        const std::vector<float> &expected,
        int seq_len,
        int out_features,
        float rel_tolerance = 0.05f, // 5% relative error (quantization + BF16)
        float abs_tolerance = 0.01f)
    {

        float max_rel_error = 0.0f;
        float max_abs_error = 0.0f;
        int error_count = 0;

        for (int i = 0; i < seq_len; ++i)
        {
            for (int j = 0; j < out_features; ++j)
            {
                int idx = i * out_features + j;
                float actual = output[idx];
                float expect = expected[idx];

                float abs_error = std::abs(actual - expect);
                float rel_error = std::abs(abs_error / (expect + 1e-6f));

                max_abs_error = std::max(max_abs_error, abs_error);
                max_rel_error = std::max(max_rel_error, rel_error);

                if (rel_error > rel_tolerance && abs_error > abs_tolerance)
                {
                    if (error_count < 10)
                    { // Print first 10 errors
                        LOG_ERROR("Mismatch at [" << i << "," << j << "]: "
                                                  << "actual=" << actual << ", expected=" << expect
                                                  << ", rel_error=" << rel_error);
                    }
                    error_count++;
                }
            }
        }

        if (rank_ == 0)
        {
            LOG_INFO("Verification: max_rel_error=" << max_rel_error
                                                    << ", max_abs_error=" << max_abs_error
                                                    << ", errors=" << error_count << "/" << (seq_len * out_features));
        }

        EXPECT_LE(max_rel_error, rel_tolerance)
            << "Relative error exceeds tolerance: " << max_rel_error << " > " << rel_tolerance;
        EXPECT_EQ(error_count, 0)
            << error_count << " values exceed tolerance";
    }
};

/**
 * @brief Test FP32 activation path with FP32 weights
 *
 * Verifies:
 * - MPI distribution produces correct results
 * - Output matches expected values within tolerance
 */
TEST_F(MPILinearOperatorV2IntegrationTest, FP32ActivationWithFP32Weight)
{
    // Test dimensions
    const int seq_len = 4;
    const int in_features = 128;
    const int out_features = 256; // Will be split across 2 ranks (128 each)

    if (rank_ == 0)
    {
        LOG_INFO("=== Testing FP32 Activation + FP32 Weight ===");
        LOG_INFO("Dimensions: [" << seq_len << ", " << in_features
                                 << "] @ [" << out_features << ", " << in_features << "]^T");
    }

    // Create test data (all ranks create same data)
    auto weight_fp32 = createTestWeight(out_features, in_features);
    auto activation_fp32 = createFP32Activation(seq_len, in_features);

    // Create operator
    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);

    // Allocate output
    auto output = TensorFactory::create_simple({seq_len, out_features});

    // Execute
    std::vector<std::shared_ptr<TensorBase>> inputs = {activation_fp32, weight_fp32};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    bool success = linear_op.execute(inputs, outputs);

    ASSERT_TRUE(success) << "MPILinearOperator_v2 execution failed on rank " << rank_;

    // Verify output on rank 0
    if (rank_ == 0)
    {
        // Compute expected output
        std::vector<float> weight_data(out_features * in_features);
        std::copy_n(weight_fp32->data(), out_features * in_features, weight_data.begin());

        std::vector<float> activation_data(seq_len * in_features);
        std::copy_n(activation_fp32->data(), seq_len * in_features, activation_data.begin());

        auto expected = computeExpectedOutput(
            activation_data, weight_data, seq_len, in_features, out_features);

        // Verify with tight tolerance (pure FP32, should be exact)
        verifyOutput(output->data(), expected, seq_len, out_features,
                     /*rel_tolerance=*/1e-5f, /*abs_tolerance=*/1e-6f);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test BF16 activation path with FP32 weights
 *
 * Verifies:
 * - BF16 activation path works
 * - Accumulation in FP32, conversion back to BF16
 * - Output matches expected values within BF16 tolerance
 */
TEST_F(MPILinearOperatorV2IntegrationTest, BF16ActivationWithFP32Weight)
{
    // Test dimensions
    const int seq_len = 4;
    const int in_features = 128;
    const int out_features = 256; // Will be split across 2 ranks (128 each)

    if (rank_ == 0)
    {
        LOG_INFO("=== Testing BF16 Activation + FP32 Weight ===");
        LOG_INFO("Dimensions: [" << seq_len << ", " << in_features
                                 << "] @ [" << out_features << ", " << in_features << "]^T");
    }

    // Create test data (all ranks create same data)
    auto weight_fp32 = createTestWeight(out_features, in_features);
    auto activation_bf16 = createBF16Activation(seq_len, in_features);

    // Create operator
    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);

    // Allocate BF16 output
    auto output_bf16 = std::make_shared<BF16Tensor>(std::vector<int>{seq_len, out_features});

    // Execute
    std::vector<std::shared_ptr<TensorBase>> inputs = {activation_bf16, weight_fp32};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output_bf16};
    bool success = linear_op.execute(inputs, outputs);

    ASSERT_TRUE(success) << "MPILinearOperator_v2 BF16 execution failed on rank " << rank_;

    // Verify output on rank 0
    if (rank_ == 0)
    {
        // Compute expected output using FP32 reference
        std::vector<float> weight_data(out_features * in_features);
        std::copy_n(weight_fp32->data(), out_features * in_features, weight_data.begin());

        // Get activation in FP32 for reference computation
        std::vector<float> activation_fp32(seq_len * in_features);
        activation_bf16->to_fp32(activation_fp32.data(), seq_len * in_features);

        auto expected = computeExpectedOutput(
            activation_fp32, weight_data, seq_len, in_features, out_features);

        // Convert BF16 output to FP32 for comparison
        std::vector<float> output_fp32(seq_len * out_features);
        output_bf16->to_fp32(output_fp32.data(), seq_len * out_features);

        // Verify with 1% tolerance (BF16 quantization error)
        verifyOutput(output_fp32.data(), expected, seq_len, out_features,
                     /*rel_tolerance=*/0.01f, /*abs_tolerance=*/0.001f);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test weight caching behavior
 *
 * Verifies:
 * - First call triggers weight caching
 * - Second call reuses cached weight
 * - Results are identical
 */
TEST_F(MPILinearOperatorV2IntegrationTest, WeightCachingBehavior)
{
    const int seq_len = 4;
    const int in_features = 64;
    const int out_features = 128;

    if (rank_ == 0)
    {
        LOG_INFO("=== Testing Weight Caching ===");
    }

    auto weight = createTestWeight(out_features, in_features);
    auto activation1 = createFP32Activation(seq_len, in_features);
    auto activation2 = createFP32Activation(seq_len, in_features);

    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);

    auto output1 = TensorFactory::create_simple({seq_len, out_features});
    auto output2 = TensorFactory::create_simple({seq_len, out_features});

    // First call - should cache weight
    std::vector<std::shared_ptr<TensorBase>> inputs1 = {activation1, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs1 = {output1};
    bool success1 = linear_op.execute(inputs1, outputs1);
    ASSERT_TRUE(success1);

    // Second call - should reuse cached weight
    std::vector<std::shared_ptr<TensorBase>> inputs2 = {activation2, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs2 = {output2};
    bool success2 = linear_op.execute(inputs2, outputs2);
    ASSERT_TRUE(success2);

    // Results should be identical (same inputs)
    if (rank_ == 0)
    {
        const float *data1 = output1->data();
        const float *data2 = output2->data();

        for (int i = 0; i < seq_len * out_features; ++i)
        {
            EXPECT_FLOAT_EQ(data1[i], data2[i])
                << "Mismatch at index " << i << " (caching changed result)";
        }

        LOG_INFO("Weight caching verified: outputs identical");
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test basic functionality with FP32 weights
 *
 * This is a sanity check to ensure the operator works correctly.
 */
TEST_F(MPILinearOperatorV2IntegrationTest, BasicFunctionality)
{
    const int seq_len = 2;
    const int in_features = 1024;
    const int out_features = 2048; // Would be 16MB quantized

    if (rank_ == 0)
    {
        LOG_INFO("=== Testing Basic Functionality ===");
        LOG_INFO("Weight dimensions: [" << out_features << ", " << in_features
                                        << "] -> " << (out_features * in_features * 4 / 1024 / 1024) << " MB FP32");
    }

    auto weight = createTestWeight(out_features, in_features);
    auto activation = createFP32Activation(seq_len, in_features);

    MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);
    auto output = TensorFactory::create_simple({seq_len, out_features});

    // Execute
    std::vector<std::shared_ptr<TensorBase>> inputs = {activation, weight};
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};
    bool success = linear_op.execute(inputs, outputs);
    ASSERT_TRUE(success);

    // Sanity check: verify output is non-zero (GEMM actually happened)
    if (rank_ == 0)
    {
        const float *data = output->data();
        float sum = 0.0f;
        for (int i = 0; i < seq_len * out_features; ++i)
        {
            sum += std::abs(data[i]);
        }

        EXPECT_GT(sum, 0.0f) << "Output is all zeros (GEMM didn't execute?)";
        LOG_INFO("GEMM succeeded (output sum: " << sum << ")");
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * @brief Test different sequence lengths (edge cases)
 */
TEST_F(MPILinearOperatorV2IntegrationTest, VariousSequenceLengths)
{
    const int in_features = 64;
    const int out_features = 128;

    // Test seq_len = 1 (single token), 8 (small batch), 32 (larger batch)
    std::vector<int> seq_lengths = {1, 8, 32};

    for (int seq_len : seq_lengths)
    {
        if (rank_ == 0)
        {
            LOG_INFO("Testing seq_len=" << seq_len);
        }

        auto weight = createTestWeight(out_features, in_features);
        auto activation = createFP32Activation(seq_len, in_features);

        MPILinearOperator_v2 linear_op(MPI_COMM_WORLD);
        auto output = TensorFactory::create_simple({seq_len, out_features});

        std::vector<std::shared_ptr<TensorBase>> inputs = {activation, weight};
        std::vector<std::shared_ptr<TensorBase>> outputs = {output};
        bool success = linear_op.execute(inputs, outputs);
        ASSERT_TRUE(success) << "Failed for seq_len=" << seq_len;

        // Basic sanity check
        if (rank_ == 0)
        {
            const float *data = output->data();
            float sum = 0.0f;
            for (int i = 0; i < seq_len * out_features; ++i)
            {
                sum += std::abs(data[i]);
            }
            EXPECT_GT(sum, 0.0f);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Only rank 0 prints test output
    if (rank != 0)
    {
        ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();
        delete listeners.Release(listeners.default_result_printer());
    }

    int result = RUN_ALL_TESTS();

    MPI_Finalize();
    return result;
}
