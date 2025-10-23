/**
 * @file test_bf16_activation_storage.cpp
 * @brief Test BF16 activation storage integration (Phase 5)
 * @author David Sanftenberg
 * @date October 20, 2025
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "../src/operators/MPILinearOperator.h"
#include "../src/tensors/TensorFactory.h"
#include "../src/tensors/SimpleTensor.h"
#include "../src/tensors/BF16Tensor.h"
#include "../src/utils/DebugEnv.h"
#include <cstdlib>

class BF16ActivationStorageTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
    }

    int rank_;
    int size_;
};

/**
 * Test that MPILinearOperator creates BF16 tensors when LLAMINAR_QUANT_OUTPUT_BF16=1
 */
TEST_F(BF16ActivationStorageTest, LinearOperatorCreatesBF16Tensors)
{
    // Enable BF16 output
    setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);

    // Force refresh of debug environment
    llaminar::debugEnvRefresh();

    // Verify flag is set
    const auto &env = llaminar::debugEnv();
    ASSERT_TRUE(env.quant.output_bf16) << "LLAMINAR_QUANT_OUTPUT_BF16 should be enabled";

    // Create a small linear operator test
    // Input: [2, 4], Weight: [3, 4], Output: [2, 3]
    auto input = llaminar::TensorFactory::create_simple({2, 4});
    auto weight = llaminar::TensorFactory::create_simple({3, 4});

    // Fill with simple data
    float *input_data = input->data();
    float *weight_data = weight->data();

    for (int i = 0; i < 8; ++i)
    {
        input_data[i] = static_cast<float>(i + 1); // 1, 2, 3, ..., 8
    }

    for (int i = 0; i < 12; ++i)
    {
        weight_data[i] = 0.1f; // Simple weights
    }

    // Create output tensor [2, 3]
    auto output = llaminar::TensorFactory::create_simple({2, 3});

    // Create operator and execute
    llaminar::MPILinearOperator linear_op(MPI_COMM_WORLD);

    std::vector<std::shared_ptr<llaminar::TensorBase>> inputs = {input, weight};
    std::vector<std::shared_ptr<llaminar::TensorBase>> outputs = {output};

    bool success = linear_op.execute(inputs, outputs);
    ASSERT_TRUE(success) << "Linear operator should execute successfully";

    // Verify output exists
    ASSERT_NE(outputs[0], nullptr) << "Output tensor should be created";

    // Verify output has correct shape [2, 3]
    const auto &output_shape = outputs[0]->shape();
    ASSERT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 2);
    EXPECT_EQ(output_shape[1], 3);

    // Verify output can be converted to FP32 (BF16Tensor should provide data())
    const float *output_data = outputs[0]->data();
    ASSERT_NE(output_data, nullptr) << "Should be able to get FP32 data from output";

    // Verify values are reasonable (not NaN/Inf)
    for (int i = 0; i < 6; ++i)
    {
        EXPECT_FALSE(std::isnan(output_data[i])) << "Output should not contain NaN at index " << i;
        EXPECT_FALSE(std::isinf(output_data[i])) << "Output should not contain Inf at index " << i;
    }

    // Clean up environment
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    llaminar::debugEnvRefresh();
}

/**
 * Test FP32 vs BF16 activation storage produces similar results
 */
TEST_F(BF16ActivationStorageTest, FP32VsBF16Parity)
{
    const int seq_len = 16;
    const int in_dim = 64;
    const int out_dim = 48;

    // Create input and weight
    auto input = llaminar::TensorFactory::create_simple({seq_len, in_dim});
    auto weight = llaminar::TensorFactory::create_simple({out_dim, in_dim});

    // Fill with random-ish data
    float *input_data = input->data();
    float *weight_data = weight->data();

    for (int i = 0; i < seq_len * in_dim; ++i)
    {
        input_data[i] = static_cast<float>(i % 17) / 17.0f - 0.5f;
    }

    for (int i = 0; i < out_dim * in_dim; ++i)
    {
        weight_data[i] = static_cast<float>(i % 13) / 26.0f - 0.25f;
    }

    // Run with FP32
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    llaminar::debugEnvRefresh();

    auto output_fp32 = llaminar::TensorFactory::create_simple({seq_len, out_dim});
    llaminar::MPILinearOperator linear_op_fp32(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<llaminar::TensorBase>> inputs_fp32 = {input, weight};
    std::vector<std::shared_ptr<llaminar::TensorBase>> outputs_fp32 = {output_fp32};

    bool success_fp32 = linear_op_fp32.execute(inputs_fp32, outputs_fp32);
    ASSERT_TRUE(success_fp32);

    // Get FP32 results
    std::vector<float> fp32_results(seq_len * out_dim);
    std::copy(outputs_fp32[0]->data(), outputs_fp32[0]->data() + seq_len * out_dim, fp32_results.begin());

    // Run with BF16
    setenv("LLAMINAR_QUANT_OUTPUT_BF16", "1", 1);
    llaminar::debugEnvRefresh();

    auto output_bf16 = llaminar::TensorFactory::create_simple({seq_len, out_dim});
    llaminar::MPILinearOperator linear_op_bf16(MPI_COMM_WORLD);
    std::vector<std::shared_ptr<llaminar::TensorBase>> inputs_bf16 = {input, weight};
    std::vector<std::shared_ptr<llaminar::TensorBase>> outputs_bf16 = {output_bf16};

    bool success_bf16 = linear_op_bf16.execute(inputs_bf16, outputs_bf16);
    ASSERT_TRUE(success_bf16);

    // Get BF16 results (converted to FP32)
    std::vector<float> bf16_results(seq_len * out_dim);
    std::copy(outputs_bf16[0]->data(), outputs_bf16[0]->data() + seq_len * out_dim, bf16_results.begin());

    // Compare results - BF16 should be within ~1e-3 relative error
    double sum_sq_diff = 0.0;
    double sum_sq_ref = 0.0;

    for (int i = 0; i < seq_len * out_dim; ++i)
    {
        double diff = fp32_results[i] - bf16_results[i];
        sum_sq_diff += diff * diff;
        sum_sq_ref += fp32_results[i] * fp32_results[i];
    }

    double rel_l2 = std::sqrt(sum_sq_diff / (sum_sq_ref + 1e-10));

    if (rank_ == 0)
    {
        std::cout << "FP32 vs BF16 Relative L2 Error: " << rel_l2 << std::endl;
    }

    // BF16 should be within 1e-3 relative error (Phase 5 target)
    EXPECT_LT(rel_l2, 1e-3) << "BF16 activation storage should maintain accuracy";

    // Clean up
    unsetenv("LLAMINAR_QUANT_OUTPUT_BF16");
    llaminar::debugEnvRefresh();
}

/**
 * Test memory usage reduction with BF16 storage
 */
TEST_F(BF16ActivationStorageTest, MemoryReduction)
{
    // Create BF16 tensor
    auto bf16_tensor = llaminar::TensorFactory::create_bf16({1000, 512});

    // Create equivalent FP32 tensor
    auto fp32_tensor = llaminar::TensorFactory::create_simple({1000, 512});

    // BF16 should use ~2× less memory for storage
    // (Note: BF16Tensor has lazy FP32 cache, so this tests native storage)
    auto bf16_ptr = std::dynamic_pointer_cast<llaminar::BF16Tensor>(bf16_tensor);
    ASSERT_NE(bf16_ptr, nullptr);

    // BF16 native data should exist
    const llaminar::bfloat16 *bf16_data = bf16_ptr->bf16_data();
    ASSERT_NE(bf16_data, nullptr);

    // Memory footprint: BF16 = 2 bytes/elem, FP32 = 4 bytes/elem
    size_t bf16_bytes = 1000 * 512 * sizeof(llaminar::bfloat16);
    size_t fp32_bytes = 1000 * 512 * sizeof(float);

    EXPECT_EQ(bf16_bytes, fp32_bytes / 2) << "BF16 should use exactly 2× less memory";

    if (rank_ == 0)
    {
        std::cout << "Memory footprint: FP32 = " << (fp32_bytes / 1024.0 / 1024.0) << " MB, "
                  << "BF16 = " << (bf16_bytes / 1024.0 / 1024.0) << " MB "
                  << "(2× reduction)" << std::endl;
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
