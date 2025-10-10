/**
 * @file test_attention_bias_validation.cpp
 * @brief Test that MPIAttentionKernel properly validates bias tensor sizes
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "kernels/MPIAttentionKernel.h"
#include "tensors/tensor_factory.h"
#include "logger.h"

using namespace llaminar;

class AttentionBiasValidationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // MPI should already be initialized by main()
    }
};

/**
 * @brief Test that kernel rejects bias tensors with incorrect sizes
 */
TEST_F(AttentionBiasValidationTest, RejectIncorrectBiasSize)
{
    // Create kernel: 4 heads, 2 KV heads, head_dim=64
    // Expected dimensions: Q bias = 4*64 = 256, K/V bias = 2*64 = 128
    const int n_head = 4;
    const int n_head_kv = 2;
    const int head_dim = 64;
    const int d_model = n_head * head_dim; // 256

    MPIAttentionKernel kernel(n_head, n_head_kv, head_dim, 10000.0f,
                              MPIAttentionKernel::DistributionStrategy::HEAD_WISE);

    // Create valid inputs
    const int seq_len = 4;
    auto input = TensorFactory::create_simple({seq_len, d_model});
    auto wq = TensorFactory::create_simple({d_model, d_model});              // [256, 256]
    auto wk = TensorFactory::create_simple({n_head_kv * head_dim, d_model}); // [128, 256]
    auto wv = TensorFactory::create_simple({n_head_kv * head_dim, d_model}); // [128, 256]
    auto wo = TensorFactory::create_simple({d_model, d_model});              // [256, 256]
    auto k_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});
    auto v_cache = TensorFactory::create_simple({seq_len, n_head_kv * head_dim});

    // Test 1: Wrong size Q bias (should be 256, give it 100)
    {
        auto bq_wrong = TensorFactory::create_simple({100}); // WRONG SIZE!
        auto bk_null = TensorFactory::create_simple({1});
        auto bv_null = TensorFactory::create_simple({1});

        std::vector<std::shared_ptr<TensorBase>> inputs = {
            input, wq, wk, wv, wo, bq_wrong, bk_null, bv_null, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs(1);

        bool result = kernel.execute(inputs, outputs);
        EXPECT_FALSE(result) << "Kernel should reject Q bias with size 100 (expected 256)";
    }

    // Test 2: Wrong size K bias (should be 128, give it 50)
    {
        auto bq_null = TensorFactory::create_simple({1});
        auto bk_wrong = TensorFactory::create_simple({50}); // WRONG SIZE!
        auto bv_null = TensorFactory::create_simple({1});

        std::vector<std::shared_ptr<TensorBase>> inputs = {
            input, wq, wk, wv, wo, bq_null, bk_wrong, bv_null, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs(1);

        bool result = kernel.execute(inputs, outputs);
        EXPECT_FALSE(result) << "Kernel should reject K bias with size 50 (expected 128)";
    }

    // Test 3: Wrong size V bias (should be 128, give it 200)
    {
        auto bq_null = TensorFactory::create_simple({1});
        auto bk_null = TensorFactory::create_simple({1});
        auto bv_wrong = TensorFactory::create_simple({200}); // WRONG SIZE!

        std::vector<std::shared_ptr<TensorBase>> inputs = {
            input, wq, wk, wv, wo, bq_null, bk_null, bv_wrong, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs(1);

        bool result = kernel.execute(inputs, outputs);
        EXPECT_FALSE(result) << "Kernel should reject V bias with size 200 (expected 128)";
    }

    // Test 4: Correct sizes should work (or at least pass validation)
    {
        auto bq_correct = TensorFactory::create_simple({d_model});              // 256
        auto bk_correct = TensorFactory::create_simple({n_head_kv * head_dim}); // 128
        auto bv_correct = TensorFactory::create_simple({n_head_kv * head_dim}); // 128

        // Fill with zeros to avoid computation issues
        std::fill_n(bq_correct->data(), bq_correct->size(), 0.0f);
        std::fill_n(bk_correct->data(), bk_correct->size(), 0.0f);
        std::fill_n(bv_correct->data(), bv_correct->size(), 0.0f);

        std::vector<std::shared_ptr<TensorBase>> inputs = {
            input, wq, wk, wv, wo, bq_correct, bk_correct, bv_correct, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs(1);

        bool result = kernel.execute(inputs, outputs);
        // Note: May still fail due to uninitialized data, but should NOT fail bias validation
        // Check logs to ensure no "bias size mismatch" error
    }

    // Test 5: nullptr and size-1 biases should be accepted
    {
        auto bq_null = TensorFactory::create_simple({1});
        auto bk_null = TensorFactory::create_simple({1});
        auto bv_null = TensorFactory::create_simple({1});

        std::vector<std::shared_ptr<TensorBase>> inputs = {
            input, wq, wk, wv, wo, bq_null, bk_null, bv_null, k_cache, v_cache};
        std::vector<std::shared_ptr<TensorBase>> outputs(1);

        bool result = kernel.execute(inputs, outputs);
        // Should NOT fail due to bias validation (size <= 1 is allowed)
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
