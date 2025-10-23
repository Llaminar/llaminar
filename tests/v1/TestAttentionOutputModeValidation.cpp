/**
 * @file TestAttentionOutputModeValidation.cpp
 * @brief Unit test to validate MPIAttentionOperator output mode configuration
 *
 * This test ensures that the critical output mode configuration is correctly set
 * for multi-rank execution. It catches the bug where LocalHeads mode (default)
 * was incorrectly used with row-partitioned weights, causing 98.6% parity failures.
 *
 * @author David Sanftenberg
 */

#include <gtest/gtest.h>
#include <mpi.h>
#include "operators/MPIAttentionOperator.h"
#include "tensors/TensorFactory.h"
#include "Logger.h"

using namespace llaminar;

class AttentionOutputModeTest : public ::testing::Test
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
 * @brief Test that LocalHeads mode with non-sharded weights on multiple ranks throws error
 *
 * This is a regression test for the critical bug where LocalHeads mode was used
 * with row-partitioned W_o weights, causing partial results without MPI reduction.
 */
TEST_F(AttentionOutputModeTest, LocalHeadsModeWithReplicatedWeightsThrows)
{
    if (size_ < 2)
    {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int n_head = 14;
    const int n_head_kv = 2;
    const int head_dim = 64;
    const int d_model = n_head * head_dim;
    const int seq_len = 5;

    // Create attention kernel with DEFAULT LocalHeads mode (this should fail!)
    auto attention_kernel = std::make_unique<MPIAttentionOperator>(
        n_head, n_head_kv, head_dim);

    // DO NOT call setOutputMode() - use the dangerous default

    // Create non-sharded (replicated) weight tensors
    auto input = TensorFactory::create_simple({seq_len, d_model});
    auto wq = TensorFactory::create_simple({d_model, d_model});
    auto wk = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    auto wv = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    auto wo = TensorFactory::create_simple({d_model, d_model}); // Row-partitioned, NOT head-sharded!
    auto k_cache = TensorFactory::create_simple({0, n_head_kv * head_dim});
    auto v_cache = TensorFactory::create_simple({0, n_head_kv * head_dim});

    // Fill with test data
    std::fill(input->data(), input->data() + input->size(), 0.1f);
    std::fill(wq->data(), wq->data() + wq->size(), 0.01f);
    std::fill(wk->data(), wk->data() + wk->size(), 0.01f);
    std::fill(wv->data(), wv->data() + wv->size(), 0.01f);
    std::fill(wo->data(), wo->data() + wo->size(), 0.01f);

    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input, wq, wk, wv, wo, k_cache, v_cache};

    auto output = TensorFactory::create_simple({seq_len, d_model});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // This should THROW because:
    // - Weights are NOT head-sharded
    // - Multiple ranks (size_ >= 2)
    // - Output mode is LocalHeads (default)
    EXPECT_THROW(
        {
            attention_kernel->execute(inputs, outputs);
        },
        std::runtime_error);

    if (rank_ == 0)
    {
        LOG_INFO("[TEST] ✓ Correctly detected invalid LocalHeads mode with non-sharded weights");
    }
}

/**
 * @brief Test that GatherHeadsPostProjection mode works correctly with replicated weights
 *
 * This validates the fix - when we explicitly set GatherHeadsPostProjection mode,
 * the kernel should not throw the configuration error (actual execution tested elsewhere).
 */
TEST_F(AttentionOutputModeTest, GatherHeadsPostProjectionModeDoesNotThrow)
{
    if (size_ < 2)
    {
        GTEST_SKIP() << "Test requires at least 2 MPI ranks";
    }

    const int n_head = 14;
    const int n_head_kv = 2;
    const int head_dim = 64;

    // Create attention kernel and EXPLICITLY set the correct mode
    auto attention_kernel = std::make_unique<MPIAttentionOperator>(
        n_head, n_head_kv, head_dim);

    // FIX: Set GatherHeadsPostProjection mode for row-partitioned weights
    attention_kernel->setOutputMode(MPIAttentionOperator::AttentionOutputMode::GatherHeadsPostProjection);

    // Verify the mode was set correctly
    EXPECT_EQ(attention_kernel->outputMode(), MPIAttentionOperator::AttentionOutputMode::GatherHeadsPostProjection);

    if (rank_ == 0)
    {
        LOG_INFO("[TEST] ✓ GatherHeadsPostProjection mode can be set correctly");
    }

    // Note: Full execution test is covered by test_mpi_attention_components.cpp
    // This test focuses on configuration validation
}

/**
 * @brief Test that single-rank execution doesn't trigger the assertion
 *
 * LocalHeads mode is technically "safe" on single rank (though not useful),
 * so the assertion should not trigger.
 */
TEST_F(AttentionOutputModeTest, SingleRankDoesNotTriggerAssertion)
{
    if (size_ != 1)
    {
        GTEST_SKIP() << "Test requires exactly 1 MPI rank";
    }

    const int n_head = 14;
    const int n_head_kv = 2;
    const int head_dim = 64;
    const int d_model = n_head * head_dim;
    const int seq_len = 5;

    // Create attention kernel with default LocalHeads mode
    auto attention_kernel = std::make_unique<MPIAttentionOperator>(
        n_head, n_head_kv, head_dim);

    // Create non-sharded tensors
    auto input = TensorFactory::create_simple({seq_len, d_model});
    auto wq = TensorFactory::create_simple({d_model, d_model});
    auto wk = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    auto wv = TensorFactory::create_simple({d_model, n_head_kv * head_dim});
    auto wo = TensorFactory::create_simple({d_model, d_model});
    auto k_cache = TensorFactory::create_simple({0, n_head_kv * head_dim});
    auto v_cache = TensorFactory::create_simple({0, n_head_kv * head_dim});

    std::fill(input->data(), input->data() + input->size(), 0.1f);
    std::fill(wq->data(), wq->data() + wq->size(), 0.01f);
    std::fill(wk->data(), wk->data() + wk->size(), 0.01f);
    std::fill(wv->data(), wv->data() + wv->size(), 0.01f);
    std::fill(wo->data(), wo->data() + wo->size(), 0.01f);

    std::vector<std::shared_ptr<TensorBase>> inputs = {
        input, wq, wk, wv, wo, k_cache, v_cache};

    auto output = TensorFactory::create_simple({seq_len, d_model});
    std::vector<std::shared_ptr<TensorBase>> outputs = {output};

    // Single rank should not trigger assertion (size_ == 1)
    EXPECT_NO_THROW(
        {
            bool success = attention_kernel->execute(inputs, outputs);
            EXPECT_TRUE(success);
        });

    LOG_INFO("[TEST] ✓ Single rank execution does not trigger assertion");
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}
