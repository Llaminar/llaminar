/**
 * @file Test__Qwen2GraphProportionalTP.cpp
 * @brief Integration tests for Qwen2Graph with TensorParallelConfig (Phase 1c)
 *
 * Tests that Qwen2Graph correctly uses TensorParallelConfig for proportional
 * head/FFN/vocab assignment instead of equal 1/world_size splits.
 */

#include <gtest/gtest.h>
#include "../../src/v2/models/qwen/Qwen2Graph.h"
#include "../../src/v2/execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "../../src/v2/config/TensorParallelConfig.h"
#include "../../src/v2/loaders/WeightManager.h"
#include "../../src/v2/utils/MPIContext.h"
#include "../../src/v2/backends/DeviceId.h"

using namespace llaminar2;

/**
 * @brief Test fixture for Qwen2GraphProportionalTP tests
 */
class Test__Qwen2GraphProportionalTP : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Model dimensions (Qwen2-0.5B style for testing)
        n_layers_ = 2;
        d_model_ = 896;
        n_heads_ = 28;
        n_kv_heads_ = 4;
        head_dim_ = 64; // 896 / 14 = 64
        d_ff_ = 4864;
        vocab_size_ = 151936;
    }

    /**
     * @brief Create GraphConfig with given TP settings
     */
    GraphConfig createConfig(
        int local_n_heads = -1,
        int local_n_kv_heads = -1,
        int head_start = 0)
    {
        GraphConfig config;
        config.n_layers = n_layers_;
        config.d_model = d_model_;
        config.n_heads = n_heads_;
        config.n_kv_heads = n_kv_heads_;
        config.head_dim = head_dim_;
        config.d_ff = d_ff_;
        config.vocab_size = vocab_size_;
        config.default_device = DeviceId::cpu();
        config.max_seq_len = 256;

        // TP parameters
        config.head_start = head_start;
        config.local_n_heads = local_n_heads;
        config.local_n_kv_heads = local_n_kv_heads;

        return config;
    }

    // Model dimensions
    int n_layers_;
    int d_model_;
    int n_heads_;
    int n_kv_heads_;
    int head_dim_;
    int d_ff_;
    int vocab_size_;
};

// =============================================================================
// Test Cases
// =============================================================================

/**
 * @brief Test that TensorParallelConfig can be added to GraphConfig
 */
TEST_F(Test__Qwen2GraphProportionalTP, ConfigAcceptsTensorParallelConfig)
{
    // Create proportional config: 73% for rank 0, 27% for rank 1
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    // Verify config is valid
    ASSERT_TRUE(tp_config->validate());
    EXPECT_EQ(tp_config->worldSize(), 2);
    EXPECT_TRUE(tp_config->isProportional());

    // Add to GraphConfig
    GraphConfig config = createConfig();
    config.tp_config = tp_config;
    config.local_rank = 0;

    // Verify config accessors work
    EXPECT_NE(config.tp_config, nullptr);
    EXPECT_NE(config.getAssignment(), nullptr);

    const auto *assignment = config.getAssignment();
    EXPECT_GT(assignment->head_count, 0);
    EXPECT_GT(assignment->kv_head_count, 0);
}

/**
 * @brief Test 73%/27% head count distribution matches expectation
 *
 * For 28 Q heads with 73%/27% split:
 * - Rank 0 (73%): 20 heads (28 * 0.73 = 20.44 → 20)
 * - Rank 1 (27%): 8 heads (remainder)
 */
TEST_F(Test__Qwen2GraphProportionalTP, ProportionalTP_HeadCount_73_27)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    ASSERT_TRUE(tp_config->validate());

    // Check rank 0 assignment
    const auto &rank0 = tp_config->forRank(0);
    EXPECT_EQ(rank0.head_start, 0);
    EXPECT_EQ(rank0.head_count, 20); // 73% of 28 = 20.44 → 20

    // Check rank 1 assignment
    const auto &rank1 = tp_config->forRank(1);
    EXPECT_EQ(rank1.head_start, 20);
    EXPECT_EQ(rank1.head_count, 8); // Remainder = 28 - 20 = 8

    // Verify total matches
    EXPECT_EQ(rank0.head_count + rank1.head_count, n_heads_);
}

/**
 * @brief Test 73%/27% KV head count distribution (GQA)
 *
 * For 4 KV heads with 73%/27% split:
 * - Rank 0: 3 KV heads (4 * 0.73 = 2.92 → 3)
 * - Rank 1: 1 KV head (remainder)
 */
TEST_F(Test__Qwen2GraphProportionalTP, ProportionalTP_KVHeadCount_73_27)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    ASSERT_TRUE(tp_config->validate());

    // Check KV head distribution
    const auto &rank0 = tp_config->forRank(0);
    const auto &rank1 = tp_config->forRank(1);

    EXPECT_EQ(rank0.kv_head_count, 3); // 73% of 4 = 2.92 → 3
    EXPECT_EQ(rank1.kv_head_count, 1); // Remainder = 4 - 3 = 1

    // Verify total matches
    EXPECT_EQ(rank0.kv_head_count + rank1.kv_head_count, n_kv_heads_);
}

/**
 * @brief Test buffer dimensions match proportional head assignment
 *
 * Q buffer should be: [seq_len, local_n_heads * head_dim]
 */
TEST_F(Test__Qwen2GraphProportionalTP, ProportionalTP_BufferSizes)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    // Rank 0 config
    GraphConfig config = createConfig();
    config.tp_config = tp_config;
    config.local_rank = 0;

    const auto *assignment = config.getAssignment();
    ASSERT_NE(assignment, nullptr);

    // Expected Q buffer dimension = head_count * head_dim
    int expected_q_dim_rank0 = assignment->head_count * head_dim_;
    EXPECT_EQ(expected_q_dim_rank0, 20 * 64); // 1280

    // Rank 1 config
    config.local_rank = 1;
    assignment = config.getAssignment();
    int expected_q_dim_rank1 = assignment->head_count * head_dim_;
    EXPECT_EQ(expected_q_dim_rank1, 8 * 64); // 512
}

/**
 * @brief Test FFN dimension distribution
 *
 * For d_ff=4864 with 73%/27% split (aligned to 32):
 * - Rank 0: 3552 (73% of 4864 ≈ 3550.72 → rounded to 32 boundary)
 * - Rank 1: 1312 (remainder, 4864 - 3552 = 1312)
 */
TEST_F(Test__Qwen2GraphProportionalTP, ProportionalTP_FFNDimension)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    ASSERT_TRUE(tp_config->validate());

    const auto &rank0 = tp_config->forRank(0);
    const auto &rank1 = tp_config->forRank(1);

    // FFN dim should be aligned to 32
    EXPECT_EQ(rank0.d_ff_count % 32, 0);
    EXPECT_EQ(rank1.d_ff_count % 32, 0);

    // Total should match
    EXPECT_EQ(rank0.d_ff_count + rank1.d_ff_count, d_ff_);

    // Rank 0 should get more (proportional to 73%)
    EXPECT_GT(rank0.d_ff_count, rank1.d_ff_count);
}

/**
 * @brief Test backward compatibility - no TensorParallelConfig
 *
 * When tp_config is nullptr, getAssignment() should return nullptr.
 */
TEST_F(Test__Qwen2GraphProportionalTP, BackwardCompatible_NoConfig)
{
    GraphConfig config = createConfig();

    // Verify tp_config defaults to nullptr
    EXPECT_EQ(config.tp_config, nullptr);
    EXPECT_EQ(config.getAssignment(), nullptr);

    // Config should still work for single-rank inference
    EXPECT_EQ(config.local_n_heads, -1);    // -1 = use full n_heads
    EXPECT_EQ(config.local_n_kv_heads, -1); // -1 = use full n_kv_heads
}

/**
 * @brief Test equal split mode (homogeneous GPUs)
 *
 * For 2 identical GPUs, equal split should give 14/14 heads.
 */
TEST_F(Test__Qwen2GraphProportionalTP, EqualSplit_HomogeneousGPUs)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(
            2, // world_size
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    ASSERT_TRUE(tp_config->validate());
    EXPECT_FALSE(tp_config->isProportional()); // Equal split is NOT proportional

    const auto &rank0 = tp_config->forRank(0);
    const auto &rank1 = tp_config->forRank(1);

    // Equal split: 28 / 2 = 14 each
    EXPECT_EQ(rank0.head_count, 14);
    EXPECT_EQ(rank1.head_count, 14);

    // KV heads: 4 / 2 = 2 each
    EXPECT_EQ(rank0.kv_head_count, 2);
    EXPECT_EQ(rank1.kv_head_count, 2);
}

/**
 * @brief Test DeviceGraphOrchestrator accepts TensorParallelConfig
 */
TEST_F(Test__Qwen2GraphProportionalTP, OrchestratorAcceptsTPConfig)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    GraphConfig config = createConfig();
    config.tp_config = tp_config;
    config.local_rank = 0;

    // Create orchestrator with config
    DeviceGraphOrchestrator orchestrator(std::make_shared<Qwen2Graph>(config, nullptr), nullptr);

    // Set TensorParallelConfig on orchestrator
    orchestrator.setTensorParallelConfig(tp_config);

    // Verify it's stored
    EXPECT_NE(orchestrator.tensorParallelConfig(), nullptr);
    EXPECT_TRUE(orchestrator.isProportionalTPEnabled());
}

/**
 * @brief Test single-device config (no parallelism)
 */
TEST_F(Test__Qwen2GraphProportionalTP, SingleDevice_NoParallelism)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::singleDevice(
            DeviceId::cuda(0),
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    ASSERT_TRUE(tp_config->validate());
    EXPECT_EQ(tp_config->worldSize(), 1);

    const auto &assignment = tp_config->forRank(0);
    EXPECT_EQ(assignment.head_count, n_heads_);
    EXPECT_EQ(assignment.kv_head_count, n_kv_heads_);
    EXPECT_EQ(assignment.d_ff_count, d_ff_);
    EXPECT_EQ(assignment.vocab_count, vocab_size_);
}

/**
 * @brief Test work fraction sums to 1.0
 */
TEST_F(Test__Qwen2GraphProportionalTP, WorkFractionSumsToOne)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    float total_fraction = 0.0f;
    for (int r = 0; r < tp_config->worldSize(); ++r)
    {
        total_fraction += tp_config->forRank(r).work_fraction;
    }

    EXPECT_NEAR(total_fraction, 1.0f, 0.01f);
}

/**
 * @brief Test config propagation from GraphConfig to assignment
 */
TEST_F(Test__Qwen2GraphProportionalTP, ConfigPropagation_AssignmentValues)
{
    auto tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::rocm(0)},
            {0.73f, 0.27f},
            n_heads_, n_kv_heads_, d_ff_, vocab_size_));

    // Create config for rank 0
    GraphConfig config = createConfig();
    config.tp_config = tp_config;
    config.local_rank = 0;

    // Manually set TP values from assignment (as InferenceRunnerFactory does)
    const auto *assignment = config.getAssignment();
    config.head_start = assignment->head_start;
    config.local_n_heads = assignment->head_count;
    config.local_n_kv_heads = assignment->kv_head_count;
    config.d_ff_local = assignment->d_ff_count;
    config.vocab_local = assignment->vocab_count;
    config.qkv_column_parallel = true;
    config.ffn_column_parallel = true;
    config.lm_head_column_parallel = true;

    // Verify values are propagated
    EXPECT_EQ(config.head_start, 0);
    EXPECT_EQ(config.local_n_heads, 20);
    EXPECT_EQ(config.local_n_kv_heads, 3);
    EXPECT_GT(config.d_ff_local, 0);
    EXPECT_LT(config.d_ff_local, d_ff_);
    EXPECT_GT(config.vocab_local, 0);
    EXPECT_LT(config.vocab_local, vocab_size_);
}
