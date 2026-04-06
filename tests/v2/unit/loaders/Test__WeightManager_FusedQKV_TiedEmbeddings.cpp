/**
 * @file Test__WeightManager_FusedQKV_TiedEmbeddings.cpp
 * @brief Regression tests for FusedQKV weight sharding and tied embedding fallback
 *
 * Bug #1 (FusedQKV): getShardedWeight() performed naive contiguous row slicing
 * for COLUMN_PARALLEL weights without FusedQKV awareness. For fused QKV weights
 * stored as [Q_all | K_all | V_all] vertically, simple [row_start, row_end)
 * slicing crosses sub-block boundaries, producing wrong Q/K/V data under TP.
 *
 * Bug #2 (Tied Embeddings): Qwen3.5 GGUF files lack output.weight, using tied
 * token_embd.weight for the LM head. Under TP, the output.weight lookup returned
 * nullptr and both ranks got full replicated embedding, causing AllGather to
 * produce duplicated logits.
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>

#include "loaders/WeightManager.h"
#include "models/qwen35/Qwen35Schema.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "mocks/MockModelLoader.h"
#include "utils/MPIContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// FusedQKV Sub-Block Sharding Tests
// =============================================================================

/**
 * @brief Test fixture for FusedQKV weight sharding regression tests
 *
 * Sets up a mock GGUF with a fused attn_qkv.weight tensor and configures
 * WeightManager with Qwen3.5 sharding config + SHARDED strategy + mock MPI.
 *
 * Qwen3.5 GDN dimensions (0.8B):
 *   n_k_heads = 16, d_k = 128 → Q sub-block = K sub-block = V sub-block = 2048 rows
 *   fused attn_qkv.weight shape = [6144, 1024] (3 × 2048 rows)
 *
 * For simplicity we use smaller dimensions:
 *   n_heads = 4, head_dim = 8 → each sub-block = 32 rows
 *   fused attn_qkv.weight shape = [96, 16] (3 × 32 rows, 16 cols)
 */
class FusedQKVShardingTest : public ::testing::Test
{
protected:
    static constexpr size_t N_HEADS = 4;
    static constexpr size_t HEAD_DIM = 8;
    static constexpr size_t HIDDEN_DIM = 16;
    static constexpr size_t SUB_BLOCK_ROWS = N_HEADS * HEAD_DIM; // 32
    static constexpr size_t TOTAL_ROWS = 3 * SUB_BLOCK_ROWS;     // 96 (Q+K+V)
    static constexpr size_t COLS = HIDDEN_DIM;                     // 16

    void SetUp() override
    {
        // Create mock loader
        mock_loader_ = std::make_shared<MockModelLoader>();
        mock_loader_->setLoaded(true);
        mock_loader_->setArchitecture("qwen3.5");
        mock_loader_->setBlockCount(1);
        mock_loader_->setEmbeddingLength(HIDDEN_DIM);
        mock_loader_->setHeadCount(N_HEADS);
        mock_loader_->setHeadCountKV(N_HEADS); // GDN: n_kv_heads = n_heads
        mock_loader_->setVocabSize(100);
        mock_loader_->setFeedForwardLength(64);

        // Create fused QKV tensor with sequential values for verification
        // Layout: [Q_heads | K_heads | V_heads] = [32 | 32 | 32] rows × 16 cols
        // Value at (row, col) = row * 1000 + col
        auto qkv_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL_ROWS, COLS});
        float *data = qkv_tensor->mutable_data();
        for (size_t r = 0; r < TOTAL_ROWS; ++r)
        {
            for (size_t c = 0; c < COLS; ++c)
            {
                data[r * COLS + c] = static_cast<float>(r * 1000 + c);
            }
        }
        mock_loader_->addTensor("blk.0.attn_qkv.weight", qkv_tensor);

        // Add minimal required tensors
        mock_loader_->addFP32RandomTensor("token_embd.weight", {100, HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("output.weight", {100, HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("output_norm.weight", {HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("blk.0.attn_norm.weight", {HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("blk.0.ffn_norm.weight", {HIDDEN_DIM});

        // Get Qwen3.5 sharding config
        Qwen35SchemaFactory schema_factory;
        sharding_config_ = schema_factory.getWeightShardingConfig();
    }

    /**
     * @brief Create a WeightManager configured for TP sharding at given rank
     */
    std::unique_ptr<WeightManager> createShardedManager(int rank, int world_size)
    {
        auto mpi = MPIContextFactory::create_mock(rank, world_size);
        auto wm = std::make_unique<WeightManager>(
            *mock_loader_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        return wm;
    }

    /**
     * @brief Verify FP32 tensor data matches expected values from sequential pattern
     * @param tensor The loaded tensor (may be TensorSlice wrapping FP32Tensor)
     * @param expected_global_rows Which global rows should be present (in order)
     */
    void verifySequentialData(const std::shared_ptr<TensorBase> &tensor,
                              const std::vector<size_t> &expected_global_rows)
    {
        ASSERT_NE(tensor, nullptr);
        const float *data = tensor->data();
        ASSERT_NE(data, nullptr);

        size_t local_cols = COLS; // columns are not split
        for (size_t local_row = 0; local_row < expected_global_rows.size(); ++local_row)
        {
            size_t global_row = expected_global_rows[local_row];
            for (size_t c = 0; c < local_cols; ++c)
            {
                float expected = static_cast<float>(global_row * 1000 + c);
                float actual = data[local_row * local_cols + c];
                EXPECT_FLOAT_EQ(actual, expected)
                    << "Mismatch at local_row=" << local_row
                    << " (global_row=" << global_row << "), col=" << c;
            }
        }
    }

    std::shared_ptr<MockModelLoader> mock_loader_;
    WeightShardingConfig sharding_config_;
};

/**
 * @brief Regression test: FusedQKV rank 0 gets first half of each sub-block
 *
 * With TP=2 and 4 heads, rank 0 should get heads 0-1 from each sub-block:
 *   Q rows [0, 16), K rows [32, 48), V rows [64, 80)
 * NOT contiguous rows [0, 48) which was the old buggy behavior.
 */
TEST_F(FusedQKVShardingTest, Rank0Gets3SubBlockSlices)
{
    auto wm = createShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);

    // Should be 48 rows total (16 from Q + 16 from K + 16 from V)
    size_t half_sub = SUB_BLOCK_ROWS / 2; // 16
    EXPECT_EQ(tensor->shape()[0], 3 * half_sub); // 48
    EXPECT_EQ(tensor->shape()[1], COLS);          // 16

    // Expected global rows: Q[0..15], K[32..47], V[64..79]
    std::vector<size_t> expected_rows;
    for (size_t r = 0; r < half_sub; ++r)
        expected_rows.push_back(r); // Q sub-block first half
    for (size_t r = 0; r < half_sub; ++r)
        expected_rows.push_back(SUB_BLOCK_ROWS + r); // K sub-block first half
    for (size_t r = 0; r < half_sub; ++r)
        expected_rows.push_back(2 * SUB_BLOCK_ROWS + r); // V sub-block first half

    verifySequentialData(tensor, expected_rows);
}

/**
 * @brief Regression test: FusedQKV rank 1 gets second half of each sub-block
 *
 * With TP=2 and 4 heads, rank 1 should get heads 2-3 from each sub-block:
 *   Q rows [16, 32), K rows [48, 64), V rows [80, 96)
 */
TEST_F(FusedQKVShardingTest, Rank1Gets3SubBlockSlices)
{
    auto wm = createShardedManager(1, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);

    size_t half_sub = SUB_BLOCK_ROWS / 2; // 16
    EXPECT_EQ(tensor->shape()[0], 3 * half_sub); // 48
    EXPECT_EQ(tensor->shape()[1], COLS);          // 16

    // Expected global rows: Q[16..31], K[48..63], V[80..95]
    std::vector<size_t> expected_rows;
    for (size_t r = 0; r < half_sub; ++r)
        expected_rows.push_back(half_sub + r); // Q second half
    for (size_t r = 0; r < half_sub; ++r)
        expected_rows.push_back(SUB_BLOCK_ROWS + half_sub + r); // K second half
    for (size_t r = 0; r < half_sub; ++r)
        expected_rows.push_back(2 * SUB_BLOCK_ROWS + half_sub + r); // V second half

    verifySequentialData(tensor, expected_rows);
}

/**
 * @brief Verify both ranks together cover all rows exactly once
 */
TEST_F(FusedQKVShardingTest, BothRanksCoverAllRows)
{
    auto wm0 = createShardedManager(0, 2);
    auto wm1 = createShardedManager(1, 2);

    auto t0 = wm0->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    auto t1 = wm1->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(t0, nullptr);
    ASSERT_NE(t1, nullptr);

    // Total local rows should sum to the original total
    EXPECT_EQ(t0->shape()[0] + t1->shape()[0], TOTAL_ROWS);

    // Each rank's sub-block should be 3 × SUB_BLOCK_ROWS/2
    size_t half_sub = SUB_BLOCK_ROWS / 2;
    EXPECT_EQ(t0->shape()[0], 3 * half_sub);
    EXPECT_EQ(t1->shape()[0], 3 * half_sub);
}

/**
 * @brief Verify FusedQKV sub-block boundaries are respected
 *
 * The key invariant: within each rank's local tensor, the Q/K/V sub-blocks
 * should be contiguous and in order [Q_local | K_local | V_local].
 * Row 0 of local tensor should be Q data (not K or V).
 */
TEST_F(FusedQKVShardingTest, SubBlockBoundariesPreserved)
{
    auto wm = createShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const float *data = tensor->data();
    size_t half_sub = SUB_BLOCK_ROWS / 2; // 16

    // First local row should be from Q sub-block (global row 0)
    EXPECT_FLOAT_EQ(data[0], 0.0f); // row=0, col=0 → 0*1000+0 = 0

    // Row at half_sub should be from K sub-block (global row = SUB_BLOCK_ROWS)
    float k_start = data[half_sub * COLS];
    float expected_k_start = static_cast<float>(SUB_BLOCK_ROWS * 1000);
    EXPECT_FLOAT_EQ(k_start, expected_k_start)
        << "K sub-block should start at local row " << half_sub;

    // Row at 2*half_sub should be from V sub-block (global row = 2*SUB_BLOCK_ROWS)
    float v_start = data[2 * half_sub * COLS];
    float expected_v_start = static_cast<float>(2 * SUB_BLOCK_ROWS * 1000);
    EXPECT_FLOAT_EQ(v_start, expected_v_start)
        << "V sub-block should start at local row " << (2 * half_sub);
}

/**
 * @brief Regression test: Without FusedQKV fix, rank 0 would get contiguous [0, 48)
 *
 * This test specifically validates that the WRONG behavior (contiguous slicing)
 * does NOT produce the same result as the correct behavior (3-sub-block slicing).
 */
TEST_F(FusedQKVShardingTest, ContiguousSlicingWouldBeWrong)
{
    auto wm = createShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const float *data = tensor->data();
    size_t half_sub = SUB_BLOCK_ROWS / 2; // 16

    // If contiguous slicing were used, row 16 would be global row 16 (still Q data).
    // With correct 3-sub-block slicing, row 16 is K data (global row 32).
    float row_at_half_sub = data[half_sub * COLS]; // Local row 16, col 0

    // Wrong value (contiguous): 16 * 1000 = 16000
    float wrong_contiguous = static_cast<float>(half_sub * 1000);
    // Correct value (3-sub-block): SUB_BLOCK_ROWS * 1000 = 32000
    float correct_fused = static_cast<float>(SUB_BLOCK_ROWS * 1000);

    EXPECT_NE(row_at_half_sub, wrong_contiguous)
        << "Row at half_sub should NOT be contiguous Q data";
    EXPECT_FLOAT_EQ(row_at_half_sub, correct_fused)
        << "Row at half_sub should be K sub-block data";
}

// =============================================================================
// Tied Embeddings LM Head Tests
// =============================================================================

/**
 * @brief Test fixture for tied embeddings regression tests
 *
 * Sets up a mock GGUF WITHOUT output.weight but WITH token_embd.weight,
 * simulating the Qwen3.5 GGUF layout where the LM head uses tied embeddings.
 */
class TiedEmbeddingsTest : public ::testing::Test
{
protected:
    static constexpr size_t VOCAB_SIZE = 100;
    static constexpr size_t HIDDEN_DIM = 16;

    void SetUp() override
    {
        mock_loader_ = std::make_shared<MockModelLoader>();
        mock_loader_->setLoaded(true);
        mock_loader_->setArchitecture("qwen3.5");
        mock_loader_->setBlockCount(1);
        mock_loader_->setEmbeddingLength(HIDDEN_DIM);
        mock_loader_->setHeadCount(4);
        mock_loader_->setHeadCountKV(4);
        mock_loader_->setVocabSize(VOCAB_SIZE);
        mock_loader_->setFeedForwardLength(64);

        // Create token_embd.weight with sequential values
        auto embd = std::make_shared<FP32Tensor>(std::vector<size_t>{VOCAB_SIZE, HIDDEN_DIM});
        float *data = embd->mutable_data();
        for (size_t r = 0; r < VOCAB_SIZE; ++r)
        {
            for (size_t c = 0; c < HIDDEN_DIM; ++c)
            {
                data[r * HIDDEN_DIM + c] = static_cast<float>(r * 1000 + c);
            }
        }
        mock_loader_->addTensor("token_embd.weight", embd);

        // Deliberately do NOT add output.weight — this is the tied embedding scenario

        // Add minimal required tensors for model validity
        mock_loader_->addFP32RandomTensor("output_norm.weight", {HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("blk.0.attn_norm.weight", {HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("blk.0.ffn_norm.weight", {HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("blk.0.attn_qkv.weight", {3 * 4 * 8, HIDDEN_DIM});

        // Get Qwen3.5 sharding config (output.weight is COLUMN_PARALLEL)
        Qwen35SchemaFactory schema_factory;
        sharding_config_ = schema_factory.getWeightShardingConfig();
    }

    std::unique_ptr<WeightManager> createShardedManager(int rank, int world_size)
    {
        auto mpi = MPIContextFactory::create_mock(rank, world_size);
        auto wm = std::make_unique<WeightManager>(
            *mock_loader_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        return wm;
    }

    std::shared_ptr<MockModelLoader> mock_loader_;
    WeightShardingConfig sharding_config_;
};

/**
 * @brief Regression test: output.weight returns non-null when GGUF has tied embeddings
 *
 * Without the fix, getWeightForDevice("output.weight") returned nullptr because
 * the tensor was not in the GGUF file.
 */
TEST_F(TiedEmbeddingsTest, OutputWeightReturnsNonNull)
{
    auto wm = createShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("output.weight", DeviceId::cpu());

    ASSERT_NE(tensor, nullptr)
        << "output.weight should fall back to token_embd.weight";
}

/**
 * @brief Regression test: Each rank gets a different slice of the LM head
 *
 * Without the fix, both ranks got the full replicated embedding (or nullptr).
 * With the fix, rank 0 gets rows [0, 50) and rank 1 gets rows [50, 100).
 */
TEST_F(TiedEmbeddingsTest, RanksGetDifferentSlices)
{
    auto wm0 = createShardedManager(0, 2);
    auto wm1 = createShardedManager(1, 2);

    auto t0 = wm0->getWeightForDevice("output.weight", DeviceId::cpu());
    auto t1 = wm1->getWeightForDevice("output.weight", DeviceId::cpu());

    ASSERT_NE(t0, nullptr);
    ASSERT_NE(t1, nullptr);

    // Each rank should get half the vocab
    size_t half_vocab = VOCAB_SIZE / 2; // 50
    EXPECT_EQ(t0->shape()[0], half_vocab);
    EXPECT_EQ(t1->shape()[0], half_vocab);
    EXPECT_EQ(t0->shape()[1], HIDDEN_DIM);
    EXPECT_EQ(t1->shape()[1], HIDDEN_DIM);

    // Verify the data is different (rank 0 → first half, rank 1 → second half)
    const float *d0 = t0->data();
    const float *d1 = t1->data();
    ASSERT_NE(d0, nullptr);
    ASSERT_NE(d1, nullptr);

    // Rank 0 row 0 should be global row 0: value = 0*1000+0 = 0
    EXPECT_FLOAT_EQ(d0[0], 0.0f);

    // Rank 1 row 0 should be global row 50: value = 50*1000+0 = 50000
    EXPECT_FLOAT_EQ(d1[0], static_cast<float>(half_vocab * 1000));
}

/**
 * @brief Regression test: Total rows across ranks equals vocab size
 */
TEST_F(TiedEmbeddingsTest, TotalRowsCoverFullVocab)
{
    auto wm0 = createShardedManager(0, 2);
    auto wm1 = createShardedManager(1, 2);

    auto t0 = wm0->getWeightForDevice("output.weight", DeviceId::cpu());
    auto t1 = wm1->getWeightForDevice("output.weight", DeviceId::cpu());

    ASSERT_NE(t0, nullptr);
    ASSERT_NE(t1, nullptr);

    EXPECT_EQ(t0->shape()[0] + t1->shape()[0], VOCAB_SIZE);
}

/**
 * @brief Regression test: When output.weight IS present, no fallback occurs
 *
 * This ensures the tied embedding fallback doesn't activate for models
 * that actually have a separate output.weight tensor.
 */
TEST_F(TiedEmbeddingsTest, NoFallbackWhenOutputWeightExists)
{
    // Add output.weight with DIFFERENT data than token_embd.weight
    auto output_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{VOCAB_SIZE, HIDDEN_DIM});
    float *data = output_tensor->mutable_data();
    for (size_t r = 0; r < VOCAB_SIZE; ++r)
    {
        for (size_t c = 0; c < HIDDEN_DIM; ++c)
        {
            data[r * HIDDEN_DIM + c] = static_cast<float>(r * 2000 + c + 1); // Different pattern
        }
    }
    mock_loader_->addTensor("output.weight", output_tensor);

    auto wm = createShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("output.weight", DeviceId::cpu());

    ASSERT_NE(tensor, nullptr);

    // Should get output.weight data (2000 pattern), NOT token_embd.weight data (1000 pattern)
    const float *d = tensor->data();
    // Row 0, col 0: output.weight value = 0*2000+0+1 = 1
    EXPECT_FLOAT_EQ(d[0], 1.0f)
        << "Should use output.weight, not token_embd.weight fallback";
    // NOT 0.0f (which would be token_embd.weight value at row 0, col 0)
}

// =============================================================================
// Qwen35 Schema Dimension Type Tests
// =============================================================================

/**
 * @brief Regression test: Qwen35 schema has correct dimension types
 *
 * Validates the schema fixes:
 * - attn_qkv.weight → FusedQKVHeads (was Heads)
 * - ssm_conv1d.weight → FusedQKVHeads (was Heads)
 * - ssm_norm.weight → Replicate (was ColumnParallel)
 */
TEST(Qwen35SchemaRegressionTest, FusedQKVDimensionType)
{
    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    EXPECT_EQ(config.getDimensionType("blk.0.attn_qkv.weight"),
              WeightDimensionType::FusedQKVHeads)
        << "attn_qkv.weight must use FusedQKVHeads for 3-sub-block sharding";
}

TEST(Qwen35SchemaRegressionTest, SSMConv1dDimensionType)
{
    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    EXPECT_EQ(config.getDimensionType("blk.0.ssm_conv1d.weight"),
              WeightDimensionType::FusedQKVHeads)
        << "ssm_conv1d.weight must use FusedQKVHeads (channels match fused QKV)";
}

TEST(Qwen35SchemaRegressionTest, SSMNormIsReplicated)
{
    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    EXPECT_EQ(config.getMode("blk.0.ssm_norm.weight"),
              WeightShardingMode::Replicate)
        << "ssm_norm.weight must be Replicate (per-state-dimension, not per-head)";
}

// =============================================================================
// GQA-Aware FusedQKV Tests (n_kv_heads < n_heads)
// =============================================================================

/**
 * @brief Test fixture for GQA-aware FusedQKV sharding
 *
 * Simulates Qwen3.5 FA layers with GQA where Q has more heads than K/V:
 *   n_heads=4, n_kv_heads=2, head_dim=8
 *   Q = 4*8 = 32 rows, K = 2*8 = 16 rows, V = 2*8 = 16 rows
 *   Total = 64 rows (NOT divisible by 3 into equal blocks)
 */
class GQAFusedQKVShardingTest : public ::testing::Test
{
protected:
    static constexpr int N_HEADS = 4;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 8;
    static constexpr size_t HIDDEN_DIM = 16;
    static constexpr size_t Q_ROWS = N_HEADS * HEAD_DIM;      // 32
    static constexpr size_t KV_ROWS = N_KV_HEADS * HEAD_DIM;  // 16
    static constexpr size_t TOTAL_ROWS = Q_ROWS + 2 * KV_ROWS; // 64
    static constexpr size_t COLS = HIDDEN_DIM;

    void SetUp() override
    {
        mock_loader_ = std::make_shared<MockModelLoader>();
        mock_loader_->setLoaded(true);
        mock_loader_->setArchitecture("qwen3.5");
        mock_loader_->setBlockCount(1);
        mock_loader_->setEmbeddingLength(HIDDEN_DIM);
        mock_loader_->setHeadCount(N_HEADS);
        mock_loader_->setHeadCountKV(N_KV_HEADS);
        mock_loader_->setVocabSize(100);
        mock_loader_->setFeedForwardLength(64);

        // Create fused QKV tensor: [Q(32) | K(16) | V(16)] = 64 rows
        // Value at (row, col) = row * 1000 + col
        auto qkv_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL_ROWS, COLS});
        float *data = qkv_tensor->mutable_data();
        for (size_t r = 0; r < TOTAL_ROWS; ++r)
            for (size_t c = 0; c < COLS; ++c)
                data[r * COLS + c] = static_cast<float>(r * 1000 + c);
        mock_loader_->addTensor("blk.0.attn_qkv.weight", qkv_tensor);

        mock_loader_->addFP32RandomTensor("token_embd.weight", {100, HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("output.weight", {100, HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("output_norm.weight", {HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("blk.0.attn_norm.weight", {HIDDEN_DIM});
        mock_loader_->addFP32RandomTensor("blk.0.ffn_norm.weight", {HIDDEN_DIM});

        Qwen35SchemaFactory schema_factory;
        sharding_config_ = schema_factory.getWeightShardingConfig();
    }

    std::unique_ptr<WeightManager> createShardedManager(int rank, int world_size)
    {
        auto mpi = MPIContextFactory::create_mock(rank, world_size);
        auto wm = std::make_unique<WeightManager>(
            *mock_loader_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        wm->setModelDimensions(N_HEADS, N_KV_HEADS, HEAD_DIM);
        return wm;
    }

    std::shared_ptr<MockModelLoader> mock_loader_;
    WeightShardingConfig sharding_config_;
};

/**
 * @brief GQA rank 0: gets first half of each unequal sub-block
 *
 * With TP=2, n_heads=4, n_kv_heads=2, head_dim=8:
 *   Q sub-block: 32 rows → rank 0 gets [0, 16)
 *   K sub-block: 16 rows → rank 0 gets [32, 40)
 *   V sub-block: 16 rows → rank 0 gets [48, 56)
 *   Total: 16 + 8 + 8 = 32 rows
 */
TEST_F(GQAFusedQKVShardingTest, Rank0GetsUnequalSubBlocks)
{
    auto wm = createShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const size_t q_half = Q_ROWS / 2;   // 16
    const size_t kv_half = KV_ROWS / 2; // 8
    EXPECT_EQ(tensor->shape()[0], q_half + 2 * kv_half); // 32
    EXPECT_EQ(tensor->shape()[1], COLS);

    // Verify data: Q rows [0..15], K rows [32..39], V rows [48..55]
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    // Q sub-block first half
    for (size_t r = 0; r < q_half; ++r)
        EXPECT_FLOAT_EQ(data[r * COLS], static_cast<float>(r * 1000))
            << "Q row " << r;

    // K sub-block first half (starts at local row q_half)
    for (size_t r = 0; r < kv_half; ++r)
        EXPECT_FLOAT_EQ(data[(q_half + r) * COLS],
                         static_cast<float>((Q_ROWS + r) * 1000))
            << "K row " << r;

    // V sub-block first half (starts at local row q_half + kv_half)
    for (size_t r = 0; r < kv_half; ++r)
        EXPECT_FLOAT_EQ(data[(q_half + kv_half + r) * COLS],
                         static_cast<float>((Q_ROWS + KV_ROWS + r) * 1000))
            << "V row " << r;
}

/**
 * @brief GQA rank 1: gets second half of each unequal sub-block
 */
TEST_F(GQAFusedQKVShardingTest, Rank1GetsUnequalSubBlocks)
{
    auto wm = createShardedManager(1, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const size_t q_half = Q_ROWS / 2;   // 16
    const size_t kv_half = KV_ROWS / 2; // 8
    EXPECT_EQ(tensor->shape()[0], q_half + 2 * kv_half); // 32
    EXPECT_EQ(tensor->shape()[1], COLS);

    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    // Q rows [16..31]
    for (size_t r = 0; r < q_half; ++r)
        EXPECT_FLOAT_EQ(data[r * COLS], static_cast<float>((q_half + r) * 1000))
            << "Q row " << r;

    // K rows [40..47]
    for (size_t r = 0; r < kv_half; ++r)
        EXPECT_FLOAT_EQ(data[(q_half + r) * COLS],
                         static_cast<float>((Q_ROWS + kv_half + r) * 1000))
            << "K row " << r;

    // V rows [56..63]
    for (size_t r = 0; r < kv_half; ++r)
        EXPECT_FLOAT_EQ(data[(q_half + kv_half + r) * COLS],
                         static_cast<float>((Q_ROWS + KV_ROWS + kv_half + r) * 1000))
            << "V row " << r;
}

/**
 * @brief GQA: both ranks together cover all rows
 */
TEST_F(GQAFusedQKVShardingTest, BothRanksCoverAllRows)
{
    auto wm0 = createShardedManager(0, 2);
    auto wm1 = createShardedManager(1, 2);
    auto t0 = wm0->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    auto t1 = wm1->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(t0, nullptr);
    ASSERT_NE(t1, nullptr);
    EXPECT_EQ(t0->shape()[0] + t1->shape()[0], TOTAL_ROWS);
}

// =============================================================================
// GDN Non-QKV Fallback Tests
// =============================================================================

/**
 * @brief When model dimensions are set and total_rows doesn't match Q+K+V,
 * FusedQKVHeads falls back to simple equal row splitting (GDN SSM weights).
 *
 * Qwen3.5 GDN layers have attn_qkv.weight with rows = 2 * inner_size,
 * which is NOT n_heads*head_dim + 2*n_kv_heads*head_dim.
 */
TEST_F(GQAFusedQKVShardingTest, GDNWeightFallsBackToSimpleRowSplit)
{
    // Create a GDN-like weight: 128 rows (doesn't match 32+16+16=64)
    const size_t gdn_rows = 128;
    auto gdn_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{gdn_rows, COLS});
    float *data = gdn_tensor->mutable_data();
    for (size_t r = 0; r < gdn_rows; ++r)
        for (size_t c = 0; c < COLS; ++c)
            data[r * COLS + c] = static_cast<float>(r * 1000 + c);
    mock_loader_->addTensor("blk.0.gdn_qkv.weight", gdn_tensor);

    // Register as FusedQKVHeads in sharding config
    sharding_config_.patterns.push_back(
        {"gdn_qkv.weight", WeightShardingMode::ColumnParallel,
         WeightDimensionType::FusedQKVHeads, "GDN QKV projection"});

    auto wm = createShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.gdn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    // Should get simple equal row split: 128/2 = 64 rows
    EXPECT_EQ(tensor->shape()[0], gdn_rows / 2);
    EXPECT_EQ(tensor->shape()[1], COLS);

    // Row 0 should be global row 0 (contiguous slice [0, 64))
    const float *d = tensor->data();
    EXPECT_FLOAT_EQ(d[0], 0.0f);
    // Last row should be global row 63
    EXPECT_FLOAT_EQ(d[(gdn_rows / 2 - 1) * COLS],
                     static_cast<float>((gdn_rows / 2 - 1) * 1000));
}
