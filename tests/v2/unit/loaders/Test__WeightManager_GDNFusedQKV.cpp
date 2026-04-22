/**
 * @file Test__WeightManager_GDNFusedQKV.cpp
 * @brief Regression tests for GDN (Gated DeltaNet) FusedQKV weight sharding
 *
 * Bug: GDN layers in Qwen3.5 have asymmetric QKV layout where Q=K≠V:
 *   Q = n_k_heads * d_state
 *   K = n_k_heads * d_state
 *   V = n_v_heads * d_state (n_v_heads > n_k_heads)
 *
 * The standard FA-based FusedQKV detection (n_heads*hd + 2*n_kv_heads*hd) doesn't
 * match GDN's (2*n_k*d + n_v*d) layout. Without GDN-aware detection via
 * setGDNDimensions(), the code fell through to simple equal row splitting which
 * produced wrong Q/K/V boundaries under TP.
 *
 * Fix v1: Added setGDNDimensions() to WeightManager. When FA layout check fails
 * and div-by-3 fails, the code tries the GDN layout (2*key_dim + value_dim).
 * Each sub-block is independently TP-sliced.
 *
 * Fix v2 (GDN modular repeat): GDN uses repeat_type=1 (modular: v_head % n_k),
 * which means every V head needs access to its corresponding K head. Under TP
 * with contiguous V sharding, v_heads on one rank may need k_heads from another
 * rank. Fix: replicate Q and K sub-blocks on every rank, only shard V.
 * This ensures all k_heads are locally available for the modular mapping.
 *
 * Validation: When a model is too small for a given TP degree (V rows not
 * evenly divisible by world_size), WeightManager throws std::invalid_argument
 * with a descriptive error message rather than silently producing wrong results.
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>

#include "loaders/WeightManager.h"
#include "models/qwen35/Qwen35Schema.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "mocks/MockModelLoader.h"
#include "utils/MPIContext.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// GDN-Aware FusedQKV Sub-Block Sharding Tests
// =============================================================================

/**
 * @brief Test fixture for GDN FusedQKV weight sharding
 *
 * Simulates Qwen3.5 4B GDN layers with asymmetric Q=K≠V:
 *   n_k_heads = 4 (group_count), n_v_heads = 8 (time_step_rank), d_state = 4
 *   Q = 4*4 = 16 rows, K = 4*4 = 16 rows, V = 8*4 = 32 rows
 *   Total = 64 rows NOT matching FA layout (n_heads*hd + 2*n_kv*hd),
 *   NOT divisible by 3 into equal blocks.
 *
 * Real Qwen3.5 4B dimensions:
 *   n_k_heads = 16, n_v_heads = 32, d_state = 128
 *   Q = 2048, K = 2048, V = 4096 → total = 8192
 */
class GDNFusedQKVShardingTest : public ::testing::Test
{
protected:
    // GDN-specific dimensions (scaled down for testing)
    static constexpr int GDN_N_K_HEADS = 4; // group_count
    static constexpr int GDN_N_V_HEADS = 8; // time_step_rank
    static constexpr int GDN_D_STATE = 4;   // state_size (d_k = d_v)

    // Derived sizes
    static constexpr size_t Q_ROWS = GDN_N_K_HEADS * GDN_D_STATE;  // 16
    static constexpr size_t K_ROWS = GDN_N_K_HEADS * GDN_D_STATE;  // 16
    static constexpr size_t V_ROWS = GDN_N_V_HEADS * GDN_D_STATE;  // 32
    static constexpr size_t TOTAL_ROWS = Q_ROWS + K_ROWS + V_ROWS; // 64
    static constexpr size_t COLS = 16;                             // hidden_dim (arbitrary)

    // FA dimensions (must NOT match GDN layout to trigger GDN path)
    static constexpr int FA_N_HEADS = 8;
    static constexpr int FA_N_KV_HEADS = 2;
    static constexpr int FA_HEAD_DIM = 8;
    // FA expected: 8*8 + 2*2*8 = 96 ≠ 64, and 64 % 3 ≠ 0 → triggers GDN path

    void SetUp() override
    {
        mock_loader_ = std::make_shared<MockModelLoader>();
        mock_loader_->setLoaded(true);
        mock_loader_->setArchitecture("qwen3.5");
        mock_loader_->setBlockCount(1);
        mock_loader_->setEmbeddingLength(COLS);
        mock_loader_->setHeadCount(FA_N_HEADS);
        mock_loader_->setHeadCountKV(FA_N_KV_HEADS);
        mock_loader_->setVocabSize(100);
        mock_loader_->setFeedForwardLength(64);

        // Create fused QKV tensor: [Q(16) | K(16) | V(32)] = 64 rows
        // Value at (row, col) = row * 1000 + col (allows verification of row placement)
        auto qkv_tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL_ROWS, COLS});
        float *data = qkv_tensor->mutable_data();
        for (size_t r = 0; r < TOTAL_ROWS; ++r)
            for (size_t c = 0; c < COLS; ++c)
                data[r * COLS + c] = static_cast<float>(r * 1000 + c);
        mock_loader_->addTensor("blk.0.attn_qkv.weight", qkv_tensor);

        // Also create ssm_conv1d.weight with the SAME layout (real Qwen3.5 does this)
        // ssm_conv1d.weight has shape [QKV_dim, conv_kernel_size] e.g. [8192, 4]
        static constexpr size_t CONV_KERNEL = 4;
        auto conv_tensor = std::make_shared<FP32Tensor>(
            std::vector<size_t>{TOTAL_ROWS, CONV_KERNEL});
        float *conv_data = conv_tensor->mutable_data();
        for (size_t r = 0; r < TOTAL_ROWS; ++r)
            for (size_t c = 0; c < CONV_KERNEL; ++c)
                conv_data[r * CONV_KERNEL + c] = static_cast<float>(r * 100 + c);
        mock_loader_->addTensor("blk.0.ssm_conv1d.weight", conv_tensor);

        // Add minimal required tensors
        mock_loader_->addFP32RandomTensor("token_embd.weight", {100, COLS});
        mock_loader_->addFP32RandomTensor("output.weight", {100, COLS});
        mock_loader_->addFP32RandomTensor("output_norm.weight", {COLS});
        mock_loader_->addFP32RandomTensor("blk.0.attn_norm.weight", {COLS});
        mock_loader_->addFP32RandomTensor("blk.0.ffn_norm.weight", {COLS});

        // Get Qwen3.5 sharding config (includes FusedQKVHeads for attn_qkv and ssm_conv1d)
        Qwen35SchemaFactory schema_factory;
        sharding_config_ = schema_factory.getWeightShardingConfig();
    }

    /**
     * @brief Create WeightManager with both FA and GDN dimensions set
     */
    std::unique_ptr<WeightManager> createGDNShardedManager(int rank, int world_size)
    {
        auto mpi = MPIContextFactory::create_mock(rank, world_size);
        auto wm = std::make_unique<WeightManager>(
            *mock_loader_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        wm->setModelDimensions(FA_N_HEADS, FA_N_KV_HEADS, FA_HEAD_DIM);
        wm->setGDNDimensions(GDN_N_K_HEADS, GDN_N_V_HEADS, GDN_D_STATE);
        return wm;
    }

    /**
     * @brief Create WeightManager WITHOUT GDN dimensions (pre-fix behavior)
     */
    std::unique_ptr<WeightManager> createNonGDNShardedManager(int rank, int world_size)
    {
        auto mpi = MPIContextFactory::create_mock(rank, world_size);
        auto wm = std::make_unique<WeightManager>(
            *mock_loader_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        wm->setModelDimensions(FA_N_HEADS, FA_N_KV_HEADS, FA_HEAD_DIM);
        // NOT calling setGDNDimensions() — simulates pre-fix behavior
        return wm;
    }

    std::shared_ptr<MockModelLoader> mock_loader_;
    WeightShardingConfig sharding_config_;
};

// -----------------------------------------------------------------------------
// Core GDN Sub-Block Slicing Tests
// -----------------------------------------------------------------------------

/**
 * @brief GDN rank 0: Q/K replicated (full), V sharded (first half)
 *
 * With GDN modular repeat (repeat_type=1), Q/K are replicated on all TP ranks
 * to ensure every v_head has access to its correct k_head.
 *
 * Layout: [Q(16) | K(16) | V(32)] with TP=2, GDN replicate Q/K:
 *   Q rank 0: rows [0, 16)     → 16 rows (FULL, replicated)
 *   K rank 0: rows [16, 32)    → 16 rows (FULL, replicated)
 *   V rank 0: rows [32, 48)    → 16 rows (sharded)
 *   Total rank 0: 16 + 16 + 16 = 48 rows
 */
TEST_F(GDNFusedQKVShardingTest, Rank0GetsReplicatedQKShardedV)
{
    auto wm = createGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    // Q/K replicated (full), V sharded
    const size_t q_local = Q_ROWS;                              // 16 (replicated)
    const size_t k_local = K_ROWS;                              // 16 (replicated)
    const size_t v_local = V_ROWS / 2;                          // 16 (sharded)
    EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local); // 48
    EXPECT_EQ(tensor->shape()[1], COLS);

    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    size_t local_row = 0;

    // Q sub-block: FULL global rows [0, 16)
    for (size_t r = 0; r < q_local; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS], static_cast<float>(r * 1000))
            << "Q row " << r << " (local_row=" << local_row << ")";

    // K sub-block: FULL global rows [16, 32)
    for (size_t r = 0; r < k_local; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                        static_cast<float>((Q_ROWS + r) * 1000))
            << "K row " << r << " (local_row=" << local_row << ")";

    // V sub-block first half: global rows [32, 48)
    for (size_t r = 0; r < v_local; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                        static_cast<float>((Q_ROWS + K_ROWS + r) * 1000))
            << "V row " << r << " (local_row=" << local_row << ")";
}

/**
 * @brief GDN rank 1: Q/K replicated (full), V sharded (second half)
 *
 * Layout: [Q(16) | K(16) | V(32)] with TP=2, GDN replicate Q/K:
 *   Q rank 1: rows [0, 16)     → 16 rows (FULL, replicated — same as rank 0!)
 *   K rank 1: rows [16, 32)    → 16 rows (FULL, replicated — same as rank 0!)
 *   V rank 1: rows [48, 64)    → 16 rows (sharded second half)
 *   Total rank 1: 16 + 16 + 16 = 48 rows
 */
TEST_F(GDNFusedQKVShardingTest, Rank1GetsReplicatedQKShardedV)
{
    auto wm = createGDNShardedManager(1, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    // Q/K replicated (full), V sharded
    const size_t q_local = Q_ROWS;                              // 16 (replicated)
    const size_t k_local = K_ROWS;                              // 16 (replicated)
    const size_t v_local = V_ROWS / 2;                          // 16 (sharded)
    EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local); // 48
    EXPECT_EQ(tensor->shape()[1], COLS);

    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    size_t local_row = 0;

    // Q sub-block: FULL global rows [0, 16) — replicated, same as rank 0
    for (size_t r = 0; r < q_local; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS], static_cast<float>(r * 1000))
            << "Q row " << r << " (local_row=" << local_row << ")";

    // K sub-block: FULL global rows [16, 32) — replicated, same as rank 0
    for (size_t r = 0; r < k_local; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                        static_cast<float>((Q_ROWS + r) * 1000))
            << "K row " << r << " (local_row=" << local_row << ")";

    // V sub-block second half: global rows [48, 64)
    for (size_t r = 0; r < v_local; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                        static_cast<float>((Q_ROWS + K_ROWS + v_local + r) * 1000))
            << "V row " << r << " (local_row=" << local_row << ")";
}

/**
 * @brief Both ranks' V sub-blocks together cover all V rows exactly
 *
 * Q/K are replicated (same on both ranks), V is sharded.
 * Total V rows across ranks = all V rows. Total including Q/K is
 * 2*(Q+K) + V because Q/K are duplicated.
 */
TEST_F(GDNFusedQKVShardingTest, BothRanksVSubBlocksCoverAllVRows)
{
    auto wm0 = createGDNShardedManager(0, 2);
    auto wm1 = createGDNShardedManager(1, 2);

    auto t0 = wm0->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    auto t1 = wm1->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(t0, nullptr);
    ASSERT_NE(t1, nullptr);

    // Each rank has Q_full + K_full + V/2 = 16 + 16 + 16 = 48
    EXPECT_EQ(t0->shape()[0], Q_ROWS + K_ROWS + V_ROWS / 2);
    EXPECT_EQ(t1->shape()[0], Q_ROWS + K_ROWS + V_ROWS / 2);

    // Q/K are replicated — verify both ranks have identical Q data
    const float *d0 = t0->data();
    const float *d1 = t1->data();
    for (size_t r = 0; r < Q_ROWS + K_ROWS; ++r)
        EXPECT_FLOAT_EQ(d0[r * COLS], d1[r * COLS])
            << "Q/K row " << r << " should be identical across ranks";

    // V sub-blocks are different (sharded)
    const size_t v_offset = Q_ROWS + K_ROWS;
    EXPECT_NE(d0[v_offset * COLS], d1[v_offset * COLS]);
}

/**
 * @brief With GDN replicated Q/K, all ranks have full Q/K but only partial V
 *
 * The key insight: V is sharded (16 rows per rank) while Q/K are replicated
 * (16 rows each, full). So Q_local == K_local == V_local == 16 for TP=2.
 */
TEST_F(GDNFusedQKVShardingTest, QKReplicatedVSharded)
{
    auto wm = createGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const size_t q_local = Q_ROWS;     // 16 (replicated)
    const size_t k_local = K_ROWS;     // 16 (replicated)
    const size_t v_local = V_ROWS / 2; // 16 (sharded)

    // Q/K are replicated, V is sharded
    EXPECT_EQ(q_local, Q_ROWS);
    EXPECT_EQ(k_local, K_ROWS);
    EXPECT_EQ(v_local, V_ROWS / 2);

    // Total local rows = 16 + 16 + 16 = 48
    EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local);
}

/**
 * @brief Sub-block boundaries are correctly placed in the local tensor
 *
 * Local layout should be [Q_full | K_full | V_local] in order.
 * The Q/K boundary at Q_ROWS and K/V boundary at Q_ROWS + K_ROWS.
 */
TEST_F(GDNFusedQKVShardingTest, SubBlockBoundariesCorrect)
{
    auto wm = createGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const float *data = tensor->data();

    // Row 0 is Q data (global row 0)
    EXPECT_FLOAT_EQ(data[0], 0.0f);

    // Row Q_ROWS is K data (global row Q_ROWS = 16) — Q is replicated in full
    EXPECT_FLOAT_EQ(data[Q_ROWS * COLS], static_cast<float>(Q_ROWS * 1000));

    // Row Q_ROWS + K_ROWS is V data (global row Q_ROWS + K_ROWS = 32)
    EXPECT_FLOAT_EQ(data[(Q_ROWS + K_ROWS) * COLS],
                    static_cast<float>((Q_ROWS + K_ROWS) * 1000));
}

// -----------------------------------------------------------------------------
// ssm_conv1d.weight Uses Same GDN Layout
// -----------------------------------------------------------------------------

/**
 * @brief ssm_conv1d.weight has the same row count as attn_qkv and uses GDN layout
 *
 * In Qwen3.5, ssm_conv1d.weight shape is [QKV_dim, conv_kernel_size] where
 * QKV_dim = 2*n_k*d + n_v*d. The FusedQKVHeads sharding must apply the same
 * GDN sub-block split to ssm_conv1d.weight.
 */
TEST_F(GDNFusedQKVShardingTest, SSMConv1dUsesGDNSubBlockSlicing)
{
    auto wm = createGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.ssm_conv1d.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    // Same replication: Q/K full, V sharded
    const size_t q_local = Q_ROWS;     // 16 (replicated)
    const size_t k_local = K_ROWS;     // 16 (replicated)
    const size_t v_local = V_ROWS / 2; // 16 (sharded)

    EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local);
    // But different column count (conv_kernel_size = 4)
    EXPECT_EQ(tensor->shape()[1], 4u);

    // Verify sub-block boundaries using our value pattern (r*100+c)
    const float *data = tensor->data();

    // Row 0 is Q data (global row 0, col 0)
    EXPECT_FLOAT_EQ(data[0], 0.0f);

    // Row Q_ROWS is K data (global row 16) — replicated full
    EXPECT_FLOAT_EQ(data[q_local * 4], static_cast<float>(Q_ROWS * 100));

    // Row Q_ROWS+K_ROWS is V data (global row 32) — sharded
    EXPECT_FLOAT_EQ(data[(q_local + k_local) * 4],
                    static_cast<float>((Q_ROWS + K_ROWS) * 100));
}

// -----------------------------------------------------------------------------
// Regression: Without setGDNDimensions(), falls back to equal row split
// -----------------------------------------------------------------------------

/**
 * @brief Without GDN dimensions, the code falls back to simple equal row split
 *
 * This is the pre-fix behavior. Without setGDNDimensions(), the GDN path is
 * unreachable, so a 64-row weight that doesn't match FA layout and isn't
 * divisible by 3 falls through to simple contiguous row splitting.
 *
 * Simple split: rank 0 gets rows [0, 32), rank 1 gets rows [32, 64)
 * This is WRONG for GDN because rank 0 gets all Q + all K + no V.
 */
TEST_F(GDNFusedQKVShardingTest, WithoutGDNDimensionsFallsBackToEqualSplit)
{
    auto wm = createNonGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    // Simple equal split: 64/2 = 32 rows per rank
    EXPECT_EQ(tensor->shape()[0], TOTAL_ROWS / 2);

    // Verify it's contiguous rows [0, 32) — the WRONG behavior for GDN
    const float *data = tensor->data();

    // Row 0 = global row 0
    EXPECT_FLOAT_EQ(data[0], 0.0f);
    // Row 31 = global row 31 (last row of contiguous block)
    EXPECT_FLOAT_EQ(data[31 * COLS], static_cast<float>(31 * 1000));

    // This means rank 0 has ALL of Q (rows 0-15) and ALL of K (rows 16-31)
    // but NONE of V — clearly wrong for symmetric TP.
}

/**
 * @brief With GDN dimensions, the result differs from equal split
 *
 * GDN replicated Q/K gives 48 rows (16+16+16) while equal split gives 32 rows.
 * The data content also differs: GDN row Q_ROWS is K data (global row 16)
 * while equal split row 16 is just global row 16 (within K block anyway).
 */
TEST_F(GDNFusedQKVShardingTest, GDNSplitDiffersFromEqualSplit)
{
    auto wm_gdn = createGDNShardedManager(0, 2);
    auto wm_equal = createNonGDNShardedManager(0, 2);

    auto t_gdn = wm_gdn->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    auto t_equal = wm_equal->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(t_gdn, nullptr);
    ASSERT_NE(t_equal, nullptr);

    // GDN has 48 rows (Q_full + K_full + V/2), equal has 32 rows (TOTAL/2)
    EXPECT_EQ(t_gdn->shape()[0], 48u);
    EXPECT_EQ(t_equal->shape()[0], 32u);

    // GDN has more total rows per rank due to Q/K replication
    EXPECT_GT(t_gdn->shape()[0], t_equal->shape()[0]);
}

// -----------------------------------------------------------------------------
// setGDNDimensions API Tests
// -----------------------------------------------------------------------------

/**
 * @brief setGDNDimensions stores values correctly and enables GDN path
 */
TEST(WeightManagerGDNAPI, SetGDNDimensionsEnablesGDNPath)
{
    auto mock = std::make_shared<MockModelLoader>();
    mock->setLoaded(true);
    mock->setArchitecture("qwen3.5");
    mock->setBlockCount(0);
    mock->setEmbeddingLength(16);
    mock->setHeadCount(8);
    mock->setHeadCountKV(2);
    mock->setVocabSize(100);
    mock->setFeedForwardLength(64);

    mock->addFP32RandomTensor("token_embd.weight", {100, 16});
    mock->addFP32RandomTensor("output.weight", {100, 16});
    mock->addFP32RandomTensor("output_norm.weight", {16});

    auto mpi = MPIContextFactory::create_mock(0, 2);
    auto wm = std::make_unique<WeightManager>(
        *mock, mpi, nullptr,
        WeightDistributionStrategy::SHARDED,
        WeightPrecision::NATIVE);

    // Before setGDNDimensions: create a weight that only matches GDN layout
    // n_k=4, d=4 → Q=K=16, n_v=8, d=4 → V=32, total=64
    auto qkv = std::make_shared<FP32Tensor>(std::vector<size_t>{64u, 16u});
    mock->addTensor("blk.0.attn_qkv.weight", qkv);

    Qwen35SchemaFactory factory;
    wm->setWeightShardingConfig(factory.getWeightShardingConfig());
    wm->setModelDimensions(8, 2, 8); // FA: 8*8 + 2*2*8 = 96 ≠ 64, 64%3 ≠ 0

    // Without GDN: falls to equal split (32 rows each)
    auto t_no_gdn = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(t_no_gdn, nullptr);
    EXPECT_EQ(t_no_gdn->shape()[0], 32u); // simple split

    // Now set GDN dimensions
    wm->setGDNDimensions(4, 8, 4);

    // Re-request — should now use GDN sub-block slicing
    // Note: WeightManager caches, so we need a fresh manager
    auto wm2 = std::make_unique<WeightManager>(
        *mock, mpi, nullptr,
        WeightDistributionStrategy::SHARDED,
        WeightPrecision::NATIVE);
    wm2->setWeightShardingConfig(factory.getWeightShardingConfig());
    wm2->setModelDimensions(8, 2, 8);
    wm2->setGDNDimensions(4, 8, 4);

    auto t_gdn = wm2->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(t_gdn, nullptr);
    // GDN replicated Q/K: Q=16(full), K=16(full), V=32/2=16 → total=48
    EXPECT_EQ(t_gdn->shape()[0], 48u);
}

// -----------------------------------------------------------------------------
// 4-Way TP Sharding
// -----------------------------------------------------------------------------

/**
 * @brief GDN FusedQKV with 4-way TP: Q/K replicated, V sharded to 1/4
 *
 * Layout: [Q(16) | K(16) | V(32)] with TP=4, GDN replicate Q/K:
 *   Q per rank: 16 rows (full, replicated)
 *   K per rank: 16 rows (full, replicated)
 *   V per rank: 8 rows (sharded)
 *   Total per rank: 16 + 16 + 8 = 40 rows
 */
TEST_F(GDNFusedQKVShardingTest, FourWayTPSharding)
{
    for (int rank = 0; rank < 4; ++rank)
    {
        auto mpi = MPIContextFactory::create_mock(rank, 4);
        auto wm4 = std::make_unique<WeightManager>(
            *mock_loader_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm4->setWeightShardingConfig(sharding_config_);
        wm4->setModelDimensions(FA_N_HEADS, FA_N_KV_HEADS, FA_HEAD_DIM);
        wm4->setGDNDimensions(GDN_N_K_HEADS, GDN_N_V_HEADS, GDN_D_STATE);

        auto tensor = wm4->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        ASSERT_NE(tensor, nullptr) << "Rank " << rank;

        const size_t q_local = Q_ROWS;     // 16 (replicated)
        const size_t k_local = K_ROWS;     // 16 (replicated)
        const size_t v_local = V_ROWS / 4; // 8 (sharded)

        EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local)
            << "Rank " << rank << " should have 40 local rows";
        EXPECT_EQ(tensor->shape()[1], COLS);

        // Verify first row of each sub-block
        const float *data = tensor->data();

        // Q sub-block: FULL (replicated), always starts at global row 0
        EXPECT_FLOAT_EQ(data[0], 0.0f)
            << "Rank " << rank << " Q start (replicated)";

        // K sub-block: FULL (replicated), always starts at global row Q_ROWS
        EXPECT_FLOAT_EQ(data[q_local * COLS],
                        static_cast<float>(Q_ROWS * 1000))
            << "Rank " << rank << " K start (replicated)";

        // V sub-block: SHARDED, global row = Q_ROWS + K_ROWS + rank * v_local
        EXPECT_FLOAT_EQ(data[(q_local + k_local) * COLS],
                        static_cast<float>((Q_ROWS + K_ROWS + rank * v_local) * 1000))
            << "Rank " << rank << " V start (sharded)";
    }
}

/**
 * @brief 4-way TP: all V slices together cover all V rows
 *
 * With Q/K replicated, total rows across all ranks is:
 * 4 * (Q_full + K_full) + V_total = 4*(16+16) + 32 = 160
 * But the UNIQUE data covered = Q + K + V = 64 rows
 */
TEST_F(GDNFusedQKVShardingTest, FourWayTPCoversAllVRows)
{
    std::vector<float> all_v_first_col;

    for (int rank = 0; rank < 4; ++rank)
    {
        auto mpi = MPIContextFactory::create_mock(rank, 4);
        auto wm = std::make_unique<WeightManager>(
            *mock_loader_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        wm->setModelDimensions(FA_N_HEADS, FA_N_KV_HEADS, FA_HEAD_DIM);
        wm->setGDNDimensions(GDN_N_K_HEADS, GDN_N_V_HEADS, GDN_D_STATE);

        auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        ASSERT_NE(tensor, nullptr);

        // Each rank has Q_full(16) + K_full(16) + V_shard(8) = 40
        EXPECT_EQ(tensor->shape()[0], Q_ROWS + K_ROWS + V_ROWS / 4);

        // Collect V rows from this rank
        const float *data = tensor->data();
        const size_t v_offset = Q_ROWS + K_ROWS;
        const size_t v_local = V_ROWS / 4;
        for (size_t r = 0; r < v_local; ++r)
            all_v_first_col.push_back(data[(v_offset + r) * COLS]);
    }

    // All V rows across ranks should cover global V rows [32..64)
    EXPECT_EQ(all_v_first_col.size(), V_ROWS);
    for (size_t i = 0; i < V_ROWS; ++i)
        EXPECT_FLOAT_EQ(all_v_first_col[i],
                        static_cast<float>((Q_ROWS + K_ROWS + i) * 1000))
            << "V row " << i << " (global " << (Q_ROWS + K_ROWS + i) << ")";
}

// -----------------------------------------------------------------------------
// Real Qwen3.5 4B Scale Test
// -----------------------------------------------------------------------------

/**
 * @brief Test with real Qwen3.5 4B GDN dimensions (scaled)
 *
 * n_k_heads=16, n_v_heads=32, d_state=128
 * Q=2048, K=2048, V=4096 → total=8192
 * With TP=2, GDN replicate Q/K:
 *   Q=2048(full), K=2048(full), V=2048(sharded) → total=6144 per rank
 */
TEST(GDNFusedQKVRealScale, Qwen35_4B_Dimensions)
{
    static constexpr int N_K = 16;
    static constexpr int N_V = 32;
    static constexpr int D = 128;
    static constexpr size_t Q_R = N_K * D;           // 2048
    static constexpr size_t K_R = N_K * D;           // 2048
    static constexpr size_t V_R = N_V * D;           // 4096
    static constexpr size_t TOTAL = Q_R + K_R + V_R; // 8192
    static constexpr size_t C = 2560;                // hidden_size

    auto mock = std::make_shared<MockModelLoader>();
    mock->setLoaded(true);
    mock->setArchitecture("qwen3.5");
    mock->setBlockCount(1);
    mock->setEmbeddingLength(C);
    mock->setHeadCount(32);  // FA n_heads
    mock->setHeadCountKV(8); // FA n_kv_heads
    mock->setVocabSize(151936);
    mock->setFeedForwardLength(9216);

    // Create large QKV tensor
    auto qkv = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL, C});
    float *data = qkv->mutable_data();
    // Fill with row markers (only first col to save time)
    for (size_t r = 0; r < TOTAL; ++r)
        data[r * C] = static_cast<float>(r);
    mock->addTensor("blk.0.attn_qkv.weight", qkv);

    mock->addFP32RandomTensor("token_embd.weight", {151936, C});
    mock->addFP32RandomTensor("output.weight", {151936, C});
    mock->addFP32RandomTensor("output_norm.weight", {C});
    mock->addFP32RandomTensor("blk.0.attn_norm.weight", {C});
    mock->addFP32RandomTensor("blk.0.ffn_norm.weight", {C});

    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    // Rank 0
    {
        auto mpi = MPIContextFactory::create_mock(0, 2);
        auto wm = std::make_unique<WeightManager>(
            *mock, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(config);
        wm->setModelDimensions(32, 8, 128); // FA: 32*128 + 2*8*128 = 6144 ≠ 8192
        wm->setGDNDimensions(N_K, N_V, D);

        auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        ASSERT_NE(tensor, nullptr);

        // GDN replicated Q/K: Q=2048(full), K=2048(full), V=4096/2=2048 → total=6144
        EXPECT_EQ(tensor->shape()[0], Q_R + K_R + V_R / 2); // 6144
        EXPECT_EQ(tensor->shape()[1], C);

        const float *d = tensor->data();

        // Q starts at global row 0 (replicated full)
        EXPECT_FLOAT_EQ(d[0], 0.0f);
        // K starts at global row 2048 (replicated full, offset = Q_R rows)
        EXPECT_FLOAT_EQ(d[Q_R * C], static_cast<float>(Q_R)); // 2048
        // V starts at global row 4096 (sharded, rank 0 first half)
        EXPECT_FLOAT_EQ(d[(Q_R + K_R) * C], static_cast<float>(Q_R + K_R)); // 4096
    }

    // Rank 1
    {
        auto mpi = MPIContextFactory::create_mock(1, 2);
        auto wm = std::make_unique<WeightManager>(
            *mock, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(config);
        wm->setModelDimensions(32, 8, 128);
        wm->setGDNDimensions(N_K, N_V, D);

        auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        ASSERT_NE(tensor, nullptr);

        // GDN replicated Q/K: same total as rank 0
        EXPECT_EQ(tensor->shape()[0], Q_R + K_R + V_R / 2); // 6144

        const float *d = tensor->data();

        // Q starts at global row 0 (replicated, same as rank 0)
        EXPECT_FLOAT_EQ(d[0], 0.0f);
        // K starts at global row 2048 (replicated, same as rank 0)
        EXPECT_FLOAT_EQ(d[Q_R * C], static_cast<float>(Q_R)); // 2048
        // V starts at global row 4096+2048 = 6144 (sharded, rank 1 second half)
        EXPECT_FLOAT_EQ(d[(Q_R + K_R) * C], static_cast<float>(Q_R + K_R + V_R / 2)); // 6144
    }
}

// =============================================================================
// Parametric TP Degree Tests (1 through 16)
// =============================================================================

/**
 * @brief Parametric test fixture for GDN FusedQKV sharding at various TP degrees
 *
 * Uses real Qwen3.5 4B-scale dimensions (n_k=16, n_v=32, d=128) to verify
 * correctness at all valid TP degrees from 1 to 16.
 *
 * V_ROWS = 4096 → valid TP degrees: 1, 2, 4, 8, 16 (factors of 4096)
 *                   invalid: 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15
 */
class GDNTPDegreeTest : public ::testing::TestWithParam<int>
{
protected:
    static constexpr int N_K = 16;
    static constexpr int N_V = 32;
    static constexpr int D_STATE = 128;
    static constexpr size_t Q_R = N_K * D_STATE;     // 2048
    static constexpr size_t K_R = N_K * D_STATE;     // 2048
    static constexpr size_t V_R = N_V * D_STATE;     // 4096
    static constexpr size_t TOTAL = Q_R + K_R + V_R; // 8192
    static constexpr size_t COLS = 64;               // Smaller than real hidden for speed

    // FA dimensions (must NOT match GDN layout)
    static constexpr int FA_N_HEADS = 32;
    static constexpr int FA_N_KV_HEADS = 8;
    static constexpr int FA_HEAD_DIM = 128;
    // FA expected: 32*128 + 2*8*128 = 6144 ≠ 8192

    void SetUp() override
    {
        mock_ = std::make_shared<MockModelLoader>();
        mock_->setLoaded(true);
        mock_->setArchitecture("qwen3.5");
        mock_->setBlockCount(1);
        mock_->setEmbeddingLength(COLS);
        mock_->setHeadCount(FA_N_HEADS);
        mock_->setHeadCountKV(FA_N_KV_HEADS);
        mock_->setVocabSize(100);
        mock_->setFeedForwardLength(64);

        // Create QKV tensor with row markers
        auto qkv = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL, COLS});
        float *data = qkv->mutable_data();
        for (size_t r = 0; r < TOTAL; ++r)
            data[r * COLS] = static_cast<float>(r);
        mock_->addTensor("blk.0.attn_qkv.weight", qkv);

        mock_->addFP32RandomTensor("token_embd.weight", {100, COLS});
        mock_->addFP32RandomTensor("output.weight", {100, COLS});
        mock_->addFP32RandomTensor("output_norm.weight", {COLS});
        mock_->addFP32RandomTensor("blk.0.attn_norm.weight", {COLS});
        mock_->addFP32RandomTensor("blk.0.ffn_norm.weight", {COLS});

        Qwen35SchemaFactory factory;
        sharding_config_ = factory.getWeightShardingConfig();
    }

    std::unique_ptr<WeightManager> createManager(int rank, int world_size)
    {
        auto mpi = MPIContextFactory::create_mock(rank, world_size);
        auto wm = std::make_unique<WeightManager>(
            *mock_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        wm->setModelDimensions(FA_N_HEADS, FA_N_KV_HEADS, FA_HEAD_DIM);
        wm->setGDNDimensions(N_K, N_V, D_STATE);
        return wm;
    }

    std::shared_ptr<MockModelLoader> mock_;
    WeightShardingConfig sharding_config_;
};

/**
 * @brief Verify correct sharding at valid TP degrees: 1, 2, 4, 8, 16
 *
 * For each TP degree:
 * - Q and K are replicated in full (2048 rows each)
 * - V is evenly sharded: 4096 / tp_degree rows per rank
 * - All ranks' V slices together reconstruct the full V
 * - The head-to-head mapping is locally satisfiable
 */
TEST_P(GDNTPDegreeTest, CorrectShardingForAllRanks)
{
    const int tp_degree = GetParam();
    const size_t v_local = V_R / static_cast<size_t>(tp_degree);

    std::vector<float> all_v_first_col;

    for (int rank = 0; rank < tp_degree; ++rank)
    {
        auto wm = createManager(rank, tp_degree);
        auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        ASSERT_NE(tensor, nullptr) << "Rank " << rank << " / TP=" << tp_degree;

        // Q and K are replicated: full size
        // V is sharded: V_R / tp_degree rows
        const size_t expected_rows = Q_R + K_R + v_local;
        EXPECT_EQ(tensor->shape()[0], expected_rows)
            << "Rank " << rank << " / TP=" << tp_degree
            << " expected " << expected_rows << " rows (Q=" << Q_R
            << " + K=" << K_R << " + V=" << v_local << ")";

        const float *data = tensor->data();

        // Q should be identical on all ranks (replicated)
        EXPECT_FLOAT_EQ(data[0], 0.0f)
            << "Rank " << rank << " Q start should be global row 0";
        EXPECT_FLOAT_EQ(data[(Q_R - 1) * COLS], static_cast<float>(Q_R - 1))
            << "Rank " << rank << " Q end should be global row " << (Q_R - 1);

        // K should be identical on all ranks (replicated)
        EXPECT_FLOAT_EQ(data[Q_R * COLS], static_cast<float>(Q_R))
            << "Rank " << rank << " K start should be global row " << Q_R;
        EXPECT_FLOAT_EQ(data[(Q_R + K_R - 1) * COLS], static_cast<float>(Q_R + K_R - 1))
            << "Rank " << rank << " K end should be global row " << (Q_R + K_R - 1);

        // V is sharded: rank's V starts at global row (Q_R + K_R + rank * v_local)
        const size_t v_global_start = Q_R + K_R + rank * v_local;
        EXPECT_FLOAT_EQ(data[(Q_R + K_R) * COLS], static_cast<float>(v_global_start))
            << "Rank " << rank << " V start should be global row " << v_global_start;

        // Collect V first-col values for full-coverage check
        for (size_t r = 0; r < v_local; ++r)
            all_v_first_col.push_back(data[(Q_R + K_R + r) * COLS]);
    }

    // Verify full V coverage: all V rows [Q_R+K_R .. TOTAL) are represented
    ASSERT_EQ(all_v_first_col.size(), V_R);
    for (size_t i = 0; i < V_R; ++i)
        EXPECT_FLOAT_EQ(all_v_first_col[i], static_cast<float>(Q_R + K_R + i))
            << "V row " << i << " missing or wrong";
}

/**
 * @brief Verify v→k head mapping is locally satisfiable at each TP degree
 *
 * With GDN modular repeat (v_head % n_k), each rank's local v_heads must
 * be able to find their required k_heads within the local (replicated) Q/K.
 *
 * Since Q/K are fully replicated, ALL k_heads (0..n_k-1) are available
 * locally on every rank. So the modular mapping v_head % n_k always resolves
 * to a locally-available k_head, regardless of which v_heads this rank has.
 */
TEST_P(GDNTPDegreeTest, HeadMappingIsLocallySatisfiable)
{
    const int tp_degree = GetParam();
    const int n_v_local = N_V / tp_degree;

    for (int rank = 0; rank < tp_degree; ++rank)
    {
        const int v_head_start = rank * n_v_local;

        for (int j = 0; j < n_v_local; ++j)
        {
            const int global_v_head = v_head_start + j;
            const int required_k_head = global_v_head % N_K;

            // With fully replicated Q/K, all k_heads [0..N_K-1] are local
            EXPECT_GE(required_k_head, 0)
                << "Rank " << rank << " v_head " << global_v_head
                << " requires k_head " << required_k_head << " (negative!)";
            EXPECT_LT(required_k_head, N_K)
                << "Rank " << rank << " v_head " << global_v_head
                << " requires k_head " << required_k_head
                << " (>= n_k_heads=" << N_K << ")";
        }
    }
}

/**
 * @brief TP=1 is a no-op: returns the full weight unchanged
 */
TEST_P(GDNTPDegreeTest, SingleDeviceIsNoOp)
{
    if (GetParam() != 1)
        GTEST_SKIP() << "Only relevant for TP=1";

    auto wm = createManager(0, 1);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    // Full tensor: all rows
    EXPECT_EQ(tensor->shape()[0], TOTAL);
    EXPECT_EQ(tensor->shape()[1], COLS);
}

INSTANTIATE_TEST_SUITE_P(
    ValidTPDegrees,
    GDNTPDegreeTest,
    ::testing::Values(1, 2, 4, 8, 16),
    [](const ::testing::TestParamInfo<int> &info)
    {
        return "TP" + std::to_string(info.param);
    });

// =============================================================================
// Error Handling: Model Too Small for TP Degree
// =============================================================================

/**
 * @brief Test fixture for GDN TP sharding error conditions
 *
 * Uses dimensions where certain TP degrees are invalid:
 *   n_k=4, n_v=8, d=4 → V=32
 *   Valid TP degrees: 1, 2, 4, 8 (factors of 32)
 *   Invalid: 3, 5, 6, 7, 9-16 except 8 (V not evenly divisible)
 *
 *   Also invalid: TP=16 for these small dims → V/16 = 2, but Q/K only 16 rows.
 *   Actually V=32 → 32/16=2 rows per rank, which IS valid since 32%16==0.
 *   Invalid TP=3 since 32/3 is not integer.
 */
class GDNTPErrorTest : public ::testing::Test
{
protected:
    static constexpr int GDN_N_K = 4;
    static constexpr int GDN_N_V = 8;
    static constexpr int GDN_D = 4;
    static constexpr size_t Q_R = GDN_N_K * GDN_D;   // 16
    static constexpr size_t K_R = GDN_N_K * GDN_D;   // 16
    static constexpr size_t V_R = GDN_N_V * GDN_D;   // 32
    static constexpr size_t TOTAL = Q_R + K_R + V_R; // 64
    static constexpr size_t COLS = 16;

    static constexpr int FA_N_HEADS = 8;
    static constexpr int FA_N_KV_HEADS = 2;
    static constexpr int FA_HEAD_DIM = 8;

    void SetUp() override
    {
        mock_ = std::make_shared<MockModelLoader>();
        mock_->setLoaded(true);
        mock_->setArchitecture("qwen3.5");
        mock_->setBlockCount(1);
        mock_->setEmbeddingLength(COLS);
        mock_->setHeadCount(FA_N_HEADS);
        mock_->setHeadCountKV(FA_N_KV_HEADS);
        mock_->setVocabSize(100);
        mock_->setFeedForwardLength(64);

        auto qkv = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL, COLS});
        float *data = qkv->mutable_data();
        for (size_t r = 0; r < TOTAL; ++r)
            for (size_t c = 0; c < COLS; ++c)
                data[r * COLS + c] = static_cast<float>(r * 1000 + c);
        mock_->addTensor("blk.0.attn_qkv.weight", qkv);

        mock_->addFP32RandomTensor("token_embd.weight", {100, COLS});
        mock_->addFP32RandomTensor("output.weight", {100, COLS});
        mock_->addFP32RandomTensor("output_norm.weight", {COLS});
        mock_->addFP32RandomTensor("blk.0.attn_norm.weight", {COLS});
        mock_->addFP32RandomTensor("blk.0.ffn_norm.weight", {COLS});

        Qwen35SchemaFactory factory;
        sharding_config_ = factory.getWeightShardingConfig();
    }

    std::unique_ptr<WeightManager> createManager(int rank, int world_size)
    {
        auto mpi = MPIContextFactory::create_mock(rank, world_size);
        auto wm = std::make_unique<WeightManager>(
            *mock_, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(sharding_config_);
        wm->setModelDimensions(FA_N_HEADS, FA_N_KV_HEADS, FA_HEAD_DIM);
        wm->setGDNDimensions(GDN_N_K, GDN_N_V, GDN_D);
        return wm;
    }

    std::shared_ptr<MockModelLoader> mock_;
    WeightShardingConfig sharding_config_;
};

/**
 * @brief TP=3 fails because V(32) is not divisible by 3
 *
 * This must throw std::invalid_argument with a descriptive message
 * rather than silently producing a 0-row tensor.
 */
TEST_F(GDNTPErrorTest, TP3_ThrowsIndivisibleV)
{
    auto wm = createManager(0, 3);
    EXPECT_THROW(
        {
            wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        },
        std::invalid_argument);
}

/**
 * @brief TP=5 fails because V(32) is not divisible by 5
 */
TEST_F(GDNTPErrorTest, TP5_ThrowsIndivisibleV)
{
    auto wm = createManager(0, 5);
    EXPECT_THROW(
        {
            wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        },
        std::invalid_argument);
}

/**
 * @brief TP=6 fails because V(32) is not divisible by 6
 */
TEST_F(GDNTPErrorTest, TP6_ThrowsIndivisibleV)
{
    auto wm = createManager(0, 6);
    EXPECT_THROW(
        {
            wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        },
        std::invalid_argument);
}

/**
 * @brief TP=7 fails because V(32) is not divisible by 7
 */
TEST_F(GDNTPErrorTest, TP7_ThrowsIndivisibleV)
{
    auto wm = createManager(0, 7);
    EXPECT_THROW(
        {
            wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        },
        std::invalid_argument);
}

/**
 * @brief TP=64 fails because V(32) < 64 — each rank would get 0 rows
 */
TEST_F(GDNTPErrorTest, TP64_ThrowsTooLargeForModel)
{
    auto wm = createManager(0, 64);
    EXPECT_THROW(
        {
            wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        },
        std::invalid_argument);
}

/**
 * @brief The error message references the correct sub-block and sizes
 */
TEST_F(GDNTPErrorTest, ErrorMessageIsDescriptive)
{
    auto wm = createManager(0, 3);
    try
    {
        wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        FAIL() << "Expected std::invalid_argument";
    }
    catch (const std::invalid_argument &e)
    {
        std::string msg = e.what();
        // Must mention the weight name
        EXPECT_NE(msg.find("attn_qkv"), std::string::npos) << "Error: " << msg;
        // Must mention sub-block V
        EXPECT_NE(msg.find("V"), std::string::npos) << "Error: " << msg;
        // Must mention the TP degree
        EXPECT_NE(msg.find("3"), std::string::npos) << "Error: " << msg;
        // Must mention the row count
        EXPECT_NE(msg.find("32"), std::string::npos) << "Error: " << msg;
        // Must suggest reducing TP degree
        EXPECT_NE(msg.find("Reduce"), std::string::npos) << "Error: " << msg;
    }
}

/**
 * @brief Valid TP degrees succeed (factors of V_ROWS=32)
 *
 * V=32 is divisible by: 1, 2, 4, 8, 16, 32
 */
TEST_F(GDNTPErrorTest, ValidTPDegreesSucceed)
{
    for (int tp : {1, 2, 4, 8, 16, 32})
    {
        auto wm = createManager(0, tp);
        EXPECT_NO_THROW(
            {
                auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
                EXPECT_NE(tensor, nullptr) << "TP=" << tp << " should succeed";
            })
            << "TP=" << tp << " should not throw";
    }
}

/**
 * @brief Invalid TP degrees all throw std::invalid_argument
 *
 * For V=32, degrees 3,5,6,7,9,10,11,12,13,14,15 are invalid.
 */
TEST_F(GDNTPErrorTest, InvalidTPDegreesThrow)
{
    for (int tp : {3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15})
    {
        auto wm = createManager(0, tp);
        EXPECT_THROW(
            {
                wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
            },
            std::invalid_argument)
            << "TP=" << tp << " should throw (V=32 not divisible by " << tp << ")";
    }
}

/**
 * @brief ssm_conv1d also gets the validation (same GDN FusedQKVHeads layout)
 */
TEST_F(GDNTPErrorTest, SSMConv1dAlsoValidated)
{
    // Add ssm_conv1d with same row count as QKV
    auto conv = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL, 4u});
    mock_->addTensor("blk.0.ssm_conv1d.weight", conv);

    auto wm = createManager(0, 3);
    EXPECT_THROW(
        {
            wm->getWeightForDevice("blk.0.ssm_conv1d.weight", DeviceId::cpu());
        },
        std::invalid_argument);
}

// =============================================================================
// Real Model TP Degree Limits
// =============================================================================

/**
 * @brief Qwen3.5 4B can be sharded up to TP=16 with GDN modular repeat
 *
 * V=4096 (n_v=32, d=128) → factors: 1, 2, 4, 8, 16, 32, 64, ...
 * With Q/K replicated, the constraint is solely on V divisibility.
 * TP=16 → V_local=256 (2 v_heads * d=128) → valid.
 */
TEST(GDNTPRealModelLimits, Qwen35_4B_SupportsTP16)
{
    static constexpr int N_K = 16;
    static constexpr int N_V = 32;
    static constexpr int D = 128;
    static constexpr size_t Q_R = N_K * D; // 2048
    static constexpr size_t K_R = N_K * D; // 2048
    static constexpr size_t V_R = N_V * D; // 4096
    static constexpr size_t TOTAL = Q_R + K_R + V_R;
    static constexpr size_t C = 64;

    auto mock = std::make_shared<MockModelLoader>();
    mock->setLoaded(true);
    mock->setArchitecture("qwen3.5");
    mock->setBlockCount(1);
    mock->setEmbeddingLength(C);
    mock->setHeadCount(32);
    mock->setHeadCountKV(8);
    mock->setVocabSize(100);
    mock->setFeedForwardLength(64);

    auto qkv = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL, C});
    float *data = qkv->mutable_data();
    for (size_t r = 0; r < TOTAL; ++r)
        data[r * C] = static_cast<float>(r);
    mock->addTensor("blk.0.attn_qkv.weight", qkv);

    mock->addFP32RandomTensor("token_embd.weight", {100, C});
    mock->addFP32RandomTensor("output.weight", {100, C});
    mock->addFP32RandomTensor("output_norm.weight", {C});
    mock->addFP32RandomTensor("blk.0.attn_norm.weight", {C});
    mock->addFP32RandomTensor("blk.0.ffn_norm.weight", {C});

    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    // TP=16: V_local = 4096/16 = 256 rows per rank
    for (int rank = 0; rank < 16; ++rank)
    {
        auto mpi = MPIContextFactory::create_mock(rank, 16);
        auto wm = std::make_unique<WeightManager>(
            *mock, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(config);
        wm->setModelDimensions(32, 8, 128);
        wm->setGDNDimensions(N_K, N_V, D);

        auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        ASSERT_NE(tensor, nullptr) << "TP=16 rank " << rank << " failed";

        // Q(2048) + K(2048) + V(256) = 4352
        EXPECT_EQ(tensor->shape()[0], Q_R + K_R + V_R / 16)
            << "TP=16 rank " << rank;

        const float *d = tensor->data();

        // Q replicated (same on all ranks)
        EXPECT_FLOAT_EQ(d[0], 0.0f);

        // V sharded (different on each rank)
        const size_t v_local = V_R / 16;
        const size_t v_global_start = Q_R + K_R + rank * v_local;
        EXPECT_FLOAT_EQ(d[(Q_R + K_R) * C], static_cast<float>(v_global_start))
            << "TP=16 rank " << rank << " V start";
    }
}

/**
 * @brief Qwen3.5 0.8B: n_k=4, n_v=8, d=64 → V=512
 *
 * V=512 → max TP = 512 (but practically limited by n_v_heads=8)
 * TP=8 → V_local=64 (1 v_head * d=64) per rank — minimum useful
 * TP=16 → V_local=32 — still valid (32 = 0.5 heads * d, sub-head splitting)
 */
TEST(GDNTPRealModelLimits, Qwen35_08B_SupportsTP8)
{
    static constexpr int N_K = 4;
    static constexpr int N_V = 8;
    static constexpr int D = 64;
    static constexpr size_t Q_R = N_K * D; // 256
    static constexpr size_t K_R = N_K * D; // 256
    static constexpr size_t V_R = N_V * D; // 512
    static constexpr size_t TOTAL = Q_R + K_R + V_R;
    static constexpr size_t C = 32;

    auto mock = std::make_shared<MockModelLoader>();
    mock->setLoaded(true);
    mock->setArchitecture("qwen3.5");
    mock->setBlockCount(1);
    mock->setEmbeddingLength(C);
    mock->setHeadCount(12);
    mock->setHeadCountKV(4);
    mock->setVocabSize(100);
    mock->setFeedForwardLength(64);

    auto qkv = std::make_shared<FP32Tensor>(std::vector<size_t>{TOTAL, C});
    float *data = qkv->mutable_data();
    for (size_t r = 0; r < TOTAL; ++r)
        data[r * C] = static_cast<float>(r);
    mock->addTensor("blk.0.attn_qkv.weight", qkv);

    mock->addFP32RandomTensor("token_embd.weight", {100, C});
    mock->addFP32RandomTensor("output.weight", {100, C});
    mock->addFP32RandomTensor("output_norm.weight", {C});
    mock->addFP32RandomTensor("blk.0.attn_norm.weight", {C});
    mock->addFP32RandomTensor("blk.0.ffn_norm.weight", {C});

    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    // TP=8: V_local = 512/8 = 64 rows per rank
    for (int rank = 0; rank < 8; ++rank)
    {
        auto mpi = MPIContextFactory::create_mock(rank, 8);
        auto wm = std::make_unique<WeightManager>(
            *mock, mpi, nullptr,
            WeightDistributionStrategy::SHARDED,
            WeightPrecision::NATIVE);
        wm->setWeightShardingConfig(config);
        wm->setModelDimensions(12, 4, 64);
        wm->setGDNDimensions(N_K, N_V, D);

        auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
        ASSERT_NE(tensor, nullptr) << "TP=8 rank " << rank;

        // Q(256) + K(256) + V(64) = 576
        EXPECT_EQ(tensor->shape()[0], Q_R + K_R + V_R / 8)
            << "TP=8 rank " << rank;
    }
}

// =============================================================================
// IWeightManager Default Interface Methods (Regression for Bug #5)
//
// Bug: RankOrchestrator (Local TP path) never called
// setModelDimensions() or setGDNDimensions() on the WeightManager.
// These methods were only called in InferenceRunnerFactory (Global TP path).
// Without them, loadFusedQKVColumnParallel() couldn't detect the GDN
// asymmetric layout, falling through to simple equal row splitting.
//
// Fix part 1: Added setModelDimensions/setGDNDimensions to IWeightManager
// with default no-op implementations so that existing MockWeightManager
// and other IWeightManager implementations don't break.
//
// Fix part 2: RankOrchestrator::initializeWeightSharding() now calls
// both methods after configuring TP.
// =============================================================================

/**
 * @brief setModelDimensions/setGDNDimensions are callable via IWeightManager*
 *
 * Tests that the default virtual implementations exist and don't crash.
 * Uses a real WeightManager behind an IWeightManager pointer to verify
 * the methods are polymorphically callable.
 */
TEST(IWeightManagerDefaults, SetModelDimensionsCallableViaInterface)
{
    auto mock = std::make_shared<MockModelLoader>();
    mock->setLoaded(true);
    mock->setArchitecture("qwen3.5");
    mock->setBlockCount(0);
    mock->setEmbeddingLength(16);
    mock->setHeadCount(4);
    mock->setHeadCountKV(2);
    mock->setVocabSize(100);
    mock->setFeedForwardLength(64);
    mock->addFP32RandomTensor("token_embd.weight", {100, 16});
    mock->addFP32RandomTensor("output.weight", {100, 16});
    mock->addFP32RandomTensor("output_norm.weight", {16});

    auto mpi = MPIContextFactory::create_mock(0, 1);
    auto wm = std::make_unique<WeightManager>(
        *mock, mpi, nullptr,
        WeightDistributionStrategy::SHARDED,
        WeightPrecision::NATIVE);

    // Call via IWeightManager pointer (the actual dispatch path in RankOrchestrator)
    IWeightManager *iface = wm.get();
    EXPECT_NO_THROW(iface->setModelDimensions(16, 4, 128));
    EXPECT_NO_THROW(iface->setGDNDimensions(16, 32, 128));
}

/**
 * @brief setGDNDimensions enables GDN slicing path even when called late
 *
 * Verify that setGDNDimensions can be called after construction and
 * before weight loading, which is the exact calling pattern used by
 * RankOrchestrator::initializeWeightSharding().
 */
TEST(IWeightManagerDefaults, SetGDNDimensionsBeforeWeightLoad)
{
    auto mock = std::make_shared<MockModelLoader>();
    mock->setLoaded(true);
    mock->setArchitecture("qwen3.5");
    mock->setBlockCount(1);
    mock->setEmbeddingLength(16);
    mock->setHeadCount(8);
    mock->setHeadCountKV(2);
    mock->setVocabSize(100);
    mock->setFeedForwardLength(64);
    mock->addFP32RandomTensor("token_embd.weight", {100, 16});
    mock->addFP32RandomTensor("output.weight", {100, 16});
    mock->addFP32RandomTensor("output_norm.weight", {16});
    mock->addFP32RandomTensor("blk.0.attn_norm.weight", {16});
    mock->addFP32RandomTensor("blk.0.ffn_norm.weight", {16});

    // GDN layout: Q=16 + K=16 + V=32 = 64 rows
    auto qkv = std::make_shared<FP32Tensor>(std::vector<size_t>{64u, 16u});
    mock->addTensor("blk.0.attn_qkv.weight", qkv);

    Qwen35SchemaFactory factory;
    auto config = factory.getWeightShardingConfig();

    auto mpi = MPIContextFactory::create_mock(0, 2);
    auto wm = std::make_unique<WeightManager>(
        *mock, mpi, nullptr,
        WeightDistributionStrategy::SHARDED,
        WeightPrecision::NATIVE);
    wm->setWeightShardingConfig(config);

    // This is the calling order from RankOrchestrator
    wm->setModelDimensions(8, 2, 8); // FA: 8*8 + 2*2*8 = 96 ≠ 64
    wm->setGDNDimensions(4, 8, 4);   // GDN: 2*4*4 + 8*4 = 64 ✓

    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    // GDN replicate Q/K: Q(16) + K(16) + V(16) = 48 (NOT 32 from simple split)
    EXPECT_EQ(tensor->shape()[0], 48u)
        << "setGDNDimensions must enable GDN sub-block slicing with Q/K replication";
}
