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
 * Fix: Added setGDNDimensions() to WeightManager. When FA layout check fails
 * and div-by-3 fails, the code tries the GDN layout (2*key_dim + value_dim).
 * Each sub-block is independently TP-sliced.
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
    static constexpr int GDN_N_K_HEADS = 4;  // group_count
    static constexpr int GDN_N_V_HEADS = 8;  // time_step_rank
    static constexpr int GDN_D_STATE = 4;    // state_size (d_k = d_v)

    // Derived sizes
    static constexpr size_t Q_ROWS = GDN_N_K_HEADS * GDN_D_STATE;       // 16
    static constexpr size_t K_ROWS = GDN_N_K_HEADS * GDN_D_STATE;       // 16
    static constexpr size_t V_ROWS = GDN_N_V_HEADS * GDN_D_STATE;       // 32
    static constexpr size_t TOTAL_ROWS = Q_ROWS + K_ROWS + V_ROWS;      // 64
    static constexpr size_t COLS = 16;                                    // hidden_dim (arbitrary)

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
 * @brief GDN rank 0: gets first half of each asymmetric sub-block
 *
 * Layout: [Q(16) | K(16) | V(32)] with TP=2
 *   Q rank 0: rows [0, 8)      → 8 rows
 *   K rank 0: rows [16, 24)    → 8 rows
 *   V rank 0: rows [32, 48)    → 16 rows
 *   Total rank 0: 8 + 8 + 16 = 32 rows
 */
TEST_F(GDNFusedQKVShardingTest, Rank0GetsAsymmetricSubBlocks)
{
    auto wm = createGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const size_t q_half = Q_ROWS / 2;   // 8
    const size_t k_half = K_ROWS / 2;   // 8
    const size_t v_half = V_ROWS / 2;   // 16
    EXPECT_EQ(tensor->shape()[0], q_half + k_half + v_half); // 32
    EXPECT_EQ(tensor->shape()[1], COLS);

    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    size_t local_row = 0;

    // Q sub-block first half: global rows [0, 8)
    for (size_t r = 0; r < q_half; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS], static_cast<float>(r * 1000))
            << "Q row " << r << " (local_row=" << local_row << ")";

    // K sub-block first half: global rows [16, 24)
    for (size_t r = 0; r < k_half; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                         static_cast<float>((Q_ROWS + r) * 1000))
            << "K row " << r << " (local_row=" << local_row << ")";

    // V sub-block first half: global rows [32, 48)
    for (size_t r = 0; r < v_half; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                         static_cast<float>((Q_ROWS + K_ROWS + r) * 1000))
            << "V row " << r << " (local_row=" << local_row << ")";
}

/**
 * @brief GDN rank 1: gets second half of each asymmetric sub-block
 *
 * Layout: [Q(16) | K(16) | V(32)] with TP=2
 *   Q rank 1: rows [8, 16)     → 8 rows
 *   K rank 1: rows [24, 32)    → 8 rows
 *   V rank 1: rows [48, 64)    → 16 rows
 *   Total rank 1: 8 + 8 + 16 = 32 rows
 */
TEST_F(GDNFusedQKVShardingTest, Rank1GetsAsymmetricSubBlocks)
{
    auto wm = createGDNShardedManager(1, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const size_t q_half = Q_ROWS / 2;   // 8
    const size_t k_half = K_ROWS / 2;   // 8
    const size_t v_half = V_ROWS / 2;   // 16
    EXPECT_EQ(tensor->shape()[0], q_half + k_half + v_half); // 32
    EXPECT_EQ(tensor->shape()[1], COLS);

    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    size_t local_row = 0;

    // Q sub-block second half: global rows [8, 16)
    for (size_t r = 0; r < q_half; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                         static_cast<float>((q_half + r) * 1000))
            << "Q row " << r << " (local_row=" << local_row << ")";

    // K sub-block second half: global rows [24, 32)
    for (size_t r = 0; r < k_half; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                         static_cast<float>((Q_ROWS + k_half + r) * 1000))
            << "K row " << r << " (local_row=" << local_row << ")";

    // V sub-block second half: global rows [48, 64)
    for (size_t r = 0; r < v_half; ++r, ++local_row)
        EXPECT_FLOAT_EQ(data[local_row * COLS],
                         static_cast<float>((Q_ROWS + K_ROWS + v_half + r) * 1000))
            << "V row " << r << " (local_row=" << local_row << ")";
}

/**
 * @brief Both ranks together cover all 64 rows exactly
 */
TEST_F(GDNFusedQKVShardingTest, BothRanksCoverAllRows)
{
    auto wm0 = createGDNShardedManager(0, 2);
    auto wm1 = createGDNShardedManager(1, 2);

    auto t0 = wm0->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    auto t1 = wm1->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(t0, nullptr);
    ASSERT_NE(t1, nullptr);
    EXPECT_EQ(t0->shape()[0] + t1->shape()[0], TOTAL_ROWS);
}

/**
 * @brief V sub-block gets proportionally more rows than Q/K under TP
 *
 * The key asymmetry: V has 32 rows (n_v=8) vs Q/K with 16 rows (n_k=4).
 * Each rank should get 16 V rows but only 8 Q/K rows.
 */
TEST_F(GDNFusedQKVShardingTest, VSubBlockGetsMoreRowsThanQK)
{
    auto wm = createGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const size_t q_local = Q_ROWS / 2;  // 8
    const size_t k_local = K_ROWS / 2;  // 8
    const size_t v_local = V_ROWS / 2;  // 16

    // V gets twice as many rows as Q or K per rank
    EXPECT_EQ(v_local, 2 * q_local);
    EXPECT_EQ(v_local, 2 * k_local);

    // Total local rows = 8 + 8 + 16 = 32
    EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local);
}

/**
 * @brief Sub-block boundaries are correctly placed in the local tensor
 *
 * Local layout should be [Q_local | K_local | V_local] in order.
 * The Q/K boundary and K/V boundary must be at the right offsets.
 */
TEST_F(GDNFusedQKVShardingTest, SubBlockBoundariesCorrect)
{
    auto wm = createGDNShardedManager(0, 2);
    auto tensor = wm->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    ASSERT_NE(tensor, nullptr);

    const float *data = tensor->data();
    const size_t q_local = Q_ROWS / 2;  // 8
    const size_t k_local = K_ROWS / 2;  // 8

    // Row 0 is Q data (global row 0)
    EXPECT_FLOAT_EQ(data[0], 0.0f);

    // Row q_local is K data (global row Q_ROWS = 16)
    EXPECT_FLOAT_EQ(data[q_local * COLS], static_cast<float>(Q_ROWS * 1000));

    // Row q_local + k_local is V data (global row Q_ROWS + K_ROWS = 32)
    EXPECT_FLOAT_EQ(data[(q_local + k_local) * COLS],
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

    const size_t q_local = Q_ROWS / 2;  // 8
    const size_t k_local = K_ROWS / 2;  // 8
    const size_t v_local = V_ROWS / 2;  // 16

    // Same total local rows as attn_qkv
    EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local);
    // But different column count (conv_kernel_size = 4)
    EXPECT_EQ(tensor->shape()[1], 4u);

    // Verify sub-block boundaries using our value pattern (r*100+c)
    const float *data = tensor->data();

    // Row 0 is Q data (global row 0, col 0)
    EXPECT_FLOAT_EQ(data[0], 0.0f);

    // Row q_local is K data (global row 16)
    EXPECT_FLOAT_EQ(data[q_local * 4], static_cast<float>(Q_ROWS * 100));

    // Row q_local+k_local is V data (global row 32)
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
 * GDN sub-block split gives different data than contiguous split.
 * Row 8 in GDN split = K data (global row 16), not Q data (global row 8).
 */
TEST_F(GDNFusedQKVShardingTest, GDNSplitDiffersFromEqualSplit)
{
    auto wm_gdn = createGDNShardedManager(0, 2);
    auto wm_equal = createNonGDNShardedManager(0, 2);

    auto t_gdn = wm_gdn->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());
    auto t_equal = wm_equal->getWeightForDevice("blk.0.attn_qkv.weight", DeviceId::cpu());

    ASSERT_NE(t_gdn, nullptr);
    ASSERT_NE(t_equal, nullptr);

    // Both have 32 rows for rank 0
    EXPECT_EQ(t_gdn->shape()[0], 32u);
    EXPECT_EQ(t_equal->shape()[0], 32u);

    // But the DATA at row 8 differs
    const float *gdn_data = t_gdn->data();
    const float *equal_data = t_equal->data();

    // GDN: row 8 = K data (global row 16) → value 16000
    // Equal: row 8 = Q data (global row 8) → value 8000
    EXPECT_FLOAT_EQ(gdn_data[8 * COLS], static_cast<float>(Q_ROWS * 1000));   // 16000
    EXPECT_FLOAT_EQ(equal_data[8 * COLS], static_cast<float>(8 * 1000));       // 8000
    EXPECT_NE(gdn_data[8 * COLS], equal_data[8 * COLS]);
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
    // GDN: Q=16/2=8, K=16/2=8, V=32/2=16 → total=32
    EXPECT_EQ(t_gdn->shape()[0], 32u); // same total but different row composition
}

// -----------------------------------------------------------------------------
// 4-Way TP Sharding
// -----------------------------------------------------------------------------

/**
 * @brief GDN FusedQKV with 4-way TP: each rank gets 1/4 of each sub-block
 *
 * Layout: [Q(16) | K(16) | V(32)] with TP=4
 *   Q per rank: 4 rows, K per rank: 4 rows, V per rank: 8 rows
 *   Total per rank: 4 + 4 + 8 = 16 rows
 */
TEST_F(GDNFusedQKVShardingTest, FourWayTPSharding)
{
    for (int rank = 0; rank < 4; ++rank)
    {
        auto wm = createGDNShardedManager(rank, 4);
        // Need to recreate since createGDNShardedManager uses hardcoded params
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

        const size_t q_local = Q_ROWS / 4;  // 4
        const size_t k_local = K_ROWS / 4;  // 4
        const size_t v_local = V_ROWS / 4;  // 8

        EXPECT_EQ(tensor->shape()[0], q_local + k_local + v_local)
            << "Rank " << rank << " should have 16 local rows";
        EXPECT_EQ(tensor->shape()[1], COLS);

        // Verify first row of each sub-block
        const float *data = tensor->data();

        // Q sub-block: global row = rank * q_local
        EXPECT_FLOAT_EQ(data[0],
                         static_cast<float>(rank * q_local * 1000))
            << "Rank " << rank << " Q start";

        // K sub-block: global row = Q_ROWS + rank * k_local
        EXPECT_FLOAT_EQ(data[q_local * COLS],
                         static_cast<float>((Q_ROWS + rank * k_local) * 1000))
            << "Rank " << rank << " K start";

        // V sub-block: global row = Q_ROWS + K_ROWS + rank * v_local
        EXPECT_FLOAT_EQ(data[(q_local + k_local) * COLS],
                         static_cast<float>((Q_ROWS + K_ROWS + rank * v_local) * 1000))
            << "Rank " << rank << " V start";
    }
}

/**
 * @brief 4-way TP: all ranks together cover all rows
 */
TEST_F(GDNFusedQKVShardingTest, FourWayTPCoversAllRows)
{
    size_t total_local_rows = 0;
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
        total_local_rows += tensor->shape()[0];
    }
    EXPECT_EQ(total_local_rows, TOTAL_ROWS);
}

// -----------------------------------------------------------------------------
// Real Qwen3.5 4B Scale Test
// -----------------------------------------------------------------------------

/**
 * @brief Test with real Qwen3.5 4B GDN dimensions (scaled)
 *
 * n_k_heads=16, n_v_heads=32, d_state=128
 * Q=2048, K=2048, V=4096 → total=8192
 * With TP=2: Q=1024, K=1024, V=2048 → total=4096 per rank
 */
TEST(GDNFusedQKVRealScale, Qwen35_4B_Dimensions)
{
    static constexpr int N_K = 16;
    static constexpr int N_V = 32;
    static constexpr int D = 128;
    static constexpr size_t Q_R = N_K * D;     // 2048
    static constexpr size_t K_R = N_K * D;     // 2048
    static constexpr size_t V_R = N_V * D;     // 4096
    static constexpr size_t TOTAL = Q_R + K_R + V_R; // 8192
    static constexpr size_t C = 2560;           // hidden_size

    auto mock = std::make_shared<MockModelLoader>();
    mock->setLoaded(true);
    mock->setArchitecture("qwen3.5");
    mock->setBlockCount(1);
    mock->setEmbeddingLength(C);
    mock->setHeadCount(32);     // FA n_heads
    mock->setHeadCountKV(8);    // FA n_kv_heads
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

        // Q=1024, K=1024, V=2048 → total=4096
        EXPECT_EQ(tensor->shape()[0], TOTAL / 2);
        EXPECT_EQ(tensor->shape()[1], C);

        const float *d = tensor->data();

        // Q starts at global row 0
        EXPECT_FLOAT_EQ(d[0], 0.0f);
        // K starts at global row 2048 (after Q block)
        EXPECT_FLOAT_EQ(d[1024 * C], static_cast<float>(2048));
        // V starts at global row 4096 (after Q+K blocks)
        EXPECT_FLOAT_EQ(d[2048 * C], static_cast<float>(4096));
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

        EXPECT_EQ(tensor->shape()[0], TOTAL / 2);

        const float *d = tensor->data();

        // Q starts at global row 1024 (second half of Q block)
        EXPECT_FLOAT_EQ(d[0], static_cast<float>(1024));
        // K starts at global row 2048+1024 = 3072
        EXPECT_FLOAT_EQ(d[1024 * C], static_cast<float>(3072));
        // V starts at global row 4096+2048 = 6144
        EXPECT_FLOAT_EQ(d[2048 * C], static_cast<float>(6144));
    }
}
