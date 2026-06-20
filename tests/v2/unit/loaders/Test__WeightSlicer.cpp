/**
 * @file Test__WeightSlicer.cpp
 * @brief Unit tests for the WeightSlicer stateless slicing component
 *
 * Tests pure slicing math with NO I/O, no model loading, no MPI.
 * The WeightSlicer takes ModelDimensions + WeightShardingConfig and
 * computes slice boundaries for weight tensors.
 */

#include <gtest/gtest.h>
#include <memory>
#include <stdexcept>

#include "loaders/WeightSlicer.h"
#include "models/qwen/Qwen2Schema.h"
#include "models/qwen35/Qwen35Schema.h"
#include "config/TensorParallelConfig.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__WeightSlicer : public ::testing::Test
{
protected:
    // Qwen2 0.5B dimensions
    static constexpr int N_HEADS = 14;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;

    // Derived
    static constexpr int Q_DIM = N_HEADS * HEAD_DIM;     // 896
    static constexpr int KV_DIM = N_KV_HEADS * HEAD_DIM; // 128
    static constexpr int FUSED_QKV = Q_DIM + 2 * KV_DIM; // 1152

    // FFN dimensions (from Qwen2 0.5B)
    static constexpr int D_FF = 4864;
    static constexpr int VOCAB = 151936;

    ModelDimensions makeDimensions(int n_heads = N_HEADS, int n_kv = N_KV_HEADS,
                                   int hd = HEAD_DIM) const
    {
        return ModelDimensions{
            .n_heads = n_heads,
            .n_kv_heads = n_kv,
            .head_dim = hd};
    }

    ModelDimensions makeGDNDimensions(int gdn_nk, int gdn_nv, int gdn_d) const
    {
        return ModelDimensions{
            .n_heads = N_HEADS,
            .n_kv_heads = N_KV_HEADS,
            .head_dim = static_cast<size_t>(HEAD_DIM),
            .gdn_n_k_heads = gdn_nk,
            .gdn_n_v_heads = gdn_nv,
            .gdn_d_state = gdn_d};
    }

    WeightShardingConfig qwen2Config() const
    {
        Qwen2SchemaFactory factory;
        return factory.getWeightShardingConfig();
    }

    WeightShardingConfig qwen35Config() const
    {
        Qwen35SchemaFactory factory;
        return factory.getWeightShardingConfig();
    }
};

// =============================================================================
// Weight Categorization
// =============================================================================

TEST_F(Test__WeightSlicer, CategorizeWeight_QKV)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    EXPECT_EQ(slicer.categorizeWeight("blk.0.attn_q.weight"),
              WeightSlicer::WeightCategory::ATTENTION_QKV);
    EXPECT_EQ(slicer.categorizeWeight("blk.0.attn_k.weight"),
              WeightSlicer::WeightCategory::ATTENTION_QKV);
    EXPECT_EQ(slicer.categorizeWeight("blk.0.attn_v.weight"),
              WeightSlicer::WeightCategory::ATTENTION_QKV);
    EXPECT_EQ(slicer.categorizeWeight("blk.5.attn_qkv.weight"),
              WeightSlicer::WeightCategory::ATTENTION_QKV);
}

TEST_F(Test__WeightSlicer, CategorizeWeight_Wo)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());
    EXPECT_EQ(slicer.categorizeWeight("blk.0.attn_output.weight"),
              WeightSlicer::WeightCategory::ATTENTION_WO);
}

TEST_F(Test__WeightSlicer, CategorizeWeight_FFN)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    EXPECT_EQ(slicer.categorizeWeight("blk.0.ffn_gate.weight"),
              WeightSlicer::WeightCategory::FFN_GATE_UP);
    EXPECT_EQ(slicer.categorizeWeight("blk.0.ffn_up.weight"),
              WeightSlicer::WeightCategory::FFN_GATE_UP);
    EXPECT_EQ(slicer.categorizeWeight("blk.0.ffn_gate_up.weight"),
              WeightSlicer::WeightCategory::FFN_GATE_UP);
    EXPECT_EQ(slicer.categorizeWeight("blk.0.ffn_down.weight"),
              WeightSlicer::WeightCategory::FFN_DOWN);
}

TEST_F(Test__WeightSlicer, CategorizeWeight_LMHead)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());
    EXPECT_EQ(slicer.categorizeWeight("output.weight"),
              WeightSlicer::WeightCategory::LM_HEAD);
}

TEST_F(Test__WeightSlicer, CategorizeWeight_Replicate)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    EXPECT_EQ(slicer.categorizeWeight("blk.0.attn_norm.weight"),
              WeightSlicer::WeightCategory::REPLICATE);
    EXPECT_EQ(slicer.categorizeWeight("token_embd.weight"),
              WeightSlicer::WeightCategory::REPLICATE);
}

// =============================================================================
// Equal Column Slice (no TensorParallelConfig)
// =============================================================================

TEST_F(Test__WeightSlicer, ColumnSlice_EqualSplit_2Way)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    auto s0 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 0, 2);
    auto s1 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 1, 2);

    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, Q_DIM / 2);
    EXPECT_EQ(s1.start, Q_DIM / 2);
    EXPECT_EQ(s1.count, Q_DIM / 2);
}

TEST_F(Test__WeightSlicer, ColumnSlice_EqualSplit_4Way)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    for (int rank = 0; rank < 4; rank++)
    {
        auto s = slicer.computeColumnSlice("blk.0.ffn_gate.weight", D_FF, rank, 4);
        EXPECT_EQ(s.start, rank * (D_FF / 4));
        EXPECT_EQ(s.count, D_FF / 4);
    }
}

// =============================================================================
// Equal Row Slice (no TensorParallelConfig)
// =============================================================================

TEST_F(Test__WeightSlicer, RowSlice_EqualSplit_2Way)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    auto s0 = slicer.computeRowSlice("blk.0.attn_output.weight", Q_DIM, 0, 2);
    auto s1 = slicer.computeRowSlice("blk.0.attn_output.weight", Q_DIM, 1, 2);

    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, Q_DIM / 2);
    EXPECT_EQ(s1.start, Q_DIM / 2);
    EXPECT_EQ(s1.count, Q_DIM / 2);
}

// =============================================================================
// FusedQKV Sub-Block Slicing (GQA layout)
// =============================================================================

TEST_F(Test__WeightSlicer, FusedQKV_GQA_DetectsSubBlocks)
{
    WeightSlicer slicer(makeDimensions(), qwen35Config());

    auto result = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", FUSED_QKV, 0, 2);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->q_total, Q_DIM);
    EXPECT_EQ(result->k_total, KV_DIM);
    EXPECT_EQ(result->v_total, KV_DIM);
    EXPECT_FALSE(result->replicate_qk);
}

TEST_F(Test__WeightSlicer, FusedQKV_GQA_SlicesCorrectly_2Way)
{
    WeightSlicer slicer(makeDimensions(), qwen35Config());

    auto r0 = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", FUSED_QKV, 0, 2);
    auto r1 = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", FUSED_QKV, 1, 2);

    ASSERT_TRUE(r0.has_value());
    ASSERT_TRUE(r1.has_value());

    // Q sub-block: 896 / 2 = 448 per rank
    EXPECT_EQ(r0->q.start, 0u);
    EXPECT_EQ(r0->q.count, Q_DIM / 2);
    EXPECT_EQ(r1->q.start, Q_DIM / 2);
    EXPECT_EQ(r1->q.count, Q_DIM / 2);

    // K sub-block: 128 / 2 = 64 per rank
    EXPECT_EQ(r0->k.start, 0u);
    EXPECT_EQ(r0->k.count, KV_DIM / 2);
    EXPECT_EQ(r1->k.start, KV_DIM / 2);
    EXPECT_EQ(r1->k.count, KV_DIM / 2);

    // V sub-block: same as K
    EXPECT_EQ(r0->v.start, 0u);
    EXPECT_EQ(r0->v.count, KV_DIM / 2);
    EXPECT_EQ(r1->v.start, KV_DIM / 2);
    EXPECT_EQ(r1->v.count, KV_DIM / 2);
}

TEST_F(Test__WeightSlicer, FusedQKV_NonFusedWeight_ReturnsNullopt)
{
    WeightSlicer slicer(makeDimensions(), qwen35Config());

    auto result = slicer.computeFusedQKVSlice("blk.0.attn_q.weight", Q_DIM, 0, 2);
    EXPECT_FALSE(result.has_value());
}

// =============================================================================
// FusedQKV Sub-Block Slicing (GDN asymmetric layout)
// =============================================================================

TEST_F(Test__WeightSlicer, FusedQKV_GDN_Asymmetric_DetectsSubBlocks)
{
    // Qwen3.5 4B: n_k=4, n_v=8, d_state=4
    // Q=16, K=16, V=32, total=64
    auto dims = makeGDNDimensions(4, 8, 4);
    WeightSlicer slicer(dims, qwen35Config());

    auto result = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", 64, 0, 2);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->q_total, 16u);
    EXPECT_EQ(result->k_total, 16u);
    EXPECT_EQ(result->v_total, 32u);
    EXPECT_TRUE(result->replicate_qk); // GDN with n_v > n_k
}

TEST_F(Test__WeightSlicer, FusedQKV_GDN_ReplicatesQK_ShardsV)
{
    auto dims = makeGDNDimensions(4, 8, 4);
    WeightSlicer slicer(dims, qwen35Config());

    auto r0 = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", 64, 0, 2);
    auto r1 = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", 64, 1, 2);

    ASSERT_TRUE(r0.has_value());
    ASSERT_TRUE(r1.has_value());

    // Q replicated: both ranks get full Q
    EXPECT_EQ(r0->q.start, 0u);
    EXPECT_EQ(r0->q.count, 16u);
    EXPECT_EQ(r1->q.start, 0u);
    EXPECT_EQ(r1->q.count, 16u);

    // K replicated: both ranks get full K
    EXPECT_EQ(r0->k.start, 0u);
    EXPECT_EQ(r0->k.count, 16u);
    EXPECT_EQ(r1->k.start, 0u);
    EXPECT_EQ(r1->k.count, 16u);

    // V sharded: 32 / 2 = 16 per rank
    EXPECT_EQ(r0->v.start, 0u);
    EXPECT_EQ(r0->v.count, 16u);
    EXPECT_EQ(r1->v.start, 16u);
    EXPECT_EQ(r1->v.count, 16u);
}

TEST_F(Test__WeightSlicer, FusedQKV_GDN_EqualHeads_NoReplication)
{
    // When n_k == n_v, no replication needed
    auto dims = makeGDNDimensions(4, 4, 4);
    WeightSlicer slicer(dims, qwen35Config());

    // Total = 2*16 + 16 = 48
    auto result = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", 48, 0, 2);
    ASSERT_TRUE(result.has_value());

    EXPECT_FALSE(result->replicate_qk); // No replication when n_k == n_v
}

TEST_F(Test__WeightSlicer, FusedQKV_GDN_IndivisibleV_Throws)
{
    // n_k=4, n_v=6, d_state=4 → Q=16, K=16, V=24
    // n_v > n_k → replicate_qk=true (only V checked for divisibility)
    // V=24 not divisible by 5 → throws
    auto dims = makeGDNDimensions(4, 6, 4);
    WeightSlicer slicer(dims, qwen35Config());

    // Total = 2*16 + 24 = 56
    EXPECT_THROW(
        slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", 56, 0, 5),
        std::invalid_argument);
}

// =============================================================================
// Proportional Slicing (with TensorParallelConfig)
// =============================================================================

TEST_F(Test__WeightSlicer, ProportionalColumnSlice_QProjection)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    auto s0 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 0, 2);
    auto s1 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 1, 2);

    // Q: 14 heads / 2 = 7 heads per rank, 7 * 64 = 448 elements
    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, 7u * HEAD_DIM);
    EXPECT_EQ(s1.start, 7u * HEAD_DIM);
    EXPECT_EQ(s1.count, 7u * HEAD_DIM);
}

TEST_F(Test__WeightSlicer, ProportionalColumnSlice_KVProjection)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    auto s0 = slicer.computeColumnSlice("blk.0.attn_k.weight", KV_DIM, 0, 2);
    auto s1 = slicer.computeColumnSlice("blk.0.attn_k.weight", KV_DIM, 1, 2);

    // K: 2 KV heads / 2 = 1 KV head per rank, 1 * 64 = 64
    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, 1u * HEAD_DIM);
    EXPECT_EQ(s1.start, 1u * HEAD_DIM);
    EXPECT_EQ(s1.count, 1u * HEAD_DIM);
}

TEST_F(Test__WeightSlicer, ProportionalColumnSlice_FFN)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    auto s0 = slicer.computeColumnSlice("blk.0.ffn_gate.weight", D_FF, 0, 2);
    auto s1 = slicer.computeColumnSlice("blk.0.ffn_gate.weight", D_FF, 1, 2);

    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, D_FF / 2);
    EXPECT_EQ(s1.start, D_FF / 2);
    EXPECT_EQ(s1.count, D_FF / 2);
}

TEST_F(Test__WeightSlicer, ProportionalRowSlice_Wo)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    auto s0 = slicer.computeRowSlice("blk.0.attn_output.weight", Q_DIM, 0, 2);
    auto s1 = slicer.computeRowSlice("blk.0.attn_output.weight", Q_DIM, 1, 2);

    // Wo: split by heads
    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, 7u * HEAD_DIM);
    EXPECT_EQ(s1.start, 7u * HEAD_DIM);
    EXPECT_EQ(s1.count, 7u * HEAD_DIM);
}

TEST_F(Test__WeightSlicer, ProportionalRowSlice_FFNDown)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    auto s0 = slicer.computeRowSlice("blk.0.ffn_down.weight", D_FF, 0, 2);
    auto s1 = slicer.computeRowSlice("blk.0.ffn_down.weight", D_FF, 1, 2);

    EXPECT_EQ(s0.start, 0u);
    EXPECT_EQ(s0.count, D_FF / 2);
    EXPECT_EQ(s1.start, D_FF / 2);
    EXPECT_EQ(s1.count, D_FF / 2);
}

// =============================================================================
// SliceSpec for Device Assignment (LOCAL TP)
// =============================================================================

TEST_F(Test__WeightSlicer, SliceForAssignment_Heads)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    const auto &a0 = tp->forRank(0);
    auto s = slicer.computeSliceForAssignment("blk.0.attn_q.weight", Q_DIM, a0);

    EXPECT_EQ(s.start, 0u);
    EXPECT_EQ(s.count, 7u * HEAD_DIM);
}

TEST_F(Test__WeightSlicer, SliceForAssignment_KVHeads)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    const auto &a1 = tp->forRank(1);
    auto s = slicer.computeSliceForAssignment("blk.0.attn_k.weight", KV_DIM, a1);

    EXPECT_EQ(s.start, 1u * HEAD_DIM);
    EXPECT_EQ(s.count, 1u * HEAD_DIM);
}

TEST_F(Test__WeightSlicer, SliceForAssignment_FFNHidden)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    const auto &a0 = tp->forRank(0);
    auto s = slicer.computeSliceForAssignment("blk.0.ffn_gate.weight", D_FF, a0);

    EXPECT_EQ(s.start, 0u);
    EXPECT_EQ(s.count, D_FF / 2);
}

TEST_F(Test__WeightSlicer, SliceForAssignment_Vocab)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen2Config(), tp);

    const auto &a0 = tp->forRank(0);
    auto s = slicer.computeSliceForAssignment("output.weight", VOCAB, a0);

    EXPECT_EQ(s.start, 0u);
    EXPECT_EQ(s.count, VOCAB / 2);
}

// =============================================================================
// FusedQKV for Device Assignment (LOCAL TP)
// =============================================================================

TEST_F(Test__WeightSlicer, FusedQKVForAssignment_2Way)
{
    auto tp = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));

    WeightSlicer slicer(makeDimensions(), qwen35Config(), tp);

    auto r0 = slicer.computeFusedQKVSliceForAssignment(
        "blk.0.attn_qkv.weight", FUSED_QKV, tp->forRank(0));
    auto r1 = slicer.computeFusedQKVSliceForAssignment(
        "blk.0.attn_qkv.weight", FUSED_QKV, tp->forRank(1));

    ASSERT_TRUE(r0.has_value());
    ASSERT_TRUE(r1.has_value());

    EXPECT_EQ(r0->q.count, Q_DIM / 2);
    EXPECT_EQ(r1->q.count, Q_DIM / 2);
    EXPECT_EQ(r0->k.count, KV_DIM / 2);
    EXPECT_EQ(r1->k.count, KV_DIM / 2);
    EXPECT_EQ(r0->v.count, KV_DIM / 2);
    EXPECT_EQ(r1->v.count, KV_DIM / 2);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(Test__WeightSlicer, ColumnSlice_SingleRank_FullTensor)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    auto s = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 0, 1);
    EXPECT_EQ(s.start, 0u);
    EXPECT_EQ(s.count, static_cast<size_t>(Q_DIM));
}

TEST_F(Test__WeightSlicer, RowSlice_SingleRank_FullTensor)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    auto s = slicer.computeRowSlice("blk.0.ffn_down.weight", D_FF, 0, 1);
    EXPECT_EQ(s.start, 0u);
    EXPECT_EQ(s.count, static_cast<size_t>(D_FF));
}

TEST_F(Test__WeightSlicer, SliceSpec_End_And_Empty)
{
    SliceSpec s{10, 20};
    EXPECT_EQ(s.end(), 30u);
    EXPECT_FALSE(s.empty());

    SliceSpec empty{5, 0};
    EXPECT_EQ(empty.end(), 5u);
    EXPECT_TRUE(empty.empty());
}

TEST_F(Test__WeightSlicer, FusedQKV_3EqualBlocks_Fallback)
{
    // Without dimension info: total_rows=300 (divisible by 3), Q=K=V=100
    ModelDimensions no_dims{};
    WeightSlicer slicer(no_dims, qwen35Config());

    auto result = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", 300, 0, 2);
    ASSERT_TRUE(result.has_value());

    EXPECT_EQ(result->q_total, 100u);
    EXPECT_EQ(result->k_total, 100u);
    EXPECT_EQ(result->v_total, 100u);
    EXPECT_FALSE(result->replicate_qk);
}

// =============================================================================
// Multi-Degree TP: Column-Parallel (Q, K, V, FFN gate/up, LM head)
// =============================================================================

TEST_F(Test__WeightSlicer, ColumnSlice_AllDegrees_Q_HeadAligned)
{
    // Llama-3 style: 32 heads, 8 KV heads, head_dim=128
    // TP degrees 2, 4, 8 are valid. TP=16/32 give per-device Q heads < KV heads
    // which is invalid for GQA (n_rep = Q_per_device / KV_per_device < 1).
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD; // 4096

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, Q, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        size_t total_count = 0;
        size_t prev_end = 0;

        for (int rank = 0; rank < tp; rank++)
        {
            auto s = slicer.computeColumnSlice("blk.0.attn_q.weight", Q, rank, tp);
            EXPECT_EQ(s.start, prev_end) << "TP=" << tp << " rank=" << rank;
            total_count += s.count;
            prev_end = s.end();
        }
        EXPECT_EQ(total_count, static_cast<size_t>(Q)) << "TP=" << tp << " coverage";
    }
}

TEST_F(Test__WeightSlicer, ColumnSlice_AllDegrees_KV_HeadAligned)
{
    // Llama-3: 32 heads, 8 KV heads, head_dim=128
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int KV_DIM_TOTAL = KV * HD; // 1024

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, D_FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        size_t total = 0;
        int kv_per_rank = KV / tp;

        for (int rank = 0; rank < tp; rank++)
        {
            auto s = slicer.computeColumnSlice("blk.0.attn_k.weight", KV_DIM_TOTAL, rank, tp);
            EXPECT_EQ(s.count, static_cast<size_t>(kv_per_rank * HD))
                << "TP=" << tp << " rank=" << rank;
            total += s.count;
        }
        EXPECT_EQ(total, static_cast<size_t>(KV_DIM_TOTAL)) << "TP=" << tp;
    }
}

TEST_F(Test__WeightSlicer, ColumnSlice_AllDegrees_FFN)
{
    // FFN d_ff=11008 (Llama-3 7B). Tests 2, 4, 8 way.
    constexpr int FF = 11008;
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        size_t total = 0;
        for (int rank = 0; rank < tp; rank++)
        {
            auto s = slicer.computeColumnSlice("blk.0.ffn_gate.weight", FF, rank, tp);
            total += s.count;
        }
        EXPECT_EQ(total, static_cast<size_t>(FF)) << "TP=" << tp;

        // Also test ffn_up (same dimension)
        total = 0;
        for (int rank = 0; rank < tp; rank++)
        {
            auto s = slicer.computeColumnSlice("blk.0.ffn_up.weight", FF, rank, tp);
            total += s.count;
        }
        EXPECT_EQ(total, static_cast<size_t>(FF)) << "TP=" << tp << " ffn_up";
    }
}

TEST_F(Test__WeightSlicer, ColumnSlice_AllDegrees_LMHead)
{
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, D_FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        size_t total = 0;
        for (int rank = 0; rank < tp; rank++)
        {
            auto s = slicer.computeColumnSlice("output.weight", VOCAB, rank, tp);
            total += s.count;
        }
        EXPECT_EQ(total, static_cast<size_t>(VOCAB)) << "TP=" << tp;
    }
}

// =============================================================================
// Multi-Degree TP: Row-Parallel (Wo, FFN down)
// =============================================================================

TEST_F(Test__WeightSlicer, RowSlice_AllDegrees_Wo)
{
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, D_FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        size_t total = 0;
        int heads_per_rank = HEADS / tp;

        for (int rank = 0; rank < tp; rank++)
        {
            auto s = slicer.computeRowSlice("blk.0.attn_output.weight", Q, rank, tp);
            EXPECT_EQ(s.count, static_cast<size_t>(heads_per_rank * HD))
                << "TP=" << tp << " rank=" << rank;
            total += s.count;
        }
        EXPECT_EQ(total, static_cast<size_t>(Q)) << "TP=" << tp;
    }
}

TEST_F(Test__WeightSlicer, RowSlice_AllDegrees_FFNDown)
{
    constexpr int FF = 11008;
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        size_t total = 0;
        for (int rank = 0; rank < tp; rank++)
        {
            auto s = slicer.computeRowSlice("blk.0.ffn_down.weight", FF, rank, tp);
            total += s.count;
        }
        EXPECT_EQ(total, static_cast<size_t>(FF)) << "TP=" << tp;
    }
}

// =============================================================================
// Multi-Degree FusedQKV: GQA layout with various TP degrees
// =============================================================================

TEST_F(Test__WeightSlicer, FusedQKV_GQA_AllDegrees_Llama3)
{
    // Llama-3 8B: 32 Q heads, 8 KV heads, head_dim=128
    // FusedQKV total = 32*128 + 2*8*128 = 4096 + 2048 = 6144
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr size_t Q_TOTAL = HEADS * HD;                 // 4096
    constexpr size_t KV_TOTAL = KV * HD;                   // 1024
    constexpr size_t FUSED_TOTAL = Q_TOTAL + 2 * KV_TOTAL; // 6144

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        WeightSlicer slicer(dims, qwen35Config());

        size_t q_total = 0, k_total = 0, v_total = 0;

        for (int rank = 0; rank < tp; rank++)
        {
            auto r = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", FUSED_TOTAL, rank, tp);
            ASSERT_TRUE(r.has_value()) << "TP=" << tp << " rank=" << rank;

            EXPECT_EQ(r->q_total, Q_TOTAL);
            EXPECT_EQ(r->k_total, KV_TOTAL);
            EXPECT_EQ(r->v_total, KV_TOTAL);
            EXPECT_FALSE(r->replicate_qk);

            EXPECT_EQ(r->q.count, Q_TOTAL / tp) << "TP=" << tp << " rank=" << rank;
            EXPECT_EQ(r->k.count, KV_TOTAL / tp) << "TP=" << tp << " rank=" << rank;
            EXPECT_EQ(r->v.count, KV_TOTAL / tp) << "TP=" << tp << " rank=" << rank;

            q_total += r->q.count;
            k_total += r->k.count;
            v_total += r->v.count;
        }

        EXPECT_EQ(q_total, Q_TOTAL) << "TP=" << tp << " Q coverage";
        EXPECT_EQ(k_total, KV_TOTAL) << "TP=" << tp << " K coverage";
        EXPECT_EQ(v_total, KV_TOTAL) << "TP=" << tp << " V coverage";
    }
}

TEST_F(Test__WeightSlicer, FusedQKV_GDN_AllDegrees_Qwen35_4B)
{
    // Qwen3.5-4B GDN: n_k=16, n_v=32, d_state=128
    // Q=2048, K=2048, V=4096, total=8192
    // n_v > n_k → replicate Q/K, shard V only
    constexpr int NK = 16;
    constexpr int NV = 32;
    constexpr int DS = 128;
    constexpr size_t Q_ROWS = NK * DS;                 // 2048
    constexpr size_t K_ROWS = NK * DS;                 // 2048
    constexpr size_t V_ROWS = NV * DS;                 // 4096
    constexpr size_t TOTAL = Q_ROWS + K_ROWS + V_ROWS; // 8192

    auto dims = makeGDNDimensions(NK, NV, DS);

    for (int tp : {2, 4, 8, 16, 32})
    {
        WeightSlicer slicer(dims, qwen35Config());

        size_t v_total = 0;

        for (int rank = 0; rank < tp; rank++)
        {
            auto r = slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", TOTAL, rank, tp);
            ASSERT_TRUE(r.has_value()) << "TP=" << tp << " rank=" << rank;

            EXPECT_TRUE(r->replicate_qk) << "TP=" << tp;

            // Q and K replicated: all ranks get full copy
            EXPECT_EQ(r->q.start, 0u) << "TP=" << tp << " rank=" << rank;
            EXPECT_EQ(r->q.count, Q_ROWS) << "TP=" << tp << " rank=" << rank;
            EXPECT_EQ(r->k.start, 0u) << "TP=" << tp << " rank=" << rank;
            EXPECT_EQ(r->k.count, K_ROWS) << "TP=" << tp << " rank=" << rank;

            // V sharded
            EXPECT_EQ(r->v.count, V_ROWS / tp) << "TP=" << tp << " rank=" << rank;
            v_total += r->v.count;
        }

        EXPECT_EQ(v_total, V_ROWS) << "TP=" << tp << " V coverage";
    }
}

TEST_F(Test__WeightSlicer, Qwen35GDNValueHeadWeights_4B_TP2_AreDStateAligned)
{
    // Regression guard for Qwen3.5-4B dense GDN: FA/Q attention uses 36 heads,
    // but GDN value projections use 32 value heads. These weights must be sliced
    // proportionally into value-head space, not by an integer 4096/36 head width.
    constexpr int FA_HEADS = 36;
    constexpr int GDN_K_HEADS = 16;
    constexpr int GDN_V_HEADS = 32;
    constexpr int D_STATE = 128;
    constexpr size_t GDN_VALUE_ROWS = GDN_V_HEADS * D_STATE;

    ModelDimensions dims{
        .n_heads = FA_HEADS,
        .n_kv_heads = 4,
        .head_dim = D_STATE,
        .gdn_n_k_heads = GDN_K_HEADS,
        .gdn_n_v_heads = GDN_V_HEADS,
        .gdn_d_state = D_STATE};
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, FA_HEADS, 4, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen35Config(), tp_cfg);

    const auto &rank1 = tp_cfg->forRank(1);
    ASSERT_EQ(rank1.head_start, 18);

    auto gate = slicer.computeSliceForAssignment(
        "blk.0.attn_gate.weight", GDN_VALUE_ROWS, rank1);
    auto ssm_out = slicer.computeSliceForAssignment(
        "blk.0.ssm_out.weight", GDN_VALUE_ROWS, rank1);

    EXPECT_EQ(gate.start, 16u * D_STATE);
    EXPECT_EQ(gate.count, 16u * D_STATE);
    EXPECT_EQ(gate.start % D_STATE, 0u);

    EXPECT_EQ(ssm_out.start, 16u * D_STATE);
    EXPECT_EQ(ssm_out.count, 16u * D_STATE);
    EXPECT_EQ(ssm_out.start % D_STATE, 0u);

    const size_t stale_heads_start = static_cast<size_t>(rank1.head_start) *
                                     (GDN_VALUE_ROWS / static_cast<size_t>(FA_HEADS));
    EXPECT_NE(gate.start, stale_heads_start);
    EXPECT_NE(ssm_out.start, stale_heads_start);
}

// =============================================================================
// TP Degree == n_heads (one head per rank)
// =============================================================================

TEST_F(Test__WeightSlicer, ColumnSlice_TPEqualsHeads_OneHeadPerRank)
{
    // 8 Q heads, TP=8: each rank gets exactly 1 head
    // Use KV=1 (MHA) — GQA with KV>1 would make TP=HEADS invalid
    // when per-device Q heads < KV heads (n_rep < 1).
    constexpr int HEADS = 8;
    constexpr int KV = 1;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(HEADS, HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    for (int rank = 0; rank < HEADS; rank++)
    {
        auto s = slicer.computeColumnSlice("blk.0.attn_q.weight", Q, rank, HEADS);
        EXPECT_EQ(s.start, static_cast<size_t>(rank * HD));
        EXPECT_EQ(s.count, static_cast<size_t>(HD));
    }
}

TEST_F(Test__WeightSlicer, ColumnSlice_TPEqualsKVHeads)
{
    // 32 Q heads, 8 KV heads, TP=8: each rank gets 1 KV head
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int KV_DIM_TOTAL = KV * HD;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(KV, HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    for (int rank = 0; rank < KV; rank++)
    {
        auto s = slicer.computeColumnSlice("blk.0.attn_v.weight", KV_DIM_TOTAL, rank, KV);
        EXPECT_EQ(s.count, static_cast<size_t>(HD)) << "rank=" << rank;
    }
}

// =============================================================================
// GQA: Various head ratios with multi-degree TP
// =============================================================================

TEST_F(Test__WeightSlicer, GQA_HighRatio_64Q_4KV_TP4)
{
    // Extreme GQA: 64 Q heads, 4 KV heads, ratio 16:1
    constexpr int HEADS = 64;
    constexpr int KV = 4;
    constexpr int HD = 64;
    constexpr int Q = HEADS * HD;         // 4096
    constexpr int KV_DIM_TOTAL = KV * HD; // 256
    constexpr int TP = 4;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(TP, HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    // Q: 64/4 = 16 heads per rank
    for (int rank = 0; rank < TP; rank++)
    {
        auto sq = slicer.computeColumnSlice("blk.0.attn_q.weight", Q, rank, TP);
        EXPECT_EQ(sq.count, static_cast<size_t>(16 * HD)) << "rank=" << rank;

        auto sk = slicer.computeColumnSlice("blk.0.attn_k.weight", KV_DIM_TOTAL, rank, TP);
        EXPECT_EQ(sk.count, static_cast<size_t>(1 * HD)) << "rank=" << rank;

        // Wo: input-parallel by heads
        auto sw = slicer.computeRowSlice("blk.0.attn_output.weight", Q, rank, TP);
        EXPECT_EQ(sw.count, static_cast<size_t>(16 * HD)) << "rank=" << rank;
    }
}

TEST_F(Test__WeightSlicer, GQA_Qwen2_14Q_2KV_TP2_And_TP7)
{
    // Qwen2: 14 Q heads, 2 KV heads
    auto dims = makeDimensions(N_HEADS, N_KV_HEADS, HEAD_DIM);

    // TP=2: 7 Q heads, 1 KV head per rank
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(2, N_HEADS, N_KV_HEADS, D_FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        auto s0 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 0, 2);
        auto s1 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 1, 2);
        EXPECT_EQ(s0.count, 7u * HEAD_DIM);
        EXPECT_EQ(s1.count, 7u * HEAD_DIM);

        auto k0 = slicer.computeColumnSlice("blk.0.attn_k.weight", KV_DIM, 0, 2);
        auto k1 = slicer.computeColumnSlice("blk.0.attn_k.weight", KV_DIM, 1, 2);
        EXPECT_EQ(k0.count, 1u * HEAD_DIM);
        EXPECT_EQ(k1.count, 1u * HEAD_DIM);
    }

    // TP=7: 2 Q heads, can't split 2 KV heads by 7
    // Equal split falls back to simple division
    {
        WeightSlicer slicer(dims, qwen2Config());

        auto s0 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 0, 7);
        auto s6 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q_DIM, 6, 7);
        EXPECT_EQ(s0.count, Q_DIM / 7) << "rank 0";
        EXPECT_EQ(s6.count, Q_DIM - 6 * (Q_DIM / 7)) << "rank 6 (remainder)";
    }
}

// =============================================================================
// FusedQKV divisibility validation at various TP degrees
// =============================================================================

TEST_F(Test__WeightSlicer, FusedQKV_GQA_IndivisibleKV_Throws)
{
    // 32 Q heads, 8 KV heads, hd=128
    // FusedQKV = 4096 + 1024 + 1024 = 6144
    // TP=16: KV=1024/16=64 (ok), Q=4096/16=256 (ok)
    // TP=5:  Q=4096/5=819.2 → not evenly divisible → throws
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr size_t FUSED_TOTAL = HEADS * HD + 2 * KV * HD; // 6144

    auto dims = makeDimensions(HEADS, KV, HD);
    WeightSlicer slicer(dims, qwen35Config());

    // TP=16: should work (4096/16=256, 1024/16=64)
    EXPECT_NO_THROW(
        slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", FUSED_TOTAL, 0, 16));

    // TP=5: Q=4096 not divisible by 5 → throws
    EXPECT_THROW(
        slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", FUSED_TOTAL, 0, 5),
        std::invalid_argument);

    // TP=3: Q=4096 not divisible by 3 → throws
    EXPECT_THROW(
        slicer.computeFusedQKVSlice("blk.0.attn_qkv.weight", FUSED_TOTAL, 0, 3),
        std::invalid_argument);
}

// =============================================================================
// Proportional (weighted) TP splits
// =============================================================================

TEST_F(Test__WeightSlicer, ProportionalSlice_73_27_Q)
{
    // 73%/27% split on 32 Q heads → 23+9=32 heads
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD; // 4096

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::cuda(1)},
            {0.73f, 0.27f},
            HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    auto s0 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q, 0, 2);
    auto s1 = slicer.computeColumnSlice("blk.0.attn_q.weight", Q, 1, 2);

    // Heads must be integers; 73% of 32 = 23.36 → rounded to integer
    EXPECT_EQ(s0.count + s1.count, static_cast<size_t>(Q));
    EXPECT_GT(s0.count, s1.count) << "Rank 0 (73%) should get more";
    // Both must be multiples of head_dim
    EXPECT_EQ(s0.count % HD, 0u);
    EXPECT_EQ(s1.count % HD, 0u);
}

TEST_F(Test__WeightSlicer, ProportionalSlice_73_27_Wo)
{
    // Row-parallel Wo should follow same head split
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::cuda(1)},
            {0.73f, 0.27f},
            HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    auto s0 = slicer.computeRowSlice("blk.0.attn_output.weight", Q, 0, 2);
    auto s1 = slicer.computeRowSlice("blk.0.attn_output.weight", Q, 1, 2);

    EXPECT_EQ(s0.count + s1.count, static_cast<size_t>(Q));
    EXPECT_GT(s0.count, s1.count);
    EXPECT_EQ(s0.count % HD, 0u);
    EXPECT_EQ(s1.count % HD, 0u);
}

TEST_F(Test__WeightSlicer, ProportionalSlice_73_27_FFN)
{
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int FF = 11008;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::cuda(1)},
            {0.73f, 0.27f},
            HEADS, KV, FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    // Column-parallel: ffn_gate
    auto sg0 = slicer.computeColumnSlice("blk.0.ffn_gate.weight", FF, 0, 2);
    auto sg1 = slicer.computeColumnSlice("blk.0.ffn_gate.weight", FF, 1, 2);
    EXPECT_EQ(sg0.count + sg1.count, static_cast<size_t>(FF));
    EXPECT_GT(sg0.count, sg1.count);

    // Row-parallel: ffn_down
    auto sd0 = slicer.computeRowSlice("blk.0.ffn_down.weight", FF, 0, 2);
    auto sd1 = slicer.computeRowSlice("blk.0.ffn_down.weight", FF, 1, 2);
    EXPECT_EQ(sd0.count + sd1.count, static_cast<size_t>(FF));
    EXPECT_GT(sd0.count, sd1.count);
}

TEST_F(Test__WeightSlicer, ProportionalSlice_4Way_Equal)
{
    // 4-way with equal fractions should behave like equalSplit
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_prop = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::proportionalSplit(
            {DeviceId::cuda(0), DeviceId::cuda(1), DeviceId::cuda(2), DeviceId::cuda(3)},
            {0.25f, 0.25f, 0.25f, 0.25f},
            HEADS, KV, D_FF, VOCAB));
    auto tp_equal = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(4, HEADS, KV, D_FF, VOCAB));

    WeightSlicer slicer_prop(dims, qwen2Config(), tp_prop);
    WeightSlicer slicer_eq(dims, qwen2Config(), tp_equal);

    for (int rank = 0; rank < 4; rank++)
    {
        auto sp = slicer_prop.computeColumnSlice("blk.0.attn_q.weight", Q, rank, 4);
        auto se = slicer_eq.computeColumnSlice("blk.0.attn_q.weight", Q, rank, 4);
        EXPECT_EQ(sp.start, se.start) << "rank=" << rank;
        EXPECT_EQ(sp.count, se.count) << "rank=" << rank;
    }
}

// =============================================================================
// SliceForAssignment: Multi-degree with DeviceShardingAssignment
// =============================================================================

TEST_F(Test__WeightSlicer, SliceForAssignment_AllDegrees_AllTypes)
{
    // Test that computeSliceForAssignment covers all 4 dimension types
    // at TP=4 for a Llama-3-like model
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;
    constexpr int KV_TOTAL = KV * HD;
    constexpr int FF = 11008;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(4, HEADS, KV, FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    for (int rank = 0; rank < 4; rank++)
    {
        const auto &a = tp_cfg->forRank(rank);

        // Heads (Q projection)
        auto sh = slicer.computeSliceForAssignment("blk.0.attn_q.weight", Q, a);
        EXPECT_EQ(sh.count, static_cast<size_t>(8 * HD)) << "rank=" << rank;

        // KVHeads
        auto skv = slicer.computeSliceForAssignment("blk.0.attn_k.weight", KV_TOTAL, a);
        EXPECT_EQ(skv.count, static_cast<size_t>(2 * HD)) << "rank=" << rank;

        // FFNHidden
        auto sf = slicer.computeSliceForAssignment("blk.0.ffn_gate.weight", FF, a);
        EXPECT_EQ(sf.count, static_cast<size_t>(a.d_ff_count)) << "rank=" << rank;

        // Vocab
        auto sv = slicer.computeSliceForAssignment("output.weight", VOCAB, a);
        EXPECT_EQ(sv.count, static_cast<size_t>(a.vocab_count)) << "rank=" << rank;
    }
}

// =============================================================================
// Replicated weights are NOT sliced
// =============================================================================

TEST_F(Test__WeightSlicer, ReplicatedWeights_NotSliced_AnyDegree)
{
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, D_FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        // Norms: replicated — proportional row/col slice defaults to full
        auto cat = slicer.categorizeWeight("blk.0.attn_norm.weight");
        EXPECT_EQ(cat, WeightSlicer::WeightCategory::REPLICATE);

        cat = slicer.categorizeWeight("token_embd.weight");
        EXPECT_EQ(cat, WeightSlicer::WeightCategory::REPLICATE);
    }
}

// =============================================================================
// Remainder handling at non-even TP degrees
// =============================================================================

TEST_F(Test__WeightSlicer, ColumnSlice_EqualSplit_RemainderGoesToLastRank)
{
    // 15 rows, 4 ranks: 3+3+3+6 → last rank gets remainder
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    size_t total = 0;
    for (int rank = 0; rank < 4; rank++)
    {
        auto s = slicer.computeColumnSlice("blk.0.attn_q.weight", 15, rank, 4);
        total += s.count;
        if (rank < 3)
        {
            EXPECT_EQ(s.count, 3u) << "rank=" << rank;
        }
        else
        {
            EXPECT_EQ(s.count, 6u) << "rank=3 (remainder)";
        }
    }
    EXPECT_EQ(total, 15u);
}

TEST_F(Test__WeightSlicer, RowSlice_EqualSplit_RemainderGoesToLastRank)
{
    WeightSlicer slicer(makeDimensions(), qwen2Config());

    size_t total = 0;
    for (int rank = 0; rank < 4; rank++)
    {
        auto s = slicer.computeRowSlice("blk.0.ffn_down.weight", 17, rank, 4);
        total += s.count;
    }
    EXPECT_EQ(total, 17u);
}

// =============================================================================
// Contiguous coverage: all ranks tile the full range with no gaps or overlaps
// =============================================================================

TEST_F(Test__WeightSlicer, ContiguousCoverage_Column_8Way)
{
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(8, HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    // Verify contiguous tiling for Q projection
    size_t prev_end = 0;
    for (int rank = 0; rank < 8; rank++)
    {
        auto s = slicer.computeColumnSlice("blk.0.attn_q.weight", Q, rank, 8);
        EXPECT_EQ(s.start, prev_end) << "Gap at rank=" << rank;
        prev_end = s.end();
    }
    EXPECT_EQ(prev_end, static_cast<size_t>(Q)) << "Should cover full range";
}

TEST_F(Test__WeightSlicer, ContiguousCoverage_Row_8Way)
{
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(8, HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    // Verify contiguous tiling for Wo (input-parallel)
    size_t prev_end = 0;
    for (int rank = 0; rank < 8; rank++)
    {
        auto s = slicer.computeRowSlice("blk.0.attn_output.weight", Q, rank, 8);
        EXPECT_EQ(s.start, prev_end) << "Gap at rank=" << rank;
        prev_end = s.end();
    }
    EXPECT_EQ(prev_end, static_cast<size_t>(Q));
}

// =============================================================================
// Multi-layer consistency: same slice for all layers
// =============================================================================

TEST_F(Test__WeightSlicer, MultiLayer_SameSlice_AllLayers)
{
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD;

    auto dims = makeDimensions(HEADS, KV, HD);
    auto tp_cfg = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(4, HEADS, KV, D_FF, VOCAB));
    WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

    // All layers should produce identical slices for rank 2
    auto ref = slicer.computeColumnSlice("blk.0.attn_q.weight", Q, 2, 4);

    for (int layer = 1; layer < 32; layer++)
    {
        std::string name = "blk." + std::to_string(layer) + ".attn_q.weight";
        auto s = slicer.computeColumnSlice(name, Q, 2, 4);
        EXPECT_EQ(s.start, ref.start) << "Layer " << layer;
        EXPECT_EQ(s.count, ref.count) << "Layer " << layer;
    }
}

// =============================================================================
// Wo vs FFN_down disambiguation at all TP degrees
// =============================================================================

TEST_F(Test__WeightSlicer, WoVsFFNDown_DifferentDimensions_AllDegrees)
{
    // Critical: both are InputParallel but Wo slices by heads, FFN_down by d_ff
    constexpr int HEADS = 32;
    constexpr int KV = 8;
    constexpr int HD = 128;
    constexpr int Q = HEADS * HD; // 4096
    constexpr int FF = 11008;

    auto dims = makeDimensions(HEADS, KV, HD);

    for (int tp : {2, 4, 8})
    {
        auto tp_cfg = std::make_shared<TensorParallelConfig>(
            TensorParallelConfig::equalSplit(tp, HEADS, KV, FF, VOCAB));
        WeightSlicer slicer(dims, qwen2Config(), tp_cfg);

        auto wo = slicer.computeRowSlice("blk.0.attn_output.weight", Q, 0, tp);
        auto fd = slicer.computeRowSlice("blk.0.ffn_down.weight", FF, 0, tp);

        // Different total sizes → different slice sizes
        size_t wo_expected = static_cast<size_t>((HEADS / tp) * HD);
        size_t fd_expected = static_cast<size_t>(FF / tp);

        EXPECT_EQ(wo.count, wo_expected) << "Wo TP=" << tp;
        EXPECT_EQ(fd.count, fd_expected) << "FFN_down TP=" << tp;
        EXPECT_NE(wo.count, fd.count) << "Wo and FFN_down must differ at TP=" << tp;
    }
}
