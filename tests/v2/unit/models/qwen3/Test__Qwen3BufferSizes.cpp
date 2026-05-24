/**
 * @file Test__Qwen3BufferSizes.cpp
 * @brief Lock-in tests for Qwen3 schema buffer sizes
 *
 * Validates that resolved buffer shapes match expected dimensions
 * for both layer buffers and model buffers. Catches regressions
 * in schema formulas that could cause excessive memory usage.
 */

#include <gtest/gtest.h>
#include "models/qwen3/Qwen3Schema.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/local_execution/graph/GraphSchema.h"

using namespace llaminar2;

// ============================================================================
// Helper
// ============================================================================

static const BufferDescriptor *findBuf(
    const StageBufferRequirements &reqs, const std::string &name)
{
    for (const auto &b : reqs.buffers)
        if (b.name == name)
            return &b;
    return nullptr;
}

// ============================================================================
// Layer Buffer Size Lock-in (Qwen3-1.7B-like dimensions)
// ============================================================================

TEST(Test__Qwen3BufferSizes, LayerBuffers_ExactShapes)
{
    Qwen3SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Qwen3-1.7B-like: d=2048, heads=16, kv_heads=4, head_dim=128, d_ff=8960
    GraphResolverConfig config{};
    config.d_model = 2048;
    config.n_heads = 16;
    config.n_kv_heads = 4;
    config.head_dim = 128;
    config.seq_len = 512;
    config.batch_size = 1;
    config.local_n_heads = 16;
    config.local_n_kv_heads = 4;
    config.local_d_ff = 8960;

    auto reqs = BufferAllocator::resolveLayerBuffers(schema, config);

    // Qwen3 has 9 layer buffers, including the stable LM-head input row.
    EXPECT_EQ(reqs.buffers.size(), 9u) << "Expected 9 layer buffers";

    // normalized: [512, 2048]
    auto *normalized = findBuf(reqs, "normalized");
    ASSERT_NE(normalized, nullptr);
    ASSERT_EQ(normalized->shape.size(), 2u);
    EXPECT_EQ(normalized->shape[0], 512u);
    EXPECT_EQ(normalized->shape[1], 2048u);

    // Q: [512, local_qkv_dim] = [512, 16*128=2048]
    auto *Q = findBuf(reqs, "Q");
    ASSERT_NE(Q, nullptr);
    ASSERT_EQ(Q->shape.size(), 2u);
    EXPECT_EQ(Q->shape[0], 512u);
    EXPECT_EQ(Q->shape[1], 2048u);

    // K: [512, local_kv_dim] = [512, 4*128=512]
    auto *K = findBuf(reqs, "K");
    ASSERT_NE(K, nullptr);
    ASSERT_EQ(K->shape.size(), 2u);
    EXPECT_EQ(K->shape[0], 512u);
    EXPECT_EQ(K->shape[1], 512u);

    // V: [512, local_kv_dim] = [512, 512]
    auto *V = findBuf(reqs, "V");
    ASSERT_NE(V, nullptr);
    ASSERT_EQ(V->shape.size(), 2u);
    EXPECT_EQ(V->shape[0], 512u);
    EXPECT_EQ(V->shape[1], 512u);

    // attn_output: [512, local_qkv_dim] = [512, 2048]
    auto *attn_output = findBuf(reqs, "attn_output");
    ASSERT_NE(attn_output, nullptr);
    ASSERT_EQ(attn_output->shape.size(), 2u);
    EXPECT_EQ(attn_output->shape[0], 512u);
    EXPECT_EQ(attn_output->shape[1], 2048u);

    // attn_proj: [512, d_model] = [512, 2048]
    auto *attn_proj = findBuf(reqs, "attn_proj");
    ASSERT_NE(attn_proj, nullptr);
    ASSERT_EQ(attn_proj->shape.size(), 2u);
    EXPECT_EQ(attn_proj->shape[0], 512u);
    EXPECT_EQ(attn_proj->shape[1], 2048u);

    // gate: [512, local_d_ff] = [512, 8960]
    auto *gate = findBuf(reqs, "gate");
    ASSERT_NE(gate, nullptr);
    ASSERT_EQ(gate->shape.size(), 2u);
    EXPECT_EQ(gate->shape[0], 512u);
    EXPECT_EQ(gate->shape[1], 8960u);

    // up: [512, local_d_ff] = [512, 8960]
    auto *up = findBuf(reqs, "up");
    ASSERT_NE(up, nullptr);
    ASSERT_EQ(up->shape.size(), 2u);
    EXPECT_EQ(up->shape[0], 512u);
    EXPECT_EQ(up->shape[1], 8960u);

    // lm_head_input_row: [1, d_model] = [1, 2048]
    auto *lm_head_input_row = findBuf(reqs, "lm_head_input_row");
    ASSERT_NE(lm_head_input_row, nullptr);
    ASSERT_EQ(lm_head_input_row->shape.size(), 2u);
    EXPECT_EQ(lm_head_input_row->shape[0], 1u);
    EXPECT_EQ(lm_head_input_row->shape[1], 2048u);
}

// ============================================================================
// Model Buffer Size Lock-in
// ============================================================================

TEST(Test__Qwen3BufferSizes, ModelBuffers_ExactShapes)
{
    Qwen3SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 2048;
    config.vocab_size = 151936;
    config.seq_len = 512;
    config.local_vocab = 151936;

    auto reqs = BufferAllocator::resolveModelBuffers(schema, config);

    EXPECT_EQ(reqs.buffers.size(), 3u) << "Should have 3 model buffers";

    // hidden: [seq_len, d_model] = [512, 2048]
    auto *hidden = findBuf(reqs, "hidden");
    ASSERT_NE(hidden, nullptr);
    ASSERT_EQ(hidden->shape.size(), 2u);
    EXPECT_EQ(hidden->shape[0], 512u);
    EXPECT_EQ(hidden->shape[1], 2048u);

    // logits: [1, vocab_size] — only 1 row (last-token logits only)
    auto *logits = findBuf(reqs, "logits");
    ASSERT_NE(logits, nullptr);
    ASSERT_EQ(logits->shape.size(), 2u);
    EXPECT_EQ(logits->shape[0], 1u);
    EXPECT_EQ(logits->shape[1], 151936u);

    // logits_local: [1, local_vocab]
    auto *logits_local = findBuf(reqs, "logits_local");
    ASSERT_NE(logits_local, nullptr);
    ASSERT_EQ(logits_local->shape.size(), 2u);
    EXPECT_EQ(logits_local->shape[0], 1u);
    EXPECT_EQ(logits_local->shape[1], 151936u);
}

TEST(Test__Qwen3BufferSizes, ModelBuffers_TP2)
{
    Qwen3SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 2048;
    config.vocab_size = 151936;
    config.seq_len = 512;
    config.local_vocab = 75968; // TP=2

    auto reqs = BufferAllocator::resolveModelBuffers(schema, config);

    // logits: [1, vocab_size] — full vocab even under TP
    auto *logits = findBuf(reqs, "logits");
    ASSERT_NE(logits, nullptr);
    EXPECT_EQ(logits->shape[0], 1u);
    EXPECT_EQ(logits->shape[1], 151936u);

    // logits_local: [1, local_vocab] — half vocab under TP=2
    auto *logits_local = findBuf(reqs, "logits_local");
    ASSERT_NE(logits_local, nullptr);
    EXPECT_EQ(logits_local->shape[0], 1u);
    EXPECT_EQ(logits_local->shape[1], 75968u);
}

// ============================================================================
// Layer Buffer Size Lock-in with TP=2
// ============================================================================

TEST(Test__Qwen3BufferSizes, LayerBuffers_TP2)
{
    Qwen3SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // TP=2: heads split in half
    GraphResolverConfig config{};
    config.d_model = 2048;
    config.n_heads = 16;
    config.n_kv_heads = 4;
    config.head_dim = 128;
    config.seq_len = 512;
    config.batch_size = 1;
    config.local_n_heads = 8;    // 16/2
    config.local_n_kv_heads = 2; // 4/2
    config.local_d_ff = 4480;    // 8960/2

    auto reqs = BufferAllocator::resolveLayerBuffers(schema, config);

    // Q: [512, 8*128=1024]
    auto *Q = findBuf(reqs, "Q");
    ASSERT_NE(Q, nullptr);
    EXPECT_EQ(Q->shape[1], 1024u);

    // K: [512, 2*128=256]
    auto *K = findBuf(reqs, "K");
    ASSERT_NE(K, nullptr);
    EXPECT_EQ(K->shape[1], 256u);

    // gate/up: [512, 4480]
    auto *gate = findBuf(reqs, "gate");
    ASSERT_NE(gate, nullptr);
    EXPECT_EQ(gate->shape[1], 4480u);
}
