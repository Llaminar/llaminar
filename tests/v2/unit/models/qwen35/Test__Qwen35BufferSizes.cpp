/**
 * @file Test__Qwen35BufferSizes.cpp
 * @brief Lock-in tests for Qwen3.5 schema buffer sizes
 *
 * Validates that resolved buffer shapes match expected dimensions
 * for both layer buffers and model buffers. Qwen3.5 has additional
 * GDN-specific and FA-specific buffers beyond the standard Qwen2/3
 * layout, plus workspace buffers for attention computation.
 */

#include <gtest/gtest.h>
#include "models/qwen35/Qwen35Schema.h"
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
// Layer Buffer Size Lock-in (Qwen3.5-4B dimensions, non-TP)
// ============================================================================

TEST(Test__Qwen35BufferSizes, LayerBuffers_ExactShapes)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Qwen3.5-4B: d=2560, heads=16, kv_heads=4, head_dim=256, d_ff=9216
    // GDN: group_count=16, time_step_rank=32, state_size=128
    // inner_size = 4096 (or computed as n_v_heads * state_size = 32 * 128 = 4096)
    GraphResolverConfig config{};
    config.d_model = 2560;
    config.n_heads = 16;
    config.n_kv_heads = 4;
    config.head_dim = 256;
    config.vocab_size = 248320;
    config.seq_len = 4096;
    config.batch_size = 1;
    config.local_n_heads = 16;
    config.local_n_kv_heads = 4;
    config.local_d_ff = 9216;
    config.local_vocab = 248320;

    // GDN custom formulas (as computed by Qwen35Graph::getResolverConfig)
    // n_k_heads=16, d_k=128, key_dim=16*128=2048
    // gdn_inner = 32*128 = 4096 (n_v_heads * state_size)
    // gdn_qkv_dim = 2*key_dim + gdn_inner = 2*2048 + 4096 = 8192
    // fa_q_full_dim = 16 * 256 * 2 = 8192
    // attn_output_dim = max(16*256, 4096) = max(4096, 4096) = 4096
    config.custom_formulas["gdn_inner_size"] = 4096;
    config.custom_formulas["gdn_qkv_dim"] = 8192;
    config.custom_formulas["gdn_time_step_rank"] = 32;
    config.custom_formulas["fa_q_full_dim"] = 8192;
    config.custom_formulas["attn_output_dim"] = 4096;

    auto reqs = BufferAllocator::resolveLayerBuffers(schema, config);

    // Qwen3.5 has 20 main layer buffers, 17 MTP verifier sidecar buffers,
    // and the compact LM-head verifier row scratch used by row-indexed MTP.
    EXPECT_EQ(reqs.buffers.size(), 38u) << "Expected 38 layer buffers";

    // ── Shared buffers ──

    // normalized: [seq_len, d_model] = [4096, 2560]
    auto *normalized = findBuf(reqs, "normalized");
    ASSERT_NE(normalized, nullptr);
    ASSERT_EQ(normalized->shape.size(), 2u);
    EXPECT_EQ(normalized->shape[0], 4096u);
    EXPECT_EQ(normalized->shape[1], 2560u);

    // residual: [seq_len, d_model] = [4096, 2560]
    auto *residual = findBuf(reqs, "residual");
    ASSERT_NE(residual, nullptr);
    ASSERT_EQ(residual->shape.size(), 2u);
    EXPECT_EQ(residual->shape[0], 4096u);
    EXPECT_EQ(residual->shape[1], 2560u);

    // attn_output: [seq_len, attn_output_dim] = [4096, 4096]
    auto *attn_output = findBuf(reqs, "attn_output");
    ASSERT_NE(attn_output, nullptr);
    ASSERT_EQ(attn_output->shape.size(), 2u);
    EXPECT_EQ(attn_output->shape[0], 4096u);
    EXPECT_EQ(attn_output->shape[1], 4096u);

    // attn_proj: [seq_len, d_model] = [4096, 2560]
    auto *attn_proj = findBuf(reqs, "attn_proj");
    ASSERT_NE(attn_proj, nullptr);
    ASSERT_EQ(attn_proj->shape.size(), 2u);
    EXPECT_EQ(attn_proj->shape[0], 4096u);
    EXPECT_EQ(attn_proj->shape[1], 2560u);

    // ── FA-specific buffers ──

    // fa_q_raw: [seq_len, fa_q_full_dim] = [4096, 8192]
    auto *fa_q_raw = findBuf(reqs, "fa_q_raw");
    ASSERT_NE(fa_q_raw, nullptr);
    ASSERT_EQ(fa_q_raw->shape.size(), 2u);
    EXPECT_EQ(fa_q_raw->shape[0], 4096u);
    EXPECT_EQ(fa_q_raw->shape[1], 8192u);

    // fa_gate: [seq_len, local_qkv_dim] = [4096, 16*256=4096]
    auto *fa_gate = findBuf(reqs, "fa_gate");
    ASSERT_NE(fa_gate, nullptr);
    ASSERT_EQ(fa_gate->shape.size(), 2u);
    EXPECT_EQ(fa_gate->shape[0], 4096u);
    EXPECT_EQ(fa_gate->shape[1], 4096u);

    // Q: [seq_len, local_qkv_dim] = [4096, 4096]
    auto *Q = findBuf(reqs, "Q");
    ASSERT_NE(Q, nullptr);
    ASSERT_EQ(Q->shape.size(), 2u);
    EXPECT_EQ(Q->shape[0], 4096u);
    EXPECT_EQ(Q->shape[1], 4096u);

    // K: [seq_len, local_kv_dim] = [4096, 4*256=1024]
    auto *K = findBuf(reqs, "K");
    ASSERT_NE(K, nullptr);
    ASSERT_EQ(K->shape.size(), 2u);
    EXPECT_EQ(K->shape[0], 4096u);
    EXPECT_EQ(K->shape[1], 1024u);

    // V: [seq_len, local_kv_dim] = [4096, 1024]
    auto *V = findBuf(reqs, "V");
    ASSERT_NE(V, nullptr);
    ASSERT_EQ(V->shape.size(), 2u);
    EXPECT_EQ(V->shape[0], 4096u);
    EXPECT_EQ(V->shape[1], 1024u);

    // ── GDN-specific buffers ──

    // gdn_qkv: [seq_len, gdn_qkv_dim] = [4096, 8192]
    auto *gdn_qkv = findBuf(reqs, "gdn_qkv");
    ASSERT_NE(gdn_qkv, nullptr);
    ASSERT_EQ(gdn_qkv->shape.size(), 2u);
    EXPECT_EQ(gdn_qkv->shape[0], 4096u);
    EXPECT_EQ(gdn_qkv->shape[1], 8192u);

    // gdn_z: [seq_len, gdn_inner_size] = [4096, 4096]
    auto *gdn_z = findBuf(reqs, "gdn_z");
    ASSERT_NE(gdn_z, nullptr);
    ASSERT_EQ(gdn_z->shape.size(), 2u);
    EXPECT_EQ(gdn_z->shape[0], 4096u);
    EXPECT_EQ(gdn_z->shape[1], 4096u);

    // gdn_alpha: [seq_len, gdn_time_step_rank] = [4096, 32]
    auto *gdn_alpha = findBuf(reqs, "gdn_alpha");
    ASSERT_NE(gdn_alpha, nullptr);
    ASSERT_EQ(gdn_alpha->shape.size(), 2u);
    EXPECT_EQ(gdn_alpha->shape[0], 4096u);
    EXPECT_EQ(gdn_alpha->shape[1], 32u);

    // gdn_beta: [seq_len, gdn_time_step_rank] = [4096, 32]
    auto *gdn_beta = findBuf(reqs, "gdn_beta");
    ASSERT_NE(gdn_beta, nullptr);
    ASSERT_EQ(gdn_beta->shape.size(), 2u);
    EXPECT_EQ(gdn_beta->shape[0], 4096u);
    EXPECT_EQ(gdn_beta->shape[1], 32u);

    // ── FFN buffers ──

    // gate: [seq_len, local_d_ff] = [4096, 9216]
    auto *gate = findBuf(reqs, "gate");
    ASSERT_NE(gate, nullptr);
    ASSERT_EQ(gate->shape.size(), 2u);
    EXPECT_EQ(gate->shape[0], 4096u);
    EXPECT_EQ(gate->shape[1], 9216u);

    // up: [seq_len, local_d_ff] = [4096, 9216]
    auto *up = findBuf(reqs, "up");
    ASSERT_NE(up, nullptr);
    ASSERT_EQ(up->shape.size(), 2u);
    EXPECT_EQ(up->shape[0], 4096u);
    EXPECT_EQ(up->shape[1], 9216u);

    // ffn_output: [seq_len, d_model] = [4096, 2560]
    auto *ffn_output = findBuf(reqs, "ffn_output");
    ASSERT_NE(ffn_output, nullptr);
    ASSERT_EQ(ffn_output->shape.size(), 2u);
    EXPECT_EQ(ffn_output->shape[0], 4096u);
    EXPECT_EQ(ffn_output->shape[1], 2560u);

    // ── Workspace buffers ──

    // workspace_scores: [batch*local_n_heads*seq_len, seq_len] = [1*16*4096, 4096] = [65536, 4096]
    auto *ws_scores = findBuf(reqs, "workspace_scores");
    ASSERT_NE(ws_scores, nullptr);
    ASSERT_EQ(ws_scores->shape.size(), 2u);
    EXPECT_EQ(ws_scores->shape[0], 65536u);
    EXPECT_EQ(ws_scores->shape[1], 4096u);

    // workspace_context: [batch*local_n_heads*seq_len, head_dim] = [65536, 256]
    auto *ws_context = findBuf(reqs, "workspace_context");
    ASSERT_NE(ws_context, nullptr);
    ASSERT_EQ(ws_context->shape.size(), 2u);
    EXPECT_EQ(ws_context->shape[0], 65536u);
    EXPECT_EQ(ws_context->shape[1], 256u);

    // workspace_mask: [batch*seq_len, seq_len] = [4096, 4096]
    auto *ws_mask = findBuf(reqs, "workspace_mask");
    ASSERT_NE(ws_mask, nullptr);
    ASSERT_EQ(ws_mask->shape.size(), 2u);
    EXPECT_EQ(ws_mask->shape[0], 4096u);
    EXPECT_EQ(ws_mask->shape[1], 4096u);

    // lm_head_input_row: [1, d_model] = [1, 2560]
    auto *lm_head_input_row = findBuf(reqs, "lm_head_input_row");
    ASSERT_NE(lm_head_input_row, nullptr);
    ASSERT_EQ(lm_head_input_row->shape.size(), 2u);
    EXPECT_EQ(lm_head_input_row->shape[0], 1u);
    EXPECT_EQ(lm_head_input_row->shape[1], 2560u);

    // lm_head_input_rows: [4, d_model] compact verifier rows for MTP target logits.
    auto *lm_head_input_rows = findBuf(reqs, "lm_head_input_rows");
    ASSERT_NE(lm_head_input_rows, nullptr);
    ASSERT_EQ(lm_head_input_rows->shape.size(), 2u);
    EXPECT_EQ(lm_head_input_rows->shape[0], 4u);
    EXPECT_EQ(lm_head_input_rows->shape[1], 2560u);

    // ── MTP sidecar buffers ──

    auto *mtp_embedding = findBuf(reqs, "mtp_embedding");
    ASSERT_NE(mtp_embedding, nullptr);
    EXPECT_EQ(mtp_embedding->shape[0], 4u);
    EXPECT_EQ(mtp_embedding->shape[1], 2560u);

    auto *mtp_concat = findBuf(reqs, "mtp_concat");
    ASSERT_NE(mtp_concat, nullptr);
    EXPECT_EQ(mtp_concat->shape[0], 4u);
    EXPECT_EQ(mtp_concat->shape[1], 5120u);

    auto *mtp_q = findBuf(reqs, "mtp_q");
    ASSERT_NE(mtp_q, nullptr);
    EXPECT_EQ(mtp_q->shape[0], 4u);
    EXPECT_EQ(mtp_q->shape[1], 4096u);

    auto *mtp_k = findBuf(reqs, "mtp_k");
    ASSERT_NE(mtp_k, nullptr);
    EXPECT_EQ(mtp_k->shape[0], 4u);
    EXPECT_EQ(mtp_k->shape[1], 1024u);

    auto *mtp_gate = findBuf(reqs, "mtp_gate");
    ASSERT_NE(mtp_gate, nullptr);
    EXPECT_EQ(mtp_gate->shape[0], 4u);
    EXPECT_EQ(mtp_gate->shape[1], 9216u);

    auto *mtp_logits = findBuf(reqs, "mtp_logits");
    ASSERT_NE(mtp_logits, nullptr);
    EXPECT_EQ(mtp_logits->shape[0], 4u);
    EXPECT_EQ(mtp_logits->shape[1], 248320u);
}

TEST(Test__Qwen35BufferSizes, LayerBuffers_MTPRequestBatchVerifierRowsScale)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 2560;
    config.n_heads = 16;
    config.n_kv_heads = 4;
    config.head_dim = 256;
    config.vocab_size = 248320;
    config.seq_len = 4096;
    config.batch_size = 1;
    config.local_n_heads = 16;
    config.local_n_kv_heads = 4;
    config.local_d_ff = 9216;
    config.local_vocab = 248320;
    config.custom_formulas["gdn_inner_size"] = 4096;
    config.custom_formulas["gdn_qkv_dim"] = 8192;
    config.custom_formulas["gdn_time_step_rank"] = 32;
    config.custom_formulas["fa_q_full_dim"] = 8192;
    config.custom_formulas["attn_output_dim"] = 4096;
    config.custom_formulas["mtp_target_query_rows"] = 8;

    auto reqs = BufferAllocator::resolveLayerBuffers(schema, config);
    auto *lm_head_input_rows = findBuf(reqs, "lm_head_input_rows");
    ASSERT_NE(lm_head_input_rows, nullptr);
    ASSERT_EQ(lm_head_input_rows->shape.size(), 2u);
    EXPECT_EQ(lm_head_input_rows->shape[0], 8u);
    EXPECT_EQ(lm_head_input_rows->shape[1], 2560u);
}

// ============================================================================
// Model Buffer Size Lock-in
// ============================================================================

TEST(Test__Qwen35BufferSizes, ModelBuffers_ExactShapes)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Qwen3.5-4B: vocab=248320, d=2560
    GraphResolverConfig config{};
    config.d_model = 2560;
    config.vocab_size = 248320;
    config.seq_len = 4096;
    config.local_vocab = 248320;

    auto reqs = BufferAllocator::resolveModelBuffers(schema, config);

    EXPECT_EQ(reqs.buffers.size(), 3u) << "Should have 3 model buffers";

    // hidden: [seq_len, d_model] = [4096, 2560]
    auto *hidden = findBuf(reqs, "hidden");
    ASSERT_NE(hidden, nullptr);
    ASSERT_EQ(hidden->shape.size(), 2u);
    EXPECT_EQ(hidden->shape[0], 4096u);
    EXPECT_EQ(hidden->shape[1], 2560u);

    // logits: [1, vocab_size] — only 1 row (last-token logits only)
    auto *logits = findBuf(reqs, "logits");
    ASSERT_NE(logits, nullptr);
    ASSERT_EQ(logits->shape.size(), 2u);
    EXPECT_EQ(logits->shape[0], 1u);
    EXPECT_EQ(logits->shape[1], 248320u);

    // logits_local: [1, local_vocab]
    auto *logits_local = findBuf(reqs, "logits_local");
    ASSERT_NE(logits_local, nullptr);
    ASSERT_EQ(logits_local->shape.size(), 2u);
    EXPECT_EQ(logits_local->shape[0], 1u);
    EXPECT_EQ(logits_local->shape[1], 248320u);
}

TEST(Test__Qwen35BufferSizes, ModelBuffers_TP2)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    GraphResolverConfig config{};
    config.d_model = 2560;
    config.vocab_size = 248320;
    config.seq_len = 4096;
    config.local_vocab = 124160; // TP=2

    auto reqs = BufferAllocator::resolveModelBuffers(schema, config);

    // logits: [1, vocab_size] — full vocab even under TP
    auto *logits = findBuf(reqs, "logits");
    ASSERT_NE(logits, nullptr);
    EXPECT_EQ(logits->shape[0], 1u);
    EXPECT_EQ(logits->shape[1], 248320u);

    // logits_local: [1, local_vocab] — half vocab under TP=2
    auto *logits_local = findBuf(reqs, "logits_local");
    ASSERT_NE(logits_local, nullptr);
    EXPECT_EQ(logits_local->shape[0], 1u);
    EXPECT_EQ(logits_local->shape[1], 124160u);
}

// ============================================================================
// Layer Buffer Sizes with TP=2
// ============================================================================

TEST(Test__Qwen35BufferSizes, LayerBuffers_TP2)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // TP=2: heads split, d_ff split
    GraphResolverConfig config{};
    config.d_model = 2560;
    config.n_heads = 16;
    config.n_kv_heads = 4;
    config.head_dim = 256;
    config.vocab_size = 248320;
    config.seq_len = 4096;
    config.batch_size = 1;
    config.local_n_heads = 8;    // 16/2
    config.local_n_kv_heads = 2; // 4/2
    config.local_d_ff = 4608;    // 9216/2
    config.local_vocab = 124160; // 248320/2

    // GDN formulas under TP=2 (n_k=8, n_v=16, d_k=128)
    // key_dim = 8*128 = 1024
    // gdn_inner = 16*128 = 2048
    // gdn_qkv_dim = 2*1024 + 2048 = 4096
    // fa_q_full_dim = 8*256*2 = 4096
    // attn_output_dim = max(8*256, 2048) = max(2048, 2048) = 2048
    config.custom_formulas["gdn_inner_size"] = 2048;
    config.custom_formulas["gdn_qkv_dim"] = 4096;
    config.custom_formulas["gdn_time_step_rank"] = 16;
    config.custom_formulas["fa_q_full_dim"] = 4096;
    config.custom_formulas["attn_output_dim"] = 2048;

    auto reqs = BufferAllocator::resolveLayerBuffers(schema, config);

    EXPECT_EQ(reqs.buffers.size(), 38u);

    // Q: [4096, 8*256=2048] under TP=2
    auto *Q = findBuf(reqs, "Q");
    ASSERT_NE(Q, nullptr);
    EXPECT_EQ(Q->shape[1], 2048u);

    // K: [4096, 2*256=512]
    auto *K = findBuf(reqs, "K");
    ASSERT_NE(K, nullptr);
    EXPECT_EQ(K->shape[1], 512u);

    // gate/up: [4096, 4608] under TP=2
    auto *gate = findBuf(reqs, "gate");
    ASSERT_NE(gate, nullptr);
    EXPECT_EQ(gate->shape[1], 4608u);

    // gdn_qkv: [4096, 4096] under TP=2
    auto *gdn_qkv = findBuf(reqs, "gdn_qkv");
    ASSERT_NE(gdn_qkv, nullptr);
    EXPECT_EQ(gdn_qkv->shape[1], 4096u);

    // attn_output: [4096, 2048] under TP=2
    auto *attn_output = findBuf(reqs, "attn_output");
    ASSERT_NE(attn_output, nullptr);
    EXPECT_EQ(attn_output->shape[1], 2048u);

    // workspace_scores: [1*8*4096, 4096] = [32768, 4096] under TP=2
    auto *ws_scores = findBuf(reqs, "workspace_scores");
    ASSERT_NE(ws_scores, nullptr);
    EXPECT_EQ(ws_scores->shape[0], 32768u);

    auto *lm_head_input_row = findBuf(reqs, "lm_head_input_row");
    ASSERT_NE(lm_head_input_row, nullptr);
    EXPECT_EQ(lm_head_input_row->shape[0], 1u);
    EXPECT_EQ(lm_head_input_row->shape[1], 2560u);

    auto *lm_head_input_rows = findBuf(reqs, "lm_head_input_rows");
    ASSERT_NE(lm_head_input_rows, nullptr);
    EXPECT_EQ(lm_head_input_rows->shape[0], 4u);
    EXPECT_EQ(lm_head_input_rows->shape[1], 2560u);

    auto *mtp_q = findBuf(reqs, "mtp_q");
    ASSERT_NE(mtp_q, nullptr);
    EXPECT_EQ(mtp_q->shape[1], 2048u);

    auto *mtp_k = findBuf(reqs, "mtp_k");
    ASSERT_NE(mtp_k, nullptr);
    EXPECT_EQ(mtp_k->shape[1], 512u);

    auto *mtp_gate = findBuf(reqs, "mtp_gate");
    ASSERT_NE(mtp_gate, nullptr);
    EXPECT_EQ(mtp_gate->shape[1], 4608u);

    auto *mtp_logits = findBuf(reqs, "mtp_logits");
    ASSERT_NE(mtp_logits, nullptr);
    EXPECT_EQ(mtp_logits->shape[0], 4u);
    EXPECT_EQ(mtp_logits->shape[1], 124160u);
}
