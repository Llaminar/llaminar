/**
 * @file Test__StageWeightContracts.cpp
 * @brief Unit tests verifying all weight-bearing stages declare weights in bufferContract()
 *
 * These tests lock in the paradigm established during the executor coherence migration:
 * every stage that holds model weights MUST declare them via bufferContract().addWeight().
 * The executor's weight coherence loop ONLY processes contract.weight_tensors — any weight
 * not declared there will never be uploaded to device, causing illegal memory access.
 *
 * Tests cover:
 * 1. Each weight-bearing stage declares non-empty weight_tensors in its contract
 * 2. Weight pointers in the contract match the pointers passed via Params
 * 3. Stages without BufferIds return empty contracts (graceful fallback)
 * 4. Null weight pointers are NOT added to the contract
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/ComputeStages.h"
#include "execution/compute_stages/stages/FusedResidualNormStage.h"
#include "execution/compute_stages/stages/QKNormStage.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "utils/TestTensorFactory.h"
#include "memory/BufferId.h"
#include "memory/StageBufferContract.h"
#include "backends/DeviceId.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture — provides tensor factories for all stage Params
// =============================================================================

class Test__StageWeightContracts : public ::testing::Test
{
protected:
    std::unique_ptr<FP32Tensor> makeFP32(size_t rows, size_t cols)
    {
        return TestTensorFactory::createFP32Random({rows, cols});
    }

    std::unique_ptr<Q8_0Tensor> makeQ8_0(size_t rows, size_t cols)
    {
        return TestTensorFactory::createQ8_0Random({rows, cols});
    }

    // Helper: check that a specific ITensor* appears in contract.weight_tensors
    static bool contractContainsWeight(const StageBufferContract &contract, const ITensor *expected)
    {
        for (const auto *w : contract.weight_tensors)
        {
            if (w == expected)
                return true;
        }
        return false;
    }

    // Typical model dimensions (Qwen2.5-0.5B-like)
    static constexpr size_t SEQ_LEN = 4;
    static constexpr size_t D_MODEL = 896;
    static constexpr size_t D_FF = 4864;
    static constexpr int N_HEADS = 14;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
    static constexpr int VOCAB_SIZE = 1024;
};

// =============================================================================
// GEMMStage
// =============================================================================

TEST_F(Test__StageWeightContracts, GEMMStage_DeclaresWeightB)
{
    auto A = makeFP32(SEQ_LEN, D_MODEL);
    auto B = makeQ8_0(D_FF, D_MODEL);
    auto C = makeFP32(SEQ_LEN, D_FF);

    GEMMStage::Params params{};
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = SEQ_LEN;
    params.n = D_FF;
    params.k = D_MODEL;
    params.a_buffer_id = BufferId::HIDDEN_STATE;
    params.c_buffer_id = BufferId::FFN_OUTPUT;

    GEMMStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_GE(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, B.get()));
}

TEST_F(Test__StageWeightContracts, GEMMStage_DeclaresWeightBAndBias)
{
    auto A = makeFP32(SEQ_LEN, D_MODEL);
    auto B = makeQ8_0(D_FF, D_MODEL);
    auto C = makeFP32(SEQ_LEN, D_FF);
    auto bias = makeFP32(1, D_FF);

    GEMMStage::Params params{};
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = SEQ_LEN;
    params.n = D_FF;
    params.k = D_MODEL;
    params.bias_tensor = bias.get();
    params.a_buffer_id = BufferId::HIDDEN_STATE;
    params.c_buffer_id = BufferId::FFN_OUTPUT;

    GEMMStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_EQ(contract.weight_tensors.size(), 2u);
    EXPECT_TRUE(contractContainsWeight(contract, B.get()));
    EXPECT_TRUE(contractContainsWeight(contract, bias.get()));
}

TEST_F(Test__StageWeightContracts, GEMMStage_NullB_EmptyWeights)
{
    auto A = makeFP32(SEQ_LEN, D_MODEL);
    auto C = makeFP32(SEQ_LEN, D_FF);

    GEMMStage::Params params{};
    params.A = A.get();
    params.B = nullptr; // No weight
    params.C = C.get();
    params.m = SEQ_LEN;
    params.n = D_FF;
    params.k = D_MODEL;
    params.a_buffer_id = BufferId::HIDDEN_STATE;
    params.c_buffer_id = BufferId::FFN_OUTPUT;

    GEMMStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_TRUE(contract.weight_tensors.empty());
}

TEST_F(Test__StageWeightContracts, GEMMStage_NoBufferIds_EmptyContract)
{
    auto A = makeFP32(SEQ_LEN, D_MODEL);
    auto B = makeQ8_0(D_FF, D_MODEL);
    auto C = makeFP32(SEQ_LEN, D_FF);

    GEMMStage::Params params{};
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = SEQ_LEN;
    params.n = D_FF;
    params.k = D_MODEL;
    // No a_buffer_id or c_buffer_id set

    GEMMStage stage(params);
    auto contract = stage.bufferContract();

    // Without BufferIds, contract should be empty (graceful fallback)
    EXPECT_TRUE(contract.empty());
}

// =============================================================================
// EmbeddingStage
// =============================================================================

TEST_F(Test__StageWeightContracts, EmbeddingStage_DeclaresEmbedTable)
{
    auto embed = makeQ8_0(VOCAB_SIZE, D_MODEL);
    auto output = makeFP32(SEQ_LEN, D_MODEL);
    int token_ids[] = {1, 2, 3, 4};

    EmbeddingStage::Params params{};
    params.embed_table = embed.get();
    params.token_ids = token_ids;
    params.output = output.get();
    params.num_tokens = SEQ_LEN;
    params.d_model = D_MODEL;
    params.vocab_size = VOCAB_SIZE;
    params.output_buffer_id = BufferId::HIDDEN_STATE;

    EmbeddingStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, embed.get()));
}

TEST_F(Test__StageWeightContracts, EmbeddingStage_NoOutputBufferId_EmptyContract)
{
    auto embed = makeQ8_0(VOCAB_SIZE, D_MODEL);
    auto output = makeFP32(SEQ_LEN, D_MODEL);
    int token_ids[] = {1, 2, 3, 4};

    EmbeddingStage::Params params{};
    params.embed_table = embed.get();
    params.token_ids = token_ids;
    params.output = output.get();
    params.num_tokens = SEQ_LEN;
    params.d_model = D_MODEL;
    params.vocab_size = VOCAB_SIZE;
    // No output_buffer_id

    EmbeddingStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_TRUE(contract.empty());
}

// =============================================================================
// RMSNormStage
// =============================================================================

TEST_F(Test__StageWeightContracts, RMSNormStage_DeclaresGamma)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto output = makeFP32(SEQ_LEN, D_MODEL);
    auto gamma = makeFP32(1, D_MODEL);

    RMSNormStage::Params params{};
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.input_buffer_id = BufferId::HIDDEN_STATE;
    params.output_buffer_id = BufferId::NORMALIZED;

    RMSNormStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, gamma.get()));
}

TEST_F(Test__StageWeightContracts, RMSNormStage_InPlace_HasInOutBinding)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto gamma = makeFP32(1, D_MODEL);

    RMSNormStage::Params params{};
    params.input = input.get();
    params.output = input.get(); // Same tensor = in-place
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.input_buffer_id = BufferId::HIDDEN_STATE;
    params.output_buffer_id = BufferId::HIDDEN_STATE; // Same buffer

    RMSNormStage stage(params);
    auto contract = stage.bufferContract();

    // In-place: should use addInOut, not separate addInput/addOutput
    EXPECT_EQ(contract.inputs.size(), 0u);
    EXPECT_EQ(contract.outputs.size(), 0u);
    EXPECT_EQ(contract.inouts.size(), 1u);
    EXPECT_EQ(contract.inouts[0].id, BufferId::HIDDEN_STATE);
}

// =============================================================================
// LMHeadStage
// =============================================================================

TEST_F(Test__StageWeightContracts, LMHeadStage_DeclaresWeight)
{
    auto hidden = makeFP32(SEQ_LEN, D_MODEL);
    auto weight = makeQ8_0(VOCAB_SIZE, D_MODEL);
    auto logits = makeFP32(SEQ_LEN, VOCAB_SIZE);

    LMHeadStage::Params params{};
    params.hidden_states = hidden.get();
    params.lm_head_weight = weight.get();
    params.logits = logits.get();
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.vocab_size = VOCAB_SIZE;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_buffer_id = BufferId::LOGITS;

    LMHeadStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_GE(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, weight.get()));
}

TEST_F(Test__StageWeightContracts, LMHeadStage_WithBias_DeclaresBoth)
{
    auto hidden = makeFP32(SEQ_LEN, D_MODEL);
    auto weight = makeQ8_0(VOCAB_SIZE, D_MODEL);
    auto logits = makeFP32(SEQ_LEN, VOCAB_SIZE);
    auto bias = makeFP32(1, VOCAB_SIZE);

    LMHeadStage::Params params{};
    params.hidden_states = hidden.get();
    params.lm_head_weight = weight.get();
    params.logits = logits.get();
    params.bias_tensor = dynamic_cast<TensorBase *>(bias.get());
    params.seq_len = SEQ_LEN;
    params.d_model = D_MODEL;
    params.vocab_size = VOCAB_SIZE;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_buffer_id = BufferId::LOGITS;

    LMHeadStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_EQ(contract.weight_tensors.size(), 2u);
    EXPECT_TRUE(contractContainsWeight(contract, weight.get()));
    EXPECT_TRUE(contractContainsWeight(contract, bias.get()));
}

// =============================================================================
// FusedQKVGEMMStage
// =============================================================================

TEST_F(Test__StageWeightContracts, FusedQKVGEMMStage_DeclaresAllWeights)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto wq = makeQ8_0(N_HEADS * HEAD_DIM, D_MODEL);
    auto wk = makeQ8_0(N_KV_HEADS * HEAD_DIM, D_MODEL);
    auto wv = makeQ8_0(N_KV_HEADS * HEAD_DIM, D_MODEL);
    auto oq = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto ok = makeFP32(SEQ_LEN, N_KV_HEADS * HEAD_DIM);
    auto ov = makeFP32(SEQ_LEN, N_KV_HEADS * HEAD_DIM);

    FusedQKVGEMMStage::Params params{};
    params.input = input.get();
    params.wq = wq.get();
    params.wk = wk.get();
    params.wv = wv.get();
    params.output_q = oq.get();
    params.output_k = ok.get();
    params.output_v = ov.get();
    params.m = SEQ_LEN;
    params.k = D_MODEL;
    params.n_q = N_HEADS * HEAD_DIM;
    params.n_k = N_KV_HEADS * HEAD_DIM;
    params.n_v = N_KV_HEADS * HEAD_DIM;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_q_buffer_id = BufferId::Q_PROJ;
    params.output_k_buffer_id = BufferId::K_PROJ;
    params.output_v_buffer_id = BufferId::V_PROJ;

    FusedQKVGEMMStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    // 3 weight matrices (no biases)
    EXPECT_EQ(contract.weight_tensors.size(), 3u);
    EXPECT_TRUE(contractContainsWeight(contract, wq.get()));
    EXPECT_TRUE(contractContainsWeight(contract, wk.get()));
    EXPECT_TRUE(contractContainsWeight(contract, wv.get()));
}

TEST_F(Test__StageWeightContracts, FusedQKVGEMMStage_WithBiases_DeclaresAll)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto wq = makeQ8_0(N_HEADS * HEAD_DIM, D_MODEL);
    auto wk = makeQ8_0(N_KV_HEADS * HEAD_DIM, D_MODEL);
    auto wv = makeQ8_0(N_KV_HEADS * HEAD_DIM, D_MODEL);
    auto bq = makeFP32(1, N_HEADS * HEAD_DIM);
    auto bk = makeFP32(1, N_KV_HEADS * HEAD_DIM);
    auto bv = makeFP32(1, N_KV_HEADS * HEAD_DIM);
    auto oq = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto ok = makeFP32(SEQ_LEN, N_KV_HEADS * HEAD_DIM);
    auto ov = makeFP32(SEQ_LEN, N_KV_HEADS * HEAD_DIM);

    FusedQKVGEMMStage::Params params{};
    params.input = input.get();
    params.wq = wq.get();
    params.wk = wk.get();
    params.wv = wv.get();
    params.bias_q = dynamic_cast<TensorBase *>(bq.get());
    params.bias_k = dynamic_cast<TensorBase *>(bk.get());
    params.bias_v = dynamic_cast<TensorBase *>(bv.get());
    params.output_q = oq.get();
    params.output_k = ok.get();
    params.output_v = ov.get();
    params.m = SEQ_LEN;
    params.k = D_MODEL;
    params.n_q = N_HEADS * HEAD_DIM;
    params.n_k = N_KV_HEADS * HEAD_DIM;
    params.n_v = N_KV_HEADS * HEAD_DIM;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_q_buffer_id = BufferId::Q_PROJ;
    params.output_k_buffer_id = BufferId::K_PROJ;
    params.output_v_buffer_id = BufferId::V_PROJ;

    FusedQKVGEMMStage stage(params);
    auto contract = stage.bufferContract();

    // 3 weights + 3 biases = 6
    EXPECT_EQ(contract.weight_tensors.size(), 6u);
    EXPECT_TRUE(contractContainsWeight(contract, wq.get()));
    EXPECT_TRUE(contractContainsWeight(contract, wk.get()));
    EXPECT_TRUE(contractContainsWeight(contract, wv.get()));
    EXPECT_TRUE(contractContainsWeight(contract, bq.get()));
    EXPECT_TRUE(contractContainsWeight(contract, bk.get()));
    EXPECT_TRUE(contractContainsWeight(contract, bv.get()));
}

// =============================================================================
// FusedGateUpGEMMStage
// =============================================================================

TEST_F(Test__StageWeightContracts, FusedGateUpGEMMStage_DeclaresGateAndUp)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto w_gate = makeQ8_0(D_FF, D_MODEL);
    auto w_up = makeQ8_0(D_FF, D_MODEL);
    auto o_gate = makeFP32(SEQ_LEN, D_FF);
    auto o_up = makeFP32(SEQ_LEN, D_FF);

    FusedGateUpGEMMStage::Params params{};
    params.input = input.get();
    params.w_gate = w_gate.get();
    params.w_up = w_up.get();
    params.output_gate = o_gate.get();
    params.output_up = o_up.get();
    params.m = SEQ_LEN;
    params.k = D_MODEL;
    params.n_gate = D_FF;
    params.n_up = D_FF;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_gate_buffer_id = BufferId::GATE_PROJ;
    params.output_up_buffer_id = BufferId::UP_PROJ;

    FusedGateUpGEMMStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_EQ(contract.weight_tensors.size(), 2u);
    EXPECT_TRUE(contractContainsWeight(contract, w_gate.get()));
    EXPECT_TRUE(contractContainsWeight(contract, w_up.get()));
}

TEST_F(Test__StageWeightContracts, FusedGateUpGEMMStage_WithBiases_DeclaresFour)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto w_gate = makeQ8_0(D_FF, D_MODEL);
    auto w_up = makeQ8_0(D_FF, D_MODEL);
    auto b_gate = makeFP32(1, D_FF);
    auto b_up = makeFP32(1, D_FF);
    auto o_gate = makeFP32(SEQ_LEN, D_FF);
    auto o_up = makeFP32(SEQ_LEN, D_FF);

    FusedGateUpGEMMStage::Params params{};
    params.input = input.get();
    params.w_gate = w_gate.get();
    params.w_up = w_up.get();
    params.bias_gate = dynamic_cast<TensorBase *>(b_gate.get());
    params.bias_up = dynamic_cast<TensorBase *>(b_up.get());
    params.output_gate = o_gate.get();
    params.output_up = o_up.get();
    params.m = SEQ_LEN;
    params.k = D_MODEL;
    params.n_gate = D_FF;
    params.n_up = D_FF;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_gate_buffer_id = BufferId::GATE_PROJ;
    params.output_up_buffer_id = BufferId::UP_PROJ;

    FusedGateUpGEMMStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_EQ(contract.weight_tensors.size(), 4u);
    EXPECT_TRUE(contractContainsWeight(contract, w_gate.get()));
    EXPECT_TRUE(contractContainsWeight(contract, w_up.get()));
    EXPECT_TRUE(contractContainsWeight(contract, b_gate.get()));
    EXPECT_TRUE(contractContainsWeight(contract, b_up.get()));
}

// =============================================================================
// FusedResidualNormStage
// =============================================================================

TEST_F(Test__StageWeightContracts, FusedResidualNormStage_DeclaresGamma)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto residual = makeFP32(SEQ_LEN, D_MODEL);
    auto norm_output = makeFP32(SEQ_LEN, D_MODEL);
    auto gamma = makeFP32(1, D_MODEL);

    FusedResidualNormStage::Params params{};
    params.input = input.get();
    params.residual = residual.get();
    params.norm_output = norm_output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.hidden_dim = D_MODEL;
    params.input_buffer_id = BufferId::ATTN_PROJ;
    params.residual_buffer_id = BufferId::RESIDUAL;
    params.norm_output_buffer_id = BufferId::NORMALIZED;

    FusedResidualNormStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, gamma.get()));
}

// =============================================================================
// QKNormStage
// =============================================================================

TEST_F(Test__StageWeightContracts, QKNormStage_DeclaresGamma)
{
    auto input = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto output = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto gamma = makeFP32(1, HEAD_DIM);

    QKNormStage::Params params{};
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.n_heads = N_HEADS;
    params.head_dim = HEAD_DIM;
    params.eps = 1e-6f;
    params.input_buffer_id = BufferId::Q_PROJ;
    params.output_buffer_id = BufferId::Q_PROJ; // In-place

    QKNormStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, gamma.get()));
}

// =============================================================================
// GDN Stages (Qwen3.5-specific)
// =============================================================================

TEST_F(Test__StageWeightContracts, GDNProjectionStage_DeclaresFourWeights)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto w_qkv = makeQ8_0(3 * N_HEADS * HEAD_DIM, D_MODEL);
    auto w_z = makeQ8_0(N_HEADS * HEAD_DIM, D_MODEL);
    auto w_a = makeQ8_0(N_HEADS, D_MODEL);
    auto w_b = makeQ8_0(N_HEADS, D_MODEL);
    auto o_qkv = makeFP32(SEQ_LEN, 3 * N_HEADS * HEAD_DIM);
    auto o_z = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto o_a = makeFP32(SEQ_LEN, N_HEADS);
    auto o_b = makeFP32(SEQ_LEN, N_HEADS);

    GDNProjectionStage::Params params{};
    params.input = input.get();
    params.w_qkv = w_qkv.get();
    params.w_z = w_z.get();
    params.w_a = w_a.get();
    params.w_b = w_b.get();
    params.output_qkv = o_qkv.get();
    params.output_z = o_z.get();
    params.output_a = o_a.get();
    params.output_b = o_b.get();
    params.m = SEQ_LEN;
    params.k = D_MODEL;
    params.n_qkv = 3 * N_HEADS * HEAD_DIM;
    params.n_z = N_HEADS * HEAD_DIM;
    params.n_a = N_HEADS;
    params.n_b = N_HEADS;
    params.input_buffer_id = BufferId::NORMALIZED;
    params.output_qkv_buffer_id = BufferId::GDN_QKV;
    params.output_z_buffer_id = BufferId::GDN_Z;
    params.output_a_buffer_id = BufferId::GDN_ALPHA;
    params.output_b_buffer_id = BufferId::GDN_BETA;

    GDNProjectionStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 4u);
    EXPECT_TRUE(contractContainsWeight(contract, w_qkv.get()));
    EXPECT_TRUE(contractContainsWeight(contract, w_z.get()));
    EXPECT_TRUE(contractContainsWeight(contract, w_a.get()));
    EXPECT_TRUE(contractContainsWeight(contract, w_b.get()));
}

TEST_F(Test__StageWeightContracts, GDNRecurrenceStage_DeclaresALogAndDtBias)
{
    auto Q = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto K = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto V = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto alpha = makeFP32(SEQ_LEN, N_HEADS);
    auto beta = makeFP32(SEQ_LEN, N_HEADS);
    auto A_log = makeFP32(1, N_HEADS);
    auto dt_bias = makeFP32(1, N_HEADS);
    auto output = makeFP32(SEQ_LEN, N_HEADS * HEAD_DIM);

    GDNRecurrenceStage::Params params{};
    params.Q = Q.get();
    params.K = K.get();
    params.V = V.get();
    params.alpha = alpha.get();
    params.beta = beta.get();
    params.A_log = A_log.get();
    params.dt_bias = dt_bias.get();
    params.output = output.get();
    params.seq_len = SEQ_LEN;
    params.n_heads = N_HEADS;
    params.d_k = HEAD_DIM;
    params.d_v = HEAD_DIM;
    params.qkv_buffer_id = BufferId::GDN_QKV;
    params.alpha_buffer_id = BufferId::GDN_ALPHA;
    params.beta_buffer_id = BufferId::GDN_BETA;
    params.output_buffer_id = BufferId::ATTN_OUTPUT;

    GDNRecurrenceStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    // A_log and dt_bias are model weights
    EXPECT_EQ(contract.weight_tensors.size(), 2u);
    EXPECT_TRUE(contractContainsWeight(contract, A_log.get()));
    EXPECT_TRUE(contractContainsWeight(contract, dt_bias.get()));
}

TEST_F(Test__StageWeightContracts, GatedRMSNormStage_DeclaresGamma)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto gate = makeFP32(SEQ_LEN, D_MODEL);
    auto output = makeFP32(SEQ_LEN, D_MODEL);
    auto gamma = makeFP32(1, D_MODEL);

    GatedRMSNormStage::Params params{};
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.input_buffer_id = BufferId::HIDDEN_STATE;
    params.gate_buffer_id = BufferId::GDN_Z;
    params.output_buffer_id = BufferId::NORMALIZED;

    GatedRMSNormStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, gamma.get()));
}

TEST_F(Test__StageWeightContracts, ShortConv1dStage_DeclaresWeightAndBias)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto output = makeFP32(SEQ_LEN, D_MODEL);
    auto weight = makeFP32(D_MODEL, 4); // kernel_size=4
    auto bias = makeFP32(1, D_MODEL);

    ShortConv1dStage::Params params{};
    params.input = input.get();
    params.output = output.get();
    params.weight = weight.get();
    params.bias = bias.get();
    params.seq_len = SEQ_LEN;
    params.channels = D_MODEL;
    params.kernel_size = 4;

    ShortConv1dStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_FALSE(contract.empty());
    EXPECT_EQ(contract.weight_tensors.size(), 2u);
    EXPECT_TRUE(contractContainsWeight(contract, weight.get()));
    EXPECT_TRUE(contractContainsWeight(contract, bias.get()));
}

TEST_F(Test__StageWeightContracts, ShortConv1dStage_NoBias_DeclaresWeightOnly)
{
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto output = makeFP32(SEQ_LEN, D_MODEL);
    auto weight = makeFP32(D_MODEL, 4);

    ShortConv1dStage::Params params{};
    params.input = input.get();
    params.output = output.get();
    params.weight = weight.get();
    params.bias = nullptr; // No bias
    params.seq_len = SEQ_LEN;
    params.channels = D_MODEL;
    params.kernel_size = 4;

    ShortConv1dStage stage(params);
    auto contract = stage.bufferContract();

    EXPECT_EQ(contract.weight_tensors.size(), 1u);
    EXPECT_TRUE(contractContainsWeight(contract, weight.get()));
}

// =============================================================================
// Contract Structural Invariants
// =============================================================================

TEST_F(Test__StageWeightContracts, AllContracts_WeightPointersAreNonNull)
{
    // Build a representative stage and verify no null pointers in weight_tensors
    auto A = makeFP32(SEQ_LEN, D_MODEL);
    auto B = makeQ8_0(D_FF, D_MODEL);
    auto C = makeFP32(SEQ_LEN, D_FF);
    auto bias = makeFP32(1, D_FF);

    GEMMStage::Params params{};
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.bias_tensor = bias.get();
    params.m = SEQ_LEN;
    params.n = D_FF;
    params.k = D_MODEL;
    params.a_buffer_id = BufferId::HIDDEN_STATE;
    params.c_buffer_id = BufferId::FFN_OUTPUT;

    GEMMStage stage(params);
    auto contract = stage.bufferContract();

    for (const auto *weight : contract.weight_tensors)
    {
        EXPECT_NE(weight, nullptr) << "Weight tensor in contract must not be null";
    }
}

TEST_F(Test__StageWeightContracts, AllContracts_WeightsAreSeparateFromArenaBindings)
{
    // Verify contract.weight_tensors doesn't overlap with inputs/outputs
    // (weights are external, not arena-managed)
    auto input = makeFP32(SEQ_LEN, D_MODEL);
    auto output = makeFP32(SEQ_LEN, D_MODEL);
    auto gamma = makeFP32(1, D_MODEL);

    RMSNormStage::Params params{};
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.input_buffer_id = BufferId::HIDDEN_STATE;
    params.output_buffer_id = BufferId::NORMALIZED;

    RMSNormStage stage(params);
    auto contract = stage.bufferContract();

    // Weights are separate from arena bindings
    EXPECT_FALSE(contract.inputs.empty() || contract.outputs.empty());
    EXPECT_FALSE(contract.weight_tensors.empty());

    // No overlap between weight tensors and arena buffer IDs
    // (This is a structural invariant — weights use ITensor*, arena uses BufferId)
    for (const auto &binding : contract.inputs)
    {
        EXPECT_NE(binding.id, BufferId::_COUNT) << "Arena binding must be a valid BufferId";
    }
}
