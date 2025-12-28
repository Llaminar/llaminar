/**
 * @file Test__StageDumpInfo.cpp
 * @brief Unit tests for ComputeStage::getDumpInfo() implementations
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests snapshot/dump info generation for all compute stages:
 * - GEMMStage
 * - FusedQKVGEMMStage
 * - FusedGateUpGEMMStage
 * - RMSNormStage
 * - RoPEStage
 * - SwiGLUStage
 * - ResidualAddStage
 * - AllreduceStage
 * - AllGatherStage
 * - MoERouterStage, MoEExpertStage, MoECombineStage
 * - KVCacheAppendStage, KVCacheGatherStage
 * - AttentionComputeStage
 * - EmbeddingStage
 * - LMHeadStage
 *
 * Each test verifies:
 * - getDumpInfo() returns non-empty info when buffers are set
 * - Scalar parameters are captured correctly
 * - Input/output buffers are referenced correctly
 */

#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <vector>

#include "execution/compute_stages/ComputeStages.h"
#include "../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class StageDumpInfoTest : public ::testing::Test
{
protected:
    // Helper to fill buffer with random FP32 values
    void fillRandom(float *data, size_t count, float min = -1.0f, float max = 1.0f)
    {
        std::uniform_real_distribution<float> dist(min, max);
        for (size_t i = 0; i < count; ++i)
        {
            data[i] = dist(rng_);
        }
    }

    // Helper to find scalar by name
    bool hasScalar(const StageDumpInfo &info, const char *name)
    {
        for (const auto &s : info.scalars)
        {
            if (std::string(s.name) == name)
                return true;
        }
        return false;
    }

    int64_t getScalarInt(const StageDumpInfo &info, const char *name)
    {
        for (const auto &s : info.scalars)
        {
            if (std::string(s.name) == name)
                return s.value;
        }
        return -1;
    }

    // Helper to find output by name
    bool hasOutput(const StageDumpInfo &info, const char *name)
    {
        for (const auto &o : info.outputs)
        {
            if (std::string(o.name) == name)
                return true;
        }
        return false;
    }

    // Helper to find input by name
    bool hasInput(const StageDumpInfo &info, const char *name)
    {
        for (const auto &i : info.inputs)
        {
            if (std::string(i.name) == name)
                return true;
        }
        return false;
    }

    // Helper to find weight by name
    bool hasWeight(const StageDumpInfo &info, const char *name)
    {
        for (const auto &w : info.weights)
        {
            if (std::string(w.name) == name)
                return true;
        }
        return false;
    }

    std::mt19937 rng_{42};
};

// =============================================================================
// GEMMStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, GEMMStage_GetDumpInfo)
{
    int m = 32, n = 64, k = 128;

    auto A = TestTensorFactory::createFP32Random({static_cast<size_t>(m), static_cast<size_t>(k)});
    auto B = TestTensorFactory::createFP32Random({static_cast<size_t>(k), static_cast<size_t>(n)});
    auto C = TestTensorFactory::createFP32({static_cast<size_t>(m), static_cast<size_t>(n)});

    GEMMStage::Params params;
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = m;
    params.n = n;
    params.k = k;

    GEMMStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "m"));
    EXPECT_TRUE(hasScalar(info, "n"));
    EXPECT_TRUE(hasScalar(info, "k"));
    EXPECT_EQ(getScalarInt(info, "m"), m);
    EXPECT_EQ(getScalarInt(info, "n"), n);
    EXPECT_EQ(getScalarInt(info, "k"), k);

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "C"));
}

// =============================================================================
// FusedQKVGEMMStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, FusedQKVGEMMStage_GetDumpInfo)
{
    int seq_len = 16;
    int d_model = 256;
    int n_heads = 4;
    int n_kv_heads = 2;
    int head_dim = 64;

    int m = seq_len;
    int k = d_model;
    int n_q = n_heads * head_dim;
    int n_k = n_kv_heads * head_dim;
    int n_v = n_kv_heads * head_dim;

    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(m), static_cast<size_t>(k)});
    auto output_q = TestTensorFactory::createFP32({static_cast<size_t>(m), static_cast<size_t>(n_q)});
    auto output_k = TestTensorFactory::createFP32({static_cast<size_t>(m), static_cast<size_t>(n_k)});
    auto output_v = TestTensorFactory::createFP32({static_cast<size_t>(m), static_cast<size_t>(n_v)});

    FusedQKVGEMMStage::Params params;
    params.input = input.get();
    params.output_q = output_q.get();
    params.output_k = output_k.get();
    params.output_v = output_v.get();
    params.m = m;
    params.k = k;
    params.n_q = n_q;
    params.n_k = n_k;
    params.n_v = n_v;

    FusedQKVGEMMStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "m"));
    EXPECT_TRUE(hasScalar(info, "k"));
    EXPECT_TRUE(hasScalar(info, "n_q"));
    EXPECT_TRUE(hasScalar(info, "n_k"));
    EXPECT_TRUE(hasScalar(info, "n_v"));

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "output_q"));
    EXPECT_TRUE(hasOutput(info, "output_k"));
    EXPECT_TRUE(hasOutput(info, "output_v"));
}

// =============================================================================
// FusedGateUpGEMMStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, FusedGateUpGEMMStage_GetDumpInfo)
{
    int seq_len = 16;
    int d_model = 256;
    int intermediate_dim = 512;

    int m = seq_len;
    int k = d_model;

    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(m), static_cast<size_t>(k)});
    auto output_gate = TestTensorFactory::createFP32({static_cast<size_t>(m), static_cast<size_t>(intermediate_dim)});
    auto output_up = TestTensorFactory::createFP32({static_cast<size_t>(m), static_cast<size_t>(intermediate_dim)});

    FusedGateUpGEMMStage::Params params;
    params.input = input.get();
    params.output_gate = output_gate.get();
    params.output_up = output_up.get();
    params.m = m;
    params.k = k;
    params.n_gate = intermediate_dim;
    params.n_up = intermediate_dim;

    FusedGateUpGEMMStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "m"));
    EXPECT_TRUE(hasScalar(info, "k"));
    EXPECT_TRUE(hasScalar(info, "n_gate"));
    EXPECT_TRUE(hasScalar(info, "n_up"));

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "output_gate"));
    EXPECT_TRUE(hasOutput(info, "output_up"));
}

// =============================================================================
// RMSNormStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, RMSNormStage_GetDumpInfo)
{
    int seq_len = 16;
    int hidden_size = 256;

    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)});
    auto gamma = TestTensorFactory::createFP32Ones({static_cast<size_t>(hidden_size)});

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.seq_len = seq_len;
    params.eps = 1e-5f;

    RMSNormStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "seq_len"));

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "output"));
}

// =============================================================================
// RoPEStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, RoPEStage_GetDumpInfo)
{
    int seq_len = 16;
    int n_heads = 4;
    int head_dim = 64;

    auto Q = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});
    auto K = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_heads * head_dim)});

    RoPEStage::Params params;
    params.Q = Q.get();
    params.K = K.get();
    params.seq_len = seq_len;
    params.n_heads = n_heads;
    params.head_dim = head_dim;
    params.pos_offset = 0;
    params.theta_base = 10000.0f;

    RoPEStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "seq_len"));
    EXPECT_TRUE(hasScalar(info, "n_heads"));
    EXPECT_TRUE(hasScalar(info, "head_dim"));

    // Check outputs (Q and K are modified in-place)
    EXPECT_TRUE(hasOutput(info, "Q"));
    EXPECT_TRUE(hasOutput(info, "K"));
}

// =============================================================================
// SwiGLUStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, SwiGLUStage_GetDumpInfo)
{
    int seq_len = 16;
    int intermediate_dim = 512;

    auto gate = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(intermediate_dim)});
    auto up = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(intermediate_dim)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(intermediate_dim)});

    SwiGLUStage::Params params;
    params.gate = gate.get();
    params.up = up.get();
    params.output = output.get();
    params.seq_len = seq_len;

    SwiGLUStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "seq_len"));

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "output"));
}

// =============================================================================
// ResidualAddStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, ResidualAddStage_GetDumpInfo)
{
    int seq_len = 16;
    int hidden_size = 256;
    size_t num_elements = static_cast<size_t>(seq_len * hidden_size);

    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)});
    auto residual = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(hidden_size)});

    ResidualAddStage::Params params;
    params.input = input.get();
    params.residual = residual.get();
    params.output = output.get();
    params.num_elements = num_elements;

    ResidualAddStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "num_elements"));
    EXPECT_EQ(getScalarInt(info, "num_elements"), static_cast<int64_t>(num_elements));

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "output"));
}

// =============================================================================
// AllreduceStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, AllreduceStage_GetDumpInfo)
{
    size_t count = 1024;

    auto buffer = TestTensorFactory::createFP32Random({count});

    AllreduceStage::Params params;
    params.buffer = buffer.get();
    params.mpi_comm = nullptr; // No actual MPI for unit test
    params.count = count;

    AllreduceStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "count"));
    EXPECT_EQ(getScalarInt(info, "count"), static_cast<int64_t>(count));

    // Allreduce has in-place buffer (input == output)
    EXPECT_TRUE(hasInput(info, "buffer") || hasOutput(info, "buffer"));
}

// =============================================================================
// AllGatherStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, AllGatherStage_GetDumpInfo)
{
    int seq_len = 16;
    int local_cols = 64;
    int world_size = 2;
    int full_cols = local_cols * world_size;

    auto local_input = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(local_cols)});
    auto full_output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(full_cols)});

    AllGatherStage::Params params;
    params.local_input = local_input.get();
    params.full_output = full_output.get();
    params.mpi_comm = nullptr;
    params.world_size = world_size;
    params.actual_seq_len = static_cast<size_t>(seq_len);

    AllGatherStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "world_size"));
    EXPECT_TRUE(hasScalar(info, "actual_seq_len"));
    EXPECT_EQ(getScalarInt(info, "world_size"), world_size);
    EXPECT_EQ(getScalarInt(info, "actual_seq_len"), seq_len);

    // Check inputs and outputs
    EXPECT_TRUE(hasInput(info, "local_input"));
    EXPECT_TRUE(hasOutput(info, "full_output"));
}

// =============================================================================
// MoERouterStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, MoERouterStage_GetDumpInfo)
{
    int seq_len = 16;
    int d_model = 256;
    int num_experts = 8;

    auto hidden = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto gate_weights = TestTensorFactory::createFP32Random({static_cast<size_t>(d_model), static_cast<size_t>(num_experts)});
    auto router_logits = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(num_experts)});

    MoERouterStage::Params params;
    params.hidden = hidden.get();
    params.gate_weights = gate_weights.get();
    params.router_logits = router_logits.get();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.num_experts = num_experts;

    MoERouterStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "seq_len"));
    EXPECT_TRUE(hasScalar(info, "d_model"));
    EXPECT_TRUE(hasScalar(info, "num_experts"));

    // Check inputs/outputs
    EXPECT_TRUE(hasInput(info, "hidden"));
    EXPECT_TRUE(hasWeight(info, "gate_weights"));
    EXPECT_TRUE(hasOutput(info, "router_logits"));
}

// =============================================================================
// MoEExpertStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, MoEExpertStage_GetDumpInfo)
{
    int d_model = 256;
    int intermediate_dim = 512;
    int num_tokens = 8;
    int expert_id = 3;

    auto input = TestTensorFactory::createFP32Random({static_cast<size_t>(num_tokens), static_cast<size_t>(d_model)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(num_tokens), static_cast<size_t>(d_model)});
    auto gate_w = TestTensorFactory::createFP32Random({static_cast<size_t>(d_model), static_cast<size_t>(intermediate_dim)});
    auto up_w = TestTensorFactory::createFP32Random({static_cast<size_t>(d_model), static_cast<size_t>(intermediate_dim)});
    auto down_w = TestTensorFactory::createFP32Random({static_cast<size_t>(intermediate_dim), static_cast<size_t>(d_model)});

    MoEExpertStage::Params params;
    params.expert_id = expert_id;
    params.input = input.get();
    params.output = output.get();
    params.gate_w = gate_w.get();
    params.up_w = up_w.get();
    params.down_w = down_w.get();
    params.d_model = d_model;
    params.intermediate_dim = intermediate_dim;

    MoEExpertStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "expert_id"));
    EXPECT_TRUE(hasScalar(info, "d_model"));
    EXPECT_TRUE(hasScalar(info, "intermediate_dim"));
    EXPECT_EQ(getScalarInt(info, "expert_id"), expert_id);

    // Check inputs/weights/outputs
    EXPECT_TRUE(hasInput(info, "input"));
    EXPECT_TRUE(hasWeight(info, "gate_w"));
    EXPECT_TRUE(hasWeight(info, "up_w"));
    EXPECT_TRUE(hasWeight(info, "down_w"));
    EXPECT_TRUE(hasOutput(info, "output"));
}

// =============================================================================
// MoECombineStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, MoECombineStage_GetDumpInfo)
{
    int seq_len = 16;
    int d_model = 256;
    int top_k = 2;

    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    // Create some expert outputs
    std::vector<std::unique_ptr<FP32Tensor>> expert_outputs_storage;
    std::vector<const TensorBase *> expert_outputs;
    for (int i = 0; i < top_k; ++i)
    {
        auto expert_out = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
        expert_outputs.push_back(expert_out.get());
        expert_outputs_storage.push_back(std::move(expert_out));
    }

    MoECombineStage::Params params;
    params.expert_outputs = &expert_outputs;
    params.output = output.get();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.top_k = top_k;

    MoECombineStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "seq_len"));
    EXPECT_TRUE(hasScalar(info, "d_model"));
    EXPECT_TRUE(hasScalar(info, "top_k"));
    EXPECT_EQ(getScalarInt(info, "top_k"), top_k);

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "output"));

    // Check expert inputs (named expert_output_0, expert_output_1, etc.)
    EXPECT_TRUE(hasInput(info, "expert_output_0"));
    EXPECT_TRUE(hasInput(info, "expert_output_1"));
}

// =============================================================================
// KVCacheAppendStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, KVCacheAppendStage_GetDumpInfo)
{
    int seq_len = 8;
    int n_kv_heads = 4;
    int head_dim = 64;

    auto K = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto V = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(n_kv_heads * head_dim)});

    KVCacheAppendStage::Params params;
    params.K = K.get();
    params.V = V.get();
    params.kv_cache = nullptr; // No actual cache for unit test
    params.layer_idx = 5;
    params.seq_len = seq_len;
    params.num_tokens = seq_len;

    KVCacheAppendStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "layer_idx"));
    EXPECT_TRUE(hasScalar(info, "seq_len"));
    EXPECT_TRUE(hasScalar(info, "num_tokens"));
    EXPECT_EQ(getScalarInt(info, "layer_idx"), 5);

    // Check inputs
    EXPECT_TRUE(hasInput(info, "K"));
    EXPECT_TRUE(hasInput(info, "V"));
}

// =============================================================================
// KVCacheGatherStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, KVCacheGatherStage_GetDumpInfo)
{
    int batch_size = 2;
    int n_kv_heads = 4;
    int head_dim = 64;
    int max_kv_len = 128;

    auto out_K = TestTensorFactory::createFP32({static_cast<size_t>(batch_size * max_kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto out_V = TestTensorFactory::createFP32({static_cast<size_t>(batch_size * max_kv_len), static_cast<size_t>(n_kv_heads * head_dim)});

    KVCacheGatherStage::Params params;
    params.out_K = out_K.get();
    params.out_V = out_V.get();
    params.kv_cache = nullptr;
    params.layer_idx = 3;
    params.batch_size = batch_size;

    KVCacheGatherStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "layer_idx"));
    EXPECT_TRUE(hasScalar(info, "batch_size"));
    EXPECT_EQ(getScalarInt(info, "layer_idx"), 3);
    EXPECT_EQ(getScalarInt(info, "batch_size"), batch_size);

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "gathered_K"));
    EXPECT_TRUE(hasOutput(info, "gathered_V"));
}

// =============================================================================
// AttentionComputeStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, AttentionComputeStage_GetDumpInfo)
{
    int seq_len = 16;
    int kv_len = 32;
    int n_heads = 8;
    int n_kv_heads = 4;
    int head_dim = 64;
    size_t hidden_size = n_heads * head_dim;

    auto Q = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), hidden_size});
    auto K = TestTensorFactory::createFP32Random({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto V = TestTensorFactory::createFP32Random({static_cast<size_t>(kv_len), static_cast<size_t>(n_kv_heads * head_dim)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), hidden_size});

    AttentionComputeStage::Params params;
    params.Q = Q.get();
    params.K = K.get();
    params.V = V.get();
    params.output = output.get();
    params.batch_size = 1;
    params.seq_len = seq_len;
    params.kv_len = kv_len;
    params.n_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.head_dim = head_dim;
    params.causal = true;

    AttentionComputeStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "seq_len"));
    EXPECT_TRUE(hasScalar(info, "kv_len"));
    EXPECT_TRUE(hasScalar(info, "n_heads"));
    EXPECT_TRUE(hasScalar(info, "n_kv_heads"));
    EXPECT_TRUE(hasScalar(info, "head_dim"));
    EXPECT_EQ(getScalarInt(info, "seq_len"), seq_len);
    EXPECT_EQ(getScalarInt(info, "kv_len"), kv_len);

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "output"));
}

// =============================================================================
// EmbeddingStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, EmbeddingStage_GetDumpInfo)
{
    int seq_len = 16;
    int vocab_size = 1024;
    int d_model = 256;

    auto embed_table = TestTensorFactory::createFP32Random({static_cast<size_t>(vocab_size), static_cast<size_t>(d_model)});
    auto output = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});

    std::vector<int> tokens(seq_len);
    for (int i = 0; i < seq_len; ++i)
    {
        tokens[i] = i % vocab_size;
    }

    EmbeddingStage::Params params;
    params.token_ids = tokens.data();
    params.embed_table = embed_table.get();
    params.output = output.get();
    params.num_tokens = seq_len;
    params.d_model = d_model;
    params.vocab_size = vocab_size;

    EmbeddingStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "num_tokens"));
    EXPECT_TRUE(hasScalar(info, "d_model"));
    EXPECT_TRUE(hasScalar(info, "vocab_size"));
    EXPECT_EQ(getScalarInt(info, "num_tokens"), seq_len);
    EXPECT_EQ(getScalarInt(info, "d_model"), d_model);

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "embeddings"));
}

// =============================================================================
// LMHeadStage Tests
// =============================================================================

TEST_F(StageDumpInfoTest, LMHeadStage_GetDumpInfo)
{
    int seq_len = 16;
    int d_model = 256;
    int vocab_size = 1024;

    auto hidden_states = TestTensorFactory::createFP32Random({static_cast<size_t>(seq_len), static_cast<size_t>(d_model)});
    auto lm_head_weight = TestTensorFactory::createFP32Random({static_cast<size_t>(vocab_size), static_cast<size_t>(d_model)});
    auto logits = TestTensorFactory::createFP32({static_cast<size_t>(seq_len), static_cast<size_t>(vocab_size)});

    LMHeadStage::Params params;
    params.hidden_states = hidden_states.get();
    params.lm_head_weight = lm_head_weight.get();
    params.logits = logits.get();
    params.seq_len = seq_len;
    params.d_model = d_model;
    params.vocab_size = vocab_size;

    LMHeadStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Check scalars
    EXPECT_TRUE(hasScalar(info, "seq_len"));
    EXPECT_TRUE(hasScalar(info, "d_model"));
    EXPECT_TRUE(hasScalar(info, "vocab_size"));

    // Check outputs
    EXPECT_TRUE(hasOutput(info, "logits"));
}

// =============================================================================
// Empty/Null Parameter Tests
// =============================================================================

TEST_F(StageDumpInfoTest, GEMMStage_GetDumpInfo_NullBuffers)
{
    // Test that getDumpInfo handles null buffers gracefully
    GEMMStage::Params params;
    params.A = nullptr;
    params.B = nullptr;
    params.C = nullptr;
    params.m = 0;
    params.n = 0;
    params.k = 0;

    GEMMStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // Should still have scalar info
    EXPECT_TRUE(hasScalar(info, "m"));
    EXPECT_TRUE(hasScalar(info, "n"));
    EXPECT_TRUE(hasScalar(info, "k"));

    // Output should be missing or empty since C is null
    // (depends on implementation - just verify no crash)
    EXPECT_NO_THROW({ auto size = info.outputs.size(); (void)size; });
}

TEST_F(StageDumpInfoTest, AllreduceStage_GetDumpInfo_NullBuffer)
{
    AllreduceStage::Params params;
    params.buffer = nullptr;
    params.mpi_comm = nullptr;
    params.count = 100;

    AllreduceStage stage(params);
    StageDumpInfo info = stage.getDumpInfo();

    // When buffer is null, getDumpInfo() doesn't add anything
    // This verifies no crash occurs with null buffer
    EXPECT_NO_THROW({ auto size = info.scalars.size(); (void)size; });
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
