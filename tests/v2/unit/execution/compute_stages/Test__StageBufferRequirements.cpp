/**
 * @file Test__StageBufferRequirements.cpp
 * @brief Unit tests for IComputeStage::getBufferRequirements() implementations
 * @author David Sanftenberg
 * @date December 2025
 *
 * Tests that each stage correctly declares its buffer requirements,
 * enabling DeviceGraphBufferManager to allocate and manage buffers.
 */

#include <gtest/gtest.h>
#include "execution/compute_stages/ComputeStages.h"
#include "execution/debug/BufferRole.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__StageBufferRequirements : public ::testing::Test
{
protected:
    // Helper to create FP32 tensor for testing
    std::unique_ptr<FP32Tensor> createFP32Tensor(size_t rows, size_t cols)
    {
        return std::make_unique<FP32Tensor>(
            std::vector<size_t>{rows, cols}, DeviceId::cpu() // CPU device
        );
    }

    // Helper to create Q8_1 tensor for testing
    std::unique_ptr<Q8_1Tensor> createQ8_1Tensor(size_t rows, size_t cols)
    {
        return std::make_unique<Q8_1Tensor>(
            std::vector<size_t>{rows, cols}, DeviceId::cpu() // CPU device
        );
    }

    // Helper to find buffer by name
    const BufferDescriptor *findBuffer(const StageBufferRequirements &reqs,
                                       const std::string &name)
    {
        for (const auto &buf : reqs.buffers)
        {
            if (buf.name == name)
                return &buf;
        }
        return nullptr;
    }

    // Constants for test dimensions
    static constexpr size_t SEQ_LEN = 32;
    static constexpr size_t D_MODEL = 896;
    static constexpr size_t D_FF = 4864;
    static constexpr int N_HEADS = 14;
    static constexpr int N_KV_HEADS = 2;
    static constexpr int HEAD_DIM = 64;
};

// =============================================================================
// ResidualAddStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, ResidualAddStage_DeclaresCorrectBuffers)
{
    auto input = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto residual = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto output = createFP32Tensor(SEQ_LEN, D_MODEL);

    ResidualAddStage::Params params{
        .input = input.get(),
        .residual = residual.get(),
        .output = output.get(),
    };
    ResidualAddStage stage(params);

    auto reqs = stage.getBufferRequirements();

    EXPECT_EQ(reqs.buffers.size(), 3);

    // Both input and residual are INPUT
    auto *in_buf = findBuffer(reqs, "input");
    ASSERT_NE(in_buf, nullptr);
    EXPECT_EQ(in_buf->role, BufferRole::INPUT);

    auto *res_buf = findBuffer(reqs, "residual");
    ASSERT_NE(res_buf, nullptr);
    EXPECT_EQ(res_buf->role, BufferRole::INPUT);

    auto *out_buf = findBuffer(reqs, "output");
    ASSERT_NE(out_buf, nullptr);
    EXPECT_EQ(out_buf->role, BufferRole::OUTPUT);
}

// =============================================================================
// RMSNormStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, RMSNormStage_DeclaresCorrectBuffers)
{
    auto input = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto output = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto gamma = createFP32Tensor(1, D_MODEL);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-6f,
    };
    RMSNormStage stage(params);

    auto reqs = stage.getBufferRequirements();

    EXPECT_EQ(reqs.buffers.size(), 3);

    // Input is INPUT
    auto *in_buf = findBuffer(reqs, "input");
    ASSERT_NE(in_buf, nullptr);
    EXPECT_EQ(in_buf->role, BufferRole::INPUT);

    // Output is OUTPUT
    auto *out_buf = findBuffer(reqs, "output");
    ASSERT_NE(out_buf, nullptr);
    EXPECT_EQ(out_buf->role, BufferRole::OUTPUT);

    // Gamma is WEIGHT (always FP32)
    auto *gamma_buf = findBuffer(reqs, "gamma");
    ASSERT_NE(gamma_buf, nullptr);
    EXPECT_EQ(gamma_buf->role, BufferRole::WEIGHT);
    EXPECT_EQ(gamma_buf->tensor_type, BufferTensorType::FP32);
}

TEST_F(Test__StageBufferRequirements, RMSNormStage_ExplicitSeqLen_UsesOverride)
{
    auto input = createFP32Tensor(128, D_MODEL); // Pre-allocated larger buffer
    auto output = createFP32Tensor(128, D_MODEL);
    auto gamma = createFP32Tensor(1, D_MODEL);

    RMSNormStage::Params params{
        .input = input.get(),
        .output = output.get(),
        .gamma = gamma.get(),
        .eps = 1e-6f,
        .seq_len = 16, // Actual sequence length (smaller than buffer)
    };
    RMSNormStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // Should use explicit seq_len (16), not tensor rows (128)
    auto *in_buf = findBuffer(reqs, "input");
    ASSERT_NE(in_buf, nullptr);
    EXPECT_EQ(in_buf->shape, (std::vector<size_t>{16, D_MODEL}));
}

// =============================================================================
// RoPEStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, RoPEStage_DeclaresInoutBuffers)
{
    const size_t q_dim = N_HEADS * HEAD_DIM;
    const size_t k_dim = N_KV_HEADS * HEAD_DIM;

    auto Q = createFP32Tensor(SEQ_LEN, q_dim);
    auto K = createFP32Tensor(SEQ_LEN, k_dim);

    RoPEStage::Params params{
        .Q = Q.get(),
        .K = K.get(),
        .n_heads = N_HEADS,
        .n_kv_heads = N_KV_HEADS,
        .head_dim = HEAD_DIM,
    };
    RoPEStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // Both Q and K are INOUT (in-place modification)
    EXPECT_EQ(reqs.buffers.size(), 2);

    auto *q_buf = findBuffer(reqs, "Q");
    ASSERT_NE(q_buf, nullptr);
    EXPECT_EQ(q_buf->role, BufferRole::INOUT);
    EXPECT_EQ(q_buf->shape, (std::vector<size_t>{SEQ_LEN, q_dim}));

    auto *k_buf = findBuffer(reqs, "K");
    ASSERT_NE(k_buf, nullptr);
    EXPECT_EQ(k_buf->role, BufferRole::INOUT);
    EXPECT_EQ(k_buf->shape, (std::vector<size_t>{SEQ_LEN, k_dim}));
}

TEST_F(Test__StageBufferRequirements, RoPEStage_OnlyQ_SingleBuffer)
{
    const size_t q_dim = N_HEADS * HEAD_DIM;
    auto Q = createFP32Tensor(SEQ_LEN, q_dim);

    RoPEStage::Params params{
        .Q = Q.get(),
        .K = nullptr, // No K tensor
        .n_heads = N_HEADS,
        .n_kv_heads = N_KV_HEADS,
        .head_dim = HEAD_DIM,
    };
    RoPEStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // Only Q buffer
    EXPECT_EQ(reqs.buffers.size(), 1);
    EXPECT_NE(findBuffer(reqs, "Q"), nullptr);
    EXPECT_EQ(findBuffer(reqs, "K"), nullptr);
}

// =============================================================================
// GEMMStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, GEMMStage_DeclaresCorrectBuffers)
{
    auto A = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto B = createFP32Tensor(D_MODEL, D_FF); // Weight
    auto C = createFP32Tensor(SEQ_LEN, D_FF);

    GEMMStage::Params params{
        .A = A.get(),
        .B = B.get(),
        .C = C.get(),
        .m = static_cast<int>(SEQ_LEN),
        .n = static_cast<int>(D_FF),
        .k = static_cast<int>(D_MODEL),
    };
    GEMMStage stage(params);

    auto reqs = stage.getBufferRequirements();

    EXPECT_EQ(reqs.buffers.size(), 3);

    // A is INPUT (activation)
    auto *a_buf = findBuffer(reqs, "A");
    ASSERT_NE(a_buf, nullptr);
    EXPECT_EQ(a_buf->role, BufferRole::INPUT);

    // B is WEIGHT
    auto *b_buf = findBuffer(reqs, "B");
    ASSERT_NE(b_buf, nullptr);
    EXPECT_EQ(b_buf->role, BufferRole::WEIGHT);

    // C is OUTPUT
    auto *c_buf = findBuffer(reqs, "C");
    ASSERT_NE(c_buf, nullptr);
    EXPECT_EQ(c_buf->role, BufferRole::OUTPUT);
}

TEST_F(Test__StageBufferRequirements, GEMMStage_WithBias_IncludesBiasBuffer)
{
    auto A = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto B = createFP32Tensor(D_MODEL, D_FF);
    auto C = createFP32Tensor(SEQ_LEN, D_FF);
    std::vector<float> bias(D_FF, 0.1f);

    GEMMStage::Params params{
        .A = A.get(),
        .B = B.get(),
        .C = C.get(),
        .m = static_cast<int>(SEQ_LEN),
        .n = static_cast<int>(D_FF),
        .k = static_cast<int>(D_MODEL),
        .bias = bias.data(),
    };
    GEMMStage stage(params);

    auto reqs = stage.getBufferRequirements();

    EXPECT_EQ(reqs.buffers.size(), 4);

    auto *bias_buf = findBuffer(reqs, "bias");
    ASSERT_NE(bias_buf, nullptr);
    EXPECT_EQ(bias_buf->role, BufferRole::WEIGHT);
    EXPECT_EQ(bias_buf->tensor_type, BufferTensorType::FP32);
}

// =============================================================================
// FusedQKVGEMMStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, FusedQKVGEMMStage_DeclaresAllOutputs)
{
    const size_t q_out = N_HEADS * HEAD_DIM;
    const size_t kv_out = N_KV_HEADS * HEAD_DIM;

    auto input = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto wq = createFP32Tensor(D_MODEL, q_out);
    auto wk = createFP32Tensor(D_MODEL, kv_out);
    auto wv = createFP32Tensor(D_MODEL, kv_out);
    auto output_q = createFP32Tensor(SEQ_LEN, q_out);
    auto output_k = createFP32Tensor(SEQ_LEN, kv_out);
    auto output_v = createFP32Tensor(SEQ_LEN, kv_out);

    FusedQKVGEMMStage::Params params{
        .input = input.get(),
        .m = static_cast<int>(SEQ_LEN),
        .k = static_cast<int>(D_MODEL),
        .wq = wq.get(),
        .output_q = output_q.get(),
        .n_q = static_cast<int>(q_out),
        .wk = wk.get(),
        .output_k = output_k.get(),
        .n_k = static_cast<int>(kv_out),
        .wv = wv.get(),
        .output_v = output_v.get(),
        .n_v = static_cast<int>(kv_out),
    };
    FusedQKVGEMMStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 1 input + 3 weights + 3 outputs = 7
    EXPECT_EQ(reqs.buffers.size(), 7);

    // Check all outputs exist
    EXPECT_NE(findBuffer(reqs, "output_q"), nullptr);
    EXPECT_NE(findBuffer(reqs, "output_k"), nullptr);
    EXPECT_NE(findBuffer(reqs, "output_v"), nullptr);

    // Check all weights exist
    EXPECT_NE(findBuffer(reqs, "wq"), nullptr);
    EXPECT_NE(findBuffer(reqs, "wk"), nullptr);
    EXPECT_NE(findBuffer(reqs, "wv"), nullptr);
}

// =============================================================================
// FusedGateUpGEMMStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, FusedGateUpGEMMStage_DeclaresGateAndUp)
{
    auto input = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto w_gate = createFP32Tensor(D_MODEL, D_FF);
    auto w_up = createFP32Tensor(D_MODEL, D_FF);
    auto output_gate = createFP32Tensor(SEQ_LEN, D_FF);
    auto output_up = createFP32Tensor(SEQ_LEN, D_FF);

    FusedGateUpGEMMStage::Params params{
        .input = input.get(),
        .m = static_cast<int>(SEQ_LEN),
        .k = static_cast<int>(D_MODEL),
        .w_gate = w_gate.get(),
        .output_gate = output_gate.get(),
        .n_gate = static_cast<int>(D_FF),
        .w_up = w_up.get(),
        .output_up = output_up.get(),
        .n_up = static_cast<int>(D_FF),
    };
    FusedGateUpGEMMStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 1 input + 2 weights + 2 outputs = 5
    EXPECT_EQ(reqs.buffers.size(), 5);

    // Check outputs
    auto *gate_out = findBuffer(reqs, "output_gate");
    ASSERT_NE(gate_out, nullptr);
    EXPECT_EQ(gate_out->role, BufferRole::OUTPUT);
    EXPECT_EQ(gate_out->shape, (std::vector<size_t>{SEQ_LEN, D_FF}));

    auto *up_out = findBuffer(reqs, "output_up");
    ASSERT_NE(up_out, nullptr);
    EXPECT_EQ(up_out->role, BufferRole::OUTPUT);
}

// =============================================================================
// Buffer Size Calculation Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, BufferDescriptor_SizeBytes_FP32)
{
    BufferDescriptor desc{
        .name = "test",
        .role = BufferRole::OUTPUT,
        .shape = {32, 896},
        .tensor_type = BufferTensorType::FP32,
    };

    EXPECT_EQ(desc.numel(), 32 * 896);
    EXPECT_EQ(desc.sizeBytes(), 32 * 896 * 4); // 4 bytes per FP32
}

TEST_F(Test__StageBufferRequirements, BufferDescriptor_SizeBytes_Q8_1)
{
    BufferDescriptor desc{
        .name = "test",
        .role = BufferRole::OUTPUT,
        .shape = {32, 896},
        .tensor_type = BufferTensorType::Q8_1,
    };

    // Q8_1: 36 bytes per 32 elements
    size_t elements = 32 * 896;
    size_t blocks = (elements + 31) / 32;
    EXPECT_EQ(desc.sizeBytes(), blocks * 36);
}

TEST_F(Test__StageBufferRequirements, StageBufferRequirements_TotalBytes)
{
    StageBufferRequirements reqs;
    reqs.addInput("a", {32, 896}, BufferTensorType::FP32);
    reqs.addOutput("b", {32, 896}, BufferTensorType::FP32);
    reqs.addScratch("temp", {32, 32}, BufferTensorType::FP32);

    // Total = 2 * (32 * 896 * 4) + (32 * 32 * 4)
    size_t expected = 2 * (32 * 896 * 4) + (32 * 32 * 4);
    EXPECT_EQ(reqs.totalBytes(), expected);
}

TEST_F(Test__StageBufferRequirements, StageBufferRequirements_TotalInputBytes)
{
    StageBufferRequirements reqs;
    reqs.addInput("a", {32, 896}, BufferTensorType::FP32);
    reqs.addInput("b", {32, 896}, BufferTensorType::FP32);
    reqs.addOutput("c", {32, 896}, BufferTensorType::FP32);

    EXPECT_EQ(reqs.totalInputBytes(), 2 * (32 * 896 * 4));
    EXPECT_EQ(reqs.totalOutputBytes(), 32 * 896 * 4);
}

// =============================================================================
// EmbeddingStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, EmbeddingStage_DeclaresCorrectBuffers)
{
    static constexpr size_t VOCAB_SIZE = 1024;
    auto embed_table = createFP32Tensor(VOCAB_SIZE, D_MODEL);
    auto output = createFP32Tensor(SEQ_LEN, D_MODEL);
    std::vector<int> token_ids(SEQ_LEN, 0);

    EmbeddingStage::Params params{
        .embed_table = embed_table.get(),
        .token_ids = token_ids.data(),
        .output = output.get(),
        .num_tokens = static_cast<int>(SEQ_LEN),
        .d_model = static_cast<int>(D_MODEL),
        .vocab_size = static_cast<int>(VOCAB_SIZE),
    };
    EmbeddingStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 1 weight (embed_table) + 1 output
    EXPECT_EQ(reqs.buffers.size(), 2);

    // Embedding table is WEIGHT
    auto *embed_buf = findBuffer(reqs, "embed_table");
    ASSERT_NE(embed_buf, nullptr);
    EXPECT_EQ(embed_buf->role, BufferRole::WEIGHT);
    EXPECT_EQ(embed_buf->shape, (std::vector<size_t>{VOCAB_SIZE, D_MODEL}));

    // Output is OUTPUT
    auto *out_buf = findBuffer(reqs, "output");
    ASSERT_NE(out_buf, nullptr);
    EXPECT_EQ(out_buf->role, BufferRole::OUTPUT);
    EXPECT_EQ(out_buf->shape, (std::vector<size_t>{SEQ_LEN, D_MODEL}));
}

TEST_F(Test__StageBufferRequirements, EmbeddingStage_NullTensors_ReturnsEmpty)
{
    EmbeddingStage::Params params{}; // All nullptrs
    EmbeddingStage stage(params);

    auto reqs = stage.getBufferRequirements();
    EXPECT_TRUE(reqs.buffers.empty());
}

// =============================================================================
// LMHeadStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, LMHeadStage_DeclaresCorrectBuffers)
{
    static constexpr size_t VOCAB_SIZE = 1024;
    auto hidden_states = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto lm_head_weight = createFP32Tensor(VOCAB_SIZE, D_MODEL);
    auto logits = createFP32Tensor(SEQ_LEN, VOCAB_SIZE);

    LMHeadStage::Params params{
        .hidden_states = hidden_states.get(),
        .lm_head_weight = lm_head_weight.get(),
        .logits = logits.get(),
        .seq_len = static_cast<int>(SEQ_LEN),
        .d_model = static_cast<int>(D_MODEL),
        .vocab_size = static_cast<int>(VOCAB_SIZE),
    };
    LMHeadStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 1 input + 1 weight + 1 output = 3
    EXPECT_EQ(reqs.buffers.size(), 3);

    // Hidden states is INPUT
    auto *hidden_buf = findBuffer(reqs, "hidden_states");
    ASSERT_NE(hidden_buf, nullptr);
    EXPECT_EQ(hidden_buf->role, BufferRole::INPUT);
    EXPECT_EQ(hidden_buf->shape, (std::vector<size_t>{SEQ_LEN, D_MODEL}));

    // LM head weight is WEIGHT
    auto *weight_buf = findBuffer(reqs, "lm_head_weight");
    ASSERT_NE(weight_buf, nullptr);
    EXPECT_EQ(weight_buf->role, BufferRole::WEIGHT);

    // Logits is OUTPUT
    auto *logits_buf = findBuffer(reqs, "logits");
    ASSERT_NE(logits_buf, nullptr);
    EXPECT_EQ(logits_buf->role, BufferRole::OUTPUT);
    EXPECT_EQ(logits_buf->shape, (std::vector<size_t>{SEQ_LEN, VOCAB_SIZE}));
}

TEST_F(Test__StageBufferRequirements, LMHeadStage_WithBias_IncludesBiasBuffer)
{
    static constexpr size_t VOCAB_SIZE = 1024;
    auto hidden_states = createFP32Tensor(SEQ_LEN, D_MODEL);
    auto lm_head_weight = createFP32Tensor(VOCAB_SIZE, D_MODEL);
    auto logits = createFP32Tensor(SEQ_LEN, VOCAB_SIZE);
    auto bias_tensor = createFP32Tensor(1, VOCAB_SIZE); // Create as TensorBase

    LMHeadStage::Params params{
        .hidden_states = hidden_states.get(),
        .lm_head_weight = lm_head_weight.get(),
        .logits = logits.get(),
        .seq_len = static_cast<int>(SEQ_LEN),
        .d_model = static_cast<int>(D_MODEL),
        .vocab_size = static_cast<int>(VOCAB_SIZE),
        .bias_tensor = bias_tensor.get(),
    };
    LMHeadStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 1 input + 2 weights (lm_head + bias) + 1 output = 4
    EXPECT_EQ(reqs.buffers.size(), 4);

    auto *bias_buf = findBuffer(reqs, "bias");
    ASSERT_NE(bias_buf, nullptr);
    EXPECT_EQ(bias_buf->role, BufferRole::WEIGHT);
    EXPECT_EQ(bias_buf->tensor_type, BufferTensorType::FP32);
}

TEST_F(Test__StageBufferRequirements, LMHeadStage_NullTensors_ReturnsEmpty)
{
    LMHeadStage::Params params{}; // All nullptrs
    LMHeadStage stage(params);

    auto reqs = stage.getBufferRequirements();
    EXPECT_TRUE(reqs.buffers.empty());
}

// =============================================================================
// AttentionComputeStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, AttentionComputeStage_DeclaresCorrectBuffers)
{
    static constexpr size_t N_HEADS = 14;
    static constexpr size_t N_KV_HEADS = 2;
    static constexpr size_t HEAD_DIM = 64;
    static constexpr size_t KV_LEN = 64; // May differ from SEQ_LEN in decode mode

    auto Q = createFP32Tensor(SEQ_LEN, N_HEADS * HEAD_DIM);
    auto K = createFP32Tensor(KV_LEN, N_KV_HEADS * HEAD_DIM);
    auto V = createFP32Tensor(KV_LEN, N_KV_HEADS * HEAD_DIM);
    auto output = createFP32Tensor(SEQ_LEN, N_HEADS * HEAD_DIM);

    AttentionComputeStage::Params params{
        .Q = Q.get(),
        .K = K.get(),
        .V = V.get(),
        .output = output.get(),
        .batch_size = 1,
        .seq_len = static_cast<int>(SEQ_LEN),
        .kv_len = static_cast<int>(KV_LEN),
        .n_heads = static_cast<int>(N_HEADS),
        .n_kv_heads = static_cast<int>(N_KV_HEADS),
        .head_dim = static_cast<int>(HEAD_DIM),
    };
    AttentionComputeStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 3 inputs (Q, K, V) + 1 output
    EXPECT_EQ(reqs.buffers.size(), 4);

    auto *q_buf = findBuffer(reqs, "Q");
    ASSERT_NE(q_buf, nullptr);
    EXPECT_EQ(q_buf->role, BufferRole::INPUT);
    EXPECT_EQ(q_buf->shape, (std::vector<size_t>{SEQ_LEN, N_HEADS * HEAD_DIM}));

    auto *k_buf = findBuffer(reqs, "K");
    ASSERT_NE(k_buf, nullptr);
    EXPECT_EQ(k_buf->role, BufferRole::INPUT);
    // K uses kv_len, not seq_len
    EXPECT_EQ(k_buf->shape, (std::vector<size_t>{KV_LEN, N_KV_HEADS * HEAD_DIM}));

    auto *out_buf = findBuffer(reqs, "output");
    ASSERT_NE(out_buf, nullptr);
    EXPECT_EQ(out_buf->role, BufferRole::OUTPUT);
}

TEST_F(Test__StageBufferRequirements, AttentionComputeStage_NullTensors_ReturnsEmpty)
{
    AttentionComputeStage::Params params{};
    AttentionComputeStage stage(params);

    auto reqs = stage.getBufferRequirements();
    EXPECT_TRUE(reqs.buffers.empty());
}

// =============================================================================
// KVCacheAppendStage Tests
// =============================================================================

TEST_F(Test__StageBufferRequirements, KVCacheAppendStage_DeclaresCorrectBuffers)
{
    static constexpr size_t N_KV_HEADS = 2;
    static constexpr size_t HEAD_DIM = 64;

    auto K = createFP32Tensor(SEQ_LEN, N_KV_HEADS * HEAD_DIM);
    auto V = createFP32Tensor(SEQ_LEN, N_KV_HEADS * HEAD_DIM);

    KVCacheAppendStage::Params params{
        .K = K.get(),
        .V = V.get(),
        .kv_cache = nullptr, // Cache is external state
        .layer_idx = 0,
        .num_tokens = static_cast<int>(SEQ_LEN),
    };
    KVCacheAppendStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 2 inputs (K, V) - cache is external
    EXPECT_EQ(reqs.buffers.size(), 2);

    auto *k_buf = findBuffer(reqs, "K");
    ASSERT_NE(k_buf, nullptr);
    EXPECT_EQ(k_buf->role, BufferRole::INPUT);
    EXPECT_EQ(k_buf->shape, (std::vector<size_t>{SEQ_LEN, N_KV_HEADS * HEAD_DIM}));

    auto *v_buf = findBuffer(reqs, "V");
    ASSERT_NE(v_buf, nullptr);
    EXPECT_EQ(v_buf->role, BufferRole::INPUT);
}

TEST_F(Test__StageBufferRequirements, KVCacheAppendStage_NullTensors_ReturnsEmpty)
{
    KVCacheAppendStage::Params params{};
    KVCacheAppendStage stage(params);

    auto reqs = stage.getBufferRequirements();
    EXPECT_TRUE(reqs.buffers.empty());
}

// =============================================================================
// AllreduceStage Tests (TensorBase* based)
// =============================================================================

TEST_F(Test__StageBufferRequirements, AllreduceStage_DeclaresInoutBuffer)
{
    auto buffer = createFP32Tensor(SEQ_LEN, D_MODEL);

    AllreduceStage::Params params{
        .mpi_ctx = nullptr, // Null ctx is ok for buffer requirements
        .buffer = buffer.get(),
    };
    AllreduceStage stage(params);

    auto reqs = stage.getBufferRequirements();

    // 1 INOUT buffer (in-place allreduce)
    EXPECT_EQ(reqs.buffers.size(), 1);

    auto *buf = findBuffer(reqs, "buffer");
    ASSERT_NE(buf, nullptr);
    EXPECT_EQ(buf->role, BufferRole::INOUT);
    EXPECT_EQ(buf->shape, (std::vector<size_t>{SEQ_LEN, D_MODEL}));
    EXPECT_EQ(buf->tensor_type, BufferTensorType::FP32);
}

TEST_F(Test__StageBufferRequirements, AllreduceStage_NullBuffer_ReturnsEmpty)
{
    AllreduceStage::Params params{
        .mpi_ctx = nullptr,
        .buffer = nullptr,
    };
    AllreduceStage stage(params);

    auto reqs = stage.getBufferRequirements();
    EXPECT_TRUE(reqs.buffers.empty());
}
