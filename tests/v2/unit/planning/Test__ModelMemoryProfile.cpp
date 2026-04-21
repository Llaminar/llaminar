#include <gtest/gtest.h>
#include "planning/ModelMemoryProfile.h"
#include "loaders/ModelLoader.h"

using namespace llaminar2;

namespace
{

// Helper: create a minimal GGUFModel resembling Qwen2.5-0.5B (2 layers only)
// block_count is set to 2 to match the actual tensor count.
// Tests that need 24 layers should use createTestModel(24).
GGUFModel createTestModel(int num_layers = 2)
{
    GGUFModel model;
    model.architecture = "qwen2";
    model.block_count = num_layers;
    model.embedding_length = 896;
    model.head_count = 14;
    model.head_count_kv = 2;
    model.key_length = 64;
    model.value_length = 64;
    model.vocab_size = 151936;
    model.context_length = 32768;

    // Add embedding tensor
    GGUFTensorInfo embd;
    embd.name = "token_embd.weight";
    embd.dimensions = {896, 151936};
    embd.type = GGUFTensorType::Q8_0;
    embd.size_bytes = 896ULL * 151936 * 34 / 32;  // Q8_0: 34 bytes per 32 elements
    embd.offset = 0;
    model.tensors.push_back(embd);

    // Add output norm
    GGUFTensorInfo out_norm;
    out_norm.name = "output_norm.weight";
    out_norm.dimensions = {896};
    out_norm.type = GGUFTensorType::F32;
    out_norm.size_bytes = 896 * 4;
    out_norm.offset = 0;
    model.tensors.push_back(out_norm);

    // Add LM head (output.weight)
    GGUFTensorInfo lm_head;
    lm_head.name = "output.weight";
    lm_head.dimensions = {896, 151936};
    lm_head.type = GGUFTensorType::Q8_0;
    lm_head.size_bytes = 896ULL * 151936 * 34 / 32;
    lm_head.offset = 0;
    model.tensors.push_back(lm_head);

    // Add 'num_layers' layers worth of tensors
    for (int layer = 0; layer < num_layers; ++layer)
    {
        std::string prefix = "blk." + std::to_string(layer) + ".";

        // attn_q
        GGUFTensorInfo q;
        q.name = prefix + "attn_q.weight";
        q.dimensions = {896, 896};
        q.type = GGUFTensorType::Q8_0;
        q.size_bytes = 896ULL * 896 * 34 / 32;
        q.offset = 0;
        model.tensors.push_back(q);

        // attn_k
        GGUFTensorInfo k;
        k.name = prefix + "attn_k.weight";
        k.dimensions = {128, 896};
        k.type = GGUFTensorType::Q8_0;
        k.size_bytes = 128ULL * 896 * 34 / 32;
        k.offset = 0;
        model.tensors.push_back(k);

        // attn_v
        GGUFTensorInfo v;
        v.name = prefix + "attn_v.weight";
        v.dimensions = {128, 896};
        v.type = GGUFTensorType::Q8_0;
        v.size_bytes = 128ULL * 896 * 34 / 32;
        v.offset = 0;
        model.tensors.push_back(v);

        // attn_output
        GGUFTensorInfo wo;
        wo.name = prefix + "attn_output.weight";
        wo.dimensions = {896, 896};
        wo.type = GGUFTensorType::Q8_0;
        wo.size_bytes = 896ULL * 896 * 34 / 32;
        wo.offset = 0;
        model.tensors.push_back(wo);

        // ffn_gate
        GGUFTensorInfo gate;
        gate.name = prefix + "ffn_gate.weight";
        gate.dimensions = {4864, 896};
        gate.type = GGUFTensorType::Q8_0;
        gate.size_bytes = 4864ULL * 896 * 34 / 32;
        gate.offset = 0;
        model.tensors.push_back(gate);

        // ffn_up
        GGUFTensorInfo up;
        up.name = prefix + "ffn_up.weight";
        up.dimensions = {4864, 896};
        up.type = GGUFTensorType::Q8_0;
        up.size_bytes = 4864ULL * 896 * 34 / 32;
        up.offset = 0;
        model.tensors.push_back(up);

        // ffn_down
        GGUFTensorInfo down;
        down.name = prefix + "ffn_down.weight";
        down.dimensions = {896, 4864};
        down.type = GGUFTensorType::Q8_0;
        down.size_bytes = 896ULL * 4864 * 34 / 32;
        down.offset = 0;
        model.tensors.push_back(down);

        // attn_norm
        GGUFTensorInfo attn_norm;
        attn_norm.name = prefix + "attn_norm.weight";
        attn_norm.dimensions = {896};
        attn_norm.type = GGUFTensorType::F32;
        attn_norm.size_bytes = 896 * 4;
        attn_norm.offset = 0;
        model.tensors.push_back(attn_norm);

        // ffn_norm
        GGUFTensorInfo ffn_norm;
        ffn_norm.name = prefix + "ffn_norm.weight";
        ffn_norm.dimensions = {896};
        ffn_norm.type = GGUFTensorType::F32;
        ffn_norm.size_bytes = 896 * 4;
        ffn_norm.offset = 0;
        model.tensors.push_back(ffn_norm);
    }

    return model;
}

} // anonymous namespace

TEST(Test__ModelMemoryProfile, FromGGUF_ExtractsArchitecture)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    EXPECT_EQ(profile.architecture, "qwen2");
    EXPECT_EQ(profile.n_layers, 2);
    EXPECT_EQ(profile.d_model, 896);
    EXPECT_EQ(profile.n_heads, 14);
    EXPECT_EQ(profile.n_kv_heads, 2);
    EXPECT_EQ(profile.head_dim, 64);
    EXPECT_EQ(profile.vocab_size, 151936);
    EXPECT_EQ(profile.max_seq_len, 32768);
}

TEST(Test__ModelMemoryProfile, FromGGUF_SumsTensorBytes)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    EXPECT_GT(profile.total_native_bytes, 0u);

    // Verify sum matches individual tensors
    size_t manual_sum = 0;
    for (const auto& t : profile.tensors)
    {
        manual_sum += t.native_bytes;
    }
    EXPECT_EQ(profile.total_native_bytes, manual_sum);
}

TEST(Test__ModelMemoryProfile, FromGGUF_ParsesLayerIndices)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    int layer0_count = 0, layer1_count = 0, non_layer_count = 0;
    for (const auto& t : profile.tensors)
    {
        if (t.layer_index == 0) layer0_count++;
        else if (t.layer_index == 1) layer1_count++;
        else if (t.layer_index == -1) non_layer_count++;
    }

    // Each layer has 9 tensors (q, k, v, wo, gate, up, down, attn_norm, ffn_norm)
    EXPECT_EQ(layer0_count, 9);
    EXPECT_EQ(layer1_count, 9);
    // Non-layer: token_embd, output_norm, output.weight
    EXPECT_EQ(non_layer_count, 3);
}

TEST(Test__ModelMemoryProfile, FromGGUF_InfersDFFFromTensorShapes)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    // d_ff should be inferred from ffn_gate dimensions[0] = 4864
    EXPECT_EQ(profile.d_ff, 4864);
}

TEST(Test__ModelMemoryProfile, WeightBytesForLayers_FullRange)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    size_t layer0 = profile.layerWeightBytes(0);
    size_t layer1 = profile.layerWeightBytes(1);
    size_t both = profile.weightBytesForLayers(0, 1);

    EXPECT_GT(layer0, 0u);
    EXPECT_EQ(both, layer0 + layer1);
}

TEST(Test__ModelMemoryProfile, WeightBytesForLayers_PPSlice)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    size_t layer0_only = profile.weightBytesForLayers(0, 0);
    size_t layer1_only = profile.weightBytesForLayers(1, 1);

    EXPECT_EQ(layer0_only, profile.layerWeightBytes(0));
    EXPECT_EQ(layer1_only, profile.layerWeightBytes(1));
}

TEST(Test__ModelMemoryProfile, EmbeddingBytes_MatchesTokenEmbd)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    // token_embd.weight: 896 × 151936 Q8_0 = 896 * 151936 * 34 / 32
    size_t expected = 896ULL * 151936 * 34 / 32;
    EXPECT_EQ(profile.embeddingBytes(), expected);
}

TEST(Test__ModelMemoryProfile, LmHeadBytes_MatchesOutputWeight)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    size_t expected = 896ULL * 151936 * 34 / 32;
    EXPECT_EQ(profile.lmHeadBytes(), expected);
}

TEST(Test__ModelMemoryProfile, NormBytes_CountsNonLayerNorms)
{
    auto model = createTestModel();
    auto profile = ModelMemoryProfile::fromGGUF(model);

    // output_norm.weight = 896 × 4 = 3584 bytes
    EXPECT_EQ(profile.normBytes(), 896u * 4);
}
