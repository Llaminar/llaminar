/**
 * @file Test__HeterogeneousLayers.cpp
 * @brief Unit tests for Phase B: Heterogeneous Layer Templates
 *
 * Tests named_templates in GraphSchema, per-layer dispatch in GraphResolver,
 * partial_rotary_factor in RoPE, AttentionOutputGateStage, GatedRMSNormStage,
 * subtract_one norm mode, HybridCacheManager, and variable GraphConfig fields.
 */

#include <gtest/gtest.h>
#include <cmath>

#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/RoPEStage.h"
#include "execution/compute_stages/stages/RMSNormStage.h"
#include "execution/cache/HybridCacheManager.h"
#include "execution/cache/IHybridCacheManager.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "models/GraphTypes.h"
#include "memory/BufferId.h"
#include "tensors/Tensors.h"

using namespace llaminar2;

namespace
{
    // Shared CPU context for stage execution tests
    std::unique_ptr<IDeviceContext> makeCPUContext()
    {
        return std::make_unique<CPUDeviceContext>(DeviceId::cpu(), 1);
    }
} // namespace

// ============================================================================
// B1: GraphSchema named_templates
// ============================================================================

TEST(Test__HeterogeneousLayers, NamedTemplatesEmpty_UsesDefault)
{
    GraphSchema schema;
    schema.name = "test";
    schema.layer_template.attention_stages.push_back(
        StageSpec{.name = "default_attn_norm", .type = StageType::RMSNorm});

    // No named templates configured — should fall back to default
    const auto &tmpl = schema.getTemplateForLayer(0);
    ASSERT_EQ(tmpl.attention_stages.size(), 1u);
    EXPECT_EQ(tmpl.attention_stages[0].name, "default_attn_norm");
}

TEST(Test__HeterogeneousLayers, NamedTemplatesDispatch)
{
    GraphSchema schema;
    schema.name = "hybrid_test";

    // Default template (should not be used when named templates are configured)
    schema.layer_template.attention_stages.push_back(
        StageSpec{.name = "default_attn", .type = StageType::RMSNorm});

    // Named templates
    LayerTemplate full_attn;
    full_attn.attention_stages.push_back(
        StageSpec{.name = "full_attn_norm", .type = StageType::RMSNorm});
    full_attn.attention_stages.push_back(
        StageSpec{.name = "qkv_proj", .type = StageType::FusedQKVGEMM});

    LayerTemplate gdn;
    gdn.attention_stages.push_back(
        StageSpec{.name = "gdn_norm", .type = StageType::RMSNorm});
    gdn.attention_stages.push_back(
        StageSpec{.name = "gdn_proj", .type = StageType::GEMM});

    schema.named_templates["full_attention"] = full_attn;
    schema.named_templates["gdn"] = gdn;
    schema.layer_template_names = {"full_attention", "gdn", "full_attention", "gdn"};

    // Layer 0: full_attention
    const auto &tmpl0 = schema.getTemplateForLayer(0);
    ASSERT_EQ(tmpl0.attention_stages.size(), 2u);
    EXPECT_EQ(tmpl0.attention_stages[0].name, "full_attn_norm");

    // Layer 1: gdn
    const auto &tmpl1 = schema.getTemplateForLayer(1);
    ASSERT_EQ(tmpl1.attention_stages.size(), 2u);
    EXPECT_EQ(tmpl1.attention_stages[0].name, "gdn_norm");

    // Layer 2: full_attention again
    const auto &tmpl2 = schema.getTemplateForLayer(2);
    EXPECT_EQ(tmpl2.attention_stages[0].name, "full_attn_norm");
}

TEST(Test__HeterogeneousLayers, NamedTemplatesFallbackOnMissingName)
{
    GraphSchema schema;
    schema.name = "fallback_test";
    schema.layer_template.attention_stages.push_back(
        StageSpec{.name = "default_attn", .type = StageType::RMSNorm});

    LayerTemplate custom;
    custom.attention_stages.push_back(
        StageSpec{.name = "custom_attn", .type = StageType::RMSNorm});
    schema.named_templates["custom"] = custom;

    // layer_template_names references a name that doesn't exist in named_templates
    schema.layer_template_names = {"nonexistent"};

    // Should fall back to default layer_template
    const auto &tmpl = schema.getTemplateForLayer(0);
    EXPECT_EQ(tmpl.attention_stages[0].name, "default_attn");
}

// ============================================================================
// B2: GraphResolver per-layer template dispatch
// ============================================================================

TEST(Test__HeterogeneousLayers, ResolverDispatchesPerLayerTemplate)
{
    GraphSchema schema;
    schema.name = "resolver_test";

    // Embedding (required by resolve)
    schema.embedding = StageSpec{
        .name = "embedding",
        .type = StageType::Embedding,
        .is_optional = true,
        .exec_policy_key = "exec_embedding"};

    // Named templates
    LayerTemplate full_attn;
    full_attn.attention_stages.push_back(
        StageSpec{.name = "attn_norm", .type = StageType::RMSNorm,
                  .is_optional = true, .exec_policy_key = "exec_rmsnorm"});

    LayerTemplate gdn;
    gdn.attention_stages.push_back(
        StageSpec{.name = "gdn_norm", .type = StageType::RMSNorm,
                  .is_optional = true, .exec_policy_key = "exec_rmsnorm"});

    schema.named_templates["full_attention"] = full_attn;
    schema.named_templates["gdn"] = gdn;
    schema.layer_template_names = {"full_attention", "gdn"};

    // Runtime config
    GraphResolverConfig runtime;
    runtime.n_layers = 2;
    runtime.d_model = 64;
    runtime.n_heads = 4;
    runtime.n_kv_heads = 4;
    runtime.head_dim = 16;
    runtime.rms_norm_eps = 1e-6f;

    TensorContext tensors;

    GraphResolver resolver;
    auto result = resolver.resolve(schema, runtime, tensors);

    // Should have resolved stages from both templates:
    // embedding (skipped if no policy) + layer0 attn_norm + layer1 gdn_norm
    bool found_attn_norm = false;
    bool found_gdn_norm = false;
    for (const auto &stage : result.stages)
    {
        if (stage.name == "layer0_attn_norm")
            found_attn_norm = true;
        if (stage.name == "layer1_gdn_norm")
            found_gdn_norm = true;
    }
    EXPECT_TRUE(found_attn_norm) << "Expected layer0 to use full_attention template";
    EXPECT_TRUE(found_gdn_norm) << "Expected layer1 to use gdn template";
}

// ============================================================================
// B5: partial_rotary_factor in StageSpec and resolver
// ============================================================================

TEST(Test__HeterogeneousLayers, PartialRotaryFactorInStageSpec)
{
    StageSpec rope_spec;
    rope_spec.name = "rope";
    rope_spec.type = StageType::RoPE;
    rope_spec.partial_rotary_factor = 0.5f;
    EXPECT_FLOAT_EQ(rope_spec.partial_rotary_factor.value(), 0.5f);
}

TEST(Test__HeterogeneousLayers, PartialRotaryFactorDefaultIsOne)
{
    RoPEStage::Params params;
    EXPECT_FLOAT_EQ(params.partial_rotary_factor, 1.0f);
}

TEST(Test__HeterogeneousLayers, PartialRotaryFactorInResolverConfig)
{
    GraphResolverConfig config;
    config.partial_rotary_factor = 0.5f;
    EXPECT_FLOAT_EQ(config.partial_rotary_factor, 0.5f);
}

// ============================================================================
// B6: AttentionOutputGateStage
// ============================================================================

TEST(Test__HeterogeneousLayers, AttentionOutputGateStage_SigmoidGating)
{
    // Create input = [1.0, 2.0, -1.0, 0.5]
    auto input = std::make_shared<FP32Tensor>(std::vector<size_t>{1, 4}, DeviceId::cpu());
    float *inp_data = input->mutable_data();
    inp_data[0] = 1.0f;
    inp_data[1] = 2.0f;
    inp_data[2] = -1.0f;
    inp_data[3] = 0.5f;

    // Create gate = [0.0, 1000.0, -1000.0, 0.0]
    // sigmoid(0) = 0.5, sigmoid(1000) ≈ 1.0, sigmoid(-1000) ≈ 0.0, sigmoid(0) = 0.5
    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, 4}, DeviceId::cpu());
    float *gate_data = gate->mutable_data();
    gate_data[0] = 0.0f;
    gate_data[1] = 1000.0f;
    gate_data[2] = -1000.0f;
    gate_data[3] = 0.0f;

    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{1, 4}, DeviceId::cpu());

    AttentionOutputGateStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.seq_len = 1;
    params.feature_dim = 4;

    auto ctx = makeCPUContext();
    AttentionOutputGateStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    const float *out = output->data();
    EXPECT_NEAR(out[0], 0.5f, 1e-4f);   // sigmoid(0) * 1.0 = 0.5
    EXPECT_NEAR(out[1], 2.0f, 1e-4f);   // sigmoid(1000) * 2.0 ≈ 2.0
    EXPECT_NEAR(out[2], 0.0f, 1e-4f);   // sigmoid(-1000) * -1.0 ≈ 0.0
    EXPECT_NEAR(out[3], 0.25f, 1e-4f);  // sigmoid(0) * 0.5 = 0.25
}

TEST(Test__HeterogeneousLayers, AttentionOutputGateStage_Type)
{
    AttentionOutputGateStage::Params params;
    params.seq_len = 1;
    params.feature_dim = 1;
    AttentionOutputGateStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::ATTENTION_OUTPUT_GATE);
}

TEST(Test__HeterogeneousLayers, AttentionOutputGateStage_InPlace)
{
    // Output can alias input
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{1, 2}, DeviceId::cpu());
    float *data = tensor->mutable_data();
    data[0] = 4.0f;
    data[1] = -2.0f;

    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, 2}, DeviceId::cpu());
    float *g = gate->mutable_data();
    g[0] = 0.0f;    // sigmoid(0) = 0.5
    g[1] = 0.0f;

    AttentionOutputGateStage::Params params;
    params.input = tensor.get();
    params.gate = gate.get();
    params.output = tensor.get();  // in-place
    params.seq_len = 1;
    params.feature_dim = 2;

    auto ctx = makeCPUContext();
    AttentionOutputGateStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    EXPECT_NEAR(tensor->data()[0], 2.0f, 1e-4f);   // 0.5 * 4.0
    EXPECT_NEAR(tensor->data()[1], -1.0f, 1e-4f);   // 0.5 * -2.0
}

// ============================================================================
// B7: GatedRMSNormStage
// ============================================================================

TEST(Test__HeterogeneousLayers, GatedRMSNormStage_Basic)
{
    const int d = 4;

    // Input: [1.0, 1.0, 1.0, 1.0] — RMS = 1.0
    auto input = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        input->mutable_data()[i] = 1.0f;

    // Gate: all 2.0
    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        gate->mutable_data()[i] = 2.0f;

    // Gamma: all 1.0
    auto gamma = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        gamma->mutable_data()[i] = 1.0f;

    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.seq_len = 1;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // RMS of [1,1,1,1] = sqrt(4/4 + eps) ≈ 1.0
    // Normalized = [1,1,1,1] / 1.0 = [1,1,1,1]
    // * gamma(1) * gate(2) = [2,2,2,2]
    for (int i = 0; i < d; ++i)
    {
        EXPECT_NEAR(output->data()[i], 2.0f, 1e-3f);
    }
}

TEST(Test__HeterogeneousLayers, GatedRMSNormStage_SubtractOne)
{
    const int d = 4;

    auto input = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        input->mutable_data()[i] = 1.0f;

    auto gate = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        gate->mutable_data()[i] = 1.0f;  // gate = 1.0 (pass-through)

    // Gamma stored as 0.5 → effective gamma = 1.5 (with subtract_one)
    auto gamma = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        gamma->mutable_data()[i] = 0.5f;

    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.subtract_one = true;
    params.seq_len = 1;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Normalized [1,1,1,1] → [1,1,1,1]
    // * (1.0 + 0.5) * 1.0 = 1.5
    for (int i = 0; i < d; ++i)
    {
        EXPECT_NEAR(output->data()[i], 1.5f, 1e-3f);
    }
}

TEST(Test__HeterogeneousLayers, GatedRMSNormStage_Type)
{
    GatedRMSNormStage::Params params;
    GatedRMSNormStage stage(params);
    EXPECT_EQ(stage.type(), ComputeStageType::GATED_RMS_NORM);
}

// ============================================================================
// B8: subtract_one in RMSNormStage params
// ============================================================================

TEST(Test__HeterogeneousLayers, RMSNormSubtractOneDefault)
{
    RMSNormStage::Params params;
    EXPECT_FALSE(params.subtract_one);
}

TEST(Test__HeterogeneousLayers, SubtractOneInStageSpec)
{
    StageSpec spec;
    spec.type = StageType::RMSNorm;
    spec.subtract_one = true;
    EXPECT_TRUE(spec.subtract_one.value());
}

// ============================================================================
// B4: HybridCacheManager
// ============================================================================

TEST(Test__HeterogeneousLayers, HybridCacheManager_AllKVCache)
{
    HybridCacheManager::Config config;
    config.n_layers = 4;
    config.layer_types = {"full_attention", "full_attention", "full_attention", "full_attention"};
    config.n_heads = 8;
    config.head_dim = 64;

    HybridCacheManager manager(config, nullptr);
    EXPECT_EQ(manager.kvCacheLayerCount(), 4);
    EXPECT_EQ(manager.gdnLayerCount(), 0);

    for (int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(manager.getLayerStateType(i), LayerStateType::KV_CACHE);
        EXPECT_EQ(manager.getGDNState(i), nullptr);
    }
}

TEST(Test__HeterogeneousLayers, HybridCacheManager_MixedLayers)
{
    HybridCacheManager::Config config;
    config.n_layers = 4;
    config.layer_types = {"full_attention", "gdn", "full_attention", "gdn"};
    config.n_heads = 8;
    config.head_dim = 64;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    EXPECT_EQ(manager.kvCacheLayerCount(), 2);
    EXPECT_EQ(manager.gdnLayerCount(), 2);
    EXPECT_EQ(manager.totalLayers(), 4);

    // Layer 0: KV cache
    EXPECT_EQ(manager.getLayerStateType(0), LayerStateType::KV_CACHE);
    EXPECT_EQ(manager.getGDNState(0), nullptr);
    EXPECT_EQ(manager.toKVCacheLayerIdx(0), 0);

    // Layer 1: GDN
    EXPECT_EQ(manager.getLayerStateType(1), LayerStateType::GDN_STATE);
    ASSERT_NE(manager.getGDNState(1), nullptr);
    EXPECT_EQ(manager.getGDNState(1)->n_heads, 8);
    EXPECT_EQ(manager.getGDNState(1)->head_dim, 64);
    EXPECT_EQ(manager.toKVCacheLayerIdx(1), -1);

    // Layer 2: KV cache (second KV layer)
    EXPECT_EQ(manager.getLayerStateType(2), LayerStateType::KV_CACHE);
    EXPECT_EQ(manager.toKVCacheLayerIdx(2), 1);

    // Layer 3: GDN (second GDN layer)
    EXPECT_EQ(manager.getLayerStateType(3), LayerStateType::GDN_STATE);
    ASSERT_NE(manager.getGDNState(3), nullptr);
}

TEST(Test__HeterogeneousLayers, HybridCacheManager_GDNStateInit)
{
    HybridCacheManager::Config config;
    config.n_layers = 1;
    config.layer_types = {"gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    auto *state = manager.getGDNState(0);
    ASSERT_NE(state, nullptr);

    // Recurrence state: [n_heads, head_dim, head_dim] = 2 * 4 * 4 = 32 floats
    EXPECT_EQ(state->recurrence_state.size(), 32u);
    // All zeros
    for (float v : state->recurrence_state)
        EXPECT_FLOAT_EQ(v, 0.0f);

    // Conv state: [n_heads, (conv_kernel-1), head_dim] = 2 * 3 * 4 = 24 floats
    EXPECT_EQ(state->conv_state.size(), 24u);
    for (float v : state->conv_state)
        EXPECT_FLOAT_EQ(v, 0.0f);
}

TEST(Test__HeterogeneousLayers, HybridCacheManager_Reset)
{
    HybridCacheManager::Config config;
    config.n_layers = 1;
    config.layer_types = {"gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    auto *state = manager.getGDNState(0);
    ASSERT_NE(state, nullptr);

    // Modify state
    state->recurrence_state[0] = 42.0f;
    state->conv_state[0] = 99.0f;

    // Reset
    manager.reset();

    EXPECT_FLOAT_EQ(state->recurrence_state[0], 0.0f);
    EXPECT_FLOAT_EQ(state->conv_state[0], 0.0f);
}

TEST(Test__HeterogeneousLayers, HybridCacheManager_MemoryBytes)
{
    HybridCacheManager::Config config;
    config.n_layers = 2;
    config.layer_types = {"gdn", "gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    HybridCacheManager manager(config, nullptr);
    // Each GDN layer: (2*4*4 + 2*3*4) * 4 bytes = (32 + 24) * 4 = 224 bytes
    EXPECT_EQ(manager.gdnMemoryBytes(), 2 * 224u);
}

TEST(Test__HeterogeneousLayers, HybridCacheManager_OutOfBounds)
{
    HybridCacheManager::Config config;
    config.n_layers = 1;
    config.layer_types = {"gdn"};
    config.n_heads = 2;
    config.head_dim = 4;

    HybridCacheManager manager(config, nullptr);
    EXPECT_EQ(manager.getLayerStateType(-1), LayerStateType::NONE);
    EXPECT_EQ(manager.getLayerStateType(5), LayerStateType::NONE);
    EXPECT_EQ(manager.getGDNState(-1), nullptr);
    EXPECT_EQ(manager.getGDNState(5), nullptr);
    EXPECT_EQ(manager.toKVCacheLayerIdx(-1), -1);
    EXPECT_EQ(manager.toKVCacheLayerIdx(5), -1);
}

// ============================================================================
// B9: Variable GraphConfig fields
// ============================================================================

TEST(Test__HeterogeneousLayers, GraphConfigLayerTypes)
{
    GraphConfig config;
    config.n_layers = 4;
    config.d_ff = 1024;
    config.n_kv_heads = 8;
    config.n_heads = 16;
    config.layer_types = {"full_attention", "gdn", "full_attention", "gdn"};
    config.layer_d_ff = {1024, 512, 1024, 512};

    EXPECT_TRUE(config.hasHeterogeneousLayers());
    EXPECT_EQ(config.getLayerDFF(0), 1024);
    EXPECT_EQ(config.getLayerDFF(1), 512);
    EXPECT_EQ(config.getLayerDFF(2), 1024);
    EXPECT_EQ(config.getLayerDFF(3), 512);
}

TEST(Test__HeterogeneousLayers, GraphConfigFallbackToDefaults)
{
    GraphConfig config;
    config.d_ff = 2048;
    config.n_kv_heads = 8;
    config.n_heads = 16;

    // Empty per-layer vectors — should fall back to global defaults
    EXPECT_FALSE(config.hasHeterogeneousLayers());
    EXPECT_EQ(config.getLayerDFF(0), 2048);
    EXPECT_EQ(config.getLayerNKVHeads(0), 8);
    EXPECT_EQ(config.getLayerNHeads(0), 16);
}

TEST(Test__HeterogeneousLayers, GraphConfigGDN)
{
    GraphConfig config;
    config.gdn.conv_kernel_size = 4;
    config.gdn.state_size = 128;

    EXPECT_TRUE(config.hasGDN());
    EXPECT_FALSE(config.hasHeterogeneousLayers());
}

TEST(Test__HeterogeneousLayers, GraphConfigPartialRotary)
{
    GraphConfig config;
    config.partial_rotary_factor = 0.5f;
    EXPECT_FLOAT_EQ(config.partial_rotary_factor, 0.5f);
}

TEST(Test__HeterogeneousLayers, GraphConfigSubtractOne)
{
    GraphConfig config;
    EXPECT_FALSE(config.rms_norm_subtract_one);
    config.rms_norm_subtract_one = true;
    EXPECT_TRUE(config.rms_norm_subtract_one);
}

TEST(Test__HeterogeneousLayers, GraphConfigAttentionOutputGate)
{
    GraphConfig config;
    EXPECT_FALSE(config.has_attention_output_gate);
    config.has_attention_output_gate = true;
    EXPECT_TRUE(config.has_attention_output_gate);
}

// ============================================================================
// BufferId: new GDN buffer IDs
// ============================================================================

TEST(Test__HeterogeneousLayers, BufferIdNewGDNIds)
{
    EXPECT_STREQ(bufferIdName(BufferId::ATTN_OUTPUT_GATE), "ATTN_OUTPUT_GATE");
    EXPECT_STREQ(bufferIdName(BufferId::GDN_CONV_STATE), "GDN_CONV_STATE");
    EXPECT_STREQ(bufferIdName(BufferId::GDN_RECURRENCE_IN), "GDN_RECURRENCE_IN");
    EXPECT_STREQ(bufferIdName(BufferId::GDN_RECURRENCE_OUT), "GDN_RECURRENCE_OUT");
}

// ============================================================================
// ComputeStageType: new types
// ============================================================================

TEST(Test__HeterogeneousLayers, ComputeStageTypeNewTypes)
{
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::ATTENTION_OUTPUT_GATE), "ATTENTION_OUTPUT_GATE");
    EXPECT_STREQ(computeStageTypeName(ComputeStageType::GATED_RMS_NORM), "GATED_RMS_NORM");
}

// ============================================================================
// StageType: new GDN types in schema
// ============================================================================

TEST(Test__HeterogeneousLayers, StageTypeNewGDNTypes)
{
    StageSpec gate_spec;
    gate_spec.type = StageType::AttentionOutputGate;
    EXPECT_EQ(gate_spec.type, StageType::AttentionOutputGate);

    StageSpec gated_norm_spec;
    gated_norm_spec.type = StageType::GatedRMSNorm;
    EXPECT_EQ(gated_norm_spec.type, StageType::GatedRMSNorm);
}

// ============================================================================
// B8: subtract_one in RMSNormStage::execute() — verifies actual execution
// ============================================================================

TEST(Test__HeterogeneousLayers, RMSNormStage_SubtractOneExecution)
{
    const int d = 4;

    // Input: [1.0, 1.0, 1.0, 1.0] — RMS = 1.0
    auto input = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        input->mutable_data()[i] = 1.0f;

    // Gamma stored as 0.5 → effective gamma = 1.5 with subtract_one
    auto gamma = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        gamma->mutable_data()[i] = 0.5f;

    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.subtract_one = true;
    params.seq_len = 1;

    auto ctx = makeCPUContext();
    RMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // RMS of [1,1,1,1] = sqrt(mean(1^2) + eps) ≈ 1.0
    // Normalized = [1,1,1,1]
    // * (1.0 + 0.5) = 1.5
    for (int i = 0; i < d; ++i)
    {
        EXPECT_NEAR(output->data()[i], 1.5f, 1e-3f)
            << "subtract_one should produce gamma_eff = 1.0 + stored, at index " << i;
    }
}

TEST(Test__HeterogeneousLayers, RMSNormStage_SubtractOneFalse_UsesGammaDirectly)
{
    const int d = 4;

    auto input = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        input->mutable_data()[i] = 1.0f;

    auto gamma = std::make_shared<FP32Tensor>(std::vector<size_t>{static_cast<size_t>(d)}, DeviceId::cpu());
    for (int i = 0; i < d; ++i)
        gamma->mutable_data()[i] = 0.5f;

    auto output = std::make_shared<FP32Tensor>(std::vector<size_t>{1, static_cast<size_t>(d)}, DeviceId::cpu());

    RMSNormStage::Params params;
    params.input = input.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.subtract_one = false;
    params.seq_len = 1;

    auto ctx = makeCPUContext();
    RMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // Without subtract_one: gamma_eff = 0.5 directly
    for (int i = 0; i < d; ++i)
    {
        EXPECT_NEAR(output->data()[i], 0.5f, 1e-3f);
    }
}

// ============================================================================
// B6: AttentionOutputGateStage multi-token test
// ============================================================================

TEST(Test__HeterogeneousLayers, AttentionOutputGateStage_MultiToken)
{
    const int seq_len = 3;
    const int d = 4;

    auto input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d)}, DeviceId::cpu());
    auto gate = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d)}, DeviceId::cpu());
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d)}, DeviceId::cpu());

    float *inp = input->mutable_data();
    float *g = gate->mutable_data();
    for (int i = 0; i < seq_len * d; ++i)
    {
        inp[i] = 2.0f;
        g[i] = 0.0f; // sigmoid(0) = 0.5
    }

    AttentionOutputGateStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.seq_len = seq_len;
    params.feature_dim = d;

    auto ctx = makeCPUContext();
    AttentionOutputGateStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    for (int i = 0; i < seq_len * d; ++i)
    {
        EXPECT_NEAR(output->data()[i], 1.0f, 1e-4f) << "at index " << i;
    }
}

// ============================================================================
// B7: GatedRMSNormStage multi-token test
// ============================================================================

TEST(Test__HeterogeneousLayers, GatedRMSNormStage_MultiToken)
{
    const int seq_len = 2;
    const int d = 4;

    auto input = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d)}, DeviceId::cpu());
    auto gate = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d)}, DeviceId::cpu());
    auto gamma = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(d)}, DeviceId::cpu());
    auto output = std::make_shared<FP32Tensor>(
        std::vector<size_t>{static_cast<size_t>(seq_len), static_cast<size_t>(d)}, DeviceId::cpu());

    for (int i = 0; i < seq_len * d; ++i)
    {
        input->mutable_data()[i] = 1.0f;
        gate->mutable_data()[i] = 1.0f; // gate pass-through
    }
    for (int i = 0; i < d; ++i)
        gamma->mutable_data()[i] = 2.0f;

    GatedRMSNormStage::Params params;
    params.input = input.get();
    params.gate = gate.get();
    params.output = output.get();
    params.gamma = gamma.get();
    params.eps = 1e-6f;
    params.seq_len = seq_len;

    auto ctx = makeCPUContext();
    GatedRMSNormStage stage(params);
    ASSERT_TRUE(stage.execute(ctx.get()));

    // RMS of [1,1,1,1] ≈ 1.0, normalized * gamma(2) * gate(1) = 2.0
    for (int i = 0; i < seq_len * d; ++i)
    {
        EXPECT_NEAR(output->data()[i], 2.0f, 1e-3f) << "at index " << i;
    }
}

// ============================================================================
// B9: GraphConfig per-layer head accessors with per-layer data
// ============================================================================

TEST(Test__HeterogeneousLayers, GraphConfig_PerLayerHeadValues)
{
    GraphConfig config;
    config.n_layers = 3;
    config.n_heads = 16;
    config.n_kv_heads = 8;
    config.layer_n_heads = {16, 0, 16};    // GDN layers have 0 heads
    config.layer_n_kv_heads = {8, 0, 8};

    EXPECT_EQ(config.getLayerNHeads(0), 16);
    EXPECT_EQ(config.getLayerNHeads(1), 0);
    EXPECT_EQ(config.getLayerNHeads(2), 16);
    EXPECT_EQ(config.getLayerNKVHeads(0), 8);
    EXPECT_EQ(config.getLayerNKVHeads(1), 0);
    EXPECT_EQ(config.getLayerNKVHeads(2), 8);
}

// ============================================================================
// B1: getTemplateForLayer bounds check
// ============================================================================

TEST(Test__HeterogeneousLayers, GetTemplateForLayer_OutOfBounds)
{
    GraphSchema schema;
    schema.name = "bounds_test";
    schema.layer_template.attention_stages.push_back(
        StageSpec{.name = "default_attn", .type = StageType::RMSNorm});

    LayerTemplate custom;
    custom.attention_stages.push_back(
        StageSpec{.name = "custom", .type = StageType::GatedRMSNorm});
    schema.named_templates["custom"] = custom;
    schema.layer_template_names = {"custom", "custom"};

    // In-bounds: should dispatch to named template
    const auto &tmpl0 = schema.getTemplateForLayer(0);
    EXPECT_EQ(tmpl0.attention_stages[0].name, "custom");

    // Out of bounds: should fall back to default
    const auto &tmplOOB = schema.getTemplateForLayer(99);
    EXPECT_EQ(tmplOOB.attention_stages[0].name, "default_attn");

    // Negative index: should fall back to default
    const auto &tmplNeg = schema.getTemplateForLayer(-1);
    EXPECT_EQ(tmplNeg.attention_stages[0].name, "default_attn");
}

// ============================================================================
// B4: HybridCacheManager implements IHybridCacheManager
// ============================================================================

TEST(Test__HeterogeneousLayers, HybridCacheManager_ImplementsInterface)
{
    HybridCacheManager::Config config;
    config.n_layers = 2;
    config.layer_types = {"full_attention", "gdn"};
    config.n_heads = 2;
    config.head_dim = 4;
    config.conv_kernel_size = 4;

    auto manager = std::make_unique<HybridCacheManager>(config, nullptr);

    // Can be referenced through the interface
    IHybridCacheManager *iface = manager.get();
    EXPECT_EQ(iface->totalLayers(), 2);
    EXPECT_EQ(iface->kvCacheLayerCount(), 1);
    EXPECT_EQ(iface->gdnLayerCount(), 1);
    EXPECT_EQ(iface->getLayerStateType(0), LayerStateType::KV_CACHE);
    EXPECT_EQ(iface->getLayerStateType(1), LayerStateType::GDN_STATE);
}
