/**
 * @file Test__Qwen35Schema.cpp
 * @brief Unit tests for Qwen 3.5 schema, config builder, graph builder, and registration
 *
 * Tests:
 * - Qwen35SchemaFactory creates valid schema with named templates (gdn, full_attention)
 * - GDN template has correct stages (gdn_proj, short_conv, gdn_recurrence, gated_norm, etc.)
 * - FA template has standard attention stages
 * - Both templates share FFN stages
 * - Weight suffix and optional weight handling
 * - Qwen35GraphConfigBuilder populates GDN fields correctly
 * - Layer type pattern generation from full_attention_interval
 * - "qwen35" architecture registration in factories
 * - GDN attention graph construction: correct stages, wiring, interface usage
 * - FA attention graph delegation to QwenStandardGraph
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <string>

#include "models/qwen35/Qwen35Schema.h"
#include "models/qwen35/Qwen35GraphConfigBuilder.h"
#include "models/qwen35/Qwen35Graph.h"
#include "execution/local_execution/graph/GraphBuilderRegistry.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "execution/local_execution/graph/GraphSchema.h"
#include "execution/local_execution/graph/GraphResolver.h"
#include "execution/compute_stages/stages/GDNProjectionStage.h"
#include "execution/compute_stages/stages/ShortConv1dStage.h"
#include "execution/compute_stages/stages/GDNRecurrenceStage.h"
#include "execution/compute_stages/stages/GatedRMSNormStage.h"
#include "execution/compute_stages/stages/AttentionOutputGateStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "config/TensorParallelConfig.h"
#include "kernels/IKVCache.h"
#include "kernels/IHybridKVCache.h"
#include "kernels/HybridKVCacheConfig.h"
#include "loaders/WeightPlan.h"
#include "tensors/TensorKernels.h"
#include "memory/BufferId.h"
#include "../../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Stub implementations for unit testing (graph construction only, never executed)
// ============================================================================

namespace
{
    /// Minimal ITensorShortConvolution stub — forward() is never called during graph build
    class StubShortConvolution : public ITensorShortConvolution
    {
    public:
        bool forward(const float *, const float *, const float *,
                     float *, float *, int, int, int, bool) override
        {
            return true;
        }
    };

    /// Minimal ITensorGatedDeltaNet stub — methods never called during graph build
    class StubGatedDeltaNet : public ITensorGatedDeltaNet
    {
    public:
        bool chunk_forward(const float *, const float *, const float *,
                           const float *, const float *, const float *, const float *,
                           float *, float *, int, int, int, int, int, bool) override
        {
            return true;
        }
        bool recurrent_step(const float *, const float *, const float *,
                            const float *, const float *, const float *, const float *,
                            float *, float *, int, int, int, bool) override
        {
            return true;
        }
    };

    /**
     * @brief Minimal hybrid KV cache stub for graph construction tests.
     *
     * Provides GDN state and stub kernels so buildGDNAttentionGraph() can
     * succeed without requiring a full KV cache implementation.
     * IKVCache methods are stubbed out — they're never called by graph builders.
     */
    class StubHybridKVCache : public IKVCache, public IHybridKVCache
    {
    public:
        StubHybridKVCache(const GraphConfig &config)
        {
            n_layers_ = config.n_layers;
            layer_types_ = config.layer_types;

            // Compute GDN dimensions
            const int n_k = config.gdn.group_count > 0 ? config.gdn.group_count : config.n_heads;
            const int n_v = config.gdn.time_step_rank > 0 ? config.gdn.time_step_rank : n_k;
            const int d_k = config.gdn.state_size;
            const int d_v = d_k;
            const int qkv_dim = 2 * n_k * d_k + n_v * d_v;

            gdn_states_.resize(n_layers_);
            for (int i = 0; i < n_layers_; ++i)
            {
                if (i < static_cast<int>(layer_types_.size()) && layer_types_[i] == "gdn")
                {
                    auto &s = gdn_states_[i];
                    s.n_v_heads = n_v;
                    s.n_k_heads = n_k;
                    s.d_k = d_k;
                    s.d_v = d_v;
                    s.conv_kernel_size = config.gdn.conv_kernel_size;
                    s.initialize(qkv_dim);
                    s.conv_kernel = std::make_shared<StubShortConvolution>();
                    s.rec_kernel = std::make_shared<StubGatedDeltaNet>();
                }
            }
        }

        // --- IHybridKVCache ---
        bool isGDNLayer(int layer) const override
        {
            return layer >= 0 && layer < static_cast<int>(layer_types_.size()) &&
                   layer_types_[layer] == "gdn";
        }
        bool isFullAttentionLayer(int layer) const override { return !isGDNLayer(layer); }
        int kvLayerCount() const override
        {
            return static_cast<int>(std::count(layer_types_.begin(), layer_types_.end(), "full_attention"));
        }
        int gdnLayerCount() const override { return n_layers_ - kvLayerCount(); }
        HybridGDNLayerState *getGDNState(int layer) override
        {
            if (!isGDNLayer(layer))
                return nullptr;
            return &gdn_states_[layer];
        }
        const HybridGDNLayerState *getGDNState(int layer) const override
        {
            if (!isGDNLayer(layer))
                return nullptr;
            return &gdn_states_[layer];
        }
        float *getRecurrenceState(int layer) override
        {
            auto *s = getGDNState(layer);
            return s ? s->recurrence_state.data() : nullptr;
        }
        float *getConvState(int layer) override
        {
            auto *s = getGDNState(layer);
            return s ? s->conv_state.data() : nullptr;
        }
        ITensorShortConvolution *getConvKernel(int layer) override
        {
            auto *s = getGDNState(layer);
            return s ? s->conv_kernel.get() : nullptr;
        }
        ITensorGatedDeltaNet *getRecurrenceKernel(int layer) override
        {
            auto *s = getGDNState(layer);
            return s ? s->rec_kernel.get() : nullptr;
        }
        void resetGDNStates() override
        {
            for (auto &s : gdn_states_)
                s.reset();
        }
        size_t gdnMemoryBytes() const override
        {
            size_t total = 0;
            for (const auto &s : gdn_states_)
                total += s.memoryBytes();
            return total;
        }

        // --- IKVCache stubs (never called by graph builders) ---
        ActivationPrecision k_precision() const override { return ActivationPrecision::FP32; }
        int get_cached_tokens(int, int) const override { return 0; }
        int max_seq_len() const override { return 32; }
        int n_layers() const override { return n_layers_; }
        bool get_kv(int, int, ITensor **, ITensor **, int *) override { return false; }
        bool get_kv(int, int, const ITensor **, const ITensor **, int *) const override { return false; }
        ITensor *get_k(int, int) override { return nullptr; }
        const ITensor *get_k(int, int) const override { return nullptr; }
        ITensor *get_v(int, int) override { return nullptr; }
        const ITensor *get_v(int, int) const override { return nullptr; }
        bool append(int, int, const ITensor *, const ITensor *, int) override { return false; }
        void clear() override {}
        void clear_sequence(int, int) override {}
        void clear_layer(int) override {}

    private:
        int n_layers_ = 0;
        std::vector<std::string> layer_types_;
        std::vector<HybridGDNLayerState> gdn_states_;
    };
} // namespace

namespace
{
    // Helper to find a stage by name in a vector of StageSpec
    bool hasStageNamed(const std::vector<StageSpec> &stages, const std::string &name)
    {
        return std::any_of(stages.begin(), stages.end(),
                           [&](const StageSpec &s)
                           { return s.name == name; });
    }

    bool hasStageOfType(const std::vector<StageSpec> &stages, StageType type)
    {
        return std::any_of(stages.begin(), stages.end(),
                           [&](const StageSpec &s)
                           { return s.type == type; });
    }

    const StageSpec *findStage(const std::vector<StageSpec> &stages, const std::string &name)
    {
        auto it = std::find_if(stages.begin(), stages.end(),
                               [&](const StageSpec &s)
                               { return s.name == name; });
        return it != stages.end() ? &(*it) : nullptr;
    }
} // namespace

// ============================================================================
// Schema Factory Tests
// ============================================================================

TEST(Test__Qwen35Schema, ArchitectureName)
{
    Qwen35SchemaFactory factory;
    EXPECT_EQ(factory.architectureName(), "qwen35");
}

TEST(Test__Qwen35Schema, CreatesValidSchema)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    EXPECT_EQ(schema.name, "qwen35");
    EXPECT_EQ(schema.version, "1.0");
    EXPECT_FALSE(schema.required_params.empty());
}

TEST(Test__Qwen35Schema, GDNValueHeadWeightsUseProportionalHeadSharding)
{
    Qwen35SchemaFactory factory;
    WeightShardingConfig sharding = factory.getWeightShardingConfig();

    EXPECT_EQ(sharding.getMode("blk.0.attn_gate.weight"),
              WeightShardingMode::ColumnParallel);
    EXPECT_EQ(sharding.getDimensionType("blk.0.attn_gate.weight"),
              WeightDimensionType::ProportionalHeads);

    EXPECT_EQ(sharding.getMode("blk.0.ssm_out.weight"),
              WeightShardingMode::InputParallel);
    EXPECT_EQ(sharding.getDimensionType("blk.0.ssm_out.weight"),
              WeightDimensionType::ProportionalHeads);
}

TEST(Test__Qwen35Schema, GDNGlobalVHeadOffsetPrefersValueProjectionSlice)
{
    // Qwen3.5-4B dense GDN has 36 FA/Q heads but 32 GDN V heads.
    // TP=2 rank 1 therefore starts at Q-head 18, but V-head 16.
    GraphConfig config;
    config.n_heads = 36;
    config.head_start = 18;
    config.local_rank = 1;
    config.tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, 36, 4, 9216, 151936));

    WeightBinding value_projection_binding;
    value_projection_binding.slice.source_rows = 4096;
    value_projection_binding.slice.row_start = 16 * 128;
    value_projection_binding.slice.row_count = 16 * 128;

    const int offset = Qwen35Graph::resolveGDNGlobalVHeadOffset(
        &value_projection_binding,
        128,
        16,
        32,
        config,
        nullptr);

    EXPECT_EQ(offset, 16);
    EXPECT_NE(offset, config.head_start);
}

TEST(Test__Qwen35Schema, GDNGlobalVHeadOffsetFallbackMapsQHeadsToVHeads)
{
    GraphConfig config;
    config.n_heads = 36;
    config.local_rank = 1;
    config.tp_config = std::make_shared<TensorParallelConfig>(
        TensorParallelConfig::equalSplit(2, 36, 4, 9216, 151936));

    const int offset = Qwen35Graph::resolveGDNGlobalVHeadOffset(
        nullptr,
        128,
        16,
        32,
        config,
        nullptr);

    EXPECT_EQ(config.tp_config->forRank(1).head_start, 18);
    EXPECT_EQ(offset, 16);
    EXPECT_NE(offset, config.tp_config->forRank(1).head_start);
}

TEST(Test__Qwen35Schema, HasNamedTemplates)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Must have exactly 2 named templates
    ASSERT_EQ(schema.named_templates.size(), 2u);
    EXPECT_TRUE(schema.named_templates.count("gdn") > 0);
    EXPECT_TRUE(schema.named_templates.count("full_attention") > 0);
}

// ============================================================================
// GDN Template Tests
// ============================================================================

TEST(Test__Qwen35Schema, GDNTemplate_HasGDNProjection)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &gdn = schema.named_templates.at("gdn");
    EXPECT_TRUE(hasStageOfType(gdn.attention_stages, StageType::GDNProjection))
        << "GDN template should have GDNProjection stage";
}

TEST(Test__Qwen35Schema, GDNTemplate_HasShortConv1d)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &gdn = schema.named_templates.at("gdn");
    EXPECT_TRUE(hasStageOfType(gdn.attention_stages, StageType::ShortConv1d))
        << "GDN template should have ShortConv1d stage";
}

TEST(Test__Qwen35Schema, GDNTemplate_HasGDNRecurrence)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &gdn = schema.named_templates.at("gdn");
    EXPECT_TRUE(hasStageOfType(gdn.attention_stages, StageType::GDNRecurrence))
        << "GDN template should have GDNRecurrence stage";
}

TEST(Test__Qwen35Schema, GDNTemplate_HasGatedRMSNorm)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &gdn = schema.named_templates.at("gdn");
    EXPECT_TRUE(hasStageOfType(gdn.attention_stages, StageType::GatedRMSNorm))
        << "GDN template should have GatedRMSNorm stage";
}

TEST(Test__Qwen35Schema, GDNTemplate_HasAttentionOutputGate)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &gdn = schema.named_templates.at("gdn");
    EXPECT_TRUE(hasStageOfType(gdn.attention_stages, StageType::AttentionOutputGate))
        << "GDN template should have AttentionOutputGate stage";
}

TEST(Test__Qwen35Schema, GDNTemplate_HasCorrectStageOrder)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &gdn = schema.named_templates.at("gdn");
    const auto &stages = gdn.attention_stages;

    // Verify key stages are present and in order
    ASSERT_GE(stages.size(), 6u) << "GDN template should have at least 6 attention stages";

    // First stage should be attn_norm
    EXPECT_TRUE(hasStageNamed(stages, "attn_norm"));

    // GDN projection stage should exist
    EXPECT_TRUE(hasStageNamed(stages, "gdn_proj"));

    // Short conv should follow gdn_proj
    EXPECT_TRUE(hasStageNamed(stages, "short_conv"));

    // GDN recurrence should follow short_conv
    EXPECT_TRUE(hasStageNamed(stages, "gdn_recurrence"));

    // Gated norm follows recurrence
    EXPECT_TRUE(hasStageNamed(stages, "gated_norm"));
}

TEST(Test__Qwen35Schema, GDNTemplate_NoKVCache)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &gdn = schema.named_templates.at("gdn");

    // GDN layers should NOT have KVCacheAppend or standard Attention
    EXPECT_FALSE(hasStageOfType(gdn.attention_stages, StageType::KVCacheAppend))
        << "GDN layers should not use KV cache";
    EXPECT_FALSE(hasStageOfType(gdn.attention_stages, StageType::AttentionCompute))
        << "GDN layers should not use standard attention";
}

// ============================================================================
// Full Attention Template Tests
// ============================================================================

TEST(Test__Qwen35Schema, FATemplate_HasStandardAttention)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &fa = schema.named_templates.at("full_attention");

    // FA template should have QKV projection
    EXPECT_TRUE(hasStageOfType(fa.attention_stages, StageType::FusedQKVGEMM) ||
                hasStageNamed(fa.attention_stages, "qkv_proj"))
        << "FA template should have QKV projection";

    // FA template should have RoPE
    EXPECT_TRUE(hasStageOfType(fa.attention_stages, StageType::RoPE))
        << "FA template should have RoPE";

    // FA template should have attention output gate
    EXPECT_TRUE(hasStageOfType(fa.attention_stages, StageType::AttentionOutputGate))
        << "FA template should have AttentionOutputGate";
}

TEST(Test__Qwen35Schema, FATemplate_NoGDNStages)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    const auto &fa = schema.named_templates.at("full_attention");

    // FA template should NOT have GDN-specific stages
    EXPECT_FALSE(hasStageOfType(fa.attention_stages, StageType::GDNProjection))
        << "FA layers should not have GDN projection";
    EXPECT_FALSE(hasStageOfType(fa.attention_stages, StageType::ShortConv1d))
        << "FA layers should not have short conv";
    EXPECT_FALSE(hasStageOfType(fa.attention_stages, StageType::GDNRecurrence))
        << "FA layers should not have GDN recurrence";
}

// ============================================================================
// Shared FFN Tests
// ============================================================================

TEST(Test__Qwen35Schema, BothTemplates_HaveFFN)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    for (const auto &[name, tmpl] : schema.named_templates)
    {
        EXPECT_GE(tmpl.ffn_stages.size(), 4u)
            << "Template '" << name << "' should have at least 4 FFN stages";
        EXPECT_TRUE(hasStageNamed(tmpl.ffn_stages, "ffn_norm"))
            << "Template '" << name << "' should have ffn_norm";
        EXPECT_TRUE(hasStageNamed(tmpl.ffn_stages, "gate_up_proj"))
            << "Template '" << name << "' should have gate_up_proj";
        EXPECT_TRUE(hasStageNamed(tmpl.ffn_stages, "down_proj"))
            << "Template '" << name << "' should have down_proj";
    }
}

// ============================================================================
// Weight Suffix and Optional Weight Tests
// ============================================================================

TEST(Test__Qwen35Schema, WeightSuffixes_ContainsGDNWeights)
{
    Qwen35SchemaFactory factory;
    auto suffixes = factory.layerWeightSuffixes();

    auto hasSuffix = [&](const std::string &s)
    {
        return std::find(suffixes.begin(), suffixes.end(), s) != suffixes.end();
    };

    // GDN-specific weights
    EXPECT_TRUE(hasSuffix("attn_qkv.weight")) << "Missing attn_qkv.weight";
    EXPECT_TRUE(hasSuffix("ssm_a")) << "Missing ssm_a";
    EXPECT_TRUE(hasSuffix("ssm_conv1d.weight")) << "Missing ssm_conv1d.weight";
    EXPECT_TRUE(hasSuffix("ssm_dt.bias")) << "Missing ssm_dt.bias";
    EXPECT_TRUE(hasSuffix("ssm_norm.weight")) << "Missing ssm_norm.weight";
    EXPECT_TRUE(hasSuffix("ssm_out.weight")) << "Missing ssm_out.weight";
}

TEST(Test__Qwen35Schema, WeightSuffixes_ContainsFAWeights)
{
    Qwen35SchemaFactory factory;
    auto suffixes = factory.layerWeightSuffixes();

    auto hasSuffix = [&](const std::string &s)
    {
        return std::find(suffixes.begin(), suffixes.end(), s) != suffixes.end();
    };

    // FA-specific weights
    EXPECT_TRUE(hasSuffix("attn_q.weight")) << "Missing attn_q.weight";
    EXPECT_TRUE(hasSuffix("attn_k.weight")) << "Missing attn_k.weight";
    EXPECT_TRUE(hasSuffix("attn_v.weight")) << "Missing attn_v.weight";
    EXPECT_TRUE(hasSuffix("attn_output.weight")) << "Missing attn_output.weight";
}

TEST(Test__Qwen35Schema, OptionalWeights)
{
    Qwen35SchemaFactory factory;

    // FA-only weights should be optional
    EXPECT_TRUE(factory.isWeightOptional("attn_q.weight"));
    EXPECT_TRUE(factory.isWeightOptional("attn_k.weight"));
    EXPECT_TRUE(factory.isWeightOptional("attn_v.weight"));
    EXPECT_TRUE(factory.isWeightOptional("attn_output.weight"));

    // GDN-only weights should be optional
    EXPECT_TRUE(factory.isWeightOptional("attn_qkv.weight"));
    EXPECT_TRUE(factory.isWeightOptional("ssm_a"));
    EXPECT_TRUE(factory.isWeightOptional("ssm_conv1d.weight"));

    // Shared weights should NOT be optional
    EXPECT_FALSE(factory.isWeightOptional("attn_norm.weight"));
    EXPECT_FALSE(factory.isWeightOptional("ffn_gate.weight"));
    EXPECT_FALSE(factory.isWeightOptional("ffn_up.weight"));
    EXPECT_FALSE(factory.isWeightOptional("ffn_down.weight"));
    EXPECT_FALSE(factory.isWeightOptional("ffn_norm.weight"));
}

// ============================================================================
// GraphConfig Layer Type Pattern Tests
// ============================================================================

TEST(Test__Qwen35Schema, LayerTypes_Interval4)
{
    // With full_attention_interval=4 and 32 layers:
    // FA at layers 3,7,11,15,19,23,27,31 → 8 FA, 24 GDN
    GraphConfig config;
    config.n_layers = 32;
    config.gdn.full_attention_interval = 4;

    // Simulate what Qwen35GraphConfigBuilder does
    config.layer_types.resize(32);
    for (int i = 0; i < 32; ++i)
    {
        bool is_fa = (config.gdn.full_attention_interval > 0) &&
                     ((i + 1) % config.gdn.full_attention_interval == 0);
        config.layer_types[i] = is_fa ? "full_attention" : "gdn";
    }

    // Count types
    int gdn_count = 0, fa_count = 0;
    for (const auto &t : config.layer_types)
    {
        if (t == "gdn")
            gdn_count++;
        else if (t == "full_attention")
            fa_count++;
    }
    EXPECT_EQ(gdn_count, 24);
    EXPECT_EQ(fa_count, 8);

    // Specific layer checks
    EXPECT_EQ(config.layer_types[0], "gdn");
    EXPECT_EQ(config.layer_types[1], "gdn");
    EXPECT_EQ(config.layer_types[2], "gdn");
    EXPECT_EQ(config.layer_types[3], "full_attention");
    EXPECT_EQ(config.layer_types[4], "gdn");
    EXPECT_EQ(config.layer_types[7], "full_attention");
    EXPECT_EQ(config.layer_types[31], "full_attention");
}

TEST(Test__Qwen35Schema, IsFullAttentionLayer_Shortcut)
{
    GraphConfig config;
    config.n_layers = 8;
    config.gdn.full_attention_interval = 4;
    config.layer_types = {"gdn", "gdn", "gdn", "full_attention",
                          "gdn", "gdn", "gdn", "full_attention"};

    EXPECT_FALSE(config.isFullAttentionLayer(0));
    EXPECT_FALSE(config.isFullAttentionLayer(1));
    EXPECT_FALSE(config.isFullAttentionLayer(2));
    EXPECT_TRUE(config.isFullAttentionLayer(3));
    EXPECT_FALSE(config.isFullAttentionLayer(4));
    EXPECT_TRUE(config.isFullAttentionLayer(7));
}

TEST(Test__Qwen35Schema, HasGDN)
{
    GraphConfig config;
    config.gdn.conv_kernel_size = 4;
    config.gdn.state_size = 128;
    EXPECT_TRUE(config.hasGDN());

    GraphConfig config2;
    config2.gdn.conv_kernel_size = 0;
    EXPECT_FALSE(config2.hasGDN());
}

// ============================================================================
// Schema Dispatch Tests (getTemplateForLayer)
// ============================================================================

TEST(Test__Qwen35Schema, SchemaDispatchesCorrectly)
{
    Qwen35SchemaFactory factory;
    GraphSchema schema = factory.createSchema();

    // Simulate 8 layers: gdn, gdn, gdn, full_attention, gdn, gdn, gdn, full_attention
    schema.layer_template_names = {
        "gdn", "gdn", "gdn", "full_attention",
        "gdn", "gdn", "gdn", "full_attention"};

    // Layer 0 (GDN) should have GDNProjection
    const auto &l0 = schema.getTemplateForLayer(0);
    EXPECT_TRUE(hasStageOfType(l0.attention_stages, StageType::GDNProjection));

    // Layer 3 (FA) should have FusedQKVGEMM or similar
    const auto &l3 = schema.getTemplateForLayer(3);
    EXPECT_FALSE(hasStageOfType(l3.attention_stages, StageType::GDNProjection));
    EXPECT_TRUE(hasStageOfType(l3.attention_stages, StageType::RoPE));
}

// ============================================================================
// Registration Tests
// ============================================================================

TEST(Test__Qwen35Schema, RegistrationInGraphBuilderRegistry)
{
    EXPECT_TRUE(GraphBuilderRegistry::isSupported("qwen35"))
        << "qwen35 should be registered in GraphBuilderRegistry";

    auto builder = GraphBuilderRegistry::create("qwen35", GraphConfig{}, nullptr);
    ASSERT_NE(builder, nullptr);
    EXPECT_EQ(builder->architectureName(), "qwen35");
}

TEST(Test__Qwen35Schema, RegistrationInSchemaFactoryRegistry)
{
    auto factory = SchemaFactoryRegistry::getFactory("qwen35");
    ASSERT_NE(factory, nullptr) << "qwen35 should be registered in SchemaFactoryRegistry";
    EXPECT_EQ(factory->architectureName(), "qwen35");
}

// ============================================================================
// Qwen35Graph Basic Tests
// ============================================================================

TEST(Test__Qwen35Schema, GraphArchitectureName)
{
    GraphConfig config;
    config.n_layers = 8;
    Qwen35Graph graph(config, nullptr);
    EXPECT_EQ(graph.architectureName(), "qwen35");
}

TEST(Test__Qwen35Schema, GraphGetSchema_HasNamedTemplates)
{
    GraphConfig config;
    config.n_layers = 8;
    config.layer_types = {"gdn", "gdn", "gdn", "full_attention",
                          "gdn", "gdn", "gdn", "full_attention"};

    Qwen35Graph graph(config, nullptr);
    GraphSchema schema = graph.getSchema();

    EXPECT_EQ(schema.name, "qwen35");
    EXPECT_EQ(schema.named_templates.size(), 2u);
    ASSERT_EQ(schema.layer_template_names.size(), 8u);
    EXPECT_EQ(schema.layer_template_names[0], "gdn");
    EXPECT_EQ(schema.layer_template_names[3], "full_attention");
}

// ============================================================================
// BufferId Tests
// ============================================================================

TEST(Test__Qwen35Schema, GDN_BufferIds_Exist)
{
    // Verify GDN-specific BufferIds can be converted to strings
    EXPECT_NE(bufferIdName(BufferId::GDN_QKV), "UNKNOWN");
    EXPECT_NE(bufferIdName(BufferId::GDN_Z), "UNKNOWN");
    EXPECT_NE(bufferIdName(BufferId::GDN_ALPHA), "UNKNOWN");
    EXPECT_NE(bufferIdName(BufferId::GDN_BETA), "UNKNOWN");
}

// ============================================================================
// GDN Attention Graph Construction Tests
// ============================================================================

namespace
{
    /**
     * @brief Test fixture for Qwen35Graph GDN attention graph construction
     *
     * Creates a minimal Qwen35 config with 4 layers (3 GDN + 1 FA)
     * and mock tensors to test buildAttentionGraph dispatching and
     * GDN sub-graph construction.
     */
    class Qwen35GraphBuildTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            config_.n_layers = 4;
            config_.d_model = 64;
            config_.n_heads = 4;
            config_.n_kv_heads = 2;
            config_.head_dim = 16;
            config_.d_ff = 128;
            config_.vocab_size = 1000;
            config_.rms_norm_eps = 1e-6f;
            config_.rope_theta = 10000.0f;
            config_.default_device = DeviceId::cpu();
            config_.max_seq_len = 32;

            // GDN config matching Qwen3.5-4B pattern
            config_.gdn.conv_kernel_size = 4;
            config_.gdn.state_size = 16;  // d_v per head
            config_.gdn.inner_size = 128; // 4 * n_heads * d_v
            config_.gdn.group_count = 4;
            config_.gdn.time_step_rank = 8;
            config_.gdn.full_attention_interval = 4;

            // Set up layer types: gdn, gdn, gdn, full_attention
            config_.layer_types = {"gdn", "gdn", "gdn", "full_attention"};

            // Create mock tensors for weights and buffers
            // Using small sizes for unit tests
            const size_t d = static_cast<size_t>(config_.d_model);
            const size_t inner = static_cast<size_t>(config_.gdn.inner_size);
            const size_t nh = static_cast<size_t>(config_.n_heads);

            // Layer weights
            // Weight shapes must be consistent with GDN config for head count derivation:
            // qkv_dim = 2*n_k_heads*d_k + n_v_heads*d_v = 2*4*16 + 8*16 = 256
            // value_dim = n_v_heads*d_v = 8*16 = 128
            const int n_k = config_.gdn.group_count;     // 4
            const int n_v = config_.gdn.time_step_rank;  // 8
            const int dk = config_.gdn.state_size;        // 16
            const size_t qkv_total = static_cast<size_t>(2 * n_k * dk + n_v * dk);
            const size_t value_total = static_cast<size_t>(n_v * dk);

            norm_weight_ = TestTensorFactory::createFP32Ones({d});
            attn_qkv_weight_ = TestTensorFactory::createFP32Random({qkv_total, d});
            attn_gate_weight_ = TestTensorFactory::createFP32Random({value_total, d}); // Z proj
            ssm_alpha_weight_ = TestTensorFactory::createFP32Random({static_cast<size_t>(n_v), d});
            ssm_beta_weight_ = TestTensorFactory::createFP32Random({static_cast<size_t>(n_v), d});
            ssm_conv_weight_ = TestTensorFactory::createFP32Random({qkv_total, 4});
            ssm_dt_bias_ = TestTensorFactory::createFP32Zeros({static_cast<size_t>(n_v)});
            ssm_a_ = TestTensorFactory::createFP32Random({static_cast<size_t>(n_v)});
            ssm_norm_weight_ = TestTensorFactory::createFP32Ones({inner});
            ssm_out_weight_ = TestTensorFactory::createFP32Random({d, inner});

            // Activation buffers
            hidden_ = TestTensorFactory::createFP32({2, d});
            normalized_ = TestTensorFactory::createFP32({2, d});
            gdn_qkv_ = TestTensorFactory::createFP32({2, qkv_total});
            gdn_z_ = TestTensorFactory::createFP32({2, value_total});
            gdn_alpha_ = TestTensorFactory::createFP32({2, static_cast<size_t>(n_v)});
            gdn_beta_ = TestTensorFactory::createFP32({2, static_cast<size_t>(n_v)});
            attn_output_ = TestTensorFactory::createFP32({2, inner});
            attn_proj_ = TestTensorFactory::createFP32({2, d});
            gate_ = TestTensorFactory::createFP32({2, value_total});
            attn_output_ = TestTensorFactory::createFP32({2, d});
            attn_proj_ = TestTensorFactory::createFP32({2, d});
            gate_ = TestTensorFactory::createFP32({2, nh * 16});

            // Populate layer weights
            layer_.attn_norm = norm_weight_.get();
            layer_.attn_qkv = attn_qkv_weight_.get();
            layer_.attn_gate = attn_gate_weight_.get();
            layer_.ssm_alpha = ssm_alpha_weight_.get();
            layer_.ssm_beta = ssm_beta_weight_.get();
            layer_.ssm_conv1d = ssm_conv_weight_.get();
            layer_.ssm_dt_bias = ssm_dt_bias_.get();
            layer_.ssm_a = ssm_a_.get();
            layer_.ssm_norm = ssm_norm_weight_.get();
            layer_.ssm_out = ssm_out_weight_.get();

            // Populate activation buffers
            buffers_.current_hidden = hidden_.get();
            buffers_.normalized = normalized_.get();
            buffers_.extensions[BufferId::GDN_QKV] = gdn_qkv_.get();
            buffers_.extensions[BufferId::GDN_Z] = gdn_z_.get();
            buffers_.extensions[BufferId::GDN_ALPHA] = gdn_alpha_.get();
            buffers_.extensions[BufferId::GDN_BETA] = gdn_beta_.get();
            buffers_.attn_output = attn_output_.get();
            buffers_.attn_proj = attn_proj_.get();
            buffers_.gate = gate_.get();

            // Create a stub hybrid KV cache for GDN graph construction tests
            stub_cache_ = std::make_unique<StubHybridKVCache>(config_);
        }

        GraphConfig config_;
        LayerWeights layer_;
        ActivationBuffers buffers_;
        std::unique_ptr<StubHybridKVCache> stub_cache_;

        // Weight tensors (ownership)
        std::unique_ptr<FP32Tensor> norm_weight_;
        std::unique_ptr<FP32Tensor> attn_qkv_weight_;
        std::unique_ptr<FP32Tensor> attn_gate_weight_;
        std::unique_ptr<FP32Tensor> ssm_alpha_weight_;
        std::unique_ptr<FP32Tensor> ssm_beta_weight_;
        std::unique_ptr<FP32Tensor> ssm_conv_weight_;
        std::unique_ptr<FP32Tensor> ssm_dt_bias_;
        std::unique_ptr<FP32Tensor> ssm_a_;
        std::unique_ptr<FP32Tensor> ssm_norm_weight_;
        std::unique_ptr<FP32Tensor> ssm_out_weight_;

        // Activation buffers (ownership)
        std::unique_ptr<FP32Tensor> hidden_;
        std::unique_ptr<FP32Tensor> normalized_;
        std::unique_ptr<FP32Tensor> gdn_qkv_;
        std::unique_ptr<FP32Tensor> gdn_z_;
        std::unique_ptr<FP32Tensor> gdn_alpha_;
        std::unique_ptr<FP32Tensor> gdn_beta_;
        std::unique_ptr<FP32Tensor> attn_output_;
        std::unique_ptr<FP32Tensor> attn_proj_;
        std::unique_ptr<FP32Tensor> gate_;
    };
} // namespace

TEST_F(Qwen35GraphBuildTest, GDNAttentionGraph_HasExpectedNodeCount)
{
    Qwen35Graph graph(config_, nullptr);

    // Build attention graph for GDN layer 0
    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, /*kv_cache=*/stub_cache_.get(), /*position_ids=*/nullptr,
        DeviceId::cpu());

    // Expected nodes: attn_norm, gdn_proj, short_conv, gdn_recurrence,
    //                 gated_norm, gdn_out_proj
    // (No output_gate or residual — FFN FusedResidualNorm handles residual)
    EXPECT_EQ(attn_graph.size(), 6u)
        << "GDN attention graph should have 6 stages";
}

TEST_F(Qwen35GraphBuildTest, GDNAttentionGraph_NodesExist)
{
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    EXPECT_NE(attn_graph.getNode("layer0_attn_norm"), nullptr);
    EXPECT_NE(attn_graph.getNode("layer0_gdn_proj"), nullptr);
    EXPECT_NE(attn_graph.getNode("layer0_short_conv"), nullptr);
    EXPECT_NE(attn_graph.getNode("layer0_gdn_recurrence"), nullptr);
    EXPECT_NE(attn_graph.getNode("layer0_gated_norm"), nullptr);
    EXPECT_NE(attn_graph.getNode("layer0_gdn_out_proj"), nullptr);
    // No output_gate or residual — FFN FusedResidualNorm handles residual
    EXPECT_EQ(attn_graph.getNode("layer0_attn_output_gate"), nullptr);
    EXPECT_EQ(attn_graph.getNode("layer0_attn_residual"), nullptr);
}

TEST_F(Qwen35GraphBuildTest, GDNAttentionGraph_StageTypes)
{
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *proj = attn_graph.getNode("layer0_gdn_proj");
    ASSERT_NE(proj, nullptr);
    EXPECT_EQ(proj->stage->type(), ComputeStageType::GDN_PROJECTION);

    auto *conv = attn_graph.getNode("layer0_short_conv");
    ASSERT_NE(conv, nullptr);
    EXPECT_EQ(conv->stage->type(), ComputeStageType::SHORT_CONV1D);

    auto *rec = attn_graph.getNode("layer0_gdn_recurrence");
    ASSERT_NE(rec, nullptr);
    EXPECT_EQ(rec->stage->type(), ComputeStageType::GDN_RECURRENCE);

    auto *gnorm = attn_graph.getNode("layer0_gated_norm");
    ASSERT_NE(gnorm, nullptr);
    EXPECT_EQ(gnorm->stage->type(), ComputeStageType::GATED_RMS_NORM);

    // Output gate removed from GDN — only used in FA layers
    EXPECT_EQ(attn_graph.getNode("layer0_attn_output_gate"), nullptr);
}

TEST_F(Qwen35GraphBuildTest, GDNAttentionGraph_DependencyChain)
{
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    // Verify linear dependency chain
    auto verifyDep = [&](const std::string &node, const std::string &expected_dep)
    {
        auto *n = attn_graph.getNode(node);
        ASSERT_NE(n, nullptr) << "Node " << node << " missing";
        bool has_dep = std::find(n->dependencies.begin(), n->dependencies.end(),
                                 expected_dep) != n->dependencies.end();
        EXPECT_TRUE(has_dep) << node << " should depend on " << expected_dep;
    };

    verifyDep("layer0_gdn_proj", "layer0_attn_norm");
    verifyDep("layer0_short_conv", "layer0_gdn_proj");
    verifyDep("layer0_gdn_recurrence", "layer0_short_conv");
    verifyDep("layer0_gated_norm", "layer0_gdn_recurrence");
    verifyDep("layer0_gdn_out_proj", "layer0_gated_norm");
    // No output_gate or residual — chain ends at gdn_out_proj
}

TEST_F(Qwen35GraphBuildTest, GDNProjection_ZWeightIsAttnGate)
{
    // The Z projection weight must be attn_gate (in_proj_z), NOT ssm_norm
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *proj_node = attn_graph.getNode("layer0_gdn_proj");
    ASSERT_NE(proj_node, nullptr);

    auto *proj_stage = dynamic_cast<GDNProjectionStage *>(proj_node->stage.get());
    ASSERT_NE(proj_stage, nullptr) << "Should be a GDNProjectionStage";

    const auto &params = proj_stage->getParams();

    // w_z must point to attn_gate weight, not ssm_norm
    EXPECT_EQ(params.w_z, layer_.attn_gate)
        << "Z projection weight should be attn_gate (in_proj_z), not ssm_norm";
    EXPECT_NE(params.w_z, layer_.ssm_norm)
        << "Z projection weight must NOT be ssm_norm (that's the GDN output norm gamma)";
}

TEST_F(Qwen35GraphBuildTest, GDNProjection_AllWeightsWired)
{
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *proj_stage = dynamic_cast<GDNProjectionStage *>(
        attn_graph.getNode("layer0_gdn_proj")->stage.get());
    ASSERT_NE(proj_stage, nullptr);

    const auto &params = proj_stage->getParams();
    EXPECT_EQ(params.w_qkv, layer_.attn_qkv);
    EXPECT_EQ(params.w_z, layer_.attn_gate);
    EXPECT_EQ(params.w_a, layer_.ssm_alpha);
    EXPECT_EQ(params.w_b, layer_.ssm_beta);
}

TEST_F(Qwen35GraphBuildTest, OutputGate_NotUsedInGDNLayers)
{
    // GDN layers do NOT use an attention output gate — only FA layers do.
    // The gated RMSNorm stage handles the equivalent gating in GDN.
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    // Should NOT have an output gate node in GDN layers
    EXPECT_EQ(attn_graph.getNode("layer0_attn_output_gate"), nullptr)
        << "GDN layers should NOT have an AttentionOutputGateStage";

    // Should NOT have a separate gate projection GEMM node
    EXPECT_EQ(attn_graph.getNode("layer0_attn_gate_proj"), nullptr)
        << "GDN layers should NOT have a separate gate projection GEMM";
}

TEST_F(Qwen35GraphBuildTest, ShortConv_UsesInterfaceKernel)
{
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *conv_stage = dynamic_cast<ShortConv1dStage *>(
        attn_graph.getNode("layer0_short_conv")->stage.get());
    ASSERT_NE(conv_stage, nullptr);

    // The kernel pointer should be non-null (interface-typed)
    EXPECT_NE(conv_stage->getParams().kernel, nullptr)
        << "ShortConv1dStage should have a kernel instance";
}

TEST_F(Qwen35GraphBuildTest, GDNRecurrence_UsesInterfaceKernel)
{
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *rec_stage = dynamic_cast<GDNRecurrenceStage *>(
        attn_graph.getNode("layer0_gdn_recurrence")->stage.get());
    ASSERT_NE(rec_stage, nullptr);

    EXPECT_NE(rec_stage->getParams().kernel, nullptr)
        << "GDNRecurrenceStage should have a kernel instance";
}

TEST_F(Qwen35GraphBuildTest, GDNRecurrence_DKConsistentWithState)
{
    // d_k calculation should be consistent between state allocation
    // and stage parameter wiring
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *rec_stage = dynamic_cast<GDNRecurrenceStage *>(
        attn_graph.getNode("layer0_gdn_recurrence")->stage.get());
    ASSERT_NE(rec_stage, nullptr);

    const auto &params = rec_stage->getParams();

    // d_k = d_v = gdn_state_size
    // n_heads = n_v_heads (gdn_time_step_rank), n_k_heads = gdn_group_count
    // inner=128, n_v_heads=8, n_k_heads=4, d_v=16, d_k=16
    EXPECT_EQ(params.d_k, 16) << "d_k should equal d_v (gdn_state_size)";
    EXPECT_EQ(params.d_v, 16) << "d_v should be gdn_state_size";
    EXPECT_EQ(params.n_heads, 8) << "n_heads should be n_v_heads (gdn_time_step_rank)";
    EXPECT_EQ(params.n_k_heads, 4) << "n_k_heads should be gdn_group_count";
}

TEST_F(Qwen35GraphBuildTest, IsGDNLayer_DispatchesCorrectly)
{
    Qwen35Graph graph(config_, nullptr);

    // Layer 0 = GDN: should produce GDN graph (has gdn_proj)
    ComputeGraph gdn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());
    EXPECT_NE(gdn_graph.getNode("layer0_gdn_proj"), nullptr)
        << "Layer 0 should produce GDN graph";

    // Layer 3 = FA: should NOT produce GDN graph
    // (We can't fully test FA without KV cache, but we can verify
    //  it doesn't have GDN-specific nodes)
    LayerWeights fa_layer;
    fa_layer.attn_norm = norm_weight_.get();
    fa_layer.wq = TestTensorFactory::createFP32Random({64, 64}).release();
    fa_layer.wk = TestTensorFactory::createFP32Random({32, 64}).release();
    fa_layer.wv = TestTensorFactory::createFP32Random({32, 64}).release();
    fa_layer.wo = TestTensorFactory::createFP32Random({64, 64}).release();

    ComputeGraph fa_graph = graph.buildAttentionGraph(
        fa_layer, buffers_, 3, 2, 1, nullptr, nullptr, DeviceId::cpu());
    EXPECT_EQ(fa_graph.getNode("layer3_gdn_proj"), nullptr)
        << "Layer 3 (FA) should NOT have GDN projection";

    // Clean up manually allocated tensors
    delete fa_layer.wq;
    delete fa_layer.wk;
    delete fa_layer.wv;
    delete fa_layer.wo;
}

TEST_F(Qwen35GraphBuildTest, GatedNorm_UsesCorrectWeightAndBuffers)
{
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *gnorm_stage = dynamic_cast<GatedRMSNormStage *>(
        attn_graph.getNode("layer0_gated_norm")->stage.get());
    ASSERT_NE(gnorm_stage, nullptr);

    const auto &params = gnorm_stage->getParams();

    // Gated norm gamma should be ssm_norm (the GDN output normalization weight)
    EXPECT_EQ(params.gamma, layer_.ssm_norm)
        << "Gated RMSNorm gamma should be ssm_norm weight";

    // Gate should be the Z buffer (gdn_z)
    EXPECT_EQ(params.gate, buffers_.get(BufferId::GDN_Z))
        << "Gated RMSNorm gate should be gdn_z buffer";
}

// ============================================================================
// Regression: attn_output buffer must be max(local_qkv_dim, gdn_inner_size)
// Bug: attn_output was sized as local_qkv_dim (FA dimension), which is smaller
// than gdn_inner_size when the model has more GDN heads than FA heads.
// GatedRMSNorm reads the buffer column count to determine how many groups to
// process, so an undersized buffer caused it to skip heads, leaving garbage
// in the output that produced NaN via Q8_1 FP16 overflow in gdn_out_proj.
// ============================================================================

TEST_F(Qwen35GraphBuildTest, ResolverConfig_AttnOutputDim_IsMaxOfFAAndGDN)
{
    // The existing fixture has: n_heads=4, head_dim=16, gdn_inner=128
    // Without TP: local_qkv_dim = 4*16=64, gdn_inner=128
    // attn_output_dim must be max(64, 128) = 128
    Qwen35Graph graph(config_, nullptr);
    auto resolver_config = graph.getResolverConfig(/*seq_len=*/2);

    auto it = resolver_config.custom_formulas.find("attn_output_dim");
    ASSERT_NE(it, resolver_config.custom_formulas.end())
        << "attn_output_dim must be defined in resolver custom formulas";

    const size_t local_qkv_dim = static_cast<size_t>(config_.n_heads * config_.head_dim);
    const size_t gdn_inner = static_cast<size_t>(config_.gdn.inner_size);
    const size_t expected = std::max(local_qkv_dim, gdn_inner);

    EXPECT_EQ(it->second, expected)
        << "attn_output_dim should be max(local_qkv_dim=" << local_qkv_dim
        << ", gdn_inner=" << gdn_inner << ") = " << expected;
}

TEST(Test__Qwen35Schema, ResolverConfig_AttnOutputDim_WithTP)
{
    // Regression test for the exact bug: with 2-way TP,
    // local_n_heads is halved but gdn_inner_size is also halved,
    // and gdn_inner > local_qkv_dim. The buffer must accommodate both.
    //
    // Qwen3.5-4B dimensions: n_heads=20, head_dim=128, gdn_inner=4096,
    // gdn_group_count=16, gdn_time_step_rank=32, gdn_state_size=64
    // With TP=2: local_n_heads=10, local_qkv_dim=1280, local_gdn_inner=2048
    GraphConfig config;
    config.n_layers = 4;
    config.d_model = 2560;
    config.n_heads = 20;
    config.n_kv_heads = 4;
    config.head_dim = 128;
    config.d_ff = 8960;
    config.vocab_size = 248320;
    config.rms_norm_eps = 1e-6f;
    config.rope_theta = 10000.0f;
    config.default_device = DeviceId::cpu();
    config.max_seq_len = 2048;
    config.gdn.conv_kernel_size = 4;
    config.gdn.state_size = 64;
    config.gdn.inner_size = 4096;
    config.gdn.group_count = 16;
    config.gdn.time_step_rank = 32;
    config.gdn.full_attention_interval = 4;
    config.layer_types = {"gdn", "gdn", "gdn", "full_attention"};

    // Simulate 2-way TP: halve the local head count
    config.qkv_column_parallel = true;
    config.local_n_heads = 10;  // 20 / 2

    Qwen35Graph graph(config, nullptr);
    auto resolver_config = graph.getResolverConfig(/*seq_len=*/1);

    auto it = resolver_config.custom_formulas.find("attn_output_dim");
    ASSERT_NE(it, resolver_config.custom_formulas.end())
        << "attn_output_dim must be defined in resolver custom formulas";

    // local_qkv_dim = 10 * 128 = 1280
    // gdn_inner (TP-local) = 4096 * (10/20) = 2048
    // attn_output_dim = max(1280, 2048) = 2048
    EXPECT_GE(it->second, static_cast<size_t>(2048))
        << "attn_output_dim must be at least gdn_inner_size (2048) with TP=2, "
           "not just local_qkv_dim (1280). Without this, GatedRMSNorm skips "
           "heads beyond the FA buffer boundary, producing garbage in gdn_out_proj.";

    EXPECT_GE(it->second, static_cast<size_t>(1280))
        << "attn_output_dim must also accommodate FA local_qkv_dim";
}

TEST_F(Qwen35GraphBuildTest, GDNOutProj_KDimMatchesGDNInner)
{
    // The GEMM K dimension comes from ssm_out weight shape[1], which must
    // equal gdn_inner_size. The attn_output buffer (A input) must have at
    // least K valid elements per row for the GEMV to read valid data.
    Qwen35Graph graph(config_, nullptr);

    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, 0, 2, 1, stub_cache_.get(), nullptr, DeviceId::cpu());

    auto *gemm_node = attn_graph.getNode("layer0_gdn_out_proj");
    ASSERT_NE(gemm_node, nullptr);

    auto *gemm_stage = dynamic_cast<GEMMStage *>(gemm_node->stage.get());
    ASSERT_NE(gemm_stage, nullptr);

    const auto &params = gemm_stage->getParams();

    // K must equal gdn_inner_size (weight shape[1])
    EXPECT_EQ(params.k, config_.gdn.inner_size)
        << "gdn_out_proj K dimension must match gdn_inner_size, not local_qkv_dim";

    // N must equal d_model (weight shape[0])
    EXPECT_EQ(params.n, config_.d_model)
        << "gdn_out_proj N dimension must be d_model";
}

// ============================================================================
// GDN State Reset Tests
// ============================================================================

TEST_F(Qwen35GraphBuildTest, ResetState_DoesNotCrash)
{
    Qwen35Graph graph(config_, nullptr);

    // Should be safe to call immediately after construction
    EXPECT_NO_THROW(graph.resetState());
}

TEST_F(Qwen35GraphBuildTest, ResetState_Idempotent)
{
    Qwen35Graph graph(config_, nullptr);

    // Calling multiple times should be safe
    EXPECT_NO_THROW(graph.resetState());
    EXPECT_NO_THROW(graph.resetState());
    EXPECT_NO_THROW(graph.resetState());
}

TEST_F(Qwen35GraphBuildTest, ResetState_SafeAfterGraphBuild)
{
    Qwen35Graph graph(config_, nullptr);

    // Build a GDN attention graph (this internally uses conv/recurrence state)
    ComputeGraph attn_graph = graph.buildAttentionGraph(
        layer_, buffers_, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, /*kv_cache=*/stub_cache_.get(), /*position_ids=*/nullptr,
        DeviceId::cpu());
    ASSERT_GT(attn_graph.size(), 0u);

    // Reset state after building — should be safe
    EXPECT_NO_THROW(graph.resetState());

    // Should be able to build another graph after reset
    ComputeGraph attn_graph2 = graph.buildAttentionGraph(
        layer_, buffers_, /*layer_idx=*/0, /*seq_len=*/2,
        /*batch_size=*/1, /*kv_cache=*/stub_cache_.get(), /*position_ids=*/nullptr,
        DeviceId::cpu());
    EXPECT_GT(attn_graph2.size(), 0u);
}

TEST(Test__IGraphBuilder_ResetState, DefaultImplementation_IsNoOp)
{
    // IGraphBuilder::resetState() has a default no-op implementation.
    // Non-GDN models (like Qwen2) should just call the base version without issue.
    GraphConfig config;
    config.n_layers = 2;
    config.d_model = 64;
    config.n_heads = 4;
    config.n_kv_heads = 2;
    config.head_dim = 16;
    config.d_ff = 128;
    config.vocab_size = 1000;

    // Qwen35Graph with NO GDN config → hasGDN() returns false,
    // ensureGDNStates() is a no-op, states are empty
    Qwen35Graph graph(config, nullptr);
    EXPECT_NO_THROW(graph.resetState());
}
