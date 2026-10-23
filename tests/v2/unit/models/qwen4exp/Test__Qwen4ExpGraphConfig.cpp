/**
 * @file Test__Qwen4ExpGraphConfig.cpp
 * @brief Graph role widths, exact PP layer spans and predictor exclusion regressions.
 *
 * Synthetic immutable directories exercise model configuration without loading
 * weights or devices. The tests prevent a widened residual from accidentally
 * widening attention/expert GEMMs or the vocabulary head.
 */
#include "Qwen4ExpMetadataFixture.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "execution/prefix_cache/PrefixCacheFingerprint.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "models/qwen4exp/Qwen4ExpSchema.h"
#include "models/GraphTypes.h"
#include "models/TerminalInputWeights.h"
#include "tensors/Tensors.h"
#include <gtest/gtest.h>

namespace llaminar2::test
{
/** @test Actual model hook retains parent state and its base snapshot after setup reload. */
TEST(Qwen4ExpGraphConfig, AttentionBasePrefixIdentitySurvivesSetupReload)
{
    using A=qwen4exp::QwenSparseAttentionArithmetic;
    auto &setup=mutableDebugEnv().attention;
    struct Restore { AttentionConfig &setup; AttentionConfig previous; ~Restore(){setup=previous;} } restore{setup,setup};
    GraphConfig config;config.default_device=DeviceId::rocm(0);
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false),config);
    config.attention_phase_schedule=qwen4exp::QwenSparseAttentionPhaseSchedule::Uniform;
    setup.qfn_sparse_attention=A::FixedFP32;Qwen4ExpGraph ordinary(config);
    setup.qfn_sparse_attention=A::TileFP16WithFP32Totals;Qwen4ExpGraph candidate(config);
    PrefixFingerprintMaterial before;candidate.appendPrefixCacheFingerprintMaterial(before);
    setup.reload();setup.qfn_sparse_attention=A::FixedFP32;
    PrefixFingerprintMaterial after,control,parent;
    candidate.appendPrefixCacheFingerprintMaterial(after);
    ordinary.appendPrefixCacheFingerprintMaterial(control);
    candidate.Qwen35MoEGraph::appendPrefixCacheFingerprintMaterial(parent);
    EXPECT_EQ(hashPrefixFingerprintFields("runtime",before.runtime),hashPrefixFingerprintFields("runtime",after.runtime));
    EXPECT_NE(hashPrefixFingerprintFields("runtime",after.runtime),hashPrefixFingerprintFields("runtime",control.runtime));
    ASSERT_FALSE(parent.moe.empty());ASSERT_EQ(after.moe.size(),parent.moe.size());
    for(std::size_t i=0;i<parent.moe.size();++i) {
        EXPECT_EQ(after.moe[i].name,parent.moe[i].name);EXPECT_EQ(after.moe[i].value,parent.moe[i].value);
    }
}

/** @brief An admitted graph owns its policy even if a later diagnostic changes setup. */
TEST(Qwen4ExpGraphConfig, HCUpActivationPolicyIsFrozenForPlanningAndReplay)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision previous;
        ~Restore() { setup.qfn_hc_up_activation = previous; }
    } restore{setup, setup.qfn_hc_up_activation};
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    GraphConfig candidate;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), candidate);
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
    GraphConfig ordinary;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), ordinary);
    EXPECT_EQ(candidate.qfn_hc_up_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_EQ(ordinary.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
    const auto copied = candidate;
    EXPECT_EQ(copied.qfn_hc_up_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    setup.qfn_hc_up_activation = kDefaultQwen4ExpHCUpActivation;
    GraphConfig default_graph;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), default_graph);
    EXPECT_EQ(default_graph.qfn_hc_up_activation, candidate.qfn_hc_up_activation);
    EXPECT_NE(default_graph.qfn_hc_up_activation, ordinary.qfn_hc_up_activation);
    EXPECT_EQ(GraphConfig{}.qfn_hc_up_activation, kDefaultQwen4ExpHCUpActivation);
}

TEST(Qwen4ExpTerminalWeights, NativeHeadInventoryAndFrozenBindingsReplaceOrdinaryNorm)
{
    Qwen4ExpSchemaFactory schema;
    const auto names = schema.terminalInputWeightNames();
    ASSERT_EQ(names.size(), 3u);
    EXPECT_STREQ(names[0], "output_hc_norm.weight");
    EXPECT_STREQ(names[1], "output_hc_down.weight");
    EXPECT_STREQ(names[2], "output_hc_up.weight");
    Qwen35MoESchemaFactory ordinary;
    ASSERT_EQ(ordinary.terminalInputWeightNames().size(), 1u);
    EXPECT_STREQ(ordinary.terminalInputWeightNames()[0], "output_norm.weight");

    InferenceStrategy strategy;
    strategy.model_id = ModelContextId{19};
    strategy.devices = {DeviceId::cpu()};
    ModelWeightSetBuilder builder(strategy);
    std::array<std::unique_ptr<FP32Tensor>, 3> operands;
    for (std::size_t index = 0; index < names.size(); ++index)
    {
        EXPECT_FALSE(schema.isWeightOptional(names[index]));
        operands[index] = std::make_unique<FP32Tensor>(std::vector<std::size_t>{1, 4});
        WeightBinding binding;
        binding.identity = makeSourceWeightIdentity(names[index], strategy.model_id, 0);
        binding.tensor = operands[index].get();
        binding.residency.home_device = DeviceId::cpu();
        binding.residency.resident_device = DeviceId::cpu();
        builder.addBinding(std::move(binding));
    }
    FrozenModelWeightSet frozen(strategy, builder.freezeBindings());
    const auto bindings = makeModelWeightBindings(frozen);
    ASSERT_NE(bindings.hyper_connection_head.norm, nullptr);
    ASSERT_NE(bindings.hyper_connection_head.down, nullptr);
    ASSERT_NE(bindings.hyper_connection_head.up, nullptr);
    EXPECT_EQ(bindings.hyper_connection_head.norm->identity.role, WeightRole::HyperConnectionNorm);
    EXPECT_EQ(bindings.hyper_connection_head.down->identity.role, WeightRole::HyperConnectionProjection);
    EXPECT_EQ(bindings.hyper_connection_head.up->identity.role, WeightRole::HyperConnectionProjection);
    const auto views = toLegacyModelWeights(bindings);
    EXPECT_EQ(views.hyper_connection_head.norm, operands[0].get());
    EXPECT_EQ(views.hyper_connection_head.down, operands[1].get());
    EXPECT_EQ(views.hyper_connection_head.up, operands[2].get());
    EXPECT_EQ(views.hyper_connection_head.inject, nullptr);
    EXPECT_EQ(views.final_norm, nullptr);
    GraphConfig flash;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), flash);
    EXPECT_TRUE(hasRequiredTerminalInputWeights(flash, views));
    for (const auto member : {&HyperConnectionWeights<TensorBase *>::norm,
                              &HyperConnectionWeights<TensorBase *>::down,
                              &HyperConnectionWeights<TensorBase *>::up})
    {
        auto incomplete = views;
        incomplete.hyper_connection_head.*member = nullptr;
        incomplete.final_norm = operands[0].get();
        EXPECT_FALSE(hasRequiredTerminalInputWeights(flash, incomplete));
    }
    GraphConfig ordinary_config;
    EXPECT_FALSE(hasRequiredTerminalInputWeights(ordinary_config, views));
    auto ordinary_weights = views;
    ordinary_weights.final_norm = operands[0].get();
    EXPECT_TRUE(hasRequiredTerminalInputWeights(ordinary_config, ordinary_weights));
}

TEST(Qwen4ExpGraphConfig, SeparatesResidualTransportFromComputeAndOutputHead)
{
    GraphConfig config;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(), config);
    ASSERT_NE(config.flash_next, nullptr);
    EXPECT_EQ(config.n_layers, 48);
    EXPECT_EQ(config.total_n_layers, 48);
    EXPECT_EQ(config.d_model, 2560);
    EXPECT_EQ(config.residualWidth(), 10240);
    EXPECT_EQ(config.terminalHeadWidth(), 2560);
    EXPECT_EQ(config.n_heads, 24);
    EXPECT_EQ(config.n_kv_heads, 2);
    EXPECT_EQ(config.moe.intermediate_size, 640);
    EXPECT_EQ(config.moe.num_experts, 512);
    EXPECT_EQ(config.moe.top_k, 10);
    EXPECT_EQ(config.gdn.inner_size, 6144);
    EXPECT_FLOAT_EQ(config.partial_rotary_factor, 0.25f);
    config.d_model = 10240;
    EXPECT_THROW(config.residualWidth(), std::invalid_argument);
    EXPECT_THROW(config.terminalHeadWidth(), std::invalid_argument);
}

TEST(Qwen4ExpGraphConfig, PipelineRangesPreserveGlobalMixerIdentity)
{
    for (const int stages : {1, 2, 4, 8})
    {
        for (int rank = 0; rank < stages; ++rank)
        {
            GraphConfig config;
            config.n_layers = 48 / stages;
            config.pp_layer_offset = rank * config.n_layers;
            config.default_device = DeviceId::cpu();
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(), config);
            EXPECT_EQ(config.n_layers, 48 / stages);
            EXPECT_EQ(config.pp_layer_offset, rank * (48 / stages));
            EXPECT_EQ(config.default_device, DeviceId::cpu());
            EXPECT_EQ(config.moe.decode_histogram_token_boundary_layer, config.n_layers - 1);
            ASSERT_EQ(config.layer_types.size(), 48u);
            ASSERT_EQ(config.layer_head_dim.size(), 48u);
            for (int layer = config.pp_layer_offset; layer < config.pp_layer_offset + config.n_layers; ++layer)
            {
                const bool sparse = (layer + 1) % 4 == 0;
                EXPECT_EQ(config.layer_types[layer], sparse ? "full_attention" : "gdn");
                EXPECT_EQ(config.layer_head_dim[layer], sparse ? 256 : 128);
            }
        }
    }
    GraphConfig invalid;
    invalid.pp_layer_offset = 24;
    invalid.n_layers = 25;
    EXPECT_THROW(Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(), invalid),
                 std::invalid_argument);
    invalid.pp_layer_offset = -1;
    EXPECT_THROW(Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(), invalid),
                 std::invalid_argument);
}

TEST(Qwen4ExpGraphConfig, PredictorIsExcludedOnlyFromACompleteExportInterval)
{
    GraphConfig full;
    full.n_layers = 49;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(), full);
    EXPECT_EQ(full.n_layers, 48);
    EXPECT_EQ(full.flash_next->predictorLayers(), 1u);
    GraphConfig plain;
    plain.d_model = 896;
    EXPECT_EQ(plain.residualWidth(), 896);
    EXPECT_EQ(plain.terminalHeadWidth(), 896);
    GraphConfig without_predictor;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), without_predictor);
    EXPECT_EQ(without_predictor.flash_next->predictorLayers(), 0u);
}
} // namespace llaminar2::test

namespace llaminar2::test
{
/** @test A frozen output-projection policy is independent of later setup and HC-up selection. */
TEST(Qwen4ExpGraphConfig, GDNOutputActivationPolicyIsFrozenAndIndependent)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision hc, gdn;
        ~Restore() { setup.qfn_hc_up_activation = hc; setup.qfn_gdn_output_activation = gdn; }
    } restore{setup, setup.qfn_hc_up_activation, setup.qfn_gdn_output_activation};
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
    setup.qfn_gdn_output_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    GraphConfig candidate;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), candidate);
    setup.qfn_gdn_output_activation = QuantizedActivationOperandPrecision::SingleQ8;
    GraphConfig ordinary;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), ordinary);
    EXPECT_EQ(candidate.qfn_gdn_output_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_EQ(candidate.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
    EXPECT_EQ(ordinary.qfn_gdn_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
    const auto frozen = candidate;
    EXPECT_EQ(frozen.qfn_gdn_output_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_EQ(GraphConfig{}.qfn_gdn_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test FA selection freezes at setup and leaves independent HC/GDN defaults unchanged. */
TEST(Qwen4ExpGraphConfig, FAOutputActivationPolicyIsFrozenAndIndependent)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision hc, gdn;
        ~Restore() { setup.qfn_hc_up_activation = hc; setup.qfn_fa_output_activation = gdn; }
    } restore{setup, setup.qfn_hc_up_activation, setup.qfn_fa_output_activation};
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
    setup.qfn_fa_output_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    GraphConfig candidate;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), candidate);
    setup.qfn_fa_output_activation = QuantizedActivationOperandPrecision::SingleQ8;
    GraphConfig ordinary;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), ordinary);
    EXPECT_EQ(candidate.qfn_fa_output_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_EQ(candidate.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
    EXPECT_EQ(ordinary.qfn_fa_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
    const auto frozen = candidate;
    EXPECT_EQ(frozen.qfn_fa_output_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_EQ(GraphConfig{}.qfn_fa_output_activation, QuantizedActivationOperandPrecision::SingleQ8);
}

/** @test A copied graph retains QKV-only policy after setup changes; default remains SingleQ8. */
TEST(Qwen4ExpGraphConfig, GDNQKVActivationPolicyIsFrozenAndIndependent)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision hc, gdn;
        ~Restore() { setup.qfn_hc_up_activation = hc; setup.qfn_gdn_qkv_activation = gdn; }
    } restore{setup, setup.qfn_hc_up_activation, setup.qfn_gdn_qkv_activation};
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
    setup.qfn_gdn_qkv_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    GraphConfig candidate;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), candidate);
    setup.qfn_gdn_qkv_activation = QuantizedActivationOperandPrecision::SingleQ8;
    GraphConfig ordinary;
    Qwen4ExpGraphConfigBuilder::populateFromDirectory(qwen4MetadataFixture(false), ordinary);
    EXPECT_EQ(candidate.qfn_gdn_qkv_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_EQ(candidate.qfn_hc_up_activation, QuantizedActivationOperandPrecision::SingleQ8);
    EXPECT_EQ(ordinary.qfn_gdn_qkv_activation, QuantizedActivationOperandPrecision::SingleQ8);
    const auto frozen = candidate;
    EXPECT_EQ(frozen.qfn_gdn_qkv_activation, QuantizedActivationOperandPrecision::TwoTermQ8);
    EXPECT_EQ(GraphConfig{}.qfn_gdn_qkv_activation, QuantizedActivationOperandPrecision::SingleQ8);
}

}
