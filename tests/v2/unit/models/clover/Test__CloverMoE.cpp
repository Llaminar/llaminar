/**
 * @file Test__CloverMoE.cpp
 * @brief Device-free Clover source, graph, weight-binding and admission contracts.
 *
 * Exercise the production architecture registration and configuration, including
 * the official Preview's implicit shared expert, exact global PP layer identity,
 * and exclusion of diffusion-only operands from the causal memory BOM. These
 * structural tests do not claim real-weight numerical or GPU qualification.
 */
#include "CloverMoEMetadataFixture.h"
#include "CloverMoESourceFixture.h"
#include "loaders/ModelContext.h"
#include "mocks/MockModelContext.h"
#include "models/clover/CloverMoEConfig.h"
#include "models/clover/CloverMoEGraphConfigBuilder.h"
#include "models/clover/CloverMoEGraph.h"
#include "models/ModelGenerationPolicy.h"
#include "execution/local_execution/graph/GraphBuilderRegistry.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "loaders/PreparedWeightRepresentationContract.h"
#include "planning/ModelMemoryProfile.h"
#include "utils/TestTensorFactory.h"
#include "mocks/MockModelLoader.h"
#include "mocks/MockTokenizer.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <unordered_map>

namespace llaminar2::test
{
    TEST(CloverMoEContract, PublishedPreviewHasThirtyGDNAndTenAttentionBlocks)
    {
        const auto source = cloverMetadataFixture(CloverFixtureExport::Preview, CloverFixtureGeometry::Published35B);
        ASSERT_EQ(source.tensors.size(), 853u);
        const CloverMoEConfig exact(source);
        GraphConfig graph;
        exact.populate(graph);
        EXPECT_EQ(graph.n_layers, 40);
        EXPECT_EQ(graph.d_model, 2048);
        EXPECT_EQ(graph.vocab_size, 248320);
        EXPECT_EQ(graph.n_heads, 16);
        EXPECT_EQ(graph.n_kv_heads, 2);
        EXPECT_EQ(graph.head_dim, 256);
        EXPECT_FLOAT_EQ(graph.partial_rotary_factor, 0.25f);
        EXPECT_EQ(std::count(graph.layer_types.begin(), graph.layer_types.end(), "gdn"), 30);
        EXPECT_EQ(std::count(graph.layer_types.begin(), graph.layer_types.end(), "full_attention"), 10);
        EXPECT_EQ(graph.gdn.group_count, 16);
        EXPECT_EQ(graph.gdn.time_step_rank, 32);
        EXPECT_EQ(graph.gdn.state_size, 128);
        EXPECT_EQ(graph.moe.num_experts, 256);
        EXPECT_EQ(graph.moe.top_k, 8);
        EXPECT_TRUE(graph.moe.norm_topk_prob);
        EXPECT_TRUE(graph.moe.has_shared_expert);
        EXPECT_TRUE(graph.moe.shared_expert_gate);
        EXPECT_EQ(graph.moe.shared_intermediate_size, 512);
    }

    TEST(CloverMoEContract, PreviewAndExplicitARPublishTheSameCausalBOM)
    {
        const auto preview = cloverMetadataFixture();
        const auto ar = cloverMetadataFixture(CloverFixtureExport::Autoregressive);
        const auto a = ModelMemoryProfile::fromGGUF(preview);
        const auto b = ModelMemoryProfile::fromGGUF(ar);
        EXPECT_EQ(a.total_native_bytes, b.total_native_bytes);
        EXPECT_EQ(a.tensors.size(), ar.tensors.size());
        EXPECT_EQ(a.tensors.size(), b.tensors.size());
        EXPECT_EQ(a.expert_shared_feed_forward_length, 256);
        EXPECT_EQ(a.mtp_layer_count, 0);
        for (const auto &tensor : a.tensors) EXPECT_FALSE(CloverMoEConfig::isDiffusionOnlyTensor(tensor.name));
        const auto bytes = a.serialize();
        const auto restored = ModelMemoryProfile::deserialize(bytes.data(), bytes.size());
        EXPECT_EQ(restored.architecture, "clover_moe");
        EXPECT_EQ(restored.total_native_bytes, a.total_native_bytes);
        EXPECT_EQ(restored.full_attention_interval, 4);
        EXPECT_EQ(restored.gdn_time_step_rank, 2);
    }

    TEST(CloverMoEContract, RealParserPreservesSourceButMaterializesOnlyCausalOperands)
    {
        for (const auto encoding : {CloverFixtureExport::Preview, CloverFixtureExport::Autoregressive})
        {
            const auto original = cloverMetadataFixture(encoding);
            CloverMoESourceFixture source(original);
            ModelLoader loader;
            loader.setUseMmap(false);
            ASSERT_TRUE(loader.loadModel(source.path()));
            EXPECT_EQ(loader.architecture(), "clover_moe");
            EXPECT_EQ(loader.tensorNames().size(), original.tensors.size());
            EXPECT_NO_THROW((void)CloverMoEConfig(*loader.ggufDirectory()));
            const auto causal = loader.materializationNames();
            const auto profile = ModelMemoryProfile::fromGGUF(*loader.ggufDirectory());
            EXPECT_EQ(causal.size(), profile.tensors.size());
            size_t consumed_bytes = 0;
            for (const auto &name : causal)
            {
                EXPECT_FALSE(CloverMoEConfig::isDiffusionOnlyTensor(name));
                const auto *tensor = loader.getModel().findTensor(name);
                ASSERT_NE(tensor, nullptr);
                consumed_bytes += tensor->size_bytes;
            }
            EXPECT_EQ(consumed_bytes, profile.total_native_bytes);
            if (encoding == CloverFixtureExport::Preview)
            {
                EXPECT_EQ(original.tensors.size() - causal.size(), 12u);
                EXPECT_TRUE(loader.hasTensor("blk.0.ssm_conv1d.weight_right"));
                EXPECT_GT(loader.totalBytes(), consumed_bytes);
            }
            else EXPECT_EQ(loader.totalBytes(), consumed_bytes);
        }
    }

    TEST(CloverMoEContract, PublicContextConfigurationRetainsTheExactSourceAndPPInterval)
    {
        CloverMoESourceFixture source;
        auto context = ModelContext::createForPPStage(source.path(), 2, 4, false, true);
        ASSERT_NE(context, nullptr);
        GraphConfig graph;
        graph.pp_layer_offset = 2;
        graph.max_seq_len = 96;
        ASSERT_TRUE(createGraphConfigBuilder("clover_moe")->populateFromModelContext(*context, graph));
        EXPECT_EQ(graph.n_layers, 2);
        EXPECT_EQ(graph.total_n_layers, 4);
        EXPECT_EQ(graph.max_seq_len, 96); // Admission, not source capacity, owns this value.
        EXPECT_EQ(graph.layer_head_dim, (std::vector<int>{128, 128, 128, 64}));
        EXPECT_EQ(graph.moe.shared_intermediate_size, 256);
        EXPECT_TRUE(context->loader()->hasTensor("blk.0.ssm_bd_history.weight"));
        MockModelContext no_directory;
        no_directory.setArchitecture("clover_moe");
        EXPECT_THROW(createGraphConfigBuilder("clover_moe")->populateFromModelContext(no_directory, graph),
            std::invalid_argument);
    }

    TEST(CloverMoEContract, EveryRequiredScalarAndCausalTensorIsAuthenticated)
    {
        const auto complete = cloverMetadataFixture(CloverFixtureExport::Autoregressive);
        for (const auto &[key, value] : complete.metadata)
        {
            if (key == "clover_moe.attention.recurrent_layers" || key == "clover_moe.expert_shared_count") continue;
            SCOPED_TRACE(key);
            auto missing = complete;
            missing.metadata.erase(key);
            EXPECT_THROW((void)CloverMoEConfig(missing), std::invalid_argument);
        }
        for (size_t index = 0; index < complete.tensors.size(); ++index)
        {
            if (complete.tensors[index].name == "output.weight") continue; // Explicit tied-embedding contract.
            SCOPED_TRACE(complete.tensors[index].name);
            auto missing = complete;
            missing.tensors.erase(missing.tensors.begin() + index);
            EXPECT_THROW((void)CloverMoEConfig(missing), std::invalid_argument);
            auto wrong = complete;
            wrong.tensors[index].dimensions.back() += 1;
            EXPECT_THROW((void)CloverMoEConfig(wrong), std::invalid_argument);
        }
    }

    TEST(CloverMoEContract, RejectsContradictoryGeometryAndUnimplementedModes)
    {
        for (const auto &[key, value] : std::vector<std::pair<std::string, uint32_t>>{
            {"attention.value_length", 128}, {"attention.head_count_kv", 3}, {"rope.dimension_count", 3},
            {"ssm.group_count", 3}, {"ssm.inner_size", 255}, {"ssm.conv_kernel", 3},
            {"expert_used_count", 5}, {"full_attention_interval", 0}, {"full_attention_interval", 1},
            {"nextn_predict_layers", 1}, {"expert_shared_count", 2}, {"expert_shared_count", 0},
            {"embedding_length", std::numeric_limits<uint32_t>::max()}})
        {
            SCOPED_TRACE(key);
            auto source = cloverMetadataFixture();
            cloverInteger(source, key, value);
            EXPECT_THROW((void)CloverMoEConfig(source), std::invalid_argument);
        }
        auto source = cloverMetadataFixture();
        source.architecture = "qwen35moe";
        EXPECT_THROW((void)CloverMoEConfig(source), std::invalid_argument);
        source = cloverMetadataFixture();
        source.metadata["clover_moe.rope.freq_base"] = cloverScalar(GGUFValueType::FLOAT32, std::numeric_limits<float>::quiet_NaN());
        EXPECT_THROW((void)CloverMoEConfig(source), std::invalid_argument);
        source = cloverMetadataFixture();
        source.tensors.push_back(source.tensors.front());
        EXPECT_THROW((void)CloverMoEConfig(source), std::invalid_argument);
        source = cloverMetadataFixture();
        source.tensors.front().name = "unsupported.weight";
        EXPECT_THROW((void)CloverMoEConfig(source), std::invalid_argument);
    }

    TEST(CloverMoEContract, PreviewDiffusionInventoryIsCompleteOrAbsent)
    {
        const auto preview = cloverMetadataFixture();
        for (size_t index = 0; index < preview.tensors.size(); ++index)
        {
            if (!CloverMoEConfig::isDiffusionOnlyTensor(preview.tensors[index].name)) continue;
            SCOPED_TRACE(preview.tensors[index].name);
            auto partial = preview;
            partial.tensors.erase(partial.tensors.begin() + index);
            EXPECT_THROW((void)CloverMoEConfig(partial), std::invalid_argument);
            auto wrong = preview;
            wrong.tensors[index].dimensions.back() += 1;
            EXPECT_THROW((void)CloverMoEConfig(wrong), std::invalid_argument);
        }
        auto causal = preview;
        std::erase_if(causal.tensors, [](const auto &tensor) {
            return CloverMoEConfig::isDiffusionOnlyTensor(tensor.name);
        });
        EXPECT_NO_THROW((void)CloverMoEConfig(causal));
    }

    TEST(CloverMoEContract, RecurrentMaskMustMatchItsCompletePeriodicSource)
    {
        for (int mutation = 0; mutation < 4; ++mutation)
        {
            auto source = cloverMetadataFixture();
            auto &mask = source.metadata["clover_moe.attention.recurrent_layers"];
            if (mutation == 0) mask.data[3] = 1;
            if (mutation == 1) mask.array_length = 3;
            if (mutation == 2) mask.array_element_type = GGUFValueType::UINT8;
            if (mutation == 3) mask.data[0] = 2;
            EXPECT_THROW((void)CloverMoEConfig(source), std::invalid_argument);
        }
        auto source = cloverMetadataFixture();
        source.metadata.erase("clover_moe.attention.recurrent_layers");
        EXPECT_NO_THROW((void)CloverMoEConfig(source));
    }

    TEST(CloverMoEContract, SharedExpertWidthIsIndependentAndAbsenceIsExplicit)
    {
        auto source = cloverMetadataFixture(CloverFixtureExport::Autoregressive);
        cloverInteger(source, "expert_shared_feed_forward_length", 512);
        for (auto &tensor : source.tensors)
        {
            if (tensor.name.ends_with("ffn_gate_shexp.weight") || tensor.name.ends_with("ffn_up_shexp.weight"))
                tensor.dimensions[0] = 512;
            if (tensor.name.ends_with("ffn_down_shexp.weight")) tensor.dimensions[1] = 512;
        }
        GraphConfig graph;
        CloverMoEConfig(source).populate(graph);
        EXPECT_EQ(graph.moe.intermediate_size, 256);
        EXPECT_EQ(graph.moe.shared_intermediate_size, 512);
        EXPECT_EQ(ModelMemoryProfile::fromGGUF(source).expert_shared_feed_forward_length, 512);
        cloverInteger(source, "expert_shared_count", 0);
        cloverInteger(source, "expert_shared_feed_forward_length", 0);
        std::erase_if(source.tensors, [](const auto &tensor) { return tensor.name.ends_with("_shexp.weight"); });
        CloverMoEConfig(source).populate(graph);
        EXPECT_FALSE(graph.moe.has_shared_expert);
        EXPECT_FALSE(graph.moe.shared_expert_gate);
        EXPECT_EQ(graph.moe.shared_intermediate_size, 0);
    }

    TEST(CloverMoEContract, PipelineProjectionPreservesGlobalLayerIdentity)
    {
        GraphConfig graph;
        graph.pp_layer_offset = 2;
        graph.n_layers = 2;
        const CloverMoEConfig exact(cloverMetadataFixture());
        exact.populate(graph);
        EXPECT_EQ(graph.n_layers, 2);
        EXPECT_EQ(graph.total_n_layers, 4);
        EXPECT_EQ(graph.layer_types[2], "gdn");
        EXPECT_EQ(graph.layer_types[3], "full_attention");
        EXPECT_EQ(graph.moe.decode_histogram_token_boundary_layer, 1);
        auto builder = GraphBuilderRegistry::create("clover_moe", graph, nullptr);
        EXPECT_EQ(builder->architectureName(), "clover_moe");
        const auto schema = builder->getSchema();
        EXPECT_EQ(schema.name, "clover_moe");
        EXPECT_EQ(schema.layer_template_names, (std::vector<std::string>{"gdn", "full_attention"}));
        graph.n_layers = 3;
        EXPECT_THROW(exact.populate(graph), std::invalid_argument);
    }

    TEST(CloverMoEContract, RegisteredSchemaRequiresRouterAndKeepsNativeExpertPlacement)
    {
        ASSERT_TRUE(GraphBuilderRegistry::isSupported("clover_moe"));
        auto schema = SchemaFactoryRegistry::getFactory("clover_moe");
        ASSERT_NE(schema, nullptr);
        EXPECT_EQ(schema->architectureName(), "clover_moe");
        EXPECT_EQ(schema->createSchema().name, "clover_moe");
        for (const auto *name : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_inp.weight"})
        {
            EXPECT_FALSE(schema->isWeightOptional(name));
            EXPECT_FALSE(schema->isWeightOptional(std::string("blk.3.") + name));
        }
        EXPECT_TRUE(schema->isWeightOptional("blk.0.attn_q.weight"));
        const auto sharding = schema->getWeightShardingConfig();
        for (const auto &pattern : sharding.patterns)
            if (pattern.pattern == "ffn_up_exps.weight") EXPECT_EQ(pattern.mode, WeightShardingMode::ExpertIdApportioned);
        EXPECT_NE(dynamic_cast<CloverMoEGraphConfigBuilder *>(createGraphConfigBuilder("clover_moe").get()), nullptr);
    }

    TEST(CloverMoEContract, BindsNativeCausalWeightsWithoutDiffusionOrNormReinterpretation)
    {
        std::unordered_map<std::string, std::shared_ptr<TensorBase>> tensors;
        for (const auto *name : {"token_embd.weight", "output_norm.weight", "blk.0.attn_qkv.weight",
             "blk.0.attn_gate.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_conv1d.weight",
             "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight", "blk.0.ssm_norm.weight", "blk.0.ssm_out.weight",
             "blk.0.post_attention_norm.weight", "blk.0.ffn_gate_inp.weight", "blk.0.ffn_gate_exps.weight",
             "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight", "blk.0.ffn_gate_shexp.weight",
             "blk.0.ffn_up_shexp.weight", "blk.0.ffn_down_shexp.weight", "blk.0.ffn_gate_inp_shexp.weight"})
            tensors.emplace(name, TestTensorFactory::createFP32({1}));
        const auto weights = createGraphConfigBuilder("clover_moe")->buildWeights([&](const std::string &name) {
            EXPECT_FALSE(CloverMoEConfig::isDiffusionOnlyTensor(name));
            const auto found = tensors.find(name);
            return found == tensors.end() ? std::shared_ptr<TensorBase>{} : found->second;
        });
        EXPECT_EQ(weights.lm_head, weights.embedding_table); // Tied head, not substituted architecture.
        EXPECT_EQ(weights.final_norm, tensors.at("output_norm.weight").get());
        const auto layer = weights.get_layer_weights(0);
        EXPECT_EQ(layer.ssm_a, tensors.at("blk.0.ssm_a").get());
        EXPECT_EQ(layer.ssm_conv1d, tensors.at("blk.0.ssm_conv1d.weight").get());
        EXPECT_EQ(layer.ffn_norm, tensors.at("blk.0.post_attention_norm.weight").get());
        EXPECT_EQ(layer.shared_expert_gate_inp, tensors.at("blk.0.ffn_gate_inp_shexp.weight").get());
        EXPECT_EQ(layer.moe_down_exps, tensors.at("blk.0.ffn_down_exps.weight").get());
    }

    TEST(CloverMoEContract, SharedGraphPreservesSamplingAndEmbeddedTemplateOwnership)
    {
        MockModelLoader loader;
        loader.setArchitecture("clover_moe");
        loader.setStringParam("general.name", "Clover-35B-A3B-Preview");
        loader.setStringParam("general.base_model.0.name", "Qwen3.5-35B-A3B");
        const auto policy = ModelGenerationPolicy::fromMetadata(loader);
        ::testing::StrictMock<MockTokenizer> tokenizer;
        EXPECT_CALL(tokenizer, setChatTemplate).Times(0);
        policy.applyChatTemplate(tokenizer);
        for (const auto mode : {ThinkingMode::Enabled, ThinkingMode::Disabled})
        {
            EXPECT_FLOAT_EQ(policy.forMode(mode).temperature, 1.0f);
            EXPECT_FLOAT_EQ(policy.forMode(mode).top_p, 1.0f);
            EXPECT_EQ(policy.forMode(mode).top_k, 0);
            EXPECT_FLOAT_EQ(policy.forMode(mode).presence_penalty, 0.0f);
        }
    }

    TEST(CloverMoEContract, GDNArithmeticPolicyHasTheSameExactSourceScope)
    {
        using Policy = PreparedWeightRepresentationContract;
        EXPECT_EQ(Policy::effectiveGDNArithmetic(WeightRole::GDNAlphaBetaProjection, "blk.0.ssm_alpha.weight",
            "clover_moe", GDNProjectionArithmetic::FP16MoEAlphaBeta), GDNProjectionArithmetic::FP16);
        EXPECT_EQ(Policy::effectiveGDNArithmetic(WeightRole::FusedQKV, "blk.0.attn_qkv.weight",
            "clover_moe", GDNProjectionArithmetic::FP16MoEAlphaBeta), GDNProjectionArithmetic::Native);
        EXPECT_FALSE(Policy::isGDNProjectionSource(WeightRole::FusedQKV, "blk.3.attn_q.weight", "clover_moe"));
        EXPECT_FALSE(CloverMoEConfig::isDiffusionOnlyTensor("blk.bad.ssm_conv1d.weight_right"));
        EXPECT_FALSE(CloverMoEConfig::isDiffusionOnlyTensor("blk.0.ssm_conv1d.weight"));
    }
}
