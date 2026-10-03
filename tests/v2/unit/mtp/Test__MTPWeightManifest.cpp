/**
 * @file Test__MTPWeightManifest.cpp
 * @brief Device-free proof of learned-predictor inventory and ownership admission.
 *
 * A plain model must reject retained MTP before pricing or allocating auxiliary
 * state. Metadata-only, incomplete and absent predictors are distinct negative
 * cases; valid NextN/dedicated layouts retain the canonical source identities.
 */
#include <gtest/gtest.h>

#include "execution/mtp/MTPWeightManifest.h"
#include "execution/mtp/MTPLearnedBlockCount.h"
#include "loaders/ModelLoader.h"
#include "loaders/WeightPlan.h"
#include "models/GraphTypes.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "mocks/MockModelLoader.h"
#include "tensors/Tensors.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

namespace
{
    /** @brief Add a small source role without materializing a model or device. */
    void addTensor(MockModelLoaderBuilder &builder, const std::string &name)
    {
        builder.addFP32Tensor(name, {4, 4});
    }

    /** @brief Populate every mandatory role of one dense NextN predictor. */
    void addNextNDepth(MockModelLoaderBuilder &builder, int source_layer)
    {
        const std::string prefix = "blk." + std::to_string(source_layer) + ".";
        for (const auto *suffix : {
                 "nextn.eh_proj.weight",
                 "nextn.hnorm.weight",
                 "nextn.enorm.weight",
                 "nextn.shared_head_norm.weight",
                 "attn_norm.weight",
                 "attn_q.weight",
                 "attn_k.weight",
                 "attn_v.weight",
                 "attn_output.weight",
                 "attn_q_norm.weight",
                 "attn_k_norm.weight",
                 "post_attention_norm.weight",
                 "ffn_gate.weight",
                 "ffn_up.weight",
                 "ffn_down.weight",
             })
        {
            addTensor(builder, prefix + suffix);
        }
    }

    /** @brief Populate every mandatory role of one routed/shared NextN predictor. */
    void addNextNMoEDepth(MockModelLoaderBuilder &builder, int source_layer)
    {
        const std::string prefix = "blk." + std::to_string(source_layer) + ".";
        for (const auto *suffix : {
                 "nextn.eh_proj.weight",
                 "nextn.hnorm.weight",
                 "nextn.enorm.weight",
                 "nextn.shared_head_norm.weight",
                 "attn_norm.weight",
                 "attn_q.weight",
                 "attn_k.weight",
                 "attn_v.weight",
                 "attn_output.weight",
                 "attn_q_norm.weight",
                 "attn_k_norm.weight",
                 "post_attention_norm.weight",
                 "ffn_gate_inp.weight",
                 "ffn_gate_exps.weight",
                 "ffn_up_exps.weight",
                 "ffn_down_exps.weight",
                 "ffn_gate_shexp.weight",
                 "ffn_up_shexp.weight",
                 "ffn_down_shexp.weight",
                 "ffn_gate_inp_shexp.weight",
             })
        {
            addTensor(builder, prefix + suffix);
        }
    }

    /** @brief Populate the dedicated `mtp.layers` source layout. */
    void addGenericMTPDepth(MockModelLoaderBuilder &builder)
    {
        for (const auto *name : {
                 "mtp.fc.weight",
                 "mtp.pre_fc_norm_hidden.weight",
                 "mtp.pre_fc_norm_embedding.weight",
                 "mtp.norm.weight",
                 "mtp.layers.0.input_layernorm.weight",
                 "mtp.layers.0.self_attn.q_proj.weight",
                 "mtp.layers.0.self_attn.k_proj.weight",
                 "mtp.layers.0.self_attn.v_proj.weight",
                 "mtp.layers.0.self_attn.o_proj.weight",
                 "mtp.layers.0.self_attn.q_norm.weight",
                 "mtp.layers.0.self_attn.k_norm.weight",
                 "mtp.layers.0.post_attention_layernorm.weight",
                 "mtp.layers.0.mlp.gate_proj.weight",
                 "mtp.layers.0.mlp.up_proj.weight",
                 "mtp.layers.0.mlp.down_proj.weight",
             })
        {
            addTensor(builder, name);
        }
    }

    /** @brief Encode a source count using either supported unsigned GGUF width. */
    template <typename Value>
    void setUnsignedMetadata(GGUFModel &model, const std::string &key, Value value)
    {
        static_assert(std::is_same_v<Value, uint32_t> || std::is_same_v<Value, uint64_t>);
        GGUFValue entry;
        entry.type = sizeof(Value) == 4 ? GGUFValueType::UINT32 : GGUFValueType::UINT64;
        entry.data.resize(sizeof(Value));
        std::memcpy(entry.data.data(), &value, sizeof(Value));
        model.metadata.emplace(key, std::move(entry));
    }

    /**
     * @brief Borrow a tiny manifest fixture as a parsed planning directory.
     *
     * Only names and geometry are copied; no cache or prepared tensor is
     * materialized. Routed parents have GGUF's [K,N,experts] rank so the memory
     * inventory receives a valid MoE source rather than a dense approximation.
     */
    GGUFModel planningDirectory(const MockModelLoader &loader)
    {
        GGUFModel model;
        model.architecture = loader.architecture();
        model.block_count = loader.blockCount();
        model.embedding_length = 4;
        model.head_count = 1;
        model.head_count_kv = 1;
        model.key_length = 4;
        model.context_length = 4096;
        model.vocab_size = 4;
        const auto names = loader.tensorNames();
        if (std::any_of(names.begin(), names.end(),
                [](const std::string &name) { return name.ends_with("_exps.weight"); }))
            setUnsignedMetadata(model, model.architecture + ".expert_count", uint32_t{4});
        for (const auto &name : names)
        {
            const bool routed = name.ends_with("_exps.weight");
            model.tensors.push_back(GGUFTensorInfo{
                .name = name,
                .dimensions = routed ? std::vector<uint64_t>{4, 4, 4}
                                     : std::vector<uint64_t>{4, 4},
                .type = GGUFTensorType::F32,
                .offset = 0,
                .size_bytes = (routed ? 64u : 16u) * sizeof(float),
                .split_idx = 0,
            });
        }
        return model;
    }

    /** @brief Own tensors before publishing borrowed immutable graph bindings. */
    struct FrozenFixture
    {
        std::vector<std::shared_ptr<TensorBase>> tensors;
        ModelWeightSetBuilder builder;

        /** @brief Create a device-free single-participant weight set. */
        FrozenFixture()
            : builder(makeStrategy())
        {
        }

        /** @return Explicit CPU ownership, with no backend initialization. */
        static InferenceStrategy makeStrategy()
        {
            InferenceStrategy strategy;
            strategy.mode = WeightInferenceMode::SingleDevice;
            strategy.model_id = ModelContextId{7};
            strategy.devices = {DeviceId::cpu()};
            return strategy;
        }

        /** @brief Retain storage before exporting its canonical source identity. */
        void add(const std::string &name)
        {
            auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{4, 4});
            WeightBinding binding;
            binding.identity = makeSourceWeightIdentity(
                name,
                ModelContextId{7},
                static_cast<uint64_t>(tensors.size() + 1));
            binding.tensor = tensor.get();
            tensors.push_back(std::move(tensor));
            builder.addBinding(std::move(binding));
        }

        /** @return Immutable bindings; this fixture continues to own their bytes. */
        FrozenModelWeightSet freeze()
        {
            return FrozenModelWeightSet(builder.strategy(), builder.freezeBindings());
        }
    };
} // namespace

/** @brief No MTP metadata or weights means no synthetic predictor or sidecar. */
TEST(Test__MTPWeightManifest, PlainDenseAndMoEModelsRejectRequiredLearnedPredictor)
{
    for (const std::string architecture : {"qwen35", "qwen35moe"})
    {
        SCOPED_TRACE(architecture);
        MockModelLoaderBuilder builder;
        builder.setArchitecture(architecture).setBlockCount(40);
        addTensor(builder, "blk.39.attn_norm.weight");
        auto loader = builder.build();
        const auto optional = discoverMTPWeightManifest(*loader, architecture, 40, false);
        EXPECT_FALSE(optional.available);
        EXPECT_EQ(optional.depth, 0);
        EXPECT_TRUE(optional.depths.empty());
        EXPECT_TRUE(optional.requiredNames().empty());
        EXPECT_EQ(mainLayerCountExcludingMTP(*loader, architecture, 40), 40);
        try
        {
            (void)requireMTPWeightManifest(*loader, architecture, 40);
            FAIL() << "A plain model was admitted with an invented learned predictor";
        }
        catch (const std::invalid_argument &error)
        {
            EXPECT_NE(std::string(error.what()).find("no MTP/nextn metadata or tensors"), std::string::npos);
            EXPECT_NE(std::string(error.what()).find("complete learned MTP/NextN weights"), std::string::npos);
        }
    }
}

/** @brief Declared or partial MTP weights are not a certificate of availability. */
TEST(Test__MTPWeightManifest, MetadataOnlyAndIncompletePredictorsFailRequiredAdmission)
{
    for (const bool partial : {false, true})
    {
        MockModelLoaderBuilder builder;
        builder.setArchitecture("qwen35moe").setBlockCount(41)
            .setInt("qwen35moe.nextn_predict_layers", 1);
        if (partial)
            addTensor(builder, "blk.40.nextn.eh_proj.weight");
        auto loader = builder.build();
        EXPECT_THROW((void)requireMTPWeightManifest(*loader, "qwen35moe", 41), std::invalid_argument);
    }
}

/** @brief Mandatory admission returns discovery's exact learned source inventory. */
TEST(Test__MTPWeightManifest, RequiredAdmissionPreservesAuthenticatedNextNAndDedicatedLayouts)
{
    for (const int layout : {0, 1, 2})
    {
        MockModelLoaderBuilder builder;
        const std::string architecture = layout == 1 ? "qwen35moe" : "qwen35";
        builder.setArchitecture(architecture).setBlockCount(layout == 2 ? 40 : 41);
        if (layout == 0) addNextNDepth(builder, 40);
        if (layout == 1) addNextNMoEDepth(builder, 40);
        if (layout == 2) addGenericMTPDepth(builder);
        auto loader = builder.build();
        const int blocks = layout == 2 ? 40 : 41;
        const auto manifest = requireMTPWeightManifest(*loader, architecture, blocks);
        const auto discovered = discoverMTPWeightManifest(*loader, architecture, blocks, true);
        EXPECT_TRUE(manifest.available);
        EXPECT_EQ(manifest.depth, 1);
        EXPECT_EQ(manifest.requiredNames(), discovered.requiredNames());
    }
}

TEST(Test__MTPWeightManifest, DiscoversNextNBlockLayoutFromQwen36Metadata)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(65)
        .setInt("qwen35.nextn_predict_layers", 1);
    addNextNDepth(builder, 64);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        65,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    ASSERT_EQ(manifest.depth, 1);
    ASSERT_EQ(manifest.depths.size(), 1u);
    EXPECT_TRUE(manifest.depths[0].nextn_block_layout);
    EXPECT_EQ(manifest.depths[0].source_layer_index, 64);
    EXPECT_EQ(manifest.depths[0].fc, "blk.64.nextn.eh_proj.weight");
    EXPECT_EQ(manifest.depths[0].wq, "blk.64.attn_q.weight");
}

TEST(Test__MTPWeightManifest, MainLayerCountExcludesTrailingNextNBlockForPlanning)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(65)
        .setInt("qwen35.nextn_predict_layers", 1);
    addNextNDepth(builder, 64);
    auto loader = builder.build();

    EXPECT_EQ(
        mainLayerCountExcludingMTP(*loader, "qwen35", 65),
        64);
}

TEST(Test__MTPWeightManifest, MainLayerCountKeepsRawBlocksWithoutNextNTensor)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(65)
        .setInt("qwen35.nextn_predict_layers", 1);
    addTensor(builder, "blk.64.attn_norm.weight");
    auto loader = builder.build();

    EXPECT_EQ(
        mainLayerCountExcludingMTP(*loader, "qwen35", 65),
        65);
}

TEST(Test__MTPWeightManifest, InfersNextNDepthWhenMetadataIsAbsent)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35").setBlockCount(64);
    addNextNDepth(builder, 64);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        64,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    EXPECT_EQ(manifest.depth, 1);
    EXPECT_EQ(manifest.depths[0].source_layer_index, 64);
}

/**
 * @brief Directory-inferred predictors enter the same main-layer and KV BOM.
 *
 * The count key is optional. A complete trailing predictor without that key
 * must not be counted as a main layer or lose its shifted cache contribution.
 * These are pure metadata/estimator calls: GPU identities do not initialize a
 * GPU, allocate a cache, or occupy an accelerator in this Unit regression.
 */
TEST(Test__MTPWeightManifest, MissingCountMetadataStillAdmitsExactlyOneShiftedCache)
{
    for (const std::string architecture : {"qwen35", "qwen35moe"})
    {
        SCOPED_TRACE(architecture);
        MockModelLoaderBuilder builder;
        builder.setArchitecture(architecture).setBlockCount(41);
        if (architecture == "qwen35moe") addNextNMoEDepth(builder, 40);
        else addNextNDepth(builder, 40);
        auto loader = builder.build();
        ASSERT_EQ(requireMTPWeightManifest(*loader, architecture, 41).depth, 1);
        EXPECT_EQ(mainLayerCountExcludingMTP(*loader, architecture, 41), 40);

        const auto model = planningDirectory(*loader);
        const auto profile = ModelMemoryProfile::fromGGUF(model);
        ASSERT_EQ(profile.mtp_layer_count, 1);
        const auto bytes = profile.serialize();
        const auto published = ModelMemoryProfile::deserialize(bytes.data(), bytes.size());
        ASSERT_EQ(published.mtp_layer_count, 1);
        for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
        {
            SCOPED_TRACE(device.toString());
            const auto estimate = PersistentStateMemoryEstimator::estimate(
                published, device, 1, 4096, 1, 1, 0, 1, 0, 39, "fp16",
                MTPStateRole::PredictorOwner);
            EXPECT_EQ(estimate.main_full_attention_layers, 40);
            EXPECT_EQ(estimate.mtp_full_attention_layers, 1);
            EXPECT_EQ(estimate.mtp_gdn_layers, 0);
            EXPECT_EQ(estimate.mtp_kv_cache_bytes,
                KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
                    1, 1, 4096, 1, 4, "fp16", device));
            EXPECT_GT(estimate.mtp_kv_cache_bytes, 0u);
            EXPECT_EQ(estimate.kv_cache_bytes,
                estimate.main_kv_cache_bytes + estimate.mtp_kv_cache_bytes);
        }
    }
}

/** @brief Loader discovery and GGUF accounting share every key and inferred layout. */
TEST(Test__MTPWeightManifest, LearnedBlockCountSharesAllMetadataKeysAndDirectoryLayouts)
{
    for (const std::string architecture : {"qwen35", "qwen35moe"})
    {
        for (const auto &key : std::vector<std::string>{
            architecture + ".nextn_predict_layers", architecture + ".mtp_num_hidden_layers",
            architecture + ".mtp.num_hidden_layers", "mtp.num_hidden_layers", "mtp_num_hidden_layers"})
        {
            for (const int count : {0, 1, 2})
            {
                SCOPED_TRACE(key + "=" + std::to_string(count));
                MockModelLoaderBuilder builder;
                builder.setArchitecture(architecture).setBlockCount(41).setInt(key, count);
                auto loader = builder.build();
                auto model32 = planningDirectory(*loader);
                auto model64 = model32;
                setUnsignedMetadata(model32, key, static_cast<uint32_t>(count));
                setUnsignedMetadata(model64, key, static_cast<uint64_t>(count));
                EXPECT_EQ(mtpLearnedBlockCount(*loader, architecture, 41), count);
                EXPECT_EQ(mtpLearnedBlockCount(model32), count);
                EXPECT_EQ(mtpLearnedBlockCount(model64), count);
                EXPECT_EQ(ModelMemoryProfile::fromGGUF(model32).mtp_layer_count, count);
                EXPECT_EQ(ModelMemoryProfile::fromGGUF(model64).mtp_layer_count, count);
                // A declared count is never enough to admit missing weights.
                EXPECT_THROW((void)requireMTPWeightManifest(*loader, architecture, 41), std::invalid_argument);
            }
        }
        enum class Layout { TrailingNextN, SeparateNextN, Dedicated };
        for (const auto layout : {Layout::TrailingNextN, Layout::SeparateNextN, Layout::Dedicated})
        {
            MockModelLoaderBuilder builder;
            const int blocks = layout == Layout::TrailingNextN ? 41 : 40;
            builder.setArchitecture(architecture).setBlockCount(blocks);
            if (layout == Layout::Dedicated) addGenericMTPDepth(builder);
            else if (architecture == "qwen35moe") addNextNMoEDepth(builder, 40);
            else addNextNDepth(builder, 40);
            auto loader = builder.build();
            const auto model = planningDirectory(*loader);
            EXPECT_EQ(mtpLearnedBlockCount(*loader, architecture, blocks), 1);
            EXPECT_EQ(mtpLearnedBlockCount(model), 1);
            EXPECT_EQ(ModelMemoryProfile::fromGGUF(model).mtp_layer_count, 1);
            EXPECT_EQ(mainLayerCountExcludingMTP(*loader, architecture, blocks), 40);
            EXPECT_EQ(requireMTPWeightManifest(*loader, architecture, blocks).depth, 1);
        }
    }
}

/** @brief Oversized/negative source counts fail before inventing runtime geometry. */
TEST(Test__MTPWeightManifest, LearnedBlockCountRejectsUnrepresentableMetadata)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35moe").setBlockCount(41);
    auto loader = builder.build();
    auto model = planningDirectory(*loader);
    const auto key = model.architecture + ".nextn_predict_layers";
    setUnsignedMetadata(model, key, uint64_t{1} + std::numeric_limits<int>::max());
    EXPECT_THROW((void)mtpLearnedBlockCount(model), std::invalid_argument);
    EXPECT_THROW((void)ModelMemoryProfile::fromGGUF(model), std::invalid_argument);
    loader->setIntParam(key, -1);
    EXPECT_THROW((void)mtpLearnedBlockCount(*loader, model.architecture, 41), std::invalid_argument);
    model.metadata.erase(key);
    model.block_count = uint64_t{1} + std::numeric_limits<int>::max();
    EXPECT_THROW((void)mtpLearnedBlockCount(model), std::invalid_argument);
    EXPECT_THROW((void)ModelMemoryProfile::fromGGUF(model), std::invalid_argument);
}

TEST(Test__MTPWeightManifest, DiscoversNextNMoEBlockLayoutFromQwen36Metadata)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35moe")
        .setBlockCount(41)
        .setInt("qwen35moe.nextn_predict_layers", 1);
    addNextNMoEDepth(builder, 40);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35moe",
        41,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    ASSERT_EQ(manifest.depth, 1);
    ASSERT_EQ(manifest.depths.size(), 1u);
    EXPECT_TRUE(manifest.depths[0].nextn_block_layout);
    EXPECT_TRUE(manifest.depths[0].moe_ffn_layout);
    EXPECT_EQ(manifest.depths[0].source_layer_index, 40);
    EXPECT_EQ(manifest.depths[0].moe_gate_exps, "blk.40.ffn_gate_exps.weight");
    EXPECT_EQ(manifest.depths[0].shared_expert_gate_inp, "blk.40.ffn_gate_inp_shexp.weight");
}

/**
 * @brief A participant replica and ExpertOverlay must never co-own experts.
 *
 * The compact predictor still needs its router and always-active shared expert
 * locally. Only the three routed parents move behind ExpertOverlay's prepared
 * residency registry.
 */
TEST(Test__MTPWeightManifest,
     ParticipantReplicaNamesRespectTypedRoutedExpertAuthority)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35moe")
        .setBlockCount(41)
        .setInt("qwen35moe.nextn_predict_layers", 1);
    addNextNMoEDepth(builder, 40);
    auto loader = builder.build();

    const MTPWeightManifest manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35moe",
        41,
        /*explicit_mtp=*/true);
    ASSERT_TRUE(manifest.available) << manifest.diagnostic;

    const auto sidecar_owned = manifest.participantReplicaNames(
        MTPRoutedExpertWeightAuthority::SidecarParticipant);
    const auto overlay_owned = manifest.participantReplicaNames(
        MTPRoutedExpertWeightAuthority::ExpertOverlay);
    const auto contains = [](const std::vector<std::string> &names,
                             const std::string &name)
    {
        return std::find(names.begin(), names.end(), name) != names.end();
    };

    for (const auto *routed_parent : {
             "blk.40.ffn_gate_exps.weight",
             "blk.40.ffn_up_exps.weight",
             "blk.40.ffn_down_exps.weight",
         })
    {
        EXPECT_TRUE(contains(sidecar_owned, routed_parent));
        EXPECT_FALSE(contains(overlay_owned, routed_parent));
    }

    for (const auto *participant_weight : {
             "blk.40.nextn.eh_proj.weight",
             "blk.40.attn_q.weight",
             "blk.40.ffn_gate_inp.weight",
             "blk.40.ffn_gate_shexp.weight",
             "blk.40.ffn_up_shexp.weight",
             "blk.40.ffn_down_shexp.weight",
             "blk.40.ffn_gate_inp_shexp.weight",
         })
    {
        EXPECT_TRUE(contains(sidecar_owned, participant_weight));
        EXPECT_TRUE(contains(overlay_owned, participant_weight));
    }
    EXPECT_EQ(sidecar_owned.size(), overlay_owned.size() + 3u);
}

TEST(Test__MTPWeightManifest, DiscoversGenericMTPLayersLayout)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(64)
        .setInt("mtp.num_hidden_layers", 1);
    addGenericMTPDepth(builder);
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        64,
        /*explicit_mtp=*/true);

    ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    ASSERT_EQ(manifest.depths.size(), 1u);
    EXPECT_FALSE(manifest.depths[0].nextn_block_layout);
    EXPECT_EQ(manifest.depths[0].fc, "mtp.fc.weight");
    EXPECT_EQ(manifest.depths[0].wq, "mtp.layers.0.self_attn.q_proj.weight");
}

TEST(Test__MTPWeightManifest, ExplicitMTPReportsMissingRequiredWeights)
{
    MockModelLoaderBuilder builder;
    builder.setArchitecture("qwen35")
        .setBlockCount(64)
        .setInt("qwen35.nextn_predict_layers", 1);
    addTensor(builder, "blk.64.nextn.eh_proj.weight");
    auto loader = builder.build();

    auto manifest = discoverMTPWeightManifest(
        *loader,
        "qwen35",
        64,
        /*explicit_mtp=*/true);

    EXPECT_FALSE(manifest.available);
    EXPECT_EQ(manifest.depth, 1);
    EXPECT_FALSE(manifest.missing_required.empty());
}

TEST(Test__MTPWeightManifest, ModelWeightBindingsExposeNextNWeights)
{
    FrozenFixture fixture;
    fixture.add("token_embd.weight");
    fixture.add("output_norm.weight");
    fixture.add("output.weight");
    for (const auto *suffix : {
             "nextn.eh_proj.weight",
             "nextn.hnorm.weight",
             "nextn.enorm.weight",
             "nextn.shared_head_norm.weight",
             "attn_norm.weight",
             "attn_q.weight",
             "attn_k.weight",
             "attn_v.weight",
             "attn_output.weight",
             "attn_q_norm.weight",
             "attn_k_norm.weight",
             "post_attention_norm.weight",
             "ffn_gate.weight",
             "ffn_up.weight",
             "ffn_down.weight",
         })
    {
        fixture.add(std::string("blk.64.") + suffix);
    }

    auto frozen = fixture.freeze();
    auto bindings = makeModelWeightBindings(frozen);
    auto legacy = toLegacyModelWeights(bindings);

    ASSERT_EQ(bindings.mtp.depth, 1);
    ASSERT_EQ(legacy.mtp.depth, 1);
    ASSERT_EQ(legacy.mtp.depths.size(), 1u);
    EXPECT_EQ(legacy.mtp.depths[0].source_layer_index, 64);
    EXPECT_TRUE(legacy.mtp.depths[0].nextn_block_layout);
    EXPECT_NE(legacy.mtp.depths[0].fc, nullptr);
    EXPECT_NE(legacy.mtp.depths[0].pre_fc_norm_hidden, nullptr);
    EXPECT_NE(legacy.mtp.depths[0].fa_block.wq, nullptr);
    EXPECT_NE(legacy.mtp.depths[0].fa_block.q_norm, nullptr);
}
