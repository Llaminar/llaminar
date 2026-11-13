/**
 * @file Test__Qwen4ExpPredictorProfile.cpp
 * @brief Device-free separate-source, typed attention and sparse-only BOM regressions.
 *
 * Exact directory fixtures never open or materialize tensor payloads. CPU and
 * GPU device IDs select metadata formulas only; no backend or GPU is created.
 */
#include "Qwen4ExpSidecarFixture.h"
#include "models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include "planning/ModelMemoryProfile.h"
#include "planning/PersistentStateMemoryEstimator.h"
#include "kernels/qwen4exp/Qwen4ExpCheckpointGeometry.h"
#include "execution/mtp/MTPWeightManifest.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstring>

namespace llaminar2::test
{
namespace
{
/** @brief Query persistent BOM through the production estimator with a bounded PP interval. */
PersistentStateEstimate estimate(const ModelMemoryProfile &profile, DeviceId device, MTPStateRole role,
    int first = 0, int last = 47)
{
    return PersistentStateMemoryEstimator::estimate(profile, device, 3, 512, 2, 2, 0, 24,
        first, last, "fp16", role);
}

/** @test Native main metadata and globals remain unchanged, and dense exports omit dormant preparation roles. */
TEST(Qwen4ExpPredictorProfile, SeparateDenseSourcePreservesMainAndProjectsExactConsumedNativeRoles)
{
    Qwen4ExpSidecarFixture f;
    const auto before = ModelMemoryProfile::fromGGUF(f.main);
    const auto profile = ModelMemoryProfile::fromGGUF(f.main, f.predictor, SharedPredictorMaterialization::Active);
    EXPECT_EQ(profile.n_layers, 48); EXPECT_EQ(profile.mainLayerCount(), 48); EXPECT_EQ(profile.totalLayerCount(), 49);
    ASSERT_TRUE(profile.flash_next_predictor);
    EXPECT_EQ(profile.flash_next_predictor->attention, Qwen4ExpPredictorAttention::Dense);
    EXPECT_EQ(profile.flash_next_predictor->hidden_norm, Qwen4ExpPredictorHiddenNorm::PerBranch);
    EXPECT_EQ(profile.flash_next_predictor->source, Qwen4ExpPredictorSource::SharedGGUF);
    EXPECT_EQ(profile.flash_next_predictor->source_layer, 48u);
    EXPECT_EQ(f.main.metadata.at("qwen4exp.block_count").asUInt64Checked(), 48u); EXPECT_EQ(f.main.tensors.size(), before.tensors.size());
    EXPECT_EQ(profile.embeddingBytes(), before.embeddingBytes()); EXPECT_EQ(profile.lmHeadBytes(), before.lmHeadBytes());
    EXPECT_EQ(profile.tensors.size(), before.tensors.size() + 28);
    size_t native = 0;
    for (const auto &t : f.predictor.tensors)
        if (t.name.find(".indexer.") == std::string::npos) native += t.size_bytes;
    EXPECT_EQ(profile.total_native_bytes, before.total_native_bytes + native);
    EXPECT_EQ(profile.weightBytesForLayers(48,48), native);
    EXPECT_EQ(std::count_if(profile.tensors.begin(), profile.tensors.end(), [](const auto &t) {
        return t.layer_index == 48 && t.name.find(".indexer.") != std::string::npos;
    }), 0);
    for (const auto &source : f.predictor.tensors)
        if (source.name.find(".indexer.") == std::string::npos)
        {
            const auto it = std::find_if(profile.tensors.begin(), profile.tensors.end(), [&](const auto &t) { return t.name == source.name; });
            ASSERT_NE(it, profile.tensors.end()); EXPECT_EQ(it->native_bytes, source.size_bytes);
            if (source.type == GGUFTensorType::Q8_0) EXPECT_EQ(it->quant_type, "Q8_0");
        }
}

/** @test Exact terminal HC roles replace generic shared-head norm without retargeting main globals. */
TEST(Qwen4ExpPredictorProfile, ManifestUsesDenseHCHeadAndNoDormantIndexerRoles)
{
    Qwen4ExpSidecarFixture f;
    const auto p=ModelMemoryProfile::fromGGUF(f.main,f.predictor,SharedPredictorMaterialization::Active);
    const auto manifest=discoverMTPWeightManifest(p,true); ASSERT_TRUE(manifest.available) << manifest.diagnostic;
    ASSERT_TRUE(manifest.flash_next); EXPECT_EQ(manifest.flash_next->attention,Qwen4ExpPredictorAttention::Dense);
    ASSERT_EQ(manifest.depths.size(),1u);
    EXPECT_EQ(manifest.depths.front().final_norm,"blk.48.nextn.hc_head_norm.weight");
    const auto required=manifest.requiredNames(); EXPECT_EQ(required.size(),28u);
    for(const auto &name:required)
    {
        EXPECT_EQ(name.find(".indexer."),std::string::npos);
        EXPECT_EQ(name.find("shared_head_norm"),std::string::npos);
    }
    EXPECT_NE(std::find(required.begin(),required.end(),"blk.48.nextn.hc_head_down.weight"),required.end());
    EXPECT_NE(std::find(required.begin(),required.end(),"blk.48.nextn.hc_head_up.weight"),required.end());
    const Qwen4ExpSidecarManifest source(f.main,f.predictor);
    EXPECT_EQ(source.tensor("token_embd.weight").source,Qwen4ExpTensorSource::Main);
    EXPECT_EQ(source.tensor("output.weight").source,Qwen4ExpTensorSource::Main);
    const auto inactive=ModelMemoryProfile::fromGGUF(f.main,f.predictor,SharedPredictorMaterialization::Inactive);
    EXPECT_FALSE(discoverMTPWeightManifest(inactive,true).available);
}

/** @test Inactive selection authenticates the original source but admits no predictor weight/cache consumers. */
TEST(Qwen4ExpPredictorProfile, AdmittedSourceWithMTPDisabledHasNoPredictorMaterializationOrState)
{
    Qwen4ExpSidecarFixture f;
    const auto main = ModelMemoryProfile::fromGGUF(f.main);
    const auto inactive = ModelMemoryProfile::fromGGUF(f.main, f.predictor, SharedPredictorMaterialization::Inactive);
    EXPECT_EQ(inactive.total_native_bytes, main.total_native_bytes); EXPECT_EQ(inactive.tensors.size(), main.tensors.size());
    EXPECT_EQ(inactive.weightBytesForLayers(48,48), 0u); ASSERT_TRUE(inactive.flash_next_predictor);
    for (auto device : {DeviceId{DeviceType::CPU,1}, DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        const auto value = estimate(inactive, device, MTPStateRole::Disabled);
        const auto control = estimate(main, device, MTPStateRole::Disabled);
        EXPECT_EQ(value.kv_cache_bytes, control.kv_cache_bytes); EXPECT_EQ(value.checkpoint_state_bytes, 0u);
        EXPECT_EQ(value.mtp_full_attention_layers, 0); EXPECT_EQ(value.mtp_gdn_layers, 0);
        EXPECT_EQ(value.mtp_sparse_attention_layers, 0);
    }
}

/** @test Dense attention owns shifted KV but never GDN or sparse auxiliary bytes. */
TEST(Qwen4ExpPredictorProfile, DenseAndSparsePredictorsHaveDistinctTypedAuxiliaryOwnership)
{
    Qwen4ExpSidecarFixture f;
    auto dense = ModelMemoryProfile::fromGGUF(f.main, f.predictor, SharedPredictorMaterialization::Active);
    auto ratios = f.predictor.metadata.at("qwen4exp.attention.compress_ratios").asUInt64Array(); ratios.back() = 4;
    f.predictor.metadata["qwen4exp.attention.compress_ratios"] = qwen4Array(GGUFValueType::UINT64, ratios);
    auto sparse = ModelMemoryProfile::fromGGUF(f.main, f.predictor, SharedPredictorMaterialization::Active);
    EXPECT_EQ(dense.flashNextLayerKind(0), Qwen4ExpLayerKind::GDN);
    EXPECT_EQ(dense.flashNextLayerKind(3), Qwen4ExpLayerKind::QSA);
    EXPECT_EQ(dense.flashNextLayerKind(48), Qwen4ExpLayerKind::Dense);
    EXPECT_EQ(sparse.flashNextLayerKind(48), Qwen4ExpLayerKind::QSA);
    EXPECT_EQ(dense.flashNextSparseLayerCount(0,48), 12u); EXPECT_EQ(sparse.flashNextSparseLayerCount(0,48), 13u);
    EXPECT_EQ(dense.flashNextSparseLayerCount(48,48), 0u); EXPECT_EQ(sparse.flashNextSparseLayerCount(48,48), 1u);
    for (auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        const auto d = estimate(dense, device, MTPStateRole::PredictorOwner);
        const auto s = estimate(sparse, device, MTPStateRole::PredictorOwner);
        EXPECT_EQ(d.main_full_attention_layers,12); EXPECT_EQ(d.main_gdn_layers,36);
        EXPECT_EQ(d.mtp_full_attention_layers,1); EXPECT_EQ(d.mtp_gdn_layers,0);
        EXPECT_EQ(d.main_sparse_attention_layers,12); EXPECT_EQ(d.mtp_sparse_attention_layers,0);
        EXPECT_EQ(s.mtp_sparse_attention_layers,1); EXPECT_EQ(d.mtp_kv_cache_bytes,s.mtp_kv_cache_bytes);
        EXPECT_GT(d.mtp_kv_cache_bytes,0u); EXPECT_EQ(d.live_recurrent_state_bytes,s.live_recurrent_state_bytes);
        const auto auxiliary = qwen4exp::Qwen4ExpCheckpointGeometry::capacity(1,0);
        if (device.is_gpu()) EXPECT_EQ(s.sequence_metadata_bytes-d.sequence_metadata_bytes,kMTPConcurrentLiveCheckpointSets*auxiliary);
        else EXPECT_EQ(s.checkpoint_state_bytes-d.checkpoint_state_bytes,kMTPConcurrentLiveCheckpointSets*auxiliary);
    }
}

/** @test PP main slices count QSA/PLE only where owned; predictor remains a separate exact cache. */
TEST(Qwen4ExpPredictorProfile, PPSlicesDoNotProjectDensePredictorIntoMainSparseOrPLEState)
{
    Qwen4ExpSidecarFixture f; const auto p = ModelMemoryProfile::fromGGUF(f.main,f.predictor,SharedPredictorMaterialization::Active);
    for (auto device : {DeviceId::cpu(),DeviceId::cuda(0),DeviceId::rocm(0)})
    {
        const auto onlyQSA = estimate(p,device,MTPStateRole::MainModelFollower,3,3);
        EXPECT_EQ(onlyQSA.main_gdn_layers,0); EXPECT_EQ(onlyQSA.main_sparse_attention_layers,1);
        EXPECT_EQ(onlyQSA.mtp_full_attention_layers,0); EXPECT_EQ(onlyQSA.mtp_sparse_attention_layers,0);
        const auto terminal = estimate(p,device,MTPStateRole::PredictorOwner,47,47);
        EXPECT_EQ(terminal.main_sparse_attention_layers,1); EXPECT_EQ(terminal.mtp_full_attention_layers,1);
        EXPECT_EQ(terminal.mtp_sparse_attention_layers,0);
    }
}

/** @test Typed source policy survives the explicit incompatible wire version without invented defaults. */
TEST(Qwen4ExpPredictorProfile, WireRoundTripAuthenticatesPolicyAndRejectsUnknownOrMissingState)
{
    Qwen4ExpSidecarFixture f; auto p = ModelMemoryProfile::fromGGUF(f.main,f.predictor,SharedPredictorMaterialization::Active);
    const auto wire=p.serialize(); const auto restored=ModelMemoryProfile::deserialize(wire.data(),wire.size());
    EXPECT_EQ(restored.flash_next_predictor,p.flash_next_predictor); EXPECT_EQ(restored.flash_next,p.flash_next);
    EXPECT_EQ(restored.total_native_bytes,p.total_native_bytes); EXPECT_EQ(restored.tensors.size(),p.tensors.size());
    EXPECT_EQ(restored.flashNextLayerKind(48),Qwen4ExpLayerKind::Dense);
    auto stale=wire; const uint32_t oldVersion=3; std::memcpy(stale.data()+4,&oldVersion,sizeof(oldVersion));
    EXPECT_THROW(ModelMemoryProfile::deserialize(stale.data(),stale.size()),std::runtime_error);
    EXPECT_THROW(ModelMemoryProfile::deserialize(wire.data(),wire.size()-1),std::runtime_error);
    auto trailing=wire; trailing.push_back(0);
    EXPECT_THROW(ModelMemoryProfile::deserialize(trailing.data(),trailing.size()),std::runtime_error);
    p.flash_next_predictor->attention=static_cast<Qwen4ExpPredictorAttention>(99);
    EXPECT_THROW(p.serialize(),std::runtime_error);
    p.flash_next_predictor.reset();
    EXPECT_THROW(p.serialize(),std::runtime_error);
}

/** @test Exact pairing and kind queries reject malformed input rather than silently adopting main policy. */
TEST(Qwen4ExpPredictorProfile, RejectsUnpairedSourcesAndUndefinedLayerIdentity)
{
    Qwen4ExpSidecarFixture f;
    auto bad=f.predictor; bad.metadata["qwen4exp.embedding_length"]=qwen4Scalar(GGUFValueType::UINT32,uint32_t(10240));
    EXPECT_THROW(ModelMemoryProfile::fromGGUF(f.main,bad,SharedPredictorMaterialization::Active),std::invalid_argument);
    EXPECT_THROW(ModelMemoryProfile::fromGGUF(f.main,f.predictor,static_cast<SharedPredictorMaterialization>(99)),std::invalid_argument);
    auto p=ModelMemoryProfile::fromGGUF(f.main,f.predictor,SharedPredictorMaterialization::Active);
    EXPECT_THROW(p.flashNextLayerKind(-1),std::invalid_argument);
    EXPECT_THROW(p.flashNextLayerKind(49),std::invalid_argument);
    p.flash_next_predictor.reset();
    EXPECT_THROW(p.flashNextLayerKind(48),std::invalid_argument);
    const auto main=ModelMemoryProfile::fromGGUF(f.main); EXPECT_EQ(main.totalLayerCount(),48); EXPECT_FALSE(main.flash_next_predictor);
    EXPECT_THROW(main.flashNextLayerKind(48),std::invalid_argument);
}
} // namespace
} // namespace llaminar2::test
