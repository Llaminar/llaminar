/**
 * @file Test__QwenSparseSnapshotState.cpp
 * @brief Device-free native descriptor, source selection, planner and PMA state-capacity regression.
 *
 * A live two-row oracle pool cannot price a native 1024-row captured slot.
 * Geometry-only tests link no engine. Full tests construct actual CPU stage
 * descriptors and price GPU metadata without opening a device or model.
 */
#include "planning/QwenSparseSnapshotMemory.h"
#include <gtest/gtest.h>
#include <limits>
#include <array>
#ifndef QFN_GEOMETRY_ONLY
#include "planning/MemoryPlanner.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "snapshots/SnapshotCapture.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "snapshots/SnapshotSemanticObservation.h"
#include "tensors/Tensors.h"
#include "../models/qwen4exp/Qwen4ExpMetadataFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#endif
using namespace llaminar2;
namespace
{
/** @return Fixed native metadata for a synthetic main-only source manifest. */
Qwen4ExpMemoryGeometry sparseModel()
{
    Qwen4ExpMemoryGeometry model;
    model.indexer_head_dimension=128;model.indexer_query_heads=4;
    model.compression_ratio=4;model.maximum_selected_tokens=2051;
    for(unsigned layer=3;layer<48;layer+=4)model.sparse_decoder_layers.push_back(layer);
    return model;
}
/** @return One actual indexer observation per semantic role and main layer. */
std::vector<GraphSnapshotSourceOutput> selection()
{
    std::vector<GraphSnapshotSourceOutput> result;
    for(int layer=3;layer<48;layer+=4)
        for(auto role:{GraphSnapshotSourceRole::QwenSparseQuery,GraphSnapshotSourceRole::QwenSparsePooledKeys,
                      GraphSnapshotSourceRole::QwenSparseSelectedIndices})
            result.push_back(GraphSnapshotSourceOutput::qwenSparse(layer,role));
    return result;
}
}
/** @test Only canonical main-layer producers can select native state replacement. */
TEST(QwenSparseSnapshotState, SourceSelectionIsExactAndRejectsAliasesAndPredictorNamespaces)
{
    for(const auto &source:selection()) {
        ASSERT_TRUE(GraphSnapshotSourceOutput::fromQwenSparseSemanticKey(source.semanticKey()));
        EXPECT_EQ(*GraphSnapshotSourceOutput::fromQwenSparseSemanticKey(source.semanticKey()),source);
    }
    for(const auto *key:{"MTP0_QSA_INDEXER_POOLED_KEYS","layer03_QSA_INDEXER_POOLED_KEYS",
        "layer-1_QSA_INDEXER_POOLED_KEYS","layer-0_QSA_INDEXER_POOLED_KEYS","layer2147483648_QSA_INDEXER_POOLED_KEYS",
        "layer3_QSA_INDEXER_POOLED_KEYS_ALLREDUCED","layer3_QSA_INDEXER_POOLED_KEYS_extra"})
        EXPECT_FALSE(GraphSnapshotSourceOutput::fromQwenSparseSemanticKey(key))<<key;
    EXPECT_THROW((void)GraphSnapshotSourceOutput::qwenSparse(-1,GraphSnapshotSourceRole::QwenSparseQuery),std::invalid_argument);
    EXPECT_THROW((void)GraphSnapshotSourceOutput::qwenSparse(3,static_cast<GraphSnapshotSourceRole>(99)),std::invalid_argument);
}
/** @test Native pools use context capacity, not prefill/decode rows, and respect exact PP source ownership. */
TEST(QwenSparseSnapshotState, NativeStateCapacityUsesContextAndEachRetainedOwnerOnce)
{
    const auto model=sparseModel();const auto selected=selection();
    for(const auto [first,last,sparse]:{std::array{0,47,12},std::array{0,7,2},std::array{8,23,4},std::array{24,47,6}})
        for(const int rows:{1,9}) {
            const auto bytes=qwenSparseSnapshotMemory(model,48,first,last,4096,1,rows,2,selected);
            EXPECT_EQ(bytes.native_state_payload_bytes,std::size_t(sparse)*1024*128*4*2);
            EXPECT_GT(bytes.authored_metadata_bytes,0u);
        }
    const auto single=qwenSparseSnapshotMemory(model,48,0,47,4096,1,9,2,selected);
    const auto batched=qwenSparseSnapshotMemory(model,48,0,47,4096,3,9,2,selected);
    EXPECT_EQ(batched.native_state_payload_bytes,3*single.native_state_payload_bytes);
    EXPECT_EQ(batched.authored_metadata_bytes,3*single.authored_metadata_bytes);
    EXPECT_EQ(single.native_state_payload_bytes,12'582'912u);
}
/** @test Metadata uses the canonical copy/layout authorities, independently of payload slack. */
TEST(QwenSparseSnapshotState, AuthoredMetadataCoversOddLegacyThenActualSelectedSlots)
{
    const auto model=sparseModel();const auto selected=selection();
    for(const int rows:{1,9}) {
        std::size_t end=4,payload=4;
        for(const auto &source:selected) {
            const auto role=source.role()==GraphSnapshotSourceRole::QwenSparseQuery?Qwen4ExpSnapshotOutput::PreparedQuery:
                source.replacesReferencePayload()?Qwen4ExpSnapshotOutput::PooledKeys:Qwen4ExpSnapshotOutput::SelectedIndices;
            const auto layout=snapshotStorageLayout(qwen4ExpSparseSnapshotShape(role,rows,4096,2051).storageShape());
            const auto slot=placeSnapshotStorageSlot(end,layout);end=slot.end_bytes;payload+=layout.payload_storage_bytes;
            EXPECT_EQ((slot.offset_bytes+layout.metadata_offset_bytes)%alignof(SnapshotObservedHeader),0u);
        }
        const auto bytes=qwenSparseSnapshotMemory(model,48,0,47,4096,1,rows,1,selected);
        EXPECT_LE(end-payload,bytes.authored_metadata_bytes);
    }
}
/** @test Missing source geometry, duplicate producers and overflow fail before reservation. */
TEST(QwenSparseSnapshotState, RejectsWrongSourceDuplicateAndOverflowWithoutWeakeningEmptySelection)
{
    const auto model=sparseModel();auto selected=selection();
    EXPECT_EQ(qwenSparseSnapshotMemory({},0,0,0,0,0,0,0,{}).totalBytes(),0u);
    EXPECT_THROW((void)qwenSparseSnapshotMemory(model,48,0,47,4096,1,9,0,selected),std::invalid_argument);
    selected.push_back(selected.front());
    EXPECT_THROW((void)qwenSparseSnapshotMemory(model,48,0,47,4096,1,9,2,selected),std::invalid_argument);
    selected={GraphSnapshotSourceOutput::qwenSparse(2,GraphSnapshotSourceRole::QwenSparsePooledKeys)};
    EXPECT_THROW((void)qwenSparseSnapshotMemory(model,48,0,47,4096,1,9,2,selected),std::invalid_argument);
    selected={GraphSnapshotSourceOutput::qwenSparse(48,GraphSnapshotSourceRole::QwenSparsePooledKeys)};
    EXPECT_THROW((void)qwenSparseSnapshotMemory(model,48,0,47,4096,1,9,2,selected),std::invalid_argument);
    selected=selection();
    EXPECT_THROW((void)qwenSparseSnapshotMemory(model,48,0,47,4096,2,9,std::numeric_limits<std::size_t>::max(),selected),std::overflow_error);
    EXPECT_THROW((void)qwenSparseSnapshotMemory(model,48,0,47,4096,1,9,std::numeric_limits<std::size_t>::max(),selected),std::overflow_error);
    auto wrong=model;++wrong.maximum_selected_tokens;
    EXPECT_THROW((void)qwenSparseSnapshotMemory(wrong,48,0,47,4096,1,9,2,selected),std::invalid_argument);
    wrong=model;wrong.compression_ratio=0;
    EXPECT_THROW((void)qwenSparseSnapshotMemory(wrong,48,0,47,4096,1,9,2,selected),std::invalid_argument);
}
/** @test Actual serving reuse roles retain three main owners, independent of bucket and NPY counts. */
TEST(ServingSnapshotArenaInventory, OrdinaryRolesKeepHistoryAndResidentInputsDistinct)
{
    const auto main = ServingSnapshotArenaInventory::ordinary(true, false, 0, 0);
    EXPECT_EQ(main.wideMainOwners(), 1u);
    EXPECT_EQ(main.scalarMainOwners(), 2u);
    EXPECT_EQ(main.mainOwners(), 3u);
    EXPECT_EQ(main.allOwners(), 3u);
    const MainSnapshotPhasePayload payload{.wide_main_bytes=900,.scalar_main_bytes=100};
    EXPECT_EQ(payload.bytesFor(main), 1100u);
    const auto with_predictor = ServingSnapshotArenaInventory::ordinary(true, true, 6, 1);
    EXPECT_EQ(with_predictor.mainOwners(), 4u);
    EXPECT_EQ(with_predictor.predictorOwners(), 6u);
    EXPECT_EQ(with_predictor.allOwners(), 10u);
    EXPECT_EQ(payload.bytesFor(with_predictor), 1200u)
        << "Predictor slots cannot multiply main reference payload or main QSA state";
    const auto requests=ServingSnapshotArenaInventory::ordinary(true,true,6,3);
    EXPECT_EQ(requests.mainOwners(),4u);
    EXPECT_EQ(payload.bytesFor(requests),1400u) << "Condition request rows are included exactly once";
    EXPECT_THROW((void)ServingSnapshotArenaInventory::ordinary(true,true,6,0),std::invalid_argument);
    EXPECT_THROW((void)ServingSnapshotArenaInventory::ordinary(true,false,1,0),std::invalid_argument);
    EXPECT_THROW((void)ServingSnapshotArenaInventory::ordinary(true,true,
        std::numeric_limits<std::size_t>::max(),1),std::overflow_error);
    EXPECT_THROW((void)(MainSnapshotPhasePayload{.wide_main_bytes=std::numeric_limits<std::size_t>::max(),
        .scalar_main_bytes=1}.bytesFor(main)),std::overflow_error);
}
#ifndef QFN_GEOMETRY_ONLY
namespace
{
/** @return Three real selected output layouts from an ordinary CPU indexer stage, without execution. */
std::vector<SnapshotStorageLayout> nativeIndexerLayouts(int rows, int blocks=1024)
{
    using S=QwenSparseAttentionStage;using F=Qwen4ExpFloatBinding;using I=Qwen4ExpIntBinding;
    FP32Tensor query({std::size_t(rows),512}),raw({std::size_t(rows),128}),partial({4,128}),
        pools({std::size_t(blocks),128}),prepared({std::size_t(rows),512}),scores({std::size_t(rows),std::size_t(blocks)}),
        gain({128}),cosine({4096,32}),sine({4096,32});
    INT32Tensor replay({3}),indices({std::size_t(rows),2051}),counts({std::size_t(rows)});
    S::Params params{.device_id=DeviceId::cpu(),.rowCapacity=rows,.tokenCapacity=4096,.replay=I(replay,BufferId::QFN_QSA_REPLAY),
        .operation=S::Indexer{F(query,BufferId::QFN_QSA_QUERY),F(raw,BufferId::QFN_QSA_RAW_KEY),
          F(partial,BufferId::QFN_QSA_PARTIAL_KEYS),F(pools,BufferId::QFN_QSA_POOLED_KEYS),
          Qwen4ExpImmutableWeightBinding(gain),Qwen4ExpImmutableWeightBinding(gain),
          Qwen4ExpImmutableWeightBinding(cosine),Qwen4ExpImmutableWeightBinding(sine),
          F(prepared,BufferId::QFN_QSA_PREPARED_QUERY),F(scores,BufferId::QFN_QSA_SCORES),
          I(indices,BufferId::QFN_QSA_SELECTED),I(counts,BufferId::QFN_QSA_SELECTED_COUNT),blocks,4096}};
    if (blocks!=1024) {
        // Assert admission failure before any dump/manifest API is called.
        EXPECT_THROW((void)S(std::move(params)),std::invalid_argument);
        return {};
    }
    S stage(std::move(params));
    std::vector<SnapshotStorageLayout> result;
    const auto dump=stage.getDumpInfoSnapshot();
    for(const auto &out:dump.outputs) {
        StageDumpInfo singleton;singleton.outputs.push_back(out);
        const auto keys=SnapshotCapture::possibleKeysForStage("layer3_qsa_indexer",singleton);
        if(keys.empty())continue;
        if(std::string_view(out.name)=="pooled_keys") {
            EXPECT_EQ(out.rows,1024u);EXPECT_EQ(out.cols,128u);EXPECT_EQ(out.byte_size,524288u);
        }
        result.push_back(snapshotStorageLayout({out.byte_size,snapshotCopyGeometry(out)}));
    }
    EXPECT_EQ(result.size(),3u);
    return result;
}
}
/** @test A larger containing tensor cannot silently change the admitted per-layer state-bank capacity. */
TEST(QwenSparseSnapshotState, RejectsOversizedNativePoolAtStageAdmission)
{
    EXPECT_TRUE(nativeIndexerLayouts(1,1025).empty());
}
/** @test Actual source dump descriptors fit the same planner/PMA bound on CUDA and ROCm without devices. */
TEST(QwenSparseSnapshotState, ActualDescriptorsAndPlannerShareStateCapacityAndOnePMAOwner)
{
    const auto profile=ModelMemoryProfile::fromGGUF(test::qwen4MetadataFixture(false));
    const auto selected=selection();
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)})
    for(const auto [first,last,layers]:{std::array{0,47,12},std::array{0,7,2},std::array{8,23,4},std::array{24,47,6}}) {
        SCOPED_TRACE(::testing::Message()<<"layers="<<first<<':'<<last);
        DevicePlanConfig cfg;cfg.world_rank=0;cfg.device=device;cfg.device_compute_units=60;
        cfg.device_total_bytes=cfg.device_free_bytes=128ull<<30;cfg.max_seq_len=4096;cfg.activation_seq_len=9;
        cfg.first_layer=first;cfg.last_layer=last;cfg.batch_size=1;
        // This is the unresolved higher-priority snapshot plan. Native table
        // placement is selected later; its host/device storage and channel BOM
        // must not leak into the frozen snapshot owner contribution.
        cfg.associated_host_memory = PhysicalMemoryResource{
            .world_rank = 0, .device = DeviceId::cpu(),
            .total_bytes = 128ull << 30, .admission_available_bytes = 128ull << 30};
        // Derive the ordinary token component from actual output descriptors,
        // keeping native state isolated from unrelated reference payload slack.
        std::size_t exact=0,ordinary=0;
        std::vector<std::size_t> arena_bytes;
        std::size_t wide_ordinary=0, scalar_ordinary=0;
        for(const int rows:{9,1,1}) {
            std::size_t end=4,phase_ordinary=4;ordinary+=4;
            const auto layouts=nativeIndexerLayouts(rows);
            for(int layer=0;layer<layers;++layer)for(const auto &layout:layouts) {
                end=placeSnapshotStorageSlot(end,layout).end_bytes;
                if(layout.payload_storage_bytes!=524288u) {
                    ordinary+=layout.payload_storage_bytes;
                    phase_ordinary+=layout.payload_storage_bytes;
                }
            }
            if(rows==9)wide_ordinary=phase_ordinary;else scalar_ordinary=phase_ordinary;
            exact+=end;arena_bytes.push_back(end);
        }
        cfg.graph_snapshot_memory.per_accelerator_bytes=ordinary;
        cfg.graph_snapshot_memory.authored_source_outputs=selected;
        const auto owners=ServingSnapshotArenaInventory::ordinary(true,false,0,0);
        cfg.graph_snapshot_memory.serving_arenas=owners;
        cfg.graph_snapshot_memory.main_phase_payload=MainSnapshotPhasePayload{
            .wide_main_bytes=wide_ordinary,.scalar_main_bytes=scalar_ordinary};
        cfg.graph_snapshot_memory.retained_forward_arena_count=owners.allOwners();
        cfg.graph_snapshot_memory.retained_main_forward_arena_count=owners.mainOwners();
        const auto contribution=qwenSparseSnapshotMemory(*profile.flash_next,48,first,last,4096,1,9,3,selected);
        const auto plan=MemoryPlanner::plan(profile,{cfg});
        ASSERT_FALSE(cfg.ngram_table_placement);
        ASSERT_EQ(plan.devices.size(),1u);
        const auto gpu_plan=std::find_if(plan.devices.begin(),plan.devices.end(),
            [device](const auto &candidate){return candidate.device()==device;});
        ASSERT_NE(gpu_plan,plan.devices.end());
        const auto capacity=ordinary+contribution.totalBytes();
        EXPECT_EQ(gpu_plan->graph_snapshot_bytes(),capacity);
        const auto *resource=plan.physicalPlan().find({0,device});ASSERT_NE(resource,nullptr);
        EXPECT_EQ(resource->bytes(PhysicalMemoryOwner::GraphSnapshotArena),capacity);
        const auto *host=plan.physicalPlan().find({0,DeviceId::cpu()});
        EXPECT_EQ(host,nullptr);
        EXPECT_EQ(resource->bytes(PhysicalMemoryOwner::ImmutableTableStorage),0u);
        EXPECT_EQ(resource->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),0u);
        ASSERT_GE(capacity,exact);
        // Use the real planner certificate, retaining the first arena lease
        // while admitting the second, exactly as setup does for prefill/decode.
        {
            auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.admit());
            PhysicalMemoryAuthority authority(certificate,0);
            auto pool=authority.reserveNewAllocations(device,PhysicalMemoryOwner::GraphSnapshotArena,capacity);
            auto prefill=pool.claimAllocation(arena_bytes[0]);
            auto decode=pool.claimAllocation(arena_bytes[1]);
            auto resident=pool.claimAllocation(arena_bytes[2]);
            EXPECT_TRUE(prefill.valid());EXPECT_TRUE(decode.valid());EXPECT_TRUE(resident.valid());
            EXPECT_EQ(pool.materializedBytes(),exact);
        }
        // The original live-reference pricing cannot admit the actual source
        // descriptor even with all four phase files' live pool rows included.
        auto legacy=cfg;legacy.graph_snapshot_memory.authored_source_outputs.clear();
        legacy.graph_snapshot_memory.main_phase_payload.reset();
        legacy.graph_snapshot_memory.per_accelerator_bytes=ordinary+std::size_t(layers)*4608;
        const auto legacy_plan=MemoryPlanner::plan(profile,{legacy});
        const auto *legacy_resource=legacy_plan.physicalPlan().find({0,device});ASSERT_NE(legacy_resource,nullptr);
        const auto legacy_bytes=legacy_resource->bytes(PhysicalMemoryOwner::GraphSnapshotArena);
        ASSERT_LT(legacy_bytes,exact);
        {
            auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(legacy_plan.admit());
            PhysicalMemoryAuthority authority(certificate,0);
            auto pool=authority.reserveNewAllocations(device,PhysicalMemoryOwner::GraphSnapshotArena,legacy_bytes);
            EXPECT_THROW((void)pool.claimAllocation(exact),std::logic_error);
        }
        // Exact-minus-one fails in the same PMA owner; no alternate accounting
        // authority or untyped margin is introduced to rescue the short claim.
        PhysicalMemoryPlanBuilder builder;
        builder.add({.world_rank=0,.device=device,.total_bytes=capacity*2,.admission_available_bytes=capacity*2},
            PhysicalMemoryOwner::GraphSnapshotArena,exact-1);
        auto certificate=std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(builder.build());
        PhysicalMemoryAuthority authority(certificate,0);
        auto short_pool=authority.reserveNewAllocations(device,PhysicalMemoryOwner::GraphSnapshotArena,exact-1);
        EXPECT_THROW((void)short_pool.claimAllocation(exact),std::logic_error);
        cfg.graph_snapshot_memory.authored_source_outputs.clear();
        const auto unselected=MemoryPlanner::plan(profile,{cfg});
        const auto *unselected_resource=unselected.physicalPlan().find({0,device});ASSERT_NE(unselected_resource,nullptr);
        EXPECT_EQ(unselected_resource->bytes(PhysicalMemoryOwner::GraphSnapshotArena),ordinary);
        cfg.device=DeviceId::cpu();cfg.cpu_execution=test::kSyntheticCPUExecutionGeometry;
        EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().graph_snapshot_bytes(),0u);
    }
}
/** @test Existing ordinary metadata keeps its declared bytes; unpaired source selections fail explicitly. */
TEST(QwenSparseSnapshotState, OrdinaryProfileAndEmptySelectionKeepExistingAdmission)
{
    ModelMemoryProfile profile;profile.architecture="qwen2";profile.n_layers=1;
    profile.d_model=64;profile.d_ff=128;profile.n_heads=profile.n_kv_heads=1;
    profile.head_dim=64;profile.vocab_size=256;profile.max_seq_len=4096;
    DevicePlanConfig cfg;cfg.world_rank=0;cfg.device=DeviceId::cuda(0);cfg.device_compute_units=60;
    cfg.device_total_bytes=cfg.device_free_bytes=128ull<<30;cfg.max_seq_len=32;cfg.activation_seq_len=9;
    cfg.graph_snapshot_memory.per_accelerator_bytes=1024;
    EXPECT_EQ(MemoryPlanner::plan(profile,{cfg}).devices.front().graph_snapshot_bytes(),1024u);
    cfg.graph_snapshot_memory.authored_source_outputs=selection();
    cfg.graph_snapshot_memory.retained_forward_arena_count=2;
    cfg.graph_snapshot_memory.retained_main_forward_arena_count=2;
    EXPECT_THROW((void)MemoryPlanner::plan(profile,{cfg}),std::invalid_argument);
}
/** @test The actual local expert descriptor is not the later completed canonical route producer. */
TEST(QwenSparseSnapshotState, ActualLocalMoEDescriptorCannotSelectCompletedRouteSnapshots)
{
    FP32Tensor routes({10,2560});
    MoEExpertComputeStage::Params params;
    params.device_id=DeviceId::cpu();params.seq_len=1;params.top_k=10;params.d_model=2560;
    params.canonical_route_contributions=&routes;
    // A legitimate complete native producer keeps its original semantic key.
    MoEExpertComputeStage complete(params);
    EXPECT_EQ(SnapshotCapture::possibleKeysForStage("layer7_moe_expert_ffn",complete.getDumpInfoSnapshot()),
        (std::vector<std::string>{"layer7_MOE_ROUTE_CONTRIBUTIONS"}));
    params.canonical_route_diagnostic_publication=MoECanonicalRouteDiagnosticPublication::ContinuationLocalBank;
    MoEExpertComputeStage stage(std::move(params));
    const auto dump=stage.getDumpInfoSnapshot();
    const auto keys=SnapshotCapture::possibleKeysForStage("layer7_moe_expert_ffn_overlay_fast",dump);
    EXPECT_EQ(keys,(std::vector<std::string>{"layer7_MOE_LOCAL_ROUTE_CONTRIBUTIONS"}));
    ASSERT_EQ(dump.outputs.size(),1u);
    EXPECT_EQ(dump.outputs[0].rows,10u);EXPECT_EQ(dump.outputs[0].cols,2560u);
    // Explicitly requested local evidence keeps the complete physical slot;
    // canonical completed-output selection retains none of these local rows.
    const auto &output=dump.outputs[0];
    ASSERT_TRUE(std::holds_alternative<std::monostate>(output.semantic_extent));
    EXPECT_THROW((void)snapshotCopyGeometry(output),std::logic_error);
    // Match executor preparation: legacy descriptors have payload-only slots,
    // rather than inventing an authored live-count header for this raw view.
    const auto layout=snapshotStorageLayout({output.byte_size,std::nullopt});
    EXPECT_EQ(layout.payload_storage_bytes,10u*2560u*sizeof(float));
    EXPECT_EQ(layout.metadata_bytes,0u);
    EXPECT_EQ(layout.storage_bytes,layout.payload_storage_bytes);
    EXPECT_EQ(layout.required_alignment,alignof(float));
}
#endif
