/**
 * @file Test__GPUAdaptiveImmutableLookupAdmission.cpp
 * @brief Device-free exact mandatory/optional mixed ngram admission and policy/wire invariants.
 *
 * The canonical original48 source fixture is metadata-only. Tests never load
 * model/native row payload, start a backend, allocate GPU storage or invent a
 * separate capacity resolver. All expected cache/frame sizes come from actual
 * source-owned geometry, and every capacity decision uses complete PMA plans.
 */
#include "planning/MemoryPlanner.h"
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "kernels/qwen4exp/PooledImmutableNativeStorage.h"
#include "transfer/TransferEngine.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "memory/BufferArena.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
using namespace llaminar2;
namespace
{
/** @return Explicit bounded new mode, independent of the unchanged Automatic preference. */
ImmutableRowTablePolicy policy(std::uint64_t rows)
{
    ImmutableRowTablePolicy p;p.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    p.window_bytes=65536;p.window_count=2;
    p.gpu_adaptive={rows,1u<<20,{.aging_accesses=97,.protected_numerator=2,.protected_denominator=5},
        {rows,1u<<20,{.aging_accesses=113,.protected_numerator=1,.protected_denominator=3}}};
    return p;
}
/** @return One full-main actual GPU PLE consumer with exact canonical CPU resource observation. */
DevicePlanConfig consumer(DeviceType backend,std::uint64_t rows=64)
{
    DevicePlanConfig c;c.world_rank=0;c.device={backend,0};c.device_compute_units=64;
    c.device_total_bytes=512ull<<30;c.device_free_bytes=500ull<<30;
    c.associated_host_memory=PhysicalMemoryResource{0,DeviceId::cpu(),512ull<<30,500ull<<30};
    c.max_seq_len=64;c.activation_seq_len=32;c.first_layer=0;c.last_layer=47;c.batch_size=1;
    c.ngram_execution_policy=policy(rows);return c;
}
/** @return Original native source words only; source/file authentication remains a separate real owner duty. */
ImmutableRowTableSource source(const ModelMemoryProfile &profile)
{
    const auto &g=*profile.flash_next;
    return {.rows=g.ple_table_rows,.row_elements=g.ple_row_elements,.format=static_cast<GGUFTensorType>(g.ple_table_gguf_type)};
}
/** @return Canonical immutable request after mandatory graph/source owners are declared. */
ImmutableRowTableAdmissionRequest request(const ModelMemoryProfile &profile,const std::vector<DevicePlanConfig> &configs)
{
    const std::vector<std::vector<DevicePlanConfig>> groups{configs};
    auto q=MemoryPlanner::immutableRowTableAdmissionRequest(profile,*configs.front().ngram_execution_policy,groups,{},
        configs.front().ngram_host_placement);
    if(!q)throw std::logic_error("Metadata source omitted its declared GPU consumer");
    return std::move(*q);
}
/** @return Exact existing owner line, or zero for an absent optional owner. */
std::size_t bytes(const PhysicalMemoryPlan &plan,PhysicalMemoryAllocatorIdentity id,PhysicalMemoryOwner owner)
{const auto *bom=plan.find(id);return bom?bom->bytes(owner):0;}
}

/** @test Mandatory frame/protocol/source/channel precede optional GPU then RAM storage, on both backend metadata paths. */
TEST(GPUAdaptiveImmutableLookupAdmission, MandatoryOwnersAndFinalOptionalSlotsUseOneCompletePMAPlan)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());
    const auto &profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    {
        auto c=consumer(backend);const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const auto q=request(profile,{c});ASSERT_TRUE(q.execution_provision);
        // FullModel is the existing typed grant for EVERY original routed
        // expert plus dense/shared weights. KV and graph rows are also frozen
        // before the optional cache solve; no zero-weight synthetic base is used.
        EXPECT_EQ(c.weight_residency.kind(),DeviceWeightResidency::Kind::FullModel);
        EXPECT_EQ(profile.expert_count,512);
        EXPECT_GT(bytes(frozen,{0,c.device},PhysicalMemoryOwner::PrimaryModelWeights),0u);
        EXPECT_GT(bytes(frozen,{0,c.device},PhysicalMemoryOwner::KVCache),0u);
        const auto zero=AdaptiveImmutableLookupGeometry::fromLayout(q.input.gpuAdaptiveLayout(q.requested,0,0,
            q.execution_provision->maximumLiveIds()),source(profile),q.requested.gpu_adaptive);
        const auto mandatory=zero.mandatoryBOM(q.input.consumers().front(),q.input.host());
        for(const auto &bom:mandatory.resources())
        {
            const PhysicalMemoryAllocatorIdentity id{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(frozen,id,PhysicalMemoryOwner::ImmutableLookupStaging),bom.bytes(PhysicalMemoryOwner::ImmutableLookupStaging));
            EXPECT_EQ(bytes(frozen,id,PhysicalMemoryOwner::ImmutableTableStorage),bom.bytes(PhysicalMemoryOwner::ImmutableTableStorage));
        }
        const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
        ASSERT_TRUE(selected.placement->isGPUAdaptive());EXPECT_TRUE(selected.physical_plan.fits());
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),source(profile),q.requested.gpu_adaptive);
        EXPECT_EQ(geometry.physical().owners().front().slots,64u);EXPECT_EQ(geometry.host().slots(),64u);
        const auto optional=geometry.slotDependentBOM(q.input.consumers().front(),q.input.host());
        for(const auto &bom:frozen.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);const PhysicalMemoryAllocatorIdentity id{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(selected.physical_plan,id,owner),bom.bytes(owner)+bytes(optional,id,owner));
        }
        auto resolved=c;resolved.ngram_table_placement=selected.placement;
        const auto projected=MemoryPlanner::plan(profile,{resolved}).physicalPlan();
        EXPECT_FALSE(projected.requiredFootprintMismatch(selected.physical_plan));
        EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(projected));
    }
}

/** @test C0 preserves mixed execution, all allowed tiers, mandatory ownership and original request identity. */
TEST(GPUAdaptiveImmutableLookupAdmission, ZeroCapacityIsSameModeAndCannotMasqueradeAsWholeSource)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto &profile=original.metadata().memoryProfile();
    auto c=consumer(DeviceType::CUDA,64);c.ngram_execution_policy->gpu_adaptive.gpu_cache_physical_byte_limit=0;
    c.ngram_execution_policy->gpu_adaptive.host.cache_physical_byte_limit=0;
    const auto q=request(profile,{c});const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
    const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);const auto &p=*selected.placement;
    EXPECT_EQ(p.lookupForConsumer({0,c.device}),ImmutableRowLookupExecution::AdaptiveGPUCache);
    EXPECT_THROW((void)p.storage(),std::logic_error);
    EXPECT_THROW((void)p.hostAdaptiveLayout(),std::logic_error);
    EXPECT_THROW((void)p.fixedHeadSourceRangeFor({0,c.device}),std::logic_error);
    EXPECT_THROW((void)immutableRowLookupUsesNativeDeviceStorage(ImmutableRowLookupExecution::AdaptiveGPUCache),std::invalid_argument);
    const auto &g=p.gpuAdaptiveLayout();EXPECT_FALSE(g.gpuCoordinator());EXPECT_EQ(g.gpuDirectoryRecordBound(),0u);
    EXPECT_EQ(g.owners().size(),2u);EXPECT_EQ(p.allocators().size(),2u);
    EXPECT_EQ(p.allocators().front(),(PhysicalMemoryAllocatorIdentity{0,c.device}));
    EXPECT_TRUE(p.containsAllocator({0,DeviceId::cpu()}));EXPECT_TRUE(p.containsAllocator({0,c.device}));
    EXPECT_FALSE(p.containsAllocator({0,DeviceId::cuda(1)}));
    for(const auto tier:{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped})EXPECT_TRUE(g.tiers().contains(tier));
    for(const auto &owner:g.owners())EXPECT_EQ(owner.cache_rows,0u);
    EXPECT_EQ(selected.physical_plan.summary(),frozen.summary());
    auto different=q.requested;++different.gpu_adaptive.maximum_gpu_cache_rows;
    EXPECT_THROW(p.requireRequestedPolicy(different),std::invalid_argument);
    different=q.requested;++different.gpu_adaptive.host.maximum_cache_rows;
    EXPECT_THROW(p.requireRequestedPolicy(different),std::invalid_argument);
    EXPECT_GT(bytes(frozen,{0,c.device},PhysicalMemoryOwner::ImmutableLookupStaging),0u);
    EXPECT_GT(bytes(frozen,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging),0u);
}

/** @test One-byte optional pressure cannot alter weights, KV, graph rows or the separately fitting host grant. */
TEST(GPUAdaptiveImmutableLookupAdmission, ExactFitAndOneByteShortOnlyChangeLowestPriorityGPUCacheSlots)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto &profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    {
        auto c=consumer(backend);auto q=request(profile,{c});const auto initial=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const auto full=ImmutableRowTablePlacementPlanner::select(q,initial);
        const auto *gpu=full.physical_plan.find({0,c.device});ASSERT_NE(gpu,nullptr);
        c.device_free_bytes=gpu->incrementalBytes();
        const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();q=request(profile,{c});
        const auto exact=ImmutableRowTablePlacementPlanner::select(q,frozen);
        const auto exactGeometry=AdaptiveImmutableLookupGeometry::fromLayout(exact.placement->gpuAdaptiveLayout(),source(profile),q.requested.gpu_adaptive);
        EXPECT_EQ(exactGeometry.physical().owners().front().slots,64u);
        --c.device_free_bytes;const auto shortFrozen=MemoryPlanner::plan(profile,{c}).physicalPlan();const auto shortQ=request(profile,{c});
        const auto shortSelected=ImmutableRowTablePlacementPlanner::select(shortQ,shortFrozen);
        const auto shortGeometry=AdaptiveImmutableLookupGeometry::fromLayout(shortSelected.placement->gpuAdaptiveLayout(),source(profile),q.requested.gpu_adaptive);
        EXPECT_LT(shortGeometry.physical().owners().front().slots,64u);EXPECT_EQ(shortGeometry.host().slots(),64u);
        EXPECT_EQ(c.activation_seq_len,32);EXPECT_EQ(c.max_seq_len,64);
        for(const auto &bom:frozen.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);if(owner==PhysicalMemoryOwner::ImmutableTableStorage || owner==PhysicalMemoryOwner::ImmutableLookupStaging)continue;
            const PhysicalMemoryAllocatorIdentity id{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(exact.physical_plan,id,owner),bytes(shortSelected.physical_plan,id,owner));
        }
    }
}

/** @test Canonical metadata catches topology, locality, source and observed-capacity drift before any materialization. */
TEST(GPUAdaptiveImmutableLookupAdmission, ForeignTopologySourceAndObservationCannotEnterTheMode)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto &profile=original.metadata().memoryProfile();
    auto valid=consumer(DeviceType::CUDA);
    for(int mutation=0;mutation<5;++mutation)
    {
        auto c=valid;
        switch(mutation)
        {
        case 0:c.device=DeviceId::cpu();break;
        case 1:c.first_layer=1;break;
        case 2:c.last_layer=7;break;
        case 3:c.total_shards=2;break;
        case 4:c.batch_size=2;break;
        }
        EXPECT_THROW((void)MemoryPlanner::plan(profile,{c}),std::invalid_argument);
    }
    auto other=consumer(DeviceType::ROCm);
    EXPECT_THROW((void)MemoryPlanner::plan(profile,{valid,other}),std::invalid_argument);
    const auto q=request(profile,{valid});const auto frozen=MemoryPlanner::plan(profile,{valid}).physicalPlan();
    auto altered=q;auto observed=q.input.consumers().front();--observed.admission_available_bytes;
    altered.input=ImmutableRowTablePlacementInput::forConsumers(q.input.extent(),q.input.host(),std::array{observed});
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(altered,frozen),std::invalid_argument);
    altered=q;++altered.requested.gpu_adaptive.gpu_policy.aging_accesses;
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(altered,frozen),std::invalid_argument);
    auto badPolicy=q.requested;badPolicy.host_adaptive.maximum_cache_rows=1;
    EXPECT_THROW(badPolicy.validate(),std::invalid_argument);
    badPolicy=q.requested;badPolicy.distribution=ImmutableRowTableDistribution::FixedHeadSharded;
    EXPECT_THROW(badPolicy.validate(),std::invalid_argument);
    auto missing=q;missing.execution_provision.reset();
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(missing,frozen),std::invalid_argument);
}

/** @test Lossless startup wire reconstructs mandatory BOM from original geometry, including both C0 and positive requests. */
TEST(GPUAdaptiveImmutableLookupAdmission, WireRoundTripIsLosslessAndMixedTruncatedOrOldArmsReject)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto &profile=original.metadata().memoryProfile();
    for(const std::uint64_t slots:{0u,64u})for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    {
        auto c=consumer(backend,slots);c.ngram_host_placement=ImmutableRowTableHostPlacement::onNode(3);
        const auto q=request(profile,{c});const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const std::array requests{q};const auto wire=serializeImmutableRowTableAdmissionRequests(requests);
        const auto parsed=parseImmutableRowTableAdmissionRequests(wire,0,&profile);ASSERT_EQ(parsed.size(),1u);
        EXPECT_EQ(parsed.front().requested,q.requested);EXPECT_EQ(parsed.front().input.hostPlacement(),q.input.hostPlacement());
        EXPECT_EQ(parsed.front().execution_provision->executionPlan().summary(),q.execution_provision->executionPlan().summary());
        EXPECT_EQ(*ImmutableRowTablePlacementPlanner::select(parsed.front(),frozen).placement,
            *ImmutableRowTablePlacementPlanner::select(q,frozen).placement);
        for(std::size_t n=0;n<wire.size();++n)
        {
            EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(std::span(wire).first(n),0,&profile),std::invalid_argument);
        }
        auto bad=wire;bad[4]=5;
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bad,0,&profile),std::invalid_argument);
        bad=wire;bad.push_back(0);
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bad,0,&profile),std::invalid_argument);
        bad=wire;bad[16]=static_cast<std::uint8_t>(ImmutableRowStoragePreference::MemoryMapped);
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bad,0,&profile),std::invalid_argument);
        auto foreign=profile;++foreign.flash_next->ple_row_elements;
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(wire,0,&foreign),std::invalid_argument);
    }
}

/** @test Reuse preserves exact original ceilings and accounts native frame and subordinate channel independently at C0. */
TEST(GPUAdaptiveImmutableLookupAdmission, RetainedZeroAndNonzeroOwnersKeepEveryDisjointClaim)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto &profile=original.metadata().memoryProfile();
    for(const std::uint64_t slots:{0u,64u})
    {
        auto c=consumer(DeviceType::CUDA,slots);const auto q=request(profile,{c});
        const auto selected=ImmutableRowTablePlacementPlanner::select(q,MemoryPlanner::plan(profile,{c}).physicalPlan());
        c.ngram_table_placement=selected.placement;
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),source(profile),q.requested.gpu_adaptive);
        const auto mandatory=geometry.mandatoryBOM(q.input.consumers().front(),q.input.host());
        const auto optional=geometry.cacheBOM(q.input.consumers().front(),q.input.host());
        c.retained_immutable_table_bytes=bytes(optional,{0,c.device},PhysicalMemoryOwner::ImmutableTableStorage);
        c.retained_immutable_lookup_staging_bytes=bytes(mandatory,{0,c.device},PhysicalMemoryOwner::ImmutableLookupStaging);
        c.retained_immutable_source_table_bytes=bytes(mandatory,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableTableStorage)+
            bytes(optional,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableTableStorage);
        c.retained_immutable_source_lookup_staging_bytes=bytes(mandatory,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging);
        const auto reused=MemoryPlanner::plan(profile,{c}).physicalPlan();
        EXPECT_EQ(reused.find({0,c.device})->alreadyResidentBytes(PhysicalMemoryOwner::ImmutableLookupStaging),c.retained_immutable_lookup_staging_bytes);
        EXPECT_EQ(reused.find({0,DeviceId::cpu()})->alreadyResidentBytes(PhysicalMemoryOwner::ImmutableLookupStaging),c.retained_immutable_source_lookup_staging_bytes);
        for(int mutation=0;mutation<4;++mutation)
        {
            auto bad=c;
            switch(mutation)
            {
            case 0:++bad.retained_immutable_table_bytes;break;
            case 1:--bad.retained_immutable_lookup_staging_bytes;break;
            case 2:--bad.retained_immutable_source_table_bytes;break;
            case 3:--bad.retained_immutable_source_lookup_staging_bytes;break;
            }
            EXPECT_THROW((void)MemoryPlanner::plan(profile,{bad}),std::invalid_argument);
        }
        auto changed=c;++changed.ngram_execution_policy->gpu_adaptive.maximum_gpu_cache_rows;
        EXPECT_THROW((void)MemoryPlanner::plan(profile,{changed}),std::invalid_argument);
        changed=c;++changed.ngram_execution_policy->gpu_adaptive.host.maximum_cache_rows;
        EXPECT_THROW((void)MemoryPlanner::plan(profile,{changed}),std::invalid_argument);
    }
}

/** @test Optional slots cannot rescue a missing mandatory bound or manufacture a native setup authority. */
TEST(GPUAdaptiveImmutableLookupAdmission, MissingMandatoryAndInvalidSetupRejectBeforeAnyBackendOrAllocation)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto &profile=original.metadata().memoryProfile();
    auto c=consumer(DeviceType::CUDA,0);const auto q=request(profile,{c});const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
    auto noMandatory=c;noMandatory.ngram_execution_policy.reset();
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(q,MemoryPlanner::plan(profile,{noMandatory}).physicalPlan()),std::invalid_argument);
    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(q.input.gpuAdaptiveLayout(q.requested,0,0,q.execution_provision->maximumLiveIds()),source(profile),q.requested.gpu_adaptive);
    EXPECT_THROW((void)qwen4exp::PooledImmutableOwnerPreparedStorage::create(qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(geometry.physical(),geometry.coordinator(),0,701,17),{},nullptr),std::invalid_argument);
    auto memory=std::make_shared<PhysicalMemoryAuthority>(std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(frozen),0);
    EXPECT_THROW((void)qwen4exp::PooledImmutableOwnerPreparedStorage::create(qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(geometry.physical(),geometry.coordinator(),0,701,17),memory,nullptr),std::invalid_argument);
    EXPECT_THROW((void)qwen4exp::PooledImmutableOwnerPreparedStorage::create(qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(geometry.physical(),geometry.coordinator(),0,701,17),memory,reinterpret_cast<void *>(1)),std::invalid_argument);
    EXPECT_EQ(memory->claimedBytes(c.device,PhysicalMemoryOwner::ImmutableTableStorage,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
    EXPECT_EQ(memory->claimedBytes(c.device,PhysicalMemoryOwner::ImmutableLookupStaging,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}

/** @test Retained GPU frame/protocol claims stay distinct from optional native cache bytes at C0. */
TEST(GPUAdaptiveImmutableLookupAdmission, DeviceRetirementIncludesActualStagingAsADistinctClass)
{
    ModelDeviceMemoryRetention row{.device=DeviceId::cuda(0),.prepared_weight_bytes=1,
        .immutable_table_bytes=0,.immutable_lookup_staging_bytes=64};
    EXPECT_TRUE(row.valid());EXPECT_EQ(row.totalBytes(),65u);
    row.immutable_table_bytes=128;EXPECT_EQ(row.totalBytes(),193u);
    row.immutable_lookup_staging_bytes=std::numeric_limits<std::size_t>::max();
    EXPECT_FALSE(row.valid());
    EXPECT_THROW((void)row.totalBytes(),std::overflow_error);
    row={.device=DeviceId::rocm(0),.immutable_lookup_staging_bytes=64};
    EXPECT_TRUE(row.valid());EXPECT_EQ(row.totalBytes(),64u);
    row.device=DeviceId::cpu();EXPECT_FALSE(row.valid());
}

/** @test Deduplication cannot underprice two mutually inconsistent layouts/policies for one original consumer. */
TEST(GPUAdaptiveImmutableLookupAdmission, SharedConsumerRejectsConflictingResolvedModesBeforeFirstOwnerCharging)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto &profile=original.metadata().memoryProfile();
    auto c=consumer(DeviceType::CUDA,64);const auto q=request(profile,{c});
    const auto selected=ImmutableRowTablePlacementPlanner::select(q,MemoryPlanner::plan(profile,{c}).physicalPlan());
    c.ngram_table_placement=selected.placement;
    auto changed=c;changed.ngram_execution_policy->gpu_adaptive.maximum_gpu_cache_rows=65;
    changed.ngram_execution_policy->gpu_adaptive.gpu_cache_physical_byte_limit=0;
    changed.ngram_table_placement.reset();const auto otherQ=request(profile,{changed});
    const auto other=ImmutableRowTablePlacementPlanner::select(otherQ,MemoryPlanner::plan(profile,{changed}).physicalPlan());
    changed.ngram_table_placement=other.placement;
    EXPECT_THROW((void)MemoryPlanner::plan(profile,{c,changed}),std::invalid_argument);
    auto plain=c;plain.ngram_table_placement.reset();plain.ngram_execution_policy.reset();
    const auto whole=ImmutableRowTablePlacementPlanner::select(ImmutableRowTablePolicy{ImmutableRowStoragePreference::VRAM},
        q.input,MemoryPlanner::plan(profile,{plain}).physicalPlan(),{PhysicalMemoryPlan{},{},{}});
    auto wholeCfg=plain;wholeCfg.ngram_execution_policy=ImmutableRowTablePolicy{ImmutableRowStoragePreference::VRAM};
    wholeCfg.ngram_table_placement=whole.placement;
    EXPECT_THROW((void)MemoryPlanner::plan(profile,{c,wholeCfg}),std::invalid_argument);
}

/** @test GPU graph arena declarations precede source setup without weakening actual boundary requirements. */
TEST(GPUAdaptiveImmutableLookupAdmission, GraphArenaDeclarationDoesNotAcquireUnpreparedSource)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource original(file.path());
    const auto &profile = original.metadata().memoryProfile();
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto slots : {0u, 4u})
        {
            auto c = consumer(backend, slots);
            const auto selected = ImmutableRowTablePlacementPlanner::select(
                request(profile, {c}), MemoryPlanner::plan(profile, {c}).physicalPlan());
            GraphConfig config;
            config.default_device = c.device;
            config.max_seq_len = c.max_seq_len;
            config.max_activation_rows = c.activation_seq_len;
            config.max_request_count = 1;
            Qwen4ExpGraphConfigBuilder::populateFromDirectory(test::qwen4MetadataFixture(false), config);
            config.ngram_table_placement = selected.placement;
            Qwen4ExpGraph graph(config);
            BufferArena arena;
            // This is the actual factory's declaration order, before any model
            // source boundary exists. No backend or physical owner is created.
            ASSERT_NO_THROW(graph.setArena(&arena));
            for (const auto role : {BufferId::QFN_POOLED_OWNER_CACHE,
                                   BufferId::QFN_POOLED_OWNER_FRAME,
                                   BufferId::QFN_NGRAM_LOOKUP_STATE})
                EXPECT_FALSE(arena.isRegistered(role));
            EXPECT_EQ(graph.config().ngram_table_placement.get(), selected.placement.get());
            try
            {
                (void)graph.retainedStorageBoundary(c.device);
                FAIL() << "An arena declaration cannot supply the missing original source owner";
            }
            catch (const std::logic_error &error)
            {
                EXPECT_NE(std::string(error.what()).find(
                    "Flash Next adaptive lookup requires its exact model-owned mixed placement"), std::string::npos);
            }
            ASSERT_NO_THROW(graph.setArena(nullptr));
        }
}

/** @test Exact root partitions, not a K-bank reserve or aggregate owner subtraction, compose every candidate. */
TEST(GPUAdaptiveImmutableLookupAdmission, SplitCandidateFloorAndGrowthEqualActualRootsAtCapacityBoundaries)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto& profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
        auto c=consumer(backend,2048);const auto q=request(profile,{c});const auto nativeSource=source(profile);
        const auto gpu=q.input.consumers().front(),host=q.input.host();
        for(const std::size_t ids:{16u,32u,256u,272u,512u}){
            const auto zero=AdaptiveImmutableLookupGeometry::fromLayout(q.input.gpuAdaptiveLayout(q.requested,0,0,ids),nativeSource,q.requested.gpu_adaptive);
            const auto floor=zero.mandatoryBOM(gpu,host);
            for(const auto slots:std::array<std::size_t,13>{0,1,2,3,7,8,15,16,17,ids-1,ids,ids+1,2*ids+1})
                for(const std::size_t ram:{0u,3u}){
                    SCOPED_TRACE(::testing::Message()<<"backend="<<static_cast<int>(backend)<<" ids="<<ids<<" slots="<<slots<<" ram="<<ram);
                    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(q.input.gpuAdaptiveLayout(q.requested,slots,ram,ids),nativeSource,q.requested.gpu_adaptive);
                    const auto growth=geometry.slotDependentBOM(gpu,host);
                    PhysicalMemoryPlanBuilder partitioned(floor),complete(geometry.mandatoryBOM(gpu,host));
                    for(const auto& line:growth.resources())partitioned.add(line);
                    // resources() borrows its owning plan; retain that plan
                    // through every aggregate contribution.
                    const auto optional=geometry.cacheBOM(gpu,host);
                    for(const auto& line:optional.resources())complete.add(line);
                    EXPECT_EQ(partitioned.build().summary(),complete.build().summary());
                    const auto* gpuGrowth=growth.find({gpu.world_rank,gpu.device});ASSERT_NE(gpuGrowth,nullptr);
                    EXPECT_EQ(gpuGrowth->totalBytes(),geometry.gpuSlotDependentBytes());
                    EXPECT_EQ(geometry.physical().owners().front().cache.bytes%64,0u);
                    EXPECT_EQ(geometry.coordinator().policyOffsets().bytes%64,0u);
                    const auto owner=qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(geometry.physical(),geometry.coordinator(),0,701,17);
                    const auto policy=qwen4exp::PooledImmutableCoordinatorNativePlan::fromGeometry(geometry.physical(),geometry.coordinator(),701,17);
                    const auto sourceGeometry=qwen4exp::PooledImmutableSourceGeometry::fromPlans(owner,policy,nativeSource);
                    const auto& o=owner.parameters();const auto& p=policy.parameters();const auto& io=sourceGeometry.parameters().offsets;
                    EXPECT_EQ(geometry.mandatoryDeviceBytes(),o.frame.bytes+o.native.bytes+p.frame.bytes+p.native.bytes+io.frame_bytes+io.native_bytes);
                    EXPECT_EQ(o.native.reply_capacity,std::min(slots,ids));
                    EXPECT_EQ(io.frame_bytes%64,0u);
                    if(!slots)EXPECT_EQ(gpuGrowth->totalBytes(),0u);
                    else EXPECT_GT(gpuGrowth->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),0u);
                }
        }
    }
    // A full15 verifier has sixteen target positions. Keep every original head
    // without growing a second counter or reducing the admitted MTP geometry.
    EXPECT_EQ(immutableRowTableMaximumLiveIds(1,1,16,16,16),256u);
}

/** @test An explicit cache ceiling pays native/source bank growth as well as cache/policy payloads. */
TEST(GPUAdaptiveImmutableLookupAdmission, SplitCandidateCacheCeilingIncludesExactNativeAndSourceGrowth)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto& profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
        auto c=consumer(backend);auto q=request(profile,{c});
        const auto full=ImmutableRowTablePlacementPlanner::select(q,MemoryPlanner::plan(profile,{c}).physicalPlan());
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(full.placement->gpuAdaptiveLayout(),source(profile),q.requested.gpu_adaptive);
        ASSERT_GT(geometry.gpuSlotDependentBytes(),geometry.gpuCacheBytes());
        c.ngram_execution_policy->gpu_adaptive.gpu_cache_physical_byte_limit=geometry.gpuCacheBytes();
        q=request(profile,{c});const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const auto bounded=ImmutableRowTablePlacementPlanner::select(q,frozen);
        const auto actual=AdaptiveImmutableLookupGeometry::fromLayout(bounded.placement->gpuAdaptiveLayout(),source(profile),q.requested.gpu_adaptive);
        EXPECT_LT(actual.physical().owners().front().slots,geometry.physical().owners().front().slots);
        EXPECT_LE(actual.gpuSlotDependentBytes(),q.requested.gpu_adaptive.gpu_cache_physical_byte_limit);
        for(const auto& line:frozen.resources())for(std::size_t index=0;index<PhysicalMemoryBOM::ownerCount();++index){
            const auto owner=PhysicalMemoryBOM::ownerAt(index);
            if(owner==PhysicalMemoryOwner::ImmutableTableStorage || owner==PhysicalMemoryOwner::ImmutableLookupStaging)continue;
            EXPECT_EQ(bytes(bounded.physical_plan,{line.resource().world_rank,line.resource().device},owner),line.bytes(owner));
        }
    }
}

/** @test Public PMA selection and resolved placement retain complete exact physical-root costs. */
TEST(GPUAdaptiveImmutableLookupAdmission, SplitCandidateResolvedPlacementUsesCompleteOriginalByteCeiling)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto& profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}){
        const auto c=consumer(backend);const auto q=request(profile,{c});
        const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
        const auto nativeSource=source(profile);
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),nativeSource,q.requested.gpu_adaptive);
        ASSERT_GT(geometry.gpuSlotDependentBytes(),geometry.gpuCacheBytes());
        const auto originalSlots=geometry.physical().owners().front().slots;
        // These exact source-bound native factories declare every original
        // cache/frame/native root; the test owns no allocation/capacity ledger.
        const auto roots=[&](const AdaptiveImmutableLookupGeometry& candidate){
            const auto owner=qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(candidate.physical(),candidate.coordinator(),0,701,17);
            const auto coordinator=qwen4exp::PooledImmutableCoordinatorNativePlan::fromGeometry(candidate.physical(),candidate.coordinator(),701,17);
            const auto sourceGeometry=qwen4exp::PooledImmutableSourceGeometry::fromPlans(owner,coordinator,nativeSource);
            const auto& o=owner.parameters();const auto& p=coordinator.parameters();const auto& io=sourceGeometry.parameters().offsets;
            return std::array<std::size_t,2>{o.cache.bytes+p.cache.bytes,
                o.frame.bytes+o.native.bytes+p.frame.bytes+p.native.bytes+io.frame_bytes+io.native_bytes};
        };
        for(const auto ceiling:std::array{geometry.gpuSlotDependentBytes(),geometry.gpuSlotDependentBytes()-1,geometry.gpuCacheBytes()}){
            auto bounded=c;bounded.ngram_execution_policy->gpu_adaptive.gpu_cache_physical_byte_limit=ceiling;
            const auto input=request(profile,{bounded});
            // Canonical PMA selection alone creates the private resolved
            // placement. No test friend, constructor bypass or fake admission
            // may introduce an invalid physical candidate.
            const auto actual=ImmutableRowTablePlacementPlanner::select(input,frozen);
            const auto retained=AdaptiveImmutableLookupGeometry::fromLayout(actual.placement->gpuAdaptiveLayout(),nativeSource,input.requested.gpu_adaptive);
            const auto zero=AdaptiveImmutableLookupGeometry::fromLayout(input.input.gpuAdaptiveLayout(input.requested,0,0,input.execution_provision->maximumLiveIds()),nativeSource,input.requested.gpu_adaptive);
            const auto fullRoots=roots(retained),floorRoots=roots(zero);
            ASSERT_GE(fullRoots[1],floorRoots[1]);
            EXPECT_EQ(retained.gpuCacheBytes(),fullRoots[0]);
            EXPECT_EQ(retained.mandatoryDeviceBytes(),fullRoots[1]);
            EXPECT_EQ(retained.gpuSlotDependentBytes(),fullRoots[0]+fullRoots[1]-floorRoots[1]);
            EXPECT_LE(retained.gpuSlotDependentBytes(),ceiling);
            const PhysicalMemoryAllocatorIdentity allocator{c.world_rank,c.device};
            EXPECT_EQ(bytes(actual.physical_plan,allocator,PhysicalMemoryOwner::ImmutableTableStorage),
                bytes(frozen,allocator,PhysicalMemoryOwner::ImmutableTableStorage)+fullRoots[0]);
            EXPECT_EQ(bytes(actual.physical_plan,allocator,PhysicalMemoryOwner::ImmutableLookupStaging),
                bytes(frozen,allocator,PhysicalMemoryOwner::ImmutableLookupStaging)+fullRoots[1]-floorRoots[1]);
            PhysicalMemoryPlanBuilder complete(frozen);
            const auto contribution=retained.slotDependentBOM(input.input.consumers().front(),input.input.host());
            for(const auto& line:contribution.resources())complete.add(line);
            EXPECT_EQ(actual.physical_plan.summary(),complete.build().summary());
            if(ceiling==geometry.gpuSlotDependentBytes())EXPECT_EQ(retained.physical().owners().front().slots,originalSlots);
            else EXPECT_LT(retained.physical().owners().front().slots,originalSlots);
        }
    }
}
