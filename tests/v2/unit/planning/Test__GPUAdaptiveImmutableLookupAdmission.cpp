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
#include "config/OrchestrationConfigDocument.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "fixtures/planning/ImmutableRowTableAdmissionWireReplayAccess.h"
#include "fixtures/planning/RecordedNgramOrdering.h"
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/RankMemoryPlanInputs.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "../../utils/PooledNgramReplayBudget.h"
#include "integration/parity/qwen4exp/Qwen4ExpCombinedLocalHotCases.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "execution/qwen4exp/PooledImmutableRetainedParticipantBudget.h"
#include "kernels/qwen4exp/PooledImmutableLocalHotTerminal.h"
#include "execution/qwen4exp/PooledImmutableDirectedChannelGeometry.h"
#include "execution/qwen4exp/PooledImmutableDirectedTransport.h"
#include "kernels/qwen4exp/PooledImmutableNativeStorage.h"
#include "transfer/TransferEngine.h"
#include "models/qwen4exp/Qwen4ExpGraph.h"
#include "models/qwen4exp/Qwen4ExpGraphConfigBuilder.h"
#include "memory/BufferArena.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <type_traits>
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
    c.ngram_execution_policy=policy(rows);c.native_ordering_events=test::recordedNgramOrdering(c.device);test::declareRecordedNgramHostServices(c);return c;
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
    auto originalConfigs=configs;std::vector<DeviceId> originalDevices;
    for(const auto& cfg:configs)if(cfg.device.is_gpu() && std::find(originalDevices.begin(),originalDevices.end(),cfg.device)==originalDevices.end())originalDevices.push_back(cfg.device);
    const auto recorded=test::NativeOrderingEventGroupReplayAccess::projection(originalDevices);
    for(auto& cfg:originalConfigs)cfg.native_ordering_events=recorded;
    test::shareRecordedNgramHostServices(originalConfigs);
    const std::vector<std::vector<DevicePlanConfig>> groups{originalConfigs};
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
        const auto mandatory=llaminar2::test::recordedNgramMandatory(zero,q.input.consumers().front(),q.input.host());
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
        const auto optional=llaminar2::test::recordedNgramOptional(geometry,q.input.consumers().front(),q.input.host());
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
    EXPECT_EQ(p.allocators().front(),g.owners().front().key.allocator);
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
            const auto owner=PhysicalMemoryBOM::ownerAt(i);if(owner==PhysicalMemoryOwner::ImmutableTableStorage || owner==PhysicalMemoryOwner::ImmutableLookupStaging || owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent)continue;
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
        // The receiver selects its own sealed resource source; certificate
        // authority and numeric bounds cannot be injected through startup wire.
        static_assert(!std::is_constructible_v<NativeOrderingEventBudgetProjection,std::uint64_t>);
        auto otherLocalSource=q;otherLocalSource.native_ordering_events=NativeOrderingEventBudgetProjection::installed();
        EXPECT_EQ(serializeImmutableRowTableAdmissionRequests(std::array{otherLocalSource}),wire);
        auto forgedCertificate=wire;
        for(const std::uint8_t byte:std::array<std::uint8_t,8>{'c','e','r','t',0xff,0xff,0xff,0xff})forgedCertificate.push_back(byte);
        EXPECT_THROW((void)test::ImmutableRowTableAdmissionWireReplayAccess::parse(forgedCertificate,0,&profile,q.native_ordering_events,q.retained_host_services),std::invalid_argument);

        const auto parsed=test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&profile,q.native_ordering_events,q.retained_host_services);ASSERT_EQ(parsed.size(),1u);
        EXPECT_EQ(parsed.front().retained_host_services,q.retained_host_services);
        // The local declaration is retained, never encoded as permission on wire.
        auto noLocalDeclaration=q;noLocalDeclaration.retained_host_services.reset();
        EXPECT_EQ(serializeImmutableRowTableAdmissionRequests(std::array{noLocalDeclaration}),wire);
        const auto wireOnly=test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&profile,q.native_ordering_events);
        ASSERT_EQ(wireOnly.size(),1u);
        ASSERT_TRUE(wireOnly.front().execution_provision);
        EXPECT_FALSE(wireOnly.front().retained_host_services); // Pricing metadata carries no local creation authority.
        EXPECT_EQ(wireOnly.front().execution_provision->consumers(),q.execution_provision->consumers());
        EXPECT_EQ(wireOnly.front().execution_provision->executionPlan().summary(),q.execution_provision->executionPlan().summary());
        EXPECT_EQ(wireOnly.front().execution_provision->requiredBase().summary(),q.execution_provision->requiredBase().summary());
        const auto priced=ImmutableRowTablePlacementPlanner::select(wireOnly.front(),frozen);
        const auto originalPriced=ImmutableRowTablePlacementPlanner::select(q,frozen);
        EXPECT_EQ(priced.physical_plan.summary(),originalPriced.physical_plan.summary());
        EXPECT_EQ(*priced.placement,*originalPriced.placement);
        EXPECT_EQ(parsed.front().requested,q.requested);EXPECT_EQ(parsed.front().input.hostPlacement(),q.input.hostPlacement());
        EXPECT_EQ(parsed.front().execution_provision->executionPlan().summary(),q.execution_provision->executionPlan().summary());
        EXPECT_EQ(*ImmutableRowTablePlacementPlanner::select(parsed.front(),frozen).placement,
            *ImmutableRowTablePlacementPlanner::select(q,frozen).placement);
        for(std::size_t n=0;n<wire.size();++n)
        {
            EXPECT_THROW((void)test::ImmutableRowTableAdmissionWireReplayAccess::parse(std::span(wire).first(n),0,&profile,q.native_ordering_events,q.retained_host_services),std::invalid_argument);
        }
        auto bad=wire;bad[4]=5;
        EXPECT_THROW((void)test::ImmutableRowTableAdmissionWireReplayAccess::parse(bad,0,&profile,q.native_ordering_events,q.retained_host_services),std::invalid_argument);
        bad=wire;bad.push_back(0);
        EXPECT_THROW((void)test::ImmutableRowTableAdmissionWireReplayAccess::parse(bad,0,&profile,q.native_ordering_events,q.retained_host_services),std::invalid_argument);
        bad=wire;bad[16]=static_cast<std::uint8_t>(ImmutableRowStoragePreference::MemoryMapped);
        EXPECT_THROW((void)test::ImmutableRowTableAdmissionWireReplayAccess::parse(bad,0,&profile,q.native_ordering_events,q.retained_host_services),std::invalid_argument);
        auto foreign=profile;++foreign.flash_next->ple_row_elements;
        EXPECT_THROW((void)test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&foreign,q.native_ordering_events,q.retained_host_services),std::invalid_argument);
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
        const auto mandatory=llaminar2::test::recordedNgramMandatory(geometry,q.input.consumers().front(),q.input.host());
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
    auto absentService=c;absentService.retained_host_services.reset();
    EXPECT_THROW((void)MemoryPlanner::plan(profile,{absentService}),std::logic_error);
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
            const auto floor=llaminar2::test::recordedNgramMandatory(zero,gpu,host);
            for(const auto slots:std::array<std::size_t,13>{0,1,2,3,7,8,15,16,17,ids-1,ids,ids+1,2*ids+1})
                for(const std::size_t ram:{0u,3u}){
                    SCOPED_TRACE(::testing::Message()<<"backend="<<static_cast<int>(backend)<<" ids="<<ids<<" slots="<<slots<<" ram="<<ram);
                    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(q.input.gpuAdaptiveLayout(q.requested,slots,ram,ids),nativeSource,q.requested.gpu_adaptive);
                    const auto growth=llaminar2::test::recordedNgramOptional(geometry,gpu,host);
                    PhysicalMemoryPlanBuilder partitioned(floor),complete(llaminar2::test::recordedNgramMandatory(geometry,gpu,host));
                    for(const auto& line:growth.resources())partitioned.add(line);
                    // resources() borrows its owning plan; retain that plan
                    // through every aggregate contribution.
                    const auto optional=geometry.cacheBOM(gpu,host);
                    for(const auto& line:optional.resources())complete.add(line);
                    EXPECT_EQ(partitioned.build().summary(),complete.build().summary());
                    const auto* gpuGrowth=growth.find({gpu.world_rank,gpu.device});ASSERT_NE(gpuGrowth,nullptr);
                    EXPECT_EQ(gpuGrowth->bytes(PhysicalMemoryOwner::ImmutableTableStorage)+gpuGrowth->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),geometry.gpuSlotDependentBytes());
                    EXPECT_GE(gpuGrowth->totalBytes(),geometry.gpuSlotDependentBytes());
                    EXPECT_EQ(geometry.physical().owners().front().cache.bytes%64,0u);
                    EXPECT_EQ(geometry.coordinator().policyOffsets().bytes%64,0u);
                    const auto owner=qwen4exp::PooledImmutableOwnerNativePlan::fromGeometry(geometry.physical(),geometry.coordinator(),0,701,17);
                    const auto policy=qwen4exp::PooledImmutableCoordinatorNativePlan::fromGeometry(geometry.physical(),geometry.coordinator(),701,17);
                    const auto sourceGeometry=qwen4exp::PooledImmutableSourceGeometry::fromPlans(owner,policy,nativeSource,1);
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
            if(owner==PhysicalMemoryOwner::ImmutableTableStorage || owner==PhysicalMemoryOwner::ImmutableLookupStaging || owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent)continue;
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
            const auto sourceGeometry=qwen4exp::PooledImmutableSourceGeometry::fromPlans(owner,coordinator,nativeSource,1);
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
            const auto contribution=llaminar2::test::recordedNgramOptional(retained,input.input.consumers().front(),input.input.host());
            for(const auto& line:contribution.resources())complete.add(line);
            EXPECT_EQ(actual.physical_plan.summary(),complete.build().summary());
            if(ceiling==geometry.gpuSlotDependentBytes())EXPECT_EQ(retained.physical().owners().front().slots,originalSlots);
            else EXPECT_LT(retained.physical().owners().front().slots,originalSlots);
        }
    }
}

/** @test TP2 retains authored coordinates, concurrent channels and one shared source/worker through final wire pricing. */
TEST(GPUAdaptiveImmutableLookupAdmission, ConsumerLocalTP2PreservesAuthoredCoordinatesAndSharedSourceBOM)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource original(file.path());
    const auto& profile = original.metadata().memoryProfile();
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm}) {
        std::vector<DevicePlanConfig> configs{consumer(backend), consumer(backend)};
        const std::array devices{DeviceId{backend, 0}, DeviceId{backend, 1}};
        const auto projection = test::NativeOrderingEventGroupReplayAccess::projection(devices);
        const auto assignment=TensorParallelConfig::equalSplit(2,profile.n_heads,profile.n_kv_heads,
            profile.d_ff,profile.vocab_size,std::vector<DeviceId>{devices[1],devices[0]});
        for (std::size_t i = 0; i < configs.size(); ++i) {
            auto& cfg = configs[i];
            cfg.device = devices[i];
            cfg.total_shards = 2;
            cfg.shard_index = static_cast<int>(1 - i);
            cfg.ngram_tp_topology = ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal;
            cfg.local_tp_backend = backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            // Use the production assignment producer, preserving reversed
            // communicator order and all FFN/vocabulary/rank provenance.
            cfg.tensor_parallel_assignment=assignment.forDevice(cfg.device);
            cfg.native_ordering_events = projection;
            test::declareRecordedNgramHostServices(cfg);
        }
        test::shareRecordedNgramHostServices(configs);
        ASSERT_EQ(configs.front().retained_host_services->instances().size(), 1u);
        EXPECT_EQ(configs.front().retained_host_services->instances().front().consumer, DeviceId::cpu());
        const auto q = request(profile, configs);
        const auto frozen = MemoryPlanner::plan(profile, configs).physicalPlan();
        const auto selected = ImmutableRowTablePlacementPlanner::select(q, frozen);
        const auto& layout = selected.placement->gpuAdaptiveLayout();
        EXPECT_EQ(layout.adaptiveGPUDomain(), ImmutableRowTableAdaptiveGPUDomain::ConsumerLocal);
        EXPECT_EQ(layout.consumers().size(), 2u);
        EXPECT_EQ(layout.owners().size(), 3u);
        const auto shared = AdaptiveImmutableLookupGeometry::sharedHostGeometry(layout, q.requested.gpu_adaptive);
        const auto first = AdaptiveImmutableLookupGeometry::forRequester(layout, source(profile), q.requested.gpu_adaptive, {0, devices[0]});
        const auto channels = qwen4exp::ImmutableLookupServiceGeometry::checked(first.channelLayout(), devices);
        EXPECT_EQ(bytes(selected.physical_plan, {0, DeviceId::cpu()}, PhysicalMemoryOwner::ImmutableTableStorage),
            shared.sourceBytes() + shared.storageBytes());
        EXPECT_EQ(bytes(selected.physical_plan, {0, DeviceId::cpu()}, PhysicalMemoryOwner::ImmutableLookupStaging),
            shared.executionBytes() + channels.hostBytes());
        const auto workerBOM = configs.front().retained_host_services->bom(
            {0, DeviceId::cpu(), RetainedHostServiceRole::ImmutableLookupIO}, *configs.front().associated_host_memory);
        EXPECT_EQ(bytes(selected.physical_plan, {0, DeviceId::cpu()}, PhysicalMemoryOwner::RetainedHostServiceStorage),
            bytes(workerBOM, {0, DeviceId::cpu()}, PhysicalMemoryOwner::RetainedHostServiceStorage));
        const auto wire = serializeImmutableRowTableAdmissionRequests(std::array{q});
        const auto parsed = test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire, 0, &profile, projection);
        ASSERT_EQ(parsed.size(), 1u);
        EXPECT_EQ(parsed.front().execution_provision->consumers(), q.execution_provision->consumers());
        for (std::size_t i = 0; i < configs.size(); ++i) {
            const auto& identity = q.execution_provision->consumers()[i];
            EXPECT_EQ(identity.tp_coordinate, configs[i].shard_index);
            EXPECT_EQ(identity.query_head_start, configs[i].tensor_parallel_assignment->head_start);
            EXPECT_EQ(identity.kv_head_start, configs[i].tensor_parallel_assignment->kv_head_start);
            EXPECT_EQ(configs[i].tensor_parallel_assignment->head_start, static_cast<int>(1 - i) * (profile.n_heads / 2));
        }
        auto forged = q.execution_provision->consumers();
        forged.front().query_head_start = profile.n_heads;
        EXPECT_THROW(test::MemoryPlannerOrderingReplayAccess::restore(profile,q.requested,q.input,forged,
            q.execution_provision->executionPlan(),q.execution_provision->requiredBase(),projection,q.retained_host_services),
            std::invalid_argument);
        forged = q.execution_provision->consumers();
        forged.back().tp_coordinate = forged.front().tp_coordinate;
        EXPECT_THROW(test::MemoryPlannerOrderingReplayAccess::restore(profile,q.requested,q.input,forged,
            q.execution_provision->executionPlan(),q.execution_provision->requiredBase(),projection,q.retained_host_services),
            std::invalid_argument);
        const auto restored = ImmutableRowTablePlacementPlanner::select(parsed.front(), frozen);
        EXPECT_EQ(*restored.placement, *selected.placement);
        EXPECT_EQ(restored.physical_plan.summary(), selected.physical_plan.summary());
        auto invalid = configs;
        invalid[1].shard_index = invalid[0].shard_index;
        EXPECT_THROW(request(profile, invalid), std::invalid_argument);
        invalid = configs;
        invalid[1].tensor_parallel_assignment.reset();
        EXPECT_THROW(request(profile, invalid), std::invalid_argument);
    }
}

/** @test The actual rank input resolver keeps original communicator ordering and query assignments for ConsumerLocal TP2. */
TEST(GPUAdaptiveImmutableLookupAdmission, ConsumerLocalTP2RankInputsKeepOriginalAssignments)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource original(file.path());
    const auto& profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        RankExecutionPlan plan;
        plan.rank=0;
        plan.hostname="consumer-local-fixture";
        plan.first_layer=0;
        plan.last_layer=47;
        plan.runtime.max_seq_len=64;
        plan.runtime.batch_size=1;
        plan.runtime.mtp.enabled=false;
        plan.runtime.mtp.draft_tokens=0;
        plan.runtime.mtp.graph_capacity_draft_tokens=0;
        plan.runtime.ngram_table_policy=policy(64);
        RankInventory inventory;
        inventory.rank=0;
        inventory.cpu_worker_threads=8;
        inventory.cpu_cores=8;
        inventory.cpu_execution=test::kSyntheticCPUExecutionGeometry;
        inventory.cpu.memory_bytes=512ull<<30;
        inventory.cpu.free_memory_bytes=500ull<<30;
        for(const auto ordinal:{1,0}) {
            plan.local_tp_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId({backend,ordinal},plan.hostname,0));
            DeviceInfo gpu;
            gpu.type=backend;
            gpu.local_device_id=ordinal;
            gpu.memory_bytes=512ull<<30;
            gpu.free_memory_bytes=500ull<<30;
            gpu.compute_units=64;
            inventory.gpus.push_back(gpu);
        }
        plan.primary_device=plan.local_tp_devices.front();
        const auto configs=buildRankMemoryPlanInputs({.model=profile,.plan=plan,.inventory=inventory});
        ASSERT_EQ(configs.size(),2u);
        for(std::size_t i=0;i<configs.size();++i) {
            EXPECT_EQ(configs[i].device,DeviceId(backend,static_cast<int>(1-i)));
            EXPECT_EQ(configs[i].shard_index,static_cast<int>(i));
            EXPECT_EQ(configs[i].ngram_tp_topology,ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal);
            ASSERT_TRUE(configs[i].tensor_parallel_assignment);
            EXPECT_EQ(configs[i].tensor_parallel_assignment->device,configs[i].device);
            EXPECT_EQ(configs[i].tensor_parallel_assignment->head_start,static_cast<int>(i)*profile.n_heads/2);
        }
        auto invalid=plan;
        invalid.local_tp_devices[1]=GlobalDeviceAddress::fromLocalDeviceId(
            backend==DeviceType::CUDA?DeviceId::rocm(0):DeviceId::cuda(0),plan.hostname,0);
        EXPECT_THROW(immutableRowTableTPTopology(profile,invalid),std::invalid_argument);
    }
}

/** @test Original generic source/PMA/wire population and full-main topology selection share exact 1/2/4/8 identities. */
TEST(GPUAdaptiveImmutableLookupAdmission, ConsumerLocalGenericMetadataPopulationBOMAndProvisionWireReplay)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource original(file.path());
    const auto& profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
      for(const auto backing:{AdaptiveImmutableRowBacking::SSD,AdaptiveImmutableRowBacking::RAM})
        for(const auto count:{1u,2u,4u,8u}) {
            std::vector<DeviceId> devices;
            std::vector<DevicePlanConfig> configs;
            for(std::size_t coordinate=0;coordinate<count;++coordinate)
                devices.push_back(DeviceId(backend,static_cast<int>(count-coordinate-1)));
            const auto projection=test::NativeOrderingEventGroupReplayAccess::projection(devices);
            const auto assignments=TensorParallelConfig::equalSplit(static_cast<int>(count),profile.n_heads,
                profile.n_kv_heads,profile.d_ff,profile.vocab_size,devices);
            for(std::size_t coordinate=0;coordinate<count;++coordinate) {
                auto cfg=consumer(backend);
                cfg.ngram_execution_policy->gpu_adaptive.source_backing=backing;
                if(backing==AdaptiveImmutableRowBacking::RAM)cfg.ngram_execution_policy->gpu_adaptive.host={};
                cfg.device=devices[coordinate];
                cfg.total_shards=static_cast<int>(count);
                cfg.shard_index=static_cast<int>(coordinate);
                cfg.native_ordering_events=projection;
                // A singleton has no TP assignment. Plural participants use
                // the same complete producer as the actual rank planner.
                if(count>1) {
                    cfg.tensor_parallel_assignment=assignments.forDevice(cfg.device);
                    cfg.ngram_tp_topology=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal;
                    cfg.local_tp_backend=backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
                }
                test::declareRecordedNgramHostServices(cfg);
                configs.push_back(std::move(cfg));
            }
            test::shareRecordedNgramHostServices(configs);
            ASSERT_EQ(configs.front().retained_host_services->instances().size(),1u);
            const auto q=request(profile,configs);
            const auto frozen=MemoryPlanner::plan(profile,configs).physicalPlan();
            const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
            const auto& layout=selected.placement->gpuAdaptiveLayout();
            EXPECT_EQ(layout.consumers().size(),count);
            EXPECT_EQ(layout.owners().size(),count+1);
            const auto local=count==1
                ? AdaptiveImmutableLookupGeometry::fromLayout(layout,source(profile),q.requested.gpu_adaptive)
                : AdaptiveImmutableLookupGeometry::forRequester(layout,source(profile),q.requested.gpu_adaptive,
                    {0,devices.front()});
            const auto shared=count==1?local.host():AdaptiveImmutableLookupGeometry::sharedHostGeometry(layout,q.requested.gpu_adaptive);
            EXPECT_EQ(shared.sourceStorage(),backing==AdaptiveImmutableRowBacking::RAM?ImmutableRowStorage::RAM:ImmutableRowStorage::MemoryMapped);
            if(backing==AdaptiveImmutableRowBacking::RAM){EXPECT_EQ(shared.slots(),0u);EXPECT_EQ(shared.storageBytes(),0u);}
            const auto channels=qwen4exp::ImmutableLookupServiceGeometry::checked(local.channelLayout(),devices);
            EXPECT_EQ(bytes(selected.physical_plan,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableTableStorage),
                shared.sourceBytes()+shared.storageBytes());
            EXPECT_EQ(bytes(selected.physical_plan,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging),
                shared.executionBytes()+channels.hostBytes());
            for(const auto device:devices) {
                const auto geometry=count==1?local:AdaptiveImmutableLookupGeometry::forRequester(
                    layout,source(profile),q.requested.gpu_adaptive,{0,device});
                EXPECT_EQ(geometry.physical().owners().size(),1u);
                EXPECT_EQ(geometry.physical().owners().front().allocator,(PhysicalMemoryAllocatorIdentity{0,device}));
                EXPECT_EQ(bytes(selected.physical_plan,{0,device},PhysicalMemoryOwner::ImmutableTableStorage),
                    geometry.gpuCacheBytes({0,device}));
            }
            const auto wire=serializeImmutableRowTableAdmissionRequests(std::array{q});
            const auto parsed=test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&profile,projection);
            ASSERT_EQ(parsed.size(),1u);
            ASSERT_TRUE(parsed.front().execution_provision);
            EXPECT_EQ(parsed.front().execution_provision->consumers(),q.execution_provision->consumers());
            const auto restored=ImmutableRowTablePlacementPlanner::select(parsed.front(),frozen);
            EXPECT_EQ(*restored.placement,*selected.placement);
            EXPECT_EQ(restored.physical_plan.summary(),selected.physical_plan.summary());
            if(count>1) {
                auto malformed=q.execution_provision->consumers();
                malformed.back().tp_coordinate=malformed.front().tp_coordinate;
                EXPECT_THROW(test::MemoryPlannerOrderingReplayAccess::restore(profile,q.requested,q.input,malformed,
                    q.execution_provision->executionPlan(),q.execution_provision->requiredBase(),projection,q.retained_host_services),
                    std::invalid_argument);
                malformed=q.execution_provision->consumers();
                malformed.front().shards=static_cast<int>(count+1);
                EXPECT_THROW(test::MemoryPlannerOrderingReplayAccess::restore(profile,q.requested,q.input,malformed,
                    q.execution_provision->executionPlan(),q.execution_provision->requiredBase(),projection,q.retained_host_services),
                    std::invalid_argument);
            }
            if(count>2) {
                RankExecutionPlan native;
                native.rank=0;
                native.hostname="metadata-only";
                native.first_layer=0;
                native.last_layer=47;
                native.runtime.ngram_table_policy=q.requested;
                for(const auto device:devices)
                    native.local_tp_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(device,native.hostname,0));
                native.primary_device=native.local_tp_devices.front();
                EXPECT_EQ(immutableRowTableTPTopology(profile,native),ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal);
                native.has_lm_head=false;
                EXPECT_THROW(immutableRowTableTPTopology(profile,native),std::invalid_argument);
            }
        }
}

namespace {
/** @brief Actual original source constructor for metadata-only shared admission. */
class SharedPlanningWorker final:public llaminar2::testing::MockWorkerGPUContext {
public:
    SharedPlanningWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
};
/** @brief Retire every original recorded source worker before abandoning its Preparing authority. */
struct SharedPlanningSource final {
    std::shared_ptr<NativeContextPreparation> source;
    std::vector<DeviceId> devices;
    /** @brief Bind the actual declared compute/storage endpoints before any source is frozen.
     * @param selected Original physical devices; ordinal equality never aliases different backends. */
    explicit SharedPlanningSource(std::vector<DeviceId> selected)
        :devices(selected){
        using Access=llaminar2::test::NativeContextPreparationReplayAccess;
        std::vector<PhysicalMemoryResource> resources{{0,DeviceId::cpu(),512ull<<30,500ull<<30}};
        for(const auto device:devices){
            Access::seed(device,[device](auto stack){return Access::construct<SharedPlanningWorker>(std::move(stack),device.ordinal);});
            resources.push_back({0,device,512ull<<30,500ull<<30});
        }
        source=Access::source<SharedPlanningWorker>(devices,resources);
        for(const auto device:devices){const auto original=source->acquire(device);EXPECT_EQ(original.memory(),source->memory());}
        source->freezeForPlanning();
    }
    ~SharedPlanningSource(){
        for(const auto device:devices){
            auto reset=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
            llaminar2::test::NativeContextPreparationReplayAccess::physicalReset(reset);
        }
        source->retireUnadmitted();
    }
};
/** @brief Exercise shared-directory pressure through the actual complete planner.
 * @param backend Original coordinator backend; no native runtime is opened.
 * @param count Original storage population, independent of compute/TP membership.
 * @param constrained Whether the last owner has a smaller or disabled grant.
 * @details A real source-owned geometry prices the coordinator ceiling. The
 * initial zero-cache BOM still contains every original weight, graph and source
 * owner. This reproduces greedy admission starving later cards without giving
 * the fixture another capacity ledger or a fabricated planning authority. */
void replayBalancedSharedCache(DeviceType backend,std::size_t count,bool constrained)
{
    std::vector<DeviceId> devices;
    for(std::size_t i=0;i<count;++i)devices.emplace_back(backend,static_cast<int>(i));
    SharedPlanningSource originals(devices);
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource model(file.path());
    const auto& profile=model.metadata().memoryProfile();
    auto requested=policy(0);
    requested.gpu_adaptive.maximum_gpu_cache_rows=0;
    requested.gpu_adaptive.gpu_cache_physical_byte_limit=0;
    requested.gpu_adaptive.placement.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
    requested.gpu_adaptive.placement.coordinator=AdaptiveGPUCacheParticipant{0,backend,0};
    for(const auto device:devices)
        requested.gpu_adaptive.placement.owners.push_back({{0,device.type,device.ordinal},128,1u<<20});
    std::vector<std::size_t> balanced(count,16);
    if(constrained && count>1) {
        balanced.back()=count==2?1:0;
        requested.gpu_adaptive.placement.owners.back().maximum_cache_rows=balanced.back();
        if(!balanced.back())requested.gpu_adaptive.placement.owners.back().cache_physical_byte_limit=0;
    }
    const auto projection=test::NativeOrderingEventGroupReplayAccess::projectionWithRemoteGraphs(devices);
    auto cfg=consumer(backend);
    cfg.native_context_preparation=originals.source;
    cfg.native_ordering_events=projection;
    cfg.ngram_gpu_peer_access=count>1?std::optional{PeerAccessCoverage::None}:std::nullopt;
    cfg.ngram_execution_policy=requested;
    std::vector<DevicePlanConfig> configs{cfg};
    ClusterInventory cluster;cluster.world_size=1;
    RankInventory rank;rank.rank=0;cluster.ranks.push_back(rank);
    (void)MemoryPlanner::declareRetainedHostServices(profile,configs,0,nullptr,cluster,test::planningHostServiceProjection());
    const auto makeRequest=[&] {
        const std::vector<std::vector<DevicePlanConfig>> groups{configs};
        auto result=MemoryPlanner::immutableRowTableAdmissionRequest(profile,requested,groups,{},configs.front().ngram_host_placement);
        if(!result)throw std::logic_error("Shared-cache pressure lost its original source request");
        return std::move(*result);
    };
    const auto initial=makeRequest();
    const auto envelope=AdaptiveImmutableLookupGeometry::fromLayout(
        initial.input.gpuAdaptiveLayout(requested,balanced,0,initial.execution_provision->maximumLiveIds()),
        source(profile),requested.gpu_adaptive);
    const PhysicalMemoryAllocatorIdentity coordinator{0,devices.front()};
    requested.gpu_adaptive.placement.owners.front().cache_physical_byte_limit=envelope.gpuSlotDependentBytes(coordinator);
    configs.front().ngram_execution_policy=requested;
    const auto q=makeRequest();
    const auto floor=MemoryPlanner::plan(profile,configs).physicalPlan();
    const auto selected=ImmutableRowTablePlacementPlanner::select(q,floor);
    EXPECT_TRUE(selected.physical_plan.fits());
    const auto actual=AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),source(profile),requested.gpu_adaptive);
    ASSERT_EQ(actual.physical().owners().size(),count);
    std::vector<std::size_t> admitted;
    for(std::size_t owner=0;owner<count;++owner) {
        const auto& row=actual.physical().owners()[owner];
        if(row.slots<balanced[owner])
            std::cerr<<"Shared cache starvation: owner="<<owner<<" population="<<count
                     <<" actual="<<row.slots<<" feasible_balanced="<<balanced[owner]<<'\n';
        EXPECT_EQ(row.allocator,(PhysicalMemoryAllocatorIdentity{0,devices[owner]}));
        EXPECT_GE(row.slots,balanced[owner])<<"owner="<<owner<<" population="<<count;
        EXPECT_LE(row.slots,requested.gpu_adaptive.placement.owners[owner].maximum_cache_rows);
        EXPECT_LE(actual.gpuSlotDependentBytes(row.allocator),requested.gpu_adaptive.placement.owners[owner].cache_physical_byte_limit);
        admitted.push_back(row.slots);
    }
    // This fixture has ample physical capacity. Any remaining one-row growth
    // must be stopped by an original owner ceiling, including shared metadata.
    for(std::size_t owner=0;owner<count;++owner) {
        if(admitted[owner]==requested.gpu_adaptive.placement.owners[owner].maximum_cache_rows)continue;
        auto more=admitted;++more[owner];
        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(
            q.input.gpuAdaptiveLayout(requested,more,0,q.execution_provision->maximumLiveIds()),source(profile),requested.gpu_adaptive);
        bool exceeds=false;
        for(std::size_t i=0;i<count;++i)
            exceeds|=geometry.gpuSlotDependentBytes({0,devices[i]})>requested.gpu_adaptive.placement.owners[i].cache_physical_byte_limit;
        EXPECT_TRUE(exceeds)<<"Unused feasible GPU row on owner="<<owner;
    }
    const auto wire=serializeImmutableRowTableAdmissionRequests(std::array{q});
    const auto parsed=test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&profile,projection);
    ASSERT_EQ(parsed.size(),1u);
    const auto replay=ImmutableRowTablePlacementPlanner::select(parsed.front(),floor);
    EXPECT_EQ(*replay.placement,*selected.placement);
    EXPECT_EQ(replay.physical_plan.summary(),selected.physical_plan.summary());
    const auto nativeGrowth=qwen4exp::PooledImmutableDirectedTransport::slotDependentBOM(actual.physical(),actual.coordinator(),
        originals.source->originalResource(DeviceId::cpu()),q.input.gpuResources());
    for(const auto& bom:floor.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i) {
        const auto owner=PhysicalMemoryBOM::ownerAt(i);
        if(owner==PhysicalMemoryOwner::ImmutableTableStorage || owner==PhysicalMemoryOwner::ImmutableLookupStaging ||
           owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent || owner==PhysicalMemoryOwner::NativeStorageGraphExecutable)continue;
        const PhysicalMemoryAllocatorIdentity id{bom.resource().world_rank,bom.resource().device};
        const auto growth=owner==PhysicalMemoryOwner::ActivationTransportStaging?bytes(nativeGrowth,id,owner):0u;
        EXPECT_EQ(bytes(selected.physical_plan,id,owner),bom.bytes(owner)+growth)<<toString(owner)<<" on "<<id.device.toString();
    }
}
/** @brief Actual grouped planner keeps original native continuation owners or opposite-vendor storage through selection/wire.
 * @param backend Original homogeneous compute backend.
 * @param cacheRows Declared zero/positive storage-owner cache envelope. */
void replaySharedPlanning(DeviceType backend=DeviceType::ROCm,std::size_t cacheRows=64,
    std::optional<std::uint64_t> localRows=std::nullopt)
{
    const auto devices=backend==DeviceType::CUDA ?
        std::vector{DeviceId::cuda(0),DeviceId::cuda(1),DeviceId::rocm(0)} :
        std::vector{DeviceId::rocm(0),DeviceId::rocm(1)};
    SharedPlanningSource originals(devices);
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource model(file.path());
    const auto& profile=model.metadata().memoryProfile();
    std::vector<DevicePlanConfig> configs{consumer(backend),consumer(backend)};
    auto requested=policy(64);
    requested.gpu_adaptive.maximum_gpu_cache_rows=0;requested.gpu_adaptive.gpu_cache_physical_byte_limit=0;
    requested.gpu_adaptive.placement.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
    requested.gpu_adaptive.placement.coordinator=AdaptiveGPUCacheParticipant{0,devices[1].type,devices[1].ordinal};
    requested.gpu_adaptive.placement.owners.clear();
    if(backend==DeviceType::CUDA)
        requested.gpu_adaptive.placement.owners.push_back({{0,devices[2].type,devices[2].ordinal},cacheRows,1048576});
    else for(const auto device:devices)
        requested.gpu_adaptive.placement.owners.push_back({{0,device.type,device.ordinal},cacheRows,1048576});
    if (localRows) requested.gpu_adaptive.local_hot_cache = AdaptiveGPULocalHotCacheRequest{
        .maximum_cache_rows=*localRows, .cache_physical_byte_limit=1048576};
    const auto projection=test::NativeOrderingEventGroupReplayAccess::projectionWithRemoteGraphs(originals.devices);
    const auto assignment=TensorParallelConfig::equalSplit(2,profile.n_heads,profile.n_kv_heads,
        profile.d_ff,profile.vocab_size,std::vector<DeviceId>{originals.devices[1],originals.devices[0]});
    for(size_t i=0;i<configs.size();++i){
        auto& cfg=configs[i];cfg.device=originals.devices[i];cfg.total_shards=2;cfg.shard_index=int(1-i);
        cfg.ngram_execution_policy=requested;cfg.ngram_tp_topology=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal;
        cfg.local_tp_backend=backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;cfg.tensor_parallel_assignment=assignment.forDevice(cfg.device);
        cfg.native_context_preparation=originals.source;cfg.native_ordering_events=projection;cfg.ngram_gpu_peer_access=backend==DeviceType::CUDA?std::nullopt:std::optional{PeerAccessCoverage::None};
        cfg.ngram_reader_peer_access={{{0,devices[1]},{0,devices[0]},PeerAccessCoverage::None}};
        test::declareRecordedNgramHostServices(cfg);
    }
    ClusterInventory cluster;cluster.world_size=1;
    RankInventory rank;rank.rank=0;cluster.ranks.push_back(rank);
    const auto declarations=MemoryPlanner::declareRetainedHostServices(profile,configs,0,nullptr,cluster,test::planningHostServiceProjection());
    ASSERT_EQ(declarations->instances().size(),1u);
    ASSERT_EQ(configs.front().retained_host_services->instances().size(),1u);
    EXPECT_EQ(configs.front().retained_host_services->instances().front().consumer,devices[1]);
    const std::vector<std::vector<DevicePlanConfig>> groups{configs};
    const auto optional=MemoryPlanner::immutableRowTableAdmissionRequest(profile,requested,groups,{},configs.front().ngram_host_placement);
    ASSERT_TRUE(optional);const auto& q=*optional;
    const auto floor=MemoryPlanner::plan(profile,configs).physicalPlan();
    const auto selected=ImmutableRowTablePlacementPlanner::select(q,floor);
    ASSERT_TRUE(selected.placement->isGPUAdaptive());
    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayoutWithLocalHot(selected.placement->gpuAdaptiveLayout(),
        source(profile),requested.gpu_adaptive,selected.placement->gpuAdaptiveLocalHotCaches());
    if (localRows)
    {
        ASSERT_EQ(selected.placement->gpuAdaptiveLocalHotCaches().size(),configs.size());
        EXPECT_EQ(geometry.maximumIds(),q.execution_provision->maximumLiveIds());
        EXPECT_EQ(geometry.coordinator().parameters().maximumUnique(),q.execution_provision->maximumLiveIds());
        EXPECT_EQ(geometry.coordinator().parameters().maximumOriginal(),configs.size()*q.execution_provision->maximumLiveIds());
        ASSERT_TRUE(geometry.localHotTransport());
        for (std::size_t i=0;i<selected.placement->gpuAdaptiveLocalHotCaches().size();++i)
        {
            const auto& local=selected.placement->gpuAdaptiveLocalHotCaches()[i];
            EXPECT_EQ(local.allocator(),selected.placement->gpuAdaptiveLayout().consumers()[i]);
            EXPECT_EQ(local.slots(),*localRows);
            EXPECT_EQ(local.extent(),selected.placement->extent());
            EXPECT_EQ(local.maximumCompactMissRows(),q.execution_provision->maximumLiveIds());
            EXPECT_LE(local.physicalBytes(),requested.gpu_adaptive.local_hot_cache->cache_physical_byte_limit);
        }
        EXPECT_THROW((void)AdaptiveImmutableLookupGeometry::fromLayout(selected.placement->gpuAdaptiveLayout(),
            source(profile),requested.gpu_adaptive),std::invalid_argument);
        const auto wire=serializeImmutableRowTableAdmissionRequests(std::array{q});
        const auto parsed=test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&profile,projection);
        ASSERT_EQ(parsed.size(),1u);
        const auto replay=ImmutableRowTablePlacementPlanner::select(parsed.front(),floor);
        EXPECT_EQ(*replay.placement,*selected.placement);
        EXPECT_EQ(replay.physical_plan.summary(),selected.physical_plan.summary());
        auto resolved=configs;
        for(auto& cfg:resolved)cfg.ngram_table_placement=selected.placement;
        const auto projected=MemoryPlanner::plan(profile,resolved).physicalPlan();
        const auto missing=projected.requiredFootprintMismatch(selected.physical_plan);
        const auto extra=selected.physical_plan.requiredFootprintMismatch(projected);
        EXPECT_FALSE(missing)<<missing.value_or("");
        EXPECT_FALSE(extra)<<extra.value_or("");
        for(const auto& bom:floor.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);
            if(owner==PhysicalMemoryOwner::ImmutableTableStorage || owner==PhysicalMemoryOwner::ImmutableLookupStaging ||
                owner==PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent || owner==PhysicalMemoryOwner::NativeStorageGraphExecutable ||
                owner==PhysicalMemoryOwner::ActivationTransportStaging)continue;
            EXPECT_EQ(bytes(selected.physical_plan,{bom.resource().world_rank,bom.resource().device},owner),bom.bytes(owner))
                <<toString(owner)<<" on "<<bom.resource().device.toString();
        }
        return;
    }
    ASSERT_TRUE(geometry.readerGeometry());
    EXPECT_EQ(geometry.physical().coordinator().device,devices[1]);
    ASSERT_EQ(geometry.physical().owners().size(),backend==DeviceType::CUDA?1u:2u);
    for(size_t owner=0;owner<geometry.physical().owners().size();++owner){
        EXPECT_EQ(geometry.physical().owners()[owner].allocator.device,devices[backend==DeviceType::CUDA?2:owner]);
        EXPECT_EQ(geometry.physical().owners()[owner].slots,cacheRows);
    }
    ASSERT_TRUE(geometry.readerGeometry()->nativeCollective());
    EXPECT_EQ(geometry.computeReaderCount(),2u);
    if(backend==DeviceType::CUDA){
        EXPECT_FALSE(geometry.physical().peerAccess());
        const auto transport=qwen4exp::PooledImmutableDirectedChannelGeometry::fromGeometry(
            geometry.physical(),geometry.coordinator(),0);
        EXPECT_GT(transport.forwardMemory().mapped_host_bytes,0u);
        EXPECT_GT(transport.reverseMemory().mapped_host_bytes,0u);
    }
    EXPECT_EQ(geometry.host().slots(),64u);
    if(backend==DeviceType::CUDA) EXPECT_EQ(geometry.mandatoryDeviceBytes({0,devices[0]}),0u);
    else EXPECT_GT(geometry.mandatoryDeviceBytes({0,devices[0]}),0u);
    for(const auto& identity:q.execution_provision->consumers()){
        const auto cfg=std::find_if(configs.begin(),configs.end(),[&](const auto& value){return value.device==identity.resource.device;});
        ASSERT_NE(cfg,configs.end());EXPECT_EQ(identity.tp_coordinate,cfg->shard_index);
    }
    const auto wire=serializeImmutableRowTableAdmissionRequests(std::array{q});
    const auto parsed=test::ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&profile,projection);
    ASSERT_EQ(parsed.size(),1u);
    EXPECT_EQ(parsed.front().execution_provision->consumers(),q.execution_provision->consumers());
    const auto restored=ImmutableRowTablePlacementPlanner::select(parsed.front(),floor);
    EXPECT_EQ(*restored.placement,*selected.placement);
    EXPECT_EQ(restored.physical_plan.summary(),selected.physical_plan.summary());
    auto resolved=configs;
    for(auto& cfg:resolved)cfg.ngram_table_placement=selected.placement;
    const auto retained=MemoryPlanner::plan(profile,resolved).physicalPlan();
    EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(retained));
    EXPECT_FALSE(retained.requiredFootprintMismatch(selected.physical_plan));
    // A compute reader that also owns rows retains its original owner frame
    // and native staging. A reader without storage still contributes zero.
    EXPECT_EQ(bytes(retained,{0,devices[0]},PhysicalMemoryOwner::ImmutableLookupStaging),
        geometry.mandatoryDeviceBytes({0,devices[0]}));
    EXPECT_GT(bytes(retained,{0,devices[0]},PhysicalMemoryOwner::ActivationTransportStaging),0u);
    const auto source_view=AdaptiveImmutableLookupGeometry::sourceLayout(selected.placement->gpuAdaptiveLayout(),requested.gpu_adaptive);
    const auto one=AdaptiveImmutableLookupGeometry::fromLayout(source_view,source(profile),requested.gpu_adaptive);
    EXPECT_EQ(bytes(retained,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableTableStorage),one.host().sourceBytes()+one.host().storageBytes());
    EXPECT_EQ(bytes(retained,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging),
        one.host().executionBytes()+qwen4exp::ImmutableLookupService::hostBytes(one.channelLayout()));
    const auto service=declarations->bom({0,devices[1],RetainedHostServiceRole::ImmutableLookupIO},*configs.front().associated_host_memory);
    for(const auto owner:{PhysicalMemoryOwner::RetainedHostServiceStorage,PhysicalMemoryOwner::RetainedHostServiceRuntime})
        EXPECT_EQ(bytes(retained,{0,DeviceId::cpu()},owner),bytes(service,{0,DeviceId::cpu()},owner));
}
}
/** @test Shared directory pressure cannot starve feasible owners; C0/small owners do not limit the others. */
TEST(GPUAdaptiveImmutableLookupAdmission, SharedDirectoryPressureBalancesEveryFeasibleOwner)
{
    /** @brief Restore the public death-test policy after this original-authority replay. */
    struct RestoreDeathStyle {
        std::string previous=GTEST_FLAG_GET(death_test_style);
        ~RestoreDeathStyle(){GTEST_FLAG_SET(death_test_style,previous);}
    } restore;
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
      for(const auto count:{1u,2u,4u,8u})
        for(const bool constrained:{false,true}) {
            SCOPED_TRACE(::testing::Message()<<"backend="<<static_cast<int>(backend)<<" owners="<<count<<" constrained="<<constrained);
            EXPECT_EXIT({replayBalancedSharedCache(backend,count,constrained);
                const auto* result=::testing::UnitTest::GetInstance()->current_test_info()->result();
                for(int part=0;part<result->total_part_count();++part)if(result->GetTestPartResult(part).failed())
                    std::cerr<<result->GetTestPartResult(part).message()<<'\n';
                std::cout.flush();std::cerr.flush();std::_Exit(::testing::Test::HasFailure()?1:0);},::testing::ExitedWithCode(0),"");
        }
}
/** @test Each replay owns its true original process/source/PMA; no authority reset bypass is used. */
TEST(GPUAdaptiveImmutableLookupAdmission, SharedCoordinatorTP2OriginalPMAAndWireReplay)
{
    for(const std::size_t rows:{0u,64u})
        EXPECT_EXIT({replaySharedPlanning(DeviceType::ROCm,rows);std::cout.flush();std::cerr.flush();std::_Exit(::testing::Test::HasFailure()?1:0);},
            ::testing::ExitedWithCode(0),"");
}

/** @test The actual original three-device source funds CUDA NoPeer readers and ROCm storage once across C0/positive wire replay. */
TEST(GPUAdaptiveImmutableLookupAdmission, SharedCoordinatorCUDATP2OriginalPMAAndMappedWireReplay)
{
    for(const std::size_t rows:{0u,64u})
        EXPECT_EXIT({replaySharedPlanning(DeviceType::CUDA,rows);std::cout.flush();std::cerr.flush();std::_Exit(::testing::Test::HasFailure()?1:0);},
            ::testing::ExitedWithCode(0),"");
}

/** @test The actual rank input resolver keeps original communicator ordering and query assignments for ConsumerLocal TP4/8. */
TEST(GPUAdaptiveImmutableLookupAdmission, ConsumerLocalTP4AndTP8RankInputsKeepOriginalAssignments)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource original(file.path());
    const auto& profile=original.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) for(const int count:{4,8}) {
        RankExecutionPlan plan;
        plan.rank=0;
        plan.hostname="consumer-local-fixture";
        plan.first_layer=0;
        plan.last_layer=47;
        plan.runtime.max_seq_len=64;
        plan.runtime.batch_size=1;
        plan.runtime.mtp.enabled=false;
        plan.runtime.mtp.draft_tokens=0;
        plan.runtime.mtp.graph_capacity_draft_tokens=0;
        plan.runtime.ngram_table_policy=policy(64);
        RankInventory inventory;
        inventory.rank=0;
        inventory.cpu_worker_threads=8;
        inventory.cpu_cores=8;
        inventory.cpu_execution=test::kSyntheticCPUExecutionGeometry;
        inventory.cpu.memory_bytes=512ull<<30;
        inventory.cpu.free_memory_bytes=500ull<<30;
        for(int ordinal=count-1;ordinal>=0;--ordinal) {
            plan.local_tp_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId({backend,ordinal},plan.hostname,0));
            DeviceInfo gpu;
            gpu.type=backend;
            gpu.local_device_id=ordinal;
            gpu.memory_bytes=512ull<<30;
            gpu.free_memory_bytes=500ull<<30;
            gpu.compute_units=64;
            inventory.gpus.push_back(gpu);
        }
        plan.primary_device=plan.local_tp_devices.front();
        const auto configs=buildRankMemoryPlanInputs({.model=profile,.plan=plan,.inventory=inventory});
        ASSERT_EQ(configs.size(),static_cast<std::size_t>(count));
        for(std::size_t i=0;i<configs.size();++i) {
            EXPECT_EQ(configs[i].device,DeviceId(backend,count-1-static_cast<int>(i)));
            EXPECT_EQ(configs[i].shard_index,static_cast<int>(i));
            EXPECT_EQ(configs[i].ngram_tp_topology,ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal);
            ASSERT_TRUE(configs[i].tensor_parallel_assignment);
            EXPECT_EQ(configs[i].tensor_parallel_assignment->device,configs[i].device);
            EXPECT_EQ(configs[i].tensor_parallel_assignment->head_start,static_cast<int>(i)*profile.n_heads/count);
        }
        auto partial=plan;partial.local_tp_devices.pop_back();
        EXPECT_THROW(immutableRowTableTPTopology(profile,partial),std::invalid_argument);
        partial=plan;partial.first_layer=1;
        EXPECT_THROW(immutableRowTableTPTopology(profile,partial),std::invalid_argument);
        auto duplicate=plan;duplicate.local_tp_devices[1]=duplicate.local_tp_devices[0];
        EXPECT_THROW(buildRankMemoryPlanInputs({.model=profile,.plan=duplicate,.inventory=inventory}),std::exception);
        auto invalid=plan;
        invalid.local_tp_devices[1]=GlobalDeviceAddress::fromLocalDeviceId(
            backend==DeviceType::CUDA?DeviceId::rocm(0):DeviceId::cuda(0),plan.hostname,0);
        EXPECT_THROW(immutableRowTableTPTopology(profile,invalid),std::invalid_argument);
    }
}

/** @test The actual canonical shared-source planner retains separate local C0/positive geometry and wire identity. */
TEST(GPUAdaptiveImmutableLookupAdmission, LocalHotCombinedCanonicalAdmissionAndReprojection)
{
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
        for(const std::uint64_t local:{0u,64u})for(const std::size_t pool:{0u,64u})
            EXPECT_EXIT({
                const auto* result=::testing::UnitTest::GetInstance()->current_test_info()->result();
                const int first=result->total_part_count();
                replaySharedPlanning(backend,pool,local);
                // GTest suppresses event forwarding in death-test children.
                // Preserve actual predicate diagnostics before process-owned retirement.
                bool failed=false;
                for(int i=first;i<result->total_part_count();++i) {
                    const auto& part=result->GetTestPartResult(i);
                    if(part.failed()) {
                        failed=true;
                        std::cerr<<part.file_name()<<':'<<part.line_number()<<": "<<part.summary()<<'\n';
                    }
                }
                std::cout.flush();std::cerr.flush();
                std::_Exit(failed?1:0);},::testing::ExitedWithCode(0),"");
}

/** @test The real local geometry/BOM preserves bounded native rows and immutable source/consumer identity. */
TEST(GPUAdaptiveImmutableLookupAdmission, LocalHotGeometryOwnsMandatoryFloorBeforeOptionalNativeSlots)
{
    const ImmutableRowTableSource original{.rows=1024,.row_elements=128,.format=GGUFTensorType::Q8_0};
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const auto count:{1u,2u,4u,8u})
    {
        const PhysicalMemoryResource host{0,DeviceId::cpu(),1ull<<30,1ull<<30};
        std::vector<DeviceId> devices;
        for(unsigned ordinal=0;ordinal<count;++ordinal)devices.emplace_back(backend,ordinal);
        const auto projection=test::NativeOrderingEventGroupReplayAccess::projection(devices);
        for(const auto device:devices)
        {
            const PhysicalMemoryResource gpu{0,device,1ull<<30,1ull<<30};
            const auto zero=qwen4exp::PooledImmutableLocalHotGeometry::declare(original,{0,device},32,0,{65536,3,4});
            const auto hot=qwen4exp::PooledImmutableLocalHotGeometry::declare(original,{0,device},32,64,{65536,3,4});
            const auto mandatory=zero.mandatoryBOM(gpu,host,projection);
            EXPECT_EQ(mandatory.summary(),hot.mandatoryBOM(gpu,host,projection).summary());
            EXPECT_EQ(zero.physicalBytes(),0u);
            EXPECT_GT(hot.physicalBytes(),64u*original.rowBytes());
            EXPECT_EQ(hot.maximumCompactMissRows(),32u);
            EXPECT_EQ(hot.extent(),ImmutableRowTableExtent::fromNative(original.format,original.rows,original.rowBytes()));
            PhysicalMemoryPlanBuilder complete(mandatory);
            const auto optional=hot.slotDependentBOM(gpu,host,projection);
            for(const auto& line:optional.resources()) complete.add(line);
            EXPECT_TRUE(complete.build().fits());
        }
    }
}

/** @test Original five policy ratios match total canonical PMA bytes, not optional caps or per-device usage. */
TEST(GPUAdaptiveImmutableLookupAdmission, CanonicalPooledReplayMatchesCompletePhysicalBudgets)
{
    EXPECT_EXIT(([]{
        try {
            SharedPlanningSource originals({DeviceId::rocm(0),DeviceId::rocm(1)});
            test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource model(file.path());
            const auto& profile=model.metadata().memoryProfile();
            namespace q=test::parity::qwen4exp;
            std::vector<ImmutableRowTablePolicy> policies;
            for(const auto& cell:q::qwen4ExpLocalHotBudgetOrdinaryCases())policies.push_back(*cell.ngram_table_policy);
            if(policies.size()!=5)throw std::logic_error("Matched benchmark lost its five original functional profiles");
            const auto projection=test::NativeOrderingEventGroupReplayAccess::projectionWithRemoteGraphs(originals.devices);
            const auto assignment=TensorParallelConfig::equalSplit(2,profile.n_heads,profile.n_kv_heads,
                profile.d_ff,profile.vocab_size,originals.devices);
            const auto resolve=[&](const ImmutableRowTablePolicy& requested) {
                std::vector<DevicePlanConfig> configs{consumer(DeviceType::ROCm),consumer(DeviceType::ROCm)};
                for(std::size_t i=0;i<configs.size();++i) {
                    auto& cfg=configs[i];cfg.device=originals.devices[i];cfg.total_shards=2;cfg.shard_index=int(i);
                    cfg.ngram_execution_policy=requested;cfg.ngram_tp_topology=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal;
                    cfg.local_tp_backend=CollectiveBackendType::RCCL;cfg.tensor_parallel_assignment=assignment.forDevice(cfg.device);
                    cfg.native_context_preparation=originals.source;cfg.native_ordering_events=projection;
                    if(requested.gpu_adaptive.placement.mode==AdaptiveGPUCachePlacementMode::DeclaredRankLocal) {
                        cfg.ngram_gpu_peer_access=PeerAccessCoverage::None;
                        cfg.ngram_reader_peer_access={{{0,DeviceId::rocm(0)},{0,DeviceId::rocm(1)},PeerAccessCoverage::None}};
                    }
                    test::declareRecordedNgramHostServices(cfg);
                }
                ClusterInventory cluster;cluster.world_size=1;RankInventory rank;rank.rank=0;cluster.ranks.push_back(rank);
                (void)MemoryPlanner::declareRetainedHostServices(profile,configs,0,nullptr,cluster,test::planningHostServiceProjection());
                const std::vector<std::vector<DevicePlanConfig>> groups{configs};
                const auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,requested,groups,{},configs.front().ngram_host_placement);
                if(!request)throw std::logic_error("Matched benchmark lost original source provision");
                const auto floor=MemoryPlanner::plan(profile,configs).physicalPlan();
                return ImmutableRowTablePlacementPlanner::select(*request,floor);
            };
            const auto original=policies;
            const auto matched=test::projectPooledNgramReplayBudgets(policies,16384,resolve);
            ASSERT_EQ(matched.size(),5u);EXPECT_EQ(policies,original);
            for(const auto& row:matched) {
                EXPECT_TRUE(row.selected.physical_plan.fits());
                EXPECT_EQ(row.within_alignment_tolerance,row.residual_bytes<=16384u);
                EXPECT_EQ(row.selected.physical_plan.totalBytes()+row.residual_bytes,row.target_bytes);
                EXPECT_EQ(row.target_bytes,matched.front().target_bytes);
                if(row.next_larger)EXPECT_GT(row.next_larger->physical_plan.totalBytes(),row.target_bytes);
                const auto fresh=resolve(row.requested);
                EXPECT_EQ(fresh.physical_plan.summary(),row.selected.physical_plan.summary());
                EXPECT_EQ(fresh.placement->gpuAdaptiveRowElements(),160u);
                EXPECT_EQ(fresh.placement->gpuAdaptiveLayout().extent().rowBytes(),90u);
                OrchestrationConfig declaration;
                declaration.ngram_table_policy=row.requested;
                const auto document=nlohmann::json::parse(serializeOrchestrationConfig(declaration));
                std::cout<<"POOLED_BENCHMARK_POLICY "<<nlohmann::json{
                    {"target_bytes",row.target_bytes},{"physical_bytes",row.selected.physical_plan.totalBytes()},
                    {"residual_bytes",row.residual_bytes},{"within_alignment_tolerance",row.within_alignment_tolerance},
                    {"ngram_table_policy",document.at("configuration").at("ngram_table_policy")},
                    {"complete_plan",row.selected.physical_plan.summary()},
                    {"next_larger_plan",row.next_larger?row.next_larger->physical_plan.summary():std::string{}}}.dump()<<std::endl;
            }
            const std::array same{policies.front(),policies.front()};
            const auto exact=test::projectPooledNgramReplayBudgets(same,0,resolve);
            EXPECT_TRUE(exact.front().within_alignment_tolerance);
            EXPECT_EQ(exact.front().residual_bytes,0u);
            auto zero=policies.front();
            test::pooled_budget_detail::caps(zero,[](auto& cap){cap=0;});
            const std::array erased{zero,zero};
            EXPECT_THROW(test::projectPooledNgramReplayBudgets(erased,16384,resolve),std::invalid_argument);
            const auto changedCeiling=[&](const ImmutableRowTablePolicy& requested) {
                auto result=resolve(requested);
                // A real alternate resource observation is built by the original BOM
                // builder; no live allocator or admission receipt is forged.
                PhysicalMemoryPlanBuilder changed;
                for(const auto& line:result.physical_plan.resources()) {
                    auto resource=line.resource();resource.total_bytes+=4096;
                    PhysicalMemoryBOMBuilder copy(resource);
                    for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i) {
                        const auto owner=PhysicalMemoryBOM::ownerAt(i);copy.add(owner,line.bytes(owner));
                    }
                    changed.add(copy.build());
                }
                if(requested!=policies.front())result.physical_plan=changed.build();
                return result;
            };
            EXPECT_THROW(test::projectPooledNgramReplayBudgets(policies,16384,changedCeiling),std::invalid_argument);
        }catch(const std::exception& error){std::cerr<<error.what()<<'\n';std::_Exit(1);}
        const auto* current=::testing::UnitTest::GetInstance()->current_test_info();
        const bool failed=current && current->result()->Failed();
        if(failed)for(int i=0;i<current->result()->total_part_count();++i){const auto& part=current->result()->GetTestPartResult(i);if(part.failed())std::cerr<<part.summary()<<'\n';}
        std::cerr.flush();std::_Exit(failed?1:0);
    }()),::testing::ExitedWithCode(0),"");
}

/** @test Real source/coordinator and query-only reader each retain exactly one fresh terminal packet beside original source/ready credit. */
TEST(GPUAdaptiveImmutableLookupAdmission, LocalTerminalHostBOMIsOncePerConsumerAcrossResolvedReuse)
{
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        EXPECT_EXIT(([&]{
            try {
                const std::vector devices{DeviceId{backend,0},DeviceId{backend,1}};
                SharedPlanningSource originals(devices);
                test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource model(file.path());
                const auto& profile=model.metadata().memoryProfile();
                const auto projection=test::NativeOrderingEventGroupReplayAccess::projectionWithRemoteGraphs(devices);
                const auto assignment=TensorParallelConfig::equalSplit(2,profile.n_heads,profile.n_kv_heads,
                    profile.d_ff,profile.vocab_size,devices);
                for(unsigned coordinator:{0u,1u})for(bool combined:{false,true}) {
                    auto requested=policy(64);auto& adaptive=requested.gpu_adaptive;
                    adaptive.source_backing=AdaptiveImmutableRowBacking::RAM;adaptive.host={};
                    adaptive.maximum_gpu_cache_rows=0;adaptive.gpu_cache_physical_byte_limit=0;
                    adaptive.placement.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
                    adaptive.placement.coordinator=AdaptiveGPUCacheParticipant{0,backend,static_cast<int>(coordinator)};
                    adaptive.placement.owners={{{0,backend,static_cast<int>(1-coordinator)},64,1048576}};
                    if(combined)adaptive.local_hot_cache=AdaptiveGPULocalHotCacheRequest{};
                    std::vector<DevicePlanConfig> configs{consumer(backend),consumer(backend)};
                    for(unsigned i=0;i<2;++i) {
                        auto& cfg=configs[i];cfg.device=devices[i];cfg.total_shards=2;cfg.shard_index=i;
                        cfg.ngram_execution_policy=requested;cfg.ngram_tp_topology=ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal;
                        cfg.local_tp_backend=backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
                        cfg.tensor_parallel_assignment=assignment.forDevice(cfg.device);
                        cfg.native_context_preparation=originals.source;cfg.native_ordering_events=projection;
                        cfg.ngram_gpu_peer_access=PeerAccessCoverage::None;
                        cfg.ngram_reader_peer_access={{{0,devices[coordinator]},{0,devices[1-coordinator]},PeerAccessCoverage::None}};
                        test::declareRecordedNgramHostServices(cfg);
                    }
                    ClusterInventory cluster;cluster.world_size=1;RankInventory rank;rank.rank=0;cluster.ranks.push_back(rank);
                    (void)MemoryPlanner::declareRetainedHostServices(profile,configs,0,nullptr,cluster,test::planningHostServiceProjection());
                    const std::vector<std::vector<DevicePlanConfig>> groups{configs};
                    const auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,requested,groups,{},configs.front().ngram_host_placement);
                    if(!request)throw std::logic_error("Terminal cost lost its original actual source request");
                    const auto unresolved=MemoryPlanner::plan(profile,configs).physicalPlan();
                    const auto selected=ImmutableRowTablePlacementPlanner::select(*request,unresolved);
                    const auto geometry=AdaptiveImmutableLookupGeometry::fromLayoutWithLocalHot(selected.placement->gpuAdaptiveLayout(),
                        source(profile),adaptive,selected.placement->gpuAdaptiveLocalHotCaches());
                    const auto& host=*configs.front().associated_host_memory;
                    const std::array<PhysicalMemoryResource,2> gpu{{{0,devices[0],configs[0].device_total_bytes,configs[0].device_free_bytes},
                        {0,devices[1],configs[1].device_total_bytes,configs[1].device_free_bytes}}};
                    PhysicalMemoryPlanBuilder original(test::AdaptiveImmutableLookupGeometryReplayAccess::mandatory(geometry,gpu,host,projection));
                    const auto cache=geometry.cacheBOM(gpu,host);
                    for(const auto& line:cache.resources())original.add(line);
                    const auto local_optional=geometry.localHotSlotDependentBOM(gpu,host,projection);
                    for(const auto& line:local_optional.resources())original.add(line);
                    const auto transport=qwen4exp::PooledImmutableRetainedParticipantBudget::completeBOM(geometry,host,gpu,projection);
                    for(const auto& line:transport.resources())original.add(line);
                    const auto original_bom=original.build();
                    const auto source_staging=original_bom.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
                    const auto packets=combined?2*qwen4exp::local_hot::terminalObservationBytes(1,sampling_math::kDeviceGenerationControlCount):0u;
                    auto resolved=configs;
                    for(auto& cfg:resolved)cfg.ngram_table_placement=selected.placement;
                    const auto projected=MemoryPlanner::plan(profile,resolved).physicalPlan();
                    EXPECT_EQ(projected.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),source_staging+packets);
                    EXPECT_FALSE(projected.requiredFootprintMismatch(selected.physical_plan));
                    EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(projected));
                    // Retained fixture inputs describe only the original model
                    // roots. The new runner's pinned terminal packet is not a
                    // source or mapped-ready resident credit.
                    const auto local_floor=geometry.localHotMandatoryBOM(gpu,host,projection);
                    const auto* ready=local_floor.find({0,DeviceId::cpu()});
                    const auto ready_bytes=ready?ready->bytes(PhysicalMemoryOwner::ImmutableLookupStaging):0u;
                    for(auto& cfg:resolved) {
                        const auto* original_gpu=original_bom.find({0,cfg.device});
                        cfg.retained_immutable_table_bytes=original_gpu->bytes(PhysicalMemoryOwner::ImmutableTableStorage);
                        cfg.retained_immutable_lookup_staging_bytes=original_gpu->bytes(PhysicalMemoryOwner::ImmutableLookupStaging);
                        if(cfg.device==devices[coordinator]) {
                            cfg.retained_immutable_source_table_bytes=original_bom.find({0,DeviceId::cpu()})->bytes(PhysicalMemoryOwner::ImmutableTableStorage);
                            cfg.retained_immutable_source_lookup_staging_bytes=source_staging-ready_bytes;
                            cfg.retained_immutable_local_ready_staging_bytes=ready_bytes;
                        }
                    }
                    const auto reused=MemoryPlanner::plan(profile,resolved).physicalPlan();
                    const auto* cpu=reused.find({0,DeviceId::cpu()});
                    EXPECT_EQ(cpu->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),source_staging+packets);
                    EXPECT_EQ(cpu->alreadyResidentBytes(PhysicalMemoryOwner::ImmutableLookupStaging),source_staging);
                    EXPECT_EQ(cpu->charge(PhysicalMemoryOwner::ImmutableLookupStaging).incrementalBytes(),packets);
                }
            }catch(const std::exception& error){std::cerr<<error.what()<<'\n';std::_Exit(1);}
            const auto* current=::testing::UnitTest::GetInstance()->current_test_info();
            const bool failed=current&&current->result()->Failed();
            if(failed)for(int i=0;i<current->result()->total_part_count();++i){const auto& part=current->result()->GetTestPartResult(i);if(part.failed())std::cerr<<part.summary()<<'\n';}
            std::cerr.flush();std::_Exit(failed?1:0);
        }()),::testing::ExitedWithCode(0),"");
    }
}
