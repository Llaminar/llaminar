/**
 * @file Test__NgramExecutionProvision.cpp
 * @brief Real complete-source lookup execution admission before table/quota finalization.
 *
 * Only GGUF metadata and canonical planning functions execute. Sparse native
 * payload extents are never loaded; no device, graph, allocator or model runner
 * is started. The tests compare source-owned BOM lines, not duplicated sizing.
 */
#include "planning/MemoryPlanner.h"
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "planning/RankMemoryPlanInputs.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/CpuAdaptiveImmutableRowCache.h"
#include <limits>
#include <algorithm>
#include <array>
#include "loaders/ModelLoader.h"
#include "models/qwen4exp/Qwen4ExpFixedHeadShards.h"
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "planning/PlanningModelMetadata.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
using namespace llaminar2;
namespace {
/** @return One observed PLE consumer with explicit backend-specific execution and shared host authority. */
DevicePlanConfig consumer(DeviceType backend,ImmutableRowStoragePreference preference)
{
    DevicePlanConfig c;c.world_rank=0;c.device=DeviceId(backend,0);c.device_compute_units=64;
    if(c.device.is_cpu())
    {
        // This metadata fixture authors its participant geometry explicitly;
        // it must not borrow host CPUID or let the planner guess CPU defaults.
        c.device_compute_units=8;c.cpu_execution=test::kSyntheticCPUExecutionGeometry;
    }
    c.device_total_bytes=512ull<<30;c.device_free_bytes=500ull<<30;
    c.associated_host_memory=PhysicalMemoryResource{0,DeviceId::cpu(),512ull<<30,500ull<<30};
    c.max_seq_len=64;c.activation_seq_len=32;c.first_layer=0;c.last_layer=47;
    c.ngram_execution_policy=ImmutableRowTablePolicy{preference,65536,3};return c;
}
/** @return Exact canonical owner line, with absent owners interpreted as zero. */
std::size_t bytes(const PhysicalMemoryPlan &plan,PhysicalMemoryAllocatorIdentity id,PhysicalMemoryOwner owner)
{const auto *bom=plan.find(id);return bom?bom->bytes(owner):0u;}
/** @return Production request preserving the declared exact NUMA policy. */
ImmutableRowTableAdmissionRequest request(const ModelMemoryProfile &profile,const std::vector<DevicePlanConfig> &configs)
{
    const std::vector<std::vector<DevicePlanConfig>> groups{configs};
    auto result=MemoryPlanner::immutableRowTableAdmissionRequest(profile,*configs.front().ngram_execution_policy,
        groups,MemoryPlanner::immutableRowTableUploadBindings(profile,groups),ImmutableRowTableHostPlacement::onNode(3));
    if(!result)throw std::logic_error("Complete source omitted its actual PLE consumer");
    return std::move(*result);
}
}
/** @test Mandatory execution is paid exactly once before lowest-priority hosted table storage. */
TEST(NgramExecutionProvision, CompleteSourceBoundIsPrepricedAndFinalHostedOwnersAreNotDuplicated)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    for(const auto preference:{ImmutableRowStoragePreference::RAM,ImmutableRowStoragePreference::MemoryMapped,
        ImmutableRowStoragePreference::Automatic})
    {
        SCOPED_TRACE(::testing::Message()<<int(backend)<<":"<<int(preference));
        auto c=consumer(backend,preference);auto incremental=c;incremental.ngram_execution_policy.reset();
        const auto old=MemoryPlanner::plan(profile,{incremental}).physicalPlan();
        const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const auto q=request(profile,{c});ASSERT_TRUE(q.execution_provision);
        EXPECT_EQ(q.execution_provision->source(),*profile.flash_next);
        EXPECT_EQ(q.execution_provision->requested(),*c.ngram_execution_policy);
        EXPECT_EQ(q.execution_provision->hostPlacement(),ImmutableRowTableHostPlacement::onNode(3));
        const auto &execution=q.execution_provision->executionPlan();
        EXPECT_GT(bytes(execution,{0,c.device},PhysicalMemoryOwner::ActivationArena),0u);
        EXPECT_GT(bytes(execution,{0,c.device},PhysicalMemoryOwner::NativeGraphExecutable),0u);
        EXPECT_GT(bytes(execution,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging),0u);
        EXPECT_EQ(q.contributions.ram.totalBytes(),0u);EXPECT_EQ(q.contributions.mmap.totalBytes(),0u);
        for(const auto &bom:execution.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);const auto id=PhysicalMemoryAllocatorIdentity{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(frozen,id,owner),bytes(old,id,owner)+bom.bytes(owner));
        }
        const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
        EXPECT_TRUE(selected.physical_plan.fits());
        for(const auto &bom:frozen.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);if(owner==PhysicalMemoryOwner::ImmutableTableStorage)continue;
            const auto id=PhysicalMemoryAllocatorIdentity{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(selected.physical_plan,id,owner),bom.bytes(owner));
        }
        EXPECT_GT(bytes(selected.physical_plan,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableTableStorage),0u);
        auto resolved=c;resolved.ngram_table_placement=selected.placement;
        const auto required=MemoryPlanner::plan(profile,{resolved}).physicalPlan();
        EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(required));
        EXPECT_FALSE(required.requiredFootprintMismatch(selected.physical_plan));
    }
}
/** @test An unpriced/stale/foreign provision cannot hide behind common activation bytes. */
TEST(NgramExecutionProvision, MissingBaseForeignIdentityAndDuplicateExecutionRejectBeforeFinalAdmission)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    auto c=consumer(DeviceType::CUDA,ImmutableRowStoragePreference::RAM);const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
    const auto q=request(profile,{c});auto old=c;old.ngram_execution_policy.reset();
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(q,MemoryPlanner::plan(profile,{old}).physicalPlan()),std::invalid_argument);
    auto changed=q;changed.requested.storage=ImmutableRowStoragePreference::MemoryMapped;
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(changed,frozen),std::invalid_argument);
    changed=q;changed.input=ImmutableRowTablePlacementInput::forConsumers(q.input.extent(),q.input.host(),q.input.consumers(),ImmutableRowTableHostPlacement::taskPolicy());
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(changed,frozen),std::invalid_argument);
    changed=q;auto foreign=std::vector(q.input.consumers().begin(),q.input.consumers().end());foreign.front().device=DeviceId::cuda(1);
    changed.input=ImmutableRowTablePlacementInput::forConsumers(q.input.extent(),q.input.host(),foreign,q.input.hostPlacement());
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(changed,frozen),std::invalid_argument);
    changed=q;const auto extent=ImmutableRowTableExtent::fromNative(q.input.extent().format(),q.input.extent().rows()-1,q.input.extent().rowBytes());
    changed.input=ImmutableRowTablePlacementInput::forConsumers(extent,q.input.host(),q.input.consumers(),q.input.hostPlacement());
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(changed,frozen),std::invalid_argument);
    changed=q;changed.contributions.ram=q.execution_provision->executionPlan();
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(changed,frozen),std::invalid_argument);
    auto smaller=c;smaller.activation_seq_len=1;const auto small=request(profile,{smaller});
    EXPECT_NE(q.execution_provision->consumers(),small.execution_provision->consumers());
    auto missing=c;missing.ngram_execution_policy.reset();const std::vector<std::vector<DevicePlanConfig>> groups{{missing}};
    EXPECT_THROW((void)MemoryPlanner::immutableRowTableAdmissionRequest(profile,*c.ngram_execution_policy,groups,{},ImmutableRowTableHostPlacement::onNode(3)),std::invalid_argument);
}
/** @test Per-consumer execution is additive while one original host table remains rank-owned. */
TEST(NgramExecutionProvision, MultipleConsumersAndTemporalRowsRetainExactSourceOwnedProvision)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    auto a=consumer(DeviceType::CUDA,ImmutableRowStoragePreference::Automatic);auto b=consumer(DeviceType::ROCm,ImmutableRowStoragePreference::Automatic);
    auto q=request(profile,{a,b});ASSERT_TRUE(q.execution_provision);ASSERT_EQ(q.execution_provision->consumers().size(),2u);
    const auto one=request(profile,{a});
    EXPECT_EQ(bytes(q.execution_provision->executionPlan(),{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging),
        2*bytes(one.execution_provision->executionPlan(),{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging));
    const auto frozen=MemoryPlanner::plan(profile,{a,b}).physicalPlan();const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
    ASSERT_EQ(selected.placement->allocators().size(),1u);EXPECT_EQ(selected.placement->consumers().size(),2u);
    EXPECT_EQ(bytes(selected.physical_plan,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableTableStorage),
        ImmutableRowTable::physicalBytesForNativeExtent(q.input.extent().nativeBytes(),selected.placement->storagePolicy()));
}

/** @test Cross-rank metadata never drops prepaid owners or accepts sender-authored source claims. */
TEST(NgramExecutionProvision, WireRoundTripReauthenticatesSourceGeometryAndExactFinalCertificate)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    {
        auto c=consumer(backend,ImmutableRowStoragePreference::MemoryMapped);
        const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();const auto q=request(profile,{c});
        const std::array requests{q};const auto wire=serializeImmutableRowTableAdmissionRequests(requests);
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(wire,0),std::invalid_argument);
        const auto decoded=parseImmutableRowTableAdmissionRequests(wire,0,&profile);ASSERT_EQ(decoded.size(),1u);
        ASSERT_TRUE(decoded.front().execution_provision);
        EXPECT_EQ(decoded.front().execution_provision->consumers(),q.execution_provision->consumers());
        EXPECT_EQ(decoded.front().execution_provision->source(),q.execution_provision->source());
        const auto local=ImmutableRowTablePlacementPlanner::select(q,frozen);
        const auto remote=ImmutableRowTablePlacementPlanner::select(decoded.front(),frozen);
        EXPECT_EQ(*local.placement,*remote.placement);EXPECT_EQ(local.physical_plan.summary(),remote.physical_plan.summary());
        // Required-base is the final framed plan. Corrupt only its GPU arena
        // line: retain all source/geometry/channel metadata and unrelated common
        // arena bytes, but omit the lookup arena contribution. Decode must
        // rederive common+execution and reject this otherwise well-formed frame.
        auto underpriced=wire;const auto &fixed=q.execution_provision->requiredBase();
        constexpr std::size_t record=28u+16u*PhysicalMemoryBOM::ownerCount();
        const auto plan_start=underpriced.size()-(4u+record*fixed.resources().size());
        std::size_t resource_index=0;
        while(resource_index<fixed.resources().size() && fixed.resources()[resource_index].resource().device!=c.device)++resource_index;
        ASSERT_LT(resource_index,fixed.resources().size());
        std::size_t owner_index=0;
        while(PhysicalMemoryBOM::ownerAt(owner_index)!=PhysicalMemoryOwner::ActivationArena)++owner_index;
        const auto full_arena=fixed.resources()[resource_index].bytes(PhysicalMemoryOwner::ActivationArena);
        const auto lookup_arena=bytes(q.execution_provision->executionPlan(),{0,c.device},PhysicalMemoryOwner::ActivationArena);
        ASSERT_GT(full_arena,lookup_arena);const auto common_arena=full_arena-lookup_arena;
        EXPECT_GT(common_arena,lookup_arena); // An execution-only coverage test would wrongly pass.
        const auto offset=plan_start+4u+resource_index*record+28u+owner_index*16u;
        ASSERT_LE(offset+8u,underpriced.size());
        for(std::size_t b=0;b<8u;++b)underpriced[offset+b]=static_cast<std::uint8_t>(static_cast<std::uint64_t>(common_arena)>>(8u*b));
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(underpriced,0,&profile),std::invalid_argument);
        auto foreign=profile;--foreign.flash_next->ple_table_rows;
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(wire,0,&foreign),std::invalid_argument);
        foreign=profile;++foreign.flash_next->residual_gate_rank;
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(wire,0,&foreign),std::invalid_argument);
        auto shrunken=q.execution_provision->consumers();shrunken.front().activation.local_vocab=1;
        auto unpriced=c;unpriced.ngram_execution_policy.reset();
        EXPECT_THROW((void)MemoryPlanner::restoreImmutableRowTableExecutionProvision(profile,q.requested,q.input,
            shrunken,q.execution_provision->executionPlan(),MemoryPlanner::plan(profile,{unpriced}).physicalPlan()),std::invalid_argument);
        auto geometry=q.execution_provision->consumers();geometry.front().rows=1;
        EXPECT_THROW((void)MemoryPlanner::restoreImmutableRowTableExecutionProvision(profile,q.requested,q.input,
            geometry,q.execution_provision->executionPlan(),q.execution_provision->requiredBase()),std::invalid_argument);
        EXPECT_THROW((void)MemoryPlanner::restoreImmutableRowTableExecutionProvision(profile,q.requested,q.input,
            q.execution_provision->consumers(),{},q.execution_provision->requiredBase()),std::invalid_argument);
        EXPECT_THROW((void)MemoryPlanner::restoreImmutableRowTableExecutionProvision(profile,q.requested,q.input,
            q.execution_provision->consumers(),q.execution_provision->executionPlan(),{}),std::invalid_argument);
        EXPECT_THROW((void)MemoryPlanner::restoreImmutableRowTableExecutionProvision(profile,q.requested,q.input,
            q.execution_provision->consumers(),q.execution_provision->executionPlan(),q.execution_provision->executionPlan()),std::invalid_argument);
        auto old=c;old.ngram_execution_policy.reset();
        EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(decoded.front(),MemoryPlanner::plan(profile,{old}).physicalPlan()),std::invalid_argument);
    }
}

/** @test Concrete resident storage needs fit the frozen Auto envelope without hosted execution owners. */
TEST(NgramExecutionProvision, ResolvedVRAMFootprintFitsAutoEnvelopeAndHasNoHostedLookupOwners)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    {
        auto c=consumer(backend,ImmutableRowStoragePreference::Automatic);
        const auto original=resolveGPUWeightLoadMemoryGeometry(maximumGGUFTensorPayloadBytes(source.loader().getModel()),
            {.staging_stream_count=1,.staging_budget_bytes=1u<<20});
        c.ordinary_weight_load_geometry=original;c.weight_load_staging={original.staging_bytes,original.host_staging_bytes};
        const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();const auto q=request(profile,{c});
        ASSERT_TRUE(q.contributions.vram);const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
        ASSERT_EQ(selected.placement->storage(),ImmutableRowStorage::VRAM);
        auto resolved=c;resolved.ngram_table_placement=selected.placement;
        const auto actual=MemoryPlanner::plan(profile,{resolved}).physicalPlan();
        const auto mismatch=selected.physical_plan.requiredFootprintMismatch(actual);
        EXPECT_FALSE(mismatch)<<mismatch.value_or("");
        EXPECT_EQ(bytes(actual,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableLookupStaging),0u);
        EXPECT_LT(bytes(actual,{0,c.device},PhysicalMemoryOwner::ActivationArena),bytes(frozen,{0,c.device},PhysicalMemoryOwner::ActivationArena));
        EXPECT_LT(bytes(actual,{0,c.device},PhysicalMemoryOwner::NativeGraphExecutable),bytes(frozen,{0,c.device},PhysicalMemoryOwner::NativeGraphExecutable));
        EXPECT_EQ(bytes(actual,{0,c.device},PhysicalMemoryOwner::ImmutableTableStorage),q.input.extent().nativeBytes());
        // This checks declared actual-mode requirements, not materialization;
        // the existing native resident-owner/stage suite proves allocation behavior.
    }
}

namespace {
/** @return An explicit host-owned cache request; original ceilings remain separate from admitted slots. */
DevicePlanConfig adaptiveConsumer(DeviceType backend,std::uint64_t slots=7,std::size_t ceiling=1u<<20)
{
    auto c=consumer(backend,ImmutableRowStoragePreference::HostAdaptiveRAMSSD);
    c.ngram_host_placement=ImmutableRowTableHostPlacement::onNode(3);
    c.ngram_execution_policy->host_adaptive={slots,ceiling,{}};
    return c;
}
/** @return Source-owned geometry for an already sealed adaptive placement, with no duplicate byte formulas. */
CpuAdaptiveImmutableRowCacheGeometry adaptiveGeometry(const ResolvedImmutableRowTablePlacement &p)
{
    return CpuAdaptiveImmutableRowCacheGeometry::fromLayout(p.hostAdaptiveLayout(),p.allocators().front(),
        p.hostAdaptiveMaximumLiveRows(),p.hostAdaptivePolicy());
}
}
/** @test A single checked source projection owns ordinary, verifier and retained-ID bounds and overflow. */
TEST(NgramExecutionProvision, AdaptiveMaximumLiveIdsUsesEveryTemporalEnvelope)
{
    EXPECT_EQ(immutableRowTableMaximumLiveIds(9,2,16,1,4),288u);
    EXPECT_EQ(immutableRowTableMaximumLiveIds(1,2,16,15,4),480u);
    EXPECT_EQ(immutableRowTableMaximumLiveIds(1,2,16,4,31),992u);
    EXPECT_THROW((void)immutableRowTableMaximumLiveIds(0,1,16,1,1),std::invalid_argument);
    EXPECT_THROW((void)immutableRowTableMaximumLiveIds(1,0,16,1,1),std::invalid_argument);
    EXPECT_THROW((void)immutableRowTableMaximumLiveIds(1,1,0,1,1),std::invalid_argument);
    EXPECT_THROW((void)immutableRowTableMaximumLiveIds(1,1,16,-1,1),std::invalid_argument);
    EXPECT_THROW((void)immutableRowTableMaximumLiveIds(2,2,std::numeric_limits<std::size_t>::max(),1,1),std::overflow_error);
}
/** @test Zero/full explicit ceilings seal truthful SSD/cache modes without changing frozen owner lines. */
TEST(NgramExecutionProvision, AdaptiveZeroAndFullSlotsMatchExactSourceBOM)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    for(const auto backend:{DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm})
    for(const std::uint64_t limit:{0u,1u,7u,17u})
    {
        auto c=adaptiveConsumer(backend,limit,limit?1u<<20:0u);
        const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();const auto q=request(profile,{c});
        const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);const auto &p=*selected.placement;
        ASSERT_TRUE(p.isHostAdaptive());EXPECT_EQ(p.hostAdaptiveRequest(),c.ngram_execution_policy->host_adaptive);
        EXPECT_EQ(p.hostPlacement(),c.ngram_host_placement);EXPECT_EQ(p.hostAdaptiveLayout().owners().front().cache_rows,limit);
        EXPECT_EQ(p.lookupForConsumer({0,c.device}),ImmutableRowLookupExecution::HostProvider);
        EXPECT_THROW((void)p.storage(),std::logic_error);
        EXPECT_THROW((void)p.storageForConsumer({0,c.device}),std::logic_error);
        const auto g=adaptiveGeometry(p);const auto host=PhysicalMemoryAllocatorIdentity{0,DeviceId::cpu()};
        EXPECT_EQ(bytes(selected.physical_plan,host,PhysicalMemoryOwner::ImmutableTableStorage),g.sourceBytes()+g.storageBytes());
        EXPECT_EQ(bytes(selected.physical_plan,host,PhysicalMemoryOwner::ImmutableLookupStaging),
            bytes(frozen,host,PhysicalMemoryOwner::ImmutableLookupStaging));
        for(const auto &bom:frozen.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
        {
            const auto owner=PhysicalMemoryBOM::ownerAt(i);if(owner==PhysicalMemoryOwner::ImmutableTableStorage)continue;
            const auto id=PhysicalMemoryAllocatorIdentity{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(selected.physical_plan,id,owner),bom.bytes(owner));
        }
        auto resolved=c;resolved.ngram_table_placement=selected.placement;
        const auto actual=MemoryPlanner::plan(profile,{resolved}).physicalPlan();
        EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(actual));
        EXPECT_FALSE(actual.requiredFootprintMismatch(selected.physical_plan));
    }
}
/** @test The exact physical-byte ceiling crosses source-owned alignment/index jumps, never a payload-only estimate. */
TEST(NgramExecutionProvision, AdaptiveByteCeilingSelectsMaximumFittingSourceGeometry)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    auto c=adaptiveConsumer(DeviceType::CUDA,31);auto q=request(profile,{c});
    const auto live=q.execution_provision->maximumLiveIds();
    const auto geometry=[&](std::size_t n){return CpuAdaptiveImmutableRowCacheGeometry::fromLayout(
        q.input.hostAdaptiveLayout(*c.ngram_execution_policy,n,live),{0,DeviceId::cpu()},live,c.ngram_execution_policy->host_adaptive.policy);};
    for(const std::size_t probe:{1u,2u,7u,16u,31u})
    for(const bool below:{false,true})
    {
        const auto exact=geometry(probe).storageBytes();ASSERT_GT(exact,0u);
        c.ngram_execution_policy->host_adaptive.cache_physical_byte_limit=exact-(below?1u:0u);
        q=request(profile,{c});const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
        std::size_t expected=0;
        for(std::size_t n=1;n<=31;++n)if(geometry(n).storageBytes()<=c.ngram_execution_policy->host_adaptive.cache_physical_byte_limit)expected=n;
        EXPECT_EQ(selected.placement->hostAdaptiveLayout().owners().front().cache_rows,expected);
        EXPECT_TRUE(selected.physical_plan.fits());
    }
}
/** @test Adaptive final merge authenticates even zero-byte GPU resource observations, as whole-tier selection does. */
TEST(NgramExecutionProvision, AdaptiveConflictingFreeObservationCannotEvadeFinalMerge)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    auto c=adaptiveConsumer(DeviceType::CUDA,0,0);const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
    auto q=request(profile,{c});auto changed=std::vector(q.input.consumers().begin(),q.input.consumers().end());
    --changed.front().admission_available_bytes;
    q.input=ImmutableRowTablePlacementInput::forConsumers(q.input.extent(),q.input.host(),changed,q.input.hostPlacement());
    // Required footprint equality intentionally ignores free observations. The canonical builder merge must not.
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(q,frozen),std::invalid_argument);
}
/** @test One host provider cannot hide conflicting immutable requests or layouts behind first-owner deduplication. */
TEST(NgramExecutionProvision, AdaptiveRetainedRequestsAndSharedResolvedOwnersStayExact)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    auto c=adaptiveConsumer(DeviceType::CUDA,7,0);const auto q=request(profile,{c});
    const auto selected=ImmutableRowTablePlacementPlanner::select(q,MemoryPlanner::plan(profile,{c}).physicalPlan());
    EXPECT_EQ(selected.placement->hostAdaptiveLayout().owners().front().cache_rows,0u);
    auto changed=*c.ngram_execution_policy;++changed.host_adaptive.maximum_cache_rows;
    auto other=c;other.ngram_execution_policy=changed;
    const auto next=ImmutableRowTablePlacementPlanner::select(request(profile,{other}),MemoryPlanner::plan(profile,{other}).physicalPlan());
    EXPECT_EQ(next.placement->hostAdaptiveLayout().owners().front().cache_rows,0u);
    EXPECT_THROW(selected.placement->requireRequestedPolicy(changed),std::invalid_argument);
    EXPECT_NO_THROW(selected.placement->requireRequestedPolicy(*c.ngram_execution_policy));
    c.ngram_table_placement=selected.placement;other.ngram_table_placement=next.placement;
    {const std::vector<std::vector<DevicePlanConfig>> groups{{c},{other}};
        EXPECT_THROW((void)MemoryPlanner::planGroups(profile,groups),std::invalid_argument);}
    other=c;++other.activation_seq_len;
    {const std::vector<std::vector<DevicePlanConfig>> groups{{c},{other}};
        EXPECT_THROW((void)MemoryPlanner::planGroups(profile,groups),std::invalid_argument);}
    other=c;other.ngram_host_placement=ImmutableRowTableHostPlacement::onNode(4);
    {const std::vector<std::vector<DevicePlanConfig>> groups{{c},{other}};
        EXPECT_THROW((void)MemoryPlanner::planGroups(profile,groups),std::invalid_argument);}
}
/** @test Adaptive schema preserves original ceilings/locality and rederives mandatory execution from canonical geometry. */
TEST(NgramExecutionProvision, AdaptiveWireRoundTripRetainsZeroFullAndRejectsUnpricedMutation)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    for(const auto slots:{0u,17u})
    {
        auto c=adaptiveConsumer(DeviceType::CUDA,slots,slots?1u<<20:0u);c.ngram_execution_policy->host_adaptive.policy.aging_accesses=37;
        const auto q=request(profile,{c});const auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();
        const std::array values{q};const auto wire=serializeImmutableRowTableAdmissionRequests(values);
        const auto decoded=parseImmutableRowTableAdmissionRequests(wire,0,&profile);ASSERT_EQ(decoded.size(),1u);
        EXPECT_EQ(decoded.front().requested,q.requested);EXPECT_EQ(decoded.front().input.hostPlacement(),c.ngram_host_placement);
        EXPECT_EQ(decoded.front().execution_provision->maximumLiveIds(),q.execution_provision->maximumLiveIds());
        const auto local=ImmutableRowTablePlacementPlanner::select(q,frozen);
        const auto remote=ImmutableRowTablePlacementPlanner::select(decoded.front(),frozen);
        EXPECT_EQ(*local.placement,*remote.placement);EXPECT_EQ(local.physical_plan.summary(),remote.physical_plan.summary());
        for(const auto version:{1u,2u,3u,4u}){auto bad=wire;bad[4]=version;EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bad,0,&profile),std::invalid_argument);}
        for(std::size_t length=0;length<wire.size();++length)
            EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(std::span(wire).first(length),0,&profile),std::invalid_argument)<<length;
        auto bad=wire;bad.push_back(0);EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bad,0,&profile),std::invalid_argument);
        bad=wire;bad[16]=static_cast<std::uint8_t>(ImmutableRowStoragePreference::RAM);
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bad,0,&profile),std::invalid_argument);
        // Byte56 is aging_accesses in schema5: zero rejects before any provision is restored.
        bad=wire;for(std::size_t i=56;i<64;++i)bad[i]=0;
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(bad,0,&profile),std::invalid_argument);
        auto geometry=q.execution_provision->consumers();++geometry.front().rows;
        EXPECT_THROW((void)MemoryPlanner::restoreImmutableRowTableExecutionProvision(profile,q.requested,q.input,
            geometry,q.execution_provision->executionPlan(),q.execution_provision->requiredBase()),std::invalid_argument);
    }
}

/** @test Optional cache slots consume only complete-PMA residual room after every frozen higher-priority owner. */
TEST(NgramExecutionProvision, AdaptiveFinalSlotsUseCanonicalFitsAfterAllFixedOwners)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    auto c=adaptiveConsumer(DeviceType::CUDA,31);auto frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();auto q=request(profile,{c});
    const auto maximum=q.execution_provision->maximumLiveIds();
    const auto g=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(q.input.hostAdaptiveLayout(q.requested,7,maximum),
        {0,DeviceId::cpu()},maximum,q.requested.host_adaptive.policy);
    const auto *host=frozen.find({0,DeviceId::cpu()});ASSERT_NE(host,nullptr);
    // Author a fresh observed budget at one exact source-geometry boundary.
    // Production selection still delegates every candidate to complete PMA fits.
    c.associated_host_memory->admission_available_bytes=host->incrementalBytes()+g.storageBytes();
    frozen=MemoryPlanner::plan(profile,{c}).physicalPlan();q=request(profile,{c});
    const auto selected=ImmutableRowTablePlacementPlanner::select(q,frozen);
    std::size_t expected=0;
    for(std::size_t n=1;n<=31;++n)
    {
        const auto shape=CpuAdaptiveImmutableRowCacheGeometry::fromLayout(q.input.hostAdaptiveLayout(q.requested,n,maximum),
            {0,DeviceId::cpu()},maximum,q.requested.host_adaptive.policy);
        PhysicalMemoryPlanBuilder complete(frozen);const auto cache=shape.cacheBOM(q.input.host());
        for(const auto &line:cache.resources())complete.add(line);
        if(complete.build().fits())expected=n;
    }
    EXPECT_EQ(selected.placement->hostAdaptiveLayout().owners().front().cache_rows,expected);
    EXPECT_LT(expected,31u);EXPECT_GE(expected,7u);
    EXPECT_EQ(bytes(selected.physical_plan,{0,c.device},PhysicalMemoryOwner::RoutedExpertWeights),
        bytes(frozen,{0,c.device},PhysicalMemoryOwner::RoutedExpertWeights));
}

/** @test Native source shards price each actual local range once and reject full-copy retained credits. */
TEST(NgramExecutionProvision, FixedHeadOwnersPriceExactRangesAndRetainedNativeCredits)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();const Qwen4ExpConfig config(source.loader().getModel());
    const auto &g=config.ngrams();
    const auto extent=ImmutableRowTableExtent::fromNative(g.table.type,g.table.dimensions[0],g.row_bytes);
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8})
    {
        std::vector<PhysicalMemoryAllocatorIdentity> owners;std::vector<PhysicalMemoryResource> resources;
        std::vector<DevicePlanConfig> configs;
        for(int n=0;n<degree;++n)
        {
            auto c=consumer(backend,ImmutableRowStoragePreference::VRAM);c.device=DeviceId(backend,n);
            // Direct declaration tests the materialization component separately
            // from the explicit production-request tests below; no Auto entry.
            c.ngram_execution_policy.reset();configs.push_back(c);owners.push_back({0,c.device});
            resources.push_back({0,c.device,c.device_total_bytes,c.device_free_bytes});
        }
        const auto placement=Qwen4ExpFixedHeadShards::declare(extent,g.head_offsets,g.head_vocab_sizes,owners);
        const auto sourceBOM=placement->fixedHeadShardStorageBOM(resources);
        for(auto &c:configs)c.ngram_table_placement=placement;
        const auto complete=MemoryPlanner::plan(profile,configs).physicalPlan();
        std::size_t sum=0;
        for(auto &c:configs)
        {
            const PhysicalMemoryAllocatorIdentity owner{0,c.device};const auto *actual=complete.find(owner);ASSERT_TRUE(actual);
            const auto expected=placement->fixedHeadSourceRangeFor(owner).nativeBytes();sum+=expected;
            EXPECT_EQ(actual->bytes(PhysicalMemoryOwner::ImmutableTableStorage),sourceBOM.find(owner)->bytes(PhysicalMemoryOwner::ImmutableTableStorage));
            EXPECT_EQ(actual->bytes(PhysicalMemoryOwner::ImmutableTableStorage),expected);
            EXPECT_EQ(actual->bytes(PhysicalMemoryOwner::ImmutableLookupStaging),0u);
            c.retained_immutable_table_bytes=expected;
        }
        EXPECT_EQ(sum,extent.nativeBytes());
        EXPECT_EQ(bytes(complete,{0,DeviceId::cpu()},PhysicalMemoryOwner::ImmutableTableStorage),0u);
        const auto retained=MemoryPlanner::plan(profile,configs).physicalPlan();
        for(const auto &c:configs)
        {
            const auto *actual=retained.find({0,c.device});ASSERT_TRUE(actual);
            EXPECT_EQ(actual->charge(PhysicalMemoryOwner::ImmutableTableStorage).already_resident_bytes,
                placement->fixedHeadSourceRangeFor({0,c.device}).nativeBytes());
        }
        auto full_copy=configs;full_copy.front().retained_immutable_table_bytes=extent.nativeBytes();
        EXPECT_THROW((void)MemoryPlanner::plan(profile,full_copy),std::invalid_argument);
        auto foreign=configs;foreign.front().device=DeviceId(backend,99);
        EXPECT_THROW((void)MemoryPlanner::plan(profile,foreign),std::invalid_argument);
    }
}

namespace {
/** @return Explicit native TP consumers in declaration order, including reversed physical order. */
std::vector<DevicePlanConfig> fixedHeadConsumers(const ModelMemoryProfile &profile,DeviceType backend,
    int degree,bool reversed,const GPUWeightLoadMemoryGeometry &upload)
{
    RankExecutionPlan plan;plan.rank=0;plan.hostname="head-fixture";plan.first_layer=0;plan.last_layer=47;
    plan.runtime.max_seq_len=64;plan.runtime.batch_size=1;
    plan.runtime.mtp.enabled=false;plan.runtime.mtp.draft_tokens=0;plan.runtime.mtp.graph_capacity_draft_tokens=0;
    plan.runtime.ngram_table_policy={ImmutableRowStoragePreference::VRAM};
    plan.runtime.ngram_table_policy.distribution=ImmutableRowTableDistribution::FixedHeadSharded;
    RankInventory inventory;inventory.rank=0;inventory.cpu_worker_threads=8;inventory.cpu_cores=8;
    inventory.cpu_execution=test::kSyntheticCPUExecutionGeometry;
    inventory.cpu.memory_bytes=512ull<<30;inventory.cpu.free_memory_bytes=500ull<<30;
    for(int p=0;p<degree;++p)
    {
        const int ordinal=reversed?degree-1-p:p;
        plan.local_tp_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(DeviceId(backend,ordinal),plan.hostname,0));
        DeviceInfo gpu;gpu.type=backend;gpu.local_device_id=ordinal;
        gpu.memory_bytes=512ull<<30;gpu.free_memory_bytes=500ull<<30;gpu.compute_units=64;
        inventory.gpus.push_back(gpu);
    }
    plan.primary_device=plan.local_tp_devices.front();
    auto result=buildRankMemoryPlanInputs({.model=profile,.plan=plan,.inventory=inventory,
        .weight_load_geometry=upload,.snapshot_capacity=std::nullopt,.captured_prefill_buckets=std::nullopt});
    for(auto &c:result)c.activation_seq_len=32; // Explicit frozen test row envelope; never chosen by table admission.
    return result;
}
}
/** @test Final selection preserves source-order coordinates and exact unequal range payload, never replicated table bytes. */
TEST(NgramExecutionProvision, FixedHeadRequestSelectsOriginalRangesInActualCommunicatorOrder)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    const auto &profile=source.metadata().memoryProfile();
    const auto serialized_profile=profile.serialize();
    const auto restored=ModelMemoryProfile::deserialize(serialized_profile.data(),serialized_profile.size());
    EXPECT_EQ(restored.flash_next,profile.flash_next);
    auto old_profile=serialized_profile;old_profile[4]=5;
    EXPECT_THROW((void)ModelMemoryProfile::deserialize(old_profile.data(),old_profile.size()),std::runtime_error);
    ASSERT_EQ(profile.flash_next->ple_head_offsets.size(),16u);
    const auto upload=resolveGPUWeightLoadMemoryGeometry(maximumGGUFTensorPayloadBytes(source.loader().getModel()),{});
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8})for(const bool reverse:{false,true})
    {
        auto configs=fixedHeadConsumers(profile,backend,degree,reverse,upload);
        const auto frozen=MemoryPlanner::plan(profile,configs).physicalPlan();const auto q=request(profile,configs);
        const auto chosen=ImmutableRowTablePlacementPlanner::select(q,frozen);
        ASSERT_TRUE(chosen.placement->isFixedHeadShards());EXPECT_TRUE(chosen.physical_plan.fits());
        std::size_t total=0;
        for(int p=0;p<degree;++p)
        {
            const auto &c=configs[p];const PhysicalMemoryAllocatorIdentity owner{0,c.device};
            const auto a=Qwen4ExpFixedHeadShards::assignmentFor(*chosen.placement,owner);
            EXPECT_EQ(a.participant,static_cast<std::size_t>(p));
            EXPECT_EQ(a.first_head,static_cast<std::size_t>(p*16/degree));
            const auto begin=p?profile.flash_next->ple_head_offsets[p*16/degree]:0;
            const auto end=p+1==degree?profile.flash_next->ple_table_rows:profile.flash_next->ple_head_offsets[(p+1)*16/degree];
            EXPECT_EQ(a.source.firstRow(),begin);EXPECT_EQ(a.source.rows(),end-begin);
            const auto native=chosen.placement->fixedHeadSourceRangeFor(owner).nativeBytes();total+=native;
            EXPECT_EQ(bytes(chosen.physical_plan,owner,PhysicalMemoryOwner::ImmutableTableStorage),native);
            // Only payload extends the frozen source-owned execution/weight/KV plan.
            for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i)
            {
                const auto o=PhysicalMemoryBOM::ownerAt(i);if(o==PhysicalMemoryOwner::ImmutableTableStorage)continue;
                EXPECT_EQ(bytes(chosen.physical_plan,owner,o),bytes(frozen,owner,o));
            }
        }
        EXPECT_EQ(total,q.input.extent().nativeBytes());
        EXPECT_NO_THROW(chosen.placement->requireRequestedPolicy(q.requested));
        auto replicated=q.requested;replicated.distribution=ImmutableRowTableDistribution::Replicated;
        EXPECT_THROW(chosen.placement->requireRequestedPolicy(replicated),std::invalid_argument);
        const std::array requests{q};const auto wire=serializeImmutableRowTableAdmissionRequests(requests);
        auto old_wire=wire;old_wire[4]=4;
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(old_wire,0,&profile),std::invalid_argument);
        auto wrong_policy=wire;wrong_policy[20]=0; // Schema5 distribution word; replicated cannot carry fixed provenance.
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(wrong_policy,0,&profile),std::invalid_argument);
        const auto parsed=parseImmutableRowTableAdmissionRequests(wire,0,&profile);ASSERT_EQ(parsed.size(),1u);
        EXPECT_EQ(parsed.front().requested,q.requested);
        EXPECT_EQ(parsed.front().execution_provision->consumers(),q.execution_provision->consumers());
        EXPECT_EQ(*ImmutableRowTablePlacementPlanner::select(parsed.front(),frozen).placement,*chosen.placement);
        auto foreign=profile;++foreign.flash_next->ple_head_offsets[1];
        EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(wire,0,&foreign),std::invalid_argument);
    }
}
/** @test Complete PMA fit alone owns the final fixed table decision; a failed payload cannot shrink higher-priority rows. */
TEST(NgramExecutionProvision, FixedHeadExactFitAndOneByteShortKeepFrozenRowsAndOwners)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    const auto upload=resolveGPUWeightLoadMemoryGeometry(maximumGGUFTensorPayloadBytes(source.loader().getModel()),{});
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})
    {
        auto initial=fixedHeadConsumers(profile,backend,2,true,upload);
        const auto first=request(profile,initial);const auto full=ImmutableRowTablePlacementPlanner::select(first,MemoryPlanner::plan(profile,initial).physicalPlan());
        for(const bool short_by_one:{false,true})
        {
            auto c=initial;
            for(std::size_t p=0;p<c.size();++p)
            {
                const auto *bom=full.physical_plan.find({0,c[p].device});ASSERT_TRUE(bom);
                c[p].device_free_bytes=bom->totalBytes()-(short_by_one && p==0?1:0);
            }
            const auto before=c;const auto q=request(profile,c);const auto frozen=MemoryPlanner::plan(profile,c).physicalPlan();
            ASSERT_TRUE(frozen.fits());
            if(short_by_one)
            {
                EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(q,frozen),PhysicalMemoryCapacityExhausted);
            }
            else EXPECT_TRUE(ImmutableRowTablePlacementPlanner::select(q,frozen).physical_plan.fits());
            for(std::size_t p=0;p<c.size();++p)
            {
                EXPECT_EQ(c[p].activation_seq_len,before[p].activation_seq_len);
                EXPECT_EQ(c[p].device_free_bytes,before[p].device_free_bytes);
                EXPECT_EQ(c[p].total_shards,before[p].total_shards);
            }
        }
    }
}
/** @test Missing original source/TP/upload authority and conflicting temporal coordinates fail closed. */
TEST(NgramExecutionProvision, FixedHeadRejectsForeignTopologyCoordinatesSourceAndUpload)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto &profile=source.metadata().memoryProfile();
    const auto upload=resolveGPUWeightLoadMemoryGeometry(maximumGGUFTensorPayloadBytes(source.loader().getModel()),{});
    const auto good=fixedHeadConsumers(profile,DeviceType::CUDA,2,false,upload);
    for(int failure=0;failure<8;++failure)
    {
        auto c=good;
        switch(failure)
        {
        case 0:c[0].ngram_tp_topology=ImmutableRowTableTPTopology::None;break;
        case 1:c[0].first_layer=1;break;
        case 2:c[0].device=DeviceId::cpu();break;
        case 3:c[0].local_tp_backend=CollectiveBackendType::HOST;break;
        case 4:c[0].shard_index=1;break;
        case 5:c[0].device=DeviceId::rocm(0);break;
        case 6:c[0].batch_size=2;break;
        case 7:c[0].ngram_execution_policy->storage=ImmutableRowStoragePreference::Automatic;break;
        }
        EXPECT_THROW((void)request(profile,c),std::invalid_argument)<<failure;
    }
    auto missing=good;for(auto &c:missing){c.ordinary_weight_load_geometry.reset();c.weight_load_staging={};}
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(request(profile,missing),MemoryPlanner::plan(profile,missing).physicalPlan()),std::invalid_argument);
    auto source_missing=profile;source_missing.flash_next->ple_head_offsets.clear();
    EXPECT_THROW((void)MemoryPlanner::plan(source_missing,good),std::invalid_argument);
    auto temporal=good;auto conflict=good.front();conflict.shard_index=1;temporal.push_back(conflict);
    EXPECT_THROW((void)request(profile,temporal),std::invalid_argument);
    const auto q=request(profile,good);
    EXPECT_THROW((void)ImmutableRowTablePlacementPlanner::select(q.requested,q.input,
        MemoryPlanner::plan(profile,good).physicalPlan(),q.contributions),std::invalid_argument);
    // All current replicated/Auto source cells retain their original discriminator.
    EXPECT_EQ(ImmutableRowTablePolicy{}.distribution,ImmutableRowTableDistribution::Replicated);
}

/**
 * @test The exact admitted phase schedule survives wire7 and rejects old/unknown schemas.
 * @details Only metadata and the existing canonical provision run. A unique full
 * common-geometry word sequence identifies the field to corrupt; no payload or
 * native allocator is created and no hard-coded byte offset guesses a consumer.
 */
TEST(NgramExecutionProvision, AttentionPhaseScheduleWireRoundTripRejectsUnknownAndOldVersion)
{
    using Schedule=qwen4exp::QwenSparseAttentionPhaseSchedule;
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto& profile=source.metadata().memoryProfile();
    for(const auto schedule:{Schedule::Uniform,Schedule::Prefill8Decode33}){
        auto c=consumer(DeviceType::ROCm,ImmutableRowStoragePreference::MemoryMapped);c.attention_phase_schedule=schedule;
        const auto q=request(profile,{c});const std::array requests{q};const auto wire=serializeImmutableRowTableAdmissionRequests(requests);
        const auto decoded=parseImmutableRowTableAdmissionRequests(wire,0,&profile);
        ASSERT_EQ(decoded.size(),1u);ASSERT_TRUE(decoded.front().execution_provision);
        EXPECT_EQ(decoded.front().execution_provision->consumers(),q.execution_provision->consumers());
        const auto& consumers=q.execution_provision->consumers();ASSERT_EQ(consumers.size(),1u);
        const auto& a=consumers.front().activation;EXPECT_EQ(a.attention_phase_schedule,schedule);
        auto prior=wire;prior.at(4)=6;EXPECT_THROW((void)parseImmutableRowTableAdmissionRequests(prior,0,&profile),std::invalid_argument);
        std::vector<std::uint8_t> words;
        for(const auto value:{std::uint32_t(a.local_d_ff),std::uint32_t(a.local_n_heads),std::uint32_t(a.local_n_kv_heads),std::uint32_t(a.local_vocab),
            std::uint32_t(a.replicated_dense_decode),std::uint32_t(a.routed_compute_policy),std::uint32_t(a.mtp_terminal_logits_layout),std::uint32_t(schedule)})
            for(int shift=0;shift<32;shift+=8)words.push_back(std::uint8_t(value>>shift));
        const auto found=std::search(wire.begin(),wire.end(),words.begin(),words.end());ASSERT_NE(found,wire.end());
        ASSERT_EQ(std::search(found+1,wire.end(),words.begin(),words.end()),wire.end());
        const auto offset=std::size_t(found-wire.begin())+words.size()-4;
        for(const std::uint32_t invalid:{2u,255u,256u,0x7fffffffu}){
            auto bad=wire;for(int byte=0;byte<4;++byte)bad.at(offset+byte)=std::uint8_t(invalid>>(8*byte));
            try{(void)parseImmutableRowTableAdmissionRequests(bad,0,&profile);FAIL()<<"Unknown schedule accepted";}
            catch(const std::invalid_argument& error){EXPECT_NE(std::string(error.what()).find("attention phase schedule"),std::string::npos);}
        }
    }
}
/** @test Consumer identity includes schedule and restore rejects unknown typed enum values. */
TEST(NgramExecutionProvision, AttentionPhaseScheduleIsPartOfConsumerAndRestoreIdentity)
{
    using Schedule=qwen4exp::QwenSparseAttentionPhaseSchedule;
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());const auto& profile=source.metadata().memoryProfile();
    auto c=consumer(DeviceType::ROCm,ImmutableRowStoragePreference::MemoryMapped);const auto q=request(profile,{c});
    const auto original=q.execution_provision->consumers();ASSERT_FALSE(original.empty());auto changed=original;
    changed.front().activation.attention_phase_schedule=Schedule::Prefill8Decode33;
    EXPECT_FALSE(changed.front()==original.front());
    changed.front().activation.attention_phase_schedule=static_cast<Schedule>(255);
    EXPECT_THROW((void)MemoryPlanner::restoreImmutableRowTableExecutionProvision(profile,q.requested,q.input,changed,
        q.execution_provision->executionPlan(),q.execution_provision->requiredBase()),std::invalid_argument);
}
