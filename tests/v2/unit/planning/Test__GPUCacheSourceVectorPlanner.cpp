/**
 * @file Test__GPUCacheSourceVectorPlanner.cpp
 * @brief Replay original two-worker source preparation through actual QFN final planning.
 *
 * Only native worker bodies and certificate observations are recorded fixtures.
 * Original Preparing PMA, mapped stack/object roots, worker pool, source freeze,
 * complete mandatory planning and final same-state admission execute unchanged.
 * This proves planning/ownership consumption, not remote captured inference.
 */
#include "planning/MemoryPlanner.h"
#include "execution/qwen4exp/PooledImmutableRetainedParticipantBudget.h"
#include "fixtures/planning/ImmutableRowTableAdmissionWireReplayAccess.h"
#include "fixtures/planning/RecordedNgramOrdering.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/ImmutableRowTableAdmissionWire.h"
#include "loaders/AdaptiveImmutableLookupGeometry.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <array>
namespace llaminar2::test {
namespace {
using Access=NativeContextPreparationReplayAccess;
/** @brief Recorded constructor on the actual original preclaimed stack/object extent. */
class SourceVectorWorker final:public llaminar2::testing::MockWorkerGPUContext {
public:
    /** @brief Retain the original constructor signature without any native device API. */
    SourceVectorWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
};
/** @brief Retire actual original mock worker generations even after a failed assertion. */
class SourceVectorRetirement final {
public:
    /** @brief Strongly retain the original source and exact declared population. */
    SourceVectorRetirement(std::shared_ptr<NativeContextPreparation> source,std::span<const DeviceId> devices)
        :source_(std::move(source)),devices_(devices.begin(),devices.end()){}
    /** @brief Use the existing trusted reset observer only for recorded device-free workers. */
    ~SourceVectorRetirement(){
        for(const auto device:devices_){auto scope=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);Access::physicalReset(scope);}
        if(source_->memory()->phase()==PhysicalMemoryAuthorityPhase::Preparing)source_->retireUnadmitted();
    }
private:
    std::shared_ptr<NativeContextPreparation> source_;
    std::vector<DeviceId> devices_;
};
/** @return Original frozen observed inventory, including both independently owned GPU allocators. */
std::array<PhysicalMemoryResource,3> observations(){
    return {{{0,DeviceId::cpu(),512ull<<30,500ull<<30},{0,DeviceId::cuda(0),512ull<<30,500ull<<30},
        {0,DeviceId::rocm(0),512ull<<30,500ull<<30}}};
}
/** @return Exact existing full-main QFN consumer; remote cache storage is not another graph config. */
DevicePlanConfig consumer(DeviceId local,bool bounded){
    DevicePlanConfig c;c.world_rank=0;c.device=local;c.device_compute_units=80;
    c.device_total_bytes=512ull<<30;c.device_free_bytes=500ull<<30;c.associated_host_memory=observations()[0];
    c.max_seq_len=64;c.activation_seq_len=32;c.first_layer=0;c.last_layer=47;c.batch_size=1;
    ImmutableRowTablePolicy p{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};p.window_count=2;
    const auto remote=local.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0);
    const auto rows=bounded?64u:0u;const auto bytes=bounded?1u<<20:0u;
    p.gpu_adaptive.host={.maximum_cache_rows=rows,.cache_physical_byte_limit=bytes};
    p.gpu_adaptive.placement={.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal,
        .owners={{{0,local.type,local.ordinal},rows,bytes},{{0,remote.type,remote.ordinal},rows,bytes}},
        .coordinator=AdaptiveGPUCacheParticipant{0,local.type,local.ordinal}};
    p.validate();c.ngram_execution_policy=std::move(p);const std::array devices{local,remote};c.native_ordering_events=NativeOrderingEventGroupReplayAccess::projectionWithRemoteGraphs(devices);test::declareRecordedNgramHostServices(c);return c;
}
/** @return Exact named PMA owner contribution; no duplicated capacity arithmetic. */
std::size_t bytes(const PhysicalMemoryPlan& plan,PhysicalMemoryAllocatorIdentity id,PhysicalMemoryOwner owner){
    const auto* bom=plan.find(id);return bom?bom->bytes(owner):0;
}
}
/** @test Actual original selected workers, mandatory roots and final optional geometry share the same authority. */
TEST(GPUCacheSourceVectorPlanner, OriginalTwoWorkerSourceFeedsFinalQFNPlanAtC0AndBoundedCapacity){
    Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto& profile=original.metadata().memoryProfile();
    for(const auto local:{DeviceId::cuda(0),DeviceId::rocm(0)})for(const bool bounded:{false,true}){
        auto cfg=consumer(local,bounded);const auto remote=local.is_cuda()?DeviceId::rocm(0):DeviceId::cuda(0);
        const std::array devices{local,remote};const auto observed=observations();
        for(const auto device:devices)Access::seed(device,[ordinal=device.ordinal](std::shared_ptr<PreparedGPUWorkerStack> root){
            return Access::construct<SourceVectorWorker>(std::move(root),ordinal);});
        auto source=Access::source<SourceVectorWorker>(std::span<const DeviceId>{devices},observed);
        SourceVectorRetirement retirement(source,devices);const auto memory=source->memory();
        {
            for(const auto device:devices){const auto permit=source->acquire(device);EXPECT_EQ(permit.memory(),memory);EXPECT_NE(permit.generation(),0u);
                const auto existing=GPUDeviceContextPool::instance().retainExistingPreparedContext(device,memory);
                EXPECT_NO_THROW(permit.requireWorker(existing.worker()));
                EXPECT_NO_THROW(existing.requireWorker(permit.worker()));}
        }
        source->freezeForPlanning();cfg.native_context_preparation=source;
        const auto mandatory_source=source->mandatoryBOM();const auto frozen=MemoryPlanner::plan(profile,{cfg}).physicalPlan();
        ASSERT_TRUE(frozen.fits());const std::vector<std::vector<DevicePlanConfig>> groups{{cfg}};
        const auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,*cfg.ngram_execution_policy,groups,{},cfg.ngram_host_placement);
        ASSERT_TRUE(request);ASSERT_TRUE(request->execution_provision);ASSERT_EQ(request->input.consumers().size(),1u);
        ASSERT_EQ(request->input.gpuResources().size(),2u);EXPECT_EQ(request->input.consumers()[0].device,local);
        EXPECT_EQ(request->input.gpuResources()[1].device,remote);
        const auto maximum=request->execution_provision->maximumLiveIds();
        const auto zero=AdaptiveImmutableLookupGeometry::fromLayout(request->input.gpuAdaptiveLayout(*cfg.ngram_execution_policy,0,0,maximum),
            {.rows=profile.flash_next->ple_table_rows,.row_elements=profile.flash_next->ple_row_elements,
             .format=static_cast<GGUFTensorType>(profile.flash_next->ple_table_gguf_type)},cfg.ngram_execution_policy->gpu_adaptive);
        const auto mandatory_lookup=llaminar2::test::recordedNgramMandatory(zero,request->input.gpuResources(),request->input.host());
        for(const auto device:devices){const PhysicalMemoryAllocatorIdentity id{0,device};
            EXPECT_EQ(bytes(frozen,id,PhysicalMemoryOwner::NativeExecutionContext),bytes(mandatory_source,id,PhysicalMemoryOwner::NativeExecutionContext));
            EXPECT_GT(bytes(frozen,id,PhysicalMemoryOwner::NativeExecutionContext),0u);
            EXPECT_EQ(bytes(frozen,id,PhysicalMemoryOwner::ImmutableLookupStaging),bytes(mandatory_lookup,id,PhysicalMemoryOwner::ImmutableLookupStaging));}
        EXPECT_GT(bytes(frozen,{0,local},PhysicalMemoryOwner::PrimaryModelWeights),0u);
        EXPECT_EQ(bytes(frozen,{0,remote},PhysicalMemoryOwner::PrimaryModelWeights),0u);
        EXPECT_EQ(bytes(frozen,{0,remote},PhysicalMemoryOwner::KVCache),0u);
        const auto selected=ImmutableRowTablePlacementPlanner::select(*request,frozen);ASSERT_TRUE(selected.physical_plan.fits());
        const auto& layout=selected.placement->gpuAdaptiveLayout();ASSERT_EQ(layout.consumers().size(),1u);ASSERT_EQ(layout.owners().size(),3u);
        const auto inventory=selected.placement->allocators();
        ASSERT_EQ(inventory.size(),layout.owners().size());
        std::size_t ownerIndex=0;
        for(const auto& allocator:inventory){
            EXPECT_EQ(&allocator,&layout.owners()[ownerIndex].key.allocator);
            EXPECT_TRUE(selected.placement->containsAllocator(allocator));
            ++ownerIndex;
        }
        for(const auto device:{DeviceId::cpu(),local,remote})EXPECT_TRUE(selected.placement->containsAllocator({0,device}));
        EXPECT_FALSE(selected.placement->containsAllocator({0,DeviceId(local.type,1)}));
        EXPECT_FALSE(selected.placement->containsAllocator({1,remote}));

        const auto geometry=AdaptiveImmutableLookupGeometry::fromLayout(layout,
            {.rows=profile.flash_next->ple_table_rows,.row_elements=profile.flash_next->ple_row_elements,
             .format=static_cast<GGUFTensorType>(profile.flash_next->ple_table_gguf_type)},cfg.ngram_execution_policy->gpu_adaptive);
        ASSERT_EQ(geometry.physical().owners().size(),2u);
        const auto channels=qwen4exp::PooledImmutableDirectedTransport::completeBOM(
            geometry.physical(),geometry.coordinator(),request->input.host(),request->input.gpuResources());
        const auto floorChannels=qwen4exp::PooledImmutableDirectedTransport::completeBOM(
            zero.physical(),zero.coordinator(),request->input.host(),request->input.gpuResources());
        // No other staging program exists in this source fixture: compare every
        // actual allocator exactly, including CPU0 rather than a GPU host alias.
        for(const auto device:{DeviceId::cpu(),local,remote}){
            const PhysicalMemoryAllocatorIdentity id{0,device};
            EXPECT_GT(bytes(channels,id,PhysicalMemoryOwner::ActivationTransportStaging),0u);
            EXPECT_EQ(bytes(frozen,id,PhysicalMemoryOwner::ActivationTransportStaging),bytes(floorChannels,id,PhysicalMemoryOwner::ActivationTransportStaging));
            EXPECT_EQ(bytes(selected.physical_plan,id,PhysicalMemoryOwner::ActivationTransportStaging),bytes(channels,id,PhysicalMemoryOwner::ActivationTransportStaging));
        }
        const auto wire=serializeImmutableRowTableAdmissionRequests(std::array{*request});
        const auto restored=ImmutableRowTableAdmissionWireReplayAccess::parse(wire,0,&profile,cfg.native_ordering_events,request->retained_host_services);
        ASSERT_EQ(restored.size(),1u);
        const auto restoredSelection=ImmutableRowTablePlacementPlanner::select(restored.front(),frozen);
        EXPECT_FALSE(restoredSelection.physical_plan.requiredFootprintMismatch(selected.physical_plan));
        EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(restoredSelection.physical_plan));

        for(std::size_t i=0;i<devices.size();++i){EXPECT_EQ(geometry.physical().owners()[i].allocator.device,devices[i]);
            EXPECT_EQ(geometry.physical().owners()[i].slots,bounded?64u:0u);}
        const auto eventOnly=NativeOrderingEventGroupReplayAccess::projection(devices);
        const auto program=qwen4exp::PooledImmutableDirectedOwnerProgram::declare(geometry.physical(),geometry.coordinator(),1);
        EXPECT_THROW(eventOnly.remoteHostBOM(program,request->input.host(),source->originalResource(remote)),std::invalid_argument);
        const auto host=cfg.native_ordering_events.remoteHostBOM(program,request->input.host(),source->originalResource(remote));
        const auto complete=qwen4exp::PooledImmutableRetainedParticipantBudget::completeBOM(
            geometry,request->input.host(),request->input.gpuResources(),cfg.native_ordering_events);
        const auto floorComplete=qwen4exp::PooledImmutableRetainedParticipantBudget::completeBOM(
            zero,request->input.host(),request->input.gpuResources(),cfg.native_ordering_events);
        const auto floorHost=cfg.native_ordering_events.remoteHostBOM(
            qwen4exp::PooledImmutableDirectedOwnerProgram::declare(zero.physical(),zero.coordinator(),1),
            request->input.host(),source->originalResource(remote));
        const auto holder=RetainedParticipantGraphSourceSlots::physicalBytes(geometry.physical().owners().size());
        EXPECT_EQ(bytes(complete,{0,DeviceId::cpu()},PhysicalMemoryOwner::NativeStorageGraphExecutable),
            bytes(host,{0,DeviceId::cpu()},PhysicalMemoryOwner::NativeStorageGraphExecutable)+holder);
        EXPECT_EQ(bytes(floorComplete,{0,DeviceId::cpu()},PhysicalMemoryOwner::NativeStorageGraphExecutable),
            bytes(floorHost,{0,DeviceId::cpu()},PhysicalMemoryOwner::NativeStorageGraphExecutable)+holder);
        // Storage-definition metadata has its own admitted purpose. It must
        // neither consume model graph capacity nor claim device payload here.
        EXPECT_GT(bytes(host,{0,DeviceId::cpu()},PhysicalMemoryOwner::NativeStorageGraphExecutable),0u);
        EXPECT_EQ(bytes(host,{0,remote},PhysicalMemoryOwner::NativeStorageGraphExecutable),0u);
        EXPECT_EQ(bytes(host,{0,DeviceId::cpu()},PhysicalMemoryOwner::NativeGraphExecutable),0u);
        EXPECT_EQ(bytes(host,{0,remote},PhysicalMemoryOwner::NativeGraphExecutable),0u);
        PhysicalMemoryPlanBuilder optionalPlan(llaminar2::test::recordedNgramOptional(geometry,request->input.gpuResources(),request->input.host()));
        const auto remoteGrowth=qwen4exp::PooledImmutableRetainedParticipantBudget::slotDependentBOM(geometry,zero,request->input.host(),request->input.gpuResources(),cfg.native_ordering_events);
        for(const auto& line:remoteGrowth.resources())optionalPlan.add(line);
        const auto optional=optionalPlan.build();
        for(const auto& bom:frozen.resources())for(std::size_t i=0;i<PhysicalMemoryBOM::ownerCount();++i){const auto owner=PhysicalMemoryBOM::ownerAt(i);
            const PhysicalMemoryAllocatorIdentity id{bom.resource().world_rank,bom.resource().device};
            EXPECT_EQ(bytes(selected.physical_plan,id,owner),bom.bytes(owner)+bytes(optional,id,owner));}
        auto resolved=cfg;resolved.ngram_table_placement=selected.placement;
        const auto projected=MemoryPlanner::plan(profile,{resolved}).physicalPlan();
        EXPECT_FALSE(projected.requiredFootprintMismatch(selected.physical_plan));EXPECT_FALSE(selected.physical_plan.requiredFootprintMismatch(projected));
        auto certificate=std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(selected.physical_plan);
        source->admitFinal(certificate);EXPECT_EQ(source->memory(),memory);EXPECT_EQ(memory->admission(),certificate);
        EXPECT_EQ(memory->phase(),PhysicalMemoryAuthorityPhase::Admitted);
        for(const auto device:{DeviceId::cpu(),local,remote}){
            const auto extent=bytes(channels,{0,device},PhysicalMemoryOwner::ActivationTransportStaging);
            auto claim=memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationTransportStaging,extent);
            EXPECT_THROW((void)memory->claimNewAllocation(device,PhysicalMemoryOwner::ActivationTransportStaging,1u),std::logic_error);
        }

    }
}
/** @test Original observations alone cannot authorize an undeclared worker or an unused preparation. */
TEST(GPUCacheSourceVectorPlanner, UndeclaredOrUnusedOriginalSourceRejectsBeforeFinalPlan){
    Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());const auto& profile=original.metadata().memoryProfile();
    const auto observed=observations();auto cfg=consumer(DeviceId::cuda(0),false);
    {
        const std::array devices{DeviceId::cuda(0),DeviceId::rocm(0)};
        auto unused=Access::source<SourceVectorWorker>(std::span<const DeviceId>{devices},observed);
        SourceVectorRetirement retirement(unused,devices);cfg.native_context_preparation=unused;
        EXPECT_THROW((void)MemoryPlanner::plan(profile,{cfg}),std::logic_error);
    }
    {
        const auto device=DeviceId::cuda(0);Access::seed(device,[](std::shared_ptr<PreparedGPUWorkerStack> root){
            return Access::construct<SourceVectorWorker>(std::move(root),0);});
        auto incomplete=Access::source<SourceVectorWorker>(device,observed);const std::array devices{device};
        SourceVectorRetirement retirement(incomplete,devices);{auto permit=incomplete->acquire(device);}
        incomplete->freezeForPlanning();cfg.native_context_preparation=incomplete;
        EXPECT_THROW((void)MemoryPlanner::plan(profile,{cfg}),std::invalid_argument);
    }
}
}
