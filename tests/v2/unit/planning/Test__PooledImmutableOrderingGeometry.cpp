/**
 * @file Test__PooledImmutableOrderingGeometry.cpp
 * @brief Actual pooled floor/optional contribution equality with sealed recorded event programs.
 *
 * Nonlinear toy bounds are private recorded inputs, not native measurements.
 * The production geometry and PMA contribution bodies remain the sole arithmetic
 * authority; no catalog, worker, source key or allocation is manufactured.
 */
#include "fixtures/loaders/AdaptiveImmutableLookupGeometryReplayAccess.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
namespace llaminar2::test { namespace {
using Allocator=PhysicalMemoryAllocatorIdentity;
/** @return Authored original observation; no physical/native query runs. */
PhysicalMemoryResource observed(Allocator id){return {id.world_rank,id.device,1ull<<34,1ull<<34};}
/** @brief Append exact source-owned BOM lines without re-authoring their byte arithmetic. */
void append(PhysicalMemoryPlanBuilder& result,const PhysicalMemoryPlan& plan){for(const auto& line:plan.resources())result.add(line);}
/** @return Original complete physical declaration, preserving empty and skewed owners. */
auto geometry(std::span<const std::uint32_t> slots,const std::vector<Allocator>& order,Allocator coordinator){
    const ImmutableRowTableSource source{.rows=4096,.row_elements=64,.format=GGUFTensorType::Q8_0};
    std::vector<ImmutableRowTableLayoutOwner> owners;bool positive=false;
    for(std::size_t i=0;i<slots.size();++i){owners.push_back({{order[i],ImmutableRowStorage::VRAM},0,slots[i],{}});positive|=slots[i]!=0;}
    owners.push_back({{{0,DeviceId::cpu()},ImmutableRowStorage::RAM},0,7,{}});
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::RAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(source.format,source.rows,source.rowBytes()),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,owners,{coordinator},{},
        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped,4096,2)}},
        {},positive?std::optional(coordinator):std::nullopt,16);
    AdaptiveVRAMRAMSSDRequest request;request.maximum_gpu_cache_rows=128;request.gpu_cache_physical_byte_limit=1u<<20;
    request.host.maximum_cache_rows=7;request.host.cache_physical_byte_limit=1u<<20;
    return AdaptiveImmutableLookupGeometry::fromLayout(layout,source,request,order);
}
/** @test Fixed C0 plus every optional event/root growth equals complete geometry before admission. */
TEST(PooledImmutableOrderingGeometry, NonlinearGroupFloorAndGrowthMatchActualPrograms){
    using Access=AdaptiveImmutableLookupGeometryReplayAccess;
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const auto count:{1u,2u,4u})for(const bool local:{false,true}){
        std::vector<Allocator> order;for(std::size_t i=0;i<count;++i)order.push_back({0,{i%2?backend:(backend==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA),static_cast<int>(i)}});
        const auto coordinator=local?order.front():Allocator{0,{backend,7}};
        std::vector<PhysicalMemoryResource> devices;std::vector<DeviceId> ids;
        for(const auto value:order){devices.push_back(observed(value));ids.push_back(value.device);}
        if(!local){devices.push_back(observed(coordinator));ids.push_back(coordinator.device);}
        std::reverse(devices.begin(),devices.end());const auto host=observed({0,DeviceId::cpu()});
        const auto projection=NativeOrderingEventGroupReplayAccess::projection(ids);
        const std::vector<std::uint32_t> empty(count,0);const auto floor=geometry(empty,order,coordinator);
        const auto frozen=Access::mandatory(floor,devices,host,projection);
        for(int replay=0;replay<20;++replay){
            std::vector<std::uint32_t> slots(count);for(std::size_t i=0;i<count;++i)slots[i]=replay%3==0?0:replay%3==1?(i?0:65):static_cast<std::uint32_t>(i+1);
            const auto actual=geometry(slots,order,coordinator);
            PhysicalMemoryPlanBuilder partitioned(frozen),complete(Access::mandatory(actual,devices,host,projection));
            append(partitioned,Access::optional(actual,devices,host,projection));append(complete,actual.cacheBOM(devices,host));
            const auto left=partitioned.build(),right=complete.build();
            EXPECT_FALSE(left.requiredFootprintMismatch(right));EXPECT_FALSE(right.requiredFootprintMismatch(left));
            PhysicalMemoryPlanBuilder expected;
            for(std::uint32_t i=0;i<count;++i)append(expected,projection.bom(
                qwen4exp::PooledImmutableOwnerNativePlan::orderingProgram(actual.physical(),actual.coordinator(),i),host,observed(order[i])));
            append(expected,projection.bom(qwen4exp::PooledImmutableCoordinatorNativePlan::orderingProgram(actual.physical(),actual.coordinator()),host,observed(coordinator)));
            append(expected,projection.bom(NativeOrderingEventProgram::pooledSource(),host,observed(coordinator)));
            append(expected,projection.bom(NativeOrderingEventProgram::lookupSubmissionRetirement(),host,observed(coordinator)));
            const auto group=expected.build();
            for(const auto& resource:group.resources()){
                ASSERT_NE(left.find(Allocator{resource.resource().world_rank,resource.resource().device}),nullptr);
                EXPECT_EQ(left.find(Allocator{resource.resource().world_rank,resource.resource().device})->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),resource.bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent));
            }
        }
    }
}
/** @test Cold installed bounds price remote resources without granting native creation authority. */
TEST(PooledImmutableOrderingGeometry, InstalledColdBoundsDoNotCreateOrQualifyWorkers){
    for(const auto device:{DeviceId::cuda(7),DeviceId::rocm(7)}) {
        SCOPED_TRACE(device.toString());
        const std::vector<Allocator> owners{{0,device}};
        const std::array slots{0u};
        const auto g=geometry(slots,owners,owners.front());
        const std::array devices{observed(owners.front())};
        const auto host=observed({0,DeviceId::cpu()});
        auto& pool=GPUDeviceContextPool::instance();
        EXPECT_THROW((void)pool.retainExistingContextProvenance(device),std::logic_error);
        const auto cold=g.mandatoryBOM(devices,host);
        const auto* hostLine=cold.find(Allocator{0,DeviceId::cpu()});
        const auto* deviceLine=cold.find(owners.front());
        ASSERT_NE(hostLine,nullptr);ASSERT_NE(deviceLine,nullptr);
        EXPECT_GT(hostLine->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
        EXPECT_GT(deviceLine->bytes(PhysicalMemoryOwner::NativeImmutableLookupOrderingEvent),0u);
        // Cold pricing carries no worker, generation or source capability. The
        // actual production creation selector must still reject this endpoint.
        EXPECT_THROW((void)NativeAuxiliaryResourceCatalog::require(device,
            NativeAuxiliaryResourceProgram::ordering(NativeOrderingEventProgram::pooledSource())),std::logic_error);
        EXPECT_THROW((void)pool.retainExistingContextProvenance(device),std::logic_error);
    }
}
}}
