/**
 * @file Test__PooledImmutableDirectedTransport.cpp
 * @brief Replay exact directed channel floor/growth/runtime capacities without devices or invented source permission.
 *
 * Recorded table/generation metadata is used only to compare pure runtime plan
 * projections. No worker, source permit, native executable or certificate is
 * created. Real captured wire extents remain covered by the original native
 * cross-vendor owner replays and the TransferEngine counted-message suite.
 */
#include "execution/qwen4exp/PooledImmutableDirectedTransport.h"
#include <gtest/gtest.h>
#include <array>
#include <map>
namespace llaminar2::test {
namespace {
namespace q=qwen4exp;
struct Shape {q::PooledImmutablePhysicalOwnerGeometry physical;q::PooledImmutableCoordinatorGeometry coordinator;std::vector<PhysicalMemoryResource> resources;};
/** @return Original declared owner order, never compute-rank order or an admission permission. */
Shape shape(DeviceId source,std::span<const std::uint32_t> slots,bool sameVendor=false,std::optional<PeerAccessCoverage> peerAccess=std::nullopt){
    const PhysicalMemoryAllocatorIdentity coordinator{0,source};std::vector<PhysicalMemoryAllocatorIdentity> order;
    std::vector<ImmutableRowTableLayoutOwner> owners;std::vector<PhysicalMemoryResource> resources{{0,source,1ULL<<34,1ULL<<34}};bool positive=false;
    // Include actual local C0 owner followed by storage-only opposite-vendor owners.
    order.push_back(coordinator);owners.push_back({{coordinator,ImmutableRowStorage::VRAM},0,0,{}});
    for(std::size_t i=0;i<slots.size();++i){const auto type=sameVendor?source.type:(source.is_cuda()?DeviceType::ROCm:DeviceType::CUDA);
        const auto device=type==DeviceType::CUDA?DeviceId::cuda(i):DeviceId::rocm(i);const PhysicalMemoryAllocatorIdentity owner{0,device};
        order.push_back(owner);owners.push_back({{owner,ImmutableRowStorage::VRAM},0,slots[i],{}});positive|=slots[i]!=0;
        resources.push_back({0,device,1ULL<<34,1ULL<<34});}
    const std::array tiers{ImmutableRowStorage::VRAM,ImmutableRowStorage::MemoryMapped};
    const auto layout=ImmutableRowTableLayoutGeometry::checked(ImmutableRowTableExtent::fromNative(GGUFTensorType::IQ4_NL,0x100000100ULL,90),
        ImmutableRowTableTiers::of(tiers),ImmutableRowTableResidency::Adaptive,std::move(owners),{coordinator},{},
        {{{0,DeviceId::cpu()},ImmutableRowTableStoragePolicy::forStorage(ImmutableRowStorage::MemoryMapped)}},{},positive?std::optional(coordinator):std::nullopt,512,ImmutableRowTableAdaptiveGPUDomain::SharedCoordinator,peerAccess);
    auto physical=q::PooledImmutablePhysicalOwnerGeometry::fromLayout(layout,order,coordinator);
    auto policy=q::PooledImmutableCoordinatorGeometry::fromOwners(physical,{65536,1,2},1024);
    return {std::move(physical),std::move(policy),std::move(resources)};
}
/** @return Typed source contribution by actual allocator; no admission/free arithmetic is duplicated. */
std::map<DeviceId,std::size_t> bytes(const PhysicalMemoryPlan& plan){std::map<DeviceId,std::size_t> result;for(const auto& item:plan.resources())result[item.resource().device]+=item.bytes(PhysicalMemoryOwner::ActivationTransportStaging);return result;}
const PhysicalMemoryResource host{0,DeviceId::cpu(),1ULL<<34,1ULL<<34};
}
TEST(PooledImmutableDirectedTransport, OrderedZeroPartialAndShrinkingCandidateGrowthIsExact){
    for(const auto source:{DeviceId::cuda(7),DeviceId::rocm(7)})for(const auto count:{1u,2u,4u})for(unsigned replay=0;replay<20;++replay){
        std::vector<std::uint32_t> slots(count);for(unsigned i=0;i<count;++i)slots[i]=std::array{0u,1u,63u,64u,65u,512u,0u,2u}[(replay+i)%8];
        const auto s=shape(source,slots);auto mandatory=bytes(q::PooledImmutableDirectedTransport::mandatoryBOM(s.physical,s.coordinator,host,s.resources));
        const auto growth=bytes(q::PooledImmutableDirectedTransport::slotDependentBOM(s.physical,s.coordinator,host,s.resources));
        const auto complete=bytes(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,s.resources));
        for(const auto& [device,value]:growth)mandatory[device]+=value;
        EXPECT_EQ(mandatory,complete);
        EXPECT_EQ(complete.at(source),count*2*sizeof(CapturedTransferCursor));
        for(const auto& owner:s.physical.owners())if(owner.allocator.device!=source)EXPECT_EQ(complete.at(owner.allocator.device),2*sizeof(CapturedTransferCursor));
    }
}
TEST(PooledImmutableDirectedTransport, PlanningAndOriginalRuntimePinEdgeUseIdenticalPhysicalChannels){
    for(const auto capacity:{0u,1u,64u,512u}){
        const std::array slots{capacity,0u,capacity/2};const auto s=shape(DeviceId::cuda(7),slots);
        const auto coordinator=q::PooledImmutableCoordinatorNativePlan::fromGeometry(s.physical,s.coordinator,701,17);
        std::size_t hostBytes=0;
        for(std::uint32_t coordinate=1;coordinate<s.physical.owners().size();++coordinate){
            const auto owner=q::PooledImmutableOwnerNativePlan::fromGeometry(s.physical,s.coordinator,coordinate,701,17);
            const auto edge=q::PooledImmutablePinEdge::fromPlans(coordinator,owner);
            const auto free=q::PooledImmutableFreeEdge::fromOriginal(edge);
            EXPECT_EQ(edge.hasCommands(),owner.parameters().native.reply_capacity!=0);
            EXPECT_EQ(free.hasPayload(),edge.hasCommands());
            if(!edge.hasCommands()){
                EXPECT_EQ(edge.maximumCommandBytes(),0u);EXPECT_EQ(edge.replySource().bytes,0u);
                EXPECT_FALSE(edge.incomingCount());EXPECT_FALSE(edge.replyCountDestination());
                EXPECT_EQ(free.maximumDecisionBytes(),0u);EXPECT_FALSE(free.decisionCount());
            }
            const auto planned=q::PooledImmutableDirectedChannelGeometry::fromGeometry(s.physical,s.coordinator,coordinate);
            EXPECT_EQ(edge.channelMemory().mapped_host_bytes,planned.forwardMemory().mapped_host_bytes);
            EXPECT_EQ(edge.returnChannelMemory().mapped_host_bytes,planned.reverseMemory().mapped_host_bytes);
            hostBytes+=edge.channelMemory().mapped_host_bytes+edge.returnChannelMemory().mapped_host_bytes;
        }
        EXPECT_EQ(bytes(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,s.resources)).at(DeviceId::cpu()),hostBytes);
    }
}
/** @test Explicit NoPeer retains compact mapped channel costs and original per-device graph identity. */
TEST(PooledImmutableDirectedTransport, ExplicitNoPeerKeepsExactFloorGrowthAndNativeOwnerIdentity)
{
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto count : {1u, 3u, 7u})
            for (const auto capacity : {0u, 64u}) {
                const DeviceId source{type,static_cast<int>(count)};
                const std::vector<std::uint32_t> slots(count, capacity);
                const auto s = shape(source, slots, true, PeerAccessCoverage::None);
                auto floor = bytes(q::PooledImmutableDirectedTransport::mandatoryBOM(s.physical,s.coordinator,host,s.resources));
                const auto growth = bytes(q::PooledImmutableDirectedTransport::slotDependentBOM(s.physical,s.coordinator,host,s.resources));
                const auto full = bytes(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,s.resources));
                for (const auto& [device, value] : growth) floor[device] += value;
                EXPECT_EQ(floor, full);
                const auto coordinator = q::PooledImmutableCoordinatorNativePlan::fromGeometry(s.physical,s.coordinator,701,17);
                for (std::uint32_t i = 1; i < s.physical.owners().size(); ++i) {
                    const auto owner = q::PooledImmutableOwnerNativePlan::fromGeometry(s.physical,s.coordinator,i,701,17);
                    const auto edge = q::PooledImmutablePinEdge::fromPlans(coordinator,owner);
                    EXPECT_EQ(owner.peerAccess(), PeerAccessCoverage::None);
                    EXPECT_TRUE(edge.capturedChannelTransport());
                    EXPECT_NO_THROW(edge.channelMemory());
                }
            }
}
TEST(PooledImmutableDirectedTransport, ForeignResourcesAndUnsupportedNativeRoutesCannotAcquireMappedBOM){
    const std::array slots{64u,0u};const auto s=shape(DeviceId::cuda(7),slots);
    auto missing=s.resources;missing.pop_back();EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,missing),std::invalid_argument);
    auto duplicate=s.resources;duplicate.push_back(duplicate.front());EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,host,duplicate),std::invalid_argument);
    const auto native=shape(DeviceId::cuda(7),slots,true);EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(native.physical,native.coordinator,host,native.resources),std::invalid_argument);
    auto foreign=host;foreign.world_rank=1;EXPECT_THROW(q::PooledImmutableDirectedTransport::completeBOM(s.physical,s.coordinator,foreign,s.resources),std::invalid_argument);
    EXPECT_THROW(q::PooledImmutableDirectedTransport::create({}),std::invalid_argument);
}
}
