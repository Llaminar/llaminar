/**
 * @file Test__OriginalMappedNativeCohorts.cpp
 * @brief Device-free original-rank cohort selection and rejection replay.
 *
 * Sparse physical NUMA IDs and nonconsecutive GPU ordinals are recorded inputs,
 * never runtime discovery. Positive odd/unequal widths prove selection follows
 * authentic ownership and UUIDs rather than rank/ordinal arithmetic or a count
 * allowlist. No test initializes MPI, a backend, native worker or accelerator.
 */
#include "fixtures/backends/OriginalMappedNativeCohorts.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;
using namespace llaminar2::test;
namespace {
/** One complete immutable recorded hardware publication, retaining both original ranks. */
std::shared_ptr<ClusterInventory> recordedMappedInventory(DeviceType source,std::size_t source_count,
    DeviceType peer,std::size_t peer_count,bool reverse=false){
    auto result=std::make_shared<ClusterInventory>();result->world_size=2;result->ranks.resize(2);
    for(int rank=0;rank<2;++rank){auto& owner=result->ranks[rank];owner.rank=rank;owner.node_id=7;owner.hostname="recorded-original-host";
        owner.cpu.type=DeviceType::CPU;owner.cpu.numa_node=rank?23:11;}
    for(int role=0;role<2;++role){auto& owner=result->ranks[role^(reverse?1:0)];const auto family=role?peer:source;const auto count=role?peer_count:source_count;
        for(std::size_t index=0;index<count;++index){DeviceInfo gpu;gpu.type=family;gpu.local_device_id=int(3*index+2);gpu.numa_node=owner.cpu.numa_node;
            gpu.uuid="physical-role"+std::to_string(role)+"-gpu"+std::to_string(index);owner.gpus.push_back(gpu);}}
    return result;
}
}
TEST(OriginalMappedNativeCohorts, PreservesSingletonOddUnequalWidthsAndOriginalNUMA){
    for(const auto counts:{std::pair{1u,1u},std::pair{1u,3u},std::pair{3u,5u},std::pair{5u,2u}})
        for(const bool reverse:{false,true}){
            const auto inventory=recordedMappedInventory(DeviceType::CUDA,counts.first,DeviceType::ROCm,counts.second,reverse);
            const auto bound=OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,counts.first},{DeviceType::ROCm,counts.second});
            EXPECT_EQ(bound.inventory().get(),inventory.get());EXPECT_EQ(bound.sourceRank(),reverse?1:0);EXPECT_EQ(bound.peerRank(),reverse?0:1);
            EXPECT_EQ(bound.sourceDevices().size(),counts.first);EXPECT_EQ(bound.peerDevices().size(),counts.second);
            for(std::size_t index=0;index<counts.first;++index)EXPECT_EQ(bound.sourceDevices()[index],DeviceId::cuda(int(3*index+2)));
            const auto reversed=bound.reversed();EXPECT_EQ(reversed.sourceRank(),bound.peerRank());EXPECT_EQ(reversed.peerRank(),bound.sourceRank());
            EXPECT_EQ(reversed.sourceDevices(),bound.peerDevices());EXPECT_EQ(reversed.inventory().get(),inventory.get());
            EXPECT_TRUE(bound.localDevices(9).empty());
        }
}
TEST(OriginalMappedNativeCohorts, SameBackendOddUnequalWidthsRetainDistinctOriginalOwners){
    const auto inventory=recordedMappedInventory(DeviceType::ROCm,3,DeviceType::ROCm,5);
    const auto bound=OriginalMappedNativeCohorts::observe(inventory,{DeviceType::ROCm,3},{DeviceType::ROCm,5});
    EXPECT_EQ(bound.sourceRank(),0);EXPECT_EQ(bound.peerRank(),1);
    EXPECT_EQ(bound.sourceDevices().size(),3u);EXPECT_EQ(bound.peerDevices().size(),5u);
    EXPECT_NE(inventory->ranks[bound.sourceRank()].cpu.numa_node,inventory->ranks[bound.peerRank()].cpu.numa_node);
}
TEST(OriginalMappedNativeCohorts, NeverPoolsSameNUMADevicesIntoTwoOriginalRanks){
    auto inventory=recordedMappedInventory(DeviceType::ROCm,9,DeviceType::CUDA,1);
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::ROCm,2},{DeviceType::ROCm,4}),std::runtime_error);
    inventory->ranks[1].gpus.clear();inventory->ranks[1].cpu.numa_node=inventory->ranks[0].cpu.numa_node;
    inventory->ranks[1].gpus=inventory->ranks[0].gpus;
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::ROCm,1},{DeviceType::ROCm,1}),std::runtime_error);
}
TEST(OriginalMappedNativeCohorts, RejectsRemoteUnknownNUMAAndAbsentPhysicalCount){
    for(const int invalid_numa:{-1,99}){auto inventory=recordedMappedInventory(DeviceType::CUDA,3,DeviceType::ROCm,5);
        for(auto& gpu:inventory->ranks[0].gpus)gpu.numa_node=invalid_numa;
        EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,3},{DeviceType::ROCm,5}),std::runtime_error);}
    const auto inventory=recordedMappedInventory(DeviceType::CUDA,3,DeviceType::ROCm,5);
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,4},{DeviceType::ROCm,5}),std::runtime_error);
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,3},{DeviceType::ROCm,6}),std::runtime_error);
}
TEST(OriginalMappedNativeCohorts, RejectsUUIDAliasingAndRelabelledOriginalMembership){
    auto inventory=recordedMappedInventory(DeviceType::CUDA,1,DeviceType::ROCm,1);
    inventory->ranks[1].gpus.front().uuid=inventory->ranks[0].gpus.front().uuid;
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,1},{DeviceType::ROCm,1}),std::invalid_argument);
    inventory=recordedMappedInventory(DeviceType::CUDA,1,DeviceType::ROCm,1);inventory->ranks[1].rank=0;
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,1},{DeviceType::ROCm,1}),std::invalid_argument);
    inventory=recordedMappedInventory(DeviceType::CUDA,1,DeviceType::ROCm,1);inventory->world_size=1;
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,1},{DeviceType::ROCm,1}),std::invalid_argument);
}
TEST(OriginalMappedNativeCohorts, RejectsZeroCountsCPUBackendAndCrossHostChannels){
    const auto inventory=recordedMappedInventory(DeviceType::CUDA,1,DeviceType::ROCm,1);
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,0},{DeviceType::ROCm,1}),std::invalid_argument);
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CPU,1},{DeviceType::ROCm,1}),std::invalid_argument);
    inventory->ranks[1].node_id=8;
    EXPECT_THROW((void)OriginalMappedNativeCohorts::observe(inventory,{DeviceType::CUDA,1},{DeviceType::ROCm,1}),std::runtime_error);
}
