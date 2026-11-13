/**
 * @file Test__PlanningPCIeCommunicationCost.cpp
 * @brief Device-free replay of observed PCIe cut geometry and exact live-byte bounds.
 *
 * Synthetic negotiated links prove wire-capacity lower bounds only. No test
 * treats these capacities as measured peer service or collective throughput.
 */
#include "planning/PlanningPCIeCommunicationCost.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstdio>
#include <limits>

using namespace llaminar2;
namespace {
    /** @return Canonical distinct endpoint BDFs; bank bridges are independent observed ancestors. */
    std::string endpointBDF(size_t ordinal)
    {
        char value[32];std::snprintf(value,sizeof(value),"0000:%02x:00.0",unsigned(0x20+ordinal));return value;
    }
    /** @return One original-rank GPU inventory with two explicitly observed shared uplinks. */
    RankInventory cutInventory(size_t count,size_t first_bank)
    {
        RankInventory rank;rank.rank=0;rank.node_id=0;rank.local_rank=0;rank.hostname="cut-fixture";
        rank.cpu.type=DeviceType::CPU;rank.cpu.numa_node=0;
        for(size_t i=0;i<count;++i){DeviceInfo gpu;gpu.type=DeviceType::ROCm;gpu.local_device_id=int(i);
            gpu.name="Recorded GPU";gpu.uuid="cut-fixture-"+std::to_string(i);gpu.numa_node=0;
            gpu.pci_bus_address=endpointBDF(i);gpu.pcie_bottleneck_bdf=i<first_bank?"0000:10:00.0":"0000:11:00.0";
            gpu.pcie_upstream_bdfs={gpu.pcie_bottleneck_bdf,"0000:01:00.0"};
            gpu.pcie_width=8;gpu.pcie_speed_gts=8;gpu.pcie_gen=3;rank.gpus.push_back(std::move(gpu));}
        return rank;
    }
    /** @return Explicitly ordered native participants, with no cardinality or topology whitelist. */
    std::vector<DeviceId> cutDevices(size_t count)
    {std::vector<DeviceId> devices;for(size_t i=0;i<count;++i)devices.push_back(DeviceId::rocm(int(i)));return devices;}
    /** @return PCIe3 x8 encoded wire capacity, excluding any invented protocol efficiency factor. */
    double cutCapacity() {return 8.e9*8.*(128./130.)/8.;}
}

/** @test A shared uplink is crossed exactly on source/destination XOR, never by a same-bank edge. */
TEST(PlanningPCIeCommunicationCost, SameBankDoesNotCrossUplinkAndPartialBytesRemainLive)
{
    const auto rank=cutInventory(8,4);const PlanningPCIeCommunicationCost cost(rank);
    const std::array local{PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::rocm(3),17}};
    const auto same=cost.transfers(local);EXPECT_DOUBLE_EQ(same.seconds,0.);
    EXPECT_EQ(same.known_cuts,0u);EXPECT_EQ(same.unknown_cuts,0u);
    const std::array cross{PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::rocm(4),17}};
    const auto bound=cost.transfers(cross);EXPECT_DOUBLE_EQ(bound.seconds,17./cutCapacity());
    EXPECT_EQ(bound.known_cuts,2u);EXPECT_EQ(bound.unknown_cuts,0u);
    for(const auto &cut:bound.cuts){EXPECT_EQ(cut.outbound_bytes+cut.inbound_bytes,17u);ASSERT_TRUE(cut.capacity_bytes_per_second);
        EXPECT_DOUBLE_EQ(*cut.capacity_bytes_per_second,cutCapacity());}
}

/** @test Simultaneous users sum on each directed cut; full duplex takes the maximum direction. */
TEST(PlanningPCIeCommunicationCost, SharedDirectedLinkSumsAndOppositeDirectionsAreFullDuplex)
{
    const PlanningPCIeCommunicationCost cost(cutInventory(8,4));
    const std::array outgoing{PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::rocm(4),100},
        PlanningPCIeDirectedTransfer{DeviceId::rocm(1),DeviceId::rocm(5),200}};
    const auto one_way=cost.transfers(outgoing);EXPECT_DOUBLE_EQ(one_way.seconds,300./cutCapacity());
    const std::array duplex{outgoing[0],outgoing[1],PlanningPCIeDirectedTransfer{DeviceId::rocm(6),DeviceId::rocm(2),250}};
    const auto both=cost.transfers(duplex);EXPECT_DOUBLE_EQ(both.seconds,300./cutCapacity());
    for(const auto &cut:both.cuts){EXPECT_EQ(std::max(cut.outbound_bytes,cut.inbound_bytes),300u);
        EXPECT_EQ(std::min(cut.outbound_bytes,cut.inbound_bytes),250u);}
}

/** @test Allreduce needs one live payload in each direction of a nontrivial cut, independent of split counts. */
TEST(PlanningPCIeCommunicationCost, AllreduceOnePayloadPerCutSupportsUnequalAndLargeGroups)
{
    for(const size_t count:{1u,3u,5u,8u,9u}){
        const auto devices=cutDevices(count);const PlanningPCIeCommunicationCost cost(cutInventory(count,count==1?1:count/2));
        const auto bound=cost.allreduce(devices,113);
        EXPECT_DOUBLE_EQ(bound.seconds,count==1?0.:113./cutCapacity());
        for(const auto &cut:bound.cuts){EXPECT_EQ(cut.outbound_bytes,113u);EXPECT_EQ(cut.inbound_bytes,113u);}
    }
    for(const size_t split:{2u,3u}){
        const PlanningPCIeCommunicationCost cost(cutInventory(8,split));
        const auto bound=cost.allreduce(cutDevices(8),1027);
        EXPECT_DOUBLE_EQ(bound.seconds,1027./cutCapacity());
        for(const auto &cut:bound.cuts){EXPECT_EQ(cut.outbound_bytes,1027u);EXPECT_EQ(cut.inbound_bytes,1027u);}
    }
}

/** @test Unique owner fragments cross a shared cut once each, including skew and empty participants. */
TEST(PlanningPCIeCommunicationCost, AllgatherChargesUniqueOwnerFragmentsWithoutConsumerFanout)
{
    for(const size_t count:{1u,3u,5u,8u,9u}){
        const size_t split=count==1?1:count/2;const auto devices=cutDevices(count);
        const PlanningPCIeCommunicationCost cost(cutInventory(count,split));
        std::vector<size_t> owners;size_t inside=0,outside=0;
        for(size_t i=0;i<count;++i){owners.push_back(i+1);(i<split?inside:outside)+=i+1;}
        const auto bound=cost.allgather(devices,owners);
        EXPECT_DOUBLE_EQ(bound.seconds,count==1?0.:double(std::max(inside,outside))/cutCapacity());
        for(const auto &cut:bound.cuts){EXPECT_EQ(std::max(cut.outbound_bytes,cut.inbound_bytes),std::max(inside,outside));
            EXPECT_EQ(cut.outbound_bytes+cut.inbound_bytes,inside+outside);}
    }
    const PlanningPCIeCommunicationCost cost(cutInventory(8,4));
    const std::array<size_t,8> skew{0,0,1,2,0,0,0,7};
    const auto partial=cost.allgather(cutDevices(8),skew);EXPECT_DOUBLE_EQ(partial.seconds,7./cutCapacity());
    for(const auto &cut:partial.cuts){EXPECT_EQ(std::max(cut.outbound_bytes,cut.inbound_bytes),7u);
        EXPECT_EQ(std::min(cut.outbound_bytes,cut.inbound_bytes),3u);}
    const std::array<size_t,2> short_shape{1,2};
    EXPECT_THROW(cost.allgather(cutDevices(8),short_shape),std::invalid_argument);
    const std::array<size_t,8> zeros{};
    EXPECT_THROW(cost.allgather(cutDevices(8),zeros),std::invalid_argument);
    const std::array<size_t,8> overflow{std::numeric_limits<size_t>::max(),1,0,0,0,0,0,0};
    EXPECT_THROW(cost.allgather(cutDevices(8),overflow),std::overflow_error);
}

/** @test Only observed negotiated lanes and supported encoding define capacity; capability maxima never replace them. */
TEST(PlanningPCIeCommunicationCost, NegotiatedWidthAndSpeedChangeThePhysicalBound)
{
    const auto devices=cutDevices(4);auto rank=cutInventory(4,2);
    const PlanningPCIeCommunicationCost frozen(rank);
    const auto original=frozen.allreduce(devices,4096);
    for(auto &gpu:rank.gpus){gpu.pcie_width=4;gpu.pcie_max_width=32;gpu.pcie_max_speed_gts=32;}
    EXPECT_DOUBLE_EQ(PlanningPCIeCommunicationCost(rank).allreduce(devices,4096).seconds,2*original.seconds);
    EXPECT_DOUBLE_EQ(frozen.allreduce(devices,4096).seconds,original.seconds);
    for(auto &gpu:rank.gpus)gpu.pcie_speed_gts=16;
    EXPECT_DOUBLE_EQ(PlanningPCIeCommunicationCost(rank).allreduce(devices,4096).seconds,original.seconds);
    for(auto &gpu:rank.gpus){gpu.pcie_width=8;gpu.pcie_speed_gts=5;}
    EXPECT_DOUBLE_EQ(PlanningPCIeCommunicationCost(rank).allreduce(devices,4096).seconds,4096./(5.e9*8.*.8/8.));
}

/** @test Missing or inconsistent original facts remain unknown instead of receiving fabricated bandwidth. */
TEST(PlanningPCIeCommunicationCost, MissingConflictingAndUnknownRatesCannotInventCapacity)
{
    for(int defect=0;defect<6;++defect){SCOPED_TRACE(defect);auto rank=cutInventory(4,2);
        if(defect==0)for(auto &gpu:rank.gpus)gpu.pcie_speed_gts=9;
        if(defect==1)for(auto &gpu:rank.gpus)gpu.pcie_width=0;
        if(defect==2)for(auto &gpu:rank.gpus)gpu.pcie_upstream_bdfs.clear();
        if(defect==3){rank.gpus[0].pcie_speed_gts=16;rank.gpus[2].pcie_width=4;}
        if(defect==4)for(auto &gpu:rank.gpus)gpu.pcie_bottleneck_bdf.clear();
        if(defect==5)for(auto &gpu:rank.gpus)gpu.pcie_speed_gts=std::numeric_limits<double>::quiet_NaN();
        const auto bound=PlanningPCIeCommunicationCost(rank).allreduce(cutDevices(4),71);
        EXPECT_DOUBLE_EQ(bound.seconds,0.);EXPECT_FALSE(bound.evidence.empty());
        if(defect==2 || defect==4)EXPECT_FALSE(bound.topology_complete);
        else EXPECT_GT(bound.unknown_cuts,0u);
        for(const auto &cut:bound.cuts)EXPECT_FALSE(cut.capacity_bytes_per_second.has_value());
    }
}

/** @test Foreign, duplicate, mixed-backend, empty and zero/self traffic cannot enter a native cut quote. */
TEST(PlanningPCIeCommunicationCost, RejectsMalformedForeignOrMixedTraffic)
{
    auto rank=cutInventory(4,2);auto cuda=rank.gpus[0];cuda.type=DeviceType::CUDA;cuda.uuid="cuda-foreign-class";rank.gpus.push_back(cuda);
    const PlanningPCIeCommunicationCost cost(rank);const std::vector<DeviceId> empty;
    EXPECT_THROW(cost.allreduce(empty,1),std::invalid_argument);
    EXPECT_THROW(cost.allreduce(cutDevices(4),0),std::invalid_argument);
    const std::array duplicate{DeviceId::rocm(0),DeviceId::rocm(0)};
    EXPECT_THROW(cost.allreduce(duplicate,1),std::invalid_argument);
    const std::array mixed{DeviceId::rocm(0),DeviceId::cuda(0)};
    EXPECT_THROW(cost.allreduce(mixed,1),std::invalid_argument);
    const std::array cpu{DeviceId::cpu()};
    EXPECT_THROW(cost.allreduce(cpu,1),std::invalid_argument);
    const std::array foreign{DeviceId::rocm(99)};
    EXPECT_THROW(cost.allreduce(foreign,1),std::invalid_argument);
    EXPECT_DOUBLE_EQ(cost.transfers({}).seconds,0.);
    const std::array overflow{PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::rocm(2),std::numeric_limits<size_t>::max()},
        PlanningPCIeDirectedTransfer{DeviceId::rocm(1),DeviceId::rocm(3),1}};
    EXPECT_THROW(cost.transfers(overflow),std::overflow_error);
    auto remote=rank;remote.gpus[0].numa_node=1;
    const PlanningPCIeCommunicationCost remote_cost(remote);
    const std::array unowned{DeviceId::rocm(0)};
    EXPECT_THROW(remote_cost.allreduce(unowned,1),std::invalid_argument);
    auto duplicate_rank=rank;duplicate_rank.gpus.push_back(rank.gpus[0]);
    EXPECT_THROW((void)PlanningPCIeCommunicationCost(duplicate_rank),std::invalid_argument);
    auto invalid_rank=rank;invalid_rank.rank=-1;
    EXPECT_THROW((void)PlanningPCIeCommunicationCost(invalid_rank),std::invalid_argument);
    for(const auto edge:{PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::rocm(0),1},
        PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::rocm(1),0},
        PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::cuda(0),1},
        PlanningPCIeDirectedTransfer{DeviceId::rocm(0),DeviceId::rocm(99),1}}){
        EXPECT_THROW(cost.transfers(std::span(&edge,1)),std::invalid_argument);
    }
}
