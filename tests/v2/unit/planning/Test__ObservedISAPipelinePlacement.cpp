/**
 * @file Test__ObservedISAPipelinePlacement.cpp
 * @brief Device-free contracts for exact ISA, UUID and original-rank pipeline binding.
 *
 * Fake observation values exercise the pure cold planner, never fake MPI or
 * GPU execution. Arbitrary sparse ordinals and misleading names prove that
 * cardinality cannot replace the original physical architecture/NUMA facts.
 */
#include "../../utils/ObservedISAPipelinePlacement.h"
#include "../../utils/OriginalControllerFabricObservation.h"
#include "../../utils/ObservedNativeReplayOwner.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
using namespace llaminar2;
using namespace llaminar2::test;
namespace {
/** @return Two original NUMA owners observing eight MI50s and an interleaved XTX. */
ClusterInventory observed() {
    ClusterInventory result;result.world_size=2;
    for(int rank=0;rank<2;++rank) {
        RankInventory owner;owner.rank=rank;owner.local_rank=rank;owner.node_id=0;
        owner.hostname="original-host";owner.cpu.numa_node=rank?11:7;
        for(const int ordinal:{9,4,5,3,2,1,0,6,7}) {
            DeviceInfo gpu;gpu.type=DeviceType::ROCm;gpu.local_device_id=ordinal;gpu.numa_node=11;
            gpu.uuid="original-physical-"+std::to_string(ordinal);
            gpu.name=ordinal==5?"misleading MI50":"irrelevant name";
            gpu.native_isa=ordinal==5?"gfx1100":"gfx906:sramecc+:xnack-";
            gpu.pcie_upstream_bdfs={ordinal==9||ordinal==4||ordinal==3||ordinal==2?"island-A":"island-B","host-root"};
            owner.gpus.push_back(std::move(gpu));
        }
        owner.p2p_rocm_count=9;owner.p2p_rocm.assign(81,true);
        result.ranks.push_back(std::move(owner));
    }
    return result;
}
}
/** @test Original ownership and exact ISA survive unequal and multi-stage geometry replay. */
TEST(ObservedISAPipelinePlacement, ArbitraryStagesPreserveOriginalOwnerAndExcludeForeignISA) {
    const auto inventory=observed();
    for(int replay=0;replay<20;++replay)
        for(const auto sizes:{std::vector<std::size_t>{4,4},std::vector<std::size_t>{1,2},
                std::vector<std::size_t>{2,1},std::vector<std::size_t>{1,3,4},std::vector<std::size_t>{3,5}}) {
            const auto selected=ObservedISAPipelinePlacement::require(inventory,DeviceType::ROCm,"gfx906",sizes);
            EXPECT_EQ(selected.discovery_owner,1);
            EXPECT_EQ(selected.selection().discoveryRanks(),std::vector<int>{1});
            ASSERT_EQ(selected.stages.size(),sizes.size());
            std::set<std::string> uuids;
            for(std::size_t stage=0;stage<sizes.size();++stage) {
                EXPECT_EQ(selected.stages[stage].size(),sizes[stage]);
                for(const auto device:selected.stages[stage]) {
                    const auto &gpu=ObservedPeerGrouping(inventory.ranks[1]).device(device);
                    EXPECT_EQ(std::string_view(gpu.native_isa).substr(0,6),"gfx906");
                    EXPECT_EQ(gpu.numa_node,11);
                    EXPECT_TRUE(uuids.insert(gpu.uuid).second);
                    EXPECT_NE(device.ordinal,5);
                }
            }
            if(sizes==std::vector<std::size_t>{4,4})
                for(const auto &stage:selected.stages) {
                    const auto &bridge=ObservedPeerGrouping(inventory.ranks[1]).device(stage.front()).pcie_upstream_bdfs.front();
                    for(const auto device:stage)
                        EXPECT_EQ(ObservedPeerGrouping(inventory.ranks[1]).device(device).pcie_upstream_bdfs.front(),bridge);
                }
        }
}
/** @test Count requests never override physical identity, absent ISA or original NUMA ownership. */
TEST(ObservedISAPipelinePlacement, RejectsMissingForeignDuplicateAndOversizedCohorts) {
    const std::array<std::size_t,2> sizes{4,4};
    for(int defect=0;defect<5;++defect) {
        auto inventory=observed();
        for(auto &rank:inventory.ranks) {
            if(defect==0) rank.gpus.front().native_isa.clear();
            if(defect==1) rank.gpus.front().native_isa="gfx1100";
            if(defect==2) rank.gpus.front().uuid=rank.gpus.back().uuid;
            if(defect==3) rank.gpus.front().numa_node=99;
        }
        if(defect==4) inventory.ranks.pop_back();
        EXPECT_THROW((void)ObservedISAPipelinePlacement::require(inventory,DeviceType::ROCm,"gfx906",sizes),std::invalid_argument);
    }
    const auto inventory=observed();
    EXPECT_THROW((void)ObservedISAPipelinePlacement::require(inventory,DeviceType::ROCm,"gfx906",std::array<std::size_t,2>{4,5}),std::invalid_argument);
    EXPECT_THROW((void)ObservedISAPipelinePlacement::require(inventory,DeviceType::ROCm,"gfx906",std::array<std::size_t,2>{0,8}),std::invalid_argument);
    EXPECT_THROW((void)ObservedISAPipelinePlacement::require(inventory,DeviceType::ROCm,"gfx906",std::array<std::size_t,2>{std::numeric_limits<std::size_t>::max(),1}),std::invalid_argument);
    EXPECT_THROW((void)ObservedISAPipelinePlacement::require(inventory,DeviceType::CPU,"gfx906",sizes),std::invalid_argument);
}

/** @test Original peer admission bypasses edge-free owners before native creation. */
TEST(ObservedNativeReplayOwner, DirectedPeerAdmissionSelectsLaterPhysicalOwnerRepeatedly) {
    ClusterInventory inventory;inventory.world_size=3;
    for(int rank=0;rank<3;++rank) {
        RankInventory owner;owner.rank=rank;owner.cpu.numa_node=rank+7;
        for(int ordinal=0;ordinal<3;++ordinal) {
            DeviceInfo gpu;gpu.type=DeviceType::ROCm;gpu.local_device_id=rank*10+ordinal;
            gpu.numa_node=owner.cpu.numa_node;owner.gpus.push_back(gpu);
        }
        inventory.ranks.push_back(owner);
    }
    const std::array families{NativeReplayFamily{DeviceType::ROCm,3,{}}};
    bool missing=false;
    const auto query=[&](DeviceId destination,DeviceId source)->std::optional<bool> {
        if(missing && destination.ordinal==10 && source.ordinal==12) return std::nullopt;
        return destination.ordinal==12 && source.ordinal==10;
    };
    const auto accepts=[&](const RankInventory &rank) {
        return observedNativeDirectedPeerPair(rank,DeviceType::ROCm,query).has_value();
    };
    for(int replay=0;replay<20;++replay) {
        const auto selected=selectObservedNativeReplayOwner(inventory,families,accepts);
        ASSERT_TRUE(selected);EXPECT_EQ(selected->discovery_owner,1);EXPECT_EQ(selected->devices.size(),3u);
        const auto pair=observedNativeDirectedPeerPair(inventory.ranks[1],DeviceType::ROCm,query);
        ASSERT_TRUE(pair);EXPECT_EQ((*pair)[0].ordinal,10);EXPECT_EQ((*pair)[1].ordinal,12);
    }
    const std::array smaller{NativeReplayFamily{DeviceType::ROCm,2,{}}};
    const auto beyond_prefix=selectObservedNativeReplayOwner(inventory,smaller,accepts);
    ASSERT_TRUE(beyond_prefix);EXPECT_EQ(beyond_prefix->discovery_owner,1);
    EXPECT_EQ(beyond_prefix->devices.size(),2u);
    missing=true;
    EXPECT_THROW(selectObservedNativeReplayOwner(inventory,families,accepts),std::invalid_argument);
    missing=false;
    // Changing physical ownership removes the only edge without borrowing a remote card.
    inventory.ranks[1].gpus[2].numa_node=99;
    EXPECT_FALSE(selectObservedNativeReplayOwner(inventory,families,accepts));
    inventory.ranks[1].gpus[2].numa_node=8;
    const auto no_edges=[](DeviceId,DeviceId)->std::optional<bool>{return false;};
    EXPECT_FALSE(selectObservedNativeReplayOwner(inventory,families,[&](const RankInventory &rank) {
        return observedNativeDirectedPeerPair(rank,DeviceType::ROCm,no_edges).has_value();
    }));
}

namespace {
/**
 * @brief Author complete five-rank observations with original sparse NUMA and GPU identities.
 * @param cuda_count Actual CUDA cohort on rank3/NUMA7, independent of local hardware.
 * @param rocm_count Actual ROCm cohort on rank1/NUMA11; its final card has another ISA.
 * @return Complete original publication, including misleading same-CUDA-owner singleton ROCm visibility.
 */
ClusterInventory crossVendorObserved(std::size_t cuda_count=2,std::size_t rocm_count=4) {
    ClusterInventory result;result.world_size=5;
    std::vector<DeviceInfo> visible;
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm}) {
        const auto count=backend==DeviceType::CUDA?cuda_count:rocm_count;
        for(std::size_t index=0;index<count;++index) {
            DeviceInfo gpu;gpu.type=backend;
            gpu.local_device_id=static_cast<int>(100+index*3);
            gpu.numa_node=backend==DeviceType::CUDA?7:11;
            gpu.uuid=std::string(backend==DeviceType::CUDA?"cuda-":"rocm-")+std::to_string(index);
            gpu.native_isa=backend==DeviceType::CUDA?"sm_80":index+1==count?"gfx1100":"gfx906:sramecc+:xnack-";
            gpu.pcie_upstream_bdfs={"original-bridge","original-root"};visible.push_back(std::move(gpu));
        }
    }
    DeviceInfo small;small.type=DeviceType::ROCm;small.local_device_id=3;small.numa_node=7;
    small.uuid="small-cuda-owner-rocm";small.native_isa="gfx906";visible.push_back(small);
    for(int rank=0;rank<result.world_size;++rank) {
        RankInventory owner;owner.rank=rank;owner.local_rank=rank;owner.node_id=23;
        owner.hostname="original-host";owner.cpu.numa_node=rank==3?7:rank==1?11:20+rank;
        owner.gpus=visible;owner.p2p_cuda_count=static_cast<int>(cuda_count);
        owner.p2p_rocm_count=static_cast<int>(rocm_count+1);
        owner.p2p_cuda.assign(cuda_count*cuda_count,true);
        owner.p2p_rocm.assign((rocm_count+1)*(rocm_count+1),true);
        result.ranks.push_back(std::move(owner));
    }
    return result;
}
/** @brief Check complete observed ownership and disjoint UUID identity in authored stage order. */
void expectOriginalCrossVendorPlacement(const ClusterInventory& inventory,
    std::span<const ObservedNativePipelineStage> request,const ObservedCrossVendorPipelinePlacement& binding) {
    ASSERT_EQ(binding.stages.size(),request.size());ASSERT_EQ(binding.owners.size(),request.size());
    std::set<std::string> selected;
    for(std::size_t stage=0;stage<request.size();++stage) {
        EXPECT_EQ(binding.owners[stage],request[stage].backend==DeviceType::CUDA?3:1);
        const auto& owner=inventory.getRank(binding.owners[stage]);
        EXPECT_EQ(binding.stages[stage].size(),request[stage].width);
        EXPECT_TRUE(ObservedPeerGrouping(owner).sameDomain(binding.stages[stage]));
        for(const auto id:binding.stages[stage]) {
            const auto& gpu=ObservedPeerGrouping(owner).device(id);
            EXPECT_TRUE(rankOwnsGPUDevice(owner,gpu));EXPECT_EQ(id.type,request[stage].backend);
            EXPECT_TRUE(selected.insert(gpu.uuid).second);
            if(!request[stage].native_isa.empty())
                EXPECT_EQ(std::string_view(gpu.native_isa).substr(0,gpu.native_isa.find(':')),request[stage].native_isa);
        }
    }
}
}

/** @test Complete geometry resolves the same-owner ROCm distractor and partitions repeated vendors. */
TEST(ObservedCrossVendorPipelinePlacement, UnevenAndAlternatingStagesKeepOriginalOwnersAndDisjointDevices) {
    auto inventory=crossVendorObserved(2,6);
    for(const bool reverse:{false,true})
        for(const auto widths:{std::vector<std::size_t>{1,2},std::vector<std::size_t>{2,1},
                std::vector<std::size_t>{1,2,1},std::vector<std::size_t>{1,3,1,2}}) {
            // Reversed four-stage geometry authors CUDA demand five across its two stages.
            const auto required_cuda=reverse && widths.size()==4?5u:2u;
            auto observed=crossVendorObserved(required_cuda,6);
            std::vector<ObservedNativePipelineStage> request;
            for(std::size_t stage=0;stage<widths.size();++stage)
                request.push_back({bool(stage%2)==reverse?DeviceType::CUDA:DeviceType::ROCm,widths[stage]});
            const auto binding=ObservedCrossVendorPipelinePlacement::require(observed,request);
            expectOriginalCrossVendorPlacement(observed,request,binding);
            std::reverse(observed.ranks.begin(),observed.ranks.end());
            const auto reordered=ObservedCrossVendorPipelinePlacement::require(observed,request);
            EXPECT_EQ(reordered.owners,binding.owners);EXPECT_EQ(reordered.stages,binding.stages);
        }
    // Observed native transport is still legal without peer edges; the selector
    // prefers genuine links but cannot replace NCCL/RCCL's transport authority.
    for(auto& rank:inventory.ranks) {rank.p2p_cuda.clear();rank.p2p_rocm.clear();}
    const std::array request{ObservedNativePipelineStage{DeviceType::CUDA,1},ObservedNativePipelineStage{DeviceType::ROCm,1}};
    expectOriginalCrossVendorPlacement(inventory,request,ObservedCrossVendorPipelinePlacement::require(inventory,request));
}

/** @test Exact filters and wildcard stages jointly reserve each later stage's only eligible ISA. */
TEST(ObservedCrossVendorPipelinePlacement, ArchitectureFilteredStagesCannotBeConsumedByAnEarlierWildcard) {
    const auto inventory=crossVendorObserved(1,3);
    const std::array request{ObservedNativePipelineStage{DeviceType::CUDA,1,"sm_80"},
        ObservedNativePipelineStage{DeviceType::ROCm,1},ObservedNativePipelineStage{DeviceType::ROCm,2,"gfx906"}};
    const auto binding=ObservedCrossVendorPipelinePlacement::require(inventory,request);
    expectOriginalCrossVendorPlacement(inventory,request,binding);
    ASSERT_EQ(binding.stages[1].size(),1u);
    EXPECT_EQ(ObservedPeerGrouping(inventory.getRank(1)).device(binding.stages[1][0]).native_isa,"gfx1100");
}

/** @test Odd and larger-than-local counts have no participant-mask, rank-order or two-rank ceiling. */
TEST(ObservedCrossVendorPipelinePlacement, OddAndLargeCountsUseObservedGeometryWithoutLocalHardwareCaps) {
    for(const auto counts:{std::pair<std::size_t,std::size_t>{3,5},{5,7},{65,67}}) {
        const auto inventory=crossVendorObserved(counts.first,counts.second);
        const std::array request{ObservedNativePipelineStage{DeviceType::ROCm,counts.second},
            ObservedNativePipelineStage{DeviceType::CUDA,counts.first}};
        expectOriginalCrossVendorPlacement(inventory,request,ObservedCrossVendorPipelinePlacement::require(inventory,request));
    }
}

/** @test Malformed observation and incomplete joint demands fail before native work. */
TEST(ObservedCrossVendorPipelinePlacement, RejectsMissingForeignContradictoryAndAliasedOriginalBindings) {
    const std::array request{ObservedNativePipelineStage{DeviceType::CUDA,2},ObservedNativePipelineStage{DeviceType::ROCm,2}};
    for(int defect=0;defect<9;++defect) {
        auto inventory=crossVendorObserved();
        auto& cuda=inventory.ranks[3];auto& rocm=inventory.ranks[1];
        if(defect==0)inventory.ranks.pop_back();
        if(defect==1)inventory.ranks[4].rank=3;
        if(defect==2)cuda.gpus.front().uuid.clear();
        if(defect==3)cuda.gpus.front().numa_node=99;
        if(defect==4)rocm.hostname="foreign-host";
        if(defect==5)rocm.node_id=99;
        if(defect==6)cuda.gpus.front().local_device_id=-1;
        if(defect==7)cuda.gpus[1].uuid=cuda.gpus[0].uuid;
        if(defect==8)cuda.gpus[1].local_device_id=cuda.gpus[0].local_device_id;
        EXPECT_THROW((void)ObservedCrossVendorPipelinePlacement::require(inventory,request),std::invalid_argument)<<defect;
    }
    auto alias=crossVendorObserved(1,1);
    for(auto& rank:alias.ranks)rank.gpus[1].uuid=rank.gpus[0].uuid;
    const std::array singleton{ObservedNativePipelineStage{DeviceType::CUDA,1},ObservedNativePipelineStage{DeviceType::ROCm,1}};
    EXPECT_THROW((void)ObservedCrossVendorPipelinePlacement::require(alias,singleton),std::invalid_argument);
    const auto inventory=crossVendorObserved();
    for(const auto invalid:{
            std::vector<ObservedNativePipelineStage>{{DeviceType::CUDA,0},{DeviceType::ROCm,1}},
            std::vector<ObservedNativePipelineStage>{{DeviceType::CPU,1},{DeviceType::ROCm,1}},
            std::vector<ObservedNativePipelineStage>{{DeviceType::CUDA,1},{DeviceType::CUDA,1}},
            std::vector<ObservedNativePipelineStage>{{DeviceType::CUDA,1},{DeviceType::ROCm,1,"missing"}},
            std::vector<ObservedNativePipelineStage>{{DeviceType::CUDA,1},{DeviceType::ROCm,1,"gfx906:xnack-"}},
            std::vector<ObservedNativePipelineStage>{{DeviceType::CUDA,2},{DeviceType::ROCm,1},{DeviceType::CUDA,1}},
            std::vector<ObservedNativePipelineStage>{{DeviceType::CUDA,std::numeric_limits<std::size_t>::max()},
                {DeviceType::ROCm,1},{DeviceType::CUDA,1}}})
        EXPECT_THROW((void)ObservedCrossVendorPipelinePlacement::require(inventory,invalid),std::invalid_argument);
}

/** @test Multiple complete owners are ranked by native peer quality and immutable original identity. */
TEST(ObservedCrossVendorPipelinePlacement, MultipleValidOwnersPreferQualityThenStableIdentityAcrossPublicationOrder) {
    auto inventory=crossVendorObserved();
    auto& extra=inventory.ranks[4];
    for(int index=0;index<2;++index) {
        DeviceInfo gpu;gpu.type=DeviceType::ROCm;gpu.local_device_id=700+index;gpu.numa_node=extra.cpu.numa_node;
        gpu.uuid="other-owner-"+std::to_string(index);gpu.native_isa="gfx906";
        gpu.pcie_upstream_bdfs={"other-original-bridge"};extra.gpus.push_back(gpu);
    }
    extra.p2p_rocm_count=7;extra.p2p_rocm.assign(49,true);
    inventory.ranks[1].p2p_rocm.assign(25,false);
    const std::array request{ObservedNativePipelineStage{DeviceType::CUDA,1},ObservedNativePipelineStage{DeviceType::ROCm,2}};
    const auto preferred=ObservedCrossVendorPipelinePlacement::require(inventory,request);
    EXPECT_EQ(preferred.owners,(std::vector<int>{3,4}));
    std::reverse(inventory.ranks.begin(),inventory.ranks.end());
    const auto reordered=ObservedCrossVendorPipelinePlacement::require(inventory,request);
    EXPECT_EQ(reordered.owners,preferred.owners);EXPECT_EQ(reordered.stages,preferred.stages);
    // Equal peer quality is legal too: original physical host/NUMA/rank/UUID
    // order breaks the tie, never the order of the discovery publication.
    for(auto& rank:inventory.ranks) {rank.p2p_rocm.clear();rank.p2p_cuda.clear();}
    const auto tied=ObservedCrossVendorPipelinePlacement::require(inventory,request);
    EXPECT_EQ(tied.owners,(std::vector<int>{3,1}));
    std::reverse(inventory.ranks.begin(),inventory.ranks.end());
    for(auto& rank:inventory.ranks)std::reverse(rank.gpus.begin(),rank.gpus.end());
    const auto shuffled=ObservedCrossVendorPipelinePlacement::require(inventory,request);
    EXPECT_EQ(shuffled.owners,tied.owners);EXPECT_EQ(shuffled.stages,tied.stages);
    // A second owner observing the same UUID with conflicting physical NUMA
    // remains contradictory evidence; it cannot become a lower-quality choice.
    auto contradictory=crossVendorObserved();
    auto gpu=contradictory.ranks[1].gpus[2];gpu.numa_node=contradictory.ranks[4].cpu.numa_node;
    contradictory.ranks[4].gpus.push_back(gpu);
    EXPECT_THROW((void)ObservedCrossVendorPipelinePlacement::require(contradictory,request),std::invalid_argument);
}

/** @test Telemetry borrows the session owner even after another NUMA gains a native singleton. */
TEST(ObservedCrossVendorPipelinePlacement, ControllerFabricSingletonObservationKeepsCompleteDeclaredOwners)
{
    auto inventory=crossVendorObserved(2,4);
    const auto incidental=inventory.getRank(3).gpus.back();
    ASSERT_EQ(incidental.type,DeviceType::ROCm);ASSERT_EQ(incidental.numa_node,7);
    for(auto& rank:inventory.ranks)
        std::erase_if(rank.gpus,[&](const auto& gpu){return gpu.uuid==incidental.uuid;});
    const std::array complete{ObservedNativePipelineStage{DeviceType::CUDA,2},
        ObservedNativePipelineStage{DeviceType::ROCm,4}};
    const auto placement=ObservedCrossVendorPipelinePlacement::require(inventory,complete);
    MoEOverlayDeviceControllerTopology topology;
    for(std::size_t stage=0;stage<complete.size();++stage) {
        const auto& owner=inventory.getRank(placement.owners.at(stage));
        for(const auto device:placement.stages.at(stage)) {
            MoEExpertOwnerParticipant participant;
            participant.participant_id=static_cast<int>(topology.participants.size());
            participant.device=device;participant.world_rank=owner.rank;participant.world_rank_known=true;
            participant.address=GlobalDeviceAddress::fromLocalDeviceId(device,owner.hostname,owner.cpu.numa_node);
            topology.participants.push_back(participant);
        }
    }
    for(const auto stage:{0u,1u}) {
        const auto family=complete[stage].backend;
        const auto selected=originalControllerFabricObservation(inventory,topology,family,1);
        EXPECT_EQ(selected.first,placement.owners.at(stage));ASSERT_EQ(selected.second.size(),1u);
        EXPECT_EQ(selected.second.front().toLocalDeviceId(),placement.stages.at(stage).front());
        EXPECT_THROW((void)originalControllerFabricObservation(inventory,topology,family,0),std::invalid_argument);
        EXPECT_THROW((void)originalControllerFabricObservation(inventory,topology,family,
            static_cast<int>(complete[stage].width+1)),std::invalid_argument);
    }
    // The complete session is already bound. A newly visible legitimate ROCm
    // singleton on the CUDA owner must not change its telemetry participant.
    for(auto& rank:inventory.ranks)rank.gpus.push_back(incidental);
    std::size_t singleton_owners=0;
    for(const auto& rank:inventory.ranks)
        singleton_owners+=std::any_of(rank.gpus.begin(),rank.gpus.end(),[&](const auto& gpu){
            return gpu.type==DeviceType::ROCm && rankOwnsGPUDevice(rank,gpu);
        });
    ASSERT_EQ(singleton_owners,2u);
    EXPECT_EQ(originalControllerFabricObservation(inventory,topology,DeviceType::ROCm,1).first,1);
    // Reversing the canonical rank-indexed roster supplies a foreign owner
    // record for the retained participant. The observation must reject it.
    std::reverse(inventory.ranks.begin(),inventory.ranks.end());
    EXPECT_THROW((void)originalControllerFabricObservation(
        inventory,topology,DeviceType::ROCm,1),std::invalid_argument);
    std::reverse(inventory.ranks.begin(),inventory.ranks.end());
    // ClusterInventory::getRank uses the canonical rank-indexed roster. GPU
    // publication order may vary; moving rank records to another rank's index
    // would forge the original owner lookup instead of testing observation.
    for(auto& rank:inventory.ranks) {
        ASSERT_EQ(&inventory.getRank(rank.rank),&rank);
        std::reverse(rank.gpus.begin(),rank.gpus.end());
    }
    EXPECT_EQ(originalControllerFabricObservation(inventory,topology,DeviceType::ROCm,1).first,1);
    auto foreign=topology;foreign.participants.back().world_rank=3;
    EXPECT_THROW((void)originalControllerFabricObservation(inventory,foreign,DeviceType::ROCm,1),std::invalid_argument);
    auto alias=topology;alias.participants.push_back(alias.participants.back());
    EXPECT_THROW((void)originalControllerFabricObservation(inventory,alias,DeviceType::ROCm,1),std::invalid_argument);
}
