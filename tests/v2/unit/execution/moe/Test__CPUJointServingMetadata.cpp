/**
 * @file Test__CPUJointServingMetadata.cpp
 * @brief Actual source-map and immutable channel-role replay over CPU TP1..8.
 *
 * Declared NUMA endpoints below are pure metadata, not native-ready contexts.
 * Native qualification uses one rank per physical socket. This host proves two
 * sockets; four/eight-socket execution remains unqualified here.
 */
#include "execution/moe/MoECPUExpertShardReturnRole.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include "execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include "execution/moe/MoEOverlaySparseCollective.h"
#include "utils/PerfStatsCollector.h"
#include "execution/moe/CPUExpertShardSourceSelection.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "tensors/TensorClasses.h"
#include "utils/CPUForwardSamplingTestFixture.h"
#include "execution/local_execution/device/DeviceContext.h"
#include <numeric>
#include <gtest/gtest.h>
#include <array>
#include <cstdlib>
using namespace llaminar2;
namespace {
/** @brief Declare original CPU tensor membership without claiming native readiness. */
MoERoutedExpertPlacementPlan declaration(int participants) {
    MoERoutedExpertPlacementPlan plan;plan.enabled=true;plan.first_model_layer=7;
    plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
    plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="original_cpu";
    RoutedExpertDomain domain;domain.name="original_cpu";domain.scope=ExecutionDomainScope::NODE_LOCAL;
    domain.backend=CollectiveBackendType::UPI;domain.owner_rank=-1;
    domain.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    domain.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
    domain.routed_decode_assignment_policy=domain.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
    for(int p=0;p<participants;++p){domain.participants.push_back(GlobalDeviceAddress::cpu(p));domain.world_ranks.push_back(p);}
    plan.domains={domain};plan.routed_tiers={{.name="actual_cpu",.domain=domain.name,.priority=0,.max_experts_per_layer=4,.fallback=true}};
    plan.placements={{.layer=7,.routed_expert_tier=std::vector<int>(4,0)}};
    return plan;
}
/** @brief Preserve the source-native projection geometry used by metadata replay. */
MoEOverlayLayerWeightManifest source() {
    const auto native=ExpertWeightFormat::floating(TensorType::FP32);
    return {7,{{{ExpertTierWeightProjection::Gate,2048,256,native},
        {ExpertTierWeightProjection::Up,2048,256,native},{ExpertTierWeightProjection::Down,256,2048,native}}}};
}
}
/** @test TP1 preserves the explicit tensor policy and its ordinary full-matrix physical/ownership path. */
TEST(CPUJointServingMetadata, ExplicitTensorTP1UsesOriginalWholeMatrixWithoutJointAuthority) {
    auto plan=declaration(1);EXPECT_FALSE(MoEExpertOwnerMap::usesCPUJointFragments(plan));
    const auto whole=MoEExpertOwnerMap::build(plan);
    const auto explicit_source=MoEExpertOwnerMap::buildShardPlacement(plan,{source()});
    EXPECT_EQ(plan.domains.front().routed_compute_policy,RoutedExpertComputePolicy::TensorSharded);
    auto apportioned=plan;apportioned.domains.front().routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    EXPECT_EQ(whole.owners(),MoEExpertOwnerMap::build(apportioned).owners());
    EXPECT_EQ(explicit_source.owners(),whole.owners());
    for(int expert=0;expert<4;++expert) {
        const auto* actual=whole.ownerFor(7,expert);ASSERT_NE(actual,nullptr);
        EXPECT_FALSE(actual->shard_cohort);EXPECT_EQ(whole.fragmentCountForExpert(7,expert),1u);
    }
    const std::array<int32_t,1> layers{7};
    EXPECT_EQ(originalCPUExpertShardReturnRole(&whole,{.world_rank=0,.participant_id=0},layers),MoEOverlayActivationReturnRole::Ordinary);
    auto unused=declaration(2).domains.front();unused.name="unused_joint_domain";plan.domains.push_back(unused);
    EXPECT_FALSE(MoEExpertOwnerMap::usesCPUJointFragments(plan));
}
/** @test Every2..8 source member computes; exactly original cohort ordinal0 publishes the combined payload. */
TEST(CPUJointServingMetadata, OriginalPopulationAndReturnRolesThroughEight) {
    const std::array<int32_t,1> layers{7};
    for(int population=2;population<=8;++population)for(int replay=0;replay<20;++replay) {
        auto plan=declaration(population);EXPECT_TRUE(MoEExpertOwnerMap::usesCPUJointFragments(plan));
        auto no_quota=plan;no_quota.placements.clear();EXPECT_TRUE(MoEExpertOwnerMap::usesCPUJointFragments(no_quota));
        const auto actual=MoEExpertOwnerMap::buildShardPlacement(plan,{source()});
        const auto* first=actual.placementFor(7,0);ASSERT_TRUE(first);ASSERT_TRUE(first->shard_cohort);
        ASSERT_EQ(first->shard_cohort->participants().size(),size_t(population));
        for(int member=0;member<population;++member) {
            EXPECT_EQ(actual.expertsForParticipant(7,member),(std::vector<int>{0,1,2,3}));
            EXPECT_EQ(actual.fragmentCountForExpert(7,member%4),size_t(population));
            const auto role=originalCPUExpertShardReturnRole(&actual,{.world_rank=member,.participant_id=member},layers);
            EXPECT_EQ(role,member==0?MoEOverlayActivationReturnRole::StaticJointRoot:MoEOverlayActivationReturnRole::StaticJointContributor);
            EXPECT_TRUE(moeOverlayActivationReturnCountsMatch(uint32_t(role),member==0?3:0,member==0?7:0,3,7));
            EXPECT_FALSE(moeOverlayActivationReturnCountsMatch(uint32_t(role),member==0?0:3,member==0?0:7,3,7));
        }
        EXPECT_THROW(originalCPUExpertShardReturnRole(&actual,{.world_rank=91,.participant_id=0},layers),std::invalid_argument);
        const std::array<int32_t,2> duplicate{7,7},missing_before{6,7},missing_after{7,8};
        for(const auto& invalid:{duplicate,missing_before,missing_after})
            EXPECT_THROW(originalCPUExpertShardReturnRole(&actual,{.world_rank=0,.participant_id=0},invalid),std::invalid_argument);
    }
    for(const uint32_t invalid:{3u,256u,257u,UINT32_MAX}) {
        EXPECT_FALSE(moeOverlayActivationReturnRoleValid(invalid));
        EXPECT_FALSE(moeOverlayActivationReturnCountsMatch(invalid,0,0,0,0));
    }
    EXPECT_FALSE(moeOverlayActivationReturnCountsMatch(0,0,0,3,7)); // Ordinary contract is unchanged.
}

namespace {
/** @brief Record the actual stage call then fail before any transport/output publication. */
class RootExtentRecorder final : public llaminar2::test::ScriptedGlobalTPContext {
public:
    /** @brief Bind the original participant population and one source ordinal. */
    RootExtentRecorder(int population,int member,bool complete_empty=false):population_(population),member_(member),ranks_(population),complete_empty_(complete_empty) {
        std::iota(ranks_.begin(),ranks_.end(),0);
    }
    /** @brief Return the original collective population. */
    int degree() const override{return population_;}
    /** @brief Return this original source ordinal. */
    int myIndex() const override{return member_;}
    /** @brief Preserve the production intra-node collective scope. */
    TPScope scope() const override{return TPScope::NODE_LOCAL;}
    /** @brief Return original MPI membership in reduction order. */
    const std::vector<int>& worldRanks() const override{return ranks_;}
    /** @brief Return this source's original CPU/NUMA endpoint. */
    GlobalDeviceAddress localDevice() const override{return GlobalDeviceAddress::cpu(member_);}
    /** @brief Record live send and common receive extents, then reject before transport. */
    bool gatherVariableFloatRecordsToRoot(const float*,size_t count,float*,size_t capacity,
        size_t width,int root,size_t& gathered,const std::string&,std::span<RootedFloatRecordRange>) override {
        ++calls;observed_count=count;observed_capacity=capacity;observed_width=width;observed_root=root;
        gathered=0;return complete_empty_; // Scripted empty-record progress is not native/ticket readiness.
    }
    int calls=0,observed_root=-1;size_t observed_count=0,observed_capacity=0,observed_width=0;
private:
    int population_,member_;std::vector<int> ranks_;bool complete_empty_;
};
}
/** @test Asymmetric root/contributor storage cannot alter the original collective extent or live sends. */
TEST(CPUJointServingMetadata, ActualStageSharesRootExtentAcrossAsymmetricParticipantBanks) {
    constexpr int H=2560,K=8,M=1;
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const MoEOverlayLayerWeightManifest native{7,{{{ExpertTierWeightProjection::Gate,2048,H,format},
        {ExpertTierWeightProjection::Up,2048,H,format},{ExpertTierWeightProjection::Down,H,2048,format}}}};
    auto context=IDeviceContext::create(DeviceId::cpu());
    for(int population=2;population<=8;++population) {
        const auto plan=declaration(population);
        const auto map=MoEExpertOwnerMap::buildShardPlacement(plan,{native});
        for(int member=0;member<population;++member)for(size_t live:{size_t(K),size_t(1),size_t(0)}) {
            auto selection=CPUExpertShardSourceSelection::fromOriginalPlacement(map,7,member);
            RootExtentRecorder tp(population,member);
            const size_t local_capacity=member==0?size_t(population*K):size_t(K);
            FP32Tensor packed({local_capacity*(H+1)+1}),indices({M,K}),weights({M,K}),combined({M*K,H});
            std::fill_n(combined.mutable_data(),combined.numel(),-991.0f);
            std::vector<RootedFloatRecordRange> ranges(member==0?population:0);
            std::vector<int32_t> maps(population*K);
            std::vector<float> required(K);
            MoECPUExpertShardRouteStage::Params parameters{
                selection,&tp,&packed,&indices,&weights,member==0?&combined:nullptr,
                ranges,maps,required,M,K,H,0,"record_original_extent"};
            EXPECT_THROW(MoECPUExpertShardRouteStage{parameters},std::invalid_argument);
            parameters.standalone_execution_phase=ExpertHistogramSource::DecodeToken;
            ASSERT_TRUE(canonical_moe_route_record::writeRecordCount(packed.mutable_data(),packed.numel(),live));
            MoECPUExpertShardRouteStage stage(parameters);
            EXPECT_FALSE(stage.execute(context.get()));
            ASSERT_EQ(tp.calls,1);EXPECT_EQ(tp.observed_count,live);
            EXPECT_EQ(tp.observed_capacity,size_t(population*K));
            EXPECT_EQ(tp.observed_width,size_t(H+1));EXPECT_EQ(tp.observed_root,0);
            EXPECT_TRUE(std::all_of(combined.data(),combined.data()+combined.numel(),[](float v){return v==-991.0f;}));
            if(member==0) {
                FP32Tensor undersized({size_t(K*(H+1)+1)});
                parameters.packed_records=&undersized;
                EXPECT_THROW(MoECPUExpertShardRouteStage{parameters},std::invalid_argument);
            }
        }
    }
}

/** @test Phase comes from the original sparse transaction, even for empty padded decode or one-row prefill. */
TEST(CPUJointServingMetadata, ActualStagePublishesOriginalSparsePhase) {
    struct ObservationScope {
        std::optional<std::string> summary,filter;
        /** @brief Retain caller configuration and enable isolated passive counter observations. */
        ObservationScope() {
            if(const auto* value=std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))summary=value;
            if(const auto* value=std::getenv("LLAMINAR_PERF_STATS_FILTER"))filter=value;
            setenv("LLAMINAR_PERF_STATS_SUMMARY","1",1);
            setenv("LLAMINAR_PERF_STATS_FILTER","moe_canonical_publication",1);
            PerfStatsCollector::reloadConfigurationFromEnvironment();PerfStatsCollector::reset();
        }
        /** @brief Restore the original environment after success or a failed assertion. */
        ~ObservationScope() {
            if(summary)setenv("LLAMINAR_PERF_STATS_SUMMARY",summary->c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            if(filter)setenv("LLAMINAR_PERF_STATS_FILTER",filter->c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_FILTER");
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
    } observation;
    constexpr int H=7,K=1;
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const MoEOverlayLayerWeightManifest native{7,{{{ExpertTierWeightProjection::Gate,64,H,format},
        {ExpertTierWeightProjection::Up,64,H,format},{ExpertTierWeightProjection::Down,H,64,format}}}};
    const auto placement=MoEExpertOwnerMap::buildShardPlacement(declaration(2),{native});
    const auto selected=CPUExpertShardSourceSelection::fromOriginalPlacement(placement,7,1);
    auto context=IDeviceContext::create(DeviceId::cpu());
    for(const auto [capacity,phase]:std::array<std::pair<int,ExpertHistogramSource>,4>{{
        {64,ExpertHistogramSource::DecodeToken},{1,ExpertHistogramSource::PrefillChunk},
        {64,ExpertHistogramSource::GroupedVerifier},{9,ExpertHistogramSource::GroupedVerifier}}}) {
        RootExtentRecorder tp(2,1,true);
        FP32Tensor packed({size_t(capacity*(H+1)+1)});
        ASSERT_TRUE(canonical_moe_route_record::writeRecordCount(packed.mutable_data(),packed.numel(),0));
        std::vector<int32_t> slots(capacity),experts(capacity);std::vector<float> weights(capacity);
        MoEOverlaySparseRows sparse;sparse.key=phase==ExpertHistogramSource::GroupedVerifier
            ?makeMTPMoEOverlayCollectiveKey(1,1,capacity==64?0:3,7,0,0,1,MoEOverlayCollectiveDirection::Dispatch)
            :makeMoEOverlayCollectiveKey(1,1,7,0,0,1,MoEOverlayCollectiveDirection::Dispatch);
        sparse.key.histogram_source=phase;sparse.target_participant=1;sparse.d_model=H;sparse.top_k=K;
        sparse.entry_capacity=capacity;sparse.original_route_slots_host=slots.data();
        sparse.expert_ids_host=experts.data();sparse.route_weights_host=weights.data();
        MoEOverlayReturnRows output;
        MoECPUExpertShardRouteStage::Params params;
        params.original_source=selected;params.tp=&tp;params.packed_records=&packed;
        params.rows=capacity;params.top_k=K;params.columns=H;params.root_participant=0;
        params.stage_name="phase_contract_"+std::to_string(capacity)+"_"+std::to_string(int(phase));
        params.original_sparse_routes=&sparse;params.original_return_rows=&output;
        params.standalone_execution_phase=phase;
        EXPECT_THROW(MoECPUExpertShardRouteStage{params},std::invalid_argument);
        params.standalone_execution_phase.reset();
        MoECPUExpertShardRouteStage stage(params);
        ASSERT_TRUE(stage.execute(context.get()));EXPECT_EQ(tp.calls,1);
        const char *expected=phase==ExpertHistogramSource::DecodeToken?"decode":
            phase==ExpertHistogramSource::PrefillChunk?"prefill":"grouped_verifier";
        const auto records=PerfStatsCollector::snapshot({"moe_canonical_publication"});
        int matches=0;
        for(const auto &record:records)if(record.tags.contains("stage") && record.tags.at("stage")==params.stage_name) {
            EXPECT_EQ(record.phase,expected);
            if(record.name=="cpu_packed_route_gather_calls"){++matches;EXPECT_EQ(record.value,1.0);}
            if(record.name=="cpu_packed_route_gather_local_bytes")EXPECT_EQ(record.value,0.0);
        }
        EXPECT_EQ(matches,1);
        for(const auto invalid:{ExpertHistogramSource::SyntheticTest,static_cast<ExpertHistogramSource>(99)}) {
            RootExtentRecorder rejected_tp(2,1,true);params.tp=&rejected_tp;
            sparse.key.histogram_source=invalid;
            MoECPUExpertShardRouteStage rejected(params);
            EXPECT_FALSE(rejected.execute(context.get()));EXPECT_EQ(rejected_tp.calls,0);
        }
    }
}
