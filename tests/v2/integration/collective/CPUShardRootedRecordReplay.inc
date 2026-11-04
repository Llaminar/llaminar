/**
 * @file CPUShardRootedRecordReplay.inc
 * @brief Actual original rooted communicator source boundaries and live counts.
 */
TEST_F(Test__GlobalTPContext_MPI, CPUShardSourceRangesPreserveLivePrefixes)
{
    auto ctx=createTwoRankContext(1843);
    ASSERT_NE(ctx,nullptr);
    constexpr size_t width=4,capacity=16;
    std::array<float,capacity*width> records;
    std::array<RootedFloatRecordRange,2> ranges;
    // Nonzero root prevents any accidental conflation of root and participant0.
    constexpr int root=1;
    for(size_t replay=0;replay<20;++replay) {
        const size_t count=replay%4==0?0:replay%4==1?size_t(world_rank_==0?7:1):
                           replay%4==2?size_t(world_rank_==0?0:2):1;
        records.fill(-777);
        for(size_t record=0;record<count;++record)for(size_t column=0;column<width;++column)
            records[record*width+column]=float(world_rank_*100+record*width+column);
        size_t gathered=99;
        auto rangeOutput=world_rank_==root?std::span(ranges):std::span<RootedFloatRecordRange>{};
        ASSERT_TRUE(ctx->gatherVariableFloatRecordsToRoot(records.data(),count,records.data(),capacity,
            width,root,gathered,"cpu_shard_source_ranges",rangeOutput));
        if(world_rank_==root) {
            if(!validRootedFloatRecordRanges(ranges,gathered)) {
                ADD_FAILURE()<<"Original rooted gather did not preserve complete source boundaries";
                continue;
            }
            EXPECT_EQ(ranges[root].first_record,0);
            EXPECT_EQ(ranges[root].record_count,count);
            for(const auto& range:ranges)
                for(size_t record=0;record<range.record_count;++record)for(size_t column=0;column<width;++column)
                    EXPECT_EQ(records[(range.first_record+record)*width+column],
                              float(range.participant*100+record*width+column));
            EXPECT_EQ(records[gathered*width],-777);
        } else EXPECT_EQ(gathered,0);
    }
}

#include "execution/compute_stages/stages/MoECPUExpertShardRouteStage.h"
#include "execution/moe/CPUExpertShardSourceSelection.h"
#include "execution/moe/MoEExpertShardCohort.h"
#include "execution/moe/CanonicalMoERouteRecord.h"
#include "utils/PerfStatsCollector.h"
#include <cstdlib>
#include <optional>
/** @test Real rooted transport and production shard stage use the exact original rank/NUMA cohort. */
TEST_F(Test__GlobalTPContext_MPI, CPUShardStageUsesOriginalCohortAndUnweightedRows) {
    const auto inventory=MPIContextFactory::global()->clusterInventory();
    const auto &first=inventory->ranks.at(0),&second=inventory->ranks.at(1);
    ASSERT_NE(first.cpu.numa_node,second.cpu.numa_node)<<"Canonical CPU2 test requires original socket placement";
    MoERoutedExpertPlacementPlan plan;
    plan.enabled=true;plan.first_model_layer=3;plan.topology=RoutedExpertPlacementTopology::TieredOverlay;
    plan.continuation_domain=plan.base_model_domain=plan.shared_expert_domain="original_cpu_joint";
    RoutedExpertDomain domain;domain.name=plan.continuation_domain;domain.scope=ExecutionDomainScope::NODE_LOCAL;
    domain.backend=CollectiveBackendType::UPI;domain.owner_rank=-1;domain.world_ranks={0,1};
    domain.participants={GlobalDeviceAddress::cpu(first.cpu.numa_node,first.hostname),
                         GlobalDeviceAddress::cpu(second.cpu.numa_node,second.hostname)};
    domain.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    domain.routed_phase_policy=RoutedExpertPhasePolicy::Uniform;
    domain.routed_decode_assignment_policy=domain.routed_prefill_assignment_policy=RoutedExpertAssignmentPolicy::StaticOwner;
    plan.domains={domain};
    plan.routed_tiers={{.name="original_joint",.domain=domain.name,.priority=0,.max_experts_per_layer=4,.fallback=true}};
    plan.placements={{.layer=3,.routed_expert_tier=std::vector<int>(4,0)}};
    // The local contributor bank is below the UPI latency boundary while
    // the declared root receive extent is above it. All members must still
    // select the same original transport; live record counts stay compact.
    constexpr int H=2560,K=8;
    const auto format=ExpertWeightFormat::floating(TensorType::FP32);
    const MoEOverlayLayerWeightManifest source{3,{{{ExpertTierWeightProjection::Gate,512,H,format},
        {ExpertTierWeightProjection::Up,512,H,format},{ExpertTierWeightProjection::Down,H,512,format}}}};
    const auto placement=MoEExpertOwnerMap::buildShardPlacement(plan,{source});
    const auto selected=CPUExpertShardSourceSelection::fromOriginalPlacement(placement,3,world_rank_);
    auto tp=createTwoRankContext(1844);
    ASSERT_EQ(tp->scope(),TPScope::NODE_LOCAL);
    // Keep the original asymmetric-storage threshold regression and also test
    // padded capacity. Both use the same original MPI payload lane: the common
    // root extent exceeds the shared-memory latency threshold even at M=1.
    for(const int M:{1,64}) {
    FP32Tensor records({size_t((world_rank_==1?2:1)*M*K*(H+1)+1)}),
        indices({size_t(M),size_t(K)}),weights({size_t(M),size_t(K)}),
        combined({size_t(M*K),size_t(H)});
    std::array<RootedFloatRecordRange,2> ranges;
    std::vector<int32_t> maps(2*M*K);
    std::vector<float> required(M*K);
    auto context=IDeviceContext::create(DeviceId::cpu());
    MoECPUExpertShardRouteStage::Params p{selected,tp.get(),&records,&indices,&weights,world_rank_==1?&combined:nullptr,
        world_rank_==1?std::span(ranges):std::span<RootedFloatRecordRange>{},maps,required,M,K,H,1,"original_cpu_shards"};
    p.standalone_execution_phase=ExpertHistogramSource::DecodeToken; // Authored record-contract invocation, independent of reusable M capacity.
    auto foreign=p;foreign.original_source=CPUExpertShardSourceSelection::fromOriginalPlacement(placement,3,1-world_rank_);
    EXPECT_THROW(MoECPUExpertShardRouteStage{foreign},std::invalid_argument);
    auto changed=p;changed.columns=H+1;
    EXPECT_THROW(MoECPUExpertShardRouteStage{changed},std::invalid_argument);
    auto aliased=p;aliased.route_weights=&records;
    EXPECT_THROW(MoECPUExpertShardRouteStage{aliased},std::invalid_argument);
    MoECPUExpertShardRouteStage stage(p);
    // Scoped test collection policy, restored even on an assertion failure.
    struct MeasurementScope {
        std::optional<std::string> summary,filter;
        MeasurementScope() {
            if(const auto* value=std::getenv("LLAMINAR_PERF_STATS_SUMMARY"))summary=value;
            if(const auto* value=std::getenv("LLAMINAR_PERF_STATS_FILTER"))filter=value;
            setenv("LLAMINAR_PERF_STATS_SUMMARY","1",1);
            setenv("LLAMINAR_PERF_STATS_FILTER","moe_canonical_publication",1);
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        ~MeasurementScope() {
            if(summary)setenv("LLAMINAR_PERF_STATS_SUMMARY",summary->c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            if(filter)setenv("LLAMINAR_PERF_STATS_FILTER",filter->c_str(),1);else unsetenv("LLAMINAR_PERF_STATS_FILTER");
            PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
    } measurement;
    ASSERT_TRUE(PerfStatsCollector::isDomainEnabled("moe_canonical_publication"));
    const auto observedBytes=[&]() {
        double bytes=0;
        for(const auto& record:PerfStatsCollector::snapshot({"moe_canonical_publication"}))
            if(record.name=="cpu_packed_route_gather_local_bytes" &&
               record.tags.at("stage")==p.stage_name && record.tags.at("participant")==std::to_string(world_rank_))
                bytes+=record.value;
        return bytes;
    };
    for(int replay=0;replay<20;++replay) {
        const int liveRows=std::min(M,std::array<int,4>{9,0,1,9}[replay%4]);
        const float unused=-777.0f-float(world_rank_);
        std::fill_n(records.mutable_data(),records.numel(),unused);
        const double bytesBefore=observedBytes();
        std::fill_n(combined.mutable_data(),combined.numel(),-991);
        size_t count=0;
        for(int slot=0;slot<M*K;++slot) {
            indices.mutable_data()[slot]=float(slot%4);
            const bool live=slot/K<liveRows && slot%3!=1;
            weights.mutable_data()[slot]=live?.25f:0;
            if(!live)continue;
            auto *record=canonical_moe_route_record::record(records.mutable_data(),count++,H);
            for(int h=0;h<H;++h)record[h]=float(world_rank_*11+slot+h)*.125f;
            ASSERT_TRUE(canonical_moe_route_record::writeFlatRouteSlot(record,H,slot));
        }
        ASSERT_TRUE(canonical_moe_route_record::writeRecordCount(records.mutable_data(),records.numel(),count));
        ASSERT_TRUE(stage.execute(context.get()));
        EXPECT_EQ(observedBytes()-bytesBefore,double(count*(H+1)*sizeof(float)));
        if(liveRows==0)EXPECT_EQ(count,0u);else EXPECT_GT(count,0u);
        const size_t payloadRecords=world_rank_==1?2*count:count;
        EXPECT_TRUE(std::all_of(records.data()+payloadRecords*(H+1),records.data()+records.numel()-1,
            [&](float value){return value==unused;})) << "capacity padding was copied";
        if(world_rank_==1) {
            ASSERT_TRUE(validRootedFloatRecordRanges(ranges,2*count));
            EXPECT_EQ(ranges[0].record_count,count);EXPECT_EQ(ranges[1].record_count,count);
            EXPECT_EQ(ranges[1].first_record,0u);EXPECT_EQ(ranges[0].first_record,count);
            EXPECT_EQ(ranges[0].record_count+ranges[1].record_count,2*count);
        }
        if(world_rank_==1)for(int slot=0;slot<M*K;++slot)for(int h=0;h<H;++h)
            EXPECT_EQ(combined.data()[slot*H+h],weights.data()[slot]!=0?
                float(slot+h)*.125f+float(11+slot+h)*.125f:-991);
    }
}
}
