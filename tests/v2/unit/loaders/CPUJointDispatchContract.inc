/**
 * @file CPUJointDispatchContract.inc
 * @brief Original joint source lease drives both dispatch stages without scalar fragment owners.
 *
 * The recorder inspects actual stage-produced packets and never completes a
 * transport or claims native CPU/GPU readiness. Source preparation and lease
 * construction use the existing original GGUF fixture without snapshot edits.
 */
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoESparseDispatchStage.h"
namespace {
/** @brief Dispatch-only observation at the existing public collective boundary. */
class JointDispatchContractRecorder final : public llaminar2::IMoEOverlaySparseCollectiveContext {
public:
    int calls=0;
    llaminar2::MoEOverlaySparseRows observed;
    /** @brief Borrow the actual outbound packet for immediate contract checks.
     * @return Accepted observation with no completed transport or publication. */
    llaminar2::MoEOverlayCollectiveResult dispatch(const llaminar2::MoEOverlayCollectiveKey&,
        const llaminar2::MoEOverlaySparseRows& rows,llaminar2::MoEOverlaySparseRows*,llaminar2::IDeviceContext*) override {
        ++calls;observed=rows;return {.ok=true,.collective_complete=false};
    }
    /** @brief Reject a return operation outside this dispatch-only fixture.
     * @return An explicit failure; no return payload is synthesized. */
    llaminar2::MoEOverlayCollectiveResult returnReduce(const llaminar2::MoEOverlayCollectiveKey&,
        const llaminar2::MoEOverlayReturnRows&,llaminar2::MoEOverlayReturnRows*,llaminar2::IDeviceContext*) override {
        return {.ok=false,.error="Dispatch contract never fabricates a return"};
    }
    /** @brief Observe cancellation without a native transport or pending work to retire. */
    void abort(const llaminar2::MoEOverlayCollectiveKey&,int)override {}
};
}
/** @test Real Static publication preserves joint fanout and scalar GPU ownership for partial/empty work. */
TEST(CPUJointDispatchContract, OriginalLeaseTargetsEverySourceCohortAndKeepsWholeOwnerScalar)
{
    using namespace llaminar2;
    const auto original=staticJointOriginal();
    auto authority=std::make_shared<MoEOverlayResidencyAuthority>(staticJointConfig(original));
    const auto snapshot=authority->snapshot();ASSERT_TRUE(snapshot);ASSERT_TRUE(snapshot->valid());
    ASSERT_TRUE(snapshot->owner_map.owners().empty());
    ASSERT_EQ(&snapshot->originalOwnerMap(),original.preparation->cpuJointOwnerMap().get());
    const auto members=staticJointMembers(original);ASSERT_EQ(members.size(),2u);
    const auto* whole=snapshot->originalOwnerMap().placementFor(7,0);ASSERT_NE(whole,nullptr);
    ASSERT_FALSE(whole->shard_cohort);ASSERT_GE(whole->owner_participant,0);
    constexpr int capacity=4,K=4,H=8;
    FP32Tensor indices({capacity,K}),weights({capacity,K}),hidden({capacity,H});
    const std::array<int,4> ids{0,64,511,3};
    for(int row=0;row<capacity;++row)for(int col=0;col<H;++col)hidden.mutable_data()[row*H+col]=float(row*H+col+1);
    MoEExpertDispatchOutput output;
    MoEExpertDispatchStage::Params params;
    params.device_id=DeviceId::cpu();params.routing_indices=&indices;params.routing_weights=&weights;params.hidden=&hidden;
    params.seq_len=capacity;params.top_k=K;params.d_model=H;params.continuation_domain=original.plan->continuation_domain;
    params.placement=original.plan->placements.front();params.routed_tiers=original.plan->routed_tiers;
    params.residency_authority=authority;params.output=&output;
    MoEExpertDispatchStage dispatch(params);
    llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(),ComputeBackendType::CPU);
    for(const int live:{3,0,1,3}) {
        SCOPED_TRACE(live);
        for(int row=0;row<capacity;++row)for(int k=0;k<K;++k) {
            indices.mutable_data()[row*K+k]=float(ids[k]);
            weights.mutable_data()[row*K+k]=row<live?(k==2?.5f:k==1?.25f:.125f):0;
        }
        ASSERT_TRUE(dispatch.execute(&ctx));ASSERT_TRUE(output.residency_lease);
        EXPECT_EQ(&(**output.residency_lease).originalOwnerMap(),&snapshot->originalOwnerMap());
        ASSERT_EQ(output.tiers.size(),2u);
        ASSERT_EQ(output.tiers[1].entries.size(),size_t(capacity*2)); // Descriptor/evidence retains original zero weights.
        for(const auto& entry:output.tiers[1].entries)EXPECT_EQ(entry.destination_participant,-1);
        for(const auto& entry:output.tiers[0].entries)EXPECT_EQ(entry.destination_participant,whole->owner_participant);
        for(const int target:members) {
            JointDispatchContractRecorder recorder;MoEOverlayCollectiveWorkspace workspace;
            workspace.ensureCapacity(capacity,capacity*K,H,K,DeviceId::cpu());
            auto inbound=workspace.dispatchReceive(7,1);
            auto outbound=workspace.localExpertInput(7,1);
            std::fill_n(outbound.hidden_rows_fp32,capacity*H,-731.0f);
            MoESparseDispatchStage::Params sparse;
            sparse.device_id=DeviceId::cpu();sparse.collective_context=&recorder;sparse.workspace=&workspace;
            sparse.key.layer_idx=7;sparse.key.tier_idx=1;sparse.key.domain_id=1;sparse.key.generation_id=11;
            sparse.source_participant=whole->owner_participant;sparse.target_participant=target;sparse.key.participant_id=target;
            sparse.hidden=&hidden;sparse.routing_indices=&indices;sparse.routing_weights=&weights;sparse.seq_len=capacity;sparse.top_k=K;sparse.d_model=H;
            sparse.dispatch_output=&output;sparse.tier_index=1;sparse.inbound_rows=&inbound;
            MoESparseDispatchStage consumer(sparse);ASSERT_TRUE(consumer.execute(&ctx));
            EXPECT_FALSE(consumer.manualGraphBoundaryComplete());ASSERT_EQ(recorder.calls,1);
            const auto& packet=recorder.observed;
            EXPECT_EQ(packet.target_participant,target);EXPECT_EQ(packet.residency_epoch,snapshot->epoch);
            ASSERT_EQ(packet.live_row_count,size_t(live));ASSERT_EQ(packet.live_entry_count,size_t(live*2));
            for(int row=0;row<live;++row) {
                EXPECT_EQ(packet.row_ids_host[row],row);EXPECT_EQ(packet.entry_offsets_host[row],row*2);
                for(int k=0;k<2;++k) {
                    const int e=row*2+k;
                    EXPECT_EQ(packet.expert_ids_host[e],ids[k+1]);
                    EXPECT_EQ(packet.original_route_slots_host[e],row*K+k+1);
                    EXPECT_EQ(packet.route_weights_host[e],weights.data()[row*K+k+1]);
                }
                for(int col=0;col<H;++col)EXPECT_EQ(packet.hidden_rows_fp32[row*H+col],hidden.data()[row*H+col]);
            }
            EXPECT_EQ(packet.entry_offsets_host[live],live*2);
            EXPECT_TRUE(std::all_of(packet.hidden_rows_fp32+live*H,packet.hidden_rows_fp32+capacity*H,
                [](float value){return value==-731.0f;}));
            if(live) {
                auto stale=output;++stale.residency_epoch;sparse.dispatch_output=&stale;
                MoESparseDispatchStage staleStage(sparse);EXPECT_FALSE(staleStage.execute(&ctx));EXPECT_EQ(recorder.calls,1);
                sparse.dispatch_output=&output;sparse.target_participant=whole->owner_participant;sparse.key.participant_id=whole->owner_participant;
                MoESparseDispatchStage foreign(sparse);EXPECT_FALSE(foreign.execute(&ctx));EXPECT_EQ(recorder.calls,1);
            }
        }
    }
    // Authentic source map alone is not a joint dispatch lease; no snapshot field is manufactured.
    auto missing=params;missing.residency_authority.reset();missing.owner_map=original.preparation->cpuJointOwnerMap();
    MoEExpertDispatchOutput rejected;missing.output=&rejected;
    MoEExpertDispatchStage missingLease(missing);EXPECT_FALSE(missingLease.execute(&ctx));EXPECT_FALSE(rejected.residency_lease);
    auto wrongEpoch=params;MoEExpertDispatchStage staleProducer(wrongEpoch);
    staleProducer.updateMoEOverlayCollectiveRuntimeParams({.generation_id=11,.step_id=13,
        .execution_semantics=IComputeStage::MoEOverlayCollectiveRuntimeParams::ExecutionSemantics::Prefill,
        .placement_epoch=snapshot->epoch+1});
    EXPECT_FALSE(staleProducer.execute(&ctx));
}

/** @test Joint-only zero selection precedes packing/counts and preserves original holes. */
TEST(CPUJointDispatchContract, ZeroWeightsCompactBeforePublicationWithoutLosingOriginalSlots)
{
    using namespace llaminar2;
    const auto original=staticJointOriginal();
    auto authority=std::make_shared<MoEOverlayResidencyAuthority>(staticJointConfig(original));
    const auto members=staticJointMembers(original);ASSERT_EQ(members.size(),2u);
    const auto* sourceOwner=authority->snapshot()->originalOwnerMap().placementFor(7,0);ASSERT_NE(sourceOwner,nullptr);
    constexpr int rows=3,K=2,H=8;
    FP32Tensor indices({rows,K}),weights({rows,K}),hidden({rows,H});
    for(int i=0;i<rows*K;++i)indices.mutable_data()[i]=float(i%2?511:64);
    for(int i=0;i<rows*H;++i)hidden.mutable_data()[i]=float(i+7);
    MoEExpertDispatchOutput output;MoEExpertDispatchStage::Params p;
    p.device_id=DeviceId::cpu();p.routing_indices=&indices;p.routing_weights=&weights;p.hidden=&hidden;
    p.seq_len=rows;p.top_k=K;p.d_model=H;p.continuation_domain=original.plan->continuation_domain;
    p.placement=original.plan->placements.front();p.routed_tiers=original.plan->routed_tiers;
    p.residency_authority=authority;p.output=&output;
    MoEExpertDispatchStage dispatch(p);llaminar2::testing::MockDeviceContext ctx(DeviceId::cpu(),ComputeBackendType::CPU);
    const std::array<std::array<float,rows*K>,3> patterns{{{{0,1,0,0,0,0}},{{0,0,0,0,0,0}},{{1,0,1,0,0,1}}}};
    for(const auto& pattern:patterns) {
        std::copy(pattern.begin(),pattern.end(),weights.mutable_data());ASSERT_TRUE(dispatch.execute(&ctx));
        ASSERT_TRUE(output.residency_lease);ASSERT_EQ(output.tiers[1].entries.size(),size_t(rows*K));
        std::vector<int32_t> expectedSlots,expectedRows,firstSlots;
        for(int row=0;row<rows;++row)for(int k=0;k<K;++k)if(pattern[row*K+k]!=0) {
            expectedSlots.push_back(row*K+k);
            if(expectedRows.empty() || expectedRows.back()!=row)expectedRows.push_back(row);
        }
        for(const int target:members) {
            JointDispatchContractRecorder recorder;MoEOverlayCollectiveWorkspace workspace;
            workspace.ensureCapacity(rows,rows*K,H,K,DeviceId::cpu());
            auto inbound=workspace.dispatchReceive(7,1);auto outbound=workspace.localExpertInput(7,1);
            std::fill_n(outbound.hidden_rows_fp32,rows*H,-733.0f);
            MoESparseDispatchStage::Params q;q.device_id=DeviceId::cpu();q.collective_context=&recorder;q.workspace=&workspace;
            q.key.generation_id=23;q.key.layer_idx=7;q.key.tier_idx=1;q.key.domain_id=1;q.key.participant_id=target;
            q.source_participant=sourceOwner->owner_participant;q.target_participant=target;q.hidden=&hidden;q.routing_indices=&indices;q.routing_weights=&weights;q.seq_len=rows;q.top_k=K;q.d_model=H;
            q.dispatch_output=&output;q.tier_index=1;q.inbound_rows=&inbound;
            MoESparseDispatchStage consumer(q);ASSERT_TRUE(consumer.execute(&ctx));EXPECT_FALSE(consumer.manualGraphBoundaryComplete());
            ASSERT_EQ(recorder.calls,1);const auto& packet=recorder.observed;
            ASSERT_EQ(packet.live_entry_count,expectedSlots.size());ASSERT_EQ(packet.live_row_count,expectedRows.size());
            const std::vector<int32_t> observedSlots(packet.original_route_slots_host,packet.original_route_slots_host+packet.live_entry_count);
            EXPECT_EQ(observedSlots,expectedSlots);
            if(target==members.front())firstSlots=observedSlots;else EXPECT_EQ(observedSlots,firstSlots);
            for(size_t entry=0;entry<expectedSlots.size();++entry) {
                EXPECT_EQ(packet.route_weights_host[entry],pattern[expectedSlots[entry]]);
                EXPECT_EQ(packet.expert_ids_host[entry],int(indices.data()[expectedSlots[entry]]));
            }
            for(size_t row=0;row<expectedRows.size();++row) {
                EXPECT_EQ(packet.row_ids_host[row],expectedRows[row]);
                int localRoute=0;
                for(size_t entry=0;entry<expectedSlots.size();++entry)if(expectedSlots[entry]/K==expectedRows[row]) {
                    EXPECT_EQ(packet.compact_route_slots_host[entry],int(row)*K+localRoute);
                    ++localRoute;
                }
                for(int col=0;col<H;++col)EXPECT_EQ(packet.hidden_rows_fp32[row*H+col],hidden.data()[expectedRows[row]*H+col]);
            }
            EXPECT_TRUE(std::all_of(packet.hidden_rows_fp32+expectedRows.size()*H,packet.hidden_rows_fp32+rows*H,
                [](float value){return value==-733.0f;}));
        }
    }
}
