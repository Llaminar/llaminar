/**
 * @file Test__PipelineMaintenanceAdmission.cpp
 * @brief Device-free actual submission-body replay for independent stage maintenance admission.
 *
 * Descriptor recorders prove root/member/extent and failure retirement, without
 * claiming native execution. Explicit native pipeline replays certify capture.
 */
#include "execution/local_execution/orchestrators/PipelineMaintenanceAdmission.h"
#include "execution/mtp/MTPGraphOwnerPlan.h"
#include "execution/mtp/HostedDeviceGenerationLifecycle.h"
#include "kernels/common/SamplingMath.h"
#include "mocks/MockLocalTPContext.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <climits>
#include <numeric>
namespace llaminar2::test {
/** @brief Access to the actual private compiler bodies, with no alternative implementation. */
struct PipelineMaintenanceAdmissionReplayAccess {
    static bool enqueue(ILocalTPContext &n, ILocalTPContext *t,int first,int m,const void *s,void *r,void *stream) {
        return PipelineMaintenanceAdmission::enqueueOriginalMinimum(n,t,first,m,s,r,stream);
    }
    static void join(ILocalTPContext &n,std::span<std::future<void>> f,std::span<std::atomic<bool>> e,
        std::barrier<> &a,std::barrier<> &b,std::barrier<> &c) {
        PipelineMaintenanceAdmission::joinOriginalRecordings(n,f,e,a,b,c,{});
    }
};
namespace {
using namespace sampling_math;
/** @brief Record actual native operands without constructing a GPU context. */
class OriginalPlanRecorder final : public MockLocalTPContext {
public:
    struct Call { const void *source; void *result; size_t count; CollectiveDataType type; CollectiveOp op; int root,member; void *stream; };
    std::vector<Call> reductions,broadcasts;
    std::atomic<bool> aborted{false};
    bool reduceRawOnStream(const void *s,void *r,size_t c,CollectiveDataType t,CollectiveOp o,int root,int member,void *stream,const std::string &) override {
        reductions.push_back({s,r,c,t,o,root,member,stream}); return true;
    }
    bool broadcastRawOnStream(const void *s,void *r,size_t c,CollectiveDataType t,int root,int member,void *stream,const std::string &) override {
        broadcasts.push_back({s,r,c,t,CollectiveOp::ALLREDUCE_MIN,root,member,stream}); return true;
    }
    bool signalNativeCompletionAbort() override { aborted.store(true); return true; }
};
/** @brief Freeze metadata coordinates only, never hardware execution membership. */
void configure(OriginalPlanRecorder &c,int first,int count) {
    std::vector<GlobalDeviceAddress> devices;
    for(int i=0;i<count;++i) devices.push_back(GlobalDeviceAddress::rocm(first+i));
    c.setDevices(std::move(devices)); c.setBackend(CollectiveBackendType::RCCL);
}
TEST(PipelineMaintenanceAdmission, UnequalArbitraryWidthsUseActualTailLeaderAndOneScalar) {
    for(const auto &widths:std::vector<std::vector<int>>{{1,2},{2,1},{1,2,4},{3,5,11},{17,1,9,2}}) {
        const int degree=std::accumulate(widths.begin(),widths.end(),0),first=degree-widths.back();
        OriginalPlanRecorder n,t; configure(n,0,degree); configure(t,first,widths.back());
        std::vector<int32_t> source(degree,7),result(degree,0);
        for(int m=0;m<degree;++m) {
            void *stream=reinterpret_cast<void *>(std::uintptr_t(64+m));
            ASSERT_TRUE(PipelineMaintenanceAdmissionReplayAccess::enqueue(n,widths.back()>1?&t:nullptr,first,m,&source[m],&result[m],stream));
            const auto &call=n.reductions.back();
            EXPECT_EQ(call.root,first); EXPECT_EQ(call.member,m); EXPECT_EQ(call.count,1u);
            EXPECT_EQ(call.type,CollectiveDataType::INT32); EXPECT_EQ(call.op,CollectiveOp::ALLREDUCE_MIN);
            EXPECT_EQ(call.source,&source[m]); EXPECT_EQ(call.result,&result[m]); EXPECT_EQ(call.stream,stream);
        }
        EXPECT_EQ(n.reductions.size(),size_t(degree));
        EXPECT_EQ(t.broadcasts.size(),widths.back()>1?size_t(widths.back()):0u);
        for(size_t i=0;i<t.broadcasts.size();++i) {
            const auto &call=t.broadcasts[i]; EXPECT_EQ(call.root,0); EXPECT_EQ(call.member,int(i));
            EXPECT_EQ(call.count,1u); EXPECT_EQ(call.result,&result[first+i]); EXPECT_EQ(call.source,call.result);
        }
    }
}
TEST(PipelineMaintenanceAdmission, ForeignCoordinatesRejectBeforeNativeSubmission) {
    OriginalPlanRecorder n,t; configure(n,0,5); configure(t,2,3); int s=3,r=0; void *stream=reinterpret_cast<void *>(64);
    for(int m:{-1,5}) EXPECT_FALSE(PipelineMaintenanceAdmissionReplayAccess::enqueue(n,&t,2,m,&s,&r,stream));
    for(void *bad:std::array<void *,3>{nullptr,reinterpret_cast<void *>(1),reinterpret_cast<void *>(2)})
        EXPECT_FALSE(PipelineMaintenanceAdmissionReplayAccess::enqueue(n,&t,2,0,&s,&r,bad));
    configure(t,3,3);
    EXPECT_FALSE(PipelineMaintenanceAdmissionReplayAccess::enqueue(n,&t,2,0,&s,&r,stream));
    EXPECT_TRUE(n.reductions.empty()); EXPECT_TRUE(t.broadcasts.empty());
}
TEST(PipelineMaintenanceAdmission, FactoryRejectsMismatchedNativeCountBeforeAnyGpuAccess) {
    class MismatchedCount final : public MockLocalTPContext {
    public:
        int degree() const override { return 5; }
    } native;
    native.setDevices({GlobalDeviceAddress::rocm(0),GlobalDeviceAddress::rocm(1),
        GlobalDeviceAddress::rocm(2),GlobalDeviceAddress::rocm(3)});
    native.setBackend(CollectiveBackendType::RCCL);
    OriginalPlanRecorder tail; configure(tail,2,3);
    PhysicalMemoryPlanBuilder plan;
    plan.add({.world_rank=0,.device=DeviceId::cpu(),.total_bytes=1024,.admission_available_bytes=1024},
        PhysicalMemoryOwner::ExecutionWorkspace,1);
    const auto memory=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
    std::array<PipelineMaintenanceSource,5> inactive;
    std::array<void *,5> streams; streams.fill(reinterpret_cast<void *>(64));
    try {
        (void)PipelineMaintenanceAdmission::create(native,2,&tail,inactive,memory,streams);
        FAIL()<<"Malformed original membership reached GPU preparation";
    } catch(const std::invalid_argument &error) {
        EXPECT_STREQ(error.what(),"Pipeline maintenance admission requires its complete original local native cohort");
    }
    EXPECT_TRUE(tail.broadcasts.empty());
    EXPECT_EQ(memory->claimedBytes(DeviceId::cpu(),PhysicalMemoryOwner::ExecutionWorkspace,
        PhysicalMemoryMaterializationKind::NewAllocation),0u);
}
TEST(PipelineMaintenanceAdmission, LaterPreEntryFailureDrainsEarlierBlockedCapture) {
    for(int replay=0;replay<20;++replay) {
        OriginalPlanRecorder n; std::barrier a(2),b(2),c(2); std::array<std::atomic<bool>,2> entered{};
        std::array<std::future<void>,2> futures;
        futures[0]=std::async(std::launch::async,[&]{entered[0].store(true,std::memory_order_release); a.arrive_and_wait(); b.arrive_and_wait(); c.arrive_and_wait();});
        std::promise<void> failure; futures[1]=failure.get_future();
        failure.set_exception(std::make_exception_ptr(std::runtime_error("worker rejected before entry")));
        EXPECT_THROW(PipelineMaintenanceAdmissionReplayAccess::join(n,futures,entered,a,b,c),std::runtime_error);
        EXPECT_TRUE(n.aborted.load()); EXPECT_FALSE(futures[0].valid()); EXPECT_FALSE(futures[1].valid());
    }
}
TEST(PipelineMaintenanceAdmission, IndependentCadenceLimitsActualVerifierWithoutClockOverwrite) {
    const std::array<int,3> initial{2,5,INT_MAX},recurring{3,7,INT_MAX};
    for(int replay=0;replay<20;++replay) {
        auto clocks=initial;
        for(int round=0;round<24;++round) {
            const auto before=clocks; const int minimum=*std::min_element(clocks.begin(),clocks.end());
            std::array<int,kDeviceGenerationControlCount> control{};
            control[kDeviceGenerationControlOk]=1; control[kDeviceGenerationControlRemainingTokenCount]=128;
            control[kDeviceGenerationControlActiveVerifierRowCount]=4;
            const int budget=sampling_math::prepare_device_generation_transaction_budget(4,minimum,control.data());
            EXPECT_EQ(budget,std::min(4,minimum)); EXPECT_EQ(clocks,before);
            for(size_t stage=0;stage<2;++stage) { ASSERT_LE(budget,clocks[stage]); clocks[stage]-=budget; if(!clocks[stage]) clocks[stage]=recurring[stage]; }
            EXPECT_EQ(clocks[2],INT_MAX);
        }
    }
}
TEST(PipelineMaintenanceAdmission, ExactOriginalGenerationMaintenanceHandoffRetainsFailedFrontiers) {
    const auto edge=DeviceEventEdge::at(DeviceTimelinePoint::DeviceGenerationStateReady)
        .from(DeviceTimelineRole::DeviceGenerationController).to(DeviceTimelineRole::MoERebalanceMaintenance);
    EXPECT_TRUE(edge.validForConsumption());
    DeviceGenerationStateHandoff handoff;
    void *generation=reinterpret_cast<void *>(64),*maintenance=reinterpret_cast<void *>(128);
    ASSERT_TRUE(handoff.publish(DeviceGenerationStatePublicationKind::Admission,generation,1,[]{return true;}));
    ASSERT_TRUE(handoff.borrow(generation,DeviceTimelineRole::DeviceGenerationController,1,
        [&](void *producer,void *consumer){return producer==generation && consumer==generation;}));
    EXPECT_FALSE(handoff.publish(DeviceGenerationStatePublicationKind::CommittedTransaction,generation,1,[]{return false;}));
    EXPECT_TRUE(handoff.borrowed()); EXPECT_EQ(handoff.frontierStream(),generation);
    ASSERT_TRUE(handoff.publish(DeviceGenerationStatePublicationKind::CommittedTransaction,generation,1,[]{return true;}));
    EXPECT_FALSE(handoff.borrow(maintenance,DeviceTimelineRole::MoERebalanceMaintenance,1,[](void *,void *){return false;}));
    EXPECT_TRUE(handoff.published()); EXPECT_EQ(handoff.frontierStream(),generation);
    ASSERT_TRUE(handoff.borrow(maintenance,DeviceTimelineRole::MoERebalanceMaintenance,1,
        [&](void *producer,void *consumer){return producer==generation && consumer==maintenance;}));
    EXPECT_EQ(handoff.borrowerRole(),DeviceTimelineRole::MoERebalanceMaintenance);
    ASSERT_TRUE(handoff.publish(DeviceGenerationStatePublicationKind::CommittedTransaction,maintenance,1,[]{return true;}));
    ASSERT_TRUE(handoff.borrow(generation,DeviceTimelineRole::DeviceGenerationController,1,
        [&](void *producer,void *consumer){return producer==maintenance && consumer==generation;}));
    EXPECT_EQ(handoff.frontierStream(),generation);
}
TEST(PipelineMaintenanceAdmission, NativePreparationReclassifiesExistingOwnersWithoutAddingExecutables) {
    MTPRuntimeConfig c; c.enabled=true; c.draft_tokens=3; c.graph_capacity_draft_tokens=3;
    const MTPGraphOwnerPlan flat(c,MTPVerifierGraphFamily::ExactRows,MTPGenerationParentRole::Controller);
    const auto native=flat.withVerifierPreparationTopology(MTPGraphOwnerPlan::VerifierPreparationTopology::NativePipelineMaintenance);
    EXPECT_EQ(native.auxiliaryExecutableSlotCount(),flat.auxiliaryExecutableSlotCount());
    EXPECT_LT(native.boundedHelperExecutableSlotCount(),flat.boundedHelperExecutableSlotCount());
    EXPECT_EQ(native.generalAuxiliaryExecutableSlotCount()-flat.generalAuxiliaryExecutableSlotCount(),flat.boundedHelperExecutableSlotCount()-native.boundedHelperExecutableSlotCount());
    const MTPGraphOwnerPlan follower(c,MTPVerifierGraphFamily::ExactRows,MTPGenerationParentRole::PipelineFollower);
    EXPECT_THROW((void)follower.withVerifierPreparationTopology(MTPGraphOwnerPlan::VerifierPreparationTopology::NativePipelineMaintenance),std::invalid_argument);
}
} // namespace
} // namespace llaminar2::test
