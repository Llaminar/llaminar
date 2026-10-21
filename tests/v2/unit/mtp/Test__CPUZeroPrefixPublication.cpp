/**
 * @file Test__CPUZeroPrefixPublication.cpp
 * @brief Actual CPU publication keeps typed prior commits distinct from snapshot NoOp.
 *
 * Small independent canonical CPU owners expose exact accepted write intervals.
 * All assertions invoke the real generic publisher, including scalar routing,
 * rather than manually calling a retained stage and bypassing its orchestration.
 */
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "execution/compute_stages/IComputeStage.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"
#include "utils/DebugEnv.h"
#include <cstdlib>
#include <optional>
#include <string>
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstring>
using namespace llaminar2;
namespace {
/** @brief Native prefix stage with an immutable two-word prior and four independent candidate words. */
class NativePrior final:public IComputeStage {
public:
    NativePrior():IComputeStage(DeviceId::cpu()),write_{live.data(),sizeof(live)}{}
    bool execute(IDeviceContext*)override{return true;}
    ComputeStageType type()const override{return ComputeStageType::COPY;}
    bool supportsBackend(ComputeBackendType b)const override{return b==ComputeBackendType::CPU;}
    bool hasVerifierStateCapture()const override{return captured;}
    bool requiresVerifierStateCaptureForPublication()const override{return true;}
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
    /** @brief Authenticate the decision and complete accepted interval without changing one live word. */
    CPUVerifierStateRestorePlan planCPUVerifierStateRestoreRow(int row)override {
        ++plans;
        if(invalid || row>=4)return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
        return {.status=CPUVerifierStateRestorePlanStatus::Ready,
            .kind=CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit,
            .native_destinations=empty ? std::span<const CPUVerifierStateRestoreDestination>{} :
                std::span<const CPUVerifierStateRestoreDestination>{&write_,1}};
    }
    /** @brief Restore the suffix of an independently concatenated prior/candidate sequence. */
    bool restoreVerifierStateCaptureRow(int row,void* stream)override {
        if(stream || invalid || row>=4)return false;++commits;if(empty)return true;
        std::array<int,6> sequence{10,11,20,21,22,23};
        const int accepted=row<0 ? 0 :row+1;
        std::copy_n(sequence.data()+accepted,2,live.data());return true;
    }
    void clearVerifierStateCaptureBindingAfterPublication()override{++clears;}
    std::array<int,2> live{91,92};bool invalid=false,empty=false,captured=true;
    int plans=0,commits=0,clears=0;
private:CPUVerifierStateRestoreDestination write_;
};
/** @brief Legacy snapshots retain the explicit negative-row NoOp contract. */
class LegacySnapshot final:public IComputeStage {
public:
    LegacySnapshot():IComputeStage(DeviceId::cpu()){}
    bool execute(IDeviceContext*)override{return true;}
    ComputeStageType type()const override{return ComputeStageType::COPY;}
    bool supportsBackend(ComputeBackendType b)const override{return b==ComputeBackendType::CPU;}
    bool hasVerifierStateCapture()const override{return true;}
    StageDumpInfo buildDumpInfoImpl()const override{return {};}
    /** @brief A legacy negative decision requires no copy or capture retirement. */
    CPUVerifierStateRestorePlan planCPUVerifierStateRestoreRow(int row)override {
        ++plans;if(invalid)return {.status=CPUVerifierStateRestorePlanStatus::Invalid};
        if(row<0 || alwaysNoOp)return {.status=CPUVerifierStateRestorePlanStatus::NoOp};
        return {.status=CPUVerifierStateRestorePlanStatus::Ready,.destination=destination,
            .source=source.data(),.bytes=sizeof(source)};
    }
    void clearVerifierStateCaptureBindingAfterPublication()override{++clears;}
    std::array<int,2> live{61,62},source{71,72};int* destination=live.data();
    bool invalid=false,alwaysNoOp=false;int plans=0,clears=0;
};
/** @brief One host-owned MTP frame; accepted drafts and flat state rows remain distinct. */
MTPSpecStepPlan stepFor(int row) {
    MTPSpecStepPlan step;step.request_index=0;step.request_id=39;step.draft_count=3;
    step.target_rows=4;step.accepted_count=row<0?0:row;return step;
}
/** @brief Invoke the canonical host vector publication without altering stage policy. */
MTPSpecStatePublicationResult batch(int row,std::vector<IComputeStage*> stages,bool required=true,void* stream=nullptr) {
    MTPSpecStepPlanBatch plans;plans.ok=true;plans.request_count=1;plans.steps={stepFor(row)};
    return publishAcceptedMTPSpecStateFromVerifierRows(plans,&row,stages,DeviceId::cpu(),stream,required);
}
/** @brief Enable only the public CPU MTP record family, restoring process configuration after the assertion. */
class ScopedMTPStats {
public:
    ScopedMTPStats() {
        for(std::size_t i=0;i<names.size();++i)if(const auto* value=std::getenv(names[i]))old[i]=value;
        ::setenv(names[0],"1",1);::setenv(names[1],"mtp",1);
        mutableDebugEnv().reload();PerfStatsCollector::reloadConfigurationFromEnvironment();PerfStatsCollector::reset();
    }
    ~ScopedMTPStats() {
        PerfStatsCollector::reset();
        for(std::size_t i=0;i<names.size();++i)if(old[i])::setenv(names[i],old[i]->c_str(),1);else ::unsetenv(names[i]);
        mutableDebugEnv().reload();PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
private:
    const std::array<const char*,2> names{"LLAMINAR_PERF_STATS_JSON","LLAMINAR_PERF_STATS_FILTER"};
    std::array<std::optional<std::string>,2> old;
};
/** @brief Keep this model-free validation on one physical CPU worker. */
class CPUZeroPrefixPublication:public testing::Test {
    llaminar::v2::ThreadCountGuard threads_{1};
};
}
/** @brief Both CPU entrypoints publish full, partial, input0-only and true-zero decisions from one immutable prior. */
TEST_F(CPUZeroPrefixPublication, SingleAndBatchShareFullPartialAndZeroPublication) {
    for(bool scalar:{false,true}) {
        NativePrior native;
        const std::array<int,4> rows{3,1,0,-1};
        const std::array<std::array<int,2>,4> expected{{{22,23},{20,21},{11,20},{10,11}}};
        for(int i=0;i<4;++i) {
            const auto result=scalar ? publishAcceptedMTPSpecStateFromVerifierRow(stepFor(rows[i]),rows[i],
                {&native},DeviceId::cpu(),nullptr,true) :batch(rows[i],{&native});
            ASSERT_TRUE(result.ok)<<result.error;EXPECT_EQ(native.live,expected[i]);
            EXPECT_EQ(result.restored_stage_count,1);EXPECT_EQ(result.skipped_stage_count,0);
            EXPECT_EQ(result.accepted_count,stepFor(rows[i]).accepted_count);
            if(scalar)EXPECT_EQ(result.request_id,39);
        }
        EXPECT_EQ(native.plans,4);EXPECT_EQ(native.commits,4);EXPECT_EQ(native.clears,4);
    }
}
/** @brief A native original-prior restore is planned even when all legacy snapshots are NoOp. */
TEST_F(CPUZeroPrefixPublication, MixedNativeAndLegacyNoOpHaveExactCounters) {
    NativePrior native;LegacySnapshot legacy;
    auto result=batch(-1,{&native,&legacy});ASSERT_TRUE(result.ok)<<result.error;
    EXPECT_EQ(native.live,(std::array{10,11}));EXPECT_EQ(legacy.live,(std::array{61,62}));
    EXPECT_EQ(native.commits,1);EXPECT_EQ(legacy.clears,0);
    EXPECT_EQ(result.restored_stage_count,1);EXPECT_EQ(result.skipped_stage_count,1);
    result=batch(-1,{&legacy});ASSERT_TRUE(result.ok)<<result.error;
    EXPECT_EQ(result.restored_stage_count,0);EXPECT_EQ(result.skipped_stage_count,1);
    EXPECT_EQ(legacy.clears,0);
}
/** @brief Invalid later native or snapshot plans cannot mutate an earlier ready prior restore. */
TEST_F(CPUZeroPrefixPublication, LaterInvalidStagePreservesEveryAcceptedByte) {
    for(bool nativeFailure:{false,true})for(int row:{3,-1}) {
        NativePrior earlier,later;LegacySnapshot legacy;later.invalid=legacy.invalid=true;
        const auto before=earlier.live;
        const auto result=batch(row,{&earlier,nativeFailure?static_cast<IComputeStage*>(&later):&legacy});
        EXPECT_FALSE(result.ok);EXPECT_EQ(earlier.live,before);EXPECT_EQ(earlier.commits,0);EXPECT_EQ(earlier.clears,0);
        EXPECT_EQ(later.commits,0);EXPECT_EQ(legacy.clears,0);
    }
}
/** @brief Overlap is rejected across native and legacy plans before either accepted owner is written. */
TEST_F(CPUZeroPrefixPublication, LaterSnapshotOverlapRejectsBeforeAnyWrite) {
    NativePrior native;LegacySnapshot legacy;legacy.destination=native.live.data();const auto before=native.live;
    const auto result=batch(1,{&native,&legacy});EXPECT_FALSE(result.ok);
    EXPECT_EQ(native.live,before);EXPECT_EQ(native.commits,0);EXPECT_EQ(native.clears,0);EXPECT_EQ(legacy.clears,0);
}
/** @brief Empty native publication remains a validated operation with zero physical payload. */
TEST_F(CPUZeroPrefixPublication, EmptyNativeOperationAndRequiredCaptureRemainExplicit) {
    NativePrior native;native.empty=true;const auto before=native.live;
    auto result=batch(-1,{&native});ASSERT_TRUE(result.ok)<<result.error;
    EXPECT_EQ(native.live,before);EXPECT_EQ(native.commits,1);EXPECT_EQ(result.restored_stage_count,1);
    NativePrior missing;missing.captured=false;
    result=batch(-1,{&native,&missing});EXPECT_FALSE(result.ok);EXPECT_EQ(native.commits,1);
    LegacySnapshot noop;noop.alwaysNoOp=true;result=batch(1,{&noop});EXPECT_FALSE(result.ok);EXPECT_EQ(noop.clears,0);
}
/** @brief CPU/GPU stream authority failures happen before every planning or write boundary. */
TEST_F(CPUZeroPrefixPublication, CPURejectsGPUStreamBeforePlan) {
    NativePrior native;const auto before=native.live;
    auto result=batch(-1,{&native},true,reinterpret_cast<void*>(1));EXPECT_FALSE(result.ok);
    result=publishAcceptedMTPSpecStateFromVerifierRow(stepFor(1),1,{&native},DeviceId::cpu(),reinterpret_cast<void*>(1),true);
    EXPECT_FALSE(result.ok);EXPECT_EQ(native.plans,0);EXPECT_EQ(native.live,before);
}

/** @brief Scalar request attempts remain observable while shared CPU execution reports each physical restore once. */
TEST_F(CPUZeroPrefixPublication, ScalarPerfStatsSurviveSharedPlanningWithoutDuplicatedPayloadBytes) {
    ScopedMTPStats stats;NativePrior native;
    for(int row:{3,0,-1}) {
        const auto result=publishAcceptedMTPSpecStateFromVerifierRow(stepFor(row),row,{&native},DeviceId::cpu(),nullptr,true);
        ASSERT_TRUE(result.ok)<<result.error;
    }
    ASSERT_TRUE(batch(1,{&native}).ok);
    const auto records=PerfStatsCollector::snapshot();
    const auto sum=[&](const char* name) {
        double total=0;for(const auto& record:records)if(record.domain=="mtp" && record.name==name &&
            record.kind==PerfStatRecord::Kind::Counter)total+=record.value;return total;
    };
    EXPECT_DOUBLE_EQ(sum("spec_state_restore_row_requests"),3.0);
    EXPECT_DOUBLE_EQ(sum("spec_state_host_batch_restore_requests"),4.0);
    EXPECT_DOUBLE_EQ(sum("spec_state_cpu_parallel_restore_bytes"),4.0*sizeof(native.live));
    EXPECT_EQ(native.commits,4);
}
