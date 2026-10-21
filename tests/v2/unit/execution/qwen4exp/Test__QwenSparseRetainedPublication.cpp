/**
 * @file Test__QwenSparseRetainedPublication.cpp
 * @brief Actual stage retention and accepted-prefix hooks with compact CPU origins.
 */
#include "execution/compute_stages/stages/qwen4exp/QwenSparseAttentionStage.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "execution/mtp/MTPSpecStatePublisher.h"
#include "tensors/Tensors.h"
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>
using namespace llaminar2;
namespace {
using Stage=QwenSparseAttentionStage;
using Float=Qwen4ExpFloatBinding;
using Int=Qwen4ExpIntBinding;
/** @brief Prepare exact original evidence without adding any accepted-state authority. */
QwenSparseTransactionPriorBinding freezePrior(INT32Tensor& prior,INT32Tensor& traffic,
    const int* replay,const float* accepted,int request,int capacity,int context)
{
    const auto offset=std::size_t(request)*sizeof(qwen4exp::QwenSparseTransactionPrior);
    auto* record=reinterpret_cast<qwen4exp::QwenSparseTransactionPrior*>(
        static_cast<std::byte*>(prior.raw_mutable_data())+offset);
    auto* observed=reinterpret_cast<qwen4exp::QwenSparsePriorTraffic*>(traffic.raw_mutable_data())+request;
    qwen4exp::cpuQwenSparseSnapshotPrior({reinterpret_cast<const qwen4exp::QwenSparseReplay*>(replay),
        accepted,record,observed,capacity,context});
    return {Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR,offset)};
}

}
TEST(QwenSparseRetainedPublication, CompactCPUOriginsPreserveRequestHolesAndOriginalTail)
{
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int requests=3,capacity=4,width=128;
    FP32Tensor source({requests*capacity,width}),retained({requests*capacity,width}),partial({requests,4*width});
    INT32Tensor replay({requests*3}),observed({requests*2}),prior({requests*388}),priorTraffic({requests*2});
    const std::array<int,requests> live{2,3,0},prefix{1,2,3},origins{0,2,5};
    auto *state=static_cast<int *>(replay.raw_mutable_data());
    for(int r=0;r<requests;++r) {
        state[r*3]=prefix[r];state[r*3+1]=live[r];state[r*3+2]=17;
        for(int row=0;row<capacity;++row)for(int d=0;d<width;++d)
            source.mutable_data()[(r*capacity+row)*width+d]=float(r*10000+row*128+d);
        for(int row=0;row<4;++row)for(int d=0;d<width;++d)
            partial.mutable_data()[(r*4+row)*width+d]=float(-r*10000-row*128-d-1);
    }
    std::fill_n(retained.mutable_data(),retained.numel(),-12345.0f);
    std::vector<float> original(partial.data(),partial.data()+partial.numel());
    std::vector<std::unique_ptr<Stage>> stages;
    for(int r=0;r<requests;++r) {
        const auto offset=std::size_t(r)*capacity*width*sizeof(float);
        const auto originalPrior=freezePrior(prior,priorTraffic,state+r*3,partial.data()+r*4*width,r,capacity,32);
        stages.push_back(std::make_unique<Stage>(Stage::Params{.device_id=DeviceId::cpu(),
            .rowCapacity=capacity,.tokenCapacity=32,.replay=originalPrior.storage,
            .operation=Stage::RetainCandidates{Float(source,BufferId::QFN_QSA_RAW_KEY,offset),
                Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS,offset),
                Int(observed,BufferId::QFN_QSA_CANDIDATE_COPY_BYTES,r*sizeof(std::uint64_t)),
                Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS,r*4*width*sizeof(float)),r,requests,
                RequestRowOrigins::nativeWords(origins.data()),originalPrior}}));
        ASSERT_TRUE(stages.back()->hasVerifierStateCapture());
        ASSERT_TRUE(stages.back()->requiresVerifierStateCaptureForPublication());
        ASSERT_TRUE(stages.back()->execute(&context));
    }
    const auto *bytes=static_cast<const std::uint64_t *>(observed.raw_data());
    for(int r=0;r<requests;++r) {
        EXPECT_EQ(bytes[r],std::uint64_t(live[r])*width*sizeof(float));
        for(int row=0;row<capacity;++row)for(int d=0;d<width;++d)
            EXPECT_EQ(retained.data()[(r*capacity+row)*width+d],row<live[r]?
                source.data()[(r*capacity+row)*width+d]:-12345.0f);
    }
    const std::array<int,requests> acceptedRows{0,4,-1};
    for(auto &stage:stages)ASSERT_TRUE(stage->restoreVerifierStateCaptureRows(acceptedRows.data(),requests,nullptr));
    for(int d=0;d<width;++d) {
        EXPECT_EQ(partial.data()[d],original[d]);
        EXPECT_EQ(partial.data()[width+d],source.data()[d]);
        EXPECT_EQ(partial.data()[4*width+d],source.data()[(capacity+2)*width+d]);
    }
    EXPECT_EQ(std::memcmp(partial.data()+8*width,original.data()+8*width,4*width*sizeof(float)),0);
    // No pool or native-KV storage is present in this operation's typed operands.
    for(auto &stage:stages)for(const auto &input:stage->bufferContract().inputs)
        EXPECT_NE(input.id,BufferId::QFN_QSA_POOLED_KEYS);
}

TEST(QwenSparseRetainedPublication, CPUPlanNamesExactWritesAndRejectsMalformedAcceptanceBeforeMutation)
{
    llaminar::v2::ThreadCountGuard threads(1);
    CPUDeviceContext context(DeviceId::cpu(),1);
    FP32Tensor source({4,128}),retained({4,128}),partial({4,128});
    INT32Tensor replay({3}),observed({2}),prior({388}),priorTraffic({2});
    auto *state=static_cast<int *>(replay.raw_mutable_data());state[0]=3;state[1]=4;state[2]=0;
    std::fill_n(partial.mutable_data(),partial.numel(),-11.0f);
    std::fill_n(source.mutable_data(),source.numel(),7.25f);
    const auto originalPrior=freezePrior(prior,priorTraffic,state,partial.data(),0,4,32);
    Stage stage({.device_id=DeviceId::cpu(),.rowCapacity=4,.tokenCapacity=32,
        .replay=originalPrior.storage,.operation=Stage::RetainCandidates{
            Float(source,BufferId::QFN_QSA_RAW_KEY),Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS),
            Int(observed,BufferId::QFN_QSA_CANDIDATE_COPY_BYTES),Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),
            0,1,RequestRowOrigins::fixedStride(19,4),originalPrior}});
    ASSERT_TRUE(stage.execute(&context));
    const auto invalid=stage.planCPUVerifierStateRestoreRow(23);
    EXPECT_EQ(invalid.status,CPUVerifierStateRestorePlanStatus::Invalid);
    EXPECT_TRUE(std::all_of(partial.data(),partial.data()+partial.numel(),[](float value){return value==-11.0f;}));
    const auto boundary=stage.planCPUVerifierStateRestoreRow(19);
    ASSERT_TRUE(boundary.ready());EXPECT_TRUE(boundary.native_destinations.empty());
    const auto selected=stage.planCPUVerifierStateRestoreRow(20);
    ASSERT_TRUE(selected.ready());ASSERT_EQ(selected.native_destinations.size(),1u);
    EXPECT_EQ(selected.kind,CPUVerifierStateRestorePlanKind::NativeAcceptedPrefixCommit);
    EXPECT_EQ(selected.native_destinations.front().bytes,128*sizeof(float));
    ASSERT_TRUE(stage.restoreVerifierStateCaptureRow(20,nullptr));
    for(int d=0;d<128;++d)EXPECT_EQ(partial.data()[d],7.25f);
}

/** @test Real retained QSA tails remain unchanged when a later compact request has invalid acceptance. */
TEST(QwenSparseRetainedPublication, CPUBatchedPublicationValidatesEveryRetainedRequestBeforeMutation)
{
    llaminar::v2::ThreadCountGuard threads(1);CPUDeviceContext context(DeviceId::cpu(),1);
    constexpr int requests=3,capacity=4,width=128;
    FP32Tensor source({requests*capacity,width}),retained({requests*capacity,width}),partial({requests,4*width});
    INT32Tensor replay({requests*3}),observed({requests*2}),prior({requests*388}),priorTraffic({requests*2});
    const std::array<int,requests> live{2,3,2},origin{0,2,5};
    std::fill_n(source.mutable_data(),source.numel(),19.25f);
    std::fill_n(partial.mutable_data(),partial.numel(),-53.75f);
    auto *state=static_cast<int *>(replay.raw_mutable_data());
    std::vector<std::unique_ptr<Stage>> owners;std::vector<IComputeStage *> stages;
    for(int request=0;request<requests;++request) {
        state[request*3]=1;state[request*3+1]=live[request];state[request*3+2]=17;
        const auto offset=std::size_t(request)*capacity*width*sizeof(float);
        const auto originalPrior=freezePrior(prior,priorTraffic,state+request*3,
            partial.data()+request*4*width,request,capacity,32);
        owners.push_back(std::make_unique<Stage>(Stage::Params{.device_id=DeviceId::cpu(),.rowCapacity=capacity,
            .tokenCapacity=32,.replay=originalPrior.storage,
            .operation=Stage::RetainCandidates{Float(source,BufferId::QFN_QSA_RAW_KEY,offset),
                Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS,offset),
                Int(observed,BufferId::QFN_QSA_CANDIDATE_COPY_BYTES,request*sizeof(std::uint64_t)),
                Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS,request*4*width*sizeof(float)),request,requests,
                RequestRowOrigins::nativeWords(origin.data()),originalPrior}}));
        ASSERT_TRUE(owners.back()->execute(&context));stages.push_back(owners.back().get());
    }
    MTPSpecStepPlanBatch batch;batch.ok=true;batch.request_count=requests;
    for(int request=0;request<requests;++request)batch.steps.push_back({.request_index=request,.draft_count=3,.accepted_count=1});
    const std::vector<float> original(partial.data(),partial.data()+partial.numel());
    const int invalid[requests]{0,2,7};
    auto result=publishAcceptedMTPSpecStateFromVerifierRows(batch,invalid,stages,DeviceId::cpu(),nullptr,true);
    EXPECT_FALSE(result.ok);
    EXPECT_EQ(std::memcmp(partial.data(),original.data(),partial.size_bytes()),0);
    const int valid[requests]{0,2,-1};
    result=publishAcceptedMTPSpecStateFromVerifierRows(batch,valid,stages,DeviceId::cpu(),nullptr,true);
    // Rejected request metadata must agree with its no-advance decision.
    EXPECT_FALSE(result.ok);EXPECT_EQ(std::memcmp(partial.data(),original.data(),partial.size_bytes()),0);
    batch.steps[2].accepted_count=0;
    result=publishAcceptedMTPSpecStateFromVerifierRows(batch,valid,stages,DeviceId::cpu(),nullptr,true);
    ASSERT_TRUE(result.ok)<<result.error;
    for(int request=0;request<requests;++request)for(int row=0;row<4;++row) {
        const float expected=request<2 && row==1 ? 19.25f : -53.75f;
        EXPECT_TRUE(std::all_of(partial.data()+(request*4+row)*width,
            partial.data()+(request*4+row+1)*width,[expected](float value){return value==expected;}));
    }
}

/** @test The dedicated SparseMTP counter role cannot be substituted by core read observations. */
TEST(QwenSparseRetainedPublication, RejectsBorrowingAttentionObservationCounterRole)
{
    FP32Tensor source({4,128}),retained({4,128}),partial({4,128});
    INT32Tensor replay({3}),counter({2}),prior({388});
    const auto params=[&](BufferId role) {
        return Stage::Params{.device_id=DeviceId::cpu(),.rowCapacity=4,.tokenCapacity=32,
            .replay=Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR),.operation=Stage::RetainCandidates{
                Float(source,BufferId::QFN_QSA_RAW_KEY),Float(retained,BufferId::QFN_QSA_CANDIDATE_RAW_KEYS),
                Int(counter,role),Float(partial,BufferId::QFN_QSA_PARTIAL_KEYS),
                0,1,RequestRowOrigins::fixedStride(0,4),
                {Int(prior,BufferId::QFN_QSA_TRANSACTION_PRIOR)}}};
    };
    EXPECT_THROW(Stage(params(BufferId::QFN_QSA_OBSERVATIONS)),std::invalid_argument);
    EXPECT_NO_THROW(Stage(params(BufferId::QFN_QSA_CANDIDATE_COPY_BYTES)));
}
