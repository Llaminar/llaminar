/**
 * @file Test__QwenSparseTerminalCounters.cpp
 * @brief Device-free canonical reset, exact byte arithmetic, alignment and overflow proof.
 *
 * These tests exercise the existing controller transition helpers and fixed-word
 * observations, without GPU calls, a model or linking a mixed prepared-engine ABI.
 */
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/QwenSparseTerminalCounters.h"
#include "kernels/qwen4exp/QwenSparseAttention.h"
#include <limits>
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <cstring>
using namespace llaminar2;using namespace llaminar2::sampling_math;using namespace llaminar2::qwen4exp;
namespace {
/** @brief Aligned canonical controller plus dirty guards excluded from reset/terminal bytes. */
struct Row {alignas(8)std::array<std::int32_t,kDeviceGenerationControlCount+4>words{};
    QwenSparseWorkCounterRow binding(){return {words.data(),kDeviceGenerationControlCount};}
    void admit(){ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::fixed(1),words.data()));}
};
}
/** @brief New requests and valid response continuation both reset reporting, never learner state. */
TEST(QwenSparseTerminalCounters, AdmissionAndContinuationResetDirtyTotalsWithoutChangingGuards){
    Row row;row.words.fill(0x12345678);row.admit();
    QwenSparseCounterTotals total;ASSERT_TRUE(readQwenSparseTerminalCounters(row.words.data(),kDeviceGenerationControlCount,total));
    EXPECT_EQ(total.prior_header|total.prior_payload|total.candidate_payload|total.accepted_payload,0u);
    EXPECT_EQ(row.words.back(),0x12345678);
    ASSERT_TRUE(addQwenSparseHostCounters(row.words.data(),kDeviceGenerationControlCount,{16,1536,2560,1536}));
    row.words[kDeviceGenerationControlRequestComplete]=1;row.words[kDeviceGenerationControlRemainingTokenCount]=0;
    row.words[kDeviceGenerationControlTransactionCount]=1;row.words[kDeviceGenerationControlTransactionCommitBudget]=0;
    DeviceGenerationDispatchTicket ticket{};ASSERT_TRUE(initialize_device_generation_dispatch_ticket(17,41,&ticket));
    ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::fixed(1),row.words.data(),
        DeviceGenerationLeadingRowDisposition::PendingResponse,{DeviceGenerationAdmissionKind::ContinueResponse,17,41,&ticket}));
    ASSERT_TRUE(readQwenSparseTerminalCounters(row.words.data(),kDeviceGenerationControlCount,total));
    EXPECT_EQ(total.prior_header|total.prior_payload|total.candidate_payload|total.accepted_payload,0u);
    EXPECT_EQ(row.words.back(),0x12345678);
    Row ordinary;ordinary.words.fill(-91);ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::ordinary(),ordinary.words.data()));
    ASSERT_TRUE(readQwenSparseTerminalCounters(ordinary.words.data(),kDeviceGenerationControlCount,total));
    EXPECT_EQ(total.prior_header|total.prior_payload|total.candidate_payload|total.accepted_payload,0u);
}
/** @brief Word pairs preserve low-word carry, reject bad alignment and never wrap on overflow. */
TEST(QwenSparseTerminalCounters, ExactPairsCarryAndOverflowPreserveAllEarlierWords){
    Row row;row.admit();auto counter=row.binding().metric(QwenSparseWorkCounter::CandidatePayload);
    ASSERT_TRUE(counter.valid());ASSERT_TRUE(addNativeCounter64(counter,0xffffffffull));ASSERT_TRUE(addNativeCounter64(counter,7));
    EXPECT_EQ(readNativeCounter64(counter.words),0x100000006ull);
    auto invalid=NativeCounter64{row.words.data()+1};ASSERT_FALSE(invalid.valid());const auto before=row.words;
    EXPECT_FALSE(addNativeCounter64(invalid,1));EXPECT_EQ(row.words,before);
    auto final=row.binding().metric(QwenSparseWorkCounter::AcceptedPayload);ASSERT_TRUE(addNativeCounter64(final,UINT64_MAX-2));
    const auto limit=row.words;EXPECT_FALSE(addNativeCounter64(final,3));EXPECT_EQ(row.words,limit);
    EXPECT_FALSE(addQwenSparseHostCounters(row.words.data(),kDeviceGenerationControlCount,{16,0,8,3}));EXPECT_EQ(row.words,limit);
    EXPECT_FALSE((QwenSparseWorkCounterRow{row.words.data(),52}).valid());
}
/** @brief QSA retains its exact32B; fixed-head48B, adaptive72B and immutable scope40B and utility state/events32B extend the same canonical row. */
TEST(QwenSparseTerminalCounters, TerminalAndCanonicalRowExtentsAddExactlyFourPairs){
    static_assert(kDeviceGenerationControlQsaPriorHeaderBytesLow==kDeviceGenerationControlInitialDraftDepth+1);
    static_assert(kDeviceGenerationControlCount*sizeof(std::int32_t)==432);
    static_assert(kDeviceGenerationControlDepthUtilityPhase==100);
    static_assert(kDeviceGenerationControlDepthUtilityProbes-kDeviceGenerationControlDepthUtilityPhase==5);
    static_assert(kDeviceGenerationControlCount-kDeviceGenerationControlDepthUtilityProbes==3);
    EXPECT_EQ((kDeviceGenerationControlQsaAcceptedPayloadBytesHigh+1-kDeviceGenerationControlQsaPriorHeaderBytesLow)*sizeof(std::int32_t),4*sizeof(std::uint64_t));
    for(int requests:{1,3,8}){
        const auto bytes=std::size_t(requests)*kDeviceGenerationControlCount*sizeof(std::int32_t);
        EXPECT_EQ(bytes,std::size_t(requests)*(208+32+48+72+40+20+12));
        EXPECT_EQ((kDeviceGenerationControlCount*sizeof(std::int32_t))%alignof(std::uint64_t),0u);
    }
}

/** @brief Actual CPU primitives report only live snapshot/candidate/tail bytes, including revised zero acceptance. */
TEST(QwenSparseTerminalCounters, ActualOperationsAccumulateRevisedAcceptanceAndEmptyTraffic){
    Row row;row.admit();auto kernels=createCPUQwenSparseAttentionKernels();QwenSparseReplay replay{7,5,91};
    std::array<float,512> accepted;std::array<float,5*128> raw,retained;
    for(int i=0;i<512;++i)accepted[i]=i+0.5f;
    for(int i=0;i<5*128;++i)raw[i]=1000+i;
    retained.fill(std::numeric_limits<float>::quiet_NaN());
    const auto original=accepted;QwenSparseTransactionPrior prior{};QwenSparsePriorTraffic traffic{};
    cpuQwenSparseSnapshotPrior({&replay,accepted.data(),&prior,&traffic,5,128,row.binding()});
    alignas(8)std::uint64_t copied=0;
    kernels->retainCandidates({raw.data(),retained.data(),&prior.replay.live_rows,&copied,5,row.binding()},nullptr);
    ASSERT_EQ(copied,sizeof(raw));ASSERT_EQ(std::memcmp(raw.data(),retained.data(),sizeof(raw)),0);
    const auto immutable=prior;
    std::uint64_t expectedWrites=0;
    for(int acceptedRows:{5,0,1,0}){
        auto acceptance=AcceptedRequestPrefixes::fixed(DeviceRequestRowRanges::deviceCounted(1,5,&prior.replay.live_rows),acceptedRows);
        cpuQwenSparseCommitPartial({&prior.replay,prior.partial,retained.data(),acceptance,
            accepted.data(),&prior,128,row.binding()});
        expectedWrites+=std::uint64_t((7+acceptedRows)%4)*128*4;
        QwenSparseCounterTotals observed;ASSERT_TRUE(readQwenSparseTerminalCounters(row.words.data(),kDeviceGenerationControlCount,observed));
        EXPECT_EQ(observed.prior_header,16u);EXPECT_EQ(observed.prior_payload,1536u);
        EXPECT_EQ(observed.candidate_payload,2560u);EXPECT_EQ(observed.accepted_payload,expectedWrites);
        EXPECT_EQ(std::memcmp(&prior,&immutable,sizeof(prior)),0);
    }
    EXPECT_EQ(std::memcmp(accepted.data(),original.data(),3*128*sizeof(float)),0);
    replay.live_rows=0;const auto frozen=accepted;const auto retainedBefore=retained;
    cpuQwenSparseSnapshotPrior({&replay,accepted.data(),&prior,&traffic,5,128,row.binding()});
    kernels->retainCandidates({raw.data(),retained.data(),&prior.replay.live_rows,&copied,5,row.binding()},nullptr);
    auto empty=AcceptedRequestPrefixes::fixed(DeviceRequestRowRanges::deviceCounted(1,5,&prior.replay.live_rows),0);
    cpuQwenSparseCommitPartial({&prior.replay,prior.partial,retained.data(),empty,accepted.data(),&prior,128,row.binding()});
    QwenSparseCounterTotals observed;ASSERT_TRUE(readQwenSparseTerminalCounters(row.words.data(),kDeviceGenerationControlCount,observed));
    EXPECT_EQ(observed.prior_header,32u);EXPECT_EQ(observed.prior_payload,1536u);EXPECT_EQ(observed.candidate_payload,2560u);
    EXPECT_EQ(observed.accepted_payload,expectedWrites);EXPECT_EQ(copied,0u);
    EXPECT_EQ(std::memcmp(accepted.data(),frozen.data(),sizeof(accepted)),0);
    EXPECT_EQ(std::memcmp(retained.data(),retainedBefore.data(),sizeof(retained)),0);
    row.admit();ASSERT_TRUE(readQwenSparseTerminalCounters(row.words.data(),kDeviceGenerationControlCount,observed));
    EXPECT_EQ(observed.prior_header|observed.prior_payload|observed.candidate_payload|observed.accepted_payload,0u);
}
/** @brief Canonical INT32 transaction budgets cannot silently overflow UINT64 observations. */
TEST(QwenSparseTerminalCounters, LifetimeBoundRejectsOverflowBeforeCapture){
    EXPECT_TRUE(validQwenSparseCounterLifetimeBound(48,16));
    EXPECT_TRUE(validQwenSparseCounterLifetimeBound(48,1024));
    EXPECT_FALSE(validQwenSparseCounterLifetimeBound(0,16));
    EXPECT_FALSE(validQwenSparseCounterLifetimeBound(48,0));
    EXPECT_FALSE(validQwenSparseCounterLifetimeBound(INT_MAX,INT_MAX));
}
