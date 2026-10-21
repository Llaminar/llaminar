/**
 * @file Test__QwenFixedHeadTerminalCounters.cpp
 * @brief Device-free exact terminal pair, request/window lifetime and native contribution semantics.
 *
 * Actual controller helpers are exercised on host-owned words. These tests do
 * not simulate a collective or claim native execution; the actual captured
 * flat/cohort native fixture owns contributed-byte proof on both backends.
 */
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/QwenFixedHeadTerminalCounters.h"
#include "execution/mtp/DeviceGenerationCollectiveReceipt.h"
#include <gtest/gtest.h>
#include <array>
#include <algorithm>
#include <limits>
#include <type_traits>
using namespace llaminar2;using namespace llaminar2::sampling_math;using namespace llaminar2::qwen4exp;
namespace {
/** @brief Original aligned word row with guards excluded from every canonical extent. */
struct Row {
    alignas(8) std::array<std::int32_t,kDeviceGenerationControlCount+4> words{};
    /** @return Exact typed borrowed row; no allocation or observation occurs. */
    QwenFixedHeadCounterRow binding(){return {words.data(),kDeviceGenerationControlCount};}
    /** @return Exact immutable terminal projection after this fixture's serialized operations. */
    QwenFixedHeadNativeSendTotals totals(){QwenFixedHeadNativeSendTotals out;
        EXPECT_TRUE(readQwenFixedHeadNativeSendCounters(words.data(),kDeviceGenerationControlCount,out));return out;}
};
/** @brief Author only host-owned observation values for controller-transition testing. */
void totals(Row &row,std::uint64_t calls,std::uint64_t ids,std::uint64_t bytes) {
    ASSERT_TRUE(addNativeCounter64(row.binding().metric(QwenFixedHeadCounter::Calls),calls));
    ASSERT_TRUE(addNativeCounter64(row.binding().metric(QwenFixedHeadCounter::LiveIds),ids));
    ASSERT_TRUE(addNativeCounter64(row.binding().metric(QwenFixedHeadCounter::NativeBytes),bytes));
    ASSERT_TRUE(addNativeCounter64(row.binding().metric(QwenFixedHeadCounter::HeaderNativeSendBytes),calls*8));
    ASSERT_TRUE(addNativeCounter64(row.binding().metric(QwenFixedHeadCounter::PayloadNativeSendBytes),bytes));
}
}
/** @test Original controller ordinals/QSA32B stay fixed; fixed observations add exactly48B/request. */
TEST(QwenFixedHeadTerminalCounters, CanonicalExtentAddsExactlySixPairsAndRejectsRawReceiptPointers)
{
    static_assert(kDeviceGenerationControlInitialDraftDepth==51);
    static_assert(kDeviceGenerationControlQsaPriorHeaderBytesLow==52);
    static_assert(kDeviceGenerationControlQsaAcceptedPayloadBytesHigh==59);
    static_assert(kDeviceGenerationControlCount == 90);
    static_assert(!std::is_constructible_v<DeviceGenerationCollectiveReceipt,void *>);
    for(const std::size_t requests:{1u,3u,8u}) {
        EXPECT_EQ(requests * kDeviceGenerationControlCount * sizeof(int), requests * (240u + 48u + 72u));
        EXPECT_EQ((requests*kDeviceGenerationControlCount*sizeof(int))%alignof(std::uint64_t),0u);
    }
    Row row;QwenFixedHeadNativeSendTotals unchanged{1,2,3,4,5,6};const auto original=unchanged;
    EXPECT_FALSE(readQwenFixedHeadNativeSendCounters(row.words.data(),60,unchanged));EXPECT_EQ(unchanged,original);
    EXPECT_FALSE(readQwenFixedHeadNativeSendCounters(nullptr,kDeviceGenerationControlCount,unchanged));EXPECT_EQ(unchanged,original);
}
/** @test Initial response admission retains real prefill work once; continuation resets only its observation window. */
TEST(QwenFixedHeadTerminalCounters, PrefillSurvivesInitialAdmissionAndContinuationCannotDoublePublish)
{
    Row row;row.words.back()=0x12345678;totals(row,2,24,24*90);const auto prefill=row.totals();
    DeviceGenerationInitialization initial;initial.forward_failure_policy=DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
    ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::fixed(1),row.words.data(),
        DeviceGenerationLeadingRowDisposition::PendingResponse,initial));
    EXPECT_EQ(row.totals(),prefill);EXPECT_EQ(row.words.back(),0x12345678);
    totals(row,1,8,8*90);EXPECT_EQ(row.totals().calls,3u);
    row.words[kDeviceGenerationControlRequestComplete]=1;row.words[kDeviceGenerationControlRemainingTokenCount]=0;
    row.words[kDeviceGenerationControlTransactionCount]=1;row.words[kDeviceGenerationControlTransactionCommitBudget]=0;
    DeviceGenerationDispatchTicket ticket{};ASSERT_TRUE(initialize_device_generation_dispatch_ticket(17,41,&ticket));
    DeviceGenerationInitialization next{DeviceGenerationAdmissionKind::ContinueResponse,17,41,&ticket};
    next.forward_failure_policy=DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
    ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::fixed(1),row.words.data(),
        DeviceGenerationLeadingRowDisposition::PendingResponse,next));
    EXPECT_EQ(row.totals(),QwenFixedHeadNativeSendTotals{});EXPECT_EQ(row.words.back(),0x12345678);
    totals(row,1,8,8*90);EXPECT_EQ(row.totals().calls,1u);
    ASSERT_TRUE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::ordinary(),row.words.data()));
    EXPECT_EQ(row.totals(),QwenFixedHeadNativeSendTotals{});
}
/** @test Header progress includes empty/error work, while source rows and payload exclude receive/decode expansion. */
TEST(QwenFixedHeadTerminalCounters, OwnedSourceContributionRelationsAreExactAndOverflowSafe)
{
    for(const int participants:{2,4,8})for(const int live:{0,1,3,15}) {
        const std::uint64_t ids=static_cast<std::uint64_t>(live)*(16/participants);
        QwenFixedHeadCounterTotals value{1,ids,ids*90,0,8,ids*90};
        EXPECT_TRUE(validQwenFixedHeadContributions(value,90,16/participants));
        auto bad=value;bad.payload_bytes++;EXPECT_FALSE(validQwenFixedHeadContributions(bad,90,16/participants));
        bad=value;bad.header_bytes++;EXPECT_FALSE(validQwenFixedHeadContributions(bad,90,16/participants));
        bad=value;bad.native_bytes*=participants;bad.payload_bytes=bad.native_bytes;
        if(ids) {EXPECT_FALSE(validQwenFixedHeadContributions(bad,90,16/participants));}
    }
    EXPECT_TRUE(validQwenFixedHeadContributions({1,0,0,1,8,0},90,8));
    EXPECT_FALSE(validQwenFixedHeadContributions({1,0,0,2,8,0},90,8));
    EXPECT_FALSE(validQwenFixedHeadContributions({UINT64_MAX,0,0,0,0,0},90,8));
    EXPECT_FALSE(validQwenFixedHeadContributions({1,UINT64_MAX,0,0,8,0},90,8));
    // Coherent byte equations cannot make25 owned heads a whole TP2 token prefix.
    EXPECT_FALSE(validQwenFixedHeadContributions({1,25,2250,0,8,2250},90,8));
    EXPECT_FALSE(validQwenFixedHeadContributions({1,8,720,0,8,720},90,3));
    EXPECT_TRUE(validQwenFixedHeadCounterLifetime(262144,8,640));
    EXPECT_FALSE(validQwenFixedHeadCounterLifetime(INT_MAX,8,640));
    EXPECT_FALSE(validQwenFixedHeadCounterLifetime(1,3,90));
    EXPECT_FALSE(validQwenFixedHeadCounterLifetime(1,8,std::numeric_limits<std::size_t>::max()));
    Row row;auto counter=row.binding().metric(QwenFixedHeadCounter::NativeBytes);
    ASSERT_TRUE(addNativeCounter64(counter,0xffffffffull));ASSERT_TRUE(addNativeCounter64(counter,7));
    EXPECT_EQ(row.totals().native_bytes,0x100000006ull);
    ASSERT_TRUE(addNativeCounter64(counter,UINT64_MAX-row.totals().native_bytes));const auto before=row.words;
    EXPECT_FALSE(addNativeCounter64(counter,1));EXPECT_EQ(row.words,before);
    EXPECT_FALSE((QwenFixedHeadCounterRow{row.words.data()+1,kDeviceGenerationControlCount}).valid());
}
/** @test A failed prefill remains failed at response admission and is never turned into a healthy terminal observation. */
TEST(QwenFixedHeadTerminalCounters, FailedPrefillCannotBeClearedByResponseAdmission)
{
    Row row;totals(row,1,0,0);ASSERT_TRUE(addNativeCounter64(row.binding().metric(QwenFixedHeadCounter::Failures),1));
    row.words[kDeviceGenerationControlErrorCode]=static_cast<int>(DeviceGenerationError::ImmutableStorageFailure);
    const auto failed=row.totals();DeviceGenerationInitialization initial;
    initial.forward_failure_policy=DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
    EXPECT_FALSE(initialize_device_generation_control(4,4,DeviceGenerationPolicy::ordinary(),row.words.data(),
        DeviceGenerationLeadingRowDisposition::PendingResponse,initial));
    EXPECT_EQ(row.totals(),failed);EXPECT_NE(row.words[kDeviceGenerationControlErrorCode],0);
}

/** @test Actual outgoing sends must exactly match the audited ring multiplicity before logical publication. */
TEST(QwenFixedHeadTerminalCounters, NativeRingSendProjectionRejectsMissingExcessAndNondivisibleTraffic)
{
    for(const int p:{2,4,8})for(const int live:{0,1,15})for(const bool error:{false,true}) {
        const auto ids=error?0ull:static_cast<unsigned long long>(live)*(16/p);
        const QwenFixedHeadNativeSendTotals raw{2,ids,ids*90,static_cast<std::uint64_t>(error),16u*(p-1),ids*90*(p-1)};
        QwenFixedHeadCounterTotals out{99,99,99,99,99,99};
        ASSERT_TRUE(projectQwenFixedHeadTerminalContributions(raw,p,90,out));
        EXPECT_EQ(out,(QwenFixedHeadCounterTotals{2,ids,ids*90,static_cast<std::uint64_t>(error),16,ids*90}));
        const auto unchanged=out;
        const auto reject=[&](auto bad){EXPECT_FALSE(projectQwenFixedHeadTerminalContributions(bad,p,90,out));EXPECT_EQ(out,unchanged);};
        auto bad=raw;--bad.header_send_bytes;reject(bad);bad=raw;++bad.header_send_bytes;reject(bad);
        bad=raw;bad.header_send_bytes+=8*(p-1);reject(bad); // Divisible but unexecuted extra header.
        bad=raw;bad.payload_send_bytes+=90*(p-1);reject(bad); // Divisible but unexecuted payload.
        bad=raw;++bad.payload_send_bytes;reject(bad);
        if(ids){bad=raw;--bad.payload_send_bytes;reject(bad);}
    }
    QwenFixedHeadCounterTotals out{1,2,3,4,5,6};const auto unchanged=out;
    EXPECT_FALSE(projectQwenFixedHeadTerminalContributions({1,25,2250,0,8,2250},2,90,out));
    EXPECT_FALSE(projectQwenFixedHeadTerminalContributions({},3,90,out));
    EXPECT_FALSE(projectQwenFixedHeadTerminalContributions({UINT64_MAX/8,0,0,0,0,0},8,90,out));
    EXPECT_FALSE(projectQwenFixedHeadTerminalContributions({1,UINT64_MAX/2,UINT64_MAX/2,0,56,0},8,1,out));
    EXPECT_EQ(out,unchanged);
}
