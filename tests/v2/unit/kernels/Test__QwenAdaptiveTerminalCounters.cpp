/**
 * @file Test__QwenAdaptiveTerminalCounters.cpp
 * @brief Device-free counter tuple, true owner transitions and reporting-window regressions.
 *
 * These call actual shared operations/controller helpers. Pure metadata response
 * validation is an algorithm-only stand-in; native source ACK, stream and stage
 * execution remain the explicit CUDA/ROCm integration obligation.
 */
#include "kernels/common/SamplingMath.h"
#include "kernels/qwen4exp/QwenAdaptiveTerminalCounters.h"
#include "qwen4exp/AdaptiveImmutableRowCacheTestFixture.h"
#include <array>
#include <cstring>
#include <gtest/gtest.h>
using namespace llaminar2;
using namespace llaminar2::qwen4exp;
using namespace llaminar2::sampling_math;
namespace
{
/** @brief Aligned original host-owned row plus guards outside the canonical extent. */
struct Row
{
    alignas(8) std::array<std::int32_t, kDeviceGenerationControlCount + 2> words{};
    /** @return Actual complete observation tuple under serialized fixture ownership. */
    QwenAdaptiveCounterTotals read() const
    {
        QwenAdaptiveCounterTotals v;
        EXPECT_TRUE(readQwenAdaptiveCounters(words.data(), kDeviceGenerationControlCount, v));
        return v;
    }
};
} // namespace
TEST(QwenAdaptiveTerminalCounters, ExactTupleOverflowFailsBeforeAnyWordChanges)
{
    static_assert(kDeviceGenerationControlAdaptiveCallsLow == 72);
    static_assert(kDeviceGenerationControlCount == 108);
    Row r;
    r.words.front() = 123;
    r.words.back() = 456;
    const QwenAdaptiveCounterTotals first{2, 32, 1, 1, 1, 90, 1, 0, 0};
    ASSERT_TRUE(accumulateQwenAdaptiveCounters(r.words.data(), kDeviceGenerationControlCount, first));
    EXPECT_EQ(r.read(), first);
    EXPECT_EQ(r.words.front(), 123);
    EXPECT_EQ(r.words.back(), 456);
    for (unsigned metric = 0; metric < 9; ++metric)
    {
        auto full = r;
        full.words[kDeviceGenerationControlAdaptiveCallsLow + 2 * metric] = -1;
        full.words[kDeviceGenerationControlAdaptiveCallsLow + 2 * metric + 1] = -1;
        const auto before = full.words;
        EXPECT_FALSE(accumulateQwenAdaptiveCounters(full.words.data(), kDeviceGenerationControlCount,
                                                    {1, 1, 1, 1, 1, 1, 1, 1, 1}));
        EXPECT_EQ(full.words, before);
    }
    const auto before = r.words;
    EXPECT_FALSE(accumulateQwenAdaptiveCounters(r.words.data(), 72, first));
    EXPECT_FALSE(accumulateQwenAdaptiveCounters(nullptr, kDeviceGenerationControlCount, first));
    EXPECT_EQ(r.words, before);
}
TEST(QwenAdaptiveTerminalCounters, SourceBytesAndCancellationWorkHaveExactDistinctSemantics)
{
    const QwenAdaptiveCounterTotals success{3, 32, 1, 1, 1, 90, 1, 0, 0};
    EXPECT_TRUE(validQwenAdaptiveCounters(success, 90, 16));
    auto bad = success;
    ++bad.source_bytes;
    EXPECT_FALSE(validQwenAdaptiveCounters(bad, 90, 16));
    bad = success;
    ++bad.completed_live_ids;
    EXPECT_FALSE(validQwenAdaptiveCounters(bad, 90, 16));
    bad = success;
    bad.promotions = 2;
    EXPECT_FALSE(validQwenAdaptiveCounters(bad, 90, 16));
    bad = success;
    bad.evictions = 2;
    EXPECT_FALSE(validQwenAdaptiveCounters(bad, 90, 16));
    EXPECT_TRUE(
        validQwenAdaptiveCounters({1, 0, 0, 1, 0, 0, 0, 1, 1}, 90, 16)); // cancelled after real eviction
    EXPECT_TRUE(
        validQwenAdaptiveCounters({1, 0, 0, 1, 1, 90, 1, 1, 1}, 90, 16)); // ACK/Ready preceded cancellation
    EXPECT_TRUE(validQwenAdaptiveCounters({1, 0, 0, 0, 0, 0, 0, 0, 1}, 90, 16)); // cancelled before policy
    EXPECT_FALSE(validQwenAdaptiveCounters({0, 0, 1, 0, 0, 0, 0, 0, 0}, 90, 16));
    EXPECT_FALSE(validQwenAdaptiveCounters({1, 0, 0, 1, 1, 90, 1, 1, 0}, 90, 16));
}
TEST(QwenAdaptiveTerminalCounters, OriginalOwnerCountsReadyAndEvictionRatherThanGenerationDifferences)
{
    namespace a = adaptive_lookup;
    using Fixture = test::adaptive::ActualMetadata<a::Slot32>;
    Fixture f(1);
    auto op = f.operations();
    const std::array<std::uint64_t, 1> cold{0}, hot{9};
    f.complete(cold);
    EXPECT_EQ(f.frame.promotions, 1u);
    EXPECT_EQ(f.frame.evictions, 0u);
    f.complete(hot);
    EXPECT_EQ(f.frame.promotions, 0u);
    EXPECT_EQ(f.frame.evictions, 0u); // declined equal frequency
    ASSERT_TRUE(op.begin(1));
    ASSERT_TRUE(op.beginPolicy(1));
    const auto reserved = op.observe(9);
    ASSERT_NE(reserved.slot, a::None);
    EXPECT_EQ(f.frame.evictions, 1u);
    EXPECT_EQ(f.frame.promotions, 0u);
    ASSERT_TRUE(op.prepare());
    EXPECT_FALSE(op.fail(a::Status::Cancelled));
    ASSERT_TRUE(op.release(reserved));
    EXPECT_FALSE(op.finish());
    EXPECT_EQ(f.frame.promotions, 0u);
    EXPECT_EQ(f.frame.evictions, 1u); // dropping Reserved is not promotion/eviction
    ASSERT_TRUE(op.begin(1));
    ASSERT_TRUE(op.beginPolicy(1));
    const auto ready = op.observe(9);
    ASSERT_NE(ready.slot, a::None);
    ASSERT_TRUE(op.prepare());
    f.validateResponse(1);
    ASSERT_TRUE(op.makeReady(ready));
    EXPECT_EQ(f.frame.promotions, 1u);
    EXPECT_EQ(f.frame.evictions, 0u);
    EXPECT_FALSE(op.fail(a::Status::Cancelled));
    ASSERT_TRUE(op.release(ready));
    EXPECT_FALSE(op.finish());
    EXPECT_NE(op.find(9), a::None);
    EXPECT_EQ(f.frame.promotions, 1u); // real Ready before cancellation survives
    f.complete(hot);
    EXPECT_EQ(f.frame.promotions, 0u);
    EXPECT_EQ(f.frame.evictions, 0u);
}
TEST(QwenAdaptiveTerminalCounters, InitialPrefillPreservesOnceContinuationAndRequestResetClearReporting)
{
    Row r;
    const QwenAdaptiveCounterTotals prefill{2, 32, 1, 1, 1, 90, 1, 0, 0};
    ASSERT_TRUE(accumulateQwenAdaptiveCounters(r.words.data(), kDeviceGenerationControlCount, prefill));
    DeviceGenerationInitialization initial;
    initial.forward_failure_policy = DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
    ASSERT_TRUE(initialize_device_generation_control(4, 4, DeviceGenerationPolicy::fixed(1), r.words.data(),
                                                     DeviceGenerationLeadingRowDisposition::PendingResponse,
                                                     initial));
    EXPECT_EQ(r.read(), prefill);
    r.words[kDeviceGenerationControlRequestComplete] = 1;
    r.words[kDeviceGenerationControlRemainingTokenCount] = 0;
    r.words[kDeviceGenerationControlTransactionCount] = 1;
    r.words[kDeviceGenerationControlTransactionCommitBudget] = 0;
    DeviceGenerationDispatchTicket ticket{};
    ASSERT_TRUE(initialize_device_generation_dispatch_ticket(17, 41, &ticket));
    DeviceGenerationInitialization next{DeviceGenerationAdmissionKind::ContinueResponse, 17, 41, &ticket};
    next.forward_failure_policy = DeviceGenerationForwardFailurePolicy::PreserveUntilRequestReset;
    ASSERT_TRUE(initialize_device_generation_control(4, 4, DeviceGenerationPolicy::fixed(1), r.words.data(),
                                                     DeviceGenerationLeadingRowDisposition::PendingResponse,
                                                     next));
    EXPECT_EQ(r.read(), QwenAdaptiveCounterTotals{});
    ASSERT_TRUE(accumulateQwenAdaptiveCounters(r.words.data(), kDeviceGenerationControlCount, prefill));
    ASSERT_TRUE(
        initialize_device_generation_control(4, 4, DeviceGenerationPolicy::ordinary(), r.words.data()));
    EXPECT_EQ(r.read(), QwenAdaptiveCounterTotals{});
}

/** @test Reporting remains exactly nine pairs while the whole ABI includes immutable scope and disjoint utility state/events. */
TEST(QwenAdaptiveTerminalCounters, ReportingExtentPreservesImmutableScopeAndRejectsLegacyTruncation)
{
    static_assert(kDeviceGenerationControlAdaptiveFailedFramesHigh + 1 ==
                  kDeviceGenerationControlLearnedScopeWidthPolicy);
    static_assert(kDeviceGenerationControlLearnedScopeComputeCapabilityMinor + 1 ==
                  kDeviceGenerationControlDepthUtilityPhase);
    static_assert(kDeviceGenerationControlDepthUtilityReturns + 1 == kDeviceGenerationControlCount);
    static_assert(kDeviceGenerationControlCount == 108);
    Row row;
    for (std::size_t word = 0; word < row.words.size(); ++word)
        if (word < kDeviceGenerationControlAdaptiveCallsLow || word > kDeviceGenerationControlAdaptiveFailedFramesHigh)
            row.words[word] = static_cast<std::int32_t>(0x12340000u + word);
    const auto original = row.words;
    const QwenAdaptiveCounterTotals delta{2, 32, 1, 1, 1, 90, 1, 0, 0};
    // The old reporting-end extent is no longer a complete controller. Both
    // readers and writers must reject it without mutating output or storage.
    auto unchanged = delta;
    EXPECT_FALSE(readQwenAdaptiveCounters(row.words.data(),
        kDeviceGenerationControlLearnedScopeWidthPolicy, unchanged));
    EXPECT_EQ(unchanged, delta);
    EXPECT_FALSE(accumulateQwenAdaptiveCounters(row.words.data(),
        kDeviceGenerationControlLearnedScopeWidthPolicy, delta));
    EXPECT_EQ(row.words, original);
    // The former100-word ABI includes scope but omits the utility owner;
    // reporting bindings still require the complete new controller extent.
    EXPECT_FALSE(readQwenAdaptiveCounters(row.words.data(), kDeviceGenerationControlDepthUtilityPhase, unchanged));
    EXPECT_EQ(unchanged, delta);
    EXPECT_FALSE(accumulateQwenAdaptiveCounters(row.words.data(), kDeviceGenerationControlDepthUtilityPhase, delta));
    EXPECT_EQ(row.words, original);
    ASSERT_TRUE(accumulateQwenAdaptiveCounters(row.words.data(), kDeviceGenerationControlCount, delta));
    EXPECT_EQ(row.read(), delta);
    for (std::size_t word = 0; word < row.words.size(); ++word)
        if (word < kDeviceGenerationControlAdaptiveCallsLow || word > kDeviceGenerationControlAdaptiveFailedFramesHigh)
            EXPECT_EQ(row.words[word], original[word]) << word;
    const auto complete = row.words;
    EXPECT_FALSE(accumulateQwenAdaptiveCounters(row.words.data(), kDeviceGenerationControlCount + 1, delta));
    EXPECT_EQ(row.words, complete);
}
