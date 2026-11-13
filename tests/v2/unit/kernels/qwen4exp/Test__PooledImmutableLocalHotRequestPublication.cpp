/**
 * @file Test__PooledImmutableLocalHotRequestPublication.cpp
 * @brief Exercise the actual request publication owner through repeats, reset and partial failure.
 *
 * Callbacks stand for ordered native enqueue/publication and exact-event acquire.
 * These device-free lifecycle tests prove once-request submission; native tests
 * separately prove the producer event and device cache reset arithmetic.
 */
#include "execution/qwen4exp/PooledImmutableLocalHotRequestPublication.h"
#include <gtest/gtest.h>
#include <array>
#include <limits>
#include <stdexcept>
#include <vector>

namespace llaminar2::qwen4exp {
TEST(PooledImmutableLocalHotRequestPublication, RepeatedPrefillAndMTPEntriesOnlyAcquireOriginalPublication)
{
    PooledImmutableLocalHotRequestPublication owner;
    std::vector<int> operations;
    // Different normal forward entries may use different actual streams. The
    // request source grant is still one producer; every caller must acquire it.
    for (const auto stream : std::array{7, 8, 8, 9, 7})
        owner.enter(12, [&] { operations.push_back(-stream); },
            [&] { operations.push_back(1); }, [&] { operations.push_back(stream); });
    EXPECT_EQ(operations, (std::vector<int>{-7, 1, 7, -8, 8, -8, 8, -9, 9, -7, 7}));
    owner.enter(13, [&] { operations.push_back(-8); },
        [&] { operations.push_back(2); }, [&] { operations.push_back(8); });
    EXPECT_EQ(operations, (std::vector<int>{-7, 1, 7, -8, 8, -8, 8, -9, 9, -7, 7, -8, 2, 8}));
}

TEST(PooledImmutableLocalHotRequestPublication, RejectsStaleEpochWithoutSubmittingAndNeverWraps)
{
    PooledImmutableLocalHotRequestPublication owner;
    int published = 0, acquired = 0;
    const auto publish = [&] { ++published; };
    const auto acquire = [&] { ++acquired; };
    EXPECT_THROW(owner.enter(0, [] {}, publish, acquire), std::logic_error);
    owner.enter(3, [] {}, publish, acquire);
    EXPECT_THROW(owner.enter(2, [] {}, publish, acquire), std::logic_error);
    owner.enter(3, [] {}, publish, acquire);
    owner.enter(std::numeric_limits<std::uint64_t>::max(), [] {}, publish, acquire);
    EXPECT_THROW(owner.enter(1, [] {}, publish, acquire), std::logic_error);
    EXPECT_EQ(published, 2);
    EXPECT_EQ(acquired, 3);
}

TEST(PooledImmutableLocalHotRequestPublication, FailedNativeSubmissionAndFailedAcquireRemainSticky)
{
    for (bool fail_publish : {false, true}) {
        PooledImmutableLocalHotRequestPublication owner;
        int published = 0, acquired = 0;
        const auto publish = [&] {
            ++published;
            if (fail_publish) throw std::runtime_error("partial native enqueue");
        };
        const auto acquire = [&] {
            ++acquired;
            throw std::runtime_error("producer event acquisition");
        };
        EXPECT_THROW(owner.enter(4, [] {}, publish, acquire), std::runtime_error);
        EXPECT_THROW(owner.enter(4, [] {}, publish, acquire), std::logic_error);
        EXPECT_THROW(owner.enter(5, [] {}, publish, acquire), std::logic_error);
        EXPECT_EQ(published, 1);
        EXPECT_EQ(acquired, fail_publish ? 0 : 1);
    }
}

TEST(PooledImmutableLocalHotRequestPublication, RecursiveProducerCannotDoubleSubmitRequest)
{
    PooledImmutableLocalHotRequestPublication owner;
    int published = 0, acquired = 0;
    const auto acquire = [&] { ++acquired; };
    const auto publish = [&] {
        ++published;
        owner.enter(9, [] {}, [&] { ++published; }, acquire);
    };
    EXPECT_THROW(owner.enter(9, [] {}, publish, acquire), std::logic_error);
    EXPECT_THROW(owner.enter(10, [] {}, [] {}, acquire), std::logic_error);
    EXPECT_EQ(published, 1);
    EXPECT_EQ(acquired, 0);
}

TEST(PooledImmutableLocalHotRequestPublication, UnacquiredControllerCannotEnqueueAdmissionOrMaskFailure)
{
    PooledImmutableLocalHotRequestPublication owner;
    int published = 0, acquired = 0;
    const auto controller = [] { throw std::runtime_error("original controller event unavailable"); };
    const auto publish = [&] { ++published; };
    const auto acquire = [&] { ++acquired; };
    EXPECT_THROW(owner.enter(4, controller, publish, acquire), std::runtime_error);
    EXPECT_EQ(published, 0);
    EXPECT_EQ(acquired, 0);
    EXPECT_THROW(owner.enter(5, [] {}, publish, acquire), std::logic_error);
}

TEST(PooledImmutableLocalHotRequestPublication, OriginalTerminalAllowsNewIntervalInSameSessionWithoutClearingRoots)
{
    PooledImmutableLocalHotRequestPublication owner;
    int published = 0, acquired = 0;
    const auto publish = [&] { ++published; };
    const auto acquire = [&] { ++acquired; };
    owner.enter(7, [] {}, publish, acquire);
    owner.enter(7, [] {}, publish, acquire);
    EXPECT_EQ(published, 1);
    EXPECT_THROW(owner.retireAfterTerminal(6), std::logic_error);
    owner.retireAfterTerminal(7);
    owner.retireAfterTerminal(7); // Original grouped terminal aggregation is idempotent.
    owner.enter(7, [] {}, publish, acquire);
    EXPECT_EQ(published, 2);
    EXPECT_EQ(acquired, 3);
    owner.retireAfterTerminal(7);
    EXPECT_THROW(owner.enter(8, [] {}, [] { throw std::runtime_error("partial enqueue"); }, acquire), std::runtime_error);
    owner.retireAfterTerminal(8);
    EXPECT_THROW(owner.enter(9, [] {}, publish, acquire), std::logic_error);
}
} // namespace llaminar2::qwen4exp
