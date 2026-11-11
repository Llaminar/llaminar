/**
 * @file Test__OpenMPWorkshare.cpp
 * @brief Device-free lexical retained-team semantics for the shared workshare helpers.
 *
 * The shared OpenMP helper family must reuse the exact enclosing team whenever a
 * participant is inside any parallel region -- active, serialized
 * (num_threads(1)) or inactive (if(false)) -- instead of forking an unwanted
 * inner team based on the active-only omp_in_parallel() predicate. Every test
 * observes the predicates from INSIDE the work function: omp_get_level(),
 * omp_in_parallel(), omp_get_num_threads() and the canonical
 * llaminar::v2::teamPlacement()/inLexicalWorkshare() authority. The Active
 * label carries the exact OpenMP meaning (enclosed by an active region; an
 * active ancestor makes a nested-serialized one-thread team report active) and
 * never claims current team width > 1. Standalone policy (condition-true full
 * team, condition-false 1-thread region, empty calls without a team/barrier)
 * is unchanged. This target is device-free and does not certify factories,
 * model inference or economy.
 */
#include "utils/OpenMPUtils.h"
#include <gtest/gtest.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace llaminar2::test
{
namespace
{
using llaminar::v2::TeamPlacement;
using llaminar::v2::OpenMPTeamForkScope;
using llaminar::v2::beginWorkshare;
using llaminar::v2::inActiveParallelRegion;
using llaminar::v2::inLexicalWorkshare;
using llaminar::v2::isInParallelRegion;
using llaminar::v2::noteOpenMPTeamFork;
using llaminar::v2::openMPTeamForkSnapshot;
using llaminar::v2::teamPlacement;

/** @brief Keep requested retained teams exact without leaking caller policy. */
class TeamPolicy final
{
public:
    TeamPolicy() : threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic()) { omp_set_dynamic(0); }
    ~TeamPolicy() { omp_set_num_threads(threads_); omp_set_dynamic(dynamic_); }
private:
    int threads_, dynamic_;
};
/** @brief Keep a requested thread-count ICV while a fixture runs. */
class NumThreadsGuard final
{
public:
    explicit NumThreadsGuard(int threads) : previous_(omp_get_max_threads()) { omp_set_num_threads(threads); }
    ~NumThreadsGuard() { omp_set_num_threads(previous_); }
private:
    int previous_;
};
} // namespace

/** @test Standalone conditional-true uses the exact admitted width; false one thread. */
TEST(OpenMPWorkshare, StandalonePolicyIsUnchanged)
{
    const TeamPolicy policy;
    const NumThreadsGuard admitted(3);
    std::atomic<int> iterations{0}, wrong{0};
    auto full = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 8; ++i)
        {
            // The fork must use the exact admitted team width (3), not merely >1.
            if (omp_get_level() != 1 || omp_get_num_threads() != 3)
                wrong.fetch_add(1, std::memory_order_relaxed);
            iterations.fetch_add(1, std::memory_order_relaxed);
        }
    };
    OMP_WORKSHARE_REGION_IF(full, true);
    EXPECT_EQ(iterations.load(), 8);
    auto single = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 8; ++i)
        {
            if (omp_get_level() != 1 || omp_get_num_threads() != 1)
                wrong.fetch_add(1, std::memory_order_relaxed);
            iterations.fetch_add(1, std::memory_order_relaxed);
        }
    };
    OMP_WORKSHARE_REGION_IF(single, false);
    EXPECT_EQ(iterations.load(), 16);
    EXPECT_EQ(wrong.load(), 0);
}

/** @test Retained active team: the work function reuses the exact 3-thread team. */
TEST(OpenMPWorkshare, RetainedActiveTeamReusesExactTeam)
{
    const TeamPolicy policy;
    std::atomic<int> iterations{0}, badDepth{0};
    auto work = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 9; ++i)
        {
            // No nested region: the work body sees the outer team directly.
            EXPECT_EQ(omp_get_num_threads(), 3);
            EXPECT_EQ(omp_get_level(), 1);
            EXPECT_EQ(omp_in_parallel(), 1);
            if (omp_get_level() != 1) badDepth.fetch_add(1, std::memory_order_relaxed);
            iterations.fetch_add(1, std::memory_order_relaxed);
        }
    };
    int outerTeam = 0;
#pragma omp parallel num_threads(3)
    {
        if (omp_get_thread_num() == 0) outerTeam = omp_get_num_threads();
        OMP_WORKSHARE_REGION(work);
    }
    EXPECT_EQ(outerTeam, 3);
    EXPECT_EQ(iterations.load(), 9);
    EXPECT_EQ(badDepth.load(), 0);
}

/** @test Retained one-thread team (no active ancestor) never forks an inner region. */
TEST(OpenMPWorkshare, RetainedSerializedTeamReusesExactTeam)
{
    const TeamPolicy policy;
    std::atomic<int> iterations{0}, wrong{0};
    auto work = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 3; ++i)
        {
            // Old active-only predicate forked a new region at a deeper level.
            if (omp_get_level() != 1 || omp_get_num_threads() != 1)
                wrong.fetch_add(1, std::memory_order_relaxed);
            EXPECT_FALSE(inActiveParallelRegion());
            EXPECT_TRUE(teamPlacement() == TeamPlacement::Retained);
            EXPECT_TRUE(inLexicalWorkshare());
            iterations.fetch_add(1, std::memory_order_relaxed);
        }
    };
#pragma omp parallel num_threads(1)
    {
        // Absent an active ancestor, a top-level one-thread region is inactive.
        EXPECT_EQ(omp_get_level(), 1);
        EXPECT_FALSE(omp_in_parallel());
        EXPECT_TRUE(teamPlacement() == TeamPlacement::Retained);
        EXPECT_FALSE(inActiveParallelRegion());
        OMP_WORKSHARE_REGION(work);
    }
    EXPECT_EQ(iterations.load(), 3);
    EXPECT_EQ(wrong.load(), 0);
}

/** @test Retained inactive (if(false)) team never forks an inner region. */
TEST(OpenMPWorkshare, RetainedInactiveTeamReusesExactTeam)
{
    const TeamPolicy policy;
    std::atomic<int> iterations{0}, wrong{0};
    auto work = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 3; ++i)
        {
            if (omp_get_level() != 1 || omp_get_num_threads() != 1)
                wrong.fetch_add(1, std::memory_order_relaxed);
            iterations.fetch_add(1, std::memory_order_relaxed);
        }
    };
#pragma omp parallel if(false)
    {
        EXPECT_EQ(omp_get_level(), 1);
        EXPECT_FALSE(omp_in_parallel());
        EXPECT_TRUE(teamPlacement() == TeamPlacement::Retained);
        OMP_WORKSHARE_REGION(work);
    }
    EXPECT_EQ(iterations.load(), 3);
    EXPECT_EQ(wrong.load(), 0);
}

/** @test REGION_IF retained variants: active3, inactive and nested stay direct. */
TEST(OpenMPWorkshare, RegionIfRetainedVariantsStayDirect)
{
    const TeamPolicy policy;
    std::atomic<int> iterations{0}, wrong{0};
    int expectedLevel = 1;
    auto work = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 6; ++i)
        {
            if (omp_get_level() != expectedLevel) wrong.fetch_add(1, std::memory_order_relaxed);
            iterations.fetch_add(1, std::memory_order_relaxed);
        }
    };
#pragma omp parallel num_threads(3)
    {
        OMP_WORKSHARE_REGION_IF(work, true);
        OMP_WORKSHARE_REGION_IF(work, false);
    }
    EXPECT_EQ(iterations.load(), 12);
    EXPECT_EQ(wrong.load(), 0);
    iterations.store(0);
#pragma omp parallel if(false)
    {
        OMP_WORKSHARE_REGION_IF(work, true);
        OMP_WORKSHARE_REGION_IF(work, false);
    }
    EXPECT_EQ(iterations.load(), 12);
    EXPECT_EQ(wrong.load(), 0);
    iterations.store(0);
    expectedLevel = 2;
#pragma omp parallel num_threads(2)
    {
        if (omp_get_num_threads() != 2) wrong.fetch_add(1000, std::memory_order_relaxed);
#pragma omp parallel num_threads(1)
        {
            OMP_WORKSHARE_REGION_IF(work, true);
            OMP_WORKSHARE_REGION_IF(work, false);
        }
    }
    // Two outer threads each create a serialized inner team, and each inner
    // team runs both 6-iteration workshares: 2 * (6 + 6) = 24.
    EXPECT_EQ(iterations.load(), 24);
    EXPECT_EQ(wrong.load(), 0);
}

/** @test Nested serialized retained team authenticates both widths and placement helpers. */
TEST(OpenMPWorkshare, NestedSerializedTeamDepthIsStable)
{
    const TeamPolicy policy;
    std::atomic<int> wrong{0};
    auto work = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 2; ++i)
            // Outer active (1) + inner serialized (2): no third fork.
            if (omp_get_level() != 2) wrong.fetch_add(1, std::memory_order_relaxed);
    };
#pragma omp parallel num_threads(2)
    {
        int team = omp_get_num_threads();
        if (team != 2) wrong.fetch_add(1000, std::memory_order_relaxed);
#pragma omp parallel num_threads(1)
        {
            int innerWidth = omp_get_num_threads();
            if (innerWidth != 1) wrong.fetch_add(2000, std::memory_order_relaxed);
            // Exact OpenMP meaning: an active ancestor exists, so the Active
            // label applies even though the current team has one thread.
            if (!(teamPlacement() == TeamPlacement::Active)) wrong.fetch_add(3000, std::memory_order_relaxed);
            if (!inLexicalWorkshare()) wrong.fetch_add(4000, std::memory_order_relaxed);
            if (!isInParallelRegion()) wrong.fetch_add(5000, std::memory_order_relaxed);
            if (!inActiveParallelRegion()) wrong.fetch_add(6000, std::memory_order_relaxed);
            OMP_WORKSHARE_REGION(work);
        }
    }
    EXPECT_EQ(wrong.load(), 0);
}

/** @test COLLAPSE2/3 forms reuse the retained team and condition false stays direct. */
TEST(OpenMPWorkshare, CollapseFormsReuseRetainedTeam)
{
    const TeamPolicy policy;
    std::atomic<int> iterations{0}, wrong{0};
    auto work2 = [&]() {
#pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 3; ++j)
            {
                if (omp_get_num_threads() != 3 || omp_get_level() != 1) wrong.fetch_add(1, std::memory_order_relaxed);
                iterations.fetch_add(1, std::memory_order_relaxed);
            }
    };
    auto work3 = [&]() {
#pragma omp for collapse(3) schedule(static)
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 2; ++k)
                {
                    if (omp_get_num_threads() != 3 || omp_get_level() != 1) wrong.fetch_add(1, std::memory_order_relaxed);
                    iterations.fetch_add(1, std::memory_order_relaxed);
                }
    };
#pragma omp parallel num_threads(3)
    {
        OMP_WORKSHARE_COLLAPSE2_IF(work2, true);
        OMP_WORKSHARE_COLLAPSE3_IF(work3, false);
    }
    EXPECT_EQ(iterations.load(), 12 + 12);
    EXPECT_EQ(wrong.load(), 0);
}

/** @test beginWorkshare: standalone empty returns; retained empty rendezvous before outer join. */
TEST(OpenMPWorkshare, BeginWorkshareEmptySemantics)
{
    const TeamPolicy policy;
    // Standalone empty: no team, no barrier, plain return.
    EXPECT_FALSE(beginWorkshare(false));
    // Retained active3 entry witness: every member reaches the empty boundary
    // before any member may return (entry rendezvous); this does not by itself
    // prove post-work completion of a later barrier.
    std::atomic<int> arrived{0}, early{0};
#pragma omp parallel num_threads(3)
    {
        if (omp_get_thread_num() == 2) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        arrived.fetch_add(1, std::memory_order_release);
        EXPECT_FALSE(beginWorkshare(false));
        if (arrived.load(std::memory_order_acquire) != 3) early.fetch_add(1, std::memory_order_relaxed);
    }
    EXPECT_EQ(arrived.load(), 3);
    EXPECT_EQ(early.load(), 0);
    // Retained serialized/inactive empty boundaries are valid no-ops.
#pragma omp parallel num_threads(1)
    { EXPECT_FALSE(beginWorkshare(false)); EXPECT_EQ(omp_get_level(), 1); }
#pragma omp parallel if(false)
    { EXPECT_FALSE(beginWorkshare(false)); EXPECT_EQ(omp_get_level(), 1); }
}

/** @test Repeated retained calls keep one inner depth and no per-call fork. */
TEST(OpenMPWorkshare, RepeatedRetainedCallsKeepConstantDepth)
{
    const TeamPolicy policy;
    std::atomic<int> iterations{0}, wrong{0};
    auto work = [&]() {
#pragma omp for schedule(static)
        for (int i = 0; i < 6; ++i)
        {
            if (omp_get_num_threads() != 3 || omp_get_level() != 1)
                wrong.fetch_add(1, std::memory_order_relaxed);
            iterations.fetch_add(1, std::memory_order_relaxed);
        }
    };
#pragma omp parallel num_threads(3)
    {
        for (int repeat = 0; repeat < 10; ++repeat) OMP_WORKSHARE_REGION(work);
    }
    EXPECT_EQ(iterations.load(), 60);
    EXPECT_EQ(wrong.load(), 0);
}

/** @test Cross-member completion is verified by every member before the outer join. */
TEST(OpenMPWorkshare, ImmediateAfterReturnBeforeOuterJoin)
{
    const TeamPolicy policy;
    constexpr std::size_t kSlots = 12, kGuard = 16;
    std::array<std::atomic<std::uint64_t>, kGuard> shared;
    std::array<std::atomic<int>, kGuard> owner;
    for (std::size_t i = 0; i < kGuard; ++i)
    {
        shared[i].store(0xfeedfacecafebeefull, std::memory_order_relaxed);
        owner[i].store(-1, std::memory_order_relaxed);
    }
    std::atomic<int> recordedTeam{0}, missing{0}, corrupted{0}, mastered{0}, skewFailed{0};
    auto work = [&]() {
        const int team = omp_get_num_threads();
        recordedTeam.store(team, std::memory_order_relaxed);
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static)
        for (std::size_t i = 0; i < kSlots; ++i)
        {
            // Condition-based skew: the member owning the last (static, default
            // chunk = ceil(N/team)) chunk waits until every other member has
            // written its own slots (50us polling, ~200ms hard cap with a fail
            // flag), then a fixed 2ms before writing. Oversubscribed cores do
            // not turn this into a long spin.
            if (i == kSlots - 4)
            {
                bool othersDone = false;
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                do {
                    othersDone = true;
                    for (std::size_t j = 0; j < kSlots - 4; ++j)
                        if (owner[j].load(std::memory_order_acquire) < 0) { othersDone = false; break; }
                    if (!othersDone) std::this_thread::sleep_for(std::chrono::microseconds(50));
                } while (!othersDone && std::chrono::steady_clock::now() < deadline);
                if (!othersDone) skewFailed.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            owner[i].store(tid, std::memory_order_release);
            shared[i].store(std::uint64_t(0xc0ffee0000000000ull | i), std::memory_order_release);
        }
    };
#pragma omp parallel num_threads(3)
    {
        OMP_WORKSHARE_REGION(work);
        // Immediately after return, before the outer join and with NO masking
        // barrier, EVERY member verifies ALL 12 slots (written by the
        // distributed static schedule, each slot owned by one member), the
        // delayed last chunk's non-master ownership, and the guard region.
        int localMissing = 0;
        bool corrupt = false;
        for (std::size_t i = 0; i < kGuard; ++i)
        {
            const auto value = shared[i].load(std::memory_order_acquire);
            const auto who = owner[i].load(std::memory_order_acquire);
            if (i < kSlots)
            {
                if (value != std::uint64_t(0xc0ffee0000000000ull | i) || who < 0) ++localMissing;
                if (i >= kSlots - 4 && who == 0) mastered.fetch_add(1, std::memory_order_relaxed);
            }
            else if (value != 0xfeedfacecafebeefull) corrupt = true;
        }
        if (corrupt) corrupted.fetch_add(1, std::memory_order_relaxed);
        if (localMissing) missing.fetch_add(1, std::memory_order_relaxed);
    }
    EXPECT_EQ(recordedTeam.load(), 3);
    EXPECT_EQ(corrupted.load(), 0);
    EXPECT_EQ(missing.load(), 0);
    EXPECT_EQ(mastered.load(), 0);
    EXPECT_EQ(skewFailed.load(), 0);
}

/** @test OMP_SINGLE executes exactly once in a retained active team. */
TEST(OpenMPWorkshare, RetainedSingleExecutesExactlyOnce)
{
    const TeamPolicy policy;
    std::atomic<int> singles{0};
    int outerTeam = 0;
    auto once = [&]() { singles.fetch_add(1, std::memory_order_relaxed); };
#pragma omp parallel num_threads(3)
    {
        if (omp_get_thread_num() == 0) outerTeam = omp_get_num_threads();
        OMP_SINGLE(once);
    }
    EXPECT_EQ(outerTeam, 3);
    EXPECT_EQ(singles.load(), 1);
}

/** @test OMP_SINGLE_NOWAIT keeps its no-forced-barrier semantics (bounded scheduling). */
TEST(OpenMPWorkshare, SingleNoWaitHasNoForcedBarrier)
{
    const TeamPolicy policy;
    std::atomic<int> returned{0};
    std::atomic<bool> failed{false};
    int outerTeam = 0;
#pragma omp parallel num_threads(3)
    {
        if (omp_get_thread_num() == 0) outerTeam = omp_get_num_threads();
        OMP_SINGLE_NOWAIT([&]() {
            // Bounded ORDERED scheduling handshake: wait until the two
            // non-single threads have returned from the nowait macro using
            // 50us sleep-based polling with a generous ~200ms hard cap. A
            // forced barrier after the single would keep them blocked here, so
            // the cap trips and the failure flag is set. This is a bounded
            // scheduling witness, not a timing-free proof.
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
            while (std::chrono::steady_clock::now() < deadline &&
                   returned.load(std::memory_order_acquire) < 2)
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            if (returned.load(std::memory_order_acquire) < 2)
                failed.store(true, std::memory_order_relaxed);
        });
        returned.fetch_add(1, std::memory_order_release);
    }
    EXPECT_EQ(outerTeam, 3);
    EXPECT_EQ(returned.load(), 3);
    EXPECT_FALSE(failed.load());
}

/** @test REGION_SYNC's explicit barrier is observable after skewed nowait work. */
TEST(OpenMPWorkshare, RegionSyncAddsExplicitBarrierAfterNowaitWork)
{
    const TeamPolicy policy;
    constexpr std::size_t kSlots = 12, kGuard = 16;
    std::array<std::atomic<std::uint64_t>, kGuard> shared;
    std::array<std::atomic<int>, kGuard> owner;
    for (std::size_t i = 0; i < kGuard; ++i)
    {
        shared[i].store(0xfeedfacecafebeefull, std::memory_order_relaxed);
        owner[i].store(-1, std::memory_order_relaxed);
    }
    std::atomic<int> recordedTeam{0}, missing{0}, corrupted{0}, mastered{0}, skewFailed{0};
    auto work = [&]() {
        const int team = omp_get_num_threads();
        recordedTeam.store(team, std::memory_order_relaxed);
        const int tid = omp_get_thread_num();
#pragma omp for schedule(static) nowait
        for (std::size_t i = 0; i < kSlots; ++i)
        {
            // Condition-based skew on the last chunk: wait until every other
            // member's slots are written (50us polling, ~200ms hard cap with a
            // fail flag), then a fixed 2ms before writing. Without the explicit
            // SYNC barrier a fast member could observe incomplete slots
            // (skew-assisted positive check; barrier-removal negative is the
            // tester's separate SYNC mutant, not the old-predicate control).
            if (i == kSlots - 4)
            {
                bool othersDone = false;
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
                do {
                    othersDone = true;
                    for (std::size_t j = 0; j < kSlots - 4; ++j)
                        if (owner[j].load(std::memory_order_acquire) < 0) { othersDone = false; break; }
                    if (!othersDone) std::this_thread::sleep_for(std::chrono::microseconds(50));
                } while (!othersDone && std::chrono::steady_clock::now() < deadline);
                if (!othersDone) skewFailed.fetch_add(1, std::memory_order_relaxed);
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            owner[i].store(tid, std::memory_order_release);
            shared[i].store(std::uint64_t(0xc0ffee0000000000ull | i), std::memory_order_release);
        }
    };
#pragma omp parallel num_threads(3)
    {
        OMP_WORKSHARE_REGION_SYNC(work);
        int localMissing = 0;
        bool corrupt = false;
        for (std::size_t i = 0; i < kGuard; ++i)
        {
            const auto value = shared[i].load(std::memory_order_acquire);
            const auto who = owner[i].load(std::memory_order_acquire);
            if (i < kSlots)
            {
                if (value != std::uint64_t(0xc0ffee0000000000ull | i) || who < 0) ++localMissing;
                if (i >= kSlots - 4 && who == 0) mastered.fetch_add(1, std::memory_order_relaxed);
            }
            else if (value != 0xfeedfacecafebeefull) corrupt = true;
        }
        if (corrupt) corrupted.fetch_add(1, std::memory_order_relaxed);
        if (localMissing) missing.fetch_add(1, std::memory_order_relaxed);
    }
    EXPECT_EQ(recordedTeam.load(), 3);
    EXPECT_EQ(corrupted.load(), 0);
    EXPECT_EQ(missing.load(), 0);
    EXPECT_EQ(mastered.load(), 0);
    EXPECT_EQ(skewFailed.load(), 0);
}

/** @test OMP_LAYER_PARALLEL forks standalone and reuses the retained team. */
TEST(OpenMPWorkshare, LayerParallelReusesRetainedTeam)
{
    const TeamPolicy policy;
    std::atomic<int> wrong{0};
    OMP_LAYER_PARALLEL({
        if (omp_get_level() != 1) wrong.fetch_add(1, std::memory_order_relaxed);
    });
    EXPECT_EQ(wrong.load(), 0);
#pragma omp parallel num_threads(3)
    {
        OMP_LAYER_PARALLEL({
            if (omp_get_num_threads() != 3 || omp_get_level() != 1)
                wrong.fetch_add(1, std::memory_order_relaxed);
        });
    }
    EXPECT_EQ(wrong.load(), 0);
}

/** @test The canonical placement authority classifies standalone, retained and active. */
TEST(OpenMPWorkshare, CanonicalPlacementAuthority)
{
    const TeamPolicy policy;
    EXPECT_TRUE(teamPlacement() == TeamPlacement::Standalone);
    EXPECT_FALSE(inLexicalWorkshare());
    EXPECT_FALSE(isInParallelRegion());
#pragma omp parallel num_threads(1)
    {
        // Absent an active ancestor, a one-thread region is inactive.
        EXPECT_TRUE(teamPlacement() == TeamPlacement::Retained);
        EXPECT_TRUE(inLexicalWorkshare());
        EXPECT_FALSE(inActiveParallelRegion());
        EXPECT_FALSE(isInParallelRegion());
    }
#pragma omp parallel if(false)
    {
        EXPECT_TRUE(teamPlacement() == TeamPlacement::Retained);
        EXPECT_TRUE(inLexicalWorkshare());
        EXPECT_FALSE(inActiveParallelRegion());
        EXPECT_FALSE(isInParallelRegion());
    }
#pragma omp parallel num_threads(3)
    {
        EXPECT_TRUE(teamPlacement() == TeamPlacement::Active);
        EXPECT_TRUE(inLexicalWorkshare());
        EXPECT_TRUE(inActiveParallelRegion());
        EXPECT_TRUE(isInParallelRegion());
    }
    EXPECT_FALSE(isInParallelRegion());
    EXPECT_FALSE(inLexicalWorkshare());
}
/**
 * @brief The exact fork/join contract of the one team-fork authority.
 *
 * Every macro family must observe exactly its own real fork: standalone forks
 * count once, retained reuse counts nothing, and the if(false) serial branch
 * still counts because it creates (and costs) a real one-thread team. The
 * counter is process-wide and monotonic across test shards, so every assertion
 * uses deltas between snapshots, never absolute values.
 */
TEST(OpenMPWorkshare, TeamForkCounterCountsExactlyTheRealForks)
{
    const TeamPolicy policy;
    const NumThreadsGuard threads(4);

    // Standalone workshare: one fork per invocation.
    std::uint64_t before = openMPTeamForkSnapshot();
    auto one_work = [] {};
    OMP_WORKSHARE_REGION(one_work);
    OMP_WORKSHARE_REGION(one_work);
    EXPECT_EQ(openMPTeamForkSnapshot() - before, 2u);

    // Standalone sync workshare: the explicit barrier adds no fork.
    before = openMPTeamForkSnapshot();
    OMP_WORKSHARE_REGION_SYNC(one_work);
    EXPECT_EQ(openMPTeamForkSnapshot() - before, 1u);

    // Standalone retained-team forms still fork for real participants.
    before = openMPTeamForkSnapshot();
    OMP_WORKSHARE_REGION_IF(one_work, true);
    EXPECT_EQ(openMPTeamForkSnapshot() - before, 1u);
    // The false branch creates a real one-thread team: it forks and costs.
    OMP_WORKSHARE_REGION_IF(one_work, false);
    EXPECT_EQ(openMPTeamForkSnapshot() - before, 2u);

    before = openMPTeamForkSnapshot();
    OMP_WORKSHARE_COLLAPSE2_IF(one_work, true);
    OMP_WORKSHARE_COLLAPSE2_IF(one_work, false);
    OMP_WORKSHARE_COLLAPSE3_IF(one_work, true);
    OMP_WORKSHARE_COLLAPSE3_IF(one_work, false);
    EXPECT_EQ(openMPTeamForkSnapshot() - before, 4u);

    before = openMPTeamForkSnapshot();
    OMP_LAYER_PARALLEL({});
    EXPECT_EQ(openMPTeamForkSnapshot() - before, 1u);

    // Retained reuse re-executes the exact team without any fork at all.
    // The baseline is captured outside the region and stays immutable: the
    // team members only read it, so no shared cell is ever written from
    // inside the parallel region.
    before = openMPTeamForkSnapshot();
#pragma omp parallel num_threads(2)
    {
        OMP_WORKSHARE_REGION(one_work);
        OMP_WORKSHARE_REGION_SYNC(one_work);
        OMP_WORKSHARE_REGION_IF(one_work, false);
        OMP_WORKSHARE_COLLAPSE2_IF(one_work, true);
        OMP_LAYER_PARALLEL({});
        EXPECT_EQ(openMPTeamForkSnapshot() - before, 0u);
    }
}

/**
 * @brief Owner-scoped attribution stays exact under overlap and nesting.
 *
 * Two threads hold concurrent scopes and fork through the production macro
 * family at the same time: each scope's cell must observe exactly its own
 * forks (concurrent executors count each other only in the process total),
 * and the process total must stay complete. A nested scope observes the same
 * fork as its enclosing scope (both are current owners), and the enclosing
 * scope keeps counting after the inner one ends. No production serialization
 * is involved: the overlap is the contract.
 */
TEST(OpenMPWorkshare, TeamForkScopeAttributesExactlyUnderOverlapAndNesting)
{
    const TeamPolicy policy;
    const NumThreadsGuard threads(4);
    constexpr std::uint64_t kForksPerThread = 8;

    auto one_work = [] {};
    std::atomic<std::uint64_t> scope_a{0}, scope_b{0};
    const std::uint64_t total_before = openMPTeamForkSnapshot();
    std::barrier<> rendezvous(2);
    auto owner = [&](std::atomic<std::uint64_t> &observed)
    {
        OpenMPTeamForkScope scope;
        rendezvous.arrive_and_wait(); // Both owners are current at once.
        for (std::uint64_t fork = 0; fork < kForksPerThread; ++fork)
            OMP_WORKSHARE_REGION(one_work);
        observed.store(scope.forks(), std::memory_order_relaxed);
    };
    std::thread a([&] { owner(scope_a); });
    std::thread b([&] { owner(scope_b); });
    a.join();
    b.join();
    EXPECT_EQ(scope_a.load(), kForksPerThread);
    EXPECT_EQ(scope_b.load(), kForksPerThread);
    EXPECT_EQ(openMPTeamForkSnapshot() - total_before, 2 * kForksPerThread);

    // Nesting: the inner scope and the enclosing scope both observe the same
    // fork; the enclosing scope keeps observing after the inner one ends.
    const std::uint64_t nesting_before = openMPTeamForkSnapshot();
    OpenMPTeamForkScope outer;
    OMP_WORKSHARE_REGION(one_work);
    std::uint64_t inner_seen = 0;
    {
        OpenMPTeamForkScope inner;
        OMP_WORKSHARE_REGION(one_work);
        OMP_WORKSHARE_REGION(one_work);
        inner_seen = inner.forks();
    }
    EXPECT_EQ(inner_seen, 2u);
    EXPECT_EQ(outer.forks(), 3u);
    OMP_WORKSHARE_REGION(one_work);
    EXPECT_EQ(outer.forks(), 4u);
    EXPECT_EQ(openMPTeamForkSnapshot() - nesting_before, 4u);
}

/**
 * @brief The ownership boundary for team workers is explicit, not assumed.
 *
 * Thread-local owner stacks do not propagate into a forked team: a worker
 * that executes a fork contributes to the process total only and must never
 * be attributed to an owner cell. The decode discipline "workers never fork"
 * is enforced by the retained-team authority, and this regression pins the
 * accounting consequence: worker-context forks move the total and leave
 * every owner cell untouched.
 */
TEST(OpenMPWorkshare, TeamForkWorkerContextForksNeverReachOwnerCells)
{
    const TeamPolicy policy;
    const NumThreadsGuard threads(2);
    auto idle = [] {};

    OpenMPTeamForkScope outer;
    OMP_WORKSHARE_REGION(idle); // One owner-thread fork for a known baseline.
    const std::uint64_t total_before = openMPTeamForkSnapshot();
    const std::uint64_t outer_before = outer.forks();
#pragma omp parallel num_threads(2)
    {
        // Only a true worker (non-zero team member) forks here: the team's
        // thread zero IS the owner thread, and a fork it executes legitimately
        // attributes to the owner. The worker's fork must not.
        if (omp_get_thread_num() != 0)
        {
            ::llaminar::v2::noteOpenMPTeamFork();
#pragma omp parallel
            {
            }
            EXPECT_EQ(outer.forks(), outer_before)
                << "no worker fork may reach an owner cell";
        }
    }
    EXPECT_EQ(openMPTeamForkSnapshot() - total_before, 1u)
        << "the worker fork moves the total exactly once";
    EXPECT_EQ(outer.forks(), outer_before)
        << "no worker fork may reach an owner cell";
}

/**
 * @brief Reverse teardown restores the owner stack exactly.
 *
 * The cost harness tears nested scopes down inner-first on purpose; this
 * regression proves the accounting survives it: after the enforced reverse
 * teardown, a fresh owner observes only its own forks - no freed cell is
 * still on the stack, and no subsequent note can dereference torn-down
 * storage.
 */
TEST(OpenMPWorkshare, TeamForkScopeStackRestoredAfterReverseTeardown)
{
    auto idle = [] {};
    const std::uint64_t before = openMPTeamForkSnapshot();
    {
        OpenMPTeamForkScope outer;
        {
            OpenMPTeamForkScope inner;
            OMP_WORKSHARE_REGION(idle);
            EXPECT_EQ(inner.forks(), 1u);
        } // Inner dies first: the enforced reverse teardown.
        OMP_WORKSHARE_REGION(idle);
        EXPECT_EQ(outer.forks(), 2u);
    } // Outer dies after the inner, completing the LIFO order.
    // A fresh owner after the teardown must observe exactly its own forks.
    OpenMPTeamForkScope fresh;
    OMP_WORKSHARE_REGION(idle);
    EXPECT_EQ(fresh.forks(), 1u);
    EXPECT_EQ(openMPTeamForkSnapshot() - before, 3u);
}

/**
 * @brief The counter's own admission and publication contract.
 *
 * Default serving materializes no "omp" record at all (the off-default
 * asserted only against a bound that is generous against a loaded shared
 * runner (the fork it describes costs microseconds).
 *
 * The collector-visible publication contracts (off-default shape, family
 * selection, cost recorded as a PerfStats record) live in the bootstrap
 * fixture, which is the witness that links the collector.
 */
namespace
{
    /**
     * @brief Amortized ns per note under one ownership configuration.
     *
     * Every measuring thread constructs its OWN same-thread owner scope
     * inside the thread, the timed window excludes thread creation and
     * teardown, and each run verifies the exact cell and process deltas it
     * must produce. The returned figure is per-owner note latency: the
     * concurrent variant divides the concurrent wall time by the notes ONE
     * thread executed, not by the sum of both threads' notes (which would be
     * aggregate throughput, not latency).
     */
    struct NoteCostRun
    {
        double ns_per_note = 0.0; ///< Amortized per-owner note latency.
        std::uint64_t process_delta = 0; ///< Exact process-total advance observed.
    };

    NoteCostRun measureNoteCost(std::uint64_t notes, int owners, bool concurrent_threads)
    {
        // Off the clock: the exact scope configuration. owners==0 constructs
        // NO scope at all (the no-owner figure must not secretly carry one); 1
        // constructs one; 2 constructs two lexically nested scopes whose
        // teardown is inner-first and asserted as restored before returning.
        const std::uint64_t before = openMPTeamForkSnapshot();
        std::optional<OpenMPTeamForkScope> outer;
        std::optional<OpenMPTeamForkScope> inner;
        if (!concurrent_threads && owners >= 1)
            outer.emplace();
        if (!concurrent_threads && owners >= 2)
            inner.emplace();

        // Off the clock (concurrent): BOTH measuring threads construct their
        // own same-thread owner scopes, and both sides have signalled
        // readiness BEFORE the clock starts - so contention inside the timed
        // window is established, not hoped for, and both owners are real.
        std::atomic<bool> worker_ready{false};
        std::atomic<bool> start_gate{false};
        std::atomic<bool> worker_notes_done{false};
        std::atomic<bool> verify_gate{false};
        std::thread worker;
        std::optional<OpenMPTeamForkScope> main_owner;
        if (concurrent_threads)
        {
            main_owner.emplace(); // The measuring thread owns a cell too.
            worker = std::thread([&]
            {
                OpenMPTeamForkScope scope; // Same-thread owner, built off the clock.
                worker_ready.store(true, std::memory_order_release);
                while (!start_gate.load(std::memory_order_acquire))
                    std::this_thread::yield();
                for (std::uint64_t note = 0; note < notes; ++note)
                    noteOpenMPTeamFork();
                worker_notes_done.store(true, std::memory_order_release);
                // Verification and teardown happen only after the main thread
                // stopped the clock and opened the verify gate: the timed
                // window contains the notes and nothing else.
                while (!verify_gate.load(std::memory_order_acquire))
                    std::this_thread::yield();
                EXPECT_EQ(scope.forks(), notes);
            });
            while (!worker_ready.load(std::memory_order_acquire))
                std::this_thread::yield();
        }

        // The timed window: notes only. Start after both owners (when any) are
        // live and ready; stop before any verification, join or teardown.
        const auto start = std::chrono::steady_clock::now();
        if (concurrent_threads)
            start_gate.store(true, std::memory_order_release);
        for (std::uint64_t note = 0; note < notes; ++note)
            noteOpenMPTeamFork();
        if (concurrent_threads)
            while (!worker_notes_done.load(std::memory_order_acquire))
                std::this_thread::yield();
        const auto stop = std::chrono::steady_clock::now();
        if (concurrent_threads)
            verify_gate.store(true, std::memory_order_release);

        // Off the clock: verification, join and teardown.
        const auto elapsed_ns = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start)
                .count());
        NoteCostRun result;
        result.ns_per_note =
            elapsed_ns / static_cast<double>(notes);
        if (concurrent_threads)
        {
            EXPECT_EQ(main_owner->forks(), notes);
            worker.join(); // The worker verified and destroyed its own scope.
            result.process_delta = 2 * notes;
        }
        else
        {
            EXPECT_EQ(outer ? outer->forks() : notes, notes) << "owners " << owners;
            if (inner)
                EXPECT_EQ(inner->forks(), notes) << "owners " << owners;
            // The total is checked BEFORE the teardown probe so the probe's
            // own fork stays out of the measured figure.
            EXPECT_EQ(openMPTeamForkSnapshot() - before, notes)
                << "nested owners " << owners;
            if (outer)
            {
                // Enforced LIFO teardown: the inner scope dies FIRST, and the
                // restored stack is asserted before the outer follows - the
                // outer must never die above a live inner.
                inner.reset();
                {
                    OpenMPTeamForkScope restored;
                    auto local_idle = [] {};
                    OMP_WORKSHARE_REGION(local_idle);
                    EXPECT_EQ(restored.forks(), 1u);
                    // STACK IDENTITY, not just a self-count: the probe fork
                    // must attribute to BOTH the fresh scope and the
                    // surviving outer (the walked stack is exactly
                    // [restored, outer]). A lost or wrong previous link would
                    // leave the outer at `notes` and fail here.
                    EXPECT_TRUE(outer.has_value());
                    EXPECT_EQ(outer->forks(), notes + 1)
                        << "the probe must walk restored AND the surviving outer";
                }
                outer.reset();
                result.process_delta = notes + 1; // Notes plus the probe.
            }
            else
            {
                // owners==0: nothing was alive, so the stack is already the
                // restored original and there is nothing to tear down.
                inner.reset();
                result.process_delta = notes;
            }
            // The ORIGINAL stack is restored after the teardown: a fork now
            // must attribute to nothing (the total is its only witness).
            const std::uint64_t after_teardown = openMPTeamForkSnapshot();
            {
                auto local_idle = [] {};
                OMP_WORKSHARE_REGION(local_idle);
            }
            EXPECT_EQ(openMPTeamForkSnapshot() - after_teardown, 1u);
            result.process_delta += 1; // The post-teardown fork.
        }
        // Every configuration counts each fork exactly once in the total
        // (the nested branch's figure includes its restoration probe and the
        // post-teardown fork that proves the original stack is restored).
        EXPECT_EQ(openMPTeamForkSnapshot() - before, result.process_delta)
            << (concurrent_threads ? "concurrent" : "nested") << " owners, " << owners;
        return result;
    }
} // namespace

TEST(OpenMPWorkshare, TeamForkCounterPublicationCostAndDefault)
{
    // The note's own cost is MEASURED under the ownership configurations
    // production actually runs - no owner (utility threads), the one owner a
    // decode window or stage holds, the two nested owners (window + stage),
    // and two concurrent owners - because production's note does a stack walk
    // over global+owner atomics and a bare ns figure would understate it.
    // Each variant is bounded generously (the fork it describes costs
    // microseconds); the collector-visible evidence cell in the bootstrap
    // fixture records the same variants.
    constexpr std::uint64_t kMeasuredNotes = 1u << 20;
    const double no_owner = measureNoteCost(kMeasuredNotes, 0, false).ns_per_note;
    const double one_owner = measureNoteCost(kMeasuredNotes, 1, false).ns_per_note;
    const double two_owners = measureNoteCost(kMeasuredNotes, 2, false).ns_per_note;
    const double concurrent = measureNoteCost(kMeasuredNotes, 1, true).ns_per_note;
    for (const auto &[label, cost] : std::initializer_list<std::pair<const char *, double>>{
             {"no_owner", no_owner}, {"one_owner", one_owner},
             {"two_nested_owners", two_owners}, {"concurrent_two_threads", concurrent}})
    {
        EXPECT_LT(cost, 100.0) << label;
        std::cout << "[evidence] openmp_team_fork_note_self_cost_" << label << "="
                  << cost << " ns per owner note, amortized concurrent wall time "
                            "including rendezvous over " << kMeasuredNotes
                  << " notes per measuring thread (per-owner latency, not aggregate "
                     "throughput; figures measured with the production owner walk; "
                     "readiness permits concurrency and does not prove scheduler overlap)"
                  << std::endl;
    }
    EXPECT_GE(two_owners, no_owner * 0.5)
        << "the two-owner walk must not be cheaper than a fabrication would allow";
}

} // namespace llaminar2::test
