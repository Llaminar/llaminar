/**
 * @file Test__RegisterGuard.cpp
 * @brief Unit tests for RegisterGuard conflict detection
 */

#include <gtest/gtest.h>
#include "kernels/cpu/jit/RegisterGuard.h"

using namespace llaminar2::jit;

class Test__RegisterGuard : public ::testing::Test
{
protected:
    RegisterTracker tracker;
};

TEST_F(Test__RegisterGuard, BorrowAndRelease)
{
    EXPECT_FALSE(tracker.is_borrowed<Score0>());
    EXPECT_EQ(tracker.borrowed_count(), 0);

    {
        auto guard = tracker.borrow<Score0>();
        EXPECT_TRUE(tracker.is_borrowed<Score0>());
        EXPECT_EQ(tracker.borrowed_count(), 1);

        // Can access the register - Score0 is XMM-based
        EXPECT_EQ(guard.xmm().getIdx(), 20);
        // Note: guard.zmm() would fail to compile for Score0 (TypedXmm)
    }

    // Released when guard goes out of scope
    EXPECT_FALSE(tracker.is_borrowed<Score0>());
    EXPECT_EQ(tracker.borrowed_count(), 0);
}

TEST_F(Test__RegisterGuard, MultipleBorrowsNonConflicting)
{
    auto guard1 = tracker.borrow<Score0>();   // reg 20
    auto guard2 = tracker.borrow<Score1>();   // reg 21
    auto guard3 = tracker.borrow<Scratch4>(); // reg 24 (safe, doesn't conflict)
    auto guard4 = tracker.borrow<Accum0>();   // reg 0

    EXPECT_EQ(tracker.borrowed_count(), 4);
    EXPECT_TRUE(tracker.is_borrowed(20));
    EXPECT_TRUE(tracker.is_borrowed(21));
    EXPECT_TRUE(tracker.is_borrowed(24));
    EXPECT_TRUE(tracker.is_borrowed(0));
}

TEST_F(Test__RegisterGuard, MoveSemantics)
{
    auto guard1 = tracker.borrow<Score0>();
    EXPECT_TRUE(tracker.is_borrowed<Score0>());

    // Move to new guard
    auto guard2 = std::move(guard1);
    EXPECT_TRUE(tracker.is_borrowed<Score0>()); // Still borrowed
    EXPECT_EQ(tracker.borrowed_count(), 1);     // Still just one

    // Original guard no longer owns it
    // (guard1 is in moved-from state, destructor won't double-release)
}

TEST_F(Test__RegisterGuard, ExplicitRelease)
{
    auto guard = tracker.borrow<Score0>();
    EXPECT_TRUE(tracker.is_borrowed<Score0>());

    guard.release();
    EXPECT_FALSE(tracker.is_borrowed<Score0>());

    // Destructor won't try to release again (would assert if it did)
}

TEST_F(Test__RegisterGuard, ConflictDetection_SameRegTwice)
{
    auto guard1 = tracker.borrow<Score0>();

    // This should FAIL at runtime (assertion)
    // Uncomment to verify conflict detection works:
    // auto guard2 = tracker.borrow<Score0>();  // ASSERT: already borrowed!
}

// Death tests to verify conflict detection actually works
TEST_F(Test__RegisterGuard, DeathTest_SameRegTwice)
{
    auto guard1 = tracker.borrow<Score0>();

    // Assert fires when trying to borrow the same register twice
    EXPECT_DEATH(
        { auto guard2 = tracker.borrow<Score0>(); },
        ".*" // Any death message - the assert fires
    );
}

TEST_F(Test__RegisterGuard, DeathTest_XmmZmmAlias)
{
    // Score0 uses xmm20, Scratch0 uses zmm20 - SAME PHYSICAL REGISTER
    auto guard_score = tracker.borrow<Score0>();

    // Assert fires when trying to borrow aliased register
    EXPECT_DEATH(
        { auto guard_scratch = tracker.borrow<Scratch0>(); },
        ".*" // Any death message - the assert fires
    );
}

TEST_F(Test__RegisterGuard, SafeScratchForFA2)
{
    // While using Score0-3 (xmm20-23), we can still use Scratch4-5 (zmm24-25)
    auto score0 = tracker.borrow<Score0>();
    auto score1 = tracker.borrow<Score1>();
    auto score2 = tracker.borrow<Score2>();
    auto score3 = tracker.borrow<Score3>();

    // These should NOT conflict (different physical registers)
    auto scratch4 = tracker.borrow<Scratch4>(); // zmm24
    auto scratch5 = tracker.borrow<Scratch5>(); // zmm25

    EXPECT_EQ(tracker.borrowed_count(), 6);
}

TEST_F(Test__RegisterGuard, DebugString)
{
    auto s0 = tracker.borrow<Score0>();
    auto s1 = tracker.borrow<Scratch4>();

    std::string debug = tracker.debug_string();
    EXPECT_NE(debug.find("r20"), std::string::npos);
    EXPECT_NE(debug.find("r24"), std::string::npos);
    EXPECT_NE(debug.find("Score"), std::string::npos);
    EXPECT_NE(debug.find("Scratch"), std::string::npos);
}

TEST_F(Test__RegisterGuard, Reset)
{
    {
        auto guard = tracker.borrow<Score0>();
        EXPECT_EQ(tracker.borrowed_count(), 1);
        guard.release(); // Explicitly release before reset
    }

    tracker.reset();
    EXPECT_EQ(tracker.borrowed_count(), 0);

    // Can borrow again after reset
    auto guard2 = tracker.borrow<Score0>();
    EXPECT_EQ(tracker.borrowed_count(), 1);
}

// Compile-time static assertions
TEST_F(Test__RegisterGuard, StaticConflictDetection)
{
    // These are compile-time checks from RegisterGuard.h
    static_assert(registers_conflict_v<Score0, Scratch0>,
                  "Score0 and Scratch0 must conflict");
    static_assert(registers_conflict_v<Score1, Scratch1>,
                  "Score1 and Scratch1 must conflict");
    static_assert(!registers_conflict_v<Scratch4, Score0>,
                  "Scratch4 should not conflict with Score0");
    static_assert(!registers_conflict_v<Accum0, Score0>,
                  "Accum0 should not conflict with Score0");
}

// Test the ScopedRegisterSet helper
TEST_F(Test__RegisterGuard, ScopedRegisterSet)
{
    {
        ScopedRegisterSet<Score0, Score1, Scratch4> regs(tracker);

        EXPECT_EQ(tracker.borrowed_count(), 3);
        EXPECT_TRUE(tracker.is_borrowed<Score0>());
        EXPECT_TRUE(tracker.is_borrowed<Score1>());
        EXPECT_TRUE(tracker.is_borrowed<Scratch4>());

        // Access individual registers
        EXPECT_EQ(regs.get<0>().xmm().getIdx(), 20);
        EXPECT_EQ(regs.get<1>().xmm().getIdx(), 21);
        EXPECT_EQ(regs.get<2>().zmm().getIdx(), 24);
    }

    // All released
    EXPECT_EQ(tracker.borrowed_count(), 0);
}

// ============================================================================
// Tests for assert_available() - detecting raw accessor conflicts
// ============================================================================

TEST_F(Test__RegisterGuard, AssertAvailable_NotBorrowed)
{
    // When nothing is borrowed, assert_available should pass
    tracker.assert_available<Score0>("test context");
    tracker.assert_available<Accum4>("test context");
    tracker.assert_available<Scratch0>("test context");

    // Also test the physical index version
    tracker.assert_available(20, "Score0", "test context");
    tracker.assert_available(4, "Accum4", "test context");
}

TEST_F(Test__RegisterGuard, AssertAvailable_DifferentRegister)
{
    // Borrow Score0, but check Accum4 - should pass
    auto guard = tracker.borrow<Score0>();
    tracker.assert_available<Accum4>("test context"); // Different register, OK

    // Also verify Score0 IS borrowed
    EXPECT_TRUE(tracker.is_borrowed<Score0>());
}

TEST_F(Test__RegisterGuard, AssertAvailable_AfterRelease)
{
    {
        auto guard = tracker.borrow<Accum4>();
        EXPECT_TRUE(tracker.is_borrowed<Accum4>());
    }
    // After guard destruction, should be available again
    tracker.assert_available<Accum4>("after release");
}

// Death test: assert_available when register IS borrowed
TEST_F(Test__RegisterGuard, DeathTest_AssertAvailable_WhenBorrowed)
{
    auto guard = tracker.borrow<Accum4>();

    // This should trigger an assertion failure with helpful message
    EXPECT_DEATH(
        tracker.assert_available<Accum4>("trying to use raw accessor"),
        "RAW ACCESSOR CONFLICT DETECTED");
}

// Death test: assert_available detects XMM/ZMM aliasing
TEST_F(Test__RegisterGuard, DeathTest_AssertAvailable_AliasConflict)
{
    // Borrow Score0 (xmm20), then try to assert_available on Scratch0 (zmm20)
    // These alias the same physical register!
    auto guard = tracker.borrow<Score0>();

    EXPECT_DEATH(
        tracker.assert_available<Scratch0>("trying zmm20 while xmm20 borrowed"),
        "RAW ACCESSOR CONFLICT DETECTED");
}

// Test that demonstrates the original bug pattern
// This is the exact pattern that caused the FA2 parity regression:
// 1. Borrow Accum4/5 for V loading
// 2. Try to use Accum4/5 raw accessors for spilled context
TEST_F(Test__RegisterGuard, DeathTest_OriginalBugPattern)
{
    // Simulate the bug: borrow accum4-5 for V loading
    auto guard_v_lo = tracker.borrow<Accum4>();
    auto guard_v_hi = tracker.borrow<Accum5>();

    // Now try the pattern that was buggy: using accum4/5 raw accessors
    // This WOULD have been detected if we had assert_available checks!
    EXPECT_DEATH(
        tracker.assert_available<Accum4>("spilled context lo"),
        "RAW ACCESSOR CONFLICT DETECTED");

    EXPECT_DEATH(
        tracker.assert_available<Accum5>("spilled context hi"),
        "RAW ACCESSOR CONFLICT DETECTED");

    // But accum6-7 (the fix) should work fine
    tracker.assert_available<Accum6>("spilled context lo (fixed)");
    tracker.assert_available<Accum7>("spilled context hi (fixed)");
}
