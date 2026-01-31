/**
 * @file Test__RegisterAllocation.cpp
 * @brief Unit tests for compile-time register allocation framework
 */

#include <gtest/gtest.h>
#include "kernels/cpu/jit/RegisterAllocation.h"
#include "kernels/cpu/jit/RegisterGuard.h"

using namespace llaminar2::jit;

// ============================================================================
// Zone Definition Tests
// ============================================================================

TEST(Test__RegisterAllocation, ZoneRangesAreCorrect)
{
    // Accumulator zone: zmm0-zmm7
    EXPECT_EQ(AccumulatorZone::base_index, 0);
    EXPECT_EQ(AccumulatorZone::count, 8);

    // Q vector zone: zmm8-zmm15
    EXPECT_EQ(QVectorZone::base_index, 8);
    EXPECT_EQ(QVectorZone::count, 8);

    // State zone: zmm16-zmm19
    EXPECT_EQ(StateZone::base_index, 16);
    EXPECT_EQ(StateZone::count, 4);

    // Scratch zone: zmm20-zmm25
    EXPECT_EQ(ScratchZone::base_index, 20);
    EXPECT_EQ(ScratchZone::count, 6);

    // Score zone: xmm20-xmm23 (overlaps scratch)
    EXPECT_EQ(ScoreZone::base_index, 20);
    EXPECT_EQ(ScoreZone::count, 4);
}

TEST(Test__RegisterAllocation, ZoneOverlapDetection)
{
    // These should NOT overlap
    EXPECT_FALSE((zones_overlap_v<AccumulatorZone, QVectorZone>));
    EXPECT_FALSE((zones_overlap_v<QVectorZone, StateZone>));
    EXPECT_FALSE((zones_overlap_v<StateZone, ScratchZone>));
    EXPECT_FALSE((zones_overlap_v<AccumulatorZone, StateZone>));
    EXPECT_FALSE((zones_overlap_v<AccumulatorZone, ScratchZone>));

    // ScoreZone and ScratchZone DO overlap (by design - XMM/ZMM aliasing)
    EXPECT_TRUE((zones_overlap_v<ScoreZone, ScratchZone>));
}

// ============================================================================
// Typed Register Tests
// ============================================================================

TEST(Test__RegisterAllocation, TypedZmmAbsoluteIndices)
{
    // Accumulators: zmm0-zmm7
    EXPECT_EQ(Accum0::absolute_index, 0);
    EXPECT_EQ(Accum7::absolute_index, 7);

    // State registers: zmm16-zmm19
    EXPECT_EQ(StateMax::absolute_index, 16);
    EXPECT_EQ(StateSum::absolute_index, 17);
    EXPECT_EQ(StateWeight::absolute_index, 18);
    EXPECT_EQ(StateCorr::absolute_index, 19);

    // Scratch registers: zmm20-zmm25
    EXPECT_EQ(Scratch0::absolute_index, 20);
    EXPECT_EQ(Scratch4::absolute_index, 24);
    EXPECT_EQ(Scratch5::absolute_index, 25);

    // Score registers (XMM): xmm20-xmm23
    EXPECT_EQ(Score0::absolute_index, 20);
    EXPECT_EQ(Score3::absolute_index, 23);
}

TEST(Test__RegisterAllocation, TypedZmmConvertsToXbyak)
{
    // TypedZmm are now pure tag types - verify we can construct Xbyak registers
    // from their absolute_index (the new pattern)
    Xbyak::Zmm zmm0 = Xbyak::Zmm(Accum0::absolute_index);
    EXPECT_EQ(zmm0.getIdx(), 0);

    Xbyak::Zmm zmm16 = Xbyak::Zmm(StateMax::absolute_index);
    EXPECT_EQ(zmm16.getIdx(), 16);

    Xbyak::Zmm zmm24 = Xbyak::Zmm(Scratch4::absolute_index);
    EXPECT_EQ(zmm24.getIdx(), 24);

    Xbyak::Zmm zmm25 = Xbyak::Zmm(Scratch5::absolute_index);
    EXPECT_EQ(zmm25.getIdx(), 25);
}

TEST(Test__RegisterAllocation, TypedXmmConvertsToXbyak)
{
    // TypedXmm are now pure tag types - verify we can construct Xbyak registers
    Xbyak::Xmm xmm20 = Xbyak::Xmm(Score0::absolute_index);
    EXPECT_EQ(xmm20.getIdx(), 20);

    Xbyak::Xmm xmm23 = Xbyak::Xmm(Score3::absolute_index);
    EXPECT_EQ(xmm23.getIdx(), 23);
}

TEST(Test__RegisterAllocation, XmmZmmAliasing)
{
    // Score0 (xmm20) aliases Scratch0 (zmm20)
    EXPECT_EQ(Score0::absolute_index, Scratch0::absolute_index);

    // Can get ZMM from XMM absolute_index (shows the aliasing)
    Xbyak::Zmm aliased_zmm = Xbyak::Zmm(Score0::absolute_index);
    EXPECT_EQ(aliased_zmm.getIdx(), 20);
}

// ============================================================================
// Zone Type Trait Tests (C++17 compatible)
// ============================================================================

TEST(Test__RegisterAllocation, ZoneTypeTraitsWork)
{
    // Scratch registers should pass the scratch zone check
    EXPECT_TRUE((is_zone_v<Scratch0, ScratchZone>));
    EXPECT_TRUE((is_zone_v<Scratch4, ScratchZone>));

    // Non-scratch registers should fail
    EXPECT_FALSE((is_zone_v<Accum0, ScratchZone>));
    EXPECT_FALSE((is_zone_v<StateMax, ScratchZone>));

    // is_not_zone_v should be the inverse
    EXPECT_FALSE((is_not_zone_v<Scratch0, ScratchZone>));
    EXPECT_TRUE((is_not_zone_v<Accum0, ScratchZone>));

    // is_any_zone_v should work with multiple zones
    EXPECT_TRUE((is_any_zone_v<Scratch0, AccumulatorZone, ScratchZone>));
    EXPECT_TRUE((is_any_zone_v<Accum0, AccumulatorZone, ScratchZone>));
    EXPECT_FALSE((is_any_zone_v<StateMax, AccumulatorZone, ScratchZone>));
}

TEST(Test__RegisterAllocation, SafeScratchForFA2)
{
    // Scratch4 and Scratch5 are safe during FA2 (don't alias score registers)
    // local_index >= 4
    EXPECT_GE(Scratch4::local_index, 4);
    EXPECT_GE(Scratch5::local_index, 4);

    // Scratch0-3 are NOT safe (alias Score0-3)
    EXPECT_LT(Scratch0::local_index, 4);
    EXPECT_LT(Scratch1::local_index, 4);
    EXPECT_LT(Scratch2::local_index, 4);
    EXPECT_LT(Scratch3::local_index, 4);
}

// ============================================================================
// Register Set Conflict Detection
// ============================================================================

TEST(Test__RegisterAllocation, RegisterSetNoConflicts)
{
    // These should compile without conflict
    using SafeSet1 = UsesRegisters<Accum0, Accum1, StateMax>;
    EXPECT_EQ(SafeSet1::count, 3u);

    using SafeSet2 = UsesRegisters<Scratch4, Scratch5, StateCorr>;
    EXPECT_EQ(SafeSet2::count, 3u);
}

// ============================================================================
// Compile-Time Safety Demonstrations
// ============================================================================

// This test verifies that conflicting registers fail at compile time
// Uncomment to see the static_assert fire:
// TEST(Test__RegisterAllocation, RegisterSetDetectsConflicts) {
//     // This should NOT compile - Score0 and Scratch0 have same absolute index!
//     using ConflictSet = UsesRegisters<Score0, Scratch0>;
// }

// To verify the SFINAE constraint works, uncomment this - it should fail:
// template<typename Reg, require_safe_scratch_for_fa2<Reg> = true>
// void unsafe_test_fn(const Reg&) {}
// TEST(Test__RegisterAllocation, SFINAEBlocksUnsafeScratch) {
//     // This should NOT compile - Scratch0 aliases Score0 (local_index < 4)
//     unsafe_test_fn(Scratch0{});
// }

// ============================================================================
// Real-World Usage Patterns
// ============================================================================

TEST(Test__RegisterAllocation, FA2TileRegistersDocumented)
{
    // Document the FA2 tile register usage to catch future conflicts

    // During FA2 tile processing:
    // - Score0-Score3 (xmm20-23) hold the 4 dot product scores
    // - Scratch4 (zmm24) can be used for tile_max
    // - Scratch5 (zmm25) can be used for other scratch

    // This is the fix we made: tile_max uses Scratch4 (zmm24)
    // NOT Scratch0 (zmm20) which would clobber Score0

    EXPECT_NE(Scratch4::absolute_index, Score0::absolute_index);
    EXPECT_NE(Scratch4::absolute_index, Score1::absolute_index);
    EXPECT_NE(Scratch4::absolute_index, Score2::absolute_index);
    EXPECT_NE(Scratch4::absolute_index, Score3::absolute_index);

    // Scratch0-3 DO conflict with scores
    EXPECT_EQ(Scratch0::absolute_index, Score0::absolute_index);
    EXPECT_EQ(Scratch1::absolute_index, Score1::absolute_index);
    EXPECT_EQ(Scratch2::absolute_index, Score2::absolute_index);
    EXPECT_EQ(Scratch3::absolute_index, Score3::absolute_index);
}

// ============================================================================
// SFINAE-based constrained function example (C++17)
// ============================================================================

namespace example
{

    // Function that only accepts scratch registers using SFINAE
    template <typename Reg, require_zone<Reg, ScratchZone> = true>
    constexpr bool accepts_scratch(const Reg &)
    {
        return true;
    }

    // Function that only accepts safe scratch for FA2 (Scratch4, Scratch5)
    template <typename Reg, require_safe_scratch_for_fa2<Reg> = true>
    constexpr bool accepts_safe_fa2_scratch(const Reg &)
    {
        return true;
    }

} // namespace example

TEST(Test__RegisterAllocation, SFINAEConstrainedFunctions)
{
    // accepts_scratch works for any scratch register
    EXPECT_TRUE(example::accepts_scratch(Scratch0{}));
    EXPECT_TRUE(example::accepts_scratch(Scratch4{}));
    EXPECT_TRUE(example::accepts_scratch(Scratch5{}));

    // accepts_safe_fa2_scratch only works for Scratch4, Scratch5
    // (These would fail to compile for Scratch0-3)
    EXPECT_TRUE(example::accepts_safe_fa2_scratch(Scratch4{}));
    EXPECT_TRUE(example::accepts_safe_fa2_scratch(Scratch5{}));

    // These lines would NOT compile (uncomment to verify):
    // example::accepts_scratch(Accum0{});  // Not a scratch register
    // example::accepts_safe_fa2_scratch(Scratch0{});  // local_index < 4
}

// ============================================================================
// Encoding Constraint Tests
// ============================================================================

TEST(Test__RegisterAllocation, EncodingConstraints)
{
    // LOW registers (0-15) are VEX-safe
    EXPECT_TRUE(Accum0::is_low_register);
    EXPECT_TRUE(Accum0::is_vex_safe);
    EXPECT_FALSE(Accum0::is_high_register);
    EXPECT_FALSE(Accum0::is_evex_only);

    EXPECT_TRUE(Input7::is_low_register); // zmm15
    EXPECT_TRUE(Input7::is_vex_safe);

    // HIGH registers (16-31) require EVEX
    EXPECT_FALSE(StateMax::is_low_register); // zmm16
    EXPECT_FALSE(StateMax::is_vex_safe);
    EXPECT_TRUE(StateMax::is_high_register);
    EXPECT_TRUE(StateMax::is_evex_only);

    EXPECT_TRUE(Scratch0::is_high_register); // zmm20
    EXPECT_TRUE(Scratch0::is_evex_only);

    // Verify zone-level encoding info
    EXPECT_TRUE(AccumulatorZone::all_low);
    EXPECT_FALSE(AccumulatorZone::has_high);
    EXPECT_EQ(AccumulatorZone::low_count, 8);
    EXPECT_EQ(AccumulatorZone::high_count, 0);

    EXPECT_TRUE(ScratchZone::all_high);
    EXPECT_FALSE(ScratchZone::has_low);
    EXPECT_EQ(ScratchZone::low_count, 0);
    EXPECT_EQ(ScratchZone::high_count, 6);
}

TEST(Test__RegisterAllocation, EncodingTypeTraits)
{
    // Type traits for encoding
    EXPECT_TRUE(is_vex_safe_v<Accum0>);
    EXPECT_TRUE(is_low_register_v<Accum0>);
    EXPECT_FALSE(is_evex_only_v<Accum0>);
    EXPECT_FALSE(is_high_register_v<Accum0>);

    EXPECT_FALSE(is_vex_safe_v<Scratch0>);
    EXPECT_FALSE(is_low_register_v<Scratch0>);
    EXPECT_TRUE(is_evex_only_v<Scratch0>);
    EXPECT_TRUE(is_high_register_v<Scratch0>);
}

// ============================================================================
// RegisterPool Tests
// ============================================================================

TEST(Test__RegisterAllocation, RegisterPoolBorrowAny)
{
    RegisterTracker tracker;
    auto scratch_pool = tracker.pool<ScratchZone>();

    // Should be able to borrow all 6 scratch registers
    EXPECT_EQ(scratch_pool.available_count(), 6);

    auto guard1 = scratch_pool.borrow_any();
    EXPECT_TRUE(guard1.has_value());
    EXPECT_EQ(scratch_pool.available_count(), 5);

    auto guard2 = scratch_pool.borrow_any();
    auto guard3 = scratch_pool.borrow_any();
    auto guard4 = scratch_pool.borrow_any();
    auto guard5 = scratch_pool.borrow_any();
    auto guard6 = scratch_pool.borrow_any();
    EXPECT_EQ(scratch_pool.available_count(), 0);

    // 7th borrow should fail
    auto guard7 = scratch_pool.borrow_any();
    EXPECT_FALSE(guard7.has_value());
}

TEST(Test__RegisterAllocation, RegisterPoolBorrowLow)
{
    RegisterTracker tracker;

    // ScratchZone has no LOW registers - should always fail
    auto scratch_pool = tracker.pool<ScratchZone>();
    EXPECT_EQ(scratch_pool.available_low_count(), 0);
    auto scratch_low = scratch_pool.borrow_low();
    EXPECT_FALSE(scratch_low.has_value());

    // AccumulatorZone is all LOW - should work
    auto accum_pool = tracker.pool<AccumulatorZone>();
    EXPECT_EQ(accum_pool.available_low_count(), 8);
    auto accum_low = accum_pool.borrow_low();
    EXPECT_TRUE(accum_low.has_value());
    EXPECT_TRUE(accum_low->is_low());
    EXPECT_TRUE(accum_low->is_vex_safe());
    EXPECT_EQ(accum_pool.available_low_count(), 7);
}

TEST(Test__RegisterAllocation, RegisterPoolBorrowHigh)
{
    RegisterTracker tracker;

    // AccumulatorZone has no HIGH registers - should always fail
    auto accum_pool = tracker.pool<AccumulatorZone>();
    EXPECT_EQ(accum_pool.available_high_count(), 0);
    auto accum_high = accum_pool.borrow_high();
    EXPECT_FALSE(accum_high.has_value());

    // ScratchZone is all HIGH - should work
    auto scratch_pool = tracker.pool<ScratchZone>();
    EXPECT_EQ(scratch_pool.available_high_count(), 6);
    auto scratch_high = scratch_pool.borrow_high();
    EXPECT_TRUE(scratch_high.has_value());
    EXPECT_TRUE(scratch_high->is_high());
    EXPECT_TRUE(scratch_high->is_evex_only());
    EXPECT_EQ(scratch_pool.available_high_count(), 5);
}

TEST(Test__RegisterAllocation, DynamicGuardXbyakAccessors)
{
    RegisterTracker tracker;
    auto pool = tracker.pool<AccumulatorZone>();

    auto guard = pool.borrow_any();
    EXPECT_TRUE(guard.has_value());

    // Should be able to get Xbyak registers
    Xbyak::Zmm zmm = guard->zmm();
    Xbyak::Ymm ymm = guard->ymm();
    Xbyak::Xmm xmm = guard->xmm();

    // All should have same register index
    EXPECT_EQ(zmm.getIdx(), ymm.getIdx());
    EXPECT_EQ(ymm.getIdx(), xmm.getIdx());

    // Since AccumulatorZone is LOW, vex_safe accessors should work
    Xbyak::Ymm ymm_safe = guard->ymm_vex_safe();
    Xbyak::Xmm xmm_safe = guard->xmm_vex_safe();
    EXPECT_EQ(ymm_safe.getIdx(), ymm.getIdx());
    EXPECT_EQ(xmm_safe.getIdx(), xmm.getIdx());
}

TEST(Test__RegisterAllocation, PoolAndStaticBorrowInteract)
{
    RegisterTracker tracker;

    // Borrow Scratch0 statically
    auto static_guard = tracker.borrow<Scratch0>();

    // Pool should see one less available
    auto scratch_pool = tracker.pool<ScratchZone>();
    EXPECT_EQ(scratch_pool.available_count(), 5);

    // Dynamic borrow should skip the already-borrowed register
    auto dynamic_guard = scratch_pool.borrow_any();
    EXPECT_TRUE(dynamic_guard.has_value());
    EXPECT_NE(dynamic_guard->absolute_index(), Scratch0::absolute_index);
}
