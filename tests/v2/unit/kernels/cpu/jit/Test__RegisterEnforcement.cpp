/**
 * @file Test__RegisterEnforcement.cpp
 * @brief Unit tests for compile-time register enforcement system
 *
 * Tests the type traits, concepts, and enforcement mechanisms from
 * RegisterEnforcement.h to ensure raw Xbyak register usage is properly
 * detected and rejected.
 */

#include <gtest/gtest.h>

#include "kernels/cpu/jit/RegisterAllocation.h"
#include "kernels/cpu/jit/RegisterGuard.h"
#include "kernels/cpu/jit/RegisterEnforcement.h"

using namespace llaminar2::jit;

// ============================================================================
// Type Trait Tests
// ============================================================================

TEST(Test__RegisterEnforcement, TypedZmmDetection)
{
    // TypedZmm types should be detected
    EXPECT_TRUE(is_typed_zmm_v<Accum0>);
    EXPECT_TRUE(is_typed_zmm_v<Accum7>);
    EXPECT_TRUE(is_typed_zmm_v<Input0>);
    EXPECT_TRUE(is_typed_zmm_v<Input7>);
    EXPECT_TRUE(is_typed_zmm_v<Scratch0>);
    EXPECT_TRUE(is_typed_zmm_v<Scratch5>);
    EXPECT_TRUE(is_typed_zmm_v<StateMax>);
    EXPECT_TRUE(is_typed_zmm_v<StateSum>);
    EXPECT_TRUE(is_typed_zmm_v<StateWeight>);
    EXPECT_TRUE(is_typed_zmm_v<StateCorr>);

    // Raw Xbyak types should NOT be detected as typed
    EXPECT_FALSE(is_typed_zmm_v<Xbyak::Zmm>);
    EXPECT_FALSE(is_typed_zmm_v<Xbyak::Ymm>);
    EXPECT_FALSE(is_typed_zmm_v<Xbyak::Xmm>);

    // TypedXmm should NOT be detected as TypedZmm
    EXPECT_FALSE(is_typed_zmm_v<Score0>);
    EXPECT_FALSE(is_typed_zmm_v<Score3>);
}

TEST(Test__RegisterEnforcement, TypedXmmDetection)
{
    // TypedXmm types should be detected
    EXPECT_TRUE(is_typed_xmm_v<Score0>);
    EXPECT_TRUE(is_typed_xmm_v<Score1>);
    EXPECT_TRUE(is_typed_xmm_v<Score2>);
    EXPECT_TRUE(is_typed_xmm_v<Score3>);

    // Raw Xbyak types should NOT be detected
    EXPECT_FALSE(is_typed_xmm_v<Xbyak::Xmm>);
    EXPECT_FALSE(is_typed_xmm_v<Xbyak::Zmm>);

    // TypedZmm should NOT be detected as TypedXmm
    EXPECT_FALSE(is_typed_xmm_v<Accum0>);
    EXPECT_FALSE(is_typed_xmm_v<Scratch0>);
}

TEST(Test__RegisterEnforcement, RegisterGuardDetection)
{
    RegisterTracker tracker;

    // Borrow a register to get a guard
    auto guard = tracker.borrow<Scratch4>();

    // Guard should be detected
    EXPECT_TRUE(is_register_guard_v<decltype(guard)>);
    EXPECT_TRUE(is_register_guard_v<RegisterGuard<Scratch4>>);
    EXPECT_TRUE(is_register_guard_v<RegisterGuard<Accum0>>);

    // Non-guards should NOT be detected
    EXPECT_FALSE(is_register_guard_v<Scratch4>);
    EXPECT_FALSE(is_register_guard_v<Xbyak::Zmm>);
    EXPECT_FALSE(is_register_guard_v<int>);
}

TEST(Test__RegisterEnforcement, TypedRegisterCombined)
{
    // All typed registers should satisfy is_typed_register_v
    EXPECT_TRUE(is_typed_register_v<Accum0>);
    EXPECT_TRUE(is_typed_register_v<Input3>);
    EXPECT_TRUE(is_typed_register_v<Scratch5>);
    EXPECT_TRUE(is_typed_register_v<Score0>);
    EXPECT_TRUE(is_typed_register_v<StateMax>);

    // Guards should also satisfy is_typed_register_v
    EXPECT_TRUE(is_typed_register_v<RegisterGuard<Scratch4>>);

    // Raw Xbyak should NOT satisfy
    EXPECT_FALSE(is_typed_register_v<Xbyak::Zmm>);
    EXPECT_FALSE(is_typed_register_v<Xbyak::Xmm>);
}

TEST(Test__RegisterEnforcement, RawXbyakDetection)
{
    // Raw Xbyak types should be detected
    EXPECT_TRUE(is_raw_xbyak_register_v<Xbyak::Zmm>);
    EXPECT_TRUE(is_raw_xbyak_register_v<Xbyak::Ymm>);
    EXPECT_TRUE(is_raw_xbyak_register_v<Xbyak::Xmm>);

    // Typed registers should NOT be detected as raw
    EXPECT_FALSE(is_raw_xbyak_register_v<Accum0>);
    EXPECT_FALSE(is_raw_xbyak_register_v<Score0>);
    EXPECT_FALSE(is_raw_xbyak_register_v<RegisterGuard<Scratch4>>);
}

// ============================================================================
// Kernel Register Manifest Tests
// ============================================================================

TEST(Test__RegisterEnforcement, ManifestNoConflicts)
{
    // Manifest with unique registers should have no conflicts
    using GoodManifest = KernelRegisterManifest<Scratch4, Scratch5, Input6>;
    EXPECT_FALSE(GoodManifest::has_conflicts());
    EXPECT_EQ(GoodManifest::count, 3u);
}

TEST(Test__RegisterEnforcement, ManifestZoneCheck)
{
    using ScratchOnlyManifest = KernelRegisterManifest<Scratch4, Scratch5>;

    // Should be from ScratchZone
    EXPECT_TRUE(ScratchOnlyManifest::all_from_zones<ScratchZone>());

    // Should NOT be from AccumulatorZone
    EXPECT_FALSE(ScratchOnlyManifest::all_from_zones<AccumulatorZone>());

    // Mixed manifest
    using MixedManifest = KernelRegisterManifest<Scratch4, Input6, Accum0>;
    EXPECT_TRUE((MixedManifest::all_from_zones<ScratchZone, QVectorZone, AccumulatorZone>()));
    EXPECT_FALSE(MixedManifest::all_from_zones<ScratchZone>());
}

TEST(Test__RegisterEnforcement, ManifestUsesRegister)
{
    using TestManifest = KernelRegisterManifest<Scratch4, Scratch5, Input6>;

    EXPECT_TRUE(TestManifest::uses_register<Scratch4>());
    EXPECT_TRUE(TestManifest::uses_register<Scratch5>());
    EXPECT_TRUE(TestManifest::uses_register<Input6>());

    EXPECT_FALSE(TestManifest::uses_register<Scratch0>());
    EXPECT_FALSE(TestManifest::uses_register<Accum0>());
}

TEST(Test__RegisterEnforcement, ManifestOverlap)
{
    using Manifest1 = KernelRegisterManifest<Scratch4, Scratch5>;
    using Manifest2 = KernelRegisterManifest<Scratch5, Input6>;
    using Manifest3 = KernelRegisterManifest<Accum0, Accum1>;

    // Manifest1 and Manifest2 overlap (both use Scratch5)
    EXPECT_TRUE(Manifest1::overlaps_with(Manifest2{}));

    // Manifest1 and Manifest3 don't overlap
    EXPECT_FALSE(Manifest1::overlaps_with(Manifest3{}));
}

TEST(Test__RegisterEnforcement, FastExpManifestValid)
{
    // Verify the predefined FastExp manifest
    EXPECT_FALSE(FastExpManifest::has_conflicts());
    EXPECT_EQ(FastExpManifest::count, 4u);
    EXPECT_TRUE((FastExpManifest::all_from_zones<ScratchZone, QVectorZone>()));
}

TEST(Test__RegisterEnforcement, ReservedConstantRegisters)
{
    // Verify the new constant/reserved register types exist and have correct indices
    static_assert(Const128::absolute_index == 26, "Const128 should be zmm26");
    static_assert(ConstScale::absolute_index == 27, "ConstScale should be zmm27");
    static_assert(ConstNegInf::absolute_index == 28, "ConstNegInf should be zmm28");
    static_assert(Const16::absolute_index == 28, "Const16 should alias zmm28 (shares with ConstNegInf)");
    static_assert(ConstOne::absolute_index == 29, "ConstOne should be zmm29");
    static_assert(ConstLog2e::absolute_index == 30, "ConstLog2e should be zmm30");
    static_assert(ConstExpMin::absolute_index == 31, "ConstExpMin should be zmm31");

    // Verify they're all from ReservedZone
    EXPECT_TRUE((is_zone_v<Const128, ReservedZone>));
    EXPECT_TRUE((is_zone_v<ConstScale, ReservedZone>));
    EXPECT_TRUE((is_zone_v<ConstNegInf, ReservedZone>));
    EXPECT_TRUE((is_zone_v<Const16, ReservedZone>));
    EXPECT_TRUE((is_zone_v<ConstOne, ReservedZone>));
    EXPECT_TRUE((is_zone_v<ConstLog2e, ReservedZone>));
    EXPECT_TRUE((is_zone_v<ConstExpMin, ReservedZone>));

    // Verify they're detected as typed registers
    EXPECT_TRUE(is_typed_zmm_v<Const128>);
    EXPECT_TRUE(is_typed_zmm_v<ConstExpMin>);

    // Verify zone_name
    EXPECT_STREQ(Const128::zone_type::name, "Reserved");

    // Manifest with constants should not have conflicts (unless intentional aliases)
    using ConstantManifest = KernelRegisterManifest<Const128, ConstScale, ConstOne, ConstLog2e, ConstExpMin>;
    EXPECT_FALSE(ConstantManifest::has_conflicts());
    EXPECT_EQ(ConstantManifest::count, 5u);
    EXPECT_TRUE(ConstantManifest::all_from_zones<ReservedZone>());

    // Note: ConstNegInf and Const16 are intentional aliases (both zmm28)
    // This is by design - different constants are loaded based on mode (decode vs prefill)
    static_assert(ConstNegInf::absolute_index == Const16::absolute_index,
                  "ConstNegInf and Const16 should alias zmm28");
}

// ============================================================================
// Register Name Helper Tests
// ============================================================================

TEST(Test__RegisterEnforcement, RegisterNameFromIndex)
{
    // Accumulator zone
    EXPECT_STREQ(RegisterNameFromIndex<0>::zone_name(), "Accumulator");
    EXPECT_EQ(RegisterNameFromIndex<0>::local_index(), 0);
    EXPECT_STREQ(RegisterNameFromIndex<7>::zone_name(), "Accumulator");
    EXPECT_EQ(RegisterNameFromIndex<7>::local_index(), 7);

    // QVector/Input zone
    EXPECT_STREQ(RegisterNameFromIndex<8>::zone_name(), "QVector/Input");
    EXPECT_EQ(RegisterNameFromIndex<8>::local_index(), 0);
    EXPECT_STREQ(RegisterNameFromIndex<15>::zone_name(), "QVector/Input");
    EXPECT_EQ(RegisterNameFromIndex<15>::local_index(), 7);

    // State zone
    EXPECT_STREQ(RegisterNameFromIndex<16>::zone_name(), "State");
    EXPECT_EQ(RegisterNameFromIndex<16>::local_index(), 0);
    EXPECT_STREQ(RegisterNameFromIndex<16>::suggested_type(), "StateMax");
    EXPECT_STREQ(RegisterNameFromIndex<17>::suggested_type(), "StateSum");
    EXPECT_STREQ(RegisterNameFromIndex<18>::suggested_type(), "StateWeight");
    EXPECT_STREQ(RegisterNameFromIndex<19>::suggested_type(), "StateCorr");

    // Scratch zone
    EXPECT_STREQ(RegisterNameFromIndex<20>::zone_name(), "Scratch");
    EXPECT_EQ(RegisterNameFromIndex<20>::local_index(), 0);
    EXPECT_STREQ(RegisterNameFromIndex<25>::zone_name(), "Scratch");
    EXPECT_EQ(RegisterNameFromIndex<25>::local_index(), 5);

    // Reserved zone
    EXPECT_STREQ(RegisterNameFromIndex<26>::zone_name(), "Reserved");
    EXPECT_EQ(RegisterNameFromIndex<26>::local_index(), 0);
}

// ============================================================================
// to_xbyak_* Helper Tests
// ============================================================================

TEST(Test__RegisterEnforcement, ToXbyakZmm)
{
    // TypedZmm
    Scratch4 typed_zmm;
    Xbyak::Zmm raw = to_xbyak_zmm(typed_zmm);
    EXPECT_EQ(raw.getIdx(), 24);

    // RegisterGuard
    RegisterTracker tracker;
    auto guard = tracker.borrow<Scratch5>();
    raw = to_xbyak_zmm(guard);
    EXPECT_EQ(raw.getIdx(), 25);
}

TEST(Test__RegisterEnforcement, ToXbyakXmm)
{
    // TypedXmm
    Score0 typed_xmm;
    Xbyak::Xmm raw = to_xbyak_xmm(typed_xmm);
    EXPECT_EQ(raw.getIdx(), 20);

    // TypedZmm (gets low 128 bits)
    Accum0 typed_zmm;
    raw = to_xbyak_xmm(typed_zmm);
    EXPECT_EQ(raw.getIdx(), 0);
}

// ============================================================================
// RegisterGuard Enhancement Tests
// ============================================================================

TEST(Test__RegisterEnforcement, BorrowSetWorks)
{
    RegisterTracker tracker;

    // Borrow multiple registers at once
    auto regs = tracker.borrow_set<Scratch4, Scratch5>();

    // Verify we can access them
    EXPECT_EQ(regs.get<0>().zmm().getIdx(), 24);
    EXPECT_EQ(regs.get<1>().zmm().getIdx(), 25);

    // Verify they're tracked as borrowed
    EXPECT_TRUE(tracker.is_borrowed<Scratch4>());
    EXPECT_TRUE(tracker.is_borrowed<Scratch5>());
}

TEST(Test__RegisterEnforcement, BorrowSetReleasesOnDestruction)
{
    RegisterTracker tracker;

    {
        auto regs = tracker.borrow_set<Scratch4, Scratch5, Input6>();
        EXPECT_TRUE(tracker.is_borrowed<Scratch4>());
        EXPECT_TRUE(tracker.is_borrowed<Scratch5>());
        EXPECT_TRUE(tracker.is_borrowed<Input6>());
    }

    // After scope ends, all should be released
    EXPECT_FALSE(tracker.is_borrowed<Scratch4>());
    EXPECT_FALSE(tracker.is_borrowed<Scratch5>());
    EXPECT_FALSE(tracker.is_borrowed<Input6>());
}

// ============================================================================
// Static Assert Verification (compile-time tests)
// ============================================================================

// These are compile-time checks. If the code compiles, the tests pass.
namespace compile_time_tests
{
    // Verify typed registers pass the macro
    template <typename T>
    void verify_typed_register(const T &reg)
    {
        REQUIRE_TYPED_REGISTER(reg);
    }

    // This function template would fail to compile if called with raw Xbyak
    template <typename T, require_typed_zmm<T> = true>
    void only_accepts_typed_zmm(const T &)
    {
    }

    void test_compile_time_enforcement()
    {
        Accum0 acc;
        Scratch4 scratch;
        Score0 score;

        // These should compile fine
        verify_typed_register(acc);
        verify_typed_register(scratch);
        verify_typed_register(score);

        only_accepts_typed_zmm(acc);
        only_accepts_typed_zmm(scratch);

        // These would NOT compile (uncomment to verify):
        // Xbyak::Zmm raw(20);
        // verify_typed_register(raw);  // static_assert failure
        // only_accepts_typed_zmm(raw); // SFINAE rejection
    }

} // namespace compile_time_tests

TEST(Test__RegisterEnforcement, CompileTimeEnforcementCompiles)
{
    // If this test runs, the compile-time checks passed
    compile_time_tests::test_compile_time_enforcement();
}

// ============================================================================
// C++20 Concept Tests (if available)
// ============================================================================

#if __cplusplus >= 202002L

TEST(Test__RegisterEnforcement, Cpp20Concepts)
{
    // Verify concepts work as expected
    static_assert(TypedZmmRegister<Accum0>);
    static_assert(TypedZmmRegister<Scratch5>);
    static_assert(!TypedZmmRegister<Xbyak::Zmm>);

    static_assert(TypedXmmRegister<Score0>);
    static_assert(!TypedXmmRegister<Accum0>);

    static_assert(TypedRegister<Accum0>);
    static_assert(TypedRegister<Score0>);
    static_assert(!TypedRegister<Xbyak::Zmm>);

    static_assert(NotRawXbyakRegister<Accum0>);
    static_assert(!NotRawXbyakRegister<Xbyak::Zmm>);
}

#endif
