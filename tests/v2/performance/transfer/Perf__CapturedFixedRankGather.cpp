/**
 * @file Perf__CapturedFixedRankGather.cpp
 * @brief Opt-in Release paired native versus mapped fixed-message component timing.
 *
 * The same shared actual fixture retains the original buffers, PMA and native
 * captures. Counts 1/97 use fullyActive(1), H2560 and 31 native-event samples per
 * path with rotated host launch order. The existing original MPI inventory
 * selects the whole NUMA-local cohort and retains inactive-rank retirement.
 * No production dispatch or arithmetic
 * setting is changed. Timings include device waits and host-arrival skew.
 */
#include "../../integration/transfer/CapturedFixedRankGatherProbe.h"
#include "../../fixtures/backends/OriginalRankLocalNativeReplay.h"
#include <gtest/gtest.h>

#ifdef HAVE_CUDA
/** @test Paired CUDA P2 fixed-message component cost, not a model benchmark. */
TEST(CapturedFixedRankGatherEconomy, CUDA2)
{
    llaminar2::test::originalRankLocalReplay(llaminar2::DeviceType::CUDA, 2, [&](const auto &cohort) {
        ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe(cohort, true));
    });
}
#endif
#ifdef HAVE_ROCM
/** @test Twelve actual 10 KiB edges on P4, original and mapped components paired. */
TEST(CapturedFixedRankGatherEconomy, ROCm4)
{
    llaminar2::test::originalRankLocalReplay(llaminar2::DeviceType::ROCm, 4, [&](const auto &cohort) {
        ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe(cohort, true));
    });
}
#endif
