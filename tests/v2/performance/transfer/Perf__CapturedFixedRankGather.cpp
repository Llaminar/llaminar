/**
 * @file Perf__CapturedFixedRankGather.cpp
 * @brief Opt-in Release paired native versus mapped fixed-message component timing.
 *
 * The same shared actual fixture retains the original buffers, PMA and native
 * captures. Counts 1/97 use fullyActive(1), H2560 and 31 native-event samples per
 * path with rotated host launch order. No production dispatch or arithmetic
 * setting is changed. Timings include device waits and host-arrival skew.
 */
#include "../../integration/transfer/CapturedFixedRankGatherProbe.h"
#include <gtest/gtest.h>

#ifdef HAVE_CUDA
/** @test Paired CUDA P2 fixed-message component cost, not a model benchmark. */
TEST(CapturedFixedRankGatherEconomy, CUDA2)
{
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({llaminar2::DeviceId::cuda(0), llaminar2::DeviceId::cuda(1)}, true));
}
#endif
#ifdef HAVE_ROCM
/** @test Twelve actual 10 KiB edges on P4, original and mapped components paired. */
TEST(CapturedFixedRankGatherEconomy, ROCm4)
{
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({llaminar2::DeviceId::rocm(0), llaminar2::DeviceId::rocm(1),
        llaminar2::DeviceId::rocm(2), llaminar2::DeviceId::rocm(3)}, true));
}
#endif
