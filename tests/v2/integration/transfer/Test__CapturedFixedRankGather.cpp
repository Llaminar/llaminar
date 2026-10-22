/**
 * @file Test__CapturedFixedRankGather.cpp
 * @brief Real retained CUDA/ROCm fixed-rank channel/fold correctness gates.
 *
 * The shared component owns original PMA resources and compares each complete
 * captured mapped transaction with the unchanged native gather/full-bank fold.
 * These explicit GPU tests are distinct from device-free geometry contracts.
 */
#include "CapturedFixedRankGatherProbe.h"
#include <gtest/gtest.h>

#ifdef HAVE_CUDA
/** @test Reversed actual CUDA membership, retained tails and original FP32 bytes. */
TEST(CapturedFixedRankGather, CUDA2)
{
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({llaminar2::DeviceId::cuda(1), llaminar2::DeviceId::cuda(0)}, false));
}
#endif
#ifdef HAVE_ROCM
/** @test Permuted ROCm P4 membership keeps communicator indices separate from ordinals. */
TEST(CapturedFixedRankGather, ROCm4)
{
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({llaminar2::DeviceId::rocm(2), llaminar2::DeviceId::rocm(0),
        llaminar2::DeviceId::rocm(3), llaminar2::DeviceId::rocm(1)}, false));
}
#endif
