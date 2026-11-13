/**
 * @file Test__CapturedFixedRankGather.cpp
 * @brief Real retained CUDA/ROCm fixed-rank channel/fold correctness gates.
 *
 * The shared component owns original PMA resources and compares each complete
 * captured mapped transaction with the unchanged native gather/full-bank fold.
 * These explicit GPU tests are distinct from device-free geometry contracts.
 */
#include "CapturedFixedRankGatherProbe.h"
#include "../../fixtures/backends/OriginalRankLocalNativeReplay.h"
using llaminar2::test::originalRankLocalReplay;
#include <gtest/gtest.h>
#include <array>

#ifdef HAVE_CUDA
/** @test Reversed actual CUDA membership, retained tails and original FP32 bytes. */
TEST(CapturedFixedRankGather, CUDA2)
{
    originalRankLocalReplay(llaminar2::DeviceType::CUDA, 2, [&](const auto &cohort) {
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({cohort[1], cohort[0]}, false));

    });
}
/** @test Exact fixed-message full-scalar bounds, independent original fold and retained cursor identity. */
TEST(CapturedFixedRankGatherBatch16KiBBoundaries, CUDA2)
{
    originalRankLocalReplay(llaminar2::DeviceType::CUDA, 2, [&](const auto &cohort) {
    using Shape = llaminar2::fixed_rank_component_test::ProbeShape;
    // Byte-granular +/-1 edges are covered by the counted protocol fixture.
    // This FP32 fold uses the nearest whole-scalar edges, never fractional data.
    const std::array shapes{Shape{4097, 1, false}, Shape{2047, 1, false}, Shape{2048, 1, false},
        Shape{2049, 1, false}, Shape{2560, 1, false}, Shape{4095, 1, false}, Shape{4096, 1, true}};
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({cohort[1], cohort[0]}, false, shapes));

    });
}

#endif
#ifdef HAVE_ROCM
/** @test Permuted ROCm P4 membership keeps communicator indices separate from ordinals. */
TEST(CapturedFixedRankGather, ROCm4)
{
    originalRankLocalReplay(llaminar2::DeviceType::ROCm, 4, [&](const auto &cohort) {
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({cohort[2], cohort[0],
        cohort[3], cohort[1]}, false));

    });
}
/** @test Exact fixed-message full-scalar bounds, independent original fold and retained cursor identity. */
TEST(CapturedFixedRankGatherBatch16KiBBoundaries, ROCm4)
{
    originalRankLocalReplay(llaminar2::DeviceType::ROCm, 4, [&](const auto &cohort) {
    using Shape = llaminar2::fixed_rank_component_test::ProbeShape;
    // Byte-granular +/-1 edges are covered by the counted protocol fixture.
    // This FP32 fold uses the nearest whole-scalar edges, never fractional data.
    const std::array shapes{Shape{4097, 1, false}, Shape{2047, 1, false}, Shape{2048, 1, false},
        Shape{2049, 1, false}, Shape{2560, 1, false}, Shape{4095, 1, false}, Shape{4096, 1, true}};
    ASSERT_NO_THROW(llaminar2::fixed_rank_component_test::probe({cohort[2], cohort[0],
        cohort[3], cohort[1]}, false, shapes));

    });
}

#endif
