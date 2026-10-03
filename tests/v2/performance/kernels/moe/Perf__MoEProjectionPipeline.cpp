/**
 * @file Perf__MoEProjectionPipeline.cpp
 * @brief Release timing of the real six-node projection graph, including grouping.
 *
 * The full graph is compared with unsharded complete-expert compute, not with
 * an inferred sum of kernel timers. Inputs, exact arithmetic and prepared
 * formats are matched. This remains a synthetic stage experiment: production
 * model throughput and physical admission need their own subsequent proofs.
 */
#include <gtest/gtest.h>
#include "utils/MoEProjectionPipelineFixture.h"
#include <cstdio>

namespace llaminar2::test
{
/** @brief One shape per invocation isolates timing/profiler evidence. */
class MoEProjectionPipelinePerf : public ::testing::TestWithParam<int> {};

#ifdef HAVE_ROCM
TEST_P(MoEProjectionPipelinePerf, ROCm2)
{
    ProjectionPipelineTiming timing;
    runProjectionPipelineFixture(DeviceId::rocm(0), 2, {
        .geometry = {.experts = 256, .width = 2048, .intermediate = 512, .top_k = 8},
        .rows = {GetParam()}, .format = "IQ2_S/IQ4_NL", .timing_warmups = 5, .timing_samples = 31}, &timing);
    ASSERT_FALSE(HasFailure());
    std::printf("LOWERED_PROJECTION_PIPELINE,ROCm,participants=2,rows=%d,gate_up=IQ2_S,down=IQ4_NL,"
        "median_us=%.3f,p10_us=%.3f,p90_us=%.3f,unsharded_compute_us=%.3f,compute_ratio=%.4f,byte_exact=1\n",
        GetParam(), timing.median_us, timing.p10_us, timing.p90_us,
        timing.unsharded_compute_us, timing.unsharded_compute_us / timing.median_us);
}
#endif
INSTANTIATE_TEST_SUITE_P(Captured, MoEProjectionPipelinePerf, ::testing::Values(1, 16, 448, 512),
    [](const ::testing::TestParamInfo<int> &info) { return "Rows" + std::to_string(info.param); });
}
