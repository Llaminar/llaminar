/**
 * @file Perf__MoEProjectionPipeline.cpp
 * @brief Release timing of the real six-node projection graph, including grouping.
 *
 * The full graph is compared with unsharded complete-expert compute, not with
 * an inferred sum of kernel timers. Inputs, exact arithmetic and prepared
 * formats are matched. This remains a synthetic stage experiment: production
 * model throughput and physical admission need their own subsequent proofs.
 * A1 cases change only explicit synthetic fixture geometry, ordinal membership
 * and counted transport; source labels never claim actual GGUF model payloads.
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
#ifdef HAVE_ROCM
/**
 * @brief Time A1's exact P/E/H/F/top-k/M geometry through the existing counted pipeline.
 * @param format Explicit synthetic original source codebooks, never actual model payloads.
 *
 * The native ordinal membership matches A1; source preparation and the complete-
 * expert oracle remain fixture-owned. All changing-owner/empty/reset checks run
 * before timing, and output bytes plus device-published useful packet extents
 * are checked outside every paired sample. This does not time shared FFN or
 * canonical full-H sums, and is not a real-model speed or quality certificate.
 */
void measureA1Geometry(const std::string &format)
{
    ProjectionPipelineTiming timing;
    runProjectionPipelineFixture(DeviceId::rocm(0), 4, {
        .geometry = {.experts = 512, .width = 2560, .intermediate = 640, .top_k = 10},
        .rows = {1}, .format = format,
        .exchange = ProjectionExchangeProof::DeviceCounted,
        .participant_order = ProjectionParticipantOrder::Ordinal,
        .service_phase = MoEOverlayServicePhaseHint::Decode,
        .timing_warmups = 5, .timing_samples = 31}, &timing);
    ASSERT_FALSE(::testing::Test::HasFailure());
    ASSERT_TRUE(timing.counted_extent_checks);
    std::printf("A1_GEOMETRY_PROJECTION_PIPELINE,scope=synthetic_fixture,ROCm,participants=4,rows=1,experts=512,"
        "width=2560,intermediate=640,top_k=10,source=%s,ordinal_members=0:1:2:3,exchange=device_counted,"
        "median_us=%.3f,p10_us=%.3f,p90_us=%.3f,unsharded_compute_us=%.3f,byte_exact=1,"
        "useful_packet_extents_checked=1,primitive_link_byte_receipt=unavailable,model_weights=0\n",
        format.c_str(), timing.median_us, timing.p10_us, timing.p90_us, timing.unsharded_compute_us);
}

/** @brief Same A1 geometry with a legal original 32-column Q8 source. */
TEST(MoEProjectionPipelineA1Perf, SyntheticQ8_0) { measureA1Geometry("Q8_0"); }
/** @brief Documented dominant UD-Q4_K_XL original codebooks; all payload values remain synthetic. */
TEST(MoEProjectionPipelineA1Perf, SyntheticQ4_K_Q5_1) { measureA1Geometry("Q4_K/Q5_1"); }
/** @brief Original Q4_1/Q5_1 sources produce representative physical dispatch IDs 5/7. */
TEST(MoEProjectionPipelineA1Perf, SyntheticQ4_1_Q5_1) { measureA1Geometry("Q4_1/Q5_1"); }
/** @brief Independently typed gate/up and down sources at the same A1 geometry. */
TEST(MoEProjectionPipelineA1Perf, SyntheticIQ2S_IQ4NL) { measureA1Geometry("IQ2_S/IQ4_NL"); }
#endif

INSTANTIATE_TEST_SUITE_P(Captured, MoEProjectionPipelinePerf, ::testing::Values(1, 16, 448, 512),
    [](const ::testing::TestParamInfo<int> &info) { return "Rows" + std::to_string(info.param); });
}
