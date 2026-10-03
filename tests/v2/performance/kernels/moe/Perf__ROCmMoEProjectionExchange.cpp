/**
 * @file Perf__ROCmMoEProjectionExchange.cpp
 * @brief Paired Release economy of complete two-device projection transactions.
 *
 * Includes compute, bitwise pack/consume, native RCCL and column assembly. The
 * shared fixture checks exact output on every measured replay; no performance
 * threshold is installed in the production preflight gate.
 */
#include <gtest/gtest.h>
#include "../../../utils/ROCmMoEProjectionExchangeFixture.h"
#include <cstdio>

namespace
{
    /** @brief One exact geometry per launch keeps profiler evidence isolated. */
    class ROCmMoEProjectionExchangePerf : public ::testing::TestWithParam<int> {};
    TEST_P(ROCmMoEProjectionExchangePerf, CapturedCompleteTransaction)
    {
        using namespace llaminar2::test::projection_exchange;
        Timing timing;
        run({.rows = GetParam(), .width = 2048, .intermediate = 512,
            .experts = 256, .top_k = 8, .warmups = 5, .samples = 21}, &timing);
        std::printf("COMPLETE_PROJECTION_EXCHANGE,ROCm,rows=%d,experts=256,top_k=8,median_us=%.3f,p10_us=%.3f,p90_us=%.3f,control_us=%.3f,speedup=%.4f,byte_exact=1\n",
            GetParam(), timing.median_us, timing.p10_us, timing.p90_us,
            timing.control_us, timing.control_us / timing.median_us);
    }
    TEST_P(ROCmMoEProjectionExchangePerf, CapturedIndependentRoutes)
    {
        using namespace llaminar2::test::projection_exchange;
        Timing timing;
        run({.rows = GetParam(), .width = 2048, .intermediate = 512,
            .experts = 256, .top_k = 8, .warmups = 5, .samples = 21,
            .down_publication = DownPublication::IndependentRoutes}, &timing);
        std::printf("PARALLEL_PROJECTION_EXCHANGE,ROCm,rows=%d,experts=256,top_k=8,median_us=%.3f,p10_us=%.3f,p90_us=%.3f,control_us=%.3f,speedup=%.4f,byte_exact=1\n",
            GetParam(), timing.median_us, timing.p10_us, timing.p90_us,
            timing.control_us, timing.control_us / timing.median_us);
    }
    TEST_P(ROCmMoEProjectionExchangePerf, CapturedQwen36ProjectionFormats)
    {
        using namespace llaminar2::test::projection_exchange;
        Timing timing;
        // The model's IQ3_S file name is not every tensor's codebook. Its
        // dominant routed pair executes IQ2_S gate/up and IQ4_NL down.
        run({.rows = GetParam(), .width = 2048, .intermediate = 512,
            .experts = 256, .top_k = 8, .warmups = 5, .samples = 21,
            .down_publication = DownPublication::IndependentRoutes,
            .gate_up_format = "IQ2_S", .down_format = "IQ4_NL"}, &timing);
        std::printf("MIXED_PROJECTION_EXCHANGE,ROCm,rows=%d,gate_up=IQ2_S,down=IQ4_NL,experts=256,top_k=8,median_us=%.3f,p10_us=%.3f,p90_us=%.3f,control_us=%.3f,speedup=%.4f,byte_exact=1\n",
            GetParam(), timing.median_us, timing.p10_us, timing.p90_us,
            timing.control_us, timing.control_us / timing.median_us);
    }
    INSTANTIATE_TEST_SUITE_P(Captured, ROCmMoEProjectionExchangePerf, ::testing::Values(1, 16, 64, 448, 512),
        [](const ::testing::TestParamInfo<int> &info) { return "Rows" + std::to_string(info.param); });
}
