/**
 * @file Perf__MoERouterOwnedExchange.cpp
 * @brief Opt-in full-transaction routing economy, separate from functional gates.
 *
 * Both arithmetic and actual counted transport are included. Live prefix sizes
 * deliberately differ from retained storage, matching partial production
 * buckets. Nine paired max-participant device-event samples follow two warmups.
 * This is a component experiment, not a whole-model throughput certificate.
 */
#include "utils/MoERouterOwnedExchangeFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <utility>

namespace
{
    /** @brief Report median paired graph intervals, never sum overlapping GPUs. */
    void measure(llaminar2::DeviceId first, int participants)
    {
        for (const auto [capacity, rows] : {std::pair{64, 64}, {128, 128}, {256, 256},
                 {512, 448}, {512, 512}, {512, 64}})
        {
            llaminar2::test::MoERouterOwnedExchangeTiming timing;
            llaminar2::test::proveMoERouterOwnedExchange(first, participants, capacity, 2048, 256, {rows}, &timing);
            ASSERT_EQ(timing.replicated_us.size(), 9u);
            ASSERT_EQ(timing.distributed_us.size(), 9u);
            for (size_t sample = 0; sample < timing.replicated_us.size(); ++sample)
                std::printf("ROUTER_SAMPLE backend=%s participants=%d capacity=%d live=%d sample=%zu replicated_us=%.6f distributed_us=%.6f\n",
                    first.is_cuda() ? "CUDA" : "ROCm", participants, capacity, rows, sample,
                    timing.replicated_us[sample], timing.distributed_us[sample]);
            std::sort(timing.replicated_us.begin(), timing.replicated_us.end());
            std::sort(timing.distributed_us.begin(), timing.distributed_us.end());
            std::printf("ROUTER_EXCHANGE backend=%s participants=%d capacity=%d live=%d replicated_us=%.6f distributed_us=%.6f speedup=%.6f\n",
                first.is_cuda() ? "CUDA" : "ROCm", participants, capacity, rows, timing.replicated_us[4],
                timing.distributed_us[4], timing.replicated_us[4] / timing.distributed_us[4]);
        }
    }
}
#ifdef HAVE_CUDA
TEST(MoERouterOwnedExchangePerf, CUDA2) { measure(llaminar2::DeviceId::cuda(0), 2); }
#endif
#ifdef HAVE_ROCM
TEST(MoERouterOwnedExchangePerf, ROCm2) { measure(llaminar2::DeviceId::rocm(0), 2); }
TEST(MoERouterOwnedExchangePerf, ROCm4) { measure(llaminar2::DeviceId::rocm(0), 4); }
#endif
