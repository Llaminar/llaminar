/**
 * @file MoERouterOwnedExchangeFixture.h
 * @brief Real-device correctness and opt-in economy proof for distributed routing.
 *
 * Functional gates never measure or assert timing. The separate performance
 * entry point times the complete captured producer/exchange/publication and
 * its replicated-router control, with identical arithmetic and input values.
 */
#pragma once
#include "backends/DeviceId.h"
#include <vector>

namespace llaminar2::test
{
    /** @brief Max-participant native event intervals; no sum of overlapping GPUs. */
    struct MoERouterOwnedExchangeTiming
    {
        std::vector<double> replicated_us;
        std::vector<double> distributed_us;
    };

    /** @brief Exercise retained graphs under changing live-prefix geometry.
     * @param first Backend and first physical ordinal in the domain.
     * @param participants Actual homogeneous GPU count, reversed in logical order.
     * @param capacity Whole-matrix captured row capacity.
     * @param width Router hidden columns, a multiple of 32.
     * @param experts Router output columns, at least eight.
     * @param live_counts Ordered request lengths, including optional empty replays.
     * @param timing Optional explicit performance experiment; requires one live count.
     * @throws std::exception On missing hardware, broken publication, wrong bits or spill evidence. */
    void proveMoERouterOwnedExchange(DeviceId first, int participants, int capacity,
        int width, int experts, const std::vector<int> &live_counts,
        MoERouterOwnedExchangeTiming *timing = nullptr);
}
