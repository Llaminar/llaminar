/**
 * @file PeerPushReplayValues.h
 * @brief Adversarial contribution values shared by native peer-push replays.
 *
 * Cancellation and signed zero verify arithmetic order, while a separate lane
 * family changes every rank's payload on each repeat. A two-rank replay must
 * not consist entirely of 2^24 + 1: that rounded result could conceal stale or
 * missing small contributions even when its byte comparison passes.
 */
#pragma once
#include <cstddef>

namespace llaminar2::test
{
    /**
     * @brief Produce deterministic finite input bits for one actual participant.
     * @param rank Original communicator member, with no fixed count assumption.
     * @param operation Distinct output binding within the captured transaction.
     * @param column Element offset, selecting cancellation/zero/change lanes.
     * @param repeat Dirty replay index; pass zero for fixed-work timing samples.
     * @return FP32 contribution whose changing lanes affect the ordered sum.
     */
    inline float peerPushReplayValue(int rank, int operation, std::size_t column, int repeat)
    {
        if (column%11==0) return -0.0f;
        if (column%5==1)
            return float((rank+1)*16+operation*3+int(column%17)) + float(repeat)*0.25f;
        if (rank==0) return operation?8388608.0f:16777216.0f;
        if (rank==1) return 1.0f;
        if (rank==2) return -16777216.0f;
        return float(1+rank+operation+column%7+repeat%3);
    }
}
