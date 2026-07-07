#pragma once

namespace llaminar2
{
    namespace collective_timeout_policy
    {
        inline int effectiveCollectTimeoutMs(int configured_timeout_ms,
                                             bool cold_start_completed)
        {
            (void)cold_start_completed;
            return configured_timeout_ms;
        }

        inline int effectiveWorkerJoinTimeoutMs(int configured_timeout_ms)
        {
            (void)configured_timeout_ms;
            // LLAMINAR_TP_COLLECT_TIMEOUT_MS is a collective/rendezvous timeout,
            // not a wall-clock limit for an entire per-device forward. Snapshot
            // and parity runs can legitimately spend more than one collective
            // timeout inside a single forward while every collective is healthy.
            return 0;
        }
    } // namespace collective_timeout_policy
} // namespace llaminar2
