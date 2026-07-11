#pragma once

namespace llaminar2
{
    namespace collective_timeout_policy
    {
        /** Canonical upper bound for one collective rendezvous. */
        inline constexpr int kDefaultCollectiveTimeoutMs = 30000;

        inline int effectiveCollectTimeoutMs(int configured_timeout_ms,
                                             bool cold_start_completed)
        {
            (void)cold_start_completed;
            return configured_timeout_ms > 0
                       ? configured_timeout_ms
                       : kDefaultCollectiveTimeoutMs;
        }

        inline int effectiveWorkerJoinTimeoutMs(int configured_timeout_ms)
        {
            (void)configured_timeout_ms;
            /*
             * LLAMINAR_TP_COLLECT_TIMEOUT_MS bounds one collective rendezvous,
             * not an entire participant forward. A forward can legitimately
             * execute many healthy collectives and expensive graph diagnostics.
             * The backend timeout aborts a genuinely stuck collective; the
             * enclosing worker join remains the completion fence for the whole
             * participant operation.
             */
            return 0;
        }
    } // namespace collective_timeout_policy
} // namespace llaminar2
