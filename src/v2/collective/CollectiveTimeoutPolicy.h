#pragma once

namespace llaminar2
{
    namespace collective_timeout_policy
    {
        /** Canonical upper bound for one collective rendezvous. */
        inline constexpr int kDefaultCollectiveTimeoutMs = 30000;

        /**
         * @brief Resolve the deadline for one collective rendezvous.
         *
         * Collective startup and steady-state execution intentionally share the
         * same deadline. First-use allocation, communicator initialization, and
         * graph-capture preparation must complete before participants enter the
         * rendezvous; extending the collective wait would only hide asymmetric
         * launch order or missing participants.
         *
         * @param configured_timeout_ms Explicit timeout from runtime policy, or
         *        a non-positive value to use the canonical 30-second default.
         * @return Timeout for one collective rendezvous, in milliseconds.
         */
        inline int effectiveCollectTimeoutMs(int configured_timeout_ms)
        {
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
