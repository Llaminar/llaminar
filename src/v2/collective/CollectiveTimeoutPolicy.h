#pragma once

#include <algorithm>

namespace llaminar2
{
    namespace collective_timeout_policy
    {
        constexpr int kColdStartCollectTimeoutMs = 300000;

        inline int effectiveCollectTimeoutMs(int configured_timeout_ms,
                                             bool cold_start_completed)
        {
            if (configured_timeout_ms <= 0 || cold_start_completed)
                return configured_timeout_ms;
            return std::max(configured_timeout_ms, kColdStartCollectTimeoutMs);
        }
    } // namespace collective_timeout_policy
} // namespace llaminar2
