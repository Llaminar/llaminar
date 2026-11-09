/**
 * @file CPUPerfStatsTimingCollectionScope.h
 * @brief Reuse the existing quiescent CPU trainer recording policy around clocks.
 *
 * This test-only scope preserves the caller's filter and immutable collector
 * snapshots. It is never entered during production execution or profiling.
 */
#pragma once
#include "utils/PerfStatsCollector.h"
#include <cstdlib>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
namespace llaminar2::test
{
/**
 * @brief Test-owner recording suppression; all workers must be quiescent on entry/exit.
 *
 * Only the collection filter changes. Existing immutable PerfStats policies own
 * publication/lifetime; production execution, request topology and record data
 * are untouched. This cold owner control never surrounds profiler evidence.
 */
class CPUExactTimingCollectionScope final
{
public:
    /** @brief Retain the exact original filter, then withdraw every timing record. */
    CPUExactTimingCollectionScope()
    {
        if (const char *value = std::getenv("LLAMINAR_PERF_STATS_FILTER")) prior_filter_ = value;
        if (setenv("LLAMINAR_PERF_STATS_FILTER", "cpu_exact_timing_disabled", 1) != 0)
            throw std::runtime_error("Cannot suppress CPU exact timing collection");
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    /** @brief Restore the caller's exact policy before a later proof/export. */
    ~CPUExactTimingCollectionScope() noexcept
    {
        const int status = prior_filter_
            ? setenv("LLAMINAR_PERF_STATS_FILTER", prior_filter_->c_str(), 1)
            : unsetenv("LLAMINAR_PERF_STATS_FILTER");
        if (status != 0) std::terminate();
        PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    CPUExactTimingCollectionScope(const CPUExactTimingCollectionScope &) = delete;
    CPUExactTimingCollectionScope &operator=(const CPUExactTimingCollectionScope &) = delete;
private:
    std::optional<std::string> prior_filter_; ///< Restored after the team retires its timed reads.
};

}
