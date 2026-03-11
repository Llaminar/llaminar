/**
 * @file BenchmarkMode.h
 * @brief Benchmark mode (--benchmark)
 */

#pragma once

#include "app/modes/IExecutionMode.h"

namespace llaminar2
{

    class BenchmarkMode : public IExecutionMode
    {
    public:
        const char *name() const override { return "benchmark"; }
        bool matches(const OrchestrationConfig &config) const override;
        int execute(AppContext &ctx) override;
    };

} // namespace llaminar2
