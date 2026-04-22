/**
 * @file BenchmarkCommand.h
 * @brief 'llaminar benchmark' — measure prefill/decode performance.
 *
 * Supports two modes:
 *   - Local benchmark: single-device or local multi-device (TP/PP)
 *   - Cluster benchmark: distributed across machines via MPI
 *
 * All device/parallelism flags from OrchestrationConfig are available.
 */

#pragma once

#include "app/ICommand.h"

namespace llaminar2
{

    class BenchmarkCommand : public ICommand
    {
    public:
        const char *name() const override { return "benchmark"; }
        const char *description() const override { return "Measure prefill/decode performance"; }
        int execute(int argc, char *argv[]) override;
    };

} // namespace llaminar2
