/**
 * @file PlanCommand.h
 * @brief 'llaminar plan' — analyze cluster and produce an execution plan.
 *
 * Queries device inventory, reads model header (architecture, layer count),
 * applies a placement strategy, and writes a serialized OrchestrationConfig
 * YAML file that can be consumed by 'llaminar serve --config <plan.yaml>'.
 */

#pragma once

#include "app/ICommand.h"

namespace llaminar2
{

    class PlanCommand : public ICommand
    {
    public:
        const char *name() const override { return "plan"; }
        const char *description() const override { return "Analyze cluster and produce execution plan"; }
        int execute(int argc, char *argv[]) override;
    };

} // namespace llaminar2
