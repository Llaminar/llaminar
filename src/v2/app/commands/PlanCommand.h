/**
 * @file PlanCommand.h
 * @brief 'llaminar plan' — analyze cluster and produce an execution plan.
 *
 * Queries device inventory, reads model header (architecture, layer count),
 * applies the shared typed candidate/admission planner, and writes a lossless
 * versioned OrchestrationConfig document that can be consumed by
 * 'llaminar serve --config <plan.yaml>'.
 */

#pragma once

#include "app/ICommand.h"

namespace llaminar2
{

    /** @brief Plan one automatic inference request and emit its apply document. */
    class PlanCommand : public ICommand
    {
    public:
        /** @return Public subcommand identity used for MPI self-launch. */
        const char *name() const override { return "plan"; }
        /** @return Short purpose for root CLI help. */
        const char *description() const override { return "Analyze cluster and produce execution plan"; }
        /**
         * @brief Parse, discover, admit and serialize the shared serving policy.
         * @param argc Argument count without the subcommand token.
         * @param argv Executable and shared runtime/presentation arguments.
         * @return Zero only after successful help/validation or completed planning.
         */
        int execute(int argc, char *argv[]) override;
    };

} // namespace llaminar2
