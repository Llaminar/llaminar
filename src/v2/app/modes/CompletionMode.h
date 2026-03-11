/**
 * @file CompletionMode.h
 * @brief Standard one-shot completion mode (default, -p "...")
 */

#pragma once

#include "app/modes/IExecutionMode.h"

namespace llaminar2
{

    class CompletionMode : public IExecutionMode
    {
    public:
        const char *name() const override { return "completion"; }
        bool matches(const OrchestrationConfig &config) const override;
        int execute(AppContext &ctx) override;
    };

} // namespace llaminar2
