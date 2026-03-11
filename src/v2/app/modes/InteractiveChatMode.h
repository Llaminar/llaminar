/**
 * @file InteractiveChatMode.h
 * @brief Interactive chat mode (--chat)
 */

#pragma once

#include "app/modes/IExecutionMode.h"

namespace llaminar2
{

    class InteractiveChatMode : public IExecutionMode
    {
    public:
        const char *name() const override { return "interactive-chat"; }
        bool matches(const OrchestrationConfig &config) const override;
        int execute(AppContext &ctx) override;
    };

} // namespace llaminar2
