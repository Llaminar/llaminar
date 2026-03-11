/**
 * @file SingleShotChatMode.h
 * @brief Single-shot chat mode (--chat-single)
 */

#pragma once

#include "app/modes/IExecutionMode.h"

namespace llaminar2
{

    class SingleShotChatMode : public IExecutionMode
    {
    public:
        const char *name() const override { return "single-shot-chat"; }
        bool matches(const OrchestrationConfig &config) const override;
        int execute(AppContext &ctx) override;
    };

} // namespace llaminar2
