/**
 * @file ServerMode.h
 * @brief HTTP server mode (--serve) with OpenAI-compatible REST API
 */

#pragma once

#include "app/modes/IExecutionMode.h"

namespace llaminar2
{

    class ServerMode : public IExecutionMode
    {
    public:
        const char *name() const override { return "server"; }
        bool matches(const OrchestrationConfig &config) const override;
        int execute(AppContext &ctx) override;
    };

} // namespace llaminar2
