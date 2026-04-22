/**
 * @file OneshotCommand.h
 * @brief 'llaminar oneshot' — single inference (completion, chat) and exit.
 *
 * Consolidates CompletionMode, SingleShotChatMode, and InteractiveChatMode
 * under a single subcommand. Benchmark mode has its own subcommand
 * ('llaminar benchmark'). The existing OrchestrationConfig flags control
 * which mode runs.
 */

#pragma once

#include "app/ICommand.h"

namespace llaminar2
{

    class OneshotCommand : public ICommand
    {
    public:
        const char *name() const override { return "oneshot"; }
        const char *description() const override { return "Run single inference and exit"; }
        int execute(int argc, char *argv[]) override;
    };

} // namespace llaminar2
