/**
 * @file LegacyCommand.h
 * @brief Backward-compatible command that runs the original flat-flag AppLifecycle path.
 *
 * Used as the fallback when no subcommand token is found, or when argv[1]
 * starts with '-'. This preserves 100% backward compatibility with existing
 * invocations like: llaminar2 -m model.gguf -p "Hello" -n 50
 */

#pragma once

#include "app/ICommand.h"

namespace llaminar2
{

    class LegacyCommand : public ICommand
    {
    public:
        const char *name() const override { return "legacy"; }
        const char *description() const override { return "Classic flag-based invocation (backward compat)"; }
        int execute(int argc, char *argv[]) override;
    };

} // namespace llaminar2
