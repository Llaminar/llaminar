/**
 * @file ServeCommand.h
 * @brief 'llaminar serve' — start OpenAI-compatible HTTP server.
 */

#pragma once

#include "app/ICommand.h"

namespace llaminar2
{

    class ServeCommand : public ICommand
    {
    public:
        const char *name() const override { return "serve"; }
        const char *description() const override { return "Start OpenAI-compatible HTTP server"; }
        int execute(int argc, char *argv[]) override;
    };

} // namespace llaminar2
