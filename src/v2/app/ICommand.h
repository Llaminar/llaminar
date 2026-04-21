/**
 * @file ICommand.h
 * @brief Interface for CLI subcommands (plan, serve, oneshot, describe)
 *
 * Each subcommand owns its full lifecycle: flag parsing, initialization,
 * and execution. This decouples heavy commands (serve, oneshot) that need
 * MPI + model loading from lightweight ones (describe) that don't.
 */

#pragma once

namespace llaminar2
{

    /**
     * @brief Interface for a top-level CLI subcommand.
     *
     * Registered with SubcommandRouter. When dispatched, receives argc/argv
     * with the subcommand token already stripped (argv[0] is the binary name,
     * argv[1] is the first flag).
     */
    class ICommand
    {
    public:
        virtual ~ICommand() = default;

        /// Subcommand token (e.g. "plan", "serve", "oneshot", "describe")
        virtual const char *name() const = 0;

        /// One-line description for top-level --help
        virtual const char *description() const = 0;

        /// Execute the command. Returns process exit code.
        /// argc/argv have the subcommand token shifted out.
        virtual int execute(int argc, char *argv[]) = 0;
    };

} // namespace llaminar2
