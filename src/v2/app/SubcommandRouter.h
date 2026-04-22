/**
 * @file SubcommandRouter.h
 * @brief Routes argv[1] to a registered ICommand.
 *
 * Dispatch logic:
 *   1. argc < 2, argv[1] starts with '-', or argv[1] == "help" → print help, return 0
 *   2. argv[1] matches a registered command → cmd->execute(argc-1, argv+1)
 *   3. No match                            → error message, return 1
 */

#pragma once

#include "app/ICommand.h"
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{

    class SubcommandRouter
    {
    public:
        /// Register a subcommand. Ownership transferred.
        SubcommandRouter &add(std::unique_ptr<ICommand> cmd);

        /// Set the fallback command for legacy (no-subcommand) invocations.
        /// Must be set before dispatch().
        SubcommandRouter &setFallback(std::unique_ptr<ICommand> cmd);

        /**
         * @brief Dispatch based on argv[1].
         *
         * @param argc  Original argc from main()
         * @param argv  Original argv from main()
         * @return Process exit code
         */
        int dispatch(int argc, char *argv[]) const;

        /// Generate the top-level help text listing all subcommands.
        std::string getTopLevelHelp(const char *binary_name = "llaminar2") const;

        /// Find a command by name (for testing). Returns nullptr if not found.
        const ICommand *find(const std::string &name) const;

        /// Number of registered (non-fallback) commands.
        size_t size() const { return commands_.size(); }

    private:
        std::vector<std::unique_ptr<ICommand>> commands_;
        std::unique_ptr<ICommand> fallback_;

        bool isHelpRequest(const char *arg) const;
        bool looksLikeFlag(const char *arg) const;
    };

} // namespace llaminar2
