/**
 * @file SubcommandRouter.cpp
 * @brief Routes argv[1] to a registered ICommand implementation.
 */

#include "app/SubcommandRouter.h"
#include "app/Splash.h"
#include "utils/Logger.h"
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace llaminar2
{

    SubcommandRouter &SubcommandRouter::add(std::unique_ptr<ICommand> cmd)
    {
        commands_.push_back(std::move(cmd));
        return *this;
    }

    SubcommandRouter &SubcommandRouter::setFallback(std::unique_ptr<ICommand> cmd)
    {
        fallback_ = std::move(cmd);
        return *this;
    }

    int SubcommandRouter::dispatch(int argc, char *argv[]) const
    {
        // No arguments, first arg is a flag, or bare "help" → print help
        if (argc < 2 || looksLikeFlag(argv[1]) ||
            std::strcmp(argv[1], "help") == 0)
        {
            printSplash();
            std::cout << getTopLevelHelp(argv[0]) << std::endl;
            return 0;
        }

        // Look up subcommand
        const char *token = argv[1];
        for (const auto &cmd : commands_)
        {
            if (std::strcmp(cmd->name(), token) == 0)
            {
                // Shift argv: keep argv[0] (binary name), skip argv[1] (subcommand)
                // Build a new argv: [argv[0], argv[2], argv[3], ...]
                std::vector<char *> shifted;
                shifted.push_back(argv[0]);
                for (int i = 2; i < argc; ++i)
                    shifted.push_back(argv[i]);

                return cmd->execute(static_cast<int>(shifted.size()), shifted.data());
            }
        }

        std::cerr << "Unknown subcommand: '" << token
                  << "'. Use --help for available commands.\n";
        return 1;
    }

    std::string SubcommandRouter::getTopLevelHelp(const char *binary_name) const
    {
        std::ostringstream os;
        os << "Usage: " << (binary_name ? binary_name : "llaminar2")
           << " <command> [options]\n\n"
           << "Commands:\n";

        // Find the longest command name for alignment
        size_t max_name = 0;
        for (const auto &cmd : commands_)
            max_name = std::max(max_name, std::strlen(cmd->name()));

        for (const auto &cmd : commands_)
        {
            os << "  " << std::left << std::setw(static_cast<int>(max_name + 2))
               << cmd->name() << cmd->description() << "\n";
        }

        os << "\nRun '" << (binary_name ? binary_name : "llaminar2")
           << " <command> --help' for command-specific options.";

        return os.str();
    }

    const ICommand *SubcommandRouter::find(const std::string &name) const
    {
        for (const auto &cmd : commands_)
        {
            if (cmd->name() == name)
                return cmd.get();
        }
        return nullptr;
    }

    bool SubcommandRouter::isHelpRequest(const char *arg) const
    {
        if (!arg)
            return false;
        return std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0;
    }

    bool SubcommandRouter::looksLikeFlag(const char *arg) const
    {
        return arg && arg[0] == '-';
    }

} // namespace llaminar2
