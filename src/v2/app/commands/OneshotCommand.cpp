/**
 * @file OneshotCommand.cpp
 * @brief 'llaminar oneshot' — parse flags, bootstrap, run single inference.
 *
 * Reuses OrchestrationConfigParser for flag parsing (all inference/sampling/
 * device/parallelism flags are available). After parsing, dispatches to the
 * appropriate IExecutionMode.
 */

#include "app/commands/OneshotCommand.h"
#include "app/AppContext.h"
#include "app/commands/CommandValidation.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "app/Splash.h"
#include "app/modes/InteractiveChatMode.h"
#include "app/modes/SingleShotChatMode.h"
#include "app/modes/CompletionMode.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/Logger.h"
#include <iostream>
#include <memory>
#include <vector>

namespace llaminar2
{

    int OneshotCommand::execute(int argc, char *argv[])
    {
        initializeLogging();
        printSplash();

        OrchestrationConfigParser parser;
        OrchestrationConfig config;
        try
        {
            config = parser.parseArgs(argc, argv);
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }

        if (config.show_help)
        {
            std::cout << OrchestrationConfigParser::getHelpText() << std::endl;
            return 0;
        }

        // Reject serve mode under oneshot
        if (config.serve_mode)
        {
            std::cerr << "Error: --serve is not valid with 'oneshot'. Use 'llaminar2 serve' instead.\n";
            return 1;
        }

        // Reject benchmark mode under oneshot — it has its own subcommand
        if (config.benchmark_mode)
        {
            std::cerr << "Error: --benchmark is not valid with 'oneshot'. Use 'llaminar2 benchmark' instead.\n";
            return 1;
        }

        if (!command_validation::printConfigErrors(config))
        {
            return 1;
        }

        if (config.validate_only)
        {
            command_validation::printValidateOnlySuccess(config);
            return 0;
        }

        // SubcommandRouter strips argv[1] ("oneshot") before calling us.
        // Re-inject it so MPIBootstrapPhase's selfLaunchMPI re-creates
        // the correct command line: llaminar2 oneshot <flags...>
        static const char *kSubcmd = "oneshot";
        std::vector<char *> full_argv;
        full_argv.push_back(argv[0]);
        full_argv.push_back(const_cast<char *>(kSubcmd));
        for (int i = 1; i < argc; ++i)
            full_argv.push_back(argv[i]);
        int full_argc = static_cast<int>(full_argv.size());

        // MPI Bootstrap
        MPIBootstrapPhase bootstrap;
        auto bs_result = bootstrap.execute(config, full_argc, full_argv.data());
        if (bs_result.action == BootstrapResult::Action::EXIT)
            return bs_result.exit_code;

        // Runtime Initialization
        RuntimeInitPhase init;
        auto ctx_opt = init.execute(config, argc, argv);
        if (!ctx_opt)
            return config.dry_run ? 0 : 1;
        auto ctx = std::move(*ctx_opt);

        // Dispatch: mode chain (no BenchmarkMode — use 'llaminar2 benchmark')
        std::vector<std::unique_ptr<IExecutionMode>> modes;
        modes.push_back(std::make_unique<InteractiveChatMode>());
        modes.push_back(std::make_unique<SingleShotChatMode>());
        modes.push_back(std::make_unique<CompletionMode>()); // catch-all

        for (auto &mode : modes)
        {
            if (mode->matches(config))
                return mode->execute(ctx);
        }

        return 1;
    }

} // namespace llaminar2
