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

        // MPI Bootstrap
        MPIBootstrapPhase bootstrap;
        auto bs_result = bootstrap.execute(config, argc, argv);
        if (bs_result.action == BootstrapResult::Action::EXIT)
            return bs_result.exit_code;

        // Runtime Initialization
        RuntimeInitPhase init;
        auto ctx_opt = init.execute(config, argc, argv);
        if (!ctx_opt)
            return 1;
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
