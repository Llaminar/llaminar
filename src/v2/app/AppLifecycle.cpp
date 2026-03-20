/**
 * @file AppLifecycle.cpp
 * @brief Top-level application orchestrator implementation
 */

#include "app/AppLifecycle.h"
#include "utils/Logger.h"
#include "config/OrchestrationConfigParser.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "app/modes/InteractiveChatMode.h"
#include "app/modes/SingleShotChatMode.h"
#include "app/modes/BenchmarkMode.h"
#include "app/modes/ServerMode.h"
#include "app/modes/CompletionMode.h"
#include <iostream>
#include <vector>
#include <memory>

namespace llaminar2
{

    int AppLifecycle::run(int argc, char *argv[])
    {
        initializeLogging();

        OrchestrationConfigParser parser;
        OrchestrationConfig config = parser.parseArgs(argc, argv);

        // Handle early exits (no MPI needed)
        if (config.show_help)
        {
            std::cout << OrchestrationConfigParser::getHelpText() << std::endl;
            return 0;
        }

        if (config.list_devices)
        {
            MPIBootstrapPhase::listDevices();
            return 0;
        }

        // MPI Bootstrap: topology planning and self-launch if needed
        MPIBootstrapPhase bootstrap;
        auto bs_result = bootstrap.execute(config, argc, argv);
        if (bs_result.action == BootstrapResult::Action::EXIT)
            return bs_result.exit_code;

        // Runtime Initialization: MPI, DeviceManager, runner, tokenizer
        RuntimeInitPhase init;
        auto ctx_opt = init.execute(config, argc, argv);
        if (!ctx_opt)
            return 1;
        auto ctx = std::move(*ctx_opt);

        // Build execution mode chain (first match wins, CompletionMode is catch-all)
        std::vector<std::unique_ptr<IExecutionMode>> modes;
        modes.push_back(std::make_unique<InteractiveChatMode>());
        modes.push_back(std::make_unique<SingleShotChatMode>());
        modes.push_back(std::make_unique<BenchmarkMode>());
        modes.push_back(std::make_unique<ServerMode>());
        modes.push_back(std::make_unique<CompletionMode>());

        for (auto &mode : modes)
        {
            if (mode->matches(config))
            {
                return mode->execute(ctx);
            }
        }

        // Should never reach here (CompletionMode always matches)
        return 1;
    }

} // namespace llaminar2
