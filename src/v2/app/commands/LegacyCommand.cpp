/**
 * @file LegacyCommand.cpp
 * @brief Runs the original flat-flag AppLifecycle path.
 *
 * This is the exact body of the old AppLifecycle::run(), extracted so that
 * SubcommandRouter can delegate to it for backward compatibility.
 */

#include "app/commands/LegacyCommand.h"
#include "app/AppContext.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "app/Splash.h"
#include "app/modes/InteractiveChatMode.h"
#include "app/modes/SingleShotChatMode.h"
#include "app/modes/BenchmarkMode.h"
#include "app/modes/ServerMode.h"
#include "app/modes/CompletionMode.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/Logger.h"
#include "utils/MPIBootstrap.h"
#include <iostream>
#include <memory>
#include <vector>

namespace llaminar2
{

    int LegacyCommand::execute(int argc, char *argv[])
    {
        initializeLogging();
        printSplash();

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

        if (config.show_topology)
        {
            auto topo = MPIBootstrap::detectCPUTopology();
            std::cout << "\n=== CPU Topology ===\n"
                      << "  Detection method  : " << topo.detection_method << "\n"
                      << "  Sockets           : " << topo.num_sockets << "\n"
                      << "  Physical cores    : " << topo.physical_cores << "\n"
                      << "  Logical cores     : " << topo.logical_cores << "\n"
                      << "  Cores/socket      : " << topo.cores_per_socket << "\n"
                      << "  Threads/core      : " << topo.threads_per_core << "\n"
                      << "  NUMA nodes        : " << topo.numa_nodes << "\n"
                      << "  Hyperthreading    : " << (topo.hyperthreading ? "yes" : "no") << "\n"
                      << std::endl;
            return 0;
        }

        if (config.show_numa)
        {
            auto topo = MPIBootstrap::detectCPUTopology();
            std::cout << "\n=== NUMA Configuration ===\n"
                      << "  NUMA nodes        : " << topo.numa_nodes << "\n"
                      << "  Sockets           : " << topo.num_sockets << "\n"
                      << "  Cores per node    : " << (topo.physical_cores / std::max(1, topo.numa_nodes)) << "\n"
                      << std::endl;
            return 0;
        }

        if (config.validate_only)
        {
            std::cout << "Configuration is valid." << std::endl;
            if (config.explain_placement || config.verbose_level > 0)
            {
                std::cout << "\n"
                          << config.toString() << std::endl;
            }
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

        return 1;
    }

} // namespace llaminar2
