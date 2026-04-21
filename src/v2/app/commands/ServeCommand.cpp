/**
 * @file ServeCommand.cpp
 * @brief 'llaminar serve' — parse flags, bootstrap, start HTTP server.
 *
 * Reuses OrchestrationConfigParser for all shared flags. Forces serve_mode=true
 * so ServerMode is always selected (no --serve flag needed on the CLI).
 */

#include "app/commands/ServeCommand.h"
#include "app/AppContext.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "app/Splash.h"
#include "app/modes/ServerMode.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/Logger.h"
#include <iostream>
#include <memory>

namespace llaminar2
{

    int ServeCommand::execute(int argc, char *argv[])
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

        // Force serve mode — 'llaminar serve' implies --serve
        config.serve_mode = true;

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

        // Run server directly
        ServerMode server;
        return server.execute(ctx);
    }

} // namespace llaminar2
