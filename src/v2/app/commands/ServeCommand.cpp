/**
 * @file ServeCommand.cpp
 * @brief 'llaminar serve' — parse flags, bootstrap, start HTTP server.
 *
 * Reuses OrchestrationConfigParser for all shared flags. Forces serve_mode=true
 * so ServerMode is always selected (no --serve flag needed on the CLI).
 */

#include "app/commands/ServeCommand.h"
#include "app/AppContext.h"
#include "app/commands/CommandValidation.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "app/Splash.h"
#include "app/modes/ServerMode.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/Logger.h"
#include <iostream>
#include <memory>
#include <vector>

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

        if (!command_validation::printConfigErrors(config))
        {
            return 1;
        }

        if (config.validate_only)
        {
            command_validation::printValidateOnlySuccess(config);
            return 0;
        }

        // SubcommandRouter strips argv[1] ("serve") before calling us.
        // Re-inject it so MPIBootstrapPhase's selfLaunchMPI re-creates
        // the correct command line: llaminar2 serve <flags...>
        static const char *kSubcmd = "serve";
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

        // Run server directly
        ServerMode server;
        return server.execute(ctx);
    }

} // namespace llaminar2
