/**
 * @file BenchmarkCommand.cpp
 * @brief 'llaminar benchmark' — parse flags, bootstrap MPI, run benchmark.
 *
 * Uses the same heavy pipeline as OneshotCommand (MPI bootstrap → RuntimeInit
 * → AppContext) but forces benchmark_mode=true and skips irrelevant mode dispatch.
 */

#include "app/commands/BenchmarkCommand.h"
#include "app/AppContext.h"
#include "app/MPIBootstrapPhase.h"
#include "app/RuntimeInitPhase.h"
#include "app/Splash.h"
#include "app/modes/BenchmarkMode.h"
#include "config/OrchestrationConfigParser.h"
#include "utils/Logger.h"
#include <iostream>
#include <vector>

namespace llaminar2
{

    int BenchmarkCommand::execute(int argc, char *argv[])
    {
        initializeLogging();
        printSplash();

        // Parse all orchestration flags (device, TP, PP, model, etc.)
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

        // Force benchmark mode — this is the whole point of this subcommand
        config.benchmark_mode = true;

        if (config.model_path.empty())
        {
            std::cerr << "Error: Model path required (-m)\n\n"
                      << OrchestrationConfigParser::getHelpText() << std::endl;
            return 1;
        }

        // SubcommandRouter strips argv[1] ("benchmark") before calling us.
        // Re-inject it so MPIBootstrapPhase's selfLaunchMPI re-creates
        // the correct command line: llaminar2 benchmark <flags...>
        static const char *kSubcmd = "benchmark";
        std::vector<char *> full_argv;
        full_argv.push_back(argv[0]);
        full_argv.push_back(const_cast<char *>(kSubcmd));
        for (int i = 1; i < argc; ++i)
            full_argv.push_back(argv[i]);
        int full_argc = static_cast<int>(full_argv.size());

        // MPI Bootstrap (may exec into mpirun, replacing this process)
        MPIBootstrapPhase bootstrap;
        auto bs_result = bootstrap.execute(config, full_argc, full_argv.data());
        if (bs_result.action == BootstrapResult::Action::EXIT)
            return bs_result.exit_code;

        // Runtime Initialization (MPI_Init, model load, runner creation)
        RuntimeInitPhase init;
        auto ctx_opt = init.execute(config, argc, argv);
        if (!ctx_opt)
            return 1;
        auto ctx = std::move(*ctx_opt);

        // Run benchmark directly — no mode chain needed
        BenchmarkMode mode;
        return mode.execute(ctx);
    }

} // namespace llaminar2
