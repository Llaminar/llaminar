/**
 * @file AppLifecycle.cpp
 * @brief Top-level application orchestrator: subcommand routing + legacy fallback.
 * The executable owns the GPU process scope; dispatch validates that boundary
 * without creating a second owner or deferring native retirement to static exit.
 */

#include "app/AppLifecycle.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include "app/SubcommandRouter.h"
#include "app/commands/BenchmarkCommand.h"
#include "app/commands/DescribeCommand.h"
#include "app/commands/OneshotCommand.h"
#include "app/commands/ServeCommand.h"
#include "app/commands/PlanCommand.h"

namespace llaminar2
{

    int AppLifecycle::run(int argc, char *argv[])
    {
        GPUDeviceProcessLifetime::requireActive();
        SubcommandRouter router;

        // Register subcommands
        router.add(std::make_unique<OneshotCommand>());
        router.add(std::make_unique<ServeCommand>());
        router.add(std::make_unique<BenchmarkCommand>());
        router.add(std::make_unique<PlanCommand>());
        router.add(std::make_unique<DescribeCommand>());

        return router.dispatch(argc, argv);
    }

} // namespace llaminar2
