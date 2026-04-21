/**
 * @file AppLifecycle.cpp
 * @brief Top-level application orchestrator: subcommand routing + legacy fallback.
 */

#include "app/AppLifecycle.h"
#include "app/SubcommandRouter.h"
#include "app/commands/LegacyCommand.h"
#include "app/commands/DescribeCommand.h"
#include "app/commands/OneshotCommand.h"
#include "app/commands/ServeCommand.h"
#include "app/commands/PlanCommand.h"

namespace llaminar2
{

    int AppLifecycle::run(int argc, char *argv[])
    {
        SubcommandRouter router;

        // Register subcommands
        router.add(std::make_unique<OneshotCommand>());
        router.add(std::make_unique<ServeCommand>());
        router.add(std::make_unique<PlanCommand>());
        router.add(std::make_unique<DescribeCommand>());

        // Legacy fallback: full flat-flag path (backward compat)
        router.setFallback(std::make_unique<LegacyCommand>());

        return router.dispatch(argc, argv);
    }

} // namespace llaminar2
