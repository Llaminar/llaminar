/**
 * @file Test__Commands.cpp
 * @brief Unit tests for the CLI subcommand implementations.
 *
 * Tests command metadata, DescribeCommand --help, and PlanCommand --help
 * and error handling. Does NOT test full inference pipelines (those require
 * models and MPI).
 */

#include <gtest/gtest.h>
#include "app/ICommand.h"
#include "app/SubcommandRouter.h"
#include "app/commands/DescribeCommand.h"
#include "app/commands/OneshotCommand.h"
#include "app/commands/ServeCommand.h"
#include "app/commands/PlanCommand.h"
#include "app/commands/LegacyCommand.h"
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    struct ArgvBuilder
    {
        std::vector<std::string> storage;
        std::vector<char *> ptrs;

        template <typename... Args>
        ArgvBuilder(Args... args) : storage{args...}
        {
            for (auto &s : storage)
                ptrs.push_back(s.data());
        }

        int argc() const { return static_cast<int>(ptrs.size()); }
        char **argv() { return ptrs.data(); }
    };
} // namespace

// ============================================================================
// Command metadata tests
// ============================================================================

TEST(Test__Commands, AllCommandsHaveCorrectNames)
{
    OneshotCommand oneshot;
    ServeCommand serve;
    PlanCommand plan;
    DescribeCommand describe;
    LegacyCommand legacy;

    EXPECT_STREQ(oneshot.name(), "oneshot");
    EXPECT_STREQ(serve.name(), "serve");
    EXPECT_STREQ(plan.name(), "plan");
    EXPECT_STREQ(describe.name(), "describe");
    EXPECT_STREQ(legacy.name(), "legacy");
}

TEST(Test__Commands, AllCommandsHaveDescriptions)
{
    OneshotCommand oneshot;
    ServeCommand serve;
    PlanCommand plan;
    DescribeCommand describe;

    // Descriptions should be non-empty
    EXPECT_GT(std::strlen(oneshot.description()), 0u);
    EXPECT_GT(std::strlen(serve.description()), 0u);
    EXPECT_GT(std::strlen(plan.description()), 0u);
    EXPECT_GT(std::strlen(describe.description()), 0u);
}

// ============================================================================
// PlanCommand tests
// ============================================================================

TEST(Test__Commands, PlanHelpReturns0)
{
    PlanCommand plan;
    ArgvBuilder args("llaminar2", "--help");
    EXPECT_EQ(plan.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, PlanShortHelpReturns0)
{
    PlanCommand plan;
    ArgvBuilder args("llaminar2", "-h");
    EXPECT_EQ(plan.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, PlanRequiresModel)
{
    PlanCommand plan;
    // No -m flag → should fail with exit code 1
    ArgvBuilder args("llaminar2");
    EXPECT_EQ(plan.execute(args.argc(), args.argv()), 1);
}

TEST(Test__Commands, PlanRejectsInvalidStrategy)
{
    PlanCommand plan;
    ArgvBuilder args("llaminar2", "-m", "model.gguf", "-s", "bogus");
    EXPECT_EQ(plan.execute(args.argc(), args.argv()), 1);
}

TEST(Test__Commands, PlanAcceptsValidStrategy)
{
    PlanCommand plan;
    ArgvBuilder args("llaminar2", "-m", "/tmp/nonexistent_model.gguf", "-s", "cpu-only");
    // Should succeed (prints plan to stdout; model doesn't need to exist for planning)
    EXPECT_EQ(plan.execute(args.argc(), args.argv()), 0);
}

// ============================================================================
// DescribeCommand tests
// ============================================================================

TEST(Test__Commands, DescribeHelpReturns0)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--help");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeNoArgsSucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2");
    // Default: prints topology, NUMA, and devices
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeTopologyOnlySucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--topology-only");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeRejectsInvalidFormat)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--format", "xml");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 1);
}

// ============================================================================
// Integration: full router with real commands
// ============================================================================

TEST(Test__Commands, RouterWithRealCommandsFindsAll)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());
    router.add(std::make_unique<ServeCommand>());
    router.add(std::make_unique<PlanCommand>());
    router.add(std::make_unique<DescribeCommand>());

    EXPECT_NE(router.find("oneshot"), nullptr);
    EXPECT_NE(router.find("serve"), nullptr);
    EXPECT_NE(router.find("plan"), nullptr);
    EXPECT_NE(router.find("describe"), nullptr);
    EXPECT_EQ(router.find("bogus"), nullptr);
    EXPECT_EQ(router.size(), 4u);
}

TEST(Test__Commands, RouterHelpTextIncludesRealCommands)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());
    router.add(std::make_unique<ServeCommand>());
    router.add(std::make_unique<PlanCommand>());
    router.add(std::make_unique<DescribeCommand>());

    std::string help = router.getTopLevelHelp("llaminar2");
    EXPECT_NE(help.find("oneshot"), std::string::npos);
    EXPECT_NE(help.find("serve"), std::string::npos);
    EXPECT_NE(help.find("plan"), std::string::npos);
    EXPECT_NE(help.find("describe"), std::string::npos);
}
