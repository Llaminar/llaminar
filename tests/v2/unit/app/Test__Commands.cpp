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
#include "app/commands/BenchmarkCommand.h"
#include <cstring>
#include <cstdio>
#include <fstream>
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
    BenchmarkCommand benchmark;

    EXPECT_STREQ(oneshot.name(), "oneshot");
    EXPECT_STREQ(serve.name(), "serve");
    EXPECT_STREQ(plan.name(), "plan");
    EXPECT_STREQ(describe.name(), "describe");
    EXPECT_STREQ(benchmark.name(), "benchmark");
}

TEST(Test__Commands, AllCommandsHaveDescriptions)
{
    OneshotCommand oneshot;
    ServeCommand serve;
    PlanCommand plan;
    DescribeCommand describe;
    BenchmarkCommand benchmark;

    // Descriptions should be non-empty
    EXPECT_GT(std::strlen(oneshot.description()), 0u);
    EXPECT_GT(std::strlen(serve.description()), 0u);
    EXPECT_GT(std::strlen(plan.description()), 0u);
    EXPECT_GT(std::strlen(describe.description()), 0u);
    EXPECT_GT(std::strlen(benchmark.description()), 0u);
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
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "-m", "/opt/llaminar-models/qwen2.5-0.5b-instruct-q4_0.gguf", "-s", "cpu-only");
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
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap");
    // Default: prints topology, NUMA, and devices
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeTopologyOnlySucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "--topology-only");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeRejectsInvalidFormat)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--format", "xml");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 1);
}

TEST(Test__Commands, DescribeJsonFormatSucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "--format", "json");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeYamlFormatSucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "--format", "yaml");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeDevicesOnlySucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "--devices-only");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeNumaOnlySucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "--numa-only");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, DescribeFileOutputSucceeds)
{
    DescribeCommand describe;
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "--format", "json", "-o", "/tmp/test_describe_output.json");
    EXPECT_EQ(describe.execute(args.argc(), args.argv()), 0);

    // Verify file was created and contains valid JSON
    std::ifstream f("/tmp/test_describe_output.json");
    ASSERT_TRUE(f.is_open());
    std::string content((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());
    EXPECT_FALSE(content.empty());
    EXPECT_NE(content.find("\"nodes\""), std::string::npos);
    EXPECT_NE(content.find("\"summary\""), std::string::npos);
    std::remove("/tmp/test_describe_output.json");
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
    router.add(std::make_unique<BenchmarkCommand>());

    EXPECT_NE(router.find("oneshot"), nullptr);
    EXPECT_NE(router.find("serve"), nullptr);
    EXPECT_NE(router.find("plan"), nullptr);
    EXPECT_NE(router.find("describe"), nullptr);
    EXPECT_NE(router.find("benchmark"), nullptr);
    EXPECT_EQ(router.find("bogus"), nullptr);
    EXPECT_EQ(router.size(), 5u);
}

TEST(Test__Commands, RouterHelpTextIncludesRealCommands)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());
    router.add(std::make_unique<ServeCommand>());
    router.add(std::make_unique<PlanCommand>());
    router.add(std::make_unique<DescribeCommand>());
    router.add(std::make_unique<BenchmarkCommand>());

    std::string help = router.getTopLevelHelp("llaminar2");
    EXPECT_NE(help.find("oneshot"), std::string::npos);
    EXPECT_NE(help.find("serve"), std::string::npos);
    EXPECT_NE(help.find("plan"), std::string::npos);
    EXPECT_NE(help.find("describe"), std::string::npos);
    EXPECT_NE(help.find("benchmark"), std::string::npos);
}

TEST(Test__Commands, RouterHelpTextDoesNotMentionLegacy)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());

    std::string help = router.getTopLevelHelp("llaminar2");
    EXPECT_EQ(help.find("Legacy"), std::string::npos);
    EXPECT_EQ(help.find("legacy"), std::string::npos);
}

TEST(Test__Commands, RouterNoArgsPrintsHelpReturns0)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());

    // Just the binary name — should print help and return 0
    ArgvBuilder args("llaminar2");
    EXPECT_EQ(router.dispatch(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, RouterBareFlags_PrintsHelpReturns0)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());

    // Legacy-style invocation (flags without subcommand) should print help
    ArgvBuilder args("llaminar2", "-m", "model.gguf", "-p", "Hello");
    EXPECT_EQ(router.dispatch(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, RouterUnknownSubcommand_Returns1)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());

    ArgvBuilder args("llaminar2", "bogus");
    EXPECT_EQ(router.dispatch(args.argc(), args.argv()), 1);
}

TEST(Test__Commands, RouterBareHelp_PrintsHelpReturns0)
{
    SubcommandRouter router;
    router.add(std::make_unique<OneshotCommand>());

    // "help" (no dash) should print help, not error
    ArgvBuilder args("llaminar2", "help");
    EXPECT_EQ(router.dispatch(args.argc(), args.argv()), 0);
}

// ============================================================================
// BenchmarkCommand tests
// ============================================================================

TEST(Test__Commands, BenchmarkHelpReturns0)
{
    BenchmarkCommand benchmark;
    ArgvBuilder args("llaminar2", "--help");
    EXPECT_EQ(benchmark.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, BenchmarkShortHelpReturns0)
{
    BenchmarkCommand benchmark;
    ArgvBuilder args("llaminar2", "-h");
    EXPECT_EQ(benchmark.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, BenchmarkRequiresModel)
{
    BenchmarkCommand benchmark;
    // No -m flag → should fail with exit code 1
    ArgvBuilder args("llaminar2");
    EXPECT_EQ(benchmark.execute(args.argc(), args.argv()), 1);
}

// ============================================================================
// OneshotCommand --benchmark rejection
// ============================================================================

TEST(Test__Commands, OneshotRejectsBenchmarkFlag)
{
    OneshotCommand oneshot;
    // --benchmark under oneshot should be rejected with exit code 1
    ArgvBuilder args("llaminar2", "--no-mpi-bootstrap", "--benchmark",
                     "-m", "/opt/llaminar-models/qwen2.5-0.5b-instruct-q4_0.gguf",
                     "-p", "Hello");
    EXPECT_EQ(oneshot.execute(args.argc(), args.argv()), 1);
}

TEST(Test__Commands, OneshotRejectsTensorParallelMoEBeforeRuntime)
{
    OneshotCommand oneshot;
    ArgvBuilder args("llaminar2", "--moe-expert-mode", "tensor-parallel",
                     "-m", "/tmp/does-not-need-to-exist.gguf", "-p", "test");
    EXPECT_EQ(oneshot.execute(args.argc(), args.argv()), 1);
}

TEST(Test__Commands, OneshotValidateOnlyReturns0BeforeRuntime)
{
    OneshotCommand oneshot;
    ArgvBuilder args("llaminar2", "--validate-only", "--moe-expert-mode", "expert-parallel");
    EXPECT_EQ(oneshot.execute(args.argc(), args.argv()), 0);
}

TEST(Test__Commands, ServeRejectsTensorParallelMoEBeforeRuntime)
{
    ServeCommand serve;
    ArgvBuilder args("llaminar2", "--moe-expert-mode", "tensor-parallel",
                     "-m", "/tmp/does-not-need-to-exist.gguf");
    EXPECT_EQ(serve.execute(args.argc(), args.argv()), 1);
}
