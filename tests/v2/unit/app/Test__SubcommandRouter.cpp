/**
 * @file Test__SubcommandRouter.cpp
 * @brief Unit tests for SubcommandRouter dispatch logic.
 *
 * Uses stub ICommand implementations to verify routing, fallback,
 * help generation, and argv shifting without pulling in production deps.
 */

#include <gtest/gtest.h>
#include "app/SubcommandRouter.h"
#include "app/ICommand.h"
#include <cstring>
#include <string>
#include <vector>

using namespace llaminar2;

// ============================================================================
// Test helpers
// ============================================================================

namespace
{
    /// Stub command that records its invocation for verification.
    class StubCommand : public ICommand
    {
    public:
        explicit StubCommand(const char *name, const char *desc = "stub")
            : name_(name), desc_(desc) {}

        const char *name() const override { return name_; }
        const char *description() const override { return desc_; }

        int execute(int argc, char *argv[]) override
        {
            executed = true;
            received_argc = argc;
            received_args.clear();
            for (int i = 0; i < argc; ++i)
                received_args.emplace_back(argv[i]);
            return exit_code;
        }

        bool executed = false;
        int received_argc = 0;
        std::vector<std::string> received_args;
        int exit_code = 0;

    private:
        const char *name_;
        const char *desc_;
    };

    /// Helper to build a modifiable argv from string literals.
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
// Routing tests
// ============================================================================

TEST(Test__SubcommandRouter, DispatchesToRegisteredCommand)
{
    SubcommandRouter router;
    auto *cmd = new StubCommand("serve");
    router.add(std::unique_ptr<ICommand>(cmd));
    router.setFallback(std::make_unique<StubCommand>("legacy"));

    ArgvBuilder args("llaminar2", "serve", "--port", "8080");
    int rc = router.dispatch(args.argc(), args.argv());

    EXPECT_TRUE(cmd->executed);
    EXPECT_EQ(rc, 0);
    // argv should be shifted: [llaminar2, --port, 8080]
    ASSERT_EQ(cmd->received_argc, 3);
    EXPECT_EQ(cmd->received_args[0], "llaminar2");
    EXPECT_EQ(cmd->received_args[1], "--port");
    EXPECT_EQ(cmd->received_args[2], "8080");
}

TEST(Test__SubcommandRouter, DispatchesToCorrectCommandAmongMany)
{
    SubcommandRouter router;
    auto *plan = new StubCommand("plan");
    auto *serve = new StubCommand("serve");
    auto *describe = new StubCommand("describe");
    router.add(std::unique_ptr<ICommand>(plan));
    router.add(std::unique_ptr<ICommand>(serve));
    router.add(std::unique_ptr<ICommand>(describe));

    ArgvBuilder args("llaminar2", "describe");
    router.dispatch(args.argc(), args.argv());

    EXPECT_FALSE(plan->executed);
    EXPECT_FALSE(serve->executed);
    EXPECT_TRUE(describe->executed);
}

TEST(Test__SubcommandRouter, PropagatesExitCode)
{
    SubcommandRouter router;
    auto *cmd = new StubCommand("fail");
    cmd->exit_code = 42;
    router.add(std::unique_ptr<ICommand>(cmd));

    ArgvBuilder args("llaminar2", "fail");
    int rc = router.dispatch(args.argc(), args.argv());
    EXPECT_EQ(rc, 42);
}

// ============================================================================
// Legacy fallback tests
// ============================================================================

TEST(Test__SubcommandRouter, FallsBackWhenNoSubcommandGiven)
{
    SubcommandRouter router;
    auto *legacy = new StubCommand("legacy");
    router.add(std::make_unique<StubCommand>("serve"));
    router.setFallback(std::unique_ptr<ICommand>(legacy));

    // No argv[1] at all
    ArgvBuilder args("llaminar2");
    router.dispatch(args.argc(), args.argv());

    EXPECT_TRUE(legacy->executed);
    // Fallback receives full original argv
    EXPECT_EQ(legacy->received_argc, 1);
    EXPECT_EQ(legacy->received_args[0], "llaminar2");
}

TEST(Test__SubcommandRouter, FallsBackWhenArgv1IsFlag)
{
    SubcommandRouter router;
    auto *legacy = new StubCommand("legacy");
    router.add(std::make_unique<StubCommand>("serve"));
    router.setFallback(std::unique_ptr<ICommand>(legacy));

    ArgvBuilder args("llaminar2", "-m", "model.gguf", "-p", "Hello");
    router.dispatch(args.argc(), args.argv());

    EXPECT_TRUE(legacy->executed);
    EXPECT_EQ(legacy->received_argc, 5);
}

TEST(Test__SubcommandRouter, FallsBackWhenUnknownSubcommand)
{
    SubcommandRouter router;
    auto *legacy = new StubCommand("legacy");
    router.add(std::make_unique<StubCommand>("serve"));
    router.setFallback(std::unique_ptr<ICommand>(legacy));

    ArgvBuilder args("llaminar2", "bogus_word");
    router.dispatch(args.argc(), args.argv());

    EXPECT_TRUE(legacy->executed);
    EXPECT_EQ(legacy->received_argc, 2);
}

// ============================================================================
// Help tests
// ============================================================================

TEST(Test__SubcommandRouter, TopLevelHelpPrintsAndReturns0)
{
    SubcommandRouter router;
    auto *legacy = new StubCommand("legacy");
    router.add(std::make_unique<StubCommand>("serve", "Start HTTP server"));
    router.add(std::make_unique<StubCommand>("plan", "Create execution plan"));
    router.setFallback(std::unique_ptr<ICommand>(legacy));

    ArgvBuilder args("llaminar2", "--help");
    int rc = router.dispatch(args.argc(), args.argv());

    EXPECT_EQ(rc, 0);
    EXPECT_FALSE(legacy->executed); // help handled by router, not fallback
}

TEST(Test__SubcommandRouter, ShortHelpAlsoWorks)
{
    SubcommandRouter router;
    auto *legacy = new StubCommand("legacy");
    router.setFallback(std::unique_ptr<ICommand>(legacy));

    ArgvBuilder args("llaminar2", "-h");
    int rc = router.dispatch(args.argc(), args.argv());
    EXPECT_EQ(rc, 0);
    EXPECT_FALSE(legacy->executed);
}

TEST(Test__SubcommandRouter, HelpTextContainsAllCommands)
{
    SubcommandRouter router;
    router.add(std::make_unique<StubCommand>("serve", "Start HTTP server"));
    router.add(std::make_unique<StubCommand>("plan", "Create execution plan"));
    router.add(std::make_unique<StubCommand>("describe", "Show inventory"));

    std::string help = router.getTopLevelHelp("llaminar2");

    EXPECT_NE(help.find("serve"), std::string::npos);
    EXPECT_NE(help.find("plan"), std::string::npos);
    EXPECT_NE(help.find("describe"), std::string::npos);
    EXPECT_NE(help.find("Start HTTP server"), std::string::npos);
    EXPECT_NE(help.find("llaminar2"), std::string::npos);
}

// ============================================================================
// Argv shifting tests
// ============================================================================

TEST(Test__SubcommandRouter, ShiftsArgvCorrectly)
{
    SubcommandRouter router;
    auto *cmd = new StubCommand("oneshot");
    router.add(std::unique_ptr<ICommand>(cmd));

    ArgvBuilder args("llaminar2", "oneshot", "-m", "model.gguf", "-p", "Hello world");
    router.dispatch(args.argc(), args.argv());

    ASSERT_EQ(cmd->received_argc, 5);
    EXPECT_EQ(cmd->received_args[0], "llaminar2");  // binary name preserved
    EXPECT_EQ(cmd->received_args[1], "-m");
    EXPECT_EQ(cmd->received_args[2], "model.gguf");
    EXPECT_EQ(cmd->received_args[3], "-p");
    EXPECT_EQ(cmd->received_args[4], "Hello world");
}

TEST(Test__SubcommandRouter, SubcommandWithNoAdditionalArgs)
{
    SubcommandRouter router;
    auto *cmd = new StubCommand("describe");
    router.add(std::unique_ptr<ICommand>(cmd));

    ArgvBuilder args("llaminar2", "describe");
    router.dispatch(args.argc(), args.argv());

    ASSERT_EQ(cmd->received_argc, 1);
    EXPECT_EQ(cmd->received_args[0], "llaminar2");
}

// ============================================================================
// Registration API tests
// ============================================================================

TEST(Test__SubcommandRouter, FindReturnsRegisteredCommand)
{
    SubcommandRouter router;
    router.add(std::make_unique<StubCommand>("serve"));
    router.add(std::make_unique<StubCommand>("plan"));

    EXPECT_NE(router.find("serve"), nullptr);
    EXPECT_NE(router.find("plan"), nullptr);
    EXPECT_EQ(router.find("bogus"), nullptr);
}

TEST(Test__SubcommandRouter, SizeTracksRegisteredCommands)
{
    SubcommandRouter router;
    EXPECT_EQ(router.size(), 0u);

    router.add(std::make_unique<StubCommand>("a"));
    EXPECT_EQ(router.size(), 1u);

    router.add(std::make_unique<StubCommand>("b"));
    EXPECT_EQ(router.size(), 2u);
}

TEST(Test__SubcommandRouter, NoFallbackReturns1)
{
    SubcommandRouter router;
    router.add(std::make_unique<StubCommand>("serve"));
    // No fallback set

    ArgvBuilder args("llaminar2", "-m", "model.gguf");
    int rc = router.dispatch(args.argc(), args.argv());
    EXPECT_EQ(rc, 1);
}

TEST(Test__SubcommandRouter, UnknownWithNoFallbackReturns1)
{
    SubcommandRouter router;
    router.add(std::make_unique<StubCommand>("serve"));

    ArgvBuilder args("llaminar2", "bogus");
    int rc = router.dispatch(args.argc(), args.argv());
    EXPECT_EQ(rc, 1);
}
