/**
 * @file Test__GPUCleanupFailureQualification.cpp
 * @brief Device-free guard against converting native teardown faults into passing tests.
 *
 * The five affected custom entrypoints must retain default fatal signal policy.
 * Native callbacks are not invoked here: source ownership proves those binaries
 * install no overriding signal handler, while real POSIX child signals replay
 * both the removed success conversion and the required fatal outcome. Normal
 * deliberate _exit(result) policy remains distinct from fault qualification.
 */
#include <gtest/gtest.h>
#include <array>
#include <csignal>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <sys/resource.h>
#include <unistd.h>

namespace
{
    /** @brief Original negative control: a teardown fault was reported as successful exit. */
    void originalCleanupSuccessHandler(int) { ::_exit(0); }

    /** @brief Keep deliberate signal probes from generating filesystem core payloads. */
    void disableCorePayload()
    {
        const rlimit limit{0, 0};
        if (setrlimit(RLIMIT_CORE, &limit) != 0) ::_exit(98);
    }
}

/** @test Actual canonical source ownership excludes every former signal interception path. */
TEST(GPUCleanupFailureQualification, CanonicalCustomMainsNeverInstallSuccessConvertingHandlers)
{
    constexpr std::array files{
        "integration/orchestrators/Test__RankOrchestrator_PP.cpp",
        "integration/execution/runner/Test__PPStageRunner.cpp",
        "integration/execution/runner/Test__Scenario7_MultiDomainPP.cpp",
        "integration/execution/runner/Test__OrchestrationRunner.cpp",
        "integration/collective/backends/Test__RCCLAllreduceAccuracy.cpp"};
    const std::regex installation(R"(\b(sigaction|signal)\s*\()");
    const std::regex comments(R"(/\*[\s\S]*?\*/|//[^\n]*)");
    for (const char *file : files)
    {
        SCOPED_TRACE(file);
        std::ifstream input(std::string(LLAMINAR_TEST_V2_SOURCE_ROOT) + '/' + file);
        ASSERT_TRUE(input.good());
        const std::string source{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        const auto code = std::regex_replace(source, comments, "");
        EXPECT_FALSE(std::regex_search(code, installation));
        for (const char *retired : {"cleanup_crash_handler", "sigabrt_handler", "install_crash_handlers",
                                    "AssertionTracker", "RCCLCleanupListener", "g_tests_passed"})
            EXPECT_EQ(code.find(retired), std::string::npos) << retired;
        EXPECT_NE(code.find("_exit(result)"), std::string::npos)
            << "Normal explicit test-result exit must remain separate from fatal cleanup signals";
    }
}

/** @test The removed control passes despite a fault; the installed default policy remains fatal. */
TEST(GPUCleanupFailureQualification, RealSignalsCannotQualifyAsSuccessfulCleanup)
{
    for (const int signal_number : {SIGABRT, SIGSEGV})
    {
        SCOPED_TRACE(signal_number);
        EXPECT_EXIT({
            disableCorePayload();
            std::signal(signal_number, originalCleanupSuccessHandler);
            std::raise(signal_number);
            ::_exit(99);
        }, ::testing::ExitedWithCode(0), "");
        EXPECT_EXIT({
            disableCorePayload();
            std::signal(signal_number, SIG_DFL);
            std::raise(signal_number);
            ::_exit(99);
        }, ::testing::KilledBySignal(signal_number), "");
    }
}
