#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>

namespace
{
    namespace fs = std::filesystem;

    std::string shellQuote(const std::string &value)
    {
        std::string quoted = "'";
        for (const char ch : value)
        {
            if (ch == '\'')
                quoted += "'\\''";
            else
                quoted.push_back(ch);
        }
        quoted.push_back('\'');
        return quoted;
    }

    std::string readFile(const fs::path &path)
    {
        std::ifstream input(path);
        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }

    bool contains(const std::string &text, const std::string &needle)
    {
        return text.find(needle) != std::string::npos;
    }

    struct CommandResult
    {
        int exit_code = -1;
        std::string output;
    };

    CommandResult runCommand(const std::string &command)
    {
        CommandResult result;
        FILE *pipe = popen(command.c_str(), "r");
        if (pipe == nullptr)
        {
            result.output = "popen failed";
            return result;
        }

        char buffer[512];
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
            result.output += buffer;

        const int status = pclose(pipe);
        if (WIFEXITED(status))
            result.exit_code = WEXITSTATUS(status);
        else
            result.exit_code = -1;
        return result;
    }
} // namespace

TEST(Perf__MoEGraphNativeOverlayAnalysis, SyntheticFixtureProducesConservativeReport)
{
    const fs::path script_path("scripts/analyze_moe_overlay_benchmarks.py");
    const fs::path fixture_dir("tests/v2/performance/moe/fixtures/moe_overlay_sample_run");
    ASSERT_TRUE(fs::exists(script_path)) << "missing analyzer script: " << script_path;
    ASSERT_TRUE(fs::is_directory(fixture_dir)) << "missing synthetic fixture: " << fixture_dir;

    const fs::path output_path = fs::temp_directory_path() / "moe_overlay_analysis_gate.md";
    const std::string command = "python3 " + shellQuote(script_path.string()) + " " +
                                shellQuote(fixture_dir.string()) + " --output " +
                                shellQuote(output_path.string()) + " 2>&1";

    const CommandResult result = runCommand(command);
    ASSERT_EQ(result.exit_code, 0) << result.output;
    ASSERT_TRUE(fs::exists(output_path)) << "analyzer did not create output file";

    const std::string report = readFile(output_path);
    ASSERT_FALSE(report.empty());
    EXPECT_TRUE(contains(report, "# MoE Graph-Native Overlay Benchmark Analysis"));
    EXPECT_TRUE(contains(report, "## All-GPU Configs"));
    EXPECT_TRUE(contains(report, "## Mixed GPU/CPU Configs"));
    EXPECT_TRUE(contains(report, "CPU fallback rows"));
    EXPECT_TRUE(contains(report, "Dense bytes avoided"));
    EXPECT_TRUE(contains(report, "## Expected Observations"));
    EXPECT_TRUE(contains(report, "All-GPU vs mixed timing: observed"));
    EXPECT_TRUE(contains(report, "CPU fallback row correlation: observed"));
    EXPECT_TRUE(contains(report, "GPU budget throughput trend: insufficient data"));
    EXPECT_TRUE(contains(report, "Prefill/decode kernel shape: insufficient data"));
    EXPECT_TRUE(contains(report, "Sparse transport bottleneck check: diagnostic"));
    EXPECT_TRUE(contains(report, "missing moe_overlay_profile.csv"));
}