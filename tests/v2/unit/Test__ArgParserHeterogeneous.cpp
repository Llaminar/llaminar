/**
 * @file Test__ArgParserHeterogeneous.cpp
 * @brief Unit tests for heterogeneous multi-domain CLI argument parsing
 * @author David Sanftenberg
 * @date January 2026
 *
 * Tests Phase 6.5 heterogeneous parallelism CLI flags:
 * - --heterogeneous
 * - --cpu-fraction
 * - --no-gpu-tp / --no-cpu-tp
 * - --min-layers-per-domain
 */

#include <gtest/gtest.h>
#include "utils/ArgParser.h"

using namespace llaminar2;

// ============================================================================
// Helper to convert string array to argc/argv
// ============================================================================

class ArgvHelper
{
public:
    ArgvHelper(std::initializer_list<const char *> args)
    {
        for (const char *arg : args)
        {
            strings_.push_back(arg);
        }
        for (auto &s : strings_)
        {
            argv_.push_back(const_cast<char *>(s.c_str()));
        }
    }

    int argc() const { return static_cast<int>(argv_.size()); }
    char **argv() { return argv_.data(); }

private:
    std::vector<std::string> strings_;
    std::vector<char *> argv_;
};

// ============================================================================
// Default Values Tests
// ============================================================================

TEST(Test__ArgParserHeterogeneous, DefaultValuesWithoutFlags)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;

    // Heterogeneous mode defaults
    EXPECT_FALSE(ctx.heterogeneous_mode);
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.2f);
    EXPECT_FALSE(ctx.disable_gpu_tp);
    EXPECT_FALSE(ctx.disable_cpu_tp);
    EXPECT_EQ(ctx.min_layers_per_domain, 2);
}

// ============================================================================
// --heterogeneous Flag Tests
// ============================================================================

TEST(Test__ArgParserHeterogeneous, ParsesHeterogeneousFlag)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--heterogeneous"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.heterogeneous_mode);
}

TEST(Test__ArgParserHeterogeneous, HeterogeneousFlagWithOtherOptions)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--heterogeneous",
                    "-p", "Hello", "-n", "50", "--mpi-procs", "4"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.heterogeneous_mode);
    EXPECT_EQ(ctx.prompt, "Hello");
    EXPECT_EQ(ctx.n_predict, 50);
    EXPECT_EQ(ctx.mpi_procs, 4);
}

// ============================================================================
// --cpu-fraction Tests
// ============================================================================

TEST(Test__ArgParserHeterogeneous, ParsesCpuFractionWithSpace)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--cpu-fraction", "0.3"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.3f);
}

TEST(Test__ArgParserHeterogeneous, ParsesCpuFractionWithEquals)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--cpu-fraction=0.5"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.5f);
}

TEST(Test__ArgParserHeterogeneous, ParsesCpuFractionZero)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--cpu-fraction", "0.0"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.0f);
}

TEST(Test__ArgParserHeterogeneous, ParsesCpuFractionOne)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--cpu-fraction", "1.0"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 1.0f);
}

TEST(Test__ArgParserHeterogeneous, RejectsCpuFractionNegative)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--cpu-fraction", "-0.1"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("--cpu-fraction"), std::string::npos);
    EXPECT_NE(ctx.error.find("0.0"), std::string::npos);
}

TEST(Test__ArgParserHeterogeneous, RejectsCpuFractionGreaterThanOne)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--cpu-fraction", "1.5"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("--cpu-fraction"), std::string::npos);
    EXPECT_NE(ctx.error.find("1.0"), std::string::npos);
}

// ============================================================================
// --no-gpu-tp and --no-cpu-tp Tests
// ============================================================================

TEST(Test__ArgParserHeterogeneous, ParsesNoGpuTp)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--no-gpu-tp"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.disable_gpu_tp);
    EXPECT_FALSE(ctx.disable_cpu_tp);
}

TEST(Test__ArgParserHeterogeneous, ParsesNoCpuTp)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--no-cpu-tp"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_FALSE(ctx.disable_gpu_tp);
    EXPECT_TRUE(ctx.disable_cpu_tp);
}

TEST(Test__ArgParserHeterogeneous, RejectsBothTpDisabled)
{
    // Can't disable both GPU and CPU TP in heterogeneous mode
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--heterogeneous",
                    "--no-gpu-tp", "--no-cpu-tp"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("--no-gpu-tp"), std::string::npos);
    EXPECT_NE(ctx.error.find("--no-cpu-tp"), std::string::npos);
}

TEST(Test__ArgParserHeterogeneous, AllowsBothTpDisabledWithoutHeterogeneousMode)
{
    // When not in heterogeneous mode, this combination is allowed
    // (it's essentially ignored, but we don't block it)
    ArgvHelper args{"llaminar2", "-m", "model.gguf",
                    "--no-gpu-tp", "--no-cpu-tp"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    // Should be valid (flags are parsed, just no heterogeneous mode active)
    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.disable_gpu_tp);
    EXPECT_TRUE(ctx.disable_cpu_tp);
    EXPECT_FALSE(ctx.heterogeneous_mode);
}

// ============================================================================
// --min-layers-per-domain Tests
// ============================================================================

TEST(Test__ArgParserHeterogeneous, ParsesMinLayersPerDomainWithSpace)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--min-layers-per-domain", "4"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_EQ(ctx.min_layers_per_domain, 4);
}

TEST(Test__ArgParserHeterogeneous, ParsesMinLayersPerDomainWithEquals)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--min-layers-per-domain=6"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_EQ(ctx.min_layers_per_domain, 6);
}

TEST(Test__ArgParserHeterogeneous, ParsesMinLayersPerDomainOne)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--min-layers-per-domain", "1"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_EQ(ctx.min_layers_per_domain, 1);
}

TEST(Test__ArgParserHeterogeneous, RejectsMinLayersPerDomainZero)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--min-layers-per-domain", "0"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("--min-layers-per-domain"), std::string::npos);
    EXPECT_NE(ctx.error.find(">= 1"), std::string::npos);
}

TEST(Test__ArgParserHeterogeneous, RejectsMinLayersPerDomainNegative)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--min-layers-per-domain", "-2"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_FALSE(ctx.error.empty());
    EXPECT_NE(ctx.error.find("--min-layers-per-domain"), std::string::npos);
}

// ============================================================================
// Combined Configuration Tests
// ============================================================================

TEST(Test__ArgParserHeterogeneous, FullHeterogeneousConfiguration)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf",
                    "--heterogeneous",
                    "--cpu-fraction", "0.35",
                    "--min-layers-per-domain", "3",
                    "--no-cpu-tp"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.heterogeneous_mode);
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.35f);
    EXPECT_EQ(ctx.min_layers_per_domain, 3);
    EXPECT_FALSE(ctx.disable_gpu_tp);
    EXPECT_TRUE(ctx.disable_cpu_tp);
}

TEST(Test__ArgParserHeterogeneous, HeterogeneousWithMpiProcs)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf",
                    "--heterogeneous",
                    "--mpi-procs", "8",
                    "--cpu-fraction", "0.25"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.heterogeneous_mode);
    EXPECT_EQ(ctx.mpi_procs, 8);
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.25f);
}

TEST(Test__ArgParserHeterogeneous, HeterogeneousGpuOnlyConfig)
{
    // GPU-only heterogeneous (multi-GPU TP without CPU participation)
    ArgvHelper args{"llaminar2", "-m", "model.gguf",
                    "--heterogeneous",
                    "--no-cpu-tp",
                    "--cpu-fraction", "0.0"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.heterogeneous_mode);
    EXPECT_TRUE(ctx.disable_cpu_tp);
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.0f);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Test__ArgParserHeterogeneous, CpuFractionEdgeCaseBoundary)
{
    // Test boundary value just within range
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--cpu-fraction", "0.999"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_NEAR(ctx.cpu_compute_fraction, 0.999f, 0.001f);
}

TEST(Test__ArgParserHeterogeneous, MinLayersPerDomainLargeValue)
{
    ArgvHelper args{"llaminar2", "-m", "model.gguf", "--min-layers-per-domain", "100"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_EQ(ctx.min_layers_per_domain, 100);
}

TEST(Test__ArgParserHeterogeneous, OrderIndependence)
{
    // Flags can appear in any order
    ArgvHelper args{"llaminar2",
                    "--cpu-fraction", "0.4",
                    "--heterogeneous",
                    "-m", "model.gguf",
                    "--min-layers-per-domain", "5"};
    ArgContext ctx = ArgParser::parse(args.argc(), args.argv());

    EXPECT_TRUE(ctx.error.empty()) << "Parse error: " << ctx.error;
    EXPECT_TRUE(ctx.heterogeneous_mode);
    EXPECT_FLOAT_EQ(ctx.cpu_compute_fraction, 0.4f);
    EXPECT_EQ(ctx.min_layers_per_domain, 5);
}
