#include "config/OrchestrationConfigParser.h"
#include "execution/moe/MoEExpertParallelPlan.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{
    namespace fs = std::filesystem;

    constexpr int kQwen35RoutedExpertCount = 256;

    struct ConfigExpectation
    {
        std::string file_name;
        bool mixed_gpu_cpu = false;
        bool rebalanced = false;
        int expected_mpi_ranks = 2;
    };

    const std::vector<ConfigExpectation> &requiredConfigs()
    {
        static const std::vector<ConfigExpectation> configs = {
            {"rocm2_replicated_static.yaml", false, false, 1},
            {"rocm2_cpu2_replicated_static.yaml", true, false, 2},
            {"cuda_hot_rocm_warm_static.yaml", false, false, 2},
            {"cuda_hot_rocm_warm_rebalanced.yaml", false, true, 2},
            {"cuda_hot_rocm_warm_cpu_cold_static.yaml", true, false, 3},
            {"cuda_hot_rocm_warm_cpu_cold_rebalanced.yaml", true, true, 3},
            {"all_gpu_capacity_low.yaml", false, true, 2},
            {"all_gpu_capacity_medium.yaml", false, true, 2},
            {"all_gpu_capacity_high.yaml", false, true, 2},
            {"all_gpu_capacity_all_fit.yaml", false, true, 2},
            {"mixed_gpu_cpu_hot_cache_low.yaml", true, true, 3},
            {"mixed_gpu_cpu_hot_cache_medium.yaml", true, true, 3},
            {"mixed_gpu_cpu_hot_cache_high.yaml", true, true, 3},
        };
        return configs;
    }

    std::string readTextFile(const fs::path &path)
    {
        std::ifstream stream(path);
        if (!stream)
            return {};

        std::ostringstream buffer;
        buffer << stream.rdbuf();
        return buffer.str();
    }

    bool containsText(const std::string &text, const std::string &needle)
    {
        return text.find(needle) != std::string::npos;
    }

    bool isTruthyEnv(const char *value)
    {
        if (!value)
            return false;

        const std::string normalized(value);
        return normalized == "1" || normalized == "true" || normalized == "TRUE" ||
               normalized == "yes" || normalized == "YES" || normalized == "on" ||
               normalized == "ON";
    }

    std::string joinErrors(const std::vector<std::string> &errors)
    {
        std::ostringstream output;
        for (const auto &error_message : errors)
        {
            output << "\n - " << error_message;
        }
        return output.str();
    }

    std::vector<std::string> forbiddenConfigTokens()
    {
        return {
            std::string("compute=") + std::string("tensor_parallel_") + "experts",
            std::string("tensor_parallel_") + "experts",
            std::string("MoEOverlay") + "DomainRuntime",
            std::string("CPUFallback") + "ParticipantRunner",
            std::string("Local") + "TPStage",
            std::string("LLAMINAR_MOE_LEGACY_OVERLAY_") + "DOMAIN_RUNTIME",
        };
    }

    int countBenchmarkMetadataRanks(const std::string &text)
    {
        const std::string marker = "mpi_ranks:";
        const size_t marker_position = text.find(marker);
        if (marker_position == std::string::npos)
            return -1;

        const size_t value_start = marker_position + marker.size();
        std::string digits;
        for (size_t char_index = value_start; char_index < text.size(); ++char_index)
        {
            const char current_char = text[char_index];
            if (current_char >= '0' && current_char <= '9')
            {
                digits.push_back(current_char);
            }
            else if (!digits.empty())
            {
                break;
            }
        }

        if (digits.empty())
            return -1;
        return std::stoi(digits);
    }

    std::string commandForConfig(const ConfigExpectation &expectation, const fs::path &config_path, const std::string &model_path)
    {
        std::ostringstream command;
        command << "LLAMINAR_PROFILING=1 ";
        command << "LLAMINAR_MOE_EP_PROFILE_CSV=benchmark_results/moe_overlay/<run>/";
        command << config_path.stem().string() << "/moe_overlay_profile.csv ";
        command << "build_v2_release/llaminar2 benchmark ";
        command << "--config " << config_path.string() << ' ';
        command << "--mpi-procs " << expectation.expected_mpi_ranks << ' ';
        command << "-m " << model_path;
        return command.str();
    }
} // namespace

TEST(Perf__MoEGraphNativeOverlayHarness, ConfigsAreGraphNativeBenchmarkReady)
{
    const fs::path config_dir("configs/moe_overlay");
    ASSERT_TRUE(fs::is_directory(config_dir)) << "missing config directory: " << config_dir;

    llaminar2::OrchestrationConfigParser parser;

    for (const auto &expectation : requiredConfigs())
    {
        const fs::path config_path = config_dir / expectation.file_name;
        SCOPED_TRACE(config_path.string());

        ASSERT_TRUE(fs::exists(config_path)) << "required benchmark config is missing";
        const std::string text = readTextFile(config_path);
        ASSERT_FALSE(text.empty()) << "config file is empty or unreadable";

        EXPECT_TRUE(containsText(text, "compute=replicated_experts"));
        for (const auto &forbidden_token : forbiddenConfigTokens())
        {
            EXPECT_FALSE(containsText(text, forbidden_token)) << forbidden_token;
        }

        EXPECT_TRUE(containsText(text, "benchmark_results/moe_overlay"));
        EXPECT_TRUE(containsText(text, "LLAMINAR_MOE_OVERLAY_MODEL"));
        EXPECT_TRUE(containsText(text, "LLAMINAR_PROFILING=1"));
        EXPECT_TRUE(containsText(text, "LLAMINAR_MOE_EP_PROFILE_CSV"));
        EXPECT_TRUE(containsText(text, "benchmark: topology="));
        EXPECT_TRUE(containsText(text, "benchmark: model_requirement="));
        EXPECT_TRUE(containsText(text, "benchmark: policy="));
        EXPECT_EQ(countBenchmarkMetadataRanks(text), expectation.expected_mpi_ranks);

        auto orchestration_config = parser.parseYamlFile(config_path.string());
        ASSERT_TRUE(orchestration_config.moe_expert_parallel_plan)
            << "config did not produce a MoE expert-parallel plan";

        const auto &plan = *orchestration_config.moe_expert_parallel_plan;
        EXPECT_TRUE(plan.enabled);
        EXPECT_EQ(plan.execution_kind, llaminar2::MoEExpertExecutionKind::TieredExpertOverlay);

        const auto validation_result = llaminar2::validateMoEExpertParallelPlan(plan);
        EXPECT_TRUE(validation_result.ok()) << joinErrors(validation_result.errors);

        for (const auto &domain : plan.domains)
        {
            EXPECT_EQ(domain.compute_kind, llaminar2::ExpertDomainComputeKind::ReplicatedExperts)
                << domain.name;
        }

        int fallback_count = 0;
        int non_fallback_capacity = 0;
        bool saw_cpu_cold_domain = false;
        for (const auto &tier : plan.routed_tiers)
        {
            if (tier.fallback)
            {
                ++fallback_count;
            }
            else
            {
                non_fallback_capacity += tier.max_experts_per_layer;
            }

            if (tier.domain == "cpu_cold")
                saw_cpu_cold_domain = true;
        }

        if (expectation.mixed_gpu_cpu)
        {
            EXPECT_EQ(fallback_count, 1) << "mixed GPU/CPU configs must declare exactly one fallback tier";
            EXPECT_TRUE(saw_cpu_cold_domain) << "mixed GPU/CPU configs must route fallback work to cpu_cold";
        }
        else
        {
            EXPECT_EQ(fallback_count, 0) << "all-GPU configs must not declare a fallback tier";
            EXPECT_GE(non_fallback_capacity, kQwen35RoutedExpertCount)
                << "all-GPU configs must cover every routed expert by non-fallback capacity";
            EXPECT_FALSE(saw_cpu_cold_domain) << "all-GPU configs must not include cpu_cold routed tiers";
        }

        if (expectation.rebalanced)
        {
            EXPECT_EQ(plan.residency_policy, llaminar2::ExpertResidencyPolicy::RoutedTierRebalanced);
            EXPECT_TRUE(containsText(text, "moe_rebalance: dynamic"));
        }
        else
        {
            EXPECT_EQ(plan.residency_policy, llaminar2::ExpertResidencyPolicy::StaticById);
        }
    }
}

TEST(Perf__MoEGraphNativeOverlayHarness, OptInBenchmarkModePrintsCommandsOnly)
{
    if (!isTruthyEnv(std::getenv("LLAMINAR_RUN_MOE_OVERLAY_BENCHMARKS")))
    {
        SUCCEED() << "set LLAMINAR_RUN_MOE_OVERLAY_BENCHMARKS=1 to print benchmark commands";
        return;
    }

    const char *model_env = std::getenv("LLAMINAR_MOE_OVERLAY_MODEL");
    ASSERT_NE(model_env, nullptr) << "set LLAMINAR_MOE_OVERLAY_MODEL to the Qwen3.5 MoE GGUF path";

    const std::string model_path(model_env);
    ASSERT_TRUE(fs::exists(model_path)) << "model path does not exist: " << model_path;

    const fs::path config_dir("configs/moe_overlay");
    for (const auto &expectation : requiredConfigs())
    {
        const fs::path config_path = config_dir / expectation.file_name;
        ASSERT_TRUE(fs::exists(config_path));
        std::cout << commandForConfig(expectation, config_path, model_path) << '\n';
    }
}
