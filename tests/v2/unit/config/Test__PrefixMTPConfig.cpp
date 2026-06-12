#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/OrchestrationConfigParser.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "models/GraphTypes.h"
#include "utils/DebugEnv.h"

using namespace llaminar2;

namespace
{
    class ArgvHelper
    {
    public:
        ArgvHelper(std::initializer_list<const char *> args)
        {
            for (const char *arg : args)
                strings_.push_back(arg);
            for (auto &arg : strings_)
                argv_.push_back(const_cast<char *>(arg.c_str()));
        }

        int argc() const { return static_cast<int>(argv_.size()); }
        char **argv() { return argv_.data(); }

    private:
        std::vector<std::string> strings_;
        std::vector<char *> argv_;
    };

    class ScopedEnvVar
    {
    public:
        explicit ScopedEnvVar(const char *name)
            : name_(name)
        {
            if (const char *current = std::getenv(name_.c_str()))
                previous_ = current;
        }

        ~ScopedEnvVar()
        {
            if (previous_)
                ::setenv(name_.c_str(), previous_->c_str(), 1);
            else
                ::unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

        void set(const char *value)
        {
            ::setenv(name_.c_str(), value, 1);
            mutableDebugEnv().reload();
        }

        void unset()
        {
            ::unsetenv(name_.c_str());
            mutableDebugEnv().reload();
        }

    private:
        std::string name_;
        std::optional<std::string> previous_;
    };
} // namespace

TEST(Test__PrefixMTPConfig, DefaultsAreDisabled)
{
    OrchestrationConfig config;

    EXPECT_FALSE(config.prefix_cache.enabled);
    EXPECT_EQ(config.prefix_cache.storage_mode, PrefixCacheStorageMode::Tiered);
    EXPECT_EQ(config.prefix_cache.block_size, 64);
    EXPECT_EQ(config.prefix_cache.ram_budget_bytes, 4ull * 1024ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.device_budget_bytes, 256ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.disk_budget_bytes, 0u);
    EXPECT_EQ(config.prefix_cache.terminal_state, PrefixCacheTerminalStateMode::Auto);
    EXPECT_EQ(config.prefix_cache.moe_policy, PrefixCacheMoEPolicy::PlacementFingerprint);

    EXPECT_FALSE(config.mtp.enabled);
    EXPECT_EQ(config.mtp.draft_tokens, 1);
    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::Greedy);
    EXPECT_TRUE(config.mtp.require_terminal_hidden_for_full_hit);
    EXPECT_EQ(config.mtp.depth_policy.mode, MTPDepthPolicyMode::Fixed);
    EXPECT_EQ(config.mtp.depth_policy.min_depth, 1);
    EXPECT_EQ(config.mtp.depth_policy.max_depth, 0);
    EXPECT_EQ(config.mtp.depth_policy.initial_depth, 0);
    EXPECT_EQ(config.mtp.depth_policy.window_size, 16);
    EXPECT_EQ(config.mtp.depth_policy.min_samples, 4);
    EXPECT_EQ(config.mtp.depth_policy.cooldown_steps, 8);
    EXPECT_EQ(config.mtp.depth_policy.promote_consecutive_windows, 3);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.promote_full_accept_rate, 1.0);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.demote_zero_accept_rate, 0.30);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.demote_acceptance_rate, 0.55);
}

TEST(Test__PrefixMTPConfig, ROCmTopKSmallKPartialBlockOverrideIsValidated)
{
    ScopedEnvVar override_env("LLAMINAR_ROCM_TOPK_SMALLK_PARTIAL_BLOCKS");

    override_env.unset();
    EXPECT_EQ(debugEnv().rocm.topk_smallk_partial_blocks, 0)
        << "Unset ROCm sampler cap should keep the production auto policy.";

    override_env.set("64");
    EXPECT_EQ(debugEnv().rocm.topk_smallk_partial_blocks, 64);

    override_env.set("33");
    EXPECT_EQ(debugEnv().rocm.topk_smallk_partial_blocks, 0)
        << "Unsupported caps should not silently enter the launch policy.";

    override_env.set("128");
    EXPECT_EQ(debugEnv().rocm.topk_smallk_partial_blocks, 128);
}

TEST(Test__PrefixMTPConfig, ValidateIgnoresDepthPolicyWhenMTPDisabled)
{
    OrchestrationConfig config;
    config.mtp.enabled = false;
    config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    config.mtp.depth_policy.min_depth = 3;
    config.mtp.depth_policy.max_depth = 1;
    config.mtp.depth_policy.initial_depth = 9;
    config.mtp.depth_policy.promote_consecutive_windows = 0;

    const auto errors = config.validate();

    EXPECT_TRUE(errors.empty())
        << "Disabled MTP must not make no-MTP baselines depend on adaptive-depth knobs";
}

TEST(Test__PrefixMTPConfig, ValidateFixedDepthIgnoresAdaptiveOnlyKnobs)
{
    OrchestrationConfig config;
    config.mtp.enabled = true;
    config.mtp.draft_tokens = 3;
    config.mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;
    config.mtp.depth_policy.min_depth = 1;
    config.mtp.depth_policy.max_depth = 0;
    config.mtp.depth_policy.initial_depth = 0;
    config.mtp.depth_policy.promote_consecutive_windows = 0;
    config.mtp.depth_policy.window_size = 0;
    config.mtp.depth_policy.min_samples = 0;

    EXPECT_TRUE(config.validate().empty())
        << "Fixed-depth lanes are normalized to min=max=initial=draft_tokens at controller setup.";

    config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    const auto errors = config.validate();
    EXPECT_FALSE(errors.empty())
        << "Dynamic depth must still reject the same invalid adaptive knobs.";
}

TEST(Test__PrefixMTPConfig, ParserAcceptsPrefixCacheAndMTPFlags)
{
    ArgvHelper args({
        "llaminar2",
        "--prefix-cache",
        "--prefix-cache-storage", "ram",
        "--prefix-cache-block-size", "32",
        "--prefix-cache-vram-budget-mb", "128",
        "--prefix-cache-ram-budget-mb", "2048",
        "--prefix-cache-disk-budget-mb", "512",
        "--prefix-cache-disk-dir", "/tmp/llaminar-prefix",
        "--prefix-cache-terminal-state", "always",
        "--prefix-cache-moe-policy", "invalidate-on-rebalance",
        "--mtp",
        "--mtp-draft-tokens", "2",
        "--mtp-verify-mode", "speculative-sampling",
        "--mtp-depth-policy", "dynamic",
        "--mtp-min-draft-tokens", "1",
        "--mtp-max-draft-tokens", "3",
        "--mtp-initial-draft-tokens", "2",
        "--mtp-depth-window", "8",
        "--mtp-depth-min-samples", "4",
        "--mtp-depth-cooldown", "2",
        "--mtp-depth-promote-windows", "3",
        "--mtp-depth-promote-full-accept", "0.70",
        "--mtp-depth-demote-zero-accept", "0.25",
        "--mtp-depth-demote-acceptance", "0.60",
    });

    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.prefix_cache.enabled);
    EXPECT_EQ(config.prefix_cache.storage_mode, PrefixCacheStorageMode::Ram);
    EXPECT_EQ(config.prefix_cache.block_size, 32);
    EXPECT_EQ(config.prefix_cache.device_budget_bytes, 128ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.ram_budget_bytes, 2048ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.disk_budget_bytes, 512ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.disk_dir, "/tmp/llaminar-prefix");
    EXPECT_EQ(config.prefix_cache.terminal_state, PrefixCacheTerminalStateMode::Always);
    EXPECT_EQ(config.prefix_cache.moe_policy, PrefixCacheMoEPolicy::InvalidateOnRebalance);

    EXPECT_TRUE(config.mtp.enabled);
    EXPECT_EQ(config.mtp.draft_tokens, 2);
    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
    EXPECT_EQ(config.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
    EXPECT_EQ(config.mtp.depth_policy.min_depth, 1);
    EXPECT_EQ(config.mtp.depth_policy.max_depth, 3);
    EXPECT_EQ(config.mtp.depth_policy.initial_depth, 2);
    EXPECT_EQ(config.mtp.depth_policy.window_size, 8);
    EXPECT_EQ(config.mtp.depth_policy.min_samples, 4);
    EXPECT_EQ(config.mtp.depth_policy.cooldown_steps, 2);
    EXPECT_EQ(config.mtp.depth_policy.promote_consecutive_windows, 3);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.promote_full_accept_rate, 0.70);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.demote_zero_accept_rate, 0.25);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.demote_acceptance_rate, 0.60);
}

TEST(Test__PrefixMTPConfig, ParserRejectsInvalidPrefixAndMTPEnums)
{
    {
        ArgvHelper args({"llaminar2", "--prefix-cache-storage", "cloud"});
        auto parser = createOrchestrationConfigParser();
        EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args({"llaminar2", "--prefix-cache-terminal-state", "maybe"});
        auto parser = createOrchestrationConfigParser();
        EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args({"llaminar2", "--prefix-cache-moe-policy", "reuse-anyway"});
        auto parser = createOrchestrationConfigParser();
        EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args({"llaminar2", "--mtp-verify-mode", "oracle"});
        auto parser = createOrchestrationConfigParser();
        EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args({"llaminar2", "--mtp-depth-policy", "random-walk"});
        auto parser = createOrchestrationConfigParser();
        EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args({"llaminar2", "--mtp-depth-demote-zero-accept", "1.5"});
        auto parser = createOrchestrationConfigParser();
        EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args({"llaminar2", "--mtp-initial-draft-tokens", "-1"});
        auto parser = createOrchestrationConfigParser();
        EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
}

TEST(Test__PrefixMTPConfig, ParserAcceptsDynamicDepthZeroBypassPolicy)
{
    ArgvHelper args({
        "llaminar2",
        "--mtp",
        "--mtp-draft-tokens", "3",
        "--mtp-depth-policy", "dynamic",
        "--mtp-min-draft-tokens", "0",
        "--mtp-initial-draft-tokens", "1",
        "--mtp-max-draft-tokens", "3",
    });

    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mtp.enabled);
    EXPECT_EQ(config.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
    EXPECT_EQ(config.mtp.depth_policy.min_depth, 0);
    EXPECT_EQ(config.mtp.depth_policy.initial_depth, 1);
    EXPECT_EQ(config.mtp.depth_policy.max_depth, 3);
}

TEST(Test__PrefixMTPConfig, YamlSectionsParsePrefixCacheAndMTP)
{
    const std::string yaml = R"yaml(
prefix_cache:
  enabled: true
  storage: device
  block_size: 16
  ram_budget_mb: 1024
  vram_budget_mb: 64
  disk_budget_mb: 128
  disk_dir: "/tmp/prefix"
  terminal_state: off
  moe_policy: disabled
mtp:
  enabled: true
  draft_tokens: 3
  verify_mode: greedy
  require_terminal_hidden_for_full_hit: false
  depth_policy: observe
  min_draft_tokens: 1
  max_draft_tokens: 3
  initial_draft_tokens: 2
  depth_window: 12
  depth_min_samples: 6
  depth_cooldown: 3
  depth_promote_windows: 4
  depth_promote_full_accept: 0.8
  depth_demote_zero_accept: 0.2
  depth_demote_acceptance: 0.55
)yaml";

    OrchestrationConfigParser parser;
    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.prefix_cache.enabled);
    EXPECT_EQ(config.prefix_cache.storage_mode, PrefixCacheStorageMode::Device);
    EXPECT_EQ(config.prefix_cache.block_size, 16);
    EXPECT_EQ(config.prefix_cache.ram_budget_bytes, 1024ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.device_budget_bytes, 64ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.disk_budget_bytes, 128ull * 1024ull * 1024ull);
    EXPECT_EQ(config.prefix_cache.disk_dir, "/tmp/prefix");
    EXPECT_EQ(config.prefix_cache.terminal_state, PrefixCacheTerminalStateMode::Off);
    EXPECT_EQ(config.prefix_cache.moe_policy, PrefixCacheMoEPolicy::Disabled);

    EXPECT_TRUE(config.mtp.enabled);
    EXPECT_EQ(config.mtp.draft_tokens, 3);
    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::Greedy);
    EXPECT_FALSE(config.mtp.require_terminal_hidden_for_full_hit);
    EXPECT_EQ(config.mtp.depth_policy.mode, MTPDepthPolicyMode::Observe);
    EXPECT_EQ(config.mtp.depth_policy.min_depth, 1);
    EXPECT_EQ(config.mtp.depth_policy.max_depth, 3);
    EXPECT_EQ(config.mtp.depth_policy.initial_depth, 2);
    EXPECT_EQ(config.mtp.depth_policy.window_size, 12);
    EXPECT_EQ(config.mtp.depth_policy.min_samples, 6);
    EXPECT_EQ(config.mtp.depth_policy.cooldown_steps, 3);
    EXPECT_EQ(config.mtp.depth_policy.promote_consecutive_windows, 4);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.promote_full_accept_rate, 0.8);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.demote_zero_accept_rate, 0.2);
    EXPECT_DOUBLE_EQ(config.mtp.depth_policy.demote_acceptance_rate, 0.55);
}

TEST(Test__PrefixMTPConfig, RuntimeConfigSurvivesPlanRunnerAndGraphCopies)
{
    OrchestrationConfig source;
    source.prefix_cache.enabled = true;
    source.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
    source.prefix_cache.block_size = 24;
    source.prefix_cache.ram_budget_bytes = 99;
    source.prefix_cache.device_budget_bytes = 77;
    source.prefix_cache.disk_budget_bytes = 55;
    source.prefix_cache.disk_dir = "/tmp/unit-prefix";
    source.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Always;
    source.prefix_cache.moe_policy = PrefixCacheMoEPolicy::InvalidateOnRebalance;
    source.mtp.enabled = true;
    source.mtp.draft_tokens = 2;
    source.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
    source.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    source.mtp.depth_policy.max_depth = 3;
    source.mtp.depth_policy.window_size = 8;

    RuntimeConfig runtime = RuntimeConfig::fromOrchestrationConfig(
        source.max_seq_len,
        source.activation_precision,
        source.kv_cache_precision,
        source.fused_attention_backend,
        source.moe_expert_mode,
        source.moe_hot_expert_cache,
        source.moe_rebalance,
        source.prefix_cache,
        source.mtp);

    RankExecutionPlan plan;
    plan.runtime = runtime;
    InferenceRunnerConfig runner_config = InferenceRunnerConfig::fromPlan(plan);

    GraphConfig graph_config;
    graph_config.prefix_cache = runner_config.prefix_cache;
    graph_config.mtp = runner_config.mtp;

    EXPECT_TRUE(graph_config.prefix_cache.enabled);
    EXPECT_EQ(graph_config.prefix_cache.storage_mode, PrefixCacheStorageMode::Ram);
    EXPECT_EQ(graph_config.prefix_cache.block_size, 24);
    EXPECT_EQ(graph_config.prefix_cache.ram_budget_bytes, 99u);
    EXPECT_EQ(graph_config.prefix_cache.device_budget_bytes, 77u);
    EXPECT_EQ(graph_config.prefix_cache.disk_budget_bytes, 55u);
    EXPECT_EQ(graph_config.prefix_cache.disk_dir, "/tmp/unit-prefix");
    EXPECT_EQ(graph_config.prefix_cache.terminal_state, PrefixCacheTerminalStateMode::Always);
    EXPECT_EQ(graph_config.prefix_cache.moe_policy, PrefixCacheMoEPolicy::InvalidateOnRebalance);

    EXPECT_TRUE(graph_config.mtp.enabled);
    EXPECT_EQ(graph_config.mtp.draft_tokens, 2);
    EXPECT_EQ(graph_config.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
    EXPECT_EQ(graph_config.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
    EXPECT_EQ(graph_config.mtp.depth_policy.max_depth, 3);
    EXPECT_EQ(graph_config.mtp.depth_policy.window_size, 8);
}

TEST(Test__PrefixMTPConfig, ExplanationIncludesResolvedPrefixCacheAndMTPSettings)
{
    OrchestrationConfig config;
    config.prefix_cache.enabled = true;
    config.prefix_cache.storage_mode = PrefixCacheStorageMode::Ram;
    config.prefix_cache.block_size = 24;
    config.prefix_cache.ram_budget_bytes = 99;
    config.prefix_cache.device_budget_bytes = 77;
    config.prefix_cache.disk_budget_bytes = 55;
    config.prefix_cache.disk_dir = "/tmp/unit-prefix";
    config.prefix_cache.terminal_state = PrefixCacheTerminalStateMode::Always;
    config.prefix_cache.moe_policy = PrefixCacheMoEPolicy::InvalidateOnRebalance;
    config.mtp.enabled = true;
    config.mtp.draft_tokens = 2;
    config.mtp.verify_mode = MTPVerifyMode::Greedy;
    config.mtp.require_terminal_hidden_for_full_hit = false;
    config.mtp.depth_policy.mode = MTPDepthPolicyMode::Observe;
    config.mtp.depth_policy.max_depth = 3;
    config.mtp.depth_policy.window_size = 8;

    const std::string explanation = config.toString();

    EXPECT_NE(explanation.find("prefix_cache:"), std::string::npos);
    EXPECT_NE(explanation.find("enabled: true"), std::string::npos);
    EXPECT_NE(explanation.find("storage: ram"), std::string::npos);
    EXPECT_NE(explanation.find("block_size: 24"), std::string::npos);
    EXPECT_NE(explanation.find("ram_budget_bytes: 99"), std::string::npos);
    EXPECT_NE(explanation.find("device_budget_bytes: 77"), std::string::npos);
    EXPECT_NE(explanation.find("disk_budget_bytes: 55"), std::string::npos);
    EXPECT_NE(explanation.find("disk_dir: /tmp/unit-prefix"), std::string::npos);
    EXPECT_NE(explanation.find("terminal_state: always"), std::string::npos);
    EXPECT_NE(explanation.find("moe_policy: invalidate-on-rebalance"), std::string::npos);
    EXPECT_NE(explanation.find("mtp:"), std::string::npos);
    EXPECT_NE(explanation.find("draft_tokens: 2"), std::string::npos);
    EXPECT_NE(explanation.find("verify_mode: greedy"), std::string::npos);
    EXPECT_NE(explanation.find("depth_policy: observe"), std::string::npos);
    EXPECT_NE(explanation.find("max_draft_tokens: 3"), std::string::npos);
    EXPECT_NE(explanation.find("depth_window: 8"), std::string::npos);
    EXPECT_NE(explanation.find("require_terminal_hidden_for_full_hit: false"), std::string::npos);
}
