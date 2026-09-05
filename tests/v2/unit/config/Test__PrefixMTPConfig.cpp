#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "config/OrchestrationConfigParser.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/mtp/MTPGraphOwnerPlan.h"
#include "execution/mtp/MTPDeviceGenerationPolicy.h"
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

TEST(Test__PrefixMTPConfig, PrefixRestoreDefaultsToBoundedTieredStorage)
{
    OrchestrationConfig config;

    EXPECT_TRUE(config.prefix_cache.enabled);
    EXPECT_EQ(config.prefix_cache.storage_mode, PrefixCacheStorageMode::Tiered);
    EXPECT_EQ(config.prefix_cache.block_size, 64);
    EXPECT_EQ(
        config.prefix_cache.ram_budget_bytes,
        kDefaultPrefixCacheRamBudgetBytes);
    EXPECT_EQ(
        config.prefix_cache.device_budget_bytes,
        kDefaultPrefixCacheDeviceBudgetBytes);
    EXPECT_EQ(
        config.prefix_cache.disk_budget_bytes,
        kDefaultPrefixCacheDiskBudgetBytes);
    const char *home = std::getenv("HOME");
    ASSERT_NE(home, nullptr);
    EXPECT_EQ(
        config.prefix_cache.disk_dir,
        (std::filesystem::path(home) / ".llaminar" / "kvcache").string());
    EXPECT_EQ(config.prefix_cache.terminal_state, PrefixCacheTerminalStateMode::Auto);
    EXPECT_EQ(config.prefix_cache.moe_policy, PrefixCacheMoEPolicy::PlacementFingerprint);

    EXPECT_FALSE(config.mtp.enabled);
    EXPECT_EQ(config.mtp.draft_tokens, 1);
    EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, 0);
    EXPECT_EQ(config.mtp.max_request_batch, 1);
    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::Greedy);
    EXPECT_EQ(
        config.mtp.terminal_head_policy,
        MTPTerminalHeadPolicy::MirroredFullVocabulary)
        << "LocalTP MTP defaults to mirrored verifier heads; disabled MTP "
           "configurations simply never activate the flag.";
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

TEST(Test__PrefixMTPConfig, CommandLineCanExplicitlyDisableDefaultPrefixRestore)
{
    ArgvHelper args({"llaminar2", "--no-prefix-cache"});

    auto parser = createOrchestrationConfigParser();
    const auto config = parser->parseArgs(args.argc(), args.argv());

    EXPECT_FALSE(config.prefix_cache.enabled);
    EXPECT_EQ(config.prefix_cache.storage_mode, PrefixCacheStorageMode::Tiered);
    EXPECT_EQ(
        config.prefix_cache.ram_budget_bytes,
        kDefaultPrefixCacheRamBudgetBytes);
    EXPECT_EQ(
        config.prefix_cache.device_budget_bytes,
        kDefaultPrefixCacheDeviceBudgetBytes);
    EXPECT_EQ(
        config.prefix_cache.disk_budget_bytes,
        kDefaultPrefixCacheDiskBudgetBytes);
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

TEST(Test__PrefixMTPConfig, ValidateRejectsInvalidRequestBatchWhenMTPEnabled)
{
    OrchestrationConfig config;
    config.mtp.enabled = true;
    config.mtp.max_request_batch = 0;

    const auto errors = config.validate();

    ASSERT_FALSE(errors.empty());
    EXPECT_NE(errors.front().find("MTP max request batch must be > 0"),
              std::string::npos);
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
        "--mtp-graph-capacity-draft-tokens", "15",
        "--mtp-max-request-batch", "4",
        "--mtp-verify-mode", "speculative-sampling",
        "--mtp-terminal-head-policy", "mirrored-full-vocabulary",
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
    EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, 15);
    EXPECT_EQ(config.mtp.max_request_batch, 4);
    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
    EXPECT_EQ(
        config.mtp.terminal_head_policy,
        MTPTerminalHeadPolicy::MirroredFullVocabulary);
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

TEST(Test__PrefixMTPConfig, MTPRequestBatchCapacityResolvesRunnerBatchSize)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.max_request_batch = 4;

    EXPECT_EQ(resolveRuntimeBatchSizeForMTP(/*configured_batch_size=*/1, mtp), 4)
        << "MTP request batching must reserve enough runner-owned per-request state.";
    EXPECT_EQ(resolveRuntimeBatchSizeForMTP(/*configured_batch_size=*/8, mtp), 8)
        << "The general runner batch-size knob still wins when it is larger.";

    mtp.enabled = false;
    EXPECT_EQ(resolveRuntimeBatchSizeForMTP(/*configured_batch_size=*/1, mtp), 1)
        << "Disabled MTP must not quietly inflate normal runner capacity.";
}

TEST(Test__PrefixMTPConfig, MTPGraphCapacityUsesMaximumPolicyDepthAndRequestCount)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 15;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;

    EXPECT_EQ(resolveMTPMaximumDraftDepth(mtp), 15);
    EXPECT_EQ(resolveMTPMaxTargetQueryRows(mtp), 16)
        << "A fixed fifteen-draft transaction owns fifteen comparison rows and one bonus row.";

    mtp.draft_tokens = 3;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    mtp.depth_policy.initial_depth = 3;
    mtp.depth_policy.max_depth = 15;
    EXPECT_EQ(resolveMTPMaximumDraftDepth(mtp), 15)
        << "Graph planning must reserve the promotion ceiling, not only the warm-start depth.";
    EXPECT_EQ(resolveMTPMaxTargetQueryRows(mtp), 16);

    mtp.max_request_batch = 2;
    EXPECT_EQ(resolveMTPMaxTargetQueryRows(mtp), 32)
        << "Request batching flattens one complete target row group per request.";

    mtp.depth_policy.max_depth = 31;
    EXPECT_EQ(resolveMTPMaximumDraftDepth(mtp), 31);
    EXPECT_EQ(resolveMTPMaxTargetQueryRows(mtp), 64)
        << "Sixteen verifier rows are a certification default, not an architectural maximum.";

    mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;
    mtp.draft_tokens = 2;
    mtp.graph_capacity_draft_tokens = 15;
    EXPECT_EQ(resolveMTPMaximumExecutionDraftDepth(mtp), 2)
        << "Retained over-capacity must not widen a fixed execution policy.";
    EXPECT_EQ(resolveMTPMaximumDraftDepth(mtp), 15)
        << "One retained graph family may cover several fixed-depth requests.";
    EXPECT_EQ(resolveMTPMaxTargetQueryRows(mtp), 32)
        << "The retained request-batched verifier shape uses graph capacity.";
}

TEST(Test__PrefixMTPConfig, ValidateRejectsGraphCapacityBelowExecutionPolicy)
{
    OrchestrationConfig config;
    config.mtp.enabled = true;
    config.mtp.draft_tokens = 3;
    config.mtp.graph_capacity_draft_tokens = 2;

    const auto errors = config.validate();

    EXPECT_NE(
        std::find_if(
            errors.begin(),
            errors.end(),
            [](const std::string &error)
            {
                return error.find("graph capacity draft tokens must cover") !=
                       std::string::npos;
            }),
        errors.end());
}

TEST(Test__PrefixMTPConfig, RetainedGraphCapacityCoversPrefillAndMTPShapes)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 15;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;

    EXPECT_EQ(resolveRetainedGraphRowCapacity(/*prefill_rows=*/9, mtp), 16)
        << "A small captured-prefill bucket must not truncate depth-fifteen verification.";
    EXPECT_EQ(resolveRetainedGraphRowCapacity(/*prefill_rows=*/32, mtp), 32)
        << "A larger prefill graph already covers the verifier shape.";

    mtp.draft_tokens = 3;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    mtp.depth_policy.initial_depth = 3;
    mtp.depth_policy.max_depth = 15;
    EXPECT_EQ(resolveRetainedGraphRowCapacity(/*prefill_rows=*/9, mtp), 16)
        << "Dynamic planning must reserve the promotion ceiling, not its initial depth.";

    mtp.enabled = false;
    EXPECT_EQ(resolveRetainedGraphRowCapacity(/*prefill_rows=*/9, mtp), 9)
        << "Non-MTP retained graphs preserve their selected prefill capacity.";
}

/**
 * @brief Execution-off services may retain one explicit MTP setup envelope.
 *
 * This is the lifecycle used by a process-resident parity campaign and by a
 * serving process that changes request policy without changing model
 * placement. It must reserve capacity without accidentally enabling MTP.
 */
TEST(Test__PrefixMTPConfig, DisabledExecutionMayRetainMTPGraphCapacity)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = false;
    mtp.draft_tokens = 1;
    mtp.graph_capacity_draft_tokens = 15;

    EXPECT_TRUE(retainsMTPGraphCapacity(mtp));
    EXPECT_EQ(resolveMTPRetainedDraftCapacity(mtp), 15);
    EXPECT_EQ(resolveMTPRetainedTargetQueryRows(mtp), 16);
    EXPECT_EQ(resolveRetainedGraphRowCapacity(/*prefill_rows=*/9, mtp), 16);
    EXPECT_FALSE(mtp.enabled)
        << "retained setup capacity must never select an execution policy";

    const MTPGraphOwnerPlan owner_plan(mtp);
    EXPECT_TRUE(owner_plan.retainsGraphCapacity());
    EXPECT_EQ(owner_plan.draftDepth(), 15);
    EXPECT_EQ(owner_plan.requestCapacity(), 1);
    EXPECT_EQ(owner_plan.verifierRowsPerRequest(), 16);
    EXPECT_EQ(owner_plan.flattenedTargetRows(), 16);
    EXPECT_EQ(owner_plan.sidecarGraphSlots(), 21u);
    EXPECT_EQ(owner_plan.terminalHiddenGraphSlots(), 35u);
    EXPECT_EQ(owner_plan.draftPublicationGraphSlots(), 15u);
    EXPECT_EQ(owner_plan.verifierPreparationGraphSlots(), 32u);
    EXPECT_EQ(owner_plan.controllerGraphSlots(), 4u);
    EXPECT_EQ(owner_plan.auxiliaryExecutableSlotCount(), 107u);
}

TEST(Test__PrefixMTPConfig,
     TypedRequestPolicyChangesExecutionWithoutChangingPhysicalCapacity)
{
    MTPRuntimeConfig retained;
    retained.enabled = true;
    retained.draft_tokens = 15;
    retained.graph_capacity_draft_tokens = 15;
    retained.max_request_batch = 4;
    retained.sidecar_dense_policy =
        MTPSidecarDensePolicy::ReplicatedPerParticipant;
    retained.terminal_head_policy =
        MTPTerminalHeadPolicy::MirroredFullVocabulary;

    MTPRequestPolicy request;
    request.enabled = true;
    request.draft_tokens = 2;
    request.verify_mode = MTPVerifyMode::Greedy;
    request.depth_policy.mode = MTPDepthPolicyMode::Fixed;

    EXPECT_FALSE(validateMTPRequestPolicy(request, retained).has_value());
    const MTPRuntimeConfig active =
        composeMTPRequestConfig(retained, request);
    EXPECT_TRUE(active.enabled);
    EXPECT_EQ(resolveMTPMaximumExecutionDraftDepth(active), 2);
    EXPECT_EQ(resolveMTPRetainedDraftCapacity(active), 15);
    EXPECT_EQ(active.max_request_batch, retained.max_request_batch);
    EXPECT_EQ(active.sidecar_dense_policy, retained.sidecar_dense_policy);
    EXPECT_EQ(active.terminal_head_policy, retained.terminal_head_policy);

    request.enabled = false;
    const MTPRuntimeConfig disabled =
        composeMTPRequestConfig(retained, request);
    EXPECT_FALSE(disabled.enabled);
    EXPECT_EQ(resolveMTPRetainedDraftCapacity(disabled), 15)
        << "Disabling one request must not retire its reusable graph family";
}

TEST(Test__PrefixMTPConfig,
     TypedRequestPolicyRejectsDepthBeyondRetainedEnvelope)
{
    MTPRuntimeConfig retained;
    retained.enabled = false;
    retained.graph_capacity_draft_tokens = 3;

    MTPRequestPolicy fixed;
    fixed.enabled = true;
    fixed.draft_tokens = 4;
    fixed.depth_policy.mode = MTPDepthPolicyMode::Fixed;
    const auto fixed_error =
        validateMTPRequestPolicy(fixed, retained);
    ASSERT_TRUE(fixed_error.has_value());
    EXPECT_NE(fixed_error->find("exceeds"), std::string::npos);

    MTPRequestPolicy dynamic;
    dynamic.enabled = true;
    dynamic.draft_tokens = 2;
    dynamic.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    dynamic.depth_policy.min_depth = 1;
    dynamic.depth_policy.max_depth = 15;
    dynamic.depth_policy.initial_depth = 2;
    const auto dynamic_error =
        validateMTPRequestPolicy(dynamic, retained);
    ASSERT_TRUE(dynamic_error.has_value());
    EXPECT_NE(dynamic_error->find("exceeds"), std::string::npos);
}

TEST(Test__PrefixMTPConfig,
     TypedRequestPolicyProjectsAllMutableFieldsRoundTrip)
{
    MTPRuntimeConfig source;
    source.enabled = true;
    source.draft_tokens = 3;
    source.verify_mode = MTPVerifyMode::SpeculativeSampling;
    source.require_terminal_hidden_for_full_hit = false;
    source.depth_policy.mode = MTPDepthPolicyMode::Observe;
    source.depth_policy.min_depth = 1;
    source.depth_policy.max_depth = 3;
    source.depth_policy.initial_depth = 2;

    const MTPRequestPolicy policy = makeMTPRequestPolicy(source);
    EXPECT_TRUE(policy.enabled);
    EXPECT_EQ(policy.draft_tokens, 3);
    EXPECT_EQ(policy.verify_mode, MTPVerifyMode::SpeculativeSampling);
    EXPECT_FALSE(policy.require_terminal_hidden_for_full_hit);
    EXPECT_EQ(policy.depth_policy.mode, MTPDepthPolicyMode::Observe);
    EXPECT_EQ(policy.depth_policy.min_depth, 1);
    EXPECT_EQ(policy.depth_policy.max_depth, 3);
    EXPECT_EQ(policy.depth_policy.initial_depth, 2);
}

/**
 * @brief The request/device policy conversion is exact and backend-neutral.
 */
TEST(Test__PrefixMTPConfig,
     TypedRequestPolicySealsExactDynamicDeviceControllerPolicy)
{
    MTPRuntimeConfig active;
    active.enabled = true;
    active.draft_tokens = 7;
    active.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    active.depth_policy.min_depth = 1;
    active.depth_policy.max_depth = 15;
    active.depth_policy.initial_depth = 7;
    active.depth_policy.window_size = 19;
    active.depth_policy.min_samples = 5;
    active.depth_policy.cooldown_steps = 2;
    active.depth_policy.promote_consecutive_windows = 4;
    active.depth_policy.promote_full_accept_rate = 0.987654;
    active.depth_policy.demote_zero_accept_rate = 0.234567;
    active.depth_policy.demote_acceptance_rate = 0.543210;

    const auto device = resolveMTPDeviceGenerationDepthPolicy(active);
    EXPECT_TRUE(device.valid());
    EXPECT_EQ(
        device.mode,
        sampling_math::DeviceGenerationDepthPolicyMode::Dynamic);
    EXPECT_EQ(device.minimum_depth, 1);
    EXPECT_EQ(device.maximum_depth, 15);
    EXPECT_EQ(device.initial_depth, 7);
    EXPECT_EQ(device.window_size, 19);
    EXPECT_EQ(device.minimum_samples, 5);
    EXPECT_EQ(device.cooldown_steps, 2);
    EXPECT_EQ(device.promote_consecutive_windows, 4);
    EXPECT_EQ(device.promote_full_accept_rate_ppm, 987654);
    EXPECT_EQ(device.demote_zero_accept_rate_ppm, 234567);
    EXPECT_EQ(device.demote_acceptance_rate_ppm, 543210);

    DeviceGenerationAdmissionRequest omitted;
    omitted.request_count = 1;
    omitted.max_new_tokens = 32;
    EXPECT_FALSE(omitted.valid())
        << "An admission may not infer depth policy from retained setup";
    omitted.depth_policy = device;
    EXPECT_TRUE(omitted.valid());
}

/**
 * @brief Unknown host modes remain invalid through the device ABI conversion.
 */
TEST(Test__PrefixMTPConfig,
     TypedRequestPolicyDoesNotNormalizeUnknownDeviceControllerMode)
{
    MTPRuntimeConfig active;
    active.enabled = true;
    active.depth_policy.mode = static_cast<MTPDepthPolicyMode>(99);

    EXPECT_FALSE(resolveMTPDeviceGenerationDepthPolicy(active).valid());
}

TEST(Test__PrefixMTPConfig,
     GraphOwnerPlanScalesEveryRequestIndexedDirectory)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 3;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;
    mtp.max_request_batch = 2;

    const MTPGraphOwnerPlan owner_plan(mtp);
    EXPECT_EQ(owner_plan.draftDepth(), 3);
    EXPECT_EQ(owner_plan.requestCapacity(), 2);
    EXPECT_EQ(owner_plan.verifierRowsPerRequest(), 4);
    EXPECT_EQ(owner_plan.flattenedTargetRows(), 8);
    EXPECT_EQ(owner_plan.stochasticTargetRows(), 8);
    EXPECT_EQ(owner_plan.stochasticDraftRows(), 6);
    EXPECT_EQ(owner_plan.sidecarKVOnlyBatchGraphSlots(), 7u);
    EXPECT_EQ(owner_plan.sidecarGraphSlots(), 13u);
    EXPECT_EQ(owner_plan.genericTerminalHiddenGraphSlots(), 1u);
    EXPECT_EQ(owner_plan.terminalHiddenContiguousGraphSlots(), 8u);
    EXPECT_EQ(owner_plan.terminalHiddenDeviceAcceptedGraphSlots(), 2u);
    EXPECT_EQ(owner_plan.terminalHiddenRequestTerminalGraphSlots(), 2u);
    EXPECT_EQ(owner_plan.terminalHiddenShiftedPrefillGraphSlots(), 16u);
    EXPECT_EQ(owner_plan.terminalHiddenGraphSlots(), 29u);
    EXPECT_EQ(owner_plan.draftPublicationGraphSlots(), 6u);
    EXPECT_EQ(owner_plan.verifierPreparationGraphSlots(), 16u);
    EXPECT_EQ(owner_plan.auxiliaryExecutableSlotCount(), 68u);
}

TEST(Test__PrefixMTPConfig,
     DisabledGraphOwnerPlanRetainsOnlyNonGraphSamplingGeometry)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = false;

    const MTPGraphOwnerPlan owner_plan(mtp);
    EXPECT_FALSE(owner_plan.retainsGraphCapacity());
    EXPECT_EQ(owner_plan.stochasticTargetRows(), 4);
    EXPECT_EQ(owner_plan.stochasticDraftRows(), 3);
    EXPECT_EQ(owner_plan.auxiliaryExecutableSlotCount(), 0u);
    EXPECT_EQ(owner_plan.sidecarKVOnlyBatchGraphSlots(), 0u);
    EXPECT_EQ(owner_plan.terminalHiddenGraphSlots(), 0u);
}

TEST(Test__PrefixMTPConfig, DisabledExecutionRejectsNegativeRetainedCapacity)
{
    OrchestrationConfig config;
    config.mtp.enabled = false;
    config.mtp.graph_capacity_draft_tokens = -1;

    const auto errors = config.validate();
    EXPECT_NE(
        std::find_if(
            errors.begin(),
            errors.end(),
            [](const std::string &error)
            {
                return error.find(
                           "graph capacity draft tokens must be >= 0") !=
                       std::string::npos;
            }),
        errors.end());
}

TEST(Test__PrefixMTPConfig, MTPTerminalHiddenArchiveCoversRequestAndVerifierRows)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = true;
    mtp.draft_tokens = 3;
    mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;
    mtp.max_request_batch = 1;

    EXPECT_EQ(resolveMTPTerminalHiddenRowCapacity(1, mtp), 4)
        << "A scalar fixed-d3 request must retain all four verifier target rows.";
    EXPECT_EQ(resolveMTPTerminalHiddenRowCapacity(8, mtp), 8)
        << "A larger general request batch must still have one terminal row per request.";

    mtp.max_request_batch = 3;
    EXPECT_EQ(resolveMTPTerminalHiddenRowCapacity(8, mtp), 12)
        << "Flattened grouped-verifier rows win when they exceed request capacity.";

    mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    mtp.depth_policy.max_depth = 15;
    EXPECT_EQ(resolveMTPTerminalHiddenRowCapacity(8, mtp), 48)
        << "The archive must reserve dynamic depth's promotion ceiling before capture.";
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
    {
        ArgvHelper args({"llaminar2", "--mtp-max-request-batch", "0"});
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
  graph_capacity_draft_tokens: 15
  max_request_batch: 2
  verify_mode: greedy
  terminal_head_policy: mirrored-full-vocabulary
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
    EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, 15);
    EXPECT_EQ(config.mtp.max_request_batch, 2);
    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::Greedy);
    EXPECT_EQ(
        config.mtp.terminal_head_policy,
        MTPTerminalHeadPolicy::MirroredFullVocabulary);
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
    source.mtp.graph_capacity_draft_tokens = 15;
    source.mtp.max_request_batch = 4;
    source.mtp.verify_mode = MTPVerifyMode::SpeculativeSampling;
    source.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
    source.mtp.depth_policy.max_depth = 3;
    source.mtp.depth_policy.window_size = 8;
    source.moe_routed_prefill.assignment_window_tokens = 64;
    source.moe_routed_prefill.least_loaded_min_routed_rows = 4096;

    RuntimeConfig runtime = RuntimeConfig::fromOrchestrationConfig(
        source.max_seq_len,
        source.batch_size,
        source.activation_precision,
        source.kv_cache_precision,
        source.fused_attention_backend,
        source.routed_expert_compute_policy,
        source.moe_hot_expert_cache,
        source.moe_routed_prefill,
        source.moe_rebalance,
        source.prefix_cache,
        source.mtp);

    RankExecutionPlan plan;
    plan.runtime = runtime;
    plan.runtime.resident_graph_rows = 2048;
    InferenceRunnerConfig runner_config = InferenceRunnerConfig::fromPlan(plan);
    RankOrchestrator::Config rank_config =
        RankOrchestrator::Config::fromPlan(plan);

    GraphConfig graph_config;
    graph_config.prefix_cache = runner_config.prefix_cache;
    graph_config.mtp = runner_config.mtp;

    EXPECT_EQ(runner_config.moe_routed_prefill.assignment_window_tokens, 64);
    EXPECT_EQ(runner_config.moe_routed_prefill.least_loaded_min_routed_rows,
              4096u);
    EXPECT_EQ(rank_config.moe_routed_prefill.assignment_window_tokens, 64);

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
    EXPECT_EQ(graph_config.mtp.graph_capacity_draft_tokens, 15);
    EXPECT_EQ(graph_config.mtp.max_request_batch, 4);
    EXPECT_EQ(graph_config.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
    EXPECT_EQ(graph_config.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
    EXPECT_EQ(graph_config.mtp.depth_policy.max_depth, 3);
    EXPECT_EQ(graph_config.mtp.depth_policy.window_size, 8);
    EXPECT_EQ(runtime.batch_size, 4);
    EXPECT_EQ(runner_config.batch_size, 4);
    EXPECT_EQ(runner_config.activation_seq_len, 2048);
    EXPECT_EQ(rank_config.resident_graph_rows, 2048);
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
    config.mtp.graph_capacity_draft_tokens = 15;
    config.mtp.max_request_batch = 3;
    config.mtp.verify_mode = MTPVerifyMode::Greedy;
    config.mtp.terminal_head_policy =
        MTPTerminalHeadPolicy::MirroredFullVocabulary;
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
    EXPECT_NE(
        explanation.find("graph_capacity_draft_tokens: 15"),
        std::string::npos);
    EXPECT_NE(explanation.find("max_request_batch: 3"), std::string::npos);
    EXPECT_NE(explanation.find("verify_mode: greedy"), std::string::npos);
    EXPECT_NE(
        explanation.find("terminal_head_policy: mirrored-full-vocabulary"),
        std::string::npos);
    EXPECT_NE(explanation.find("depth_policy: observe"), std::string::npos);
    EXPECT_NE(explanation.find("max_draft_tokens: 3"), std::string::npos);
    EXPECT_NE(explanation.find("depth_window: 8"), std::string::npos);
    EXPECT_NE(explanation.find("require_terminal_hidden_for_full_hit: false"), std::string::npos);
}

/**
 * @brief Every shape-dependent forward family requires eager publication.
 *
 * The serial stochastic oracle intentionally disables MTP while preserving the
 * production phase-split dense policy. This matrix locks in replicated or
 * mirrored decode topology and ExpertOverlay graph families; durable MoE
 * maintenance itself runs through the background RCU authority and does not
 * install a second graph-side Dynamic family.
 */
TEST(Test__PrefixMTPConfig, ShapeDependentForwardPoliciesRequireEagerFamilyManifest)
{
    GraphConfig config;
    config.dense_tp_enabled = true;

    EXPECT_FALSE(config.requiresEagerWorkspaceFamilyManifest());

    config.mtp.enabled = true;
    EXPECT_TRUE(config.requiresEagerWorkspaceFamilyManifest())
        << "CPU stages and captured GPU graphs both retain bound workspace addresses";

    config.mtp.enabled = false;
    config.mtp.graph_capacity_draft_tokens = 15;
    EXPECT_TRUE(config.requiresEagerWorkspaceFamilyManifest())
        << "A capacity-only model context must declare the same graph family as an enabled lease";
    EXPECT_FALSE(config.usesMTPGroupedDecodeEquivalentRows());
    config.compute_all_position_logits = true;
    EXPECT_TRUE(config.usesMTPGroupedDecodeEquivalentRows())
        << "A retained verifier declaration must select production grouped arithmetic even when execution is off";
    config.compute_all_position_logits = false;

    config.mtp.graph_capacity_draft_tokens = 0;
    config.dense_tp_decode_replicated = true;
    EXPECT_TRUE(config.requiresEagerWorkspaceFamilyManifest());

    config.dense_tp_decode_replicated = false;
    config.dense_tp_decode_mirrored_embedding = true;
    EXPECT_TRUE(config.requiresEagerWorkspaceFamilyManifest());

    config.dense_tp_enabled = false;
    EXPECT_FALSE(config.requiresEagerWorkspaceFamilyManifest())
        << "A replicated single-device graph has no phase-split TP topology";

    config.moe.routed_expert_plan =
        std::make_shared<MoERoutedExpertPlacementPlan>();
    config.moe.routed_expert_plan->enabled = true;
    config.moe.routed_expert_plan->topology =
        RoutedExpertPlacementTopology::TieredOverlay;
    EXPECT_TRUE(config.requiresEagerWorkspaceFamilyManifest())
        << "Tiered overlay startup must resolve every participant's initial prepared bank";
}
