/**
 * @file Test__ModelParityBenchmarkProjection.cpp
 * @brief Device-free proof that canonical benchmarks time production defaults.
 *
 * Model/topology and selected modes still come from the canonical matrix, but
 * short-trace correctness stress must not set benchmark depth or economics.
 * Parse real public CLI projections to prove this boundary without launching
 * a model, initializing an accelerator or enforcing noisy timing thresholds.
 */
#include "integration/parity/ModelParityDefinition.h"
#include "integration/parity/qwen36/Qwen36ModelParityDefinitions.h"
#include "integration/parity/qwen38/Qwen38ModelParityDefinitions.h"
#include "config/OrchestrationConfigParser.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

namespace llaminar2::test::parity
{
    namespace
    {
        /** @return Real parser result without executing startup or discovery. */
        OrchestrationConfig parseBenchmarkIntent(std::vector<std::string> args)
        {
            args.insert(args.begin(), "llaminar2");
            std::vector<char *> argv;
            for (auto &arg : args) argv.push_back(arg.data());
            return OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        }

        /** @return Existing dense/MoE definitions across CPU and both GPU vendors. */
        std::vector<ModelParityDefinition> benchmarkDefinitions()
        {
            auto definitions = qwen38::qwen38DenseMultiDeviceDefinitions();
            definitions.push_back(qwen36::qwen36MoECPU2NodeTPParityDefinition());
            for (const auto device : {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)})
                definitions.push_back(qwen38::qwen38DenseParityDefinition(
                    qwen36::qwen36SingleDeviceTopology("single", device), "/references"));
            return definitions;
        }
    }

    TEST(ModelParityBenchmarkProjection, PreservesIntentButUsesProductionDefaults)
    {
        const auto defaults = parseBenchmarkIntent({"--mtp", "--mtp-depth-policy", "dynamic"});
        int tags = 0;
        for (const auto &definition : benchmarkDefinitions())
            for (const auto &cell : expandModelParityDefinition(definition))
            {
                if (!cell.e2e_certification) continue;
                SCOPED_TRACE(cell.testName());
                ++tags;
                const auto args = modelParityBenchmarkArguments(cell);
                const auto config = parseBenchmarkIntent(args);
                const auto intent = resolveOrchestrationIntent(config);
                ASSERT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(intent));
                std::vector<DeviceType> devices;
                for (const auto &participant : cell.topology.participants)
                    devices.push_back(participant.address.device_type);
                const auto &automatic = std::get<AutomaticOrchestrationRequest>(intent);
                EXPECT_TRUE(automatic.allows(modelParityAutomaticStrategy(cell), devices));
                devices.pop_back();
                EXPECT_FALSE(automatic.allows(modelParityAutomaticStrategy(cell), devices));
                EXPECT_EQ(config.activation_precision, "fp32");
                EXPECT_EQ(config.kv_cache_precision, "fp16");
                EXPECT_TRUE(config.prefix_cache.enabled);
                EXPECT_TRUE(config.mtp.enabled);
                EXPECT_EQ(config.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
                EXPECT_EQ(config.mtp.verify_mode, defaults.mtp.verify_mode);
                EXPECT_EQ(config.mtp.draft_tokens, defaults.mtp.draft_tokens);
                EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, defaults.mtp.graph_capacity_draft_tokens);
                EXPECT_EQ(config.mtp.depth_policy.initial_depth, defaults.mtp.depth_policy.initial_depth);
                EXPECT_EQ(config.mtp.depth_policy.min_depth, defaults.mtp.depth_policy.min_depth);
                EXPECT_EQ(config.mtp.depth_policy.max_depth, defaults.mtp.depth_policy.max_depth);
                EXPECT_EQ(config.mtp.depth_policy.window_size, defaults.mtp.depth_policy.window_size);
                EXPECT_EQ(config.mtp.depth_policy.min_samples, defaults.mtp.depth_policy.min_samples);
                EXPECT_EQ(config.mtp.depth_policy.cooldown_steps, defaults.mtp.depth_policy.cooldown_steps);
                EXPECT_EQ(config.mtp.depth_policy.promote_consecutive_windows,
                          defaults.mtp.depth_policy.promote_consecutive_windows);
                EXPECT_EQ(config.moe_rebalance.window_size, defaults.moe_rebalance.window_size);
                EXPECT_EQ(config.moe_rebalance.migration_transfer_slots, defaults.moe_rebalance.migration_transfer_slots);
                EXPECT_EQ(config.moe_rebalance.dynamic_max_swaps_per_layer, defaults.moe_rebalance.dynamic_max_swaps_per_layer);
                EXPECT_EQ(config.moe_rebalance.device_min_maintenance_period_tokens,
                          defaults.moe_rebalance.device_min_maintenance_period_tokens);
                EXPECT_EQ(config.moe_routed_prefill.overlay_segment_rows, defaults.moe_routed_prefill.overlay_segment_rows);

                std::ostringstream out;
                PrintTo(cell, &out);
                const auto exported = nlohmann::json::parse(out.str());
                EXPECT_EQ(exported["benchmark"]["schema"], 1);
                EXPECT_EQ(exported["benchmark"]["policy"], "production_defaults");
                EXPECT_EQ(exported["benchmark"]["args"], args);
                EXPECT_EQ(exported["benchmark"]["context_length"], cell.e2e_certification->context_length);
                EXPECT_EQ(exported["e2e"]["server_args"], modelParityServerArguments(
                    cell, ModelParityRuntimePlacementProjection::AutomaticCell));
                EXPECT_NE(exported["benchmark"]["args"], exported["e2e"]["server_args"]);
            }
        EXPECT_EQ(tags, 8);
    }

    TEST(ModelParityBenchmarkProjection, StressEconomicsCannotChangeTimingPolicy)
    {
        for (auto cell : expandModelParityDefinition(qwen36::qwen36MoECPU2NodeTPParityDefinition()))
        {
            if (!cell.e2e_certification) continue;
            const auto expected = modelParityBenchmarkArguments(cell);
            const auto stress = modelParityServerArguments(cell);
            cell.dynamic_rebalance.window_size = 123;
            cell.dynamic_rebalance.migration_transfer_slots = 49;
            cell.dynamic_rebalance.dynamic_max_swaps_per_layer = 31;
            cell.dynamic_rebalance.device_min_maintenance_period_tokens = 1024;
            cell.moe_routed_prefill.overlay_segment_rows = 64;
            cell.tp_allreduce_precision_override = "fp32";
            EXPECT_NE(modelParityServerArguments(cell), stress);
            EXPECT_EQ(modelParityBenchmarkArguments(cell), expected);
        }
    }

    TEST(ModelParityBenchmarkProjection, OffFixedRandomAndStaticIntentIsNotRelabeled)
    {
        for (auto cell : expandModelParityDefinition(qwen36::qwen36MoECPU2NodeTPParityDefinition()))
        {
            // Eligibility belongs to the definition. Exercise every selected
            // mode here without expanding the real certification membership.
            cell.e2e_certification = ModelParityE2EProfile{};
            const auto config = parseBenchmarkIntent(modelParityBenchmarkArguments(cell));
            EXPECT_EQ(config.mtp.enabled, cell.mtpEnabled());
            if (cell.mtpEnabled() && !cell.usesDynamicMTPDepth())
            {
                EXPECT_EQ(config.mtp.depth_policy.mode, MTPDepthPolicyMode::Fixed);
                EXPECT_EQ(config.mtp.draft_tokens, cell.requestedMTPDraftDepth());
            }
            ASSERT_TRUE(cell.expert_overlay);
            EXPECT_EQ(config.moe_rebalance.mode, cell.expert_overlay->movement == ModelParityExpertMovement::Dynamic
                ? MoERebalanceRuntimeMode::Dynamic : MoERebalanceRuntimeMode::Off);
            const auto args = modelParityBenchmarkArguments(cell);
            const auto owner = std::find(args.begin(), args.end(), "--moe-routed-expert-owner-order");
            ASSERT_NE(owner, args.end());
            EXPECT_EQ(*std::next(owner), cell.expert_overlay->owner_order == RoutedExpertOwnerOrder::Ordinal
                ? "ordinal" : "random");
        }
    }

    TEST(ModelParityBenchmarkProjection, UntaggedCasesCannotBecomeBenchmarks)
    {
        auto cell = expandModelParityDefinition(qwen36::qwen36MoECPU2NodeTPParityDefinition()).front();
        cell.e2e_certification.reset();
        EXPECT_THROW(modelParityBenchmarkArguments(cell), std::invalid_argument);
        std::ostringstream out;
        PrintTo(cell, &out);
        EXPECT_TRUE(nlohmann::json::parse(out.str())["benchmark"].is_null());
    }
}
