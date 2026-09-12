/**
 * @file Test__OrchestrationConfigParser.cpp
 * @brief Unit tests for OrchestrationConfigParser
 *
 * Tests:
 * - CLI parsing for simple options (--tp, --pp, --device)
 * - CLI parsing for --define-domain and --pp-stage
 * - Model and inference configuration options
 * - Sampling configuration
 * - Chat and benchmark modes
 * - Heterogeneous mode options
 * - YAML parsing for domain-based config
 * - Error handling for malformed input
 * - Validation of enum-type arguments
 * - Startup CLI publication into an already-read kernel environment snapshot
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "config/OrchestrationConfigParser.h"
#include "execution/moe/DeviceMoERebalancePolicyShared.h"
#include "utils/DebugEnv.h"

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
// Factory Function Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, CreateParser_ReturnsNonNull)
{
    auto parser = createOrchestrationConfigParser();
    EXPECT_NE(parser, nullptr);
}

// ============================================================================
// CLI Parsing - Simple Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_EmptyArgs_ReturnsDefaults)
{
    ArgvHelper args{"llaminar2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 1);
    EXPECT_EQ(config.pp_degree, 1);
    EXPECT_FALSE(config.dry_run);
    EXPECT_EQ(config.routed_expert_compute_policy, RoutedExpertComputePolicy::Apportioned);
    EXPECT_EQ(config.routed_expert_owner_order, RoutedExpertOwnerOrder::Ordinal);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Percent);
    EXPECT_FLOAT_EQ(config.moe_hot_expert_cache.percent, 10.0f);
    EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 25);
    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
    EXPECT_EQ(config.moe_rebalance.window_size, 256);
    EXPECT_EQ(
        config.moe_rebalance.migration_payoff_horizon_tokens,
        moe_rebalance_policy::kDefaultMigrationPayoffHorizonTokens);
    EXPECT_EQ(
        config.moe_rebalance.migration_transfer_slots,
        moe_rebalance_policy::kDefaultMigrationTransferSlots);
    EXPECT_FALSE(config.moe_rebalance.migration_execution_streams.has_value());
    EXPECT_EQ(
        config.moe_rebalance.resolvedMigrationExecutionStreams(),
        moe_rebalance_policy::kDefaultMigrationExecutionStreams);
    EXPECT_FALSE(config.moe_rebalance.migration_cycles_per_wave.has_value());
    EXPECT_EQ(
        config.moe_rebalance.resolvedMigrationCyclesPerWave(),
        moe_rebalance_policy::kDefaultMigrationTransferSlots);
    EXPECT_EQ(
        config.moe_rebalance.dynamic_max_swaps_per_layer,
        moe_rebalance_policy::kDefaultDynamicMaxSwapsPerLayer);
    EXPECT_EQ(
        config.moe_rebalance.dynamic_max_plan_entries_per_wave,
        moe_rebalance_policy::kDefaultDynamicMaxPlanEntriesPerWave);
    EXPECT_EQ(config.moe_routed_prefill.assignment_window_tokens, 0);
    EXPECT_EQ(config.moe_routed_prefill.least_loaded_min_routed_rows, 8192u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_numerator, 1u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_denominator, 1u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_numerator, 13u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_denominator, 10u);
    EXPECT_TRUE(config.moe_routed_prefill.llep_enable_balanced_skip);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement, 0u);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement_divisor,
              moe_rebalance_policy::kDefaultDeviceMinLoadSpreadImprovementDivisor);
    EXPECT_EQ(config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot, 256u);
    EXPECT_EQ(
        config.moe_rebalance
            .device_min_foreign_rows_per_critical_path_payload_slot,
        0u);
    EXPECT_EQ(config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot, 128u);
    EXPECT_EQ(config.moe_rebalance.device_max_post_wave_load_spread_per_mille, 100u);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_RoutedExpertOwnerOrder)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{
        "llaminar2",
        "--moe-routed-expert-owner-order", "random"};

    const auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.routed_expert_owner_order, RoutedExpertOwnerOrder::Random);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidRoutedExpertOwnerOrderThrows)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{
        "llaminar2",
        "--moe-routed-expert-owner-order", "frequency"};

    EXPECT_THROW(
        (void)parser.parseArgs(args.argc(), args.argv()),
        std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DryRun)
{
    ArgvHelper args{"llaminar2", "--dry-run"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.dry_run);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ExplainPlacement)
{
    ArgvHelper args{"llaminar2", "--explain-placement"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.explain_placement);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowTopology)
{
    ArgvHelper args{"llaminar2", "--show-topology"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_topology);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowNuma)
{
    ArgvHelper args{"llaminar2", "--show-numa"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_numa);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ValidateOnly)
{
    ArgvHelper args{"llaminar2", "--validate-only"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.validate_only);
}

// ============================================================================
// CLI Parsing - TP Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_WithSpace)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_WithEquals)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree=4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDegree_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-tp", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPScope)
{
    ArgvHelper args{"llaminar2", "--tp-scope", "rank_local"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_scope, TPScope::RANK_LOCAL);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPDevices)
{
    ArgvHelper args{"llaminar2", "--tp-devices", "cuda:0,cuda:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_devices.size(), 2);
    EXPECT_EQ(config.tp_devices[0].device_type, DeviceType::CUDA);
    EXPECT_EQ(config.tp_devices[0].device_ordinal, 0);
    EXPECT_EQ(config.tp_devices[1].device_ordinal, 1);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPWeights)
{
    ArgvHelper args{"llaminar2", "--tp-weights", "0.73,0.27"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_weights.size(), 2);
    EXPECT_FLOAT_EQ(config.tp_weights[0], 0.73f);
    EXPECT_FLOAT_EQ(config.tp_weights[1], 0.27f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPLocal)
{
    ArgvHelper args{"llaminar2", "--tp-local", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_local_degree, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPGlobal)
{
    ArgvHelper args{"llaminar2", "--tp-global", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_global_degree, 4);
}

// ============================================================================
// CLI Parsing - PP Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_PPDegree_WithSpace)
{
    ArgvHelper args{"llaminar2", "--pipeline-parallelism-degree", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_degree, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPDegree_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-pp", "3"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_degree, 3);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPSplit)
{
    ArgvHelper args{"llaminar2", "--pp-split", "weighted"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_split, PPSplitMode::WEIGHTED);
}

// ============================================================================
// CLI Parsing - Device Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Device)
{
    ArgvHelper args{"llaminar2", "--device", "cuda:0"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.device_for_this_rank->device_ordinal, 0);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-d", "rocm:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::ROCm);
    EXPECT_EQ(config.device_for_this_rank->device_ordinal, 1);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_CpuShorthand_EnablesGlobalCpuTpIntent)
{
    ArgvHelper args{"llaminar2", "-d", "cpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_for_this_rank->numa_node, 0);
    EXPECT_FALSE(config.device_for_this_rank_numa_explicit);
    EXPECT_TRUE(config.cpu_global_tp_all_local);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_CpuExplicitNuma)
{
    ArgvHelper args{"llaminar2", "-d", "cpu:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_for_this_rank->numa_node, 1);
    EXPECT_TRUE(config.device_for_this_rank_numa_explicit);
    EXPECT_FALSE(config.cpu_global_tp_all_local);
}

/** Serialized unknown NUMA must remain unresolved in every configuration surface. */
TEST(Test__OrchestrationConfigParser, DeviceNumaIntentSurvivesAddressSerialization)
{
    OrchestrationConfigParser parser;
    for (const auto *backend : {"cpu", "cuda", "rocm"})
    {
        for (const int numa : {-1, 0, 1})
        {
            const auto address = std::string("localhost:") +
                std::to_string(numa) + ":" + backend + ":0";
            SCOPED_TRACE(address);
            ArgvHelper args{"llaminar2", "--device", address.c_str()};
            const auto cli = parser.parseArgs(args.argc(), args.argv());
            const auto yaml = parser.parseYamlString("device: " + address + "\n");
            const auto map = "0=" + address;
            ArgvHelper mapped_args{"llaminar2", "--device-map", map.c_str()};
            const auto mapped = parser.parseArgs(mapped_args.argc(), mapped_args.argv());
            for (const auto *config : {&cli, &yaml})
            {
                ASSERT_TRUE(config->device_for_this_rank);
                EXPECT_EQ(config->device_for_this_rank->numa_node, numa);
                EXPECT_EQ(config->device_for_this_rank_numa_explicit, numa >= 0);
                EXPECT_FALSE(config->cpu_global_tp_all_local);
            }
            ASSERT_EQ(mapped.device_map_numa_explicit.size(), 1u);
            EXPECT_EQ(mapped.device_map_numa_explicit.front().second, numa >= 0);
        }
    }
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DeviceMode)
{
    ArgvHelper args{"llaminar2", "--device-mode", "round_robin"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::ROUND_ROBIN);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DeviceMap)
{
    ArgvHelper args{"llaminar2", "--device-map", "0=cuda:0,1=cuda:1,2=rocm:0"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::EXPLICIT);
    EXPECT_EQ(config.device_map.size(), 3);
    EXPECT_EQ(config.device_map[0].first, 0);
    EXPECT_EQ(config.device_map[0].second.device_type, DeviceType::CUDA);
    EXPECT_EQ(config.device_map[1].first, 1);
    EXPECT_EQ(config.device_map[2].first, 2);
    EXPECT_EQ(config.device_map[2].second.device_type, DeviceType::ROCm);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DeviceMap_CpuEntriesTrackNumaExplicitness)
{
    ArgvHelper args{"llaminar2", "--device-map", "0=cpu,1=cpu:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_EQ(config.device_map.size(), 2);
    ASSERT_EQ(config.device_map_numa_explicit.size(), 2);

    EXPECT_EQ(config.device_map[0].first, 0);
    EXPECT_EQ(config.device_map[0].second.device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_map[0].second.numa_node, 0);
    EXPECT_FALSE(config.device_map_numa_explicit[0].second);

    EXPECT_EQ(config.device_map[1].first, 1);
    EXPECT_EQ(config.device_map[1].second.device_type, DeviceType::CPU);
    EXPECT_EQ(config.device_map[1].second.numa_node, 1);
    EXPECT_TRUE(config.device_map_numa_explicit[1].second);
}

// ============================================================================
// CLI Parsing - Named Domains
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_DefineDomain)
{
    ArgvHelper args{"llaminar2", "--define-domain", "gpu_tp=cuda:0,cuda:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.domain_definitions.size(), 1);
    EXPECT_EQ(config.domain_definitions[0].name, "gpu_tp");
    EXPECT_EQ(config.domain_definitions[0].devices.size(), 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DefineDomain_Multiple)
{
    ArgvHelper args{"llaminar2",
                    "--define-domain", "fast=cuda:0,cuda:1",
                    "--define-domain", "slow=rocm:0,rocm:1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.domain_definitions.size(), 2);
    EXPECT_EQ(config.domain_definitions[0].name, "fast");
    EXPECT_EQ(config.domain_definitions[1].name, "slow");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DefineDomain_WithWeightsAndBackend)
{
    ArgvHelper args{"llaminar2",
                    "--define-domain", "mixed=cuda:0,rocm:0;weights=0.6,0.4;backend=heterogeneous"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.domain_definitions.size(), 1);
    auto &domain = config.domain_definitions[0];
    EXPECT_EQ(domain.name, "mixed");
    EXPECT_EQ(domain.weights.size(), 2);
    EXPECT_FLOAT_EQ(domain.weights[0], 0.6f);
    EXPECT_EQ(domain.backend, CollectiveBackendType::HETEROGENEOUS);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_DefineDomain_WithScopeOwnerRanksBackend)
{
    ArgvHelper args{"llaminar2",
                    "--define-domain", "rocm_socket0=0:rocm:0,0:rocm:1;scope=rank_local;backend=rccl;owner=0",
                    "--define-domain", "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_EQ(config.domain_definitions.size(), 2u);
    EXPECT_EQ(config.domain_definitions[0].scope, TPScope::RANK_LOCAL);
    ASSERT_TRUE(config.domain_definitions[0].owner_rank.has_value());
    EXPECT_EQ(*config.domain_definitions[0].owner_rank, 0);
    EXPECT_EQ(config.domain_definitions[0].backend, CollectiveBackendType::RCCL);

    EXPECT_EQ(config.domain_definitions[1].scope, TPScope::NODE_LOCAL);
    EXPECT_EQ(config.domain_definitions[1].backend, CollectiveBackendType::UPI);
    ASSERT_EQ(config.domain_definitions[1].explicit_ranks.size(), 2u);
    EXPECT_EQ(config.domain_definitions[1].explicit_ranks[0], 0);
    EXPECT_EQ(config.domain_definitions[1].explicit_ranks[1], 1);
}

TEST(Test__OrchestrationConfigParser,
     ParseArgs_MoEAutoScopeDefersRankOwnershipToInventoryBinding)
{
    ArgvHelper args{
        "llaminar2",
        "--moe-routed-expert-placement", "tiered-overlay",
        "--moe-routed-expert-continuation-domain", "portable_gpu_pool",
        "--moe-routed-expert-shared-domain", "portable_gpu_pool",
        "--moe-routed-expert-domain",
        "portable_gpu_pool=cuda:0,cuda:1;scope=auto;backend=nccl;routed_compute=apportioned",
        "--moe-routed-expert-tier",
        "priority_0@portable_gpu_pool;priority=0",
    };
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    ASSERT_EQ(config.moe_routed_expert_plan->domains.size(), 1u);
    const auto &domain = config.moe_routed_expert_plan->domains.front();
    EXPECT_EQ(domain.scope, ExecutionDomainScope::AUTO);
    EXPECT_EQ(domain.owner_rank, -1);
    EXPECT_TRUE(domain.world_ranks.empty());
}

TEST(Test__OrchestrationConfigParser, Phase9B_NamedAndOverlayDomainsShareCanonicalNormalization)
{
    OrchestrationConfigParser parser;
    ArgvHelper named_args{"llaminar2",
                          "--define-domain", "rocm_hot=0:rocm:0,0:rocm:1;weights=0.60,0.40;scope=rank_local;backend=rccl;owner=0"};
    ArgvHelper overlay_args{"llaminar2",
                            "--moe-routed-expert-placement", "tiered-overlay",
                            "--moe-routed-expert-continuation-domain", "rocm_hot",
                            "--moe-routed-expert-shared-domain", "rocm_hot",
                            "--moe-routed-expert-domain", "rocm_hot=0:rocm:0,0:rocm:1;weights=0.60,0.40;scope=rank_local;backend=rccl;routed_compute=apportioned;owner=0",
                            "--moe-routed-expert-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;routed_compute=apportioned;ranks=0,1",
                            "--moe-routed-expert-tier", "hot@rocm_hot;priority=0",
                            "--moe-routed-expert-tier", "cold@cpu_cold;priority=1"};

    const auto named_config = parser.parseArgs(named_args.argc(), named_args.argv());
    const auto overlay_config = parser.parseArgs(overlay_args.argc(), overlay_args.argv());

    ASSERT_EQ(named_config.domain_definitions.size(), 1u);
    ASSERT_NE(overlay_config.moe_routed_expert_plan, nullptr);
    ASSERT_EQ(overlay_config.moe_routed_expert_plan->domains.size(), 2u);

    const auto named_domain = named_config.domain_definitions[0].toExecutionDomainDefinition();
    const auto overlay_domain = overlay_config.moe_routed_expert_plan->domains[0].toExecutionDomainDefinition();

    EXPECT_EQ(named_domain.name, overlay_domain.name);
    EXPECT_EQ(named_domain.participants, overlay_domain.participants);
    EXPECT_EQ(named_domain.weights, overlay_domain.weights);
    EXPECT_EQ(named_domain.scope, overlay_domain.scope);
    EXPECT_EQ(named_domain.backend, overlay_domain.backend);
    EXPECT_EQ(named_domain.owner_rank, overlay_domain.owner_rank);
    EXPECT_EQ(named_domain.ranks, overlay_domain.ranks);
    EXPECT_EQ(named_domain.routed_compute_policy, RoutedExpertComputePolicy::Unspecified);
    EXPECT_EQ(overlay_domain.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);

    const auto inventory = overlay_config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 2u);
    EXPECT_EQ(inventory[0].logicalIdentity(), "rocm_hot");
    EXPECT_EQ(inventory[1].logicalIdentity(), "cpu_cold");
    EXPECT_FALSE(overlay_config.moe_routed_expert_plan->routed_tiers[0].fallback);
    EXPECT_TRUE(overlay_config.moe_routed_expert_plan->routed_tiers[1].fallback);
}

TEST(Test__OrchestrationConfigParser, Phase9B_OverlayDenseTPOptIn)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--moe-routed-expert-placement", "tiered-overlay",
                    "--moe-routed-expert-continuation-domain", "cuda_hot",
                    "--moe-routed-expert-base-model-domain", "cuda_hot",
                    "--moe-routed-expert-shared-domain", "cuda_hot",
                    "--moe-continuation-dense-tp", "true",
                    "--moe-continuation-dense-decode-replicated", "true",
                    "--moe-routed-expert-domain", "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;routed_compute=apportioned;owner=0",
                    "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    EXPECT_TRUE(config.moe_routed_expert_plan->continuation_domain_spec.dense_tp_enabled);
    EXPECT_TRUE(config.moe_routed_expert_plan->continuation_domain_spec.dense_decode_replicated);
    EXPECT_EQ(config.moe_routed_expert_plan->continuation_domain_spec.dense_policy,
              DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
    EXPECT_EQ(config.moe_routed_expert_plan->continuation_domain_spec.effectiveDensePolicy(),
              DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
    EXPECT_EQ(config.moe_routed_expert_plan->continuation_domain, "cuda_hot");

    const auto inventory = config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 1u);
    EXPECT_EQ(inventory[0].logicalIdentity(), "cuda_hot");
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDensePolicyNamesPhaseSplitHybrid)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--moe-routed-expert-placement", "tiered-overlay",
                    "--moe-routed-expert-continuation-domain", "cuda_hot",
                    "--moe-routed-expert-base-model-domain", "cuda_hot",
                    "--moe-routed-expert-shared-domain", "cuda_hot",
                    "--moe-continuation-dense-policy", "prefill-tensor-parallel-decode-replicated",
                    "--moe-routed-expert-domain", "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;routed_compute=apportioned;owner=0",
                    "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    const auto &spec = config.moe_routed_expert_plan->continuation_domain_spec;
    EXPECT_EQ(spec.dense_policy, DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
    EXPECT_EQ(spec.effectiveDensePolicy(), DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
    EXPECT_TRUE(spec.dense_tp_enabled);
    EXPECT_TRUE(spec.dense_decode_replicated);

    const auto inventory = config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 1u);
    EXPECT_EQ(inventory[0].routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDensePolicyNamesDecodeMirroredEmbedding)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--moe-routed-expert-placement", "tiered-overlay",
                    "--moe-routed-expert-continuation-domain", "cuda_hot",
                    "--moe-routed-expert-base-model-domain", "cuda_hot",
                    "--moe-routed-expert-shared-domain", "cuda_hot",
                    "--moe-continuation-dense-policy", "tensor-parallel-decode-mirrored-embedding",
                    "--moe-routed-expert-domain", "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;routed_compute=apportioned;owner=0",
                    "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    const auto &spec = config.moe_routed_expert_plan->continuation_domain_spec;
    EXPECT_EQ(spec.dense_policy, DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding);
    EXPECT_EQ(spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding);
    EXPECT_TRUE(spec.dense_tp_enabled);
    EXPECT_FALSE(spec.dense_decode_replicated);
    EXPECT_TRUE(spec.dense_decode_mirrored_embedding);
}

TEST(Test__OrchestrationConfigParser, MoEExecutionPolicyKeepsAllAxesIndependent)
{
    const auto policy = makeMoEExecutionPolicy(
        DenseParallelPolicy::PrefillTensorParallelDecodeReplicated,
        RoutedExpertComputePolicy::Apportioned,
        RoutedExpertAssignmentPolicy::StaticOwner,
        RoutedExpertAssignmentPolicy::LeastLoadedResident);

    EXPECT_EQ(
        policy.dense,
        DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
    EXPECT_EQ(policy.routed_compute, RoutedExpertComputePolicy::Apportioned);
    EXPECT_EQ(policy.routed_phase, RoutedExpertPhasePolicy::Uniform);
    EXPECT_EQ(
        policy.routed_decode_assignment,
        RoutedExpertAssignmentPolicy::StaticOwner);
    EXPECT_EQ(
        policy.routed_prefill_assignment,
        RoutedExpertAssignmentPolicy::LeastLoadedResident);
}

TEST(Test__OrchestrationConfigParser, MoEExecutionPolicyDescriptionNamesEveryAxis)
{
    const auto policy = makeMoEExecutionPolicy(
        DenseParallelPolicy::TensorParallelDecodeMirroredEmbedding,
        RoutedExpertComputePolicy::TensorSharded,
        RoutedExpertAssignmentPolicy::StaticOwner,
        RoutedExpertAssignmentPolicy::StaticOwner);

    EXPECT_EQ(
        describeMoEExecutionPolicy(policy),
        "dense=tensor-parallel-decode-mirrored-embedding,"
        "routed_compute=tensor-sharded,routed_phase=uniform,"
        "routed_decode_assignment=static-owner,"
        "routed_prefill_assignment=static-owner");
}

TEST(Test__OrchestrationConfigParser, AmbiguousCompositePolicyAliasesAreRejected)
{
    EXPECT_FALSE(parseDenseParallelPolicy("phase-split-hybrid-tp-ae").has_value());
    EXPECT_FALSE(parseDenseParallelPolicy("tp").has_value());
    EXPECT_FALSE(parseDenseParallelPolicy("dense-tp").has_value());
    EXPECT_FALSE(parseDenseParallelPolicy("full").has_value());
    EXPECT_FALSE(parseDenseParallelPolicy("decode-mirrored-embedding").has_value());
    EXPECT_FALSE(parseRoutedExpertComputePolicy("apportioned-experts").has_value());
    EXPECT_FALSE(parseRoutedExpertComputePolicy("expert-parallel").has_value());
    EXPECT_FALSE(parseRoutedExpertPhasePolicy("hybrid").has_value());
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDomainParsesPhaseSplitIndependently)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{
        "llaminar2",
        "--moe-routed-expert-placement", "tiered-overlay",
        "--moe-routed-expert-continuation-domain", "cuda_hot",
        "--moe-routed-expert-shared-domain", "cuda_hot",
        "--moe-routed-expert-domain",
        "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;"
        "routed_compute=replicated;"
        "routed_phase=prefill-apportioned-decode-replicated;"
        "routed_decode_assignment=static-owner;"
        "routed_prefill_assignment=static-owner;owner=0",
        "--moe-routed-expert-tier",
        "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    ASSERT_EQ(config.moe_routed_expert_plan->domains.size(), 1u);
    const auto &domain = config.moe_routed_expert_plan->domains.front();
    EXPECT_EQ(
        domain.routed_compute_policy,
        RoutedExpertComputePolicy::Replicated);
    EXPECT_EQ(
        domain.routed_phase_policy,
        RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated);
    EXPECT_EQ(
        domain.routed_decode_assignment_policy,
        RoutedExpertAssignmentPolicy::StaticOwner);
    EXPECT_EQ(
        domain.routed_prefill_assignment_policy,
        RoutedExpertAssignmentPolicy::StaticOwner);

    const auto inventory = config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 1u);
    EXPECT_EQ(
        inventory.front().routed_phase_policy,
        RoutedExpertPhasePolicy::PrefillApportionedDecodeReplicated);
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDomainRejectsPhaseSplitWithoutReplicas)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{
        "llaminar2",
        "--moe-routed-expert-placement", "tiered-overlay",
        "--moe-routed-expert-continuation-domain", "cuda_hot",
        "--moe-routed-expert-shared-domain", "cuda_hot",
        "--moe-routed-expert-domain",
        "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;"
        "routed_compute=apportioned;"
        "routed_phase=prefill-apportioned-decode-replicated;owner=0",
        "--moe-routed-expert-tier",
        "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDomainParsesDecodeAndPrefillAssignmentIndependently)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--moe-routed-expert-placement", "tiered-overlay",
                    "--moe-routed-expert-continuation-domain", "cuda_hot",
                    "--moe-routed-expert-shared-domain", "cuda_hot",
                    "--moe-routed-expert-domain", "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;routed_compute=apportioned;routed_decode_assignment=static-owner;routed_prefill_assignment=least-loaded-resident;owner=0",
                    "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    ASSERT_EQ(config.moe_routed_expert_plan->domains.size(), 1u);
    EXPECT_EQ(config.moe_routed_expert_plan->domains[0].routed_compute_policy,
              RoutedExpertComputePolicy::Apportioned);
    EXPECT_EQ(config.moe_routed_expert_plan->domains[0].routed_decode_assignment_policy,
              RoutedExpertAssignmentPolicy::StaticOwner);
    EXPECT_EQ(config.moe_routed_expert_plan->domains[0].routed_prefill_assignment_policy,
              RoutedExpertAssignmentPolicy::LeastLoadedResident);

    const auto inventory = config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 1u);
    EXPECT_EQ(inventory[0].routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
    EXPECT_EQ(inventory[0].routed_decode_assignment_policy,
              RoutedExpertAssignmentPolicy::StaticOwner);
    EXPECT_EQ(inventory[0].routed_prefill_assignment_policy,
              RoutedExpertAssignmentPolicy::LeastLoadedResident);
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDomainRejectsAmbiguousAllWorkAssignment)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{
        "llaminar2",
        "--moe-routed-expert-placement", "tiered-overlay",
        "--moe-routed-expert-continuation-domain", "cuda_hot",
        "--moe-routed-expert-shared-domain", "cuda_hot",
        "--moe-routed-expert-domain",
        "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;"
        "routed_compute=apportioned;"
        "routed_assignment=least-loaded-resident;owner=0",
        "--moe-routed-expert-tier",
        "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    EXPECT_THROW(
        parser.parseArgs(args.argc(), args.argv()),
        std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDomainRejectsLeastLoadedAsComputeKind)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--moe-routed-expert-placement", "tiered-overlay",
                    "--moe-routed-expert-continuation-domain", "cuda_hot",
                    "--moe-routed-expert-shared-domain", "cuda_hot",
                    "--moe-routed-expert-domain", "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;routed_compute=least-loaded-resident;owner=0",
                    "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, MoEOverlayDomainRejectsLegacyExpertParallelComputeAlias)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--moe-routed-expert-placement", "tiered-overlay",
                    "--moe-routed-expert-continuation-domain", "cuda_hot",
                    "--moe-routed-expert-shared-domain", "cuda_hot",
                    "--moe-routed-expert-domain", "cuda_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;routed_compute=expert_parallel;owner=0",
                    "--moe-routed-expert-tier", "hot@cuda_hot;priority=0;max-experts-per-layer=256"};

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, Phase9B_DomainIdentityIsNameScopedForSharedParticipants)
{
    const auto first = ExecutionDomainDefinition::parse(
        "continuation=0:cuda:0;scope=single;backend=auto;routed_compute=apportioned");
    const auto second = ExecutionDomainDefinition::parse(
        "shared_experts=0:cuda:0;scope=single;backend=auto;routed_compute=apportioned");

    EXPECT_TRUE(first.samePhysicalParticipants(second));
    EXPECT_NE(first.logicalIdentity(), second.logicalIdentity());
}

TEST(Test__OrchestrationConfigParser, Phase9B_PPStageRemainsLayerPlacementNotMoEOverlay)
{
    ArgvHelper args{"llaminar2",
                    "--define-domain", "gpu_tp=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;owner=0",
                    "--pp-stage", "0=gpu_tp:0-3"};
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_EQ(config.domain_definitions.size(), 1u);
    ASSERT_EQ(config.pp_stage_definitions.size(), 1u);
    EXPECT_EQ(config.pp_stage_definitions[0].domain_name, "gpu_tp");
    EXPECT_EQ(config.pp_stage_definitions[0].first_layer, 0);
    EXPECT_EQ(config.pp_stage_definitions[0].last_layer, 3);
    EXPECT_EQ(config.moe_routed_expert_plan, nullptr);

    const auto inventory = config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 1u);
    EXPECT_EQ(inventory[0].logicalIdentity(), "gpu_tp");
    EXPECT_EQ(inventory[0].routed_compute_policy, RoutedExpertComputePolicy::Unspecified);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPStage)
{
    ArgvHelper args{"llaminar2", "--pp-stage", "0=gpu_tp:0-13"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_stage_definitions.size(), 1);
    EXPECT_EQ(config.pp_stage_definitions[0].stage_id, 0);
    EXPECT_EQ(config.pp_stage_definitions[0].domain_name, "gpu_tp");
    EXPECT_EQ(config.pp_stage_definitions[0].first_layer, 0);
    EXPECT_EQ(config.pp_stage_definitions[0].last_layer, 13);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PPStage_Multiple)
{
    ArgvHelper args{"llaminar2",
                    "--pp-stage", "0=stage0:0-13",
                    "--pp-stage", "1=stage1:14-27"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.pp_stage_definitions.size(), 2);
    EXPECT_EQ(config.pp_stage_definitions[0].stage_id, 0);
    EXPECT_EQ(config.pp_stage_definitions[1].stage_id, 1);
}

// ============================================================================
// CLI Parsing - Layer Placement
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_CPULayers)
{
    ArgvHelper args{"llaminar2", "--cpu-layers", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.cpu_layers, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_CPULayersFirst)
{
    ArgvHelper args{"llaminar2", "--cpu-layers", "4", "--cpu-layers-first"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.cpu_layers, 4);
    EXPECT_TRUE(config.cpu_layers_first);
}

// ============================================================================
// CLI Parsing - Backend
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Backend)
{
    ArgvHelper args{"llaminar2", "--backend", "nccl"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.default_backend, CollectiveBackendType::NCCL);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Backend_ShortFlag)
{
    ArgvHelper args{"llaminar2", "-b", "rccl"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.default_backend, CollectiveBackendType::RCCL);
}

// ============================================================================
// CLI Parsing - Combined Options
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ComplexConfig)
{
    ArgvHelper args{"llaminar2",
                    "--tensor-parallelism-degree", "2",
                    "--pipeline-parallelism-degree", "2",
                    "--device", "cuda:0",
                    "--cpu-layers", "2",
                    "--backend", "nccl",
                    "--dry-run"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_degree, 2);
    EXPECT_EQ(config.pp_degree, 2);
    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.cpu_layers, 2);
    EXPECT_EQ(config.default_backend, CollectiveBackendType::NCCL);
    EXPECT_TRUE(config.dry_run);
}

// ============================================================================
// CLI Parsing - Error Handling
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MissingTPValue_Throws)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidTPValue_Throws)
{
    ArgvHelper args{"llaminar2", "--tensor-parallelism-degree", "abc"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidDevice_Throws)
{
    ArgvHelper args{"llaminar2", "--device", "invalid:device"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidTPScope_Throws)
{
    ArgvHelper args{"llaminar2", "--tp-scope", "invalid"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidDomainDefinition_Throws)
{
    ArgvHelper args{"llaminar2", "--define-domain", "invalid"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidPPStage_Throws)
{
    ArgvHelper args{"llaminar2", "--pp-stage", "invalid"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

// ============================================================================
// YAML Parsing Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseYamlString_SimpleConfig)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
tp_degree: 4
pp_degree: 2
dry_run: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_degree, 4);
    EXPECT_EQ(config.pp_degree, 2);
    EXPECT_TRUE(config.dry_run);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_TPConfig)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
tp_degree: 2
tp_scope: rank_local
tp_devices: [cuda:0, cuda:1]
tp_weights: [0.73, 0.27]
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_degree, 2);
    EXPECT_EQ(config.tp_scope, TPScope::RANK_LOCAL);
    EXPECT_EQ(config.tp_devices.size(), 2);
    EXPECT_EQ(config.tp_weights.size(), 2);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_TPAllreducePrecision)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
tp_allreduce_precision: bf16
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_allreduce_precision_override, "bf16");
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_NamedDomainLists)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
domains:
    - "rocm_socket0=0:rocm:0,0:rocm:1;scope=rank_local;backend=rccl;owner=0"
    - "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"
pp_stages:
    - "0=rocm_socket0:0-13"
    - "1=cpu_sockets:14-27"
)";

    auto config = parser.parseYamlString(yaml);

    ASSERT_EQ(config.domain_definitions.size(), 2u);
    EXPECT_EQ(config.domain_definitions[0].name, "rocm_socket0");
    EXPECT_EQ(config.domain_definitions[0].scope, TPScope::RANK_LOCAL);
    ASSERT_TRUE(config.domain_definitions[0].owner_rank.has_value());
    EXPECT_EQ(*config.domain_definitions[0].owner_rank, 0);
    EXPECT_EQ(config.domain_definitions[0].backend, CollectiveBackendType::RCCL);
    EXPECT_EQ(config.domain_definitions[1].scope, TPScope::NODE_LOCAL);
    ASSERT_EQ(config.domain_definitions[1].explicit_ranks.size(), 2u);
    EXPECT_EQ(config.domain_definitions[1].explicit_ranks[1], 1);

    ASSERT_EQ(config.pp_stage_definitions.size(), 2u);
    EXPECT_EQ(config.pp_stage_definitions[0].domain_name, "rocm_socket0");
    EXPECT_EQ(config.pp_stage_definitions[1].domain_name, "cpu_sockets");
    EXPECT_EQ(config.pp_stage_definitions[1].first_layer, 14);
    EXPECT_EQ(config.pp_stage_definitions[1].last_layer, 27);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_AllIntrospectionFlags)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
dry_run: true
explain_placement: true
show_topology: true
show_numa: true
validate_only: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.dry_run);
    EXPECT_TRUE(config.explain_placement);
    EXPECT_TRUE(config.show_topology);
    EXPECT_TRUE(config.show_numa);
    EXPECT_TRUE(config.validate_only);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_DeviceConfig)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
device: cuda:0
device_mode: explicit
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.device_mode, DeviceAssignmentMode::EXPLICIT);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_LayerPlacement)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
cpu_layers: 4
cpu_layers_first: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.cpu_layers, 4);
    EXPECT_TRUE(config.cpu_layers_first);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_Backend)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
backend: nccl
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.default_backend, CollectiveBackendType::NCCL);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_CommentsIgnored)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
# This is a comment
tp_degree: 2
# Another comment
pp_degree: 3
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_degree, 2);
    EXPECT_EQ(config.pp_degree, 3);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_EmptyString_ReturnsDefaults)
{
    OrchestrationConfigParser parser;

    auto config = parser.parseYamlString("");

    EXPECT_EQ(config.tp_degree, 1);
    EXPECT_EQ(config.pp_degree, 1);
    EXPECT_EQ(config.routed_expert_compute_policy, RoutedExpertComputePolicy::Apportioned);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Percent);
    EXPECT_FLOAT_EQ(config.moe_hot_expert_cache.percent, 10.0f);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_MoENestedBlock)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
moe:
    routed_expert_compute_policy: replicated
    routed_expert_owner_order: random
    hot_expert_cache: 12
    residency_maintenance: observe
    residency_maintenance_window: 64
    residency_maintenance_max_window: 512
    residency_maintenance_window_growth: 2.5
    migration_payoff_horizon_tokens: 16384
    migration_transfer_slots: 3
    migration_execution_streams: 2
    migration_cycles_per_wave: 2
    routed_prefill_assignment_window_tokens: 96
    overlay_prefill_segment_rows: 320
    routed_prefill_least_loaded_min_routed_rows: 2048
    routed_prefill_llep_alpha_numerator: 3
    routed_prefill_llep_alpha_denominator: 4
    routed_prefill_llep_lambda_numerator: 7
    routed_prefill_llep_lambda_denominator: 6
    routed_prefill_llep_enable_balanced_skip: false
    dynamic_imbalance_threshold_permille: 1125
    dynamic_min_improvement_permille: 20
    dynamic_max_swaps_per_layer: 6
    dynamic_max_plan_entries_per_wave: 24
    dynamic_min_window_activations: 32
    device_rebalance_maintenance_slack_tokens: 3
    device_rebalance_min_maintenance_period_tokens: 67
    device_rebalance_initial_maintenance_period_tokens: 35
    device_min_load_spread_improvement: 44
    device_min_load_spread_improvement_divisor: 15
    device_min_wave_spread_improvement_per_payload_slot: 192
    device_min_foreign_rows_per_critical_path_payload_slot: 320
    device_min_router_spread_improvement_per_payload_slot: 384
    device_max_post_wave_load_spread_permille: 75
    release_raw_expert_weights: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.routed_expert_compute_policy, RoutedExpertComputePolicy::Replicated);
    EXPECT_EQ(config.routed_expert_owner_order, RoutedExpertOwnerOrder::Random);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Count);
    EXPECT_EQ(config.moe_hot_expert_cache.count, 12);
    EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 12);
    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Observe);
    EXPECT_EQ(config.moe_rebalance.window_size, 64);
    EXPECT_EQ(config.moe_rebalance.max_window_size, 512);
    EXPECT_FLOAT_EQ(config.moe_rebalance.window_growth_factor, 2.5f);
    EXPECT_EQ(
        config.moe_rebalance.migration_payoff_horizon_tokens,
        16'384u);
    EXPECT_EQ(config.moe_rebalance.migration_transfer_slots, 3u);
    ASSERT_TRUE(config.moe_rebalance.migration_execution_streams.has_value());
    EXPECT_EQ(*config.moe_rebalance.migration_execution_streams, 2u);
    EXPECT_EQ(
        config.moe_rebalance.resolvedMigrationExecutionStreams(),
        2u);
    ASSERT_TRUE(config.moe_rebalance.migration_cycles_per_wave.has_value());
    EXPECT_EQ(*config.moe_rebalance.migration_cycles_per_wave, 2u);
    EXPECT_EQ(config.moe_rebalance.resolvedMigrationCyclesPerWave(), 2u);
    EXPECT_EQ(config.moe_routed_prefill.assignment_window_tokens, 96);
    EXPECT_EQ(config.moe_routed_prefill.overlay_segment_rows, 320);
    EXPECT_EQ(config.moe_routed_prefill.least_loaded_min_routed_rows, 2048u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_numerator, 3u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_denominator, 4u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_numerator, 7u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_denominator, 6u);
    EXPECT_FALSE(config.moe_routed_prefill.llep_enable_balanced_skip);
    EXPECT_EQ(config.moe_rebalance.dynamic_imbalance_threshold_per_mille, 1125u);
    EXPECT_EQ(config.moe_rebalance.dynamic_min_improvement_per_mille, 20u);
    EXPECT_EQ(config.moe_rebalance.dynamic_max_swaps_per_layer, 6u);
    EXPECT_EQ(config.moe_rebalance.dynamic_max_plan_entries_per_wave, 24u);
    EXPECT_EQ(config.moe_rebalance.dynamic_min_window_activations, 32u);
    EXPECT_EQ(config.moe_rebalance.device_maintenance_slack_tokens, 3);
    EXPECT_EQ(config.moe_rebalance.device_min_maintenance_period_tokens, 67);
    EXPECT_EQ(config.moe_rebalance.device_initial_maintenance_period_tokens, 35);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement, 44u);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement_divisor, 15u);
    EXPECT_EQ(config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot, 192u);
    EXPECT_EQ(
        config.moe_rebalance
            .device_min_foreign_rows_per_critical_path_payload_slot,
        320u);
    EXPECT_EQ(config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot, 384u);
    EXPECT_EQ(config.moe_rebalance.device_max_post_wave_load_spread_per_mille, 75u);
    EXPECT_TRUE(config.moe_rebalance.release_raw_expert_weights);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_MoEFlatKeys)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
routed_expert_compute_policy: apportioned
moe_hot_expert_cache: 25%
moe_residency_maintenance: off
moe_residency_maintenance_window: 128
moe_residency_maintenance_max_window: 1024
moe_residency_maintenance_window_growth: 1.25
moe_migration_payoff_horizon_tokens: 32768
moe_migration_transfer_slots: 4
moe_migration_execution_streams: 2
moe_migration_cycles_per_wave: 3
moe_routed_prefill_assignment_window_tokens: 192
moe_overlay_prefill_segment_rows: 384
moe_routed_prefill_least_loaded_min_routed_rows: 4096
moe_routed_prefill_llep_alpha_numerator: 5
moe_routed_prefill_llep_alpha_denominator: 8
moe_routed_prefill_llep_lambda_numerator: 9
moe_routed_prefill_llep_lambda_denominator: 7
moe_routed_prefill_llep_enable_balanced_skip: false
moe_dynamic_imbalance_threshold_permille: 1050
moe_dynamic_min_improvement_permille: 0
moe_dynamic_max_swaps_per_layer: 8
moe_dynamic_max_plan_entries_per_wave: 32
moe_dynamic_min_window_activations: 16
moe_device_rebalance_maintenance_slack_tokens: 5
moe_device_rebalance_min_maintenance_period_tokens: 133
moe_device_rebalance_initial_maintenance_period_tokens: 69
moe_device_rebalance_min_load_spread_improvement: 64
moe_device_rebalance_min_load_spread_improvement_divisor: 20
moe_device_rebalance_min_wave_spread_improvement_per_payload_slot: 256
moe_device_rebalance_min_foreign_rows_per_critical_path_payload_slot: 640
moe_device_rebalance_min_router_spread_improvement_per_payload_slot: 512
moe_device_rebalance_max_post_wave_load_spread_permille: 80
moe_release_raw_expert_weights: false
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.routed_expert_compute_policy, RoutedExpertComputePolicy::Apportioned);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Percent);
    EXPECT_FLOAT_EQ(config.moe_hot_expert_cache.percent, 25.0f);
    EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 64);
    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Off);
    EXPECT_EQ(config.moe_rebalance.window_size, 128);
    EXPECT_EQ(config.moe_rebalance.max_window_size, 1024);
    EXPECT_FLOAT_EQ(config.moe_rebalance.window_growth_factor, 1.25f);
    EXPECT_EQ(
        config.moe_rebalance.migration_payoff_horizon_tokens,
        32'768u);
    EXPECT_EQ(config.moe_rebalance.migration_transfer_slots, 4u);
    ASSERT_TRUE(config.moe_rebalance.migration_execution_streams.has_value());
    EXPECT_EQ(*config.moe_rebalance.migration_execution_streams, 2u);
    EXPECT_EQ(
        config.moe_rebalance.resolvedMigrationExecutionStreams(),
        2u);
    ASSERT_TRUE(config.moe_rebalance.migration_cycles_per_wave.has_value());
    EXPECT_EQ(*config.moe_rebalance.migration_cycles_per_wave, 3u);
    EXPECT_EQ(config.moe_rebalance.resolvedMigrationCyclesPerWave(), 3u);
    EXPECT_EQ(config.moe_routed_prefill.assignment_window_tokens, 192);
    EXPECT_EQ(config.moe_routed_prefill.overlay_segment_rows, 384);
    EXPECT_EQ(config.moe_routed_prefill.least_loaded_min_routed_rows, 4096u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_numerator, 5u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_denominator, 8u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_numerator, 9u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_denominator, 7u);
    EXPECT_FALSE(config.moe_routed_prefill.llep_enable_balanced_skip);
    EXPECT_EQ(config.moe_rebalance.dynamic_imbalance_threshold_per_mille, 1050u);
    EXPECT_EQ(config.moe_rebalance.dynamic_min_improvement_per_mille, 0u);
    EXPECT_EQ(config.moe_rebalance.dynamic_max_swaps_per_layer, 8u);
    EXPECT_EQ(config.moe_rebalance.dynamic_max_plan_entries_per_wave, 32u);
    EXPECT_EQ(config.moe_rebalance.dynamic_min_window_activations, 16u);
    EXPECT_EQ(config.moe_rebalance.device_maintenance_slack_tokens, 5);
    EXPECT_EQ(config.moe_rebalance.device_min_maintenance_period_tokens, 133);
    EXPECT_EQ(config.moe_rebalance.device_initial_maintenance_period_tokens, 69);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement, 64u);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement_divisor, 20u);
    EXPECT_EQ(config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot, 256u);
    EXPECT_EQ(
        config.moe_rebalance
            .device_min_foreign_rows_per_critical_path_payload_slot,
        640u);
    EXPECT_EQ(config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot, 512u);
    EXPECT_EQ(config.moe_rebalance.device_max_post_wave_load_spread_per_mille, 80u);
    EXPECT_FALSE(config.moe_rebalance.release_raw_expert_weights);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_RoutedExpertPlacementBlock)
{
    OrchestrationConfigParser parser;
    const std::string yaml = R"(
moe_routed_expert_placement:
    enabled: true
    topology: tiered-overlay
    continuation_domain: gpu_hot
    base_model_domain: gpu_hot
    shared_expert_domain: gpu_hot
    domains:
        - "gpu_hot=0:cuda:0,0:cuda:1;scope=rank_local;backend=nccl;routed_compute=apportioned;routed_decode_assignment=static-owner;routed_prefill_assignment=static-owner;owner=0"
    routed_tiers:
        - "hot@gpu_hot;priority=0"
    )";

    const auto config = parser.parseYamlString(yaml);

    ASSERT_NE(config.moe_routed_expert_plan, nullptr);
    EXPECT_TRUE(config.moe_routed_expert_plan->enabled);
    EXPECT_EQ(
        config.moe_routed_expert_plan->topology,
        RoutedExpertPlacementTopology::TieredOverlay);
    ASSERT_EQ(config.moe_routed_expert_plan->domains.size(), 1u);
    EXPECT_EQ(
        config.moe_routed_expert_plan->domains.front().routed_compute_policy,
        RoutedExpertComputePolicy::Apportioned);
    ASSERT_EQ(config.moe_routed_expert_plan->routed_tiers.size(), 1u);
    EXPECT_TRUE(config.moe_routed_expert_plan->routed_tiers.front().fallback);
}

TEST(Test__OrchestrationConfigParser, RejectsAuthoredFallbackTier)
{
    ArgvHelper args{
        "llaminar2",
        "--moe-routed-expert-placement", "tiered-overlay",
        "--moe-routed-expert-tier", "priority0@gpu;priority=0;fallback=true",
    };
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, RejectsObsoleteRoutedExpertYamlNames)
{
    OrchestrationConfigParser parser;

    EXPECT_THROW(
        parser.parseYamlString("moe:\n  expert_mode: replicated\n"),
        std::invalid_argument);
    EXPECT_THROW(
        parser.parseYamlString("moe_expert_parallel:\n  enabled: true\n"),
        std::invalid_argument);
    EXPECT_THROW(
        parser.parseYamlString("moe_expert_mode: apportioned-experts\n"),
        std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_QuotedValues)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
device: "cuda:0"
tp_scope: 'rank_local'
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.tp_scope, TPScope::RANK_LOCAL);
}

TEST(Test__OrchestrationConfigParser, ParseYamlFile_NonExistent_Throws)
{
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseYamlFile("/nonexistent/path/config.yaml"), std::invalid_argument);
}

// ============================================================================
// Help Text Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, GetHelpText_ContainsKeyOptions)
{
    std::string help = OrchestrationConfigParser::getHelpText();

    EXPECT_TRUE(help.find("--tensor-parallelism-degree") != std::string::npos);
    EXPECT_TRUE(help.find("--pipeline-parallelism-degree") != std::string::npos);
    EXPECT_TRUE(help.find("--device") != std::string::npos);
    EXPECT_TRUE(help.find("--define-domain") != std::string::npos);
    EXPECT_TRUE(help.find("--pp-stage") != std::string::npos);
    EXPECT_TRUE(help.find("--backend") != std::string::npos);
    EXPECT_TRUE(help.find("--config") != std::string::npos);
    EXPECT_TRUE(help.find("--moe-routed-expert-compute") != std::string::npos);
    EXPECT_TRUE(help.find("--moe-routed-expert-owner-order") != std::string::npos);
    EXPECT_TRUE(help.find("--moe-routed-expert-placement") != std::string::npos);
    EXPECT_TRUE(help.find("--moe-routed-expert-domain") != std::string::npos);
    EXPECT_TRUE(help.find("--moe-hot-expert-cache") != std::string::npos);
    EXPECT_EQ(help.find("--moe-expert-mode"), std::string::npos);
    EXPECT_EQ(help.find("--moe-expert-overlay"), std::string::npos);
}
// ============================================================================
// CLI Parsing - Model Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ModelPath)
{
    ArgvHelper args{"llaminar2", "-m", "models/test.gguf"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.model_path, "models/test.gguf");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ModelPath_LongForm)
{
    ArgvHelper args{"llaminar2", "--model", "path/to/model.gguf"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.model_path, "path/to/model.gguf");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ContextLength)
{
    ArgvHelper args{"llaminar2", "-c", "4096"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.max_seq_len, 4096);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MMap)
{
    ArgvHelper args{"llaminar2", "--no-mmap"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FALSE(config.use_mmap);
}

// ============================================================================
// CLI Parsing - Inference Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Prompt)
{
    ArgvHelper args{"llaminar2", "-p", "Hello, world!"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.prompt, "Hello, world!");
    EXPECT_TRUE(config.prompt_was_explicitly_provided);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Prompt_LongForm)
{
    ArgvHelper args{"llaminar2", "--prompt", "Test prompt"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.prompt, "Test prompt");
    EXPECT_TRUE(config.prompt_was_explicitly_provided);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NPredict)
{
    ArgvHelper args{"llaminar2", "-n", "100"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.n_predict, 100);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NPredict_LongForm)
{
    ArgvHelper args{"llaminar2", "--n-predict", "50"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.n_predict, 50);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_BatchSize)
{
    ArgvHelper args{"llaminar2", "--batch-size", "8"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.batch_size, 8);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Threads)
{
    ArgvHelper args{"llaminar2", "--threads", "16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.n_threads, 16);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Seed)
{
    ArgvHelper args{"llaminar2", "-s", "42"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.seed, 42);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Seed_LongForm)
{
    ArgvHelper args{"llaminar2", "--seed", "12345"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.seed, 12345);
}

// ============================================================================
// CLI Parsing - Sampling Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Temperature)
{
    ArgvHelper args{"llaminar2", "-t", "0.5"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.temperature, 0.5f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Temperature_LongForm)
{
    ArgvHelper args{"llaminar2", "--temperature", "1.2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.temperature, 1.2f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TopK)
{
    ArgvHelper args{"llaminar2", "--top-k", "50"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.top_k, 50);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TopP)
{
    ArgvHelper args{"llaminar2", "--top-p", "0.95"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.top_p, 0.95f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Deterministic)
{
    ArgvHelper args{"llaminar2", "--deterministic"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.deterministic);
}

/**
 * @brief Restore the process environment and canonical snapshot after a test.
 *
 * Logging may read DebugEnv before CLI parsing. Tests deliberately reproduce
 * that order without leaving deterministic policy enabled for other tests.
 */
class ScopedDeterministicEnvironment
{
public:
    /** @brief Save the caller's inherited deterministic environment value. */
    ScopedDeterministicEnvironment()
    {
        if (const char *value = std::getenv("LLAMINAR_DETERMINISTIC"))
            previous_ = value;
    }

    /** @brief Restore both exported state and its canonical parsed snapshot. */
    ~ScopedDeterministicEnvironment()
    {
        if (previous_)
            setenv("LLAMINAR_DETERMINISTIC", previous_->c_str(), 1);
        else
            unsetenv("LLAMINAR_DETERMINISTIC");
        mutableDebugEnv().reload();
    }

private:
    std::optional<std::string> previous_;
};

TEST(Test__OrchestrationConfigParser,
     DeterministicPublishesIntoPreviouslyReadKernelPolicy)
{
    ScopedDeterministicEnvironment restore;
    ASSERT_EQ(setenv("LLAMINAR_DETERMINISTIC", "0", 1), 0);
    mutableDebugEnv().reload();
    ASSERT_FALSE(debugEnv().gemm.deterministic);

    // Reproduce early logging/splash reads, including policy defaults that
    // deterministic execution must override on both accelerator backends.
    mutableDebugEnv().gemm.cuda_concurrent_prefill = true;
    mutableDebugEnv().rocm.concurrent_prefill = true;
    mutableDebugEnv().rocm.concurrent_decode = true;
    mutableDebugEnv().rocm.gdn_concurrent_decode = true;

    ArgvHelper args{"llaminar2", "--deterministic"};
    OrchestrationConfigParser parser;
    const auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.deterministic);
    EXPECT_FLOAT_EQ(config.temperature, 0.0f);
    EXPECT_STREQ(std::getenv("LLAMINAR_DETERMINISTIC"), "1");
    EXPECT_TRUE(debugEnv().gemm.deterministic);
    EXPECT_FALSE(debugEnv().gemm.cuda_concurrent_prefill);
    EXPECT_FALSE(debugEnv().rocm.concurrent_prefill);
    EXPECT_FALSE(debugEnv().rocm.concurrent_decode);
    EXPECT_FALSE(debugEnv().rocm.gdn_concurrent_decode);

    // MPI initialization reparses argv. Publishing the same startup policy
    // twice must be idempotent, not toggle a backend into a different mode.
    (void)parser.parseArgs(args.argc(), args.argv());
    EXPECT_TRUE(debugEnv().gemm.deterministic);
    EXPECT_FALSE(debugEnv().rocm.concurrent_decode);
}

// ============================================================================
// CLI Parsing - Chat Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ChatMode)
{
    ArgvHelper args{"llaminar2", "--chat"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.chat_mode);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ChatSingle)
{
    ArgvHelper args{"llaminar2", "--chat-single"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.single_shot_chat);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_SystemPrompt)
{
    ArgvHelper args{"llaminar2", "--system", "You are a helpful assistant."};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.system_prompt, "You are a helpful assistant.");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ChatTemplate)
{
    ArgvHelper args{"llaminar2", "--chat-template", "chatml"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.chat_template_override, "chatml");
}

// ============================================================================
// CLI Parsing - Benchmark Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_BenchmarkMode)
{
    ArgvHelper args{"llaminar2", "--benchmark"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.benchmark_mode);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_BenchmarkJsonOutput)
{
    ArgvHelper args{"llaminar2", "--benchmark", "--benchmark-json-output", "/tmp/llaminar-benchmark.json"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.benchmark_mode);
    EXPECT_EQ(config.benchmark_json_output_path, "/tmp/llaminar-benchmark.json");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_BenchmarkPromptFile)
{
    ArgvHelper args{"llaminar2", "--benchmark", "--prompt-file", "/tmp/fixed-prompt.txt"};
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.benchmark_mode);
    EXPECT_TRUE(config.benchmark_prompt_file_was_provided);
    EXPECT_EQ(config.benchmark_prompt_file_path, "/tmp/fixed-prompt.txt");
}

// ============================================================================
// CLI Parsing - Fused Attention
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_FusedAttention)
{
    ArgvHelper args{"llaminar2", "--fused-attention"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.use_fused_attention);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_FusedAttentionBackend)
{
    ArgvHelper args{"llaminar2", "--fused-attention-backend", "reference"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.fused_attention_backend, FusedAttentionBackend::REFERENCE);
}

// ============================================================================
// CLI Parsing - MPI Bootstrap
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIProcs)
{
    ArgvHelper args{"llaminar2", "--mpi-procs", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.mpi_procs, 4);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIDryRun)
{
    ArgvHelper args{"llaminar2", "--mpi-dry-run"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_dry_run);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIVerbose)
{
    ArgvHelper args{"llaminar2", "--mpi-verbose"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_verbose);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoMPIBootstrap)
{
    ArgvHelper args{"llaminar2", "--no-mpi-bootstrap"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_no_bootstrap);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIOversubscribe)
{
    ArgvHelper args{"llaminar2", "--mpi-oversubscribe"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.mpi_oversubscribe);
}

// ============================================================================
// CLI Parsing - Verbosity
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_VerboseLevel1)
{
    ArgvHelper args{"llaminar2", "-v"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.verbose_level, 1);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_VerboseLevel2)
{
    ArgvHelper args{"llaminar2", "-vv"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.verbose_level, 2);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ListDevices)
{
    ArgvHelper args{"llaminar2", "--list-devices"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.list_devices);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowHelp)
{
    ArgvHelper args{"llaminar2", "-h"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_help);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ShowHelp_LongForm)
{
    ArgvHelper args{"llaminar2", "--help"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.show_help);
}

// ============================================================================
// CLI Parsing - Memory Constraints
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MaxGPUMemory)
{
    ArgvHelper args{"llaminar2", "--max-gpu-memory", "8000"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.max_gpu_memory_mb.has_value());
    EXPECT_EQ(config.max_gpu_memory_mb.value(), 8000u);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MaxCPUMemory)
{
    ArgvHelper args{"llaminar2", "--max-cpu-memory", "16000"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.max_cpu_memory_mb.has_value());
    EXPECT_EQ(config.max_cpu_memory_mb.value(), 16000u);
}

// ============================================================================
// CLI Parsing - MoE Configuration
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESharedGPU)
{
    ArgvHelper args{"llaminar2", "--moe-shared-gpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.moe_shared_experts_gpu);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESharedCPU)
{
    ArgvHelper args{"llaminar2", "--moe-shared-cpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FALSE(config.moe_shared_experts_gpu);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESparseGPU)
{
    ArgvHelper args{"llaminar2", "--moe-sparse-gpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FALSE(config.moe_sparse_experts_cpu);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoESparseCPU)
{
    ArgvHelper args{"llaminar2", "--moe-sparse-cpu"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.moe_sparse_experts_cpu);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_RoutedExpertComputePolicy)
{
    {
        ArgvHelper args{"llaminar2", "--moe-routed-expert-compute", "apportioned"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.routed_expert_compute_policy, RoutedExpertComputePolicy::Apportioned);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-routed-expert-compute", "replicated"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.routed_expert_compute_policy, RoutedExpertComputePolicy::Replicated);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-routed-expert-compute", "tensor-sharded"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.routed_expert_compute_policy, RoutedExpertComputePolicy::TensorSharded);
    }
}

TEST(Test__OrchestrationConfigParser, RejectsObsoleteRoutedExpertCliNamesAndValues)
{
    OrchestrationConfigParser parser;

    ArgvHelper old_option{
        "llaminar2", "--moe-expert-mode", "apportioned-experts"};
    EXPECT_THROW(
        parser.parseArgs(old_option.argc(), old_option.argv()),
        std::invalid_argument);

    ArgvHelper old_placement_option{
        "llaminar2", "--moe-expert-overlay", "tiered"};
    EXPECT_THROW(
        parser.parseArgs(
            old_placement_option.argc(),
            old_placement_option.argv()),
        std::invalid_argument);

    ArgvHelper old_value{
        "llaminar2", "--moe-routed-expert-compute", "sharded-experts"};
    EXPECT_THROW(
        parser.parseArgs(old_value.argc(), old_value.argv()),
        std::invalid_argument);

    ArgvHelper old_llep_maintenance_option{
        "llaminar2", "--moe-device-llep-alpha-numerator", "1"};
    EXPECT_THROW(
        parser.parseArgs(
            old_llep_maintenance_option.argc(),
            old_llep_maintenance_option.argv()),
        std::invalid_argument)
        << "Current-batch LLEP parameters must not be accepted through the "
           "durable residency-maintenance namespace.";
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoEHotExpertCache)
{
    {
        ArgvHelper args{"llaminar2", "--moe-hot-expert-cache", "10%"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Percent);
        EXPECT_FLOAT_EQ(config.moe_hot_expert_cache.percent, 10.0f);
        EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 25);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-hot-expert-cache", "24"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Count);
        EXPECT_EQ(config.moe_hot_expert_cache.count, 24);
        EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 24);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-hot-expert-cache", "off"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Off);
        EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 0);
    }
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MoERebalance)
{
    ArgvHelper args{"llaminar2",
                    "--moe-residency-maintenance", "observe",
                    "--moe-residency-maintenance-window", "128",
                    "--moe-residency-maintenance-max-window", "2048",
                    "--moe-residency-maintenance-window-growth", "2.0",
                    "--moe-migration-payoff-horizon-tokens", "65536",
                    "--moe-migration-transfer-slots", "5",
                    "--moe-migration-execution-streams", "3",
                    "--moe-migration-cycles-per-wave", "4",
                    "--moe-routed-prefill-assignment-window", "384",
                    "--moe-overlay-prefill-segment-rows", "448",
                    "--moe-routed-prefill-least-loaded-min-routed-rows", "1024",
                    "--moe-routed-prefill-llep-alpha-numerator", "2",
                    "--moe-routed-prefill-llep-alpha-denominator", "3",
                    "--moe-routed-prefill-llep-lambda-numerator", "11",
                    "--moe-routed-prefill-llep-lambda-denominator", "8",
                    "--moe-routed-prefill-llep-disable-balanced-skip",
                    "--moe-dynamic-imbalance-threshold-permille", "1100",
                    "--moe-dynamic-min-improvement-permille", "10",
                    "--moe-dynamic-max-swaps-per-layer", "7",
                    "--moe-dynamic-max-plan-entries-per-wave", "28",
                    "--moe-dynamic-min-window-activations", "48",
                    "--moe-device-rebalance-maintenance-slack-tokens", "2",
                    "--moe-device-rebalance-min-maintenance-period-tokens", "130",
                    "--moe-device-rebalance-initial-maintenance-period-tokens", "66",
                    "--moe-device-rebalance-min-load-spread-improvement", "72",
                    "--moe-device-rebalance-min-load-spread-improvement-divisor", "25",
                    "--moe-device-rebalance-min-wave-spread-improvement-per-payload-slot", "144",
                    "--moe-device-rebalance-min-foreign-rows-per-critical-path-payload-slot", "216",
                    "--moe-device-rebalance-min-router-spread-improvement-per-payload-slot", "288",
                    "--moe-device-rebalance-max-post-wave-load-spread-permille", "90",
                    "--moe-release-raw-expert-weights"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Observe);
    EXPECT_EQ(config.moe_rebalance.window_size, 128);
    EXPECT_EQ(config.moe_rebalance.max_window_size, 2048);
    EXPECT_FLOAT_EQ(config.moe_rebalance.window_growth_factor, 2.0f);
    EXPECT_EQ(
        config.moe_rebalance.migration_payoff_horizon_tokens,
        65'536u);
    EXPECT_EQ(config.moe_rebalance.migration_transfer_slots, 5u);
    ASSERT_TRUE(config.moe_rebalance.migration_execution_streams.has_value());
    EXPECT_EQ(*config.moe_rebalance.migration_execution_streams, 3u);
    EXPECT_EQ(
        config.moe_rebalance.resolvedMigrationExecutionStreams(),
        3u);
    ASSERT_TRUE(config.moe_rebalance.migration_cycles_per_wave.has_value());
    EXPECT_EQ(*config.moe_rebalance.migration_cycles_per_wave, 4u);
    EXPECT_EQ(config.moe_rebalance.resolvedMigrationCyclesPerWave(), 4u);
    EXPECT_EQ(config.moe_routed_prefill.assignment_window_tokens, 384);
    EXPECT_EQ(config.moe_routed_prefill.overlay_segment_rows, 448);
    EXPECT_EQ(config.moe_routed_prefill.least_loaded_min_routed_rows, 1024u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_numerator, 2u);
    EXPECT_EQ(config.moe_routed_prefill.llep_alpha_denominator, 3u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_numerator, 11u);
    EXPECT_EQ(config.moe_routed_prefill.llep_lambda_denominator, 8u);
    EXPECT_FALSE(config.moe_routed_prefill.llep_enable_balanced_skip);
    EXPECT_EQ(config.moe_rebalance.dynamic_imbalance_threshold_per_mille, 1100u);
    EXPECT_EQ(config.moe_rebalance.dynamic_min_improvement_per_mille, 10u);
    EXPECT_EQ(config.moe_rebalance.dynamic_max_swaps_per_layer, 7u);
    EXPECT_EQ(config.moe_rebalance.dynamic_max_plan_entries_per_wave, 28u);
    EXPECT_EQ(config.moe_rebalance.dynamic_min_window_activations, 48u);
    EXPECT_EQ(config.moe_rebalance.device_maintenance_slack_tokens, 2);
    EXPECT_EQ(config.moe_rebalance.device_min_maintenance_period_tokens, 130);
    EXPECT_EQ(config.moe_rebalance.device_initial_maintenance_period_tokens, 66);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement, 72u);
    EXPECT_EQ(config.moe_rebalance.device_min_load_spread_improvement_divisor, 25u);
    EXPECT_EQ(config.moe_rebalance.device_min_wave_spread_improvement_per_payload_slot, 144u);
    EXPECT_EQ(
        config.moe_rebalance
            .device_min_foreign_rows_per_critical_path_payload_slot,
        216u);
    EXPECT_EQ(config.moe_rebalance.device_min_router_spread_improvement_per_payload_slot, 288u);
    EXPECT_EQ(config.moe_rebalance.device_max_post_wave_load_spread_per_mille, 90u);
    EXPECT_TRUE(config.moe_rebalance.release_raw_expert_weights);
}

TEST(Test__OrchestrationConfigParser,
     ParseArgs_LLEPIsNotAResidencyMaintenanceMode)
{
    ArgvHelper args{
        "llaminar2",
        "--moe-residency-maintenance",
        "llep"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(
        parser.parseArgs(args.argc(), args.argv()),
        std::invalid_argument)
        << "Current-batch LLEP belongs to routed_prefill_assignment, not "
           "durable residency maintenance.";
}

TEST(Test__OrchestrationConfigParser,
     ParseArgs_MigrationTransferSlotsMustBePositive)
{
    ArgvHelper args{
        "llaminar2",
        "--moe-migration-transfer-slots",
        "0"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(
        parser.parseArgs(args.argc(), args.argv()),
        std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser,
     ParseArgs_MigrationCyclesPerWaveMustBePositive)
{
    ArgvHelper args{
        "llaminar2",
        "--moe-migration-cycles-per-wave",
        "0"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(
        parser.parseArgs(args.argc(), args.argv()),
        std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser,
     ParseArgs_MigrationExecutionStreamsMustBePositive)
{
    ArgvHelper args{
        "llaminar2",
        "--moe-migration-execution-streams",
        "0"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(
        parser.parseArgs(args.argc(), args.argv()),
        std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser,
     Validate_RejectsUndefinedProgrammaticLLEPRatios)
{
    OrchestrationConfig config;
    config.moe_routed_prefill.llep_alpha_numerator = 0;
    config.moe_routed_prefill.llep_alpha_denominator = 0;
    config.moe_routed_prefill.llep_lambda_numerator = 0;
    config.moe_routed_prefill.llep_lambda_denominator = 0;

    const auto errors = config.validate();
    const auto contains = [&](const std::string &needle)
    {
        return std::any_of(
            errors.begin(),
            errors.end(),
            [&](const std::string &error)
            {
                return error.find(needle) != std::string::npos;
            });
    };

    EXPECT_TRUE(contains("LLEP alpha numerator must be > 0"));
    EXPECT_TRUE(contains("LLEP alpha denominator must be > 0"));
    EXPECT_TRUE(contains("LLEP lambda numerator must be > 0"));
    EXPECT_TRUE(contains("LLEP lambda denominator must be > 0"));
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidMoEConfig_Throws)
{
    {
        ArgvHelper args{"llaminar2", "--moe-routed-expert-compute", "mystery"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-hot-expert-cache", "101%"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-hot-expert-cache", "-1"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args{
            "llaminar2",
            "--moe-residency-maintenance",
            "mystery"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-dynamic-max-swaps-per-layer", "-1"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args{
            "llaminar2",
            "--moe-migration-payoff-horizon-tokens",
            "0"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(
            parser.parseArgs(args.argc(), args.argv()),
            std::invalid_argument);
    }
    {
        ArgvHelper args{
            "llaminar2",
            "--moe-device-rebalance-maintenance-slack-tokens",
            "-1"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
    {
        ArgvHelper args{
            "llaminar2",
            "--moe-device-rebalance-min-maintenance-period-tokens",
            "2147483648"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
}

// ============================================================================
// CLI Parsing - Precision
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision)
{
    ArgvHelper args{"llaminar2", "--activation-precision", "fp32"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.activation_precision, "fp32");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision_Alias)
{
    ArgvHelper args{"llaminar2", "--act-prec", "fp32"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.activation_precision, "fp32");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPAllreducePrecision)
{
    ArgvHelper args{"llaminar2", "--tp-allreduce-precision", "fp16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_allreduce_precision_override, "fp16");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPAllreducePrecision_AliasNormalizesValue)
{
    ArgvHelper args{"llaminar2", "--allreduce-precision", "f16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_allreduce_precision_override, "fp16");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TPAllreducePrecision_InvalidThrows)
{
    ArgvHelper args{"llaminar2", "--tp-allreduce-precision", "int4"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, Validate_TPAllreducePrecisionRejectsInvalidYamlValue)
{
    OrchestrationConfig config;
    config.tp_allreduce_precision_override = "int4";

    const auto errors = config.validate();

    ASSERT_FALSE(errors.empty());
    EXPECT_NE(errors.front().find("Invalid tp_allreduce_precision_override"), std::string::npos);
}

// ============================================================================
// CLI Parsing - Weight Sharding
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ShardWeights)
{
    ArgvHelper args{"llaminar2", "--shard-weights"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.shard_weights);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoShardWeights)
{
    ArgvHelper args{"llaminar2", "--no-shard-weights"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.disable_weight_sharding);
}

// ============================================================================
// CLI Parsing - Heterogeneous Mode
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_HeterogeneousMode)
{
    ArgvHelper args{"llaminar2", "--heterogeneous"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.heterogeneous_mode);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_CPUFraction)
{
    ArgvHelper args{"llaminar2", "--cpu-fraction", "0.3"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_FLOAT_EQ(config.cpu_compute_fraction, 0.3f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoGPUTP)
{
    ArgvHelper args{"llaminar2", "--no-gpu-tp"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.disable_gpu_tp);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NoCPUTP)
{
    ArgvHelper args{"llaminar2", "--no-cpu-tp"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_TRUE(config.disable_cpu_tp);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MinLayersPerDomain)
{
    ArgvHelper args{"llaminar2", "--min-layers-per-domain", "4"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.min_layers_per_domain, 4);
}

// ============================================================================
// Validation Tests - Invalid Arguments
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_UnknownArgument_Throws)
{
    ArgvHelper args{"llaminar2", "--unknown-flag"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidStrategy_Throws)
{
    ArgvHelper args{"llaminar2", "--strategy", "invalid-strategy"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidActivationPrecision_Throws)
{
    ArgvHelper args{"llaminar2", "--activation-precision", "fp64"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidCPUFraction_TooHigh_Throws)
{
    ArgvHelper args{"llaminar2", "--cpu-fraction", "1.5"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidCPUFraction_Negative_Throws)
{
    ArgvHelper args{"llaminar2", "--cpu-fraction", "-0.1"};
    OrchestrationConfigParser parser;

    EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_HeterogeneousWithSingleRank_ValidationWarning)
{
    // This test checks that --heterogeneous with TP=1 produces a validation warning
    // (heterogeneous mode requires TP >= 2 to be meaningful)
    ArgvHelper args{"llaminar2", "--heterogeneous", "--tensor-parallelism-degree", "1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    // Config should still parse, but validation would show issues
    EXPECT_TRUE(config.heterogeneous_mode);
    EXPECT_EQ(config.tp_degree, 1);
}

// ============================================================================
// Combined Flags Test
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_FullInferenceConfig)
{
    ArgvHelper args{"llaminar2",
                    "-m", "model.gguf",
                    "-p", "Hello world",
                    "-n", "100",
                    "-t", "0.7",
                    "--top-k", "50",
                    "--top-p", "0.9",
                    "-s", "42",
                    "-d", "cuda:0",
                    "-tp", "2"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.model_path, "model.gguf");
    EXPECT_EQ(config.prompt, "Hello world");
    EXPECT_EQ(config.n_predict, 100);
    EXPECT_FLOAT_EQ(config.temperature, 0.7f);
    EXPECT_EQ(config.top_k, 50);
    EXPECT_FLOAT_EQ(config.top_p, 0.9f);
    EXPECT_EQ(config.seed, 42);
    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.tp_degree, 2);
}

// ============================================================================
// Help Text Extended Tests
// ============================================================================

TEST(Test__OrchestrationConfigParser, GetHelpText_ContainsAllShortFlags)
{
    std::string help = OrchestrationConfigParser::getHelpText();

    // All documented short flags should be in help text
    EXPECT_TRUE(help.find("-m") != std::string::npos);
    EXPECT_TRUE(help.find("-p") != std::string::npos);
    EXPECT_TRUE(help.find("-n") != std::string::npos);
    EXPECT_TRUE(help.find("-t") != std::string::npos);
    EXPECT_TRUE(help.find("-s") != std::string::npos);
    EXPECT_TRUE(help.find("-d") != std::string::npos);
    EXPECT_TRUE(help.find("-c") != std::string::npos);
    EXPECT_TRUE(help.find("-tp") != std::string::npos);
    EXPECT_TRUE(help.find("-pp") != std::string::npos);
    EXPECT_TRUE(help.find("-b") != std::string::npos);
    EXPECT_TRUE(help.find("-v") != std::string::npos);
    EXPECT_TRUE(help.find("-h") != std::string::npos);
}

TEST(Test__OrchestrationConfigParser, GetHelpText_ContainsExamples)
{
    std::string help = OrchestrationConfigParser::getHelpText();

    EXPECT_TRUE(help.find("Examples:") != std::string::npos);
    EXPECT_TRUE(help.find("llaminar2") != std::string::npos);
}

// ============================================================================
// Precision Tests (activation + kv-cache, with all short aliases)
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_KvCachePrecision_LongForm)
{
    ArgvHelper args({"llaminar2", "--kv-cache-precision", "q16_1"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.kv_cache_precision, "q16_1");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_KvCachePrecision_ShortAliasFlag)
{
    ArgvHelper args({"llaminar2", "--kv-prec", "fp16"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.kv_cache_precision, "fp16");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_KvCachePrecision_AcceptsShortValueAliases)
{
    // The kv-cache parser normalises to lowercase and accepts short value
    // aliases (f32, f16, q8, q16, i16) in addition to canonical forms.
    for (const auto &alias : {"f32", "f16", "q8", "q16", "i16", "tq4", "tq"})
    {
        ArgvHelper args({"llaminar2", "--kv-cache-precision", alias});
        auto parser = createOrchestrationConfigParser();
        auto config = parser->parseArgs(args.argc(), args.argv());
        EXPECT_EQ(config.kv_cache_precision, alias) << "alias=" << alias;
    }
}

TEST(Test__OrchestrationConfigParser, ParseArgs_KvCachePrecision_InvalidThrows)
{
    ArgvHelper args({"llaminar2", "--kv-cache-precision", "nonsense"});
    auto parser = createOrchestrationConfigParser();
    EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision_InvalidReportsUnimplemented)
{
    // Admission reports the actual production support rather than advertising
    // tensor dtypes as complete model activation modes.
    ArgvHelper args({"llaminar2", "--activation-precision", "int4"});
    auto parser = createOrchestrationConfigParser();
    try
    {
        parser->parseArgs(args.argc(), args.argv());
        FAIL() << "expected throw";
    }
    catch (const std::invalid_argument &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("fp32"), std::string::npos);
        EXPECT_NE(msg.find("unimplemented"), std::string::npos);
    }
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision_AllThreeAliases)
{
    for (const auto &flag : {"--activation-precision", "--activation-prec", "--act-prec"})
    {
        ArgvHelper args({"llaminar2", flag, "fp32"});
        auto parser = createOrchestrationConfigParser();
        auto config = parser->parseArgs(args.argc(), args.argv());
        EXPECT_EQ(config.activation_precision, "fp32") << "flag=" << flag;
    }
}

TEST(Test__OrchestrationConfigParser, UnimplementedActivationsRejectCLIYamlAndConfig)
{
    for (const auto *precision : {"fp16", "bf16", "q8_1", "q16_1", "hybrid", "hybridq16"})
    {
        SCOPED_TRACE(precision);
        OrchestrationConfigParser parser;
        for (const auto *flag : {"--activation-precision", "--activation-prec", "--act-prec"})
        {
            ArgvHelper args{"llaminar2", flag, precision};
            EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
        }
        EXPECT_THROW(
            parser.parseYamlString(std::string("activation_precision: ") + precision),
            std::invalid_argument);
        OrchestrationConfig config;
        config.activation_precision = precision;
        const auto errors = config.validate();
        EXPECT_TRUE(std::any_of(errors.begin(), errors.end(), [](const auto &error)
        {
            return error.find("unimplemented") != std::string::npos &&
                   error.find("fp32") != std::string::npos;
        }));
    }
}

// ============================================================================
// MPI profile (enum) + backend / scope / split enum coverage
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIProfile_Tuned)
{
    ArgvHelper args({"llaminar2", "--mpi-profile", "tuned"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.mpi_profile, MPIProfile::TUNED);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_MPIProfile_InvalidThrows)
{
    ArgvHelper args({"llaminar2", "--mpi-profile", "turbo"});
    auto parser = createOrchestrationConfigParser();
    try
    {
        parser->parseArgs(args.argc(), args.argv());
        FAIL() << "expected throw";
    }
    catch (const std::invalid_argument &e)
    {
        std::string msg = e.what();
        EXPECT_NE(msg.find("--mpi-profile"), std::string::npos);
        // valid_values whitelist should mention both accepted values.
        EXPECT_NE(msg.find("auto"), std::string::npos);
        EXPECT_NE(msg.find("tuned"), std::string::npos);
    }
}

TEST(Test__OrchestrationConfigParser, ParseArgs_TpScope_NodeLocal)
{
    ArgvHelper args({"llaminar2", "--tp-scope", "node_local"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.tp_scope, TPScope::NODE_LOCAL);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Backend_Heterogeneous)
{
    ArgvHelper args({"llaminar2", "--backend", "heterogeneous"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.default_backend, CollectiveBackendType::HETEROGENEOUS);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_PpSplit_InvalidThrows)
{
    ArgvHelper args({"llaminar2", "--pp-split", "lopsided"});
    auto parser = createOrchestrationConfigParser();
    EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

// ============================================================================
// Device shorthand semantics
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_CpuMarksAllLocal)
{
    // Bare `-d cpu` should set cpu_global_tp_all_local=true and leave
    // numa_explicit=false (the orchestrator will fan out across NUMA nodes).
    ArgvHelper args({"llaminar2", "-d", "cpu"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_TRUE(config.cpu_global_tp_all_local);
    EXPECT_FALSE(config.device_for_this_rank_numa_explicit);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Device_InvalidCpuNumaThrows)
{
    ArgvHelper args({"llaminar2", "-d", "cpu:abc"});
    auto parser = createOrchestrationConfigParser();
    EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

// ============================================================================
// Verbosity: -v / -vv / -vvv / repeated -v
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Verbose_Triple)
{
    ArgvHelper args({"llaminar2", "-vvv"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.verbose_level, 3);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Verbose_RepeatedIncrements)
{
    ArgvHelper args({"llaminar2", "-v", "-v", "-v"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.verbose_level, 3);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Verbose_RepeatedIncrementsClampAt3)
{
    ArgvHelper args({"llaminar2", "-v", "-v", "-v", "-v", "-v"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.verbose_level, 3);
}

// ============================================================================
// Deterministic: temperature forcing + env var side effect
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Deterministic_ForcesTemperatureZero)
{
    // Even when --temperature is supplied non-zero before --deterministic,
    // the deterministic flag must zero it out.
    ArgvHelper args({"llaminar2", "--temperature", "0.8", "--deterministic"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_TRUE(config.deterministic);
    EXPECT_FLOAT_EQ(config.temperature, 0.0f);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Deterministic_SetsEnvVar)
{
    unsetenv("LLAMINAR_DETERMINISTIC");
    ArgvHelper args({"llaminar2", "--deterministic"});
    auto parser = createOrchestrationConfigParser();
    (void)parser->parseArgs(args.argc(), args.argv());
    const char *env = std::getenv("LLAMINAR_DETERMINISTIC");
    ASSERT_NE(env, nullptr);
    EXPECT_STREQ(env, "1");
    unsetenv("LLAMINAR_DETERMINISTIC");
}

// ============================================================================
// Server mode flags
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Serve_WithHostAndPort)
{
    ArgvHelper args({"llaminar2", "--serve", "--host", "0.0.0.0", "--port", "9000"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_TRUE(config.serve_mode);
    EXPECT_EQ(config.serve_host, "0.0.0.0");
    EXPECT_EQ(config.serve_port, 9000);
}

// ============================================================================
// MPI bootstrap flags
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_MpiHostfile)
{
    ArgvHelper args({"llaminar2", "--mpi-hostfile", "/etc/hosts.txt"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.hostfile, "/etc/hosts.txt");
}

// ============================================================================
// NYI flags still parse (back-compat)
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_NYIFlags_StillAccepted)
{
    // All NYI flags must still be parseable so existing scripts don't break.
    ArgvHelper args({
        "llaminar2",
        "--topology",
        "PP(a)",
        "--topology-file",
        "topo.yaml",
        "--max-gpu-memory",
        "8192",
        "--max-cpu-memory",
        "16384",
        "--cpu-layers",
        "4",
        "--cpu-layers-first",
    });
    auto parser = createOrchestrationConfigParser();
    OrchestrationConfig config;
    EXPECT_NO_THROW(config = parser->parseArgs(args.argc(), args.argv()));
    EXPECT_EQ(config.topology_string, "PP(a)");
    EXPECT_EQ(config.topology_file_path, "topo.yaml");
    EXPECT_EQ(config.max_gpu_memory_mb, 8192u);
    EXPECT_EQ(config.max_cpu_memory_mb, 16384u);
    EXPECT_EQ(config.cpu_layers, 4);
    EXPECT_TRUE(config.cpu_layers_first);
}

// ============================================================================
// Cross-flag validation
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_Heterogeneous_NoGpuTp_NoCpuTp_Throws)
{
    ArgvHelper args({"llaminar2", "--heterogeneous", "--no-gpu-tp", "--no-cpu-tp"});
    auto parser = createOrchestrationConfigParser();
    EXPECT_THROW(parser->parseArgs(args.argc(), args.argv()), std::invalid_argument);
}

// ============================================================================
// Negative-number value support for int options (regression guard)
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_NegativeSeedSpaceForm)
{
    ArgvHelper args({"llaminar2", "--seed", "-1"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.seed, -1);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_NegativeNPredictSpaceForm)
{
    ArgvHelper args({"llaminar2", "-n", "-1"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.n_predict, -1);
}

// ============================================================================
// Equals-form coverage for assorted option types
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_EqualsForm_OnEnum)
{
    ArgvHelper args({"llaminar2", "--tp-scope=rank_local"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.tp_scope, TPScope::RANK_LOCAL);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_EqualsForm_OnString)
{
    ArgvHelper args({"llaminar2", "--model=/tmp/model.gguf"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.model_path, "/tmp/model.gguf");
}

// ============================================================================
// Last-write-wins for repeated value flags
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_RepeatedFlag_LastWriteWins)
{
    ArgvHelper args({"llaminar2", "--temperature", "0.5", "--temperature", "0.9"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_FLOAT_EQ(config.temperature, 0.9f);
}
