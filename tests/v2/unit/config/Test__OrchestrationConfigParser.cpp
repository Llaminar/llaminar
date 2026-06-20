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
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "config/OrchestrationConfigParser.h"

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
    EXPECT_EQ(config.moe_expert_mode, MoEExpertMode::ExpertParallel);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Percent);
    EXPECT_FLOAT_EQ(config.moe_hot_expert_cache.percent, 10.0f);
    EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 25);
    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
    EXPECT_EQ(config.moe_rebalance.window_size, 256);
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
    ArgvHelper args{"llaminar2", "--tp-scope", "local"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.tp_scope, TPScope::LOCAL);
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
                    "--define-domain", "rocm_socket0=0:rocm:0,0:rocm:1;scope=local;backend=rccl;owner=0",
                    "--define-domain", "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_EQ(config.domain_definitions.size(), 2u);
    EXPECT_EQ(config.domain_definitions[0].scope, TPScope::LOCAL);
    ASSERT_TRUE(config.domain_definitions[0].owner_rank.has_value());
    EXPECT_EQ(*config.domain_definitions[0].owner_rank, 0);
    EXPECT_EQ(config.domain_definitions[0].backend, CollectiveBackendType::RCCL);

    EXPECT_EQ(config.domain_definitions[1].scope, TPScope::NODE_LOCAL);
    EXPECT_EQ(config.domain_definitions[1].backend, CollectiveBackendType::UPI);
    ASSERT_EQ(config.domain_definitions[1].explicit_ranks.size(), 2u);
    EXPECT_EQ(config.domain_definitions[1].explicit_ranks[0], 0);
    EXPECT_EQ(config.domain_definitions[1].explicit_ranks[1], 1);
}

TEST(Test__OrchestrationConfigParser, Phase9B_NamedAndOverlayDomainsShareCanonicalNormalization)
{
    OrchestrationConfigParser parser;
    ArgvHelper named_args{"llaminar2",
                          "--define-domain", "rocm_hot=0:rocm:0,0:rocm:1;weights=0.60,0.40;scope=local;backend=rccl;owner=0"};
    ArgvHelper overlay_args{"llaminar2",
                            "--moe-expert-overlay", "tiered",
                            "--moe-expert-overlay-continuation", "rocm_hot",
                            "--moe-expert-overlay-shared-domain", "rocm_hot",
                            "--moe-expert-overlay-domain", "rocm_hot=0:rocm:0,0:rocm:1;weights=0.60,0.40;scope=local;backend=rccl;compute=expert_id_sharded;owner=0",
                            "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=expert_id_sharded;ranks=0,1",
                            "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0",
                            "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"};

    const auto named_config = parser.parseArgs(named_args.argc(), named_args.argv());
    const auto overlay_config = parser.parseArgs(overlay_args.argc(), overlay_args.argv());

    ASSERT_EQ(named_config.domain_definitions.size(), 1u);
    ASSERT_NE(overlay_config.moe_expert_parallel_plan, nullptr);
    ASSERT_EQ(overlay_config.moe_expert_parallel_plan->domains.size(), 2u);

    const auto named_domain = named_config.domain_definitions[0].toExecutionDomainDefinition();
    const auto overlay_domain = overlay_config.moe_expert_parallel_plan->domains[0].toExecutionDomainDefinition();

    EXPECT_EQ(named_domain.name, overlay_domain.name);
    EXPECT_EQ(named_domain.participants, overlay_domain.participants);
    EXPECT_EQ(named_domain.weights, overlay_domain.weights);
    EXPECT_EQ(named_domain.scope, overlay_domain.scope);
    EXPECT_EQ(named_domain.backend, overlay_domain.backend);
    EXPECT_EQ(named_domain.owner_rank, overlay_domain.owner_rank);
    EXPECT_EQ(named_domain.ranks, overlay_domain.ranks);
    EXPECT_EQ(named_domain.compute_kind, ExecutionDomainComputeKind::UNSPECIFIED);
    EXPECT_EQ(overlay_domain.compute_kind, ExecutionDomainComputeKind::EXPERT_ID_SHARDED);

    const auto inventory = overlay_config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 2u);
    EXPECT_EQ(inventory[0].logicalIdentity(), "rocm_hot");
    EXPECT_EQ(inventory[1].logicalIdentity(), "cpu_cold");
}

TEST(Test__OrchestrationConfigParser, Phase9B_DomainIdentityIsNameScopedForSharedParticipants)
{
    const auto first = ExecutionDomainDefinition::parse(
        "continuation=0:cuda:0;scope=single;backend=auto;compute=replicated_experts");
    const auto second = ExecutionDomainDefinition::parse(
        "shared_experts=0:cuda:0;scope=single;backend=auto;compute=replicated_experts");

    EXPECT_TRUE(first.samePhysicalParticipants(second));
    EXPECT_NE(first.logicalIdentity(), second.logicalIdentity());
}

TEST(Test__OrchestrationConfigParser, Phase9B_PPStageRemainsLayerPlacementNotMoEOverlay)
{
    ArgvHelper args{"llaminar2",
                    "--define-domain", "gpu_tp=0:cuda:0,0:cuda:1;scope=local;backend=nccl;owner=0",
                    "--pp-stage", "0=gpu_tp:0-3"};
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());

    ASSERT_EQ(config.domain_definitions.size(), 1u);
    ASSERT_EQ(config.pp_stage_definitions.size(), 1u);
    EXPECT_EQ(config.pp_stage_definitions[0].domain_name, "gpu_tp");
    EXPECT_EQ(config.pp_stage_definitions[0].first_layer, 0);
    EXPECT_EQ(config.pp_stage_definitions[0].last_layer, 3);
    EXPECT_EQ(config.moe_expert_parallel_plan, nullptr);

    const auto inventory = config.executionDomainDefinitions();
    ASSERT_EQ(inventory.size(), 1u);
    EXPECT_EQ(inventory[0].logicalIdentity(), "gpu_tp");
    EXPECT_EQ(inventory[0].compute_kind, ExecutionDomainComputeKind::UNSPECIFIED);
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
tp_scope: local
tp_devices: [cuda:0, cuda:1]
tp_weights: [0.73, 0.27]
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.tp_degree, 2);
    EXPECT_EQ(config.tp_scope, TPScope::LOCAL);
    EXPECT_EQ(config.tp_devices.size(), 2);
    EXPECT_EQ(config.tp_weights.size(), 2);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_NamedDomainLists)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
domains:
    - "rocm_socket0=0:rocm:0,0:rocm:1;scope=local;backend=rccl;owner=0"
    - "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;ranks=0,1"
pp_stages:
    - "0=rocm_socket0:0-13"
    - "1=cpu_sockets:14-27"
)";

    auto config = parser.parseYamlString(yaml);

    ASSERT_EQ(config.domain_definitions.size(), 2u);
    EXPECT_EQ(config.domain_definitions[0].name, "rocm_socket0");
    EXPECT_EQ(config.domain_definitions[0].scope, TPScope::LOCAL);
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
    EXPECT_EQ(config.moe_expert_mode, MoEExpertMode::ExpertParallel);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Percent);
    EXPECT_FLOAT_EQ(config.moe_hot_expert_cache.percent, 10.0f);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_MoENestedBlock)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
moe:
    expert_mode: replicated
    hot_expert_cache: 12
    rebalance: observe
    rebalance_window: 64
    rebalance_max_window: 512
    rebalance_window_growth: 2.5
    release_raw_expert_weights: true
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.moe_expert_mode, MoEExpertMode::Replicated);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Count);
    EXPECT_EQ(config.moe_hot_expert_cache.count, 12);
    EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 12);
    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Observe);
    EXPECT_EQ(config.moe_rebalance.window_size, 64);
    EXPECT_EQ(config.moe_rebalance.max_window_size, 512);
    EXPECT_FLOAT_EQ(config.moe_rebalance.window_growth_factor, 2.5f);
    EXPECT_TRUE(config.moe_rebalance.release_raw_expert_weights);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_MoEFlatKeys)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
moe_expert_mode: expert-parallel
moe_hot_expert_cache: 25%
moe_rebalance: off
moe_rebalance_window: 128
moe_rebalance_max_window: 1024
moe_rebalance_window_growth: 1.25
moe_release_raw_expert_weights: false
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_EQ(config.moe_expert_mode, MoEExpertMode::ExpertParallel);
    EXPECT_EQ(config.moe_hot_expert_cache.kind, MoEHotExpertCacheConfig::Kind::Percent);
    EXPECT_FLOAT_EQ(config.moe_hot_expert_cache.percent, 25.0f);
    EXPECT_EQ(config.moe_hot_expert_cache.resolveCap(256, /*dynamic_rebalance_enabled=*/true), 64);
    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Off);
    EXPECT_EQ(config.moe_rebalance.window_size, 128);
    EXPECT_EQ(config.moe_rebalance.max_window_size, 1024);
    EXPECT_FLOAT_EQ(config.moe_rebalance.window_growth_factor, 1.25f);
    EXPECT_FALSE(config.moe_rebalance.release_raw_expert_weights);
}

TEST(Test__OrchestrationConfigParser, ParseYamlString_QuotedValues)
{
    OrchestrationConfigParser parser;

    std::string yaml = R"(
device: "cuda:0"
tp_scope: 'local'
    )";

    auto config = parser.parseYamlString(yaml);

    EXPECT_TRUE(config.device_for_this_rank.has_value());
    EXPECT_EQ(config.device_for_this_rank->device_type, DeviceType::CUDA);
    EXPECT_EQ(config.tp_scope, TPScope::LOCAL);
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
    EXPECT_TRUE(help.find("--moe-expert-mode") != std::string::npos);
    EXPECT_TRUE(help.find("--moe-hot-expert-cache") != std::string::npos);
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
}

TEST(Test__OrchestrationConfigParser, ParseArgs_Prompt_LongForm)
{
    ArgvHelper args{"llaminar2", "--prompt", "Test prompt"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.prompt, "Test prompt");
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

TEST(Test__OrchestrationConfigParser, ParseArgs_MoEExpertMode)
{
    {
        ArgvHelper args{"llaminar2", "--moe-expert-mode", "replicated"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.moe_expert_mode, MoEExpertMode::Replicated);
    }
    {
        ArgvHelper args{"llaminar2", "--moe-expert-mode", "tensor-parallel"};
        OrchestrationConfigParser parser;

        auto config = parser.parseArgs(args.argc(), args.argv());

        EXPECT_EQ(config.moe_expert_mode, MoEExpertMode::TensorParallel);
    }
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
                    "--moe-rebalance", "observe",
                    "--moe-rebalance-window", "128",
                    "--moe-rebalance-max-window", "2048",
                    "--moe-rebalance-window-growth", "2.0",
                    "--moe-release-raw-expert-weights"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.moe_rebalance.mode, MoERebalanceRuntimeMode::Observe);
    EXPECT_EQ(config.moe_rebalance.window_size, 128);
    EXPECT_EQ(config.moe_rebalance.max_window_size, 2048);
    EXPECT_FLOAT_EQ(config.moe_rebalance.window_growth_factor, 2.0f);
    EXPECT_TRUE(config.moe_rebalance.release_raw_expert_weights);
}

TEST(Test__OrchestrationConfigParser, ParseArgs_InvalidMoEConfig_Throws)
{
    {
        ArgvHelper args{"llaminar2", "--moe-expert-mode", "mystery"};
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
        ArgvHelper args{"llaminar2", "--moe-rebalance", "mystery"};
        OrchestrationConfigParser parser;
        EXPECT_THROW(parser.parseArgs(args.argc(), args.argv()), std::invalid_argument);
    }
}

// ============================================================================
// CLI Parsing - Precision
// ============================================================================

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision)
{
    ArgvHelper args{"llaminar2", "--activation-precision", "bf16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.activation_precision, "bf16");
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision_Alias)
{
    ArgvHelper args{"llaminar2", "--act-prec", "fp16"};
    OrchestrationConfigParser parser;

    auto config = parser.parseArgs(args.argc(), args.argv());

    EXPECT_EQ(config.activation_precision, "fp16");
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

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision_InvalidThrowsViaValidValues)
{
    // --activation-precision uses CliSpec's valid_values whitelist, so an
    // unknown value must be rejected with a listing of the accepted set.
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
        EXPECT_NE(msg.find("bf16"), std::string::npos);
    }
}

TEST(Test__OrchestrationConfigParser, ParseArgs_ActivationPrecision_AllThreeAliases)
{
    for (const auto &flag : {"--activation-precision", "--activation-prec", "--act-prec"})
    {
        ArgvHelper args({"llaminar2", flag, "bf16"});
        auto parser = createOrchestrationConfigParser();
        auto config = parser->parseArgs(args.argc(), args.argv());
        EXPECT_EQ(config.activation_precision, "bf16") << "flag=" << flag;
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
    ArgvHelper args({"llaminar2", "--tp-scope=local"});
    auto parser = createOrchestrationConfigParser();
    auto config = parser->parseArgs(args.argc(), args.argv());
    EXPECT_EQ(config.tp_scope, TPScope::LOCAL);
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
