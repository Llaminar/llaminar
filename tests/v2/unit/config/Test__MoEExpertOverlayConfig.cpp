#include <gtest/gtest.h>

#include "config/OrchestrationConfigParser.h"
#include "execution/moe/MoEExpertOverlayExecutionPlan.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{

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

const MoEExpertParallelPlan &requirePlan(const OrchestrationConfig &config)
{
    EXPECT_NE(config.moe_expert_parallel_plan, nullptr);
    return *config.moe_expert_parallel_plan;
}

void expectValidOverlay(const OrchestrationConfig &config)
{
    const auto errors = config.validate();
    EXPECT_TRUE(errors.empty()) << (errors.empty() ? "" : errors.front());
}

bool hasErrorContaining(const std::vector<std::string> &errors, const std::string &needle)
{
    return std::any_of(errors.begin(), errors.end(), [&](const std::string &error) {
        return error.find(needle) != std::string::npos;
    });
}

void expectParseThrowsContaining(std::initializer_list<const char *> argv, const std::string &needle)
{
    ArgvHelper args(argv);
    OrchestrationConfigParser parser;
    try
    {
        (void)parser.parseArgs(args.argc(), args.argv());
        FAIL() << "Expected parseArgs to throw";
    }
    catch (const std::invalid_argument &e)
    {
        EXPECT_NE(std::string(e.what()).find(needle), std::string::npos) << e.what();
    }
}

void expectTwoTierRocmCpuPlan(const MoEExpertParallelPlan &plan)
{
    EXPECT_TRUE(plan.enabled);
    EXPECT_EQ(plan.execution_kind, MoEExpertExecutionKind::TieredExpertOverlay);
    EXPECT_EQ(plan.continuation_domain, "rocm_hot");
    EXPECT_EQ(plan.effectiveBaseModelDomain(), "rocm_hot");
    EXPECT_EQ(plan.shared_expert_domain, "rocm_hot");
    EXPECT_EQ(plan.residency_policy, ExpertResidencyPolicy::HistogramTieredCache);

    ASSERT_EQ(plan.domains.size(), 2u);
    EXPECT_EQ(plan.domains[0].name, "rocm_hot");
    EXPECT_EQ(plan.domains[0].kind, ExpertDomainKind::LocalTP);
    EXPECT_EQ(plan.domains[0].backend, CollectiveBackendType::RCCL);
    EXPECT_EQ(plan.domains[0].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    EXPECT_EQ(plan.domains[0].owner_rank, 0);
    ASSERT_EQ(plan.domains[0].participants.size(), 2u);
    EXPECT_EQ(plan.domains[0].participants[0].device_type, DeviceType::ROCm);

    EXPECT_EQ(plan.domains[1].name, "cpu_cold");
    EXPECT_EQ(plan.domains[1].kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(plan.domains[1].backend, CollectiveBackendType::UPI);
    EXPECT_EQ(plan.domains[1].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    ASSERT_EQ(plan.domains[1].world_ranks.size(), 2u);
    EXPECT_EQ(plan.domains[1].world_ranks[0], 0);
    EXPECT_EQ(plan.domains[1].world_ranks[1], 1);
    ASSERT_EQ(plan.domains[1].participants.size(), 2u);
    EXPECT_EQ(plan.domains[1].participants[0].device_type, DeviceType::CPU);

    ASSERT_EQ(plan.routed_tiers.size(), 2u);
    EXPECT_EQ(plan.routed_tiers[0].name, "hot");
    EXPECT_EQ(plan.routed_tiers[0].domain, "rocm_hot");
    EXPECT_EQ(plan.routed_tiers[0].priority, 0);
    EXPECT_EQ(plan.routed_tiers[0].max_experts_per_layer, 8);
    EXPECT_EQ(plan.routed_tiers[0].memory_budget_bytes, 0u);
    EXPECT_FALSE(plan.routed_tiers[0].fallback);
    EXPECT_EQ(plan.routed_tiers[1].name, "cold");
    EXPECT_TRUE(plan.routed_tiers[1].fallback);
}

void expectThreeTierCudaRocmCpuPlan(const MoEExpertParallelPlan &plan)
{
    EXPECT_TRUE(plan.enabled);
    EXPECT_EQ(plan.execution_kind, MoEExpertExecutionKind::TieredExpertOverlay);
    EXPECT_EQ(plan.continuation_domain, "cuda_fast");
    EXPECT_EQ(plan.effectiveBaseModelDomain(), "cuda_fast");
    EXPECT_EQ(plan.shared_expert_domain, "cuda_fast");
    EXPECT_EQ(plan.residency_policy, ExpertResidencyPolicy::StaticById);

    ASSERT_EQ(plan.domains.size(), 3u);
    EXPECT_EQ(plan.domains[0].name, "cuda_fast");
    EXPECT_EQ(plan.domains[0].kind, ExpertDomainKind::SingleDevice);
    EXPECT_EQ(plan.domains[0].compute_kind, ExpertDomainComputeKind::ReplicatedExperts);
    EXPECT_EQ(plan.domains[0].owner_rank, 0);
    ASSERT_EQ(plan.domains[0].participants.size(), 1u);
    EXPECT_EQ(plan.domains[0].participants[0].device_type, DeviceType::CUDA);

    EXPECT_EQ(plan.domains[1].name, "rocm_warm");
    EXPECT_EQ(plan.domains[1].kind, ExpertDomainKind::LocalTP);
    EXPECT_EQ(plan.domains[1].backend, CollectiveBackendType::RCCL);
    EXPECT_EQ(plan.domains[1].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    EXPECT_EQ(plan.domains[1].owner_rank, 1);

    EXPECT_EQ(plan.domains[2].name, "cpu_cold");
    EXPECT_EQ(plan.domains[2].kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(plan.domains[2].backend, CollectiveBackendType::UPI);
    EXPECT_EQ(plan.domains[2].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    ASSERT_EQ(plan.domains[2].world_ranks.size(), 2u);
    EXPECT_EQ(plan.domains[2].world_ranks[0], 0);
    EXPECT_EQ(plan.domains[2].world_ranks[1], 2);

    ASSERT_EQ(plan.routed_tiers.size(), 3u);
    EXPECT_EQ(plan.routed_tiers[0].name, "hottest");
    EXPECT_EQ(plan.routed_tiers[0].domain, "cuda_fast");
    EXPECT_EQ(plan.routed_tiers[0].priority, 0);
    EXPECT_EQ(plan.routed_tiers[1].name, "warm");
    EXPECT_EQ(plan.routed_tiers[1].domain, "rocm_warm");
    EXPECT_EQ(plan.routed_tiers[1].priority, 1);
    EXPECT_EQ(plan.routed_tiers[2].name, "cold");
    EXPECT_TRUE(plan.routed_tiers[2].fallback);
}

MoEExpertOverlayExecutionPlan resolveOverlayExecutionPlan(
    const OrchestrationConfig &config,
    int current_world_rank,
    int world_size)
{
    EXPECT_NE(config.moe_expert_parallel_plan, nullptr);
    return resolveMoEExpertOverlayExecutionPlan(
        config.moe_expert_parallel_plan,
        MoEExpertOverlayExecutionPlanResolverOptions{
            .current_world_rank = current_world_rank,
            .world_size = world_size,
        });
}

void expectEquivalentExecutionPlanDiagnostics(
    const OrchestrationConfig &cli_config,
    const OrchestrationConfig &yaml_config,
    int current_world_rank,
    int world_size)
{
    const auto cli_plan = resolveOverlayExecutionPlan(cli_config, current_world_rank, world_size);
    const auto yaml_plan = resolveOverlayExecutionPlan(yaml_config, current_world_rank, world_size);

    EXPECT_EQ(cli_plan.continuation_domain, yaml_plan.continuation_domain);
    EXPECT_EQ(cli_plan.shared_expert_domain, yaml_plan.shared_expert_domain);
    EXPECT_EQ(cli_plan.continuation_root_rank, yaml_plan.continuation_root_rank);
    EXPECT_EQ(cli_plan.diagnostics(), yaml_plan.diagnostics());
}

} // namespace

TEST(Test__MoEExpertOverlayConfig, ParseArgs_ConstructsRocmLocalTPAndCpuNodeLocalTPTieredPlan)
{
    ArgvHelper args{"llaminar2",
                    "--moe-expert-overlay", "tiered",
                    "--moe-expert-overlay-continuation", "rocm_hot",
                    "--moe-expert-overlay-shared-domain", "rocm_hot",
                    "--moe-expert-overlay-residency", "histogram",
                    "--moe-expert-overlay-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0",
                    "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                    "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0;max-experts-per-layer=8;memory-mb=auto",
                    "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"};
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());

    expectTwoTierRocmCpuPlan(requirePlan(config));
    expectValidOverlay(config);
}

TEST(Test__MoEExpertOverlayConfig, ParseYaml_ConstructsRocmLocalTPAndCpuNodeLocalTPTieredPlan)
{
    OrchestrationConfigParser parser;
    const std::string yaml = R"(
moe_expert_parallel:
  enabled: true
  execution_kind: tiered_expert_overlay
  continuation_domain: rocm_hot
  shared_expert_domain: rocm_hot
  residency:
    mode: histogram
  domains:
    - "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0"
    - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1"
  routed_tiers:
    - "hot@rocm_hot;priority=0;max-experts-per-layer=8;memory-mb=auto"
    - "cold@cpu_cold;priority=1;fallback=true"
)";

    const auto config = parser.parseYamlString(yaml);

    expectTwoTierRocmCpuPlan(requirePlan(config));
    expectValidOverlay(config);
    ASSERT_EQ(config.domain_definitions.size(), 2u);
    EXPECT_EQ(config.domain_definitions[0].name, "rocm_hot");
    EXPECT_TRUE(config.pp_stage_definitions.empty());
}

TEST(Test__MoEExpertOverlayConfig, LayoutA_CliAndYamlResolveIdenticalExecutionPlanWithoutDeviceFlag)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--define-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0",
                    "--define-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                    "--moe-expert-overlay", "tiered",
                    "--moe-expert-overlay-continuation", "rocm_hot",
                    "--moe-expert-overlay-base-domain", "rocm_hot",
                    "--moe-expert-overlay-shared-domain", "rocm_hot",
                    "--moe-expert-overlay-residency", "histogram",
                    "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0;max-experts-per-layer=8;memory-mb=auto",
                    "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"};
    const auto cli_config = parser.parseArgs(args.argc(), args.argv());

    const auto yaml_config = parser.parseYamlString(R"(
domains:
    - "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0"
    - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1"
moe_expert_parallel:
    enabled: true
    execution_kind: tiered
    continuation_domain: rocm_hot
    base_model_domain: rocm_hot
    shared_expert_domain: rocm_hot
    residency:
        mode: histogram
    routed_tiers:
        - "hot@rocm_hot;priority=0;max-experts-per-layer=8;memory-mb=auto"
        - "cold@cpu_cold;priority=1;fallback=true"
)");

    EXPECT_FALSE(cli_config.device_for_this_rank.has_value());
    EXPECT_FALSE(yaml_config.device_for_this_rank.has_value());
    ASSERT_EQ(cli_config.domain_definitions.size(), 2u);
    ASSERT_EQ(yaml_config.domain_definitions.size(), 2u);
    ASSERT_EQ(cli_config.moe_expert_parallel_plan->domains.size(), 2u);
    ASSERT_EQ(yaml_config.moe_expert_parallel_plan->domains.size(), 2u);
    expectValidOverlay(cli_config);
    expectValidOverlay(yaml_config);
    expectEquivalentExecutionPlanDiagnostics(cli_config, yaml_config, 0, 2);
}

TEST(Test__MoEExpertOverlayConfig, ParseArgs_ConfigYamlCanBeCompletedByCliBeforeOverlayValidation)
{
    const auto path = std::filesystem::temp_directory_path() / "llaminar_moe_overlay_base.yaml";
    {
        std::ofstream out(path);
        out << R"(
moe_expert_parallel:
    enabled: true
    execution_kind: tiered
    continuation_domain: rocm_hot
    shared_expert_domain: rocm_hot
    residency:
        mode: histogram
    domains:
        - "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0"
        - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1"
)";
    }

    const std::string path_string = path.string();
    ArgvHelper args{"llaminar2",
                    "--config", path_string.c_str(),
                    "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0;max-experts-per-layer=8;memory-mb=auto",
                    "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"};
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());

    expectTwoTierRocmCpuPlan(requirePlan(config));
    expectValidOverlay(config);

    std::filesystem::remove(path);
}

TEST(Test__MoEExpertOverlayConfig, ParseArgs_ConstructsCudaRocmCpuThreeTierPlan)
{
    ArgvHelper args{"llaminar2",
                    "--moe-expert-overlay", "tiered",
                    "--moe-expert-overlay-continuation", "cuda_fast",
                    "--moe-expert-overlay-shared-domain", "cuda_fast",
                    "--moe-expert-overlay-residency", "static-by-id",
                    "--moe-expert-overlay-domain", "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts;owner=0",
                    "--moe-expert-overlay-domain", "rocm_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=1",
                    "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,2",
                    "--moe-expert-overlay-tier", "hottest@cuda_fast;priority=0;max-experts-per-layer=4;memory-mb=512",
                    "--moe-expert-overlay-tier", "warm@rocm_warm;priority=1;max-experts-per-layer=8;memory-mb=auto",
                    "--moe-expert-overlay-tier", "cold@cpu_cold;priority=2;fallback=true"};
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());

    expectThreeTierCudaRocmCpuPlan(requirePlan(config));
    EXPECT_EQ(config.moe_expert_parallel_plan->routed_tiers[0].memory_budget_bytes, 512ULL * 1024ULL * 1024ULL);
    expectValidOverlay(config);
}

TEST(Test__MoEExpertOverlayConfig, ParseYaml_ConstructsCudaRocmCpuThreeTierPlan)
{
    OrchestrationConfigParser parser;
    const std::string yaml = R"(
moe_expert_parallel:
  enabled: true
  execution_kind: tiered
  continuation_domain: cuda_fast
  shared_expert_domain: cuda_fast
  residency:
    mode: static-by-id
  domains:
    - "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts;owner=0"
    - "rocm_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=1"
    - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,2"
  routed_tiers:
    - "hottest@cuda_fast;priority=0;max-experts-per-layer=4;memory-mb=512"
    - "warm@rocm_warm;priority=1;max-experts-per-layer=8;memory-mb=auto"
    - "cold@cpu_cold;priority=2;fallback=true"
)";

    const auto config = parser.parseYamlString(yaml);

    expectThreeTierCudaRocmCpuPlan(requirePlan(config));
    EXPECT_EQ(config.moe_expert_parallel_plan->routed_tiers[0].memory_budget_bytes, 512ULL * 1024ULL * 1024ULL);
    expectValidOverlay(config);
}

TEST(Test__MoEExpertOverlayConfig, LayoutB_CliAndYamlResolveIdenticalExecutionPlanWithoutDeviceFlag)
{
    OrchestrationConfigParser parser;
    ArgvHelper args{"llaminar2",
                    "--define-domain", "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts;owner=0",
                    "--define-domain", "rocm_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=1",
                    "--define-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,2",
                    "--moe-expert-overlay", "tiered",
                    "--moe-expert-overlay-continuation", "cuda_fast",
                    "--base-model-domain", "cuda_fast",
                    "--moe-expert-overlay-shared-domain", "cuda_fast",
                    "--moe-expert-overlay-residency", "static-by-id",
                    "--moe-expert-overlay-tier", "hottest@cuda_fast;priority=0;max-experts-per-layer=4;memory-mb=512",
                    "--moe-expert-overlay-tier", "warm@rocm_warm;priority=1;max-experts-per-layer=8;memory-mb=auto",
                    "--moe-expert-overlay-tier", "cold@cpu_cold;priority=2;fallback=true"};
    const auto cli_config = parser.parseArgs(args.argc(), args.argv());

    const auto yaml_config = parser.parseYamlString(R"(
domains:
    - "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts;owner=0"
    - "rocm_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=1"
    - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,2"
moe_expert_parallel:
    enabled: true
    execution_kind: tiered
    continuation_domain: cuda_fast
    base_model_domain: cuda_fast
    shared_expert_domain: cuda_fast
    residency:
        mode: static-by-id
    routed_tiers:
        - "hottest@cuda_fast;priority=0;max-experts-per-layer=4;memory-mb=512"
        - "warm@rocm_warm;priority=1;max-experts-per-layer=8;memory-mb=auto"
        - "cold@cpu_cold;priority=2;fallback=true"
)");

    EXPECT_FALSE(cli_config.device_for_this_rank.has_value());
    EXPECT_FALSE(yaml_config.device_for_this_rank.has_value());
    ASSERT_EQ(cli_config.domain_definitions.size(), 3u);
    ASSERT_EQ(yaml_config.domain_definitions.size(), 3u);
    ASSERT_EQ(cli_config.moe_expert_parallel_plan->domains.size(), 3u);
    ASSERT_EQ(yaml_config.moe_expert_parallel_plan->domains.size(), 3u);
    expectValidOverlay(cli_config);
    expectValidOverlay(yaml_config);
    expectEquivalentExecutionPlanDiagnostics(cli_config, yaml_config, 0, 3);
}

TEST(Test__MoEExpertOverlayConfig, Phase9C_OverlayDomainAliasFeedsCanonicalInventory)
{
    OrchestrationConfigParser parser;
    ArgvHelper alias_args{"llaminar2",
                          "--moe-expert-overlay", "tiered",
                          "--moe-expert-overlay-continuation", "rocm_hot",
                          "--moe-expert-overlay-shared-domain", "rocm_hot",
                          "--moe-expert-overlay-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0",
                          "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                          "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0",
                          "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"};
    ArgvHelper named_args{"llaminar2",
                          "--define-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0",
                          "--define-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                          "--moe-expert-overlay", "tiered",
                          "--moe-expert-overlay-continuation", "rocm_hot",
                          "--moe-expert-overlay-shared-domain", "rocm_hot",
                          "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0",
                          "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"};

    const auto alias_config = parser.parseArgs(alias_args.argc(), alias_args.argv());
    const auto named_config = parser.parseArgs(named_args.argc(), named_args.argv());

    const auto alias_inventory = alias_config.executionDomainDefinitions();
    const auto named_inventory = named_config.executionDomainDefinitions();
    ASSERT_EQ(alias_inventory.size(), named_inventory.size());
    for (size_t index = 0; index < alias_inventory.size(); ++index)
    {
        EXPECT_EQ(alias_inventory[index].name, named_inventory[index].name);
        EXPECT_EQ(alias_inventory[index].participants, named_inventory[index].participants);
        EXPECT_EQ(alias_inventory[index].backend, named_inventory[index].backend);
        EXPECT_EQ(alias_inventory[index].compute_kind, named_inventory[index].compute_kind);
    }
}

TEST(Test__MoEExpertOverlayConfig, Phase9C_OverlayDomainAliasRejectsConflictingDefineDomain)
{
    expectParseThrowsContaining({"llaminar2",
                                 "--define-domain", "rocm_hot=0:rocm:0;scope=single;backend=auto;compute=replicated_experts;owner=0",
                                 "--moe-expert-overlay", "tiered",
                                 "--moe-expert-overlay-continuation", "rocm_hot",
                                 "--moe-expert-overlay-shared-domain", "rocm_hot",
                                 "--moe-expert-overlay-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0",
                                 "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                                 "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0",
                                 "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"},
                                "Conflicting execution domain definition");
}

TEST(Test__MoEExpertOverlayConfig, ParseArgs_ConstructsSingleDomainCpuSocketExpertParallelPlan)
{
    ArgvHelper args{"llaminar2",
                    "--moe-expert-overlay", "single-domain",
                    "--moe-expert-overlay-continuation", "cpu_sockets",
                    "--moe-expert-overlay-shared-domain", "cpu_sockets",
                    "--moe-expert-overlay-domain", "cpu_sockets=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=expert_id_sharded",
                    "--moe-expert-overlay-tier", "routed@cpu_sockets;priority=0;fallback=true"};
    OrchestrationConfigParser parser;

    const auto config = parser.parseArgs(args.argc(), args.argv());
    const auto &plan = requirePlan(config);

    EXPECT_TRUE(plan.enabled);
    EXPECT_EQ(plan.execution_kind, MoEExpertExecutionKind::SingleDomainExpertSharded);
    ASSERT_EQ(plan.domains.size(), 1u);
    EXPECT_EQ(plan.domains[0].kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(plan.domains[0].compute_kind, ExpertDomainComputeKind::ExpertIdSharded);
    ASSERT_EQ(plan.routed_tiers.size(), 1u);
    EXPECT_TRUE(plan.routed_tiers[0].fallback);
    expectValidOverlay(config);
}

TEST(Test__MoEExpertOverlayConfig, Validate_RejectsInvalidOverlayConfigs)
{
    OrchestrationConfig missing_tiers;
    missing_tiers.moe_expert_parallel_plan = std::make_shared<MoEExpertParallelPlan>();
    missing_tiers.moe_expert_parallel_plan->enabled = true;
    missing_tiers.moe_expert_parallel_plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
    missing_tiers.moe_expert_parallel_plan->continuation_domain = "gpu";
    missing_tiers.moe_expert_parallel_plan->shared_expert_domain = "gpu";
    missing_tiers.moe_expert_parallel_plan->domains.push_back(ExpertComputeDomain{
        .name = "gpu",
        .kind = ExpertDomainKind::SingleDevice,
        .backend = CollectiveBackendType::AUTO,
        .participants = {GlobalDeviceAddress::cuda(0)},
        .compute_kind = ExpertDomainComputeKind::ReplicatedExperts,
    });

    auto errors = missing_tiers.validate();
    EXPECT_TRUE(hasErrorContaining(errors, "routed tier"));

    OrchestrationConfig pp_overlap = missing_tiers;
    pp_overlap.moe_expert_parallel_plan->routed_tiers.push_back(ExpertRoutedTier{
        .name = "hot",
        .domain = "gpu",
        .priority = 0,
        .max_experts_per_layer = 0,
        .memory_budget_bytes = 0,
        .fallback = true,
    });
    pp_overlap.pp_stage_definitions.push_back(PPStageDefinition::parse("0=gpu:0-1"));
    errors = pp_overlap.validate();
    EXPECT_TRUE(hasErrorContaining(errors, "same-layer expert roles"));

    OrchestrationConfig off_with_domain;
    off_with_domain.moe_expert_parallel_plan = std::make_shared<MoEExpertParallelPlan>();
    off_with_domain.moe_expert_parallel_plan->enabled = false;
    off_with_domain.moe_expert_parallel_plan->domains.push_back(missing_tiers.moe_expert_parallel_plan->domains[0]);
    errors = off_with_domain.validate();
    EXPECT_TRUE(hasErrorContaining(errors, "off/disabled"));

    OrchestrationConfig unsupported_auxiliary;
    unsupported_auxiliary.moe_expert_parallel_plan = std::make_shared<MoEExpertParallelPlan>();
    unsupported_auxiliary.moe_expert_parallel_plan->enabled = true;
    unsupported_auxiliary.moe_expert_parallel_plan->execution_kind = MoEExpertExecutionKind::TieredExpertOverlay;
    unsupported_auxiliary.moe_expert_parallel_plan->continuation_domain = "cuda_fast";
    unsupported_auxiliary.moe_expert_parallel_plan->shared_expert_domain = "cuda_fast";
    unsupported_auxiliary.moe_expert_parallel_plan->domains = {
        ExpertComputeDomain{
            .name = "cuda_fast",
            .kind = ExpertDomainKind::SingleDevice,
            .backend = CollectiveBackendType::AUTO,
            .participants = {GlobalDeviceAddress::cuda(0)},
            .owner_rank = 0,
            .compute_kind = ExpertDomainComputeKind::ReplicatedExperts,
        },
        ExpertComputeDomain{
            .name = "remote_gpu",
            .kind = ExpertDomainKind::SingleDevice,
            .backend = CollectiveBackendType::NCCL,
            .participants = {GlobalDeviceAddress::cuda(1)},
            .owner_rank = 1,
            .compute_kind = ExpertDomainComputeKind::ReplicatedExperts,
        },
        ExpertComputeDomain{
            .name = "cpu_cold",
            .kind = ExpertDomainKind::NodeLocalTP,
            .backend = CollectiveBackendType::UPI,
            .participants = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)},
            .world_ranks = {0, 2},
            .compute_kind = ExpertDomainComputeKind::TensorParallelExperts,
        },
    };
    unsupported_auxiliary.moe_expert_parallel_plan->routed_tiers = {
        ExpertRoutedTier{.name = "fast", .domain = "cuda_fast", .priority = 0},
        ExpertRoutedTier{.name = "remote", .domain = "remote_gpu", .priority = 1},
        ExpertRoutedTier{.name = "cold", .domain = "cpu_cold", .priority = 2, .fallback = true},
    };
    errors = unsupported_auxiliary.validate();
    EXPECT_TRUE(hasErrorContaining(errors, "auxiliary domain 'remote_gpu'"));
    EXPECT_TRUE(hasErrorContaining(errors, "no Phase 6 worker implementation"));
}

TEST(Test__MoEExpertOverlayConfig, ParseArgs_InvalidConfigsThrowBeforeExecution)
{
    expectParseThrowsContaining({"llaminar2",
                                 "--moe-expert-overlay", "tiered",
                                 "--moe-expert-overlay-continuation", "gpu",
                                 "--moe-expert-overlay-shared-domain", "gpu",
                                 "--moe-expert-overlay-domain", "gpu=0:cuda:0;scope=single;backend=auto;compute=replicated_experts"},
                                "routed tier");

    expectParseThrowsContaining({"llaminar2",
                                 "--moe-expert-overlay", "off",
                                 "--moe-expert-overlay-domain", "gpu=0:cuda:0;scope=single;backend=auto;compute=replicated_experts"},
                                "off/disabled");

    expectParseThrowsContaining({"llaminar2",
                                 "--moe-expert-overlay", "tiered",
                                 "--moe-expert-overlay-continuation", "gpu",
                                 "--moe-expert-overlay-shared-domain", "gpu",
                                 "--moe-expert-overlay-domain", "gpu=0:cuda:0;scope=single;backend=auto;compute=replicated_experts",
                                 "--moe-expert-overlay-tier", "hot@gpu;priority=0;fallback=true",
                                 "--pp-stage", "0=gpu:0-1"},
                                "same-layer expert roles");

    expectParseThrowsContaining({"llaminar2",
                                 "-d", "cpu",
                                 "--moe-expert-overlay", "tiered",
                                 "--moe-expert-overlay-continuation", "rocm_hot",
                                 "--moe-expert-overlay-shared-domain", "rocm_hot",
                                 "--moe-expert-overlay-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0",
                                 "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                                 "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0",
                                 "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"},
                                "Conflicting options: --device/-d cpu and --moe-expert-overlay-continuation rocm_hot");

    expectParseThrowsContaining({"llaminar2",
                                 "-d", "cuda:0",
                                 "--moe-expert-overlay", "tiered",
                                 "--moe-expert-overlay-continuation", "cuda_fast",
                                 "--moe-expert-overlay-shared-domain", "cuda_fast",
                                 "--moe-expert-overlay-domain", "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts;owner=0",
                                 "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                                 "--moe-expert-overlay-tier", "fast@cuda_fast;priority=0",
                                 "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"},
                                "Conflicting options: --device/-d cuda:0 and --moe-expert-overlay-continuation cuda_fast");

    expectParseThrowsContaining({"llaminar2",
                                 "-d", "cpu",
                                 "--define-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts;owner=0",
                                 "--define-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts;ranks=0,1",
                                 "--moe-expert-overlay", "tiered",
                                 "--moe-expert-overlay-continuation", "rocm_hot",
                                 "--moe-expert-overlay-base-domain", "rocm_hot",
                                 "--moe-expert-overlay-shared-domain", "rocm_hot",
                                 "--moe-expert-overlay-tier", "hot@rocm_hot;priority=0",
                                 "--moe-expert-overlay-tier", "cold@cpu_cold;priority=1;fallback=true"},
                                "Conflicting options: --device/-d cpu and --moe-expert-overlay-base-domain rocm_hot");
}
