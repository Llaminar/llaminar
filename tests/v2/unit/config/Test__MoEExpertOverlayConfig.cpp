#include <gtest/gtest.h>

#include "config/OrchestrationConfigParser.h"

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
    EXPECT_EQ(plan.shared_expert_domain, "rocm_hot");
    EXPECT_EQ(plan.residency_policy, ExpertResidencyPolicy::HistogramTieredCache);

    ASSERT_EQ(plan.domains.size(), 2u);
    EXPECT_EQ(plan.domains[0].name, "rocm_hot");
    EXPECT_EQ(plan.domains[0].kind, ExpertDomainKind::LocalTP);
    EXPECT_EQ(plan.domains[0].backend, CollectiveBackendType::RCCL);
    EXPECT_EQ(plan.domains[0].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
    ASSERT_EQ(plan.domains[0].participants.size(), 2u);
    EXPECT_EQ(plan.domains[0].participants[0].device_type, DeviceType::ROCm);

    EXPECT_EQ(plan.domains[1].name, "cpu_cold");
    EXPECT_EQ(plan.domains[1].kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(plan.domains[1].backend, CollectiveBackendType::UPI);
    EXPECT_EQ(plan.domains[1].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);
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
    EXPECT_EQ(plan.shared_expert_domain, "cuda_fast");
    EXPECT_EQ(plan.residency_policy, ExpertResidencyPolicy::StaticById);

    ASSERT_EQ(plan.domains.size(), 3u);
    EXPECT_EQ(plan.domains[0].name, "cuda_fast");
    EXPECT_EQ(plan.domains[0].kind, ExpertDomainKind::SingleDevice);
    EXPECT_EQ(plan.domains[0].compute_kind, ExpertDomainComputeKind::ReplicatedExperts);
    ASSERT_EQ(plan.domains[0].participants.size(), 1u);
    EXPECT_EQ(plan.domains[0].participants[0].device_type, DeviceType::CUDA);

    EXPECT_EQ(plan.domains[1].name, "rocm_warm");
    EXPECT_EQ(plan.domains[1].kind, ExpertDomainKind::LocalTP);
    EXPECT_EQ(plan.domains[1].backend, CollectiveBackendType::RCCL);
    EXPECT_EQ(plan.domains[1].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);

    EXPECT_EQ(plan.domains[2].name, "cpu_cold");
    EXPECT_EQ(plan.domains[2].kind, ExpertDomainKind::NodeLocalTP);
    EXPECT_EQ(plan.domains[2].backend, CollectiveBackendType::UPI);
    EXPECT_EQ(plan.domains[2].compute_kind, ExpertDomainComputeKind::TensorParallelExperts);

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

} // namespace

TEST(Test__MoEExpertOverlayConfig, ParseArgs_ConstructsRocmLocalTPAndCpuNodeLocalTPTieredPlan)
{
    ArgvHelper args{"llaminar2",
                    "--moe-expert-overlay", "tiered",
                    "--moe-expert-overlay-continuation", "rocm_hot",
                    "--moe-expert-overlay-shared-domain", "rocm_hot",
                    "--moe-expert-overlay-residency", "histogram",
                    "--moe-expert-overlay-domain", "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts",
                    "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts",
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
    - "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts"
    - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts"
  routed_tiers:
    - "hot@rocm_hot;priority=0;max-experts-per-layer=8;memory-mb=auto"
    - "cold@cpu_cold;priority=1;fallback=true"
)";

    const auto config = parser.parseYamlString(yaml);

    expectTwoTierRocmCpuPlan(requirePlan(config));
    expectValidOverlay(config);
    EXPECT_TRUE(config.domain_definitions.empty());
    EXPECT_TRUE(config.pp_stage_definitions.empty());
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
        - "rocm_hot=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts"
        - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts"
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
                    "--moe-expert-overlay-domain", "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts",
                    "--moe-expert-overlay-domain", "rocm_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts",
                    "--moe-expert-overlay-domain", "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts",
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
    - "cuda_fast=0:cuda:0;scope=single;backend=auto;compute=replicated_experts"
    - "rocm_warm=0:rocm:0,0:rocm:1;scope=local;backend=rccl;compute=tensor_parallel_experts"
    - "cpu_cold=0:cpu:0,1:cpu:0;scope=node_local;backend=upi;compute=tensor_parallel_experts"
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
}
