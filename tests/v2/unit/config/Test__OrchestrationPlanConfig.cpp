/**
 * @file Test__OrchestrationPlanConfig.cpp
 * @brief Device-free contracts for reusable plans and compact tier declarations.
 *
 * Exercise the public parser and launcher with synthetic configuration only.
 * These tests neither discover devices nor initialize MPI; physical placement
 * remains a later inventory-bound operation and cannot be inferred from names.
 */
#include "config/ExpertTierDefinition.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"
#include "app/MPIBootstrapPhase.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <unistd.h>

using namespace llaminar2;

namespace
{
    /** @return Parsed argv whose backing strings live through the parse call. */
    OrchestrationConfig parse(std::vector<std::string> arguments)
    {
        arguments.insert(arguments.begin(), "llaminar2");
        std::vector<char *> argv;
        for (auto &argument : arguments) argv.push_back(argument.data());
        return OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
    }

    /** @brief Own a unique small config fixture, never a user-selected path. */
    class PlanFile
    {
    public:
        /** @brief Materialize the exact config used by the public --config test. */
        explicit PlanFile(const std::string &contents)
        {
            char pattern[] = "/tmp/llaminar-plan-unit-XXXXXX";
            const int descriptor = mkstemp(pattern);
            if (descriptor < 0) throw std::runtime_error("Cannot create plan test fixture");
            close(descriptor);
            path = pattern;
            std::ofstream output(path);
            output << contents;
        }
        /** @brief Remove only this fixture's uniquely created file. */
        ~PlanFile() { std::remove(path.c_str()); }
        std::string path;
    };
}

TEST(OrchestrationPlanConfig, AutomaticFilterOverridesReplaceYamlSetsWithoutWidening)
{
    const PlanFile input("planning:\n  mode: auto\n  only_backends: cuda,rocm\n  only_strategies: tp,pp\n");
    const auto config = parse({"--only-backends", "rocm", "--only-strategies", "tp", "--config=" + input.path});
    const auto resolved = resolveOrchestrationIntent(config);
    ASSERT_TRUE(std::holds_alternative<AutomaticOrchestrationRequest>(resolved));
    const auto &request = std::get<AutomaticOrchestrationRequest>(resolved);
    EXPECT_TRUE(request.allows(DeviceType::ROCm));
    EXPECT_FALSE(request.allows(DeviceType::CUDA));
    EXPECT_FALSE(request.allows(DeviceType::CPU));
    EXPECT_TRUE(request.allows(OrchestrationStrategy::TensorParallel));
    EXPECT_FALSE(request.allows(OrchestrationStrategy::PipelineParallel));
}

TEST(OrchestrationPlanConfig, CommandPresentationPreservesTheCompleteServingPolicy)
{
    const std::vector<std::string> policy{
        "--auto", "--model", "model.gguf", "--mpi-hostfile", "cluster.hosts",
        "--context-length", "8192", "--kv-cache-precision", "fp32",
        "--only-backends", "rocm,cpu", "--only-strategies", "expert-overlay",
        "--auto-hosts", "all",
        "--mtp", "--mtp-depth-policy", "dynamic", "--mtp-max-draft-tokens", "15",
        "--mtp-graph-capacity-draft-tokens", "15", "--moe-residency-maintenance", "dynamic",
        "--prefix-cache", "--prefix-cache-storage", "tiered"};
    const auto expected = parse(policy);
    std::string output;
    CliSpec<OrchestrationConfig> presentation;
    presentation.add({.long_name = "--output", .value_label = "<path>",
        .setter = [&output](OrchestrationConfig &, const std::string &value) { output = value; }});
    auto arguments = policy;
    arguments.insert(arguments.begin(), "llaminar2");
    arguments.push_back("--output=plan.json");
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    const auto actual = OrchestrationConfigParser{}.parseCommandArgs(argv.size(), argv.data(), presentation);
    EXPECT_EQ(output, "plan.json");
    EXPECT_EQ(serializeOrchestrationConfig(actual), serializeOrchestrationConfig(expected));
    const auto help = OrchestrationConfigParser::getCommandHelpText(presentation, "plan");
    for (const auto *flag : {"--output", "--mtp", "--kv-cache-precision", "--auto", "--mpi-hostfile", "--auto-hosts"})
        EXPECT_NE(help.find(flag), std::string::npos) << flag;
}

TEST(OrchestrationPlanConfig, AutomaticHostPolicySurvivesDocumentAndCliOverride)
{
    const auto request = parse({"--auto-hosts", "all"});
    const auto restored = deserializeOrchestrationConfig(serializeOrchestrationConfig(request));
    EXPECT_EQ(restored.automatic_planning.host_participation, AutomaticHostParticipation::AllDiscovered);
    const PlanFile input("planning:\n  mode: auto\n  hosts: all\n");
    EXPECT_EQ(parse({"--config", input.path, "--auto-hosts", "best-subset"})
                  .automatic_planning.host_participation, AutomaticHostParticipation::BestSubset);
}

TEST(OrchestrationPlanConfig, CommandExtensionCannotRedefineServingPolicy)
{
    CliSpec<OrchestrationConfig> conflicting;
    conflicting.add({.long_name = "--output", .aliases = {"--kv-cache-precision"},
                     .value_label = "<value>"});
    char executable[] = "llaminar2";
    char *argv[]{executable};
    EXPECT_THROW(OrchestrationConfigParser{}.parseCommandArgs(1, argv, conflicting), std::invalid_argument);
    EXPECT_THROW(OrchestrationConfigParser::getCommandHelpText(conflicting, "plan"), std::invalid_argument);
}

TEST(OrchestrationPlanConfig, SavedExplicitPlacementRejectsAutomaticSearchInsteadOfIgnoringIt)
{
    const PlanFile input("device: rocm:0\nmodel_path: model.gguf\n");
    const auto applied = parse({"--config", input.path});
    EXPECT_TRUE(std::holds_alternative<ApplyOrchestrationRequest>(resolveOrchestrationIntent(applied)));
    const auto automatic = parse({"--auto", "--config", input.path});
    EXPECT_THROW((void)resolveOrchestrationIntent(automatic), std::invalid_argument);
}

TEST(OrchestrationPlanConfig, CompactTiersResolveRolesByPriorityNotDeclarationOrder)
{
    auto config = parse({"--expert-tier", "storage=cpu:0,cpu:1;priority=70",
                         "--expert-tier", "compute=rocm:0,rocm:1;priority=-5"});
    ASSERT_TRUE(config.moe_routed_expert_plan);
    const auto &plan = *config.moe_routed_expert_plan;
    EXPECT_TRUE(plan.enabled);
    EXPECT_EQ(plan.continuation_domain, "compute");
    EXPECT_EQ(plan.shared_expert_domain, "compute");
    EXPECT_EQ(plan.effectiveBaseModelDomain(), "compute");
    EXPECT_EQ(plan.residency_policy,
              RoutedExpertResidencyPolicy::RoutedTierRebalanced);
    EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
    ASSERT_EQ(plan.routed_tiers.size(), 2u);
    EXPECT_TRUE(plan.routed_tiers[0].fallback);
    EXPECT_FALSE(plan.routed_tiers[1].fallback);
    for (const auto &tier : plan.routed_tiers)
    {
        EXPECT_EQ(tier.memory_budget_bytes, 0u);
        EXPECT_EQ(tier.max_experts_per_layer, 0);
    }
    for (const auto &domain : plan.domains)
    {
        EXPECT_EQ(domain.scope, ExecutionDomainScope::AUTO);
        EXPECT_EQ(domain.backend, CollectiveBackendType::AUTO);
        EXPECT_EQ(domain.owner_rank, -1);
        EXPECT_TRUE(domain.world_ranks.empty());
        EXPECT_EQ(domain.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
    }
    // Normalization is idempotent: replaying a config does not append domains
    // or change which single authority owns the declaration.
    EXPECT_TRUE(normalizeMoERoutedExpertPlacementDomains(config).empty());
    EXPECT_EQ(config.moe_routed_expert_plan->domains.size(), 2u);
    EXPECT_EQ(config.domain_definitions.size(), 2u);
}

TEST(OrchestrationPlanConfig, ExplicitRoleAndCapacityOverridesSurviveCompactDefaults)
{
    const auto config = parse({
        "--expert-tier", "primary=cuda:0;priority=0;memory-mb=4096;max-experts-per-layer=9",
        "--expert-tier", "secondary=cuda:1;priority=8;routed_compute=replicated",
        "--moe-routed-expert-continuation-domain", "secondary",
        "--moe-routed-expert-shared-domain", "primary"});
    const auto &plan = *config.moe_routed_expert_plan;
    EXPECT_EQ(plan.continuation_domain, "secondary");
    EXPECT_EQ(plan.shared_expert_domain, "primary");
    EXPECT_EQ(plan.routed_tiers[0].memory_budget_bytes, 4096ull * 1024 * 1024);
    EXPECT_EQ(plan.routed_tiers[0].max_experts_per_layer, 9);
    EXPECT_EQ(plan.domains[1].routed_compute_policy, RoutedExpertComputePolicy::Replicated);
}

TEST(OrchestrationPlanConfig, CompactDomainsHaveBackendSymmetricDefaults)
{
    for (const auto *devices : {"cpu:0", "cuda:0", "rocm:0", "cuda:0,cuda:1", "rocm:0,rocm:1"})
    {
        SCOPED_TRACE(devices);
        const auto declaration = ExpertTierDefinition::parse(std::string("t=") + devices + ";priority=0");
        EXPECT_EQ(declaration.domain.scope, ExecutionDomainScope::AUTO);
        EXPECT_EQ(declaration.domain.routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
        EXPECT_EQ(declaration.tier.memory_budget_bytes, 0u);
    }
}

TEST(OrchestrationPlanConfig, DensePolicyExplicitnessSurvivesArgumentOrdering)
{
    for (const bool override_first : {true, false})
    {
        const std::vector<std::string> tier = {"--expert-tier", "t=cuda:0,cuda:1;priority=0"};
        const std::vector<std::string> policy = {"--moe-continuation-dense-policy", "replicated"};
        auto arguments = override_first ? policy : tier;
        const auto &tail = override_first ? tier : policy;
        arguments.insert(arguments.end(), tail.begin(), tail.end());
        const auto config = parse(arguments);
        EXPECT_EQ(config.moe_routed_expert_plan->continuation_domain_spec.effectiveDensePolicy(),
                  DenseParallelPolicy::Replicated);
        EXPECT_EQ(config.moe_routed_expert_plan->continuation_dense_policy_intent,
                  MoEContinuationDensePolicyIntent::Explicit);
    }
    const auto config = parse({"--expert-tier", "s=rocm:0;priority=0"});
    EXPECT_EQ(config.moe_routed_expert_plan->continuation_domain_spec.effectiveDensePolicy(),
              DenseParallelPolicy::Replicated);
}

TEST(OrchestrationPlanConfig, ExpandedDomainsAcceptTheSameOmittedDefaults)
{
    const auto config = parse({"--moe-routed-expert-placement", "tiered-overlay",
                              "--moe-routed-expert-domain", "accelerator=cuda:0",
                              "--moe-routed-expert-tier", "a@accelerator;priority=3"});
    EXPECT_EQ(config.moe_routed_expert_plan->continuation_domain, "accelerator");
    EXPECT_EQ(config.moe_routed_expert_plan->residency_policy,
              RoutedExpertResidencyPolicy::RoutedTierRebalanced);
}

TEST(OrchestrationPlanConfig, ExpertOverlayResidencyDefaultTracksMaintenanceIntent)
{
    const auto dynamic = parse({
        "--expert-tier", "compute=cuda:0,cuda:1;priority=0"});
    ASSERT_TRUE(dynamic.moe_routed_expert_plan);
    EXPECT_EQ(dynamic.moe_rebalance.mode,
              MoERebalanceRuntimeMode::Dynamic);
    EXPECT_EQ(dynamic.moe_routed_expert_plan->residency_policy,
              RoutedExpertResidencyPolicy::RoutedTierRebalanced);

    const auto disabled = parse({
        "--expert-tier", "compute=cuda:0,cuda:1;priority=0",
        "--moe-residency-maintenance", "off"});
    ASSERT_TRUE(disabled.moe_routed_expert_plan);
    EXPECT_EQ(disabled.moe_routed_expert_plan->residency_policy,
              RoutedExpertResidencyPolicy::StaticById);

    const auto explicit_static = parse({
        "--expert-tier", "compute=cuda:0,cuda:1;priority=0",
        "--moe-routed-expert-residency", "static-by-id"});
    ASSERT_TRUE(explicit_static.moe_routed_expert_plan);
    EXPECT_EQ(explicit_static.moe_routed_expert_plan->residency_policy,
              RoutedExpertResidencyPolicy::StaticById);
}

TEST(OrchestrationPlanConfig, CompactDeclarationsRejectAmbiguityAndOverflow)
{
    for (const auto *declaration : {
             "x=cuda:0", "x=cuda:0;priority=2junk", "x=cuda:0;priority=0;priority=1",
             "x=cuda:0;priority=0;memory-mb=-1", "x=cuda:0;priority=0;memory-mb=18446744073709551615",
             "x=cuda:0;priority=0;max-experts-per-layer=-1", "x=cuda:0;priority=0;fallback=true",
             "x=cuda:0;priority=0;scope=global", "x=cuda:0;priority=0;backend=nccl;backend=rccl"})
    {
        SCOPED_TRACE(declaration);
        EXPECT_THROW(ExpertTierDefinition::parse(declaration), std::invalid_argument);
    }
    EXPECT_THROW(parse({"--expert-tier", "x=cuda:0;priority=0", "--expert-tier", "y=rocm:0;priority=0"}), std::invalid_argument);
}

TEST(OrchestrationPlanConfig, YamlCarriesModelWorkloadClusterAndCompactTiers)
{
    const auto config = OrchestrationConfigParser{}.parseYamlString(R"(
model_path: models/model.gguf
max_seq_len: 8192
batch_size: 1
hostfile: cluster.hosts
mpi_procs: 4
expert_tiers:
  - "compute=rocm:0,rocm:1;priority=0"
  - "storage=cpu:0,cpu:1;priority=1"
kv_cache_precision: fp16
)");
    EXPECT_EQ(config.model_path, "models/model.gguf");
    EXPECT_EQ(config.max_seq_len, 8192);
    EXPECT_EQ(config.batch_size, 1);
    EXPECT_EQ(config.hostfile, "cluster.hosts");
    EXPECT_EQ(config.mpi_procs, 4);
    ASSERT_TRUE(config.moe_routed_expert_plan);
    EXPECT_EQ(config.moe_routed_expert_plan->routed_tiers.size(), 2u);
    EXPECT_EQ(config.moe_routed_expert_plan->continuation_domain, "compute");
}

TEST(OrchestrationPlanConfig, ScalarAfterNestedMtpSectionReturnsToRoot)
{
    const auto config = OrchestrationConfigParser{}.parseYamlString(
        "mtp:\n  enabled: true\nmodel_path: example.gguf\nmax_seq_len: 4096\n");
    EXPECT_EQ(config.model_path, "example.gguf");
    EXPECT_EQ(config.max_seq_len, 4096);
    EXPECT_TRUE(config.mtp.enabled);
}

TEST(OrchestrationPlanConfig, ConfigEqualsAsLastArgumentIsActuallyLoaded)
{
    const PlanFile file("model_path: from-plan.gguf\nmax_seq_len: 8192\n");
    EXPECT_EQ(parse({"--config=" + file.path}).model_path, "from-plan.gguf");
    const auto overridden = parse({"--config=" + file.path, "--model", "override.gguf"});
    EXPECT_EQ(overridden.model_path, "override.gguf");
    EXPECT_EQ(overridden.max_seq_len, 8192);
}

TEST(OrchestrationPlanConfig, HostfileAliasPreservesClusterIntent)
{
    EXPECT_EQ(parse({"--hostfile", "cluster.hosts"}).hostfile, "cluster.hosts");
    EXPECT_EQ(parse({"--mpi-hostfile=cluster.hosts"}).hostfile, "cluster.hosts");
}

TEST(OrchestrationPlanConfig, HostfileLaunchDelegatesSlotsAndRemoteGeometryToMPI)
{
    OrchestrationConfig request;
    request.hostfile = "cluster.hosts";
    const auto launch = MPIBootstrapPhase::discoveryLaunchConfig(request);
    EXPECT_EQ(launch.num_procs, 0);
    EXPECT_TRUE(launch.cpu_set.empty());
    EXPECT_EQ(launch.omp_threads_per_rank, 0);
    CPUTopology topology;
    topology.num_sockets = 2;
    topology.cores_per_socket = 28;
    char binary[] = "llaminar2";
    char subcommand[] = "serve";
    char flag[] = "--hostfile";
    char file[] = "cluster.hosts";
    char *argv[] = {binary, subcommand, flag, file};
    const auto command = MPIBootstrap::buildMPIRunCommand(4, argv, launch, topology);
    EXPECT_EQ(std::count(command.begin(), command.end(), "--hostfile"), 2);
    EXPECT_EQ(std::count(command.begin(), command.end(), "-np"), 0);
    EXPECT_EQ(std::count(command.begin(), command.end(), "--cpu-set"), 0);
    EXPECT_FALSE(std::any_of(command.begin(), command.end(),
        [](const auto &argument) { return argument.find(":PE=") != std::string::npos; }));
    request.mpi_procs = 7;
    EXPECT_EQ(MPIBootstrapPhase::discoveryLaunchConfig(request).num_procs, 7);
}

/** @brief Auto discovery must not pin every requested rank to NUMA zero. */
TEST(OrchestrationPlanConfig, AutomaticLocalMpiWorldRetainsSocketDiscovery)
{
    OrchestrationConfig request;
    request.planning_mode = OrchestrationPlanningMode::Automatic;
    request.mpi_procs = 2;
    const auto launch = MPIBootstrapPhase::discoveryLaunchConfig(request);
    EXPECT_EQ(launch.num_procs, 2);
    EXPECT_TRUE(launch.cpu_set.empty());
    EXPECT_TRUE(launch.bind_to_socket);
    EXPECT_TRUE(launch.map_by_socket);
    EXPECT_EQ(launch.omp_threads_per_rank, 0);
    request.planning_mode = OrchestrationPlanningMode::Apply;
    EXPECT_THROW(MPIBootstrapPhase::discoveryLaunchConfig(request),
                 std::invalid_argument);
}

TEST(OrchestrationPlanConfig, SavedLocalSelectionCannotNarrowTheDiscoveryLaunchToItsCpuPin)
{
    OrchestrationConfig request;
    request.device_for_this_rank = GlobalDeviceAddress::cpu(7);
    request.device_for_this_rank_numa_explicit = true;
    request.execution_rank_selection = ExecutionRankSelection({3, 1});
    EXPECT_THROW(MPIBootstrapPhase::discoveryLaunchConfig(request), std::invalid_argument);
    request.mpi_procs = 4;
    const auto launch = MPIBootstrapPhase::discoveryLaunchConfig(request);
    EXPECT_EQ(launch.num_procs, 4);
    EXPECT_TRUE(launch.cpu_set.empty());
    EXPECT_EQ(launch.omp_threads_per_rank, 0);
    EXPECT_TRUE(launch.bind_to_socket);
    EXPECT_TRUE(launch.map_by_socket);
    request.mpi_procs = 2;
    EXPECT_THROW(MPIBootstrapPhase::discoveryLaunchConfig(request), std::invalid_argument);
    request.mpi_procs = 0;
    request.hostfile = "cluster.hosts";
    EXPECT_EQ(MPIBootstrapPhase::discoveryLaunchConfig(request).num_procs, 0);
}

TEST(OrchestrationPlanConfig, SavedSelectionCannotConstructAnImplicitWorldRunner)
{
    OrchestrationConfig request;
    request.device_for_this_rank = GlobalDeviceAddress::cpu(0);
    request.execution_rank_selection = ExecutionRankSelection({1});
    auto factory = createOrchestrationRunnerFactory();
    // This rejection precedes every context lookup, model read or device query.
    // The positive explicit-context path is exercised by frontend MPI preflight.
    EXPECT_EQ(factory->createFromOrchestrationConfig(request), nullptr);
}
