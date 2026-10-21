/**
 * @file Test__ModelParityDiagnosticBenchmark.cpp
 * @brief Device-free declared-placement benchmark projection of actual QFN cells.
 *
 * Real matrix objects and the public parser prove static8 tiers, storage, CPU
 * endpoint and the original hybrid stage split. No model, device, fabricated
 * E2E tag, resolved allocation or alternate topology is created. Existing
 * certified automatic arguments and server arguments remain separate surfaces.
 */
#include "integration/parity/ModelParityDefinition.h"
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveParityCases.h"
#include "integration/parity/qwen38/Qwen38ModelParityDefinitions.h"
#include "config/OrchestrationConfig.h"
#include "config/OrchestrationConfigParser.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <sstream>

namespace llaminar2::test::parity
{
namespace
{
/** @return Actual public CLI parse without startup or model reads. */
OrchestrationConfig parseDiagnostic(std::vector<std::string> args)
{
    args.insert(args.begin(), "llaminar2");
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
}

/**
 * @brief Compare the original declared placement after public normalization.
 * @param cell Actual canonical case; its expert plan is copied by config creation.
 * @note Normalization reconciles typed declarations only. No device inventory,
 *       placement resolution, model reads or execution participates in this test.
 */
void requireDeclaredRoundTrip(const ModelParityCase &cell)
{
    auto expected = cell.makeOrchestrationConfig(cell.model.model_path, 0);
    // The typed cell stores overlay domains in its owned expert plan. The
    // public parser reconciles those dense/routed declarations into the shared
    // domain inventory, so compare the original plan after that same semantic
    // normalization rather than comparing its unprojected vector length.
    const auto normalization_errors = normalizeMoERoutedExpertPlacementDomains(expected);
    ASSERT_TRUE(normalization_errors.empty());
    const auto args = modelParityDiagnosticBenchmarkArguments(cell);
    const auto actual = parseDiagnostic(args);
    EXPECT_EQ(actual.activation_precision, expected.activation_precision);
    EXPECT_EQ(actual.kv_cache_precision, expected.kv_cache_precision);
    EXPECT_EQ(actual.ngram_table_policy, expected.ngram_table_policy);
    EXPECT_EQ(actual.default_backend, expected.default_backend);
    EXPECT_EQ(actual.mtp.enabled, cell.mtpEnabled());
    EXPECT_EQ(actual.tp_degree, expected.tp_degree);
    EXPECT_EQ(actual.tp_scope, expected.tp_scope);
    EXPECT_EQ(actual.tp_devices, expected.tp_devices);
    EXPECT_EQ(actual.device_map, expected.device_map);
    EXPECT_EQ(actual.device_for_this_rank, expected.device_for_this_rank);
    ASSERT_EQ(actual.domain_definitions.size(), expected.domain_definitions.size());
    for (std::size_t i = 0; i < actual.domain_definitions.size(); ++i)
        EXPECT_EQ(modelParityDomainArgument(actual.domain_definitions[i].toExecutionDomainDefinition()),
                  modelParityDomainArgument(expected.domain_definitions[i].toExecutionDomainDefinition()));
    ASSERT_EQ(actual.pp_stage_definitions.size(), expected.pp_stage_definitions.size());
    for (std::size_t i = 0; i < actual.pp_stage_definitions.size(); ++i)
    {
        const auto &a = actual.pp_stage_definitions[i], &e = expected.pp_stage_definitions[i];
        EXPECT_EQ(a.stage_id, e.stage_id);
        EXPECT_EQ(a.domain_name, e.domain_name);
        EXPECT_EQ(a.first_layer, e.first_layer);
        EXPECT_EQ(a.last_layer, e.last_layer);
    }
    ASSERT_EQ(bool(actual.moe_routed_expert_plan), bool(expected.moe_routed_expert_plan));
    if (expected.moe_routed_expert_plan)
    {
        const auto &a = *actual.moe_routed_expert_plan, &e = *expected.moe_routed_expert_plan;
        EXPECT_EQ(a.topology, e.topology);
        EXPECT_EQ(a.residency_policy, e.residency_policy);
        EXPECT_EQ(a.owner_order, e.owner_order);
        EXPECT_EQ(a.continuation_domain, e.continuation_domain);
        EXPECT_EQ(a.base_model_domain, e.base_model_domain);
        EXPECT_EQ(a.shared_expert_domain, e.shared_expert_domain);
        EXPECT_EQ(a.continuation_domain_spec.effectiveDensePolicy(), e.continuation_domain_spec.effectiveDensePolicy());
        ASSERT_EQ(a.dense_domains.size(), e.dense_domains.size());
        for (std::size_t i = 0; i < a.dense_domains.size(); ++i)
            EXPECT_EQ(modelParityDomainArgument(a.dense_domains[i]),
                      modelParityDomainArgument(e.dense_domains[i]));
        ASSERT_EQ(a.domains.size(), e.domains.size());
        for (std::size_t i = 0; i < a.domains.size(); ++i)
            EXPECT_EQ(modelParityDomainArgument(a.domains[i].toExecutionDomainDefinition()),
                      modelParityDomainArgument(e.domains[i].toExecutionDomainDefinition()));
        ASSERT_EQ(a.routed_tiers.size(), e.routed_tiers.size());
        for (std::size_t i = 0; i < a.routed_tiers.size(); ++i)
        {
            EXPECT_EQ(a.routed_tiers[i].name, e.routed_tiers[i].name);
            EXPECT_EQ(a.routed_tiers[i].domain, e.routed_tiers[i].domain);
            EXPECT_EQ(a.routed_tiers[i].priority, e.routed_tiers[i].priority);
            EXPECT_EQ(a.routed_tiers[i].max_experts_per_layer, e.routed_tiers[i].max_experts_per_layer);
            EXPECT_EQ(a.routed_tiers[i].memory_budget_bytes, e.routed_tiers[i].memory_budget_bytes);
        }
    }
    for (const char *forbidden : {"--auto", "--only-backends", "--auto-device-counts", "--prefix-cache",
             "--moe-dynamic-max-swaps-per-layer", "--moe-residency-maintenance-window",
             "--moe-overlay-prefill-segment-rows", "--mtp-depth-window"})
        EXPECT_EQ(std::count(args.begin(), args.end(), forbidden), 0) << forbidden;
}
}

TEST(ModelParityDiagnosticBenchmark, OriginalCPUAndHybridDeclaredRoundTrip)
{
    int cpu = 0, hybrid = 0;
    for (const auto &cell : qwen4exp::qwen4ExpProductionCases())
    {
        if (!cell.ngram_table_policy ||
            cell.ngram_table_policy->storage != ImmutableRowStoragePreference::MemoryMapped ||
            cell.mtp != ModelParityMTP::Off || cell.expert_overlay) continue;
        ASSERT_FALSE(cell.e2e_certification);
        requireDeclaredRoundTrip(cell);
        if (cell.topology.kind == ModelParityTopologyKind::SingleDevice) ++cpu;
        else
        {
            ++hybrid;
            const auto config = parseDiagnostic(modelParityDiagnosticBenchmarkArguments(cell));
            ASSERT_EQ(config.pp_stage_definitions.size(), 3u);
            EXPECT_EQ(config.pp_stage_definitions[0].first_layer, 0);
            EXPECT_EQ(config.pp_stage_definitions[0].last_layer, 7);
            EXPECT_EQ(config.pp_stage_definitions[1].first_layer, 8);
            EXPECT_EQ(config.pp_stage_definitions[1].last_layer, 19);
            EXPECT_EQ(config.pp_stage_definitions[2].first_layer, 20);
            EXPECT_EQ(config.pp_stage_definitions[2].last_layer, 47);
        }
    }
    EXPECT_EQ(cpu, 1);
    EXPECT_EQ(hybrid, 1);
}

TEST(ModelParityDiagnosticBenchmark, ActualCUDAAndROCmStaticEightAdaptiveRoundTrip)
{
    int cells = 0;
    for (const auto &cell : qwen4exp::qwen4ExpGPUAdaptiveProductionCases())
    {
        ASSERT_FALSE(cell.e2e_certification);
        requireDeclaredRoundTrip(cell);
        const auto config = parseDiagnostic(modelParityDiagnosticBenchmarkArguments(cell));
        ASSERT_NE(config.moe_routed_expert_plan, nullptr);
        ASSERT_EQ(config.moe_routed_expert_plan->routed_tiers.size(), 2u);
        EXPECT_EQ(config.moe_routed_expert_plan->routed_tiers[0].max_experts_per_layer, 8);
        EXPECT_EQ(config.moe_routed_expert_plan->routed_tiers[1].max_experts_per_layer, 0);
        EXPECT_EQ(config.ngram_table_policy.storage, ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD);
        ++cells;
    }
    EXPECT_EQ(cells, 4);
}

TEST(ModelParityDiagnosticBenchmark, DiscoveryRemainsExplicitlyUncertified)
{
    for (const auto &cell : qwen4exp::qwen4ExpGPUAdaptiveProductionCases())
    {
        std::ostringstream text;
        PrintTo(cell, &text);
        const auto record = nlohmann::json::parse(text.str());
        EXPECT_TRUE(record.at("e2e").is_null());
        EXPECT_TRUE(record.at("benchmark").is_null());
        const auto &diagnostic = record.at("diagnostic_benchmark");
        EXPECT_EQ(diagnostic.at("policy"), "diagnostic_declared");
        EXPECT_EQ(diagnostic.at("certification_eligible"), false);
        EXPECT_EQ(diagnostic.at("context_length"), cell.model.max_seq_len);
        EXPECT_EQ(diagnostic.at("args"), modelParityDiagnosticBenchmarkArguments(cell));
        EXPECT_EQ(record.at("runtime").at("server_args"), modelParityServerArguments(cell));
        EXPECT_THROW(modelParityBenchmarkArguments(cell), std::invalid_argument);
    }
}

TEST(ModelParityDiagnosticBenchmark, OriginalCertifiedAutomaticPolicyStaysSeparate)
{
    int tags = 0;
    for (const auto &definition : qwen38::qwen38DenseMultiDeviceDefinitions())
        for (const auto &cell : expandModelParityDefinition(definition))
        {
            if (!cell.e2e_certification) continue;
            ++tags;
            const auto automatic = modelParityBenchmarkArguments(cell);
            EXPECT_EQ(automatic, modelParityBenchmarkSemanticArguments(cell,
                model_parity_runtime_detail::BenchmarkPlacement::AutomaticProduction));
            EXPECT_EQ(std::count(automatic.begin(), automatic.end(), "--auto"), 1);
            EXPECT_EQ(std::count(automatic.begin(), automatic.end(), "--pp-stage"), 0);
            const auto diagnostic = modelParityDiagnosticBenchmarkArguments(cell);
            EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), "--auto"), 0);
            EXPECT_EQ(std::count(diagnostic.begin(), diagnostic.end(), "--mpi-procs"), 1);
            EXPECT_THROW(modelParityBenchmarkSemanticArguments(cell,
                static_cast<model_parity_runtime_detail::BenchmarkPlacement>(255)), std::invalid_argument);
        }
    EXPECT_GT(tags, 0);
}
} // namespace llaminar2::test::parity
