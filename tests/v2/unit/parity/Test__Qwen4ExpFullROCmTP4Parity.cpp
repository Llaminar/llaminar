/**
 * @file Test__Qwen4ExpFullROCmTP4Parity.cpp
 * @brief Device-free original-source, full-GPU topology and public-config regressions.
 *
 * Actual PMA admission and resident expert/source materialization belong to the
 * real SnapshotAdmission cells. These tests exercise the canonical expander,
 * original numerical contract and public parser, without GPU initialization,
 * weight loading, model reads or a second placement/capacity calculation.
 */
#include "integration/parity/qwen4exp/Qwen4ExpFullROCmTP4ParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpFixedHeadParityCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"
#include <nlohmann/json.hpp>
#include "config/GDNProjectionArithmeticPolicy.h"
#include "config/MoERouterArithmeticPolicy.h"
#include <gtest/gtest.h>
#include <set>
#include <sstream>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Existing public argument consumers; each uses the same canonical placement formatter. */
enum class PublicProjection { Server, DiagnosticBenchmark, CorpusObserver };

/**
 * @brief Parse the actual declared public arguments without model/device discovery.
 * @param cell Original complete model/placement declaration.
 * @param projection Existing server, diagnostic benchmark or corpus observation boundary.
 * @return Public parser-owned configuration; corpus explicitly disables prefix caching.
 * @note Corpus history-reset semantics differ from parity's prefix proof. This
 *       changes no placement/precision/source and does not mint expected_config.
 */
OrchestrationConfig publicConfig(const ModelParityCase &cell,
                                 PublicProjection projection = PublicProjection::Server)
{
    auto args = projection == PublicProjection::Server
        ? modelParityServerArguments(cell) : modelParityDiagnosticBenchmarkArguments(cell);
    if (projection == PublicProjection::CorpusObserver)
    {
        args.push_back("--no-prefix-cache");
    }
    args.insert(args.begin(), {"llaminar2", "--model", cell.model.model_path});
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
}

/** @brief Require exactly four original physical ROCm endpoints, in communicator order. */
void expectFourROCm(const std::vector<GlobalDeviceAddress> &participants)
{
    ASSERT_EQ(participants.size(), 4u);
    for (int index = 0; index < 4; ++index)
    {
        EXPECT_EQ(participants[static_cast<std::size_t>(index)], GlobalDeviceAddress::rocm(index));
    }
}
} // namespace

/** @test One new case preserves every inherited source/numerical/path obligation. */
TEST(Qwen4ExpFullROCmTP4Parity, OneCasePreservesOriginalSourceAndPerStepNumericalContract)
{
    const auto &cells = qwen4exp::qwen4ExpFullROCmTP4ProductionCases();
    ASSERT_EQ(cells.size(), 1u);
    EXPECT_EQ(qwen4exp::qwen4ExpProductionCases().size(), 20u);
    EXPECT_EQ(qwen4exp::qwen4ExpFixedHeadProductionCases().size(), 64u);
    const auto source = expandModelParityDefinition(qwen4exp::qwen4ExpProductionDefinition(
        qwen4exp::Qwen4ExpParityTopology::CPU)).front();
    std::set<std::string> names;
    for (const auto &cell : cells)
    {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.model.model_path, source.model.model_path);
        ASSERT_TRUE(cell.model.shared_predictor);
        EXPECT_EQ(cell.model.shared_predictor->path(), qwen4exp::kPredictorModelPath);
        EXPECT_EQ(cell.model.reference_directory, source.model.reference_directory);
        EXPECT_EQ(cell.model.token_ids, source.model.token_ids);
        EXPECT_EQ(cell.model.transformer_layers, 48);
        EXPECT_EQ(cell.model.decode_steps, 2);
        EXPECT_EQ(cell.activation_precision, ActivationPrecision::FP32);
        EXPECT_EQ(cell.kv_cache_precision, KVCachePrecision::FP16);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 0);
        EXPECT_FALSE(cell.e2e_certification);
        ASSERT_TRUE(cell.ngram_table_policy);
        EXPECT_EQ(cell.ngram_table_policy->storage, ImmutableRowStoragePreference::RAM);
        EXPECT_EQ(cell.graph_snapshot_policy.required_prefill_snapshot_keys,
                  source.graph_snapshot_policy.required_prefill_snapshot_keys);
        EXPECT_EQ(cell.graph_snapshot_policy.required_decode_snapshot_keys,
                  source.graph_snapshot_policy.required_decode_snapshot_keys);
        EXPECT_TRUE(cell.graph_snapshot_policy.require_graph_execution_on_gpu);
        EXPECT_TRUE(cell.graph_snapshot_policy.require_snapshot_publication);
        EXPECT_EQ(cell.thresholds.early_layers_count, source.thresholds.early_layers_count);
        EXPECT_EQ(cell.thresholds.min_early_layers_passed, source.thresholds.min_early_layers_passed);
        ASSERT_TRUE(cell.thresholds.final_logit_mean_kl);
        EXPECT_EQ(cell.thresholds.final_logit_mean_kl->prefill_maximum,
                  source.thresholds.final_logit_mean_kl->prefill_maximum);
        EXPECT_EQ(cell.thresholds.final_logit_mean_kl->decode_maximum,
                  source.thresholds.final_logit_mean_kl->decode_maximum);
        EXPECT_EQ(cell.thresholds.final_logit_mean_kl->decode_steps, 2);
        EXPECT_EQ(cell.prefix_restore_geometry, source.prefix_restore_geometry);
        EXPECT_EQ(cell.collective_evidence_source, source.collective_evidence_source);
        EXPECT_EQ(cell.topology.collective, Collective::RCCL);
        EXPECT_EQ(cell.topology.mpi_ranks, 1);
        std::vector<GlobalDeviceAddress> participants;
        for (const auto &participant : cell.topology.participants)
        {
            EXPECT_EQ(participant.world_rank, 0);
            participants.push_back(participant.address);
        }
        expectFourROCm(participants);
        const auto parsed = publicConfig(cell);
        EXPECT_EQ(parsed.model_path, cell.model.model_path);
        EXPECT_EQ(parsed.shared_predictor, cell.model.shared_predictor);
        EXPECT_EQ(parsed.ngram_table_policy, *cell.ngram_table_policy);
        std::ostringstream discovery;
        ASSERT_NO_THROW(PrintTo(cell, &discovery));
        EXPECT_NE(discovery.str().find("\"server_args\":["), std::string::npos);
        EXPECT_NE(discovery.str().find("\"e2e\":null,\"benchmark\":null"), std::string::npos);
        // Bound both original GTest artifact components, including the longer admission body.
        for (const auto *body : {"ProductionParity", "SnapshotAdmission"})
        {
            EXPECT_LT(std::string("Qwen4ExpFullROCmTP4_Qwen4ExpMainParityTest_").size() +
                      std::string(body).size() + 1u + cell.testName().size(), 256u);
        }
    }
    EXPECT_EQ(resolveMoERouterArithmetic(MoERouterArithmetic::ModelDefault, "qwen4exp"),
              MoERouterArithmetic::FP16);
    EXPECT_EQ(kDefaultGDNProjectionArithmetic, GDNProjectionArithmetic::FP16MoEAlphaBeta);
}

/** @test TP4 owns only one complete final GPU tier, with no eight-expert limit or CPU overflow. */
TEST(Qwen4ExpFullROCmTP4Parity, TP4PublicProjectionRetainsSoleFullCoverageGpuAuthority)
{
    const auto &cell = qwen4exp::qwen4ExpFullROCmTP4ProductionCases().front();
    ASSERT_EQ(cell.topology.kind, ModelParityTopologyKind::RankLocalTensorParallel);
    ASSERT_EQ(cell.expert_overlay, std::optional{ModelParityExpertOverlayPolicy{}});
    for (const auto &config : {cell.makeOrchestrationConfig(cell.model.model_path, 0),
                               publicConfig(cell), publicConfig(cell, PublicProjection::DiagnosticBenchmark)})
    {
        ASSERT_TRUE(config.moe_routed_expert_plan);
        const auto &plan = *config.moe_routed_expert_plan;
        ASSERT_EQ(plan.domains.size(), 1u);
        ASSERT_EQ(plan.routed_tiers.size(), 1u);
        ASSERT_EQ(plan.dense_domains.size(), 1u);
        EXPECT_EQ(plan.topology, RoutedExpertPlacementTopology::SingleDomain);
        EXPECT_EQ(plan.residency_policy, RoutedExpertResidencyPolicy::StaticById);
        EXPECT_EQ(plan.owner_order, RoutedExpertOwnerOrder::Ordinal);
        EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
        EXPECT_EQ(plan.continuation_domain_spec.hidden_layout, MoEContinuationActivationLayout::ReplicatedHidden);
        expectFourROCm(plan.domains.front().participants);
        EXPECT_EQ(plan.domains.front().backend, CollectiveBackendType::RCCL);
        EXPECT_EQ(plan.domains.front().scope, ExecutionDomainScope::RANK_LOCAL);
        EXPECT_EQ(plan.domains.front().routed_compute_policy, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
        auto automatic = plan.domains.front().toExecutionDomainDefinition();
        automatic.routed_compute_policy = RoutedExpertComputePolicy::Automatic;
        EXPECT_EQ(automatic.resolveRoutedComputePolicy(1), RoutedExpertComputePolicy::GateUpOwnedDownColumns);
        EXPECT_TRUE(plan.routed_tiers.front().fallback); // Canonical final coverage, not runtime substitution.
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer, 0);
        EXPECT_EQ(plan.routed_tiers.front().memory_budget_bytes, 0u);
        EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(plan, {.layer_count=48, .routed_expert_count=512}).ok());
        EXPECT_TRUE(config.pp_stage_definitions.empty());
    }
}

/**
 * @test Corpus engine configuration and benchmark export share the original placement/source.
 * @note This authenticates declaration/serialization, not live PMA residency or execution.
 */
TEST(Qwen4ExpFullROCmTP4Parity, DiagnosticBenchmarkAndCorpusConfigRetainExactDeclaredPlacement)
{
    const auto &cell = qwen4exp::qwen4ExpFullROCmTP4ProductionCases().front();
    std::ostringstream stream;
    PrintTo(cell, &stream);
    const auto document = nlohmann::json::parse(stream.str());
    const auto args = modelParityDiagnosticBenchmarkArguments(cell);
    EXPECT_EQ(document.at("diagnostic_benchmark").at("args").get<std::vector<std::string>>(), args);
    EXPECT_FALSE(document.at("diagnostic_benchmark").at("certification_eligible").get<bool>());
    EXPECT_EQ(document.at("diagnostic_benchmark").at("context_length"), cell.model.max_seq_len);
    EXPECT_TRUE(document.at("benchmark").is_null());
    EXPECT_TRUE(document.at("e2e").is_null());
    const auto parsed = publicConfig(cell, PublicProjection::CorpusObserver);
    ASSERT_TRUE(parsed.shared_predictor);
    EXPECT_EQ(parsed.shared_predictor->path(), qwen4exp::kPredictorModelPath);
    EXPECT_FALSE(parsed.mtp.enabled);
    EXPECT_EQ(parsed.mtp_activation_policy, MTPActivationPolicy::Disabled);
    EXPECT_FALSE(parsed.prefix_cache.enabled); // Corpus observer owns independent history resets.
    EXPECT_EQ(parsed.ngram_table_policy, *cell.ngram_table_policy);
    EXPECT_EQ(parsed.activation_precision, "fp32");
    EXPECT_EQ(parsed.kv_cache_precision, "fp16");
    // The existing corpus topology.expected_config uses this exact public document;
    // it never synthesizes a second expert plan or a source-identity receipt.
    const auto wire = serializeOrchestrationConfig(parsed);
    EXPECT_EQ(serializeOrchestrationConfig(deserializeOrchestrationConfig(wire)), wire);
    ASSERT_TRUE(parsed.moe_routed_expert_plan);
    EXPECT_EQ(parsed.moe_routed_expert_plan->routed_tiers.size(), 1u);
    EXPECT_EQ(parsed.moe_routed_expert_plan->domains.front().participants,
              cell.topology.expert_overlay_plan->domains.front().participants);
}

/** @test The real resolved-quota validator cannot accept partial512-expert source coverage.
 * @note Synthetic quota metadata tests validation only; it grants no PMA capacity or native residency. */
TEST(Qwen4ExpFullROCmTP4Parity, ResolvedFinalGpuQuotaMustCoverAll512OriginalExperts)
{
    auto plan = *qwen4exp::qwen4ExpFullROCmTP4ProductionCases().front().topology.expert_overlay_plan;
    plan.routed_tiers.front().resolved_live_experts_per_layer.assign(48, 512);
    EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(plan, {.layer_count=48, .routed_expert_count=512}).ok());
    plan.routed_tiers.front().resolved_live_experts_per_layer[23] = 511;
    EXPECT_FALSE(validateMoERoutedExpertPlacementPlan(plan, {.layer_count=48, .routed_expert_count=512}).ok());
}

/** @test The new declaration never relabels the old eight-GPU-expert plus CPU control. */
TEST(Qwen4ExpFullROCmTP4Parity, KeepsCpuOverlayControlDistinct)
{
    const std::array<int,4> ordinals{0,1,2,3};
    const auto old = qwen4exp::qwen4ExpFixedHeadParityTopology(DeviceType::ROCm, ordinals);
    ASSERT_TRUE(old.expert_overlay_plan);
    EXPECT_EQ(old.expert_overlay_plan->routed_tiers.front().max_experts_per_layer, 8);
    EXPECT_EQ(old.expert_overlay_plan->domains.back().participants,
              (std::vector{GlobalDeviceAddress::cpu(0)}));
    EXPECT_NE(old.test_id, qwen4exp::qwen4ExpFullROCmTP4ProductionCases().front().topology.test_id);
}
} // namespace llaminar2::test::parity
