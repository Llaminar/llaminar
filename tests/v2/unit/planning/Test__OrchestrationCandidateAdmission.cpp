/**
 * @file Test__OrchestrationCandidateAdmission.cpp
 * @brief Device-free GGUF-to-complete-candidate admission regressions.
 *
 * These tests compose the real metadata parser, rank compiler and canonical BOM
 * admission without allocating model tensors or touching accelerators. Sparse
 * fixture payloads are not numerical model or graph-execution certification.
 */
#include "planning/OrchestrationCandidateAdmission.h"
#include "planning/RankMemoryPlanInputs.h"
#include "config/OrchestrationConfigDocument.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>

using namespace llaminar2;

namespace
{
    /** @return Separate physical hosts with sparse CPU and GPU ordinal identities. */
    ClusterInventory inventory(DeviceType backend, int remote_hosts = 0)
    {
        ClusterInventory result;
        result.world_size = remote_hosts + 1;
        for (int rank = 0; rank < result.world_size; ++rank)
        {
            RankInventory record;
            record.rank = rank;
            record.node_id = rank;
            record.local_rank = 0;
            record.hostname = "host-" + std::to_string(rank);
            record.cpu.numa_node = 3;
            record.cpu.memory_bytes = 64ull << 30;
            record.cpu.free_memory_bytes = 48ull << 30;
            record.cpu.compute_units = record.cpu_cores = 8;
            record.cpu_worker_threads = 8;
            record.numa_nodes = 1;
            if (backend != DeviceType::CPU && rank == 0)
                record.gpus.push_back({.type = backend, .local_device_id = 2,
                    .memory_bytes = 32ull << 30, .free_memory_bytes = 24ull << 30,
                    .compute_units = 64, .uuid = "observed-card", .numa_node = 3});
            result.ranks.push_back(std::move(record));
        }
        result.buildNodeAggregations();
        return result;
    }

    /** @return One complete explicit candidate from the shared automatic expander. */
    AutomaticOrchestrationCandidate candidate(const PlanningModelSource &source,
        DeviceType backend, int remote_hosts = 0)
    {
        OrchestrationConfig request;
        request.max_seq_len = 1024;
        request.mtp.enabled = source.metadata().memoryProfile().mtp_layer_count > 0;
        request.mtp.graph_capacity_draft_tokens = request.mtp.enabled ? 15 : 0;
        request.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        request.mtp.depth_policy.max_depth = 15;
        request.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        request.model_path = source.path();
        request.automatic_planning.only_backends = remote_hosts ? std::vector{backend, DeviceType::CPU} : std::vector{backend};
        request.automatic_planning.only_strategies = {remote_hosts ? OrchestrationStrategy::ExpertOverlay : OrchestrationStrategy::SingleDevice};
        std::optional<AutomaticOrchestrationCandidate> selected;
        visitAutomaticOrchestrationCandidates(request, source.metadata(), inventory(backend, remote_hosts),
            [&](auto value) {
                if (value.membership.discoveryRanks().size() == static_cast<size_t>(remote_hosts + 1))
                    selected = std::move(value);
            });
        if (!selected) throw std::logic_error("Fixture expander emitted no complete membership");
        return std::move(*selected);
    }

    /** @return Small explicit capture ladder, with unchanged production upload policy. */
    OrchestrationCandidateMemoryPolicy policy()
    {
        return {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1,
            .maximum_cached_prefill_buckets = 8};
    }
}

TEST(OrchestrationCandidateAdmission, OrdinaryBOMEqualsTheRuntimeBuilderOnEveryBackend)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        auto proposal = candidate(source, backend);
        proposal.config.model_path = file.path();
        proposal.config.max_cpu_memory_mb = 32768;
        proposal.config.max_gpu_memory_mb = 16384;
        const auto admitted = AdmittedOrchestrationCandidate::admit(std::move(proposal), source, policy());
        ASSERT_EQ(admitted.rankPlans().size(), 1u);
        ASSERT_EQ(admitted.devicePlans().size(), 1u);
        EXPECT_EQ(admitted.devicePlans()[0].world_rank, 0);
        ASSERT_TRUE(admitted.config().execution_rank_selection);
        EXPECT_EQ(*admitted.config().execution_rank_selection, admitted.membership().selection());
        EXPECT_FALSE(admitted.overlayCapacity());
        const auto configs = buildRankMemoryPlanInputs({
            .model = source.metadata().memoryProfile(), .plan = admitted.rankPlans()[0],
            .inventory = admitted.membership().inventory().ranks[0],
            .weight_load_geometry = resolveGPUWeightLoadMemoryGeometry(maximumGGUFTensorPayloadBytes(source.loader().getModel()), policy().weight_load),
            .captured_prefill_buckets = policy().prefill_bucket_rows,
            .max_gpu_memory_bytes = 16ull << 30, .max_cpu_memory_bytes = 32ull << 30});
        const auto expected = MemoryPlanner::planLargestFittingResidentGraphRows(
            source.metadata().memoryProfile(), configs, policy().prefill_bucket_rows);
        EXPECT_EQ(expected.resident_graph_rows, admitted.rankPlans()[0].runtime.resident_graph_rows);
        ASSERT_EQ(expected.device_inputs.size(), 1u);
        EXPECT_EQ(admitted.devicePlans()[0].activation_seq_len, expected.device_inputs[0].activation_seq_len);
        EXPECT_EQ(admitted.devicePlans()[0].device, expected.device_inputs[0].device);
        EXPECT_EQ(expected.memory_plan.admit().plan().totalBytes(), admitted.physicalAdmission().plan().totalBytes());
        EXPECT_FALSE(expected.memory_plan.admit().plan().requiredFootprintMismatch(admitted.physicalAdmission().plan()));
        EXPECT_EQ(serializeOrchestrationConfig(deserializeOrchestrationConfig(serializeOrchestrationConfig(admitted.config()))),
            serializeOrchestrationConfig(admitted.config()));
    }
}

TEST(OrchestrationCandidateAdmission, ZeroUserCapacityFailsTypedWithoutChangingTopology)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        auto proposal = candidate(source, backend);
        proposal.config.model_path = file.path();
        proposal.config.max_cpu_memory_mb = proposal.config.max_gpu_memory_mb = 0;
        EXPECT_THROW((void)AdmittedOrchestrationCandidate::admit(proposal, source, policy()), PhysicalMemoryCapacityExhausted);
        EXPECT_EQ(proposal.strategy, OrchestrationStrategy::SingleDevice);
    }
}

TEST(OrchestrationCandidateAdmission, InvalidInputIsNotCapacityRejection)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto proposal = candidate(source, DeviceType::CPU);
    for (const int defect : {0, 1, 2})
    {
        auto malformed = proposal;
        auto settings = policy();
        if (defect == 0) malformed.config.model_path += ".different";
        if (defect == 1) settings.prefill_bucket_rows.clear();
        if (defect == 2) settings.prefill_bucket_rows.push_back(0);
        try
        {
            (void)AdmittedOrchestrationCandidate::admit(std::move(malformed), source, settings);
            FAIL() << "Malformed candidate was admitted";
        }
        catch (const PhysicalMemoryCapacityExhausted &)
        {
            FAIL() << "Invalid input must not advance a capacity search";
        }
        catch (const std::invalid_argument &) {}
    }
}

TEST(OrchestrationCandidateAdmission, RemoteOverlayAdmitsEveryRankAndRetainsDepthFifteen)
{
    for (const auto format : {GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16})
    {
        test::PlanningGGUFFixture file(true, true, format);
        PlanningModelSource source(file.path());
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const int hosts : {1, 2})
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Dynamic})
        {
            SCOPED_TRACE(::testing::Message() << deviceTypeToString(backend) << " hosts=" << hosts
                << " format=" << static_cast<int>(format) << " mode=" << static_cast<int>(mode));
            auto proposal = candidate(source, backend, hosts);
            proposal.config.model_path = file.path();
            proposal.config.moe_rebalance.mode = mode;
            proposal.config.moe_routed_expert_plan->residency_policy = mode == MoERebalanceRuntimeMode::Off
                ? RoutedExpertResidencyPolicy::StaticById : RoutedExpertResidencyPolicy::RoutedTierRebalanced;
            const auto admitted = AdmittedOrchestrationCandidate::admit(std::move(proposal), source, policy());
            ASSERT_TRUE(admitted.overlayCapacity());
            EXPECT_EQ(admitted.overlayCapacity()->physical_memory_admission.get(), &admitted.physicalAdmission());
            EXPECT_EQ(admitted.rankPlans().size(), static_cast<size_t>(hosts + 1));
            EXPECT_TRUE(admitted.physicalAdmission().plan().fits());
            ASSERT_EQ(admitted.devicePlans().size(), static_cast<size_t>(hosts + 1));
            for (const auto &device : admitted.devicePlans())
            {
                EXPECT_TRUE(device.weight_residency.selectsRoutedExperts());
                EXPECT_EQ(device.weight_residency.selectedRoutedExpertsForLayer(0), 0)
                    << "Fixed BOM inputs must not masquerade as final routed quotas";
                EXPECT_GT(device.activation_seq_len, 0);
            }
            for (const auto &rank : admitted.rankPlans())
            {
                EXPECT_EQ(resolveMTPRetainedDraftCapacity(rank.runtime.mtp), 15);
                EXPECT_GE(rank.runtime.resident_graph_rows, resolveMTPRetainedTargetQueryRows(rank.runtime.mtp));
                EXPECT_NE(admitted.physicalAdmission().plan().find({rank.rank, DeviceId::cpu()}), nullptr);
            }
        }
    }
}

/**
 * @brief Discovery-time capacity evidence must not become a fixed live quota.
 *
 * GPU context creation happens after automatic search. A changed free-memory
 * observation can alter the maximal resident expert count without changing
 * the selected topology or automatic capacity policy.
 */
TEST(OrchestrationCandidateAdmission, AutomaticOverlayKeepsLiveExpertQuotaAdaptive)
{
    test::PlanningGGUFFixture file(true, true, GGUFTensorType::F16);
    PlanningModelSource source(file.path());
    auto proposal = candidate(source, DeviceType::ROCm, 1);
    proposal.config.model_path = file.path();
    const auto admitted = AdmittedOrchestrationCandidate::admit(
        proposal, source, policy());
    ASSERT_TRUE(admitted.overlayCapacity());
    ASSERT_TRUE(admitted.config().moe_routed_expert_plan);
    ASSERT_EQ(admitted.config().moe_routed_expert_plan->routed_tiers.size(), 2u);
    for (const auto &tier : admitted.config().moe_routed_expert_plan->routed_tiers)
        EXPECT_TRUE(tier.resolved_live_experts_per_layer.empty());

    auto lower_free_inventory = proposal.membership.inventory();
    lower_free_inventory.ranks[0].gpus[0].free_memory_bytes -= 1ull << 20;
    lower_free_inventory.buildNodeAggregations();
    proposal.membership = ExecutionRankMembership(
        lower_free_inventory, proposal.membership.discoveryRanks());
    const auto refreshed = AdmittedOrchestrationCandidate::admit(
        std::move(proposal), source, policy());
    ASSERT_TRUE(refreshed.overlayCapacity());
    for (const auto &tier : refreshed.config().moe_routed_expert_plan->routed_tiers)
        EXPECT_TRUE(tier.resolved_live_experts_per_layer.empty());
    const auto restored = deserializeOrchestrationConfig(
        serializeOrchestrationConfig(refreshed.config()));
    for (const auto &tier : restored.moe_routed_expert_plan->routed_tiers)
        EXPECT_TRUE(tier.resolved_live_experts_per_layer.empty());
}

TEST(OrchestrationCandidateAdmission, PipelineAdmissionIncludesTheNonRootPhysicalBOM)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    OrchestrationConfig request;
    request.model_path = file.path();
    request.max_seq_len = 1024;
    request.automatic_planning.only_backends = {DeviceType::CPU};
    request.automatic_planning.only_strategies = {OrchestrationStrategy::PipelineParallel};
    int seen = 0;
    visitAutomaticOrchestrationCandidates(request, source.metadata(), inventory(DeviceType::CPU, 1),
        [&](auto value) {
            ++seen;
            const auto complete = AdmittedOrchestrationCandidate::admit(value, source, policy());
            EXPECT_EQ(complete.rankPlans().size(), 2u);
            EXPECT_EQ(complete.physicalAdmission().plan().resources().size(), 2u);
            auto exhausted = value.membership.inventory();
            exhausted.ranks[1].cpu.free_memory_bytes = 0;
            value.membership = ExecutionRankMembership(exhausted, {0, 1});
            EXPECT_THROW((void)AdmittedOrchestrationCandidate::admit(std::move(value), source, policy()), PhysicalMemoryCapacityExhausted);
        });
    EXPECT_EQ(seen, 2);
}
