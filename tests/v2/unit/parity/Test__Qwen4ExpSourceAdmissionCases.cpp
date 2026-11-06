/**
 * @file Test__Qwen4ExpSourceAdmissionCases.cpp
 * @brief Device-free coverage of exact QFN source-only topology and storage requests.
 *
 * These tests use the canonical expander, public CLI parser and configuration
 * validator without loading weights, consulting a reference pack or probing
 * accelerators. Physical fit and native materialization remain separate real
 * source-admission obligations; an absent backend or unsupported public policy
 * cannot be relabelled as a positive admission by these declaration tests.
 * Native proof expectations and public runtime projections must agree on an
 * explicitly smaller dynamic MTP ceiling, independent of retained capacity.
 */
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpPooledCurrentDemandCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpSharedContinuationMTPSourceCases.h"
#include "fixtures/models/Qwen4ExpReplicatedPLECohortReplayAccess.h"
#include "planning/ImmutableRowTablePlacementPlanner.h"
#include "execution/qwen4exp/PooledImmutableParticipantExecutionPolicy.h"
#include "integration/parity/qwen4exp/Qwen4ExpCPUJointParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionEvidence.h"
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpNgramAdmissionCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpROCm4OverlayPolicyParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpAllGPUQ6MTPParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCI.h"
#include "execution/moe/MoEOverlayDeviceControllerTopology.h"
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "integration/parity/qwen4exp/Qwen4ExpOrdinaryGenerationWorkload.h"
#include "integration/parity/ParityHardwareRequirements.h"
#include "config/ConfigValidator.h"
#include "app/RuntimeInitPhase.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "config/OrchestrationConfigParser.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <tuple>
#include <unistd.h>

namespace llaminar2::test::parity::qwen4exp
{

namespace
{
/**
 * @brief Parse the actual public server arguments without startup or source I/O.
 * @param cell Exact source-admission input.
 * @return Parsed request retaining all authored placement and table policies.
 */
OrchestrationConfig sourcePublicConfig(const ModelParityCase &cell)
{
    auto args = modelParityServerArguments(cell);
    args.insert(args.begin(), "llaminar2");
    // The canonical exporter deliberately omits the model. The real HTTP
    // owner appends its independently bound source after these policy args;
    // mirror that public boundary rather than accepting an empty model path.
    args.insert(args.end(), {"-m", cell.model.model_path});
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(arg.data());
    return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
}

/**
 * @brief Check a complete, non-overlapping public 48-layer pipeline partition.
 * @param config Native or CLI-projected pipeline request.
 * @param cell Original ordered participants and per-stage collectives.
 */
void expectSourcePipeline(const OrchestrationConfig &config, const ModelParityCase &cell)
{
    const auto &topology = cell.topology;
    ASSERT_FALSE(config.moe_routed_expert_plan);
    ASSERT_EQ(config.domain_definitions.size(), topology.pipeline_stage_sizes.size());
    ASSERT_EQ(config.pp_stage_definitions.size(), topology.pipeline_stage_sizes.size());
    std::size_t participant = 0;
    int next_layer = 0;
    for (std::size_t stage = 0; stage < topology.pipeline_stage_sizes.size(); ++stage)
    {
        const auto &domain = config.domain_definitions[stage];
        const auto &placement = config.pp_stage_definitions[stage];
        const auto size = static_cast<std::size_t>(topology.pipeline_stage_sizes[stage]);
        ASSERT_EQ(domain.devices.size(), size);
        EXPECT_EQ(domain.scope, TPScope::RANK_LOCAL);
        EXPECT_EQ(domain.owner_rank, 0);
        EXPECT_EQ(placement.domain_name, domain.name);
        EXPECT_EQ(placement.stage_id, static_cast<int>(stage));
        EXPECT_EQ(placement.first_layer, next_layer);
        EXPECT_GE(placement.last_layer, placement.first_layer);
        next_layer = placement.last_layer + 1;
        for (std::size_t index = 0; index < size; ++index)
            EXPECT_EQ(domain.devices[index], topology.participants[participant++].address);
        if (size > 1)
        {
            const auto expected = topology.pipeline_tensor_parallel_collectives[stage];
            EXPECT_EQ(domain.backend, expected == Collective::NCCL ? CollectiveBackendType::NCCL :
                expected == Collective::RCCL ? CollectiveBackendType::RCCL : CollectiveBackendType::HOST);
            EXPECT_EQ(domain.weights.size(), size);
        }
    }
    EXPECT_EQ(participant, topology.participants.size());
    EXPECT_EQ(next_layer, 48);
}
} // namespace

/** @test Combined local cache intent survives typed and public projection, including engaged C0. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotCacheProjectionRetainsAllCapsAndPolicy)
{
    for(const auto* seeds:{&qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmissionSeeds(),
                          &qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmissionSeeds()})
        for(const auto& original:*seeds)
            for(const std::uint64_t slots:{0u,64u}) {
                auto cell=original;
                auto& local=cell.ngram_table_policy->gpu_adaptive.local_hot_cache;
                local=AdaptiveGPULocalHotCacheRequest{
                    .maximum_cache_rows=slots,.cache_physical_byte_limit=slots?std::size_t{1}<<20:0,
                    .policy={.aging_accesses=17,.protected_numerator=2,.protected_denominator=3}};
                const auto native=cell.makeOrchestrationConfig(cell.model.model_path,0);
                const auto parsed=sourcePublicConfig(cell);
                EXPECT_EQ(native.ngram_table_policy,*cell.ngram_table_policy);
                EXPECT_EQ(parsed.ngram_table_policy,*cell.ngram_table_policy);
                EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
                EXPECT_NE(cell.testName(),original.testName());
                const auto originalArgs=modelParityNgramStorageArguments(original);
                EXPECT_EQ(std::find(originalArgs.begin(),originalArgs.end(),"--ngram-local-cache-max-rows"),originalArgs.end());
            }
}

/** @test Every local policy scalar/override affects matrix identity; argv export cannot silently lose overrides. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotCacheIdentityDistinguishesEveryPolicyAndOverride)
{
    auto base=qwen4ExpSharedCoordinatorTP2AdaptiveSourceAdmissionSeeds().front();
    const auto unchanged=base.testName();
    std::set<std::string> names{unchanged};
    const AdaptiveGPULocalHotCacheRequest original{};
    for(int field=0;field<10;++field) {
        auto cell=base;
        auto& local=cell.ngram_table_policy->gpu_adaptive.local_hot_cache;
        local=original;
        switch(field) {
        case 0: break; // Engaged C0 is not absent/shared-only.
        case 1: local->maximum_cache_rows=1;break;
        case 2: local->cache_physical_byte_limit=1;break;
        case 3: local->policy.aging_accesses=17;break;
        case 4: local->policy.protected_numerator=0;break;
        case 5: local->policy.protected_denominator=7;break;
        case 6: local->consumer_overrides={{{0,DeviceType::ROCm,0},0,0}};break;
        case 7: local->consumer_overrides={{{0,DeviceType::ROCm,1},0,0}};break;
        case 8: local->consumer_overrides={{{0,DeviceType::ROCm,0},1,0}};break;
        case 9: local->consumer_overrides={{{0,DeviceType::ROCm,0},0,1}};break;
        }
        cell.ngram_table_policy->validate();
        EXPECT_TRUE(names.insert(cell.testName()).second)<<field;
        if(!local->consumer_overrides.empty()) {
            EXPECT_EQ(cell.makeOrchestrationConfig(cell.model.model_path,0).ngram_table_policy,*cell.ngram_table_policy);
            EXPECT_THROW(modelParityNgramStorageArguments(cell),std::invalid_argument);
        }
    }
    EXPECT_EQ(base.testName(),unchanged);
}

/** @test Declared dynamic depth survives fixture/runtime/CLI projection independently of graph capacity. */
TEST(Qwen4ExpSourceAdmissionCases, DeclaredDynamicCeilingMatchesFixtureRuntimeAndPublicArguments)
{
    const auto &cases = qwen4ExpCPUJointWorld2MTPExactCases();
    const auto original = std::find_if(cases.begin(), cases.end(), [](const auto &cell) {
        return cell.mtp == ModelParityMTP::DynamicDepth;
    });
    ASSERT_NE(original, cases.end());
    ASSERT_EQ(original->model.maximum_mtp_draft_depth, 3);
    for (const int ceiling : {1, 3, 6, 15})
        for (const int retained : {ceiling, 15})
        {
            SCOPED_TRACE(::testing::Message() << "ceiling=" << ceiling << " retained=" << retained);
            auto cell = *original;
            cell.model.maximum_mtp_draft_depth = ceiling;
            cell.retained_mtp_draft_capacity = retained;
            EXPECT_EQ(cell.requestedMTPDraftDepth(), ceiling);
            const auto fixture = cell.toTestConfig();
            EXPECT_EQ(fixture.mtp_expectation, ParityMTPExpectation::DynamicDepth);
            EXPECT_EQ(fixture.mtp_expected_draft_depth, ceiling);
            EXPECT_EQ(fixture.mtp_expected_graph_capacity, retained);
            const auto runtime = cell.makeOrchestrationConfig(cell.model.model_path, 0);
            const auto parsed = sourcePublicConfig(cell);
            for (const auto *config : {&runtime, &parsed})
            {
                EXPECT_EQ(config->mtp.draft_tokens, ceiling);
                EXPECT_EQ(config->mtp.depth_policy.max_depth, ceiling);
                EXPECT_EQ(config->mtp.depth_policy.initial_depth, ceiling);
                EXPECT_EQ(config->mtp.graph_capacity_draft_tokens, retained);
                EXPECT_EQ(resolveMTPMaximumExecutionDraftDepth(config->mtp), ceiling);
            }
            EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
        }
    // Off and explicit fixed-depth requests never derive their depth from the
    // dynamic ceiling or from the size of retained executable storage.
    auto fixed = *original;
    for (const auto [policy, expected] : std::array{
             std::pair{ModelParityMTP::Off, 0}, std::pair{ModelParityMTP::Depth1, 1},
             std::pair{ModelParityMTP::Depth3, 3}, std::pair{ModelParityMTP::Depth15, 15}})
    {
        fixed.mtp = policy;
        EXPECT_EQ(fixed.requestedMTPDraftDepth(), expected);
    }
}

/** @test Supported-intent, unqualified candidate and expected-rejection identities are complete and disjoint. */
TEST(Qwen4ExpSourceAdmissionCases, SupportedCandidateAndRejectedCasesPreserveAllOriginalIdentities)
{
    const auto &original = qwen4ExpProductionCases();
    const auto &host = qwen4ExpHostAdaptiveProductionCases();
    const auto &fixed = qwen4ExpFixedHeadProductionCases();
    const auto &adaptive = qwen4ExpGPUAdaptiveProductionCases();
    const auto &full = qwen4ExpFullROCmTP4ProductionCases();
    const auto &policies = qwen4ExpROCm4OverlayPolicyProductionCases();
    ASSERT_EQ(original.size() + host.size() + fixed.size() + adaptive.size() +
        full.size() + policies.size(), 112u);
    std::set<std::string> names;
    for (const auto *family : {&original, &host, &fixed, &adaptive, &full, &policies})
        for (const auto &cell : *family)
            ASSERT_TRUE(names.insert(cell.testName()).second);
    const auto &extended = qwen4ExpExtendedSourceAdmissionCases();
    const auto &candidates = qwen4ExpComposedCandidateSourceAdmissionCases();
    const auto &rejected = qwen4ExpRejectedSourceAdmissionCases();
    ASSERT_EQ(extended.size(), 8u);
    ASSERT_EQ(candidates.size(), 16u);
    ASSERT_EQ(rejected.size(), 12u);
    EXPECT_EQ(112u + extended.size(), 120u);
    std::size_t supported_cpu = 0;
    std::size_t unqualified_composed = 0;
    std::size_t internally_sharded = 0;
    for (const auto *family : {&extended, &candidates, &rejected})
    for (const auto &cell : *family)
    {
        SCOPED_TRACE(cell.testName());
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.model.model_path, kMainModelPath);
        EXPECT_EQ(cell.model.reference_directory, original.front().model.reference_directory);
        EXPECT_EQ(cell.model.transformer_layers, 48);
        EXPECT_EQ(cell.model.token_ids, original.front().model.token_ids);
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 0);
        EXPECT_EQ(cell.activation_precision, ActivationPrecision::FP32);
        EXPECT_EQ(cell.kv_cache_precision, KVCachePrecision::FP16);
        EXPECT_FALSE(cell.e2e_certification);
        EXPECT_FALSE(cell.paired_workload);
        if (family == &rejected)
        {
            ASSERT_TRUE(cell.expert_overlay);
            EXPECT_EQ(*cell.expert_overlay, ModelParityExpertOverlayPolicy{});
            ++internally_sharded;
        }
        else if (family == &candidates)
        {
            EXPECT_FALSE(cell.expert_overlay);
            EXPECT_EQ(cell.topology.kind, ModelParityTopologyKind::RankLocalPipelineParallel);
            EXPECT_TRUE(std::any_of(cell.topology.participants.begin(), cell.topology.participants.end(),
                [](const auto &participant) { return participant.address.isGPU(); }));
            ++unqualified_composed;
        }
        else
        {
            EXPECT_FALSE(cell.expert_overlay);
            EXPECT_TRUE(std::all_of(cell.topology.participants.begin(), cell.topology.participants.end(),
                [](const auto &participant) { return participant.address.isCPU(); }));
            ++supported_cpu;
        }
        EXPECT_LT(std::string("Qwen4ExpExtendedSourceAdmission_Qwen4ExpSourceAdmissionTest_FullSourceAdmission_").size() +
            cell.testName().size(), 256u);
    }
    EXPECT_EQ(supported_cpu, 8u);
    EXPECT_EQ(unqualified_composed, 16u);
    EXPECT_EQ(internally_sharded, 12u);
    EXPECT_EQ(names.size(), 148u);
}

/** @test Host tiers/cache ceilings and exact CPU/GPU physical counts remain independent axes. */
TEST(Qwen4ExpSourceAdmissionCases, ComposedHardwareAndHostStorageRequestsAreExplicit)
{
    using Counts = std::tuple<std::size_t, int, int>;
    using StorageAxes = std::set<std::tuple<ImmutableRowStoragePreference, std::uint64_t, std::size_t>>;
    const std::map<std::string, Counts> expected{
        {"CPU01_TP2", {2, 0, 0}}, {"CPU0_CPU1_PP2", {2, 0, 0}},
        {"ROCm0123_TP2_PP2", {0, 0, 4}}, {"CUDA01_CPU01_TP2_PP2", {2, 2, 0}},
        {"ROCm01_CPU01_TP2_PP2", {2, 0, 2}}, {"CUDA01_ROCm01_CPU01_TP2_PP3", {2, 2, 2}}};
    std::map<std::string, StorageAxes> axes;
    for (const auto *family : {&qwen4ExpExtendedSourceAdmissionCases(),
             &qwen4ExpComposedCandidateSourceAdmissionCases()})
    for (const auto &cell : *family)
    {
        if (cell.expert_overlay)
            continue;
        SCOPED_TRACE(cell.testName());
        const bool cpu_only = std::all_of(cell.topology.participants.begin(), cell.topology.participants.end(),
            [](const auto &participant) { return participant.address.isCPU(); });
        EXPECT_EQ(cpu_only, family == &qwen4ExpExtendedSourceAdmissionCases());
        std::vector<GlobalDeviceAddress> addresses;
        std::set<int> numa;
        for (const auto &participant : cell.topology.participants)
        {
            EXPECT_EQ(participant.world_rank, 0);
            addresses.push_back(participant.address);
            if (participant.address.device_type == DeviceType::CPU)
            {
                EXPECT_TRUE(participant.address.hasValidNuma());
                EXPECT_TRUE(numa.insert(participant.address.numa_node).second);
            }
        }
        const auto hardware = ParityHardwareRequirements::fromParticipants(addresses);
        EXPECT_EQ((Counts{numa.size(), hardware.cuda_devices, hardware.rocm_devices}), expected.at(cell.topology.test_id));
        if (!numa.empty())
            EXPECT_EQ(numa, (std::set<int>{0, 1}));
        ASSERT_TRUE(cell.ngram_table_policy);
        const auto &policy = *cell.ngram_table_policy;
        EXPECT_EQ(policy.distribution, ImmutableRowTableDistribution::Replicated);
        axes[cell.topology.test_id].emplace(policy.storage,
            policy.host_adaptive.maximum_cache_rows, policy.host_adaptive.cache_physical_byte_limit);
    }
    EXPECT_EQ(axes.size(), 6u);
    for (const auto &[identity, storage] : axes)
    {
        SCOPED_TRACE(identity);
        EXPECT_EQ(storage, (StorageAxes{
            {ImmutableRowStoragePreference::RAM, 0, 0},
            {ImmutableRowStoragePreference::MemoryMapped, 0, 0},
            {ImmutableRowStoragePreference::HostAdaptiveRAMSSD, 0, 0},
            {ImmutableRowStoragePreference::HostAdaptiveRAMSSD, 64, 1u << 20}}));
    }
}

/** @test Public projection preserves both supported-intent and unqualified-candidate requests without granting runtime support. */
TEST(Qwen4ExpSourceAdmissionCases, PublicProjectionPreservesLegalPoliciesAndLayerOwnership)
{
    for (const auto *family : {&qwen4ExpExtendedSourceAdmissionCases(),
             &qwen4ExpComposedCandidateSourceAdmissionCases()})
    for (const auto &cell : *family)
    {
        SCOPED_TRACE(cell.testName());
        const auto native = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        const auto parsed = sourcePublicConfig(cell);
        EXPECT_EQ(native.ngram_table_policy, *cell.ngram_table_policy);
        EXPECT_EQ(parsed.ngram_table_policy, *cell.ngram_table_policy);
        for (const auto &error : ConfigValidator::createStandard().validateToStrings(parsed))
            ADD_FAILURE() << "Invalid source-admission public configuration (syntax does not certify runtime support): " << error;
        if (cell.topology.kind == ModelParityTopologyKind::RankLocalPipelineParallel)
        {
            expectSourcePipeline(native, cell);
            expectSourcePipeline(parsed, cell);
            const auto &stages = native.pp_stage_definitions;
            if (cell.topology.test_id == "CUDA01_ROCm01_CPU01_TP2_PP3")
            {
                EXPECT_EQ(stages[0].last_layer, 7);
                EXPECT_EQ(stages[1].last_layer, 19);
            }
            else if (cell.topology.test_id == "CUDA01_CPU01_TP2_PP2" ||
                cell.topology.test_id == "ROCm01_CPU01_TP2_PP2")
                EXPECT_EQ(stages[0].last_layer, 15);
            else
                EXPECT_EQ(stages[0].last_layer, 23);
        }
        else
        {
            EXPECT_EQ(native.tp_devices, (std::vector{GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)}));
            EXPECT_EQ(parsed.tp_devices, native.tp_devices);
            EXPECT_EQ(native.tp_degree, 2);
        }
    }
}

/** @test GPU count requirements never turn TP4/8 into a smaller visible backend or confuse table/expert sharding. */
TEST(Qwen4ExpSourceAdmissionCases, RejectedCrossTierShardingKeepsBackendDegreeAndIndependentTableAxes)
{
    std::map<std::pair<DeviceType, std::size_t>, std::set<ImmutableRowTableDistribution>> axes;
    for (const auto &cell : qwen4ExpRejectedSourceAdmissionCases())
    {
        if (!cell.expert_overlay)
            continue;
        const auto &plan = *cell.topology.expert_overlay_plan;
        const auto &domain = plan.domains.front();
        std::vector<GlobalDeviceAddress> addresses;
        for (const auto &participant : cell.topology.participants)
            addresses.push_back(participant.address);
        const auto requirements = ParityHardwareRequirements::fromParticipants(addresses);
        const auto degree = domain.participants.size();
        const auto backend = domain.participants.front().device_type;
        EXPECT_TRUE(degree == 2 || degree == 4 || degree == 8);
        EXPECT_EQ(requirements.cuda_devices, backend == DeviceType::CUDA ? static_cast<int>(degree) : 0);
        EXPECT_EQ(requirements.rocm_devices, backend == DeviceType::ROCm ? static_cast<int>(degree) : 0);
        axes[{backend, degree}].insert(cell.ngram_table_policy->distribution);
        if (degree > 2)
            EXPECT_TRUE(parityHardwareInventoryFailure(requirements, {.cuda_devices = 2, .rocm_devices = 2}));
    }
    EXPECT_EQ(axes.size(), 6u);
    for (const auto &[geometry, distributions] : axes)
        EXPECT_EQ(distributions, (std::set{ImmutableRowTableDistribution::Replicated,
            ImmutableRowTableDistribution::FixedHeadSharded}));
}

/** @test Structural parsing preserves an unsupported request; only the separate rejection fixture proves the production diagnostic. */
TEST(Qwen4ExpSourceAdmissionCases, RejectedCrossTierPoliciesAreNotSubstitutedByPublicProjection)
{
    ASSERT_EQ(qwen4ExpRejectedSourceAdmissionCases().size(), 12u);
    for (const auto &cell : qwen4ExpRejectedSourceAdmissionCases())
    {
        SCOPED_TRACE(cell.testName());
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto &requested = *cell.topology.expert_overlay_plan;
        const auto native = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        const auto parsed = sourcePublicConfig(cell);
        // Public syntax accepts these exact declarations. This assertion does
        // not claim native support: PMA's unsupported cross-tier diagnostic is
        // exercised by the explicitly negative source-admission fixture.
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
        for (const auto *config : {&native, &parsed})
        {
            EXPECT_EQ(config->model_path, kMainModelPath);
            EXPECT_FALSE(config->mtp.enabled);
            EXPECT_EQ(config->ngram_table_policy, *cell.ngram_table_policy);
            ASSERT_TRUE(config->moe_routed_expert_plan);
            const auto &plan = *config->moe_routed_expert_plan;
            EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(plan).ok());
            ASSERT_EQ(plan.domains.size(), 2u);
            ASSERT_EQ(plan.routed_tiers.size(), 2u);
            EXPECT_EQ(plan.domains.front().participants, requested.domains.front().participants);
            EXPECT_EQ(plan.domains.front().backend, requested.domains.front().backend);
            EXPECT_EQ(plan.domains.front().routed_compute_policy, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            EXPECT_EQ(plan.domains.back().participants, (std::vector{GlobalDeviceAddress::cpu(0)}));
            EXPECT_EQ(plan.domains.back().routed_compute_policy, RoutedExpertComputePolicy::Apportioned);
            EXPECT_EQ(plan.continuation_domain, requested.continuation_domain);
            EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(), DenseParallelPolicy::TensorParallel);
            EXPECT_EQ(plan.routed_tiers.front().domain, requested.routed_tiers.front().domain);
            EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer, 8);
            EXPECT_EQ(plan.routed_tiers.front().priority, requested.routed_tiers.front().priority);
            EXPECT_EQ(plan.routed_tiers.front().memory_budget_bytes, 0u);
            EXPECT_EQ(plan.routed_tiers.back().domain, requested.routed_tiers.back().domain);
            EXPECT_EQ(plan.routed_tiers.back().max_experts_per_layer, 0);
            EXPECT_EQ(plan.routed_tiers.back().priority, requested.routed_tiers.back().priority);
        }
    }
}

/** @test Projection requires no reference files and malformed physical/stage/policy requests fail explicitly. */
TEST(Qwen4ExpSourceAdmissionCases, NoGoldDependencyAndInvalidDeclarationsCannotEnterAnotherMode)
{
    auto cell = qwen4ExpExtendedSourceAdmissionCases().front();
    cell.model.reference_directory = "/nonexistent/device-free-source-admission-reference";
    EXPECT_NO_THROW((void)cell.makeOrchestrationConfig(kMainModelPath, 0));
    EXPECT_THROW(qwen4ExpComposedSourceAdmissionDefinition(Qwen4ExpParityTopology::CPU), std::invalid_argument);
    const std::array duplicates{0, 0};
    EXPECT_THROW(qwen4ExpInternallyShardedSourceAdmissionDefinition(DeviceType::CUDA, duplicates), std::invalid_argument);
    EXPECT_THROW(qwen4ExpInternallyShardedSourceAdmissionDefinition(DeviceType::CPU, std::array{0, 1}), std::invalid_argument);
    auto invalid = qwen4ExpComposedSourceAdmissionDefinition(Qwen4ExpParityTopology::CPU2TP);
    invalid.topology.participants.back().address = GlobalDeviceAddress::cpu(0);
    EXPECT_THROW(expandModelParityDefinition(invalid), std::invalid_argument);
    invalid = qwen4ExpComposedSourceAdmissionDefinition(Qwen4ExpParityTopology::ROCm4TP2PP2);
    invalid.topology.pipeline_stage_sizes = {2, 1};
    EXPECT_THROW(expandModelParityDefinition(invalid), std::invalid_argument);
}

/** @test Two CPU participants require two actual NUMA domains even though both device ordinals are zero. */
TEST(Qwen4ExpSourceAdmissionCases, FrozenCPUInventoryAuthenticatesNUMAAddressesRatherThanDeviceOrdinals)
{
    const auto cell = qwen4ExpExtendedSourceAdmissionCases().front();
    ASSERT_EQ(cell.topology.test_id, "CPU01_TP2");
    ASSERT_EQ(cell.topology.participants.size(), 2u);
    EXPECT_EQ(cell.topology.participants[0].address.device_ordinal, 0);
    EXPECT_EQ(cell.topology.participants[1].address.device_ordinal, 0);
    EXPECT_EQ(cell.topology.participants[0].address.numa_node, 0);
    EXPECT_EQ(cell.topology.participants[1].address.numa_node, 1);

    CPUSocketInfo first;
    first.socket_id = first.numa_node = 0;
    first.model_name = "synthetic source-admission CPU";
    first.physical_cores = {0, 1};
    first.memory_bytes = 1024;
    first.available_memory_bytes = 768;
    HardwareInventory hardware;
    hardware.cpu_sockets = {first};
    EXPECT_THROW(requireQwen4ExpSourceCPUInventory(cell, 0, hardware), std::invalid_argument);

    auto second = first;
    second.socket_id = second.numa_node = 1;
    second.physical_cores = {2, 3};
    hardware.cpu_sockets.push_back(second);
    EXPECT_NO_THROW(requireQwen4ExpSourceCPUInventory(cell, 0, hardware));

    // Two socket records that both expose NUMA0 still cannot satisfy CPU1.
    hardware.cpu_sockets.back().numa_node = 0;
    EXPECT_THROW(requireQwen4ExpSourceCPUInventory(cell, 0, hardware), std::invalid_argument);
    hardware.cpu_sockets.back().numa_node = 1;
    auto alias = cell;
    alias.topology.participants.back().address = GlobalDeviceAddress::cpu(0);
    EXPECT_THROW(requireQwen4ExpSourceCPUInventory(alias, 0, hardware), std::invalid_argument);
    alias = cell;
    alias.topology.participants.back().address.numa_node = NUMA_NODE_UNKNOWN;
    EXPECT_THROW(requireQwen4ExpSourceCPUInventory(alias, 0, hardware), std::invalid_argument);
    alias = cell;
    alias.topology.participants.back().address.device_ordinal = 1;
    EXPECT_THROW(requireQwen4ExpSourceCPUInventory(alias, 0, hardware), std::invalid_argument);
    EXPECT_THROW(requireQwen4ExpSourceCPUInventory(cell, 1, hardware), std::invalid_argument);
}
namespace
{
/** @brief Retain a unique tiny metadata-test root independently of the prefix child under test. */
struct SourceEvidenceDirectory final
{
    std::filesystem::path path;
    /** @brief Create a unique owned test root without any model or reference files. */
    SourceEvidenceDirectory()
    {
        auto pattern = (std::filesystem::temp_directory_path() / "qfn-admission-evidence-XXXXXX").string();
        const auto *created = ::mkdtemp(pattern.data());
        if (!created)
            throw std::runtime_error("Cannot create QFN source-admission metadata fixture");
        path = created;
    }
    /** @brief Retire only this fixture's test root after every child owner has unwound. */
    ~SourceEvidenceDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
        if (error)
            ADD_FAILURE() << "Cannot retire QFN source-admission metadata fixture: " << error.message();
    }
};

/** @return Two exact synthetic NUMA localities and four ordinals per GPU vendor, without discovery. */
HardwareInventory sourceEvidenceHardware()
{
    HardwareInventory hardware;
    for (int numa = 0; numa < 2; ++numa)
    {
        CPUSocketInfo socket;
        socket.socket_id = socket.numa_node = numa;
        socket.physical_cores = {numa * 2, numa * 2 + 1};
        socket.memory_bytes = 1024;
        socket.available_memory_bytes = 768;
        hardware.cpu_sockets.push_back(socket);
    }
    for (int ordinal = 0; ordinal < 4; ++ordinal)
    {
        ComputeDevice gpu{};
        gpu.device_id = ordinal;
        gpu.numa_node = ordinal % 2;
        gpu.type = ComputeBackendType::GPU_CUDA;
        hardware.cuda_devices.push_back(gpu);
        gpu.type = ComputeBackendType::GPU_ROCM;
        hardware.rocm_devices.push_back(gpu);
    }
    return hardware;
}

/**
 * @brief Author one resolved endpoint using only this fixture's explicit host/localities.
 * @param address Original public CPU/GPU address, never a DeviceId that drops CPU NUMA.
 * @return Same endpoint with the synthetic host and frozen GPU NUMA binding.
 */
GlobalDeviceAddress sourceEvidenceResolved(GlobalDeviceAddress address)
{
    address.hostname = "fixture-host";
    if (address.isGPU())
        address.numa_node = address.device_ordinal % 2;
    return address;
}

/**
 * @brief Supply complete synthetic actual domain observations for a concrete public request.
 * @param authored Existing public configuration; no model or reference is consulted.
 * @param actual Independently authored TP/PP plan to receive domain observations.
 */
void sourceEvidenceDomains(const OrchestrationConfig &authored, RankExecutionPlan &actual)
{
    int id = 0;
    for (const auto &domain : authored.executionDomainDefinitions())
    {
        TPDomainParticipation observed;
        observed.domain_id = id++;
        observed.domain_name = domain.name;
        observed.weights = domain.weights;
        observed.backend = domain.backend;
        for (const auto &address : domain.participants)
            observed.devices.push_back(sourceEvidenceResolved(address));
        actual.my_domains.push_back(std::move(observed));
    }
}

/** @return A full 48-layer flat CPU2 request with its canonically synthesized MoE domain and no pipeline. */
RankExecutionPlan sourceEvidenceCPUPlan(const OrchestrationConfig &authored)
{
    RankExecutionPlan actual;
    actual.hostname = "fixture-host";
    actual.last_layer = 47;
    actual.tp_scope = TPScope::RANK_LOCAL;
    actual.local_tp_devices = {sourceEvidenceResolved(GlobalDeviceAddress::cpu(0)),
        sourceEvidenceResolved(GlobalDeviceAddress::cpu(1))};
    actual.local_tp_weights = authored.tp_weights;
    actual.local_tp_backend = authored.default_backend;
    actual.primary_device = actual.local_tp_devices.front();
    const auto canonical = canonicalQwen4ExpSourceAdmissionConfig(authored, 0);
    sourceEvidenceDomains(canonical, actual);
    // applyOverlayRole publishes the synthesized continuation's exact weights;
    // the empty public flat-TP weight vector is not a second live authority.
    actual.local_tp_weights = actual.my_domains.front().weights;
    return actual;
}

/** @return Literal 8/12/28-layer TP2-in-PP3 ownership, independent of any native plan builder. */
RankExecutionPlan sourceEvidencePipelinePlan(const OrchestrationConfig &authored)
{
    RankExecutionPlan actual;
    actual.hostname = "fixture-host";
    actual.last_layer = 47;
    actual.local_pp_devices = {sourceEvidenceResolved(GlobalDeviceAddress::cuda(0)),
        sourceEvidenceResolved(GlobalDeviceAddress::rocm(0)), sourceEvidenceResolved(GlobalDeviceAddress::cpu(0))};
    actual.local_pp_layer_boundaries = {0, 8, 20, 48};
    actual.local_pp_backend = ExecutionPlanBuilder{}.selectBackend(actual.local_pp_devices);
    actual.local_pp_stage_tp_info = {
        {{sourceEvidenceResolved(GlobalDeviceAddress::cuda(0)), sourceEvidenceResolved(GlobalDeviceAddress::cuda(1))},
            {0.5f, 0.5f}, CollectiveBackendType::NCCL},
        {{sourceEvidenceResolved(GlobalDeviceAddress::rocm(0)), sourceEvidenceResolved(GlobalDeviceAddress::rocm(1))},
            {0.5f, 0.5f}, CollectiveBackendType::RCCL},
        {{sourceEvidenceResolved(GlobalDeviceAddress::cpu(0)), sourceEvidenceResolved(GlobalDeviceAddress::cpu(1))},
            {0.5f, 0.5f}, CollectiveBackendType::HOST}};
    actual.primary_device = actual.local_pp_devices.front();
    sourceEvidenceDomains(authored, actual);
    return actual;
}
} // namespace

/** @test Actual CPU2 membership, order, collective and root cannot collapse to CPU0 aliases after public projection. */
TEST(Qwen4ExpSourceAdmissionCases, ActualCPUExecutionPlanRejectsDroppedReorderedAndAliasedNUMAEndpoints)
{
    const auto cell = expandModelParityDefinition(
        qwen4ExpComposedSourceAdmissionDefinition(Qwen4ExpParityTopology::CPU2TP)).front();
    const auto authored = cell.makeOrchestrationConfig(kMainModelPath, 0);
    const auto hardware = sourceEvidenceHardware();
    const auto actual = sourceEvidenceCPUPlan(authored);
    ASSERT_EQ(actual.local_tp_devices[0].device_ordinal, actual.local_tp_devices[1].device_ordinal);
    EXPECT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, actual, hardware));
    const std::vector<std::pair<const char *, std::function<void(RankExecutionPlan &)>>> mutations{
        {"dropped NUMA1", [](auto &p) { p.local_tp_devices.pop_back(); }},
        {"reordered NUMA endpoints", [](auto &p) { std::swap(p.local_tp_devices[0], p.local_tp_devices[1]); }},
        {"CPU1 aliases CPU0", [](auto &p) { p.local_tp_devices[1] = p.local_tp_devices[0]; }},
        {"wrong collective", [](auto &p) { p.local_tp_backend = CollectiveBackendType::NCCL; }},
        {"wrong TP scope", [](auto &p) { p.tp_scope = TPScope::AUTO; }},
        {"wrong explicit weights", [](auto &p) { p.local_tp_weights = {0.75f, 0.25f}; }},
        {"foreign primary NUMA", [](auto &p) { p.primary_device = p.local_tp_devices[1]; }},
        {"wrong owning rank", [](auto &p) { p.rank = 1; }},
        {"foreign endpoint host", [](auto &p) { p.local_tp_devices[1].hostname = "foreign-host"; }},
        {"unintended pipeline", [](auto &p) { p.local_pp_devices = {p.primary_device}; }}};
    for (const auto &[label, mutate] : mutations)
    {
        SCOPED_TRACE(label);
        auto changed = actual;
        mutate(changed);
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, changed, hardware), std::invalid_argument);
    }
    auto missing = hardware;
    missing.cpu_sockets.pop_back();
    EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, actual, missing), std::invalid_argument);
}

/** @test Original world-rank domains prove CPU fragments independently of a follower's control endpoint. */
TEST(Qwen4ExpSourceAdmissionCases, ActualCPUJointPlanRequiresOriginalRankDomainsAndFollowerRole)
{
    const auto hardware = sourceEvidenceHardware();
    for (const auto &cell : qwen4ExpCPUJointWorld2MTPSourceAdmissionCases())
    {
        SCOPED_TRACE(cell.testName());
        const bool active_mtp = cell.mtp != ModelParityMTP::Off;
        ASSERT_EQ(cell.topology.mpi_ranks, 2);
        for (int rank = 0; rank < cell.topology.mpi_ranks; ++rank)
        {
            SCOPED_TRACE(rank);
            const auto authored = cell.makeOrchestrationConfig(kMainModelPath, rank);
            RankExecutionPlan actual;
            actual.rank = rank;
            actual.runtime.mtp = authored.mtp;
            actual.hostname = "fixture-host";
            actual.last_layer = 47;
            sourceEvidenceDomains(authored, actual);
            ASSERT_EQ(actual.my_domains.size(), 2u);
            if (rank == 0)
            {
                actual.primary_device = sourceEvidenceResolved(GlobalDeviceAddress::rocm(0));
                actual.local_tp_backend = CollectiveBackendType::RCCL;
            }
            else
            {
                actual.my_domains.erase(actual.my_domains.begin());
                actual.my_domains.front().my_index_in_domain = 1;
                // This is exactly the follower control endpoint, not NUMA0 ownership.
                actual.primary_device = GlobalDeviceAddress::cpu();
                actual.local_tp_backend = CollectiveBackendType::AUTO;
            }
            EXPECT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, rank, authored, actual, hardware));
            if (active_mtp)
            {
                const auto serving = qwen4ExpMTPSourceServingParticipants(cell, rank, authored, actual, hardware,
                    Qwen4ExpMTPSourceProgramFamily::CompleteSparse, Qwen4ExpMTPHostSetup::OverlayMaterialized);
                if (rank == 0)
                {
                    ASSERT_EQ(serving.size(), 1u);
                    EXPECT_EQ(serving.front().device, DeviceId::rocm(0));
                    EXPECT_TRUE(serving.front().completion_required);
                }
                else
                {
                    ASSERT_EQ(serving.size(), 1u);
                    EXPECT_EQ(serving.front().role, Qwen4ExpMTPSourceParticipantRole::ExpertFragment);
                    EXPECT_FALSE(serving.front().completion_required);
                    EXPECT_EQ(requireQwen4ExpMTPSourceSetupFamilies({}, {}, serving), 0u);
                    // A continuation completion on an expert-only rank is foreign evidence.
                    PerfStatRecord foreign;
                    foreign.domain = "forward_graph"; foreign.phase = "setup";
                    foreign.device = DeviceId::cpu().toString();
                    foreign.name = "serving_graph_family_participant_completions";
                    foreign.value = 1;
                    const std::array foreign_rows{foreign};
                    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, foreign_rows, serving), std::logic_error);
                    auto invalid = serving; invalid.front().completion_required = true;
                    EXPECT_THROW(requireQwen4ExpMTPSourceSetupFamilies({}, {}, invalid), std::invalid_argument);
                }
            }
            const std::vector<std::pair<const char *, std::function<void(RankExecutionPlan &)>>> mutations{
                {"missing original CPU domain", [](auto &p) { p.my_domains.pop_back(); }},
                {"wrong CPU participant index", [](auto &p) { p.my_domains.back().my_index_in_domain ^= 1; }},
                {"aliased CPU NUMA endpoint", [](auto &p) { p.my_domains.back().devices[1] = p.my_domains.back().devices[0]; }},
                {"changed CPU collective", [](auto &p) { p.my_domains.back().backend = CollectiveBackendType::HOST; }},
                {"extra original domain copy", [](auto &p) { p.my_domains.push_back(p.my_domains.back()); }},
                {"unintended local TP", [](auto &p) { p.local_tp_devices = p.my_domains.back().devices; }},
                {"unintended pipeline", [](auto &p) { p.local_pp_devices = {p.primary_device}; }},
                {"foreign world rank", [](auto &p) { p.rank ^= 1; }},
                {"lost original source layer", [](auto &p) { p.last_layer = 46; }},
            };
            for (const auto &[label, mutate] : mutations)
            {
                SCOPED_TRACE(label);
                auto changed = actual;
                mutate(changed);
                EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, rank, authored, changed, hardware), std::invalid_argument);
                if (active_mtp) EXPECT_THROW(qwen4ExpMTPSourceServingParticipants(cell, rank, authored, changed, hardware,
                    Qwen4ExpMTPSourceProgramFamily::CompleteSparse, Qwen4ExpMTPHostSetup::OverlayMaterialized),
                    std::invalid_argument);
            }
            if (rank != 0)
            {
                auto foreign_continuation = actual;
                foreign_continuation.primary_device = sourceEvidenceResolved(GlobalDeviceAddress::rocm(0));
                EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, rank, authored, foreign_continuation, hardware), std::invalid_argument);
            }
        }
    }
}

/** @test PP roots alone cannot certify complete stage TP groups, layers, domain ownership or collectives. */
TEST(Qwen4ExpSourceAdmissionCases, ActualPipelineRejectsForeignStageRootsMembershipAndLayerBoundaries)
{
    const auto cell = expandModelParityDefinition(
        qwen4ExpComposedSourceAdmissionDefinition(Qwen4ExpParityTopology::CUDA2ROCm2CPU2TPPP)).front();
    const auto authored = cell.makeOrchestrationConfig(kMainModelPath, 0);
    const auto hardware = sourceEvidenceHardware();
    const auto actual = sourceEvidencePipelinePlan(authored);
    EXPECT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, actual, hardware));
    const std::vector<std::pair<const char *, std::function<void(RankExecutionPlan &)>>> mutations{
        {"dropped stage root", [](auto &p) { p.local_pp_devices.pop_back(); }},
        {"reordered stage roots", [](auto &p) { std::swap(p.local_pp_devices[0], p.local_pp_devices[1]); }},
        {"foreign stage root", [](auto &p) { p.local_pp_devices[1] = sourceEvidenceResolved(GlobalDeviceAddress::rocm(1)); }},
        {"dropped TP member", [](auto &p) { p.local_pp_stage_tp_info[0].devices.pop_back(); }},
        {"reordered TP members", [](auto &p) { std::swap(p.local_pp_stage_tp_info[0].devices[0], p.local_pp_stage_tp_info[0].devices[1]); }},
        {"aliased CPU1 in final stage", [](auto &p) { p.local_pp_stage_tp_info[2].devices[1] = p.local_pp_stage_tp_info[2].devices[0]; }},
        {"wrong stage collective", [](auto &p) { p.local_pp_stage_tp_info[1].tp_backend = CollectiveBackendType::NCCL; }},
        {"wrong aggregate pipeline collective", [](auto &p) { p.local_pp_backend = CollectiveBackendType::HOST; }},
        {"wrong stage weights", [](auto &p) { p.local_pp_stage_tp_info[1].tp_weights = {0.75f, 0.25f}; }},
        {"shifted layer boundary", [](auto &p) { p.local_pp_layer_boundaries[1] = 9; }},
        {"truncated final layer", [](auto &p) { p.local_pp_layer_boundaries.back() = 47; }},
        {"lost rank layer", [](auto &p) { p.last_layer = 46; }},
        {"outer flat TP replaces composition", [](auto &p) { p.local_tp_devices = p.local_pp_stage_tp_info[0].devices; }},
        {"missing published domain", [](auto &p) { p.my_domains.pop_back(); }},
        {"duplicate published domain alias", [](auto &p) { p.my_domains[1].domain_name = p.my_domains[0].domain_name; }},
        {"foreign domain collective", [](auto &p) { p.my_domains[0].backend = CollectiveBackendType::HOST; }},
        {"foreign domain member", [](auto &p) { p.my_domains[1].devices[1] = sourceEvidenceResolved(GlobalDeviceAddress::rocm(2)); }},
        {"foreign rank hostname", [](auto &p) { p.hostname = "foreign-host"; }},
        {"wrong resolved GPU NUMA", [](auto &p) { p.local_pp_stage_tp_info[0].devices[1].numa_node = 0; }}};
    for (const auto &[label, mutate] : mutations)
    {
        SCOPED_TRACE(label);
        auto changed = actual;
        mutate(changed);
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, changed, hardware), std::invalid_argument);
    }
    auto wrong_stage = authored;
    wrong_stage.pp_stage_definitions[0].domain_name = wrong_stage.pp_stage_definitions[1].domain_name;
    EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, wrong_stage, actual, hardware), std::invalid_argument);
}

/** @test Only local host aliases and the same frozen GPU's previously unknown NUMA may normalize. */
TEST(Qwen4ExpSourceAdmissionCases, EndpointNormalizationNeverChangesExplicitCPUOrGpuIdentity)
{
    const auto hardware = sourceEvidenceHardware();
    const auto requested = GlobalDeviceAddress::cuda(1);
    const auto actual = sourceEvidenceResolved(requested);
    EXPECT_NO_THROW(requireQwen4ExpSourceEndpoint(requested, actual, "fixture-host", hardware));
    EXPECT_NO_THROW(requireQwen4ExpSourceEndpoint(requested, requested, "fixture-host", hardware));
    auto explicit_gpu = requested;
    explicit_gpu.numa_node = 0;
    EXPECT_THROW(requireQwen4ExpSourceEndpoint(explicit_gpu, actual, "fixture-host", hardware), std::invalid_argument);
    auto foreign = actual;
    foreign.hostname = "foreign-host";
    EXPECT_THROW(requireQwen4ExpSourceEndpoint(requested, foreign, "fixture-host", hardware), std::invalid_argument);
    foreign = actual;
    foreign.device_type = DeviceType::ROCm;
    EXPECT_THROW(requireQwen4ExpSourceEndpoint(requested, foreign, "fixture-host", hardware), std::invalid_argument);
    auto absent = hardware;
    absent.cuda_devices.clear();
    EXPECT_THROW(requireQwen4ExpSourceEndpoint(requested, actual, "fixture-host", absent), std::invalid_argument);
}

/** @test A named-overlay TP topology retains its sole authority instead of requiring a competing flat public map. */
TEST(Qwen4ExpSourceAdmissionCases, NamedFullROCmTPPlanHasNoCompetingAuthoredFlatMap)
{
    const auto &cell = qwen4ExpFullROCmTP4ProductionCases().front();
    const auto authored = cell.makeOrchestrationConfig(kMainModelPath, 0);
    ASSERT_TRUE(authored.moe_routed_expert_plan);
    ASSERT_TRUE(authored.tp_devices.empty());
    ASSERT_EQ(cell.topology.kind, ModelParityTopologyKind::RankLocalTensorParallel);
    const auto hardware = sourceEvidenceHardware();
    RankExecutionPlan actual;
    actual.hostname = "fixture-host";
    actual.last_layer = 47;
    sourceEvidenceDomains(authored, actual);
    const auto &domain = actual.my_domains.front();
    actual.local_tp_devices = domain.devices;
    actual.local_tp_weights = domain.weights;
    actual.local_tp_backend = domain.backend;
    actual.tp_scope = TPScope::RANK_LOCAL;
    actual.primary_device = domain.devices.front();
    EXPECT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, actual, hardware));
    for (const auto &mutate : std::vector<std::function<void(RankExecutionPlan &)>>{
        [](auto &p) { p.local_tp_devices.pop_back(); },
        [](auto &p) { p.tp_scope = TPScope::AUTO; },
        [](auto &p) { p.local_tp_weights = {0.5f, 0.5f}; },
        [](auto &p) { p.local_tp_backend = CollectiveBackendType::NCCL; }})
    {
        auto changed = actual;
        mutate(changed);
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, changed, hardware), std::invalid_argument);
    }
}

/** @test A SINGLE named continuation is an exact root, not a fabricated one-participant LocalTP domain. */
TEST(Qwen4ExpSourceAdmissionCases, NamedSingletonContinuationRetainsEmptyLocalTPAndExactRoot)
{
    auto definition = qwen4ExpProductionDefinition(Qwen4ExpParityTopology::ROCmCPU);
    definition.features.ngram_storage = {{.storage = ImmutableRowStoragePreference::RAM}};
    const auto cell = expandModelParityDefinition(definition).front();
    const auto authored = cell.makeOrchestrationConfig(kMainModelPath, 0);
    ASSERT_TRUE(authored.moe_routed_expert_plan);
    ASSERT_EQ(authored.moe_routed_expert_plan->domains.front().scope, ExecutionDomainScope::SINGLE);
    const auto hardware = sourceEvidenceHardware();
    RankExecutionPlan actual;
    actual.hostname = "fixture-host";
    actual.last_layer = 47;
    actual.primary_device = sourceEvidenceResolved(GlobalDeviceAddress::rocm(0));
    sourceEvidenceDomains(authored, actual);
    ASSERT_TRUE(actual.local_tp_devices.empty());
    ASSERT_TRUE(actual.local_tp_weights.empty());
    ASSERT_EQ(actual.tp_scope, TPScope::AUTO);
    EXPECT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, actual, hardware));
    const std::vector<std::pair<const char *, std::function<void(RankExecutionPlan &)>>> mutations{
        {"fabricated singleton TP group", [](auto &p) { p.local_tp_devices = {p.primary_device}; }},
        {"stray TP weights", [](auto &p) { p.local_tp_weights = {1.0f}; }},
        {"wrong singleton scope", [](auto &p) { p.tp_scope = TPScope::RANK_LOCAL; }},
        {"foreign root", [](auto &p) { p.primary_device = sourceEvidenceResolved(GlobalDeviceAddress::rocm(1)); }},
        {"missing CPU capacity member", [](auto &p) { p.my_domains.back().devices.clear(); }}};
    for (const auto &[label, mutate] : mutations)
    {
        SCOPED_TRACE(label);
        auto changed = actual;
        mutate(changed);
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, changed, hardware), std::invalid_argument);
    }
}

/** @test Existing outer PP transport policy is independent of strict stage-internal TP collectives. */
TEST(Qwen4ExpSourceAdmissionCases, OuterPipelineCollectiveUsesCanonicalSelectorOnExactStageRoots)
{
    const auto hardware = sourceEvidenceHardware();
    for (const auto mode : {Qwen4ExpParityTopology::CPU2PP, Qwen4ExpParityTopology::CUDA2CPU2TPPP})
    {
        // The composed candidate's metadata can be checked without claiming
        // that native TP-in-PP materialization is currently supported.
        const auto cell = expandModelParityDefinition(qwen4ExpComposedSourceAdmissionDefinition(mode)).front();
        const auto authored = cell.makeOrchestrationConfig(kMainModelPath, 0);
        RankExecutionPlan actual;
        actual.hostname = "fixture-host";
        actual.last_layer = 47;
        if (mode == Qwen4ExpParityTopology::CPU2PP)
        {
            actual.local_pp_devices = {sourceEvidenceResolved(GlobalDeviceAddress::cpu(0)),
                sourceEvidenceResolved(GlobalDeviceAddress::cpu(1))};
            actual.local_pp_layer_boundaries = {0, 24, 48};
            actual.local_pp_stage_tp_info = {
                {{actual.local_pp_devices[0]}, {}, CollectiveBackendType::AUTO},
                {{actual.local_pp_devices[1]}, {}, CollectiveBackendType::AUTO}};
        }
        else
        {
            actual.local_pp_devices = {sourceEvidenceResolved(GlobalDeviceAddress::cuda(0)),
                sourceEvidenceResolved(GlobalDeviceAddress::cpu(0))};
            actual.local_pp_layer_boundaries = {0, 16, 48};
            actual.local_pp_stage_tp_info = {
                {{sourceEvidenceResolved(GlobalDeviceAddress::cuda(0)), sourceEvidenceResolved(GlobalDeviceAddress::cuda(1))},
                    {0.5f, 0.5f}, CollectiveBackendType::NCCL},
                {{sourceEvidenceResolved(GlobalDeviceAddress::cpu(0)), sourceEvidenceResolved(GlobalDeviceAddress::cpu(1))},
                    {0.5f, 0.5f}, CollectiveBackendType::HOST}};
        }
        actual.primary_device = actual.local_pp_devices.front();
        sourceEvidenceDomains(authored, actual);
        actual.local_pp_backend = ExecutionPlanBuilder{}.selectBackend(actual.local_pp_devices);
        EXPECT_EQ(actual.local_pp_backend, mode == Qwen4ExpParityTopology::CPU2PP
            ? CollectiveBackendType::UPI : CollectiveBackendType::HOST);
        EXPECT_NE(actual.local_pp_backend, authored.default_backend);
        EXPECT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, actual, hardware));
        auto wrong = actual;
        wrong.local_pp_backend = authored.default_backend;
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, wrong, hardware), std::invalid_argument);
        wrong = actual;
        wrong.local_pp_stage_tp_info.back().tp_backend = CollectiveBackendType::RCCL;
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, wrong, hardware), std::invalid_argument);
    }
}

/** @test Same-sized foreign paths, offsets, formats and reshaped rows cannot masquerade as original GGUF table metadata. */
TEST(Qwen4ExpSourceAdmissionCases, NativeTableSourceIdentityRejectsSameByteForeignCoordinates)
{
    GGUFModel directory;
    directory.split_count = 2;
    directory.data_offset = 128;
    directory.split_paths = {"/synthetic/qfn/part0.gguf", "/synthetic/qfn/part1.gguf"};
    directory.split_data_offsets = {128, 1024};
    directory.tensors.push_back({.name = "per_layer_token_embd.weight", .dimensions = {4, 32},
        .type = GGUFTensorType::F16, .offset = 64, .size_bytes = 256, .split_idx = 1});
    const auto artifacts = directory.split_paths;
    const ImmutableRowTableSource source{.path = artifacts[1], .tensor_name = "per_layer_token_embd.weight",
        .file_offset = 1088, .rows = 4, .row_elements = 32, .format = GGUFTensorType::F16};
    ASSERT_EQ(source.tensorBytes(), 256u);
    EXPECT_NO_THROW(requireQwen4ExpSourceTableIdentity(directory, artifacts, source));
    const std::vector<std::pair<const char *, std::function<void(ImmutableRowTableSource &)>>> mutations{
        {"same-sized foreign shard", [](auto &s) { s.path = "/synthetic/foreign/part1.gguf"; }},
        {"wrong native tensor", [](auto &s) { s.tensor_name = "different.weight"; }},
        {"shifted source offset", [](auto &s) { ++s.file_offset; }},
        {"same-byte foreign format", [](auto &s) { s.format = GGUFTensorType::BF16; }},
        {"same-byte transposed geometry", [](auto &s) { s.rows = 32; s.row_elements = 4; }}};
    for (const auto &[label, mutate] : mutations)
    {
        SCOPED_TRACE(label);
        auto changed = source;
        mutate(changed);
        EXPECT_EQ(changed.tensorBytes(), source.tensorBytes());
        EXPECT_THROW(requireQwen4ExpSourceTableIdentity(directory, artifacts, changed), std::invalid_argument);
    }
    auto wrong_artifacts = artifacts;
    std::swap(wrong_artifacts[0], wrong_artifacts[1]);
    EXPECT_THROW(requireQwen4ExpSourceTableIdentity(directory, wrong_artifacts, source), std::invalid_argument);
    auto incomplete = directory;
    incomplete.split_data_offsets.pop_back();
    EXPECT_THROW(requireQwen4ExpSourceTableIdentity(incomplete, artifacts, source), std::invalid_argument);
    auto overflow = directory;
    overflow.tensors.front().offset = std::numeric_limits<std::uint64_t>::max();
    EXPECT_THROW(requireQwen4ExpSourceTableIdentity(overflow, artifacts, source), std::invalid_argument);
}

/** @test Exact child cleanup is idempotent on normal/exception exits and never removes sibling admission evidence. */
TEST(Qwen4ExpSourceAdmissionCases, PrefixDirectoryRetiresOnlyOwnedChildOnNormalAndExceptionExit)
{
    const SourceEvidenceDirectory root;
    const auto prefix = root.path / "prefix-cache";
    const auto evidence = root.path / "source-admission.json";
    std::ofstream(evidence) << "retained metadata";
    {
        Qwen4ExpAdmissionPrefixDirectory owner(prefix);
        std::ofstream(prefix / "dummy-cache") << "tiny fixture";
        EXPECT_TRUE(std::filesystem::exists(prefix / "dummy-cache"));
        EXPECT_NO_THROW(owner.cleanup());
        EXPECT_FALSE(std::filesystem::exists(prefix));
        EXPECT_NO_THROW(owner.cleanup());
        EXPECT_TRUE(std::filesystem::exists(evidence));
    }
    EXPECT_FALSE(std::filesystem::exists(prefix));
    EXPECT_THROW(([&] {
        Qwen4ExpAdmissionPrefixDirectory owner(prefix);
        std::ofstream(prefix / "dummy-cache") << "tiny fixture";
        throw std::runtime_error("synthetic admission unwind");
    }()), std::runtime_error);
    EXPECT_FALSE(std::filesystem::exists(prefix));
    EXPECT_TRUE(std::filesystem::exists(evidence));
    std::string retained;
    std::ifstream stream(evidence);
    std::getline(stream, retained);
    EXPECT_EQ(retained, "retained metadata");
    EXPECT_THROW(Qwen4ExpAdmissionPrefixDirectory(root.path / "source-admission"), std::invalid_argument);
    EXPECT_TRUE(std::filesystem::exists(evidence));
}
/** @test Full-rank layer coverage cannot omit either original embedding or terminal head ownership. */
TEST(Qwen4ExpSourceAdmissionCases, FullModelExecutionPlanRequiresEmbeddingAndLMHeadOwners)
{
    const auto hardware = sourceEvidenceHardware();
    for (const auto mode : {Qwen4ExpParityTopology::CPU2TP, Qwen4ExpParityTopology::CUDA2ROCm2CPU2TPPP})
    {
        const auto cell = expandModelParityDefinition(qwen4ExpComposedSourceAdmissionDefinition(mode)).front();
        const auto authored = cell.makeOrchestrationConfig(kMainModelPath, 0);
        const auto actual = mode == Qwen4ExpParityTopology::CPU2TP
            ? sourceEvidenceCPUPlan(authored) : sourceEvidencePipelinePlan(authored);
        ASSERT_EQ(actual.first_layer, 0);
        ASSERT_EQ(actual.last_layer, 47);
        ASSERT_TRUE(actual.has_embedding);
        ASSERT_TRUE(actual.has_lm_head);
        EXPECT_NO_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, actual, hardware));
        auto missing_embedding = actual;
        missing_embedding.has_embedding = false;
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, missing_embedding, hardware), std::invalid_argument);
        auto missing_head = actual;
        missing_head.has_lm_head = false;
        EXPECT_THROW(requireQwen4ExpSourceExecutionPlan(cell, 0, authored, missing_head, hardware), std::invalid_argument);
    }
}

/** @test Existing reports are immutable while missing or legitimately precreated empty output directories are accepted. */
TEST(Qwen4ExpSourceAdmissionCases, EvidenceNamespaceRefusesPriorResultsAndAcceptsEmptyDirectories)
{
    const SourceEvidenceDirectory root;
    const auto output = root.path / "fresh-evidence";
    EXPECT_NO_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(output));
    EXPECT_TRUE(std::filesystem::is_directory(output));
    EXPECT_TRUE(std::filesystem::is_empty(output));
    EXPECT_NO_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(output));
    const auto prior = output / "admission.csv";
    std::ofstream(prior) << "preserve prior result";
    EXPECT_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(output), std::invalid_argument);
    std::ifstream retained(prior);
    std::string contents;
    std::getline(retained, contents);
    EXPECT_EQ(contents, "preserve prior result");
    EXPECT_TRUE(std::filesystem::exists(prior));
    const auto empty = root.path / "framework-empty-evidence";
    ASSERT_TRUE(std::filesystem::create_directory(empty));
    EXPECT_NO_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(empty));
    EXPECT_TRUE(std::filesystem::is_empty(empty));
    EXPECT_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(prior), std::invalid_argument);
    EXPECT_THROW(prepareQwen4ExpAdmissionEvidenceDirectory("relative-evidence"), std::invalid_argument);
    const auto alias = root.path / "evidence-alias";
    std::filesystem::create_directory_symlink(empty, alias);
    EXPECT_THROW(prepareQwen4ExpAdmissionEvidenceDirectory(alias), std::invalid_argument);
    EXPECT_TRUE(std::filesystem::exists(empty));
    EXPECT_TRUE(std::filesystem::exists(prior));
}

/** @test Admission preserves Dynamic maintenance and all typed policy fields. */
TEST(Qwen4ExpSourceAdmissionCases, MaintenancePolicyRejectsStaticAndChangedCadenceOrCapacity)
{
    MoERebalanceRuntimeConfig authored;
    authored.mode = MoERebalanceRuntimeMode::Dynamic;
    EXPECT_NO_THROW(requireQwen4ExpSourceMaintenancePolicy(authored, authored));
    auto changed = authored;
    // Static residency projects to maintenance Off in ModelParityCase.
    changed.mode = MoERebalanceRuntimeMode::Off;
    EXPECT_THROW(requireQwen4ExpSourceMaintenancePolicy(authored, changed), std::invalid_argument);
    changed = authored;
    ++changed.window_size;
    EXPECT_THROW(requireQwen4ExpSourceMaintenancePolicy(authored, changed), std::invalid_argument);
    changed = authored;
    ++changed.migration_transfer_slots;
    EXPECT_THROW(requireQwen4ExpSourceMaintenancePolicy(authored, changed), std::invalid_argument);
    changed = authored;
    changed.migration_cycles_per_wave = authored.resolvedMigrationCyclesPerWave();
    EXPECT_THROW(requireQwen4ExpSourceMaintenancePolicy(authored, changed), std::invalid_argument);
    changed = authored;
    changed.release_raw_expert_weights = !authored.release_raw_expert_weights;
    EXPECT_THROW(requireQwen4ExpSourceMaintenancePolicy(authored, changed), std::invalid_argument);
}

/** @brief A provider's exact bytes coexist with physically live GPU channels. */
TEST(Qwen4ExpSourceAdmissionCases, HostLookupStagingJoinsCompleteOwnerWithPlacementConsumers)
{
    const PhysicalMemoryAllocatorIdentity host{0, DeviceId::cpu()};
    const std::vector<PhysicalMemoryAllocatorIdentity> cpu{host};
    const std::vector<PhysicalMemoryAllocatorIdentity> mixed{
        {0, DeviceId::cuda(0)}, {0, DeviceId::rocm(0)}, host};
    // Literal extents from the independently observed mixed native admission:
    // the provider owns 589824 bytes; mapped channels add another 806912.
    constexpr std::size_t provider = 589824u;
    std::vector<PhysicalMemoryOwnerAttestation> rows{{.identity = host,
        .owner = PhysicalMemoryOwner::ImmutableLookupStaging,
        .planned_new_bytes = provider, .materialized_new_bytes = provider,
        .committed_new_bytes = provider}};
    EXPECT_NO_THROW(requireQwen4ExpHostLookupStaging(rows, host, provider, cpu));
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, provider, mixed), std::invalid_argument);
    rows.front().planned_new_bytes = 1396736u;
    rows.front().materialized_new_bytes = 1396736u;
    rows.front().committed_new_bytes = 1396736u;
    EXPECT_NO_THROW(requireQwen4ExpHostLookupStaging(rows, host, provider, mixed));
    // CPU-only staging must not silently retain unrelated/obsolete channel bytes.
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, provider, cpu), std::invalid_argument);
    // Whole RAM/mmap providers have zero own staging but GPU channels remain live.
    EXPECT_NO_THROW(requireQwen4ExpHostLookupStaging(rows, host, 0u, mixed));
    rows.front().planned_new_bytes = 0u;
    rows.front().materialized_new_bytes = 0u;
    rows.front().committed_new_bytes = 0u;
    EXPECT_NO_THROW(requireQwen4ExpHostLookupStaging(rows, host, 0u, cpu));
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 0u, mixed), std::invalid_argument);
}

/** @brief Reservations and partial commitment cannot certify a retained channel. */
TEST(Qwen4ExpSourceAdmissionCases, HostLookupStagingRejectsReservationOnlyAndIncompleteOwners)
{
    const PhysicalMemoryAllocatorIdentity host{0, DeviceId::cpu()};
    const std::vector<PhysicalMemoryAllocatorIdentity> consumers{{0, DeviceId::rocm(0)}, host};
    const PhysicalMemoryOwnerAttestation complete{.identity = host,
        .owner = PhysicalMemoryOwner::ImmutableLookupStaging,
        .planned_new_bytes = 1396736u, .materialized_new_bytes = 1396736u,
        .committed_new_bytes = 1396736u};
    std::vector<PhysicalMemoryOwnerAttestation> rows{complete};
    rows.front().materialized_new_bytes = 589824u;
    EXPECT_TRUE(rows.front().committed());
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers), std::invalid_argument);
    rows.front() = complete;
    rows.front().committed_new_bytes = 589824u;
    EXPECT_TRUE(rows.front().complete());
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers), std::invalid_argument);
    rows.front() = complete;
    rows.front().planned_resident_bytes = 4096u;
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers), std::invalid_argument);
    rows.front().adopted_resident_bytes = 4096u;
    EXPECT_NO_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers));
}

/** @brief Owner identity and checked live extents remain mandatory for mixed lookup. */
TEST(Qwen4ExpSourceAdmissionCases, HostLookupStagingRejectsForeignOwnerAndOverflow)
{
    const PhysicalMemoryAllocatorIdentity host{0, DeviceId::cpu()};
    const std::vector<PhysicalMemoryAllocatorIdentity> consumers{{0, DeviceId::cuda(0)}, host};
    std::vector<PhysicalMemoryOwnerAttestation> rows{{.identity = {1, DeviceId::cpu()},
        .owner = PhysicalMemoryOwner::ImmutableLookupStaging,
        .planned_new_bytes = 1396736u, .materialized_new_bytes = 1396736u,
        .committed_new_bytes = 1396736u}};
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers), std::invalid_argument);
    rows.front().identity = host;
    rows.front().owner = PhysicalMemoryOwner::ImmutableTableStorage;
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers), std::invalid_argument);
    rows.clear();
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers), std::invalid_argument);
    rows.push_back({.identity = host, .owner = PhysicalMemoryOwner::ImmutableLookupStaging,
        .planned_new_bytes = std::numeric_limits<std::size_t>::max(), .planned_resident_bytes = 1u,
        .materialized_new_bytes = std::numeric_limits<std::size_t>::max(),
        .committed_new_bytes = std::numeric_limits<std::size_t>::max(), .adopted_resident_bytes = 1u});
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, consumers), std::overflow_error);
    const std::vector<PhysicalMemoryAllocatorIdentity> empty;
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, host, 589824u, empty), std::invalid_argument);
    EXPECT_THROW(requireQwen4ExpHostLookupStaging(rows, {0, DeviceId::cuda(0)}, 589824u, consumers), std::invalid_argument);
}

/** @test Planned or reserved CPU bytes cannot replace actual weight and arena claims. */
TEST(Qwen4ExpSourceAdmissionCases, CPUStorageMaterializationRejectsMissingLiveOwners)
{
    using Owner = PhysicalMemoryOwner;
    const PhysicalMemoryAllocatorIdentity cpu{0, DeviceId::cpu()};
    std::vector<PhysicalMemoryOwnerAttestation> rows;
    for (const auto owner : {Owner::ActivationArena, Owner::RoutedExpertWeights,
            Owner::PrimaryModelWeights, Owner::AdditionalModelWeights, Owner::ExecutionWorkspace})
        rows.push_back({.identity=cpu, .owner=owner, .planned_new_bytes=4096,
            .materialized_new_bytes=64, .committed_new_bytes=64});
    EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(rows));
    for (std::size_t index : {0u, 1u, 4u})
    {
        auto changed = rows;
        changed[index].materialized_new_bytes = 0;
        changed[index].committed_new_bytes = 4096;
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(changed), std::logic_error);
    }
    rows[2].materialized_new_bytes = 0;
    EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(rows));
    rows[3].materialized_new_bytes = 0;
    EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows), std::logic_error);
    auto foreign = rows[2];
    foreign.identity.device = DeviceId{DeviceType::CPU, 1};
    foreign.materialized_new_bytes = 64;
    rows.push_back(foreign);
    EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows), std::logic_error);
    rows.back().identity.device = DeviceId::cuda(0);
    EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows), std::logic_error);
    rows[2].adopted_resident_bytes = rows[2].planned_resident_bytes = 64;
    rows.push_back({.identity=cpu, .owner=Owner::PrefixArchiveStaging,
        .planned_new_bytes=8192, .committed_new_bytes=8192});
    EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(rows));
}

/** @test Frozen empty CPU tiers retain capacity without inventing live computation. */
TEST(Qwen4ExpSourceAdmissionCases, CPUStorageMaterializationUsesFrozenComputeOwners)
{
    using Owner = PhysicalMemoryOwner;
    const PhysicalMemoryAllocatorIdentity cpu{0, DeviceId::cpu()};
    // An original all-GPU compute plan can retain a genuine RAM ngram source
    // on CPU0 without executing CPU layers or materializing CPU workspaces.
    auto gpu_only = std::make_shared<MoERoutedExpertPlacementPlan>(
        *qwen4ExpAllGPUQ6MTPDefinition().topology.expert_overlay_plan);
    gpu_only->placements = {{.layer = 0, .routed_expert_tier = {0, 1}}};
    gpu_only->routed_tiers[0].resolved_live_experts_per_layer = {1};
    gpu_only->routed_tiers[1].resolved_live_experts_per_layer = {1};
    const std::array lookup_owners{cpu};
    std::vector<PhysicalMemoryOwnerAttestation> source_rows{
        {.identity=cpu, .owner=Owner::ActivationArena, .planned_new_bytes=4096},
        {.identity=cpu, .owner=Owner::ExecutionWorkspace, .planned_new_bytes=8192}};
    EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(source_rows, gpu_only), std::logic_error);
    EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(source_rows, gpu_only, ImmutableRowAllocatorInventory{lookup_owners}));
    for (const auto foreign : {PhysicalMemoryAllocatorIdentity{1, DeviceId::cpu()},
                              PhysicalMemoryAllocatorIdentity{0, DeviceId{DeviceType::CPU,1}}}) {
        const std::array foreign_owners{foreign};
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(source_rows, gpu_only, ImmutableRowAllocatorInventory{foreign_owners}), std::logic_error);
    }
    for (const auto owner : {Owner::RoutedExpertWeights, Owner::PrimaryModelWeights, Owner::AdditionalModelWeights}) {
        auto missing = source_rows;
        missing.push_back({.identity=cpu, .owner=owner, .planned_new_bytes=64});
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(missing, gpu_only, ImmutableRowAllocatorInventory{lookup_owners}), std::logic_error);
    }
    for (const int first : {0, 17})
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(
            *qwen4ExpHardwareTargetDefinition(Qwen4ExpHardwareTarget::ROCm4).topology.expert_overlay_plan);
        ASSERT_EQ(plan->routed_tiers.size(), 2u);
        plan->first_model_layer = first;
        plan->placements = {{.layer = first, .routed_expert_tier = {0, 0}},
                            {.layer = first + 1, .routed_expert_tier = {0, 0}}};
        plan->routed_tiers.front().resolved_live_experts_per_layer = {2, 2};
        plan->routed_tiers.back().resolved_live_experts_per_layer = {0, 0};
        std::vector<PhysicalMemoryOwnerAttestation> rows{
            {.identity = cpu, .owner = Owner::ActivationArena, .planned_new_bytes = 4096},
            {.identity = cpu, .owner = Owner::ExecutionWorkspace, .planned_new_bytes = 8192},
            {.identity = cpu, .owner = Owner::PrefixHostTier, .planned_new_bytes = 16384}};
        // This is the old capacity-only assertion's exact failing observation.
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows), std::logic_error);
        EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(rows, plan));
        const auto changed = [&] { return std::make_shared<MoERoutedExpertPlacementPlan>(*plan); };
        auto unresolved = changed();
        unresolved->routed_tiers.back().resolved_live_experts_per_layer.clear();
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, unresolved), std::logic_error);
        auto prequota = changed();
        prequota->placements.clear();
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, prequota), std::logic_error);
        auto incomplete = changed();
        incomplete->routed_tiers.back().resolved_live_experts_per_layer = {0};
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, incomplete), std::logic_error);
        auto later = changed();
        later->routed_tiers.front().resolved_live_experts_per_layer = {2, 1};
        later->routed_tiers.back().resolved_live_experts_per_layer = {0, 1};
        later->placements.back().routed_expert_tier = {0, 1};
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, later), std::logic_error);
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, later,
            ImmutableRowAllocatorInventory{lookup_owners}), std::logic_error);
        auto contradicted = changed();
        contradicted->placements.back().routed_expert_tier = {0, 1};
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, contradicted), std::logic_error);
        auto shared = changed();
        shared->shared_expert_domain = shared->routed_tiers.back().domain;
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, shared), std::logic_error);
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(rows, shared,
            ImmutableRowAllocatorInventory{lookup_owners}), std::logic_error);

        // Absence from the admitted compute list is not permission to drop
        // actual weight claims, a different CPU's claims or another rank's rows.
        auto foreign = rows;
        foreign.front().identity.device = DeviceId{DeviceType::CPU, 1};
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(foreign, plan), std::logic_error);
        foreign = rows;
        foreign.front().identity.world_rank = 1;
        EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(foreign, plan), std::logic_error);
        for (const auto owner : {Owner::RoutedExpertWeights, Owner::PrimaryModelWeights})
        {
            auto missing_weights = rows;
            missing_weights.push_back({.identity = cpu, .owner = owner, .planned_new_bytes = 4096});
            EXPECT_THROW(requireQwen4ExpCPUStorageMaterialization(missing_weights, plan), std::logic_error);
        }
        rows[0].materialized_new_bytes = rows[0].committed_new_bytes = 64;
        rows[1].materialized_new_bytes = rows[1].committed_new_bytes = 128;
        EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(rows, later));
        EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(rows, shared));
        EXPECT_NO_THROW(requireQwen4ExpCPUStorageMaterialization(rows, plan));
    }
}

/** @test Explicit depth3 source families do not reduce the unchanged full15 default. */
TEST(Qwen4ExpSourceAdmissionCases, ExplicitRetainedThreeFamilyPreservesFullDefaultAndSourceProvenance)
{
    const auto &bounded = qwen4ExpCUDAFixedHeadFourExpertRetained3SourceAdmissionCases();
    ASSERT_EQ(bounded.size(), 4u);
    const auto original = qwen4ExpCUDAFixedHeadEightExpertSourceSeed();
    const std::array seeds{original};
    const auto full = qwen4ExpExpandSourceAdmissionMTP(seeds);
    std::set<std::string> names;
    for (std::size_t i = 0; i < bounded.size(); ++i)
    {
        const auto &cell = bounded[i];
        const bool active = cell.mtp != ModelParityMTP::Off;
        EXPECT_EQ(cell.mtp, kQwen4ExpSourceAdmissionMTPPolicies[i]);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, active ? 3 : 0);
        EXPECT_EQ(full[i].retained_mtp_draft_capacity, active ? 15 : 0);
        EXPECT_NE(cell.testName().find("GpuExperts4_RetainedMTP3"), std::string::npos);
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_TRUE(names.insert(full[i].testName()).second);
        EXPECT_LT(cell.testName().size(), 256u);
        EXPECT_EQ(cell.model.model_path, original.model.model_path);
        EXPECT_EQ(cell.model.maximum_mtp_draft_depth,
            active && original.model.maximum_mtp_draft_depth == 0 ? 3 : original.model.maximum_mtp_draft_depth);
        EXPECT_EQ(full[i].model.maximum_mtp_draft_depth,
            active && original.model.maximum_mtp_draft_depth == 0 ? 15 : original.model.maximum_mtp_draft_depth);
        EXPECT_EQ(cell.model.reference_directory, original.model.reference_directory);
        EXPECT_EQ(cell.ngram_table_policy, original.ngram_table_policy);
        EXPECT_EQ(cell.topology.expert_overlay_plan->routed_tiers.front().max_experts_per_layer, 4);
        EXPECT_EQ(full[i].topology.expert_overlay_plan->routed_tiers.front().max_experts_per_layer, 8);
        const auto config = qwen4ExpSourceAdmissionConfig(cell, cell.model.model_path, 0);
        EXPECT_EQ(config.mtp.enabled, active);
        EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, active ? 3 : 0);
        if (active)
        {
            ASSERT_TRUE(config.shared_predictor);
            EXPECT_EQ(config.shared_predictor->path(), kPredictorModelPath);
            EXPECT_EQ(config.mtp.draft_tokens, cell.mtp == ModelParityMTP::Depth1 ? 1 : 3);
            EXPECT_EQ(config.mtp.verify_mode, OrchestrationConfig::defaults().mtp.verify_mode);
            EXPECT_FALSE(config.deterministic);
            if (cell.usesDynamicMTPDepth())
            {
                EXPECT_EQ(config.mtp.depth_policy.max_depth, 3);
                EXPECT_EQ(config.mtp.depth_policy.initial_depth, 0);
            }
        }
        else EXPECT_EQ(config.shared_predictor, original.model.shared_predictor);
    }
    Qwen4ExpMTPAdmissionCase paired{.main = original,
        .policy = ModelParityMTP::DynamicDepth, .retained_depth = 3};
    EXPECT_EQ(paired.requestedDepth(), 3);
    for (const int invalid : {0, -1, 16})
    {
        paired.retained_depth = invalid;
        EXPECT_THROW((void)paired.sourceAdmissionConfig(Qwen4ExpMTPAdmissionExecution::Active,
            original.model.model_path, 0), std::invalid_argument);
    }
    paired.policy = ModelParityMTP::Depth3;
    paired.retained_depth = 1;
    EXPECT_THROW((void)paired.sourceAdmissionConfig(Qwen4ExpMTPAdmissionExecution::Active,
        original.model.model_path, 0), std::invalid_argument);
    EXPECT_THROW((void)qwen4ExpExpandSourceAdmissionMTP(seeds,
        static_cast<Qwen4ExpSourceMTPEnvelope>(255)), std::invalid_argument);
}

/** @test A separate layer split preserves original full15 inputs and every table/expert policy. */
TEST(Qwen4ExpSourceAdmissionCases, ExplicitFifteenLayerCUDAHostFamilyPreservesOriginalFull15)
{
    const auto originals = expandModelParityDefinition(
        qwen4ExpComposedSourceAdmissionDefinition(Qwen4ExpParityTopology::CUDA2CPU2TPPP));
    const auto &fitting = qwen4ExpCUDA2CPU2FifteenLayerSourceAdmissionCases();
    ASSERT_EQ(originals.size(), 4u);
    ASSERT_EQ(fitting.size(), originals.size());
    const auto full_original = qwen4ExpExpandSourceAdmissionMTP(originals);
    const auto full_fitting = qwen4ExpExpandSourceAdmissionMTP(fitting);
    std::set<std::string> names;
    for (std::size_t i = 0; i < full_fitting.size(); ++i)
    {
        const auto &cell = full_fitting[i];
        const auto &original = full_original[i];
        SCOPED_TRACE(cell.testName());
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_TRUE(names.insert(original.testName()).second);
        EXPECT_EQ(cell.model.model_path, original.model.model_path);
        EXPECT_EQ(cell.model.shared_predictor, original.model.shared_predictor);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, original.retained_mtp_draft_capacity);
        EXPECT_EQ(cell.mtp, original.mtp);
        EXPECT_EQ(cell.ngram_table_policy, original.ngram_table_policy);
        EXPECT_EQ(cell.expert_overlay, original.expert_overlay);
        EXPECT_EQ(cell.topology.pipeline_stage_sizes, original.topology.pipeline_stage_sizes);
        EXPECT_EQ(cell.topology.pipeline_tensor_parallel_collectives, original.topology.pipeline_tensor_parallel_collectives);
        const auto request = qwen4ExpSourceAdmissionConfig(cell, cell.model.model_path, 0);
        const auto original_request = qwen4ExpSourceAdmissionConfig(original, original.model.model_path, 0);
        expectSourcePipeline(request, cell);
        ASSERT_EQ(request.pp_stage_definitions.size(), 2u);
        EXPECT_EQ(request.pp_stage_definitions[0].last_layer, 14);
        EXPECT_EQ(request.pp_stage_definitions[1].first_layer, 15);
        EXPECT_EQ(request.pp_stage_definitions[1].last_layer, 47);
        EXPECT_EQ(original_request.pp_stage_definitions[0].last_layer, 15);
        EXPECT_EQ(original_request.pp_stage_definitions[1].first_layer, 16);
        EXPECT_EQ(request.ngram_table_policy, original_request.ngram_table_policy);
        EXPECT_EQ(request.moe_rebalance.mode, original_request.moe_rebalance.mode);
        EXPECT_EQ(request.mtp.graph_capacity_draft_tokens, cell.mtp == ModelParityMTP::Off ? 0 : 15);
        if (cell.mtp != ModelParityMTP::Off)
        {
            EXPECT_EQ(resolveMTPRetainedTargetQueryRows(request.mtp), 16);
            EXPECT_EQ(request.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
            EXPECT_FALSE(request.deterministic);
        }
    }
    // The public CLI projection owns the same explicit layer split before any
    // backend initialization, source loading or physical-capacity selection.
    for (const auto &cell : fitting)
    {
        const auto parsed = sourcePublicConfig(cell);
        expectSourcePipeline(parsed, cell);
        EXPECT_EQ(parsed.pp_stage_definitions[0].last_layer, 14);
        EXPECT_EQ(parsed.pp_stage_definitions[1].first_layer, 15);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
    }
}

/** @test Bounded capacity controls preserve the original larger positive declaration. */
TEST(Qwen4ExpSourceAdmissionCases, ExplicitFixedHeadCapacityControlsKeepOriginalPolicies)
{
    const auto original = qwen4ExpCUDAFixedHeadEightExpertSourceSeed();
    const std::array seeds{original};
    const auto positives = qwen4ExpExpandSourceAdmissionMTP(seeds);
    const auto &negatives = qwen4ExpCUDAFixedHeadCapacityRejectionCases();
    ASSERT_EQ(positives.size(), 4u);
    ASSERT_EQ(negatives.size(), 4u);
    std::set<std::string> names;
    for (std::size_t i = 0; i < positives.size(); ++i)
    {
        const auto &positive = positives[i];
        const auto &negative = negatives[i];
        EXPECT_TRUE(names.insert(positive.testName()).second);
        EXPECT_TRUE(names.insert(negative.testName()).second);
        EXPECT_LT(negative.testName().size(), 256u);
        EXPECT_EQ(negative.cell.mtp, kQwen4ExpSourceAdmissionMTPPolicies[i]);
        EXPECT_EQ(positive.mtp, negative.cell.mtp);
        EXPECT_EQ(positive.retained_mtp_draft_capacity, negative.cell.retained_mtp_draft_capacity);
        EXPECT_EQ(positive.retained_mtp_draft_capacity, positive.mtp == ModelParityMTP::Off ? 0 : 15);
        EXPECT_EQ(negative.cell.model.model_path, positive.model.model_path);
        EXPECT_EQ(negative.cell.model.shared_predictor, positive.model.shared_predictor);
        EXPECT_EQ(negative.cell.model.maximum_mtp_draft_depth, positive.model.maximum_mtp_draft_depth);
        EXPECT_EQ(negative.cell.ngram_table_policy, positive.ngram_table_policy);
        ASSERT_TRUE(negative.cell.expert_overlay);
        ASSERT_TRUE(positive.expert_overlay);
        EXPECT_EQ(negative.cell.expert_overlay->movement, positive.expert_overlay->movement);
        EXPECT_EQ(negative.cell.expert_overlay->owner_order, positive.expert_overlay->owner_order);
        EXPECT_EQ(negative.cell.topology.expert_overlay_plan->routed_tiers.size(),
            positive.topology.expert_overlay_plan->routed_tiers.size());
        EXPECT_EQ(negative.gpu_memory_cap_mb, 13u * 1024u);
        const auto p = qwen4ExpSourceAdmissionCapacityConfig(positive, positive.model.model_path, 0);
        const auto n = qwen4ExpSourceAdmissionCapacityConfig(negative.cell, negative.cell.model.model_path, 0,
            negative.gpu_memory_cap_mb);
        EXPECT_FALSE(p.max_gpu_memory_mb.has_value());
        ASSERT_TRUE(n.max_gpu_memory_mb.has_value());
        EXPECT_EQ(*n.max_gpu_memory_mb, 13312u);
        EXPECT_EQ(n.memoryLimitBytes(DeviceType::CUDA), std::size_t{13} * 1024 * 1024 * 1024);
        EXPECT_EQ(p.ngram_table_policy, n.ngram_table_policy);
        EXPECT_EQ(p.mtp.graph_capacity_draft_tokens, n.mtp.graph_capacity_draft_tokens);
        EXPECT_EQ(p.mtp.depth_policy.initial_depth, n.mtp.depth_policy.initial_depth);
        EXPECT_EQ(n.moe_routed_expert_plan->routed_tiers.front().max_experts_per_layer, 8);
        ASSERT_EQ(positive.topology.participants.size(), negative.cell.topology.participants.size());
        for (std::size_t participant = 0; participant < positive.topology.participants.size(); ++participant)
            EXPECT_EQ(positive.topology.participants[participant].address,
                negative.cell.topology.participants[participant].address);
    }
    EXPECT_EQ(original.topology.expert_overlay_plan->routed_tiers.front().max_experts_per_layer, 8);
    EXPECT_THROW((void)qwen4ExpSourceAdmissionCapacityConfig(original, original.model.model_path, 0, 0),
        std::invalid_argument);
}

/** @test The physical shortage has a distinct capped regression, retaining both original expert policies and all active MTP modes. */
TEST(Qwen4ExpSourceAdmissionCases, ROCmReplicatedCapacityControlsRetainAllOriginalOwners)
{
    const auto original = qwen4ExpExpandSourceAdmissionMTP(qwen4ExpROCm4OverlayPolicyProductionCases());
    const auto &controls = qwen4ExpROCm4ReplicatedCapacityRejectionCases();
    ASSERT_EQ(controls.size(), 6u);
    std::set<std::pair<RoutedExpertComputePolicy, ModelParityMTP>> axes;
    std::set<std::string> names;
    for (const auto &control : controls)
    {
        SCOPED_TRACE(control.testName());
        auto restored = control.cell;
        const auto suffix = restored.topology.test_id.rfind("_Capacity32GiB");
        ASSERT_NE(suffix, std::string::npos);
        restored.topology.test_id.resize(suffix);
        const auto positive = std::find_if(original.begin(), original.end(),
            [&](const auto &cell) { return cell.testName() == restored.testName(); });
        ASSERT_NE(positive, original.end());
        EXPECT_TRUE(names.insert(control.testName()).second);
        EXPECT_LT(control.testName().size(), 256u);
        EXPECT_NE(control.cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(control.cell.retained_mtp_draft_capacity, 15);
        EXPECT_EQ(control.cell.model.model_path, positive->model.model_path);
        EXPECT_EQ(control.cell.model.shared_predictor, positive->model.shared_predictor);
        EXPECT_EQ(control.cell.ngram_table_policy, positive->ngram_table_policy);
        EXPECT_EQ(control.boundary, Qwen4ExpSourceCapacityBoundary::RoutedExpertResidency);
        EXPECT_EQ(control.gpu_memory_cap_mb, 32768u);
        const auto p = qwen4ExpSourceAdmissionCapacityConfig(*positive, positive->model.model_path, 0);
        const auto n = qwen4ExpSourceAdmissionCapacityConfig(control.cell, control.cell.model.model_path, 0,
            control.gpu_memory_cap_mb);
        EXPECT_FALSE(p.max_gpu_memory_mb.has_value());
        EXPECT_EQ(n.memoryLimitBytes(DeviceType::ROCm), std::size_t{32} * 1024 * 1024 * 1024);
        EXPECT_EQ(n.mtp.graph_capacity_draft_tokens, p.mtp.graph_capacity_draft_tokens);
        EXPECT_EQ(n.mtp.depth_policy.initial_depth, p.mtp.depth_policy.initial_depth);
        const auto &plan = *n.moe_routed_expert_plan;
        ASSERT_EQ(plan.domains.size(), 1u);
        ASSERT_EQ(plan.routed_tiers.size(), 1u);
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer, 0);
        EXPECT_EQ(plan.domains.front().participants, p.moe_routed_expert_plan->domains.front().participants);
        EXPECT_EQ(plan.continuation_domain_spec.effectiveDensePolicy(),
            DenseParallelPolicy::PrefillTensorParallelDecodeReplicated);
        EXPECT_TRUE(axes.emplace(plan.domains.front().routed_compute_policy, control.cell.mtp).second);
    }
    EXPECT_EQ(axes.size(), 6u);
}

/** @test A capacity control cannot pass from a different boundary or an unrelated initialization failure. */
TEST(Qwen4ExpSourceAdmissionCases, CapacityDiagnosticRequiresItsAuthoredBoundary)
{
    const std::string table = "Explicit fixed head native ranges exceed complete frozen PMA capacity; "
        "immutable_table_storage=17000000000 deficit_bytes=3000000000";
    const std::string experts = "ExpertOverlay has no resident captured-prefill shape with complete expert coverage "
        "under the physical BOM: ExpertOverlay capacity step threw: 'overlay-candidate.complete-bom': "
        "ExpertOverlay tier 'all_gpu' cannot admit its fallback remainder for layer 40: "
        "requires_additional_bytes=2048000 remaining_bytes=842040 fixed_bytes=11130185704 "
        "staging_bytes=536870624 live_expert_bytes=22324838400";
    EXPECT_TRUE(qwen4ExpSourceCapacityDiagnosticMatches(Qwen4ExpSourceCapacityBoundary::FixedHeadTable, table));
    EXPECT_TRUE(qwen4ExpSourceCapacityDiagnosticMatches(Qwen4ExpSourceCapacityBoundary::RoutedExpertResidency, experts));
    EXPECT_FALSE(qwen4ExpSourceCapacityDiagnosticMatches(Qwen4ExpSourceCapacityBoundary::FixedHeadTable, experts));
    EXPECT_FALSE(qwen4ExpSourceCapacityDiagnosticMatches(Qwen4ExpSourceCapacityBoundary::RoutedExpertResidency, table));
    for (const auto *error : {"", "GPU allocation failed", "RCCL communicator creation failed",
            "ExpertOverlay fixed-bom schema is invalid", "captured prefill is unsupported"})
        EXPECT_FALSE(qwen4ExpSourceCapacityDiagnosticMatches(Qwen4ExpSourceCapacityBoundary::RoutedExpertResidency, error));
    EXPECT_THROW(qwen4ExpSourceCapacityDiagnosticMatches(static_cast<Qwen4ExpSourceCapacityBoundary>(99), experts),
        std::invalid_argument);
}

/**
 * @test Every existing source-admission identity retains all four MTP policies,
 * including candidate and precise-rejection families without reclassification.
 */
TEST(Qwen4ExpSourceAdmissionCases, FullSourceMatrixHasTotalFourPolicyMTPDimension)
{
    const std::array families{
        &qwen4ExpProductionCases(), &qwen4ExpHostAdaptiveProductionCases(),
        &qwen4ExpFixedHeadProductionCases(), &qwen4ExpGPUAdaptiveSourceAdmissionSeeds(),
        &qwen4ExpFullROCmTP4ProductionCases(), &qwen4ExpROCm4OverlayPolicyProductionCases(),
        &qwen4ExpExtendedSourceAdmissionCases(), &qwen4ExpComposedCandidateSourceAdmissionCases(),
        &qwen4ExpRejectedSourceAdmissionCases(), &qwen4ExpNgramAdmissionCases(),
        &qwen4ExpCUDA2CPU2FifteenLayerSourceAdmissionCases()};
    const std::set policies{ModelParityMTP::Off, ModelParityMTP::Depth1,
        ModelParityMTP::Depth3, ModelParityMTP::DynamicDepth};
    std::set<std::string> names;
    size_t seeds = 0, active = 0, off = 0;
    for (const auto *family : families)
    {
        const auto expanded = qwen4ExpExpandSourceAdmissionMTP(*family);
        ASSERT_EQ(expanded.size(), family->size() * policies.size());
        seeds += family->size();
        for (size_t index = 0; index < family->size(); ++index)
        {
            const auto &main = family->at(index);
            std::set<ModelParityMTP> observed;
            for (size_t policy = 0; policy < policies.size(); ++policy)
            {
                const auto &cell = expanded[index * policies.size() + policy];
                SCOPED_TRACE(cell.testName());
                ASSERT_TRUE(names.insert(cell.testName()).second);
                EXPECT_LT(cell.testName().size(), 256u)
                    << "The actual source-admission directory component must fit NAME_MAX";
                observed.insert(cell.mtp);
                EXPECT_EQ(cell.model.model_path, main.model.model_path);
                EXPECT_EQ(cell.model.reference_directory, main.model.reference_directory);
                EXPECT_EQ(cell.model.maximum_mtp_draft_depth,
                    cell.mtp != ModelParityMTP::Off && main.model.maximum_mtp_draft_depth == 0
                        ? cell.retained_mtp_draft_capacity : main.model.maximum_mtp_draft_depth);
                EXPECT_EQ(cell.model.token_ids, main.model.token_ids);
                EXPECT_EQ(cell.topology.test_id, main.topology.test_id);
                EXPECT_EQ(cell.topology.pipeline_stage_sizes, main.topology.pipeline_stage_sizes);
                EXPECT_EQ(cell.topology.pipeline_weights, main.topology.pipeline_weights);
                ASSERT_EQ(cell.topology.participants.size(), main.topology.participants.size());
                for (size_t member = 0; member < main.topology.participants.size(); ++member)
                {
                    EXPECT_EQ(cell.topology.participants[member].address, main.topology.participants[member].address);
                    EXPECT_EQ(cell.topology.participants[member].world_rank, main.topology.participants[member].world_rank);
                }
                EXPECT_EQ(cell.ngram_table_policy, main.ngram_table_policy);
                EXPECT_EQ(cell.expert_overlay, main.expert_overlay);
                EXPECT_EQ(cell.activation_precision, main.activation_precision);
                EXPECT_EQ(cell.kv_cache_precision, main.kv_cache_precision);
                EXPECT_EQ(cell.e2e_certification, main.e2e_certification);
                EXPECT_EQ(cell.paired_workload.has_value(), main.paired_workload.has_value());
                const auto config = qwen4ExpSourceAdmissionConfig(cell, cell.model.model_path, 0);
                EXPECT_EQ(config.ngram_table_policy, main.makeOrchestrationConfig(main.model.model_path, 0).ngram_table_policy);
                if (cell.mtp == ModelParityMTP::Off)
                {
                    ++off;
                    EXPECT_EQ(cell.testName(), main.testName());
                    EXPECT_EQ(cell.model.shared_predictor, main.model.shared_predictor);
                    EXPECT_EQ(cell.retained_mtp_draft_capacity, 0);
                    EXPECT_FALSE(config.mtp.enabled);
                    const auto original = main.makeOrchestrationConfig(main.model.model_path, 0);
                    EXPECT_EQ(config.deterministic, original.deterministic);
                    EXPECT_EQ(config.mtp.verify_mode, original.mtp.verify_mode);
                }
                else
                {
                    ++active;
                    ASSERT_TRUE(cell.model.shared_predictor);
                    EXPECT_EQ(config.shared_predictor, cell.model.shared_predictor);
                    EXPECT_TRUE(config.mtp.enabled);
                    const auto production = OrchestrationConfig::defaults();
                    EXPECT_FALSE(config.deterministic);
                    EXPECT_EQ(config.deterministic, production.deterministic);
                    EXPECT_EQ(config.mtp.verify_mode, production.mtp.verify_mode);
                    EXPECT_EQ(config.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
                    const MTPDepthPolicyConfig cadence;
                    EXPECT_EQ(config.mtp.depth_policy.window_size, cadence.window_size);
                    EXPECT_EQ(config.mtp.depth_policy.min_samples, cadence.min_samples);
                    EXPECT_EQ(config.mtp.depth_policy.cooldown_steps, cadence.cooldown_steps);
                    EXPECT_EQ(config.mtp.depth_policy.promote_consecutive_windows, cadence.promote_consecutive_windows);
                    EXPECT_EQ(config.mtp.depth_policy.use_generated_policy, cadence.use_generated_policy);
                    EXPECT_EQ(config.mtp.depth_policy.backend, MTPDepthPolicyBackend::Any);
                    EXPECT_EQ(config.mtp.depth_policy.model_class, MTPDepthPolicyModelClass::Any);
                    EXPECT_EQ(config.mtp.draft_tokens, cell.requestedMTPDraftDepth());
                    EXPECT_EQ(config.mtp.graph_capacity_draft_tokens, 15);
                    EXPECT_EQ(resolveMTPRetainedTargetQueryRows(config.mtp), 16)
                        << "All active policies retain the same one-request depth15 physical shape";
                    EXPECT_EQ(config.mtp.depth_policy.initial_depth, 0);
                    EXPECT_EQ(config.mtp.depth_policy.mode, cell.mtp == ModelParityMTP::DynamicDepth
                        ? MTPDepthPolicyMode::Dynamic : MTPDepthPolicyMode::Fixed);
                    if (cell.mtp == ModelParityMTP::DynamicDepth)
                    {
                        EXPECT_EQ(config.mtp.depth_policy.min_depth, 1);
                        EXPECT_EQ(config.mtp.depth_policy.max_depth, 15);
                    }
                }
            }
            EXPECT_EQ(observed, policies);
        }
    }
    EXPECT_EQ(seeds, 167u);
    EXPECT_EQ(off, 167u);
    EXPECT_EQ(active, 501u);
    EXPECT_EQ(names.size(), 668u);
}

/** @test Missing paired sources and invalid dimension inputs fail without any source I/O. */
TEST(Qwen4ExpSourceAdmissionCases, SourceMTPDimensionRejectsMissingPairAndInvalidSeeds)
{
    const auto main = qwen4ExpProductionCases().front();
    const std::array seeds{main};
    const auto expanded = qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(expanded.size(), 4u);
    auto missing = expanded[1];
    missing.model.shared_predictor.reset();
    EXPECT_THROW(qwen4ExpSourceAdmissionConfig(missing, main.model.model_path, 0), std::invalid_argument);
    auto envelope = expanded[1];
    envelope.retained_mtp_draft_capacity = 1;
    EXPECT_THROW(qwen4ExpSourceAdmissionConfig(envelope, main.model.model_path, 0), std::invalid_argument);
    auto unknown = expanded[1];
    for (const auto omitted : {ModelParityMTP::Depth2, ModelParityMTP::Depth5, ModelParityMTP::Depth15})
    {
        unknown.mtp = omitted;
        EXPECT_THROW(qwen4ExpSourceAdmissionConfig(unknown, main.model.model_path, 0), std::invalid_argument);
    }
    unknown.mtp = static_cast<ModelParityMTP>(255);
    EXPECT_THROW(qwen4ExpSourceAdmissionConfig(unknown, main.model.model_path, 0), std::invalid_argument);
    const std::array active_seed{expanded[1]};
    EXPECT_THROW(qwen4ExpExpandSourceAdmissionMTP(active_seed), std::invalid_argument);
    auto retained = main;
    retained.retained_mtp_draft_capacity = 1;
    const std::array retained_seed{retained};
    EXPECT_THROW(qwen4ExpExpandSourceAdmissionMTP(retained_seed), std::invalid_argument);
}

/** @test Full source projection does not silently change existing A1 diagnostic semantics. */
TEST(Qwen4ExpSourceAdmissionCases, SourceMTPProjectionPreservesLegacyDiagnosticAndExplicitInactiveControl)
{
    auto main = qwen4ExpProductionCases().front();
    main.ngram_table_policy = ImmutableRowTablePolicy{.storage = ImmutableRowStoragePreference::RAM};
    const Qwen4ExpMTPAdmissionCase paired{.main = main, .policy = ModelParityMTP::DynamicDepth, .retained_depth = 15};
    const auto diagnostic = paired.config(Qwen4ExpMTPAdmissionExecution::Active);
    EXPECT_EQ(diagnostic.mtp.depth_policy.initial_depth, 15);
    EXPECT_TRUE(diagnostic.deterministic);
    EXPECT_EQ(diagnostic.mtp.verify_mode, MTPVerifyMode::Greedy);
    const auto qualification = paired.config(Qwen4ExpMTPAdmissionExecution::Active,
        Qwen4ExpMTPAdmissionPurpose::RuntimeQualification);
    EXPECT_EQ(qualification.mtp.depth_policy.initial_depth, 15);
    EXPECT_FALSE(qualification.deterministic);
    EXPECT_EQ(qualification.mtp.verify_mode, modelParityHttpMTPVerifyMode());
    const auto serving = paired.config(Qwen4ExpMTPAdmissionExecution::Active,
        Qwen4ExpMTPAdmissionPurpose::ProductionServing);
    EXPECT_EQ(serving.mtp.depth_policy.initial_depth, 0);
    EXPECT_FALSE(serving.deterministic);
    EXPECT_EQ(serving.mtp.verify_mode, modelParityHttpMTPVerifyMode());
    const auto source = paired.sourceAdmissionConfig(Qwen4ExpMTPAdmissionExecution::Active, main.model.model_path, 0);
    EXPECT_EQ(source.mtp.depth_policy.initial_depth, 0);
    EXPECT_FALSE(source.deterministic);
    EXPECT_EQ(source.mtp.verify_mode, MTPVerifyMode::SpeculativeSampling);
    const auto inactive = paired.sourceAdmissionConfig(Qwen4ExpMTPAdmissionExecution::Inactive, main.model.model_path, 0);
    EXPECT_FALSE(inactive.mtp.enabled);
    EXPECT_EQ(inactive.mtp.graph_capacity_draft_tokens, 0);
    EXPECT_EQ(inactive.shared_predictor, std::optional{paired.predictor});
    // This separately authored inactive pair is not substituted for legacy Off:
    // a four-file main-only Off case must keep its original source closure.
    EXPECT_EQ(qwen4ExpSourceAdmissionConfig(main, main.model.model_path, 0).shared_predictor,
        main.model.shared_predictor);
}

} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp {
/** @test Canonical source admission cannot treat missing source birth as an observation. */
TEST(Qwen4ExpSourceAdmissionCases, ConsumedNativeSourceRejectsAbsentOriginalPreparation)
{
    const std::optional<::llaminar2::qwen4exp::PreparedImmutableSourceWorkers> unused;
    EXPECT_THROW((void)requireQwen4ExpConsumedPreparedSource(nullptr,unused),std::logic_error);
}
/** @test A real ModelContext retaining a real recorded source is not proof its lookup used it.
 * No worker is acquired/created: this exact early false-positive boundary must
 * reject before the canonical observer can inspect any original native handles.
 */
TEST(Qwen4ExpSourceAdmissionCases, ConsumedNativeSourceRejectsPreboundUnusedOriginalAuthority)
{
    for(const auto device:{DeviceId::cuda(88),DeviceId::rocm(88)}){
        auto context=ModelContext::createForTesting("recorded-unused-source.gguf",nullptr,48,false);
        const auto source=::llaminar2::test::NativeContextPreparationReplayAccess::source<::llaminar2::testing::MockWorkerGPUContext>(device);
        context->bindNativeContextPreparation(source);
        const auto bound=context->nativeContextPreparation();
        ASSERT_EQ(bound,source); // Bound-only control would incorrectly pass.
        const std::optional<::llaminar2::qwen4exp::PreparedImmutableSourceWorkers> unused;
        try{(void)requireQwen4ExpConsumedPreparedSource(bound,unused);FAIL()<<"Unused source certified actual lookup consumption";}
        catch(const std::logic_error& error){EXPECT_NE(std::string(error.what()).find("prebound but not consumed"),std::string::npos);}
    }
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Source-only expansion retains every original GPU/tier/expert/MTP axis without adding topology. */
TEST(Qwen4ExpSourceAdmissionCases, GPUAdaptiveSourceHasTotalExpertPolicyAndMTPProduct)
{
    const auto& seeds=qwen4ExpGPUAdaptiveSourceAdmissionSeeds();
    ASSERT_EQ(seeds.size(),16u);
    const auto cells=qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(cells.size(),64u);
    using Axis=std::tuple<DeviceType,bool,ModelParityExpertMovement,RoutedExpertOwnerOrder,ModelParityMTP>;
    std::set<Axis> axes;std::set<std::string> names;
    for(const auto& cell:cells){
        SCOPED_TRACE(cell.testName());ASSERT_TRUE(cell.expert_overlay);ASSERT_TRUE(cell.ngram_table_policy);
        ASSERT_TRUE(cell.ngram_table_policy->isGPUAdaptive());
        ASSERT_TRUE(cell.topology.expert_overlay_plan);const auto& plan=*cell.topology.expert_overlay_plan;
        ASSERT_EQ(plan.domains.size(),2u);ASSERT_EQ(plan.routed_tiers.size(),2u);
        const auto continuation=std::find_if(plan.domains.begin(),plan.domains.end(),[&](const auto& domain){return domain.name==plan.continuation_domain;});
        ASSERT_NE(continuation,plan.domains.end());ASSERT_EQ(continuation->participants.size(),1u);
        const auto compute=continuation->participants.front();EXPECT_TRUE(compute.isCUDA() || compute.isROCm());
        EXPECT_EQ(compute.device_ordinal,0);EXPECT_EQ(plan.domains.back().participants,(std::vector{GlobalDeviceAddress::cpu(0)}));
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer,8u);
        EXPECT_EQ(cell.topology.participants.size(),2u);EXPECT_EQ(cell.topology.mpi_ranks,1);
        EXPECT_EQ(cell.topology.kind,ModelParityTopologyKind::RankLocalMultiDomain);
        const auto& request=cell.ngram_table_policy->gpu_adaptive;
        const bool bounded=request.maximum_gpu_cache_rows!=0;
        EXPECT_EQ(request.maximum_gpu_cache_rows,bounded?64u:0u);
        EXPECT_EQ(request.gpu_cache_physical_byte_limit,bounded?1u<<20:0u);
        EXPECT_EQ(request.host.maximum_cache_rows,bounded?64u:0u);
        EXPECT_EQ(request.host.cache_physical_byte_limit,bounded?1u<<20:0u);
        EXPECT_TRUE(axes.emplace(compute.device_type,bounded,cell.expert_overlay->movement,cell.expert_overlay->owner_order,cell.mtp).second);
        EXPECT_TRUE(names.insert(cell.testName()).second);EXPECT_LT(cell.testName().size(),256u);
        EXPECT_EQ(cell.model.transformer_layers,48);EXPECT_EQ(cell.model.model_path,kMainModelPath);
        const auto config=qwen4ExpSourceAdmissionConfig(cell,cell.model.model_path,0);
        const auto original=std::find_if(seeds.begin(),seeds.end(),[&](const auto& seed){
            return seed.topology.test_id==cell.topology.test_id && seed.expert_overlay==cell.expert_overlay && seed.ngram_table_policy==cell.ngram_table_policy;
        });
        ASSERT_NE(original,seeds.end());
        const auto authored=original->makeOrchestrationConfig(original->model.model_path,0);
        // Reuse the canonical all-fields policy equality: MTP/source expansion
        // cannot replace the original Qwen MoE maintenance/transfer policy.
        EXPECT_NO_THROW(requireQwen4ExpSourceMaintenancePolicy(authored.moe_rebalance,config.moe_rebalance));
        ASSERT_TRUE(config.moe_routed_expert_plan);
        EXPECT_EQ(config.routed_expert_owner_order,cell.expert_overlay->owner_order);
        EXPECT_EQ(config.moe_routed_expert_plan->residency_policy,
            cell.expert_overlay->movement==ModelParityExpertMovement::Dynamic?RoutedExpertResidencyPolicy::RoutedTierRebalanced:RoutedExpertResidencyPolicy::StaticById);
        EXPECT_EQ(config.mtp.enabled,cell.mtp!=ModelParityMTP::Off);
        EXPECT_EQ(cell.retained_mtp_draft_capacity,cell.mtp==ModelParityMTP::Off?0:15);
    }
    EXPECT_EQ(axes.size(),64u);EXPECT_EQ(names.size(),64u);
}
/** @test Original16 source identities and all4 numerical cases remain byte-identical declarations. */
TEST(Qwen4ExpSourceAdmissionCases, GPUAdaptiveSourcePreservesOriginalSixteenAndNumericalFour)
{
    const auto& numerical=qwen4ExpGPUAdaptiveProductionCases();ASSERT_EQ(numerical.size(),4u);
    std::vector<std::string> before;
    for(const auto& cell:numerical){std::ostringstream json;::llaminar2::test::parity::PrintTo(cell,&json);before.push_back(json.str());}
    const auto legacy=qwen4ExpExpandSourceAdmissionMTP(numerical);
    const auto expanded=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpGPUAdaptiveSourceAdmissionSeeds());
    ASSERT_EQ(legacy.size(),16u);ASSERT_EQ(expanded.size(),64u);
    std::size_t preserved=0;
    for(const auto& original:legacy){
        const auto actual=std::find_if(expanded.begin(),expanded.end(),[&](const auto& cell){return cell.testName()==original.testName();});
        ASSERT_NE(actual,expanded.end());std::ostringstream oldJson,newJson;
        ::llaminar2::test::parity::PrintTo(original,&oldJson);::llaminar2::test::parity::PrintTo(*actual,&newJson);
        EXPECT_EQ(oldJson.str(),newJson.str());++preserved;
    }
    EXPECT_EQ(expanded.size()-preserved,48u);
    ASSERT_EQ(qwen4ExpGPUAdaptiveProductionCases().size(),before.size());
    for(std::size_t i=0;i<before.size();++i){std::ostringstream after;::llaminar2::test::parity::PrintTo(qwen4ExpGPUAdaptiveProductionCases()[i],&after);EXPECT_EQ(before[i],after.str());}
}
/** @test Dynamic MTP depth and required physical expert movement are independent source intents. */
TEST(Qwen4ExpSourceAdmissionCases, GPUAdaptiveDynamicDepthDoesNotCertifyPhysicalExpertMovement)
{
    const auto cells=qwen4ExpExpandSourceAdmissionMTP(qwen4ExpGPUAdaptiveSourceAdmissionSeeds());
    std::size_t movement=0,dynamicDepth=0,both=0;
    for(const auto& cell:cells){
        const bool physical=cell.expert_overlay->movement==ModelParityExpertMovement::Dynamic;
        EXPECT_EQ(cell.requiresPhysicalExpertMovement(),physical);
        const auto config=qwen4ExpSourceAdmissionConfig(cell,cell.model.model_path,0);
        // Source configuration retains the real policy. Numerical movement
        // journal/exercise observations remain a separate inference gate.
        if(physical){++movement;EXPECT_EQ(config.moe_rebalance.mode,MoERebalanceRuntimeMode::Dynamic);}
        if(cell.mtp==ModelParityMTP::DynamicDepth){++dynamicDepth;EXPECT_EQ(config.mtp.depth_policy.initial_depth,0);if(physical)++both;}
    }
    EXPECT_EQ(movement,32u);EXPECT_EQ(dynamicDepth,16u);EXPECT_EQ(both,8u);
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test The sole new source seed keeps original world2 socket roles and the complete public parser projection. */
TEST(Qwen4ExpSourceAdmissionCases, CPUJointWorld2StaticRAMOffUsesExactSocketRoles) {
    const auto& cases=qwen4ExpCPUJointWorld2SourceAdmissionCases();ASSERT_EQ(cases.size(),1u);
    const auto& cell=cases.front();EXPECT_EQ(cell.mtp,ModelParityMTP::Off);
    EXPECT_EQ(cell.retained_mtp_draft_capacity,0);EXPECT_FALSE(cell.model.shared_predictor);
    EXPECT_EQ(cell.activation_precision,ActivationPrecision::FP32);
    EXPECT_EQ(cell.kv_cache_precision,KVCachePrecision::FP16);
    ASSERT_TRUE(cell.ngram_table_policy);EXPECT_EQ(cell.ngram_table_policy->storage,ImmutableRowStoragePreference::RAM);
    EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
    EXPECT_EQ(cell.topology.kind,ModelParityTopologyKind::NodeMultiDomain);EXPECT_EQ(cell.topology.mpi_ranks,2);
    ASSERT_EQ(cell.topology.participants.size(),3u);
    EXPECT_EQ(cell.topology.participants[0],(ModelParityParticipant{GlobalDeviceAddress::rocm(0),0}));
    EXPECT_EQ(cell.topology.participants[1],(ModelParityParticipant{GlobalDeviceAddress::cpu(0),0}));
    EXPECT_EQ(cell.topology.participants[2],(ModelParityParticipant{GlobalDeviceAddress::cpu(1),1}));
    const auto parsed=sourcePublicConfig(cell);
    EXPECT_EQ(parsed.mpi_procs,2);EXPECT_EQ(parsed.moe_rebalance.mode,MoERebalanceRuntimeMode::Off);
    ASSERT_TRUE(parsed.moe_routed_expert_plan);ASSERT_EQ(parsed.moe_routed_expert_plan->domains.size(),2u);
    EXPECT_TRUE(MoEExpertOwnerMap::usesCPUJointFragments(*parsed.moe_routed_expert_plan));
    EXPECT_EQ(parsed.moe_routed_expert_plan->routed_tiers.front().max_experts_per_layer,64);
    EXPECT_EQ(parsed.moe_routed_expert_plan->routed_tiers.back().max_experts_per_layer,448);
    const auto& cpu=parsed.moe_routed_expert_plan->domains.back();
    EXPECT_EQ(cpu.scope,ExecutionDomainScope::NODE_LOCAL);EXPECT_EQ(cpu.backend,CollectiveBackendType::UPI);
    EXPECT_EQ(cpu.routed_compute_policy,RoutedExpertComputePolicy::TensorSharded);
    EXPECT_EQ(cpu.world_ranks,(std::vector<int>{0,1}));
    EXPECT_EQ(cpu.participants,(std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cpu(0),GlobalDeviceAddress::cpu(1)}));
    EXPECT_TRUE(qwen4ExpCPUJointSourceHasContinuation(cell,0));EXPECT_FALSE(qwen4ExpCPUJointSourceHasContinuation(cell,1));
    EXPECT_THROW(qwen4ExpCPUJointSourceHasContinuation(cell,2),std::invalid_argument);
    for(int rank=0;rank<2;++rank) {
        const auto native=qwen4ExpSourceAdmissionConfig(cell,cell.model.model_path,rank);
        ASSERT_TRUE(native.moe_routed_expert_plan);
        EXPECT_EQ(native.moe_routed_expert_plan->domains.back().toExecutionDomainDefinition().toString(),cpu.toExecutionDomainDefinition().toString());
        EXPECT_EQ(native.shared_predictor,parsed.shared_predictor);
        EXPECT_EQ(native.ngram_table_policy,parsed.ngram_table_policy);
        EXPECT_FALSE(native.mtp.enabled);EXPECT_EQ(native.mtp.graph_capacity_draft_tokens,0);
        EXPECT_TRUE(native.prefix_cache.enabled); // Existing canonical source policy, not prefix-Off observer equivalence.
    }
    // Original CPU2TP remains rank-local; never relabel it as one rank per socket.
    const auto local=qwen4ExpProductionTopology(Qwen4ExpParityTopology::CPU2TP);
    EXPECT_EQ(local.mpi_ranks,1);ASSERT_EQ(local.participants.size(),2u);
    EXPECT_EQ(local.participants[0].world_rank,0);EXPECT_EQ(local.participants[1].world_rank,0);
}
} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp {
/** @test Active source requests retain the original world2 source/socket authority for every rank. */
TEST(Qwen4ExpSourceAdmissionCases, CPUJointWorld2MTPPoliciesKeepOriginalSourceAndSocketRoles) {
    const auto &seed=qwen4ExpCPUJointWorld2SourceAdmissionCases().front();
    const auto &cases=qwen4ExpCPUJointWorld2MTPSourceAdmissionCases();ASSERT_EQ(cases.size(),4u);
    std::set<ModelParityMTP> policies;
    for(const auto &cell:cases) {
        EXPECT_TRUE(policies.insert(cell.mtp).second);
        EXPECT_EQ(cell.topology.test_id,seed.topology.test_id);
        EXPECT_EQ(cell.topology.participants,seed.topology.participants);
        EXPECT_EQ(cell.topology.expert_overlay_plan,seed.topology.expert_overlay_plan);
        EXPECT_EQ(cell.expert_overlay,seed.expert_overlay);
        EXPECT_EQ(cell.activation_precision,seed.activation_precision);EXPECT_EQ(cell.kv_cache_precision,seed.kv_cache_precision);
        const auto parsed=sourcePublicConfig(cell);
        EXPECT_EQ(parsed.mpi_procs,2);
        EXPECT_EQ(cell.topology.mpi_ranks,2);
        for(int rank=0;rank<2;++rank) {
            const auto config=qwen4ExpSourceAdmissionConfig(cell,cell.model.model_path,rank);
            // The launcher owns world size; in-process source setup discovers
            // that existing MPI world rather than launching another one.
            EXPECT_EQ(config.mpi_procs,0);
            EXPECT_EQ(config.moe_rebalance.mode,MoERebalanceRuntimeMode::Off);
            EXPECT_EQ(config.shared_predictor,parsed.shared_predictor);
            EXPECT_EQ(config.mtp.enabled,parsed.mtp.enabled);
            ASSERT_TRUE(config.moe_routed_expert_plan);
            EXPECT_TRUE(MoEExpertOwnerMap::usesCPUJointFragments(*config.moe_routed_expert_plan));
            EXPECT_EQ(config.moe_routed_expert_plan->domains.back().world_ranks,(std::vector<int>{0,1}));
            if(cell.mtp==ModelParityMTP::Off) {
                EXPECT_FALSE(config.mtp.enabled);EXPECT_EQ(cell.retained_mtp_draft_capacity,0);
            } else {
                EXPECT_TRUE(config.mtp.enabled);EXPECT_TRUE(config.shared_predictor);
                EXPECT_EQ(config.mtp.graph_capacity_draft_tokens,3);
                EXPECT_EQ(config.mtp.depth_policy.mode,cell.mtp==ModelParityMTP::DynamicDepth?MTPDepthPolicyMode::Dynamic:MTPDepthPolicyMode::Fixed);
            }
        }
    }
    EXPECT_EQ(policies,(std::set<ModelParityMTP>{ModelParityMTP::Off,ModelParityMTP::Depth1,ModelParityMTP::Depth3,ModelParityMTP::DynamicDepth}));
    const auto &exact=qwen4ExpCPUJointWorld2MTPExactCases();ASSERT_EQ(exact.size(),3u);
    for(const auto &cell:exact) {
        EXPECT_NE(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.model.maximum_mtp_draft_depth,3);
        EXPECT_EQ(cell.inventory_role,ModelParityInventoryRole::NumericalDiagnostic);
        EXPECT_EQ(cell.collective_evidence_source,ParityCollectiveEvidenceSource::PostCollectiveSnapshot);
        EXPECT_TRUE(cell.thresholds.allreduce_stages.empty());
        EXPECT_EQ(cell.topology.participants,seed.topology.participants);
        const auto config=cell.toTestConfig();EXPECT_NE(config.mtp_expectation,ParityMTPExpectation::Disabled);
        EXPECT_EQ(config.collective_evidence_source,ParityCollectiveEvidenceSource::PostCollectiveSnapshot);
        EXPECT_TRUE(config.thresholds.allreduce_stages.empty());
    }
}
}

namespace llaminar2::test::parity::qwen4exp
{
/** @test Original CPU rank affinity cannot hide an explicitly owned GPU on another socket. */
TEST(Qwen4ExpSourceAdmissionCases, CPUJointExactStartupPreservesHostWideGPUAndExactCPUSockets)
{
    for (const auto &cell : qwen4ExpCPUJointWorld2MTPSourceAdmissionCases())
    {
        for (int rank = 0; rank < cell.topology.mpi_ranks; ++rank)
        {
            const auto startup = cell.makeOrchestrationConfig(kMainModelPath, rank);
            EXPECT_TRUE(RuntimeInitPhase::requiresHostWideAcceleratorVisibility(startup, rank));
            // Physical GPU locality comes from the retained inventory, not its
            // owning rank number or the CPU fragment's original NUMA endpoint.
            auto hardware = sourceEvidenceHardware();
            ASSERT_FALSE(hardware.rocm_devices.empty());
            hardware.rocm_devices.front().numa_node = 1;
            auto gpu = GlobalDeviceAddress::rocm(0);
            gpu.hostname = "fixture-host";
            gpu.numa_node = 1;
            EXPECT_NO_THROW(requireQwen4ExpSourceEndpoint(
                GlobalDeviceAddress::rocm(0), gpu, "fixture-host", hardware));
            auto foreign = gpu;
            foreign.numa_node = 0;
            EXPECT_THROW(requireQwen4ExpSourceEndpoint(
                GlobalDeviceAddress::rocm(0), foreign, "fixture-host", hardware), std::invalid_argument);
            auto cpu = GlobalDeviceAddress::cpu(rank);
            cpu.hostname = "fixture-host";
            EXPECT_NO_THROW(requireQwen4ExpSourceEndpoint(
                GlobalDeviceAddress::cpu(rank), cpu, "fixture-host", hardware));
            auto wrong_cpu = cpu;
            wrong_cpu.numa_node ^= 1;
            EXPECT_THROW(requireQwen4ExpSourceEndpoint(
                GlobalDeviceAddress::cpu(rank), wrong_cpu, "fixture-host", hardware), std::invalid_argument);
        }
    }
    auto cpu_only = OrchestrationConfig::defaults();
    cpu_only.device_mode = DeviceAssignmentMode::EXPLICIT;
    cpu_only.device_for_this_rank = GlobalDeviceAddress::cpu(1);
    EXPECT_FALSE(RuntimeInitPhase::requiresHostWideAcceleratorVisibility(cpu_only, 1));
}
} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp {
/** @test Independent consumer sources receive the same real cold recreation obligation on every GPU count. */
TEST(Qwen4ExpSourceAdmissionCases, ConsumerLocalRecreationRetainsOriginalGPUOwners) {
    EXPECT_FALSE(qwen4ExpSourceReusesContinuationOwners(nullptr));
    const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<30,1ull<<29};};
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})
    for(const auto count:{1,2,4,8})
    for(const auto backing:{AdaptiveImmutableRowBacking::RAM,AdaptiveImmutableRowBacking::SSD}) {
        std::vector<PhysicalMemoryResource> consumers;
        for(int ordinal=0;ordinal<count;++ordinal)consumers.push_back(resource({type,ordinal}));
        ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
        policy.gpu_adaptive.source_backing=backing;
        policy.gpu_adaptive.maximum_gpu_cache_rows=64;
        policy.gpu_adaptive.gpu_cache_physical_byte_limit=1u<<20;
        const auto input=ImmutableRowTablePlacementInput::forConsumers(
            ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),128,64),resource(DeviceId::cpu()),consumers);
        const std::vector<std::size_t> slots(count,32);
        const auto placement=llaminar2::test::Qwen4ExpReplicatedPLECohortReplayAccess::declaration(
            input.gpuAdaptiveLayout(policy,slots,0,16),policy.gpu_adaptive,16);
        EXPECT_EQ(placement->consumers().size(),static_cast<std::size_t>(count));
        EXPECT_TRUE(qwen4ExpSourceReusesContinuationOwners(placement.get()));
        EXPECT_FALSE(qwen4ExpSourceReusesSharedContinuationOwners(placement.get()));
        EXPECT_EQ(placement->gpuAdaptiveRequest().source_backing,backing);
        EXPECT_EQ(placement->gpuAdaptiveRequest().placement.mode,AdaptiveGPUCachePlacementMode::ConsumerLocal);
    }
}

/** @test Actual CPU follower absence and checked GPU ownership select only original reuse cases. */
TEST(Qwen4ExpSourceAdmissionCases, SharedContinuationReuseEligibilityPreservesFollowerAndOwnerRoles) {
    using namespace llaminar2;
    const auto& follower=qwen4ExpCPUJointWorld2SourceAdmissionCases().front();
    ASSERT_FALSE(qwen4ExpCPUJointSourceHasContinuation(follower,1));
    EXPECT_FALSE(qwen4ExpSourceReusesSharedContinuationOwners(nullptr));
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const bool remote:{false,true}){
        const std::string name=type==DeviceType::CUDA?"cuda":"rocm";
        const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<30,1ull<<29};};
        const std::array consumers{resource({type,1}),resource({type,0})};
        const auto other=type==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA;
        std::vector<PhysicalMemoryResource> devices(consumers.begin(),consumers.end());
        if(remote)devices.push_back(resource({other,0}));
        ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
        const std::string owner=remote?(other==DeviceType::CUDA?"cuda":"rocm"):name;
        policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement(remote?
            "0:"+owner+":0/64/1048576@0:"+name+":1":
            "0:"+name+":0/64/1048576,0:"+name+":1/64/1048576@0:"+name+":1");
        const std::array readerEdges{ImmutableRowTableReaderPeerAccess{{0,{type,1}},{0,{type,0}},PeerAccessCoverage::None}};
        const auto input=ImmutableRowTablePlacementInput::forConsumers(
            ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),128,64),resource(DeviceId::cpu()),consumers)
            .withGPUCacheResources(policy,devices).withReaderPeerAccess(readerEdges);
        const std::vector<size_t> slots(remote?1:2,32);
        const auto placement=llaminar2::test::Qwen4ExpReplicatedPLECohortReplayAccess::declaration(
            input.gpuAdaptiveLayout(policy,slots,0,16),policy.gpu_adaptive,16);
        EXPECT_EQ(qwen4ExpSourceReusesSharedContinuationOwners(placement.get()),!remote);
        EXPECT_EQ(qwen4ExpSourceReusesContinuationOwners(placement.get()),!remote);
    }
}

}

namespace llaminar2::test::parity::qwen4exp {
/** @test Canonical source-only projection changes only active MTP and preserves matched seed identity. */
TEST(Qwen4ExpSourceAdmissionCases, SharedContinuationMTPProjectionPreservesCanonicalMatchedSeeds) {
    const auto& cells=qwen4ExpSharedContinuationMTPSourceAdmissionCases();
    size_t seeds=0,ssd=0,ram=0;
    std::set<std::string> original_names,names;
    for(const auto* original:{&qwen4ExpHardwareMatchedCacheEconomyCases(),
                              &qwen4ExpHardwareMatchedCacheRAMBackingEconomyCases()})
    for(const auto& seed:*original){
        EXPECT_TRUE(original_names.insert(seed.testName()).second);
        EXPECT_EQ(seed.mtp,ModelParityMTP::Off);EXPECT_EQ(seed.retained_mtp_draft_capacity,0);
        const bool eligible=qwen4ExpSharedContinuationSourceSeed(seed);
        if(!seed.ngram_table_policy->isGPUAdaptive() ||
           seed.ngram_table_policy->gpu_adaptive.placement.mode==AdaptiveGPUCachePlacementMode::ConsumerLocal)
            EXPECT_FALSE(eligible);
        if(!eligible)continue;
        ++seeds;
        if(seed.ngram_table_policy->gpu_adaptive.source_backing==AdaptiveImmutableRowBacking::RAM)++ram;
        else ++ssd;
        const auto expected=qwen4ExpExpandSourceAdmissionMTP(std::vector<ModelParityCase>{seed});
        for(const auto& reference:expected)if(reference.mtp!=ModelParityMTP::Off){
            const auto found=std::find_if(cells.begin(),cells.end(),[&](const auto& cell){return cell.testName()==reference.testName();});
            ASSERT_NE(found,cells.end());
            EXPECT_EQ(found->retained_mtp_draft_capacity,15);EXPECT_TRUE(found->model.shared_predictor);
            EXPECT_EQ(found->topology.expert_overlay_plan,seed.topology.expert_overlay_plan);
            EXPECT_EQ(found->topology.participants,seed.topology.participants);
            EXPECT_EQ(found->topology.test_id,seed.topology.test_id);
            EXPECT_EQ(found->model.model_path,seed.model.model_path);
            EXPECT_EQ(found->ngram_table_policy,seed.ngram_table_policy);
            EXPECT_EQ(found->activation_precision,seed.activation_precision);EXPECT_EQ(found->kv_cache_precision,seed.kv_cache_precision);
            EXPECT_EQ(found->topology.mpi_ranks,seed.topology.mpi_ranks);
            const auto source_config=seed.toTestConfig(),mtp_config=found->toTestConfig();
            ASSERT_TRUE(source_config.local_hardware_requirements);ASSERT_TRUE(mtp_config.local_hardware_requirements);
            EXPECT_EQ(mtp_config.local_hardware_requirements->cuda_devices,source_config.local_hardware_requirements->cuda_devices);
            EXPECT_EQ(mtp_config.local_hardware_requirements->rocm_devices,source_config.local_hardware_requirements->rocm_devices);
        }
    }
    EXPECT_EQ(seeds,8u);EXPECT_EQ(ssd,4u);EXPECT_EQ(ram,4u);
    EXPECT_EQ(cells.size(),24u);
    for(const auto& cell:cells) {
        EXPECT_NE(cell.mtp,ModelParityMTP::Off);
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_FALSE(original_names.contains(cell.testName()));
    }
    EXPECT_FALSE(qwen4ExpSharedContinuationSourceSeed(qwen4ExpCPUJointWorld2SourceAdmissionCases().front()));
    for(const auto& cell:qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmissionSeeds())
        EXPECT_FALSE(qwen4ExpSharedContinuationSourceSeed(cell));
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Q6 candidates retain original physical roles and the exact64-cell setup product. */
TEST(Qwen4ExpSourceAdmissionCases, ExpertFollowerOwnersRetainOriginalQ6AndAllMTPPolicies) {
    const auto& seeds=qwen4ExpExpertFollowerOwnerSourceAdmissionSeeds();
    ASSERT_EQ(seeds.size(),16u);
    const auto expanded=qwen4ExpExpandSourceAdmissionMTP(seeds);
    ASSERT_EQ(expanded.size(),64u);
    std::set<std::string> names;
    size_t follower=0,both=0,ram=0,positive=0,dynamic=0;
    for(const auto& seed:seeds){
        EXPECT_TRUE(names.insert(seed.testName()).second);
        ASSERT_TRUE(seed.ngram_table_policy);ASSERT_TRUE(seed.expert_overlay);
        const auto& request=seed.ngram_table_policy->gpu_adaptive;
        ASSERT_TRUE(request.placement.coordinator);
        EXPECT_EQ(request.placement.coordinator->device(),DeviceId::cuda(0));
        EXPECT_EQ(request.placement.owners.size()==4 || request.placement.owners.size()==6,true);
        const bool has_continuation=request.placement.owners.size()==6;
        has_continuation?++both:++follower;
        size_t cuda=0,rocm=0;
        for(const auto& owner:request.placement.owners){
            EXPECT_EQ(owner.participant.world_rank,0);
            if(owner.participant.device().is_cuda())++cuda;else if(owner.participant.device().is_rocm())++rocm;
            EXPECT_TRUE(owner.maximum_cache_rows==0 || owner.maximum_cache_rows==64);
        }
        EXPECT_EQ(cuda,has_continuation?2u:0u);EXPECT_EQ(rocm,4u);
        if(request.source_backing==AdaptiveImmutableRowBacking::RAM)++ram;
        if(request.placement.owners.front().maximum_cache_rows==64)++positive;
        if(seed.expert_overlay->movement==ModelParityExpertMovement::Dynamic)++dynamic;
        EXPECT_EQ(request.host.maximum_cache_rows,0u);
        ASSERT_TRUE(seed.topology.expert_overlay_plan);
        const auto& plan=*seed.topology.expert_overlay_plan;
        ASSERT_EQ(plan.routed_tiers.size(),3u);
        EXPECT_EQ(plan.routed_tiers[0].max_experts_per_layer,8);
        EXPECT_EQ(plan.routed_tiers[1].max_experts_per_layer,8);
        const auto domain=std::find_if(plan.domains.begin(),plan.domains.end(),[](const auto& d){return d.name=="rocm_followers";});
        ASSERT_NE(domain,plan.domains.end());EXPECT_EQ(domain->participants.size(),4u);
        const auto cells=qwen4ExpExpandSourceAdmissionMTP(std::vector<ModelParityCase>{seed});
        for(const auto& cell:cells){
            EXPECT_EQ(cell.topology.expert_overlay_plan,seed.topology.expert_overlay_plan);
            EXPECT_EQ(cell.ngram_table_policy,seed.ngram_table_policy);
            EXPECT_EQ(cell.expert_overlay,seed.expert_overlay);
            EXPECT_EQ(cell.model.model_path,seed.model.model_path);
            EXPECT_EQ(cell.topology.participants,seed.topology.participants);
        }
    }
    EXPECT_EQ(follower,8u);EXPECT_EQ(both,8u);EXPECT_EQ(ram,8u);EXPECT_EQ(positive,8u);EXPECT_EQ(dynamic,8u);
    names.clear();for(const auto& cell:expanded)EXPECT_TRUE(names.insert(cell.testName()).second);
    for(const auto policy:{ModelParityExpertOverlayPolicy{},ModelParityExpertOverlayPolicy{
            .owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic}}){
        const auto execution=qwen4ExpExpertFollowerOwnerExecutionCases(policy);
        EXPECT_EQ(execution.size(),32u);EXPECT_EQ(qwen4ExpExpertFollowerOwnerNativeExactCases(policy).size(),24u);
        for(const auto& cell:execution)EXPECT_EQ(cell.expert_overlay,std::optional{policy});
    }
    EXPECT_THROW(qwen4ExpExpertFollowerOwnerPressureDefinition(Qwen4ExpHardwareNgramOwners::Continuation),std::invalid_argument);
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Original Q6 roles require exact known rank identity, including equal foreign ordinals. */
TEST(Qwen4ExpSourceAdmissionCases, ExpertLookupOwnerSelectionRequiresOriginalKnownRank) {
    const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<30,1ull<<29};};
    const std::array consumers{resource(DeviceId::cuda(0)),resource(DeviceId::cuda(1))};
    std::vector<PhysicalMemoryResource> devices(consumers.begin(),consumers.end());
    for(int ordinal=0;ordinal<4;++ordinal)devices.push_back(resource(DeviceId::rocm(ordinal)));
    const std::array readerEdges{ImmutableRowTableReaderPeerAccess{
        {0,DeviceId::cuda(0)},{0,DeviceId::cuda(1)},PeerAccessCoverage::None}};
    for(const auto& cell:qwen4ExpExpertFollowerOwnerSourceAdmissionSeeds()){
        ASSERT_TRUE(cell.ngram_table_policy);
        const auto& policy=*cell.ngram_table_policy;
        const auto input=ImmutableRowTablePlacementInput::forConsumers(
            ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),128,64),resource(DeviceId::cpu()),consumers)
            .withGPUCacheResources(policy,devices).withReaderPeerAccess(readerEdges);
        const std::vector<size_t> slots(policy.gpu_adaptive.placement.owners.size(),0);
        const auto placement=llaminar2::test::Qwen4ExpReplicatedPLECohortReplayAccess::declaration(
            input.gpuAdaptiveLayout(policy,slots,0,16),policy.gpu_adaptive,16);
        const auto owners=qwen4ExpSourceExpertLookupOwners(cell,placement.get());
        ASSERT_EQ(owners.size(),4u);
        for(int ordinal=0;ordinal<4;++ordinal)
            EXPECT_NE(std::find(owners.begin(),owners.end(),PhysicalMemoryAllocatorIdentity{0,DeviceId::rocm(ordinal)}),owners.end());
        auto changed=cell;
        auto plan=std::make_shared<MoERoutedExpertPlacementPlan>(*cell.topology.expert_overlay_plan);
        changed.topology.expert_overlay_plan=plan;
        const auto domain=std::find_if(plan->domains.begin(),plan->domains.end(),[](const auto& d){return d.name=="rocm_followers";});
        ASSERT_NE(domain,plan->domains.end());
        domain->owner_rank=1;domain->world_ranks.clear();
        EXPECT_TRUE(qwen4ExpSourceExpertLookupOwners(changed,placement.get()).empty())
            << "Foreign rank's same ROCm ordinals cannot authorize local storage owners";
        domain->owner_rank=-1;
        EXPECT_TRUE(qwen4ExpSourceExpertLookupOwners(changed,placement.get()).empty())
            << "Unresolved original rank cannot authorize a local expert owner";
        domain->world_ranks={1,0,0,0};
        const auto partial=qwen4ExpSourceExpertLookupOwners(changed,placement.get());
        EXPECT_EQ(partial.size(),3u);
        EXPECT_EQ(std::find(partial.begin(),partial.end(),PhysicalMemoryAllocatorIdentity{0,DeviceId::rocm(0)}),partial.end());
        EXPECT_FALSE(qwen4ExpSourceReusesSharedContinuationOwners(placement.get()));
    }
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Moving MTP proof retains the original physical source and exact oracle contract. */
TEST(Qwen4ExpSourceAdmissionCases, CPUJointMovingMTPExactRetainsOriginalRequests)
{
    const auto &original=qwen4ExpCPUJointWorld2MTPExactCases();
    const auto &moving=qwen4ExpCPUJointWorld2DynamicMTPExactCases();
    ASSERT_EQ(moving.size(),3u);
    std::set<ModelParityMTP> policies;
    for(const auto &cell:moving) {
        policies.insert(cell.mtp);
        const auto source=std::find_if(original.begin(),original.end(),[&](const auto &value){return value.mtp==cell.mtp;});
        ASSERT_NE(source,original.end());
        EXPECT_EQ(cell.topology.participants,source->topology.participants);
        EXPECT_EQ(cell.topology.mpi_ranks,2);
        EXPECT_EQ(cell.retained_mtp_draft_capacity,3);
        EXPECT_EQ(cell.model.maximum_mtp_draft_depth,3);
        EXPECT_EQ(cell.model.shared_predictor,source->model.shared_predictor);
        EXPECT_EQ(cell.ngram_table_policy,source->ngram_table_policy);
        EXPECT_EQ(cell.expert_overlay->owner_order,source->expert_overlay->owner_order);
        EXPECT_EQ(cell.expert_overlay->movement,ModelParityExpertMovement::Dynamic);
        EXPECT_EQ(cell.dynamic_evidence,ModelParityDynamicEvidence::EconomicMovement);
        EXPECT_EQ(source->expert_overlay->movement,ModelParityExpertMovement::Static);
        EXPECT_EQ(cell.collective_evidence_source,ParityCollectiveEvidenceSource::PostCollectiveSnapshot);
        EXPECT_NE(cell.testName(),source->testName());
    }
    EXPECT_EQ(policies,(std::set<ModelParityMTP>{ModelParityMTP::Depth1,ModelParityMTP::Depth3,ModelParityMTP::DynamicDepth}));
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Every nonempty original continuation mask preserves native readers, coordinator and follower roles. */
TEST(Qwen4ExpSourceAdmissionCases, OwnerSubsetMasksExhaustOriginalHardwareDomains)
{
    std::size_t masks=0,axes=0,pure_coordinators=0;
    for(const auto target:kQwen4ExpHardwareTargets) {
        const auto role=target==Qwen4ExpHardwareTarget::CUDA2ROCm4
            ? Qwen4ExpHardwareNgramOwners::Both:Qwen4ExpHardwareNgramOwners::Continuation;
        const auto original=qwen4ExpHardwareTargetDefinition(target,role);
        const auto& plan=*original.topology.expert_overlay_plan;
        const auto domain=std::find_if(plan.domains.begin(),plan.domains.end(),
            [&](const auto& value){return value.name==plan.continuation_domain;});
        ASSERT_NE(domain,plan.domains.end());
        const auto degree=domain->participants.size();
        const auto end=std::uint64_t{1}<<degree;
        EXPECT_THROW(qwen4ExpOwnerSubsetPressureDefinition(target,0),std::invalid_argument);
        EXPECT_THROW(qwen4ExpOwnerSubsetPressureDefinition(target,end),std::invalid_argument);
        for(std::uint64_t mask=1;mask<end;++mask) {
            const auto definition=qwen4ExpOwnerSubsetPressureDefinition(target,mask);
            ++masks;if(!(mask&1))++pure_coordinators;
            EXPECT_EQ(definition.topology.participants,original.topology.participants);
            ASSERT_EQ(definition.topology.expert_overlay_plan->domains.size(),plan.domains.size());
            ASSERT_EQ(definition.topology.expert_overlay_plan->routed_tiers.size(),plan.routed_tiers.size());
            for(std::size_t i=0;i<plan.domains.size();++i)
                EXPECT_EQ(definition.topology.expert_overlay_plan->domains[i].participants,plan.domains[i].participants);
            for(std::size_t i=0;i<plan.routed_tiers.size();++i)
                EXPECT_EQ(definition.topology.expert_overlay_plan->routed_tiers[i].max_experts_per_layer,
                    plan.routed_tiers[i].max_experts_per_layer);
            ASSERT_EQ(definition.features.ngram_storage.size(),4u);
            std::set<std::pair<AdaptiveImmutableRowBacking,bool>> profiles;
            for(const auto& policy:definition.features.ngram_storage) {
                ++axes;ASSERT_TRUE(policy.isGPUAdaptive());ASSERT_NO_THROW(policy.validate());
                const auto& request=policy.gpu_adaptive;
                ASSERT_TRUE(request.placement.coordinator);
                EXPECT_EQ(request.placement.coordinator->world_rank,domain->owner_rank);
                EXPECT_EQ(request.placement.coordinator->device(),
                    (DeviceId{domain->participants.front().device_type,domain->participants.front().device_ordinal}));
                std::size_t selected=0,followers=0;
                for(std::size_t coordinate=0;coordinate<degree;++coordinate) {
                    const auto& address=domain->participants[coordinate];
                    const auto found=std::find_if(request.placement.owners.begin(),request.placement.owners.end(),
                        [&](const auto& owner){return owner.participant.world_rank==domain->owner_rank &&
                            owner.participant.device()==DeviceId{address.device_type,address.device_ordinal};});
                    EXPECT_EQ(found!=request.placement.owners.end(),bool(mask&(std::uint64_t{1}<<coordinate)));
                    if(found!=request.placement.owners.end())++selected;
                }
                for(const auto& owner:request.placement.owners) {
                    EXPECT_EQ(owner.participant.world_rank,0);
                    if(target==Qwen4ExpHardwareTarget::CUDA2ROCm4 && owner.participant.device().is_rocm())++followers;
                }
                EXPECT_EQ(followers,target==Qwen4ExpHardwareTarget::CUDA2ROCm4?4u:0u);
                EXPECT_EQ(request.placement.owners.size(),selected+followers);
                profiles.insert({request.source_backing,request.placement.owners.front().maximum_cache_rows!=0});
            }
            EXPECT_EQ(profiles.size(),4u);
        }
    }
    EXPECT_EQ(masks,26u);EXPECT_EQ(axes,104u);EXPECT_EQ(pure_coordinators,10u);
}

/** @test Ten representative requests reuse original source/MTP policies without expanding the expensive product. */
TEST(Qwen4ExpSourceAdmissionCases, OwnerSubsetRealModelCandidatesKeepBoundedOriginalProduct)
{
    const auto& seeds=qwen4ExpOwnerSubsetSourceAdmissionSeeds();ASSERT_EQ(seeds.size(),10u);
    const auto source=qwen4ExpExpandSourceAdmissionMTP(seeds,Qwen4ExpSourceMTPEnvelope::Retained3);
    ASSERT_EQ(source.size(),40u);
    std::set<std::string> names;
    std::size_t moving=0;
    for(const auto& seed:seeds) {
        EXPECT_TRUE(names.insert(seed.testName()).second);
        EXPECT_EQ(seed.mtp,ModelParityMTP::Off);EXPECT_EQ(seed.retained_mtp_draft_capacity,0);
        EXPECT_EQ(seed.model.transformer_layers,48);
        ASSERT_TRUE(seed.ngram_table_policy);ASSERT_TRUE(seed.ngram_table_policy->isGPUAdaptive());
        ASSERT_TRUE(seed.expert_overlay);
        if(seed.expert_overlay->movement==ModelParityExpertMovement::Dynamic)++moving;
        const auto config=seed.toTestConfig();ASSERT_TRUE(config.local_hardware_requirements);
        EXPECT_TRUE(config.moe_routed_expert_plan);
    }
    EXPECT_EQ(moving,2u);names.clear();
    for(std::size_t index=0;index<source.size();++index) {
        const auto& cell=source[index];const auto& seed=seeds[index/4];
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.topology.expert_overlay_plan,seed.topology.expert_overlay_plan);
        EXPECT_EQ(cell.topology.participants,seed.topology.participants);
        EXPECT_EQ(cell.ngram_table_policy,seed.ngram_table_policy);
        EXPECT_EQ(cell.expert_overlay,seed.expert_overlay);
        EXPECT_EQ(cell.model.model_path,seed.model.model_path);
        EXPECT_EQ(cell.retained_mtp_draft_capacity,cell.mtp==ModelParityMTP::Off?0:3);
    }
    names.clear();std::size_t numerical=0,exact=0;
    for(const auto policy:{ModelParityExpertOverlayPolicy{},ModelParityExpertOverlayPolicy{
            .owner_order=RoutedExpertOwnerOrder::Ordinal,.movement=ModelParityExpertMovement::Dynamic}}) {
        const auto cells=qwen4ExpOwnerSubsetExecutionCases(policy);
        const auto native=qwen4ExpOwnerSubsetNativeExactCases(policy);
        EXPECT_EQ(cells.size(),policy.movement==ModelParityExpertMovement::Static?32u:8u);
        EXPECT_EQ(native.size(),policy.movement==ModelParityExpertMovement::Static?24u:6u);
        numerical+=cells.size();exact+=native.size();
        for(const auto& cell:cells) {
            EXPECT_TRUE(names.insert(cell.testName()).second);EXPECT_EQ(cell.expert_overlay,std::optional{policy});
            EXPECT_EQ(cell.retained_mtp_draft_capacity,3);EXPECT_EQ(cell.model.maximum_mtp_draft_depth,3);
            EXPECT_EQ(cell.inventory_role,ModelParityInventoryRole::NumericalDiagnostic);
            ASSERT_TRUE(cell.model.shared_predictor);EXPECT_EQ(cell.model.transformer_layers,48);
            EXPECT_TRUE(cell.toTestConfig().local_hardware_requirements);
        }
        for(const auto& cell:native)EXPECT_NE(cell.mtp,ModelParityMTP::Off);
    }
    EXPECT_EQ(numerical,40u);EXPECT_EQ(exact,30u);
    EXPECT_THROW(qwen4ExpOwnerSubsetExecutionCases({.owner_order=RoutedExpertOwnerOrder::Random,
        .movement=ModelParityExpertMovement::Static}),std::invalid_argument);
}

/** @test Partial owners, including an unowned coordinator, enter the same real cold recreation source gate. */
TEST(Qwen4ExpSourceAdmissionCases, OwnerSubsetSourceReuseRequiresOriginalNonemptyMembership)
{
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const std::size_t degree:{2u,4u}) {
        const auto resource=[](DeviceId device){return PhysicalMemoryResource{0,device,1ull<<30,1ull<<29};};
        std::vector<PhysicalMemoryResource> resources;
        std::vector<PhysicalMemoryAllocatorIdentity> consumers;
        std::vector<ImmutableRowTableReaderPeerAccess> edges;
        for(std::size_t coordinate=0;coordinate<degree;++coordinate) {
            resources.push_back(resource({type,int(coordinate)}));consumers.push_back({0,{type,int(coordinate)}});
            if(coordinate)edges.push_back({consumers.front(),consumers.back(),PeerAccessCoverage::None});
        }
        for(std::uint64_t mask=1;mask<(std::uint64_t{1}<<degree);++mask) {
            ImmutableRowTablePolicy policy{.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD};
            auto& declared=policy.gpu_adaptive.placement;
            declared.mode=AdaptiveGPUCachePlacementMode::DeclaredRankLocal;
            declared.coordinator=AdaptiveGPUCacheParticipant{0,type,0};
            for(std::size_t coordinate=0;coordinate<degree;++coordinate)
                if(mask&(std::uint64_t{1}<<coordinate))declared.owners.push_back({{0,type,int(coordinate)},64,1u<<20});
            ASSERT_NO_THROW(policy.validate());ASSERT_TRUE(qwen4ExpSharedContinuationOwnerSet(declared,consumers));
            const auto input=ImmutableRowTablePlacementInput::forConsumers(
                ImmutableRowTableExtent::fromNative(static_cast<GGUFTensorType>(0),128,64),resource(DeviceId::cpu()),resources)
                .withGPUCacheResources(policy,resources).withReaderPeerAccess(edges);
            const std::vector<std::size_t> slots(declared.owners.size(),32);
            const auto placement=llaminar2::test::Qwen4ExpReplicatedPLECohortReplayAccess::declaration(
                input.gpuAdaptiveLayout(policy,slots,0,16),policy.gpu_adaptive,16);
            EXPECT_TRUE(qwen4ExpSourceReusesSharedContinuationOwners(placement.get()));
            auto changed=declared;changed.owners.clear();EXPECT_FALSE(qwen4ExpSharedContinuationOwnerSet(changed,consumers));
            changed=declared;changed.owners.push_back(changed.owners.front());EXPECT_FALSE(qwen4ExpSharedContinuationOwnerSet(changed,consumers));
            changed=declared;changed.owners.front().participant.world_rank=1;EXPECT_FALSE(qwen4ExpSharedContinuationOwnerSet(changed,consumers));
            changed=declared;changed.coordinator->world_rank=1;EXPECT_FALSE(qwen4ExpSharedContinuationOwnerSet(changed,consumers));
        }
    }
}
} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp
{
/** @test Exactly three original source/math policies retain GPU-only Q6 and ordinary RAM PLE. */
TEST(Qwen4ExpSourceAdmissionCases, AllGPUQ6RetainsOriginalSourceRAMAndThreeMTPPolicies)
{
    const auto& cases = qwen4ExpAllGPUQ6MTPNativeExactCases();
    ASSERT_EQ(cases.size(), 3u);
    std::set<ModelParityMTP> policies;
    std::set<std::string> names;
    for (const auto& cell : cases) {
        policies.insert(cell.mtp);
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.inventory_role, ModelParityInventoryRole::NumericalDiagnostic);
        EXPECT_EQ(cell.model.model_path, kMainModelPath);
        ASSERT_TRUE(cell.model.shared_predictor);
        EXPECT_EQ(cell.model.shared_predictor->path(), kPredictorModelPath);
        EXPECT_EQ(cell.model.maximum_mtp_draft_depth, 3);
        EXPECT_EQ(cell.retained_mtp_draft_capacity, 3);
        EXPECT_EQ(cell.topology.mpi_ranks, 1);
        ASSERT_EQ(cell.topology.participants.size(), 6u);
        int cuda = 0, rocm = 0;
        for (const auto& participant : cell.topology.participants) {
            EXPECT_EQ(participant.world_rank, std::optional<int>{0});
            EXPECT_TRUE(participant.address.isGPU());
            cuda += participant.address.device_type == DeviceType::CUDA;
            rocm += participant.address.device_type == DeviceType::ROCm;
        }
        EXPECT_EQ(cuda, 2); EXPECT_EQ(rocm, 4);
        ASSERT_TRUE(cell.ngram_table_policy);
        EXPECT_EQ(cell.ngram_table_policy->storage, ImmutableRowStoragePreference::RAM);
        EXPECT_FALSE(cell.ngram_table_policy->isGPUAdaptive());
        EXPECT_EQ(cell.expert_overlay, (std::optional{ModelParityExpertOverlayPolicy{
            .owner_order = RoutedExpertOwnerOrder::Ordinal, .movement = ModelParityExpertMovement::Dynamic}}));
        ASSERT_TRUE(cell.topology.expert_overlay_plan);
        const auto& plan = *cell.topology.expert_overlay_plan;
        EXPECT_EQ(resolveMoEOverlayAuthorityExecutionKind(plan), MoEOverlayAuthorityExecutionKind::DeviceResident);
        EXPECT_FALSE(MoEExpertOwnerMap::usesCPUJointFragments(plan));
        ASSERT_EQ(plan.routed_tiers.size(), 2u);
        EXPECT_EQ(plan.routed_tiers.front().max_experts_per_layer, 8);
        EXPECT_EQ(plan.routed_tiers.back().max_experts_per_layer, 0);
        EXPECT_TRUE(plan.routed_tiers.back().fallback);
        EXPECT_TRUE(validateMoERoutedExpertPlacementPlan(plan).ok());
    }
    EXPECT_EQ(policies, (std::set{ModelParityMTP::Depth1, ModelParityMTP::Depth3, ModelParityMTP::DynamicDepth}));
}

/** @test Actual metadata owner/controller builders select the existing world1 mapped GraphService branch. */
TEST(Qwen4ExpSourceAdmissionCases, AllGPUQ6UsesTwoOriginalNativeGroupsAndMappedControl)
{
    auto plan = *qwen4ExpAllGPUQ6MTPNativeExactCases().front().topology.expert_overlay_plan;
    plan.authority_execution = resolveMoEOverlayAuthorityExecutionKind(plan);
    EXPECT_EQ(MoEOverlayCapacityAdmission::selectMigrationStorage(plan, MoERebalanceRuntimeMode::Dynamic),
              MoEOverlayMigrationStorageKind::PhysicalResidencyFabric);
    // This one representative 512-expert metadata row exercises the original
    // owner-map constructor. It grants no physical slot, prepared source or PMA admission.
    std::vector<int> tiers(512, 1);
    std::fill_n(tiers.begin(), 8, 0);
    plan.placements = {{.layer = 0, .routed_expert_tier = std::move(tiers)}};
    const auto owners = MoEExpertOwnerMap::build(plan);
    const std::array node_ids{0};
    const auto topology = resolveMoEOverlayDeviceControllerTopology(plan, owners, {.world_rank_node_ids = node_ids});
    ASSERT_TRUE(topology.valid());
    EXPECT_EQ(topology.leader_world_rank, 0);
    ASSERT_EQ(topology.groups.size(), 2u);
    EXPECT_EQ(topology.groups[0].participant_ids, (std::vector<int>{0, 1}));
    EXPECT_EQ(topology.groups[1].participant_ids, (std::vector<int>{2, 3, 4, 5}));
    EXPECT_EQ(topology.groups[0].inter_group_transport, MoEOverlayDeviceControllerInterGroupTransport::LeaderLocal);
    EXPECT_EQ(topology.groups[1].inter_group_transport, MoEOverlayDeviceControllerInterGroupTransport::NodeLocalMapped);
    for (const auto& group : topology.groups) {
        EXPECT_EQ(group.root_world_rank, 0);
        EXPECT_EQ(group.intra_group_transport, MoEOverlayDeviceControllerIntraGroupTransport::NativeCollective);
    }
    EXPECT_TRUE(topology.usesNodeLocalMappedControl());
    std::map<int, int> expert_counts;
    for (const auto& owner : owners.owners()) ++expert_counts[owner.owner_participant];
    EXPECT_EQ(expert_counts, (std::map<int, int>{{0, 4}, {1, 4}, {2, 126}, {3, 126}, {4, 126}, {5, 126}}));
    // The old Q6 row is intentionally still host-owned; storage owners alone
    // never turn its routed CPU capacity tier into a GPU finite-scope test.
    const auto original = qwen4ExpHardwareTargetDefinition(Qwen4ExpHardwareTarget::CUDA2ROCm4);
    EXPECT_EQ(resolveMoEOverlayAuthorityExecutionKind(*original.topology.expert_overlay_plan),
              MoEOverlayAuthorityExecutionKind::HostResident);
}

/** @test Source admission and public runtime arguments preserve the same three retained3 policies. */
TEST(Qwen4ExpSourceAdmissionCases, AllGPUQ6SourceAdmissionAndPublicRuntimeKeepOriginalPolicies)
{
    const auto& cells = qwen4ExpAllGPUQ6MTPNativeExactCases();
    const auto admissions = qwen4ExpSourceAdmissionParameters(cells);
    ASSERT_EQ(admissions.size(), 3u);
    for (const auto& cell : cells) {
        const auto runtime = cell.makeOrchestrationConfig(cell.model.model_path, 0);
        const auto source = qwen4ExpSourceAdmissionConfig(cell, cell.model.model_path, 0);
        const auto public_request = sourcePublicConfig(cell);
        for (const auto* config : {&runtime, &source, &public_request}) {
            ASSERT_TRUE(config->moe_routed_expert_plan);
            EXPECT_EQ(resolveMoEOverlayAuthorityExecutionKind(*config->moe_routed_expert_plan),
                      MoEOverlayAuthorityExecutionKind::DeviceResident);
            EXPECT_EQ(config->moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
            EXPECT_EQ(config->ngram_table_policy.storage, ImmutableRowStoragePreference::RAM);
            EXPECT_TRUE(config->mtp.enabled);
            EXPECT_EQ(resolveMTPMaximumDraftDepth(config->mtp), 3);
            ASSERT_TRUE(config->shared_predictor);
            EXPECT_EQ(config->shared_predictor->path(), kPredictorModelPath);
        }
        EXPECT_EQ(runtime.mtp.depth_policy.mode, source.mtp.depth_policy.mode);
        EXPECT_EQ(runtime.mtp.depth_policy.mode, public_request.mtp.depth_policy.mode);
        EXPECT_EQ(runtime.mtp.draft_tokens, source.mtp.draft_tokens);
        EXPECT_EQ(runtime.mtp.draft_tokens, public_request.mtp.draft_tokens);
    }
}

/** @test Functional Off generation projects two unchanged original retained3 shared-owner requests. */
TEST(Qwen4ExpSourceAdmissionCases, OwnerSubsetOrdinaryGenerationKeepsOriginalOffRequests)
{
    const auto cells=qwen4ExpOwnerSubsetOrdinaryGenerationCases();ASSERT_EQ(cells.size(),2u);
    const auto original=qwen4ExpOwnerSubsetExecutionCases(ModelParityExpertOverlayPolicy{});
    std::set<std::string> names;
    for(const auto& cell:cells) {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        EXPECT_EQ(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.retained_mtp_draft_capacity,3);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        ASSERT_TRUE(cell.ngram_table_policy);const auto& storage=cell.ngram_table_policy->gpu_adaptive;
        EXPECT_EQ(storage.source_backing,AdaptiveImmutableRowBacking::RAM);
        ASSERT_EQ(storage.placement.owners.size(),1u);
        EXPECT_EQ(storage.placement.owners.front().participant.device().ordinal,1);
        EXPECT_TRUE(std::any_of(original.begin(),original.end(),[&](const auto& old){return old.testName()==cell.testName();}));
    }
}

} // namespace llaminar2::test::parity::qwen4exp

namespace llaminar2::test::parity::qwen4exp {
/** @test Explicit current-demand candidates preserve exact original seeds and truthful role/backend coverage. */
TEST(Qwen4ExpSourceAdmissionCases, PooledCurrentDemandCandidatesPreserveSupportedOriginalSeeds)
{
    const auto originalMatched=qwen4ExpHardwareMatchedCacheEconomyCases();
    const auto originalPure=qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmissionSeeds();
    const auto& cases=qwen4ExpPooledCurrentDemandCases();ASSERT_EQ(cases.size(),4u);
    unsigned pure=0,inlineOwners=0;std::set<DeviceType> inlineBackends;std::set<std::uint64_t> pureCaps;
    for(const auto& candidate:cases){
        EXPECT_EQ(candidate.mtp,ModelParityMTP::Off);EXPECT_EQ(candidate.retained_mtp_draft_capacity,0);
        EXPECT_EQ(candidate.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        EXPECT_EQ(candidate.inventory_role,ModelParityInventoryRole::NumericalDiagnostic);
        const auto role=qwen4ExpCurrentDemandRole(candidate);const auto& pool=candidate.ngram_table_policy->gpu_adaptive;
        const auto& original=role==Qwen4ExpCurrentDemandRole::PureReader?originalPure:originalMatched;
        const auto found=std::find_if(original.begin(),original.end(),[&](const auto& value){return value.topology.test_id+"_PooledEarlyCurrent"==candidate.topology.test_id && value.ngram_table_policy==candidate.ngram_table_policy;});
        ASSERT_NE(found,original.end());auto normalized=candidate;
        normalized.topology.test_id=found->topology.test_id;normalized.inventory_role=found->inventory_role;
        EXPECT_EQ(::testing::PrintToString(normalized),::testing::PrintToString(*found));
        if(role==Qwen4ExpCurrentDemandRole::PureReader){
            ++pure;ASSERT_EQ(pool.placement.owners.size(),1u);
            EXPECT_EQ(pool.placement.coordinator->device().type,DeviceType::CUDA);
            EXPECT_EQ(pool.placement.owners.front().participant.device().type,DeviceType::ROCm);
            pureCaps.insert(pool.placement.owners.front().maximum_cache_rows);
        }else{
            ++inlineOwners;ASSERT_EQ(pool.placement.owners.size(),2u);
            inlineBackends.insert(pool.placement.coordinator->device().type);
            for(const auto& owner:pool.placement.owners)EXPECT_EQ(owner.cache_physical_byte_limit,std::size_t{256}<<20);
        }
    }
    EXPECT_EQ(pure,2u);EXPECT_EQ(inlineOwners,2u);
    EXPECT_EQ(pureCaps,(std::set<std::uint64_t>{0,64}));
    EXPECT_EQ(inlineBackends,(std::set<DeviceType>{DeviceType::CUDA,DeviceType::ROCm}));
    for(std::size_t index=0;index<originalMatched.size();++index)
        EXPECT_EQ(::testing::PrintToString(originalMatched[index]),::testing::PrintToString(qwen4ExpHardwareMatchedCacheEconomyCases()[index]));
    for(std::size_t index=0;index<originalPure.size();++index)
        EXPECT_EQ(::testing::PrintToString(originalPure[index]),::testing::PrintToString(qwen4ExpSharedCoordinatorCUDATP2AdaptiveSourceAdmissionSeeds()[index]));
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Existing typed MTP expansion preserves all original current-demand seed/owner identities without duplicate Off launches. */
TEST(Qwen4ExpSourceAdmissionCases,PooledCurrentDemandMTPPreservesOriginalSeeds)
{
 const auto& seeds=qwen4ExpPooledCurrentDemandCases();const auto& active=qwen4ExpPooledCurrentMTPDemandCases();
 ASSERT_EQ(seeds.size(),4u);ASSERT_EQ(active.size(),12u);std::set<std::string> names;
 std::map<ModelParityMTP,int> policies;
 const auto expected=qwen4ExpExpandSourceAdmissionMTP(seeds,Qwen4ExpSourceMTPEnvelope::Retained3);
 for(const auto& cell:active){ASSERT_NE(cell.mtp,ModelParityMTP::Off);EXPECT_EQ(cell.retained_mtp_draft_capacity,3);
  ASSERT_TRUE(cell.model.shared_predictor);EXPECT_EQ(cell.inventory_role,ModelParityInventoryRole::NumericalDiagnostic);
  EXPECT_TRUE(names.insert(cell.testName()).second);++policies[cell.mtp];
  const auto original=std::find_if(expected.begin(),expected.end(),[&](const auto& row){return row.testName()==cell.testName();});
  ASSERT_NE(original,expected.end());EXPECT_EQ(::testing::PrintToString(cell),::testing::PrintToString(*original));
  EXPECT_NO_THROW(qwen4ExpCurrentDemandRole(cell));
 }
 EXPECT_EQ(policies[ModelParityMTP::Depth1],4);EXPECT_EQ(policies[ModelParityMTP::Depth3],4);EXPECT_EQ(policies[ModelParityMTP::DynamicDepth],4);
 for(const auto& seed:seeds){EXPECT_EQ(seed.mtp,ModelParityMTP::Off);EXPECT_EQ(seed.retained_mtp_draft_capacity,0);}
}

}

namespace llaminar2::test::parity::qwen4exp {
/** @test Combined C0 is a lossless bounded projection of the original ROCm2 RAM owner-subset cell. */
TEST(Qwen4ExpSourceAdmissionCases, CombinedLocalHotC0CandidatesRetainOriginalRequests)
{
    const auto candidates=qwen4ExpCombinedLocalHotC0Cases();ASSERT_EQ(candidates.size(),2u);
    const auto original=qwen4ExpOwnerSubsetExecutionCases(ModelParityExpertOverlayPolicy{});
    std::set<ModelParityMTP> modes;
    for(const auto& cell:candidates) {
        ASSERT_TRUE(cell.ngram_table_policy);const auto& request=cell.ngram_table_policy->gpu_adaptive;
        ASSERT_TRUE(request.local_hot_cache);
        EXPECT_EQ(request.local_hot_cache->maximum_cache_rows,0u);
        EXPECT_EQ(request.local_hot_cache->cache_physical_byte_limit,0u);
        EXPECT_TRUE(request.local_hot_cache->consumer_overrides.empty());
        EXPECT_EQ(request.source_backing,AdaptiveImmutableRowBacking::RAM);
        EXPECT_EQ(request.placement.mode,AdaptiveGPUCachePlacementMode::DeclaredRankLocal);
        ASSERT_EQ(request.placement.owners.size(),1u);
        EXPECT_EQ(request.placement.owners.front().participant.device(),DeviceId::rocm(1));
        EXPECT_EQ(cell.retained_mtp_draft_capacity,3);
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        EXPECT_TRUE(modes.insert(cell.mtp).second);
        auto normalized=cell;
        normalized.ngram_table_policy->gpu_adaptive.local_hot_cache.reset();
        const std::string suffix="_CombinedLocalHotC0Candidate";
        ASSERT_TRUE(normalized.topology.test_id.ends_with(suffix));
        normalized.topology.test_id.resize(normalized.topology.test_id.size()-suffix.size());
        const auto found=std::find_if(original.begin(),original.end(),[&](const auto& old){return old.testName()==normalized.testName();});
        ASSERT_NE(found,original.end());
        EXPECT_EQ(::testing::PrintToString(normalized),::testing::PrintToString(*found));
        const auto native=cell.makeOrchestrationConfig(cell.model.model_path,0);
        const auto parsed=sourcePublicConfig(cell);
        EXPECT_EQ(native.ngram_table_policy,*cell.ngram_table_policy);
        EXPECT_EQ(parsed.ngram_table_policy,*cell.ngram_table_policy);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
    }
    EXPECT_EQ(modes,(std::set<ModelParityMTP>{ModelParityMTP::Off,ModelParityMTP::Depth1}));
    const auto ordinary=qwen4ExpCombinedLocalHotC0OrdinaryCases();ASSERT_EQ(ordinary.size(),1u);
    const auto exact=qwen4ExpCombinedLocalHotC0ExactCases();ASSERT_EQ(exact.size(),1u);
    EXPECT_EQ(ordinary.front().mtp,ModelParityMTP::Off);EXPECT_EQ(exact.front().mtp,ModelParityMTP::Depth1);
}

}

namespace llaminar2::test::parity::qwen4exp {
/** @test Five optional-cap comparisons retain exact original source/numerical/retained policies. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotBudgetProfilesPreserveOriginalModelAndDistinctHierarchies)
{
    const auto originals=qwen4ExpCombinedLocalHotC0Cases();
    const auto candidates=qwen4ExpLocalHotBudgetCases();ASSERT_EQ(candidates.size(),10u);
    EXPECT_EQ(qwen4ExpLocalHotBudgetOrdinaryCases().size(),5u);EXPECT_EQ(qwen4ExpLocalHotBudgetExactCases().size(),5u);
    std::set<std::string> names;std::size_t consumerLocal=0,sharedOnly=0,combined=0,zeroPositive=0,unequal=0;
    for(const auto& cell:candidates) {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        const auto found=std::find_if(originals.begin(),originals.end(),[&](const auto& original){return original.mtp==cell.mtp;});ASSERT_NE(found,originals.end());
        auto restored=cell;restored.ngram_table_policy=found->ngram_table_policy;restored.topology.test_id=found->topology.test_id;
        EXPECT_EQ(::testing::PrintToString(restored),::testing::PrintToString(*found));
        const auto consumers=qwen4ExpLocalHotOriginalConsumers(cell);const auto& g=cell.ngram_table_policy->gpu_adaptive;
        std::size_t total=0;
        if(g.placement.mode==AdaptiveGPUCachePlacementMode::ConsumerLocal) {
            ++consumerLocal;EXPECT_FALSE(g.local_hot_cache);EXPECT_TRUE(g.placement.owners.empty());
            total=g.gpu_cache_physical_byte_limit*consumers.size();
        } else {
            EXPECT_EQ(g.placement.mode,AdaptiveGPUCachePlacementMode::DeclaredRankLocal);
            EXPECT_EQ(g.placement.coordinator,found->ngram_table_policy->gpu_adaptive.placement.coordinator);
            ASSERT_EQ(g.placement.owners.size(),found->ngram_table_policy->gpu_adaptive.placement.owners.size());
            for(std::size_t i=0;i<g.placement.owners.size();++i) {
                EXPECT_EQ(g.placement.owners[i].participant,found->ngram_table_policy->gpu_adaptive.placement.owners[i].participant);
                total+=g.placement.owners[i].cache_physical_byte_limit;
            }
            if(!g.local_hot_cache)++sharedOnly;
            else {
                ++combined;g.local_hot_cache->requireConsumers(consumers);EXPECT_EQ(g.local_hot_cache->policy,g.gpu_policy);
                std::set<std::size_t> limits;
                for(const auto& consumer:consumers){const auto request=g.local_hot_cache->ceilingsFor(consumer);limits.insert(request.cache_physical_byte_limit);total+=request.cache_physical_byte_limit;}
                if(limits.size()>1){++unequal;if(limits.contains(0))++zeroPositive;}
            }
        }
        EXPECT_EQ(total,std::size_t{128}<<20); // Optional ceilings only; mandatory native footprints are deliberately separate.
        EXPECT_EQ(cell.ngram_table_policy->gpu_adaptive.source_backing,AdaptiveImmutableRowBacking::RAM);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(cell.makeOrchestrationConfig(cell.model.model_path,0)).empty());
    }
    EXPECT_EQ(consumerLocal,2u);EXPECT_EQ(sharedOnly,2u);EXPECT_EQ(combined,6u);EXPECT_EQ(unequal,4u);EXPECT_EQ(zeroPositive,2u);
}
/** @test Unequal and zero/positive requests use the actual lossless public document and preserve every consumer. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotBudgetLosslessRuntimeDocumentUsesPublicParser)
{
    std::size_t documents=0;
    for(const auto& cell:qwen4ExpLocalHotBudgetCases()) {
        const auto record=nlohmann::json::parse(::testing::PrintToString(cell));
        if(modelParityRuntimeRepresentation(cell)==ModelParityRuntimeRepresentation::Arguments) {
            EXPECT_TRUE(record["runtime"].contains("server_args"));EXPECT_FALSE(record["runtime"].contains("configuration_document"));
            EXPECT_EQ(sourcePublicConfig(cell).ngram_table_policy,*cell.ngram_table_policy);continue;
        }
        ++documents;EXPECT_THROW(modelParityServerArguments(cell),std::invalid_argument);
        EXPECT_THROW(modelParityNgramStorageArguments(cell),std::invalid_argument);
        ASSERT_FALSE(record["runtime"].contains("server_args"));
        const auto& benchmark=record["diagnostic_benchmark"];
        ASSERT_FALSE(benchmark.is_null());EXPECT_FALSE(benchmark.contains("args"));
        EXPECT_EQ(benchmark["configuration_policy"],"lossless_declared_document_v1");
        EXPECT_EQ(benchmark["configuration_document"],record["runtime"]["configuration_document"]);
        EXPECT_EQ(record["runtime"]["configuration_policy"],"lossless_declared_document_v1");
        const auto document=record["runtime"]["configuration_document"].dump();
        const auto restored=deserializeOrchestrationConfig(document);EXPECT_EQ(restored.ngram_table_policy,*cell.ngram_table_policy);
        const auto path=std::filesystem::temp_directory_path()/("qfn-lossless-local-cache-"+std::to_string(::getpid())+"-"+std::to_string(documents)+".json");
        struct Remove {std::filesystem::path path;~Remove(){std::error_code ignored;std::filesystem::remove(path,ignored);}} remove{path};
        {std::ofstream out(path);ASSERT_TRUE(out);out<<document;ASSERT_TRUE(out.good());}
        std::vector<std::string> args{"llaminar2","--config",path.string()};std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
        const auto parsed=OrchestrationConfigParser{}.parseArgs(int(argv.size()),argv.data());
        EXPECT_EQ(parsed.ngram_table_policy,*cell.ngram_table_policy);EXPECT_EQ(parsed.model_path,cell.model.model_path);
        ASSERT_TRUE(parsed.shared_predictor);EXPECT_EQ(parsed.shared_predictor->path(),cell.model.shared_predictor->path());
        EXPECT_EQ(parsed.mtp.draft_tokens,cell.makeOrchestrationConfig(cell.model.model_path,0).mtp.draft_tokens);
        EXPECT_EQ(parsed.mtp.graph_capacity_draft_tokens,3);EXPECT_EQ(parsed.mtp.verify_mode,modelParityHttpMTPVerifyMode());
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
    }
    EXPECT_EQ(documents,4u);
}
/** @test Consumer identity and cap swaps cannot reuse another candidate or an equal scalar projection. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotBudgetIdentityRetainsUnequalConsumersAndZeroArm)
{
    const auto original=qwen4ExpCombinedLocalHotC0Cases().front();
    auto cell=qwen4ExpLocalHotBudgetCase(original,Qwen4ExpLocalHotBudgetProfile::CombinedUnequal);
    const auto name=cell.testName();const auto runtime=modelParityServerConfiguration(cell);
    ASSERT_TRUE(runtime.ngram_table_policy.gpu_adaptive.local_hot_cache);
    auto& overrides=cell.ngram_table_policy->gpu_adaptive.local_hot_cache->consumer_overrides;
    ASSERT_GE(overrides.size(),2u);std::swap(overrides.front().cache_physical_byte_limit,overrides.back().cache_physical_byte_limit);
    EXPECT_NE(cell.testName(),name);EXPECT_NE(cell.ngram_table_policy,runtime.ngram_table_policy);
    const auto zero=qwen4ExpLocalHotBudgetCase(original,Qwen4ExpLocalHotBudgetProfile::CombinedZeroPositive);
    EXPECT_NE(zero.testName(),name);EXPECT_NE(zero.testName(),original.testName());
    auto bad=*zero.ngram_table_policy->gpu_adaptive.local_hot_cache;
    bad.consumer_overrides.push_back(bad.consumer_overrides.front());EXPECT_THROW(bad.validate(),std::invalid_argument);
    bad=*zero.ngram_table_policy->gpu_adaptive.local_hot_cache;bad.consumer_overrides.front().participant.ordinal=std::numeric_limits<int>::max();
    EXPECT_THROW(bad.requireConsumers(qwen4ExpLocalHotOriginalConsumers(zero)),std::invalid_argument);
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test The deliberate recurrent subset adds six requests without expanding initial or benchmark policy products. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotRecurrentMembershipPreservesOriginalR3Policies)
{
    const auto cases=qwen4ExpLocalHotRecurrentCases();
    ASSERT_EQ(cases.size(),6u);
    EXPECT_EQ(qwen4ExpCombinedLocalHotC0Cases().size(),2u);
    EXPECT_EQ(qwen4ExpLocalHotBudgetCases().size(),10u);
    EXPECT_EQ(qwen4ExpLocalHotBudgetOrdinaryCases().size(),5u);
    EXPECT_EQ(qwen4ExpLocalHotBudgetExactCases().size(),5u);
    const auto original=qwen4ExpOwnerSubsetExecutionCases(ModelParityExpertOverlayPolicy{});
    std::set<std::string> identities;
    std::map<ModelParityMTP,std::size_t> modes;
    for(const auto& cell:cases)
    {
        EXPECT_TRUE(identities.insert(cell.testName()).second);
        EXPECT_TRUE(cell.mtp==ModelParityMTP::Depth3 || cell.mtp==ModelParityMTP::DynamicDepth);
        ++modes[cell.mtp];
        EXPECT_EQ(cell.retained_mtp_draft_capacity,3);
        EXPECT_EQ(cell.model.maximum_mtp_draft_depth,3);
        EXPECT_EQ(cell.model.shared_predictor,std::optional{SharedPredictorGGUFSource{std::string(kPredictorModelPath)}});
        EXPECT_EQ(cell.expert_overlay,std::optional{ModelParityExpertOverlayPolicy{}});
        const auto& gpu=cell.ngram_table_policy->gpu_adaptive;
        EXPECT_EQ(gpu.source_backing,AdaptiveImmutableRowBacking::RAM);
        EXPECT_EQ(gpu.placement.mode,AdaptiveGPUCachePlacementMode::DeclaredRankLocal);
        ASSERT_TRUE(gpu.local_hot_cache);
        const auto seed=std::find_if(original.begin(),original.end(),[&](const auto& value){
            return value.mtp==cell.mtp && cell.topology.test_id.starts_with(value.topology.test_id+"_");});
        ASSERT_NE(seed,original.end());
        auto restored=cell;
        restored.ngram_table_policy=seed->ngram_table_policy;
        restored.topology.test_id=seed->topology.test_id;
        EXPECT_EQ(::testing::PrintToString(restored),::testing::PrintToString(*seed));
        EXPECT_EQ(cell.makeOrchestrationConfig(cell.model.model_path,0).ngram_table_policy,*cell.ngram_table_policy);
    }
    EXPECT_EQ(modes[ModelParityMTP::Depth3],3u);
    EXPECT_EQ(modes[ModelParityMTP::DynamicDepth],3u);
    EXPECT_THROW(qwen4ExpCombinedLocalHotC0CasesForMTP({}),std::invalid_argument);
    EXPECT_THROW(qwen4ExpCombinedLocalHotC0CasesForMTP({ModelParityMTP::Depth3,ModelParityMTP::Depth3}),std::invalid_argument);
    EXPECT_THROW(qwen4ExpCombinedLocalHotC0CasesForMTP({static_cast<ModelParityMTP>(255)}),std::invalid_argument);
}
/** @test Lossless configuration preserves recurrent MTP and exact unequal consumer ceilings through the public parser. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotRecurrentLosslessRuntimePreservesOriginalMTP)
{
    std::size_t documents=0;
    for(const auto& cell:qwen4ExpLocalHotRecurrentCases())
    {
        const auto expected=modelParityServerConfiguration(cell);
        const auto record=nlohmann::json::parse(::testing::PrintToString(cell));
        const auto parsed=[&]{
            if(modelParityRuntimeRepresentation(cell)==ModelParityRuntimeRepresentation::Arguments)
                return sourcePublicConfig(cell);
            ++documents;
            EXPECT_THROW(modelParityServerArguments(cell),std::invalid_argument);
            EXPECT_FALSE(record["runtime"].contains("server_args"));
            EXPECT_EQ(record["runtime"]["configuration_policy"],"lossless_declared_document_v1");
            const auto serialized=record["runtime"]["configuration_document"].dump();
            const auto path=std::filesystem::temp_directory_path()/("qfn-recurrent-local-cache-"+std::to_string(::getpid())+"-"+std::to_string(documents)+".json");
            struct Remove {std::filesystem::path path;~Remove(){std::error_code ignored;std::filesystem::remove(path,ignored);}} remove{path};
            {std::ofstream out(path);if(!out)throw std::runtime_error("Cannot write immutable recurrent config fixture");out<<serialized;}
            std::vector<std::string> args{"llaminar2","--config",path.string()};
            std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
            return OrchestrationConfigParser{}.parseArgs(int(argv.size()),argv.data());
        }();
        EXPECT_EQ(parsed.ngram_table_policy,expected.ngram_table_policy);
        EXPECT_EQ(nlohmann::json::parse(serializeOrchestrationConfig(parsed)).at("configuration").at("mtp"),
                  nlohmann::json::parse(serializeOrchestrationConfig(expected)).at("configuration").at("mtp"));
        EXPECT_EQ(parsed.mtp.graph_capacity_draft_tokens,3);
        EXPECT_EQ(parsed.mtp.verify_mode,modelParityHttpMTPVerifyMode());
        EXPECT_EQ(parsed.model_path,expected.model_path);
        EXPECT_EQ(parsed.shared_predictor,expected.shared_predictor);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
        auto changed=cell;
        changed.mtp=cell.mtp==ModelParityMTP::Depth3?ModelParityMTP::DynamicDepth:ModelParityMTP::Depth3;
        EXPECT_NE(changed.testName(),cell.testName());
    }
    EXPECT_EQ(documents,2u);
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test Generic original-seed selection preserves CUDA source/roles and the unchanged ROCm families. */
TEST(Qwen4ExpSourceAdmissionCases, CUDALocalHotCandidatesPreserveOriginalSourceAndConfiguration)
{
    const auto cases=qwen4ExpCUDALocalHotCases();
    ASSERT_EQ(cases.size(),4u);
    EXPECT_EQ(qwen4ExpCUDALocalHotOrdinaryCases().size(),2u);
    EXPECT_EQ(qwen4ExpCUDALocalHotExactCases().size(),2u);
    EXPECT_EQ(qwen4ExpCombinedLocalHotC0Cases().size(),2u);
    EXPECT_EQ(qwen4ExpLocalHotBudgetCases().size(),10u);
    EXPECT_EQ(qwen4ExpLocalHotRecurrentCases().size(),6u);
    const auto originals=qwen4ExpOwnerSubsetExecutionCases(ModelParityExpertOverlayPolicy{});
    std::set<std::string> names;
    std::map<ModelParityMTP,std::size_t> modes;
    std::size_t c0=0,equal=0;
    for (const auto& cell:cases)
    {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        ++modes[cell.mtp];
        const auto original=std::find_if(originals.begin(),originals.end(),[&](const auto& value){
            return value.mtp==cell.mtp && cell.topology.test_id.starts_with(value.topology.test_id+"_");});
        ASSERT_NE(original,originals.end());
        auto restored=cell;
        restored.topology.test_id=original->topology.test_id;
        restored.ngram_table_policy=original->ngram_table_policy;
        EXPECT_EQ(::testing::PrintToString(restored),::testing::PrintToString(*original));
        const auto& g=cell.ngram_table_policy->gpu_adaptive;
        EXPECT_EQ(g.source_backing,AdaptiveImmutableRowBacking::RAM);
        ASSERT_EQ(g.placement.owners.size(),1u);
        EXPECT_EQ(g.placement.owners.front().participant.device(),DeviceId::cuda(1));
        ASSERT_TRUE(g.placement.coordinator);
        EXPECT_EQ(g.placement.coordinator->device(),DeviceId::cuda(0));
        ASSERT_TRUE(g.local_hot_cache);
        std::size_t optional=0;
        for(const auto& owner:g.placement.owners)optional+=owner.cache_physical_byte_limit;
        for(const auto& consumer:qwen4ExpLocalHotOriginalConsumers(cell))
            optional+=g.local_hot_cache->ceilingsFor(consumer).cache_physical_byte_limit;
        if(cell.topology.test_id.ends_with("_CombinedLocalHotC0Candidate"))
        {
            ++c0;EXPECT_EQ(optional,0u);
            EXPECT_EQ(g.placement.owners.front().maximum_cache_rows,0u);
        }
        else {++equal;EXPECT_EQ(optional,std::size_t{128}<<20);}
        const auto parsed=sourcePublicConfig(cell);
        EXPECT_EQ(parsed.ngram_table_policy,*cell.ngram_table_policy);
        EXPECT_EQ(parsed.mtp.graph_capacity_draft_tokens,3);
        EXPECT_EQ(parsed.mtp.draft_tokens,cell.makeOrchestrationConfig(cell.model.model_path,0).mtp.draft_tokens);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
    }
    EXPECT_EQ(c0,2u);EXPECT_EQ(equal,2u);
    EXPECT_EQ(modes[ModelParityMTP::Off],2u);EXPECT_EQ(modes[ModelParityMTP::Depth1],2u);
    EXPECT_THROW(qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
        {Qwen4ExpHardwareTarget::CUDA2,0,AdaptiveImmutableRowBacking::RAM,
         Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Static},
        {ModelParityMTP::Depth1}),std::logic_error);
    EXPECT_THROW(qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
        {Qwen4ExpHardwareTarget::CUDA2,2,AdaptiveImmutableRowBacking::RAM,
         Qwen4ExpOwnerSubsetCachePressure::Positive,ModelParityExpertMovement::Dynamic},
        {ModelParityMTP::Depth1}),std::invalid_argument);
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test The actual source-factory policy accepts original Q6 ownership, preserving CPU controls and native-only errors. */
TEST(Qwen4ExpSourceAdmissionCases, OriginalHostedSourceUsesActualMixedGPUContract)
{
    using Policy=llaminar2::qwen4exp::PooledImmutableParticipantExecutionPolicy;
    const auto definition=qwen4ExpAllGPUQ6MTPDefinition();
    const auto original=resolveMoEExpertOverlayExecutionPlan(definition.topology.expert_overlay_plan,
        {.current_world_rank=0,.world_size=1});
    ASSERT_EQ(original.world_size,1);
    ASSERT_TRUE(original.current_rank.ownsTransactionAuthority());
    ASSERT_EQ(original.generationExecutionDomain(),DeviceGenerationExecutionDomain::HostedBoundary);
    for(const auto& domain:original.domains)for(const auto& participant:domain.participants)
        ASSERT_TRUE(participant.local_device.is_gpu());
    for(int repeat=0;repeat<20;++repeat)
    {
        EXPECT_NO_THROW(Policy::requireHostedSourceExecution(original));
        auto foreign=original;foreign.world_size=2;
        EXPECT_THROW(Policy::requireHostedSourceExecution(foreign),std::invalid_argument);
        foreign=original;foreign.current_rank.execution_kind=OverlayRankExecutionKind::RelayOnly;
        EXPECT_THROW(Policy::requireHostedSourceExecution(foreign),std::invalid_argument);
        foreign=original;foreign.domains.pop_back();
        EXPECT_THROW(Policy::requireHostedSourceExecution(foreign),std::invalid_argument);
    }
    const auto legacy=qwen4ExpCombinedLocalHotC0Cases().front().topology.expert_overlay_plan;
    const auto control=resolveMoEExpertOverlayExecutionPlan(legacy,{.current_world_rank=0,.world_size=1});
    EXPECT_NO_THROW(Policy::requireHostedSourceExecution(control));
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test The combined Q6 projection preserves the actual GPU-only parent, follower subset and original movement policy. */
TEST(Qwen4ExpSourceAdmissionCases, AllGPUQ6LocalHotCandidatesPreserveOriginalOwnerAndRuntimeContracts)
{
    const auto cases=qwen4ExpAllGPUQ6LocalHotCases();ASSERT_EQ(cases.size(),4u);
    EXPECT_EQ(qwen4ExpAllGPUQ6LocalHotOrdinaryCases().size(),2u);
    EXPECT_EQ(qwen4ExpAllGPUQ6LocalHotExactCases().size(),2u);
    const auto& originals=qwen4ExpAllGPUQ6MTPNativeExactCases();
    const auto fixed=std::find_if(originals.begin(),originals.end(),[](const auto& cell){return cell.mtp==ModelParityMTP::Depth1;});
    ASSERT_NE(fixed,originals.end());std::set<std::string> names;std::size_t zero=0,positive=0;
    for(const auto& cell:cases)
    {
        EXPECT_TRUE(names.insert(cell.testName()).second);
        auto restored=cell;restored.topology.test_id=fixed->topology.test_id;
        restored.ngram_table_policy=fixed->ngram_table_policy;restored.mtp=fixed->mtp;
        EXPECT_EQ(::testing::PrintToString(restored),::testing::PrintToString(*fixed));
        EXPECT_EQ(cell.retained_mtp_draft_capacity,3);
        ASSERT_TRUE(cell.expert_overlay);EXPECT_EQ(cell.expert_overlay->movement,ModelParityExpertMovement::Dynamic);
        const auto& plan=*cell.topology.expert_overlay_plan;
        EXPECT_EQ(resolveMoEOverlayAuthorityExecutionKind(plan),MoEOverlayAuthorityExecutionKind::DeviceResident);
        for(const auto& participant:cell.topology.participants)EXPECT_FALSE(participant.address.isCPU());
        const auto resolved=resolveMoEExpertOverlayExecutionPlan(cell.topology.expert_overlay_plan,{.current_world_rank=0,.world_size=1});
        EXPECT_NO_THROW(llaminar2::qwen4exp::PooledImmutableParticipantExecutionPolicy::requireHostedSourceExecution(resolved));
        const auto& g=cell.ngram_table_policy->gpu_adaptive;EXPECT_EQ(g.source_backing,AdaptiveImmutableRowBacking::RAM);
        ASSERT_TRUE(g.local_hot_cache);ASSERT_EQ(g.placement.owners.size(),2u);
        EXPECT_EQ(g.placement.owners[0].participant.device(),DeviceId::rocm(0));
        EXPECT_EQ(g.placement.owners[1].participant.device(),DeviceId::rocm(2));
        const auto consumers=qwen4ExpLocalHotOriginalConsumers(cell);
        std::size_t bytes=0;for(const auto& owner:g.placement.owners)bytes+=owner.cache_physical_byte_limit;
        for(const auto& consumer:consumers)bytes+=g.local_hot_cache->ceilingsFor(consumer).cache_physical_byte_limit;
        if(cell.topology.test_id.ends_with("_CombinedLocalHotC0Candidate")){++zero;EXPECT_EQ(bytes,0u);}
        else {++positive;EXPECT_EQ(bytes,std::size_t{128}<<20);}
        const auto config=sourcePublicConfig(cell);
        EXPECT_EQ(config.ngram_table_policy,*cell.ngram_table_policy);
        EXPECT_EQ(config.mtp.graph_capacity_draft_tokens,3);
        EXPECT_EQ(config.moe_routed_expert_plan->residency_policy,RoutedExpertResidencyPolicy::RoutedTierRebalanced);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(config).empty());
    }
    EXPECT_EQ(zero,2u);EXPECT_EQ(positive,2u);
}
}

namespace llaminar2::test::parity::qwen4exp {
/** @test SSD candidates preserve original bounded backing and lossless local-only caps. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotSSDProjectionPreservesBoundedSourceAndConsumerOwnership)
{
    const auto cases = qwen4ExpLocalHotSSDCases();
    ASSERT_EQ(cases.size(), 4u);
    EXPECT_EQ(qwen4ExpLocalHotSSDOrdinaryCases().size(), 2u);
    EXPECT_EQ(qwen4ExpLocalHotSSDExactCases().size(), 2u);
    std::set<std::string> identities;
    for (const auto& cell : cases)
    {
        EXPECT_TRUE(identities.insert(cell.testName()).second);
        const auto& policy = *cell.ngram_table_policy;
        const auto& gpu = policy.gpu_adaptive;
        EXPECT_EQ(gpu.source_backing, AdaptiveImmutableRowBacking::SSD);
        EXPECT_GT(policy.window_bytes, 0u);
        EXPECT_GT(policy.window_count, 0u);
        const auto consumers = qwen4ExpLocalHotOriginalConsumers(cell);
        ASSERT_EQ(consumers.size(), 2u);
        ASSERT_TRUE(gpu.local_hot_cache);
        gpu.local_hot_cache->requireConsumers(consumers);
        EXPECT_EQ(gpu.local_hot_cache->ceilingsFor(consumers.front()).cache_physical_byte_limit, 0u);
        EXPECT_EQ(gpu.local_hot_cache->ceilingsFor(consumers.back()).cache_physical_byte_limit, std::size_t{64} << 20);
        ASSERT_EQ(gpu.placement.owners.size(), 1u);
        EXPECT_EQ(gpu.placement.owners.front().participant, consumers.front());
        EXPECT_EQ(gpu.placement.owners.front().cache_physical_byte_limit, std::size_t{64} << 20);
        const auto target = consumers.front().backend == DeviceType::ROCm
            ? Qwen4ExpHardwareTarget::ROCm2 : Qwen4ExpHardwareTarget::CUDA2;
        const auto pressure = target == Qwen4ExpHardwareTarget::ROCm2
            ? Qwen4ExpOwnerSubsetCachePressure::Positive : Qwen4ExpOwnerSubsetCachePressure::Zero;
        const auto originals = qwen4ExpCombinedLocalHotC0CasesFromOriginalSeed(
            {target, 1, AdaptiveImmutableRowBacking::SSD, pressure, ModelParityExpertMovement::Static},
            {ModelParityMTP::Off, ModelParityMTP::Depth1});
        const auto original = std::find_if(originals.begin(), originals.end(),
            [&](const auto& value) { return value.mtp == cell.mtp; });
        ASSERT_NE(original, originals.end());
        auto restored = cell;
        restored.ngram_table_policy = original->ngram_table_policy;
        restored.topology.test_id = original->topology.test_id;
        EXPECT_EQ(::testing::PrintToString(restored), ::testing::PrintToString(*original));
        EXPECT_EQ(policy.window_bytes, original->ngram_table_policy->window_bytes);
        EXPECT_EQ(policy.window_count, original->ngram_table_policy->window_count);
        EXPECT_EQ(gpu.host, original->ngram_table_policy->gpu_adaptive.host);
        const auto exported = nlohmann::json::parse(::testing::PrintToString(cell));
        ASSERT_TRUE(exported["runtime"].contains("configuration_document"));
        const auto parsed = deserializeOrchestrationConfig(exported["runtime"]["configuration_document"].dump());
        EXPECT_EQ(parsed.ngram_table_policy, policy);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(parsed).empty());
    }
}
}

#include "integration/parity/qwen4exp/Qwen4ExpLocalPairedCases.h"
namespace llaminar2::test::parity::qwen4exp {
/** @test Explicit caps change only authored optional ceilings and retain numerical admission. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotExplicitBudgetCandidatesPreserveOriginalPolicies)
{
    const auto cells = qwen4ExpLocalHotMeasuredBudgetCandidates();
    ASSERT_EQ(cells.size(), 2u);
    const auto seed = qwen4ExpCombinedLocalHotC0CasesForMTP({ModelParityMTP::Off}).front();
    const std::array profiles{Qwen4ExpLocalHotBudgetProfile::ConsumerLocal,
        Qwen4ExpLocalHotBudgetProfile::CombinedEqual};
    for (std::size_t index = 0; index < cells.size(); ++index)
    {
        const auto& cell = cells[index];
        const auto original = qwen4ExpLocalHotBudgetCase(seed, profiles[index]);
        auto restored = cell;
        restored.topology.test_id = original.topology.test_id;
        auto& restored_gpu = restored.ngram_table_policy->gpu_adaptive;
        const auto& original_gpu = original.ngram_table_policy->gpu_adaptive;
        if (index == 0)
            restored_gpu.gpu_cache_physical_byte_limit = original_gpu.gpu_cache_physical_byte_limit;
        else
        {
            ASSERT_EQ(restored_gpu.placement.owners.size(), 1u);
            restored_gpu.placement.owners.front().cache_physical_byte_limit =
                original_gpu.placement.owners.front().cache_physical_byte_limit;
            restored_gpu.local_hot_cache->cache_physical_byte_limit =
                original_gpu.local_hot_cache->cache_physical_byte_limit;
        }
        EXPECT_EQ(::testing::PrintToString(restored), ::testing::PrintToString(original));
        EXPECT_EQ(cell.mtp, ModelParityMTP::Off);
        EXPECT_EQ(cell.ngram_table_policy->gpu_adaptive.source_backing, AdaptiveImmutableRowBacking::RAM);
        EXPECT_EQ(cell.testName().find("MatchedOptional"), std::string::npos);
        EXPECT_NE(cell.testName().find("Candidate"), std::string::npos);
        EXPECT_NE(cell.testName().find("AuthoredOptionalBudgets"), std::string::npos);
        const auto record = nlohmann::json::parse(::testing::PrintToString(cell));
        EXPECT_EQ(sourcePublicConfig(cell).ngram_table_policy, *cell.ngram_table_policy);
        EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(
            cell.makeOrchestrationConfig(cell.model.model_path, 0)).empty());
        const auto id = qfnLocalPairedCellId("Qwen4ExpLocalHotMeasuredBudgetNumerical", cell);
        EXPECT_EQ(id, "Qwen4ExpLocalHotMeasuredBudgetNumerical/Qwen4ExpLocalHotNumericalTest.ProductionParity/" + cell.testName());
        EXPECT_THROW(qfnRequireLocalHotNumericalBinding(cell), std::invalid_argument);
    }
    const auto& consumer = cells[0].ngram_table_policy->gpu_adaptive;
    EXPECT_EQ(consumer.gpu_cache_physical_byte_limit, 104697856u);
    EXPECT_FALSE(consumer.local_hot_cache);
    const auto& combined = cells[1].ngram_table_policy->gpu_adaptive;
    EXPECT_EQ(combined.placement.owners.front().cache_physical_byte_limit, 71344128u);
    EXPECT_EQ(combined.local_hot_cache->cache_physical_byte_limit, 35672064u);
    EXPECT_EQ(qwen4ExpLocalHotBudgetCases().size(), 10u);
    EXPECT_THROW(qwen4ExpLocalHotBudgetCase(seed, profiles[0],
        (Qwen4ExpLocalHotExplicitOptionalBudgets{1, 2})), std::invalid_argument);
    EXPECT_THROW(qwen4ExpLocalHotBudgetCase(seed, Qwen4ExpLocalHotBudgetProfile::SharedOnly,
        (Qwen4ExpLocalHotExplicitOptionalBudgets{1, 2})), std::invalid_argument);
    EXPECT_THROW(qwen4ExpLocalHotBudgetCase(seed, Qwen4ExpLocalHotBudgetProfile::CombinedUnequal,
        (Qwen4ExpLocalHotExplicitOptionalBudgets{1, 2})), std::invalid_argument);
}
/** @brief Dynamic public generation consumes the original sustained declaration; static stays short. */
TEST(Qwen4ExpSourceAdmissionCases, OrdinaryGenerationBindsDeclaredMovementHorizonAndInput)
{
    const std::vector<int32_t> checkpoint{7,11,13};
    SamplingParams short_sampling;
    short_sampling.temperature=0;short_sampling.top_k=1;short_sampling.top_p=1;short_sampling.seed=19;
    auto dynamic=qwen4ExpAllGPUQ6LocalHotOrdinaryCases().front();
    ASSERT_TRUE(dynamic.requiresPhysicalExpertMovement());
    const auto original=::testing::PrintToString(dynamic);
    unsigned encodes=0;
    const auto encoder=[&](const auto& messages,bool add,const std::string& tools,bool thinking){
        ++encodes;EXPECT_TRUE(add);EXPECT_TRUE(tools.empty());EXPECT_FALSE(thinking);
        EXPECT_EQ(messages.size(),2u);
        EXPECT_EQ(messages[0].role,"system");EXPECT_EQ(messages[0].content,dynamic.model.generation_prompt.system());
        EXPECT_EQ(messages[1].role,"user");EXPECT_EQ(messages[1].content,dynamic.model.generation_prompt.user());
        return std::vector<int>{21,22,23,24};
    };
    const auto workload=Qwen4ExpOrdinaryGenerationWorkload::fromOriginalCase(dynamic,checkpoint,short_sampling,encoder);
    EXPECT_EQ(encodes,1u);EXPECT_TRUE(workload.sustained());EXPECT_EQ(workload.originalCase(),dynamic.testName());
    const auto first=workload.request(Qwen4ExpOrdinaryRequestPhase::First);
    const auto reuse=workload.request(Qwen4ExpOrdinaryRequestPhase::Reuse);
    EXPECT_EQ(first.tokens,(std::vector<int32_t>{21,22,23,24}));EXPECT_EQ(first.tokens,reuse.tokens);
    EXPECT_EQ(first.maximum_tokens,dynamic.generation_workload.maximumTokens());
    EXPECT_EQ(first.minimum_tokens,dynamic.generation_workload.minimumTokens());
    EXPECT_EQ(first.maximum_tokens,384);EXPECT_EQ(reuse.maximum_tokens,first.maximum_tokens);
    EXPECT_EQ(reuse.minimum_tokens,first.minimum_tokens);EXPECT_EQ(first.sampling,reuse.sampling);
    EXPECT_FLOAT_EQ(first.sampling.temperature,float(kModelParityGenerationTemperature));
    EXPECT_EQ(first.sampling.top_k,kModelParityGenerationTopK);
    EXPECT_FLOAT_EQ(first.sampling.top_p,float(kModelParityGenerationTopP));
    EXPECT_EQ(first.sampling.seed,dynamic.model.generation_seed.value());
    EXPECT_EQ(::testing::PrintToString(dynamic),original);
    const auto exported=nlohmann::json::parse(original);
    const auto& request=exported.at("runtime").at("generation").at("requests")[0].at("body");
    EXPECT_EQ(request.at("max_tokens"),first.maximum_tokens);
    EXPECT_EQ(request.at("messages")[0].at("content"),dynamic.model.generation_prompt.system());
    EXPECT_EQ(request.at("messages")[1].at("content"),dynamic.model.generation_prompt.user());
    EXPECT_EQ(request.at("seed"),first.sampling.seed);
    EXPECT_DOUBLE_EQ(request.at("temperature").get<double>(),kModelParityGenerationTemperature);
    EXPECT_EQ(request.at("top_k"),first.sampling.top_k);
    EXPECT_DOUBLE_EQ(request.at("top_p").get<double>(),kModelParityGenerationTopP);

    auto statics=qwen4ExpLocalHotBudgetCases();
    const auto found=std::find_if(statics.begin(),statics.end(),[](const auto& cell){return cell.mtp==ModelParityMTP::Off;});
    ASSERT_NE(found,statics.end());ASSERT_FALSE(found->requiresPhysicalExpertMovement());
    const auto static_original=::testing::PrintToString(*found);
    const auto short_workload=Qwen4ExpOrdinaryGenerationWorkload::fromOriginalCase(*found,checkpoint,short_sampling,encoder);
    EXPECT_FALSE(short_workload.sustained());EXPECT_EQ(encodes,1u);
    const auto short_first=short_workload.request(Qwen4ExpOrdinaryRequestPhase::First);
    const auto short_reuse=short_workload.request(Qwen4ExpOrdinaryRequestPhase::Reuse);
    EXPECT_EQ(short_first.tokens,checkpoint);EXPECT_EQ(short_first.sampling,short_sampling);
    EXPECT_EQ(short_first.maximum_tokens,3);EXPECT_EQ(short_first.minimum_tokens,3);
    EXPECT_EQ(short_reuse.maximum_tokens,2);EXPECT_EQ(short_reuse.minimum_tokens,2);
    EXPECT_EQ(::testing::PrintToString(*found),static_original);

    dynamic.generation_workload=ModelParityGenerationWorkload(512,384);
    const auto longer=Qwen4ExpOrdinaryGenerationWorkload::fromOriginalCase(dynamic,checkpoint,short_sampling,encoder);
    EXPECT_EQ(longer.request(Qwen4ExpOrdinaryRequestPhase::First).maximum_tokens,512);
    EXPECT_EQ(longer.request(Qwen4ExpOrdinaryRequestPhase::Reuse).minimum_tokens,384);
    EXPECT_THROW(workload.request(static_cast<Qwen4ExpOrdinaryRequestPhase>(9)),std::invalid_argument);
    EXPECT_THROW(Qwen4ExpOrdinaryGenerationWorkload::fromOriginalCase(dynamic,checkpoint,short_sampling,
        [](const auto&,bool,const auto&,bool){return std::vector<int>{};}),std::invalid_argument);
    dynamic.mtp=ModelParityMTP::Depth1;
    EXPECT_THROW(Qwen4ExpOrdinaryGenerationWorkload::fromOriginalCase(dynamic,checkpoint,short_sampling,encoder),std::invalid_argument);
}

}

namespace llaminar2::test::parity::qwen4exp {
/** @test Refinement changes only exact optional caps and preserves prior candidates/paired guards. */
TEST(Qwen4ExpSourceAdmissionCases, LocalHotRefinedBudgetPreservesMeasuredPolicies)
{
    const auto refined = qwen4ExpLocalHotRefinedBudgetCandidates();
    ASSERT_EQ(refined.size(), 1u);
    const auto previous = qwen4ExpLocalHotMeasuredBudgetCandidates();
    ASSERT_EQ(previous.size(), 2u);
    EXPECT_EQ(previous[0].ngram_table_policy->gpu_adaptive.gpu_cache_physical_byte_limit, 104697856u);
    auto restored = refined.front();
    auto& gpu = restored.ngram_table_policy->gpu_adaptive;
    EXPECT_EQ(gpu.placement.owners.front().cache_physical_byte_limit, 70432384u);
    EXPECT_EQ(gpu.local_hot_cache->cache_physical_byte_limit, 35216192u);
    EXPECT_EQ(gpu.placement.owners.front().cache_physical_byte_limit,
        2 * gpu.local_hot_cache->cache_physical_byte_limit);
    gpu.placement.owners.front().cache_physical_byte_limit =
        previous[1].ngram_table_policy->gpu_adaptive.placement.owners.front().cache_physical_byte_limit;
    gpu.local_hot_cache->cache_physical_byte_limit =
        previous[1].ngram_table_policy->gpu_adaptive.local_hot_cache->cache_physical_byte_limit;
    restored.topology.test_id = previous[1].topology.test_id;
    EXPECT_EQ(::testing::PrintToString(restored), ::testing::PrintToString(previous[1]));
    EXPECT_EQ(qwen4ExpLocalHotBudgetCases().size(), 10u);
    EXPECT_EQ(qwen4ExpLocalHotNumericalCases().size(), 6u);
    const auto key = qfnLocalPairedCellId("Qwen4ExpLocalHotRefinedBudgetNumerical", refined.front());
    EXPECT_EQ(key, "Qwen4ExpLocalHotRefinedBudgetNumerical/Qwen4ExpLocalHotNumericalTest.ProductionParity/" + refined.front().testName());
    auto deliberately_unbound = refined.front();
    deliberately_unbound.paired_workload.reset();
    EXPECT_THROW(qfnRequireLocalHotNumericalBinding(deliberately_unbound), std::invalid_argument);
    EXPECT_TRUE(ConfigValidator::createStandard().validateToStrings(
        refined.front().makeOrchestrationConfig(refined.front().model.model_path, 0)).empty());
}
}
