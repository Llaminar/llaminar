/**
 * @file Test__Qwen4ExpSourceAdmissionCases.cpp
 * @brief Device-free coverage of exact QFN source-only topology and storage requests.
 *
 * These tests use the canonical expander, public CLI parser and configuration
 * validator without loading weights, consulting a reference pack or probing
 * accelerators. Physical fit and native materialization remain separate real
 * source-admission obligations; an absent backend or unsupported public policy
 * cannot be relabelled as a positive admission by these declaration tests.
 */
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpSourceAdmissionEvidence.h"
#include "integration/parity/qwen4exp/Qwen4ExpGPUAdaptiveParityCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpNgramAdmissionCases.h"
#include "integration/parity/qwen4exp/Qwen4ExpROCm4OverlayPolicyParityCases.h"
#include "integration/parity/ModelParityRuntimeExport.h"
#include "integration/parity/ParityHardwareRequirements.h"
#include "config/ConfigValidator.h"
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
        EXPECT_EQ(cell.model.maximum_mtp_draft_depth, original.model.maximum_mtp_draft_depth);
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

/**
 * @test Every existing source-admission identity retains all four MTP policies,
 * including candidate and precise-rejection families without reclassification.
 */
TEST(Qwen4ExpSourceAdmissionCases, FullSourceMatrixHasTotalFourPolicyMTPDimension)
{
    const std::array families{
        &qwen4ExpProductionCases(), &qwen4ExpHostAdaptiveProductionCases(),
        &qwen4ExpFixedHeadProductionCases(), &qwen4ExpGPUAdaptiveProductionCases(),
        &qwen4ExpFullROCmTP4ProductionCases(), &qwen4ExpROCm4OverlayPolicyProductionCases(),
        &qwen4ExpExtendedSourceAdmissionCases(), &qwen4ExpComposedCandidateSourceAdmissionCases(),
        &qwen4ExpRejectedSourceAdmissionCases(), &qwen4ExpNgramAdmissionCases()};
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
                EXPECT_EQ(cell.model.maximum_mtp_draft_depth, main.model.maximum_mtp_draft_depth);
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
    EXPECT_EQ(seeds, 151u);
    EXPECT_EQ(off, 151u);
    EXPECT_EQ(active, 453u);
    EXPECT_EQ(names.size(), 604u);
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
