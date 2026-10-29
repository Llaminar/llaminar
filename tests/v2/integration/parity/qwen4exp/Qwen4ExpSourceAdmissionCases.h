/**
 * @file Qwen4ExpSourceAdmissionCases.h
 * @brief Bounded gold-free setup coverage through canonical QFN case projection.
 *
 * Two CPU-NUMA blueprints and four unqualified composed TP/PP candidates add
 * four explicit host-table requests each. Twelve rejection cases retain unsupported internally sharded
 * expert requests across GPU-plus-CPU tiers, pairing RAM with fixed-head VRAM.
 * These are plain source-admission fixture inputs, never numerical campaign
 * registrations or claims that absent hardware or an unadmitted BOM is supported.
 * Reference metadata remains immutable provenance; the plain fixture does not
 * enter a numerical setup, open a reference pack, or generate gold.
 */
#pragma once

#include "Qwen4ExpFixedHeadParityCases.h"
#include "Qwen4ExpMTPAdmissionCases.h"
#include "backends/HardwareInventory.h"

#include <array>
#include <algorithm>
#include <limits>
#include <span>
#include <iterator>
#include <set>
#include <stdexcept>

namespace llaminar2::test::parity::qwen4exp
{
/** @brief Complete setup-only MTP dimension; numerical reference families are unchanged. */
inline constexpr std::array kQwen4ExpSourceAdmissionMTPPolicies{
    ModelParityMTP::Off, ModelParityMTP::Depth1,
    ModelParityMTP::Depth3, ModelParityMTP::DynamicDepth};

/** @brief Explicit source-only retained family, independent of numerical reference capacity. */
enum class Qwen4ExpSourceMTPEnvelope
{
    Full15,
    Retained3,
};

/**
 * @brief Expand every existing topology/storage/placement input over the same MTP dimension.
 * @param main_cases Original MTPOff declarations from any source-admission family.
 * @param envelope Explicit physical family; unchanged callers retain the full15 default.
 * @return Four cells per original, with Off identity/source closure preserved exactly.
 * @throws std::invalid_argument For an already-expanded or retained-active seed.
 * @throws std::overflow_error For an unrepresentable output count.
 *
 * Active cells explicitly select the existing original predictor source and
 * retain the selected explicit envelope (full15 by default). They do not change reference metadata,
 * declare numerical MTP support, or alter the fixture's positive/candidate/
 * rejection classification. No topology list or hardware decision lives here.
 */
inline std::vector<ModelParityCase> qwen4ExpExpandSourceAdmissionMTP(
    std::span<const ModelParityCase> main_cases,
    Qwen4ExpSourceMTPEnvelope envelope = Qwen4ExpSourceMTPEnvelope::Full15)
{
    if (envelope != Qwen4ExpSourceMTPEnvelope::Full15 && envelope != Qwen4ExpSourceMTPEnvelope::Retained3)
        throw std::invalid_argument("Unknown explicit QFN source MTP envelope");
    const int retained_depth = envelope == Qwen4ExpSourceMTPEnvelope::Full15
        ? kModelParityRequiredMaximumMTPDepth : 3;
    if (main_cases.size() > std::numeric_limits<size_t>::max() / kQwen4ExpSourceAdmissionMTPPolicies.size())
        throw std::overflow_error("QFN source-admission MTP case count overflows");
    std::vector<ModelParityCase> result;
    result.reserve(main_cases.size() * kQwen4ExpSourceAdmissionMTPPolicies.size());
    for (const auto &main : main_cases)
    {
        if (main.mtp != ModelParityMTP::Off || main.retained_mtp_draft_capacity != 0)
            throw std::invalid_argument("QFN source-admission MTP dimension requires unchanged MTPOff seeds");
        for (const auto policy : kQwen4ExpSourceAdmissionMTPPolicies)
        {
            auto cell = main;
            if (policy != ModelParityMTP::Off)
            {
                const Qwen4ExpMTPAdmissionCase paired{
                    .main = main,
                    .predictor = main.model.shared_predictor.value_or(
                        SharedPredictorGGUFSource{std::string(kPredictorModelPath)}),
                    .policy = policy, .retained_depth = retained_depth};
                (void)paired.requestedDepth();
                cell.model.shared_predictor = paired.predictor;
                cell.mtp = paired.policy;
                cell.retained_mtp_draft_capacity = paired.retained_depth;
            }
            result.push_back(std::move(cell));
        }
    }
    return result;
}

/**
 * @brief Project one setup-only MTP cell without changing its authored storage or topology.
 * @param cell Existing Off declaration or a cell from the shared four-policy expansion.
 * @param resolved_model_path Exact main source selected by the harness.
 * @param rank Owning runtime MPI rank.
 * @return Production configuration with its explicit active predictor and declared retention.
 * @throws std::invalid_argument For missing active predictor, unsupported policy or a changed envelope.
 *
 * The existing paired admission type owns the single source/depth projection.
 * Its explicit source-only purpose preserves every authored storage policy and
 * leaves the diagnostic/runtime projections unchanged. Dynamic keeps
 * initial_depth zero for the public production resolver; no inference or
 * controller execution is claimed by source setup.
 */
inline OrchestrationConfig qwen4ExpSourceAdmissionConfig(const ModelParityCase &cell,
    const std::string &resolved_model_path, int rank)
{
    if (std::find(kQwen4ExpSourceAdmissionMTPPolicies.begin(),
            kQwen4ExpSourceAdmissionMTPPolicies.end(), cell.mtp) == kQwen4ExpSourceAdmissionMTPPolicies.end())
        throw std::invalid_argument("QFN source-admission has an unsupported MTP policy");
    if (cell.mtp == ModelParityMTP::Off)
        return cell.makeOrchestrationConfig(resolved_model_path, rank);
    if (!cell.model.shared_predictor ||
        (cell.retained_mtp_draft_capacity != kModelParityRequiredMaximumMTPDepth &&
         cell.retained_mtp_draft_capacity != 3))
        throw std::invalid_argument("Active QFN source admission requires its explicit paired source and declared depth3 or depth15 envelope");
    auto main = cell;
    main.mtp = ModelParityMTP::Off;
    main.retained_mtp_draft_capacity = 0;
    const Qwen4ExpMTPAdmissionCase paired{.main = main, .predictor = *cell.model.shared_predictor,
        .policy = cell.mtp, .retained_depth = cell.retained_mtp_draft_capacity};
    return paired.sourceAdmissionConfig(Qwen4ExpMTPAdmissionExecution::Active, resolved_model_path, rank);
}

/**
 * @brief Select the existing exact CUDA TP2 VRAM Dynamic/Random declaration.
 * @return Independent source-only seed; existing numerical/positive cases stay unchanged.
 * @throws std::logic_error If the canonical producer no longer names exactly one seed.
 */
inline ModelParityCase qwen4ExpCUDAFixedHeadEightExpertSourceSeed()
{
    const std::array ordinals{0, 1};
    const auto candidates = expandModelParityDefinition(
        qwen4ExpFixedHeadParityDefinition(DeviceType::CUDA, ordinals));
    std::optional<ModelParityCase> selected;
    for (const auto &cell : candidates)
    {
        if (!cell.ngram_table_policy || !cell.ngram_table_policy->isFixedHeadSharded() ||
            !cell.expert_overlay || cell.expert_overlay->movement != ModelParityExpertMovement::Dynamic ||
            cell.expert_overlay->owner_order != RoutedExpertOwnerOrder::Random)
            continue;
        if (selected) throw std::logic_error("CUDA fixed-head source control has ambiguous canonical seeds");
        selected = cell;
    }
    if (!selected) throw std::logic_error("CUDA fixed-head source control lost its canonical seed");
    return std::move(*selected);
}

/**
 * @return One separately named four-expert CUDA TP2 family with retained depth3.
 *
 * This is an authored request, not an automatic response to capacity failure.
 * The Off control keeps inactive/no-retention semantics while retaining the
 * explicit family ceiling in its name. All active policies own the same depth3
 * backing envelope; the original full15 producers remain unchanged.
 */
inline const std::vector<ModelParityCase> &qwen4ExpCUDAFixedHeadFourExpertRetained3SourceAdmissionCases()
{
    static const auto cases = [] {
        auto seed = qwen4ExpCUDAFixedHeadEightExpertSourceSeed();
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(*seed.topology.expert_overlay_plan);
        plan->routed_tiers.front().max_experts_per_layer = 4;
        seed.topology.expert_overlay_plan = std::move(plan);
        const auto marker = seed.topology.test_id.rfind("GpuExperts8");
        if (marker == std::string::npos) throw std::logic_error("CUDA fixed-head seed lost its explicit grant identity");
        seed.topology.test_id.replace(marker, std::string("GpuExperts8").size(), "GpuExperts4");
        seed.topology.test_id += "_RetainedMTP3";
        const std::array seeds{seed};
        return qwen4ExpExpandSourceAdmissionMTP(seeds, Qwen4ExpSourceMTPEnvelope::Retained3);
    }();
    return cases;
}

/** @brief Explicit public capacity input, distinct from unsupported policy rejection. */
struct Qwen4ExpSourceCapacityAdmissionCase
{
    ModelParityCase cell;
    std::size_t gpu_memory_cap_mb;
    /** @return Existing source identity including the authored capacity-control suffix. */
    std::string testName() const { return cell.testName(); }
};

/**
 * @return The unchanged eight-expert request under an explicit thirteen-GiB public cap.
 *
 * max_gpu_memory_mb uses binary MiB. Each original fixed head shard alone is
 * larger than thirteen GiB, so the cap makes rejection independent of larger
 * installed GPU capacities. Thirteen GiB also admits the actual smallest
 * full15 pre-table BOM, so this control reaches the required table-capacity
 * rejection instead of accepting an unrelated earlier allocation failure.
 * Every MTP policy keeps its original retained envelope.
 */
inline const std::vector<Qwen4ExpSourceCapacityAdmissionCase> &qwen4ExpCUDAFixedHeadCapacityRejectionCases()
{
    static const auto cases = [] {
        auto seed = qwen4ExpCUDAFixedHeadEightExpertSourceSeed();
        seed.topology.test_id += "_Capacity13GiB";
        const std::array seeds{seed};
        std::vector<Qwen4ExpSourceCapacityAdmissionCase> result;
        for (auto &cell : qwen4ExpExpandSourceAdmissionMTP(seeds))
            result.push_back({std::move(cell), 13u * 1024u});
        return result;
    }();
    return cases;
}

/**
 * @brief Apply an explicitly declared public memory ceiling before admission.
 * @param cell Exact source request, unchanged in topology, placement and MTP policy.
 * @param model_path Harness-resolved original main source.
 * @param rank Owning runtime rank.
 * @param cap_mb Optional authored binary-MiB GPU cap; absent preserves original config.
 * @return Canonical source configuration with only its explicit ceiling added.
 * @throws std::invalid_argument For a zero ceiling.
 */
inline OrchestrationConfig qwen4ExpSourceAdmissionCapacityConfig(const ModelParityCase &cell,
    const std::string &model_path, int rank, std::optional<std::size_t> cap_mb = std::nullopt)
{
    auto config = qwen4ExpSourceAdmissionConfig(cell, model_path, rank);
    if (cap_mb)
    {
        if (!*cap_mb) throw std::invalid_argument("Source-admission capacity ceiling must be positive");
        config.max_gpu_memory_mb = *cap_mb;
    }
    return config;
}

/**
 * @brief Authenticate exact rank-local CPU domains against frozen startup inventory.
 * @param cell Canonical source-admission participant request.
 * @param rank Current owning MPI rank, within the declared topology.
 * @param hardware Already-published process inventory; this function never detects hardware.
 * @throws std::invalid_argument For an invalid rank, ambiguous CPU address,
 *         duplicate NUMA locality, or locality absent from the observation.
 *
 * GlobalDeviceAddress::cpu(n) stores n in numa_node while device_ordinal stays
 * zero. Neither CPUBackend::deviceCount(), socket count nor worker count proves
 * the presence of an explicitly requested NUMA domain. The same production
 * inventory projection used for CPU admission must authenticate each address.
 */
inline void requireQwen4ExpSourceCPUInventory(
    const ModelParityCase &cell, int rank, const HardwareInventory &hardware)
{
    if (rank < 0 || rank >= cell.topology.mpi_ranks)
        throw std::invalid_argument("QFN source admission CPU inventory has an invalid owning rank");
    std::set<int> localities;
    for (const auto &participant : cell.topology.participants)
    {
        if (participant.world_rank != rank || participant.address.device_type != DeviceType::CPU)
            continue;
        const auto &address = participant.address;
        if (!address.hasValidNuma() || address.device_ordinal != 0)
            throw std::invalid_argument("QFN source admission requires an exact canonical CPU NUMA address");
        if (!localities.insert(address.numa_node).second)
            throw std::invalid_argument("QFN source admission cannot alias two CPU participants to one NUMA locality");
        // cpuDevice is a pure projection of this frozen observation. Passing
        // device_ordinal here would silently authenticate CPU1 as CPU NUMA0.
        (void)hardware.cpuDevice(address.numa_node);
    }
}

/**
 * @brief Declare explicit whole-host policies for a new physical composition.
 * @param mode One of the six CPU-NUMA/composed source-admission topologies.
 * @return Original model/source/precision blueprint with RAM, SSD and two cache ceilings.
 * @throws std::invalid_argument For a topology outside the reviewed bounded extension.
 */
inline ModelParityDefinition qwen4ExpComposedSourceAdmissionDefinition(
    Qwen4ExpParityTopology mode)
{
    switch (mode)
    {
    case Qwen4ExpParityTopology::CPU2TP:
    case Qwen4ExpParityTopology::CPU2PP:
    case Qwen4ExpParityTopology::ROCm4TP2PP2:
    case Qwen4ExpParityTopology::CUDA2CPU2TPPP:
    case Qwen4ExpParityTopology::ROCm2CPU2TPPP:
    case Qwen4ExpParityTopology::CUDA2ROCm2CPU2TPPP:
        break;
    default:
        throw std::invalid_argument("QFN composed source admission requires an explicit reviewed CPU2/TP-PP topology");
    }
    auto definition = qwen4ExpProductionDefinition(mode);
    const auto adaptive = qwen4ExpHostAdaptiveStorageAxes();
    definition.features.ngram_storage.insert(definition.features.ngram_storage.end(),
        adaptive.begin(), adaptive.end());
    return definition;
}

/**
 * @brief Declare an unsupported cross-tier shard request for exact rejection coverage.
 * @param backend Homogeneous CUDA or ROCm continuation backend.
 * @param ordinals Original ordered physical TP2/4/8 participants.
 * @return Existing fixed-head/RAM definition with the unsupported routed compute request retained.
 * @throws std::invalid_argument If either the existing topology or the modified public plan is invalid.
 *
 * The original eight-GPU-expert ceiling and uncapped final CPU tier remain
 * authored inputs. Production supports this shard policy only within one native
 * GPU tier; cross-tier slice publication must fail precisely, without silently
 * changing to apportioned experts. Table sharding is an independent request.
 */
inline ModelParityDefinition qwen4ExpInternallyShardedSourceAdmissionDefinition(
    DeviceType backend, std::span<const int> ordinals)
{
    auto definition = qwen4ExpFixedHeadParityDefinition(backend, ordinals);
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>(
        *definition.topology.expert_overlay_plan);
    plan->domains.front().routed_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
    if (!validateMoERoutedExpertPlacementPlan(*plan).ok())
        throw std::invalid_argument("QFN internally sharded source admission has an invalid GPU/CPU policy");
    definition.topology.expert_overlay_plan = std::move(plan);
    // Compact identities retain the actual communicator order while leaving
    // room for the plain fixture's explicit FullSourceAdmission registration.
    definition.topology.test_id = backend == DeviceType::CUDA ? "CUDA" : "ROCm";
    for (const auto ordinal : ordinals)
        definition.topology.test_id += std::to_string(ordinal);
    definition.topology.test_id += "_CPU0_TP" + std::to_string(ordinals.size()) +
        "_Experts8_OwnedColumns";
    return definition;
}

/**
 * @return Eight CPU-NUMA MTP-off positive source-admission inputs.
 * @note The six existing numerical families remain unchanged. Active MTP and
 *       adaptive-GPU cross-products are separate, still-unqualified work.
 *
 * The canonical expander owns policy identities and public configuration.
 * Selection below compares typed values rather than names. CPU0/CPU1 denote
 * distinct physical NUMA domains; the runtime helper must authenticate both
 * through the already-published HardwareInventory, never CPU worker count.
 */
inline const std::vector<ModelParityCase> &qwen4ExpExtendedSourceAdmissionCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        constexpr std::array composed{
            Qwen4ExpParityTopology::CPU2TP,
            Qwen4ExpParityTopology::CPU2PP};
        for (const auto topology : composed)
        {
            auto expanded = expandModelParityDefinition(
                qwen4ExpComposedSourceAdmissionDefinition(topology));
            result.insert(result.end(), std::make_move_iterator(expanded.begin()),
                std::make_move_iterator(expanded.end()));
        }
        if (result.size() != 8)
            throw std::logic_error("QFN CPU-NUMA source admission requires exactly eight explicit cells");
        return result;
    }();
    return cases;
}

/**
 * @return Sixteen positive composed-GPU admission candidates awaiting implementation.
 * @note Candidate registration still requires successful native materialization;
 *       it never converts a missing implementation into an expected rejection.
 *
 * Native domain-to-domain graph composition and movable CPU TP maintenance
 * are explicit remaining obligations. Preserve all requested participants,
 * movement policies and table tiers so focused future runs reproduce these gaps.
 */
inline const std::vector<ModelParityCase> &qwen4ExpComposedCandidateSourceAdmissionCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        constexpr std::array composed{
            Qwen4ExpParityTopology::ROCm4TP2PP2,
            Qwen4ExpParityTopology::CUDA2CPU2TPPP,
            Qwen4ExpParityTopology::ROCm2CPU2TPPP,
            Qwen4ExpParityTopology::CUDA2ROCm2CPU2TPPP};
        for (const auto topology : composed)
        {
            auto expanded = expandModelParityDefinition(
                qwen4ExpComposedSourceAdmissionDefinition(topology));
            result.insert(result.end(), std::make_move_iterator(expanded.begin()),
                std::make_move_iterator(expanded.end()));
        }
        if (result.size() != 16)
            throw std::logic_error("QFN composed candidate admission requires exactly 16 explicit cells");
        return result;
    }();
    return cases;
}

/**
 * @return Twelve explicit cross-tier expert-sharding rejection cases.
 * @note These cases must never enter the positive materialization fixture.
 *
 * Preserve backend, degree, CPU tier, expert ceiling and ngram policy exactly.
 * The rejection fixture authenticates the production diagnostic; no lower
 * device count, different expert policy or alternative table tier is admitted.
 */
inline const std::vector<ModelParityCase> &qwen4ExpRejectedSourceAdmissionCases()
{
    static const auto cases = [] {
        std::vector<ModelParityCase> result;
        const std::array<std::vector<int>, 3> orders{{{0, 1}, {0, 1, 2, 3},
            {0, 1, 2, 3, 4, 5, 6, 7}}};
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
            for (const auto &order : orders)
                for (auto &cell : expandModelParityDefinition(
                         qwen4ExpInternallyShardedSourceAdmissionDefinition(backend, order)))
                    if (cell.mtp == ModelParityMTP::Off &&
                        cell.expert_overlay == std::optional{ModelParityExpertOverlayPolicy{}})
                        result.push_back(std::move(cell));
        if (result.size() != 12)
            throw std::logic_error("QFN cross-tier shard rejection requires exactly 12 explicit cells");
        return result;
    }();
    return cases;
}
} // namespace llaminar2::test::parity::qwen4exp
