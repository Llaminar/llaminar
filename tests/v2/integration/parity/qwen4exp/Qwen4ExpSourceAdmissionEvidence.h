/**
 * @file Qwen4ExpSourceAdmissionEvidence.h
 * @brief Device-free source/plan proofs and owned cache cleanup for QFN admission.
 *
 * These test-only checks consume the production plan and frozen startup/source
 * metadata. They never discover hardware, read tensor payloads, allocate model
 * state, or substitute a requested topology after admission. Explicit CPU NUMA
 * identities remain distinct even though kernel DeviceId conversion drops NUMA.
 */
#pragma once
#include "Qwen4ExpSourceAdmissionCases.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/moe/MoEExpertOverlayAuthorityPlan.h"
#include "loaders/ImmutableRowTable.h"
#include "loaders/ModelLoader.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "execution/runner/IOrchestrationRunner.h"
#include "execution/moe/MoEPipelinePreparedPlan.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "tensors/TensorKernels.h"
#include "utils/PerfStatsCollector.h"
#include "planning/RankMemoryPlanInputs.h"
#include "execution/mtp/MTPServingForwardCaptureGeometry.h"
#include "execution/mtp/MTPConditionForwardPurpose.h"
#include "execution/mtp/MTPVerifierOutcomeGraph.h"
#include <map>
#include <tuple>
#include <array>
#include <set>
#include "planning/PhysicalMemoryAuthority.h"
#include <gtest/gtest.h>
#include <algorithm>
#include "models/qwen4exp/Qwen4ExpConfig.h"
#include "kernels/common/SparseQueryExecutionProgram.h"
#include <sstream>
#include <filesystem>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace llaminar2::test::parity::qwen4exp
{

/** @brief One planned serving leaf; expert-only allocation resources are absent. */
struct Qwen4ExpMTPSourceServingParticipant
{
    std::string identity;            ///< Ordered PP/TP identity from the actual execution plan.
    DeviceId device;                 ///< Kernel/counter identity, including shared CPU multiplicity.
    std::vector<int> verifier_rows;  ///< Exact production retained GPU family, empty on CPU.
    bool completion_required = true; ///< Ordinary non-PP host readiness deliberately defers forward setup.
    std::size_t sparse_program_count = 1; ///< Exact immutable model program inventory; nonsparse models retain one.
};
/** @brief Explicit validated model inventory; absence is never interpreted as nonsparse. */
enum class Qwen4ExpMTPSourceProgramFamily { RankingOnly, CompleteSparse };

/** @brief Authenticated host startup boundary; ordinary non-PP setup may defer, overlay setup seals its family. */
enum class Qwen4ExpMTPHostSetup { OrdinaryDeferred, OverlayMaterialized };

/** @brief Project only actual serving PP/TP leaves and their native verifier policy.
 * @param plan Actual initialized rank execution plan, not the PMA resource list.
 * @param programs Explicit program inventory derived from validated main model mixers.
 * @param host_setup Resolved overlay startup contract, independent of observed completions.
 * @return Complete ordered serving leaf manifest, excluding expert-only domains.
 */
inline std::vector<Qwen4ExpMTPSourceServingParticipant>
qwen4ExpMTPSourceServingParticipants(const RankExecutionPlan &plan, Qwen4ExpMTPSourceProgramFamily programs, Qwen4ExpMTPHostSetup host_setup)
{
    std::vector<Qwen4ExpMTPSourceServingParticipant> result;
    if (host_setup != Qwen4ExpMTPHostSetup::OrdinaryDeferred &&
        host_setup != Qwen4ExpMTPHostSetup::OverlayMaterialized)
        throw std::invalid_argument("Unknown source host setup lifecycle");
    const auto domain = declaredRankDeviceGenerationDomain(plan);
    const auto append = [&](std::string identity, const GlobalDeviceAddress &address)
    {
        const auto device = address.toLocalDeviceId();
        if (!device.is_valid())
            throw std::invalid_argument("MTP source serving plan contains an invalid leaf");
        Qwen4ExpMTPSourceServingParticipant leaf{
            std::move(identity), device, {}, device.is_gpu() || plan.usesLocalPP() ||
                host_setup == Qwen4ExpMTPHostSetup::OverlayMaterialized};
        if (device.is_gpu())
        {
            const auto role =
                mtpVerifierExecutionRole(declaredDeviceGenerationExecutionPolicy(device.type, domain));
            for (const auto &geometry : resolveMTPServingForwardCaptureFamily(
                     plan.runtime.mtp, resolveMTPRetainedTargetQueryRows(plan.runtime.mtp),
                     resolveMTPVerifierGraphFamily(role)))
                leaf.verifier_rows.push_back(geometry.verifier_rows);
            switch (programs)
            {
            case Qwen4ExpMTPSourceProgramFamily::RankingOnly: break;
            case Qwen4ExpMTPSourceProgramFamily::CompleteSparse:
                leaf.sparse_program_count = kSparseQueryExecutionProgramCount; break;
            default: throw std::invalid_argument("Unknown source sparse-query program family");
            }
            if (leaf.verifier_rows.empty())
                throw std::logic_error("Active GPU source plan has no retained verifier family");
        }
        result.push_back(std::move(leaf));
    };
    if (!plan.local_pp_devices.empty())
    {
        if (!plan.local_pp_stage_tp_info.empty() &&
            plan.local_pp_stage_tp_info.size() != plan.local_pp_devices.size())
            throw std::invalid_argument("MTP source PP plan has an incomplete TP manifest");
        for (size_t stage = 0; stage < plan.local_pp_devices.size(); ++stage)
        {
            if (!plan.local_pp_stage_tp_info.empty() && !plan.local_pp_stage_tp_info[stage].devices.empty())
                for (size_t member = 0; member < plan.local_pp_stage_tp_info[stage].devices.size(); ++member)
                    append("pp" + std::to_string(stage) + "/tp" + std::to_string(member),
                           plan.local_pp_stage_tp_info[stage].devices[member]);
            else
                append("pp" + std::to_string(stage) + "/tp0", plan.local_pp_devices[stage]);
        }
    }
    else if (!plan.local_tp_devices.empty())
        for (size_t member = 0; member < plan.local_tp_devices.size(); ++member)
            append("tp" + std::to_string(member), plan.local_tp_devices[member]);
    else
        append("primary", plan.primary_device);
    return result;
}
/** @brief Authenticate exact completed leaves and every required GPU row/outcome.
 * @param before Setup observation before this source runner initializes.
 * @param after Observation while original initialized owners remain live.
 * @param expected Actual plan's serving leaves, including CPU multiplicity.
 * @return Exact number of setup-completed participants; explicitly deferred hosts are excluded.
 * @throws std::logic_error For missing/extra/stale/reset/incomplete setup evidence.
 *
 * GPU roles are main-model condition/verifier forwards, including PP followers;
 * predictor bodies and generation-controller execution are separate ownership.
 * Ordinary non-PP host initialization deliberately does not enter the GPU/native
 * serving-family materializer. Its source/physical owners are checked separately;
 * no host forward-family completion is asserted or manufactured here.
 */
inline size_t
requireQwen4ExpMTPSourceSetupFamilies(std::span<const PerfStatRecord> before,
                                      std::span<const PerfStatRecord> after,
                                      std::span<const Qwen4ExpMTPSourceServingParticipant> expected)
{
    if (expected.empty())
        throw std::invalid_argument("MTP source evidence needs its complete serving manifest");
    std::set<std::string> identities;
    std::set<DeviceId> gpu_owners;
    std::map<std::string, double> required_completed, completed;
    using Family = std::tuple<std::string, std::string, std::string, std::string, std::string>;
    std::map<Family, double> required_families, families;
    for (const auto &leaf : expected)
    {
        if (leaf.identity.empty() || !identities.insert(leaf.identity).second || !leaf.device.is_valid())
            throw std::invalid_argument("MTP source serving manifest has an invalid/duplicate identity");
        const auto device = leaf.device.toString();
        if (leaf.completion_required)
            ++required_completed[device];
        if (!leaf.device.is_gpu())
            continue;
        if (!leaf.completion_required)
            throw std::invalid_argument("GPU serving setup cannot be deferred as host readiness");
        if (!gpu_owners.insert(leaf.device).second)
            throw std::invalid_argument("MTP serving manifest repeats an exclusive GPU owner");
        if (leaf.verifier_rows.empty())
            throw std::invalid_argument("GPU source serving leaf lacks exact verifier rows");
        if (leaf.sparse_program_count != 1 && leaf.sparse_program_count != kSparseQueryExecutionProgramCount)
            throw std::invalid_argument("Invalid retained sparse-query program inventory");
        for (std::size_t program = 0; program < leaf.sparse_program_count; ++program)
        {
            for (const auto purpose : kMTPConditionForwardPurposes)
                ++required_families[{device, "mtp_condition", "1", std::to_string(static_cast<int>(purpose)), std::to_string(program)}];
            std::set<int> unique_rows;
            for (const int rows : leaf.verifier_rows)
            {
                if (rows < 2 || !unique_rows.insert(rows).second)
                    throw std::invalid_argument("Invalid retained verifier rows");
                for (const auto outcome :
                     {MTPVerifierOutcomeGraphMode::Greedy, MTPVerifierOutcomeGraphMode::Disabled})
                    ++required_families[{device, "mtp_grouped_verifier", std::to_string(rows),
                                         std::to_string(static_cast<int>(outcome)), std::to_string(program)}];
            }
        }
    }
    const auto matches = [](const auto &a, const auto &b)
    {
        return a.kind == b.kind && a.domain == b.domain && a.name == b.name && a.phase == b.phase &&
               a.device == b.device && a.tags == b.tags;
    };
    for (const auto &old : before)
        if (old.domain == "forward_graph" && old.phase == "setup" &&
            (old.name == "serving_graph_family_participant_completions" ||
             old.name == "serving_graph_family_materializations"))
        {
            const auto found = std::find_if(after.begin(), after.end(),
                                            [&](const auto &record) { return matches(old, record); });
            if (found == after.end() || found->value < old.value)
                throw std::logic_error("MTP source setup evidence reset during admission");
        }
    for (const auto &record : after)
    {
        if (record.kind != PerfStatRecord::Kind::Counter || record.domain != "forward_graph" ||
            record.phase != "setup")
            continue;
        const auto old = std::find_if(before.begin(), before.end(),
                                      [&](const auto &prior) { return matches(prior, record); });
        const double added = record.value - (old == before.end() ? 0 : old->value);
        if (added < 0)
            throw std::logic_error("MTP source setup evidence decreased");
        if (added == 0)
            continue;
        if (record.name == "serving_graph_family_participant_completions")
            completed[record.device] += added;
        if (record.name != "serving_graph_family_materializations")
            continue;
        const auto role = record.tags.find("role");
        if (role == record.tags.end() ||
            (role->second != "mtp_condition" && role->second != "mtp_grouped_verifier"))
            continue;
        const auto submission = record.tags.find("submission"), rows = record.tags.find("physical_rows");
        const auto program = record.tags.find("sparse_query_program");
        const auto detail =
            record.tags.find(role->second == "mtp_condition" ? "commit_purpose" : "outcome_mode");
        if (submission == record.tags.end() || submission->second != "materialized_unlaunched" ||
            rows == record.tags.end() || detail == record.tags.end() || program == record.tags.end())
            throw std::logic_error("MTP source family has incomplete immutable role/row identity");
        families[{record.device, role->second, rows->second, detail->second, program->second}] += added;
    }
    if (completed != required_completed)
    {
        std::ostringstream diagnostic;
        diagnostic << "MTP source serving participant count mismatch:";
        auto devices = required_completed;
        devices.insert(completed.begin(), completed.end());
        for (const auto &[device, unused] : devices)
        {
            const double required = required_completed.contains(device) ? required_completed.at(device) : 0;
            const double actual = completed.contains(device) ? completed.at(device) : 0;
            if (actual != required)
                diagnostic << " [" << device << " expected=" << required << " actual=" << actual << ']';
        }
        throw std::logic_error(diagnostic.str());
    }
    if (families != required_families)
    {
        std::ostringstream diagnostic;
        diagnostic << "MTP source retained family mismatch (device,role,rows,detail,program):";
        auto keys = required_families;
        keys.insert(families.begin(), families.end());
        for (const auto &[key, unused] : keys)
        {
            const auto expected_count = required_families.contains(key) ? required_families.at(key) : 0;
            const auto actual_count = families.contains(key) ? families.at(key) : 0;
            if (actual_count == expected_count) continue;
            const auto &[device, role, rows, detail, program] = key;
            diagnostic << " [" << device << ',' << role << ',' << rows << ',' << detail << ',' << program
                       << " expected=" << expected_count << " actual=" << actual_count << ']';
        }
        throw std::logic_error(diagnostic.str());
    }
    return std::count_if(expected.begin(), expected.end(),
                         [](const auto &leaf) { return leaf.completion_required; });
}

/**
 * @brief Observe only existing prepared expert records under their published scopes.
 * @param reuse Existing production certificate and frozen domain authorities.
 * @return Prepared engine records for source48 from each actual owning authority.
 *
 * Ordinary prepared experts belong to PreparedWeightStore slabs, while routed
 * overlay participants publish scope aliases in ExpertGemmRegistry. This observation
 * neither requires an overlay registry on a whole-source CPU model nor accepts
 * a populated slab without its original source format and owned engine lifetime.
 *
 * This is an existence count, never physical bytes or a second admission ledger.
 * Dense/norm prepared bindings have no public name iterator; initialized native
 * graph construction enforces their existing frozen operand contracts instead.
 */
inline std::size_t qwen4ExpSourcePreparedPredictorExpertRecords(const ModelContextReuseContract &reuse)
{
    std::set<std::string> domains{std::string{}};
    if (reuse.prepared_routed_weight_plan)
        for (const auto &domain : reuse.prepared_routed_weight_plan->domains) domains.insert(domain.name);
    if (reuse.pipeline_prepared_plan)
        for (const auto &stage : reuse.pipeline_prepared_plan->stages())
            for (const auto &domain : stage.placement->domains) domains.insert(domain.name);
    // WeightManager's const public accessor keeps the original model owner.
    const auto manager = reuse.context->concreteWeightManager();
    const auto &registry = manager->expertGemmRegistry();
    const auto store = manager->preparedWeightStoreIfInitialized();
    const auto &loader = reuse.context->concreteLoader();
    const auto *predictor = loader.ggufPredictorDirectory();
    if (!store || !predictor)
        throw std::logic_error("Predictor expert observation requires its original source and prepared store");
    const auto *gate = predictor->findTensor("blk.48.ffn_gate_exps.weight");
    if (!gate || gate->dimensions.size() != 3)
        throw std::logic_error("Predictor expert observation requires the original source48 expert extent");
    const auto expert_count = static_cast<int>(gate->dimensions[2]);
    const auto native = buildMoEOverlayLayerWeightManifestFromGGUF(*predictor, 1, expert_count, 48);
    const std::array roles{WeightRole::MoEExpertGate, WeightRole::MoEExpertUp, WeightRole::MoEExpertDown};
    std::size_t count = 0;
    for (const auto &resource : reuse.physical_memory_authority->admission()->plan().resources())
    {
        if (resource.resource().world_rank != reuse.physical_memory_authority->worldRank()) continue;
        const auto device = resource.resource().device;
        for (const auto &domain : domains)
            count += registry.countEnginesForLayerInDomain(domain, device, 48);
        for (std::size_t role = 0; role < roles.size(); ++role)
        {
            const auto &projection = native.front().projections[role];
            const ExpertSlabDescriptor descriptor{
                .layer_idx = 48, .role = roles[role], .device = device,
                .num_experts = expert_count, .local_expert_start = 0,
                .local_expert_count = expert_count,
                .rows_per_expert = static_cast<std::size_t>(projection.N),
                .cols_per_expert = static_cast<std::size_t>(projection.K)};
            const auto slab = store->findExpertSlab(descriptor);
            if (!slab) continue; // Absent/inactive authorities own no source48 slab.
            EXPECT_EQ(slab->model_id, store->modelId());
            EXPECT_EQ(slab->layer_idx, 48);
            EXPECT_EQ(slab->role, roles[role]);
            EXPECT_EQ(slab->device, device);
            const auto available = store->expertAvailabilityMask(*slab);
            if (available.size() != static_cast<std::size_t>(expert_count))
                throw std::logic_error("Predictor slab has an incompatible expert availability extent");
            for (int expert = 0; expert < expert_count; ++expert)
            {
                const auto lifetime = store->expertGemmKernelLifetime(*slab, expert);
                const auto *engine = store->expertGemmKernel(*slab, expert);
                EXPECT_EQ(lifetime.get(), engine);
                EXPECT_EQ(available[expert], engine != nullptr);
                if (!engine) continue;
                EXPECT_TRUE(engine->hasWeights());
                if(device.is_cpu())
                {
                    NativeVnniSourceIdentity original;
                    EXPECT_TRUE(engine->weights_converted());
                    EXPECT_TRUE(engine->exportNativeVNNISourceIdentity(original));
                    EXPECT_EQ(ExpertWeightFormat::nativeVnni(original), projection.format);
                }
                ++count;
            }
        }
    }
    return count;
}


/**
 * @brief Reject any change to the authored maintenance policy during admission.
 * @param authored Original typed maintenance request.
 * @param actual Resolved configuration or actual execution-plan policy.
 * @throws std::invalid_argument If any typed policy field differs.
 *
 * Defaulted value equality includes every policy field, preserving explicit
 * optional values rather than collapsing them into resolved defaults.
 */
inline void requireQwen4ExpSourceMaintenancePolicy(const MoERebalanceRuntimeConfig &authored,
    const MoERebalanceRuntimeConfig &actual)
{
    if (!(authored == actual))
        throw std::invalid_argument("QFN admission changed the authored maintenance policy");
}

/**
 * @brief Authenticate provider staging within the complete host lookup owner.
 * @param attestation One coherent PMA snapshot from the retained model authority.
 * @param allocator Exact host provider allocator, including its world rank.
 * @param provider_bytes Actual retained provider staging, independently checked
 *        against its canonical geometry by the admission observer.
 * @param consumers Consumers from the same resolved immutable-table placement.
 * @throws std::exception For missing/foreign ownership, incomplete live storage,
 *         incomplete commitment, overflow, missing GPU channels or stale CPU-only bytes.
 *
 * A host-backed GPU lookup retains mapped channel storage in the same PMA owner
 * as provider staging. The provider observation deliberately excludes those
 * channels. PMA's complete typed owner attestation, rather than an observer-side
 * channel ledger or a capacity reservation, proves their physical materialization.
 * CPU-only placements have no channel allocation and retain exact provider equality.
 */
inline void requireQwen4ExpHostLookupStaging(
    std::span<const PhysicalMemoryOwnerAttestation> attestation,
    PhysicalMemoryAllocatorIdentity allocator, std::size_t provider_bytes,
    std::span<const PhysicalMemoryAllocatorIdentity> consumers)
{
    if (!allocator.device.is_cpu() || consumers.empty())
        throw std::invalid_argument("QFN host lookup staging requires its host allocator and declared consumers");
    const auto line = std::find_if(attestation.begin(), attestation.end(), [&](const auto &row) {
        return row.identity == allocator && row.owner == PhysicalMemoryOwner::ImmutableLookupStaging;
    });
    if (line == attestation.end())
        throw std::invalid_argument("QFN host lookup staging has no exact allocator/owner attestation");
    if (!line->complete() || !line->committed())
        throw std::invalid_argument("QFN host lookup staging requires complete live materialization and commitment");
    if (line->materialized_new_bytes >
        std::numeric_limits<std::size_t>::max() - line->adopted_resident_bytes)
        throw std::overflow_error("QFN host lookup staging live extent overflows");
    const auto live = line->materialized_new_bytes + line->adopted_resident_bytes;
    const bool gpu_consumer = std::any_of(consumers.begin(), consumers.end(), [](const auto &consumer) {
        return consumer.device.is_gpu();
    });
    if (gpu_consumer ? live <= provider_bytes : live != provider_bytes)
        throw std::invalid_argument(gpu_consumer
            ? "QFN host-backed GPU lookup is missing live channel storage beyond the provider"
            : "QFN CPU-only lookup staging differs from its exact provider owner");
}

/**
 * @brief Reject the formerly unclaimed CPU storage categories after real setup.
 * @param rows Coherent rank-local attestation from the existing model authority.
 * @throws std::logic_error If a CPU compute allocation category has no live claim.
 *
 * This is a materialization canary, not equality to a maximum admission budget.
 * Per-allocation unit regressions prove exact byte geometry and lifetime. A
 * prepared bank may consume less than its envelope, and primary/additional
 * copies may share immutable sources. Unused prefix capacity remains untouched.
 */
inline void requireQwen4ExpCPUStorageMaterialization(
    std::span<const PhysicalMemoryOwnerAttestation> rows)
{
    const auto planned = [](const auto &row) {
        return row.planned_new_bytes != 0 || row.planned_resident_bytes != 0;
    };
    const auto live = [](const auto &row) {
        return row.materialized_new_bytes != 0 || row.adopted_resident_bytes != 0;
    };
    for (const auto &row : rows)
    {
        if (!row.identity.device.is_cpu() || !planned(row)) continue;
        if (row.owner == PhysicalMemoryOwner::ActivationArena ||
            row.owner == PhysicalMemoryOwner::RoutedExpertWeights ||
            row.owner == PhysicalMemoryOwner::ExecutionWorkspace)
        {
            if (!live(row))
                throw std::logic_error("QFN CPU compute storage is admitted but has no live physical claim");
        }
        else if (row.owner == PhysicalMemoryOwner::PrimaryModelWeights ||
                 row.owner == PhysicalMemoryOwner::AdditionalModelWeights)
        {
            // These two owner lines may contain unmaterialized graph-family
            // capacity. At least one actual prepared bank must be represented
            // on this exact physical allocator; a sibling CPU/GPU cannot count.
            const bool prepared_live = std::any_of(rows.begin(), rows.end(), [&](const auto &other) {
                return other.identity == row.identity &&
                    (other.owner == PhysicalMemoryOwner::PrimaryModelWeights ||
                     other.owner == PhysicalMemoryOwner::AdditionalModelWeights) && live(other);
            });
            if (!prepared_live)
                throw std::logic_error("QFN CPU prepared weights have no live physical claim");
        }
    }
}

/**
 * @brief Create an empty admission namespace without overwriting previous evidence.
 * @param path Absolute case/rank output directory; an existing empty directory is legitimate.
 * @throws std::invalid_argument For a relative, aliased, non-directory or nonempty output.
 * @throws std::filesystem::filesystem_error If the owned directory cannot be created or inspected.
 *
 * Parent frameworks may create the empty child first. Existing reports, failure
 * evidence and incomplete attempts are never deleted to make another run fit.
 * Disposable prefix cleanup remains a separate final-owner lifetime operation.
 */
inline void prepareQwen4ExpAdmissionEvidenceDirectory(const std::filesystem::path &path)
{
    if (!path.is_absolute() || std::filesystem::is_symlink(path))
        throw std::invalid_argument("QFN admission evidence requires an absolute non-symlink output directory");
    if (std::filesystem::exists(path))
    {
        if (!std::filesystem::is_directory(path) || !std::filesystem::is_empty(path))
            throw std::invalid_argument("QFN admission evidence already exists; preserve it and select a fresh artifact root");
        return;
    }
    std::filesystem::create_directories(path);
    if (!std::filesystem::is_directory(path) || !std::filesystem::is_empty(path))
        throw std::invalid_argument("QFN admission evidence output is not an empty owned directory");
}

/**
 * @brief Own only one cell/rank's disposable prefix-cache directory.
 *
 * Admission evidence lives in the parent and is never deleted here. This guard
 * is declared before every runner/model owner; its destructor therefore runs
 * after their retirement even when a GoogleTest assertion returns early.
 */
class Qwen4ExpAdmissionPrefixDirectory final
{
public:
    /** @brief Create this exact fixture child before any writer can borrow it. @param path Owned prefix child. */
    explicit Qwen4ExpAdmissionPrefixDirectory(std::filesystem::path path) : path_(std::move(path))
    {
        if (!path_.is_absolute() || path_.filename() != "prefix-cache")
            throw std::invalid_argument("QFN admission cleanup owns only an absolute prefix-cache child");
        std::filesystem::create_directories(path_);
    }
    /** @brief Remove the owned child on assertion/exception unwinding; report actual cleanup failures. */
    ~Qwen4ExpAdmissionPrefixDirectory()
    {
        try { cleanup(); }
        catch (const std::exception &error) { ADD_FAILURE() << "Admission prefix cleanup failed: " << error.what(); }
    }
    Qwen4ExpAdmissionPrefixDirectory(const Qwen4ExpAdmissionPrefixDirectory &) = delete;
    Qwen4ExpAdmissionPrefixDirectory &operator=(const Qwen4ExpAdmissionPrefixDirectory &) = delete;
    /** @brief Remove only the owned child after all native/archive owners retire. @throws filesystem_error On failure. */
    void cleanup()
    {
        std::filesystem::remove_all(path_);
        if (std::filesystem::exists(path_))
            throw std::runtime_error("Admission prefix child remains after cleanup");
    }
private:
    std::filesystem::path path_;
};

/**
 * @brief Authenticate one actual endpoint against its exact authored address.
 * @param expected Original public address, including explicit CPU locality.
 * @param actual Address embedded in the actual production execution plan.
 * @param hostname Owning rank's authenticated execution-plan hostname.
 * @param hardware Frozen startup inventory, never a live rediscovery.
 * @throws std::invalid_argument For foreign host/vendor/ordinal/locality.
 *
 * Production resolution may replace a local host alias by its owner hostname
 * and an unknown GPU NUMA by that exact inventory device's locality. It cannot
 * rewrite an explicit locality or interchange two equally sized participants.
 */
inline void requireQwen4ExpSourceEndpoint(const GlobalDeviceAddress &expected,
    const GlobalDeviceAddress &actual, const std::string &hostname, const HardwareInventory &hardware)
{
    if (expected.device_type != actual.device_type || expected.device_ordinal != actual.device_ordinal)
        throw std::invalid_argument("QFN admitted execution endpoint changed vendor or ordinal");
    const bool host_matches = expected.isLocal()
        ? (actual.isLocal() || (!hostname.empty() && actual.hostname == hostname))
        : actual.hostname == expected.hostname;
    if (!host_matches)
        throw std::invalid_argument("QFN admitted execution endpoint changed owning host");
    if (expected.hasValidNuma())
    {
        if (actual.numa_node != expected.numa_node)
            throw std::invalid_argument("QFN admitted execution endpoint changed explicit NUMA locality");
        if (expected.isCPU()) (void)hardware.cpuDevice(expected.numa_node);
        return;
    }
    if (expected.isCPU())
        throw std::invalid_argument("QFN source admission requires explicit CPU NUMA locality");
    // Flat TP retains the authored unknown locality; named domains resolve it.
    // Both are valid representations, but any positive resolution must agree
    // with the same frozen device rather than an unrelated participant.
    if (actual.numa_node == expected.numa_node) return;
    const auto &devices = expected.isCUDA() ? hardware.cuda_devices : hardware.rocm_devices;
    const auto found = std::find_if(devices.begin(), devices.end(), [&](const auto &device) {
        return device.device_id == expected.device_ordinal;
    });
    if (found == devices.end() || actual.numa_node != found->numa_node)
        throw std::invalid_argument("QFN admitted GPU NUMA resolution is absent from frozen inventory");
}

/**
 * @brief Require exact ordered endpoint membership, not just participant count.
 * @param expected Authored membership of one actual TP/domain/stage.
 * @param actual Production plan membership in the same logical order.
 * @param hostname Owning rank's authenticated hostname.
 * @param hardware Frozen startup inventory.
 * @throws std::invalid_argument For missing, reordered or aliased endpoints.
 */
inline void requireQwen4ExpSourceEndpoints(std::span<const GlobalDeviceAddress> expected,
    std::span<const GlobalDeviceAddress> actual, const std::string &hostname, const HardwareInventory &hardware)
{
    if (expected.size() != actual.size())
    {
        const auto describe = [](std::span<const GlobalDeviceAddress> members) {
            std::string result = "[";
            // Native cases are bounded to eight GPUs plus CPU capacity. Keep
            // malformed-fixture diagnostics bounded too; addresses are metadata.
            for (std::size_t index = 0; index < std::min<std::size_t>(members.size(), 16); ++index)
            {
                if (index) result += ",";
                result += members[index].toString();
            }
            if (members.size() > 16) result += ",...";
            return result + "]";
        };
        throw std::invalid_argument("QFN admitted execution membership has a different extent: expected_count=" +
            std::to_string(expected.size()) + " actual_count=" + std::to_string(actual.size()) +
            " expected=" + describe(expected) + " actual=" + describe(actual));
    }
    for (std::size_t index = 0; index < expected.size(); ++index)
        requireQwen4ExpSourceEndpoint(expected[index], actual[index], hostname, hardware);
}

/**
 * @brief Apply the production implicit-MoE normalization to an authored flat TP request.
 * @param authored Original configuration before source admission resolves it.
 * @param rank Exact owning rank of this reviewed rank-local QFN source case.
 * @return Original intent with only its canonical single-tier authority installed.
 * @throws std::invalid_argument For invalid canonical synthesis or installation.
 *
 * QFN source metadata proves routed experts. Ordinary multi-device LocalTP is
 * therefore promoted by production to one named ExpertOverlay authority before
 * materialization; accepting an arbitrary extra resolved domain would hide a
 * topology defect. This calls the existing pure production normalizer instead.
 */
inline OrchestrationConfig canonicalQwen4ExpSourceAdmissionConfig(
    const OrchestrationConfig &authored, int rank)
{
    auto expected = authored;
    if (!authored.moe_routed_expert_plan && authored.pp_stage_definitions.empty() && authored.tp_devices.size() > 1)
    {
        const auto normalized = normalizeMoEExpertOverlayAuthorityPlan({
            .requested_plan = {}, .model_has_routed_experts = true,
            .world_rank = rank, .local_tp_participants = authored.tp_devices,
            .local_tp_weights = authored.tp_weights, .local_tp_backend = authored.default_backend,
            .has_cross_rank_tensor_parallel = false, .has_pipeline_parallel = false,
            .routed_compute_policy = authored.routed_expert_compute_policy,
            .owner_order = authored.routed_expert_owner_order,
            .residency_maintenance = authored.moe_rebalance.mode});
        if (!normalized.synthesized() || !normalized.plan)
            throw std::invalid_argument("QFN flat MoE TP did not produce its canonical authority");
        // Match ResolvedRankOrchestration: install the synthesized requested
        // authority first, then replace its simple-TP selectors through the
        // canonical resolved-plan publication boundary. The installer rejects
        // a resolved plan with no prior requested authority by design.
        expected.moe_routed_expert_plan = normalized.plan;
        const auto errors = installResolvedMoEExpertOverlayPlan(expected, normalized.plan,
            MoEExpertOverlayPlanInstallOrigin::SynthesizedSimpleTP);
        if (!errors.empty()) throw std::invalid_argument(errors.front());
    }
    return expected;
}

/**
 * @brief Authenticate actual TP/PP ownership after the public projector materializes it.
 * @param cell Canonical original-source case, including physical topology kind.
 * @param rank Current owning MPI rank.
 * @param authored Immutable public configuration before production normalization.
 * @param actual Actual production RankExecutionPlan used by the initialized runner.
 * @param hardware Frozen startup inventory used for permissible address binding.
 * @throws std::invalid_argument For missing endpoints, changed order/collective,
 *         wrong stage membership/layer range, or an unreviewed topology.
 *
 * Named-domain proofs include every capacity/execution domain. Authored flat
 * CPU2 TP canonically gains one implicit MoE domain, and its actual TP list must
 * independently retain the same two NUMA endpoints. Composed PP verifies full stage TP groups as well as their roots;
 * a correct flattened root list alone cannot certify TP inside each stage.
 */
inline void requireQwen4ExpSourceExecutionPlan(const ModelParityCase &cell, int rank,
    const OrchestrationConfig &authored, const RankExecutionPlan &actual, const HardwareInventory &hardware)
{
    if (rank < 0 || rank >= cell.topology.mpi_ranks || actual.rank != rank)
        throw std::invalid_argument("QFN source admission plan changed owning rank");
    if (cell.topology.mpi_ranks != 1)
        throw std::invalid_argument("QFN gold-free plan proof requires a reviewed rank-local topology");
    const auto expected = canonicalQwen4ExpSourceAdmissionConfig(authored, rank);
    const auto domains = expected.executionDomainDefinitions();
    if (actual.my_domains.size() != domains.size())
        throw std::invalid_argument("QFN actual execution domain inventory has a different extent");
    for (const auto &domain : domains)
    {
        const auto found = std::find_if(actual.my_domains.begin(), actual.my_domains.end(), [&](const auto &value) {
            return value.domain_name == domain.name;
        });
        if (found == actual.my_domains.end() ||
            std::count_if(actual.my_domains.begin(), actual.my_domains.end(), [&](const auto &value) {
                return value.domain_name == domain.name;
            }) != 1)
            throw std::invalid_argument("QFN actual execution domain is missing or duplicated");
        requireQwen4ExpSourceEndpoints(domain.participants, found->devices, actual.hostname, hardware);
        if (found->weights != domain.weights || found->backend != domain.backend)
            throw std::invalid_argument("QFN actual execution domain changed weights or collective");
    }
    if (!actual.has_embedding || !actual.has_lm_head)
        throw std::invalid_argument("QFN actual rank-local plan lost embedding or LM-head source ownership");
    if (actual.first_layer != 0 || actual.last_layer != cell.model.transformer_layers - 1)
        throw std::invalid_argument("QFN actual rank-local plan lost original source layers");
    if (cell.topology.kind == ModelParityTopologyKind::RankLocalPipelineParallel)
    {
        const auto &stages = authored.pp_stage_definitions;
        if (stages.size() < 2 || actual.local_pp_devices.size() != stages.size() ||
            actual.local_pp_stage_tp_info.size() != stages.size() ||
            actual.local_pp_layer_boundaries.size() != stages.size() + 1)
            throw std::invalid_argument("QFN actual PP stage inventory is incomplete");
        bool composed = false;
        std::vector<GlobalDeviceAddress> expected_roots;
        for (std::size_t index = 0; index < stages.size(); ++index)
        {
            const auto &stage = stages[index];
            const auto domain = std::find_if(domains.begin(), domains.end(), [&](const auto &value) {
                return value.name == stage.domain_name;
            });
            if (stage.stage_id != static_cast<int>(index) || domain == domains.end() || domain->participants.empty() ||
                stage.first_layer != actual.local_pp_layer_boundaries[index] ||
                stage.last_layer + 1 != actual.local_pp_layer_boundaries[index + 1])
                throw std::invalid_argument("QFN actual PP layer boundaries or stage domain changed");
            expected_roots.push_back(domain->participants.front());
            const auto &tp = actual.local_pp_stage_tp_info[index];
            requireQwen4ExpSourceEndpoints(domain->participants, tp.devices, actual.hostname, hardware);
            requireQwen4ExpSourceEndpoint(domain->participants.front(), actual.local_pp_devices[index], actual.hostname, hardware);
            if (tp.tp_weights != domain->weights || tp.tp_backend != domain->backend)
                throw std::invalid_argument("QFN actual PP stage TP weights or collective changed");
            composed = composed || domain->participants.size() > 1;
        }
        // Outer PP edges select transport from physical stage roots. They do
        // not inherit the explicit collective used inside each TP stage.
        if (actual.local_pp_backend != ExecutionPlanBuilder{}.selectBackend(expected_roots))
            throw std::invalid_argument("QFN actual PP transport collective changed");
        if (actual.local_pp_layer_boundaries.front() != 0 ||
            actual.local_pp_layer_boundaries.back() != cell.model.transformer_layers ||
            (composed && !actual.local_tp_devices.empty()))
            throw std::invalid_argument("QFN actual PP was truncated or replaced by outer flat TP");
        requireQwen4ExpSourceEndpoint(domains.front().participants.front(), actual.primary_device, actual.hostname, hardware);
        return;
    }
    if (!actual.local_pp_devices.empty() || !actual.local_pp_layer_boundaries.empty() ||
        !actual.local_pp_stage_tp_info.empty())
        throw std::invalid_argument("QFN non-PP source admission acquired an unintended pipeline");
    if (expected.moe_routed_expert_plan)
    {
        const auto &name = expected.moe_routed_expert_plan->continuation_domain;
        const auto continuation = std::find_if(domains.begin(), domains.end(), [&](const auto &domain) {
            return domain.name == name;
        });
        if (continuation == domains.end() || continuation->participants.empty())
            throw std::invalid_argument("QFN named overlay lost its continuation domain");
        requireQwen4ExpSourceEndpoint(continuation->participants.front(), actual.primary_device, actual.hostname, hardware);
        if (continuation->scope == ExecutionDomainScope::RANK_LOCAL)
        {
            requireQwen4ExpSourceEndpoints(continuation->participants, actual.local_tp_devices, actual.hostname, hardware);
            if (actual.tp_scope != TPScope::RANK_LOCAL || actual.local_tp_backend != continuation->backend ||
                actual.local_tp_weights != continuation->weights)
                throw std::invalid_argument("QFN actual continuation TP scope, weights or collective changed");
        }
        else if (continuation->scope == ExecutionDomainScope::SINGLE)
        {
            // Match the production overlay role projection: the one real
            // continuation root is not an invented one-participant LocalTP.
            if (continuation->participants.size() != 1 || !actual.local_tp_devices.empty() ||
                !actual.local_tp_weights.empty() || actual.tp_scope != TPScope::AUTO)
                throw std::invalid_argument("QFN singleton continuation acquired a fabricated LocalTP group");
        }
        else throw std::invalid_argument("QFN source admission has an unreviewed continuation scope");
        return; // Named overlay is sole authority, including all-GPU TP4.
    }
    if (cell.topology.kind == ModelParityTopologyKind::RankLocalTensorParallel)
    {
        if (authored.tp_devices.size() < 2 || authored.tp_degree != static_cast<int>(authored.tp_devices.size()))
            throw std::invalid_argument("QFN authored flat TP membership is incomplete");
        requireQwen4ExpSourceEndpoints(authored.tp_devices, actual.local_tp_devices, actual.hostname, hardware);
        if (actual.local_tp_weights != authored.tp_weights || actual.local_tp_backend != authored.default_backend)
            throw std::invalid_argument("QFN actual flat TP weights or collective changed");
        requireQwen4ExpSourceEndpoint(authored.tp_devices.front(), actual.primary_device, actual.hostname, hardware);
        return;
    }
    if (cell.topology.kind == ModelParityTopologyKind::SingleDevice)
    {
        if (cell.topology.participants.size() != 1 || !actual.local_tp_devices.empty())
            throw std::invalid_argument("QFN single-device source admission changed physical membership");
        requireQwen4ExpSourceEndpoint(cell.topology.participants.front().address, actual.primary_device, actual.hostname, hardware);
        return;
    }
    throw std::invalid_argument("QFN gold-free plan proof has an unreviewed topology kind");
}

/**
 * @brief Join an actual native table owner to the original GGUF PLE coordinates.
 * @param directory Original main GGUF directory already retained by ModelContext.
 * @param artifacts Ordered original source paths, including an optional inactive predictor.
 * @param source Actual host/resident/adaptive owner metadata; never a copied payload.
 * @throws std::invalid_argument For foreign split/name/offset/format/extent or overflow.
 *
 * File metadata authentication remains the owner's setup-only operation. This
 * pure join proves that the authenticated file/tensor is the intended original
 * source, rather than a same-sized foreign table or a transformed row format.
 */
inline void requireQwen4ExpSourceTableIdentity(const GGUFModel &directory,
    std::span<const std::string> artifacts, const ImmutableRowTableSource &source)
{
    const auto *native = directory.findTensor("per_layer_token_embd.weight");
    if (!native || native->dimensions.size() != 2)
        throw std::invalid_argument("QFN source identity lacks original native PLE geometry");
    const auto split = static_cast<std::size_t>(native->split_idx);
    if (split >= artifacts.size() || (directory.split_count > 1 && split >= directory.split_data_offsets.size()))
        throw std::invalid_argument("QFN source identity has an incomplete original shard inventory");
    const auto offset = directory.split_count > 1 ? directory.split_data_offsets[split] : directory.data_offset;
    if (native->offset > std::numeric_limits<std::uint64_t>::max() - offset)
        throw std::invalid_argument("QFN original table source offset overflows");
    if (source.path != artifacts[split] || source.tensor_name != native->name ||
        source.file_offset != offset + native->offset || source.format != native->type ||
        source.rows != native->dimensions[0] || source.row_elements != native->dimensions[1] ||
        source.tensorBytes() != native->size_bytes)
        throw std::invalid_argument("QFN retained native table differs from original GGUF source coordinates");
}
} // namespace llaminar2::test::parity::qwen4exp
