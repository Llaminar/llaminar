/**
 * @file Test__Qwen4Exp_MTP_Admission.cpp
 * @brief Real paired-source serving admission without numerical reference setup.
 *
 * Plain GoogleTest fixtures enter the same production runner factory, startup
 * policy, PMA and initialization as serving. They never inherit parity SetUp,
 * resolve a gold pack, enable snapshots or execute model inference. The existing
 * MPI integration main retires collective/GPU owners after runner destruction.
 */
#include "Qwen4ExpMTPAdmissionCases.h"
#include "utils/TestOrchestrationHelper.h"
#include "backends/BackendManager.h"
#include "config/OrchestrationStartupPolicy.h"
#include "execution/moe/MoEPipelinePreparedPlan.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "tensors/TensorKernels.h"
#include "execution/mtp/MTPWeightManifest.h"
#include "loaders/ModelContext.h"
#include "loaders/PreparedWeightStore.h"
#include "models/qwen4exp/Qwen4ExpSidecarManifest.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>

namespace llaminar2::test::parity::qwen4exp
{
namespace
{
/**
 * @brief Observe only existing prepared expert records under their published scopes.
 * @param reuse Existing production certificate and frozen domain authorities.
 * @return Prepared engine records for source48 from each actual owning authority.
 *
 * Ordinary CPU experts belong to PreparedWeightStore slabs, while routed overlay
 * participants publish scope aliases in ExpertGemmRegistry. This observation
 * neither requires an overlay registry on a whole-source CPU model nor accepts
 * a populated slab without its original source format and owned engine lifetime.
 *
 * This is an existence count, never physical bytes or a second admission ledger.
 * Dense/norm prepared bindings have no public name iterator; initialized native
 * graph construction enforces their existing frozen operand contracts instead.
 */
std::size_t preparedPredictorExpertRecords(const ModelContextReuseContract &reuse)
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
        const auto device = resource.resource().device;
        for (const auto &domain : domains)
            count += registry.countEnginesForLayerInDomain(domain, device, 48);
        if (!device.is_cpu()) continue;
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
            if (!slab) continue; // Inactive predictors and GPU-only participants own no CPU slab.
            EXPECT_EQ(slab->model_id, store->modelId());
            EXPECT_EQ(slab->layer_idx, 48);
            EXPECT_EQ(slab->role, roles[role]);
            EXPECT_EQ(slab->device, device);
            const auto available = store->expertAvailabilityMask(*slab);
            if (available.size() != static_cast<std::size_t>(expert_count))
                throw std::logic_error("Predictor CPU slab has an incompatible expert availability extent");
            for (int expert = 0; expert < expert_count; ++expert)
            {
                const auto lifetime = store->expertGemmKernelLifetime(*slab, expert);
                const auto *engine = store->expertGemmKernel(*slab, expert);
                EXPECT_EQ(lifetime.get(), engine);
                EXPECT_EQ(available[expert], engine != nullptr);
                if (!engine) continue;
                NativeVnniSourceIdentity original;
                EXPECT_TRUE(engine->weights_converted());
                EXPECT_TRUE(engine->exportNativeVNNISourceIdentity(original));
                EXPECT_EQ(ExpertWeightFormat::nativeVnni(original), projection.format);
                ++count;
            }
        }
    }
    return count;
}

/**
 * @brief Require every field in one canonical domain representation to agree.
 * @param actual Observed raw-wrapper projection or effective typed definition.
 * @param expected Expected definition in that same representation.
 */
void expectSameDomainDefinition(const ExecutionDomainDefinition &actual, const ExecutionDomainDefinition &expected)
{
    EXPECT_EQ(actual.name, expected.name);
    EXPECT_EQ(actual.participants, expected.participants);
    EXPECT_EQ(actual.weights, expected.weights);
    EXPECT_EQ(actual.backend, expected.backend);
    EXPECT_EQ(actual.scope, expected.scope);
    EXPECT_EQ(actual.owner_rank, expected.owner_rank);
    EXPECT_EQ(actual.ranks, expected.ranks);
    EXPECT_EQ(actual.routed_compute_policy, expected.routed_compute_policy);
    EXPECT_EQ(actual.routed_phase_policy, expected.routed_phase_policy);
    EXPECT_EQ(actual.routed_decode_assignment_policy, expected.routed_decode_assignment_policy);
    EXPECT_EQ(actual.routed_prefill_assignment_policy, expected.routed_prefill_assignment_policy);
}

/**
 * @brief Authenticate the published named inventory through canonical overlay installation.
 * @param authored Frozen original request, including typed domain authorities without duplicate names.
 * @param resolved Runner configuration after inventory binding and domain publication.
 *
 * Raw domain_definitions is a publication surface, not the complete authored
 * topology. The canonical union includes routed/dense domains. Reinstalling a
 * copy of the actual bound plan validates original intent, exact selectors and
 * rank ownership; complete field comparisons then reject missing/foreign aliases.
 */
void expectResolvedDomainInventory(OrchestrationConfig authored, const OrchestrationConfig &resolved)
{
    const auto requested = authored.executionDomainDefinitions();
    ASSERT_EQ(resolved.domain_definitions.size(), requested.size());
    if (!authored.moe_routed_expert_plan)
    {
        EXPECT_EQ(resolved.domain_definitions.size(), authored.domain_definitions.size());
        return; // Existing PP interval/device checks below authenticate ordinary named stages.
    }
    ASSERT_NE(resolved.moe_routed_expert_plan, nullptr);
    const auto errors = installResolvedMoEExpertOverlayPlan(authored,
        std::make_shared<MoERoutedExpertPlacementPlan>(*resolved.moe_routed_expert_plan),
        MoEExpertOverlayPlanInstallOrigin::UserDeclaredDomains);
    ASSERT_TRUE(errors.empty()) << (errors.empty() ? "" : errors.front());
    const auto expected = authored.executionDomainDefinitions();
    ASSERT_EQ(expected.size(), requested.size());
    const auto effective = resolved.executionDomainDefinitions();
    ASSERT_EQ(effective.size(), requested.size());
    std::set<std::string> observed;
    for (const auto &published : resolved.domain_definitions)
    {
        const auto actual = published.toExecutionDomainDefinition();
        ASSERT_TRUE(observed.insert(actual.name).second) << actual.name;
        const auto found = std::find_if(expected.begin(), expected.end(), [&](const auto &domain) {
            return domain.name == actual.name;
        });
        ASSERT_NE(found, expected.end()) << actual.name;
        // The legacy TPScope wrapper cannot represent SINGLE and intentionally
        // publishes AUTO. Authenticate that exact projection independently from
        // the effective typed authority; never relax SINGLE to arbitrary scope.
        expectSameDomainDefinition(actual, DomainDefinition::fromExecutionDomainDefinition(*found)
            .toExecutionDomainDefinition());
        const auto typed = std::find_if(effective.begin(), effective.end(), [&](const auto &domain) {
            return domain.name == actual.name;
        });
        ASSERT_NE(typed, effective.end()) << actual.name;
        expectSameDomainDefinition(*typed, *found);
    }
}

/**
 * @brief Admit one explicit source pair and retain only setup evidence.
 * @param cell Canonical main topology/precision/storage declaration plus exact predictor source.
 * @param execution Fixed1/R1/retained1 or the separate inactive-source control.
 *
 * All assertions occur while the real runner and its certificate remain alive.
 * unique_ptr teardown covers fatal assertion/exception paths. No device-state
 * probe, model prefill/decode, snapshot copy or gold regeneration is invoked.
 */
void admitPairedSource(const Qwen4ExpMTPAdmissionCase &cell, Qwen4ExpMTPAdmissionExecution execution)
{
    const bool active = execution == Qwen4ExpMTPAdmissionExecution::FixedOne;
    auto config = cell.config(execution);
    publishOrchestrationStartupPolicy(config);
    initCPUBackend(-1);
    for (const auto &participant : cell.main.topology.participants)
    {
        const auto type = participant.address.device_type;
        if (type == DeviceType::CPU) continue;
        auto *backend = type == DeviceType::CUDA ? getCUDABackend() : getROCmBackend();
        if (!backend || participant.address.device_ordinal >= backend->deviceCount())
            GTEST_SKIP() << "Missing declared admission backend " << participant.address.toShortString();
    }
    ASSERT_TRUE(std::filesystem::is_regular_file(config.model_path)) << config.model_path;
    ASSERT_TRUE(config.shared_predictor);
    ASSERT_TRUE(std::filesystem::is_regular_file(config.shared_predictor->path())) << config.shared_predictor->path();
    const char *artifact_root = std::getenv("LLAMINAR_PRODUCTION_PARITY_ARTIFACT_ROOT");
    const auto base = artifact_root ? std::filesystem::path(artifact_root)
        : std::filesystem::current_path() / "parity-results" / "qwen-flash-next" / "mtp-admission";
    ASSERT_TRUE(base.is_absolute()) << "Admission artifact root must be absolute";
    const std::string identity = active ? cell.testName() : cell.testName() + "_ExplicitNoMTP";
    const auto output = base / identity;
    // As in the canonical parity fixture, this cell owns only its generated
    // child directory. Repeated CTest runs cannot inherit an earlier success.
    std::error_code artifact_error;
    std::filesystem::remove_all(output, artifact_error);
    ASSERT_FALSE(artifact_error) << artifact_error.message();
    ASSERT_TRUE(std::filesystem::create_directories(output, artifact_error));
    ASSERT_FALSE(artifact_error) << artifact_error.message();
    config.prefix_cache.disk_dir = (output / "prefix-cache-rank-0").string();
    // Freeze requested domain intent before the runner resolves shared plan metadata.
    auto authored_topology = config;
    if (config.moe_routed_expert_plan)
        authored_topology.moe_routed_expert_plan =
            std::make_shared<MoERoutedExpertPlacementPlan>(*config.moe_routed_expert_plan);
    auto runner = TestOrchestrationHelper::create(config);
    ASSERT_NE(runner, nullptr);
    runner->disableSnapshotCapture();
    ASSERT_TRUE(runner->initialize()) << runner->lastError();
    ASSERT_TRUE(runner->isInitialized());
    EXPECT_EQ(runner->executionPlan().layerCount(), 48);
    EXPECT_EQ(runner->config().model_path, config.model_path);
    EXPECT_EQ(runner->config().shared_predictor, config.shared_predictor);
    EXPECT_EQ(runner->config().max_seq_len, 4096);
    EXPECT_EQ(runner->config().batch_size, 1);
    const auto &runtime = runner->executionPlan().runtime;
    EXPECT_EQ(runtime.max_seq_len, 4096);
    EXPECT_EQ(runtime.mtp.enabled, active);
    EXPECT_EQ(runtime.mtp.draft_tokens, 1);
    EXPECT_EQ(runtime.mtp.graph_capacity_draft_tokens, active ? 1 : 0);
    EXPECT_EQ(runtime.mtp.max_request_batch, 1);
    EXPECT_EQ(runtime.mtp.depth_policy.mode, MTPDepthPolicyMode::Fixed);
    EXPECT_EQ(resolveMTPMaximumExecutionDraftDepth(runtime.mtp), 1);
    EXPECT_EQ(retainsMTPGraphCapacity(runtime.mtp), active);
    expectResolvedDomainInventory(std::move(authored_topology), runner->config());
    EXPECT_EQ(runner->config().pp_stage_definitions.size(), config.pp_stage_definitions.size());
    EXPECT_EQ(runner->config().ngram_table_policy, config.ngram_table_policy);
    if (cell.main.topology.kind == ModelParityTopologyKind::RankLocalPipelineParallel)
    {
        const auto &plan = runner->executionPlan();
        EXPECT_EQ(plan.local_pp_layer_boundaries, (std::vector<int>{0, 8, 20, 48}));
        ASSERT_EQ(plan.local_pp_devices.size(), 3u);
        const std::array types{DeviceType::CUDA, DeviceType::ROCm, DeviceType::CPU};
        for (std::size_t index = 0; index < types.size(); ++index)
        {
            EXPECT_EQ(plan.local_pp_devices[index].device_type, types[index]);
            EXPECT_EQ(plan.local_pp_devices[index].device_ordinal, 0);
        }
    }
    EXPECT_TRUE(runner->getSnapshotKeys().empty());
    const auto reuse = runner->modelContextReuseContract();
    ASSERT_TRUE(reuse);
    ASSERT_NE(reuse->context, nullptr);
    ASSERT_NE(reuse->physical_memory_authority, nullptr);
    ASSERT_NE(reuse->physical_memory_authority->admission(), nullptr);
    const auto model = reuse->context;
    EXPECT_EQ(runner->modelContextForDiagnostics(), model.get());
    EXPECT_EQ(model->path(), config.model_path);
    EXPECT_EQ(model->totalBlockCount(), 48);
    const auto &loader = model->concreteLoader();
    const auto *main = loader.ggufDirectory();
    const auto *predictor = loader.ggufPredictorDirectory();
    const auto *manifest = loader.predictorManifest();
    ASSERT_NE(main, nullptr);
    ASSERT_NE(predictor, nullptr);
    ASSERT_NE(manifest, nullptr);
    EXPECT_EQ(main->block_count, 48u);
    EXPECT_EQ(predictor->block_count, 49u);
    EXPECT_EQ(manifest->sourceLayer(), 48u);
    EXPECT_EQ(manifest->attention(), Qwen4ExpPredictorAttention::Dense);
    EXPECT_EQ(manifest->hiddenNorm(), Qwen4ExpPredictorHiddenNorm::PerBranch);
    EXPECT_FALSE(manifest->sparseSelector());
    EXPECT_EQ(manifest->predictorMetadata().cacheOrigin(), MTPShiftedCacheOrigin::VisibleZeroHiddenTokenZero);
    EXPECT_EQ(loader.predictorMaterialization(), active ? SharedPredictorMaterialization::Active : SharedPredictorMaterialization::Inactive);
    const auto paths = model->artifactPaths();
    ASSERT_EQ(main->split_paths.size(), 4u);
    ASSERT_EQ(paths.size(), 5u);
    EXPECT_TRUE(std::equal(main->split_paths.begin(), main->split_paths.end(), paths.begin()));
    EXPECT_EQ(paths.front(), config.model_path);
    EXPECT_EQ(paths.back(), config.shared_predictor->path());
    const auto names = loader.materializationNames();
    for (const auto &source : manifest->sources())
    {
        const bool selected = std::find(names.begin(), names.end(), source.tensor.name) != names.end();
        if (source.source == Qwen4ExpTensorSource::Predictor)
        {
            EXPECT_EQ(selected, active && source.consumed) << source.tensor.name;
            EXPECT_EQ(main->findTensor(source.tensor.name), nullptr);
        }
        else
        {
            const auto *original = main->findTensor(source.tensor.name);
            ASSERT_NE(original, nullptr);
            EXPECT_EQ(predictor->findTensor(source.tensor.name), nullptr);
            EXPECT_EQ(source.tensor.dimensions, original->dimensions);
            EXPECT_EQ(source.tensor.offset, original->offset);
            EXPECT_EQ(source.tensor.split_idx, original->split_idx);
            EXPECT_EQ(source.tensor.type, original->type);
            EXPECT_EQ(source.tensor.size_bytes, original->size_bytes);
        }
    }
    if (active)
    {
        const auto learned = requireMTPWeightManifest(loader, "qwen4exp", 48);
        ASSERT_TRUE(learned.available);
        ASSERT_EQ(learned.depths.size(), 1u);
        EXPECT_EQ(learned.depths.front().depth_index, 0);
        EXPECT_EQ(learned.depths.front().source_layer_index, 48);
        EXPECT_EQ(learned.flash_next, std::optional(manifest->predictorMetadata()));
        // The typed composer enforces depth0->cache0 during actual initialized
        // graph construction; no diagnostic reads mutable GPU cache frontiers.
        const auto records = preparedPredictorExpertRecords(*reuse);
        EXPECT_GT(records, 0u);
        const bool whole_source_cpu = cell.main.topology.participants.size() == 1 &&
            cell.main.topology.participants.front().address.device_type == DeviceType::CPU;
        if (whole_source_cpu)
        {
            // Whole-source CPU setup owns all three original512-expert banks
            // directly in PreparedWeightStore, without overlay registry aliases.
            EXPECT_EQ(records, 3u * predictor->findTensor("blk.48.ffn_gate_exps.weight")->dimensions[2]);
        }
    }
    else EXPECT_EQ(preparedPredictorExpertRecords(*reuse), 0u);
    const auto manager = model->concreteWeightManager();
    const auto store = manager->preparedWeightStoreIfInitialized();
    ASSERT_NE(store, nullptr);
    EXPECT_GT(store->size(), 0u);
    EXPECT_EQ(store->physicalMemoryAuthority(), reuse->physical_memory_authority);
    const auto placement = model->resolvedImmutableRowTablePlacement();
    ASSERT_NE(placement, nullptr);
    ASSERT_NE(runtime.ngram_table_placement, nullptr);
    EXPECT_EQ(*runtime.ngram_table_placement, *placement);
    const auto expected_tier = cell.main.ngram_table_policy->storage == ImmutableRowStoragePreference::RAM
        ? ImmutableRowStorage::RAM : ImmutableRowStorage::MemoryMapped;
    EXPECT_EQ(placement->storage(), expected_tier);
    ASSERT_EQ(placement->allocators().size(), 1u);
    const auto host = placement->allocators().front();
    EXPECT_TRUE(host.device.is_cpu());
    const auto table = model->hostImmutableRowProvider();
    ASSERT_NE(table, nullptr);
    EXPECT_EQ(table->wholeStorage(), expected_tier);
    EXPECT_EQ(table->source().tensorBytes(), 28800138240ull);
    EXPECT_EQ(table->source().tensorBytes(), placement->extent().nativeBytes());
    // Observe actual retained leases once, through the closed provider; the
    // source extent and admission BOM are not a second live-byte authority.
    const auto retained_owner = table->retainedOwnerObservation();
    EXPECT_EQ(retained_owner.allocator(), host);
    EXPECT_EQ(retained_owner.authority().get(), reuse->physical_memory_authority.get());
    EXPECT_EQ(retained_owner.hostPlacement(), placement->hostPlacement());
    EXPECT_EQ(retained_owner.lookupStagingBytes(), 0u);
    std::size_t table_owners = 0;
    const auto &memory = reuse->physical_memory_authority;
    for (const auto &bom : memory->admission()->plan().resources())
    {
        EXPECT_EQ(bom.bytes(PhysicalMemoryOwner::GraphSnapshotArena), 0u);
        if (bom.bytes(PhysicalMemoryOwner::ImmutableTableStorage))
        {
            ++table_owners;
            EXPECT_EQ(bom.resource().world_rank, host.world_rank);
            EXPECT_EQ(bom.resource().device, host.device);
            EXPECT_EQ(bom.bytes(PhysicalMemoryOwner::ImmutableTableStorage), retained_owner.tableStorageBytes());
        }
        EXPECT_EQ(model->residentImmutableRowTableBytes(bom.resource().device), 0u);
    }
    EXPECT_EQ(table_owners, 1u);
    EXPECT_EQ(memory->claimedBytes(host.device, PhysicalMemoryOwner::ImmutableTableStorage,
              PhysicalMemoryMaterializationKind::NewAllocation), retained_owner.tableStorageBytes());
    std::ofstream attestation(output / "admission_memory.txt");
    ASSERT_TRUE(attestation.is_open());
    attestation << memory->rankAttestationSummary(true);
    attestation.flush();
    ASSERT_TRUE(attestation.good());
    std::ofstream csv(output / "admission.csv");
    ASSERT_TRUE(csv.is_open());
    csv << "cell,main_source,predictor_source,main_layers,source_layer,cache_layer,mtp_active,retained_depth,snapshots_enabled,numerical_inference_executed,native_table_bytes\n"
        << identity << ',' << config.model_path << ',' << config.shared_predictor->path()
        << ",48,48," << (active ? 0 : -1) << ',' << active << ',' << (active ? 1 : 0) << ",0,0," << table->source().tensorBytes() << '\n';
    csv.flush();
    ASSERT_TRUE(csv.good());
    runner->shutdown();
    EXPECT_FALSE(runner->isInitialized());
}

/** @brief Eight full-model source admissions; no parity/gold fixture lifecycle. */
class Qwen4ExpMTPAdmissionTest : public ::testing::TestWithParam<Qwen4ExpMTPAdmissionCase> {};
/** @test Actual fixed1/R1 paired source is prepared under its complete serving certificate. */
TEST_P(Qwen4ExpMTPAdmissionTest, PairedSourceInitializesWithoutReferencePack)
{ admitPairedSource(GetParam(), Qwen4ExpMTPAdmissionExecution::FixedOne); }
INSTANTIATE_TEST_SUITE_P(Qwen4ExpMTPAdmission, Qwen4ExpMTPAdmissionTest,
    ::testing::ValuesIn(qwen4ExpMTPAdmissionCases()),
    [](const ::testing::TestParamInfo<Qwen4ExpMTPAdmissionCase> &info) { return info.param.testName(); });

/** @test Selecting a sidecar with explicit no-MTP admits metadata but no predictor materialization. */
TEST(Qwen4ExpMTPPairedSourceControl, CPU0MmapExplicitNoMTP)
{
    const auto &cases = qwen4ExpMTPAdmissionCases();
    const auto found = std::find_if(cases.begin(), cases.end(), [](const auto &cell) {
        return cell.main.topology.kind == ModelParityTopologyKind::SingleDevice &&
               cell.main.ngram_table_policy->storage == ImmutableRowStoragePreference::MemoryMapped;
    });
    ASSERT_NE(found, cases.end());
    admitPairedSource(*found, Qwen4ExpMTPAdmissionExecution::Inactive);
}
} // namespace
} // namespace llaminar2::test::parity::qwen4exp
