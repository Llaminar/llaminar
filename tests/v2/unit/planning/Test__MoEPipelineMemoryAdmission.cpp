/**
 * @file Test__MoEPipelineMemoryAdmission.cpp
 * @brief Device-free GGUF-to-PMA regressions for four/eight-device MoE pipelines.
 *
 * Public parsing, model manifests, stage projection and aggregate admission are
 * exercised together. Sparse GGUF fixtures contain directory metadata only; no
 * accelerator, model payload or inference is substituted. Runtime setup tests
 * construct the same histogram, residency and empty prepared-bank owners used
 * by production, then hand them to the checked pipeline binding.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "planning/OriginalRankCapacityExchange.h"
#include "collective/NativeCPUCollectiveProgram.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include "planning/OrchestrationCandidateAdmission.h"
#include "planning/PlanningForwardWeightWork.h"
#include "planning/MoEOverlayPlanningInputs.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"
#include "execution/mpi_orchestration/ExecutionPlanBuilder.h"
#include "execution/moe/MoEOverlayResidencySetup.h"
#include "execution/moe/MoEOverlayPipelineStageBinding.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayEconomyCalibrationPlanner.h"
#include "execution/moe/MoEPipelinePreparedPlan.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/runner/PreparedWeightAuthorityIdentity.h"
#include "loaders/ModelContext.h"
#include "../../mocks/MockMPIContext.h"
#include "utils/MPIContext.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/QuantizedVerifierFormats.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "backends/GPUDeviceProcessLifetime.h"
#include <array>
#include <map>
#include <set>

namespace llaminar2::test
{
namespace
{
    /** @return Production model/loader adapter over a sparse directory; no payload or mmap is loaded. */
    std::shared_ptr<ModelContext> metadataContext(const std::string &path)
    {
        ModelContextConfig config;
        config.mpi_ctx = MPIContextFactory::self();
        config.use_mmap = false;
        auto context = ModelContext::create(path, config);
        if (!context) throw std::runtime_error("Cannot construct pipeline metadata context");
        return context;
    }

    /** @return Observed distinct GPU groups and one shared host, without backend discovery. */
    ClusterInventory inventory(int width)
    {
        ClusterInventory result;
        result.world_size = 1;
        RankInventory rank;
        rank.rank = rank.node_id = rank.local_rank = 0;
        rank.hostname = "pipeline-host";
        rank.cpu.numa_node = 0;
        rank.numa_nodes = 1;
        rank.cpu_cores = rank.cpu_worker_threads = 8;
        rank.cpu_execution = kSyntheticCPUExecutionGeometry;
        rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = rank.cpu_memory_bytes = 256ULL << 30;
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            for (int ordinal = 0; ordinal < width; ++ordinal)
                rank.gpus.push_back({.type = backend, .local_device_id = ordinal,
                    .memory_bytes = 32ULL << 30, .free_memory_bytes = 24ULL << 30,
                    .compute_units = 64, .uuid = std::string(deviceTypeToString(backend)) + "-observed-" + std::to_string(ordinal), .numa_node = 0});
            auto &matrix = backend == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
            (backend == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count) = width;
            matrix.assign(size_t(width * width), false);
            for (int ordinal = 0; ordinal < width; ++ordinal) matrix[ordinal * width + ordinal] = true;
        }
        result.ranks.push_back(std::move(rank));
        result.buildNodeAggregations();
        return result;
    }

    /** @return Authored two-stage topology parsed through the public CLI grammar. */
    AutomaticOrchestrationCandidate proposal(const PlanningModelSource &source, int width, bool reverse)
    {
        const auto domain = [width](std::string name, bool rocm) {
            std::string members;
            for (int ordinal = width - 1; ordinal >= 0; --ordinal)
                members += (members.empty() ? "" : ",") + std::string(rocm ? "rocm:" : "cuda:") + std::to_string(ordinal);
            return name + "=" + members + ";scope=rank_local;backend=" + (rocm ? "rccl" : "nccl") + ";owner=0";
        };
        std::vector<std::string> args{"llaminar2", "--define-domain", domain("head", reverse),
            "--define-domain", domain("tail", !reverse), "--pp-stage", "0=head:0-0", "--pp-stage", "1=tail:1-1"};
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        config.model_path = source.path();
        config.max_seq_len = 128;
        config.moe_routed_prefill.overlay_segment_rows = 32;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        config.prefix_cache.enabled = true;
        config.prefix_cache.storage_mode = PrefixCacheStorageMode::Tiered;
        config.prefix_cache.ram_budget_bytes = 16ULL << 30;
        config.prefix_cache.disk_budget_bytes = 32ULL << 30;
        config.prefix_cache.disk_dir = "/metadata-only/shared-prefix";
        return {OrchestrationStrategy::PipelineParallel, ExecutionRankMembership(inventory(width), {0}), std::move(config)};
    }

    /** @return Explicit capture inventory small enough for fast metadata-only regression. */
    OrchestrationCandidateMemoryPolicy policy()
    { return {.prefill_bucket_rows = {1, 8, 32}, .minimum_prefill_sequence_rows = 1, .maximum_cached_prefill_buckets = 16, .retained_host_services = planningHostServiceBudgets(1)}; }

    /** @return Every supported source codebook, excluding the runtime-only Q8_1 representation. */
    std::vector<TensorType> formats()
    {
        std::vector<TensorType> result{TensorType::FP32, TensorType::FP16, TensorType::BF16};
        for (const auto &format : quantizedVerifierFormats())
            if (format.tensor_type != TensorType::Q8_1) result.push_back(format.tensor_type);
        return result;
    }

    /** @return Source-resolved model dimensions with an independently specified stage interval. */
    MoERoutedExpertModelMetadata metadataFor(const ModelLoader &loader, int first, int main_end, int layers)
    {
        const auto &arch = loader.architecture();
        return {.num_layers = layers, .num_experts = loader.getInt(arch + ".expert_count", 0),
            .d_model = loader.getInt(arch + ".embedding_length", 0),
            .routed_intermediate_size = loader.getInt(arch + ".expert_feed_forward_length", 0),
            .shared_intermediate_size = loader.getInt(arch + ".expert_feed_forward_length", 0),
            .main_inference_layer_count = main_end, .first_model_layer = first};
    }

    /** @return Concrete placements frozen only after their original quota certificate is installed. */
    MoERoutedExpertPlacementPlan freeze(const AdmittedMoEPipelineStage &stage, const MoERoutedExpertModelMetadata &metadata)
    {
        const auto admitted = MoEOverlayCapacityResolver::installResolvedQuotas(
            *stage.topology().config().moe_routed_expert_plan, *stage.capacity());
        return MoERoutedExpertPlacementPlanner::plan(admitted, metadata).planned_plan;
    }

    /** @brief Independent expected expert work, summed only over owners of each main layer. */
    void expectWork(const AdmittedOrchestrationCandidate &admitted, const PlanningModelMetadata &metadata,
        RoutedExpertComputePolicy compute, PlanningMainForwardPhase phase, int width)
    {
        const auto work = compilePlanningForwardWeightWork(metadata, admitted, phase);
        ASSERT_EQ(work.size(), size_t(width * 2));
        std::map<int, double> routed_rows;
        for (const auto &participant : work)
        {
            ASSERT_EQ(participant.first_layer, participant.last_layer);
            ASSERT_EQ(participant.routed.size(), 1u);
            const auto &routed = participant.routed.front();
            EXPECT_EQ(routed.layer, participant.first_layer);
            routed_rows[routed.layer] += routed.uniformExpectation(7).routed_rows;
            if (compute == RoutedExpertComputePolicy::GateUpOwnedDownColumns)
            {
                ASSERT_TRUE(routed.projection_ownership);
                EXPECT_EQ(routed.projectionMatrix(WeightRole::MoEExpertDown).rows, size_t(256 / width));
            }
        }
        ASSERT_EQ(routed_rows.size(), 2u);
        const double copies = compute == RoutedExpertComputePolicy::Replicated ? width : 1;
        for (const auto &[layer, rows] : routed_rows)
            EXPECT_DOUBLE_EQ(rows, metadata.memoryProfile().expert_used_count * 7.0 * copies) << layer;
    }
}

/** @brief Physical participant count is an explicit preflight axis. */
class MoEPipelineMemoryAdmission : public ::testing::TestWithParam<int> {};

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test All source formats and serving modes retain complete compact stage authority. */
TEST_P(MoEPipelineMemoryAdmission, AllFormatsMTPModesAndBothVendorOrders)
{
    const int width = GetParam() / 2;
    // Q8_1 is a runtime-prepared format, not a supported GGUF source encoding.
    EXPECT_THROW(PlanningGGUFFixture::sourceType(TensorType::Q8_1), std::invalid_argument);
    ASSERT_EQ(formats().size(), 23u);
    for (const auto format : formats())
    {
        PlanningGGUFFixture file(true, true, PlanningGGUFFixture::sourceType(format));
        PlanningModelSource source(file.path());
        for (const bool reverse : {false, true})
            for (const int lane : {0, 1, 2})
                for (const auto compute : {RoutedExpertComputePolicy::Apportioned,
                    RoutedExpertComputePolicy::Automatic, RoutedExpertComputePolicy::GateUpOwnedDownColumns})
                {
                    SCOPED_TRACE(::testing::Message() << GetParam() << '/' << int(format) << '/' << reverse << '/' << lane << '/' << int(compute));
                    auto input = proposal(source, width, reverse);
                    input.config.routed_expert_compute_policy = compute;
                    input.config.mtp_activation_policy = lane == 0 ? MTPActivationPolicy::Disabled : MTPActivationPolicy::Automatic;
                    input.config.mtp.depth_policy.mode = lane == 1 ? MTPDepthPolicyMode::Fixed : MTPDepthPolicyMode::Dynamic;
                    input.config.mtp.draft_tokens = 7;
                    input.config.mtp.graph_capacity_draft_tokens = lane ? 15 : 0;
                    const auto original = serializeOrchestrationConfig(input.config);
                    const auto admitted = AdmittedOrchestrationCandidate::admit(input, source, policy());
                    ASSERT_TRUE(admitted.pipelineMemory());
                    ASSERT_FALSE(admitted.overlayCapacity());
                    EXPECT_FALSE(admitted.config().moe_routed_expert_plan);
                    EXPECT_TRUE(admitted.rankPlans().front().usesLocalPP());
                    EXPECT_TRUE(admitted.physicalAdmission().plan().fits());
                    ASSERT_EQ(admitted.pipelineMemory()->stages().size(), 2u);
                    ASSERT_EQ(admitted.devicePlans().size(), size_t(GetParam()));
                    for (int index = 0; index < 2; ++index)
                    {
                        const auto &stage = admitted.pipelineMemory()->stages()[index];
                        EXPECT_EQ(stage.capacity()->physical_memory_admission.get(), &admitted.physicalAdmission());
                        EXPECT_EQ(stage.rankPlan().first_layer, index);
                        EXPECT_EQ(stage.rankPlan().last_layer, index);
                        EXPECT_EQ(stage.graphFamily().first_model_layer, index);
                        EXPECT_EQ(stage.graphFamily().main_layer_count, 1);
                        EXPECT_EQ(stage.graphFamily().mtp_source_layers,
                            index == 1 && lane ? std::vector<int>{2} : std::vector<int>{});
                        EXPECT_EQ(stage.weightManifest().size(), size_t(index == 1 && lane ? 2 : 1));
                        EXPECT_EQ(stage.weightManifest().front().layer_idx, index);
                        EXPECT_EQ(&admitted.expertLayer(index)->capacity, stage.capacity().get());
                        EXPECT_EQ(stage.rankPlan().runtime.resident_graph_rows, admitted.rankPlans().front().runtime.resident_graph_rows);
                        for (const auto &tier : stage.topology().config().moe_routed_expert_plan->routed_tiers)
                            EXPECT_TRUE(tier.resolved_live_experts_per_layer.empty());
                    }
                    if (lane) EXPECT_EQ(admitted.expertLayer(2)->capacity.physical_memory_admission.get(), &admitted.physicalAdmission());
                    else EXPECT_THROW((void)admitted.expertLayer(2), std::invalid_argument);
                    EXPECT_THROW((void)admitted.expertLayer(-1), std::invalid_argument);
                    EXPECT_THROW((void)admitted.expertLayer(3), std::invalid_argument);
                    expectWork(admitted, source.metadata(), compute, PlanningMainForwardPhase::Prefill, width);
                    expectWork(admitted, source.metadata(), compute, PlanningMainForwardPhase::Decode, width);
                    EXPECT_EQ(serializeOrchestrationConfig(input.config), original);
                }
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Shared host and later-stage GPU pressure cannot admit only the leading child. */
TEST_P(MoEPipelineMemoryAdmission, AggregatePressureAndFailureLeaveInputUnchanged)
{
    PlanningGGUFFixture file(true, true);
    PlanningModelSource source(file.path());
    for (const bool reverse : {false, true})
    {
        auto input = proposal(source, GetParam() / 2, reverse);
        const auto original = serializeOrchestrationConfig(input.config);
        for (const bool host_pressure : {false, true})
        {
            auto observed = input.membership.inventory();
            if (host_pressure)
                observed.ranks.front().cpu.free_memory_bytes = size_t(GetParam() / 2 + 1) * (16ULL << 30);
            else
                observed.ranks.front().gpus.back().free_memory_bytes = 0;
            observed.buildNodeAggregations();
            auto rejected = input;
            rejected.membership = ExecutionRankMembership(observed, {0});
            EXPECT_THROW((void)AdmittedOrchestrationCandidate::admit(rejected, source, policy()), PhysicalMemoryCapacityExhausted);
            EXPECT_EQ(serializeOrchestrationConfig(rejected.config), original);
        }
        // A failure cannot poison shared-owner deduplication or publish partial grants.
        EXPECT_TRUE(AdmittedOrchestrationCandidate::admit(input, source, policy()).physicalAdmission().plan().fits());
        EXPECT_EQ(serializeOrchestrationConfig(input.config), original);
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

/** @test Invalid capture policy remains an input error and cannot select a smaller shape. */
TEST_P(MoEPipelineMemoryAdmission, InvalidPolicyIsNotCapacityExhaustion)
{
    PlanningGGUFFixture file(true, true);
    PlanningModelSource source(file.path());
    const auto input = proposal(source, GetParam() / 2, false);
    for (const int defect : {0, 1, 2})
    {
        auto settings = policy();
        if (defect == 0) settings.prefill_bucket_rows.push_back(0);
        if (defect == 1) settings.prefill_bucket_rows.clear();
        if (defect == 2) settings.maximum_cached_prefill_buckets = 1;
        EXPECT_THROW((void)AdmittedOrchestrationCandidate::admit(input, source, settings), std::invalid_argument);
    }
}

INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelineMemoryAdmission, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });

/** @brief Runtime construction retains the same explicit four/eight-participant axis. */
class MoEPipelineResidencySetup : public ::testing::TestWithParam<int> {};

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Every source format and maintenance/MTP policy constructs exact stage owners. */
TEST_P(MoEPipelineResidencySetup, AllFormatsModesAndBothVendorOrders)
{
    for (const auto format : formats())
    {
        PlanningGGUFFixture file(true, true, PlanningGGUFFixture::sourceType(format));
        PlanningModelSource source(file.path());
        for (const bool reverse : {false, true})
            for (const int lane : {0, 1, 2})
                for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe, MoERebalanceRuntimeMode::Dynamic})
                    for (const auto compute : {RoutedExpertComputePolicy::Apportioned, RoutedExpertComputePolicy::GateUpOwnedDownColumns})
                    {
                        SCOPED_TRACE(::testing::Message() << GetParam() << '/' << int(format) << '/' << reverse << '/' << lane
                            << '/' << int(mode) << '/' << int(compute));
                        auto input = proposal(source, GetParam() / 2, reverse);
                        input.config.moe_rebalance.mode = mode;
                        input.config.routed_expert_compute_policy = compute;
                        input.config.mtp_activation_policy = lane ? MTPActivationPolicy::Automatic : MTPActivationPolicy::Disabled;
                        input.config.mtp.depth_policy.mode = lane == 1 ? MTPDepthPolicyMode::Fixed : MTPDepthPolicyMode::Dynamic;
                        input.config.mtp.draft_tokens = 7;
                        input.config.mtp.graph_capacity_draft_tokens = lane ? 15 : 0;
                        const auto admitted = AdmittedOrchestrationCandidate::admit(input, source, policy());
                        ASSERT_TRUE(admitted.pipelineMemory());
                        auto memory = std::make_shared<PhysicalMemoryAuthority>(admitted.pipelineMemory()->physicalAdmission(), 0);
                        const auto unclaimed = memory->rankAttestationSummary();
                        std::vector<std::shared_ptr<const MoEOverlayPipelineStageBinding>> bindings;
                        for (int index = 0; index < 2; ++index)
                        {
                            const auto &stage = admitted.pipelineMemory()->stages()[index];
                            const int layers = index == 1 && lane ? 2 : 1;
                            const auto metadata = metadataFor(source.loader(), index, index + 1, layers);
                            const auto plan = freeze(stage, metadata);
                            const auto setup = MoEOverlayResidencySetup::create({
                                .plan = plan, .metadata = metadata, .loader = source.loader(),
                                .rebalance = stage.topology().config().moe_rebalance,
                                .storage_policy = stage.storagePolicy(), .capacity = *stage.capacity(), .memory = memory});
                            ASSERT_TRUE(setup.authority());
                            EXPECT_EQ(setup.authority()->maintenanceMode(), mode);
                            EXPECT_EQ(bool(setup.histogram()), mode != MoERebalanceRuntimeMode::Off);
                            EXPECT_EQ(bool(setup.interference()), mode == MoERebalanceRuntimeMode::Dynamic);
                            if (setup.histogram())
                            {
                                EXPECT_EQ(setup.histogram()->firstModelLayer(), index);
                                EXPECT_EQ(setup.histogram()->endModelLayer(), index + layers);
                                EXPECT_EQ(setup.histogram()->config().token_boundary_layer_idx, index);
                                EXPECT_EQ(setup.histogram()->config().num_layers, layers);
                                EXPECT_EQ(setup.histogram()->config().top_k, 2);
                            }
                            const auto &catalog = setup.authority()->economyLayerCatalog();
                            ASSERT_EQ(bool(catalog), mode == MoERebalanceRuntimeMode::Dynamic);
                            if (catalog)
                            {
                                EXPECT_EQ(catalog->firstModelLayer(), index);
                                EXPECT_EQ(catalog->layerCount(), size_t(layers));
                            }
                            const auto snapshot = setup.authority()->snapshot();
                            ASSERT_TRUE(snapshot);
                            const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                                [&](const auto &value) { return value.name == plan.effectiveBaseModelDomain(); });
                            ASSERT_NE(domain, plan.domains.end());
                            bindings.push_back(MoEOverlayPipelineStageBinding::seal(stage.topology().scope(), metadata,
                                domain->participants, {.authority = setup.authority(), .residency = setup.residency(), .histogram = setup.histogram()}));
                            bindings.back()->requireDestination(stage.topology().scope(), domain->participants, domain->weights, domain->backend, mode);
                            for (const int participant : setup.residency()->localParticipantIds())
                            {
                                const auto endpoint = setup.residency()->endpoint(participant);
                                ASSERT_TRUE(endpoint);
                                EXPECT_EQ(endpoint->firstModelLayer(), index);
                                EXPECT_EQ(endpoint->numLayers(), layers);
                                EXPECT_EQ(endpoint->numExperts(), 8);
                            }
                        }
                        EXPECT_NE(bindings[0]->owners().authority, bindings[1]->owners().authority);
                        EXPECT_NE(bindings[0]->owners().residency, bindings[1]->owners().residency);
                        // GPU controller state is never allocated or mirrored by this host setup boundary.
                        EXPECT_EQ(memory->rankAttestationSummary(), unclaimed);
                        EXPECT_TRUE(bindings[1]->owners().authority->retainsHistogram(bindings[1]->owners().histogram.get()));
                    }
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Equal-sized foreign intervals/certificates and unfinished quota policy fail before publication. */
TEST_P(MoEPipelineResidencySetup, RejectsForeignOwnersWithoutPublishingPartialState)
{
    PlanningGGUFFixture file(true, true);
    PlanningModelSource source(file.path());
    const auto input = proposal(source, GetParam() / 2, false);
    const auto admitted = AdmittedOrchestrationCandidate::admit(input, source, policy());
    const auto other = AdmittedOrchestrationCandidate::admit(input, source, policy());
    const auto &stage = admitted.pipelineMemory()->stages()[1];
    const auto valid_metadata = metadataFor(source.loader(), 1, 2, 2);
    const auto valid_plan = freeze(stage, valid_metadata);
    const auto memory = std::make_shared<PhysicalMemoryAuthority>(admitted.pipelineMemory()->physicalAdmission(), 0);
    const auto before = memory->rankAttestationSummary();
    for (int defect = 0; defect < 15; ++defect)
    {
        SCOPED_TRACE(defect);
        auto plan = valid_plan;
        auto metadata = valid_metadata;
        auto rebalance = stage.topology().config().moe_rebalance;
        auto storage = stage.storagePolicy();
        auto capacity = *stage.capacity();
        int rank = 0;
        if (defect == 0) metadata.first_model_layer = 0;
        if (defect == 1) metadata.main_inference_layer_count = 1;
        if (defect == 2) metadata.num_layers = 1;
        if (defect == 3) capacity.layer_footprints[1].layer_idx = 1;
        if (defect == 4) capacity.physical_memory_admission = other.pipelineMemory()->physicalAdmission();
        if (defect == 5) plan.placements.clear();
        if (defect == 6) plan.first_model_layer = 0;
        if (defect == 7) storage.overlay_world_size = 2;
        if (defect == 8) rebalance.mode = static_cast<MoERebalanceRuntimeMode>(99);
        if (defect == 9) rank = 1;
        if (defect == 10) plan.routed_tiers.front().resolved_live_experts_per_layer.clear();
        if (defect == 11) plan.routed_tiers.front().resolved_live_experts_per_layer.front() = 0;
        if (defect == 12) storage.migration_storage = MoEOverlayMigrationStorageKind::Disabled;
        if (defect == 13) metadata.num_experts = 7;
        if (defect == 14)
        {
            // The aggregate PMA also contains the sibling's devices. That
            // does not authorize this stage to consume their expert banks.
            for (auto &domain : plan.domains)
            {
                domain.backend = CollectiveBackendType::NCCL;
                for (auto &participant : domain.participants)
                    participant = GlobalDeviceAddress::cuda(participant.toLocalDeviceId().ordinal, 0);
            }
        }
        EXPECT_THROW((void)MoEOverlayResidencySetup::create({
            .plan = plan, .metadata = metadata, .loader = source.loader(), .rebalance = rebalance,
            .storage_policy = storage, .capacity = capacity, .memory = memory, .world_rank = rank}), std::invalid_argument);
        EXPECT_EQ(memory->rankAttestationSummary(), before);
    }
    EXPECT_NO_THROW((void)MoEOverlayResidencySetup::create({
        .plan = valid_plan, .metadata = valid_metadata, .loader = source.loader(),
        .rebalance = stage.topology().config().moe_rebalance, .storage_policy = stage.storagePolicy(),
        .capacity = *stage.capacity(), .memory = memory}));
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

// Every test of this suite requires a compiled GPU backend family.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelineResidencySetup, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });
#endif

/** @brief End-to-end metadata handoff keeps every declared physical participant. */
class MoEPipelinePreparedPlanTest : public ::testing::TestWithParam<int> {};

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Retained immutable placements create fresh stage owners in every source/MTP lane. */
TEST_P(MoEPipelinePreparedPlanTest, AllFormatsModesAndFreshRuntimeLifetimes)
{
    for (const auto format : formats())
    {
        PlanningGGUFFixture file(true, true, PlanningGGUFFixture::sourceType(format));
        PlanningModelSource source(file.path());
        auto model = metadataContext(file.path());
        for (const bool reverse : {false, true})
            for (const int lane : {0, 1, 2, 3})
                for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe,
                    MoERebalanceRuntimeMode::Dynamic})
                {
                    SCOPED_TRACE(::testing::Message() << GetParam() << '/' << int(format) << '/' << reverse << '/' << lane << '/' << int(mode));
                    auto input = proposal(source, GetParam() / 2, reverse);
                    input.config.moe_rebalance.mode = mode;
                    input.config.mtp_activation_policy = lane == 0 || lane == 3 ? MTPActivationPolicy::Disabled : MTPActivationPolicy::Automatic;
                    input.config.mtp.depth_policy.mode = lane == 1 ? MTPDepthPolicyMode::Fixed : MTPDepthPolicyMode::Dynamic;
                    input.config.mtp.draft_tokens = 7;
                    input.config.mtp.graph_capacity_draft_tokens = lane ? 15 : 0;
                    const auto admitted = AdmittedOrchestrationCandidate::admit(input, source, policy());
                    ExecutionPlanBuilder builder;
                    const auto topology = ResolvedRankOrchestration::resolve(input.config, source.metadata(), input.membership.inventory(), builder, 0);
                    const auto prepared = MoEPipelinePreparedPlan::create(admitted.pipelineMemory(), model);
                    ASSERT_EQ(prepared->stages().size(), 2u);
                    EXPECT_EQ(pipelineRoutedWeightAuthorityRequestIdentity(topology),
                        pipelineRoutedWeightAuthorityRequestIdentity(*prepared->admission()));
                    const auto memory = std::make_shared<PhysicalMemoryAuthority>(prepared->admission()->physicalAdmission(), 0);
                    const auto before = memory->rankAttestationSummary();
                    const auto context = MPIContextFactory::self();
                    auto first = prepared->createRuntimeBindings(model->concreteLoader(), memory, context, topology);
                    auto second = prepared->createRuntimeBindings(model->concreteLoader(), memory, context, topology);
                    ASSERT_EQ(first.size(), 2u);
                    ASSERT_EQ(second.size(), 2u);
                    for (int stage = 0; stage < 2; ++stage)
                    {
                        const auto &retained = prepared->stages()[stage];
                        const int owned_layers = stage == 1 && lane ? 2 : 1;
                        EXPECT_EQ(retained.metadata.first_model_layer, stage);
                        EXPECT_EQ(retained.metadata.num_layers, owned_layers);
                        ASSERT_EQ(retained.placement->placements.size(), size_t(owned_layers));
                        EXPECT_EQ(retained.placement->placements.back().layer, stage + owned_layers - 1);
                        EXPECT_EQ(first[stage]->owners().mpi, context);
                        EXPECT_EQ(first[stage]->owners().residency->localParticipantIds().size(), size_t(GetParam() / 2));
                        EXPECT_NE(first[stage]->owners().authority, second[stage]->owners().authority);
                        EXPECT_NE(first[stage]->owners().residency, second[stage]->owners().residency);
                        if (mode != MoERebalanceRuntimeMode::Off)
                            EXPECT_NE(first[stage]->owners().histogram, second[stage]->owners().histogram);
                        EXPECT_EQ(first[stage]->owners().authority->snapshot()->epoch, second[stage]->owners().authority->snapshot()->epoch);
                    }
                    std::weak_ptr<MoEOverlayResidencyAuthority> retired = first[0]->owners().authority;
                    first.clear();
                    EXPECT_TRUE(retired.expired());
                    EXPECT_TRUE(second[0]->owners().authority->snapshot()->valid());
                    second.clear();
                    EXPECT_EQ(memory->rankAttestationSummary(), before);
                    EXPECT_FALSE(model->concreteWeightManager()->preparedWeightStoreIfInitialized());
                }
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

/** @test Actual retained context identity reaches every fresh CPU/CUDA/ROCm pipeline registry. */
TEST(MoEPipelinePreparedSource, OriginalContextAuthenticatesEveryStageAndRejectsForeignRebinding)
{
    PlanningGGUFFixture file(true,true);
    PlanningModelSource source(file.path());
    auto model=metadataContext(file.path());
    auto foreign=metadataContext(file.path()); // Same path/metadata is not the same owner.
    const auto context=MPIContextFactory::self();
    const std::array<std::array<DeviceType,2>,3> backends{{
        {DeviceType::CPU,DeviceType::CUDA},{DeviceType::CUDA,DeviceType::ROCm},{DeviceType::ROCm,DeviceType::CPU}}};
    for(const auto& pair:backends)for(const int depth:{0,1,3,-1}) {
        SCOPED_TRACE(::testing::Message()<<int(pair[0])<<'/'<<int(pair[1])<<" depth="<<depth);
        auto input=proposal(source,1,false);
        input.config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
        auto observed=inventory(1);observed.ranks.front().cpu.numa_node=0;
        observed.ranks.front().numa_nodes=1;observed.buildNodeAggregations();
        input.membership=ExecutionRankMembership(observed,{0});
        ASSERT_EQ(input.config.domain_definitions.size(),2u);
        for(size_t stage=0;stage<pair.size();++stage) {
            auto& domain=input.config.domain_definitions[stage];
            const auto backend=pair[stage];
            domain.devices={backend==DeviceType::CPU?GlobalDeviceAddress::cpu(0):
                backend==DeviceType::CUDA?GlobalDeviceAddress::cuda(0):GlobalDeviceAddress::rocm(0)};
            domain.backend=backend==DeviceType::CPU?CollectiveBackendType::UPI:
                backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
        }
        input.config.mtp_activation_policy=depth==0?MTPActivationPolicy::Disabled:MTPActivationPolicy::Automatic;
        input.config.mtp.depth_policy.mode=depth<0?MTPDepthPolicyMode::Dynamic:MTPDepthPolicyMode::Fixed;
        input.config.mtp.draft_tokens=depth>0?depth:3;
        input.config.mtp.graph_capacity_draft_tokens=depth==0?0:15;
        const auto admitted=AdmittedOrchestrationCandidate::admit(input,source,policy());
        const auto prepared=MoEPipelinePreparedPlan::create(admitted.pipelineMemory(),model);
        auto memory=std::make_shared<PhysicalMemoryAuthority>(prepared->admission()->physicalAdmission(),0);
        const auto before=memory->rankAttestationSummary();
        ExecutionPlanBuilder builder;
        const auto topology=ResolvedRankOrchestration::resolve(input.config,source.metadata(),observed,builder,0);
        for(int lifetime=0;lifetime<2;++lifetime) {
            const auto bindings=prepared->createRuntimeBindings(model->concreteLoader(),memory,context,topology);
            ASSERT_EQ(bindings.size(),2u);
            for(const auto& binding:bindings) {
                const auto& registry=binding->owners().residency;
                ASSERT_NE(registry,nullptr);
                EXPECT_EQ(registry->originalModel(),ModelContextId{reinterpret_cast<uint64_t>(model.get())});
                const auto preparation=registry->projectionPreparation();std::string error;
                EXPECT_FALSE(registry->bindOriginalPreparedContext(foreign,preparation,&error));
                EXPECT_EQ(error,"Prepared source requires its actual original ModelContext identity");
                EXPECT_FALSE(registry->acquireOriginalPreparedSource());
                ASSERT_TRUE(registry->bindOriginalPreparedContext(model,preparation,&error))<<error;
                auto use=registry->acquireOriginalPreparedSource(&error);ASSERT_TRUE(use)<<error;
                EXPECT_EQ(&use->registry(),&model->concreteWeightManager()->expertGemmRegistry());
                EXPECT_EQ(use->preparation(),preparation);
                auto alias=std::shared_ptr<ModelContext>(model.get(),[](ModelContext*){});
                EXPECT_FALSE(registry->bindOriginalPreparedContext(alias,preparation,&error));
                EXPECT_EQ(error,"Prepared source cannot replace its original shared context/manager owner");
                EXPECT_TRUE(registry->bindOriginalPreparedContext(model,preparation,&error))<<error;
            }
        }
        EXPECT_EQ(memory->rankAttestationSummary(),before);
        EXPECT_FALSE(model->concreteWeightManager()->preparedWeightStoreIfInitialized());
    }
}

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Foreign source/certificate/policy cannot consume a retained stage's capacity. */
TEST_P(MoEPipelinePreparedPlanTest, RejectsForeignSourcesAndCapacityPolicyBeforePublication)
{
    PlanningGGUFFixture file(true, true);
    PlanningModelSource source(file.path());
    auto model = metadataContext(file.path());
    auto input = proposal(source, GetParam() / 2, false);
    input.config.mtp.graph_capacity_draft_tokens = 15;
    input.config.routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
    const auto admitted = AdmittedOrchestrationCandidate::admit(input, source, policy());
    const auto prepared = MoEPipelinePreparedPlan::create(admitted.pipelineMemory(), model);
    ExecutionPlanBuilder builder;
    const auto resolve = [&](const OrchestrationConfig &config) {
        return ResolvedRankOrchestration::resolve(config, source.metadata(), input.membership.inventory(), builder, 0);
    };
    const auto topology = resolve(input.config);
    const auto context = MPIContextFactory::self();
    const auto memory = std::make_shared<PhysicalMemoryAuthority>(admitted.pipelineMemory()->physicalAdmission(), 0);
    const auto before = memory->rankAttestationSummary();
    EXPECT_THROW((void)MoEPipelinePreparedPlan::create(nullptr, model), std::invalid_argument);
    EXPECT_THROW((void)prepared->createRuntimeBindings(source.loader(), memory, context, topology), std::invalid_argument);
    EXPECT_THROW((void)prepared->createRuntimeBindings(model->concreteLoader(), nullptr, context, topology), std::invalid_argument);
    EXPECT_THROW((void)prepared->createRuntimeBindings(model->concreteLoader(), memory, nullptr, topology), std::invalid_argument);
    EXPECT_THROW((void)prepared->createRuntimeBindings(model->concreteLoader(), memory,
        std::make_shared<MockMPIContext>(0, 2), topology), std::invalid_argument);
    const auto other_admission = AdmittedOrchestrationCandidate::admit(input, source, policy());
    EXPECT_THROW((void)prepared->createRuntimeBindings(model->concreteLoader(),
        std::make_shared<PhysicalMemoryAuthority>(other_admission.pipelineMemory()->physicalAdmission(), 0), context, topology), std::invalid_argument);
    for (int defect = 0; defect < 11; ++defect)
    {
        SCOPED_TRACE(defect);
        auto changed = input.config;
        if (defect == 0) changed.moe_rebalance.window_size *= 2;
        if (defect == 1) changed.moe_rebalance.max_window_size *= 2;
        if (defect == 2) changed.moe_rebalance.migration_transfer_slots += 1;
        if (defect == 3) changed.max_seq_len *= 2;
        if (defect == 4) changed.prefix_cache.ram_budget_bytes /= 2;
        if (defect == 5) changed.prefix_cache.disk_dir += "/foreign";
        if (defect == 6) changed.routed_expert_compute_policy = RoutedExpertComputePolicy::GateUpOwnedDownColumns;
        if (defect == 7) changed.moe_rebalance.mode = MoERebalanceRuntimeMode::Observe;
        if (defect == 8) changed.mtp.graph_capacity_draft_tokens = 7;
        if (defect == 9) changed = proposal(source, GetParam() / 2, true).config;
        if (defect == 10) changed.mtp.sidecar_dense_policy =
            changed.mtp.sidecar_dense_policy == MTPSidecarDensePolicy::TensorParallel
                ? MTPSidecarDensePolicy::ReplicatedPerParticipant : MTPSidecarDensePolicy::TensorParallel;
        EXPECT_THROW((void)prepared->createRuntimeBindings(model->concreteLoader(), memory, context, resolve(changed)), std::invalid_argument);
        EXPECT_EQ(memory->rankAttestationSummary(), before);
    }
    PlanningGGUFFixture foreign_file(true, true, GGUFTensorType::Q4_0);
    auto foreign = metadataContext(foreign_file.path());
    EXPECT_THROW((void)MoEPipelinePreparedPlan::create(admitted.pipelineMemory(), foreign), std::invalid_argument);
    // Replace only this test's sparse metadata file. An identical path cannot
    // disguise a changed projection format between discovery and model loading.
    std::filesystem::rename(foreign_file.path(), file.path());
    foreign = metadataContext(file.path());
    EXPECT_THROW((void)MoEPipelinePreparedPlan::create(admitted.pipelineMemory(), foreign), std::invalid_argument);
    std::weak_ptr<ModelContext> lifetime = model;
    model.reset();
    EXPECT_TRUE(lifetime.expired()); // The prepared plan cannot pin a retired model.
    EXPECT_THROW((void)prepared->createRuntimeBindings(source.loader(), memory, context, topology), std::invalid_argument);
    EXPECT_EQ(memory->rankAttestationSummary(), before);
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Active depth may change within retained physical geometry without readmission. */
TEST_P(MoEPipelinePreparedPlanTest, ReusesRetainedCapacityAcrossActiveMTPModes)
{
    PlanningGGUFFixture file(true, true);
    PlanningModelSource source(file.path());
    auto model = metadataContext(file.path());
    auto input = proposal(source, GetParam() / 2, false);
    input.config.mtp.graph_capacity_draft_tokens = 15;
    const auto admitted = AdmittedOrchestrationCandidate::admit(input, source, policy());
    const auto prepared = MoEPipelinePreparedPlan::create(admitted.pipelineMemory(), model);
    auto memory = std::make_shared<PhysicalMemoryAuthority>(prepared->admission()->physicalAdmission(), 0);
    ExecutionPlanBuilder builder;
    for (const int depth : {0, 1, 3, 7, 15})
    {
        auto config = input.config;
        config.mtp_activation_policy = depth ? MTPActivationPolicy::Automatic : MTPActivationPolicy::Disabled;
        config.mtp.depth_policy.mode = MTPDepthPolicyMode::Fixed;
        config.mtp.draft_tokens = std::max(1, depth);
        const auto current = ResolvedRankOrchestration::resolve(config, source.metadata(), input.membership.inventory(), builder, 0);
        auto runtime = current.rankPlan().runtime;
        prepared->admission()->publishRuntimeCapacity(runtime);
        EXPECT_EQ(runtime.mtp.enabled, current.rankPlan().runtime.mtp.enabled);
        EXPECT_EQ(runtime.mtp.draft_tokens, current.rankPlan().runtime.mtp.draft_tokens);
        EXPECT_EQ(runtime.mtp.depth_policy.mode, current.rankPlan().runtime.mtp.depth_policy.mode);
        EXPECT_EQ(runtime.mtp.graph_capacity_draft_tokens, 15);
        EXPECT_EQ(runtime.resident_graph_rows, prepared->admission()->rankPlan().runtime.resident_graph_rows);
        const auto bindings = prepared->createRuntimeBindings(model->concreteLoader(), memory, MPIContextFactory::self(), current);
        ASSERT_EQ(bindings.size(), 2u);
        EXPECT_EQ(bindings[1]->owners().residency->endpoint(0)->numLayers(), 2);
        EXPECT_EQ(memory->admission(), admitted.pipelineMemory()->physicalAdmission());
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

// Every test of this suite requires a compiled GPU backend family.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelinePreparedPlanTest, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });
#endif

/** @test Correction scratch cannot borrow an authority admitted for single-bank activations. */
TEST(PreparedWeightAuthorityIdentity, Qwen4ExpHCUpWorkspaceSelectionSeparatesRetainedAuthority)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision previous;
        ~Restore() { setup.qfn_hc_up_activation = previous; }
    } restore{setup, setup.qfn_hc_up_activation};
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    OrchestrationConfig config;
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
    const auto ordinary = routedWeightAuthorityRequestIdentity(plan, config);
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    const auto corrected = routedWeightAuthorityRequestIdentity(plan, config);
    EXPECT_NE(ordinary, corrected);
    EXPECT_EQ(corrected, routedWeightAuthorityRequestIdentity(plan, config));
    setup.qfn_hc_up_activation = QuantizedActivationOperandPrecision::SingleQ8;
    EXPECT_EQ(ordinary, routedWeightAuthorityRequestIdentity(plan, config));
}

/** @test A one-field generation-boundary change cannot reuse the original physical admission identity. */
TEST(PreparedWeightAuthorityIdentity, PipelineGenerationDomainSeparatesRetainedAuthority)
{
    PlanningGGUFFixture file(true, true);
    PlanningModelSource source(file.path());
    auto input = proposal(source, 1, false);
    input.config.mtp_activation_policy = MTPActivationPolicy::Disabled;
    input.config.mtp.graph_capacity_draft_tokens = 15;
    ExecutionPlanBuilder builder;
    auto topology = ResolvedRankOrchestration::resolve(input.config, source.metadata(),
        input.membership.inventory(), builder, 0);
    const auto admitted = AdmittedOrchestrationCandidate::admit(input, source, policy());
    ASSERT_TRUE(admitted.pipelineMemory());
    const auto original = pipelineRoutedWeightAuthorityRequestIdentity(topology);
    EXPECT_EQ(original, pipelineRoutedWeightAuthorityRequestIdentity(*admitted.pipelineMemory()));
    for (std::size_t index = 0; index < topology.pipelineStages().size(); ++index)
    {
        auto changed_topology = topology;
        auto changed_admission = *admitted.pipelineMemory();
        // Deliberate mutation of non-const copied test values isolates one
        // capacity field; production exposes no mutable scope accessor.
        auto &scope = const_cast<FactoryPPStageConfig &>(changed_topology.pipelineStages().at(index).scope());
        auto &admitted_scope = const_cast<FactoryPPStageConfig &>(changed_admission.stages().at(index).topology().scope());
        ASSERT_EQ(scope.generation_domain, DeviceGenerationExecutionDomain::HostedBoundary);
        scope.generation_domain = DeviceGenerationExecutionDomain::NativeDomain;
        admitted_scope.generation_domain = scope.generation_domain;
        const auto changed = pipelineRoutedWeightAuthorityRequestIdentity(changed_topology);
        EXPECT_NE(changed, original);
        EXPECT_EQ(changed, pipelineRoutedWeightAuthorityRequestIdentity(changed_admission));
    }
    EXPECT_EQ(original, pipelineRoutedWeightAuthorityRequestIdentity(topology));
}

/** @test Dense and MoE model reuse share the exact retained MTP weight-ownership gate. */
TEST(PreparedWeightAuthorityIdentity, MTPReuseRequiresPredictorAndHeadPlacement)
{
    for (const auto sidecar : {MTPSidecarDensePolicy::TensorParallel, MTPSidecarDensePolicy::ReplicatedPerParticipant})
        for (const auto head : {MTPTerminalHeadPolicy::VocabularySharded, MTPTerminalHeadPolicy::MirroredFullVocabulary})
        {
            MTPRuntimeConfig retained;
            retained.enabled = true;
            retained.graph_capacity_draft_tokens = 15;
            retained.sidecar_dense_policy = sidecar;
            retained.terminal_head_policy = head;
            for (const bool enabled : {false, true})
                for (const int depth : {1, 3, 7, 15})
                    for (const auto mode : {MTPDepthPolicyMode::Fixed, MTPDepthPolicyMode::Dynamic})
                    {
                        auto current = retained;
                        current.enabled = enabled;
                        current.draft_tokens = depth;
                        current.depth_policy.mode = mode;
                        ASSERT_TRUE(retainedMTPWeightAuthorityMatches(retained, current));
                        for (int changed = 0; changed < 4; ++changed)
                        {
                            SCOPED_TRACE(::testing::Message() << int(sidecar) << '/' << int(head)
                                << '/' << enabled << '/' << depth << '/' << int(mode) << '/' << changed);
                            auto invalid = current;
                            if (changed == 0)
                            {
                                invalid.graph_capacity_draft_tokens = 7;
                                invalid.draft_tokens = 1; // Active depth cannot retain the original larger envelope.
                                invalid.depth_policy.max_depth = 7;
                            }
                            if (changed == 1) invalid.max_request_batch += 1;
                            if (changed == 2) invalid.sidecar_dense_policy = sidecar == MTPSidecarDensePolicy::TensorParallel
                                ? MTPSidecarDensePolicy::ReplicatedPerParticipant : MTPSidecarDensePolicy::TensorParallel;
                            if (changed == 3) invalid.terminal_head_policy = head == MTPTerminalHeadPolicy::VocabularySharded
                                ? MTPTerminalHeadPolicy::MirroredFullVocabulary : MTPTerminalHeadPolicy::VocabularySharded;
                            EXPECT_FALSE(retainedMTPWeightAuthorityMatches(retained, invalid));
                        }
                    }
        }
}

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Ordinary model-wide setup shares the same constructor on every backend. */
TEST(MoEOverlayResidencySetup, ModelWideModesRetireDemandClaimsAndFailedConstruction)
{
    // Qwen's packed query/gate projection determines the actual query width.
    // Admit enough source heads for eight positive attention slices.
    PlanningGGUFFixture file(true, true, GGUFTensorType::F32, 256, std::nullopt, 32);
    PlanningModelSource source(file.path());
    ASSERT_GE(source.metadata().memoryProfile().n_heads, 8);
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const int width : {1, 2, 4, 8})
        {
            if (backend == DeviceType::CPU && width != 1) continue;
            for (const bool retain_mtp : {false, true})
                for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe, MoERebalanceRuntimeMode::Dynamic})
                {
                    SCOPED_TRACE(::testing::Message() << int(backend) << '/' << width << '/' << retain_mtp << '/' << int(mode));
                    const auto exercise = [&] {
                    const auto spelling = backend == DeviceType::CPU ? "cpu" : backend == DeviceType::CUDA ? "cuda" : "rocm";
                    std::string declaration = "native=";
                    for (int index = 0; index < width; ++index)
                        declaration += std::string(index ? "," : "") + spelling + ":" + std::to_string(index);
                    declaration += ";priority=0";
                    std::vector<std::string> args{"llaminar2", "--expert-tier", declaration};
                    std::vector<char *> argv;
                    for (auto &arg : args) argv.push_back(arg.data());
                    auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
                    config.model_path = source.path();
                    config.max_seq_len = 128;
                    config.moe_routed_prefill.overlay_segment_rows = 32;
                    config.moe_rebalance.mode = mode;
                    config.moe_rebalance.window_size = 8;
                    config.moe_rebalance.max_window_size = 32;
                    config.mtp_activation_policy = retain_mtp ? MTPActivationPolicy::Automatic : MTPActivationPolicy::Disabled;
                    auto observed = inventory(width);
                    observed.ranks.front().cpu.numa_node = 0;
                    observed.ranks.front().numa_nodes = 1;
                    observed.buildNodeAggregations();
                    const auto admitted = AdmittedOrchestrationCandidate::admit(
                        {OrchestrationStrategy::ExpertOverlay, ExecutionRankMembership(observed, {0}), config}, source, policy());
                    ASSERT_TRUE(admitted.overlayCapacity());
                    auto capacity = *admitted.overlayCapacity();
                    const auto metadata = metadataFor(source.loader(), 0, 2, retain_mtp ? 3 : 2);
                    const auto bound = MoEOverlayCapacityResolver::installResolvedQuotas(*admitted.config().moe_routed_expert_plan, capacity);
                    const auto plan = MoERoutedExpertPlacementPlanner::plan(bound, metadata).planned_plan;
                    const auto storage = resolveMoEOverlayCapacityAdmissionPolicy(plan, admitted.config(), 1, metadata.num_layers, 8);
                    const bool host_dynamic = backend == DeviceType::CPU && mode == MoERebalanceRuntimeMode::Dynamic;
                    // Host Dynamic alone does not authorize demand storage:
                    // a singleton has no peer to move an expert to. Preserve
                    // the production NoPeer policy instead of inventing a bank.
                    const bool needs_host_demand = host_dynamic &&
                        storage.migration_storage != MoEOverlayMigrationStorageKind::NoPeer;
                    if (host_dynamic)
                    {
                        // Discovery returns a topology-wide certificate. Live
                        // setup retains this rank's demand descriptor from the
                        // same fixed-owner builder, as the production runner does.
                        ExecutionPlanBuilder builder;
                        const auto compiled = ResolvedRankOrchestration::resolve(
                            admitted.config(), source.metadata(), observed, builder, 0);
                        const auto &rank = admitted.rankPlans().front();
                        const auto &mtp = rank.runtime.mtp;
                        const int decode_rows = retainsMTPGraphCapacity(mtp) ? resolveMTPRetainedTargetQueryRows(mtp) : 1;
                        const auto family = resolveMoEOverlayInferenceGraphFamilyIdentity(
                            source.loader(), source.loader().architecture(), source.loader().blockCount(),
                            retain_mtp ? MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                            1, std::max(decode_rows, 32), decode_rows, 1, resolveMTPRetainedDraftCapacity(mtp));
                        const auto settings = policy();
                        const auto inputs = buildMoEOverlayMemoryPlanInputs({
                            .model = source.metadata().memoryProfile(), .rank_plan = rank, .config = admitted.config(),
                            .inventory = observed, .execution = *compiled.overlayExecution(), .capacity_policy = storage,
                            .retained_mtp = mtp, .graph_family = family,
                            .prefill = {.bucket_rows = settings.prefill_bucket_rows,
                                .minimum_sequence_rows = settings.minimum_prefill_sequence_rows,
                                .maximum_cached_buckets = settings.maximum_cached_prefill_buckets}},
                            rank.runtime.resident_graph_rows);
                        capacity.host_demand_memory = inputs.local_capacity.host_demand_memory;
                    }
                    EXPECT_EQ(capacity.host_demand_memory.has_value(), needs_host_demand);
                    const auto memory = std::make_shared<PhysicalMemoryAuthority>(capacity.physical_memory_admission, 0);
                    const auto before = memory->rankAttestationSummary();
                    {
                        const auto setup = MoEOverlayResidencySetup::create({
                            .plan = plan, .metadata = metadata, .loader = source.loader(), .rebalance = config.moe_rebalance,
                            .storage_policy = storage, .capacity = capacity, .memory = memory});
                        ASSERT_TRUE(setup.authority());
                        EXPECT_EQ(setup.residency()->localParticipantIds().size(), size_t(width));
                        if (setup.histogram())
                        {
                            EXPECT_EQ(setup.histogram()->firstModelLayer(), 0);
                            EXPECT_EQ(setup.histogram()->endModelLayer(), metadata.num_layers);
                            EXPECT_EQ(setup.histogram()->config().token_boundary_layer_idx, 1);
                        }
                        EXPECT_EQ(memory->rankAttestationSummary() != before, needs_host_demand);
                    }
                    EXPECT_EQ(memory->rankAttestationSummary(), before);
                    if (host_dynamic)
                    {
                        auto no_demand = capacity;
                        no_demand.host_demand_memory.reset();
                        if (needs_host_demand)
                        {
                            EXPECT_THROW((void)MoEOverlayResidencySetup::create({
                                .plan = plan, .metadata = metadata, .loader = source.loader(), .rebalance = config.moe_rebalance,
                                .storage_policy = storage, .capacity = no_demand, .memory = memory}), std::logic_error);
                        }
                        else
                        {
                            ASSERT_EQ(storage.migration_storage, MoEOverlayMigrationStorageKind::NoPeer);
                            EXPECT_NO_THROW((void)MoEOverlayResidencySetup::create({
                                .plan = plan, .metadata = metadata, .loader = source.loader(), .rebalance = config.moe_rebalance,
                                .storage_policy = storage, .capacity = no_demand, .memory = memory}));
                            EXPECT_EQ(memory->rankAttestationSummary(), before);
                        }
                        // Invalid maintenance still fails with exactly the same
                        // ledger state, including a no-bank singleton setup.
                        auto invalid = config.moe_rebalance;
                        invalid.window_growth_factor = 0;
                        EXPECT_THROW((void)MoEOverlayResidencySetup::create({
                            .plan = plan, .metadata = metadata, .loader = source.loader(), .rebalance = invalid,
                            .storage_policy = storage, .capacity = capacity, .memory = memory}), std::invalid_argument);
                        EXPECT_EQ(memory->rankAttestationSummary(), before);
                    }
                    };
                    EXPECT_NO_THROW(exercise());
                }
        }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)
}

namespace llaminar2::test
{
/** @test A larger GDN residual bank cannot reuse an authority admitted only for HC or ordinary activations. */
TEST(PreparedWeightAuthorityIdentity, Qwen4ExpGDNOutputWorkspaceSelectionSeparatesRetainedAuthority)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision hc, gdn;
        ~Restore() { setup.qfn_hc_up_activation = hc; setup.qfn_gdn_output_activation = gdn; }
    } restore{setup, setup.qfn_hc_up_activation, setup.qfn_gdn_output_activation};
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    OrchestrationConfig config;
    std::set<std::string> identities;
    for (const auto hc : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
    for (const auto gdn : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        setup.qfn_hc_up_activation = hc;
        setup.qfn_gdn_output_activation = gdn;
        const auto identity = routedWeightAuthorityRequestIdentity(plan, config);
        EXPECT_TRUE(identities.insert(identity).second);
        EXPECT_EQ(identity, routedWeightAuthorityRequestIdentity(plan, config));
    }
    EXPECT_EQ(identities.size(), 4u);
}

} // namespace llaminar2::test

namespace llaminar2::test {
/** @test Retained PMA/prepared authority cannot hide an explicit distribution change behind the same VRAM tier. */
TEST(PreparedWeightAuthorityIdentity, FixedHeadNgramDistributionSeparatesRetainedRequestAuthority)
{
    auto plan=std::make_shared<MoERoutedExpertPlacementPlan>();plan->enabled=true;
    OrchestrationConfig c;c.ngram_table_policy={ImmutableRowStoragePreference::VRAM};
    const auto replicated=routedWeightAuthorityRequestIdentity(plan,c);
    c.ngram_table_policy.distribution=ImmutableRowTableDistribution::FixedHeadSharded;
    const auto sharded=routedWeightAuthorityRequestIdentity(plan,c);
    EXPECT_NE(replicated,sharded);EXPECT_EQ(sharded,routedWeightAuthorityRequestIdentity(plan,c));
    c.ngram_table_policy.distribution=ImmutableRowTableDistribution::Replicated;
    EXPECT_EQ(replicated,routedWeightAuthorityRequestIdentity(plan,c));
}
}

namespace llaminar2::test {
/** @test Retained capacity identity freezes all independent projection policies in both directions. */
TEST(PreparedWeightAuthorityIdentity, Qwen4ExpFAOutputWorkspaceSelectionSeparatesRetainedAuthority)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision hc, gdn, fa;
        ~Restore() { setup.qfn_hc_up_activation = hc; setup.qfn_gdn_output_activation = gdn; setup.qfn_fa_output_activation = fa; }
    } restore{setup, setup.qfn_hc_up_activation, setup.qfn_gdn_output_activation, setup.qfn_fa_output_activation};
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    OrchestrationConfig config;
    std::set<std::string> identities;
    for (const auto hc : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
    for (const auto gdn : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
    for (const auto fa : {QuantizedActivationOperandPrecision::SingleQ8, QuantizedActivationOperandPrecision::TwoTermQ8})
    {
        setup.qfn_hc_up_activation = hc;
        setup.qfn_gdn_output_activation = gdn;
        setup.qfn_fa_output_activation = fa;
        const auto identity = routedWeightAuthorityRequestIdentity(plan, config);
        EXPECT_TRUE(identities.insert(identity).second);
        EXPECT_EQ(identity, routedWeightAuthorityRequestIdentity(plan, config));
    }
    EXPECT_EQ(identities.size(), 8u);
}

}

namespace llaminar2::test
{
/** @test QKV activation scratch separates existing retained capacity authority, without inventing a routed owner. */
TEST(PreparedWeightAuthorityIdentity, Qwen4ExpGDNQKVSelectionSeparatesRetainedAuthority)
{
    auto &setup = mutableDebugEnv().gemm;
    struct Restore
    {
        GemmConfig &setup;
        QuantizedActivationOperandPrecision previous;
        ~Restore() { setup.qfn_gdn_qkv_activation = previous; }
    } restore{setup, setup.qfn_gdn_qkv_activation};
    auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
    plan->enabled = true;
    OrchestrationConfig config;
    setup.qfn_gdn_qkv_activation = QuantizedActivationOperandPrecision::SingleQ8;
    const auto ordinary = routedWeightAuthorityRequestIdentity(plan, config);
    setup.qfn_gdn_qkv_activation = QuantizedActivationOperandPrecision::TwoTermQ8;
    const auto corrected = routedWeightAuthorityRequestIdentity(plan, config);
    EXPECT_NE(ordinary, corrected);
    EXPECT_EQ(corrected, routedWeightAuthorityRequestIdentity(plan, config));
    EXPECT_TRUE(routedWeightAuthorityRequestIdentity(nullptr, config).empty());
    setup.qfn_gdn_qkv_activation = QuantizedActivationOperandPrecision::SingleQ8;
    EXPECT_EQ(ordinary, routedWeightAuthorityRequestIdentity(plan, config));
}
} // namespace llaminar2::test

namespace llaminar2::test {
namespace {
/** @brief Only native constructor results are doubled; actual source/pool/PMA planning remains unchanged. */
class PipelineSourceWorker final:public testing::MockWorkerGPUContext {
public:
    /** @param ordinal Original selected device. @param root Exact source-owned context object extent. */
    PipelineSourceWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack> root):MockWorkerGPUContext(ordinal){(void)root;}
};
/** @brief Retire actual recorded source generations through the existing original reset observer. */
struct PipelineSourceRetirement final {
    std::shared_ptr<NativeContextPreparation> source;
    std::vector<DeviceId> devices;
    /** @brief No direct native SDK reset or alternate capacity owner is created. */
    ~PipelineSourceRetirement(){
        for(const auto device:devices){auto reset=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
            NativeContextPreparationReplayAccess::physicalReset(reset);}
        if(source->memory()->phase()==PhysicalMemoryAuthorityPhase::Preparing)source->retireUnadmitted();
    }
};
/** @return The original public pipeline producer with actual same-vendor consecutive endpoints. */
OrchestrationConfig originalNativePipeline(const PlanningModelSource& source,DeviceType backend,bool cpuTail=false){
    auto config=proposal(source,1,backend==DeviceType::ROCm).config;
    config.domain_definitions.back().devices={GlobalDeviceAddress::fromLocalDeviceId(DeviceId(backend,1),"localhost",0)};
    config.domain_definitions.back().backend=backend==DeviceType::CUDA?CollectiveBackendType::NCCL:CollectiveBackendType::RCCL;
    if(cpuTail){config.domain_definitions.back().devices={GlobalDeviceAddress::cpu(0)};config.domain_definitions.back().backend=CollectiveBackendType::HOST;}
    config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
    config.mtp_activation_policy=MTPActivationPolicy::Disabled;
    config.mtp.graph_capacity_draft_tokens=0;
    return config;
}
/** @return Exact original physical observation registry; no free-memory query or capacity arithmetic. */
std::vector<PhysicalMemoryResource> sourceResources(const ClusterInventory& observed,std::span<const DeviceId> devices){
    const auto& rank=observed.ranks.front();
    std::vector<PhysicalMemoryResource> result{{0,DeviceId::cpu(),rank.cpu.memory_bytes,rank.cpu.free_memory_bytes}};
    for(const auto device:devices){const auto it=std::find_if(rank.gpus.begin(),rank.gpus.end(),[&](const auto& gpu){return gpu.type==device.type && gpu.local_device_id==device.ordinal;});
        if(it==rank.gpus.end())throw std::logic_error("Recorded pipeline omitted selected device");
        result.push_back({0,device,it->memory_bytes,it->free_memory_bytes});}
    return result;
}
/** @brief Bind recorded constructor bodies before the actual source attempts them. */
void seedPipelineWorkers(std::span<const DeviceId> devices,bool failLast=false){
    for(std::size_t i=0;i<devices.size();++i){const auto device=devices[i];const bool fail=failLast && i+1==devices.size();
        NativeContextPreparationReplayAccess::seed(device,[device,fail](std::shared_ptr<PreparedGPUWorkerStack> root){
            auto worker=NativeContextPreparationReplayAccess::construct<PipelineSourceWorker>(std::move(root),device.ordinal);
            if(fail)throw std::runtime_error("original pipeline native prefix construction failed");return worker;});}
}
/** @brief Existing compiler injection preserves the entire production topology and binds only an inherited runtime owner. */
class SourceBoundPlanBuilder final:public ExecutionPlanBuilder {
public:
    /** @param source Original closed source, never a fake Ready capability. */
    explicit SourceBoundPlanBuilder(std::shared_ptr<NativeContextPreparation> source):source_(std::move(source)){}
    RankExecutionPlan buildPlanForRank(const OrchestrationConfig& cfg,const ModelConfig& model,const ClusterInventory& inv,int rank)override{
        auto plan=ExecutionPlanBuilder::buildPlanForRank(cfg,model,inv,rank);plan.runtime.native_context_preparation=source_;return plan;
    }
private:std::shared_ptr<NativeContextPreparation> source_;
};
/** @return One exact existing memory policy with a separately passed original serving source. */
MoEPipelineMemoryPolicy sourcePolicy(std::span<const DeviceId> devices,const ModelLoader& loader,std::shared_ptr<NativeContextPreparation> source={}){
    static constexpr std::array rows{1,8};
    return {.prefill={.bucket_rows=rows,.minimum_sequence_rows=1,.maximum_cached_buckets=4},
        .gpu_weight_load={.policy=configuredGPUWeightLoadMemoryPolicy(),.maximum_source_bytes=maximumGGUFTensorPayloadBytes(loader.getModel())},
        .native_ordering_events=NativeOrderingEventGroupReplayAccess::projection(devices),
        .retained_host_services=planningHostServiceProjection(),.native_context_preparation=std::move(source)};
}
/**
 * @brief Reuse all original source-admission contracts for the actual local stage GPU set.
 * @param topology Original compiled rank projection; remote devices never enter its event quote.
 * @param loader Original sparse GGUF directory supplying the real maximum tensor extent.
 * @param rows Explicit retained prefill inventory owned by the calling fixture.
 * @return Existing complete GPU-load/event/host-service policy with only the requested prefill envelope changed.
 */
MoEPipelineMemoryPolicy globalSourcePolicy(const ResolvedRankOrchestration &topology,
    const ModelLoader &loader, std::span<const int> rows)
{
    std::set<DeviceId> unique;
    for (const auto &stage : topology.pipelineStages())
        for (const auto device : stage.overlayExecution().currentRankPlan().local_devices)
            if (device.is_gpu()) unique.insert(device);
    const std::vector<DeviceId> devices(unique.begin(), unique.end());
    auto result = sourcePolicy(devices, loader);
    result.prefill = {.bucket_rows = rows, .minimum_sequence_rows = 1, .maximum_cached_buckets = 16};
    return result;
}
}
/** @test Actual original source backing survives immutable topology compilation and is charged once across every stage. */
TEST(MoEPipelineNativePreparation, OriginalSourceReachesParentAllStagesAndFinalPMAOnce){
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const bool cpuTail:{false,true})ASSERT_EXIT(([&]{
        {PlanningGGUFFixture file(true);PlanningModelSource model(file.path());auto context=metadataContext(file.path());
         auto observed=inventory(2);observed.ranks.front().cpu.numa_node=0;observed.ranks.front().numa_nodes=1;observed.buildNodeAggregations();
         const auto config=originalNativePipeline(model,backend,cpuTail);
         ExecutionPlanBuilder compiler;const auto topology=ResolvedRankOrchestration::resolve(config,model.metadata(),observed,compiler,0);
         ASSERT_FALSE(topology.rankPlan().runtime.native_context_preparation);
         std::vector<DeviceId> devices{DeviceId(backend,0)};if(!cpuTail)devices.push_back(DeviceId(backend,1));seedPipelineWorkers(devices);
         auto source=NativeContextPreparationReplayAccess::source<PipelineSourceWorker>(devices,sourceResources(observed,devices));
         PipelineSourceRetirement retirement{source,{devices.begin(),devices.end()}};
         for(const auto device:devices){auto permit=source->acquire(device);}
         source->freezeForPlanning();
         const auto admission=AdmittedMoEPipelineMemory::admit(topology,model.metadata(),context->concreteLoader(),observed,sourcePolicy(devices,context->concreteLoader(),source));
         EXPECT_EQ(admission.rankPlan().runtime.native_context_preparation,source);
         ASSERT_EQ(admission.stages().size(),2u);
         for(const auto& stage:admission.stages()){
             EXPECT_EQ(stage.rankPlan().runtime.native_context_preparation,source);
             for(const auto& cfg:stage.devicePlans())EXPECT_EQ(cfg.native_context_preparation,source);
             EXPECT_FALSE(stage.topology().rankPlan().runtime.native_context_preparation); // Arithmetic topology stays immutable.
         }
         for(const auto& original:source->mandatoryBOM().resources()){
             const auto* actual=admission.physicalAdmission()->plan().find({original.resource().world_rank,original.resource().device});
             ASSERT_NE(actual,nullptr);
             EXPECT_EQ(actual->charge(PhysicalMemoryOwner::NativeExecutionContext).incrementalBytes(),
                 original.charge(PhysicalMemoryOwner::NativeExecutionContext).incrementalBytes());
         }
         auto runtime=topology.rankPlan().runtime;admission.publishRuntimeCapacity(runtime);
         EXPECT_EQ(runtime.native_context_preparation,source);
         auto foreign=std::shared_ptr<NativeContextPreparation>(source.get(),[](auto*){});
         auto wrong=runtime;wrong.native_context_preparation=foreign;const auto rows=wrong.resident_graph_rows;
         EXPECT_THROW(admission.publishRuntimeCapacity(wrong),std::logic_error);EXPECT_EQ(wrong.resident_graph_rows,rows);
         SourceBoundPlanBuilder inherited(source);const auto originalBound=ResolvedRankOrchestration::resolve(config,model.metadata(),observed,inherited,0);
         EXPECT_THROW((void)AdmittedMoEPipelineMemory::admit(originalBound,model.metadata(),context->concreteLoader(),observed,sourcePolicy(devices,context->concreteLoader())),std::logic_error);
         EXPECT_THROW((void)AdmittedMoEPipelineMemory::admit(originalBound,model.metadata(),context->concreteLoader(),observed,sourcePolicy(devices,context->concreteLoader(),foreign)),std::logic_error);
         EXPECT_NO_THROW(source->admitFinal(admission.physicalAdmission()));
         EXPECT_EQ(source->memory()->phase(),PhysicalMemoryAuthorityPhase::Admitted);
        }
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?1:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test Source-free discovery remains legitimate but cannot publish over an existing serving owner. */
TEST(MoEPipelineNativePreparation, DiscoveryStaysSourceFreeAndCannotEraseServingSource){
    GTEST_FLAG_SET(death_test_style,"threadsafe");ASSERT_EXIT(([&]{
        {PlanningGGUFFixture file(true);PlanningModelSource model(file.path());auto context=metadataContext(file.path());const auto observed=inventory(2);
         const auto config=originalNativePipeline(model,DeviceType::CUDA);ExecutionPlanBuilder compiler;
         const auto topology=ResolvedRankOrchestration::resolve(config,model.metadata(),observed,compiler,0);
         const std::array devices{DeviceId::cuda(0),DeviceId::cuda(1)};seedPipelineWorkers(devices);
         auto source=NativeContextPreparationReplayAccess::source<PipelineSourceWorker>(devices,sourceResources(observed,devices));
         PipelineSourceRetirement retirement{source,{devices.begin(),devices.end()}};
         for(const auto device:devices){auto permit=source->acquire(device);}source->freezeForPlanning();
         const auto discovery=AdmittedMoEPipelineMemory::admit(topology,model.metadata(),context->concreteLoader(),observed,sourcePolicy(devices,context->concreteLoader()));
         EXPECT_FALSE(discovery.rankPlan().runtime.native_context_preparation);
         RuntimeConfig bound;bound.native_context_preparation=source;
         EXPECT_THROW(discovery.publishRuntimeCapacity(bound),std::logic_error);EXPECT_EQ(bound.native_context_preparation,source);
         try{source->admitFinal(discovery.physicalAdmission());FAIL()<<"Source-free certificate admitted retained native backing";}
         catch(const std::logic_error& error){EXPECT_EQ(std::string(error.what()),"Final source plan omitted/relabelled preparation backing as residency");}
         EXPECT_EQ(source->memory()->phase(),PhysicalMemoryAuthorityPhase::Preparing);
        }
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?1:0);
    }()),::testing::ExitedWithCode(0),"");
}
/** @test Source-backed Apply adds only original birth graphs to the identical retained pipeline envelope. */
TEST(MoEPipelineNativePreparation, RetainedGraphInventoryMatchesSourceFreeAndOriginalApply)
{
    GTEST_FLAG_SET(death_test_style,"threadsafe");
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (const int capacity : {0, 15}) ASSERT_EXIT(([&] {
            {
                PlanningGGUFFixture file(true, capacity > 0);
                PlanningModelSource model(file.path());
                auto context = metadataContext(file.path());
                auto observed = inventory(2);
                observed.ranks.front().cpu.numa_node = 0;
                observed.ranks.front().numa_nodes = 1;
                observed.buildNodeAggregations();
                auto config = originalNativePipeline(model, backend);
                config.mtp.graph_capacity_draft_tokens = capacity;
                ExecutionPlanBuilder compiler;
                const auto topology = ResolvedRankOrchestration::resolve(config, model.metadata(), observed, compiler, 0);
                const std::array devices{DeviceId(backend, 0), DeviceId(backend, 1)};
                const auto discovery = AdmittedMoEPipelineMemory::admit(topology, model.metadata(),
                    context->concreteLoader(), observed, sourcePolicy(devices, context->concreteLoader()));
                seedPipelineWorkers(devices);
                auto source = NativeContextPreparationReplayAccess::source<PipelineSourceWorker>(
                    devices, sourceResources(observed, devices));
                PipelineSourceRetirement retirement{source, {devices.begin(), devices.end()}};
                for (const auto device : devices) { auto permit = source->acquire(device); }
                source->freezeForPlanning();
                const auto applied = AdmittedMoEPipelineMemory::admit(topology, model.metadata(),
                    context->concreteLoader(), observed, sourcePolicy(devices, context->concreteLoader(), source));
                for (const auto device : devices)
                {
                    const auto *before = discovery.physicalAdmission()->plan().find({0, device});
                    const auto *after = applied.physicalAdmission()->plan().find({0, device});
                    const auto *birth = source->mandatoryBOM().find({0, device});
                    ASSERT_NE(before, nullptr);
                    ASSERT_NE(after, nullptr);
                    ASSERT_NE(birth, nullptr);
                    EXPECT_EQ(after->bytes(PhysicalMemoryOwner::NativeGraphExecutable),
                        before->bytes(PhysicalMemoryOwner::NativeGraphExecutable) +
                        birth->bytes(PhysicalMemoryOwner::NativeGraphExecutable));
                }
            }
            GPUDeviceProcessLifetime::retireActive();
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        }()), ::testing::ExitedWithCode(0), "");
}
/** @test A failed original constructor prefix cannot be frozen/admitted and retains both reset obligations. */
TEST(MoEPipelineNativePreparation, PartialOriginalPreparationRejectsBeforePlanningAndRetiresEveryAttempt){
    GTEST_FLAG_SET(death_test_style,"threadsafe");ASSERT_EXIT(([&]{
        {PlanningGGUFFixture file(true);PlanningModelSource model(file.path());auto context=metadataContext(file.path());const auto observed=inventory(2);
         const auto config=originalNativePipeline(model,DeviceType::CUDA);ExecutionPlanBuilder compiler;
         const auto topology=ResolvedRankOrchestration::resolve(config,model.metadata(),observed,compiler,0);
         const std::array devices{DeviceId::cuda(0),DeviceId::cuda(1)};seedPipelineWorkers(devices,true);
         auto source=NativeContextPreparationReplayAccess::source<PipelineSourceWorker>(devices,sourceResources(observed,devices));
         PipelineSourceRetirement retirement{source,{devices.begin(),devices.end()}};
         {auto first=source->acquire(devices[0]);}
         EXPECT_THROW(source->acquire(devices[1]),std::runtime_error);
         EXPECT_THROW(source->freezeForPlanning(),std::logic_error);
         EXPECT_THROW((void)AdmittedMoEPipelineMemory::admit(topology,model.metadata(),context->concreteLoader(),observed,sourcePolicy(devices,context->concreteLoader(),source)),std::logic_error);
         EXPECT_EQ(source->memory()->phase(),PhysicalMemoryAuthorityPhase::Preparing);
        }
        GPUDeviceProcessLifetime::retireActive();std::_Exit(::testing::Test::HasFailure()?1:0);
    }()),::testing::ExitedWithCode(0),"");
}
} // namespace llaminar2::test

namespace llaminar2::test {
// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/** @test Actual per-rank admission/preparation preserves source, layers and sole PMA on every owner. */
TEST(MoEGlobalPipelineMemory, LocalClaimsMTPEnvelopeAndPartialFailureRemainRankOwned)
{
    static constexpr std::array prefill_rows{1, 8, 32};
    PlanningGGUFFixture file(true, true, GGUFTensorType::Q4_0);
    PlanningModelSource source(file.path());
    auto observed = inventory(2);
    auto peer = observed.ranks.front();
    peer.rank = peer.local_rank = 1;
    peer.cpu.numa_node = 17;
    std::erase_if(observed.ranks.front().gpus, [](const auto &gpu) { return gpu.type != DeviceType::CUDA; });
    std::erase_if(peer.gpus, [](const auto &gpu) { return gpu.type != DeviceType::ROCm; });
    for (auto &gpu : peer.gpus) gpu.numa_node = peer.cpu.numa_node;
    observed.ranks.push_back(std::move(peer));
    observed.world_size = 2;
    observed.buildNodeAggregations();
    auto input = proposal(source, 2, false);
    input.config.mpi_procs = 2;
    input.config.domain_definitions.back().owner_rank = 1;
    for (const bool retained_mtp : {false, true})
    {
        input.config.mtp_activation_policy = retained_mtp ? MTPActivationPolicy::Automatic : MTPActivationPolicy::Disabled;
        input.config.mtp.graph_capacity_draft_tokens = retained_mtp ? 15 : 0;
        for (const auto mode : {MoERebalanceRuntimeMode::Off, MoERebalanceRuntimeMode::Observe, MoERebalanceRuntimeMode::Dynamic})
        {
            input.config.moe_rebalance.mode = mode;
            for (int rank = 0; rank < 2; ++rank)
            {
                SCOPED_TRACE(::testing::Message() << rank << '/' << retained_mtp << '/' << int(mode));
                ExecutionPlanBuilder builder;
                const auto topology = ResolvedRankOrchestration::resolve(input.config, source.metadata(), observed, builder, rank);
                const auto admitted = std::make_shared<const AdmittedMoEPipelineMemory>(AdmittedMoEPipelineMemory::admit(
                    topology, source.metadata(), source.loader(), observed,
                    globalSourcePolicy(topology, source.loader(), prefill_rows)));
                ASSERT_TRUE(admitted->globalPipeline());
                ASSERT_EQ(admitted->stages().size(), 1u);
                const auto &stage = admitted->stages().front();
                EXPECT_EQ(stage.topology().stageId(), rank);
                EXPECT_EQ(stage.rankPlan().rank, rank);
                EXPECT_EQ(stage.weightManifest().front().layer_idx, rank);
                EXPECT_EQ(stage.weightManifest().size(), rank == 1 && retained_mtp ? 2u : 1u);
                for (const auto &resource : admitted->physicalAdmission()->plan().resources())
                    EXPECT_EQ(resource.resource().world_rank, rank);
                auto model = metadataContext(file.path());
                const auto prepared = MoEPipelinePreparedPlan::create(admitted, model);
                ASSERT_EQ(prepared->stages().size(), 1u);
                EXPECT_EQ(prepared->stages().front().metadata.first_model_layer, rank);
                auto memory = std::make_shared<PhysicalMemoryAuthority>(admitted->physicalAdmission(), rank);
                const auto unclaimed = memory->rankAttestationSummary();
                const auto mpi = std::make_shared<MockMPIContext>(rank, 2);
                auto bindings = prepared->createRuntimeBindings(model->concreteLoader(), memory, mpi, topology);
                ASSERT_EQ(bindings.size(), 1u);
                EXPECT_EQ(bindings.front()->owners().mpi, mpi);
                ASSERT_EQ(bindings.front()->owners().residency->localParticipantIds().size(), 2u);
                EXPECT_EQ(bindings.front()->scope().first_layer, rank);
                std::weak_ptr<MoEOverlayResidencyAuthority> owner = bindings.front()->owners().authority;
                bindings.clear();
                EXPECT_TRUE(owner.expired());
                EXPECT_EQ(memory->rankAttestationSummary(), unclaimed);
                EXPECT_THROW((void)prepared->createRuntimeBindings(model->concreteLoader(), memory,
                    std::make_shared<MockMPIContext>(1-rank, 2), topology), std::invalid_argument);
                EXPECT_EQ(memory->rankAttestationSummary(), unclaimed);
                auto changed = input.config;
                // A remote-only edit must not be accepted as the same retained
                // global topology merely because this rank's local stage fits.
                auto &remote = changed.domain_definitions[1-rank].devices;
                std::reverse(remote.begin(), remote.end());
                const auto foreign = ResolvedRankOrchestration::resolve(changed, source.metadata(), observed, builder, rank);
                EXPECT_THROW((void)prepared->createRuntimeBindings(model->concreteLoader(), memory, mpi, foreign), std::invalid_argument);
                EXPECT_EQ(memory->rankAttestationSummary(), unclaimed);
                EXPECT_FALSE(model->concreteWeightManager()->preparedWeightStoreIfInitialized());
            }
        }
    }
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test One rank may prepare separated global stages without taking another rank's interval or CPU storage. */
TEST(MoEGlobalPipelineMemory, ReversedAndRepeatedRankStagesKeepExactAdmissionAndProjection)
{
    static constexpr std::array prefill_rows{1, 8, 32};
    PlanningGGUFFixture file(true, false, GGUFTensorType::Q4_0, 256, std::nullopt, 8,
        PlanningGGUFFixture::Geometry::Directory, 4);
    PlanningModelSource source(file.path());
    for (const auto &owners : {std::vector<int>{1,0}, std::vector<int>{2,0,2,1}})
    {
        const int count = *std::max_element(owners.begin(), owners.end()) + 1;
        auto observed = inventory(1);
        const auto seed = observed.ranks.front();
        observed.ranks.clear(); observed.world_size = count;
        for (int rank = 0; rank < count; ++rank)
        {
            auto entry = seed;
            entry.rank = entry.local_rank = rank;
            entry.cpu.numa_node = 9 + rank * 7;
            std::erase_if(entry.gpus, [rank](const auto &gpu) {
                return gpu.type != (rank % 2 ? DeviceType::ROCm : DeviceType::CUDA); });
            for (auto &gpu : entry.gpus)
            {
                gpu.local_device_id += 4 * rank;
                gpu.uuid += "-rank" + std::to_string(rank);
                gpu.numa_node = entry.cpu.numa_node;
            }
            observed.ranks.push_back(std::move(entry));
        }
        observed.buildNodeAggregations();
        std::vector<std::string> arguments{"original-pipeline", "--mpi-procs", std::to_string(count), "--no-mtp"};
        for (size_t stage = 0; stage < owners.size(); ++stage)
        {
            const int owner = owners[stage]; const bool cuda = owner % 2 == 0;
            arguments.insert(arguments.end(), {"--define-domain", "part" + std::to_string(stage) + "=" +
                (cuda ? "cuda:" : "rocm:") + std::to_string(4 * owner) + ";scope=single;backend=" +
                (cuda ? "nccl" : "rccl") + ";owner=" + std::to_string(owner), "--pp-stage",
                std::to_string(stage) + "=part" + std::to_string(stage) + ":" + std::to_string(stage) + "-" +
                std::to_string(stage + 1 == owners.size() ? 3 : stage)});
        }
        std::vector<char *> argv;for (auto &arg : arguments) argv.push_back(arg.data());
        auto config = OrchestrationConfigParser{}.parseArgs(argv.size(), argv.data());
        config.model_path = file.path(); config.max_seq_len = 128;
        config.moe_routed_prefill.overlay_segment_rows = 32;
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        for (int rank = 0; rank < count; ++rank)
        {
            ExecutionPlanBuilder builder;
            const auto topology = ResolvedRankOrchestration::resolve(config, source.metadata(), observed, builder, rank);
            const auto admitted = std::make_shared<const AdmittedMoEPipelineMemory>(AdmittedMoEPipelineMemory::admit(
                topology, source.metadata(), source.loader(), observed,
                globalSourcePolicy(topology, source.loader(), prefill_rows)));
            auto model = metadataContext(file.path());
            const auto prepared = MoEPipelinePreparedPlan::create(admitted, model);
            auto memory = std::make_shared<PhysicalMemoryAuthority>(admitted->physicalAdmission(), rank);
            const auto before = memory->rankAttestationSummary();
            auto bindings = prepared->createRuntimeBindings(model->concreteLoader(), memory,
                std::make_shared<MockMPIContext>(rank, count), topology);
            size_t local = 0;
            for (size_t stage = 0; stage < owners.size(); ++stage)
                if (owners[stage] == rank)
                {
                    const auto &claim = admitted->stages().at(local);
                    const auto &binding = bindings.at(local++);
                    EXPECT_EQ(claim.topology().stageId(), stage);
                    EXPECT_EQ(binding->scope().first_layer, stage);
                    EXPECT_EQ(binding->owners().mpi->rank(), rank);
                    const auto &domain = binding->plan()->domains.front();
                    InferenceRunnerConfig destination;
                    destination.pp_stage_config = binding->scope();
                    destination.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
                    destination.cpu_serial_expert_storage.push_back(nullptr);
                    EXPECT_THROW(binding->installRuntime(destination, domain.participants, domain.weights,
                        domain.backend), std::invalid_argument);
                    EXPECT_FALSE(destination.moe_routed_expert_plan);
                    EXPECT_FALSE(destination.moe_expert_overlay_residency_authority);
                    EXPECT_EQ(destination.cpu_serial_expert_storage.size(), 1u);
                    destination.cpu_serial_expert_storage.clear();
                    binding->installRuntime(destination, domain.participants, domain.weights, domain.backend);
                    EXPECT_EQ(destination.moe_expert_overlay_residency_authority, binding->owners().authority);
                    EXPECT_THROW(binding->installRuntime(destination, domain.participants, domain.weights,
                        domain.backend), std::invalid_argument);
                }
            EXPECT_EQ(local, bindings.size());
            bindings.clear();
            EXPECT_EQ(memory->rankAttestationSummary(), before);
            for (const auto &resource : admitted->physicalAdmission()->plan().resources())
                EXPECT_EQ(resource.resource().world_rank, rank);
        }
    }
}
} // namespace llaminar2::test

namespace llaminar2::test {
/** @test Original CPU global shards reach the canonical input adapter without becoming local TP. */
TEST(MoEGlobalPipelineMemory, OriginalCPUStageMemoryInputsRequireExactProgramScopeAndRank)
{
    PlanningGGUFFixture file(true,true,GGUFTensorType::Q4_0,256,std::nullopt,8,
        PlanningGGUFFixture::Geometry::FullAttentionGraph,2);
    PlanningModelSource source(file.path());
    auto observed=inventory(0);
    const auto seed=observed.ranks.front();observed.ranks.clear();observed.world_size=2;
    for(int rank=0;rank<2;++rank) {
        auto owner=seed;owner.rank=owner.local_rank=rank;owner.cpu.numa_node=7+rank*10;owner.numa_nodes=2;
        observed.ranks.push_back(std::move(owner));
    }
    observed.buildNodeAggregations();
    for(const auto compute:{RoutedExpertComputePolicy::TensorSharded,RoutedExpertComputePolicy::Apportioned}) {
        OrchestrationConfig config;config.model_path=file.path();config.mpi_procs=2;config.max_seq_len=64;
        config.mtp_activation_policy=MTPActivationPolicy::Disabled;config.mtp.graph_capacity_draft_tokens=0;
        config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
        config.moe_routed_prefill.overlay_segment_rows=8;
        DomainDefinition cpu;cpu.name="original_cpu";cpu.scope=TPScope::NODE_LOCAL;
        cpu.backend=CollectiveBackendType::UPI;cpu.routed_compute_policy=compute;
        for(const auto& owner:observed.ranks) {
            cpu.devices.push_back(GlobalDeviceAddress::cpu(owner.cpu.numa_node,owner.hostname));
            cpu.explicit_ranks.push_back(owner.rank);
        }
        DomainDefinition tail;tail.name="cpu_tail";tail.owner_rank=0;tail.scope=TPScope::AUTO;
        tail.devices={cpu.devices.front()};tail.backend=CollectiveBackendType::HOST;
        tail.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
        config.domain_definitions={cpu,tail};config.pp_stage_definitions={{0,cpu.name,0,0},{1,tail.name,1,1}};
        for(int rank=0;rank<2;++rank) {
            SCOPED_TRACE(::testing::Message()<<rank<<"/"<<int(compute));
            ExecutionPlanBuilder builder;
            const auto topology=ResolvedRankOrchestration::resolve(config,source.metadata(),observed,builder,rank);
            ASSERT_TRUE(topology.globalPipeline());
            const auto programs=originalCPUPipelinePrograms(topology.config(),*topology.globalPipeline(),observed);
            ASSERT_EQ(programs.size(),1u);
            const auto& child=topology.pipelineStages().front();
            const auto& scope=child.scope();const auto& mtp=child.rankPlan().runtime.mtp;
            const auto family=resolveMoEOverlayInferenceGraphFamilyIdentity(source.loader(),source.loader().architecture(),
                source.loader().blockCount(),MoEOverlayMTPGraphFamilyPolicy::MainOnly,1,8,1,1,0)
                    .forRoutedLayerInterval(scope.first_layer,scope.layerCount());
            const auto manifest=buildMoEOverlayLayerWeightManifestFromLoader(source.loader(),scope.layerCount(),8,scope.first_layer);
            const auto storage=resolveMoEOverlayCapacityAdmissionPolicy(*child.config().moe_routed_expert_plan,
                child.config(),2,scope.layerCount(),8);
            static constexpr std::array rows{1,8};
            const auto settings=globalSourcePolicy(topology,source.loader(),rows);
            MoEOverlayMemoryPlanInputRequest request{
                .model=source.metadata().memoryProfile(),.rank_plan=child.rankPlan(),.config=child.config(),
                .inventory=observed,.execution=child.overlayExecution(),.capacity_policy=storage,.retained_mtp=mtp,
                .graph_family=family,.prefill=settings.prefill,.gpu_weight_load=settings.gpu_weight_load,
                .pipeline_stage=scope,.original_expert_manifest=manifest,
                .native_source_placement=source.loader().nativeWeightSourcePlacement(),.original_cpu_stage_program=&programs.front()};
            const auto inputs=buildMoEOverlayMemoryPlanInputs(request,8);
            EXPECT_EQ(inputs.local_capacity.original_cpu_stage_program,&programs.front());
            EXPECT_EQ(inputs.local_capacity.rank_plan,&child.rankPlan());
            EXPECT_TRUE(inputs.local_capacity.rank_plan->usesGlobalTP());
            EXPECT_TRUE(inputs.local_capacity.rank_plan->local_tp_devices.empty());
            EXPECT_EQ(inputs.local_capacity.rank_inventory,&observed.ranks[rank]);
            EXPECT_EQ(inputs.local_capacity.generation_domain,DeviceGenerationExecutionDomain::HostedBoundary);
            EXPECT_EQ(inputs.local_capacity.graph_family.first_model_layer,0);
            EXPECT_EQ(inputs.local_capacity.graph_family.main_layer_count,1);
            EXPECT_EQ(inputs.local_capacity.resident_graph_rows,8);
            request.original_cpu_stage_program=nullptr;
            EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(request,8),std::invalid_argument);
            request.original_cpu_stage_program=&programs.front();
            auto foreign=child.rankPlan();foreign.primary_device.numa_node=999;
            // Rebind the borrowed immutable rank reference through aggregate
            // reconstruction; no sealed topology is mutated to create a valid case.
            MoEOverlayMemoryPlanInputRequest wrong_rank{
                .model=request.model,.rank_plan=foreign,.config=request.config,.inventory=observed,
                .execution=request.execution,.capacity_policy=storage,.retained_mtp=mtp,.graph_family=family,
                .prefill=settings.prefill,.gpu_weight_load=settings.gpu_weight_load,.pipeline_stage=scope,
                .original_expert_manifest=manifest,.native_source_placement=request.native_source_placement,
                .original_cpu_stage_program=&programs.front()};
            EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(wrong_rank,8),std::invalid_argument);
            foreign=child.rankPlan();foreign.weight_shard.shard_index=1-rank;
            EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(wrong_rank,8),std::invalid_argument);
            auto foreign_stage=topology.globalPipeline()->stages.front();
            foreign_stage.first_layer=foreign_stage.last_layer=1;foreign_stage.has_embedding=false;foreign_stage.has_lm_head=true;
            const auto foreign_program=OriginalCPUPipelineProgram::checked(foreign_stage,observed);
            request.original_cpu_stage_program=&foreign_program;
            EXPECT_THROW((void)buildMoEOverlayMemoryPlanInputs(request,8),std::invalid_argument);
        }
    }
}
/** @test Unequal original local stage sets feed one complete CPU PMA plan, preserving stage and allocator identity. */
TEST(MoEGlobalPipelineMemory, OriginalCPUStageCapacityExchangePreservesUnequalOriginalOwnership)
{
    PlanningGGUFFixture file(true,true,GGUFTensorType::Q4_0,256,std::nullopt,8,
        PlanningGGUFFixture::Geometry::FullAttentionGraph,2);
    PlanningModelSource source(file.path());
    auto observed=inventory(0);
    const auto seed=observed.ranks.front();
    observed.ranks.clear();
    observed.world_size=2;
    for(int rank=0;rank<2;++rank) {
        auto owner=seed;
        owner.rank=owner.local_rank=rank;
        owner.cpu.numa_node=7+rank*10;
        owner.numa_nodes=2;
        observed.ranks.push_back(std::move(owner));
    }
    observed.buildNodeAggregations();
    OrchestrationConfig config;
    config.model_path=file.path();config.mpi_procs=2;config.max_seq_len=64;
    config.mtp_activation_policy=MTPActivationPolicy::Disabled;config.mtp.graph_capacity_draft_tokens=0;
    config.moe_rebalance.mode=MoERebalanceRuntimeMode::Off;
    config.moe_routed_prefill.overlay_segment_rows=8;
    DomainDefinition head;
    head.name="head";head.scope=TPScope::NODE_LOCAL;head.backend=CollectiveBackendType::UPI;
    head.routed_compute_policy=RoutedExpertComputePolicy::TensorSharded;
    for(const auto& owner:observed.ranks) {
        head.devices.push_back(GlobalDeviceAddress::cpu(owner.cpu.numa_node,owner.hostname));
        head.explicit_ranks.push_back(owner.rank);
    }
    DomainDefinition tail;
    tail.name="tail";tail.owner_rank=0;tail.scope=TPScope::AUTO;tail.backend=CollectiveBackendType::HOST;
    tail.devices={head.devices.front()};tail.routed_compute_policy=RoutedExpertComputePolicy::Apportioned;
    config.domain_definitions={head,tail};config.pp_stage_definitions={{0,head.name,0,0},{1,tail.name,1,1}};
    ExecutionPlanBuilder first_builder,second_builder;
    const auto first=ResolvedRankOrchestration::resolve(config,source.metadata(),observed,first_builder,0);
    const auto second=ResolvedRankOrchestration::resolve(config,source.metadata(),observed,second_builder,1);
    ASSERT_TRUE(first.globalPipeline());
    const auto& topology=*first.globalPipeline();
    const auto budget=[](int rank,size_t planned,size_t resident) {
        PhysicalMemoryBOMBuilder builder(PhysicalMemoryResource{rank,DeviceId::cpu(),4096,2048});
        builder.add(PhysicalMemoryOwner::NativeCPUCollectiveDomain,planned,resident);
        return MoEOverlayBoundPhysicalMemoryBudget{
            MoEOverlayLocalCapacityPlanner::physicalResourceId(rank,DeviceId::cpu()),
            PhysicalMemoryAdmissionCertificate(builder.build())};
    };
    const std::vector first_head{budget(0,11,1)}, first_tail{budget(0,17,2)}, second_head{budget(1,23,3)};
    const std::array first_parts{OriginalPipelineFixedContribution{0,first_head},OriginalPipelineFixedContribution{1,first_tail}};
    const std::array second_parts{OriginalPipelineFixedContribution{0,second_head}};
    const auto first_frame=OriginalPipelineCapacitySendFrame::prepare(topology,0,first_parts);
    const auto second_frame=OriginalPipelineCapacitySendFrame::prepare(topology,1,second_parts);
    const auto first_read=parseOriginalPipelineCapacityFrame(first_frame.bytes(),0,topology);
    const auto second_read=parseOriginalPipelineCapacityFrame(second_frame.bytes(),1,topology);
    ASSERT_EQ(first_read.size(),2u);ASSERT_EQ(second_read.size(),2u);
    ASSERT_EQ(first_read[0].budgets.size(),1u);ASSERT_EQ(first_read[1].budgets.size(),1u);
    ASSERT_EQ(second_read[0].budgets.size(),1u);EXPECT_TRUE(second_read[1].budgets.empty());
    EXPECT_EQ(first_read[0].budgets[0].bom().totalBytes(),11u);
    EXPECT_EQ(first_read[1].budgets[0].bom().totalBytes(),17u);
    EXPECT_EQ(second_read[0].budgets[0].bom().totalBytes(),23u);
    EXPECT_EQ(second_read[0].budgets[0].worldRank(),1);
    EXPECT_EQ(second_read[0].budgets[0].bom().charge(PhysicalMemoryOwner::NativeCPUCollectiveDomain).already_resident_bytes,3u);
    EXPECT_THROW(parseOriginalPipelineCapacityFrame(first_frame.bytes(),1,topology),std::invalid_argument);
    EXPECT_THROW(OriginalPipelineCapacitySendFrame::prepare(topology,0,second_parts),std::invalid_argument);
    const std::array foreign_part{OriginalPipelineFixedContribution{0,first_head}};
    EXPECT_THROW(OriginalPipelineCapacitySendFrame::prepare(topology,1,foreign_part),std::invalid_argument);
    const std::array duplicate_parts{first_parts[0],first_parts[0]};
    EXPECT_THROW(OriginalPipelineCapacitySendFrame::prepare(topology,0,duplicate_parts),std::invalid_argument);
    auto bytes=std::vector<std::uint8_t>(first_frame.bytes().begin(),first_frame.bytes().end());
    bytes.push_back(0);
    EXPECT_THROW(parseOriginalPipelineCapacityFrame(bytes,0,topology),std::invalid_argument);
    bytes.resize(bytes.size()-2);
    EXPECT_THROW(parseOriginalPipelineCapacityFrame(bytes,0,topology),std::invalid_argument);
    bytes.assign(first_frame.bytes().begin(),first_frame.bytes().end());
    bytes.at(36)=1; // Sender rank inside the first CAP4 record, independently of its outer sender.
    EXPECT_THROW(parseOriginalPipelineCapacityFrame(bytes,0,topology),std::invalid_argument);

    static constexpr std::array rows{1,8};
    const auto first_plan=AdmittedMoEPipelineMemory::admit(first,source.metadata(),source.loader(),observed,
        globalSourcePolicy(first,source.loader(),rows));
    const auto second_plan=AdmittedMoEPipelineMemory::admit(second,source.metadata(),source.loader(),observed,
        globalSourcePolicy(second,source.loader(),rows));
    EXPECT_EQ(topology.embeddingPublisherRank(), 0);
    auto nonzero_leader = topology;
    nonzero_leader.world_size = 3;
    nonzero_leader.stages.front().participating_ranks = {1, 2};
    EXPECT_EQ(nonzero_leader.embeddingPublisherRank(), 1);
    nonzero_leader.stages.front().participating_ranks = {2, 1};
    EXPECT_EQ(nonzero_leader.embeddingPublisherRank(), 1);
    auto invalid_head = nonzero_leader;
    invalid_head.stages.front().participating_ranks = {1, 1};
    EXPECT_THROW((void)invalid_head.embeddingPublisherRank(), std::logic_error);
    invalid_head = nonzero_leader;
    invalid_head.stages.back().has_embedding = true;
    EXPECT_THROW((void)invalid_head.embeddingPublisherRank(), std::logic_error);
    auto singleton_head = topology;
    singleton_head.stages.front().is_global_tp = false;
    singleton_head.stages.front().participating_ranks.clear();
    singleton_head.stages.front().owning_rank = 1;
    EXPECT_EQ(singleton_head.embeddingPublisherRank(), 1);

    // Seal actual production runtime bindings against the same immutable PMA
    // plan on each rank, before any model weights or native domains exist.
    auto model = metadataContext(file.path());
    const auto prove_binding = [&](const auto& admitted, const auto& resolved, int rank) {
        const auto sealed = std::make_shared<const AdmittedMoEPipelineMemory>(admitted);
        const auto prepared = MoEPipelinePreparedPlan::create(sealed, model);
        const auto memory = std::make_shared<PhysicalMemoryAuthority>(sealed->physicalAdmission(), rank);
        const auto context = std::make_shared<MockMPIContext>(rank, 2);
        const auto bindings = prepared->createRuntimeBindings(model->concreteLoader(), memory, context, resolved);
        ASSERT_EQ(bindings.size(), rank == 0 ? 2u : 1u);
        const auto& head_binding = *bindings.front();
        const auto& owners = head_binding.owners();
        const auto snapshot = owners.authority->snapshot();
        ASSERT_TRUE(snapshot);
        ASSERT_EQ(snapshot->ownershipKind(), MoEOverlayResidencyOwnershipKind::CPUJointFragments);
        EXPECT_EQ(snapshot->jointOwnerMap().get(), &owners.residency->initialOwnerMap());
        EXPECT_EQ(&snapshot->originalOwnerMap(), &owners.residency->initialOwnerMap());
        const auto& local = owners.residency->localParticipantIds();
        ASSERT_EQ(local.size(), 1u);
        const auto* participant = snapshot->originalOwnerMap().participantForId(local.front());
        ASSERT_NE(participant, nullptr);
        EXPECT_EQ(participant->world_rank, rank);
        EXPECT_EQ(participant->address, head.devices.at(rank));
        ASSERT_EQ(snapshot->originalOwnerMap().owners().size(), 8u);
        for (const auto& row : snapshot->originalOwnerMap().owners())
            ASSERT_TRUE(row.shard_cohort);

        if (rank == 0) {
            const auto second_bindings = prepared->createRuntimeBindings(model->concreteLoader(), memory, context, resolved);
            auto foreign = owners;
            foreign.residency = second_bindings.front()->owners().residency;
            ASSERT_NE(&foreign.residency->initialOwnerMap(), &snapshot->originalOwnerMap());
            EXPECT_THROW((void)MoEOverlayPipelineStageBinding::seal(head_binding.scope(),
                head_binding.metadata(), head.devices, std::move(foreign)), std::invalid_argument);
        }
    };
    prove_binding(first_plan, first, 0);
    prove_binding(second_plan, second, 1);
    ASSERT_EQ(first_plan.stages().size(),2u);ASSERT_EQ(second_plan.stages().size(),1u);
    ASSERT_EQ(first_plan.stages()[0].topology().stageId(),0);
    ASSERT_EQ(first_plan.stages()[1].topology().stageId(),1);
    ASSERT_EQ(second_plan.stages()[0].topology().stageId(),0);
    const auto& first_resources=first_plan.physicalAdmission()->plan().resources();
    const auto& second_resources=second_plan.physicalAdmission()->plan().resources();
    ASSERT_EQ(first_resources.size(),2u);ASSERT_EQ(second_resources.size(),2u);
    for(const auto& resource:first_resources) {
        const auto peer=std::find_if(second_resources.begin(),second_resources.end(),
            [&](const auto& item){return item.resource().sameAllocator(resource.resource());});
        ASSERT_NE(peer,second_resources.end());
        EXPECT_EQ(resource.resource().device,DeviceId::cpu());
        EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::NativeCPUCollectiveDomain),NativeCPUCollectiveProgram::domainBytes());
        EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::NativeCPUCollectiveProcessPool),NativeCPUCollectiveProgram::processPoolBytes());
        for(size_t index=0;index<PhysicalMemoryBOM::ownerCount();++index) {
            const auto owner=PhysicalMemoryBOM::ownerAt(index);
            EXPECT_EQ(resource.charge(owner).planned_bytes,peer->charge(owner).planned_bytes);
            EXPECT_EQ(resource.charge(owner).already_resident_bytes,peer->charge(owner).already_resident_bytes);
        }
    }
}

/** @test Stage framing preserves sparse and unequal rank membership beyond the serving fixture's two sockets. */
TEST(MoEGlobalPipelineMemory, OriginalCPUStageCapacityExchangeUsesDeclaredCohortWidths)
{
    for (const int population : {3, 5, 9, 17}) {
        SCOPED_TRACE(population);
        GlobalPPStageSpec head;
        head.stage_id = 0;
        head.first_layer = head.last_layer = 0;
        head.has_embedding = true;
        head.is_global_tp = true;
        GlobalPPStageSpec middle;
        middle.stage_id = 1;
        middle.first_layer = middle.last_layer = 1;
        middle.is_global_tp = true;
        GlobalPPStageSpec tail;
        tail.stage_id = 2;
        tail.first_layer = tail.last_layer = 2;
        tail.has_lm_head = true;
        tail.owning_rank = population - 2;
        for (int rank = 0; rank < population; ++rank) {
            if (rank % 2 == 0 || (population > 3 && rank == 1))
                head.participating_ranks.push_back(rank);
            if (rank % 2 != 0 || rank == population - 1)
                middle.participating_ranks.push_back(rank);
        }
        const auto topology = GlobalPPTopology::build({head, middle, tail}, 3, population);
        ASSERT_TRUE(topology.validate().empty());
        for (int rank = 0; rank < population; ++rank) {
            SCOPED_TRACE(rank);
            PhysicalMemoryBOMBuilder builder(PhysicalMemoryResource{rank, DeviceId::cpu(), 4096, 2048});
            builder.add(PhysicalMemoryOwner::NativeCPUCollectiveDomain, rank + 1);
            const std::vector local{MoEOverlayBoundPhysicalMemoryBudget{
                MoEOverlayLocalCapacityPlanner::physicalResourceId(rank, DeviceId::cpu()),
                PhysicalMemoryAdmissionCertificate(builder.build())}};
            std::vector<OriginalPipelineFixedContribution> contributions;
            // Reverse the source order to prove the parser publishes stage
            // identity order, independent of local submission order.
            const std::array expected{rank % 2 == 0 || (population > 3 && rank == 1),
                rank % 2 != 0 || rank == population - 1,
                rank == population - 2};
            for (int stage = static_cast<int>(expected.size()) - 1; stage >= 0; --stage)
                if (expected[stage])
                    contributions.push_back({stage, local});
            const auto frame = OriginalPipelineCapacitySendFrame::prepare(topology, rank, contributions);
            const auto decoded = parseOriginalPipelineCapacityFrame(frame.bytes(), rank, topology);
            ASSERT_EQ(decoded.size(), expected.size());
            for (std::size_t stage = 0; stage < expected.size(); ++stage) {
                EXPECT_EQ(decoded[stage].stage_id, static_cast<int>(stage));
                if (expected[stage]) {
                    ASSERT_EQ(decoded[stage].budgets.size(), 1u);
                    EXPECT_EQ(decoded[stage].budgets.front().worldRank(), rank);
                    EXPECT_EQ(decoded[stage].budgets.front().bom().totalBytes(), static_cast<std::size_t>(rank + 1));
                } else {
                    EXPECT_TRUE(decoded[stage].budgets.empty());
                }
            }
        }
    }
}

}
