/**
 * @file Test__MoEPipelineMemoryAdmission.cpp
 * @brief Device-free GGUF-to-PMA regressions for four/eight-device MoE pipelines.
 *
 * Public parsing, model manifests, stage projection and aggregate admission are
 * exercised together. Sparse GGUF fixtures contain directory metadata only; no
 * accelerator, model payload or inference is substituted. Runtime setup tests
 * construct the same histogram, residency and empty prepared-bank owners used
 * by production, then hand them to the checked pipeline binding.
 */
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

INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelineResidencySetup, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });

/** @brief End-to-end metadata handoff keeps every declared physical participant. */
class MoEPipelinePreparedPlanTest : public ::testing::TestWithParam<int> {};

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

INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelinePreparedPlanTest, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });

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
