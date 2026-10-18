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
#include <map>

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
        rank.cpu_cores = rank.cpu_worker_threads = 8;
        rank.cpu_execution = kSyntheticCPUExecutionGeometry;
        rank.cpu.memory_bytes = rank.cpu.free_memory_bytes = rank.cpu_memory_bytes = 256ULL << 30;
        for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        {
            for (int ordinal = 0; ordinal < width; ++ordinal)
                rank.gpus.push_back({.type = backend, .local_device_id = ordinal,
                    .memory_bytes = 32ULL << 30, .free_memory_bytes = 24ULL << 30,
                    .compute_units = 64, .uuid = std::string(deviceTypeToString(backend)) + "-observed-" + std::to_string(ordinal)});
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
    { return {.prefill_bucket_rows = {1, 8, 32}, .minimum_prefill_sequence_rows = 1, .maximum_cached_prefill_buckets = 16}; }

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
        const auto bindings = prepared->createRuntimeBindings(model->concreteLoader(), memory, MPIContextFactory::self(), current);
        ASSERT_EQ(bindings.size(), 2u);
        EXPECT_EQ(bindings[1]->owners().residency->endpoint(0)->numLayers(), 2);
        EXPECT_EQ(memory->admission(), admitted.pipelineMemory()->physicalAdmission());
    }
}

INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelinePreparedPlanTest, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });

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
                        EXPECT_EQ(memory->rankAttestationSummary() != before, host_dynamic);
                    }
                    EXPECT_EQ(memory->rankAttestationSummary(), before);
                    if (host_dynamic)
                    {
                        auto no_demand = capacity;
                        no_demand.host_demand_memory.reset();
                        EXPECT_THROW((void)MoEOverlayResidencySetup::create({
                            .plan = plan, .metadata = metadata, .loader = source.loader(), .rebalance = config.moe_rebalance,
                            .storage_policy = storage, .capacity = no_demand, .memory = memory}), std::logic_error);
                        // The histogram claims its banks before the authority rejects
                        // invalid growth. Exception unwinding must retire those bytes.
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
