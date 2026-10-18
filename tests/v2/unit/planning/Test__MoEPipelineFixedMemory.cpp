/**
 * @file Test__MoEPipelineFixedMemory.cpp
 * @brief Device-free proofs of complete local pipeline fixed-owner admission.
 *
 * Real memory adapters, transfer ownership and PMA operate on immutable model
 * metadata and observed resources. No weights, GPU context or prefix payload is
 * allocated. Shared archive backing must appear once while every actual stage
 * cache and channel endpoint remains represented in the joint fixed BOM.
 */
#include "planning/MoEOverlayMemoryPlanInputs.h"
#include "planning/MoEOverlayPlanningInputs.h"
#include "planning/RankMemoryPlanInputs.h"
#include "execution/prefix_cache/PrefixArchiveIOGeometry.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

namespace llaminar2
{
namespace
{
    /** @brief Authored stage geometry; ordinals are assigned once per physical backend. */
    struct StageGeometry
    {
        DeviceType backend;
        int width;
        int layers = 2;
    };

    /** @brief Complete stage declarations sharing one full source and observation. */
    struct Fixture
    {
        ModelMemoryProfile model;
        RankExecutionPlan parent;
        ClusterInventory inventory;
        std::vector<RankExecutionPlan> ranks;
        std::vector<OrchestrationConfig> configs;
        std::vector<MoEExpertOverlayExecutionPlan> executions;
        std::vector<MoEOverlayCapacityAdmissionPolicy> policies;
        std::vector<MoEOverlayInferenceGraphFamilyIdentity> families;
        std::vector<int> buckets{1, 8, 32, 128};

        /**
         * @brief Declare real owner geometry without starting a native backend.
         * @param reverse Swap the CUDA/ROCm stage order.
         * @param width Physical participants in each stage.
         * @param dense Actual dense weight layout, independent of channel leadership.
         */
        Fixture(bool reverse, int width, DenseParallelPolicy dense = DenseParallelPolicy::Replicated)
            : Fixture({{reverse ? DeviceType::ROCm : DeviceType::CUDA, width},
                       {reverse ? DeviceType::CUDA : DeviceType::ROCm, width}}, dense) {}

        /**
         * @brief Declare arbitrary PP stages with distinct physical membership.
         * @param stages Ordered backend, TP width and owned main-layer count.
         * @param dense Dense layout shared by the authored stage declarations.
         */
        explicit Fixture(const std::vector<StageGeometry> &stages,
                         DenseParallelPolicy dense = DenseParallelPolicy::Replicated)
            : ranks(stages.size()), configs(stages.size()), executions(stages.size()),
              policies(stages.size()), families(stages.size())
        {
            model.architecture = "qwen35moe";
            model.n_layers = std::accumulate(stages.begin(), stages.end(), 0,
                [](int sum, const auto &stage) { return sum + stage.layers; });
            model.d_model = model.expert_feed_forward_length = 256;
            model.d_ff = 512;
            model.n_heads = model.n_kv_heads = 8;
            model.head_dim = 32;
            model.vocab_size = 512;
            model.max_seq_len = 256;
            model.expert_count = 8;
            model.expert_used_count = 2;
            for (int layer = 0; layer < model.n_layers; ++layer)
                for (const auto suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"})
                    model.tensors.push_back({.name = "blk." + std::to_string(layer) + "." + suffix,
                        .native_bytes = size_t(model.expert_count) * 4u * 256u * 256u, .quant_type = "F32",
                        .elements = size_t(model.expert_count) * 256u * 256u, .K = 256u, .layer_index = layer});
            parent.rank = 0;
            parent.first_layer = 0;
            parent.last_layer = model.n_layers - 1;
            parent.has_embedding = parent.has_lm_head = true;
            parent.local_pp_layer_boundaries = {0};
            for (const auto &stage : stages)
                parent.local_pp_layer_boundaries.push_back(parent.local_pp_layer_boundaries.back() + stage.layers);
            parent.local_pp_stage_tp_info.resize(stages.size());
            inventory.world_size = 1;
            inventory.ranks.resize(1);
            auto &observed = inventory.ranks.front();
            observed.rank = 0;
            observed.cpu_cores = observed.cpu_worker_threads = 8;
            observed.cpu_execution = test::kSyntheticCPUExecutionGeometry;
            observed.cpu.memory_bytes = observed.cpu.free_memory_bytes = observed.cpu_memory_bytes = 256ULL << 30;
            std::map<DeviceType, int> next_ordinal;
            for (size_t stage = 0; stage < stages.size(); ++stage)
            {
                const auto [backend, width, layers] = stages[stage];
                auto &rank = ranks[stage];
                rank.rank = 0;
                rank.first_layer = parent.local_pp_layer_boundaries[stage];
                rank.last_layer = rank.first_layer + layers - 1;
                rank.has_embedding = stage == 0;
                rank.has_lm_head = stage + 1 == stages.size();
                rank.runtime.max_seq_len = 256;
                rank.runtime.batch_size = 1;
                rank.runtime.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
                rank.runtime.routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
                rank.runtime.prefix_cache.enabled = true;
                rank.runtime.prefix_cache.ram_budget_bytes = 16ULL << 30;
                rank.runtime.prefix_cache.disk_budget_bytes = 32ULL << 30;
                rank.runtime.prefix_cache.disk_dir = "/cache/shared";
                auto overlay = std::make_shared<MoERoutedExpertPlacementPlan>();
                overlay->enabled = true;
                overlay->topology = RoutedExpertPlacementTopology::SingleDomain;
                overlay->authority_execution = MoEOverlayAuthorityExecutionKind::DeviceResident;
                overlay->first_model_layer = rank.first_layer;
                overlay->base_model_domain = overlay->continuation_domain = overlay->shared_expert_domain = "child";
                overlay->continuation_domain_spec.domain = "child";
                overlay->continuation_domain_spec.setDensePolicy(dense);
                RoutedExpertDomain domain;
                domain.name = "child";
                domain.scope = width == 1 ? ExecutionDomainScope::SINGLE : ExecutionDomainScope::RANK_LOCAL;
                domain.owner_rank = 0;
                domain.backend = backend == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
                domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
                for (int member = 0; member < width; ++member)
                {
                    const int ordinal = next_ordinal[backend]++;
                    const auto address = GlobalDeviceAddress::fromLocalDeviceId(DeviceId(backend, ordinal));
                    domain.participants.push_back(address);
                    if (member == 0) rank.primary_device = address;
                    if (width > 1) rank.local_tp_devices.push_back(address);
                    observed.gpus.push_back({.type = backend, .local_device_id = ordinal,
                        .memory_bytes = 64ULL << 30, .free_memory_bytes = 64ULL << 30, .compute_units = 60});
                }
                rank.local_tp_backend = domain.backend;
                parent.local_pp_devices.push_back(rank.primary_device);
                parent.local_pp_stage_tp_info[stage] = {.devices = domain.participants, .tp_backend = domain.backend};
                overlay->domains = {domain};
                overlay->routed_tiers = {{.name = "owned", .domain = "child", .priority = 0, .fallback = true}};
                configs[stage].moe_routed_expert_plan = overlay;
                configs[stage].moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
                configs[stage].routed_expert_compute_policy = RoutedExpertComputePolicy::Apportioned;
                executions[stage] = resolveMoEExpertOverlayExecutionPlan(overlay, {.current_world_rank = 0, .world_size = 1});
                families[stage] = {.graph_family_generation = 1, .main_layer_count = layers,
                    .max_graph_rows = 128, .max_decode_rows = 1, .max_request_count = 1,
                    .first_model_layer = rank.first_layer};
                policies[stage] = resolveMoEOverlayCapacityAdmissionPolicy(*overlay, configs[stage], 1, layers, model.expert_count);
            }
        }

        /** @return Exact borrowed inputs from the production stage-memory adapter. */
        std::vector<MoEOverlayLocalCapacityPlannerInput> inputs(int rows = 32) const
        {
            std::vector<MoEOverlayLocalCapacityPlannerInput> result;
            for (size_t stage = 0; stage < ranks.size(); ++stage)
            {
                const auto &rank = ranks[stage];
                auto projected = buildMoEOverlayMemoryPlanInputs({
                    .model = model, .rank_plan = rank, .config = configs[stage], .inventory = inventory,
                    .execution = executions[stage], .capacity_policy = policies[stage],
                    .retained_mtp = rank.runtime.mtp, .graph_family = families[stage],
                    .prefill = {.bucket_rows = buckets, .minimum_sequence_rows = 1, .maximum_cached_buckets = 16},
                    .gpu_weight_load = {.maximum_source_bytes = 65536},
                    .pipeline_stage = FactoryPPStageConfig{.first_layer = rank.first_layer, .last_layer = rank.last_layer + 1,
                        .has_embedding = rank.has_embedding, .has_lm_head = rank.has_lm_head}}, rows);
                result.push_back(std::move(projected.local_capacity));
            }
            return result;
        }

        /** @return Exact routed tensor directory for the declared owned stage. */
        std::vector<MoEOverlayLayerWeightManifest> manifest(size_t stage) const
        {
            std::vector<MoEOverlayLayerWeightManifest> result;
            for (int layer = ranks[stage].first_layer; layer <= ranks[stage].last_layer; ++layer)
                result.push_back({.layer_idx = layer, .projections = {{
                    {.projection = ExpertTierWeightProjection::Gate, .N = 256, .K = 256,
                     .format = ExpertWeightFormat::floating(TensorType::FP32)},
                    {.projection = ExpertTierWeightProjection::Up, .N = 256, .K = 256,
                     .format = ExpertWeightFormat::floating(TensorType::FP32)},
                    {.projection = ExpertTierWeightProjection::Down, .N = 256, .K = 256,
                     .format = ExpertWeightFormat::floating(TensorType::FP32)}}}});
            return result;
        }

        /**
         * @brief Submit all fixed owners and routed source rows to the real joint resolver.
         * @param fixed Per-stage contributions from the common fixed-memory transaction.
         * @return One shared complete certificate plus the stage-local quota grants.
         */
        std::vector<MoEOverlayResolvedCapacityPlan> admit(
            const std::vector<MoEOverlayLocalCapacityPlannerResult> &fixed) const
        {
            std::vector<std::vector<MoEOverlayLayerWeightManifest>> manifests;
            for (size_t stage = 0; stage < ranks.size(); ++stage) manifests.push_back(manifest(stage));
            std::vector<MoEOverlayStageCapacityRequest> requests;
            for (size_t stage = 0; stage < ranks.size(); ++stage)
                requests.push_back({*configs[stage].moe_routed_expert_plan, model.expert_count,
                    manifests[stage], fixed.at(stage).physical_budgets, policies[stage]});
            return MoEOverlayCapacityAdmission::resolvePipelineCapacity(requests);
        }
    };
}

/** @test The real fixed planner carries all caches and exactly the two channel leaders. */
TEST(MoEPipelineFixedMemory, SharedArchiveAndPhysicalChannelsSurviveStageProjection)
{
    for (const bool reverse : {false, true})
        for (const int width : {1, 2, 4})
            for (const auto dense : {DenseParallelPolicy::Replicated, DenseParallelPolicy::TensorParallel,
                                    DenseParallelPolicy::PrefillTensorParallelDecodeReplicated})
            {
                SCOPED_TRACE(::testing::Message() << reverse << '/' << width << '/' << int(dense));
                if (width == 1 && dense != DenseParallelPolicy::Replicated)
                {
                    EXPECT_THROW((void)Fixture(reverse, width, dense), std::invalid_argument);
                    continue;
                }
                Fixture fixture(reverse, width, dense);
                const auto inputs = fixture.inputs();
                const auto plans = MoEOverlayLocalCapacityPlanner::planPipeline(fixture.parent, inputs);
                ASSERT_EQ(plans.size(), 2u);
                PhysicalMemoryPlanBuilder aggregate;
                for (size_t stage = 0; stage < 2; ++stage)
                {
                    ASSERT_EQ(plans[stage].device_inputs.size(), size_t(width));
                    for (size_t member = 0; member < size_t(width); ++member)
                    {
                        const auto &config = plans[stage].device_inputs[member];
                        EXPECT_EQ(config.first_layer, fixture.ranks[stage].first_layer);
                        EXPECT_EQ(config.last_layer, fixture.ranks[stage].last_layer);
                        EXPECT_FALSE(config.local_pipeline_backend);
                        if (member == 0)
                        {
                            ASSERT_EQ(config.captured_pipeline_boundaries.size(), 1u);
                            EXPECT_EQ(config.captured_pipeline_boundaries.front(), stage == 0
                                ? PipelineBoundarySide::EarlierDomain : PipelineBoundarySide::LaterDomain);
                        }
                        else EXPECT_TRUE(config.captured_pipeline_boundaries.empty());
                    }
                    for (const auto &bom : plans[stage].physical_memory_admission->plan().resources()) aggregate.add(bom);
                }
                const auto admitted = PhysicalMemoryPlanAdmissionCertificate(aggregate.build());
                const auto *host = admitted.plan().find({0, DeviceId::cpu()});
                ASSERT_NE(host, nullptr);
                EXPECT_EQ(host->bytes(PhysicalMemoryOwner::PrefixHostTier), size_t(2 * width) * (16ULL << 30));
                EXPECT_EQ(host->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), PrefixArchiveIOGeometry::scratchBytes());
                const auto channel = PipelineTransferMemory::forRows(fixture.model.d_model, 32, 1);
                EXPECT_EQ(host->bytes(PhysicalMemoryOwner::ActivationTransportStaging), channel.host_bytes_per_boundary);
                for (size_t stage = 0; stage < 2; ++stage)
                    for (int member = 0; member < width; ++member)
                    {
                        const auto device = fixture.parent.local_pp_stage_tp_info[stage].devices[member].toLocalDeviceId();
                        const auto *gpu = admitted.plan().find({0, device});
                        ASSERT_NE(gpu, nullptr);
                        EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::ActivationTransportStaging),
                            member == 0 ? channel.device_bytes_per_boundary : 0u);
                    }
                const auto complete = fixture.admit(plans);
                ASSERT_EQ(complete.size(), 2u);
                ASSERT_EQ(complete[0].physical_memory_admission, complete[1].physical_memory_admission);
                const auto *complete_host = complete[0].physical_memory_admission->plan().find({0, DeviceId::cpu()});
                ASSERT_NE(complete_host, nullptr);
                EXPECT_EQ(complete_host->bytes(PhysicalMemoryOwner::PrefixHostTier), size_t(2 * width) * (16ULL << 30));
                EXPECT_EQ(complete_host->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), PrefixArchiveIOGeometry::scratchBytes());
                EXPECT_EQ(complete_host->bytes(PhysicalMemoryOwner::ActivationTransportStaging), channel.host_bytes_per_boundary);
            }
}

/** @test Input groups cannot lose or reorder the authored stage membership. */
TEST(MoEPipelineFixedMemory, RejectsMissingForeignAndOverflowingStageOwnership)
{
    for (int mutation = 0; mutation < 11; ++mutation)
    {
        SCOPED_TRACE(mutation);
        Fixture fixture(false, 2);
        auto inputs = fixture.inputs();
        switch (mutation)
        {
            case 0: inputs.pop_back(); break;
            case 1: std::swap(inputs[0], inputs[1]); break;
            case 2: fixture.ranks[1].first_layer = 0; break;
            case 3: fixture.ranks[1].has_embedding = true; break;
            case 4: fixture.ranks[0].local_tp_devices.pop_back(); break;
            case 5: fixture.ranks[0].local_tp_devices[1] = GlobalDeviceAddress::cuda(5); break;
            case 6: fixture.parent.local_pp_layer_boundaries[1] = std::numeric_limits<int>::min(); break;
            case 7: fixture.parent.last_layer = std::numeric_limits<int>::max(); break;
            case 8: inputs[1].model_profile = nullptr; break;
            case 9: inputs[1].resident_graph_rows *= 2; break;
            case 10: inputs[1].activation_channel_row_capacity *= 2; break;
        }
        EXPECT_THROW((void)MoEOverlayLocalCapacityPlanner::planPipeline(fixture.parent, inputs), std::invalid_argument);
    }
}

/** @test Individually fitting RAM tiers require one combined physical admission. */
TEST(MoEPipelineFixedMemory, SharedHostCapacityMustAdmitAllStageContributions)
{
    Fixture fixture(false, 2);
    auto &host = fixture.inventory.ranks[0];
    host.cpu.memory_bytes = host.cpu.free_memory_bytes = host.cpu_memory_bytes = 48ULL << 30;
    const auto inputs = fixture.inputs();
    const auto plans = MoEOverlayLocalCapacityPlanner::planPipeline(fixture.parent, inputs);
    ASSERT_EQ(plans.size(), 2u);
    PhysicalMemoryPlanBuilder combined;
    for (const auto &stage : plans)
        for (const auto &bom : stage.physical_memory_admission->plan().resources()) combined.add(bom);
    EXPECT_THROW((void)PhysicalMemoryPlanAdmissionCertificate(combined.build()), PhysicalMemoryCapacityExhausted);
    EXPECT_THROW((void)fixture.admit(plans), PhysicalMemoryCapacityExhausted);
}

/** @test Invalid child or host identity must fail before publishing transfer owners. */
TEST(MoEPipelineFixedMemory, TransferBindingRejectsForeignOrDuplicateOwners)
{
    Fixture fixture(false, 2);
    const auto transfer = PipelineStageTransferMemory::forStage(fixture.parent, 0);
    DevicePlanConfig valid;
    valid.world_rank = 0;
    valid.device = fixture.parent.local_pp_devices[0].toLocalDeviceId();
    valid.first_layer = 0;
    valid.last_layer = 1;
    valid.associated_host_memory = PhysicalMemoryResource{.world_rank = 0, .device = DeviceId::cpu(),
        .total_bytes = 256ULL << 30, .admission_available_bytes = 256ULL << 30};
    for (int mutation = 0; mutation < 8; ++mutation)
    {
        auto config = valid;
        switch (mutation)
        {
            case 0: config.associated_host_memory.reset(); break;
            case 1: config.associated_host_memory->world_rank = 1; break;
            case 2: config.first_layer = 1; break;
            case 3: config.last_layer = 2; break;
            case 4: config.world_rank = 1; break;
            case 5: config.device = DeviceId::cuda(5); break;
            case 6: config.execution_role = DeviceExecutionMemoryRole::RoutedExpertParticipant; break;
            case 7: transfer.bind(config); break;
        }
        EXPECT_THROW(transfer.bind(config), std::invalid_argument);
    }
    auto nonleader = valid;
    nonleader.device = fixture.parent.local_pp_stage_tp_info[0].devices[1].toLocalDeviceId();
    nonleader.associated_host_memory.reset();
    EXPECT_FALSE(transfer.requiresHostMemory(nonleader.device));
    transfer.bind(nonleader);
    EXPECT_TRUE(nonleader.captured_pipeline_boundaries.empty());
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(fixture.parent, 2), std::invalid_argument);
    fixture.parent.local_pp_stage_tp_info[0].devices[1] = fixture.parent.local_pp_stage_tp_info[0].devices[0];
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(fixture.parent, 0), std::invalid_argument);
}
/** @brief Explicit four-/eight-device tests remain discoverable in preflight evidence. */
class MoEPipelineMultiDeviceMemory : public ::testing::TestWithParam<int> {};

/** @test Every middle stage, cache owner and channel survives complete joint admission. */
TEST_P(MoEPipelineMultiDeviceMemory, CompleteFixedOwnersAndExactBoundaryExtents)
{
    const int count = GetParam();
    for (const auto first_backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto other_backend = first_backend == DeviceType::CUDA ? DeviceType::ROCm : DeviceType::CUDA;
        const auto alternating = [&](int width) {
            std::vector<StageGeometry> stages;
            for (int stage = 0; stage < count / width; ++stage)
                stages.push_back({stage % 2 ? other_backend : first_backend, width, stage % 2 ? 3 : 1});
            return stages;
        };
        const std::vector<std::vector<StageGeometry>> shapes{
            std::vector<StageGeometry>(count, {first_backend, 1}), alternating(1), alternating(2),
            {{first_backend, count / 2, 1}, {other_backend, count / 2, 3}},
            {{first_backend, 1, 3}, {other_backend, count - 1, 1}}};
        for (size_t shape = 0; shape < shapes.size(); ++shape)
        {
            SCOPED_TRACE(::testing::Message() << count << '/' << int(first_backend) << '/' << shape);
            Fixture fixture(shapes[shape]);
            const auto plans = MoEOverlayLocalCapacityPlanner::planPipeline(fixture.parent, fixture.inputs());
            const auto complete = fixture.admit(plans);
            ASSERT_EQ(complete.size(), shapes[shape].size());
            const auto certificate = complete.front().physical_memory_admission;
            ASSERT_TRUE(certificate);
            const auto &physical = certificate->plan();
            const auto *host = physical.find({0, DeviceId::cpu()});
            ASSERT_NE(host, nullptr);
            EXPECT_EQ(host->bytes(PhysicalMemoryOwner::PrefixHostTier), size_t(count) * (16ULL << 30));
            EXPECT_EQ(host->bytes(PhysicalMemoryOwner::PrefixArchiveStaging), PrefixArchiveIOGeometry::scratchBytes());
            const auto channel = PipelineTransferMemory::forRows(fixture.model.d_model, 32, 1);
            size_t cross_backend_boundaries = 0;
            size_t actual_members = 0;
            for (size_t stage = 0; stage < plans.size(); ++stage)
            {
                EXPECT_EQ(complete[stage].physical_memory_admission, certificate);
                const bool incoming = stage && shapes[shape][stage - 1].backend != shapes[shape][stage].backend;
                const bool outgoing = stage + 1 < plans.size() &&
                    shapes[shape][stage + 1].backend != shapes[shape][stage].backend;
                cross_backend_boundaries += outgoing;
                const auto &members = plans[stage].device_inputs;
                ASSERT_EQ(members.size(), size_t(shapes[shape][stage].width));
                actual_members += members.size();
                for (size_t member = 0; member < members.size(); ++member)
                {
                    const auto &config = members[member];
                    const auto *gpu = physical.find({0, config.device});
                    ASSERT_NE(gpu, nullptr);
                    EXPECT_EQ(config.first_layer, fixture.ranks[stage].first_layer);
                    EXPECT_EQ(config.last_layer, fixture.ranks[stage].last_layer);
                    EXPECT_EQ(config.captured_pipeline_boundaries.size(), member == 0 ? incoming + outgoing : 0);
                    EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::ActivationTransportStaging),
                        member == 0 ? size_t(incoming + outgoing) * channel.device_bytes_per_boundary : 0u);
                }
            }
            EXPECT_EQ(actual_members, size_t(count));
            EXPECT_EQ(physical.resources().size(), size_t(count + 1));
            EXPECT_EQ(host->bytes(PhysicalMemoryOwner::ActivationTransportStaging),
                cross_backend_boundaries * channel.host_bytes_per_boundary);
        }
    }
}

/** @test The joint transaction rejects overcommit hidden by individually fitting stages. */
TEST_P(MoEPipelineMultiDeviceMemory, AggregateHostBudgetCannotBeReusedByEveryStage)
{
    const int count = GetParam();
    std::vector<StageGeometry> stages;
    for (int stage = 0; stage < count; ++stage)
        stages.push_back({stage % 2 ? DeviceType::CUDA : DeviceType::ROCm, 1});
    Fixture fixture(stages);
    auto &host = fixture.inventory.ranks[0];
    host.cpu.memory_bytes = host.cpu.free_memory_bytes = host.cpu_memory_bytes = size_t(count - 1) * (16ULL << 30);
    const auto plans = MoEOverlayLocalCapacityPlanner::planPipeline(fixture.parent, fixture.inputs());
    ASSERT_EQ(plans.size(), size_t(count));
    EXPECT_THROW((void)fixture.admit(plans), PhysicalMemoryCapacityExhausted);
}

/** @test Later-stage corruption cannot be hidden by validating only the first two peers. */
TEST_P(MoEPipelineMultiDeviceMemory, RejectsForeignMiddleAndTerminalScope)
{
    const int count = GetParam();
    for (int mutation = 0; mutation < 5; ++mutation)
    {
        SCOPED_TRACE(::testing::Message() << count << '/' << mutation);
        std::vector<StageGeometry> stages;
        for (int stage = 0; stage < count; ++stage)
            stages.push_back({stage % 2 ? DeviceType::CUDA : DeviceType::ROCm, 1});
        Fixture fixture(stages);
        auto inputs = fixture.inputs();
        switch (mutation)
        {
            case 0: std::swap(inputs[count / 2], inputs.back()); break;
            case 1: fixture.ranks.back().has_lm_head = false; break;
            case 2: inputs.back().graph_family.first_model_layer = 0; break;
            case 3: fixture.ranks[count / 2].last_layer++; break;
            case 4: inputs.back().activation_channel_row_capacity++; break;
        }
        EXPECT_THROW((void)MoEOverlayLocalCapacityPlanner::planPipeline(fixture.parent, inputs), std::invalid_argument);
    }
}

INSTANTIATE_TEST_SUITE_P(DeviceCounts, MoEPipelineMultiDeviceMemory, ::testing::Values(4, 8),
    [](const auto &info) { return "Devices" + std::to_string(info.param); });
} // namespace llaminar2
