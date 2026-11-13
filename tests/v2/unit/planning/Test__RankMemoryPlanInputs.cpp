/**
 * @file Test__RankMemoryPlanInputs.cpp
 * @brief Device-free proof of the shared runtime/automatic-planner BOM boundary.
 *
 * Sweep backend and TP degree while retaining the exact graph, shifted MTP KV,
 * mirrored weights, prefix and fresh-load owners. No model loading or driver
 * calls occur; these are policy/geometry proofs before physical admission.
 *
 * Original QFN declarations and actual nonordinal endpoint fixtures cover every
 * supported head degree without loading model tensors or initializing GPUs.
 * CPU NUMA pipeline domains preserve every logical shard while the actual PMA
 * projection aggregates their storage into the single rank-local host allocator.
 */
#include "planning/RankMemoryPlanInputs.h"
#include "fixtures/backends/OriginalRankLocalNativeCohort.h"
#include "planning/PipelineTransferMemory.h"
#include "execution/mtp/DeviceGenerationExecutionPolicy.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/TensorParallelSourceDeclaration.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "kernels/attention/AttentionHeadMapping.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "config/OrchestrationConfig.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include "fixtures/backends/NativeContextPreparationReplayAccess.h"
#include "mocks/MockWorkerGPUContext.h"
#include "backends/NativeAuxiliaryResourceCatalog.h"
#include <gtest/gtest.h>
#include <limits>
#include <array>
#include <algorithm>
#include <vector>
#include <utility>

using namespace llaminar2;

namespace
{
    /** @brief Actual original mapped mock constructor; no native SDK is called. */
    class ReadAheadPolicyWorker final:public llaminar2::testing::MockWorkerGPUContext {
    public:
        /** @brief Preserve the original prepared-stack constructor boundary. */
        ReadAheadPolicyWorker(int ordinal,std::shared_ptr<PreparedGPUWorkerStack>):MockWorkerGPUContext(ordinal){}
    };
    /** @brief Preserve original same-PMA roots until the recorded physical reset observer runs. */
    struct ReadAheadPolicySourceRetirement final {
        std::shared_ptr<NativeContextPreparation> source;
        DeviceId device;
        /** @brief Retire the exact mock worker generation before abandoning its original Preparing authority. */
        ~ReadAheadPolicySourceRetirement(){
            auto reset=GPUDeviceContextPool::instance().beginExclusiveGenerationRetirement(device);
            test::NativeContextPreparationReplayAccess::physicalReset(reset);
            source->retireUnadmitted();
        }
    };

    /** @brief Complete synthetic metadata and rank observation for BOM assembly. */
    struct Fixture
    {
        ModelMemoryProfile model;
        RankExecutionPlan plan;
        RankInventory inventory;

        /** @brief Construct realistic GQA geometry without allocating its weights. */
        Fixture(DeviceType backend, int degree)
        {
            model.architecture = "qwen2";
            model.n_layers = 4;
            model.d_model = 512;
            model.d_ff = 1536;
            model.n_heads = 32;
            model.n_kv_heads = 2;
            model.head_dim = 16;
            model.vocab_size = 320;
            model.max_seq_len = 4096;
            plan.rank = inventory.rank = 0;
            plan.first_layer = 0;
            plan.last_layer = 3;
            plan.has_embedding = plan.has_lm_head = true;
            plan.runtime.max_seq_len = 4096;
            plan.runtime.batch_size = 1;
            plan.runtime.prefix_cache.enabled = true;
            inventory.cpu_cores = 8;
            inventory.cpu_worker_threads = 3; // Requested team is intentionally smaller than physical capacity.
            inventory.cpu_execution = test::kSyntheticCPUExecutionGeometry;
            inventory.cpu.memory_bytes = 128ull << 30;
            inventory.cpu.free_memory_bytes = 96ull << 30;
            for (int ordinal = 0; ordinal < degree; ++ordinal)
            {
                const auto device = DeviceId(backend, ordinal);
                const auto address = GlobalDeviceAddress::fromLocalDeviceId(device, "node", 0);
                if (ordinal == 0) plan.primary_device = address;
                if (degree > 1) plan.local_tp_devices.push_back(address);
                if (!device.is_gpu()) continue;
                DeviceInfo gpu;
                gpu.type = backend;
                gpu.local_device_id = ordinal;
                gpu.uuid = "fixture-" + std::to_string(ordinal);
                gpu.memory_bytes = 32ull << 30;
                gpu.free_memory_bytes = 24ull << 30;
                gpu.compute_units = 60;
                inventory.gpus.push_back(gpu);
            }
        }

        /** @return Exact ordinary serving policy with its fresh upload geometry. */
        RankMemoryPlanInputRequest request() const
        {
            return {
                .model = model, .plan = plan, .inventory = inventory,
                .weight_load_geometry = resolveGPUWeightLoadMemoryGeometry(4096, {}),
                .snapshot_capacity = std::nullopt,
                .captured_prefill_buckets = std::vector<int>{32, 64, 128, 256},
            };
        }
    };

    /**
     * @brief Declare one GPU domain and one CPU NUMA domain without device work.
     * @param backend Accelerator vendor for the peer domain.
     * @param width Number of independently authored workers in each domain.
     * @param cpu_first Whether CPU owns the earlier pipeline interval.
     * @param reverse_members Whether communicator order runs opposite NUMA order.
     * @return Complete rank fixture with distinct compute and storage identities.
     */
    Fixture cpuPipelineFixture(DeviceType backend, int width, bool cpu_first, bool reverse_members)
    {
        Fixture fixture(backend, width);
        std::vector<GlobalDeviceAddress> cpu, gpu;
        for (int ordinal = 0; ordinal < width; ++ordinal)
        {
            cpu.push_back(GlobalDeviceAddress::cpu(ordinal, "node"));
            gpu.push_back(GlobalDeviceAddress::fromLocalDeviceId(DeviceId(backend, ordinal), "node", 0));
        }
        if (reverse_members)
        {
            std::reverse(cpu.begin(), cpu.end());
            std::reverse(gpu.begin(), gpu.end());
        }
        fixture.plan.local_tp_devices.clear();
        fixture.plan.local_pp_layer_boundaries = {0, 2, 4};
        fixture.plan.local_pp_stage_tp_info = cpu_first
            ? std::vector<RankExecutionPlan::LocalPPStageTPInfo>{{.devices = cpu}, {.devices = gpu}}
            : std::vector<RankExecutionPlan::LocalPPStageTPInfo>{{.devices = gpu}, {.devices = cpu}};
        fixture.plan.local_pp_devices = {
            fixture.plan.local_pp_stage_tp_info[0].devices.front(),
            fixture.plan.local_pp_stage_tp_info[1].devices.front()};
        fixture.plan.primary_device = fixture.plan.local_pp_devices.front();
        return fixture;
    }
}

TEST(RankMemoryPlanInputs, CPUWorkspaceRequiresObservedWorkersNotPhysicalCoreFallback)
{
    Fixture fixture(DeviceType::CPU, 1);
    for (int workers : {1, 3, 16})
    {
        fixture.inventory.cpu_worker_threads = workers;
        const auto configs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(configs.size(), 1u);
        EXPECT_EQ(configs.front().device_compute_units, workers);
        EXPECT_EQ(fixture.inventory.cpu_cores, 8);
    }
    fixture.inventory.cpu_worker_threads = 0;
    EXPECT_THROW(buildRankMemoryPlanInputs(fixture.request()), std::invalid_argument);
}

TEST(RankMemoryPlanInputs, BackendAndDegreeSweepPreservesExactTPAndEveryOwner)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (int degree = 1; degree <= 8; ++degree)
        {
            SCOPED_TRACE(std::to_string(static_cast<int>(backend)) + ":" + std::to_string(degree));
            Fixture fixture(backend, degree);
            const auto request = fixture.request();
            const auto configs = buildRankMemoryPlanInputs(request);
            ASSERT_EQ(configs.size(), static_cast<std::size_t>(degree));
            for (int shard = 0; shard < degree; ++shard)
            {
                const auto &cfg = configs[shard];
                EXPECT_EQ(cfg.world_rank, 0);
                EXPECT_EQ(cfg.cpu_execution, backend == DeviceType::CPU
                    ? fixture.inventory.cpu_execution : CPUExecutionGeometry{});
                EXPECT_EQ(cfg.shard_index, shard);
                EXPECT_EQ(cfg.total_shards, degree);
                EXPECT_EQ(cfg.first_layer, 0);
                EXPECT_EQ(cfg.last_layer, 3);
                EXPECT_EQ(cfg.max_seq_len, 4096);
                EXPECT_TRUE(cfg.prefix_cache.enabled);
                EXPECT_EQ(cfg.prepared_weight_admission, PreparedWeightAdmission::AllocateCompleteSet);
                EXPECT_EQ(cfg.retained_workspace_bytes, 0u);
                EXPECT_EQ(cfg.additional_weight_sets, resolveAdditionalPersistentWeightSets(
                    DenseParallelPolicy::TensorParallel, degree, fixture.plan.runtime.mtp));
                if (degree > 1)
                {
                    ASSERT_TRUE(cfg.tensor_parallel_assignment);
                    EXPECT_EQ(cfg.local_kv_heads, cfg.tensor_parallel_assignment->kv_head_count);
                    EXPECT_GT(cfg.local_kv_heads, 0);
                    ASSERT_TRUE(cfg.local_tp_backend);
                    EXPECT_NE(*cfg.local_tp_backend, CollectiveBackendType::AUTO);
                }
                else EXPECT_FALSE(cfg.local_tp_backend);
                if (backend == DeviceType::CPU)
                {
                    EXPECT_EQ(cfg.device_compute_units, fixture.inventory.cpuWorkerThreads());
                    EXPECT_FALSE(cfg.associated_host_memory);
                    EXPECT_FALSE(cfg.weight_load_staging.enabled());
                    EXPECT_FALSE(cfg.captured_serving_graphs.enabled());
                }
                else
                {
                    ASSERT_TRUE(cfg.associated_host_memory);
                    EXPECT_EQ(cfg.associated_host_memory->admission_available_bytes, 96ull << 30);
                    EXPECT_EQ(cfg.weight_load_staging.device_bytes, request.weight_load_geometry->staging_bytes);
                    EXPECT_EQ(cfg.weight_load_staging.host_bytes, request.weight_load_geometry->host_staging_bytes);
                    EXPECT_EQ(cfg.captured_serving_graphs.prefill_bucket_rows, *request.captured_prefill_buckets);
                }
            }
        }
}

/** @test Disabled speculation still retains its declared per-request seed rows. */
TEST(RankMemoryPlanInputs, SeedCapacityTracksRetainedRequestsAcrossBackendsAndTP)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const int degree : {1, 2, 4, 8})
            for (const bool mtp : {false, true})
            {
                Fixture fixture(backend, degree);
                fixture.plan.runtime.mtp.enabled = mtp;
                fixture.plan.runtime.mtp.max_request_batch = 3;
                for (const auto &config : buildRankMemoryPlanInputs(fixture.request()))
                    EXPECT_EQ(config.generation_request_capacity, 3);
            }
}

TEST(RankMemoryPlanInputs, MTPBoundedAndRetainedOffKeepTheFullFamily)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (int depth : {0, 1, 2, 3, 15})
        {
            Fixture fixture(backend, 2);
            auto &mtp = fixture.plan.runtime.mtp;
            mtp.enabled = depth > 0;
            mtp.draft_tokens = std::max(1, depth);
            mtp.graph_capacity_draft_tokens = 15;
            const auto configs = buildRankMemoryPlanInputs(fixture.request());
            for (const auto &cfg : configs)
            {
                EXPECT_TRUE(cfg.mtp_enabled) << "Retained capacity is independent of request enablement";
                EXPECT_EQ(cfg.mtp_target_query_rows, resolveMTPRetainedTargetQueryRows(mtp));
                EXPECT_EQ(cfg.mtp_shifted_kv_head_layout, resolveMTPShiftedKVHeadLayout(mtp, true, 2));
                EXPECT_EQ(cfg.mtp_terminal_logits_layout,
                    resolveMTPTerminalLogitsLayout(true, mtp.terminal_head_policy));
                if (backend != DeviceType::CPU)
                {
                    const auto execution = declaredDeviceGenerationExecutionPolicy(
                        backend, DeviceGenerationExecutionDomain::NativeDomain);
                    const auto family = resolveMTPVerifierGraphFamily(mtpVerifierExecutionRole(execution));
                    const auto expected = resolveCapturedServingGraphMemoryInventory(
                        {32, 64, 128, 256},
                        mtp,
                        family,
                        MTPGenerationParentRole::Controller,
                        DeviceGenerationExecutionDomain::NativeDomain);
                    EXPECT_EQ(cfg.captured_serving_graphs.fixed_executable_count, expected.fixed_executable_count);
                    EXPECT_EQ(cfg.captured_serving_graphs.mtp_graph_owners.sidecarGraphSlots(),
                              expected.mtp_graph_owners.sidecarGraphSlots());
                }
            }
        }
}

TEST(RankMemoryPlanInputs, SnapshotInventoryCoversTheWholeRetainedFamily)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 2);
        fixture.plan.runtime.mtp.enabled = true;
        fixture.plan.runtime.mtp.graph_capacity_draft_tokens = 15;
        auto request = fixture.request();
        request.snapshot_capacity = GraphSnapshotMemoryCapacity{
            .per_accelerator_bytes = 8192, .effective_kv = GraphSnapshotMemoryCapacity::EffectiveKV{}};
        for (const auto &cfg : buildRankMemoryPlanInputs(request))
        {
            ASSERT_TRUE(cfg.graph_snapshot_memory.effective_kv);
            EXPECT_EQ(cfg.graph_snapshot_memory.per_accelerator_bytes, 8192u);
            EXPECT_EQ(cfg.graph_snapshot_memory.effective_kv->retained_arena_count,
                cfg.graph_snapshot_memory.serving_arenas->allOwners());
            EXPECT_EQ(cfg.graph_snapshot_memory.retained_main_forward_arena_count,
                cfg.graph_snapshot_memory.serving_arenas->mainOwners());
        }
    }
}

TEST(RankMemoryPlanInputs, DistributedShardDoesNotInventALocalCollective)
{
    Fixture fixture(DeviceType::CUDA, 1);
    fixture.plan.weight_shard.total_shards = 2;
    fixture.plan.weight_shard.shard_index = 1;
    const auto configs = buildRankMemoryPlanInputs(fixture.request());
    ASSERT_EQ(configs.size(), 1u);
    EXPECT_EQ(configs[0].shard_index, 1);
    EXPECT_EQ(configs[0].total_shards, 2);
    EXPECT_FALSE(configs[0].local_tp_backend);
    EXPECT_FALSE(configs[0].tensor_parallel_assignment);
}

TEST(RankMemoryPlanInputs, InvalidObservationsFailBeforeAdmission)
{
    Fixture fixture(DeviceType::CUDA, 1);
    auto request = fixture.request();
    request.captured_prefill_buckets = std::vector<int>{};
    EXPECT_THROW(buildRankMemoryPlanInputs(request), std::invalid_argument);
    request.captured_prefill_buckets = std::vector<int>{32, 64};
    fixture.inventory.rank = 1;
    EXPECT_THROW(buildRankMemoryPlanInputs(request), std::invalid_argument);
    fixture.inventory.rank = 0;
    request.snapshot_capacity = GraphSnapshotMemoryCapacity{};
    EXPECT_THROW(buildRankMemoryPlanInputs(request), std::invalid_argument);
    request.snapshot_capacity.reset();
    fixture.inventory.gpus.clear();
    EXPECT_THROW(buildRankMemoryPlanInputs(request), std::invalid_argument);
}

TEST(RankMemoryPlanInputs, PrefixDisabledStillChargesTheFreshUploadHostRing)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 1);
        fixture.plan.runtime.prefix_cache.enabled = false;
        const auto configs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(configs.size(), 1u);
        EXPECT_GT(configs[0].weight_load_staging.host_bytes, 0u);
        ASSERT_TRUE(configs[0].associated_host_memory);
        EXPECT_EQ(configs[0].associated_host_memory->world_rank, fixture.plan.rank);
    }
}

TEST(RankMemoryPlanInputs, ExplicitMemoryLimitsReachEveryCPUAndGPUOwner)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    for (const auto limit : {0ull, 1ull << 30, 256ull << 30})
    {
        Fixture fixture(backend, 2);
        auto request = fixture.request();
        request.max_gpu_memory_bytes = limit;
        request.max_cpu_memory_bytes = limit;
        const auto configs = buildRankMemoryPlanInputs(request);
        for (const auto &cfg : configs)
        {
            EXPECT_EQ(cfg.device_free_bytes, PhysicalMemoryAuthority::admissionCapacity(
                backend == DeviceType::CPU ? 96ull << 30 : 24ull << 30, limit));
            if (cfg.associated_host_memory)
                EXPECT_EQ(cfg.associated_host_memory->admission_available_bytes,
                    PhysicalMemoryAuthority::admissionCapacity(96ull << 30, limit));
        }
        // Applying a policy must not rewrite the hardware observation or grant
        // free RAM when a caller explicitly requested a zero-byte ceiling.
        EXPECT_EQ(fixture.inventory.cpu.free_memory_bytes, 96ull << 30);
        if (limit == 0)
            EXPECT_THROW((void)MemoryPlanner::plan(fixture.model, configs).admit(), PhysicalMemoryCapacityExhausted);
    }
}

TEST(RankMemoryPlanInputs, MemoryIntentConversionIsCheckedAndVendorSymmetric)
{
    OrchestrationConfig config;
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        EXPECT_FALSE(config.memoryLimitBytes(backend));
    config.max_cpu_memory_mb = 123;
    config.max_gpu_memory_mb = 789;
    EXPECT_EQ(config.memoryLimitBytes(DeviceType::CPU), 123ull << 20);
    EXPECT_EQ(config.memoryLimitBytes(DeviceType::CUDA), 789ull << 20);
    EXPECT_EQ(config.memoryLimitBytes(DeviceType::ROCm), 789ull << 20);
    config.max_cpu_memory_mb = config.max_gpu_memory_mb = 0;
    EXPECT_EQ(config.memoryLimitBytes(DeviceType::CPU), 0u);
    EXPECT_EQ(config.memoryLimitBytes(DeviceType::CUDA), 0u);
    config.max_cpu_memory_mb = config.max_gpu_memory_mb = std::numeric_limits<size_t>::max();
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        EXPECT_THROW((void)config.memoryLimitBytes(backend), std::overflow_error);
    EXPECT_THROW((void)config.memoryLimitBytes(static_cast<DeviceType>(-1)), std::invalid_argument);
}

TEST(RankMemoryPlanInputs, ExplicitCeilingCannotHideInvalidHardwareObservation)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 1);
        auto &observed = backend == DeviceType::CPU ? fixture.inventory.cpu : fixture.inventory.gpus.front();
        observed.free_memory_bytes = observed.memory_bytes + 1;
        auto request = fixture.request();
        request.max_gpu_memory_bytes = request.max_cpu_memory_bytes = 1;
        EXPECT_THROW((void)buildRankMemoryPlanInputs(request), std::invalid_argument);
    }
}

TEST(RankMemoryPlanInputs, AdaptiveCeilingAndNotInitialDepthOwnsCapacity)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 1);
        auto &mtp = fixture.plan.runtime.mtp;
        mtp.enabled = true;
        mtp.draft_tokens = 1;
        mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        mtp.depth_policy.initial_depth = 1;
        mtp.depth_policy.max_depth = 15;
        const auto configs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(configs.size(), 1u);
        EXPECT_EQ(configs[0].mtp_target_query_rows, resolveMTPRetainedTargetQueryRows(mtp));
        EXPECT_GE(configs[0].mtp_target_query_rows, 16);
    }
}

/** @test Native PP transport preserves each singleton/TP domain's layers, shards and exact byte endpoints. */
TEST(RankMemoryPlanInputs, PipelineTensorDomainsKeepTheirExactLayerIntervals)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const auto &widths : {std::vector<int>{2, 2}, std::vector<int>{1, 2},
            std::vector<int>{2, 1}, std::vector<int>{1, 2, 1}})
    {
        int participants = 0;
        for (const int width : widths) participants += width;
        Fixture fixture(backend, participants);
        const auto devices = fixture.plan.local_tp_devices;
        fixture.plan.local_tp_devices.clear();
        fixture.plan.local_pp_layer_boundaries = widths.size() == 2
            ? std::vector<int>{0, 2, 4} : std::vector<int>{0, 1, 3, 4};
        std::size_t offset = 0;
        for (const int width : widths)
        {
            RankExecutionPlan::LocalPPStageTPInfo stage;
            stage.devices.assign(devices.begin() + offset, devices.begin() + offset + width);
            fixture.plan.local_pp_devices.push_back(stage.devices.front());
            fixture.plan.local_pp_stage_tp_info.push_back(std::move(stage));
            offset += width;
        }
        const auto configs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(configs.size(), static_cast<std::size_t>(participants));
        std::size_t config_index = 0, endpoint_sum = 0;
        for (std::size_t stage = 0; stage < widths.size(); ++stage)
        for (int shard = 0; shard < widths[stage]; ++shard)
        {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(backend)
                << " stages=" << widths.size() << " stage=" << stage << " shard=" << shard);
            const auto &cfg = configs[config_index];
            EXPECT_EQ(cfg.device, devices[config_index].toLocalDeviceId());
            EXPECT_EQ(cfg.first_layer, fixture.plan.local_pp_layer_boundaries[stage]);
            EXPECT_EQ(cfg.last_layer, fixture.plan.local_pp_layer_boundaries[stage + 1] - 1);
            EXPECT_EQ(cfg.owns_embedding, stage == 0);
            EXPECT_EQ(cfg.shard_index, shard);
            EXPECT_EQ(cfg.total_shards, widths[stage]);
            ASSERT_TRUE(cfg.local_pipeline_backend);
            EXPECT_EQ(*cfg.local_pipeline_backend, backend == DeviceType::CUDA
                ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
            EXPECT_TRUE(cfg.captured_pipeline_boundaries.empty());
            EXPECT_FALSE(cfg.pipeline_native_activation_fanout);
            EXPECT_EQ(cfg.pipeline_forward_only_graph_owner, PipelineForwardOnlyGraphOwner::Participant);
            // Every stage member receives adjacent activation/metadata edges.
            // Only its actual leader sends the peer stage's individual rows.
            std::size_t endpoints = std::size_t(stage > 0) + std::size_t(stage + 1 < widths.size());
            if (shard == 0)
            {
                if (stage > 0) endpoints += widths[stage - 1];
                if (stage + 1 < widths.size()) endpoints += widths[stage + 1];
            }
            EXPECT_EQ(cfg.native_pipeline_byte_endpoints, endpoints);
            endpoint_sum += cfg.native_pipeline_byte_endpoints;
            if (widths[stage] > 1)
            {
                ASSERT_TRUE(cfg.tensor_parallel_assignment);
                EXPECT_EQ(cfg.tensor_parallel_assignment->device, cfg.device);
            }
            else EXPECT_FALSE(cfg.tensor_parallel_assignment);
            ++config_index;
        }
        std::size_t expected_endpoint_sum = 0;
        for (std::size_t stage = 1; stage < widths.size(); ++stage)
            expected_endpoint_sum += 2u * (widths[stage - 1] + widths[stage]);
        EXPECT_EQ(endpoint_sum, expected_endpoint_sum);
        fixture.plan.local_pp_layer_boundaries.pop_back();
        EXPECT_THROW(buildRankMemoryPlanInputs(fixture.request()), std::invalid_argument);
    }
}

/** @test NUMA compute membership stays complete while physical host capacity is counted once. */
TEST(RankMemoryPlanInputs, CPUNumaPipelineDomainsShareStorageWithoutLosingShards)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int width : {1, 2, 4, 8})
    for (const bool cpu_first : {false, true})
    for (const bool reversed : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << static_cast<int>(backend) << ":" << width
            << " cpu_first=" << cpu_first << " reversed=" << reversed);
        const auto fixture = cpuPipelineFixture(backend, width, cpu_first, reversed);
        const auto configs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(configs.size(), size_t(width * 2));
        size_t host_kv_bytes = 0;
        for (int stage = 0; stage < 2; ++stage)
        for (int shard = 0; shard < width; ++shard)
        {
            const auto &cfg = configs[stage * width + shard];
            const bool cpu = stage == (cpu_first ? 0 : 1);
            EXPECT_EQ(cfg.device, fixture.plan.local_pp_stage_tp_info[stage].devices[shard].toLocalDeviceId());
            EXPECT_EQ(cfg.device.is_cpu(), cpu);
            EXPECT_EQ(cfg.shard_index, shard);
            EXPECT_EQ(cfg.total_shards, width);
            EXPECT_EQ(cfg.first_layer, stage * 2);
            EXPECT_EQ(cfg.last_layer, stage * 2 + 1);
            EXPECT_FALSE(cfg.local_pipeline_backend);
            EXPECT_TRUE(cfg.captured_pipeline_boundaries.empty());
            EXPECT_EQ(cfg.pipeline_forward_only_graph_owner, cpu
                ? PipelineForwardOnlyGraphOwner::None : PipelineForwardOnlyGraphOwner::Participant);
            if (width > 1)
            {
                ASSERT_TRUE(cfg.tensor_parallel_assignment);
                EXPECT_EQ(cfg.tensor_parallel_assignment->local_rank, shard);
            }
            if (cpu)
                host_kv_bytes += KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
                    2, 1, cfg.max_seq_len, width == 1 ? fixture.model.n_kv_heads : cfg.local_kv_heads,
                    fixture.model.head_dim, cfg.kv_precision, cfg.device);
        }
        const auto memory = MemoryPlanner::plan(fixture.model, configs);
        ASSERT_TRUE(memory.fits());
        EXPECT_NO_THROW((void)memory.admit());
        const auto &resources = memory.physicalPlan().resources();
        EXPECT_EQ(resources.size(), size_t(width + 1));
        ASSERT_EQ(std::count_if(resources.begin(), resources.end(),
            [](const auto &bom) { return bom.resource().device.is_cpu(); }), 1);
        const auto host = std::find_if(resources.begin(), resources.end(),
            [](const auto &bom) { return bom.resource().device.is_cpu(); });
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::KVCache), host_kv_bytes);
    }
}

/** @test The physical projection cannot conceal duplicate, foreign or invalid authored endpoints. */
TEST(RankMemoryPlanInputs, CPUNumaPipelineRejectsInvalidAuthoredMembership)
{
    const auto valid = cpuPipelineFixture(DeviceType::CUDA, 2, false, false);
    EXPECT_NO_THROW((void)PipelineStageTransferMemory::forStage(valid.plan, 1));
    auto plan = valid.plan;
    plan.local_pp_stage_tp_info[1].devices[1] = plan.local_pp_stage_tp_info[1].devices[0];
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 1), std::invalid_argument);
    // A meaningless CPU ordinal cannot turn the same NUMA worker into a new participant.
    plan.local_pp_stage_tp_info[1].devices[1].device_ordinal = 7;
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 1), std::invalid_argument);
    plan = valid.plan;
    std::reverse(plan.local_pp_stage_tp_info[1].devices.begin(), plan.local_pp_stage_tp_info[1].devices.end());
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 1), std::invalid_argument);
    plan = valid.plan;
    plan.local_pp_stage_tp_info[1].devices[1].hostname = "other-host";
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 1), std::invalid_argument);
    plan = valid.plan;
    plan.local_pp_stage_tp_info[1].devices[1].device_type = static_cast<DeviceType>(127);
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 1), std::invalid_argument);
    plan = valid.plan;
    plan.local_pp_stage_tp_info[1].devices[1].device_ordinal = -1;
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 1), std::invalid_argument);
    plan = valid.plan;
    plan.local_pp_devices[0].hostname = "other-host";
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 1), std::invalid_argument);
    plan = valid.plan;
    plan.local_pp_stage_tp_info[0].devices[1] = plan.local_pp_stage_tp_info[0].devices[0];
    plan.local_pp_stage_tp_info[0].devices[1].numa_node = 1;
    EXPECT_THROW((void)PipelineStageTransferMemory::forStage(plan, 0), std::invalid_argument);
}

/** @test Only an installed native PP communicator contributes its fence BOM. */
TEST(RankMemoryPlanInputs, NativePipelineMemoryIsNotTensorParallelScratch)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 2);
        fixture.plan.local_pp_devices = fixture.plan.local_tp_devices;
        fixture.plan.local_tp_devices.clear();
        fixture.plan.local_pp_layer_boundaries = {0, 2, 4};
        const auto configs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(configs.size(), 2u);
        for (const auto &config : configs)
        {
            EXPECT_FALSE(config.local_tp_backend.has_value());
            if (backend == DeviceType::CPU) EXPECT_FALSE(config.local_pipeline_backend.has_value());
            else EXPECT_EQ(config.local_pipeline_backend, backend == DeviceType::CUDA
                ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL);
        }
    }
}

/** @test One local query over replicated KV retains its exact global ownership and complete cache capacity. */
TEST(RankMemoryPlanInputs, SingleQueryReplicatedKVShardRetainsExactOwnershipAndCapacity)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const bool reversed : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(backend) << " reversed=" << reversed);
            Fixture fixture(backend, 8);
            fixture.model.n_heads = 8;
            if (reversed)
            {
                std::reverse(fixture.plan.local_tp_devices.begin(), fixture.plan.local_tp_devices.end());
                fixture.plan.primary_device = fixture.plan.local_tp_devices.front();
            }
            const auto configs = buildRankMemoryPlanInputs(fixture.request());
            ASSERT_EQ(configs.size(), 8u);
            const auto memory = MemoryPlanner::plan(fixture.model, configs);
            ASSERT_TRUE(memory.fits());
            EXPECT_NO_THROW((void)memory.admit());
            /** @brief Independent logical cache requirements grouped by the actual allocator. */
            struct ExpectedCacheOwner
            {
                PhysicalMemoryAllocatorIdentity identity;
                size_t bytes = 0;
                size_t participants = 0;
            };
            std::vector<ExpectedCacheOwner> expected_owners;
            for (int participant = 0; participant < 8; ++participant)
            {
                const auto &cfg = configs[participant];
                ASSERT_TRUE(cfg.tensor_parallel_assignment);
                const auto &assignment = *cfg.tensor_parallel_assignment;
                const auto device = fixture.plan.local_tp_devices[participant].toLocalDeviceId();
                EXPECT_EQ(cfg.device, device);
                EXPECT_EQ(assignment.device, device);
                EXPECT_EQ(cfg.shard_index, participant);
                EXPECT_EQ(assignment.local_rank, participant);
                EXPECT_EQ(assignment.head_start, participant);
                EXPECT_EQ(assignment.head_count, 1);
                EXPECT_EQ(assignment.kv_head_start, 0);
                EXPECT_EQ(assignment.kv_head_count, 2);
                EXPECT_EQ(cfg.local_kv_heads, 2);
                EXPECT_TRUE(attention::isValidHeadMapping({1, 2, participant, 4}));
                const auto expected = KVCacheMemoryEstimator::estimate(KVCacheFamily::AttentionOnly,
                    4, 1, 4096, 2, fixture.model.head_dim, cfg.kv_precision, device);
                EXPECT_GT(expected, 0u);
                const PhysicalMemoryAllocatorIdentity identity{cfg.world_rank, device};
                auto owner = std::find_if(expected_owners.begin(), expected_owners.end(),
                    [&](const auto &entry) { return entry.identity == identity; });
                if (owner == expected_owners.end())
                {
                    expected_owners.push_back({identity, expected, 1});
                }
                else
                {
                    ASSERT_LE(expected, std::numeric_limits<size_t>::max() - owner->bytes);
                    owner->bytes += expected;
                    ++owner->participants;
                }
            }
            // Logical CPU roles share the rank host allocator; GPU ordinals own distinct banks.
            const size_t expected_owner_count = backend == DeviceType::CPU ? 1 : 8;
            const size_t expected_participants_per_owner = backend == DeviceType::CPU ? 8 : 1;
            ASSERT_EQ(expected_owners.size(), expected_owner_count);
            EXPECT_EQ(std::count_if(memory.devices.begin(), memory.devices.end(),
                [](const auto &entry) { return entry.kv_cache_bytes() > 0; }), expected_owner_count);
            const auto &physical_resources = memory.physicalPlan().resources();
            EXPECT_EQ(std::count_if(physical_resources.begin(), physical_resources.end(),
                [](const auto &entry) { return entry.bytes(PhysicalMemoryOwner::KVCache) > 0; }),
                expected_owner_count);
            for (const auto &expected : expected_owners)
            {
                EXPECT_EQ(expected.participants, expected_participants_per_owner);
                const auto planned = std::find_if(memory.devices.begin(), memory.devices.end(),
                    [&](const auto &entry)
                    {
                        const auto &resource = entry.bom().resource();
                        return resource.world_rank == expected.identity.world_rank &&
                               resource.device == expected.identity.device;
                    });
                ASSERT_NE(planned, memory.devices.end());
                EXPECT_EQ(planned->kv_cache_bytes(), expected.bytes);
                const auto *owner = memory.physicalPlan().find(expected.identity);
                ASSERT_NE(owner, nullptr);
                EXPECT_EQ(owner->bytes(PhysicalMemoryOwner::KVCache), expected.bytes);
            }
        }
}

/** @test Fewer global queries than participants still leaves an invalid empty shard. */
TEST(RankMemoryPlanInputs, EmptyQueryParticipantCannotBecomeAnAdmissibleCandidate)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 8);
        fixture.model.n_heads = 4; // Integral global4/2 GQA still leaves four query-empty participants.
        ASSERT_EQ(fixture.model.n_heads % fixture.model.n_kv_heads, 0);
        EXPECT_THROW(buildRankMemoryPlanInputs(fixture.request()), std::invalid_argument);
    }
}

/** @test Reverse vendor order and uneven TP domains assign channels only to actual leaders. */
TEST(RankMemoryPlanInputs, CapturedPipelineChannelsFollowDomainsNotDeviceOrdinals)
{
    for (const bool reverse : {false, true})
    for (const int width : {1, 2, 4, 8})
    {
        const auto first = reverse ? DeviceType::ROCm : DeviceType::CUDA;
        const auto second = reverse ? DeviceType::CUDA : DeviceType::ROCm;
        Fixture fixture(first, width);
        Fixture other(second, width);
        fixture.inventory.gpus.insert(fixture.inventory.gpus.end(), other.inventory.gpus.begin(), other.inventory.gpus.end());
        fixture.plan.runtime.prefix_cache.enabled = false;
        fixture.plan.local_tp_devices.clear();
        std::vector<GlobalDeviceAddress> earlier, later;
        for (int i = 0; i < width; ++i)
        {
            const int ordinal = (i + 1) % width; // Domain leader is not sorted by ordinal.
            earlier.push_back(GlobalDeviceAddress::fromLocalDeviceId(DeviceId(first, ordinal), "node", 0));
            later.push_back(GlobalDeviceAddress::fromLocalDeviceId(DeviceId(second, ordinal), "node", 0));
        }
        fixture.plan.local_pp_devices = {earlier.front(), later.front()};
        fixture.plan.local_pp_layer_boundaries = {0, 2, 4};
        fixture.plan.local_pp_stage_tp_info = {{.devices = earlier}, {.devices = later}};
        auto request = fixture.request();
        request.weight_load_geometry.reset(); // Host ownership below must come from the channel itself.
        const auto configs = buildRankMemoryPlanInputs(request);
        ASSERT_EQ(configs.size(), size_t(width * 2));
        for (int i = 0; i < width * 2; ++i)
        {
            const bool leader = i % width == 0;
            const bool source = i < width;
            const auto &cfg = configs[i];
            EXPECT_FALSE(cfg.local_pipeline_backend);
            if (!leader) EXPECT_TRUE(cfg.captured_pipeline_boundaries.empty());
            else EXPECT_EQ(cfg.captured_pipeline_boundaries, std::vector{source
                ? PipelineBoundarySide::EarlierDomain : PipelineBoundarySide::LaterDomain});
            EXPECT_EQ(cfg.pipeline_native_activation_fanout, !source && width > 1);
            EXPECT_EQ(cfg.associated_host_memory.has_value(), (source && leader) || (!source && width > 1));
        }
        // Remove only the new typed bank contribution to authenticate its exact
        // incremental physical quote. Channel/payload/graph geometry stays fixed.
        auto old_configs = configs;
        for (auto &cfg : old_configs) cfg.pipeline_native_activation_fanout = false;
        const auto actual = MemoryPlanner::plan(fixture.model, configs);
        const auto original = MemoryPlanner::plan(fixture.model, old_configs);
        const auto totals = [](const auto &plan, bool host) {
            size_t bytes = 0;
            for (const auto &bom : plan.physicalPlan().resources())
                if (bom.resource().device.is_cpu() == host)
                    bytes += bom.bytes(PhysicalMemoryOwner::ActivationTransportStaging);
            return bytes;
        };
        const size_t members = width > 1 ? size_t(width) : 0;
        EXPECT_EQ(totals(actual, false) - totals(original, false), members * PipelineTransferMemory::fanoutDeviceBytes());
        EXPECT_EQ(totals(actual, true) - totals(original, true), members * PipelineTransferMemory::fanoutHostBytes());
    }
}

/** @test Native terminal publication follows actual ownership/communicator, independently of MTP capacity. */
TEST(RankMemoryPlanInputs, NativeVocabularyPublicationFollowsResolvedOwnership)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    for (const int degree : {1, 2, 4, 8})
    for (const bool enabled : {false, true})
    for (const int retained : {0, 15})
    for (const auto head : {MTPTerminalHeadPolicy::MirroredFullVocabulary,
                           MTPTerminalHeadPolicy::VocabularySharded})
    {
        Fixture fixture(backend, degree);
        auto &mtp = fixture.plan.runtime.mtp;
        mtp.enabled = enabled;
        mtp.graph_capacity_draft_tokens = retained;
        mtp.terminal_head_policy = head;
        const bool native = backend != DeviceType::CPU && degree > 1 &&
            head == MTPTerminalHeadPolicy::VocabularySharded;
        for (const auto &cfg : buildRankMemoryPlanInputs(fixture.request()))
            EXPECT_EQ(cfg.terminalLogitsCollective(cfg.last_layer < 0 || cfg.last_layer >= fixture.model.n_layers - fixture.model.mtp_layer_count - 1), native
                ? MTPTerminalLogitsCollective::NativeLocalVocabularyAllGather
                : MTPTerminalLogitsCollective::None);
    }
    Fixture distributed(DeviceType::CUDA, 1);
    distributed.plan.weight_shard.total_shards = 2;
    distributed.plan.runtime.mtp.enabled = true;
    distributed.plan.runtime.mtp.terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded;
    EXPECT_EQ(buildRankMemoryPlanInputs(distributed.request()).front().terminalLogitsCollective(true),
              MTPTerminalLogitsCollective::None);
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture earlier(backend, 2);
        earlier.plan.runtime.mtp.enabled = true;
        earlier.plan.runtime.mtp.terminal_head_policy = MTPTerminalHeadPolicy::VocabularySharded;
        earlier.plan.last_layer = 1;
        earlier.plan.has_lm_head = false;
        for (const auto &cfg : buildRankMemoryPlanInputs(earlier.request()))
            EXPECT_EQ(cfg.terminalLogitsCollective(false), MTPTerminalLogitsCollective::None);
    }
}

/** @test Requested Automatic never becomes unresolved runtime storage during base row planning. */
TEST(RankMemoryPlanInputs, NgramStorageElectsOneRankOwnerAndKeepsExactUserPolicy)
{
    for (const auto backend : {DeviceType::CPU,DeviceType::CUDA,DeviceType::ROCm})
        for (const int degree : {1,2})
        {
            Fixture fixture(backend,degree);
            fixture.model.flash_next.emplace();fixture.model.flash_next->ple_decoder_layer=1;
            fixture.plan.runtime.ngram_table_policy={ImmutableRowStoragePreference::MemoryMapped,131072,7};
            const auto configs=buildRankMemoryPlanInputs(fixture.request());
            for (const auto &cfg:configs)
            {
                EXPECT_FALSE(cfg.ngram_table_placement);
                if (cfg.device.is_gpu()) ASSERT_TRUE(cfg.associated_host_memory);
            }
            EXPECT_EQ(fixture.plan.runtime.ngram_table_policy.storage,ImmutableRowStoragePreference::MemoryMapped);
        }
}


/** @test Snapshot BOM lanes match the production signature reuse policy, including distinct decode inputs. */
TEST(RankMemoryPlanInputs, SnapshotRolesMatchActualForwardSignatureSharing)
{
    using Class=DeviceGraphExecutor::GraphSnapshotArenaReuseClass;
    for(const auto device:{DeviceId::cuda(0),DeviceId::rocm(0)}) {
        ForwardGraphSignature prefill;prefill.device=device;prefill.seq_len=9;
        prefill.execution_role=ForwardExecutionRole::MainInference;
        EXPECT_EQ(graphSnapshotArenaReusePolicyForSignature(prefill).reuse_class,Class::PrefillOrMTPVerifierAlternative);
        auto history=prefill;history.decode=true;history.seq_len=1;
        auto resident=history;
        int32_t history_token=0,resident_token=0;
        history.device_token_ids=&history_token;resident.device_token_ids=&resident_token;
        EXPECT_FALSE(history==resident);
        EXPECT_EQ(graphSnapshotArenaReusePolicyForSignature(history).reuse_class,Class::Dedicated);
        EXPECT_EQ(graphSnapshotArenaReusePolicyForSignature(resident).reuse_class,Class::Dedicated);
        auto verifier=prefill;verifier.execution_role=ForwardExecutionRole::GroupedMTPVerifier;
        EXPECT_EQ(graphSnapshotArenaReusePolicyForSignature(verifier).reuse_class,
            graphSnapshotArenaReusePolicyForSignature(prefill).reuse_class);
        auto condition=history;condition.execution_role=ForwardExecutionRole::MTPCondition;
        EXPECT_EQ(graphSnapshotArenaReusePolicyForSignature(condition).reuse_class,Class::MTPConditionAlternative);
        const auto arenas=ServingSnapshotArenaInventory::ordinary(true,false,0,0);
        EXPECT_EQ(arenas.mainOwners(),3u);
    }
}

/** @test Canonical ordered TP declaration supplies provenance; sorted physical IDs cannot replace it. */
TEST(RankMemoryPlanInputs, FixedHeadDistributionUsesActualDeclarationOrderAndRejectsOtherTopologies)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8})
    {
        Fixture f(backend,degree);f.model=source.metadata().memoryProfile();f.plan.last_layer=47;
        f.plan.runtime.ngram_table_policy={ImmutableRowStoragePreference::VRAM};
        f.plan.runtime.ngram_table_policy.distribution=ImmutableRowTableDistribution::FixedHeadSharded;
        std::reverse(f.plan.local_tp_devices.begin(),f.plan.local_tp_devices.end());
        f.plan.primary_device=f.plan.local_tp_devices.front();
        const auto configs=buildRankMemoryPlanInputs(f.request());ASSERT_EQ(configs.size(),degree);
        for(int p=0;p<degree;++p)
        {
            EXPECT_EQ(configs[p].device,f.plan.local_tp_devices[p].toLocalDeviceId());
            EXPECT_EQ(configs[p].shard_index,p);
            EXPECT_EQ(configs[p].ngram_tp_topology,ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal);
        }
        for(int failure=0;failure<9;++failure)
        {
            auto bad=f;
            switch(failure)
            {
            case 0:bad.plan.local_pp_devices=bad.plan.local_tp_devices;break;
            case 1:bad.plan.prev_rank=1;break;
            case 2:bad.plan.first_layer=1;break;
            case 3:bad.plan.has_embedding=false;break;
            case 4:bad.plan.local_tp_devices[0]=GlobalDeviceAddress::cpu(0,"node");break;
            case 5:bad.plan.local_tp_devices[0]=GlobalDeviceAddress::fromLocalDeviceId(
                DeviceId(backend==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA,0),"node",0);break;
            case 6:bad.plan.local_tp_devices.resize(1);break;
            case 7:bad.plan.local_tp_devices[0].hostname="foreign-host";break;
            case 8:bad.plan.runtime.ngram_table_policy.storage=ImmutableRowStoragePreference::Automatic;break;
            }
            EXPECT_THROW((void)buildRankMemoryPlanInputs(bad.request()),std::invalid_argument)<<failure;
        }
    }
}

/** @test PMA consumes the same original-source assignment as graph/weight construction, not KV/P arithmetic. */
TEST(RankMemoryPlanInputs, QueryConsumerSourceMatchesCanonicalAssignmentEveryDegree)
{
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource original(file.path());
    for(const auto backend:{DeviceType::CUDA,DeviceType::ROCm})for(int degree=1;degree<=8;++degree)
    {
        Fixture f(backend,degree);f.model=original.metadata().memoryProfile();f.plan.last_layer=47;
        f.plan.runtime.mtp.enabled=false;f.plan.runtime.mtp.graph_capacity_draft_tokens=0;
        f.plan.runtime.ngram_table_policy.storage=ImmutableRowStoragePreference::RAM;
        // This is the actual declared discovery order; ordinal and logical rank deliberately differ.
        std::vector<DeviceId> members;
        for(int rank=0;rank<degree;++rank)
        {
            const int ordinal=21-rank*2;members.emplace_back(backend,ordinal);
            f.inventory.gpus[rank].local_device_id=ordinal;
            const auto address=GlobalDeviceAddress::fromLocalDeviceId(members.back(),"node",0);
            if(degree>1)f.plan.local_tp_devices[rank]=address;
            if(!rank)f.plan.primary_device=address;
        }
        const auto declared=resolveProfileTPDeclaration(f.model,TensorParallelMainPlacement::FullMainRankLocal,
            f.plan.runtime.mtp,DenseParallelPolicy::TensorParallel,members);
        const auto assignment=declared.partition(members,{},f.model.d_ff,f.model.vocab_size);
        const auto configs=buildRankMemoryPlanInputs(f.request());ASSERT_EQ(configs.size(),degree);
        for(int rank=0;rank<degree;++rank)
        {
            const auto &cfg=configs[rank];EXPECT_EQ(cfg.device,members[rank]);
            if(degree==1){EXPECT_FALSE(cfg.tensor_parallel_assignment);continue;}
            ASSERT_TRUE(cfg.tensor_parallel_assignment);const auto &actual=*cfg.tensor_parallel_assignment;
            const auto &expected=assignment.forRank(rank);
            EXPECT_EQ(actual.device,expected.device);EXPECT_EQ(actual.local_rank,rank);
            EXPECT_EQ(actual.head_start,expected.head_start);EXPECT_EQ(actual.head_count,expected.head_count);
            EXPECT_EQ(actual.kv_head_start,expected.kv_head_start);EXPECT_EQ(actual.kv_head_count,expected.kv_head_count);
            EXPECT_EQ(actual.d_ff_start,expected.d_ff_start);EXPECT_EQ(actual.d_ff_count,expected.d_ff_count);
            EXPECT_EQ(actual.vocab_start,expected.vocab_start);EXPECT_EQ(actual.vocab_count,expected.vocab_count);
            EXPECT_EQ(cfg.local_kv_heads,expected.kv_head_count);
        }
    }
}

/**
 * @test Mutable setup reload cannot change the already admitted schedule scalar.
 * @details Runtime construction is the only setup read. Rank planning and both
 * runner/config adapters receive copies after setup changes, without allocating
 * a graph or initializing any backend. Incoming mutable attention setup is restored.
 */
TEST(RankMemoryPlanInputs, AttentionPhaseScheduleSnapshotSurvivesDebugReloadAndAdapterCopies)
{
    using A = qwen4exp::QwenSparseAttentionArithmetic;
    using S = qwen4exp::QwenSparseAttentionPhaseSchedule;
    struct RestoreAttention final {
        AttentionConfig original;
        /** @brief Restore incoming setup on every assertion/unwind path. */
        ~RestoreAttention(){mutableDebugEnv().attention=std::move(original);}
    } restore{mutableDebugEnv().attention};
    mutableDebugEnv().attention.qfn_sparse_attention.reset();
    mutableDebugEnv().attention.qfn_sparse_attention_phase_schedule.reset();
    auto frozen = RuntimeConfig::fromOrchestrationConfig(4096, 1, "fp32", "fp16");
    EXPECT_FALSE(frozen.attention_setup_request.arithmetic);
    EXPECT_FALSE(frozen.attention_setup_request.schedule);
    mutableDebugEnv().attention.qfn_sparse_attention = A::FixedFP32;
    mutableDebugEnv().attention.qfn_sparse_attention_phase_schedule = S::Uniform;
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    Fixture fixture(DeviceType::ROCm, 4);
    fixture.model = source.metadata().memoryProfile();
    fixture.plan.last_layer = 47;
    fixture.plan.runtime = frozen;
    fixture.plan.runtime.ngram_table_policy.storage = ImmutableRowStoragePreference::RAM;
    const auto inputs=buildRankMemoryPlanInputs(fixture.request());ASSERT_EQ(inputs.size(),4u);
    for (const auto &input : inputs)
    {
        EXPECT_EQ(input.attention_arithmetic, A::TileFP16WithFP32Totals);
        EXPECT_EQ(input.attention_phase_schedule, S::Prefill8Decode33);
    }
    frozen.attention_participants =
        projectAdmittedQwenSparseAttentionPrograms(fixture.model, inputs, 0);
    const auto rank=RankOrchestrator::Config::fromRuntime(frozen);
    const auto runner=InferenceRunnerConfig::fromRuntime(frozen);
    EXPECT_EQ(rank.attention_participants, frozen.attention_participants);
    EXPECT_EQ(runner.attention_participants, frozen.attention_participants);
    EXPECT_EQ(qwen4exp::admittedQwenSparseAttentionPolicy(
                  runner.attention_participants, DeviceId::rocm(0), 0, 47)
                  .schedule,
              S::Prefill8Decode33);
    EXPECT_FALSE(frozen.attention_setup_request.arithmetic);
    EXPECT_FALSE(frozen.attention_setup_request.schedule);
}

/** @test Mixed PP copies the same resolved per-device policies into execution, never a rank-global
 * default. */
TEST(RankMemoryPlanInputs, AdmittedAttentionDefaultsKeepMixedParticipantsAndColdControls)
{
    using A = qwen4exp::QwenSparseAttentionArithmetic;
    using S = qwen4exp::QwenSparseAttentionPhaseSchedule;
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    std::vector<DevicePlanConfig> inputs;
    for (const auto device : {DeviceId::cpu(), DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        DevicePlanConfig cfg;
        cfg.device = device;
        cfg.world_rank = 0;
        cfg.batch_size = 1;
        cfg.generation_request_capacity = 1;
        cfg.kv_precision = "fp16"; // Already resolved by the canonical device-input adapter.
        RuntimeConfig snapshot;
        snapshot.kv_cache_precision = KVCachePrecision::FP16;
        resolveQwenSparseAttentionDeviceProgram(source.metadata().memoryProfile(), cfg, snapshot);
        inputs.push_back(cfg);
    }
    const auto projection =
        projectAdmittedQwenSparseAttentionPrograms(source.metadata().memoryProfile(), inputs, 0);
    for (const auto &input : inputs)
    {
        const auto copied = qwen4exp::admittedQwenSparseAttentionPolicy(
            projection,
            input.device,
            input.first_layer,
            input.last_layer < 0 ? source.metadata().memoryProfile().mainLayerCount() - 1
                                 : input.last_layer);
        EXPECT_EQ(copied.arithmetic, input.attention_arithmetic);
        EXPECT_EQ(copied.schedule, input.attention_phase_schedule);
        EXPECT_EQ(copied.arithmetic,
                  input.device.is_rocm() ? A::TileFP16WithFP32Totals : A::FixedFP32);
    }
    auto duplicate = inputs;
    duplicate.push_back(inputs.back());
    EXPECT_THROW(
        projectAdmittedQwenSparseAttentionPrograms(source.metadata().memoryProfile(), duplicate, 0),
        std::invalid_argument);
    RuntimeConfig control;
    control.kv_cache_precision = KVCachePrecision::FP16;
    control.attention_setup_request.arithmetic = A::FixedFP32;
    auto cfg = inputs.back();
    resolveQwenSparseAttentionDeviceProgram(source.metadata().memoryProfile(), cfg, control);
    EXPECT_EQ(cfg.attention_arithmetic, A::FixedFP32);
    EXPECT_EQ(cfg.attention_phase_schedule, S::Uniform);
    auto no_attention = inputs.front();
    no_attention.first_layer = 0;
    no_attention.last_layer = 2; // Published layers0..2 are GDN; first QSA is3.
    RuntimeConfig half;
    half.kv_cache_precision = KVCachePrecision::FP16;
    half.attention_setup_request = {A::TileFP16WithFP32Totals, S::Prefill8Decode33};
    EXPECT_NO_THROW(resolveQwenSparseAttentionDeviceProgram(
        source.metadata().memoryProfile(), no_attention, half));
    EXPECT_FALSE(projectAdmittedQwenSparseAttentionPrograms(
        source.metadata().memoryProfile(), {no_attention}, 0));
    no_attention.last_layer = 3;
    EXPECT_THROW(resolveQwenSparseAttentionDeviceProgram(
                     source.metadata().memoryProfile(), no_attention, half),
                 std::invalid_argument);
    for (const auto *storage : {"fp32", "q16_1", "tq4", "tq8"})
    {
        auto declared_storage = inputs.back();
        declared_storage.kv_precision = storage;
        RuntimeConfig omitted; // The helper must consume the admitted input, not reinterpret AUTO.
        resolveQwenSparseAttentionDeviceProgram(source.metadata().memoryProfile(), declared_storage, omitted);
        EXPECT_EQ(declared_storage.attention_arithmetic, A::FixedFP32) << storage;
        EXPECT_EQ(declared_storage.attention_phase_schedule, S::Uniform) << storage;
        omitted.attention_setup_request.arithmetic = A::TileFP16WithFP32Totals;
        EXPECT_THROW(resolveQwenSparseAttentionDeviceProgram(source.metadata().memoryProfile(), declared_storage, omitted), std::invalid_argument);
    }

}

/** @test Admission uses actual members, including nested and remote domain peers. */
TEST(RankMemoryPlanInputs, DeclaredGenerationDomainUsesParticipantsRatherThanInventory)
{
    Fixture fixture(DeviceType::CUDA, 1);
    fixture.plan.runtime.mtp.enabled = true;
    fixture.plan.runtime.mtp.graph_capacity_draft_tokens = 15;
    const auto native = buildRankMemoryPlanInputs(fixture.request()).front().captured_serving_graphs;
    const auto bounded = resolveCapturedServingGraphMemoryInventory(
        {32, 64, 128, 256},
        fixture.plan.runtime.mtp,
        MTPVerifierGraphFamily::BoundedBuckets,
        MTPGenerationParentRole::Controller,
        DeviceGenerationExecutionDomain::NativeDomain);
    EXPECT_EQ(native.fixed_executable_count, bounded.fixed_executable_count);
    // An unused discovered GPU is not a member of the captured generation domain.
    fixture.inventory.gpus.push_back({.type = DeviceType::ROCm, .local_device_id = 0,
        .memory_bytes = 32ull << 30, .free_memory_bytes = 24ull << 30, .compute_units = 60});
    EXPECT_EQ(declaredRankDeviceGenerationDomain(fixture.plan), DeviceGenerationExecutionDomain::NativeDomain);
    fixture.plan.my_domains.push_back({});
    fixture.plan.my_domains.back().devices = {fixture.plan.primary_device,
        GlobalDeviceAddress::fromLocalDeviceId(DeviceId::rocm(7), "remote", 1)};
    EXPECT_EQ(declaredRankDeviceGenerationDomain(fixture.plan), DeviceGenerationExecutionDomain::HostedBoundary);
    const auto hosted = buildRankMemoryPlanInputs(fixture.request()).front().captured_serving_graphs;
    const auto exact = resolveCapturedServingGraphMemoryInventory(
        {32, 64, 128, 256},
        fixture.plan.runtime.mtp,
        MTPVerifierGraphFamily::ExactRows,
        MTPGenerationParentRole::Controller,
        DeviceGenerationExecutionDomain::HostedBoundary);
    EXPECT_EQ(hosted.fixed_executable_count, exact.fixed_executable_count);
    EXPECT_EQ(hosted.mtp_graph_owners.generalAuxiliaryExecutableSlotCount(),
              exact.mtp_graph_owners.generalAuxiliaryExecutableSlotCount());
    EXPECT_GT(hosted.fixed_executable_count, native.fixed_executable_count);
    fixture.plan.my_domains.clear();
    fixture.plan.local_pp_stage_tp_info.resize(1);
    fixture.plan.local_pp_stage_tp_info[0].devices = {GlobalDeviceAddress::cpu()};
    EXPECT_EQ(declaredRankDeviceGenerationDomain(fixture.plan), DeviceGenerationExecutionDomain::HostedBoundary);
}

/** @test Exact hosted follower parents belong only to actual nonterminal PP stages. */
TEST(RankMemoryPlanInputs, LocalPipelinePricesExactFollowersAndTerminalControllerSeparately)
{
    Fixture fixture(DeviceType::ROCm, 2);
    fixture.plan.runtime.mtp.enabled = true;
    fixture.plan.runtime.mtp.graph_capacity_draft_tokens = 15;
    fixture.plan.local_pp_devices = fixture.plan.local_tp_devices;
    fixture.plan.local_tp_devices.clear();
    fixture.plan.local_pp_layer_boundaries = {0, 2, 4};
    const auto configs = buildRankMemoryPlanInputs(fixture.request());
    ASSERT_EQ(configs.size(), 2u);
    for (size_t stage = 0; stage < configs.size(); ++stage)
    {
        const auto role = stage == 0 ? MTPGenerationParentRole::PipelineFollower : MTPGenerationParentRole::Controller;
        const MTPGraphOwnerPlan owners(fixture.plan.runtime.mtp, MTPVerifierGraphFamily::ExactRows, role);
        EXPECT_EQ(configs[stage].captured_serving_graphs.mtp_graph_owners.generationParentGraphSlots(),
                  owners.generationParentGraphSlots());
        EXPECT_EQ(configs[stage].captured_serving_graphs.mtp_graph_owners.generalAuxiliaryExecutableSlotCount(),
                  owners.generalAuxiliaryExecutableSlotCount());
    }
    EXPECT_GT(configs[0].captured_serving_graphs.mtp_graph_owners.generationParentGraphSlots(),
              configs[1].captured_serving_graphs.mtp_graph_owners.generationParentGraphSlots());
    fixture.plan.local_pp_devices.clear();
    fixture.plan.local_pp_layer_boundaries.clear();
    fixture.plan.last_layer = 1;
    fixture.plan.has_lm_head = false;
    // Partial layer ownership alone must not turn a standalone owner into a follower.
    EXPECT_EQ(buildRankMemoryPlanInputs(fixture.request()).front().captured_serving_graphs
                  .mtp_graph_owners.generationParentGraphSlots(), 1u);
}

/** @test Only actual terminal sampling participants own a direct-history helper. */
TEST(RankMemoryPlanInputs, DirectTokenHistoryOwnerFollowsTerminalAcrossOneToEightDevices)
{
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (int devices = 1; devices <= 8; ++devices)
    for (bool pipeline : {false, true})
    for (bool retained : {false, true})
    {
        Fixture fixture(backend, devices);
        fixture.plan.runtime.mtp.enabled = false; // Admission must support later MTP requests.
        fixture.plan.runtime.mtp.graph_capacity_draft_tokens = retained ? 15 : 0;
        if (pipeline)
        {
            fixture.model.n_layers = devices;
            fixture.plan.last_layer = devices - 1;
            fixture.plan.local_tp_devices.clear();
            for (int stage = 0; stage < devices; ++stage)
            {
                fixture.plan.local_pp_devices.push_back(GlobalDeviceAddress::fromLocalDeviceId(
                    DeviceId(backend, stage), "node", 0));
                fixture.plan.local_pp_layer_boundaries.push_back(stage);
            }
            fixture.plan.local_pp_layer_boundaries.push_back(devices);
        }
        const auto inputs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(inputs.size(), size_t(devices));
        for (size_t participant = 0; participant < inputs.size(); ++participant)
        {
            const bool controller = !pipeline || participant + 1 == inputs.size();
            const auto &owners = inputs[participant].captured_serving_graphs.mtp_graph_owners;
            EXPECT_EQ(owners.directTokenHistoryGraphSlots(), retained && controller ? 1u : 0u)
                << "backend=" << int(backend) << " devices=" << devices
                << " pipeline=" << pipeline << " participant=" << participant;
        }
    }
}

/** @test Original selected policy prices exactly its installed side-family BOM
 * before final admission; fixture-only parent terminals never enter serving. */
TEST(RankMemoryPlanInputs, ReadAheadOriginalPolicyRequiresCertifiedBOMBeforeAdmission)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for(const auto policy : {ImmutableLookupSchedule::ConsumerScheduled,
                            ImmutableLookupSchedule::EarlyCurrentHostedCandidate,
                            ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate})
    {
        RuntimeConfig runtime;
        runtime.ngram_lookup_schedule=policy;
        EXPECT_EQ(RankOrchestrator::Config::fromRuntime(runtime).ngram_lookup_schedule,policy);
        EXPECT_EQ(InferenceRunnerConfig::fromRuntime(runtime).ngram_lookup_schedule,policy);
    }
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 1);
        fixture.model=source.metadata().memoryProfile();
        fixture.plan.last_layer=47;
        fixture.plan.runtime.mtp.enabled=false;
        fixture.plan.runtime.ngram_table_policy.storage=ImmutableRowStoragePreference::VRAM;
        fixture.plan.runtime.ngram_lookup_schedule=ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate;
        auto inputs=buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(inputs.size(),1u);
        const auto device=inputs.front().device;
        ASSERT_TRUE(inputs.front().associated_host_memory);
        const std::array devices{device};
        const std::array observations{*inputs.front().associated_host_memory,
            PhysicalMemoryResource{inputs.front().world_rank,device,inputs.front().device_total_bytes,inputs.front().device_free_bytes}};
        using SourceAccess=test::NativeContextPreparationReplayAccess;
        SourceAccess::seed(device,[device](std::shared_ptr<PreparedGPUWorkerStack> stack){
            return SourceAccess::construct<ReadAheadPolicyWorker>(std::move(stack),device.gpu_ordinal());});
        auto original=SourceAccess::source<ReadAheadPolicyWorker>(devices,observations);
        ReadAheadPolicySourceRetirement retirement{original,device};
        {const auto worker=original->acquire(device);EXPECT_EQ(worker.memory(),original->memory());}
        original->freezeForPlanning();
        inputs.front().native_context_preparation=original;
        const auto originalKey=GPUDeviceContextPool::instance().retainExistingContextProvenance(device).originalBirthKey();
        ASSERT_NE(originalKey,nullptr);
        ASSERT_EQ(inputs.size(),1u);
        EXPECT_EQ(inputs.front().ngram_lookup_schedule,fixture.plan.runtime.ngram_lookup_schedule);
        EXPECT_EQ(RankOrchestrator::Config::fromRuntime(fixture.plan.runtime).ngram_lookup_schedule,
                  fixture.plan.runtime.ngram_lookup_schedule);
        EXPECT_EQ(InferenceRunnerConfig::fromRuntime(fixture.plan.runtime).ngram_lookup_schedule,
                  fixture.plan.runtime.ngram_lookup_schedule);
        const auto expected=NativeAuxiliaryResourceCatalog::allowance(backend,
            NativeAuxiliaryResourceProgram::readAheadSide());
        const auto planned=MemoryPlanner::plan(fixture.model,inputs);
        const auto* host=planned.physicalPlan().find({inputs.front().world_rank,DeviceId::cpu()});
        const auto* gpu=planned.physicalPlan().find({inputs.front().world_rank,device});
        ASSERT_NE(host,nullptr);ASSERT_NE(gpu,nullptr);
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::NativeReadAheadFamily),expected.hostBytes);
        EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::NativeReadAheadFamily),expected.deviceBytes);
        EXPECT_EQ(host->bytes(PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);
        EXPECT_EQ(gpu->bytes(PhysicalMemoryOwner::NativeReadAheadParentTerminal),0u);
        // This original 24GiB-free fixture cannot fit the full 48-layer source.
        // Qualification adds its exact BOM; it never advertises extra capacity.
        EXPECT_FALSE(planned.fits());
        EXPECT_THROW((void)planned.admit(),PhysicalMemoryCapacityExhausted);
        // Planning seals a BOM, not a live creation permit or final source mutation.
        EXPECT_EQ(original->memory()->phase(),PhysicalMemoryAuthorityPhase::Preparing);
        EXPECT_EQ(GPUDeviceContextPool::instance().retainExistingContextProvenance(device).originalBirthKey().get(),originalKey.get());
        auto ordinary=inputs;
        ordinary.front().ngram_lookup_schedule=ImmutableLookupSchedule::ConsumerScheduled;
        const auto serial=MemoryPlanner::plan(fixture.model,ordinary);
        for(const auto& resource:serial.physicalPlan().resources())
            EXPECT_EQ(resource.bytes(PhysicalMemoryOwner::NativeReadAheadFamily),0u);
        auto incompatible=inputs;incompatible.front().concurrent_workspace_owners=2;
        EXPECT_THROW((void)MemoryPlanner::plan(fixture.model,incompatible),std::invalid_argument);
        EXPECT_EQ(original->memory()->phase(),PhysicalMemoryAuthorityPhase::Preparing);
    }
}

/** @test Setup selection freezes once and survives mutable environment reloads and adapters. */
TEST(RankMemoryPlanInputs, ReadAheadSelectionSnapshotSurvivesDebugMutationAndAdapters)
{
    struct Restore {
        std::optional<ImmutableLookupSchedule> original;
        ~Restore(){mutableDebugEnv().execution.qfn_ngram_lookup_schedule=original;}
    } restore{mutableDebugEnv().execution.qfn_ngram_lookup_schedule};
    mutableDebugEnv().execution.qfn_ngram_lookup_schedule=ImmutableLookupSchedule::EarlyCurrentHostedCandidate;
    const auto frozen=RuntimeConfig::fromOrchestrationConfig(4096,1,"fp32","fp16");
    mutableDebugEnv().execution.qfn_ngram_lookup_schedule=ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate;
    const auto resident=RuntimeConfig::fromOrchestrationConfig(4096,1,"fp32","fp16");
    mutableDebugEnv().execution.qfn_ngram_lookup_schedule.reset();
    const auto baseline=RuntimeConfig::fromOrchestrationConfig(4096,1,"fp32","fp16");
    EXPECT_EQ(frozen.ngram_lookup_schedule,ImmutableLookupSchedule::EarlyCurrentHostedCandidate);
    EXPECT_EQ(resident.ngram_lookup_schedule,ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate);
    EXPECT_EQ(baseline.ngram_lookup_schedule,ImmutableLookupSchedule::ConsumerScheduled);
    EXPECT_EQ(RankOrchestrator::Config::fromRuntime(frozen).ngram_lookup_schedule,frozen.ngram_lookup_schedule);
    EXPECT_EQ(InferenceRunnerConfig::fromRuntime(frozen).ngram_lookup_schedule,frozen.ngram_lookup_schedule);
}

/** @test Original Rank TP coordinates stay authored while both continuation GPUs also own shared storage. */
TEST(RankMemoryPlanInputs, SharedCoordinatorTP2KeepsOriginalComputeAndStorageRoles)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    Fixture fixture(DeviceType::ROCm,2);
    fixture.model=source.metadata().memoryProfile();fixture.plan.last_layer=47;
    fixture.inventory.p2p_rocm_count=2;fixture.inventory.p2p_rocm.assign(4,false);
    fixture.plan.runtime.ngram_table_policy.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    fixture.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
        parseAdaptiveGPUCachePlacement("0:rocm:0/64/1048576,0:rocm:1/64/1048576@0:rocm:1");
    std::reverse(fixture.plan.local_tp_devices.begin(),fixture.plan.local_tp_devices.end());
    fixture.plan.primary_device=fixture.plan.local_tp_devices.front();
    const auto inputs=buildRankMemoryPlanInputs(fixture.request());
    ASSERT_EQ(inputs.size(),2u);
    for(size_t coordinate=0;coordinate<inputs.size();++coordinate){
        EXPECT_EQ(inputs[coordinate].device,fixture.plan.local_tp_devices[coordinate].toLocalDeviceId());
        EXPECT_EQ(inputs[coordinate].shard_index,int(coordinate));
        EXPECT_EQ(inputs[coordinate].ngram_tp_topology,ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal);
        ASSERT_TRUE(inputs[coordinate].ngram_execution_policy);
        EXPECT_EQ(inputs[coordinate].ngram_execution_policy->lookupIOConsumer(inputs[coordinate].device,2),DeviceId::rocm(1));
        EXPECT_FALSE(inputs[coordinate].ngram_gpu_peer_access); // Native library owns homogeneous peer selection.
        const auto& owners=inputs[coordinate].ngram_execution_policy->gpu_adaptive.placement.owners;
        ASSERT_EQ(owners.size(),2u);
        EXPECT_EQ(owners[coordinate].participant.device(),(DeviceId::rocm(int(coordinate))));
        EXPECT_TRUE(inputs[coordinate].ngram_reader_peer_access.empty());
    }
    auto missing=fixture;
    missing.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
        parseAdaptiveGPUCachePlacement("0:rocm:1/64/1048576@0:rocm:1");
    EXPECT_NO_THROW(buildRankMemoryPlanInputs(missing.request()));
    auto foreign=fixture;
    foreign.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
        parseAdaptiveGPUCachePlacement("0:rocm:2/64/1048576@0:rocm:3");
    EXPECT_THROW(buildRankMemoryPlanInputs(foreign.request()),std::invalid_argument);
    auto partial=fixture;partial.plan.first_layer=1;
    EXPECT_THROW(buildRankMemoryPlanInputs(partial.request()),std::invalid_argument);
    auto wrongBackend=fixture;
    wrongBackend.plan.local_tp_devices[0]=GlobalDeviceAddress::cuda(0,-1,"node");
    EXPECT_THROW(buildRankMemoryPlanInputs(wrongBackend.request()),std::invalid_argument);
    auto tooMany=fixture;
    tooMany.plan.local_tp_devices.push_back(GlobalDeviceAddress::rocm(2,-1,"node"));
    EXPECT_THROW(buildRankMemoryPlanInputs(tooMany.request()),std::invalid_argument);
    auto cuda=Fixture(DeviceType::CUDA,2);cuda.model=fixture.model;cuda.plan.last_layer=47;
    cuda.plan.runtime.ngram_table_policy.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    cuda.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
        parseAdaptiveGPUCachePlacement("0:cuda:2/64/1048576@0:cuda:1");
    EXPECT_THROW(buildRankMemoryPlanInputs(cuda.request()),std::invalid_argument);
}

/** @test CUDA readers retain their original NoPeer edge and a distinct ROCm storage-only owner. */
TEST(RankMemoryPlanInputs, SharedCoordinatorCUDATP2KeepsOppositeVendorStorageAndNoPeerReader)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    Fixture fixture(DeviceType::CUDA,2);
    fixture.model=source.metadata().memoryProfile();
    fixture.plan.last_layer=47;
    auto storage=fixture.inventory.gpus.front();
    storage.type=DeviceType::ROCm;
    storage.local_device_id=0;
    storage.uuid="storage-only-rocm-0";
    fixture.inventory.gpus.push_back(storage);
    fixture.inventory.p2p_cuda_count=2;
    fixture.inventory.p2p_cuda.assign(4,false);
    fixture.plan.runtime.ngram_table_policy.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
    fixture.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
        parseAdaptiveGPUCachePlacement("0:rocm:0/64/1048576@0:cuda:1");
    std::reverse(fixture.plan.local_tp_devices.begin(),fixture.plan.local_tp_devices.end());
    fixture.plan.primary_device=fixture.plan.local_tp_devices.front();
    const auto inputs=buildRankMemoryPlanInputs(fixture.request());
    ASSERT_EQ(inputs.size(),2u);
    for(std::size_t coordinate=0;coordinate<inputs.size();++coordinate){
        const auto& input=inputs[coordinate];
        EXPECT_EQ(input.device,fixture.plan.local_tp_devices[coordinate].toLocalDeviceId());
        EXPECT_EQ(input.shard_index,static_cast<int>(coordinate));
        EXPECT_EQ(input.ngram_tp_topology,ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal);
        EXPECT_FALSE(input.ngram_gpu_peer_access); // Cross-vendor storage has no same-vendor observation.
        EXPECT_TRUE(input.ngram_reader_peer_access.empty()); // No serialized captured-channel route for native readers.
        ASSERT_TRUE(input.ngram_execution_policy);
        EXPECT_EQ(input.ngram_execution_policy->lookupIOConsumer(input.device,2),DeviceId::cuda(1));
        EXPECT_NE(input.device,DeviceId::rocm(0));
    }
    auto partial=fixture;
    partial.inventory.p2p_cuda[1]=true;
    EXPECT_NO_THROW(buildRankMemoryPlanInputs(partial.request())); // Native CCL, including partial P2P coverage.
    auto computeOwner=fixture;
    computeOwner.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
        parseAdaptiveGPUCachePlacement("0:cuda:0/64/1048576@0:cuda:1");
    EXPECT_NO_THROW(buildRankMemoryPlanInputs(computeOwner.request()));
    auto unsupportedStorage=fixture;
    unsupportedStorage.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
        parseAdaptiveGPUCachePlacement("0:cuda:2/64/1048576@0:cuda:1");
    EXPECT_THROW(buildRankMemoryPlanInputs(unsupportedStorage.request()),std::invalid_argument);
    EXPECT_THROW({
        auto wrongRank=fixture;
        wrongRank.plan.runtime.ngram_table_policy.gpu_adaptive.placement=
            parseAdaptiveGPUCachePlacement("1:rocm:0/64/1048576@0:cuda:1");
        (void)buildRankMemoryPlanInputs(wrongRank.request());
    },std::invalid_argument);
    auto partialModel=fixture;
    partialModel.plan.first_layer=1;
    EXPECT_THROW(buildRankMemoryPlanInputs(partialModel.request()),std::invalid_argument);
}

#include "NativeWeightLoadStagingContract.inc"
#include "NativeWeightSourceBOMContract.inc"


/** @brief Actual Rank inputs preserve both continuation shards and native ownership independently of P2P coverage. */
TEST(RankMemoryPlanInputs, SharedAllContinuationOwnersUseOriginalNativeTP2AcrossPeerCoverage)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm})
        for (const auto coverage : {0u, 1u, 2u})
            for (const auto capacity : {0u, 64u}) {
                Fixture fixture(type, 2);
                fixture.model = source.metadata().memoryProfile();
                fixture.plan.last_layer = 47;
                auto& matrix = type == DeviceType::CUDA ? fixture.inventory.p2p_cuda : fixture.inventory.p2p_rocm;
                auto& count = type == DeviceType::CUDA ? fixture.inventory.p2p_cuda_count : fixture.inventory.p2p_rocm_count;
                count = 2; matrix.assign(4, false);
                matrix[1] = coverage > 0; matrix[2] = coverage > 1;
                const auto backend = type == DeviceType::CUDA ? "cuda" : "rocm";
                fixture.plan.runtime.ngram_table_policy.storage = ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
                fixture.plan.runtime.ngram_table_policy.gpu_adaptive.placement = parseAdaptiveGPUCachePlacement(
                    "0:" + std::string(backend) + ":0/" + std::to_string(capacity) + "/" + std::to_string(capacity ? 1048576 : 0) + ",0:" + backend +
                    ":1/" + std::to_string(capacity / 2) + "/" + std::to_string(capacity ? 524288 : 0) + "@0:" + backend + ":1");
                std::reverse(fixture.plan.local_tp_devices.begin(), fixture.plan.local_tp_devices.end());
                fixture.plan.primary_device = fixture.plan.local_tp_devices.front();
                const auto inputs = buildRankMemoryPlanInputs(fixture.request());
                ASSERT_EQ(inputs.size(), 2u);
                for (std::size_t coordinate = 0; coordinate < inputs.size(); ++coordinate) {
                    const auto& input = inputs[coordinate];
                    EXPECT_EQ(input.device, fixture.plan.local_tp_devices[coordinate].toLocalDeviceId());
                    EXPECT_EQ(input.shard_index, static_cast<int>(coordinate));
                    EXPECT_EQ(input.ngram_tp_topology, ImmutableRowTableTPTopology::HomogeneousFullMainRankLocal);
                    ASSERT_TRUE(input.ngram_execution_policy);
                    EXPECT_EQ(input.ngram_execution_policy->lookupIOConsumer(input.device, 2), (DeviceId{type, 1}));
                    EXPECT_FALSE(input.ngram_gpu_peer_access);
                    EXPECT_TRUE(input.ngram_reader_peer_access.empty());
                }
                auto incomplete = fixture;
                incomplete.plan.runtime.ngram_table_policy.gpu_adaptive.placement.owners.pop_back();
                EXPECT_NO_THROW(buildRankMemoryPlanInputs(incomplete.request()));
                auto partialModel = fixture;
                partialModel.plan.first_layer = 1;
                EXPECT_THROW(buildRankMemoryPlanInputs(partialModel.request()), std::invalid_argument);
            }
}

/** @test A declared third same-vendor owner fails at original planning before native materialization. */
TEST(RankMemoryPlanInputs, SharedOutsideNativeCohortOwnerRejectsAtAdmission)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto type : {DeviceType::CUDA, DeviceType::ROCm}) {
        Fixture fixture(type, 2);
        fixture.model = source.metadata().memoryProfile();
        fixture.plan.last_layer = 47;
        auto storage = fixture.inventory.gpus.front();
        storage.local_device_id = 2;
        storage.uuid = "outside-native-cohort";
        fixture.inventory.gpus.push_back(storage);
        const auto backend = type == DeviceType::CUDA ? "cuda" : "rocm";
        fixture.plan.runtime.ngram_table_policy.storage = ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
        fixture.plan.runtime.ngram_table_policy.gpu_adaptive.placement = parseAdaptiveGPUCachePlacement(
            "0:" + std::string(backend) + ":2/64/1048576@0:" + backend + ":0");
        try {
            (void)buildRankMemoryPlanInputs(fixture.request());
            FAIL() << "Outside-cohort native owner was admitted";
        } catch (const std::invalid_argument& error) {
            EXPECT_NE(std::string(error.what()).find("outside the original native communicator"), std::string::npos);
        }
    }
}

/** @test Owner subsets and mapped owners never change original native reader coordinates or cache intent. */
TEST(RankMemoryPlanInputs, SharedOwnerSubsetsPreserveOriginalNativeReadersAcrossBackends)
{
    test::Qwen4ExpPlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto type : {DeviceType::CUDA,DeviceType::ROCm})
        for (unsigned mask=1;mask<4;++mask)
            for (int coordinator=0;coordinator<2;++coordinator)
                for (const unsigned capacity : {0u,64u})
                    for (const bool mapped : {false,true}) {
                        SCOPED_TRACE(::testing::Message()<<int(type)<<":"<<mask<<":"<<coordinator<<":"<<capacity<<":"<<mapped);
                        Fixture fixture(type,2);fixture.model=source.metadata().memoryProfile();fixture.plan.last_layer=47;
                        const std::string backend=type==DeviceType::CUDA?"cuda":"rocm";
                        const std::string opposite=type==DeviceType::CUDA?"rocm":"cuda";
                        std::string owners;
                        for(unsigned owner=0;owner<2;++owner)if(mask&(1u<<owner)){
                            if(!owners.empty())owners+=",";
                            owners+="0:"+backend+":"+std::to_string(owner)+"/"+std::to_string(capacity)+"/"+std::to_string(capacity?1048576:0);
                        }
                        if(mapped){
                            auto storage=fixture.inventory.gpus.front();storage.type=type==DeviceType::CUDA?DeviceType::ROCm:DeviceType::CUDA;
                            storage.local_device_id=0;storage.uuid="opposite-original-owner";fixture.inventory.gpus.push_back(storage);
                            owners+=",0:"+opposite+":0/"+std::to_string(capacity)+"/"+std::to_string(capacity?1048576:0);
                        }
                        auto& policy=fixture.plan.runtime.ngram_table_policy;
                        policy.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
                        policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement(owners+"@0:"+backend+":"+std::to_string(coordinator));
                        const auto expected=policy.gpu_adaptive.placement;
                        std::reverse(fixture.plan.local_tp_devices.begin(),fixture.plan.local_tp_devices.end());
                        fixture.plan.primary_device=fixture.plan.local_tp_devices.front();
                        const auto inputs=buildRankMemoryPlanInputs(fixture.request());ASSERT_EQ(inputs.size(),2u);
                        for(std::size_t coordinate=0;coordinate<inputs.size();++coordinate){
                            EXPECT_EQ(inputs[coordinate].device,fixture.plan.local_tp_devices[coordinate].toLocalDeviceId());
                            EXPECT_EQ(inputs[coordinate].shard_index,int(coordinate));
                            ASSERT_TRUE(inputs[coordinate].ngram_execution_policy);
                            EXPECT_EQ(inputs[coordinate].ngram_execution_policy->gpu_adaptive.placement,expected);
                            EXPECT_EQ(inputs[coordinate].ngram_execution_policy->lookupIOConsumer(inputs[coordinate].device,2),(DeviceId{type,coordinator}));
                            EXPECT_FALSE(inputs[coordinate].ngram_gpu_peer_access);
                            EXPECT_TRUE(inputs[coordinate].ngram_reader_peer_access.empty());
                        }
                        auto foreignHost=fixture;foreignHost.plan.local_tp_devices[1].hostname="foreign";
                        EXPECT_THROW(buildRankMemoryPlanInputs(foreignHost.request()),std::invalid_argument);
                        auto foreignRank=fixture;foreignRank.plan.runtime.ngram_table_policy.gpu_adaptive.placement.owners.front().participant.world_rank=1;
                        EXPECT_THROW(buildRankMemoryPlanInputs(foreignRank.request()),std::invalid_argument);
                    }
}

/** @test Original TP4/8 owner subsets retain exact homogeneous readers and no unrelated mapped route. */
TEST(RankMemoryPlanInputs, AddressedSharedOwnersRetainOriginalReadersEveryDegree) {
    test::Qwen4ExpPlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(const auto type:{DeviceType::CUDA,DeviceType::ROCm})for(const int degree:{2,4,8})
        for(const int coordinator:{0,degree-1})for(const bool ownsSource:{false,true})for(const unsigned capacity:{0u,64u}) {
            Fixture fixture(type,degree);fixture.model=source.metadata().memoryProfile();fixture.plan.last_layer=47;
            const std::string backend=type==DeviceType::CUDA?"cuda":"rocm";
            const int owner=ownsSource?coordinator:(coordinator+1)%degree;
            auto& policy=fixture.plan.runtime.ngram_table_policy;policy.storage=ImmutableRowStoragePreference::AdaptiveVRAMRAMSSD;
            policy.gpu_adaptive.placement=parseAdaptiveGPUCachePlacement("0:"+backend+":"+std::to_string(owner)+"/"+
                std::to_string(capacity)+"/"+std::to_string(capacity?1048576:0)+"@0:"+backend+":"+std::to_string(coordinator));
            std::reverse(fixture.plan.local_tp_devices.begin(),fixture.plan.local_tp_devices.end());fixture.plan.primary_device=fixture.plan.local_tp_devices.front();
            const auto inputs=buildRankMemoryPlanInputs(fixture.request());ASSERT_EQ(inputs.size(),std::size_t(degree));
            for(std::size_t coordinate=0;coordinate<inputs.size();++coordinate){
                EXPECT_EQ(inputs[coordinate].device,fixture.plan.local_tp_devices[coordinate].toLocalDeviceId());
                EXPECT_EQ(inputs[coordinate].shard_index,int(coordinate));ASSERT_TRUE(inputs[coordinate].ngram_execution_policy);
                EXPECT_EQ(inputs[coordinate].ngram_execution_policy->gpu_adaptive.placement,policy.gpu_adaptive.placement);
            }
        }
}

/** @test Original MPI membership is independent of native width, permutation and physical NUMA labels. */
TEST(OriginalRankLocalNativeCohort, PreservesOriginalRanksPermutationsAndWidths)
{
    for (const int world : {1, 2, 5})
        for (const int width : {1, 2, 3, 4, 8, 17})
        {
            SCOPED_TRACE(::testing::Message() << "world=" << world << " width=" << width);
            ClusterInventory inventory;
            inventory.world_size = world;
            inventory.ranks.resize(world);
            for (int rank = 0; rank < world; ++rank)
            {
                inventory.ranks[rank].rank = rank;
                inventory.ranks[rank].cpu.numa_node = 31 + rank * 7;
            }
            const int rank = world - 1;
            auto &owner = inventory.ranks[rank];
            std::vector<DeviceId> devices;
            for (int index = 0; index < width; ++index)
            {
                DeviceInfo gpu;
                gpu.type = DeviceType::ROCm;
                gpu.local_device_id = 4 + index * 3;
                gpu.uuid = "original-uuid-" + std::to_string(index);
                gpu.numa_node = owner.cpu.numa_node;
                owner.gpus.push_back(gpu);
                devices.push_back(DeviceId::rocm(gpu.local_device_id));
            }
            std::reverse(devices.begin(), devices.end());
            const auto original = devices;
            const auto *borrow = &test::requireOriginalRankLocalNativeCohort(
                inventory, rank, world, owner.cpu.numa_node, devices);
            EXPECT_EQ(borrow, &owner);
            EXPECT_EQ(borrow->rank, rank);
            EXPECT_EQ(inventory.ranks.size(), static_cast<std::size_t>(world));
            EXPECT_EQ(devices, original);
        }
}

/** @test Visibility, duplicate physical aliases and changed CPU bindings cannot authorize native work. */
TEST(OriginalRankLocalNativeCohort, RejectsForeignRankNUMAOrdinalUUIDAndIncompleteMembership)
{
    ClusterInventory inventory;
    inventory.world_size = 2;
    inventory.ranks.resize(2);
    inventory.ranks[0].rank = 0;
    inventory.ranks[0].cpu.numa_node = 43;
    auto &owner = inventory.ranks[1];
    owner.rank = 1;
    owner.cpu.numa_node = 19;
    for (int index = 0; index < 2; ++index)
    {
        DeviceInfo gpu;
        gpu.type = DeviceType::CUDA;
        gpu.local_device_id = index * 7 + 3;
        gpu.numa_node = 19;
        gpu.uuid = "original-cuda-" + std::to_string(index);
        owner.gpus.push_back(gpu);
    }
    const std::array devices{DeviceId::cuda(10), DeviceId::cuda(3)};
    const auto admit = [&](const ClusterInventory &candidate, int rank = 1, int world = 2, int numa = 19) {
        return &test::requireOriginalRankLocalNativeCohort(candidate, rank, world, numa, devices);
    };
    ASSERT_NO_THROW((void)admit(inventory));
    EXPECT_THROW((void)admit(inventory, -1), std::invalid_argument);
    EXPECT_THROW((void)admit(inventory, 2), std::invalid_argument);
    EXPECT_THROW((void)admit(inventory, 1, 1), std::invalid_argument);
    EXPECT_THROW((void)admit(inventory, 1, 0), std::invalid_argument);
    EXPECT_THROW((void)admit(inventory, 1, 2, -1), std::invalid_argument);
    EXPECT_THROW((void)admit(inventory, 1, 2, 43), std::invalid_argument);
    auto changed = inventory;
    changed.ranks[1].rank = 0;
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    changed = inventory; changed.ranks[0].rank = 1;
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    changed = inventory; changed.ranks.pop_back();
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    changed = inventory; changed.ranks[1].cpu.numa_node = -1;
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    changed = inventory; changed.ranks[1].cpu.type = DeviceType::CUDA;
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    for (const int numa : {-1, 43})
    {
        changed = inventory; changed.ranks[1].gpus[0].numa_node = numa;
        EXPECT_THROW((void)admit(changed), std::invalid_argument);
    }
    changed = inventory; changed.ranks[1].gpus[0].uuid.clear();
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    changed = inventory; changed.ranks[1].gpus[1].uuid = changed.ranks[1].gpus[0].uuid;
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    changed = inventory; changed.ranks[1].gpus.push_back(changed.ranks[1].gpus[0]);
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    changed = inventory; changed.ranks[1].gpus[0].local_device_id = 99;
    EXPECT_THROW((void)admit(changed), std::invalid_argument);
    const std::array duplicates{devices[0], devices[0]};
    EXPECT_THROW((void)test::requireOriginalRankLocalNativeCohort(inventory, 1, 2, 19, duplicates), std::invalid_argument);
    const std::array host{DeviceId::cpu()};
    EXPECT_THROW((void)test::requireOriginalRankLocalNativeCohort(inventory, 1, 2, 19, host), std::invalid_argument);
    EXPECT_THROW((void)test::requireOriginalRankLocalNativeCohort(inventory, 1, 2, 19, {}), std::invalid_argument);
}
