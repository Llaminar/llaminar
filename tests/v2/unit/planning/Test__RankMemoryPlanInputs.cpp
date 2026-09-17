/**
 * @file Test__RankMemoryPlanInputs.cpp
 * @brief Device-free proof of the shared runtime/automatic-planner BOM boundary.
 *
 * Sweep backend and TP degree while retaining the exact graph, shifted MTP KV,
 * mirrored weights, prefix and fresh-load owners. No model loading or driver
 * calls occur; these are policy/geometry proofs before physical admission.
 */
#include "planning/RankMemoryPlanInputs.h"
#include "config/OrchestrationConfig.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

namespace
{
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
                    const auto expected = resolveCapturedServingGraphMemoryInventory({32, 64, 128, 256}, mtp);
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
                cfg.captured_serving_graphs.prefill_bucket_rows.size() +
                cfg.captured_serving_graphs.fixed_executable_count +
                cfg.captured_serving_graphs.mtp_graph_owners.sidecarGraphSlots());
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

TEST(RankMemoryPlanInputs, PipelineTensorDomainsKeepTheirExactLayerIntervals)
{
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        Fixture fixture(backend, 4);
        const auto devices = fixture.plan.local_tp_devices;
        fixture.plan.local_tp_devices.clear();
        fixture.plan.local_pp_devices = {devices[0], devices[2]};
        fixture.plan.local_pp_layer_boundaries = {0, 2, 4};
        fixture.plan.local_pp_stage_tp_info = {
            {.devices = {devices[0], devices[1]}},
            {.devices = {devices[2], devices[3]}}};
        const auto configs = buildRankMemoryPlanInputs(fixture.request());
        ASSERT_EQ(configs.size(), 4u);
        for (int participant = 0; participant < 4; ++participant)
        {
            const auto &cfg = configs[participant];
            EXPECT_EQ(cfg.first_layer, (participant / 2) * 2);
            EXPECT_EQ(cfg.last_layer, (participant / 2) * 2 + 1);
            EXPECT_EQ(cfg.owns_embedding, participant < 2);
            EXPECT_EQ(cfg.shard_index, participant % 2);
            EXPECT_FALSE(cfg.local_pipeline_backend.has_value());
            ASSERT_TRUE(cfg.tensor_parallel_assignment);
            EXPECT_EQ(cfg.tensor_parallel_assignment->device, DeviceId(backend, participant));
        }
        fixture.plan.local_pp_layer_boundaries.pop_back();
        EXPECT_THROW(buildRankMemoryPlanInputs(fixture.request()), std::invalid_argument);
    }
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

TEST(RankMemoryPlanInputs, ImpossibleGQAShardCannotBecomeAnAdmissibleCandidate)
{
    Fixture fixture(DeviceType::CUDA, 8);
    fixture.model.n_heads = 8; // Replicated two-head GQA needs at least two Q heads per shard.
    EXPECT_THROW(buildRankMemoryPlanInputs(fixture.request()), std::invalid_argument);
}
