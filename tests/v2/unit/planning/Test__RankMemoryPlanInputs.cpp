/**
 * @file Test__RankMemoryPlanInputs.cpp
 * @brief Device-free proof of the shared runtime/automatic-planner BOM boundary.
 *
 * Sweep backend and TP degree while retaining the exact graph, shifted MTP KV,
 * mirrored weights, prefix and fresh-load owners. No model loading or driver
 * calls occur; these are policy/geometry proofs before physical admission.
 */
#include "planning/RankMemoryPlanInputs.h"
#include "planning/PlanningModelMetadata.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "kernels/attention/AttentionHeadMapping.h"
#include "../../utils/Qwen4ExpPlanningGGUFFixture.h"
#include "execution/local_execution/engine/ForwardGraphTypes.h"
#include "config/OrchestrationConfig.h"
#include "../../utils/CPUExecutionTestGeometry.h"
#include <gtest/gtest.h>
#include <limits>
#include <algorithm>
#include <vector>

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
            EXPECT_EQ(cfg.associated_host_memory.has_value(), source && leader);
        }
    }
}

/** @test Native publication follows retained graph capacity and the actual local communicator. */
TEST(RankMemoryPlanInputs, NativeVocabularyPublicationFollowsResolvedOwnership)
{
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    for (const int degree : {1, 2, 4})
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
            retainsMTPGraphCapacity(mtp) && head == MTPTerminalHeadPolicy::VocabularySharded;
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
