/**
 * @file Test__MoELocalExpertStage_Params.cpp
 * @brief Device-free construction and ownership tests for local MoE stages.
 *
 * These tests exercise only typed parameter and compact-buffer ownership
 * contracts.  GPU placement and execution are intentionally left to explicit
 * integration suites, keeping the unit target fast and accelerator-free.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 */

#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "utils/CPUSerialArenaTestAdmission.h"
#include "utils/CPUPerLayerPacketTestAdmission.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"
#include "kernels/cpu/gemm/CPUProjectionWorkspaceContract.h"
#include "kernels/cpu/CPUInvocationWorkspace.h"
#include "utils/CPUExecutionTestGeometry.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <limits>
#include "tensors/TensorClasses.h"

namespace llaminar2::test
{
    /** @test Empty CPU banks declare the exact named scratch for future arrivals. */
    TEST(Test__MoELocalExpertStage_Params, EmptyCPUResidencyDeclaresSourceWorkspace)
    {
        for (const auto format : {TensorType::FP32, TensorType::FP16, TensorType::BF16,
                                 TensorType::Q8_0, TensorType::Q6_K, TensorType::IQ2_S})
        for (const uint32_t row_tile : {2u, 4u})
        for (const int workers : {1, 3, 28})
        {
            SCOPED_TRACE(tensorTypeName(format));
            auto geometry = kSyntheticCPUExecutionGeometry;
            geometry.maximum_native_row_tile = row_tile;
            MoELocalExpertStage::Params params;
            params.device_id = DeviceId::cpu();
            params.num_experts = 8;
            params.top_k = 3;
            params.d_model = 4096;
            params.expert_intermediate = 256;
            params.expert_weight_resolution_policy = MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
            params.cpu_workspace_source = MoELocalExpertStage::CPUExpertWorkspaceSource{
                .gate = format, .up = TensorType::BF16, .down = format,
                .workers = workers, .execution = geometry};
            MoELocalExpertStage stage(test::admitCPUPerLayerPacketForTest(params));
            auto expected = cpuSwiGLUWorkspaceRequirements(15 * 3, 256);
            for (const auto [n, k] : {std::pair{256, 4096}, std::pair{4096, 256}})
                expected.merge(CPUProjectionWorkspaceContract::sourceNative(format,
                    {.rows = 15 * 3, .n = n, .k = k, .workers = workers, .execution = geometry,
                     .numerical_policy = CPUProjectionNumericalPolicy::GPUAlignedExpert}));
            const auto actual = stage.getWorkspaceRequirements(15);
            EXPECT_EQ(actual.cpu_grouped_moe_storage, nullptr)
                << "Source-only queries declare named scratch without inventing an executor backing";
            ASSERT_EQ(actual.buffers.size(), expected.buffers.size());
            for (const auto &buffer : expected.buffers)
            {
                const auto *found = actual.find(buffer.name);
                ASSERT_NE(found, nullptr);
                EXPECT_EQ(found->size_bytes, buffer.size_bytes);
            }
        }
    }

    /** @test Missing source metadata and unrepresentable route geometry fail during setup. */
    TEST(Test__MoELocalExpertStage_Params, EmptyCPUWorkspaceRejectsMissingSourceOrOverflow)
    {
        MoELocalExpertStage::Params params;
        params.device_id = DeviceId::cpu();
        params.num_experts = 8;
        params.top_k = 3;
        params.d_model = 4096;
        params.expert_intermediate = 256;
        params.expert_weight_resolution_policy = MoELocalExpertStage::ExpertWeightResolutionPolicy::RegistryOnly;
        MoELocalExpertStage stage(test::admitCPUPerLayerPacketForTest(params));
        EXPECT_THROW(stage.getWorkspaceRequirements(1), std::logic_error);
        EXPECT_THROW(stage.getWorkspaceRequirements(std::numeric_limits<int>::max()), std::overflow_error);
    }

    namespace
    {
        template <typename, typename = void>
        struct has_runtime : std::false_type
        {
        };
        template <typename T>
        struct has_runtime<T, std::void_t<decltype(std::declval<T &>().runtime)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_moe_runtime_table : std::false_type
        {
        };
        template <typename T>
        struct has_moe_runtime_table<T, std::void_t<decltype(std::declval<T &>().moe_runtime_table)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_domain_runtime : std::false_type
        {
        };
        template <typename T>
        struct has_domain_runtime<T, std::void_t<decltype(std::declval<T &>().domain_runtime)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_runtime_service : std::false_type
        {
        };
        template <typename T>
        struct has_runtime_service<T, std::void_t<decltype(std::declval<T &>().runtime_service)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_runner : std::false_type
        {
        };
        template <typename T>
        struct has_runner<T, std::void_t<decltype(std::declval<T &>().runner)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_role_runner : std::false_type
        {
        };
        template <typename T>
        struct has_role_runner<T, std::void_t<decltype(std::declval<T &>().role_runner)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_participants : std::false_type
        {
        };
        template <typename T>
        struct has_participants<T, std::void_t<decltype(std::declval<T &>().participants)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_prepared_participants : std::false_type
        {
        };
        template <typename T>
        struct has_prepared_participants<T, std::void_t<decltype(std::declval<T &>().prepared_participants)>> : std::true_type
        {
        };

        template <typename, typename = void>
        struct has_peer_participants : std::false_type
        {
        };
        template <typename T>
        struct has_peer_participants<T, std::void_t<decltype(std::declval<T &>().peer_participants)>> : std::true_type
        {
        };

    } // namespace

    TEST(Test__MoELocalExpertStage_Params, ConstructibleWithOnlyRuntimeTableHookAndNoRunnerFields)
    {
        using Params = MoELocalExpertStage::Params;

        static_assert(StageParamsRequired<Params>);
        EXPECT_TRUE(std::is_default_constructible_v<Params>);
        EXPECT_TRUE((std::is_constructible_v<MoELocalExpertStage, Params>));

        EXPECT_FALSE(has_runtime<Params>::value);
        EXPECT_TRUE(has_moe_runtime_table<Params>::value);
        EXPECT_FALSE(has_domain_runtime<Params>::value);
        EXPECT_FALSE(has_runtime_service<Params>::value);
        EXPECT_FALSE(has_runner<Params>::value);
        EXPECT_FALSE(has_role_runner<Params>::value);
        EXPECT_FALSE(has_participants<Params>::value);
        EXPECT_FALSE(has_prepared_participants<Params>::value);
        EXPECT_FALSE(has_peer_participants<Params>::value);
    }

    /**
     * @brief A serial family binds stable shared tensors and refuses growth.
     *
     * This is the device-free half of the graph-family storage proof.  The
     * Qwen graph-lowering test separately proves that production construction
     * supplies one arena to every eligible variant.
     */
    TEST(Test__MoELocalExpertStage_Params,
         SerialCompactBufferArenaBindsStableTensorsAndRejectsOutOfPlanRows)
    {
        constexpr int kDModel = 4;
        constexpr int kTopK = 2;
        constexpr size_t kRowCapacity = 4;
        constexpr size_t kEntryCapacity = kRowCapacity * kTopK;

        auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
            admitCPUSerialArenaForTest(MoELocalExpertSerialBufferArena::Config{
                .device_id = DeviceId::cpu(),
                .row_capacity = kRowCapacity,
                .row_capacity_buckets = {2},
                .d_model = kDModel,
                .routing_top_k = kTopK,
                .cpu_grouped_scratch_storage =
                    MoELocalExpertSerialBufferArena::
                        CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
                .num_experts = 2,
                .expert_intermediate = 8,
                .logical_participant_id = 3,
                .debug_name = "unit.serial_compact"}));
        ASSERT_NE(arena, nullptr);
        ASSERT_TRUE(arena->logicalParticipantId().has_value());
        EXPECT_EQ(*arena->logicalParticipantId(), 3);
        EXPECT_TRUE(arena->supports(
            DeviceId::cpu(), kRowCapacity, kDModel, kTopK));
        EXPECT_FALSE(arena->supports(
            DeviceId::cpu(), kRowCapacity + 1u, kDModel, kTopK));
        ASSERT_EQ(arena->familyCount(), 2u);
        ASSERT_NE(arena->smallestFamilySupporting(1), nullptr);
        EXPECT_EQ(arena->smallestFamilySupporting(1)->row_capacity, 2u);
        ASSERT_NE(arena->smallestFamilySupporting(3), nullptr);
        EXPECT_EQ(
            arena->smallestFamilySupporting(3)->row_capacity,
            kRowCapacity);
        EXPECT_EQ(arena->smallestFamilySupporting(kRowCapacity + 1u), nullptr);
        ASSERT_NE(arena->cpuGroupedWorkspace(), nullptr);
        EXPECT_TRUE(arena->cpuGroupedWorkspace()->supports(
            static_cast<int>(kRowCapacity),
            kTopK,
            kDModel,
            /*expert_intermediate=*/8,
            /*num_experts=*/2));
        {
            auto lease = arena->cpuGroupedWorkspace()->acquire(
                1, 1, kDModel, 8, 2, /*layer_idx=*/0);
            EXPECT_THROW(
                (void)arena->cpuGroupedWorkspace()->acquire(
                    1, 1, kDModel, 8, 2, /*layer_idx=*/1),
                std::logic_error)
                << "A serial workspace must reject overlapping graph roles";
        }
        EXPECT_NO_THROW(
            (void)arena->cpuGroupedWorkspace()->acquire(
                1, 1, kDModel, 8, 2, /*layer_idx=*/1));

        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(
            /*max_rows=*/4,
            /*max_entries=*/kEntryCapacity,
            kDModel,
            kTopK,
            DeviceId::cpu());
        auto input = workspace.localExpertInput(/*layer_idx=*/0, /*tier_idx=*/0);
        auto output = workspace.localExpertOutput(/*layer_idx=*/0, /*tier_idx=*/0);

        MoELocalExpertStage::Params params;
        params.device_id = DeviceId::cpu();
        params.input_rows = &input;
        params.output_rows = &output;
        params.num_experts = 2;
        params.top_k = kTopK;
        params.d_model = kDModel;
        params.expert_intermediate = 8;
        params.layer_idx = 0;
        params.runtime_participant_index = 3;
        params.serial_compact_buffer_arena = arena;

        MoELocalExpertStage first(test::admitCPUPerLayerPacketForTest(params));
        MoELocalExpertStage second(test::admitCPUPerLayerPacketForTest(params));
        EXPECT_EQ(first.compactHiddenTensorForDiagnostics(), arena->hidden().get());
        EXPECT_EQ(second.compactHiddenTensorForDiagnostics(), arena->hidden().get());
        EXPECT_EQ(first.compactHiddenTensorForDiagnostics(),
                  second.compactHiddenTensorForDiagnostics());
        EXPECT_TRUE(first.preparePersistentBuffers());
        EXPECT_TRUE(second.preparePersistentBuffers());

        MoEOverlayCollectiveWorkspace small_workspace;
        small_workspace.ensureCapacity(
            /*max_rows=*/1,
            /*max_entries=*/2,
            kDModel,
            kTopK,
            DeviceId::cpu());
        auto small_input =
            small_workspace.localExpertInput(/*layer_idx=*/0, /*tier_idx=*/0);
        auto small_output =
            small_workspace.localExpertOutput(/*layer_idx=*/0, /*tier_idx=*/0);
        auto small_params = params;
        small_params.input_rows = &small_input;
        small_params.output_rows = &small_output;
        MoELocalExpertStage small_stage(test::admitCPUPerLayerPacketForTest(small_params));
        EXPECT_EQ(small_stage.compactRowCapacityForDiagnostics(), 2u);
        EXPECT_EQ(
            small_stage.compactHiddenTensorForDiagnostics(),
            arena->smallestFamilySupporting(2)->hidden.get());
        EXPECT_NE(
            small_stage.compactHiddenTensorForDiagnostics(),
            first.compactHiddenTensorForDiagnostics());

        params.runtime_participant_index = 2;
        EXPECT_THROW(
            (void)MoELocalExpertStage{test::admitCPUPerLayerPacketForTest(params)},
            std::runtime_error)
            << "A serial arena must not be rebound to another logical participant";
        params.runtime_participant_index = 3;

        auto undersized = std::make_shared<MoELocalExpertSerialBufferArena>(
            admitCPUSerialArenaForTest(MoELocalExpertSerialBufferArena::Config{
                .device_id = DeviceId::cpu(),
                .row_capacity = kRowCapacity - 1u,
                .d_model = kDModel,
                .routing_top_k = kTopK,
                .logical_participant_id = 3,
                .debug_name = "unit.undersized_serial_compact"}));
        params.serial_compact_buffer_arena = std::move(undersized);
        EXPECT_THROW(
            (void)MoELocalExpertStage{test::admitCPUPerLayerPacketForTest(params)},
            std::runtime_error);

        EXPECT_THROW(
            ([&]
             {
                 (void)MoELocalExpertSerialBufferArena{
                     MoELocalExpertSerialBufferArena::Config{
                         .device_id = DeviceId::cpu(),
                         .row_capacity = kRowCapacity,
                         .d_model = kDModel,
                         .routing_top_k = kTopK,
                         .logical_participant_id = -1,
                         .debug_name = "unit.invalid_serial_participant"}};
             }()),
            std::invalid_argument);

        EXPECT_THROW(
            ([&]
             {
                 (void)MoELocalExpertSerialBufferArena{
                     MoELocalExpertSerialBufferArena::Config{
                         .device_id = DeviceId::cpu(),
                         .row_capacity = kRowCapacity,
                         .row_capacity_buckets = {kRowCapacity + 1u},
                         .d_model = kDModel,
                         .routing_top_k = kTopK,
                         .logical_participant_id = 3,
                         .debug_name = "unit.invalid_serial_bucket"}};
             }()),
            std::invalid_argument);
    }

    TEST(Test__MoELocalExpertStage_Params,
         PowerOfTwoBucketsCoverSparseDecodeWithoutLosingExactUpperBound)
    {
        EXPECT_TRUE(
            MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(0)
                .empty());
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(1),
            (std::vector<size_t>{1}));
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(8),
            (std::vector<size_t>{1, 2, 4, 8}));
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(10),
            (std::vector<size_t>{1, 2, 4, 8, 10}));
    }

    /**
     * @brief Route-width selection is total and never under-admits live work.
     */
    TEST(Test__MoELocalExpertStage_Params,
         RouteWidthBucketsCoverEverySupportedTopKWithoutChangingOrder)
    {
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::routeWidthBucketFor(0, 8), 0);
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::routeWidthBucketFor(9, 8), 0);
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::routeWidthBucketFor(1, 8), 1);
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::routeWidthBucketFor(2, 8), 2);
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::routeWidthBucketFor(3, 8), 4);
        EXPECT_EQ(
            MoELocalExpertSerialBufferArena::routeWidthBucketFor(5, 8), 8);

        for (int model_top_k = 1; model_top_k <= 16; ++model_top_k)
        {
            int previous_bucket = 0;
            for (int required = 1; required <= model_top_k; ++required)
            {
                const int bucket =
                    MoELocalExpertSerialBufferArena::routeWidthBucketFor(
                        required, model_top_k);
                EXPECT_GE(bucket, required);
                EXPECT_LE(bucket, model_top_k);
                EXPECT_GE(bucket, previous_bucket);
                previous_bucket = bucket;
            }
            EXPECT_EQ(previous_bucket, model_top_k)
                << "The exact non-power-of-two top-k must remain a family.";
        }
    }

    /** @test Odd-shaped declared roots, grouped storage and retained views have one exact issuer. */
    TEST(Test__MoELocalExpertStage_Params, SerialDeclaredRootsMatchPhysicalClaimsAndAliasLifetime)
    {
        using Arena = MoELocalExpertSerialBufferArena;
        const DeviceId device{DeviceType::CPU, 1};
        auto config = admitCPUSerialArenaForTest(Arena::Config{
            .device_id = device, .row_capacity = 7, .row_capacity_buckets = {3,1,3},
            .d_model = 5, .routing_top_k = 2,
            .cpu_canonical_route_storage = Arena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
            .cpu_grouped_scratch_storage = Arena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
            .num_experts = 4, .expert_intermediate = 11});
        const auto rounded = [](size_t elements) { return AlignedVector<float>::requiredAllocationBytes(elements); };
        const auto q8 = [](size_t elements) { return AlignedVector<Q8_1Block>::requiredAllocationBytes(elements); };
        size_t activation = rounded(7 * 2 * 5);
        for (const size_t rows : {1u,3u,7u})
            activation += 2 * rounded(rows * 5) + 2 * rounded(rows * 2);
        const size_t workspace = rounded(7 * 5) + 2 * rounded(14 * 11) +
            rounded(14 * 5) + rounded(14 * std::max(5, 11)) + 2 * q8(14) + q8(7);
        EXPECT_EQ(config.cpu_storage_layout->physicalBytes(), activation);
        EXPECT_EQ(config.cpu_storage_layout->workspaceBytes(), workspace);
        auto live = [&](PhysicalMemoryOwner owner)
        {
            size_t bytes = 0;
            for (const auto &line : config.physical_memory_authority->rankAttestation())
                if (line.owner == owner) bytes += line.materialized_new_bytes;
            return bytes;
        };
        for (int replay = 0; replay < 3; ++replay)
        {
            auto arena = std::make_shared<Arena>(config);
            EXPECT_EQ(arena->allocationBytes(), activation + workspace);
            EXPECT_EQ(live(PhysicalMemoryOwner::ActivationArena), activation);
            EXPECT_EQ(live(PhysicalMemoryOwner::ExecutionWorkspace), workspace);
            ASSERT_EQ(arena->familyCount(), 3u);
            auto view = std::dynamic_pointer_cast<FP32Tensor>(
                arena->smallestFamilySupporting(3)->hidden->create_view({1,5}));
            auto canonical = arena->cpuCanonicalRoutes();
            auto grouped = arena->cpuGroupedWorkspace();
            ASSERT_TRUE(view); ASSERT_TRUE(canonical); ASSERT_TRUE(grouped);
            view->mutable_data()[0] = 7.0f;
            arena.reset();
            EXPECT_EQ(view->data()[0], 7.0f);
            EXPECT_EQ(live(PhysicalMemoryOwner::ActivationArena), rounded(3 * 5) + rounded(7 * 2 * 5));
            EXPECT_EQ(live(PhysicalMemoryOwner::ExecutionWorkspace), workspace);
            canonical.reset(); grouped.reset();
            EXPECT_EQ(live(PhysicalMemoryOwner::ActivationArena), rounded(3 * 5));
            EXPECT_EQ(live(PhysicalMemoryOwner::ExecutionWorkspace), 0u);
            view.reset();
            EXPECT_EQ(live(PhysicalMemoryOwner::ActivationArena), 0u);
        }
    }

    /** @test Missing/stale declarations and partial capacity failures cannot leak root claims. */
    TEST(Test__MoELocalExpertStage_Params, SerialDeclaredRootsRejectMissingStaleAndInsufficientAdmission)
    {
        using Arena = MoELocalExpertSerialBufferArena;
        Arena::Config raw{.device_id = DeviceId::cpu(), .row_capacity = 7,
            .row_capacity_buckets = {1,3}, .d_model = 5, .routing_top_k = 2};
        EXPECT_THROW((void)Arena(raw), std::invalid_argument);
        auto config = admitCPUSerialArenaForTest(raw);
        auto stale = config;
        stale.row_capacity_buckets = {1,2,3};
        EXPECT_THROW((void)Arena(stale), std::invalid_argument);
        for (const auto &line : config.physical_memory_authority->rankAttestation())
            EXPECT_EQ(line.materialized_new_bytes, 0u);
        const auto expected = config.cpu_storage_layout->physicalBytes();
        PhysicalMemoryPlanBuilder plan;
        plan.add({0, DeviceId::cpu(), expected, expected}, PhysicalMemoryOwner::ActivationArena, expected - 1);
        config.physical_memory_authority = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()), 0);
        for (int replay = 0; replay < 3; ++replay)
        {
            EXPECT_THROW((void)Arena(config), std::logic_error);
            for (const auto &line : config.physical_memory_authority->rankAttestation())
                EXPECT_EQ(line.materialized_new_bytes, 0u);
        }
    }

    /** @test Every admitted controller/endpoint owns exclusive roots; foreign workers fail closed. */
    TEST(Test__MoELocalExpertStage_Params, SerialEnvelopeOwnsEveryControllerAndRejectsUnlistedWorkers)
    {
        using Arena = MoELocalExpertSerialBufferArena;
        const DeviceId host{DeviceType::CPU, 1};
        auto admission = std::make_shared<CPUSerialExpertStorageAdmission>();
        admission->allocator = {0, host};
        admission->participant_ids = {2,7};
        admission->controller_devices = {DeviceId::cpu(), host};
        std::sort(admission->controller_devices.begin(), admission->controller_devices.end());
        admission->envelope = std::make_shared<const CPUSerialExpertStorageEnvelope>(
            host, 7, 3, 5, 2, std::vector<int>{1,3,7}, 1, 8, true,
            CPUSerialExpertFamilyPolicy::LocalPowerOfTwo, 11, 4);
        std::vector<std::shared_ptr<const CPUSerialExpertStorageAdmission>> declarations{admission};
        size_t activation = 0, workspace = 0;
        for (const auto controller : admission->controller_devices)
        {
            const auto layout = admission->envelope->forController(controller);
            activation += layout->physicalBytes() * admission->participant_ids.size();
            workspace += layout->workspaceBytes() * admission->participant_ids.size();
        }
        EXPECT_EQ(admission->physicalBytes(), activation);
        PhysicalMemoryPlanBuilder plan;
        const PhysicalMemoryResource resource{0,host,activation+workspace,activation+workspace};
        plan.add(resource, PhysicalMemoryOwner::ActivationArena, activation);
        plan.add(resource, PhysicalMemoryOwner::ExecutionWorkspace, workspace);
        const auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        std::vector<std::shared_ptr<Arena>> arenas;
        std::vector<const void *> addresses;
        for (const auto controller : admission->controller_devices)
        for (const int participant : admission->participant_ids)
        {
            EXPECT_EQ(&requireCPUSerialExpertStorageAdmission(declarations,{0,host},participant,controller), admission.get());
            auto layout = admission->envelope->forController(controller);
            std::vector<size_t> rows;
            for (const auto &family : layout->families()) rows.push_back(family.rows);
            auto arena = std::make_shared<Arena>(Arena::Config{
                .device_id=host,.row_capacity=layout->maximumRows(),.row_capacity_buckets=rows,
                .d_model=5,.routing_top_k=2,
                .cpu_canonical_route_storage=Arena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum,
                .cpu_grouped_scratch_storage=Arena::CPUGroupedScratchStoragePolicy::RetainSerialMaximum,
                .num_experts=4,.expert_intermediate=11,.logical_participant_id=participant,
                .cpu_storage_layout=layout,.physical_memory_authority=memory});
            addresses.push_back(arena->families().front().hidden->raw_data());
            arenas.push_back(std::move(arena));
        }
        for (size_t first = 0; first < addresses.size(); ++first)
            for (size_t second = first + 1; second < addresses.size(); ++second)
                EXPECT_NE(addresses[first], addresses[second]);
        for (const auto &line : memory->rankAttestation())
        {
            if(line.owner==PhysicalMemoryOwner::ActivationArena) EXPECT_EQ(line.materialized_new_bytes,activation);
            if(line.owner==PhysicalMemoryOwner::ExecutionWorkspace) EXPECT_EQ(line.materialized_new_bytes,workspace);
        }
        EXPECT_THROW(requireCPUSerialExpertStorageAdmission(declarations,{0,host},3,host),std::invalid_argument);
        EXPECT_THROW(requireCPUSerialExpertStorageAdmission(declarations,{0,host},2,DeviceId::cuda(1)),std::invalid_argument);
        auto duplicated = declarations; duplicated.push_back(admission);
        EXPECT_THROW(requireCPUSerialExpertStorageAdmission(duplicated,{0,host},2,host),std::invalid_argument);
        {
            auto first = arenas[0]->cpuGroupedWorkspace()->acquire(1,1,5,11,4,0);
            auto independent = arenas[1]->cpuGroupedWorkspace()->acquire(1,1,5,11,4,0);
            EXPECT_THROW((void)arenas[0]->cpuGroupedWorkspace()->acquire(1,1,5,11,4,0),std::logic_error);
        }
        arenas.clear();
        for(const auto &line:memory->rankAttestation()) EXPECT_EQ(line.materialized_new_bytes,0u);
    }

    /** @test Only authored bounded schedules can project new runtime family roots. */
    TEST(Test__MoELocalExpertStage_Params, SerialEnvelopeProjectsOnlyAuthoredSchedulesAndPricesBothCanonicalBanks)
    {
        const DeviceId host{DeviceType::CPU,1};
        const std::vector<int> inventory{1,3,7,15,31};
        for(const auto policy:{CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,CPUSerialExpertFamilyPolicy::PublishedSchedule})
        {
            CPUSerialExpertStorageEnvelope envelope(host,31,5,5,2,inventory,1,8,true,policy,11,4);
            for(const int prefill:{31,15,7})
            {
                const auto schedule=retainedRawPrefillGraphBucketLadder(inventory,prefill,1,8);
                const auto layout=envelope.project(prefill,5,schedule);
                EXPECT_EQ(layout->maximumRows(),static_cast<size_t>(std::max(prefill,5)));
                EXPECT_LE(layout->physicalBytes(),envelope.admitted()->physicalBytes());
                EXPECT_TRUE(layout->grouped());
            }
            const auto schedule=retainedRawPrefillGraphBucketLadder(inventory,31,1,8);
            EXPECT_THROW(envelope.project(32,5,schedule),std::invalid_argument);
            EXPECT_THROW(envelope.project(31,4,schedule),std::invalid_argument);
            EXPECT_THROW(envelope.project(31,5,std::vector<int>{1,9,31}),std::invalid_argument);
        }
        CPUSerialExpertStorageEnvelope local(host,7,3,5,2,std::vector<int>{1,3,7},1,8,true,
            CPUSerialExpertFamilyPolicy::LocalPowerOfTwo,11,4);
        const auto ordinary=local.forController(host);
        const auto published=local.forController(DeviceId::cuda(0));
        ASSERT_TRUE(ordinary->canonical()); ASSERT_TRUE(published->canonical());
        EXPECT_FALSE(ordinary->mappedCanonical()); ASSERT_TRUE(published->mappedCanonical());
        EXPECT_EQ(published->physicalBytes(),ordinary->physicalBytes()+published->mappedCanonical()->physicalBytes());
        EXPECT_EQ(published->canonical()->physicalBytes(),ordinary->canonical()->physicalBytes());
        EXPECT_EQ(published->mappedCanonical()->hostAllocator(),host);
        EXPECT_EQ(published->mappedCanonical()->consumer(),DeviceId::cuda(0));
    }


    /** @test Exact CPU1 packet backing is shared by views and fixed after setup. */
    TEST(Test__MoELocalExpertStage_Params, PerLayerCpuPacketClaimsExactBackingAndRejectsGrowth)
    {
        const DeviceId device(DeviceType::CPU,1);
        constexpr int rows=3,width=33,top_k=3;
        const auto layout=CPUCompactPacketStorageLayout::declare(rows,width,top_k,device);
        const auto expected=2*AlignedVector<float>::requiredAllocationBytes(rows*width)+
            2*AlignedVector<float>::requiredAllocationBytes(rows*top_k);
        EXPECT_EQ(layout.physicalBytes(),expected);
        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(rows,rows*top_k,width,top_k,device);
        auto input=workspace.localExpertInput(0,0);
        auto output=workspace.localExpertOutput(0,0);
        MoELocalExpertStage::Params params;
        params.device_id=device;params.input_rows=&input;params.output_rows=&output;
        params.num_experts=4;params.top_k=top_k;params.d_model=width;params.expert_intermediate=17;
        params=test::admitCPUPerLayerPacketForTest(params);
        const auto authority=params.physical_memory_authority;
        const auto live=[&] { return authority->claimedBytes(device,PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation); };
        std::shared_ptr<TensorBase> retained_view;
        {
            MoELocalExpertStage stage(params);
            EXPECT_EQ(live(),expected);
            const auto *root=stage.compactHiddenTensorForDiagnostics();
            ASSERT_NE(root,nullptr);
            EXPECT_EQ(root->home_device(),device);
            retained_view=const_cast<FP32Tensor*>(root)->create_view({1,width},0);
            ASSERT_NE(retained_view,nullptr);
            for(int replay=0;replay<3;++replay)
            {
                ASSERT_TRUE(stage.preparePersistentBuffers());
                EXPECT_EQ(stage.compactHiddenTensorForDiagnostics(),root);
                EXPECT_EQ(live(),expected);
            }
            // Packet metadata cannot reopen the stage's immutable setup capacity.
            input.row_capacity=rows+1;
            input.entry_capacity=(rows+1)*top_k;
            EXPECT_THROW((void)stage.preparePersistentBuffers(),std::logic_error);
            EXPECT_EQ(stage.compactRowCapacityForDiagnostics(),rows);
            EXPECT_EQ(live(),expected);
        }
        EXPECT_EQ(live(),layout.hidden.physicalBytes());
        retained_view.reset();
        EXPECT_EQ(live(),0u);
    }

    /** @test Missing/foreign authority and partial setup failure publish no surviving packet claims. */
    TEST(Test__MoELocalExpertStage_Params, PerLayerCpuPacketRejectsMissingAdmissionAndRollsBackPartialSetup)
    {
        const DeviceId device(DeviceType::CPU,1);
        MoEOverlayCollectiveWorkspace workspace;
        workspace.ensureCapacity(3,9,33,3,device);
        auto input=workspace.localExpertInput(0,0);
        auto output=workspace.localExpertOutput(0,0);
        MoELocalExpertStage::Params params;
        params.device_id=device;params.input_rows=&input;params.output_rows=&output;
        params.num_experts=4;params.top_k=3;params.d_model=33;params.expert_intermediate=17;
        EXPECT_THROW((void)MoELocalExpertStage{params},std::invalid_argument);
        const auto layout=CPUCompactPacketStorageLayout::declare(3,33,3,device);
        const auto authority=[&](DeviceId allocator,size_t bytes) {
            PhysicalMemoryPlanBuilder plan;
            plan.add({0,allocator,bytes,bytes},PhysicalMemoryOwner::ActivationArena,bytes);
            return std::make_shared<PhysicalMemoryAuthority>(
                std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(plan.build()),0);
        };
        params.physical_memory_authority=authority(DeviceId::cpu(),layout.physicalBytes());
        EXPECT_THROW((void)MoELocalExpertStage{params},std::invalid_argument);
        params.physical_memory_authority=authority(device,layout.physicalBytes()-1);
        EXPECT_THROW((void)MoELocalExpertStage{params},std::logic_error);
        EXPECT_EQ(params.physical_memory_authority->claimedBytes(device,PhysicalMemoryOwner::ActivationArena,
            PhysicalMemoryMaterializationKind::NewAllocation),0u);
    }
} // namespace llaminar2::test
