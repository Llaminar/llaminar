/**
 * @file Test__MoELocalExpertStage_Params.cpp
 * @brief Device-free construction and ownership tests for local MoE stages.
 *
 * These tests exercise only typed parameter and compact-buffer ownership
 * contracts.  GPU placement and execution are intentionally left to explicit
 * integration suites, keeping the unit target fast and accelerator-free.
 */

#include "execution/compute_stages/stages/MoELocalExpertStage.h"
#include "execution/compute_stages/stages/MoEExpertComputeStage.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <utility>

namespace llaminar2::test
{
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
            MoELocalExpertSerialBufferArena::Config{
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
                .debug_name = "unit.serial_compact"});
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

        MoELocalExpertStage first(params);
        MoELocalExpertStage second(params);
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
        MoELocalExpertStage small_stage(small_params);
        EXPECT_EQ(small_stage.compactRowCapacityForDiagnostics(), 2u);
        EXPECT_EQ(
            small_stage.compactHiddenTensorForDiagnostics(),
            arena->smallestFamilySupporting(2)->hidden.get());
        EXPECT_NE(
            small_stage.compactHiddenTensorForDiagnostics(),
            first.compactHiddenTensorForDiagnostics());

        params.runtime_participant_index = 2;
        EXPECT_THROW(
            (void)MoELocalExpertStage{params},
            std::runtime_error)
            << "A serial arena must not be rebound to another logical participant";
        params.runtime_participant_index = 3;

        auto undersized = std::make_shared<MoELocalExpertSerialBufferArena>(
            MoELocalExpertSerialBufferArena::Config{
                .device_id = DeviceId::cpu(),
                .row_capacity = kRowCapacity - 1u,
                .d_model = kDModel,
                .routing_top_k = kTopK,
                .logical_participant_id = 3,
                .debug_name = "unit.undersized_serial_compact"});
        params.serial_compact_buffer_arena = std::move(undersized);
        EXPECT_THROW(
            (void)MoELocalExpertStage{params},
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

} // namespace llaminar2::test
