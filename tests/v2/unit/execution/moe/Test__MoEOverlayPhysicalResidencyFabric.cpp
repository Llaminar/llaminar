/**
 * @file Test__MoEOverlayPhysicalResidencyFabric.cpp
 * @brief Device-free production-fabric tests for ExpertOverlay residency moves.
 *
 * These tests use the real CPU NativeVNNI prepared engines, physical inactive
 * slot pools, participant RCU banks, composite transport, and publication
 * authority.  Only the GPU DMA lanes are absent; no scripted transfer provider
 * or identity-only GEMM stands in for the production CPU path.
 */

#include "execution/moe/CpuExpertSlotPool.h"
#include "execution/moe/DecodeExpertHistogram.h"
#include "execution/moe/MoEOverlayParticipantMigration.h"
#include "execution/moe/MoEOverlayPhysicalResidencyFabric.h"
#include "execution/moe/MoEOverlayTierMigrationTransport.h"
#include "loaders/ModelLoader.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Build one metadata-only routed-MoE layer without reading payloads. */
    GGUFModel oneLayerExpertModel(
        GGUFTensorType type,
        std::uint64_t experts = 2)
    {
        GGUFModel model;
        model.tensors = {
            {
                .name = "blk.0.ffn_gate_exps.weight",
                .dimensions = {64, 32, experts},
                .type = type,
            },
            {
                .name = "blk.0.ffn_up_exps.weight",
                .dimensions = {64, 32, experts},
                .type = type,
            },
            {
                .name = "blk.0.ffn_down_exps.weight",
                .dimensions = {32, 64, experts},
                .type = type,
            },
        };
        return model;
    }

    /** @brief Build two process-local CPU tiers with an explicit owner order. */
    MoERoutedExpertPlacementPlan twoCpuTierPlan(
        std::vector<int> expert_tiers,
        RoutedExpertResidencyPolicy policy)
    {
        const int fixed_hot_capacity = static_cast<int>(std::count(
            expert_tiers.begin(), expert_tiers.end(), 0));
        RoutedExpertDomain hot;
        hot.name = "hot_domain";
        hot.scope = ExecutionDomainScope::SINGLE;
        hot.backend = CollectiveBackendType::HOST;
        hot.participants = {GlobalDeviceAddress::cpu(0)};
        hot.world_ranks = {0};
        hot.owner_rank = 0;
        hot.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;

        RoutedExpertDomain cold;
        cold.name = "cold_domain";
        cold.scope = ExecutionDomainScope::SINGLE;
        cold.backend = CollectiveBackendType::HOST;
        cold.participants = {GlobalDeviceAddress::cpu(1)};
        cold.world_ranks = {0};
        cold.owner_rank = 0;
        cold.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;

        RoutedExpertTier hot_tier;
        hot_tier.name = "hot";
        hot_tier.domain = hot.name;
        hot_tier.priority = 0;
        hot_tier.max_experts_per_layer = fixed_hot_capacity;

        RoutedExpertTier cold_tier;
        cold_tier.name = "cold";
        cold_tier.domain = cold.name;
        cold_tier.priority = 1;
        cold_tier.fallback = true;

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = hot.name;
        plan.shared_expert_domain = hot.name;
        plan.residency_policy = policy;
        plan.owner_order = RoutedExpertOwnerOrder::Ordinal;
        plan.domains = {std::move(hot), std::move(cold)};
        plan.routed_tiers = {std::move(hot_tier), std::move(cold_tier)};
        plan.placements = {{
            .layer = 0,
            .routed_expert_tier = std::move(expert_tiers),
        }};
        return plan;
    }

    /** @brief Q4_0 gate/up/down shapes matching a 64x32 routed MLP. */
    std::vector<CpuExpertSlotPool::ProjectionSpec> sourceProjectionSpecs()
    {
        const NativeVnniSourceIdentity source_identity{
            .codebook_id = native_vnni_formats::Q4_0.codebook_id,
            .is_superblock = native_vnni_formats::Q4_0.is_superblock,
            .present = true,
        };
        const ExpertWeightFormat format =
            ExpertWeightFormat::nativeVnni(source_identity);
        return {
            {
                .projection = ExpertTierWeightProjection::Gate,
                .N = 32,
                .K = 64,
                .format = format,
            },
            {
                .projection = ExpertTierWeightProjection::Up,
                .N = 32,
                .K = 64,
                .format = format,
            },
            {
                .projection = ExpertTierWeightProjection::Down,
                .N = 64,
                .K = 32,
                .format = format,
            },
        };
    }

    /** @brief Allocate and fill one immutable real CPU prepared expert. */
    MoEOverlayPreparedExpertTriplet makeSourceExpert(
        int participant_id,
        int expert_id,
        std::uint8_t seed)
    {
        auto pool = CpuExpertSlotPool::create({
            .participant_id = participant_id,
            .layer_idx = 0,
            .capacity = 1,
            .projections = sourceProjectionSpecs(),
            .memory_placement =
                CpuExpertSlotPool::MemoryPlacement::aggregateDomain(),
            .perf_device = "cpu-overlay-fabric-test",
        });
        auto lease = pool->acquire(expert_id, /*residency_epoch=*/1);
        if (!lease || lease->projections.size() != 3)
            throw std::runtime_error(
                "CPU overlay test could not acquire a complete source expert");

        MoEOverlayPreparedExpertTriplet result;
        for (const auto &projection : lease->projections)
        {
            for (std::size_t byte = 0;
                 byte < projection.destination_bytes.size();
                 ++byte)
            {
                projection.destination_bytes[byte] =
                    static_cast<std::uint8_t>(
                        seed + static_cast<std::uint8_t>(byte * 13u));
            }
            switch (projection.projection)
            {
            case ExpertTierWeightProjection::Gate:
                result.gate = projection.engine;
                break;
            case ExpertTierWeightProjection::Up:
                result.up = projection.engine;
                break;
            case ExpertTierWeightProjection::Down:
                result.down = projection.engine;
                break;
            }
        }
        if (!result.complete())
            throw std::runtime_error(
                "CPU overlay test source triplet is incomplete");
        return result;
    }

    /** @brief Register exact initial banks, including automatically-empty ones. */
    template <std::size_t ExpertCount>
    std::shared_ptr<MoEOverlayParticipantResidencyRegistry> makeRegistry(
        const std::shared_ptr<const MoEOverlayResidencySnapshot> &snapshot,
        const std::array<MoEOverlayPreparedExpertTriplet, ExpertCount> &experts)
    {
        static_assert(ExpertCount > 0);
        auto registry =
            std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = snapshot->owner_map,
                    .local_participant_ids = {0, 1},
                    .num_layers = 1,
                    .num_experts = static_cast<int>(ExpertCount),
                    .initial_epoch = snapshot->epoch,
                    .retained_epoch_capacity = 2,
                });

        std::string error;
        for (const int participant_id : std::array<int, 2>{0, 1})
        {
            const auto mask = snapshot->owner_map.expertMaskForParticipant(
                0, participant_id, static_cast<int>(ExpertCount));
            if (std::none_of(mask.begin(), mask.end(), [](bool resident)
                             { return resident; }))
            {
                /* The registry constructor publishes this exact empty bank. */
                continue;
            }
            std::vector<MoEOverlayPreparedExpertTriplet> resident_engines(
                ExpertCount);
            for (int expert_id = 0;
                 expert_id < static_cast<int>(ExpertCount);
                 ++expert_id)
            {
                if (mask[static_cast<std::size_t>(expert_id)])
                {
                    resident_engines[static_cast<std::size_t>(expert_id)] =
                        experts[static_cast<std::size_t>(expert_id)];
                }
            }
            if (!registry->registerInitialLayer(
                    participant_id,
                    0,
                    mask,
                    resident_engines,
                    &error))
            {
                throw std::runtime_error(error);
            }
        }
        if (!registry->allInitialBanksReady())
            throw std::runtime_error(
                "CPU overlay test registry did not publish every initial bank");
        return registry;
    }

    /** @brief Snapshot the exact final CPU bytes for a prepared triplet. */
    std::array<std::vector<std::uint8_t>, 3> packedBytes(
        const MoEOverlayPreparedExpertTriplet &triplet)
    {
        std::array<std::vector<std::uint8_t>, 3> result;
        const std::array<std::shared_ptr<ITensorGemm>, 3> engines{
            triplet.gate,
            triplet.up,
            triplet.down,
        };
        for (std::size_t projection = 0;
             projection < engines.size();
             ++projection)
        {
            const auto *packed =
                engines[projection]->exportCPUNativeVNNIPackedWeights();
            if (!packed)
                throw std::runtime_error(
                    "CPU overlay test engine cannot export packed weights");
            result[projection].assign(
                packed->native_interleaved.begin(),
                packed->native_interleaved.end());
        }
        return result;
    }
} // namespace

TEST(Test__MoEOverlayPhysicalResidencyFabric,
     GgufManifestCataloguesEveryQuantizedWeightCodebook)
{
    struct FormatCase
    {
        GGUFTensorType gguf_type;
        const NativeVnniFormatInfo *native_format;
    };
    const std::array formats{
        FormatCase{GGUFTensorType::Q4_0, &native_vnni_formats::Q4_0},
        FormatCase{GGUFTensorType::Q4_1, &native_vnni_formats::Q4_1},
        FormatCase{GGUFTensorType::Q5_0, &native_vnni_formats::Q5_0},
        FormatCase{GGUFTensorType::Q5_1, &native_vnni_formats::Q5_1},
        FormatCase{GGUFTensorType::Q8_0, &native_vnni_formats::Q8_0},
        FormatCase{GGUFTensorType::Q2_K, &native_vnni_formats::Q2_K},
        FormatCase{GGUFTensorType::Q3_K, &native_vnni_formats::Q3_K},
        FormatCase{GGUFTensorType::Q4_K, &native_vnni_formats::Q4_K},
        FormatCase{GGUFTensorType::Q5_K, &native_vnni_formats::Q5_K},
        FormatCase{GGUFTensorType::Q6_K, &native_vnni_formats::Q6_K},
        FormatCase{GGUFTensorType::Q8_K, &native_vnni_formats::Q8_K},
        FormatCase{GGUFTensorType::IQ2_XXS, &native_vnni_formats::IQ2_XXS},
        FormatCase{GGUFTensorType::IQ2_XS, &native_vnni_formats::IQ2_XS},
        FormatCase{GGUFTensorType::IQ3_XXS, &native_vnni_formats::IQ3_XXS},
        FormatCase{GGUFTensorType::IQ1_S, &native_vnni_formats::IQ1_S},
        FormatCase{GGUFTensorType::IQ4_NL, &native_vnni_formats::IQ4_NL},
        FormatCase{GGUFTensorType::IQ3_S, &native_vnni_formats::IQ3_S},
        FormatCase{GGUFTensorType::IQ2_S, &native_vnni_formats::IQ2_S},
        FormatCase{GGUFTensorType::IQ4_XS, &native_vnni_formats::IQ4_XS},
        FormatCase{GGUFTensorType::IQ1_M, &native_vnni_formats::IQ1_M},
    };

    for (const auto &format : formats)
    {
        SCOPED_TRACE(static_cast<std::uint32_t>(format.gguf_type));
        const auto manifest = buildMoEOverlayLayerWeightManifestFromGGUF(
            oneLayerExpertModel(format.gguf_type), 1, 2);
        ASSERT_EQ(manifest.size(), 1u);
        ASSERT_TRUE(manifest.front().valid());
        EXPECT_EQ(manifest.front().layer_idx, 0);

        for (std::size_t role = 0; role < 3; ++role)
        {
            const auto &projection = manifest.front().projections[role];
            EXPECT_EQ(
                projection.projection,
                static_cast<ExpertTierWeightProjection>(role));
            EXPECT_EQ(
                projection.N,
                role == static_cast<std::size_t>(
                            ExpertTierWeightProjection::Down)
                    ? 64
                    : 32);
            EXPECT_EQ(
                projection.K,
                role == static_cast<std::size_t>(
                            ExpertTierWeightProjection::Down)
                    ? 32
                    : 64);
            EXPECT_EQ(
                projection.format.native_vnni.codebook_id,
                format.native_format->codebook_id);
            EXPECT_EQ(
                projection.format.native_vnni.is_superblock,
                format.native_format->is_superblock);
            EXPECT_TRUE(projection.format.native_vnni.present);
            EXPECT_TRUE(projection.format.isNativeVnni());
        }
    }
}

TEST(Test__MoEOverlayPhysicalResidencyFabric,
     GgufManifestCataloguesEveryFloatingWeightPrecision)
{
    struct FormatCase
    {
        GGUFTensorType gguf_type;
        TensorType tensor_type;
    };
    constexpr std::array formats{
        FormatCase{GGUFTensorType::F16, TensorType::FP16},
        FormatCase{GGUFTensorType::BF16, TensorType::BF16},
        FormatCase{GGUFTensorType::F32, TensorType::FP32},
    };

    for (const auto &format : formats)
    {
        SCOPED_TRACE(static_cast<std::uint32_t>(format.gguf_type));
        const auto manifest = buildMoEOverlayLayerWeightManifestFromGGUF(
            oneLayerExpertModel(format.gguf_type), 1, 2);
        ASSERT_EQ(manifest.size(), 1u);
        ASSERT_TRUE(manifest.front().valid());
        for (const auto &projection : manifest.front().projections)
        {
            EXPECT_TRUE(projection.format.isFloating());
            EXPECT_EQ(
                projection.format.floatingTensorType(),
                std::optional<TensorType>(format.tensor_type));
            EXPECT_FALSE(projection.format.native_vnni.present);
        }
    }
}

TEST(Test__MoEOverlayPhysicalResidencyFabric,
     GgufManifestRejectsIncompleteOrIncompatibleMetadata)
{
    auto missing = oneLayerExpertModel(GGUFTensorType::Q4_0);
    missing.tensors.pop_back();
    EXPECT_THROW(
        (void)buildMoEOverlayLayerWeightManifestFromGGUF(missing, 1, 2),
        std::invalid_argument);

    EXPECT_THROW(
        (void)buildMoEOverlayLayerWeightManifestFromGGUF(
            oneLayerExpertModel(GGUFTensorType::Q4_0, 3), 1, 2),
        std::invalid_argument);

    auto bad_k = oneLayerExpertModel(GGUFTensorType::Q4_0);
    bad_k.tensors.front().dimensions[0] = 48;
    EXPECT_THROW(
        (void)buildMoEOverlayLayerWeightManifestFromGGUF(bad_k, 1, 2),
        std::invalid_argument);

    auto bad_float = oneLayerExpertModel(GGUFTensorType::BF16);
    bad_float.tensors.front().dimensions.clear();
    EXPECT_THROW(
        (void)buildMoEOverlayLayerWeightManifestFromGGUF(
            bad_float, 1, 2),
        std::invalid_argument);
}

TEST(Test__MoEOverlayPhysicalResidencyFabric,
     InitiallyEmptyColdTierGetsExactPreallocatedPools)
{
    auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
        MoEOverlayResidencyAuthority::Config{
            .initial_plan = twoCpuTierPlan(
                {0, 0}, RoutedExpertResidencyPolicy::StaticById),
            .model_metadata = {
                .num_layers = 1,
                .num_experts = 2,
                .d_model = 64,
                .routed_intermediate_size = 32,
                .routed_quant_type = "Q4_0",
            },
            .maintenance_mode = MoERebalanceRuntimeMode::Off,
            .histogram = nullptr,
            .perf_device = "cpu-hot/cpu-empty-cold",
        });
    const auto snapshot = authority->snapshot();
    ASSERT_NE(snapshot, nullptr);

    const std::array<MoEOverlayPreparedExpertTriplet, 2> experts{
        makeSourceExpert(0, 0, 17),
        makeSourceExpert(0, 1, 93),
    };
    auto registry = makeRegistry(snapshot, experts);
    ASSERT_NE(registry->endpoint(1)->acquire(snapshot->epoch), nullptr)
        << "A fully empty participant still requires its epoch-one bank";

    auto fabric = MoEOverlayPhysicalResidencyFabric::create({
        .registry = registry,
        .initial_snapshot = snapshot,
        .shadow_slots_per_endpoint_layer = 2,
        .staging_capacity_bytes = 64,
        .perf_device = "cpu-hot/cpu-empty-cold",
    });
    const auto stats = fabric->stats();
    EXPECT_EQ(stats.endpoint_layer_pools, 2u);
    EXPECT_EQ(stats.cpu_shadow_slots, 4u)
        << "Wave capacity must not be reduced to current endpoint occupancy";
    EXPECT_EQ(stats.gpu_shadow_slots, 0u);
    EXPECT_EQ(stats.persistent_transfer_lanes, 0u);
    EXPECT_EQ(stats.inference_stream_waits, 0u);
    EXPECT_EQ(stats.blocking_synchronizations, 0u);
}

TEST(Test__MoEOverlayPhysicalResidencyFabric,
     SingleShadowSlotSustainsRepeatedByteExactCpuRebalancing)
{
    const auto plan = twoCpuTierPlan(
        {0, 0, 1}, RoutedExpertResidencyPolicy::RoutedTierRebalanced);
    const auto owner_map = MoEExpertOwnerMap::build(plan);

    DecodeExpertHistogramConfig histogram_config;
    histogram_config.num_layers = 1;
    histogram_config.num_experts = 3;
    histogram_config.top_k = 1;
    histogram_config.window_size = 2;
    histogram_config.sockets = {DeviceId::cpu(), DeviceId::cpu()};
    histogram_config.ownership = owner_map.layeredOwnership(1, 3);
    auto histogram = std::make_shared<DecodeExpertHistogram>(histogram_config);
    auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
        MoEOverlayResidencyAuthority::Config{
            .initial_plan = plan,
            .model_metadata = {
                .num_layers = 1,
                .num_experts = 3,
                .d_model = 64,
                .routed_intermediate_size = 32,
                .routed_quant_type = "Q4_0",
            },
            .maintenance_mode = MoERebalanceRuntimeMode::Dynamic,
            .histogram = histogram.get(),
            .perf_device = "cpu-hot/cpu-cold",
        });
    const auto initial_snapshot = authority->snapshot();
    ASSERT_NE(initial_snapshot, nullptr);

    const std::array<MoEOverlayPreparedExpertTriplet, 3> experts{
        makeSourceExpert(0, 0, 11),
        makeSourceExpert(0, 1, 93),
        makeSourceExpert(1, 2, 177),
    };
    const std::array expected{
        packedBytes(experts[0]),
        packedBytes(experts[1]),
        packedBytes(experts[2]),
    };
    auto registry = makeRegistry(initial_snapshot, experts);
    auto fabric = MoEOverlayPhysicalResidencyFabric::create({
        .registry = registry,
        .initial_snapshot = initial_snapshot,
        .shadow_slots_per_endpoint_layer = 1,
        .staging_capacity_bytes = 64,
        .perf_device = "cpu-hot/cpu-cold",
    });

    MoEOverlayParticipantPreparedWaveFactory factory({
        .registry = registry,
        .transfer_provider = fabric,
        .perf_device = "cpu-hot/cpu-cold",
    });
    MoEOverlayTierMigrationTransport transport({
        .factory = &factory,
        .projections_per_expert = 3,
        .perf_device = "cpu-hot/cpu-cold",
    });

    const auto commit_window = [&](const std::array<std::uint64_t, 3> &counts,
                                   std::uint64_t expected_epoch)
    {
        /* Rotation clears the previous evidence bank, so each wave is driven
         * solely by this window's deliberately reversed expert temperature. */
        histogram->mergeLayerCounts(0, counts.data(), 3, false);
        const auto transaction = authority->proposeFromHistogram();
        ASSERT_TRUE(transaction.valid());
        ASSERT_EQ(transaction.candidate->epoch, expected_epoch);
        ASSERT_EQ(transaction.migrations.size(), 2u);
        ASSERT_EQ(transaction.migration_cycles.size(), 1u);

        auto progress = authority->beginApply(transaction, transport);
        ASSERT_EQ(progress.status, MoEOverlayResidencyApplyStatus::Started)
            << progress.error;

        bool committed = false;
        for (int poll = 0; poll < 4096; ++poll)
        {
            progress = authority->advanceBackground();
            ASSERT_TRUE(progress.ok()) << progress.error;
            if (progress.status == MoEOverlayResidencyApplyStatus::Committed)
            {
                committed = true;
                break;
            }
        }
        ASSERT_TRUE(committed)
            << "Bounded CPU-copy polling did not publish epoch "
            << expected_epoch;
        EXPECT_EQ(authority->pendingRetirementCount(), 0u)
            << "Ticket-free publication must recycle the old physical bank "
               "before the next wave";

        for (const auto &migration : transaction.migrations)
        {
            const auto bank = registry
                                  ->endpoint(
                                      migration.destination.owner_participant)
                                  ->acquire(transaction.candidate->epoch);
            ASSERT_NE(bank, nullptr);
            const auto &arrived =
                bank->layers[static_cast<std::size_t>(migration.layer_idx)]
                    .experts[static_cast<std::size_t>(migration.expert_id)];
            ASSERT_TRUE(arrived.complete());
            EXPECT_EQ(
                packedBytes(arrived),
                expected[static_cast<std::size_t>(migration.expert_id)])
                << "Published destination bytes differ from expert "
                << migration.expert_id << " at epoch " << expected_epoch;
        }
    };

    /*
     * Expert 1 remains in its loader-owned slot through epoch two and departs
     * only in the second wave.  Reaching epoch four with one shadow allocation
     * proves bootstrap retirement keys physical birth at epoch one while the
     * retirement fence may name a later bank.  The third arrival then consumes
     * that reclaimed slot and also proves later lease-managed reuse.
     */
    commit_window({1, 80, 100}, 2);
    commit_window({100, 1, 80}, 3);
    commit_window({80, 100, 1}, 4);

    const auto fabric_stats = fabric->stats();
    EXPECT_EQ(fabric_stats.adopted_initial_slots, 3u);
    EXPECT_EQ(fabric_stats.adopted_initial_slots_recycled, 3u);
    EXPECT_EQ(fabric_stats.cpu_shadow_slots, 2u);
    EXPECT_EQ(fabric_stats.waves_prepared, 3u);
    EXPECT_EQ(fabric_stats.waves_deferred, 0u);
    EXPECT_EQ(fabric_stats.waves_failed, 0u);
    EXPECT_EQ(fabric_stats.projection_operations_prepared, 18u);
    EXPECT_EQ(fabric_stats.cpu_copy_operations, 18u);
    EXPECT_EQ(fabric_stats.gpu_cpu_operations, 0u);
    EXPECT_EQ(fabric_stats.inference_stream_waits, 0u);
    EXPECT_EQ(fabric_stats.blocking_synchronizations, 0u);

    const auto transport_stats = transport.stats();
    EXPECT_EQ(transport_stats.waves_started, 3u);
    EXPECT_EQ(transport_stats.transfer_operations_completed, 18u);
    EXPECT_EQ(transport_stats.commits_completed, 3u);
    EXPECT_EQ(transport_stats.inference_stream_waits, 0u);
    EXPECT_EQ(transport_stats.blocking_synchronizations, 0u);

    const auto authority_stats = authority->stats();
    EXPECT_EQ(authority_stats.committed_waves, 3u);
    EXPECT_EQ(authority_stats.committed_migrations, 6u);
    EXPECT_EQ(authority_stats.promotions, 3u);
    EXPECT_EQ(authority_stats.demotions, 3u);
    EXPECT_EQ(authority_stats.cross_domain_migrations, 6u);
}
