/**
 * @file Test__MoEOverlayCapacityResolver.cpp
 * @brief Device-free proof of exact ExpertOverlay capacity and quota planning.
 *
 * These tests independently reproduce the CPU and GPU allocation arithmetic,
 * sweep every catalogued source codebook, and exercise multi-tier admission
 * against exact physical budgets.  No backend is initialized: the suite is a
 * fast protocol/BOM gate suitable for ordinary unit-test execution.
 */

#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "execution/moe/MoEOverlayCapacityResolver.h"
#include "execution/moe/MoEOverlayLocalCapacityPlanner.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "tensors/NativeVnniFormatInfo.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace
    {
        constexpr std::size_t kGpuAlignment = 256;
        constexpr std::size_t kCpuCacheLine = 64;
        constexpr std::size_t kCpuPage = 4096;

        /** @return `value` rounded to one power-of-two test alignment. */
        [[nodiscard]] constexpr std::size_t testAlignUp(
            std::size_t value,
            std::size_t alignment)
        {
            return (value + alignment - 1) & ~(alignment - 1);
        }

        /** @return One complete projection manifest with exact source identity. */
        [[nodiscard]] MoEOverlayProjectionWeightManifest projection(
            ExpertTierWeightProjection role,
            int N,
            int K,
            const NativeVnniFormatInfo &format)
        {
            return {
                .projection = role,
                .N = N,
                .K = K,
                .format = ExpertWeightFormat::nativeVnni({
                    .codebook_id = format.codebook_id,
                    .is_superblock = format.is_superblock,
                    .present = true,
                }),
            };
        }

        /** @return One valid gate/up/down layer with non-trivial alignment edges. */
        [[nodiscard]] MoEOverlayLayerWeightManifest layer(
            int layer_idx,
            const NativeVnniFormatInfo &format,
            int geometry_delta = 0)
        {
            MoEOverlayLayerWeightManifest result;
            result.layer_idx = layer_idx;
            result.projections = {{
                projection(
                    ExpertTierWeightProjection::Gate,
                    96 + geometry_delta,
                    64,
                    format),
                projection(
                    ExpertTierWeightProjection::Up,
                    80 + geometry_delta,
                    96,
                    format),
                projection(
                    ExpertTierWeightProjection::Down,
                    64 + geometry_delta,
                    128,
                    format),
            }};
            return result;
        }

        /** @return A contiguous model manifest using one source codebook. */
        [[nodiscard]] std::vector<MoEOverlayLayerWeightManifest> manifest(
            int layer_count,
            const NativeVnniFormatInfo &format,
            bool vary_geometry = false)
        {
            std::vector<MoEOverlayLayerWeightManifest> result;
            result.reserve(static_cast<std::size_t>(layer_count));
            for (int layer_idx = 0; layer_idx < layer_count; ++layer_idx)
            {
                result.push_back(layer(
                    layer_idx,
                    format,
                    vary_geometry ? layer_idx * 16 : 0));
            }
            return result;
        }

        /** @return Independently expected CPU interleaved unit stride. */
        [[nodiscard]] constexpr std::size_t expectedCpuStride(
            const NativeVnniFormatInfo &format)
        {
            if (format.codebook_id == 8)
                return 1792;
            if (format.codebook_id == 0 ||
                format.codebook_id == 4 ||
                format.codebook_id == 5)
            {
                return 1024 + 256 + (format.is_asymmetric ? 128 : 0);
            }
            return 2048 + 256 + (format.is_asymmetric ? 128 : 0);
        }

        /** @return Independently expected allocation for one CPU projection. */
        [[nodiscard]] std::size_t expectedCpuProjectionBytes(
            const MoEOverlayProjectionWeightManifest &spec,
            const NativeVnniFormatInfo &format)
        {
            const std::size_t padded_n = testAlignUp(
                static_cast<std::size_t>(spec.N), 64);
            const std::size_t units =
                (padded_n / 64) * static_cast<std::size_t>(spec.K / 32);
            const std::size_t logical = units * expectedCpuStride(format);
            return testAlignUp(
                logical,
                logical >= kCpuPage ? kCpuPage : kCpuCacheLine);
        }

        /** @return Independently expected separated GPU projection allocation. */
        [[nodiscard]] std::size_t expectedGpuProjectionBytes(
            const MoEOverlayProjectionWeightManifest &spec,
            int payload_bytes_per_block,
            bool asymmetric,
            bool has_emins)
        {
            const std::size_t blocks =
                static_cast<std::size_t>(spec.N) *
                static_cast<std::size_t>(spec.K / 32);
            std::size_t cursor = 0;
            const auto append = [&](std::size_t bytes)
            {
                cursor = testAlignUp(cursor, kGpuAlignment);
                cursor += bytes;
            };
            append(blocks * static_cast<std::size_t>(payload_bytes_per_block));
            append(blocks * sizeof(std::uint16_t));
            if (asymmetric)
                append(blocks * sizeof(std::uint16_t));
            if (has_emins)
                append(blocks * sizeof(std::uint32_t));
            return testAlignUp(cursor, kGpuAlignment);
        }

        /** @return Migration-stable GPU payload width for a CPU round trip. */
        [[nodiscard]] constexpr int expectedMigrationPayload(
            const NativeVnniFormatInfo &format)
        {
            if (format.codebook_id == 8)
                return 24;
            if (format.codebook_id == 0 ||
                format.codebook_id == 4 ||
                format.codebook_id == 5)
            {
                return 16;
            }
            return 32;
        }

        /** @return Expected complete gate/up/down footprint from independent math. */
        [[nodiscard]] MoEOverlayPreparedExpertFootprint expectedFootprint(
            const MoEOverlayLayerWeightManifest &model_layer,
            const NativeVnniFormatInfo &format)
        {
            MoEOverlayPreparedExpertFootprint expected;
            expected.layer_idx = model_layer.layer_idx;
            for (const auto &spec : model_layer.projections)
            {
                const std::size_t cpu =
                    expectedCpuProjectionBytes(spec, format);
                expected.cpu_live_bytes += cpu;
                expected.cpu_shadow_bytes += cpu;
                expected.gpu_live_bytes += expectedGpuProjectionBytes(
                    spec,
                    std::max(
                        format.payload_bytes,
                        expectedMigrationPayload(format)),
                    format.is_asymmetric,
                    format.has_emins);
                expected.gpu_shadow_bytes += expectedGpuProjectionBytes(
                    spec,
                    /*payload_bytes_per_block=*/32,
                    /*asymmetric=*/true,
                    /*has_emins=*/true);
            }
            return expected;
        }

        /** @return Exact physical budget for specified live and shadow copies. */
        [[nodiscard]] MoEOverlayPhysicalMemoryBudget exactBudget(
            std::string resource_id,
            DeviceId device,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::vector<int> &live_copies_per_layer,
            int shadow_copies_per_layer,
            std::size_t fixed_bytes = 113,
            std::size_t staging_bytes = 257,
            std::size_t reserve_bytes = 509)
        {
            if (footprints.size() != live_copies_per_layer.size() ||
                shadow_copies_per_layer < 0)
            {
                throw std::invalid_argument("invalid test budget geometry");
            }
            MoEOverlayPhysicalMemoryBudget result{
                .resource_id = std::move(resource_id),
                .device = device,
                .fixed_bytes = fixed_bytes,
                .transfer_staging_bytes = staging_bytes,
                .safety_reserve_bytes = reserve_bytes,
            };
            result.usable_budget_bytes =
                fixed_bytes + staging_bytes + reserve_bytes;
            for (std::size_t layer_idx = 0;
                 layer_idx < footprints.size();
                 ++layer_idx)
            {
                result.usable_budget_bytes +=
                    static_cast<std::size_t>(live_copies_per_layer[layer_idx]) *
                    footprints[layer_idx].liveBytes(device);
                result.usable_budget_bytes +=
                    static_cast<std::size_t>(shadow_copies_per_layer) *
                    footprints[layer_idx].shadowBytes(device);
            }
            return result;
        }

        /** @return Rank/device-qualified production-admission form of a test budget. */
        [[nodiscard]] MoEOverlayBoundPhysicalMemoryBudget boundBudget(
            int world_rank,
            const MoEOverlayPhysicalMemoryBudget &budget)
        {
            return {
                .world_rank = world_rank,
                .device = budget.device,
                .resource_id = budget.resource_id,
                .usable_budget_bytes = budget.usable_budget_bytes,
                .fixed_bytes = budget.fixed_bytes,
                .additional_transfer_staging_bytes =
                    budget.transfer_staging_bytes,
                .safety_reserve_bytes = budget.safety_reserve_bytes,
            };
        }

        /** @return One hardware-bound domain with an explicit whole-expert policy. */
        [[nodiscard]] RoutedExpertDomain boundDomain(
            std::string name,
            int world_rank,
            GlobalDeviceAddress participant,
            RoutedExpertComputePolicy policy =
                RoutedExpertComputePolicy::Apportioned)
        {
            RoutedExpertDomain result;
            result.name = std::move(name);
            result.scope = ExecutionDomainScope::SINGLE;
            result.participants = {std::move(participant)};
            result.world_ranks = {world_rank};
            result.owner_rank = world_rank;
            result.routed_compute_policy = policy;
            return result;
        }

        /** @return Small complete MoE model profile for fixed-memory planner tests. */
        [[nodiscard]] ModelMemoryProfile smallModelProfile()
        {
            ModelMemoryProfile profile;
            profile.n_layers = 2;
            profile.d_model = 64;
            profile.d_ff = 128;
            profile.n_heads = 4;
            profile.n_kv_heads = 2;
            profile.head_dim = 16;
            profile.max_seq_len = 32;
            profile.expert_count = 4;
            profile.expert_used_count = 2;
            profile.expert_feed_forward_length = 96;
            return profile;
        }

        /** @return Explicit model/load contract required by GPU capacity tests. */
        [[nodiscard]] MoEOverlayGPUWeightLoadCapacityInput
        testGPUWeightLoadCapacityInput(
            std::size_t maximum_source_bytes = 32u * 1024u * 1024u)
        {
            return {
                .policy = GPUWeightLoadMemoryPolicy{
                    .staging_stream_count = 3,
                    .staging_budget_bytes = 24u * 1024u * 1024u,
                    .safety_margin_percent = 0,
                    .minimum_safety_margin_bytes =
                        128u * 1024u * 1024u,
                },
                .maximum_source_bytes = maximum_source_bytes,
            };
        }

        /** @return One participant binding used by a test tier. */
        [[nodiscard]] MoEOverlayTierCapacityParticipant participant(
            int participant_id,
            std::string resource_id,
            std::size_t shadow_slots = 1)
        {
            return {
                .participant_id = participant_id,
                .resource_id = std::move(resource_id),
                .shadow_slots_per_layer = shadow_slots,
            };
        }

        /** @return One fully typed tier capacity request. */
        [[nodiscard]] MoEOverlayTierCapacityRequest tier(
            int tier_index,
            std::string name,
            int priority,
            bool fallback,
            MoEOverlayLiveQuotaMode quota_mode,
            std::vector<MoEOverlayTierCapacityParticipant> participants,
            std::vector<int> fixed_quotas = {},
            MoEOverlayTierCopyPolicy copy_policy =
                MoEOverlayTierCopyPolicy::Apportioned,
            std::vector<int> maximum_quotas = {})
        {
            return {
                .tier_index = tier_index,
                .tier_name = std::move(name),
                .priority = priority,
                .fallback = fallback,
                .quota_mode = quota_mode,
                .copy_policy = copy_policy,
                .fixed_live_experts_per_layer = std::move(fixed_quotas),
                .max_live_experts_per_layer = std::move(maximum_quotas),
                .participants = std::move(participants),
            };
        }
    } // namespace

    TEST(MoEOverlayCapacityResolver, EveryCataloguedCodebookHasExactCpuAndGpuFootprint)
    {
        ASSERT_EQ(native_vnni_formats::kAllSourceFormats.size(), 21u);
        for (const auto &source : native_vnni_formats::kAllSourceFormats)
        {
            SCOPED_TRACE(::testing::Message() << source.quant_type);
            ASSERT_NE(source.metadata, nullptr);
            const auto model_manifest = manifest(1, *source.metadata);
            const auto actual =
                MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
            ASSERT_EQ(actual.size(), 1u);
            const auto expected =
                expectedFootprint(model_manifest.front(), *source.metadata);

            EXPECT_EQ(actual[0].layer_idx, 0);
            EXPECT_EQ(actual[0].cpu_live_bytes, expected.cpu_live_bytes);
            EXPECT_EQ(actual[0].cpu_shadow_bytes, expected.cpu_shadow_bytes);
            EXPECT_EQ(actual[0].gpu_live_bytes, expected.gpu_live_bytes);
            EXPECT_EQ(actual[0].gpu_shadow_bytes, expected.gpu_shadow_bytes);
            EXPECT_EQ(actual[0].liveBytes(DeviceId::cpu()), expected.cpu_live_bytes);
            EXPECT_EQ(actual[0].liveBytes(DeviceId::cuda(0)), expected.gpu_live_bytes);
            EXPECT_EQ(actual[0].liveBytes(DeviceId::rocm(7)), expected.gpu_live_bytes);
            EXPECT_EQ(actual[0].shadowBytes(DeviceId::cpu()), expected.cpu_shadow_bytes);
            EXPECT_EQ(actual[0].shadowBytes(DeviceId::cuda(0)), expected.gpu_shadow_bytes);
            EXPECT_GE(actual[0].gpu_shadow_bytes, actual[0].gpu_live_bytes);
        }
    }

    TEST(MoEOverlayCapacityResolver, AutomaticTiersUseOnlyPriorityAcrossScrambledNamesAndOrder)
    {
        constexpr int kExperts = 6;
        const auto model_manifest = manifest(
            2, native_vnni_formats::Q4_K, /*vary_geometry=*/true);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);

        MoEOverlayCapacityResolverInput input;
        input.num_experts = kExperts;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget("resource-z", DeviceId::cuda(0), footprints, {2, 2}, 1),
            exactBudget("resource-a", DeviceId::rocm(0), footprints, {1, 1}, 1),
            exactBudget("resource-m", DeviceId::cpu(), footprints, {3, 3}, 1),
        };
        input.tiers = {
            tier(2, "opaque-archive", 40, true,
                 MoEOverlayLiveQuotaMode::Automatic,
                 {participant(300, "resource-m")}),
            tier(0, "opaque-zebra", -7, false,
                 MoEOverlayLiveQuotaMode::Automatic,
                 {participant(100, "resource-z")}),
            tier(1, "opaque-alpha", 12, false,
                 MoEOverlayLiveQuotaMode::Automatic,
                 {participant(200, "resource-a")}),
        };

        const auto result = MoEOverlayCapacityResolver::resolve(input);
        ASSERT_NE(result.tier(0), nullptr);
        ASSERT_NE(result.tier(1), nullptr);
        ASSERT_NE(result.tier(2), nullptr);
        EXPECT_EQ(result.tier(0)->live_experts_per_layer,
                  (std::vector<int>{2, 2}));
        EXPECT_EQ(result.tier(1)->live_experts_per_layer,
                  (std::vector<int>{1, 1}));
        EXPECT_EQ(result.tier(2)->live_experts_per_layer,
                  (std::vector<int>{3, 3}));
        EXPECT_TRUE(result.tier(0)->has_live_residency);
        EXPECT_TRUE(result.tier(1)->has_live_residency);
        EXPECT_TRUE(result.tier(2)->has_live_residency);

        for (const auto &budget : input.physical_budgets)
        {
            const auto *resource = result.resource(budget.resource_id);
            ASSERT_NE(resource, nullptr) << budget.resource_id;
            EXPECT_EQ(resource->used_bytes, budget.usable_budget_bytes);
            EXPECT_EQ(resource->remaining_bytes, 0u);
            EXPECT_EQ(resource->shadow_copies_per_layer,
                      (std::vector<int>{1, 1}));
        }
        for (std::size_t layer_idx = 0; layer_idx < 2; ++layer_idx)
        {
            EXPECT_EQ(
                result.tier(0)->live_experts_per_layer[layer_idx] +
                    result.tier(1)->live_experts_per_layer[layer_idx] +
                    result.tier(2)->live_experts_per_layer[layer_idx],
                kExperts);
        }

        MoERoutedExpertPlacementPlan placement_plan;
        placement_plan.enabled = true;
        placement_plan.topology =
            RoutedExpertPlacementTopology::TieredOverlay;
        placement_plan.continuation_domain = "opaque-domain";
        placement_plan.shared_expert_domain = "opaque-domain";
        placement_plan.residency_policy =
            RoutedExpertResidencyPolicy::StaticById;
        RoutedExpertDomain domain;
        domain.name = "opaque-domain";
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.participants = {GlobalDeviceAddress::cpu(0)};
        domain.owner_rank = 0;
        placement_plan.domains = {domain};
        placement_plan.routed_tiers = {
            RoutedExpertTier{
                .name = "opaque-zebra",
                .domain = domain.name,
                .priority = -7,
            },
            RoutedExpertTier{
                .name = "opaque-alpha",
                .domain = domain.name,
                .priority = 12,
            },
            RoutedExpertTier{
                .name = "opaque-archive",
                .domain = domain.name,
                .priority = 40,
                .fallback = true,
            },
        };

        const auto capacity_bound =
            MoEOverlayCapacityResolver::installResolvedQuotas(
                placement_plan,
                result);
        EXPECT_EQ(
            capacity_bound.routed_tiers[0]
                .resolved_live_experts_per_layer,
            (std::vector<int>{2, 2}));
        EXPECT_EQ(
            capacity_bound.routed_tiers[1]
                .resolved_live_experts_per_layer,
            (std::vector<int>{1, 1}));
        EXPECT_EQ(
            capacity_bound.routed_tiers[2]
                .resolved_live_experts_per_layer,
            (std::vector<int>{3, 3}));

        MoERoutedExpertModelMetadata metadata;
        metadata.num_layers = 2;
        metadata.num_experts = kExperts;
        metadata.d_model = 64;
        metadata.routed_intermediate_size = 96;
        metadata.routed_quant_type = "Q4_K";
        const auto placed = MoERoutedExpertPlacementPlanner::plan(
            capacity_bound,
            metadata);
        ASSERT_EQ(placed.planned_plan.placements.size(), 2u);
        EXPECT_EQ(
            placed.planned_plan.placements[0].routed_expert_tier,
            (std::vector<int>{0, 0, 1, 2, 2, 2}));
        EXPECT_EQ(
            placed.planned_plan.placements[1].routed_expert_tier,
            (std::vector<int>{0, 0, 1, 2, 2, 2}));
    }

    TEST(MoEOverlayCapacityAdmission,
         BoundRankDeviceResourcesDriveScrambledThreeTierPriorityQuotas)
    {
        constexpr int kExperts = 6;
        const auto model_manifest = manifest(
            2, native_vnni_formats::Q4_K, /*vary_geometry=*/true);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);

        const auto cuda_budget = exactBudget(
            "physical-rank7-cuda0",
            DeviceId::cuda(0),
            footprints,
            {2, 2},
            /*shadow_copies_per_layer=*/1);
        const auto rocm_budget = exactBudget(
            "physical-rank2-rocm3",
            DeviceId::rocm(3),
            footprints,
            {1, 1},
            /*shadow_copies_per_layer=*/1);
        const auto cpu_budget = exactBudget(
            "physical-rank11-cpu",
            DeviceId::cpu(),
            footprints,
            {3, 3},
            /*shadow_copies_per_layer=*/1);

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "domain-cuda";
        plan.base_model_domain = "domain-cuda";
        plan.shared_expert_domain = "domain-cuda";
        plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan.domains = {
            boundDomain(
                "domain-cpu", 11, GlobalDeviceAddress::cpu(1, "node-z")),
            boundDomain(
                "domain-cuda", 7, GlobalDeviceAddress::cuda(0, 0, "node-a")),
            boundDomain(
                "domain-rocm", 2, GlobalDeviceAddress::rocm(3, 1, "node-b")),
        };
        /* Declaration/index order deliberately opposes integer preference. */
        plan.routed_tiers = {
            RoutedExpertTier{
                .name = "violet",
                .domain = "domain-cpu",
                .priority = 91,
                .fallback = true,
            },
            RoutedExpertTier{
                .name = "square",
                .domain = "domain-cuda",
                .priority = -13,
            },
            RoutedExpertTier{
                .name = "lemur",
                .domain = "domain-rocm",
                .priority = 6,
            },
        };

        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> budgets = {
            boundBudget(2, rocm_budget),
            boundBudget(11, cpu_budget),
            boundBudget(7, cuda_budget),
        };
        const auto input = MoEOverlayCapacityAdmission::buildResolverInput(
            plan, kExperts, model_manifest, budgets);
        ASSERT_EQ(input.tiers.size(), 3u);
        EXPECT_EQ(input.tiers[0].tier_name, "violet");
        EXPECT_EQ(input.tiers[0].priority, 91);
        EXPECT_EQ(input.tiers[0].participants[0].resource_id,
                  cpu_budget.resource_id);
        EXPECT_EQ(input.tiers[1].participants[0].resource_id,
                  cuda_budget.resource_id);
        EXPECT_EQ(input.tiers[2].participants[0].resource_id,
                  rocm_budget.resource_id);

        const auto capacity = MoEOverlayCapacityResolver::resolve(input);
        ASSERT_NE(capacity.tier(0), nullptr);
        ASSERT_NE(capacity.tier(1), nullptr);
        ASSERT_NE(capacity.tier(2), nullptr);
        EXPECT_EQ(capacity.tier(0)->live_experts_per_layer,
                  (std::vector<int>{3, 3}));
        EXPECT_EQ(capacity.tier(1)->live_experts_per_layer,
                  (std::vector<int>{2, 2}));
        EXPECT_EQ(capacity.tier(2)->live_experts_per_layer,
                  (std::vector<int>{1, 1}));

        const auto capacity_bound =
            MoEOverlayCapacityAdmission::resolveAndInstall(
                plan, kExperts, model_manifest, budgets);
        MoERoutedExpertModelMetadata metadata;
        metadata.num_layers = 2;
        metadata.num_experts = kExperts;
        metadata.d_model = 64;
        metadata.routed_intermediate_size = 96;
        metadata.routed_quant_type = "Q4_K";
        const auto placed = MoERoutedExpertPlacementPlanner::plan(
            capacity_bound, metadata);
        ASSERT_EQ(placed.planned_plan.placements.size(), 2u);
        for (const auto &layer_placement : placed.planned_plan.placements)
        {
            EXPECT_EQ(
                layer_placement.routed_expert_tier,
                (std::vector<int>{1, 1, 2, 0, 0, 0}));
        }
    }

    TEST(MoEOverlayCapacityAdmission,
         SharedPhysicalResourceIsChargedOnceAcrossOpaqueLogicalTiers)
    {
        constexpr int kExperts = 2;
        const auto model_manifest = manifest(1, native_vnni_formats::Q5_K);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        const auto shared_budget = exactBudget(
            "rank4-cuda1-shared-authority",
            DeviceId::cuda(1),
            footprints,
            {2},
            /*shadow_copies_per_layer=*/2);

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "left-domain";
        plan.base_model_domain = "left-domain";
        plan.shared_expert_domain = "left-domain";
        plan.residency_policy = RoutedExpertResidencyPolicy::StaticById;
        plan.domains = {
            boundDomain(
                "left-domain", 4, GlobalDeviceAddress::cuda(1, 0, "node")),
            boundDomain(
                "right-domain", 4, GlobalDeviceAddress::cuda(1, 0, "node")),
        };
        plan.routed_tiers = {
            RoutedExpertTier{
                .name = "first-opaque-label",
                .domain = "left-domain",
                .priority = -2,
                .max_experts_per_layer = 1,
            },
            RoutedExpertTier{
                .name = "second-opaque-label",
                .domain = "right-domain",
                .priority = 44,
                .fallback = true,
            },
        };

        const auto input = MoEOverlayCapacityAdmission::buildResolverInput(
            plan,
            kExperts,
            model_manifest,
            {boundBudget(4, shared_budget)});
        ASSERT_EQ(input.physical_budgets.size(), 1u);
        ASSERT_EQ(input.tiers.size(), 2u);
        EXPECT_EQ(input.tiers[0].participants[0].resource_id,
                  shared_budget.resource_id);
        EXPECT_EQ(input.tiers[1].participants[0].resource_id,
                  shared_budget.resource_id);
        EXPECT_NE(input.tiers[0].participants[0].participant_id,
                  input.tiers[1].participants[0].participant_id);

        const auto capacity = MoEOverlayCapacityResolver::resolve(input);
        EXPECT_EQ(capacity.tier(0)->live_experts_per_layer,
                  (std::vector<int>{1}));
        EXPECT_EQ(capacity.tier(1)->live_experts_per_layer,
                  (std::vector<int>{1}));
        const auto *physical = capacity.resource(shared_budget.resource_id);
        ASSERT_NE(physical, nullptr);
        EXPECT_EQ(physical->shadow_copies_per_layer,
                  (std::vector<int>{2}));
        EXPECT_EQ(physical->live_copies_per_layer,
                  (std::vector<int>{2}));
        EXPECT_EQ(physical->used_bytes, shared_budget.usable_budget_bytes);
    }

    TEST(MoEOverlayCapacityAdmission,
         MissingBoundResourceAndTensorShardedTierFailClosed)
    {
        const auto model_manifest = manifest(1, native_vnni_formats::Q4_0);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        const auto budget = exactBudget(
            "rank3-cuda0",
            DeviceId::cuda(0),
            footprints,
            {1},
            /*shadow_copies_per_layer=*/1);

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "opaque-domain";
        plan.base_model_domain = "opaque-domain";
        plan.shared_expert_domain = "opaque-domain";
        plan.domains = {boundDomain(
            "opaque-domain", 3, GlobalDeviceAddress::cuda(0))};
        plan.routed_tiers = {RoutedExpertTier{
            .name = "opaque-tier",
            .domain = "opaque-domain",
            .priority = 8,
            .fallback = true,
        }};

        EXPECT_THROW(
            (void)MoEOverlayCapacityAdmission::buildResolverInput(
                plan,
                /*num_experts=*/1,
                model_manifest,
                {boundBudget(9, budget)}),
            std::invalid_argument);

        plan.domains[0].routed_compute_policy =
            RoutedExpertComputePolicy::TensorSharded;
        EXPECT_THROW(
            (void)MoEOverlayCapacityAdmission::buildResolverInput(
                plan,
                /*num_experts=*/1,
                model_manifest,
                {boundBudget(3, budget)}),
            std::invalid_argument);
    }

    TEST(MoEOverlayCapacityAdmission,
         TransferStagingBomPricesLocalHeterogeneousAndRelayResourcesExactly)
    {
        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "domain-a";
        plan.base_model_domain = "domain-a";
        plan.shared_expert_domain = "domain-a";
        plan.domains = {
            boundDomain("domain-a", 0, GlobalDeviceAddress::cuda(0)),
            boundDomain("domain-b", 0, GlobalDeviceAddress::rocm(2)),
            boundDomain("domain-c", 0, GlobalDeviceAddress::cpu(0)),
        };
        plan.routed_tiers = {
            RoutedExpertTier{
                .name = "alpha",
                .domain = "domain-a",
                .priority = -9,
            },
            RoutedExpertTier{
                .name = "beta",
                .domain = "domain-b",
                .priority = 7,
            },
            RoutedExpertTier{
                .name = "gamma",
                .domain = "domain-c",
                .priority = 101,
                .fallback = true,
            },
        };

        constexpr std::size_t kChunk = 4096;
        const MoEOverlayCapacityAdmissionPolicy policy{
            .materialize_migration_fabric = true,
            .shadow_slots_per_endpoint_layer = 1,
            .staging_capacity_bytes = kChunk,
            .distributed_transport = true,
            .overlay_world_size = 2,
        };
        const auto charges = MoEOverlayCapacityAdmission::transferStagingBOM(
            plan, /*world_size=*/2, policy);
        const auto bytesFor = [&](int rank, DeviceId device)
        {
            const auto found = std::find_if(
                charges.begin(), charges.end(), [&](const auto &charge)
                {
                    return charge.world_rank == rank &&
                           charge.device == device;
                });
            return found == charges.end() ? std::size_t{0} : found->bytes;
        };

        /*
         * Each GPU owns six local CPU-edge chunks plus six remote chunks.
         * Rank-zero host memory owns 12 CPU-edge, 24 heterogeneous-blob,
         * 12 remote-GPU, and nine global MPI-lane chunks. Relay rank one owns
         * only the same nine global MPI payload lanes.
         */
        EXPECT_EQ(bytesFor(0, DeviceId::cuda(0)), 12u * kChunk);
        EXPECT_EQ(bytesFor(0, DeviceId::rocm(2)), 12u * kChunk);
        EXPECT_EQ(bytesFor(0, DeviceId::cpu()), 57u * kChunk);
        EXPECT_EQ(bytesFor(1, DeviceId::cpu()), 9u * kChunk);
        EXPECT_EQ(charges.size(), 4u);
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         ContinuationShardsPreserveLocalTensorParallelIdentity)
    {
        RankExecutionPlan rank_plan;
        rank_plan.first_layer = 3;
        rank_plan.last_layer = 11;
        rank_plan.primary_device = GlobalDeviceAddress::cuda(0);
        rank_plan.local_tp_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };

        const auto shards =
            MoEOverlayLocalCapacityPlanner::continuationShards(
                rank_plan,
                /*builds_root_graph=*/true);
        ASSERT_EQ(shards.size(), 2u);
        EXPECT_EQ(shards[0].device, DeviceId::cuda(0));
        EXPECT_EQ(shards[0].shard_index, 0);
        EXPECT_EQ(shards[0].total_shards, 2);
        EXPECT_EQ(shards[0].first_layer, 3);
        EXPECT_EQ(shards[0].last_layer, 11);
        EXPECT_EQ(shards[1].device, DeviceId::cuda(1));
        EXPECT_EQ(shards[1].shard_index, 1);
        EXPECT_EQ(shards[1].total_shards, 2);
        EXPECT_TRUE(
            MoEOverlayLocalCapacityPlanner::continuationShards(
                rank_plan,
                /*builds_root_graph=*/false)
                .empty());
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         GroupsContinuationAndEndpointFixedBytesByBoundPhysicalResource)
    {
        const auto profile = smallModelProfile();

        RankExecutionPlan rank_plan;
        rank_plan.rank = 5;
        rank_plan.first_layer = 0;
        rank_plan.last_layer = 1;
        rank_plan.primary_device = GlobalDeviceAddress::cuda(0);
        rank_plan.runtime.batch_size = 1;
        rank_plan.runtime.max_seq_len = 32;
        rank_plan.runtime.kv_cache_precision = KVCachePrecision::FP16;

        RankInventory inventory;
        inventory.rank = 5;
        inventory.cpu_memory_bytes = 1024u * 1024u * 1024u;
        inventory.cpu.memory_bytes = 1024u * 1024u * 1024u;
        inventory.cpu.free_memory_bytes = 640u * 1024u * 1024u;
        inventory.gpus = {
            DeviceInfo{
                .type = DeviceType::CUDA,
                .local_device_id = 0,
                .memory_bytes = 2u * 1024u * 1024u * 1024u,
                .free_memory_bytes = 1536u * 1024u * 1024u,
                .compute_units = 82,
            },
        };

        MoERoutedExpertPlacementPlan overlay;
        overlay.enabled = true;
        overlay.topology = RoutedExpertPlacementTopology::TieredOverlay;
        overlay.continuation_domain = "domain-one";
        overlay.base_model_domain = "domain-one";
        overlay.shared_expert_domain = "domain-one";
        overlay.domains = {
            boundDomain(
                "domain-one", 5, GlobalDeviceAddress::cuda(0)),
            boundDomain(
                "domain-two", 5, GlobalDeviceAddress::cpu(0)),
        };
        overlay.routed_tiers = {
            RoutedExpertTier{
                .name = "shape",
                .domain = "domain-one",
                .priority = -1,
            },
            RoutedExpertTier{
                .name = "colour",
                .domain = "domain-two",
                .priority = 15,
                .fallback = true,
            },
        };

        const auto gpu_weight_load = testGPUWeightLoadCapacityInput();
        const auto result = MoEOverlayLocalCapacityPlanner::plan({
            .model_profile = &profile,
            .rank_plan = &rank_plan,
            .overlay_plan = &overlay,
            .rank_inventory = &inventory,
            .builds_root_graph = true,
            .require_host_memory_authority = true,
            .max_gpu_memory_bytes = 1024u * 1024u * 1024u,
            .max_cpu_memory_bytes = 512u * 1024u * 1024u,
            .resident_graph_rows = 8,
            .gpu_weight_load = gpu_weight_load,
        });
        ASSERT_EQ(result.fixed_memory_plan.devices.size(), 2u);
        ASSERT_EQ(result.physical_budgets.size(), 2u);

        const auto budgetFor = [&](DeviceId device)
            -> const MoEOverlayBoundPhysicalMemoryBudget *
        {
            const auto found = std::find_if(
                result.physical_budgets.begin(),
                result.physical_budgets.end(),
                [&](const auto &budget)
                { return budget.device == device; });
            return found == result.physical_budgets.end()
                       ? nullptr
                       : &*found;
        };
        const auto *cpu = budgetFor(DeviceId::cpu());
        const auto *cuda = budgetFor(DeviceId::cuda(0));
        ASSERT_NE(cpu, nullptr);
        ASSERT_NE(cuda, nullptr);
        EXPECT_EQ(cpu->world_rank, 5);
        EXPECT_EQ(cuda->world_rank, 5);
        EXPECT_EQ(cpu->usable_budget_bytes, 512u * 1024u * 1024u);
        EXPECT_EQ(cuda->usable_budget_bytes, 1024u * 1024u * 1024u);
        const auto expected_load_reserve = gpuWeightLoadMemoryBOM(
            /*planned_weight_bytes=*/0,
            gpu_weight_load.maximum_source_bytes,
            cuda->usable_budget_bytes,
            inventory.gpus.front().memory_bytes,
            gpu_weight_load.policy);
        EXPECT_EQ(
            cuda->additional_transfer_staging_bytes,
            expected_load_reserve.staging_bytes);
        EXPECT_EQ(
            cuda->safety_reserve_bytes,
            expected_load_reserve.safety_margin_bytes);
        /*
         * One serial participant owns the complete immutable route-family
         * ladder. Eight rows at top-k two retain capacities 1, 2, 4, 8, and
         * the exact terminal capacity 16, shared across all model layers.
         */
        constexpr std::size_t kExpectedCpuSparsePacketBytes =
            (1u + 2u + 4u + 8u + 16u) *
            (2u * 64u + 2u) * sizeof(float);
        EXPECT_EQ(cpu->fixed_bytes, kExpectedCpuSparsePacketBytes);
        EXPECT_GT(cuda->fixed_bytes, 0u);
        EXPECT_EQ(
            cuda->resource_id,
            MoEOverlayLocalCapacityPlanner::physicalResourceId(
                5, DeviceId::cuda(0)));
        for (const auto &device_plan : result.fixed_memory_plan.devices)
            EXPECT_EQ(device_plan.activation_seq_len, 8);
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         RelayHostAuthorityDoesNotInventAnExpertGraphWorkspace)
    {
        const auto profile = smallModelProfile();
        RankExecutionPlan rank_plan;
        rank_plan.rank = 1;
        rank_plan.runtime.batch_size = 1;
        rank_plan.runtime.max_seq_len = 32;
        rank_plan.runtime.kv_cache_precision = KVCachePrecision::FP16;

        RankInventory inventory;
        inventory.rank = 1;
        inventory.cpu_memory_bytes = 768u * 1024u * 1024u;
        inventory.cpu.memory_bytes = 768u * 1024u * 1024u;
        inventory.cpu.free_memory_bytes = 384u * 1024u * 1024u;

        MoERoutedExpertPlacementPlan overlay;
        overlay.enabled = true;
        overlay.topology = RoutedExpertPlacementTopology::TieredOverlay;
        overlay.continuation_domain = "elsewhere";
        overlay.base_model_domain = "elsewhere";
        overlay.shared_expert_domain = "elsewhere";
        overlay.domains = {boundDomain(
            "elsewhere", 0, GlobalDeviceAddress::cuda(0))};
        overlay.routed_tiers = {RoutedExpertTier{
            .name = "opaque",
            .domain = "elsewhere",
            .priority = 0,
            .fallback = true,
        }};

        const auto result = MoEOverlayLocalCapacityPlanner::plan({
            .model_profile = &profile,
            .rank_plan = &rank_plan,
            .overlay_plan = &overlay,
            .rank_inventory = &inventory,
            .builds_root_graph = false,
            .require_host_memory_authority = true,
        });
        EXPECT_TRUE(result.fixed_memory_plan.devices.empty());
        ASSERT_EQ(result.physical_budgets.size(), 1u);
        EXPECT_EQ(result.physical_budgets[0].device, DeviceId::cpu());
        EXPECT_EQ(result.physical_budgets[0].fixed_bytes, 0u);
        EXPECT_EQ(result.physical_budgets[0].safety_reserve_bytes, 0u);
        EXPECT_EQ(
            result.physical_budgets[0].usable_budget_bytes,
            384u * 1024u * 1024u);
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         AutomaticAdmissionAndLateLoaderUseTheSameByteEquation)
    {
        constexpr std::size_t kMiB = 1024u * 1024u;
        constexpr std::size_t kPersistentWeights = 73u * kMiB;
        constexpr std::size_t kNonWeightFixed = 19u * kMiB;
        constexpr std::size_t kTotalVram = 2ULL * 1024ULL * kMiB;
        const auto gpu_load = testGPUWeightLoadCapacityInput();
        const auto reserve = gpuWeightLoadMemoryBOM(
            /*planned_weight_bytes=*/0,
            gpu_load.maximum_source_bytes,
            /*free_vram_bytes=*/kTotalVram,
            kTotalVram,
            gpu_load.policy);
        const auto model_manifest = manifest(
            /*layers=*/1, native_vnni_formats::Q4_0);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        ASSERT_EQ(footprints.size(), 1u);
        const std::size_t live_expert_bytes =
            footprints.front().gpu_live_bytes;
        const std::size_t fixed_bytes =
            kPersistentWeights + kNonWeightFixed;
        const std::size_t initial_free =
            fixed_bytes + reserve.staging_bytes +
            reserve.safety_margin_bytes + live_expert_bytes;

        MoEOverlayCapacityResolverInput admission;
        admission.num_experts = 1;
        admission.layer_weight_manifest = model_manifest;
        admission.physical_budgets = {
            MoEOverlayPhysicalMemoryBudget{
                .resource_id = "same-equation-gpu",
                .device = DeviceId::cuda(0),
                .usable_budget_bytes = initial_free,
                .fixed_bytes = fixed_bytes,
                .transfer_staging_bytes = reserve.staging_bytes,
                .safety_reserve_bytes = reserve.safety_margin_bytes,
            },
        };
        admission.tiers = {
            tier(
                0,
                "opaque",
                0,
                true,
                MoEOverlayLiveQuotaMode::FixedPerLayer,
                {participant(
                    0, "same-equation-gpu", /*shadow_slots=*/0)},
                {1}),
        };
        const auto admitted =
            MoEOverlayCapacityResolver::resolve(admission);
        ASSERT_NE(admitted.resource("same-equation-gpu"), nullptr);
        EXPECT_EQ(
            admitted.resource("same-equation-gpu")->remaining_bytes,
            0u);

        /*
         * By load time the non-weight fixed state has already reduced the
         * backend's free-memory reading. The pool contains persistent dense
         * weights plus the admitted expert. Algebraically this is exactly the
         * same physical bill, not a similar heuristic.
         */
        const std::size_t late_free = initial_free - kNonWeightFixed;
        const auto late_load = gpuWeightLoadMemoryBOM(
            kPersistentWeights + live_expert_bytes,
            gpu_load.maximum_source_bytes,
            late_free,
            kTotalVram,
            gpu_load.policy);
        EXPECT_EQ(late_load.staging_bytes, reserve.staging_bytes);
        EXPECT_EQ(
            late_load.safety_margin_bytes,
            reserve.safety_margin_bytes);
        EXPECT_EQ(late_load.required_bytes, late_free);
        EXPECT_TRUE(late_load.fits());

        const auto one_byte_short = gpuWeightLoadMemoryBOM(
            kPersistentWeights + live_expert_bytes,
            gpu_load.maximum_source_bytes,
            late_free - 1,
            kTotalVram,
            gpu_load.policy);
        EXPECT_FALSE(one_byte_short.fits());
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         SmallerCapturedGraphCandidateCanRestoreCompleteExpertCoverage)
    {
        auto profile = smallModelProfile();
        profile.d_model = 4096;
        profile.d_ff = 4096;
        profile.n_heads = 32;
        profile.n_kv_heads = 8;
        profile.head_dim = 128;
        profile.max_seq_len = 4096;

        RankExecutionPlan rank_plan;
        rank_plan.rank = 0;
        rank_plan.first_layer = 0;
        rank_plan.last_layer = 1;
        rank_plan.primary_device = GlobalDeviceAddress::cuda(0);
        rank_plan.runtime.batch_size = 1;
        rank_plan.runtime.max_seq_len = 4096;
        rank_plan.runtime.kv_cache_precision = KVCachePrecision::FP16;

        constexpr std::size_t kLargeBudget =
            64ULL * 1024ULL * 1024ULL * 1024ULL;
        RankInventory inventory;
        inventory.rank = 0;
        inventory.cpu_memory_bytes = kLargeBudget;
        inventory.cpu.memory_bytes = kLargeBudget;
        inventory.cpu.free_memory_bytes = kLargeBudget;
        inventory.gpus = {
            DeviceInfo{
                .type = DeviceType::CUDA,
                .local_device_id = 0,
                .memory_bytes = kLargeBudget,
                .free_memory_bytes = kLargeBudget,
                .compute_units = 82,
            },
        };

        MoERoutedExpertPlacementPlan overlay;
        overlay.enabled = true;
        overlay.topology = RoutedExpertPlacementTopology::TieredOverlay;
        overlay.continuation_domain = "opaque-a";
        overlay.base_model_domain = "opaque-a";
        overlay.shared_expert_domain = "opaque-a";
        overlay.domains = {
            boundDomain(
                "opaque-a", 0, GlobalDeviceAddress::cuda(0)),
            boundDomain(
                "opaque-b", 0, GlobalDeviceAddress::cpu(0)),
        };
        overlay.routed_tiers = {
            RoutedExpertTier{
                .name = "not-a-temperature",
                .domain = "opaque-a",
                .priority = -9,
            },
            RoutedExpertTier{
                .name = "also-opaque",
                .domain = "opaque-b",
                .priority = 23,
                .fallback = true,
            },
        };

        const auto planForRows = [&](int rows)
        {
            return MoEOverlayLocalCapacityPlanner::plan({
                .model_profile = &profile,
                .rank_plan = &rank_plan,
                .overlay_plan = &overlay,
                .rank_inventory = &inventory,
                .builds_root_graph = true,
                .require_host_memory_authority = true,
                .resident_graph_rows = rows,
                .gpu_weight_load = testGPUWeightLoadCapacityInput(),
            });
        };
        auto small = planForRows(8);
        auto large = planForRows(4096);

        const auto model_manifest = manifest(
            profile.n_layers, native_vnni_formats::Q4_1);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        ASSERT_EQ(footprints.size(), 2u);

        const auto budgetFor = [](
            std::vector<MoEOverlayBoundPhysicalMemoryBudget> &budgets,
            DeviceId device)
            -> MoEOverlayBoundPhysicalMemoryBudget &
        {
            const auto found = std::find_if(
                budgets.begin(), budgets.end(), [&](const auto &budget)
                { return budget.device == device; });
            if (found == budgets.end())
                throw std::logic_error("test physical budget is missing");
            return *found;
        };

        const auto &small_gpu = budgetFor(
            small.physical_budgets, DeviceId::cuda(0));
        const auto &small_cpu = budgetFor(
            small.physical_budgets, DeviceId::cpu());
        const std::size_t gpu_limit =
            small_gpu.fixed_bytes +
            small_gpu.additional_transfer_staging_bytes +
            small_gpu.safety_reserve_bytes +
            footprints[0].gpu_live_bytes + footprints[1].gpu_live_bytes;
        const std::size_t cpu_limit =
            small_cpu.fixed_bytes + small_cpu.safety_reserve_bytes +
            3 * footprints[0].cpu_live_bytes +
            3 * footprints[1].cpu_live_bytes;
        EXPECT_GT(
            budgetFor(large.physical_budgets, DeviceId::cuda(0)).fixed_bytes,
            gpu_limit)
            << "the adversarial budget must distinguish the two graph shapes";

        for (auto *budgets : {
                 &small.physical_budgets,
                 &large.physical_budgets})
        {
            budgetFor(*budgets, DeviceId::cuda(0)).usable_budget_bytes =
                gpu_limit;
            budgetFor(*budgets, DeviceId::cpu()).usable_budget_bytes =
                cpu_limit;
        }

        const MoEOverlayCapacityAdmissionPolicy no_migration{
            .materialize_migration_fabric = false,
            .shadow_slots_per_endpoint_layer = 0,
            .staging_capacity_bytes = 0,
            .distributed_transport = false,
            .overlay_world_size = 1,
        };
        const auto small_input =
            MoEOverlayCapacityAdmission::buildResolverInput(
                overlay,
                profile.expert_count,
                model_manifest,
                small.physical_budgets,
                no_migration);
        const auto admitted =
            MoEOverlayCapacityResolver::resolve(small_input);
        ASSERT_NE(admitted.tier(0), nullptr);
        ASSERT_NE(admitted.tier(1), nullptr);
        EXPECT_EQ(
            admitted.tier(0)->live_experts_per_layer,
            (std::vector<int>{1, 1}));
        EXPECT_EQ(
            admitted.tier(1)->live_experts_per_layer,
            (std::vector<int>{3, 3}));

        const auto large_input =
            MoEOverlayCapacityAdmission::buildResolverInput(
                overlay,
                profile.expert_count,
                model_manifest,
                large.physical_budgets,
                no_migration);
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(large_input),
            std::invalid_argument);
    }

    TEST(MoEOverlayCapacityResolver, SharedPhysicalResourceAggregatesEveryTierCopy)
    {
        const auto model_manifest = manifest(1, native_vnni_formats::Q6_K);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        MoEOverlayCapacityResolverInput input;
        input.num_experts = 2;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget(
                "shared-gpu",
                DeviceId::cuda(0),
                footprints,
                {2},
                /*shadow_copies_per_layer=*/2),
        };
        input.tiers = {
            tier(0, "hot", 0, false, MoEOverlayLiveQuotaMode::FixedPerLayer,
                 {participant(0, "shared-gpu")}, {1}),
            tier(1, "cold", 1, true, MoEOverlayLiveQuotaMode::FixedPerLayer,
                 {participant(1, "shared-gpu")}, {1}),
        };

        const auto result = MoEOverlayCapacityResolver::resolve(input);
        const auto *resource = result.resource("shared-gpu");
        ASSERT_NE(resource, nullptr);
        EXPECT_EQ(resource->live_copies_per_layer, (std::vector<int>{2}));
        EXPECT_EQ(resource->shadow_copies_per_layer, (std::vector<int>{2}));
        EXPECT_EQ(resource->remaining_bytes, 0u);

        --input.physical_budgets.front().usable_budget_bytes;
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);
    }

    TEST(MoEOverlayCapacityResolver,
         CapacityFailureReportsExactLimitingPhysicalBom)
    {
        const auto model_manifest = manifest(1, native_vnni_formats::Q4_0);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        ASSERT_EQ(footprints.size(), 1u);

        MoEOverlayCapacityResolverInput input;
        input.num_experts = 2;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget(
                "diagnostic-gpu",
                DeviceId::rocm(3),
                footprints,
                {1},
                /*shadow_copies_per_layer=*/0),
        };
        input.tiers = {
            tier(
                0,
                "opaque-fallback",
                17,
                true,
                MoEOverlayLiveQuotaMode::Automatic,
                {participant(0, "diagnostic-gpu", /*shadow_slots=*/0)}),
        };

        try
        {
            (void)MoEOverlayCapacityResolver::resolve(input);
            FAIL() << "under-provisioned fallback unexpectedly admitted";
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("tier 'opaque-fallback'"), std::string::npos);
            EXPECT_NE(message.find("layer 0"), std::string::npos);
            EXPECT_NE(
                message.find("resource='diagnostic-gpu' device=ROCm:3"),
                std::string::npos);
            EXPECT_NE(
                message.find(
                    "requires_additional_bytes=" +
                    std::to_string(footprints[0].gpu_live_bytes)),
                std::string::npos);
            EXPECT_NE(message.find("remaining_bytes=0"), std::string::npos);
            EXPECT_NE(message.find("fixed_bytes=113"), std::string::npos);
            EXPECT_NE(message.find("staging_bytes=257"), std::string::npos);
            EXPECT_NE(
                message.find("safety_reserve_bytes=509"),
                std::string::npos);
            EXPECT_NE(message.find("shadow_bytes=0"), std::string::npos);
            EXPECT_NE(
                message.find(
                    "live_expert_bytes=" +
                    std::to_string(footprints[0].gpu_live_bytes)),
                std::string::npos);
        }
    }

    TEST(MoEOverlayCapacityResolver, ReplicatedTierChargesEveryParticipantCopy)
    {
        const auto model_manifest = manifest(1, native_vnni_formats::Q5_1);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        MoEOverlayCapacityResolverInput input;
        input.num_experts = 3;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget("cuda-a", DeviceId::cuda(0), footprints, {2}, 1),
            exactBudget("cuda-b", DeviceId::cuda(1), footprints, {2}, 1),
            exactBudget("cpu-cold", DeviceId::cpu(), footprints, {1}, 1),
        };
        input.tiers = {
            tier(
                0,
                "replicated-hot",
                0,
                false,
                MoEOverlayLiveQuotaMode::FixedPerLayer,
                {participant(0, "cuda-a"), participant(1, "cuda-b")},
                {2},
                MoEOverlayTierCopyPolicy::Replicated),
            tier(1, "cold", 1, true, MoEOverlayLiveQuotaMode::FixedPerLayer,
                 {participant(2, "cpu-cold")}, {1}),
        };

        const auto replicated = MoEOverlayCapacityResolver::resolve(input);
        ASSERT_NE(replicated.tier(0), nullptr);
        EXPECT_EQ(replicated.tier(0)->participant_live_copies,
                  (std::vector<std::vector<int>>{{2}, {2}}));
        EXPECT_EQ(replicated.resource("cuda-a")->live_copies_per_layer,
                  (std::vector<int>{2}));
        EXPECT_EQ(replicated.resource("cuda-b")->live_copies_per_layer,
                  (std::vector<int>{2}));

        input.tiers.front().copy_policy =
            MoEOverlayTierCopyPolicy::Apportioned;
        const auto apportioned = MoEOverlayCapacityResolver::resolve(input);
        EXPECT_EQ(apportioned.tier(0)->participant_live_copies,
                  (std::vector<std::vector<int>>{{1}, {1}}));
        EXPECT_EQ(
            apportioned.resource("cuda-a")->remaining_bytes,
            footprints.front().gpu_live_bytes);
        EXPECT_EQ(
            apportioned.resource("cuda-b")->remaining_bytes,
            footprints.front().gpu_live_bytes);
    }

    TEST(MoEOverlayCapacityResolver, EmptyTierStillPaysForMaterializedShadowBanks)
    {
        const auto model_manifest = manifest(1, native_vnni_formats::IQ2_S);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        MoEOverlayCapacityResolverInput input;
        input.num_experts = 1;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget("hot", DeviceId::cuda(0), footprints, {1}, 1),
            exactBudget("empty-cold", DeviceId::cpu(), footprints, {0}, 1),
        };
        input.tiers = {
            tier(0, "hot", 0, false, MoEOverlayLiveQuotaMode::FixedPerLayer,
                 {participant(0, "hot")}, {1}),
            tier(1, "empty-cold", 1, true,
                 MoEOverlayLiveQuotaMode::FixedPerLayer,
                 {participant(1, "empty-cold")}, {0}),
        };

        const auto result = MoEOverlayCapacityResolver::resolve(input);
        ASSERT_NE(result.tier(1), nullptr);
        EXPECT_FALSE(result.tier(1)->has_live_residency);
        const auto *cold = result.resource("empty-cold");
        ASSERT_NE(cold, nullptr);
        EXPECT_EQ(cold->live_expert_bytes, 0u);
        EXPECT_EQ(cold->shadow_bytes, footprints.front().cpu_shadow_bytes);
        EXPECT_EQ(cold->shadow_copies_per_layer, (std::vector<int>{1}));
        EXPECT_EQ(cold->remaining_bytes, 0u);

        --input.physical_budgets.back().usable_budget_bytes;
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);
    }

    TEST(MoEOverlayCapacityResolver, FallbackCoverageFailsClosedWhenOneExpertDoesNotFit)
    {
        const auto model_manifest = manifest(1, native_vnni_formats::Q8_K);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        MoEOverlayCapacityResolverInput input;
        input.num_experts = 3;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget("cold", DeviceId::cpu(), footprints, {2}, 1),
        };
        input.tiers = {
            tier(0, "cold", 0, true, MoEOverlayLiveQuotaMode::Automatic,
                 {participant(0, "cold")}),
        };

        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);
    }

    TEST(MoEOverlayCapacityResolver, AutomaticUpperBoundsLimitHotTierAndFallbackCoverage)
    {
        const auto model_manifest = manifest(2, native_vnni_formats::Q4_1);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        MoEOverlayCapacityResolverInput input;
        input.num_experts = 4;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget("hot", DeviceId::cuda(0), footprints, {4, 4}, 1),
            exactBudget("cold", DeviceId::cpu(), footprints, {3, 2}, 1),
        };
        input.tiers = {
            tier(
                0,
                "hot",
                0,
                false,
                MoEOverlayLiveQuotaMode::Automatic,
                {participant(0, "hot")},
                {},
                MoEOverlayTierCopyPolicy::Apportioned,
                {1, 2}),
            tier(
                1,
                "cold",
                1,
                true,
                MoEOverlayLiveQuotaMode::Automatic,
                {participant(1, "cold")},
                {},
                MoEOverlayTierCopyPolicy::Apportioned,
                {3, 2}),
        };

        const auto result = MoEOverlayCapacityResolver::resolve(input);
        EXPECT_EQ(result.tier(0)->live_experts_per_layer,
                  (std::vector<int>{1, 2}));
        EXPECT_EQ(result.tier(1)->live_experts_per_layer,
                  (std::vector<int>{3, 2}));

        input.tiers.back().max_live_experts_per_layer = {2, 2};
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);

        input.tiers.front().quota_mode =
            MoEOverlayLiveQuotaMode::FixedPerLayer;
        input.tiers.front().fixed_live_experts_per_layer = {2, 2};
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);
    }

    TEST(MoEOverlayCapacityResolver, MalformedTopologyAndOverflowAreRejected)
    {
        const auto model_manifest = manifest(1, native_vnni_formats::Q4_0);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        MoEOverlayCapacityResolverInput input;
        input.num_experts = 1;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget("hot", DeviceId::cuda(0), footprints, {0}, 1),
            exactBudget("cold", DeviceId::cpu(), footprints, {1}, 1),
        };
        input.tiers = {
            tier(0, "hot", 0, false, MoEOverlayLiveQuotaMode::FixedPerLayer,
                 {participant(0, "hot")}, {0}),
            tier(1, "cold", 0, true, MoEOverlayLiveQuotaMode::FixedPerLayer,
                 {participant(1, "cold")}, {1}),
        };
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);

        input.tiers[1].priority = 1;
        input.tiers[0].participants[0].resource_id = "absent";
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);

        input.tiers[0].participants[0].resource_id = "hot";
        input.tiers[0].participants[0].shadow_slots_per_layer =
            std::numeric_limits<std::size_t>::max();
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::overflow_error);
        EXPECT_THROW(
            (void)footprints.front().liveBytes(DeviceId::invalid()),
            std::invalid_argument);
    }
} // namespace llaminar2
