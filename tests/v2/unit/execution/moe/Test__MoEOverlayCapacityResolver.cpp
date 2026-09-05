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
#include "planning/CapturedGraphMemoryEstimator.h"
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
    TEST(Test__MoEOverlayCapacityResolver,
         RuntimeGPUObservationUpdatesCUDAAndROCmThroughOneAuthority)
    {
        RankInventory inventory;
        inventory.rank = 7;
        inventory.gpus = {
            DeviceInfo{
                .type = DeviceType::CUDA,
                .local_device_id = 2,
                .memory_bytes = 24'000,
                .free_memory_bytes = 23'000,
            },
            DeviceInfo{
                .type = DeviceType::ROCm,
                .local_device_id = 5,
                .memory_bytes = 32'000,
                .free_memory_bytes = 31'000,
            },
        };

        installMoEOverlayRuntimeGPUCapacityObservation(
            inventory, DeviceId::cuda(2), 24'000, 21'500);
        installMoEOverlayRuntimeGPUCapacityObservation(
            inventory, DeviceId::rocm(5), 32'000, 27'250);

        ASSERT_EQ(inventory.gpus.size(), 2u);
        EXPECT_EQ(inventory.gpus[0].memory_bytes, 24'000u);
        EXPECT_EQ(inventory.gpus[0].free_memory_bytes, 21'500u);
        EXPECT_EQ(inventory.gpus[1].memory_bytes, 32'000u);
        EXPECT_EQ(inventory.gpus[1].free_memory_bytes, 27'250u);
    }

    TEST(Test__MoEOverlayCapacityResolver,
         RuntimeGPUObservationRejectsInvalidOrUnownedDevices)
    {
        RankInventory inventory;
        inventory.rank = 3;
        inventory.gpus = {DeviceInfo{
            .type = DeviceType::CUDA,
            .local_device_id = 0,
            .memory_bytes = 24'000,
            .free_memory_bytes = 23'000,
        }};

        EXPECT_THROW(
            installMoEOverlayRuntimeGPUCapacityObservation(
                inventory, DeviceId::cpu(), 64'000, 32'000),
            std::invalid_argument);
        EXPECT_THROW(
            installMoEOverlayRuntimeGPUCapacityObservation(
                inventory, DeviceId::cuda(0), 24'000, 24'001),
            std::invalid_argument);
        EXPECT_THROW(
            installMoEOverlayRuntimeGPUCapacityObservation(
                inventory, DeviceId::rocm(0), 32'000, 31'000),
            std::invalid_argument);
    }

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

        /** @return Certified base budget before routed-expert copies are added. */
        [[nodiscard]] MoEOverlayPhysicalMemoryBudget physicalBudget(
            std::string resource_id,
            DeviceId device,
            std::size_t usable_budget_bytes,
            std::size_t fixed_bytes,
            std::size_t staging_bytes)
        {
            PhysicalMemoryBOMBuilder builder({
                .world_rank = -1,
                .device = device,
                .total_bytes = usable_budget_bytes,
                .admission_available_bytes = usable_budget_bytes,
            });
            builder
                .add(PhysicalMemoryOwner::ActivationArena, fixed_bytes)
                .add(
                    PhysicalMemoryOwner::ExpertMigrationStaging,
                    staging_bytes);
            return MoEOverlayPhysicalMemoryBudget(
                std::move(resource_id),
                PhysicalMemoryAdmissionCertificate(builder.build()));
        }

        /** @return Exact physical budget for specified live and shadow copies. */
        [[nodiscard]] MoEOverlayPhysicalMemoryBudget exactBudget(
            std::string resource_id,
            DeviceId device,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::vector<int> &live_copies_per_layer,
            int shadow_copies_per_layer,
            std::size_t fixed_bytes = 113,
            std::size_t staging_bytes = 257)
        {
            if (footprints.size() != live_copies_per_layer.size() ||
                shadow_copies_per_layer < 0)
            {
                throw std::invalid_argument("invalid test budget geometry");
            }
            std::size_t usable_budget_bytes =
                fixed_bytes + staging_bytes;
            for (std::size_t layer_idx = 0;
                 layer_idx < footprints.size();
                 ++layer_idx)
            {
                usable_budget_bytes +=
                    static_cast<std::size_t>(live_copies_per_layer[layer_idx]) *
                    footprints[layer_idx].liveBytes(device);
                usable_budget_bytes +=
                    static_cast<std::size_t>(shadow_copies_per_layer) *
                    footprints[layer_idx].shadowBytes(device);
            }
            return physicalBudget(
                std::move(resource_id),
                device,
                usable_budget_bytes,
                fixed_bytes,
                staging_bytes);
        }

        /** @return Rank/device-qualified production-admission form of a test budget. */
        [[nodiscard]] MoEOverlayBoundPhysicalMemoryBudget boundBudget(
            int world_rank,
            const MoEOverlayPhysicalMemoryBudget &budget)
        {
            const auto &source = budget.certificate().bom();
            PhysicalMemoryBOMBuilder builder(
                PhysicalMemoryResource{
                    .world_rank = world_rank,
                    .device = budget.device(),
                    .total_bytes = source.resource().total_bytes,
                    .admission_available_bytes =
                        source.resource().admission_available_bytes,
                },
                source);
            return MoEOverlayBoundPhysicalMemoryBudget(
                budget.resourceId(),
                PhysicalMemoryAdmissionCertificate(builder.build()));
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
            profile.architecture = "qwen35moe";
            profile.n_layers = 2;
            profile.d_model = 64;
            profile.d_ff = 128;
            profile.n_heads = 4;
            profile.n_kv_heads = 2;
            profile.head_dim = 16;
            profile.vocab_size = 256;
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
                .maximum_concurrent_shadow_slots = shadow_slots,
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

            const auto directory_profile =
                DeviceMoETransferSlotDirectory::
                    profileForLayerWeightManifest(model_manifest);
            ASSERT_EQ(directory_profile.allocation_specs.size(), 3u);
            const auto reusable = reusableDeviceVnniAllocationFormat(
                *source.metadata);
            std::size_t expected_slot_bytes = 0u;
            for (std::size_t projection_index = 0;
                 projection_index < directory_profile.allocation_specs.size();
                 ++projection_index)
            {
                const auto &allocation =
                    directory_profile
                        .allocation_specs[projection_index];
                EXPECT_EQ(
                    allocation.payload_bytes_per_block,
                    reusable.payload_bytes_per_block);
                EXPECT_EQ(
                    allocation.is_asymmetric,
                    reusable.has_mins);
                EXPECT_EQ(
                    allocation.has_emins,
                    reusable.has_emins);
                EXPECT_EQ(
                    allocation.codebook_id,
                    canonicalDeviceVnniCodebookId(
                        source.metadata->codebook_id));
                const auto &projection =
                    model_manifest.front()
                        .projections[projection_index];
                expected_slot_bytes += expectedGpuProjectionBytes(
                    projection,
                    reusable.payload_bytes_per_block,
                    reusable.has_mins,
                    reusable.has_emins);
            }

            const auto capacity =
                DeviceMoETransferSlotDirectory::planBufferedCapacity(
                    /*requested_active_slots=*/1u,
                    /*transfer_wave_slots=*/1u,
                    /*transfer_buffer_count=*/2u);
            const auto directory_bom =
                DeviceMoETransferSlotDirectory::allocationBOM(
                    capacity,
                    directory_profile);
            EXPECT_EQ(
                directory_bom.payload_bytes,
                expected_slot_bytes * capacity.total_slots);
            EXPECT_EQ(
                directory_bom.descriptor_bytes,
                sizeof(DeviceMoEExpertDirectoryEntry) *
                    capacity.total_slots * 2u);
            EXPECT_EQ(
                directory_bom.total_bytes,
                directory_bom.payload_bytes +
                    directory_bom.descriptor_bytes);

            const DeviceMoERebalanceWorkspaceCapacity workspace_capacity{
                .num_layers = 1u,
                .num_experts = 4u,
                .participant_count = 2u,
                .layer_window_count = 1u,
                .layer_wave_count = 1u,
                .max_hot_replicas_per_participant = 1u,
                .flags = static_cast<std::uint32_t>(
                    DeviceMoERebalanceFlags::DeferRuntimeApply),
                .local_transfer_slot_count = capacity.total_slots,
                .collective_payload_slot_capacity = 1u,
                .phase = DeviceMoERebalanceStagePhase::PlanCopyApply,
                .transfer_mode =
                    DeviceMoERebalanceTransferMode::CompactTransferSlots,
            };
            const auto payload_slot_bytes =
                DeviceMoERebalanceWorkspaceContract::
                    collectivePayloadSlotBytes(
                        directory_profile.max_wire_payload_bytes);
            const DeviceMoERebalanceWorkspaceBinding workspace_binding{
                .capacity = workspace_capacity,
                .collective_payload_slot_bytes = payload_slot_bytes,
                .workspace_suffix = "all_codebooks",
            };
            const auto workspace_requirements =
                DeviceMoERebalanceWorkspaceContract::requirements(
                    workspace_binding);
            ASSERT_EQ(workspace_requirements.buffers.size(), 19u);
            const auto *local_payload = workspace_requirements.find(
                "moe_rebalance_local_transfer_payload_all_codebooks");
            const auto *gathered_payload = workspace_requirements.find(
                "moe_rebalance_gathered_transfer_payload_all_codebooks");
            ASSERT_NE(local_payload, nullptr);
            ASSERT_NE(gathered_payload, nullptr);
            EXPECT_EQ(local_payload->size_bytes, payload_slot_bytes);
            EXPECT_EQ(
                gathered_payload->size_bytes,
                payload_slot_bytes * workspace_capacity.participant_count);
            EXPECT_EQ(payload_slot_bytes % kGpuAlignment, 0u);
            EXPECT_GE(
                DeviceMoERebalanceWorkspaceContract::allocationBytes(
                    workspace_binding),
                DeviceMoERebalanceWorkspaceContract::logicalBytes(
                    workspace_binding));
        }
    }

    TEST(MoEOverlayCapacityAdmission,
         HomogeneousGpuDirectoryIsAdmittedExactlyOnEveryParticipant)
    {
        constexpr int kLayers = 40;
        constexpr int kExperts = 256;
        constexpr std::uint32_t kHotReplicasPerParticipant = 25u;
        const auto model_manifest =
            manifest(kLayers, native_vnni_formats::Q4_K);
        const auto profile =
            DeviceMoETransferSlotDirectory::
                profileForLayerWeightManifest(model_manifest);
        DeviceMoERebalanceConfig rebalance;
        rebalance.num_layers = kLayers;
        rebalance.max_hot_replicas_per_participant =
            kHotReplicasPerParticipant;
        const auto capacity =
            DeviceMoETransferSlotDirectory::planRuntimeCapacity(
                rebalance,
                /*minimum_active_slots=*/0u,
                /*transfer_wave_slots=*/2u,
                /*transfer_buffer_count=*/2u);
        ASSERT_EQ(capacity.active_slots, 1000u);
        ASSERT_EQ(capacity.staging_slots, 4u);
        ASSERT_EQ(capacity.total_slots, 1004u);
        const auto expected_directory =
            DeviceMoETransferSlotDirectory::allocationBOM(
                capacity,
                profile);
        const DeviceMoERebalanceWorkspaceCapacity workspace_capacity{
            .num_layers = kLayers,
            .num_experts = kExperts,
            .participant_count = 2u,
            .layer_window_count = kLayers - 1u,
            .layer_wave_count = 4u,
            .max_hot_replicas_per_participant =
                kHotReplicasPerParticipant,
            .flags = static_cast<std::uint32_t>(
                         DeviceMoERebalanceFlags::DeferRuntimeApply) |
                     static_cast<std::uint32_t>(
                         DeviceMoERebalanceFlags::PlanMissingArrivals),
            .local_transfer_slot_count = capacity.total_slots,
            .collective_payload_slot_capacity = 1u,
            .phase = DeviceMoERebalanceStagePhase::PlanCopyApply,
            .transfer_mode =
                DeviceMoERebalanceTransferMode::CompactTransferSlots,
        };
        const DeviceMoERebalanceWorkspaceBinding workspace_binding{
            .capacity = workspace_capacity,
            .collective_payload_slot_bytes =
                DeviceMoERebalanceWorkspaceContract::
                    collectivePayloadSlotBytes(
                        profile.max_wire_payload_bytes),
            .workspace_suffix = "expected_admission",
        };
        const std::size_t expected_workspace =
            DeviceMoERebalanceWorkspaceContract::allocationBytes(
                workspace_binding);

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::SingleDomain;
        plan.authority_execution =
            MoEOverlayAuthorityExecutionKind::DeviceResident;
        plan.continuation_domain = "rank-local-gpus";
        plan.base_model_domain = "rank-local-gpus";
        plan.shared_expert_domain = "rank-local-gpus";
        RoutedExpertDomain domain;
        domain.name = "rank-local-gpus";
        domain.scope = ExecutionDomainScope::RANK_LOCAL;
        domain.backend = CollectiveBackendType::RCCL;
        domain.participants = {
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::rocm(1),
        };
        domain.world_ranks = {0, 0};
        domain.owner_rank = 0;
        domain.routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        plan.domains = {std::move(domain)};
        plan.routed_tiers = {RoutedExpertTier{
            .name = "priority-zero",
            .domain = "rank-local-gpus",
            .priority = 0,
            .fallback = true,
        }};

        EXPECT_EQ(
            MoEOverlayCapacityAdmission::selectMigrationStorage(
                plan, MoERebalanceRuntimeMode::Dynamic),
            MoEOverlayMigrationStorageKind::DeviceTransferDirectory);
        EXPECT_EQ(
            MoEOverlayCapacityAdmission::selectMigrationStorage(
                plan, MoERebalanceRuntimeMode::Off),
            MoEOverlayMigrationStorageKind::Disabled);
        auto cross_rank = plan;
        cross_rank.domains.front().world_ranks = {0, 1};
        EXPECT_EQ(
            MoEOverlayCapacityAdmission::selectMigrationStorage(
                cross_rank, MoERebalanceRuntimeMode::Dynamic),
            MoEOverlayMigrationStorageKind::PhysicalResidencyFabric);
        auto heterogeneous = plan;
        heterogeneous.domains.front().participants.back() =
            GlobalDeviceAddress::cuda(0);
        EXPECT_EQ(
            MoEOverlayCapacityAdmission::selectMigrationStorage(
                heterogeneous, MoERebalanceRuntimeMode::Dynamic),
            MoEOverlayMigrationStorageKind::PhysicalResidencyFabric);

        constexpr std::size_t kAvailable =
            std::size_t{8} * 1024u * 1024u * 1024u;
        const auto first = physicalBudget(
            "rank0-rocm0",
            DeviceId::rocm(0),
            kAvailable,
            /*fixed_bytes=*/4096u,
            /*staging_bytes=*/0u);
        const auto second = physicalBudget(
            "rank0-rocm1",
            DeviceId::rocm(1),
            kAvailable,
            /*fixed_bytes=*/4096u,
            /*staging_bytes=*/0u);
        const MoEOverlayCapacityAdmissionPolicy policy{
            .migration_storage =
                MoEOverlayMigrationStorageKind::DeviceTransferDirectory,
            .device_transfer_directory_capacity = capacity,
            .device_rebalance_workspace_capacity = workspace_capacity,
            .overlay_world_size = 1,
        };

        const auto input =
            MoEOverlayCapacityAdmission::buildResolverInput(
                plan,
                kExperts,
                model_manifest,
                {boundBudget(0, first), boundBudget(0, second)},
                policy);
        ASSERT_EQ(input.physical_budgets.size(), 2u);
        for (const auto &budget : input.physical_budgets)
        {
            EXPECT_EQ(
                budget.certificate().bom().bytes(
                    PhysicalMemoryOwner::ExpertMigrationStaging),
                expected_directory.total_bytes);
            EXPECT_EQ(
                budget.certificate().bom().bytes(
                    PhysicalMemoryOwner::ExecutionWorkspace),
                expected_workspace);
        }
        ASSERT_EQ(input.tiers.size(), 1u);
        ASSERT_EQ(input.tiers.front().participants.size(), 2u);
        for (const auto &participant : input.tiers.front().participants)
        {
            EXPECT_EQ(participant.shadow_slots_per_layer, 0u);
            EXPECT_EQ(participant.maximum_concurrent_shadow_slots, 0u);
        }
        EXPECT_TRUE(
            MoEOverlayCapacityAdmission::transferStagingBOM(
                plan, 1, policy)
                .empty());

        auto incomplete_policy = policy;
        incomplete_policy.device_rebalance_workspace_capacity.reset();
        EXPECT_THROW(
            (void)MoEOverlayCapacityAdmission::buildResolverInput(
                plan,
                kExperts,
                model_manifest,
                {boundBudget(0, first), boundBudget(0, second)},
                incomplete_policy),
            std::invalid_argument)
            << "A graph-owned transfer directory without its maintenance workspace must fail admission.";
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
        ASSERT_NE(result.physical_memory_admission, nullptr);
        EXPECT_EQ(
            result.physical_memory_admission->plan().resources().size(),
            input.physical_budgets.size());

        for (const auto &budget : input.physical_budgets)
        {
            const auto *resource = result.resource(budget.resourceId());
            ASSERT_NE(resource, nullptr) << budget.resourceId();
            EXPECT_EQ(
                resource->authority().get(),
                result.physical_memory_admission.get());
            EXPECT_EQ(
                &resource->bom(),
                result.physical_memory_admission->plan().find({
                    .world_rank = -1,
                    .device = budget.device(),
                }));
            EXPECT_EQ(resource->usedBytes(), budget.usableBudgetBytes());
            EXPECT_EQ(resource->remainingBytes(), 0u);
            EXPECT_EQ(resource->shadow_arrival_capacity_per_layer,
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

    TEST(MoEOverlayCapacityResolver,
         MigrationSourcesSeedEveryParticipantBeforeStrictPriorityFill)
    {
        constexpr int kExperts = 8;
        const auto model_manifest = manifest(
            1, native_vnni_formats::Q4_K);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);

        MoEOverlayCapacityResolverInput input;
        input.num_experts = kExperts;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {
            exactBudget("priority-0-a", DeviceId::cuda(0), footprints, {3}, 1),
            exactBudget("priority-0-b", DeviceId::cuda(1), footprints, {3}, 1),
            exactBudget("priority-8-a", DeviceId::rocm(0), footprints, {2}, 1),
            exactBudget("priority-8-b", DeviceId::rocm(1), footprints, {2}, 1),
            exactBudget("priority-19", DeviceId::cpu(), footprints, {1}, 1),
        };
        input.tiers = {
            tier(
                0,
                "opaque-zero",
                0,
                false,
                MoEOverlayLiveQuotaMode::Automatic,
                {
                    participant(0, "priority-0-a"),
                    participant(1, "priority-0-b"),
                }),
            tier(
                1,
                "opaque-eight",
                8,
                false,
                MoEOverlayLiveQuotaMode::Automatic,
                {
                    participant(2, "priority-8-a"),
                    participant(3, "priority-8-b"),
                },
                {},
                MoEOverlayTierCopyPolicy::Replicated),
            tier(
                2,
                "opaque-nineteen",
                19,
                true,
                MoEOverlayLiveQuotaMode::Automatic,
                {participant(4, "priority-19")}),
        };

        const auto priority_only =
            MoEOverlayCapacityResolver::resolve(input);
        EXPECT_EQ(
            priority_only.tier(0)->live_experts_per_layer,
            (std::vector<int>{6}));
        EXPECT_EQ(
            priority_only.tier(1)->live_experts_per_layer,
            (std::vector<int>{2}));
        EXPECT_EQ(
            priority_only.tier(2)->live_experts_per_layer,
            (std::vector<int>{0}));

        input.initial_residency_policy =
            MoEOverlayInitialResidencyPolicy::
                MigrationSourcePerParticipant;
        const auto seeded = MoEOverlayCapacityResolver::resolve(input);
        ASSERT_NE(seeded.tier(0), nullptr);
        ASSERT_NE(seeded.tier(1), nullptr);
        ASSERT_NE(seeded.tier(2), nullptr);
        EXPECT_EQ(
            seeded.tier(0)->live_experts_per_layer,
            (std::vector<int>{6}));
        EXPECT_EQ(
            seeded.tier(0)->participant_live_copies,
            (std::vector<std::vector<int>>{{3}, {3}}));
        EXPECT_EQ(
            seeded.tier(1)->live_experts_per_layer,
            (std::vector<int>{1}));
        EXPECT_EQ(
            seeded.tier(1)->participant_live_copies,
            (std::vector<std::vector<int>>{{1}, {1}}));
        EXPECT_EQ(
            seeded.tier(2)->live_experts_per_layer,
            (std::vector<int>{1}));
        EXPECT_EQ(
            seeded.tier(2)->participant_live_copies,
            (std::vector<std::vector<int>>{{1}}));

        input.tiers.front().quota_mode =
            MoEOverlayLiveQuotaMode::FixedPerLayer;
        input.tiers.front().fixed_live_experts_per_layer = {1};
        try
        {
            (void)MoEOverlayCapacityResolver::resolve(input);
            FAIL() << "under-seeded fixed migration tier unexpectedly admitted";
        }
        catch (const std::invalid_argument &error)
        {
            const std::string message = error.what();
            EXPECT_NE(message.find("fixed tier 'opaque-zero'"),
                      std::string::npos);
            EXPECT_NE(message.find("requires 2 to seed every participant"),
                      std::string::npos);
        }
    }

    TEST(MoEOverlayCapacityAdmission,
         MigrationFabricSelectsInitialSourceResidencyPolicy)
    {
        const auto model_manifest = manifest(
            1, native_vnni_formats::Q4_0);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        const auto cuda_budget = exactBudget(
            "rank0-cuda0", DeviceId::cuda(0), footprints, {1}, 1);
        const auto cpu_budget = exactBudget(
            "rank0-cpu", DeviceId::cpu(), footprints, {1}, 1);

        MoERoutedExpertPlacementPlan plan;
        plan.enabled = true;
        plan.topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan.continuation_domain = "domain-0";
        plan.base_model_domain = "domain-0";
        plan.shared_expert_domain = "domain-0";
        plan.domains = {
            boundDomain("domain-0", 0, GlobalDeviceAddress::cuda(0)),
            boundDomain("domain-1", 0, GlobalDeviceAddress::cpu(0)),
        };
        plan.routed_tiers = {
            RoutedExpertTier{
                .name = "tier-0",
                .domain = "domain-0",
                .priority = -4,
            },
            RoutedExpertTier{
                .name = "tier-1",
                .domain = "domain-1",
                .priority = 22,
                .fallback = true,
            },
        };
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> budgets = {
            boundBudget(0, cuda_budget),
            boundBudget(0, cpu_budget),
        };

        const auto immutable =
            MoEOverlayCapacityAdmission::buildResolverInput(
                plan, 2, model_manifest, budgets);
        EXPECT_EQ(
            immutable.initial_residency_policy,
            MoEOverlayInitialResidencyPolicy::PriorityFillOnly);

        const MoEOverlayCapacityAdmissionPolicy migration{
            .migration_storage =
                MoEOverlayMigrationStorageKind::PhysicalResidencyFabric,
            .shadow_slots_per_endpoint_layer = 1,
            .staging_capacity_bytes = 4096,
            .maximum_concurrent_cycles = 1,
            .maximum_cycles_per_layer = 1,
            .distributed_transport = false,
            .overlay_world_size = 1,
        };
        const auto movable =
            MoEOverlayCapacityAdmission::buildResolverInput(
                plan, 2, model_manifest, budgets, migration);
        EXPECT_EQ(
            movable.initial_residency_policy,
            MoEOverlayInitialResidencyPolicy::
                MigrationSourcePerParticipant);
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
            /*shadow_copies_per_layer=*/0);
        const auto rocm_budget = exactBudget(
            "physical-rank2-rocm3",
            DeviceId::rocm(3),
            footprints,
            {1, 1},
            /*shadow_copies_per_layer=*/0);
        const auto cpu_budget = exactBudget(
            "physical-rank11-cpu",
            DeviceId::cpu(),
            footprints,
            {3, 3},
            /*shadow_copies_per_layer=*/0);

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
                  cpu_budget.resourceId());
        EXPECT_EQ(input.tiers[1].participants[0].resource_id,
                  cuda_budget.resourceId());
        EXPECT_EQ(input.tiers[2].participants[0].resource_id,
                  rocm_budget.resourceId());

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
            /*shadow_copies_per_layer=*/0);

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
                  shared_budget.resourceId());
        EXPECT_EQ(input.tiers[1].participants[0].resource_id,
                  shared_budget.resourceId());
        EXPECT_NE(input.tiers[0].participants[0].participant_id,
                  input.tiers[1].participants[0].participant_id);

        const auto capacity = MoEOverlayCapacityResolver::resolve(input);
        EXPECT_EQ(capacity.tier(0)->live_experts_per_layer,
                  (std::vector<int>{1}));
        EXPECT_EQ(capacity.tier(1)->live_experts_per_layer,
                  (std::vector<int>{1}));
        const auto *physical = capacity.resource(shared_budget.resourceId());
        ASSERT_NE(physical, nullptr);
        EXPECT_EQ(physical->shadow_arrival_capacity_per_layer,
                  (std::vector<int>{0}));
        EXPECT_EQ(physical->live_copies_per_layer,
                  (std::vector<int>{2}));
        EXPECT_EQ(physical->usedBytes(), shared_budget.usableBudgetBytes());
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
            .migration_storage =
                MoEOverlayMigrationStorageKind::PhysicalResidencyFabric,
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

        auto underprovisioned_two_cycle_policy = policy;
        underprovisioned_two_cycle_policy.maximum_concurrent_cycles = 2;
        underprovisioned_two_cycle_policy.maximum_cycles_per_layer = 2;
        EXPECT_THROW(
            (void)MoEOverlayCapacityAdmission::transferStagingBOM(
                plan, /*world_size=*/2,
                underprovisioned_two_cycle_policy),
            std::invalid_argument);

        auto two_cycle_policy = underprovisioned_two_cycle_policy;
        two_cycle_policy.shadow_slots_per_endpoint_layer =
            MoEOverlayCapacityAdmissionPolicy::
                requiredShadowSlotsPerEndpointLayer(
                    two_cycle_policy.maximum_concurrent_cycles,
                    two_cycle_policy.maximum_cycles_per_layer);
        const auto two_cycle_charges =
            MoEOverlayCapacityAdmission::transferStagingBOM(
                plan, /*world_size=*/2, two_cycle_policy);
        ASSERT_EQ(two_cycle_charges.size(), charges.size());
        for (std::size_t index = 0; index < charges.size(); ++index)
        {
            EXPECT_EQ(
                two_cycle_charges[index].world_rank,
                charges[index].world_rank);
            EXPECT_EQ(
                two_cycle_charges[index].device,
                charges[index].device);
            EXPECT_EQ(two_cycle_charges[index].bytes, 2u * charges[index].bytes);
        }

        auto broad_one_cycle_per_layer_policy = two_cycle_policy;
        broad_one_cycle_per_layer_policy.maximum_concurrent_cycles = 8;
        broad_one_cycle_per_layer_policy.maximum_cycles_per_layer = 1;
        broad_one_cycle_per_layer_policy.shadow_slots_per_endpoint_layer =
            MoEOverlayCapacityAdmissionPolicy::
                requiredShadowSlotsPerEndpointLayer(
                    broad_one_cycle_per_layer_policy
                        .maximum_concurrent_cycles,
                    broad_one_cycle_per_layer_policy
                        .maximum_cycles_per_layer);
        EXPECT_EQ(
            broad_one_cycle_per_layer_policy
                .shadow_slots_per_endpoint_layer,
            1u);
        const auto broad_charges =
            MoEOverlayCapacityAdmission::transferStagingBOM(
                plan, /*world_size=*/2,
                broad_one_cycle_per_layer_policy);
        ASSERT_EQ(broad_charges.size(), charges.size());
        for (std::size_t index = 0; index < charges.size(); ++index)
        {
            EXPECT_EQ(
                broad_charges[index].bytes,
                8u * charges[index].bytes);
        }
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         NodeLocalActivationBOMPricesExactRootTargetAndMTPFamilyLanes)
    {
        MoERoutedExpertPlacementPlan overlay;
        overlay.enabled = true;
        overlay.topology = RoutedExpertPlacementTopology::TieredOverlay;
        overlay.continuation_domain = "opaque-primary";
        overlay.base_model_domain = "opaque-primary";
        overlay.shared_expert_domain = "opaque-primary";
        overlay.continuation_domain_spec.domain = "opaque-primary";
        overlay.continuation_domain_spec.logical_root_participant = 0;

        RoutedExpertDomain primary;
        primary.name = "opaque-primary";
        primary.scope = ExecutionDomainScope::RANK_LOCAL;
        primary.participants = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };
        primary.world_ranks = {0, 0};
        primary.owner_rank = 0;
        primary.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;

        RoutedExpertDomain secondary;
        secondary.name = "opaque-secondary";
        secondary.scope = ExecutionDomainScope::RANK_LOCAL;
        secondary.participants = {
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::rocm(1),
            GlobalDeviceAddress::rocm(2),
            GlobalDeviceAddress::rocm(3),
        };
        secondary.world_ranks = {1, 1, 1, 1};
        secondary.owner_rank = 1;
        secondary.routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;

        RoutedExpertDomain inter_node;
        inter_node.name = "opaque-inter-node";
        inter_node.scope = ExecutionDomainScope::SINGLE;
        inter_node.participants = {GlobalDeviceAddress::cuda(2)};
        inter_node.world_ranks = {2};
        inter_node.owner_rank = 2;
        inter_node.routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        overlay.domains = {primary, secondary, inter_node};
        overlay.routed_tiers = {
            RoutedExpertTier{
                .name = "priority-minus-seven",
                .domain = primary.name,
                .priority = -7,
            },
            RoutedExpertTier{
                .name = "priority-eleven",
                .domain = secondary.name,
                .priority = 11,
            },
            RoutedExpertTier{
                .name = "priority-ninety",
                .domain = inter_node.name,
                .priority = 90,
                .fallback = true,
            },
        };

        ClusterInventory cluster;
        cluster.world_size = 3;
        cluster.ranks.resize(3);
        for (int rank = 0; rank < 3; ++rank)
        {
            cluster.ranks[static_cast<std::size_t>(rank)].rank = rank;
            cluster.ranks[static_cast<std::size_t>(rank)].node_id =
                rank == 2 ? 9 : 4;
            cluster.ranks[static_cast<std::size_t>(rank)].hostname =
                rank == 2 ? "other-node" : "shared-node";
        }

        constexpr std::size_t kRows = 8;
        constexpr int kDModel = 16;
        constexpr int kTopK = 2;
        constexpr std::size_t kFamilies = 3;
        constexpr std::size_t kMatrixBytes =
            kRows * static_cast<std::size_t>(kDModel) * sizeof(float);
        constexpr std::size_t kLaneBytes = kFamilies * kMatrixBytes;
        constexpr std::size_t kCanonicalRouteBytes =
            kRows * static_cast<std::size_t>(kTopK) *
            static_cast<std::size_t>(kDModel) * sizeof(float);
        const auto channel_plan =
            MoEOverlayActivationChannelPlanner::plan({
                .placement_plan = &overlay,
                .cluster_inventory = &cluster,
                .row_capacity = kRows,
                .d_model = kDModel,
                .top_k = kTopK,
                .graph_family_count = kFamilies,
            });

        EXPECT_EQ(channel_plan.payload_matrix_bytes, kMatrixBytes);
        const std::size_t kGrantBytes =
            channel_plan.device_grant_bytes_per_lane;
        EXPECT_EQ(
            channel_plan.canonical_route_matrix_bytes,
            kCanonicalRouteBytes);
        ASSERT_EQ(channel_plan.channels.size(), 1u)
            << "the different-node tier must remain on the MPI path";
        const auto &channel = channel_plan.channels.front();
        EXPECT_EQ(channel.tier_index, 1);
        EXPECT_EQ(channel.domain_ordinal, 1);
        EXPECT_EQ(channel.source_world_rank, 0);
        EXPECT_EQ(channel.target_world_rank, 1);
        EXPECT_EQ(channel.source_device, DeviceId::cuda(0));
        EXPECT_EQ(
            channel.targetParticipantIds(),
            (std::vector<int>{2, 3, 4, 5}));
        ASSERT_EQ(channel.source_lanes.size(), 4u);
        ASSERT_EQ(channel.target_lanes.size(), 4u);
        for (std::size_t lane = 0; lane < 4u; ++lane)
        {
            EXPECT_EQ(channel.source_lanes[lane].device, DeviceId::cuda(0));
            EXPECT_EQ(
                channel.target_lanes[lane].device,
                DeviceId::rocm(static_cast<int>(lane)));
        }
        EXPECT_EQ(
            channel_plan.stagingBytesFor(0, DeviceId::cuda(0)),
            4u * (kLaneBytes + kGrantBytes));
        EXPECT_EQ(
            channel_plan.stagingBytesFor(0, DeviceId::cuda(1)), 0u);
        for (int device = 0; device < 4; ++device)
        {
            EXPECT_EQ(
                channel_plan.stagingBytesFor(1, DeviceId::rocm(device)),
                kLaneBytes + kGrantBytes + kCanonicalRouteBytes);
        }
        EXPECT_EQ(
            channel_plan.stagingBytesFor(0, DeviceId::cpu()),
            channel.mapping_layout.sourceOwnedBytes());
        EXPECT_EQ(
            channel_plan.stagingBytesFor(1, DeviceId::cpu()),
            channel.mapping_layout.targetOwnedBytes());
        EXPECT_EQ(
            channel_plan.stagingBytesFor(2, DeviceId::cuda(2)), 0u);

        /* The rank-local capacity adapter must add the exact same charge to
         * the continuation root only; the other TP shard retains only its
         * independent model-upload ring. */
        auto profile = smallModelProfile();
        profile.d_model = kDModel;
        RankExecutionPlan rank_plan;
        rank_plan.rank = 0;
        rank_plan.first_layer = 0;
        rank_plan.last_layer = profile.n_layers - 1;
        rank_plan.primary_device = GlobalDeviceAddress::cuda(0);
        rank_plan.local_tp_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };
        rank_plan.runtime.batch_size = 1;
        rank_plan.runtime.max_seq_len = 32;
        rank_plan.runtime.kv_cache_precision = KVCachePrecision::FP16;
        rank_plan.runtime.prefix_cache.enabled = false;
        rank_plan.runtime.prefix_cache.storage_mode =
            PrefixCacheStorageMode::Disabled;

        constexpr std::size_t kLargeBudget =
            64ULL * 1024ULL * 1024ULL * 1024ULL;
        auto &rank_zero = cluster.ranks[0];
        rank_zero.cpu_memory_bytes = kLargeBudget;
        rank_zero.cpu.memory_bytes = kLargeBudget;
        rank_zero.cpu.free_memory_bytes = kLargeBudget;
        rank_zero.gpus = {
            DeviceInfo{
                .type = DeviceType::CUDA,
                .local_device_id = 0,
                .memory_bytes = kLargeBudget,
                .free_memory_bytes = kLargeBudget,
                .compute_units = 82,
            },
            DeviceInfo{
                .type = DeviceType::CUDA,
                .local_device_id = 1,
                .memory_bytes = kLargeBudget,
                .free_memory_bytes = kLargeBudget,
                .compute_units = 82,
            },
        };
        const auto gpu_load = testGPUWeightLoadCapacityInput();
        const auto capacity = MoEOverlayLocalCapacityPlanner::plan({
            .model_profile = &profile,
            .rank_plan = &rank_plan,
            .overlay_plan = &overlay,
            .rank_inventory = &rank_zero,
            .cluster_inventory = &cluster,
            .rank_execution_kind =
                OverlayRankExecutionKind::ContinuationAuthority,
            .resident_graph_rows = static_cast<int>(kRows),
            .activation_channel_row_capacity = static_cast<int>(kRows),
            .activation_graph_family_count = kFamilies,
            .captured_graph_plan = resolveMoEOverlayCapturedGraphPlan(
                profile.n_layers,
                MoEOverlayAuthorityExecutionKind::HostResident,
                /*model_graph_identity_count=*/11u,
                /*model_graph_topology_variant_count=*/1u,
                /*auxiliary_executable_count=*/0u),
            .gpu_weight_load = gpu_load,
        });
        const auto budgetFor = [&](DeviceId device)
            -> const MoEOverlayBoundPhysicalMemoryBudget &
        {
            const auto found = std::find_if(
                capacity.physical_budgets.begin(),
                capacity.physical_budgets.end(),
                [&](const auto &budget)
                { return budget.device() == device; });
            if (found == capacity.physical_budgets.end())
                throw std::logic_error("test capacity device is absent");
            return *found;
        };
        const auto upload = resolveGPUWeightLoadMemoryGeometry(
            gpu_load.maximum_source_bytes,
            gpu_load.policy);
        EXPECT_EQ(
            budgetFor(DeviceId::cuda(0))
                .stagingBytes(),
            upload.staging_bytes +
                4u * (kLaneBytes + kGrantBytes));
        EXPECT_EQ(
            budgetFor(DeviceId::cuda(1))
                .stagingBytes(),
            upload.staging_bytes);
        const std::size_t captured_graph_bytes =
            estimateCapturedGraphExecutableBytes(
                DeviceId::cuda(0), /*executable_count=*/11u);
        EXPECT_GE(
            budgetFor(DeviceId::cuda(0)).fixedBytes(),
            captured_graph_bytes);
        EXPECT_GE(
            budgetFor(DeviceId::cuda(1)).fixedBytes(),
            captured_graph_bytes);
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         HostAuthoritySeparatesCompilationUnitsFromResidentExecutables)
    {
        constexpr int kModelLayers = 48;
        constexpr std::size_t kModelGraphIdentities = 5u;
        constexpr std::size_t kAuxiliaryExecutables = 7u;

        const auto plan =
            resolveMoEOverlayCapturedGraphPlan(
                kModelLayers,
                MoEOverlayAuthorityExecutionKind::HostResident,
                kModelGraphIdentities,
                /*model_graph_topology_variant_count=*/1u,
                kAuxiliaryExecutables);
        ASSERT_TRUE(plan.valid());
        EXPECT_EQ(
            plan.compilation.compilation_units_per_model_graph,
            49u);
        EXPECT_EQ(
            plan.resident_executables.residentExecutableCount(),
            kModelGraphIdentities + kAuxiliaryExecutables);

        const std::size_t physical_bytes =
            estimateCapturedGraphExecutableBytes(
                DeviceId::cuda(0), plan.resident_executables);
        EXPECT_EQ(
            physical_bytes,
            (kModelGraphIdentities + kAuxiliaryExecutables) *
                GPUGraphMemoryContract::kCUDAExecutableReservationBytes)
            << "Compilation children are imported into each retained parent; "
               "they are not independent executable owners.";
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         DeviceAuthorityKeepsCompleteModelGraphsMonolithic)
    {
        const auto plan =
            resolveMoEOverlayCapturedGraphPlan(
                /*model_layer_count=*/48,
                MoEOverlayAuthorityExecutionKind::DeviceResident,
                /*model_graph_identity_count=*/5u,
                /*model_graph_topology_variant_count=*/1u,
                /*auxiliary_executable_count=*/7u);
        ASSERT_TRUE(plan.valid());
        EXPECT_EQ(
            plan.resident_executables.model_graph_identity_count,
            5u);
        EXPECT_EQ(
            plan.resident_executables.model_graph_topology_variant_count,
            1u);
        EXPECT_EQ(
            plan.compilation.compilation_units_per_model_graph,
            1u);
        EXPECT_EQ(
            plan.resident_executables.auxiliary_executable_count,
            7u);
        EXPECT_THROW(
            {
                const auto unresolved =
                    resolveMoEOverlayCapturedGraphPlan(
                        48,
                        MoEOverlayAuthorityExecutionKind::Unresolved,
                        5u,
                        1u,
                        7u);
                (void)unresolved;
            },
            std::invalid_argument);
    }

    TEST(MoEOverlayLocalCapacityPlanner,
         DeviceAuthorityPricesDiagnosticAndLeanTopologyVariants)
    {
        constexpr int kModelLayers = 48;
        constexpr std::size_t kIdentitiesPerVariant = 8u;
        constexpr std::size_t kTopologyVariants = 2u;
        constexpr std::size_t kAuxiliaryExecutables = 13u;
        const auto plan =
            resolveMoEOverlayCapturedGraphPlan(
                kModelLayers,
                MoEOverlayAuthorityExecutionKind::DeviceResident,
                kIdentitiesPerVariant,
            kTopologyVariants,
            kAuxiliaryExecutables);

        EXPECT_EQ(
            plan.resident_executables.model_graph_identity_count,
            kIdentitiesPerVariant);
        EXPECT_EQ(
            plan.resident_executables.model_graph_topology_variant_count,
            kTopologyVariants);
        EXPECT_EQ(
            plan.compilation.compilation_units_per_model_graph,
            1u);
        EXPECT_EQ(
            plan.resident_executables.auxiliary_executable_count,
            kAuxiliaryExecutables);
        EXPECT_EQ(
            estimateCapturedGraphExecutableBytes(
                DeviceId::cuda(0),
                plan.resident_executables),
            (kIdentitiesPerVariant * kTopologyVariants +
             kAuxiliaryExecutables) *
                GPUGraphMemoryContract::kCUDAExecutableReservationBytes);
    }

    /**
     * @brief Prove tier identity never substitutes for rank locality.
     *
     * A one-tier NodeTP overlay still transports activations between its two
     * MPI ranks. This is the production topology that previously reached graph
     * construction with no preflight channel.
     */
    TEST(MoEOverlayLocalCapacityPlanner,
         SingleTierRemoteRankCreatesNodeLocalActivationChannel)
    {
        MoERoutedExpertPlacementPlan overlay;
        overlay.enabled = true;
        overlay.topology = RoutedExpertPlacementTopology::TieredOverlay;
        overlay.continuation_domain = "single-priority-domain";
        overlay.base_model_domain = "single-priority-domain";
        overlay.shared_expert_domain = "single-priority-domain";
        overlay.continuation_domain_spec.domain = "single-priority-domain";
        overlay.continuation_domain_spec.logical_root_participant = 0;

        RoutedExpertDomain domain;
        domain.name = "single-priority-domain";
        domain.scope = ExecutionDomainScope::RANK_LOCAL;
        domain.participants = {
            GlobalDeviceAddress::cpu(0),
            GlobalDeviceAddress::cpu(0),
        };
        domain.world_ranks = {0, 1};
        domain.owner_rank = 0;
        domain.routed_compute_policy =
            RoutedExpertComputePolicy::Apportioned;
        overlay.domains = {domain};
        overlay.routed_tiers = {RoutedExpertTier{
            .name = "only-priority",
            .domain = domain.name,
            .priority = 17,
        }};

        ClusterInventory cluster;
        cluster.world_size = 2;
        cluster.ranks.resize(2);
        for (int rank = 0; rank < 2; ++rank)
        {
            cluster.ranks[static_cast<std::size_t>(rank)].rank = rank;
            cluster.ranks[static_cast<std::size_t>(rank)].node_id = 5;
            cluster.ranks[static_cast<std::size_t>(rank)].hostname =
                "shared-node";
        }

        EXPECT_TRUE(
            MoEOverlayActivationChannelPlanner::hasRemoteRankParticipants(
                overlay));
        const auto plan = MoEOverlayActivationChannelPlanner::plan({
            .placement_plan = &overlay,
            .cluster_inventory = &cluster,
            .row_capacity = 8u,
            .d_model = 16,
            .top_k = 2,
            .graph_family_count = 3u,
        });

        ASSERT_EQ(plan.channels.size(), 1u);
        const auto &channel = plan.channels.front();
        EXPECT_EQ(channel.tier_index, 0);
        EXPECT_EQ(channel.domain_ordinal, 0);
        EXPECT_EQ(channel.source_world_rank, 0);
        EXPECT_EQ(channel.target_world_rank, 1);
        EXPECT_EQ(channel.source_device, DeviceId::cpu());
        EXPECT_EQ(channel.targetParticipantIds(), (std::vector<int>{1}));
        ASSERT_EQ(channel.source_lanes.size(), 1u);
        ASSERT_EQ(channel.target_lanes.size(), 1u);
        EXPECT_EQ(channel.source_lanes.front().device, DeviceId::cpu());
        EXPECT_EQ(channel.target_lanes.front().device, DeviceId::cpu());
        const std::size_t kGrantBytes =
            plan.device_grant_bytes_per_lane;
        EXPECT_EQ(
            plan.stagingBytesFor(0, DeviceId::cpu()),
            channel.mapping_layout.sourceOwnedBytes() + kGrantBytes);
        EXPECT_EQ(
            plan.stagingBytesFor(1, DeviceId::cpu()),
            channel.mapping_layout.targetOwnedBytes() + kGrantBytes);
    }

    /**
     * @brief Keep a colocated lower-priority participant on the local graph.
     *
     * A NodeTP CPU tier naturally has one socket participant on the same MPI
     * rank as whichever GPU inventory binding selects as continuation. The
     * activation-channel planner must create only the remote-rank lane: a
     * same-rank self-channel is invalid and its compact arena is already part
     * of the rank-local graph memory plan.
     */
    TEST(MoEOverlayLocalCapacityPlanner,
         NodeLocalActivationPlanSkipsColocatedCrossTierParticipant)
    {
        MoERoutedExpertPlacementPlan overlay;
        overlay.enabled = true;
        overlay.topology = RoutedExpertPlacementTopology::TieredOverlay;
        overlay.continuation_domain = "priority-zero-domain";
        overlay.base_model_domain = "priority-zero-domain";
        overlay.shared_expert_domain = "priority-zero-domain";
        overlay.continuation_domain_spec.domain = "priority-zero-domain";
        overlay.continuation_domain_spec.logical_root_participant = 0;

        RoutedExpertDomain continuation;
        continuation.name = "priority-zero-domain";
        continuation.scope = ExecutionDomainScope::SINGLE;
        continuation.participants = {GlobalDeviceAddress::cuda(0)};
        continuation.world_ranks = {0};
        continuation.owner_rank = 0;

        RoutedExpertDomain colocated;
        colocated.name = "priority-seven-domain";
        colocated.scope = ExecutionDomainScope::SINGLE;
        colocated.participants = {GlobalDeviceAddress::cpu(0)};
        colocated.world_ranks = {0};
        colocated.owner_rank = 0;

        RoutedExpertDomain remote;
        remote.name = "priority-forty-one-domain";
        remote.scope = ExecutionDomainScope::SINGLE;
        remote.participants = {GlobalDeviceAddress::rocm(0)};
        remote.world_ranks = {1};
        remote.owner_rank = 1;

        overlay.domains = {continuation, colocated, remote};
        overlay.routed_tiers = {
            RoutedExpertTier{
                .name = "priority-zero",
                .domain = continuation.name,
                .priority = 0,
            },
            RoutedExpertTier{
                .name = "priority-seven",
                .domain = colocated.name,
                .priority = 7,
            },
            RoutedExpertTier{
                .name = "priority-forty-one",
                .domain = remote.name,
                .priority = 41,
                .fallback = true,
            },
        };

        ClusterInventory cluster;
        cluster.world_size = 2;
        cluster.ranks.resize(2);
        for (int rank = 0; rank < 2; ++rank)
        {
            cluster.ranks[static_cast<std::size_t>(rank)].rank = rank;
            cluster.ranks[static_cast<std::size_t>(rank)].node_id = 3;
            cluster.ranks[static_cast<std::size_t>(rank)].hostname =
                "shared-node";
        }

        constexpr std::size_t kRows = 4;
        constexpr int kDModel = 8;
        constexpr int kTopK = 2;
        constexpr std::size_t kFamilies = 2;
        constexpr std::size_t kLaneBytes =
            kRows * static_cast<std::size_t>(kDModel) * sizeof(float) *
            kFamilies;
        constexpr std::size_t kCanonicalRouteBytes =
            kRows * static_cast<std::size_t>(kTopK) *
            static_cast<std::size_t>(kDModel) * sizeof(float);
        const auto plan = MoEOverlayActivationChannelPlanner::plan({
            .placement_plan = &overlay,
            .cluster_inventory = &cluster,
            .row_capacity = kRows,
            .d_model = kDModel,
            .top_k = kTopK,
            .graph_family_count = kFamilies,
        });
        const std::size_t kGrantBytes =
            plan.device_grant_bytes_per_lane;

        ASSERT_EQ(plan.channels.size(), 1u);
        EXPECT_EQ(plan.channels.front().source_world_rank, 0);
        EXPECT_EQ(plan.channels.front().target_world_rank, 1);
        EXPECT_EQ(
            plan.channels.front().targetParticipantIds(),
            (std::vector<int>{2}));
        EXPECT_EQ(
            plan.stagingBytesFor(0, DeviceId::cuda(0)),
            kLaneBytes + kGrantBytes);
        EXPECT_EQ(
            plan.stagingBytesFor(0, DeviceId::cpu()),
            plan.channels.front().mapping_layout.sourceOwnedBytes());
        EXPECT_EQ(
            plan.stagingBytesFor(1, DeviceId::rocm(0)),
            kLaneBytes + kGrantBytes + kCanonicalRouteBytes);
        EXPECT_EQ(
            plan.stagingBytesFor(1, DeviceId::cpu()),
            plan.channels.front().mapping_layout.targetOwnedBytes());
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
                OverlayRankExecutionKind::ContinuationAuthority);
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
                OverlayRankExecutionKind::ExpertOnlyFollower)
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
        rank_plan.runtime.prefix_cache.enabled = false;
        rank_plan.runtime.prefix_cache.storage_mode =
            PrefixCacheStorageMode::Disabled;

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
            .rank_execution_kind =
                OverlayRankExecutionKind::ContinuationAuthority,
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
                { return budget.device() == device; });
            return found == result.physical_budgets.end()
                       ? nullptr
                       : &*found;
        };
        const auto *cpu = budgetFor(DeviceId::cpu());
        const auto *cuda = budgetFor(DeviceId::cuda(0));
        ASSERT_NE(cpu, nullptr);
        ASSERT_NE(cuda, nullptr);
        EXPECT_EQ(cpu->worldRank(), 5);
        EXPECT_EQ(cuda->worldRank(), 5);
        EXPECT_EQ(cpu->usableBudgetBytes(), 512u * 1024u * 1024u);
        EXPECT_EQ(cuda->usableBudgetBytes(), 1024u * 1024u * 1024u);
        const auto expected_load_geometry =
            resolveGPUWeightLoadMemoryGeometry(
            gpu_weight_load.maximum_source_bytes,
            gpu_weight_load.policy);
        EXPECT_EQ(
            cuda->stagingBytes(),
            expected_load_geometry.staging_bytes);
        /*
         * One serial participant owns the complete immutable route-family
         * ladder. Eight rows at top-k two retain capacities 1, 2, 4, 8, and
         * the exact terminal capacity 16, shared across all model layers.
         */
        constexpr std::size_t kExpectedCpuSparsePacketBytes =
            (1u + 2u + 4u + 8u + 16u) *
            (2u * 64u + 2u) * sizeof(float);
        EXPECT_EQ(cpu->fixedBytes(), kExpectedCpuSparsePacketBytes);
        EXPECT_GT(cuda->fixedBytes(), 0u);
        EXPECT_EQ(
            cuda->resourceId(),
            MoEOverlayLocalCapacityPlanner::physicalResourceId(
                5, DeviceId::cuda(0)));
        for (const auto &device_plan : result.fixed_memory_plan.devices)
            EXPECT_EQ(device_plan.activation_seq_len(), 8);
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
        rank_plan.runtime.prefix_cache.enabled = false;
        rank_plan.runtime.prefix_cache.storage_mode =
            PrefixCacheStorageMode::Disabled;

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
            .rank_execution_kind =
                OverlayRankExecutionKind::ExpertOnlyFollower,
            .require_host_memory_authority = true,
        });
        EXPECT_TRUE(result.fixed_memory_plan.devices.empty());
        ASSERT_EQ(result.physical_budgets.size(), 1u);
        EXPECT_EQ(result.physical_budgets[0].device(), DeviceId::cpu());
        EXPECT_EQ(result.physical_budgets[0].fixedBytes(), 0u);
        EXPECT_EQ(
            result.physical_budgets[0].usableBudgetBytes(),
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
        const auto upload_geometry = resolveGPUWeightLoadMemoryGeometry(
            gpu_load.maximum_source_bytes,
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
            fixed_bytes + upload_geometry.staging_bytes + live_expert_bytes;

        MoEOverlayCapacityResolverInput admission;
        admission.num_experts = 1;
        admission.layer_weight_manifest = model_manifest;
        admission.physical_budgets = {physicalBudget(
            "same-equation-gpu",
            DeviceId::cuda(0),
            initial_free,
            fixed_bytes,
            upload_geometry.staging_bytes)};
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
            admitted.resource("same-equation-gpu")->remainingBytes(),
            0u);

        /*
         * By load time the non-weight fixed state has already reduced the
         * backend's free-memory reading. The pool contains persistent dense
         * weights plus the admitted expert. Algebraically this is exactly the
         * same physical bill, not a similar heuristic.
         */
        const std::size_t late_free = initial_free - kNonWeightFixed;
        const auto late_load = gpuWeightLoadMemoryBOM(
            PhysicalMemoryResource{
                .world_rank = -1,
                .device = DeviceId::cuda(0),
                .total_bytes = kTotalVram,
                .admission_available_bytes = late_free,
            },
            kPersistentWeights + live_expert_bytes,
            upload_geometry);
        EXPECT_EQ(
            late_load.bytes(PhysicalMemoryOwner::WeightLoadStaging),
            upload_geometry.staging_bytes);
        EXPECT_EQ(late_load.incrementalBytes(), late_free);
        EXPECT_TRUE(late_load.fits());

        const auto one_byte_short = gpuWeightLoadMemoryBOM(
            PhysicalMemoryResource{
                .world_rank = -1,
                .device = DeviceId::cuda(0),
                .total_bytes = kTotalVram,
                .admission_available_bytes = late_free - 1u,
            },
            kPersistentWeights + live_expert_bytes,
            upload_geometry);
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
        rank_plan.runtime.prefix_cache.enabled = false;
        rank_plan.runtime.prefix_cache.storage_mode =
            PrefixCacheStorageMode::Disabled;

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
                .rank_execution_kind =
                    OverlayRankExecutionKind::ContinuationAuthority,
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
                { return budget.device() == device; });
            if (found == budgets.end())
                throw std::logic_error("test physical budget is missing");
            return *found;
        };

        const auto &small_gpu = budgetFor(
            small.physical_budgets, DeviceId::cuda(0));
        const auto &small_cpu = budgetFor(
            small.physical_budgets, DeviceId::cpu());
        const std::size_t gpu_limit =
            small_gpu.fixedBytes() +
            small_gpu.stagingBytes() +
            footprints[0].gpu_live_bytes + footprints[1].gpu_live_bytes;
        const std::size_t cpu_limit =
            small_cpu.fixedBytes() +
            small_cpu.stagingBytes() +
            3 * footprints[0].cpu_live_bytes +
            3 * footprints[1].cpu_live_bytes;
        EXPECT_GT(
            budgetFor(large.physical_budgets, DeviceId::cuda(0)).fixedBytes(),
            gpu_limit)
            << "the adversarial budget must distinguish the two graph shapes";

        auto &small_gpu_budget = budgetFor(
            small.physical_budgets, DeviceId::cuda(0));
        small_gpu_budget = small_gpu_budget.withAvailableBytes(gpu_limit);
        auto &small_cpu_budget = budgetFor(
            small.physical_budgets, DeviceId::cpu());
        small_cpu_budget = small_cpu_budget.withAvailableBytes(cpu_limit);

        /*
         * A bound budget is now an admission certificate, so the larger graph
         * is rejected at the instant its allocator observation is narrowed.
         * It must never travel downstream as an apparently valid budget and
         * rely on the expert resolver to rediscover the fixed-memory deficit.
         */
        EXPECT_THROW(
            (void)budgetFor(
                large.physical_budgets,
                DeviceId::cuda(0))
                .withAvailableBytes(gpu_limit),
            std::invalid_argument);

        const MoEOverlayCapacityAdmissionPolicy no_migration{
            .migration_storage =
                MoEOverlayMigrationStorageKind::Disabled,
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
        EXPECT_EQ(
            resource->shadow_arrival_capacity_per_layer,
            (std::vector<int>{2}));
        EXPECT_EQ(resource->remainingBytes(), 0u);

        input.physical_budgets.front() =
            input.physical_budgets.front().withAvailableBytes(
                input.physical_budgets.front().usableBudgetBytes() - 1u);
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::invalid_argument);
    }

    TEST(MoEOverlayCapacityResolver,
         IdenticalLayerGeometriesShareTheGlobalShadowSlotBudget)
    {
        constexpr int kLayerCount = 4;
        const auto model_manifest = manifest(
            kLayerCount,
            native_vnni_formats::Q4_0,
            /*vary_geometry=*/false);
        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(model_manifest);
        ASSERT_EQ(footprints.size(), static_cast<std::size_t>(kLayerCount));

        constexpr std::size_t kFixedBytes = 113u;
        constexpr std::size_t kStagingBytes = 257u;
        std::size_t admitted_bytes = kFixedBytes + kStagingBytes;
        for (const auto &footprint : footprints)
            admitted_bytes += footprint.gpu_live_bytes;
        admitted_bytes += 2u * footprints.front().gpu_shadow_bytes;

        MoEOverlayCapacityResolverInput input;
        input.num_experts = 1;
        input.layer_weight_manifest = model_manifest;
        input.physical_budgets = {physicalBudget(
            "shared-geometry-gpu",
            DeviceId::cuda(0),
            admitted_bytes,
            kFixedBytes,
            kStagingBytes)};
        input.tiers = {tier(
            0,
            "only-tier",
            0,
            true,
            MoEOverlayLiveQuotaMode::FixedPerLayer,
            {MoEOverlayTierCapacityParticipant{
                .participant_id = 0,
                .resource_id = "shared-geometry-gpu",
                .shadow_slots_per_layer = 1,
                .maximum_concurrent_shadow_slots = 2,
            }},
            {1, 1, 1, 1})};

        const auto resolved = MoEOverlayCapacityResolver::resolve(input);
        const auto *resource = resolved.resource("shared-geometry-gpu");
        ASSERT_NE(resource, nullptr);
        EXPECT_EQ(
            resource->shadowBytes(),
            2u * footprints.front().gpu_shadow_bytes)
            << "Four compatible layers need only the globally concurrent two slots";
        EXPECT_EQ(
            resource->shadow_arrival_capacity_per_layer,
            (std::vector<int>{1, 1, 1, 1}))
            << "Logical per-layer admission remains independently bounded";
        EXPECT_EQ(resource->remainingBytes(), 0u);

        input.physical_budgets.front() =
            input.physical_budgets.front().withAvailableBytes(
                admitted_bytes - 1u);
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
            apportioned.resource("cuda-a")->remainingBytes(),
            footprints.front().gpu_live_bytes);
        EXPECT_EQ(
            apportioned.resource("cuda-b")->remainingBytes(),
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
        EXPECT_EQ(cold->liveExpertBytes(), 0u);
        EXPECT_EQ(cold->shadowBytes(), footprints.front().cpu_shadow_bytes);
        EXPECT_EQ(
            cold->shadow_arrival_capacity_per_layer,
            (std::vector<int>{1}));
        EXPECT_EQ(cold->remainingBytes(), 0u);

        input.physical_budgets.back() =
            input.physical_budgets.back().withAvailableBytes(
                input.physical_budgets.back().usableBudgetBytes() - 1u);
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
        input.tiers[0].participants[0].maximum_concurrent_shadow_slots =
            std::numeric_limits<std::size_t>::max();
        EXPECT_THROW(
            (void)MoEOverlayCapacityResolver::resolve(input),
            std::overflow_error);
        EXPECT_THROW(
            (void)footprints.front().liveBytes(DeviceId::invalid()),
            std::invalid_argument);
    }
} // namespace llaminar2
