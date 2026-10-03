/**
 * @file Test__MoEProjectionCapacity.cpp
 * @brief Device-free accounting proof for the two explicit routed projection modes.
 *
 * The production admission adapter and capacity resolver consume complete model
 * manifests. Tests independently price aligned GPU planes, including fixed down
 * slices on empty owners, and prove the exact one-byte rejection boundary. The
 * same cases cover CUDA/ROCm, all source codebooks, and FP16/BF16/FP32 without
 * allocating a backend or treating a diagnostic byte counter as an authority.
 */
#include "execution/moe/MoEOverlayCapacityAdmission.h"
#include "tensors/NativeVnniFormatInfo.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @return Every catalogued source identity, including raw floating families. */
        std::vector<ExpertWeightFormat> formats()
        {
            std::vector<ExpertWeightFormat> result;
            for (const auto &source : native_vnni_formats::kAllSourceFormats)
                result.push_back(ExpertWeightFormat::nativeVnni({
                    .codebook_id = source.metadata->codebook_id,
                    .is_superblock = source.metadata->is_superblock, .present = true}));
            for (auto type : {TensorType::FP16, TensorType::BF16, TensorType::FP32})
                result.push_back(ExpertWeightFormat::floating(type));
            return result;
        }

        /** @return Independent final alignment of the physical weight pool. */
        std::size_t aligned(std::size_t bytes) { return (bytes + 255u) & ~std::size_t{255u}; }

        /** @return Sum of aligned physical planes, not a flat proportional discount. */
        std::size_t gpuBytes(int n, int k, const ExpertWeightFormat &format)
        {
            if (format.isFloating())
                return aligned(std::size_t(n) * k * format.floatingElementBytes());
            const auto *source = native_vnni_formats::forSourceIdentity(
                format.native_vnni.codebook_id, format.native_vnni.is_superblock);
            const auto packing = reusableDeviceVnniAllocationFormat(*source);
            const auto blocks = std::size_t(n) * (k / 32);
            return aligned(blocks * packing.payload_bytes_per_block) + aligned(blocks * 2u) +
                (packing.has_mins ? aligned(blocks * 2u) : 0u) +
                (packing.has_emins ? aligned(blocks * 4u) : 0u);
        }

        /** @return Complete source matrices; array order intentionally differs from role order. */
        std::vector<MoEOverlayLayerWeightManifest> manifest(ExpertWeightFormat format, bool uniform = false)
        {
            std::vector<MoEOverlayLayerWeightManifest> result;
            for (int layer = 0; layer < 2; ++layer)
            {
                const int intermediate = uniform ? 256 : 256 * (layer + 1);
                result.push_back({.layer_idx = layer, .projections = {{
                    {.projection = ExpertTierWeightProjection::Down, .N = 768, .K = intermediate, .format = format},
                    {.projection = ExpertTierWeightProjection::Gate, .N = intermediate, .K = 768, .format = format},
                    {.projection = ExpertTierWeightProjection::Up, .N = intermediate, .K = 768, .format = format}}}});
            }
            return result;
        }

        /** @return One explicit rank-local domain; the control differs only in compute policy. */
        MoERoutedExpertPlacementPlan plan(DeviceType type, int degree, RoutedExpertComputePolicy compute)
        {
            MoERoutedExpertPlacementPlan result;
            result.enabled = true;
            result.topology = RoutedExpertPlacementTopology::SingleDomain;
            result.authority_execution = MoEOverlayAuthorityExecutionKind::DeviceResident;
            result.continuation_domain = result.base_model_domain = result.shared_expert_domain = "group";
            RoutedExpertDomain domain;
            domain.name = "group";
            domain.scope = ExecutionDomainScope::RANK_LOCAL;
            domain.backend = type == DeviceType::CUDA ? CollectiveBackendType::NCCL : CollectiveBackendType::RCCL;
            domain.owner_rank = 0;
            domain.routed_compute_policy = compute;
            for (int index = 0; index < degree; ++index)
            {
                domain.participants.push_back(type == DeviceType::CUDA
                    ? GlobalDeviceAddress::cuda(index) : GlobalDeviceAddress::rocm(index));
                domain.world_ranks.push_back(0);
            }
            result.domains = {domain};
            result.routed_tiers = {{.name = "tier", .domain = "group", .priority = 0, .fallback = true}};
            return result;
        }

        /** @return Physical base certificates with no hidden expert or staging charge. */
        std::vector<MoEOverlayBoundPhysicalMemoryBudget> budgets(const MoERoutedExpertPlacementPlan &plan)
        {
            std::vector<MoEOverlayBoundPhysicalMemoryBudget> result;
            int index = 0;
            for (const auto &address : plan.domains.front().participants)
            {
                PhysicalMemoryBOMBuilder builder({.world_rank = 0, .device = address.toLocalDeviceId(),
                    .total_bytes = 1u << 30, .admission_available_bytes = 1u << 30});
                builder.add(PhysicalMemoryOwner::ActivationArena, 4096u);
                result.emplace_back(std::to_string(index++), PhysicalMemoryAdmissionCertificate(builder.build()));
            }
            return result;
        }
    }

    /** @brief Balanced ownership retains the control's weight footprint on every backend/format. */
    TEST(MoEProjectionCapacity, CompleteAndGateUpModesPriceEveryProjectionExactly)
    {
        for (const auto format : formats())
        for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
        for (int degree : {1, 2, 3, 4, 8})
        for (int experts : {1, 24})
        {
            SCOPED_TRACE(::testing::Message() << static_cast<int>(type) << '/' << degree << '/' << experts
                << '/' << static_cast<int>(format.kind) << '/' << format.native_vnni.codebook_id);
            const auto source = manifest(format);
            const auto split_plan = plan(type, degree, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            auto input = MoEOverlayCapacityAdmission::buildResolverInput(split_plan, experts, source, budgets(split_plan), {});
            ASSERT_EQ(input.tiers.front().copy_policy, MoEOverlayTierCopyPolicy::GateUpOwnedDownColumns);
            const auto split = MoEOverlayCapacityResolver::resolve(input);
            const auto control_plan = plan(type, degree, RoutedExpertComputePolicy::Apportioned);
            const auto control = MoEOverlayCapacityAdmission::resolveCapacity(
                control_plan, experts, source, budgets(control_plan), {});
            for (int index = 0; index < degree; ++index)
            {
                const auto *resource = split.resource(std::to_string(index));
                ASSERT_NE(resource, nullptr);
                const auto owned = experts / degree + (index < experts % degree ? 1 : 0);
                std::size_t expected = 0;
                for (const auto &layer : source)
                {
                    const auto i = layer.projections.front().K;
                    expected += owned * 2u * gpuBytes(i, 768, format) +
                        experts * gpuBytes(768 / degree, i, format);
                }
                EXPECT_EQ(resource->liveExpertBytes(), expected);
                EXPECT_EQ(resource->live_copies_per_layer, (std::vector<int>{owned, owned}));
                EXPECT_EQ(resource->shadowBytes(), 0u);
                EXPECT_EQ(resource->usedBytes(), expected + 4096u);
                if (experts % degree == 0)
                    EXPECT_EQ(resource->usedBytes(), control.resource(std::to_string(index))->usedBytes());
                // This includes zero gate/up owners: the immutable down bank
                // must still be admitted in full on each participant.
                input.physical_budgets[index] = input.physical_budgets[index].withAvailableBytes(expected + 4096u);
            }
            EXPECT_NO_THROW((void)MoEOverlayCapacityResolver::resolve(input));
            const auto last = input.physical_budgets.size() - 1;
            input.physical_budgets[last] = input.physical_budgets[last].withAvailableBytes(
                split.resource(std::to_string(last))->usedBytes() - 1u);
            EXPECT_THROW((void)MoEOverlayCapacityResolver::resolve(input), MoEOverlayCapacityExhausted);
        }
    }

    /** @brief Native migration admission uses only the two movable planes and its exact workspace. */
    TEST(MoEProjectionCapacity, NativeDirectoryAndWorkspaceExcludeFixedDownOnBothBackends)
    {
        for (const auto format : formats())
        for (auto type : {DeviceType::CUDA, DeviceType::ROCm})
        {
            const auto source = manifest(format, true);
            const auto p = plan(type, 2, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
            const auto slots = DeviceMoETransferSlotDirectory::planBufferedCapacity(2, 1, 2);
            const MoEOverlayCapacityAdmissionPolicy policy{
                .migration_storage = MoEOverlayMigrationStorageKind::DeviceTransferDirectory,
                .device_transfer_directory_capacity = slots,
                .device_rebalance_workspace_capacity = DeviceMoERebalanceWorkspaceCapacity{
                    .num_layers = 2, .num_experts = 24, .participant_count = 2, .layer_window_count = 1,
                    .local_transfer_slot_count = slots.total_slots, .collective_payload_slot_capacity = 1,
                    .phase = DeviceMoERebalanceStagePhase::PlanCopyApply,
                    .transfer_mode = DeviceMoERebalanceTransferMode::CompactTransferSlots}};
            const auto input = MoEOverlayCapacityAdmission::buildResolverInput(p, 24, source, budgets(p), policy);
            const auto profile = DeviceMoETransferSlotDirectory::profileForLayerWeightManifest(source, DeviceMoEProjectionSet::GateUp);
            const auto whole = DeviceMoETransferSlotDirectory::profileForLayerWeightManifest(source);
            ASSERT_EQ(profile.allocation_specs.size(), 2u);
            EXPECT_LT(profile.max_wire_payload_bytes, whole.max_wire_payload_bytes);
            const auto directory = DeviceMoETransferSlotDirectory::allocationBOM(slots, profile);
            const auto workspace = DeviceMoERebalanceWorkspaceContract::allocationBytes({
                .capacity = *policy.device_rebalance_workspace_capacity,
                .collective_payload_slot_bytes = DeviceMoERebalanceWorkspaceContract::collectivePayloadSlotBytes(
                    profile.max_wire_payload_bytes), .workspace_suffix = "expected"});
            const auto actual = MoEOverlayCapacityResolver::resolve(input);
            for (int index = 0; index < 2; ++index)
            {
                const auto &bom = actual.resource(std::to_string(index))->bom();
                EXPECT_EQ(bom.bytes(PhysicalMemoryOwner::ExpertMigrationStaging), directory.total_bytes);
                EXPECT_EQ(bom.bytes(PhysicalMemoryOwner::ExecutionWorkspace), workspace);
                EXPECT_EQ(bom.bytes(PhysicalMemoryOwner::ExpertShadowSlots), 0u);
            }
        }
    }

    /** @brief Unsupported topologies and source geometry fail before any quota is certified. */
    TEST(MoEProjectionCapacity, RejectsUnimplementedOrInconsistentProjectionTopologies)
    {
        const auto source = manifest(ExpertWeightFormat::floating(TensorType::FP32));
        const auto p = plan(DeviceType::ROCm, 2, RoutedExpertComputePolicy::GateUpOwnedDownColumns);
        const auto valid = MoEOverlayCapacityAdmission::buildResolverInput(p, 24, source, budgets(p), {});
        auto bad = valid;
        bad.tiers.front().copy_policy = static_cast<MoEOverlayTierCopyPolicy>(255);
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolve(bad), std::invalid_argument);
        bad = valid;
        bad.layer_weight_manifest.front().projections.front().N = 767;
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolve(bad), std::invalid_argument);
        bad = valid;
        bad.tiers.front().participants.front().shadow_slots_per_layer = 1;
        bad.tiers.front().participants.front().maximum_concurrent_shadow_slots = 1;
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolve(bad), std::invalid_argument);
        bad = valid;
        auto second = bad.tiers.front();
        second.tier_index = 1;
        second.tier_name = "other";
        second.priority = -1;
        second.fallback = false;
        for (auto &participant : second.participants) participant.participant_id += 2;
        bad.tiers.push_back(second);
        EXPECT_THROW((void)MoEOverlayCapacityResolver::resolve(bad), std::invalid_argument);
    }
} // namespace llaminar2
