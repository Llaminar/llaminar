/**
 * @file DeviceMoERebalanceWorkspaceContract.cpp
 * @brief Implements the canonical device MoE rebalance workspace contract.
 */

#include "DeviceMoERebalanceWorkspaceContract.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace llaminar2
{
    namespace
    {
        constexpr std::size_t kWorkspaceAlignment = 256u;

        /** @return Checked product used by every workspace extent formula. */
        std::size_t checkedMultiply(
            std::size_t lhs,
            std::size_t rhs,
            const char *label)
        {
            if (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs)
            {
                throw std::overflow_error(
                    std::string("Device MoE rebalance workspace overflow in ") +
                    (label ? label : "unnamed product"));
            }
            return lhs * rhs;
        }

        /** @return Checked sum used by logical and aligned BOM totals. */
        std::size_t checkedAdd(
            std::size_t lhs,
            std::size_t rhs,
            const char *label)
        {
            if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
            {
                throw std::overflow_error(
                    std::string("Device MoE rebalance workspace overflow in ") +
                    (label ? label : "unnamed sum"));
            }
            return lhs + rhs;
        }

        /** @return Checked 256-byte round-up matching WorkspaceAllocator. */
        std::size_t alignedBytes(std::size_t bytes)
        {
            if (bytes >
                std::numeric_limits<std::size_t>::max() -
                    (kWorkspaceAlignment - 1u))
            {
                throw std::overflow_error(
                    "Device MoE rebalance workspace alignment overflows size_t");
            }
            return (bytes + kWorkspaceAlignment - 1u) &
                   ~(kWorkspaceAlignment - 1u);
        }

        /** @return Stable buffer name inside one workspace namespace. */
        std::string bufferName(
            const char *base,
            const std::string &suffix)
        {
            return std::string(base) + "_" + suffix;
        }

        /** @brief Reject incomplete capacity before it reaches memory admission. */
        void validateCapacity(
            const DeviceMoERebalanceWorkspaceBinding &binding)
        {
            const auto &capacity = binding.capacity;
            if (capacity.num_layers == 0u || capacity.num_experts == 0u ||
                capacity.participant_count == 0u ||
                capacity.participant_count > kDeviceMoEMaxParticipants ||
                binding.workspace_suffix.empty())
            {
                throw std::invalid_argument(
                    "Device MoE rebalance workspace requires positive model, participant, and namespace geometry");
            }

            switch (capacity.phase)
            {
            case DeviceMoERebalanceStagePhase::PlanCopyApply:
            case DeviceMoERebalanceStagePhase::CollectState:
            case DeviceMoERebalanceStagePhase::PlanAndCopy:
            case DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband:
            case DeviceMoERebalanceStagePhase::PackCollectivePayloadAfterSideband:
            case DeviceMoERebalanceStagePhase::UnpackCollectivePayloadAfterSideband:
            case DeviceMoERebalanceStagePhase::Apply:
            case DeviceMoERebalanceStagePhase::JoinTransfer:
                break;
            default:
                throw std::invalid_argument(
                    "Device MoE rebalance workspace received an invalid phase");
            }

            if (capacity.usesTransferSlots() &&
                capacity.local_transfer_slot_count == 0u)
            {
                throw std::invalid_argument(
                    "Transfer-backed device MoE rebalance workspace requires a non-empty directory");
            }
            if (capacity.usesCollectivePayloadLane() &&
                (binding.collective_payload_slot_bytes == 0u ||
                 capacity.payloadSlotCapacity() == 0u))
            {
                throw std::invalid_argument(
                    "Collective device MoE rebalance workspace requires bound payload bytes and capacity");
            }
        }

        /** @brief Append one exact persistent descriptor. */
        void append(
            WorkspaceRequirements &requirements,
            const char *base,
            const std::string &suffix,
            std::size_t bytes)
        {
            requirements.buffers.push_back({
                bufferName(base, suffix),
                bytes,
                kWorkspaceAlignment,
                true,
            });
        }
    } // namespace

    DeviceMoERebalanceWorkspaceCapacity
    DeviceMoERebalanceWorkspaceCapacity::fromRuntime(
        const DeviceMoERebalanceConfig &config,
        std::uint32_t local_transfer_slot_count,
        std::uint32_t collective_payload_slot_capacity,
        DeviceMoERebalanceStagePhase phase,
        DeviceMoERebalanceTransferMode transfer_mode) noexcept
    {
        return {
            .num_layers = config.num_layers,
            .num_experts = config.num_experts,
            .participant_count = config.participant_count,
            .layer_window_count = config.layer_window_count,
            .layer_wave_count = config.layer_wave_count,
            .max_hot_replicas_per_participant =
                config.max_hot_replicas_per_participant,
            .routed_assignment_policy = config.routed_assignment_policy,
            .flags = config.flags,
            .local_transfer_slot_count = local_transfer_slot_count,
            .collective_payload_slot_capacity =
                collective_payload_slot_capacity,
            .phase = phase,
            .transfer_mode = transfer_mode,
        };
    }

    std::size_t
    DeviceMoERebalanceWorkspaceCapacity::histogramLayerCount() const noexcept
    {
        const std::uint32_t window_count = layer_window_count == 0u
            ? num_layers
            : std::min(layer_window_count, num_layers);
        const std::uint32_t wave_count = layer_wave_count == 0u
            ? window_count
            : std::min(layer_wave_count, window_count);
        return static_cast<std::size_t>(std::max<std::uint32_t>(1u, wave_count));
    }

    std::size_t
    DeviceMoERebalanceWorkspaceCapacity::llepPlannerScratchCount() const noexcept
    {
        const std::uint32_t window_count = layer_window_count == 0u
            ? num_layers
            : std::min(layer_window_count, num_layers);
        return static_cast<std::size_t>(
            std::max<std::uint32_t>(1u, window_count));
    }

    std::size_t
    DeviceMoERebalanceWorkspaceCapacity::transferPlanCapacity() const noexcept
    {
        DeviceMoERebalanceConfig sizing;
        sizing.num_layers = num_layers;
        sizing.layer_window_count = layer_window_count;
        sizing.layer_wave_count = layer_wave_count;
        sizing.max_hot_replicas_per_participant =
            max_hot_replicas_per_participant;
        sizing.participant_count = participant_count;
        return static_cast<std::size_t>(
            deviceMoERebalanceCommandPlanCapacity(sizing, transfer_mode));
    }

    std::size_t
    DeviceMoERebalanceWorkspaceCapacity::commandBufferCount() const noexcept
    {
        return usesTransferSlots() ? 2u : 1u;
    }

    bool DeviceMoERebalanceWorkspaceCapacity::usesTransferSlots() const noexcept
    {
        return deviceMoERebalanceModeUsesTransferSlots(transfer_mode);
    }

    bool DeviceMoERebalanceWorkspaceCapacity::usesCompactTransferSlots() const noexcept
    {
        return usesTransferSlots() &&
               transfer_mode ==
                   DeviceMoERebalanceTransferMode::CompactTransferSlots;
    }

    bool DeviceMoERebalanceWorkspaceCapacity::usesFixedPayloadTransfer() const noexcept
    {
        return usesTransferSlots() &&
               deviceMoERebalanceModeMovesFixedPayloadCapacity(transfer_mode);
    }

    bool DeviceMoERebalanceWorkspaceCapacity::usesCollectivePayloadLane() const noexcept
    {
        return usesTransferSlots() &&
               deviceMoERebalanceModeUsesCollectivePayloadLane(transfer_mode);
    }

    bool DeviceMoERebalanceWorkspaceCapacity::usesReadyWaveApply() const noexcept
    {
        return usesTransferSlots() ||
               hasDeviceMoERebalanceFlag(
                   flags,
                   DeviceMoERebalanceFlags::DeferRuntimeApply);
    }

    bool DeviceMoERebalanceWorkspaceCapacity::runsController() const noexcept
    {
        return phase == DeviceMoERebalanceStagePhase::PlanCopyApply ||
               phase == DeviceMoERebalanceStagePhase::PlanAndCopy ||
               phase ==
                   DeviceMoERebalanceStagePhase::PlanAndCopyAfterSideband;
    }

    bool DeviceMoERebalanceWorkspaceCapacity::usesParallelLLEPPlanning() const noexcept
    {
        return runsController() &&
               routed_assignment_policy ==
                   kDeviceMoERebalanceAssignmentLeastLoadedResident;
    }

    std::size_t
    DeviceMoERebalanceWorkspaceCapacity::payloadSlotCapacity() const noexcept
    {
        if (!usesCollectivePayloadLane() || local_transfer_slot_count == 0u)
            return 0u;
        const std::size_t captured = collective_payload_slot_capacity == 0u
            ? static_cast<std::size_t>(local_transfer_slot_count)
            : std::min<std::size_t>(
                  collective_payload_slot_capacity,
                  local_transfer_slot_count);
        return std::min(transferPlanCapacity(), captured);
    }

    std::size_t
    DeviceMoERebalanceWorkspaceCapacity::collectivePayloadSlotCount() const noexcept
    {
        const std::size_t slots = payloadSlotCapacity();
        if (usesCompactTransferSlots())
            return slots;
        return slots * static_cast<std::size_t>(participant_count);
    }

    std::size_t DeviceMoERebalanceWorkspaceCapacity::movementJournalEdgeCapacity() const
    {
        // Retain one complete expert turnover per participant between terminal
        // observations. This is diagnostic storage, not a movement/admission
        // limit: exhaustion is explicit and cannot stall serving or wrap.
        if (!usesTransferSlots())
            return 0;
        const auto count = checkedMultiply(checkedMultiply(num_layers, num_experts,
            "movement journal layer experts"), participant_count, "movement journal participants");
        if (count > UINT32_MAX)
            throw std::overflow_error("Device movement journal exceeds its wire index capacity");
        return std::max<std::size_t>(2, count);
    }

    std::size_t DeviceMoERebalanceWorkspaceCapacity::movementJournalWaveCapacity() const
    {
        return movementJournalEdgeCapacity() / 2;
    }

    std::uint64_t
    DeviceMoERebalanceWorkspaceContract::collectivePayloadSlotBytes(
        std::size_t wire_payload_bytes)
    {
        if (wire_payload_bytes == 0u)
        {
            throw std::invalid_argument(
                "Device MoE rebalance wire payload must be non-zero");
        }
        constexpr std::uint64_t alignment = 256u;
        const std::uint64_t header = sizeof(DeviceMoEExpertDirectoryEntry);
        const std::uint64_t wire = static_cast<std::uint64_t>(wire_payload_bytes);
        if (wire > std::numeric_limits<std::uint64_t>::max() - header -
                       (alignment - 1u))
        {
            throw std::overflow_error(
                "Device MoE rebalance collective payload stride overflows uint64_t");
        }
        return ((wire + header + alignment - 1u) / alignment) * alignment;
    }

    WorkspaceRequirements DeviceMoERebalanceWorkspaceContract::requirements(
        const DeviceMoERebalanceWorkspaceBinding &binding)
    {
        validateCapacity(binding);
        WorkspaceRequirements requirements;
        const auto &capacity = binding.capacity;
        if (capacity.phase == DeviceMoERebalanceStagePhase::JoinTransfer)
            return requirements;

        const std::size_t participants = capacity.participant_count;
        const std::size_t command_buffers = capacity.commandBufferCount();
        const std::size_t plan_entries = capacity.transferPlanCapacity();
        const std::size_t local_histogram_entries = checkedMultiply(
            capacity.histogramLayerCount(),
            capacity.num_experts,
            "local histogram entries");
        const std::size_t gathered_histogram_entries = checkedMultiply(
            local_histogram_entries,
            participants,
            "gathered histogram entries");

        append(
            requirements,
            WS_LOCAL_HISTOGRAM,
            binding.workspace_suffix,
            checkedMultiply(
                local_histogram_entries,
                sizeof(std::uint64_t),
                "local histogram bytes"));
        append(
            requirements,
            WS_GATHERED_HISTOGRAM,
            binding.workspace_suffix,
            checkedMultiply(
                gathered_histogram_entries,
                sizeof(std::uint64_t),
                "gathered histogram bytes"));
        append(
            requirements,
            WS_TRANSFER_PLAN,
            binding.workspace_suffix,
            checkedMultiply(
                checkedMultiply(
                    command_buffers,
                    plan_entries,
                    "local command entries"),
                sizeof(DeviceMoERebalancePlanEntry),
                "local command bytes"));
        append(
            requirements,
            WS_TRANSFER_PLAN_COUNT,
            binding.workspace_suffix,
            checkedMultiply(
                command_buffers,
                sizeof(std::uint32_t),
                "command-count bytes"));
        append(
            requirements,
            WS_COMMAND_HEADER,
            binding.workspace_suffix,
            checkedMultiply(
                command_buffers,
                sizeof(DeviceMoERebalanceCommandBufferHeader),
                "command-header bytes"));
        append(
            requirements,
            WS_CONTROLLER_STATE,
            binding.workspace_suffix,
            sizeof(DeviceMoERebalanceGraphControllerState));
        if (capacity.usesTransferSlots())
        {
            append(requirements, WS_MOVEMENT_WAVES, binding.workspace_suffix,
                checkedMultiply(capacity.movementJournalWaveCapacity(),
                    sizeof(DeviceMoERebalanceMovementWave), "movement journal wave bytes"));
            append(requirements, WS_MOVEMENT_EDGES, binding.workspace_suffix,
                checkedMultiply(capacity.movementJournalEdgeCapacity(),
                    sizeof(DeviceMoERebalanceMovementEdge), "movement journal edge bytes"));
        }
        /* Planning writes a private RCU bank before any live table can publish it. */
        append(
            requirements,
            WS_PLACEMENT_PLAN_SCRATCH,
            binding.workspace_suffix,
            sizeof(DeviceMoEPlacementBank));
        if (capacity.usesParallelLLEPPlanning())
        {
            append(
                requirements,
                WS_LLEP_LAYER_PLANS,
                binding.workspace_suffix,
                checkedMultiply(
                    capacity.llepPlannerScratchCount(),
                    sizeof(DeviceMoELLEPLayerPlanScratch),
                    "parallel LLEP layer-plan bytes"));
        }
        append(
            requirements,
            WS_WAVE_STATE,
            binding.workspace_suffix,
            checkedMultiply(
                command_buffers,
                sizeof(DeviceMoERebalanceWaveState),
                "wave-state bytes"));
        append(
            requirements,
            WS_STATUS,
            binding.workspace_suffix,
            sizeof(DeviceMoERebalanceStatus));
        if (capacity.usesReadyWaveApply())
        {
            append(
                requirements,
                WS_APPLY_STATUS,
                binding.workspace_suffix,
                sizeof(DeviceMoERebalanceApplyStatus));
        }

        if (!capacity.usesTransferSlots())
            return requirements;

        append(
            requirements,
            WS_COPY_STATUS,
            binding.workspace_suffix,
            sizeof(DeviceMoERebalanceApplyStatus));
        append(
            requirements,
            WS_GATHERED_COPY_STATUS,
            binding.workspace_suffix,
            checkedMultiply(
                participants,
                sizeof(DeviceMoERebalanceApplyStatus),
                "gathered copy-status bytes"));
        const std::size_t gathered_command_entries = checkedMultiply(
            checkedMultiply(
                plan_entries,
                command_buffers,
                "buffered command entries"),
            participants,
            "gathered command entries");
        append(
            requirements,
            WS_GATHERED_TRANSFER_PLAN,
            binding.workspace_suffix,
            checkedMultiply(
                gathered_command_entries,
                sizeof(DeviceMoERebalancePlanEntry),
                "gathered command bytes"));
        append(
            requirements,
            WS_GATHERED_COMMAND_HEADER,
            binding.workspace_suffix,
            checkedMultiply(
                checkedMultiply(
                    participants,
                    command_buffers,
                    "gathered command headers"),
                sizeof(DeviceMoERebalanceCommandBufferHeader),
                "gathered command-header bytes"));
        append(
            requirements,
            WS_GATHERED_WAVE_STATE,
            binding.workspace_suffix,
            checkedMultiply(
                checkedMultiply(
                    participants,
                    command_buffers,
                    "gathered wave states"),
                sizeof(DeviceMoERebalanceWaveState),
                "gathered wave-state bytes"));
        append(
            requirements,
            WS_TRANSFER_SLOT_CLAIM_INDEX,
            binding.workspace_suffix,
            deviceMoETransferSlotClaimIndexBytes(
                capacity.local_transfer_slot_count));

        if (capacity.usesCompactTransferSlots())
        {
            const std::size_t local_source_entries = checkedMultiply(
                participants,
                checkedMultiply(
                    command_buffers,
                    plan_entries,
                    "buffered source descriptors"),
                "participant source descriptors");
            append(
                requirements,
                WS_LOCAL_SOURCE_DESCRIPTORS,
                binding.workspace_suffix,
                checkedMultiply(
                    local_source_entries,
                    sizeof(DeviceMoEExpertDirectoryEntry),
                    "local source-descriptor bytes"));
        }

        if (capacity.usesCollectivePayloadLane())
        {
            if (binding.collective_payload_slot_bytes >
                std::numeric_limits<std::size_t>::max())
            {
                throw std::overflow_error(
                    "Device MoE rebalance payload stride exceeds size_t");
            }
            const std::size_t payload_slot_bytes =
                static_cast<std::size_t>(
                    binding.collective_payload_slot_bytes);
            const std::size_t local_payload_bytes = checkedMultiply(
                capacity.collectivePayloadSlotCount(),
                payload_slot_bytes,
                "local collective payload bytes");
            append(
                requirements,
                WS_LOCAL_TRANSFER_PAYLOAD,
                binding.workspace_suffix,
                local_payload_bytes);
            append(
                requirements,
                WS_GATHERED_TRANSFER_PAYLOAD,
                binding.workspace_suffix,
                checkedMultiply(
                    local_payload_bytes,
                    participants,
                    "gathered collective payload bytes"));
        }

        if (capacity.usesFixedPayloadTransfer())
        {
            const std::size_t directory_entries = checkedMultiply(
                capacity.num_layers,
                capacity.num_experts,
                "local directory entries");
            append(
                requirements,
                WS_LOCAL_DIRECTORY,
                binding.workspace_suffix,
                checkedMultiply(
                    directory_entries,
                    sizeof(DeviceMoEExpertDirectoryEntry),
                    "local directory bytes"));
        }
        return requirements;
    }

    std::size_t DeviceMoERebalanceWorkspaceContract::logicalBytes(
        const DeviceMoERebalanceWorkspaceBinding &binding)
    {
        std::size_t total = 0u;
        for (const auto &descriptor : requirements(binding).buffers)
        {
            total = checkedAdd(
                total,
                descriptor.size_bytes,
                "logical workspace total");
        }
        return total;
    }

    std::size_t DeviceMoERebalanceWorkspaceContract::allocationBytes(
        const DeviceMoERebalanceWorkspaceBinding &binding)
    {
        std::size_t total = 0u;
        for (const auto &descriptor : requirements(binding).buffers)
        {
            total = checkedAdd(
                total,
                alignedBytes(descriptor.size_bytes),
                "aligned workspace total");
        }
        return total;
    }
} // namespace llaminar2
