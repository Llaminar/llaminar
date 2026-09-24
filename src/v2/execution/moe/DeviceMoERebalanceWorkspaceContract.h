/**
 * @file DeviceMoERebalanceWorkspaceContract.h
 * @brief Canonical metadata-only workspace contract for device MoE rebalance.
 *
 * Dynamic same-tier GPU maintenance is captured as a graph participant whose
 * buffers remain live beside the serving graph family.  Admission runs before
 * device pointers or prepared engines exist, so the physical-memory authority
 * and the live compute stage must project their requirements from this same
 * pointer-free capacity value.  No caller may reproduce the buffer arithmetic.
 */

#pragma once

#include "DeviceMoERebalanceController.h"
#include "../local_execution/device/WorkspaceDescriptor.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace llaminar2
{
    /**
     * @brief Typed phase of one captured device-rebalance transaction.
     *
     * Phases may share one persistent workspace, but only the complete
     * PlanCopyApply maintenance transaction is retained as a standalone serial
     * graph-family participant.  JoinTransfer contributes ordering only and
     * therefore owns no workspace buffers.
     */
    enum class DeviceMoERebalanceStagePhase
    {
        /** Collect, plan, move, and atomically apply one maintenance wave. */
        PlanCopyApply,
        /** Pack local state for a following collective sideband. */
        CollectState,
        /** Plan and enqueue transfer work without joining its stream. */
        PlanAndCopy,
        /** Plan after another stage has gathered the state sideband. */
        PlanAndCopyAfterSideband,
        /** Pack requested payloads after command-sideband exchange. */
        PackCollectivePayloadAfterSideband,
        /** Unpack gathered payloads and publish transfer completion. */
        UnpackCollectivePayloadAfterSideband,
        /** Apply an already prepared wave to the inactive runtime bank. */
        Apply,
        /** Join the auxiliary transfer stream without allocating storage. */
        JoinTransfer,
    };

    /**
     * @brief Pointer-free capacity identity for one rebalance workspace family.
     *
     * Only fields that change buffer count or size are retained.  Runtime
     * policy such as cadence and economy thresholds belongs to
     * DeviceMoERebalanceConfig but must not create a second memory identity.
     */
    struct DeviceMoERebalanceWorkspaceCapacity
    {
        std::uint32_t num_layers = 0; ///< Complete runtime-table layer capacity.
        std::uint32_t num_experts = 0; ///< Logical experts in every routed layer.
        std::uint32_t participant_count = 0; ///< Native collective degree.
        std::uint32_t layer_window_count = 0; ///< Layers visible to this cursor; zero means all.
        std::uint32_t layer_wave_count = 0; ///< Layers planned per replay; zero means the window.
        std::uint32_t max_hot_replicas_per_participant = 0; ///< Command-plan width per layer.
        std::uint32_t routed_assignment_policy =
            kDeviceMoERebalanceAssignmentStaticOwner; ///< Static-owner or LLEP planner ABI.
        std::uint32_t flags = 0; ///< Publication flags that alter retained status storage.
        std::uint32_t local_transfer_slot_count = 0; ///< Addressable device-directory entries.
        std::uint32_t collective_payload_slot_capacity = 0; ///< Non-empty payloads moved per replay.
        DeviceMoERebalanceStagePhase phase =
            DeviceMoERebalanceStagePhase::PlanCopyApply; ///< Captured transaction phase.
        DeviceMoERebalanceTransferMode transfer_mode =
            DeviceMoERebalanceTransferMode::ResidentOnly; ///< Physical arrival transport.

        /**
         * @brief Project the workspace-relevant fields from a live stage config.
         * @param config Complete runtime controller configuration.
         * @param local_transfer_slot_count Addressable transfer directory size.
         * @param collective_payload_slot_capacity Captured compact payload width.
         * @param phase Exact stage transaction phase.
         * @param transfer_mode Exact transfer implementation.
         * @return Pointer-free capacity consumed by admission and graph stages.
         */
        [[nodiscard]] static DeviceMoERebalanceWorkspaceCapacity fromRuntime(
            const DeviceMoERebalanceConfig &config,
            std::uint32_t local_transfer_slot_count,
            std::uint32_t collective_payload_slot_capacity,
            DeviceMoERebalanceStagePhase phase,
            DeviceMoERebalanceTransferMode transfer_mode) noexcept;

        /** @return Number of histogram layers retained by one planning wave. */
        [[nodiscard]] std::size_t histogramLayerCount() const noexcept;

        /** @return Number of layer plans retained by parallel LLEP planning. */
        [[nodiscard]] std::size_t llepPlannerScratchCount() const noexcept;

        /** @return Maximum command entries in one local command buffer. */
        [[nodiscard]] std::size_t transferPlanCapacity() const noexcept;

        /** @return Number of independently retained command buffers. */
        [[nodiscard]] std::size_t commandBufferCount() const noexcept;

        /** @return Whether this identity owns transfer-slot apply storage. */
        [[nodiscard]] bool usesTransferSlots() const noexcept;

        /** @return Whether payloads use compact non-empty transfer slots. */
        [[nodiscard]] bool usesCompactTransferSlots() const noexcept;

        /** @return Whether payloads use a fixed-capacity transfer arena. */
        [[nodiscard]] bool usesFixedPayloadTransfer() const noexcept;

        /** @return Whether a collective payload send/receive lane is retained. */
        [[nodiscard]] bool usesCollectivePayloadLane() const noexcept;

        /** @return Whether an apply-status publication remains graph-visible. */
        [[nodiscard]] bool usesReadyWaveApply() const noexcept;

        /** @return Whether this phase executes the device-side planner. */
        [[nodiscard]] bool runsController() const noexcept;

        /** @return Whether the controller retains per-layer LLEP scratch. */
        [[nodiscard]] bool usesParallelLLEPPlanning() const noexcept;

        /** @return Effective number of non-empty payload slots per participant. */
        [[nodiscard]] std::size_t payloadSlotCapacity() const noexcept;

        /** @return Local collective payload slots after transport-mode expansion. */
        [[nodiscard]] std::size_t collectivePayloadSlotCount() const noexcept;

        /** @return Bounded terminal-history edge capacity derived from model geometry. */
        [[nodiscard]] std::size_t movementJournalEdgeCapacity() const;
        /** @return Worst-case paired-wave count fitting the same bounded edge history. */
        [[nodiscard]] std::size_t movementJournalWaveCapacity() const;
    };

    /**
     * @brief Complete immutable binding of capacity, payload stride, and names.
     *
     * The payload stride is codebook/model dependent and is bound only after
     * the authenticated expert-weight manifest is available.  Keeping this as
     * a distinct type prevents an unbound zero-byte payload from reaching the
     * physical-memory ledger.
     */
    struct DeviceMoERebalanceWorkspaceBinding
    {
        DeviceMoERebalanceWorkspaceCapacity capacity; ///< Topology and policy capacity.
        std::uint64_t collective_payload_slot_bytes = 0; ///< Aligned bytes in one wire slot.
        std::string workspace_suffix; ///< Stable namespace for every descriptor.
    };

    /**
     * @brief Sole buffer and byte authority for captured device MoE maintenance.
     */
    class DeviceMoERebalanceWorkspaceContract final
    {
    public:
        static constexpr const char *WS_LOCAL_HISTOGRAM = "moe_rebalance_local_histogram";
        static constexpr const char *WS_GATHERED_HISTOGRAM = "moe_rebalance_gathered_histogram";
        static constexpr const char *WS_TRANSFER_PLAN = "moe_rebalance_transfer_plan";
        static constexpr const char *WS_TRANSFER_PLAN_COUNT = "moe_rebalance_transfer_plan_count";
        static constexpr const char *WS_COMMAND_HEADER = "moe_rebalance_command_header";
        static constexpr const char *WS_CONTROLLER_STATE = "moe_rebalance_controller_state";
        static constexpr const char *WS_MOVEMENT_WAVES = "moe_rebalance_movement_waves";
        static constexpr const char *WS_MOVEMENT_EDGES = "moe_rebalance_movement_edges";
        static constexpr const char *WS_PLACEMENT_PLAN_SCRATCH =
            "moe_rebalance_placement_plan_scratch";
        static constexpr const char *WS_LLEP_LAYER_PLANS = "moe_rebalance_llep_layer_plans";
        static constexpr const char *WS_GATHERED_TRANSFER_PLAN = "moe_rebalance_gathered_transfer_plan";
        static constexpr const char *WS_GATHERED_COMMAND_HEADER = "moe_rebalance_gathered_command_header";
        static constexpr const char *WS_TRANSFER_SLOT_CLAIM_INDEX =
            "moe_rebalance_transfer_slot_claim_index";
        static constexpr const char *WS_WAVE_STATE = "moe_rebalance_wave_state";
        static constexpr const char *WS_GATHERED_WAVE_STATE = "moe_rebalance_gathered_wave_state";
        static constexpr const char *WS_STATUS = "moe_rebalance_status";
        static constexpr const char *WS_LOCAL_DIRECTORY = "moe_rebalance_local_directory";
        static constexpr const char *WS_LOCAL_SOURCE_DESCRIPTORS = "moe_rebalance_local_source_descriptors";
        static constexpr const char *WS_LOCAL_TRANSFER_PAYLOAD = "moe_rebalance_local_transfer_payload";
        static constexpr const char *WS_GATHERED_TRANSFER_PAYLOAD = "moe_rebalance_gathered_transfer_payload";
        static constexpr const char *WS_COPY_STATUS = "moe_rebalance_copy_status";
        static constexpr const char *WS_GATHERED_COPY_STATUS = "moe_rebalance_gathered_copy_status";
        static constexpr const char *WS_APPLY_STATUS = "moe_rebalance_apply_status";

        /**
         * @brief Derive the aligned collective slot stride from wire payload bytes.
         * @param wire_payload_bytes Largest complete serialized expert payload.
         * @return Payload plus descriptor header rounded to 256 bytes.
         * @throws std::invalid_argument for zero payload bytes.
         * @throws std::overflow_error when the aligned stride cannot be represented.
         */
        [[nodiscard]] static std::uint64_t collectivePayloadSlotBytes(
            std::size_t wire_payload_bytes);

        /**
         * @brief Build every stable-name buffer required by one transaction.
         * @param binding Complete pointer-free workspace identity.
         * @return Exact descriptor inventory consumed by WorkspaceAllocator.
         * @throws std::invalid_argument for an incomplete or contradictory binding.
         * @throws std::overflow_error when descriptor arithmetic cannot fit size_t.
         */
        [[nodiscard]] static WorkspaceRequirements requirements(
            const DeviceMoERebalanceWorkspaceBinding &binding);

        /**
         * @brief Sum physical buffer payload bytes without alignment padding.
         * @param binding Complete workspace identity.
         * @return Exact logical bytes reported by the compute stage.
         */
        [[nodiscard]] static std::size_t logicalBytes(
            const DeviceMoERebalanceWorkspaceBinding &binding);

        /**
         * @brief Sum independently aligned physical buffer extents.
         * @param binding Complete workspace identity.
         * @return Conservative compositional allocation consumed by admission.
         *
         * The serial-family interval planner may place these persistent names in
         * holes left by other participants. Admission cannot depend on those
         * incidental holes, so it charges each canonical descriptor once with
         * the same alignment WorkspaceAllocator enforces.
         */
        [[nodiscard]] static std::size_t allocationBytes(
            const DeviceMoERebalanceWorkspaceBinding &binding);
    };
} // namespace llaminar2
