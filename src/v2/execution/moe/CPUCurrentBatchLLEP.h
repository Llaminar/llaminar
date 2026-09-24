/**
 * @file CPUCurrentBatchLLEP.h
 * @brief Typed CPU current-batch least-loaded expert transaction contracts.
 *
 * CPU LLEP is a transient graph transaction.  A begin stage plans routed-row
 * destinations and materializes only the missing packed expert replicas.  The
 * routed expert stage consumes the immutable destination publication, and a
 * restore stage removes those transient replicas before the next layer or
 * decode graph can observe them. `MoEOverlayResidencyAuthority` remains the
 * sole placement authority and pins the durable parent epoch. The device
 * orchestrator is only the physical executor for packed-weight transfer and
 * graph-local prepared-engine publication.
 */

#pragma once

#include "ExpertWeightTransfer.h"
#include "LeastLoadedExpertAssignment.h"
#include "MoEOverlayResidencyAuthority.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class IGlobalTPContext;

    /** @brief Lifecycle phase represented by one CPU LLEP graph stage. */
    enum class CPUCurrentBatchLLEPPhase : uint8_t
    {
        Begin,
        Restore,
    };

    /**
     * @brief Graph-owned publication shared by begin, expert, and restore stages.
     *
     * All storage is sized when the graph is built.  Execute-time planning only
     * overwrites existing elements, keeping the CPU production path free of
     * per-route allocation and making overflow a fatal, attributable error.
     */
    struct CPUCurrentBatchLLEPTransactionState
    {
        int layer_idx = -1;
        int seq_len = 0;
        int top_k = 0;
        int num_experts = 0;
        int participant_count = 0;
        int participant_id = -1;

        bool active = false;
        uint64_t transaction_epoch = 0;
        /** Durable ExpertOverlay epoch borrowed by this transient child. */
        uint64_t durable_parent_epoch = 0;
        /** Routed domain whose local participant indices appear below. */
        std::string durable_domain;
        /** Lease preventing parent-engine retirement until restore completes. */
        std::optional<MoEOverlayResidencyAuthority::TicketLease>
            durable_epoch_lease;

        std::vector<uint32_t> owner_participants;
        std::vector<bool> owner_mask;
        std::vector<bool> transient_resident_mask;

        std::vector<uint64_t> expert_loads;
        std::vector<uint32_t> sorted_experts;
        std::vector<uint64_t> pending_load;
        std::vector<uint64_t> assigned_load;
        std::vector<least_loaded_ep::LeastLoadedExpertAssignmentSpan> spans;
        std::vector<least_loaded_ep::LeastLoadedExpertWeightTransfer> transfers;
        least_loaded_ep::LeastLoadedExpertAssignmentStatus status;

        std::vector<uint64_t> expert_route_offsets;
        std::vector<uint64_t> expert_route_cursors;
        std::vector<uint32_t> destination_by_expert_occurrence;
        std::vector<uint32_t> destination_by_flat_route;

        std::vector<uint64_t> gathered_route_hashes;
        std::vector<uint64_t> gathered_plan_hashes;

        /** @return Number of physical route slots in this graph invocation. */
        [[nodiscard]] size_t routeSlotCount() const noexcept
        {
            return static_cast<size_t>(seq_len) * static_cast<size_t>(top_k);
        }

        /**
         * @brief Return the participant assigned to an original route slot.
         * @throws std::out_of_range when the publication is inactive or invalid.
         */
        [[nodiscard]] uint32_t destinationForFlatRoute(size_t flat_route) const;
    };

    /**
     * @brief Participant-local sparse expert endpoint borrowed by one CPU LLEP child.
     *
     * The interface deliberately exposes physical prepared-weight operations,
     * not placement decisions.  The ExpertOverlay authority supplies the
     * durable owner row and the graph stage supplies the deterministic route
     * assignment.  Implementations may borrow immutable parent-bank engines
     * and retain transient arrivals, but must not mutate the durable epoch,
     * publish a second owner map, or leave child state observable after
     * @ref discardCPUCurrentBatchLLEPTransientResidency returns.
     */
    class ICPUCurrentBatchLLEPExpertConsumer
    {
    public:
        virtual ~ICPUCurrentBatchLLEPExpertConsumer() = default;

        /** @return Transformer layer served by this exact sparse endpoint. */
        [[nodiscard]] virtual int cpuCurrentBatchLLEPLayerIndex() const noexcept = 0;

        /** @return Domain-local participant id served by this endpoint. */
        [[nodiscard]] virtual int cpuCurrentBatchLLEPParticipantId() const noexcept = 0;

        /** @return Whether this endpoint is a CPU prepared-expert consumer. */
        [[nodiscard]] virtual bool cpuCurrentBatchLLEPUsesCPU() const noexcept = 0;

        /**
         * @brief Clone one packed expert from an immutable durable parent bank.
         * @param expert_id Global expert id within the layer.
         * @param durable_parent_epoch Exact authority epoch pinned by the child.
         * @return Three complete packed CPU projections owned by the caller.
         */
        [[nodiscard]] virtual ExpertPackedWeights
        cloneCPUCurrentBatchLLEPPreparedExpert(
            int expert_id,
            uint64_t durable_parent_epoch) const = 0;

        /**
         * @brief Install every arrival needed by the already-planned child.
         * @param state Complete transaction publication, still inactive while installing.
         * @param arrivals Destination-local final prepared engines, or null when none arrive.
         * @return True only when every transient resident has complete engines.
         */
        virtual bool installCPUCurrentBatchLLEPTransientResidency(
            const CPUCurrentBatchLLEPTransactionState &state,
            const std::unordered_map<int, PreparedExpertEngines> *arrivals) = 0;

        /**
         * @brief Remove all graph-local transient arrivals and reveal only the parent bank.
         * @param state Transaction whose child residency is being closed or rolled back.
         * @return True when no transient child engine remains observable.
         */
        virtual bool discardCPUCurrentBatchLLEPTransientResidency(
            const CPUCurrentBatchLLEPTransactionState &state) noexcept = 0;
    };

    /**
     * @brief Physical executor for an authority-owned CPU LLEP child transaction.
     *
     * Implementations materialize final packed CPU expert engines directly on
     * their destination participant.  No raw-weight fallback or ownership move
     * is permitted: the parent ExpertOverlay epoch remains authoritative and
     * pinned for the whole batch. Implementations must not select owners,
     * publish a durable epoch, or retain the parent lease after restore.
     */
    class ICPUCurrentBatchLLEPPhysicalExecutor
    {
    public:
        virtual ~ICPUCurrentBatchLLEPPhysicalExecutor() = default;

        /**
         * @brief Materialize and publish a fully planned transient transaction.
         * @param state Complete deterministic route and transfer publication.
         * @param expert_consumer Exact participant-local sparse expert consumer.
         * @param tp_ctx Cross-rank CPU domain used for packed expert transfer.
         * @return true only when every requested arrival and mask is published.
         */
        virtual bool materializeCPUCurrentBatchLLEPTransaction(
            CPUCurrentBatchLLEPTransactionState &state,
            ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
            IGlobalTPContext &tp_ctx) = 0;

        /**
         * @brief Restore owner-only residency after routed expert execution.
         * @param state Active transaction to close.
         * @param expert_consumer Exact participant-local sparse expert consumer.
         * @param tp_ctx Domain whose participant identity must match the begin.
         * @return true only after transient replicas are no longer observable.
         */
        virtual bool restoreCPUCurrentBatchLLEPPhysicalState(
            CPUCurrentBatchLLEPTransactionState &state,
            ICPUCurrentBatchLLEPExpertConsumer &expert_consumer,
            IGlobalTPContext &tp_ctx) = 0;
    };

} // namespace llaminar2
