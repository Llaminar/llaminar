/**
 * @file MoEExpertDispatchStage.h
 * @brief Host-side routed-row dispatch descriptor stage.
 */

#pragma once

#include "../IComputeStage.h"
#include "../StageParamsBase.h"
#include "../../../memory/BufferId.h"
#include "../../moe/MoERoutedExpertPlacementPlan.h"
#include "../../moe/MoEExpertTokenRowTransfer.h"
#include "../../moe/MoEExpertOwnerMap.h"
#include "../../moe/MoEOverlayResidencyAuthority.h"
#include "../../moe/CPUCurrentBatchLLEP.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace llaminar2
{
    class MoEOverlayDispatchTicketStorage;

    /**
     * @brief Sole owner responsible for closing one host dispatch epoch lease.
     *
     * The mapped GPU parent and the host sparse descriptor have independent
     * residency readers. A mapped lane therefore cannot change who retires the
     * host descriptor: ordinary dispatch closes at its final ordered sparse
     * return, while current-batch LLEP closes at its typed restore boundary.
     */
    enum class MoEOverlayHostDispatchLeaseOwner : std::uint8_t
    {
        None = 0,          ///< Graph-frozen placement acquired no host lease.
        FinalSparseReturn, ///< Final ordered host return owns lease retirement.
        CurrentBatchLLEP,  ///< The current-batch LLEP restore owns retirement.
    };

    /**
     * @brief Exact action assigned to one sparse return boundary.
     */
    enum class MoEOverlayHostDispatchLeaseTerminal : std::uint8_t
    {
        Retain = 0, ///< This boundary is not the descriptor's terminal owner.
        Release,    ///< This is the final ordered host sparse return.
    };

    /**
     * @brief Resolve lease action from typed ownership and return order.
     * @param owner Unique request-lifecycle owner selected before graph wiring.
     * @param final_ordered_return Whether this is the final host return edge.
     * @return Release only for the exact final ordinary-dispatch boundary.
     */
    [[nodiscard]] constexpr MoEOverlayHostDispatchLeaseTerminal
    moeOverlayHostDispatchLeaseTerminal(
        MoEOverlayHostDispatchLeaseOwner owner,
        bool final_ordered_return) noexcept
    {
        return owner ==
                       MoEOverlayHostDispatchLeaseOwner::FinalSparseReturn &&
                   final_ordered_return
                   ? MoEOverlayHostDispatchLeaseTerminal::Release
                   : MoEOverlayHostDispatchLeaseTerminal::Retain;
    }

    struct MoEExpertDispatchEntry
    {
        int token_row = -1;
        int route_slot = -1;
        int expert_id = -1;
        float route_weight = 0.0f;
        /**
         * Exact current-batch destination, or -1 for ordinary owner-filtered
         * dispatch. Sparse transport consumes this before putting bytes on the
         * wire, so transient replicas never execute an unassigned route.
         */
        int destination_participant = -1;
    };

    struct MoEExpertTierDispatch
    {
        int tier_index = -1;
        std::string tier_name;
        std::string domain;
        bool fallback = false;
        bool transfer_required = false;
        MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::None;
        MoEExpertTransferVolume transfer_volume;
        std::vector<MoEExpertDispatchEntry> entries;

        /// Original dense token-row indices selected for this tier. Sparse
        /// transfer helpers use these indices to scatter-add returned rows.
        std::vector<int> token_rows;
    };

    struct MoEExpertDispatchOutput
    {
        int seq_len = 0;
        int logical_seq_len = 0;
        int top_k = 0;
        int d_model = 0;
        std::string continuation_domain;
        std::vector<MoEExpertTierDispatch> tiers;
        std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_lifetime;

        /**
         * Exact immutable residency epoch used to build this descriptor.
         * Zero means the stage used its graph-frozen static placement.
         */
        uint64_t residency_epoch = 0;

        /**
         * Keeps the residency epoch live through every sparse dispatch,
         * participant-local expert, and return/reduce operation.  Only the
         * final continuation-root return may release this lease.
         */
        std::shared_ptr<MoEOverlayResidencyAuthority::TicketLease>
            residency_lease;

        size_t estimatedTransferBytes() const
        {
            size_t total = 0;
            for (const auto &tier : tiers)
                total += tier.transfer_volume.totalBytes();
            return total;
        }

        size_t denseTransferBytes() const
        {
            size_t total = 0;
            for (const auto &tier : tiers)
            {
                if (tier.transfer_required)
                    total += tier.transfer_volume.denseTotalBytes();
            }
            return total;
        }
    };

    /**
     * @brief Converts router top-k tensors into per-tier host work descriptors.
     *
     * Phase 4 intentionally stops at descriptors: it does not copy hidden rows,
     * launch expert kernels, or perform cross-domain transfer.
     */
    class MoEExpertDispatchStage : public IComputeStage
    {
    public:
        /**
         * @brief Exact routing-history publication owned by a manual boundary.
         *
         * Captured heterogeneous GPU graphs already publish their router
         * tensors into a fixed-capacity host ticket.  The following manual
         * dispatch boundary may therefore merge those exact real rows without
         * adding another device transfer or synchronizing an inference stream.
         * Grouped-verifier callers must not use this contract until accepted
         * prefix length is known; speculative rows are not production demand.
         */
        struct RoutingEvidencePublication
        {
            DecodeExpertHistogram *histogram = nullptr;
            ExpertHistogramSource source =
                ExpertHistogramSource::DecodeToken;
        };

        struct Params
        {
            STAGE_PARAMS_COMMON_FIELDS;

            const ITensor *routing_indices = nullptr; ///< FP32 expert ids [seq_len, top_k] or flat
            const ITensor *routing_weights = nullptr; ///< FP32 route weights [seq_len, top_k] or flat
            const ITensor *hidden = nullptr;          ///< Optional hidden state for future sparse row transfer
            /** Captured fixed-capacity source; mutually exclusive with direct host tensor reads. */
            std::shared_ptr<MoEOverlayDispatchTicketStorage> ticket_storage;
            std::optional<BufferId> routing_indices_buffer_id;
            std::optional<BufferId> routing_weights_buffer_id;
            std::optional<BufferId> hidden_buffer_id;

            int seq_len = 0;
            int top_k = 0;
            int d_model = 0;
            std::string continuation_domain; ///< Empty means transfer metadata is informational only
            MoEExpertTransferMode transfer_mode = MoEExpertTransferMode::Auto;

            std::optional<RoutedExpertLayerPlacement> placement;
            std::vector<RoutedExpertTier> routed_tiers;

            /**
             * Graph-frozen physical owner map for ordinary routed experts.
             *
             * Heterogeneous dispatch must put each route into exactly one
             * participant subpacket before bytes cross a rank boundary.  A
             * live residency authority supersedes this initial map with the
             * owner map carried by the ticket's immutable epoch lease.
             */
            std::shared_ptr<const MoEExpertOwnerMap> owner_map;

            /**
             * Optional live residency authority. When present, `placement`
             * identifies the graph-frozen layer/topology while routing reads
             * the exact immutable snapshot retained by the output lease.
             */
            std::shared_ptr<MoEOverlayResidencyAuthority>
                residency_authority;
            /**
             * Optional active CPU LLEP child whose parent lease supplies both
             * the immutable placement and per-route participant publication.
             */
            std::shared_ptr<CPUCurrentBatchLLEPTransactionState>
                cpu_current_batch_llep_state;
            /**
             * Optional ticket-bound decode or real-prefill history owner.
             *
             * The stage validates that the source is a production phase and
             * publishes only `logical_seq_len`, never padded bucket rows.
             */
            std::optional<RoutingEvidencePublication>
                routing_evidence_publication;
            MoEExpertDispatchOutput *output = nullptr;
            std::shared_ptr<MoEExpertDispatchOutput> output_lifetime;
        };

        static_assert(StageParamsRequired<Params>);

        explicit MoEExpertDispatchStage(Params params);

        bool execute(IDeviceContext *ctx) override;
        ComputeStageType type() const override { return ComputeStageType::MOE_EXPERT_DISPATCH; }
        std::string name() const override { return "moe_expert_dispatch"; }
        bool supportsBackend(ComputeBackendType backend) const override;
        bool isGraphCapturable() const override { return false; }
        /** @return true because heterogeneous ExpertOverlay executes this host descriptor between captured device regions. */
        bool isManualGraphBoundary() const override { return true; }
        /**
         * @brief Admit overlap only when a captured GPU ticket owns every input.
         *
         * The stage's first operation acquire-waits on the publisher's mapped
         * timeline. Direct tensor bindings instead require the preceding GPU
         * executable to have completed before host execution begins.
         */
        ManualGraphBoundaryScheduling
        manualGraphBoundaryScheduling() const noexcept override
        {
            return params_.ticket_storage
                       ? ManualGraphBoundaryScheduling::ConcurrentTicketService
                       : ManualGraphBoundaryScheduling::BetweenExecutableLaunches;
        }
        bool supportsPaddedPrefillGraphCapturePreflight() const override
        {
            return params_.ticket_storage != nullptr;
        }
        bool supportsPaddedPrefillRealLengthContract() const override
        {
            return params_.ticket_storage != nullptr;
        }
        /**
         * @brief Return whether live residency must receive sequence identity.
         *
         * Static graph-frozen placement has no independently advancing bank.
         * Dynamic and LLEP dispatch must instead consume the placement epoch
         * pinned by the enclosing graph sequence.
         */
        bool hasMoEOverlayCollectiveRuntimeParams() const override
        {
            return params_.residency_authority != nullptr ||
                   params_.cpu_current_batch_llep_state != nullptr;
        }
        /**
         * @brief Store the typed transaction and exact placement epoch for the next execution.
         * @param params Sequence identity published by the orchestration owner.
         */
        void updateMoEOverlayCollectiveRuntimeParams(
            const MoEOverlayCollectiveRuntimeParams &params) override;
        StageBufferRequirements getBufferRequirements() const override;
        StageBufferContract bufferContract() const override;
        StageDumpInfo buildDumpInfoImpl() const override;

        const Params &params() const { return params_; }

    private:
        /**
         * @brief Merge the already validated ticket route prefix exactly once.
         * @param logical_seq_len Number of real rows in the fixed-capacity ticket.
         * @return Whether the configured histogram accepted the complete slice.
         */
        bool publishRoutingEvidence(int logical_seq_len) const;

        Params params_;
        /** Sequence-owned identity stamped immediately before graph execution. */
        MoEOverlayCollectiveRuntimeParams runtime_params_{};
        /** Persistent integer view of FP32 ticket expert ids; no execute allocation. */
        mutable std::vector<int> routing_evidence_expert_ids_;
        /**
         * @brief Invocation-exclusive histogram accumulator retained at setup.
         *
         * Dispatch stages execute independently, so each stage owns one dense
         * expert-count table rather than sharing mutable scratch through the
         * process-wide histogram. The merge clears and reuses this storage.
         */
        mutable std::vector<std::uint64_t>
            routing_evidence_count_scratch_;
    };

} // namespace llaminar2
