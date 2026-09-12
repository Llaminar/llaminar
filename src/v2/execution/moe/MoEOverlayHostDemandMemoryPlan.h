/**
 * @file MoEOverlayHostDemandMemoryPlan.h
 * @brief Exact host routing-evidence BOM shared by setup and publication.
 *
 * This value describes storage, not a live allocator or another demand policy.
 * PhysicalMemoryAuthority alone admits and carries the allocations. Mutable RCU
 * banks, immutable observations and the existing MPI mailbox use the same route
 * geometry; none can invent a private window or graph-row capacity at runtime.
 */
#pragma once

#include "DecodeExpertHistogram.h"

#include <cstddef>

namespace llaminar2
{
    /** @brief Whether the existing proposal publication owns an MPI mailbox. */
    enum class MoEOverlayDemandPublicationScope
    {
        ProcessLocal,
        Distributed,
    };

    /** @brief Model and retained execution geometry, resolved before tier filling. */
    struct MoEOverlayHostDemandGeometry
    {
        int num_layers = 0; ///< Routed main/sidecar layers, not all GGUF blocks.
        int num_experts = 0;
        int top_k = 0;
        int initial_window_rows = 0;
        int maximum_window_rows = 0; ///< Zero means adaptation is disabled.
        int maximum_invocation_rows = 0; ///< Whole flattened batch, never its padding.
        MoEOverlayDemandPublicationScope publication =
            MoEOverlayDemandPublicationScope::ProcessLocal;
    };

    /**
     * @brief Immutable BOM for the host authority's existing evidence lifecycle.
     *
     * Two mutable banks support inference/rotation. Immutable storage covers
     * three concurrent owners: the forecast's current observation, a newly
     * frozen/received observation, and a previously published wave still retiring.
     * They often share one payload; admission covers their maximum disjoint case.
     * Publication reuses one send/receive mailbox, not one buffer per peer.
     * Copies retained outside that production lifecycle require their own PMA
     * admission; diagnostics cannot retain unbounded observations for free.
     */
    class MoEOverlayHostDemandMemoryPlan final
    {
    public:
        /**
         * @brief Resolve checked payload bounds without allocating or querying devices.
         * @param geometry Actual model, adaptive-window and retained invocation bounds.
         * @throws std::invalid_argument For missing or contradictory geometry.
         * @throws std::overflow_error For a byte or MPI-count ABI overflow.
         */
        explicit MoEOverlayHostDemandMemoryPlan(MoEOverlayHostDemandGeometry geometry);

        /**
         * @brief Bind runtime ingress to this admitted geometry and the sole live ledger.
         * @param histogram Model/window facts resolved for the live histogram.
         * @param memory Canonical rank-local PMA; no private allocation authority.
         * @return Exact transaction storage contract, without allocating payloads.
         * @throws std::invalid_argument For a missing ledger or stale model/window identity.
         */
        [[nodiscard]] ExpertHistogramTransactionConfig bind(
            const DecodeExpertHistogramConfig &histogram,
            std::shared_ptr<PhysicalMemoryAuthority> memory) const;

        /** @return Single immutable source of route-bank geometry. */
        [[nodiscard]] const moe_overlay_economy::TransactionDemandCapacity &capacity() const noexcept;
        /** @return Resolved model and publication facts, for setup identity checks. */
        [[nodiscard]] const MoEOverlayHostDemandGeometry &geometry() const noexcept;
        /** @return Bytes of both persistent mutable route banks. */
        [[nodiscard]] std::size_t mutableBankBytes() const noexcept;
        /** @return Maximum bytes for the three overlapping immutable observations. */
        [[nodiscard]] std::size_t observationBytes() const noexcept;
        /** @return One bounded proposal mailbox, or zero for process-local publication. */
        [[nodiscard]] std::size_t mailboxBytes() const noexcept;
        /** @return ExecutionWorkspace BOM contribution; never a live available-byte ledger. */
        [[nodiscard]] std::size_t allocationBytes() const noexcept;

    private:
        MoEOverlayHostDemandGeometry geometry_;
        moe_overlay_economy::TransactionDemandCapacity capacity_;
        std::size_t mutable_bank_bytes_ = 0;
        std::size_t observation_bytes_ = 0;
        std::size_t mailbox_bytes_ = 0;
        std::size_t total_bytes_ = 0;
    };
}
