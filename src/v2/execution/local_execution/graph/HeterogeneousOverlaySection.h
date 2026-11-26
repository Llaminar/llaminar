/**
 * @file HeterogeneousOverlaySection.h
 * @brief Declarative contract and structural invariant verifier for heterogeneous MoE ticket sections.
 *
 * A heterogeneous MoE layer (e.g. GPU+CPU TieredOverlay) executes five distinct
 * phases across captured GPU work and concurrent manual CPU workers:
 *   1. Router / early trigger (`moe_routing` -> `moe_overlay_ticket_publish` /
 *      `moe_overlay_activation_dispatch_pack`), which wakes CPU workers or
 *      remote followers as early as possible.
 *   2. Pre-CPU captured GPU overlap branches (local routed experts, optional
 *      independent shared-expert FFN) running concurrently with CPU service.
 *   3. Symmetric pre-CPU ticket cutpoint (`BeforeManualBoundary`) closing the
 *      producer captured unit identically across authority and follower GPUs.
 *   4. Manual host service (`moe_expert_dispatch` -> CPU experts -> return)
 *      owned exclusively by the authority envelope.
 *   5. Post-CPU captured join (`ticket_consume` -> `ordered_reduce` ->
 *      shared-expert gate/combine -> broadcast).
 *
 * Sealing this section through @ref sealHeterogeneousOverlayTicketSection makes
 * invalid wirings — such as delaying the early CPU trigger behind a slow GPU
 * overlap branch, attaching manual work to a follower graph, or inverting the
 * post-CPU join ahead of the cutpoint — impossible at construction time.
 */

#pragma once

#include "GraphComposer.h"
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Declarative topology description of one heterogeneous MoE ticket section.
     */
    struct HeterogeneousOverlayTicketSection
    {
        GraphNodeRef router;                      ///< Participant-local GPU router producer.
        std::optional<GraphNodeRef> early_trigger; ///< Authority-only GPU early ticket/packet publisher.
        GraphNodeRef pre_cpu_gpu_cutpoint;        ///< Captured GPU terminal closing the pre-CPU unit.
        std::string ticket_unit_identity;         ///< Symmetric cross-participant ticket-unit identity.
        std::vector<GraphNodeRef> pre_cpu_overlap_branches; ///< Independent GPU work overlapping CPU inside the pre-CPU unit.
        std::optional<GraphNodeRef> manual_host_entry;      ///< Authority-only manual CPU dispatch entry.
        std::optional<GraphNodeRef> post_cpu_join;          ///< First post-CPU captured join node (e.g., ordered_reduce).
        std::vector<GraphNodeRef> rank_batch_dispatch_entries; ///< Optional additional authority host rank-batch dispatches.
    };

    /**
     * @brief Validate all structural invariants of @p section without mutating @p graph.
     * @param graph Participant-local ComputeGraph in a heterogeneous ticket envelope.
     * @param section Complete declarative section description.
     * @throws std::invalid_argument On missing/foreign handles, wrong device, or envelope mismatch.
     * @throws std::logic_error On ordering inversions that would delay early CPU wake-up or
     *         deadlock authority/follower capture rendezvous.
     */
    void validateHeterogeneousOverlayTicketSection(
        const ComputeGraph &graph,
        const HeterogeneousOverlayTicketSection &section);

    /**
     * @brief Validate and seal @p section onto @p graph with explicit typed dependency roles.
     * @param graph Participant-local ComputeGraph in a heterogeneous ticket envelope.
     * @param section Complete declarative section description.
     * @details Atomically:
     *   - Verifies envelope role parity (authority owns manual host work; follower owns none).
     *   - Verifies no GPU overlap branch precedes @ref HeterogeneousOverlayTicketSection::early_trigger.
     *   - Orders each overlap branch after @p early_trigger (when present) and before
     *     @p pre_cpu_gpu_cutpoint via @ref GraphDependencyKind::ScratchOrder.
     *   - Attaches the `BeforeManualBoundary` ticket-unit contract to @p pre_cpu_gpu_cutpoint.
     *   - Orders @p manual_host_entry (and any @p rank_batch_dispatch_entries) after
     *     @p pre_cpu_gpu_cutpoint via @ref GraphDependencyKind::HostSubmissionOrder.
     *   - Verifies @p post_cpu_join strictly follows @p pre_cpu_gpu_cutpoint and never feeds
     *     backward into @p early_trigger, @p pre_cpu_gpu_cutpoint, or @p manual_host_entry.
     */
    void sealHeterogeneousOverlayTicketSection(
        ComputeGraph &graph,
        const HeterogeneousOverlayTicketSection &section);

    /**
     * @brief Declaratively attach one independent GPU overlap branch into an already-sealed pre-CPU ticket unit.
     * @param graph Participant-local ComputeGraph in a heterogeneous ticket envelope.
     * @param branch Independent GPU branch node (e.g., shared_expert_ffn) on the same GPU.
     * @param early_trigger Optional early CPU packet/ticket publisher that must fire BEFORE @p branch.
     * @param pre_cpu_cutpoint Existing `BeforeManualBoundary` cutpoint closing the pre-CPU unit.
     * @throws std::invalid_argument On foreign handles, wrong device, or missing cutpoint contract.
     * @throws std::logic_error If @p branch is an ancestor of @p early_trigger or if attaching
     *         @p branch would introduce a cycle or delay an early CPU trigger.
     */
    void attachPreCPUOverlapBranch(
        ComputeGraph &graph,
        const GraphNodeRef &branch,
        const std::optional<GraphNodeRef> &early_trigger,
        const GraphNodeRef &pre_cpu_cutpoint);

} // namespace llaminar2
