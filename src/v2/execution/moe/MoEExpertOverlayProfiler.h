/**
 * @file MoEExpertOverlayProfiler.h
 * @brief Diagnostic-only aggregation for production MoE expert-overlay evidence.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{
    struct RoutedExpertDomain;
    struct RoutedExpertLayerPlacement;
    struct RoutedExpertTier;
    struct MoEExpertDispatchOutput;

    /**
     * @brief One aggregate-safe observation from the graph-native overlay path.
     *
     * `participant_id` is deliberately part of the row identity.  Several
     * logical expert owners may use the same physical CPU backend, so grouping
     * only by device would hide a dropped NUMA participant behind another
     * participant's successful work.
     */
    struct MoEExpertOverlayProfileRow
    {
        std::string phase = "unknown";
        int layer = -1;
        int tier_index = -1;
        /// Stable owner-map participant, or -1 for a domain-level observation.
        int participant_id = -1;
        std::string domain = "unknown";
        std::string domain_kind = "unknown";
        std::string backend = "unknown";
        int assigned_experts = 0;
        int resident_experts = 0;
        size_t routed_entries = 0;
        /// Number of locally mask-eligible sparse routes selected for compute.
        size_t active_routes = 0;
        size_t selected_rows = 0;
        size_t transfer_bytes = 0;
        size_t outbound_bytes = 0;
        size_t return_bytes = 0;
        double compute_ms = 0.0;
        double domain_reduce_ms = 0.0;
        double cross_domain_reduce_ms = 0.0;
        int participant_count = 0;
        std::string executed_experts = "unknown";
        std::string transport_mode = "unknown";
        std::string final_reduce_mode = "unknown";
        std::string accumulation_path = "unknown";
        // Graph-native phase extensions (Phase 14)
        size_t inbound_rows = 0; ///< Rows received by the current participant/root
        size_t compact_dispatch_bytes = 0;
        size_t compact_return_bytes = 0;
        size_t dense_bytes_avoided = 0;   ///< Dense bytes minus compact bytes saved
        size_t cpu_fallback_rows = 0;     ///< Rows handled by CPU expert participants
        size_t gpu_cached_rows = 0;       ///< Rows handled by GPU expert participants/cache tiers
        double scatter_ms = 0.0;          ///< Scatter-add time (gn_return_reduce)
        double import_broadcast_ms = 0.0; ///< TP import/broadcast after scatter
    };

    /**
     * @brief Collects CSV and PerfStats evidence for graph-native MoE overlay execution.
     *
     * The profiler is diagnostic-only: it observes already selected sparse
     * work and never changes routing, ownership, collective ordering, or
     * tensor residency.  Its identity fields preserve the distinction between
     * logical participants sharing a backend so campaigns can prove every
     * assigned owner actually executed.
     */
    class MoEExpertOverlayProfiler
    {
    public:
        /** @brief Return whether overlay profiling or PerfStats collection is active. */
        static bool isEnabled();
        /** @brief Return whether a human-readable overlay summary should be printed. */
        static bool shouldPrintSummary();

        /** @brief Discard all process-local observations before a new inference transaction. */
        static void reset();
        /** @brief Merge one observation into the process-local aggregate and PerfStats stream. */
        static void recordRow(MoEExpertOverlayProfileRow row);
        /** @brief Return a stable copy of every process-local aggregate row. */
        static std::vector<MoEExpertOverlayProfileRow> rows();
        /** @brief Render the current aggregate rows as a terminal-readable table. */
        static std::string renderSummary();
        /** @brief Serialize the current aggregate rows using the campaign CSV schema. */
        static std::string csvString();
        /** @brief Return the configured destination for the diagnostic CSV artifact. */
        static std::string csvPath();
        /** @brief Write the current aggregate rows to the configured or supplied CSV path. */
        static bool writeCsv(const std::string &path = {});
        /** @brief Emit enabled summary and CSV artifacts without changing inference state. */
        static void flush();

        /** @brief Record a legacy domain-level dispatch observation. */
        static void recordDispatch(
            int layer,
            const MoEExpertDispatchOutput &output,
            const RoutedExpertLayerPlacement &placement,
            const std::vector<RoutedExpertTier> &routed_tiers);

        /**
         * @brief Record compact sparse-route delivery for one graph-native edge.
         *
         * @param layer Transformer layer.
         * @param tier_index Routed tier.
         * @param domain_key Stable collective-edge descriptor.
         * @param source_participant Logical sender.
         * @param target_participant Logical receiver.
         * @param outbound_rows Sparse rows sent on the edge.
         * @param outbound_entries Sparse route entries sent on the edge.
         * @param inbound_rows Rows observed after transport.
         * @param compact_dispatch_bytes Actual compact payload size.
         * @param dense_dispatch_bytes Equivalent dense payload size.
         * @param wait_ms Transport wait time outside the compute stream.
         */
        static void recordGraphNativeSparseDispatch(
            int layer,
            int tier_index,
            const std::string &domain_key,
            int source_participant,
            int target_participant,
            size_t outbound_rows,
            size_t outbound_entries,
            size_t inbound_rows,
            size_t compact_dispatch_bytes,
            size_t dense_dispatch_bytes,
            double wait_ms);

        /**
         * @brief Record one participant-local sparse expert invocation.
         *
         * This API is called even when `active_routes` is zero.  A zero-work
         * row distinguishes a genuinely idle participant from a stage that
         * was absent from the graph, and makes an unexpected route drop
         * visible in the same CSV/PerfStats evidence as successful work.
         *
         * @param layer Transformer layer.
         * @param tier_index Routed tier.
         * @param participant_id Stable owner-map participant identity.
         * @param device_key Executing backend/device descriptor.
         * @param is_cpu Whether the participant is a CPU fallback endpoint.
         * @param inbound_rows Sparse rows delivered before local mask filtering.
         * @param active_routes Mask-eligible route entries sent to local compute.
         * @param output_rows Distinct token rows emitted after aggregation.
         * @param unique_expert_ids Mask-eligible expert ids executed locally.
         * @param compute_ms Measured local compute duration.
         */
        static void recordGraphNativeLocalExpert(
            int layer,
            int tier_index,
            int participant_id,
            const std::string &device_key,
            bool is_cpu,
            size_t inbound_rows,
            size_t active_routes,
            size_t output_rows,
            std::vector<int> unique_expert_ids,
            double compute_ms);

        /**
         * @brief Record compact result return and root-side deterministic scatter/reduction.
         *
         * @param layer Transformer layer.
         * @param tier_index Routed tier.
         * @param domain_key Stable collective-edge descriptor.
         * @param source_participant Logical sender of computed rows.
         * @param target_participant Logical return owner.
         * @param outbound_rows Rows returned from the expert participant.
         * @param inbound_rows Rows delivered to the return owner.
         * @param compact_return_bytes Actual compact return payload size.
         * @param dense_return_bytes Equivalent dense payload size.
         * @param return_wait_ms Transport wait time outside the compute stream.
         * @param scatter_ms Deterministic root-side scatter-add duration.
         * @param import_broadcast_ms Optional TP import/broadcast duration.
         */
        static void recordGraphNativeReturnReduce(
            int layer,
            int tier_index,
            const std::string &domain_key,
            int source_participant,
            int target_participant,
            size_t outbound_rows,
            size_t inbound_rows,
            size_t compact_return_bytes,
            size_t dense_return_bytes,
            double return_wait_ms,
            double scatter_ms,
            double import_broadcast_ms);
    };

} // namespace llaminar2
