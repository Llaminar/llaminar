/**
 * @file MoEExpertOverlayProfiler.h
 * @brief Lightweight Phase 9A profiling aggregation for MoE expert overlays.
 */

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace llaminar2
{
    struct ExpertComputeDomain;
    struct ExpertLayerPlacement;
    struct ExpertRoutedTier;
    struct MoEExpertDispatchOutput;
    struct MoEExpertParallelReduceDiagnostics;

    struct MoEExpertOverlayProfileRow
    {
        std::string phase = "unknown";
        int layer = -1;
        std::string domain = "unknown";
        std::string domain_kind = "unknown";
        std::string backend = "unknown";
        int assigned_experts = 0;
        int resident_experts = 0;
        size_t routed_entries = 0;
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

    class MoEExpertOverlayProfiler
    {
    public:
        static bool isEnabled();
        static bool shouldPrintSummary();

        static void reset();
        static void recordRow(MoEExpertOverlayProfileRow row);
        static std::vector<MoEExpertOverlayProfileRow> rows();
        static std::string renderSummary();
        static std::string csvString();
        static std::string csvPath();
        static bool writeCsv(const std::string &path = {});
        static void flush();

        static void recordDispatch(
            int layer,
            const MoEExpertDispatchOutput &output,
            const ExpertLayerPlacement &placement,
            const std::vector<ExpertRoutedTier> &routed_tiers);

        static void recordFinalReduce(
            int layer,
            const MoEExpertParallelReduceDiagnostics &diagnostics);

        // Graph-native stage profiling (Phase 14)
        static void recordGraphNativeSparseDispatch(
            int layer,
            const std::string &domain_key,
            int source_participant,
            int target_participant,
            size_t outbound_rows,
            size_t outbound_entries,
            size_t inbound_rows,
            size_t compact_dispatch_bytes,
            size_t dense_dispatch_bytes,
            double wait_ms);

        static void recordGraphNativeLocalExpert(
            int layer,
            const std::string &device_key,
            bool is_cpu,
            size_t input_rows,
            size_t output_rows,
            std::vector<int> unique_expert_ids,
            double compute_ms);

        static void recordGraphNativeReturnReduce(
            int layer,
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