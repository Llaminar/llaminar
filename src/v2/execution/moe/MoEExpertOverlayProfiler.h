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
    struct MoECPUFallbackTensorParallelStats;
    struct MoECPUFallbackTransferStats;
    struct MoEExpertDispatchOutput;
    struct MoEExpertOverlayLocalTPDiagnostics;
    struct MoEExpertParallelReduceDiagnostics;
    struct MoEOverlayRuntimeDomain;

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

        static void recordLocalTP(
            int layer,
            const MoEOverlayRuntimeDomain &domain,
            int resident_experts,
            const MoEExpertOverlayLocalTPDiagnostics &diagnostics);

        static void recordCPUFallback(
            int layer,
            const ExpertComputeDomain &domain,
            int resident_experts,
            size_t routed_entries,
            size_t selected_rows,
            const MoECPUFallbackTransferStats *transfer_stats,
            const MoECPUFallbackTensorParallelStats *tensor_parallel_stats,
            double compute_ms);

        static void recordFinalReduce(
            int layer,
            const MoEExpertParallelReduceDiagnostics &diagnostics);
    };

} // namespace llaminar2