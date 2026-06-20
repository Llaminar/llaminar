/**
 * @file MoEExpertOverlayProfiler.cpp
 * @brief Lightweight Phase 9A profiling aggregation for MoE expert overlays.
 */

#include "MoEExpertOverlayProfiler.h"

#include "MoEExpertOverlayRuntimePlan.h"
#include "execution/compute_stages/stages/MoEExpertDispatchStage.h"
#include "execution/compute_stages/stages/MoEExpertParallelReduceStage.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

#include "fort.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

namespace llaminar2
{
    namespace
    {
        struct ProfilerState
        {
            std::mutex mutex;
            std::vector<MoEExpertOverlayProfileRow> rows;
            size_t version = 0;
            size_t printed_version = 0;
            size_t csv_version = 0;
        };

        ProfilerState &state()
        {
            static ProfilerState instance;
            return instance;
        }

        bool sameKey(const MoEExpertOverlayProfileRow &lhs, const MoEExpertOverlayProfileRow &rhs)
        {
            return lhs.phase == rhs.phase && lhs.layer == rhs.layer &&
                   lhs.tier_index == rhs.tier_index && lhs.domain == rhs.domain;
        }

        void mergeTextField(std::string &target, const std::string &value)
        {
            if (value.empty() || value == "unknown")
                return;
            if (target.empty() || target == "unknown")
            {
                target = value;
                return;
            }
            if (target.find(value) == std::string::npos)
                target += "+" + value;
        }

        void mergeRow(MoEExpertOverlayProfileRow &target, const MoEExpertOverlayProfileRow &row)
        {
            mergeTextField(target.domain_kind, row.domain_kind);
            mergeTextField(target.backend, row.backend);
            target.assigned_experts = std::max(target.assigned_experts, row.assigned_experts);
            target.resident_experts = std::max(target.resident_experts, row.resident_experts);
            target.routed_entries += row.routed_entries;
            target.selected_rows += row.selected_rows;
            target.transfer_bytes += row.transfer_bytes;
            target.outbound_bytes += row.outbound_bytes;
            target.return_bytes += row.return_bytes;
            target.compute_ms += row.compute_ms;
            target.domain_reduce_ms += row.domain_reduce_ms;
            target.cross_domain_reduce_ms += row.cross_domain_reduce_ms;
            target.participant_count = std::max(target.participant_count, row.participant_count);
            mergeTextField(target.executed_experts, row.executed_experts);
            mergeTextField(target.transport_mode, row.transport_mode);
            mergeTextField(target.final_reduce_mode, row.final_reduce_mode);
            mergeTextField(target.accumulation_path, row.accumulation_path);
            target.inbound_rows += row.inbound_rows;
            target.compact_dispatch_bytes += row.compact_dispatch_bytes;
            target.compact_return_bytes += row.compact_return_bytes;
            target.dense_bytes_avoided += row.dense_bytes_avoided;
            target.cpu_fallback_rows += row.cpu_fallback_rows;
            target.gpu_cached_rows += row.gpu_cached_rows;
            target.scatter_ms += row.scatter_ms;
            target.import_broadcast_ms += row.import_broadcast_ms;
        }

        std::string csvEscape(const std::string &value)
        {
            if (value.find_first_of(",\"\n\r") == std::string::npos)
                return value;
            std::string escaped = "\"";
            for (char ch : value)
            {
                if (ch == '"')
                    escaped += "\"\"";
                else
                    escaped += ch;
            }
            escaped += "\"";
            return escaped;
        }

        std::string formatDouble(double value)
        {
            std::ostringstream out;
            out.setf(std::ios::fixed, std::ios::floatfield);
            out.precision(3);
            out << value;
            return out.str();
        }

        std::string joinInts(std::vector<int> values)
        {
            if (values.empty())
                return "unknown";
            std::sort(values.begin(), values.end());
            values.erase(std::unique(values.begin(), values.end()), values.end());
            std::ostringstream out;
            for (size_t i = 0; i < values.size(); ++i)
            {
                if (i > 0)
                    out << ";";
                out << values[i];
            }
            return out.str();
        }

        int countAssignedExperts(const ExpertLayerPlacement &placement, int tier_index)
        {
            return static_cast<int>(std::count(
                placement.routed_expert_tier.begin(),
                placement.routed_expert_tier.end(),
                tier_index));
        }

        std::string finalReduceTransportMode(const MoEExpertParallelReduceDiagnostics &diagnostics)
        {
            if (diagnostics.host_staged)
                return "host-staged";
            if (diagnostics.output_resident_on_continuation)
                return "continuation-device";
            return "direct";
        }

        std::string accumulationPathSummary(const MoEExpertParallelReduceDiagnostics &diagnostics)
        {
            std::set<std::string> paths;
            for (const auto &partial : diagnostics.partials)
                paths.insert(toString(partial.accumulation_path));
            if (paths.empty())
                return diagnostics.host_staged ? "HostSummedCorrectnessFallback" : "unknown";
            std::ostringstream out;
            bool first = true;
            for (const auto &path : paths)
            {
                if (!first)
                    out << ";";
                out << path;
                first = false;
            }
            return out.str();
        }

        uint64_t msToNs(double ms)
        {
            if (ms <= 0.0)
                return 0;
            return static_cast<uint64_t>(ms * 1.0e6);
        }

        PerfStatsCollector::Tags unifiedTagsForRow(const MoEExpertOverlayProfileRow &row)
        {
            PerfStatsCollector::Tags tags{
                {"layer", std::to_string(row.layer)},
                {"tier", std::to_string(row.tier_index)},
                {"domain", row.domain},
                {"domain_kind", row.domain_kind},
                {"backend", row.backend},
                {"transport", row.transport_mode},
                {"accumulation", row.accumulation_path},
            };
            if (!row.final_reduce_mode.empty() && row.final_reduce_mode != "unknown")
                tags.emplace("final_reduce", row.final_reduce_mode);
            if (!row.executed_experts.empty() && row.executed_experts != "unknown")
                tags.emplace("experts", row.executed_experts);
            return tags;
        }

        void addUnifiedCounterIfNonZero(
            const MoEExpertOverlayProfileRow &row,
            const PerfStatsCollector::Tags &tags,
            const char *name,
            double value)
        {
            if (value == 0.0)
                return;
            PerfStatsCollector::addCounter(
                "moe_overlay",
                name,
                value,
                row.phase,
                row.domain,
                tags);
        }

        void addUnifiedTimerIfNonZero(
            const MoEExpertOverlayProfileRow &row,
            const PerfStatsCollector::Tags &tags,
            const char *name,
            double ms)
        {
            const uint64_t ns = msToNs(ms);
            if (ns == 0)
                return;
            PerfStatsCollector::recordTimingNs(
                "moe_overlay",
                name,
                ns,
                row.phase,
                row.domain,
                tags);
        }

        void recordUnified(const MoEExpertOverlayProfileRow &row)
        {
            if (!PerfStatsCollector::isEnabled())
                return;

            const auto tags = unifiedTagsForRow(row);
            addUnifiedCounterIfNonZero(row, tags, "assigned_experts", row.assigned_experts);
            addUnifiedCounterIfNonZero(row, tags, "resident_experts", row.resident_experts);
            addUnifiedCounterIfNonZero(row, tags, "routed_entries", static_cast<double>(row.routed_entries));
            addUnifiedCounterIfNonZero(row, tags, "selected_rows", static_cast<double>(row.selected_rows));
            addUnifiedCounterIfNonZero(row, tags, "transfer_bytes", static_cast<double>(row.transfer_bytes));
            addUnifiedCounterIfNonZero(row, tags, "outbound_bytes", static_cast<double>(row.outbound_bytes));
            addUnifiedCounterIfNonZero(row, tags, "return_bytes", static_cast<double>(row.return_bytes));
            addUnifiedCounterIfNonZero(row, tags, "participant_count", row.participant_count);
            addUnifiedCounterIfNonZero(row, tags, "inbound_rows", static_cast<double>(row.inbound_rows));
            addUnifiedCounterIfNonZero(row, tags, "compact_dispatch_bytes", static_cast<double>(row.compact_dispatch_bytes));
            addUnifiedCounterIfNonZero(row, tags, "compact_return_bytes", static_cast<double>(row.compact_return_bytes));
            addUnifiedCounterIfNonZero(row, tags, "dense_bytes_avoided", static_cast<double>(row.dense_bytes_avoided));
            addUnifiedCounterIfNonZero(row, tags, "cpu_rows", static_cast<double>(row.cpu_fallback_rows));
            addUnifiedCounterIfNonZero(row, tags, "gpu_rows", static_cast<double>(row.gpu_cached_rows));

            addUnifiedTimerIfNonZero(row, tags, "compute", row.compute_ms);
            addUnifiedTimerIfNonZero(row, tags, "domain_reduce", row.domain_reduce_ms);
            addUnifiedTimerIfNonZero(row, tags, "cross_domain_reduce", row.cross_domain_reduce_ms);
            addUnifiedTimerIfNonZero(row, tags, "scatter", row.scatter_ms);
            addUnifiedTimerIfNonZero(row, tags, "import_broadcast", row.import_broadcast_ms);
        }
    } // namespace

    bool MoEExpertOverlayProfiler::isEnabled()
    {
        const auto &env = debugEnv();
        return env.profile.enabled ||
               env.moe_expert_overlay.trace ||
               env.moe_expert_overlay.profile_csv_enabled ||
               PerfStatsCollector::isEnabled();
    }

    bool MoEExpertOverlayProfiler::shouldPrintSummary()
    {
        const auto &env = debugEnv();
        return env.profile.enabled || env.moe_expert_overlay.trace;
    }

    void MoEExpertOverlayProfiler::reset()
    {
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        s.rows.clear();
        ++s.version;
        s.printed_version = 0;
        s.csv_version = 0;
    }

    void MoEExpertOverlayProfiler::recordRow(MoEExpertOverlayProfileRow row)
    {
        if (row.phase.empty())
            row.phase = "unknown";
        if (row.domain.empty())
            row.domain = "unknown";

        recordUnified(row);

        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        auto existing = std::find_if(s.rows.begin(), s.rows.end(), [&](const auto &candidate)
                                     { return sameKey(candidate, row); });
        if (existing != s.rows.end())
            mergeRow(*existing, row);
        else
            s.rows.push_back(std::move(row));
        ++s.version;
    }

    std::vector<MoEExpertOverlayProfileRow> MoEExpertOverlayProfiler::rows()
    {
        auto &s = state();
        std::lock_guard<std::mutex> lock(s.mutex);
        return s.rows;
    }

    std::string MoEExpertOverlayProfiler::renderSummary()
    {
        const auto snapshot = rows();
        if (snapshot.empty())
            return {};

        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);
        table << fort::header
              << "Phase" << "Layer" << "Tier" << "Domain" << "Kind" << "Backend"
              << "Assigned" << "Resident" << "Routed" << "Rows" << "Bytes"
              << "Compute ms" << "Local reduce ms" << "Final reduce ms"
              << "Participants" << "Transport" << "Accumulation"
              << "In rows" << "CPU rows" << "GPU rows" << "Dense saved"
              << "Dispatch B" << "Return B" << "Scatter ms" << "Import ms"
              << fort::endr;

        for (const auto &row : snapshot)
        {
            table << row.phase
                  << row.layer
                  << row.tier_index
                  << row.domain
                  << row.domain_kind
                  << row.backend
                  << row.assigned_experts
                  << row.resident_experts
                  << row.routed_entries
                  << row.selected_rows
                  << row.transfer_bytes
                  << formatDouble(row.compute_ms)
                  << formatDouble(row.domain_reduce_ms)
                  << formatDouble(row.cross_domain_reduce_ms)
                  << row.participant_count
                  << row.transport_mode
                  << row.accumulation_path
                  << row.inbound_rows
                  << row.cpu_fallback_rows
                  << row.gpu_cached_rows
                  << row.dense_bytes_avoided
                  << row.compact_dispatch_bytes
                  << row.compact_return_bytes
                  << formatDouble(row.scatter_ms)
                  << formatDouble(row.import_broadcast_ms)
                  << fort::endr;
        }

        std::ostringstream out;
        out << "\nMOE EXPERT OVERLAY PROFILING SUMMARY\n"
            << table.to_string();
        return out.str();
    }

    std::string MoEExpertOverlayProfiler::csvString()
    {
        const auto snapshot = rows();
        std::ostringstream out;
        out << "phase,layer,tier_index,domain,domain_kind,backend,assigned_experts,resident_experts,"
            << "routed_entries,selected_rows,transfer_bytes,outbound_bytes,return_bytes,"
            << "compute_ms,domain_reduce_ms,cross_domain_reduce_ms,participant_count,"
            << "executed_experts,transport_mode,final_reduce_mode,accumulation_path,"
            << "inbound_rows,compact_dispatch_bytes,compact_return_bytes,dense_bytes_avoided,"
            << "cpu_fallback_rows,gpu_cached_rows,scatter_ms,import_broadcast_ms\n";

        for (const auto &row : snapshot)
        {
            out << csvEscape(row.phase) << ','
                << row.layer << ','
                << row.tier_index << ','
                << csvEscape(row.domain) << ','
                << csvEscape(row.domain_kind) << ','
                << csvEscape(row.backend) << ','
                << row.assigned_experts << ','
                << row.resident_experts << ','
                << row.routed_entries << ','
                << row.selected_rows << ','
                << row.transfer_bytes << ','
                << row.outbound_bytes << ','
                << row.return_bytes << ','
                << formatDouble(row.compute_ms) << ','
                << formatDouble(row.domain_reduce_ms) << ','
                << formatDouble(row.cross_domain_reduce_ms) << ','
                << row.participant_count << ','
                << csvEscape(row.executed_experts) << ','
                << csvEscape(row.transport_mode) << ','
                << csvEscape(row.final_reduce_mode) << ','
                << csvEscape(row.accumulation_path) << ','
                << row.inbound_rows << ','
                << row.compact_dispatch_bytes << ','
                << row.compact_return_bytes << ','
                << row.dense_bytes_avoided << ','
                << row.cpu_fallback_rows << ','
                << row.gpu_cached_rows << ','
                << formatDouble(row.scatter_ms) << ','
                << formatDouble(row.import_broadcast_ms) << '\n';
        }
        return out.str();
    }

    std::string MoEExpertOverlayProfiler::csvPath()
    {
        return debugEnv().moe_expert_overlay.profile_csv_path;
    }

    bool MoEExpertOverlayProfiler::writeCsv(const std::string &path)
    {
        const std::string resolved_path = path.empty() ? csvPath() : path;
        if (resolved_path.empty())
            return false;

        std::error_code ec;
        const std::filesystem::path fs_path(resolved_path);
        if (fs_path.has_parent_path())
            std::filesystem::create_directories(fs_path.parent_path(), ec);

        std::ofstream out(resolved_path, std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            LOG_WARN("[MoEExpertOverlayProfiler] Failed to open CSV output path '" << resolved_path << "'");
            return false;
        }
        out << csvString();
        return true;
    }

    void MoEExpertOverlayProfiler::flush()
    {
        const int rank = Logger::getInstance().getRank();
        if (rank > 0)
            return;

        auto &s = state();
        size_t version = 0;
        size_t printed_version = 0;
        size_t csv_version = 0;
        bool empty = true;
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            version = s.version;
            printed_version = s.printed_version;
            csv_version = s.csv_version;
            empty = s.rows.empty();
        }

        if (empty)
            return;

        if (shouldPrintSummary() && printed_version != version)
        {
            std::cout << renderSummary();
            std::lock_guard<std::mutex> lock(s.mutex);
            s.printed_version = s.version;
        }

        if (debugEnv().moe_expert_overlay.profile_csv_enabled && csv_version != version)
        {
            if (writeCsv())
            {
                std::lock_guard<std::mutex> lock(s.mutex);
                s.csv_version = s.version;
            }
        }
    }

    void MoEExpertOverlayProfiler::recordDispatch(
        int layer,
        const MoEExpertDispatchOutput &output,
        const ExpertLayerPlacement &placement,
        const std::vector<ExpertRoutedTier> &routed_tiers)
    {
        if (!isEnabled())
            return;

        for (const auto &tier : output.tiers)
        {
            MoEExpertOverlayProfileRow row;
            row.phase = "dispatch";
            row.layer = layer;
            row.domain = tier.domain;
            row.assigned_experts = countAssignedExperts(placement, tier.tier_index);
            if (tier.tier_index >= 0 && static_cast<size_t>(tier.tier_index) < routed_tiers.size())
            {
                const auto &routed_tier = routed_tiers[static_cast<size_t>(tier.tier_index)];
                row.resident_experts = routed_tier.max_experts_per_layer > 0
                                           ? routed_tier.max_experts_per_layer
                                           : row.assigned_experts;
                row.transport_mode = routed_tier.fallback ? "fallback" : toString(tier.transfer_mode);
            }
            else
            {
                row.resident_experts = row.assigned_experts;
                row.transport_mode = toString(tier.transfer_mode);
            }
            row.routed_entries = tier.entries.size();
            row.selected_rows = tier.token_rows.size();
            row.transfer_bytes = tier.transfer_volume.totalBytes();
            row.outbound_bytes = tier.transfer_volume.outbound_bytes;
            row.return_bytes = tier.transfer_volume.return_bytes;
            recordRow(std::move(row));
        }
    }

    void MoEExpertOverlayProfiler::recordFinalReduce(
        int layer,
        const MoEExpertParallelReduceDiagnostics &diagnostics)
    {
        if (!isEnabled())
            return;

        MoEExpertOverlayProfileRow row;
        row.phase = "final_reduce";
        row.layer = layer;
        row.domain = diagnostics.continuation_domain.empty() ? "continuation" : diagnostics.continuation_domain;
        row.transfer_bytes = diagnostics.total_transfer_bytes;
        row.outbound_bytes = diagnostics.host_to_device_bytes;
        row.return_bytes = diagnostics.device_to_host_bytes;
        row.cross_domain_reduce_ms = diagnostics.reduce_ms;
        row.participant_count = static_cast<int>(diagnostics.partial_count);
        row.transport_mode = finalReduceTransportMode(diagnostics);
        row.final_reduce_mode = toString(diagnostics.mode);
        row.accumulation_path = accumulationPathSummary(diagnostics);
        recordRow(std::move(row));
    }

    void MoEExpertOverlayProfiler::recordGraphNativeSparseDispatch(
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
        double wait_ms)
    {
        if (!isEnabled())
            return;

        MoEExpertOverlayProfileRow row;
        row.phase = "gn_sparse_dispatch";
        row.layer = layer;
        row.tier_index = tier_index;
        row.domain = domain_key.empty() ? "unknown" : domain_key;
        row.participant_count = 2;
        row.selected_rows = outbound_rows;
        row.inbound_rows = inbound_rows;
        row.routed_entries = outbound_entries;
        row.outbound_bytes = compact_dispatch_bytes;
        row.transfer_bytes = compact_dispatch_bytes;
        row.compact_dispatch_bytes = compact_dispatch_bytes;
        row.dense_bytes_avoided = dense_dispatch_bytes > compact_dispatch_bytes
                                      ? dense_dispatch_bytes - compact_dispatch_bytes
                                      : 0;
        row.domain_reduce_ms = wait_ms;
        row.transport_mode = "compact";
        row.accumulation_path = source_participant == target_participant ? "local" : "remote";
        recordRow(std::move(row));
    }

    void MoEExpertOverlayProfiler::recordGraphNativeLocalExpert(
        int layer,
        int tier_index,
        const std::string &device_key,
        bool is_cpu,
        size_t input_rows,
        size_t output_rows,
        std::vector<int> unique_expert_ids,
        double compute_ms)
    {
        if (!isEnabled())
            return;

        MoEExpertOverlayProfileRow row;
        row.phase = "gn_local_expert";
        row.layer = layer;
        row.tier_index = tier_index;
        row.domain = device_key.empty() ? "unknown" : device_key;
        row.domain_kind = is_cpu ? "CPU" : "GPU";
        row.selected_rows = input_rows;
        row.inbound_rows = output_rows;
        row.assigned_experts = static_cast<int>(unique_expert_ids.size());
        row.resident_experts = row.assigned_experts;
        row.compute_ms = compute_ms;
        row.transport_mode = "local";
        row.cpu_fallback_rows = is_cpu ? input_rows : 0;
        row.gpu_cached_rows = is_cpu ? 0 : input_rows;
        row.executed_experts = joinInts(std::move(unique_expert_ids));
        row.accumulation_path = is_cpu ? "CPU" : "GPU";
        recordRow(std::move(row));
    }

    void MoEExpertOverlayProfiler::recordGraphNativeReturnReduce(
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
        double import_broadcast_ms)
    {
        if (!isEnabled())
            return;

        MoEExpertOverlayProfileRow row;
        row.phase = "gn_return_reduce";
        row.layer = layer;
        row.tier_index = tier_index;
        row.domain = domain_key.empty() ? "unknown" : domain_key;
        row.participant_count = 2;
        row.selected_rows = inbound_rows;
        row.inbound_rows = inbound_rows;
        row.routed_entries = outbound_rows;
        row.outbound_bytes = compact_return_bytes;
        row.transfer_bytes = compact_return_bytes;
        row.compact_return_bytes = compact_return_bytes;
        row.dense_bytes_avoided = dense_return_bytes > compact_return_bytes
                                      ? dense_return_bytes - compact_return_bytes
                                      : 0;
        row.domain_reduce_ms = return_wait_ms;
        row.scatter_ms = scatter_ms;
        row.import_broadcast_ms = import_broadcast_ms;
        row.transport_mode = "compact";
        row.accumulation_path = source_participant == target_participant ? "local" : "remote";
        recordRow(std::move(row));
    }

} // namespace llaminar2
