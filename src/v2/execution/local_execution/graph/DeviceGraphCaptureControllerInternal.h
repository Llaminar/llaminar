/**
 * @file DeviceGraphCaptureControllerInternal.h
 * @brief Shared internal helpers and capture/replay diagnostics for DeviceGraphCaptureController shards.
 */
#pragma once
/**
 * @file DeviceGraphCaptureController.cpp
 * @brief Native graph capture lifecycle planning and execution.
 *
 * This implementation validates graph-owned lifecycle envelopes, materializes
 * their replay units, and preserves every producer/consumer edge with streams,
 * events, or an explicitly declared immutable host ticket.
 * Ticket evidence is authored by the logical lifecycle validator, before
 * retained-parent lowering changes the number of physical compiler units.
 * Native parent evidence separately projects that physical inventory through
 * the same schema used for deferred first launch and steady replay.
 * Cold materialization runs as one job on the participant's original worker;
 * descriptor preparation and native recording therefore share its thread-local
 * capture authority as well as its exact stream. This does not move ready
 * replay or add a per-stage dispatch or device synchronization.
 */

#include "DeviceGraphCaptureController.h"
#include "GraphCaptureGuard.h"
#include "GraphCaptureStageActivity.h"
#include "RetainedParentPerfEvidence.h"

#ifdef HAVE_CUDA
#include "../../../backends/cuda/CUDAGraphCapture.h"
#endif

#include "../coherence/CoherencePolicy.h"
#include "../../../tensors/TensorClasses.h"
#include "../../../utils/DebugEnv.h"
#include "../../../utils/ForwardPassProfiler.h"
#include "../../../utils/KernelProfiler.h"
#include "../../../utils/Logger.h"
#include "../../../utils/PerfStatsCollector.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <map>
#include <cstdlib>

#ifdef HAVE_ROCM
extern "C" uint64_t *rocmDebugAllocMappedClockBuf(size_t count, int device_id, int *wall_clock_khz_out);
extern "C" void rocmDebugRecordWallClock(uint64_t *host_ptr, uint32_t slot, int device_id, void *stream);

namespace llaminar2::dgcc_detail
{
struct RocmDecodeTimelineProbe {
    bool enabled = false;
    bool dumped = false;
    int replay_count = 0;
    uint64_t *buf = nullptr;
    int khz = 25000;
    struct Entry { std::string name; std::string type; };
    std::vector<Entry> entries;
    std::unordered_map<std::string, int> name_to_slot;
    int slotFor(const std::string &name, const std::string &type) {
        auto it = name_to_slot.find(name);
        if (it != name_to_slot.end()) return it->second;
        if (entries.size() >= 4096) return -1;
        int idx = static_cast<int>(entries.size());
        entries.push_back({name, type});
        name_to_slot.emplace(name, idx);
        return idx;
    }
    static RocmDecodeTimelineProbe &instance() {
        static RocmDecodeTimelineProbe p;
        return p;
    }
    void initIfNeeded() {
        // Timeline probe is retired in production source; keep no direct getenv outside DebugEnv.h.
        return;
    }
    ~RocmDecodeTimelineProbe() {
        if (enabled && buf && !dumped && !entries.empty()) {
            replay_count = 100;
            maybeDumpOnReplay(0);
        }
    }
    void maybeDumpOnReplay(int device_id = 0) {
        if (device_id != 0 || !enabled || !buf || dumped || entries.empty()) return;
        size_t nonzero = 0;
        for (size_t i = 0; i < entries.size(); ++i) {
            if (__atomic_load_n(&buf[2 * i + 1], __ATOMIC_ACQUIRE) != 0) ++nonzero;
        }
        if (nonzero < 10) return;
        if (++replay_count < 8) return;
        dumped = true;
        FILE *f = std::fopen("/tmp/qfn-dual-rocm-prof/timeline_probe.txt", "w");
        if (!f) f = stderr;
        const double tick_to_us = 1000.0 / static_cast<double>(khz);
        std::map<std::string, std::pair<int, double>> by_type;
        std::map<std::string, std::pair<int, double>> by_suffix;
        double total_us = 0.0;
        uint64_t first_t = 0, last_t = 0;
        for (size_t i = 0; i < entries.size(); ++i) {
            uint64_t t0 = __atomic_load_n(&buf[2 * i], __ATOMIC_ACQUIRE);
            uint64_t t1 = __atomic_load_n(&buf[2 * i + 1], __ATOMIC_ACQUIRE);
            if (t0 == 0 || t1 < t0) continue;
            if (first_t == 0) first_t = t0;
            if (t1 > last_t) last_t = t1;
            double us = static_cast<double>(t1 - t0) * tick_to_us;
            auto &acc = by_type[entries[i].type];
            acc.first += 1;
            acc.second += us;
            // Strip layerNN_ prefix to aggregate by logical stage name across layers
            std::string sfx = entries[i].name;
            if (sfx.rfind("layer", 0) == 0) {
                auto pos = sfx.find('_');
                if (pos != std::string::npos) sfx = "layer*_" + sfx.substr(pos + 1);
            }
            auto &sacc = by_suffix[sfx];
            sacc.first += 1;
            sacc.second += us;
            total_us += us;
        }
        double span_ms = (last_t > first_t) ? static_cast<double>(last_t - first_t) * tick_to_us / 1000.0 : 0.0;
        std::fprintf(f, "=== ROCM_DECODE_TIMELINE_PROBE (khz=%d, entries=%zu, nonzero=%zu, stage_sum=%.3f ms, span=%.3f ms) ===\n",
                     khz, entries.size(), nonzero, total_us / 1000.0, span_ms);
        std::fprintf(f, "\n--- BY STAGE TYPE ---\n");
        std::vector<std::pair<double, std::pair<std::string, int>>> sorted;
        for (const auto &[k, v] : by_type) sorted.push_back({v.second, {k, v.first}});
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        for (size_t i = 0; i < std::min<size_t>(sorted.size(), 40); ++i) {
            const auto &[us, info] = sorted[i];
            std::fprintf(f, "  %-42s n=%4d  avg=%7.2f us  sum=%7.3f ms\n",
                         info.first.c_str(), info.second, us / std::max(1, info.second), us / 1000.0);
        }
        std::fprintf(f, "\n--- BY LOGICAL STAGE NAME (top 45) ---\n");
        sorted.clear();
        for (const auto &[k, v] : by_suffix) sorted.push_back({v.second, {k, v.first}});
        std::sort(sorted.begin(), sorted.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        for (size_t i = 0; i < std::min<size_t>(sorted.size(), 45); ++i) {
            const auto &[us, info] = sorted[i];
            std::fprintf(f, "  %-50s n=%4d  avg=%7.2f us  sum=%7.3f ms\n",
                         info.first.c_str(), info.second, us / std::max(1, info.second), us / 1000.0);
        }
        std::fprintf(f, "\n=== LAYER 10 (GDN) + LAYER 3 (QSA) + TAIL STAGES ===\n");
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto &n = entries[i].name;
            if (n.find("layer10_") == 0 || n.find("layer3_") == 0 || n.find("lm_head") != std::string::npos || n.find("sample") != std::string::npos || n.find("final") != std::string::npos || n.find("embed") != std::string::npos) {
                uint64_t t0 = buf[2 * i], t1 = buf[2 * i + 1];
                double us = (t1 >= t0 && t0 > 0) ? static_cast<double>(t1 - t0) * tick_to_us : 0.0;
                double rel = (first_t > 0 && t0 >= first_t) ? static_cast<double>(t0 - first_t) * tick_to_us : 0.0;
                std::fprintf(f, "  [%8.1f us] dur=%7.2f us  %-32s %s\n",
                             rel, us, entries[i].type.c_str(), n.c_str());
            }
        }
        if (f != stderr) std::fclose(f);
    }
};
} // namespace llaminar2::dgcc_detail
using llaminar2::dgcc_detail::RocmDecodeTimelineProbe;
#endif

namespace llaminar2
{
    namespace
    {
        enum class GraphReplayCaptureMode
        {
            Segmented,
            FullGraph
        };

        /**
         * @brief Typed host-side phases of one native graph materialization.
         *
         * These phases are diagnostic subdivisions of the single capture
         * lifecycle transition. They never authorize capture, replay, or
         * serving admission; the graph cache remains the sole lifecycle owner.
         */
        enum class CaptureMaterializationPhase : std::size_t
        {
            ArenaPrebind = 0,
            LaunchPreparation,
            SnapshotManifest,
            InputCoherence,
            NativeRecord,
            NativeFinalize,
            Count,
        };

        /** @return Stable PerfStats name for one materialization phase. */
        constexpr const char *captureMaterializationPhaseName(
            CaptureMaterializationPhase phase) noexcept
        {
            switch (phase)
            {
            case CaptureMaterializationPhase::ArenaPrebind:
                return "arena_prebind";
            case CaptureMaterializationPhase::LaunchPreparation:
                return "launch_preparation";
            case CaptureMaterializationPhase::SnapshotManifest:
                return "snapshot_manifest";
            case CaptureMaterializationPhase::InputCoherence:
                return "input_coherence";
            case CaptureMaterializationPhase::NativeRecord:
                return "native_record";
            case CaptureMaterializationPhase::NativeFinalize:
                return "native_finalize";
            case CaptureMaterializationPhase::Count:
                break;
            }
            return "invalid";
        }

        /** @brief Bounded aggregate for repeated segment-local phase samples. */
        struct CaptureMaterializationAggregate
        {
            std::uint64_t total_ns = 0u; ///< Sum across every segment sample.
            std::uint64_t max_ns = 0u;   ///< Slowest individual segment sample.
            std::size_t samples = 0u;    ///< Number of observations accumulated.

            /** @brief Add one measured duration without changing lifecycle state. */
            void observe(std::uint64_t elapsed_ns) noexcept
            {
                total_ns += elapsed_ns;
                max_ns = std::max(max_ns, elapsed_ns);
                ++samples;
            }
        };

        /**
         * @brief Host cost of recording one declarative stage into a native graph.
         *
         * Stream capture normally makes a kernel launch a microsecond-scale host
         * operation. A library or collective is allowed to perform setup while
         * recording, however, and an accidental rendezvous there can make graph
         * materialization take seconds. These observations are setup-only and
         * bounded to one entry per declarative stage; they never time graph
         * replay or introduce a device synchronization.
         */
        struct CaptureStageRecordObservation
        {
            std::string stage_name;       ///< Stable declarative graph node name.
            std::string stage_type;       ///< Stable declarative stage category.
            std::uint64_t execute_ns = 0; ///< Canonical stage-recording host cost.
            std::uint64_t snapshot_ns = 0; ///< Diagnostic-copy recording host cost.

            /** @return Complete host recording cost attributed to this stage. */
            [[nodiscard]] std::uint64_t totalNs() const noexcept
            {
                return execute_ns + snapshot_ns;
            }
        };

        /**
         * @brief Graph-wide recording cost accumulated for one stage category.
         *
         * Per-segment slow-stage reports expose first-use outliers, while this
         * aggregate exposes a small cost repeated across every transformer
         * block. Keeping only totals, a maximum, and a sample count bounds the
         * setup diagnostic independently of model depth.
         */
        struct CaptureStageRecordAggregate
        {
            std::uint64_t execute_ns = 0u;  ///< Sum of canonical record calls.
            std::uint64_t snapshot_ns = 0u; ///< Sum of diagnostic-copy calls.
            std::uint64_t max_ns = 0u;      ///< Slowest complete stage sample.
            std::size_t samples = 0u;       ///< Number of declarative stages.

            /** @brief Accumulate one already-measured declarative stage. */
            void observe(const CaptureStageRecordObservation &observation) noexcept
            {
                execute_ns += observation.execute_ns;
                snapshot_ns += observation.snapshot_ns;
                max_ns = std::max(max_ns, observation.totalNs());
                ++samples;
            }

            /** @return Complete recording cost across this stage category. */
            [[nodiscard]] std::uint64_t totalNs() const noexcept
            {
                return execute_ns + snapshot_ns;
            }
        };

        /**
         * @brief Report a bounded attribution when native recording is pathological.
         *
         * A normal graph does not emit this diagnostic. Once recording exceeds
         * one second, the slowest stages and unattributed begin/end bookkeeping
         * are logged so a production graph can be reduced without flooding the
         * log with every layer. Host clocks only surround already-setup capture
         * work and therefore cannot alter device ordering.
         */
        void logSlowCaptureStageAttribution(
            DeviceId device,
            const std::string &context,
            std::span<const CaptureStageRecordObservation> observations,
            std::uint64_t begin_ns,
            std::uint64_t close_ns,
            std::uint64_t native_record_ns)
        {
            constexpr std::uint64_t kSlowCaptureNs = 1'000'000'000ull;
            constexpr std::size_t kMaximumReportedStages = 12u;
            if (native_record_ns < kSlowCaptureNs)
                return;

            std::vector<std::size_t> order(observations.size());
            for (std::size_t index = 0u; index < order.size(); ++index)
                order[index] = index;
            const std::size_t reported =
                std::min(kMaximumReportedStages, order.size());
            std::partial_sort(
                order.begin(),
                order.begin() + static_cast<std::ptrdiff_t>(reported),
                order.end(),
                [&](std::size_t lhs, std::size_t rhs)
                {
                    return observations[lhs].totalNs() >
                           observations[rhs].totalNs();
                });

            std::uint64_t stage_total_ns = 0u;
            for (const auto &observation : observations)
                stage_total_ns += observation.totalNs();

            std::ostringstream slowest;
            for (std::size_t rank = 0u; rank < reported; ++rank)
            {
                const auto &observation = observations[order[rank]];
                if (rank != 0u)
                    slowest << ';';
                slowest << observation.stage_name
                        << "{execute_ms="
                        << static_cast<double>(observation.execute_ns) / 1.0e6
                        << ",snapshot_ms="
                        << static_cast<double>(observation.snapshot_ns) / 1.0e6
                        << '}';
            }

            const std::uint64_t accounted_ns =
                std::min(native_record_ns, begin_ns + close_ns + stage_total_ns);
            LOG_INFO(
                "[GraphCaptureStageAttribution] slow native recording"
                << " device=" << device.toString()
                << " context=" << context
                << " native_record_ms="
                << static_cast<double>(native_record_ns) / 1.0e6
                << " capture_begin_ms="
                << static_cast<double>(begin_ns) / 1.0e6
                << " capture_close_ms="
                << static_cast<double>(close_ns) / 1.0e6
                << " stage_total_ms="
                << static_cast<double>(stage_total_ns) / 1.0e6
                << " unattributed_ms="
                << static_cast<double>(native_record_ns - accounted_ns) / 1.0e6
                << " stages=" << observations.size()
                << " slowest=" << slowest.str());
        }

        /**
         * @brief Emit one bounded graph-wide attribution after materialization.
         *
         * This is deliberately a setup-only host-clock diagnostic. It neither
         * synchronizes a device nor claims execution authority; its purpose is
         * to distinguish one expensive first-use call from a modest library or
         * collective registration cost repeated at every layer.
         */
        void logCaptureStageTypeAggregate(
            DeviceId device,
            const std::string &context,
            const std::map<std::string, CaptureStageRecordAggregate> &aggregates)
        {
            constexpr std::size_t kMaximumReportedTypes = 12u;
            if (aggregates.empty())
                return;

            std::vector<std::reference_wrapper<const std::pair<
                const std::string,
                CaptureStageRecordAggregate>>>
                order;
            order.reserve(aggregates.size());
            for (const auto &entry : aggregates)
                order.emplace_back(std::cref(entry));

            const std::size_t reported =
                std::min(kMaximumReportedTypes, order.size());
            std::partial_sort(
                order.begin(),
                order.begin() + static_cast<std::ptrdiff_t>(reported),
                order.end(),
                [](const auto &lhs, const auto &rhs)
                {
                    return lhs.get().second.totalNs() >
                           rhs.get().second.totalNs();
                });

            std::uint64_t stage_total_ns = 0u;
            std::size_t stage_samples = 0u;
            for (const auto &[stage_type, aggregate] : aggregates)
            {
                (void)stage_type;
                stage_total_ns += aggregate.totalNs();
                stage_samples += aggregate.samples;
            }

            std::ostringstream categories;
            for (std::size_t rank = 0u; rank < reported; ++rank)
            {
                const auto &[stage_type, aggregate] = order[rank].get();
                if (rank != 0u)
                    categories << ';';
                categories
                    << stage_type
                    << "{total_ms="
                    << static_cast<double>(aggregate.totalNs()) / 1.0e6
                    << ",execute_ms="
                    << static_cast<double>(aggregate.execute_ns) / 1.0e6
                    << ",snapshot_ms="
                    << static_cast<double>(aggregate.snapshot_ns) / 1.0e6
                    << ",max_ms="
                    << static_cast<double>(aggregate.max_ns) / 1.0e6
                    << ",samples=" << aggregate.samples << '}';
            }

            LOG_INFO(
                "[GraphCaptureStageTypeAggregate] complete"
                << " device=" << device.toString()
                << " context=" << context
                << " stage_total_ms="
                << static_cast<double>(stage_total_ns) / 1.0e6
                << " stage_samples=" << stage_samples
                << " stage_types=" << aggregates.size()
                << " slowest_types=" << categories.str());
        }

        const char *graphExecutableLaunchPhaseName(
            DeviceGraphExecutor::GraphExecutableLaunchPhase phase)
        {
            using Phase = DeviceGraphExecutor::GraphExecutableLaunchPhase;
            switch (phase)
            {
            case Phase::InitialTransaction:
                return "initial_transaction";
            case Phase::SteadyReplay:
                return "steady_replay";
            case Phase::DiagnosticRecapture:
                return "diagnostic_recapture";
            case Phase::DiagnosticVerification:
                return "diagnostic_verification";
            }
            return "unknown";
        }

        bool applyGraphLaunchDependency(
            const DeviceGraphExecutor::GraphLaunchDependencyHook &hook,
            DeviceGraphExecutor::GraphExecutableLaunchPhase phase,
            void *execution_stream)
        {
            if (!hook)
                return true;
            if (!execution_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Captured executable launch "
                    "dependency requires the exact non-null graph stream"
                    << " phase=" << graphExecutableLaunchPhaseName(phase));
                return false;
            }
            if (!hook(phase, execution_stream))
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Captured executable launch "
                    "dependency was rejected"
                    << " phase=" << graphExecutableLaunchPhaseName(phase)
                    << " stream=" << execution_stream);
                return false;
            }
            return true;
        }

        const char *captureModeTag(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph ? "full_graph" : "segmented";
        }

        const char *captureModeSource(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_capture"
                       : "segmented_graph_capture";
        }

        const char *captureModeHostScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_replay_host"
                       : "segmented_replay_host";
        }

        const char *captureModeEventScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_replay_events"
                       : "segmented_replay_events";
        }

        const char *captureModePlanScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_capture_plan"
                       : "segmented_capture_plan";
        }

        std::string replayMetricName(GraphReplayCaptureMode mode, const char *suffix)
        {
            return std::string(mode == GraphReplayCaptureMode::FullGraph
                                   ? "full_graph_replay_"
                                   : "segmented_replay_") +
                   suffix;
        }

        std::string planMetricName(GraphReplayCaptureMode mode, const char *suffix)
        {
            return std::string(mode == GraphReplayCaptureMode::FullGraph
                                   ? "full_graph_plan_"
                                   : "segmented_plan_") +
                   suffix;
        }

        std::string planUnitMetricName(GraphReplayCaptureMode mode)
        {
            return planMetricName(mode, mode == GraphReplayCaptureMode::FullGraph
                                            ? "graphs"
                                            : "segments");
        }

        std::string planMaxUnitStagesMetricName(GraphReplayCaptureMode mode)
        {
            return planMetricName(mode, mode == GraphReplayCaptureMode::FullGraph
                                            ? "max_graph_stages"
                                            : "max_segment_stages");
        }

        const char *stageGpuPlanUnitMetricName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_replay_plan_graphs"
                       : "graph_replay_plan_segments";
        }

        const char *planUnitTypeTagName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_type"
                       : "segment_type";
        }

        const char *replayUnitCounterName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "full_graph_replay_calls"
                       : "segmented_replay_segments";
        }

        std::string replayUnitTimingName(GraphReplayCaptureMode mode)
        {
            return replayMetricName(mode, mode == GraphReplayCaptureMode::FullGraph
                                              ? "graph"
                                              : "segment");
        }

        const char *replayUnitGpuEventName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_replay.graph"
                       : "graph_replay.segment";
        }

        const char *replayUnitIndexTagName(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_index"
                       : "segment_index";
        }

        const char *replayUnitTimingScope(GraphReplayCaptureMode mode)
        {
            return mode == GraphReplayCaptureMode::FullGraph
                       ? "graph_replay_gpu_event"
                       : "segment_replay_gpu_event";
        }

        GraphReplayCaptureMode captureModeForPlan(size_t total_segments,
                                                  size_t capturable_segments,
                                                  size_t manual_segments)
        {
            return total_segments == 1 && capturable_segments == 1 && manual_segments == 0
                       ? GraphReplayCaptureMode::FullGraph
                       : GraphReplayCaptureMode::Segmented;
        }

        GraphReplayCaptureMode captureModeForCache(const DeviceGraphExecutor::GraphSegmentCache &cache)
        {
            size_t capturable_segments = 0;
            size_t manual_segments = 0;
            for (const auto &segment : cache.segments)
            {
                if (segment.capturable)
                    ++capturable_segments;
                else
                    ++manual_segments;
            }
            return captureModeForPlan(cache.segments.size(), capturable_segments, manual_segments);
        }

        /**
         * @return Observable replay topology for one sealed or planned cache.
         *
         * A retained parent may be assembled from several capture-plan units,
         * including a mapped host ticket service, but submits exactly one GPU
         * executable. Reporting it as segmented would make PerfStats claim the
         * serial child-replay path ran when it did not.
         */
        const char *replayTopologyTag(
            const DeviceGraphExecutor::GraphSegmentCache &cache) noexcept
        {
            if (DeviceGraphExecutor::isRetainedParentPlanPolicy(
                    cache.graph_replay_plan_policy))
            {
                return "retained_parent";
            }
            return captureModeTag(captureModeForCache(cache));
        }

        const char *segmentTypeName(const DeviceGraphExecutor::GraphSegment &segment)
        {
            return segment.capturable ? "capturable" : "manual";
        }

        std::string describeSegmentStages(const DeviceGraphExecutor::GraphSegment &segment)
        {
            std::ostringstream out;
            out << '[';
            for (size_t i = 0; i < segment.stage_names.size(); ++i)
            {
                if (i > 0)
                    out << ", ";
                out << segment.stage_names[i];
            }
            out << ']';
            return out.str();
        }

        /** @brief Participant role in one heterogeneous ticket segment plan. */
        enum class HeterogeneousTicketSegmentRole : std::uint8_t
        {
            Authority, ///< Owns one manual service after each typed cutpoint.
            Follower, ///< Owns only captured units aligned to those cutpoints.
        };

        /** @brief Ticket-unit markers physically contained by one plan segment. */
        struct HeterogeneousTicketSegmentMarkers
        {
            std::size_t before_manual = 0u; ///< BeforeManualBoundary contracts.
            std::size_t terminal = 0u; ///< TransactionTerminal contracts.
        };

        /** @brief Authority that proves one replay transaction has completed. */
        enum class ReplayTerminalFenceAuthority : std::uint8_t
        {
            HostObserver, ///< The caller must observe the capture-stream fence.
            CaptureStreamEvent, ///< A dependent consumer may await the exact stream event.
        };

        /** @brief Resolved terminal-fence contract for one replay invocation. */
        struct ReplayTerminalFencePlan
        {
            ReplayTerminalFenceAuthority authority =
                ReplayTerminalFenceAuthority::HostObserver; ///< Sole completion authority.

            /** @return Whether replay may return after publishing device ordering. */
            [[nodiscard]] bool defersHostObservation() const noexcept
            {
                return authority ==
                       ReplayTerminalFenceAuthority::CaptureStreamEvent;
            }
        };

        /** @return Stable diagnostic name for a heterogeneous ticket role. */
        constexpr const char *heterogeneousTicketSegmentRoleName(
            HeterogeneousTicketSegmentRole role) noexcept
        {
            switch (role)
            {
            case HeterogeneousTicketSegmentRole::Authority:
                return "authority";
            case HeterogeneousTicketSegmentRole::Follower:
                return "follower";
            }
            return "invalid";
        }

        /**
         * @brief Validate typed ticket ordering without assuming child count.
         *
         * Scalar packet frontiers may deliberately split a producer or
         * consumer unit into additional native children before retained-parent
         * composition. The ticket lifecycle is therefore defined by marker
         * adjacency, not by `captured == boundaries + 1` arithmetic. Every
         * authority marker must be followed immediately by its one manual
         * service and another captured unit. A follower has the same ordered
         * marker cutpoints but no manual segment. Both roles terminate exactly
         * once in the final captured unit.
         *
         * @param graph Source of the typed per-node ticket contracts.
         * @param segments Logical plan before retained-parent lowering.
         * @param role Authority or follower ownership of manual service.
         * @param declared_before_manual Number of declared ticket cutpoints.
         * @param declared_terminal Number of declared transaction terminals.
         * @throws std::runtime_error if ordering, ownership or terminal identity
         *         is invalid. Only a successful validation publishes evidence.
         */
        void validateHeterogeneousTicketSegmentLifecycle(
            const ComputeGraph &graph,
            std::span<const DeviceGraphExecutor::GraphSegment> segments,
            HeterogeneousTicketSegmentRole role,
            std::size_t declared_before_manual,
            std::size_t declared_terminal)
        {
            std::vector<HeterogeneousTicketSegmentMarkers> markers(
                segments.size());
            std::size_t observed_before_manual = 0u;
            std::size_t observed_terminal = 0u;
            std::size_t observed_manual_segments = 0u;
            std::size_t observed_captured_segments = 0u;

            const auto fail = [&](const std::string &reason)
            {
                std::ostringstream diagnostic;
                diagnostic
                    << "Invalid heterogeneous ticket "
                    << heterogeneousTicketSegmentRoleName(role)
                    << " segment lifecycle: " << reason
                    << "; segments=" << segments.size()
                    << " captured=" << observed_captured_segments
                    << " manual=" << observed_manual_segments
                    << " declared_boundaries=" << declared_before_manual
                    << " observed_boundaries=" << observed_before_manual
                    << " declared_terminal=" << declared_terminal
                    << " observed_terminal=" << observed_terminal;
                const std::size_t diagnostic_segments =
                    std::min<std::size_t>(segments.size(), 8u);
                for (std::size_t index = 0u;
                     index < diagnostic_segments;
                     ++index)
                {
                    const auto &segment = segments[index];
                    diagnostic << " segment[" << index << "]={"
                               << (segment.capturable ? "captured" : "manual")
                               << ",before=" << markers[index].before_manual
                               << ",terminal=" << markers[index].terminal
                               << ",identity="
                               << (segment.capture_wave_identity.empty()
                                       ? "<implicit>"
                                       : segment.capture_wave_identity)
                               << ",first="
                               << (segment.stage_names.empty()
                                       ? "<empty>"
                                       : segment.stage_names.front())
                               << ",last="
                               << (segment.stage_names.empty()
                                       ? "<empty>"
                                       : segment.stage_names.back())
                               << '}';
                }
                throw std::runtime_error(diagnostic.str());
            };

            if (segments.empty())
                fail("the plan is empty");

            for (std::size_t segment_index = 0u;
                 segment_index < segments.size();
                 ++segment_index)
            {
                const auto &segment = segments[segment_index];
                if (segment.stage_names.empty())
                    fail("a plan segment has no stages");
                if (segment.capturable)
                    ++observed_captured_segments;
                else
                    ++observed_manual_segments;

                for (const std::string &stage_name : segment.stage_names)
                {
                    const ComputeNode *const node = graph.getNode(stage_name);
                    if (!node || !node->stage)
                    {
                        fail("a segment names unresolved stage '" +
                             stage_name + "'");
                    }
                    if (!node->heterogeneous_ticket_unit_contract)
                        continue;
                    if (!segment.capturable)
                    {
                        fail("ticket-unit marker '" + stage_name +
                             "' is attached to manual work");
                    }
                    switch (node->heterogeneous_ticket_unit_contract
                                ->disposition)
                    {
                    case GraphHeterogeneousTicketUnitDisposition::
                        BeforeManualBoundary:
                        ++markers[segment_index].before_manual;
                        ++observed_before_manual;
                        break;
                    case GraphHeterogeneousTicketUnitDisposition::
                        TransactionTerminal:
                        ++markers[segment_index].terminal;
                        ++observed_terminal;
                        break;
                    }
                }
                if (markers[segment_index].before_manual > 1u ||
                    markers[segment_index].terminal > 1u ||
                    (markers[segment_index].before_manual != 0u &&
                     markers[segment_index].terminal != 0u))
                {
                    fail("one captured segment contains conflicting ticket-unit markers");
                }
            }

            if (!segments.front().capturable ||
                !segments.back().capturable)
            {
                fail("the transaction must begin and end with captured device work");
            }
            if (declared_before_manual == 0u ||
                observed_before_manual != declared_before_manual)
            {
                fail("the planned cutpoints do not match the declared ticket boundaries");
            }
            if (declared_terminal != 1u || observed_terminal != 1u ||
                markers.back().terminal != 1u)
            {
                fail("the final captured unit is not the sole transaction terminal");
            }

            for (std::size_t index = 0u; index < segments.size(); ++index)
            {
                const bool closes_ticket =
                    markers[index].before_manual == 1u;
                if (role == HeterogeneousTicketSegmentRole::Authority)
                {
                    if (closes_ticket)
                    {
                        if (index + 2u >= segments.size() ||
                            segments[index + 1u].capturable ||
                            !segments[index + 2u].capturable)
                        {
                            fail("an authority cutpoint is not followed by one manual service and captured ingress");
                        }
                    }
                    if (!segments[index].capturable)
                    {
                        if (index == 0u ||
                            markers[index - 1u].before_manual != 1u)
                        {
                            fail("manual service has no immediately preceding authenticated cutpoint");
                        }
                    }
                }
                else
                {
                    if (!segments[index].capturable)
                    {
                        fail("a follower contains participant-local manual work");
                    }
                    if (closes_ticket &&
                        (index + 1u >= segments.size() ||
                         !segments[index + 1u].capturable))
                    {
                        fail("a follower cutpoint has no following captured unit");
                    }
                }
            }

            if (role == HeterogeneousTicketSegmentRole::Authority &&
                observed_manual_segments != observed_before_manual)
            {
                fail("authority manual-service count does not match its authenticated cutpoints");
            }

            /* These are logical lifecycle counts, not physical compiler units.
             * Lowering can combine 48 CPU boundaries into one service program,
             * while graph-only GPU children are imported into one parent.
             * Publish here, from the validator's own observations, so no later
             * observer has to reconstruct adjacency from lowered child counts. */
            PerfStatsCollector::addCounter(
                "forward_graph", "heterogeneous_ticket_transactions", 1.0,
                "capture", "",
                {{"lifecycle_contract", "typed_marker_adjacency_v1"},
                 {"capturable_segments", std::to_string(observed_captured_segments)},
                 {"manual_segments", std::to_string(observed_manual_segments)},
                 {"ticket_publication_authority", "stage_owned_mapped_timeline"},
                 {"unit_boundaries", std::to_string(observed_before_manual)},
                 {"terminal_units", std::to_string(observed_terminal)},
                 {"role", heterogeneousTicketSegmentRoleName(role)}});
        }

        /**
         * @brief Resolve the sole completion authority for a replay transaction.
         *
         * A fully captured graph can publish its capture-stream event when its
         * collectives are native graph nodes. A typed heterogeneous ticket
         * transaction has a different, equally exact proof: capture planning
         * validates that it ends in one captured terminal unit, while every
         * intervening manual collective publishes its completion back to the
         * capture stream. Consequently that terminal stream event covers the
         * whole captured/manual/captured DAG and a host wait would merely add
         * scheduler latency before the event-aware consumer is enqueued.
         *
         * Ordinary segmented graphs retain a host observer. Their topology has
         * no graph-owned terminal contract proving that a downstream caller is
         * allowed to consume an asynchronously published transaction.
         *
         * @param graph Declarative graph whose envelope owns replay semantics.
         * @param cache Materialized replay plan on the exact capture stream.
         * @param has_collective_nodes Whether the source graph has collectives.
         * @param collectives_graph_capturable Whether those collectives are in
         *        native graph executables.
         * @param defer_requested Whether the caller accepts event completion.
         * @return Typed authority that must complete this replay invocation.
         * @throws std::logic_error when a typed heterogeneous plan has lost its
         *         required captured terminal or exact replay policy.
         */
        ReplayTerminalFencePlan resolveReplayTerminalFencePlan(
            const ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegmentCache &cache,
            bool has_collective_nodes,
            bool collectives_graph_capturable,
            bool defer_requested)
        {
            if (!defer_requested)
                return {};

            const bool every_unit_captured = std::all_of(
                cache.segments.begin(),
                cache.segments.end(),
                [](const DeviceGraphExecutor::GraphSegment &segment)
                {
                    return segment.capturable;
                });
            const bool complete_native_dag =
                every_unit_captured &&
                (!has_collective_nodes || cache.runtime_collective_capture_requirement==DeviceGraphExecutor::GraphSegmentCache::RuntimeCollectiveCaptureRequirement::NotRequired || collectives_graph_capturable);
            if (complete_native_dag)
            {
                return {
                    .authority =
                        ReplayTerminalFenceAuthority::CaptureStreamEvent};
            }

            if (!requiresHeterogeneousTicketSegmentation(
                    graph.nativeCaptureEnvelope()))
            {
                return {};
            }

            if (cache.graph_replay_plan_policy !=
                    DeviceGraphExecutor::GraphReplayPlanPolicy::
                        AllowHeterogeneousBoundarySegmentation ||
                cache.segments.empty() ||
                !cache.segments.front().capturable ||
                !cache.segments.back().capturable)
            {
                throw std::logic_error(
                    "A typed heterogeneous replay lost its captured/manual/"
                    "captured terminal-fence contract");
            }

            const std::string terminal_name = graph.terminalNode();
            const auto &terminal_segment = cache.segments.back();
            if (terminal_name.empty() ||
                terminal_segment.stage_names.empty() ||
                terminal_segment.stage_names.back() != terminal_name)
            {
                throw std::logic_error(
                    "A typed heterogeneous replay does not end at its declared "
                    "graph terminal");
            }

            const ComputeNode *const terminal = graph.getNode(terminal_name);
            if (!terminal || !terminal->heterogeneous_ticket_unit_contract ||
                terminal->heterogeneous_ticket_unit_contract->disposition !=
                    GraphHeterogeneousTicketUnitDisposition::
                        TransactionTerminal ||
                !terminal_segment.capture ||
                !terminal_segment.capture->hasExecutable() ||
                terminal_segment.capture->executionStream() !=
                    cache.capture_stream)
            {
                throw std::logic_error(
                    "A typed heterogeneous replay terminal is not one exact "
                    "captured executable on the cache stream");
            }

            return {
                .authority =
                    ReplayTerminalFenceAuthority::CaptureStreamEvent};
        }

        /**
         * @brief Lower an authenticated ticket lifecycle to bounded compilation
         *        units and one concurrent CPU service program.
         *
         * The declarative graph alternates GPU packet work and CPU ticket work so
         * dependency validation can prove every cutpoint. Runtime execution does
         * not alternate graph launches: every captured unit below is imported by
         * the retained-parent composer and only that one parent executable may be
         * submitted. The units are deliberately kept bounded during native graph
         * recording, however. RCCL capture registration becomes substantially
         * more expensive when hundreds of preceding compute and collective nodes
         * already inhabit the same HIP capture, while importing several graph-only
         * children into one parent has no request-time launch cost.
         *
         * Manual stages are the other concurrent program. The persistent CPU
         * worker services them in declarative order while device packet waits keep
         * their original positions in the retained parent. Consequently compile
         * sharding changes neither the typed ticket protocol nor runtime ownership.
         *
         * @param segments Validated alternating semantic segments to consume.
         * @param include_manual_service Whether this endpoint owns CPU service.
         * @return Ordered captured compilation units followed, when owned, by one
         *         CPU service program.
         */
        std::vector<DeviceGraphExecutor::GraphSegment>
        lowerHeterogeneousTicketCompilationPlan(
            std::vector<DeviceGraphExecutor::GraphSegment> segments,
            bool include_manual_service)
        {
            DeviceGraphExecutor::GraphSegment service_program;
            service_program.capturable = false;

            std::vector<DeviceGraphExecutor::GraphSegment> programs;
            programs.reserve(segments.size() +
                             (include_manual_service ? 1u : 0u));

            for (auto &segment : segments)
            {
                if (!segment.passive_capture_waves_after.empty())
                {
                    throw std::runtime_error(
                        "A heterogeneous ticket compilation unit cannot contain passive capture waves");
                }

                if (segment.capturable)
                {
                    /*
                     * Preserve the typed ticket/layer frontier and its capture
                     * identity. This is a graph-only compiler shard: it is never
                     * exposed as an independently launchable runtime program.
                     */
                    programs.push_back(std::move(segment));
                    continue;
                }

                service_program.stage_names.insert(
                    service_program.stage_names.end(),
                    std::make_move_iterator(segment.stage_names.begin()),
                    std::make_move_iterator(segment.stage_names.end()));
            }

            if (programs.empty())
            {
                throw std::runtime_error(
                    "A heterogeneous ticket compilation plan has no capturable device work");
            }
            if (include_manual_service !=
                !service_program.stage_names.empty())
            {
                throw std::runtime_error(
                    include_manual_service
                        ? "A heterogeneous ticket authority has no CPU service program"
                        : "A heterogeneous ticket follower unexpectedly owns CPU service work");
            }

            if (include_manual_service)
                programs.push_back(std::move(service_program));
            return programs;
        }

        /**
         * @brief Expose the lifecycle's actual main-stage frontier for its finite parent body.
         * @param graph Original participant graph and its declared finite body.
         * @param segments Validated compiler units, before capture or input binding.
         * @note This changes only graph-only compilation units. The complete
         * parent still executes once; no new CPU ticket or submission is added.
         * CurrentDemand forks after Probe and may wait for a later LocalRead.
         * FutureProducer forks after its actual Probe without a model read.
         * Local-only consumption forks after the selected actual Read/Discard.
         * None of these finite bodies joins the producing fragment's exit.
         */
        void exposeParticipantAuxiliaryProbeFrontier(
            const ComputeGraph& graph,
            std::vector<DeviceGraphExecutor::GraphSegment>& segments)
        {
            const auto& range=graph.retainedParticipantAuxiliaryRange();
            if(!range)return;
            const bool needsRead=range->lifecycle==RetainedGraphParticipantAuxiliaryLifecycle::CurrentDemand;
            const std::string frontier=needsRead ? "participant_auxiliary_probe" :
                range->lifecycle==RetainedGraphParticipantAuxiliaryLifecycle::FutureProducer
                    ? "participant_future_producer" : "participant_local_consumption";
            std::size_t producer=segments.size(),consumer=segments.size(),offset=0;
            bool sawRead=false;
            for(std::size_t index=0;index<segments.size();++index) {
                const auto& segment=segments[index];
                for(std::size_t row=0;row<segment.stage_names.size();++row) {
                    const auto& name=segment.stage_names[row];
                    if(needsRead && name==range->main_read) {
                        if(producer==segments.size() || sawRead)
                            throw std::logic_error("Participant LocalRead must follow its unique current probe");
                        sawRead=true;consumer=index;
                    }
                    if(name!=range->probe_ready)continue;
                    if(producer!=segments.size() || !segment.capturable)
                        throw std::logic_error("Participant probe must belong to one captured main unit");
                    producer=index;offset=row;
                }
            }
            if(producer==segments.size() || (needsRead && !sawRead))
                throw std::logic_error("Participant auxiliary lost its main probe/read frontier");
            auto& unit=segments[producer];
            const auto continuation_identity = [&]() -> std::string {
                const auto& transaction=graph.nativeTransactionCaptureContract();
                return transaction ? transaction->identity : "participant_auxiliary_continuation";
            };
            if(offset+1==unit.stage_names.size()) {
                if(unit.capture_wave_identity.empty())
                    unit.capture_wave_identity=frontier;
                if(needsRead && segments[consumer].capture_wave_identity.empty())
                    segments[consumer].capture_wave_identity=continuation_identity();
                return;
            }
            DeviceGraphExecutor::GraphSegment prefix;
            prefix.capturable=true;
            // Explicit semantic identity is shared across the participant
            // domain. Never include a device, local pointer or stage spelling.
            prefix.capture_wave_identity=unit.capture_wave_identity.empty()
                ? frontier
                : unit.capture_wave_identity+":"+frontier;
            // The new continuation half is also a shared logical frontier.
            // Leaving it implicit would derive the TP rendezvous from local
            // source/reader stage names, even when the two programs agree.
            // Existing ticket and whole-native transaction identities survive.
            if(unit.capture_wave_identity.empty())
                unit.capture_wave_identity=continuation_identity();
            prefix.stage_names.assign(unit.stage_names.begin(),unit.stage_names.begin()+offset+1);
            unit.stage_names.erase(unit.stage_names.begin(),unit.stage_names.begin()+offset+1);
            segments.insert(segments.begin()+producer,std::move(prefix));
        }

        /**
         * @brief Materialize one frozen stage-order ledger before beginCapture().
         *
         * The production executor supplies arena-derived internal edges.  Pure
         * controller tests have no BufferArena and receive an empty plan.  If
         * such a test invokes the production runStage() path, stage entry fails
         * immediately because there is no planned stage; simple state-machine
         * callback tests can still exercise begin/end orchestration without
         * constructing tensors.
         */
        std::unique_ptr<GraphCaptureDependencyLedger> planCaptureDependencies(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment,
            DeviceId device,
            void *stream,
            const std::function<std::unique_ptr<GraphCaptureDependencyLedger>(
                const DeviceGraphExecutor::GraphSegment &)> &planner,
            const std::string &context)
        {
            if (planner)
                return planner(segment);

            (void)graph;
            (void)segment;
            std::vector<GraphCaptureDependencyLedger::StagePlan> stages;

            try
            {
                return std::make_unique<GraphCaptureDependencyLedger>(
                    device, stream, std::move(stages), context);
            }
            catch (const std::exception &e)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Strict capture dependency "
                    "planning failed ("
                    << context << "): " << e.what());
                return nullptr;
            }
        }

        /**
         * @brief Return the strict cross-participant name for one capture wave.
         *
         * Raw replay-segment indexes are intentionally absent. A heterogeneous
         * root owns manual sparse segments that its sibling device graphs do not
         * contain. Capture-wave ordinals count only native-capture lifecycle
         * participation and therefore remain symmetric. Explicit identities
         * admit deliberately different stage ranges; implicit waves retain the
         * old first/last-stage structural check.
         */
        std::string graphCaptureBoundaryName(
            const char *phase,
            uint64_t current_step,
            size_t capture_wave_ordinal,
            const std::string &capture_wave_identity,
            const DeviceGraphExecutor::GraphSegment *active_segment,
            const std::string &perf_context)
        {
            std::ostringstream out;
            out << "before_begin:"
                << "phase=" << phase
                << ":step=" << current_step
                << ":wave=" << capture_wave_ordinal;
            if (!capture_wave_identity.empty())
            {
                out << ":identity=explicit:" << capture_wave_identity;
            }
            else if (active_segment)
            {
                const std::string first_stage =
                    active_segment->stage_names.empty()
                        ? std::string("<empty>")
                        : active_segment->stage_names.front();
                const std::string last_stage =
                    active_segment->stage_names.empty()
                        ? std::string("<empty>")
                        : active_segment->stage_names.back();
                out << ":identity=implicit:"
                    << first_stage << ".." << last_stage;
            }
            else
            {
                throw std::logic_error(
                    "Passive capture wave requires an explicit identity");
            }
            if (!perf_context.empty())
                out << ":context=" << perf_context;
            return out.str();
        }

        /**
         * @brief Join sibling-only capture waves after the anchor segment completes.
         *
         * The anchor may be a launched graph or a completed manual collective.
         * The begin rendezvous holds this participant while a sibling performs
         * intervening heterogeneous work. The immediate end rendezvous then
         * holds it until that sibling has finished recording and launching its
         * native graph. No empty graph, dummy transfer, device allocation, or
         * kernel is introduced on the passive participant.
         */
        bool joinPassiveCaptureWaves(
            ComputeGraph &graph,
            const DeviceGraphExecutor::GraphSegment &segment,
            const char *phase_prefix,
            uint64_t current_step,
            const std::string &perf_context,
            const DeviceGraphExecutor::GraphCaptureBoundaryHook &capture_boundary,
            void *capture_stream,
            const DeviceId &device)
        {
            if (segment.passive_capture_waves_after.empty())
                return true;
            if (!capture_boundary || !capture_stream)
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Passive capture waves require "
                    "a domain boundary hook and exact non-null stream");
                return false;
            }

            for (const auto &passive_wave :
                 segment.passive_capture_waves_after)
            {
                const std::string begin_phase =
                    std::string(phase_prefix) + "_begin";
                const std::string begin_boundary =
                    graphCaptureBoundaryName(
                        begin_phase.c_str(),
                        current_step,
                        passive_wave.ordinal,
                        passive_wave.identity,
                        nullptr,
                        perf_context);
                if (!capture_boundary(begin_boundary, capture_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Passive capture-wave begin "
                        "rendezvous failed boundary="
                        << begin_boundary);
                    return false;
                }

                const std::string end_phase =
                    std::string(phase_prefix) + "_end";
                const std::string end_boundary =
                    graphCaptureBoundaryName(
                        end_phase.c_str(),
                        current_step,
                        passive_wave.ordinal,
                        passive_wave.identity,
                        nullptr,
                        perf_context);
                if (!capture_boundary(end_boundary, capture_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Passive capture-wave end "
                        "rendezvous failed boundary="
                        << end_boundary);
                    return false;
                }

                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "passive_capture_wave_joins",
                    1.0,
                    phase_prefix,
                    device.toString(),
                    {{"identity", passive_wave.identity},
                     {"wave", std::to_string(passive_wave.ordinal)},
                     {"context", perf_context}});

                /*
                 * These graph nodes are immutable no-ops on this participant.
                 * Mark them complete only after the sibling's active wave has
                 * both launched and left capture, which is their declarative
                 * position in the shared execution schedule.
                 */
                for (const auto &stage_name :
                     passive_wave.declarative_noop_stage_names)
                {
                    graph.markCompleted(stage_name);
                }
            }
            return true;
        }

        void addContextTag(PerfStatsCollector::Tags &tags, const std::string &perf_context)
        {
            if (!perf_context.empty())
                tags.emplace("context", perf_context);
        }

        /**
         * @brief Attach immutable cache-key geometry to one replay metric.
         *
         * The descriptor belongs to the same GraphSegmentCache as the timing
         * event ring.  This is essential for deferred collection: by the time a
         * stop event is queried, another MTP transaction may already be active
         * on the host.  No runner state is sampled here.
         */
        void addReplayWorkloadTags(
            PerfStatsCollector::Tags &tags,
            const DeviceGraphExecutor::GraphSegmentCache::ReplayWorkloadGeometry &workload)
        {
            if (!workload.valid())
                return;

            tags.emplace("seq_len", std::to_string(workload.seq_len));
            tags.emplace("batch_size", std::to_string(workload.batch_size));
            tags.emplace("m", std::to_string(workload.m));
            tags.emplace(
                "all_position_rows",
                std::to_string(workload.all_position_rows));
            tags.emplace(
                "verifier_outcome_mode",
                std::to_string(workload.verifier_outcome_mode));
            tags.emplace(
                "position_policy",
                std::to_string(workload.position_policy));
            tags.emplace(
                "moe_placement_epoch",
                std::to_string(workload.moe_placement_epoch));
            tags.emplace("decode", workload.decode ? "true" : "false");
            tags.emplace(
                "all_position_logits",
                workload.all_position_logits ? "true" : "false");
            tags.emplace(
                "live_mtp_request_batch_condition",
                workload.live_mtp_request_batch_condition ? "true" : "false");
        }

        /**
         * @brief Join one physical unit to its setup-issued executable family.
         * @param segment Captured or manual unit owned by that capture plan.
         * @param perf_context Human-readable role, not an executable identity.
         * @param workload Optional immutable geometry for timing attribution.
         * @return Diagnostic tags; no replay decision consumes this metadata.
         */
        PerfStatsCollector::Tags replaySegmentTags(const DeviceGraphExecutor::GraphSegment &segment,
                                                   const std::string &perf_context,
                                                   const DeviceGraphExecutor::GraphSegmentCache::ReplayWorkloadGeometry *workload = nullptr)
        {
            PerfStatsCollector::Tags tags{
                {"executable_family", std::to_string(segment.evidence_family)},
                {"type", segmentTypeName(segment)},
                {"stage_count", std::to_string(segment.stage_names.size())}};
            if (!segment.stage_names.empty())
            {
                tags.emplace("first_stage", segment.stage_names.front());
                tags.emplace("last_stage", segment.stage_names.back());
            }
            addContextTag(tags, perf_context);
            if (workload)
                addReplayWorkloadTags(tags, *workload);
            return tags;
        }

        /**
         * @brief Describe a complete cache without conflating sibling variants.
         * @param cache Materialized capture plan and current diagnostic geometry.
         * @return Family-qualified inventory tags for existing PerfStats rows.
         */
        PerfStatsCollector::Tags replayCacheTags(const DeviceGraphExecutor::GraphSegmentCache &cache)
        {
            size_t stage_count = 0;
            bool has_capturable = false;
            bool has_manual = false;
            for (const auto &segment : cache.segments)
            {
                stage_count += segment.stage_names.size();
                has_capturable = has_capturable || segment.capturable;
                has_manual = has_manual || !segment.capturable;
            }

            const char *type = "empty";
            if (has_capturable && has_manual)
                type = "mixed";
            else if (has_capturable)
                type = "capturable";
            else if (has_manual)
                type = "manual";

            PerfStatsCollector::Tags tags{
                {"executable_family", std::to_string(cache.evidence_family)},
                {"type", type},
                {"segment_count", std::to_string(cache.segments.size())},
                {"stage_count", std::to_string(stage_count)}};
            addContextTag(tags, cache.perf_context);
            addReplayWorkloadTags(tags, cache.replay_workload);
            return tags;
        }

        /**
         * @brief Preserve family identity while naming the native graph policy.
         * @param cache Owner of all executable units being reported.
         * @param mode Existing full-graph or heterogeneous-segment policy.
         * @return Policy-qualified inventory without changing execution state.
         */
        PerfStatsCollector::Tags replayCacheTags(const DeviceGraphExecutor::GraphSegmentCache &cache,
                                                 GraphReplayCaptureMode mode)
        {
            auto tags = replayCacheTags(cache);
            if (mode == GraphReplayCaptureMode::FullGraph)
            {
                tags.erase("segment_count");
                tags.emplace("graph_count", std::to_string(cache.segments.size()));
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayHostTimingTags(PerfStatsCollector::Tags tags,
                                                           const char *timing_scope,
                                                           GraphReplayCaptureMode mode)
        {
            tags.emplace("attribution", "host_wall");
            tags.emplace("source", captureModeSource(mode));
            tags.emplace("graph_capture_scope", captureModeHostScope(mode));
            if (timing_scope && timing_scope[0] != '\0')
            {
                tags.emplace("timing_scope", timing_scope);
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayGpuEventTags(PerfStatsCollector::Tags tags,
                                                         const char *timing_scope,
                                                         GraphReplayCaptureMode mode)
        {
            tags.emplace("attribution", "gpu_event");
            tags.emplace("source", captureModeSource(mode));
            tags.emplace("graph_capture_scope", captureModeEventScope(mode));
            if (timing_scope && timing_scope[0] != '\0')
            {
                tags.emplace("timing_scope", timing_scope);
            }
            return tags;
        }

        PerfStatsCollector::Tags graphReplayMetadataTags(PerfStatsCollector::Tags tags,
                                                         const std::string &perf_context,
                                                         GraphReplayCaptureMode mode)
        {
            tags.emplace("attribution", "graph_replay_metadata");
            tags.emplace("source", captureModeSource(mode));
            tags.emplace("graph_capture_scope", captureModePlanScope(mode));
            addContextTag(tags, perf_context);
            return tags;
        }

        constexpr size_t kInvalidReplayGpuTimingSlot =
            std::numeric_limits<size_t>::max();

        /*
         * Sixteen independent intervals provide a useful sample of a backlogged
         * replay burst without turning diagnostics into an event-per-token memory
         * commitment. When all slots are in flight, replay remains untouched and
         * the cache increments an explicit unsampled count. This is a bounded
         * profiler sampling policy, never a growth or synchronization policy.
         */
        constexpr size_t kDeferredReplayGpuTimingSlotCapacity = 16;

        void emitReplayGpuTiming(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            const DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot &slot,
            float elapsed_ms)
        {
            const GraphReplayCaptureMode replay_mode =
                captureModeForCache(segment_cache);
            const uint64_t elapsed_ns = static_cast<uint64_t>(
                static_cast<double>(std::max(0.0f, elapsed_ms)) * 1.0e6);
            const char *sync_scope = slot.deferred_final_fence
                                         ? "asynchronous_event_reclaimed"
                                         : "stream_synchronized";

            if (slot.scope ==
                DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot::Scope::TotalReplay)
            {
                auto total_tags = replayCacheTags(segment_cache, replay_mode);
                total_tags.emplace("sync_scope", sync_scope);
                PerfStatsCollector::recordTimingNs(
                    "stage_gpu",
                    "graph_replay.total",
                    elapsed_ns,
                    "decode",
                    segment_cache.replay_gpu_timing_device_name,
                    graphReplayGpuEventTags(
                        std::move(total_tags),
                        "total_replay_gpu_event",
                        replay_mode));

                /*
                 * A monolithic replay has one graph unit, so its aggregate and
                 * unit interval are exactly the same pair of stream events. Do
                 * not enqueue a duplicate pair merely to emit the second view.
                 */
                if (replay_mode == GraphReplayCaptureMode::FullGraph &&
                    segment_cache.segments.size() == 1)
                {
                    auto unit_tags = replaySegmentTags(
                        segment_cache.segments.front(),
                        segment_cache.perf_context,
                        &segment_cache.replay_workload);
                    unit_tags.emplace("graph_index", "0");
                    unit_tags.emplace("sync_scope", sync_scope);
                    PerfStatsCollector::recordTimingNs(
                        "stage_gpu",
                        replayUnitGpuEventName(replay_mode),
                        elapsed_ns,
                        "decode",
                        segment_cache.replay_gpu_timing_device_name,
                        graphReplayGpuEventTags(
                            std::move(unit_tags),
                            replayUnitTimingScope(replay_mode),
                            replay_mode));
                }
                return;
            }

            if (slot.replay_unit_index >= segment_cache.segments.size())
            {
                throw std::logic_error(
                    "Replay GPU timing slot refers to a missing replay unit");
            }

            auto unit_tags = replaySegmentTags(
                segment_cache.segments[slot.replay_unit_index],
                segment_cache.perf_context,
                &segment_cache.replay_workload);
            unit_tags.emplace(
                replayUnitIndexTagName(replay_mode),
                std::to_string(slot.replay_unit_index));
            unit_tags.emplace("sync_scope", sync_scope);
            PerfStatsCollector::recordTimingNs(
                "stage_gpu",
                replayUnitGpuEventName(replay_mode),
                elapsed_ns,
                "decode",
                segment_cache.replay_gpu_timing_device_name,
                graphReplayGpuEventTags(
                    std::move(unit_tags),
                    replayUnitTimingScope(replay_mode),
                    replay_mode));
        }

        bool reclaimReplayGpuTimingNonblockingImpl(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx)
        {
            bool reclaimed_any = false;
            for (auto &slot : segment_cache.replay_gpu_timing_slots)
            {
                if (!slot.pending)
                    continue;

                bool ready = false;
                if (!gpu_ctx->queryEventChecked(slot.stop_event, ready))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Failed to query a "
                        "pending replay GPU timing event");
                    return false;
                }
                if (!ready)
                    continue;

                const float elapsed_ms = gpu_ctx->eventElapsedTime(
                    slot.start_event,
                    slot.stop_event);
                if (elapsed_ms < 0.0f)
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Failed to read a "
                        "completed replay GPU timing interval");
                    return false;
                }

                emitReplayGpuTiming(segment_cache, slot, elapsed_ms);
                slot.started = false;
                slot.pending = false;
                reclaimed_any = true;
            }

            if (reclaimed_any &&
                segment_cache.replay_gpu_timing_busy_samples > 0)
            {
                PerfStatsCollector::addCounter(
                    "forward_graph",
                    "replay_gpu_timing_busy_samples",
                    static_cast<double>(
                        segment_cache.replay_gpu_timing_busy_samples),
                    "decode",
                    segment_cache.replay_gpu_timing_device_name,
                    {{"context", segment_cache.perf_context},
                     {"sampling_policy", "bounded_nonblocking"}});
                segment_cache.replay_gpu_timing_busy_samples = 0;
            }
            return true;
        }

        bool beginReplayGpuTiming(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx,
            DeviceGraphExecutor::GraphSegmentCache::ReplayGpuTimingSlot::Scope scope,
            size_t replay_unit_index,
            bool deferred_final_fence,
            size_t &out_slot_index)
        {
            out_slot_index = kInvalidReplayGpuTimingSlot;
            if (!PerfStatsCollector::gpuStageEventTimingEnabled())
                return true;
            if (!gpu_ctx || !segment_cache.capture_stream ||
                segment_cache.replay_gpu_timing_slots.empty())
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] GPU replay timing was "
                    "selected without its capture-owned event ring");
                return false;
            }
            if (!reclaimReplayGpuTimingNonblockingImpl(segment_cache, gpu_ctx))
                return false;

            for (size_t index = 0;
                 index < segment_cache.replay_gpu_timing_slots.size();
                 ++index)
            {
                auto &slot = segment_cache.replay_gpu_timing_slots[index];
                if (slot.started || slot.pending)
                    continue;

                slot.scope = scope;
                slot.replay_unit_index = replay_unit_index;
                slot.deferred_final_fence = deferred_final_fence;
                if (!gpu_ctx->recordEventChecked(
                        slot.start_event,
                        segment_cache.capture_stream))
                {
                    LOG_ERROR(
                        "[DeviceGraphCaptureController] Failed to publish a "
                        "replay GPU timing start event");
                    return false;
                }
                slot.started = true;
                out_slot_index = index;
                return true;
            }

            ++segment_cache.replay_gpu_timing_busy_samples;
            return true;
        }

        bool finishReplayGpuTiming(
            DeviceGraphExecutor::GraphSegmentCache &segment_cache,
            IWorkerGPUContext *gpu_ctx,
            size_t slot_index)
        {
            if (slot_index == kInvalidReplayGpuTimingSlot)
                return true;
            if (!gpu_ctx || slot_index >= segment_cache.replay_gpu_timing_slots.size())
                return false;

            auto &slot = segment_cache.replay_gpu_timing_slots[slot_index];
            if (!slot.started || slot.pending)
                return false;
            if (!gpu_ctx->recordEventChecked(
                    slot.stop_event,
                    segment_cache.capture_stream))
            {
                LOG_ERROR(
                    "[DeviceGraphCaptureController] Failed to publish a replay "
                    "GPU timing stop event");
                return false;
            }
            slot.pending = true;
            return true;
        }
    }
} // namespace llaminar2
