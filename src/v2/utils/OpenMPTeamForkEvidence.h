/**
 * @file OpenMPTeamForkEvidence.h
 * @brief The two production publishers of the OpenMP team-fork counter.
 *
 * The counter authority (OpenMPUtils.h) counts every real fork; this header
 * owns how that count becomes structured evidence, so both publication
 * surfaces share one tested implementation:
 *
 * - `openmp_team_forks_decode_step`: the fork/join count of one cached-forward
 *   CPU decode step's owner scope. The engine passes the scope it constructed
 *   around the execution window; the publisher publishes the scope's own cell
 *   count directly - never a global-delta arithmetic, which other concurrent
 *   owners would corrupt. The graph-context string is materialized only after
 *   the gate accepts: the publisher owns every materialization, so an
 *   off-default process allocates nothing.
 * - `openmp_team_forks_stage`: one record per CPU graph stage, published by
 *   the graph executor's stage loops. The publisher requires the explicit CPU
 *   stage-timing gate AND the record-family selection AND a CPU device: a GPU
 *   entry, an ungated caller or an unselected export can never fabricate a
 *   fork record.
 *
 * Both publishers are ordinary functions: tests exercise them directly for
 * the off-default, on, filtered, GPU-refusal and exact-value contracts, and
 * the engine/executor call sites prove the actual path end to end.
 */
#pragma once

#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"

#include "../backends/DeviceId.h"

#include <functional>
#include <string>

namespace llaminar::v2
{
/** @brief The summary counter family published once per cached-forward CPU decode step. */
inline constexpr const char *kOpenMPTeamForkSummaryFamily = "openmp_team_forks_decode_step";
/** @brief The per-stage counter family published by the graph executor's CPU stage loops. */
inline constexpr const char *kOpenMPTeamForkStageFamily = "openmp_team_forks_stage";

/**
 * @brief Publish one cached-forward CPU decode step's team-fork summary.
 * @param scope The owner scope the engine constructed around the execution window.
 * @param device The executing device; a non-CPU device is refused by contract.
 * @param perf_context Lazy graph-context builder, invoked only after the gate
 *                    accepts, so the off-default path materializes nothing.
 * @param used_graph_replay Whether the step ran as captured-graph replay.
 * @details The published value is the scope's own cell count: concurrent
 * owners, prior totals and other threads' forks cannot move it. A GPU device
 * step executes no OpenMP fork, and publishing one would claim wake-up
 * structure the device never had. Callers publish on success only; a failed
 * step publishes no count rather than a partial one.
 */
inline void publishOpenMPTeamForkSummary(const OpenMPTeamForkScope &scope,
    llaminar2::DeviceId device, const std::function<std::string()> &perf_context,
    bool used_graph_replay)
{
    if (!device.is_cpu() ||
        !llaminar2::PerfStatsCollector::isRecordFamilyEnabled("omp", kOpenMPTeamForkSummaryFamily))
        return;
    llaminar2::PerfStatsCollector::addCounter(
        "omp", kOpenMPTeamForkSummaryFamily,
        static_cast<double>(scope.forks()),
        "decode", device.toString(),
        {{"context", perf_context()},
         {"used_graph_replay", used_graph_replay ? "true" : "false"}});
}

/**
 * @brief Publish one CPU graph stage's team-fork count from its owner scope.
 * @param scope The stage's @ref OpenMPTeamForkScope, constructed before the
 *              stage executed on the executor thread.
 * @param stage_name Exact graph node name of the executed stage.
 * @param phase Structured execution-phase name at publication time.
 * @param device The executing device; a non-CPU device is refused by contract.
 * @param index Stable schedule index of the executed entry.
 * @param stage_type Computed stage-type name of the executed node.
 * @details The record belongs to the explicit CPU stage-timing family: the
 * publisher refuses to emit one when that gate is off, even for a caller that
 * constructed a scope, so the stage evidence can never appear without the
 * instrumentation whose cost it describes.
 */
inline void publishOpenMPTeamForkStage(const OpenMPTeamForkScope &scope,
    const std::string &stage_name, const std::string &phase, llaminar2::DeviceId device,
    size_t index, const std::string &stage_type)
{
    if (!device.is_cpu() ||
        !llaminar2::PerfStatsCollector::cpuStageTimingEnabled() ||
        !llaminar2::PerfStatsCollector::isRecordFamilyEnabled("omp", kOpenMPTeamForkStageFamily))
        return;
    llaminar2::PerfStatsCollector::addCounter(
        "omp", kOpenMPTeamForkStageFamily,
        static_cast<double>(scope.forks()),
        phase, device.toString(),
        {{"attribution", "retained_or_standalone_fork_count"},
         {"source", "device_graph_executor"},
         {"stage", stage_name},
         {"stage_type", stage_type},
         {"index", std::to_string(index)}});
}

} // namespace llaminar::v2
