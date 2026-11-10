/**
 * @file OpenMPTeamForkEvidence.h
 * @brief The two production publishers of the OpenMP team-fork counter.
 *
 * The counter authority (OpenMPUtils.h) counts every real fork; this header
 * owns how that count becomes structured evidence, so both publication
 * surfaces share one tested implementation:
 *
 * - `openmp_team_forks_decode_step`: the exact fork/join count of one
 *   cached-forward CPU decode step. The engine publishes the delta of its
 *   execution-window scope under the broad PerfStats collection policy and
 *   the record-family selection, so default serving materializes nothing
 *   while the one-atomic cost remains.
 * - `openmp_team_forks_stage`: one record per CPU graph stage, published by
 *   the graph executor's stage loops under the explicit CPU stage-timing
 *   gate. Stage timing serializes waves onto one executor thread, and the
 *   stage scope is that thread's innermost owner, so the cell delta
 *   attributes the stage's forks exactly; GPU entries execute no OpenMP fork
 *   and publish no fork record.
 *
 * Both publishers are ordinary functions: tests exercise them directly for
 * the off-default, on, filtered, GPU-absent and exact-value contracts, and
 * the engine/executor call sites prove the actual path end to end.
 */
#pragma once

#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"

#include "../backends/DeviceId.h"

#include <string>

namespace llaminar::v2
{
/** @brief The summary counter family published once per cached-forward CPU decode step. */
inline constexpr const char *kOpenMPTeamForkSummaryFamily = "openmp_team_forks_decode_step";
/** @brief The per-stage counter family published by the graph executor's CPU stage loops. */
inline constexpr const char *kOpenMPTeamForkStageFamily = "openmp_team_forks_stage";

/**
 * @brief Publish one cached-forward CPU decode step's team-fork summary.
 * @param entry_forks The snapshot taken when the execution window opened.
 * @param device Identifying device string of the executing rank/device.
 * @param perf_context The forward-graph context tag of the executed signature.
 * @param used_graph_replay Whether the step ran as captured-graph replay.
 * @details Silently does nothing unless collection policy selects the family:
 * the fork sites count regardless, so an off-default process carries only the
 * one-atomic cost. Callers publish on success only; a failed step publishes
 * no count rather than a partial one.
 */
inline void publishOpenMPTeamForkSummary(std::uint64_t entry_forks,
    llaminar2::DeviceId device, const std::string &perf_context, bool used_graph_replay)
{
    // A GPU device step executes no OpenMP fork: publishing its "summary"
    // would claim CPU wake-up structure the device never had. The refusal is
    // the publisher's own contract, so no call-site mistake can fabricate a
    // GPU decode step's fork count.
    if (!device.is_cpu() ||
        !llaminar2::PerfStatsCollector::isRecordFamilyEnabled("omp", kOpenMPTeamForkSummaryFamily))
        return;
    llaminar2::PerfStatsCollector::addCounter(
        "omp", kOpenMPTeamForkSummaryFamily,
        static_cast<double>(openMPTeamForkSnapshot() - entry_forks),
        "decode", device.toString(),
        {{"context", perf_context},
         {"used_graph_replay", used_graph_replay ? "true" : "false"}});
}

/**
 * @brief Publish one CPU graph stage's team-fork count from its owner scope.
 * @param scope The stage's @ref OpenMPTeamForkScope, constructed before the
 *              stage executed on the executor thread.
 * @param stage_name Exact graph node name of the executed stage.
 * @param phase Structured execution-phase name at publication time.
 * @param device Identifying device string of the executing device context.
 * @param index Stable schedule index of the executed entry.
 * @param stage_type Computed stage-type name of the executed node.
 */
inline void publishOpenMPTeamForkStage(const OpenMPTeamForkScope &scope,
    const std::string &stage_name, const std::string &phase, llaminar2::DeviceId device,
    size_t index, const std::string &stage_type)
{
    // GPU entries execute no OpenMP fork: a fork record for one would claim
    // wake-up structure the device never had. The refusal is the publisher's
    // own contract, so a misattributed call site cannot fabricate one.
    if (!device.is_cpu() ||
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
