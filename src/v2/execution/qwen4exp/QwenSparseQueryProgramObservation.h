/**
 * @file QwenSparseQueryProgramObservation.h
 * @brief Original successful decode program observations shared by direct and composed forwards.
 *
 * Call only after a real forward succeeds. A composed owner calls after all
 * original participant terminals complete; declaration and capture emit no
 * observations. Each call counts one logical forward, never native launches.
 */
#pragma once
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "utils/PerfStatsCollector.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace llaminar2 {
/**
 * @brief Publish the original selected program of one successful decode forward.
 * @param device Actual original participant, without probing backend state.
 * @param program Immutable program belonging to the executed forward signature.
 * @param role Actual model-forward role; verifier work cannot claim main inference.
 * @param physical_rows Actual positive retained forward width.
 * @throws std::invalid_argument For an unknown program/role or invalid width/device.
 */
inline void recordQwenSparseQueryDecodeForward(DeviceId device, SparseQueryExecutionProgram program,
    ForwardExecutionRole role, std::int64_t physical_rows)
{
    if (!device.is_valid() || physical_rows <= 0 ||
        (program != SparseQueryExecutionProgram::ProvenNoRanking && program != SparseQueryExecutionProgram::RankingPossible) ||
        (role != ForwardExecutionRole::MainInference && role != ForwardExecutionRole::GroupedMTPVerifier &&
            role != ForwardExecutionRole::MTPCondition))
        throw std::invalid_argument("Sparse query decode observation requires its original device/program/role/width");
    PerfStatsCollector::addCounter("qsa", "query_program_forward_submissions", 1.0,
        "decode", device.toString(),
        {{"program", program == SparseQueryExecutionProgram::ProvenNoRanking ? "proven_no_ranking" : "ranking_possible"},
         {"role", std::to_string(static_cast<int>(role))}, {"physical_rows", std::to_string(physical_rows)},
         {"observation", "successful_forward_submission_not_native_replay_count"}});
}
} // namespace llaminar2
