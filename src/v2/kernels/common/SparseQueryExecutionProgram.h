/**
 * @file SparseQueryExecutionProgram.h
 * @brief Immutable sparse-query topology selected before retained graph execution.
 */
#pragma once
#include <cstddef>
#include <stdexcept>
namespace llaminar2
{
/** @brief Closed query program; no live counter or runtime fallback is represented. */
enum class SparseQueryExecutionProgram
{
    RankingPossible,
    ProvenNoRanking
};
/** @brief Number of explicitly retained sparse-query programs. */
inline constexpr std::size_t kSparseQueryExecutionProgramCount = 2;
/**
 * @brief Map a closed program to its retained owner slot.
 * @param program Immutable admitted query topology.
 * @return Stable RankingPossible/ProvenNoRanking slot, respectively zero/one.
 * @throws std::invalid_argument For an unknown program; never silently aliases a slot.
 */
[[nodiscard]] constexpr std::size_t sparseQueryExecutionProgramIndex(
    SparseQueryExecutionProgram program)
{
    switch (program)
    {
    case SparseQueryExecutionProgram::RankingPossible: return 0;
    case SparseQueryExecutionProgram::ProvenNoRanking: return 1;
    }
    throw std::invalid_argument("Unknown sparse-query execution program");
}
}
