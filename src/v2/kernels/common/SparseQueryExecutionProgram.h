/**
 * @file SparseQueryExecutionProgram.h
 * @brief Immutable sparse-query topology selected before retained graph execution.
 */
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace llaminar2
{
/** @brief Closed query program; no live counter or runtime fallback is represented. */
enum class SparseQueryExecutionProgram
{
    RankingPossible,
    ProvenNoRanking
};
/**
 * @brief Immutable prefill range admitted from original request I/O, never a live KV mirror.
 * @details A request proof can project only contained scheduler chunks. The
 * separate cold declaration is legal solely for unlaunched materialization;
 * execution owners reject it before any cache lookup or native submission.
 * Numeric ranges are request data and are intentionally absent from graph keys.
 */
class SparseQueryPrefillProof final
{
public:
    /**
     * @brief Retain the original accepted contiguous logical token range.
     * @param begin Existing admitted request's nonnegative logical prefix.
     * @param rows Existing accepted real input count, excluding physical padding.
     * @return Immutable range proof with checked representable exclusive end.
     * @throws std::invalid_argument For invalid/overflowing original geometry.
     */
    [[nodiscard]] static SparseQueryPrefillProof admittedRange(int begin, int rows)
    {
        if (begin < 0 || rows <= 0 || std::int64_t(begin) + rows > std::numeric_limits<int>::max())
            throw std::invalid_argument("Sparse prefill proof requires a complete admitted logical range");
        return SparseQueryPrefillProof(begin, begin + rows);
    }
    /**
     * @brief Declare one existing retained program without fabricating a live request.
     * @param program Closed program selected by the original cold compilation scope.
     * @return Cold-only proof; never legal for Execute submission.
     * @throws std::invalid_argument For an unknown retained program.
     */
    [[nodiscard]] static SparseQueryPrefillProof materialization(SparseQueryExecutionProgram program)
    {
        if (program != SparseQueryExecutionProgram::RankingPossible && program != SparseQueryExecutionProgram::ProvenNoRanking)
            throw std::invalid_argument("Sparse prefill materialization requires a known retained program");
        SparseQueryPrefillProof proof(0, 0);
        proof.materialization_ = true;
        proof.cold_program_ = program;
        return proof;
    }
    /**
     * @brief Project an original immutable scheduler chunk, without reading device state.
     * @param begin Chunk's absolute logical start from the admitted schedule.
     * @param rows Chunk's real count, excluding physical bucket padding.
     * @return Narrowed request proof, or the unchanged cold declaration.
     * @throws std::invalid_argument For a foreign, empty, or overflowing range.
     */
    [[nodiscard]] SparseQueryPrefillProof chunk(int begin, int rows) const
    {
        if (materialization_) return *this;
        auto projected = admittedRange(begin, rows);
        if (projected.begin_ < begin_ || projected.end_ > end_)
            throw std::invalid_argument("Sparse prefill chunk is outside its original admitted request");
        return projected;
    }
    /** @return Exact original admission metadata equality, independent of graph/cache identity. */
    [[nodiscard]] bool operator==(const SparseQueryPrefillProof &) const noexcept = default;
    /** @return Whether this proof names cold topology rather than an executable request. */
    [[nodiscard]] bool isMaterialization() const noexcept { return materialization_; }
    /**
     * @brief Select the complete supported program from original admitted visibility.
     * @param maximum The kernel contract's canonical maximum no-ranking visible count.
     * @return The exact retained program; cold declarations preserve their selected program.
     * @throws std::invalid_argument For a malformed canonical bound.
     */
    [[nodiscard]] SparseQueryExecutionProgram program(int maximum) const
    {
        if (maximum <= 0) throw std::invalid_argument("Sparse prefill proof requires the canonical visible bound");
        return materialization_ ? cold_program_ : end_ <= maximum
            ? SparseQueryExecutionProgram::ProvenNoRanking : SparseQueryExecutionProgram::RankingPossible;
    }
    /** @return Whether this executable proof names exactly the submitted logical range. */
    [[nodiscard]] bool namesRange(int begin, int rows) const noexcept
    {
        return !materialization_ && begin >= 0 && rows > 0 && begin == begin_ && std::int64_t(begin) + rows == end_;
    }
private:
    /** @brief Construct only through checked admission or explicit cold declaration. */
    SparseQueryPrefillProof(int begin, int end) : begin_(begin), end_(end) {}
    int begin_, end_; ///< Original immutable logical I/O extent, never a current device cursor.
    bool materialization_ = false; ///< Closed source distinction, checked against submission intent.
    SparseQueryExecutionProgram cold_program_ = SparseQueryExecutionProgram::RankingPossible;
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
