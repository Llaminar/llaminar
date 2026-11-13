/**
 * @file OrdinaryTerminalProjectionCapacity.h
 * @brief Immutable ordinary terminal-row demand within existing graph admission.
 *
 * Selected-final-row execution retains one output row per admitted request.
 * An explicit diagnostic may declare a larger flattened output-row maximum
 * before admission. This value owns no bytes, live count or allocator: the
 * existing batch/resident-activation envelope bounds it, and PMA prices the
 * native transpose through VocabularyGatherWorkspaceContract.
 */
#pragma once
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
/** @brief Checked ordinary projection demand; request reset never mutates it. */
class OrdinaryTerminalProjectionCapacity final
{
public:
    /** @brief Select the ordinary one-final-row-per-request family. */
    constexpr OrdinaryTerminalProjectionCapacity() noexcept = default;
    /** @return Default demand, independent of prompt/KV or MTP capacity. */
    [[nodiscard]] static constexpr OrdinaryTerminalProjectionCapacity selectedFinalRowsPerRequest() noexcept
    { return {}; }
    /**
     * @brief Declare the maximum actual flattened diagnostic output rows.
     * @param rows Positive maximum, already including every request.
     * @return Immutable demand validated against physical graph bounds later.
     * @throws std::invalid_argument For a nonpositive declaration.
     */
    [[nodiscard]] static OrdinaryTerminalProjectionCapacity projectedRows(int rows)
    {
        if (rows <= 0) throw std::invalid_argument("Ordinary terminal projection rows must be positive");
        return OrdinaryTerminalProjectionCapacity(rows);
    }
    /** @return Whether setup authored a diagnostic output-row maximum. */
    [[nodiscard]] constexpr bool hasExplicitRows() const noexcept { return rows_ > 0; }
    /**
     * @brief Resolve demand against the already admitted flattened activation bound.
     * @param requests Positive admitted request population.
     * @param activation_rows Positive admitted flattened activation rows.
     * @return Maximum selected final rows or explicitly authored output rows.
     * @throws std::invalid_argument For missing/inconsistent/oversized geometry.
     */
    [[nodiscard]] int resolve(int requests, int activation_rows) const
    {
        if (requests <= 0 || activation_rows < requests)
            throw std::invalid_argument("Ordinary terminal projection requires an admitted positive request/activation envelope");
        if (rows_ > activation_rows)
            throw std::invalid_argument("Ordinary terminal projection exceeds admitted activation rows");
        return std::max(requests, rows_);
    }
    /**
     * @brief Resolve from the planner's existing per-request resident row axes.
     * @param requests Positive admitted request population.
     * @param sequence_rows Positive selected resident rows per request, not KV context.
     * @return Maximum flattened terminal output rows.
     * @throws std::invalid_argument For absent geometry or oversized demand.
     * @throws std::overflow_error For an unrepresentable flattened row bound.
     */
    [[nodiscard]] int resolveForSequence(int requests, int sequence_rows) const
    {
        return resolve(requests, checkedFlattenedRows(requests, sequence_rows));
    }
    /**
     * @brief Check flattened input rows before any narrow arithmetic or enqueue.
     * @param requests Positive request population.
     * @param sequence_rows Positive physical rows per request.
     * @return Exact representable request-by-row product.
     * @throws std::invalid_argument For nonpositive geometry.
     * @throws std::overflow_error For an unrepresentable product.
     */
    [[nodiscard]] static int checkedFlattenedRows(int requests, int sequence_rows)
    {
        if (requests <= 0 || sequence_rows <= 0)
            throw std::invalid_argument("Ordinary terminal projection requires positive request/resident rows");
        if (sequence_rows > std::numeric_limits<int>::max() / requests)
            throw std::overflow_error("Ordinary terminal activation row bound overflows int");
        return requests * sequence_rows;
    }
    /**
     * @brief Check one actual publication against ordinary and retained-MTP demand.
     * @param rows Positive projected operation rows; never tensor capacity.
     * @param requests Positive admitted request population.
     * @param activation_rows Admitted flattened ordinary activation bound.
     * @param retained_mtp_rows Existing retained verifier maximum, zero when absent.
     * @throws std::invalid_argument For invalid or unadmitted publication geometry.
     *
     * MainInference can legitimately execute an MTP parity verifier. Retained
     * capacity, not the transient execution role or enabled flag, owns that union.
     */
    void requirePublication(int rows, int requests, int activation_rows, int retained_mtp_rows) const
    {
        if (rows <= 0 || retained_mtp_rows < 0 || rows > std::max(resolve(requests, activation_rows), retained_mtp_rows))
            throw std::invalid_argument("Native ordinary terminal publication exceeds admitted projected-row capacity");
    }
    friend constexpr bool operator==(const OrdinaryTerminalProjectionCapacity &, const OrdinaryTerminalProjectionCapacity &) noexcept = default;
private:
    /** @brief Only the checked diagnostic factory may author explicit rows. */
    explicit constexpr OrdinaryTerminalProjectionCapacity(int rows) noexcept : rows_(rows) {}
    int rows_ = 0; ///< Zero denotes selected-final-row policy, never an empty operation.
};
} // namespace llaminar2
