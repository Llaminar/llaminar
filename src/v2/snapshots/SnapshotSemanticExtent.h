/**
 * @file SnapshotSemanticExtent.h
 * @brief Explicit authored extents for point-in-time diagnostic tensor observations.
 *
 * Token matrices, persistent state prefixes and counted index rows have different
 * mathematical axes. Their native metadata is captured beside the payload and
 * interpreted only by the diagnostic observer. No count is returned to model
 * execution and no diagnostic rule changes production padding or arithmetic.
 */
#pragma once
#include "SnapshotContiguousRows.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <variant>

namespace llaminar2
{
class ITensor;
/** @brief Borrowed INT32 authority; its exact tensor, subview and lifetime are capture identity. */
struct SnapshotInt32Binding
{
    ITensor *tensor=nullptr;
    std::size_t byteOffset=0;
    bool operator==(const SnapshotInt32Binding &) const=default;
};
/** @brief One token row per live count; an absent count explicitly selects all admitted rows. */
struct SnapshotTokenRows
{
    SnapshotInt32Binding liveRows;
    bool operator==(const SnapshotTokenRows &) const=default;
};
/** @brief Latest complete state prefix, never a sequence of token chunks. */
struct SnapshotStatePrefixRows
{
    SnapshotInt32Binding frontier; ///< Two adjacent native words: prior rows, then produced live rows.
    int sourceCapacity=0; ///< Maximum prior+produced source rows admitted by the execution owner.
    int sourceRowsPerStateRow=1; ///< Number of source rows completing one state row (four for QSA).
    bool operator==(const SnapshotStatePrefixRows &) const=default;
};
/** @brief Authored index prefixes; only diagnostic host reconstruction creates trailing -1 cells. */
struct SnapshotCountedSelectionRows
{
    SnapshotInt32Binding liveRows; ///< Optional explicit all-active policy when absent.
    SnapshotInt32Binding counts; ///< One native authored-column count per physical token row.
    bool operator==(const SnapshotCountedSelectionRows &) const=default;
};
/** @brief Legacy inference, explicit tokens, latest state or counted index rows. */
using SnapshotSemanticExtent=std::variant<std::monostate,SnapshotTokenRows,
    SnapshotStatePrefixRows,SnapshotCountedSelectionRows>;

/** @brief Required point-in-time metadata written by the diagnostic copy kernel. */
struct alignas(8) SnapshotObservedHeader
{
    std::int32_t priorRows=0;
    std::int32_t liveRows=0;
    std::uint64_t payloadBytes=0; ///< Actual authored bytes copied, excluding retained capacity.
};
static_assert(sizeof(SnapshotObservedHeader)==16);

/** @brief Checked mathematical dimensions reconstructed from acquired native metadata. */
struct SnapshotObservedShape
{
    std::size_t rows=0;
    std::size_t payloadBytes=0;
    std::size_t offsetBytes=0; ///< Required packed-row offsets, zero for ordinary/state prefixes.
};

/** @brief Validate an observed header without touching a live execution pointer.
 * @param extent Frozen mathematical axis and native count authority.
 * @param header Already acquired point-in-time native metadata.
 * @param capacity Admitted output rows, never an assumed live extent.
 * @param columns Exact logical columns per row.
 * @param elementBytes Native scalar bytes (four or eight).
 * @return Live shape and exact next observation extents; counted offsets require separate validation.
 */
inline SnapshotObservedShape validateSnapshotObservedHeader(const SnapshotSemanticExtent &extent,
    const SnapshotObservedHeader &header,std::size_t capacity,std::size_t columns,std::size_t elementBytes)
{
    if(!capacity||!columns||(elementBytes!=4&&elementBytes!=8)||
       columns>std::numeric_limits<std::size_t>::max()/elementBytes||
       capacity>std::numeric_limits<std::size_t>::max()/(columns*elementBytes)||header.liveRows<0||header.priorRows<0)
        throw std::invalid_argument("Diagnostic observation has invalid admitted geometry or negative native counts");
    const auto rowBytes=columns*elementBytes;
    std::size_t rows=static_cast<std::size_t>(header.liveRows),offsetBytes=0;
    if(const auto *state=std::get_if<SnapshotStatePrefixRows>(&extent))
    {
        const auto observed=resolveSnapshotContiguousRows(
            {SnapshotContiguousRowKind::StatePrefixRows,capacity,state->sourceCapacity,state->sourceRowsPerStateRow},
            header.priorRows,header.liveRows);
        if(!observed.valid)
            throw std::logic_error("Diagnostic state prefix exceeds its native source frontier");
        rows=observed.rows;
    }
    else
    {
        if(header.priorRows||std::holds_alternative<std::monostate>(extent))
            throw std::logic_error("Diagnostic token/count header does not match its frozen mathematical axis");
        if(std::holds_alternative<SnapshotCountedSelectionRows>(extent))
        {
            if(rows==std::numeric_limits<std::size_t>::max()||rows+1>std::numeric_limits<std::size_t>::max()/8)
                throw std::overflow_error("Diagnostic counted-row metadata overflows observation storage");
            offsetBytes=(rows+1)*sizeof(std::uint64_t);
        }
    }
    if(rows>capacity||header.payloadBytes>rows*rowBytes||
       (!offsetBytes&&header.payloadBytes!=rows*rowBytes)||header.payloadBytes%elementBytes)
        throw std::logic_error("Diagnostic payload bytes disagree with their exact native live extent");
    return {rows,static_cast<std::size_t>(header.payloadBytes),offsetBytes};
}

/** @brief Authenticate captured packed offsets and every native row count before reading payload.
 * @param offsets Acquired exclusive offsets, one more than the observed live token rows.
 * @param shape Header-derived exact live rows and payload byte count.
 * @param columns Admitted per-row selected-index capacity.
 * @param elementBytes Exact scalar width of the selected indices.
 */
inline void validateSnapshotSelectionOffsets(std::span<const std::uint64_t> offsets,
    SnapshotObservedShape shape,std::size_t columns,std::size_t elementBytes)
{
    if(offsets.size()!=shape.rows+1||offsets.empty()||offsets.front()!=0||!shape.offsetBytes||
       (elementBytes!=4&&elementBytes!=8))
        throw std::logic_error("Diagnostic selection requires its complete captured row-offset metadata");
    for(std::size_t row=0;row<shape.rows;++row)
        if(offsets[row+1]<offsets[row]||offsets[row+1]-offsets[row]>columns)
            throw std::logic_error("Diagnostic selected count exceeds its authored native row prefix");
    if(offsets.back()>std::numeric_limits<std::uint64_t>::max()/elementBytes||
       offsets.back()*elementBytes!=shape.payloadBytes)
        throw std::logic_error("Diagnostic selected payload disagrees with captured native row counts");
}
} // namespace llaminar2
