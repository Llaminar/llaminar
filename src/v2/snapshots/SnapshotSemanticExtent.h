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
#include "SnapshotAuthoredTokenRowsObservation.h"
#include "../kernels/common/ProjectionRowWork.h"
#include "../backends/DeviceId.h"
#include <vector>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <variant>

namespace llaminar2
{
class ITensor;
class ProjectionRowPublicationTable;
/** @brief Borrowed INT32 authority; its exact tensor, subview and lifetime are capture identity. */
struct SnapshotInt32Binding
{
    const ITensor *tensor=nullptr;
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
/**
 * @brief Borrow an existing canonical native count owner without inventing a tensor.
 *
 * The original controller/input owns the pointer and publication lifetime. The
 * diagnostic freezes device, entire native owner capacity and scalar subview;
 * it never dereferences this borrow separately from ProjectionRowWork::observe*.
 */
struct SnapshotNativeInt32Borrow
{
    DeviceId device=DeviceId::invalid();
    const std::int32_t *data=nullptr;
    std::size_t capacity=0,wordOffset=0;
    bool operator==(const SnapshotNativeInt32Borrow &) const=default;
};
/** @brief Optional inherited agreement authority: original tensor or canonical native borrow. */
using SnapshotInheritedInt32Binding=std::variant<std::monostate,SnapshotInt32Binding,SnapshotNativeInt32Borrow>;
/** @brief Original tensor/subview authorities for one request publication. */
struct SnapshotProjectionPublicationBinding
{
    SnapshotInt32Binding frontier,live;
    SnapshotInheritedInt32Binding inheritedLive;
    bool operator==(const SnapshotProjectionPublicationBinding &) const=default;
};
/**
 * @brief Borrow the exact prepared projection work and its original scalar owners.
 *
 * The stage retains and seals any immutable indirect table through its existing
 * SerialGraphFamily workspace publication. The work's table address/generation
 * belongs to capture identity; this descriptor is not another table or live ledger.
 * Primary frontier/live words name original tensor/subviews; inherited agreement
 * may borrow the existing controller/input native owner. These borrows remain
 * valid until the captured diagnostic transaction and terminal observer retire.
 */
struct SnapshotAuthoredTokenRows
{
    ProjectionRowWork work;
    std::vector<SnapshotProjectionPublicationBinding> publications; ///< Only original requests intersecting this source slice, in order.
    std::shared_ptr<const ProjectionRowPublicationTable> publicationTable; ///< Original immutable workspace publication; absent for strided work.
    /** @brief Compare complete immutable owner/publication/geometry identity, excluding live contents. */
    bool operator==(const SnapshotAuthoredTokenRows &other) const {
        return work.captureIdentity()==other.work.captureIdentity()&&publications==other.publications&&publicationTable==other.publicationTable;
    }
};
/** @return First original request intersecting the immutable source slice. */
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline int snapshotProjectionFirstRequest(ProjectionRowWork work) noexcept {return work.requestForRow(0);}
/** @return Required original-request records; unrelated requests do not communicate metadata. */
#if defined(__CUDACC__) || defined(__HIPCC__)
__host__ __device__
#endif
inline int snapshotProjectionRequestCount(ProjectionRowWork work) noexcept {
    return work.requestForRow(work.physicalRows()-1)-work.requestForRow(0)+1;
}
/** @brief Legacy inference, tokens, state, counted indices or authored request intervals. */
using SnapshotSemanticExtent=std::variant<std::monostate,SnapshotTokenRows,
    SnapshotStatePrefixRows,SnapshotCountedSelectionRows,SnapshotAuthoredTokenRows>;

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
    std::size_t offsetBytes=0; ///< Required offsets/request records, zero for ordinary/state prefixes.
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
        if(const auto *authored=std::get_if<SnapshotAuthoredTokenRows>(&extent))
        {
            if(capacity>static_cast<std::size_t>(INT32_MAX)||authored->work.physicalRows()!=static_cast<int>(capacity)||
               static_cast<std::size_t>(snapshotProjectionRequestCount(authored->work))>(std::numeric_limits<std::size_t>::max()-sizeof(SnapshotObservedHeader))/sizeof(SnapshotAuthoredTokenRowRecord))
                throw std::logic_error("Diagnostic authored work disagrees with its admitted source slice");
            offsetBytes=static_cast<std::size_t>(snapshotProjectionRequestCount(authored->work))*sizeof(SnapshotAuthoredTokenRowRecord);
        }
        else if(std::holds_alternative<SnapshotCountedSelectionRows>(extent))
        {
            if(rows==std::numeric_limits<std::size_t>::max()||rows+1>std::numeric_limits<std::size_t>::max()/8)
                throw std::overflow_error("Diagnostic counted-row metadata overflows observation storage");
            offsetBytes=(rows+1)*sizeof(std::uint64_t);
        }
    }
    if(rows>capacity||header.payloadBytes>rows*rowBytes||
       (!std::holds_alternative<SnapshotCountedSelectionRows>(extent)&&header.payloadBytes!=rows*rowBytes)||header.payloadBytes%elementBytes)
        throw std::logic_error("Diagnostic payload bytes disagree with their exact native live extent");
    return {rows,static_cast<std::size_t>(header.payloadBytes),offsetBytes};
}

/**
 * @brief Authenticate every acquired request interval without rereading live words.
 * @param extent Frozen original work, independent of mutable publication contents.
 * @param records Complete point-in-time native records, including empty requests.
 * @param shape Already checked total payload shape.
 * @return Immutable provenance retained beside the exact packed payload.
 * @throws std::logic_error For missing records, wrong coordinates/reasons or invented bytes.
 * @details Canonical numerical comparison separately checks original model history;
 * this structural check cannot certify the producer's own threshold as model truth.
 */
inline SnapshotAuthoredTokenRowsObservation validateSnapshotAuthoredTokenRows(
    const SnapshotAuthoredTokenRows &extent,std::span<const SnapshotAuthoredTokenRowRecord> records,
    SnapshotObservedShape shape)
{
    const auto id=extent.work.captureIdentity();
    if(records.size()!=static_cast<std::size_t>(snapshotProjectionRequestCount(extent.work))||id.source_capacity<=0)
        throw std::logic_error("Authored query observation lacks complete original request records");
    std::uint64_t rows=0;
    for(std::size_t record=0;record<records.size();++record)
    {
        const int request=snapshotProjectionFirstRequest(extent.work)+static_cast<int>(record);
        const auto &r=records[record];
        const auto span=extent.work.resolveFor(request,r.liveRows,r.prefixRows);
        const auto noRanking=r.liveRows==0||static_cast<std::int64_t>(r.prefixRows)+r.liveRows<=id.first_required_logical_row;
        if(!span.valid()||r.rowOffset!=request*id.logical_rows*id.repeats_per_row-id.first_physical_row||
           r.rowCapacity!=id.logical_rows||r.sourceCapacity!=id.source_capacity||
           r.firstRequiredLogicalRow!=id.first_required_logical_row||r.firstRow!=span.first||r.endRow!=span.end||
           r.payloadRowOffset!=rows||r.reserved||
           r.emptyReason!=(noRanking?SnapshotEmptyReason::RankingNotRequired:SnapshotEmptyReason::None))
            throw std::logic_error("Authored query metadata contradicts original request coordinates");
        rows+=static_cast<unsigned>(span.size());
    }
    if(rows!=shape.rows)
        throw std::logic_error("Authored query payload invents or omits required interval rows");
    return {{records.begin(),records.end()},static_cast<std::size_t>(id.physical_rows),
        id.repeats_per_row,id.first_physical_row};
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
