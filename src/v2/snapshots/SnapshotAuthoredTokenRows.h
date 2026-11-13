/**
 * @file SnapshotAuthoredTokenRows.h
 * @brief Metadata-only conservation and replica agreement for acquired row intervals.
 *
 * These diagnostic checks authenticate packed offsets and original coordinates.
 * They read no payload or live execution state and do not determine canonical
 * query demand; parity separately supplies independently authenticated history.
 */
#pragma once
#include "SnapshotInfo.h"
#include <algorithm>
#include <limits>
#include <span>
#include <stdexcept>

namespace llaminar2
{
/**
 * @brief Check complete acquired records against their frozen source and packed geometry.
 * @param snapshot Owned immutable acquired diagnostic publication.
 * @return Whether every original record, packed offset, type and payload extent agrees.
 * @details A request clipped outside a source slice may be physically empty while
 * still requiring work elsewhere; only an originally empty logical demand has
 * RankingNotRequired. This rule never treats allocation capacity as payload.
 */
inline bool validSnapshotAuthoredTokenRows(const SnapshotInfo &snapshot) noexcept
{
    if(!snapshot || !snapshot.lifetime_owner || snapshot.isInteger() ||
       snapshot.semantic_axis!=SnapshotSemanticAxis::AuthoredTokenRows || !snapshot.authored_token_rows ||
       !std::holds_alternative<std::span<const float>>(snapshot.typed_values) || !snapshot.cols)
        return false;
    const auto &observation=*snapshot.authored_token_rows;
    constexpr auto maximum=std::numeric_limits<std::int32_t>::max();
    if(observation.requests.empty() || observation.repeatsPerRow<=0 || observation.sliceFirstRow<0 ||
       !observation.physicalRows || observation.physicalRows>static_cast<std::size_t>(maximum))return false;
    const auto capacity=observation.requests.front().rowCapacity;
    if(capacity<=0 || capacity>maximum/observation.repeatsPerRow ||
       observation.physicalRows>static_cast<std::size_t>(maximum-observation.sliceFirstRow))return false;
    const auto stride=static_cast<std::int64_t>(capacity)*observation.repeatsPerRow;
    const auto firstRequest=observation.sliceFirstRow/stride;
    const auto lastRequest=(observation.sliceFirstRow+static_cast<std::int64_t>(observation.physicalRows)-1)/stride;
    if(observation.requests.size()!=static_cast<std::size_t>(lastRequest-firstRequest+1))return false;
    std::uint64_t packed=0;
    for(std::size_t request=0;request<observation.requests.size();++request)
    {
        const auto &record=observation.requests[request];
        if(record.rowCapacity!=capacity || record.prefixRows<0 || record.liveRows<0 ||
           record.liveRows>capacity || record.sourceCapacity<capacity || record.firstRequiredLogicalRow<0 ||
           static_cast<std::int64_t>(record.prefixRows)+record.liveRows>record.sourceCapacity || record.reserved)
            return false;
        const auto base=(firstRequest+static_cast<std::int64_t>(request))*stride-observation.sliceFirstRow;
        if(base!=record.rowOffset)return false;
        const auto logicalFirst=std::clamp(static_cast<std::int64_t>(record.firstRequiredLogicalRow)-record.prefixRows,
            std::int64_t{0},static_cast<std::int64_t>(record.liveRows));
        const auto end=std::clamp(base+static_cast<std::int64_t>(record.liveRows)*observation.repeatsPerRow,
            std::int64_t{0},static_cast<std::int64_t>(observation.physicalRows));
        const auto first=std::clamp(base+logicalFirst*observation.repeatsPerRow,std::int64_t{0},end);
        const auto reason=logicalFirst==record.liveRows?SnapshotEmptyReason::RankingNotRequired:SnapshotEmptyReason::None;
        if(record.firstRow!=first || record.endRow!=end || record.emptyReason!=reason ||
           record.payloadRowOffset!=packed)return false;
        packed+=static_cast<std::uint64_t>(end-first);
    }
    if(packed>observation.physicalRows || snapshot.rows!=packed ||
       snapshot.rows>std::numeric_limits<std::size_t>::max()/snapshot.cols ||
       snapshot.size!=snapshot.rows*snapshot.cols)return false;
    const auto values=std::get<std::span<const float>>(snapshot.typed_values);
    return values.size()==snapshot.size && (!snapshot.size || (snapshot.data && values.data()==snapshot.data));
}

/**
 * @brief Require every declared participant to retain identical acquired row provenance.
 * @param replicas Immutable observations retained in actual participant order.
 * @throws std::logic_error For absent/mixed/malformed or disagreeing metadata.
 * @details Value comparison remains canonical parity's responsibility. A complete
 * publication cannot hide an absent or differently authored participant here.
 */
inline void validateSnapshotAuthoredTokenRowReplicas(std::span<const SnapshotInfo> replicas)
{
    if(replicas.empty() || !validSnapshotAuthoredTokenRows(replicas.front()))
        throw std::logic_error("Authored token snapshot has no valid owned first participant");
    const auto &first=replicas.front();
    for(const auto &replica:replicas)
        if(!validSnapshotAuthoredTokenRows(replica) || replica.authored_token_rows!=first.authored_token_rows ||
           replica.rows!=first.rows || replica.cols!=first.cols || replica.size!=first.size ||
           replica.publication!=first.publication || replica.row_ownership!=first.row_ownership)
            throw std::logic_error("Authored token snapshot participant metadata disagrees or is incomplete");
}
} // namespace llaminar2
