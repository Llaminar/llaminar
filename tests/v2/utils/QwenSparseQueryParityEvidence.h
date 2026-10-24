/**
 * @file QwenSparseQueryParityEvidence.h
 * @brief Independent original-history authentication for canonical QSA query snapshots.
 *
 * A native interval authenticates acquired provenance but never chooses expected
 * work. Empty required checkpoints retain structural evidence without numerical
 * pass or sample credit. Useful rows keep their original reference coordinates.
 */
#pragma once
#include "QwenSparseQueryExpectedInterval.h"
#include "snapshots/SnapshotAuthoredTokenRows.h"
#include <string_view>

namespace llaminar2::test::parity
{
/** @brief Numerical values, authored query values and unauthored query scratch are distinct. */
enum class ParityCheckpointEvidenceKind {ComparedValues,AuthoredQuery,UnauthoredQuery};
/** @return Whether the exact canonical checkpoint is an indexer query operand. */
inline bool isQwenSparseQueryCheckpoint(std::string_view stage) noexcept
{return stage=="QSA_INDEXER_RAW_QUERY" || stage=="QSA_INDEXER_QUERY";}
/** @brief Acquired immutable evidence plus independently supplied original history. */
struct QwenSparseQueryEvidence
{
    QwenSparseQueryHistory history;
    SnapshotInfo observation; ///< Retains original immutable owner and acquired record; never inference state.
    std::size_t referenceElements=0,referenceFirstRow=0;
    int expectedRepeats=1; ///< Frozen checkpoint row representation, never selected by observed metadata.
    bool valid=false,unauthored=false;
};
/**
 * @brief Authenticate one actual canonical request before loading/comparing useful values.
 * @param actual Owned acquired publication with original row records.
 * @param history Independent original prompt and submitted Decode row history.
 * @param referenceElements Header-authenticated full-history or invocation-row reference extent.
 * @param expectedRepeats Frozen physical rows per logical query; canonical raw/prepared use one.
 * @return Exact structural evidence and original reference row offset; no numerical verdict.
 * @details The current canonical runner submits one request. Extra acquired
 * requests are rejected rather than assigned an invented reference history.
 */
inline QwenSparseQueryEvidence inspectQwenSparseQueryEvidence(const SnapshotInfo &actual,
    QwenSparseQueryHistory history,std::size_t referenceElements,int expectedRepeats=1)
{
    QwenSparseQueryEvidence result{.history=history,.observation=actual,
        .referenceElements=referenceElements,.expectedRepeats=expectedRepeats};
    const auto expected=expectedQwenSparseQueryInterval(history);
    if(!expected || expectedRepeats<=0 || 512%expectedRepeats ||
       !validSnapshotAuthoredTokenRows(actual))return result;
    const auto &acquired=*actual.authored_token_rows;
    if(acquired.requests.size()!=1 || acquired.sliceFirstRow!=0 || acquired.repeatsPerRow!=expectedRepeats ||
       actual.cols!=static_cast<std::size_t>(512/expectedRepeats))return result;
    const auto &record=acquired.requests.front();
    if(record.prefixRows!=expected->prefixRows || record.liveRows!=history.liveRows || record.rowOffset!=0 ||
       record.sourceCapacity!=history.sourceCapacity || record.firstRequiredLogicalRow!=2051 ||
       acquired.physicalRows!=static_cast<std::size_t>(record.rowCapacity)*expectedRepeats ||
       record.firstRow!=static_cast<std::int64_t>(expected->firstRow)*expectedRepeats ||
       record.endRow!=static_cast<std::int64_t>(expected->endRow)*expectedRepeats || record.payloadRowOffset!=0)
        return result;
    if(referenceElements%512)return result;
    const auto referenceRows=referenceElements/512;
    if(referenceRows!=static_cast<std::size_t>(history.liveRows) &&
       referenceRows!=static_cast<std::size_t>(expected->prefixRows)+history.liveRows)return result;
    result.referenceFirstRow=referenceRows==static_cast<std::size_t>(history.liveRows)
        ?expected->firstRow:static_cast<std::size_t>(expected->prefixRows)+expected->firstRow;
    result.unauthored=expected->firstRow==expected->endRow;
    result.valid=true;
    return result;
}
/**
 * @brief Recheck retained structural evidence without granting numerical equality.
 * @tparam Stage Actual canonical comparison record or a device-free contract fixture.
 * @param stage Exact original checkpoint record.
 * @param requireQueryProof Decode requires original history for both required Q checkpoints.
 * @return Whether its typed raw witness is complete, independent of numerical tolerance.
 */
template<class Stage> bool validParityCheckpointEvidence(const Stage &stage,bool requireQueryProof=false)
{
    if(stage.checkpoint_evidence_kind==ParityCheckpointEvidenceKind::ComparedValues)
        return stage.evidence_valid && !stage.query_work_evidence &&
            !(requireQueryProof && isQwenSparseQueryCheckpoint(stage.stage_name));
    if(!requireQueryProof || !isQwenSparseQueryCheckpoint(stage.stage_name) || !stage.query_work_evidence)return false;
    const auto &proof=*stage.query_work_evidence;
    const auto checked=inspectQwenSparseQueryEvidence(proof.observation,proof.history,
        proof.referenceElements,proof.expectedRepeats);
    if(!proof.valid || !checked.valid || proof.unauthored!=checked.unauthored ||
       proof.referenceFirstRow!=checked.referenceFirstRow)return false;
    if(stage.checkpoint_evidence_kind==ParityCheckpointEvidenceKind::AuthoredQuery)
        return !checked.unauthored && stage.evidence_valid && stage.total_elements==proof.observation.size &&
            !stage.integer_mismatch_count && !stage.is_routing_stage && !stage.route_conditioned_proof;
    return stage.checkpoint_evidence_kind==ParityCheckpointEvidenceKind::UnauthoredQuery && checked.unauthored &&
        !stage.passed && !stage.evidence_valid && stage.total_elements==0 && !stage.integer_mismatch_count &&
        !stage.is_routing_stage && !stage.route_conditioned_proof;
}
} // namespace llaminar2::test::parity
