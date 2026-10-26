/**
 * @file QwenSparseQueryParityEvidence.h
 * @brief Independent original-history authentication for canonical QSA query snapshots.
 *
 * A native interval authenticates acquired provenance but never chooses expected
 * work. Empty required checkpoints retain structural evidence without numerical
 * pass or sample credit. Useful rows keep their original reference coordinates.
 * A retained program may omit both query producers entirely. That case requires
 * fresh submission evidence for the admitted participants and an independently
 * empty logical interval; it must never manufacture a native snapshot payload.
 */
#pragma once
#include "QwenSparseQueryExpectedInterval.h"
#include "snapshots/SnapshotAuthoredTokenRows.h"
#include <string_view>
#include "utils/PerfStatsCollector.h"
#include <map>
#include <set>
#include <functional>
#include <nlohmann/json.hpp>
#include <limits>

namespace llaminar2::test::parity
{
/** @brief Numerical values, authored query values and unauthored query scratch are distinct. */
enum class ParityCheckpointEvidenceKind {ComparedValues,AuthoredQuery,UnauthoredQuery};
/** @return Whether the exact canonical checkpoint is an indexer query operand. */
inline bool isQwenSparseQueryCheckpoint(std::string_view stage) noexcept
{return stage=="QSA_INDEXER_RAW_QUERY" || stage=="QSA_INDEXER_QUERY";}
/** @brief Retained existing path counters bracketing exactly one submitted Decode. */
struct QwenSparseQueryProgramObservation
{
    std::vector<PerfStatRecord> before, after;
    std::vector<std::string> expected_devices; ///< Frozen actual admitted participant identities, not observed device guesses.
    /**
     * @brief Capture existing counters before one diagnostic forward.
     * @param devices Participant identities from the admitted execution plan.
     * @param read Reader for the existing immutable diagnostic counter view.
     * @return An observation awaiting the same forward's completion.
     */
    static QwenSparseQueryProgramObservation begin(std::vector<std::string> devices,
        const std::function<std::vector<PerfStatRecord>()> &read)
    {return {read(),{},std::move(devices)};}
    /**
     * @brief Complete the same observation after successful submission.
     * @param read Reader for the same counter domain used before the forward.
     */
    void finish(const std::function<std::vector<PerfStatRecord>()> &read){after=read();}
};
/**
 * @brief Serialize actual bracketing path records and their admitted participant witness.
 * @param observed Existing records retained around the exact forward submission.
 * @return Versioned tiny metadata JSON; no native snapshot values or synthetic rows.
 */
inline std::string qwenSparseQueryProgramObservationJSON(const QwenSparseQueryProgramObservation &observed)
{
    const auto records=[](const std::vector<PerfStatRecord> &source) {
        auto result=nlohmann::json::array();
        for(const auto &record:source) {
            if(record.domain!="qsa" || record.name!="query_program_forward_submissions")continue;
            result.push_back({{"device",record.device},{"phase",record.phase},
                {"kind",static_cast<int>(record.kind)},{"count",record.count},{"tags",record.tags}});
        }
        return result;
    };
    return nlohmann::json{{"schema","qsa.program_omission.v1"},
        {"participant_source","admitted_rank_local_model_devices"},
        {"expected_devices",observed.expected_devices},{"before",records(observed.before)},
        {"after",records(observed.after)}}.dump();
}
/** @brief Acquired immutable evidence plus independently supplied original history. */
struct QwenSparseQueryEvidence
{
    QwenSparseQueryHistory history;
    SnapshotInfo observation; ///< Retains original immutable owner and acquired record; never inference state.
    std::size_t referenceElements=0,referenceFirstRow=0;
    int expectedRepeats=1; ///< Frozen checkpoint row representation, never selected by observed metadata.
    bool valid=false,unauthored=false;
    std::optional<QwenSparseQueryProgramObservation> omitted_program; ///< Path evidence only; original history decides necessity.
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
 * @brief Authenticate absent query work from fresh path evidence AND independent history.
 * @param observed Existing counters around this exact successful single-row main Decode.
 * @param history Original input history, never inferred from the observed selected program.
 * @param referenceElements Header-authenticated reference extent; zero is presence-only validation.
 * @return Unauthored structural evidence without claiming a native snapshot or numerical sample.
 */
inline QwenSparseQueryEvidence inspectOmittedQwenSparseQueryEvidence(
    const QwenSparseQueryProgramObservation &observed, QwenSparseQueryHistory history,
    std::size_t referenceElements)
{
    QwenSparseQueryEvidence result{.history=history,.referenceElements=referenceElements};
    const auto interval=expectedQwenSparseQueryInterval(history);
    if(!interval || history.liveRows!=1 || interval->firstRow!=interval->endRow)return result;
    if(referenceElements && (referenceElements%512 ||
       (referenceElements/512!=1 && referenceElements/512!=std::size_t(interval->prefixRows)+1)))return result;
    const std::set<std::string> expected(observed.expected_devices.begin(),observed.expected_devices.end());
    if(expected.empty() || expected.size()!=observed.expected_devices.size() || (expected.contains("") || expected.contains("INVALID") || expected.contains("Unknown")))return result;
    struct Counted {PerfStatRecord record;uint64_t count=0;};
    using Counts=std::map<std::string,Counted>;
    const auto counts=[](const std::vector<PerfStatRecord> &records) {
        Counts values;
        for(const auto &record:records) {
            if(record.domain!="qsa" || record.name!="query_program_forward_submissions")continue;
            std::string key=record.device+"|"+record.phase+"|"+std::to_string(static_cast<int>(record.kind));
            for(const auto &[name,value]:record.tags)key+="|"+name+"="+value;
            auto &entry=values[key];entry.record=record;
            if(record.count>std::numeric_limits<uint64_t>::max()-entry.count)
                return std::optional<Counts>{};
            entry.count+=record.count;
        }
        return std::optional<Counts>{std::move(values)};
    };
    const auto before=counts(observed.before),after=counts(observed.after);
    if(!before || !after)return result;
    std::set<std::string> submitted;
    for(const auto &[key,entry]:*before) {
        const auto it=after->find(key);
        if(it==after->end() || it->second.count<entry.count)return result;
    }
    for(const auto &[key,entry]:*after) {
        const auto it=before->find(key);const auto prior=it==before->end()?0:it->second.count;
        if(entry.count<prior)return result;
        if(entry.count==prior)continue;
        const auto &record=entry.record;
        const std::map<std::string,std::string> schema{
            {"program","proven_no_ranking"},{"role","0"},{"physical_rows","1"},
            {"observation","successful_forward_submission_not_native_replay_count"}};
        if(record.phase!="decode" || record.kind!=PerfStatRecord::Kind::Counter ||
           record.tags!=schema || !expected.contains(record.device) || entry.count-prior!=1 ||
           !submitted.insert(record.device).second)return result;
    }
    if(submitted!=expected)return result;
    if(referenceElements)
        result.referenceFirstRow=referenceElements/512==std::size_t(history.liveRows)
            ?std::size_t(interval->firstRow):std::size_t(interval->prefixRows+interval->firstRow);
    result.valid=result.unauthored=true;
    result.omitted_program=observed;
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
    if(proof.omitted_program && (!proof.referenceElements || proof.expectedRepeats!=1 ||
       proof.observation.data || proof.observation.size || proof.observation.rows || proof.observation.cols ||
       proof.observation.lifetime_owner || proof.observation.authored_token_rows ||
       std::visit([](const auto values){return values.data()!=nullptr || !values.empty();},proof.observation.typed_values)))return false;
    const auto checked=proof.omitted_program
        ?inspectOmittedQwenSparseQueryEvidence(*proof.omitted_program,proof.history,proof.referenceElements)
        :inspectQwenSparseQueryEvidence(proof.observation,proof.history,
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
