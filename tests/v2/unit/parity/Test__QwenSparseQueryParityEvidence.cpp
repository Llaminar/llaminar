/**
 * @file Test__QwenSparseQueryParityEvidence.cpp
 * @brief Device-free original-history and required-query interval parity regressions.
 *
 * Tests use tiny owned diagnostic values and the actual canonical static
 * finalizers/inventory/writer. They instantiate no model, runner, or backend.
 */
#include "integration/parity/ParityTestBase.h"
#include <gtest/gtest.h>
#include <array>
#include <chrono>
#include <fstream>
#include <sstream>

namespace llaminar2::test::parity
{
namespace
{
/** @brief Expose actual static evidence consumers without constructing the model fixture. */
class QueryWorkProbe:public ParityTestBase
{
public:
    using ParityTestBase::acceptsFinalLogitRawEvidence;
    using ParityTestBase::acceptsRequiredRawParityStages;
    using ParityTestBase::finalizeDecodeLayerNumerics;
    using ParityTestBase::finalizePrefillLayerNumerics;
};
/**
 * @brief Construct owned acquired metadata independently from the tested history oracle.
 * @param prefix Original acquired prefix. @param live Original acquired live count.
 * @param first Original logical authored first row. @param end Original logical authored end.
 * @param repeat Frozen physical row representation for this fixture.
 * @return A small retained observation with explicit original coordinates.
 */
SnapshotInfo queryObservation(int prefix,int live,int first,int end,int repeat=1)
{
    auto owner=std::make_shared<const std::vector<float>>(std::size_t(end-first)*512,1.f);
    SnapshotAuthoredTokenRowsObservation observation{
        .requests={{.prefixRows=prefix,.liveRows=live,.rowOffset=0,.rowCapacity=16,.sourceCapacity=4096,
            .firstRequiredLogicalRow=2051,.firstRow=first*repeat,.endRow=end*repeat,
            .emptyReason=first==end?SnapshotEmptyReason::RankingNotRequired:SnapshotEmptyReason::None}},
        .physicalRows=std::size_t(16*repeat),.repeatsPerRow=repeat};
    return SnapshotInfo{.data=owner->empty()?nullptr:owner->data(),.size=owner->size(),
        .rows=std::size_t((end-first)*repeat),.cols=std::size_t(512/repeat),.lifetime_owner=owner,
        .typed_values=std::span<const float>(*owner),.semantic_axis=SnapshotSemanticAxis::AuthoredTokenRows,
        .authored_token_rows=std::move(observation)};
}
/** @brief Create an empty required-query record without granting numerical equality. */
StageComparisonResult structural(const char *stage,QwenSparseQueryEvidence proof)
{
    StageComparisonResult result;result.stage_name=stage;
    result.checkpoint_evidence_kind=ParityCheckpointEvidenceKind::UnauthoredQuery;
    result.query_work_evidence=std::move(proof);
    result.cosine_similarity=result.rel_l2_norm=result.max_abs_diff=result.snr_db=result.rmse=
        result.error_entropy=std::numeric_limits<float>::quiet_NaN();
    return result;
}
/** @brief Create a real authored useful-query numerical record using the canonical comparator. */
StageComparisonResult authored(const char *stage,QwenSparseQueryEvidence proof)
{
    const auto &snapshot=proof.observation;
    auto result=compareParityTensorData(snapshot.data,snapshot.data,snapshot.size,stage,.99f);
    result.checkpoint_evidence_kind=ParityCheckpointEvidenceKind::AuthoredQuery;
    result.query_work_evidence=std::move(proof);
    return result;
}
/** @brief Parse one simple canonical CSV row without project script imports. */
std::vector<std::string> fields(const std::string &line)
{
    std::vector<std::string> result;std::stringstream input(line);std::string field;
    while(std::getline(input,field,','))result.push_back(field);
    if(!line.empty()&&line.back()==',')result.emplace_back();
    return result;
}
}

/** @test Every boundary row follows the independent complete-pool equation, including empty live buckets. */
TEST(QwenSparseQueryParityEvidence, OriginalHistoryDerivesExactSuffixWithoutNativeThreshold)
{
    for(int prefix=2047;prefix<=2053;++prefix)
        for(int capacity:{1,3,16,17,64})for(int live=0;live<=capacity;++live)
        {
            const auto expected=expectedQwenSparseQueryInterval({std::size_t(prefix),0,live,4096});
            ASSERT_TRUE(expected);EXPECT_EQ(expected->prefixRows,prefix);EXPECT_EQ(expected->endRow,live);
            for(int row=0;row<live;++row)
                EXPECT_EQ(row>=expected->firstRow,(prefix+row+1)/4>512);
        }
    EXPECT_FALSE(expectedQwenSparseQueryInterval({std::size_t(INT32_MAX),1,1,INT32_MAX}));
    EXPECT_FALSE(expectedQwenSparseQueryInterval({2050,0,-1,4096}));
    EXPECT_FALSE(expectedQwenSparseQueryInterval({2050,0,3,2052}));
}

/** @test Native false skips, extra unused rows, stale prefixes and wrong original-coordinate suffixes fail. */
TEST(QwenSparseQueryParityEvidence, RejectsFalseSkipExtraAuthoredAndWrongSuffixAt2052)
{
    const QwenSparseQueryHistory shortHistory{2050,0,1,4096};
    const auto shortProof=inspectQwenSparseQueryEvidence(queryObservation(2050,1,1,1),shortHistory,512);
    ASSERT_TRUE(shortProof.valid);EXPECT_TRUE(shortProof.unauthored);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(queryObservation(2050,1,0,1),shortHistory,512).valid);
    const QwenSparseQueryHistory needed{2050,1,1,4096};
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(queryObservation(2051,1,1,1),needed,512).valid);
    ASSERT_TRUE(inspectQwenSparseQueryEvidence(queryObservation(2051,1,0,1),needed,512).valid);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(queryObservation(2047,1,1,1),{2048,0,1,4096},512).valid);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence({},shortHistory,512).valid);
    const QwenSparseQueryHistory crossing{2050,0,3,4096};
    const auto useful=inspectQwenSparseQueryEvidence(queryObservation(2050,3,1,3),crossing,3*512);
    ASSERT_TRUE(useful.valid);EXPECT_FALSE(useful.unauthored);EXPECT_EQ(useful.referenceFirstRow,1u);
    const auto full=inspectQwenSparseQueryEvidence(queryObservation(2050,3,1,3),crossing,2053*512);
    ASSERT_TRUE(full.valid);EXPECT_EQ(full.referenceFirstRow,2051u);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(queryObservation(2050,3,0,3),crossing,3*512).valid);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(queryObservation(2050,3,2,3),crossing,3*512).valid);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(queryObservation(2050,3,1,2),crossing,3*512).valid);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(queryObservation(2050,3,1,3),crossing,2*512).valid);
    const auto expanded=inspectQwenSparseQueryEvidence(queryObservation(2050,3,1,3,4),crossing,3*512,4);
    ASSERT_TRUE(expanded.valid);EXPECT_EQ(expanded.referenceFirstRow,1u);
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(expanded.observation,crossing,3*512,1).valid);
}

/** @test Original ownership, record inventory, packing, threshold and empty reason remain independently mandatory. */
TEST(QwenSparseQueryParityEvidence, MalformedAcquiredMetadataCannotAuthenticateEmptyScratch)
{
    const QwenSparseQueryHistory history{2050,0,1,4096};
    const auto good=queryObservation(2050,1,1,1);
    for(int mutation=0;mutation<14;++mutation)
    {
        auto actual=good;auto &observation=*actual.authored_token_rows;auto &record=observation.requests.front();
        switch(mutation)
        {
            case 0:actual.lifetime_owner.reset();break;
            case 1:record.firstRequiredLogicalRow=2052;break;
            case 2:record.prefixRows=2049;break;
            case 3:record.sourceCapacity=8192;break;
            case 4:record.rowOffset=1;break;
            case 5:record.payloadRowOffset=1;break;
            case 6:record.emptyReason=SnapshotEmptyReason::None;break;
            case 7:record.reserved=1;break;
            case 8:observation.physicalRows=17;break;
            case 9:observation.sliceFirstRow=1;break;
            case 10:observation.requests.push_back(record);break;
            case 11:actual.semantic_axis=SnapshotSemanticAxis::TokenRows;break;
            case 12:actual.typed_values=std::span<const std::int64_t>{};break;
            case 13:actual.authored_token_rows.reset();break;
        }
        EXPECT_FALSE(inspectQwenSparseQueryEvidence(actual,history,512).valid)<<mutation;
    }
}

/** @test Packed holes and sliced head-expanded metadata retain original request coordinates and all replica owners. */
TEST(QwenSparseQueryParityEvidence, PackedRequestsAndIntersectingSlicesKeepExactOriginalCoordinates)
{
    const auto owner=std::make_shared<const std::vector<float>>(3*512,1.f);
    SnapshotAuthoredTokenRowsObservation observation{
        .requests={
            {.prefixRows=2047,.liveRows=3,.rowOffset=0,.rowCapacity=3,.sourceCapacity=4096,
             .firstRequiredLogicalRow=2051,.firstRow=3,.endRow=3,.emptyReason=SnapshotEmptyReason::RankingNotRequired},
            {.prefixRows=2050,.liveRows=3,.rowOffset=3,.rowCapacity=3,.sourceCapacity=4096,
             .firstRequiredLogicalRow=2051,.firstRow=4,.endRow=6,.payloadRowOffset=0},
            {.prefixRows=2052,.liveRows=1,.rowOffset=6,.rowCapacity=3,.sourceCapacity=4096,
             .firstRequiredLogicalRow=2051,.firstRow=6,.endRow=7,.payloadRowOffset=2}},
        .physicalRows=9};
    SnapshotInfo full{.data=owner->data(),.size=owner->size(),.rows=3,.cols=512,.lifetime_owner=owner,
        .typed_values=std::span<const float>(*owner),.semantic_axis=SnapshotSemanticAxis::AuthoredTokenRows,
        .authored_token_rows=observation};
    ASSERT_TRUE(validSnapshotAuthoredTokenRows(full));
    for(int participant=0;participant<2;++participant)
    {
        std::array replicas{full,full};EXPECT_NO_THROW(validateSnapshotAuthoredTokenRowReplicas(replicas));
        replicas[participant].authored_token_rows->requests[2].payloadRowOffset=1;
        EXPECT_THROW(validateSnapshotAuthoredTokenRowReplicas(replicas),std::logic_error);
        replicas={full,full};replicas[participant].lifetime_owner.reset();
        EXPECT_THROW(validateSnapshotAuthoredTokenRowReplicas(replicas),std::logic_error);
    }
    auto extra=full;extra.authored_token_rows->requests.push_back(observation.requests.back());
    EXPECT_FALSE(validSnapshotAuthoredTokenRows(extra));
    const auto slicedOwner=std::make_shared<const std::vector<float>>(2*512,2.f);
    auto sliced=full;sliced.data=slicedOwner->data();sliced.size=slicedOwner->size();sliced.rows=8;sliced.cols=128;
    sliced.lifetime_owner=slicedOwner;sliced.typed_values=std::span<const float>(*slicedOwner);
    sliced.authored_token_rows=SnapshotAuthoredTokenRowsObservation{
        .requests={{.prefixRows=2050,.liveRows=3,.rowOffset=-4,.rowCapacity=3,.sourceCapacity=4096,
            .firstRequiredLogicalRow=2051,.firstRow=0,.endRow=8}},
        .physicalRows=8,.repeatsPerRow=4,.sliceFirstRow=16};
    EXPECT_TRUE(validSnapshotAuthoredTokenRows(sliced));
    // The current canonical caller has no independent request1 history and
    // must reject this otherwise structurally valid sliced observation.
    EXPECT_FALSE(inspectQwenSparseQueryEvidence(sliced,{2050,0,3,4096},3*512,4).valid);
}

/** @test Both raw and prepared Q keep required identity while structural records receive no numerical sample credit. */
TEST(QwenSparseQueryParityEvidence, ActualCanonicalInventoryAndAggregatesKeepBothRequiredQueries)
{
    const auto proof=inspectQwenSparseQueryEvidence(queryObservation(2050,1,1,1),{2050,0,1,4096},512);
    auto raw=structural("QSA_INDEXER_RAW_QUERY",proof),prepared=structural("QSA_INDEXER_QUERY",proof);
    const std::array values{1.f,2.f};
    const auto attention=compareParityTensorData(values.data(),values.data(),values.size(),"ATTENTION_OUTPUT",.99f);
    LayerStats layer;layer.layer_idx=0;layer.stage_results={raw,prepared,attention};
    QueryWorkProbe::finalizeDecodeLayerNumerics(layer,ParityConfig{},true);
    ASSERT_TRUE(layer.evidence_valid);EXPECT_TRUE(layer.passed);EXPECT_EQ(layer.stages_compared,1);
    const std::vector<std::string> required{"layer0_QSA_INDEXER_RAW_QUERY","layer0_QSA_INDEXER_QUERY","layer0_ATTENTION_OUTPUT"};
    EXPECT_TRUE(QueryWorkProbe::acceptsRequiredRawParityStages(std::vector{layer},required,ParityForwardPhase::Decode));
    ParityConfig config;config.early_layers_count=1;config.graph_snapshot_policy.required_decode_snapshot_keys=required;
    EXPECT_TRUE(QueryWorkProbe::acceptsFinalLogitRawEvidence(std::vector{layer},config,ParityForwardPhase::Decode));
    auto missing=layer;missing.stage_results.erase(missing.stage_results.begin());
    EXPECT_FALSE(QueryWorkProbe::acceptsRequiredRawParityStages(std::vector{missing},required,ParityForwardPhase::Decode));
    auto duplicate=layer;duplicate.stage_results.push_back(raw);
    EXPECT_FALSE(QueryWorkProbe::acceptsRequiredRawParityStages(std::vector{duplicate},required,ParityForwardPhase::Decode));
    layer.stage_results={raw,prepared};QueryWorkProbe::finalizeDecodeLayerNumerics(layer,ParityConfig{},true);
    EXPECT_FALSE(layer.passed);EXPECT_FALSE(layer.evidence_valid);EXPECT_EQ(layer.stages_compared,0);
    layer.stage_results={raw,prepared,attention};QueryWorkProbe::finalizePrefillLayerNumerics(layer,ParityConfig{});
    EXPECT_FALSE(layer.passed);
    raw.passed=true;EXPECT_FALSE(validParityCheckpointEvidence(raw,true));
    const auto useful=inspectQwenSparseQueryEvidence(queryObservation(2050,3,1,3),{2050,0,3,4096},3*512);
    auto numeric=authored("QSA_INDEXER_QUERY",useful);EXPECT_TRUE(validParityCheckpointEvidence(numeric,true));
    numeric.evidence_valid=false;EXPECT_FALSE(validParityCheckpointEvidence(numeric,true));
}

/** @test The actual writer emits typed original-row provenance with blank numerical fields for an empty query. */
TEST(QwenSparseQueryParityEvidence, WriterKeepsRankingNotRequiredDistinctFromNumericalEquality)
{
    /** @brief Retire this test's uniquely owned temporary CSV directory. */
    struct Temporary
    {
        std::filesystem::path path=std::filesystem::temp_directory_path()/
            ("llaminar-query-interval-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        /** @brief Remove only owned diagnostic files; never throw from retirement. */
        ~Temporary(){std::error_code error;std::filesystem::remove_all(path,error);}
    } temporary;
    const auto proof=inspectQwenSparseQueryEvidence(queryObservation(2050,1,1,1),{2050,0,1,4096},512);
    LayerStats layer;layer.layer_idx=3;layer.stage_results={structural("QSA_INDEXER_RAW_QUERY",proof)};
    DecodeParitySummary summary;summary.steps_total=1;DecodeStepStats step;step.step_idx=0;step.layer_stats={layer};summary.step_stats={step};
    ASSERT_TRUE(ParityCSVArtifactWriter::writeDecode(temporary.path,"CPU0",summary));
    std::ifstream input(temporary.path/"decode_stages.csv");ASSERT_TRUE(input);
    std::string header,line;ASSERT_TRUE(std::getline(input,header));ASSERT_TRUE(std::getline(input,line));
    const auto names=fields(header),row=fields(line);ASSERT_EQ(names.size(),row.size());
    const auto value=[&](const std::string &name){const auto p=std::find(names.begin(),names.end(),name);
        return p==names.end()?std::string("missing"):row[std::size_t(p-names.begin())];};
    EXPECT_EQ(value("checkpoint_evidence_kind"),"unauthored_query");EXPECT_EQ(value("structural_valid"),"1");
    EXPECT_EQ(value("query_empty_reason"),"ranking_not_required");EXPECT_EQ(value("passed"),"0");
    EXPECT_EQ(value("canonical_passed"),"0");EXPECT_EQ(value("cosine"),"");EXPECT_EQ(value("llaminar_elements"),"");
    EXPECT_EQ(value("history_prompt_rows"),"2050");EXPECT_EQ(value("observed_prefix_rows"),"2050");
    EXPECT_EQ(value("observed_first_row"),"1");EXPECT_EQ(value("observed_end_row"),"1");
    EXPECT_EQ(value("observed_payload_rows"),"0");EXPECT_EQ(value("observed_payload_columns"),"512");
    EXPECT_EQ(value("observed_payload_elements"),"0");EXPECT_EQ(value("reference_first_row"),"1");
}
} // namespace llaminar2::test::parity
