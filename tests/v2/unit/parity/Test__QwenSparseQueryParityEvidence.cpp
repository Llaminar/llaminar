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
#include <cstdlib>

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
    std::vector<std::string> result;
    std::string field;
    bool quoted=false;
    for(std::size_t i=0;i<line.size();++i) {
        const char c=line[i];
        if(c=='"') {
            if(quoted && i+1<line.size() && line[i+1]=='"'){field+='"';++i;}
            else quoted=!quoted;
        } else if(c==',' && !quoted){result.push_back(field);field.clear();}
        else field+=c;
    }
    result.push_back(field);
    if(quoted)throw std::invalid_argument("Unterminated test CSV quote");
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

namespace llaminar2::test::parity
{
namespace
{
/** @brief Existing counter schema authored independently for admission negatives. */
PerfStatRecord omissionRecord(uint64_t count,const char *program="proven_no_ranking")
{
    PerfStatRecord record;
    record.domain="qsa";record.name="query_program_forward_submissions";
    record.phase="decode";record.device="CPU";record.count=count;
    record.tags={{"program",program},{"role","0"},{"physical_rows","1"},
        {"observation","successful_forward_submission_not_native_replay_count"}};
    return record;
}
}
/** @test Both omitted query roles retain structural proof and no numerical credit. */
TEST(QwenSparseQueryParityEvidence, ProgramOmissionRequiresFreshPathAndIndependentHistory)
{
    const QwenSparseQueryProgramObservation path{{omissionRecord(7)},{omissionRecord(8)},{"CPU"}};
    const auto proof=inspectOmittedQwenSparseQueryEvidence(path,{2050,0,1,4096},512);
    ASSERT_TRUE(proof.valid);EXPECT_TRUE(proof.unauthored);
    EXPECT_FALSE(proof.observation.lifetime_owner);
    for(const char *role:{"QSA_INDEXER_RAW_QUERY","QSA_INDEXER_QUERY"}) {
        auto stage=structural(role,proof);
        EXPECT_TRUE(validParityCheckpointEvidence(stage,true));
        EXPECT_FALSE(stage.passed);EXPECT_FALSE(stage.evidence_valid);EXPECT_EQ(stage.total_elements,0u);
    }
    EXPECT_FALSE(validParityCheckpointEvidence(structural("QSA_INDEXER_POOLED_KEYS",proof),true));
    EXPECT_FALSE(validParityCheckpointEvidence(structural("QSA_SELECTED_INDICES",proof),true));
    EXPECT_FALSE(inspectOmittedQwenSparseQueryEvidence(path,{2051,0,1,4096},512).valid);
    EXPECT_FALSE(inspectOmittedQwenSparseQueryEvidence(path,{2050,1,1,4096},512).valid);
    EXPECT_FALSE(inspectOmittedQwenSparseQueryEvidence(path,{2050,0,2,4096},1024).valid);
    EXPECT_FALSE(inspectOmittedQwenSparseQueryEvidence(path,{2050,0,1,2050},512).valid);
    EXPECT_FALSE(inspectOmittedQwenSparseQueryEvidence(path,{2050,0,1,4096},513).valid);
    auto presence=structural("QSA_INDEXER_RAW_QUERY",inspectOmittedQwenSparseQueryEvidence(path,{2050,0,1,4096},0));
    EXPECT_FALSE(validParityCheckpointEvidence(presence,true));
    auto fabricated=structural("QSA_INDEXER_QUERY",proof);
    fabricated.query_work_evidence->observation=queryObservation(2050,1,1,1);
    EXPECT_FALSE(validParityCheckpointEvidence(fabricated,true));
    fabricated=structural("QSA_INDEXER_QUERY",proof);
    fabricated.query_work_evidence->expectedRepeats=2;
    EXPECT_FALSE(validParityCheckpointEvidence(fabricated,true));
    auto foreign=path;foreign.expected_devices={"CUDA:0"};
    EXPECT_FALSE(inspectOmittedQwenSparseQueryEvidence(foreign,{2050,0,1,4096},512).valid);
}

/** @test Stale, reset, foreign program/role and multiple-forward observations cannot waive a key. */
TEST(QwenSparseQueryParityEvidence, ProgramOmissionRejectsStaleAndMismatchedCounters)
{
    const auto valid=[](QwenSparseQueryProgramObservation path) {
        path.expected_devices={"CPU"};
        return inspectOmittedQwenSparseQueryEvidence(path,{9,0,1,4096},512).valid;
    };
    EXPECT_FALSE(valid({{}, {}}));
    EXPECT_FALSE(valid({{omissionRecord(8)},{omissionRecord(8)}}));
    EXPECT_FALSE(valid({{omissionRecord(8)},{omissionRecord(7)}}));
    EXPECT_FALSE(valid({{omissionRecord(7)},{omissionRecord(9)}}));
    EXPECT_FALSE(valid({{}, {omissionRecord(1,"ranking_possible")}}));
    auto wrong=omissionRecord(1);wrong.tags["role"]="1";
    EXPECT_FALSE(valid({{}, {wrong}}));
    wrong=omissionRecord(1);wrong.tags["physical_rows"]="2";
    EXPECT_FALSE(valid({{}, {wrong}}));
    wrong=omissionRecord(1);wrong.tags["observation"]="capture_only";
    EXPECT_FALSE(valid({{}, {wrong}}));
    EXPECT_FALSE(valid({{}, {omissionRecord(1),wrong}}));
    wrong=omissionRecord(1);wrong.tags["role"]="1";
    EXPECT_FALSE(valid({{}, {omissionRecord(1),wrong}}));
    wrong=omissionRecord(1);wrong.tags["physical_rows"]="2";
    EXPECT_FALSE(valid({{}, {omissionRecord(1),wrong}}));
    EXPECT_FALSE(valid({{}, {omissionRecord(1),omissionRecord(1,"ranking_possible")}}));
}
/** @test The actual capture helper brackets one invocation and retains immutable expected devices. */
TEST(QwenSparseQueryParityEvidence, ProgramOmissionCaptureHelperBracketsExactlyOnce)
{
    int reads=0;
    const auto read=[&] {return std::vector{omissionRecord(++reads==1?7:8)};};
    auto path=QwenSparseQueryProgramObservation::begin({"CPU"},read);
    EXPECT_EQ(reads,1);EXPECT_TRUE(path.after.empty());
    EXPECT_FALSE(inspectOmittedQwenSparseQueryEvidence(path,{9,0,1,4096},512).valid);
    path.finish(read);EXPECT_EQ(reads,2);
    EXPECT_TRUE(inspectOmittedQwenSparseQueryEvidence(path,{9,0,1,4096},512).valid);
}
/** @test CSV marks omitted program evidence without inventing acquired snapshot coordinates. */
TEST(QwenSparseQueryParityEvidence, ProgramOmissionCSVKeepsNativeFieldsBlank)
{
    struct Temporary {
        std::filesystem::path path=std::filesystem::temp_directory_path()/
            ("llaminar-query-omission-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        ~Temporary(){std::error_code error;std::filesystem::remove_all(path,error);}
    } temporary;
    const QwenSparseQueryProgramObservation path{{omissionRecord(7)},{omissionRecord(8)},{"CPU"}};
    LayerStats layer;layer.layer_idx=3;
    layer.stage_results={
        structural("QSA_INDEXER_QUERY",inspectOmittedQwenSparseQueryEvidence(path,{9,0,1,4096},512)),
        structural("QSA_INDEXER_RAW_QUERY",inspectOmittedQwenSparseQueryEvidence(path,{9,0,1,4096},512))};
    DecodeParitySummary summary;summary.steps_total=1;DecodeStepStats step;step.step_idx=0;
    step.layer_stats={layer};summary.step_stats={step};
    ASSERT_TRUE(ParityCSVArtifactWriter::writeDecode(temporary.path,"CPU0",summary));
    std::ifstream input(temporary.path/"decode_stages.csv");ASSERT_TRUE(input);
    std::string header,line;ASSERT_TRUE(std::getline(input,header));ASSERT_TRUE(std::getline(input,line));
    const auto names=fields(header),row=fields(line);ASSERT_EQ(names.size(),row.size());
    const auto value=[&](const std::string &name){const auto it=std::find(names.begin(),names.end(),name);
        return it==names.end()?std::string("missing"):row[std::size_t(it-names.begin())];};
    EXPECT_EQ(value("structural_valid"),"1");
    EXPECT_EQ(value("query_empty_reason"),"program_omitted_ranking_not_required");
    EXPECT_EQ(value("observed_prefix_rows"),"");EXPECT_EQ(value("history_prompt_rows"),"9");
    EXPECT_EQ(value("passed"),"0");EXPECT_EQ(value("cosine"),"");
    const auto witness=nlohmann::json::parse(value("query_program_observation"));
    EXPECT_EQ(witness.at("schema"),"qsa.program_omission.v1");
    EXPECT_EQ(witness.at("expected_devices"),nlohmann::json::array({"CPU"}));
    if(const char *destination=std::getenv("LLAMINAR_QUERY_OMISSION_CSV_EXPORT")) {
        const std::filesystem::path target(destination);
        ASSERT_TRUE(target.is_absolute());ASSERT_FALSE(std::filesystem::exists(target));
        ASSERT_TRUE(std::filesystem::copy_file(temporary.path/"decode_stages.csv",target));
    }
}

}

namespace llaminar2::test::parity
{
/**
 * @test Exact admitted CUDA/ROCm and mixed participant sets require every fresh submission.
 * @details These are device-free metadata fixtures using real DeviceId spellings;
 * they claim no native execution or support for a new mixed serving topology.
 */
TEST(QwenSparseQueryParityEvidence, ProgramOmissionAdmittedParticipantSetsAreExact)
{
    std::vector<std::vector<std::string>> sets;
    for (auto backend : {DeviceType::CUDA, DeviceType::ROCm})
        for (int degree : {1, 2, 4, 8})
        {
            std::vector<std::string> devices;
            for (int ordinal = 0; ordinal < degree; ++ordinal)
                devices.push_back(DeviceId(backend, ordinal).toString());
            sets.push_back(std::move(devices));
        }
    sets.push_back({DeviceId::cpu().toString(), DeviceId(DeviceType::CUDA, 0).toString(),
                    DeviceId(DeviceType::ROCm, 1).toString()});
    for (const auto &devices : sets)
    {
        SCOPED_TRACE(::testing::PrintToString(devices));
        QwenSparseQueryProgramObservation path;
        path.expected_devices = devices;
        for (const auto &device : devices)
        {
            auto before = omissionRecord(7), after = omissionRecord(8);
            before.device = after.device = device;
            path.before.push_back(before);
            path.after.push_back(after);
        }
        const auto inspect = [](const auto &observation) {
            return inspectOmittedQwenSparseQueryEvidence(observation, {9, 0, 1, 4096}, 512);
        };
        const auto proof = inspect(path);
        ASSERT_TRUE(proof.valid);
        for (const char *role : {"QSA_INDEXER_RAW_QUERY", "QSA_INDEXER_QUERY"})
            EXPECT_TRUE(validParityCheckpointEvidence(structural(role, proof), true));
        auto broken = path;
        broken.after.pop_back();
        EXPECT_FALSE(inspect(broken).valid);
        broken = path;
        broken.expected_devices.push_back(devices.front());
        EXPECT_FALSE(inspect(broken).valid);
        broken = path;
        broken.expected_devices.pop_back();
        EXPECT_FALSE(inspect(broken).valid);
        broken = path;
        auto unexpected = omissionRecord(1);
        unexpected.device = DeviceId(DeviceType::CUDA, 99).toString();
        broken.after.push_back(unexpected);
        EXPECT_FALSE(inspect(broken).valid);
        broken = path;
        broken.after.push_back(broken.after.front());
        EXPECT_FALSE(inspect(broken).valid);
        broken = path;
        broken.after.back().count = 6;
        EXPECT_FALSE(inspect(broken).valid);
        broken = path;
        broken.after.back().count = 7;
        EXPECT_FALSE(inspect(broken).valid);
    }
}
}
