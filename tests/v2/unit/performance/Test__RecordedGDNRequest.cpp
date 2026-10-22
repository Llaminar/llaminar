/**
 * @file Test__RecordedGDNRequest.cpp
 * @brief Device-free adversaries for complete recorded GDN source/trajectory admission.
 *
 * Authored tiny metadata is deliberately not a native observation or a timing
 * label. These guards prove complete snapshot closure, modulo-linked ownership
 * and fixed original arithmetic without opening any model or tensor payload.
 */
#include "../../performance/kernels/recorded/RecordedGDNRequest.h"
#include <gtest/gtest.h>
#include <set>
namespace
{
using namespace llaminar2;
using namespace llaminar2::test::recorded_gdn;
using nlohmann::json;
/** @brief Authored stat identity for an unrun fixture file. */
json file(const std::string &path,std::uint64_t bytes)
{return {{"path",path},{"device",1},{"inode",2},{"bytes",bytes},{"mtime_ns",3},{"ctime_ns",4}};}
/** @brief Exact observed source geometry with deliberately nonexistent fixture paths. */
json fixture(int degree=4,int rank=0,int rows=1)
{
    json sources=json::array();
    const std::array<const char *,4> names={"blk.0.attn_qkv.weight","blk.0.attn_gate.weight","blk.0.ssm_alpha.weight","blk.0.ssm_beta.weight"};
    const std::array<const char *,4> keys={"layer0_QKV_PROJECTION","layer0_GDN_Z_PROJECTION","layer0_GDN_ALPHA","layer0_GDN_BETA"};
    const std::array<int,4> widths={10240,6144,48,48};
    for(std::size_t i=0;i<4;++i)sources.push_back({{"source_tensor",names[i]},{"snapshot_key",keys[i]},
        {"original_n",widths[i]},{"source_gguf_type",i<2?8:0},{"source_format",i<2?"Q8_0":"F32"},
        {"prepared_format",i<2?"NativeVNNI_Q8_0":"FP16"},{"source_execution_codebook",i<2?json(19):json(nullptr)},
        {"oracle",file("/unrun/"+std::to_string(i)+".f32",std::uint64_t(rows)*widths[i]*4)}});
    return {{"schema","llaminar.recorded_gdn_projection.v1"},{"scope","local_shape"},{"operation_boundary","gdn_projection_bundle_compute"},
        {"input_origin","llaminar_captured_stage_output"},{"producer_stage","layer0_attn_norm"},{"semantic_tensor","layer0_ATTENTION_NORM"},
        {"dtype","float32_le"},{"mtp_enabled",false},{"mtp_graph_capacity",0},{"backend","rocm"},{"device_ordinal",0},
        {"output","/unrun/result.json"},{"observation_id","RecordedGDNFixture.ProductionParity"},{"snapshot_backend","original-case"},
        {"phase",rows==1?"decode":"prefill"},{"step",rows==1?0:-1},{"slice_degree",degree},{"slice_participant",rank},
        {"observation_physical_rows",rows},{"k",2560},{"gdn_key_heads",16},{"gdn_value_heads",48},{"gdn_state",128},
        {"warmup",8},{"samples",31},{"generation",1},{"model_context",1},{"sources",sources},
        {"input",file("/unrun/input.f32",std::uint64_t(rows)*2560*4)},
        {"snapshot_manifest",file("/unrun/manifest.csv",100)},{"observation_receipt",file("/unrun/receipt.json",100)},
        {"model_artifacts",json::array({file("/unrun/model.gguf",1024)})},
        {"six_precision_selectors",{{"LLAMINAR_GDN_PROJECTION_ARITHMETIC","fp16_moe_alpha_beta"},{"LLAMINAR_MOE_ROUTER_ARITHMETIC","fp16"},
            {"LLAMINAR_QFN_FA_OUTPUT_ACTIVATION","single_q8"},{"LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION","single_q8"},
            {"LLAMINAR_QFN_GDN_QKV_ACTIVATION","single_q8"},{"LLAMINAR_QFN_HC_UP_ACTIVATION","two_term_q8"}}}};
}
/** @brief Authored canonical typed rows, without a payload or runtime provenance claim. */
std::vector<std::string> manifest(const Request &r)
{
    std::vector<std::string> lines={"backend,phase,step,key,elements,exported_elements,file,scalar_type"};
    const auto append=[&](const std::string &key,const FileIdentity &file)
    {lines.push_back(r.snapshot_backend+","+r.phase+","+std::to_string(r.step)+","+key+","+std::to_string(file.bytes/4)+","+std::to_string(file.bytes/4)+","+file.path+",FP32");};
    append("layer0_ATTENTION_NORM",r.input);for(const auto &s:r.sources)append(s.key,s.oracle);return lines;
}
/** @brief Authored original-export receipt structure used only to test fail-closed metadata checks. */
json receipt(const Request &r)
{
    auto environment=r.precision;environment["LLAMINAR_PARITY_EXPORT_SNAPSHOT_KEYS"]=snapshotKeys;
    environment["LLAMINAR_PARITY_EXPORT_SNAPSHOT_MAX_ELEMENTS"]="0";
    return {{"complete",true},{"passed",true},{"returncode",0},{"native_exit",{{"exit_valid",true}}},{"driver_window_clean",true},
        {"observation_errors",json::array()},{"received_signals",json::array()},
        {"gtest_cases",json::array({{{"classname","RecordedGDNFixture"},{"name","ProductionParity"},{"status","run"},{"result","completed"},{"failures",0},{"skipped",0}}})},
        {"command",json::array({"/unrun/actual-test","--gtest_filter="+r.observation})},{"source_models",r.document.at("model_artifacts")},
        {"snapshot_manifests",json::array({r.document.at("snapshot_manifest")})},{"explicit_environment",environment}};
}
TEST(RecordedGDNRequest,NativePreparedMetadataUsesSemanticBoolean)
{
    // This is authored device-free descriptor metadata, never a native result.
    DeviceNativeVNNIMatrixDesc descriptor{};
    descriptor.n=2560;descriptor.k=2560;descriptor.codebook_id=19;descriptor.source_codebook_id=19;
    for(const std::uint8_t flag:{std::uint8_t(0),std::uint8_t(1)})
    {
        descriptor.source_is_superblock=flag;
        const auto metadata=json::parse(nativePreparedSourceMetadata(descriptor).dump());
        ASSERT_TRUE(metadata.at("source_is_superblock").is_boolean());
        EXPECT_EQ(metadata.at("source_is_superblock").get<bool>(),flag!=0);
        EXPECT_TRUE(metadata.at("codebook").is_number_unsigned());EXPECT_EQ(metadata.at("codebook"),19);
        EXPECT_TRUE(metadata.at("source_codebook").is_number_unsigned());EXPECT_EQ(metadata.at("source_codebook"),19);
        EXPECT_EQ(metadata.at("n"),descriptor.n);EXPECT_EQ(metadata.at("k"),descriptor.k);
        // The original byte-valued publisher must fail the semantic type guard.
        const json original={{"source_is_superblock",descriptor.source_is_superblock}};
        EXPECT_TRUE(original.at("source_is_superblock").is_number_unsigned());
        EXPECT_FALSE(original.at("source_is_superblock").is_boolean());
    }
}
TEST(RecordedGDNRequest,AllOriginalDecodeSlicesAndExactNineRowPrefill)
{
    for(int rows:{1,9})for(int degree:{1,2,4})for(int rank=0;rank<degree;++rank)
    {
        const auto r=parse(fixture(degree,rank,rows));EXPECT_EQ(r.rows,rows);
        EXPECT_NO_THROW(validateSnapshotManifest(r,manifest(r)));
        EXPECT_NO_THROW(validateObservationReceipt(r,receipt(r)));
    }
}
TEST(RecordedGDNRequest,ModuloLinkedValueSpansMustNotBecomeContiguousEqualRows)
{
    const auto spans=sourceSpans(parse(fixture(4,1)));
    EXPECT_EQ(spans[0],(std::vector<GDNHeadSpan>{{512,512},{2560,512},{4608,512},{6656,512},{8704,512}}));
    EXPECT_EQ(spans[1],(std::vector<GDNHeadSpan>{{512,512},{2560,512},{4608,512}}));
    EXPECT_EQ(spans[2],(std::vector<GDNHeadSpan>{{4,4},{20,4},{36,4}}));EXPECT_EQ(spans[3],spans[2]);
    EXPECT_NE(spans[1],(std::vector<GDNHeadSpan>{{1536,1536}}));
}
TEST(RecordedGDNRequest,EveryOriginalOutputRowHasOneParticipantOwner)
{
    for(int degree:{1,2,4})
    {
        std::array<std::set<int>,4> seen;
        for(int rank=0;rank<degree;++rank)
        {
            const auto r=parse(fixture(degree,rank));const auto spans=sourceSpans(r);
            for(std::size_t i=0;i<4;++i)for(const auto span:spans[i])for(int row=span.start;row<span.end();++row)
            {EXPECT_GE(row,0);EXPECT_LT(row,r.sources[i].original_n);EXPECT_TRUE(seen[i].insert(row).second);}
        }
        const auto r=parse(fixture(degree));for(std::size_t i=0;i<4;++i)EXPECT_EQ(seen[i].size(),std::size_t(r.sources[i].original_n));
    }
}
TEST(RecordedGDNRequest,RejectsMissingReorderedForeignAndChangedSiblingArithmetic)
{
    for(const char *key:{"sources","input","snapshot_manifest","observation_receipt","model_artifacts","six_precision_selectors"})
    {auto j=fixture();j.erase(key);EXPECT_THROW(parse(j),json::exception)<<key;}
    auto j=fixture();j["sources"].erase(j["sources"].begin()+3);EXPECT_THROW(parse(j),std::invalid_argument);
    j=fixture();std::swap(j["sources"][0],j["sources"][1]);EXPECT_THROW(parse(j),std::invalid_argument);
    for(std::size_t i=0;i<4;++i)
    {
        j=fixture();j["sources"][i]["original_n"]=1;EXPECT_THROW(parse(j),std::invalid_argument);
        j=fixture();j["sources"][i]["oracle"]["bytes"]=4;EXPECT_THROW(parse(j),std::invalid_argument);
        j=fixture();j["sources"][i]["source_gguf_type"]=1;EXPECT_THROW(parse(j),std::invalid_argument);
        j=fixture();j["sources"][i]["prepared_format"]="FP32";EXPECT_THROW(parse(j),std::invalid_argument);
    }
    for(const char *key:{"LLAMINAR_GDN_PROJECTION_ARITHMETIC","LLAMINAR_QFN_GDN_QKV_ACTIVATION"})
    {j=fixture();j["six_precision_selectors"][key]="foreign";EXPECT_THROW(parse(j),std::invalid_argument);}
}
TEST(RecordedGDNRequest,RejectsInventedRowsAndUnrepresentableOriginalGeometry)
{
    for(const auto &[key,value]:std::vector<std::pair<std::string,json>>{{"observation_physical_rows",512},{"gdn_key_heads",15},
        {"gdn_value_heads",47},{"gdn_state",0},{"slice_degree",3},{"slice_participant",4},{"k",2559},{"step",-1},
        {"samples",30},{"generation",0},{"model_context",-1},{"mtp_enabled",true},{"input_origin","synthetic"}})
    {auto j=fixture();j[key]=value;EXPECT_THROW(parse(j),std::invalid_argument)<<key;}
    auto j=fixture(4,0,9);j["observation_physical_rows"]=512;EXPECT_THROW(parse(j),std::invalid_argument);
}
TEST(RecordedGDNRequest,RejectsPartialDuplicateMixedAndConflictingManifestClosure)
{
    const auto r=parse(fixture());const auto original=manifest(r);
    for(std::size_t i=1;i<original.size();++i)
    {
        auto rows=original;rows.erase(rows.begin()+i);EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument);
        rows=original;rows.push_back(rows[i]);EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument);
        rows=original;auto fields=llaminar2::test::recorded_observation::csvFields(rows[i]);fields[5]="1";
        rows[i].clear();for(const auto &f:fields){if(!rows[i].empty())rows[i]+=',';rows[i]+=f;}EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument);
    }
    auto rows=original;rows.push_back(r.snapshot_backend+",decode,0,layer0_GDN_ALPHA,48,48,/foreign/file.f32,FP32");
    EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument);
}
TEST(RecordedGDNRequest,ReceiptMustRetainCompleteOriginalFiveSnapshotExportAndPrecision)
{
    const auto r=parse(fixture());auto j=receipt(r);j["explicit_environment"]["LLAMINAR_PARITY_EXPORT_SNAPSHOT_KEYS"]="FINAL_NORM,LM_HEAD";
    EXPECT_THROW(validateObservationReceipt(r,j),std::invalid_argument);
    j=receipt(r);j["explicit_environment"]["LLAMINAR_QFN_HC_UP_ACTIVATION"]="single_q8";EXPECT_THROW(validateObservationReceipt(r,j),std::invalid_argument);
    j=receipt(r);j["gtest_cases"][0]["failures"]=1;EXPECT_THROW(validateObservationReceipt(r,j),std::invalid_argument);
}
TEST(RecordedGDNRequest,CohortRetainsEveryOriginalSourceAndInputOnEveryMember)
{
    for(int degree:{1,2,4})
    {
        json devices=json::array();for(int rank=0;rank<degree;++rank)devices.push_back(rank);
        json j={{"schema","llaminar.recorded_gdn_cohort.v1"},{"scope","actual_tp"},{"operation_boundary","gdn_projection_bundle_compute"},
            {"member_template",fixture(degree)},{"device_ordinals",devices}};
        const auto cohort=CohortRequest::fromJson(j);ASSERT_EQ(cohort.members().size(),std::size_t(degree));
        for(int rank=0;rank<degree;++rank)
        {const auto &m=cohort.members()[rank];EXPECT_EQ(m.participant,rank);EXPECT_EQ(m.ordinal,rank);EXPECT_EQ(m.input.path,cohort.source().input.path);EXPECT_EQ(m.document.at("sources"),cohort.source().document.at("sources"));}
        j["device_ordinals"]=json::array();EXPECT_THROW(CohortRequest::fromJson(j),std::invalid_argument);
    }
    json j={{"schema","llaminar.recorded_gdn_cohort.v1"},{"scope","actual_tp"},{"operation_boundary","gdn_projection_bundle_compute"},
        {"member_template",fixture()},{"device_ordinals",json::array({0,1,1,3})}};
    EXPECT_THROW(CohortRequest::fromJson(j),std::invalid_argument);
}
} // namespace
