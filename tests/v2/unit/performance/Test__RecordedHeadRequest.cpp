/**
 * @file Test__RecordedHeadRequest.cpp
 * @brief Fast device-free adversaries for actual-row/slice/manifest qualification.
 *
 * These tests open no payload/model, initialize no backend and mint no native
 * observation. They prove closed parser boundaries; real recorded bytes and
 * LMHeadStage/GEMM capture remain an explicit model-dependent diagnostic gate.
 * Concurrent admission tests use only the actual performance-local host gate;
 * no model, backend, GPU context or runtime library is initialized.
 */
#include "../../performance/kernels/recorded/RecordedHeadCohort.h"
#include <atomic>
#include <future>
#include <gtest/gtest.h>
namespace
{
using namespace llaminar2::test::recorded_head;
using nlohmann::json;
/** @brief Small authored metadata fixture, deliberately not a runtime receipt. */
json fixture()
{
    auto f=[](const char *path,std::uint64_t bytes){return json{{"path",path},{"device",1},{"inode",2},{"bytes",bytes},{"mtime_ns",3},{"ctime_ns",4}};};
    return {{"schema","llaminar.recorded_head_projection.v1"},{"scope","local_shape"},
        {"input_origin","llaminar_captured_stage_output"},{"producer_stage","final_norm"},{"semantic_tensor","FINAL_NORM"},
        {"dtype","float32_le"},{"head_operand_precision","single_q8"},{"source_format","Q8_0"},{"source_gguf_type",8},
        {"source_execution_codebook",19},{"mtp_enabled",false},{"mtp_graph_capacity",0},{"serial_equivalent_partition_width",0},
        {"backend","rocm"},{"device_ordinal",0},{"output","/unrun/result.json"},{"observation_id","QfnRecordedFixture.ProductionParity"},
        {"phase","decode"},{"source_tensor","output.weight"},{"original_n",248320},{"k",2560},
        {"slice_degree",4},{"slice_participant",2},{"first_column",124160},{"local_n",62080},
        {"observation_physical_rows",1},{"observation_selected_row",0},{"step",3},
        {"snapshot_backend","typed-test-case"},{"snapshot_key","FINAL_NORM"},{"logits_key","LM_HEAD"},
        {"warmup",8},{"samples",31},{"generation",1},{"model_context",1},
        {"input",f("/unrun/final_norm.f32",10240)},{"logits",f("/unrun/lm_head.f32",993280)},
        {"snapshot_manifest",f("/unrun/manifest.csv",100)},{"observation_receipt",f("/unrun/controller.json",100)},
        {"model_artifacts",json::array({f("/unrun/model.gguf",1024)})},
        {"six_precision_selectors",{{"LLAMINAR_GDN_PROJECTION_ARITHMETIC","fp16_moe_alpha_beta"},{"LLAMINAR_MOE_ROUTER_ARITHMETIC","fp16"},
            {"LLAMINAR_QFN_FA_OUTPUT_ACTIVATION","single_q8"},{"LLAMINAR_QFN_GDN_OUTPUT_ACTIVATION","single_q8"},
            {"LLAMINAR_QFN_GDN_QKV_ACTIVATION","single_q8"},{"LLAMINAR_QFN_HC_UP_ACTIVATION","two_term_q8"}}}};
}
/** @brief Exact canonical exported header/rows, whose provenance stays fixture-only. */
std::vector<std::string> manifest(const Request &r)
{
    const auto phase=r.phase=="decode"?"decode":"prefill";
    return {"backend,phase,step,key,elements,exported_elements,file,scalar_type",
        r.snapshot_backend+","+phase+","+std::to_string(r.step)+","+r.snapshot_key+","+std::to_string(r.input.bytes/4)+","+std::to_string(r.input.bytes/4)+","+r.input.path+",FP32",
        r.snapshot_backend+","+phase+","+std::to_string(r.step)+","+r.logits_key+","+std::to_string(r.logits.bytes/4)+","+std::to_string(r.logits.bytes/4)+","+r.logits.path+",FP32"};
}
/** @brief Authored shape of the existing root focused export result, never a native receipt. */
json controller(const json &request)
{
    auto env=request.at("six_precision_selectors");env["LLAMINAR_PARITY_EXPORT_SNAPSHOT_KEYS"]="FINAL_NORM,LM_HEAD";
    env["LLAMINAR_PARITY_EXPORT_SNAPSHOT_MAX_ELEMENTS"]="0";
    return {{"complete",true},{"passed",true},{"returncode",0},{"driver_window_clean",true},
        {"native_exit",{{"exit_valid",true}}},{"observation_errors",json::array()},{"received_signals",json::array()},
        {"command",json::array({"/unrun/actual-original-test","--gtest_filter=QfnRecordedFixture.ProductionParity"})},
        {"gtest_cases",json::array({{{"classname","QfnRecordedFixture"},{"name","ProductionParity"},{"status","run"},
            {"result","completed"},{"failures",0},{"skipped",0}}})},
        {"source_models",request.at("model_artifacts")},{"snapshot_manifests",json::array({request.at("snapshot_manifest")})},
        {"explicit_environment",env}};
}
TEST(RecordedHeadRequest,RequiresActualPassedExportCellSourceAndManifest)
{
    const auto j=fixture();const auto r=parse(j);auto c=controller(j);
    EXPECT_NO_THROW(validateObservationReceipt(r,c));
    EXPECT_THROW(validateObservationReceipt(r,json::object()),json::exception);
    for(const char *key:{"complete","passed","driver_window_clean"})
    {auto bad=c;bad[key]=false;EXPECT_THROW(validateObservationReceipt(r,bad),std::invalid_argument)<<key;}
    auto bad=c;bad["gtest_cases"][0]["skipped"]=1;EXPECT_THROW(validateObservationReceipt(r,bad),std::invalid_argument);
    bad=c;bad["gtest_cases"].push_back(bad["gtest_cases"][0]);EXPECT_THROW(validateObservationReceipt(r,bad),std::invalid_argument);
    bad=c;bad["command"][1]="--gtest_filter=different";EXPECT_THROW(validateObservationReceipt(r,bad),std::invalid_argument);
    bad=c;bad["source_models"][0]["inode"]=99;EXPECT_THROW(validateObservationReceipt(r,bad),std::invalid_argument);
    bad=c;bad["snapshot_manifests"][0]["bytes"]=99;EXPECT_THROW(validateObservationReceipt(r,bad),std::invalid_argument);
    bad=c;bad["explicit_environment"]["LLAMINAR_QFN_HC_UP_ACTIVATION"]="single_q8";
    EXPECT_THROW(validateObservationReceipt(r,bad),std::invalid_argument);
}
TEST(RecordedHeadRequest,OriginalAndAllEqualSlices)
{
    for(int degree:{1,2,4})for(int rank=0;rank<degree;++rank)
    {auto j=fixture();j["slice_degree"]=degree;j["slice_participant"]=rank;j["first_column"]=rank*(248320/degree);j["local_n"]=248320/degree;
     const auto r=parse(j);EXPECT_NO_THROW(validateSnapshotManifest(r,manifest(r)));}
}
TEST(RecordedHeadRequest,OriginalPositivePartitionPolicyRemainsFixedAcrossPhysicalWidths)
{
    for(int degree:{1,2,4})
    {auto j=fixture();j["slice_degree"]=degree;j["slice_participant"]=0;j["first_column"]=0;j["local_n"]=248320/degree;
     j["serial_equivalent_partition_width"]=62080;EXPECT_EQ(parse(j).serial_width,62080);}
    auto j=fixture();j["serial_equivalent_partition_width"]=-1;EXPECT_THROW(parse(j),std::invalid_argument);
}
TEST(RecordedHeadRequest,PrefillUsesExactRecordedTerminalRow)
{
    auto j=fixture();j["phase"]="prefill_terminal";j["step"]=-1;j["observation_physical_rows"]=512;j["observation_selected_row"]=506;j["input"]["bytes"]=512*2560*4;
    auto r=parse(j);EXPECT_EQ(r.selected_row,506);EXPECT_NO_THROW(validateSnapshotManifest(r,manifest(r)));
    j["observation_selected_row"]=512;EXPECT_THROW(parse(j),std::invalid_argument);
}
TEST(RecordedHeadRequest,RejectsSyntheticWrongSourceAndArithmetic)
{
    for(const auto &[key,value]:std::vector<std::pair<std::string,json>>{{"scope","actual_tp"},{"input_origin","synthetic"},
        {"producer_stage","lm_head"},{"semantic_tensor","LM_HEAD"},{"dtype","INT64"},{"source_tensor","token_embd.weight"},
        {"source_format","Q8_1"},{"source_gguf_type",9},{"source_execution_codebook",7},{"head_operand_precision","two_term_q8"},
        {"mtp_enabled",true},{"mtp_graph_capacity",1},{"serial_equivalent_partition_width",62081}})
    {auto j=fixture();j[key]=value;EXPECT_THROW(parse(j),std::invalid_argument)<<key;}
}
TEST(RecordedHeadRequest,RejectsMalformedGeometryAndNumerics)
{
    for(const auto &[key,value]:std::vector<std::pair<std::string,json>>{{"k",0},{"k",2559},{"slice_degree",3},
        {"slice_participant",4},{"first_column",0},{"local_n",62079},{"samples",30},{"warmup",0},{"generation",0},
        {"samples",31.5},{"model_context",-1},{"observation_selected_row",1},{"step",-1}})
    {auto j=fixture();j[key]=value;EXPECT_THROW(parse(j),std::invalid_argument)<<key;}
}
TEST(RecordedHeadRequest,RejectsMissingOrUnboundAssets)
{
    for(const auto &key:{"input","logits","snapshot_manifest","observation_receipt","model_artifacts","six_precision_selectors"})
    {auto j=fixture();j.erase(key);EXPECT_THROW(parse(j),json::exception)<<key;}
    auto j=fixture();j["input"]["bytes"]=10239;EXPECT_THROW(parse(j),std::invalid_argument);
    j=fixture();j["input"]["inode"]=-1;EXPECT_THROW(parse(j),std::invalid_argument);
    j=fixture();j["model_artifacts"].push_back(j["model_artifacts"][0]);EXPECT_THROW(parse(j),std::invalid_argument);
}
TEST(RecordedHeadRequest,RejectsTruncatedDuplicateForeignAndUntypedCanonicalRows)
{
    const auto r=parse(fixture());const auto original=manifest(r);
    auto rows=original;rows.push_back(rows[1]);EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument);
    rows=original;rows.erase(rows.begin()+1);EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument);
    for(std::size_t field:{0u,1u,2u,3u,4u,5u,7u})
    {rows=original;auto cells=csvFields(rows[1]);cells[field]="foreign";rows[1].clear();for(auto &c:cells){if(!rows[1].empty())rows[1]+=',';rows[1]+=c;}
     EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument)<<field;}
    rows=original;rows[0]="backend,phase,step,key,elements,exported_elements,file";
    EXPECT_THROW(validateSnapshotManifest(r,rows),std::invalid_argument);
}
TEST(RecordedHeadRequest,CanonicalQuotedPathsAndMalformedCSV)
{
    EXPECT_EQ(csvFields("a,\"b,c\",\"d\"\"e\""),(std::vector<std::string>{"a","b,c","d\"e"}));
    EXPECT_THROW(csvFields("a,\"unterminated"),std::invalid_argument);
    EXPECT_THROW(csvFields("a,\"closed\"junk"),std::invalid_argument);
    EXPECT_THROW(csvFields("a,unquoted\"quote"),std::invalid_argument);
}
}

namespace
{
/** @brief Authored metadata wrapper; its synthetic fixture is never a native receipt. */
json cohortFixture(int degree=4)
{
    auto source=fixture();source["slice_degree"]=degree;source["slice_participant"]=0;
    source["first_column"]=0;source["local_n"]=248320/degree;
    source["serial_equivalent_partition_width"]=62080;
    const std::vector<int> devices=degree==1?std::vector<int>{3}:degree==2?std::vector<int>{3,1}:std::vector<int>{3,1,7,0};
    source["device_ordinal"]=devices.front();
    return {{"schema","llaminar.recorded_head_cohort.v1"},{"scope","actual_tp"},
        {"operation_boundary","partitioned_projection_compute"},{"member_template",source},{"device_ordinals",devices}};
}
TEST(RecordedHeadCohort,DerivesCompleteOriginalSlicesOnExplicitNonordinalMembers)
{
    for(int degree:{1,2,4})
    {
        const auto j=cohortFixture(degree);const auto cohort=CohortRequest::fromJson(j);
        ASSERT_EQ(cohort.members().size(),std::size_t(degree));
        std::size_t next=0;
        for(int rank=0;rank<degree;++rank)
        {
            const auto &r=cohort.members().at(rank);
            EXPECT_EQ(r.ordinal,j["device_ordinals"][rank]);EXPECT_EQ(r.participant,rank);
            EXPECT_EQ(r.degree,degree);EXPECT_EQ(r.n,248320/degree);EXPECT_EQ(r.first,int(next));
            EXPECT_EQ(r.serial_width,62080);next+=std::size_t(r.n);
            auto expected=j["member_template"];expected["device_ordinal"]=r.ordinal;
            expected["slice_participant"]=rank;expected["first_column"]=r.first;
            EXPECT_EQ(r.document,expected);EXPECT_EQ(r.observation,cohort.source().observation);
        }
        EXPECT_EQ(next,248320u);EXPECT_EQ(cohort.document(),j);
    }
}
TEST(RecordedHeadCohort,WorkloadRetainsActualSourcePrecisionAndPolicyAcrossDegrees)
{
    const auto one=cohortWorkloadIdentity(CohortRequest::fromJson(cohortFixture(1)).source());
    for(int degree:{2,4})
    {
        const auto cohort=CohortRequest::fromJson(cohortFixture(degree));
        for(const auto &member:cohort.members())EXPECT_EQ(cohortWorkloadIdentity(member),one);
    }
    for(const char *field:{"generation","serial_equivalent_partition_width","model_context"})
    {
        auto j=cohortFixture();j["member_template"][field]=field==std::string("serial_equivalent_partition_width")?0:2;
        EXPECT_NE(cohortWorkloadIdentity(CohortRequest::fromJson(j).source()),one);
    }
    auto j=cohortFixture();j["member_template"]["six_precision_selectors"]["LLAMINAR_QFN_HC_UP_ACTIVATION"]="single_q8";
    EXPECT_NE(cohortWorkloadIdentity(CohortRequest::fromJson(j).source()),one);
    j=cohortFixture();j["member_template"]["input"]["mtime_ns"]=5;
    EXPECT_NE(cohortWorkloadIdentity(CohortRequest::fromJson(j).source()),one);
}
TEST(RecordedHeadCohort,RejectsIncompleteDuplicateOrAmbiguousCohorts)
{
    for(const auto &[key,value]:std::vector<std::pair<std::string,json>>{
        {"scope","local_shape"},{"operation_boundary","complete_lm_head"},{"schema","unknown"},
        {"device_ordinals",json::array({3,1})},{"device_ordinals",json::array({3,1,7,7})},
        {"device_ordinals",json::array({3,1,7,-1})},{"device_ordinals",json::array({3,1,7,0.5})}})
    {auto j=cohortFixture();j[key]=value;EXPECT_THROW(CohortRequest::fromJson(j),std::invalid_argument)<<key;}
    auto j=cohortFixture();j["unused_override"]=true;EXPECT_THROW(CohortRequest::fromJson(j),std::invalid_argument);
    j=cohortFixture();j["member_template"]["device_ordinal"]=0;EXPECT_THROW(CohortRequest::fromJson(j),std::invalid_argument);
    j=cohortFixture();j["member_template"]["slice_participant"]=1;j["member_template"]["first_column"]=62080;
    EXPECT_THROW(CohortRequest::fromJson(j),std::invalid_argument);
    for(const char *field:{"schema","scope","operation_boundary","member_template","device_ordinals"})
    {j=cohortFixture();j.erase(field);EXPECT_THROW(CohortRequest::fromJson(j),std::exception)<<field;}
    j=cohortFixture();j["member_template"]["input_origin"]="synthetic";
    EXPECT_THROW(CohortRequest::fromJson(j),std::invalid_argument);
}
TEST(RecordedHeadCohort,CompleteRankMaximumPrecedesAnyMedian)
{
    const std::array<CohortRankInterval,2> first{{{0,0,9},{0,1,1}}},second{{{1,1,9},{1,0,1}}},third{{{2,0,1},{2,1,1}}};
    EXPECT_EQ(cohortMaximum(first,2,0),9);EXPECT_EQ(cohortMaximum(second,2,1),9);EXPECT_EQ(cohortMaximum(third,2,2),1);
    const std::array<CohortRankInterval,4> all{{{7,3,4},{7,1,2},{7,0,1},{7,2,3}}};
    EXPECT_EQ(cohortMaximum(all,4,7),4);
    EXPECT_THROW(cohortMaximum(std::span(all).first(3),4,7),std::invalid_argument);
    EXPECT_THROW(cohortMaximum(all,4,6),std::invalid_argument);
    auto bad=all;bad[3].rank=0;EXPECT_THROW(cohortMaximum(bad,4,7),std::invalid_argument);
    bad=all;bad[0].rank=4;EXPECT_THROW(cohortMaximum(bad,4,7),std::invalid_argument);
    for(double value:{0.,-1.,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()})
    {bad=all;bad[0].latency_us=value;EXPECT_THROW(cohortMaximum(bad,4,7),std::invalid_argument);}
}
TEST(RecordedHeadCohort,ImmediateWorkerRejectionCannotStrandReadiness)
{
    std::promise<void> pending;auto live=pending.get_future();EXPECT_NO_THROW(requireQueuedCohortAdmission(live));
    std::promise<void> rejected;auto error=rejected.get_future();rejected.set_exception(std::make_exception_ptr(std::runtime_error("worker stopped")));
    EXPECT_THROW(requireQueuedCohortAdmission(error),std::runtime_error);
    std::promise<void> completed;auto early=completed.get_future();completed.set_value();
    EXPECT_THROW(requireQueuedCohortAdmission(early),std::logic_error);
    std::future<void> absent;EXPECT_THROW(requireQueuedCohortAdmission(absent),std::invalid_argument);
}
TEST(RecordedHeadCohort,AllMembersAreReadyBeforeControllerRelease)
{
    CohortRoundGate gate(2);std::promise<void> entered;
    auto first=std::async(std::launch::async,[&]{entered.set_value();gate.arriveAndWait(0);return true;});
    entered.get_future().wait();EXPECT_EQ(first.wait_for(std::chrono::milliseconds(0)),std::future_status::timeout);
    auto second=std::async(std::launch::async,[&]{gate.arriveAndWait(1);return true;});
    EXPECT_THROW(gate.releaseHostNanoseconds(),std::logic_error);
    gate.releaseWhenReady();EXPECT_GT(gate.releaseHostNanoseconds(),0);EXPECT_TRUE(first.get());EXPECT_TRUE(second.get());
    EXPECT_THROW(gate.releaseWhenReady(),std::logic_error);
}
TEST(RecordedHeadCohort,AdmissionFailureWakesQueuedMembersWithoutNativeSubmission)
{
    CohortRoundGate gate(2);std::promise<void> entered;std::atomic<int> submitted{0};
    auto task=std::async(std::launch::async,[&]
    {entered.set_value();try{gate.arriveAndWait(0);++submitted;return false;}catch(const std::exception &){return true;}});
    entered.get_future().wait();gate.abort();gate.abort();
    EXPECT_THROW(gate.releaseWhenReady(),std::runtime_error);EXPECT_TRUE(task.get());EXPECT_EQ(submitted,0);
}
TEST(RecordedHeadCohort,DuplicateAndForeignRanksAbortEveryWaitingMember)
{
    for(int second_rank:{0,2})
    {
        CohortRoundGate gate(2);std::promise<void> entered;
        auto task=std::async(std::launch::async,[&]
        {entered.set_value();try{gate.arriveAndWait(0);return false;}catch(const std::exception &){return true;}});
        entered.get_future().wait();
        // A same-rank call can beat the first callback to the mutex; either
        // call must abort, and both errors are observed without GPU execution.
        auto second=std::async(std::launch::async,[&]
        {try{gate.arriveAndWait(second_rank);return false;}catch(const std::exception &){return true;}});
        EXPECT_THROW(gate.releaseWhenReady(),std::runtime_error);EXPECT_TRUE(second.get());EXPECT_TRUE(task.get());
    }
}
} // namespace
