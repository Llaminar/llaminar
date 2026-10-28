/**
 * @file Test__MTPDepthTraceReplay.cpp
 * @brief Independent golden chronology and fail-closed offline controller replay.
 *
 * Authored edge cases and genuine captured TP4 inputs supply independent
 * decisions/endpoints, including real disjoint confirmation and trial outcomes.
 * Historical survival inputs retain their original unsupported frontier.
 * Compiled IDs are added only to explicitly synthetic
 * documents; recorded requests retain their producer IDs and capture receipts.
 */
#include "MTPDepthTraceReplay.h"
#include "fixtures/MTPDepthTraceGolden.h"
#include "fixtures/MTPDepthTraceRecorded.h"
#include "fixtures/MTPDepthTraceDirectionalRecorded.h"
#include "fixtures/MTPDepthTraceSurvivalRecorded.h"
#include "fixtures/MTPDepthTraceConfirmationRecorded.h"
#include "utils/MTPDepthTransactionTraceCPU.h"
#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <utility>
namespace llaminar2::test::mtp_trace_replay
{
namespace
{
using namespace sampling_math;
using Json=nlohmann::json;
/** @return Independently authored initial static and fresh resident state. */
std::array<int,kDeviceGenerationControlCount> initialControl()
{
    std::array<int,kDeviceGenerationControlCount> c{};
    c[kDeviceGenerationControlOk]=1;c[kDeviceGenerationControlRemainingTokenCount]=34;
    c[kDeviceGenerationControlCurrentDraftDepth]=3;c[kDeviceGenerationControlActiveVerifierRowCount]=4;
    c[kDeviceGenerationControlMinimumDraftDepth]=1;c[kDeviceGenerationControlMaximumDraftDepth]=5;
    c[kDeviceGenerationControlDepthPolicyMode]=static_cast<int>(DeviceGenerationPolicyMode::Dynamic);
    c[kDeviceGenerationControlDepthWindowSize]=4;c[kDeviceGenerationControlDepthMinimumSamples]=4;
    c[kDeviceGenerationControlDepthPromoteConsecutiveWindows]=1;
    c[kDeviceGenerationControlDepthPromoteFullAcceptRatePPM]=1000000;
    c[kDeviceGenerationControlDepthDemoteZeroAcceptRatePPM]=300000;
    c[kDeviceGenerationControlDepthDemoteAcceptanceRatePPM]=550000;
    c[kDeviceGenerationControlDepthLastRecommendedDepth]=3;c[kDeviceGenerationControlInitialDraftDepth]=3;
    return c;
}
/** @return Exact complete active host config, not device-derived PPM approximations. */
Json hostInitial()
{
    return {{"schema","llaminar.mtp.cpu_depth_transaction_trace_initial.v3"},{"compiled_identity",compiledIdentity()},
        {"verify_mode",0},{"configured_draft_tokens",3},{"initial_depth",3},{"initial_incoming_debt",false},{"policy",{
        {"mode",static_cast<int>(MTPDepthPolicyMode::Dynamic)},{"backend",0},{"model_class",0},
        {"scope",{{"verifier_width_policy",0},{"tp_width",0},{"pp_stages",0},{"retained_depth",0},{"hidden_dim",0},
            {"main_layers",0},{"expert_count",0},{"experts_per_token",0},{"compute_capability_major",0},{"compute_capability_minor",0}}},
        {"min_depth",1},{"max_depth",5},{"initial_depth",3},{"window_size",4},{"min_samples",4},
        {"cooldown_steps",0},{"promote_consecutive_windows",1},{"use_generated_policy",false},
        {"promote_full_accept_rate",1.0},{"demote_zero_accept_rate",.3},{"demote_acceptance_rate",.55}}}};
}
/** @return A complete synthetic golden request; no function under test authors its endpoints. */
Json request(bool host=false)
{
    Json rows=Json::array();int ordinal=0;
    for(const auto &r:host?golden::kHost:golden::kResident)
        rows.push_back({{"ordinal",++ordinal},{"packed",r.packed},{"token_witness",r.token_witness},{"evaluated_windows",r.evaluated_windows}});
    auto c=initialControl();auto terminal=c;
    terminal[kDeviceGenerationControlTransactionCount]=13;terminal[kDeviceGenerationControlResponseTokenCount]=34;
    terminal[kDeviceGenerationControlPublishedStateCommitCount]=33;
    terminal[kDeviceGenerationControlNextLeadingCommittedOutputCount]=1;
    terminal[kDeviceGenerationControlLastTransactionEmittedTokenCount]=1;
    terminal[kDeviceGenerationControlRemainingTokenCount]=0;terminal[kDeviceGenerationControlRequestComplete]=1;
    terminal[kDeviceGenerationControlDepthEvaluatedWindows]=3;terminal[kDeviceGenerationControlDepthUpdates]=2;
    terminal[kDeviceGenerationControlDepthPromotions]=1;terminal[kDeviceGenerationControlDepthDemotions]=1;
    return {{"schema","llaminar.mtp.depth_trace_replay_request.v3"},{"intent","baseline"},{"authority",host?"cpu_host":"native_resident"},
        {"compiled_identity",compiledIdentity()},{"capture",{{"observation_identity","authored_golden_fixture"},
            {"device",host?"CPU":"ROCm:0"},{"receipt_sha256",std::string(64,'1')},{"configuration_sha256",std::string(64,'2')}}},
        {"initial",host?hostInitial():Json{{"phase","validated_pre_first_response_append"},{"control",c}}},
        {"transactions",rows},{"terminal",host?Json{{"transactions",13},{"current_depth",3},{"evaluated_windows",3},{"emitted_tokens",34},{"published_state_tokens",33},{"model_stopped",false},{"utility_probes",0},{"utility_keeps",0},{"utility_returns",0}}
            :Json{{"transactions",13},{"control",terminal}}},{"output","/tmp/mtp-trace-golden-never-written-by-pure-replay.json"}};
}
/** @return Authored stopped-correction request with one eligible row and explicit outgoing debt.
 * This partial window has R1/A1; the stop itself contributes no adaptive sample.
 * The last correction has raw debt1 but no next carry, and ends before budget8.
 */
Json stoppedRequest(bool host)
{
    auto j=request(host);j["transactions"]=Json::array();int ordinal=0;
    for(const auto &r:host?golden::kStoppedHost:golden::kStoppedResident)
        j["transactions"].push_back({{"ordinal",++ordinal},{"packed",r.packed},{"token_witness",r.token_witness},{"evaluated_windows",r.evaluated_windows}});
    if(host)
    {
        j["terminal"]={{"transactions",2},{"current_depth",3},{"evaluated_windows",0},
            {"emitted_tokens",4},{"published_state_tokens",3},{"model_stopped",true},
            {"utility_probes",0},{"utility_keeps",0},{"utility_returns",0}};
    }
    else
    {
        auto c=initialControl();c[kDeviceGenerationControlRemainingTokenCount]=8;j["initial"]["control"]=c;
        c[kDeviceGenerationControlResponseTokenCount]=4;c[kDeviceGenerationControlRemainingTokenCount]=4;
        c[kDeviceGenerationControlRequestComplete]=1;c[kDeviceGenerationControlModelStopped]=1;
        c[kDeviceGenerationControlTransactionCount]=2;c[kDeviceGenerationControlPublishedStateCommitCount]=3;
        c[kDeviceGenerationControlNextLeadingCommittedOutputCount]=0;c[kDeviceGenerationControlLastTransactionEmittedTokenCount]=1;
        c[kDeviceGenerationControlDepthStepsSinceChange]=1;c[kDeviceGenerationControlDepthWindowVerifierRuns]=1;
        c[kDeviceGenerationControlDepthWindowAttemptedTokens]=3;c[kDeviceGenerationControlDepthWindowAcceptedTokens]=1;
        c[kDeviceGenerationControlDepthWindowRejectedTokens]=2;c[kDeviceGenerationControlDepthWindowRollbacks]=1;
        c[kDeviceGenerationControlDepthWindowAcceptedPrefixSum]=1;
        j["terminal"]={{"transactions",2},{"control",c}};
    }
    return j;
}
/** @brief Project captured words through the independently pinned ownership set.
 * @param terminal Immutable captured/authored terminal row, not replay-generated state.
 * @return Exact expected word/value pairs; omission of a controller owner fails tests.
 */
Json expectedResidentTerminalProjection(const Json &terminal)
{
    // Pin this explicit index set independently of production owned(word), so
    // accidentally narrowing that predicate cannot make the fixture self-pass.
    constexpr std::array<int,53> words{0,11,
        13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,
        46,47,48,49,50,51,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107};
    Json values=Json::array();
    for(int word:words)values.push_back({{"word",word},{"value",terminal.at(word)}});
    return {{"schema","llaminar.mtp.resident_controller_terminal_projection.v1"},{"words",values}};
}
/** @brief Compare independently specified decisions and counters, not a regenerated oracle. */
void expectGolden(const Json &result)
{
    EXPECT_EQ(result.at("status"),"complete_baseline_replay");EXPECT_EQ(result.at("transactions"),13);
    EXPECT_EQ(result.at("final_depth"),3);EXPECT_EQ(result.at("evaluated_windows"),3);
    EXPECT_EQ(result.at("updates"),2);EXPECT_EQ(result.at("promotions"),1);EXPECT_EQ(result.at("demotions"),1);
    ASSERT_EQ(result.at("windows").size(),3u);
    EXPECT_EQ(result.at("decisions")[1].at("budget_limited"),true);
    EXPECT_EQ(result.at("decisions")[1].at("accepted_prefix"),0);
    EXPECT_EQ(result.at("decisions")[1].at("rollback"),result.at("authority")=="native_resident");
    EXPECT_EQ(result.at("emitted_tokens"),34);EXPECT_EQ(result.at("published_state_tokens"),33);
    EXPECT_FALSE(result.at("model_stopped").get<bool>());
    EXPECT_EQ(result.at("decisions")[0].at("newly_emitted_tokens"),4);
    EXPECT_EQ(result.at("decisions")[0].at("published_state_tokens"),3);
    EXPECT_EQ(result.at("decisions")[2].at("newly_emitted_tokens"),3);
    EXPECT_EQ(result.at("decisions")[2].at("published_state_tokens"),4);
    const auto &w=result.at("windows");
    EXPECT_EQ(w[0].at("transaction"),5);EXPECT_EQ(w[0].at("runs"),4);EXPECT_EQ(w[0].at("accepted"),8);
    EXPECT_EQ(w[0].at("zero"),0);EXPECT_EQ(w[0].at("full"),1);EXPECT_EQ(w[0].at("new_depth"),3);
    EXPECT_EQ(w[1].at("transaction"),9);EXPECT_EQ(w[1].at("accepted"),12);EXPECT_EQ(w[1].at("full"),4);EXPECT_EQ(w[1].at("new_depth"),4);
    EXPECT_EQ(w[2].at("transaction"),13);EXPECT_EQ(w[2].at("accepted"),0);EXPECT_EQ(w[2].at("zero"),4);EXPECT_EQ(w[2].at("new_depth"),3);
}
}

TEST(MTPDepthTraceReplay, ResidentGoldenPreservesActualOrderAndWindows)
{
    expectGolden(replay(request()));
    auto other_producer=request();other_producer["terminal"]["control"][kDeviceGenerationControlQsaPriorHeaderBytesLow]=12345;
    expectGolden(replay(other_producer)); // Unrelated producers are explicitly outside this replay.
}
TEST(MTPDepthTraceReplay, HostGoldenUsesActualControllerAndRoundTripDoubles)
{
    auto j=request(true);
    j["initial"]["policy"]["demote_zero_accept_rate"]=std::nextafter(.3,1.0);
    const auto parsed=parseDocument(j.dump());
    EXPECT_EQ(parsed["initial"]["policy"]["demote_zero_accept_rate"].get<double>(),std::nextafter(.3,1.0));
    expectGolden(replay(parsed));
    MTPDepthPolicyConfig c;c.mode=MTPDepthPolicyMode::Dynamic;c.min_depth=1;c.max_depth=5;c.initial_depth=3;
    c.window_size=4;c.min_samples=4;c.cooldown_steps=0;c.promote_consecutive_windows=1;c.use_generated_policy=false;
    c.demote_zero_accept_rate=std::nextafter(.3,1.0);
    MTPDepthController owner(c,3);
    MTPPublicationWitness publication;ASSERT_TRUE(MTPPublicationWitness::fromCounts(4,0,3,false,&publication));
    const auto actual=cpuDepthTransactionTraceInitialToJson(owner,3,publication);
    EXPECT_EQ(actual.at("policy").at("demote_zero_accept_rate").get<double>(),std::nextafter(.3,1.0));
    auto actual_request=j;actual_request["initial"]=actual;expectGolden(replay(actual_request));
}
/** @test Genuine v2 TP4 captures remain immutable and cannot masquerade as publication v3.
 * They require the retained historical replay binary. New policy replay needs
 * genuine source-matched v3 captures; the authored golden does not replace them.
 */
TEST(MTPDepthTraceReplay, RecordedTP4RepetitionV2IsRejectedWithoutRestamping)
{
    for(const auto bytes:{recorded::kGreedyRepetition,recorded::kStochasticRepetition})
    {
        const auto original=parseDocument(bytes);ASSERT_EQ(original.at("schema"),"llaminar.mtp.depth_trace_replay_request.v2");
        const auto snapshot=original.dump();
        for(int repeat=0;repeat<3;++repeat)
        {
            try{(void)replay(original);FAIL()<<"v2 publication was silently reconstructed";}
            catch(const std::invalid_argument &error){EXPECT_NE(std::string(error.what()).find("unknown replay schema"),std::string::npos);}
            EXPECT_EQ(original.dump(),snapshot);
        }
    }
}
/** @test The genuine directional v2 request is rejected rather than padded with debt guesses. */
TEST(MTPDepthTraceReplay, RecordedTP4DownwardV2IsRejectedWithoutRestamping)
{
    const auto original=parseDocument(recorded::kStochasticLongPythonDirectional);
    ASSERT_EQ(original.at("schema"),"llaminar.mtp.depth_trace_replay_request.v2");
    const auto snapshot=original.dump();
    for(int repeat=0;repeat<3;++repeat)
    {
        EXPECT_THROW((void)replay(original),std::invalid_argument);
        EXPECT_EQ(original.dump(),snapshot);
    }
}

/** @test Recorded input framing and ABI identity fail closed without inventing missing work. */
TEST(MTPDepthTraceReplay, RecordedTP4RepetitionRejectsMissingAndIncompatibleEvidence)
{
    for (const auto text : {recorded::kGreedyRepetition, recorded::kStochasticRepetition})
    {
        auto truncated = parseDocument(text);
        truncated["transactions"].erase(truncated["transactions"].end() - 1);
        EXPECT_THROW((void)replay(truncated), std::invalid_argument);
        auto incompatible = parseDocument(text);
        incompatible["compiled_identity"]["sources"]["control_layout"] = std::string(64, '0');
        EXPECT_THROW((void)replay(incompatible), std::invalid_argument);
        auto missing_initial = parseDocument(text);
        missing_initial["initial"].erase("control");
        EXPECT_THROW((void)replay(missing_initial), std::invalid_argument);
    }
}
TEST(MTPDepthTraceReplay, FixedClippedAndFreshResetStayExact)
{
    for(bool host:{false,true})
    {
        auto j=request(host);j["transactions"].erase(j["transactions"].begin()+3,j["transactions"].end());
        j["terminal"]["transactions"]=3;
        if(host)
        {
            j["initial"]["policy"]["mode"]=static_cast<int>(MTPDepthPolicyMode::Fixed);
            j["initial"]["policy"]["min_depth"]=3;j["initial"]["policy"]["max_depth"]=3;
            j["terminal"]["evaluated_windows"]=0;j["terminal"]["emitted_tokens"]=8;
            j["terminal"]["published_state_tokens"]=8;
        }
        else
        {
            auto &c=j["initial"]["control"];c[kDeviceGenerationControlRemainingTokenCount]=8;
            c[kDeviceGenerationControlDepthPolicyMode]=static_cast<int>(DeviceGenerationPolicyMode::Fixed);
            c[kDeviceGenerationControlMinimumDraftDepth]=3;c[kDeviceGenerationControlMaximumDraftDepth]=3;
            auto t=c;t[kDeviceGenerationControlTransactionCount]=3;t[kDeviceGenerationControlResponseTokenCount]=8;
            t[kDeviceGenerationControlRemainingTokenCount]=0;t[kDeviceGenerationControlRequestComplete]=1;
            t[kDeviceGenerationControlPublishedStateCommitCount]=8;t[kDeviceGenerationControlLastTransactionEmittedTokenCount]=3;
            j["terminal"]["control"]=t;
        }
        const auto first=replay(j);EXPECT_EQ(first.at("evaluated_windows"),0);EXPECT_EQ(first.at("transactions"),3);
        EXPECT_EQ(first,replay(j)); // No controller state can leak across independent traces.
    }
}
/** @test Observe records retain actual recommendations while the applied selector stays fixed. */
TEST(MTPDepthTraceReplay, ObserveGoldenKeepsRecommendationsSeparateFromAppliedDepth)
{
    for(bool host:{false,true})
    {
        auto j=request(host);
        for(size_t index=0;index<j["transactions"].size();++index)
        {
            auto &row=j["transactions"][index];
            MTPDepthTransactionTraceRecord raw{row["packed"].get<uint32_t>(),row["token_witness"].get<int32_t>(),row["evaluated_windows"].get<int32_t>()};
            MTPDepthTransactionTraceValues v;ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(raw,&v));
            // These observed selectors/recommendations are authored expectations,
            // not outputs generated by either production controller.
            v.pre_depth=3;v.post_depth=3;
            // Host non-evaluations return the current selector; the resident
            // row retains its previous evaluated recommendation between windows.
            v.recommended_depth=host?(index==8?4:index==12?2:3):(index<8?3:index<12?4:2);
            if(host){v.requested_depth=3;if(v.effective_depth>0)v.effective_depth=3;}
            ASSERT_TRUE(packMTPDepthTransactionTraceRecord(v,&raw));row["packed"]=raw.packed;
        }
        if(host)
        {
            j["initial"]["policy"]["mode"]=static_cast<int>(MTPDepthPolicyMode::Observe);
            j["terminal"]["current_depth"]=3;
        }
        else
        {
            j["initial"]["control"][kDeviceGenerationControlDepthPolicyMode]=static_cast<int>(DeviceGenerationPolicyMode::Observe);
            auto &t=j["terminal"]["control"];t[kDeviceGenerationControlDepthPolicyMode]=static_cast<int>(DeviceGenerationPolicyMode::Observe);
            t[kDeviceGenerationControlDepthUpdates]=0;t[kDeviceGenerationControlDepthPromotions]=0;t[kDeviceGenerationControlDepthDemotions]=0;
            t[kDeviceGenerationControlDepthStepsSinceChange]=12;t[kDeviceGenerationControlDepthLastRecommendedDepth]=2;
        }
        const auto result=replay(j);EXPECT_EQ(result["final_depth"],3);EXPECT_EQ(result["updates"],0);
        EXPECT_EQ(result["windows"][1]["recommended_depth"],4);EXPECT_EQ(result["windows"][1]["new_depth"],3);
        EXPECT_EQ(result["windows"][2]["recommended_depth"],2);EXPECT_EQ(result["windows"][2]["new_depth"],3);
    }
}
TEST(MTPDepthTraceReplay, RejectsMissingCorruptAndOutOfScopeInput)
{
    for(int bad=0;bad<12;++bad)
    {
        SCOPED_TRACE(bad);auto j=request();
        if(bad==0)j.erase("initial");
        if(bad==1)j["transactions"][0]["packed"]=true;
        if(bad==2)j["transactions"][0]["packed"]=3.0;
        if(bad==3)j["transactions"][0]["packed"]="3";
        if(bad==4)j["transactions"][0]["packed"]=uint64_t{1}<<32;
        if(bad==5)j["transactions"][0]["packed"]=golden::kResident[0].packed|(1u<<29);
        if(bad==6)j["transactions"][1]["ordinal"]=1;
        if(bad==7)j["transactions"][0]["ordinal"]=0;
        if(bad==8)j["initial"]["control"][kDeviceGenerationControlCurrentDraftDepth]=0;
        if(bad==9)j["transactions"][1]["token_witness"]=3;
        if(bad==10)j["transactions"][12]["token_witness"]=35;
        if(bad==11)j["unexpected"]=1;
        EXPECT_THROW((void)replay(j),std::exception);
    }
    for(bool host:{false,true})
    {
        auto hostile=request(host);
        if(host)hostile["initial"]["policy"]["cooldown_steps"]=INT32_MAX;
        else
        {
            hostile["initial"]["control"][kDeviceGenerationControlDepthCooldownSteps]=INT32_MAX;
            hostile["initial"]["control"][kDeviceGenerationControlDepthStepsSinceChange]=INT32_MAX;
            hostile["terminal"]["control"][kDeviceGenerationControlDepthCooldownSteps]=INT32_MAX;
        }
        try { (void)replay(hostile);FAIL()<<"Unsafe signed counter accepted"; }
        catch(const std::invalid_argument &error)
        { EXPECT_NE(std::string(error.what()).find("cooldown exceeds safe finite replay horizon"),std::string::npos); }
    }
    EXPECT_THROW((void)parseDocument("{\"x\":1,\"x\":2}"),std::exception);
}
TEST(MTPDepthTraceReplay, RejectsCompiledPolicyAndConfigurationIdentityMismatch)
{
    auto changed=request();auto key=changed["compiled_identity"]["sources"].begin().key();
    changed["compiled_identity"]["sources"][key]=std::string(64,'0');EXPECT_THROW((void)replay(changed),std::exception);
    auto candidate=request();candidate["intent"]="candidate";candidate["compiled_identity"]["sources"]["policy_table"]=std::string(64,'0');
    const auto candidate_result=replay(candidate);EXPECT_EQ(candidate_result.at("status"),"same_observed_depth_path");
    EXPECT_FALSE(candidate_result.at("baseline_validated_by_this_executable").get<bool>());
    candidate["compiled_identity"]["sources"]["control_layout"]=std::string(64,'0');EXPECT_THROW((void)replay(candidate),std::exception);
    changed=request();changed["terminal"]["control"][kDeviceGenerationControlDepthMinimumSamples]=5;
    EXPECT_THROW((void)replay(changed),std::exception);
    changed=request();changed["initial"]["control"][kDeviceGenerationControlLearnedScopeTPWidth]=4;
    EXPECT_THROW((void)replay(changed),std::exception);
    changed=request(true);changed["initial"]["initial_depth"]=4;EXPECT_THROW((void)replay(changed),std::exception);
    changed=request(true);changed["initial"]["policy"]["demote_acceptance_rate"]="0.55";EXPECT_THROW((void)replay(changed),std::exception);
    changed=request();changed["initial"]["control"][kDeviceGenerationControlRemainingTokenCount]=35;
    changed["terminal"]["control"][kDeviceGenerationControlRemainingTokenCount]=1;
    EXPECT_THROW((void)replay(changed),std::exception);
    changed["terminal"]["control"][kDeviceGenerationControlModelStopped]=1;
    EXPECT_THROW((void)replay(changed),std::exception);
}
TEST(MTPDepthTraceReplay, RejectsTruncatedAndWrongCPUFinalWitnesses)
{
    for(bool host:{false,true})
    {
        auto j=request(host);j["transactions"].erase(j["transactions"].end()-1);EXPECT_THROW((void)replay(j),std::exception);
    }
    auto j=request(true);j["terminal"]["current_depth"]=4;EXPECT_THROW((void)replay(j),std::exception);
    j=request(true);j["terminal"]["evaluated_windows"]=2;EXPECT_THROW((void)replay(j),std::exception);
    j=request(true);j["terminal"]["transactions"]=12;EXPECT_THROW((void)replay(j),std::exception);
}
TEST(MTPDepthTraceReplay, CandidateStopsAtFirstUnsupportedDepthWithoutFutureFeedback)
{
    auto j=request();DeviceGenerationPolicy p;p.initial_depth=3;p.minimum_depth=1;p.maximum_depth=5;
    p.mode=DeviceGenerationPolicyMode::Dynamic;p.window_size=4;p.minimum_samples=4;p.cooldown_steps=0;
    p.promote_consecutive_windows=1;p.learned.enabled=false;p.promote_full_accept_rate_ppm=0;
    const auto result=replayResidentCandidate(j,p);
    EXPECT_EQ(result.at("status"),"unsupported_candidate_depth");EXPECT_EQ(result.at("consumed_transactions"),5);
    EXPECT_EQ(result.at("first_divergence_transaction"),5);EXPECT_EQ(result.at("expected_depth"),3);EXPECT_EQ(result.at("candidate_depth"),4);
    EXPECT_EQ(result.at("decisions").size(),5u);
    auto unsafe=p;unsafe.cooldown_steps=INT32_MAX;
    EXPECT_THROW((void)replayResidentCandidate(j,unsafe),std::invalid_argument);
    for(int changed=0;changed<4;++changed)
    {
        auto other=p;
        if(changed==0)other.learned.backend=MTPDepthPolicyBackend::CUDA;
        if(changed==1)other.learned.model_class=MTPDepthPolicyModelClass::Dense;
        if(changed==2)other.learned.verify_mode=MTPVerifyMode::SpeculativeSampling;
        if(changed==3)other.learned.scope={.verifier_width_policy=MTPVerifierPhysicalWidthPolicy::ExactLogicalRows,
            .tp_width=4,.pp_stages=1,.retained_depth=15,.hidden_dim=2560,.main_layers=48,.expert_count=512,
            .experts_per_token=10,.compute_capability_major=90,.compute_capability_minor=6};
        EXPECT_THROW((void)replayResidentCandidate(j,other),std::exception);
    }
}
/** @test The exact endpoint projection includes every resident owner and excludes other producers. */
TEST(MTPDepthTraceReplay, ResidentTerminalProjectionIsCompleteAndRepeatable)
{
    auto j=request();
    const auto expected=expectedResidentTerminalProjection(j.at("terminal").at("control"));
    const auto result=replay(j);
    EXPECT_EQ(result.at("controller_terminal_projection"),expected);
    EXPECT_EQ(replay(j).at("controller_terminal_projection"),expected);
    ASSERT_EQ(result.at("controller_terminal_projection").at("words").size(),53u);
    j["terminal"]["control"][kDeviceGenerationControlQsaPriorHeaderBytesLow]=12345;
    EXPECT_EQ(replay(j).at("controller_terminal_projection"),expected);
    // CPU authority owns a different configuration/state contract, never a
    // fabricated resident 108-word row.
    EXPECT_FALSE(replay(request(true)).contains("controller_terminal_projection"));
}
/** @test Candidate endpoint evidence does not silently impose origin state equality. */
TEST(MTPDepthTraceReplay, ResidentTerminalProjectionPreservesCandidateSemantics)
{
    auto j=request();
    j["intent"]="candidate";
    j["compiled_identity"]["sources"]["policy_table"]=std::string(64,'0');
    const auto producer=j.at("compiled_identity");
    // Same selector path can have a different endpoint. A recorded regression
    // must explicitly compare this evidence; ordinary candidate replay may not.
    j["terminal"]["control"][kDeviceGenerationControlDepthUtilityProbes]=1;
    const auto result=replay(j);
    EXPECT_EQ(result.at("status"),"same_observed_depth_path");
    EXPECT_EQ(result.at("origin_compiled_identity"),producer);
    EXPECT_FALSE(result.at("baseline_validated_by_this_executable").get<bool>());
    EXPECT_FALSE(result.at("economy_claim").get<bool>());
    EXPECT_NE(result.at("controller_terminal_projection"),
        expectedResidentTerminalProjection(j.at("terminal").at("control")));
    EXPECT_EQ(result.at("controller_terminal_projection"),
        expectedResidentTerminalProjection(request().at("terminal").at("control")));
    j["intent"]="baseline";
    j["compiled_identity"]=compiledIdentity();
    EXPECT_THROW((void)replay(j),std::exception);
}
/** @test A first unsupported selector has no fictitious final state or future feedback. */
TEST(MTPDepthTraceReplay, ResidentTerminalProjectionAbsentAtUnsupportedFrontier)
{
    auto j=request();DeviceGenerationPolicy p;p.initial_depth=3;p.minimum_depth=1;p.maximum_depth=5;
    p.mode=DeviceGenerationPolicyMode::Dynamic;p.window_size=4;p.minimum_samples=4;p.cooldown_steps=0;
    p.promote_consecutive_windows=1;p.learned.enabled=false;p.promote_full_accept_rate_ppm=0;
    const auto result=replayResidentCandidate(j,p);
    EXPECT_EQ(result.at("status"),"unsupported_candidate_depth");
    EXPECT_EQ(result.at("consumed_transactions"),5);
    EXPECT_EQ(result.at("decisions").size(),5u);
    EXPECT_FALSE(result.contains("controller_terminal_projection"));
}
/** @test Exact V/Y endpoints preserve carry and exclude a stopped correction from the partial window. */
TEST(MTPDepthTraceReplay, PublicationV3StopCarryAndPartialWindowRemainExact)
{
    for(bool host:{false,true})
    {
        const auto j=stoppedRequest(host);Json previous;
        for(int repeat=0;repeat<3;++repeat)
        {
            const auto result=replay(j);
            EXPECT_EQ(result.at("emitted_tokens"),4);EXPECT_EQ(result.at("published_state_tokens"),3);
            EXPECT_TRUE(result.at("model_stopped").get<bool>());EXPECT_EQ(result.at("evaluated_windows"),0);
            EXPECT_TRUE(result.at("windows").empty());EXPECT_EQ(result.at("updates"),0);
            const auto &last=result.at("decisions").at(1);
            EXPECT_EQ(last.at("newly_emitted_tokens"),1);EXPECT_EQ(last.at("published_state_tokens"),1);
            EXPECT_TRUE(last.at("incoming_debt").get<bool>());EXPECT_TRUE(last.at("outgoing_debt").get<bool>());
            EXPECT_TRUE(last.at("model_stopped").get<bool>());EXPECT_FALSE(last.at("rollback").get<bool>());
            if(host)EXPECT_EQ(last.at("reason"),toString(MTPDepthDecisionReason::ModelStopped));
            if(repeat)EXPECT_EQ(result,previous);previous=result;
            expectGolden(replay(request(host))); // A stopped partial window never leaks into a fresh request.
        }
    }
}
/** @test Publication and emission totals are independent inputs, and stop/debt cannot be guessed. */
TEST(MTPDepthTraceReplay, PublicationV3RejectsMissingDebtAndIndependentEndpointCorruption)
{
    for(bool host:{false,true})for(int defect=0;defect<7;++defect)
    {
        SCOPED_TRACE(host);
        SCOPED_TRACE(defect);
        auto j=stoppedRequest(host);
        if(defect==0)
        {
            if(host)j["terminal"]["published_state_tokens"]=4;
            else j["terminal"]["control"][kDeviceGenerationControlPublishedStateCommitCount]=4;
        }
        if(defect==1)
        {
            if(host)j["terminal"]["emitted_tokens"]=5;
            else j["terminal"]["control"][kDeviceGenerationControlResponseTokenCount]=5;
        }
        if(defect==2)
        {
            if(host)j["terminal"]["model_stopped"]=false;
            else j["terminal"]["control"][kDeviceGenerationControlModelStopped]=0;
        }
        if(defect==3)j["transactions"][1]["packed"]=j["transactions"][1]["packed"].get<uint32_t>()^(uint32_t{1}<<29);
        if(defect==4)j["transactions"][0]["packed"]=j["transactions"][0]["packed"].get<uint32_t>()|(uint32_t{1}<<31);
        if(defect==5)
        {
            if(host)j["terminal"].erase("published_state_tokens");
            else j["terminal"]["control"][kDeviceGenerationControlNextLeadingCommittedOutputCount]=1;
        }
        if(defect==6)j["transactions"].erase(j["transactions"].end()-1);
        EXPECT_THROW((void)replay(j),std::invalid_argument);
    }
    auto missing=request(true);missing["initial"].erase("initial_incoming_debt");
    EXPECT_THROW((void)replay(missing),std::invalid_argument);
    missing=request(true);missing["terminal"]["model_stopped"]=0;
    EXPECT_THROW((void)replay(missing),std::invalid_argument);
}
TEST(MTPDepthTraceReplay, CLIWritesFreshResultWithoutSDKModelsOrAccelerators)
{
    const auto directory=std::filesystem::temp_directory_path()/("llaminar-mtp-replay-"+std::to_string(getpid()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    struct Cleanup{std::filesystem::path p;~Cleanup(){std::error_code error;std::filesystem::remove_all(p,error);}} cleanup{directory};
    auto j=request();j["output"]=(directory/"result.json").string();const auto input=directory/"request.json";
    {std::ofstream out(input);out<<j.dump();}
    std::string executable="replay",flag="--replay-depth-trace",path=input.string();
    char *argv[]={executable.data(),flag.data(),path.data()};EXPECT_EQ(runCLI(3,argv),0);
    ASSERT_TRUE(std::filesystem::is_regular_file(directory/"result.json"));EXPECT_NE(runCLI(3,argv),0);
    for(int defect=0;defect<2;++defect)
    {
        auto invalid=request();invalid["output"]=(directory/("invalid-"+std::to_string(defect)+".json")).string();
        if(defect==0)invalid["schema"]="llaminar.mtp.depth_trace_replay_request.v2";
        else invalid["transactions"].erase(invalid["transactions"].end()-1);
        {std::ofstream out(input);out<<invalid.dump();}
        EXPECT_NE(runCLI(3,argv),0);EXPECT_FALSE(std::filesystem::exists(invalid["output"].get<std::string>()));
    }
}
}

namespace llaminar2::test::mtp_trace_replay
{
/** @test The compact ABI, finite admission and exact byte contract remain total. */
TEST(MTPDepthTraceReplay, CodecAdmissionAndExactExtentsRemainTotal)
{
    static_assert(sizeof(MTPDepthTransactionTraceRecord)==12);
    static_assert(sizeof(MTPDepthTransactionTrace)==3520);
    static_assert(sizeof(MTPDiagnosticRecordStorage)==3912);
    static_assert(alignof(MTPDiagnosticRecordStorage)==8);
    static_assert(kDeviceGenerationControlCount==108);
    MTPDiagnosticRecordStorage destination;EXPECT_EQ(destination.depth_transaction_trace.magic,0u);
    EXPECT_EQ(mtpDepthTransactionTraceBytes(0),0u);
    EXPECT_EQ(mtpDepthTransactionTraceBytes(1),offsetof(MTPDepthTransactionTrace,records)+12);
    EXPECT_EQ(mtpDepthTransactionTraceBytes(256),sizeof(MTPDepthTransactionTrace));
    EXPECT_THROW(mtpDepthTransactionTraceBytes(-1),std::out_of_range);
    EXPECT_THROW(mtpDepthTransactionTraceBytes(257),std::out_of_range);
    for(bool phase:{false,true})for(bool mirror:{false,true})for(bool trace:{false,true})
    {
        if(trace&&(phase||mirror))EXPECT_THROW(resolveMTPDiagnosticRecordMode(phase,mirror,trace),std::invalid_argument);
        else EXPECT_EQ(resolveMTPDiagnosticRecordMode(phase,mirror,trace),trace?MTPDiagnosticRecordMode::DepthTransactionTrace:
            phase||mirror?MTPDiagnosticRecordMode::FirstTransaction:MTPDiagnosticRecordMode::Disabled);
    }
    EXPECT_TRUE(validMTPDepthTransactionTraceScope(MTPDepthTransactionTraceScope::BoundedBenchmark,0,1));
    EXPECT_FALSE(validMTPDepthTransactionTraceScope(MTPDepthTransactionTraceScope::Other,0,1));
    EXPECT_FALSE(validMTPDepthTransactionTraceScope(MTPDepthTransactionTraceScope::BoundedBenchmark,1,1));
    EXPECT_FALSE(validMTPDepthTransactionTraceScope(MTPDepthTransactionTraceScope::BoundedBenchmark,0,0));
    EXPECT_TRUE(validMTPDepthTransactionTraceAdmission(DeviceGenerationAdmissionKind::NewRequest,1,1));
    EXPECT_TRUE(validMTPDepthTransactionTraceAdmission(DeviceGenerationAdmissionKind::NewRequest,1,256));
    EXPECT_FALSE(validMTPDepthTransactionTraceAdmission(DeviceGenerationAdmissionKind::ContinueResponse,1,256));
    for(int n:{-1,0,257})EXPECT_FALSE(validMTPDepthTransactionTraceAdmission(DeviceGenerationAdmissionKind::NewRequest,1,n));
    EXPECT_FALSE(validMTPDepthTransactionTraceAdmission(DeviceGenerationAdmissionKind::NewRequest,2,256));
    for(int depth=1;depth<=15;++depth)for(int prefix=0;prefix<=depth;++prefix)
    for(int post=1;post<=15;++post)for(bool clipped:{false,true})for(bool host:{false,true})
    {
        const MTPDepthTransactionTraceValues input{.pre_depth=depth,.accepted_prefix=prefix,.post_depth=post,
            .rollback=prefix<depth,.budget_limited=clipped,.incoming_debt=true,.outgoing_debt=true,.model_stopped=true,
            .requested_depth=host?depth:0,.effective_depth=host?depth:0,.host_observation=host,
            .recommended_depth=post,.token_witness=host?16:256,.evaluated_windows=256};
        MTPDepthTransactionTraceRecord packed{};ASSERT_TRUE(packMTPDepthTransactionTraceRecord(input,&packed));
        MTPDepthTransactionTraceValues decoded;ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(packed,&decoded));
        EXPECT_EQ(decoded.pre_depth,depth);EXPECT_EQ(decoded.accepted_prefix,prefix);EXPECT_EQ(decoded.post_depth,post);
        EXPECT_EQ(decoded.rollback,prefix<depth);EXPECT_EQ(decoded.budget_limited,clipped);EXPECT_EQ(decoded.host_observation,host);
        EXPECT_EQ(decoded.requested_depth,host?depth:0);EXPECT_EQ(decoded.effective_depth,host?depth:0);
        EXPECT_EQ(decoded.token_witness,host?16:256);EXPECT_EQ(decoded.evaluated_windows,256);
    }
    for(int field=0;field<9;++field)
    {
        MTPDepthTransactionTraceValues v{.pre_depth=3,.accepted_prefix=2,.post_depth=3,.rollback=true,
            .recommended_depth=3,.token_witness=3,.evaluated_windows=0};
        if(field==0)v.pre_depth=16;if(field==1)v.accepted_prefix=4;if(field==2)v.post_depth=0;
        if(field==3)v.recommended_depth=16;if(field==4)v.requested_depth=1;
        if(field==5)v.token_witness=0;if(field==6)v.token_witness=257;
        if(field==7)v.evaluated_windows=-1;if(field==8)v.evaluated_windows=257;
        MTPDepthTransactionTraceRecord unchanged{123,456,7};
        EXPECT_FALSE(packMTPDepthTransactionTraceRecord(v,&unchanged));
        EXPECT_EQ(unchanged.packed,123u);EXPECT_EQ(unchanged.token_witness,456);EXPECT_EQ(unchanged.evaluated_windows,7);
    }
    for(int bit:{29,30,31})
    {
        auto raw=golden::kResident[0];raw.packed|=uint32_t{1}<<bit;
        MTPDepthTransactionTraceValues out;out.pre_depth=7;
        ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(raw,&out));
        EXPECT_TRUE(bit==29?out.incoming_debt:bit==30?out.outgoing_debt:out.model_stopped);
    }
}
/** @test Actual append records fixed transactions and lazily replaces only active evidence. */
TEST(MTPDepthTraceReplay, ProducerAppendFreshResetLeavesUnusedTailUntouched)
{
    std::array<int,kDeviceGenerationControlCount> c{};std::array<int32_t,16> response{};
    const int32_t tokens[]{10,11,12};std::array<int,kSpeculativeBatchMetaCount> meta{};
    auto p=DeviceGenerationPolicy::fixed(2);MTPDepthTransactionTrace trace;
    std::memset(&trace,0x5a,sizeof(trace));
    auto append=[&]()
    {
        EXPECT_EQ(prepare_device_generation_transaction_budget(3,3,c.data()),3);
        meta.fill(0);meta[kSpecBatchMetaOk]=1;meta[kSpecBatchMetaReadyToken]=-1;meta[kSpecBatchMetaOutputCount]=3;
        meta[kSpecBatchMetaAcceptedSpeculativePrefix]=1;meta[kSpecBatchMetaTargetVerifierStateCommitCount]=2;
        meta[kSpecBatchMetaConsumedVerifierRows]=2;meta[kSpecBatchMetaAllSpeculativeAccepted]=0;meta[kSpecBatchMetaSampledTerminal]=0;
        meta[kSpecBatchMetaLeadingCommittedOutputCount]=c[kDeviceGenerationControlNextLeadingCommittedOutputCount];
        return append_speculative_outcome_to_device_generation(tokens,3,meta.data(),kSpeculativeBatchMetaCount,
            response.data(),16,c.data(),&trace);
    };
    ASSERT_TRUE(initialize_device_generation_control(12,16,p,c.data()));ASSERT_TRUE(append());ASSERT_TRUE(append());
    EXPECT_EQ(trace.initial_control[kDeviceGenerationControlTransactionCount],0);
    EXPECT_EQ(trace.initial_control[kDeviceGenerationControlResponseTokenCount],0);
    MTPDepthTransactionTraceValues v;ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(trace.records[0],&v));
    EXPECT_EQ(v.pre_depth,2);EXPECT_EQ(v.post_depth,2);EXPECT_EQ(v.accepted_prefix,1);EXPECT_EQ(v.evaluated_windows,0);
    EXPECT_FALSE(v.incoming_debt);EXPECT_TRUE(v.outgoing_debt);EXPECT_FALSE(v.model_stopped);
    ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(trace.records[1],&v));EXPECT_TRUE(v.incoming_debt);
    EXPECT_TRUE(v.outgoing_debt);EXPECT_EQ(c[kDeviceGenerationControlPublishedStateCommitCount],4);
    EXPECT_EQ(c[kDeviceGenerationControlResponseTokenCount],5);
    const auto stale=trace.records[1];
    ASSERT_TRUE(initialize_device_generation_control(12,16,p,c.data()));ASSERT_TRUE(append());
    EXPECT_EQ(c[kDeviceGenerationControlTransactionCount],1);
    EXPECT_EQ(trace.records[1].packed,stale.packed);EXPECT_EQ(trace.records[1].token_witness,stale.token_witness);
    const auto *tail=reinterpret_cast<const unsigned char *>(&trace.records[2]);
    const auto *end=reinterpret_cast<const unsigned char *>(&trace)+sizeof(trace);
    for(;tail!=end;++tail)EXPECT_EQ(*tail,0x5a);
}
/** @test V3 rejects v2/v1 wire and incomplete resident/publication state. */
TEST(MTPDepthTraceReplay, UtilityTraceV3RejectsHistoricalABIAndTokenUnits)
{
    for(const char* schema:{"llaminar.mtp.depth_trace_replay_request.v1","llaminar.mtp.depth_trace_replay_request.v2"})
    {
        auto historical=request();historical["schema"]=schema;
        EXPECT_THROW((void)replay(historical),std::exception);
    }
    auto short_control=request();short_control["initial"]["control"].erase(
        short_control["initial"]["control"].begin()+100,short_control["initial"]["control"].end());
    EXPECT_THROW((void)replay(short_control),std::exception);
    auto host=request(true);host["transactions"][0]["token_witness"]=-1;
    EXPECT_THROW((void)replay(host),std::exception);
    host=request(true);host["transactions"][0]["response_token_count"]=3;
    EXPECT_THROW((void)replay(host),std::exception);
}
/** @test Actual host emission is lossless and independently bounded by production output totals. */
TEST(MTPDepthTraceReplay, UtilityTraceV3UsesActualCPUEmissionAndIndependentTerminal)
{
    auto actual=request(true);
    // A rejection can emit a correction as well as the accepted prefix. This
    // authored input is deliberately different from accepted_prefix+1.
    ASSERT_EQ(actual["transactions"][3]["token_witness"],3);
    ASSERT_EQ(actual["terminal"]["emitted_tokens"],34);
    ASSERT_EQ(actual["terminal"]["published_state_tokens"],33);
    const auto result=replay(actual);
    EXPECT_EQ(result["decisions"][3]["accepted_prefix"],1);
    EXPECT_EQ(result["decisions"][3]["newly_emitted_tokens"],3);
    EXPECT_EQ(result["decisions"][3]["token_witness_scope"],"cpu_transaction_emitted");
    actual["terminal"]["emitted_tokens"]=33;
    EXPECT_THROW((void)replay(actual),std::exception);
    actual=request(true);actual["terminal"]["utility_probes"]=1;
    EXPECT_THROW((void)replay(actual),std::exception);
    actual=request(true);actual["transactions"].erase(actual["transactions"].end()-1);
    EXPECT_THROW((void)replay(actual),std::exception);
}
/** @test Every appended learner/reporting word belongs to exact baseline replay. */
TEST(MTPDepthTraceReplay, UtilityTraceV3ValidatesCompleteFinalStateAndCounters)
{
    for(int word=kDeviceGenerationControlDepthUtilityPhase;word<=kDeviceGenerationControlDepthUtilityReturns;++word)
    {
        auto altered=request();altered["terminal"]["control"][word]=1;
        EXPECT_THROW((void)replay(altered),std::exception);
        altered=request();altered["initial"]["control"][word]=1;
        EXPECT_THROW((void)replay(altered),std::exception);
    }
}
/** @test The actual CPU producer requires an explicit emitted delta, never A+R. */
TEST(MTPDepthTraceReplay, UtilityTraceV3CPUProducerRequiresActualEmission)
{
    MTPDepthPolicyConfig config;config.mode=MTPDepthPolicyMode::Fixed;config.initial_depth=3;
    config.min_depth=3;config.max_depth=3;config.use_generated_policy=false;
    MTPDepthController controller(config,3);
    MTPDepthObservation observation{.requested_depth=3,.effective_depth=3,.accepted_speculative_prefix=1,
        .budget_limited=false,.rollback=false,.newly_emitted_tokens=3u};
    MTPPublicationWitness publication;ASSERT_TRUE(MTPPublicationWitness::fromCounts(3,0,2,false,&publication));
    observation.publication=publication;
    const auto decision=controller.recordStep(observation);
    const auto raw=makeCPUDepthTransactionTraceRecord(observation,decision,1,0);
    MTPDepthTransactionTraceValues decoded;ASSERT_TRUE(unpackMTPDepthTransactionTraceRecord(raw,&decoded));
    EXPECT_EQ(decoded.token_witness,3);EXPECT_TRUE(decoded.host_observation);
    const auto actual_publication=observation.publication;observation.publication.reset();
    EXPECT_THROW((void)makeCPUDepthTransactionTraceRecord(observation,decision,1,0),std::invalid_argument);
    observation.publication=actual_publication;
    observation.newly_emitted_tokens.reset();
    EXPECT_THROW((void)makeCPUDepthTransactionTraceRecord(observation,decision,1,0),std::invalid_argument);
    observation.newly_emitted_tokens=0u;
    EXPECT_THROW((void)makeCPUDepthTransactionTraceRecord(observation,decision,1,0),std::invalid_argument);
    observation.newly_emitted_tokens=5u;
    EXPECT_THROW((void)makeCPUDepthTransactionTraceRecord(observation,decision,1,0),std::invalid_argument);
}

/** @test Historical survival v3 source identities and feedback remain immutable.
 * The original controller selected1/6 atT4; confirmation keeps5 instead.
 * Candidate replay consumes exactly that common prefix and stops before any
 * next-depth observation. This is not a new baseline or a correctness exemption:
 * the new authored confirmation sequence and fresh captured traces own new proof.
 */
TEST(MTPDepthTraceReplay, HistoricalSurvivalV3RejectsBaselineAndStopsAtConfirmationFrontier)
{
    using namespace recorded_survival_v3;
    for(int repeat=0;repeat<3;++repeat)
    for(const auto bytes:{kGreedyTrainPython,kStochasticTrainPython})
    {
        const auto original=parseDocument(bytes);const auto snapshot=original.dump();
        ASSERT_NE(original.at("compiled_identity"),compiledIdentity());
        auto baseline=original;baseline["intent"]="baseline";
        EXPECT_THROW((void)replay(baseline),std::invalid_argument);
        const auto result=replay(original);
        ASSERT_EQ(result.at("status"),"unsupported_candidate_depth");
        EXPECT_EQ(result.at("consumed_transactions"),4);EXPECT_EQ(result.at("first_divergence_transaction"),4);
        EXPECT_EQ(result.at("candidate_depth"),5);
        const auto raw=original.at("transactions").at(3).at("packed").get<uint32_t>();
        EXPECT_EQ(result.at("expected_depth"),static_cast<int>((raw>>8)&15u));
        ASSERT_EQ(result.at("decisions").size(),4u);
        EXPECT_EQ(result.at("decisions").back().at("new_depth"),5);
        EXPECT_EQ(result.at("decisions").back().at("promotion_streak"),1);
        EXPECT_EQ(result.at("decisions").back().at("recommended_depth"),static_cast<int>((raw>>8)&15u));
        EXPECT_EQ(result.at("origin_compiled_identity"),original.at("compiled_identity"));
        EXPECT_EQ(result.at("current_compiled_identity"),compiledIdentity());
        EXPECT_FALSE(result.at("baseline_validated_by_this_executable").get<bool>());
        EXPECT_FALSE(result.at("economy_claim").get<bool>());
        EXPECT_FALSE(result.contains("controller_terminal_projection"));
        EXPECT_FALSE(result.contains("transactions"));EXPECT_FALSE(result.contains("windows"));
        EXPECT_EQ(original.dump(),snapshot);
    }
}

/** @test Genuine v3 observations reject missing tails, corrupted publication and incompatible ABI.
 * Mutations are local negative inputs; the embedded original is never repaired,
 * relabeled, or given newly invented feedback. Independent V/Y endpoints are
 * distinct from the controller-owned projection compared above.
 */
TEST(MTPDepthTraceReplay, RecordedSurvivalV3RejectsMutationAndTruncation)
{
    using namespace recorded_survival_v3;
    for(const auto bytes:{kGreedyTrainPython,kStochasticTrainPython})
    {
        const auto original=parseDocument(bytes);const auto snapshot=original.dump();
        auto bad=original;bad["transactions"].erase(bad["transactions"].end()-1);
        EXPECT_THROW((void)replay(bad),std::invalid_argument);
        for(int word:{1,12})
        {
            bad=original;bad["terminal"]["control"][word]=bad["terminal"]["control"][word].get<int>()+1;
            EXPECT_THROW((void)replay(bad),std::invalid_argument);
        }
        bad=original;bad["transactions"][0]["packed"]=bad["transactions"][0]["packed"].get<uint32_t>()^(uint32_t{1}<<29);
        EXPECT_THROW((void)replay(bad),std::invalid_argument);
        for(const auto field:{"control_layout","transaction_trace","publication_witness"})
        {
            bad=original;bad["compiled_identity"]["sources"][field]=std::string(64,'0');
            EXPECT_THROW((void)replay(bad),std::invalid_argument);
        }
        EXPECT_EQ(original.dump(),snapshot);
    }
}

/** @brief Independently author a completed confirmation request and partial target trial.
 * @param host Select actual host-controller or resident arithmetic ownership.
 * @return A NEW synthetic v3 golden, never rewritten captured model feedback.
 *
 * Budget19 keeps every first12 D5 observation eligible: T12 starts with7 tokens,
 * exceeding physical rows6. Three D1 full outcomes then leave only1 token; final
 * clipped output closes the response without finishing Pending's fourth sample.
 * First twelve P1 transactions give Y13/P12 (initial L0, then L1/rawD1).
 * Three target full outcomes give Y1+2+2/P2+2+2; last clipped gives Y1/P1.
 * The trial reference is ONLY third-window R4/V4, never pooled R12/V12.
 * Synthetic portable CPU metadata deliberately exercises the measured arithmetic
 * owner; it is not a claim of CPU hardware or loaded TP4-model admission.
 */
Json confirmationGoldenRequest(bool host)
{
    auto j=request(host);auto c=initialControl();
    c[kDeviceGenerationControlRemainingTokenCount]=19;
    c[kDeviceGenerationControlCurrentDraftDepth]=5;c[kDeviceGenerationControlActiveVerifierRowCount]=6;
    c[kDeviceGenerationControlMaximumDraftDepth]=15;c[kDeviceGenerationControlInitialDraftDepth]=5;
    c[kDeviceGenerationControlDepthLastRecommendedDepth]=5;c[kDeviceGenerationControlDepthWindowSize]=16;
    c[kDeviceGenerationControlDepthCooldownSteps]=8;c[kDeviceGenerationControlDepthStepsSinceChange]=8;
    c[kDeviceGenerationControlDepthPromoteConsecutiveWindows]=3;
    c[kDeviceGenerationControlLearnedDepthEnabled]=1;c[kDeviceGenerationControlLearnedDepthBackend]=3;
    c[kDeviceGenerationControlLearnedDepthModelClass]=2;
    const std::array<int,10> scope{2,4,1,15,2560,48,512,10,90,6};
    for(size_t i=0;i<scope.size();++i)c[90+i]=scope[i];
    if(host)
    {
        auto &h=j["initial"];h["configured_draft_tokens"]=5;h["initial_depth"]=5;
        auto &p=h["policy"];p["backend"]=3;p["model_class"]=2;p["max_depth"]=15;p["initial_depth"]=5;
        p["window_size"]=16;p["cooldown_steps"]=8;p["promote_consecutive_windows"]=3;p["use_generated_policy"]=true;
        p["scope"]={{"verifier_width_policy",2},{"tp_width",4},{"pp_stages",1},{"retained_depth",15},
            {"hidden_dim",2560},{"main_layers",48},{"expert_count",512},{"experts_per_token",10},
            {"compute_capability_major",90},{"compute_capability_minor",6}};
    }
    else j["initial"]["control"]=c;
    j["transactions"]=Json::array();int cumulative=0;
    for(int tx=1;tx<=16;++tx)
    {
        const int pre=tx<=12?5:1,post=tx<12?5:1;
        const int accepted=(tx>=13&&tx<=15)?1:0;
        const bool incoming=tx>=2&&tx<=13,outgoing=tx<=12,clipped=tx==16;
        const int y=tx==1?2:(tx==14||tx==15)?2:1;
        cumulative+=y;
        const int recommended=tx<4?5:1,windows=tx<12?tx/4:3;
        // Direct wire arithmetic independently specifies the codec input. The
        // production codec/transition does not generate its expected oracle.
        uint32_t packed=static_cast<uint32_t>(pre)|(static_cast<uint32_t>(accepted)<<4)|
            (static_cast<uint32_t>(post)<<8)|(static_cast<uint32_t>(!host&&!clipped&&accepted<pre)<<12)|
            (static_cast<uint32_t>(clipped)<<13)|(static_cast<uint32_t>(recommended)<<25)|
            (static_cast<uint32_t>(incoming)<<29)|(static_cast<uint32_t>(outgoing)<<30);
        if(host)packed|=(uint32_t{1}<<24)|(static_cast<uint32_t>(pre)<<14)|
            (static_cast<uint32_t>(clipped?0:pre)<<19);
        j["transactions"].push_back({{"ordinal",tx},{"packed",packed},{"token_witness",host?y:cumulative},
            {"evaluated_windows",windows}});
    }
    if(host)
        j["terminal"]={{"transactions",16},{"current_depth",1},{"evaluated_windows",3},{"emitted_tokens",19},
            {"published_state_tokens",19},{"model_stopped",false},{"utility_probes",1},{"utility_keeps",0},{"utility_returns",0}};
    else
    {
        auto terminal=c;terminal[kDeviceGenerationControlRemainingTokenCount]=0;
        terminal[kDeviceGenerationControlResponseTokenCount]=19;terminal[kDeviceGenerationControlRequestComplete]=1;
        terminal[kDeviceGenerationControlTransactionCount]=16;terminal[kDeviceGenerationControlPublishedStateCommitCount]=19;
        terminal[kDeviceGenerationControlLastTransactionEmittedTokenCount]=1;
        terminal[kDeviceGenerationControlCurrentDraftDepth]=1;terminal[kDeviceGenerationControlActiveVerifierRowCount]=2;
        terminal[kDeviceGenerationControlDepthLastRecommendedDepth]=1;terminal[kDeviceGenerationControlDepthStepsSinceChange]=3;
        terminal[kDeviceGenerationControlDepthWindowVerifierRuns]=3;terminal[kDeviceGenerationControlDepthWindowAttemptedTokens]=3;
        terminal[kDeviceGenerationControlDepthWindowAcceptedTokens]=3;terminal[kDeviceGenerationControlDepthWindowFullAccepts]=3;
        terminal[kDeviceGenerationControlDepthWindowAcceptedPrefixSum]=3;
        terminal[kDeviceGenerationControlDepthEvaluatedWindows]=3;terminal[kDeviceGenerationControlDepthUpdates]=1;
        terminal[kDeviceGenerationControlDepthDemotions]=1;
        terminal[kDeviceGenerationControlDepthUtilityPhase]=1;terminal[kDeviceGenerationControlDepthUtilityReferenceDepth]=5;
        terminal[kDeviceGenerationControlDepthUtilityReferenceRuns]=4;terminal[kDeviceGenerationControlDepthUtilityReferenceVerifiedProgress]=4;
        terminal[kDeviceGenerationControlDepthUtilityWindowEmitted]=5;terminal[kDeviceGenerationControlDepthUtilityProbes]=1;
        j["terminal"]={{"transactions",16},{"control",terminal}};
    }
    return j;
}

/** @test Three disjoint measured windows defer the selector and charge only the third reference. */
TEST(MTPDepthTraceReplay, ConfirmationGoldenPreservesTargetStreakAndThirdWindowReference)
{
    for(int repeat=0;repeat<3;++repeat)for(bool host:{false,true})
    {
        const auto input=confirmationGoldenRequest(host);const auto snapshot=input.dump();const auto result=replay(input);
        EXPECT_EQ(result.at("status"),"complete_baseline_replay");EXPECT_EQ(result.at("transactions"),16);
        EXPECT_EQ(result.at("emitted_tokens"),19);EXPECT_EQ(result.at("published_state_tokens"),19);
        EXPECT_EQ(result.at("evaluated_windows"),3);EXPECT_EQ(result.at("updates"),1);EXPECT_EQ(result.at("demotions"),1);
        EXPECT_EQ(result.at("utility_probes"),1);EXPECT_EQ(result.at("utility_keeps"),0);EXPECT_EQ(result.at("utility_returns"),0);
        const auto &rows=result.at("decisions");ASSERT_EQ(rows.size(),16u);
        for(size_t tx=0;tx<rows.size();++tx)
        {
            const auto &row=rows[tx];const int expected_streak=tx<3?0:tx<7?1:tx<11?2:0;
            EXPECT_EQ(row.at("promotion_streak"),expected_streak)<<"tx"<<tx+1;
            EXPECT_EQ(row.at("new_depth"),tx<11?5:1)<<"tx"<<tx+1;
            EXPECT_EQ(row.at("recommended_depth"),tx<3?5:1)<<"tx"<<tx+1;
        }
        const auto &windows=result.at("windows");ASSERT_EQ(windows.size(),3u);
        for(size_t i=0;i<3;++i)
        {
            EXPECT_EQ(windows[i].at("transaction"),4*(i+1));EXPECT_EQ(windows[i].at("runs"),4);
            EXPECT_EQ(windows[i].at("accepted"),0);EXPECT_EQ(windows[i].at("zero"),4);EXPECT_EQ(windows[i].at("full"),0);
            EXPECT_EQ(windows[i].at("recommended_depth"),1);EXPECT_EQ(windows[i].at("new_depth"),i<2?5:1);
        }
        EXPECT_TRUE(rows.back().at("budget_limited").get<bool>());
        if(!host)EXPECT_EQ(result.at("controller_terminal_projection"),
            expectedResidentTerminalProjection(input.at("terminal").at("control")));
        EXPECT_EQ(input.dump(),snapshot);
    }
}

/** @test A fresh source cannot borrow an old baseline identity or missing confirmation endpoint. */
TEST(MTPDepthTraceReplay, ConfirmationEndpointAndFreshStateCorruptionAreRejected)
{
    auto bad=confirmationGoldenRequest(false);
    bad["terminal"]["control"][kDeviceGenerationControlDepthPromotionStreak]=1;
    EXPECT_THROW((void)replay(bad),std::invalid_argument);
    bad=confirmationGoldenRequest(false);bad["initial"]["control"][kDeviceGenerationControlDepthPromotionStreak]=1;
    EXPECT_THROW((void)replay(bad),std::invalid_argument);
    bad=confirmationGoldenRequest(false);bad["initial"]["control"][kDeviceGenerationControlDepthLastRecommendedDepth]=1;
    EXPECT_THROW((void)replay(bad),std::invalid_argument);
    for(bool host:{false,true})
    {
        bad=confirmationGoldenRequest(host);bad["transactions"].erase(bad["transactions"].end()-1);
        EXPECT_THROW((void)replay(bad),std::invalid_argument);
    }
}

/** @brief Reconstruct a retained numeric projection without invoking controller arithmetic.
 * @param table Exact saved baseline constants, column names and ordered numeric rows.
 * @return The original decision/window JSON, including all retained fields.
 */
Json expandRecordedConfirmationTable(const Json &table)
{
    Json rows=Json::array();
    for(const auto &values:table.at("rows"))
    {
        Json row=table.at("constants");
        if(values.size()!=table.at("columns").size())throw std::invalid_argument("corrupt retained chronology table");
        for(size_t i=0;i<values.size();++i)row[table.at("columns")[i].get<std::string>()]=values[i];
        rows.push_back(std::move(row));
    }
    return rows;
}

/** @test Real TRAIN natural prompts replay through the production controller with exact chronology.
 * Original producer identity remains immutable. Candidate intent allows a future
 * implementation to identify its first unsupported depth; this exact regression
 * fails there and never borrows feedback beyond that frontier. Expected decisions
 * are saved original baseline output, independent of this invocation.
 */
TEST(MTPDepthTraceReplay, RecordedConfirmationV3TrainPythonReplaysExactObservedBaseline)
{
    using namespace recorded_confirmation_v3;
    for(int repeat=0;repeat<3;++repeat)
    for(const auto pair:{std::pair{kGreedyTrainPython,kGreedyTrainPythonExpected},
                         std::pair{kStochasticTrainPython,kStochasticTrainPythonExpected}})
    {
        const auto input=parseDocument(pair.first);const auto snapshot=input.dump();
        auto expected=parseDocument(pair.second);
        ASSERT_EQ(input.at("schema"),"llaminar.mtp.depth_trace_replay_request.v3");
        ASSERT_EQ(input.at("intent"),"candidate");
        const auto result=replay(input);
        ASSERT_NE(result.at("status"),"unsupported_candidate_depth")
            <<"Actual recorded feedback ends at the first unsupported selector frontier: "<<result.dump();
        ASSERT_EQ(result.at("status"),"same_observed_depth_path");
        EXPECT_EQ(result.at("origin_compiled_identity"),input.at("compiled_identity"));
        EXPECT_EQ(result.at("current_compiled_identity"),compiledIdentity());
        EXPECT_EQ(result.at("capture"),input.at("capture"));
        EXPECT_FALSE(result.at("baseline_validated_by_this_executable").get<bool>());
        EXPECT_FALSE(result.at("economy_claim").get<bool>());
        expected["decisions"]=expandRecordedConfirmationTable(expected.at("decisions"));
        expected["windows"]=expandRecordedConfirmationTable(expected.at("windows"));
        // Only explicit invocation metadata differs from the original baseline.
        for(const auto key:{"status","intent","baseline_validated_by_this_executable","current_compiled_identity"})
            expected[key]=result.at(key);
        EXPECT_EQ(result,expected);
        EXPECT_EQ(result.at("controller_terminal_projection"),
            expectedResidentTerminalProjection(input.at("terminal").at("control")));
        EXPECT_EQ(input.dump(),snapshot);
    }
}

/** @test Genuine v3 observations reject missing tails, corrupted publication and incompatible ABI.
 * Mutations are local negative inputs; the embedded original is never repaired,
 * relabeled, or given newly invented feedback. Independent V/Y endpoints are
 * distinct from the controller-owned projection compared above.
 */
TEST(MTPDepthTraceReplay, RecordedConfirmationV3RejectsMutationAndTruncation)
{
    using namespace recorded_confirmation_v3;
    for(const auto bytes:{kGreedyTrainPython,kStochasticTrainPython})
    {
        const auto original=parseDocument(bytes);const auto snapshot=original.dump();
        auto bad=original;bad["transactions"].erase(bad["transactions"].end()-1);
        EXPECT_THROW((void)replay(bad),std::invalid_argument);
        for(int word:{1,12})
        {
            bad=original;bad["terminal"]["control"][word]=bad["terminal"]["control"][word].get<int>()+1;
            EXPECT_THROW((void)replay(bad),std::invalid_argument);
        }
        bad=original;bad["transactions"][0]["packed"]=bad["transactions"][0]["packed"].get<uint32_t>()^(uint32_t{1}<<29);
        EXPECT_THROW((void)replay(bad),std::invalid_argument);
        for(const auto field:{"control_layout","transaction_trace","publication_witness"})
        {
            bad=original;bad["compiled_identity"]["sources"][field]=std::string(64,'0');
            EXPECT_THROW((void)replay(bad),std::invalid_argument);
        }
        EXPECT_EQ(original.dump(),snapshot);
    }
}

}
