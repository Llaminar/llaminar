/**
 * @file MTPDepthTraceReplay.cpp
 * @brief Replay immutable transaction feedback through the canonical controllers.
 *
 * Resident replay invokes SamplingMath's actual observation transition; host
 * replay invokes the real MTPDepthController. Only owner/static fields belong
 * to the endpoint comparison: QSA, head, response and maintenance producers are
 * not rerun by this diagnostic. Candidate feedback ends at first depth change.
 */
#include "MTPDepthTraceReplay.h"
#include "execution/mtp/MTPDepthReplayBuildIdentity.h"
#include "execution/mtp/MTPDepthController.h"
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace llaminar2::test::mtp_trace_replay
{
using Json = nlohmann::json;
using namespace sampling_math;
namespace
{
/** @brief Reject invalid input without substituting defaults. */
void require(bool ok, const std::string &message) { if (!ok) throw std::invalid_argument(message); }
/** @brief Require exactly a closed set of object members. */
void keys(const Json &j, std::initializer_list<const char *> expected)
{
    require(j.is_object(), "expected JSON object");
    std::set<std::string> wanted; for (auto key : expected) wanted.emplace(key);
    std::set<std::string> actual; for (auto it=j.begin();it!=j.end();++it) actual.emplace(it.key());
    require(actual == wanted, "missing/unknown object fields");
}
/** @return An exact integer in range; bools, strings and integral floats are invalid. */
int64_t integer(const Json &j, int64_t minimum, int64_t maximum)
{
    require(j.is_number_integer(), "expected exact integer");
    if (j.is_number_unsigned()) require(j.get<uint64_t>() <= static_cast<uint64_t>(maximum), "unsigned integer overflow");
    const auto value=j.get<int64_t>(); require(value>=minimum && value<=maximum,"integer out of range"); return value;
}
/** @return Exact finite floating input, preserving CPU round-trip precision. */
double rate(const Json &j)
{
    require(j.is_number() && !j.is_boolean(),"expected numeric CPU rate");
    const double result=j.get<double>();require(std::isfinite(result)&&result>=0&&result<=1,"invalid CPU rate");return result;
}
/** @return A nonempty metadata string, never a model/runtime path to inspect. */
std::string text(const Json &j) { require(j.is_string()&&!j.get_ref<const std::string &>().empty(),"expected nonempty string");return j.get<std::string>(); }
/** @brief Validate a source receipt/config digest without reading payloads. */
void hash(const Json &j)
{
    const auto value=text(j);require(value.size()==64&&value.find_first_not_of("0123456789abcdef")==std::string::npos,"invalid source/config digest");
}
/** @return Exactly the existing control ABI, without numeric coercions. */
std::array<int,kDeviceGenerationControlCount> control(const Json &j)
{
    require(j.is_array()&&j.size()==kDeviceGenerationControlCount,"control must have exactly108 words");
    std::array<int,kDeviceGenerationControlCount> out{};
    for (int i=0;i<kDeviceGenerationControlCount;++i) out[i]=static_cast<int>(integer(j[i],INT32_MIN,INT32_MAX));
    return out;
}
/** @return A checked policy reconstructed only from authority-owned static words. */
DeviceGenerationPolicy policy(const std::array<int,kDeviceGenerationControlCount> &c)
{
    DeviceGenerationPolicy p;
    p.mode=static_cast<DeviceGenerationPolicyMode>(c[kDeviceGenerationControlDepthPolicyMode]);
    p.initial_depth=c[kDeviceGenerationControlInitialDraftDepth];
    p.minimum_depth=c[kDeviceGenerationControlMinimumDraftDepth];p.maximum_depth=c[kDeviceGenerationControlMaximumDraftDepth];
    p.window_size=c[kDeviceGenerationControlDepthWindowSize];p.minimum_samples=c[kDeviceGenerationControlDepthMinimumSamples];
    p.cooldown_steps=c[kDeviceGenerationControlDepthCooldownSteps];p.promote_consecutive_windows=c[kDeviceGenerationControlDepthPromoteConsecutiveWindows];
    p.promote_full_accept_rate_ppm=c[kDeviceGenerationControlDepthPromoteFullAcceptRatePPM];
    p.demote_zero_accept_rate_ppm=c[kDeviceGenerationControlDepthDemoteZeroAcceptRatePPM];
    p.demote_acceptance_rate_ppm=c[kDeviceGenerationControlDepthDemoteAcceptanceRatePPM];
    require(c[kDeviceGenerationControlLearnedDepthEnabled]==0||c[kDeviceGenerationControlLearnedDepthEnabled]==1,"invalid learned enabled word");
    p.learned={.enabled=c[kDeviceGenerationControlLearnedDepthEnabled]!=0,
        .backend=static_cast<MTPDepthPolicyBackend>(c[kDeviceGenerationControlLearnedDepthBackend]),
        .model_class=static_cast<MTPDepthPolicyModelClass>(c[kDeviceGenerationControlLearnedDepthModelClass]),
        .verify_mode=static_cast<MTPVerifyMode>(c[kDeviceGenerationControlLearnedDepthVerifyMode]),
        .scope={.verifier_width_policy=static_cast<MTPVerifierPhysicalWidthPolicy>(c[kDeviceGenerationControlLearnedScopeWidthPolicy]),
            .tp_width=c[kDeviceGenerationControlLearnedScopeTPWidth],.pp_stages=c[kDeviceGenerationControlLearnedScopePPStages],
            .retained_depth=c[kDeviceGenerationControlLearnedScopeRetainedDepth],.hidden_dim=c[kDeviceGenerationControlLearnedScopeHiddenDim],
            .main_layers=c[kDeviceGenerationControlLearnedScopeMainLayers],.expert_count=c[kDeviceGenerationControlLearnedScopeExpertCount],
            .experts_per_token=c[kDeviceGenerationControlLearnedScopeExpertsPerToken],
            .compute_capability_major=c[kDeviceGenerationControlLearnedScopeComputeCapabilityMajor],
            .compute_capability_minor=c[kDeviceGenerationControlLearnedScopeComputeCapabilityMinor]}};
    require(p.valid()&&!p.isOrdinary()&&!p.isForwardOnly(),"invalid/out-of-scope resident policy"); return p;
}
/** @return True only for controller/static fields this replay actually owns. */
bool owned(int word)
{
    return word==kDeviceGenerationControlOk||word==kDeviceGenerationControlErrorCode||
        (word>=kDeviceGenerationControlCurrentDraftDepth&&word<=kDeviceGenerationControlDepthLastRecommendedDepth)||
        (word>=kDeviceGenerationControlLearnedDepthEnabled&&word<=kDeviceGenerationControlInitialDraftDepth)||
        (word>=kDeviceGenerationControlLearnedScopeWidthPolicy&&word<=kDeviceGenerationControlLearnedScopeComputeCapabilityMinor)||
        (word>=kDeviceGenerationControlDepthUtilityPhase&&word<=kDeviceGenerationControlDepthUtilityReturns);
}
/** @brief Export exactly the resident fields this replay actually reconstructs.
 * @param c Final replay-owned row; external append/QSA/sampler owners are excluded.
 * @return Explicit ABI word/value pairs for comparison with immutable capture data.
 *
 * This projection is evidence, not a new candidate acceptance condition. A
 * same-depth candidate may intentionally change its cadence or learner state;
 * a recorded regression can independently require exact endpoint equality.
 */
Json ownedControlProjection(const std::array<int,kDeviceGenerationControlCount> &c)
{
    Json words=Json::array();
    for(int word=0;word<kDeviceGenerationControlCount;++word)
        if(owned(word))words.push_back({{"word",word},{"value",c[word]}});
    return {{"schema","llaminar.mtp.resident_controller_terminal_projection.v1"},{"words",words}};
}
/** @brief Reject missing identity against this executable's own compiled constants. */
void identity(const Json &j, bool baseline)
{
    const auto own=compiledIdentity();keys(j,{"schema","sources"});require(j.at("schema")==own.at("schema"),"source identity schema mismatch");
    require(j.at("sources").is_object()&&j.at("sources").size()==own.at("sources").size(),"source identity fields mismatch");
    for(auto it=own.at("sources").begin();it!=own.at("sources").end();++it)
    {
        require(j.at("sources").contains(it.key()),"missing compiled source identity");hash(j.at("sources").at(it.key()));
    }
    if(baseline) require(j==own,"compiled controller/policy source identity mismatch");
    else for(const char *abi:{"control_layout","transaction_trace","policy_types","policy_scope","verifier_width_policy","publication_witness"})
        require(j.at("sources").at(abi)==own.at("sources").at(abi),"candidate trace/control ABI identity mismatch");
}
/** @brief Validate closed request/custody/transaction framing before any controller call. */
void envelope(const Json &j)
{
    keys(j,{"schema","intent","authority","compiled_identity","capture","initial","transactions","terminal","output"});
    require(j.at("intent")=="baseline"||j.at("intent")=="candidate","unknown replay intent");
    require(j.at("schema")=="llaminar.mtp.depth_trace_replay_request.v3","unknown replay schema");
    require(j.at("authority")=="native_resident"||j.at("authority")=="cpu_host","unknown trace authority");
    identity(j.at("compiled_identity"),j.at("intent")=="baseline");
    const auto &capture=j.at("capture");keys(capture,{"observation_identity","device","receipt_sha256","configuration_sha256"});
    (void)text(capture.at("observation_identity"));(void)text(capture.at("device"));hash(capture.at("receipt_sha256"));hash(capture.at("configuration_sha256"));
    (void)text(j.at("output"));require(j.at("transactions").is_array(),"transactions must be an array");
    const int n=static_cast<int>(integer(j.at("terminal").at("transactions"),1,kMTPDepthTransactionTraceCapacity));
    require(j.at("transactions").size()==static_cast<size_t>(n),"missing/duplicated transaction or terminal count mismatch");
    for (int i=0;i<n;++i)
    {
        const auto &r=j.at("transactions")[i];keys(r,{"ordinal","packed","token_witness","evaluated_windows"});
        require(integer(r.at("ordinal"),1,n)==i+1,"transaction ordinal gap/duplicate/order mismatch");
        MTPDepthTransactionTraceRecord raw{static_cast<uint32_t>(integer(r.at("packed"),0,UINT32_MAX)),
            static_cast<int32_t>(integer(r.at("token_witness"),1,256)),
            static_cast<int32_t>(integer(r.at("evaluated_windows"),0,256))};
        MTPDepthTransactionTraceValues value;
        require(unpackMTPDepthTransactionTraceRecord(raw,&value),"corrupt packed transaction");
        require(value.host_observation==(j.at("authority")=="cpu_host"),"wrong transaction owner");
        require(value.evaluated_windows<=i+1,"impossible evaluated-window count");
    }
}
/** @return The actual lossless decoded inputs; input shape was already checked. */
MTPDepthTransactionTraceValues values(const Json &r)
{
    MTPDepthTransactionTraceRecord raw{r.at("packed").get<uint32_t>(),r.at("token_witness").get<int32_t>(),r.at("evaluated_windows").get<int32_t>()};
    MTPDepthTransactionTraceValues v;require(unpackMTPDepthTransactionTraceRecord(raw,&v),"invalid record");return v;
}
/** @return The unchanged decoded inputs, never inferred from window aggregates. */
Json observedInputs(const MTPDepthTransactionTraceValues &v)
{
    return {{"accepted_prefix",v.accepted_prefix},{"rollback",v.rollback},{"budget_limited",v.budget_limited},
        {"token_witness",v.token_witness},{"token_witness_scope",v.host_observation?"cpu_transaction_emitted":"native_committed_response"},
        {"host_observation",v.host_observation},{"incoming_debt",v.incoming_debt},
        {"outgoing_debt",v.outgoing_debt},{"model_stopped",v.model_stopped},
        {"requested_depth",v.requested_depth},{"effective_depth",v.effective_depth}};
}
/** @brief Replay resident arithmetic; alternate policy stops before unsupported feedback. */
Json resident(const Json &j,const DeviceGenerationPolicy *candidate, bool candidate_mode)
{
    keys(j.at("initial"),{"phase","control"});keys(j.at("terminal"),{"transactions","control"});
    require(j.at("initial").at("phase")=="validated_pre_first_response_append","wrong initial snapshot phase");
    auto c=control(j.at("initial").at("control"));const auto terminal=control(j.at("terminal").at("control"));
    const auto p=policy(c);
    require(c[kDeviceGenerationControlOk]==1&&c[kDeviceGenerationControlErrorCode]==0&&
        c[kDeviceGenerationControlTransactionCount]==0&&c[kDeviceGenerationControlResponseTokenCount]==0&&
        c[kDeviceGenerationControlRequestComplete]==0&&c[kDeviceGenerationControlRemainingTokenCount]>0&&
        c[kDeviceGenerationControlRemainingTokenCount]<=256,"resident snapshot is not fresh admitted response");
    const int budget=c[kDeviceGenerationControlRemainingTokenCount];
    // Canonical transitions increment this signed counter on every eligible
    // input. Reject hostile wire values before invoking unchanged arithmetic.
    require(p.cooldown_steps<=INT32_MAX-static_cast<int>(j.at("transactions").size()),
        "cooldown exceeds safe finite replay horizon");
    require(device_generation_policy_matches(c.data(),p)&&c[kDeviceGenerationControlCurrentDraftDepth]==p.initial_depth&&
        c[kDeviceGenerationControlActiveVerifierRowCount]==p.initial_depth+1&&
        c[kDeviceGenerationControlDepthLastRecommendedDepth]==p.initial_depth&&
        c[kDeviceGenerationControlDepthStepsSinceChange]==p.cooldown_steps,"initial policy/config mismatch");
    for(int w=kDeviceGenerationControlDepthPromotionStreak;w<=kDeviceGenerationControlDepthDemotions;++w)
        require(c[w]==0,"initial learner state is not fresh");
    require(c[kDeviceGenerationControlLearnedDepthMatchedWindows]==0,"initial matched count is not fresh");
    for(int word=kDeviceGenerationControlDepthUtilityPhase;word<=kDeviceGenerationControlDepthUtilityReturns;++word)
        require(c[word]==0,"initial utility state/counters are not fresh");
    // Validate the COMPLETE recorded origin path independently before any
    // candidate consumes inputs. Reading framing is not replaying future feedback.
    int observed_depth=p.initial_depth,observed_response=0,observed_windows=0;
    int observed_leading=c[kDeviceGenerationControlNextLeadingCommittedOutputCount];
    std::int64_t observed_published=c[kDeviceGenerationControlPublishedStateCommitCount];
    require((observed_leading==0||observed_leading==1)&&observed_published==0,
        "initial publication endpoint is not fresh");
    bool observed_stopped=false;
    for(const auto &raw:j.at("transactions"))
    {
        const auto v=values(raw);
        require(v.pre_depth==observed_depth,"origin trace depth continuity mismatch");
        require(v.token_witness>observed_response&&v.token_witness<=budget&&v.token_witness-observed_response+static_cast<int>(v.incoming_debt)<=v.pre_depth+1,"nonmonotonic/out-of-budget origin progress");
        require(v.evaluated_windows>=observed_windows&&v.evaluated_windows<=observed_windows+1,"origin window continuity mismatch");
        MTPPublicationWitness publication;
        require(!observed_stopped && v.incoming_debt==(observed_leading!=0) &&
            MTPPublicationWitness::fromEmitted(static_cast<uint32_t>(v.token_witness-observed_response),
                v.incoming_debt,v.outgoing_debt,v.model_stopped,&publication),
            "origin publication debt/progress mismatch");
        observed_published+=publication.publishedStateTokens();
        observed_leading=!v.model_stopped&&v.outgoing_debt?1:0;observed_stopped=v.model_stopped;
        observed_depth=v.post_depth;observed_response=v.token_witness;observed_windows=v.evaluated_windows;
    }
    require(terminal[kDeviceGenerationControlTransactionCount]==j.at("transactions").size()&&
        terminal[kDeviceGenerationControlResponseTokenCount]==observed_response&&terminal[kDeviceGenerationControlOk]==1&&
        terminal[kDeviceGenerationControlErrorCode]==0&&terminal[kDeviceGenerationControlRequestComplete]==1&&
        terminal[kDeviceGenerationControlRemainingTokenCount]==budget-observed_response&&
        (terminal[kDeviceGenerationControlModelStopped]==0||terminal[kDeviceGenerationControlModelStopped]==1)&&
        (terminal[kDeviceGenerationControlRemainingTokenCount]==0||terminal[kDeviceGenerationControlModelStopped]==1)&&
        terminal[kDeviceGenerationControlCurrentDraftDepth]==observed_depth&&
        terminal[kDeviceGenerationControlActiveVerifierRowCount]==observed_depth+1&&
        terminal[kDeviceGenerationControlDepthEvaluatedWindows]==observed_windows&&
        terminal[kDeviceGenerationControlPublishedStateCommitCount]==observed_published&&
        terminal[kDeviceGenerationControlNextLeadingCommittedOutputCount]==observed_leading&&
        terminal[kDeviceGenerationControlModelStopped]==static_cast<int>(observed_stopped),
        "origin terminal publication/progress witness mismatch");
    require(device_generation_policy_matches(terminal.data(),p),"origin static policy/config identity mismatch");
    if(candidate)
    {
        require(candidate->valid()&&!candidate->isOrdinary()&&!candidate->isForwardOnly(),"invalid candidate");
        require(candidate->cooldown_steps<=INT32_MAX-static_cast<int>(j.at("transactions").size()),
            "candidate cooldown exceeds safe finite replay horizon");
        require(candidate->learned.backend==p.learned.backend&&candidate->learned.model_class==p.learned.model_class&&
            candidate->learned.verify_mode==p.learned.verify_mode&&candidate->learned.scope==p.learned.scope,
            "candidate changes immutable verifier/domain identity");
        require(initialize_device_generation_control(budget,256,*candidate,c.data()),"candidate initialization failed");
    }
    Json decisions=Json::array(),windows=Json::array();int response=0;
    for(size_t i=0;i<j.at("transactions").size();++i)
    {
        const auto v=values(j.at("transactions")[i]);
        if(candidate_mode&&c[kDeviceGenerationControlCurrentDraftDepth]!=v.pre_depth)
            return {{"status","unsupported_candidate_depth"},{"consumed_transactions",i},{"first_divergence_transaction",i+1},{"expected_depth",v.pre_depth},{"candidate_depth",c[kDeviceGenerationControlCurrentDraftDepth]},{"decisions",decisions},{"economy_claim",false}};
        require(c[kDeviceGenerationControlCurrentDraftDepth]==v.pre_depth,"baseline pre-depth mismatch at transaction"+std::to_string(i+1));
        require(v.token_witness>response&&v.token_witness<=budget&&v.token_witness-response+static_cast<int>(v.incoming_debt)<=v.pre_depth+1,"nonmonotonic/out-of-budget response progress");
        const int newly_emitted=v.token_witness-response;response=v.token_witness;
        const auto before=c;c[kDeviceGenerationControlTransactionCount]=static_cast<int>(i+1);c[kDeviceGenerationControlResponseTokenCount]=response;
        c[kDeviceGenerationControlLastTransactionEmittedTokenCount]=newly_emitted;
        MTPPublicationWitness publication;
        require(MTPPublicationWitness::fromEmitted(static_cast<uint32_t>(newly_emitted),
            v.incoming_debt,v.outgoing_debt,v.model_stopped,&publication),"resident publication witness rejected");
        require(record_device_generation_depth_observation(v.accepted_prefix,v.rollback,v.budget_limited,c.data(),publication),"resident observation rejected");
        decisions.push_back({{"ordinal",i+1},{"old_depth",v.pre_depth},{"recommended_depth",c[kDeviceGenerationControlDepthLastRecommendedDepth]},
            {"new_depth",c[kDeviceGenerationControlCurrentDraftDepth]},{"evaluated_windows",c[kDeviceGenerationControlDepthEvaluatedWindows]}});
        decisions.back().update(observedInputs(v));
        decisions.back()["newly_emitted_tokens"]=newly_emitted;
        decisions.back()["published_state_tokens"]=publication.publishedStateTokens();
        // This is the actual word26 owner: measured confirmation count, legacy
        // promotion hysteresis elsewhere. Never infer it from a recommendation.
        decisions.back()["promotion_streak"]=c[kDeviceGenerationControlDepthPromotionStreak];
        decisions.back()["utility_phase"]=c[kDeviceGenerationControlDepthUtilityPhase];
        decisions.back()["utility_probes"]=c[kDeviceGenerationControlDepthUtilityProbes];
        decisions.back()["utility_keeps"]=c[kDeviceGenerationControlDepthUtilityKeeps];
        decisions.back()["utility_returns"]=c[kDeviceGenerationControlDepthUtilityReturns];
        if(c[kDeviceGenerationControlDepthEvaluatedWindows]!=before[kDeviceGenerationControlDepthEvaluatedWindows])
            windows.push_back({{"transaction",i+1},{"runs",before[kDeviceGenerationControlDepthWindowVerifierRuns]+1},
                {"accepted",before[kDeviceGenerationControlDepthWindowAcceptedTokens]+v.accepted_prefix},
                {"zero",before[kDeviceGenerationControlDepthWindowZeroAccepts]+(v.accepted_prefix==0)},
                {"full",before[kDeviceGenerationControlDepthWindowFullAccepts]+(v.accepted_prefix==v.pre_depth)},
                {"old_depth",v.pre_depth},{"recommended_depth",c[kDeviceGenerationControlDepthLastRecommendedDepth]},
                {"new_depth",c[kDeviceGenerationControlCurrentDraftDepth]}});
        if(candidate_mode&&c[kDeviceGenerationControlCurrentDraftDepth]!=v.post_depth)
            return {{"status","unsupported_candidate_depth"},{"consumed_transactions",i+1},{"first_divergence_transaction",i+1},
                {"expected_depth",v.post_depth},{"candidate_depth",c[kDeviceGenerationControlCurrentDraftDepth]},{"decisions",decisions},{"economy_claim",false}};
        if(!candidate_mode)require(c[kDeviceGenerationControlCurrentDraftDepth]==v.post_depth&&
            c[kDeviceGenerationControlDepthLastRecommendedDepth]==v.recommended_depth&&
            c[kDeviceGenerationControlDepthEvaluatedWindows]==v.evaluated_windows,"baseline post-depth/recommendation/window mismatch");
    }
    require(terminal[kDeviceGenerationControlTransactionCount]==j.at("transactions").size()&&
        terminal[kDeviceGenerationControlResponseTokenCount]==response&&terminal[kDeviceGenerationControlOk]==1&&
        terminal[kDeviceGenerationControlErrorCode]==0&&terminal[kDeviceGenerationControlRequestComplete]==1&&
        terminal[kDeviceGenerationControlRemainingTokenCount]==budget-response&&
        (terminal[kDeviceGenerationControlModelStopped]==0||terminal[kDeviceGenerationControlModelStopped]==1)&&
        (terminal[kDeviceGenerationControlRemainingTokenCount]==0||terminal[kDeviceGenerationControlModelStopped]==1),
        "terminal transaction/progress/completeness mismatch");
    if(!candidate_mode)for(int w=0;w<kDeviceGenerationControlCount;++w)if(owned(w))
        require(c[w]==terminal[w],"terminal controller/static config mismatch at word"+std::to_string(w));
    return {{"status",candidate_mode?"same_observed_depth_path":"complete_baseline_replay"},{"authority","native_resident"},
        {"transactions",j.at("transactions").size()},{"final_depth",c[kDeviceGenerationControlCurrentDraftDepth]},
        {"evaluated_windows",c[kDeviceGenerationControlDepthEvaluatedWindows]},
        {"published_state_tokens",observed_published},
        {"emitted_tokens",response},{"model_stopped",observed_stopped},
        {"updates",c[kDeviceGenerationControlDepthUpdates]},{"promotions",c[kDeviceGenerationControlDepthPromotions]},
        {"demotions",c[kDeviceGenerationControlDepthDemotions]},
        {"utility_probes",c[kDeviceGenerationControlDepthUtilityProbes]},
        {"utility_keeps",c[kDeviceGenerationControlDepthUtilityKeeps]},
        {"utility_returns",c[kDeviceGenerationControlDepthUtilityReturns]},
        {"controller_terminal_projection",ownedControlProjection(c)},
        {"decisions",decisions},{"windows",windows},{"economy_claim",false}};
}
/** @return Exact resolved host policy, never a conversion from device PPM words. */
MTPDepthPolicyConfig hostPolicy(const Json &p)
{
    keys(p,{"mode","backend","model_class","scope","min_depth","max_depth","initial_depth","window_size","min_samples",
        "cooldown_steps","promote_consecutive_windows","use_generated_policy","promote_full_accept_rate","demote_zero_accept_rate","demote_acceptance_rate"});
    MTPDepthPolicyConfig c;
    c.mode=static_cast<MTPDepthPolicyMode>(integer(p.at("mode"),0,2));
    c.backend=static_cast<MTPDepthPolicyBackend>(integer(p.at("backend"),0,3));c.model_class=static_cast<MTPDepthPolicyModelClass>(integer(p.at("model_class"),0,2));
    const auto &s=p.at("scope");keys(s,{"verifier_width_policy","tp_width","pp_stages","retained_depth","hidden_dim","main_layers","expert_count","experts_per_token","compute_capability_major","compute_capability_minor"});
    c.scope={.verifier_width_policy=static_cast<MTPVerifierPhysicalWidthPolicy>(integer(s.at("verifier_width_policy"),0,2)),
        .tp_width=static_cast<int>(integer(s.at("tp_width"),0,INT32_MAX)),.pp_stages=static_cast<int>(integer(s.at("pp_stages"),0,INT32_MAX)),
        .retained_depth=static_cast<int>(integer(s.at("retained_depth"),0,15)),.hidden_dim=static_cast<int>(integer(s.at("hidden_dim"),0,INT32_MAX)),
        .main_layers=static_cast<int>(integer(s.at("main_layers"),0,INT32_MAX)),.expert_count=static_cast<int>(integer(s.at("expert_count"),0,INT32_MAX)),
        .experts_per_token=static_cast<int>(integer(s.at("experts_per_token"),0,INT32_MAX)),
        .compute_capability_major=static_cast<int>(integer(s.at("compute_capability_major"),0,INT32_MAX)),
        .compute_capability_minor=static_cast<int>(integer(s.at("compute_capability_minor"),0,INT32_MAX))};
    c.min_depth=static_cast<int>(integer(p.at("min_depth"),1,15));c.max_depth=static_cast<int>(integer(p.at("max_depth"),1,15));
    c.initial_depth=static_cast<int>(integer(p.at("initial_depth"),1,15));c.window_size=static_cast<int>(integer(p.at("window_size"),1,INT32_MAX));
    c.min_samples=static_cast<int>(integer(p.at("min_samples"),1,INT32_MAX));c.cooldown_steps=static_cast<int>(integer(p.at("cooldown_steps"),0,INT32_MAX));
    c.promote_consecutive_windows=static_cast<int>(integer(p.at("promote_consecutive_windows"),1,INT32_MAX));
    require(p.at("use_generated_policy").is_boolean(),"expected learned boolean");c.use_generated_policy=p.at("use_generated_policy").get<bool>();
    c.promote_full_accept_rate=rate(p.at("promote_full_accept_rate"));c.demote_zero_accept_rate=rate(p.at("demote_zero_accept_rate"));c.demote_acceptance_rate=rate(p.at("demote_acceptance_rate"));return c;
}
/** @brief Replay actual host semantics and independent terminal completeness witnesses. */
Json host(const Json &j,bool candidate_mode)
{
    const auto &initial=j.at("initial");keys(initial,{"schema","compiled_identity","verify_mode","configured_draft_tokens","initial_depth","initial_incoming_debt","policy"});
    require(initial.at("schema")=="llaminar.mtp.cpu_depth_transaction_trace_initial.v3","wrong CPU initial schema");require(initial.at("compiled_identity")==j.at("compiled_identity"),"CPU initial/origin source identity mismatch");
    keys(j.at("terminal"),{"transactions","current_depth","evaluated_windows","emitted_tokens","published_state_tokens","model_stopped","utility_probes","utility_keeps","utility_returns"});
    const auto config=hostPolicy(initial.at("policy"));const auto verify=static_cast<MTPVerifyMode>(integer(initial.at("verify_mode"),0,1));
    require(config.cooldown_steps<=INT32_MAX-static_cast<int>(j.at("transactions").size()),
        "CPU cooldown exceeds safe finite replay horizon");
    MTPDepthController controller(config,static_cast<int>(integer(initial.at("configured_draft_tokens"),1,15)),verify);
    require(controller.currentDepth()==integer(initial.at("initial_depth"),1,15),"CPU initial/config mismatch");
    for(const char *counter:{"utility_probes","utility_keeps","utility_returns"})
        (void)integer(j.at("terminal").at(counter),0,static_cast<std::int64_t>(j.at("transactions").size()));
    require(initial.at("initial_incoming_debt").is_boolean()&&j.at("terminal").at("model_stopped").is_boolean(),
        "CPU debt/stop witnesses require exact bools");
    int observed_depth=controller.currentDepth(),observed_windows=0,observed_emitted=0,observed_published=0;
    bool observed_leading=initial.at("initial_incoming_debt").get<bool>(),observed_stopped=false;
    for(const auto &raw:j.at("transactions"))
    {
        const auto v=values(raw);require(v.pre_depth==observed_depth&&v.requested_depth==v.pre_depth,
            "CPU origin depth continuity mismatch");
        require(v.effective_depth>=0&&v.effective_depth<=v.requested_depth&&v.accepted_prefix<=v.effective_depth&&
            v.token_witness+static_cast<int>(v.incoming_debt)<=v.effective_depth+1,"invalid CPU effective input");
        require(v.evaluated_windows>=observed_windows&&v.evaluated_windows<=observed_windows+1,"CPU origin window continuity mismatch");
        MTPPublicationWitness publication;
        require(!observed_stopped&&v.incoming_debt==observed_leading&&
            MTPPublicationWitness::fromEmitted(static_cast<uint32_t>(v.token_witness),
                v.incoming_debt,v.outgoing_debt,v.model_stopped,&publication),"CPU origin publication mismatch");
        observed_published+=static_cast<int>(publication.publishedStateTokens());
        observed_leading=!v.model_stopped&&v.outgoing_debt;observed_stopped=v.model_stopped;
        observed_depth=v.post_depth;observed_windows=v.evaluated_windows;observed_emitted+=v.token_witness;
    }
    require(observed_emitted==integer(j.at("terminal").at("emitted_tokens"),1,256),"CPU independent emitted-token witness mismatch");
    require(observed_published==integer(j.at("terminal").at("published_state_tokens"),0,257)&&
        observed_stopped==j.at("terminal").at("model_stopped").get<bool>(),"CPU independent publication/stop witness mismatch");
    require(observed_depth==integer(j.at("terminal").at("current_depth"),1,15)&&
        observed_windows==integer(j.at("terminal").at("evaluated_windows"),0,256),"CPU origin terminal witness mismatch");
    Json decisions=Json::array(),windows=Json::array();
    for(size_t i=0;i<j.at("transactions").size();++i)
    {
        const auto v=values(j.at("transactions")[i]);
        if(candidate_mode&&v.pre_depth!=controller.currentDepth())
            return {{"status","unsupported_candidate_depth"},{"consumed_transactions",i},{"first_divergence_transaction",i+1},
                {"expected_depth",v.pre_depth},{"candidate_depth",controller.currentDepth()},{"decisions",decisions},{"economy_claim",false}};
        require(v.pre_depth==controller.currentDepth()&&v.requested_depth==controller.requestedDepthForStep(),"CPU pre/requested depth mismatch");
        require(v.effective_depth>=0&&v.effective_depth<=v.requested_depth&&v.accepted_prefix<=v.effective_depth&&
            v.token_witness+static_cast<int>(v.incoming_debt)<=v.effective_depth+1,"invalid CPU effective input");
        MTPPublicationWitness publication;
        require(MTPPublicationWitness::fromEmitted(static_cast<uint32_t>(v.token_witness),v.incoming_debt,
            v.outgoing_debt,v.model_stopped,&publication),"CPU publication witness rejected");
        const auto d=controller.recordStep({.requested_depth=v.requested_depth,.effective_depth=v.effective_depth,
            .accepted_speculative_prefix=v.accepted_prefix,.budget_limited=v.budget_limited,.rollback=v.rollback,
            .newly_emitted_tokens=static_cast<std::uint32_t>(v.token_witness),.publication=publication});
        if(!candidate_mode)require(d.new_depth==v.post_depth&&d.recommended_depth==v.recommended_depth&&controller.stats().windows==static_cast<uint64_t>(v.evaluated_windows),"CPU post-state/window mismatch");
        decisions.push_back({{"ordinal",i+1},{"old_depth",d.old_depth},{"recommended_depth",d.recommended_depth},{"new_depth",d.new_depth},
            {"evaluated_windows",controller.stats().windows},{"reason",toString(d.reason)}});
        decisions.back().update(observedInputs(v));
        decisions.back()["newly_emitted_tokens"]=v.token_witness;
        decisions.back()["published_state_tokens"]=publication.publishedStateTokens();
        // Borrow the actual mode-exclusive measured owner, never last_decision_.
        if(const auto *confirmation=controller.proposalConfirmation())
            decisions.back()["promotion_streak"]=confirmation->completed_windows;
        decisions.back()["utility_probes"]=controller.stats().utility_probes;
        decisions.back()["utility_keeps"]=controller.stats().utility_keeps;
        decisions.back()["utility_returns"]=controller.stats().utility_returns;
        if(d.evaluated)windows.push_back({{"transaction",i+1},{"runs",d.window.verifier_runs},{"accepted",d.window.accepted_draft_tokens},
            {"zero",d.window.zero_accepts},{"full",d.window.full_accepts},{"old_depth",d.old_depth},{"recommended_depth",d.recommended_depth},{"new_depth",d.new_depth}});
        if(candidate_mode&&d.new_depth!=v.post_depth)
            return {{"status","unsupported_candidate_depth"},{"consumed_transactions",i+1},{"first_divergence_transaction",i+1},
                {"expected_depth",v.post_depth},{"candidate_depth",d.new_depth},{"decisions",decisions},{"economy_claim",false}};
    }
    if(!candidate_mode)require(controller.stats().utility_probes==static_cast<std::uint64_t>(integer(j.at("terminal").at("utility_probes"),0,256))&&
        controller.stats().utility_keeps==static_cast<std::uint64_t>(integer(j.at("terminal").at("utility_keeps"),0,256))&&
        controller.stats().utility_returns==static_cast<std::uint64_t>(integer(j.at("terminal").at("utility_returns"),0,256)),"CPU terminal utility counters mismatch");
    if(!candidate_mode)require(controller.currentDepth()==integer(j.at("terminal").at("current_depth"),1,15)&&
        controller.stats().windows==static_cast<uint64_t>(integer(j.at("terminal").at("evaluated_windows"),0,256)),"CPU terminal depth/window mismatch");
    return {{"status",candidate_mode?"same_observed_depth_path":"complete_baseline_replay"},{"authority","cpu_host"},{"transactions",j.at("transactions").size()},
        {"final_depth",controller.currentDepth()},{"emitted_tokens",observed_emitted},
        {"published_state_tokens",observed_published},{"model_stopped",observed_stopped},
        {"evaluated_windows",controller.stats().windows},{"updates",controller.stats().updates},
        {"promotions",controller.stats().promotions},{"demotions",controller.stats().demotions},
        {"utility_probes",controller.stats().utility_probes},{"utility_keeps",controller.stats().utility_keeps},
        {"utility_returns",controller.stats().utility_returns},{"decisions",decisions},{"windows",windows},{"economy_claim",false}};
}
}
Json compiledIdentity()
{
    Json hashes=Json::object();for(const auto &[key,value]:mtp_depth_replay_identity::kSourceHashes)hashes[std::string(key)]=std::string(value);
    return {{"schema",std::string(mtp_depth_replay_identity::kSchema)},{"sources",hashes}};
}
Json parseDocument(std::string_view input)
{
    std::vector<std::set<std::string>> objects;
    return Json::parse(input.begin(),input.end(),[&](int,Json::parse_event_t event,Json &value){
        if(event==Json::parse_event_t::object_start)objects.emplace_back();
        if(event==Json::parse_event_t::key)require(objects.back().insert(value.get<std::string>()).second,"duplicate JSON member");
        if(event==Json::parse_event_t::object_end)objects.pop_back();return true;});
}
Json replay(const Json &j)
{
    envelope(j);const bool candidate=j.at("intent")=="candidate";
    auto result=j.at("authority")=="native_resident"?resident(j,nullptr,candidate):host(j,candidate);
    result["intent"]=j.at("intent");result["origin_compiled_identity"]=j.at("compiled_identity");
    result["current_compiled_identity"]=compiledIdentity();result["capture"]=j.at("capture");
    result["baseline_validated_by_this_executable"]=!candidate;result["initial_state_semantics"]="captured_actual_fresh_state";return result;
}
Json replayResidentCandidate(const Json &j,const DeviceGenerationPolicy &candidate)
{
    auto request=j;request["intent"]="candidate";envelope(request);
    require(request.at("authority")=="native_resident","candidate resident policy needs resident input");
    auto result=resident(request,&candidate,true);result["intent"]="candidate";
    result["origin_compiled_identity"]=request.at("compiled_identity");result["current_compiled_identity"]=compiledIdentity();
    result["baseline_validated_by_this_executable"]=false;return result;
}
int runCLI(int argc,char **argv)
{
    try
    {
        require(argc==3&&std::string_view(argv[1])=="--replay-depth-trace","use --replay-depth-trace <absolute request.json>");
        const std::filesystem::path input=argv[2];require(input.is_absolute()&&std::filesystem::is_regular_file(input)&&std::filesystem::file_size(input)<=1024*1024,"invalid/oversized request file");
        std::ifstream in(input);std::stringstream bytes;bytes<<in.rdbuf();const auto request=parseDocument(bytes.str());
        const std::filesystem::path output=text(request.at("output"));const auto temporary=output.string()+".tmp";
        require(output.is_absolute()&&std::filesystem::is_directory(output.parent_path())&&!std::filesystem::exists(output)&&
            !std::filesystem::is_symlink(output)&&!std::filesystem::exists(temporary)&&!std::filesystem::is_symlink(temporary),"output must be a fresh absolute file");
        const auto result=replay(request);std::ofstream out(temporary,std::ios::binary);out<<result.dump(2)<<'\n';out.close();require(bool(out),"result write failed");std::filesystem::rename(temporary,output);
        std::cout<<"MTP_DEPTH_TRACE_REPLAY "<<output.string()<<'\n';return 0;
    }
    catch(const std::exception &error){std::cerr<<"MTP_DEPTH_TRACE_REPLAY_REJECTED "<<error.what()<<'\n';return 1;}
}
}
