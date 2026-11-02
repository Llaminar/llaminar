/**
 * @file PooledNgramReplaySchedule.h
 * @brief One fixed immutable prelude/sample trajectory shared by lookup timing actors.
 *
 * Two untimed preludes establish the declared input history. Warm controls
 * repeat their own bank from empty capacity; other scenarios retain disjoint
 * preludes without claiming a cache outcome. Measured
 * frames are never selected from cache observations. Actors retain their actual
 * placement/tier policy and report observed work rather than inventing it.
 */
#pragma once
#include "PooledNgramReplayTrace.h"
#include <string_view>
namespace llaminar2::test::ngram_replay {
inline constexpr std::uint32_t PreludeFrames=2;
inline constexpr const char* Schedule="policy_independent_preludes_and_alternating_samples_v2";
/** @brief Closed measured traffic declarations; duplicate control remains separate. */
enum class Scenario {WarmHits,AlternatingSets,Mixed,Churn,DuplicateControl,Empty};
/** @brief One input bank and nominal diagnostic phase, without any runtime state. */
struct ScheduledFrame {Pattern pattern;std::string_view phase;};
/**
 * @brief Return one original untimed prelude or measured frame.
 * @param scenario Fixed traffic declaration.
 * @param prelude True only for the two required untimed frames.
 * @param ordinal Ordinal within the prelude or measured samples, respectively.
 * @return Immutable original input bank and phase.
 * @throws std::invalid_argument For unknown scenarios or extra prelude frames.
 */
inline ScheduledFrame scheduledFrame(Scenario scenario,bool prelude,std::uint64_t ordinal){
    switch(scenario){case Scenario::WarmHits:case Scenario::AlternatingSets:case Scenario::Mixed:
        case Scenario::Churn:case Scenario::DuplicateControl:case Scenario::Empty:break;
        default:throw std::invalid_argument("Unknown fixed ngram replay scenario");}
    if(prelude){if(ordinal>=PreludeFrames)throw std::invalid_argument("Ngram replay has exactly two declared prelude frames");
        if(scenario==Scenario::WarmHits)return {Pattern::DistinctA,"prelude_a"};
        if(scenario==Scenario::DuplicateControl)return {Pattern::DuplicateHeavy,"prelude_duplicate_a"};
        if(!ordinal)return {Pattern::DistinctB,"prelude_b"};
        if(scenario==Scenario::Churn)return {Pattern::DistinctC,"prelude_c"};
        if(scenario==Scenario::DuplicateControl)return {Pattern::DuplicateHeavy,"prelude_duplicate_a"};
        return {Pattern::DistinctA,"prelude_a"};}
    switch(scenario){
    case Scenario::WarmHits:return {Pattern::DistinctA,"warm_hits"};
    case Scenario::AlternatingSets:return {ordinal%2?Pattern::DistinctC:Pattern::DistinctB,"alternating_sets"};
    case Scenario::Mixed:return {ordinal%2?Pattern::MixedC:Pattern::MixedB,"mixed"};
    case Scenario::Churn:return {ordinal%2?Pattern::DistinctB:Pattern::DistinctA,ordinal==0?"hot_set_a":ordinal%2?"hot_set_b":"hot_set_a_return"};
    case Scenario::DuplicateControl:return {Pattern::DuplicateHeavy,"warm_hits"};
    case Scenario::Empty:return {Pattern::Empty,"empty"};
    }
    throw std::invalid_argument("Unknown fixed ngram replay scenario");
}
/**
 * @brief Select one invocation from either matched two-prelude or explicit no-prelude execution.
 * @param scenario Immutable traffic declaration.
 * @param preludes Two for matched comparisons; zero for separately labelled file-cold controls.
 * @param index Monotonic invocation ordinal including any preludes.
 * @return Exact shared prelude/sample bank and phase.
 * @throws std::invalid_argument For an undeclared prelude count.
 */
inline ScheduledFrame scheduledInvocation(Scenario scenario,std::uint32_t preludes,std::uint64_t index){
    if(preludes!=0 && preludes!=PreludeFrames)throw std::invalid_argument("Ngram comparisons require two preludes, or explicit zero-prelude cold control");
    return index<preludes?scheduledFrame(scenario,true,index):scheduledFrame(scenario,false,index-preludes);
}
/**
 * @brief Fixed actual authored-remote lifecycle inputs, independent of device/cache observations.
 * @param shape Original admitted trace geometry. @param ordinal Replay position within one request.
 * @return Large, small, empty, then full-capacity duplicate-heavy input, repeated identically after reset.
 * @throws std::invalid_argument For invalid original trace geometry, through the shared author.
 */
inline std::vector<std::uint64_t> authoredRemoteLifecycleIds(Shape shape,std::uint64_t ordinal){
    const auto phase=ordinal%4;
    shape.tokens=(phase==0 || phase==3)?shape.capacity_tokens:1;
    return ids(shape,phase==2?Pattern::Empty:phase==3?Pattern::DuplicateHeavy:Pattern::DistinctA);
}
} // namespace llaminar2::test::ngram_replay
