/**
 * @file ReadAheadScheduling.h
 * @brief Immutable phase-specific lookup scheduling, independent of row geometry.
 *
 * One-token prefill and decode remain different phases. A decode-only resident
 * candidate deliberately preserves the consumer-scheduled prefill policy; a
 * failed or unqualified decode never changes its selected lane. This small
 * header is shared by graph construction, cache identity and device-free replay.
 */
#pragma once
#include <cstdint>
#include <stdexcept>
#include <span>
#include <string_view>

namespace llaminar2 {
/** @brief Mathematical phase selected explicitly, never inferred from row count. */
enum class ForwardExecutionPhase : std::uint8_t
{
    Prefill, ///< Prompt ingestion with the configured prefill topology.
    Decode, ///< Serial or grouped decode-equivalent execution.
};
/** @brief Explicit immutable-source schedule; candidate selections are never automatic. */
enum class ImmutableLookupSchedule : std::uint8_t
{
    ConsumerScheduled, ///< Every phase gathers at its first consumer.
    EarlyCurrentHostedCandidate, ///< Every supported phase splits hosted publication/consumption.
    EarlyCurrentResidentDecodeGPUCandidate, ///< Resident prefill stays consumer-scheduled; decode uses a side lane.
};
/** @brief Parse one complete diagnostic token; aliases and partial matches are forbidden.
 * @param value Exact setup-only token from the canonical DebugEnv parser.
 * @return Original typed schedule, never a substituted automatic mode.
 * @throws std::invalid_argument For empty, unknown, truncated or whitespace-bearing values.
 */
[[nodiscard]] constexpr ImmutableLookupSchedule parseImmutableLookupSchedule(std::string_view value)
{
    if(value=="consumer_scheduled")return ImmutableLookupSchedule::ConsumerScheduled;
    if(value=="early_current_hosted_candidate")return ImmutableLookupSchedule::EarlyCurrentHostedCandidate;
    if(value=="early_current_resident_decode_gpu_candidate")return ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate;
    throw std::invalid_argument("LLAMINAR_QFN_NGRAM_LOOKUP_SCHEDULE requires consumer_scheduled, early_current_hosted_candidate or early_current_resident_decode_gpu_candidate");
}
/** @brief One concrete invocation's required implementation. */
enum class ImmutableLookupInvocationSchedule : std::uint8_t
{
    ConsumerScheduled, ///< Original consumer-local gather implementation.
    HostedPublishConsume, ///< Original retained host-service Publish/Consume protocol.
    ResidentSideLane, ///< Exact-event native side-lane gather and joined consumer.
};
/** @brief Resolve policy and phase without observing model/controller/device state.
 * @throws std::invalid_argument For an unknown policy or phase; no substituted implementation exists.
 */
[[nodiscard]] constexpr ImmutableLookupInvocationSchedule resolveImmutableLookupInvocationSchedule(
    ImmutableLookupSchedule policy,ForwardExecutionPhase phase)
{
    if(phase!=ForwardExecutionPhase::Prefill && phase!=ForwardExecutionPhase::Decode)
        throw std::invalid_argument("Unknown immutable lookup invocation phase");
    switch(policy)
    {
    case ImmutableLookupSchedule::ConsumerScheduled:return ImmutableLookupInvocationSchedule::ConsumerScheduled;
    case ImmutableLookupSchedule::EarlyCurrentHostedCandidate:return ImmutableLookupInvocationSchedule::HostedPublishConsume;
    case ImmutableLookupSchedule::EarlyCurrentResidentDecodeGPUCandidate:
        return phase==ForwardExecutionPhase::Prefill?ImmutableLookupInvocationSchedule::ConsumerScheduled:
            ImmutableLookupInvocationSchedule::ResidentSideLane;
    }
    throw std::invalid_argument("Unknown immutable lookup scheduling policy");
}
/** @brief Complete typed phase/implementation capture identity, with a fixed schema version. */
[[nodiscard]] constexpr std::uint64_t immutableLookupPhaseIdentity(
    ForwardExecutionPhase phase,ImmutableLookupInvocationSchedule schedule)
{
    if((phase!=ForwardExecutionPhase::Prefill && phase!=ForwardExecutionPhase::Decode) ||
       (schedule!=ImmutableLookupInvocationSchedule::ConsumerScheduled &&
        schedule!=ImmutableLookupInvocationSchedule::HostedPublishConsume &&
        schedule!=ImmutableLookupInvocationSchedule::ResidentSideLane))
        throw std::invalid_argument("Invalid immutable lookup phase identity");
    return (1ull<<32) | ((static_cast<std::uint64_t>(phase)+1)<<8) |
        (static_cast<std::uint64_t>(schedule)+1);
}
/**
 * @brief Immutable recording authority distinct from complete-parent submission.
 *
 * Cache setup plans the existing owned recording stream before preparation and
 * identity calculation. Sealing authenticates the actual capture object's stream
 * before recording. Sealing changes permission, never pointer identity, so first
 * materialization cannot masquerade as a variant change requiring recapture.
 */
class ReadAheadRecordingBinding final
{
public:
    /** @brief Bind source-owned recording/submission streams once, without assuming equality. */
    bool plan(void* recording,void* submission)
    {
        if(!recording || !submission)throw std::invalid_argument("Read-ahead stream planning requires exact non-null authorities");
        if(recording_ && (recording_!=recording || submission_!=submission))
            throw std::logic_error("Read-ahead recording/submission authority changed");
        if(recording_)return false;
        recording_=recording;submission_=submission;return true;
    }
    /** @brief Authenticate the actual capture object's owned stream after preparation. */
    void seal(void* actual)
    {require(actual);sealed_=true;}
    /** @brief Reject an unrelated stream before using prepared addresses. */
    void require(void* actual)const
    {if(!recording_ || actual!=recording_)throw std::logic_error("Read-ahead uses a stale or unplanned recording stream");}
    /** @brief Require the authenticated capture owner before any event/node recording. */
    void requireSealed(void* actual)const
    {require(actual);if(!sealed_)throw std::logic_error("Read-ahead recording owner was not sealed before capture");}
    /** @return Complete pre-prepare/post-seal identity; permission state is deliberately excluded. */
    std::uint64_t identity()const noexcept
    {
        if(!recording_)return 0;
        auto value=static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(recording_));
        value^=static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(submission_));
        value*=1099511628211ull;
        // Retain both coordinates even when the two authorities happen to match.
        value^=static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(recording_));
        return value?value:1;
    }
private:
    void* recording_=nullptr;void* submission_=nullptr;bool sealed_=false;
};

/** @brief Source-declared closure policy; never selected after a native failure. */
enum class ReadAheadJoinPolicy : std::uint8_t
{
    AtConsumer, ///< One native unit contains producer and consumer.
    FirstDeclaredCaptureFrontier, ///< Close at the first original host cutpoint or consumer, whichever is earlier.
};
/** @brief One immutable source graph frontier, independent of request execution state. */
struct ReadAheadJoinFrontier
{
    std::string_view name;
    enum class Kind : std::uint8_t {Ordinary,BeforeDeclaredHostBoundary,TransactionTerminal};
    Kind kind=Kind::Ordinary;
};
/**
 * @brief Resolve the exact cold join anchor while preserving later consumer ownership.
 * @return Existing primary node at which the side result becomes ordered.
 * @throws std::invalid_argument For missing/reordered producers or an unsupported capture crossing.
 */
[[nodiscard]] inline std::string_view resolveReadAheadJoinAnchor(
    std::span<const ReadAheadJoinFrontier> order,std::string_view fork,std::string_view producer,
    std::string_view consumer,ReadAheadJoinPolicy policy)
{
    if(policy!=ReadAheadJoinPolicy::AtConsumer && policy!=ReadAheadJoinPolicy::FirstDeclaredCaptureFrontier)
        throw std::invalid_argument("Unknown read-ahead join placement policy");
    bool sawFork=false,sawProducer=false;
    for(const auto& node:order)
    {
        if(node.kind!=ReadAheadJoinFrontier::Kind::Ordinary &&
           node.kind!=ReadAheadJoinFrontier::Kind::BeforeDeclaredHostBoundary &&
           node.kind!=ReadAheadJoinFrontier::Kind::TransactionTerminal)
            throw std::invalid_argument("Unknown declared read-ahead frontier kind");
        if(node.name==fork){if(sawFork || sawProducer)throw std::invalid_argument("Repeated/reordered read-ahead fork");sawFork=true;}
        if(node.name==producer){if(!sawFork || sawProducer)throw std::invalid_argument("Read-ahead producer precedes its fork");sawProducer=true;}
        if(node.name==consumer)
        {
            if(!sawProducer)throw std::invalid_argument("Read-ahead consumer precedes its producer");
            return consumer;
        }
        if(sawFork && node.kind==ReadAheadJoinFrontier::Kind::BeforeDeclaredHostBoundary)
        {
            if(!sawProducer || policy==ReadAheadJoinPolicy::AtConsumer)
                throw std::invalid_argument("Read-ahead cannot cross an unjoined declared host boundary");
            // The actual later consumer must still exist and remain ordered.
            bool found=false;
            for(const auto& later:order)if(later.name==consumer)found=true;
            if(!found)throw std::invalid_argument("Read-ahead lost its actual later PLE consumer");
            return node.name;
        }
    }
    throw std::invalid_argument("Read-ahead join plan has missing fork/producer/consumer");
}

}
