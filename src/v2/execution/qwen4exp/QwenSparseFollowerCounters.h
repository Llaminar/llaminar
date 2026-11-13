/**
 * @file QwenSparseFollowerCounters.h
 * @brief Exact follower counter reset and immutable terminal-result projection.
 *
 * The caller borrows its original device-generation control bank and existing
 * terminal-ready stream/event. Only this final host observation owns copied
 * words; they never select execution or become a live GPU state mirror. No
 * GPU staging region, additional allocator or per-transaction copy is created.
 */
#pragma once
#include "kernels/qwen4exp/QwenSparseTerminalCounters.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
namespace llaminar2 {
/** @brief Frozen terminal reporting ownership; tail's existing DGO bridge must remain its sole projector. */
enum class QwenSparseTerminalOwner { DeviceControllerTail, PipelineFollower };
/** @brief Four exact UINT64 observations within the existing host terminal result. */
struct alignas(8) QwenSparseFollowerCounterObservation {
    static constexpr int first_word=sampling_math::kDeviceGenerationControlQsaPriorHeaderBytesLow;
    // QSA reporting is a closed four-pair schema. Later controller observations
    // belong to their own owners and may never enlarge this follower reset/copy.
    static constexpr int last_word=sampling_math::kDeviceGenerationControlQsaAcceptedPayloadBytesHigh;
    static constexpr int word_count=last_word-first_word+1;
    static constexpr std::size_t byte_count=word_count*sizeof(std::int32_t);
    std::array<std::int32_t,word_count> words{};
    /** @brief Decode only a completed terminal observation, without UINT64 pointer aliasing. */
    qwen4exp::QwenSparseCounterTotals totals()const noexcept {
        const auto pair=[&](qwen4exp::QwenSparseWorkCounter metric){
            return readNativeCounter64(words.data()+static_cast<int>(metric)-first_word);};
        return {pair(qwen4exp::QwenSparseWorkCounter::PriorHeader),pair(qwen4exp::QwenSparseWorkCounter::PriorPayload),
            pair(qwen4exp::QwenSparseWorkCounter::CandidatePayload),pair(qwen4exp::QwenSparseWorkCounter::AcceptedPayload)};
    }
};
static_assert(QwenSparseFollowerCounterObservation::first_word==52);
static_assert(QwenSparseFollowerCounterObservation::last_word==59);
static_assert(QwenSparseFollowerCounterObservation::last_word<sampling_math::kDeviceGenerationControlCount);
static_assert(QwenSparseFollowerCounterObservation::word_count==8);
static_assert(sizeof(QwenSparseFollowerCounterObservation)==32);
/** @brief Enqueue only the original reporting span reset before the existing admission publication.
 * @param row Exact original request-owned controller row, never a model arena scratch bank.
 * @param submit Submission accepting original span and exact32B, on the caller's admission stream.
 * @return False without submission for malformed bindings; native failures remain fatal to admission.
 */
template<class Submit>bool resetQwenSparseFollowerCounters(qwen4exp::QwenSparseWorkCounterRow row,Submit&&submit){
    if(!row.valid())return false;
    return std::forward<Submit>(submit)(row.control+QwenSparseFollowerCounterObservation::first_word,
        QwenSparseFollowerCounterObservation::byte_count);
}
/** @brief Enqueue one exact final field into the existing host terminal result, before its existing event.
 * @param owner Frozen pipeline role; a controller tail submits no duplicate copy.
 * @param row Original request-owned device controller after its complete writer frontier.
 * @param result Host-owned terminal-result field, retained through the existing terminal-ready event.
 * @param submit Explicit terminal copy accepting destination/source/exact32B on the existing stream.
 * @return False without submission for malformed source binding.
 */
template<class Submit>bool observeQwenSparseFollowerCounters(QwenSparseTerminalOwner owner,
    qwen4exp::QwenSparseWorkCounterRow row,QwenSparseFollowerCounterObservation&result,Submit&&submit){
    if(owner==QwenSparseTerminalOwner::DeviceControllerTail)return true;
    if(owner!=QwenSparseTerminalOwner::PipelineFollower || !row.valid())return false;
    return std::forward<Submit>(submit)(result.words.data(),
        row.control+QwenSparseFollowerCounterObservation::first_word,QwenSparseFollowerCounterObservation::byte_count);
}
} // namespace llaminar2
