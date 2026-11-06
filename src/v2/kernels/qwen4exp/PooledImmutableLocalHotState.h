/**
 * @file PooledImmutableLocalHotState.h
 * @brief One bounded native query frame over the existing canonical cache metadata.
 *
 * Adaptive FrameControl owns cache-policy progress. This separate record owns
 * only invocation identity and consumed/discarded response lifetime. It is not
 * a residency map: slots/directory/frequency remain unchanged Operations data.
 */
#pragma once
#include "PooledImmutableLocalHotAccess.h"
#include "PooledImmutableLocalHotTransportProtocol.h"
namespace llaminar2::qwen4exp::local_hot {
/** @brief Distinct response-consumption frontiers, never inferred from L2 receipt alone. */
enum class Disposition : std::uint32_t { Idle,Probing,Probed,DataReady,Consuming,Consumed,Discarded,Cancelled };
/** @brief Exact current frame identity; only its owning native stages may mutate it. */
struct alignas(64) FrameIdentity {
    local_hot_transport::QueryIdentity query{};
    PooledImmutableLocalHotAccessIntent intent=PooledImmutableLocalHotAccessIntent::CurrentInvocation; ///< Effective consumption intent; immutable wire purpose remains FutureOrigin after adoption.
    Disposition disposition=Disposition::Idle;
    std::uint64_t retained_native_bytes=0;
    adaptive_lookup::Status terminal_error=adaptive_lookup::Status::Success;
    std::uint32_t flags=0;
    std::uint64_t source_request_invocation=0; ///< Last actual once-per-request I/O invocation; survives query/reset.
};
static_assert(sizeof(FrameIdentity)==64);
inline constexpr std::uint32_t ReadCompleted=1u,FutureOrigin=2u,LateAdoption=4u,PredictionDiscard=8u;
/** @brief Actual consumer identity; a matching row prediction cannot replace request authority. */
struct DemandIdentity {
    local_hot_transport::QueryIdentity query{};
    std::uint64_t source_request_invocation=0;
};
/** @brief Explicit local adoption result; prediction mismatch is reusable discard, never success. */
enum class AdoptionResult : std::uint32_t {
    Accepted,
    PredictionMiss,
    InvalidIdentity,
    InvalidState,
};
/** @brief Order-independent integer-OR comparison bits shared by host replay and native reduction. */
enum class DemandComparison : std::uint32_t {
    Equal=0,
    DifferentRow=1,
    InvalidRecord=2,
};
/** @brief Complete original demand prefix; no storage, pin or execution authority is minted. */
template<class Row> struct ActualDemand {
    DemandIdentity identity{};
    const Row* ordered_ids=nullptr;
    std::uint32_t original_ids=0;
};
/**
 * @brief Actual completed transition counters; original IDs and unique native rows stay distinct.
 * Receive-completed wire bytes are projected by the original receipt producer;
 * no field makes an admission or residency decision. No timing
 * probe is needed: on-time/late are relative to actual demand adoption.
 */
struct alignas(64) Counters {
    std::uint64_t read_original_ids=0,local_hit_original_ids=0,all_hit_wait_avoided_original_ids=0;
    std::uint64_t consumed_original_ids=0,future_on_time_native_rows=0,future_late_native_rows=0;
    std::uint64_t future_useful_native_rows=0,future_wasted_native_rows=0;
    std::uint64_t promotions=0,evictions=0;
    std::uint64_t request_key_payload_bytes=0,native_result_payload_bytes=0;
    std::uint64_t lookup_control_header_bytes=0,acknowledgement_bytes=0,request_admission_bytes=0;
    std::uint64_t reserved=0; ///< Always zero in the live producer record; no execution state is stored here.
    /** @return Exact immutable observation equality, including every receive-completed category. */
    bool operator==(const Counters&)const=default;
};
static_assert(sizeof(Counters)==128);
} // namespace llaminar2::qwen4exp::local_hot
