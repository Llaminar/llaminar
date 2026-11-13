/**
 * @file MTPDepthTransactionTrace.h
 * @brief Lossless bounded controller inputs in one existing diagnostic bank.
 *
 * One fresh scalar response has at most256 committed nonempty transactions.
 * The original TransactionCount owns record indexing; this evidence never
 * supplies model/sampler/controller values or changes captured execution.
 * Actual initial device control is retained at the validated boundary before
 * the first response append. External producer counters are not replay inputs.
 */
#pragma once
#include "DeviceGenerationInitialization.h"
#include "../../execution/mtp/MTPPublicationWitness.h"
#include "DeviceGenerationControlLayout.h"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MTP_TRACE_HD __host__ __device__
#else
#define LLAMINAR_MTP_TRACE_HD
#endif
namespace llaminar2::sampling_math
{
/** @brief Exactly one interpretation of the already admitted physical bank. */
enum class MTPDiagnosticRecordMode : std::uint8_t { Disabled, FirstTransaction, DepthTransactionTrace };
/** @brief Reject competing first-transaction and complete-transaction writers.
 * @param phase Existing phase diagnostic selection.
 * @param mirror Existing mirror diagnostic selection.
 * @param trace Explicit transaction trace selection.
 * @return One immutable mode, without automatic substitution.
 * @throws std::invalid_argument For conflicting diagnostic owners.
 */
inline MTPDiagnosticRecordMode resolveMTPDiagnosticRecordMode(bool phase, bool mirror, bool trace)
{
    if (trace && (phase || mirror)) throw std::invalid_argument("MTP transaction trace conflicts with first-transaction diagnostics");
    return trace ? MTPDiagnosticRecordMode::DepthTransactionTrace
        : phase || mirror ? MTPDiagnosticRecordMode::FirstTransaction : MTPDiagnosticRecordMode::Disabled;
}
/** @brief Finite request/collector lifetime, never an HTTP history buffer. */
enum class MTPDepthTransactionTraceScope { BoundedBenchmark, Other };
/** @brief Require exactly one measured, unwarmed request per collector epoch.
 * @param scope Actual application entrypoint. @param warmups Warmup requests.
 * @param measurements Measured requests sharing a collector epoch.
 * @return Whether bounded raw-record export cannot silently merge requests.
 */
inline constexpr bool validMTPDepthTransactionTraceScope(MTPDepthTransactionTraceScope scope, int warmups, int measurements) noexcept
{
    return scope == MTPDepthTransactionTraceScope::BoundedBenchmark && warmups == 0 && measurements == 1;
}
inline constexpr int kMTPDepthTransactionTraceCapacity = 256;
inline constexpr std::uint32_t kMTPDepthTransactionTraceMagic = 0x4d545054u;
inline constexpr std::uint32_t kMTPDepthTransactionTraceVersion = 3;
/** @brief Exact scalar inputs and post-observation witnesses; no rounded rates. */
struct MTPDepthTransactionTraceValues
{
    int pre_depth = 0, accepted_prefix = 0, post_depth = 0;
    bool rollback = false, budget_limited = false;
    bool incoming_debt = false, outgoing_debt = false, model_stopped = false;
    int requested_depth = 0, effective_depth = 0;
    bool host_observation = false;
    int recommended_depth = 0, token_witness = 0, evaluated_windows = 0;
};
/** @brief Three native words; the owner tag fixes the token witness unit.
 * Resident witnesses are cumulative response counts; host witnesses are actual
 * emitted tokens in this transaction. Bits29/30/31 retain actual incoming debt,
 * raw outgoing publication debt and model stop. Published progress is verified
 * independently against the actual terminal publisher ledger, never acceptance.
 */
struct MTPDepthTransactionTraceRecord
{
    std::uint32_t packed;
    std::int32_t token_witness, evaluated_windows;
};
static_assert(sizeof(MTPDepthTransactionTraceRecord) == 12);
inline constexpr int kMTPDepthTransactionTraceRecordWords = 3;
/** @brief Validate then encode every field without masking invalid input.
 * @param v Actual observation; GPU-only unused host fields must be zero.
 * @param out Written only after the complete contract is proved.
 * @return False for an unrepresentable or wrong-owner observation.
 */
LLAMINAR_MTP_TRACE_HD inline bool packMTPDepthTransactionTraceRecord(const MTPDepthTransactionTraceValues &v,
    MTPDepthTransactionTraceRecord *out) noexcept
{
    if (!out || v.pre_depth < 0 || v.pre_depth > 15 || v.accepted_prefix < 0 || v.accepted_prefix > 15 ||
        v.post_depth < 0 || v.post_depth > 15 || v.recommended_depth < 0 || v.recommended_depth > 15 ||
        v.requested_depth < 0 || v.requested_depth > 31 || v.effective_depth < 0 || v.effective_depth > 31 ||
        v.evaluated_windows < 0 || v.evaluated_windows > kMTPDepthTransactionTraceCapacity ||
        (v.host_observation ? (v.token_witness <= 0 || v.token_witness > 16) :
            (v.pre_depth == 0 || v.post_depth == 0 || v.accepted_prefix > v.pre_depth ||
             v.requested_depth != 0 || v.effective_depth != 0 ||
             v.token_witness <= 0 || v.token_witness > kMTPDepthTransactionTraceCapacity))) return false;
    MTPPublicationWitness publication;
    if (v.host_observation && !MTPPublicationWitness::fromEmitted(
            static_cast<std::uint32_t>(v.token_witness), v.incoming_debt, v.outgoing_debt,
            v.model_stopped, &publication)) return false;
    out->packed = static_cast<std::uint32_t>(v.pre_depth) |
        (static_cast<std::uint32_t>(v.accepted_prefix) << 4) | (static_cast<std::uint32_t>(v.post_depth) << 8) |
        (static_cast<std::uint32_t>(v.rollback) << 12) | (static_cast<std::uint32_t>(v.budget_limited) << 13) |
        (static_cast<std::uint32_t>(v.requested_depth) << 14) | (static_cast<std::uint32_t>(v.effective_depth) << 19) |
        (static_cast<std::uint32_t>(v.host_observation) << 24) | (static_cast<std::uint32_t>(v.recommended_depth) << 25) |
        (static_cast<std::uint32_t>(v.incoming_debt) << 29) | (static_cast<std::uint32_t>(v.outgoing_debt) << 30) |
        (static_cast<std::uint32_t>(v.model_stopped) << 31);
    out->token_witness = v.token_witness;
    out->evaluated_windows = v.evaluated_windows;
    return true;
}
/** @brief Decode the closed v3 schema; its three publication bits use the former reserved space.
 * @param record Existing immutable record. @param out Exact decoded values.
 * @return False without updating output for malformed or unrepresentable bytes.
 */
LLAMINAR_MTP_TRACE_HD inline bool unpackMTPDepthTransactionTraceRecord(const MTPDepthTransactionTraceRecord &record,
    MTPDepthTransactionTraceValues *out) noexcept
{
    if (!out) return false;
    const auto p = record.packed;
    const MTPDepthTransactionTraceValues v{
        .pre_depth = static_cast<int>(p & 15u), .accepted_prefix = static_cast<int>((p >> 4) & 15u),
        .post_depth = static_cast<int>((p >> 8) & 15u),
        .rollback = ((p >> 12) & 1u) != 0, .budget_limited = ((p >> 13) & 1u) != 0,
        .incoming_debt = ((p >> 29) & 1u) != 0, .outgoing_debt = ((p >> 30) & 1u) != 0,
        .model_stopped = ((p >> 31) & 1u) != 0,
        .requested_depth = static_cast<int>((p >> 14) & 31u),
        .effective_depth = static_cast<int>((p >> 19) & 31u), .host_observation = ((p >> 24) & 1u) != 0,
        .recommended_depth = static_cast<int>((p >> 25) & 15u),
        .token_witness = record.token_witness, .evaluated_windows = record.evaluated_windows};
    MTPDepthTransactionTraceRecord checked;
    if (!packMTPDepthTransactionTraceRecord(v, &checked)) return false;
    *out = v; return true;
}
/** @brief Actual initial control plus ordered committed observations in the old bank. */
struct MTPDepthTransactionTrace
{
    std::uint32_t magic, version;
    std::int32_t record_words, capacity;
    std::int32_t initial_control[kDeviceGenerationControlCount];
    MTPDepthTransactionTraceRecord records[kMTPDepthTransactionTraceCapacity];
};
static_assert(sizeof(MTPDepthTransactionTrace) == 3520);
static_assert(std::is_trivially_copyable_v<MTPDepthTransactionTrace>);
/** @brief Exact fresh response capacity without a parallel transaction cursor.
 * @param kind Actual admission lifetime. @param requests Actual independent rows.
 * @param tokens Admitted positive generated-token budget.
 * @return Whether the existing nonempty-transaction invariant bounds all records.
 */
inline constexpr bool validMTPDepthTransactionTraceAdmission(DeviceGenerationAdmissionKind kind, int requests, int tokens) noexcept
{
    return kind == DeviceGenerationAdmissionKind::NewRequest && requests == 1 && tokens > 0 && tokens <= kMTPDepthTransactionTraceCapacity;
}
/** @brief Exact populated terminal span; inactive tail capacity never transfers.
 * @param records Authenticated terminal TransactionCount.
 * @return Zero for no records, otherwise header, initial control and actual records.
 * @throws std::out_of_range For an impossible terminal extent.
 */
inline std::size_t mtpDepthTransactionTraceBytes(int records)
{
    if (records < 0 || records > kMTPDepthTransactionTraceCapacity) throw std::out_of_range("MTP transaction trace count exceeds256 records");
    return records == 0 ? 0 : offsetof(MTPDepthTransactionTrace, records) + static_cast<std::size_t>(records) * sizeof(MTPDepthTransactionTraceRecord);
}
} // namespace llaminar2::sampling_math
#undef LLAMINAR_MTP_TRACE_HD
