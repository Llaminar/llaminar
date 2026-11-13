/**
 * @file Qwen4ExpPLERetained.h
 * @brief Exact immutable PLE transaction snapshots and accepted-tail publication.
 *
 * SnapshotPrior must precede the first verifier consumer/mutator of accepted
 * history. Its outputs (including the captured live-count word) are immutable
 * until every publication/republication of that transaction retires. Candidate
 * retention follows the token and normalized-key producers. Publication never
 * reruns the convolution: it splices the exact two-token and nine-row suffixes.
 * All buffers are persistent, admitted and owned by the execution participant.
 */
#pragma once
#include "../common/AcceptedRequestPrefixes.h"
#include <cstddef>
#include <cstdint>
namespace llaminar2::qwen4exp {
/** @brief Snapshot exactly one request, publishing its transaction live word. */
struct PLESnapshotPrior {
    DeviceRequestRowRanges sourceRows; int requestIndex,channels;
    const std::int32_t *acceptedTokens; const float *acceptedConvolution;
    std::int32_t *priorTokens; float *priorConvolution;
    std::int32_t *transactionLive; ///< Complete request-vector; only requestIndex is written.
    std::uint64_t *copiedBytes; ///< Required exact payload observation; metadata excluded.
};
/** @brief Retain only this request's live token/normalized candidates. */
struct PLERetainCandidates {
    DeviceRequestRowRanges transactionRows; int requestIndex,channels;
    const std::int32_t *tokens; const float *normalized;
    std::int32_t *retainedTokens; float *retainedNormalized;
    std::uint64_t *copiedBytes;
};
/** @brief Splice one accepted request from the immutable transaction snapshot. */
struct PLEPublishAccepted {
    AcceptedRequestPrefixes acceptance; int requestIndex,channels;
    const std::int32_t *priorTokens; const float *priorConvolution;
    const std::int32_t *retainedTokens; const float *retainedNormalized;
    std::int32_t *acceptedTokens; float *acceptedConvolution;
};
/** @brief Reject missing, overflowing, misaligned or overlapping snapshot storage. */
void validatePLESnapshot(const PLESnapshotPrior &p);
/** @brief Reject candidate extent/alias mistakes without reading live device words. */
void validatePLERetention(const PLERetainCandidates &p);
/** @brief Reject publication aliases/geometry before CPU execution or native launch. */
void validatePLEPublication(const PLEPublishAccepted &p);
/** @brief Validate all requests' native decisions without mutation; CPU owners only. */
bool validPLEDecisions(const AcceptedRequestPrefixes &p) noexcept;
/** @brief Snapshot original tails, or only publish zero metadata for an empty request. */
void cpuPLESnapshotPrior(const PLESnapshotPrior &p);
/** @brief Copy exactly live candidates; dirty capacity is neither read nor written. */
void cpuPLERetainCandidates(const PLERetainCandidates &p);
/** @brief Validate the entire decision vector before restoring/splicing one request. */
void cpuPLEPublishAccepted(const PLEPublishAccepted &p);
/** @brief Submit snapshot on an exact non-default CUDA stream. */
void cudaPLESnapshotPrior(const PLESnapshotPrior &p,void *stream);
/** @brief Submit exact live retention on an exact non-default CUDA stream. */
void cudaPLERetainCandidates(const PLERetainCandidates &p,void *stream);
/** @brief Validate all native decisions, then submit exact CUDA tail splice. */
void cudaPLEPublishAccepted(const PLEPublishAccepted &p,void *stream);
/** @brief Submit snapshot on an exact non-default HIP stream. */
void rocmPLESnapshotPrior(const PLESnapshotPrior &p,void *stream);
/** @brief Submit exact live retention on an exact non-default HIP stream. */
void rocmPLERetainCandidates(const PLERetainCandidates &p,void *stream);
/** @brief Validate all native decisions, then submit exact HIP tail splice. */
void rocmPLEPublishAccepted(const PLEPublishAccepted &p,void *stream);
} // namespace llaminar2::qwen4exp
