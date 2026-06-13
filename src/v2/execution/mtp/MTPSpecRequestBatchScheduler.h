/**
 * @file MTPSpecRequestBatchScheduler.h
 * @brief Admission control for vLLM-style request-batched MTP transactions.
 */
#pragma once

#include "MTPDecodeCatchup.h"
#include "MTPSpecDecodeMetadata.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Sampling/verifier contract used by an admitted speculative batch.
     *
     * The scheduler intentionally separates greedy and stochastic modes because
     * they have different verifier-output plumbing.  Mixing them in one graph
     * transaction would make state publication hard to reason about.
     */
    enum class MTPSpecRequestBatchMode
    {
        GREEDY,
        STOCHASTIC,
    };

    /**
     * @brief Where a request's compact verifier tokens currently live.
     *
     * Multi-request verifier execution only has a host-token contract today.
     * Device-token rows need a declared workspace binding before they can enter
     * request-batched graph capture.
     */
    enum class MTPSpecVerifierInputPlacement
    {
        HOST_TOKENS,
        DEVICE_TOKEN_ROW,
    };

    /**
     * @brief Per-request admission result for diagnostics and tests.
     */
    enum class MTPSpecRequestBatchAdmissionStatus
    {
        ADMITTED,
        DEFERRED,
        REJECTED,
    };

    /**
     * @brief Static scheduler limits for one speculative transaction attempt.
     */
    struct MTPSpecRequestBatchSchedulerConfig
    {
        int max_request_batch = 1;
        int max_draft_tokens = 0;
        MTPSpecRequestBatchMode mode = MTPSpecRequestBatchMode::GREEDY;
    };

    /**
     * @brief One request that may be batched into the next verifier transaction.
     *
     * `greedy_request.draft_tokens` uses the existing MTP catch-up naming: it is
     * the compact verifier token row consumed by the target graph.  The current
     * metadata code caps this vector with `shape.max_draft_tokens`.
     */
    struct MTPSpecSchedulableRequest
    {
        int request_id = -1;
        bool ready = true;
        MTPSpecRequestBatchMode mode = MTPSpecRequestBatchMode::GREEDY;
        MTPSpecVerifierInputPlacement verifier_input =
            MTPSpecVerifierInputPlacement::HOST_TOKENS;
        std::string compatibility_key;
        int vocab_size = 0;
        int32_t base_cached_tokens = 0;
        bool requires_shifted_kv_publication = true;
        MTPDecodeCatchupGreedyRequest greedy_request;
    };

    /**
     * @brief Recorded reason for admitting, deferring, or rejecting a request.
     */
    struct MTPSpecRequestBatchAdmission
    {
        int request_id = -1;
        MTPSpecRequestBatchAdmissionStatus status =
            MTPSpecRequestBatchAdmissionStatus::DEFERRED;
        std::string reason;
    };

    /**
     * @brief Scheduler output that can be adapted to verifier transaction calls.
     */
    struct MTPSpecRequestBatch
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeMetadataShape shape;
        int request_count = 0;
        MTPSpecRequestBatchMode mode = MTPSpecRequestBatchMode::GREEDY;
        std::string compatibility_key;
        int vocab_size = 0;
        bool requires_shifted_kv_publication = false;

        std::vector<int> request_ids;
        std::vector<MTPDecodeCatchupGreedyRequest> greedy_requests;
        std::vector<int32_t> base_cached_tokens;
        std::vector<MTPSpecRequestBatchAdmission> admissions;
    };

    /**
     * @brief Owns CPU-side request grouping before graph verifier execution.
     *
     * This class does not execute model code.  It makes the admission decision
     * explicit and testable so server or benchmark code cannot accidentally
     * claim request batching while still running one request at a time.
     */
    class MTPSpecRequestBatchScheduler
    {
    public:
        explicit MTPSpecRequestBatchScheduler(
            MTPSpecRequestBatchSchedulerConfig config);

        /**
         * @brief Build the next compatible request batch from pending requests.
         *
         * Requests are considered in stable input order.  The first admitted
         * request establishes the compatibility key and vocabulary.  Later
         * incompatible requests are deferred instead of rejected so they can be
         * tried in a later batch with matching peers.
         */
        MTPSpecRequestBatch buildNextBatch(
            const std::vector<MTPSpecSchedulableRequest> &pending) const;

        const MTPSpecRequestBatchSchedulerConfig &config() const
        {
            return config_;
        }

    private:
        MTPSpecRequestBatchSchedulerConfig config_;
    };

} // namespace llaminar2
