/**
 * @file MTPSpecRequestBatchOwner.h
 * @brief Two-phase ownership for request-batched MTP verifier work.
 */
#pragma once

#include "MTPSpecRequestBatchScheduler.h"

#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Owns pending request-batch records until publication commits.
     *
     * `MTPSpecRequestBatchScheduler` answers "which requests are compatible
     * right now?". This owner answers "who is still responsible for those
     * requests?".  Scheduling a batch reserves requests but deliberately does
     * not remove them. The caller removes admitted requests only after verifier
     * execution and accepted-state publication commit successfully. If either
     * step fails, `releaseInFlightBatch()` returns the requests to the pending
     * pool without rebuilding them.
     *
     * The class is intentionally CPU-only. It should be used by runner/server
     * orchestration layers before they hand a scheduled batch to graph-facing
     * metadata and backend kernels.
     */
    class MTPSpecRequestBatchOwner
    {
    public:
        /**
         * @brief Add a request to the pending pool.
         *
         * The request id must be non-negative and unique. Pending mutations are
         * rejected while a batch is in flight so callers cannot race ownership
         * against verifier/publication completion.
         */
        bool enqueueRequest(
            MTPSpecSchedulableRequest request,
            std::string *error = nullptr);

        /**
         * @brief Remove a request that has not been reserved by a batch.
         */
        bool eraseRequest(int request_id, std::string *error = nullptr);

        /**
         * @brief Toggle a request's ready bit before scheduling.
         */
        bool setRequestReady(
            int request_id,
            bool ready,
            std::string *error = nullptr);

        /**
         * @brief Reserve the next compatible batch without removing requests.
         *
         * A successful schedule stores an in-flight batch. A later call to
         * `commitInFlightBatch()` removes its admitted request ids; a call to
         * `releaseInFlightBatch()` leaves the pending pool unchanged.
         */
        MTPSpecRequestBatch scheduleNextBatch(
            const MTPSpecRequestBatchScheduler &scheduler);

        /**
         * @brief Commit the in-flight batch and remove its admitted requests.
         */
        bool commitInFlightBatch(std::string *error = nullptr);

        /**
         * @brief Release the in-flight reservation after a failed transaction.
         */
        bool releaseInFlightBatch(std::string *error = nullptr);

        bool hasInFlightBatch() const
        {
            return in_flight_batch_.has_value();
        }

        const std::optional<MTPSpecRequestBatch> &inFlightBatch() const
        {
            return in_flight_batch_;
        }

        const std::vector<MTPSpecSchedulableRequest> &pendingRequests() const
        {
            return pending_;
        }

        size_t pendingCount() const
        {
            return pending_.size();
        }

        bool empty() const
        {
            return pending_.empty();
        }

    private:
        bool ensureMutable(std::string *error) const;
        bool findPendingIndex(int request_id, size_t *index) const;
        static void setError(std::string *error, std::string message);

        std::vector<MTPSpecSchedulableRequest> pending_;
        std::optional<MTPSpecRequestBatch> in_flight_batch_;
    };

} // namespace llaminar2
