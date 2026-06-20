#include "MTPSpecRequestBatchOwner.h"

#include <algorithm>
#include <unordered_set>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPSpecRequestBatch ownerFailure(std::string error)
        {
            MTPSpecRequestBatch batch;
            batch.ok = false;
            batch.error = std::move(error);
            return batch;
        }
    } // namespace

    bool MTPSpecRequestBatchOwner::enqueueRequest(
        MTPSpecSchedulableRequest request,
        std::string *error)
    {
        if (!ensureMutable(error))
            return false;
        if (request.request_id < 0)
        {
            setError(error, "MTP request batch owner rejects negative request ids");
            return false;
        }

        size_t existing = 0;
        if (findPendingIndex(request.request_id, &existing))
        {
            setError(error, "MTP request batch owner rejects duplicate request ids");
            return false;
        }

        pending_.push_back(std::move(request));
        return true;
    }

    bool MTPSpecRequestBatchOwner::eraseRequest(
        int request_id,
        std::string *error)
    {
        if (!ensureMutable(error))
            return false;

        size_t index = 0;
        if (!findPendingIndex(request_id, &index))
        {
            setError(error, "MTP request batch owner cannot erase an unknown request id");
            return false;
        }

        pending_.erase(pending_.begin() + static_cast<std::ptrdiff_t>(index));
        return true;
    }

    bool MTPSpecRequestBatchOwner::setRequestReady(
        int request_id,
        bool ready,
        std::string *error)
    {
        if (!ensureMutable(error))
            return false;

        size_t index = 0;
        if (!findPendingIndex(request_id, &index))
        {
            setError(error, "MTP request batch owner cannot update an unknown request id");
            return false;
        }

        pending_[index].ready = ready;
        return true;
    }

    MTPSpecRequestBatch MTPSpecRequestBatchOwner::scheduleNextBatch(
        const MTPSpecRequestBatchScheduler &scheduler)
    {
        if (in_flight_batch_.has_value())
        {
            return ownerFailure(
                "MTP request batch owner already has an in-flight batch");
        }

        MTPSpecRequestBatch batch = scheduler.buildNextBatch(pending_);
        if (batch.ok)
            in_flight_batch_ = batch;
        return batch;
    }

    bool MTPSpecRequestBatchOwner::commitInFlightBatch(std::string *error)
    {
        if (!in_flight_batch_.has_value())
        {
            setError(error, "MTP request batch owner has no in-flight batch to commit");
            return false;
        }

        const std::vector<int> &request_ids = in_flight_batch_->request_ids;
        std::unordered_set<int> admitted_ids(
            request_ids.begin(),
            request_ids.end());

        const size_t old_size = pending_.size();
        pending_.erase(
            std::remove_if(
                pending_.begin(),
                pending_.end(),
                [&admitted_ids](const MTPSpecSchedulableRequest &request) {
                    return admitted_ids.find(request.request_id) !=
                           admitted_ids.end();
                }),
            pending_.end());

        const size_t removed = old_size - pending_.size();
        if (removed != admitted_ids.size())
        {
            setError(
                error,
                "MTP request batch owner lost ownership before commit");
            return false;
        }

        in_flight_batch_.reset();
        return true;
    }

    bool MTPSpecRequestBatchOwner::releaseInFlightBatch(std::string *error)
    {
        if (!in_flight_batch_.has_value())
        {
            setError(error, "MTP request batch owner has no in-flight batch to release");
            return false;
        }

        in_flight_batch_.reset();
        return true;
    }

    bool MTPSpecRequestBatchOwner::ensureMutable(std::string *error) const
    {
        if (!in_flight_batch_.has_value())
            return true;

        setError(
            error,
            "MTP request batch owner cannot mutate pending requests while a batch is in flight");
        return false;
    }

    bool MTPSpecRequestBatchOwner::findPendingIndex(
        int request_id,
        size_t *index) const
    {
        for (size_t i = 0; i < pending_.size(); ++i)
        {
            if (pending_[i].request_id == request_id)
            {
                if (index)
                    *index = i;
                return true;
            }
        }
        return false;
    }

    void MTPSpecRequestBatchOwner::setError(
        std::string *error,
        std::string message)
    {
        if (error)
            *error = std::move(message);
    }

} // namespace llaminar2
