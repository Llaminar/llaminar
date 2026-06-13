#include "MTPSpecRequestBatchScheduler.h"

#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPSpecRequestBatch failure(
            const MTPSpecRequestBatchSchedulerConfig &config,
            std::string error,
            std::vector<MTPSpecRequestBatchAdmission> admissions = {})
        {
            MTPSpecRequestBatch batch;
            batch.ok = false;
            batch.error = std::move(error);
            batch.mode = config.mode;
            batch.shape.max_requests = config.max_request_batch;
            batch.shape.max_draft_tokens = config.max_draft_tokens;
            batch.admissions = std::move(admissions);
            return batch;
        }

        MTPSpecRequestBatchAdmission admission(
            const MTPSpecSchedulableRequest &request,
            MTPSpecRequestBatchAdmissionStatus status,
            std::string reason)
        {
            MTPSpecRequestBatchAdmission result;
            result.request_id = request.request_id;
            result.status = status;
            result.reason = std::move(reason);
            return result;
        }

        int verifierTokenCount(const MTPSpecSchedulableRequest &request)
        {
            return static_cast<int>(request.greedy_request.draft_tokens.size());
        }

        bool compatibleWithSeed(
            const MTPSpecSchedulableRequest &request,
            const std::string &seed_key,
            int seed_vocab)
        {
            return request.compatibility_key == seed_key &&
                   request.vocab_size == seed_vocab;
        }

        const char *modeName(MTPSpecRequestBatchMode mode)
        {
            switch (mode)
            {
            case MTPSpecRequestBatchMode::GREEDY:
                return "greedy";
            case MTPSpecRequestBatchMode::STOCHASTIC:
                return "stochastic";
            }
            return "unknown";
        }
    } // namespace

    MTPSpecRequestBatchScheduler::MTPSpecRequestBatchScheduler(
        MTPSpecRequestBatchSchedulerConfig config)
        : config_(config)
    {
    }

    MTPSpecRequestBatch MTPSpecRequestBatchScheduler::buildNextBatch(
        const std::vector<MTPSpecSchedulableRequest> &pending) const
    {
        std::vector<MTPSpecRequestBatchAdmission> admissions;
        admissions.reserve(pending.size());

        if (config_.max_request_batch <= 0)
            return failure(config_, "MTP request batch scheduler has non-positive max_request_batch");
        if (config_.max_draft_tokens <= 0)
            return failure(config_, "MTP request batch scheduler has non-positive max_draft_tokens");
        if (pending.empty())
            return failure(config_, "MTP request batch scheduler has no pending requests");

        MTPSpecRequestBatch batch;
        batch.ok = true;
        batch.mode = config_.mode;
        batch.shape.max_requests = config_.max_request_batch;
        batch.shape.max_draft_tokens = config_.max_draft_tokens;

        bool seeded = false;
        for (const MTPSpecSchedulableRequest &request : pending)
        {
            if (!request.ready)
            {
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::DEFERRED,
                              "request is not ready for speculative verification"));
                continue;
            }

            if (request.request_id < 0)
            {
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::REJECTED,
                              "request id is negative"));
                continue;
            }

            if (request.mode != config_.mode)
            {
                std::ostringstream msg;
                msg << "request uses " << modeName(request.mode)
                    << " MTP but scheduler is building "
                    << modeName(config_.mode) << " batches";
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::DEFERRED,
                              msg.str()));
                continue;
            }

            if (request.verifier_input !=
                MTPSpecVerifierInputPlacement::HOST_TOKENS)
            {
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::REJECTED,
                              "request-batched verifier execution requires host verifier tokens until device-token workspace support lands"));
                continue;
            }

            const int token_count = verifierTokenCount(request);
            if (token_count <= 0)
            {
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::REJECTED,
                              "request has no verifier tokens"));
                continue;
            }
            if (token_count > config_.max_draft_tokens)
            {
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::REJECTED,
                              "request verifier token count exceeds max_draft_tokens"));
                continue;
            }

            if (!seeded)
            {
                batch.compatibility_key = request.compatibility_key;
                batch.vocab_size = request.vocab_size;
                seeded = true;
            }
            else if (!compatibleWithSeed(
                         request,
                         batch.compatibility_key,
                         batch.vocab_size))
            {
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::DEFERRED,
                              "request compatibility key or vocabulary differs from the current batch seed"));
                continue;
            }

            if (batch.request_count >= config_.max_request_batch)
            {
                admissions.push_back(
                    admission(request,
                              MTPSpecRequestBatchAdmissionStatus::DEFERRED,
                              "request batch is already at max_request_batch"));
                continue;
            }

            batch.request_ids.push_back(request.request_id);
            batch.greedy_requests.push_back(request.greedy_request);
            batch.base_cached_tokens.push_back(request.base_cached_tokens);
            batch.requires_shifted_kv_publication =
                batch.requires_shifted_kv_publication ||
                request.requires_shifted_kv_publication;
            ++batch.request_count;

            admissions.push_back(
                admission(request,
                          MTPSpecRequestBatchAdmissionStatus::ADMITTED,
                          "admitted"));
        }

        batch.admissions = std::move(admissions);
        if (batch.request_count <= 0)
        {
            return failure(
                config_,
                "MTP request batch scheduler found no admissible requests",
                std::move(batch.admissions));
        }

        return batch;
    }

} // namespace llaminar2
