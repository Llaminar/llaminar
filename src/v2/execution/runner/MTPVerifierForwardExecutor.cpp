#include "MTPVerifierForwardExecutor.h"

#include "../local_execution/orchestrators/IInferenceRunner.h"

#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPVerifierForwardExecutionResult verifierForwardFailure(
            MTPSpecDecodeVerifierGraphForwardPlan graph_plan,
            std::string error)
        {
            MTPVerifierForwardExecutionResult result;
            result.ok = false;
            result.graph_plan = std::move(graph_plan);
            result.error = std::move(error);
            return result;
        }
    } // namespace

    MTPVerifierForwardExecutionResult executeMTPSpecVerifierForward(
        IInferenceRunner &runner,
        const MTPSpecDecodeVerifierInputPlan &plan,
        const MTPVerifierForwardExecutionOptions &options)
    {
        MTPSpecDecodeVerifierGraphForwardPlan graph_plan =
            buildMTPSpecDecodeVerifierGraphForwardPlan(plan);
        if (!graph_plan.ok)
        {
            return verifierForwardFailure(
                std::move(graph_plan),
                std::string("MTP verifier graph forward plan failed: ") +
                    graph_plan.error);
        }

        const bool batched = graph_plan.request_count > 1;
        if (batched && !options.allow_batched_host_forward)
        {
            return verifierForwardFailure(
                std::move(graph_plan),
                "Batched MTP verifier forward is disabled by caller options");
        }

        bool forward_ok = false;
        MTPVerifierForwardExecutionResult result;
        result.graph_plan = std::move(graph_plan);
        result.used_batch_forward = batched;
        result.used_device_token_ids = options.device_token_ids != nullptr;

        if (batched)
        {
            forward_ok = options.device_token_ids
                             ? runner.forwardBatchWithDeviceTokenIds(
                                   result.graph_plan.token_batches,
                                   options.device_token_ids,
                                   result.graph_plan.padded_seq_len)
                             : runner.forward_batch(result.graph_plan.token_batches);
        }
        else
        {
            const int seq_len = result.graph_plan.sequence_lengths.empty()
                                    ? plan.total_verifier_input_tokens
                                    : result.graph_plan.sequence_lengths.front();
            /*
             * The host shadow vector is still supplied when the graph reads a
             * device token row. It gives diagnostics and CPU-side state code a
             * stable logical view while the embedding stage consumes the
             * runner-owned device buffer.
             */
            forward_ok = options.device_token_ids
                             ? runner.forwardWithDeviceTokenIds(
                                   plan.verifier_input_tokens.data(),
                                   options.device_token_ids,
                                   seq_len)
                             : runner.forward(
                                   plan.verifier_input_tokens.data(),
                                   seq_len);
        }

        if (!forward_ok)
        {
            result.ok = false;
            result.error = batched
                               ? "MTP verifier batched forward failed"
                               : "MTP verifier forward failed";
            return result;
        }

        result.ok = true;
        return result;
    }

    MTPGreedyVerifierBatchTransactionResult executeMTPGreedyVerifierBatchTransaction(
        IInferenceRunner &runner,
        const MTPGreedyVerifierBatchTransactionRequest &request)
    {
        MTPGreedyVerifierBatchTransactionResult result;

        auto fail = [&](std::string error) -> MTPGreedyVerifierBatchTransactionResult
        {
            result.ok = false;
            result.error = std::move(error);
            return result;
        };

        if (!request.shape.valid())
            return fail("MTP greedy verifier batch transaction has invalid shape");
        if (request.requests.empty())
            return fail("MTP greedy verifier batch transaction has no requests");
        if (request.request_ids.size() != request.requests.size())
            return fail("MTP greedy verifier batch transaction request-id vector mismatch");
        if (request.base_cached_tokens.size() != request.requests.size())
            return fail("MTP greedy verifier batch transaction base-cache vector mismatch");
        if (static_cast<int>(request.requests.size()) > request.shape.max_requests)
            return fail("MTP greedy verifier batch transaction exceeds max_requests");

        std::vector<MTPSpecDecodeVerifierDraftRequest> verifier_requests;
        verifier_requests.reserve(request.requests.size());
        for (size_t i = 0; i < request.requests.size(); ++i)
        {
            MTPSpecDecodeVerifierDraftRequest verifier_request;
            verifier_request.request_id = request.request_ids[i];
            verifier_request.draft_tokens = request.requests[i].draft_tokens;
            verifier_requests.push_back(std::move(verifier_request));
        }

        result.verifier_input_plan =
            buildMTPSpecDecodeVerifierInputPlan(
                request.shape,
                verifier_requests);
        if (!result.verifier_input_plan.ok)
        {
            return fail(std::string("MTP greedy verifier input plan failed: ") +
                        result.verifier_input_plan.error);
        }

        const int compact_row_count =
            result.verifier_input_plan.compact_logit_row_count;
        if (compact_row_count <= 0)
            return fail("MTP greedy verifier input plan produced no compact rows");

        bool row_indexed_enabled = false;
        bool all_position_enabled = false;
        auto cleanup_row_indexed = [&]() -> bool
        {
            bool ok = true;
            runner.clearMTPSpecVerifierInputPlan();
            if (all_position_enabled)
            {
                all_position_enabled = false;
                ok = runner.setComputeAllPositionLogits(false) && ok;
            }
            if (!row_indexed_enabled)
                return ok;
            row_indexed_enabled = false;
            return runner.setComputeRowIndexedAllPositionLogits(false, 0) && ok;
        };

        if (!runner.setComputeRowIndexedAllPositionLogits(true, compact_row_count))
            return fail("MTP greedy verifier batch could not enable row-indexed logits");
        row_indexed_enabled = true;

        if (!runner.setMTPSpecVerifierInputPlan(result.verifier_input_plan))
        {
            const bool cleanup_ok = cleanup_row_indexed();
            return fail(
                cleanup_ok
                    ? "MTP greedy verifier batch could not install row plan"
                    : "MTP greedy verifier batch could not install row plan and cleanup failed");
        }

        if (!runner.setComputeAllPositionLogits(true))
        {
            const bool cleanup_ok = cleanup_row_indexed();
            return fail(
                cleanup_ok
                    ? "MTP greedy verifier batch could not enable all-position logits"
                    : "MTP greedy verifier batch could not enable all-position logits and cleanup failed");
        }
        all_position_enabled = true;

        result.forward =
            executeMTPSpecVerifierForward(
                runner,
                result.verifier_input_plan,
                request.forward_options);
        if (!result.forward.ok)
        {
            const bool cleanup_ok = cleanup_row_indexed();
            return fail(
                cleanup_ok
                    ? std::string("MTP greedy verifier forward failed: ") +
                          result.forward.error
                    : std::string("MTP greedy verifier forward failed and cleanup failed: ") +
                          result.forward.error);
        }

        result.sampled_verifier_rows.assign(
            static_cast<size_t>(compact_row_count),
            kMTPSpecDecodeInvalidToken);
        if (!runner.sampleGreedyFromAllPositionLogitsOnDeviceRows(
                /*start_row=*/0,
                compact_row_count,
                result.sampled_verifier_rows.data()))
        {
            const bool cleanup_ok = cleanup_row_indexed();
            return fail(
                cleanup_ok
                    ? "MTP greedy verifier batch could not sample compact rows"
                    : "MTP greedy verifier batch could not sample compact rows and cleanup failed");
        }

        if (!cleanup_row_indexed())
            return fail("MTP greedy verifier batch could not disable row-indexed logits");

        result.catchup =
            buildAllPositionMTPDecodeCatchupGreedyBatchResult(
                request.requests,
                result.sampled_verifier_rows);
        if (!result.catchup.ok)
        {
            return fail(std::string("MTP greedy verifier catch-up batch failed: ") +
                        result.catchup.error);
        }

        result.transaction_plan =
            request.publication_contract ==
                    MTPSpecTransactionPublicationContract::
                        DecodeEquivalentReplayPublicationRequired
                ? buildMTPSpecTransactionBatchPlanFromGreedyCatchupsForReplayPublication(
                      request.shape,
                      request.request_ids,
                      request.vocab_size,
                      request.requests,
                      result.catchup.results,
                      request.base_cached_tokens)
                : buildMTPSpecTransactionBatchPlanFromGreedyCatchups(
                      request.shape,
                      request.request_ids,
                      request.vocab_size,
                      request.requests,
                      result.catchup.results,
                      request.base_cached_tokens);
        if (!result.transaction_plan.ok)
        {
            return fail(std::string("MTP greedy verifier transaction plan failed: ") +
                        result.transaction_plan.error);
        }

        result.ok = true;
        return result;
    }

    MTPGreedyVerifierBatchTransactionResult executeMTPGreedyVerifierScheduledBatchTransaction(
        IInferenceRunner &runner,
        const MTPSpecRequestBatch &scheduled_batch,
        MTPVerifierForwardExecutionOptions forward_options,
        MTPSpecTransactionPublicationContract publication_contract)
    {
        MTPGreedyVerifierBatchTransactionResult result;

        auto fail = [&](std::string error) -> MTPGreedyVerifierBatchTransactionResult
        {
            result.ok = false;
            result.error = std::move(error);
            return result;
        };

        if (!scheduled_batch.ok)
        {
            return fail(
                std::string("cannot execute invalid scheduled MTP batch: ") +
                scheduled_batch.error);
        }
        if (scheduled_batch.mode != MTPSpecRequestBatchMode::GREEDY)
            return fail("scheduled MTP batch is not greedy");
        if (scheduled_batch.request_count <= 0)
            return fail("scheduled MTP batch has no admitted requests");
        if (scheduled_batch.verifier_input ==
                MTPSpecVerifierInputPlacement::DEVICE_TOKEN_ROW &&
            forward_options.device_token_ids == nullptr)
        {
            return fail("scheduled MTP device-token batch requires device_token_ids");
        }
        if (scheduled_batch.verifier_input ==
                MTPSpecVerifierInputPlacement::HOST_TOKENS &&
            forward_options.device_token_ids != nullptr)
        {
            return fail("scheduled MTP host-token batch received device_token_ids");
        }

        const size_t request_count =
            static_cast<size_t>(scheduled_batch.request_count);
        if (scheduled_batch.request_ids.size() != request_count ||
            scheduled_batch.greedy_requests.size() != request_count ||
            scheduled_batch.base_cached_tokens.size() != request_count)
        {
            return fail("scheduled MTP batch vectors do not match request_count");
        }

        MTPGreedyVerifierBatchTransactionRequest request;
        request.shape = scheduled_batch.shape;
        request.request_ids = scheduled_batch.request_ids;
        request.vocab_size = scheduled_batch.vocab_size;
        request.requests = scheduled_batch.greedy_requests;
        request.base_cached_tokens = scheduled_batch.base_cached_tokens;
        request.forward_options = forward_options;
        request.publication_contract = publication_contract;
        return executeMTPGreedyVerifierBatchTransaction(runner, request);
    }

    MTPOwnedGreedyVerifierBatchTransactionResult
    executeOwnedMTPGreedyVerifierScheduledBatchTransaction(
        IInferenceRunner &runner,
        MTPSpecRequestBatchOwner &owner,
        const MTPSpecRequestBatchScheduler &scheduler,
        MTPVerifierForwardExecutionOptions forward_options,
        MTPSpecTransactionPublicationContract publication_contract)
    {
        MTPOwnedGreedyVerifierBatchTransactionResult result;

        result.scheduled_batch = owner.scheduleNextBatch(scheduler);
        if (!result.scheduled_batch.ok)
        {
            result.ok = false;
            result.error =
                std::string("owned MTP verifier scheduling failed: ") +
                result.scheduled_batch.error;
            return result;
        }

        result.transaction =
            executeMTPGreedyVerifierScheduledBatchTransaction(
                runner,
                result.scheduled_batch,
                forward_options,
                publication_contract);

        if (!result.transaction.ok)
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error =
                std::string("owned MTP verifier transaction failed: ") +
                result.transaction.error;
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        std::string commit_error;
        result.committed = owner.commitInFlightBatch(&commit_error);
        if (!result.committed)
        {
            result.ok = false;
            result.error =
                std::string("owned MTP verifier transaction succeeded but commit failed: ") +
                commit_error;
            return result;
        }

        result.ok = true;
        return result;
    }

    MTPOwnedGreedyVerifierBatchTransactionResult
    executeOwnedMTPGreedyVerifierScheduledBatchTransactionAndPublish(
        IInferenceRunner &runner,
        MTPSpecRequestBatchOwner &owner,
        const MTPSpecRequestBatchScheduler &scheduler,
        MTPGreedyVerifierBatchPublicationFn publish,
        MTPVerifierForwardExecutionOptions forward_options,
        MTPSpecTransactionPublicationContract publication_contract)
    {
        MTPOwnedGreedyVerifierBatchTransactionResult result;

        if (!publish)
        {
            result.ok = false;
            result.error =
                "owned MTP verifier publication callback is required";
            return result;
        }

        result.scheduled_batch = owner.scheduleNextBatch(scheduler);
        if (!result.scheduled_batch.ok)
        {
            result.ok = false;
            result.error =
                std::string("owned MTP verifier scheduling failed: ") +
                result.scheduled_batch.error;
            return result;
        }

        result.transaction =
            executeMTPGreedyVerifierScheduledBatchTransaction(
                runner,
                result.scheduled_batch,
                forward_options,
                publication_contract);

        if (!result.transaction.ok)
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error =
                std::string("owned MTP verifier transaction failed: ") +
                result.transaction.error;
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        if (result.transaction.transaction_plan
                .requiresDecodeEquivalentReplayPublication())
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error =
                "owned MTP greedy transaction requires decode-equivalent replay publication";
            const std::string &reason =
                result.transaction.transaction_plan.publication_contract_reason;
            if (!reason.empty())
            {
                result.error += ": ";
                result.error += reason;
            }
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        std::string publish_error;
        result.published =
            publish(result.transaction.transaction_plan, &publish_error);
        if (!result.published)
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error = "owned MTP verifier publication failed";
            if (!publish_error.empty())
            {
                result.error += ": ";
                result.error += publish_error;
            }
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        std::string commit_error;
        result.committed = owner.commitInFlightBatch(&commit_error);
        if (!result.committed)
        {
            result.ok = false;
            result.error =
                std::string("owned MTP verifier publication succeeded but commit failed: ") +
                commit_error;
            return result;
        }

        result.ok = true;
        return result;
    }

    MTPDeviceOutcomeBatchTransactionResult executeMTPDeviceOutcomeScheduledBatchTransaction(
        const MTPSpecRequestBatch &scheduled_batch,
        std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes,
        MTPSpecTransactionPublicationContract publication_contract)
    {
        MTPDeviceOutcomeBatchTransactionResult result;
        result.scheduled_batch = scheduled_batch;
        result.device_outcomes = std::move(device_outcomes);

        auto fail = [&](std::string error) -> MTPDeviceOutcomeBatchTransactionResult
        {
            result.ok = false;
            result.error = std::move(error);
            return result;
        };

        if (!scheduled_batch.ok)
        {
            return fail(
                std::string("cannot execute invalid scheduled MTP outcome batch: ") +
                scheduled_batch.error);
        }
        if (scheduled_batch.mode != MTPSpecRequestBatchMode::STOCHASTIC)
            return fail("scheduled MTP outcome batch is not stochastic");
        if (scheduled_batch.request_count <= 0)
            return fail("scheduled MTP outcome batch has no admitted requests");

        const size_t request_count =
            static_cast<size_t>(scheduled_batch.request_count);
        if (scheduled_batch.request_ids.size() != request_count ||
            scheduled_batch.greedy_requests.size() != request_count ||
            scheduled_batch.base_cached_tokens.size() != request_count)
        {
            return fail("scheduled MTP outcome batch vectors do not match request_count");
        }
        if (result.device_outcomes.size() != request_count)
            return fail("scheduled MTP outcome batch result vector mismatch");

        result.transaction_plan =
            publication_contract ==
                    MTPSpecTransactionPublicationContract::
                        DecodeEquivalentReplayPublicationRequired
                ? buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomesForReplayPublication(
                      scheduled_batch.shape,
                      scheduled_batch.request_ids,
                      scheduled_batch.vocab_size,
                      scheduled_batch.greedy_requests,
                      result.device_outcomes,
                      scheduled_batch.base_cached_tokens)
                : buildMTPSpecTransactionBatchPlanFromDeviceRejectionOutcomes(
                      scheduled_batch.shape,
                      scheduled_batch.request_ids,
                      scheduled_batch.vocab_size,
                      scheduled_batch.greedy_requests,
                      result.device_outcomes,
                      scheduled_batch.base_cached_tokens);
        if (!result.transaction_plan.ok)
        {
            return fail(
                std::string("scheduled MTP outcome transaction plan failed: ") +
                result.transaction_plan.error);
        }

        result.ok = true;
        return result;
    }

    MTPOwnedDeviceOutcomeBatchTransactionResult
    executeOwnedMTPDeviceOutcomeScheduledBatchTransactionAndPublish(
        MTPSpecRequestBatchOwner &owner,
        const MTPSpecRequestBatchScheduler &scheduler,
        MTPDeviceOutcomeBatchProducerFn produce,
        MTPGreedyVerifierBatchPublicationFn publish,
        MTPSpecTransactionPublicationContract publication_contract)
    {
        MTPOwnedDeviceOutcomeBatchTransactionResult result;

        if (!produce)
        {
            result.ok = false;
            result.error =
                "owned MTP outcome producer callback is required";
            return result;
        }
        if (!publish)
        {
            result.ok = false;
            result.error =
                "owned MTP outcome publication callback is required";
            return result;
        }

        result.scheduled_batch = owner.scheduleNextBatch(scheduler);
        if (!result.scheduled_batch.ok)
        {
            result.ok = false;
            result.error =
                std::string("owned MTP outcome scheduling failed: ") +
                result.scheduled_batch.error;
            return result;
        }

        std::string produce_error;
        result.produced =
            produce(result.scheduled_batch,
                    &result.device_outcomes,
                    &produce_error);
        if (!result.produced)
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error = "owned MTP outcome production failed";
            if (!produce_error.empty())
            {
                result.error += ": ";
                result.error += produce_error;
            }
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        MTPDeviceOutcomeBatchTransactionResult planned =
            executeMTPDeviceOutcomeScheduledBatchTransaction(
                result.scheduled_batch,
                result.device_outcomes,
                publication_contract);
        result.transaction_plan = planned.transaction_plan;
        if (!planned.ok)
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error =
                std::string("owned MTP outcome transaction failed: ") +
                planned.error;
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        /*
         * Grouped outcome proof is not the same as safe live-state
         * publication.  A replay-required plan must be routed to a future
         * replay/publication helper, never through this direct publisher.
         */
        if (result.transaction_plan.requiresDecodeEquivalentReplayPublication())
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error =
                "owned MTP outcome transaction requires decode-equivalent replay publication";
            if (!result.transaction_plan.publication_contract_reason.empty())
            {
                result.error += ": ";
                result.error +=
                    result.transaction_plan.publication_contract_reason;
            }
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        std::string publish_error;
        result.published = publish(result.transaction_plan, &publish_error);
        if (!result.published)
        {
            std::string release_error;
            result.released = owner.releaseInFlightBatch(&release_error);
            result.ok = false;
            result.error = "owned MTP outcome publication failed";
            if (!publish_error.empty())
            {
                result.error += ": ";
                result.error += publish_error;
            }
            if (!result.released)
            {
                result.error += "; release failed: ";
                result.error += release_error;
            }
            return result;
        }

        std::string commit_error;
        result.committed = owner.commitInFlightBatch(&commit_error);
        if (!result.committed)
        {
            result.ok = false;
            result.error =
                std::string("owned MTP outcome publication succeeded but commit failed: ") +
                commit_error;
            return result;
        }

        result.ok = true;
        return result;
    }

} // namespace llaminar2
