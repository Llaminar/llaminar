#pragma once

#include "../mtp/MTPDecodeCatchup.h"
#include "../mtp/MTPSpecDecodeMetadata.h"
#include "../mtp/MTPSpecRequestBatchOwner.h"
#include "../mtp/MTPSpecRequestBatchScheduler.h"
#include "../mtp/MTPSpecTransactionDriver.h"

#include <functional>
#include <string>
#include <vector>

namespace llaminar2
{
    class IInferenceRunner;

    /**
     * @brief Optional execution knobs for a target-verifier graph forward.
     *
     * The device-token pointer represents either a runner-owned compact token
     * row or a padded request-batch token matrix that GPU graph capture can
     * read without a host-to-device copy. Batched callers must provide rows
     * laid out as `[request_count, padded_seq_len]`, matching the graph
     * forward plan.
     */
    struct MTPVerifierForwardExecutionOptions
    {
        const void *device_token_ids = nullptr;
        bool allow_batched_host_forward = true;
    };

    /**
     * @brief Result of executing one verifier input plan through a runner.
     */
    struct MTPVerifierForwardExecutionResult
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeVerifierGraphForwardPlan graph_plan;
        bool used_batch_forward = false;
        bool used_device_token_ids = false;
    };

    /**
     * @brief Execute a logical MTP verifier input plan in graph coordinates.
     *
     * `MTPSpecDecodeVerifierInputPlan` is compact and logical: requests are
     * flattened back-to-back. Device graphs see padded batch coordinates, so
     * this helper materializes `MTPSpecDecodeVerifierGraphForwardPlan` first
     * and then chooses the matching runner entrypoint:
     *
     * - one request + host tokens -> `forward()`
     * - one request + device token row -> `forwardWithDeviceTokenIds()`
     * - multiple requests + host tokens -> `forward_batch()`
     * - multiple requests + device token rows -> `forwardBatchWithDeviceTokenIds()`
     *
     * Keeping this as a single utility prevents future scheduler batching from
     * quietly diverging from the SingleDevice verifier path.
     */
    MTPVerifierForwardExecutionResult executeMTPSpecVerifierForward(
        IInferenceRunner &runner,
        const MTPSpecDecodeVerifierInputPlan &plan,
        const MTPVerifierForwardExecutionOptions &options = {});

    /**
     * @brief Input for one greedy request-batched verifier transaction.
     *
     * Every request supplies the compact all-position verifier input
     * `[main_token, draft_0, ...]` through `requests`.  The executor runs one
     * verifier graph, greedily samples the compact verifier rows, and converts
     * the result into one publication-ready transaction plan.
     */
    struct MTPGreedyVerifierBatchTransactionRequest
    {
        MTPSpecDecodeMetadataShape shape;
        std::vector<int> request_ids;
        int vocab_size = 0;
        std::vector<MTPDecodeCatchupGreedyRequest> requests;
        std::vector<int32_t> base_cached_tokens;
        MTPVerifierForwardExecutionOptions forward_options = {};
        MTPSpecTransactionPublicationContract publication_contract =
            MTPSpecTransactionPublicationContract::
                DirectAcceptedStatePublication;
    };

    /**
     * @brief Result of an executable greedy verifier batch transaction.
     */
    struct MTPGreedyVerifierBatchTransactionResult
    {
        bool ok = false;
        std::string error;

        MTPSpecDecodeVerifierInputPlan verifier_input_plan;
        MTPVerifierForwardExecutionResult forward;
        std::vector<int32_t> sampled_verifier_rows;
        MTPDecodeCatchupGreedyBatchResult catchup;
        MTPSpecTransactionBatchPlan transaction_plan;
    };

    /**
     * @brief Execute a compact greedy verifier batch and build publication plans.
     *
     * The helper owns the short-lived all-position verifier modes: it enables
     * row-indexed all-position logits, installs the compact row plan, executes
     * the verifier forward, samples rows `0..N-1`, and then disables the modes
     * even on failure.  It does not mutate live state publication itself;
     * callers apply the returned `transaction_plan` atomically.
     */
    MTPGreedyVerifierBatchTransactionResult executeMTPGreedyVerifierBatchTransaction(
        IInferenceRunner &runner,
        const MTPGreedyVerifierBatchTransactionRequest &request);

    /**
     * @brief Execute a scheduler-produced greedy verifier transaction batch.
     *
     * This is the narrow handoff between request admission and graph execution.
     * Keeping the adapter here avoids teaching the scheduler about runner-layer
     * entrypoints while still proving that scheduled batches are immediately
     * consumable by the existing vLLM-style transaction helper.
     */
    MTPGreedyVerifierBatchTransactionResult executeMTPGreedyVerifierScheduledBatchTransaction(
        IInferenceRunner &runner,
        const MTPSpecRequestBatch &scheduled_batch,
        MTPVerifierForwardExecutionOptions forward_options = {},
        MTPSpecTransactionPublicationContract publication_contract =
            MTPSpecTransactionPublicationContract::
                DirectAcceptedStatePublication);

    /**
     * @brief Result of scheduling and executing an owned greedy verifier batch.
     */
    struct MTPOwnedGreedyVerifierBatchTransactionResult
    {
        bool ok = false;
        std::string error;

        MTPSpecRequestBatch scheduled_batch;
        MTPGreedyVerifierBatchTransactionResult transaction;
        bool published = false;
        bool committed = false;
        bool released = false;
    };

    /**
     * @brief Schedule, execute, and complete one owned greedy verifier batch.
     *
     * This is the first executable handoff above raw scheduler admission. It
     * reserves a compatible batch through `MTPSpecRequestBatchOwner`, executes
     * the existing verifier transaction helper, commits admitted request ids on
     * success, and releases the reservation unchanged on failure. The helper
     * deliberately does not perform live state publication; callers still apply
     * the returned transaction plan atomically before considering the request
     * complete.
     */
    MTPOwnedGreedyVerifierBatchTransactionResult
    executeOwnedMTPGreedyVerifierScheduledBatchTransaction(
        IInferenceRunner &runner,
        MTPSpecRequestBatchOwner &owner,
        const MTPSpecRequestBatchScheduler &scheduler,
        MTPVerifierForwardExecutionOptions forward_options = {},
        MTPSpecTransactionPublicationContract publication_contract =
            MTPSpecTransactionPublicationContract::
                DirectAcceptedStatePublication);

    /**
     * @brief Callback used to publish accepted verifier state before commit.
     *
     * The publisher receives the fully validated transaction plan. It should
     * apply KV, terminal hidden/logits, GDN, short-conv, and shifted MTP state
     * through the normal runner/backend publication path. Returning false keeps
     * ownership pending so the caller can retry or replay safely.
     */
    using MTPGreedyVerifierBatchPublicationFn =
        std::function<bool(const MTPSpecTransactionBatchPlan &, std::string *)>;

    /**
     * @brief Schedule, verify, publish, and commit one owned greedy batch.
     *
     * This is the ownership contract intended for benchmark/server request
     * batching. Requests are reserved while the verifier graph runs. The
     * reservation commits only after `publish` succeeds, and it is released
     * unchanged after verifier, sampling, transaction-planning, or publication
     * failure.
     */
    MTPOwnedGreedyVerifierBatchTransactionResult
    executeOwnedMTPGreedyVerifierScheduledBatchTransactionAndPublish(
        IInferenceRunner &runner,
        MTPSpecRequestBatchOwner &owner,
        const MTPSpecRequestBatchScheduler &scheduler,
        MTPGreedyVerifierBatchPublicationFn publish,
        MTPVerifierForwardExecutionOptions forward_options = {},
        MTPSpecTransactionPublicationContract publication_contract =
            MTPSpecTransactionPublicationContract::
                DirectAcceptedStatePublication);

    /**
     * @brief Result of reducing a scheduled stochastic verifier batch.
     *
     * GPU stochastic verifier kernels should return compact device outcomes,
     * not host-visible logits. This object records the publication-ready
     * transaction plan produced from those compact outcomes.
     */
    struct MTPDeviceOutcomeBatchTransactionResult
    {
        bool ok = false;
        std::string error;

        MTPSpecRequestBatch scheduled_batch;
        std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes;
        MTPSpecTransactionBatchPlan transaction_plan;
    };

    /**
     * @brief Build a transaction plan from a scheduled stochastic outcome batch.
     */
    MTPDeviceOutcomeBatchTransactionResult executeMTPDeviceOutcomeScheduledBatchTransaction(
        const MTPSpecRequestBatch &scheduled_batch,
        std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes,
        MTPSpecTransactionPublicationContract publication_contract =
            MTPSpecTransactionPublicationContract::
                DirectAcceptedStatePublication);

    /**
     * @brief Callback that produces compact stochastic verifier outcomes.
     *
     * The producer is called only after the owner has reserved a concrete
     * scheduled batch, letting backend code execute a graph for exactly those
     * requests and write one compact outcome per admitted request.
     */
    using MTPDeviceOutcomeBatchProducerFn =
        std::function<bool(const MTPSpecRequestBatch &,
                           std::vector<MTPDeviceRejectionBatchOutcome> *,
                           std::string *)>;

    /**
     * @brief Result of owned stochastic outcome publication.
     */
    struct MTPOwnedDeviceOutcomeBatchTransactionResult
    {
        bool ok = false;
        std::string error;

        MTPSpecRequestBatch scheduled_batch;
        std::vector<MTPDeviceRejectionBatchOutcome> device_outcomes;
        MTPSpecTransactionBatchPlan transaction_plan;
        bool produced = false;
        bool published = false;
        bool committed = false;
        bool released = false;
    };

    /**
     * @brief Schedule, produce, publish, and commit one stochastic batch.
     *
     * This mirrors the greedy publication-aware helper but leaves verifier
     * execution to a producer callback. That matches the vLLM-style GPU path:
     * graph-captured device work reduces stochastic rows into compact outcomes,
     * then shared host code validates the outcomes before live-state
     * publication and request ownership commit.
     */
    MTPOwnedDeviceOutcomeBatchTransactionResult
    executeOwnedMTPDeviceOutcomeScheduledBatchTransactionAndPublish(
        MTPSpecRequestBatchOwner &owner,
        const MTPSpecRequestBatchScheduler &scheduler,
        MTPDeviceOutcomeBatchProducerFn produce,
        MTPGreedyVerifierBatchPublicationFn publish,
        MTPSpecTransactionPublicationContract publication_contract =
            MTPSpecTransactionPublicationContract::
                DirectAcceptedStatePublication);

} // namespace llaminar2
