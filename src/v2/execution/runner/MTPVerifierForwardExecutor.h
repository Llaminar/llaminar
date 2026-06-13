#pragma once

#include "../mtp/MTPDecodeCatchup.h"
#include "../mtp/MTPSpecDecodeMetadata.h"
#include "../mtp/MTPSpecTransactionDriver.h"

#include <string>
#include <vector>

namespace llaminar2
{
    class IInferenceRunner;

    /**
     * @brief Optional execution knobs for a target-verifier graph forward.
     *
     * The device-token pointer is intentionally single-request only today. It
     * represents a runner-owned compact token row that GPU graph capture can
     * read without a host-to-device copy. Request-batched device-token input
     * needs a separate workspace contract, so this helper hard-fails that case.
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
     * - multiple requests -> `forward_batch()`
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

} // namespace llaminar2
