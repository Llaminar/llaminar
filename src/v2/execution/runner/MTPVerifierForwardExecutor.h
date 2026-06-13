#pragma once

#include "../mtp/MTPSpecDecodeMetadata.h"

#include <string>

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

} // namespace llaminar2
