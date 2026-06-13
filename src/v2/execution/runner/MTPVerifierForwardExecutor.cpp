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
        if (batched && options.device_token_ids)
        {
            return verifierForwardFailure(
                std::move(graph_plan),
                "Batched MTP verifier forward cannot use the single-row device token input hook");
        }
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
            forward_ok = runner.forward_batch(result.graph_plan.token_batches);
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

} // namespace llaminar2
