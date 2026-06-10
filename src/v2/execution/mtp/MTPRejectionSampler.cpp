/**
 * @file MTPRejectionSampler.cpp
 * @brief Reference implementation for vLLM-style MTP rejection sampling.
 */

#include "MTPRejectionSampler.h"

#include "../../kernels/common/SamplingMath.h"

#include <algorithm>
#include <sstream>
#include <utility>

namespace llaminar2
{
    namespace
    {
        MTPRejectionSampleRowResult rejectionSampleFailure(
            int32_t draft_token,
            std::string reason)
        {
            MTPRejectionSampleRowResult result;
            result.ok = false;
            result.draft_token = draft_token;
            result.error = std::move(reason);
            return result;
        }

        MTPDecodeCatchupGreedyResult stochasticCatchupFailure(
            std::string reason)
        {
            MTPDecodeCatchupGreedyResult result;
            result.ok = false;
            result.error = std::move(reason);
            return result;
        }

        MTPRejectionBatchOutcome stochasticOutcomeFailure(
            std::string reason)
        {
            MTPRejectionBatchOutcome result;
            result.ok = false;
            result.error = std::move(reason);
            return result;
        }

        bool tokenIsStop(
            const std::vector<int32_t> &stop_tokens,
            int32_t token)
        {
            return std::find(stop_tokens.begin(), stop_tokens.end(), token) !=
                   stop_tokens.end();
        }

        float probabilityOfToken(
            const std::vector<SamplingDistributionEntry> &distribution,
            int32_t token)
        {
            return Sampler::probability_of_token(distribution, token);
        }

        std::vector<SamplingDistributionEntry> residualDistribution(
            const std::vector<SamplingDistributionEntry> &target,
            const std::vector<SamplingDistributionEntry> &draft)
        {
            // Reuse the public sampler residual math so the MTP verifier does
            // not grow a subtly different p-q implementation.
            return Sampler::residual_distribution(target, draft);
        }
    } // namespace

    int32_t sampleMTPDistributionWithThreshold(
        const std::vector<SamplingDistributionEntry> &distribution,
        float threshold)
    {
        if (distribution.empty())
            return -1;

        std::vector<int> token_ids(distribution.size(), -1);
        std::vector<float> probs(distribution.size(), 0.0f);
        for (size_t i = 0; i < distribution.size(); ++i)
        {
            token_ids[i] = distribution[i].token_id;
            probs[i] = distribution[i].probability;
        }

        return sampling_math::sample_distribution_with_threshold(
            token_ids.data(),
            probs.data(),
            static_cast<int>(distribution.size()),
            threshold);
    }

    MTPRejectionSampleRowResult sampleMTPRejectionRowFromDistributions(
        const std::vector<SamplingDistributionEntry> &target_distribution,
        const std::vector<SamplingDistributionEntry> &draft_distribution,
        int32_t draft_token,
        float accept_threshold,
        float residual_threshold)
    {
        if (draft_token < 0)
            return rejectionSampleFailure(draft_token, "draft token is invalid");
        if (target_distribution.empty())
            return rejectionSampleFailure(draft_token, "target distribution is empty");
        if (draft_distribution.empty())
            return rejectionSampleFailure(draft_token, "draft distribution is empty");

        MTPRejectionSampleRowResult result;
        result.ok = true;
        result.draft_token = draft_token;
        result.accept_threshold =
            sampling_math::clamp_unit_threshold(accept_threshold);
        result.accept_probability =
            Sampler::speculative_accept_probability(
                probabilityOfToken(target_distribution, draft_token),
                probabilityOfToken(draft_distribution, draft_token));

        if (result.accept_threshold < result.accept_probability)
        {
            result.accepted = true;
            result.token = draft_token;
            return result;
        }

        std::vector<SamplingDistributionEntry> residual =
            residualDistribution(target_distribution, draft_distribution);
        const std::vector<SamplingDistributionEntry> &source =
            residual.empty() ? target_distribution : residual;

        result.accepted = false;
        result.token =
            sampleMTPDistributionWithThreshold(source, residual_threshold);
        if (result.token < 0)
        {
            return rejectionSampleFailure(
                draft_token,
                "residual distribution sampling produced no token");
        }
        return result;
    }

    MTPRejectionBatchOutcome summarizeAllPositionMTPRejectionBatch(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<MTPRejectionSampleRowResult> &verified_rows,
        std::optional<int32_t> bonus_ready_token)
    {
        if (request.draft_tokens.empty())
            return stochasticOutcomeFailure(
                "stochastic all-position verifier received no draft tokens");
        if (verified_rows.size() > request.draft_tokens.size() - 1)
        {
            return stochasticOutcomeFailure(
                "stochastic verifier row count exceeds draft comparison rows");
        }

        MTPRejectionBatchOutcome result;
        result.ok = true;
        result.output_tokens.reserve(request.draft_tokens.size());
        result.verifier_tokens.reserve(verified_rows.size());

        const int32_t first_token = request.draft_tokens.front();
        result.output_tokens.push_back(first_token);
        result.target_verifier_state_commit_count = 1;

        if (tokenIsStop(request.stop_tokens, first_token))
        {
            result.stopped_on_output = true;
            return result;
        }

        bool ended_by_rejection_or_stop = false;
        for (size_t row = 0; row < verified_rows.size(); ++row)
        {
            const int draft_idx = static_cast<int>(row) + 1;
            const int32_t expected_draft =
                request.draft_tokens[static_cast<size_t>(draft_idx)];
            const MTPRejectionSampleRowResult &verified = verified_rows[row];
            if (!verified.ok)
                return stochasticOutcomeFailure(verified.error);
            if (verified.draft_token != expected_draft)
            {
                std::ostringstream msg;
                msg << "stochastic verifier row " << row
                    << " draft token mismatch: row=" << verified.draft_token
                    << ", expected=" << expected_draft;
                return stochasticOutcomeFailure(msg.str());
            }
            if (verified.token < 0)
            {
                std::ostringstream msg;
                msg << "stochastic verifier row " << row
                    << " produced an invalid token";
                return stochasticOutcomeFailure(msg.str());
            }
            if (verified.accepted && verified.token != expected_draft)
            {
                return stochasticOutcomeFailure(
                    "accepted stochastic verifier row did not return the draft token");
            }

            result.verifier_tokens.push_back(verified.token);
            result.output_tokens.push_back(verified.token);
            ++result.consumed_verifier_rows;
            if (verified.accepted)
            {
                ++result.accepted_speculative_prefix;
            }
            else
            {
                result.all_speculative_accepted = false;
                result.rejected_verified_token = verified.token;
                ended_by_rejection_or_stop = true;
            }

            if (tokenIsStop(request.stop_tokens, verified.token))
            {
                result.stopped_on_output = true;
                ended_by_rejection_or_stop = true;
            }
            if (ended_by_rejection_or_stop)
                break;
        }

        if (!ended_by_rejection_or_stop &&
            verified_rows.size() < request.draft_tokens.size() - 1)
        {
            return stochasticOutcomeFailure(
                "stochastic verifier rows ended before accept/reject/stop decision");
        }

        result.target_verifier_state_commit_count =
            std::min<int>(
                static_cast<int>(request.draft_tokens.size()),
                result.accepted_speculative_prefix + 1);

        if (!result.stopped_on_output && result.all_speculative_accepted)
        {
            if (!bonus_ready_token.has_value() || *bonus_ready_token < 0)
            {
                return stochasticOutcomeFailure(
                    "stochastic verifier accepted all drafts without a bonus ready token");
            }
            result.ready_token = *bonus_ready_token;
            result.sampled_terminal = true;
        }

        return result;
    }

    MTPDecodeCatchupGreedyResult buildAllPositionMTPDecodeCatchupStochasticResult(
        const MTPDecodeCatchupGreedyRequest &request,
        const std::vector<MTPRejectionSampleRowResult> &verified_rows,
        std::optional<int32_t> bonus_ready_token)
    {
        MTPRejectionBatchOutcome outcome =
            summarizeAllPositionMTPRejectionBatch(
                request,
                verified_rows,
                bonus_ready_token);
        if (!outcome.ok)
            return stochasticCatchupFailure(outcome.error);

        MTPDecodeCatchupGreedyResult result;
        result.ok = true;
        result.main_forward_token_count =
            static_cast<int>(request.draft_tokens.size());
        result.accepted_tokens = std::move(outcome.output_tokens);
        result.verifier_tokens = std::move(outcome.verifier_tokens);
        result.accepted_speculative_prefix =
            outcome.accepted_speculative_prefix;
        result.target_verifier_state_commit_count =
            outcome.target_verifier_state_commit_count;
        result.ready_token = outcome.ready_token;
        result.rejected_verified_token = outcome.rejected_verified_token;
        result.stopped_on_output = outcome.stopped_on_output;
        result.all_speculative_accepted = outcome.all_speculative_accepted;
        result.shifted_commit_count =
            static_cast<int>(result.accepted_tokens.size());

        std::ostringstream trace;
        trace << "stochastic_rows=" << verified_rows.size()
              << ", accepted_prefix=" << result.accepted_speculative_prefix
              << ", publish_state_count="
              << result.target_verifier_state_commit_count
              << ", ready_token=" << result.ready_token;
        result.debug_trace = trace.str();
        return result;
    }

    MTPRejectionBatchOutcome summarizeDeviceMTPRejectionBatchOutcome(
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDeviceRejectionBatchOutcome &device_outcome)
    {
        if (request.draft_tokens.empty())
            return stochasticOutcomeFailure(
                "device stochastic verifier received no draft tokens");
        if (!device_outcome.ok)
            return stochasticOutcomeFailure(
                "device stochastic verifier batch outcome is invalid");
        if (device_outcome.output_token_count < 1 ||
            device_outcome.output_token_count >
                static_cast<int>(device_outcome.output_tokens.size()))
        {
            return stochasticOutcomeFailure(
                "device stochastic verifier batch outcome token count is invalid");
        }
        if (device_outcome.output_token_count >
            static_cast<int>(request.draft_tokens.size()))
        {
            return stochasticOutcomeFailure(
                "device stochastic verifier emitted more tokens than the draft batch");
        }
        if (device_outcome.consumed_verifier_rows < 0 ||
            device_outcome.consumed_verifier_rows >
                static_cast<int>(request.draft_tokens.size()) - 1)
        {
            return stochasticOutcomeFailure(
                "device stochastic verifier consumed row count is invalid");
        }
        if (device_outcome.accepted_speculative_prefix < 0 ||
            device_outcome.accepted_speculative_prefix >
                device_outcome.consumed_verifier_rows)
        {
            return stochasticOutcomeFailure(
                "device stochastic verifier accepted prefix is invalid");
        }
        if (device_outcome.target_verifier_state_commit_count < 0 ||
            device_outcome.target_verifier_state_commit_count >
                static_cast<int>(request.draft_tokens.size()))
        {
            return stochasticOutcomeFailure(
                "device stochastic verifier state commit count is invalid");
        }
        if (device_outcome.sampled_terminal &&
            device_outcome.ready_token < 0)
        {
            return stochasticOutcomeFailure(
                "device stochastic verifier sampled terminal token is invalid");
        }

        MTPRejectionBatchOutcome result;
        result.ok = true;
        result.output_tokens.reserve(
            static_cast<size_t>(device_outcome.output_token_count));
        for (int i = 0; i < device_outcome.output_token_count; ++i)
        {
            const int32_t token =
                device_outcome.output_tokens[static_cast<size_t>(i)];
            if (token < 0)
            {
                return stochasticOutcomeFailure(
                    "device stochastic verifier emitted an invalid token");
            }
            result.output_tokens.push_back(token);
            if (i > 0)
                result.verifier_tokens.push_back(token);
        }

        result.consumed_verifier_rows =
            device_outcome.consumed_verifier_rows;
        result.accepted_speculative_prefix =
            device_outcome.accepted_speculative_prefix;
        result.target_verifier_state_commit_count =
            device_outcome.target_verifier_state_commit_count;
        result.ready_token = device_outcome.ready_token;
        result.rejected_verified_token =
            device_outcome.rejected_verified_token;
        result.stopped_on_output = device_outcome.stopped_on_output;
        result.all_speculative_accepted =
            device_outcome.all_speculative_accepted;
        result.sampled_terminal = device_outcome.sampled_terminal;
        return result;
    }

    MTPDecodeCatchupGreedyResult buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
        const MTPDecodeCatchupGreedyRequest &request,
        const MTPDeviceRejectionBatchOutcome &device_outcome)
    {
        MTPRejectionBatchOutcome outcome =
            summarizeDeviceMTPRejectionBatchOutcome(
                request,
                device_outcome);
        if (!outcome.ok)
            return stochasticCatchupFailure(outcome.error);

        MTPDecodeCatchupGreedyResult result;
        result.ok = true;
        result.main_forward_token_count =
            static_cast<int>(request.draft_tokens.size());
        result.accepted_tokens = std::move(outcome.output_tokens);
        result.verifier_tokens = std::move(outcome.verifier_tokens);
        result.accepted_speculative_prefix =
            outcome.accepted_speculative_prefix;
        result.target_verifier_state_commit_count =
            outcome.target_verifier_state_commit_count;
        result.ready_token = outcome.ready_token;
        result.rejected_verified_token = outcome.rejected_verified_token;
        result.stopped_on_output = outcome.stopped_on_output;
        result.all_speculative_accepted = outcome.all_speculative_accepted;
        result.shifted_commit_count =
            static_cast<int>(result.accepted_tokens.size());

        std::ostringstream trace;
        trace << "device_stochastic_rows=" << outcome.consumed_verifier_rows
              << ", accepted_prefix=" << result.accepted_speculative_prefix
              << ", publish_state_count="
              << result.target_verifier_state_commit_count
              << ", ready_token=" << result.ready_token;
        result.debug_trace = trace.str();
        return result;
    }

} // namespace llaminar2
