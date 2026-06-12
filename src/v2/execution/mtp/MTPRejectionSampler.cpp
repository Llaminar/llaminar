/**
 * @file MTPRejectionSampler.cpp
 * @brief Reference implementation for vLLM-style MTP rejection sampling.
 */

#include "MTPRejectionSampler.h"

#include "../../kernels/common/SamplingMath.h"

#include <algorithm>
#include <cmath>
#include <limits>
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

        MTPFullLogitRowStats fullLogitStatsFailure(
            std::string reason)
        {
            MTPFullLogitRowStats result;
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

        bool isActiveLogit(float value)
        {
            return std::isfinite(value);
        }

        float probabilityFromStatsUnchecked(
            const float *logits,
            const MTPFullLogitRowStats &stats,
            int32_t token)
        {
            if (!stats.ok || !logits || token < 0 || !(stats.exp_sum > 0.0))
                return 0.0f;

            const float logit = logits[token];
            if (!isActiveLogit(logit))
                return 0.0f;

            return static_cast<float>(
                std::exp(static_cast<double>(logit - stats.max_logit)) /
                stats.exp_sum);
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

    int32_t sampleMTPRecoveredTokenFromProbabilities(
        const float *target_probabilities,
        const float *draft_probabilities,
        const float *inverse_rejection_samples,
        int vocab_size,
        int32_t draft_token,
        bool no_draft_probabilities)
    {
        if (!target_probabilities || !inverse_rejection_samples ||
            vocab_size <= 0 || draft_token < 0 || draft_token >= vocab_size ||
            (!no_draft_probabilities && !draft_probabilities))
        {
            return -1;
        }

        int32_t best_token = 0;
        float best_value = -1.0f;
        for (int token = 0; token < vocab_size; ++token)
        {
            float probability = target_probabilities[token];
            if (!(probability > 0.0f))
                probability = 0.0f;

            if (no_draft_probabilities)
            {
                if (token == draft_token)
                    probability = 0.0f;
            }
            else
            {
                const float draft_probability =
                    draft_probabilities[token] > 0.0f
                        ? draft_probabilities[token]
                        : 0.0f;
                probability = std::max(0.0f, probability - draft_probability);
            }

            const float inverse_sample = inverse_rejection_samples[token];
            const float value =
                (inverse_sample > 0.0f) ? probability * inverse_sample : 0.0f;
            if (value > best_value ||
                (value == best_value && token < best_token))
            {
                best_value = value;
                best_token = token;
            }
        }
        return best_token;
    }

    MTPRejectionSampleRowResult sampleMTPRejectionRowFromProbabilities(
        const float *target_probabilities,
        const float *draft_probabilities,
        const float *inverse_rejection_samples,
        int vocab_size,
        int32_t draft_token,
        float accept_threshold,
        bool no_draft_probabilities)
    {
        if (!target_probabilities)
            return rejectionSampleFailure(draft_token, "target probabilities pointer is null");
        if (!inverse_rejection_samples)
            return rejectionSampleFailure(draft_token, "inverse rejection samples pointer is null");
        if (vocab_size <= 0)
            return rejectionSampleFailure(draft_token, "probability vocab size is invalid");
        if (draft_token < 0 || draft_token >= vocab_size)
            return rejectionSampleFailure(draft_token, "draft token is invalid");
        if (!no_draft_probabilities && !draft_probabilities)
            return rejectionSampleFailure(draft_token, "draft probabilities pointer is null");

        const float target_probability =
            target_probabilities[draft_token] > 0.0f
                ? target_probabilities[draft_token]
                : 0.0f;
        const float draft_probability =
            no_draft_probabilities
                ? 1.0f
                : (draft_probabilities[draft_token] > 0.0f
                       ? draft_probabilities[draft_token]
                       : 0.0f);

        MTPRejectionSampleRowResult result;
        result.ok = true;
        result.draft_token = draft_token;
        result.accept_threshold =
            sampling_math::clamp_unit_threshold(accept_threshold);
        result.accept_probability =
            Sampler::speculative_accept_probability(
                target_probability,
                draft_probability);

        if (result.accept_threshold <= result.accept_probability)
        {
            result.accepted = true;
            result.token = draft_token;
            return result;
        }

        result.token = sampleMTPRecoveredTokenFromProbabilities(
            target_probabilities,
            draft_probabilities,
            inverse_rejection_samples,
            vocab_size,
            draft_token,
            no_draft_probabilities);
        if (result.token < 0)
        {
            return rejectionSampleFailure(
                draft_token,
                "full-probability recovered-token sampling produced no token");
        }
        result.accepted = false;
        return result;
    }

    MTPFullLogitRowStats computeMTPFullLogitRowStats(
        const float *logits,
        int vocab_size)
    {
        if (!logits)
            return fullLogitStatsFailure("processed logits pointer is null");
        if (vocab_size <= 0)
            return fullLogitStatsFailure("processed logits vocab size is invalid");

        MTPFullLogitRowStats result;
        result.max_logit = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < vocab_size; ++i)
        {
            const float value = logits[i];
            if (!isActiveLogit(value))
                continue;
            if (result.argmax_token < 0 ||
                value > result.max_logit ||
                (value == result.max_logit && i < result.argmax_token))
            {
                result.argmax_token = i;
                result.max_logit = value;
            }
        }

        if (result.argmax_token < 0)
            return fullLogitStatsFailure("processed logits row has no active tokens");

        double exp_sum = 0.0;
        for (int i = 0; i < vocab_size; ++i)
        {
            const float value = logits[i];
            if (!isActiveLogit(value))
                continue;
            exp_sum += std::exp(static_cast<double>(value - result.max_logit));
        }
        if (!(exp_sum > 0.0))
            return fullLogitStatsFailure("processed logits row has zero probability mass");

        result.ok = true;
        result.exp_sum = exp_sum;
        return result;
    }

    float probabilityFromMTPFullLogits(
        const float *logits,
        int vocab_size,
        const MTPFullLogitRowStats &stats,
        int32_t token)
    {
        if (token < 0 || token >= vocab_size)
            return 0.0f;
        return probabilityFromStatsUnchecked(logits, stats, token);
    }

    int32_t sampleMTPTokenFromProcessedLogits(
        const float *logits,
        int vocab_size,
        float threshold)
    {
        MTPFullLogitRowStats stats =
            computeMTPFullLogitRowStats(logits, vocab_size);
        if (!stats.ok)
            return -1;

        const double target_mass =
            static_cast<double>(
                sampling_math::clamp_unit_threshold(threshold)) *
            stats.exp_sum;
        double cumulative = 0.0;
        int32_t selected = stats.argmax_token;
        for (int i = 0; i < vocab_size; ++i)
        {
            const float value = logits[i];
            if (!isActiveLogit(value))
                continue;
            cumulative += std::exp(static_cast<double>(value - stats.max_logit));
            if (target_mass <= cumulative)
            {
                selected = i;
                break;
            }
        }
        return selected;
    }

    MTPRejectionSampleRowResult sampleMTPRejectionRowFromProcessedLogits(
        const float *target_logits,
        const float *draft_logits,
        int vocab_size,
        int32_t draft_token,
        float accept_threshold,
        float residual_threshold)
    {
        if (draft_token < 0 || draft_token >= vocab_size)
            return rejectionSampleFailure(draft_token, "draft token is invalid");

        MTPFullLogitRowStats target_stats =
            computeMTPFullLogitRowStats(target_logits, vocab_size);
        if (!target_stats.ok)
            return rejectionSampleFailure(draft_token, target_stats.error);
        MTPFullLogitRowStats draft_stats =
            computeMTPFullLogitRowStats(draft_logits, vocab_size);
        if (!draft_stats.ok)
            return rejectionSampleFailure(draft_token, draft_stats.error);

        MTPRejectionSampleRowResult result;
        result.ok = true;
        result.draft_token = draft_token;
        result.accept_threshold =
            sampling_math::clamp_unit_threshold(accept_threshold);
        const float target_probability =
            probabilityFromStatsUnchecked(target_logits, target_stats, draft_token);
        const float draft_probability =
            probabilityFromStatsUnchecked(draft_logits, draft_stats, draft_token);
        result.accept_probability =
            Sampler::speculative_accept_probability(
                target_probability,
                draft_probability);

        if (result.accept_threshold < result.accept_probability)
        {
            result.accepted = true;
            result.token = draft_token;
            return result;
        }

        double residual_total = 0.0;
        for (int token = 0; token < vocab_size; ++token)
        {
            const float p =
                probabilityFromStatsUnchecked(target_logits, target_stats, token);
            if (!(p > 0.0f))
                continue;
            const float q =
                probabilityFromStatsUnchecked(draft_logits, draft_stats, token);
            residual_total += std::max(0.0, static_cast<double>(p) - q);
        }

        const bool use_residual = residual_total > 0.0;
        const double sample_total =
            use_residual ? residual_total : target_stats.exp_sum;
        const double target_mass =
            static_cast<double>(
                sampling_math::clamp_unit_threshold(residual_threshold)) *
            sample_total;
        double cumulative = 0.0;
        int32_t selected = target_stats.argmax_token;
        for (int token = 0; token < vocab_size; ++token)
        {
            double weight = 0.0;
            if (use_residual)
            {
                const float p =
                    probabilityFromStatsUnchecked(target_logits, target_stats, token);
                if (p > 0.0f)
                {
                    const float q =
                        probabilityFromStatsUnchecked(draft_logits, draft_stats, token);
                    weight = std::max(0.0, static_cast<double>(p) - q);
                }
            }
            else if (isActiveLogit(target_logits[token]))
            {
                weight = std::exp(static_cast<double>(
                    target_logits[token] - target_stats.max_logit));
            }

            if (!(weight > 0.0))
                continue;
            cumulative += weight;
            if (target_mass <= cumulative)
            {
                selected = token;
                break;
            }
        }

        result.accepted = false;
        result.token = selected;
        if (result.token < 0)
        {
            return rejectionSampleFailure(
                draft_token,
                "processed-logit residual sampling produced no token");
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

    MTPRejectionBatchOutcome summarizeAllPositionMTPRejectionBatchFromProcessedLogits(
        const MTPDecodeCatchupGreedyRequest &request,
        const float *target_logits,
        const float *draft_logits,
        int verifier_row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const std::vector<float> &accept_thresholds,
        const std::vector<float> &residual_thresholds,
        const float *bonus_target_logits,
        float bonus_threshold)
    {
        if (request.draft_tokens.empty())
            return stochasticOutcomeFailure(
                "processed-logit stochastic verifier received no draft tokens");
        if (!target_logits || !draft_logits)
            return stochasticOutcomeFailure(
                "processed-logit stochastic verifier received null logits");
        if (verifier_row_count < 0 ||
            verifier_row_count >
                static_cast<int>(request.draft_tokens.size()) - 1)
        {
            return stochasticOutcomeFailure(
                "processed-logit verifier row count is invalid");
        }
        if (vocab_size <= 0 ||
            target_row_stride < vocab_size ||
            draft_row_stride < vocab_size)
        {
            return stochasticOutcomeFailure(
                "processed-logit verifier shape is invalid");
        }
        if (accept_thresholds.size() < static_cast<size_t>(verifier_row_count) ||
            residual_thresholds.size() < static_cast<size_t>(verifier_row_count))
        {
            return stochasticOutcomeFailure(
                "processed-logit verifier thresholds are incomplete");
        }

        std::vector<MTPRejectionSampleRowResult> rows;
        rows.reserve(static_cast<size_t>(verifier_row_count));
        for (int row = 0; row < verifier_row_count; ++row)
        {
            const int32_t draft_token =
                request.draft_tokens[static_cast<size_t>(row + 1)];
            rows.push_back(sampleMTPRejectionRowFromProcessedLogits(
                target_logits + static_cast<size_t>(row) * target_row_stride,
                draft_logits + static_cast<size_t>(row) * draft_row_stride,
                vocab_size,
                draft_token,
                accept_thresholds[static_cast<size_t>(row)],
                residual_thresholds[static_cast<size_t>(row)]));
        }

        std::optional<int32_t> bonus_ready_token = std::nullopt;
        if (bonus_target_logits)
        {
            const int32_t token = sampleMTPTokenFromProcessedLogits(
                bonus_target_logits,
                vocab_size,
                bonus_threshold);
            if (token >= 0)
                bonus_ready_token = token;
        }

        return summarizeAllPositionMTPRejectionBatch(
            request,
            rows,
            bonus_ready_token);
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

    MTPDecodeCatchupGreedyResult buildAllPositionMTPDecodeCatchupFromProcessedLogits(
        const MTPDecodeCatchupGreedyRequest &request,
        const float *target_logits,
        const float *draft_logits,
        int verifier_row_count,
        int vocab_size,
        int target_row_stride,
        int draft_row_stride,
        const std::vector<float> &accept_thresholds,
        const std::vector<float> &residual_thresholds,
        const float *bonus_target_logits,
        float bonus_threshold)
    {
        MTPRejectionBatchOutcome outcome =
            summarizeAllPositionMTPRejectionBatchFromProcessedLogits(
                request,
                target_logits,
                draft_logits,
                verifier_row_count,
                vocab_size,
                target_row_stride,
                draft_row_stride,
                accept_thresholds,
                residual_thresholds,
                bonus_target_logits,
                bonus_threshold);
        if (!outcome.ok)
            return stochasticCatchupFailure(outcome.error);

        MTPDeviceRejectionBatchOutcome device_outcome;
        device_outcome.ok = true;
        device_outcome.output_token_count =
            static_cast<int>(outcome.output_tokens.size());
        for (int i = 0; i < device_outcome.output_token_count; ++i)
        {
            device_outcome.output_tokens[static_cast<size_t>(i)] =
                outcome.output_tokens[static_cast<size_t>(i)];
        }
        device_outcome.accepted_speculative_prefix =
            outcome.accepted_speculative_prefix;
        device_outcome.target_verifier_state_commit_count =
            outcome.target_verifier_state_commit_count;
        device_outcome.ready_token = outcome.ready_token;
        device_outcome.rejected_verified_token =
            outcome.rejected_verified_token;
        device_outcome.stopped_on_output = outcome.stopped_on_output;
        device_outcome.all_speculative_accepted =
            outcome.all_speculative_accepted;
        device_outcome.consumed_verifier_rows =
            outcome.consumed_verifier_rows;
        device_outcome.sampled_terminal = outcome.sampled_terminal;

        return buildAllPositionMTPDecodeCatchupFromDeviceBatchOutcome(
            request,
            device_outcome);
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
