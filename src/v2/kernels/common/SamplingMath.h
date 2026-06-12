/**
 * @file SamplingMath.h
 * @brief Shared CPU/CUDA/ROCm stochastic sampling math.
 */
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_SAMPLING_HD __host__ __device__ inline
#else
#define LLAMINAR_SAMPLING_HD inline
#endif

namespace llaminar2::sampling_math
{
    constexpr int kMaxTopK = 256;
    constexpr int kSpeculativeBatchMaxRows = 4;
    constexpr int kSpeculativeBatchMaxOutputTokens =
        kSpeculativeBatchMaxRows + 1;
    constexpr int kSpeculativeBatchMaxStopTokens = 8;
    constexpr int kSpeculativeBatchMetaCount = 10;
    constexpr float kMaxUnitThreshold = 0.99999994f;

    enum SpeculativeBatchMetaIndex : int
    {
        kSpecBatchMetaOk = 0,
        kSpecBatchMetaOutputCount = 1,
        kSpecBatchMetaAcceptedSpeculativePrefix = 2,
        kSpecBatchMetaTargetVerifierStateCommitCount = 3,
        kSpecBatchMetaReadyToken = 4,
        kSpecBatchMetaRejectedVerifiedToken = 5,
        kSpecBatchMetaStoppedOnOutput = 6,
        kSpecBatchMetaAllSpeculativeAccepted = 7,
        kSpecBatchMetaConsumedVerifierRows = 8,
        kSpecBatchMetaSampledTerminal = 9
    };

    LLAMINAR_SAMPLING_HD uint64_t splitmix64(uint64_t x)
    {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    }

    LLAMINAR_SAMPLING_HD float uniform01(uint64_t seed, uint64_t offset)
    {
        const uint64_t bits = splitmix64(seed + offset);
        return static_cast<float>((bits >> 40) & 0xFFFFFFull) *
               (1.0f / 16777216.0f);
    }

    LLAMINAR_SAMPLING_HD float clamp_unit_threshold(float threshold)
    {
        return fminf(fmaxf(threshold, 0.0f), kMaxUnitThreshold);
    }

    /**
     * @brief Convert a uniform draw into vLLM-style inverse exponential noise.
     *
     * vLLM's stochastic rejection sampler picks recovered tokens by maximizing
     * `probability * inv_q[token]`, where `inv_q` is the reciprocal of an
     * exponential random variable. Keeping this tiny transform shared prevents
     * CPU, CUDA, and ROCm from drifting on clamp behavior near zero.
     */
    LLAMINAR_SAMPLING_HD float inverse_exponential_from_uniform(float uniform)
    {
        constexpr float kMinUniform = 1.0f / 16777216.0f;
        constexpr float kMinExponential = 1.0e-20f;
        const float u = fminf(fmaxf(uniform, kMinUniform), kMaxUnitThreshold);
        const float exponential = fmaxf(-logf(u), kMinExponential);
        return 1.0f / exponential;
    }

    LLAMINAR_SAMPLING_HD float speculative_accept_probability(
        float target_probability,
        float draft_probability)
    {
        if (!(target_probability > 0.0f) || !(draft_probability > 0.0f))
            return 0.0f;
        return fminf(1.0f, target_probability / draft_probability);
    }

    LLAMINAR_SAMPLING_HD float distribution_probability(
        const int *token_ids,
        const float *probs,
        int k,
        int token_id)
    {
        for (int i = 0; i < k; ++i)
        {
            if (token_ids[i] == token_id)
                return probs[i];
        }
        return 0.0f;
    }

    /**
     * @brief Sample a compact probability table and optionally return p(token).
     *
     * The selected probability is the stored compact-table probability for the
     * token that won the threshold scan. Keeping this helper shared prevents
     * CUDA, ROCm, and CPU-side verifier plumbing from drifting on edge cases
     * such as inactive token slots or clamped thresholds.
     */
    LLAMINAR_SAMPLING_HD int sample_distribution_with_threshold_and_probability(
        const int *token_ids,
        const float *probs,
        int k,
        float threshold,
        float *out_probability)
    {
        if (out_probability)
            *out_probability = 0.0f;

        float total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (token_ids[i] >= 0 && probs[i] > 0.0f)
                total += probs[i];
        }
        if (!(total > 0.0f))
            return -1;

        const float r = clamp_unit_threshold(threshold) * total;
        float cumulative = 0.0f;
        int selected = -1;
        for (int i = 0; i < k; ++i)
        {
            if (token_ids[i] < 0 || !(probs[i] > 0.0f))
                continue;
            if (selected < 0)
                selected = token_ids[i];
            cumulative += probs[i];
            if (r <= cumulative)
            {
                selected = token_ids[i];
                if (out_probability)
                    *out_probability = probs[i];
                break;
            }
        }
        if (out_probability && selected >= 0 && *out_probability == 0.0f)
            *out_probability = distribution_probability(token_ids, probs, k, selected);
        return selected;
    }

    LLAMINAR_SAMPLING_HD int build_topk_topp_distribution_from_sorted(
        const float *sorted_logits,
        const int *sorted_token_ids,
        int k,
        float top_p,
        float temperature,
        int *out_token_ids,
        float *out_probs,
        float *scratch_weights)
    {
        const float temp = temperature > 0.0f ? temperature : 1.0f;
        const float max_logit = sorted_logits[0];
        float total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (sorted_token_ids[i] < 0)
            {
                scratch_weights[i] = 0.0f;
                continue;
            }
            const float w = expf((sorted_logits[i] - max_logit) / temp);
            scratch_weights[i] = w;
            total += w;
        }

        int nucleus = k;
        if (total > 0.0f && top_p > 0.0f && top_p < 1.0f)
        {
            float cumulative = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                cumulative += scratch_weights[i] / total;
                if (cumulative >= top_p)
                {
                    nucleus = i + 1;
                    break;
                }
            }
        }

        float nucleus_total = 0.0f;
        for (int i = 0; i < nucleus; ++i)
            nucleus_total += scratch_weights[i];

        int active = 0;
        for (int i = 0; i < k; ++i)
        {
            if (i < nucleus && nucleus_total > 0.0f && sorted_token_ids[i] >= 0)
            {
                out_token_ids[i] = sorted_token_ids[i];
                out_probs[i] = scratch_weights[i] / nucleus_total;
                ++active;
            }
            else
            {
                out_token_ids[i] = -1;
                out_probs[i] = 0.0f;
            }
        }
        return active;
    }

    LLAMINAR_SAMPLING_HD int sample_topk_topp_from_sorted_with_threshold(
        const float *sorted_logits,
        const int *sorted_token_ids,
        int k,
        float top_p,
        float temperature,
        float threshold,
        float *scratch_weights)
    {
        const float temp = temperature > 0.0f ? temperature : 1.0f;
        const float max_logit = sorted_logits[0];
        float total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (sorted_token_ids[i] < 0)
            {
                scratch_weights[i] = 0.0f;
                continue;
            }
            const float w = expf((sorted_logits[i] - max_logit) / temp);
            scratch_weights[i] = w;
            total += w;
        }

        if (!(total > 0.0f))
            return sorted_token_ids[0] >= 0 ? sorted_token_ids[0] : 0;

        int nucleus = k;
        if (top_p > 0.0f && top_p < 1.0f)
        {
            float cumulative = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                cumulative += scratch_weights[i] / total;
                if (cumulative >= top_p)
                {
                    nucleus = i + 1;
                    break;
                }
            }
        }

        float nucleus_total = 0.0f;
        for (int i = 0; i < nucleus; ++i)
            nucleus_total += scratch_weights[i];
        if (!(nucleus_total > 0.0f))
            return sorted_token_ids[0] >= 0 ? sorted_token_ids[0] : 0;

        const float r = clamp_unit_threshold(threshold) * nucleus_total;
        float cumulative = 0.0f;
        int selected = sorted_token_ids[0] >= 0 ? sorted_token_ids[0] : 0;
        for (int i = 0; i < nucleus; ++i)
        {
            cumulative += scratch_weights[i];
            if (r <= cumulative)
            {
                selected = sorted_token_ids[i] >= 0 ? sorted_token_ids[i] : selected;
                break;
            }
        }
        return selected;
    }

    LLAMINAR_SAMPLING_HD int sample_distribution_with_threshold(
        const int *token_ids,
        const float *probs,
        int k,
        float threshold)
    {
        return sample_distribution_with_threshold_and_probability(
            token_ids,
            probs,
            k,
            threshold,
            nullptr);
    }

    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds_and_draft_probability(
        const int *target_token_ids,
        const float *target_probs,
        const int *draft_token_ids,
        const float *draft_probs,
        int k,
        int draft_token,
        float sampled_draft_probability,
        bool has_sampled_draft_probability,
        float accept_threshold,
        float residual_threshold,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        const float p = distribution_probability(
            target_token_ids, target_probs, k, draft_token);
        const float q =
            has_sampled_draft_probability && sampled_draft_probability > 0.0f
                ? sampled_draft_probability
                : distribution_probability(draft_token_ids, draft_probs, k, draft_token);
        const float accept_probability = speculative_accept_probability(p, q);
        const float threshold = clamp_unit_threshold(accept_threshold);

        if (out_accept_probability)
            *out_accept_probability = accept_probability;
        if (out_accept_threshold)
            *out_accept_threshold = threshold;

        if (threshold < accept_probability)
        {
            *out_token = draft_token;
            *out_accepted = 1;
            return;
        }

        float residual_weights[kMaxTopK];
        float residual_total = 0.0f;
        for (int i = 0; i < k; ++i)
        {
            if (target_token_ids[i] < 0)
            {
                residual_weights[i] = 0.0f;
                continue;
            }
            const float q_i = distribution_probability(
                draft_token_ids,
                draft_probs,
                k,
                target_token_ids[i]);
            const float w = fmaxf(0.0f, target_probs[i] - q_i);
            residual_weights[i] = w;
            residual_total += w;
        }

        if (!(residual_total > 0.0f))
        {
            residual_total = 0.0f;
            for (int i = 0; i < k; ++i)
            {
                residual_weights[i] = target_token_ids[i] >= 0 ? target_probs[i] : 0.0f;
                residual_total += residual_weights[i];
            }
        }

        const float r = clamp_unit_threshold(residual_threshold) * residual_total;
        float cumulative = 0.0f;
        int selected = target_token_ids[0] >= 0 ? target_token_ids[0] : draft_token;
        for (int i = 0; i < k; ++i)
        {
            cumulative += residual_weights[i];
            if (r <= cumulative)
            {
                selected = target_token_ids[i] >= 0 ? target_token_ids[i] : selected;
                break;
            }
        }

        *out_token = selected;
        *out_accepted = 0;
    }

    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds(
        const int *target_token_ids,
        const float *target_probs,
        const int *draft_token_ids,
        const float *draft_probs,
        int k,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        speculative_verify_with_thresholds_and_draft_probability(
            target_token_ids,
            target_probs,
            draft_token_ids,
            draft_probs,
            k,
            draft_token,
            0.0f,
            false,
            accept_threshold,
            residual_threshold,
            out_token,
            out_accepted,
            out_accept_probability,
            out_accept_threshold);
    }

    /**
     * @brief Reduce row-wise speculative verifier decisions into one commit plan.
     *
     * Row kernels decide the stochastic accept/reject token independently. This
     * reducer applies the autoregressive semantics: emit tokens until the first
     * rejection or stop token, count the accepted speculative prefix, and expose
     * a ready token only when every verifier row accepted. The metadata layout
     * is fixed by SpeculativeBatchMetaIndex so host tests and GPU kernels cannot
     * drift.
     */
    LLAMINAR_SAMPLING_HD void summarize_speculative_verify_batch(
        int first_token,
        const int *row_tokens,
        const int *row_accepted,
        int row_count,
        const int *stop_tokens,
        int stop_token_count,
        int bonus_ready_token,
        int has_bonus_ready_token,
        int *out_tokens,
        int *out_meta)
    {
        if (!out_tokens || !out_meta ||
            row_count < 0 ||
            row_count > kSpeculativeBatchMaxRows ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens)
        {
            if (out_meta)
                out_meta[kSpecBatchMetaOk] = 0;
            return;
        }

        for (int i = 0; i < kSpeculativeBatchMaxOutputTokens; ++i)
            out_tokens[i] = -1;
        for (int i = 0; i < kSpeculativeBatchMetaCount; ++i)
            out_meta[i] = 0;

        if (first_token < 0)
        {
            out_meta[kSpecBatchMetaOk] = 0;
            return;
        }

        int output_count = 1;
        int consumed_rows = 0;
        int accepted_prefix = 0;
        int rejected_token = -1;
        bool stopped = false;
        for (int i = 0; i < stop_token_count; ++i)
        {
            if (stop_tokens && stop_tokens[i] == first_token)
            {
                stopped = true;
                break;
            }
        }
        bool all_accepted = true;
        out_tokens[0] = first_token;

        for (int row = 0; !stopped && row < row_count; ++row)
        {
            if (!row_tokens || !row_accepted || row_tokens[row] < 0)
            {
                out_meta[kSpecBatchMetaOk] = 0;
                return;
            }

            const int token = row_tokens[row];
            const bool accepted = row_accepted[row] != 0;
            out_tokens[output_count++] = token;
            ++consumed_rows;

            if (accepted)
            {
                ++accepted_prefix;
            }
            else
            {
                all_accepted = false;
                rejected_token = token;
            }

            for (int i = 0; i < stop_token_count; ++i)
            {
                if (stop_tokens && stop_tokens[i] == token)
                {
                    stopped = true;
                    break;
                }
            }
            if (!accepted)
                break;
        }

        int ready_token = -1;
        int sampled_terminal = 0;
        if (!stopped && all_accepted)
        {
            if (!has_bonus_ready_token || bonus_ready_token < 0)
            {
                out_meta[kSpecBatchMetaOk] = 0;
                return;
            }
            ready_token = bonus_ready_token;
            sampled_terminal = 1;
        }

        const int commit_count =
            accepted_prefix + 1 < row_count + 1
                ? accepted_prefix + 1
                : row_count + 1;

        out_meta[kSpecBatchMetaOk] = 1;
        out_meta[kSpecBatchMetaOutputCount] = output_count;
        out_meta[kSpecBatchMetaAcceptedSpeculativePrefix] = accepted_prefix;
        out_meta[kSpecBatchMetaTargetVerifierStateCommitCount] = commit_count;
        out_meta[kSpecBatchMetaReadyToken] = ready_token;
        out_meta[kSpecBatchMetaRejectedVerifiedToken] = rejected_token;
        out_meta[kSpecBatchMetaStoppedOnOutput] = stopped ? 1 : 0;
        out_meta[kSpecBatchMetaAllSpeculativeAccepted] = all_accepted ? 1 : 0;
        out_meta[kSpecBatchMetaConsumedVerifierRows] = consumed_rows;
        out_meta[kSpecBatchMetaSampledTerminal] = sampled_terminal;
    }

    /**
     * @brief Decide whether a speculative batch needs a bonus ready token.
     *
     * GPU lazy verifier kernels use this before sampling the bonus distribution:
     * if the first token stops, any verifier row stops, or any verifier row
     * rejects, the bonus token is not semantically consumed and should not burn
     * a stochastic sample.  The full reducer still owns the final metadata; this
     * helper only answers the cheap "is bonus needed?" question from the same
     * shared semantics.
     */
    LLAMINAR_SAMPLING_HD bool speculative_batch_needs_bonus_ready_token(
        int first_token,
        const int *row_tokens,
        const int *row_accepted,
        int row_count,
        const int *stop_tokens,
        int stop_token_count)
    {
        if (first_token < 0 ||
            row_count < 0 ||
            row_count > kSpeculativeBatchMaxRows ||
            stop_token_count < 0 ||
            stop_token_count > kSpeculativeBatchMaxStopTokens)
        {
            return false;
        }

        for (int i = 0; i < stop_token_count; ++i)
        {
            if (stop_tokens && stop_tokens[i] == first_token)
                return false;
        }

        for (int row = 0; row < row_count; ++row)
        {
            if (!row_tokens || !row_accepted || row_tokens[row] < 0)
                return false;

            const int token = row_tokens[row];
            if (row_accepted[row] == 0)
                return false;

            for (int i = 0; i < stop_token_count; ++i)
            {
                if (stop_tokens && stop_tokens[i] == token)
                    return false;
            }
        }

        return true;
    }

    /**
     * @brief Summarize greedy speculative verifier rows from device tokens.
     *
     * `verifier_tokens[row]` is the target-model greedy token for verifier row
     * `row`. `draft_tokens[0]` is the first already-sampled target token and
     * `draft_tokens[row + 1]` is the speculative draft token checked by
     * `verifier_tokens[row]`. `verifier_tokens[compare_row_count]` is the bonus
     * ready token consumed only when every speculative row accepts.
     */
    LLAMINAR_SAMPLING_HD void summarize_greedy_speculative_verify_batch(
        int first_token,
        const int *verifier_tokens,
        const int *draft_tokens,
        int compare_row_count,
        const int *stop_tokens,
        int stop_token_count,
        int *out_tokens,
        int *out_meta)
    {
        if (!verifier_tokens || !draft_tokens || compare_row_count < 0 ||
            compare_row_count > kSpeculativeBatchMaxRows)
        {
            if (out_meta)
                out_meta[kSpecBatchMetaOk] = 0;
            return;
        }

        int row_accepted[kSpeculativeBatchMaxRows] = {0, 0, 0, 0};
        for (int row = 0; row < compare_row_count; ++row)
        {
            row_accepted[row] =
                verifier_tokens[row] == draft_tokens[row + 1] ? 1 : 0;
        }

        const int bonus_ready_token = verifier_tokens[compare_row_count];
        summarize_speculative_verify_batch(
            first_token,
            verifier_tokens,
            row_accepted,
            compare_row_count,
            stop_tokens,
            stop_token_count,
            bonus_ready_token,
            /*has_bonus_ready_token=*/1,
            out_tokens,
            out_meta);
    }

} // namespace llaminar2::sampling_math

#undef LLAMINAR_SAMPLING_HD
