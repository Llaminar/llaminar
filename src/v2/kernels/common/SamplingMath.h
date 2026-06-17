/**
 * @file SamplingMath.h
 * @brief Shared CPU/CUDA/ROCm stochastic sampling math.
 */
#pragma once

#include <cmath>
#include <cstddef>
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
    constexpr uint64_t kInverseSampleDomain = 0xA0761D6478BD642FULL;
    constexpr uint64_t kMTPSpecDrawPurposesPerToken = 8;

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

    /**
     * @brief Deterministic MTP stochastic draw keyed by token position.
     *
     * vLLM-style speculative decoding may consume a draw in different graph
     * shapes: a bonus-ready row in one step can become the first token in the
     * next step, and verifier rows may be reduced entirely on device.  The draw
     * must therefore be keyed by logical output position and purpose instead
     * of by the host call order.  Keeping this helper shared lets CPU tests,
     * CUDA kernels, and ROCm kernels prove the same seeded thresholds.
     */
    LLAMINAR_SAMPLING_HD float mtp_spec_threshold_from_seed(
        uint64_t seed,
        int logical_position,
        int draw_purpose)
    {
        const uint64_t position =
            static_cast<uint64_t>(logical_position > 0 ? logical_position : 0);
        const uint64_t purpose =
            static_cast<uint64_t>(draw_purpose >= 0 ? draw_purpose : 0);
        return uniform01(
            seed,
            position * kMTPSpecDrawPurposesPerToken + purpose);
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
     * @brief Verify a sampled greedy draft token against a compact target table.
     *
     * The vLLM-style greedy MTP proposal is a one-hot draft distribution:
     * `q(draft_token) = 1` and `q(other) = 0`.  The compact verifier can apply
     * the same rejection-sampling math without materializing a full draft
     * probability row.  On rejection, the residual distribution is therefore
     * the target distribution with the draft token removed.
     *
     * This helper deliberately uses the vLLM/no-draft acceptance convention
     * `threshold <= p/q`.  The older compact draft-table helper keeps its
     * historical strict comparison for backwards compatibility.
     */
    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds_one_hot_draft(
        const int *target_token_ids,
        const float *target_probs,
        int k,
        int draft_token,
        float accept_threshold,
        float residual_threshold,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        const float p = distribution_probability(
            target_token_ids, target_probs, k, draft_token);
        const float accept_probability = speculative_accept_probability(p, 1.0f);
        const float threshold = clamp_unit_threshold(accept_threshold);

        if (out_accept_probability)
            *out_accept_probability = accept_probability;
        if (out_accept_threshold)
            *out_accept_threshold = threshold;

        if (threshold <= accept_probability)
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

            const float q_i = target_token_ids[i] == draft_token ? 1.0f : 0.0f;
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

    /**
     * @brief Verify one-hot greedy draft against a compact vLLM target table.
     *
     * This is the compact-table equivalent of the processed/full-probability
     * vLLM rejection kernels. Acceptance still uses `q(draft)=1`; rejection
     * samples the recovered token with inverse-exponential noise keyed by the
     * logical position and absolute vocabulary token id. Passing the full
     * vocabulary size preserves the same RNG offsets as the full-vocab path
     * even though this helper scans only the active compact support.
     */
    LLAMINAR_SAMPLING_HD void speculative_verify_with_thresholds_one_hot_draft_vllm_recovered(
        const int *target_token_ids,
        const float *target_probs,
        int k,
        int vocab_size,
        int draft_token,
        float accept_threshold,
        uint64_t inverse_sample_seed,
        int logical_position,
        int *out_token,
        int *out_accepted,
        float *out_accept_probability,
        float *out_accept_threshold)
    {
        const float p = distribution_probability(
            target_token_ids, target_probs, k, draft_token);
        const float accept_probability = speculative_accept_probability(p, 1.0f);
        const float threshold = clamp_unit_threshold(accept_threshold);

        if (out_accept_probability)
            *out_accept_probability = accept_probability;
        if (out_accept_threshold)
            *out_accept_threshold = threshold;

        if (threshold <= accept_probability)
        {
            *out_token = draft_token;
            *out_accepted = 1;
            return;
        }

        const uint64_t safe_position =
            static_cast<uint64_t>(logical_position > 0 ? logical_position : 0);
        const uint64_t safe_vocab =
            static_cast<uint64_t>(vocab_size > 0 ? vocab_size : k);
        float best_value = -1.0f;
        int best_token = -1;
        for (int i = 0; i < k; ++i)
        {
            const int token = target_token_ids[i];
            if (token < 0)
                continue;

            const float q_i = token == draft_token ? 1.0f : 0.0f;
            const float probability = fmaxf(0.0f, target_probs[i] - q_i);
            const uint64_t offset =
                safe_position * safe_vocab + static_cast<uint64_t>(token);
            const float uniform =
                uniform01(inverse_sample_seed ^ kInverseSampleDomain, offset);
            const float inverse_sample =
                inverse_exponential_from_uniform(uniform);
            const float value = probability * inverse_sample;
            if (value > best_value ||
                (value == best_value &&
                 (best_token < 0 || token < best_token)))
            {
                best_value = value;
                best_token = token;
            }
        }

        *out_token = best_token >= 0
                         ? best_token
                         : (target_token_ids[0] >= 0 ? target_token_ids[0] : draft_token);
        *out_accepted = 0;
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
     * @brief Derive live-state publication rows from compact verifier metadata.
     *
     * The compact stochastic verifier summary intentionally has two different
     * counts:
     *
     * - kSpecBatchMetaAcceptedSpeculativePrefix counts accepted MTP draft rows.
     * - kSpecBatchMetaTargetVerifierStateCommitCount counts verifier input
     *   rows whose target-model state may be published. This includes row zero,
     *   the first main-model token.
     *
     * Accepted-state publication must use the second count. Keeping this tiny
     * helper shared between CPU tests and GPU kernels prevents CUDA, ROCm, and
     * CPU from drifting on the off-by-one boundary after a rejection.
     */
    LLAMINAR_SAMPLING_HD void derive_speculative_publication_metadata(
        const int *meta,
        int meta_stride,
        int request_index,
        int padded_state_rows_per_request,
        int base_cached_tokens,
        int max_state_commit_rows,
        int *out_restore_row,
        int *out_target_cached_tokens,
        int *out_accepted_state_count,
        int *out_ok,
        const int32_t *output_tokens = nullptr,
        int output_token_stride = 0,
        int32_t *out_next_condition_token = nullptr,
        int *out_all_drafts_accepted = nullptr,
        int *out_stopped = nullptr)
    {
        if (out_restore_row)
            *out_restore_row = -1;
        if (out_target_cached_tokens)
            *out_target_cached_tokens = base_cached_tokens;
        if (out_accepted_state_count)
            *out_accepted_state_count = 0;
        if (out_next_condition_token)
            *out_next_condition_token = -1;
        if (out_all_drafts_accepted)
            *out_all_drafts_accepted = 0;
        if (out_stopped)
            *out_stopped = 0;
        if (out_ok)
            *out_ok = 0;

        if (!meta ||
            meta_stride < kSpeculativeBatchMetaCount ||
            request_index < 0 ||
            padded_state_rows_per_request <= 0 ||
            base_cached_tokens < 0 ||
            max_state_commit_rows < 0 ||
            max_state_commit_rows > padded_state_rows_per_request)
        {
            return;
        }

        const int *request_meta =
            meta + static_cast<size_t>(request_index) *
                       static_cast<size_t>(meta_stride);
        if (request_meta[kSpecBatchMetaOk] == 0)
            return;

        const int accepted_state_count =
            request_meta[kSpecBatchMetaTargetVerifierStateCommitCount];
        if (accepted_state_count < 0 ||
            accepted_state_count > max_state_commit_rows)
        {
            return;
        }

        if (out_accepted_state_count)
            *out_accepted_state_count = accepted_state_count;
        if (out_target_cached_tokens)
            *out_target_cached_tokens =
                base_cached_tokens + accepted_state_count;
        if (out_all_drafts_accepted)
            *out_all_drafts_accepted =
                request_meta[kSpecBatchMetaAllSpeculativeAccepted] != 0 ? 1 : 0;
        if (out_stopped)
            *out_stopped =
                request_meta[kSpecBatchMetaStoppedOnOutput] != 0 ? 1 : 0;
        if (out_next_condition_token && output_tokens && output_token_stride > 0)
        {
            const int ready_token =
                request_meta[kSpecBatchMetaReadyToken];
            const bool sampled_terminal =
                request_meta[kSpecBatchMetaSampledTerminal] != 0;
            if (sampled_terminal && ready_token >= 0)
            {
                *out_next_condition_token = ready_token;
            }

            const int output_count =
                request_meta[kSpecBatchMetaOutputCount];
            if (*out_next_condition_token < 0 &&
                output_count > 0 && output_count <= output_token_stride)
            {
                *out_next_condition_token =
                    output_tokens[static_cast<size_t>(request_index) *
                                      static_cast<size_t>(output_token_stride) +
                                  static_cast<size_t>(output_count - 1)];
            }
        }
        if (out_restore_row && accepted_state_count > 0)
        {
            *out_restore_row =
                request_index * padded_state_rows_per_request +
                accepted_state_count - 1;
        }
        if (out_ok)
            *out_ok = 1;
    }

    /**
     * @brief Derive shifted MTP-KV publication counts from compact metadata.
     *
     * The main verifier publication advances the target model cache to
     * `base + accepted_state_count`.  Depth `d` of the shifted sidecar cache is
     * one-or-more rows behind the main model and must instead expose
     * `max(0, target - d - 1)` rows.  The accepted count passed to a ring cache
     * is the delta from its prior shifted length so wrapped heads advance by the
     * same number of newly valid shifted rows as the old host publisher used.
     */
    LLAMINAR_SAMPLING_HD void derive_shifted_speculative_publication_metadata(
        const int *meta,
        int meta_stride,
        int request_index,
        int padded_state_rows_per_request,
        int base_cached_tokens,
        int max_state_commit_rows,
        int mtp_depth,
        int *out_target_cached_tokens,
        int *out_accepted_state_count,
        int *out_ok)
    {
        if (out_target_cached_tokens)
            *out_target_cached_tokens = 0;
        if (out_accepted_state_count)
            *out_accepted_state_count = 0;
        if (out_ok)
            *out_ok = 0;

        if (mtp_depth < 0)
            return;

        int restore_row = -1;
        int main_target_cached_tokens = base_cached_tokens;
        int main_accepted_state_count = 0;
        int ok = 0;
        derive_speculative_publication_metadata(
            meta,
            meta_stride,
            request_index,
            padded_state_rows_per_request,
            base_cached_tokens,
            max_state_commit_rows,
            &restore_row,
            &main_target_cached_tokens,
            &main_accepted_state_count,
            &ok);

        if (!ok)
            return;

        const int shift = mtp_depth + 1;
        const int base_shifted =
            base_cached_tokens > shift ? base_cached_tokens - shift : 0;
        const int target_shifted =
            main_target_cached_tokens > shift
                ? main_target_cached_tokens - shift
                : 0;
        const int accepted_shifted =
            target_shifted >= base_shifted
                ? target_shifted - base_shifted
                : 0;

        if (out_target_cached_tokens)
            *out_target_cached_tokens = target_shifted;
        if (out_accepted_state_count)
            *out_accepted_state_count = accepted_shifted;
        if (out_ok)
            *out_ok = 1;
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
