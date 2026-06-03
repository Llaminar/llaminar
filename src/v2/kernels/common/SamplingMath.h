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
    constexpr float kMaxUnitThreshold = 0.99999994f;

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

    LLAMINAR_SAMPLING_HD int sample_distribution_with_threshold(
        const int *token_ids,
        const float *probs,
        int k,
        float threshold)
    {
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
                break;
            }
        }
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
        const float p = distribution_probability(
            target_token_ids, target_probs, k, draft_token);
        const float q = distribution_probability(
            draft_token_ids, draft_probs, k, draft_token);
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

} // namespace llaminar2::sampling_math

#undef LLAMINAR_SAMPLING_HD
