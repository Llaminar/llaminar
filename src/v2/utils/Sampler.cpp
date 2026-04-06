/**
 * @file Sampler.cpp
 * @brief Implementation of token sampling strategies
 * @author David Sanftenberg
 * @date 2025
 */

#include "Sampler.h"
#include <limits>
#include <stdexcept>

namespace llaminar2
{

    Sampler::Sampler(unsigned int seed)
    {
        set_seed(seed);
    }

    void Sampler::set_seed(unsigned int seed)
    {
        if (seed == 0)
        {
            std::random_device rd;
            rng_.seed(rd());
        }
        else
        {
            rng_.seed(seed);
        }
    }

    int Sampler::sample(const std::vector<float> &logits, const SamplingParams &params)
    {
        return sample(logits.data(), logits.size(), params);
    }

    int Sampler::sample(const float *logits, size_t vocab_size, const SamplingParams &params)
    {
        if (!logits || vocab_size == 0)
        {
            throw std::invalid_argument("Cannot sample from empty logits");
        }

        // Greedy sampling (deterministic) - zero-copy path
        // Note: penalties still apply even for greedy (they shift the argmax)
        if (params.is_greedy() && !params.has_penalties())
        {
            return sample_greedy(logits, vocab_size);
        }

        // Non-greedy paths (or greedy with penalties) need a mutable vector
        std::vector<float> logits_vec(logits, logits + vocab_size);

        // Apply presence/frequency penalties before temperature and filtering
        if (params.has_penalties())
        {
            apply_penalties(logits_vec, params);
        }

        // Greedy with penalties: argmax on penalized logits
        if (params.is_greedy())
        {
            return sample_greedy(logits_vec);
        }

        // Apply top-k if specified
        if (params.top_k > 0 && params.top_k < static_cast<int>(vocab_size))
        {
            return sample_top_k(logits_vec, params.top_k, params.temperature);
        }

        // Apply top-p if specified
        if (params.top_p < 1.0f)
        {
            return sample_top_p(logits_vec, params.top_p, params.temperature);
        }

        // Temperature sampling only
        if (params.temperature != 1.0f)
        {
            return sample_temperature(logits_vec, params.temperature);
        }

        // Standard sampling (temperature = 1.0, no filtering)
        auto probs = softmax(logits_vec);
        return sample_from_probs(probs);
    }

    int Sampler::sample_greedy(const std::vector<float> &logits)
    {
        if (logits.empty())
        {
            throw std::invalid_argument("Cannot sample from empty logits");
        }

        return std::max_element(logits.begin(), logits.end()) - logits.begin();
    }

    int Sampler::sample_greedy(const float *logits, size_t vocab_size)
    {
        if (!logits || vocab_size == 0)
        {
            throw std::invalid_argument("Cannot sample from empty logits");
        }

        return std::max_element(logits, logits + vocab_size) - logits;
    }

    int Sampler::sample_temperature(const std::vector<float> &logits, float temperature)
    {
        if (temperature == 0.0f)
        {
            return sample_greedy(logits);
        }

        auto scaled_logits = apply_temperature(logits, temperature);
        auto probs = softmax(scaled_logits);
        return sample_from_probs(probs);
    }

    int Sampler::sample_top_k(const std::vector<float> &logits, int k, float temperature)
    {
        if (k <= 0)
        {
            throw std::invalid_argument("Top-k must be positive");
        }
        if (k >= static_cast<int>(logits.size()))
        {
            // k >= vocab_size, no filtering needed
            return sample_temperature(logits, temperature);
        }

        // Apply temperature scaling
        auto scaled_logits = apply_temperature(logits, temperature);

        // Create index-value pairs
        std::vector<std::pair<int, float>> indexed_logits;
        indexed_logits.reserve(scaled_logits.size());
        for (size_t i = 0; i < scaled_logits.size(); ++i)
        {
            indexed_logits.emplace_back(i, scaled_logits[i]);
        }

        // Partial sort to get top-k (more efficient than full sort)
        std::partial_sort(
            indexed_logits.begin(),
            indexed_logits.begin() + k,
            indexed_logits.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

        // Extract top-k logits
        std::vector<float> top_k_logits(k);
        std::vector<int> top_k_indices(k);
        for (int i = 0; i < k; ++i)
        {
            top_k_indices[i] = indexed_logits[i].first;
            top_k_logits[i] = indexed_logits[i].second;
        }

        // Softmax over top-k
        auto probs = softmax(top_k_logits);

        // Sample from top-k distribution
        int sampled_idx = sample_from_probs(probs);

        // Map back to original vocabulary index
        return top_k_indices[sampled_idx];
    }

    int Sampler::sample_top_p(const std::vector<float> &logits, float p, float temperature)
    {
        if (p <= 0.0f || p > 1.0f)
        {
            throw std::invalid_argument("Top-p must be in range (0.0, 1.0]");
        }

        // Apply temperature scaling
        auto scaled_logits = apply_temperature(logits, temperature);

        // Create index-value pairs
        std::vector<std::pair<int, float>> indexed_logits;
        indexed_logits.reserve(scaled_logits.size());
        for (size_t i = 0; i < scaled_logits.size(); ++i)
        {
            indexed_logits.emplace_back(i, scaled_logits[i]);
        }

        // Sort by logit value (descending)
        std::sort(
            indexed_logits.begin(),
            indexed_logits.end(),
            [](const auto &a, const auto &b)
            { return a.second > b.second; });

        // Convert to probabilities and accumulate
        float max_logit = indexed_logits[0].second;
        std::vector<float> probs;
        std::vector<int> indices;
        float cumsum = 0.0f;
        float total_exp = 0.0f;

        // First pass: compute exp(logit - max) for numerical stability
        std::vector<float> exp_vals;
        for (const auto &[idx, logit] : indexed_logits)
        {
            float exp_val = std::exp(logit - max_logit);
            exp_vals.push_back(exp_val);
            total_exp += exp_val;
        }

        // Second pass: accumulate normalized probabilities until threshold
        for (size_t i = 0; i < indexed_logits.size(); ++i)
        {
            float prob = exp_vals[i] / total_exp;
            cumsum += prob;

            indices.push_back(indexed_logits[i].first);
            probs.push_back(indexed_logits[i].second);

            if (cumsum >= p)
            {
                break;
            }
        }

        // Ensure at least one token is included
        if (probs.empty())
        {
            return indexed_logits[0].first;
        }

        // Softmax over nucleus
        auto nucleus_probs = softmax(probs);

        // Sample from nucleus
        int sampled_idx = sample_from_probs(nucleus_probs);

        // Map back to original vocabulary index
        return indices[sampled_idx];
    }

    std::vector<float> Sampler::apply_temperature(const std::vector<float> &logits, float temperature)
    {
        if (temperature == 1.0f)
        {
            return logits; // No scaling needed
        }

        std::vector<float> scaled(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            scaled[i] = logits[i] / temperature;
        }
        return scaled;
    }

    std::vector<float> Sampler::softmax(const std::vector<float> &logits)
    {
        if (logits.empty())
        {
            return {};
        }

        // Find max for numerical stability
        float max_logit = *std::max_element(logits.begin(), logits.end());

        // Compute exp(logit - max)
        std::vector<float> exp_vals(logits.size());
        float sum_exp = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i)
        {
            exp_vals[i] = std::exp(logits[i] - max_logit);
            sum_exp += exp_vals[i];
        }

        // Normalize
        std::vector<float> probs(logits.size());
        for (size_t i = 0; i < logits.size(); ++i)
        {
            probs[i] = exp_vals[i] / sum_exp;
        }

        return probs;
    }

    int Sampler::sample_from_probs(const std::vector<float> &probs)
    {
        if (probs.empty())
        {
            throw std::invalid_argument("Cannot sample from empty probability distribution");
        }

        // Generate random value in [0, 1)
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);

        // Cumulative sampling
        float cumsum = 0.0f;
        for (size_t i = 0; i < probs.size(); ++i)
        {
            cumsum += probs[i];
            if (r < cumsum)
            {
                return i;
            }
        }

        // Fallback to last token (handles floating point rounding)
        return probs.size() - 1;
    }

    void Sampler::record_token(int token_id)
    {
        token_counts_[token_id]++;
    }

    void Sampler::reset_history()
    {
        token_counts_.clear();
    }

    void Sampler::apply_penalties(std::vector<float> &logits, const SamplingParams &params)
    {
        if (token_counts_.empty())
        {
            return; // No history, nothing to penalize
        }

        // OpenAI-style penalties (applied additively to logits):
        //   logit[t] -= presence_penalty * (1 if t in history else 0)
        //   logit[t] -= frequency_penalty * count(t)
        for (const auto &[token_id, count] : token_counts_)
        {
            if (token_id >= 0 && token_id < static_cast<int>(logits.size()))
            {
                if (params.presence_penalty != 0.0f)
                {
                    logits[token_id] -= params.presence_penalty;
                }
                if (params.frequency_penalty != 0.0f)
                {
                    logits[token_id] -= params.frequency_penalty * static_cast<float>(count);
                }
            }
        }
    }

} // namespace llaminar2
