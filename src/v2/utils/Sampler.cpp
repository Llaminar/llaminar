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

        auto distribution = compute_distribution(logits, vocab_size, params);
        return sample_from_distribution(distribution);
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

    std::vector<SamplingDistributionEntry> Sampler::compute_distribution(
        const float *logits,
        size_t vocab_size,
        const SamplingParams &params)
    {
        if (!logits || vocab_size == 0)
        {
            throw std::invalid_argument("Cannot build distribution from empty logits");
        }

        std::vector<float> logits_vec(logits, logits + vocab_size);
        if (params.has_penalties())
        {
            apply_penalties(logits_vec, params);
        }

        if (params.is_greedy())
        {
            return {{sample_greedy(logits_vec), 1.0f}};
        }

        float temperature = params.temperature;
        if (temperature <= 0.0f)
        {
            temperature = 1.0f;
        }

        std::vector<std::pair<int, float>> candidates;
        candidates.reserve(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i)
        {
            candidates.emplace_back(static_cast<int>(i), logits_vec[i] / temperature);
        }

        int candidate_count = static_cast<int>(candidates.size());
        if (params.top_k > 0 && params.top_k < candidate_count)
        {
            candidate_count = params.top_k;
            std::partial_sort(
                candidates.begin(),
                candidates.begin() + candidate_count,
                candidates.end(),
                [](const auto &a, const auto &b)
                {
                    return a.second > b.second;
                });
            candidates.resize(static_cast<size_t>(candidate_count));
        }
        else
        {
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const auto &a, const auto &b)
                {
                    return a.second > b.second;
                });
        }

        const float max_logit = candidates.front().second;
        std::vector<float> exp_vals(candidates.size(), 0.0f);
        float total_exp = 0.0f;
        for (size_t i = 0; i < candidates.size(); ++i)
        {
            exp_vals[i] = std::exp(candidates[i].second - max_logit);
            total_exp += exp_vals[i];
        }
        if (!(total_exp > 0.0f))
        {
            return {{candidates.front().first, 1.0f}};
        }

        size_t nucleus_count = candidates.size();
        if (params.top_p > 0.0f && params.top_p < 1.0f)
        {
            float cumulative = 0.0f;
            for (size_t i = 0; i < candidates.size(); ++i)
            {
                cumulative += exp_vals[i] / total_exp;
                if (cumulative >= params.top_p)
                {
                    nucleus_count = i + 1;
                    break;
                }
            }
        }

        float nucleus_exp = 0.0f;
        for (size_t i = 0; i < nucleus_count; ++i)
        {
            nucleus_exp += exp_vals[i];
        }
        if (!(nucleus_exp > 0.0f))
        {
            return {{candidates.front().first, 1.0f}};
        }

        std::vector<SamplingDistributionEntry> distribution;
        distribution.reserve(nucleus_count);
        for (size_t i = 0; i < nucleus_count; ++i)
        {
            distribution.push_back({candidates[i].first, exp_vals[i] / nucleus_exp});
        }
        return distribution;
    }

    int Sampler::sample_from_distribution(
        const std::vector<SamplingDistributionEntry> &distribution)
    {
        if (distribution.empty())
        {
            throw std::invalid_argument("Cannot sample from empty distribution");
        }

        const float r = random_uniform_01();
        float cumulative = 0.0f;
        for (const auto &entry : distribution)
        {
            cumulative += entry.probability;
            if (r < cumulative)
            {
                return entry.token_id;
            }
        }
        return distribution.back().token_id;
    }

    int Sampler::sample_from_residual_distribution(
        const std::vector<SamplingDistributionEntry> &target,
        const std::vector<SamplingDistributionEntry> &draft)
    {
        if (target.empty())
        {
            throw std::invalid_argument("Cannot sample residual from empty target distribution");
        }

        auto residual = residual_distribution(target, draft);
        if (residual.empty())
        {
            return sample_from_distribution(target);
        }
        return sample_from_distribution(residual);
    }

    std::vector<SamplingDistributionEntry> Sampler::residual_distribution(
        const std::vector<SamplingDistributionEntry> &target,
        const std::vector<SamplingDistributionEntry> &draft)
    {
        if (target.empty())
        {
            throw std::invalid_argument("Cannot compute residual from empty target distribution");
        }

        std::unordered_map<int, float> draft_probs;
        draft_probs.reserve(draft.size());
        for (const auto &entry : draft)
        {
            draft_probs[entry.token_id] = entry.probability;
        }

        std::vector<SamplingDistributionEntry> residual;
        residual.reserve(target.size());
        float total = 0.0f;
        for (const auto &entry : target)
        {
            const auto it = draft_probs.find(entry.token_id);
            const float q = it == draft_probs.end() ? 0.0f : it->second;
            const float p = std::max(0.0f, entry.probability - q);
            if (p > 0.0f)
            {
                residual.push_back({entry.token_id, p});
                total += p;
            }
        }

        if (!(total > 0.0f))
        {
            return {};
        }
        for (auto &entry : residual)
        {
            entry.probability /= total;
        }
        return residual;
    }

    float Sampler::speculative_accept_probability(
        float target_probability,
        float draft_probability)
    {
        return draft_probability > 0.0f
                   ? std::min(1.0f, std::max(0.0f, target_probability) / draft_probability)
                   : 0.0f;
    }

    float Sampler::probability_of_token(
        const std::vector<SamplingDistributionEntry> &distribution,
        int token_id)
    {
        for (const auto &entry : distribution)
        {
            if (entry.token_id == token_id)
            {
                return entry.probability;
            }
        }
        return 0.0f;
    }

    float Sampler::random_uniform_01()
    {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(rng_);
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
        float r = random_uniform_01();

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
        dry_token_history_.push_back(token_id);
    }

    void Sampler::reset_history()
    {
        token_counts_.clear();
        dry_token_history_.clear();
    }

    void Sampler::initDryBreakers(const std::vector<std::string> &breaker_strings,
                                   const TokenizeFunc &tokenize)
    {
        dry_breakers_.clear();
        for (const auto &s : breaker_strings)
        {
            auto tokens = tokenize(s);
            if (tokens.empty())
                continue;
            int head = tokens.back(); // Match from end of history
            std::vector<int> tail(tokens.begin(), tokens.end() - 1);
            dry_breakers_.emplace(head, std::move(tail));
        }
        dry_breakers_initialized_ = true;
    }

    void Sampler::compute_dry_penalties(std::unordered_map<int, float> &penalty_map,
                                         const SamplingParams &params, int vocab_size)
    {
        if (params.dry_multiplier == 0.0f || params.dry_base < 1.0f || params.dry_penalty_last_n == 0)
            return;

        const auto &history = dry_token_history_;
        if (history.empty())
            return;

        int history_size = static_cast<int>(history.size());

        // Window: how far back to look
        int window = params.dry_penalty_last_n;
        if (window < 0 || window > history_size)
            window = history_size;

        // We need at least 1 token of history to detect repeats
        if (window < 1)
            return;

        // Build reversed view of the history window for Z-algorithm
        std::vector<int> reversed(window);
        for (int i = 0; i < window; ++i)
            reversed[i] = history[history_size - 1 - i];

        // Step 1: Scan for sequence breakers to compute rep_limit
        // (Aligned with llama.cpp's breaker scan approach)
        int rep_limit = window;
        if (dry_breakers_initialized_)
        {
            for (int i = 0; i < window; ++i)
            {
                int token = reversed[i]; // = history[history_size - 1 - i]
                auto its = dry_breakers_.equal_range(token);
                if (its.first == dry_breakers_.end())
                    continue;
                int longest_match = -1;
                for (auto it = its.first; it != its.second; ++it)
                {
                    int seq_len = static_cast<int>(it->second.size());
                    if (seq_len > longest_match && seq_len <= i)
                    {
                        bool match = true;
                        for (int offset = 0; offset < seq_len; ++offset)
                        {
                            // reversed[i - offset - 1] = history token preceding the head
                            if (it->second[offset] != reversed[i - offset - 1])
                            {
                                match = false;
                                break;
                            }
                        }
                        if (match)
                            longest_match = seq_len;
                    }
                }
                if (longest_match >= 0)
                {
                    rep_limit = i - longest_match;
                    break;
                }
            }
        }
        if (rep_limit < params.dry_allowed_length)
            return;

        // Step 2: Z-algorithm on reversed history
        int n = static_cast<int>(reversed.size());
        std::vector<int> z(n, 0);
        int l = 0, r = 0;
        for (int i = 1; i < n; ++i)
        {
            if (i < r)
                z[i] = std::min(r - i, z[i - l]);
            while (i + z[i] < n && reversed[z[i]] == reversed[i + z[i]])
                z[i]++;
            if (i + z[i] > r)
            {
                l = i;
                r = i + z[i];
            }
        }

        // Step 3: Collect max repeat length per token that would extend a repeat
        std::unordered_map<int, int> dry_max_token_repeat;
        for (int i = 1; i < n; ++i)
        {
            int repeat_len = std::min(z[i], rep_limit);
            if (repeat_len < params.dry_allowed_length)
                continue;

            // The token that would extend the repeat = reversed[i-1] = history[n-i]
            int penalty_token = reversed[i - 1];

            if (penalty_token < 0 || penalty_token >= vocab_size)
                continue;

            auto it = dry_max_token_repeat.find(penalty_token);
            if (it == dry_max_token_repeat.end() || it->second < repeat_len)
                dry_max_token_repeat[penalty_token] = repeat_len;
        }

        // Step 4: Apply penalties, with overflow protection and breaker exemption
        // Prevent pow() overflow by clamping exponent
        static const float FLOAT_MAX_LOG = 88.7228391f;
        int max_exponent = 0;
        if (params.dry_base > 1.000001f)
            max_exponent = static_cast<int>(FLOAT_MAX_LOG / std::log(params.dry_base));

        for (const auto &[token, repeat_len] : dry_max_token_repeat)
        {
            // Exempt single-token sequence breakers from penalty (aligned with llama.cpp)
            if (dry_breakers_initialized_)
            {
                auto range = dry_breakers_.equal_range(token);
                bool is_single_token_breaker = false;
                for (auto it = range.first; it != range.second; ++it)
                {
                    if (it->second.empty())
                    {
                        is_single_token_breaker = true;
                        break;
                    }
                }
                if (is_single_token_breaker)
                    continue;
            }

            int repeat_exp = repeat_len - params.dry_allowed_length;
            if (max_exponent > 0 && repeat_exp > max_exponent)
                repeat_exp = max_exponent;

            float penalty = params.dry_multiplier *
                            std::pow(params.dry_base, static_cast<float>(repeat_exp));

            // Add DRY penalty on top of any existing presence/frequency penalty
            auto it = penalty_map.find(token);
            if (it == penalty_map.end())
                penalty_map[token] = penalty;
            else
                it->second += penalty;
        }
    }

    std::vector<LogitPenalty> Sampler::compute_penalty_map(const SamplingParams &params, int vocab_size)
    {
        std::unordered_map<int, float> penalty_map;

        // Presence + frequency penalties
        if (params.presence_penalty != 0.0f || params.frequency_penalty != 0.0f)
        {
            for (const auto &[token_id, count] : token_counts_)
            {
                if (token_id < 0 || token_id >= vocab_size)
                    continue;
                float p = 0.0f;
                if (params.presence_penalty != 0.0f)
                    p += params.presence_penalty;
                if (params.frequency_penalty != 0.0f)
                    p += params.frequency_penalty * static_cast<float>(count);
                if (p != 0.0f)
                    penalty_map[token_id] = p;
            }
        }

        // DRY penalty
        compute_dry_penalties(penalty_map, params, vocab_size);

        // Convert to sparse vector
        std::vector<LogitPenalty> result;
        result.reserve(penalty_map.size());
        for (const auto &[token_id, penalty] : penalty_map)
        {
            result.push_back({token_id, penalty});
        }
        return result;
    }

    void Sampler::apply_penalties(std::vector<float> &logits, const SamplingParams &params)
    {
        // Compute the full penalty map and apply it
        auto penalties = compute_penalty_map(params, static_cast<int>(logits.size()));
        for (const auto &entry : penalties)
        {
            logits[entry.token_id] -= entry.penalty;
        }
    }

} // namespace llaminar2
