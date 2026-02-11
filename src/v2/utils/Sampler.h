/**
 * @file Sampler.h
 * @brief Token sampling strategies for LLM text generation
 * @author David Sanftenberg
 * @date 2025
 *
 * Provides various sampling methods for selecting next tokens from logits:
 * - Greedy sampling (argmax)
 * - Temperature scaling
 * - Top-k sampling
 * - Top-p (nucleus) sampling
 *
 * Used during autoregressive decode phase to generate text token by token.
 */

#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace llaminar2
{

    /**
     * @brief Parameters for sampling configuration
     */
    struct SamplingParams
    {
        float temperature = 1.0f; ///< Temperature for scaling logits (0.0 = greedy, >1.0 = more random)
        int top_k = 0;            ///< Top-k filtering (0 = disabled, >0 = keep only top-k tokens)
        float top_p = 1.0f;       ///< Top-p (nucleus) filtering (1.0 = disabled, <1.0 = cumulative prob threshold)
        unsigned int seed = 0;    ///< Random seed (0 = random, >0 = deterministic)

        /**
         * @brief Check if sampling is greedy (deterministic argmax)
         */
        bool is_greedy() const
        {
            return temperature == 0.0f || (top_k == 1 && top_p >= 1.0f);
        }
    };

    /**
     * @brief Token sampler for LLM text generation
     *
     * Supports multiple sampling strategies:
     * - Greedy: Always select highest probability token
     * - Temperature: Scale logits before softmax to control randomness
     * - Top-k: Only consider top-k highest probability tokens
     * - Top-p: Only consider tokens with cumulative probability <= p
     *
     * Example usage:
     * @code
     * Sampler sampler;
     * SamplingParams params;
     * params.temperature = 0.8f;
     * params.top_k = 40;
     * params.top_p = 0.95f;
     *
     * std::vector<float> logits = pipeline->get_logits();
     * int next_token = sampler.sample(logits, params);
     * @endcode
     */
    class Sampler
    {
    public:
        /**
         * @brief Construct sampler with optional seed
         * @param seed Random seed (0 = use random device, >0 = deterministic)
         */
        explicit Sampler(unsigned int seed = 0);

        /**
         * @brief Sample next token from logits
         *
         * Applies temperature scaling, top-k/top-p filtering, then samples.
         *
         * @param logits Raw logits from model (vocab_size elements)
         * @param params Sampling parameters
         * @return Token ID (index into vocabulary)
         */
        int sample(const std::vector<float> &logits, const SamplingParams &params);

        /**
         * @brief Sample next token from raw logits pointer (zero-copy)
         *
         * Avoids vector allocation for greedy path. For non-greedy sampling,
         * constructs a vector internally only when needed.
         *
         * @param logits Raw logits pointer from model
         * @param vocab_size Number of logit values
         * @param params Sampling parameters
         * @return Token ID (index into vocabulary)
         */
        int sample(const float *logits, size_t vocab_size, const SamplingParams &params);

        /**
         * @brief Greedy sampling (always select argmax)
         *
         * Deterministic: Always returns token with highest logit value.
         *
         * @param logits Raw logits from model
         * @return Token ID with highest logit
         */
        int sample_greedy(const std::vector<float> &logits);

        /**
         * @brief Greedy sampling from raw pointer (zero-copy)
         *
         * Deterministic argmax directly on raw logits pointer.
         * Avoids the ~600KB heap allocation per token for typical LLM vocab sizes.
         *
         * @param logits Raw logits pointer
         * @param vocab_size Number of logit values
         * @return Token ID with highest logit
         */
        int sample_greedy(const float *logits, size_t vocab_size);

        /**
         * @brief Sample with temperature scaling
         *
         * Logits are divided by temperature before converting to probabilities:
         * - temperature = 0.0: Greedy (argmax)
         * - temperature = 1.0: No scaling (standard softmax)
         * - temperature > 1.0: More random (flatter distribution)
         * - temperature < 1.0: Less random (sharper distribution)
         *
         * @param logits Raw logits from model
         * @param temperature Temperature scaling factor
         * @return Sampled token ID
         */
        int sample_temperature(const std::vector<float> &logits, float temperature);

        /**
         * @brief Top-k sampling
         *
         * Only consider the k tokens with highest probability.
         * Renormalizes probabilities over top-k candidates and samples.
         *
         * @param logits Raw logits from model
         * @param k Number of top tokens to consider
         * @param temperature Temperature scaling (applied before top-k filtering)
         * @return Sampled token ID from top-k candidates
         */
        int sample_top_k(const std::vector<float> &logits, int k, float temperature = 1.0f);

        /**
         * @brief Top-p (nucleus) sampling
         *
         * Select tokens with cumulative probability <= p.
         * Dynamically adjusts number of candidates based on probability distribution.
         *
         * @param logits Raw logits from model
         * @param p Cumulative probability threshold (0.0-1.0)
         * @param temperature Temperature scaling (applied before top-p filtering)
         * @return Sampled token ID from nucleus
         */
        int sample_top_p(const std::vector<float> &logits, float p, float temperature = 1.0f);

        /**
         * @brief Set random seed for reproducibility
         * @param seed Random seed (0 = use random device)
         */
        void set_seed(unsigned int seed);

    private:
        std::mt19937 rng_; ///< Random number generator

        /**
         * @brief Apply temperature scaling to logits
         * @param logits Input logits
         * @param temperature Scaling factor
         * @return Temperature-scaled logits
         */
        std::vector<float> apply_temperature(const std::vector<float> &logits, float temperature);

        /**
         * @brief Convert logits to probabilities via softmax
         * @param logits Input logits
         * @return Probability distribution (sums to 1.0)
         */
        std::vector<float> softmax(const std::vector<float> &logits);

        /**
         * @brief Sample token index from probability distribution
         * @param probs Probability distribution
         * @return Sampled index
         */
        int sample_from_probs(const std::vector<float> &probs);
    };

} // namespace llaminar2
