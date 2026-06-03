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
#include <unordered_map>
#include <functional>
#include <string>

namespace llaminar2
{

    /**
     * @brief A single logit penalty entry for GPU-side penalty application
     *
     * Represents an additive penalty to subtract from a token's logit.
     * Used to upload a sparse penalty map to the GPU, avoiding a full D2H
     * transfer of the logits tensor (~600KB) just to apply penalties.
     */
    struct LogitPenalty
    {
        int32_t token_id; ///< Token to penalize
        float penalty;    ///< Amount to subtract from logit (positive = penalize)
    };

    struct SamplingDistributionEntry
    {
        int token_id = -1;
        float probability = 0.0f;
    };

    /**
     * @brief Parameters for sampling configuration
     */
    struct SamplingParams
    {
        float temperature = 1.0f; ///< Temperature for scaling logits (0.0 = greedy, >1.0 = more random)
        int top_k = 0;            ///< Top-k filtering (0 = disabled, >0 = keep only top-k tokens)
        float top_p = 1.0f;       ///< Top-p (nucleus) filtering (1.0 = disabled, <1.0 = cumulative prob threshold)
        unsigned int seed = 0;    ///< Random seed (0 = random, >0 = deterministic)

        // Repetition penalty parameters (applied before softmax)
        float presence_penalty = 0.0f;  ///< Penalize tokens that appeared at all (OpenAI-style, additive)
        float frequency_penalty = 0.0f; ///< Penalize tokens proportional to frequency (OpenAI-style, additive)

        // DRY (Don't Repeat Yourself) penalty parameters
        // Detects repeated N-gram patterns and penalizes tokens that would extend them.
        // Algorithm: reverse Z-algorithm on token history to find suffix matches.
        // Penalty: multiplier * base^(repeat_len - allowed_length)
        float dry_multiplier = 0.0f;    ///< DRY penalty strength (0 = disabled)
        float dry_base = 1.75f;         ///< Exponential base for penalty scaling
        int dry_allowed_length = 2;     ///< Repeats up to this length are ignored
        int dry_penalty_last_n = -1;    ///< Token window to scan (-1 = full context, 0 = disabled)
        std::vector<std::string> dry_sequence_breakers = {"\n", ":", "\"", "*"}; ///< Reset repeat detection

        /**
         * @brief Check if sampling is greedy (deterministic argmax)
         */
        bool is_greedy() const
        {
            return temperature == 0.0f || (top_k == 1 && top_p >= 1.0f);
        }

        /**
         * @brief Check if any repetition penalties are active
         */
        bool has_penalties() const
        {
            return presence_penalty != 0.0f || frequency_penalty != 0.0f ||
                   (dry_multiplier != 0.0f && dry_penalty_last_n != 0);
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
         * @brief Build the normalized sampling distribution for the current history.
         *
         * Applies additive penalties, temperature, top-k, and top-p. When both
         * top_k and top_p are set, top_p is applied within the top-k candidate
         * set; this matches common chat defaults and the GPU sampler path.
         */
        std::vector<SamplingDistributionEntry> compute_distribution(
            const float *logits,
            size_t vocab_size,
            const SamplingParams &params);

        int sample_from_distribution(const std::vector<SamplingDistributionEntry> &distribution);
        int sample_from_residual_distribution(
            const std::vector<SamplingDistributionEntry> &target,
            const std::vector<SamplingDistributionEntry> &draft);
        static std::vector<SamplingDistributionEntry> residual_distribution(
            const std::vector<SamplingDistributionEntry> &target,
            const std::vector<SamplingDistributionEntry> &draft);
        static float speculative_accept_probability(
            float target_probability,
            float draft_probability);
        static float probability_of_token(
            const std::vector<SamplingDistributionEntry> &distribution,
            int token_id);
        float random_uniform_01();

        /**
         * @brief Set random seed for reproducibility
         * @param seed Random seed (0 = use random device)
         */
        void set_seed(unsigned int seed);

        /**
         * @brief Record a token as having been generated (for penalty tracking)
         *
         * Must be called after each token is sampled so that presence/frequency
         * penalties can be applied in subsequent samples.
         *
         * @param token_id The generated token ID
         */
        void record_token(int token_id);

        /**
         * @brief Reset token generation history (e.g., new conversation)
         */
        void reset_history();

        /**
         * @brief Initialize DRY sequence breakers from string patterns
         *
         * Tokenizes each breaker string and builds a multimap from head tokens
         * to tail token sequences. Must be called before DRY penalties are used.
         *
         * @param breaker_strings Strings that reset repeat detection (e.g., "\n", ":")
         * @param tokenize Function that converts a string to token IDs
         */
        using TokenizeFunc = std::function<std::vector<int>(const std::string &)>;
        void initDryBreakers(const std::vector<std::string> &breaker_strings,
                             const TokenizeFunc &tokenize);

        /**
         * @brief Compute a sparse penalty map for GPU-side penalty application
         *
         * Combines presence, frequency, and DRY penalties into a single sparse
         * map of (token_id, penalty) entries. Upload this to the GPU to apply
         * penalties without a full D2H of the logits tensor.
         *
         * @param params Sampling parameters with penalty configuration
         * @param vocab_size Vocabulary size (for bounds checking)
         * @return Vector of LogitPenalty entries to subtract from logits
         */
        std::vector<LogitPenalty> compute_penalty_map(const SamplingParams &params, int vocab_size);

    private:
        std::mt19937 rng_; ///< Random number generator
        std::unordered_map<int, int> token_counts_; ///< Token ID → generation count

        // DRY state
        std::vector<int> dry_token_history_;  ///< Ring buffer of recent token IDs
        size_t dry_history_capacity_ = 0;     ///< Max tokens to keep (dry_penalty_last_n)
        std::unordered_multimap<int, std::vector<int>> dry_breakers_; ///< Head token → tail sequences
        bool dry_breakers_initialized_ = false;

        /**
         * @brief Apply DRY penalty to a penalty map
         *
         * Uses the reverse Z-algorithm on token history to find repeated N-gram
         * patterns and compute penalties for tokens that would extend them.
         *
         * @param penalty_map Map to merge DRY penalties into
         * @param params Sampling parameters with DRY configuration
         * @param vocab_size Vocabulary size (for bounds checking)
         */
        void compute_dry_penalties(std::unordered_map<int, float> &penalty_map,
                                   const SamplingParams &params, int vocab_size);

        /**
         * @brief Apply presence and frequency penalties to logits (in-place)
         *
         * OpenAI-style penalties applied before temperature scaling:
         * logit[token] -= presence_penalty * (1 if token appeared) + frequency_penalty * count
         *
         * @param logits Logits to modify in-place
         * @param params Sampling parameters with penalty values
         */
        void apply_penalties(std::vector<float> &logits, const SamplingParams &params);

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
