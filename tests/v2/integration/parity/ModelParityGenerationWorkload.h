/**
 * @file ModelParityGenerationWorkload.h
 * @brief Typed continuous-generation budget, model-owned input and sampling seed.
 *
 * A long generation can expose delayed token drift from a tiny batch-invariance
 * violation. It cannot establish a bound on KL divergence or replace the deep
 * grouped-versus-serial arithmetic proof. MTP and its serial control inherit
 * this same workload from their model/topology definition, not from a runner's
 * interpretation of a test name. Short requests may not be summed to satisfy it.
 */
#pragma once

#include <stdexcept>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace llaminar2::test::parity
{
    /** Immutable request ordering; prefix restore is mandatory, not another axis. */
    enum class GenerationPrefixProbe { Fresh, FullRestore, PartialRestore };

    /** A fresh story and a continuation that retains its complete cached prompt. */
    enum class GenerationNarrative { Harbor, Mountain };

    /** One request and the serial baseline it must match, including repeat hits. */
    struct GenerationProbe
    {
        std::string_view id;
        GenerationPrefixProbe prefix;
        GenerationNarrative narrative;
    };

    /**
     * @brief Positive model-owned seed within the public sampler's uint32 ABI.
     *
     * Seed zero selects nondeterministic sampling and cannot establish a stable
     * serial control. Larger integers must not silently narrow at the HTTP
     * boundary. Initial acquisition may deliberately choose another seed, but
     * every topology, KV format and MTP policy for that model inherits it.
     */
    class ModelParityGenerationSeed
    {
    public:
        /**
         * @param seed Exact positive public request seed; defaults preserve existing workloads.
         * @throws std::invalid_argument for zero, negative or non-uint32 values.
         */
        explicit ModelParityGenerationSeed(std::int64_t seed = 4242)
        {
            if (seed <= 0 || seed > std::numeric_limits<std::uint32_t>::max())
                throw std::invalid_argument("generation requires a positive uint32 sampling seed");
            seed_ = static_cast<std::uint32_t>(seed);
        }

        /** @return Immutable request seed, not a topology or runner override. */
        [[nodiscard]] std::uint32_t value() const noexcept { return seed_; }
        /** @return Whether two model declarations select the same sampled workload. */
        friend bool operator==(const ModelParityGenerationSeed &, const ModelParityGenerationSeed &) = default;

    private:
        std::uint32_t seed_;
    };

    /**
     * @brief Validated per-request budget and minimum observed output length.
     *
     * Every case requires at least 384 committed tokens in one request. The
     * default requests exactly that many to bound routine regression cost;
     * definitions may ask for longer evidence, never a shorter substitute.
     */
    class ModelParityGenerationWorkload
    {
    public:
        /** Minimum continuous horizon shared by MTP and MTP-off controls. */
        static constexpr int kMinimumCompletionTokens = 384;

        /** All generated cells exercise the same fresh/full/partial lifecycle. */
        static constexpr std::array<GenerationProbe, 4> kProbes{{
            {"fresh_harbor", GenerationPrefixProbe::Fresh, GenerationNarrative::Harbor},
            {"full_harbor", GenerationPrefixProbe::FullRestore, GenerationNarrative::Harbor},
            {"partial_mountain", GenerationPrefixProbe::PartialRestore, GenerationNarrative::Mountain},
            {"full_mountain", GenerationPrefixProbe::FullRestore, GenerationNarrative::Mountain},
        }};

        /**
         * @return Shared instructions; the completed seed prompt owns the cache boundary.
         *
         * Exact bytes are workload identity. This is a long-generation task,
         * independent of each model's deliberately short HF checkpoint prompt.
         */
        [[nodiscard]] static std::string_view systemPrompt() noexcept
        {
            return "You are a novelist writing a long, detailed story. Write at least two thousand words. "
                "Begin directly with the story and continue the scene without a concluding summary. "
                "Use clear sentences, concrete physical details, dialogue, and a consistent chronology. "
                "Let the characters notice small changes in their surroundings and explain what they do next. "
                "Keep the same names and places throughout. Describe the objects they carry, the sounds they hear, "
                "the weather, the route they follow, and the choices they make. Introduce discoveries gradually "
                "instead of rushing to an ending. Each paragraph should move the story forward while preserving "
                "the facts established in previous paragraphs. Do not discuss these instructions or outline a plan. "
                "Do not write a title, a word count, or a list. The reader should be able to follow the journey "
                "from one location to the next and understand why each character acts. Keep writing the story "
                "in ordinary prose for the requested length.";
        }

        /** @return Initial user turn, retained verbatim by the continuation probe. */
        [[nodiscard]] static std::string_view userPrompt() noexcept
        {
            return "Write about a lighthouse keeper discovering an unusual map in an old harbor.";
        }

        /**
         * @return Fixed assistant history continuing the seed, not a sampled answer.
         *
         * Retaining the complete prior prompt gives hybrid models a restorable
         * terminal recurrent-state checkpoint. Merely changing a user suffix
         * shares some KV blocks but can leave no recurrent-state checkpoint at
         * that shorter boundary. The HTTP observer verifies the actual token
         * prefix too: chat-template differences must fail rather than fake a hit.
         */
        [[nodiscard]] static std::string_view assistantContinuation() noexcept
        {
            return "The keeper unfolded the map on the harbor wall. A narrow line crossed the water "
                "and climbed into the mountains, ending at an abandoned observatory. Several days later, "
                "a mountain guide named Mira reached its locked gate with the keeper's map in her coat. "
                "She heard footsteps inside and called out to whoever was waiting beyond the door.";
        }

        /**
         * @param maximum_tokens Public HTTP response token budget.
         * @param minimum_tokens Actual committed tokens required from each response.
         * @param readiness_seconds Startup budget within the exact-cell watchdog.
         * @throws std::invalid_argument if the horizon is too short or unachievable.
         */
        explicit ModelParityGenerationWorkload(
            int maximum_tokens = kMinimumCompletionTokens,
            int minimum_tokens = kMinimumCompletionTokens,
            int readiness_seconds = 60)
            : maximum_tokens_(maximum_tokens), minimum_tokens_(minimum_tokens),
              readiness_seconds_(readiness_seconds)
        {
            if (minimum_tokens < kMinimumCompletionTokens || maximum_tokens < minimum_tokens)
                throw std::invalid_argument(
                    "generation regression requires at least 384 continuous committed tokens within its response budget");
            if (readiness_seconds <= 0 || readiness_seconds > 600)
                throw std::invalid_argument("generation readiness must fit the 600-second exact-cell watchdog");
        }

        /** @return Response limit, not evidence that this many tokens were generated. */
        [[nodiscard]] int maximumTokens() const noexcept { return maximum_tokens_; }

        /** @return Minimum actual output count; early EOS below this is insufficient. */
        [[nodiscard]] int minimumTokens() const noexcept { return minimum_tokens_; }

        /** @return Model/topology-owned startup allowance, never a name heuristic. */
        [[nodiscard]] int readinessSeconds() const noexcept { return readiness_seconds_; }

        /** @return A validated startup override without duplicating horizon defaults. */
        [[nodiscard]] ModelParityGenerationWorkload withReadiness(int seconds) const
        {
            return ModelParityGenerationWorkload(maximum_tokens_, minimum_tokens_, seconds);
        }

        /** @return Whether serial and speculative cells share an identical horizon. */
        friend bool operator==(const ModelParityGenerationWorkload &,
                               const ModelParityGenerationWorkload &) = default;

    private:
        int maximum_tokens_;
        int minimum_tokens_;
        int readiness_seconds_;
    };
    /**
     * @brief Model-owned text for the common fresh/full/partial request protocol.
     *
     * A small model may naturally finish a particular continuation too early
     * to exercise the required horizon. Initial corpus acquisition can choose
     * suitable long-form text here without changing token budgets, sampling,
     * prefix semantics or an execution policy. Exact exported message bytes
     * remain control identity; an approved stream must never be silently retried
     * with different text. Backend and topology do not select these prompts.
     */
    class ModelParityGenerationPrompt
    {
    public:
        /**
         * @param system Long-form writing instructions, retained across requests.
         * @param user Initial user turn that owns the cached prompt boundary.
         * @param continuation Fixed assistant text extending that whole prompt.
         * @param followup Optional next user request, where the model's chat
         *                 template preserves the complete cached token prefix.
         * @throws std::invalid_argument if any required message is empty.
         */
        explicit ModelParityGenerationPrompt(
            std::string system = std::string(ModelParityGenerationWorkload::systemPrompt()),
            std::string user = std::string(ModelParityGenerationWorkload::userPrompt()),
            std::string continuation = std::string(ModelParityGenerationWorkload::assistantContinuation()),
            std::optional<std::string> followup = std::nullopt)
            : system_(std::move(system)), user_(std::move(user)), continuation_(std::move(continuation)),
              followup_(std::move(followup))
        {
            if (system_.empty() || user_.empty() || continuation_.empty())
                throw std::invalid_argument("generation requires nonempty system, user and continuation text");
            if (followup_ && followup_->empty())
                throw std::invalid_argument("a supplied generation follow-up user turn must be nonempty");
        }

        /** @return Exact model-owned system message, not a harness override. */
        [[nodiscard]] const std::string &system() const noexcept { return system_; }
        /** @return Exact initial user message shared by every probe. */
        [[nodiscard]] const std::string &user() const noexcept { return user_; }
        /** @return Fixed, closed assistant message; never sampled output or an open completion prefix. */
        [[nodiscard]] const std::string &continuation() const noexcept { return continuation_; }
        /**
         * @return Fixed model-owned next user turn, or no additional turn.
         *
         * This is prompt identity, never a retry or an output-length workaround
         * chosen by a runner. The HTTP proof must authenticate the actual token
         * prefix; identical earlier messages alone cannot establish that fact.
         */
        [[nodiscard]] const std::optional<std::string> &followup() const noexcept { return followup_; }
        /** @return Whether the complete prompt identity is unchanged. */
        friend bool operator==(const ModelParityGenerationPrompt &, const ModelParityGenerationPrompt &) = default;

    private:
        std::string system_;
        std::string user_;
        std::string continuation_;
        std::optional<std::string> followup_;
    };
} // namespace llaminar2::test::parity
