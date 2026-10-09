/**
 * @file TextPreTokenizer.h
 * @brief Metadata-selected Unicode boundaries for byte-level BPE.
 *
 * Each policy owns normalization as well as its permitted merge boundaries.
 * Qwen requires NFC; GPT-2 and Llama3 preserve the supplied Unicode spelling.
 * BPE may merge bytes only within the spans declared by the model's tokenizer.
 * An immutable compiled pattern is shared across calls; each call owns its
 * matching state and borrows its input, so concurrent requests cannot race.
 */
#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace llaminar2
{
    /** @brief Supported, distinct GGUF byte-BPE pre-tokenization contracts. */
    enum class TextPreTokenizerProfile { GPT2, Qwen2, Qwen35, Llama3 };

    /**
     * @brief Resolve an exact tokenizer.ggml.pre value without guessing a model family.
     * @param name Declared GGUF pre-tokenizer identifier.
     * @return The corresponding Unicode splitting policy.
     * @throws std::invalid_argument If the identifier is missing or unsupported.
     */
    TextPreTokenizerProfile parseTextPreTokenizerProfile(std::string_view name);

    /** @brief Immutable Unicode splitter with invocation-local matching storage. */
    class TextPreTokenizer final
    {
    public:
        /**
         * @brief Compile the selected model's exact Unicode expression once.
         * @param profile The admitted model policy.
         * @throws std::runtime_error If the expression cannot be compiled.
         */
        explicit TextPreTokenizer(TextPreTokenizerProfile profile);
        /** @brief Release the owned compiled expression. */
        ~TextPreTokenizer();
        TextPreTokenizer(const TextPreTokenizer&) = delete;
        TextPreTokenizer& operator=(const TextPreTokenizer&) = delete;

        /**
         * @brief Apply the admitted policy's Unicode normalization before splitting.
         * @param text Ordinary UTF-8 text; embedded NULs are part of the input.
         * @return NFC for Qwen policies, unchanged bytes for GPT-2 and Llama3.
         * @throws std::runtime_error If Unicode normalization fails.
         */
        std::string normalize(std::string_view text) const;

        /**
         * @brief Partition UTF-8 into contiguous, nonempty byte ranges for BPE.
         * @param text Valid UTF-8 whose storage must outlive the returned views.
         * @return Ordered views covering every input byte exactly once.
         * @throws std::runtime_error On invalid UTF-8 or an incomplete match.
         */
        std::vector<std::string_view> split(std::string_view text) const;

    private:
        struct Pattern;
        std::unique_ptr<Pattern> pattern_;
    };
}
