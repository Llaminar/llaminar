/**
 * @file TextPreTokenizer.cpp
 * @brief Unicode-exact pre-tokenization using pinned PCRE2 Unicode tables.
 *
 * These expressions are the model-declared policies, applied before byte BPE.
 * Qwen35 additionally groups combining marks with letters; Llama3 groups at
 * most three digits. Whitespace lookahead preserves Python indentation.
 * Qwen's NFC normalization is length-aware and preserves embedded NULs,
 * emoji modifiers and joiners. UTF-8 is validated once per splitting subject,
 * not once per word of a long prompt.
 */
#include "TextPreTokenizer.h"

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <utf8proc.h>
#include <array>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace
    {
        /** @brief Return the upstream expression for an admitted typed policy. */
        std::string_view expression(TextPreTokenizerProfile profile)
        {
            // Source: openai/gpt-2 src/encoder.py, Meta llama3/llama/tokenizer.py,
            // and the official Qwen tokenizer.json files. Pinned reference
            // provenance and independent span/token oracles live in the tests.
            switch (profile)
            {
            case TextPreTokenizerProfile::GPT2:
                return R"('s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+)";
            case TextPreTokenizerProfile::Qwen2:
                return R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
            case TextPreTokenizerProfile::Qwen35:
                return R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
            case TextPreTokenizerProfile::Llama3:
                return R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
            case TextPreTokenizerProfile::DeepSeekV4:
                // Final stage of the DeepSeek-V4 sequence; earlier stages are deepSeekV4Stages().
                return R"([!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
            }
            throw std::invalid_argument("Invalid TextPreTokenizerProfile");
        }

        /**
         * @brief DeepSeek-V4 (tokenizer.json @ 6821d6ad) isolated split stages, in order.
         * @details Digits in runs of up to three, then CJK runs, then the
         * general expression returned by expression(DeepSeekV4).
         */
        const std::array<const char *, 2> &deepSeekV4Stages()
        {
            static const std::array<const char *, 2> stages{R"(\p{N}{1,3})", R"([一-龥぀-ゟ゠-ヿ]+)"};
            return stages;
        }

        /** @brief Preserve the regex library's precise compile/match diagnostic. */
        std::string errorMessage(int code)
        {
            std::array<PCRE2_UCHAR, 256> message{};
            if (pcre2_get_error_message(code, message.data(), message.size()) < 0)
                return "PCRE2 error " + std::to_string(code);
            return reinterpret_cast<const char*>(message.data());
        }
    }

    /** @brief RAII owner of the immutable compiled policy; matching state is never shared. */
    struct TextPreTokenizer::Pattern
    {
        TextPreTokenizerProfile profile;
        std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)> code{nullptr, pcre2_code_free};
        /// DeepSeekV4 only: earlier isolated-split stages, applied before @c code.
        std::vector<std::unique_ptr<pcre2_code, decltype(&pcre2_code_free)>> stages;
    };

    TextPreTokenizerProfile parseTextPreTokenizerProfile(std::string_view name)
    {
        if (name == "gpt-2") return TextPreTokenizerProfile::GPT2;
        if (name == "qwen2") return TextPreTokenizerProfile::Qwen2;
        if (name == "qwen35") return TextPreTokenizerProfile::Qwen35;
        if (name == "llama-bpe") return TextPreTokenizerProfile::Llama3;
        if (name == "joyai-llm") return TextPreTokenizerProfile::DeepSeekV4;
        throw std::invalid_argument("Unsupported tokenizer.ggml.pre='" + std::string(name)
                                    + "'; expected gpt-2, qwen2, qwen35, llama-bpe or joyai-llm");
    }

    TextPreTokenizer::TextPreTokenizer(TextPreTokenizerProfile profile)
        : pattern_(std::make_unique<Pattern>())
    {
        pattern_->profile = profile;
        const auto regex = expression(profile);
        int error = 0;
        PCRE2_SIZE offset = 0;
        pattern_->code.reset(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(regex.data()), regex.size(),
                                          PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr));
        if (!pattern_->code)
            throw std::runtime_error("Pre-tokenizer expression at byte " + std::to_string(offset)
                                     + ": " + errorMessage(error));
        if (profile == TextPreTokenizerProfile::DeepSeekV4)
            for (const char *stage : deepSeekV4Stages())
            {
                pattern_->stages.emplace_back(pcre2_compile(reinterpret_cast<PCRE2_SPTR>(stage), PCRE2_ZERO_TERMINATED,
                                                            PCRE2_UTF | PCRE2_UCP, &error, &offset, nullptr),
                                              pcre2_code_free);
                if (!pattern_->stages.back())
                    throw std::runtime_error("Pre-tokenizer stage expression at byte " + std::to_string(offset)
                                             + ": " + errorMessage(error));
            }
    }

    TextPreTokenizer::~TextPreTokenizer() = default;

    std::string TextPreTokenizer::normalize(std::string_view text) const
    {
        if (text.empty()) return {};
        if (pattern_->profile == TextPreTokenizerProfile::GPT2 ||
            pattern_->profile == TextPreTokenizerProfile::Llama3 ||
            pattern_->profile == TextPreTokenizerProfile::DeepSeekV4)
            return std::string(text);
        if (text.size() > static_cast<size_t>(std::numeric_limits<utf8proc_ssize_t>::max()))
            throw std::length_error("Pre-tokenizer input exceeds Unicode normalizer extent");
        utf8proc_uint8_t* output = nullptr;
        // Never use NULLTERM: JSON chat messages may contain U+0000 followed
        // by more live text. COMPOSE is NFC, without compatibility folding,
        // case folding or stripping of accents, selectors, joiners or controls.
        const auto length = utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(text.data()),
            static_cast<utf8proc_ssize_t>(text.size()), &output,
            static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
        const std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> owned(output, std::free);
        if (length < 0)
            throw std::runtime_error(std::string("Pre-tokenizer NFC failed: ") + utf8proc_errmsg(length));
        return std::string(reinterpret_cast<const char*>(owned.get()), static_cast<size_t>(length));
    }

    namespace
    {
        /**
         * @brief Hugging Face "Isolated" split of one piece: every non-overlapping
         *        match and every gap between matches becomes its own span.
         */
        void isolatedSplit(const pcre2_code *code, std::string_view piece, std::vector<std::string_view> &out)
        {
            const std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
                pcre2_match_data_create_from_pattern(code, nullptr), pcre2_match_data_free);
            if (!match) throw std::bad_alloc();
            PCRE2_SIZE offset = 0, last = 0;
            while (offset < piece.size())
            {
                const int count = pcre2_match(code, reinterpret_cast<PCRE2_SPTR>(piece.data()), piece.size(),
                                              offset, offset == 0 ? 0 : PCRE2_NO_UTF_CHECK, match.get(), nullptr);
                if (count == PCRE2_ERROR_NOMATCH) break;
                if (count <= 0)
                    throw std::runtime_error("Pre-tokenization stage failed: " + errorMessage(count));
                const auto *range = pcre2_get_ovector_pointer(match.get());
                if (range[1] <= range[0]) // empty match: advance one code point
                {
                    offset = range[0] + 1;
                    while (offset < piece.size() && (static_cast<unsigned char>(piece[offset]) & 0xC0) == 0x80) ++offset;
                    continue;
                }
                if (range[0] > last) out.push_back(piece.substr(last, range[0] - last));
                out.push_back(piece.substr(range[0], range[1] - range[0]));
                last = offset = range[1];
            }
            if (last < piece.size()) out.push_back(piece.substr(last));
        }
    }

    std::vector<std::string_view> TextPreTokenizer::split(std::string_view text) const
    {
        std::vector<std::string_view> spans;
        if (text.empty()) return spans;
        if (pattern_->profile == TextPreTokenizerProfile::DeepSeekV4)
        {
            std::vector<std::string_view> pieces{text};
            for (const auto &stage : pattern_->stages)
            {
                std::vector<std::string_view> next;
                for (const auto piece : pieces) isolatedSplit(stage.get(), piece, next);
                pieces = std::move(next);
            }
            for (const auto piece : pieces) isolatedSplit(pattern_->code.get(), piece, spans);
            return spans;
        }
        const std::unique_ptr<pcre2_match_data, decltype(&pcre2_match_data_free)> match(
            pcre2_match_data_create_from_pattern(pattern_->code.get(), nullptr), pcre2_match_data_free);
        if (!match) throw std::bad_alloc();
        PCRE2_SIZE offset = 0;
        while (offset < text.size())
        {
            // The first match validates the complete subject. Subsequent calls
            // borrow that same immutable subject at proven UTF-8 boundaries.
            const uint32_t options = PCRE2_ANCHORED | (offset == 0 ? 0 : PCRE2_NO_UTF_CHECK);
            const int count = pcre2_match(pattern_->code.get(),
                reinterpret_cast<PCRE2_SPTR>(text.data()), text.size(), offset, options, match.get(), nullptr);
            if (count <= 0)
                throw std::runtime_error("Pre-tokenization failed at byte " + std::to_string(offset)
                                         + ": " + errorMessage(count));
            const auto* range = pcre2_get_ovector_pointer(match.get());
            if (range[0] != offset || range[1] <= offset || range[1] > text.size())
                throw std::runtime_error("Pre-tokenization did not cover the next input bytes");
            spans.push_back(text.substr(offset, range[1] - offset));
            offset = range[1];
        }
        return spans;
    }
}
