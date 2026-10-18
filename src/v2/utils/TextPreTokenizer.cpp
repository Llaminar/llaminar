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
            }
            throw std::invalid_argument("Invalid TextPreTokenizerProfile");
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
    };

    TextPreTokenizerProfile parseTextPreTokenizerProfile(std::string_view name)
    {
        if (name == "gpt-2") return TextPreTokenizerProfile::GPT2;
        if (name == "qwen2") return TextPreTokenizerProfile::Qwen2;
        if (name == "qwen35") return TextPreTokenizerProfile::Qwen35;
        if (name == "llama-bpe") return TextPreTokenizerProfile::Llama3;
        throw std::invalid_argument("Unsupported tokenizer.ggml.pre='" + std::string(name)
                                    + "'; expected gpt-2, qwen2, qwen35 or llama-bpe");
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
    }

    TextPreTokenizer::~TextPreTokenizer() = default;

    std::string TextPreTokenizer::normalize(std::string_view text) const
    {
        if (text.empty()) return {};
        if (pattern_->profile == TextPreTokenizerProfile::GPT2 ||
            pattern_->profile == TextPreTokenizerProfile::Llama3)
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

    std::vector<std::string_view> TextPreTokenizer::split(std::string_view text) const
    {
        std::vector<std::string_view> spans;
        if (text.empty()) return spans;
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
