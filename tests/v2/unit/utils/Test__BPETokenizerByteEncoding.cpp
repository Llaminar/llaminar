/**
 * @file Test__BPETokenizerByteEncoding.cpp
 * @brief Model-free production tokenizer regressions against independent HF oracles.
 *
 * Small vocabulary projections and golden spans/IDs are frozen from upstream
 * tokenizer policies, with source hashes in the fixture. Tests exercise real
 * GGUF admission and BPE, never a copied byte encoder. Indentation can round-
 * trip with incorrect IDs, so both exact IDs and decoded bytes are asserted.
 */
#include "utils/Tokenizer.h"
#include "loaders/ModelLoader.h"
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <future>

namespace
{
    using namespace llaminar2;
    using nlohmann::json;

    /** @brief Read the small independent oracle once without models or network. */
    const json& fixture()
    {
        static const json value = [] {
            std::ifstream stream(LLAMINAR_BPE_FIXTURE_PATH);
            if (!stream) throw std::runtime_error("Missing frozen byte-BPE reference fixture");
            return json::parse(stream);
        }();
        return value;
    }

    /** @brief Store GGUF's length-prefixed UTF-8 bytes, without a terminator. */
    GGUFValue stringValue(const std::string& value)
    {
        const uint64_t length = value.size();
        GGUFValue result{.type = GGUFValueType::STRING,
                         .data = std::vector<uint8_t>(sizeof(length) + length)};
        std::memcpy(result.data.data(), &length, sizeof(length));
        std::memcpy(result.data.data() + sizeof(length), value.data(), length);
        return result;
    }

    /** @brief Store arrays in the same representation as the GGUF loader. */
    GGUFValue arrayValue(const std::vector<std::string>& value)
    {
        return {.type = GGUFValueType::ARRAY, .array_length = value.size(), .string_array_value = value};
    }

    /** @brief Build complete metadata including authoritative added-token types. */
    std::map<std::string, GGUFValue> metadata(const std::string& profile)
    {
        const auto& vocab = fixture()["vocabularies"][fixture()["profiles"][profile]["vocabulary"].get<std::string>()];
        auto tokens = vocab["tokens"].get<std::vector<std::string>>();
        tokens.insert(tokens.end(), {"<|im_start|>", "<|im_end|>", "<think>", "</think>"});
        std::vector<uint32_t> types(tokens.size(), 1);
        std::fill(types.end() - 4, types.end(), 3);
        GGUFValue typeValue{.type = GGUFValueType::ARRAY,
                            .data = std::vector<uint8_t>(types.size() * sizeof(uint32_t)),
                            .array_length = types.size()};
        std::memcpy(typeValue.data.data(), types.data(), typeValue.data.size());
        return {{"tokenizer.ggml.model", stringValue("gpt2")},
                {"tokenizer.ggml.pre", stringValue(profile)},
                {"tokenizer.ggml.tokens", arrayValue(tokens)},
                {"tokenizer.ggml.merges", arrayValue(vocab["merges"].get<std::vector<std::string>>())},
                {"tokenizer.ggml.token_type", std::move(typeValue)}};
    }

    TEST(BPETokenizer, DeclaredUnicodeBoundariesMatchIndependentReference)
    {
        for (const auto& [name, reference] : fixture()["profiles"].items())
        {
            TextPreTokenizer splitter(parseTextPreTokenizerProfile(name));
            for (size_t i = 0; i < fixture()["inputs"].size(); ++i)
            {
                SCOPED_TRACE(name + " case " + std::to_string(i));
                const auto text = fixture()["inputs"][i].get<std::string>();
                const auto spans = splitter.split(text);
                std::vector<std::string> actual;
                size_t offset = 0;
                for (const auto span : spans)
                {
                    EXPECT_FALSE(span.empty());
                    EXPECT_EQ(span.data(), text.data() + offset);
                    offset += span.size();
                    actual.emplace_back(span);
                }
                EXPECT_EQ(offset, text.size());
                EXPECT_EQ(actual, reference["cases"][i]["spans"].get<std::vector<std::string>>());
            }
        }
    }

    TEST(BPETokenizer, IndentationIdentifiersAndEmojiHaveExactReferenceTokenIds)
    {
        for (const auto& [name, reference] : fixture()["profiles"].items())
        {
            const auto tokenizer = BPETokenizer::create(metadata(name));
            ASSERT_NE(tokenizer, nullptr);
            for (size_t i = 0; i < fixture()["inputs"].size(); ++i)
            {
                SCOPED_TRACE(name + " case " + std::to_string(i));
                const auto text = fixture()["inputs"][i].get<std::string>();
                const auto ids = tokenizer->encode(text, false, false);
                EXPECT_EQ(ids, reference["cases"][i]["ids"].get<std::vector<int>>());
                EXPECT_EQ(tokenizer->decode(ids, false), reference["cases"][i]["normalized"].get<std::string>());
            }
        }
    }

    TEST(BPETokenizer, SpecialTokensDelimitOrdinaryUnicodeWithoutChangingIds)
    {
        const auto tokenizer = BPETokenizer::create(metadata("qwen35"));
        ASSERT_NE(tokenizer, nullptr);
        const std::string text = "<|im_start|>user\n👩🏽‍💻 café\n<|im_end|><think>    def f():\n</think>";
        std::vector<int> expected{1000};
        for (const auto& [part, terminator] : std::vector<std::pair<std::string, int>>{
                 {"user\n👩🏽‍💻 café\n", 1001}, {"", 1002}, {"    def f():\n", 1003}})
        {
            const auto ids = tokenizer->encode(part, false, false);
            expected.insert(expected.end(), ids.begin(), ids.end());
            expected.push_back(terminator);
        }
        EXPECT_EQ(tokenizer->encode(text, false, false), expected);
        EXPECT_EQ(tokenizer->decode(expected, false),
                  "<|im_start|>user\n👩🏽‍💻 café\n<|im_end|><think>    def f():\n</think>");
    }

    TEST(BPETokenizer, UnicodeNFCUsesTheDeclaredProfileAndPreservesEmojiAndNuls)
    {
        const std::vector<std::pair<std::string, std::string>> examples{
            {"e\u0301 cafe\u0301", "é café"},
            {"\u212a \u212b \u1100\u1161\u11a8", "K Å 각"},
            {"🙂 👩🏽‍💻 🇬🇧 ❤️ 1️⃣ 🚀", "🙂 👩🏽‍💻 🇬🇧 ❤️ 1️⃣ 🚀"},
            {std::string("e\u0301\0suffix", 10), std::string("é\0suffix", 9)},
            {"ﬁ ① Ａ", "ﬁ ① Ａ"}}; // NFC must not become compatibility normalization.
        for (const auto name : {"qwen2", "qwen35", "gpt-2", "llama-bpe"})
        {
            const auto tokenizer = BPETokenizer::create(metadata(name));
            ASSERT_NE(tokenizer, nullptr);
            for (const auto& [original, normalized] : examples)
            {
                SCOPED_TRACE(name);
                const auto ids = tokenizer->encode(original, false, false);
                const auto& expected = std::string_view(name).starts_with("qwen") ? normalized : original;
                EXPECT_EQ(tokenizer->decode(ids, false), expected);
                EXPECT_EQ(ids, tokenizer->encode(expected, false, false));
            }
        }
    }

    TEST(BPETokenizer, VocabularyWithoutAddedTokensStillHonorsBoundaries)
    {
        auto values = metadata("qwen35");
        // All entries are ordinary, including the four fixture spellings.
        auto& types = values["tokenizer.ggml.token_type"].data;
        const uint32_t ordinary = 1;
        for (size_t offset = 0; offset < types.size(); offset += sizeof(ordinary))
            std::memcpy(types.data() + offset, &ordinary, sizeof(ordinary));
        const auto tokenizer = BPETokenizer::create(values);
        ASSERT_NE(tokenizer, nullptr);
        const auto text = fixture()["inputs"][2].get<std::string>();
        EXPECT_EQ(tokenizer->encode(text, false, false),
                  fixture()["profiles"]["qwen35"]["cases"][2]["ids"].get<std::vector<int>>());
    }

    TEST(BPETokenizer, AdmissionRejectsUnknownMissingAndWrongTypedPolicies)
    {
        for (const auto key : {"tokenizer.ggml.pre", "tokenizer.ggml.model"})
        {
            auto values = metadata("qwen35");
            values.erase(key);
            EXPECT_EQ(BPETokenizer::create(values), nullptr);
            values[key] = stringValue("unknown");
            EXPECT_EQ(BPETokenizer::create(values), nullptr);
            values[key] = arrayValue({"qwen35"});
            EXPECT_EQ(BPETokenizer::create(values), nullptr);
        }
        EXPECT_THROW(parseTextPreTokenizerProfile(""), std::invalid_argument);
        EXPECT_THROW(parseTextPreTokenizerProfile("Qwen3.8-27B"), std::invalid_argument);
        auto values = metadata("qwen35");
        values["tokenizer.ggml.tokens"].string_array_value[0] = "missing-byte";
        EXPECT_EQ(BPETokenizer::create(values), nullptr);
    }

    TEST(BPETokenizer, EveryByteDecodesThroughTheProductionVocabulary)
    {
        // The first 256 upstream GPT byte-BPE tokens cover every byte once.
        const auto tokenizer = BPETokenizer::create(metadata("qwen35"));
        ASSERT_NE(tokenizer, nullptr);
        std::array<bool, 256> seen{};
        for (int id = 0; id < 256; ++id)
        {
            const auto value = tokenizer->decode_token(id);
            ASSERT_EQ(value.size(), 1);
            const auto byte = static_cast<unsigned char>(value[0]);
            EXPECT_FALSE(seen[byte]);
            seen[byte] = true;
        }
        EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));
    }

    TEST(BPETokenizer, InvalidUtf8NeverProducesPartialPromptTokens)
    {
        const auto tokenizer = BPETokenizer::create(metadata("qwen35"));
        ASSERT_NE(tokenizer, nullptr);
        for (const auto& invalid : {std::string("\x80"), std::string("\xc0\xaf"), std::string("\xed\xa0\x80"),
                                    std::string("\xf4\x90\x80\x80"), std::string("\xf0\x9f\x91")})
            for (const auto& prefix : {std::string(), std::string("    valid text\n<|im_start|>")})
                EXPECT_THROW(tokenizer->encode(prefix + invalid, false, false), std::runtime_error);
    }

    TEST(BPETokenizer, LongPromptAndConcurrentRequestsRetainIndependentMatchState)
    {
        const auto tokenizer = BPETokenizer::create(metadata("qwen35"));
        ASSERT_NE(tokenizer, nullptr);
        const std::string line = "    def answer():\n        return 123 # 👩🏽‍💻\n";
        const auto lineTokens = tokenizer->encode(line, false, false);
        std::string text;
        std::vector<int> expected;
        for (int i = 0; i < 4096; ++i)
        {
            text += line;
            expected.insert(expected.end(), lineTokens.begin(), lineTokens.end());
        }
        EXPECT_EQ(tokenizer->encode(text, false, false), expected);
        std::vector<std::future<bool>> calls;
        for (int thread = 0; thread < 8; ++thread)
            calls.push_back(std::async(std::launch::async, [tokenizer, line, lineTokens] {
                for (int repeat = 0; repeat < 20; ++repeat)
                    if (tokenizer->encode(line, false, false) != lineTokens) return false;
                return true;
            }));
        for (auto& call : calls) EXPECT_TRUE(call.get());
    }

    TEST(BPETokenizer, ContextSizedRunsPreserveCompleteLinearCoverage)
    {
        // A long coding prompt may contain a single huge literal or whitespace
        // run. Match limits must not impose a hidden small prompt restriction.
        for (const auto& [name, reference] : fixture()["profiles"].items())
        {
            TextPreTokenizer splitter(parseTextPreTokenizerProfile(name));
            for (const char byte : {' ', 'a', '1', '_'})
            {
                SCOPED_TRACE(name + " byte " + std::to_string(byte));
                const std::string text(1 << 20, byte);
                const auto spans = splitter.split(text);
                size_t offset = 0;
                for (const auto span : spans)
                {
                    ASSERT_FALSE(span.empty());
                    ASSERT_EQ(span.data(), text.data() + offset);
                    offset += span.size();
                }
                EXPECT_EQ(offset, text.size());
            }
        }
    }
}
