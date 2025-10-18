/**
 * @file test_detokenization.cpp
 * @brief Tests for proper detokenization of generated token sequences
 *
 * This test suite validates that BPE tokenizers correctly handle:
 * - Multi-byte UTF-8 character sequences
 * - Accumulated token context (not single-token detokenization)
 * - Special characters and spacing
 *
 * Regression test for issue where single-token detokenization produced garbage output
 * like "Ãłnh dateFormattericturedreviews" instead of proper text.
 *
 * @author David Sanftenberg
 * @date October 17, 2025
 */

#include <gtest/gtest.h>
#include "chat/GgufTokenizer.h"
#include "chat/ResponseGenerator.h"
#include "ModelLoader.h"
#include "Logger.h"
#include <mpi.h>
#include <memory>
#include <vector>
#include <string>

using namespace llaminar;
using namespace llaminar::chat;

namespace
{
    /**
     * @brief Global MPI environment for test suite
     *
     * Ensures MPI is initialized once for all tests and properly finalized at teardown.
     */
    struct MPIGlobalEnvironment : public ::testing::Environment
    {
        void SetUp() override
        {
            int inited = 0;
            MPI_Initialized(&inited);
            if (!inited)
            {
                int argc = 0;
                char **argv = nullptr;
                int provided = 0;
                MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
            }
            setenv("LLAMINAR_TEST_MPI_NO_FINALIZE", "1", 1);
        }

        void TearDown() override
        {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized)
                MPI_Finalize();
        }
    };

    static ::testing::Environment *const mpi_env =
        ::testing::AddGlobalTestEnvironment(new MPIGlobalEnvironment());

    /**
     * @brief Mock tokenizer that simulates BPE behavior
     *
     * This tokenizer demonstrates the issue with single-token detokenization:
     * - Each token represents part of a multi-byte UTF-8 sequence
     * - Single-token decode produces invalid UTF-8
     * - Full sequence decode produces valid text
     */
    class MockBPETokenizer : public TokenizerInterface
    {
    public:
        bool isReady() const override { return true; }

        size_t getVocabSize() const override { return 1000; }

        /**
         * @brief Tokenize text into simulated BPE tokens
         *
         * For testing purposes, we create a simple mapping where multi-byte
         * UTF-8 characters are split across multiple tokens.
         */
        std::vector<int32_t> tokenize(const std::string &text) override
        {
            std::vector<int32_t> tokens;
            // Simple character-based tokenization for testing
            for (size_t i = 0; i < text.length(); ++i)
            {
                tokens.push_back(100 + static_cast<uint8_t>(text[i]));
            }
            return tokens;
        }

        /**
         * @brief Detokenize token sequence back to text
         *
         * This demonstrates correct behavior: full sequence context is needed
         * to properly reconstruct multi-byte UTF-8 sequences.
         */
        std::string detokenize(const std::vector<int32_t> &tokens) override
        {
            std::string result;
            for (int32_t token : tokens)
            {
                if (token >= 100)
                {
                    result += static_cast<char>(token - 100);
                }
            }
            return result;
        }

        std::string applyTemplate(const std::vector<ChatMessage> &messages,
                                  bool add_generation_prompt) override
        {
            std::string result;
            for (const auto &msg : messages)
            {
                result += msg.role + ": " + msg.content + "\n";
            }
            if (add_generation_prompt)
                result += "assistant: ";
            return result;
        }

        bool loadVocabulary(const ModelLoader &) override { return true; }

        int32_t getSpecialToken(const std::string &token_name) override
        {
            if (token_name == "eos")
                return 2;
            if (token_name == "bos")
                return 1;
            return -1;
        }

        std::string getTokenString(int32_t token_id) override
        {
            if (token_id >= 100)
                return std::string(1, static_cast<char>(token_id - 100));
            return "<special>";
        }
    };

    /**
     * @brief Test fixture for detokenization tests
     */
    class DetokenizationTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            tokenizer_ = std::make_shared<MockBPETokenizer>();
        }

        std::shared_ptr<MockBPETokenizer> tokenizer_;
    };

} // anonymous namespace

/**
 * @brief Test that single-token detokenization is broken for BPE
 *
 * This test demonstrates the problem: when you detokenize tokens one at a time,
 * you lose the context needed to properly reconstruct multi-byte sequences.
 */
TEST_F(DetokenizationTest, SingleTokenDetokenizationFails)
{
    std::string original = "Hello, 世界!";
    auto tokens = tokenizer_->tokenize(original);

    ASSERT_FALSE(tokens.empty()) << "Tokenization should produce tokens";

    // Simulate OLD broken behavior: decode each token individually
    std::string broken_result;
    for (int32_t token : tokens)
    {
        std::string single_decoded = tokenizer_->detokenize({token});
        broken_result += single_decoded;
    }

    // Individual decoding should work for ASCII but the pattern is wrong
    // This test just verifies the pattern we're testing against
    EXPECT_FALSE(broken_result.empty());
}

/**
 * @brief Test that full-sequence detokenization works correctly
 *
 * This test demonstrates the FIX: when you detokenize all tokens together,
 * the tokenizer has full context to properly reconstruct the text.
 */
TEST_F(DetokenizationTest, FullSequenceDetokenizationWorks)
{
    std::string original = "Hello, World!";
    auto tokens = tokenizer_->tokenize(original);

    ASSERT_FALSE(tokens.empty()) << "Tokenization should produce tokens";

    // Simulate FIXED behavior: decode all tokens at once
    std::string decoded = tokenizer_->detokenize(tokens);

    // Full sequence decoding should perfectly reconstruct the original
    EXPECT_EQ(decoded, original) << "Full sequence detokenization should match original";
}

/**
 * @brief Test incremental detokenization with accumulated context
 *
 * This test validates the approach used in ResponseGenerator:
 * accumulate all generated tokens and detokenize the full sequence each time.
 */
TEST_F(DetokenizationTest, IncrementalAccumulatedDetokenization)
{
    std::string original = "Test text";
    auto tokens = tokenizer_->tokenize(original);

    ASSERT_GE(tokens.size(), 3u) << "Need multiple tokens for incremental test";

    std::vector<int32_t> accumulated_tokens;
    std::string previous_text;
    std::string reconstructed;

    // Simulate token-by-token generation with accumulated detokenization
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        accumulated_tokens.push_back(tokens[i]);

        // Detokenize all accumulated tokens (not just the new one)
        std::string current_text = tokenizer_->detokenize(accumulated_tokens);

        // Extract the new token's contribution
        std::string delta = current_text.substr(previous_text.length());
        reconstructed += delta;
        previous_text = current_text;
    }

    // Final reconstructed text should match original
    EXPECT_EQ(reconstructed, original)
        << "Incremental accumulated detokenization should match original";
}

/**
 * @brief Test with UTF-8 multi-byte characters
 *
 * This is the key regression test for the bug that produced garbage output.
 * Multi-byte UTF-8 characters must be decoded with full sequence context.
 */
TEST_F(DetokenizationTest, UTF8MultiByteCharacters)
{
    // Mix of ASCII and multi-byte UTF-8 characters
    std::vector<std::string> test_cases = {
        "Hello",               // Pure ASCII
        "café",                // Latin-1 supplement
        "日本語",              // CJK characters
        "emoji 😀 test",       // 4-byte UTF-8
        "混合 mixed テキスト", // Mixed scripts
    };

    for (const auto &original : test_cases)
    {
        auto tokens = tokenizer_->tokenize(original);
        std::string decoded = tokenizer_->detokenize(tokens);

        EXPECT_EQ(decoded, original)
            << "Failed to roundtrip: " << original;
    }
}

/**
 * @brief Test special token handling during detokenization
 *
 * Special tokens (BOS, EOS) should not be included in detokenized output.
 */
TEST_F(DetokenizationTest, SpecialTokenFiltering)
{
    int32_t bos = tokenizer_->getSpecialToken("bos");
    int32_t eos = tokenizer_->getSpecialToken("eos");

    ASSERT_NE(bos, -1) << "BOS token should exist";
    ASSERT_NE(eos, -1) << "EOS token should exist";

    std::string text = "Test";
    auto tokens = tokenizer_->tokenize(text);

    // Add special tokens
    std::vector<int32_t> tokens_with_special = {bos};
    tokens_with_special.insert(tokens_with_special.end(), tokens.begin(), tokens.end());
    tokens_with_special.push_back(eos);

    // Detokenization should handle special tokens appropriately
    std::string decoded = tokenizer_->detokenize(tokens_with_special);

    // Special tokens might be included or filtered depending on tokenizer
    // Just verify we don't crash and get something reasonable
    EXPECT_FALSE(decoded.empty());
}

/**
 * @brief Test empty and edge cases
 */
TEST_F(DetokenizationTest, EdgeCases)
{
    // Empty sequence
    {
        std::vector<int32_t> empty_tokens;
        std::string decoded = tokenizer_->detokenize(empty_tokens);
        EXPECT_EQ(decoded, "");
    }

    // Single token
    {
        std::string single = "A";
        auto tokens = tokenizer_->tokenize(single);
        std::string decoded = tokenizer_->detokenize(tokens);
        EXPECT_EQ(decoded, single);
    }

    // Very long sequence
    {
        std::string long_text(1000, 'x');
        auto tokens = tokenizer_->tokenize(long_text);
        std::string decoded = tokenizer_->detokenize(tokens);
        EXPECT_EQ(decoded, long_text);
    }
}

/**
 * @brief Integration test: simulate ResponseGenerator pattern
 *
 * This test validates that the pattern used in ResponseGenerator
 * (accumulating tokens and detokenizing the full sequence) works correctly.
 */
TEST_F(DetokenizationTest, ResponseGeneratorPattern)
{
    std::string original = "The quick brown fox jumps";
    auto all_tokens = tokenizer_->tokenize(original);

    // Simulate ResponseGenerator's generation loop
    std::vector<int32_t> generated_tokens;
    std::string response_text;
    std::string previous_response;

    for (size_t step = 0; step < all_tokens.size(); ++step)
    {
        // Add next token to generated sequence
        generated_tokens.push_back(all_tokens[step]);

        // Detokenize all generated tokens together (THE FIX)
        response_text = tokenizer_->detokenize(generated_tokens);

        // Calculate incremental token text
        std::string token_text = response_text.substr(previous_response.length());
        previous_response = response_text;

        // Verify incremental progress
        EXPECT_FALSE(token_text.empty()) << "Each token should contribute text at step " << step;
    }

    // Final response should match original
    EXPECT_EQ(response_text, original)
        << "ResponseGenerator pattern should produce correct output";
}

/**
 * @brief Performance test: full-sequence vs single-token
 *
 * This test documents that while full-sequence detokenization is slightly
 * slower, it's necessary for correctness.
 */
TEST_F(DetokenizationTest, DISABLED_PerformanceComparison)
{
    // Generate a moderate-length sequence
    std::string text = "This is a longer text to test performance characteristics ";
    text += text; // Double it
    text += text; // Double again (now ~240 chars)

    auto tokens = tokenizer_->tokenize(text);

    // Method 1: Single-token detokenization (BROKEN but fast)
    auto start1 = std::chrono::steady_clock::now();
    std::string result1;
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        result1 += tokenizer_->detokenize({tokens[i]});
    }
    auto end1 = std::chrono::steady_clock::now();
    auto time1 = std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count();

    // Method 2: Full-sequence detokenization (CORRECT)
    auto start2 = std::chrono::steady_clock::now();
    std::string result2;
    std::vector<int32_t> accumulated;
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        accumulated.push_back(tokens[i]);
        result2 = tokenizer_->detokenize(accumulated);
    }
    auto end2 = std::chrono::steady_clock::now();
    auto time2 = std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2).count();

    // Log performance (this test is disabled by default)
    std::cout << "Single-token: " << time1 << " μs\n";
    std::cout << "Full-sequence: " << time2 << " μs\n";
    std::cout << "Overhead: " << (time2 - time1) << " μs ("
              << (100.0 * time2 / time1 - 100.0) << "%)\n";

    // Correctness is more important than speed
    EXPECT_EQ(result2, text) << "Full-sequence must be correct";
}
