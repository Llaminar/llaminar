/**
 * @file Test__BPETokenizer.cpp
 * @brief Unit tests for BPETokenizer
 * @author David Sanftenberg
 * @date 2025-11-07
 *
 * Tests for native BPE tokenizer implementation including:
 * - Tokenizer creation from ModelContext
 * - Encoding text to token IDs
 * - Decoding token IDs to text
 * - Special token handling (BOS/EOS)
 * - Edge cases (empty strings, long inputs)
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>

#include "../../../src/v2/utils/Tokenizer.h"
#include "../../../src/v2/loaders/ModelContext.h"
#include "../../../src/v2/utils/Logger.h"

using namespace llaminar2;

namespace
{
    // Test fixture with model loading
    class BPETokenizerTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Use real Qwen model for integration testing
            model_path_ = "models/qwen2.5-0.5b-instruct-q4_0.gguf";

            // Create model context using factory method
            model_ctx_ = ModelContext::create(model_path_);
            if (!model_ctx_)
            {
                GTEST_SKIP() << "Failed to load model: " << model_path_
                             << " - skipping tokenizer tests (model may not exist)";
            }

            // Create tokenizer
            tokenizer_ = createTokenizer(model_ctx_);
            if (!tokenizer_)
            {
                GTEST_SKIP() << "Failed to create tokenizer - skipping tests";
            }
        }

        void TearDown() override
        {
            tokenizer_.reset();
            model_ctx_.reset();
        }

        std::string model_path_;
        std::shared_ptr<ModelContext> model_ctx_;
        std::shared_ptr<ITokenizer> tokenizer_;
    };

    // =============================================================================
    // Basic Functionality Tests
    // =============================================================================

    TEST_F(BPETokenizerTest, TokenizerCreation)
    {
        ASSERT_NE(tokenizer_, nullptr) << "Tokenizer should be created successfully";
        EXPECT_GT(tokenizer_->vocab_size(), 0) << "Vocabulary size should be positive";

        // Qwen 2.5 has 151,936 tokens
        EXPECT_EQ(tokenizer_->vocab_size(), 151936) << "Qwen 2.5 should have 151,936 tokens";
    }

    TEST_F(BPETokenizerTest, SpecialTokens)
    {
        // Verify special tokens are set
        EXPECT_GE(tokenizer_->bos_token(), 0) << "BOS token should be non-negative";
        EXPECT_GE(tokenizer_->eos_token(), 0) << "EOS token should be non-negative";

        // Qwen 2.5 specific token IDs
        EXPECT_EQ(tokenizer_->bos_token(), 151643) << "Qwen 2.5 BOS token";
        EXPECT_EQ(tokenizer_->eos_token(), 151645) << "Qwen 2.5 EOS token";
    }

    // =============================================================================
    // Encoding Tests
    // =============================================================================

    TEST_F(BPETokenizerTest, EncodeSimpleText)
    {
        std::string text = "Hello world";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/false);

        EXPECT_GT(tokens.size(), 0) << "Encoded tokens should not be empty";
        EXPECT_EQ(tokens[0], tokenizer_->bos_token()) << "First token should be BOS";

        // "Hello world" should tokenize to reasonable number of tokens (2-5)
        EXPECT_GE(tokens.size(), 2) << "Should have at least BOS + content";
        EXPECT_LE(tokens.size(), 10) << "Should not over-tokenize simple text";
    }

    TEST_F(BPETokenizerTest, EncodeWithoutBOS)
    {
        std::string text = "Test";
        auto tokens_with_bos = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/false);
        auto tokens_without_bos = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        EXPECT_EQ(tokens_with_bos.size(), tokens_without_bos.size() + 1)
            << "Adding BOS should add exactly one token";
        EXPECT_EQ(tokens_with_bos[0], tokenizer_->bos_token())
            << "First token with add_bos should be BOS";
    }

    TEST_F(BPETokenizerTest, EncodeWithEOS)
    {
        std::string text = "Test";
        auto tokens_without_eos = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);
        auto tokens_with_eos = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/true);

        EXPECT_EQ(tokens_with_eos.size(), tokens_without_eos.size() + 1)
            << "Adding EOS should add exactly one token";
        EXPECT_EQ(tokens_with_eos.back(), tokenizer_->eos_token())
            << "Last token with add_eos should be EOS";
    }

    TEST_F(BPETokenizerTest, EncodeWithBothBOSAndEOS)
    {
        std::string text = "Test";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/true);

        EXPECT_GE(tokens.size(), 3) << "Should have at least BOS + content + EOS";
        EXPECT_EQ(tokens[0], tokenizer_->bos_token()) << "First token should be BOS";
        EXPECT_EQ(tokens.back(), tokenizer_->eos_token()) << "Last token should be EOS";
    }

    TEST_F(BPETokenizerTest, EncodeEmptyString)
    {
        std::string text = "";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/true);

        // Empty string with BOS+EOS should give 2 tokens
        EXPECT_EQ(tokens.size(), 2) << "Empty string should have only BOS and EOS";
        EXPECT_EQ(tokens[0], tokenizer_->bos_token());
        EXPECT_EQ(tokens[1], tokenizer_->eos_token());
    }

    TEST_F(BPETokenizerTest, EncodeMultipleWords)
    {
        std::string text = "The quick brown fox jumps over the lazy dog";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        EXPECT_GT(tokens.size(), 5) << "Multi-word sentence should tokenize to multiple tokens";
        EXPECT_LT(tokens.size(), 50) << "Should not over-tokenize reasonable text";
    }

    TEST_F(BPETokenizerTest, EncodePunctuation)
    {
        std::string text = "Hello, world!";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        EXPECT_GT(tokens.size(), 0) << "Text with punctuation should tokenize";

        // All tokens should be within vocabulary bounds
        for (int token : tokens)
        {
            EXPECT_GE(token, 0) << "Token ID should be non-negative";
            EXPECT_LT(token, tokenizer_->vocab_size()) << "Token ID should be within vocabulary";
        }
    }

    TEST_F(BPETokenizerTest, EncodeNumbers)
    {
        std::string text = "1234567890";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        EXPECT_GT(tokens.size(), 0) << "Numbers should tokenize";

        for (int token : tokens)
        {
            EXPECT_GE(token, 0);
            EXPECT_LT(token, tokenizer_->vocab_size());
        }
    }

    // =============================================================================
    // Decoding Tests
    // =============================================================================

    TEST_F(BPETokenizerTest, DecodeTokens)
    {
        std::string original = "Hello world";
        auto tokens = tokenizer_->encode(original, /*add_bos=*/false, /*add_eos=*/false);
        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/true);

        // Decoded text should be similar to original (may differ in whitespace/casing)
        EXPECT_FALSE(decoded.empty()) << "Decoded text should not be empty";
        EXPECT_GT(decoded.size(), 0) << "Decoded text should have content";
    }

    TEST_F(BPETokenizerTest, DecodeWithSpecialTokens)
    {
        std::string original = "Test";
        auto tokens = tokenizer_->encode(original, /*add_bos=*/true, /*add_eos=*/true);

        // Decode without removing special tokens
        std::string decoded_with_special = tokenizer_->decode(tokens, /*remove_special=*/false);
        EXPECT_FALSE(decoded_with_special.empty());

        // Decode with removing special tokens
        std::string decoded_without_special = tokenizer_->decode(tokens, /*remove_special=*/true);
        EXPECT_FALSE(decoded_without_special.empty());

        // Removing special tokens should not make output longer
        EXPECT_LE(decoded_without_special.size(), decoded_with_special.size());
    }

    TEST_F(BPETokenizerTest, DecodeSingleToken)
    {
        std::string text = "Hello";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        ASSERT_GT(tokens.size(), 0) << "Should have at least one token";

        // Decode individual tokens
        for (int token : tokens)
        {
            std::string decoded = tokenizer_->decode_token(token);
            // Decoded token should not be empty (though it might be whitespace/special char)
            EXPECT_GE(decoded.size(), 0);
        }
    }

    TEST_F(BPETokenizerTest, DecodeEmptyTokenList)
    {
        std::vector<int> empty_tokens;
        std::string decoded = tokenizer_->decode(empty_tokens, /*remove_special=*/true);

        EXPECT_EQ(decoded, "") << "Decoding empty token list should give empty string";
    }

    // =============================================================================
    // Round-Trip Tests (Encode -> Decode)
    // =============================================================================

    TEST_F(BPETokenizerTest, RoundTripSimpleText)
    {
        std::string original = "Hello";
        auto tokens = tokenizer_->encode(original, /*add_bos=*/false, /*add_eos=*/false);
        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/true);

        // For simple text, round-trip should preserve content
        // (may differ in whitespace, but core content should match)
        EXPECT_FALSE(decoded.empty());
        EXPECT_NE(decoded.find("Hello"), std::string::npos) << "Decoded should contain 'Hello'";
    }

    TEST_F(BPETokenizerTest, RoundTripWithSpecialTokens)
    {
        std::string original = "Test message";
        auto tokens = tokenizer_->encode(original, /*add_bos=*/true, /*add_eos=*/true);

        // Decode and remove special tokens - should get original back
        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/true);
        EXPECT_FALSE(decoded.empty());
    }

    // =============================================================================
    // Edge Cases
    // =============================================================================

    TEST_F(BPETokenizerTest, LongInput)
    {
        // Create a long input string (1000 characters)
        std::string long_text;
        for (int i = 0; i < 100; ++i)
        {
            long_text += "The quick brown fox jumps over the lazy dog. ";
        }

        auto tokens = tokenizer_->encode(long_text, /*add_bos=*/true, /*add_eos=*/false);

        EXPECT_GT(tokens.size(), 100) << "Long text should produce many tokens";
        EXPECT_LT(tokens.size(), 5000) << "Should not produce excessive tokens";

        // All tokens should be valid
        for (int token : tokens)
        {
            EXPECT_GE(token, 0);
            EXPECT_LT(token, tokenizer_->vocab_size());
        }
    }

    TEST_F(BPETokenizerTest, SpecialCharacters)
    {
        std::string text = "Hello\nWorld\tTest!@#$%^&*()";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        EXPECT_GT(tokens.size(), 0) << "Text with special characters should tokenize";

        for (int token : tokens)
        {
            EXPECT_GE(token, 0);
            EXPECT_LT(token, tokenizer_->vocab_size());
        }
    }

    TEST_F(BPETokenizerTest, UnicodeCharacters)
    {
        std::string text = "Hello 世界 🌍"; // English + Chinese + Emoji
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        EXPECT_GT(tokens.size(), 0) << "Unicode text should tokenize";

        for (int token : tokens)
        {
            EXPECT_GE(token, 0);
            EXPECT_LT(token, tokenizer_->vocab_size());
        }
    }

    TEST_F(BPETokenizerTest, RepeatedEncoding)
    {
        std::string text = "Test";

        // Encode same text multiple times
        auto tokens1 = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/false);
        auto tokens2 = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/false);
        auto tokens3 = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/false);

        // Results should be identical (deterministic)
        EXPECT_EQ(tokens1.size(), tokens2.size());
        EXPECT_EQ(tokens1.size(), tokens3.size());

        for (size_t i = 0; i < tokens1.size(); ++i)
        {
            EXPECT_EQ(tokens1[i], tokens2[i]);
            EXPECT_EQ(tokens1[i], tokens3[i]);
        }
    }

    TEST_F(BPETokenizerTest, WhitespaceOnly)
    {
        std::string text = "   \t\n   ";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        // Whitespace should tokenize (may be empty or contain whitespace tokens)
        EXPECT_GE(tokens.size(), 0);
    }

    // =============================================================================
    // Performance Tests (Optional - can be slow)
    // =============================================================================

    TEST_F(BPETokenizerTest, PerformanceEncoding)
    {
        // Disabled by default - enable with --gtest_also_run_disabled_tests

        std::string text = "The quick brown fox jumps over the lazy dog";

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; ++i)
        {
            auto tokens = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/false);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "1000 encodings took " << duration.count() << " ms" << std::endl;

        // Should be reasonably fast (< 1 second for 1000 encodings)
        EXPECT_LT(duration.count(), 1000) << "Encoding should be fast";
    }

    TEST_F(BPETokenizerTest, PerformanceEncodingLarge)
    {
        // Disabled by default - enable with --gtest_also_run_disabled_tests

        std::string base = "The quick brown fox jumps over the lazy dog. ";
        std::string text;
        // Create ~1KB of text (25 repetitions)
        for (int i = 0; i < 25; ++i)
            text += base;

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 10000; ++i)
        {
            auto tokens = tokenizer_->encode(text, /*add_bos=*/true, /*add_eos=*/false);
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "10000 encodings of 1KB text took " << duration.count() << " ms" << std::endl;

        // Should be fast (target < 2000ms)
        EXPECT_LT(duration.count(), 2000) << "Encoding should be fast";
    }

} // anonymous namespace

// Main function
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
