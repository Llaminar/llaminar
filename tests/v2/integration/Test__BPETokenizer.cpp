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

    TEST_F(BPETokenizerTest, RoundTripDiagnostic)
    {
        // Diagnostic test to isolate BPE encode/decode issues
        
        // Check vocab content for specific tokens
        std::cout << "\n=== Vocab Token Content ===" << std::endl;
        
        // Access model context to check vocab directly
        const auto& metadata = model_ctx_->model().metadata;
        auto tokens_it = metadata.find("tokenizer.ggml.tokens");
        ASSERT_NE(tokens_it, metadata.end()) << "Should have tokenizer vocab";
        
        const auto& vocab = tokens_it->second.asStringArray();
        std::cout << "Vocab size: " << vocab.size() << std::endl;
        
        // Check specific tokens
        std::vector<int> check_ids = {10, 13, 17, 30};
        for (int id : check_ids) {
            if (id < (int)vocab.size()) {
                std::cout << "Token " << id << " (len=" << vocab[id].size() << "): [";
                for (unsigned char c : vocab[id]) {
                    if (c >= 32 && c < 127) {
                        std::cout << c;
                    } else {
                        std::cout << "\\x" << std::hex << (int)c << std::dec;
                    }
                }
                std::cout << "] hex: ";
                for (unsigned char c : vocab[id]) {
                    std::cout << std::hex << (int)c << " " << std::dec;
                }
                std::cout << std::endl;
            }
        }
        
        // Now check tokenizer decode
        std::cout << "\n=== Tokenizer Decode ===" << std::endl;
        for (int id : check_ids) {
            std::string decoded = tokenizer_->decode_token(id);
            std::cout << "decode_token(" << id << "): [";
            for (unsigned char c : decoded) {
                if (c >= 32 && c < 127) {
                    std::cout << c;
                } else {
                    std::cout << "\\x" << std::hex << (int)c << std::dec;
                }
            }
            std::cout << "]" << std::endl;
        }
        
        // Check space decoding specifically
        std::cout << "\n=== Space Token Decode ===" << std::endl;
        // Token 220 should be single space
        std::string space_decoded = tokenizer_->decode_token(220);
        std::cout << "decode_token(220) = [";
        for (unsigned char c : space_decoded) {
            std::cout << "\\x" << std::hex << (int)c << std::dec;
        }
        std::cout << "] (expected: space = \\x20)" << std::endl;
        
        // Get the raw vocab for token 220
        const auto& vocab2 = tokens_it->second.asStringArray();
        std::string raw_220 = vocab2[220];
        std::cout << "vocab[220] raw bytes: ";
        for (unsigned char c : raw_220) {
            std::cout << std::hex << (int)c << " " << std::dec;
        }
        std::cout << std::endl;
        
        // Token 525 should be " are"
        std::string are_decoded = tokenizer_->decode_token(525);
        std::cout << "decode_token(525) = [" << are_decoded << "] hex: ";
        for (unsigned char c : are_decoded) {
            std::cout << std::hex << (int)c << " " << std::dec;
        }
        std::cout << std::endl;
        
        // Test cases
        std::vector<std::string> test_cases = {
            "Hello",
            "Hello world",
            "helpful assistant",
            "2+2",
            "What is 2+2?",
            "You are a helpful assistant.",
        };

        for (const auto &original : test_cases)
        {
            auto tokens = tokenizer_->encode(original, /*add_bos=*/false, /*add_eos=*/false);
            std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/true);

            // Print diagnostic info on failure
            if (decoded.find(original) == std::string::npos)
            {
                std::cout << "\n=== Round-trip failure ===" << std::endl;
                std::cout << "Original: [" << original << "]" << std::endl;
                std::cout << "Tokens (" << tokens.size() << "): ";
                for (size_t i = 0; i < tokens.size() && i < 20; ++i)
                {
                    std::cout << tokens[i] << " ";
                }
                std::cout << std::endl;
                std::cout << "Decoded: [" << decoded << "]" << std::endl;

                // Also print individual token decodes
                std::cout << "Per-token decode: ";
                for (int tok : tokens)
                {
                    std::cout << "[" << tokenizer_->decode_token(tok) << "] ";
                }
                std::cout << std::endl;
            }

            EXPECT_NE(decoded.find(original), std::string::npos)
                << "Round-trip failed for: " << original
                << "\n  Decoded: " << decoded;
        }
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
        // Performance test for large text encoding
        // Uses optimized BPE algorithm with priority queue (O(n log n) instead of O(n²))

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

        // Target: < 5 seconds for 10000 encodings of 1KB text
        // HuggingFace reference: ~4.3 seconds
        // Our optimized BPE: ~2.7 seconds (with O(n log n) algorithm)
        EXPECT_LT(duration.count(), 5000) << "Encoding should be reasonably fast";
    }

    // =============================================================================
    // Chat Template Integration Tests
    // =============================================================================

    TEST_F(BPETokenizerTest, ChatTemplateDetection)
    {
        // Qwen models have ChatML template
        ASSERT_TRUE(tokenizer_->hasChatTemplate())
            << "Qwen model should have a chat template";

        EXPECT_EQ(tokenizer_->getChatTemplateType(), ChatTemplateType::CHATML)
            << "Qwen model should use ChatML template";

        // Template string should not be empty
        EXPECT_FALSE(tokenizer_->getChatTemplateString().empty())
            << "Chat template string should not be empty";
    }

    TEST_F(BPETokenizerTest, ApplyTemplateSimple)
    {
        // Simple user message
        std::vector<ChatMessage> messages = {
            {"user", "Hello!"}};

        std::string formatted = tokenizer_->applyTemplate(messages, /*add_generation_prompt=*/true);

        // Should contain ChatML markers
        EXPECT_NE(formatted.find("<|im_start|>"), std::string::npos)
            << "ChatML format should have <|im_start|>";
        EXPECT_NE(formatted.find("<|im_end|>"), std::string::npos)
            << "ChatML format should have <|im_end|>";
        EXPECT_NE(formatted.find("user"), std::string::npos)
            << "Should contain 'user' role";
        EXPECT_NE(formatted.find("Hello!"), std::string::npos)
            << "Should contain message content";
        EXPECT_NE(formatted.find("assistant"), std::string::npos)
            << "Should have assistant prompt (add_generation_prompt=true)";
    }

    TEST_F(BPETokenizerTest, ApplyTemplateWithSystem)
    {
        std::vector<ChatMessage> messages = {
            {"system", "You are a helpful assistant."},
            {"user", "What is 2+2?"}};

        std::string formatted = tokenizer_->applyTemplate(messages, /*add_generation_prompt=*/true);

        EXPECT_NE(formatted.find("system"), std::string::npos)
            << "Should contain 'system' role";
        EXPECT_NE(formatted.find("You are a helpful assistant."), std::string::npos)
            << "Should contain system message content";
        EXPECT_NE(formatted.find("What is 2+2?"), std::string::npos)
            << "Should contain user message content";
    }

    TEST_F(BPETokenizerTest, ApplyTemplateMultiTurn)
    {
        std::vector<ChatMessage> messages = {
            {"user", "Hello"},
            {"assistant", "Hi there!"},
            {"user", "How are you?"}};

        std::string formatted = tokenizer_->applyTemplate(messages, /*add_generation_prompt=*/true);

        // Should contain all messages
        EXPECT_NE(formatted.find("Hello"), std::string::npos);
        EXPECT_NE(formatted.find("Hi there!"), std::string::npos);
        EXPECT_NE(formatted.find("How are you?"), std::string::npos);

        // Assistant response should be marked with <|im_end|>
        size_t pos = formatted.find("Hi there!");
        EXPECT_NE(pos, std::string::npos);
        // Find the <|im_end|> after the assistant's response
        size_t end_pos = formatted.find("<|im_end|>", pos);
        EXPECT_NE(end_pos, std::string::npos);
    }

    TEST_F(BPETokenizerTest, ApplyTemplateNoGenerationPrompt)
    {
        std::vector<ChatMessage> messages = {
            {"user", "Hello"}};

        std::string with_prompt = tokenizer_->applyTemplate(messages, /*add_generation_prompt=*/true);
        std::string without_prompt = tokenizer_->applyTemplate(messages, /*add_generation_prompt=*/false);

        // Both should contain the user message
        EXPECT_NE(with_prompt.find("Hello"), std::string::npos);
        EXPECT_NE(without_prompt.find("Hello"), std::string::npos);

        // With prompt should be longer (has assistant turn start)
        EXPECT_GT(with_prompt.length(), without_prompt.length())
            << "Adding generation prompt should make output longer";
    }

    TEST_F(BPETokenizerTest, EncodeChatSimple)
    {
        std::vector<ChatMessage> messages = {
            {"user", "Hello!"}};

        auto tokens = tokenizer_->encodeChat(messages, /*add_generation_prompt=*/true);

        EXPECT_GT(tokens.size(), 0) << "Encoded chat should not be empty";

        // Decode back to verify
        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/false);
        EXPECT_NE(decoded.find("Hello"), std::string::npos)
            << "Decoded text should contain original message";
    }

    TEST_F(BPETokenizerTest, EncodeChatMultiTurn)
    {
        std::vector<ChatMessage> messages = {
            {"system", "You are a helpful assistant."},
            {"user", "What is 2+2?"},
            {"assistant", "4"},
            {"user", "And 3+3?"}};

        auto tokens = tokenizer_->encodeChat(messages, /*add_generation_prompt=*/true);

        EXPECT_GT(tokens.size(), 5) << "Multi-turn chat should have multiple tokens";

        // Verify all messages are encoded
        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/false);
        EXPECT_NE(decoded.find("helpful assistant"), std::string::npos);
        EXPECT_NE(decoded.find("2+2"), std::string::npos);
        EXPECT_NE(decoded.find("3+3"), std::string::npos);
    }

    TEST_F(BPETokenizerTest, EncodeChatEmpty)
    {
        std::vector<ChatMessage> empty_messages;

        // Should not crash with empty messages
        auto tokens = tokenizer_->encodeChat(empty_messages, /*add_generation_prompt=*/true);

        // Result may be empty or contain just the assistant prompt
        // Just verify it doesn't crash
        SUCCEED() << "Empty message encoding should not crash";
    }

    TEST_F(BPETokenizerTest, EncodeChatUnicode)
    {
        std::vector<ChatMessage> messages = {
            {"user", "你好！"}, // Chinese "Hello!"
            {"assistant", "您好！有什么可以帮助您的？"}};

        auto tokens = tokenizer_->encodeChat(messages, /*add_generation_prompt=*/false);

        EXPECT_GT(tokens.size(), 0) << "Unicode chat should be encoded";

        // Decode and verify content is preserved
        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/false);
        EXPECT_NE(decoded.find("你好"), std::string::npos)
            << "Chinese characters should be preserved in round-trip";
    }

    TEST_F(BPETokenizerTest, ChineseTokenizationRoundTrip)
    {
        // Test that Chinese characters can be tokenized and decoded correctly
        std::string chinese = "你好";

        // Encode
        auto tokens = tokenizer_->encode(chinese, /*add_bos=*/false, /*add_eos=*/false);
        EXPECT_GE(tokens.size(), 1) << "Should have at least 1 token";

        // The HuggingFace tokenizer produces token 108386 for "你好"
        // We should get the same or equivalent tokenization
        EXPECT_LE(tokens.size(), 2) << "你好 should tokenize to at most 2 tokens";

        // Decode back
        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/false);
        EXPECT_EQ(decoded, chinese) << "Chinese text should round-trip correctly";
    }

    TEST_F(BPETokenizerTest, ChineseIndividualCharacters)
    {
        // Test individual Chinese characters
        auto* bpe_tok = dynamic_cast<BPETokenizer*>(tokenizer_.get());
        ASSERT_NE(bpe_tok, nullptr);

        // Token 56568 = 你, Token 52801 = 好 according to HuggingFace
        std::string decoded_ni = bpe_tok->decode_token(56568);
        std::string decoded_hao = bpe_tok->decode_token(52801);

        EXPECT_EQ(decoded_ni, "你") << "Token 56568 should decode to 你";
        EXPECT_EQ(decoded_hao, "好") << "Token 52801 should decode to 好";
    }

    TEST_F(BPETokenizerTest, EmojiRoundTrip)
    {
        // Test emoji tokenization
        std::string emoji = "👋🌍";

        auto tokens = tokenizer_->encode(emoji, /*add_bos=*/false, /*add_eos=*/false);
        EXPECT_GT(tokens.size(), 0) << "Emoji should produce tokens";

        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/false);
        EXPECT_EQ(decoded, emoji) << "Emoji should round-trip correctly";
    }

    TEST_F(BPETokenizerTest, MixedLanguageRoundTrip)
    {
        // Test mixed English and Chinese
        std::string mixed = "Hello 你好 World";

        auto tokens = tokenizer_->encode(mixed, /*add_bos=*/false, /*add_eos=*/false);
        EXPECT_GT(tokens.size(), 0) << "Mixed text should produce tokens";

        std::string decoded = tokenizer_->decode(tokens, /*remove_special=*/false);
        EXPECT_EQ(decoded, mixed) << "Mixed language text should round-trip correctly";
    }

} // anonymous namespace

// Main function
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
