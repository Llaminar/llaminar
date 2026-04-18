/**
 * @file Test__StopTokens.cpp
 * @brief Integration tests for stop token detection
 * @author David Sanftenberg
 * @date 2025-06-28
 *
 * Tests for stop token functionality including:
 * - stop_tokens() method returns appropriate tokens for chat template
 * - is_stop_token() correctly identifies stop tokens
 * - Template-specific stop token detection (ChatML, LLaMA3, PHI3, etc.)
 */

#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include <climits>

#include "utils/Tokenizer.h"
#include "loaders/ModelContext.h"
#include "utils/Logger.h"

using namespace llaminar2;

namespace
{
    // Test fixture with model loading
    class StopTokensTest : public ::testing::Test
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
                             << " - skipping stop token tests (model may not exist)";
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
    // Basic Stop Token Tests
    // =============================================================================

    TEST_F(StopTokensTest, StopTokensNotEmpty)
    {
        auto stop_tokens = tokenizer_->stop_tokens();
        EXPECT_FALSE(stop_tokens.empty())
            << "stop_tokens() should return at least one stop token";
    }

    TEST_F(StopTokensTest, StopTokensContainEOS)
    {
        auto stop_tokens = tokenizer_->stop_tokens();
        int eos = tokenizer_->eos_token();

        bool contains_eos = std::find(stop_tokens.begin(), stop_tokens.end(), eos) != stop_tokens.end();
        EXPECT_TRUE(contains_eos)
            << "stop_tokens() should include EOS token (" << eos << ")";
    }

    TEST_F(StopTokensTest, StopTokensAreValid)
    {
        auto stop_tokens = tokenizer_->stop_tokens();
        int vocab_size = tokenizer_->vocab_size();

        for (int token : stop_tokens)
        {
            EXPECT_GE(token, 0) << "Stop token should be non-negative";
            EXPECT_LT(token, vocab_size) << "Stop token should be within vocabulary";
        }
    }

    // =============================================================================
    // is_stop_token() Tests
    // =============================================================================

    TEST_F(StopTokensTest, IsStopToken_EOSReturnsTrue)
    {
        int eos = tokenizer_->eos_token();
        EXPECT_TRUE(tokenizer_->is_stop_token(eos))
            << "is_stop_token() should return true for EOS token";
    }

    TEST_F(StopTokensTest, IsStopToken_AllStopTokensReturnTrue)
    {
        auto stop_tokens = tokenizer_->stop_tokens();

        for (int token : stop_tokens)
        {
            EXPECT_TRUE(tokenizer_->is_stop_token(token))
                << "is_stop_token() should return true for stop token " << token;
        }
    }

    TEST_F(StopTokensTest, IsStopToken_RegularTokensReturnFalse)
    {
        // Common tokens that should NOT be stop tokens
        // Token 1 = "!" in Qwen, Token 2 = "\"", etc.
        std::vector<int> regular_tokens = {1, 2, 3, 100, 1000, 10000};
        auto stop_tokens = tokenizer_->stop_tokens();

        for (int token : regular_tokens)
        {
            // Skip if this token happens to be in stop_tokens
            if (std::find(stop_tokens.begin(), stop_tokens.end(), token) != stop_tokens.end())
            {
                continue;
            }

            EXPECT_FALSE(tokenizer_->is_stop_token(token))
                << "is_stop_token() should return false for regular token " << token;
        }
    }

    TEST_F(StopTokensTest, IsStopToken_BOSIsNotStopToken)
    {
        int bos = tokenizer_->bos_token();
        auto stop_tokens = tokenizer_->stop_tokens();

        // BOS should typically not be a stop token
        bool bos_is_stop = std::find(stop_tokens.begin(), stop_tokens.end(), bos) != stop_tokens.end();

        // This test documents expected behavior - BOS is not typically a stop token
        if (!bos_is_stop)
        {
            EXPECT_FALSE(tokenizer_->is_stop_token(bos))
                << "BOS token should not be a stop token";
        }
    }

    // =============================================================================
    // ChatML-Specific Tests (Qwen uses ChatML)
    // =============================================================================

    TEST_F(StopTokensTest, ChatML_ContainsImEnd)
    {
        // Qwen 2.5 uses ChatML template which has <|im_end|> as stop token
        // <|im_end|> is token 151645 in Qwen 2.5
        auto stop_tokens = tokenizer_->stop_tokens();

        // The EOS token IS <|im_end|> in Qwen 2.5
        int eos = tokenizer_->eos_token();
        EXPECT_EQ(eos, 151645) << "Qwen 2.5 EOS token should be 151645 (<|im_end|>)";

        bool contains_im_end = std::find(stop_tokens.begin(), stop_tokens.end(), 151645) != stop_tokens.end();
        EXPECT_TRUE(contains_im_end)
            << "ChatML models should have <|im_end|> (151645) as stop token";
    }

    TEST_F(StopTokensTest, ChatML_ImEndIsStopToken)
    {
        // <|im_end|> token ID for Qwen 2.5
        const int im_end_token = 151645;

        EXPECT_TRUE(tokenizer_->is_stop_token(im_end_token))
            << "is_stop_token() should return true for <|im_end|> token";
    }

    // =============================================================================
    // Edge Cases
    // =============================================================================

    TEST_F(StopTokensTest, IsStopToken_NegativeTokenReturnsFalse)
    {
        // Negative token IDs should not be stop tokens
        EXPECT_FALSE(tokenizer_->is_stop_token(-1))
            << "is_stop_token() should return false for negative token ID";
        EXPECT_FALSE(tokenizer_->is_stop_token(-100))
            << "is_stop_token() should return false for negative token ID";
    }

    TEST_F(StopTokensTest, IsStopToken_OutOfRangeTokenReturnsFalse)
    {
        // Token IDs beyond vocabulary should not be stop tokens
        int vocab_size = tokenizer_->vocab_size();

        EXPECT_FALSE(tokenizer_->is_stop_token(vocab_size))
            << "is_stop_token() should return false for out-of-range token";
        EXPECT_FALSE(tokenizer_->is_stop_token(vocab_size + 1000))
            << "is_stop_token() should return false for out-of-range token";
        EXPECT_FALSE(tokenizer_->is_stop_token(INT_MAX))
            << "is_stop_token() should return false for INT_MAX";
    }

    TEST_F(StopTokensTest, StopTokensConsistent)
    {
        // Calling stop_tokens() multiple times should return the same result
        auto stop_tokens1 = tokenizer_->stop_tokens();
        auto stop_tokens2 = tokenizer_->stop_tokens();

        EXPECT_EQ(stop_tokens1.size(), stop_tokens2.size())
            << "stop_tokens() should return consistent results";

        for (size_t i = 0; i < stop_tokens1.size(); ++i)
        {
            EXPECT_EQ(stop_tokens1[i], stop_tokens2[i])
                << "stop_tokens() should return same tokens on repeated calls";
        }
    }

    // =============================================================================
    // Generation Integration Test
    // =============================================================================

    TEST_F(StopTokensTest, StopTokenCanBeDetectedInSequence)
    {
        // Simulate a generated sequence that ends with a stop token
        auto stop_tokens = tokenizer_->stop_tokens();
        ASSERT_FALSE(stop_tokens.empty()) << "Need at least one stop token for this test";

        // Create a mock generated sequence
        std::vector<int> generated = {100, 200, 300, 400, stop_tokens[0]};

        // Verify we can detect the stop token
        bool found_stop = false;
        for (int token : generated)
        {
            if (tokenizer_->is_stop_token(token))
            {
                found_stop = true;
                break;
            }
        }

        EXPECT_TRUE(found_stop)
            << "Should be able to detect stop token in generated sequence";
    }

    TEST_F(StopTokensTest, NoFalsePositivesInEncodedText)
    {
        // Encode some regular text and verify no tokens are stop tokens
        // (except possibly the last if we add EOS)
        std::string text = "The quick brown fox jumps over the lazy dog";
        auto tokens = tokenizer_->encode(text, /*add_bos=*/false, /*add_eos=*/false);

        ASSERT_FALSE(tokens.empty()) << "Encoded text should not be empty";

        for (size_t i = 0; i < tokens.size(); ++i)
        {
            EXPECT_FALSE(tokenizer_->is_stop_token(tokens[i]))
                << "Regular text token " << tokens[i] << " at position " << i
                << " should not be a stop token";
        }
    }

} // namespace
