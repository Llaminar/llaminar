/**
 * @file MockTokenizer.h
 * @brief Mock implementation of ITokenizer for unit testing
 */

#pragma once

#include "utils/Tokenizer.h"
#include <gmock/gmock.h>

namespace llaminar2::test
{

    class MockTokenizer : public ITokenizer
    {
    public:
        MOCK_METHOD(std::vector<int>, encode,
                    (const std::string &text, bool add_bos, bool add_eos),
                    (const, override));

        MOCK_METHOD(std::string, decode,
                    (const std::vector<int> &tokens, bool remove_special),
                    (const, override));

        MOCK_METHOD(std::string, decode_token,
                    (int token), (const, override));

        MOCK_METHOD(int, bos_token, (), (const, override));
        MOCK_METHOD(int, eos_token, (), (const, override));
        MOCK_METHOD(std::vector<int>, stop_tokens, (), (const, override));
        MOCK_METHOD(bool, is_stop_token, (int token_id), (const, override));
        MOCK_METHOD(int, vocab_size, (), (const, override));

        // Chat template
        MOCK_METHOD(std::string, getChatTemplateString, (), (const, override));
        MOCK_METHOD(ChatTemplateType, getChatTemplateType, (), (const, override));
        MOCK_METHOD(bool, hasChatTemplate, (), (const, override));
        MOCK_METHOD(void, setChatTemplate, (std::unique_ptr<ChatTemplate> tmpl), (override));

        MOCK_METHOD(std::vector<int>, encodeChat,
                    (const std::vector<ChatMessage> &messages, bool add_generation_prompt),
                    (const, override));

        MOCK_METHOD(std::string, applyTemplate,
                    (const std::vector<ChatMessage> &messages, bool add_generation_prompt),
                    (const, override));
    };

} // namespace llaminar2::test
