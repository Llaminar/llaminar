/**
 * @file Test__ChatTemplate.cpp
 * @brief Unit tests for ChatTemplate class
 * @author David Sanftenberg
 * @date 2025
 */

#include <gtest/gtest.h>
#include "utils/ChatTemplate.h"

using namespace llaminar2;

// ============================================================================
// Template Detection Tests
// ============================================================================

TEST(Test__ChatTemplate, DetectsChatML)
{
    // Qwen-style ChatML template
    std::string template_str = R"(
        {% for message in messages %}
        <|im_start|>{{ message['role'] }}
        {{ message['content'] }}<|im_end|>
        {% endfor %}
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::CHATML);
}

TEST(Test__ChatTemplate, DetectsLlama3)
{
    std::string template_str = R"(
        {% for message in messages %}
        <|start_header_id|>{{ message['role'] }}<|end_header_id|>
        {{ message['content'] }}<|eot_id|>
        {% endfor %}
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::LLAMA3);
}

TEST(Test__ChatTemplate, DetectsLlama2)
{
    std::string template_str = R"(
        [INST] <<SYS>>
        {{ system }}
        <</SYS>>
        {{ user }} [/INST] {{ assistant }}
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::LLAMA2);
}

TEST(Test__ChatTemplate, DetectsMistralV1)
{
    std::string template_str = R"(
        {{ bos_token }} [INST] {{ system }} {{ user }} [/INST]
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::MISTRAL_V1);
}

TEST(Test__ChatTemplate, DetectsMistralV7)
{
    std::string template_str = R"(
        [SYSTEM_PROMPT] {{ system }} [/SYSTEM_PROMPT]
        [INST] {{ user }} [/INST]
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::MISTRAL_V7);
}

TEST(Test__ChatTemplate, DetectsPhi3)
{
    std::string template_str = R"(
        <|system|>
        {{ system }}<|end|>
        <|user|>
        {{ user }}<|end|>
        <|assistant|>
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::PHI3);
}

TEST(Test__ChatTemplate, DetectsPhi4)
{
    std::string template_str = R"(
        <|im_start|>system<|im_sep|>{{ system }}<|im_end|>
        <|im_start|>user<|im_sep|>{{ user }}<|im_end|>
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::PHI4);
}

TEST(Test__ChatTemplate, DetectsGemma)
{
    std::string template_str = R"(
        <start_of_turn>user
        {{ user }}<end_of_turn>
        <start_of_turn>model
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::GEMMA);
}

TEST(Test__ChatTemplate, DetectsZephyr)
{
    std::string template_str = R"(
        <|system|>
        {{ system }}<|endoftext|>
        <|user|>
        {{ user }}<|endoftext|>
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::ZEPHYR);
}

TEST(Test__ChatTemplate, DetectsVicuna)
{
    std::string template_str = R"(
        {{ system }}
        USER: {{ user }}
        ASSISTANT: {{ assistant }}
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::VICUNA);
}

TEST(Test__ChatTemplate, DetectsDeepSeek)
{
    std::string template_str = R"(
        {{ system }}
        ### Instruction:
        {{ user }}
        ### Response:
        {{ assistant }}<|EOT|>
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::DEEPSEEK);
}

TEST(Test__ChatTemplate, DetectsCommandR)
{
    std::string template_str = R"(
        <|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{ system }}<|END_OF_TURN_TOKEN|>
        <|START_OF_TURN_TOKEN|><|USER_TOKEN|>{{ user }}<|END_OF_TURN_TOKEN|>
    )";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::COMMAND_R);
}

TEST(Test__ChatTemplate, FallsBackToUnknown)
{
    std::string template_str = "Some random template without known markers";

    auto tmpl = ChatTemplate::create(template_str);
    EXPECT_EQ(tmpl->type(), ChatTemplateType::UNKNOWN);
    EXPECT_TRUE(tmpl->isUnknown());
}

// ============================================================================
// Template Application Tests
// ============================================================================

TEST(Test__ChatTemplate, ApplyChatMLBasic)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"user", "Hello!"}};

    std::string result = tmpl->apply(messages, true);

    EXPECT_NE(result.find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(result.find("Hello!"), std::string::npos);
    EXPECT_NE(result.find("<|im_end|>"), std::string::npos);
    EXPECT_NE(result.find("<|im_start|>assistant"), std::string::npos);
}

TEST(Test__ChatTemplate, ApplyChatMLWithSystem)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"system", "You are a helpful assistant."},
        {"user", "What is 2+2?"}};

    std::string result = tmpl->apply(messages, true);

    // Verify order: system before user
    size_t sys_pos = result.find("<|im_start|>system");
    size_t user_pos = result.find("<|im_start|>user");
    size_t ass_pos = result.find("<|im_start|>assistant");

    EXPECT_NE(sys_pos, std::string::npos);
    EXPECT_NE(user_pos, std::string::npos);
    EXPECT_NE(ass_pos, std::string::npos);
    EXPECT_LT(sys_pos, user_pos);
    EXPECT_LT(user_pos, ass_pos);
}

TEST(Test__ChatTemplate, ApplyChatMLMultiTurn)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"user", "Hello!"},
        {"assistant", "Hi there!"},
        {"user", "How are you?"}};

    std::string result = tmpl->apply(messages, true);

    // Count occurrences of im_start
    size_t count = 0;
    size_t pos = 0;
    while ((pos = result.find("<|im_start|>", pos)) != std::string::npos)
    {
        count++;
        pos++;
    }

    // 3 messages + 1 generation prompt = 4
    EXPECT_EQ(count, 4);
}

TEST(Test__ChatTemplate, ApplyChatMLNoGenerationPrompt)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"user", "Hello!"}};

    std::string result = tmpl->apply(messages, false);

    // Should NOT have assistant prompt at end
    size_t last_im_start = result.rfind("<|im_start|>");
    EXPECT_NE(last_im_start, std::string::npos);

    std::string after_last = result.substr(last_im_start);
    EXPECT_EQ(after_last.find("assistant"), std::string::npos);
}

TEST(Test__ChatTemplate, ApplyLlama3Basic)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::LLAMA3);

    std::vector<ChatMessage> messages = {
        {"user", "Hello!"}};

    std::string result = tmpl->apply(messages, true);

    EXPECT_NE(result.find("<|start_header_id|>user<|end_header_id|>"), std::string::npos);
    EXPECT_NE(result.find("Hello!"), std::string::npos);
    EXPECT_NE(result.find("<|eot_id|>"), std::string::npos);
    EXPECT_NE(result.find("<|start_header_id|>assistant<|end_header_id|>"), std::string::npos);
}

TEST(Test__ChatTemplate, ApplyGemmaConvertsAssistantToModel)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::GEMMA);

    std::vector<ChatMessage> messages = {
        {"user", "Hello!"},
        {"assistant", "Hi!"}};

    std::string result = tmpl->apply(messages, true);

    // Gemma uses "model" instead of "assistant"
    EXPECT_NE(result.find("<start_of_turn>model"), std::string::npos);
    EXPECT_EQ(result.find("<start_of_turn>assistant"), std::string::npos);
}

TEST(Test__ChatTemplate, ApplyGemmaMergesSystemWithUser)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::GEMMA);

    std::vector<ChatMessage> messages = {
        {"system", "Be helpful."},
        {"user", "Hello!"}};

    std::string result = tmpl->apply(messages, true);

    // System should be merged with user message
    size_t user_start = result.find("<start_of_turn>user");
    EXPECT_NE(user_start, std::string::npos);

    // Both system content and user content should appear after user tag
    std::string after_user = result.substr(user_start);
    EXPECT_NE(after_user.find("Be helpful"), std::string::npos);
    EXPECT_NE(after_user.find("Hello!"), std::string::npos);
}

TEST(Test__ChatTemplate, ApplyUnknownFallback)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::UNKNOWN);

    std::vector<ChatMessage> messages = {
        {"system", "Be helpful."},
        {"user", "Hello!"}};

    std::string result = tmpl->apply(messages, true);

    // Fallback should use simple prefixes
    EXPECT_NE(result.find("System: Be helpful"), std::string::npos);
    EXPECT_NE(result.find("User: Hello"), std::string::npos);
    EXPECT_NE(result.find("Assistant:"), std::string::npos);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Test__ChatTemplate, HandlesEmptyMessages)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages;
    std::string result = tmpl->apply(messages, true);

    // Should still add generation prompt
    EXPECT_NE(result.find("<|im_start|>assistant"), std::string::npos);
}

TEST(Test__ChatTemplate, HandlesEmptyContent)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"user", ""}};

    std::string result = tmpl->apply(messages, true);

    // Should still format correctly with empty content
    EXPECT_NE(result.find("<|im_start|>user"), std::string::npos);
    EXPECT_NE(result.find("<|im_end|>"), std::string::npos);
}

TEST(Test__ChatTemplate, HandlesSpecialCharacters)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"user", "Hello <world> & \"friends\"!"}};

    std::string result = tmpl->apply(messages, true);

    // Special characters should be preserved as-is
    EXPECT_NE(result.find("<world>"), std::string::npos);
    EXPECT_NE(result.find("& \"friends\""), std::string::npos);
}

TEST(Test__ChatTemplate, HandlesUnicode)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"user", "こんにちは! 🎉 مرحبا"}};

    std::string result = tmpl->apply(messages, true);

    // Unicode should be preserved
    EXPECT_NE(result.find("こんにちは"), std::string::npos);
    EXPECT_NE(result.find("🎉"), std::string::npos);
    EXPECT_NE(result.find("مرحبا"), std::string::npos);
}

TEST(Test__ChatTemplate, HandlesNewlines)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::CHATML);

    std::vector<ChatMessage> messages = {
        {"user", "Line 1\nLine 2\nLine 3"}};

    std::string result = tmpl->apply(messages, true);

    // Newlines should be preserved
    EXPECT_NE(result.find("Line 1\nLine 2\nLine 3"), std::string::npos);
}

// ============================================================================
// Type Name Tests
// ============================================================================

TEST(Test__ChatTemplate, TypeNameReturnsCorrectStrings)
{
    EXPECT_STREQ(chatTemplateTypeName(ChatTemplateType::CHATML), "ChatML");
    EXPECT_STREQ(chatTemplateTypeName(ChatTemplateType::LLAMA3), "LLaMA 3");
    EXPECT_STREQ(chatTemplateTypeName(ChatTemplateType::LLAMA2), "LLaMA 2");
    EXPECT_STREQ(chatTemplateTypeName(ChatTemplateType::MISTRAL_V1), "Mistral v1");
    EXPECT_STREQ(chatTemplateTypeName(ChatTemplateType::GEMMA), "Gemma");
    EXPECT_STREQ(chatTemplateTypeName(ChatTemplateType::UNKNOWN), "Unknown");
}

// ============================================================================
// Factory Method Tests
// ============================================================================

TEST(Test__ChatTemplate, CreateFromTypeWorks)
{
    auto tmpl = ChatTemplate::create(ChatTemplateType::LLAMA3);

    EXPECT_EQ(tmpl->type(), ChatTemplateType::LLAMA3);
    EXPECT_TRUE(tmpl->rawTemplate().empty());
}

TEST(Test__ChatTemplate, CreateFromStringPreservesRawTemplate)
{
    std::string template_str = "<|im_start|>test<|im_end|>";
    auto tmpl = ChatTemplate::create(template_str);

    EXPECT_EQ(tmpl->rawTemplate(), template_str);
}
