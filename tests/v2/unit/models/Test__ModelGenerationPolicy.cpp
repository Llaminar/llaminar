/**
 * @file Test__ModelGenerationPolicy.cpp
 * @brief Device-free regressions for revision-owned Qwen generation defaults.
 *
 * These independent numeric expectations are transcribed from the pinned model
 * cards below (Qwen2.5 publishes its values in generation_config.json). General
 * text defaults take precedence over task-specific examples and stale generation
 * configs. Test every audited size, both reasoning modes, default mode, exact
 * metadata identity and neutral penalties without loading weights or networking.
 * The pinned official Qwen3.8 template proves default reasoning and historical
 * thinking retention survive startup even though its graph architecture is qwen35.
 */
#include "models/ModelGenerationPolicy.h"
#include "../../mocks/MockModelLoader.h"
#include "../../mocks/MockTokenizer.h"
#include "app/ChatTemplateResolver.h"
#include "Qwen38ChatTemplate.generated.h"
#include <gtest/gtest.h>
#include <array>
#include <string_view>

using namespace llaminar2;

namespace
{
    /** @brief Independently specified temperature, nucleus and presence law. */
    struct ModeExpectation { float temperature, top_p, presence; };
    /** @brief One pinned official card and its general-text recommendations. */
    struct CardExpectation
    {
        const char *name;
        const char *architecture;
        bool thinking_by_default;
        ModeExpectation thinking, direct;
        float repetition;
        const char *source;
    };
    constexpr auto cards = std::to_array<CardExpectation>({
        {"Qwen2.5-0.5B-Instruct", "qwen2", false, {0.7f, 0.8f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.1f,
         "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/generation_config.json"},
        {"Qwen2.5-1.5B-Instruct", "qwen2", false, {0.7f, 0.8f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.1f,
         "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/generation_config.json"},
        {"Qwen2.5-3B-Instruct", "qwen2", false, {0.7f, 0.8f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.05f,
         "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/aa8e72537993ba99e69dfaafa59ed015b17504d1/generation_config.json"},
        {"Qwen2.5-7B-Instruct", "qwen2", false, {0.7f, 0.8f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.05f,
         "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/generation_config.json"},
        {"Qwen2.5-14B-Instruct", "qwen2", false, {0.7f, 0.8f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.05f,
         "https://huggingface.co/Qwen/Qwen2.5-14B-Instruct/blob/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8/generation_config.json"},
        {"Qwen2.5-32B-Instruct", "qwen2", false, {0.7f, 0.8f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.05f,
         "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct/blob/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd/generation_config.json"},
        {"Qwen2.5-72B-Instruct", "qwen2", false, {0.7f, 0.8f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.05f,
         "https://huggingface.co/Qwen/Qwen2.5-72B-Instruct/blob/495f39366efef23836d0cfae4fbe635880d2be31/generation_config.json"},
        {"Qwen3-0.6B", "qwen3", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/README.md"},
        {"Qwen3-1.7B", "qwen3", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-1.7B/blob/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/README.md"},
        {"Qwen3-4B", "qwen3", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/README.md"},
        {"Qwen3-8B", "qwen3", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-8B/blob/b968826d9c46dd6066d109eabc6255188de91218/README.md"},
        {"Qwen3-14B", "qwen3", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-14B/blob/40c069824f4251a91eefaf281ebe4c544efd3e18/README.md"},
        {"Qwen3-32B", "qwen3", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-32B/blob/9216db5781bf21249d130ec9da846c4624c16137/README.md"},
        {"Qwen3-30B-A3B", "qwen3moe", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39/README.md"},
        {"Qwen3-235B-A22B", "qwen3moe", true, {0.6f, 0.95f, 0.0f}, {0.7f, 0.8f, 0.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/8efa61729e24bd65b1d152b5ab5409052aa80e65/README.md"},
        {"Qwen3.5-0.8B", "qwen35", false, {1.0f, 0.95f, 1.5f}, {1.0f, 1.0f, 2.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/2fc06364715b967f1860aea9cf38778875588b17/README.md"},
        {"Qwen3.5-2B", "qwen35", false, {1.0f, 0.95f, 1.5f}, {1.0f, 1.0f, 2.0f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/README.md"},
        {"Qwen3.5-4B", "qwen35", true, {1.0f, 0.95f, 1.5f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/README.md"},
        {"Qwen3.5-9B", "qwen35", true, {1.0f, 0.95f, 1.5f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md"},
        {"Qwen3.5-27B", "qwen35", true, {1.0f, 0.95f, 1.5f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/README.md"},
        {"Qwen3.5-35B-A3B", "qwen35moe", true, {1.0f, 0.95f, 1.5f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/README.md"},
        {"Qwen3.5-122B-A10B", "qwen35moe", true, {1.0f, 0.95f, 1.5f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.5-122B-A10B/blob/dc4d348443bc740c68e2d77492492c11606384d5/README.md"},
        {"Qwen3.6-27B", "qwen35", true, {1.0f, 0.95f, 0.0f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/README.md"},
        {"Qwen3.6-35B-A3B", "qwen35moe", true, {1.0f, 0.95f, 1.5f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/README.md"},
        {"Qwen3.8-27B", "qwen35", true, {1.0f, 0.95f, 0.0f}, {0.7f, 0.8f, 1.5f}, 1.0f,
         "https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md"},
    });

    /**
     * @brief Assert the complete law, including parameters the cards leave neutral.
     * @param actual Resolved engine recommendation.
     * @param expected Independently transcribed mode-specific card values.
     * @param repetition Published multiplicative factor.
     */
    void expectLaw(const SamplingParams &actual, const ModeExpectation &expected, float repetition)
    {
        EXPECT_FLOAT_EQ(actual.temperature, expected.temperature);
        EXPECT_FLOAT_EQ(actual.top_p, expected.top_p);
        EXPECT_EQ(actual.top_k, 20);
        EXPECT_FLOAT_EQ(actual.presence_penalty, expected.presence);
        EXPECT_FLOAT_EQ(actual.frequency_penalty, 0.0f);
        EXPECT_FLOAT_EQ(actual.repetition_penalty, repetition);
        EXPECT_FLOAT_EQ(actual.dry_multiplier, 0.0f);
        EXPECT_EQ(actual.seed, 0u);
    }
}

/** @test Every card selects both modes and its documented default from GGUF metadata. */
TEST(Test__ModelGenerationPolicy, OfficialCardsCoverAllSizesModesAndMetadataForms)
{
    ASSERT_EQ(cards.size(), 25u);
    for (const auto &card : cards)
    {
        SCOPED_TRACE(card.source);
        for (int form = 0; form < 8; ++form)
        {
            SCOPED_TRACE(form);
            test::MockModelLoader loader;
            loader.setArchitecture(card.architecture);
            const std::string name = card.name;
            switch (form)
            {
            case 0: loader.setStringParam("general.name", name); break;
            case 1: loader.setStringParam("general.basename", name); break;
            case 2: loader.setStringParam("general.base_model.0.name", name); break;
            case 3: loader.setStringParam("general.base_model.0.repo_url", "https://huggingface.co/Qwen/" + name); break;
            case 4: loader.setStringParam("general.repo_url", "https://huggingface.co/Qwen/" + name + "/"); break;
            case 5:
            {
                std::string display = "  " + name + "  ";
                for (auto &c : display) if (c == '-') c = ' ';
                loader.setStringParam("general.name", display);
                break;
            }
            case 6:
            case 7:
            {
                const auto dash = name.find('-');
                std::string size = name.substr(dash + 1);
                if (size.ends_with("-Instruct"))
                {
                    size.resize(size.size() - std::string_view("-Instruct").size());
                    loader.setStringParam("general.finetune", "Instruct");
                }
                loader.setStringParam("general.basename", name.substr(0, dash) + (form == 7 ? "-" + size : ""));
                if (form == 6) loader.setStringParam("general.size_label", size);
                break;
            }
            }
            const auto policy = ModelGenerationPolicy::fromMetadata(loader);
            ::testing::StrictMock<test::MockTokenizer> tokenizer;
            const bool maintained = name.starts_with("Qwen3.5-") || name.starts_with("Qwen3.6-");
            EXPECT_CALL(tokenizer, setChatTemplate).Times(maintained ? 1 : 0);
            policy.applyChatTemplate(tokenizer);
            expectLaw(policy.forMode(ThinkingMode::Enabled), card.thinking, card.repetition);
            expectLaw(policy.forMode(ThinkingMode::Disabled), card.direct, card.repetition);
            const auto mode = card.thinking_by_default ? ThinkingMode::Enabled : ThinkingMode::Disabled;
            EXPECT_EQ(policy.defaultThinkingMode(), mode);
            expectLaw(policy.forMode(ThinkingMode::ModelDefault),
                card.thinking_by_default ? card.thinking : card.direct, card.repetition);
        }
    }
}

/** @test Current model revision owns policy even when it cites an older base model. */
TEST(Test__ModelGenerationPolicy, CurrentIdentityPrecedesBaseProvenance)
{
    for (const char *key : {"general.name", "general.basename"})
    {
        test::MockModelLoader loader;
        loader.setArchitecture("qwen35");
        loader.setStringParam(key, "Qwen3.6-27B");
        loader.setStringParam("general.base_model.0.name", "Qwen3.5-27B");
        const auto policy = ModelGenerationPolicy::fromMetadata(loader);
        expectLaw(policy.forMode(ThinkingMode::Enabled), {1.0f, 0.95f, 0.0f}, 1.0f);
    }
}

/** @test Similar names, unrelated URLs and mismatched architectures never claim a card. */
TEST(Test__ModelGenerationPolicy, UnmatchedIdentityKeepsExplicitGenericDefaults)
{
    for (const auto &[architecture, name] : std::to_array<std::pair<const char *, const char *>>({
        {"qwen2", "Qwen2.5-7B"}, {"qwen2", "Qwen2.5-Coder-7B-Instruct"},
        {"qwen2", "Qwen2.5-Math-7B-Instruct"}, {"qwen3", "Qwen3-4B-Instruct-2507"},
        {"qwen3", "Qwen3-8B-Base"}, {"qwen35", "Qwen3.80-27B"},
        {"qwen35", "Qwen3.8-270B"}, {"qwen35", "/models/Qwen3.8-27B.gguf"},
        {"llama", "Qwen3.8-27B"}, {"qwen35moe", "Qwen3.6-27B"}, {"qwen35", ""}}))
    {
        SCOPED_TRACE(std::string(architecture) + "/" + name);
        test::MockModelLoader loader;
        loader.setArchitecture(architecture);
        loader.setStringParam("general.name", name);
        loader.setStringParam("general.repo_url", "https://huggingface.co/Someone/Qwen3.8-27B");
        const auto policy = ModelGenerationPolicy::fromMetadata(loader);
        for (auto mode : {ThinkingMode::Enabled, ThinkingMode::Disabled, ThinkingMode::ModelDefault})
        {
            const auto &actual = policy.forMode(mode);
            EXPECT_FLOAT_EQ(actual.temperature, 1.0f);
            EXPECT_FLOAT_EQ(actual.top_p, 1.0f);
            EXPECT_EQ(actual.top_k, 0);
            EXPECT_FLOAT_EQ(actual.presence_penalty, 0.0f);
            EXPECT_FLOAT_EQ(actual.frequency_penalty, 0.0f);
            EXPECT_FLOAT_EQ(actual.repetition_penalty, 1.0f);
            EXPECT_FLOAT_EQ(actual.dry_multiplier, 0.0f);
        }
    }
}

/** @test Qwen3.8 preserves the model-declared xhigh and full reasoning history. */
TEST(Test__ModelGenerationPolicy, Qwen38RetainsOfficialReasoningAndHistoricalThinking)
{
    test::MockModelLoader loader;
    loader.setArchitecture("qwen35");
    loader.setStringParam("general.name", "Qwen3.8-27B");
    loader.setStringParam("general.base_model.0.name", "Qwen3.5-27B");
    auto tokenizer = std::make_shared<::testing::NiceMock<test::MockTokenizer>>();
    auto active = ChatTemplate::create(std::string(test::qwen38::kOfficialChatTemplate));
    ON_CALL(*tokenizer, setChatTemplate).WillByDefault(
        [&](std::unique_ptr<ChatTemplate> replacement) { active = std::move(replacement); });
    ModelGenerationPolicy::fromMetadata(loader).applyChatTemplate(*tokenizer);
    ChatTemplateResolver::resolve("", tokenizer, 0);
    ASSERT_TRUE(active->hasJinjaSupport());
    EXPECT_EQ(active->rawTemplate(), test::qwen38::kOfficialChatTemplate);
    ChatMessage prior{"assistant", "First answer 🙂"};
    prior.reasoning_content = "Retained earlier reasoning 👩🏽‍💻";
    const std::vector<ChatMessage> messages{
        {"system", "Build the Python application."}, {"user", "First task"}, prior,
        {"user", "Continue with the next task 🚀"}};
    for (const bool thinking : {true, false})
    {
        const auto rendered = active->apply(messages, true, thinking);
        EXPECT_NE(rendered.find(*prior.reasoning_content), std::string::npos);
        EXPECT_NE(rendered.find("First answer 🙂"), std::string::npos);
        EXPECT_NE(rendered.find("Continue with the next task 🚀"), std::string::npos);
        if (thinking)
        {
            EXPECT_NE(rendered.find("Reasoning effort is set to xhigh."), std::string::npos);
            EXPECT_TRUE(rendered.ends_with("<|im_start|>assistant\n<think>\n"));
        }
        else
            EXPECT_TRUE(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }
}

/** @test Revision defaults never override a newer or unknown model, or a user's choice. */
TEST(Test__ModelGenerationPolicy, TemplateSelectionIsRevisionOwnedAndExplicitOverridesWin)
{
    for (const auto &card : cards)
    {
        SCOPED_TRACE(card.name);
        test::MockModelLoader loader;
        loader.setArchitecture(card.architecture);
        loader.setStringParam("general.name", card.name);
        auto tokenizer = std::make_shared<::testing::NiceMock<test::MockTokenizer>>();
        auto active = ChatTemplate::create(std::string(test::qwen38::kOfficialChatTemplate));
        ON_CALL(*tokenizer, setChatTemplate).WillByDefault(
            [&](std::unique_ptr<ChatTemplate> replacement) { active = std::move(replacement); });
        ModelGenerationPolicy::fromMetadata(loader).applyChatTemplate(*tokenizer);
        const std::string_view name = card.name;
        const bool maintained = name.starts_with("Qwen3.5-") || name.starts_with("Qwen3.6-");
        EXPECT_EQ(active->rawTemplate() != test::qwen38::kOfficialChatTemplate, maintained);
        ChatTemplateResolver::resolve("llama3", tokenizer, 0);
        EXPECT_EQ(active->type(), ChatTemplateType::LLAMA3);
    }
    for (const char *name : {"Qwen3.80-27B", "Qwen3.8-270B", "/models/Qwen3.8-27B.gguf", ""})
    {
        test::MockModelLoader loader;
        loader.setArchitecture("qwen35");
        loader.setStringParam("general.name", name);
        ::testing::StrictMock<test::MockTokenizer> tokenizer;
        EXPECT_CALL(tokenizer, setChatTemplate).Times(0);
        ModelGenerationPolicy::fromMetadata(loader).applyChatTemplate(tokenizer);
    }
}
