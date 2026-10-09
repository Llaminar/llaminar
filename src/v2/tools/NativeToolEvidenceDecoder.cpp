/**
 * @file NativeToolEvidenceDecoder.cpp
 * @brief Decode retained token IDs without model execution or tensor payload reads.
 *
 * The image-bound OpenCode audit loads tokenizer metadata and the same immutable
 * model generation policy as serving. Complete answer/reasoning bytes and native
 * tool content are exported alongside a production streaming-parser replay. The
 * Python auditor independently authenticates XML argument values; replay alone
 * never certifies the parser which produced the HTTP response.
 */
#include "app/modes/ChatCompletionHandler.h"
#include "execution/local_execution/graph/SchemaFactoryRegistry.h"
#include "loaders/ModelContext.h"
#include "models/ModelGenerationPolicy.h"
#include "utils/Tokenizer.h"
#include "utils/ToolCallParser.h"
#include <cstdint>
#include <iostream>
#include <string_view>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

/**
 * @brief Decode a complete JSON request array from stdin into one JSON result.
 * @param argc Argument count; a GGUF path and optional --describe-model mode.
 * @param argv Executable name, optional metadata mode, and the served GGUF path.
 * @return Zero on complete evidence, nonzero with a diagnostic on any failure.
 */
int main(int argc, char **argv)
{
    try
    {
        const bool describe = argc == 3 && std::string_view(argv[1]) == "--describe-model";
        if (argc != 2 && !describe)
            throw std::invalid_argument("Expected the served GGUF metadata path");
        const auto context = llaminar2::ModelContext::create(argv[describe ? 2 : 1]);
        if (!context)
            throw std::runtime_error("Unable to load model metadata");
        const auto tokenizer = llaminar2::createTokenizer(context);
        if (!tokenizer)
            throw std::runtime_error("Unable to create the model tokenizer");
        const auto policy = llaminar2::ModelGenerationPolicy::fromMetadata(*context->loader());
        policy.applyChatTemplate(*tokenizer);
        const auto factory = llaminar2::SchemaFactoryRegistry::getFactory(context->architecture());
        if (!factory || factory->getToolCallFormat() != llaminar2::ToolCallFormat::QWEN_3_XML)
            throw std::invalid_argument("Native OpenCode evidence requires the Qwen XML grammar");
        if (describe)
        {
            // Model-owned training context and the production cache block
            // geometry bound PMA admission probes. No tensor payload is read.
            const auto context_tokens = context->loader()->contextLength();
            if (context_tokens == 0 || context_tokens > static_cast<uint64_t>(INT32_MAX))
                throw std::invalid_argument("Model metadata has no supported training context");
            std::cout << nlohmann::json({{"schema", 1}, {"architecture", context->architecture()},
                {"context_tokens", context_tokens},
                {"context_alignment_tokens", llaminar2::PrefixCacheRuntimeConfig{}.block_size},
                {"default_thinking", policy.defaultThinkingMode() == llaminar2::ThinkingMode::Enabled},
                {"tool_format", "qwen_3_xml"}}).dump() << '\n';
            return 0;
        }
        nlohmann::json input;
        std::cin >> input;
        if (!input.is_array() || input.empty())
            throw std::invalid_argument("Expected a nonempty native response array");
        auto output = nlohmann::json::array();
        for (const auto &request : input)
        {
            const auto tokens = request.at("tokens").get<std::vector<int>>();
            const auto raw = tokenizer->decode(tokens, false);
            const auto &thinking_override = request.at("enable_thinking");
            if (!thinking_override.is_null() && !thinking_override.is_boolean())
                throw std::invalid_argument("Reasoning override must be boolean or null");
            const bool thinking = thinking_override.is_null()
                ? policy.defaultThinkingMode() == llaminar2::ThinkingMode::Enabled
                : thinking_override.get<bool>();
            std::string reasoning, native_content;
            if (thinking && tokenizer->hasChatTemplate() && tokenizer->getChatTemplate().isThinkingModel())
            {
                llaminar2::StreamingThinkSplitter splitter(tokenizer->getChatTemplate().thinkingEndTag());
                for (const auto &part : {splitter.process(raw), splitter.flush()})
                    (part.field == "content" ? native_content : reasoning) += part.text;
            }
            else
            {
                native_content = raw;
            }
            llaminar2::StreamingToolCallSplitter parser(factory->getToolCallFormat(), request.at("tools"));
            auto events = parser.process(native_content);
            auto terminal = parser.flush();
            events.insert(events.end(), std::make_move_iterator(terminal.begin()), std::make_move_iterator(terminal.end()));
            auto calls = nlohmann::json::array();
            std::string content;
            for (const auto &event : events)
            {
                if (event.kind == llaminar2::StreamingToolCallSplitter::Event::Kind::Content)
                    content += event.content;
                else
                    calls.push_back({{"name", event.tool_call.name},
                                     {"arguments", nlohmann::json::parse(event.tool_call.arguments)}});
            }
            output.push_back({{"id", request.at("id")}, {"tool_format", "qwen_3_xml"}, {"raw", raw},
                              {"reasoning", reasoning}, {"native_content", native_content},
                              {"calls", calls}, {"content", content}});
        }
        std::cout << output.dump() << '\n';
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Native tool evidence decode failed: " << error.what() << '\n';
        return 1;
    }
}
