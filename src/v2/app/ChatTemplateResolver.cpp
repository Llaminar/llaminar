/**
 * @file ChatTemplateResolver.cpp
 * @brief Resolves chat template override strings to ChatTemplateType
 */

#include "app/ChatTemplateResolver.h"
#include "utils/Logger.h"
#include "utils/Tokenizer.h"
#include <algorithm>

namespace llaminar2
{

    void ChatTemplateResolver::resolve(const std::string &override_str,
                                       const std::shared_ptr<ITokenizer> &tokenizer,
                                       int rank)
    {
        if (override_str.empty())
            return;

        ChatTemplateType override_type = ChatTemplateType::UNKNOWN;
        std::string tmpl_lower = override_str;
        std::transform(tmpl_lower.begin(), tmpl_lower.end(), tmpl_lower.begin(), ::tolower);

        if (tmpl_lower == "chatml")
            override_type = ChatTemplateType::CHATML;
        else if (tmpl_lower == "llama3")
            override_type = ChatTemplateType::LLAMA3;
        else if (tmpl_lower == "llama2")
            override_type = ChatTemplateType::LLAMA2;
        else if (tmpl_lower == "mistral" || tmpl_lower == "mistral_v1")
            override_type = ChatTemplateType::MISTRAL_V1;
        else if (tmpl_lower == "mistral_v3")
            override_type = ChatTemplateType::MISTRAL_V3;
        else if (tmpl_lower == "mistral_v7")
            override_type = ChatTemplateType::MISTRAL_V7;
        else if (tmpl_lower == "phi3")
            override_type = ChatTemplateType::PHI3;
        else if (tmpl_lower == "phi4")
            override_type = ChatTemplateType::PHI4;
        else if (tmpl_lower == "gemma")
            override_type = ChatTemplateType::GEMMA;
        else if (tmpl_lower == "deepseek")
            override_type = ChatTemplateType::DEEPSEEK;
        else if (tmpl_lower == "deepseek2")
            override_type = ChatTemplateType::DEEPSEEK2;
        else if (tmpl_lower == "deepseek3")
            override_type = ChatTemplateType::DEEPSEEK3;
        else if (tmpl_lower == "zephyr")
            override_type = ChatTemplateType::ZEPHYR;
        else if (tmpl_lower == "vicuna")
            override_type = ChatTemplateType::VICUNA;
        else if (tmpl_lower == "command_r" || tmpl_lower == "command-r")
            override_type = ChatTemplateType::COMMAND_R;
        else
        {
            if (rank == 0)
            {
                LOG_WARN("Unknown chat template '" << override_str << "', using model's template");
            }
            return;
        }

        tokenizer->setChatTemplate(ChatTemplate::create(override_type));
        if (rank == 0)
        {
            LOG_DEBUG("Using chat template override: " << override_str);
        }
    }

} // namespace llaminar2
