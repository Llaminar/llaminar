/**
 * @file ChatTemplateResolver.h
 * @brief Resolves chat template override strings to ChatTemplateType
 */

#pragma once

#include "utils/ChatTemplate.h"
#include <memory>
#include <string>

namespace llaminar2
{

    class ITokenizer;

    /**
     * @brief Resolves chat template override strings to ChatTemplateType
     *
     * Maps CLI strings like "chatml", "llama3" to ChatTemplateType enums
     * and applies the override to the tokenizer.
     */
    class ChatTemplateResolver
    {
    public:
        /**
         * @brief Resolve a chat template override string and apply to tokenizer
         *
         * @param override_str Template name from --chat-template CLI flag
         * @param tokenizer Tokenizer to apply the template to
         * @param rank MPI rank (for logging on rank 0 only)
         */
        static void resolve(const std::string &override_str,
                            const std::shared_ptr<ITokenizer> &tokenizer,
                            int rank);
    };

} // namespace llaminar2
