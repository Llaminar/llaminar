/**
 * @file QwenThinkingPolicy.h
 * @brief Shared model-owned continuation text for Qwen thinking-budget closure.
 *
 * Budget exhaustion can interrupt arbitrary reasoning text. The documented
 * paragraph boundary is part of the token sequence, not display formatting:
 * omitting it can join the instruction to a partial heading and induce answer
 * loops in both native inference and the independent Hugging Face reference.
 * Schemas expose this policy; HTTP and other callers must preserve its bytes.
 */
#pragma once

#include <string>

namespace llaminar2
{
    /**
     * @brief Return Qwen's thinking-budget continuation including its boundaries.
     * @return Assistant continuation text to encode without sequence BOS/EOS.
     *
     * The leading blank paragraph and space separate the instruction from any
     * interrupted reasoning. The closing marker returns to answer generation;
     * it is not EOS and must not terminate the inference request.
     * @see https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html#thinking-budget
     */
    inline std::string qwenStopThinkingPrompt()
    {
        return "\n\n Considering the limited time by the user, I have to give the "
               "solution based on the thinking directly now.\n</think>\n\n";
    }
}
