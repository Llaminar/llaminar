#pragma once

#include <iostream>
#include <string>

namespace llaminar2::console_output
{
    inline void printPromptAndResponseHeader(const std::string &prompt)
    {
        std::cout << "\nPrompt:\n"
                  << prompt
                  << "\n\nResponse:\n"
                  << std::flush;
    }
} // namespace llaminar2::console_output
