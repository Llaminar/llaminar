#pragma once

#include "config/OrchestrationConfig.h"
#include <iostream>

namespace llaminar2
{
    namespace command_validation
    {
        inline bool printConfigErrors(const OrchestrationConfig &config)
        {
            auto errors = config.validate();
            if (errors.empty())
            {
                return true;
            }

            std::cerr << "Configuration validation failed:\n";
            for (const auto &error : errors)
            {
                std::cerr << "  - " << error << "\n";
            }
            return false;
        }

        inline void printValidateOnlySuccess(const OrchestrationConfig &config)
        {
            std::cout << "Configuration is valid." << std::endl;
            if (config.explain_placement || config.verbose_level > 0)
            {
                std::cout << "\n"
                          << config.toString() << std::endl;
            }
        }
    } // namespace command_validation
} // namespace llaminar2