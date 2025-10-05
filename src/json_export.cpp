/**
 * @file json_export.cpp
 * @brief Implementation of JSON export utilities
 * @author David Sanftenberg
 */

#include "json_export.h"
#include "logger.h"
#include <iomanip>

namespace llaminar
{
    bool exportToJson(const std::string &filepath, const GenerationData &data)
    {
        try
        {
            std::ofstream out(filepath);
            if (!out.is_open())
            {
                LOG_ERROR("Failed to open JSON output file: " << filepath);
                return false;
            }

            // Write JSON with proper formatting
            out << "{\n";

            // Prompt tokens
            out << "  \"prompt_tokens\": [";
            for (size_t i = 0; i < data.prompt_tokens.size(); ++i)
            {
                if (i > 0)
                    out << ", ";
                out << data.prompt_tokens[i];
            }
            out << "],\n";

            // Generated tokens
            out << "  \"generated_tokens\": [";
            for (size_t i = 0; i < data.generated_tokens.size(); ++i)
            {
                if (i > 0)
                    out << ", ";
                out << data.generated_tokens[i];
            }
            out << "],\n";

            // Logits (array of arrays)
            out << "  \"logits\": [\n";
            for (size_t step = 0; step < data.logits.size(); ++step)
            {
                out << "    [";
                const auto &step_logits = data.logits[step];
                for (size_t i = 0; i < step_logits.size(); ++i)
                {
                    if (i > 0)
                        out << ", ";
                    // Use scientific notation for compactness
                    out << std::scientific << std::setprecision(6) << step_logits[i];
                }
                out << "]";
                if (step + 1 < data.logits.size())
                {
                    out << ",";
                }
                out << "\n";
            }
            out << "  ]\n";
            out << "}\n";

            out.close();
            LOG_INFO("Exported generation data to " << filepath
                                                    << " (" << data.generated_tokens.size() << " tokens, "
                                                    << data.logits.size() << " logit arrays)");
            return true;
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Failed to export JSON: " << e.what());
            return false;
        }
    }

} // namespace llaminar
