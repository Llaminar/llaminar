/**
 * @file SchemaFactoryRegistry.cpp
 * @brief Implementation of SchemaFactoryRegistry
 * @author David Sanftenberg
 * @date February 2026
 */

#include "SchemaFactoryRegistry.h"
#include "../../../models/qwen/Qwen2Schema.h"
#include "../../../models/qwen3/Qwen3Schema.h"

namespace llaminar2
{

    std::unique_ptr<ISchemaFactory> SchemaFactoryRegistry::getFactory(const std::string &architecture)
    {
        // Normalize architecture string to lowercase for comparison
        std::string arch_lower = architecture;
        std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });

        if (arch_lower == "qwen2")
        {
            return std::make_unique<Qwen2SchemaFactory>();
        }

        if (arch_lower == "qwen3")
        {
            return std::make_unique<Qwen3SchemaFactory>();
        }

        // Add more architectures here as they are implemented:
        // else if (arch_lower == "llama") {
        //     return std::make_unique<LlamaSchemaFactory>();
        // }
        // else if (arch_lower == "deepseek") {
        //     return std::make_unique<DeepSeekSchemaFactory>();
        // }

        throw std::runtime_error(
            "SchemaFactoryRegistry: Unsupported architecture '" + architecture + "'. "
                                                                                 "Supported architectures: qwen2, qwen3. "
                                                                                 "To add support for a new architecture, create a FooSchemaFactory "
                                                                                 "class and register it in SchemaFactoryRegistry::getFactory().");
    }

    WeightShardingConfig SchemaFactoryRegistry::getWeightShardingConfig(const std::string &architecture)
    {
        auto factory = getFactory(architecture);
        return factory->getWeightShardingConfig();
    }

    StageShardingConfig SchemaFactoryRegistry::getStageShardingConfig(const std::string &architecture)
    {
        auto factory = getFactory(architecture);
        return factory->getStageShardingConfig();
    }

    bool SchemaFactoryRegistry::isSupported(const std::string &architecture)
    {
        std::string arch_lower = architecture;
        std::transform(arch_lower.begin(), arch_lower.end(), arch_lower.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });

        return arch_lower == "qwen2" || arch_lower == "qwen3";
        // Add more architectures as they are implemented:
        // || arch_lower == "llama"
        // || arch_lower == "deepseek"
    }

    std::vector<std::string> SchemaFactoryRegistry::supportedArchitectures()
    {
        return {"qwen2", "qwen3"};
        // Add more as implemented:
        // return {"qwen2", "llama", "deepseek"};
    }

} // namespace llaminar2
