/**
 * @file GraphBuilderRegistry.cpp
 * @brief Implementation of GraphBuilderRegistry
 * @author David Sanftenberg
 * @date April 2026
 */

#include "GraphBuilderRegistry.h"
#include "../../../models/ModelRegistrations.h"
#include "utils/Logger.h"

#include <algorithm>
#include <stdexcept>

namespace llaminar2
{

    std::unordered_map<std::string, GraphBuilderRegistry::Registration> &
    GraphBuilderRegistry::registry()
    {
        static std::unordered_map<std::string, Registration> s_registry;
        return s_registry;
    }

    std::mutex &GraphBuilderRegistry::mutex()
    {
        static std::mutex s_mutex;
        return s_mutex;
    }

    std::string GraphBuilderRegistry::normalize(const std::string &architecture)
    {
        std::string result = architecture;
        std::transform(result.begin(), result.end(), result.begin(),
                       [](unsigned char c)
                       { return std::tolower(c); });
        return result;
    }

    void GraphBuilderRegistry::registerFactory(const std::string &architecture, FactoryFn factory,
                                               ArrivalDeclarationFn arrival_declaration)
    {
        std::lock_guard<std::mutex> lock(mutex());
        std::string key = normalize(architecture);

        auto &reg = registry();
        if (reg.find(key) != reg.end())
        {
            LOG_WARN("[GraphBuilderRegistry] Overwriting existing factory for '" << key << "'");
        }

        reg[key] = {std::move(factory), arrival_declaration};
        LOG_DEBUG("[GraphBuilderRegistry] Registered graph builder for '" << key << "'");
    }

    void GraphBuilderRegistry::ensureBuiltins()
    {
        registerBuiltinModels();
    }

    std::shared_ptr<IGraphBuilder> GraphBuilderRegistry::create(
        const std::string &architecture,
        const GraphConfig &config,
        std::shared_ptr<IMPIContext> mpi_ctx)
    {
        ensureBuiltins();
        std::lock_guard<std::mutex> lock(mutex());
        std::string key = normalize(architecture);

        auto &reg = registry();
        auto it = reg.find(key);
        if (it == reg.end())
        {
            std::string supported;
            for (const auto &[name, _] : reg)
            {
                if (!supported.empty())
                    supported += ", ";
                supported += name;
            }

            throw std::runtime_error(
                "GraphBuilderRegistry: Unsupported architecture '" + architecture + "'. "
                                                                                    "Registered architectures: [" +
                supported + "]. "
                            "To add support, register a factory in your model's .cpp file using "
                            "REGISTER_GRAPH_BUILDER(\"your_arch\", factory_fn).");
        }

        return it->second.factory(config, std::move(mpi_ctx));
    }

    GraphBuilderRegistry::ArrivalDeclarationFn
    GraphBuilderRegistry::requireArrivalDeclaration(const std::string &architecture)
    {
        ensureBuiltins();
        std::lock_guard<std::mutex> lock(mutex());
        const auto registration = registry().find(normalize(architecture));
        if (registration == registry().end() || !registration->second.arrival_declaration)
            throw std::invalid_argument("CUDA ExpertOverlay graph builder '" + architecture +
                "' has no native arrival-site declaration for physical-memory admission");
        return registration->second.arrival_declaration;
    }

    bool GraphBuilderRegistry::isSupported(const std::string &architecture)
    {
        ensureBuiltins();
        std::lock_guard<std::mutex> lock(mutex());
        return registry().count(normalize(architecture)) > 0;
    }

    std::vector<std::string> GraphBuilderRegistry::supportedArchitectures()
    {
        ensureBuiltins();
        std::lock_guard<std::mutex> lock(mutex());
        std::vector<std::string> result;
        result.reserve(registry().size());
        for (const auto &[name, _] : registry())
            result.push_back(name);
        std::sort(result.begin(), result.end());
        return result;
    }

} // namespace llaminar2
