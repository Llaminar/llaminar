/**
 * @file GraphBuilderRegistry.h
 * @brief Registry for architecture-specific IGraphBuilder factories
 * @author David Sanftenberg
 * @date April 2026
 *
 * Provides model-agnostic dispatch from architecture string to IGraphBuilder.
 * Model-specific code registers itself during static initialization, so this
 * file has ZERO includes of model-specific headers.
 *
 * Usage:
 *   auto builder = GraphBuilderRegistry::create("qwen2", config, mpi_ctx);
 *
 * Registration (in model TU):
 *   REGISTER_GRAPH_BUILDER("qwen2", [](const GraphConfig& cfg, std::shared_ptr<IMPIContext> mpi) {
 *       return std::make_shared<QwenStandardGraph>(cfg, std::move(mpi));
 *   });
 */

#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{

    // Forward declarations — NO model-specific includes
    class IGraphBuilder;
    class IMPIContext;
    struct GraphConfig;

    /**
     * @brief Registry for IGraphBuilder factory functions
     *
     * Self-registration pattern: model-specific TUs register their factory
     * functions during static initialization. Generic infrastructure (e.g.,
     * InferenceRunnerFactory) calls create() with the architecture string.
     */
    class GraphBuilderRegistry
    {
    public:
        /// Factory function signature: (GraphConfig, IMPIContext) → IGraphBuilder
        using FactoryFn = std::function<std::shared_ptr<IGraphBuilder>(
            const GraphConfig &config,
            std::shared_ptr<IMPIContext> mpi_ctx)>;

        /**
         * @brief Register a factory for an architecture
         *
         * Called during static initialization from model TUs.
         * Thread-safe (guarded by mutex for static init ordering).
         *
         * @param architecture Lowercase architecture name (e.g., "qwen2")
         * @param factory Factory function that creates IGraphBuilder instances
         */
        static void registerFactory(const std::string &architecture, FactoryFn factory);

        /**
         * @brief Create an IGraphBuilder for the given architecture
         *
         * @param architecture Architecture name (case-insensitive)
         * @param config Graph configuration
         * @param mpi_ctx MPI context (nullptr for single-rank)
         * @return Shared pointer to the created IGraphBuilder
         * @throws std::runtime_error if architecture is not registered
         */
        static std::shared_ptr<IGraphBuilder> create(
            const std::string &architecture,
            const GraphConfig &config,
            std::shared_ptr<IMPIContext> mpi_ctx);

        /**
         * @brief Check if an architecture is registered
         */
        static bool isSupported(const std::string &architecture);

        /**
         * @brief Get list of all registered architecture names
         */
        static std::vector<std::string> supportedArchitectures();

    private:
        static std::unordered_map<std::string, FactoryFn> &registry();
        static std::mutex &mutex();
        static std::string normalize(const std::string &architecture);

        /// Lazily register built-in models on first access (std::call_once)
        static void ensureBuiltins();
    };

    /**
     * @brief Helper for static self-registration of graph builders
     *
     * Usage in model TU (.cpp file):
     *   static GraphBuilderRegistrar s_qwen2_registrar("qwen2", [](auto& cfg, auto mpi) {
     *       return std::make_shared<QwenStandardGraph>(cfg, std::move(mpi));
     *   });
     */
    struct GraphBuilderRegistrar
    {
        GraphBuilderRegistrar(const std::string &architecture,
                              GraphBuilderRegistry::FactoryFn factory)
        {
            GraphBuilderRegistry::registerFactory(architecture, std::move(factory));
        }
    };

/// Convenience macro for graph builder registration
#define REGISTER_GRAPH_BUILDER(arch, factory_fn)                          \
    static ::llaminar2::GraphBuilderRegistrar s_graph_builder_reg_##arch( \
        #arch, factory_fn)

} // namespace llaminar2
