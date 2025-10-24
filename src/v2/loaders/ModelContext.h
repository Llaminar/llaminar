/**
 * @file ModelContext.h
 * @brief Model context for pipeline initialization
 *
 * Wraps GGUF model metadata and provides convenient access for pipelines.
 * Eliminates need for pipelines to hardcode architecture parameters.
 *
 * @author David Sanftenberg
 */

#pragma once

#include "ModelLoader.h"
#include <memory>
#include <string>

namespace llaminar2
{

    /**
     * @brief Model context for pipeline initialization
     *
     * Contains:
     * - Model file path
     * - Parsed GGUF metadata (architecture, hyperparameters)
     * - ModelLoader for on-demand tensor loading
     *
     * Usage:
     *   auto ctx = ModelContext::create("model.gguf");
     *   auto pipeline = PipelineFactory::create(ctx->architecture(), ctx, mpi_ctx, device_idx);
     *   // Pipeline reads hyperparameters from ctx->model()
     */
    class ModelContext
    {
    public:
        /**
         * @brief Create model context from GGUF file
         *
         * @param model_path Path to GGUF model file
         * @param factory Optional TensorFactory for NUMA-aware allocation
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> create(
            const std::string &model_path,
            TensorFactory *factory = nullptr);

        /**
         * @brief Create test-only model context (doesn't load actual model)
         *
         * For unit tests that need a ModelContext but don't need real model data.
         *
         * @param model_path Dummy path (can be anything)
         * @return Shared pointer to context (always succeeds)
         */
        static std::shared_ptr<ModelContext> createForTesting(
            const std::string &model_path = "test.gguf")
        {
            return std::shared_ptr<ModelContext>(new ModelContext(model_path, nullptr));
        }

        /**
         * @brief Get model file path
         */
        const std::string &path() const { return model_path_; }

        /**
         * @brief Get GGUF model metadata
         */
        const GGUFModel &model() const { return loader_.getModel(); }

        /**
         * @brief Get architecture string
         */
        const std::string &architecture() const { return loader_.getModel().architecture; }

        /**
         * @brief Get ModelLoader for tensor loading
         */
        ModelLoader &loader() { return loader_; }
        const ModelLoader &loader() const { return loader_; }

    private:
        // Private constructor - use create() factory method
        explicit ModelContext(const std::string &model_path, TensorFactory *factory = nullptr);

        std::string model_path_;
        ModelLoader loader_;
    };

} // namespace llaminar2
