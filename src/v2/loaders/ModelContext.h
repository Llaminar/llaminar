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
#include "WeightManager.h"
#include "WeightPlacementMap.h"
#include "../utils/MPIContext.h"
#include "../tensors/Tensors.h"
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
     * - WeightManager for distributed weight allocation
     *
     * Usage:
     *   auto ctx = ModelContext::create("model.gguf", mpi_ctx);
     *   auto pipeline = PipelineFactory::create(ctx->architecture(), ctx, mpi_ctx, device_idx);
     *   // Pipeline reads hyperparameters from ctx->model()
     *   // Pipeline loads weights via ctx->getWeight(name, device_idx)
     */
    class ModelContext
    {
    public:
        /**
         * @brief Create model context from GGUF file
         *
         * @param model_path Path to GGUF model file
         * @param mpi_ctx MPI context for distributed weight management (nullptr = single rank)
         * @param placement_map Fine-grained weight placement decisions (nullptr = default all to device 0)
         * @param factory Optional TensorFactory for NUMA-aware allocation
         * @param strategy Weight distribution strategy (default: REPLICATED)
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> create(
            const std::string &model_path,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
            TensorFactory *factory = nullptr,
            WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED);

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

        /**
         * @brief Get weight tensor by name
         *
         * Loads from GGUF with appropriate distribution strategy.
         *
         * @param name GGUF tensor name (e.g., "token_embd.weight", "blk.0.attn_q.weight")
         * @param device_idx Device to place tensor on (-1 = CPU, ≥0 = GPU)
         * @return Shared pointer to tensor, or nullptr on error
         */
        std::shared_ptr<TensorBase> getWeight(const std::string &name, int device_idx = -1)
        {
            return weight_manager_->getWeight(name, device_idx);
        }

        /**
         * @brief Get weight manager
         */
        std::shared_ptr<WeightManager> weightManager() { return weight_manager_; }

    private:
        // Private constructor - use create() factory method
        private:
        /**
         * @brief Private constructor (use create() instead)
         */
        ModelContext(const std::string &model_path,
                     std::shared_ptr<MPIContext> mpi_ctx,
                     std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
                     TensorFactory *factory = nullptr,
                     WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED);

        std::string model_path_;
        ModelLoader loader_;
        std::shared_ptr<WeightManager> weight_manager_;
    };

} // namespace llaminar2
