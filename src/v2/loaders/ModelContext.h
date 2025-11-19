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
        /**
         * @param model_path Path to GGUF model file
         * @param mpi_ctx MPI context for distributed weight management (nullptr = single rank)
         * @param placement_map Fine-grained weight placement decisions (nullptr = default all to device 0)
         * @param factory Optional TensorFactory for NUMA-aware allocation
         * @param strategy Weight distribution strategy (default: REPLICATED)
         * @param weight_precision Weight loading precision (affects dequantization, default: NATIVE)
         * @return Shared pointer to context, or nullptr on error
         */
        static std::shared_ptr<ModelContext> create(
            const std::string &model_path,
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            std::shared_ptr<WeightPlacementMap> placement_map = nullptr,
            TensorFactory *factory = nullptr,
            WeightDistributionStrategy strategy = WeightDistributionStrategy::REPLICATED,
            WeightPrecision weight_precision = WeightPrecision::NATIVE);

        /**
         * @brief Create test-only model context (doesn't load actual model)
         *
         * For unit tests that need a ModelContext but don't need real model data.
         * Initializes a minimal valid GGUFModel structure to prevent undefined behavior.
         *
         * @param model_path Dummy path (can be anything)
         * @param mpi_ctx Optional MPI context for multi-rank tests (use nullptr for single-rank)
         * @return Shared pointer to context (always succeeds)
         */
        static std::shared_ptr<ModelContext> createForTesting(
            const std::string &model_path = "test.gguf",
            std::shared_ptr<MPIContext> mpi_ctx = nullptr,
            uint32_t block_count = 1)
        {
            // Create TensorFactory from MPI context (if provided) to prevent ModelLoader
            // from creating internal MPI_COMM_NULL context that conflicts with test's MPI_COMM_WORLD
            std::unique_ptr<TensorFactory> owned_factory;
            TensorFactory *factory = nullptr;
            if (mpi_ctx)
            {
                owned_factory = std::make_unique<TensorFactory>(*mpi_ctx);
                factory = owned_factory.get();
            }

            auto ctx = std::shared_ptr<ModelContext>(
                new ModelContext(model_path, mpi_ctx, nullptr, factory, WeightDistributionStrategy::REPLICATED));
            
            // Store owned factory so it lives as long as the context
            if (owned_factory)
            {
                ctx->owned_test_factory_ = std::move(owned_factory);
            }

            // Initialize minimal valid model structure to prevent accessing uninitialized memory
            ctx->loader_.initializeTestModel(block_count);
            return ctx;
        }        /**
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
        std::unique_ptr<TensorFactory> owned_test_factory_; // For createForTesting() only - must be declared BEFORE loader_!
        ModelLoader loader_;
        std::shared_ptr<WeightManager> weight_manager_;
    };

} // namespace llaminar2
