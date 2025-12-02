/**
 * @file ModelContext.cpp
 * @brief Model context implementation
 * @author David Sanftenberg
 */

#include "ModelContext.h"
#include "../utils/Logger.h"
#include <iostream>

namespace llaminar2
{

    ModelContext::ModelContext(const std::string &model_path,
                               std::shared_ptr<MPIContext> mpi_ctx,
                               std::shared_ptr<WeightPlacementMap> placement_map,
                               TensorFactory *factory,
                               WeightDistributionStrategy strategy)
        : model_path_(model_path), loader_(factory)
    {
        // WeightManager will be created after model is loaded (see create())
        // placement_map is passed through to WeightManager
    }

    std::shared_ptr<ModelContext> ModelContext::create(
        const std::string &model_path,
        std::shared_ptr<MPIContext> mpi_ctx,
        std::shared_ptr<WeightPlacementMap> placement_map,
        TensorFactory *factory,
        WeightDistributionStrategy strategy,
        WeightPrecision weight_precision)
    {
        // Create TensorFactory from MPI context if not provided
        // This ensures proper NUMA-aware allocation and avoids the ModelLoader
        // creating an internal single-rank MPI context
        std::unique_ptr<TensorFactory> owned_factory;
        if (!factory && mpi_ctx)
        {
            owned_factory = std::make_unique<TensorFactory>(*mpi_ctx);
            factory = owned_factory.get();
        }

        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, placement_map, factory, strategy));

        // Store owned factory so it lives as long as the context
        if (owned_factory)
        {
            ctx->owned_test_factory_ = std::move(owned_factory);
        }

        // Load model metadata
        if (!ctx->loader_.loadModel(model_path))
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path);
            return nullptr;
        }

        // Create WeightManager with loaded model and placement map
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, mpi_ctx, placement_map, strategy, weight_precision);

        return ctx;
    }

} // namespace llaminar2
