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
        ComputePrecision precision)
    {
        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, placement_map, factory, strategy));

        // Load model metadata
        if (!ctx->loader_.loadModel(model_path))
        {
            LOG_ERROR("[ModelContext] Failed to load model: " << model_path);
            return nullptr;
        }

        // Create WeightManager with loaded model and placement map
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, mpi_ctx, placement_map, strategy, precision);

        return ctx;
    }

} // namespace llaminar2
