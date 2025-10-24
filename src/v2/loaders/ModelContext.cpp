/**
 * @file ModelContext.cpp
 * @brief Model context implementation
 * @author David Sanftenberg
 */

#include "ModelContext.h"
#include <iostream>

namespace llaminar2
{

    ModelContext::ModelContext(const std::string &model_path,
                               std::shared_ptr<MPIContext> mpi_ctx,
                               TensorFactory *factory,
                               WeightDistributionStrategy strategy)
        : model_path_(model_path), loader_(factory)
    {
        // WeightManager will be created after model is loaded (see create())
    }

    std::shared_ptr<ModelContext> ModelContext::create(
        const std::string &model_path,
        std::shared_ptr<MPIContext> mpi_ctx,
        TensorFactory *factory,
        WeightDistributionStrategy strategy)
    {
        auto ctx = std::shared_ptr<ModelContext>(
            new ModelContext(model_path, mpi_ctx, factory, strategy));

        // Load model metadata
        if (!ctx->loader_.loadModel(model_path))
        {
            std::cerr << "[ModelContext] Failed to load model: " << model_path << std::endl;
            return nullptr;
        }

        // Create WeightManager with loaded model
        ctx->weight_manager_ = std::make_shared<WeightManager>(
            ctx->loader_, mpi_ctx, strategy);

        return ctx;
    }

} // namespace llaminar2
