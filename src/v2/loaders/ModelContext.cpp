/**
 * @file ModelContext.cpp
 * @brief Model context implementation
 * @author David Sanftenberg
 */

#include "ModelContext.h"
#include <iostream>

namespace llaminar2
{

    ModelContext::ModelContext(const std::string &model_path, TensorFactory *factory)
        : model_path_(model_path), loader_(factory)
    {
    }

    std::shared_ptr<ModelContext> ModelContext::create(
        const std::string &model_path,
        TensorFactory *factory)
    {
        auto ctx = std::shared_ptr<ModelContext>(new ModelContext(model_path, factory));

        // Load model metadata
        if (!ctx->loader_.loadModel(model_path))
        {
            std::cerr << "[ModelContext] Failed to load model: " << model_path << std::endl;
            return nullptr;
        }

        return ctx;
    }

} // namespace llaminar2
