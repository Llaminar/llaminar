/**
 * @file CloverMoEGraphConfigBuilder.cpp
 * @brief Preserve the model's exact causal geometry and participant execution policy.
 *
 * Validate before invoking shared configuration code so malformed metadata cannot
 * resize buffers or select an accidental dense/Qwen path. Canonical GGUF weights
 * already have the shared graph's norm, convolution and tiled GDN head layout.
 */
#include "CloverMoEGraphConfigBuilder.h"
#include "CloverMoEConfig.h"
#include "../../interfaces/IModelContext.h"
#include "../../loaders/IModelLoader.h"
#include <stdexcept>

namespace llaminar2
{
    bool CloverMoEGraphConfigBuilder::populateFromModelContext(IModelContext &ctx, GraphConfig &config)
    {
        const auto loader = ctx.loader();
        if (!loader || !loader->ggufDirectory())
            throw std::invalid_argument("clover_moe: source admission requires the complete GGUF directory");
        const CloverMoEConfig exact(*loader->ggufDirectory());
        if (!QwenStandardGraphConfigBuilder::populateFromModelContext(ctx, config)) return false;
        exact.populate(config);
        return true;
    }
}
