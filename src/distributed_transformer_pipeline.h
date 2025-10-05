/**
 * @file distributed_transformer_pipeline.h
 * @brief Compatibility shim - redirects to qwen_pipeline.h
 * @deprecated Use qwen_pipeline.h directly. This file maintained for backward compatibility.
 *
 * The DistributedTransformerPipeline has been renamed to QwenPipeline to reflect
 * that it's Qwen-specific and abstracted behind base classes.
 */
#pragma once

#warning "distributed_transformer_pipeline.h is deprecated. Use qwen_pipeline.h instead."

#include "qwen_pipeline.h"

// Type aliases for backward compatibility
namespace llaminar
{
    using DistributedTransformerPipeline = QwenPipeline;

    inline std::unique_ptr<QwenPipeline> createDistributedTransformerPipeline(const ModelConfig &config)
    {
        return createQwenPipeline(config);
    }
}
