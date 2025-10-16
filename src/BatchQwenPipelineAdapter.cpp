/**
 * @file BatchQwenPipelineAdapter.cpp
 * @brief Factory registration for BatchQwenPipeline
 * @author David Sanftenberg
 * @date 2025-01-16
 */

#include "BatchQwenPipelineAdapter.h"
#include "BatchQwenPipeline.h"
#include "AbstractPipeline.h"
#include <memory>

namespace llaminar
{
    static std::unique_ptr<AbstractPipeline> createBatchQwen(const ModelConfig &cfg)
    {
        return std::make_unique<BatchQwenPipeline>(cfg);
    }

    void registerBatchQwenPipeline()
    {
        PipelineFactory::instance().registerCreator("qwen_batch", &createBatchQwen);
    }
}
