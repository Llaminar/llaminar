/**
 * @file BatchQwenPipelineAdapter.h
 * @brief Factory registration for BatchQwenPipeline
 * @author David Sanftenberg
 * @date 2025-01-16
 */
#pragma once

namespace llaminar
{
    /**
     * @brief Register BatchQwenPipeline with PipelineFactory under "qwen_batch"
     *
     * This allows factory-based instantiation of the true batched pipeline implementation.
     * The legacy "qwen" registration points to QwenPipelineAdapter (sequential fallback).
     */
    void registerBatchQwenPipeline();
}
