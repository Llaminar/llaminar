/**
 * @file SequentialBatchBenchmark.h
 * @brief Sequential batch benchmark for measuring throughput across multiple sequences
 * @author David Sanftenberg
 * @date 2025-10-15
 *
 * This implements a simple but effective batching strategy:
 * - Process N sequences sequentially (not in parallel)
 * - Weights are cached and reused across sequences (via weight caching)
 * - Measure aggregate throughput = total_tokens / total_time
 *
 * This demonstrates the benefit of weight reuse without complex tensor reshaping.
 */

#pragma once

#include <memory>
#include <vector>
#include <string>
#include "AbstractPipeline.h"
#include "QwenPipelineAdapter.h"
#include "chat/TokenizerInterface.h"
#include "ArgumentParser.h"
#include "BenchmarkRunner.h"

namespace llaminar
{
    namespace benchmark
    {

        /**
         * @brief Metrics for sequential batch processing
         */
        struct SequentialBatchMetrics
        {
            int batch_size = 0;
            int total_sequences_processed = 0;

            // Per-phase aggregates
            int total_prefill_tokens = 0;
            double total_prefill_time_ms = 0.0;

            int total_decode_tokens = 0;
            double total_decode_time_ms = 0.0;

            // Aggregate throughput
            double aggregate_tokens_per_sec = 0.0;

            // Per-sequence average
            double avg_tokens_per_sequence = 0.0;
            double avg_time_per_sequence_ms = 0.0;

            void print() const;
        };

        /**
         * @brief Run sequential batch benchmark
         *
         * Processes multiple sequences one after another, leveraging weight caching
         * to demonstrate batching benefits without complex parallelization.
         *
         * @param pipeline The model pipeline
         * @param weights The model weights
         * @param tokenizer The tokenizer
         * @param prompts Vector of prompts to process
         * @param n_predict Number of tokens to generate per sequence
         * @return SequentialBatchMetrics with aggregate statistics
         */
        SequentialBatchMetrics runSequentialBatch(
            AbstractPipeline &pipeline,
            const QwenModelWeights &weights,
            chat::TokenizerInterface &tokenizer,
            const std::vector<std::string> &prompts,
            int n_predict);

    } // namespace benchmark
} // namespace llaminar
