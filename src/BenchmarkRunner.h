/**
 * @file BenchmarkRunner.h
 * @brief Clean benchmark runner for inference performance measurement
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#pragma once

#include <memory>
#include <string>
#include <chrono>
#include "AbstractPipeline.h"
#include "QwenPipelineAdapter.h"
#include "chat/TokenizerInterface.h"
#include "ArgumentParser.h"

namespace llaminar
{
    namespace benchmark
    {

        /**
         * @brief Performance metrics from a benchmark run
         */
        struct BenchmarkMetrics
        {
            // Prefill metrics
            int prefill_tokens = 0;
            double prefill_time_ms = 0.0;
            double prefill_tokens_per_sec = 0.0;

            // Decode metrics
            int decode_tokens = 0;
            double decode_time_ms = 0.0;
            double decode_tokens_per_sec = 0.0;

            // Total metrics
            double total_time_ms = 0.0;
            double total_tokens_per_sec = 0.0;

            // Model info
            std::string model_path;
            std::string backend; // "OpenBLAS", "COSMA", or "Mixed"

            /**
             * @brief Print benchmark results in a clean format
             */
            void print() const;
        };

        /**
         * @brief Run inference benchmark with minimal logging
         *
         * This function runs a single inference pass with the given prompt,
         * measuring prefill and decode performance separately.
         *
         * @param pipeline The model pipeline to use
         * @param weights The model weights
         * @param tokenizer The tokenizer interface
         * @param params The runtime parameters (prompt, n_predict, etc.)
         * @return BenchmarkMetrics containing performance data
         */
        BenchmarkMetrics runInferenceBenchmark(
            AbstractPipeline &pipeline,
            const QwenModelWeights &weights,
            chat::TokenizerInterface &tokenizer,
            const LlaminarParams &params);

    } // namespace benchmark
} // namespace llaminar
