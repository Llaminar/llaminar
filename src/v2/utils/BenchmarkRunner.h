/**
 * @file BenchmarkRunner.h
 * @brief Benchmark runner for measuring prefill and decode performance
 * @author David Sanftenberg
 * @date 2025
 *
 * Provides clean performance measurement for LLM inference:
 * - Separate prefill/decode timing
 * - Throughput metrics (tokens/second)
 * - Minimal logging during benchmark for accurate timing
 * - Greedy sampling for deterministic, reproducible results
 *
 * Usage:
 *   BenchmarkRunner runner(pipeline, tokenizer, mpi_ctx);
 *   BenchmarkResult result = runner.run(args);
 *   runner.printResults(result);
 */

#pragma once

#include "MPIContext.h"
#include "Tokenizer.h"
#include "Sampler.h"
#include "../execution/local_execution/orchestrators/IInferenceRunner.h"
#include "../config/OrchestrationConfig.h"
#include <memory>
#include <string>
#include <vector>
#include <chrono>

namespace llaminar2
{

    /**
     * @brief Results from a benchmark run
     */
    struct BenchmarkResult
    {
        // Prefill phase
        int prefill_tokens = 0;              ///< Number of tokens in prefill
        double prefill_time_ms = 0.0;        ///< Time for prefill phase (ms)
        double prefill_tokens_per_sec = 0.0; ///< Prefill throughput (tok/s)
        bool prefill_success = false;

        // Decode phase
        int decode_tokens = 0;              ///< Number of tokens generated
        double decode_time_ms = 0.0;        ///< Time for decode phase (ms)
        double decode_tokens_per_sec = 0.0; ///< Decode throughput (tok/s)
        bool decode_success = false;

        // Overall
        double total_time_ms = 0.0; ///< Total benchmark time (ms)
        bool success = false;

        // Generated text (for verification)
        std::string generated_text;
    };

    /**
     * @brief Benchmark runner for prefill and decode performance measurement
     *
     * Measures performance in two distinct phases:
     *
     * 1. **Prefill Phase**: Process the initial prompt tokens in parallel.
     *    This is the "time to first token" metric.
     *
     * 2. **Decode Phase**: Generate tokens one at a time autoregressively.
     *    This measures incremental generation throughput.
     *
     * Features:
     * - Clean output with minimal logging during measurement
     * - Greedy sampling for deterministic results
     * - MPI-aware (all ranks participate, rank 0 reports)
     * - Professional formatted output with box drawing
     */
    class BenchmarkRunner
    {
    public:
        /**
         * @brief Construct benchmark runner
         * @param runner Initialized inference runner for inference
         * @param tokenizer Tokenizer for encode/decode
         * @param mpi_ctx MPI context for distributed execution
         */
        BenchmarkRunner(
            std::shared_ptr<IInferenceRunner> runner,
            std::shared_ptr<ITokenizer> tokenizer,
            std::shared_ptr<MPIContext> mpi_ctx);

        /**
         * @brief Run the benchmark
         *
         * Executes prefill and decode phases, measuring timing for each.
         * Uses greedy sampling (temperature=0) for deterministic results.
         *
         * @param config Parsed orchestration config (prompt, n_predict, etc.)
         * @return Benchmark results with timing metrics
         */
        BenchmarkResult run(const OrchestrationConfig &config);

        /**
         * @brief Print benchmark results in a formatted table
         * @param result Benchmark results to print
         */
        void printResults(const BenchmarkResult &result) const;

    private:
        std::shared_ptr<IInferenceRunner> runner_;
        std::shared_ptr<ITokenizer> tokenizer_;
        std::shared_ptr<MPIContext> mpi_ctx_;

        /**
         * @brief Generate a default benchmark prompt if none provided
         * @return A standardized prompt for consistent benchmarking
         */
        std::string generateDefaultPrompt() const;

        /**
         * @brief Run prefill phase and measure timing
         * @param tokens Input token IDs
         * @return Pair of (success, time_ms)
         */
        std::pair<bool, double> runPrefill(const std::vector<int> &tokens);

        /**
         * @brief Run decode phase and measure timing
         * @param n_tokens Number of tokens to generate
         * @param eos_token_id EOS token ID for early stopping
         * @return Tuple of (success, time_ms, tokens_generated, generated_text)
         */
        std::tuple<bool, double, int, std::string> runDecode(int n_tokens, int eos_token_id);

    };

} // namespace llaminar2
