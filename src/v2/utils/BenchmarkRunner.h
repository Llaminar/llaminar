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
 *   BenchmarkRunner runner(pipeline, tokenizer);
 *   BenchmarkResult result = runner.run(args);
 *   runner.printResults(result);
 */

#pragma once

#include "Tokenizer.h"
#include "Sampler.h"
#include "../execution/local_execution/orchestrators/IInferenceRunner.h"
#include "../config/OrchestrationConfig.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

namespace llaminar2
{

    /**
     * @brief Origin of the exact text tokenized by a benchmark run.
     *
     * The source is part of the benchmark result contract. In particular, an
     * inline prompt that happens to match an old default sentinel remains an
     * inline prompt; no prompt value is interpreted as an implicit mode switch.
     */
    enum class BenchmarkPromptSource : uint8_t
    {
        BuiltIn, ///< No prompt option was supplied; use the stable built-in corpus.
        Inline,  ///< Prompt bytes came directly from `-p` / `--prompt`.
        File     ///< Prompt bytes came from `--prompt-file`.
    };

    /**
     * @brief Fully resolved, content-addressed benchmark prompt.
     *
     * Resolution happens before benchmark timing begins. File input is read in
     * binary mode so trailing newlines and every other byte remain significant.
     * The digest lets benchmark artifacts prove that their prompt inputs match
     * without embedding potentially sensitive prompt text in logs or JSON.
     */
    struct ResolvedBenchmarkPrompt
    {
        BenchmarkPromptSource source = BenchmarkPromptSource::BuiltIn; ///< Selected source kind.
        std::string text;      ///< Exact bytes passed to the tokenizer.
        std::string file_path; ///< User-supplied path for file-backed input.
        std::string sha256;    ///< SHA-256 of `text`, lowercase hexadecimal.
    };

    /**
     * @brief Return the stable diagnostic name for a prompt source.
     * @param source Source kind to describe.
     * @return `built_in`, `inline`, or `file`.
     */
    const char *benchmarkPromptSourceToString(BenchmarkPromptSource source) noexcept;

    /**
     * @brief Resolve and authenticate the exact prompt for a benchmark run.
     *
     * `--prompt` and `--prompt-file` are mutually exclusive. Empty explicit
     * input, unreadable files, and failed content authentication throw instead
     * of silently selecting the built-in corpus.
     *
     * @param config Parsed benchmark configuration.
     * @return Exact prompt bytes together with source and SHA-256 identity.
     * @throws std::invalid_argument for ambiguous or empty prompt options.
     * @throws std::runtime_error for file I/O or SHA-256 failures.
     */
    ResolvedBenchmarkPrompt resolveBenchmarkPrompt(const OrchestrationConfig &config);

    /**
     * @brief Inter-step overhead profiling data from the decode loop.
     *
     * Tracks time spent BETWEEN forward() calls: sampling, token broadcast,
     * and other loop housekeeping. Structured timings are exported in the
     * `decode_loop` PerfStats domain; the human table is legacy-only.
     */
    struct DecodeLoopProfile
    {
        double sampler_total_us = 0.0;    ///< Total sampling time (argmax or GPU argmax)
        double inter_step_total_us = 0.0; ///< Total time between forward() return and next forward() call
        int decode_tokens = 0;            ///< Number of decode iterations measured

        bool empty() const { return decode_tokens == 0; }
    };

    /**
     * @brief One contiguous decode interval aligned to the maintenance cadence.
     *
     * Latencies are partitioned after inference from samples the benchmark
     * already owns, so this evidence adds no work to the production token loop.
     * A window is a timing coordinate, not a claim that asynchronous movement
     * completed exactly on its final token.
     */
    struct BenchmarkDecodeWindowResult
    {
        int start_token = 0; ///< Zero-based emitted-token offset.
        int token_count = 0; ///< Samples in this interval.
        double time_ms = 0.0; ///< Sum of emitted-token wall latencies.
        double tokens_per_sec = 0.0; ///< Interval-local generation rate.
    };

    /**
     * @brief One measured request in a repeated production benchmark.
     *
     * Aggregate throughput is useful for a stable headline number, but it
     * erases the trajectory of an adaptive placement policy.  These records
     * retain exact request timings together with completed ExpertOverlay
     * movement evidence delimited by the request boundaries. The counters are
     * a passive projection of the sole host/device optimization authority and
     * therefore remain available when PerfStats is disabled without downloading
     * a device owner map or synchronizing an inference stream.
     */
    struct BenchmarkIterationResult
    {
        int iteration = 0; ///< One-based measured iteration ordinal.
        int prefill_tokens = 0; ///< Real prompt tokens processed this request.
        double prefill_time_ms = 0.0; ///< Wall time for production prefill.
        double prefill_tokens_per_sec = 0.0; ///< Request-local prefill rate.
        int decode_tokens = 0; ///< Tokens emitted by production decode.
        double decode_time_ms = 0.0; ///< Wall time for production decode.
        double decode_tokens_per_sec = 0.0; ///< Request-local decode rate.
        int decode_after_prefill_tokens = 0; ///< Emitted tokens excluding the terminal-prefill sample.
        /**
         * Request-local generation rate after excluding the one output per
         * logical request already paid for by prefill. This prevents a short
         * decode from presenting terminal-prefill sampling as decode compute.
         */
        double decode_after_prefill_tokens_per_sec = 0.0;

        /** Completed movement epoch immediately before this request's prefill. */
        std::uint64_t moe_runtime_movement_epoch_start = 0;
        /** Completed movement epoch observed after this request's decode. */
        std::uint64_t moe_runtime_movement_epoch = 0;

        /** Completed durable movement transactions during this request. */
        std::uint64_t dynamic_movement_transactions = 0;
        /** Completed device-authored movement commands during this request. */
        std::uint64_t dynamic_movement_commands = 0;
        /** Physically transferred durable expert bytes during this request. */
        std::uint64_t dynamic_physical_bytes = 0;
        /** Cross-priority moves toward a smaller integer priority. */
        std::uint64_t dynamic_promotions = 0;
        /** Cross-priority moves toward a larger integer priority. */
        std::uint64_t dynamic_demotions = 0;
        /** Equal-priority moves that rebalance participant skew. */
        std::uint64_t dynamic_same_priority_moves = 0;

        /** Decode trajectory partitioned by the configured maintenance window. */
        std::vector<BenchmarkDecodeWindowResult> decode_windows;
    };

    /**
     * @brief Results from a benchmark run
     */
    struct BenchmarkResult
    {
        int measurement_iterations = 3; ///< Number of measured benchmark iterations averaged.
        int warmup_iterations = 1;      ///< Number of pre-measurement warmup iterations.

        // Exact benchmark input identity (the prompt text itself is deliberately omitted).
        BenchmarkPromptSource prompt_source = BenchmarkPromptSource::BuiltIn;
        std::string prompt_file_path; ///< Source path when `prompt_source == File`.
        size_t prompt_bytes = 0;      ///< Exact byte count before tokenization.
        std::string prompt_sha256;    ///< SHA-256 of the exact pre-tokenization bytes.

        // Prefill phase
        int prefill_tokens = 0;              ///< Number of tokens in prefill
        double prefill_time_ms = 0.0;        ///< Time for prefill phase (ms)
        double prefill_tokens_per_sec = 0.0; ///< Prefill throughput (tok/s)
        bool prefill_success = false;

        // Decode phase
        int decode_tokens = 0;              ///< Number of tokens generated
        double decode_time_ms = 0.0;        ///< Time for decode phase (ms)
        double decode_tokens_per_sec = 0.0; ///< Decode throughput (tok/s)
        /** Tokens whose logits were produced after the prefill boundary. */
        int decode_after_prefill_tokens = 0;
        /**
         * Throughput after removing the terminal-prefill output from each
         * logical request. This is the steady decode comparison metric; the
         * ordinary decode rate remains end-user emitted-token throughput.
         */
        double decode_after_prefill_tokens_per_sec = 0.0;
        std::vector<double> decode_token_latencies_ms; ///< Per emitted token latency samples (ms/token)
        double decode_latency_mean_ms = 0.0;            ///< Mean per-token decode latency
        double decode_latency_p50_ms = 0.0;             ///< p50 per-token decode latency
        double decode_latency_p90_ms = 0.0;             ///< p90 per-token decode latency
        bool decode_success = false;

        // Overall
        double total_time_ms = 0.0; ///< Total benchmark time (ms)
        bool success = false;
        std::string failure_reason;

        // Generated text (for verification)
        std::string generated_text;
        std::vector<int32_t> generated_token_ids;

        // Prefix-cache / MTP observability captured after the benchmark loop.
        PrefixRuntimeStateSnapshot prefix_state;

        /** Ordered measured-request trajectory for adaptive-policy analysis. */
        std::vector<BenchmarkIterationResult> iterations;
    };

    /**
     * @brief Serialize a benchmark result to stable machine-readable JSON.
     *
     * The JSON schema is intentionally compact and carries the counters needed
     * to explain prefix-cache and MTP benchmark results without parsing logs.
     *
     * @param result Benchmark result to serialize.
     * @param config Optional config used to include requested benchmark knobs.
     * @return Pretty-printed JSON document.
     */
    std::string benchmarkResultToJsonString(
        const BenchmarkResult &result,
        const OrchestrationConfig *config = nullptr);

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
     * - Compatible with an MPI-coordinated runner owned by one request controller
     * - Professional formatted output with box drawing
     */
    class BenchmarkRunner
    {
    public:
        /**
         * @brief Construct benchmark runner
         * @param runner Initialized inference runner for inference
         * @param tokenizer Tokenizer for encode/decode
         *
         * Distributed participation belongs to the supplied inference runner:
         * the application executes one BenchmarkRunner on the coordinated root
         * while non-root ranks remain in the runner's production worker loop.
         */
        BenchmarkRunner(
            std::shared_ptr<IInferenceRunner> runner,
            std::shared_ptr<ITokenizer> tokenizer);

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
        void printResults(const BenchmarkResult &result);

        /**
         * @brief Set callback invoked after warmup run completes
         * @param cb Callback (e.g., for MoE expert rebalancing)
         */
        void setPostWarmupCallback(std::function<void()> cb) { post_warmup_cb_ = std::move(cb); }

        /**
         * @brief Set callback invoked after each decode step
         * @param cb Callback (e.g., for incremental MoE expert rebalancing)
         */
        void setDecodeStepCallback(std::function<void()> cb) { decode_step_cb_ = std::move(cb); }

    private:
        std::shared_ptr<IInferenceRunner> runner_;
        std::shared_ptr<ITokenizer> tokenizer_;
        std::function<void()> post_warmup_cb_;
        std::function<void()> decode_step_cb_;
        DecodeLoopProfile decode_loop_profile_; ///< Accumulated across benchmark iterations
        SamplingParams decode_sampling_params_; ///< Sampling params used by orchestrated decodeStep()
        int decode_request_batch_ = 1;          ///< Active logical request batch for MTP benchmark decode.
        std::string last_failure_reason_;

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
         * @param ignore_stop_tokens If true, never stop on EOS/stop tokens (for throughput benchmarks)
         * @return Decode run result with timing, text, and generated token ids.
         */
        struct DecodeRunResult
        {
            bool success = false;
            double time_ms = 0.0;
            int tokens_generated = 0;
            std::string generated_text;
            std::vector<int32_t> generated_token_ids;
            std::vector<double> token_latencies_ms;
        };

        DecodeRunResult runDecode(int n_tokens, int eos_token_id, bool ignore_stop_tokens = false);
    };

} // namespace llaminar2
