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
 * - Requested sampling policy, shared by serial and speculative decode
 *
 * Usage:
 *   BenchmarkRunner runner(pipeline, tokenizer);
 *   BenchmarkResult result = runner.run(args);
 *   runner.printResults(result);
 */

#pragma once

#include "Tokenizer.h"
#include "Sampler.h"
#include "PerfStatsCollector.h"
#include "../execution/local_execution/orchestrators/IInferenceRunner.h"
#include "../config/OrchestrationConfig.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <optional>
#include <utility>

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
     * @brief Immutable prompt bytes and their one canonical tokenizer result.
     * Benchmark campaigns prepare each natural prompt once before their first
     * request. Only BenchmarkRunner may construct this value; replay never
     * rereads an input file or substitutes another tokenization.
     */
    class PreparedBenchmarkInput final
    {
    public:
        /** @return Exact immutable prompt origin, bytes and SHA-256. */
        const ResolvedBenchmarkPrompt &prompt() const noexcept { return prompt_; }
        /** @return Exact natural token IDs, including their original order. */
        const std::vector<int> &tokenIds() const noexcept { return tokens_; }
    private:
        friend class BenchmarkRunner;
        /** @brief Seal an already validated tokenizer result with its source. */
        PreparedBenchmarkInput(ResolvedBenchmarkPrompt prompt, std::vector<int> tokens)
            : prompt_(std::move(prompt)), tokens_(std::move(tokens)) {}
        ResolvedBenchmarkPrompt prompt_;
        std::vector<int> tokens_;
    };

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

        /**
         * Exact tokens emitted by this measured production request.
         *
         * Repeated benchmarks previously retained tokens only for the final
         * request.  That made an adaptive runtime failure impossible to align
         * with the request/epoch that introduced it.  This is generic output
         * evidence captured after inference; it neither controls nor probes an
         * ExpertOverlay authority.
         */
        std::vector<int32_t> generated_token_ids;

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

        /** MTP draft predictions attempted by this exact request. */
        std::uint64_t mtp_draft_steps = 0;
        /** MTP draft tokens accepted by this exact request. */
        std::uint64_t mtp_accepted_tokens = 0;
        /** MTP draft tokens rejected by this exact request. */
        std::uint64_t mtp_rejected_tokens = 0;
        /** Grouped verifier transactions submitted by this exact request. */
        std::uint64_t mtp_verifier_runs = 0;
        /** Physical target rows evaluated across those verifier transactions. */
        std::uint64_t mtp_verifier_token_count = 0;

        /** Decode trajectory partitioned by the configured maintenance window. */
        std::vector<BenchmarkDecodeWindowResult> decode_windows;
    };

    /**
     * @brief Result-owned passive arithmetic observations at the pre-measurement boundary.
     *
     * Retained graph recording can emit host metadata only during setup/warmup.
     * Copying those exact records before the measured reset preserves the
     * observed route and prepared operands without retaining warmup work in the
     * measured counters. Counts describe host recording/submission observations;
     * they are neither native graph replay counts nor token counts. Missing
     * records do not prove that an arithmetic path was absent.
     */
    struct BenchmarkSetupArithmeticObservation
    {
        bool observed_before_measurement = false; ///< False when the run never reached the measurement boundary.
        bool collection_enabled = false; ///< Collector policy at that boundary; this snapshot never enables it.
        std::vector<PerfStatRecord> records; ///< Independent copies of exact tagged arithmetic families, not collector aliases.
    };

    /**
     * @brief Results from a benchmark run
     */
    struct BenchmarkResult
    {
        int measurement_iterations = 3; ///< Number of measured benchmark iterations averaged.
        int warmup_iterations = 1;      ///< Number of pre-measurement warmup iterations.
        BenchmarkSetupArithmeticObservation setup_warmup_arithmetic; ///< Independent setup/warmup evidence, unaffected by collector resets.

        // Exact benchmark input identity (the prompt text itself is deliberately omitted).
        BenchmarkPromptSource prompt_source = BenchmarkPromptSource::BuiltIn;
        std::string prompt_file_path; ///< Source path when `prompt_source == File`.
        size_t prompt_bytes = 0;      ///< Exact byte count before tokenization.
        std::string prompt_sha256;    ///< SHA-256 of the exact pre-tokenization bytes.
        std::vector<int> prompt_token_ids; ///< Exact once-tokenized natural input; never inferred from output tokens.

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
     * - Request-owned sampling; use temperature zero for greedy comparisons
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
         * Preserves the requested temperature, top-k, top-p, and seed across
         * serial and speculative execution. Request temperature zero for a
         * greedy comparison; MTP never implicitly changes the sampling law.
         *
         * @param config Parsed orchestration config (prompt, n_predict, etc.)
         * @return Benchmark results with timing metrics
         */
        BenchmarkResult run(const OrchestrationConfig &config);

        /**
         * @brief Resolve/tokenize an immutable input without executing inference.
         * @param config Exact prompt source; all bytes remain significant.
         * @return Sealed prompt and nonempty natural token sequence.
         * @throws std::runtime_error For unreadable/empty/failed tokenizer input.
         */
        PreparedBenchmarkInput prepareInput(const OrchestrationConfig &config) const;

        /**
         * @brief Run the same production timing path with a previously sealed input.
         * @param config Exact request sampling/MTP/decode policy.
         * @param input Once-tokenized immutable natural prompt.
         * @return Ordinary BenchmarkResult with the exact input token IDs.
         */
        BenchmarkResult run(const OrchestrationConfig &config, const PreparedBenchmarkInput &input);

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
        /** @brief Shared timing implementation; preparation never changes the inference path. */
        BenchmarkResult runImpl(const OrchestrationConfig &config,
            std::optional<std::reference_wrapper<const PreparedBenchmarkInput>> input);
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
