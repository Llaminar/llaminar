/**
 * @file BenchmarkRunner.cpp
 * @brief Implementation of benchmark runner for prefill/decode performance
 * @author David Sanftenberg
 * @date 2025
 */

#include "BenchmarkRunner.h"
#include "Logger.h"
#include "DebugEnv.h"
#include "KernelProfiler.h"
#include "KVCacheProfiler.h"
#include "CUDAKernelProfiler.h"
#include "ROCmKernelProfiler.h"
#include "WeightLoadingProfiler.h"
#include "../execution/local_execution/graph/IGraphExecutor.h"

#include "../backends/BackendManager.h"
#include "../backends/IBackend.h"
#include "fort.hpp"
#include <algorithm>
#include <iomanip>
#include <print>
#include <sstream>
#include <mpi.h>
#include <numeric>
#include <nlohmann/json.hpp>

namespace llaminar2
{

    // Number of benchmark iterations (after warmup)
    static constexpr int BENCHMARK_ITERATIONS = 3;
    static constexpr int WARMUP_ITERATIONS = 1;
    static constexpr int PREFILL_GRAPH_WARMUP_ITERATIONS = 2;

    // Log GPU memory on all GPUs (enabled via LLAMINAR_BENCH_MEM_LOG=1).
    static void logGPUMemorySnapshot(const char *label)
    {
        if (!debugEnv().runtime_debug.benchmark_memory_log)
            return;

        auto log_backend = [&](IBackend *b, const char *name)
        {
            if (!b)
                return;
            int n = b->deviceCount();
            for (int i = 0; i < n; ++i)
            {
                size_t free_b = b->deviceMemoryFree(i);
                size_t total_b = b->deviceMemoryTotal(i);
                if (total_b == 0)
                    continue;
                double free_mb = free_b / (1024.0 * 1024.0);
                double total_mb = total_b / (1024.0 * 1024.0);
                double used_mb = total_mb - free_mb;
                LOG_INFO("[MemProbe] " << label << " " << name << ":" << i
                                       << " used=" << std::fixed << std::setprecision(1) << used_mb
                                       << " MB free=" << free_mb << " MB / " << total_mb << " MB");
            }
        };
        log_backend(getCUDABackend(), "CUDA");
        log_backend(getROCmBackend(), "ROCm");
    }

    static bool hasPrefixOrMTPStats(const PrefixRuntimeStateSnapshot &snapshot)
    {
        return snapshot.prefix_cache_config_enabled ||
               snapshot.prefix_cache_ready ||
               snapshot.prefix_cache_bypassed ||
               snapshot.prefix_cache_lookups != 0 ||
               snapshot.prefix_cache_hits != 0 ||
               snapshot.prefix_cache_partial_hits != 0 ||
               snapshot.prefix_cache_misses != 0 ||
               snapshot.prefix_cache_matched_blocks != 0 ||
               snapshot.prefix_cache_matched_tokens != 0 ||
               snapshot.prefix_cache_stores != 0 ||
               snapshot.prefix_cache_inserts != 0 ||
               snapshot.prefix_cache_evictions != 0 ||
               snapshot.prefix_cache_promotions != 0 ||
               snapshot.prefix_cache_disk_hydrations != 0 ||
               snapshot.prefix_cache_terminal_state_hits != 0 ||
               snapshot.prefix_cache_ram_bytes != 0 ||
               snapshot.prefix_cache_device_bytes != 0 ||
               snapshot.prefix_cache_disk_bytes != 0 ||
               snapshot.prefix_cache_hybrid_state_bytes != 0 ||
               snapshot.prefix_cache_mtp_state_bytes != 0 ||
               snapshot.prefix_cache_bypasses != 0 ||
               snapshot.prefix_cache_unsupported_backend_bypasses != 0 ||
               snapshot.prefix_cache_fingerprint_bypasses != 0 ||
               snapshot.prefix_cache_terminal_state_bypasses != 0 ||
               snapshot.prefix_request.enabled ||
               snapshot.prefix_request.requested_tokens != 0 ||
               snapshot.prefix_request.matched_tokens != 0 ||
               snapshot.prefix_request.bypassed ||
               snapshot.mtp_config_enabled ||
               snapshot.mtp_bypassed ||
               snapshot.mtp_draft_steps != 0 ||
               snapshot.mtp_accepted_tokens != 0 ||
               snapshot.mtp_rejected_tokens != 0 ||
               snapshot.mtp_rollbacks != 0 ||
               snapshot.mtp_bypasses != 0 ||
               snapshot.mtp_verifier_runs != 0 ||
               snapshot.mtp_verifier_token_count != 0 ||
               snapshot.mtp_request.enabled ||
               snapshot.mtp_request.bypassed ||
               snapshot.mtp_request.draft_steps != 0 ||
               snapshot.mtp_request.accepted_tokens != 0 ||
               snapshot.mtp_request.rejected_tokens != 0 ||
               snapshot.mtp_request.rollbacks != 0 ||
               snapshot.prefill_chunk_schedules != 0 ||
               snapshot.prefill_chunk_successful_schedules != 0 ||
               snapshot.prefill_chunks != 0 ||
               snapshot.prefill_chunk_real_tokens != 0 ||
               snapshot.prefill_chunk_padded_tokens != 0 ||
               snapshot.prefill_chunk_failures != 0;
    }

    static nlohmann::json prefixRequestToJson(const PrefixCacheRequestSummary &request)
    {
        return nlohmann::json{
            {"enabled", request.enabled},
            {"bypassed", request.bypassed},
            {"bypass_reason", request.bypass_reason},
            {"hit", request.hit},
            {"partial_hit", request.partial_hit},
            {"requested_tokens", request.requested_tokens},
            {"matched_tokens", request.matched_tokens},
            {"matched_blocks", request.matched_blocks},
            {"terminal_logits_restored", request.terminal_logits_restored},
            {"terminal_hidden_restored", request.terminal_hidden_restored},
            {"mtp_state_restored", request.mtp_state_restored},
            {"hybrid_state_restored", request.hybrid_state_restored},
            {"storage_tier", request.storage_tier},
        };
    }

    static nlohmann::json mtpRequestToJson(const MTPRequestSummary &request)
    {
        return nlohmann::json{
            {"enabled", request.enabled},
            {"bypassed", request.bypassed},
            {"bypass_reason", request.bypass_reason},
            {"draft_steps", request.draft_steps},
            {"accepted_tokens", request.accepted_tokens},
            {"rejected_tokens", request.rejected_tokens},
            {"rollbacks", request.rollbacks},
            {"acceptance_rate", request.acceptance_rate},
        };
    }

    static double mtpTokenAcceptanceRate(uint64_t accepted_tokens, uint64_t rejected_tokens)
    {
        const uint64_t attempted_tokens = accepted_tokens + rejected_tokens;
        return attempted_tokens > 0
                   ? static_cast<double>(accepted_tokens) / static_cast<double>(attempted_tokens)
                   : 0.0;
    }

    std::string benchmarkResultToJsonString(
        const BenchmarkResult &result,
        const OrchestrationConfig *config)
    {
        const auto &state = result.prefix_state;

        nlohmann::json doc{
            {"schema", "llaminar.benchmark.v1"},
            {"success", result.success},
            {"prefill_success", result.prefill_success},
            {"decode_success", result.decode_success},
            {"tokens", {{"prefill", result.prefill_tokens},
                         {"decode", result.decode_tokens},
                         {"total", result.prefill_tokens + result.decode_tokens}}},
            {"timing_ms", {{"prefill", result.prefill_time_ms},
                            {"decode", result.decode_time_ms},
                            {"total", result.total_time_ms}}},
            {"throughput_tokens_per_sec", {{"prefill", result.prefill_tokens_per_sec},
                                            {"decode", result.decode_tokens_per_sec},
                                            {"overall", result.total_time_ms > 0.0
                                                            ? ((result.prefill_tokens + result.decode_tokens) * 1000.0) /
                                                                  result.total_time_ms
                                                            : 0.0}}},
            {"generated_text_bytes", result.generated_text.size()},
            {"runtime_state", {{"initialized", state.initialized},
                                {"architecture", state.architecture},
                                {"execution_path", state.execution_path},
                                {"primary_device", state.primary_device.toString()},
                                {"current_position", state.current_position},
                                {"session_epoch", state.session_epoch},
                                {"prefill_logits_ready", state.prefill_logits_ready},
                                {"has_hidden", state.has_hidden},
                                {"has_logits", state.has_logits},
                                {"kv_cache_count", state.kv_caches.size()},
                                {"mtp_kv_cache_count", state.mtp_kv_caches.size()},
                                {"gdn_layer_count", state.gdn_layers.size()},
                                {"cached_tokens", state.totalCachedTokens()},
                                {"mtp_cached_tokens", state.totalMTPCachedTokens()}}},
            {"prefix_cache", {{"config_enabled", state.prefix_cache_config_enabled},
                               {"ready", state.prefix_cache_ready},
                               {"bypassed", state.prefix_cache_bypassed},
                               {"bypass_reason", state.prefix_cache_bypass_reason},
                               {"lookups", state.prefix_cache_lookups},
                               {"hits", state.prefix_cache_hits},
                               {"partial_hits", state.prefix_cache_partial_hits},
                               {"misses", state.prefix_cache_misses},
                               {"matched_blocks", state.prefix_cache_matched_blocks},
                               {"matched_tokens", state.prefix_cache_matched_tokens},
                               {"stores", state.prefix_cache_stores},
                               {"inserts", state.prefix_cache_inserts},
                               {"evictions", state.prefix_cache_evictions},
                               {"promotions", state.prefix_cache_promotions},
                               {"disk_hydrations", state.prefix_cache_disk_hydrations},
                               {"terminal_state_hits", state.prefix_cache_terminal_state_hits},
                               {"bypasses", state.prefix_cache_bypasses},
                               {"unsupported_backend_bypasses", state.prefix_cache_unsupported_backend_bypasses},
                               {"fingerprint_bypasses", state.prefix_cache_fingerprint_bypasses},
                               {"terminal_state_bypasses", state.prefix_cache_terminal_state_bypasses},
                               {"ram_bytes", state.prefix_cache_ram_bytes},
                               {"device_bytes", state.prefix_cache_device_bytes},
                               {"disk_bytes", state.prefix_cache_disk_bytes},
                               {"hybrid_state_bytes", state.prefix_cache_hybrid_state_bytes},
                               {"mtp_state_bytes", state.prefix_cache_mtp_state_bytes},
                               {"request", prefixRequestToJson(state.prefix_request)}}},
            {"mtp", {{"config_enabled", state.mtp_config_enabled},
                      {"bypassed", state.mtp_bypassed},
                      {"bypass_reason", state.mtp_bypass_reason},
                      {"draft_steps", state.mtp_draft_steps},
                      {"accepted_tokens", state.mtp_accepted_tokens},
                      {"rejected_tokens", state.mtp_rejected_tokens},
                      {"rollbacks", state.mtp_rollbacks},
                      {"bypasses", state.mtp_bypasses},
                      {"verifier_runs", state.mtp_verifier_runs},
                      {"verifier_token_count", state.mtp_verifier_token_count},
                      {"acceptance_rate", mtpTokenAcceptanceRate(
                                              state.mtp_accepted_tokens,
                                              state.mtp_rejected_tokens)},
                      {"request", mtpRequestToJson(state.mtp_request)}}},
            {"prefill_chunks", {{"schedules", state.prefill_chunk_schedules},
                                 {"successful_schedules", state.prefill_chunk_successful_schedules},
                                 {"chunks", state.prefill_chunks},
                                 {"real_tokens", state.prefill_chunk_real_tokens},
                                 {"padded_tokens", state.prefill_chunk_padded_tokens},
                                 {"failures", state.prefill_chunk_failures}}},
        };

        if (config)
        {
            nlohmann::json config_json{
                {"benchmark_mode", config->benchmark_mode},
                {"n_predict", config->n_predict},
                {"prefix_cache_enabled", config->prefix_cache.enabled},
                {"mtp_enabled", config->mtp.enabled},
            };
            if (!config->model_path.empty())
                config_json["model_path"] = config->model_path;
            if (!config->benchmark_json_output_path.empty())
                config_json["benchmark_json_output_path"] = config->benchmark_json_output_path;
            if (config->device_for_this_rank)
                config_json["device"] = config->device_for_this_rank->toString();
            doc["config"] = std::move(config_json);
        }

        return doc.dump(2);
    }

    static std::string formatByteCount(uint64_t bytes)
    {
        constexpr double kib = 1024.0;
        constexpr double mib = 1024.0 * kib;
        constexpr double gib = 1024.0 * mib;

        std::ostringstream oss;
        if (bytes >= static_cast<uint64_t>(gib))
        {
            oss << std::fixed << std::setprecision(2) << (bytes / gib) << " GiB";
        }
        else if (bytes >= static_cast<uint64_t>(mib))
        {
            oss << std::fixed << std::setprecision(2) << (bytes / mib) << " MiB";
        }
        else if (bytes >= static_cast<uint64_t>(kib))
        {
            oss << std::fixed << std::setprecision(2) << (bytes / kib) << " KiB";
        }
        else
        {
            oss << bytes << " B";
        }
        return oss.str();
    }

    BenchmarkRunner::BenchmarkRunner(
        std::shared_ptr<IInferenceRunner> runner,
        std::shared_ptr<ITokenizer> tokenizer,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : runner_(std::move(runner)), tokenizer_(std::move(tokenizer)), mpi_ctx_(std::move(mpi_ctx))
    {
    }

    std::string BenchmarkRunner::generateDefaultPrompt() const
    {
        // A standardized prompt that tokenizes to ~512 tokens
        // This is a comprehensive text covering various topics to exercise the model
        return "The following is a comprehensive analysis of machine learning systems "
               "and their applications in modern computing environments. "
               "We will explore the fundamental concepts, examine practical implementations, "
               "and discuss the future directions of this rapidly evolving field. "
               "Machine learning has transformed how we approach problem-solving across "
               "numerous domains, from natural language processing to computer vision, "
               "from autonomous vehicles to medical diagnosis. "
               "The key to understanding these systems lies in grasping the underlying "
               "mathematical foundations while also appreciating the engineering challenges "
               "involved in deploying them at scale. "
               "Let us begin our exploration with an overview of the main paradigms: "
               "supervised learning, unsupervised learning, and reinforcement learning. "
               "Each of these approaches has its own strengths and is suited to different "
               "types of problems. In supervised learning, we train models using labeled data, "
               "where the correct output is known for each input example. "
               "This approach is particularly effective for classification and regression tasks. "
               "Unsupervised learning, on the other hand, deals with finding patterns in data "
               "without explicit labels. Clustering, dimensionality reduction, and anomaly detection "
               "are common applications. Reinforcement learning takes a different approach, "
               "where agents learn optimal behaviors through interaction with an environment, "
               "receiving rewards or penalties based on their actions. "
               "Deep learning, a subset of machine learning, has revolutionized the field "
               "by enabling the training of neural networks with many layers. "
               "These deep neural networks can learn hierarchical representations of data, "
               "automatically extracting features at multiple levels of abstraction. "
               "Convolutional neural networks have become the standard for image processing, "
               "while recurrent neural networks and transformers excel at sequential data. "
               "The transformer architecture, introduced in 2017, has become particularly influential, "
               "forming the basis for large language models like GPT, BERT, and LLaMA. "
               "These models are trained on vast amounts of text data and can perform "
               "a wide range of natural language tasks with impressive accuracy. "
               "The training process involves optimizing millions or billions of parameters "
               "using gradient descent and backpropagation algorithms. "
               "Modern training infrastructure relies on specialized hardware like GPUs and TPUs, "
               "distributed computing frameworks, and sophisticated optimization techniques. "
               "Transfer learning has emerged as a powerful paradigm, allowing models "
               "pre-trained on large datasets to be fine-tuned for specific tasks "
               "with relatively little additional data. This approach has democratized "
               "access to state-of-the-art AI capabilities for researchers and practitioners "
               "who may not have the resources to train large models from scratch. "
               "As we look to the future, several exciting developments are on the horizon. "
               "Multimodal models that can process text, images, audio, and video together "
               "are becoming increasingly sophisticated. Federated learning enables "
               "training on distributed data while preserving privacy. "
               "Neural architecture search automates the design of optimal network structures. "
               "And new hardware accelerators promise to make AI more efficient and accessible. "
               "The ethical implications of these technologies cannot be overlooked. "
               "Issues of bias, fairness, transparency, and accountability must be addressed "
               "as AI systems become more prevalent in society. Responsible AI development "
               "requires collaboration between technologists, policymakers, and the public "
               "to ensure these powerful tools benefit humanity as a whole.";
    }

    std::pair<bool, double> BenchmarkRunner::runPrefill(const std::vector<int> &tokens)
    {
        // Synchronize all ranks before timing (skip for single-rank)
        if (mpi_ctx_->world_size() > 1)
            mpi_ctx_->barrier();

        auto start = std::chrono::high_resolution_clock::now();

        bool success = runner_->forward(tokens.data(), tokens.size());

        // Synchronize after forward and propagate failures to every rank.
        success = synchronizeSuccess(success, "prefill");

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return {success, time_ms};
    }

    std::tuple<bool, double, int, std::string> BenchmarkRunner::runDecode(int n_tokens, int eos_token_id, bool ignore_stop_tokens)
    {
        Sampler sampler(42); // Fixed seed for reproducibility
        std::string generated_text;
        generated_text.reserve(n_tokens * 4); // Pre-allocate ~4 bytes/token to avoid reallocs
        int tokens_generated = 0;

        // Sampler profiling (enabled when LLAMINAR_PROFILING=1)
        const bool profile_sampler = debugEnv().profile.enabled;
        double sampler_total_us = 0.0;
        double inter_step_total_us = 0.0;

        // Synchronize before timing decode phase (skip for single-rank)
        if (mpi_ctx_->world_size() > 1)
            mpi_ctx_->barrier();

        auto start = std::chrono::high_resolution_clock::now();
        // Track the end of the last forward() call for inter-step measurement
        auto last_forward_end = start;

        if (runner_->supportsDecodeStep())
        {
            SamplingParams greedy_params;
            greedy_params.temperature = 0.0f;
            greedy_params.seed = 42;
            runner_->setDecodeSamplingParams(greedy_params);

            while (tokens_generated < n_tokens)
            {
                const int remaining = n_tokens - tokens_generated;
                runner_->setDecodeStepTokenBudget(remaining);
                DecodeStepOutput step = runner_->decodeStepForBenchmark();
                runner_->setDecodeStepTokenBudget(0);

                if (!synchronizeSuccess(step.error.empty(), "decode step"))
                {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    return {false, time_ms, tokens_generated, generated_text};
                }

                int step_token_count = mpi_ctx_->rank() == 0
                                           ? static_cast<int>(std::min<size_t>(
                                                 step.tokens.size(),
                                                 static_cast<size_t>(remaining)))
                                           : 0;
                if (mpi_ctx_->world_size() > 1)
                    mpi_ctx_->broadcast_int32(&step_token_count, 1, 0);

                if (step_token_count <= 0)
                {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    if (step.is_complete)
                        return {true, time_ms, tokens_generated, generated_text};
                    LOG_ERROR("Benchmark decode step produced no tokens");
                    return {false, time_ms, tokens_generated, generated_text};
                }

                int stop_reached = 0;
                if (mpi_ctx_->rank() == 0)
                {
                    for (int j = 0; j < step_token_count; ++j)
                    {
                        const int32_t token = step.tokens[static_cast<size_t>(j)];
                        if (!tokenizer_->is_stop_token(token))
                        {
                            generated_text += tokenizer_->decode_token(token);
                        }
                        else if (!ignore_stop_tokens)
                        {
                            stop_reached = 1;
                            break;
                        }
                    }
                }
                if (mpi_ctx_->world_size() > 1)
                    mpi_ctx_->broadcast_int32(&stop_reached, 1, 0);

                tokens_generated += step_token_count;

                if (stop_reached != 0)
                    break;

                const bool maintenance_success = runner_->maybeApplyDecodeBoundaryMaintenance();
                if (!synchronizeSuccess(maintenance_success, "decode maintenance"))
                {
                    auto end = std::chrono::high_resolution_clock::now();
                    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    return {false, time_ms, tokens_generated, generated_text};
                }
            }

            const bool decode_success = synchronizeSuccess(true, "decode complete");
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            return {decode_success, time_ms, tokens_generated, generated_text};
        }

        for (int i = 0; i < n_tokens; ++i)
        {
            int next_token = -1;

            // Rank 0: Sample next token (greedy for deterministic benchmark)
            if (mpi_ctx_->rank() == 0)
            {
                auto t0 = profile_sampler ? std::chrono::high_resolution_clock::now() : std::chrono::high_resolution_clock::time_point{};

                // Try GPU-side argmax first (avoids ~600 KB D2H + CPU scan)
                next_token = runner_->sampleGreedyOnDevice();

                if (next_token < 0)
                {
                    // GPU argmax not available (CPU device or unsupported backend).
                    // Fall back to host-side argmax over gathered logits.
                    const float *logit_data = runner_->logits();
                    if (!logit_data)
                    {
                        LOG_ERROR("CPU sampling fallback failed at decode step " << i
                                                                                 << ": logits() returned nullptr.");
                        auto end = std::chrono::high_resolution_clock::now();
                        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                        return {false, time_ms, tokens_generated, generated_text};
                    }
                    const int vs = runner_->vocab_size();
                    next_token = static_cast<int>(
                        std::distance(logit_data, std::max_element(logit_data, logit_data + vs)));
                }

                if (profile_sampler)
                {
                    auto t1 = std::chrono::high_resolution_clock::now();
                    sampler_total_us += std::chrono::duration<double, std::micro>(t1 - t0).count();
                }

                // Collect generated text for verification (but don't print during benchmark)
                if (!tokenizer_->is_stop_token(next_token))
                {
                    generated_text += tokenizer_->decode_token(next_token);
                }
            }

            // Broadcast token to all ranks (skip for single-rank)
            if (mpi_ctx_->world_size() > 1)
                mpi_ctx_->broadcast_int32(&next_token, 1, 0);

            // Check for stop token (unless benchmarking throughput)
            if (!ignore_stop_tokens && tokenizer_->is_stop_token(next_token))
            {
                break;
            }

            tokens_generated++;

            // Measure inter-step gap: time from last forward() return to this forward() call
            if (profile_sampler && i > 0)
            {
                auto pre_forward = std::chrono::high_resolution_clock::now();
                inter_step_total_us += std::chrono::duration<double, std::micro>(pre_forward - last_forward_end).count();
            }

            // Forward the token through pipeline
            const bool forward_success = runner_->forward(&next_token, 1);

            if (profile_sampler)
                last_forward_end = std::chrono::high_resolution_clock::now();

            if (!synchronizeSuccess(forward_success, "decode forward"))
            {
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                return {false, time_ms, tokens_generated, generated_text};
            }

            // Optional per-step callback (e.g., incremental MoE expert rebalancing)
            if (decode_step_cb_)
                decode_step_cb_();
        }

        // Synchronize after decode phase (skip for single-rank)
        const bool decode_success = synchronizeSuccess(true, "decode complete");

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        if (!decode_success)
            return {false, time_ms, tokens_generated, generated_text};

        // Accumulate inter-step profiling data across benchmark iterations
        if (profile_sampler && mpi_ctx_->rank() == 0 && tokens_generated > 0)
        {
            decode_loop_profile_.sampler_total_us += sampler_total_us;
            decode_loop_profile_.inter_step_total_us += inter_step_total_us;
            decode_loop_profile_.decode_tokens += tokens_generated;

            double avg_us = sampler_total_us / tokens_generated;
            double pct = (sampler_total_us / 1000.0) / time_ms * 100.0;
            LOG_INFO("Sampler profiling: " << std::fixed << std::setprecision(1)
                                           << avg_us << " µs/token avg, "
                                           << sampler_total_us / 1000.0 << " ms total ("
                                           << pct << "% of decode time)");
        }

        return {true, time_ms, tokens_generated, generated_text};
    }

    bool BenchmarkRunner::synchronizeSuccess(bool local_success, const char *phase) const
    {
        if (!mpi_ctx_ || mpi_ctx_->world_size() <= 1)
            return local_success;

        const float local = local_success ? 1.0f : 0.0f;
        float global_sum = 0.0f;

        try
        {
            mpi_ctx_->allreduce_sum(&local, &global_sum, 1);
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Benchmark " << phase << " failure synchronization failed: " << e.what());
            return false;
        }
        catch (...)
        {
            LOG_ERROR("Benchmark " << phase << " failure synchronization failed: unknown exception");
            return false;
        }

        const bool global_success = global_sum >= static_cast<float>(mpi_ctx_->world_size()) - 0.5f;
        if (!global_success && mpi_ctx_->rank() == 0)
        {
            LOG_ERROR("Benchmark " << phase << " failed on at least one rank (success_sum="
                                   << global_sum << "/" << mpi_ctx_->world_size() << ")");
        }
        return global_success;
    }

    BenchmarkResult BenchmarkRunner::run(const OrchestrationConfig &config)
    {
        BenchmarkResult result;
        auto capture_and_return = [&]() -> BenchmarkResult
        {
            result.prefix_state = runner_ ? runner_->prefixStateProbe() : PrefixRuntimeStateSnapshot{};
            return result;
        };

        // Determine prompt (use default if not provided or empty)
        std::string prompt = config.prompt;
        if (prompt.empty() || prompt == "Hello, my name is")
        {
            prompt = generateDefaultPrompt();
            if (mpi_ctx_->rank() == 0)
            {
                LOG_DEBUG("Using default benchmark prompt (~512 tokens)");
            }
        }

        // Tokenize prompt (rank 0 only, then broadcast)
        std::vector<int> tokens;
        int token_count = 0;

        if (mpi_ctx_->rank() == 0)
        {
            tokens = tokenizer_->encode(prompt, /*add_bos=*/false, /*add_eos=*/false);
            token_count = static_cast<int>(tokens.size());

            if (tokens.empty())
            {
                LOG_ERROR("Failed to tokenize benchmark prompt");
                token_count = -1;
            }
        }

        // Broadcast token count (skip for single-rank)
        if (mpi_ctx_->world_size() > 1)
            mpi_ctx_->broadcast_int32(&token_count, 1, 0);

        if (token_count <= 0)
        {
            return capture_and_return(); // Return empty result on error
        }

        // Broadcast tokens to all ranks
        if (mpi_ctx_->rank() != 0)
        {
            tokens.resize(token_count);
        }
        if (mpi_ctx_->world_size() > 1)
            mpi_ctx_->broadcast_int32(tokens.data(), token_count, 0);

        result.prefill_tokens = token_count;

        // Determine number of decode tokens
        // -1 means "use default" (128 for benchmark)
        // 0 means "skip decode phase"
        // >0 means use that value
        int n_decode = config.n_predict;
        if (n_decode < 0)
        {
            n_decode = 128; // Default for benchmark
        }

        if (mpi_ctx_->rank() == 0)
        {
            LOG_DEBUG("Benchmark configuration:");
            LOG_DEBUG("  Prefill tokens: " << token_count);
            LOG_DEBUG("  Decode tokens:  " << n_decode);
            LOG_DEBUG("  Warmup runs:    " << WARMUP_ITERATIONS);
            LOG_DEBUG("  Benchmark runs: " << BENCHMARK_ITERATIONS);
            LOG_DEBUG("");
        }

        // Enable GPU-side greedy sampling to skip D2H logits gather during decode.
        // Only enabled on GPU — CPU has no device-side argmax, so logits must be
        // gathered to host for CPU-side sampling.
        const bool has_gpu = runner_->primaryDeviceId().is_gpu();
        runner_->setSkipLogitsGatherDecode(has_gpu);

        // ========================================================================
        // Warmup Phase - Run once to warm up caches, JIT, etc.
        // ========================================================================
        if (mpi_ctx_->rank() == 0)
        {
            LOG_INFO("Running warmup...");
        }

        // Reset pipeline state before warmup
        runner_->clear_cache();
        logGPUMemorySnapshot("before-warmup");

        // Suppress GPU stage timeline during warmup — warmup includes one-time costs
        // (weight H2D transfers, buffer allocation, kernel JIT) that inflate overhead
        // numbers and don't reflect steady-state performance.
        runner_->setSuppressTimeline(true);

        // Skip D2H logits gather for prefill — prefill logits are never consumed
        // in the benchmark flow (sampling happens during decode via GPU-side argmax).
        // This eliminates ~405ms of PCIe traffic for TP=2 prefill.
        runner_->setSkipLogitsGatherPrefill(true);

        auto warmPrefillGraphCapture = [&]() -> bool
        {
            if (!debugEnv().execution.gpu_graphs)
                return true;

            if (mpi_ctx_->rank() == 0)
            {
                LOG_INFO("Preparing prefill graph capture for steady-state benchmark...");
            }

            for (int iter = 0; iter < PREFILL_GRAPH_WARMUP_ITERATIONS; ++iter)
            {
                runner_->clear_cache();
                auto [graph_warmup_success, graph_warmup_time] = runPrefill(tokens);
                if (!graph_warmup_success)
                {
                    if (mpi_ctx_->rank() == 0)
                    {
                        LOG_ERROR("Prefill graph warmup failed on iteration " << (iter + 1));
                    }
                    return false;
                }
            }

            return true;
        };

        // Warmup prefill
        auto [warmup_prefill_success, warmup_prefill_time] = runPrefill(tokens);
        if (!warmup_prefill_success)
        {
            if (mpi_ctx_->rank() == 0)
            {
                LOG_ERROR("Warmup prefill failed");
            }
            return capture_and_return();
        }

        // Warmup decode (if requested)
        if (n_decode > 0)
        {
            int eos_token = tokenizer_->eos_token();
            auto [warmup_decode_success, warmup_decode_time, warmup_tokens, warmup_text] =
                runDecode(n_decode, eos_token, /*ignore_stop_tokens=*/true);
            if (!warmup_decode_success)
            {
                if (mpi_ctx_->rank() == 0)
                {
                    LOG_ERROR("Warmup decode failed");
                }
                return capture_and_return();
            }
        }

        if (!warmPrefillGraphCapture())
        {
            return capture_and_return();
        }

        if (mpi_ctx_->rank() == 0)
        {
            LOG_INFO("Warmup complete.");
        }
        logGPUMemorySnapshot("after-warmup");

        // Post-warmup callback (e.g., MoE expert rebalancing)
        if (post_warmup_cb_)
        {
            post_warmup_cb_();

            // Re-warm caches after post-warmup work (e.g., MPI expert weight
            // transfers can evict hot data from LLC, causing the first benchmark
            // iteration to measure cold-cache performance).
            if (mpi_ctx_->rank() == 0)
                LOG_DEBUG("Re-warming caches after post-warmup setup...");
            runner_->clear_cache();
            auto [rw_ok, rw_time] = runPrefill(tokens);
            if (rw_ok && n_decode > 0)
            {
                int eos_token = tokenizer_->eos_token();
                runDecode(n_decode, eos_token, /*ignore_stop_tokens=*/true);
            }
            if (rw_ok && !warmPrefillGraphCapture())
            {
                return capture_and_return();
            }
        }

        if (mpi_ctx_->rank() == 0)
        {
            LOG_INFO("Running " << BENCHMARK_ITERATIONS << " benchmark iterations...");
        }

        // Reset profiling after warmup (only track actual benchmark iterations)
        if (KernelProfiler::isEnabled())
        {
            KernelProfiler::reset();
            KVCacheProfiler::reset();
            CUDAKernelProfiler::reset();
            ROCmKernelProfiler::reset();
        }
        // Also reset executor overhead stats so warmup overhead isn't counted
        runner_->resetExecutorStats();

        // Re-enable GPU stage timeline for actual benchmark runs.
        // Timeline is suppressed only during warmup (which includes one-time costs).
        runner_->setSuppressTimeline(false);

        // Accumulate prefill timelines instead of printing per-iteration.
        // flushStageTimeline() after the loop will print averaged tables.
        runner_->setAccumulatePrefill(true);

        // ========================================================================
        // Benchmark Iterations - Run multiple times and average
        // ========================================================================
        std::vector<double> prefill_times;
        std::vector<double> decode_times;
        std::vector<int> decode_token_counts;
        std::string last_generated_text;

        logGPUMemorySnapshot("pre-iter-loop");

        for (int iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            // Reset pipeline state before each iteration
            runner_->clear_cache();
            logGPUMemorySnapshot(("after-clear-cache iter=" + std::to_string(iter + 1)).c_str());

            if (mpi_ctx_->rank() == 0)
            {
                LOG_DEBUG("  Iteration " << (iter + 1) << "/" << BENCHMARK_ITERATIONS << "...");
            }

            // Run prefill
            KernelProfiler::setCurrentPhase(KernelProfiler::Phase::PREFILL);
            CUDAKernelProfiler::setCurrentPhase(CUDAKernelProfiler::Phase::PREFILL);
            ROCmKernelProfiler::setCurrentPhase(ROCmKernelProfiler::Phase::PREFILL);
            KVCacheProfiler::setCurrentPhase(KVCacheProfiler::Phase::PREFILL);
            GraphExecutorStats::setCurrentPhase(ExecutionPhase::PREFILL);
            auto [prefill_success, prefill_time] = runPrefill(tokens);
            if (!prefill_success)
            {
                if (mpi_ctx_->rank() == 0)
                {
                    LOG_ERROR("Prefill failed on iteration " << (iter + 1));
                }
                logGPUMemorySnapshot(("prefill-fail iter=" + std::to_string(iter + 1)).c_str());
                return capture_and_return();
            }
            prefill_times.push_back(prefill_time);
            logGPUMemorySnapshot(("after-prefill iter=" + std::to_string(iter + 1)).c_str());

            // Run decode (if requested)
            if (n_decode > 0)
            {
                KernelProfiler::setCurrentPhase(KernelProfiler::Phase::DECODE);
                CUDAKernelProfiler::setCurrentPhase(CUDAKernelProfiler::Phase::DECODE);
                ROCmKernelProfiler::setCurrentPhase(ROCmKernelProfiler::Phase::DECODE);
                KVCacheProfiler::setCurrentPhase(KVCacheProfiler::Phase::DECODE);
                GraphExecutorStats::setCurrentPhase(ExecutionPhase::DECODE);
                int eos_token = tokenizer_->eos_token();
                auto [decode_success, decode_time, tokens_generated, generated_text] =
                    runDecode(n_decode, eos_token, /*ignore_stop_tokens=*/true);
                if (!decode_success)
                {
                    if (mpi_ctx_->rank() == 0)
                    {
                        LOG_ERROR("Decode failed on iteration " << (iter + 1));
                    }
                    return capture_and_return();
                }
                decode_times.push_back(decode_time);
                decode_token_counts.push_back(tokens_generated);
                last_generated_text = generated_text;
                logGPUMemorySnapshot(("after-decode iter=" + std::to_string(iter + 1)).c_str());
            }

            if (mpi_ctx_->rank() == 0)
            {
                LOG_DEBUG("    Prefill: " << std::fixed << std::setprecision(2) << prefill_time << " ms"
                                          << (n_decode > 0 ? ", Decode: " + std::to_string(static_cast<int>(decode_times.back())) + " ms" : ""));
            }
        }

        // ========================================================================
        // Calculate Averages
        // ========================================================================

        // Prefill averages
        double avg_prefill_time = std::accumulate(prefill_times.begin(), prefill_times.end(), 0.0) / prefill_times.size();
        result.prefill_time_ms = avg_prefill_time;
        result.prefill_tokens_per_sec = (result.prefill_tokens * 1000.0) / avg_prefill_time;
        result.prefill_success = true;

        // Decode averages (if applicable)
        if (n_decode > 0 && !decode_times.empty())
        {
            double avg_decode_time = std::accumulate(decode_times.begin(), decode_times.end(), 0.0) / decode_times.size();
            int avg_decode_tokens = std::accumulate(decode_token_counts.begin(), decode_token_counts.end(), 0) / decode_token_counts.size();

            result.decode_time_ms = avg_decode_time;
            result.decode_tokens = avg_decode_tokens;
            result.decode_tokens_per_sec = (avg_decode_tokens * 1000.0) / avg_decode_time;
            result.decode_success = true;
            result.generated_text = last_generated_text;
        }
        else
        {
            result.decode_success = true;
            result.decode_tokens = 0;
            result.decode_time_ms = 0.0;
            result.decode_tokens_per_sec = 0.0;
        }

        // Calculate totals
        result.total_time_ms = result.prefill_time_ms + result.decode_time_ms;
        result.success = result.prefill_success && result.decode_success;

        if (mpi_ctx_->rank() == 0)
        {
            LOG_INFO("Benchmark complete.");
        }

        return capture_and_return();
    }

    void BenchmarkRunner::printResults(const BenchmarkResult &result)
    {
        if (mpi_ctx_->rank() != 0)
        {
            return; // Only rank 0 prints
        }

        std::print("\n");

        // Title table
        {
            fort::utf8_table title;
            title.set_border_style(FT_DOUBLE2_STYLE);
            std::ostringstream title_ss;
            title_ss << "BENCHMARK RESULTS (average of " << BENCHMARK_ITERATIONS << " runs after warmup)";
            title << title_ss.str() << fort::endr;
            title[0][0].set_cell_text_align(fort::text_align::center);
            title.row(0).set_cell_row_type(fort::row_type::header);
            std::print("{}", title.to_string());
        }

        // Results table
        fort::utf8_table table;
        table.set_border_style(FT_DOUBLE2_STYLE);

        table << fort::header << "Phase" << "Metric" << "Value" << fort::endr;
        table.column(0).set_cell_text_align(fort::text_align::left);
        table.column(1).set_cell_text_align(fort::text_align::left);
        table.column(2).set_cell_text_align(fort::text_align::right);

        // Prefill results
        if (result.prefill_tokens > 0)
        {
            std::ostringstream tokens_ss, time_ss, throughput_ss;
            tokens_ss << result.prefill_tokens << " tokens";
            time_ss << std::fixed << std::setprecision(2) << result.prefill_time_ms << " ms";
            throughput_ss << std::fixed << std::setprecision(2) << result.prefill_tokens_per_sec << " tok/s";

            table << "PREFILL" << "Tokens" << tokens_ss.str() << fort::endr;
            table << "" << "Time" << time_ss.str() << fort::endr;
            table << "" << "Throughput" << throughput_ss.str() << fort::endr;
        }
        else
        {
            table << "PREFILL" << "(SKIPPED)" << "" << fort::endr;
        }

        table << fort::separator;

        // Decode results
        if (result.decode_tokens > 0)
        {
            std::ostringstream tokens_ss, time_ss, throughput_ss;
            tokens_ss << result.decode_tokens << " tokens";
            time_ss << std::fixed << std::setprecision(2) << result.decode_time_ms << " ms";
            throughput_ss << std::fixed << std::setprecision(2) << result.decode_tokens_per_sec << " tok/s";

            table << "DECODE" << "Tokens" << tokens_ss.str() << fort::endr;
            table << "" << "Time" << time_ss.str() << fort::endr;
            table << "" << "Throughput" << throughput_ss.str() << fort::endr;
        }
        else
        {
            table << "DECODE" << "(SKIPPED)" << "" << fort::endr;
        }

        table << fort::separator;

        // Total
        {
            std::ostringstream time_ss;
            time_ss << std::fixed << std::setprecision(2) << result.total_time_ms << " ms";
            table << "TOTAL" << "Time" << time_ss.str() << fort::endr;

            if (result.prefill_tokens + result.decode_tokens > 0 && result.total_time_ms > 0)
            {
                double total_tokens = result.prefill_tokens + result.decode_tokens;
                double overall_throughput = (total_tokens * 1000.0) / result.total_time_ms;
                std::ostringstream throughput_ss;
                throughput_ss << std::fixed << std::setprecision(2) << overall_throughput << " tok/s (avg)";
                table << "" << "Overall" << throughput_ss.str() << fort::endr;
            }
        }

        std::print("{}", table.to_string());

        const auto &prefix_state = result.prefix_state;
        if (hasPrefixOrMTPStats(prefix_state))
        {
            fort::utf8_table state_table;
            state_table.set_border_style(FT_DOUBLE2_STYLE);
            state_table << fort::header << "PREFIX / MTP STATE" << "Value" << fort::endr;
            state_table.column(0).set_cell_text_align(fort::text_align::left);
            state_table.column(1).set_cell_text_align(fort::text_align::right);

            {
                std::ostringstream status;
                status << (prefix_state.prefix_cache_config_enabled ? "enabled" : "disabled")
                       << ", " << (prefix_state.prefix_cache_ready ? "ready" : "not ready");
                if (prefix_state.prefix_cache_bypassed)
                {
                    status << ", bypassed";
                    if (!prefix_state.prefix_cache_bypass_reason.empty())
                    {
                        status << " (" << prefix_state.prefix_cache_bypass_reason << ")";
                    }
                }
                state_table << "Prefix cache" << status.str() << fort::endr;
            }
            if (prefix_state.prefix_request.enabled ||
                prefix_state.prefix_request.requested_tokens != 0 ||
                prefix_state.prefix_request.matched_tokens != 0 ||
                prefix_state.prefix_request.bypassed)
            {
                const auto &request = prefix_state.prefix_request;
                std::ostringstream summary;
                if (request.bypassed)
                {
                    summary << "bypassed";
                    if (!request.bypass_reason.empty())
                        summary << " (" << request.bypass_reason << ")";
                }
                else if (request.hit)
                {
                    summary << "hit";
                }
                else if (request.partial_hit)
                {
                    summary << "partial-hit";
                }
                else
                {
                    summary << "miss";
                }
                summary << ", requested " << request.requested_tokens
                        << ", matched " << request.matched_tokens
                        << " tokens/" << request.matched_blocks
                        << " blocks, tier " << request.storage_tier;
                if (request.terminal_logits_restored || request.terminal_hidden_restored)
                {
                    summary << ", terminal logits "
                            << (request.terminal_logits_restored ? "restored" : "not restored")
                            << ", hidden "
                            << (request.terminal_hidden_restored ? "restored" : "not restored");
                }
                state_table << "Prefix request" << summary.str() << fort::endr;
            }
            if (prefix_state.prefix_cache_bypasses != 0 ||
                prefix_state.prefix_cache_unsupported_backend_bypasses != 0 ||
                prefix_state.prefix_cache_fingerprint_bypasses != 0 ||
                prefix_state.prefix_cache_terminal_state_bypasses != 0)
            {
                std::ostringstream bypasses;
                bypasses << prefix_state.prefix_cache_bypasses << " total, "
                         << prefix_state.prefix_cache_unsupported_backend_bypasses << " unsupported, "
                         << prefix_state.prefix_cache_fingerprint_bypasses << " fingerprint, "
                         << prefix_state.prefix_cache_terminal_state_bypasses << " terminal-state";
                state_table << "Bypasses" << bypasses.str() << fort::endr;
            }
            {
                std::ostringstream lookups;
                lookups << prefix_state.prefix_cache_lookups << " lookups, "
                        << prefix_state.prefix_cache_hits << " hits, "
                        << prefix_state.prefix_cache_partial_hits << " partial, "
                        << prefix_state.prefix_cache_misses << " misses";
                state_table << "Lookup results" << lookups.str() << fort::endr;
            }
            {
                std::ostringstream matched;
                matched << prefix_state.prefix_cache_matched_blocks << " blocks, "
                        << prefix_state.prefix_cache_matched_tokens << " tokens";
                state_table << "Matched prefix" << matched.str() << fort::endr;
            }
            {
                std::ostringstream storage;
                storage << prefix_state.prefix_cache_stores << " stores, "
                        << prefix_state.prefix_cache_evictions << " evictions, "
                        << prefix_state.prefix_cache_promotions << " promotions, "
                        << prefix_state.prefix_cache_disk_hydrations << " hydrations";
                state_table << "Storage events" << storage.str() << fort::endr;
            }
            {
                std::ostringstream bytes;
                bytes << "RAM " << formatByteCount(prefix_state.prefix_cache_ram_bytes)
                      << ", device " << formatByteCount(prefix_state.prefix_cache_device_bytes)
                      << ", disk " << formatByteCount(prefix_state.prefix_cache_disk_bytes);
                state_table << "Resident bytes" << bytes.str() << fort::endr;
            }
            {
                std::ostringstream payload;
                payload << "hybrid " << formatByteCount(prefix_state.prefix_cache_hybrid_state_bytes)
                        << ", MTP " << formatByteCount(prefix_state.prefix_cache_mtp_state_bytes)
                        << ", terminal hits " << prefix_state.prefix_cache_terminal_state_hits;
                state_table << "Payload state" << payload.str() << fort::endr;
            }
            if (prefix_state.prefill_chunk_schedules != 0 ||
                prefix_state.prefill_chunk_successful_schedules != 0 ||
                prefix_state.prefill_chunks != 0 ||
                prefix_state.prefill_chunk_real_tokens != 0 ||
                prefix_state.prefill_chunk_padded_tokens != 0 ||
                prefix_state.prefill_chunk_failures != 0)
            {
                std::ostringstream chunked;
                chunked << prefix_state.prefill_chunk_successful_schedules
                        << "/" << prefix_state.prefill_chunk_schedules
                        << " schedules, "
                        << prefix_state.prefill_chunks << " chunks, "
                        << prefix_state.prefill_chunk_real_tokens << " real tokens, "
                        << prefix_state.prefill_chunk_padded_tokens << " padded tokens, "
                        << prefix_state.prefill_chunk_failures << " failures";
                state_table << "Prefill chunks" << chunked.str() << fort::endr;
            }

            if (prefix_state.mtp_draft_steps != 0 ||
                prefix_state.mtp_accepted_tokens != 0 ||
                prefix_state.mtp_rejected_tokens != 0 ||
                prefix_state.mtp_rollbacks != 0 ||
                prefix_state.mtp_bypasses != 0 ||
                prefix_state.mtp_verifier_runs != 0 ||
                prefix_state.mtp_verifier_token_count != 0 ||
                prefix_state.mtp_config_enabled ||
                prefix_state.mtp_bypassed)
            {
                state_table << fort::separator;
                {
                    std::ostringstream status;
                    status << (prefix_state.mtp_config_enabled ? "enabled" : "disabled");
                    if (prefix_state.mtp_bypassed)
                    {
                        status << ", bypassed";
                        if (!prefix_state.mtp_bypass_reason.empty())
                        {
                            status << " (" << prefix_state.mtp_bypass_reason << ")";
                        }
                    }
                    state_table << "MTP" << status.str() << fort::endr;
                }
                if (prefix_state.mtp_request.enabled ||
                    prefix_state.mtp_request.bypassed ||
                    prefix_state.mtp_request.draft_steps != 0 ||
                    prefix_state.mtp_request.accepted_tokens != 0 ||
                    prefix_state.mtp_request.rejected_tokens != 0 ||
                    prefix_state.mtp_request.rollbacks != 0)
                {
                    const auto &request = prefix_state.mtp_request;
                    std::ostringstream mtp_request;
                    mtp_request << request.draft_steps << " draft steps, "
                                << request.accepted_tokens << " accepted, "
                                << request.rejected_tokens << " rejected, "
                                << request.rollbacks << " rollbacks, "
                                << std::fixed << std::setprecision(2)
                                << (request.acceptance_rate * 100.0) << "% acceptance";
                    if (request.bypassed)
                    {
                        mtp_request << ", bypassed";
                        if (!request.bypass_reason.empty())
                            mtp_request << " (" << request.bypass_reason << ")";
                    }
                    state_table << "MTP request" << mtp_request.str() << fort::endr;
                }
                {
                    std::ostringstream mtp;
                    mtp << prefix_state.mtp_draft_steps << " draft steps, "
                        << prefix_state.mtp_accepted_tokens << " accepted, "
                        << prefix_state.mtp_rejected_tokens << " rejected, "
                        << prefix_state.mtp_rollbacks << " rollbacks, "
                        << prefix_state.mtp_bypasses << " bypasses, "
                        << prefix_state.mtp_verifier_runs << " verifier runs, "
                        << prefix_state.mtp_verifier_token_count << " verifier tokens";
                    state_table << "MTP decode" << mtp.str() << fort::endr;
                }
            }

            std::print("{}", state_table.to_string());
        }

        // Flush accumulated GPU stage timeline (prefill + decode + TP orchestrator)
        // This prints after BENCHMARK RESULTS for cleaner output ordering.
        if (runner_)
        {
            runner_->flushStageTimeline();
        }

        // Print kernel profiling summary if enabled
        if (KernelProfiler::isEnabled())
        {
            // Kernel profilers accumulate stats across ALL benchmark iterations,
            // but result.prefill_time_ms/decode_time_ms are averages.
            // Scale wall clocks and token counts by iteration count so %
            // calculations use the total accumulated wall clock as denominator.
            uint64_t total_tokens = (result.prefill_tokens + result.decode_tokens) * BENCHMARK_ITERATIONS;
            double total_prefill_ms = result.prefill_time_ms * BENCHMARK_ITERATIONS;
            double total_decode_ms = result.decode_time_ms * BENCHMARK_ITERATIONS;
            uint64_t total_prefill_tokens = result.prefill_tokens * BENCHMARK_ITERATIONS;
            uint64_t total_decode_tokens = result.decode_tokens * BENCHMARK_ITERATIONS;

            KernelProfiler::printSummary(total_tokens, total_prefill_ms, total_decode_ms,
                                         total_prefill_tokens, total_decode_tokens);

            KVCacheProfiler::printSummary();

            // Skip per-kernel profiling when GPU stage timing is active —
            // the GPU Stage Timeline provides strictly superior coverage
            // (all stages, GPU-event precision) vs hand-instrumented subset.
            if (!debugEnv().gpu_stage_timing)
            {
                CUDAKernelProfiler::printSummary(total_tokens, total_prefill_ms, total_decode_ms,
                                                 total_prefill_tokens, total_decode_tokens);
                ROCmKernelProfiler::printSummary(total_tokens, total_prefill_ms, total_decode_ms,
                                                 total_prefill_tokens, total_decode_tokens);
            }
        }

        // Print executor overhead profiling if enabled (LLAMINAR_PROFILING=1)
        if (KernelProfiler::isEnabled() && runner_)
        {
            const auto *stats = runner_->executorStats();
            if (stats && stats->total_stages_executed > 0)
            {
                uint64_t ep_prefill = result.prefill_tokens * BENCHMARK_ITERATIONS;
                uint64_t ep_decode = result.decode_tokens * BENCHMARK_ITERATIONS;
                stats->printProfilingSummary(ep_prefill, ep_decode);
            }
        }

        // Print weight loading profiling if enabled
        if (KernelProfiler::isEnabled())
        {
            std::string wl_summary = WeightLoadingProfiler::getSummary();
            if (!wl_summary.empty())
            {
                std::print("{}", wl_summary);
            }
        }

        // Print decode loop inter-step overhead (sampling + broadcast + housekeeping)
        if (KernelProfiler::isEnabled() && !decode_loop_profile_.empty())
        {
            const auto &dlp = decode_loop_profile_;
            double avg_sampler_us = dlp.sampler_total_us / dlp.decode_tokens;
            double avg_inter_step_us = dlp.inter_step_total_us / dlp.decode_tokens;
            double avg_other_us = avg_inter_step_us - avg_sampler_us;
            if (avg_other_us < 0)
                avg_other_us = 0;
            double decode_wall_ms = result.decode_time_ms;
            double inter_step_pct = (dlp.inter_step_total_us / 1000.0 / BENCHMARK_ITERATIONS) / decode_wall_ms * 100.0;

            fort::utf8_table tbl;
            tbl.set_border_style(FT_DOUBLE2_STYLE);
            tbl << fort::header << "DECODE LOOP OVERHEAD" << "AVG/token" << "% of decode" << fort::endr;
            {
                std::ostringstream s;
                s << std::fixed << std::setprecision(1) << avg_sampler_us << " μs";
                std::ostringstream p;
                p << std::fixed << std::setprecision(1)
                  << (dlp.sampler_total_us / 1000.0 / BENCHMARK_ITERATIONS) / decode_wall_ms * 100.0 << "%";
                tbl << "  Sampling (argmax)" << s.str() << p.str() << fort::endr;
            }
            {
                std::ostringstream s;
                s << std::fixed << std::setprecision(1) << avg_other_us << " μs";
                std::ostringstream p;
                double other_total_us = dlp.inter_step_total_us - dlp.sampler_total_us;
                if (other_total_us < 0)
                    other_total_us = 0;
                p << std::fixed << std::setprecision(1)
                  << (other_total_us / 1000.0 / BENCHMARK_ITERATIONS) / decode_wall_ms * 100.0 << "%";
                tbl << "  Other (broadcast, prep)" << s.str() << p.str() << fort::endr;
            }
            tbl << fort::separator;
            {
                std::ostringstream s;
                s << std::fixed << std::setprecision(1) << avg_inter_step_us << " μs";
                std::ostringstream p;
                p << std::fixed << std::setprecision(1) << inter_step_pct << "%";
                tbl << "  TOTAL inter-step" << s.str() << p.str() << fort::endr;
            }
            tbl.column(0).set_cell_text_align(fort::text_align::left);
            tbl.column(1).set_cell_text_align(fort::text_align::right);
            tbl.column(2).set_cell_text_align(fort::text_align::right);
            std::print("{}", tbl.to_string());
        }

        // Status
        if (result.success)
        {
            std::print("\n✓ Benchmark completed successfully.\n");
        }
        else
        {
            std::print("\n✗ Benchmark failed.\n");
        }

        std::println("");
    }

} // namespace llaminar2
