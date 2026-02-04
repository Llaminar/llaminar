/**
 * @file BenchmarkRunner.cpp
 * @brief Implementation of benchmark runner for prefill/decode performance
 * @author David Sanftenberg
 * @date 2025
 */

#include "BenchmarkRunner.h"
#include "Logger.h"
#include "KernelProfiler.h"
#include "CUDAKernelProfiler.h"
#include "ROCmKernelProfiler.h"
#include "../execution/local_execution/graph/IGraphExecutor.h"
#include "fort.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <mpi.h>
#include <numeric>

namespace llaminar2
{

    // Number of benchmark iterations (after warmup)
    static constexpr int BENCHMARK_ITERATIONS = 3;
    static constexpr int WARMUP_ITERATIONS = 1;

    BenchmarkRunner::BenchmarkRunner(
        std::shared_ptr<IInferenceRunner> runner,
        std::shared_ptr<ITokenizer> tokenizer,
        std::shared_ptr<MPIContext> mpi_ctx)
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
        // Synchronize all ranks before timing
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();

        bool success = runner_->forward(tokens.data(), tokens.size());

        // Synchronize after forward for accurate timing
        MPI_Barrier(MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return {success, time_ms};
    }

    std::tuple<bool, double, int, std::string> BenchmarkRunner::runDecode(int n_tokens, int eos_token_id)
    {
        Sampler sampler(42); // Fixed seed for reproducibility
        std::string generated_text;
        int tokens_generated = 0;

        // Synchronize before timing decode phase
        MPI_Barrier(MPI_COMM_WORLD);

        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < n_tokens; ++i)
        {
            int next_token = -1;

            // Rank 0: Get logits and sample (greedy for deterministic benchmark)
            if (mpi_ctx_->rank() == 0)
            {
                const float *logits = runner_->logits();
                size_t vocab_size = tokenizer_->vocab_size();
                std::vector<float> logits_vec(logits, logits + vocab_size);

                // Greedy sampling (argmax) for deterministic benchmark
                next_token = sampler.sample_greedy(logits_vec);

                // Collect generated text for verification (but don't print during benchmark)
                if (!tokenizer_->is_stop_token(next_token))
                {
                    generated_text += tokenizer_->decode_token(next_token);
                }
            }

            // Broadcast token to all ranks
            MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Check for stop token
            if (tokenizer_->is_stop_token(next_token))
            {
                break;
            }

            tokens_generated++;

            // Forward the token through pipeline
            if (!runner_->forward(&next_token, 1))
            {
                // Synchronize on failure
                MPI_Barrier(MPI_COMM_WORLD);
                auto end = std::chrono::high_resolution_clock::now();
                double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
                return {false, time_ms, tokens_generated, generated_text};
            }
        }

        // Synchronize after decode phase
        MPI_Barrier(MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        return {true, time_ms, tokens_generated, generated_text};
    }

    BenchmarkResult BenchmarkRunner::run(const OrchestrationConfig &config)
    {
        BenchmarkResult result;

        // Determine prompt (use default if not provided or empty)
        std::string prompt = config.prompt;
        if (prompt.empty() || prompt == "Hello, my name is")
        {
            prompt = generateDefaultPrompt();
            if (mpi_ctx_->rank() == 0)
            {
                LOG_INFO("Using default benchmark prompt (~512 tokens)");
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

        // Broadcast token count
        MPI_Bcast(&token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (token_count <= 0)
        {
            return result; // Return empty result on error
        }

        // Broadcast tokens to all ranks
        if (mpi_ctx_->rank() != 0)
        {
            tokens.resize(token_count);
        }
        MPI_Bcast(tokens.data(), token_count, MPI_INT, 0, MPI_COMM_WORLD);

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
            LOG_INFO("Benchmark configuration:");
            LOG_INFO("  Prefill tokens: " << token_count);
            LOG_INFO("  Decode tokens:  " << n_decode);
            LOG_INFO("  Warmup runs:    " << WARMUP_ITERATIONS);
            LOG_INFO("  Benchmark runs: " << BENCHMARK_ITERATIONS);
            LOG_INFO("");
        }

        // ========================================================================
        // Warmup Phase - Run once to warm up caches, JIT, etc.
        // ========================================================================
        if (mpi_ctx_->rank() == 0)
        {
            LOG_INFO("Running warmup...");
        }

        // Reset pipeline state before warmup
        runner_->clear_cache();

        // Warmup prefill
        auto [warmup_prefill_success, warmup_prefill_time] = runPrefill(tokens);
        if (!warmup_prefill_success)
        {
            if (mpi_ctx_->rank() == 0)
            {
                LOG_ERROR("Warmup prefill failed");
            }
            return result;
        }

        // Warmup decode (if requested)
        if (n_decode > 0)
        {
            int eos_token = tokenizer_->eos_token();
            auto [warmup_decode_success, warmup_decode_time, warmup_tokens, warmup_text] =
                runDecode(n_decode, eos_token);
            if (!warmup_decode_success)
            {
                if (mpi_ctx_->rank() == 0)
                {
                    LOG_ERROR("Warmup decode failed");
                }
                return result;
            }
        }

        if (mpi_ctx_->rank() == 0)
        {
            LOG_INFO("Warmup complete.");
            LOG_INFO("");
            LOG_INFO("Running " << BENCHMARK_ITERATIONS << " benchmark iterations...");
        }

        // Reset profiling after warmup (only track actual benchmark iterations)
        if (KernelProfiler::isEnabled())
        {
            KernelProfiler::reset();
            CUDAKernelProfiler::reset();
            ROCmKernelProfiler::reset();
        }

        // ========================================================================
        // Benchmark Iterations - Run multiple times and average
        // ========================================================================
        std::vector<double> prefill_times;
        std::vector<double> decode_times;
        std::vector<int> decode_token_counts;
        std::string last_generated_text;

        for (int iter = 0; iter < BENCHMARK_ITERATIONS; ++iter)
        {
            // Reset pipeline state before each iteration
            runner_->clear_cache();

            if (mpi_ctx_->rank() == 0)
            {
                LOG_DEBUG("  Iteration " << (iter + 1) << "/" << BENCHMARK_ITERATIONS << "...");
            }

            // Run prefill
            auto [prefill_success, prefill_time] = runPrefill(tokens);
            if (!prefill_success)
            {
                if (mpi_ctx_->rank() == 0)
                {
                    LOG_ERROR("Prefill failed on iteration " << (iter + 1));
                }
                return result;
            }
            prefill_times.push_back(prefill_time);

            // Run decode (if requested)
            if (n_decode > 0)
            {
                int eos_token = tokenizer_->eos_token();
                auto [decode_success, decode_time, tokens_generated, generated_text] =
                    runDecode(n_decode, eos_token);
                if (!decode_success)
                {
                    if (mpi_ctx_->rank() == 0)
                    {
                        LOG_ERROR("Decode failed on iteration " << (iter + 1));
                    }
                    return result;
                }
                decode_times.push_back(decode_time);
                decode_token_counts.push_back(tokens_generated);
                last_generated_text = generated_text;
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
            LOG_INFO("");
            LOG_INFO("Benchmark iterations complete.");
        }

        return result;
    }

    void BenchmarkRunner::printResults(const BenchmarkResult &result) const
    {
        if (mpi_ctx_->rank() != 0)
        {
            return; // Only rank 0 prints
        }

        std::cout << "\n";

        // Title table
        {
            fort::utf8_table title;
            title.set_border_style(FT_DOUBLE2_STYLE);
            std::ostringstream title_ss;
            title_ss << "BENCHMARK RESULTS (average of " << BENCHMARK_ITERATIONS << " runs after warmup)";
            title << title_ss.str() << fort::endr;
            title[0][0].set_cell_text_align(fort::text_align::center);
            title.row(0).set_cell_row_type(fort::row_type::header);
            std::cout << title.to_string();
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

        std::cout << table.to_string();

        // Print kernel profiling summary if enabled
        if (KernelProfiler::isEnabled())
        {
            uint64_t total_tokens = result.prefill_tokens + result.decode_tokens;
            KernelProfiler::printSummary(total_tokens);
            CUDAKernelProfiler::printSummary(total_tokens);
            ROCmKernelProfiler::printSummary(total_tokens);
        }

        // Print executor overhead profiling if enabled (LLAMINAR_PROFILING=1)
        if (KernelProfiler::isEnabled() && runner_)
        {
            const auto *stats = runner_->executorStats();
            if (stats && stats->total_stages_executed > 0)
            {
                // Pass decode tokens for per-token calculations
                stats->printProfilingSummary(result.decode_tokens);
            }
        }

        // Status
        if (result.success)
        {
            std::cout << "\n✓ Benchmark completed successfully.\n";
        }
        else
        {
            std::cout << "\n✗ Benchmark failed.\n";
        }

        std::cout << std::endl;
    }

} // namespace llaminar2
