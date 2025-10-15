/**
 * @file BenchmarkRunner.cpp
 * @brief Implementation of clean benchmark runner
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#include "BenchmarkRunner.h"
#include "Logger.h"
#include "MpiContext.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>

namespace llaminar
{
    namespace benchmark
    {

        void BenchmarkMetrics::print() const
        {
            auto rank = MPIContext::capture().rank;
            if (rank != 0)
                return; // Only rank 0 prints

            std::cout << "\n";
            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                    INFERENCE BENCHMARK                       ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Model: " << std::left << std::setw(54) << model_path << "║\n";
            std::cout << "║ Backend: " << std::left << std::setw(52) << backend << "║\n";

            // Only show prefill phase if tokens > 0
            if (prefill_tokens > 0)
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ PREFILL PHASE                                                ║\n";
                std::cout << "║   Tokens:       " << std::right << std::setw(8) << prefill_tokens << " tokens                              ║\n";
                std::cout << "║   Time:         " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << prefill_time_ms << " ms                                 ║\n";
                std::cout << "║   Throughput:   " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << prefill_tokens_per_sec << " tok/s                             ║\n";
            }
            else
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ PREFILL PHASE                                   (SKIPPED)    ║\n";
            }

            // Only show decode phase if tokens > 0
            if (decode_tokens > 0)
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ DECODE PHASE                                                 ║\n";
                std::cout << "║   Tokens:       " << std::right << std::setw(8) << decode_tokens << " tokens                              ║\n";
                std::cout << "║   Time:         " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << decode_time_ms << " ms                                 ║\n";
                std::cout << "║   Throughput:   " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << decode_tokens_per_sec << " tok/s                             ║\n";
            }
            else
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ DECODE PHASE                                    (SKIPPED)    ║\n";
            }

            // Show total only if both phases ran
            if (prefill_tokens > 0 && decode_tokens > 0)
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ TOTAL                                                        ║\n";
                std::cout << "║   Tokens:       " << std::right << std::setw(8) << (prefill_tokens + decode_tokens) << " tokens                              ║\n";
                std::cout << "║   Time:         " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << total_time_ms << " ms                                 ║\n";
                std::cout << "║   Throughput:   " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << total_tokens_per_sec << " tok/s                             ║\n";
            }

            std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
            std::cout << "\n";
        }

        BenchmarkMetrics runInferenceBenchmark(
            AbstractPipeline &pipeline,
            const QwenModelWeights &weights,
            chat::TokenizerInterface &tokenizer,
            const LlaminarParams &params)
        {
            BenchmarkMetrics metrics;
            metrics.model_path = params.model_file;

            auto rank = MPIContext::capture().rank;

            // Tokenize the prompt (rank 0 only)
            std::vector<int> tokens;
            if (rank == 0)
            {
                std::cout << "Tokenizing prompt..." << std::flush;
                tokens = tokenizer.tokenize(params.prompt);
                std::cout << " done (" << tokens.size() << " tokens)\n";
                std::cout << "Tokens: [";
                for (size_t i = 0; i < std::min(tokens.size(), size_t(10)); ++i)
                {
                    std::cout << tokens[i];
                    if (i < std::min(tokens.size(), size_t(10)) - 1)
                        std::cout << ", ";
                }
                if (tokens.size() > 10)
                    std::cout << ", ...";
                std::cout << "]\n";
                std::cout << "Running prefill..." << std::flush;
            }

            // Broadcast token count and tokens to all ranks (MPI synchronization)
            int token_count = 0;
            if (rank == 0)
            {
                token_count = static_cast<int>(tokens.size());
            }
            MPI_Bcast(&token_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (rank != 0)
            {
                tokens.resize(token_count);
            }
            MPI_Bcast(tokens.data(), token_count, MPI_INT, 0, MPI_COMM_WORLD);

            metrics.prefill_tokens = token_count;

            // ============================================
            // PREFILL PHASE (skip if token_count == 0)
            // ============================================
            StageContext stage_ctx;

            if (token_count > 0)
            {
                auto prefill_start = std::chrono::high_resolution_clock::now();

                // Run prefill (all tokens at once)
                if (!pipeline.prefill(tokens, weights, stage_ctx))
                {
                    if (rank == 0)
                    {
                        std::cerr << "\nPrefill failed!\n";
                    }
                    return metrics;
                }

                auto prefill_end = std::chrono::high_resolution_clock::now();
                metrics.prefill_time_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
                metrics.prefill_tokens_per_sec = (metrics.prefill_tokens * 1000.0) / metrics.prefill_time_ms;

                if (rank == 0)
                {
                    std::cout << " done (" << std::fixed << std::setprecision(2)
                              << metrics.prefill_time_ms << " ms, "
                              << metrics.prefill_tokens_per_sec << " tok/s)\n";
                    std::cout << "Running decode..." << std::flush;
                }
            }
            else
            {
                if (rank == 0)
                {
                    std::cout << " skipped (0 tokens)\n";
                    std::cout << "Running decode..." << std::flush;
                }
            }

            // ============================================
            // DECODE PHASE (skip if n_predict == 0)
            // ============================================
            std::vector<int> generated_tokens;
            int max_new_tokens = params.n_predict;

            if (max_new_tokens == 0)
            {
                if (rank == 0)
                {
                    std::cout << " skipped (0 tokens requested)\n";
                }

                // Finalize metrics
                metrics.decode_tokens = 0;
                metrics.decode_time_ms = 0.0;
                metrics.decode_tokens_per_sec = 0.0;
                metrics.total_time_ms = metrics.prefill_time_ms;
                metrics.total_tokens_per_sec = metrics.prefill_tokens_per_sec;
                metrics.backend = "OpenBLAS";

                return metrics;
            }

            auto decode_start = std::chrono::high_resolution_clock::now();

            // Get EOS token ID
            int32_t eos_token = tokenizer.getSpecialToken("<|endoftext|>");
            if (eos_token < 0)
            {
                eos_token = tokenizer.getSpecialToken("<|im_end|>"); // Try alternative
            }

            for (int i = 0; i < max_new_tokens; ++i)
            {
                // Fetch logits from pipeline (after prefill or previous decode)
                std::shared_ptr<TensorBase> latest_logits;
                if (!pipeline.logits(latest_logits) || !latest_logits)
                {
                    if (rank == 0)
                    {
                        std::cerr << "\nFailed to fetch logits at decode step " << i << "\n";
                    }
                    break;
                }

                // Greedy sampling: pick token with highest logit (rank 0 only)
                int next_token = 0;
                if (rank == 0)
                {
                    // Get last row of logits tensor (2D: [seq_len, vocab_size])
                    const auto &shape = latest_logits->shape();
                    if (shape.size() == 2)
                    {
                        int rows = shape[0];
                        int cols = shape[1]; // vocab_size
                        const float *logits_data = latest_logits->data();
                        size_t offset = (rows - 1) * cols;

                        float max_logit = logits_data[offset];
                        for (int j = 1; j < cols; ++j)
                        {
                            if (logits_data[offset + j] > max_logit)
                            {
                                max_logit = logits_data[offset + j];
                                next_token = j;
                            }
                        }
                    }
                }

                // Broadcast next token to all ranks
                MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);

                // Check for EOS token
                if (eos_token >= 0 && next_token == eos_token)
                {
                    break;
                }

                generated_tokens.push_back(next_token);

                // Decode single token
                if (!pipeline.decode(next_token, weights, stage_ctx))
                {
                    if (rank == 0)
                    {
                        std::cerr << "\nDecode failed at token " << i << "\n";
                    }
                    break;
                }
            }

            auto decode_end = std::chrono::high_resolution_clock::now();
            metrics.decode_tokens = static_cast<int>(generated_tokens.size());
            metrics.decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

            if (metrics.decode_tokens > 0)
            {
                metrics.decode_tokens_per_sec = (metrics.decode_tokens * 1000.0) / metrics.decode_time_ms;
            }

            // Total metrics
            metrics.total_time_ms = metrics.prefill_time_ms + metrics.decode_time_ms;
            int total_tokens = metrics.prefill_tokens + metrics.decode_tokens;
            if (total_tokens > 0)
            {
                metrics.total_tokens_per_sec = (total_tokens * 1000.0) / metrics.total_time_ms;
            }

            // Detect backend
            metrics.backend = "OpenBLAS"; // Default, could be enhanced to detect COSMA usage

            if (rank == 0)
            {
                std::cout << " done (" << std::fixed << std::setprecision(2)
                          << metrics.decode_time_ms << " ms, "
                          << metrics.decode_tokens_per_sec << " tok/s)\n";

                // Decode generated text
                std::string generated_text;
                for (int token_id : generated_tokens)
                {
                    generated_text += tokenizer.detokenize({token_id});
                }
                std::cout << "\nGenerated text:\n"
                          << generated_text << "\n";
            }

            return metrics;
        }

    } // namespace benchmark
} // namespace llaminar
