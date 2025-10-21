/**
 * @file BenchmarkRunner.cpp
 * @brief Implementation of clean benchmark runner
 * @author David Sanftenberg
 * @date 2025-10-15
 */

#include "BenchmarkRunner.h"
#include "Logger.h"
#include "MpiContext.h"
#include "MemoryTracker.h"
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

            // Capture memory usage at end of benchmark
            double memory_mb = MemoryTracker::getResidentMemoryMB();

            std::cout << "\n";
            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                    INFERENCE BENCHMARK                       ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Model: " << std::left << std::setw(54) << model_path << "║\n";
            std::cout << "║ Backend: " << std::left << std::setw(52) << backend << "║\n";
            if (memory_mb > 0)
            {
                std::cout << "║ Memory Usage: " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << memory_mb << " MB                                  ║\n";
            }

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

        void BatchBenchmarkMetrics::print() const
        {
            auto rank = MPIContext::capture().rank;
            if (rank != 0)
                return; // Only rank 0 prints

            std::cout << "\n";
            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║                   BATCH INFERENCE BENCHMARK                  ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Model: " << std::left << std::setw(54) << model_path << "║\n";
            std::cout << "║ Backend: " << std::left << std::setw(52) << backend << "║\n";
            std::cout << "║ Batch Size: " << std::right << std::setw(6) << batch_size << " sequences" << std::left << std::setw(34) << "" << "║\n";
            std::cout << "║ Completed: " << std::right << std::setw(7) << sequences_completed << " sequences" << std::left << std::setw(34) << "" << "║\n";

            // Prefill phase
            if (prefill_tokens_total > 0)
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ PREFILL PHASE                                                ║\n";
                std::cout << "║   Total Tokens:     " << std::right << std::setw(8) << prefill_tokens_total << " tokens                          ║\n";
                std::cout << "║   Time:             " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << prefill_time_ms << " ms                             ║\n";
                std::cout << "║   Throughput:       " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << prefill_throughput << " tok/s (aggregate)             ║\n";
                std::cout << "║   Latency/Token:    " << std::right << std::setw(10) << std::fixed << std::setprecision(4) << prefill_latency_per_token << " ms/token                      ║\n";
            }
            else
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ PREFILL PHASE                                   (SKIPPED)    ║\n";
            }

            // Decode phase
            if (decode_tokens_total > 0)
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ DECODE PHASE                                                 ║\n";
                std::cout << "║   Total Tokens:     " << std::right << std::setw(8) << decode_tokens_total << " tokens                          ║\n";
                std::cout << "║   Time:             " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << decode_time_ms << " ms                             ║\n";
                std::cout << "║   Throughput:       " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << decode_throughput << " tok/s (aggregate)             ║\n";
                std::cout << "║   Latency/Token:    " << std::right << std::setw(10) << std::fixed << std::setprecision(4) << decode_latency_per_token << " ms/token                      ║\n";
            }
            else
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ DECODE PHASE                                    (SKIPPED)    ║\n";
            }

            // Total and performance analysis
            if (prefill_tokens_total > 0 || decode_tokens_total > 0)
            {
                std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                std::cout << "║ TOTAL                                                        ║\n";
                std::cout << "║   Total Tokens:     " << std::right << std::setw(8) << (prefill_tokens_total + decode_tokens_total) << " tokens                          ║\n";
                std::cout << "║   Time:             " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << total_time_ms << " ms                             ║\n";
                std::cout << "║   Throughput:       " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << total_throughput << " tok/s (aggregate)             ║\n";

                if (memory_bandwidth_gbps > 0.0)
                {
                    std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
                    std::cout << "║ PERFORMANCE ANALYSIS                                         ║\n";
                    std::cout << "║   Memory Bandwidth: " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << memory_bandwidth_gbps << " GB/s                          ║\n";
                    std::cout << "║   Efficiency:       " << std::right << std::setw(10) << std::fixed << std::setprecision(1) << efficiency_percent << " %                             ║\n";
                }
            }

            std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
            std::cout << "\n";
        }

        BatchBenchmarkMetrics runBatchBenchmark(
            AbstractPipeline &pipeline,
            const QwenModelWeights &weights,
            chat::TokenizerInterface &tokenizer,
            const LlaminarParams &params,
            int batch_size,
            size_t model_size_bytes)
        {
            BatchBenchmarkMetrics metrics;
            metrics.model_path = params.model_file;
            metrics.batch_size = batch_size;

            auto rank = MPIContext::capture().rank;

            // Generate or use prompts for batch
            std::vector<std::vector<int>> token_batches;

            if (rank == 0)
            {
                std::cout << "Preparing batch of " << batch_size << " sequences...\n";

                // For now, use the same prompt for all sequences
                // TODO: Support varied prompts via parameters
                std::vector<int> base_tokens = tokenizer.tokenize(params.prompt);

                std::cout << "Base prompt tokenized: " << base_tokens.size() << " tokens\n";

                for (int i = 0; i < batch_size; ++i)
                {
                    token_batches.push_back(base_tokens);
                }

                std::cout << "Batch prepared: " << batch_size << " sequences × "
                          << base_tokens.size() << " tokens\n";
                std::cout << "Running prefillBatch..." << std::flush;
            }

            // Broadcast batch configuration to all ranks
            int tokens_per_seq = 0;
            if (rank == 0)
            {
                tokens_per_seq = static_cast<int>(token_batches[0].size());
            }
            MPI_Bcast(&tokens_per_seq, 1, MPI_INT, 0, MPI_COMM_WORLD);

            // Non-rank-0 processes prepare empty token batches
            if (rank != 0)
            {
                token_batches.resize(batch_size);
                for (int i = 0; i < batch_size; ++i)
                {
                    token_batches[i].resize(tokens_per_seq);
                }
            }

            // Broadcast each sequence
            for (int i = 0; i < batch_size; ++i)
            {
                MPI_Bcast(token_batches[i].data(), tokens_per_seq, MPI_INT, 0, MPI_COMM_WORLD);
            }

            // ============================================
            // PREFILL BATCH PHASE
            // ============================================
            StageContext stage_ctx;
            std::shared_ptr<TensorBase> prefill_logits;

            if (tokens_per_seq > 0)
            {
                auto prefill_start = std::chrono::high_resolution_clock::now();

                // Run prefillBatch (all sequences at once)
                if (!pipeline.prefillBatch(token_batches, weights, stage_ctx, prefill_logits))
                {
                    if (rank == 0)
                    {
                        std::cerr << "\nPrefillBatch failed!\n";
                    }
                    return metrics;
                }

                auto prefill_end = std::chrono::high_resolution_clock::now();

                metrics.prefill_tokens_total = batch_size * tokens_per_seq;
                metrics.prefill_time_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
                metrics.prefill_throughput = (metrics.prefill_tokens_total * 1000.0) / metrics.prefill_time_ms;
                metrics.prefill_latency_per_token = metrics.prefill_time_ms / metrics.prefill_tokens_total;

                if (rank == 0)
                {
                    std::cout << " done (" << std::fixed << std::setprecision(2)
                              << metrics.prefill_time_ms << " ms, "
                              << metrics.prefill_throughput << " tok/s)\n";
                    std::cout << "Running decodeBatch..." << std::flush;
                }
            }

            // ============================================
            // DECODE BATCH PHASE
            // ============================================
            int max_new_tokens = params.n_predict;

            if (max_new_tokens == 0)
            {
                if (rank == 0)
                {
                    std::cout << " skipped (0 tokens requested)\n";
                }

                metrics.sequences_completed = batch_size;
                metrics.total_time_ms = metrics.prefill_time_ms;
                metrics.total_throughput = metrics.prefill_throughput;
                metrics.backend = "OpenBLAS";

                return metrics;
            }

            auto decode_start = std::chrono::high_resolution_clock::now();

            // Track generated tokens for each sequence
            std::vector<std::vector<int>> generated_tokens(batch_size);
            std::vector<bool> seq_finished(batch_size, false);

            // Get EOS token ID
            int32_t eos_token = tokenizer.getSpecialToken("<|endoftext|>");
            if (eos_token < 0)
            {
                eos_token = tokenizer.getSpecialToken("<|im_end|>");
            }

            int active_sequences = batch_size;

            for (int step = 0; step < max_new_tokens && active_sequences > 0; ++step)
            {
                // Fetch logits from pipeline
                std::shared_ptr<TensorBase> decode_logits;
                if (!pipeline.logits(decode_logits) || !decode_logits)
                {
                    if (rank == 0)
                    {
                        std::cerr << "\nFailed to fetch logits at decode step " << step << "\n";
                    }
                    break;
                }

                // Greedy sampling for each sequence (rank 0 only)
                std::vector<int> next_tokens(batch_size, 0);

                if (rank == 0)
                {
                    const auto &shape = decode_logits->shape();
                    if (shape.size() == 2 && shape[0] == batch_size)
                    {
                        int vocab_size = shape[1];
                        const float *logits_data = decode_logits->data();

                        for (int seq = 0; seq < batch_size; ++seq)
                        {
                            if (seq_finished[seq])
                                continue;

                            size_t offset = seq * vocab_size;
                            float max_logit = logits_data[offset];
                            int best_token = 0;

                            for (int j = 1; j < vocab_size; ++j)
                            {
                                if (logits_data[offset + j] > max_logit)
                                {
                                    max_logit = logits_data[offset + j];
                                    best_token = j;
                                }
                            }

                            next_tokens[seq] = best_token;

                            // Check for EOS
                            if (eos_token >= 0 && best_token == eos_token)
                            {
                                seq_finished[seq] = true;
                                active_sequences--;
                            }
                            else
                            {
                                generated_tokens[seq].push_back(best_token);
                            }
                        }
                    }
                }

                // Broadcast next tokens to all ranks
                MPI_Bcast(next_tokens.data(), batch_size, MPI_INT, 0, MPI_COMM_WORLD);

                // Broadcast finished flags
                std::vector<int> finished_flags(batch_size);
                if (rank == 0)
                {
                    for (int i = 0; i < batch_size; ++i)
                    {
                        finished_flags[i] = seq_finished[i] ? 1 : 0;
                    }
                }
                MPI_Bcast(finished_flags.data(), batch_size, MPI_INT, 0, MPI_COMM_WORLD);

                if (rank != 0)
                {
                    for (int i = 0; i < batch_size; ++i)
                    {
                        seq_finished[i] = (finished_flags[i] != 0);
                    }
                    active_sequences = 0;
                    for (int i = 0; i < batch_size; ++i)
                    {
                        if (!seq_finished[i])
                            active_sequences++;
                    }
                }

                // Decode batch (all sequences)
                if (!pipeline.decodeBatch(next_tokens, weights, stage_ctx, decode_logits))
                {
                    if (rank == 0)
                    {
                        std::cerr << "\nDecodeBatch failed at step " << step << "\n";
                    }
                    break;
                }
            }

            auto decode_end = std::chrono::high_resolution_clock::now();

            // Calculate decode metrics
            int total_decode_tokens = 0;
            if (rank == 0)
            {
                for (const auto &seq_tokens : generated_tokens)
                {
                    total_decode_tokens += seq_tokens.size();
                }
            }
            MPI_Bcast(&total_decode_tokens, 1, MPI_INT, 0, MPI_COMM_WORLD);

            metrics.decode_tokens_total = total_decode_tokens;
            metrics.decode_time_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

            if (metrics.decode_tokens_total > 0)
            {
                metrics.decode_throughput = (metrics.decode_tokens_total * 1000.0) / metrics.decode_time_ms;
                metrics.decode_latency_per_token = metrics.decode_time_ms / metrics.decode_tokens_total;
            }

            // Total metrics
            metrics.sequences_completed = batch_size;
            metrics.total_time_ms = metrics.prefill_time_ms + metrics.decode_time_ms;
            int total_tokens = metrics.prefill_tokens_total + metrics.decode_tokens_total;
            if (total_tokens > 0)
            {
                metrics.total_throughput = (total_tokens * 1000.0) / metrics.total_time_ms;
            }

            // Estimate memory bandwidth (if model size provided)
            if (model_size_bytes > 0 && metrics.total_time_ms > 0)
            {
                // Rough estimate: each token processes the model once
                double total_data_gb = (total_tokens * model_size_bytes) / (1024.0 * 1024.0 * 1024.0);
                double time_sec = metrics.total_time_ms / 1000.0;
                metrics.memory_bandwidth_gbps = total_data_gb / time_sec;

                // Assume theoretical max bandwidth (conservative estimate)
                const double THEORETICAL_MAX_GBPS = 100.0; // Typical DDR4 system
                metrics.efficiency_percent = (metrics.memory_bandwidth_gbps / THEORETICAL_MAX_GBPS) * 100.0;
            }

            metrics.backend = "OpenBLAS"; // TODO: Detect actual backend

            if (rank == 0)
            {
                std::cout << " done (" << std::fixed << std::setprecision(2)
                          << metrics.decode_time_ms << " ms, "
                          << metrics.decode_throughput << " tok/s)\n";

                // Print sample generated text (first sequence only)
                if (!generated_tokens.empty() && !generated_tokens[0].empty())
                {
                    std::string sample_text;
                    for (int token_id : generated_tokens[0])
                    {
                        sample_text += tokenizer.detokenize({token_id});
                    }
                    std::cout << "\nGenerated text (sequence 0):\n"
                              << sample_text << "\n";
                }
            }

            return metrics;
        }

    } // namespace benchmark
} // namespace llaminar
