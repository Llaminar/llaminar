/**
 * @file SequentialBatchBenchmark.cpp
 * @brief Implementation of sequential batch benchmarking
 * @author David Sanftenberg
 */

#include "SequentialBatchBenchmark.h"
#include "Logger.h"
#include "MpiContext.h"
#include "QwenPipeline.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <mpi.h>

namespace llaminar
{
    namespace benchmark
    {

        void SequentialBatchMetrics::print() const
        {
            std::cout << "\n";
            std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║ SEQUENTIAL BATCH BENCHMARK RESULTS                           ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Batch Configuration                                          ║\n";
            std::cout << "║   Batch Size:   " << std::right << std::setw(8) << batch_size << " sequences                           ║\n";
            std::cout << "║   Processed:    " << std::right << std::setw(8) << total_sequences_processed << " sequences                           ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ PREFILL (Aggregate)                                          ║\n";
            std::cout << "║   Total Tokens: " << std::right << std::setw(8) << total_prefill_tokens << " tokens                              ║\n";
            std::cout << "║   Total Time:   " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << total_prefill_time_ms << " ms                                 ║\n";
            std::cout << "║   Throughput:   " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << (total_prefill_tokens * 1000.0 / total_prefill_time_ms) << " tok/s                             ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ DECODE (Aggregate)                                           ║\n";
            std::cout << "║   Total Tokens: " << std::right << std::setw(8) << total_decode_tokens << " tokens                              ║\n";
            std::cout << "║   Total Time:   " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << total_decode_time_ms << " ms                                 ║\n";
            std::cout << "║   Throughput:   " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << (total_decode_tokens * 1000.0 / total_decode_time_ms) << " tok/s                             ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ TOTAL (All Sequences)                                        ║\n";
            std::cout << "║   Total Tokens: " << std::right << std::setw(8) << (total_prefill_tokens + total_decode_tokens) << " tokens                              ║\n";
            std::cout << "║   Total Time:   " << std::right << std::setw(9) << std::fixed << std::setprecision(2)
                      << (total_prefill_time_ms + total_decode_time_ms) << " ms                                 ║\n";
            std::cout << "║   Throughput:   " << std::right << std::setw(10) << std::fixed << std::setprecision(2)
                      << aggregate_tokens_per_sec << " tok/s                             ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Per-Sequence Averages                                        ║\n";
            std::cout << "║   Avg Tokens:   " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << avg_tokens_per_sequence << " tokens/seq                        ║\n";
            std::cout << "║   Avg Time:     " << std::right << std::setw(9) << std::fixed << std::setprecision(2) << avg_time_per_sequence_ms << " ms/seq                            ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
            std::cout << "\n";
        }

        SequentialBatchMetrics runSequentialBatch(
            AbstractPipeline &pipeline,
            const QwenModelWeights &weights,
            chat::TokenizerInterface &tokenizer,
            const std::vector<std::string> &prompts,
            int n_predict)
        {
            SequentialBatchMetrics metrics;
            metrics.batch_size = static_cast<int>(prompts.size());

            auto rank = MPIContext::capture().rank;

            if (rank == 0)
            {
                std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
                std::cout << "║ Starting Sequential Batch Benchmark                         ║\n";
                std::cout << "║   Batch Size: " << std::setw(8) << metrics.batch_size << " sequences                               ║\n";
                std::cout << "║   Tokens/seq: " << std::setw(8) << n_predict << " tokens                                 ║\n";
                std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
            }

            // Process each sequence sequentially
            for (size_t seq_idx = 0; seq_idx < prompts.size(); ++seq_idx)
            {
                // Reset pipeline position for new sequence
                // Cast to QwenPipeline to access setSequencePosition
                auto *qwen_pipeline = dynamic_cast<QwenPipeline *>(&pipeline);
                if (qwen_pipeline)
                {
                    qwen_pipeline->setSequencePosition(0);
                }

                if (rank == 0)
                {
                    std::cout << "[Sequence " << (seq_idx + 1) << "/" << prompts.size() << "] ";
                    std::cout << "Prompt: \"" << prompts[seq_idx].substr(0, 50);
                    if (prompts[seq_idx].length() > 50)
                        std::cout << "...";
                    std::cout << "\"\n";
                    std::cout << "  Tokenizing..." << std::flush;
                }

                // Tokenize prompt (rank 0 only)
                std::vector<int> tokens;
                if (rank == 0)
                {
                    tokens = tokenizer.tokenize(prompts[seq_idx]);
                    std::cout << " " << tokens.size() << " tokens\n";
                    std::cout << "  Prefill..." << std::flush;
                }

                // Broadcast tokens to all ranks
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

                // Prefill phase
                auto prefill_start = std::chrono::high_resolution_clock::now();
                StageContext stage_ctx;
                if (!pipeline.prefill(tokens, weights, stage_ctx))
                {
                    if (rank == 0)
                    {
                        std::cerr << "\n  ERROR: Prefill failed for sequence " << (seq_idx + 1) << "\n";
                    }
                    return metrics;
                }
                auto prefill_end = std::chrono::high_resolution_clock::now();
                double prefill_ms = std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

                metrics.total_prefill_tokens += token_count;
                metrics.total_prefill_time_ms += prefill_ms;

                if (rank == 0)
                {
                    std::cout << " done (" << std::fixed << std::setprecision(1) << prefill_ms << "ms, "
                              << std::setprecision(2) << (token_count * 1000.0 / prefill_ms) << " tok/s)\n";
                }

                // Decode phase
                if (n_predict > 0)
                {
                    if (rank == 0)
                    {
                        std::cout << "  Decode..." << std::flush;
                    }

                    auto decode_start = std::chrono::high_resolution_clock::now();
                    int tokens_generated = 0;

                    for (int i = 0; i < n_predict; ++i)
                    {
                        // Get logits from last prefill/decode step
                        std::shared_ptr<TensorBase> logits;
                        if (!pipeline.logits(logits))
                        {
                            if (rank == 0)
                            {
                                std::cerr << "\n  ERROR: Failed to get logits at token " << i << "\n";
                            }
                            break;
                        }

                        // Greedy sampling: argmax
                        int next_token = 0;
                        if (rank == 0)
                        {
                            const float *logits_data = logits->data();
                            size_t vocab_size = logits->size();
                            float max_logit = logits_data[0];
                            for (size_t j = 1; j < vocab_size; ++j)
                            {
                                if (logits_data[j] > max_logit)
                                {
                                    max_logit = logits_data[j];
                                    next_token = static_cast<int>(j);
                                }
                            }
                        }

                        // Broadcast selected token to all ranks
                        MPI_Bcast(&next_token, 1, MPI_INT, 0, MPI_COMM_WORLD);

                        // Decode single token
                        if (!pipeline.decode({next_token}, weights, stage_ctx))
                        {
                            if (rank == 0)
                            {
                                std::cerr << "\n  ERROR: Decode failed at token " << i << "\n";
                            }
                            break;
                        }

                        tokens_generated++;
                    }

                    auto decode_end = std::chrono::high_resolution_clock::now();
                    double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();

                    metrics.total_decode_tokens += tokens_generated;
                    metrics.total_decode_time_ms += decode_ms;

                    if (rank == 0)
                    {
                        std::cout << " done (" << tokens_generated << " tokens, "
                                  << std::fixed << std::setprecision(1) << decode_ms << "ms, "
                                  << std::setprecision(2) << (tokens_generated * 1000.0 / decode_ms) << " tok/s)\n";
                    }
                }

                metrics.total_sequences_processed++;

                if (rank == 0)
                {
                    std::cout << "\n";
                }
            }

            // Calculate aggregate metrics
            double total_time_ms = metrics.total_prefill_time_ms + metrics.total_decode_time_ms;
            int total_tokens = metrics.total_prefill_tokens + metrics.total_decode_tokens;
            metrics.aggregate_tokens_per_sec = (total_tokens * 1000.0) / total_time_ms;
            metrics.avg_tokens_per_sequence = static_cast<double>(total_tokens) / metrics.total_sequences_processed;
            metrics.avg_time_per_sequence_ms = total_time_ms / metrics.total_sequences_processed;

            if (rank == 0)
            {
                metrics.print();
            }

            return metrics;
        }

    } // namespace benchmark
} // namespace llaminar
