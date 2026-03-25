/**
 * @file CompletionMode.cpp
 * @brief Standard one-shot completion mode (default, -p "...")
 */

#include "app/modes/CompletionMode.h"
#include "app/AppContext.h"
#include "utils/Logger.h"
#include "utils/Sampler.h"
#include <mpi.h>
#include <iostream>
#include <climits>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

namespace llaminar2
{

    bool CompletionMode::matches(const OrchestrationConfig & /*config*/) const
    {
        // Default fallback mode — always matches
        return true;
    }

    int CompletionMode::execute(AppContext &ctx)
    {
        auto &config = ctx.config;
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        // Tokenize prompt
        std::vector<int32_t> tokens;
        try
        {
            auto encoded = tokenizer->encode(config.prompt, /*add_bos=*/false, /*add_eos=*/false);
            tokens.assign(encoded.begin(), encoded.end());

            if (tokens.empty())
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_ERROR("Tokenization resulted in empty token sequence");
                }
                runner->shutdown();
                MPI_Finalize();
                return 1;
            }

            if (mpi_ctx->rank() == 0)
            {
                LOG_INFO("Tokenized prompt: " << tokens.size() << " tokens");
                std::ostringstream token_ids_str;
                token_ids_str << "Token IDs: [";
                for (size_t i = 0; i < tokens.size(); ++i)
                {
                    token_ids_str << tokens[i];
                    if (i < tokens.size() - 1)
                        token_ids_str << ", ";
                }
                token_ids_str << "]";
                LOG_INFO(token_ids_str.str());
            }
        }
        catch (const std::exception &e)
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Error tokenizing prompt: " << e.what());
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        // Set up sampling parameters
        SamplingParams sampling_params;
        sampling_params.temperature = config.temperature;
        sampling_params.top_k = config.top_k;
        sampling_params.top_p = config.top_p;
        sampling_params.seed = config.seed;

        if (mpi_ctx->rank() == 0)
        {
            LOG_DEBUG("Sampling parameters:");
            LOG_DEBUG("  temperature: " << sampling_params.temperature);
            LOG_DEBUG("  top_k: " << sampling_params.top_k);
            LOG_DEBUG("  top_p: " << sampling_params.top_p);
            LOG_DEBUG("  seed: " << sampling_params.seed);
        }

        // Run prefill
        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running prefill (" << tokens.size() << " tokens)...");
        }

        if (!runner->prefill(tokens))
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Error: Prefill forward pass failed: " << runner->lastError());
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            if (config.n_predict == -1)
            {
                LOG_DEBUG("Prefill complete. Generating tokens until EOS...\n");
            }
            else
            {
                LOG_DEBUG("Prefill complete. Generating " << config.n_predict << " tokens...\n");
            }
        }

        // Configure GPU-side sampling
        runner->setSamplingParams(sampling_params);
        runner->setSkipLogitsGatherDecode(false); // Ensure logits are gathered for comparison

        // Logit snapshots for R1 vs R2 comparison
        int vocab_size = runner->vocabSize();
        std::vector<std::vector<float>> r1_logits_per_step;

        // Generate tokens autoregressively
        int max_tokens = (config.n_predict == -1) ? INT_MAX : config.n_predict;
        std::vector<int32_t> round1_tokens;
        for (int i = 0; i < max_tokens; ++i)
        {
            LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Starting decode iteration " << i);

            GenerationResult result = runner->decodeStep();

            if (!result.success())
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_ERROR("\nError: Decode step failed at token " << (i + 1) << ": " << result.error);
                }
                runner->shutdown();
                MPI_Finalize();
                return 1;
            }

            if (result.tokens.empty())
            {
                LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] No token generated at iteration " << i);
                break;
            }

            int32_t next_token = result.tokens[0];
            round1_tokens.push_back(next_token);

            // Save logit snapshot for R1
            {
                const float *lg = runner->lastLogits();
                if (lg && vocab_size > 0)
                {
                    r1_logits_per_step.emplace_back(lg, lg + vocab_size);
                }
            }

            LOG_DEBUG("[Rank " << mpi_ctx->rank() << "] Generated token: " << next_token);

            if (mpi_ctx->rank() == 0 && !tokenizer->is_stop_token(next_token))
            {
                std::string token_text = tokenizer->decode_token(next_token);
                std::cout << token_text << std::flush;
            }

            if (result.is_complete || tokenizer->is_stop_token(next_token))
            {
                if (mpi_ctx->rank() == 0 && config.verbose_level > 0)
                {
                    LOG_DEBUG("\nGeneration stopped: stop token " << next_token << " encountered");
                }
                break;
            }
        }

        // Flush accumulated GPU stage timeline for decode phase
        runner->flushStageTimeline();

        if (mpi_ctx->rank() == 0)
        {
            std::cout << "\n"
                      << std::endl;
            LOG_DEBUG("Generation complete.");
        }

        // =====================================================================
        // Round 2: Clear cache and re-run same prompt (reproduction test)
        // =====================================================================
        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("========== ROUND 2: clearCache + re-prefill same tokens ==========");
        }

        runner->clearCache();

        // Re-prefill same tokens
        if (!runner->prefill(tokens))
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Error: Round 2 prefill failed: " << runner->lastError());
            }
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        runner->setSamplingParams(sampling_params);
        runner->setSkipLogitsGatherDecode(false); // Ensure logits are gathered for comparison

        if (mpi_ctx->rank() == 0)
        {
            std::cout << "[Round 2] ";
        }

        // Decode same number of tokens
        std::vector<std::vector<float>> r2_logits_per_step;
        std::vector<int32_t> round2_tokens;
        for (int i = 0; i < max_tokens; ++i)
        {
            GenerationResult result = runner->decodeStep();

            if (!result.success())
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_ERROR("\nRound 2 decode failed at token " << (i + 1) << ": " << result.error);
                }
                break;
            }

            if (result.tokens.empty())
                break;

            int32_t next_token = result.tokens[0];
            round2_tokens.push_back(next_token);

            // Save logit snapshot for R2
            {
                const float *lg = runner->lastLogits();
                if (lg && vocab_size > 0)
                {
                    r2_logits_per_step.emplace_back(lg, lg + vocab_size);
                }
            }

            if (mpi_ctx->rank() == 0 && !tokenizer->is_stop_token(next_token))
            {
                std::string token_text = tokenizer->decode_token(next_token);
                std::cout << token_text << std::flush;
            }

            if (result.is_complete || tokenizer->is_stop_token(next_token))
                break;
        }

        // Flush accumulated GPU stage timeline for round 2 decode phase
        runner->flushStageTimeline();

        if (mpi_ctx->rank() == 0)
        {
            std::cout << "\n"
                      << std::endl;

            // Compare tokens
            bool match = (round1_tokens.size() == round2_tokens.size());
            if (match)
            {
                for (size_t i = 0; i < round1_tokens.size(); ++i)
                {
                    if (round1_tokens[i] != round2_tokens[i])
                    {
                        match = false;
                        break;
                    }
                }
            }

            std::cout << "========== TOKEN COMPARISON ==========" << std::endl;
            std::cout << "Round 1 tokens: [";
            for (size_t i = 0; i < round1_tokens.size(); ++i)
            {
                std::cout << round1_tokens[i];
                if (i < round1_tokens.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "Round 2 tokens: [";
            for (size_t i = 0; i < round2_tokens.size(); ++i)
            {
                std::cout << round2_tokens[i];
                if (i < round2_tokens.size() - 1)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "MATCH: ";
            if (match)
            {
                std::cout << "YES ✓" << std::endl;
            }
            else
            {
                // Compute step-0 RMSE to classify: GPU non-determinism vs real bug
                double step0_rmse = -1.0;
                if (!r1_logits_per_step.empty() && !r2_logits_per_step.empty() && r1_logits_per_step[0].size() == r2_logits_per_step[0].size())
                {
                    double sq = 0;
                    int V = static_cast<int>(r1_logits_per_step[0].size());
                    for (int v = 0; v < V; ++v)
                    {
                        double d = r1_logits_per_step[0][v] - r2_logits_per_step[0][v];
                        sq += d * d;
                    }
                    step0_rmse = std::sqrt(sq / V);
                }
                if (step0_rmse >= 0 && step0_rmse < 0.5)
                    std::cout << "NO — GPU non-determinism (step0 RMSE=" << step0_rmse << ")" << std::endl;
                else
                    std::cout << "NO ✗ — BUG! (step0 RMSE=" << step0_rmse << ")" << std::endl;
            }
            std::cout << "======================================" << std::endl;

            // Logit comparison (always show first 3 steps; full detail on mismatch)
            {
                size_t n_steps = std::min(r1_logits_per_step.size(), r2_logits_per_step.size());
                for (size_t step = 0; step < n_steps; ++step)
                {
                    const auto &l1 = r1_logits_per_step[step];
                    const auto &l2 = r2_logits_per_step[step];
                    if (l1.size() != l2.size())
                        continue;
                    int V = static_cast<int>(l1.size());

                    // Find max absolute difference
                    float max_diff = 0;
                    int max_diff_idx = 0;
                    double sum_sq_diff = 0;
                    int n_diff = 0;
                    for (int v = 0; v < V; ++v)
                    {
                        float d = std::fabs(l1[v] - l2[v]);
                        sum_sq_diff += static_cast<double>(d) * d;
                        if (d > 0)
                            n_diff++;
                        if (d > max_diff)
                        {
                            max_diff = d;
                            max_diff_idx = v;
                        }
                    }
                    double rmse = std::sqrt(sum_sq_diff / V);

                    // Top-5 tokens from each round
                    std::vector<int> idx1(V), idx2(V);
                    std::iota(idx1.begin(), idx1.end(), 0);
                    std::iota(idx2.begin(), idx2.end(), 0);
                    std::partial_sort(idx1.begin(), idx1.begin() + 5, idx1.end(),
                                      [&](int a, int b)
                                      { return l1[a] > l1[b]; });
                    std::partial_sort(idx2.begin(), idx2.begin() + 5, idx2.end(),
                                      [&](int a, int b)
                                      { return l2[a] > l2[b]; });

                    std::cout << "--- Step " << step << " logit comparison ---" << std::endl;
                    std::cout << "  RMSE=" << rmse << " max_diff=" << max_diff
                              << " @token=" << max_diff_idx << " n_diff=" << n_diff << "/" << V << std::endl;

                    // Detailed output only on mismatch or first 3 steps
                    if (!match || step < 3)
                    {
                        std::cout << "  R1 top5: ";
                        for (int t = 0; t < 5; ++t)
                            std::cout << idx1[t] << "(" << l1[idx1[t]] << ") ";
                        std::cout << std::endl;
                        std::cout << "  R2 top5: ";
                        for (int t = 0; t < 5; ++t)
                            std::cout << idx2[t] << "(" << l2[idx2[t]] << ") ";
                        std::cout << std::endl;
                    }

                    // Show if the step's sampled tokens differ
                    if (step < round1_tokens.size() && step < round2_tokens.size() && round1_tokens[step] != round2_tokens[step])
                    {
                        int t1 = round1_tokens[step], t2 = round2_tokens[step];
                        std::cout << "  DIVERGENT: R1 sampled " << t1 << "(logit=" << l1[t1]
                                  << ") R2 sampled " << t2 << "(logit=" << l2[t2] << ")" << std::endl;
                        std::cout << "  R1 logit[" << t2 << "]=" << l1[t2]
                                  << " R2 logit[" << t1 << "]=" << l2[t1] << std::endl;
                    }
                }
            }
        }

        if (mpi_ctx->world_size() > 1)
        {
            mpi_ctx->barrier();
        }

        runner->shutdown();

        if (mpi_ctx->world_size() > 1)
        {
            mpi_ctx->barrier();
        }

        MPI_Finalize();
        return 0;
    }

} // namespace llaminar2
