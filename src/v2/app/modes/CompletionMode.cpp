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

        // Generate tokens autoregressively
        int max_tokens = (config.n_predict == -1) ? INT_MAX : config.n_predict;
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
