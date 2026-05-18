/**
 * @file SingleShotChatMode.cpp
 * @brief Single-shot chat mode (--chat-single)
 */

#include "app/modes/SingleShotChatMode.h"
#include "app/AppContext.h"
#include "app/MPIShutdown.h"
#include "utils/Logger.h"
#include "utils/ChatTemplate.h"
#include "utils/Sampler.h"
#include <iostream>
#include <string>
#include <vector>

namespace llaminar2
{
    namespace
    {
        int finalizeAfterUnhandledException(AppContext &ctx, const char *mode_name, const std::string &detail)
        {
            const bool has_mpi = ctx.mpi_ctx != nullptr;
            const bool is_root = !has_mpi || ctx.mpi_ctx->rank() == 0;
            const bool notify_workers = has_mpi && ctx.mpi_ctx->world_size() > 1 && ctx.mpi_ctx->rank() == 0;

            if (is_root)
                LOG_ERROR(mode_name << " failed with unhandled exception: " << detail);

            if (ctx.runner)
            {
                if (notify_workers)
                    ctx.runner->abortMPIWorkers(detail);
                ctx.runner->shutdown();
            }
            mpiShutdown();
            return 1;
        }
    } // namespace

    bool SingleShotChatMode::matches(const OrchestrationConfig &config) const
    {
        return config.single_shot_chat;
    }

    int SingleShotChatMode::execute(AppContext &ctx)
    try
    {
        auto &config = ctx.config;
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        const bool mpi_coordinated = mpi_ctx->world_size() > 1;
        if (mpi_coordinated && mpi_ctx->rank() != 0)
        {
            LOG_DEBUG("Rank " << mpi_ctx->rank()
                             << " entering MPI worker loop for single-shot chat inference");
            runner->setMPICoordinatedMode(true);
            runner->runMPIWorkerLoop();
            runner->shutdown();
            mpiShutdown();
            return 0;
        }

        if (mpi_coordinated)
        {
            runner->setMPICoordinatedMode(true);
        }

        auto shutdownAndFinalize = [&](int exit_code) -> int
        {
            if (mpi_coordinated)
            {
                runner->shutdownMPIWorkers();
            }
            runner->shutdown();
            mpiShutdown();
            return exit_code;
        };

        if (!tokenizer->hasChatTemplate())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
            }
            return shutdownAndFinalize(1);
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running single-shot chat...");
        }

        // Build conversation and encode with chat template on rank 0. Coordinated
        // prefill broadcasts the tokens to worker ranks when MPI is enabled.
        std::vector<int32_t> token_ids;
        int token_count = 0;

        try
        {
            std::vector<ChatMessage> conversation;
            if (!config.system_prompt.empty())
            {
                conversation.push_back(ChatMessage("system", config.system_prompt));
            }
            conversation.push_back(ChatMessage("user", config.prompt));

            auto encoded = tokenizer->encodeChat(conversation, /*add_generation_prompt=*/true);
            token_ids.assign(encoded.begin(), encoded.end());
            token_count = static_cast<int>(token_ids.size());

            if (token_ids.empty())
            {
                LOG_ERROR("Failed to encode conversation with chat template");
                token_count = -1;
            }
            else
            {
                LOG_DEBUG("Encoded " << token_count << " tokens with chat template");
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Error encoding conversation with chat template: " << e.what());
            return shutdownAndFinalize(1);
        }

        if (token_count <= 0)
        {
            return shutdownAndFinalize(1);
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_DEBUG("Running prefill (" << token_count << " tokens)...");
        }

        if (!runner->prefill(token_ids))
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat prefill failed: " << runner->lastError());
            }
            return shutdownAndFinalize(1);
        }

        // Determine max tokens: -1 means unlimited
        int max_tokens = config.n_predict;
        if (max_tokens < 0)
        {
            max_tokens = config.max_seq_len - token_count;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Generating response (max " << max_tokens << " tokens)...\n");
        }

        // Configure sampling params from CLI config
        SamplingParams sampling_params;
        sampling_params.temperature = config.temperature;
        sampling_params.top_k = config.top_k;
        sampling_params.top_p = config.top_p;
        sampling_params.seed = config.seed;
        runner->setSamplingParams(sampling_params);

        // Decode loop
        for (int i = 0; i < max_tokens; ++i)
        {
            GenerationResult result = runner->decodeStep();

            if (!result.success())
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_ERROR("Decode step failed: " << result.error);
                }
                return shutdownAndFinalize(1);
            }

            if (result.tokens.empty())
            {
                break;
            }

            int32_t next_token = result.tokens[0];

            if (mpi_ctx->rank() == 0 && !tokenizer->is_stop_token(next_token))
            {
                std::string token_text = tokenizer->decode_token(next_token);
                std::cout << token_text << std::flush;
            }

            if (result.is_complete || tokenizer->is_stop_token(next_token))
            {
                if (mpi_ctx->rank() == 0)
                {
                    LOG_DEBUG("Stop token encountered (" << next_token << "), stopping generation");
                }
                break;
            }
        }

        // Flush accumulated GPU stage timeline for decode phase
        runner->flushStageTimeline();

        if (mpi_ctx->rank() == 0)
        {
            std::cout << std::endl;
            LOG_INFO("Chat generation complete.");
        }

        return shutdownAndFinalize(0);
    }
    catch (const std::exception &e)
    {
        return finalizeAfterUnhandledException(ctx, "Single-shot chat mode", e.what());
    }
    catch (...)
    {
        return finalizeAfterUnhandledException(ctx, "Single-shot chat mode", "unknown non-std exception");
    }

} // namespace llaminar2
