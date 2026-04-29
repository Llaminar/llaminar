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
#include <vector>

namespace llaminar2
{

    bool SingleShotChatMode::matches(const OrchestrationConfig &config) const
    {
        return config.single_shot_chat;
    }

    int SingleShotChatMode::execute(AppContext &ctx)
    {
        auto &config = ctx.config;
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        if (!tokenizer->hasChatTemplate())
        {
            if (mpi_ctx->rank() == 0)
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
            }
            if (mpi_ctx->world_size() > 1)
                mpi_ctx->barrier();
            runner->shutdown();
            mpiShutdown();
            return 1;
        }

        if (mpi_ctx->rank() == 0)
        {
            LOG_INFO("Running single-shot chat...");
        }

        // Build conversation and encode with chat template (rank 0 only, then broadcast)
        std::vector<int32_t> token_ids;
        int token_count = 0;

        if (mpi_ctx->rank() == 0)
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

        // Broadcast token count first
        if (mpi_ctx->world_size() > 1)
            mpi_ctx->broadcast_int32(&token_count, 1, 0);

        if (token_count <= 0)
        {
            runner->shutdown();
            mpiShutdown();
            return 1;
        }

        // Resize token_ids on non-rank-0 processes and broadcast the tokens
        if (mpi_ctx->rank() != 0)
        {
            token_ids.resize(token_count);
        }
        if (mpi_ctx->world_size() > 1)
            mpi_ctx->broadcast_int32(token_ids.data(), token_count, 0);

        // All ranks participate in prefill
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
            runner->shutdown();
            mpiShutdown();
            return 1;
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
                runner->shutdown();
                mpiShutdown();
                return 1;
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

        runner->shutdown();
        mpiShutdown();
        return 0;
    }

} // namespace llaminar2
