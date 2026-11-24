/**
 * @file SingleShotChatMode.cpp
 * @brief Single-shot chat mode (--chat-single)
 *
 * Request termination belongs to this mode; process finalization belongs to
 * the caller's MPIProcessSession. Returning first retires mode-local adapters
 * and handlers before the runner, contexts and outer session are destroyed.
 */

#include <iterator>
#include <fstream>
#include "app/modes/SingleShotChatMode.h"
#include "app/AppContext.h"
#include "app/modes/ConsoleOutput.h"
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
        /**
         * @brief Terminate the request channel while retaining outer MPI ownership.
         * @param ctx Active caller-owned application resources.
         * @param mode_name Diagnostic name of this request frontend.
         * @param detail Failure shared with followers before runner retirement.
         * @return Nonzero command exit; the caller's scope finalizes MPI later.
         */
        int shutdownAfterUnhandledException(AppContext &ctx, const char *mode_name, const std::string &detail)
        {
            const bool has_mpi = ctx.mpi_ctx != nullptr;
            const bool is_authority =
                ctx.runner &&
                ctx.coordinatedRequestRole() ==
                    CoordinatedRequestRole::Authority;
            const bool notify_workers =
                has_mpi && ctx.mpi_ctx->world_size() > 1 && is_authority;

            if (is_authority)
                LOG_ERROR(mode_name << " failed with unhandled exception: " << detail);

            if (ctx.runner)
            {
                if (notify_workers)
                    ctx.runner->abortMPIWorkers(detail);
                ctx.runner->shutdown();
            }
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
        const bool is_authority =
            ctx.coordinatedRequestRole() ==
            CoordinatedRequestRole::Authority;
        if (mpi_coordinated && !is_authority)
        {
            LOG_DEBUG("Rank " << mpi_ctx->rank()
                             << " entering MPI worker loop for single-shot chat inference");
            runner->setMPICoordinatedMode(true);
            runner->runMPIWorkerLoop();
            runner->shutdown();
            return 0;
        }

        if (mpi_coordinated)
        {
            runner->setMPICoordinatedMode(true);
        }

        auto shutdownAndReturn = [&](int exit_code) -> int
        {
            if (mpi_coordinated)
            {
                runner->shutdownMPIWorkers();
            }
            runner->shutdown();
            return exit_code;
        };

        if (!tokenizer->hasChatTemplate())
        {
            LOG_ERROR("Chat mode requires a model with a chat template.");
            LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
            return shutdownAndReturn(1);
        }

        LOG_INFO("Running single-shot chat...");

        // The continuation authority owns chat encoding. Coordinated prefill
        // broadcasts the resulting tokens to followers when MPI is enabled.
        std::vector<int32_t> token_ids;
        int token_count = 0;

        try
        {
            std::vector<ChatMessage> conversation;
            if (!config.system_prompt.empty())
            {
                conversation.push_back(ChatMessage("system", config.system_prompt));
            }
            ChatMessage user("user", config.prompt);
            std::vector<std::vector<std::uint8_t>> images;
            if (!config.image_paths.empty())
            {
                if (!ctx.image_encoder)
                    throw std::invalid_argument("--image requires --mmproj");
                // Images precede the prompt text, each as its own content part
                // (OpenAI order: image parts, then the question).
                for (const auto &path : config.image_paths)
                {
                    std::ifstream file(path, std::ios::binary);
                    if (!file) throw std::invalid_argument("cannot read image " + path);
                    images.emplace_back(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
                    user.content_parts.push_back(ctx.image_encoder->placeholderText());
                }
                user.content_parts.push_back(config.prompt);
                user.content.clear();
                for (std::size_t i = 0; i < user.content_parts.size(); ++i)
                    user.content += (i ? "\n" : "") + user.content_parts[i];
            }
            conversation.push_back(std::move(user));

            auto encoded = tokenizer->encodeChat(conversation, /*add_generation_prompt=*/true);
            token_ids.assign(encoded.begin(), encoded.end());
            if (!images.empty())
            {
                auto expanded = ctx.image_encoder->expand(token_ids, images);
                token_ids = std::move(expanded.tokens);
                if (!runner->setNextPromptEmbeddingRows(std::move(expanded.rows)))
                    throw std::runtime_error("runner rejected image input: " + runner->lastError());
                LOG_INFO("Attached " << images.size() << " image(s); prompt expanded to " << token_ids.size()
                                     << " tokens");
            }
            token_count = static_cast<int>(token_ids.size());

            if (token_ids.empty())
            {
                LOG_ERROR("Failed to encode conversation with chat template");
                token_count = -1;
            }
            else
            {
                LOG_DEBUG("Encoded " << token_count << " tokens with chat template: " << [&] {
                    std::string ids;
                    for (const auto id : token_ids) ids += std::to_string(id) + " ";
                    return ids;
                }());
            }
        }
        catch (const std::exception &e)
        {
            LOG_ERROR("Error encoding conversation with chat template: " << e.what());
            return shutdownAndReturn(1);
        }

        if (token_count <= 0)
        {
            return shutdownAndReturn(1);
        }

        LOG_DEBUG("Running prefill (" << token_count << " tokens)...");

        SamplingParams sampling_params;
        sampling_params.temperature = config.temperature;
        sampling_params.top_k = config.top_k;
        sampling_params.top_p = config.top_p;
        sampling_params.seed = config.seed;
        runner->setSamplingParams(sampling_params);
        runner->setStopTokens(tokenizer->stop_tokens());

        if (!runner->prefill(token_ids))
        {
            LOG_ERROR("Chat prefill failed: " << runner->lastError());
            return shutdownAndReturn(1);
        }

        // Determine max tokens: -1 means unlimited
        int max_tokens = config.n_predict;
        if (max_tokens < 0)
        {
            max_tokens = config.max_seq_len - token_count;
        }

        LOG_INFO("Generating response (max " << max_tokens << " tokens)...");
        console_output::printPromptAndResponseHeader(config.prompt);

        // Decode loop. MTP can return multiple accepted tokens in one step, so
        // keep the loop bounded by emitted tokens rather than decode calls.
        int generated_tokens = 0;
        bool stop_generation = false;
        while (generated_tokens < max_tokens && !stop_generation)
        {
            runner->setDecodeStepTokenBudget(max_tokens - generated_tokens);
            GenerationResult result = runner->decodeStep();
            runner->setDecodeStepTokenBudget(0);

            if (!result.success())
            {
                LOG_ERROR("Decode step failed: " << result.error);
                return shutdownAndReturn(1);
            }

            if (result.tokens.empty())
            {
                break;
            }

            for (size_t token_idx = 0; token_idx < result.tokens.size() && generated_tokens < max_tokens; ++token_idx)
            {
                const int32_t next_token = result.tokens[token_idx];
                const bool is_final_returned_token = token_idx + 1 == result.tokens.size();
                const bool is_stop = tokenizer->is_stop_token(next_token) ||
                                     (result.is_complete && is_final_returned_token);

                ++generated_tokens;

                if (!is_stop)
                {
                    std::string token_text = tokenizer->decode_token(next_token);
                    std::cout << token_text << std::flush;
                }

                if (is_stop)
                {
                    LOG_DEBUG("Stop token encountered (" << next_token << "), stopping generation");
                    stop_generation = true;
                    break;
                }
            }
        }

        std::cout << "\n"
                  << std::flush;

        // Flush accumulated GPU stage timeline for decode phase
        runner->flushStageTimeline();

        std::cout << std::endl;
        LOG_INFO("Chat generation complete.");

        return shutdownAndReturn(0);
    }
    catch (const std::exception &e)
    {
        return shutdownAfterUnhandledException(ctx, "Single-shot chat mode", e.what());
    }
    catch (...)
    {
        return shutdownAfterUnhandledException(ctx, "Single-shot chat mode", "unknown non-std exception");
    }

} // namespace llaminar2
