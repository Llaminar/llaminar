/**
 * @file InteractiveChatMode.cpp
 * @brief Interactive chat mode (--chat)
 *
 * Request termination belongs to this mode; process finalization belongs to
 * the caller's MPIProcessSession. Returning first retires mode-local adapters
 * and handlers before the runner, contexts and outer session are destroyed.
 */

#include "app/modes/InteractiveChatMode.h"
#include "app/AppContext.h"
#include "app/InferenceRunnerAdapter.h"
#include "utils/Logger.h"
#include "utils/ChatUI.h"

namespace llaminar2
{
    namespace
    {
        /**
         * @brief End a failed UI request without finalizing its outer session.
         * @param ctx Active context retaining every request/communicator owner.
         * @param detail Exception diagnostic to publish to waiting followers.
         * @return Nonzero command exit status after runner shutdown.
         */
        int shutdownAfterUnhandledException(AppContext &ctx, const std::string &detail)
        {
            if (ctx.runner)
            {
                if (ctx.coordinatedRequestRole() == CoordinatedRequestRole::Authority)
                {
                    LOG_ERROR("Interactive chat failed: " << detail);
                    if (ctx.mpi_ctx->world_size() > 1)
                        ctx.runner->abortMPIWorkers(detail);
                }
                ctx.runner->shutdown();
            }
            return 1;
        }
    }

    bool InteractiveChatMode::matches(const OrchestrationConfig &config) const
    {
        return config.chat_mode;
    }

    int InteractiveChatMode::execute(AppContext &ctx)
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
        // Followers may already be waiting for commands. Open the authority's
        // terminal channel before any precondition can reject the request.
        if (mpi_coordinated)
            runner->setMPICoordinatedMode(true);
        if (is_authority)
        {
            if (!tokenizer->hasChatTemplate())
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
                if (mpi_coordinated)
                    runner->shutdownMPIWorkers();
                runner->shutdown();
                return 1;
            }

            LOG_INFO("Starting interactive chat mode...");

            ChatUIConfig chat_config;
            chat_config.system_prompt = config.system_prompt;
            chat_config.max_tokens = config.n_predict;
            chat_config.temperature = config.temperature;
            chat_config.top_k = config.top_k;
            chat_config.top_p = config.top_p;

            auto adapter = std::make_shared<InferenceRunnerAdapter>(runner.get());
            ChatUI chat_ui(tokenizer, adapter, chat_config);
            int result = chat_ui.run();

            // Release followers from the coordinated command loop.
            if (mpi_coordinated)
                runner->shutdownMPIWorkers();

            runner->shutdown();
            return result;
        }
        else
        {
            // Followers participate only when the authority admits a command.
            if (mpi_coordinated)
            {
                runner->runMPIWorkerLoop();
            }
            runner->shutdown();
            return 0;
        }
    }
    catch (const std::exception &error)
    {
        return shutdownAfterUnhandledException(ctx, error.what());
    }
    catch (...)
    {
        return shutdownAfterUnhandledException(ctx, "unknown exception");
    }

} // namespace llaminar2
