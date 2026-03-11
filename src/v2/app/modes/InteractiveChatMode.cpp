/**
 * @file InteractiveChatMode.cpp
 * @brief Interactive chat mode (--chat)
 */

#include "app/modes/InteractiveChatMode.h"
#include "app/AppContext.h"
#include "app/InferenceRunnerAdapter.h"
#include "utils/Logger.h"
#include "utils/ChatUI.h"
#include <mpi.h>

namespace llaminar2
{

    bool InteractiveChatMode::matches(const OrchestrationConfig &config) const
    {
        return config.chat_mode;
    }

    int InteractiveChatMode::execute(AppContext &ctx)
    {
        auto &config = ctx.config;
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        if (mpi_ctx->rank() == 0)
        {
            if (!tokenizer->hasChatTemplate())
            {
                LOG_ERROR("Chat mode requires a model with a chat template.");
                LOG_ERROR("Use --chat-template to specify one (e.g., --chat-template chatml)");
                MPI_Finalize();
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

            runner->shutdown();
            MPI_Finalize();
            return result;
        }
        else
        {
            // Non-rank-0 processes wait for chat to complete
            if (mpi_ctx->world_size() > 1)
                MPI_Barrier(mpi_ctx->comm());
            runner->shutdown();
            MPI_Finalize();
            return 0;
        }
    }

} // namespace llaminar2
