/**
 * @file InteractiveChatMode.cpp
 * @brief Interactive chat mode (--chat)
 */

#include "app/modes/InteractiveChatMode.h"
#include "app/AppContext.h"
#include "app/MPIShutdown.h"
#include "app/InferenceRunnerAdapter.h"
#include "utils/Logger.h"
#include "utils/ChatUI.h"

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
                mpiShutdown();
                return 1;
            }

            LOG_INFO("Starting interactive chat mode...");

            ChatUIConfig chat_config;
            chat_config.system_prompt = config.system_prompt;
            chat_config.max_tokens = config.n_predict;
            chat_config.temperature = config.temperature;
            chat_config.top_k = config.top_k;
            chat_config.top_p = config.top_p;

            // Enable coordinated mode so rank 0 broadcasts to worker ranks
            if (mpi_ctx->world_size() > 1)
                runner->setMPICoordinatedMode(true);

            auto adapter = std::make_shared<InferenceRunnerAdapter>(runner.get());
            ChatUI chat_ui(tokenizer, adapter, chat_config);
            int result = chat_ui.run();

            // Signal non-root ranks to exit their worker loops
            if (mpi_ctx->world_size() > 1)
                runner->shutdownMPIWorkers();

            runner->shutdown();
            mpiShutdown();
            return result;
        }
        else
        {
            // Non-rank-0 processes: enter MPI worker loop to participate
            // in inference collectives when rank 0 initiates them.
            if (mpi_ctx->world_size() > 1)
            {
                runner->setMPICoordinatedMode(true);
                runner->runMPIWorkerLoop();
            }
            runner->shutdown();
            mpiShutdown();
            return 0;
        }
    }

} // namespace llaminar2
