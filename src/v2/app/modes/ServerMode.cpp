/**
 * @file ServerMode.cpp
 * @brief HTTP server mode with OpenAI-compatible REST API
 *
 * Endpoints:
 *   GET  /health                  — Liveness check
 *   POST /v1/chat/completions     — OpenAI-compatible chat completion
 */

#include "app/modes/ServerMode.h"
#include "app/modes/ChatCompletionHandler.h"
#include "app/AppContext.h"
#include "utils/Logger.h"

// cpp-httplib (header-only)
#include "httplib.h"

// nlohmann/json (header-only)
#include "nlohmann/json.hpp"

#include <mpi.h>
#include <iostream>
#include <mutex>
#include <atomic>
#include <csignal>

using json = nlohmann::json;

namespace llaminar2
{

    // Global signal handling for clean shutdown
    static std::atomic<bool> g_shutdown_requested{false};
    static httplib::Server *g_server_ptr = nullptr;

    static void signal_handler(int /*sig*/)
    {
        g_shutdown_requested.store(true);
        if (g_server_ptr)
            g_server_ptr->stop();
    }

    bool ServerMode::matches(const OrchestrationConfig &config) const
    {
        return config.serve_mode;
    }

    int ServerMode::execute(AppContext &ctx)
    {
        auto &config = ctx.config;
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        if (mpi_ctx->world_size() > 1 && mpi_ctx->rank() != 0)
        {
            // Non-root ranks: participate in MPI collectives but don't run HTTP
            // For now, single-rank server only
            LOG_WARN("Server mode only supports single-rank. Rank " << mpi_ctx->rank() << " exiting.");
            runner->shutdown();
            MPI_Finalize();
            return 0;
        }

        if (!tokenizer->hasChatTemplate())
        {
            LOG_ERROR("Server mode requires a model with a chat template.");
            runner->shutdown();
            MPI_Finalize();
            return 1;
        }

        httplib::Server svr;
        g_server_ptr = &svr;

        // Install signal handlers for graceful shutdown
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Mutex to serialize inference requests (single model instance)
        std::mutex inference_mutex;

        // ─── GET /health ─────────────────────────────────────────────
        svr.Get("/health", [](const httplib::Request &, httplib::Response &res)
                {
            json response = {{"status", "ok"}};
            res.set_content(response.dump(), "application/json"); });

        // ─── POST /v1/chat/completions ───────────────────────────────
        ChatCompletionHandler handler(*runner, *tokenizer);

        svr.Post("/v1/chat/completions",
                 [&](const httplib::Request &req, httplib::Response &res)
                 {
                     std::lock_guard<std::mutex> lock(inference_mutex);

                     auto response = handler.handleRawRequest(req.body);
                     res.status = response.http_status;
                     res.set_content(response.json_body, "application/json");
                 });

        // Start listening
        LOG_INFO("Llaminar server starting on " << config.serve_host << ":" << config.serve_port);

        if (!svr.listen(config.serve_host, config.serve_port))
        {
            if (!g_shutdown_requested.load())
            {
                LOG_ERROR("Failed to start server on " << config.serve_host << ":" << config.serve_port);
                runner->shutdown();
                MPI_Finalize();
                return 1;
            }
        }

        LOG_INFO("Server shut down.");
        g_server_ptr = nullptr;
        runner->shutdown();
        MPI_Finalize();
        return 0;
    }

} // namespace llaminar2
