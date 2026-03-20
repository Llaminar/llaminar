/**
 * @file ServerMode.cpp
 * @brief HTTP server mode with OpenAI-compatible REST API
 *
 * Endpoints:
 *   GET  /health                  — Liveness check
 *   POST /v1/chat/completions     — OpenAI-compatible chat completion
 */

#include "app/modes/ServerMode.h"
#include "app/AppContext.h"
#include "utils/Logger.h"
#include "utils/ChatTemplate.h"

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
        svr.Post("/v1/chat/completions",
                 [&](const httplib::Request &req, httplib::Response &res)
                 {
                     std::lock_guard<std::mutex> lock(inference_mutex);

                     // Parse request body
                     json body;
                     try
                     {
                         body = json::parse(req.body);
                     }
                     catch (const json::parse_error &e)
                     {
                         res.status = 400;
                         json err = {{"error", {{"message", std::string("Invalid JSON: ") + e.what()}, {"type", "invalid_request_error"}}}};
                         res.set_content(err.dump(), "application/json");
                         return;
                     }

                     // Validate required fields
                     if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
                     {
                         res.status = 400;
                         json err = {{"error", {{"message", "\"messages\" field is required and must be a non-empty array"}, {"type", "invalid_request_error"}}}};
                         res.set_content(err.dump(), "application/json");
                         return;
                     }

                     // Extract parameters
                     int max_tokens = body.value("max_tokens", 128);
                     float temperature = body.value("temperature", 0.0f);

                     // Build conversation from messages
                     std::vector<ChatMessage> conversation;
                     for (const auto &msg : body["messages"])
                     {
                         if (!msg.contains("role") || !msg.contains("content"))
                         {
                             res.status = 400;
                             json err = {{"error", {{"message", "Each message must have \"role\" and \"content\" fields"}, {"type", "invalid_request_error"}}}};
                             res.set_content(err.dump(), "application/json");
                             return;
                         }
                         conversation.emplace_back(
                             msg["role"].get<std::string>(),
                             msg["content"].get<std::string>());
                     }

                     // Clear KV cache for fresh conversation
                     runner->clearCache();

                     // Encode with chat template
                     auto token_ids = tokenizer->encodeChat(conversation, /*add_generation_prompt=*/true);
                     if (token_ids.empty())
                     {
                         res.status = 500;
                         json err = {{"error", {{"message", "Failed to encode conversation with chat template"}, {"type", "server_error"}}}};
                         res.set_content(err.dump(), "application/json");
                         return;
                     }

                     int prompt_tokens = static_cast<int>(token_ids.size());

                     // Convert to int32_t
                     std::vector<int32_t> input_ids(token_ids.begin(), token_ids.end());

                     // Prefill
                     if (!runner->prefill(input_ids))
                     {
                         res.status = 500;
                         json err = {{"error", {{"message", std::string("Prefill failed: ") + runner->lastError()}, {"type", "server_error"}}}};
                         res.set_content(err.dump(), "application/json");
                         return;
                     }

                     // Decode loop
                     std::string generated_text;
                     int completion_tokens = 0;

                     for (int i = 0; i < max_tokens; ++i)
                     {
                         GenerationResult result = runner->decodeStep();

                         if (!result.success())
                         {
                             res.status = 500;
                             json err = {{"error", {{"message", std::string("Decode failed: ") + result.error}, {"type", "server_error"}}}};
                             res.set_content(err.dump(), "application/json");
                             return;
                         }

                         if (result.tokens.empty())
                             break;

                         int32_t next_token = result.tokens[0];
                         completion_tokens++;

                         if (result.is_complete || tokenizer->is_stop_token(next_token))
                             break;

                         generated_text += tokenizer->decode_token(next_token);
                     }

                     // Build OpenAI-compatible response
                     json response = {
                         {"id", "chatcmpl-llaminar"},
                         {"object", "chat.completion"},
                         {"choices", json::array({json{{"index", 0},
                                                       {"message", {{"role", "assistant"}, {"content", generated_text}}},
                                                       {"finish_reason", "stop"}}})},
                         {"usage", {{"prompt_tokens", prompt_tokens}, {"completion_tokens", completion_tokens}, {"total_tokens", prompt_tokens + completion_tokens}}}};

                     res.set_content(response.dump(), "application/json");
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
