/**
 * @file ServerMode.cpp
 * @brief HTTP server mode with OpenAI-compatible REST API
 *
 * Endpoints:
 *   GET  /health                  — Liveness check
 *   POST /v1/chat/completions     — OpenAI-compatible chat completion (streaming + non-streaming)
 */

#include "app/modes/ServerMode.h"
#include "app/modes/ChatCompletionHandler.h"
#include "app/AppContext.h"
#include "app/MPIShutdown.h"
#include "utils/Logger.h"

// cpp-httplib (header-only)
#include "httplib.h"

// nlohmann/json (header-only)
#include "nlohmann/json.hpp"

#include <iostream>
#include <mutex>
#include <atomic>
#ifdef __linux__
#include <malloc.h>
#include <fstream>
#endif
#include <csignal>
#include <exception>
#include <filesystem>
#include <string>

using json = nlohmann::json;

namespace llaminar2
{
    namespace
    {
        int finalizeAfterUnhandledException(AppContext &ctx, const std::string &detail)
        {
            const bool has_mpi = ctx.mpi_ctx != nullptr;
            const bool notify_workers = has_mpi && ctx.mpi_ctx->world_size() > 1 && ctx.mpi_ctx->rank() == 0;
            const bool is_root = !has_mpi || ctx.mpi_ctx->rank() == 0;

            if (is_root)
                LOG_ERROR("Server mode failed with unhandled exception: " << detail);

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

    std::unique_ptr<httplib::TaskQueue> createSerializedInferenceTaskQueue()
    {
        return std::make_unique<httplib::ThreadPool>(1);
    }

    int ServerMode::execute(AppContext &ctx)
    try
    {
        auto &config = ctx.config;
        auto &mpi_ctx = ctx.mpi_ctx;
        auto &runner = ctx.runner;
        auto &tokenizer = ctx.tokenizer;

        if (mpi_ctx->world_size() > 1 && mpi_ctx->rank() != 0)
        {
            // Non-root ranks: enter MPI worker loop to participate in
            // inference collectives (allreduce for Global TP) when rank 0
            // initiates them. Returns when rank 0 sends SHUTDOWN.
            LOG_INFO("Rank " << mpi_ctx->rank()
                             << " entering MPI worker loop for inference participation");
            runner->setMPICoordinatedMode(true);
            runner->runMPIWorkerLoop();
            runner->shutdown();
            mpiShutdown();
            return 0;
        }

        if (!tokenizer->hasChatTemplate())
        {
            LOG_ERROR("Server mode requires a model with a chat template.");
            if (mpi_ctx->world_size() > 1)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            mpiShutdown();
            return 1;
        }

        // Enable coordinated mode so rank 0 broadcasts commands to workers
        if (mpi_ctx->world_size() > 1)
            runner->setMPICoordinatedMode(true);

        // Extract model name from path for response metadata
        std::string model_name = std::filesystem::path(config.model_path).stem().string();

        httplib::Server svr;
        g_server_ptr = &svr;

        // Inference is serialized on a single model instance. Keep HTTP handling
        // on one stable worker so OpenMP does not initialize per-request teams
        // on a large rotating httplib thread pool.
        svr.new_task_queue = [] { return createSerializedInferenceTaskQueue().release(); };

        // Install signal handlers for graceful shutdown
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        // Mutex to serialize inference requests (single model instance)
        std::mutex inference_mutex;

        // CORS headers for Open WebUI and other browser-based clients
        svr.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                                 {"Access-Control-Allow-Methods", "GET, POST, OPTIONS"},
                                 {"Access-Control-Allow-Headers", "Content-Type, Authorization"}});

        // Handle CORS preflight
        svr.Options("/v1/chat/completions",
                    [](const httplib::Request &, httplib::Response &res)
                    {
                        res.status = 204;
                    });

        // ─── GET /health ─────────────────────────────────────────────
        svr.Get("/health", [](const httplib::Request &, httplib::Response &res)
                {
            json response = {{"status", "ok"}};
            res.set_content(response.dump(), "application/json"); });

        // ─── POST /v1/chat/completions ───────────────────────────────
        ChatCompletionHandler handler(*runner, *tokenizer, model_name);

        svr.Post("/v1/chat/completions",
                 [&](const httplib::Request &req, httplib::Response &res)
                 {
                     std::lock_guard<std::mutex> lock(inference_mutex);

                     // Check if streaming was requested
                     ChatCompletionResponse parse_error;
                     auto parsed_request = ChatCompletionHandler::parseRequest(req.body, parse_error);

                     if (!parsed_request)
                     {
                         res.status = parse_error.http_status;
                         res.set_content(parse_error.json_body, "application/json");
                         return;
                     }

                     if (parsed_request->stream)
                     {
                         // SSE streaming response
                         res.set_chunked_content_provider(
                             "text/event-stream",
                             [&handler, request = std::move(*parsed_request)](size_t /*offset*/, httplib::DataSink &sink) -> bool
                             {
                                 auto chunk_cb = [&sink](const std::string &sse_line) -> bool
                                 {
                                     return sink.write(sse_line.c_str(), sse_line.size());
                                 };

                                 auto response = handler.handleStreamingRequest(request, chunk_cb);

                                 if (!response.ok && !response.json_body.empty())
                                 {
                                     // Error before streaming started — emit error as SSE
                                     std::string error_sse = "data: " + response.json_body + "\n\ndata: [DONE]\n\n";
                                     sink.write(error_sse.c_str(), error_sse.size());
                                 }

                                 sink.done();
                                 return true;
                             });
                     }
                     else
                     {
                         // Non-streaming response
                         auto response = handler.handleRequest(*parsed_request);
                         res.status = response.http_status;
                         res.set_content(response.json_body, "application/json");
                     }
                 });

        // Start listening
        LOG_INFO("Llaminar server starting on " << config.serve_host << ":" << config.serve_port);

        // Report RSS at server-ready point (after arena + KV cache init)
#ifdef __linux__
        {
            ::malloc_trim(0); // Return freed init memory to OS
            std::ifstream proc_status("/proc/self/status");
            std::string line;
            while (std::getline(proc_status, line))
            {
                if (line.compare(0, 6, "VmRSS:") == 0 ||
                    line.compare(0, 8, "RssAnon:") == 0)
                {
                    LOG_INFO("[ServerReady] " << line);
                }
            }
        }
#endif

        if (!svr.listen(config.serve_host, config.serve_port))
        {
            if (!g_shutdown_requested.load())
            {
                LOG_ERROR("Failed to start server on " << config.serve_host << ":" << config.serve_port);
                if (mpi_ctx->world_size() > 1)
                    runner->shutdownMPIWorkers();
                runner->shutdown();
                mpiShutdown();
                return 1;
            }
        }

        LOG_INFO("Server shut down.");
        g_server_ptr = nullptr;

        // Signal non-root ranks to exit their worker loops
        if (mpi_ctx->world_size() > 1)
            runner->shutdownMPIWorkers();

        runner->shutdown();
        mpiShutdown();
        return 0;
    }
    catch (const std::exception &e)
    {
        return finalizeAfterUnhandledException(ctx, e.what());
    }
    catch (...)
    {
        return finalizeAfterUnhandledException(ctx, "unknown exception");
    }

} // namespace llaminar2
