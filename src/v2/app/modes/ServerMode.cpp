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
#include <chrono>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#ifdef __linux__
#include <malloc.h>
#include <fstream>
#endif
#include <csignal>
#include <exception>
#include <filesystem>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

namespace llaminar2
{
    namespace
    {
        using SteadyClock = std::chrono::steady_clock;

        struct RequestLogContext
        {
            SteadyClock::time_point started_at{};
            std::shared_ptr<std::string> streamed_response_body;
        };

        class RequestLogState
        {
        public:
            void start(const httplib::Request &req)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                contexts_[&req].started_at = SteadyClock::now();
            }

            void attachStreamedResponseBody(const httplib::Request &req,
                                            std::shared_ptr<std::string> body)
            {
                if (!body)
                    return;

                std::lock_guard<std::mutex> lock(mutex_);
                contexts_[&req].streamed_response_body = std::move(body);
            }

            RequestLogContext finish(const httplib::Request &req)
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = contexts_.find(&req);
                if (it == contexts_.end())
                    return {};

                RequestLogContext context = std::move(it->second);
                contexts_.erase(it);
                return context;
            }

        private:
            std::mutex mutex_;
            std::unordered_map<const httplib::Request *, RequestLogContext> contexts_;
        };

        bool traceAccessLoggingEnabled()
        {
#if defined(NDEBUG)
            return false;
#else
            return Logger::getInstance().shouldLog(LogLevel::TRACE);
#endif
        }

        std::string escapeAccessField(const std::string &value)
        {
            if (value.empty())
                return "-";

            std::string escaped;
            escaped.reserve(value.size());
            for (char c : value)
            {
                switch (c)
                {
                case '\\':
                    escaped += "\\\\";
                    break;
                case '"':
                    escaped += "\\\"";
                    break;
                case '\n':
                    escaped += "\\n";
                    break;
                case '\r':
                    escaped += "\\r";
                    break;
                case '\t':
                    escaped += "\\t";
                    break;
                default:
                    escaped += c;
                    break;
                }
            }
            return escaped;
        }

        std::string requestTarget(const httplib::Request &req)
        {
            if (!req.target.empty())
                return req.target;
            if (!req.path.empty())
                return req.path;
            return "-";
        }

        std::string remoteAddress(const httplib::Request &req)
        {
            if (req.remote_addr.empty())
                return "-";
            if (req.remote_port < 0)
                return req.remote_addr;
            return req.remote_addr + ":" + std::to_string(req.remote_port);
        }

        std::optional<double> elapsedMs(const RequestLogContext &context)
        {
            if (context.started_at == SteadyClock::time_point{})
                return std::nullopt;

            return std::chrono::duration<double, std::milli>(
                       SteadyClock::now() - context.started_at)
                .count();
        }

        std::optional<size_t> responseBodyBytes(const httplib::Response &res,
                                                const RequestLogContext &context)
        {
            if (context.streamed_response_body)
                return context.streamed_response_body->size();
            if (!res.body.empty())
                return res.body.size();
            if (res.has_header("Content-Length"))
                return static_cast<size_t>(res.get_header_value_u64("Content-Length"));
            if (res.status == 204 || res.status == 304)
                return static_cast<size_t>(0);
            return std::nullopt;
        }

        std::string formatDuration(const std::optional<double> &duration_ms)
        {
            if (!duration_ms)
                return "-";

            std::ostringstream out;
            out << std::fixed << std::setprecision(1) << *duration_ms << "ms";
            return out.str();
        }

        std::string formatBytes(const std::optional<size_t> &bytes)
        {
            if (!bytes)
                return "-";
            return std::to_string(*bytes) + "B";
        }

        std::string formatAccessLogLine(const httplib::Request &req,
                                        const httplib::Response &res,
                                        const RequestLogContext &context)
        {
            const std::string version = req.version.empty() ? "HTTP/?" : req.version;
            const std::string referer = req.get_header_value("Referer");
            const std::string user_agent = req.get_header_value("User-Agent");

            std::ostringstream out;
            out << "[Access] "
                << remoteAddress(req)
                << " - - \""
                << escapeAccessField(req.method.empty() ? "-" : req.method) << ' '
                << escapeAccessField(requestTarget(req)) << ' '
                << escapeAccessField(version) << "\" "
                << res.status << ' '
                << formatBytes(responseBodyBytes(res, context)) << ' '
                << formatDuration(elapsedMs(context)) << " \""
                << escapeAccessField(referer) << "\" \""
                << escapeAccessField(user_agent) << "\"";
            return out.str();
        }

        size_t headerCount(const httplib::Headers &headers)
        {
            return headers.size();
        }

        std::string contentType(const httplib::Response &res)
        {
            return res.get_header_value("Content-Type");
        }

        std::string formatDebugLogLine(const httplib::Request &req,
                                       const httplib::Response &res,
                                       const RequestLogContext &context)
        {
            std::ostringstream out;
            out << "[HTTP] "
                << (req.method.empty() ? "-" : req.method) << ' '
                << requestTarget(req)
                << " from=" << remoteAddress(req)
                << " request_headers=" << headerCount(req.headers)
                << " request_body=" << req.body.size() << "B"
                << " status=" << res.status
                << " response_headers=" << headerCount(res.headers)
                << " response_body=" << formatBytes(responseBodyBytes(res, context))
                << " content_type=\"" << escapeAccessField(contentType(res)) << "\""
                << " duration=" << formatDuration(elapsedMs(context));
            return out.str();
        }

        void appendHeaders(std::ostringstream &out, const httplib::Headers &headers)
        {
            for (const auto &[name, value] : headers)
            {
                out << name << ": " << value << '\n';
            }
        }

        std::string responseBodyForTrace(const httplib::Response &res,
                                         const RequestLogContext &context)
        {
            if (context.streamed_response_body)
                return *context.streamed_response_body;
            return res.body;
        }

        std::string formatRequestTrace(const httplib::Request &req)
        {
            std::ostringstream out;
            out << (req.method.empty() ? "-" : req.method) << ' '
                << requestTarget(req) << ' '
                << (req.version.empty() ? "HTTP/?" : req.version) << '\n';
            appendHeaders(out, req.headers);
            out << '\n'
                << req.body;
            return out.str();
        }

        std::string formatResponseTrace(const httplib::Response &res,
                                        const RequestLogContext &context)
        {
            std::ostringstream out;
            out << "HTTP/1.1 " << res.status;
            if (const char *reason = httplib::status_message(res.status))
                out << ' ' << reason;
            out << '\n';
            appendHeaders(out, res.headers);
            out << '\n'
                << responseBodyForTrace(res, context);
            return out.str();
        }

        void logServedRequest(const httplib::Request &req,
                              const httplib::Response &res,
                              RequestLogState &log_state)
        {
            RequestLogContext context = log_state.finish(req);

            LOG_INFO(formatAccessLogLine(req, res, context));
            LOG_DEBUG(formatDebugLogLine(req, res, context));
            LOG_TRACE("[HTTP] request\n" << formatRequestTrace(req));
            LOG_TRACE("[HTTP] response\n" << formatResponseTrace(res, context));
        }

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
            LOG_DEBUG("Rank " << mpi_ctx->rank()
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
        RequestLogState request_log_state;

        // Inference is serialized on a single model instance. Keep HTTP handling
        // on one stable worker so OpenMP does not initialize per-request teams
        // on a large rotating httplib thread pool.
        svr.new_task_queue = []
        { return createSerializedInferenceTaskQueue().release(); };

        // Install signal handlers for graceful shutdown
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        svr.set_pre_routing_handler(
            [&request_log_state](const httplib::Request &req, httplib::Response &) {
                request_log_state.start(req);
                return httplib::Server::HandlerResponse::Unhandled;
            });

        svr.set_logger(
            [&request_log_state](const httplib::Request &req, const httplib::Response &res) {
                logServedRequest(req, res, request_log_state);
            });

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
                         auto streamed_response_body = traceAccessLoggingEnabled()
                                                           ? std::make_shared<std::string>()
                                                           : nullptr;
                         request_log_state.attachStreamedResponseBody(req, streamed_response_body);

                         // SSE streaming response
                         res.set_chunked_content_provider(
                             "text/event-stream",
                             [&handler,
                              request = std::move(*parsed_request),
                              streamed_response_body](size_t /*offset*/, httplib::DataSink &sink) -> bool
                             {
                                 auto chunk_cb = [&sink, streamed_response_body](const std::string &sse_line) -> bool
                                 {
                                     if (streamed_response_body)
                                         streamed_response_body->append(sse_line);
                                     return sink.write(sse_line.c_str(), sse_line.size());
                                 };

                                 auto response = handler.handleStreamingRequest(request, chunk_cb);

                                 if (!response.ok && !response.json_body.empty())
                                 {
                                     // Error before streaming started — emit error as SSE
                                     std::string error_sse = "data: " + response.json_body + "\n\ndata: [DONE]\n\n";
                                     if (streamed_response_body)
                                         streamed_response_body->append(error_sse);
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

        // Bind the socket before announcing readiness. cpp-httplib combines
        // bind(2) and listen(2) in bind_to_port(), then listen_after_bind()
        // enters the blocking accept loop.
        const std::string serve_endpoint = config.serve_host + ":" + std::to_string(config.serve_port);
        LOG_INFO("Llaminar server starting on " << serve_endpoint);

        if (!svr.bind_to_port(config.serve_host, config.serve_port))
        {
            if (!g_shutdown_requested.load())
            {
                LOG_ERROR("Failed to start server on " << serve_endpoint);
            }
            if (mpi_ctx->world_size() > 1)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            mpiShutdown();
            return 1;
        }

        LOG_INFO("Llaminar is ready and serving on " << serve_endpoint);

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

        if (!svr.listen_after_bind())
        {
            if (!g_shutdown_requested.load())
            {
                LOG_ERROR("Server stopped unexpectedly while serving on " << serve_endpoint);
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
