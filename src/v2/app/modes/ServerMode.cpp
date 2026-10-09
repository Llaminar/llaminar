/**
 * @file ServerMode.cpp
 * @brief HTTP server mode with OpenAI-compatible REST API
 *
 * Request termination belongs to this mode; process finalization belongs to
 * the caller's MPIProcessSession. Returning first retires mode-local adapters
 * and handlers before the runner, contexts and outer session are destroyed.
 *
 * Endpoints:
 *   GET  /stats                   — Passive current-period cache/token/MTP/timing counters
 *   PUT  /stats                   — Reset observations without interrupting inference
 *   GET  /health                  — Liveness check
 *   GET  /v1/models               — OpenAI-compatible loaded-model discovery
 *   POST /v1/chat/completions     — OpenAI-compatible chat completion (streaming + non-streaming)
 *
 * The resolved request authority, not MPI rank zero, owns HTTP serving.
 * Every rank publishes its immutable membership in PerfStats so an external
 * certificate can require complete participant evidence without steering
 * inference or adding a diagnostic collective to the execution lifecycle.
 * One dedicated inference worker retains CPU/OpenMP locality. Bounded request
 * admission leaves independent HTTP capacity for passive stats and discovery
 * while generations are active or queued. Connections end after each complete
 * response so idle clients cannot retain HTTP workers. The listener and its
 * children use the rank's admitted CPU partition; they must not inherit the
 * OpenMP initial thread's single-place mask and compete with model service.
 */

#include "app/modes/ServerMode.h"
#include "app/modes/ServerRankMembership.h"
#include "app/modes/ServerExecutionEvidence.h"
#include "app/modes/ChatCompletionHandler.h"
#include "app/modes/SerializedInferenceExecutor.h"
#include "app/modes/HttpServiceThreadAffinity.h"
#include "app/modes/HttpProcessStats.h"
#include "app/modes/MoEMovementTransportJson.h"
#include "app/AppContext.h"
#include "loaders/ModelContext.h"
#include "utils/Assertions.h"
#include "utils/DebugEnv.h"
#include "utils/Logger.h"
#include "utils/PerfStatsCollector.h"

// cpp-httplib (header-only)
#include "httplib.h"

// nlohmann/json (header-only)
#include "nlohmann/json.hpp"

#include <iostream>
#include <unistd.h>
#include <mutex>
#include <atomic>
#include <chrono>
#include <iomanip>
#include <memory>
#include <optional>
#include <sstream>
#include <thread>
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
    std::string openAIModelListResponse(const std::string &model_name)
    {
        /*
         * This endpoint names the already-admitted model only.  It must not
         * scan directories, load a second GGUF, or touch runner state: HTTP
         * model discovery is intentionally independent of the live graph.
         */
        return json{{"object", "list"},
                    {"data", json::array({{
                         {"id", model_name},
                         {"object", "model"},
                         {"created", 0},
                         {"owned_by", "llaminar"},
                     }})}}
            .dump();
    }

    void registerOpenAIModelDiscoveryEndpoint(httplib::Server &server,
                                              std::string model_name)
    {
        // The route owns its copy: config/runner setup may leave scope before
        // the HTTP event loop stops, while discovery must remain immutable.
        server.Get("/v1/models", [model_name = std::move(model_name)](
                                     const httplib::Request &,
                                     httplib::Response &res)
                   {
                       res.set_content(openAIModelListResponse(model_name),
                                       "application/json");
                   });
    }

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

        void flushPerfStatsFromEnv()
        {
            if (!PerfStatsCollector::flushFromEnv())
            {
                LOG_WARN("[ServerMode] Failed to flush PerfStats artifact(s)");
            }
        }

        /**
         * @brief Export completed physical history after this rank's clean drain.
         * @param runner Retired runner retaining its canonical terminal diagnostics.
         *
         * Live stats remain bounded counters. The optional final sidecar preserves
         * exact transaction membership for generation and parity observers; export
         * failure propagates to the serving lifecycle instead of certifying a
         * successful run with missing transport evidence.
         */
        void exportTerminalMovement(const IOrchestrationRunner &runner)
        {
            const auto path = PerfStatsCollector::jsonExportPath();
            if (path.empty() || (!PerfStatsCollector::isDomainEnabled("moe_overlay_controller") &&
                !PerfStatsCollector::isDomainEnabled("moe_overlay_residency")))
                return;
            writeMoEMovementTransportJson(path, std::max(0, Logger::getInstance().getRank()),
                runner.moeOptimizationMovementLedger());
        }

        bool shutdownEndpointEnabled()
        {
            return DebugEnv::isTruthyEnvValue(
                DebugEnv::envValue("LLAMINAR_ENABLE_SERVER_SHUTDOWN_ENDPOINT"));
        }

        /**
         * @brief Terminate the request channel while retaining outer MPI ownership.
         * @param ctx Active caller-owned application resources.
         * @param detail Failure shared with followers before runner retirement.
         * @return Nonzero command exit; the caller's scope finalizes MPI later.
         */
        int shutdownAfterUnhandledException(AppContext &ctx, const std::string &detail)
        {
            const bool has_mpi = ctx.mpi_ctx != nullptr;
            const bool is_authority =
                ctx.runner &&
                ctx.coordinatedRequestRole() ==
                    CoordinatedRequestRole::Authority;
            const bool notify_workers =
                has_mpi && ctx.mpi_ctx->world_size() > 1 && is_authority;

            if (is_authority)
                LOG_ERROR("Server mode failed with unhandled exception: " << detail);

            if (ctx.runner)
            {
                if (notify_workers)
                    ctx.runner->abortMPIWorkers(detail);
                ctx.runner->shutdown();
            }
            flushPerfStatsFromEnv();
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

    void configureInferenceHttpServer(httplib::Server &server,
                                      const SerializedInferenceExecutor &executor)
    {
        server.new_task_queue = [workers = executor.httpWorkerCount()] {
            return std::make_unique<httplib::ThreadPool>(workers).release();
        };
        // A completed connection releases HTTP ownership. Inference thread
        // locality belongs solely to the executor, never a rotating web worker.
        server.set_keep_alive_max_count(1);
    }

    bool listenInferenceHttpServer(httplib::Server &server)
    {
        // The inference executor already owns its stable thread. Only the
        // listener and HTTP children created by listen_after_bind inherit
        // this rank-local service mask. Restore it before runner shutdown.
        const HttpServiceThreadAffinity service_affinity;
        return server.listen_after_bind();
    }

    void registerRuntimeStatsEndpoint(httplib::Server &server,
                                      ChatCompletionHandler &handler,
                                      const SerializedInferenceExecutor &executor,
                                      std::string model_name, json description,
                                      HttpResponseObserver response_observer,
                                      std::shared_ptr<HttpProcessStats> process_stats)
    {
        // HTTP closures retain one periodic sampler. Neither GET nor PUT has
        // an OS refresh entrypoint, including when its cached data is stale.
        if (!process_stats) process_stats = std::make_shared<HttpProcessStats>();
        server.set_logger([&handler, observer = std::move(response_observer)](
                              const httplib::Request &request, const httplib::Response &response) {
            handler.recordHttpResponse(response.status);
            if (observer) observer(request, response);
        });
        server.Put("/stats", [&handler](const httplib::Request &, httplib::Response &response) {
            const auto epoch = handler.resetRuntimeStats();
            response.set_header("Cache-Control", "no-store");
            response.set_content(json{{"object", "llaminar.stats.reset"}, {"epoch", epoch}}.dump(),
                                 "application/json");
        });
        server.Options("/stats", [](const httplib::Request &, httplib::Response &response) {
            response.status = 204;
        });
        server.Get("/stats", [&handler, &executor, model_name = std::move(model_name),
                              description = std::move(description), process_stats = std::move(process_stats)](
                               const httplib::Request &, httplib::Response &response) {
            auto result = handler.runtimeStats();
            const auto queue = executor.snapshot();
            result["model"] = model_name;
            result["topology"] = description.at("topology");
            result["configuration"] = description.at("configuration");
            result["resources"] = process_stats->snapshot();
            result["inference_queue"] = {{"capacity", queue.capacity}, {"admitted", queue.admitted},
                                         {"queued", queue.queued}, {"active", queue.active}};
            response.set_header("Cache-Control", "no-store");
            response.set_content(result.dump(), "application/json");
        });
    }

    nlohmann::json serverRuntimeDescription(const RankExecutionPlan &plan,
                                            const OrchestrationConfig &config, int world_size)
    {
        const auto participants = serverExecutionParticipants(plan, config, world_size);
        const auto policy = serverExecutionPolicyTags(plan, config);
        json devices = json::array();
        for (const auto &[device, role] : participants)
            devices.push_back({{"id", device.toString()},
                {"role", role == ServerParticipantRole::ModelGraph ? "model_graph" : "expert_only"}});
        json domains = json::array();
        for (const auto &domain : plan.my_domains)
        {
            json members = json::array();
            for (const auto &address : domain.devices) members.push_back(address.toString());
            domains.push_back({{"id", domain.domain_id}, {"name", domain.domain_name},
                {"devices", std::move(members)}, {"backend", collectiveBackendTypeToString(domain.backend)}});
        }
        const auto &runtime = plan.runtime;
        return {
            {"topology", {{"source", "resolved_execution_plan"}, {"scope", "serving_rank"},
                {"rank", plan.rank}, {"world_size", world_size},
                {"strategy", policy.at("execution_strategy")}, {"devices", std::move(devices)},
                {"pipeline_domains", serverPipelineDomainTags(plan)}, {"tp_domains", std::move(domains)},
                {"global_tp_domain_size", plan.global_tp_domain_size},
                {"expert_overlay", policy.at("expert_overlay") == "true"}}},
            {"configuration", {{"context_tokens", config.max_seq_len},
                {"mtp", {{"enabled", runtime.mtp.enabled},
                    {"verify_mode", mtpVerifyModeToString(runtime.mtp.verify_mode)},
                    {"depth_policy", mtpDepthPolicyModeToString(runtime.mtp.depth_policy.mode)},
                    {"min_depth", runtime.mtp.depth_policy.min_depth},
                    {"max_depth", resolveMTPMaximumExecutionDraftDepth(runtime.mtp)}}},
                {"prefix_cache", {{"enabled", runtime.prefix_cache.enabled},
                    {"ram_budget_bytes_per_participant", runtime.prefix_cache.ram_budget_bytes},
                    {"disk_budget_bytes", runtime.prefix_cache.disk_budget_bytes}}}}}};
    }

    void registerChatCompletionEndpoint(httplib::Server &server, ChatCompletionHandler &handler,
                                        SerializedInferenceExecutor &inference_executor,
                                        HttpStreamObserver stream_observer)
    {
        server.Post("/v1/chat/completions",
                 [&handler, &inference_executor, stream_observer = std::move(stream_observer)](const httplib::Request &req, httplib::Response &res)
                 {
                     const HttpRequestArrival arrival;
                     // Parse before spending an inference reservation.
                     ChatCompletionResponse parse_error;
                     auto parsed_request = ChatCompletionHandler::parseRequest(req.body, parse_error);

                     if (!parsed_request)
                     {
                         res.status = parse_error.http_status;
                         res.set_content(parse_error.json_body, "application/json");
                         return;
                     }

                     auto admission = inference_executor.tryReserve();
                     if (!admission)
                     {
                         res.status = 503;
                         res.set_content(json{{"error", {{"message", "Inference request capacity is full"},
                             {"type", "server_busy"}, {"code", "inference_queue_full"}}}}.dump(),
                             "application/json");
                         return;
                     }

                     if (parsed_request->stream)
                     {
                         auto streamed_response_body = traceAccessLoggingEnabled()
                                                           ? std::make_shared<std::string>()
                                                           : nullptr;
                         if (stream_observer) stream_observer(req, streamed_response_body);

                         // SSE streaming response
                         res.set_chunked_content_provider(
                             "text/event-stream",
                             [&handler,
                              admission = std::move(admission),
                              request = std::move(*parsed_request),
                              streamed_response_body, arrival](size_t /*offset*/, httplib::DataSink &sink) -> bool
                             {
                                 auto chunk_cb = [&sink, streamed_response_body](const std::string &sse_line) -> bool
                                 {
                                     if (streamed_response_body)
                                         streamed_response_body->append(sse_line);
                                     return sink.write(sse_line.c_str(), sse_line.size());
                                 };

                                 admission->run([&] { return handler.publishStreamingRequest(request, chunk_cb, arrival); });

                                 sink.done();
                                 return true;
                             });
                     }
                     else
                     {
                         // Non-streaming response
                         auto response = admission->run([&] { return handler.handleRequest(*parsed_request, arrival); });
                         res.status = response.http_status;
                         res.set_content(response.json_body, "application/json");
                     }
                 });
    }

    int ServerMode::execute(AppContext &ctx)
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
        // Copy immutable topology only for observation. This runs once before
        // serving/participating; graph and request decisions never consult it.
        if (PerfStatsCollector::isDomainEnabled("server"))
        {
            // Read the already-published context snapshot. A second hostname
            // probe or MPI exchange here would introduce another topology
            // authority and a new startup failure/ordering boundary.
            const auto inventory = mpi_ctx->clusterInventory();
            LLAMINAR_ASSERT_NOT_NULL(inventory, "server cluster inventory");
            PerfStatsCollector::addCounter(
                "server", "rank_membership", 1.0, "startup", {},
                serverRankMembershipTags(*inventory, mpi_ctx->rank(),
                                         runner->coordinatedRootRank()));
            PerfStatsCollector::addCounter(
                "server", "process_identity", 1.0, "startup", {},
                serverProcessIdentityTags(static_cast<int>(::getpid())));
            const auto participants = serverExecutionParticipants(
                runner->executionPlan(), runner->config(), mpi_ctx->world_size());
            // Reuse loaded metadata; observing attention ownership must not
            // reopen the model or infer a layer family from missing timings.
            const auto *model = dynamic_cast<const ModelContext *>(runner->modelContextForDiagnostics());
            LLAMINAR_ASSERT_NOT_NULL(model, "server loaded model metadata");
            PerfStatsCollector::addCounter(
                "server", "execution_topology", 1.0, "startup", {},
                serverExecutionTopologyTags(participants, runner->executionPlan(),
                    ModelMemoryProfile::fromGGUF(model->model())));
            for (const auto &domain : serverPipelineDomainTags(runner->executionPlan()))
                PerfStatsCollector::addCounter(
                    "server", "execution_pipeline_domain", 1.0, "startup", {}, domain);
            for (const auto &[device, role] : participants)
            {
                (void)role;
                PerfStatsCollector::addCounter(
                    "server", "execution_participant", 1.0, "startup", device.toString(),
                    serverExecutionParticipantTags(device, inventory->ranks.at(mpi_ctx->rank()),
                                                   runner->executionPlan().numa_node));
            }
            // Only the service authority publishes request policy. Expert-only
            // followers do not own an independent MTP/prefix configuration.
            if (is_authority)
                PerfStatsCollector::addCounter(
                    "server", "execution_policy", 1.0, "startup", {},
                    serverExecutionPolicyTags(runner->executionPlan(), runner->config()));
        }
        if (mpi_coordinated && !is_authority)
        {
            // Followers enter the MPI command loop and participate in the
            // exact graph/collective sequence admitted by the authority.
            LOG_DEBUG("Rank " << mpi_ctx->rank()
                              << " entering MPI worker loop for inference participation");
            runner->setMPICoordinatedMode(true);
            runner->runMPIWorkerLoop();
            runner->shutdown();
            exportTerminalMovement(*runner);
            flushPerfStatsFromEnv();
            return 0;
        }

        // The authority must open the command channel before any early-exit
        // path so followers blocked in their receive loop can always observe a
        // terminal command.
        if (mpi_coordinated)
            runner->setMPICoordinatedMode(true);

        if (!tokenizer->hasChatTemplate())
        {
            LOG_ERROR("Server mode requires a model with a chat template.");
            if (mpi_coordinated)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            flushPerfStatsFromEnv();
            return 1;
        }

        /* Do not bind or advertise the HTTP endpoint until the same generic
         * production-readiness lifecycle used by benchmark mode is complete. */
        if (!runner->prepareForInference())
        {
            LOG_ERROR(
                "Inference runtime did not become ready for serving: "
                << runner->lastError());
            if (mpi_coordinated)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            flushPerfStatsFromEnv();
            return 1;
        }

        // Extract model name from path for response metadata
        std::string model_name = std::filesystem::path(config.model_path).stem().string();

        ChatCompletionHandler handler(*runner, *tokenizer, model_name);
        SerializedInferenceExecutor inference_executor;
        httplib::Server svr;
        g_server_ptr = &svr;
        RequestLogState request_log_state;

        // The dedicated worker retains model/OpenMP ownership. HTTP waiters
        // cannot consume the capacity reserved for passive control requests.
        configureInferenceHttpServer(svr, inference_executor);

        // Install signal handlers for graceful shutdown
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
#ifdef SIGPIPE
        // HTTP clients can close a connection while the server is still
        // writing a response. Treat that as an ordinary request failure path
        // instead of letting the process die from SIGPIPE.
        std::signal(SIGPIPE, SIG_IGN);
#endif

        svr.set_pre_routing_handler(
            [&request_log_state](const httplib::Request &req, httplib::Response &) {
                request_log_state.start(req);
                return httplib::Server::HandlerResponse::Unhandled;
            });

        // CORS headers for Open WebUI and other browser-based clients
        svr.set_default_headers({{"Access-Control-Allow-Origin", "*"},
                                 {"Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS"},
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

        // ─── GET /v1/models ──────────────────────────────────────────
        // OpenAI clients commonly discover a server's usable model before
        // presenting it in their UI.  The response is immutable for this
        // server lifetime because its model was admitted before bind(2).
        registerOpenAIModelDiscoveryEndpoint(svr, model_name);
        HttpProcessStatsOptions process_options;
        process_options.filesystems.push_back(std::filesystem::absolute(config.model_path).parent_path());
        if (runner->executionPlan().runtime.prefix_cache.enabled &&
            runner->executionPlan().runtime.prefix_cache.disk_budget_bytes != 0)
            process_options.filesystems.push_back(runner->executionPlan().runtime.prefix_cache.disk_dir);
        registerRuntimeStatsEndpoint(svr, handler, inference_executor, model_name,
            serverRuntimeDescription(runner->executionPlan(), runner->config(), mpi_ctx->world_size()),
            [&request_log_state](const httplib::Request &req, const httplib::Response &res) {
                logServedRequest(req, res, request_log_state);
            }, std::make_shared<HttpProcessStats>(std::move(process_options)));

        if (shutdownEndpointEnabled())
        {
            svr.Post("/admin/shutdown",
                     [](const httplib::Request &, httplib::Response &res)
                     {
                         json response = {{"status", "shutting_down"}};
                         res.status = 202;
                         res.set_content(response.dump(), "application/json");

                         std::thread([] {
                             std::this_thread::sleep_for(std::chrono::milliseconds(500));
                             g_shutdown_requested.store(true);
                             if (g_server_ptr)
                                 g_server_ptr->stop();
                         }).detach();
                     });
        }

        // ─── POST /v1/chat/completions ───────────────────────────────

        registerChatCompletionEndpoint(svr, handler, inference_executor,
            [&request_log_state](const httplib::Request &request, std::shared_ptr<std::string> body) {
                request_log_state.attachStreamedResponseBody(request, std::move(body));
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
            if (mpi_coordinated)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            flushPerfStatsFromEnv();
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

        if (!listenInferenceHttpServer(svr))
        {
            if (!g_shutdown_requested.load())
            {
                LOG_ERROR("Server stopped unexpectedly while serving on " << serve_endpoint);
                if (mpi_coordinated)
                    runner->shutdownMPIWorkers();
                runner->shutdown();
                flushPerfStatsFromEnv();
                return 1;
            }
        }

        // All HTTP workers have drained their responses. Join inference before
        // retiring model/MPI owners; no maintenance or GPU cancellation occurs.
        inference_executor.shutdown();
        LOG_INFO("Server shut down.");
        g_server_ptr = nullptr;

        // Release every follower from the coordinated command loop.
        if (mpi_coordinated)
            runner->shutdownMPIWorkers();

        runner->shutdown();
        exportTerminalMovement(*runner);
        flushPerfStatsFromEnv();
        return 0;
    }
    catch (const std::exception &e)
    {
        return shutdownAfterUnhandledException(ctx, e.what());
    }
    catch (...)
    {
        return shutdownAfterUnhandledException(ctx, "unknown exception");
    }

} // namespace llaminar2
