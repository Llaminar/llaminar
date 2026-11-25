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

#include "app/modes/ServingBatchDriver.h"
#include "app/modes/ServingResponseLeasePool.h"
#include "app/modes/ServerMode.h"
#include "app/modes/ServerRankMembership.h"
#include "app/modes/ServerExecutionEvidence.h"
#include "app/modes/ChatCompletionHandler.h"
#include "app/modes/SSEPublishChannel.h"
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
        configureInferenceHttpServer(server, executor.httpWorkerCount());
    }

    void configureInferenceHttpServer(httplib::Server &server, std::size_t workers)
    {
        server.new_task_queue = [workers] {
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

    InferenceQueueStatsSource serializedInferenceQueueStats(const SerializedInferenceExecutor &executor)
    {
        return std::make_shared<std::function<json()>>([&executor] {
            const auto queue = executor.snapshot();
            return json{{"mode", "serialized"}, {"capacity", queue.capacity}, {"admitted", queue.admitted},
                        {"queued", queue.queued}, {"active", queue.active}};
        });
    }

    void registerRuntimeStatsEndpoint(httplib::Server &server,
                                      ChatCompletionHandler &handler,
                                      const SerializedInferenceExecutor &executor,
                                      std::string model_name, json description,
                                      HttpResponseObserver response_observer,
                                      std::shared_ptr<HttpProcessStats> process_stats)
    {
        registerRuntimeStatsEndpoint(server, handler, serializedInferenceQueueStats(executor),
                                     std::move(model_name), std::move(description),
                                     std::move(response_observer), std::move(process_stats));
    }

    void registerRuntimeStatsEndpoint(httplib::Server &server,
                                      ChatCompletionHandler &handler,
                                      InferenceQueueStatsSource queue_stats,
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
        server.Get("/stats", [&handler, queue_stats, model_name = std::move(model_name),
                              description = std::move(description), process_stats = std::move(process_stats)](
                               const httplib::Request &, httplib::Response &response) {
            auto result = handler.runtimeStats();
            result["model"] = model_name;
            result["topology"] = description.at("topology");
            result["configuration"] = description.at("configuration");
            result["resources"] = process_stats->snapshot();
            result["inference_queue"] = (*queue_stats)();
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
                         // §10.1 INDEPENDENT WRITERS (v52, accepted channel):
                         // run() is ONLY the producer (enqueue, no socket); the
                         // httplib PROVIDER THREAD is the network writer (drains,
                         // owns every sink.write). The WRITER LIFECYCLE is a
                         // NONCOPYABLE owner/state allocated BEFORE the producer
                         // launch (no unowned thread on any throw), close+join
                         // exactly once, and the terminal verdict joins the
                         // producer outcome with the transport outcome: an
                         // exception mid-stream can never falsely Drain.
                         auto retained_admission = admission;
                         auto stream_request = std::move(*parsed_request);
                         // QUEUED-PAYLOAD WINDOW QUOTA (PROVISIONAL POLICY):
                         // max_entries = response-limit tokens x 2 (reasoning and
                         // answer can both publish per token) + a provisional
                         // framing-line allowance (role/finish/usage/DONE/tool-
                         // flush). This is an honest provisional quota policy:
                         // exact handler publication geometry is an OPEN
                         // derivation — NOT certified. Bytes are UNBOUNDED here
                         // (no guessed per-token byte constant; physical/admitted
                         // byte authority (PMA-backed output window) is a
                         // separate open admission piece, NOT claimed).
                         auto channel = std::make_shared<SSEPublishChannel>(
                             SSEPublishChannelBudget::forRequest(
                                 stream_request.max_tokens));
                         auto transport_open = std::make_shared<std::atomic<bool>>(true);
                         auto writer_finished = std::make_shared<std::atomic<bool>>(false);
                         auto writer_failed = std::make_shared<std::atomic<bool>>(false);
                         auto producer_error = std::make_shared<std::exception_ptr>(nullptr);
                         struct SSEWriterLifecycle
                         {
                             std::shared_ptr<SSEPublishChannel> channel;
                             std::shared_ptr<std::atomic<bool>> retired;
                             std::unique_ptr<std::thread> producer;
                             SSEWriterLifecycle(std::shared_ptr<SSEPublishChannel> ch)
                                 : channel(std::move(ch)),
                                   retired(std::make_shared<std::atomic<bool>>(false)) {}
                             ~SSEWriterLifecycle()
                             {
                                 if (!retired->exchange(true))
                                 {
                                     channel->close();
                                     if (producer && producer->joinable())
                                         producer->join();
                                 }
                             }
                             SSEWriterLifecycle(const SSEWriterLifecycle &) = delete;
                             SSEWriterLifecycle &operator=(const SSEWriterLifecycle &) = delete;
                         };
                         // OWNER/STATE FIRST (retired flag + channel ownership);
                         // the thread is installed into it only afterwards, so a
                         // failed launch never leaves an unowned joinable thread.
                         auto lifecycle = std::make_shared<SSEWriterLifecycle>(channel);
                         lifecycle->producer = std::make_unique<std::thread>(
                             [&handler, producer_error, retained_admission, channel, transport_open,
                              stream_request = std::move(stream_request), arrival] mutable {
                                 try
                                 {
                                     retained_admission->run([&] {
                                         auto enqueue = [channel, transport_open](const std::string &line) -> bool
                                         {
                                             if (!transport_open->load())
                                                 return false;
                                             return channel->push(line);
                                         };
                                         return handler.publishStreamingRequest(stream_request, enqueue, arrival);
                                     });
                                 }
                                 catch (...)
                                 {
                                     *producer_error = std::current_exception();
                                 }
                                 channel->close();
                             });
                         res.set_chunked_content_provider(
                             "text/event-stream",
                             [channel, transport_open, writer_finished, writer_failed,
                              streamed_response_body](size_t /*offset*/, httplib::DataSink &sink) -> bool
                             {
                                 for (;;)
                                 {
                                     auto line = channel->popOrClosed();
                                     if (!line)
                                         break;
                                     if (streamed_response_body)
                                         streamed_response_body->append(*line);
                                     if (!sink.write(line->c_str(), line->size()))
                                     {
                                         writer_failed->store(true);
                                         transport_open->store(false);
                                         channel->close();
                                         return false;
                                     }
                                 }
                                 sink.done();
                                 writer_finished->store(true);
                                 return true;
                             },
                             [retained_admission, lifecycle, transport_open,
                              writer_finished, writer_failed, producer_error](bool native_success)
                             {
                                 if (!lifecycle->retired->exchange(true))
                                 {
                                     transport_open->store(false);
                                     lifecycle->channel->close();
                                     if (lifecycle->producer && lifecycle->producer->joinable())
                                         lifecycle->producer->join();
                                 }
                                 // Join the PRODUCER outcome with the TRANSPORT
                                 // outcome: an exception mid-stream (truncated or
                                 // empty output with successful native framing)
                                 // must never be reported as Drained. Fatal
                                 // invalid-state/join semantics are NOT
                                 // suppressed here (no try/catch around close or
                                 // join); only the terminal verdict is computed
                                 // without throwing.
                                 const bool producer_ok =
                                     (producer_error == nullptr || *producer_error == nullptr);
                                 const bool writer_ok =
                                     writer_finished->load() && !writer_failed->load();
                                 retained_admission->publicationTerminal(
                                     native_success && writer_ok && producer_ok);
                             });
                     }
                     else
                     {
                         // Non-streaming response
                         auto response = admission->run([&] { return handler.handleRequest(*parsed_request, arrival); });
                         res.status = response.http_status;
                         // §10.1 TERMINAL OWNERSHIP (S2): the body is handed to the
                         // transport as a ONE-SHOT provider, so the reservation is
                         // RETAINED through write_response_core and the typed close runs
                         // at the provider boundary only when the body write was
                         // accepted. A rejected write leaves the reservation on the
                         // honest abandoned path (destructor release) — never a
                         // fabricated success close.
                         const auto body = std::make_shared<std::string>(response.json_body);
                         // The reservation is RETAINED through the transport write
                         // (captured by the provider). NON-CHUNKED VERDICT: the close
                         // runs inside the provider, gated on the accepted body write;
                         // a rejected write leaves the honest abandoned path.
                         res.set_content_provider(
                             "application/json",
                             [admission = std::move(admission), body](size_t, httplib::DataSink &sink) -> bool
                             {
                                 if (!sink.write(body->data(), body->size()))
                                     return false;
                                 sink.done();
                                 admission->publicationInFlight();
                                 admission->publicationDetached();
                                 admission->publicationFinished();
                                 return true;
                             },
                             [](bool) {});
                     }
                 });
    }

    /**
     * @brief Chat endpoint for continuous batching.
     *
     * Each request runs its chat handler on its own thread; generation goes
     * through the handler's batched lane into the runner-owned serving
     * session, so concurrent requests share execution. Admission takes a
     * PMA-backed response lease from @p leases (R responses, S + Q executing);
     * beyond that the server answers 503 without queueing.
     */
    void registerBatchedChatCompletionEndpoint(
        httplib::Server &server, ChatCompletionHandler &handler, ServingResponseLeasePool &leases,
        HttpStreamObserver stream_observer, std::function<std::string()> serving_failure)
    {
        server.Post("/v1/chat/completions",
                 [&handler, &leases, stream_observer = std::move(stream_observer),
                  serving_failure = std::move(serving_failure)](
                     const httplib::Request &req, httplib::Response &res)
                 {
                     if (const auto failure = serving_failure(); !failure.empty())
                     {
                         res.status = 503;
                         res.set_content(json{{"error", {{"message", failure}, {"type", "server_error"},
                             {"code", "serving_failed"}}}}.dump(), "application/json");
                         return;
                     }
                     const HttpRequestArrival arrival;
                     ChatCompletionResponse parse_error;
                     auto parsed_request = ChatCompletionHandler::parseRequest(req.body, parse_error);
                     if (!parsed_request)
                     {
                         res.status = parse_error.http_status;
                         res.set_content(parse_error.json_body, "application/json");
                         return;
                     }
                     // Response lease (R) + execution claim (S + Q); the last
                     // owner (handler call or streaming producer) releases it.
                     auto admitted = leases.tryAcquire();
                     if (!admitted)
                     {
                         res.status = 503;
                         res.set_content(json{{"error", {{"message", "Inference request capacity is full"},
                             {"type", "server_busy"}, {"code", "inference_queue_full"}}}}.dump(),
                             "application/json");
                         return;
                     }

                     if (!parsed_request->stream)
                     {
                         // Non-streaming clients are never written to while
                         // waiting: probe the socket so a disconnect cancels.
                         auto cancellation = std::make_shared<ServingRequestCancellation>();
                         cancellation->setConnectionProbe(req.is_connection_closed);
                         ServingBatchDriver::RequestScope scope(cancellation, admitted);
                         const auto response = handler.handleRequest(*parsed_request, arrival);
                         res.status = response.http_status;
                         res.set_content(response.json_body, "application/json");
                         return;
                     }

                     auto streamed_response_body = traceAccessLoggingEnabled()
                                                       ? std::make_shared<std::string>()
                                                       : nullptr;
                     if (stream_observer) stream_observer(req, streamed_response_body);
                     auto stream_request = std::move(*parsed_request);
                     // The queued SSE bytes of one response are bounded by the
                     // per-response budget the PMA serving pool committed: a
                     // slow reader blocks its own producer (and so its grants),
                     // never other requests and never unpriced memory.
                     auto budget = SSEPublishChannelBudget::forRequest(stream_request.max_tokens);
                     budget.max_buffered_bytes =
                         static_cast<std::size_t>(ServingPoolPerResponseBudget::defaultPolicy().bytes_per_response);
                     auto channel = std::make_shared<SSEPublishChannel>(budget);
                     auto transport_open = std::make_shared<std::atomic<bool>>(true);
                     // The producer runs the handler; the provider thread is the
                     // only socket writer. Closing the channel or the transport
                     // retires the producer, which cancels its serving request.
                     // Transport close cancels the serving request at once,
                     // even while it is queued or waiting for its consumer.
                     auto cancellation = std::make_shared<ServingRequestCancellation>();
                     auto producer = std::make_shared<std::thread>(
                         [&handler, channel, transport_open, admitted, arrival, cancellation,
                          stream_request = std::move(stream_request)]() mutable {
                             ServingBatchDriver::RequestScope scope(cancellation, admitted);
                             try
                             {
                                 auto enqueue = [channel, transport_open](const std::string &line) -> bool {
                                     return transport_open->load() && channel->push(line);
                                 };
                                 (void)handler.publishStreamingRequest(stream_request, enqueue, arrival);
                             }
                             catch (const std::exception &error)
                             {
                                 LOG_ERROR("[Server] batched streaming request failed: " << error.what());
                             }
                             channel->close();
                         });
                     res.set_chunked_content_provider(
                         "text/event-stream",
                         [channel, transport_open, cancellation, streamed_response_body](size_t, httplib::DataSink &sink) -> bool {
                             // Idle polls between SSE keep-alive comments (10 s).
                             constexpr int kHeartbeatPolls = 50;
                             int idle_polls = 0;
                             for (;;)
                             {
                                 std::string line;
                                 const auto status =
                                     channel->popOrClosedFor(std::chrono::milliseconds(200), line);
                                 if (status == SSEPublishChannel::PopStatus::Closed)
                                     break;
                                 // While the request waits (queued, prefilling or
                                 // paced) probe the client (writability plus a
                                 // socket liveness peek) so a disconnect frees
                                 // its slot without waiting for the next token,
                                 // and periodically send an SSE comment, which a
                                 // reset connection turns into a write error.
                                 bool alive = true;
                                 if (status == SSEPublishChannel::PopStatus::Line)
                                 {
                                     idle_polls = 0;
                                     alive = sink.write(line.c_str(), line.size());
                                 }
                                 else if (++idle_polls >= kHeartbeatPolls)
                                 {
                                     idle_polls = 0;
                                     static constexpr char kHeartbeat[] = ": keep-alive\n\n";
                                     alive = sink.write(kHeartbeat, sizeof(kHeartbeat) - 1);
                                 }
                                 else
                                 {
                                     alive = sink.is_writable();
                                 }
                                 if (!alive)
                                 {
                                     transport_open->store(false);
                                     cancellation->cancel();
                                     channel->close();
                                     return false;
                                 }
                                 if (status == SSEPublishChannel::PopStatus::Line && streamed_response_body)
                                     streamed_response_body->append(line);
                             }
                             sink.done();
                             return true;
                         },
                         [producer, channel, transport_open, cancellation](bool) {
                             transport_open->store(false);
                             cancellation->cancel();
                             channel->close();
                             if (producer->joinable())
                                 producer->join();
                         });
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
                for (const auto &record : serverExecutionParticipantRecords(
                         device, inventory->ranks.at(mpi_ctx->rank()), runner->executionPlan(),
                         runner->config(), mpi_ctx->world_size()))
                    PerfStatsCollector::addCounter(
                        "server", "execution_participant", 1.0, "startup", device.toString(), record);
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
        // Adopted §9.4: one resolved serving capacity backs the execution response
        // pool. Automatic intent preserves the existing accepted-response
        // semantics (active + queued <= kDefaultCapacity); authored intent is
        // exact and rejected when the installed envelope cannot serve it.
        std::optional<ResolvedServingCapacity> resolved_capacity;
        try
        {
            resolved_capacity.emplace(resolveInstalledServingCapacity(config.servingCapacityRequest()));
        }
        catch (const std::invalid_argument &error)
        {
            std::cerr << "Serving capacity admission failed: " << error.what() << "\n";
            runner->shutdownMPIWorkers();
            runner->shutdown();
            return 1;
        }
        // PMA-RESERVED SERVING POOL CAPACITY (REQUIRED): serving requires the
        // runner's published physical-memory authority. A missing authority is a
        // fatal startup diagnostic — production never runs an unbacked pool and
        // never falls back. The executor additionally requires the admitted
        // ServingResponsePool owner line (exact typed demand; addServingPoolDemand
        // in the memory plan) and reserves the capacity there (typed-layout
        // materialization belongs to the real serving-engine slice).
        const auto serving_memory_authority = runner->physicalMemoryAuthority();
        if (!serving_memory_authority)
        {
            LOG_ERROR("Server mode requires the runner's physical-memory authority; "
                      "refusing to start with an unbacked serving pool.");
            runner->shutdownMPIWorkers();
            runner->shutdown();
            flushPerfStatsFromEnv();
            return 1;
        }
        // One response-storage authority per serving mode, both committing the
        // admitted ServingResponsePool owner line: S = 1 serializes on the
        // executor's stable worker; S > 1 (continuous batching) admits against
        // PMA-backed response leases and per-slot decision storage.
        const int batch_capacity = resolved_capacity->residentSlots();
        if (batch_capacity > 1 &&
            (mpi_coordinated || runner->servingBatchCapacity() < batch_capacity))
        {
            // Authored S > 1 is exact: never silently serialize requests.
            LOG_ERROR("--serving-resident-slots " << batch_capacity
                      << " requests continuous batching, which is unavailable: "
                      << (mpi_coordinated ? std::string("multi-rank continuous batching is not implemented yet")
                          : runner->servingBatchCapacity() > 0
                              ? "the runner admitted " + std::to_string(runner->servingBatchCapacity()) + " request slots"
                              : runner->servingBatchUnsupportedReason()));
            if (mpi_coordinated)
                runner->shutdownMPIWorkers();
            runner->shutdown();
            flushPerfStatsFromEnv();
            return 1;
        }
        std::optional<SerializedInferenceExecutor> serialized_executor;
        std::unique_ptr<ServingResponseLeasePool> response_leases;
        if (batch_capacity > 1)
            response_leases = std::make_unique<ServingResponseLeasePool>(
                *resolved_capacity, serving_memory_authority, ServingPoolPerResponseBudget::defaultPolicy(),
                ServingSlotDecisionGeometry::forVocabulary(runner->vocabSize()));
        else
            serialized_executor.emplace(*resolved_capacity, serving_memory_authority,
                                        ServingPoolPerResponseBudget::defaultPolicy());
        httplib::Server svr;
        g_server_ptr = &svr;
        RequestLogState request_log_state;

        // The dedicated worker retains model/OpenMP ownership. HTTP waiters
        // cannot consume the capacity reserved for passive control requests.
        configureInferenceHttpServer(svr, static_cast<std::size_t>(servingHttpWorkerCount(*resolved_capacity)));

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
        // Fatal serving policy: a failed continuous-batching session keeps the
        // process up but reports 503 on /health and inference, so the
        // operator's supervisor restarts it. Installed below when batching.
        auto serving_failure = std::make_shared<std::function<std::string()>>([] { return std::string{}; });
        svr.Get("/health", [serving_failure](const httplib::Request &, httplib::Response &res)
                {
            const auto failure = (*serving_failure)();
            json response = failure.empty() ? json{{"status", "ok"}}
                                             : json{{"status", "failed"}, {"reason", failure}};
            if (!failure.empty())
                res.status = 503;
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
        // The serving mode chosen below replaces this source before listening.
        auto queue_stats = serialized_executor
            ? serializedInferenceQueueStats(*serialized_executor)
            : std::make_shared<std::function<json()>>([] { return json::object(); });
        registerRuntimeStatsEndpoint(svr, handler, queue_stats, model_name,
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

        // Continuous batching (§9.4): resolved resident slots S > 1 run every
        // chat request concurrently through the runner-owned serving session;
        // one driver thread owns it. S = 1 keeps the serialized executor.
        std::unique_ptr<ServingBatchDriver> batch_driver;
        auto stream_observer = [&request_log_state](const httplib::Request &request, std::shared_ptr<std::string> body) {
            request_log_state.attachStreamedResponseBody(request, std::move(body));
        };
        if (batch_capacity > 1)
        {
            // The decision store is materialized from the committed serving
            // pool (one FP32 vocabulary row per resident slot).
            batch_driver = std::make_unique<ServingBatchDriver>(
                *runner, batch_capacity, config.max_seq_len,
                response_leases->claimDecisionStorage(
                    ServingSlotDecisionGeometry::forVocabulary(runner->vocabSize()).bytesFor(batch_capacity)));
            // Install lanes (and snapshot request metadata) before the driver
            // thread owns the runner.
            handler.setGenerationLaneFactory(
                [driver = batch_driver.get()] { return driver->makeLane(); },
                [driver = batch_driver.get()](const std::function<void(IOrchestrationRunner &)> &access) {
                    driver->call(access);
                });
            std::string batch_error;
            if (!batch_driver->start(batch_error))
            {
                LOG_ERROR("Continuous-batching session failed to open: " << batch_error);
                runner->shutdown();
                flushPerfStatsFromEnv();
                return 1;
            }
            *serving_failure = [driver = batch_driver.get()] { return driver->failure(); };
            registerBatchedChatCompletionEndpoint(svr, handler, *response_leases, stream_observer,
                                                  [driver = batch_driver.get()] { return driver->failure(); });
            *queue_stats = [leases = response_leases.get(), driver = batch_driver.get()] {
                const auto lease_state = leases->snapshot();
                const auto occupancy = driver->occupancy();
                return json{{"mode", "continuous_batching"},
                            {"response_leases", lease_state.response_leases},
                            {"responses", lease_state.responses},
                            {"execution_capacity", lease_state.execution_capacity},
                            {"executing", lease_state.executing},
                            {"resident_slots", driver->capacity()},
                            {"active", occupancy.resident}, {"queued", occupancy.queued},
                            {"committed_bytes", lease_state.committed_bytes},
                            {"materialized_bytes", lease_state.materialized_bytes}};
            };
            LOG_INFO("Continuous batching: S=" << batch_capacity << " resident slots, R="
                     << resolved_capacity->responseLeases() << " response leases, Q="
                     << resolved_capacity->queueCapacity() << " (PMA serving pool "
                     << response_leases->snapshot().committed_bytes << " bytes)");
        }
        else
        {
            registerChatCompletionEndpoint(svr, handler, *serialized_executor, stream_observer);
        }

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
            if (batch_driver)
                batch_driver->stop(); // the driver thread must stop using the runner first
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
                if (batch_driver)
                    batch_driver->stop(); // the driver thread must stop using the runner first
                runner->shutdown();
                flushPerfStatsFromEnv();
                return 1;
            }
        }

        // All HTTP workers have drained their responses. Join inference before
        // retiring model/MPI owners; no maintenance or GPU cancellation occurs.
        if (batch_driver)
            batch_driver->stop();
        if (response_leases)
            response_leases->close();
        if (serialized_executor)
            serialized_executor->shutdown();
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
