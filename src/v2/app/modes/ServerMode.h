/**
 * @file ServerMode.h
 * @brief HTTP serving, connection ownership, and OpenAI-compatible discovery.
 *
 * One stable inference worker owns the loaded model's serialized lifetime.
 * Bounded generation admission leaves independent HTTP capacity for passive
 * stats and discovery during long requests. Connections retire after a complete
 * response so idle clients do not retain HTTP workers.
 */

#pragma once

#include "app/modes/IExecutionMode.h"

#include <string>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>

namespace httplib
{
    class Server;
    struct Request;
    struct Response;
}

namespace llaminar2
{
    class HttpProcessStats;
    class ChatCompletionHandler;
    class SerializedInferenceExecutor;
    struct RankExecutionPlan;
    struct OrchestrationConfig;

    /** @brief Optional access-log observer; the stats owner always counts responses first. */
    using HttpResponseObserver = std::function<void(const httplib::Request &, const httplib::Response &)>;
    /** @brief Optional full-response trace attachment, only enabled by explicit trace logging. */
    using HttpStreamObserver = std::function<void(const httplib::Request &, std::shared_ptr<std::string>)>;

    /**
     * @brief Freeze admitted topology and serving policy once, before binding HTTP.
     * @param plan Runner-owned resolved rank execution plan.
     * @param config Runner-owned resolved orchestration configuration.
     * @param world_size Exact execution communicator size.
     * @return Immutable topology/policy metadata, explicitly scoped for multi-rank plans.
     */
    nlohmann::json serverRuntimeDescription(const RankExecutionPlan &plan,
                                            const OrchestrationConfig &config, int world_size);

    /**
     * @brief Install the production chat route, including bounded admission and SSE publication.
     * @param server HTTP owner whose lifetime is shorter than the handler and executor.
     * @param handler Serialized request implementation.
     * @param executor Sole inference worker and response reservation authority.
     * @param stream_observer Optional trace-only body attachment, never a token/state probe.
     */
    void registerChatCompletionEndpoint(httplib::Server &server, ChatCompletionHandler &handler,
                                        SerializedInferenceExecutor &executor,
                                        HttpStreamObserver stream_observer = {});

    /**
     * @brief Serialize the loaded model as an OpenAI-compatible model-list response.
     *
     * The server owns exactly one loaded model instance, so discovery is a
     * stable control-plane description and never consults the inference
     * runner.  Keeping this formatter separately testable prevents clients
     * such as OpenWebUI from needing a special health-only integration.
     *
     * @param model_name Stable identifier advertised by the loaded server.
     * @return JSON body for `GET /v1/models`.
     */
    std::string openAIModelListResponse(const std::string &model_name);

    /**
     * @brief Register immutable OpenAI-compatible model discovery on a server.
     *
     * The supplied model name is copied into the route closure, so the route
     * remains valid for the complete HTTP server lifetime without consulting
     * the live inference runner.
     *
     * @param server HTTP server that owns the route.
     * @param model_name Stable identifier for the already-admitted model.
     */
    void registerOpenAIModelDiscoveryEndpoint(httplib::Server &server,
                                              std::string model_name);

    /**
     * @brief Register passive server-lifetime statistics on the serving endpoint.
     * @param server HTTP owner of GET /stats.
     * @param handler Handler owning immutable completed request observations.
     * @param executor Host inference queue whose snapshot never visits a runner.
     * @param model_name Immutable identity of the admitted model.
     * @param description Frozen result of serverRuntimeDescription(), copied into the route.
     * @param response_observer Optional access logger sharing the completed-response callback.
     * @param process_stats Shared periodic OS sampler; omitted selects one default sampler.
     */
    void registerRuntimeStatsEndpoint(httplib::Server &server,
                                      ChatCompletionHandler &handler,
                                      const SerializedInferenceExecutor &executor,
                                      std::string model_name,
                                      nlohmann::json description,
                                      HttpResponseObserver response_observer = {},
                                      std::shared_ptr<HttpProcessStats> process_stats = {});

    /**
     * @brief Couple HTTP capacity to bounded, serialized inference admission.
     *
     * The HTTP pool has two more workers than admitted inference responses.
     * Queued generations therefore cannot occupy every stats/discovery worker.
     * Inference itself always runs on the executor's one stable thread, including
     * SSE content providers. Connections end only after their complete response.
     *
     * @param server Server configured before listening.
     * @param executor Sole inference ownership and admission authority.
     */
    void configureInferenceHttpServer(httplib::Server &server,
                                      const SerializedInferenceExecutor &executor);

    /**
     * @brief Run the bound listener and HTTP pool within the rank's admitted CPU partition.
     * @param server Server with an already-bound listening socket and registered routes.
     * @return The native listener's result after all HTTP workers have retired.
     *
     * The existing inference worker keeps its compute placement. On return or
     * exception the calling thread regains its exact pre-listen affinity.
     */
    bool listenInferenceHttpServer(httplib::Server &server);

    /** @brief Own the public HTTP lifetime of one admitted inference runner. */
    class ServerMode : public IExecutionMode
    {
    public:
        /** @brief Return the stable execution-mode name. @return "server". */
        const char *name() const override { return "server"; }
        /**
         * @brief Report whether the admitted configuration selects serving.
         * @param config Validated orchestration intent.
         * @return Whether the server execution mode was requested.
         */
        bool matches(const OrchestrationConfig &config) const override;
        /**
         * @brief Serve requests until shutdown, retiring the runner before MPI.
         * @param ctx Caller-owned application and admitted runner resources.
         * @return Zero for clean shutdown or nonzero for fatal serving failure.
         */
        int execute(AppContext &ctx) override;
    };

} // namespace llaminar2
