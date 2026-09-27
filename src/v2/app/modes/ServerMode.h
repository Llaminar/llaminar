/**
 * @file ServerMode.h
 * @brief HTTP serving, connection ownership, and OpenAI-compatible discovery.
 *
 * One stable HTTP worker owns the loaded model's serialized request lifetime.
 * The connection policy must release that worker when a response completes;
 * an idle client must not retain the model's only request executor.
 */

#pragma once

#include "app/modes/IExecutionMode.h"

#include <string>

namespace httplib
{
    class Server;
}

namespace llaminar2
{

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
     * @brief Configure HTTP execution and connection ownership for one model.
     *
     * Inference uses mutable request/KV state, therefore the server serializes
     * HTTP work on one stable worker rather than allowing a rotating web-worker
     * pool to initialize new OpenMP teams or enter the live model concurrently.
     * Each connection ends after one complete response, including an SSE stream,
     * so idle keep-alive clients cannot monopolize the worker. HTTP clients may
     * reconnect normally; inference concurrency and streaming are unchanged.
     * Configure this before listening; callers must not replace the queue or
     * connection policy independently.
     *
     * @param server Server whose request executor and connections are configured.
     */
    void configureSerializedInferenceHttpServer(httplib::Server &server);

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
