/**
 * @file ServerMode.h
 * @brief HTTP server mode (--serve) with OpenAI-compatible REST API
 */

#pragma once

#include "app/modes/IExecutionMode.h"

#include <memory>
#include <string>

namespace httplib
{
    class Server;
    class TaskQueue;
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
     * @brief Create the single-worker HTTP task queue for one model instance.
     *
     * Inference uses mutable request/KV state, therefore the server serializes
     * HTTP work at this boundary rather than allowing a rotating web-worker
     * pool to enter the live model concurrently.
     *
     * @return Owned task queue whose one worker executes HTTP inference jobs.
     */
    std::unique_ptr<httplib::TaskQueue> createSerializedInferenceTaskQueue();

    class ServerMode : public IExecutionMode
    {
    public:
        const char *name() const override { return "server"; }
        bool matches(const OrchestrationConfig &config) const override;
        int execute(AppContext &ctx) override;
    };

} // namespace llaminar2
