/**
 * @file Test__ServerMode.cpp
 * @brief Model-free proofs of HTTP discovery and serialized connection ownership.
 *
 * Real loopback clients exercise the production server configuration without
 * loading a model. An idle keep-alive client must not capture the sole worker,
 * and SSE must retain its connection only until the complete stream is sent.
 */
#include <gtest/gtest.h>

#include "app/modes/ServerMode.h"
#include "httplib.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Join a test queue before its captured fixture storage is retired. */
    class ScopedTaskQueueShutdown
    {
    public:
        /** @brief Borrow the queue whose worker must be joined on scope exit. */
        explicit ScopedTaskQueueShutdown(httplib::TaskQueue &queue) : queue_(queue) {}
        /** @brief Join queued work, including on a failed test assertion. */
        ~ScopedTaskQueueShutdown() { queue_.shutdown(); }

    private:
        httplib::TaskQueue &queue_;
    };

    /** @brief Stop and join a pre-bound HTTP listener on every test exit. */
    class ScopedHttpListener
    {
    public:
        /** @brief Start listening on the server's already-bound loopback socket. */
        explicit ScopedHttpListener(httplib::Server &server)
            : server_(server), thread_([&server] { server.listen_after_bind(); }) {}

        /** @brief Stop accepting work and join before route captures are destroyed. */
        ~ScopedHttpListener()
        {
            server_.stop();
            thread_.join();
        }

    private:
        httplib::Server &server_;
        std::thread thread_;
    };

    /**
     * @brief Prove a completed response releases the only worker to another client.
     * @param streaming Exercise a chunked SSE response instead of a fixed body.
     *
     * The first client deliberately requests keep-alive and remains alive while
     * a second client calls the server. The read bound is a deadlock detector,
     * not an inference performance threshold: neither route performs model work.
     */
    void expectCompletedConnectionReleasesWorker(bool streaming)
    {
        httplib::Server server;
        configureSerializedInferenceHttpServer(server);
        server.Get("/response", [streaming](const httplib::Request &, httplib::Response &response) {
            if (streaming)
            {
                response.set_chunked_content_provider(
                    "text/event-stream", [](size_t, httplib::DataSink &sink) {
                        const std::string body = "data: ready\n\ndata: [DONE]\n\n";
                        if (!sink.write(body.data(), body.size()))
                            return false;
                        sink.done();
                        return true;
                    });
            }
            else
                response.set_content("ready", "text/plain");
        });
        server.Get("/probe", [](const httplib::Request &, httplib::Response &response) {
            response.set_content("available", "text/plain");
        });
        const int port = server.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);
        ScopedHttpListener listener(server);

        httplib::Client idle_client("127.0.0.1", port);
        idle_client.set_keep_alive(true);
        idle_client.set_read_timeout(std::chrono::seconds(1));
        const auto first = idle_client.Get("/response");
        ASSERT_TRUE(first);
        EXPECT_EQ(first->status, 200);
        EXPECT_EQ(first->get_header_value("Connection"), "close");
        EXPECT_EQ(first->body, streaming ? "data: ready\n\ndata: [DONE]\n\n" : "ready");

        httplib::Client next_client("127.0.0.1", port);
        next_client.set_read_timeout(std::chrono::seconds(1));
        const auto second = next_client.Get("/probe");
        // Release the intentionally idle connection even on the red path, so
        // listener teardown never waits for the library's five-second timeout.
        idle_client.stop();
        ASSERT_TRUE(second) << "A completed connection retained the sole HTTP worker";
        EXPECT_EQ(second->status, 200);
        EXPECT_EQ(second->body, "available");
    }
}

TEST(Test__ServerMode, SerializedInferenceTaskQueue_RunsJobsOnOneWorkerThread)
{
    httplib::Server server;
    configureSerializedInferenceHttpServer(server);
    std::unique_ptr<httplib::TaskQueue> queue(server.new_task_queue());
    ASSERT_NE(queue, nullptr);

    constexpr int task_count = 6;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::thread::id> worker_ids;
    worker_ids.reserve(task_count);
    // Join before captured synchronization and observation storage is retired,
    // including on the timeout/assertion path.
    ScopedTaskQueueShutdown shutdown(*queue);

    for (int i = 0; i < task_count; ++i)
    {
        ASSERT_TRUE(queue->enqueue([&]
                                   {
                                       {
                                           std::lock_guard<std::mutex> lock(mutex);
                                           worker_ids.push_back(std::this_thread::get_id());
                                       }
                                       cv.notify_one();
                                   }));
    }

    bool completed = false;
    {
        std::unique_lock<std::mutex> lock(mutex);
        completed = cv.wait_for(lock, std::chrono::seconds(5), [&]
                                { return worker_ids.size() == task_count; });
    }

    ASSERT_TRUE(completed);
    ASSERT_FALSE(worker_ids.empty());
    const auto expected_worker = worker_ids.front();
    for (const auto &worker_id : worker_ids)
        EXPECT_EQ(worker_id, expected_worker);
}

/** @brief An idle non-streaming client cannot retain the serialized worker. */
TEST(Test__ServerMode, SerializedHTTPConnection_ReleasesWorkerAfterOrdinaryResponse)
{
    expectCompletedConnectionReleasesWorker(false);
}

/** @brief Complete SSE data is sent before releasing the worker and connection. */
TEST(Test__ServerMode, SerializedHTTPConnection_ReleasesWorkerAfterStreamingResponse)
{
    expectCompletedConnectionReleasesWorker(true);
}

/**
 * @brief Model discovery advertises the one admitted model using the OpenAI schema.
 *
 * OpenWebUI calls this endpoint before presenting a selectable connection.
 * The payload must therefore use the standard list/model shape without
 * reaching into a live runner or allocating model state.
 */
TEST(Test__ServerMode, OpenAIModelListResponse_AdvertisesLoadedModel)
{
    EXPECT_EQ(
        openAIModelListResponse("Qwen3.5-122B-A10B-UD-Q8_K_XL"),
        R"({"data":[{"created":0,"id":"Qwen3.5-122B-A10B-UD-Q8_K_XL","object":"model","owned_by":"llaminar"}],"object":"list"})");
}

/**
 * @brief Registered model discovery returns the OpenAI payload over HTTP.
 *
 * This guards the actual route registration that OpenWebUI exercises.  A
 * formatter-only test would not catch the regression where the server omitted
 * the GET handler and responded with 404.
 */
TEST(Test__ServerMode, OpenAIModelDiscoveryRoute_ReturnsLoadedModel)
{
    httplib::Server server;
    registerOpenAIModelDiscoveryEndpoint(server, "Qwen3.5-122B-A10B-UD-Q8_K_XL");

    const int port = server.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    std::thread serving_thread([&server] { server.listen_after_bind(); });

    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds(2));
    const auto response = client.Get("/v1/models");

    server.stop();
    serving_thread.join();

    ASSERT_TRUE(response);
    EXPECT_EQ(response->status, 200);
    EXPECT_EQ(response->get_header_value("Content-Type"), "application/json");
    EXPECT_EQ(
        response->body,
        R"({"data":[{"created":0,"id":"Qwen3.5-122B-A10B-UD-Q8_K_XL","object":"model","owned_by":"llaminar"}],"object":"list"})");
}
