#include <gtest/gtest.h>

#include "app/modes/ServerMode.h"
#include "httplib.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    class ScopedTaskQueueShutdown
    {
    public:
        explicit ScopedTaskQueueShutdown(httplib::TaskQueue &queue) : queue_(queue) {}
        ~ScopedTaskQueueShutdown() { queue_.shutdown(); }

    private:
        httplib::TaskQueue &queue_;
    };
}

TEST(Test__ServerMode, SerializedInferenceTaskQueue_RunsJobsOnOneWorkerThread)
{
    auto queue = createSerializedInferenceTaskQueue();
    ASSERT_NE(queue, nullptr);
    ScopedTaskQueueShutdown shutdown(*queue);

    constexpr int task_count = 6;
    std::mutex mutex;
    std::condition_variable cv;
    std::vector<std::thread::id> worker_ids;
    worker_ids.reserve(task_count);

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
