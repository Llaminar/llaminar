/**
 * @file Test__HttpProcessStatsEndpoint.cpp
 * @brief Loopback proof that public GET/PUT stats traffic never executes OS collection.
 *
 * A blocked second collection leaves the first immutable snapshot available.
 * Concurrent real HTTP requests and resets exercise the production route while
 * mocks reject any need for inference, models or accelerators.
 */
#include "app/modes/HttpProcessStats.h"
#include "app/modes/ServerMode.h"
#include "app/modes/ChatCompletionHandler.h"
#include "app/modes/SerializedInferenceExecutor.h"
#include "mocks/MockOrchestrationRunner.h"
#include "mocks/MockTokenizer.h"
#include "httplib.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <atomic>
#include <semaphore>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace std::chrono_literals;
using json = nlohmann::json;

namespace
{
    /** @brief Stop the loopback listener before destroying handler-owned route captures. */
    struct HttpListener
    {
        httplib::Server &server;
        std::jthread worker;
        /** @brief Serve an already bound loopback socket using production HTTP placement. */
        explicit HttpListener(httplib::Server &owner)
            : server(owner), worker([&owner] { listenInferenceHttpServer(owner); }) {}
        /** @brief Terminate and join on every test exit. */
        ~HttpListener() { server.stop(); worker.join(); }
    };
    /** @brief Retire an intentionally blocked diagnostic read on every assertion exit. */
    struct ReleaseProbe
    {
        std::binary_semaphore &gate;
        /** @brief Publish completion before sampler destruction joins its worker. */
        ~ReleaseProbe() { gate.release(); }
    };
}

TEST(Test__HttpProcessStatsEndpoint, ConcurrentGETAndPUTNeverTriggerOrWaitForOSReads)
{
    std::atomic<unsigned> probes = 0;
    std::binary_semaphore blocked{0}, release{0};
    auto sampler = std::make_shared<HttpProcessStats>(HttpProcessStatsOptions{.interval = 2ms}, [&] {
        if (++probes == 2) { blocked.release(); release.acquire(); }
        return json{{"process", {{"rss_bytes", 4321}}}};
    });
    ReleaseProbe unblock{release};
    ASSERT_TRUE(blocked.try_acquire_for(2s));
    testing::NiceMock<MockOrchestrationRunner> runner;
    testing::NiceMock<MockTokenizer> tokenizer;
    runner.simulateInitialized();
    ChatCompletionHandler handler(runner, tokenizer, "fixture-model");
    SerializedInferenceExecutor executor;
    httplib::Server server;
    configureInferenceHttpServer(server, executor);
    registerRuntimeStatsEndpoint(server, handler, executor, "fixture-model",
        {{"topology", json::object()}, {"configuration", json::object()}}, {}, sampler);
    const int port = server.bind_to_any_port("127.0.0.1");
    ASSERT_GT(port, 0);
    HttpListener listener(server);
    std::atomic<unsigned> errors = 0;
    std::vector<std::jthread> clients;
    for (unsigned thread = 0; thread < 4; ++thread)
        clients.emplace_back([&] {
            httplib::Client client("127.0.0.1", port);
            client.set_read_timeout(2, 0);
            for (unsigned request = 0; request < 100; ++request)
            {
                if (request % 10 == 0)
                {
                    const auto reset = client.Put("/stats", "", "application/json");
                    if (!reset || reset->status != 200) ++errors;
                }
                const auto response = client.Get("/stats");
                if (!response || response->status != 200) { ++errors; continue; }
                try
                {
                    const auto value = json::parse(response->body).at("resources");
                    if (value.at("process").at("rss_bytes") != 4321 ||
                        value.at("collection").at("successful_samples") != 1) ++errors;
                }
                catch (...) { ++errors; }
            }
        });
    clients.clear();
    EXPECT_EQ(errors.load(), 0u);
    EXPECT_EQ(probes.load(), 2u);
}
