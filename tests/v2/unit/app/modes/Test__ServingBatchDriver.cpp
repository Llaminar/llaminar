/**
 * @file Test__ServingBatchDriver.cpp
 * @brief Device-free lifecycle tests of the continuous-batching HTTP driver.
 *
 * A scripted fake runner implements the runner serving-session ABI. The tests
 * prove the driver's exactly-once outcome contract: every submitted request's
 * mailbox reaches one terminal state on finish, cancellation, shutdown,
 * session failure or a throwing step, so no lane can wait forever; plus the
 * lane's grant pacing, forced-token fail-fast and transport cancellation.
 */

#include "app/modes/ServingBatchDriver.h"
#include "MockOrchestrationRunner.h"
#include "MockTokenizer.h"
#include "app/modes/ChatCompletionHandler.h"
#include "app/modes/ServerMode.h"
#include "httplib.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace llaminar2::test
{
    namespace
    {
        /**
         * @brief Fake serving session: admits up to `resident_capacity`
         *        requests and emits one token per granted request per step.
         */
        class FakeServingRunner : public ::testing::NiceMock<MockOrchestrationRunner>
        {
        public:
            std::atomic<int> resident_capacity{4};
            std::atomic<bool> reject_submissions{false};
            std::atomic<bool> throw_on_step{false};
            std::atomic<bool> fail_forward{false};
            std::atomic<bool> block_steps{false};
            std::atomic<int> steps{0};
            std::atomic<bool> ended{false};

            void releaseSteps()
            {
                {
                    std::lock_guard lock(gate_mutex_);
                    block_steps = false;
                }
                gate_.notify_all();
            }

            std::vector<std::uint64_t> cancelled() const
            {
                std::lock_guard lock(mutex_);
                return cancelled_;
            }

            bool beginServingBatch(int) override { return true; }
            void endServingBatch() override { ended = true; }
            int servingBatchCapacity() const override { return resident_capacity; }
            const std::string &lastError() const override { return error_; }
            std::string servingBatchFailure() const override
            {
                std::lock_guard lock(mutex_);
                return failure_;
            }

            bool submitServingRequest(ServingBatchRequest request) override
            {
                std::lock_guard lock(mutex_);
                if (reject_submissions)
                {
                    error_ = "fake rejection";
                    return false;
                }
                queue_.push_back(Row{std::move(request)});
                return true;
            }

            void cancelServingRequest(std::uint64_t id) override
            {
                std::lock_guard lock(mutex_);
                cancels_.push_back(id);
            }

            void grantServingTokens(std::uint64_t id, int tokens) override
            {
                std::lock_guard lock(mutex_);
                if (Row *row = find(id))
                    row->grant += tokens;
            }

            void forceServingToken(std::uint64_t id, int32_t token) override
            {
                std::lock_guard lock(mutex_);
                if (Row *row = find(id))
                    row->forced.push_back(token);
            }

            bool servingBatchHasRunnableWork() const override
            {
                std::lock_guard lock(mutex_);
                if (!cancels_.empty() || fail_forward)
                    return true;
                if (!queue_.empty() && static_cast<int>(resident_.size()) < resident_capacity)
                    return true;
                for (const auto &row : resident_)
                    if (row.grant > 0 || !row.forced.empty())
                        return true;
                return false;
            }

            std::vector<ServingBatchEvent> stepServingBatch() override
            {
                {
                    std::unique_lock lock(gate_mutex_);
                    gate_.wait(lock, [&] { return !block_steps; });
                }
                ++steps;
                if (throw_on_step)
                    throw std::runtime_error("fake step failure");
                std::lock_guard lock(mutex_);
                std::vector<ServingBatchEvent> events;
                auto finish = [&](const Row &row, ServingFinishReason reason, std::string error = {}) {
                    ServingBatchEvent event;
                    event.id = row.request.id;
                    event.finish = reason;
                    event.error = std::move(error);
                    events.push_back(std::move(event));
                };
                for (const auto id : cancels_)
                {
                    cancelled_.push_back(id);
                    for (auto *list : {&queue_, &resident_})
                        for (auto it = list->begin(); it != list->end(); ++it)
                            if (it->request.id == id)
                            {
                                finish(*it, ServingFinishReason::Cancelled);
                                list->erase(it);
                                break;
                            }
                }
                cancels_.clear();
                if (fail_forward)
                {
                    for (auto *list : {&queue_, &resident_})
                        for (const auto &row : *list)
                            finish(row, ServingFinishReason::Error, "fake forward failure");
                    queue_.clear();
                    resident_.clear();
                    fail_forward = false;
                    failure_ = "fake forward failure"; // the runner session seals
                    return events;
                }
                while (!queue_.empty() && static_cast<int>(resident_.size()) < resident_capacity)
                {
                    resident_.push_back(std::move(queue_.front()));
                    queue_.erase(queue_.begin());
                }
                for (auto it = resident_.begin(); it != resident_.end();)
                {
                    ServingBatchEvent event;
                    event.id = it->request.id;
                    if (!it->forced.empty())
                    {
                        event.tokens.push_back(it->forced.front());
                        it->forced.erase(it->forced.begin());
                        ++it->produced;
                    }
                    else if (it->grant > 0)
                    {
                        event.tokens.push_back(static_cast<int32_t>(it->request.id * 1000 + it->produced));
                        --it->grant;
                        ++it->produced;
                    }
                    if (it->produced >= it->request.max_new_tokens)
                        event.finish = ServingFinishReason::Length;
                    if (!event.tokens.empty() || event.finished())
                        events.push_back(std::move(event));
                    if (it->produced >= it->request.max_new_tokens)
                        it = resident_.erase(it);
                    else
                        ++it;
                }
                return events;
            }

        private:
            struct Row
            {
                ServingBatchRequest request;
                int grant = 0;
                int produced = 0;
                std::vector<int32_t> forced;
            };

            Row *find(std::uint64_t id)
            {
                for (auto *list : {&queue_, &resident_})
                    for (auto &row : *list)
                        if (row.request.id == id)
                            return &row;
                return nullptr;
            }

            mutable std::mutex mutex_;
            std::vector<Row> queue_;
            std::vector<Row> resident_;
            std::vector<std::uint64_t> cancels_;
            std::vector<std::uint64_t> cancelled_;
            std::string error_;
            std::string failure_;
            std::mutex gate_mutex_;
            std::condition_variable gate_;
        };

        constexpr int kContext = 64; // lanes request context - prompt tokens

        std::vector<int32_t> prompt(int length) { return std::vector<int32_t>(static_cast<std::size_t>(length), 7); }

        /** @brief Run @p fn on another thread; fail the test instead of hanging. */
        template <typename F>
        auto bounded(F fn)
        {
            auto future = std::async(std::launch::async, std::move(fn));
            EXPECT_EQ(future.wait_for(std::chrono::seconds(10)), std::future_status::ready)
                << "operation did not complete (lane would hang)";
            return future.get();
        }
    } // namespace

    TEST(ServingBatchDriverLifecycle, LaneStreamsGrantedTokensAndCompletesAtLength)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto lane = driver.makeLane();
        ASSERT_TRUE(lane->prefill(prompt(60))); // max_new = 64 - 60 = 4
        std::vector<int32_t> tokens;
        bool complete = false;
        for (int i = 0; i < 8 && !complete; ++i)
        {
            const auto step = bounded([&] { return lane->decodeStep(2); });
            ASSERT_TRUE(step.error.empty()) << step.error;
            ASSERT_LE(step.tokens.size(), 2u);
            tokens.insert(tokens.end(), step.tokens.begin(), step.tokens.end());
            complete = step.is_complete;
        }
        EXPECT_TRUE(complete);
        EXPECT_EQ(tokens.size(), 4u);
        driver.stop();
        EXPECT_TRUE(runner.ended);
    }

    TEST(ServingBatchDriverLifecycle, StopFailsAPendingSubmission)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        // Hold the driver thread inside a step so the submission stays queued.
        runner.block_steps = true;
        auto busy = driver.makeLane();
        ASSERT_TRUE(busy->prefill(prompt(4)));
        driver.grant(1, 1); // wake the loop into the blocked step
        while (runner.steps.load() == 0 && !runner.block_steps.load())
            std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        auto lane = driver.makeLane();
        auto prefilled = std::async(std::launch::async, [&] { return lane->prefill(prompt(4)); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        auto stopped = std::async(std::launch::async, [&] { driver.stop(); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        runner.releaseSteps();
        ASSERT_EQ(prefilled.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        EXPECT_FALSE(prefilled.get());
        EXPECT_NE(lane->lastError().find("shutting down"), std::string::npos) << lane->lastError();
        stopped.get();
        // The resident lane is failed too, not left waiting.
        const auto step = bounded([&] { return busy->decodeStep(1); });
        EXPECT_FALSE(step.error.empty());
    }

    TEST(ServingBatchDriverLifecycle, SubmitAfterStopFailsImmediately)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        driver.stop();
        auto lane = driver.makeLane();
        EXPECT_FALSE(bounded([&] { return lane->prefill(prompt(4)); }));
        EXPECT_FALSE(lane->lastError().empty());
    }

    TEST(ServingBatchDriverLifecycle, RejectedSubmissionReportsTheRunnerError)
    {
        FakeServingRunner runner;
        runner.reject_submissions = true;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto lane = driver.makeLane();
        EXPECT_FALSE(bounded([&] { return lane->prefill(prompt(4)); }));
        EXPECT_EQ(lane->lastError(), "fake rejection");
    }

    TEST(ServingBatchDriverLifecycle, TransportCancellationWhileQueuedResolvesTheLane)
    {
        FakeServingRunner runner;
        runner.resident_capacity = 0; // every request stays queued
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto cancellation = std::make_shared<ServingRequestCancellation>();
        std::unique_ptr<IRequestGenerationLane> lane;
        {
            ServingBatchDriver::RequestScope scope(cancellation);
            lane = driver.makeLane();
        }
        ASSERT_TRUE(lane->prefill(prompt(4)));
        auto pending = std::async(std::launch::async, [&] { return lane->decodeStep(4); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        cancellation->cancel(); // the client disconnected while queued
        ASSERT_EQ(pending.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        const auto step = pending.get();
        EXPECT_TRUE(step.tokens.empty());
        EXPECT_NE(step.error.find("cancelled"), std::string::npos) << step.error;
        EXPECT_EQ(runner.cancelled(), std::vector<std::uint64_t>{1});
    }

    TEST(ServingBatchDriverLifecycle, CancellationBeforeSubmissionNeverSubmits)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto cancellation = std::make_shared<ServingRequestCancellation>();
        cancellation->cancel();
        ServingBatchDriver::RequestScope scope(cancellation);
        auto lane = driver.makeLane();
        EXPECT_FALSE(lane->prefill(prompt(4)));
        EXPECT_NE(lane->lastError().find("cancelled"), std::string::npos);
    }

    TEST(ServingBatchDriverLifecycle, DestroyingAnUnfinishedLaneCancelsItsRequest)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        {
            auto lane = driver.makeLane();
            ASSERT_TRUE(lane->prefill(prompt(4)));
            const auto step = bounded([&] { return lane->decodeStep(1); });
            ASSERT_EQ(step.tokens.size(), 1u);
        }
        for (int i = 0; i < 200 && runner.cancelled().empty(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        EXPECT_EQ(runner.cancelled(), std::vector<std::uint64_t>{1});
    }

    TEST(ServingBatchDriverLifecycle, ThrowingStepSealsTheSessionAndFailsEveryLane)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto a = driver.makeLane();
        auto b = driver.makeLane();
        ASSERT_TRUE(a->prefill(prompt(4)));
        ASSERT_TRUE(b->prefill(prompt(4)));
        runner.throw_on_step = true;
        const auto step = bounded([&] { return a->decodeStep(1); });
        EXPECT_NE(step.error.find("fake step failure"), std::string::npos) << step.error;
        const auto other = bounded([&] { return b->decodeStep(1); });
        EXPECT_FALSE(other.error.empty());
        EXPECT_TRUE(driver.closed());
        auto late = driver.makeLane();
        EXPECT_FALSE(bounded([&] { return late->prefill(prompt(4)); }));
    }

    TEST(ServingBatchDriverLifecycle, ForwardFailureErrorsResidentAndQueuedLanes)
    {
        FakeServingRunner runner;
        runner.resident_capacity = 1;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto resident = driver.makeLane();
        auto queued = driver.makeLane();
        ASSERT_TRUE(resident->prefill(prompt(4)));
        ASSERT_TRUE(queued->prefill(prompt(4)));
        ASSERT_EQ(bounded([&] { return resident->decodeStep(1); }).tokens.size(), 1u);
        auto waiting = std::async(std::launch::async, [&] { return queued->decodeStep(1); });
        runner.fail_forward = true;
        driver.grant(1, 1); // wake the driver
        const auto failed = bounded([&] { return resident->decodeStep(1); });
        EXPECT_EQ(failed.error, "fake forward failure");
        ASSERT_EQ(waiting.wait_for(std::chrono::seconds(10)), std::future_status::ready);
        EXPECT_EQ(waiting.get().error, "fake forward failure");
        // The runner seal closes the driver: serving reports the failure and
        // admits nothing more.
        for (int i = 0; i < 200 && !driver.closed(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        EXPECT_TRUE(driver.closed());
        EXPECT_NE(driver.failure().find("fake forward failure"), std::string::npos) << driver.failure();
        auto late = driver.makeLane();
        EXPECT_FALSE(bounded([&] { return late->prefill(prompt(4)); }));
    }

    TEST(ServingBatchDriverLifecycle, ForcedTokenFailsFastWithAnOutstandingGrant)
    {
        FakeServingRunner runner;
        runner.resident_capacity = 0; // grants never execute
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto lane = driver.makeLane();
        ASSERT_TRUE(lane->prefill(prompt(4)));
        auto waiting = std::async(std::launch::async, [&] { return lane->decodeStep(3); });
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const auto forced = lane->forceDecodeToken(42);
        EXPECT_NE(forced.error.find("outstanding"), std::string::npos) << forced.error;
        driver.stop();
        EXPECT_EQ(waiting.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    }

    TEST(ServingBatchDriverLifecycle, ForcedTokenIsTheNextOutputWithoutAGrant)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto lane = driver.makeLane();
        ASSERT_TRUE(lane->prefill(prompt(4)));
        const auto forced = bounded([&] { return lane->forceDecodeToken(42); });
        ASSERT_TRUE(forced.error.empty()) << forced.error;
        EXPECT_EQ(forced.tokens, std::vector<int32_t>{42});
    }

    TEST(ServingBatchDriverLifecycle, RunnerAccessRunsOnTheDriverThreadAndFailsAfterStop)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        std::thread::id seen;
        bounded([&] {
            driver.call([&](IOrchestrationRunner &) { seen = std::this_thread::get_id(); });
            return 0;
        });
        EXPECT_NE(seen, std::thread::id{});
        EXPECT_NE(seen, std::this_thread::get_id());
        EXPECT_THROW(driver.call([](IOrchestrationRunner &) { throw std::logic_error("read failed"); }),
                     std::logic_error);
        EXPECT_FALSE(driver.closed()); // a failed read does not seal serving
        driver.stop();
        EXPECT_THROW(driver.call([](IOrchestrationRunner &) {}), std::runtime_error);
    }

    TEST(ServingBatchDriverLifecycle, MaintenanceRunsPerStepOnTheDriverAndFailureSeals)
    {
        FakeServingRunner runner;
        std::atomic<std::uint64_t> maintained{0};
        std::atomic<bool> fail{false};
        ON_CALL(runner, maybeApplyMoERebalance(::testing::_))
            .WillByDefault(::testing::Invoke([&](std::uint64_t tokens) {
                maintained += tokens;
                return !fail.load();
            }));
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        auto lane = driver.makeLane();
        ASSERT_TRUE(lane->prefill(prompt(4)));
        const auto first = bounded([&] { return lane->decodeStep(2); });
        ASSERT_GE(first.tokens.size(), 1u);
        EXPECT_GE(maintained.load(), first.tokens.size()) << "committed tokens reach maintenance from the driver";
        fail = true;
        const auto failed = bounded([&] { return lane->decodeStep(2); });
        for (int i = 0; i < 200 && !driver.closed(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        EXPECT_TRUE(driver.closed());
        EXPECT_NE(driver.failure().find("maintenance"), std::string::npos) << driver.failure();
        (void)failed;
    }

    TEST(ServingResponseLeasePool, ExecutionClaimsEndAtRowFinishResponsesAtPublication)
    {
        // S = 2, R = 4, Q = 0: two requests may execute; four responses may
        // be outstanding while finished rows still publish.
        ServingResponseLeasePool pool(resolveInstalledServingCapacity(ServingCapacityRequest::authored(2, 4, 0)));
        auto a = pool.tryAcquire();
        auto b = pool.tryAcquire();
        ASSERT_TRUE(a && b);
        EXPECT_FALSE(pool.tryAcquire()) << "S + Q = 2 executing";
        a->executionFinished();
        a->executionFinished(); // idempotent
        auto c = pool.tryAcquire();
        ASSERT_TRUE(c) << "a finished executing; its response still publishes";
        b->executionFinished();
        c->executionFinished();
        auto d = pool.tryAcquire();
        ASSERT_TRUE(d);
        EXPECT_FALSE(pool.tryAcquire()) << "R = 4 responses outstanding";
        a.reset();
        EXPECT_TRUE(pool.tryAcquire()) << "publication completed frees a response lease";
        const auto snapshot = pool.snapshot();
        EXPECT_EQ(snapshot.response_leases, 4u);
        EXPECT_EQ(snapshot.execution_capacity, 2u);
        pool.close();
        EXPECT_FALSE(pool.tryAcquire());
    }

    TEST(ServingResponseLeasePool, LaneReleasesItsExecutionClaimWhenTheRowFinishes)
    {
        FakeServingRunner runner;
        ServingBatchDriver driver(runner, 4, kContext);
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        ServingResponseLeasePool pool(resolveInstalledServingCapacity(ServingCapacityRequest::authored(1, 2, 0)));
        auto lease = pool.tryAcquire();
        ASSERT_TRUE(lease);
        std::unique_ptr<IRequestGenerationLane> lane;
        {
            ServingBatchDriver::RequestScope scope(std::make_shared<ServingRequestCancellation>(), lease);
            lane = driver.makeLane();
        }
        ASSERT_TRUE(lane->prefill(prompt(62))); // max_new = 2
        EXPECT_FALSE(pool.tryAcquire()) << "executing";
        bool complete = false;
        for (int i = 0; i < 4 && !complete; ++i)
            complete = bounded([&] { return lane->decodeStep(1); }).is_complete;
        ASSERT_TRUE(complete);
        auto next = pool.tryAcquire();
        EXPECT_TRUE(next) << "row finished: the execution claim is free while the response publishes";
    }

    TEST(BatchedServingEndpoint, DisconnectWhileQueuedReleasesAdmissionWithinABound)
    {
        // Real HTTP serving over the scripted runner: the request can never
        // enter a slot (resident capacity 0), so only the transport probe can
        // retire it. A client that disconnects must free its lease quickly.
        FakeServingRunner runner;
        runner.resident_capacity = 0;
        ::testing::NiceMock<MockTokenizer> tokenizer;
        ON_CALL(tokenizer, encodeChat(::testing::_, ::testing::_, ::testing::_, ::testing::_))
            .WillByDefault(::testing::Return(std::vector<int>(8, 7)));
        ON_CALL(tokenizer, decode_token(::testing::_)).WillByDefault(::testing::Return("x"));
        ChatCompletionHandler handler(runner, tokenizer, "fixture-model");
        ServingBatchDriver driver(runner, 2, kContext);
        handler.setGenerationLaneFactory(
            [&driver] { return driver.makeLane(); },
            [&driver](const std::function<void(IOrchestrationRunner &)> &access) { driver.call(access); });
        std::string error;
        ASSERT_TRUE(driver.start(error)) << error;
        ServingResponseLeasePool leases(resolveInstalledServingCapacity(ServingCapacityRequest::authored(2, 4, 2)));
        httplib::Server server;
        configureInferenceHttpServer(server, std::size_t{6});
        registerBatchedChatCompletionEndpoint(server, handler, leases, {}, [] { return std::string{}; });
        const int port = server.bind_to_any_port("127.0.0.1");
        ASSERT_GT(port, 0);
        std::thread listener([&] { listenInferenceHttpServer(server); });

        const std::string body = R"({"messages":[{"role":"user","content":"hello"}],"max_tokens":8})";
        const std::string request = "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                                    "Content-Type: application/json\r\nContent-Length: " +
                                    std::to_string(body.size()) + "\r\n\r\n" + body;
        const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        ASSERT_GE(fd, 0);
        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<uint16_t>(port));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        ASSERT_EQ(::connect(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)), 0);
        ASSERT_EQ(::send(fd, request.data(), request.size(), 0), static_cast<ssize_t>(request.size()));

        auto eventually = [](auto predicate) {
            for (int i = 0; i < 300; ++i) // 3 s bound
            {
                if (predicate())
                    return true;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            return predicate();
        };
        ASSERT_TRUE(eventually([&] { return leases.snapshot().executing == 1; })) << "request admitted and queued";
        ::close(fd); // the client goes away while its request is queued
        EXPECT_TRUE(eventually([&] { return runner.cancelled() == std::vector<std::uint64_t>{1}; }))
            << "transport probe cancels the queued serving request";
        EXPECT_TRUE(eventually([&] {
            const auto state = leases.snapshot();
            return state.executing == 0 && state.responses == 0;
        })) << "execution claim and response lease are released";

        server.stop();
        listener.join();
        driver.stop();
    }
} // namespace llaminar2::test
