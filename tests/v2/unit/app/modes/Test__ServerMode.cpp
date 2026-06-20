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
