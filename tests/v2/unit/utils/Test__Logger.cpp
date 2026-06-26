#include <gtest/gtest.h>

#include "utils/Logger.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace llaminar2;

namespace
{
    class ScopedLogLevel
    {
    public:
        explicit ScopedLogLevel(LogLevel level)
            : previous_(Logger::getInstance().getLogLevel())
        {
            Logger::getInstance().setLogLevel(level);
        }

        ~ScopedLogLevel()
        {
            Logger::getInstance().setLogLevel(previous_);
        }

    private:
        LogLevel previous_;
    };
}

TEST(Test__Logger, ConcurrentFileTeeWritesCompleteLines)
{
    ScopedLogLevel scoped_level(LogLevel::ERROR);

    auto path = std::filesystem::temp_directory_path() /
                ("llaminar_logger_concurrent_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".log");

    auto &logger = Logger::getInstance();
    logger.closeLogFile();
    ASSERT_TRUE(logger.setLogFile(path.string()));

    constexpr int kThreads = 8;
    constexpr int kLinesPerThread = 128;
    constexpr const char *kPayload = "payload=logger-concurrency-regression";

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int thread_idx = 0; thread_idx < kThreads; ++thread_idx)
    {
        threads.emplace_back([thread_idx] {
            ScopedDeviceLog device_prefix("logger-test-" + std::to_string(thread_idx));
            for (int line_idx = 0; line_idx < kLinesPerThread; ++line_idx)
            {
                LOG_ERROR("thread=" << thread_idx << " line=" << line_idx << " " << kPayload);
            }
        });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }
    logger.closeLogFile();

    std::ifstream in(path);
    ASSERT_TRUE(in.is_open());

    size_t line_count = 0;
    std::string line;
    while (std::getline(in, line))
    {
        ++line_count;
        const auto first_payload = line.find(kPayload);
        ASSERT_NE(first_payload, std::string::npos) << line;
        EXPECT_EQ(first_payload, line.rfind(kPayload))
            << "Concurrent log writes must not merge two records into one file line: " << line;
    }

    EXPECT_EQ(line_count, static_cast<size_t>(kThreads * kLinesPerThread));
    std::filesystem::remove(path);
}
