#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <execinfo.h>
#include <initializer_list>
#include <mpi.h>
#include <string>
#include <thread>

namespace llaminar::test_util
{

    /// RAII watchdog that aborts the process with a stack trace if a test runs longer than the
    /// configured timeout. The timeout can be overridden by environment variables.
    class TestTimeoutGuard
    {
    public:
        TestTimeoutGuard(std::string scope,
                         std::chrono::milliseconds timeout,
                         std::chrono::milliseconds poll_interval = std::chrono::milliseconds(250))
            : scope_(std::move(scope)),
              timeout_(timeout),
              poll_interval_(poll_interval),
              done_(false)
        {
            if (timeout_.count() <= 0)
            {
                done_.store(true, std::memory_order_relaxed);
                return;
            }
            start_time_ = std::chrono::steady_clock::now();
            thread_ = std::thread([this]()
                                  { this->run(); });
        }

        TestTimeoutGuard(const TestTimeoutGuard &) = delete;
        TestTimeoutGuard &operator=(const TestTimeoutGuard &) = delete;

        ~TestTimeoutGuard() { disarm(); }

        /// Explicitly stop the watchdog (idempotent).
        void disarm()
        {
            bool expected = false;
            if (done_.compare_exchange_strong(expected, true, std::memory_order_release))
            {
                if (thread_.joinable())
                {
                    thread_.join();
                }
            }
            else
            {
                if (thread_.joinable())
                {
                    thread_.join();
                }
            }
        }

        /// Resolve timeout from a list of environment variables (first positive integer in ms wins).
        static std::chrono::milliseconds ResolveTimeout(
            std::initializer_list<const char *> env_vars,
            std::chrono::milliseconds fallback = std::chrono::milliseconds(60000))
        {
            for (const char *name : env_vars)
            {
                if (!name)
                {
                    continue;
                }
                const char *value = std::getenv(name);
                if (!value || *value == '\0')
                {
                    continue;
                }
                char *end = nullptr;
                long long parsed = std::strtoll(value, &end, 10);
                if (end == value)
                {
                    continue; // no digits
                }
                while (*end == ' ' || *end == '\t')
                {
                    ++end;
                }
                if (*end != '\0')
                {
                    continue; // mixed content, ignore
                }
                if (parsed >= 0)
                {
                    return std::chrono::milliseconds(parsed);
                }
            }
            return fallback;
        }

    private:
        void run()
        {
            while (!done_.load(std::memory_order_acquire))
            {
                std::this_thread::sleep_for(poll_interval_);
                if (done_.load(std::memory_order_acquire))
                {
                    break;
                }
                auto now = std::chrono::steady_clock::now();
                if (now - start_time_ > timeout_)
                {
                    emit_timeout_and_abort();
                    break;
                }
            }
        }

        void emit_timeout_and_abort() const
        {
            int initialized = 0;
            int finalized = 0;
            MPI_Initialized(&initialized);
            MPI_Finalized(&finalized);
            int rank = 0;
            int world = 1;
            if (initialized && !finalized)
            {
                MPI_Comm_rank(MPI_COMM_WORLD, &rank);
                MPI_Comm_size(MPI_COMM_WORLD, &world);
            }
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time_);
            std::fprintf(stderr,
                         "[WATCHDOG][%s][rank %d/%d] Timeout after %lld ms (limit %lld ms).\n",
                         scope_.c_str(), rank, world,
                         static_cast<long long>(elapsed.count()),
                         static_cast<long long>(timeout_.count()));
            dump_stack(rank);
            std::fflush(stderr);
            std::raise(SIGABRT);
        }

        void dump_stack(int rank) const
        {
            constexpr int kFrames = 128;
            void *frames[kFrames];
            int count = ::backtrace(frames, kFrames);
            if (count <= 0)
            {
                std::fprintf(stderr, "[WATCHDOG][%s][rank %d] <no stack trace available>\n",
                             scope_.c_str(), rank);
                return;
            }
            char **symbols = ::backtrace_symbols(frames, count);
            std::fprintf(stderr, "[WATCHDOG][%s][rank %d] === STACK TRACE (%d frames) ===\n",
                         scope_.c_str(), rank, count);
            if (!symbols)
            {
                for (int i = 0; i < count; ++i)
                {
                    std::fprintf(stderr, "[WATCHDOG][%s][rank %d] frame[%d] = %p\n",
                                 scope_.c_str(), rank, i, frames[i]);
                }
                return;
            }
            for (int i = 0; i < count; ++i)
            {
                std::fprintf(stderr, "[WATCHDOG][%s][rank %d] %s\n",
                             scope_.c_str(), rank, symbols[i]);
            }
            std::free(symbols);
        }

        std::string scope_;
        std::chrono::milliseconds timeout_;
        std::chrono::milliseconds poll_interval_;
        std::chrono::steady_clock::time_point start_time_;
        std::atomic<bool> done_;
        std::thread thread_;
    };

} // namespace llaminar::test_util
