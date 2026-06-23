#pragma once

/**
 * @file Logger.h
 * @brief Thread-safe logging infrastructure for Llaminar V2
 * @author David Sanftenberg
 */

#include "LogLevel.h"
#include "DebugEnv.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstdio>
#include <ctime>
#include <deque>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <vector>

namespace llaminar2
{

    /**
     * @brief Singleton logger with buffered output and file/line tracking
     *
     * Features:
     * - Thread-safe with mutex protection
     * - Configurable log level filtering
     * - Timestamped output with millisecond precision
     * - File/line location tracking
     * - Recent log buffer for debugging
     * - Environment variable configuration (LLAMINAR_LOG_LEVEL)
     */
    class Logger
    {
    public:
        static Logger &getInstance()
        {
            static Logger *instance = new Logger();
            return *instance;
        }

        void setLogLevel(LogLevel level)
        {
            current_level_ = level;
        }

        LogLevel getLogLevel() const
        {
            return current_level_;
        }

        /**
         * @brief Set the MPI rank for log messages
         * @param rank The MPI rank number (0-based)
         */
        void setRank(int rank)
        {
            rank_ = rank;
            has_rank_ = true;
        }

        /**
         * @brief Get the current MPI rank
         * @return The MPI rank, or -1 if not set
         */
        int getRank() const
        {
            return has_rank_ ? rank_ : -1;
        }

        /**
         * @brief Set a thread-local device prefix for log messages
         *
         * This prefix is automatically included in all log messages from the
         * calling thread. Use ScopedDeviceLog for RAII-style management.
         *
         * @param prefix The device prefix (e.g., "rocm:0", "cuda:1")
         */
        static void setThreadDevicePrefix(const std::string &prefix)
        {
            thread_device_prefix() = prefix;
        }

        /**
         * @brief Clear the thread-local device prefix
         */
        static void clearThreadDevicePrefix()
        {
            thread_device_prefix().clear();
        }

        /**
         * @brief Get the current thread-local device prefix
         * @return The device prefix, or empty string if not set
         */
        static const std::string &getThreadDevicePrefix()
        {
            return thread_device_prefix();
        }

        /**
         * @brief Set a log file path to tee all log output to a file
         *
         * When set, every log message is written to both stderr and the file.
         * Before opening, any existing log file at @p path is rotated through
         * a sliding window of up to 10 previous runs (logrotate-style):
         *
         *   test_log.txt     ->  test_log.1.txt
         *   test_log.1.txt   ->  test_log.2.txt
         *   ...
         *   test_log.9.txt   ->  test_log.10.txt
         *   test_log.10.txt  (deleted)
         *
         * This preserves the logs of the prior 10 runs for post-mortem
         * diagnosis of flaky failures — the current run's log would otherwise
         * be lost on truncation-on-open. The new file is then opened in
         * truncate mode. Call closeLogFile() to flush and close.
         *
         * @param path Filesystem path for the log file
         * @return true if the file was opened successfully
         */
        bool setLogFile(const std::string &path)
        {
            std::lock_guard<std::mutex> lk(buffer_mutex_);
            if (log_file_.is_open())
                log_file_.close();

            // Rotate prior logs so flaky failures keep their evidence across
            // up to 10 retries.
            try
            {
                constexpr int kMaxRotations = 10;
                std::filesystem::path p(path);
                std::string stem = p.stem().string();     // "test_log"
                std::string ext = p.extension().string(); // ".txt"

                auto rotated_path = [&](int n) {
                    std::filesystem::path rp = p;
                    rp.replace_filename(stem + "." + std::to_string(n) + ext);
                    return rp;
                };

                std::error_code ec;
                // Drop the oldest slot if present.
                std::filesystem::remove(rotated_path(kMaxRotations), ec);
                // Shift N-1 -> N, N-2 -> N-1, ..., 1 -> 2.
                for (int i = kMaxRotations - 1; i >= 1; --i)
                {
                    auto src = rotated_path(i);
                    auto dst = rotated_path(i + 1);
                    if (std::filesystem::exists(src, ec))
                    {
                        std::filesystem::rename(src, dst, ec);
                    }
                }
                // Move current log to slot .1
                if (std::filesystem::exists(p, ec))
                {
                    std::filesystem::rename(p, rotated_path(1), ec);
                }
            }
            catch (...)
            {
                // Never let log rotation break logging itself.
            }

            log_file_.open(path, std::ios::out | std::ios::trunc);
            return log_file_.is_open();
        }

        /**
         * @brief Close the log file if open
         */
        void closeLogFile()
        {
            std::lock_guard<std::mutex> lk(buffer_mutex_);
            if (log_file_.is_open())
            {
                log_file_.flush();
                log_file_.close();
            }
        }

        bool shouldLog(LogLevel level) const
        {
            return static_cast<int>(level) <= static_cast<int>(current_level_);
        }

        void log(LogLevel level, const std::string &message, const std::string &file = "", int line = 0)
        {
            if (!shouldLog(level))
            {
                return;
            }

            std::string timestamp = getCurrentTimestamp();
            std::string level_str = logLevelToString(level);
            std::string location = "";

            if (!file.empty() && line > 0)
            {
                // Extract just the filename from the full path
                size_t pos = file.find_last_of("/\\");
                std::string filename = (pos != std::string::npos) ? file.substr(pos + 1) : file;
                location = " [" + filename + ":" + std::to_string(line) + "]";
            }

            std::string full_line = "[" + timestamp + "] [" + level_str + "]";

            // Add rank if set
            if (has_rank_)
            {
                full_line += " [" + std::to_string(rank_) + "]";
            }

            // Add thread-local device prefix if set
            const std::string &device_prefix = thread_device_prefix();
            if (!device_prefix.empty())
            {
                full_line += " [" + device_prefix + "]";
            }

            full_line += location + " " + message;

            {
                std::lock_guard<std::mutex> lk(buffer_mutex_);
                recent_.push_back(full_line);
                if (recent_.size() > max_buffer_)
                    recent_.pop_front();
            }

            std::cerr << full_line << std::endl;

            // Tee to log file if open
            if (log_file_.is_open())
            {
                log_file_ << full_line << '\n';
            }
        }

        std::string logLevelToString(LogLevel level) const
        {
            switch (level)
            {
            case LogLevel::ERROR:
                return "ERROR";
            case LogLevel::WARN:
                return "WARN ";
            case LogLevel::INFO:
                return "INFO ";
            case LogLevel::VERBOSITY_DEBUG:
                return "DEBUG";
            case LogLevel::TRACE:
                return "TRACE";
            default:
                return "INFO ";
            }
        }

        LogLevel stringToLogLevel(const std::string &level_str) const
        {
            if (level_str == "ERROR" || level_str == "error")
                return LogLevel::ERROR;
            if (level_str == "WARN" || level_str == "warn")
                return LogLevel::WARN;
            if (level_str == "INFO" || level_str == "info")
                return LogLevel::INFO;
            if (level_str == "DEBUG" || level_str == "debug")
                return LogLevel::VERBOSITY_DEBUG;
            if (level_str == "TRACE" || level_str == "trace")
                return LogLevel::TRACE;
            return LogLevel::INFO; // default
        }

        /**
         * @brief Get recent log lines for debugging
         * @return Vector of recent log messages
         */
        std::vector<std::string> recent_lines() const
        {
            std::lock_guard<std::mutex> lk(buffer_mutex_);
            return std::vector<std::string>(recent_.begin(), recent_.end());
        }

        std::string getCurrentTimestamp() const
        {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          now.time_since_epoch()) %
                      1000;

            char buf[16]; // "HH:MM:SS.mmm" = 12 chars + null
            std::tm tm_buf;
            std::strftime(buf, sizeof(buf), "%H:%M:%S", localtime_r(&time_t, &tm_buf));
            // Append ".mmm"
            std::snprintf(buf + 8, sizeof(buf) - 8, ".%03d", static_cast<int>(ms.count()));
            return std::string(buf, 12);
        }

    private:
        Logger() : current_level_(LogLevel::INFO), rank_(0), has_rank_(false)
        {
            const auto &env = debugEnv();

            if (!env.logger.log_level.empty())
            {
                current_level_ = stringToLogLevel(env.logger.log_level);
            }

            if (env.logger.buffer_lines > 0)
            {
                max_buffer_ = static_cast<size_t>(env.logger.buffer_lines);
            }
        }

        /**
         * @brief Thread-local storage for device prefix
         *
         * Each thread can have its own device prefix, enabling multi-device
         * parallel execution with clear log attribution.
         */
        static std::string &thread_device_prefix()
        {
            static thread_local std::string *prefix = new std::string();
            return *prefix;
        }

        LogLevel current_level_;
        int rank_;
        bool has_rank_;
        mutable std::deque<std::string> recent_;
        mutable size_t max_buffer_ = 2048; // keep last N lines
        mutable std::mutex buffer_mutex_;
        std::ofstream log_file_;
    };

    /**
     * @brief RAII helper for scoped device logging
     *
     * Sets the thread-local device prefix on construction and clears it on destruction.
     * Use this to automatically tag all log messages within a scope with the device ID.
     *
     * Example:
     * @code
     * void executeOnDevice(DeviceId device) {
     *     ScopedDeviceLog scoped_log(device);
     *     LOG_INFO("Starting execution");  // Logs as "[rocm:0] Starting execution"
     *     // ... work ...
     * }  // Prefix automatically cleared
     * @endcode
     */
    class ScopedDeviceLog
    {
    public:
        /**
         * @brief Construct with device ID string
         * @param device_str The device string (e.g., "rocm:0", "cuda:1", "cpu")
         */
        explicit ScopedDeviceLog(const std::string &device_str)
            : previous_prefix_(Logger::getThreadDevicePrefix())
        {
            Logger::setThreadDevicePrefix(device_str);
        }

        /**
         * @brief Construct with DeviceId object
         * @param device The DeviceId to use for logging
         */
        template <typename DeviceIdType>
        explicit ScopedDeviceLog(const DeviceIdType &device)
            : previous_prefix_(Logger::getThreadDevicePrefix())
        {
            Logger::setThreadDevicePrefix(device.to_string());
        }

        ~ScopedDeviceLog()
        {
            // Restore previous prefix (supports nesting)
            Logger::setThreadDevicePrefix(previous_prefix_);
        }

        // Non-copyable, non-movable
        ScopedDeviceLog(const ScopedDeviceLog &) = delete;
        ScopedDeviceLog &operator=(const ScopedDeviceLog &) = delete;
        ScopedDeviceLog(ScopedDeviceLog &&) = delete;
        ScopedDeviceLog &operator=(ScopedDeviceLog &&) = delete;

    private:
        std::string previous_prefix_;
    };

} // namespace llaminar2

// Convenience macros for logging with file/line tracking
#define LOG_ERROR(msg)                                                                                           \
    do                                                                                                           \
    {                                                                                                            \
        if (::llaminar2::Logger::getInstance().shouldLog(::llaminar2::LogLevel::ERROR))                          \
        {                                                                                                        \
            std::ostringstream oss;                                                                              \
            oss << msg;                                                                                          \
            ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::ERROR, oss.str(), __FILE__, __LINE__); \
        }                                                                                                        \
    } while (0)

#define LOG_WARN(msg)                                                                                           \
    do                                                                                                          \
    {                                                                                                           \
        if (::llaminar2::Logger::getInstance().shouldLog(::llaminar2::LogLevel::WARN))                          \
        {                                                                                                       \
            std::ostringstream oss;                                                                             \
            oss << msg;                                                                                         \
            ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::WARN, oss.str(), __FILE__, __LINE__); \
        }                                                                                                       \
    } while (0)

#define LOG_INFO(msg)                                                                                           \
    do                                                                                                          \
    {                                                                                                           \
        if (::llaminar2::Logger::getInstance().shouldLog(::llaminar2::LogLevel::INFO))                          \
        {                                                                                                       \
            std::ostringstream oss;                                                                             \
            oss << msg;                                                                                         \
            ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::INFO, oss.str(), __FILE__, __LINE__); \
        }                                                                                                       \
    } while (0)

// LOG_DEBUG: Always runtime-gated in all build types. The shouldLog() check
// is branch-predicted away when log level < DEBUG, so the overhead is negligible.
// This allows LLAMINAR_LOG_LEVEL=DEBUG to work even in Release builds.
#define LOG_DEBUG(msg)                                                                                                     \
    do                                                                                                                     \
    {                                                                                                                      \
        if (::llaminar2::Logger::getInstance().shouldLog(::llaminar2::LogLevel::VERBOSITY_DEBUG))                          \
        {                                                                                                                  \
            std::ostringstream oss;                                                                                        \
            oss << msg;                                                                                                    \
            ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::VERBOSITY_DEBUG, oss.str(), __FILE__, __LINE__); \
        }                                                                                                                  \
    } while (0)

// LOG_TRACE: Compiled out in Release/E2ERelease/Integration (NDEBUG) builds.
// Only active in Debug builds. TRACE generates extremely verbose output that
// can materially slow down even the shouldLog() branch prediction.
#if defined(NDEBUG)

#define LOG_TRACE(msg) \
    do                 \
    {                  \
    } while (0)

#else

#define LOG_TRACE(msg)                                                                                           \
    do                                                                                                           \
    {                                                                                                            \
        if (::llaminar2::Logger::getInstance().shouldLog(::llaminar2::LogLevel::TRACE))                          \
        {                                                                                                        \
            std::ostringstream oss;                                                                              \
            oss << msg;                                                                                          \
            ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::TRACE, oss.str(), __FILE__, __LINE__); \
        }                                                                                                        \
    } while (0)

#endif // NDEBUG

// Simple logging without location info
#define LOG(level, msg)                                               \
    do                                                                \
    {                                                                 \
        if (::llaminar2::Logger::getInstance().shouldLog(level))      \
        {                                                             \
            std::ostringstream oss;                                   \
            oss << msg;                                               \
            ::llaminar2::Logger::getInstance().log(level, oss.str()); \
        }                                                             \
    } while (0)

/**
 * @brief Initialize logging from environment variable
 *
 * Reads LLAMINAR_LOG_LEVEL environment variable and sets the global log level.
 * Valid values: ERROR, WARN, INFO (default), DEBUG, TRACE
 *
 * Call this early in main() to configure logging.
 */
inline void initializeLogging()
{
    const auto &env = ::llaminar2::debugEnv();
    if (!env.logger.log_level.empty())
    {
        ::llaminar2::Logger::getInstance().setLogLevel(
            ::llaminar2::Logger::getInstance().stringToLogLevel(env.logger.log_level));
    }
}
