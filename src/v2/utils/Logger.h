#pragma once

/**
 * @file Logger.h
 * @brief Thread-safe logging infrastructure for Llaminar V2
 * @author David Sanftenberg
 */

#include "LogLevel.h"
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <deque>
#include <mutex>
#include <vector>
#include <cstdlib>

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
            static Logger instance;
            return instance;
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
        static void setThreadDevicePrefix(const std::string& prefix)
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
        static const std::string& getThreadDevicePrefix()
        {
            return thread_device_prefix();
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
            const std::string& device_prefix = thread_device_prefix();
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

            std::cout << full_line << std::endl;
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

            std::ostringstream oss;
            oss << std::put_time(std::localtime(&time_t), "%H:%M:%S");
            oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
            return oss.str();
        }

    private:
        Logger() : current_level_(LogLevel::INFO), rank_(0), has_rank_(false)
        {
            // Check environment variable for log level override
            const char *level_env = std::getenv("LLAMINAR_LOG_LEVEL");
            if (level_env)
            {
                current_level_ = stringToLogLevel(std::string(level_env));
            }

            // Check environment variable for buffer size override
            const char *buffer_env = std::getenv("LLAMINAR_LOG_BUFFER_LINES");
            if (buffer_env)
            {
                int lines = std::atoi(buffer_env);
                if (lines > 0)
                    max_buffer_ = static_cast<size_t>(lines);
            }
        }

        /**
         * @brief Thread-local storage for device prefix
         * 
         * Each thread can have its own device prefix, enabling multi-device
         * parallel execution with clear log attribution.
         */
        static std::string& thread_device_prefix()
        {
            thread_local std::string prefix;
            return prefix;
        }

        LogLevel current_level_;
        int rank_;
        bool has_rank_;
        mutable std::deque<std::string> recent_;
        mutable size_t max_buffer_ = 2048; // keep last N lines
        mutable std::mutex buffer_mutex_;
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
        explicit ScopedDeviceLog(const std::string& device_str)
            : previous_prefix_(Logger::getThreadDevicePrefix())
        {
            Logger::setThreadDevicePrefix(device_str);
        }

        /**
         * @brief Construct with DeviceId object
         * @param device The DeviceId to use for logging
         */
        template<typename DeviceIdType>
        explicit ScopedDeviceLog(const DeviceIdType& device)
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
        ScopedDeviceLog(const ScopedDeviceLog&) = delete;
        ScopedDeviceLog& operator=(const ScopedDeviceLog&) = delete;
        ScopedDeviceLog(ScopedDeviceLog&&) = delete;
        ScopedDeviceLog& operator=(ScopedDeviceLog&&) = delete;

    private:
        std::string previous_prefix_;
    };

} // namespace llaminar2

// Convenience macros for logging with file/line tracking
#define LOG_ERROR(msg)                                                                                       \
    do                                                                                                       \
    {                                                                                                        \
        std::ostringstream oss;                                                                              \
        oss << msg;                                                                                          \
        ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::ERROR, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_WARN(msg)                                                                                       \
    do                                                                                                      \
    {                                                                                                       \
        std::ostringstream oss;                                                                             \
        oss << msg;                                                                                         \
        ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::WARN, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_INFO(msg)                                                                                       \
    do                                                                                                      \
    {                                                                                                       \
        std::ostringstream oss;                                                                             \
        oss << msg;                                                                                         \
        ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::INFO, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_DEBUG(msg)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        std::ostringstream oss;                                                                                        \
        oss << msg;                                                                                                    \
        ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::VERBOSITY_DEBUG, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_TRACE(msg)                                                                                       \
    do                                                                                                       \
    {                                                                                                        \
        std::ostringstream oss;                                                                              \
        oss << msg;                                                                                          \
        ::llaminar2::Logger::getInstance().log(::llaminar2::LogLevel::TRACE, oss.str(), __FILE__, __LINE__); \
    } while (0)

// Simple logging without location info
#define LOG(level, msg)                                           \
    do                                                            \
    {                                                             \
        std::ostringstream oss;                                   \
        oss << msg;                                               \
        ::llaminar2::Logger::getInstance().log(level, oss.str()); \
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
    const char *log_level_env = std::getenv("LLAMINAR_LOG_LEVEL");
    if (log_level_env)
    {
        ::llaminar2::Logger::getInstance().setLogLevel(
            ::llaminar2::Logger::getInstance().stringToLogLevel(log_level_env));
    }
}
