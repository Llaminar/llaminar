#pragma once

#include "log_level.h"
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <deque>
#include <mutex>
#include "utils/debug_env.h"

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

        std::string full_line = "[" + timestamp + "] [" + level_str + "]" + location + " " + message;
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

private:
    Logger() : current_level_(LogLevel::INFO)
    {
        // Allow overriding ring buffer size via environment variable
        if (const char *env = std::getenv("LLAMINAR_LOG_BUFFER_LINES"))
        {
            try
            {
                size_t v = static_cast<size_t>(std::stoull(env));
                if (v > 0 && v < 2000000)
                { // cap to a reasonable upper bound
                    max_buffer_ = v;
                }
            }
            catch (...)
            {
                // ignore parse errors
            }
        }
    }
    LogLevel current_level_;
    mutable std::deque<std::string> recent_;
    mutable size_t max_buffer_ = 2048; // keep last N lines
    mutable std::mutex buffer_mutex_;

public:
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
};

// Convenience macros for logging
#define LOG_ERROR(msg)                                                             \
    do                                                                             \
    {                                                                              \
        std::ostringstream oss;                                                    \
        oss << msg;                                                                \
        Logger::getInstance().log(LogLevel::ERROR, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_WARN(msg)                                                             \
    do                                                                            \
    {                                                                             \
        std::ostringstream oss;                                                   \
        oss << msg;                                                               \
        Logger::getInstance().log(LogLevel::WARN, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_INFO(msg)                                                             \
    do                                                                            \
    {                                                                             \
        std::ostringstream oss;                                                   \
        oss << msg;                                                               \
        Logger::getInstance().log(LogLevel::INFO, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_DEBUG(msg)                                                                       \
    do                                                                                       \
    {                                                                                        \
        std::ostringstream oss;                                                              \
        oss << msg;                                                                          \
        Logger::getInstance().log(LogLevel::VERBOSITY_DEBUG, oss.str(), __FILE__, __LINE__); \
    } while (0)

#define LOG_TRACE(msg)                                                             \
    do                                                                             \
    {                                                                              \
        std::ostringstream oss;                                                    \
        oss << msg;                                                                \
        Logger::getInstance().log(LogLevel::TRACE, oss.str(), __FILE__, __LINE__); \
    } while (0)

// Simple logging without location info
#define LOG(level, msg)                              \
    do                                               \
    {                                                \
        std::ostringstream oss;                      \
        oss << msg;                                  \
        Logger::getInstance().log(level, oss.str()); \
    } while (0)

// Initialize logging from environment variable if set
inline void initializeLogging()
{
    // Use centralized snapshot; calling debugEnv() here will perform lazy parse once.
    const auto &snap = ::llaminar::debugEnv();
    if (snap.logging.log_level_active)
    {
        Logger::getInstance().setLogLevel(
            Logger::getInstance().stringToLogLevel(snap.logging.log_level.c_str()));
    }
}