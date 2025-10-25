#pragma once

/**
 * @file LogLevel.h
 * @brief Log level enumeration for Llaminar V2
 * @author David Sanftenberg
 */

namespace llaminar2
{

    /**
     * @brief Log levels for verbosity control
     *
     * Levels are ordered by severity (ERROR is most critical, TRACE is most verbose).
     * When a log level is set, all messages at that level and below are shown.
     */
    enum class LogLevel
    {
        ERROR = 0,           ///< Critical errors that prevent operation
        WARN = 1,            ///< Warnings about concerning conditions
        INFO = 2,            ///< Important runtime information (default)
        VERBOSITY_DEBUG = 3, ///< Detailed debugging information
        TRACE = 4            ///< Very verbose execution tracing
    };

} // namespace llaminar2
