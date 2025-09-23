// Shared logging bootstrap for all test binaries.
// Ensures consistent log level escalation when diagnostic env vars are set.
// Linked into every test target via CMake so that static object constructor runs early.

#include "logger.h"
#include <cstdlib>

namespace {
struct LoggingBootstrap {
    LoggingBootstrap() {
        // If user explicitly set LLAMINAR_LOG_LEVEL, honor it (initializeLogging reads it).
        // Otherwise pick a default: DEBUG for general tests, TRACE if diagnostic env hints present.
        const char* env_level = std::getenv("LLAMINAR_LOG_LEVEL");
        bool diag = (std::getenv("LLAMINAR_COSMA_DIAG") || std::getenv("LLAMINAR_COSMA_TEST_TRACE") || std::getenv("LLAMINAR_COSMA_DIAG_DEEP"));
        if (!env_level) {
            Logger::getInstance().setLogLevel(diag ? LogLevel::TRACE : LogLevel::VERBOSITY_DEBUG);
        } else {
            // initializeLogging() will parse this later if not already parsed
            Logger::getInstance().setLogLevel(Logger::getInstance().stringToLogLevel(env_level));
        }
        // Ensure subsequent code that calls initializeLogging won't downgrade a higher requested level.
    }
};
static LoggingBootstrap _logging_bootstrap_instance; // NOLINT
} // anonymous namespace
