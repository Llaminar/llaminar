/**
 * @file IExecutionMode.h
 * @brief Interface for pluggable execution modes
 */

#pragma once

namespace llaminar2
{

    struct AppContext;
    struct OrchestrationConfig;

    /**
     * @brief Interface for pluggable execution modes
     *
     * Each mode (chat, single-shot, benchmark, completion, server)
     * implements this interface. AppLifecycle iterates registered modes;
     * the first one whose matches() returns true gets dispatched.
     */
    class IExecutionMode
    {
    public:
        virtual ~IExecutionMode() = default;

        /// Human-readable mode name (for logging/diagnostics)
        virtual const char *name() const = 0;

        /// Does this mode handle the given config?
        virtual bool matches(const OrchestrationConfig &config) const = 0;

        /// Execute the mode. Returns process exit code.
        virtual int execute(AppContext &ctx) = 0;
    };

} // namespace llaminar2
