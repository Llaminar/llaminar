/**
 * @file AppLifecycle.h
 * @brief Top-level application orchestrator: bootstrap -> init -> mode dispatch
 */

#pragma once

namespace llaminar2
{

    /**
     * @brief Owns the full startup lifecycle: parse -> bootstrap -> init -> dispatch
     *
     * Usage:
     *   AppLifecycle app;
     *   return app.run(argc, argv);
     */
    class AppLifecycle
    {
    public:
        /**
         * @brief Dispatch one command inside the executable's lexical GPU lifetime.
         * @param argc Process argument count. @param argv Process argument values.
         * @return Command exit status after command-owned runtime resources unwind.
         * @throws std::logic_error If no active process GPU owner precedes dispatch.
         */
        int run(int argc, char *argv[]);
    };

} // namespace llaminar2
