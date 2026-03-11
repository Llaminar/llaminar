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
        int run(int argc, char *argv[]);
    };

} // namespace llaminar2
