/**
 * @file IntentionalFaultProcess.h
 * @brief Keep intentional fatal test children out of host crash collectors.
 */
#pragma once

#include <sys/resource.h>
#include <unistd.h>
#if defined(__linux__)
#include <sys/prctl.h>
#endif

namespace llaminar2::test
{
    /**
     * @brief Disable dump collection in the current fault-injection child.
     *
     * Linux piped core handlers can run despite RLIMIT_CORE being zero.
     * Suppressing dumps still preserves the child's original terminating signal.
     */
    inline void disableIntentionalFaultCoreDump()
    {
        const rlimit disabled{0, 0};
        if (setrlimit(RLIMIT_CORE, &disabled) != 0)
            ::_exit(98);
#if defined(__linux__)
        if (prctl(PR_SET_DUMPABLE, 0L, 0L, 0L, 0L) != 0)
            ::_exit(98);
#endif
    }
}
