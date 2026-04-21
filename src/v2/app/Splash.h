/**
 * @file Splash.h
 * @brief Llaminar startup splash screen (ASCII shower + ANSI blue gradient).
 */
#pragma once

namespace llaminar2
{

    /**
     * Print the Llaminar splash screen to stdout.
     *
     * No-op when:
     *   - stdout is not a TTY (piped, redirected, in a CI log)
     *   - the process is an MPI-spawned rank (only the original launcher prints)
     *   - the NO_COLOR environment variable is set
     *   - the LLAMINAR_NO_SPLASH environment variable is set
     */
    void printSplash();

} // namespace llaminar2
