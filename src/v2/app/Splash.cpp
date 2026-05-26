/**
 * @file Splash.cpp
 * @brief Llaminar startup splash screen implementation.
 */
#include "app/Splash.h"
#include "utils/MPIBootstrap.h"
#include "utils/DebugEnv.h"

#include <cstdio>
#include <iostream>
#include <unistd.h>

namespace llaminar2
{
    namespace
    {
        // ANSI 256-color blue/cyan/water palette, light → deep.
        constexpr const char *kReset = "\033[0m";
        constexpr const char *kBold = "\033[1m";
        constexpr const char *kSky = "\033[38;5;159m";  // very pale cyan
        constexpr const char *kIce = "\033[38;5;123m";  // light cyan
        constexpr const char *kAqua = "\033[38;5;87m";  // aqua
        constexpr const char *kAzure = "\033[38;5;45m"; // bright blue
        constexpr const char *kOcean = "\033[38;5;33m"; // deep blue
        constexpr const char *kDeep = "\033[38;5;27m";  // deeper blue
        constexpr const char *kDim = "\033[38;5;24m";   // muted teal

        bool splashEnabled()
        {
            // 1. Skip if stdout is not a TTY (pipes, files, CI logs).
            if (!isatty(fileno(stdout)))
                return false;
            // 2. Skip if user opted out.
            if (debugEnv().runtime_debug.no_splash)
                return false;
            // 3. Honour NO_COLOR (https://no-color.org/) and Llaminar's color opt-out.
            if (debugEnv().runtime_debug.no_color_output)
                return false;
            // 4. Skip on MPI-spawned ranks; only the original launcher prints.
            auto info = MPIBootstrap::detectMPIEnvironment();
            if (info.is_mpi_process)
                return false;
            return true;
        }
    } // namespace

    void printSplash()
    {
        if (!splashEnabled())
            return;

        // Layout note:
        // Layout note:
        //   Splash is left-justified. Each row's "stream zone" before the
        //   wordmark is exactly 23 chars wide:
        //       leading_spaces(r) + "\  \  \  \  \  \"(16) + trailing_spaces(7-r)
        //   The 6 backslashes shift +1 column per row to form perfectly
        //   parallel 45° diagonals (laminar flow), but the wordmark stays
        //   straight at column 24 because trailing pad shrinks by the same amount.
        std::cout
            << '\n'
            // ── 6 parallel streams + fixed-column wordmark ────────────────
            << kAqua << "\\  \\  \\  \\  \\  \\" << kReset << "         "
            << kBold << kOcean << "██╗     ██╗      █████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██████╗" << kReset << '\n'
            << " " << kAqua << "\\  \\  \\  \\  \\  \\" << kReset << "        "
            << kBold << kOcean << "██║     ██║     ██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██╔══██╗" << kReset << '\n'
            << "  " << kAzure << "\\  \\  \\  \\  \\  \\" << kReset << "       "
            << kBold << kOcean << "██║     ██║     ███████║██╔████╔██║██║██╔██╗ ██║███████║██████╔╝" << kReset << '\n'
            << "   " << kAzure << "\\  \\  \\  \\  \\  \\" << kReset << "      "
            << kBold << kDeep << "██║     ██║     ██╔══██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║██╔══██╗" << kReset << '\n'
            << "    " << kOcean << "\\  \\  \\  \\  \\  \\" << kReset << "     "
            << kBold << kDeep << "███████╗███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║██║  ██║" << kReset << '\n'
            << "     " << kDeep << "\\  \\  \\  \\  \\  \\" << kReset << "    "
            << kDim << "╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝" << kReset << '\n'
            // ── ripples beneath the landing zone + tagline ────────────────
            << "      " << kDim << "~  ~  ~  ~  ~  ~" << kReset << "                "
            << kDim << "l l m   i n f e r e n c e" << kReset << '\n'
            << '\n';
        std::cout.flush();
    }

} // namespace llaminar2
