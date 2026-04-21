/**
 * @file Splash.cpp
 * @brief Llaminar startup splash screen implementation.
 */
#include "app/Splash.h"
#include "utils/MPIBootstrap.h"

#include <cstdio>
#include <cstdlib>
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
            if (std::getenv("LLAMINAR_NO_SPLASH"))
                return false;
            // 3. Honour NO_COLOR (https://no-color.org/).
            if (std::getenv("NO_COLOR"))
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

        // Tilted shower head (45° down-right) with a column of perfectly
        // parallel diagonal water streams — laminar flow.  Streams shift
        // +1 column right per row so each "\" continues the diagonal of
        // the row above it. The wordmark sits to the right of the streams.
        std::cout
            << '\n'
            // ── tilted shower head ────────────────────────────────────────
            << "  " << kAqua  << "      .-~~~-." << kReset << '\n'
            << "  " << kAqua  << "     ( " << kSky << "o o" << kAqua << "  '\\." << kReset << '\n'
            << "  " << kAzure << "      \\_" << kIce << "o_o_o" << kAzure << "_'\\." << kReset << '\n'
            // ── 6 parallel streams + wordmark ─────────────────────────────
            << "  " << "                " << kAqua  << "\\  \\  \\  \\  \\  \\"  << kReset << "  "
            << kBold << kOcean << " ██╗     ██╗      █████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██████╗ " << kReset << '\n'
            << "  " << "                 " << kAqua  << "\\  \\  \\  \\  \\  \\"  << kReset << "  "
            << kBold << kOcean << " ██║     ██║     ██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██╔══██╗" << kReset << '\n'
            << "  " << "                  " << kAzure << "\\  \\  \\  \\  \\  \\"  << kReset << "  "
            << kBold << kOcean << " ██║     ██║     ███████║██╔████╔██║██║██╔██╗ ██║███████║██████╔╝" << kReset << '\n'
            << "  " << "                   " << kAzure << "\\  \\  \\  \\  \\  \\"  << kReset << "  "
            << kBold << kDeep  << " ██║     ██║     ██╔══██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║██╔══██╗" << kReset << '\n'
            << "  " << "                    " << kOcean << "\\  \\  \\  \\  \\  \\"  << kReset << "  "
            << kBold << kDeep  << " ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║██║  ██║" << kReset << '\n'
            << "  " << "                     " << kDeep  << "\\  \\  \\  \\  \\  \\"  << kReset << "  "
            << kDim           << " ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝" << kReset << '\n'
            // ── ripples beneath where the streams land ────────────────────
            << "  " << "                      " << kDim  << "~  ~  ~  ~  ~  ~" << kReset << "      "
            << kDim << "l a m i n a r   l l m   i n f e r e n c e" << kReset << '\n'
            << '\n';
        std::cout.flush();
    }

} // namespace llaminar2
