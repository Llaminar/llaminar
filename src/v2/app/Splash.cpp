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

        // Layout note:
        //   The wordmark is held at a FIXED column on every row (column 44 from
        //   the start of the line). Each row's "stream zone" before the
        //   wordmark is exactly 41 chars wide:
        //       leading_spaces(16+r) + "\  \  \  \  \  \"(16) + trailing_spaces(7-r)
        //   The 6 backslashes shift +1 column per row to form perfectly
        //   parallel 45° diagonals (laminar flow), but the wordmark stays
        //   straight because trailing pad shrinks by the same amount.
        //
        //   The shower head is a flat perforated disc spanning columns 5-17.
        //   Its bottom-right spout `\` lands at column 18 — exactly one
        //   diagonal step (+1 row, +1 col) from row 0's leftmost stream `\`
        //   at column 19, so water visibly emerges from the spout into the
        //   laminar flow column.
        std::cout
            << '\n'
            // ── perforated showerhead (flat disc, spout at bottom-right) ──
            //   Row 1 (dome top):    cols 6-17  ".-========-."
            //   Row 2 (face+holes):  cols 5-17  "| o o o o o |"
            //   Row 3 (bottom+spout):cols 6-17  "'-========-'"  + spout `\` at col 18
            << "  " << kAqua  << "   .-========-." << kReset << '\n'
            << "  " << kAqua  << "  | "
                   << kSky    << "o o o o o"
                   << kAqua   << " |" << kReset << '\n'
            << "  " << kAzure << "   '-========-'" << kIce << "\\" << kReset << '\n'
            // ── 6 parallel streams + fixed-column wordmark ────────────────
            << "  " << "                " << kAqua  << "\\  \\  \\  \\  \\  \\"  << kReset << "         "
            << kBold << kOcean << "██╗     ██╗      █████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██████╗" << kReset << '\n'
            << "  " << "                 " << kAqua  << "\\  \\  \\  \\  \\  \\"  << kReset << "        "
            << kBold << kOcean << "██║     ██║     ██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██╔══██╗" << kReset << '\n'
            << "  " << "                  " << kAzure << "\\  \\  \\  \\  \\  \\"  << kReset << "       "
            << kBold << kOcean << "██║     ██║     ███████║██╔████╔██║██║██╔██╗ ██║███████║██████╔╝" << kReset << '\n'
            << "  " << "                   " << kAzure << "\\  \\  \\  \\  \\  \\"  << kReset << "      "
            << kBold << kDeep  << "██║     ██║     ██╔══██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║██╔══██╗" << kReset << '\n'
            << "  " << "                    " << kOcean << "\\  \\  \\  \\  \\  \\"  << kReset << "     "
            << kBold << kDeep  << "███████╗███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║██║  ██║" << kReset << '\n'
            << "  " << "                     " << kDeep  << "\\  \\  \\  \\  \\  \\"  << kReset << "    "
            << kDim           << "╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝" << kReset << '\n'
            // ── ripples beneath the landing zone + tagline ────────────────
            << "  " << "                      " << kDim  << "~  ~  ~  ~  ~  ~" << kReset << "                "
            << kDim << "l l m   i n f e r e n c e" << kReset << '\n'
            << '\n';
        std::cout.flush();
    }

} // namespace llaminar2
