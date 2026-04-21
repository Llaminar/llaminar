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
        constexpr const char *kSky = "\033[38;5;159m";    // very pale cyan
        constexpr const char *kIce = "\033[38;5;123m";    // light cyan
        constexpr const char *kAqua = "\033[38;5;87m";    // aqua
        constexpr const char *kAzure = "\033[38;5;45m";   // bright blue
        constexpr const char *kOcean = "\033[38;5;33m";   // deep blue
        constexpr const char *kDeep = "\033[38;5;27m";    // deeper blue
        constexpr const char *kDim = "\033[38;5;24m";     // muted teal

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

        // Shower head: rounded top, perforated underside.
        // Water column: gradient cyan ribbons of varying density.
        // Wordmark: blocky LLAMINAR in azure/ocean blue.
        std::cout
            << '\n'
            << "  " << kAqua << "      .-~~~~~~-.        " << kReset << '\n'
            << "  " << kAqua << "    .'  " << kSky << "o o o o" << kAqua << "  '.      " << kReset << '\n'
            << "  " << kAqua << "   /   " << kSky << "o o o o o" << kAqua << "   \\     " << kReset << '\n'
            << "  " << kAzure << "  '._" << kIce << ".o.o.o.o.o._" << kAzure << ".'    " << kReset << '\n'
            << "  " << kAzure << "     " << kIce << "│ │ │ │ │ │      " << kReset
            << kBold << kOcean << " ██╗     ██╗      █████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██████╗ " << kReset << '\n'
            << "  " << kAzure << "     " << kAqua << "╵ │ ╵ │ ╵ │      " << kReset
            << kBold << kOcean << " ██║     ██║     ██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██╔══██╗" << kReset << '\n'
            << "  " << kOcean << "       " << kAqua << "│ ╵ │ ╵ │       " << kReset
            << kBold << kDeep  << " ██║     ██║     ███████║██╔████╔██║██║██╔██╗ ██║███████║██████╔╝" << kReset << '\n'
            << "  " << kOcean << "       " << kAqua << "╵   ╵   ╵       " << kReset
            << kBold << kDeep  << " ██║     ██║     ██╔══██║██║╚██╔╝██║██║██║╚██╗██║██╔══██║██╔══██╗" << kReset << '\n'
            << "  " << kDeep  << "                       " << kReset
            << kBold << kDeep  << " ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║██║  ██║" << kReset << '\n'
            << "  " << kDim   << "        ~ ~  ~ ~       " << kReset
            << kDim           << " ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═╝" << kReset << '\n'
            << "  " << kDim   << "      ~  ~ ~  ~ ~        " << kReset
            << kDim           << "        l a m i n a r   l l m   i n f e r e n c e" << kReset << '\n'
            << '\n';
        std::cout.flush();
    }

} // namespace llaminar2
