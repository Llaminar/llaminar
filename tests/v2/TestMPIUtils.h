/**
 * @file TestMPIUtils.h
 * @brief Shared helpers for MPI-enabled tests (sanitizer-friendly wrappers)
 */

#pragma once

#include <mpi.h>

// Detect whether leak sanitizer hooks are available
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define LLAMINAR_TESTS_HAS_LSAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define LLAMINAR_TESTS_HAS_LSAN 1
#endif

#ifdef LLAMINAR_TESTS_HAS_LSAN
#include <sanitizer/lsan_interface.h>

extern "C" __attribute__((weak, used)) const char *__lsan_default_suppressions()
{
    // Suppress known leaks from Open MPI's libevent/hwloc progress threads and their worker symbols.
    return "leak:libevent_core\n"
           "leak:event_base_loop\n"
           "leak:opal_progress\n"
           "leak:opal_libevent2022_progress\n"
           "leak:libopen-pal\n"
           "leak:libmpi\n"
           "leak:hwloc\n";
}
#endif

namespace llaminar2::tests
{
    inline void disable_mpi_leak_reports_once()
    {
#ifdef LLAMINAR_TESTS_HAS_LSAN
        // Open MPI spawns libevent progress threads that never release their resources.
        // Suppress leak checking once so allocations made after MPI_Init do not trip LSAN.
        static bool leak_checks_disabled = false;
        if (!leak_checks_disabled)
        {
            __lsan_disable();
            leak_checks_disabled = true;
        }
#endif
    }

    /**
     * @brief Wrap MPI_Init_thread so that leak sanitizer ignores hwloc allocations.
     */
    inline void mpi_init_thread_sanitizer_safe(int *argc, char ***argv, int required, int *provided)
    {
        disable_mpi_leak_reports_once();
        MPI_Init_thread(argc, argv, required, provided);
    }

    /**
     * @brief Wrap MPI_Finalize with leak sanitizer suppression.
     */
    inline void mpi_finalize_sanitizer_safe()
    {
        disable_mpi_leak_reports_once();
        MPI_Finalize();
    }
}

#undef LLAMINAR_TESTS_HAS_LSAN
