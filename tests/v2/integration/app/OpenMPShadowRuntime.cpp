/**
 * @file OpenMPShadowRuntime.cpp
 * @brief Harmless wrong-SONAME fixture for native loader binding preflight.
 *
 * This DSO is only dependency-traced, never used to execute a parallel region.
 * A private alias gives it the selected OpenMP runtime's filename while its
 * native SONAME remains different, reproducing a GPU SDK compatibility alias.
 */

/** @brief Provide a recognizable symbol without initializing any runtime. */
extern "C" int omp_get_max_threads()
{
    return 1;
}
