/**
 * @file CUDADriverVisibilityAudit.cpp
 * @brief Test-only ELF audit module that models a host without an NVIDIA driver.
 *
 * glibc asks this module before resolving a shared library, including dlopen.
 * Denying just libcuda reproduces a CPU-only host without hiding CUDA runtime,
 * cuBLAS, ROCm or the full shared core. No replacement driver or fake success is
 * supplied. This module is loaded only by its explicit preflight subprocess.
 */
#include <link.h>
#include <cstring>

extern "C"
{
/** @return The supported glibc audit ABI; no application state is inspected. */
unsigned int la_version(unsigned int)
{
    return LAV_CURRENT;
}

/**
 * @brief Make every NVIDIA driver lookup fail, preserving all other libraries.
 * @param name Candidate ELF library name or resolved pathname.
 * @return Original name, or a deliberately nonexistent exact test path.
 */
char *la_objsearch(const char *name, uintptr_t *, unsigned int)
{
    const char *base = std::strrchr(name, '/');
    base = base ? base + 1 : name;
    if (std::strcmp(base, "libcuda.so") == 0 ||
        std::strncmp(base, "libcuda.so.", 11) == 0)
    {
        static char unavailable[] = "/llaminar-test-no-nvidia-driver/libcuda.so.1";
        return unavailable;
    }
    return const_cast<char *>(name);
}
}
