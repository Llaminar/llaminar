/**
 * @file CUDAHardwareQueryObserver.cpp
 * @brief Host-only ELF observers forwarding actual CUDA discovery without substitution.
 *
 * The selected shared cudart remains the implementation authority. Each exported
 * observer counts its real invocation and forwards the exact arguments/status
 * through RTLD_NEXT; no count, property or identity is fabricated. Keeping these
 * definitions outside nvcc avoids colliding with its device-runtime builtins.
 */
#include "CUDAHardwareQueryObserver.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <atomic>
#include <cstdio>
#include <exception>

namespace
{
    std::atomic<unsigned> count_queries{0}, property_queries{0}, pci_queries{0};

    /**
     * @brief Resolve the actual dynamic runtime following this executable's observer.
     * @tparam Function Exact host runtime function pointer type.
     * @param name Exported native runtime symbol; never a device-runtime builtin.
     * @return Forwarding target; absence terminates this native qualification proof.
     */
    template <typename Function> Function nativeSymbol(const char *name)
    {
        (void)dlerror();
        auto function = reinterpret_cast<Function>(dlsym(RTLD_NEXT, name));
        if (!function)
        {
            const char *error = dlerror();
            std::fprintf(stderr, "CUDA discovery observer cannot resolve shared runtime symbol %s: %s\n",
                name, error ? error : "no matching native symbol");
            std::terminate();
        }
        return function;
    }
}

namespace llaminar2::test::cuda_metadata_native
{
    std::array<unsigned, 3> queryCounts() noexcept
    {
        return {count_queries.load(std::memory_order_relaxed),
                property_queries.load(std::memory_order_relaxed),
                pci_queries.load(std::memory_order_relaxed)};
    }
}

/** @brief Observe count discovery and forward the exact native status/output. */
extern "C" cudaError_t CUDARTAPI cudaGetDeviceCount(int *count)
{
    count_queries.fetch_add(1, std::memory_order_relaxed);
    static const auto native = nativeSymbol<decltype(&cudaGetDeviceCount)>("cudaGetDeviceCount");
    return native(count);
}
/** @brief Observe property discovery, including callers in a shared backend. */
extern "C" cudaError_t CUDARTAPI cudaGetDeviceProperties(cudaDeviceProp *properties, int ordinal)
{
    property_queries.fetch_add(1, std::memory_order_relaxed);
    static const auto native = nativeSymbol<decltype(&cudaGetDeviceProperties)>("cudaGetDeviceProperties");
    return native(properties, ordinal);
}
/** @brief Observe PCI discovery without replacing its native identity authority. */
extern "C" cudaError_t CUDARTAPI cudaDeviceGetPCIBusId(char *pci, int length, int ordinal)
{
    pci_queries.fetch_add(1, std::memory_order_relaxed);
    static const auto native = nativeSymbol<decltype(&cudaDeviceGetPCIBusId)>("cudaDeviceGetPCIBusId");
    return native(pci, length, ordinal);
}
