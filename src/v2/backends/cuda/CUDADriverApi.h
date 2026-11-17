/**
 * @file CUDADriverApi.h
 * @brief Typed, immutable CUDA Driver API binding without an ELF driver dependency.
 *
 * Full-backend binaries must also run on CPU-only cluster members. Loading the
 * executable must therefore not require the host NVIDIA driver. CUDA backend
 * preparation binds every required native entry point once, before recording
 * graphs. Calls still use the original driver operations, exact streams and
 * native return codes; no feature substitution or runtime re-resolution occurs.
 */
#pragma once

#include <cuda.h>

#include "CUDADiscoveryAdmission.h"

namespace llaminar2
{
/**
 * @brief Complete required Driver API table, retained for the process lifetime.
 *
 * Binding does not initialize a driver context. Only CUDA callers acquire this
 * table, so CPU/ROCm startup and test-name discovery need no NVIDIA installation.
 * The library remains loaded through static destruction and CUDA runtime reset;
 * a model or context generation must never invalidate native code addresses.
 */
class CUDADriverApi final
{
private:
    void *const library_; ///< Pinned native code; deliberately not closed at teardown.

public:
    /**
     * @return The one completely bound native API table.
     * @throws std::runtime_error If the driver or any exact SDK symbol is absent.
     * @note CUDA preparation calls this before capture. Subsequent calls only
     *       read the C++ thread-safe initialized table; they do not call dlsym.
     */
    static const CUDADriverApi &instance();

    /**
     * @brief Observe the optional native driver's exact loader binding status.
     * @return Typed binding result for discovery admission; never an exception.
     * @note This is a pure observation for the discovery policy: it performs one
     *       dlopen/dlclose cycle with the binding's exact flags and SONAME and
     *       never pins the library, resolves further symbols, initializes a
     *       driver context or replaces the bound table. Only a positively
     *       absent library is reported as non-installation; a present library
     *       that cannot bind, for any reason, is reported as present.
     */
    static CUDADriverBindingStatus probeDriverBinding() noexcept;

#define LLAMINAR_CUDA_DRIVER_ENTRY(member, symbol) \
    /** Exact SDK ABI; non-null after complete construction. */ \
    decltype(&::symbol) const member;
#include "CUDADriverFunctions.def"
#undef LLAMINAR_CUDA_DRIVER_ENTRY

private:
    /** @brief Bind the complete table; the factory retains or retires the handle. */
    explicit CUDADriverApi(void *library);
};
} // namespace llaminar2
