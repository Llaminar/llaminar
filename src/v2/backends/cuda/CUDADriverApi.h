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
