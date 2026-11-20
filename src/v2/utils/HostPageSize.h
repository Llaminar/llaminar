/**
 * @file HostPageSize.h
 * @brief One authority for the host's base page size.
 *
 * NUMA memory policy (mbind), madvise, first-touch placement and page-rounded
 * admission all act on whole base pages. AArch64 kernels may run 4 KiB,
 * 16 KiB or 64 KiB pages (Grace/Neoverse-V2 servers commonly 64 KiB). A
 * hard-coded 4 KiB page makes mbind/madvise fail with EINVAL on misaligned
 * ranges, and a sub-page madvise(MADV_DONTNEED) range is rounded by the
 * kernel to the whole page, discarding neighbouring live bytes. Every such
 * owner reads the size here instead of assuming it.
 *
 * The value is observed once per process; it cannot change while running.
 * Allocation owners pass it into pure helpers, so their arithmetic is tested
 * for every page size without a test-only override of this function.
 */
#pragma once

#include <cstddef>
#include <stdexcept>
#ifdef __linux__
#include <unistd.h>
#endif

namespace llaminar2
{
    /**
     * @brief Validate one observed base page size.
     * @param bytes Raw sysconf(_SC_PAGESIZE) result.
     * @return The page size in bytes.
     * @throws std::runtime_error Unless the value is a positive power of two.
     */
    [[nodiscard]] inline std::size_t validatedHostPageBytes(long bytes)
    {
        if (bytes <= 0 || (bytes & (bytes - 1)) != 0)
            throw std::runtime_error("Host base page size is not a positive power of two");
        return static_cast<std::size_t>(bytes);
    }

    /**
     * @brief Return the process's host base page size.
     * @return Bytes per base page, observed once.
     * @throws std::runtime_error When the page size cannot be observed.
     */
    [[nodiscard]] inline std::size_t hostPageBytes()
    {
#ifdef __linux__
        static const std::size_t bytes = validatedHostPageBytes(::sysconf(_SC_PAGESIZE));
        return bytes;
#else
        throw std::runtime_error("Host page size observation requires Linux");
#endif
    }
} // namespace llaminar2
