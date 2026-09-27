/**
 * @file HostRegistrationPageBoundary.h
 * @brief Isolate registered host-range edges from neighbouring huge-page reclaim.
 *
 * Linux can expand a neighbour's small MADV_DONTNEED into a whole huge-PMD
 * invalidation. A ROCm USERPTR buffer object then evicts its process queues and
 * revalidates live pages even though no registered byte was discarded. Marking
 * just the first and last base pages NOHUGEPAGE before registration splits that
 * shared PMD while no GPU owns the range. Interior huge pages remain eligible.
 * This changes VMA policy only: no allocation, accounting, data copy or GPU
 * synchronization is introduced. TransferEngine still owns registration life.
 */
#pragma once

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <system_error>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace llaminar2
{
    /**
     * @brief Immutable base-page boundaries of one positive host registration.
     *
     * Geometry describes virtual page policy, not a second physical-memory
     * ledger. Only the public builder can construct a valid nonempty boundary.
     */
    class HostRegistrationPageBoundary
    {
    public:
        /**
         * @brief Resolve the distinct edge pages without overflowing the address.
         * @param address First byte submitted to the native registration API.
         * @param bytes Positive logical registration extent.
         * @param page_bytes Runtime base-page size, a positive power of two.
         * @return One or two page starts, never an address outside the pin span.
         * @throws std::invalid_argument For null/empty or invalid page geometry.
         * @throws std::overflow_error If the last registered address overflows.
         */
        [[nodiscard]] static HostRegistrationPageBoundary forRange(
            std::uintptr_t address, std::size_t bytes, std::size_t page_bytes)
        {
            if (!address || !bytes || !page_bytes ||
                (page_bytes & (page_bytes - 1u)) != 0u)
                throw std::invalid_argument("Host registration requires a positive range and power-of-two page size");
            if (bytes - 1u > std::numeric_limits<std::uintptr_t>::max() - address)
                throw std::overflow_error("Host registration address range overflows");
            const auto mask = ~(static_cast<std::uintptr_t>(page_bytes) - 1u);
            return HostRegistrationPageBoundary(
                address & mask, (address + bytes - 1u) & mask, page_bytes);
        }

        /** @return Distinct base-page starts, in increasing address order. */
        [[nodiscard]] std::span<const std::uintptr_t> pages() const noexcept
        {
            return {pages_.data(), count_};
        }

        /** @return Length of each boundary page, not a physical capacity charge. */
        [[nodiscard]] std::size_t pageBytes() const noexcept { return page_bytes_; }

        /**
         * @brief Install edge isolation before the native runtime pins the range.
         * @param pointer Stable caller-owned host storage, not a GPU alias.
         * @param bytes Exact positive registration extent.
         * @throws std::system_error If Linux cannot establish either boundary.
         *
         * The advice is retained until the storage owner reuses/unmaps its
         * pages. Re-enabling huge pages at unregister could merge an edge with
         * a neighbouring registration that is still live. No byte is discarded
         * and no permission is changed, so concurrent host readers remain valid.
         */
        static void prepare(void *pointer, std::size_t bytes)
        {
#ifdef __linux__
            const long page_bytes = ::sysconf(_SC_PAGESIZE);
            if (page_bytes <= 0)
                throw std::runtime_error("Cannot resolve host registration page size");
            const auto boundary = forRange(reinterpret_cast<std::uintptr_t>(pointer),
                bytes, static_cast<std::size_t>(page_bytes));
            for (const auto page : boundary.pages())
                if (::madvise(reinterpret_cast<void *>(page), boundary.pageBytes(),
                              MADV_NOHUGEPAGE) != 0)
                    throw std::system_error(errno, std::generic_category(),
                        "Cannot isolate host registration from neighbouring huge-page reclaim");
#else
            (void)pointer;
            (void)bytes;
            throw std::runtime_error("Host registration page isolation requires Linux");
#endif
        }

    private:
        /** @brief Construct only checked page geometry supplied by forRange(). */
        HostRegistrationPageBoundary(std::uintptr_t first, std::uintptr_t last,
                                     std::size_t page_bytes) noexcept
            : pages_{first, last}, count_(first == last ? 1u : 2u), page_bytes_(page_bytes) {}

        std::array<std::uintptr_t, 2> pages_;
        std::size_t count_;
        std::size_t page_bytes_;
    };
}
