/**
 * @file HostRegistrationPageRange.h
 * @brief Keep every caller-registered host page outside transparent huge pages.
 *
 * KFD USERPTR mappings can be invalidated when khugepaged coalesces an interior
 * PMD, even when no registered byte is discarded. Protecting only edge pages
 * cannot prevent that operation. Before native registration, one NOHUGEPAGE
 * advice covers the complete base-page-rounded range, preserving payload and
 * the huge-page eligibility of adjacent, wholly unregistered pages. This is a
 * virtual-memory policy, not an allocation or a second capacity ledger.
 * TransferEngine retains the original storage and registration lifetime.
 */
#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <system_error>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace llaminar2
{
    /**
     * @brief Checked half-open page span containing one positive registration.
     *
     * Partial first/last pages belong to the span because native registration
     * pins those same pages. The size describes VMA policy only and never adds
     * a physical-memory charge. Only forRange() constructs admitted geometry.
     */
    class HostRegistrationPageRange
    {
    public:
        /**
         * @brief Round outward to every touched base page without wrapping.
         * @param address Nonzero first logical registration byte.
         * @param bytes Positive logical registration extent.
         * @param page_bytes Runtime base-page size, a positive power of two.
         * @return Complete page-aligned span with representable exclusive end.
         * @throws std::invalid_argument For null/empty or invalid page geometry.
         * @throws std::overflow_error For logical, rounded-end or size overflow.
         */
        [[nodiscard]] static HostRegistrationPageRange forRange(
            std::uintptr_t address, std::size_t bytes, std::size_t page_bytes)
        {
            constexpr auto maximum = std::numeric_limits<std::uintptr_t>::max();
            if (!address || !bytes || !page_bytes || page_bytes > maximum ||
                (page_bytes & (page_bytes - 1u)) != 0u)
                throw std::invalid_argument("Host registration requires a positive range and power-of-two page size");
            if (bytes - 1u > maximum - address)
                throw std::overflow_error("Host registration address range overflows");
            const auto page = static_cast<std::uintptr_t>(page_bytes);
            const auto mask = ~(page - 1u);
            const auto first = address & mask;
            const auto last = (address + bytes - 1u) & mask;
            // The logical last byte may fit while its containing page's
            // exclusive end does not. Never pass a wrapped extent to madvise.
            if (last > maximum - page)
                throw std::overflow_error("Host registration rounded page end overflows");
            const auto extent = last + page - first;
            if (extent > std::numeric_limits<std::size_t>::max())
                throw std::overflow_error("Host registration page span size overflows");
            return HostRegistrationPageRange(first, static_cast<std::size_t>(extent), page_bytes);
        }

        /** @return First containing page's address, including a partial first page. */
        [[nodiscard]] std::uintptr_t beginAddress() const noexcept { return begin_; }
        /** @return Representable exclusive end after the last containing page. */
        [[nodiscard]] std::uintptr_t endAddress() const noexcept { return begin_ + bytes_; }
        /** @return Complete advice extent, not a physical capacity charge. */
        [[nodiscard]] std::size_t sizeBytes() const noexcept { return bytes_; }
        /** @return Base-page granularity used to authenticate this span. */
        [[nodiscard]] std::size_t pageBytes() const noexcept { return page_bytes_; }

        /**
         * @brief Exclude the entire caller-owned page span from THP before pinning.
         * @param pointer Stable caller-owned host storage, never a GPU alias.
         * @param bytes Exact positive logical registration extent.
         * @throws std::system_error If Linux cannot establish the full policy.
         * @throws std::exception For invalid/overflowing geometry or page discovery.
         *
         * One advice changes neither bytes nor permissions and introduces no
         * allocation, transfer or GPU synchronization. Keep the policy after
         * unregister: a page may still contain another registration or a later
         * capture may reuse it. Re-enabling THP during unpin would invalidate
         * that owner's lifetime. Only the storage owner may later reuse/unmap
         * its pages; wholly adjacent pages remain eligible throughout.
         */
        static void prepare(void *pointer, std::size_t bytes)
        {
#ifdef __linux__
            const long page_bytes = ::sysconf(_SC_PAGESIZE);
            if (page_bytes <= 0)
                throw std::runtime_error("Cannot resolve host registration page size");
            const auto range = forRange(reinterpret_cast<std::uintptr_t>(pointer),
                bytes, static_cast<std::size_t>(page_bytes));
            if (::madvise(reinterpret_cast<void *>(range.beginAddress()),
                          range.sizeBytes(), MADV_NOHUGEPAGE) != 0)
                throw std::system_error(errno, std::generic_category(),
                    "Cannot exclude the full host registration span from huge-page coalescing");
#else
            (void)pointer;
            (void)bytes;
            throw std::runtime_error("Host registration page policy requires Linux");
#endif
        }

    private:
        /** @brief Retain only complete geometry checked by forRange(). */
        HostRegistrationPageRange(std::uintptr_t begin, std::size_t bytes,
                                  std::size_t page_bytes) noexcept
            : begin_(begin), bytes_(bytes), page_bytes_(page_bytes) {}

        std::uintptr_t begin_;
        std::size_t bytes_;
        std::size_t page_bytes_;
    };
}
