/**
 * @file HostPageMappingTestUtils.h
 * @brief Owned Linux host pages and VMA-policy evidence for registration tests.
 *
 * The fixture deliberately lets a registered payload share a huge-page-sized
 * mapping with an unrelated reclaimable tail. Production registration must
 * isolate its edges without losing bytes or disabling interior huge pages.
 * Reading smaps checks installed policy without relying on memory fragmentation
 * or on the kernel printing its rate-limited workqueue warning.
 */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

namespace llaminar2::test
{
    /** @brief Test-only owner of a huge-page-aligned 32 MiB anonymous range. */
    class HostPageMapping
    {
    public:
        static constexpr std::size_t huge_bytes = 2u * 1024u * 1024u;
        static constexpr std::size_t payload_bytes = 32u * 1024u * 1024u;
        static constexpr std::size_t live_bytes = 31u * 1024u * 1024u;

        /** @brief Request huge pages before first touch, without requiring their availability. */
        HostPageMapping()
        {
            base_ = ::mmap(nullptr, reservation_bytes_, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base_ == MAP_FAILED)
                throw std::runtime_error("Host-page test mapping failed");
            const auto address = (reinterpret_cast<std::uintptr_t>(base_) + huge_bytes - 1u) &
                                 ~(static_cast<std::uintptr_t>(huge_bytes) - 1u);
            data_ = reinterpret_cast<unsigned char *>(address);
            if (::madvise(data_, payload_bytes, MADV_HUGEPAGE) != 0)
            {
                ::munmap(base_, reservation_bytes_);
                throw std::runtime_error("Host-page test huge-page policy failed");
            }
            std::memset(data_, 0x53, payload_bytes);
        }

        /** @brief Retire the mapping after every native registration/graph owner. */
        ~HostPageMapping() { (void)::munmap(base_, reservation_bytes_); }
        HostPageMapping(const HostPageMapping &) = delete;
        HostPageMapping &operator=(const HostPageMapping &) = delete;

        /** @return The stable first payload byte, not necessarily the reservation base. */
        [[nodiscard]] unsigned char *data() const noexcept { return data_; }

        /** @return Exact runtime base-page bytes used by this Linux fixture. */
        [[nodiscard]] static std::size_t pageBytes()
        {
            const auto size = ::sysconf(_SC_PAGESIZE);
            if (size <= 0) throw std::runtime_error("Host-page test cannot resolve page size");
            return static_cast<std::size_t>(size);
        }

        /** @brief Reclaim only the unregistered tail's first page, never a live byte. */
        void reclaimNeighbour() const
        {
            if (::madvise(data_ + live_bytes, pageBytes(), MADV_DONTNEED) != 0)
                throw std::runtime_error("Host-page test neighbour reclamation failed");
        }

        /** @return Whether the byte's current VMA contains the requested VmFlags token. */
        [[nodiscard]] static bool hasFlag(const void *pointer, const char *flag)
        {
            std::ifstream maps("/proc/self/smaps");
            if (!maps) throw std::runtime_error("Host-page test cannot read smaps");
            const auto address = reinterpret_cast<std::uintptr_t>(pointer);
            bool selected = false;
            for (std::string line; std::getline(maps, line);)
            {
                unsigned long start = 0, end = 0;
                if (std::sscanf(line.c_str(), "%lx-%lx", &start, &end) == 2)
                    selected = address >= start && address < end;
                else if (selected && line.starts_with("VmFlags:"))
                    return (line + " ").find(std::string(" ") + flag + " ") != std::string::npos;
            }
            throw std::runtime_error("Host-page test address has no VMA policy");
        }

    private:
        static constexpr std::size_t reservation_bytes_ = payload_bytes + huge_bytes;
        void *base_ = nullptr;
        unsigned char *data_ = nullptr;
    };
}
