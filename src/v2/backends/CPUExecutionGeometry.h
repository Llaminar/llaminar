/**
 * @file CPUExecutionGeometry.h
 * @brief Immutable CPU kernel-policy observations carried by cluster inventory.
 *
 * These values describe execution geometry, not physical host/rank identity or
 * a memory balance. The observing process and its kernels use the same cached
 * record. Remote planning consumes the published record and must not substitute
 * its own CPUID, ISA or worker configuration. PhysicalMemoryAuthority alone
 * admits the workspace requirements derived from it.
 */
#pragma once
#include <cstdint>

namespace llaminar2
{
    /** @brief Exact private/shared cache dimensions consumed by CPU tile policy. */
    struct CPUCacheGeometry
    {
        std::uint64_t private_l2_bytes = 0;
        std::uint64_t shared_l3_bytes = 0;
        std::uint32_t private_l2_ways = 0;
        std::uint32_t shared_l3_ways = 0;
        /** @return Whether the observation contains every required cache dimension. */
        constexpr bool isValid() const noexcept
        {
            return private_l2_bytes && shared_l3_bytes && private_l2_ways && shared_l3_ways;
        }
        /** @return Exact observation equality, without inferring physical locality. */
        bool operator==(const CPUCacheGeometry &) const = default;
    };

    /** @brief Process-selected CPU execution geometry; workers remain rank-plan owned. */
    struct CPUExecutionGeometry
    {
        CPUCacheGeometry cache;
        std::uint32_t maximum_native_row_tile = 0; ///< Two for AVX2, four for AVX512.
        /** @return Whether NativeVNNI can price and execute this exact observation. */
        constexpr bool isValid() const noexcept
        {
            return cache.isValid() && (maximum_native_row_tile == 2 || maximum_native_row_tile == 4);
        }
        /** @return Exact policy observation equality, not a physical-node comparison. */
        bool operator==(const CPUExecutionGeometry &) const = default;
        /**
         * @brief Read the observing process's one immutable CPU policy record.
         * @return Real CPUID cache dimensions and selected ISA; unknown fields stay zero.
         *
         * Detection is cached once, before execution. Unsupported/incomplete
         * NativeVNNI geometry is rejected by its requirement/dispatch boundary;
         * discovery must not fabricate cache sizes or exclude floating work.
         */
        static const CPUExecutionGeometry &local();
    };
}
