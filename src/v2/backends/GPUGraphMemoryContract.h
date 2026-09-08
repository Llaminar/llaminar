/**
 * @file GPUGraphMemoryContract.h
 * @brief Shared admission and runtime-attestation policy for native GPU graphs.
 *
 * CUDA and HIP retain native graph storage in backend-owned pools. A memory
 * sample around one instantiation observes growth of that shared pool, not
 * bytes attributable to the executable being instantiated: one graph can grow
 * the pool for several later graphs, which then report zero. Planning must
 * therefore reserve the complete retained family, and runtime certification
 * must never compare one pool-growth event with one executable slot.
 */

#pragma once

#include "DeviceId.h"

#include <cstddef>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Certified physical shape, independent of a graph's semantic role. */
    enum class GPUGraphExecutableClass
    {
        General, ///< Complete forwards, nested parents, and unbounded auxiliaries.
        BoundedFlatHelper, ///< At most 64 flat kernel/copy/memset nodes; no control or child nodes.
    };

    /**
     * @brief Canonical opaque-driver pool contract for native graph families.
     *
     * The values are backend certification units, not anonymous safety margins.
     * CUDA graph-pool growth has been observed in 2 MiB slabs. The canonical
     * 3,050-node production-checkpoint certificate consumes eleven slabs on a
     * cold CUDA process; HIP graphs from one through 3,050 nodes consume one
     * two-MiB slab. A separate 128-owner/64-kernel flat-helper certificate fits
     * two MiB per owner on both backends (including cold and warm lifetimes).
     * The declaration must pass a native node-kind/count check before receiving
     * this smaller class; sidecars and control graphs are not small by name.
     * @ref PhysicalMemoryAuthority reserves the class-specific complete family
     * before model placement. Runtime capture records
     * individual pool-growth deltas as evidence; a complete-family certificate
     * compares their aggregate with the family reservation. A future driver or
     * larger family must extend that certificate instead of drawing from model
     * capacity owned by another BOM line.
     */
    class GPUGraphMemoryContract final
    {
    public:
        /** Native-node ceiling certified by the complete small-helper family. */
        static constexpr std::size_t kBoundedFlatHelperMaxNodes = 64u;
        /** Both backends certify 128 retained 64-node helpers within this unit. */
        static constexpr std::size_t kBoundedFlatHelperReservationBytes =
            2ULL * 1024ULL * 1024ULL;

        /**
         * @brief Validate a typed class and return its bounded-shape obligation.
         * @throws std::invalid_argument for an unknown enum value.
         */
        [[nodiscard]] static bool requiresBoundedFlatShape(
            GPUGraphExecutableClass executable_class)
        {
            switch (executable_class)
            {
            case GPUGraphExecutableClass::General:
                return false;
            case GPUGraphExecutableClass::BoundedFlatHelper:
                return true;
            }
            throw std::invalid_argument("Unknown GPU graph executable memory class");
        }

        /** CUDA driver-pool reservation unit for one retained graph slot. */
        static constexpr std::size_t kCUDAExecutableReservationBytes =
            22ULL * 1024ULL * 1024ULL;

        /** HIP driver-pool reservation unit for one retained graph slot. */
        static constexpr std::size_t kROCmExecutableReservationBytes =
            2ULL * 1024ULL * 1024ULL;

        /**
         * @brief Return the reservation owned by one native executable.
         * @param device Exact GPU backend/device identity.
         * @param executable_class Declared and capture-enforced physical shape.
         * @throws std::invalid_argument when @p device is not CUDA or ROCm.
         */
        [[nodiscard]] static std::size_t
        reservationBytesPerExecutable(
            DeviceId device,
            GPUGraphExecutableClass executable_class =
                GPUGraphExecutableClass::General)
        {
            const bool bounded = requiresBoundedFlatShape(executable_class);
            if (!device.is_cuda() && !device.is_rocm())
                throw std::invalid_argument(
                    "Native graph memory contracts require a CUDA or ROCm device");
            if (bounded)
                return kBoundedFlatHelperReservationBytes;
            if (device.is_cuda())
                return kCUDAExecutableReservationBytes;
            if (device.is_rocm())
                return kROCmExecutableReservationBytes;
            throw std::invalid_argument(
                "Native graph memory contracts require a CUDA or ROCm device");
        }

        /**
         * @brief Verify aggregate pool growth against one family admission.
         * @param device Exact GPU backend/device identity.
         * @param observed_pool_growth_bytes Sum of positive setup-only
         *        free-memory deltas across the complete materialized family.
         * @param admitted_family_bytes Bytes reserved for that exact family by
         *        the physical-memory authority.
         * @return True when the complete runtime observation fits admission.
         *
         * A caller must not pass a single executable's slot reservation as
         * @p admitted_family_bytes. CUDA may charge a large shared-pool growth
         * event to the first executable and zero to several siblings.
         */
        [[nodiscard]] static bool acceptsFamilyObservation(
            DeviceId device,
            std::size_t observed_pool_growth_bytes,
            std::size_t admitted_family_bytes)
        {
            (void)reservationBytesPerExecutable(device);
            return observed_pool_growth_bytes <= admitted_family_bytes;
        }
    };
} // namespace llaminar2
