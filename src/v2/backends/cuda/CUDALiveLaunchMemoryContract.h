/**
 * @file CUDALiveLaunchMemoryContract.h
 * @brief Incremental pool geometry for CUDA final owners with live arrival launches.
 */
#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    /**
     * @brief Quote the incremental device pool for one uploaded CUDA owner.
     * @param sites Final imports, not reusable source recordings or slot capacity.
     * @return Zero for an ordinary executable; otherwise the upload and update-node extent.
     * @throws std::overflow_error If the aligned reservation cannot be represented.
     *
     * The owner uploads once. Each import binds one fixed-size update node and
     * its control edge, irrespective of the shape or live arrivals. Add the
     * per-import bytes before rounding the whole owner to a CUDA pool slab.
     * CUDAMoEArrivalFinalOwnersFitNativePool checks the affine bound against
     * complete matched families on discrete devices. Integrated devices consume
     * the same quote; their system-wide deltas are diagnostics, not pool proof.
     */
    [[nodiscard]] inline std::size_t quoteCUDALiveLaunchReservationBytes(std::size_t sites)
    {
        if (sites == 0u)
            return 0u;
        constexpr std::size_t slab = 2u * 1024u * 1024u;
        constexpr std::size_t upload = 3'478'016u;
        constexpr std::size_t per_site = 22u * 1024u;
        if (sites > (std::numeric_limits<std::size_t>::max() - upload - (slab - 1u)) / per_site)
            throw std::overflow_error("CUDA live-launch graph-owner reservation overflows size_t");
        return (upload + per_site * sites + (slab - 1u)) / slab * slab;
    }
} // namespace llaminar2
