/**
 * @file CapturedCUDAArrivalOwnerInventory.h
 * @brief Additional opaque CUDA graph-pool geometry for existing retained owners.
 *
 * Group only physical executables with the same role and site count. Each
 * owner receives its own upload charge and rounding; recording-only source
 * fragments do not increment the population of executable owners.
 */
#pragma once

#include "backends/cuda/CUDALiveLaunchMemoryContract.h"

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace llaminar2
{
    /** @brief Capture role of a physical owner that can lower arrival sites. */
    enum class CUDAArrivalOwnerRole
    {
        Prefill,
        PrefixRehydration,
        PrefixDecode,
        RestoredPrefixBridge,
        MaintenanceCopy,
        StandaloneCadence,
        GenerationParent,
        PipelineFollower,
    };

    /** @brief Complete affected subset of an already declared native graph family. */
    class CapturedCUDAArrivalOwnerInventory final
    {
    public:
        struct Population
        {
            CUDAArrivalOwnerRole role;
            std::size_t owners;
            std::size_t sites_per_owner;
        };

        /** @brief Add physical owners, omitting owners that instantiate without live arrivals. */
        void add(Population population)
        {
            if (population.owners != 0u && population.sites_per_owner != 0u)
                populations_.push_back(population);
        }

        /** @return The retained role and site populations priced by this family. */
        [[nodiscard]] std::span<const Population> populations() const noexcept { return populations_; }

        /** @return Number of physical owners to which the incremental quote applies. */
        [[nodiscard]] std::size_t ownerCount() const
        {
            std::size_t owners = 0;
            for (const auto &population : populations_)
            {
                if (population.owners > (std::numeric_limits<std::size_t>::max() - owners))
                    throw std::overflow_error("CUDA arrival executable count overflows size_t");
                owners += population.owners;
            }
            return owners;
        }

        /** @return Physical maintenance executables outside ordinary/MTP/model forward slots. */
        [[nodiscard]] std::size_t standaloneMaintenanceExecutableCount() const
        {
            std::size_t owners = 0;
            for (const auto &population : populations_)
            {
                if (population.role != CUDAArrivalOwnerRole::MaintenanceCopy &&
                    population.role != CUDAArrivalOwnerRole::StandaloneCadence)
                    continue;
                if (population.owners > std::numeric_limits<std::size_t>::max() - owners)
                    throw std::overflow_error("CUDA standalone maintenance executable count overflows size_t");
                owners += population.owners;
            }
            return owners;
        }

        /** @return Additional family pool bytes, with the same per-owner quote as native lowering. */
        [[nodiscard]] std::size_t reservationBytes() const
        {
            std::size_t bytes = 0;
            for (const auto &population : populations_)
            {
                const auto per_owner = quoteCUDALiveLaunchReservationBytes(population.sites_per_owner);
                if (population.owners > (std::numeric_limits<std::size_t>::max() - bytes) / per_owner)
                    throw std::overflow_error("CUDA arrival native graph family reservation overflows size_t");
                bytes += per_owner * population.owners;
            }
            return bytes;
        }

    private:
        std::vector<Population> populations_;
    };
} // namespace llaminar2
