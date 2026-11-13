/**
 * @file MoELocalTPMaintenanceDomain.h
 * @brief Exact maintenance membership borrowed from the original LocalTP owner.
 *
 * A LocalTP context's resolved scope establishes one original rank's ownership;
 * cold planning and stage birth authenticate its actual endpoints. Graph
 * workers must borrow that authority instead of guessing locality from an
 * address's display hostname or performing new inventory/MPI discovery.
 * Fully qualified hostnames and sparse native ordinals remain unchanged.
 */
#pragma once
#include "collective/ILocalTPContext.h"

namespace llaminar2
{
    /**
     * @brief Check the actual captured native family retained by one LocalTP context.
     * @param context Original stage-local collective owner, already admitted and initialized.
     * @param device Exact graph participant's native device.
     * @param device_index Participant index in the context's original ordered membership.
     * @param maximum_participants Controller ABI's admitted participant bound.
     * @return True only for a homogeneous NCCL/CUDA or RCCL/ROCm captured family with exact membership.
     * The native allgather capability proves graph capture support; a backend
     * name or global MPI world size cannot grant it. This query is setup-only
     * metadata: it creates no context, collective, worker, allocation or transfer.
     */
    inline bool isHomogeneousGPULocalTPMaintenanceDomain(const ILocalTPContext &context,
        DeviceId device, int device_index, int maximum_participants)
    {
        const int degree = context.degree();
        if (context.scope() != TPScope::RANK_LOCAL || (!device.is_cuda() && !device.is_rocm()) || degree <= 1 ||
            degree > maximum_participants || device_index < 0 || device_index >= degree ||
            !context.supportsRawAllgatherOnStreamGraphCapture()) return false;
        const auto backend = context.backend();
        if ((device.is_cuda() && backend != CollectiveBackendType::NCCL) ||
            (device.is_rocm() && backend != CollectiveBackendType::RCCL)) return false;
        const auto &participants = context.devices();
        if (participants.size() != static_cast<size_t>(degree)) return false;
        for (size_t index = 0; index < participants.size(); ++index)
        {
            const auto &participant = participants[index];
            if (!participant.isGPU() || participant.device_type != device.type ||
                !participant.toLocalDeviceId().is_gpu()) return false;
            // Native ordinals address this one process's devices. Different
            // display hostname spellings cannot make a duplicated GPU distinct.
            for (size_t prior = 0; prior < index; ++prior)
                if (participants[prior].toLocalDeviceId() == participant.toLocalDeviceId()) return false;
        }
        return participants[static_cast<size_t>(device_index)].toLocalDeviceId() == device;
    }
}
