/**
 * @file MoEOverlayDynamicPolicyTestMemory.h
 * @brief Exact component-payload PMA for original native controller fixtures.
 *
 * This fixture budget admits only the new persistent policy scratch against
 * the actual already-selected backend device. It cannot create a native
 * context, select another participant or change the original controller roster.
 */
#pragma once
#include "execution/moe/MoEOverlayDynamicPolicyWorkspaceOwner.h"
#include <mpi.h>

namespace llaminar2::test
{
    /** @brief Admit one exact scratch payload on the fixture's existing owner.
     * @param device Original native authority or CPU replay owner. @param shape Original captured shape.
     * @param stream Exact existing native consumer stream, or null for CPU.
     * @return Persistent owner with its sole PMA allocation lease.
     * @throws std::logic_error For absent original backend/rank or insufficient observed capacity. */
    inline std::unique_ptr<MoEOverlayDynamicPolicyWorkspaceOwner> admitDynamicPolicyWorkspaceForTest(
        DeviceId device, MoEOverlayDynamicPolicyShape shape, void *stream)
    {
        const auto bytes = MoEOverlayDynamicPolicyWorkspaceOwner::allocationBytes(shape);
        std::size_t total = bytes, available = bytes;
        if (device.is_gpu())
        {
            auto *backend = getBackendFor(device);
            if (!backend) throw std::logic_error("Policy fixture has no original native backend");
            total = backend->deviceMemoryTotal(device.toKernelDeviceIndex());
            available = backend->deviceMemoryFree(device.toKernelDeviceIndex());
        }
        if (bytes > total || bytes > available)
            throw std::logic_error("Policy fixture cannot admit its actual scratch allocation");
        int initialized = 0, rank = 0;
        if (MPI_Initialized(&initialized) != MPI_SUCCESS ||
            (initialized && MPI_Comm_rank(MPI_COMM_WORLD, &rank) != MPI_SUCCESS))
            throw std::logic_error("Policy fixture cannot resolve its original MPI rank");
        PhysicalMemoryPlanBuilder budget;
        budget.add({rank, device, total, available}, PhysicalMemoryOwner::ExecutionWorkspace, bytes);
        auto memory = std::make_shared<PhysicalMemoryAuthority>(
            std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()), rank);
        return std::make_unique<MoEOverlayDynamicPolicyWorkspaceOwner>(memory, device, shape, stream);
    }
}
