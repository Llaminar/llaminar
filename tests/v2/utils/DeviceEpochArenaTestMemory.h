/**
 * @file DeviceEpochArenaTestMemory.h
 * @brief Declare exact project-owned epoch storage for original component fixtures.
 *
 * This creates only a PMA budget for the actual epoch allocation. It cannot
 * create a native context/worker, source permit, model identity or placement.
 * Existing native fixtures retain their original backend/context and streams.
 */
#pragma once
#include "execution/moe/DeviceMoEOverlayEpochArena.h"
#include "planning/PhysicalMemoryAuthority.h"
#include "backends/BackendManager.h"
#include "backends/IBackend.h"
#include <mpi.h>
#include <stdexcept>
namespace llaminar2::test {
/** @brief Supply only a missing component-payload budget, preserving any actual supplied source/PMA.
 * @param config Original device, initial bank and request-slot declaration.
 * @return Same declaration with exact packed bytes under ExecutionWorkspace.
 * @throws std::logic_error For absent native backend or insufficient actual device memory.
 */
inline DeviceMoEOverlayEpochArena::Config admitEpochArenaForTest(DeviceMoEOverlayEpochArena::Config config) {
    if(config.memory_authority)return config;
    const auto bytes=DeviceMoEOverlayEpochArena::allocationBytesForSlots(config.request_slot_capacity);
    std::size_t total=bytes,available=bytes;
    if(config.device_id.is_gpu()) {
        auto* backend=getBackendFor(config.device_id);
        if(!backend)throw std::logic_error("Epoch component fixture has no original native backend");
        total=backend->deviceMemoryTotal(config.device_id.toKernelDeviceIndex());
        available=backend->deviceMemoryFree(config.device_id.toKernelDeviceIndex());
        if(total<bytes || available<bytes)throw std::logic_error("Epoch component fixture cannot admit its actual packed storage");
    }
    int initialized=0,rank=0;
    if(MPI_Initialized(&initialized)!=MPI_SUCCESS ||
       (initialized && MPI_Comm_rank(MPI_COMM_WORLD,&rank)!=MPI_SUCCESS))
        throw std::logic_error("Epoch component fixture cannot resolve its original MPI rank");
    PhysicalMemoryPlanBuilder budget;
    budget.add({rank,config.device_id,total,available},PhysicalMemoryOwner::ExecutionWorkspace,bytes);
    config.memory_authority=std::make_shared<PhysicalMemoryAuthority>(
        std::make_shared<PhysicalMemoryPlanAdmissionCertificate>(budget.build()),rank);
    return config;
}
}
