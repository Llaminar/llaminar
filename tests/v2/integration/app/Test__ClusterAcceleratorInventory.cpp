/**
 * @file Test__ClusterAcceleratorInventory.cpp
 * @brief Model-free public-command discovery across CPU and GPU NUMA localities.
 *
 * Use the same command/MPI entrypoint as plan and describe, not a preinitialized
 * serving manager. Every rank must publish all driver-visible accelerators even
 * when its CPU affinity belongs to another socket. Repeated physical visibility
 * must not multiply cluster capacity, and repeated reads must reuse the exact
 * context-owned publication. The registered CUDA and ROCm lanes exercise real
 * drivers separately; no model, synthetic capacity or inference is involved.
 *
 * The custom entry owns one lexical native GPU lifetime without probing a
 * device. Native workers/caches retire before process exit and MPI finalization.
 */
#include "backends/GPUDeviceProcessLifetime.h"
#include "app/commands/CommandMPI.h"
#include "config/OrchestrationConfig.h"
#include "utils/MPIBootstrap.h"
#include "utils/DebugEnv.h"
#include "planning/RankHardwareOwnership.h"
#include <string_view>
#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace llaminar2;

/** @brief Enter production command discovery and collectively check its observation. */
int main(int argc, char **argv)
{
    llaminar2::GPUDeviceProcessLifetime gpu_process_lifetime;

    // The extra registered case exercises the real GPU-targeted self-launch,
    // not a test-authored taskset or an already MPI-bound two-rank discovery.
    const bool single_rank = std::any_of(argv + 1, argv + argc,
        [](const char *arg) { return std::string_view(arg) == "--single-rank-gpu-bootstrap"; });
    if (!single_rank && !MPIBootstrap::detectMPIEnvironment().is_mpi_process)
        throw std::runtime_error("Accelerator inventory proof requires its registered MPI launch");
    OrchestrationConfig request;
    if (single_rank)
    {
        const auto &startup = debugEnv().backend_startup;
        if (startup.cudaEnabled() == startup.rocmEnabled())
            throw std::runtime_error("Single-rank GPU bootstrap proof requires exactly one enabled vendor");
        request.device_for_this_rank = startup.cudaEnabled()
            ? GlobalDeviceAddress::cuda(0) : GlobalDeviceAddress::rocm(0);
        request.mpi_procs = 1;
    }
    auto [session, early_exit] = CommandMPI::bootstrap({
        .subcommand = "plan", .argc = argc, .argv = argv,
        .no_mpi_bootstrap = false, .hostfile = {}, .request = &request});
    if (early_exit) return *early_exit;
    const auto *hardware = DeviceManager::instance().hardware();
    if (!hardware) throw std::runtime_error("Command discovery published no hardware observation");
    const auto &inventory = session.inventory();
    const auto &local = inventory.ranks.at(session.context()->rank());
    const auto expected = hardware->cuda_device_count() + hardware->rocm_device_count();
    int passed = expected > 0 && local.cpu.numa_node >= 0 &&
        local.gpus.size() == static_cast<std::size_t>(expected);
    for (const auto *devices : {&hardware->cuda_devices, &hardware->rocm_devices})
        for (const auto &device : *devices)
            passed &= std::any_of(local.gpus.begin(), local.gpus.end(), [&](const auto &gpu) {
                return gpu.uuid == device.uuid && gpu.local_device_id == device.device_id &&
                    gpu.numa_node == device.numa_node && gpu.memory_bytes == device.total_memory_bytes;
            });
    if (single_rank)
    {
        passed &= inventory.world_size == 1;
        const auto selected = request.device_for_this_rank->toLocalDeviceId();
        const auto found = std::find_if(local.gpus.begin(), local.gpus.end(), [&](const auto &gpu) {
            return gpu.type == selected.type && gpu.local_device_id == selected.ordinal;
        });
        passed &= found != local.gpus.end() && rankOwnsGPUDevice(local, *found);
    }
    // Every test process observes the same real physical host. UUID union,
    // rather than summing rank records, is the cluster hardware identity rule.
    passed &= inventory.node_count == 1 && inventory.total_gpus == expected;
    const auto first = session.context()->clusterInventory();
    const auto second = session.context()->clusterInventory();
    passed &= first == second && first.get() == &inventory && DeviceManager::instance().hardware() == hardware;
    MPI_Allreduce(MPI_IN_PLACE, &passed, 1, MPI_INT, MPI_MIN, session.communicator());
    if (!passed)
        std::cerr << "Inventory dropped a visible GPU: rank=" << local.rank
                  << " cpu_numa=" << local.cpu.numa_node
                  << " driver_gpus=" << expected << " published_gpus=" << local.gpus.size() << '\n';
    return passed ? 0 : 1;
}
