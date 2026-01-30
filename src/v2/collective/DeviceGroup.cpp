/**
 * @file DeviceGroup.cpp
 * @brief DeviceGroup and DeviceGroupFactory implementations
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include "DeviceGroup.h"
#include "../execution/mpi_orchestration/DeviceInventory.h"

namespace llaminar2
{

    // =========================================================================
    // DeviceGroupFactory implementations
    // =========================================================================

    DeviceGroup DeviceGroupFactory::createLocalCUDAGroup(
        const RankInventory &inventory,
        int local_device_idx)
    {
        DeviceGroupBuilder builder;
        builder.setName("cuda_gpus_rank" + std::to_string(inventory.rank))
            .setScope(CollectiveScope::LOCAL)
            .setLocalRank(local_device_idx);

        for (const auto &gpu : inventory.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
            {
                builder.addDevice(DeviceId::cuda(gpu.local_device_id));
            }
        }

        return builder.build();
    }

    DeviceGroup DeviceGroupFactory::createLocalROCmGroup(
        const RankInventory &inventory,
        int local_device_idx)
    {
        DeviceGroupBuilder builder;
        builder.setName("rocm_gpus_rank" + std::to_string(inventory.rank))
            .setScope(CollectiveScope::LOCAL)
            .setLocalRank(local_device_idx);

        for (const auto &gpu : inventory.gpus)
        {
            if (gpu.type == DeviceType::ROCm)
            {
                builder.addDevice(DeviceId::rocm(gpu.local_device_id));
            }
        }

        return builder.build();
    }

    DeviceGroup DeviceGroupFactory::createLocalAllDevicesGroup(
        const RankInventory &inventory,
        int local_device_idx)
    {
        DeviceGroupBuilder builder;
        builder.setName("all_devices_rank" + std::to_string(inventory.rank))
            .setScope(CollectiveScope::LOCAL)
            .setLocalRank(local_device_idx);

        // Add all GPUs
        for (const auto &gpu : inventory.gpus)
        {
            if (gpu.type == DeviceType::CUDA)
            {
                builder.addDevice(DeviceId::cuda(gpu.local_device_id));
            }
            else if (gpu.type == DeviceType::ROCm)
            {
                builder.addDevice(DeviceId::rocm(gpu.local_device_id));
            }
        }

        // Add CPU
        builder.addDevice(DeviceId::cpu());

        return builder.build();
    }

    DeviceGroup DeviceGroupFactory::createLocalCPUGroup(int rank)
    {
        DeviceGroupBuilder builder;
        builder.setName("cpu_rank" + std::to_string(rank))
            .setScope(CollectiveScope::LOCAL)
            .setLocalRank(0)
            .addDevice(DeviceId::cpu());

        return builder.build();
    }

    DeviceGroup DeviceGroupFactory::createGlobalGroup(
        const ClusterInventory &inventory,
        int rank,
        DeviceId local_device)
    {
        DeviceGroupBuilder builder;
        builder.setName("global_group")
            .setScope(CollectiveScope::GLOBAL)
            .setLocalRank(rank);

        // Add one device per rank (placeholder - actual implementation
        // would iterate over cluster inventory)
        for (int r = 0; r < inventory.world_size; ++r)
        {
            if (r == rank)
            {
                builder.addDevice(local_device);
            }
            else
            {
                // For other ranks, add a placeholder device
                // Real implementation would query ClusterInventory
                builder.addDevice(DeviceId::cpu());
            }
        }

        return builder.build();
    }

    DeviceGroup DeviceGroupFactory::createSubgroup(
        const DeviceGroup &parent,
        DeviceType type,
        int local_device_idx)
    {
        DeviceGroupBuilder builder;
        builder.setName(parent.name + "_subgroup_" + deviceTypeToString(type))
            .setScope(parent.scope)
            .setLocalRank(local_device_idx);

        for (const auto &device : parent.devices)
        {
            if (device.type == type)
            {
                builder.addDevice(device);
            }
        }

        return builder.build();
    }

    std::vector<DeviceGroup> DeviceGroupFactory::partitionByType(const DeviceGroup &group)
    {
        std::vector<DeviceGroup> subgroups;

        // Check for each device type
        if (group.cuda_count > 0)
        {
            subgroups.push_back(createSubgroup(group, DeviceType::CUDA, 0));
        }
        if (group.rocm_count > 0)
        {
            subgroups.push_back(createSubgroup(group, DeviceType::ROCm, 0));
        }
        if (group.cpu_count > 0)
        {
            subgroups.push_back(createSubgroup(group, DeviceType::CPU, 0));
        }

        return subgroups;
    }

} // namespace llaminar2
