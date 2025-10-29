/**
 * @file WeightManager.cpp
 * @brief Weight distribution and caching implementation
 * @author David Sanftenberg
 */

#include "WeightManager.h"
#include "../utils/Logger.h"
#include <iostream>

namespace llaminar2
{

    WeightManager::WeightManager(ModelLoader &loader,
                                 std::shared_ptr<MPIContext> mpi_ctx,
                                 std::shared_ptr<WeightPlacementMap> placement_map,
                                 WeightDistributionStrategy strategy)
        : loader_(loader), mpi_ctx_(mpi_ctx), placement_map_(placement_map), strategy_(strategy)
    {
        int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;

        if (rank == 0)
        {
            std::string strategy_name;
            switch (strategy_)
            {
            case WeightDistributionStrategy::REPLICATED:
                strategy_name = "REPLICATED (full copy per rank)";
                break;
            case WeightDistributionStrategy::SHARDED:
                strategy_name = "SHARDED (partitioned across ranks)";
                break;
            case WeightDistributionStrategy::INTERLEAVED:
                strategy_name = "INTERLEAVED (NUMA-aware global)";
                break;
            }
            LOG_INFO("[WeightManager] Initialized with strategy: " << strategy_name);
        }
    }

    std::shared_ptr<TensorBase> WeightManager::getWeight(const std::string &name, int device_idx, int layer_idx)
    {
        // Check cache first
        auto it = cache_.find(name);
        if (it != cache_.end())
        {
            return it->second;
        }

        // Determine device from placement map if not explicitly provided
        int target_device = device_idx;
        if (target_device < 0 && placement_map_)
        {
            target_device = placement_map_->getDeviceForWeight(name, layer_idx);
        }
        if (target_device < 0)
        {
            target_device = 0; // Default to device 0
        }

        // Load based on strategy
        std::shared_ptr<TensorBase> tensor;

        switch (strategy_)
        {
        case WeightDistributionStrategy::REPLICATED:
            tensor = getReplicatedWeight(name, target_device);
            break;

        case WeightDistributionStrategy::SHARDED:
            tensor = getShardedWeight(name, target_device);
            break;

        case WeightDistributionStrategy::INTERLEAVED:
            tensor = getInterleavedWeight(name, target_device);
            break;

        default:
            LOG_ERROR("[WeightManager] Unknown strategy: " << static_cast<int>(strategy_));
            return nullptr;
        }

        // Cache the loaded tensor
        if (tensor)
        {
            cache_[name] = tensor;
        }

        return tensor;
    }

    std::shared_ptr<TensorBase> WeightManager::getReplicatedWeight(const std::string &name, int device_idx)
    {
        // Phase 1: Simple replication - each rank loads independently
        // No MPI coordination needed

        auto tensor = loader_.loadTensor(name, device_idx);
        if (!tensor)
        {
            int rank = mpi_ctx_ ? mpi_ctx_->rank() : 0;
            LOG_ERROR("[WeightManager] Rank " << rank << " failed to load: " << name);
            return nullptr;
        }

        return tensor;
    }

    std::shared_ptr<TensorBase> WeightManager::getShardedWeight(const std::string &name, int device_idx)
    {
        // Phase 2: Sharded strategy (not yet implemented)
        // TODO: Implement column-wise sharding for linear layer weights
        //
        // For weight matrix [out_dim, in_dim]:
        // 1. Get tensor shape from loader
        // 2. Calculate this rank's slice: out_dim / world_size
        // 3. Load only this rank's slice from GGUF
        // 4. Kernels will need to do Allreduce after matmul
        //
        // Example:
        //   int rank = mpi_ctx_->rank();
        //   int world_size = mpi_ctx_->world_size();
        //   auto shape = loader_.getTensorShape(name);
        //   size_t slice_start = (shape[0] / world_size) * rank;
        //   size_t slice_size = shape[0] / world_size;
        //   return loader_.loadTensorSlice(name, slice_start, slice_size, device_idx);

        LOG_ERROR("[WeightManager] SHARDED strategy not yet implemented, falling back to REPLICATED");
        return getReplicatedWeight(name, device_idx);
    }

    std::shared_ptr<TensorBase> WeightManager::getInterleavedWeight(const std::string &name, int device_idx)
    {
        // Phase 3: Interleaved strategy (not yet implemented)
        // TODO: Implement NUMA-aware allocation with page interleaving
        //
        // For shared-memory multi-socket systems:
        // 1. Allocate with NUMA interleave policy (mbind)
        // 2. Memory distributed across NUMA nodes
        // 3. All ranks can access, but with varying latency
        // 4. Good for systems with fast interconnect (NVLink, etc.)

        LOG_ERROR("[WeightManager] INTERLEAVED strategy not yet implemented, falling back to REPLICATED");
        return getReplicatedWeight(name, device_idx);
    }

} // namespace llaminar2
