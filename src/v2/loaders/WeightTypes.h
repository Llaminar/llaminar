/**
 * @file WeightTypes.h
 * @brief Shared type definitions for weight management
 *
 * Enums and type aliases used across IWeightManager, WeightManagerConfig,
 * WeightManager, and their tests. Extracted to break the circular dependency
 * where these types were defined in WeightManager.h but needed by
 * IWeightManager.h and WeightManagerConfig.h.
 *
 * @author David Sanftenberg
 */

#pragma once

#include <functional>
#include <memory>
#include <string>

namespace llaminar2
{

    // Forward declarations
    class TensorBase;

    /**
     * @brief Weight distribution strategy
     */
    enum class WeightDistributionStrategy
    {
        REPLICATED,       ///< Full copy per rank (2x memory on 2-socket, no communication)
        SHARDED,          ///< Partition across ranks (1x memory, Allreduce after matmul)
        INTERLEAVED,      ///< NUMA-aware global (shared memory, remote access penalty)
        LAYER_PARTITIONED ///< Load only weights for assigned layers (for Pipeline Parallelism)
    };

    /**
     * @brief Sharding mode for a weight tensor
     */
    enum class ShardingMode
    {
        REPLICATE,        ///< Not sharded, full copy on each rank (norms, biases)
        COLUMN_PARALLEL,  ///< Split output dimension (rows of weight) - for Gate/Up, QKV
        ROW_PARALLEL,     ///< Split output dimension (rows of weight) + allreduce - for Wo
        INPUT_PARALLEL,   ///< Split input dimension (columns of weight) + allreduce - for Down
        EXPERT_PARALLEL   ///< Split expert dimension of 3D MoE tensors - for expert weights
    };

    /**
     * @brief Pre-packing progress callback for weight preloading
     *
     * Called during preloading with:
     * @param current Number of weights packed so far
     * @param total Total number of weights to pack
     * @param name Current weight name being packed
     */
    using PreloadProgressCallback = std::function<void(size_t current, size_t total, const std::string &name)>;

    /**
     * @brief Transforms a weight tensor before GEMM packing.
     *
     * Applied once per GEMM weight during packGemmWeights(). The callback
     * receives the weight name and original tensor, and returns either:
     *   - A new tensor (e.g., rotated copy) to replace the original
     *   - The same tensor if no preprocessing is needed
     *
     * The returned tensor becomes the cache entry, so subsequent
     * getWeightForDevice() calls see the preprocessed version.
     */
    using WeightPreprocessor = std::function<std::shared_ptr<TensorBase>(
        const std::string &name, std::shared_ptr<TensorBase> tensor)>;

} // namespace llaminar2
