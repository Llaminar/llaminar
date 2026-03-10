/**
 * @file PerStageBufferPool.h
 * @brief Per-PP-stage buffer allocation for heterogeneous execution
 *
 * Part of the Unified PP Graph Architecture (Phase 1).
 * Each PP stage gets its own ActivationBuffers allocated on the
 * stage's primary device. Enables heterogeneous PP execution where
 * different stages run on different device types.
 *
 * @see docs/v2/UNIFIED_PP_GRAPH_ARCHITECTURE_PLAN.md
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#pragma once

#include "BufferSpec.h"
#include "DomainAwareBufferManager.h"
#include "../../../config/PipelineConfig.h"
#include "../../../models/GraphTypes.h" // For ActivationBuffers
#include "../../../tensors/TensorClasses.h"
#include <map>
#include <memory>
#include <vector>

namespace llaminar2
{

    // Forward declarations
    class TensorFactory;
    class MPIContext;

    /**
     * @brief Per-PP-stage buffer allocation
     *
     * Manages separate ActivationBuffers for each PP stage, with each
     * buffer set allocated on the stage's primary device.
     *
     * ## Design
     *
     * In Pipeline Parallelism (PP), different stages may execute on different
     * devices. For example:
     * - Stage 0: Layers 0-13 on CUDA GPU
     * - Stage 1: Layers 14-27 on ROCm GPU
     *
     * Each stage needs its own activation buffers on its device. This class:
     * 1. Allocates ActivationBuffers per stage
     * 2. Places buffers on each stage's primary device
     * 3. Provides lookup by stage ID or layer index
     *
     * ## Usage
     *
     * @code
     * // Create pipeline config (2 stages)
     * auto config = PipelineConfig::pipelineParallel2Stage(
     *     28, DeviceId::cuda(0), 14, DeviceId::rocm(0),
     *     CollectiveBackendType::PCIE_BAR);
     *
     * // Create buffer spec from model config
     * BufferSpec spec;
     * spec.d_model = 896;
     * spec.n_heads = 14;
     * // ... etc
     *
     * // Initialize buffer pool
     * PerStageBufferPool pool;
     * pool.initialize(config, spec);
     *
     * // Access buffers by stage
     * auto& stage0_buffers = pool.forStage(0);
     *
     * // Or by layer (finds owning stage)
     * auto& layer5_buffers = pool.forLayer(5);
     * @endcode
     *
     * ## Thread Safety
     *
     * NOT thread-safe. Should be used from a single thread or with
     * external synchronization.
     */
    class PerStageBufferPool
    {
    public:
        PerStageBufferPool() = default;
        ~PerStageBufferPool();

        // Non-copyable
        PerStageBufferPool(const PerStageBufferPool &) = delete;
        PerStageBufferPool &operator=(const PerStageBufferPool &) = delete;

        // Movable
        PerStageBufferPool(PerStageBufferPool &&) = default;
        PerStageBufferPool &operator=(PerStageBufferPool &&) = default;

        /**
         * @brief Initialize buffer pools for all PP stages
         * @param config Pipeline configuration with stage→device mapping
         * @param spec Buffer specification (shapes, dtypes)
         * @param mpi_ctx Optional MPI context for NUMA-aware allocation
         * @return true if allocation succeeded on all devices
         */
        bool initialize(const PipelineConfig &config, const PPStageBufferSpec &spec,
                        const MPIContext *mpi_ctx = nullptr);

        /**
         * @brief Check if pool has been initialized
         */
        [[nodiscard]] bool isInitialized() const { return config_ != nullptr; }

        /**
         * @brief Get buffers for a specific PP stage
         * @param stage_id PP stage index (0, 1, 2, ...)
         * @return Reference to stage's activation buffers
         * @throws std::out_of_range if stage_id invalid
         */
        ActivationBuffers &forStage(int stage_id);
        [[nodiscard]] const ActivationBuffers &forStage(int stage_id) const;

        /**
         * @brief Get buffers for the stage that owns a layer
         * @param layer_idx Layer index (0 to n_layers-1)
         * @return Reference to owning stage's buffers
         * @throws std::out_of_range if layer not covered by any stage
         */
        ActivationBuffers &forLayer(int layer_idx);
        [[nodiscard]] const ActivationBuffers &forLayer(int layer_idx) const;

        /**
         * @brief Get the device for a specific stage
         * @param stage_id PP stage index
         * @return Primary device for the stage
         */
        [[nodiscard]] DeviceId deviceForStage(int stage_id) const;

        /**
         * @brief Get number of stages
         */
        [[nodiscard]] int numStages() const;

        /**
         * @brief Release all allocated buffers
         */
        void release();

        /**
         * @brief Get allocation statistics
         */
        [[nodiscard]] const DomainAllocationStats &stats() const { return stats_; }

    private:
        /// Allocate buffers for a single stage
        bool allocateStageBuffers(int stage_id, DeviceId device, const PPStageBufferSpec &spec);

        /// Create a tensor on the specified device
        std::unique_ptr<TensorBase> createTensor(DeviceId device,
                                                 const std::vector<size_t> &shape,
                                                 TensorType dtype);

        /// Stage ID → allocated buffers
        std::map<int, ActivationBuffers> stage_buffers_;

        /// Stage ID → owning tensor storage (keeps tensors alive)
        std::map<int, std::vector<std::unique_ptr<TensorBase>>> tensor_storage_;

        /// Reference to pipeline config for stage→device lookup
        const PipelineConfig *config_ = nullptr;

        /// Buffer specification used for allocation
        PPStageBufferSpec spec_;

        /// Allocation statistics
        DomainAllocationStats stats_;

        /// Tensor factory for device-aware allocation
        std::unique_ptr<TensorFactory> tensor_factory_;
    };

} // namespace llaminar2
