/**
 * @file TensorFactory.h
 * @brief Factory for creating tensors with NUMA-aware allocation
 * @author David Sanftenberg
 */

#pragma once

#include "Tensors.h"
#include "../utils/MPIContext.h"
#include <memory>
#include <vector>
#include <cstddef>

#ifdef HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

namespace llaminar2
{
    /**
     * @brief Factory for creating tensors with NUMA-aware allocation
     *
     * Handles:
     * - NUMA node binding based on MPI rank
     * - Automatic tensor type selection based on data format
     * - Memory allocation on local NUMA node for optimal performance
     *
     * Thread-safe: Can be called from multiple threads, but each thread
     * should ideally be pinned to the same NUMA node as its MPI rank.
     */
    class TensorFactory
    {
    public:
        /**
         * @brief Initialize factory with MPI context
         * @param mpi_ctx MPI context for rank-to-NUMA mapping
         */
        explicit TensorFactory(const MPIContext &mpi_ctx);

        /**
         * @brief Create FP32 tensor with NUMA-aware allocation
         * @param shape Tensor dimensions
         * @param device_idx Optional device index for tracking (-1 = CPU, >=0 = GPU)
         * @return FP32 tensor allocated on local NUMA node
         */
        std::unique_ptr<FP32Tensor> createFP32(const std::vector<size_t> &shape, int device_idx = -1);

        /**
         * @brief Create FP16 tensor with NUMA-aware allocation
         * @param shape Tensor dimensions
         * @return FP16 tensor allocated on local NUMA node
         */
        std::unique_ptr<FP16Tensor> createFP16(const std::vector<size_t> &shape);

        /**
         * @brief Create FP16 tensor from existing data
         * @param shape Tensor dimensions
         * @param fp16_data Raw FP16 data (uint16_t)
         * @return FP16 tensor
         */
        std::unique_ptr<FP16Tensor> createFP16(const std::vector<size_t> &shape,
                                               const std::vector<uint16_t> &fp16_data);

        /**
         * @brief Create BF16 tensor with NUMA-aware allocation
         * @param shape Tensor dimensions
         * @return BF16 tensor allocated on local NUMA node
         */
        std::unique_ptr<BF16Tensor> createBF16(const std::vector<size_t> &shape);

        /**
         * @brief Create BF16 tensor from existing data
         * @param shape Tensor dimensions
         * @param bf16_data Raw BF16 data (uint16_t)
         * @return BF16 tensor
         */
        std::unique_ptr<BF16Tensor> createBF16(const std::vector<size_t> &shape,
                                               const std::vector<uint16_t> &bf16_data);

        /**
         * @brief Create quantized tensor from raw GGUF data
         * @param type Quantization type
         * @param shape Tensor dimensions
         * @param raw_data Raw quantized blocks
         * @return Quantized tensor of appropriate type
         */
        std::unique_ptr<TensorBase> createQuantized(TensorType type,
                                                    const std::vector<size_t> &shape,
                                                    const std::vector<uint8_t> &raw_data);

        /**
         * @brief Get NUMA node for current MPI rank
         * @return NUMA node index, or -1 if NUMA not available
         */
        int getNumaNode() const { return numa_node_; }

        /**
         * @brief Get MPI rank
         * @return MPI rank
         */
        int getMPIRank() const { return mpi_rank_; }

        /**
         * @brief Check if NUMA support is available
         * @return true if libnuma is available and system has multiple NUMA nodes
         */
        static bool isNumaAvailable();

    private:
        int mpi_rank_;
        int numa_node_; // NUMA node for this rank

        /**
         * @brief Bind current thread to NUMA node
         * Called before allocating large tensors
         */
        void bindToNumaNode();

        /**
         * @brief Get NUMA node for given MPI rank
         * @param rank MPI rank
         * @return NUMA node index, or -1 if not available
         */
        static int getNumaNodeForRank(int rank);
    };

} // namespace llaminar2
