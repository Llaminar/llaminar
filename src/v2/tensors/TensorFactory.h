/**
 * @file TensorFactory.h
 * @brief Factory for creating tensors with NUMA-aware allocation
 * @author David Sanftenberg
 */

#pragma once

#include "Tensors.h"
#include "../backends/DeviceId.h"
#include "../utils/MPIContext.h"
#include "../execution/RuntimeConfig.h"
#include <memory>
#include <vector>
#include <cstddef>

#ifdef HAVE_NUMA
#include <numa.h>
#include <numaif.h>
#endif

// Forward declaration for BAR-backed tensor support
namespace llaminar2
{
    class DirectP2PEngine;
}

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
         * @param device Device identifier (default: CPU)
         * @return FP32 tensor allocated on local NUMA node
         */
        std::unique_ptr<FP32Tensor> createFP32(const std::vector<size_t> &shape, DeviceId device = DeviceId::cpu());

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
         * @brief Create INT32 tensor with NUMA-aware allocation
         * @param shape Tensor dimensions
         * @return INT32 tensor allocated on local NUMA node
         */
        std::unique_ptr<INT32Tensor> createINT32(const std::vector<size_t> &shape);

        /**
         * @brief Create Q8_1 tensor with NUMA-aware allocation
         * @param shape Tensor dimensions (logical element count)
         * @param device Device identifier (default: CPU)
         * @return Q8_1 tensor with properly sized block storage
         */
        std::unique_ptr<Q8_1Tensor> createQ8_1(const std::vector<size_t> &shape, DeviceId device = DeviceId::cpu());

        /**
         * @brief Create Q8_1 tensor from existing raw data
         * @param shape Tensor dimensions (logical element count)
         * @param raw_data Raw Q8_1 block data
         * @return Q8_1 tensor
         */
        std::unique_ptr<Q8_1Tensor> createQ8_1(const std::vector<size_t> &shape,
                                               const std::vector<uint8_t> &raw_data);

        /**
         * @brief Create Q16_1 tensor with NUMA-aware allocation
         *
         * Q16_1 provides 256× finer quantization than Q8_1 with FP32 scale.
         * Ideal for high-precision residual stream storage.
         *
         * @param shape Tensor dimensions (logical element count)
         * @param device Device identifier (default: CPU)
         * @return Q16_1 tensor with properly sized block storage
         */
        std::unique_ptr<Q16_1Tensor> createQ16_1(const std::vector<size_t> &shape,
                                                 DeviceId device = DeviceId::cpu());

        /**
         * @brief Create Q16_1 tensor with custom block size
         *
         * Creates Q16_1 tensor with variable block sizes for head-aligned
         * quantization in models with non-standard head dimensions.
         *
         * @param shape Tensor dimensions (logical element count)
         * @param block_size Block size for quantization (Q16_32, Q16_64, Q16_128, Q16_192)
         * @param device Device identifier (default: CPU)
         * @return Q16_1 tensor with specified block size
         */
        std::unique_ptr<Q16_1Tensor> createQ16_1(const std::vector<size_t> &shape,
                                                 Q16BlockSize block_size,
                                                 DeviceId device = DeviceId::cpu());

        /**
         * @brief Create activation tensor based on precision setting
         *
         * Convenience method that creates the appropriate tensor type based on
         * ActivationPrecision enum. Useful for creating activation buffers.
         *
         * @param shape Tensor dimensions
         * @param precision Activation precision (FP32, BF16, FP16, Q8_1)
         * @param device Device identifier (default: CPU)
         * @return Tensor of appropriate type
         */
        std::unique_ptr<TensorBase> createActivation(const std::vector<size_t> &shape,
                                                     ActivationPrecision precision,
                                                     DeviceId device = DeviceId::cpu());

        /**
         * @brief Create activation tensor with explicit head_dim for Q16 block size
         *
         * For Q16_1 and HybridQ16 precisions, uses optimal_q16_block_size(head_dim)
         * to determine the block size. This ensures Q tensor block size matches
         * the KV cache block size (which also uses optimal_q16_block_size).
         *
         * @param shape Tensor dimensions
         * @param precision Activation precision
         * @param head_dim Attention head dimension (used for Q16 block size)
         * @param device Device identifier
         * @return Tensor of appropriate type
         */
        std::unique_ptr<TensorBase> createActivation(const std::vector<size_t> &shape,
                                                     ActivationPrecision precision,
                                                     int head_dim,
                                                     DeviceId device);

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

        /**
         * @brief Enable mapped memory allocation for GPU tensors
         *
         * When enabled, FP32 tensors targeting GPU devices will be allocated using
         * zero-copy mapped memory (cudaHostAllocMapped / hipHostMallocMapped).
         * This enables direct host access without memcpy, ideal for:
         * - Snapshot/debugging modes where host needs to read GPU output
         * - Logits tensors that must be read by host for sampling
         *
         * @param enable true to use mapped memory for GPU FP32 tensors
         */
        void setUseMappedMemoryForGPU(bool enable) { use_mapped_memory_for_gpu_ = enable; }

        /**
         * @brief Check if mapped memory is enabled for GPU tensors
         * @return true if mapped memory allocation is enabled
         */
        bool useMappedMemoryForGPU() const { return use_mapped_memory_for_gpu_; }

        // =========================================================================
        // BAR-Backed Tensor Support (for PCIeBAR heterogeneous tensor parallelism)
        // =========================================================================

        /**
         * @brief Set DirectP2PEngine context for BAR-backed tensor allocation
         *
         * Must be called before createFP32BARBacked() can be used.
         * Typically called during LOCAL TP initialization when PCIeBAR backend is selected.
         *
         * @param p2p DirectP2PEngine context with mapped BAR regions
         */
        void setDirectP2P(std::shared_ptr<DirectP2PEngine> p2p);

        /**
         * @brief Check if BAR-backed tensor allocation is available
         * @return true if DirectP2PEngine context is set and BAR regions are mapped
         */
        bool canCreateBARBacked() const;

        /**
         * @brief Create FP32 tensor backed by PCIe BAR memory
         *
         * Creates a tensor in ROCm device's BAR-exposed VRAM region.
         * - ROCm device writes at full VRAM bandwidth (~900 GB/s)
         * - CUDA device reads through PCIe BAR (~2.65 GB/s)
         *
         * Use for row-parallel output buffers (FFN_DOWN, attention Wo) in
         * heterogeneous LOCAL TP with PCIeBAR backend.
         *
         * **Memory model**:
         * - Buffer physically resides in AMD VRAM (not system DRAM)
         * - CUDA writes via PCIe posted writes (~2.65 GB/s on PCIe 3.0)
         * - AMD accesses locally (full VRAM bandwidth ~900 GB/s)
         *
         * @param shape Tensor dimensions
         * @param rocm_device ROCm device that owns the BAR memory (writes locally)
         * @param cuda_device CUDA device that will read through PCIe
         * @return FP32 tensor with BAR-backed memory, or nullptr on failure
         * @throws std::runtime_error if DirectP2PEngine not set or BAR not available
         *
         * @see BARBackedTensor.h for detailed design documentation
         */
        std::unique_ptr<FP32Tensor> createFP32BARBacked(
            const std::vector<size_t> &shape,
            DeviceId rocm_device,
            DeviceId cuda_device);

    private:
        int mpi_rank_;
        int numa_node_;                          // NUMA node for this rank
        bool use_mapped_memory_for_gpu_ = false; // When true, FP32 GPU tensors use zero-copy mapped memory
        std::shared_ptr<DirectP2PEngine> direct_p2p_;  // For BAR-backed tensor allocation

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
