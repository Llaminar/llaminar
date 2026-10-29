/**
 * @file TensorFactory.h
 * @brief Tensor construction preserving the admitted rank CPU allocation domain.
 *
 * CPUBackend supplies aggregate (-1) or explicit node-bound allocation intent.
 * Current CPU affinity and MPI rank numbers never redefine that resource domain.
 * @author David Sanftenberg
 */

#pragma once

#include "Tensors.h"
#include "../backends/DeviceId.h"
#include "../utils/MPIContext.h"
#include "../execution/config/RuntimeConfig.h"
#include <memory>
#include <vector>
#include <cstddef>

#include <numa.h>
#include <numaif.h>

namespace llaminar2
{
    /**
     * @brief Factory for creating tensors with NUMA-aware allocation
     *
     * Handles:
     * - NUMA policy from the initialized rank CPU backend
     * - Automatic tensor type selection based on data format
     * - Aggregate placement or the explicitly admitted NUMA node
     *
     * Aggregate factories never alter a calling thread's memory policy. Bound
     * factories retain the existing persistent MPOL_BIND behavior on each calling
     * thread; they do not restore that policy at destruction. CPU affinity alone
     * is never evidence that a single-node memory constraint was admitted.
     */
    class TensorFactory
    {
    public:
        /**
         * @brief Snapshot the existing CPU backend allocation domain.
         * @param mpi_ctx MPI context retained only for rank diagnostics.
         *
         * cpuBackendNUMANode() is authoritative: -1 preserves aggregate intent
         * and the caller's current policy, including an externally imposed bind.
         * A non-negative admitted node retains explicit MPOL_BIND semantics.
         * When no CPU backend exists the authority reports aggregate (-1); this
         * constructor neither initializes a backend nor invents rank placement.
         */
        explicit TensorFactory(const IMPIContext &mpi_ctx);

        /**
         * @brief Create FP32 tensor with NUMA-aware allocation
         * @param shape Tensor dimensions
         * @param device Device identifier (default: CPU)
         * @return FP32 tensor allocated on local NUMA node
         */
        std::unique_ptr<FP32Tensor> createFP32(const std::vector<size_t> &shape, DeviceId device = DeviceId::cpu());

        /**
         * @brief Adopt initialized aligned FP32 storage on this factory's NUMA node.
         * @param shape Tensor dimensions.
         * @param host_data Exact element storage; ownership is transferred.
         * @param device Logical home device for subsequent preparation.
         * @return FP32 tensor owning the original allocation.
         */
        std::unique_ptr<FP32Tensor> createFP32Owned(
            const std::vector<size_t> &shape,
            AlignedVector<float> host_data,
            DeviceId device = DeviceId::cpu());

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
         * @brief Adopt initialized aligned FP16 storage without copying.
         * @param shape Tensor dimensions.
         * @param fp16_data Exact element storage; ownership is transferred.
         * @return FP16 tensor owning the original allocation.
         */
        std::unique_ptr<FP16Tensor> createFP16Owned(
            const std::vector<size_t> &shape,
            AlignedVector<uint16_t> fp16_data);

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
         * @brief Adopt initialized aligned BF16 storage without copying.
         * @param shape Tensor dimensions.
         * @param bf16_data Exact element storage; ownership is transferred.
         * @return BF16 tensor owning the original allocation.
         */
        std::unique_ptr<BF16Tensor> createBF16Owned(
            const std::vector<size_t> &shape,
            AlignedVector<uint16_t> bf16_data);

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
         * @param raw_data Raw quantized blocks copied into aligned tensor storage.
         * @return Quantized tensor of appropriate type
         */
        std::unique_ptr<TensorBase> createQuantized(TensorType type,
                                                    const std::vector<size_t> &shape,
                                                    const std::vector<uint8_t> &raw_data);

        /**
         * @brief Adopt initialized aligned native-codebook bytes without copying.
         * @param type Quantization type.
         * @param shape Tensor dimensions.
         * @param raw_data Exact packed bytes; ownership is transferred.
         * @return Quantized tensor owning the original aligned allocation.
         */
        std::unique_ptr<TensorBase> createQuantizedOwned(
            TensorType type,
            const std::vector<size_t> &shape,
            AlignedVector<uint8_t> raw_data);

        /**
         * @brief Create quantized tensor with zero-copy mmap-backed data
         *
         * The tensor borrows data directly from a memory-mapped file region
         * instead of copying. The mmap_lifetime_owner keeps the mapped region
         * alive as long as the tensor exists.
         *
         * @param type Quantization type
         * @param shape Tensor dimensions
         * @param mmap_data Pointer into mmap'd region (must remain valid)
         * @param byte_size Size of the tensor data in bytes
         * @param mmap_lifetime_owner Shared pointer that keeps the mmap region alive
         * @return Quantized tensor referencing mmap'd memory (no copy)
         */
        std::unique_ptr<TensorBase> createQuantizedZeroCopy(TensorType type,
                                                            const std::vector<size_t> &shape,
                                                            const uint8_t *mmap_data,
                                                            size_t byte_size,
                                                            std::shared_ptr<void> mmap_lifetime_owner);

        /**
         * @brief Ensure NUMA memory binding is active for the calling thread
         *
         * For an explicitly node-bound CPU backend, sets strict MPOL_BIND so
         * subsequent allocations use that admitted node. The policy persists
         * beyond the call and is not restored when the factory is destroyed.
         *
         * Call this before allocating temporary buffers (e.g. file I/O scratch)
         * that should be NUMA-local. The create* methods already call this
         * internally, but callers who allocate their own buffers before
         * passing data to the factory should call this first.
         *
         * Aggregate (-1) is a no-op: it neither installs a singleton policy nor
         * clears an external caller policy. The current CPU and MPI rank do not
         * narrow the aggregate CPU resource.
         */
        void ensureNumaBinding();

        /**
         * @brief Parallel NUMA-aware memcpy for large buffers
         *
         * Uses OpenMP to copy contiguous ranges; first-touch placement follows
         * each worker's existing policy. This operation does not infer a NUMA
         * node or change worker policy. Small buffers (< 128 KB) use std::memcpy.
         *
         * @param dst  Destination buffer (should be allocated with NUMA binding)
         * @param src  Source buffer
         * @param bytes Number of bytes to copy
         */
        static void numaMemcpy(void *dst, const void *src, size_t bytes);

        /**
         * @brief Return the snapshotted CPU backend allocation domain.
         * @return Explicit admitted node, or -1 for aggregate CPU allocation.
         */
        int getNumaNode() const { return numa_node_; }

        /**
         * @brief Get MPI rank
         * @return MPI rank
         */
        int getMPIRank() const { return mpi_rank_; }

        /**
         * @brief Check if this system exposes more than one NUMA node
         * @return true if NUMA policy APIs are available and multiple nodes are configured
         */
        static bool isNumaAvailable();

    private:
        int mpi_rank_;
        int numa_node_; // CPU backend allocation authority; -1 means aggregate.

        /**
         * @brief Install the existing persistent policy for an admitted node.
         * Called only for non-negative CPU backend allocation authority. The
         * aggregate branch never invokes this method or changes caller policy.
         */
        void bindToNumaNode();

    };

} // namespace llaminar2
