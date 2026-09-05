/**
 * @file PreparedEmbeddingWeights.h
 * @brief GPU-resident prepared embedding table following the PreparedGemmWeights pattern
 *
 * Holds repacked EmbedQ8 embedding data in device memory. Created during weight
 * loading (Phase 4), cached in KernelFactory by {tensor, device} key.
 *
 * Unlike the previous workspace-based approach, this struct owns its GPU
 * allocation directly — no dependency on workspace lifecycle or lazy
 * initialization during the first forward pass.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace llaminar2
{
    class PhysicalMemoryAllocationLease;

    /**
     * @brief GPU-resident prepared embedding weights
     *
     * Contains repacked EmbedQ8Block data uploaded to a specific GPU device.
     * Owns the device allocation and frees it on destruction.
     */
    struct PreparedEmbeddingWeights
    {
        /**
         * @brief Compute the exact backing allocation for an embedding shard.
         * @param vocab_size Number of rows resident in this shard.
         * @param d_model Logical embedding width.
         * @return Checked bytes occupied by the universal EmbedQ8 layout.
         */
        [[nodiscard]] static size_t allocationBytes(
            size_t vocab_size,
            int d_model);

        /**
         * @brief Bind the allocation's sole live physical-memory claim.
         * @param lease Exact claim acquired before `device_data` allocation.
         * @throws std::invalid_argument for an invalid or size-mismatched claim.
         * @throws std::logic_error when a claim is already installed.
         */
        void bindPhysicalMemoryLease(
            PhysicalMemoryAllocationLease lease);

        void *device_data = nullptr; ///< GPU pointer to EmbedQ8Block array
        size_t byte_size = 0;        ///< Total bytes allocated on device
        size_t blocks_per_row = 0;   ///< EmbedQ8 blocks per vocabulary entry
        size_t vocab_size = 0;       ///< Number of vocabulary entries in this shard
        size_t vocab_offset = 0;     ///< First vocab index in this shard (0 when unsharded)
        size_t total_vocab = 0;      ///< Total vocab size across all shards
        int d_model = 0;             ///< Embedding dimension
        DeviceId device_id;          ///< Device this data lives on

        ~PreparedEmbeddingWeights();

        // Non-copyable, movable
        PreparedEmbeddingWeights();
        PreparedEmbeddingWeights(const PreparedEmbeddingWeights &) = delete;
        PreparedEmbeddingWeights &operator=(const PreparedEmbeddingWeights &) = delete;
        PreparedEmbeddingWeights(PreparedEmbeddingWeights &&other) noexcept;
        PreparedEmbeddingWeights &operator=(PreparedEmbeddingWeights &&other) noexcept;

    private:
        /** Ledger claim released only after the matching backing allocation. */
        std::unique_ptr<PhysicalMemoryAllocationLease>
            physical_memory_lease_;

        /** @brief Free owned storage while preserving noexcept teardown. */
        void releaseDeviceData() noexcept;
    };

    /**
     * @brief Handle for a prepared embedding weight entry
     *
     * Analogous to PreparedGemmHandle: stable identity object keyed by
     * (tensor, device) in the KernelFactory registry.
     */
    struct PreparedEmbeddingHandle
    {
        const class TensorBase *tensor = nullptr;          ///< Source embedding tensor
        DeviceId device_id;                                ///< Target device
        std::shared_ptr<PreparedEmbeddingWeights> weights; ///< Prepared data
    };

} // namespace llaminar2
