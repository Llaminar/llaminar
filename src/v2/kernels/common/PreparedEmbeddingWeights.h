/**
 * @file PreparedEmbeddingWeights.h
 * @brief Model-owned source-native embedding allocation and its exact representation
 *
 * Quantized source blocks are copied without requantization or scale conversion.
 * The prepared store owns the stable handle and its physical-memory lease before
 * graph capture. Lookup consumes this allocation directly on its exact device;
 * no runtime weight preparation or workspace-owned replacement exists.
 */

#pragma once

#include "../../backends/DeviceId.h"
#include "NativeEmbeddingLayout.h"
#include "NativeEmbeddingIQTables.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace llaminar2
{
    class PhysicalMemoryAllocationLease;

    /**
     * @brief GPU-resident prepared embedding weights
     *
     * Contains unchanged native source blocks uploaded to one exact device.
     * Owns the device allocation and frees it on destruction.
     */
    struct PreparedEmbeddingWeights
    {
        /**
         * @brief Bind the allocation's sole live physical-memory claim.
         * @param lease Exact claim acquired before `device_data` allocation.
         * @throws std::invalid_argument for an invalid or size-mismatched claim.
         * @throws std::logic_error when a claim is already installed.
         */
        void bindPhysicalMemoryLease(
            PhysicalMemoryAllocationLease lease);

        void *device_data = nullptr; ///< Backend pointer to unchanged source-native blocks.
        NativeEmbeddingLayout layout{}; ///< Sole physical format/stride/extent authority.
        NativeEmbeddingIQTables iq_tables{}; ///< Borrowed immutable backend codebooks bound before capture.
        size_t vocab_size = 0;       ///< Number of vocabulary entries in this shard
        size_t vocab_offset = 0;     ///< First vocab index in this shard (0 when unsharded)
        size_t total_vocab = 0;      ///< Total vocab size across all shards
        int d_model = 0;             ///< Embedding dimension
        DeviceId device_id;          ///< Device this data lives on

        /** @brief Release native backing before retiring its sole physical-memory lease. */
        ~PreparedEmbeddingWeights();

        // Non-copyable, movable
        /** @brief Construct an empty handle with no physical allocation authority. */
        PreparedEmbeddingWeights();
        PreparedEmbeddingWeights(const PreparedEmbeddingWeights &) = delete;
        PreparedEmbeddingWeights &operator=(const PreparedEmbeddingWeights &) = delete;
        /**
         * @brief Transfer the complete physical allocation and borrowed codebooks.
         * @param other Previous sole allocation owner, empty after this move.
         */
        PreparedEmbeddingWeights(PreparedEmbeddingWeights &&other) noexcept;
        /**
         * @brief Retire this allocation and adopt the complete source representation.
         * @param other Previous sole allocation owner, empty after this move.
         * @return This new allocation owner.
         */
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
