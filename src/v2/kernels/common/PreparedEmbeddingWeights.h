/**
 * @file PreparedEmbeddingWeights.h
 * @brief Model-owned source-native embedding allocation and its exact representation
 *
 * Quantized source blocks are copied without requantization or scale conversion.
 * The prepared store owns the stable handle and its physical-memory lease before
 * graph capture. Lookup consumes this allocation directly on its exact device;
 * no runtime weight preparation or workspace-owned replacement exists.
 * CPU preparation may instead retain the original final native tensor. Its
 * allocation lease stays on that tensor; this handle neither copies nor frees
 * its pointer as an independently allocated backend buffer.
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
    class TensorBase;

    /**
     * @brief Prepared source-native embedding storage on one exact endpoint.
     *
     * Contains unchanged native source blocks uploaded to one exact device.
     * Owns a copied backend allocation or retains an already-final CPU tensor.
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

        /**
         * @brief Retain final native CPU tensor storage without another allocation.
         * @param tensor Owning CPU matrix matching the declared layout and endpoint.
         * @throws std::invalid_argument For missing, borrowed, mapped or mismatched storage.
         * @throws std::logic_error If this handle already owns backing or a separate lease.
         * @details The original tensor remains live execution data. Cleanup may
         * release source windows, but must not release this tensor's raw bytes.
         */
        void bindFinalCPUStorage(std::shared_ptr<const TensorBase> tensor);

        /** @return Retained final CPU owner, or empty for an independently copied backend allocation. */
        const std::shared_ptr<const TensorBase> &retainedFinalCPUTensor() const noexcept
        {
            return final_cpu_tensor_;
        }

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
        std::shared_ptr<const TensorBase> final_cpu_tensor_; ///< Sole CPU backing owner; its existing lease is never duplicated here.

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
