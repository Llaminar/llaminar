/**
 * @file GpuExpertSlotPool.h
 * @brief Persistent GPU slots with epoch-safe ExpertOverlay ownership.
 *
 * Rebalanced GPU arrivals share fixed per-layer shapes. This pool preallocates
 * bounded compute-facing slots plus optional transient transfer slots. Active
 * leases are keyed by `(expert, residency epoch)`, so an old RCU bank and its
 * candidate successor may retain the same logical expert concurrently without
 * overwriting either version. Legacy Dynamic/LLEP callers use epoch zero; the
 * arbitrary-tier ExpertOverlay path must always supply its positive epoch.
 */

#pragma once

#include "ExpertWeightFormat.h"
#include "../../backends/DeviceId.h"
#include "../../loaders/gpu_pipeline/WeightVRAMPool.h"
#include "../../tensors/NativeVnniFormatInfo.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaminar2
{
    class IBackend;
    class LoadOrchestrator;

    /**
     * @brief Model-lifetime GPU allocation pool with lease-driven slot reuse.
     *
     * `LoadOrchestrator` owns every physical allocation. A returned lifetime
     * token is the only authority that releases a logical assignment, allowing
     * prepared GEMM aliases and residency banks to pin a slot independently of
     * the maintenance operation that filled it.
     */
    class GpuExpertSlotPool final
        : public std::enable_shared_from_this<GpuExpertSlotPool>
    {
    public:
        /** @brief Allocation geometry for one gate/up/down projection. */
        struct ProjectionSpec
        {
            std::string label; ///< Stable projection name.
            int N = 0;        ///< Output-column count.
            int K = 0;        ///< Reduction dimension; quantized formats require /32.
            int payload_bytes_per_block = 0; ///< Quantized allocation bytes/block.
            bool is_asymmetric = false;      ///< Allocate a minima region.
            bool has_emins = false;          ///< Allocate extended minima.
            uint8_t codebook_id = 0;         ///< Initial descriptor codebook.
            ExpertWeightFormat format; ///< Original arithmetic identity.
        };

        /** @brief One projection's persistent allocation and exact geometry. */
        struct ProjectionSlot
        {
            ProjectionSpec spec;
            WeightVRAMPool::WeightSlot slot;
            uint32_t blocks_per_row = 0;
        };

        /** @brief Compute-facing slot retained by an expert/epoch lease. */
        struct AcquiredSlot
        {
            int slot_index = -1;
            int expert_id = -1;
            uint64_t residency_epoch = 0;
            std::shared_ptr<void> lifetime;
            std::vector<ProjectionSlot> projections;
        };

        /** @brief Transient staging slot retained by an expert/epoch lease. */
        struct TransferSlot
        {
            int slot_index = -1;
            int expert_id = -1;
            uint64_t residency_epoch = 0;
            std::shared_ptr<void> lifetime;
            std::vector<ProjectionSlot> projections;
        };

        /**
         * @brief Plan and allocate every active and staging projection.
         * @throws std::invalid_argument for invalid capacity or geometry.
         * @throws std::runtime_error when device allocation fails.
         */
        static std::shared_ptr<GpuExpertSlotPool> create(
            IBackend *backend,
            DeviceId device,
            int device_ordinal,
            int layer_idx,
            int active_capacity,
            std::vector<ProjectionSpec> specs,
            int transfer_capacity = 0);

        /** @return Active capacity covering cache churn and one arrival batch. */
        static int recommendedCapacity(int num_experts, size_t arrival_batch_size);

        /** @return Staging capacity covering one bounded rolling wave. */
        static int recommendedTransferCapacity(int num_experts, size_t arrival_batch_size);

        /**
         * @brief Acquire a legacy non-RCU active slot under epoch zero.
         * @param expert_id Logical expert identity.
         * @return Lease, or no value on duplicate identity/capacity pressure.
         */
        std::optional<AcquiredSlot> acquire(int expert_id);

        /**
         * @brief Acquire an active slot for one exact ExpertOverlay epoch.
         * @param expert_id Logical expert identity.
         * @param residency_epoch Positive RCU candidate epoch.
         * @return Lease, or no value on invalid/duplicate identity or pressure.
         */
        std::optional<AcquiredSlot> acquire(
            int expert_id,
            uint64_t residency_epoch);

        /** @brief Acquire a legacy epoch-zero transfer slot. */
        std::optional<TransferSlot> acquireTransferSlot(int expert_id);

        /** @brief Acquire a transfer slot for one exact ExpertOverlay epoch. */
        std::optional<TransferSlot> acquireTransferSlot(
            int expert_id,
            uint64_t residency_epoch);

        /** @return Number of compute-facing slots. */
        size_t activeCapacity() const;
        /** @return Number of transient transfer slots. */
        size_t transferCapacity() const;
        /** @return Backward-compatible alias for active capacity. */
        size_t capacity() const;
        /** @return Number of retained active identities. */
        size_t usedSlots() const;
        /** @return Number of retained transfer identities. */
        size_t usedTransferSlots() const;
        /** @return Number of immediately acquirable active slots. */
        size_t availableSlots() const;
        /** @return Number of immediately acquirable transfer slots. */
        size_t availableTransferSlots() const;

        /** @return Legacy epoch-zero active slot for @p expert_id. */
        std::optional<int> slotForExpert(int expert_id) const;

        /** @return Active slot for one exact expert/epoch identity. */
        std::optional<int> slotForExpert(
            int expert_id,
            uint64_t residency_epoch) const;

        /** @return Legacy epoch-zero transfer slot for @p expert_id. */
        std::optional<int> transferSlotForExpert(int expert_id) const;

        /** @return Transfer slot for one exact expert/epoch identity. */
        std::optional<int> transferSlotForExpert(
            int expert_id,
            uint64_t residency_epoch) const;

    private:
        /** @brief Exact logical assignment of one physical slot. */
        struct SlotIdentity
        {
            int expert_id = -1;
            uint64_t residency_epoch = 0;

            bool operator==(const SlotIdentity &) const = default;
        };

        /** @brief Stable hash for the complete expert/epoch identity. */
        struct SlotIdentityHash
        {
            size_t operator()(const SlotIdentity &identity) const noexcept;
        };

        /** @brief Retain validated configuration and allocated arena owner. */
        GpuExpertSlotPool(IBackend *backend,
                          DeviceId device,
                          int device_ordinal,
                          int layer_idx,
                          int active_capacity,
                          int transfer_capacity,
                          std::vector<ProjectionSpec> specs,
                          std::shared_ptr<LoadOrchestrator> orchestrator);

        /** @return Unique model-lifetime allocation name for an active slot. */
        static std::string activeSlotName(int slot_index, const std::string &label);
        /** @return Unique model-lifetime allocation name for a staging slot. */
        static std::string transferSlotName(int slot_index, const std::string &label);

        /** @brief Release only when index and complete identity still match. */
        void releaseSlot(int slot_index, SlotIdentity identity) noexcept;

        /** @brief Release a staging slot only for its exact retained identity. */
        void releaseTransferSlot(
            int slot_index,
            SlotIdentity identity) noexcept;

        IBackend *backend_ = nullptr;
        DeviceId device_;
        int device_ordinal_ = -1;
        int layer_idx_ = -1;
        int active_capacity_ = 0;
        int transfer_capacity_ = 0;
        std::vector<ProjectionSpec> specs_;
        std::shared_ptr<LoadOrchestrator> orchestrator_;

        mutable std::mutex mutex_;
        std::vector<int> expert_by_slot_;
        std::vector<uint64_t> epoch_by_slot_;
        std::unordered_map<SlotIdentity, int, SlotIdentityHash>
            slot_by_identity_;
        std::vector<int> expert_by_transfer_slot_;
        std::vector<uint64_t> epoch_by_transfer_slot_;
        std::unordered_map<SlotIdentity, int, SlotIdentityHash>
            transfer_slot_by_identity_;
    };

} // namespace llaminar2
