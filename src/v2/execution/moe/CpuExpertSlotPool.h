/**
 * @file CpuExpertSlotPool.h
 * @brief Persistent NUMA-owned CPU arrival slots for ExpertOverlay epochs.
 *
 * GPU-to-CPU migration writes final CPU NativeVNNI or row-major floating-point
 * bytes directly into these preallocated slots. Each slot owns stable
 * gate/up/down GEMM objects created before inference begins. A lease is
 * identified by `(expert, epoch)`, allowing an old and candidate epoch to
 * retain the same logical expert concurrently until RCU ticket retirement
 * releases the old engine aliases. Exact-geometry layers may share one pool;
 * each lease is therefore identified by `(layer, expert, epoch)`.
 */

#pragma once

#include "ExpertWeightFormat.h"
#include "ExpertTierWeightStream.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    class ITensorGemm;

    /**
     * @brief Preallocated CPU execution slots owned by one participant/geometry.
     *
     * Slot buffers are never resized after construction. The physical transfer
     * writes every byte before `MoEOverlayPreparedProjectionOperation` publishes
     * the corresponding stable engine. The pool itself retains engine objects;
     * aliasing shared pointers retain an epoch lease and prevent buffer reuse
     * while any inference bank can still reach the engine.
     */
    class CpuExpertSlotPool final
        : public std::enable_shared_from_this<CpuExpertSlotPool>
    {
    public:
        /** @brief Typed memory placement for final CPU execution bytes. */
        class MemoryPlacement final
        {
        public:
            /** @brief Whether pages have one exact NodeTP owner. */
            enum class Scope
            {
                AggregateDomain, ///< Device-free or deliberately aggregate CPU domain.
                BoundNode,       ///< Every slot page is strictly bound to one NUMA node.
            };

            MemoryPlacement() = delete;

            /** @brief Construct an explicit aggregate-domain placement. */
            static MemoryPlacement aggregateDomain() noexcept
            {
                return MemoryPlacement(Scope::AggregateDomain, -1);
            }

            /**
             * @brief Construct strict placement on one NUMA node.
             * @throws std::invalid_argument when @p node is negative.
             */
            static MemoryPlacement boundNode(int node);

            /** @return Declared placement scope. */
            [[nodiscard]] Scope scope() const noexcept { return scope_; }

            /** @return Exact NUMA node, or -1 for an aggregate domain. */
            [[nodiscard]] int node() const noexcept { return node_; }

            /** @return Whether construction must prove an exact node binding. */
            [[nodiscard]] bool requiresNodeBinding() const noexcept
            {
                return scope_ == Scope::BoundNode;
            }

        private:
            MemoryPlacement(Scope scope, int node) noexcept
                : scope_(scope), node_(node)
            {
            }

            Scope scope_;
            int node_;
        };

        /** @brief Immutable prepared format for one projection in every slot. */
        struct ProjectionSpec
        {
            ExpertTierWeightProjection projection =
                ExpertTierWeightProjection::Gate;
            int N = 0; ///< CPU GEMM output columns.
            int K = 0; ///< CPU GEMM reduction dimension.
            ExpertWeightFormat format; ///< Exact quantized or floating provenance.

            /** @brief Compare complete projection geometry and provenance. */
            bool operator==(const ProjectionSpec &) const = default;
        };

        /** @brief One writable final buffer and stable execution engine alias. */
        struct ProjectionLease
        {
            ExpertTierWeightProjection projection =
                ExpertTierWeightProjection::Gate;
            std::span<std::uint8_t> destination_bytes;
            std::shared_ptr<ITensorGemm> engine;
        };

        /** @brief Complete gate/up/down ownership for one expert and epoch. */
        struct Lease
        {
            int slot_index = -1;
            int layer_idx = -1;
            int expert_id = -1;
            std::uint64_t residency_epoch = 0;
            /** Shared control block that returns the slot after all aliases die. */
            std::shared_ptr<void> lifetime;
            std::vector<ProjectionLease> projections;
        };

        /** @brief Construction inputs that must be resolved before inference. */
        struct Config
        {
            int participant_id = -1;
            int layer_idx = -1;
            int capacity = 0;
            std::vector<ProjectionSpec> projections;
            MemoryPlacement memory_placement;
            std::string perf_device;
        };

        /**
         * @brief Allocate final buffers and construct every stable CPU engine.
         * @param config Exact endpoint, capacity, formats, and NUMA policy.
         * @return Shared pool retained by engine lease control blocks.
         * @throws std::invalid_argument for invalid geometry/format/topology.
         * @throws std::runtime_error when strict NUMA binding cannot be installed.
         */
        static std::shared_ptr<CpuExpertSlotPool> create(
            Config config,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
            PhysicalMemoryOwner owner);

        /**
         * @brief Materialize an explicitly unadmitted pool for isolated tests.
         *
         * Naming the bypass keeps test fixtures concise without allowing a
         * production caller to omit aggregate CPU admission accidentally.
         */
        static std::shared_ptr<CpuExpertSlotPool> createForTest(Config config);

        /**
         * @brief Reserve one inactive slot for an exact candidate epoch.
         * @param expert_id Logical routed expert.
         * @param residency_epoch Candidate epoch that will own the slot.
         * @return Complete writable triplet, or no value on capacity pressure.
         */
        [[nodiscard]] std::optional<Lease> acquire(
            int expert_id,
            std::uint64_t residency_epoch);

        /**
         * @brief Reserve one slot for a layer sharing this exact geometry.
         *
         * CPU expert engines are projection-geometry objects; no arithmetic
         * state is specific to a transformer-layer number. Including the
         * logical layer in the lease identity therefore permits a bounded
         * geometry arena to recycle physical bytes across layers without
         * allowing equal expert ids from two layers to alias.
         *
         * @param layer_idx Logical transformer layer receiving the expert.
         * @param expert_id Logical expert identity within that layer.
         * @param residency_epoch Positive candidate RCU epoch.
         * @return Complete writable triplet, or no value on pressure/duplicate.
         */
        [[nodiscard]] std::optional<Lease> acquireForLayer(
            int layer_idx,
            int expert_id,
            std::uint64_t residency_epoch);

        /** @return Total persistent slots planned for this endpoint. */
        [[nodiscard]] std::size_t capacity() const noexcept;

        /** @return Slots retained by any transfer, bank, or inference lease. */
        [[nodiscard]] std::size_t usedSlots() const noexcept;

        /** @return Immediately reservable inactive slots. */
        [[nodiscard]] std::size_t availableSlots() const noexcept;

        /** @return Exact physical RAM retained by every projection allocation. */
        [[nodiscard]] std::size_t allocationBytes() const noexcept
        {
            return allocation_bytes_;
        }

        /**
         * @brief Find a setup-layer slot for one exact expert/epoch identity.
         * @return Slot index when that setup-layer RCU version remains retained.
         */
        [[nodiscard]] std::optional<int> slotFor(
            int expert_id,
            std::uint64_t residency_epoch) const noexcept;

        /** @return Endpoint participant owning these buffers. */
        [[nodiscard]] int participantId() const noexcept
        {
            return config_.participant_id;
        }

        /** @return Setup layer whose geometry defines this reusable pool. */
        [[nodiscard]] int layerIndex() const noexcept
        {
            return config_.layer_idx;
        }

        /** @return Exact configured NUMA policy. */
        [[nodiscard]] const MemoryPlacement &memoryPlacement() const noexcept
        {
            return config_.memory_placement;
        }

    private:
        struct PreparedProjection;
        struct Slot;

        /** @brief Take already materialized slots after complete validation. */
        CpuExpertSlotPool(
            Config config,
            std::vector<Slot> slots,
            std::optional<PhysicalMemoryAllocationLease> memory_lease,
            std::size_t allocation_bytes);

        /** @brief Shared checked implementation for production and test creation. */
        static std::shared_ptr<CpuExpertSlotPool> createImpl(
            Config config,
            std::shared_ptr<PhysicalMemoryAuthority> memory_authority,
            PhysicalMemoryOwner owner,
            bool admitted);

        /** @brief Return a slot only for the exact generation that acquired it. */
        void release(
            int slot_index,
            int layer_idx,
            int expert_id,
            std::uint64_t residency_epoch) noexcept;

        Config config_;
        /** Claim remains live until every owned projection has been destroyed. */
        std::optional<PhysicalMemoryAllocationLease> memory_lease_;
        std::vector<Slot> slots_;
        std::size_t allocation_bytes_ = 0u;
        mutable std::mutex mutex_;
    };

} // namespace llaminar2
