/**
 * @file PhysicalMemoryBOM.h
 * @brief Canonical CPU/GPU physical-memory accounting and admission contract.
 *
 * Every production allocation is charged to exactly one typed physical
 * `(world rank, device)` resource.  Estimators, ExpertOverlay capacity
 * resolution, final preflight, and runtime attestation all consume this same
 * bill of materials (BOM); none of those callers may maintain an independent
 * subtotal.  An already-resident allocation remains visible in the complete
 * footprint while its typed credit is removed exactly once from incremental
 * allocation demand.
 */

#pragma once

#include "backends/DeviceId.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace llaminar2
{
    /**
     * @brief Concrete owner classes that can consume CPU RAM or GPU VRAM.
     *
     * The enum is deliberately storage-oriented.  A logical feature that owns
     * memory in two physical domains publishes two charges (for example a GPU
     * activation channel and its CPU-pinned transport region).  This prevents
     * feature names or device heuristics from deciding where bytes are billed.
     */
    enum class PhysicalMemoryOwner : std::uint8_t
    {
        PrimaryModelWeights = 0,   ///< Main prepared model-weight authority.
        AdditionalModelWeights,   ///< Concurrent replicated/mirrored weight views.
        ModelSourcePayload,        ///< Non-file-backed GGUF/source tensor storage.
        KVCache,                  ///< Live full-attention key/value storage.
        RecurrentLiveState,       ///< Live GDN/recurrent state.
        RecurrentCheckpointState, ///< MTP rollback/checkpoint state.
        SequenceMetadata,         ///< Device request positions and sequence tables.
        PrefixArchiveStaging,     ///< Bounded host/device archive/restore scratch.
        PrefixDeviceTier,         ///< Bounded accelerator-hot prefix storage.
        PrefixHostTier,           ///< Bounded RAM prefix storage.
        ActivationArena,          ///< Graph-stable activation tensors.
        ExecutionWorkspace,       ///< Persistent kernel workspace lease.
        LocalCollective,          ///< LocalTP collective buffers/scratch.
        NativeGraphExecutable,    ///< Opaque CUDA/HIP executable storage.
        GraphSnapshotArena,       ///< Graph-resident diagnostic checkpoint storage.
        WeightLoadStaging,        ///< Bounded setup upload/repack ring.
        ActivationTransportStaging, ///< Node-local/cross-rank activation lanes.
        ExpertMigrationStaging,   ///< Background expert transfer/repack lanes.
        ExpertShadowSlots,        ///< Inactive RCU expert-arrival banks.
        RoutedExpertWeights,      ///< Live prepared routed experts.
        Count,                    ///< Sentinel; never a valid charge owner.
    };

    /** @return Stable diagnostic spelling for one physical allocation owner. */
    [[nodiscard]] inline constexpr std::string_view toString(
        PhysicalMemoryOwner owner) noexcept
    {
        switch (owner)
        {
        case PhysicalMemoryOwner::PrimaryModelWeights:
            return "primary_model_weights";
        case PhysicalMemoryOwner::AdditionalModelWeights:
            return "additional_model_weights";
        case PhysicalMemoryOwner::ModelSourcePayload:
            return "model_source_payload";
        case PhysicalMemoryOwner::KVCache:
            return "kv_cache";
        case PhysicalMemoryOwner::RecurrentLiveState:
            return "recurrent_live_state";
        case PhysicalMemoryOwner::RecurrentCheckpointState:
            return "recurrent_checkpoint_state";
        case PhysicalMemoryOwner::SequenceMetadata:
            return "sequence_metadata";
        case PhysicalMemoryOwner::PrefixArchiveStaging:
            return "prefix_archive_staging";
        case PhysicalMemoryOwner::PrefixDeviceTier:
            return "prefix_device_tier";
        case PhysicalMemoryOwner::PrefixHostTier:
            return "prefix_host_tier";
        case PhysicalMemoryOwner::ActivationArena:
            return "activation_arena";
        case PhysicalMemoryOwner::ExecutionWorkspace:
            return "execution_workspace";
        case PhysicalMemoryOwner::LocalCollective:
            return "local_collective";
        case PhysicalMemoryOwner::NativeGraphExecutable:
            return "native_graph_executable";
        case PhysicalMemoryOwner::GraphSnapshotArena:
            return "graph_snapshot_arena";
        case PhysicalMemoryOwner::WeightLoadStaging:
            return "weight_load_staging";
        case PhysicalMemoryOwner::ActivationTransportStaging:
            return "activation_transport_staging";
        case PhysicalMemoryOwner::ExpertMigrationStaging:
            return "expert_migration_staging";
        case PhysicalMemoryOwner::ExpertShadowSlots:
            return "expert_shadow_slots";
        case PhysicalMemoryOwner::RoutedExpertWeights:
            return "routed_expert_weights";
        case PhysicalMemoryOwner::Count:
            break;
        }
        return "invalid";
    }

    /**
     * @brief Typed identity of one independently allocatable memory domain.
     *
     * CPU currently denotes the rank-local host/NUMA allocation authority;
     * CUDA and ROCm ordinals denote separate VRAM allocators.  `world_rank`
     * may be `-1` only for device-free estimates that have not yet been bound
     * to an orchestration topology.
     */
    struct PhysicalMemoryResource
    {
        int world_rank = -1;
        DeviceId device = DeviceId::invalid();
        std::size_t total_bytes = 0;
        std::size_t admission_available_bytes = 0;

        /** @return Whether the resource has coherent allocator geometry. */
        [[nodiscard]] bool valid() const noexcept
        {
            return world_rank >= -1 && device.is_valid() &&
                   (device.is_cpu() || device.is_gpu()) && total_bytes > 0 &&
                   admission_available_bytes <= total_bytes;
        }

        /** @return Stable human/wire identity derived only from typed fields. */
        [[nodiscard]] std::string id() const
        {
            if (!valid())
                return "invalid-memory-resource";
            return "rank=" +
                   (world_rank < 0 ? std::string("unbound")
                                   : std::to_string(world_rank)) +
                   ";device=" + device.toString();
        }

        /** @return Whether two BOMs charge the same physical allocator. */
        [[nodiscard]] bool sameAllocator(
            const PhysicalMemoryResource &other) const noexcept
        {
            return world_rank == other.world_rank && device == other.device &&
                   total_bytes == other.total_bytes &&
                   admission_available_bytes ==
                       other.admission_available_bytes;
        }
    };

    /** @brief Immutable planned and already-resident bytes for one owner. */
    struct PhysicalMemoryCharge
    {
        std::size_t planned_bytes = 0;
        std::size_t already_resident_bytes = 0;

        /** @return Bytes that still have to be allocated after admission. */
        [[nodiscard]] std::size_t incrementalBytes() const noexcept
        {
            return planned_bytes - already_resident_bytes;
        }
    };

    class PhysicalMemoryBOMBuilder;

    /**
     * @brief Immutable complete bill of materials for one physical allocator.
     *
     * Complete and incremental totals are intentionally separate.  Driver or
     * kernel availability is sampled after some reusable allocations may
     * already exist, so comparing complete footprint to current free bytes
     * would double-charge them.  Every credit is attached to the owner whose
     * allocation was concretely certified; there is no anonymous reserve or
     * global subtraction.
     */
    class PhysicalMemoryBOM final
    {
    public:
        /** @return Number of valid enumerable allocation-owner categories. */
        [[nodiscard]] static constexpr std::size_t ownerCount() noexcept
        {
            return kOwnerCount;
        }

        /**
         * @brief Convert a bounded wire/table index to its typed owner.
         * @throws std::out_of_range when index does not name an owner.
         */
        [[nodiscard]] static PhysicalMemoryOwner ownerAt(
            std::size_t index)
        {
            if (index >= kOwnerCount)
                throw std::out_of_range(
                    "Physical memory owner index is out of range");
            return static_cast<PhysicalMemoryOwner>(index);
        }

        /** @return Physical allocator against which every line is charged. */
        [[nodiscard]] const PhysicalMemoryResource &resource() const noexcept
        {
            return resource_;
        }

        /** @return Planned/retained line for one typed owner. */
        [[nodiscard]] const PhysicalMemoryCharge &charge(
            PhysicalMemoryOwner owner) const
        {
            const auto index = ownerIndex(owner);
            return charges_[index];
        }

        /** @return Complete bytes retained by one owner. */
        [[nodiscard]] std::size_t bytes(
            PhysicalMemoryOwner owner) const
        {
            return charge(owner).planned_bytes;
        }

        /** @return Already-resident credit attached to one owner. */
        [[nodiscard]] std::size_t alreadyResidentBytes(
            PhysicalMemoryOwner owner) const
        {
            return charge(owner).already_resident_bytes;
        }

        /** @return Complete physical footprint across every owner. */
        [[nodiscard]] std::size_t totalBytes() const noexcept
        {
            return total_bytes_;
        }

        /** @return Sum of concrete already-resident owner credits. */
        [[nodiscard]] std::size_t alreadyResidentBytes() const noexcept
        {
            return already_resident_bytes_;
        }

        /** @return Bytes that must be allocated from the sampled availability. */
        [[nodiscard]] std::size_t incrementalBytes() const noexcept
        {
            return total_bytes_ - already_resident_bytes_;
        }

        /** @return Whether every incremental byte fits the physical resource. */
        [[nodiscard]] bool fits() const noexcept
        {
            return incrementalBytes() <=
                   resource_.admission_available_bytes;
        }

        /** @return Incremental bytes beyond capacity, or zero on a fit. */
        [[nodiscard]] std::size_t deficitBytes() const noexcept
        {
            return fits()
                       ? 0u
                       : incrementalBytes() -
                             resource_.admission_available_bytes;
        }

        /** @return Capacity not assigned to any admitted owner. */
        [[nodiscard]] std::size_t remainingBytes() const noexcept
        {
            return fits()
                       ? resource_.admission_available_bytes -
                             incrementalBytes()
                       : 0u;
        }

        /** @return Compact deterministic diagnostic listing every nonzero line. */
        [[nodiscard]] std::string summary() const
        {
            std::ostringstream output;
            output << resource_.id();
            for (std::size_t index = 0; index < charges_.size(); ++index)
            {
                const auto &entry = charges_[index];
                if (entry.planned_bytes == 0)
                    continue;
                output << ' ' << toString(
                    static_cast<PhysicalMemoryOwner>(index))
                       << '=' << entry.planned_bytes;
                if (entry.already_resident_bytes != 0)
                    output << "(resident="
                           << entry.already_resident_bytes << ')';
            }
            output << " total=" << totalBytes()
                   << " incremental=" << incrementalBytes()
                   << " available="
                   << resource_.admission_available_bytes;
            return output.str();
        }

    private:
        friend class PhysicalMemoryBOMBuilder;

        static constexpr std::size_t kOwnerCount =
            static_cast<std::size_t>(PhysicalMemoryOwner::Count);

        PhysicalMemoryResource resource_;
        std::array<PhysicalMemoryCharge, kOwnerCount> charges_{};
        std::size_t total_bytes_ = 0;
        std::size_t already_resident_bytes_ = 0;

        /** @brief Reject the sentinel and corrupted enum values at the API edge. */
        [[nodiscard]] static std::size_t ownerIndex(
            PhysicalMemoryOwner owner)
        {
            const auto index = static_cast<std::size_t>(owner);
            if (index >= kOwnerCount)
                throw std::invalid_argument(
                    "Physical memory BOM received an invalid owner");
            return index;
        }
    };

    /**
     * @brief Sole mutable construction surface for a physical-memory BOM.
     *
     * Builders are setup-only value objects.  They may be initialized from an
     * immutable base BOM when a later authority (for example ExpertOverlay
     * quota resolution) adds owners, but the earlier BOM is never mutated.
     */
    class PhysicalMemoryBOMBuilder final
    {
    public:
        /** @brief Begin an empty BOM for one observed physical allocator. */
        explicit PhysicalMemoryBOMBuilder(PhysicalMemoryResource resource)
        {
            if (!resource.valid())
                throw std::invalid_argument(
                    "Physical memory BOM requires a valid resource");
            bom_.resource_ = std::move(resource);
        }

        /** @brief Extend an existing immutable BOM without changing its identity. */
        explicit PhysicalMemoryBOMBuilder(const PhysicalMemoryBOM &base)
            : bom_(base)
        {
            if (!bom_.resource_.valid())
                throw std::invalid_argument(
                    "Physical memory BOM extension requires a valid base");
        }

        /**
         * @brief Copy owner lines onto a newly bound physical resource.
         *
         * This is the only legal topology/capacity rebind and is used when a
         * device-free estimate becomes rank-bound or a fresh allocator
         * observation supersedes an older one. No charge is inferred from the
         * old resource geometry.
         */
        PhysicalMemoryBOMBuilder(
            PhysicalMemoryResource resource,
            const PhysicalMemoryBOM &charges)
            : PhysicalMemoryBOMBuilder(std::move(resource))
        {
            for (std::size_t index = 0;
                 index < PhysicalMemoryBOM::ownerCount();
                 ++index)
            {
                const auto owner = PhysicalMemoryBOM::ownerAt(index);
                const auto &entry = charges.charge(owner);
                add(
                    owner,
                    entry.planned_bytes,
                    entry.already_resident_bytes);
            }
        }

        /**
         * @brief Add one exact owner charge with an optional retained credit.
         * @param owner Concrete allocation owner.
         * @param planned_bytes Complete bytes owned after materialization.
         * @param already_resident_bytes Portion already excluded from the
         *        resource's sampled available bytes.
         * @return This builder for fluent setup composition.
         * @throws std::invalid_argument when retained bytes exceed planned.
         * @throws std::overflow_error when any owner or total wraps.
         */
        PhysicalMemoryBOMBuilder &add(
            PhysicalMemoryOwner owner,
            std::size_t planned_bytes,
            std::size_t already_resident_bytes = 0)
        {
            if (already_resident_bytes > planned_bytes)
                throw std::invalid_argument(
                    "Physical memory retained credit exceeds its owner charge");
            const auto index = PhysicalMemoryBOM::ownerIndex(owner);
            auto &entry = bom_.charges_[index];
            entry.planned_bytes = checkedAdd(
                entry.planned_bytes, planned_bytes, toString(owner));
            entry.already_resident_bytes = checkedAdd(
                entry.already_resident_bytes,
                already_resident_bytes,
                toString(owner));
            if (entry.already_resident_bytes > entry.planned_bytes)
                throw std::invalid_argument(
                    "Physical memory accumulated retained credit exceeds its owner charge");
            bom_.total_bytes_ = checkedAdd(
                bom_.total_bytes_, planned_bytes, "complete BOM");
            bom_.already_resident_bytes_ = checkedAdd(
                bom_.already_resident_bytes_,
                already_resident_bytes,
                "retained BOM credit");
            return *this;
        }

        /** @return Immutable snapshot of every charge installed so far. */
        [[nodiscard]] PhysicalMemoryBOM build() const
        {
            return bom_;
        }

    private:
        PhysicalMemoryBOM bom_;

        /** @brief Add byte counts without allowing an affordable-looking wrap. */
        [[nodiscard]] static std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            std::string_view description)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    "Physical memory " + std::string(description) +
                    " charge overflows size_t");
            }
            return left + right;
        }
    };

    /**
     * @brief Immutable proof that one complete BOM fit its sampled allocator.
     *
     * Only this type may cross from planning into runtime materialization or
     * be used as the base of a later capacity-resolution phase.  Constructing
     * it from an over-budget BOM fails immediately, so a runtime owner can
     * never receive an uncertified plan by accident.
     */
    class PhysicalMemoryAdmissionCertificate final
    {
    public:
        /**
         * @brief Certify a fitting immutable BOM.
         * @throws std::invalid_argument when the BOM is over budget.
         */
        explicit PhysicalMemoryAdmissionCertificate(PhysicalMemoryBOM bom)
            : bom_(std::move(bom))
        {
            if (!bom_.fits())
            {
                throw std::invalid_argument(
                    "Physical memory admission failed for " +
                    bom_.resource().id() + ": deficit_bytes=" +
                    std::to_string(bom_.deficitBytes()));
            }
        }

        /** @return The sole immutable byte authority certified by this proof. */
        [[nodiscard]] const PhysicalMemoryBOM &bom() const noexcept
        {
            return bom_;
        }

    private:
        PhysicalMemoryBOM bom_;
    };
} // namespace llaminar2
