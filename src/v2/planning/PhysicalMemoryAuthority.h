/**
 * @file PhysicalMemoryAuthority.h
 * @brief Topology-wide CPU/GPU admission and live-allocation authority.
 *
 * Per-resource arithmetic lives in @ref PhysicalMemoryBOM. This file supplies
 * the three typed transitions above it: a mutable setup builder coalesces all
 * logical contributors, an immutable certificate admits the complete physical
 * topology, and a thread-safe materialization ledger issues RAII leases for
 * the exact allocations that remain live. No caller may compare independent
 * feature subtotals against the same allocator or consume an anonymous safety
 * reserve.
 */

#pragma once

#include "PhysicalMemoryBOM.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llaminar2
{
    /** @brief Stable join key for one rank-local CPU or accelerator allocator. */
    struct PhysicalMemoryAllocatorIdentity
    {
        int world_rank = -1;
        DeviceId device = DeviceId::invalid();

        /** @return Whether this identity names a supported physical allocator. */
        [[nodiscard]] bool valid() const noexcept
        {
            return world_rank >= -1 && device.is_valid() &&
                   (device.is_cpu() || device.is_gpu());
        }

        /** @return Stable diagnostic spelling without capacity information. */
        [[nodiscard]] std::string toString() const;

        bool operator==(
            const PhysicalMemoryAllocatorIdentity &) const = default;
    };

    class PhysicalMemoryPlanBuilder;

    /** @brief Immutable aggregate BOM for every physical allocator in a plan. */
    class PhysicalMemoryPlan final
    {
    public:
        /** @return Deterministically ordered, unique physical-resource BOMs. */
        [[nodiscard]] const std::vector<PhysicalMemoryBOM> &resources()
            const noexcept
        {
            return resources_;
        }

        /** @return Exact BOM for @p identity, or nullptr when it is absent. */
        [[nodiscard]] const PhysicalMemoryBOM *find(
            PhysicalMemoryAllocatorIdentity identity) const noexcept;

        /** @return Whether every unique allocator admits its incremental bytes. */
        [[nodiscard]] bool fits() const noexcept;

        /** @return Sum of complete physical footprints with overflow checking. */
        [[nodiscard]] std::size_t totalBytes() const;

        /** @return Sum of incremental allocation demand with overflow checking. */
        [[nodiscard]] std::size_t incrementalBytes() const;

        /** @return Sum of sampled allocatable bytes with overflow checking. */
        [[nodiscard]] std::size_t admissionAvailableBytes() const;

        /**
         * @brief Compare a later runner's needs with this admitted envelope.
         *
         * Model-owned prepared weights and sealed workspace backing can span
         * runner generations.  Their admission object must span those same
         * generations: minting a second certificate from a later free-memory
         * sample would give the retained bytes two accounting authorities.
         * This comparison therefore treats this plan as the immutable upper
         * bound and accepts a later plan only when every required owner line
         * fits inside it on the same physical allocator.
         *
         * `admission_available_bytes` and `already_resident_bytes` are
         * deliberately observations, not identity.  The former changes as
         * runner-owned allocations retire; the latter changes when an
         * allocation made by the first generation becomes retained input to
         * the next.  Physical total capacity must remain identical. Extra
         * resources or owner capacity in this envelope are legal because
         * setup-only staging can disappear in a later generation.
         *
         * @param required Complete footprint requested by the later runner.
         * @return Empty when `required` fits; otherwise the first deterministic
         *         incompatibility diagnostic.
         */
        [[nodiscard]] std::optional<std::string>
        requiredFootprintMismatch(
            const PhysicalMemoryPlan &required) const;

        /** @return Deterministic multiline diagnostics for every allocator. */
        [[nodiscard]] std::string summary() const;

    private:
        friend class PhysicalMemoryPlanBuilder;
        std::vector<PhysicalMemoryBOM> resources_;
    };

    /**
     * @brief Sole topology-wide mutable accounting surface.
     *
     * Each logical subsystem contributes a typed BOM. Contributions with the
     * same `(world rank, device)` key are merged before admission; conflicting
     * observations of that allocator are rejected instead of racing to become
     * authoritative.
     */
    class PhysicalMemoryPlanBuilder final
    {
    public:
        /** @brief Start an empty topology-wide plan. */
        PhysicalMemoryPlanBuilder() = default;

        /** @brief Extend an immutable plan without mutating the original. */
        explicit PhysicalMemoryPlanBuilder(const PhysicalMemoryPlan &base);

        /**
         * @brief Merge one complete typed resource BOM.
         * @return This builder for fluent setup composition.
         * @throws std::invalid_argument for conflicting resource observations.
         * @throws std::overflow_error when aggregate owner arithmetic wraps.
         */
        PhysicalMemoryPlanBuilder &add(const PhysicalMemoryBOM &bom);

        /**
         * @brief Add one owner directly against an observed resource.
         * @return This builder for fluent setup composition.
         */
        PhysicalMemoryPlanBuilder &add(
            PhysicalMemoryResource resource,
            PhysicalMemoryOwner owner,
            std::size_t planned_bytes,
            std::size_t already_resident_bytes = 0u);

        /** @return Immutable plan sorted by rank and typed device identity. */
        [[nodiscard]] PhysicalMemoryPlan build() const;

    private:
        PhysicalMemoryPlan plan_;
    };

    /**
     * @brief Immutable proof that the complete CPU/GPU topology was admitted.
     *
     * This is the only value accepted by the live materialization ledger. The
     * constructor checks every physical resource at once, preventing several
     * independently fitting logical plans from overcommitting one allocator.
     */
    class PhysicalMemoryPlanAdmissionCertificate final
    {
    public:
        /**
         * @brief Certify a non-empty fitting aggregate plan.
         * @throws std::invalid_argument when any resource is over budget.
         */
        explicit PhysicalMemoryPlanAdmissionCertificate(
            PhysicalMemoryPlan plan);

        /** @return Complete immutable topology-wide byte authority. */
        [[nodiscard]] const PhysicalMemoryPlan &plan() const noexcept
        {
            return plan_;
        }

        /**
         * @brief Derive the legacy single-resource proof without re-accounting.
         * @throws std::out_of_range when @p identity is absent.
         */
        [[nodiscard]] PhysicalMemoryAdmissionCertificate certificateFor(
            PhysicalMemoryAllocatorIdentity identity) const;

    private:
        PhysicalMemoryPlan plan_;
    };

    /** @brief Distinguishes new allocation from adoption of certified residency. */
    enum class PhysicalMemoryMaterializationKind : std::uint8_t
    {
        NewAllocation = 0,
        AdoptedResident,
    };

    /**
     * @brief Atomic owner-level view of one admitted allocator ledger.
     *
     * This value is produced while the materialization mutex is held, so its
     * commitment and live-allocation columns describe one coherent instant.
     * Diagnostics and setup gates consume this typed view instead of joining
     * independent allocator telemetry, PerfStats counters, or repeated ledger
     * queries that can race asynchronous setup workers.
     */
    struct PhysicalMemoryOwnerAttestation
    {
        PhysicalMemoryAllocatorIdentity identity; ///< Physical allocator.
        PhysicalMemoryOwner owner =
            PhysicalMemoryOwner::Count; ///< Exact BOM owner line.
        std::size_t planned_new_bytes = 0u; ///< Admitted new allocation.
        std::size_t planned_resident_bytes = 0u; ///< Admitted retained credit.
        std::size_t materialized_new_bytes = 0u; ///< Currently live new bytes.
        std::size_t committed_new_bytes = 0u; ///< Direct plus reserved capacity.
        std::size_t adopted_resident_bytes = 0u; ///< Live retained allocation.

        /** @return Whether every admitted byte is physically live now. */
        [[nodiscard]] bool complete() const noexcept
        {
            return materialized_new_bytes == planned_new_bytes &&
                   adopted_resident_bytes == planned_resident_bytes;
        }

        /** @return Whether every admitted byte has a unique live/reserved owner. */
        [[nodiscard]] bool committed() const noexcept
        {
            return committed_new_bytes == planned_new_bytes &&
                   adopted_resident_bytes == planned_resident_bytes;
        }
    };

    namespace detail
    {
        struct PhysicalMemoryMaterializationState;
        struct PhysicalMemoryReservationState;
    }

    /**
     * @brief Move-only RAII proof for one currently live physical allocation.
     *
     * Destroying the lease returns its exact owner bytes to the ledger. Runtime
     * owners retain this token beside the allocation, so reclamation cannot
     * leave accounting stale and an asynchronous owner cannot outlive its
     * topology-wide authority.
     */
    class PhysicalMemoryAllocationLease final
    {
    public:
        PhysicalMemoryAllocationLease() = default;
        ~PhysicalMemoryAllocationLease();

        PhysicalMemoryAllocationLease(
            const PhysicalMemoryAllocationLease &) = delete;
        PhysicalMemoryAllocationLease &operator=(
            const PhysicalMemoryAllocationLease &) = delete;
        PhysicalMemoryAllocationLease(
            PhysicalMemoryAllocationLease &&other) noexcept;
        PhysicalMemoryAllocationLease &operator=(
            PhysicalMemoryAllocationLease &&other) noexcept;

        /** @return Whether this token currently owns a ledger claim. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Exact physical bytes represented by this live claim. */
        [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

        /** @return Typed owner against which the bytes were admitted. */
        [[nodiscard]] PhysicalMemoryOwner owner() const noexcept
        {
            return owner_;
        }

    private:
        friend class PhysicalMemoryMaterializationLedger;

        PhysicalMemoryAllocationLease(
            std::shared_ptr<detail::PhysicalMemoryMaterializationState> state,
            std::size_t resource_index,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind,
            std::size_t bytes);

        /** @brief Release a live claim exactly once. */
        void release() noexcept;

        std::shared_ptr<detail::PhysicalMemoryMaterializationState> state_;
        std::size_t resource_index_ = 0u;
        PhysicalMemoryOwner owner_ = PhysicalMemoryOwner::Count;
        PhysicalMemoryMaterializationKind kind_ =
            PhysicalMemoryMaterializationKind::NewAllocation;
        std::size_t bytes_ = 0u;
    };

    /**
     * @brief Move-only proof for one live allocation inside an owner reserve.
     *
     * The parent reserve commits capacity once against the topology BOM. Each
     * child lease then records the physical bytes currently materialized inside
     * that capacity without charging the owner line a second time. A child
     * retains the reservation state, so destroying the public reservation
     * handle cannot make accounting disappear before its allocation is freed.
     */
    class PhysicalMemorySuballocationLease final
    {
    public:
        PhysicalMemorySuballocationLease() = default;
        ~PhysicalMemorySuballocationLease();

        PhysicalMemorySuballocationLease(
            const PhysicalMemorySuballocationLease &) = delete;
        PhysicalMemorySuballocationLease &operator=(
            const PhysicalMemorySuballocationLease &) = delete;
        PhysicalMemorySuballocationLease(
            PhysicalMemorySuballocationLease &&other) noexcept;
        PhysicalMemorySuballocationLease &operator=(
            PhysicalMemorySuballocationLease &&other) noexcept;

        /** @return Whether this token represents live physical bytes. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Exact physical bytes represented by this child claim. */
        [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    private:
        friend class PhysicalMemoryOwnerReservation;

        PhysicalMemorySuballocationLease(
            std::shared_ptr<detail::PhysicalMemoryReservationState> state,
            std::size_t bytes);

        /** @brief Return the child bytes to their retained reserve once. */
        void release() noexcept;

        std::shared_ptr<detail::PhysicalMemoryReservationState> state_;
        std::size_t bytes_ = 0u;
    };

    /**
     * @brief Typed capacity authority for a lazily populated owner pool.
     *
     * Fixed blocks use @ref PhysicalMemoryAllocationLease directly. Pools whose
     * population changes over their lifetime first reserve an exact capacity,
     * then obtain one child lease before each backing allocation. Reservation
     * and materialization are deliberately distinct observables: committed
     * bytes prevent another allocator from consuming the same BOM capacity,
     * while claimed bytes report how much memory is physically live now.
     *
     * This handle is move-only and thread-safe. Destroying it closes the pool
     * to new children; existing children retain the committed capacity until
     * the final physical allocation is retired.
     */
    class PhysicalMemoryOwnerReservation final
    {
    public:
        PhysicalMemoryOwnerReservation() = default;
        ~PhysicalMemoryOwnerReservation() = default;

        PhysicalMemoryOwnerReservation(
            const PhysicalMemoryOwnerReservation &) = delete;
        PhysicalMemoryOwnerReservation &operator=(
            const PhysicalMemoryOwnerReservation &) = delete;
        PhysicalMemoryOwnerReservation(
            PhysicalMemoryOwnerReservation &&) noexcept = default;
        PhysicalMemoryOwnerReservation &operator=(
            PhysicalMemoryOwnerReservation &&) noexcept = default;

        /** @return Whether this handle can issue child allocation claims. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Exact new-allocation capacity committed to this pool. */
        [[nodiscard]] std::size_t capacityBytes() const noexcept;

        /** @return Bytes currently represented by live child leases. */
        [[nodiscard]] std::size_t materializedBytes() const;

        /** @return Capacity still available for a child allocation. */
        [[nodiscard]] std::size_t remainingBytes() const;

        /** @return Typed BOM owner of every child in this pool. */
        [[nodiscard]] PhysicalMemoryOwner owner() const noexcept;

        /**
         * @brief Pre-claim one physical allocation inside this reserve.
         *
         * Call this immediately before the backing allocator. If allocation
         * fails, destroy the returned lease; if it succeeds, retain the lease
         * beside the pointer and destroy it only after that pointer is freed.
         *
         * @throws std::invalid_argument for zero bytes or a closed reserve.
         * @throws std::logic_error when the child would exceed capacity.
         */
        [[nodiscard]] PhysicalMemorySuballocationLease claimAllocation(
            std::size_t bytes) const;

    private:
        friend class PhysicalMemoryMaterializationLedger;

        explicit PhysicalMemoryOwnerReservation(
            std::shared_ptr<detail::PhysicalMemoryReservationState> state);

        std::shared_ptr<detail::PhysicalMemoryReservationState> state_;
    };

    /**
     * @brief Thread-safe live-allocation ledger derived from one admission proof.
     *
     * Setup workers may claim independent allocations concurrently. Each claim
     * is bounded by its precise owner line; bytes cannot move between owners or
     * resources and retained credits cannot masquerade as new allocations.
     */
    class PhysicalMemoryMaterializationLedger final
    {
    public:
        /**
         * @brief Create an empty live ledger for the shared admitted plan.
         *
         * A value-copy constructor is deliberately absent: admission and
         * materialization must retain the same authority object identity.
         */
        explicit PhysicalMemoryMaterializationLedger(
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
                certificate);

        /** @return Exact immutable authority from which claims are issued. */
        [[nodiscard]] const std::shared_ptr<
            const PhysicalMemoryPlanAdmissionCertificate> &admission()
            const noexcept;

        /**
         * @brief Claim bytes newly allocated after the capacity observation.
         * @throws std::invalid_argument for zero bytes or invalid owner.
         * @throws std::out_of_range for an absent physical resource.
         * @throws std::logic_error when the owner line would be exceeded.
         */
        [[nodiscard]] PhysicalMemoryAllocationLease claimNewAllocation(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner,
            std::size_t bytes);

        /**
         * @brief Adopt bytes already resident in the certified owner line.
         * @throws std::logic_error when retained credit would be exceeded.
         */
        [[nodiscard]] PhysicalMemoryAllocationLease adoptResidentAllocation(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner,
            std::size_t bytes);

        /**
         * @brief Commit capacity for a lazily populated new-allocation pool.
         * @throws std::logic_error when direct claims and reservations would
         *         exceed the owner's admitted incremental bytes.
         */
        [[nodiscard]] PhysicalMemoryOwnerReservation reserveNewAllocations(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner,
            std::size_t bytes);

        /** @return Currently live bytes of one kind for an owner/resource. */
        [[nodiscard]] std::size_t claimedBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind) const;

        /** @return Capacity committed to lazy new allocations for one owner. */
        [[nodiscard]] std::size_t reservedBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner) const;

        /**
         * @return Direct live bytes plus reserved capacity for one claim kind.
         *
         * New-allocation reservations count at full capacity; their child
         * allocations are already contained within that capacity. Adopted
         * residency has no lazy form and therefore equals its live claim.
         */
        [[nodiscard]] std::size_t committedBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind) const;

        /**
         * @return Admitted new-allocation bytes not held by a direct claim or
         *         lazy-pool reservation.
         */
        [[nodiscard]] std::size_t remainingAdmittedNewAllocationBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner) const;

        /** @return Whether every certified owner byte has a live RAII claim. */
        [[nodiscard]] bool complete() const;

        /**
         * @brief Require complete live materialization before graph publication.
         * @throws std::logic_error with owner/resource diagnostics when partial.
         */
        void requireComplete() const;

        /** @return Whether every owner byte is live or explicitly reserved. */
        [[nodiscard]] bool committed() const;

        /**
         * @brief Require complete capacity commitment before publication.
         * @throws std::logic_error with owner/resource diagnostics when partial.
         */
        void requireCommitted() const;

        /**
         * @brief Snapshot every admitted owner under one ledger lock.
         *
         * Zero-byte lines are included deliberately. A complete fixed-size
         * owner table makes setup reports deterministic and lets tests prove
         * that a newly introduced owner cannot disappear from attestation.
         */
        [[nodiscard]] std::vector<PhysicalMemoryOwnerAttestation>
        attestation() const;

    private:
        std::shared_ptr<detail::PhysicalMemoryMaterializationState> state_;

        /** @brief Shared checked implementation for new and retained claims. */
        [[nodiscard]] PhysicalMemoryAllocationLease claim(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind,
            std::size_t bytes);
    };

    /**
     * @brief Rank-bound production authority for all live CPU/GPU allocations.
     *
     * Admission is topology-wide so several logical features cannot each fit
     * independently against the same allocator. Materialization is rank-local:
     * one process may claim only the CPU and accelerator resources physically
     * owned by its MPI rank. This object binds those two scopes once and is the
     * dependency passed to concrete allocators; callers never carry a loose
     * certificate, ledger, and integer rank that could disagree.
     */
    class PhysicalMemoryAuthority final
    {
    public:
        /**
         * @brief Bind a fitting topology admission to one materializing rank.
         * @param admission Shared immutable topology-wide proof.
         * @param world_rank Rank whose physical allocators this process owns.
         * @throws std::invalid_argument for a null proof, negative rank, or a
         *         plan with no resource owned by that rank.
         */
        PhysicalMemoryAuthority(
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
                admission,
            int world_rank);

        PhysicalMemoryAuthority(const PhysicalMemoryAuthority &) = delete;
        PhysicalMemoryAuthority &operator=(
            const PhysicalMemoryAuthority &) = delete;

        /** @return MPI rank whose physical allocators may be claimed. */
        [[nodiscard]] int worldRank() const noexcept { return world_rank_; }

        /** @return Exact topology-wide admission object retained by the ledger. */
        [[nodiscard]] const std::shared_ptr<
            const PhysicalMemoryPlanAdmissionCertificate> &admission()
            const noexcept;

        /** @return Whether this rank's plan contains @p device. */
        [[nodiscard]] bool contains(DeviceId device) const noexcept;

        /**
         * @brief Return complete planned bytes for one local owner/device.
         * @throws std::out_of_range when the device is not admitted for this rank.
         */
        [[nodiscard]] std::size_t plannedBytes(
            DeviceId device,
            PhysicalMemoryOwner owner) const;

        /**
         * @brief Claim a newly allocated local CPU/GPU region.
         * @return Move-only lease that releases the live claim on destruction.
         */
        [[nodiscard]] PhysicalMemoryAllocationLease claimNewAllocation(
            DeviceId device,
            PhysicalMemoryOwner owner,
            std::size_t bytes);

        /**
         * @brief Adopt one concretely retained local allocation.
         * @return Move-only lease bounded by the owner's certified residency.
         */
        [[nodiscard]] PhysicalMemoryAllocationLease adoptResidentAllocation(
            DeviceId device,
            PhysicalMemoryOwner owner,
            std::size_t bytes);

        /** @brief Commit a local owner's lazy new-allocation pool capacity. */
        [[nodiscard]] PhysicalMemoryOwnerReservation reserveNewAllocations(
            DeviceId device,
            PhysicalMemoryOwner owner,
            std::size_t bytes);

        /** @return Currently live bytes for one local owner and claim kind. */
        [[nodiscard]] std::size_t claimedBytes(
            DeviceId device,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind) const;

        /** @return Local lazy-pool capacity committed for one owner. */
        [[nodiscard]] std::size_t reservedBytes(
            DeviceId device,
            PhysicalMemoryOwner owner) const;

        /** @return Local bytes committed by direct claims or pool capacity. */
        [[nodiscard]] std::size_t committedBytes(
            DeviceId device,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind) const;

        /**
         * @return Rank-local owner capacity still available for a new claim.
         *
         * This is an admission-ledger query, not allocator telemetry. The
         * subsequent allocation must still claim its exact bytes because
         * concurrent setup owners may consume capacity after this snapshot.
         */
        [[nodiscard]] std::size_t remainingAdmittedNewAllocationBytes(
            DeviceId device,
            PhysicalMemoryOwner owner) const;

        /** @return Whether every owner planned on this rank has a live lease. */
        [[nodiscard]] bool rankComplete() const;

        /**
         * @brief Require complete rank-local materialization before publication.
         * @throws std::logic_error naming the first partial owner line.
         */
        void requireRankComplete() const;

        /** @return Whether every local owner byte is live or reserved. */
        [[nodiscard]] bool rankCommitted() const;

        /** @brief Require every local owner byte to be live or reserved. */
        void requireRankCommitted() const;

        /**
         * @brief Snapshot the owner lines physically materialized by this rank.
         * @return Deterministic resource-major, owner-major attestation rows.
         */
        [[nodiscard]] std::vector<PhysicalMemoryOwnerAttestation>
        rankAttestation() const;

        /**
         * @brief Render the coherent rank attestation for setup diagnostics.
         * @param include_zero_lines Whether owners with no admitted/live bytes
         *        should be shown.
         * @return One deterministic line per selected physical owner.
         */
        [[nodiscard]] std::string rankAttestationSummary(
            bool include_zero_lines = false) const;

    private:
        /** @return Checked rank-local allocator identity for @p device. */
        [[nodiscard]] PhysicalMemoryAllocatorIdentity identity(
            DeviceId device) const;

        int world_rank_ = -1;
        PhysicalMemoryMaterializationLedger ledger_;
    };
} // namespace llaminar2
