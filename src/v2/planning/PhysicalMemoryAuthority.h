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
#include <span>
#include <string>
#include <vector>
#include <utility>

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
    class NativeStoragePmaBinding;
    class NativeStoragePreparationPmaBinding;
    class NativeMandatoryStorageDeclarations;
    class NativePreparationPlanningSnapshot;
    namespace test
    {
        struct NativeStoragePreparationReplayAccess;
    }

    /** @brief Immutable aggregate BOM for every physical allocator in a plan. */
    class PhysicalMemoryPlan final
    {
    public:
        /**
         * @return Deterministically ordered, unique physical-resource BOMs borrowed from this retained plan.
         * @note The caller retains the lvalue plan for the complete borrow. A
         * temporary plan cannot expose a vector reference beyond its lifetime.
         */
        [[nodiscard]] const std::vector<PhysicalMemoryBOM> &resources()
            const & noexcept
        {
            return resources_;
        }
        /** @brief Reject borrowing resource storage from a temporary plan, including const rvalues. */
        const std::vector<PhysicalMemoryBOM> &resources() const && noexcept = delete;

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

        /**
         * @brief Add the per-owner envelope of mutually exclusive setup plans.
         * @param alternatives Complete plans whose temporary owners retire before
         *        the next alternative executes. Shared source storage may persist.
         * @return This builder; existing concurrent contributions remain additive.
         * @throws std::invalid_argument for conflicting resources or residency credits.
         *
         * This is admission arithmetic, not permission to overlap executions.
         * The enclosing setup transaction must serialize these alternatives.
         * Retained credits require allocation identity and cannot be combined by
         * a maximum; this interface deliberately admits new storage only. Failure
         * leaves the builder unchanged, including when the final addition overflows.
         */
        PhysicalMemoryPlanBuilder &addMutuallyExclusive(
            std::span<const PhysicalMemoryPlan> alternatives);

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
         * @throws PhysicalMemoryCapacityExhausted when any valid resource is over budget.
         * @throws std::invalid_argument when the aggregate is empty or malformed.
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

    /** @brief One-way canonical authority lifecycle; cancellation never erases root ownership. */
    enum class PhysicalMemoryAuthorityPhase : std::uint8_t
    {
        Preparing,
        Admitted,
        Cancelled
    };
    /** @return Explicit canonical lifecycle spelling for passive source attestations. */
    [[nodiscard]] inline constexpr std::string_view toString(
        PhysicalMemoryAuthorityPhase phase) noexcept
    {
        switch (phase)
        {
        case PhysicalMemoryAuthorityPhase::Preparing:
            return "Preparing";
        case PhysicalMemoryAuthorityPhase::Admitted:
            return "Admitted";
        case PhysicalMemoryAuthorityPhase::Cancelled:
            return "Cancelled";
        }
        return "Invalid";
    }

    /** @brief Actual native source phase, independently frozen in each root descriptor. */
    enum class PhysicalMemoryNativeRootPhase : std::uint32_t
    {
        ColdProbe = 1,
        RetainedSetup = 2,
        GraphRecordingHostMetadata = 3
    };

    /**
     * @brief Immutable source-issued identity/geometry attached to an actual native root lease.
     * No pointer lookup, byte balance or alias registry is represented here.
     */
    struct PhysicalMemoryNativeRootDeclaration final
    {
        std::uint64_t owner_identity = 0; ///< Actual native root record, bound before Create.
        std::uint64_t shared_anchor_identity =
            0; ///< Stable source root-cell identity, never dereferenced by PMA.
        std::size_t requested_bytes = 0; ///< Exact native logical allocation request.
        std::size_t physical_bytes = 0;  ///< Actual allocator-rounded physical creation extent.
        std::size_t allocation_granularity =
            0; ///< Granularity from the selected native allocator properties.
        std::uint64_t allocator_flags =
            0; ///< Frozen native allocator/handle/access policy identity.
        DeviceId device =
            DeviceId::invalid(); ///< Physical allocator; host placement always uses CPU0.
        std::int32_t host_numa =
            -1; ///< Placement metadata, independently preserved from allocator identity.
        std::uint32_t owner_family = 0;   ///< SDK-issued actual native owner family.
        std::uint32_t lifetime = 0;       ///< SDK-issued source ownership/retirement lifetime.
        std::uint32_t allocator_kind = 0; ///< Exact selected allocator; no implicit substitution.
        PhysicalMemoryNativeRootPhase source_phase = PhysicalMemoryNativeRootPhase::RetainedSetup;
        bool operator==(const PhysicalMemoryNativeRootDeclaration &) const = default;
        /**
         * @brief Compare immutable geometry while permitting only actual owner binding.
         * @param other Actual callback declaration for this same source root cell.
         * @return Whether every frozen source field matches; only owner_identity is ignored.
         */
        [[nodiscard]] bool samePreparedGeometry(
            const PhysicalMemoryNativeRootDeclaration &other) const noexcept
        {
            auto left = *this;
            auto right = other;
            left.owner_identity = 0;
            right.owner_identity = 0;
            return left == right;
        }
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
        PhysicalMemoryAuthorityPhase phase =
            PhysicalMemoryAuthorityPhase::Admitted; ///< Atomic source admission phase.
        PhysicalMemoryOwner owner =
            PhysicalMemoryOwner::Count; ///< Exact BOM owner line.
        std::size_t planned_new_bytes = 0u; ///< Admitted new allocation.
        std::size_t planned_resident_bytes = 0u; ///< Admitted retained credit.
        std::size_t materialized_new_bytes = 0u; ///< Currently live new bytes.
        std::size_t committed_new_bytes = 0u; ///< Direct plus reserved capacity.
        std::size_t adopted_resident_bytes = 0u; ///< Live retained allocation.

        std::size_t preparation_committed_bytes =
            0u; ///< Exact source preparation backing, included once after admission.
        std::size_t preparation_materialized_bytes = 0u; ///< Published early physical roots.
        std::size_t native_declared_preclaimed_bytes =
            0u; ///< Unpublished native children within source BOM reservations.

        /** @return All canonical source commitments, including exact native preparation/SDK roots.
         */
        [[nodiscard]] std::size_t totalCommittedNewBytes() const noexcept
        {
            return committed_new_bytes;
        }
        /** @return All canonical physical source bytes; preparation/SDK children occur exactly
         * once. */
        [[nodiscard]] std::size_t totalMaterializedNewBytes() const noexcept
        {
            return materialized_new_bytes;
        }

        /** @return Whether every admitted byte is physically live now. */
        [[nodiscard]] bool complete() const noexcept
        {
            return phase == PhysicalMemoryAuthorityPhase::Admitted &&
                   materialized_new_bytes == planned_new_bytes &&
                   adopted_resident_bytes == planned_resident_bytes &&
                   native_declared_preclaimed_bytes == 0u;
        }

        /** @return Whether every admitted byte has a unique live/reserved owner. */
        [[nodiscard]] bool committed() const noexcept
        {
            return phase == PhysicalMemoryAuthorityPhase::Admitted &&
                   committed_new_bytes == planned_new_bytes &&
                   adopted_resident_bytes == planned_resident_bytes;
        }
    };

    class PhysicalMemoryOwnerReservation;

    namespace detail
    {
        struct PhysicalMemoryMaterializationState;
        struct PhysicalMemoryReservationState;
        struct PhysicalMemoryNativeSetupEpochState;
        struct PhysicalMemoryPreparationRootState;
        struct PhysicalMemoryNativeRootPermitState;
        struct PhysicalMemoryPreparationSnapshotState;
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
     * @brief Unique preclaim and physical-publication authority for one native root.
     *
     * The existing PMA state and immutable source demand remain retained until
     * the actual native allocation and every physical alias have retired. Only
     * this move-only lease can publish materialization or return its exact claim.
     */
    class PhysicalMemoryNativeSetupLease final
    {
    public:
        /** @brief Construct an inactive lease without claiming capacity. */
        PhysicalMemoryNativeSetupLease();
        /** @brief Return the exact preclaim after its native owner frees backing. */
        ~PhysicalMemoryNativeSetupLease();
        PhysicalMemoryNativeSetupLease(const PhysicalMemoryNativeSetupLease &) = delete;
        PhysicalMemoryNativeSetupLease &operator=(const PhysicalMemoryNativeSetupLease &) = delete;
        PhysicalMemoryNativeSetupLease(PhysicalMemoryNativeSetupLease &&other) noexcept;
        PhysicalMemoryNativeSetupLease &operator=(PhysicalMemoryNativeSetupLease &&other) noexcept;
        /** @return Whether this token uniquely retains its native-root preclaim. */
        [[nodiscard]] bool valid() const noexcept
        {
            return bool(epoch_) && bytes_ != 0;
        }
        /** @return Actual allocator-declared physical creation extent. */
        [[nodiscard]] std::size_t bytes() const noexcept
        {
            return bytes_;
        }
        /** @return Whether successful native creation has been published. */
        [[nodiscard]] bool materialized() const noexcept
        {
            return materialized_;
        }
        /**
         * @brief Publish successful physical creation before exposing any consumer.
         * @throws std::logic_error For inactive, duplicate or closed-epoch publication.
         */
        void materialize();

    private:
        friend class PhysicalMemoryNativeSetupEpoch;
        /** @brief Bind one already-reserved native root without another ledger. */
        PhysicalMemoryNativeSetupLease(
            std::shared_ptr<detail::PhysicalMemoryNativeSetupEpochState> epoch,
            std::size_t resource_index, PhysicalMemoryOwner owner, std::size_t bytes);
        /** @brief Return committed/materialized columns exactly once. */
        void release() noexcept;
        std::shared_ptr<detail::PhysicalMemoryNativeSetupEpochState> epoch_;
        std::shared_ptr<detail::PhysicalMemoryReservationState> declared_reservation_;
        std::unique_ptr<detail::PhysicalMemoryPreparationRootState> preparation_root_;
        std::shared_ptr<detail::PhysicalMemoryNativeRootPermitState> permit_;
        std::size_t resource_index_ = 0;
        PhysicalMemoryOwner owner_ = PhysicalMemoryOwner::Count;
        std::size_t bytes_ = 0;
        bool materialized_ = false;
    };

    /**
     * @brief Shared immutable setup identity with one-way capture/retirement closure.
     *
     * Callback copies share this exact epoch. Claim, materialization, closure
     * and retirement synchronize through the existing PMA materialization mutex.
     * A retired session is required before an independent source re-admission;
     * process-lived roots cannot be hidden in a second source authority.
     */
    class PhysicalMemoryNativeSetupEpoch final
    {
    public:
        /** @brief Construct an inactive lifecycle handle without creating an authority. */
        PhysicalMemoryNativeSetupEpoch() = default;
        /** @return Original same-session identity retained through final source admission. */
        [[nodiscard]] std::uint64_t identity() const noexcept;
        /** @brief Enter cold host recording after every required GPU root is prepared. */
        void beginHostRecording() const;
        /**
         * @brief Close native growth before executable publication/replay.
         * Repeated successful sealing is idempotent while all required roots remain published.
         * @throws std::logic_error For abandoned setup or missing/unpublished retained roots.
         */
        void sealForExecution() const;
        /** @return Whether growth has permanently closed for this native session. */
        [[nodiscard]] bool sealed() const;
        /** @return Whether every actual source/root/alias has physically retired. */
        [[nodiscard]] bool retired() const;
        /** @brief Reject independent source admission until final SDK root retirement. */
        void requireRetired() const;

    private:
        friend class PhysicalMemoryMaterializationLedger;
        friend class NativeStoragePreparationPmaBinding;
        friend class NativeMandatoryStorageDeclarations;
        /**
         * @brief Preclaim an exact source preparation root before native Create.
         * @param declaration SDK-issued CPU0 allocator geometry and stable actual owner anchor.
         * @return Unique physical-root lease retaining the original state and epoch.
         * @throws std::invalid_argument For malformed/duplicate creator declarations.
         * @throws PhysicalMemoryCapacityExhausted Before Create when original CPU0 capacity is
         * exhausted.
         * @throws std::logic_error If Preparing growth has closed.
         */
        [[nodiscard]] PhysicalMemoryNativeSetupLease claimPreparation(
            const PhysicalMemoryNativeRootDeclaration &declaration) const;
        /**
         * @brief Close early growth and snapshot actual declarations under the canonical mutex.
         * @return Immutable source floor/identity used before optional placement.
         * @throws std::logic_error For pending publication, cancellation or repeated closure.
         * Allocation failure leaves the original open phase and every token unchanged.
         */
        [[nodiscard]] std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState>
        closePreparationForPlanning() const;
        /**
         * @brief Cancel early setup while all actual tokens remain releasable.
         * @throws std::logic_error If final admission already transferred ownership.
         */
        void cancelPreparation() const;
        friend class NativeStoragePmaBinding;
        /**
         * @brief Close failed admitted setup without publishing an executable.
         * @throws std::logic_error For a source that never reached Admitted.
         * This changes no physical counters and does not recycle descriptor permits.
         * Abandoned setup can never publish an executable. Repeated abandonment is idempotent;
         * destruction after a successful executable seal preserves that successful state.
         */
        void abandonSetup() const;
        /**
         * @brief Consume one exact immutable descriptor before native Create.
         * @param declaration Actual native owner request, including its frozen source anchor.
         * @return Sole root lease after descriptor consumption and source preclaim commit
         * atomically.
         * @throws std::logic_error For changed/consumed descriptors or an incompatible setup phase.
         */
        [[nodiscard]] PhysicalMemoryNativeSetupLease claimPreparedNative(
            const PhysicalMemoryNativeRootDeclaration &declaration) const;
        /**
         * @brief Build future root permits without publishing inventory or capacity.
         * @param declarations Exact immutable SDK creator declarations, excluding existing
         * preparation roots.
         * @param reservations Prebuilt source owner lifetimes, still unpublished.
         * @return Stable descriptor objects attached to those same owner lifetimes.
         * @throws std::logic_error If source extents do not exactly cover their declared
         * reservation.
         */
        [[nodiscard]] std::vector<std::shared_ptr<detail::PhysicalMemoryNativeRootPermitState>>
        prepareNativePermits(std::span<const PhysicalMemoryNativeRootDeclaration> declarations,
                             const std::vector<PhysicalMemoryOwnerReservation> &reservations) const;
        /**
         * @brief Prebuild exact source reservations without publishing counters.
         * @param snapshot Original closed preparation identity and retained root floor.
         * @param mandatory Source-native creator BOM merged before optional fill.
         * @param final Complete immutable certificate using the original observation registry.
         * @return Unpublished owner handles; allocation failure leaves the source unchanged.
         * @throws std::logic_error For stale floors, missing mandatory storage or wrong lifecycle.
         */
        [[nodiscard]] std::vector<PhysicalMemoryOwnerReservation> prepareNativeAdmission(
            const std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> &snapshot,
            const PhysicalMemoryPlan &mandatory,
            const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &final) const;
        /**
         * @brief Publish the complete prebuilt final bundle atomically in the same state.
         * @param snapshot Original closed identity, revalidated under the one canonical mutex.
         * @param mandatory Exact SDK creator BOM already priced before optional fill.
         * @param final Immutable complete source certificate; never replaced after success.
         * @param reservations Future native source owner handles, currently unpublished.
         * @param permits Complete immutable future descriptor inventory with unconsumed lifecycle
         * state.
         * @param retainedPreparations Prebuilt exact original preparation declaration inventory.
         * @param context Private unpublished callback-binding transfer record.
         * @param arm Noexcept ownership/arm operation; it performs no native allocation or
         * callback.
         * @throws Before mutation for any stale/partial/changed source bundle.
         * After validation only noexcept pointer/vector moves, counters and arm remain.
         */
        void publishNativeAdmission(
            const std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> &snapshot,
            const PhysicalMemoryPlan &mandatory,
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> final,
            const std::vector<PhysicalMemoryOwnerReservation> &reservations,
            std::vector<std::shared_ptr<detail::PhysicalMemoryNativeRootPermitState>> permits,
            std::vector<PhysicalMemoryNativeRootDeclaration> retainedPreparations, void *context,
            void (*arm)(void *) noexcept) const;
        /** @brief Bind this handle to the sole canonical epoch state. */
        explicit PhysicalMemoryNativeSetupEpoch(
            std::shared_ptr<detail::PhysicalMemoryNativeSetupEpochState> epoch)
            : epoch_(std::move(epoch))
        {
        }
        /** @brief Admit one declared child under the same PMA mutex and epoch boundary. */
        std::shared_ptr<detail::PhysicalMemoryNativeSetupEpochState> epoch_;
    };

    /** @brief Closed source preparation snapshot; no additional live allocation ledger. */
    class NativePreparationPlanningSnapshot final
    {
    public:
        NativePreparationPlanningSnapshot(const NativePreparationPlanningSnapshot &) = delete;
        NativePreparationPlanningSnapshot &operator=(const NativePreparationPlanningSnapshot &) =
            delete;
        NativePreparationPlanningSnapshot(NativePreparationPlanningSnapshot &&) noexcept = default;
        NativePreparationPlanningSnapshot &operator=(
            NativePreparationPlanningSnapshot &&) noexcept = default;
        /**
         * @return Original single observation registry with stable resource indices.
         * @throws std::logic_error If final publication consumed this snapshot.
         */
        [[nodiscard]] std::span<const PhysicalMemoryResource> initialObservations() const;
        /**
         * @return Exact retained preparation backing, merged before optional placement.
         * @throws std::logic_error If final publication consumed this snapshot.
         */
        [[nodiscard]] const PhysicalMemoryPlan &retainedPreparationBOM() const;
        /** @return Original native session epoch; final callbacks retain this exact identity. */
        [[nodiscard]] std::uint64_t epochIdentity() const;
        /** @return Rank owning the original physical observations for this session. */
        [[nodiscard]] int worldRank() const;
        /** @return Whether this unconsumed snapshot still binds its original epoch. */
        [[nodiscard]] bool valid() const noexcept
        {
            return bool(snapshot_);
        }

    private:
        friend class NativeStoragePreparationPmaBinding;
        friend struct test::NativeStoragePreparationReplayAccess;
        friend class NativeMandatoryStorageDeclarations;
        /** @brief Bind immutable values captured under the canonical state mutex. */
        explicit NativePreparationPlanningSnapshot(
            std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> value)
            : snapshot_(std::move(value))
        {
        }
        /** @brief Exact actual-root declarations, accessible only to SDK creator projection. */
        [[nodiscard]] std::span<const PhysicalMemoryNativeRootDeclaration> declarations() const;
        std::shared_ptr<const detail::PhysicalMemoryPreparationSnapshotState> snapshot_;
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

        /** @return Immutable physical allocator identity retained by this exact reservation. */
        [[nodiscard]] PhysicalMemoryAllocatorIdentity allocatorIdentity() const noexcept;

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
        friend class PhysicalMemoryNativeSetupEpoch;
        friend class NativeStoragePreparationPmaBinding;

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
        [[nodiscard]] const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &
        admission() const;

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

        /**
         * @return Source-BOM live bytes of one kind for an owner/resource.
         * Exact preparation and SDK roots occur in these canonical source totals.
         */
        [[nodiscard]] std::size_t claimedBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind) const;

        /** @return Capacity committed to lazy new allocations for one owner. */
        [[nodiscard]] std::size_t reservedBytes(
            PhysicalMemoryAllocatorIdentity identity,
            PhysicalMemoryOwner owner) const;

        /**
         * @return Source-BOM direct live bytes plus reserved capacity for one claim kind.
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

        /**
         * @return New bytes still required by this allocator's admitted owners.
         *
         * Lazy reservations are commitments, not backing allocations. Their
         * unmaterialized children remain demand even when the whole owner pool
         * has already been reserved. The snapshot is taken under one ledger
         * lock and cannot join inconsistent per-owner queries.
         */
        [[nodiscard]] std::size_t unmaterializedNewBytes(
            PhysicalMemoryAllocatorIdentity identity) const;

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
        friend class PhysicalMemoryAuthority;
        friend class NativeStoragePreparationPmaBinding;
        /**
         * @brief Start the same canonical state without inventing an admission certificate.
         * @param observations Original unique allocator observations, canonically sorted internally
         * once.
         * @throws std::invalid_argument For conflicting/duplicate resources or unsupported CPU
         * ordinals.
         */
        explicit PhysicalMemoryMaterializationLedger(
            std::span<const PhysicalMemoryResource> observations);
        /** @brief Open the one native session epoch while central authority is Preparing. */
        [[nodiscard]] PhysicalMemoryNativeSetupEpoch beginPreparationEpoch(int world_rank);
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
         * @brief Apply a user ceiling to an observed allocator's admission capacity.
         * @param observed_available_bytes Immutable observation, not a subsystem balance.
         * @param explicit_limit_bytes Optional user constraint; zero deliberately admits no bytes.
         * @return Capacity supplied to the canonical BOM, never a safety reserve or live debit.
         */
        [[nodiscard]] static std::size_t admissionCapacity(
            std::size_t observed_available_bytes,
            std::optional<std::size_t> explicit_limit_bytes) noexcept;

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

        /** @return Canonical phase copied under the materialization mutex. */
        [[nodiscard]] PhysicalMemoryAuthorityPhase phase() const;

        /** @return MPI rank whose physical allocators may be claimed. */
        [[nodiscard]] int worldRank() const noexcept { return world_rank_; }

        /** @return Exact topology-wide admission object retained by the ledger. */
        [[nodiscard]] const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &
        admission() const;

        /** @return Whether this rank's plan contains @p device. */
        [[nodiscard]] bool contains(DeviceId device) const;

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

        /**
         * @brief Admit an exactly measured, context-lifetime native allocation.
         * @param observed Fresh same-rank physical capacity after the reversible
         *        driver-footprint probe has restored its original storage.
         * @param bytes Exact measured footprint, never a per-thread estimate.
         * @return Unique lease to retain beside the native context until reset.
         * @throws PhysicalMemoryCapacityExhausted when the context allocation
         *         plus every outstanding owner of this plan cannot fit.
         * @throws std::out_of_range for a GPU absent from this rank's admission.
         * @throws std::invalid_argument for changed allocator geometry, foreign
         *         rank, CPU storage, or a zero-byte footprint.
         *
         * The native context outlives a sample/model graph family, so it needs
         * its own immutable certificate and ledger lease. This authority first
         * protects the source plan's still-unmaterialized demand. It does not
         * transfer bytes out of graph reservations, enlarge a graph allowance,
         * or give the caller capacity arithmetic to duplicate. The lease keeps
         * its certificate alive after the source authority is destroyed.
         */
        [[nodiscard]] PhysicalMemoryAllocationLease claimMeasuredRuntimeContextStorage(
            PhysicalMemoryResource observed,
            std::size_t bytes) const;

        /**
         * @return Source-BOM live bytes for one local owner and claim kind.
         * Preparation and exact SDK roots occur in the same source-BOM totals.
         */
        [[nodiscard]] std::size_t claimedBytes(
            DeviceId device,
            PhysicalMemoryOwner owner,
            PhysicalMemoryMaterializationKind kind) const;

        /** @return Local lazy-pool capacity committed for one owner. */
        [[nodiscard]] std::size_t reservedBytes(
            DeviceId device,
            PhysicalMemoryOwner owner) const;

        /**
         * @return Source-BOM local commitment by direct claims or pool capacity.
         * Preparation and exact SDK roots occur in the same source-BOM totals.
         */
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
        friend class NativeStoragePmaBinding;
        friend class NativeStoragePreparationPmaBinding;
        struct PreparingNativeStorageTag
        {
        };
        /**
         * @brief Construct only the typed native preparation authority.
         * @param observations Original physical observations; no later free-memory sample is
         * accepted.
         * @param world_rank Rank owning the required CPU0 preparation allocator.
         * @throws std::invalid_argument If the rank has no original CPU0 resource.
         */
        PhysicalMemoryAuthority(std::span<const PhysicalMemoryResource> observations,
                                int world_rank, PreparingNativeStorageTag);
        /** @return Checked rank-local allocator identity for @p device. */
        [[nodiscard]] PhysicalMemoryAllocatorIdentity identity(
            DeviceId device) const;

        int world_rank_ = -1;
        PhysicalMemoryMaterializationLedger ledger_;
    };
} // namespace llaminar2
