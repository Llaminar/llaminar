/**
 * @file MoEOverlayDevicePhysicalSlotLedger.h
 * @brief Physical allocation lifetimes for device-owned ExpertOverlay epochs.
 *
 * Device-resident policy deliberately has no host owner-map mirror.  The byte
 * transport still needs to retain the prepared engine payloads backing raw GPU
 * descriptors and to know when a departed slot can be recycled.  This ledger
 * records only those physical facts.  It cannot observe histograms, score a
 * placement, change a movement command, or publish a device runtime bank.
 * One closed internal payload alternative retains either the ordinary declared
 * engine family or the original authenticated F fragment. Separate typed public
 * receipts prevent fragment engines from certifying whole-expert semantics.
 * Whole and fragment coordinates may coexist in one ledger; their fixed source
 * families cannot change when logical experts migrate between those targets.
 */

#pragma once

#include "MoEOverlayDevicePhysicalMovement.h"
#include "MoEOverlayParticipantResidency.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <variant>

namespace llaminar2
{
    /** @brief Stable process-local identity of one physical expert allocation. */
    struct MoEOverlayDevicePhysicalSlotKey
    {
        int participant_id = -1;
        int layer_idx = -1;
        int expert_id = -1;

        /** @return Whether every dense physical coordinate is non-negative. */
        [[nodiscard]] bool valid() const noexcept
        {
            return participant_id >= 0 && layer_idx >= 0 && expert_id >= 0;
        }

        bool operator==(
            const MoEOverlayDevicePhysicalSlotKey &) const = default;
        bool operator<(
            const MoEOverlayDevicePhysicalSlotKey &other) const noexcept;
    };

    /** @brief One setup-enrolled physical source and its prepared lifetimes. */
    struct MoEOverlayDeviceInitialPhysicalSlot
    {
        MoEOverlayDevicePhysicalSlotKey key;
        std::uint64_t entered_epoch = 0u;
        bool bootstrap_allocation = true;
        MoEOverlayPreparedExpertPayload payload;

        /** @return Whether identity, epoch, and the declared payload family are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief One completed destination payload awaiting device RCU publication. */
    struct MoEOverlayDeviceStagedPhysicalArrival
    {
        MoEOverlayDevicePhysicalSlotKey key;
        MoEOverlayPreparedExpertPayload payload;

        /** @return Whether identity and all prepared engines are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief A departed physical allocation returned after reader retirement. */
    struct MoEOverlayDeviceRetiredPhysicalSlot
    {
        MoEOverlayDevicePhysicalSlotKey key;
        std::uint64_t entered_epoch = 0u;
        bool bootstrap_allocation = false;
        MoEOverlayPreparedExpertPayload payload;
    };

    /**
     * @brief One immutable entry in a quiescent device-owned physical epoch.
     *
     * Shared prepared-engine ownership makes this snapshot cheap: copying an
     * entry retains its declared engine lifetimes but never copies weight bytes. The
     * terminal model-context sealer uses these entries instead of consulting a
     * deliberately stale host participant-bank mirror.
     */
    struct MoEOverlayDeviceActivePhysicalSlot
    {
        MoEOverlayDevicePhysicalSlotKey key;
        std::uint64_t entered_epoch = 0u;
        bool bootstrap_allocation = false;
        MoEOverlayPreparedExpertPayload payload;

        /** @return Whether identity, birth epoch, and engine payload are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Complete process-local physical inventory at one durable epoch.
     *
     * This value contains allocation lifetimes only. It cannot express an
     * owner policy, histogram, runtime selector, or candidate transaction.
     */
    struct MoEOverlayDevicePhysicalInventorySnapshot
    {
        std::uint64_t epoch = 0u;
        std::vector<MoEOverlayDeviceActivePhysicalSlot> slots;

        /** @return Whether the epoch is positive and every retained slot is valid. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Typed fragment allocation receipt; never convertible to a whole expert. */
    struct MoEOverlayDeviceFragmentPhysicalSlot
    {
        MoEOverlayDevicePhysicalSlotKey key;
        std::uint64_t entered_epoch = 0u;
        bool bootstrap_allocation = false;
        MoEOverlayPreparedFragmentPayload payload;

        /** @return Whether this receipt retains a complete authenticated fragment. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Completed local fragment awaiting the same device RCU publication. */
    struct MoEOverlayDeviceStagedFragmentArrival
    {
        MoEOverlayDevicePhysicalSlotKey key;
        MoEOverlayPreparedFragmentPayload payload;
    };

    /** @brief Quiescent fragment inventory, retaining original immutable provenance. */
    struct MoEOverlayDeviceFragmentInventorySnapshot
    {
        std::uint64_t epoch = 0u;
        std::vector<MoEOverlayDeviceFragmentPhysicalSlot> slots;

        /** @return Whether the epoch and every retained original fragment coordinate are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief Closed physical lifetime; a fragment never acquires whole-expert semantics. */
    using MoEOverlayDevicePreparedPhysicalPayload =
        std::variant<MoEOverlayPreparedExpertPayload, MoEOverlayPreparedFragmentPayload>;

    /** @brief Exact local member receipt for a mixed whole/cohort wave. */
    struct MoEOverlayDevicePreparedPhysicalArrival
    {
        MoEOverlayDevicePhysicalSlotKey key;
        MoEOverlayDevicePreparedPhysicalPayload payload;
    };

    /** @brief Allocation lifetime retained until the original old-reader barrier. */
    struct MoEOverlayDevicePreparedPhysicalSlot
    {
        MoEOverlayDevicePhysicalSlotKey key;
        std::uint64_t entered_epoch = 0u;
        bool bootstrap_allocation = false;
        MoEOverlayDevicePreparedPhysicalPayload payload;
    };

    /** @brief Quiescent mixed inventory; payload alternatives preserve original family. */
    struct MoEOverlayDevicePreparedPhysicalInventory
    {
        std::uint64_t epoch = 0u;
        std::vector<MoEOverlayDevicePreparedPhysicalSlot> slots;
    };

    /**
     * @brief Thread-safe physical lifetime follower for device-owned Dynamic RCU.
     *
     * Lifecycle is strictly `begin -> stage -> publish -> retire`, or
     * `begin -> abort`. `publish` advances physical epoch identity only after
     * destination payloads are complete; `retire` releases sources only after
     * the device controller's old-reader barrier. Exactly one wave may exist,
     * matching the controller's one-transaction ABI.
     */
    class MoEOverlayDevicePhysicalSlotLedger final
    {
    public:
        /** @brief Complete setup-time enrollment of process-local allocations. */
        struct Config
        {
            std::uint64_t initial_epoch = 0u;
            /** Sorted local overlay IDs. A rank outside the overlay follows each epoch with empty inventory. */
            std::vector<int> local_participant_ids;
            std::vector<MoEOverlayDeviceInitialPhysicalSlot> initial_slots;
            /** Immutable family accepted for sources and every future arrival. */
            DeviceMoEProjectionSet movable_projections = DeviceMoEProjectionSet::CompleteExpert;
            /** Immutable global layer interval, including ranks with no local slots. */
            int first_model_layer = 0;
            std::uint32_t num_layers = 0u;
            /** Exact per-layer expert geometry of the admitted runtime. */
            std::uint32_t num_experts = 0u;
        };

        /** @brief One original registry-authenticated local participant/layer family. */
        struct FragmentFamily
        {
            int participant_id = -1;
            int layer_idx = -1;
            std::shared_ptr<const MoEOverlayFragmentFamily> family;
        };

        /** @brief Closed fragment enrollment; ordinary payloads cannot inhabit this arm. */
        struct FragmentConfig
        {
            std::uint64_t initial_epoch = 0u;
            std::vector<int> local_participant_ids;
            std::vector<MoEOverlayDeviceFragmentPhysicalSlot> initial_slots;
            std::vector<FragmentFamily> families;
            int first_model_layer = 0;
            std::uint32_t num_layers = 0u;
            std::uint32_t num_experts = 0u;
        };

        /**
         * @brief One physical ledger with fixed whole and original fragment coordinates.
         * Whole rows are those absent from families; each declared fragment row
         * retains its original source contract even while no expert is resident.
         */
        struct MixedConfig
        {
            Config whole;
            std::vector<FragmentFamily> families;
            std::vector<MoEOverlayDeviceFragmentPhysicalSlot> initial_fragments;
        };

        /**
         * @brief Enroll mixed targets without creating another storage authority.
         * @param config Exact whole inventory and immutable original fragment rows.
         * @return One ledger following the existing begin/stage/publish/retire lifecycle.
         * @throws std::invalid_argument For conflicting coordinates or foreign families.
         */
        [[nodiscard]] static std::unique_ptr<MoEOverlayDevicePhysicalSlotLedger> createMixed(MixedConfig config);

        /**
         * @brief Stage every exact local member in one atomic mixed wave.
         * @param batch Same authenticated movement identity as begin.
         * @param arrivals Complete original-member receipts, with no duplicate or foreign key.
         * @param error Rejection diagnostic; failure leaves the wave unstaged.
         * @return True only after all local obligations are complete.
         */
        [[nodiscard]] bool stagePrepared(const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<MoEOverlayDevicePreparedPhysicalArrival> arrivals, std::string *error = nullptr) noexcept;

        /**
         * @brief Retire mixed source lifetimes after the original reader barrier.
         * @param batch Same published movement identity.
         * @param retired Output retaining every departed local source.
         * @param error Rejection diagnostic; no source is removed on rejection.
         * @return True after the original wave becomes terminal.
         */
        [[nodiscard]] bool retirePrepared(const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<MoEOverlayDevicePreparedPhysicalSlot> *retired, std::string *error = nullptr) noexcept;

        /**
         * @brief Borrow exact quiescent inventory without flattening payload alternatives.
         * @param epoch Exact durable epoch, with no wave still active.
         * @param error Rejection diagnostic.
         * @return Immutable shared physical lifetimes, never a selected-owner mirror.
         */
        [[nodiscard]] std::optional<MoEOverlayDevicePreparedPhysicalInventory> preparedSnapshot(
            std::uint64_t epoch, std::string *error = nullptr) const noexcept;

        /**
         * @brief Borrow only fixed source metadata for a participant/layer row.
         * @param participant Original global member identity.
         * @param layer Original model layer.
         * @return Original retained family, or null for a whole-expert row.
         */
        [[nodiscard]] std::shared_ptr<const MoEOverlayFragmentFamily> fragmentFamily(int participant, int layer) const noexcept;

        /**
         * @brief Retain original fragment families and enroll complete local sources.
         * @param config Exact model geometry, local members and authenticated families.
         * @throws std::invalid_argument For foreign families, coordinates, epochs or payloads.
         */
        [[nodiscard]] static std::unique_ptr<MoEOverlayDevicePhysicalSlotLedger> createFragments(FragmentConfig config);

        /**
         * @brief Borrow a fragment source without converting its projection semantics.
         * @param batch Exact pending movement identity.
         * @param key Original local source coordinate named by that movement.
         * @param error Optional rejection diagnostic.
         * @return Retained original fragment, or no value on a wrong arm or identity.
         */
        [[nodiscard]] std::optional<MoEOverlayPreparedFragmentPayload> sourceFragmentPayload(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            const MoEOverlayDevicePhysicalSlotKey &key, std::string *error = nullptr) const noexcept;

        /**
         * @brief Stage exactly the local fragment arrivals of the authenticated wave.
         * @param batch Exact pending movement identity.
         * @param arrivals Registry-sealed fragments for this candidate epoch.
         * @param error Optional rejection diagnostic.
         * @return True only when every original destination family is complete.
         */
        [[nodiscard]] bool stageFragments(const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<MoEOverlayDeviceStagedFragmentArrival> arrivals,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Release fragment sources only after the caller's old-reader barrier.
         * @param batch Exact published movement identity.
         * @param retired Receives the original source lifetimes.
         * @param error Optional rejection diagnostic.
         * @return True when the same pending wave becomes terminal.
         */
        [[nodiscard]] bool retireFragments(const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<MoEOverlayDeviceFragmentPhysicalSlot> *retired,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Borrow the quiescent inventory through the fragment-only interface.
         * @param epoch Exact current durable epoch.
         * @param error Optional rejection diagnostic.
         * @return Complete fragment inventory, or no value for a wrong arm/lifecycle.
         */
        [[nodiscard]] std::optional<MoEOverlayDeviceFragmentInventorySnapshot> fragmentSnapshot(
            std::uint64_t epoch, std::string *error = nullptr) const noexcept;

        /**
         * @brief Validate and enroll the immutable initial physical inventory.
         * @throws std::invalid_argument for invalid epochs, participants,
         *         stage geometry, foreign/duplicate slots, or incomplete engines.
         */
        explicit MoEOverlayDevicePhysicalSlotLedger(Config config);

        /** @brief Destroy only after no background movement call is active. */
        ~MoEOverlayDevicePhysicalSlotLedger();

        MoEOverlayDevicePhysicalSlotLedger(
            const MoEOverlayDevicePhysicalSlotLedger &) = delete;
        MoEOverlayDevicePhysicalSlotLedger &operator=(
            const MoEOverlayDevicePhysicalSlotLedger &) = delete;

        /**
         * @brief Reserve lifecycle identity for one exact durable command wave.
         * @param batch Valid non-empty device-authored Dynamic movement.
         * @param error Optional exact rejection diagnostic.
         * @return True when every local source exists and destination is free.
         */
        [[nodiscard]] bool begin(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Resolve a retained local source for asynchronous byte movement.
         * @param batch Exact active batch identity.
         * @param key Source coordinate named by one migration.
         * @param error Optional exact rejection diagnostic.
         * @return Copy of the shared payload lifetime, or no value on rejection.
         */
        [[nodiscard]] std::optional<MoEOverlayPreparedExpertPayload>
        sourcePayload(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            const MoEOverlayDevicePhysicalSlotKey &key,
            std::string *error = nullptr) const noexcept;

        /**
         * @brief Retain every completed local destination before RCU apply.
         * @param batch Exact active batch identity.
         * @param arrivals Exactly one complete payload per local destination.
         * @param error Optional exact rejection diagnostic.
         * @return True after the inactive physical inventory is complete.
         */
        [[nodiscard]] bool stage(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<MoEOverlayDeviceStagedPhysicalArrival> arrivals,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Make staged destinations durable before device RCU publication.
         * @param batch Exact active batch identity.
         * @param error Optional exact rejection diagnostic.
         * @return True after physical epoch identity advances atomically.
         *
         * This is allocation-lifetime bookkeeping only. It cannot mutate the
         * device runtime selector, so inference continues on the base epoch
         * until the controller's bounded publication epoch admits E+1.
         */
        [[nodiscard]] bool publish(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Remove old sources after every device reader has drained.
         * @param batch Exact published batch identity.
         * @param retired Receives lifetimes whose destruction may recycle slots.
         * @param error Optional exact rejection diagnostic.
         * @return True after the transaction becomes terminal and reusable.
         */
        [[nodiscard]] bool retire(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::vector<MoEOverlayDeviceRetiredPhysicalSlot> *retired,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Discard an unpublished wave after its operations are aborted.
         * @param batch Exact begun or staged batch identity.
         * @param error Optional exact rejection diagnostic.
         * @return True when no staged arrival can later be published.
         */
        [[nodiscard]] bool abort(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            std::string *error = nullptr) noexcept;

        /**
         * @brief Retain an immutable copy of the current quiescent inventory.
         * @param epoch Exact durable epoch certified by the device controller.
         * @param error Optional exact lifecycle or inventory diagnostic.
         * @return Complete process-local snapshot, or no value on rejection.
         *
         * A snapshot is legal only after the transaction pipeline has drained:
         * the requested epoch must equal the ledger's current epoch and no
         * begun/staged/published wave may remain. Copying shared payloads does
         * not copy, download, synchronize, or repack their weight bytes.
         */
        [[nodiscard]] std::optional<
            MoEOverlayDevicePhysicalInventorySnapshot>
        snapshot(
            std::uint64_t epoch,
            std::string *error = nullptr) const noexcept;

        /** @return Durable physical epoch currently represented by active slots. */
        [[nodiscard]] std::uint64_t currentEpoch() const noexcept;

        /** @return Number of process-local active physical expert allocations. */
        [[nodiscard]] std::size_t activeSlotCount() const noexcept;

        /** @return Whether a device transaction currently owns the inactive set. */
        [[nodiscard]] bool hasPendingWave() const noexcept;

    private:
        struct MixedEnrollment {};
        /** @brief Enroll original immutable families before accepting mixed physical arrivals. */
        MoEOverlayDevicePhysicalSlotLedger(MixedConfig config, MixedEnrollment);
        struct FragmentEnrollment {};
        /** @brief Private named-factory constructor keeps existing aggregate Config calls unambiguous. */
        MoEOverlayDevicePhysicalSlotLedger(FragmentConfig config, FragmentEnrollment);
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace llaminar2
