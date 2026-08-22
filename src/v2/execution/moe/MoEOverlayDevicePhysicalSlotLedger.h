/**
 * @file MoEOverlayDevicePhysicalSlotLedger.h
 * @brief Physical allocation lifetimes for device-owned ExpertOverlay epochs.
 *
 * Device-resident policy deliberately has no host owner-map mirror.  The byte
 * transport still needs to retain the prepared engine triplets backing raw GPU
 * descriptors and to know when a departed slot can be recycled.  This ledger
 * records only those physical facts.  It cannot observe histograms, score a
 * placement, change a movement command, or publish a device runtime bank.
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
        MoEOverlayPreparedExpertTriplet triplet;

        /** @return Whether identity, epoch, and all three engines are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief One completed destination triplet awaiting device RCU publication. */
    struct MoEOverlayDeviceStagedPhysicalArrival
    {
        MoEOverlayDevicePhysicalSlotKey key;
        MoEOverlayPreparedExpertTriplet triplet;

        /** @return Whether identity and all prepared engines are complete. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /** @brief A departed physical allocation returned after reader retirement. */
    struct MoEOverlayDeviceRetiredPhysicalSlot
    {
        MoEOverlayDevicePhysicalSlotKey key;
        std::uint64_t entered_epoch = 0u;
        bool bootstrap_allocation = false;
        MoEOverlayPreparedExpertTriplet triplet;
    };

    /**
     * @brief Thread-safe physical lifetime follower for device-owned Dynamic RCU.
     *
     * Lifecycle is strictly `begin -> stage -> publish -> retire`, or
     * `begin -> abort`. `publish` advances physical epoch identity only after
     * destination triplets are complete; `retire` releases sources only after
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
            std::vector<int> local_participant_ids;
            std::vector<MoEOverlayDeviceInitialPhysicalSlot> initial_slots;
        };

        /**
         * @brief Validate and enroll the immutable initial physical inventory.
         * @throws std::invalid_argument for invalid epochs, participants,
         *         duplicate slots, or incomplete prepared engines.
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
         * @return Copy of the shared triplet lifetime, or no value on rejection.
         */
        [[nodiscard]] std::optional<MoEOverlayPreparedExpertTriplet>
        sourceTriplet(
            const MoEOverlayDevicePhysicalMovementBatch &batch,
            const MoEOverlayDevicePhysicalSlotKey &key,
            std::string *error = nullptr) const noexcept;

        /**
         * @brief Retain every completed local destination before RCU apply.
         * @param batch Exact active batch identity.
         * @param arrivals Exactly one complete triplet per local destination.
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

        /** @return Durable physical epoch currently represented by active slots. */
        [[nodiscard]] std::uint64_t currentEpoch() const noexcept;

        /** @return Number of process-local active physical expert allocations. */
        [[nodiscard]] std::size_t activeSlotCount() const noexcept;

        /** @return Whether a device transaction currently owns the inactive set. */
        [[nodiscard]] bool hasPendingWave() const noexcept;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace llaminar2
