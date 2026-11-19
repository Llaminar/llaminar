/**
 * @file ROCmPeerPushResources.h
 * @brief PMA-owned receiver-local storage for the original RCCL participant.
 *
 * This is the collective infrastructure's explicit uncached-allocation boundary.
 * Every allocation consumes the original LocalCollective reservation before
 * creation and keeps its child lease until physical free. It creates no worker,
 * communicator, capacity ledger, host execution counter or replay allocation.
 */
#pragma once
#include "collective/PeerPushReduction.h"
#include "planning/PhysicalMemoryAuthority.h"
#include <memory>
#include <span>

namespace llaminar2
{
    /** @brief One original participant's one-shot cold materialization and publication. */
    class ROCmPeerPushResources final
    {
    public:
        /**
         * @brief Materialize exact quoted bytes on the already selected original worker.
         * @param device Original physical owner, matching the PMA reservation.
         * @param reserve Canonical LocalCollective capacity; no second reservation is made.
         * @param layout Immutable common progress/storage geometry.
         * @param rank Original communicator index.
         * @param abort Borrowed original native abort word, retained by the enclosing coordinator.
         * @param stream Explicit original setup stream; capture is rejected.
         * @return Unpublished storage that must receive the complete peer directory once.
         * @throws std::exception For invalid ownership, admission or native creation failure.
         */
        static std::unique_ptr<ROCmPeerPushResources> create(DeviceId device,
            const PhysicalMemoryOwnerReservation &reserve, PeerPushLayout layout, int rank,
            const volatile std::uint32_t *abort, void *stream);

        /** @brief Free backing before its PMA claims; the coordinator retires all consumers first. */
        ~ROCmPeerPushResources();
        ROCmPeerPushResources(const ROCmPeerPushResources &) = delete;
        ROCmPeerPushResources &operator=(const ROCmPeerPushResources &) = delete;

        /** @return Immutable local addresses for the cold common peer-directory publication. */
        [[nodiscard]] PeerPushEndpoint endpoint() const noexcept { return endpoint_; }

        /**
         * @brief Publish the complete peer directory and initial flags before capture.
         * @param peers Exact original communicator order, including this endpoint unchanged.
         * @param stream Same explicit stream that materialized this participant.
         * @throws std::exception For duplicate publication, missing peers or native errors.
         * The single cold join closes initialization; no hot-path join is needed.
         */
        void publish(std::span<const PeerPushEndpoint> peers, void *stream);

        /** @return Frozen graph binding; unpublished/failed storage cannot execute. */
        [[nodiscard]] const PeerPushBinding &binding() const;

    private:
        /** @brief Bind identities before any allocation, so partial creation always has an owner. */
        ROCmPeerPushResources(DeviceId device, PeerPushLayout layout, int rank,
                             const volatile std::uint32_t *abort, void *stream);
        enum class State { Materializing, Unpublished, Publishing, Ready };
        DeviceId device_;
        void *stream_ = nullptr;
        void *slab_ = nullptr;
        PeerPushEndpoint *table_ = nullptr;
        PeerPushEndpoint endpoint_;
        PeerPushBinding binding_;
        PhysicalMemorySuballocationLease slab_lease_;
        PhysicalMemorySuballocationLease table_lease_;
        State state_ = State::Materializing;
    };
}
