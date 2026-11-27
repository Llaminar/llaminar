/**
 * @file CUDAPeerPushResources.h
 * @brief PMA-owned receiver-local P2P inbox storage for the original NCCL participant.
 *
 * Exact CUDA sibling of ROCmPeerPushResources. Every allocation consumes the
 * original LocalCollective PMA reservation before creation and retains its
 * suballocation lease until physical release. Creates no extra worker,
 * communicator, parallel memory ledger, or replay-time allocation.
 */
#pragma once

#include "collective/PeerPushReduction.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <memory>
#include <span>

namespace llaminar2
{
    /** @brief One original CUDA participant's cold materialization and peer-directory publication. */
    class CUDAPeerPushResources final
    {
    public:
        /**
         * @brief Materialize exact quoted bytes on the already-selected CUDA participant.
         * @param device Original physical owner matching the PMA reservation.
         * @param reserve Canonical LocalCollective device capacity.
         * @param layout Immutable common progress/storage geometry.
         * @param rank Original communicator index.
         * @param abort Device-readable abort word owned by the enclosing coordinator attachment.
         * @param stream Explicit original setup stream; capture is rejected.
         * @return Unpublished storage awaiting one cold peer-directory publication.
         */
        static std::unique_ptr<CUDAPeerPushResources> create(DeviceId device,
            const PhysicalMemoryOwnerReservation &reserve, PeerPushLayout layout, int rank,
            const volatile std::uint32_t *abort, void *stream);

        /** @brief Release physical backing before returning its PMA leases. */
        ~CUDAPeerPushResources();
        CUDAPeerPushResources(const CUDAPeerPushResources &) = delete;
        CUDAPeerPushResources &operator=(const CUDAPeerPushResources &) = delete;

        /** @return Immutable local addresses for the cold common peer-directory publication. */
        [[nodiscard]] PeerPushEndpoint endpoint() const noexcept { return endpoint_; }

        /** @brief Publish the complete peer directory once before graph capture. */
        void publish(std::span<const PeerPushEndpoint> peers, void *stream);

        /** @return Frozen graph binding; unpublished or failed storage cannot execute. */
        [[nodiscard]] const PeerPushBinding &binding() const;

    private:
        CUDAPeerPushResources(DeviceId device, PeerPushLayout layout, int rank,
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
} // namespace llaminar2
