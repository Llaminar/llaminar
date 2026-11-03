/**
 * @file CapturedTransferChannelReplayProbe.h
 * @brief Closed native fixture access to original channel storage after exact event joins.
 *
 * Payload poisoning occurs only while the original lane is idle. Protocol and
 * cursor observations are terminal test assertions, never live scheduling or a
 * second progress authority. Production has no host observation entry point.
 */
#pragma once
#include "transfer/CapturedTransferChannel.h"

namespace llaminar2
{
    /** @brief Borrow original owners solely for test setup and terminal observations. */
    struct CapturedTransferChannelReplayProbe
    {
        /** @return Exact payload; poison only before launch and inspect after original event joins. */
        static std::span<std::byte> payload(const CapturedTransferChannel &channel)
        {
            if(channel.payloadStorage()!=CapturedTransferPayloadStorage::MappedHost)
                throw std::invalid_argument("Peer payload requires an original device observation");
            return {static_cast<std::byte *>(channel.mapped_->mutableHostData(CapturedTransferChannel::payload_offset)),
                channel.capacityBytes()};
        }
        /** @return Original producer VRAM slab; native tests poison before launch and observe only after exact terminal joins. */
        static std::shared_ptr<DeviceTransferBuffer> terminalPeerPayload(const CapturedTransferChannel& channel)
        {
            if(channel.payloadStorage()!=CapturedTransferPayloadStorage::PeerDevice || !channel.peer_payload_)
                throw std::invalid_argument("Native peer probe requires its original producer payload owner");
            return channel.peer_payload_;
        }
        /** @return Original mapped control; callers may read only after every submitted endpoint joined. */
        static const CapturedTransferChannelControl &terminalControl(const CapturedTransferChannel &channel)
        {
            return *static_cast<const CapturedTransferChannelControl *>(channel.mapped_->mutableHostData());
        }
        /**
         * @brief Retain the exact original cursor for a terminal TransferEngine readback.
         * @param channel Original same-PMA channel.
         * @param endpoint Original producer or consumer role.
         * @return Original physical cursor owner, never copied/replaced protocol storage.
         * @throws std::invalid_argument For an unknown endpoint.
         */
        static std::shared_ptr<DeviceTransferBuffer> terminalCursor(const CapturedTransferChannel &channel,
            CapturedTransferEndpoint endpoint)
        {
            if(endpoint != CapturedTransferEndpoint::Producer && endpoint != CapturedTransferEndpoint::Consumer)
                throw std::invalid_argument("Native cancellation probe requires an original endpoint");
            return channel.endpoints_[endpoint == CapturedTransferEndpoint::Producer ? 0 : 1].cursor;
        }
    };
}
