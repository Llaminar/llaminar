/**
 * @file PooledImmutableDirectedChannelGeometry.h
 * @brief One identity-free authority for the existing phase-reused directed channel capacities.
 *
 * Runtime Pin/Free bindings and pre-quota BOM inputs share these exact maxima.
 * Geometry contains no table identity, mutable live count or admission token.
 * The mandatory envelope retains protocol metadata at C0; actual device-counted
 * payload extents remain owned by the existing captured transfer publications.
 */
#pragma once
#include "kernels/qwen4exp/PooledImmutableNativePlan.h"
#include "transfer/CapturedTransferChannel.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
/** @brief Exact two-channel shape; allocation capacity never becomes traffic extent. */
class PooledImmutableDirectedChannelGeometry final {
public:
    /** @brief Project the exact runtime source/owner plans through the shared original cohort check. */
    static PooledImmutableDirectedChannelGeometry fromPlans(const PooledImmutableCoordinatorNativePlan& c,
        const PooledImmutableOwnerNativePlan& o){
        c.requireOwnerPlan(o);return shape(o.parameters().native.reply_capacity,o.parameters().row_bytes,c.parameters().policy.participants(),
            ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(c.allocator(), o.allocator(), o.peerAccess()));
    }
    /** @brief Project final physical capacities before an actual table/channel identity exists. */
    static PooledImmutableDirectedChannelGeometry fromGeometry(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate){
        const auto offsets=PooledImmutableOwnerNativePlan::rootOffsets(physical,coordinator,coordinate);
        return shape(offsets.reply_capacity,physical.rowBytes(),coordinator.parameters().participants(),
            ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(physical.coordinator(), physical.owners()[coordinate].allocator, physical.peerAccess()));
    }
    /** @brief Price the same ordered protocol population with no optional payload bank. */
    static PooledImmutableDirectedChannelGeometry mandatory(const PooledImmutablePhysicalOwnerGeometry& physical,
        const PooledImmutableCoordinatorGeometry& coordinator,std::uint32_t coordinate){
        (void)PooledImmutableOwnerNativePlan::rootOffsets(physical,coordinator,coordinate);
        return shape(0,physical.rowBytes(),coordinator.parameters().participants(),
            ImmutableRowTableDirectedTransportPolicy::usesPeerPayload(physical.coordinator(), physical.owners()[coordinate].allocator, physical.peerAccess()));
    }
    /** @return Original maximum coordinator-to-owner publication capacity. */
    std::size_t forwardMaximum()const noexcept{return forward_;}
    /** @return Original maximum owner-to-coordinator publication capacity. */
    std::size_t reverseMaximum()const noexcept{return reverse_;}
    /** @return Existing TransferEngine physical allocation geometry for the forward channel. */
    CapturedTransferChannelMemory forwardMemory()const{return CapturedTransferChannel::memoryFor(forward_, payload_storage_);}
    /** @return Existing TransferEngine physical allocation geometry for the return channel. */
    CapturedTransferChannelMemory reverseMemory()const{return CapturedTransferChannel::memoryFor(reverse_, payload_storage_);}
private:
    /** @brief Preserve existing invocation/free/source/publication/read/retire maxima in one source. */
    static PooledImmutableDirectedChannelGeometry shape(std::uint32_t replies,std::size_t rowBytes,std::uint32_t participants, bool peer){
        namespace n=pooled_native;namespace p=adaptive_lookup::pooled;
        if(!rowBytes || !participants)throw std::invalid_argument("Directed channel requires original row/cohort geometry");
        const auto product=[](std::size_t a,std::size_t b){if(a && b>std::numeric_limits<std::size_t>::max()/a)throw std::overflow_error("Directed channel extent overflow");return a*b;};
        const auto payload=product(replies,rowBytes);
        return {std::max({sizeof(n::Invocation),sizeof(n::FreePlanHeader),sizeof(n::SourceFrontier),product(replies,sizeof(n::Fill)),payload,
                         product(replies,sizeof(p::OwnerCommand)),product(participants,sizeof(p::FreeAllocationReceipt)),product(participants,sizeof(p::PublishHeader))}),
                std::max({sizeof(n::Availability),sizeof(p::FreeAllocationReceipt),sizeof(n::OwnerSourceSealReceipt),payload,
                         product(replies,sizeof(p::ReleaseReceipt)),product(replies,sizeof(p::Grant))}),
                peer ? CapturedTransferPayloadStorage::PeerDevice : CapturedTransferPayloadStorage::MappedHost};
    }
    PooledImmutableDirectedChannelGeometry(std::size_t forward,std::size_t reverse, CapturedTransferPayloadStorage storage)
        :forward_(forward),reverse_(reverse),payload_storage_(storage){}
    std::size_t forward_,reverse_;
    CapturedTransferPayloadStorage payload_storage_;
};
}
