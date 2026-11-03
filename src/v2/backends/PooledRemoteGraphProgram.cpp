/** @file PooledRemoteGraphProgram.cpp
 * @brief Validate complete source-owned remote geometry and immutable class/capture identities.
 */
#include "PooledRemoteGraphProgram.h"
#include "loaders/ImmutableRowTableLayoutGeometry.h"
#include "kernels/qwen4exp/PooledImmutableNativeTypes.h"
#include "transfer/CapturedTransferChannelProtocol.h"
#include <algorithm>
#include <stdexcept>
namespace llaminar2 {
PooledRemoteGraphArgumentLayout PooledRemoteGraphProgramClass::argumentLayout()noexcept{
    // Owner kernels take one OwnerArgs value. Fused/copy transfer kernels take
    // binding,destination,source; acquire kernels take the smaller binding alone.
    struct TransferArguments {CapturedTransferChannelDeviceBinding binding;void* destination;const void* source;};
    return {3,std::max(sizeof(qwen4exp::pooled_native::OwnerArgs),sizeof(TransferArguments)),
        std::max(alignof(qwen4exp::pooled_native::OwnerArgs),alignof(TransferArguments))};
}
std::string PooledRemoteGraphProgramClass::identity()const{
    constexpr char digits[]="0123456789abcdef";std::string out="pooled-directed-remote-owner-v1:";
    for(const auto byte:dispatch_){out+=digits[byte>>4];out+=digits[byte&15];}
    out+=':';out+=std::to_string(limits_.maximumNodes);out+=':';out+=std::to_string(limits_.maximumEdges);
    out+=':';out+=std::to_string(limits_.allowedKinds);return out;
}
PooledRemoteGraphProgram::PooledRemoteGraphProgram(PooledRemoteGraphGeometry geometry,
    std::array<std::uint8_t,32> dispatch,GPUFlatGraphLimits limits):geometry_(geometry),class_(dispatch,limits){
    const auto& g=geometry_;
    ImmutableRowTableDirectedTransportPolicy::require(g.coordinator, g.owner, g.peerAccess);
    if(!g.owner.device.is_valid() || !g.coordinator.device.is_valid() || !g.owner.device.is_gpu() || !g.coordinator.device.is_gpu() ||
       g.owner==g.coordinator || g.owner.world_rank!=g.coordinator.world_rank ||
       g.owner.world_rank < -1 || !g.participants || g.coordinate>=g.participants ||
       !g.maximumOriginalIds || !g.maximumUniqueIds || g.maximumUniqueIds>g.maximumOriginalIds ||
       !g.sourceRows || !g.rowBytes || g.slotStride<g.rowBytes || g.slotStride%4 ||
       !g.nativeLayoutIdentity || (g.rowKeyBits!=32 && g.rowKeyBits!=64) ||
       g.nonemptyRootMask!=(g.capacity?7u:6u) || !limits.valid() ||
       std::none_of(dispatch.begin(),dispatch.end(),[](auto byte){return byte!=0;}))
        throw std::invalid_argument("Remote graph program requires its complete original mapped-transport source geometry/dispatch");
}
std::uint64_t PooledRemoteGraphProgram::captureIdentity()const noexcept{
    std::uint64_t value=1469598103934665603ULL;
    const auto mix=[&](std::uint64_t item){value=(value^item)*1099511628211ULL;};
    const auto allocator=[&](PhysicalMemoryAllocatorIdentity a){mix(a.world_rank);mix(static_cast<unsigned>(a.device.type));mix(a.device.ordinal);};
    allocator(geometry_.coordinator);allocator(geometry_.owner);mix(geometry_.coordinate);mix(geometry_.participants);
    mix(geometry_.capacity);mix(geometry_.maximumOriginalIds);mix(geometry_.maximumUniqueIds);mix(geometry_.sourceRows);
    mix(geometry_.rowBytes);mix(geometry_.slotStride);mix(geometry_.nativeLayoutIdentity);mix(geometry_.rowKeyBits);mix(geometry_.nonemptyRootMask);
    mix(geometry_.peerAccess.has_value());
    if (geometry_.peerAccess) mix(static_cast<unsigned>(*geometry_.peerAccess));
    for(const auto byte:class_.dispatchSignature())mix(byte);
    mix(limits().maximumNodes);mix(limits().maximumEdges);mix(limits().allowedKinds);return value?value:1;
}
}
