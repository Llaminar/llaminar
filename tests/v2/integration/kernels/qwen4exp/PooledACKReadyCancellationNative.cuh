/**
 * @file PooledACKReadyCancellationNative.cuh
 * @brief Authenticate exact source operands and cancel after ACK/fill before Ready.
 *
 * Inspection reads immutable launch metadata before instantiation. The retained
 * diagnostic node uses current device state, no host probe or payload read/copy.
 */
#pragma once
#include "PooledACKReadyCancellation.h"
#include "kernels/common/SamplingMath.h"
#include <cstring>
#include <stdexcept>
#include <vector>
namespace llaminar2::test::pooled_ack::ACK_CANCEL_NAMESPACE {
namespace q=qwen4exp;namespace n=q::pooled_native;namespace s=q::pooled_source;namespace p=n::p;namespace a=q::adaptive_lookup;
/** @brief Reject incompatible original graph metadata before setup mutation. */
inline void require(bool value,const char* reason){if(!value)throw std::runtime_error(reason);}
/** @brief Check actual source consumption and acknowledged original reservation at the selected frontier. */
template<class Slot> __device__ bool validReservation(const s::DeviceArgs& x,unsigned unique){
    const auto view=n::coordinatorView(x.coordinator);const auto& row=view.unique[unique];
    if(!x.plan.owner.owner.capacity())return row.phase==p::UniquePhase::SourceOnly;
    if(row.phase!=p::UniquePhase::FreeGranted && row.phase!=p::UniquePhase::VictimGranted)return false;
    const auto owner=n::ownerView<Slot>(x.owner);const auto& grant=row.grant;
    if(grant.slot>=x.plan.owner.owner.capacity())return false;
    const auto& slot=owner.slots[grant.slot];
    return slot.row==900 && slot.generation==grant.generation &&
        a::PhysicalSlot<Slot>::state(slot)==a::SlotState::Reserved && (slot.flags&a::FramePin) &&
        n::progress(x.owner).delivery==n::Delivery::Copied && owner.frame->phase==p::OwnerPhase::Publishable;
}
/** @brief Cancel only original target900 frames; warm/full/partial/empty/reset use the same retained graph. */
__global__ void cancelAfterACKBeforeReady(s::DeviceArgs x){
    if(blockIdx.x || threadIdx.x)return;
    const auto view=n::coordinatorView(x.coordinator);unsigned target=~0u;
    for(unsigned i=0;i<view.frame->unique_rows;++i)if(view.unique[i].row==900)target=i;
    if(target==~0u)return;
    const auto& io=s::protocol(x);const auto& frontier=s::frontier(x);
    const bool valid=*x.healthy && !*x.error && s::stage(x)==s::ResponseStage::Consumed &&
        frontier.status==a::Status::Success && frontier.disposition==n::SourceDisposition::Consumed &&
        io.idCount && io.nativeBytes==io.idCount*x.plan.owner.row_bytes && io.issuedSequence==io.consumedSequence &&
        (x.plan.owner.owner.rowWidth()==a::RowKeyWidth::UInt32?validReservation<a::Slot32>(x,target):validReservation<a::Slot64>(x,target));
    (void)sampling_math::fail_device_generation_control(x.control,valid?
        sampling_math::DeviceGenerationError::InvalidController:sampling_math::DeviceGenerationError::ImmutableStorageFailure);
}
/** @brief Original missing check replay: preserve the actual Ready edge while authorizing nothing from canonical health. */
__global__ void originalUncheckedAuthorization(s::DeviceArgs){ }
/** @return Complete original root/channel/controller identity, excluding irrelevant structure padding. */
inline bool same(const s::DeviceArgs& a,const s::DeviceArgs& b){
    return a.plan.identity==b.plan.identity && a.owner.plan.identity==b.owner.plan.identity &&
        a.coordinator.plan.identity==b.coordinator.plan.identity && a.owner.cache==b.owner.cache &&
        a.owner.frame==b.owner.frame && a.owner.native==b.owner.native && a.coordinator.cache==b.coordinator.cache &&
        a.coordinator.frame==b.coordinator.frame && a.coordinator.native==b.coordinator.native && a.frame==b.frame && a.native==b.native &&
        a.channel.base.request==b.channel.base.request && a.channel.base.completion==b.channel.base.completion &&
        a.channel.base.acknowledgement==b.channel.base.acknowledgement && a.channel.base.ids==b.channel.base.ids &&
        a.channel.base.nativeRows==b.channel.base.nativeRows && a.channel.base.table==b.channel.base.table &&
        a.channel.base.maximumIds==b.channel.base.maximumIds && a.channel.base.rowBytes==b.channel.base.rowBytes &&
        a.channel.intents==b.channel.intents && a.channel.acknowledgements==b.channel.acknowledgements &&
        a.healthy==b.healthy && a.error==b.error && a.control==b.control;
}
/** @brief Preserve every native production dependency and add one diagnostic predecessor to the exact authorization. */
inline void splice(IGPUGraphCapture& capture,const s::DeviceArgs& expected,void* stream,AuthorizationReplay replay){
    require(stream && reinterpret_cast<std::uintptr_t>(stream)>2 && expected.plan.coordinator.device.ACK_DEVICE_PREDICATE(),
        "Pooled ACK splice requires its exact original backend stream");
    auto* owner=dynamic_cast<ACK_CAPTURE*>(&capture);require(owner && owner->deviceOrdinal()==expected.plan.coordinator.device.ordinal,
        "Pooled ACK splice belongs to another capture participant");
    std::vector<GPUGraphKernelNodeInfo> records;std::string error;
    require(capture.inspectKernelNodes(records,&error),"Pooled ACK inspection failed");std::uintptr_t identity=0;
    for(const auto& record:records)if(record.name.find("pooled_source")!=std::string::npos && record.name.find("beforeReady")!=std::string::npos){
        require(record.name_resolved && !record.nesting_depth && record.function_identity && !identity,"Pooled ACK authorization is ambiguous");
        identity=record.function_identity;
    }
    require(identity,"Pooled ACK authorization is absent");const auto graph=owner->graph();std::size_t count=0;
    ACK_CHECK(ACK_RT(GraphGetNodes)(graph,nullptr,&count));std::vector<ACK_NODE> nodes(count);ACK_CHECK(ACK_RT(GraphGetNodes)(graph,nodes.data(),&count));
    ACK_NODE authorization{};
    for(const auto node:nodes){ACK_NODE_TYPE type{};ACK_CHECK(ACK_RT(GraphNodeGetType)(node,&type));if(type!=ACK_RT(GraphNodeTypeKernel))continue;
        const auto parameters=originalParameters(node);if(originalIdentity(parameters)!=identity)continue;
        require(!authorization && parameters.kernelParams && parameters.kernelParams[0],"Pooled ACK source launch is ambiguous");
        auto actual=expected;std::memcpy(&actual,parameters.kernelParams[0],sizeof(actual));
        require(same(actual,expected),"Pooled ACK source launch changed its original operands");authorization=node;
    }
    require(authorization,"Pooled ACK authorization has no original native node");std::size_t n=0;
    ACK_CHECK(ACK_DEPENDENCIES(authorization,nullptr,&n));require(n,"Pooled ACK authorization lost its acquired publication dependency");
    std::vector<ACK_NODE> dependencies(n);ACK_CHECK(ACK_DEPENDENCIES(authorization,dependencies.data(),&n));
    auto argument=expected;void* arguments[]{&argument};ACK_KERNEL_PARAMS addition{};
    addition.func=reinterpret_cast<void*>(cancelAfterACKBeforeReady);addition.gridDim=dim3(1);addition.blockDim=dim3(1);addition.kernelParams=arguments;
    ACK_NODE cancellation{};ACK_CHECK(ACK_RT(GraphAddKernelNode)(&cancellation,graph,dependencies.data(),n,&addition));
    ACK_CHECK(ACK_ADD_DEPENDENCIES(graph,&cancellation,&authorization,1));
    if(replay==AuthorizationReplay::OriginalUnchecked){
        // Authenticate the production node first, then replace only its missing
        // health check with the original unchecked behavior. Same operands,
        // predecessor, cancelled controller and owner Ready nodes remain.
        ACK_KERNEL_PARAMS unchecked{};unchecked.func=reinterpret_cast<void*>(originalUncheckedAuthorization);
        unchecked.gridDim=dim3(1);unchecked.blockDim=dim3(1);unchecked.kernelParams=arguments;
        ACK_CHECK(ACK_RT(GraphKernelNodeSetParams)(authorization,&unchecked));
    }

}
}
