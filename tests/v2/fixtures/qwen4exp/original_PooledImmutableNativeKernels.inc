/**
 * @file PooledImmutableNativeKernels.cpp
 * @brief Checked native callers binding exact persistent roots, action schemas and producer spans.
 *
 * Public enqueue never allocates, transfers, waits, rebinds storage or observes
 * live values. The graph executor declares/publishes actual canonical roots and
 * acquires original producer edges. Setup uses only its exact once-only stream.
 */
#include "PooledImmutableNativeKernels.h"
#include "PooledImmutableNativeOperands.h"
#include "PooledImmutableNativeRootStorage.h"
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
namespace n=pooled_native;
/** @brief Reject every native default-stream spelling before further work. */
void stream(void* exact)
{
    if(!exact || reinterpret_cast<std::uintptr_t>(exact)<=2)
        throw std::invalid_argument("Pooled native phase requires its exact non-default stream");
}
/** @return Exact checked span intersection without arithmetic wrap. */
bool overlaps(const void* x,std::size_t a,const void* y,std::size_t b)
{
    if(!a || !b)return false;const auto first=reinterpret_cast<std::uintptr_t>(x),second=reinterpret_cast<std::uintptr_t>(y);
    if(!x || !y || first>~std::uintptr_t{0}-a || second>~std::uintptr_t{0}-b)return true;
    return first<second+b && second<first+a;
}
/** @return A read view belongs entirely to one named admitted phase input bank. */
bool contained(std::size_t offset,std::size_t bytes,std::size_t first,std::size_t capacity)
{return offset>=first && offset-first<=capacity && bytes<=capacity-(offset-first);}
/**
 * @brief Reject hidden root aliasing, allowing only named phase-owned incoming frame/header banks.
 * @note Immutable invocation must remain external through the whole frame;
 * payload/count/control aliases can never overwrite actual physical authority.
 */
void ownerAliases(const PooledImmutableOwnerOperands& inputs,const PooledImmutableNativeRoots& roots,const n::OwnerParameters& plan)
{
    const std::array<std::shared_ptr<TensorBase>,3> writable{roots.cache,roots.frame,roots.native};
    for(std::size_t role=0;role<inputs.spans().size();++role){
        if(!inputs.spans()[role])continue;const auto& input=*inputs.spans()[role];
        const auto* data=static_cast<const std::byte*>(input.root->gpu_data_ptr())+input.offset;
        for(std::size_t root=0;root<writable.size();++root){
            if(!writable[root] || !overlaps(data,input.bytes,writable[root]->gpu_data_ptr(),writable[root]->size_bytes()))continue;
            bool allowed=false;
            if(input.root==writable[root] && root==1){
                if(role==static_cast<std::size_t>(PooledNativeOperand::Records) ||
                    role==static_cast<std::size_t>(PooledNativeOperand::Payload))
                    allowed=contained(input.offset,input.bytes,plan.frame.commands,std::size_t{plan.owner.maximumUnique()}*sizeof(adaptive_lookup::pooled::OwnerCommand));
                if(role==static_cast<std::size_t>(PooledNativeOperand::Frontier))
                    allowed=contained(input.offset,input.bytes,plan.frame.pin_headers,std::size_t{plan.owner.participants()}*sizeof(adaptive_lookup::pooled::PinHeader)) ||
                        contained(input.offset,input.bytes,plan.frame.publish_headers,std::size_t{plan.owner.participants()}*sizeof(adaptive_lookup::pooled::PublishHeader));
            }
            if(input.root==writable[root] && root==2 && role==static_cast<std::size_t>(PooledNativeOperand::Frontier))
                allowed=contained(input.offset,input.bytes,plan.native.header,64);
            if(!allowed)throw std::invalid_argument("Pooled owner operand aliases physical authority or an incompatible retained bank");
        }
    }
}
/** @brief Coordinator inputs must remain independently acquired observations, outside its mutable authority roots. */
void coordinatorAliases(const PooledImmutableCoordinatorOperands& inputs,const PooledImmutableNativeRoots& roots)
{
    const std::array<std::shared_ptr<TensorBase>,3> writable{roots.cache,roots.frame,roots.native};
    for(const auto& input:inputs.spans())if(input){
        const auto* data=static_cast<const std::byte*>(input->root->gpu_data_ptr())+input->offset;
        for(const auto& root:writable)if(root && overlaps(data,input->bytes,root->gpu_data_ptr(),root->size_bytes()))
            throw std::invalid_argument("Pooled coordinator input aliases its retained policy/frame/output authority");
    }
}
}
void IPooledImmutableNativeKernels::initialize(PooledImmutableOwnerPreparedStorage& storage,void* exact)const
{
    stream(exact);const auto& plan=storage.plan_.parameters();
    storage.roots_->initialize(backend(),exact,[&]{
        initializeOwner({plan,storage.roots_->address(0),storage.roots_->address(1),storage.roots_->address(2),{}},exact);
    });
}
void IPooledImmutableNativeKernels::initialize(PooledImmutableCoordinatorPreparedStorage& storage,void* exact)const
{
    stream(exact);const auto& plan=storage.plan_.parameters();
    storage.roots_->initialize(backend(),exact,[&]{
        initializeCoordinator({plan,storage.roots_->address(0),storage.roots_->address(1),storage.roots_->address(2)},storage.plan_.owners(),exact);
    });
}
void IPooledImmutableNativeKernels::enqueue(n::OwnerAction action,PooledImmutableOwnerPreparedStorage& storage,
    const PooledImmutableOwnerOperands& inputs,void* exact)const
{
    stream(exact);storage.roots_->requireInitialized();inputs.require(action,storage.plan_);
    if(backend()!=storage.plan_.parameters().device.type)throw std::invalid_argument("Pooled owner phase selected a foreign backend bundle");
    ownerAliases(inputs,storage.roots(),storage.plan_.parameters());
    launch(action,{storage.plan_.parameters(),storage.roots_->address(0),storage.roots_->address(1),storage.roots_->address(2),inputs.native()},exact);
}
void IPooledImmutableNativeKernels::enqueue(n::CoordinatorAction action,PooledImmutableCoordinatorPreparedStorage& storage,
    const PooledImmutableCoordinatorOperands& inputs,void* exact)const
{
    stream(exact);storage.roots_->requireInitialized();inputs.require(action,storage.plan_);
    if(backend()!=storage.plan_.parameters().device.type)throw std::invalid_argument("Pooled coordinator phase selected a foreign backend bundle");
    coordinatorAliases(inputs,storage.roots());
    const n::CoordinatorArgs args{storage.plan_.parameters(),storage.roots_->address(0),storage.roots_->address(1),storage.roots_->address(2),inputs.native()};
    if(action==n::CoordinatorAction::Begin){const auto query=inputs.query();launch(action,args,&query,exact);}
    else launch(action,args,nullptr,exact);
}
}
