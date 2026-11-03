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
#include "PooledImmutableNativeAliasPolicy.h"
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
/** @brief Metadata guard is shared verbatim with CPU regression; native addresses remain checked here. */
void ownerAliases(n::OwnerAction action,const PooledImmutableOwnerOperands& inputs,const PooledImmutableNativeRoots& roots,const PooledImmutableOwnerNativePlan& plan)
{
    detail::requireOwnerAliasDeclarations(action,inputs,roots,plan);
    const std::array<std::shared_ptr<TensorBase>,3> writable{roots.cache,roots.frame,roots.native};
    for(const auto& input:inputs.spans())if(input)for(const auto& root:writable)if(root && input->root!=root){
        const auto* data=static_cast<const std::byte*>(input->root->gpu_data_ptr())+input->offset;
        if(overlaps(data,input->bytes,root->gpu_data_ptr(),root->size_bytes()))
            throw std::invalid_argument("Pooled owner input physically overlaps another original authority root");
    }
}
/** @brief Retained receive banks are the only internal coordinator input aliases; foreign physical overlaps remain fatal. */
void coordinatorAliases(n::CoordinatorAction action,const PooledImmutableCoordinatorOperands& inputs,const PooledImmutableNativeRoots& roots,const PooledImmutableCoordinatorNativePlan& plan)
{
    detail::requireCoordinatorAliasDeclarations(action,inputs,roots,plan);
    const std::array<std::shared_ptr<TensorBase>,3> writable{roots.cache,roots.frame,roots.native};
    for(const auto& input:inputs.spans())if(input)for(const auto& root:writable)if(root && input->root!=root){
        const auto* data=static_cast<const std::byte*>(input->root->gpu_data_ptr())+input->offset;
        if(overlaps(data,input->bytes,root->gpu_data_ptr(),root->size_bytes()))
            throw std::invalid_argument("Pooled coordinator input physically overlaps another original authority root");
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
    ownerAliases(action,inputs,storage.roots(),storage.plan_);
    launch(action,{storage.plan_.parameters(),storage.roots_->address(0),storage.roots_->address(1),storage.roots_->address(2),inputs.native(),storage.invocationReceiveCompletion(exact)},exact);
}
void IPooledImmutableNativeKernels::enqueue(n::CoordinatorAction action,PooledImmutableCoordinatorPreparedStorage& storage,
    const PooledImmutableCoordinatorOperands& inputs,void* exact)const
{
    stream(exact);storage.roots_->requireInitialized();inputs.require(action,storage.plan_);
    if(backend()!=storage.plan_.parameters().device.type)throw std::invalid_argument("Pooled coordinator phase selected a foreign backend bundle");
    coordinatorAliases(action,inputs,storage.roots(),storage.plan_);
    const n::CoordinatorArgs args{storage.plan_.parameters(),storage.roots_->address(0),storage.roots_->address(1),storage.roots_->address(2),inputs.native()};
    if(action==n::CoordinatorAction::Begin){const auto query=inputs.query();launch(action,args,&query,exact);}
    else launch(action,args,nullptr,exact);
}
}
