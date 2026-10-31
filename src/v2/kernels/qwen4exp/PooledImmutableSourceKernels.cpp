/**
 * @file PooledImmutableSourceKernels.cpp
 * @brief Exact-stream original-source callers with canonical root/runtime and controller seals.
 *
 * Setup alone materializes metadata. Captured callers allocate, rebind, read back
 * and synchronize nothing; source publication/acquisition and last-reader edges
 * stay visible as nominal graph operations. The actual service retains backing.
 */
#include "PooledImmutableSourceKernels.h"
#include "PooledImmutableSourcePreparedStorage.h"
#include "PooledImmutableNativeRootStorage.h"
#include "execution/mtp/DeviceGenerationControlBinding.h"
#include <limits>
#include <stdexcept>
namespace llaminar2::qwen4exp {
namespace {
/** @brief Reject every default stream spelling before touching native state. */
void exact(void* stream){if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2)throw std::invalid_argument("Pooled source requires its exact non-default stream");}
/** @return Checked native root address; no allocation or coherence operation occurs. */
std::byte* address(const std::shared_ptr<TensorBase>& root){return root?static_cast<std::byte*>(root->gpu_data_ptr()):nullptr;}
/** @return Exact nonwrapping overlap, including invalid pointer spans. */
bool overlaps(const void* x,std::size_t bytes,const void* y,std::size_t count){
    if(!bytes || !count)return false;
    const auto a=reinterpret_cast<std::uintptr_t>(x),b=reinterpret_cast<std::uintptr_t>(y);
    return !a || !b || bytes>~std::uintptr_t{0}-a || count>~std::uintptr_t{0}-b || (a<b+count && b<a+bytes);
}
}
void IPooledImmutableSourceKernels::requireStorage(const PooledImmutableSourcePreparedStorage& storage)const{
    if(storage.geometry().parameters().identity!=geometry().parameters().identity)
        throw std::invalid_argument("Pooled source native caller names a foreign original storage registration");
    source().requireFor(geometry());storage.captureIdentity();
}
pooled_source::DeviceArgs IPooledImmutableSourceKernels::args(const PooledImmutableSourcePreparedStorage& storage,
    const DeviceGenerationControlBinding* control)const{
    const auto own=storage.owner()->roots(),coord=storage.coordinator()->roots(),root=storage.roots();
    return {geometry().parameters(),{geometry().parameters().owner,address(own.cache),address(own.frame),address(own.native),{}},
        {geometry().parameters().coordinator,address(coord.cache),address(coord.frame),address(coord.native),{}},
        address(root.frame),address(root.native),source().deviceView(),control?control->healthy():nullptr,control?control->error():nullptr,control?control->row():nullptr};
}
void IPooledImmutableSourceKernels::initialize(PooledImmutableSourcePreparedStorage& storage,void* stream)const{
    exact(stream);requireStorage(storage);
    storage.roots_->initialize(geometry().allocator().device.type,stream,[&]{initializeNative(args(storage,nullptr),stream);});
}
void IPooledImmutableSourceKernels::admit(PooledImmutableSourcePreparedStorage& storage,std::uint64_t generation,
    std::uint64_t invocation,void* stream)const{
    exact(stream);requireStorage(storage);storage.roots_->requireInitialized();
    if(isGraphCaptureActive() || !generation || !invocation)throw std::invalid_argument("Pooled source admission requires the real positive service generation and worker invocation outside capture");
    admitNative(args(storage,nullptr),generation,invocation,stream);
    TransferEngine::publishDeviceWrite(storage.roots().native.get(),geometry().allocator().device,stream);
}
void IPooledImmutableSourceKernels::enqueue(PooledSourceAction action,PooledImmutableSourcePreparedStorage& storage,
    const DeviceGenerationControlBinding& control,PooledSourceQuery query,void* stream)const{
    exact(stream);requireStorage(storage);storage.roots_->requireInitialized();control.requireFor(geometry().allocator().device);
    const auto& plan=geometry().parameters();
    if(action!=PooledSourceAction::Plan && action!=PooledSourceAction::Publish && action!=PooledSourceAction::Acquire &&
       action!=PooledSourceAction::Read && action!=PooledSourceAction::ReadersRetired && action!=PooledSourceAction::ObserveRetired && action!=PooledSourceAction::BeforeReady)
        throw std::invalid_argument("Unknown pooled source operation");
    const auto count=std::size_t{plan.coordinator.policy.maximumOriginal()}*plan.row_elements;
    if(count>std::numeric_limits<std::size_t>::max()/sizeof(float))throw std::overflow_error("Pooled source original output extent overflows");
    if((action==PooledSourceAction::Read)?(!query.output || reinterpret_cast<std::uintptr_t>(query.output)%4 || (!query.elements || query.elements>count || query.elements%plan.row_elements)):
        (query.output || query.elements))throw std::invalid_argument("Pooled source output belongs exactly to the actual original-order consumer");
    const auto controlBytes=control.owner()->size_bytes();const auto* controller=control.owner()->gpu_data_ptr();
    for(const auto roots:{storage.roots(),storage.owner()->roots(),storage.coordinator()->roots()})
        for(const auto& root:{roots.cache,roots.frame,roots.native})if(root) {
            if(overlaps(controller,controlBytes,root->gpu_data_ptr(),root->size_bytes()) ||
               overlaps(query.output,query.elements*sizeof(float),root->gpu_data_ptr(),root->size_bytes()))
                throw std::invalid_argument("Pooled source controller/output aliases its immutable original authority roots");
        }
    if(overlaps(query.output,query.elements*sizeof(float),controller,controlBytes))
        throw std::invalid_argument("Pooled source output aliases canonical request control");
    launch(action,args(storage,&control),query,stream);
}
void IPooledImmutableSourceKernels::closeWhenTerminal(PooledImmutableSourcePreparedStorage& storage,
    const DeviceGenerationControlBinding& control,void* stream)const{
    exact(stream);requireStorage(storage);storage.roots_->requireInitialized();control.requireFor(geometry().allocator().device);
    closeNative(args(storage,&control),control.row()+sampling_math::kDeviceGenerationControlRequestComplete,stream);
}
void IPooledImmutableSourceKernels::closeForReset(PooledImmutableSourcePreparedStorage& storage,
    const DeviceGenerationControlBinding& control,void* stream)const{
    exact(stream);requireStorage(storage);storage.roots_->requireInitialized();control.requireFor(geometry().allocator().device);
    closeNative(args(storage,&control),nullptr,stream);
}
std::uint64_t IPooledImmutableSourceKernels::captureIdentity(const PooledImmutableSourcePreparedStorage& storage)const{
    requireStorage(storage);auto h=storage.captureIdentity();
    for(auto word:{source().captureIdentity(),codeIdentity(),static_cast<std::uint64_t>(kImmutableLookupSubordinateProtocol)})h=(h^word)*1099511628211ULL;
    return h?h:1;
}
}
