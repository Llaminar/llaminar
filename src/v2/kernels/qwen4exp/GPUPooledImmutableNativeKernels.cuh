/**
 * @file GPUPooledImmutableNativeKernels.cuh
 * @brief Exact-stream participant-local CUDA/HIP owner/coordinator launch bundles.
 *
 * All live counts remain device-owned. Independent initialization, coalescing
 * and native-byte copies are parallel; one ordered writer retains policy and
 * physical mutation authority. Every launch is fixed before capture, with no
 * allocation, host callbacks, host count reads, synchronization or transport.
 */
#pragma once
#if defined(QFN_POOLED_NATIVE_CUDA)
#include <cuda_runtime.h>
#include <cuda/atomic>
#define QFN_NATIVE_NAMESPACE cuda_pooled_native
#define QFN_NATIVE_STREAM cudaStream_t
#define QFN_NATIVE_ERROR cudaGetLastError
#define QFN_NATIVE_SUCCESS cudaSuccess
#define QFN_NATIVE_ERROR_TEXT cudaGetErrorString
#define QFN_NATIVE_DEVICE cudaGetDevice
#define QFN_NATIVE_FACTORY createCUDAPooledImmutableNativeKernels
#define QFN_NATIVE_BACKEND DeviceType::CUDA
#else
#include <hip/hip_runtime.h>
#define QFN_NATIVE_NAMESPACE rocm_pooled_native
#define QFN_NATIVE_STREAM hipStream_t
#define QFN_NATIVE_ERROR hipGetLastError
#define QFN_NATIVE_SUCCESS hipSuccess
#define QFN_NATIVE_ERROR_TEXT hipGetErrorString
#define QFN_NATIVE_DEVICE hipGetDevice
#define QFN_NATIVE_FACTORY createROCmPooledImmutableNativeKernels
#define QFN_NATIVE_BACKEND DeviceType::ROCm
#endif
#include "PooledImmutableNativeKernels.h"
#include "PooledImmutableNativeOps.h"
#include "PooledImmutableOwnerLaunchPlan.h"
#include "PooledImmutableReadOps.h"
#include <stdexcept>
#include <string>
#include <type_traits>
#include "GPUPooledImmutableCoalescing.cuh"
#include "GPUPooledImmutableExactIdentity.cuh"
#include "GPUPooledImmutableOwnerValidation.cuh"
namespace llaminar2::qwen4exp::QFN_NATIVE_NAMESPACE
{
/** @brief Seed mandatory owner controls once; no physical slot is made Ready or zero-filled. */
template<class Slot> __global__ void seedOwnerKernel(n::OwnerArgs a){n::seedOwner<Slot>(a);}
/** @brief Independent virgin slot initialization, ordered before the one canonical owner seal. */
template<class Slot> __global__ __launch_bounds__(Threads,2) void initializeSlotsKernel(n::OwnerArgs a)
{
    auto op=p::OwnerOperations<Slot>(n::ownerView<Slot>(a));
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<a.plan.owner.capacity();i+=std::uint64_t{gridDim.x}*Threads)
        op.initializeSlot(static_cast<unsigned>(i));
}
/** @brief Publish the actual once-only owner setup identity after all slot producers. */
template<class Slot> __global__ void sealOwnerKernel(n::OwnerArgs a){n::sealOwner<Slot>(a);}
/** @return The same owner operations over mapped or native-inline immutable arguments. */
template<class Args> __host__ __device__ const n::OwnerArgs& originalOwnerArgs(const Args& value)
{
    if constexpr (std::is_same_v<Args, n::NativeOwnerArgs>) return value.owner;
    else return value;
}
/** @return Block-uniform original native cancellation without physical release or first-fault erasure. */
template<class Slot> __device__ bool abortedNativeOwner(const n::NativeOwnerArgs& value)
{
    __shared__ std::uint32_t cancelled;
    if (threadIdx.x == 0) {
        std::uint32_t abort = 0;
        if (value.original_invocation.abortFlag) {
#if defined(QFN_POOLED_NATIVE_CUDA)
            auto* address = const_cast<std::uint32_t*>(value.original_invocation.abortFlag);
            abort = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_system>(*address).load(cuda::memory_order_acquire);
#else
            abort = __hip_atomic_load(value.original_invocation.abortFlag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
        }
        cancelled = n::suppressAbortedNativeOwner<Slot>(value.owner, value.original_invocation, abort) ? 1u : 0u;
    }
    __syncthreads();
    return cancelled != 0;
}
/** @return Rare cancellation drain over only original endpoint and owner state; normal work reads no mapped word. */
template<class Slot> __device__ bool cancelledIdleOwner(const n::OwnerArgs& a)
{
    if (!a.original_invocation.hasCancelledCursor()) return false;
#if defined(QFN_POOLED_NATIVE_CUDA)
    auto* address = const_cast<std::uint64_t*>(a.original_invocation.cancellationAddress());
    const auto cancellation = cuda::atomic_ref<std::uint64_t, cuda::thread_scope_system>(*address).load(cuda::memory_order_acquire);
#else
    const auto cancellation = __hip_atomic_load(a.original_invocation.cancellationAddress(), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
#endif
    return n::cancelledIdleOwner<Slot>(a, cancellation);
}
/** @return Original mapped cancellation, or native parent abort for the explicit native-only specialization. */
template<class Slot, class Args> __device__ bool cancelledOwnerWork(const Args& value)
{
    if constexpr (std::is_same_v<Args, n::NativeOwnerArgs>)
        if (abortedNativeOwner<Slot>(value)) return true;
    return cancelledIdleOwner<Slot>(originalOwnerArgs(value));
}
/** @brief One real ordered owner state mutation; no row payload work runs here. */
template<n::OwnerAction Action,class Slot,class Args> __global__ void ownerPhase(Args supplied)
{
    const auto& a = originalOwnerArgs(supplied);
    if (cancelledOwnerWork<Slot>(supplied)) return;
    bool succeeded;
    if constexpr(Action==n::OwnerAction::Pin || Action==n::OwnerAction::AcceptFree || Action==n::OwnerAction::AcceptVictim)
        succeeded=n::CooperativeOwnerValidation<Slot>::template run<Action>(a);
    else succeeded=n::owner<Action,Slot>(a);
    if constexpr(Action==n::OwnerAction::AcquireReaderRetirement || Action==n::OwnerAction::RetiredWire || Action==n::OwnerAction::ReadRows || Action==n::OwnerAction::PublishWire || Action==n::OwnerAction::ReadyWire || Action==n::OwnerAction::PublicationAcquiredWire){
        if(!succeeded){
#if defined(QFN_POOLED_NATIVE_CUDA)
            __trap();
#else
            __builtin_trap();
#endif
        }
    }
}
/** @brief Copy only the current immutable actual grant prefix to reusable directed transport staging. */
template<n::OwnerAction Action, class Slot,class Args> __global__ __launch_bounds__(Threads,2) void publishReplies(Args supplied)
{
    const auto& a = originalOwnerArgs(supplied);
    if (cancelledOwnerWork<Slot>(supplied)) return;
    const auto bytes=n::progress(a).reply_bytes;
    if(bytes%sizeof(p::Grant) || bytes/sizeof(p::Grant)>a.plan.native.reply_capacity)return;
    const auto count=bytes/sizeof(p::Grant);
    const auto* source=Action==n::OwnerAction::Pin?
        n::at<p::Grant>(a.frame,a.plan.frame.initial_pins):n::grants(a)+n::allocationFrame(a).wave_offset;
    auto* output=n::at<p::Grant>(a.native,a.plan.native.replies);
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<count;i+=std::uint64_t{gridDim.x}*Threads)output[i]=source[i];
}
/** @brief Fill exactly live unchanged native row bytes; slot stride padding is never copied. */
template<class Slot,class Args> __global__ __launch_bounds__(Threads,2) void copyRows(Args supplied)
{
    const auto& a = originalOwnerArgs(supplied);
    if (cancelledOwnerWork<Slot>(supplied)) return;
    if(n::progress(a).delivery!=n::Delivery::Validated || n::ownerView<Slot>(a).frame->status!=adaptive_lookup::Status::Success)return;
    const auto count=n::allocationFrame(a).count;
    const auto bytes=std::uint64_t{count}*a.plan.row_bytes;
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<bytes;i+=std::uint64_t{gridDim.x}*Threads){
        const auto row=i/a.plan.row_bytes,byte=i%a.plan.row_bytes;
        a.cache[a.plan.cache.payload+std::uint64_t{n::grants(a)[row].slot}*a.plan.slot_stride+byte]=a.input.payload[i];
    }
}
/** @brief Pack only actual validated pinned bytes, excluding all native slot stride padding. */
template<class Slot,class Args> __global__ __launch_bounds__(Threads,2) void copyHits(Args supplied){
    const auto& a = originalOwnerArgs(supplied);
    if (cancelledOwnerWork<Slot>(supplied)) return;
    if(n::progress(a).delivery!=n::Delivery::Validated)return;
    const auto bytes=n::at<n::OwnerIngress>(a.native,a.plan.native.ingress)->payload_bytes;
    for(auto i=std::uint64_t{blockIdx.x}*Threads+threadIdx.x;i<bytes;i+=std::uint64_t{gridDim.x}*Threads)
        a.native[a.plan.native.source_payload+i]=*n::hitReadByte(a,i);
}
/** @brief Publish fresh receipt64/count only after every ordered packed-byte writer. */
template<class Slot,class Args> __global__ void hitsComplete(Args supplied)
{
    if (!cancelledOwnerWork<Slot>(supplied)) (void)n::completeHitRead<Slot>(originalOwnerArgs(supplied));
}
/** @brief Actual copy completion is published only after every disjoint payload writer's launch completes. */
template<class Slot,class Args> __global__ void copyComplete(Args supplied)
{
    if (!cancelledOwnerWork<Slot>(supplied)) n::copied<Slot>(originalOwnerArgs(supplied));
}
/** @brief Seed only actual coordinator controls. */
__global__ void seedCoordinatorKernel(n::CoordinatorSetupArgs a){n::seedCoordinator(a);}
/** @brief Write one exact immutable owner geometry record from setup scalars, without a host payload mirror. */
__global__ void initializeOwnerGeometryKernel(p::CoordinatorOwnerGeometry* destination,p::CoordinatorOwnerGeometry value)
{*destination=value;}
/** @brief Parallel independent directory/radix/frequency setup, with no serial whole-bank initialization. */
__global__ __launch_bounds__(Threads,2) void initializePolicyKernel(n::CoordinatorSetupArgs a)
{
    auto op=p::CoordinatorOperations(n::coordinatorView(a));
    const auto lane=std::uint64_t{blockIdx.x}*Threads+threadIdx.x,stride=std::uint64_t{gridDim.x}*Threads;
    for(auto i=lane;i<a.plan.policy.entries();i+=stride)p::CoordinatorCoalescingAccess::entry(op,static_cast<unsigned>(i));
    for(auto i=lane;i+1<a.plan.policy.entries();i+=stride)p::CoordinatorCoalescingAccess::node(op,static_cast<unsigned>(i));
    for(auto i=lane;i<std::uint64_t{4}*a.plan.policy.laneWidth();i+=stride)p::CoordinatorCoalescingAccess::frequency(op,i);
}
/** @brief Seal actual coordinator controls after every static geometry and independent setup producer. */
__global__ void sealCoordinatorKernel(n::CoordinatorSetupArgs a)
{auto op=p::CoordinatorOperations(n::coordinatorView(a));p::CoordinatorCoalescingAccess::seal(op);}
/** @brief One ordered policy/consensus operation; independent coalescing and payload copies use their own producers. */
template<n::CoordinatorAction Action> __global__ void coordinatorPhase(n::CoordinatorArgs a)
{n::submitCoordinator<Action>(a);}
/** @brief Reject default streams and foreign current contexts before any enqueue; no context guessing or substitution. */
void require(DeviceId device,void* stream)
{
    if(!stream || reinterpret_cast<std::uintptr_t>(stream)<=2 || device.type!=QFN_NATIVE_BACKEND)
        throw std::invalid_argument("Pooled native enqueue requires the exact non-default backend stream");
    int current=-1;
    if(QFN_NATIVE_DEVICE(&current)!=QFN_NATIVE_SUCCESS || current!=device.ordinal)
        throw std::invalid_argument("Pooled native enqueue current device differs from the sealed original participant");
}
/** @brief Authenticate launch submission without a blocking runtime wait. */
void checked()
{
    const auto error=QFN_NATIVE_ERROR();
    if(error!=QFN_NATIVE_SUCCESS)throw std::runtime_error(std::string("Pooled native launch failed: ")+QFN_NATIVE_ERROR_TEXT(error));
}
/** @return Static bounded launch grid; actual counts remain inside native kernels. */
unsigned blocks(std::uint64_t maximum){return static_cast<unsigned>(std::max<std::uint64_t>(1,std::min<std::uint64_t>((maximum+Threads-1)/Threads,65535)));}
/** @brief Backend bundle owns code only; all state, roots and exact dependencies remain external retained bindings. */
class Kernels final:public IPooledImmutableNativeKernels
{
public:
    /** @return Actual compile-time native backend with no device discovery. */
    DeviceType backend()const noexcept override{return QFN_NATIVE_BACKEND;}
    /** @return Version, exact identity algorithm, thread geometry and deterministic coalescing policy are immutable capture identity. */
    std::uint64_t codeIdentity()const noexcept override
    {return (n::Version<<48)|(static_cast<std::uint64_t>(QFN_NATIVE_BACKEND)<<40)|(std::uint64_t{1}<<32)|(std::uint64_t{Threads}<<16)|SmallRows;}
protected:
    /** @brief Initialize one checked owner once on its exact setup stream. */
    void initializeOwner(const n::OwnerArgs& a,void* exact)const override
    {
        require(a.plan.device,exact);const auto stream=static_cast<QFN_NATIVE_STREAM>(exact);
        if(a.plan.owner.rowWidth()==adaptive_lookup::RowKeyWidth::UInt32)setupOwner<adaptive_lookup::Slot32>(a,stream);
        else setupOwner<adaptive_lookup::Slot64>(a,stream);
    }
    /** @brief Initialize actual immutable owner coordinates and parallel policy banks on one exact setup stream. */
    void initializeCoordinator(const n::CoordinatorSetupArgs& a,std::span<const p::CoordinatorOwnerGeometry> owners,void* exact)const override
    {
        require(a.plan.device,exact);const auto stream=static_cast<QFN_NATIVE_STREAM>(exact);
        seedCoordinatorKernel<<<1,1,0,stream>>>(a);checked();
        for(std::size_t i=0;i<owners.size();++i){
            initializeOwnerGeometryKernel<<<1,1,0,stream>>>(n::at<p::CoordinatorOwnerGeometry>(a.frame,a.plan.frame.owners)+i,owners[i]);checked();
        }
        initializePolicyKernel<<<blocks(std::max<std::uint64_t>(a.plan.policy.entries(),std::uint64_t{4}*a.plan.policy.laneWidth())),Threads,0,stream>>>(a);checked();
        sealCoordinatorKernel<<<1,1,0,stream>>>(a);checked();
    }
    /** @brief Record exact native owner phase and every inseparable private publication/copy producer. */
    void launch(n::OwnerAction action,const n::OwnerArgs& a,void* exact)const override
    {
        require(a.plan.device,exact);const auto stream=static_cast<QFN_NATIVE_STREAM>(exact);
        if(a.plan.owner.rowWidth()==adaptive_lookup::RowKeyWidth::UInt32)ownerAction<adaptive_lookup::Slot32>(action,a,stream);
        else ownerAction<adaptive_lookup::Slot64>(action,a,stream);
    }
    /** @brief Record native-only inline owner entry; original mapped arguments/dispatch remain distinct. */
    void launchNativeOwner(n::OwnerAction action, const n::NativeOwnerArgs& a, void* exact) const override
    {
        require(a.owner.plan.device, exact);
        const auto stream = static_cast<QFN_NATIVE_STREAM>(exact);
        if (a.owner.plan.owner.rowWidth() == adaptive_lookup::RowKeyWidth::UInt32)
            ownerAction<adaptive_lookup::Slot32>(action, a, stream);
        else ownerAction<adaptive_lookup::Slot64>(action, a, stream);
    }
    /** @brief Record the exact fixed coordinator phase without host count readbacks or hidden collectives. */
    void launch(n::CoordinatorAction action,const n::CoordinatorArgs& a,const n::CoordinatorQuery* query,void* exact)const override
    {
        require(a.plan.device,exact);const auto stream=static_cast<QFN_NATIVE_STREAM>(exact);
        switch(action){
        case n::CoordinatorAction::Begin:
            if(!query)throw std::invalid_argument("Pooled native Begin has no original query producer");
            switch(query->original.id_width){
            case n::OriginalIdWidth::UInt32:begin<n::OriginalIdWidth::UInt32>(n::CoordinatorBeginArgs{a,*query},stream);return;
            case n::OriginalIdWidth::UInt64:begin<n::OriginalIdWidth::UInt64>(n::CoordinatorBeginArgs{a,*query},stream);return;
            }
            throw std::invalid_argument("Pooled native Begin received an unsupported original producer width");
#define QFN_COORDINATOR_CASE(Name) case n::CoordinatorAction::Name:coordinatorPhase<n::CoordinatorAction::Name><<<1,1,0,stream>>>(a);checked();return;
        QFN_COORDINATOR_CASE(ImportLocalPinHeader)
        QFN_COORDINATOR_CASE(ImportLocalAvailability)
        QFN_COORDINATOR_CASE(ImportLocalFreeReceipt)
        QFN_COORDINATOR_CASE(ImportLocalVictimReceipt)
        QFN_COORDINATOR_CASE(PinCommands)
        QFN_COORDINATOR_CASE(AcquirePins)
        QFN_COORDINATOR_CASE(SealPins)
        case n::CoordinatorAction::PlanFree:
            observationRuns<<<static_cast<unsigned>((std::uint64_t{a.plan.policy.maximumOriginal()}+Threads-1)/Threads),Threads,0,stream>>>(a);checked();
            coordinatorPhase<n::CoordinatorAction::PlanFree><<<1,1,0,stream>>>(a);checked();return;
        QFN_COORDINATOR_CASE(AcquireFree)
        QFN_COORDINATOR_CASE(CompleteFree)
        QFN_COORDINATOR_CASE(PlanVictims)
        QFN_COORDINATOR_CASE(AcquireVictims)
        QFN_COORDINATOR_CASE(CompleteVictims)
        QFN_COORDINATOR_CASE(SealSource)
        QFN_COORDINATOR_CASE(Publish)
        QFN_COORDINATOR_CASE(Cancel)
        QFN_COORDINATOR_CASE(Finish)
#undef QFN_COORDINATOR_CASE
        }
        throw std::invalid_argument("Unknown pooled native coordinator phase");
    }
private:
    /** @brief Compose separate seed/parallel-slot/seal producers so setup cannot race an initialization guard. */
    template<class Slot> static void setupOwner(const n::OwnerArgs& a,QFN_NATIVE_STREAM stream)
    {
        seedOwnerKernel<Slot><<<1,1,0,stream>>>(a);checked();
        initializeSlotsKernel<Slot><<<blocks(a.plan.owner.capacity()),Threads,0,stream>>>(a);checked();
        sealOwnerKernel<Slot><<<1,1,0,stream>>>(a);checked();
    }
    /** @brief Execute one closed owner phase specialization and publish only its actual live reply prefix. */
    template<n::OwnerAction Action,class Slot,class Args> static void submitOwner(const Args& supplied,QFN_NATIVE_STREAM stream)
    {
        const auto& a = originalOwnerArgs(supplied);
        constexpr auto lanes=Action==n::OwnerAction::Pin || Action==n::OwnerAction::AcceptFree || Action==n::OwnerAction::AcceptVictim?Threads:1;
        ownerPhase<Action,Slot><<<1,lanes,0,stream>>>(supplied);checked();
        if constexpr(n::OwnerLaunchPlan<Action>::publishesReplies)
            if(a.plan.native.reply_capacity){publishReplies<Action,Slot><<<blocks(a.plan.native.reply_capacity),Threads,0,stream>>>(supplied);checked();}
        if constexpr(n::OwnerLaunchPlan<Action>::readsRows){
            if(a.plan.native.reply_capacity){copyHits<Slot><<<blocks(std::uint64_t{a.plan.native.reply_capacity}*a.plan.row_bytes),Threads,0,stream>>>(supplied);checked();}
            hitsComplete<Slot><<<1,1,0,stream>>>(supplied);checked();
        }
        if constexpr(n::OwnerLaunchPlan<Action>::appliesSource){
            if(a.plan.native.reply_capacity){
                copyRows<Slot><<<blocks(std::uint64_t{a.plan.native.reply_capacity}*a.plan.row_bytes),Threads,0,stream>>>(supplied);checked();
            }
            copyComplete<Slot><<<1,1,0,stream>>>(supplied);checked();
        }
    }
    /** @brief Closed owner dispatch keeps each candidate's compiler resource evidence isolated by kernel symbol. */
    template<class Slot,class Args> static void ownerAction(n::OwnerAction action,const Args& a,QFN_NATIVE_STREAM stream)
    {
        switch(action){
#define QFN_OWNER_CASE(Name) case n::OwnerAction::Name:submitOwner<n::OwnerAction::Name,Slot>(a,stream);return;
        QFN_OWNER_CASE(ReadRows)
        QFN_OWNER_CASE(Begin)
        QFN_OWNER_CASE(Pin)
        QFN_OWNER_CASE(SealPins)
        QFN_OWNER_CASE(AcquirePins)
        QFN_OWNER_CASE(Availability)
        QFN_OWNER_CASE(AcceptFree)
        QFN_OWNER_CASE(ExecuteFree)
        QFN_OWNER_CASE(FreeReceipt)
        QFN_OWNER_CASE(AcquireFree)
        QFN_OWNER_CASE(AcceptVictim)
        QFN_OWNER_CASE(ExecuteVictim)
        QFN_OWNER_CASE(VictimReceipt)
        QFN_OWNER_CASE(SealSource)
        QFN_OWNER_CASE(ApplySourceDelivery)
        QFN_OWNER_CASE(ValidateFills)
        QFN_OWNER_CASE(AcquirePublication)
        QFN_OWNER_CASE(MakeReady)
        QFN_OWNER_CASE(ReadyReceipt)
        QFN_OWNER_CASE(Release)
        QFN_OWNER_CASE(CancelFree)
        QFN_OWNER_CASE(CancelVictim)
        QFN_OWNER_CASE(CancelSource)
        QFN_OWNER_CASE(Retire)
        QFN_OWNER_CASE(DrainSourceFailure)
        QFN_OWNER_CASE(PublishWire)
        QFN_OWNER_CASE(ReadyWire)
        QFN_OWNER_CASE(PublicationAcquiredWire)
        QFN_OWNER_CASE(RetiredWire)
        QFN_OWNER_CASE(AcquireReaderRetirement)
#undef QFN_OWNER_CASE
        }
        throw std::invalid_argument("Unknown pooled native owner phase");
    }
    /** @brief Fixed radix partition DAG; every native loop uses its actual original/unique extent. */
    template<bool Positions> static void radix(const n::CoordinatorBeginArgs& a,QFN_NATIVE_STREAM stream)
    {
        bool from_a=!Positions;
        const unsigned passes=n::radixPasses(Positions?static_cast<std::uint64_t>(a.query.original.rows.physicalRows()):a.plan.policy.sourceRows());
        // Each radix block owns one partition; unlike grid-stride element kernels, it must cover every admitted partition.
        const auto maximum=Positions?std::min<std::uint64_t>(a.plan.policy.maximumUnique(),a.query.original.rows.physicalRows()):
            static_cast<std::uint64_t>(a.query.original.rows.physicalRows());
        const auto grid=static_cast<unsigned>((maximum+Threads-1)/Threads);
        for(unsigned pass=0;pass<passes;++pass){
            radixCount<Positions><<<grid,Threads,0,stream>>>(a,pass,from_a);checked();
            radixOffsets<Positions><<<1,Threads,0,stream>>>(a);checked();
            radixScatter<Positions><<<grid,Threads,0,stream>>>(a,pass,from_a);checked();
            from_a=!from_a;
        }
    }
    /** @brief Inseparable parallel Begin bundle selected only by immutable captured row capacity. */
    template<n::OriginalIdWidth Width> static void begin(const n::CoordinatorBeginArgs& a,QFN_NATIVE_STREAM stream)
    {
        const auto maximum=static_cast<std::uint64_t>(a.query.original.rows.physicalRows());
        openOriginal<Width><<<1,1,0,stream>>>(a);checked();
        if(maximum<=SmallRows){coalesceSmall<Width><<<1,Threads,0,stream>>>(a);checked();}
        else{
            const auto grid=blocks(maximum);
            seedKeys<Width><<<grid,Threads,0,stream>>>(a);checked();
            radix<false>(a,stream);
            markGroups<<<grid,Threads,0,stream>>>(a);checked();
            scanGroups<<<1,Threads,0,stream>>>(a);checked();
            materializeGroups<<<grid,Threads,0,stream>>>(a);checked();
            radix<true>(a,stream);
            materializeUnique<<<blocks(std::min<std::uint64_t>(a.plan.policy.maximumUnique(),maximum)),Threads,0,stream>>>(a);checked();
            remapOriginal<<<grid,Threads,0,stream>>>(a);checked();
        }
        finishOriginal<<<1,1,0,stream>>>(a);checked();
    }
};
}
namespace llaminar2::qwen4exp {
/** @brief Construct code only; the checked storage/operand wrapper owns all actual runtime authentication. */
std::unique_ptr<IPooledImmutableNativeKernels> QFN_NATIVE_FACTORY()
{return std::make_unique<QFN_NATIVE_NAMESPACE::Kernels>();}
}
#include "GPUPooledImmutableReplicatedReader.cuh"
#include "GPUPooledImmutableLocalHotTransport.cuh"
#undef QFN_NATIVE_NAMESPACE
#undef QFN_NATIVE_STREAM
#undef QFN_NATIVE_ERROR
#undef QFN_NATIVE_SUCCESS
#undef QFN_NATIVE_ERROR_TEXT
#undef QFN_NATIVE_DEVICE
#undef QFN_NATIVE_FACTORY
#undef QFN_NATIVE_BACKEND
