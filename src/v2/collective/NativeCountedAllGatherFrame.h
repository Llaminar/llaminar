/**
 * @file NativeCountedAllGatherFrame.h
 * @brief Collective-local device authority for completed native byte broadcasts.
 *
 * One original participant owns this frame and the following degree-sized banks.
 * Setup clears them once. Captured preparation advances the invocation sequence;
 * request reset never rewinds it or clears a sticky native failure. Extents remain
 * authored by the existing packet kernels. Operands and receipts borrow this same
 * PMA allocation and are immutable throughout one ordered broadcast composition.
 */
#pragma once
#include "NativeCollectiveCompletion.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace llaminar2 {
/** @brief Closed cold initialization versus captured invocation advancement. */
enum class NativeCountedAllGatherPreparation : std::uint8_t { Initialize, Advance };
/** @brief Exactly one participant's persistent invocation/error authority. */
struct NativeCountedAllGatherFrame final {
    std::uint64_t table, admission, sequence;
    std::int32_t first_error;
    std::uint32_t reserved;
};
static_assert(sizeof(NativeCountedAllGatherFrame)==32);
static_assert(alignof(NativeCountedAllGatherFrame)==alignof(std::uint64_t) &&
    sizeof(NativeCountedAllGatherFrame)%alignof(std::uint64_t)==0 &&
    sizeof(std::uint64_t)%alignof(ncclLlaminarCompletionOperand)==0 &&
    sizeof(ncclLlaminarCompletionOperand)%alignof(ncclLlaminarByteCompletion)==0 &&
    sizeof(ncclLlaminarByteCompletion)%alignof(unsigned long long)==0,
    "Every actual-degree bank offset preserves the compiled native ABI alignment");

/** @brief Checked O(N) metadata geometry; participant count has no policy cap. */
struct NativeCountedAllGatherLayout final {
    const std::size_t degree;
    /** @brief Reject zero, foreign native coordinates and arithmetic overflow. */
    explicit NativeCountedAllGatherLayout(std::size_t count):degree(count) {
        constexpr auto per=sizeof(std::uint64_t)+sizeof(ncclLlaminarCompletionOperand)+
            sizeof(ncclLlaminarByteCompletion)+sizeof(unsigned long long);
        if(!count || count>std::size_t(std::numeric_limits<int>::max()))
            throw std::invalid_argument("Native counted allgather requires representable positive membership");
        if(count>(std::numeric_limits<std::size_t>::max()-sizeof(NativeCountedAllGatherFrame))/per)
            throw std::overflow_error("Native counted allgather metadata overflow");
    }
    /** @return Existing source-major count bank, including the local producer. */
    std::size_t extents()const noexcept{return sizeof(NativeCountedAllGatherFrame);}
    /** @return Immutable-for-invocation completion operands. */
    std::size_t operands()const noexcept{return extents()+degree*sizeof(std::uint64_t);}
    /** @return Actual native-written source-major completion receipts. */
    std::size_t receipts()const noexcept{return operands()+degree*sizeof(ncclLlaminarCompletionOperand);}
    /** @return Passive actual outgoing payload counters, including forwarding links. */
    std::size_t counters()const noexcept{return receipts()+degree*sizeof(ncclLlaminarByteCompletion);}
    /** @return Exact device allocation request, excluding every payload bank. */
    std::size_t bytes()const noexcept{return counters()+degree*sizeof(unsigned long long);}
};

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_COUNTED_HD __host__ __device__
#else
#define LLAMINAR_COUNTED_HD
#endif
/**
 * @brief Shared device transition, replayed directly by device-free contract tests.
 * @param memory Same initialized participant-local bank. @param degree Exact admitted degree.
 * @param table Capture-frozen semantic identity. @param admission Actual admitted packet geometry.
 * @return True after sealing one new invocation; false leaves an absorbing error.
 * The single writer never clears an error or rewinds sequence during request reset.
 */
LLAMINAR_COUNTED_HD inline bool prepareNativeCountedAllGatherInvocation(void* memory,std::size_t degree,
    std::uint64_t table,std::uint64_t admission,unsigned long long *outgoing = nullptr) {
    auto* frame=static_cast<NativeCountedAllGatherFrame*>(memory);
    if(!table || !admission || frame->reserved || frame->first_error || frame->sequence==~std::uint64_t(0)) {
        if(!frame->first_error)frame->first_error=1;
        return false;
    }
    frame->table=table;frame->admission=admission;++frame->sequence;
    auto* operands=reinterpret_cast<ncclLlaminarCompletionOperand*>(
        static_cast<unsigned char*>(memory)+sizeof(*frame)+degree*sizeof(std::uint64_t));
    auto* receipts=reinterpret_cast<ncclLlaminarByteCompletion*>(operands+degree);
    auto* counters=reinterpret_cast<unsigned long long*>(receipts+degree);
    for(std::size_t root=0;root<degree;++root)
        operands[root]={&frame->table,&frame->admission,&frame->sequence,&frame->first_error,
            receipts+root,outgoing ? outgoing : counters+root,((table^(root+1))*1099511628211ULL)|1ULL,1,0};
    return true;
}
/**
 * @brief Shared exact completion predicate after the whole native composition.
 * @param memory Same invocation bank. @param degree Actual original membership.
 * @param original Original native authority. @param aborted Acquired original native abort word.
 * @return Whether every root has a fresh exact live-count receipt; false sets sticky error.
 */
LLAMINAR_COUNTED_HD inline bool completeNativeCountedAllGatherInvocation(void* memory,std::size_t degree,
    ncclLlaminarCompletionAuthority original,std::uint32_t aborted) {
    auto* frame=static_cast<NativeCountedAllGatherFrame*>(memory);
    auto* extents=reinterpret_cast<std::uint64_t*>(frame+1);
    auto* operands=reinterpret_cast<ncclLlaminarCompletionOperand*>(extents+degree);
    auto* receipts=reinterpret_cast<ncclLlaminarByteCompletion*>(operands+degree);
    for(std::size_t root=0;root<degree;++root)
        if(!ncclLlaminarCompletionMatches(receipts[root],frame->table,frame->admission,frame->sequence,
            extents[root],operands[root].message,original,aborted,frame->first_error)) {
            if(!frame->first_error)frame->first_error=1;
            return false;
        }
    return true;
}
#undef LLAMINAR_COUNTED_HD
#if defined(__CUDACC__) || defined(__HIPCC__)
/** @brief Seal exactly one participant invocation on its existing exact stream. */
static __global__ void prepareNativeCountedAllGatherFrame(void* memory,std::size_t degree,
    std::uint64_t table,std::uint64_t admission,unsigned long long *outgoing) {
    if(!prepareNativeCountedAllGatherInvocation(memory,degree,table,admission,outgoing)) {
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
}
/**
 * @brief Reject incomplete/aborted native work before dependent packet publication.
 * @param memory Same invocation bank after every original broadcast launch.
 * @param degree Exact admitted native membership.
 * @param original Original communicator identity, never selected from a receipt.
 * Only the terminal test boundary may observe receipts on the host. Production
 * failure is a device trap before the downstream import can consume stale data.
 */
static __global__ void validateNativeCountedAllGatherFrame(void* memory,std::size_t degree,
    ncclLlaminarCompletionAuthority original) {
    // Match the native ABI's system-acquire of the original absorbing flag;
    // never trust authority pointers copied out of the receipt itself.
    std::uint32_t aborted;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(aborted) : "l"(original.abortFlag) : "memory");
#elif defined(__HIP_DEVICE_COMPILE__)
    aborted=__atomic_load_n(original.abortFlag,__ATOMIC_ACQUIRE);
#else
    aborted=*original.abortFlag;
#endif
    if(!completeNativeCountedAllGatherInvocation(memory,degree,original,aborted)) {
#if defined(__HIP_DEVICE_COMPILE__)
        __builtin_trap();
#else
        asm volatile("trap;");
#endif
    }
}

#endif
} // namespace llaminar2
