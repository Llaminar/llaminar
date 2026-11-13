/**
 * @file NativeCounter64.h
 * @brief Exact two-word terminal counters with checked host/device additions.
 *
 * Arena/controller ownership stays with its original execution owner. A borrowed
 * counter has two INT32 words and mandatory8-byte alignment for native UINT64
 * atomics. GPU writers use compare/exchange, so independent graph branches need
 * no artificial ordering. Host writers use their existing serialized ownership.
 * Overflow never wraps a terminal observation into a plausible smaller total.
 */
#pragma once
#include <cstdint>
#include <cstddef>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_NATIVE_COUNTER_HD __host__ __device__ inline
#else
#define LLAMINAR_NATIVE_COUNTER_HD inline
#endif
namespace llaminar2 {
/** @brief Borrow exactly two adjacent native INT32 words; never allocate or own them. */
struct NativeCounter64 {
    std::int32_t*words=nullptr;
    /** @return Whether the enclosing operation explicitly requested aggregation. */
    LLAMINAR_NATIVE_COUNTER_HD bool bound()const{return words!=nullptr;}
    /** @return Whether native UINT64 atomics can address this original aligned pair. */
    LLAMINAR_NATIVE_COUNTER_HD bool valid()const {
        return words && reinterpret_cast<std::uintptr_t>(words)%alignof(std::uint64_t)==0;
    }
};
/** @brief Decode an ordered terminal pair without UINT64 aliasing on the host.
 * @param words Two readable INT32 words, low first, after terminal completion.
 * @return Exact UINT64 terminal count; caller owns binding validation.
 */
LLAMINAR_NATIVE_COUNTER_HD std::uint64_t readNativeCounter64(const std::int32_t*words) {
    return std::uint64_t(static_cast<std::uint32_t>(words[0])) |
        (std::uint64_t(static_cast<std::uint32_t>(words[1]))<<32);
}
/** @brief Add actual bytes with native UINT64 atomicity or serialized host ownership.
 * @param counter Original aligned request-owned observation pair.
 * @param delta Actual performed work, not storage capacity.
 * @return False for invalid alignment/binding or overflow, preserving this count.
 * @details GPU successful additions commute exactly in unsigned integer arithmetic.
 * Every writer must finish before the existing terminal event/copy observes pairs.
 * Zero deltas issue no atomic operation and write no counter word.
 */
LLAMINAR_NATIVE_COUNTER_HD bool addNativeCounter64(NativeCounter64 counter,std::uint64_t delta) {
    if(!counter.valid())return false;
    if(!delta)return true;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    static_assert(sizeof(unsigned long long)==sizeof(std::uint64_t));
    auto*address=reinterpret_cast<unsigned long long*>(counter.words);
    auto observed=atomicCAS(address,0ull,0ull);
    for(;;) {
        if(delta>UINT64_MAX-observed)return false;
        const auto previous=atomicCAS(address,observed,observed+delta);
        if(previous==observed)return true;
        observed=previous;
    }
#else
    const auto before=readNativeCounter64(counter.words);
    if(delta>UINT64_MAX-before)return false;
    const auto total=before+delta;
    counter.words[0]=static_cast<std::int32_t>(static_cast<std::uint32_t>(total));
    counter.words[1]=static_cast<std::int32_t>(static_cast<std::uint32_t>(total>>32));
    return true;
#endif
}
} // namespace llaminar2
#undef LLAMINAR_NATIVE_COUNTER_HD
