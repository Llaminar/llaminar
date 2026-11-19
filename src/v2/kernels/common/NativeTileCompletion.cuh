/**
 * @file NativeTileCompletion.cuh
 * @brief Shared CUDA/HIP device-scope completion publication primitive.
 *
 * Captured transfers retain their original acquire/release join. Arithmetic
 * tiles instead release every arrival and acquire only in the unique final
 * producer, which alone consumes remote partials. CTA barriers collect every
 * local writer before arrival and distribute the final acquired payload.
 * No thread waits for an unscheduled block, and no arithmetic uses atomics.
 */
#pragma once
#include "NativeTileCompletion.h"
#if defined(__CUDACC__)
#include <cuda/atomic>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif
namespace llaminar2::native_tile_completion
{
/** @param counter Exact device-owned tile counter. @return Earlier completed producers, acquiring their stores. */
__device__ __forceinline__ std::uint32_t join(std::uint32_t* counter)
{
#if defined(__CUDA_ARCH__)
    ::cuda::atomic_ref<std::uint32_t,::cuda::thread_scope_device> reference(*counter);
    return reference.fetch_add(1u,::cuda::memory_order_acq_rel);
#elif defined(__HIP_DEVICE_COMPILE__)
    return __hip_atomic_fetch_add(counter,1u,__ATOMIC_ACQ_REL,__HIP_MEMORY_SCOPE_AGENT);
#else
    __builtin_unreachable(); // The host compiler pass cannot execute a device-only primitive.
#endif
}
/**
 * @brief Release one completed producer and acquire payload only for its unique publisher.
 * @param counter Exact device-owned counter, initialized before the first retained replay.
 * @param contributors Positive immutable original split-K physical producer count.
 * @return Publish only after acquiring every earlier release arrival; Retire consumes no payload.
 * @details The caller's CTA barrier makes all local partial stores happen before
 * its leader's release RMW. All arrivals in this counter epoch are RMWs, so the
 * final RMW reads a member of the release sequence headed by EACH earlier
 * release. C++ [atomics.fences]/4 pairs those releases with the following acquire
 * fence. The caller then retains its CTA barrier before any consumer lane reads
 * the ordered partial bank. Reset remains a release store after publication,
 * and exact-stream graph order separates epochs; no overlapping epoch is legal.
 * This protocol-specific operation does not weaken captured-transfer join().
 */
__device__ __forceinline__ NativeTileCompletionDecision arriveForPublication(
    std::uint32_t* counter,std::uint32_t contributors)
{
    std::uint32_t ticket{};
#if defined(__CUDA_ARCH__)
    ::cuda::atomic_ref<std::uint32_t,::cuda::thread_scope_device> reference(*counter);
    ticket=reference.fetch_add(1u,::cuda::memory_order_release);
#elif defined(__HIP_DEVICE_COMPILE__)
    ticket=__hip_atomic_fetch_add(counter,1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
#else
    __builtin_unreachable(); // The host pass cannot execute this device-only operation.
#endif
    const auto result=nativeTileCompletionDecision(ticket,contributors);
    if(result==NativeTileCompletionDecision::Publish)
    {
#if defined(__CUDA_ARCH__)
        ::cuda::atomic_thread_fence(::cuda::memory_order_acquire,::cuda::thread_scope_device);
#elif defined(__HIP_DEVICE_COMPILE__)
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
#endif
    }
    return result;
}
/** @param counter Exact last-owned tile counter; all publication lanes have joined before reset. */
__device__ __forceinline__ void reset(std::uint32_t* counter)
{
#if defined(__CUDA_ARCH__)
    ::cuda::atomic_ref<std::uint32_t,::cuda::thread_scope_device> reference(*counter);
    reference.store(0u,::cuda::memory_order_release);
#elif defined(__HIP_DEVICE_COMPILE__)
    __hip_atomic_store(counter,0u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
#endif
}
}
