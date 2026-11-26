/**
 * @file ROCmResidentGridBarrier.cuh
 * @brief Instance-owned device-scope joins for an occupancy-admitted ordinary grid.
 *
 * Every admitted CTA reaches each join, including inactive numerical work.
 * The owning graph's stream/event edges order replays; concurrent instances
 * own distinct serial-family storage. No host reset or device singleton is used.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>
#include "kernels/common/NativeLayoutBarrierLayout.h"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
namespace llaminar2::rocm {
/**
 * @brief Instance-owned reusable sense barriers with a positive typed extent.
 * @tparam PhaseCount Distinct joins reached once by every CTA per invocation.
 * @details The original cold residency/stream/lease owner remains authoritative;
 * changing this extent does not reserve a grid or permit overlapping users.
 */
template<unsigned PhaseCount>
struct ResidentGridBarrierFor {
    using Layout=ResidentGridBarrierLayout<PhaseCount>; ///< Canonical stored-byte extent.
    static_assert(Layout::phases==PhaseCount,"Resident grid barrier layout must instantiate its positive extent");
    std::uint32_t* words; ///< Exact instance-owned subregion, retained by the existing owner.
    unsigned phase=0; ///< Once-per-invocation join ordinal; a fresh phase object starts at zero.
    /**
     * @brief Release all CTA writes, join the entire resident grid, and acquire peers.
     * @details All lanes participate. The final leader resets arrival before
     * publishing the flipped generation. A subsequent invocation cannot reuse
     * a slot before every prior CTA has observed its generation: it must first
     * retire that prior invocation through its exact stream/event edge. Within
     * an invocation each admitted slot is used exactly once.
     */
    __device__ __forceinline__ void sync()
    {
        if(!words||phase>=Layout::phases)__builtin_trap();
        auto* arrival=words+Layout::words_per_phase*phase;
        auto* generation=arrival+1;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
        // On gfx906, L1D is write-through to L2 (TCC), while scalar memory
        // atomics and glc loads/stores execute directly in L2 without L1
        // vector-atomic traffic. Using the exact two-word sense protocol avoids
        // counter wraparound and preserves arbitrary gridDim.x reuse across replays.
        const unsigned grid=__builtin_amdgcn_readfirstlane(gridDim.x);
        __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        __asm__ volatile("s_barrier" ::: "memory");
        if(threadIdx.x<64u)
        {
            unsigned before=0u;
            __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                             : "=s"(before) : "s"(generation) : "memory");
            unsigned ticket=1u;
            __asm__ volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                             : "+s"(ticket) : "s"(arrival) : "memory");
            if(ticket>=grid)__builtin_trap();
            if(ticket+1u==grid)
            {
                unsigned zero=0u;
                unsigned next=before^1u;
                __asm__ volatile(
                    "s_store_dword %0, %1, 0x0 glc\n\t"
                    "s_waitcnt lgkmcnt(0)\n\t"
                    "s_store_dword %2, %3, 0x0 glc\n\t"
                    "s_waitcnt lgkmcnt(0)"
                    :: "s"(zero), "s"(arrival), "s"(next), "s"(generation) : "memory");
            }
            else
            {
                while(true)
                {
                    unsigned cur=0u;
                    __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                                     : "=s"(cur) : "s"(generation) : "memory");
                    if(cur!=before)break;
                    __builtin_amdgcn_s_sleep(1);
                }
            }
            __asm__ volatile("buffer_wbinvl1_vol" ::: "memory");
        }
        __asm__ volatile("s_barrier" ::: "memory");
#else
        __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
        __syncthreads();
        if(threadIdx.x==0)
        {
            const auto before=__hip_atomic_load(generation,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
            const auto ticket=__hip_atomic_fetch_add(arrival,1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
            if(ticket>=gridDim.x)__builtin_trap();
            if(ticket+1u==gridDim.x)
            {
                // The final RMW reads every earlier release sequence. Acquire
                // only here before republishing those CTA payloads in sense.
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
                __hip_atomic_store(arrival,0u,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(generation,before^1u,__ATOMIC_RELEASE,__HIP_MEMORY_SCOPE_AGENT);
            }
            else
            {
                while(__hip_atomic_load(generation,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_AGENT)==before)
                    __builtin_amdgcn_s_sleep(1);
                // Only waiters acquire the successful sense load. The final
                // leader already acquired all arrivals before publishing sense.
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
            }
        }
        // HIP's global+local workgroup release/barrier/acquire distributes
        // the leader's imported payload to every lane without another agent fence.
        __syncthreads();
#endif
        ++phase;
    }
};
/** @brief Preserve the standalone FFN's exact two-phase/four-word barrier ABI. */
using ResidentGridBarrier=ResidentGridBarrierFor<NativeLayoutBarrierLayout::phases>;
}
