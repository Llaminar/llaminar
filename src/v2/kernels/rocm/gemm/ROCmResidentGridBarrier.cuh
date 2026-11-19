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
                    __builtin_amdgcn_s_sleep(0);
                // Only waiters acquire the successful sense load. The final
                // leader already acquired all arrivals before publishing sense.
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
            }
        }
        // HIP's global+local workgroup release/barrier/acquire distributes
        // the leader's imported payload to every lane without another agent fence.
        __syncthreads();
        ++phase;
    }
};
/** @brief Preserve the standalone FFN's exact two-phase/four-word barrier ABI. */
using ResidentGridBarrier=ResidentGridBarrierFor<NativeLayoutBarrierLayout::phases>;
}
