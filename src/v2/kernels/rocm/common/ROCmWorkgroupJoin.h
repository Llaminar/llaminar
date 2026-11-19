/**
 * @file ROCmWorkgroupJoin.h
 * @brief Explicit workgroup memory scope for phase-local ROCm joins.
 *
 * LDS publication and retirement need a local-memory edge, not a drain of
 * unrelated global stores. Cross-CTA/public-global publication remains with
 * the original device-scope joins and native kernel completion. This wraps
 * the selected HIP SDK's compiler-aware workgroup barrier on other targets.
 * gfx906 uses the source-authenticated Castagnia local primitive explicitly;
 * its compiler-aware SDK barrier otherwise drains unrelated VMEM as well.
 */
#pragma once
#include <hip/hip_runtime.h>

namespace llaminar2::rocm {
/** @brief Address spaces whose producer/consumer edge this workgroup join owns. */
enum class WorkgroupJoinMemory {
    GlobalAndLDS, ///< Existing standalone HIP __syncthreads semantics.
    LDSOnly ///< Every cross-lane dependency is in this CTA's local storage.
};
/**
 * @brief Join all physical CTA lanes with the declared compiler/hardware memory scope.
 * @tparam Memory Exact producer/consumer edge; no runtime mode or inferred scope.
 * @pre Every physical lane reaches the join under the same participation as
 * the replaced __syncthreads. LDSOnly callers must not use the join to publish
 * or acquire another lane's global writes. Their required global loads have
 * already produced register/LDS operands; unrelated global stores may remain
 * pending until the existing device-scope publication or kernel completion.
 * @details gfx906 uses the exact pinned source local wait/barrier with a
 * compiler memory clobber: LDS stores/reads retire before peer reuse while
 * unrelated global requests need not drain. Other targets use the SDK local
 * release/barrier/acquire arm. Convergence and full-CTA participation remain
 * required on both paths; the memory clobber prevents compiler motion of LDS
 * accesses across the native join. No global-memory publication is claimed.
 */
template<WorkgroupJoinMemory Memory>
__device__ __forceinline__ __attribute__((convergent)) void workgroupJoin()
{
    static_assert(Memory==WorkgroupJoinMemory::GlobalAndLDS || Memory==WorkgroupJoinMemory::LDSOnly);
    if constexpr(Memory==WorkgroupJoinMemory::LDSOnly)
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
        // Exact hc-persist.cu::lds_sync primitive, selected explicitly for
        // gfx906. The compiler's intrinsic barrier also drains VMEM here;
        // this phase owns only an LDS dependency, proved by the caller.
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
        __work_group_barrier(static_cast<__cl_mem_fence_flags>(__CLK_LOCAL_MEM_FENCE));
#endif
    }
    else
        __syncthreads();
}
}
