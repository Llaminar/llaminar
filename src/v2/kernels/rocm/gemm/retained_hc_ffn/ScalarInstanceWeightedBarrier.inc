/**
 * @file ScalarInstanceWeightedBarrier.inc
 * @brief Single-RMW weighted scalar arrivals on the existing instance phase bank.
 *
 * All positive contributions sum to one immutable stride, so a generation can
 * advance only after every admitted CTA arrives. Existing event-exclusive reuse,
 * payload release/acquire and cache obligations remain authoritative.
 */
struct ScalarInstanceWeightedJoin {
    using Layout=::llaminar2::FullNativeHCFFNBarrierLayout;
    static constexpr unsigned stride=ScalarInstancePackedJoin::stride;
    static constexpr unsigned mask=stride-1;
    unsigned* words; ///< Same retained56B allocation; second words stay unused.
    unsigned phase=0; ///< Invocation ordinal; stored generations persist.
    /**
     * @brief Publish one positive weighted arrival and acquire the completed generation.
     * @return Prior counter for the diagnostic issuer; real callers discard it.
     * @details CTA0 adds S-N+1 and each other CTA adds1. Only wave0 issues the
     * scalar RMW. No last detector, padding RMW or reset store exists. The same
     * all-wave entry release/drains and every post-poll acquire/invalidation stay.
     */
    __device__ __forceinline__ unsigned sync() {
        if(!words||phase>=5)__builtin_trap();
        const unsigned word=phase<3?2*phase:Layout::suffix_word_offset+2*(phase-3);
        unsigned* counter=words+word;
        unsigned old=0;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        lds_sync();
        if(__builtin_amdgcn_readfirstlane(threadIdx.x)<64) {
            const unsigned members=gridDim.x;
            if(!members||members>stride)__builtin_trap();
            const unsigned increment=blockIdx.x==0?stride-members+1:1;
            old=ScalarInstancePackedJoin::add(counter,increment);
            const unsigned before=old&~mask;
            while((ScalarInstancePackedJoin::load(counter)&~mask)==before)__builtin_amdgcn_s_sleep(0);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
            // The acquire queues the CU-wide vector L1 invalidation. Retire
            // its VMEM acknowledgment before the LDS-only join releases
            // other waves to consume cross-CTA global writes.
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        lds_sync();
        ++phase;
        return old;
    }
};
