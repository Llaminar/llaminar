/**
 * @file ScalarInstancePackedBarrier.inc
 * @brief Packed scalar arrival/generation protocol on existing instance phase words.
 *
 * Stride is immutable throughout this retained counter's lifetime. This private
 * body already seals NB60; S64 derives from that admitted bound, never from a
 * changing live membership on dirty storage. Existing retirement owns reuse.
 */
struct ScalarInstancePackedJoin {
    using Layout=::llaminar2::FullNativeHCFFNBarrierLayout;
    static constexpr unsigned stride=std::bit_ceil(unsigned(NB));
    static constexpr unsigned mask=stride-1;
    static_assert(stride==64&&stride>NB&&Layout::suffix_word_offset==10);
    unsigned* words; ///< Same56B owner; second word in every live pair stays unused.
    unsigned phase=0; ///< Invocation-local ordinal; generation bits persist.
    /** @brief Completed coherent mutable scalar load. @param address Original owned word. @return Raw counter. */
    __device__ __forceinline__ static unsigned load(unsigned* address) {
        unsigned value;
        asm volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
            : "=&s"(value) : "s"(address) : "memory");
        return value;
    }
    /** @brief Completed scalar RMW. @param address Original counter. @param increment Positive arrival/padding. @return Prior raw counter. */
    __device__ __forceinline__ static unsigned add(unsigned* address,unsigned increment) {
        asm volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
            : "+&s"(increment) : "s"(address) : "memory");
        return increment;
    }
    /**
     * @brief Publish arrivals, complete last-member padding, acquire changed generation.
     * @details All original entry/exit joins, drains, agent fences and cache
     * invalidation remain. No absolute comparison/division, host reset or extra
     * allocation exists. The following invocation cannot reuse storage before
     * the original exact completion edge. Only wave0 issues scalar side effects.
     */
    __device__ __forceinline__ void sync() {
        if(!words||phase>=5)__builtin_trap();
        const unsigned word=phase<3?2*phase:Layout::suffix_word_offset+2*(phase-3);
        unsigned* counter=words+word;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        lds_sync();
        if(__builtin_amdgcn_readfirstlane(threadIdx.x)<64) {
            const unsigned members=gridDim.x;
            if(!members||members>stride)__builtin_trap();
            const unsigned old=add(counter,1);
            const unsigned ticket=old&mask;
            const unsigned before=old&~mask;
            if(ticket>=members)__builtin_trap();
            if(ticket+1==members) {
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
                __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
                if(stride!=members)add(counter,stride-members);
            }
            while((load(counter)&~mask)==before)__builtin_amdgcn_s_sleep(0);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
            asm volatile("buffer_wbinvl1_vol" ::: "memory");
        }
        lds_sync();
        ++phase;
    }
};
