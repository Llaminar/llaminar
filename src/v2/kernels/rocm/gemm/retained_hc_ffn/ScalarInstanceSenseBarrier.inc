/**
 * @file ScalarInstanceSenseBarrier.inc
 * @brief Source-review candidate: scalar issuer for canonical instance arrival/sense slots.
 *
 * The existing caller owns residency, PMA bytes and exact event-exclusive reuse.
 * This helper neither admits concurrency nor allocates/reset slots on the host.
 * Its hardware ordering/lowering still requires strict ISA/native qualification.
 */
struct ScalarInstanceSenseJoin {
    using Layout=::llaminar2::FullNativeHCFFNBarrierLayout;
    unsigned* words; ///< Original instance-owned retained56B bank.
    unsigned phase=0; ///< Fresh invocation ordinal; stored senses persist.
    static_assert(Layout::words_per_phase==2&&Layout::suffix_word_offset==10&&Layout::suffix_words==4);
    /** @brief Completed coherent scalar word load, never an immutable/AS4 borrow. */
    __device__ __forceinline__ static unsigned load(unsigned* address) {
        unsigned value;
        asm volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
            : "=&s"(value) : "s"(address) : "memory");
        return value;
    }
    /** @brief One scalar RMW per signaling wave; return is completed before use. */
    __device__ __forceinline__ static unsigned arrive(unsigned* address) {
        unsigned ticket=1;
        asm volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
            : "+&s"(ticket) : "s"(address) : "memory");
        return ticket;
    }
    /** @brief Complete reset/publication as scalar side effects, not64 lane stores. */
    __device__ __forceinline__ static void replace(unsigned* address,unsigned value) {
        asm volatile("s_atomic_swap %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
            : "+&s"(value) : "s"(address) : "memory");
    }
    /**
     * @brief Release payloads, reset arrival then publish flipped sense, acquire peers.
     * @details All lanes enter both original LDS joins. Only physical wave0 may
     * execute scalar side effects; actual ISA must skip them for other waves.
     * Existing event retirement prevents slot reuse until every CTA has exited.
     */
    __device__ __forceinline__ void sync() {
        if(!words||phase>=5)__builtin_trap();
        const unsigned word=phase<3?2*phase:Layout::suffix_word_offset+2*(phase-3);
        unsigned* arrival=words+word;
        unsigned* sense=arrival+1;
        __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        lds_sync();
        if(__builtin_amdgcn_readfirstlane(threadIdx.x)<64) {
            const unsigned before=load(sense);
            const unsigned ticket=arrive(arrival);
            if(ticket>=gridDim.x)__builtin_trap();
            if(ticket+1==gridDim.x) {
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
                replace(arrival,0);
                __builtin_amdgcn_fence(__ATOMIC_RELEASE,"agent");
                replace(sense,before^1u);
            } else {
                while(load(sense)==before)__builtin_amdgcn_s_sleep(0);
                __builtin_amdgcn_fence(__ATOMIC_ACQUIRE,"agent");
            }
            asm volatile("buffer_wbinvl1_vol" ::: "memory");
        }
        lds_sync();
        ++phase;
    }
};
