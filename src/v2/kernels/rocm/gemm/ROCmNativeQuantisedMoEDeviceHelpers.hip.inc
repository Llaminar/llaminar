/**
 * @file ROCmNativeQuantisedMoEDeviceHelpers.hip.inc
 * @brief Shared inline device MoE decode/fold helpers for grouped decode, resident FFN, and native-layout HC/FFN shards.
 */
#pragma once
#include "ROCmNativeQuantisedDecodeCommon.hip.inc"


/**
 * @brief Decode one canonical Q4 four-code group from one packed nibble word.
 * @tparam FMT Q4_0 or Q4_1 physical decoder.
 * @param raw Original four nibble-pairs, without scale/minimum metadata.
 * @param high Select original elements16..31 rather than0..15.
 * @return Four original signed/unsigned int8 codes packed in one int32.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ int32_t moe_decode_q4_group(uint32_t raw,bool high)
{
    static_assert(FMT==NATIVE_QUANTISED_Q4_0 || FMT==NATIVE_QUANTISED_Q4_1);
    const uint32_t nibbles=(high?raw>>4:raw)&0x0F0F0F0Fu;
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::center_4bit==8)
        return static_cast<int32_t>((nibbles+0x78787878u)^0x80808080u);
    else return static_cast<int32_t>(nibbles);
}
/**
 * @brief Decode one canonical Q5 four-code group, retaining original centering.
 * @tparam FMT Q5_0 or Q5_1 physical decoder.
 * @param raw Original four nibble-pairs.
 * @param high_bits Original32-bit fifth-bit plane.
 * @param group Original group index0..7; indices4..7 select high nibbles.
 * @return Four original signed/unsigned int8 codes packed in one int32.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ int32_t moe_decode_q5_group(uint32_t raw,uint32_t high_bits,unsigned group)
{
    static_assert(FMT==NATIVE_QUANTISED_Q5_0 || FMT==NATIVE_QUANTISED_Q5_1);
    const uint32_t nibbles=(group>=4?raw>>4:raw)&0x0F0F0F0Fu;
    const uint32_t hb4=(high_bits>>(group*4))&0xFu;
    // Share the exact four-bit deposit; keep group selection and Q5 centering
    // unchanged while avoiding four independently live shift/mask terms.
    const uint32_t hb=::llaminar2::native_quantised::q5HighBitsToPackedBytes(hb4);
    uint32_t value=nibbles|hb;
    if constexpr(NATIVE_QUANTISEDTraits<FMT>::center_5bit==16)
    {value^=0x80808080u;value-=0x10101010u;value^=0x80808080u;}
    return static_cast<int32_t>(value);
}

/**
 * @brief Load one packed input word at the current ordered integer-dot boundary.
 * @tparam Ordered NativeLayout's scheduling-only raw-input dependency; false for legacy callers.
 * @param payload Original canonical payload. @param offset Original byte offset.
 * @param previous_dot Prior integer dot; never changes loaded bits.
 * @return Original word, with no allocation, fence or numerical operation.
 */
template<bool Ordered>
__device__ __forceinline__ uint32_t moe_native_group_word(const uint8_t* payload,unsigned offset,int32_t previous_dot)
{
    uint32_t raw=*reinterpret_cast<const uint32_t*>(payload+offset);
    if constexpr(Ordered)asm volatile("" : "+v"(raw) : "v"(previous_dot));
    return raw;
}
/**
 * @brief Load one bounded packed byte at the current integer-dot boundary.
 * @tparam Ordered Whether to retain the scheduling-only dependency.
 * @param payload Original payload. @param offset In-bounds original byte offset.
 * @param previous_dot Prior integer dot. @return Original byte widened without sign extension.
 */
template<bool Ordered>
__device__ __forceinline__ uint32_t moe_native_group_byte(const uint8_t* payload,unsigned offset,int32_t previous_dot)
{
    uint32_t raw=payload[offset];
    if constexpr(Ordered)asm volatile("" : "+v"(raw) : "v"(previous_dot));
    return raw;
}
/**
 * @brief Default immutable device-table lookup authority for canonical IQ decode.
 * @details Explicit full-entry views may borrow byte-identical phase-local LDS;
 * all existing owners retain these original module-table reads by default.
 */
struct NativeQuantisedGridLookup {
    /** @brief Read the original four-code IQ3 entry. @tparam FMT Original format. @param index Original decoded index. @return Unchanged packed bytes. */
    template<NativeQuantisedFormat FMT> __device__ __forceinline__ uint32_t iq3(unsigned index) const {return iq3_grid_lookup<FMT>(index);}
    /** @brief Read the original eight-code IQ2 entry. @tparam FMT Original format. @param index Original decoded index. @return Unchanged packed bytes. */
    template<NativeQuantisedFormat FMT> __device__ __forceinline__ uint64_t iq2(unsigned index) const {return iq2_grid_lookup<FMT>(index);}
    /** @brief Read the original IQ1 entry. @param index Original decoded index. @return Unchanged packed bytes. */
    __device__ __forceinline__ uint64_t iq1(unsigned index) const {return iq1_grid_lookup(index);}
};
/**
 * @brief Decode exactly one original four-code group, without expanding future groups.
 * @tparam FMT Canonical physical codebook from the existing supported inventory.
 * @tparam Ordered Apply NativeLayout's register-only dependency to every raw packed input.
 * @tparam Grid Explicit immutable lookup view; existing owners use original device tables.
 * @param payload Original canonical block, including original IQ sign/tail bytes.
 * @param group Original output group0..7. @param previous_dot Prior integer result, unchanged by decode.
 * @param table_view Lookup authority with byte-identical entries, valid for this complete consumer.
 * @return The identical four signed/unsigned int8 codes packed in one integer word.
 * @details The legacy eight-group API delegates here with Ordered=false. Ordered
 * mode delays raw inputs before mapping, rather than serializing already-expanded
 * values. Grid lookups and bit mappings are the original canonical operations.
 */
template<NativeQuantisedFormat FMT,bool Ordered=false,class Grid=NativeQuantisedGridLookup>
__device__ __forceinline__ int32_t moe_decode_native_quantised_group(
    const uint8_t* __restrict__ payload,unsigned group,int32_t previous_dot=0,Grid table_view={})
{
    using Traits=NATIVE_QUANTISEDTraits<FMT>;
    static_assert(NATIVE_QUANTISEDGroupedMoESupported<FMT>::value,
        "Grouped MoE decode requires a supported native-quantised format");
    if constexpr(Traits::uses_lut)
    {
        constexpr uint32_t CB_LO_A=0xBFAD9881u,CB_LO_B=0xF6EADDCFu;
        constexpr uint32_t CB_HI_A=0x26190D01u,CB_HI_B=0x71594535u;
        constexpr uint32_t MUX_IDENTITY=0x03020100u,MUX_BIT3_MASK=0x04040404u;
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot);
        const uint32_t nibbles=(group>=4?raw>>4:raw)&0x0F0F0F0Fu;
        const uint32_t sel=nibbles&0x07070707u;
        const uint32_t r0=__builtin_amdgcn_perm(CB_LO_B,CB_LO_A,sel);
        const uint32_t r1=__builtin_amdgcn_perm(CB_HI_B,CB_HI_A,sel);
        const uint32_t mux=MUX_IDENTITY|((nibbles>>1)&MUX_BIT3_MASK);
        return static_cast<int32_t>(__builtin_amdgcn_perm(r1,r0,mux));
    }
    else if constexpr(Traits::is_5bit)
    {
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot);
        const uint32_t high=moe_native_group_word<Ordered>(payload,16,previous_dot);
        return moe_decode_q5_group<FMT>(raw,high,group);
    }
    else if constexpr(Traits::is_3bit)
    {
        constexpr uint32_t BIT_SPREAD=0x00204081u;
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group/4)*4,previous_dot);
        const uint32_t high=moe_native_group_word<Ordered>(payload,8,previous_dot);
        const uint32_t low2=(raw>>((group%4)*2))&0x03030303u;
        const uint32_t hb4=(high>>(group*4))&0xFu;
        const uint32_t hb=((hb4*BIT_SPREAD)&0x01010101u)<<2;
        uint32_t value=(low2|hb)^0x80808080u;
        value-=0x04040404u;value^=0x80808080u;
        return static_cast<int32_t>(value);
    }
    else if constexpr(Traits::is_2bit)
    {
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group/4)*4,previous_dot);
        return static_cast<int32_t>((raw>>((group%4)*2))&0x03030303u);
    }
    else if constexpr(Traits::is_6bit)
    {
        const uint32_t raw=moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot);
        // Q6's24-byte payload has two complete aligned high-bit words.
        // Keep that word, rather than its extracted byte, behind the current
        // dot boundary; byte extraction and dilation then remain group-local.
        const uint32_t high_word=moe_native_group_word<Ordered>(payload,16+(group/4)*4,previous_dot);
        const uint32_t high=(high_word>>((group%4)*8))&0xFFu;
        const uint32_t nibbles=(group>=4?raw>>4:raw)&0x0F0F0F0Fu;
        const uint32_t hb=rocm_native_quantised_expand_q6_high_fields(high);
        uint32_t value=(nibbles|hb)^0x80808080u;
        value-=0x20202020u;value^=0x80808080u;
        return static_cast<int32_t>(value);
    }
    else if constexpr(Traits::is_iq3_grid)
    {
        const uint32_t qs=moe_native_group_word<Ordered>(payload,(group/4)*4,previous_dot);
        const uint32_t signs=moe_native_group_word<Ordered>(payload,8,previous_dot);
        int idx=(qs>>((group%4)*8))&0xFF;
        uint8_t sign_byte;
        if constexpr(Traits::has_qh)
        {
            idx|=static_cast<int>((signs>>group)&1)<<8;
            sign_byte=group<6?static_cast<uint8_t>(signs>>((group/2+1)*8)):
                static_cast<uint8_t>(moe_native_group_byte<Ordered>(payload,12,previous_dot));
        }
        else sign_byte=static_cast<uint8_t>(signs>>((group/2)*8));
        const uint8_t sign_nibble=group&1?sign_byte>>4:sign_byte&0xF;
        return static_cast<int32_t>(iq_apply_signs_4(table_view.template iq3<FMT>(idx),sign_nibble));
    }
    else if constexpr(Traits::is_iq2_grid)
    {
        const unsigned lookup=group/2;
        const uint32_t qs=moe_native_group_word<Ordered>(payload,0,previous_dot);
        const uint32_t signs=moe_native_group_word<Ordered>(payload,4,previous_dot);
        int idx=(qs>>(lookup*8))&0xFF;
        uint8_t sign_byte;
        if constexpr(Traits::has_qh)
        {
            if constexpr(Traits::qh_bits_per_lookup==1)idx|=static_cast<int>((signs>>lookup)&1)<<8;
            else idx|=static_cast<int>((signs>>(2*lookup))&3)<<8;
            sign_byte=lookup<3?static_cast<uint8_t>(signs>>((lookup+1)*8)):
                static_cast<uint8_t>(moe_native_group_byte<Ordered>(payload,8,previous_dot));
        }
        else sign_byte=static_cast<uint8_t>(signs>>(lookup*8));
        const uint64_t grid=table_view.template iq2<FMT>(idx);
        const uint32_t word=group&1?static_cast<uint32_t>(grid>>32):static_cast<uint32_t>(grid);
        const uint8_t sign_nibble=group&1?sign_byte>>4:sign_byte&0xF;
        return static_cast<int32_t>(iq_apply_signs_4(word,sign_nibble));
    }
    else if constexpr(Traits::is_iq1_grid)
    {
        const unsigned lookup=group/2;
        const uint32_t qs=moe_native_group_byte<Ordered>(payload,lookup,previous_dot);
        unsigned high;
        if constexpr(FMT==NATIVE_QUANTISED_IQ1_S)
        {
            const unsigned h0=moe_native_group_byte<Ordered>(payload,4,previous_dot);
            const unsigned h1=moe_native_group_byte<Ordered>(payload,5,previous_dot);
            high=((h0|(h1<<8))>>(3*lookup))&7;
        }
        else high=(moe_native_group_byte<Ordered>(payload,4+lookup/2,previous_dot)>>((lookup%2)*4))&7;
        const uint64_t grid=table_view.iq1(qs|(high<<8));
        return group&1?static_cast<int32_t>(static_cast<uint32_t>(grid>>32)):
            static_cast<int32_t>(static_cast<uint32_t>(grid));
    }
    else if constexpr(Traits::is_8bit)
        return static_cast<int32_t>(moe_native_group_word<Ordered>(payload,group*4,previous_dot));
    else return moe_decode_q4_group<FMT>(moe_native_group_word<Ordered>(payload,(group%4)*4,previous_dot),group>=4);
}

/**
 * @brief Decode the original eight groups through the canonical single-group mapping.
 * @tparam FMT Existing supported physical decoder.
 * @param payload Original canonical block. @param packed_groups Receives groups0..7 unchanged.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void moe_decode_native_quantised_groups(
    const uint8_t* __restrict__ payload,int32_t* packed_groups)
{
    #pragma unroll
    for(unsigned group=0;group<8;++group)
        packed_groups[group]=moe_decode_native_quantised_group<FMT>(payload,group);
}

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ float moe_native_quantised_dot_desc_range(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K,
    int b_start,
    int b_end);

/*
 * Grouped MoE decode is defined before the generic verifier-row kernels in this
 * translation unit, but both paths must call the same arithmetic primitives.
 * Keep these declarations here so the grouped dot cannot grow a second copy of
 * the serial M=1 accumulation tree.
 */
template <NativeQuantisedFormat FMT>
__device__ __forceinline__ int32_t native_quantised_activation_sum_a(
    const int32_t* __restrict__ a4);

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ void native_quantised_small_m_accumulate_decoded_block(
    float& f_acc,
    const int32_t* __restrict__ a4,
    const int32_t* __restrict__ packed_groups,
    const uint8_t* __restrict__ payload,
    const uint16_t* __restrict__ scale_base,
    const uint16_t* __restrict__ min_base,
    const uint32_t* __restrict__ emin_base,
    size_t linear,
    float scale_ab,
    int32_t sum_a);

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ float moe_native_quantised_dot_desc_range(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K,
    int b_start,
    int b_end)
{
    if (!A_int8 ||
        !scales_A_blockwise ||
        n < 0 ||
        n >= N ||
        K <= 0 ||
        (K % 32) != 0 ||
        !native_quantised_desc_shape_ok<FMT>(desc, N, K))
    {
        return 0.0f;
    }

    const int blocks_per_row = K / 32;
    const uint8_t* payload_base = desc.payload;
    const uint16_t* scale_base = static_cast<const uint16_t*>(desc.scales);
    const uint16_t* min_base = static_cast<const uint16_t*>(desc.mins);
    const uint32_t* emin_base = static_cast<const uint32_t*>(desc.emins);
    float acc = 0.0f;

    if (b_start < 0) b_start = 0;
    if (b_end > blocks_per_row) b_end = blocks_per_row;

    for (int b = b_start; b < b_end; ++b)
    {
        const int32_t* a4 = reinterpret_cast<const int32_t*>(A_int8 + b * 32);
        int32_t packed_groups[8];
        const size_t linear = static_cast<size_t>(b) * N + static_cast<size_t>(n);
        const uint8_t* payload = payload_base + linear * NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
        moe_decode_native_quantised_groups<FMT>(payload, packed_groups);

        const float scale_ab = scales_A_blockwise[b];
        const int32_t sum_a = native_quantised_activation_sum_a<FMT>(a4);
        // Mixed IQ4_NL/IQ4_XS pairs share only integer decode. Each actual
        // descriptor retains its own scale program and optional raw word.
        if constexpr (FMT == NATIVE_QUANTISED_IQ4_XS_SOURCE)
        {
            if (desc.codebook_id == NATIVE_QUANTISED_IQ4_NL)
                native_quantised_small_m_accumulate_decoded_block<NATIVE_QUANTISED_IQ4_NL>(
                    acc, a4, packed_groups, payload, scale_base, min_base,
                    emin_base, linear, scale_ab, sum_a);
            else
                native_quantised_small_m_accumulate_decoded_block<FMT>(
                    acc, a4, packed_groups, payload, scale_base, min_base,
                    emin_base, linear, scale_ab, sum_a);
        }
        else
        {
            native_quantised_small_m_accumulate_decoded_block<FMT>(
                acc, a4, packed_groups, payload, scale_base, min_base,
                emin_base, linear, scale_ab, sum_a);
        }
    }

    return acc;
}

template <NativeQuantisedFormat FMT>
__device__ __forceinline__ float moe_native_quantised_dot_desc(
    const DeviceNativeQuantisedMatrixDesc& desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K)
{
    return moe_native_quantised_dot_desc_range<FMT>(
        desc, n, A_int8, scales_A_blockwise, N, K, 0, K / 32);
}

/**
 * @brief Return the canonical source policy carried by a device descriptor.
 *
 * Q8_1 and Q8_K keep distinct source identities but intentionally share the
 * Q8_0 accelerator arithmetic policy.  Expanded codebook 23 never supplies a
 * policy of its own: its authoritative source identity selects the reduction
 * tree that existed before migration.
 */
__device__ __forceinline__ uint8_t
moe_native_quantised_descriptor_policy_codebook(
    const DeviceNativeQuantisedMatrixDesc* desc)
{
    if (!desc)
        return 0xffu;
    uint8_t policy = desc->source_identity_present
                         ? desc->source_codebook_id
                         : desc->codebook_id;
    if (policy == 20 || policy == 21)
        policy = 19;
    return policy;
}

/**
 * @brief Resolve one descriptor's exact serial-M1 K-partition count on device.
 */
__device__ __forceinline__ int moe_native_quantised_descriptor_k_partitions(
    const DeviceNativeQuantisedMatrixDesc* desc,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31)
{
    const uint8_t policy =
        moe_native_quantised_descriptor_policy_codebook(desc);
    if (policy >= ROCmMoESerialM1PolicyTable::kCodebookCapacity)
        return 0;

    uint64_t packed = 0;
    switch (policy >> 3u)
    {
    case 0:
        packed = policy_partitions_0_7;
        break;
    case 1:
        packed = policy_partitions_8_15;
        break;
    case 2:
        packed = policy_partitions_16_23;
        break;
    case 3:
        packed = policy_partitions_24_31;
        break;
    default:
        return 0;
    }
    const unsigned shift = static_cast<unsigned>(policy & 7u) * 8u;
    return static_cast<int>((packed >> shift) & uint64_t{0xff});
}

/**
 * @brief Decode one descriptor using its physical execution codebook.
 * @tparam DecoderMask Cold immutable compiled decoder family; original generic callers retain the complete mask.
 *
 * The switch is uniform for every lane in a `(route, K-partition)` block.  It
 * therefore preserves the specialized decoder bodies while allowing a single
 * captured graph to consume compact and streamed-expanded expert slots.
 */
template<std::uint32_t DecoderMask=llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders>
__device__ __forceinline__ float moe_native_quantised_dot_desc_range_dynamic(
    const DeviceNativeQuantisedMatrixDesc* desc,
    int n,
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    int N,
    int K,
    int b_start,
    int b_end)
{
    static_assert(DecoderMask!=0&&(DecoderMask&~llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders)==0);
    if (!desc)
        return 0.0f;
    switch (static_cast<NativeQuantisedFormat>(desc->codebook_id))
    {
    case NATIVE_QUANTISED_Q4_0:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q4_0))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q4_0>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ4_NL:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ4_NL))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ4_NL>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ4_XS_SOURCE:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ4_XS_SOURCE))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ4_XS_SOURCE>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q4_1:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q4_1))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q4_1>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q5_0:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q5_0))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q5_0>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q5_1:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q5_1))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q5_1>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q6_K:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q6_K))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q6_K>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q3_K:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q3_K))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q3_K>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q2_K:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q2_K))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q2_K>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ3_S:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ3_S))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ3_S>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ3_XXS:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ3_XXS))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ3_XXS>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ2_S:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ2_S))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ2_S>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ2_XS:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ2_XS))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ2_XS>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ2_XXS:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ2_XXS))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ2_XXS>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ1_S:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ1_S))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ1_S>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_IQ1_M:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_IQ1_M))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_IQ1_M>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_Q8_0:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_Q8_0))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_Q8_0>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    case NATIVE_QUANTISED_EXPANDED_INT8_MIN:
        if constexpr((DecoderMask&(1u<<NATIVE_QUANTISED_EXPANDED_INT8_MIN))!=0)
            return moe_native_quantised_dot_desc_range<NATIVE_QUANTISED_EXPANDED_INT8_MIN>(
                *desc, n, A_int8, scales_A_blockwise, N, K, b_start, b_end);
        break;
    default:
        break;
    }
    // Both unknown physical formats and formats excluded by this admitted
    // compiled family are fatal; no wider decoder recovery is available.
    __builtin_trap();
    return 0.0f;
}

/**
 * @brief List exactly the emitted paired programs, not a second format policy.
 * @details The proof below exhausts the existing execution-decoder inventory
 * and canonical gate/up compatibility predicate. Extending either authority
 * without extending this closed emitted registry fails compilation.
 */
#define LLAMINAR_MOE_PAIRED_K32_PROGRAMS(Apply) \
    Apply(NATIVE_QUANTISED_Q4_0,NATIVE_QUANTISED_Q4_0) \
    Apply(NATIVE_QUANTISED_IQ4_NL,NATIVE_QUANTISED_IQ4_NL) \
    Apply(NATIVE_QUANTISED_IQ4_XS_SOURCE,NATIVE_QUANTISED_IQ4_XS_SOURCE) \
    Apply(NATIVE_QUANTISED_Q4_1,NATIVE_QUANTISED_Q4_1) \
    Apply(NATIVE_QUANTISED_Q5_0,NATIVE_QUANTISED_Q5_0) \
    Apply(NATIVE_QUANTISED_Q5_1,NATIVE_QUANTISED_Q5_1) \
    Apply(NATIVE_QUANTISED_Q6_K,NATIVE_QUANTISED_Q6_K) \
    Apply(NATIVE_QUANTISED_Q3_K,NATIVE_QUANTISED_Q3_K) \
    Apply(NATIVE_QUANTISED_Q2_K,NATIVE_QUANTISED_Q2_K) \
    Apply(NATIVE_QUANTISED_IQ3_S,NATIVE_QUANTISED_IQ3_S) \
    Apply(NATIVE_QUANTISED_IQ3_XXS,NATIVE_QUANTISED_IQ3_XXS) \
    Apply(NATIVE_QUANTISED_IQ2_S,NATIVE_QUANTISED_IQ2_S) \
    Apply(NATIVE_QUANTISED_IQ2_XS,NATIVE_QUANTISED_IQ2_XS) \
    Apply(NATIVE_QUANTISED_IQ2_XXS,NATIVE_QUANTISED_IQ2_XXS) \
    Apply(NATIVE_QUANTISED_IQ1_S,NATIVE_QUANTISED_IQ1_S) \
    Apply(NATIVE_QUANTISED_IQ1_M,NATIVE_QUANTISED_IQ1_M) \
    Apply(NATIVE_QUANTISED_Q8_0,NATIVE_QUANTISED_Q8_0) \
    Apply(NATIVE_QUANTISED_EXPANDED_INT8_MIN,NATIVE_QUANTISED_EXPANDED_INT8_MIN) \
    Apply(NATIVE_QUANTISED_IQ4_NL,NATIVE_QUANTISED_IQ4_XS_SOURCE) \
    Apply(NATIVE_QUANTISED_IQ4_XS_SOURCE,NATIVE_QUANTISED_IQ4_NL)

/** @return Whether the emitted programs exactly cover the canonical supported compatibility matrix. */
constexpr bool moePairedK32RegistryMatchesCanonical() noexcept
{
#define LLAMINAR_MOE_PAIRED_K32_RECORD(Gate,Up) std::array<unsigned,2>{unsigned(Gate),unsigned(Up)},
    constexpr std::array programs{LLAMINAR_MOE_PAIRED_K32_PROGRAMS(LLAMINAR_MOE_PAIRED_K32_RECORD)};
#undef LLAMINAR_MOE_PAIRED_K32_RECORD
    constexpr auto supported=llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders;
    unsigned gate_inventory=0,up_inventory=0;
    for(const auto& program:programs)
    {
        if(program[0]>=32||program[1]>=32)return false;
        gate_inventory|=1u<<program[0];up_inventory|=1u<<program[1];
    }
    if(gate_inventory!=supported||up_inventory!=supported)return false;
    for(unsigned gate=0;gate<32;++gate)
    for(unsigned up=0;up<32;++up)
    {
        unsigned matches=0;
        for(const auto& program:programs)matches+=(program[0]==gate&&program[1]==up);
        const bool canonical=(supported&(1u<<gate))&&(supported&(1u<<up))&&
            llaminar2::areNativeQuantisedGateUpCodebooksCompatible(static_cast<uint8_t>(gate),static_cast<uint8_t>(up));
        if(matches!=(canonical?1u:0u))return false;
    }
    return true;
}
static_assert(moePairedK32RegistryMatchesCanonical(),
    "Paired K32 programs must exactly cover the canonical decoder/compatibility inventory");

/**
 * @brief Consume one role's unchanged decoder and rounded K32 contribution.
 * @tparam FMT Original physical decoder, independent of the other role.
 * @param desc Original descriptor; original source-policy arithmetic stays outside execution dispatch.
 * @param linear Exact original [K32,N] coordinate.
 * @param a4 Once-loaded original eight activation words for this K32.
 * @param scale Original published FP32 activation scale.
 * @param sum Exact original eight-dot4 activation sum, shared only as an integer value.
 * @param accumulator This role's original increasing-block FP32 chain.
 * @note Only one decoded8word weight operand is live at a time. No local aggregate
 * address escapes this force-inlined call, and no new scratch allocation exists.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void moePairedK32Commit(const DeviceNativeQuantisedMatrixDesc& desc,size_t linear,
    const int32_t* a4,float scale,int32_t sum,float& accumulator)
{
    const auto* payload=desc.payload+linear*NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
    int32_t groups[8];
    moe_decode_native_quantised_groups<FMT>(payload,groups);
    native_quantised_small_m_accumulate_decoded_block<FMT>(accumulator,a4,groups,payload,
        static_cast<const uint16_t*>(desc.scales),static_cast<const uint16_t*>(desc.mins),
        static_cast<const uint32_t*>(desc.emins),linear,scale,sum);
}
/**
 * @brief Commit the unchanged canonical block from immutable planes captured before the loop.
 * @tparam FMT Original independently admitted execution decoder.
 * @param payload_base Original prepared payload plane; no descriptor aggregate copy.
 * @param scales Original primary metadata plane. @param mins Original secondary metadata plane.
 * @param emins Original dual-minimum plane. @param linear Original [K32,N] coordinate.
 * @param a4 Shared once-loaded activation words. @param scale Original FP32 sidecar value.
 * @param sum Shared exact integer sum. @param accumulator Original role-owned ascending chain.
 * @note The caller reads these pointers once before any rounded commit. Keeping
 * scalar plane values separate avoids descriptor rereads across numerical
 * compiler barriers without introducing another live allocation or byte authority.
 */
template<NativeQuantisedFormat FMT>
__device__ __forceinline__ void moePairedK32CommitPlanes(const uint8_t* payload_base,
    const uint16_t* scales,const uint16_t* mins,const uint32_t* emins,size_t linear,
    const int32_t* a4,float scale,int32_t sum,float& accumulator)
{
    const auto* payload=payload_base+linear*NATIVE_QUANTISEDTraits<FMT>::payload_bytes;
    int32_t groups[8];
    moe_decode_native_quantised_groups<FMT>(payload,groups);
    native_quantised_small_m_accumulate_decoded_block<FMT>(accumulator,a4,groups,payload,
        scales,mins,emins,linear,scale,sum);
}
/**
 * @brief Walk only the union of two original partition ranges, sharing identical activations.
 * @tparam Gate Original gate physical format selected once before the K32 loop.
 * @tparam Up Original up physical format selected by the same canonical compatibility registry.
 * @param gate Original gate descriptor. @param up Original up descriptor.
 * @param n Original column lane. @param input Original Q8 hidden row.
 * @param scales Original FP32 block scales. @param N Original column count.
 * @param K Original reduction width. @param gate_begin Gate's original first K32.
 * @param gate_end Gate's original exclusive last K32. @param up_begin Up's original first K32.
 * @param up_end Up's original exclusive last K32. @param gate_acc Original gate partial result.
 * @param up_acc Original up partial result.
 * @details Each role sees exactly its original ascending block subsequence and
 * canonical per-format contribution helper. Gaps are skipped before any input
 * load. Invalid original shapes contribute+0 exactly as the original range dot.
 * Integer sum sharing does not change IQ dual-half/subgroup correction helpers.
 */
template<NativeQuantisedFormat Gate,NativeQuantisedFormat Up>
__device__ __forceinline__ void moePairedK32RangeTyped(const DeviceNativeQuantisedMatrixDesc& gate,
    const DeviceNativeQuantisedMatrixDesc& up,int n,const int8_t* input,const float* scales,int N,int K,
    int gate_begin,int gate_end,int up_begin,int up_end,float& gate_acc,float& up_acc)
{
    static_assert(llaminar2::areNativeQuantisedGateUpCodebooksCompatible(uint8_t(Gate),uint8_t(Up)));
    gate_acc=0.0f;up_acc=0.0f;
    if(!input||!scales||n<0||n>=N||K<=0||K%32)return;
    const bool valid_gate=native_quantised_desc_shape_ok<Gate>(gate,N,K);
    const bool valid_up=native_quantised_desc_shape_ok<Up>(up,N,K);
    if(valid_gate&&valid_up&&gate_begin==up_begin&&gate_end==up_end)
    {
        // Authenticate the equality once. Each role still owns its original
        // range; equal ranges permit unconditional commits to independent chains.
        // Read immutable plane addresses before the loop, rather than requesting
        // descriptor fields again after a canonical rounded-operation barrier.
        const uint8_t* gate_payload=gate.payload;
        const uint16_t* gate_scales=static_cast<const uint16_t*>(gate.scales);
        const uint16_t* gate_mins=static_cast<const uint16_t*>(gate.mins);
        const uint32_t* gate_emins=static_cast<const uint32_t*>(gate.emins);
        const uint8_t* up_payload=up.payload;
        const uint16_t* up_scales=static_cast<const uint16_t*>(up.scales);
        const uint16_t* up_mins=static_cast<const uint16_t*>(up.mins);
        const uint32_t* up_emins=static_cast<const uint32_t*>(up.emins);
        #pragma unroll 1
        for(int b=gate_begin;b<gate_end;++b)
        {
            const auto* source=reinterpret_cast<const int32_t*>(input+size_t(b)*32u);
            int32_t activation[8];
            #pragma unroll
            for(int word=0;word<8;++word)activation[word]=source[word];
            const float scale=scales[b];
            int32_t sum=0;
            if constexpr(NATIVE_QUANTISEDTraits<Gate>::is_asymmetric||NATIVE_QUANTISEDTraits<Up>::is_asymmetric)
            {
                #pragma unroll
                for(int word=0;word<8;++word)
                    sum=::llaminar2::rocm_integer_contract::signedDot4(activation[word],0x01010101,sum);
            }
            const size_t linear=size_t(b)*size_t(N)+size_t(n);
            moePairedK32CommitPlanes<Gate>(gate_payload,gate_scales,gate_mins,gate_emins,
                linear,activation,scale,sum,gate_acc);
            moePairedK32CommitPlanes<Up>(up_payload,up_scales,up_mins,up_emins,
                linear,activation,scale,sum,up_acc);
        }
        return;
    }
    // Unequal ranges or one invalid original role retain the existing exact
    // union walk, including skipped gaps, role-specific+0 and source metadata.
    const int first=min(gate_begin,up_begin),last=max(gate_end,up_end);
    #pragma unroll 1
    for(int b=first;b<last;++b)
    {
        const bool active_gate=valid_gate&&b>=gate_begin&&b<gate_end;
        const bool active_up=valid_up&&b>=up_begin&&b<up_end;
        if(!active_gate&&!active_up)continue; // No activation read or padded contribution in a gap.
        const auto* source=reinterpret_cast<const int32_t*>(input+size_t(b)*32u);
        int32_t activation[8];
        #pragma unroll
        for(int word=0;word<8;++word)activation[word]=source[word];
        const float scale=scales[b];
        const bool needs_sum=(active_gate&&NATIVE_QUANTISEDTraits<Gate>::is_asymmetric)||
            (active_up&&NATIVE_QUANTISEDTraits<Up>::is_asymmetric);
        int32_t sum=0;
        if(needs_sum)
        {
            #pragma unroll
            for(int word=0;word<8;++word)
                sum=::llaminar2::rocm_integer_contract::signedDot4(activation[word],0x01010101,sum);
        }
        const size_t linear=size_t(b)*size_t(N)+size_t(n);
        if(active_gate)moePairedK32Commit<Gate>(gate,linear,activation,scale,sum,gate_acc);
        if(active_up)moePairedK32Commit<Up>(up,linear,activation,scale,sum,up_acc);
    }
}
/**
 * @brief Select exactly one admitted format pair before entering the original K32 partition walk.
 * @param gate Original gate descriptor. @param up Original up descriptor.
 * @param n Original column lane. @param input Original Q8 hidden row.
 * @param scales Original FP32 block scales. @param N Original column count.
 * @param K Original reduction width. @param gate_begin Gate's original first K32.
 * @param gate_end Gate's original exclusive last K32. @param up_begin Up's original first K32.
 * @param up_end Up's original exclusive last K32. @param gate_acc Original gate partial result.
 * @param up_acc Original up partial result.
 * @note The registry is compile-time checked against the existing canonical
 * predicate. An unknown or incompatible pair traps; no per-block dispatcher,
 * wider Cartesian family or automatic generic recovery exists.
 */
__device__ __forceinline__ void moePairedK32Range(const DeviceNativeQuantisedMatrixDesc& gate,
    const DeviceNativeQuantisedMatrixDesc& up,int n,const int8_t* input,const float* scales,int N,int K,
    int gate_begin,int gate_end,int up_begin,int up_end,float& gate_acc,float& up_acc)
{
    const unsigned key=(unsigned(gate.codebook_id)<<8)|unsigned(up.codebook_id);
    switch(key)
    {
#define LLAMINAR_MOE_PAIRED_K32_CASE(Gate,Up) \
    case (unsigned(Gate)<<8)|unsigned(Up): \
        moePairedK32RangeTyped<Gate,Up>(gate,up,n,input,scales,N,K,gate_begin,gate_end,up_begin,up_end,gate_acc,up_acc);return;
        LLAMINAR_MOE_PAIRED_K32_PROGRAMS(LLAMINAR_MOE_PAIRED_K32_CASE)
#undef LLAMINAR_MOE_PAIRED_K32_CASE
    default:__builtin_trap();return;
    }
}
#undef LLAMINAR_MOE_PAIRED_K32_PROGRAMS

/**
 * @brief Load gate/up descriptors from a static table or the active runtime bank.
 */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ bool moe_native_quantised_load_gate_up_descriptors(
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    int expert_id,
    int num_experts,
    const DeviceNativeQuantisedMatrixDesc*& gate_desc,
    const DeviceNativeQuantisedMatrixDesc*& up_desc)
{
    if (expert_id < 0 || expert_id >= num_experts)
        return false;
    if constexpr (RUNTIME_TABLE)
    {
        const uint32_t execution_bank =
            moe_runtime_execution_bank(runtime);
        if (execution_bank > 1u)
            return false;
        const DeviceMoEPlacementBankView& bank =
            moe_runtime_placement_banks(runtime)[execution_bank];
        if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id)) return false;
        gate_desc = &bank.experts[expert_id].gate;
        up_desc = &bank.experts[expert_id].up;
    }
    else
    {
        gate_desc = &gate_descs[expert_id];
        up_desc = &up_descs[expert_id];
    }
    return true;
}

/** @brief Load a down descriptor from a static table or active runtime bank. */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ bool moe_native_quantised_load_down_descriptor(
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    int expert_id,
    int num_experts,
    const DeviceNativeQuantisedMatrixDesc*& down_desc)
{
    if (expert_id < 0 || expert_id >= num_experts)
        return false;
    if constexpr (RUNTIME_TABLE)
    {
        const uint32_t execution_bank =
            moe_runtime_execution_bank(runtime);
        if (execution_bank > 1u)
            return false;
        const DeviceMoEPlacementBankView& bank =
            moe_runtime_placement_banks(runtime)[execution_bank];
        if (!moe_runtime_native_descriptor_selected(runtime, bank, expert_id)) return false;
        down_desc = &bank.experts[expert_id].down;
    }
    else
    {
        down_desc = &down_descs[expert_id];
    }
    return true;
}

/**
 * @brief Produce descriptor-local gate/up split-K partials for a stable graph.
 */
/**
 * @brief Execute one original indexed64-column split-K producer task.
 * @details Original wrapper and last-producer entry share the exact body;
 * physical completion metadata cannot change source-policy arithmetic.
 */
template <bool RUNTIME_TABLE,bool PairedK32=false>
__device__ __forceinline__ void moeGateUpOriginalPartialTask(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    unsigned logical_tile,unsigned logical_route,unsigned logical_part,unsigned logical_lane)
{
    constexpr int TILE_N = 64;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    const int k_part = logical_part;
    if (route >= num_active || n >= N ||
        k_part >= maximum_k_partitions)
    {
        return;
    }

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const DeviceNativeQuantisedMatrixDesc* gate_desc = nullptr;
    const DeviceNativeQuantisedMatrixDesc* up_desc = nullptr;
    if (!moe_native_quantised_load_gate_up_descriptors<RUNTIME_TABLE>(
            gate_descs, up_descs, runtime, expert_ids[route], num_experts,
            gate_desc, up_desc))
    {
        gate_partials[partial_index] = 0.0f;
        up_partials[partial_index] = 0.0f;
        return;
    }

    const int gate_partitions =
        moe_native_quantised_descriptor_k_partitions(
            gate_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    const int up_partitions =
        moe_native_quantised_descriptor_k_partitions(
            up_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (gate_partitions <= 0 || up_partitions <= 0)
    {
        // A published descriptor without generated arithmetic evidence is fatal.
        __builtin_trap();
        return;
    }

    const int blocks_per_row = K / 32;
    if constexpr(PairedK32)
    {
        const int gate_count=(blocks_per_row+gate_partitions-1)/gate_partitions;
        const int up_count=(blocks_per_row+up_partitions-1)/up_partitions;
        const int gate_begin=k_part<gate_partitions?min(blocks_per_row,k_part*gate_count):blocks_per_row;
        const int up_begin=k_part<up_partitions?min(blocks_per_row,k_part*up_count):blocks_per_row;
        float gate_acc=0.f,up_acc=0.f;
        moePairedK32Range(*gate_desc,*up_desc,n,A_int8,scales_A_blockwise,N,K,
            gate_begin,min(blocks_per_row,gate_begin+gate_count),
            up_begin,min(blocks_per_row,up_begin+up_count),gate_acc,up_acc);
        gate_partials[partial_index]=gate_acc;up_partials[partial_index]=up_acc;
        return;
    }
    if (k_part < gate_partitions)
    {
        const int blocks_per_part =
            (blocks_per_row + gate_partitions - 1) / gate_partitions;
        const int b_start = k_part * blocks_per_part;
        const int b_end = min(blocks_per_row, b_start + blocks_per_part);
        gate_partials[partial_index] = b_start < b_end
            ? moe_native_quantised_dot_desc_range_dynamic(
                  gate_desc, n, A_int8, scales_A_blockwise,
                  N, K, b_start, b_end)
            : 0.0f;
    }
    else
    {
        gate_partials[partial_index] = 0.0f;
    }

    if (k_part < up_partitions)
    {
        const int blocks_per_part =
            (blocks_per_row + up_partitions - 1) / up_partitions;
        const int b_start = k_part * blocks_per_part;
        const int b_end = min(blocks_per_row, b_start + blocks_per_part);
        up_partials[partial_index] = b_start < b_end
            ? moe_native_quantised_dot_desc_range_dynamic(
                  up_desc, n, A_int8, scales_A_blockwise,
                  N, K, b_start, b_end)
            : 0.0f;
    }
    else
    {
        up_partials[partial_index] = 0.0f;
    }
}

/** @brief Closed fold load scheduling: staged arithmetic remains the original one-part loop. */
enum class MoEGateUpFoldLoads {Original,TwoPartialLookahead};
/**
 * @brief Load two independent partials before executing their original ordered additions.
 * @param partials Original immutable source-policy partial bank after publication.
 * @param partitions Authenticated positive original descriptor-local partition count.
 * @param num_active Exact original route count used by every producer.
 * @param route Original route index, unchanged by load scheduling.
 * @param columns Original output width/producer bank stride.
 * @param column Original active column index.
 * @return Bit-identical direct partial for P1, otherwise the original ascending-partition RN sum.
 * @details At most two partial values are pending. The odd final partition is
 * loaded only when in bounds; no padding or another source policy is consumed.
 * Each addition still depends on the immediately preceding accumulator. These
 * independent memory loads do not authorize reassociation or a wider reduction.
 */
__device__ __forceinline__ float moeGateUpTwoPartialLookaheadSum(
    const float* __restrict__ partials,int partitions,int num_active,int route,int columns,int column)
{
    const size_t first=static_cast<size_t>(route)*static_cast<size_t>(columns)+static_cast<size_t>(column);
    if(partitions==1)return partials[first]; // Preserve the original signed-zero direct publication.
    float sum=0.0f;
    int part=0;
    #pragma unroll 1
    for(;part+1<partitions;part+=2)
    {
        const size_t first_index=(static_cast<size_t>(part)*static_cast<size_t>(num_active)+
            static_cast<size_t>(route))*static_cast<size_t>(columns)+static_cast<size_t>(column);
        const size_t second_index=(static_cast<size_t>(part+1)*static_cast<size_t>(num_active)+
            static_cast<size_t>(route))*static_cast<size_t>(columns)+static_cast<size_t>(column);
        const float first_value=partials[first_index];
        const float second_value=partials[second_index];
        sum=rocm_native_quantised_accumulate_rn(sum,first_value);
        sum=rocm_native_quantised_accumulate_rn(sum,second_value);
    }
    if(part<partitions)
    {
        const size_t tail_index=(static_cast<size_t>(part)*static_cast<size_t>(num_active)+
            static_cast<size_t>(route))*static_cast<size_t>(columns)+static_cast<size_t>(column);
        sum=rocm_native_quantised_accumulate_rn(sum,partials[tail_index]);
    }
    return sum;
}

/**
 * @brief Reduce gate/up partials and publish the exact Q8 SwiGLU row in-place.
 *
 * Production grouped M=1 decode consumes gate/up only through the following
 * down projection, for both explicit routing and runtime-table routing.
 * Materializing two FP32 rows and launching a second kernel to read them back
 * is therefore avoidable.  This reducer retains the exact descriptor-local
 * ascending-partition additions used above, then performs the same 32-value
 * SwiGLU maximum reduction and Q8 rounding as
 * `moe_grouped_swiglu_quantize_blockwise_kernel`.
 *
 * One 64-lane workgroup owns two independent Q8 blocks.  XOR masks never set
 * wavefront bit 5, so the lower and upper 32 lanes cannot exchange values on a
 * wave64 device.  Invalid routes intentionally leave their Q8 scratch
 * untouched: the down kernel authenticates the same negative expert id before
 * reading that scratch and publishes a zero canonical route row.
 */
template <bool RUNTIME_TABLE,MoEGateUpFoldLoads Loads=MoEGateUpFoldLoads::Original>
__device__ __forceinline__ void moeGateUpOriginalFoldSwiGLUQ8Task(
    const float* __restrict__ gate_partials,
    const float* __restrict__ up_partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    int8_t* __restrict__ swiglu_int8,
    float* __restrict__ swiglu_scales,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    unsigned logical_tile,unsigned logical_route,unsigned logical_lane)
{
    static_assert(Loads==MoEGateUpFoldLoads::Original||Loads==MoEGateUpFoldLoads::TwoPartialLookahead);
    constexpr int TILE_N = 64;
    constexpr int Q8_BLOCK = 32;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    if (route >= num_active)
        return;

    const DeviceNativeQuantisedMatrixDesc* gate_desc = nullptr;
    const DeviceNativeQuantisedMatrixDesc* up_desc = nullptr;
    if (!moe_native_quantised_load_gate_up_descriptors<RUNTIME_TABLE>(
            gate_descs, up_descs, runtime, expert_ids[route], num_experts,
            gate_desc, up_desc))
    {
        return;
    }

    const int gate_partitions =
        moe_native_quantised_descriptor_k_partitions(
            gate_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    const int up_partitions =
        moe_native_quantised_descriptor_k_partitions(
            up_desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (gate_partitions <= 0 || up_partitions <= 0)
    {
        __builtin_trap();
        return;
    }

    const bool active_lane = n < N;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (active_lane)
    {
        /*
         * Preserve the one-partition signed-zero rule and the exact
         * increasing-partition FP32 addition order of the standalone reducer.
         */
        if constexpr(Loads==MoEGateUpFoldLoads::TwoPartialLookahead)
        {
            gate_sum=moeGateUpTwoPartialLookaheadSum(gate_partials,gate_partitions,num_active,route,N,n);
            up_sum=moeGateUpTwoPartialLookaheadSum(up_partials,up_partitions,num_active,route,N,n);
        }
        else
        {
            const size_t first_partial_index =
                static_cast<size_t>(route) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            gate_sum = gate_partials[first_partial_index];
            if (gate_partitions > 1)
            {
                gate_sum = 0.0f;
                #pragma unroll 1
                for (int part = 0; part < gate_partitions; ++part)
                {
                    const size_t partial_index =
                        (static_cast<size_t>(part) *
                             static_cast<size_t>(num_active) +
                         static_cast<size_t>(route)) * static_cast<size_t>(N) +
                        static_cast<size_t>(n);
                    gate_sum = rocm_native_quantised_accumulate_rn(
                        gate_sum, gate_partials[partial_index]);
                }
            }

            up_sum = up_partials[first_partial_index];
            if (up_partitions > 1)
            {
                up_sum = 0.0f;
                #pragma unroll 1
                for (int part = 0; part < up_partitions; ++part)
                {
                    const size_t partial_index =
                        (static_cast<size_t>(part) *
                             static_cast<size_t>(num_active) +
                         static_cast<size_t>(route)) * static_cast<size_t>(N) +
                        static_cast<size_t>(n);
                    up_sum = rocm_native_quantised_accumulate_rn(
                        up_sum, up_partials[partial_index]);
                }
            }
        }
    }

    const float swiglu = active_lane
                             ? llaminar2::rocm::decode_equivalent::swigluValue(
                                   gate_sum, up_sum)
                             : 0.0f;
    float block_abs_max = fabsf(swiglu);
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        block_abs_max = fmaxf(
            block_abs_max,
            __shfl_xor(block_abs_max, mask));

    if (!active_lane)
        return;

    const float scale =
        llaminar2::rocm::decode_equivalent::q8Scale(block_abs_max);
    const int q8_block = n / Q8_BLOCK;
    const int lane = logical_lane & (Q8_BLOCK - 1);
    if (lane == 0)
    {
        swiglu_scales[
            static_cast<size_t>(route) * static_cast<size_t>(N / Q8_BLOCK) +
            static_cast<size_t>(q8_block)] = scale;
    }
    swiglu_int8[
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n)] = static_cast<int8_t>(
            llaminar2::rocm::decode_equivalent::quantizeQ8FromScale(
                swiglu, scale));
}

/**
 * @brief Keep the original64-thread split-K grid and publish each tile from its last completed producer.
 * @details Integer acquire/release tickets order the existing global partial
 * stores. Every nonfinal CTA retires immediately; no spinning, cooperative
 * launch, global residency bound, floating atomic or altered fold is involved.
 */
template<bool PairedK32>
__device__ __forceinline__ void moeGateUpLastProducerBody(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ gate_descs,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ up_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ gate_partials,
    float* __restrict__ up_partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int8_t* swiglu_int8,float* swiglu_scales,llaminar2::NativeTileCompletionView completion)
{
    using llaminar2::NativeTileCompletionDecision;
    const unsigned tiles=(static_cast<unsigned>(N)+63u)/64u;
    if(!completion.counters||completion.tiles!=tiles||completion.routes!=static_cast<unsigned>(num_active)||
        completion.contributors!=static_cast<unsigned>(maximum_k_partitions)||blockDim.x!=64u||
        blockIdx.x>=tiles||blockIdx.y>=completion.routes||blockIdx.z>=completion.contributors)__builtin_trap();
    moeGateUpOriginalPartialTask<false,PairedK32>(
        A_int8, scales_A_blockwise, gate_descs, up_descs, runtime, expert_ids, gate_partials, up_partials, num_active, N, K, num_experts, policy_partitions_0_7, policy_partitions_8_15, policy_partitions_16_23, policy_partitions_24_31, maximum_k_partitions,
        blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x);
    // A full CTA joins all original partial writers before its leader releases
    // them into the modification-order chain. Tail lanes never return here.
    __syncthreads();
    __shared__ unsigned decision;
    auto* counter=completion.counters+size_t(blockIdx.y)*completion.tiles+blockIdx.x;
    if(threadIdx.x==0)
    {
        const auto result=llaminar2::native_tile_completion::arriveForPublication(
            counter,completion.contributors);
        if(result==NativeTileCompletionDecision::Invalid)__builtin_trap();
        decision=static_cast<unsigned>(result);
    }
    // This distributes the final leader's acquired global payload to every
    // original consumer lane before it reads the unchanged ordered bank.
    __syncthreads();
    if(decision!=static_cast<unsigned>(NativeTileCompletionDecision::Publish))return;
    moeGateUpOriginalFoldSwiGLUQ8Task<false,MoEGateUpFoldLoads::TwoPartialLookahead>(
        gate_partials,up_partials,gate_descs,up_descs,nullptr,expert_ids,
        swiglu_int8,swiglu_scales,num_active,N,num_experts,
        policy_partitions_0_7,policy_partitions_8_15,policy_partitions_16_23,policy_partitions_24_31,
        blockIdx.x,blockIdx.y,threadIdx.x);
    // Invalid routes still discharge and reset their protocol without reading
    // stale Q8 bytes. All valid publication writes finish before reset/reuse.
    __syncthreads();
    if(threadIdx.x==0)llaminar2::native_tile_completion::reset(counter);
}

/**
 * @brief Produce descriptor-local down partials for table or runtime routing.
 * @param A_int8 Original route-major Q8 down input.
 * @param scales_A_blockwise Original route-major Q8 scales.
 * @param down_descs Original compact immutable/runtime-published descriptor table.
 * @param runtime Original runtime bank for the original runtime-table specialization.
 * @param expert_ids Original selected slot IDs, including invalid/empty slots.
 * @param partials Original partition-major [part,route,H] publication.
 * @param num_active Original selected slot extent.
 * @param N Original H output width.
 * @param K Original participant-local F reduction width.
 * @param num_experts Original descriptor table extent.
 * @param policy_partitions_0_7 Original source-policy partition entries0..7.
 * @param policy_partitions_8_15 Original source-policy partition entries8..15.
 * @param policy_partitions_16_23 Original source-policy partition entries16..23.
 * @param policy_partitions_24_31 Original source-policy partition entries24..31.
 * @param maximum_k_partitions Original physical grid-Z extent, including zero partial obligations.
 * @param logical_tile Original64-column grid-X coordinate.
 * @param logical_route Original selected-slot grid-Y coordinate.
 * @param logical_part Original split-K grid-Z coordinate.
 * @param logical_lane Original64-thread column lane.
 */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ void moeDownOriginalPartialTask(
    const int8_t* __restrict__ A_int8,
    const float* __restrict__ scales_A_blockwise,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    float* __restrict__ partials,
    int num_active,
    int N,
    int K,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int maximum_k_partitions,
    int logical_tile,int logical_route,int logical_part,int logical_lane)
{
    constexpr int TILE_N = 64;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    const int k_part = logical_part;
    if (route >= num_active || n >= N ||
        k_part >= maximum_k_partitions)
    {
        return;
    }

    const size_t partial_index =
        (static_cast<size_t>(k_part) * static_cast<size_t>(num_active) +
         static_cast<size_t>(route)) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const DeviceNativeQuantisedMatrixDesc* desc = nullptr;
    if (!moe_native_quantised_load_down_descriptor<RUNTIME_TABLE>(
            down_descs, runtime, expert_ids[route], num_experts, desc))
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int route_partitions =
        moe_native_quantised_descriptor_k_partitions(
            desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (route_partitions <= 0)
    {
        __builtin_trap();
        return;
    }
    if (k_part >= route_partitions)
    {
        partials[partial_index] = 0.0f;
        return;
    }

    const int blocks_per_row = K / 32;
    const int blocks_per_part =
        (blocks_per_row + route_partitions - 1) / route_partitions;
    const int b_start = k_part * blocks_per_part;
    const int b_end = min(blocks_per_row, b_start + blocks_per_part);
    const int8_t* route_a =
        A_int8 + static_cast<size_t>(route) * static_cast<size_t>(K);
    const float* route_scales =
        scales_A_blockwise +
        static_cast<size_t>(route) * static_cast<size_t>(blocks_per_row);
    partials[partial_index] = b_start < b_end
        ? moe_native_quantised_dot_desc_range_dynamic(
              desc, n, route_a, route_scales, N, K, b_start, b_end)
        : 0.0f;
}

/**
 * @brief Evaluate one original partition without changing its dot expression.
 * @param descriptor Original admitted source/execution descriptor.
 * @param n Participant-local output column.
 * @param activation Exact published Q8 row.
 * @param scales Original per-block Q8 scales.
 * @param N Original output width.
 * @param K Original reduction width.
 * @param part Original serial-M1 partition index.
 * @param partitions Original descriptor-local partition count.
 * @return Byte-identical original split-K partial, including empty partitions.
 */
template<std::uint32_t DecoderMask=llaminar2::rocm::ResidentMoEFFNDecoderProgram::all_decoders>
__device__ __forceinline__ float moe_native_quantised_block_partial(
    const DeviceNativeQuantisedMatrixDesc* descriptor, int n,
    const int8_t* activation, const float* scales, int N, int K,
    int part, int partitions)
{
    const int blocks = K / 32;
    const int width = (blocks + partitions - 1) / partitions;
    const int first = part * width;
    const int end = min(blocks, first + width);
    return first < end
        ? moe_native_quantised_dot_desc_range_dynamic<DecoderMask>(
              descriptor, n, activation, scales, N, K, first, end)
        : 0.0f;
}

/**
 * @brief Fold CTA-owned original partitions with the serial signed-zero rule.
 * @param partials Shared partition-major plane written before the CTA barrier.
 * @param column Original column within the 32-column publication tile.
 * @param partitions Exact descriptor-local arithmetic partition count.
 * @return Original increasing-partition explicit-RN result.
 */
__device__ __forceinline__ float moe_native_quantised_block_fold(
    const float* partials, int column, int partitions)
{
    if (partitions == 1)
        return partials[column];
    float value = 0.0f;
    #pragma unroll 1
    for (int part = 0; part < partitions; ++part)
        value = rocm_native_quantised_accumulate_rn(
            value, partials[part * 32 + column]);
    return value;
}

/**
 * @brief Reduce one route's descriptor-local K tree, then apply router weight.
 * @param partials Original partition-major partial plane.
 * @param down_descs Original compact descriptor table.
 * @param runtime Original runtime bank for the runtime-table specialization.
 * @param expert_ids Original selected route IDs.
 * @param route_weights Original selected route weights; invalid descriptors never read them.
 * @param route_output Original public route-major weighted contribution plane.
 * @param num_active Original selected slot extent.
 * @param N Original H width.
 * @param num_experts Original descriptor table extent.
 * @param policy_partitions_0_7 Original source-policy partition entries0..7.
 * @param policy_partitions_8_15 Original source-policy partition entries8..15.
 * @param policy_partitions_16_23 Original source-policy partition entries16..23.
 * @param policy_partitions_24_31 Original source-policy partition entries24..31.
 * @param logical_tile Original64-column output tile.
 * @param logical_route Original selected slot, without compaction or reordering.
 * @param logical_lane Original64-thread column lane.
 */
template <bool RUNTIME_TABLE>
__device__ __forceinline__ void moeDownOriginalWeightedFoldTask(
    const float* __restrict__ partials,
    const DeviceNativeQuantisedMatrixDesc* __restrict__ down_descs,
    const DeviceMoELayerRuntimeView* __restrict__ runtime,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    float* __restrict__ route_output,
    int num_active,
    int N,
    int num_experts,
    uint64_t policy_partitions_0_7,
    uint64_t policy_partitions_8_15,
    uint64_t policy_partitions_16_23,
    uint64_t policy_partitions_24_31,
    int logical_tile,int logical_route,int logical_lane)
{
    constexpr int TILE_N = 64;
    const int n = logical_tile * TILE_N + logical_lane;
    const int route = logical_route;
    if (route >= num_active || n >= N)
        return;

    const size_t output_index =
        static_cast<size_t>(route) * static_cast<size_t>(N) +
        static_cast<size_t>(n);
    const DeviceNativeQuantisedMatrixDesc* desc = nullptr;
    if (!moe_native_quantised_load_down_descriptor<RUNTIME_TABLE>(
            down_descs, runtime, expert_ids[route], num_experts, desc))
    {
        route_output[output_index] = 0.0f;
        return;
    }

    const int route_partitions =
        moe_native_quantised_descriptor_k_partitions(
            desc,
            policy_partitions_0_7,
            policy_partitions_8_15,
            policy_partitions_16_23,
            policy_partitions_24_31);
    if (route_partitions <= 0)
    {
        __builtin_trap();
        return;
    }

    float expert_sum = partials[output_index];
    if (route_partitions > 1)
    {
        expert_sum = 0.0f;
        #pragma unroll 1
        for (int part = 0; part < route_partitions; ++part)
        {
            const size_t partial_index =
                (static_cast<size_t>(part) *
                     static_cast<size_t>(num_active) +
                 static_cast<size_t>(route)) * static_cast<size_t>(N) +
                static_cast<size_t>(n);
            expert_sum = rocm_native_quantised_accumulate_rn(
                expert_sum, partials[partial_index]);
        }
    }
    // Raw fragment rows are weighted by their completed continuation cohort.
    route_output[output_index] = route_weights
        ? rocm_native_quantised_weight_route_rn(route_weights[route], expert_sum) : expert_sum;
}

/**
 * @brief Complete original down partitions, then weighted canonical routes, without waiting for an unscheduled CTA.
 * @details Physical128-thread CTAs own two adjacent ORIGINAL64-column tasks.
 * Each lane passes the exact original virtual tile/lane to unchanged helpers.
 */

/**
 * @brief Original descriptor partial fold indexed independently of a native grid.
 * @param partials Original partition-major [part,route,column] plane.
 * @param route Original selected slot, never a compacted ordinal.
 * @param column Original output column.
 * @param routes Exact original selected slot count.
 * @param columns Original physical column stride.
 * @param partitions Original descriptor-local serial-M1 arithmetic count.
 * @return Existing increasing-partition RN result, including one-part signed zero.
 */
__device__ __forceinline__ float residentMoEOrderedPartial(const float* partials,int route,int column,
    int routes,int columns,int partitions)
{
    const auto first=static_cast<size_t>(route)*columns+column;
    if(partitions==1)return partials[first];
    float sum=0.f;
    #pragma unroll 1
    for(int part=0;part<partitions;++part)
        sum=rocm_native_quantised_accumulate_rn(sum,partials[(static_cast<size_t>(part)*routes+route)*columns+column]);
    return sum;
}

/** @brief Authenticate the exact physical role envelope and original source-policy entry; absence is fatal. */
__device__ __forceinline__ int residentMoEPartitions(const DeviceNativeQuantisedMatrixDesc* desc,
    llaminar2::rocm::ResidentMoEPartitionPolicy policy,std::uint32_t execution_mask)
{
    if(!desc||!llaminar2::rocm::residentMoEFFNDecoderAllows(execution_mask,desc->codebook_id))__builtin_trap();
    const int count=moe_native_quantised_descriptor_k_partitions(desc,policy.p0,policy.p1,policy.p2,policy.p3);
    if(count<=0||count>int(policy.maximum))__builtin_trap();
    return count;
}

