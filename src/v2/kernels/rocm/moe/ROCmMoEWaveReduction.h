/**
 * @file ROCmMoEWaveReduction.h
 * @brief Fixed-wave MoE reduction arithmetic shared by routers, expert gates and native replay.
 *
 * gfx906 retains its original DPP instruction sequence. Other ordered-wave64
 * programs perform the same ascending XOR butterfly with portable shuffles;
 * hardware preference does not choose a different floating-point sum tree.
 * Maximum selection retains its original implementation. This contract does
 * not qualify separately implemented future wave32/MMA programs.
 */
#pragma once
#include <hip/hip_runtime.h>
#include <type_traits>
#include "kernels/common/ROCmCompiledWaveContract.h"

namespace llaminar2::rocm::moe_wave_contract
{
#if defined(__HIP_DEVICE_COMPILE__)
    static_assert(__builtin_amdgcn_is_invocable(__builtin_amdgcn_inverse_ballot_w64),
        "Ordered MoE reduction requires explicit HIP wave64 compilation");
#endif
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
#define LLAMINAR_ROCM_MOE_GFX906_DPP 1
#endif

#if defined(LLAMINAR_ROCM_MOE_GFX906_DPP)
#define LLAMINAR_ROCM_MOE_DPP_F32(name, barrier, dpp_ctrl, vop_instr)       \
    __device__ __forceinline__ float name(float x)                          \
    {                                                                       \
        float r;                                                            \
        asm volatile(                                                       \
            barrier                                                         \
            vop_instr " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r)                                                       \
            : "v"(x)                                                        \
            : "memory");                                                    \
        return r;                                                           \
    }

    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_add_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_add_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_add_xor8_f32, "s_nop 1\n", "row_ror:8", "v_add_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_max_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_max_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")
    LLAMINAR_ROCM_MOE_DPP_F32(rocm_moe_dpp_max_xor8_f32, "s_nop 1\n", "row_ror:8", "v_max_f32_dpp")

#undef LLAMINAR_ROCM_MOE_DPP_F32

    /**
     * @brief Exchange one lane word through the original gfx906 DPP/swizzle sequence.
     * @tparam off Fixed XOR partner distance in {1, 2, 4, 8, 16}.
     * @param x Original lane bits. @return Partner lane bits without arithmetic.
     */
    template <int off>
    __device__ __forceinline__ int rocm_moe_dpp_xfer_xor_i32(int x)
    {
        int d;
        if constexpr (off == 1)
        {
            asm volatile("s_nop 4\n"
                         "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        else if constexpr (off == 2)
        {
            asm volatile("s_nop 4\n"
                         "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        else if constexpr (off == 4)
        {
            // The first move writes d before the final masked move still reads
            // x. Early-clobber forbids destructive input/output register aliasing.
            asm volatile("v_mov_b32 %0, %1\n"
                         "s_nop 1\n"
                         "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                         "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                         : "=&v"(d)
                         : "v"(x)
                         : "memory");
        }
        else if constexpr (off == 8)
        {
            asm volatile("s_nop 4\n"
                         "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        else
        {
            static_assert(off == 16, "unsupported xor offset");
            asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                         "s_waitcnt lgkmcnt(0)\n"
                         : "=v"(d)
                         : "v"(x)
                         : "memory");
        }
        return d;
    }

    /** @brief Transfer an FP32 word across XOR-4 partners. @param x Lane value. @return Partner value with identical bits. */
    __device__ __forceinline__ float rocm_moe_dpp_xfer_xor4_f32(float x)
    {
        return __int_as_float(rocm_moe_dpp_xfer_xor_i32<4>(__float_as_int(x)));
    }

    /** @brief Transfer an FP32 word across XOR-16 partners. @param x Lane value. @return Partner value with identical bits. */
    __device__ __forceinline__ float rocm_moe_dpp_xfer_xor16_f32(float x)
    {
        return __int_as_float(rocm_moe_dpp_xfer_xor_i32<16>(__float_as_int(x)));
    }
#endif

    /**
     * @brief Transfer one exact32-bit word left inside its original16-lane row.
     * @tparam Shift Original constant row distance,1/2/4/8/12.
     * @tparam T Original float or int word; no numeric conversion takes place.
     * @param value Original lane bits. @return Row-local source bits, zero out of row.
     * @pre Every lane of the physical wave participates. Consumers must prove
     * their required source stays in the same row; this is not a full-wave shuffle.
     * @details This is the existing ordered-bank update_dpp primitive. Its empty
     * value boundary prevents folding the bit move into source-swapped arithmetic.
     * No memory fence, arithmetic, readiness promise or workgroup join is added.
     */
    template<unsigned Shift,class T>
    __device__ __forceinline__ T rowShiftNativeBits(T value)
    {
        static_assert(Shift==1||Shift==2||Shift==4||Shift==8||Shift==12);
        static_assert(sizeof(T)==4&&(std::is_same_v<T,float>||std::is_same_v<T,int>));
        if(__builtin_amdgcn_is_invocable(__builtin_amdgcn_update_dpp))
        {
            const int bits=__builtin_bit_cast(int,value);
            int moved=__builtin_amdgcn_update_dpp(0,bits,0x100+Shift,0xf,0xf,true);
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(moved));
#endif
            return __builtin_bit_cast(T,moved);
        }
        __builtin_trap();
    }

    /** @brief Internal transfer selection; numerical quantization has one body. */
    enum class QuantizerXorPolicy { Indexed, TargetNative };

    /**
     * @brief Transfer one exact word inside its original32-lane XOR group.
     * @tparam Mask Original descending partner mask, one of16,8,4,2,1.
     * @tparam T Original float or int32 lane word; no numeric conversion occurs.
     * @param value Current lane's exact bits. @return Original XOR partner bits.
     * @pre Every source lane in the original32-lane group participates. Full64,
     * lower32 and upper32 active masks are valid; partial32 groups are not.
     * @details gfx906 reuses the canonical DPP/swizzle sequence. Other compiled
     * targets keep their SDK-native shuffle and its invocability checks. This
     * transfers bits only; it never combines/reorders floating operands.
     */
    template<int Mask,class T>
    __device__ __forceinline__ T xor32NativeBits(T value)
    {
        static_assert(Mask==16||Mask==8||Mask==4||Mask==2||Mask==1);
        static_assert(sizeof(T)==4&&(std::is_same_v<T,float>||std::is_same_v<T,int>));
#if defined(LLAMINAR_ROCM_MOE_GFX906_DPP)
        const int bits=__builtin_bit_cast(int,value);
        return __builtin_bit_cast(T,rocm_moe_dpp_xfer_xor_i32<Mask>(bits));
#else
        return __shfl_xor(value,Mask,32);
#endif
    }

    /**
     * @brief Select the bit transfer used by the unchanged quantizer stage.
     * @tparam Policy Existing indexed users or explicit NativeLayout native transfers.
     * @tparam T Original float/int lane word. @param value Current original bits.
     * @param mask Original loop mask16,8,4,2,1. @return Exact original partner bits.
     * @details The canonical quantizer unrolls its original mask loop, so every
     * native call resolves one constant arm. An invalid mask is fatal; no
     * alternative numerical or runtime path is selected to handle it.
     */
    template<QuantizerXorPolicy Policy,class T>
    __device__ __forceinline__ T quantizerXor32(T value,int mask)
    {
        static_assert(Policy==QuantizerXorPolicy::Indexed||Policy==QuantizerXorPolicy::TargetNative);
        if constexpr(Policy==QuantizerXorPolicy::Indexed)return __shfl_xor(value,mask,32);
        else
        {
            switch(mask)
            {
            case 16:return xor32NativeBits<16>(value);
            case 8:return xor32NativeBits<8>(value);
            case 4:return xor32NativeBits<4>(value);
            case 2:return xor32NativeBits<2>(value);
            case 1:return xor32NativeBits<1>(value);
            default:__builtin_trap();
            }
        }
    }

    /**
     * @brief Reduce a complete wave using the established adjacent-pair-first FP32 tree.
     * @param x One lane's contribution, including explicit zero for inactive work.
     * @return Sum replicated across the ordered 64-lane wave.
     */
    __device__ __forceinline__ float rocm_moe_wave_reduce_sum_f32(float x)
    {
#if defined(LLAMINAR_ROCM_MOE_GFX906_DPP)
        x = rocm_moe_dpp_add_xor1_f32(x);
        x = rocm_moe_dpp_add_xor2_f32(x);
        x += rocm_moe_dpp_xfer_xor4_f32(x);
        x = rocm_moe_dpp_add_xor8_f32(x);
        x += rocm_moe_dpp_xfer_xor16_f32(x);
        x += __shfl_xor(x, 32);
#else
        // Match the existing gfx906 DPP parenthesization exactly: pair
        // adjacent lanes first, then merge groups of 4, 8, 16, 32 and 64.
        for (int offset = 1; offset < static_cast<int>(rocmOrderedWaveWidth); offset <<= 1)
            x += __shfl_xor(x, offset, static_cast<int>(rocmOrderedWaveWidth));
#endif
        return x;
    }

    /**
     * @brief Preserve the existing native maximum tree and its lane publication semantics.
     * @param x One lane's comparison value in a complete ordered wave.
     * @return Reduced maximum at lane zero; callers retain their original broadcast boundary.
     */
    __device__ __forceinline__ float rocm_moe_wave_reduce_max_f32(float x)
    {
#if defined(LLAMINAR_ROCM_MOE_GFX906_DPP)
        x = rocm_moe_dpp_max_xor1_f32(x);
        x = rocm_moe_dpp_max_xor2_f32(x);
        x = fmaxf(x, rocm_moe_dpp_xfer_xor4_f32(x));
        x = rocm_moe_dpp_max_xor8_f32(x);
        x = fmaxf(x, rocm_moe_dpp_xfer_xor16_f32(x));
        x = fmaxf(x, __shfl_xor(x, 32));
#else
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            x = fmaxf(x, __shfl_down(x, offset));
#endif
        return x;
    }


} // namespace llaminar2::rocm::moe_wave_contract
