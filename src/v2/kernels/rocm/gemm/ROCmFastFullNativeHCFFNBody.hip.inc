/**
 * @file ROCmFastFullNativeHCFFNBody.hip.inc
 * @brief Sharded L1-I-cache-resident all-codebook HC + Router + MoE FFN megakernel body.
 *
 * Included by ROCmFastFullNativeHCFFNShard{0,1a,1b,2}.hip. Each specialization instantiates
 * only its sealed (GateCB, DownCB, SharedCB, AllowExpanded) physical decoders for production
 * AcquiredBankOnly decode (zero observation/printf overhead), delivering:
 *   - Compact L1-I-cache-resident code, zero VGPR spills, minimal SGPR spills via
 *     scalar-uniform control flow and phase-local kernarg reloads (phaseKernargField)
 *   - Bank-conflict-free 48-byte AS3 LDS Q8 blocks (2x ds_read_b128 + 1x ds_read_b64)
 *   - Single-validation runtime bank with cached histogram pointers (zero printf/FLAT bloat)
 *   - Idle-CTA-prefetched HC down + 1-pass register-folded HC up across arbitrary gridDim.x
 *   - 4-way load-ahead Router (AS3 FP16 hidden) + 2-col parallel Shared Gate/Up + Gate
 *   - Compile-time UsePanel Gate+Up inner loop with running payload pointers & zero-branch P=16 fold
 *   - Scalar-dispatched 1-wave-per-route 64-col coalesced Down with unit-width fast path
 */
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "ROCmFastFullNativeHCFFN.h"
#include "ROCmMappedRetainedHCFFN.h"
#include "ROCmFullNativeHCFFNPrefix.h"
#include "ROCmNativeLayoutDownGeometry.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "ROCmQ8ResidualNativeDecode.cuh"
#include "ROCmFullNativeHCFFNPrefixBody.hip.inc"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/rocm/common/ROCmQ4PairedDot.h"
#include "kernels/rocm/common/ROCmWorkgroupJoin.h"
#include "kernels/rocm/common/ROCmWaveAddress.h"
#include "kernels/rocm/common/ROCmKernargField.h"
#include "kernels/rocm/ops/ROCmSwiGLUDecodeEquivalentMath.h"
#include "kernels/rocm/moe/ROCmNativeRuntimeBankView.h"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"
#include "kernels/common/DeviceQ8ActivationNumericalContract.h"
#include "kernels/common/DeviceOrderedRouteFold.h"
#include "kernels/common/NativeEmbeddingIQTables.h"
#include "kernels/common/NativeQuantisedDescriptorValidation.h"

namespace llaminar2::rocm::fast_full_hc_ffn {

using DeviceMoELayerRuntimeView = native_runtime_abi::DeviceMoELayerRuntimeView;
using DeviceMoEPlacementBankView = native_runtime_abi::DeviceMoEPlacementBankView;
using DeviceMoEExpertDescriptorView = native_runtime_abi::DeviceMoEExpertDescriptorView;

#if defined(__HIP_DEVICE_COMPILE__)
template<class T> using LdsValue = T __attribute__((opencl_local));
template<class T> using LdsPtr = T __attribute__((opencl_local))*;
template<class T> using GlobalPtr = T __attribute__((address_space(1)))*;
#else
template<class T> using LdsValue = T;
template<class T> using LdsPtr = T*;
template<class T> using GlobalPtr = T*;
#endif

using RawInt4 = int __attribute__((ext_vector_type(4)));
using RawFloat4 = float __attribute__((ext_vector_type(4)));
using RawUint2 = std::uint32_t __attribute__((ext_vector_type(2)));

/** @brief Cast a __shared__ base address to an explicit 32-bit AS3 pointer. */
template<class T, class U>
__device__ __forceinline__ LdsPtr<T> asLds(U* ptr)
{
    return (LdsPtr<T>)((LdsPtr<unsigned char>)ptr);
}

/** @brief Cast a globally resident address to an explicit 64-bit AS1 pointer. */
template<class T>
__device__ __forceinline__ GlobalPtr<T> asGlobal(T* ptr)
{
    return reinterpret_cast<GlobalPtr<T>>(reinterpret_cast<std::uintptr_t>(ptr));
}

/** @brief Load 128-bit int4 from AS3 LDS via native ds_read_b128. */
__device__ __forceinline__ int4 ldsLoadInt4(LdsPtr<const RawInt4> ptr, unsigned index)
{
    const RawInt4 v = ptr[index];
    return make_int4(v[0], v[1], v[2], v[3]);
}

/** @brief Load 128-bit float4 from AS3 LDS via native ds_read_b128. */
__device__ __forceinline__ float4 ldsLoadFloat4(LdsPtr<const RawFloat4> ptr, unsigned index)
{
    const RawFloat4 v = ptr[index];
    return make_float4(v[0], v[1], v[2], v[3]);
}

/** @brief Store 128-bit float4 to AS3 LDS via native ds_write_b128. */
__device__ __forceinline__ void ldsStoreFloat4(LdsPtr<RawFloat4> ptr, unsigned index, float4 v)
{
    RawFloat4 raw;
    raw[0] = v.x;
    raw[1] = v.y;
    raw[2] = v.z;
    raw[3] = v.w;
    ptr[index] = raw;
}

/** @brief Store 128-bit int4 to AS3 LDS via native ds_write_b128. */
__device__ __forceinline__ void ldsStoreInt4(LdsPtr<RawInt4> ptr, unsigned index, int4 v)
{
    RawInt4 raw;
    raw[0] = v.x;
    raw[1] = v.y;
    raw[2] = v.z;
    raw[3] = v.w;
    ptr[index] = raw;
}

/** @brief Load 64-bit uint2 from AS3 LDS via native ds_read_b64. */
__device__ __forceinline__ uint2 ldsLoadUint2(LdsPtr<const RawUint2> ptr)
{
    const RawUint2 v = ptr[0];
    return make_uint2(v[0], v[1]);
}

/** @brief Store 64-bit uint2 to AS3 LDS via native ds_write_b64. */
__device__ __forceinline__ void ldsStoreUint2(LdsPtr<RawUint2> ptr, std::uint32_t lo, std::uint32_t hi)
{
    RawUint2 raw;
    raw[0] = lo;
    raw[1] = hi;
    ptr[0] = raw;
}

/** @brief Load 32-bit packed __half2 from AS3 LDS via native ds_read_b32. */
__device__ __forceinline__ __half2 ldsLoadHalf2(LdsPtr<const std::uint32_t> ptr, unsigned index)
{
    return __builtin_bit_cast(__half2, ptr[index]);
}

/** @brief Store 32-bit packed __half2 to AS3 LDS via native ds_write_b32. */
__device__ __forceinline__ void ldsStoreHalf2(LdsPtr<std::uint32_t> ptr, unsigned index, __half2 v)
{
    ptr[index] = __builtin_bit_cast(std::uint32_t, v);
}

/**
 * @brief 16-byte-aligned, 0%-bank-conflict 48-byte AS3 LDS Q8 activation block.
 *
 * Stride = 12 dwords: across 8 concurrent 5-block Gate/Up partitions,
 * (part * 5 * 12) % 32 visits {0, 28, 24, 20, 16, 12, 8, 4} — a bijection of
 * all 8 four-bank 128-bit LDS groups — while packing {scale, sum} at offset +32
 * for a single ds_read_b64 and removing two live LDS base pointers across all phases.
 */
struct alignas(16) FastQ8Block48 {
    std::int8_t codes[32];  ///< 32 INT8 activation codes (2x ds_read_b128).
    float scale;            ///< Positive finite FP16-rounded activation scale (+32).
    std::int32_t sum;       ///< Exact integer sum of the 32 quantized codes (+36).
    std::uint32_t pad[2];   ///< Padding to 48 B (12 dwords) for bank-conflict bijection.
};
static_assert(sizeof(FastQ8Block48) == 48u && alignof(FastQ8Block48) == 16u &&
              offsetof(FastQ8Block48, scale) == 32u && offsetof(FastQ8Block48, sum) == 36u);

/** @brief Fast 16-bit fixed-point division for small divisors (d <= 64, n < 65536). */
struct FastDiv16 {
    unsigned d;
    unsigned m;

    __device__ __forceinline__ static FastDiv16 make(unsigned divisor)
    {
        return {divisor, (divisor <= 1u) ? 65536u : ((65536u + divisor - 1u) / divisor)};
    }

    __device__ __forceinline__ unsigned div(unsigned n) const
    {
        return (d <= 1u) ? n : ((n * m) >> 16);
    }

    __device__ __forceinline__ void divmod(unsigned n, unsigned& q, unsigned& r) const
    {
        q = div(n);
        r = n - q * d;
    }
};

/** @brief Exact 32-bit unsigned division/remainder using a precomputed __umulhi reciprocal. */
struct FastDiv32 {
    unsigned d;
    unsigned magic;

    __device__ __forceinline__ static FastDiv32 make(unsigned divisor)
    {
        FastDiv32 fd;
        fd.d = divisor;
        fd.magic = (divisor <= 1u) ? 0xFFFFFFFFu : (0xFFFFFFFFu / divisor);
        return fd;
    }

    __device__ __forceinline__ unsigned div(unsigned n) const
    {
        if (d <= 1u) return n;
        unsigned q = __umulhi(n, magic);
        if (n - q * d >= d) ++q;
        return q;
    }

    __device__ __forceinline__ void divmod(unsigned n, unsigned& q, unsigned& r) const
    {
        if (d <= 1u) { q = n; r = 0u; return; }
        q = __umulhi(n, magic);
        r = n - q * d;
        if (r >= d) { ++q; r -= d; }
    }
};

/** @brief Bank-conflict-free AS3 LDS index for packed scale|min words in 64-row Gate/Up tiles. */
__device__ __forceinline__ constexpr unsigned fastMetaPanelIndex(unsigned row, unsigned block)
{
    return (block >> 4) * 1024u + (row >> 2) * 64u + (((block & 15u) ^ (row >> 3)) << 2) + (row & 3u);
}

/** @brief Recover wave-uniform scalar, float, or 64-bit address via SGPR readfirstlane. */
template<class T>
__device__ __forceinline__ T waveUniform(T value)
{
    if constexpr (std::is_pointer_v<T>)
        return reinterpret_cast<T>(waveUniformAddress(reinterpret_cast<std::uintptr_t>(value)));
    else if constexpr (std::is_same_v<T, float>)
        return __builtin_bit_cast(float, static_cast<unsigned>(__builtin_amdgcn_readfirstlane(__builtin_bit_cast(unsigned, value))));
    else
        return static_cast<T>(__builtin_amdgcn_readfirstlane(static_cast<unsigned>(value)));
}

/** @brief Materialize exact physical wave64 lane without extending prior lifetimes. */
__device__ __forceinline__ unsigned lane64()
{
#if defined(__HIP_DEVICE_COMPILE__)
    unsigned lane;
    asm volatile("v_mbcnt_lo_u32_b32 %0, -1, 0\n\t"
                 "v_mbcnt_hi_u32_b32 %0, -1, %0" : "=&v"(lane));
    return lane;
#else
    return static_cast<unsigned>(threadIdx.x) & 63u;
#endif
}

/** @brief Bit-exact absolute wave64 lane transfer. */
__device__ __forceinline__ float waveShuffleAbs(float value, unsigned source)
{
    const auto bits = __builtin_bit_cast(std::int32_t, value);
#if __has_builtin(__builtin_amdgcn_wave_shuffle)
    const auto moved = __builtin_amdgcn_wave_shuffle(bits, static_cast<int>(source));
#else
    const auto moved = __builtin_amdgcn_ds_bpermute(static_cast<int>(source << 2), bits);
#endif
    return __builtin_bit_cast(float, moved);
}

/** @brief Branch-free 4-bank ascending fold for P=16 + tiny unlikely loop for P!=16 (L1I-compact). */
__device__ __forceinline__ float fastFoldFourColumns(float partial, unsigned parts, unsigned lane)
{
    const unsigned u_parts = waveUniform(parts);
    if (__builtin_expect(u_parts == 16u, 1)) {
        float acc = 0.0f;
        acc = rocm_native_quantised_accumulate_rn(acc, partial);
        acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<4>(partial));
        acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<8>(partial));
        acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<12>(partial));
        {
            unsigned cur = lane;
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(cur) : "v"(acc));
#endif
            const float b1 = waveShuffleAbs(partial, (cur + 16u) & 63u);
            acc = rocm_native_quantised_accumulate_rn(acc, b1);
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<4>(b1));
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<8>(b1));
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<12>(b1));
        }
        {
            unsigned cur = lane;
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(cur) : "v"(acc));
#endif
            const float b2 = waveShuffleAbs(partial, (cur + 32u) & 63u);
            acc = rocm_native_quantised_accumulate_rn(acc, b2);
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<4>(b2));
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<8>(b2));
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<12>(b2));
        }
        {
            unsigned cur = lane;
#if defined(__HIP_DEVICE_COMPILE__)
            asm volatile("" : "+v"(cur) : "v"(acc));
#endif
            const float b3 = waveShuffleAbs(partial, (cur + 48u) & 63u);
            acc = rocm_native_quantised_accumulate_rn(acc, b3);
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<4>(b3));
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<8>(b3));
            acc = rocm_native_quantised_accumulate_rn(acc, moe_wave_contract::rowShiftNativeBits<12>(b3));
        }
        return acc;
    }
    if (u_parts == 1u) return waveShuffleAbs(partial, lane & 3u);
    float acc = 0.0f;
    const unsigned col = lane & 3u;
#pragma unroll 1
    for (unsigned p = 0u; p < u_parts; ++p)
        acc = rocm_native_quantised_accumulate_rn(acc, waveShuffleAbs(partial, (p << 2) + col));
    return acc;
}

/** @brief Fold 1..16 ascending K-partitions from strided lanes for odd-row tails. */
__device__ __forceinline__ float fastFoldStrided(float partial, unsigned parts, unsigned col, unsigned stride)
{
    const unsigned u_parts = waveUniform(parts);
    if (u_parts == 1u) return waveShuffleAbs(partial, col);
    float acc = 0.0f;
#pragma unroll 1
    for (unsigned p = 0u; p < u_parts; ++p)
        acc = rocm_native_quantised_accumulate_rn(acc, waveShuffleAbs(partial, stride * p + col));
    return acc;
}

/** @brief Exact AS1 global load for immutable weight/activation bytes. */
template<class T>
__device__ __forceinline__ T globalWeight(const void* pointer)
{
    const auto* source = reinterpret_cast<const unsigned char __attribute__((address_space(1)))*>(
        reinterpret_cast<std::uintptr_t>(pointer));
    T value;
    __builtin_memcpy(&value, source, sizeof(T));
    return value;
}

/** @brief Single-RMW weighted scalar resident grid barrier across arbitrary gridDim.x <= 65536. */
struct FastWeightedGridBarrier {
    static constexpr unsigned kStride = 65536u;
    static constexpr unsigned kMask = kStride - 1u;
    std::uint32_t* words;

    /** @brief Validate words and gridDim.x ONCE at kernel entry (eliminates 10 s_trap CFG diamonds). */
    __device__ __forceinline__ void validateEntry() const
    {
        if (__builtin_expect(!words || gridDim.x == 0u || gridDim.x > kStride, 0))
            __builtin_trap();
    }

    template<unsigned Phase>
    __device__ __forceinline__ void sync() const
    {
        static_assert(Phase < 5u);
        constexpr unsigned kWord = (Phase < 3u)
            ? (2u * Phase)
            : (FullNativeHCFFNBarrierLayout::suffix_word_offset + 2u * (Phase - 3u));
        std::uint32_t* counter = words + kWord;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        if (__builtin_amdgcn_readfirstlane(threadIdx.x) < 64u) {
            const unsigned members = __builtin_amdgcn_readfirstlane(gridDim.x);
            const unsigned bid0 = __builtin_amdgcn_readfirstlane(blockIdx.x);
            const unsigned inc = (bid0 == 0u) ? (kStride - members + 1u) : 1u;
            unsigned old = inc;
            asm volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                         : "+&s"(old) : "s"(counter) : "memory");
            const unsigned before = old & ~kMask;
            while (true) {
                unsigned cur = 0u;
                asm volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                             : "=&s"(cur) : "s"(counter) : "memory");
                if ((cur & ~kMask) != before) break;
                __builtin_amdgcn_s_sleep(1);
            }
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
            asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        }
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
#else
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (threadIdx.x == 0u) {
            const unsigned members = gridDim.x;
            const unsigned inc = (blockIdx.x == 0u) ? (kStride - members + 1u) : 1u;
            const unsigned old = __hip_atomic_fetch_add(counter, inc, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            const unsigned before = old & ~kMask;
            while ((__hip_atomic_load(counter, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) & ~kMask) == before)
                __builtin_amdgcn_s_sleep(1);
            __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
        }
        __syncthreads();
#endif
    }
};

/** @brief Phase-local kernarg field loaders for arg 0 (NativeLayoutRouterFFNArgs). */
#define LLAMINAR_FAST_ROUTER_FIELD(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs, decltype(NativeLayoutRouterFFNArgs::Field), \
        offsetof(NativeLayoutRouterFFNArgs, Field)>()

#define LLAMINAR_FAST_FFN_FIELD(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs, decltype(ResidentMoEFFNArgs::Field), \
        offsetof(NativeLayoutRouterFFNArgs, ffn) + offsetof(ResidentMoEFFNArgs, Field)>()

#define LLAMINAR_FAST_SUFFIX_POLICY_MAX(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs, unsigned, \
        offsetof(NativeLayoutRouterFFNArgs, ffn) + offsetof(ResidentMoEFFNArgs, Field) + \
        offsetof(ResidentMoEPartitionPolicy, maximum)>()

#define LLAMINAR_FAST_SUFFIX_MASK(Field) \
    phaseKernargField<NativeLayoutRouterFFNArgs, std::uint32_t, \
        offsetof(NativeLayoutRouterFFNArgs, ffn) + offsetof(ResidentMoEFFNArgs, execution) + \
        offsetof(ResidentMoEFFNExecutionMasks, Field)>()

/** @brief Compact phase-local projection metadata held in wave-uniform SGPRs or static LDS. */
struct FastPhaseProjection {
    const std::uint8_t* payload;
    const std::uint16_t* scales;
    const std::uint16_t* mins;
    const std::uint32_t* emins;
    unsigned parts;
    unsigned partition_blocks;
    std::uint8_t codebook;
    bool ready;
};

/** @brief Validated immutable execution-bank & histogram pointers acquired ONCE per invocation. */
struct FastValidatedRuntimeBank {
    DeviceMoELayerRuntimeView* runtime;
    const DeviceMoEExpertDescriptorView* experts;
    const std::uint8_t* local_compute_mask;
    std::uint64_t* hist_selected;
    std::uint64_t* hist_local;
    unsigned expert_count;
    int participant_id;
};

/** @brief Publish a validated projection record via 1x s_load_dwordx8 + 1x s_load_dword on __GFX9__. */
__device__ __forceinline__ FastPhaseProjection makeFastProjectionDesc(
    const DeviceNativeQuantisedMatrixDesc* desc,
    unsigned policy_parts,
    std::uint32_t execution_mask,
    unsigned reduction)
{
    FastPhaseProjection r{};
    r.parts = 1u;
    r.partition_blocks = reduction >> 5;
    const auto* udesc = waveUniform(desc);
    if (!udesc) return r;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
    static_assert(offsetof(DeviceNativeQuantisedMatrixDesc, payload) == 0 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, scales) == 8 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, mins) == 16 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, emins) == 24 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, codebook_id) == 44 &&
                  sizeof(DeviceNativeQuantisedMatrixDesc) >= 48);
    using ScalarWords8 = unsigned __attribute__((ext_vector_type(8)));
    const std::uint64_t addr = reinterpret_cast<std::uint64_t>(udesc);
    ScalarWords8 ptr_words;
    unsigned cb_word = 0u;
    asm volatile("s_load_dwordx8 %0, %2, 0\n\t"
                 "s_load_dword %1, %2, 44\n\t"
                 "s_waitcnt lgkmcnt(0)"
                 : "=&s"(ptr_words), "=&s"(cb_word)
                 : "s"(addr)
                 : "memory");
    const auto* pay = reinterpret_cast<const std::uint8_t*>(
        std::uint64_t(ptr_words[0]) | (std::uint64_t(ptr_words[1]) << 32));
    const auto* sc = reinterpret_cast<const std::uint16_t*>(
        std::uint64_t(ptr_words[2]) | (std::uint64_t(ptr_words[3]) << 32));
    if (__builtin_expect(!pay || !sc, 0)) return r;
    const std::uint8_t cb = static_cast<std::uint8_t>(cb_word & 0xFFu);
    if (__builtin_expect(!residentMoEFFNDecoderAllows(execution_mask, cb), 0))
        __builtin_trap();
    r.parts = policy_parts;
    r.partition_blocks = ((reduction >> 5) + policy_parts - 1u) / policy_parts;
    r.payload = pay;
    r.scales = sc;
    r.mins = reinterpret_cast<const std::uint16_t*>(
        std::uint64_t(ptr_words[4]) | (std::uint64_t(ptr_words[5]) << 32));
    r.emins = reinterpret_cast<const std::uint32_t*>(
        std::uint64_t(ptr_words[6]) | (std::uint64_t(ptr_words[7]) << 32));
    r.codebook = cb;
    r.ready = true;
    return r;
#else
    const auto& d = *udesc;
    if (!d.payload || !d.scales) return r;
    if (!residentMoEFFNDecoderAllows(execution_mask, d.codebook_id))
        __builtin_trap();
    r.parts = policy_parts;
    r.partition_blocks = ((reduction >> 5) + policy_parts - 1u) / policy_parts;
    r.payload = d.payload;
    r.scales = static_cast<const std::uint16_t*>(d.scales);
    r.mins = static_cast<const std::uint16_t*>(d.mins);
    r.emins = static_cast<const std::uint32_t*>(d.emins);
    r.codebook = d.codebook_id;
    r.ready = true;
    return r;
#endif
}

/** @brief Sentinel codebook selecting runtime all-18-codebook dispatch for arbitrary mixed GGUF quants. */
constexpr std::uint8_t kFastDynamicCodebook = 0xFFu;

/** @brief Single-flight scalar load of local_mask + proj_set + weight_format + gate/up descriptors on __GFX9__. */
struct FastGateUpDescPair {
    FastPhaseProjection gate;
    FastPhaseProjection up;
    bool local_native;
};

template<std::uint8_t GateCB, bool AllowExpanded>
__device__ __forceinline__ FastGateUpDescPair makeFastGateUpDescPair(
    const DeviceMoEExpertDescriptorView* experts_table,
    const std::uint8_t* local_mask,
    unsigned expert,
    unsigned policy_parts,
    std::uint32_t execution_mask,
    unsigned reduction)
{
    constexpr bool kDynamic = (GateCB == kFastDynamicCodebook);
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<kDynamic ? 19u : GateCB>;
    constexpr bool kNeedMins = kDynamic || Traits::requires_secondary_metadata || AllowExpanded;
    constexpr bool kNeedEmins = kDynamic || (GateCB == 10u);
    FastGateUpDescPair out{};
    out.gate.parts = 1u;
    out.gate.partition_blocks = reduction >> 5;
    out.up.parts = 1u;
    out.up.partition_blocks = reduction >> 5;
    const unsigned u_exp = waveUniform(expert);
    const auto* row = waveUniform(experts_table) + u_exp;
    const auto* mask_base = waveUniform(local_mask);
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
    static_assert(offsetof(DeviceMoEExpertDescriptorView, gate) == 0 &&
                  offsetof(DeviceMoEExpertDescriptorView, up) == 56 &&
                  offsetof(DeviceMoEExpertDescriptorView, weight_format) == 232 &&
                  offsetof(DeviceMoEExpertDescriptorView, projection_set) == 236 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, payload) == 0 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, scales) == 8 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, mins) == 16 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, emins) == 24 &&
                  offsetof(DeviceNativeQuantisedMatrixDesc, codebook_id) == 44);
    using ScalarWords4 = unsigned __attribute__((ext_vector_type(4)));
    using ScalarWords2 = unsigned __attribute__((ext_vector_type(2)));
    const std::uint64_t row_addr = reinterpret_cast<std::uint64_t>(row);
    const std::uint64_t mask_word_addr = reinterpret_cast<std::uint64_t>(mask_base + (u_exp & ~3u));
    ScalarWords4 g_ps, u_ps;
    ScalarWords2 g_m{}, u_m{}, g_em{}, u_em{}, meta_words;
    unsigned g_cb_word = 0u, u_cb_word = 0u, mask_word = 0u;
    if constexpr (kNeedEmins) {
        asm volatile("s_load_dword %9, %11, 0\n\t"
                     "s_load_dwordx2 %8, %10, 232\n\t"
                     "s_load_dwordx4 %0, %10, 0\n\t"
                     "s_load_dwordx2 %2, %10, 16\n\t"
                     "s_load_dwordx2 %4, %10, 24\n\t"
                     "s_load_dword %6, %10, 44\n\t"
                     "s_load_dwordx4 %1, %10, 56\n\t"
                     "s_load_dwordx2 %3, %10, 72\n\t"
                     "s_load_dwordx2 %5, %10, 80\n\t"
                     "s_load_dword %7, %10, 100\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&s"(g_ps), "=&s"(u_ps), "=&s"(g_m), "=&s"(u_m), "=&s"(g_em), "=&s"(u_em),
                       "=&s"(g_cb_word), "=&s"(u_cb_word), "=&s"(meta_words), "=&s"(mask_word)
                     : "s"(row_addr), "s"(mask_word_addr)
                     : "memory");
    } else if constexpr (kNeedMins) {
        asm volatile("s_load_dword %7, %9, 0\n\t"
                     "s_load_dwordx2 %6, %8, 232\n\t"
                     "s_load_dwordx4 %0, %8, 0\n\t"
                     "s_load_dwordx2 %2, %8, 16\n\t"
                     "s_load_dword %4, %8, 44\n\t"
                     "s_load_dwordx4 %1, %8, 56\n\t"
                     "s_load_dwordx2 %3, %8, 72\n\t"
                     "s_load_dword %5, %8, 100\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&s"(g_ps), "=&s"(u_ps), "=&s"(g_m), "=&s"(u_m),
                       "=&s"(g_cb_word), "=&s"(u_cb_word), "=&s"(meta_words), "=&s"(mask_word)
                     : "s"(row_addr), "s"(mask_word_addr)
                     : "memory");
    } else {
        asm volatile("s_load_dword %5, %7, 0\n\t"
                     "s_load_dwordx2 %4, %6, 232\n\t"
                     "s_load_dwordx4 %0, %6, 0\n\t"
                     "s_load_dword %2, %6, 44\n\t"
                     "s_load_dwordx4 %1, %6, 56\n\t"
                     "s_load_dword %3, %6, 100\n\t"
                     "s_waitcnt lgkmcnt(0)"
                     : "=&s"(g_ps), "=&s"(u_ps),
                       "=&s"(g_cb_word), "=&s"(u_cb_word), "=&s"(meta_words), "=&s"(mask_word)
                     : "s"(row_addr), "s"(mask_word_addr)
                     : "memory");
    }
    const std::uint8_t is_local = static_cast<std::uint8_t>((mask_word >> ((u_exp & 3u) * 8u)) & 0xFFu);
    if (is_local == 0u) return out;
    const unsigned wt_fmt = meta_words[0] & 0xFFFFu;
    const unsigned proj_set = meta_words[1];
    if (__builtin_expect(proj_set != static_cast<unsigned>(DeviceMoEProjectionSet::IntermediateShard), 0))
        __builtin_trap();
    if (__builtin_expect(wt_fmt != static_cast<unsigned>(DeviceMoEWeightFormat::NativeQuantised), 0))
        return out;
    out.local_native = true;
    const auto decode_one = [&](const ScalarWords4& ps, const ScalarWords2& mn, const ScalarWords2& em,
                                unsigned cb_word, FastPhaseProjection& r) {
        const auto* pay = reinterpret_cast<const std::uint8_t*>(
            std::uint64_t(ps[0]) | (std::uint64_t(ps[1]) << 32));
        const auto* sc = reinterpret_cast<const std::uint16_t*>(
            std::uint64_t(ps[2]) | (std::uint64_t(ps[3]) << 32));
        if (__builtin_expect(!pay || !sc, 0)) return;
        const std::uint8_t cb = static_cast<std::uint8_t>(cb_word & 0xFFu);
        if (__builtin_expect(!residentMoEFFNDecoderAllows(execution_mask, cb), 0))
            __builtin_trap();
        r.parts = policy_parts;
        r.partition_blocks = ((reduction >> 5) + policy_parts - 1u) / policy_parts;
        r.payload = pay;
        r.scales = sc;
        if constexpr (kNeedMins)
            r.mins = reinterpret_cast<const std::uint16_t*>(std::uint64_t(mn[0]) | (std::uint64_t(mn[1]) << 32));
        if constexpr (kNeedEmins)
            r.emins = reinterpret_cast<const std::uint32_t*>(std::uint64_t(em[0]) | (std::uint64_t(em[1]) << 32));
        r.codebook = cb;
        r.ready = true;
    };
    decode_one(g_ps, g_m, g_em, g_cb_word, out.gate);
    decode_one(u_ps, u_m, u_em, u_cb_word, out.up);
    return out;
#else
    if (globalWeight<std::uint8_t>(mask_base + u_exp) == 0u) return out;
    const auto* row_g = asGlobal(row);
    if (static_cast<unsigned>(row_g->projection_set) != static_cast<unsigned>(DeviceMoEProjectionSet::IntermediateShard))
        __builtin_trap();
    if (static_cast<unsigned>(row_g->weight_format) != static_cast<unsigned>(DeviceMoEWeightFormat::NativeQuantised))
        return out;
    out.local_native = true;
    out.gate = makeFastProjectionDesc(&row->gate, policy_parts, execution_mask, reduction);
    out.up = makeFastProjectionDesc(&row->up, policy_parts, execution_mask, reduction);
    return out;
#endif
}

/** @brief Store FastPhaseProjection fieldwise into AS3 LDS without generic assignment. */
__device__ __forceinline__ void ldsStoreProj(LdsPtr<FastPhaseProjection> dst, const FastPhaseProjection& src)
{
    dst->payload = src.payload;
    dst->scales = src.scales;
    dst->mins = src.mins;
    dst->emins = src.emins;
    dst->parts = src.parts;
    dst->partition_blocks = src.partition_blocks;
    dst->codebook = src.codebook;
    dst->ready = src.ready;
}

/** @brief Trivially-constructible LDS copy of IQ table pointers (keeps union FastFullNativeHCFFNShared trivial). */
struct FastIQTablesLds {
    const std::uint32_t* iq3s;
    const std::uint32_t* iq3xxs;
    const std::uint64_t* iq2s;
    const std::uint64_t* iq2xs;
    const std::uint64_t* iq2xxs;
    const std::uint64_t* iq1s;
};

/** @brief Static LDS for fast suffix phases (fits strictly inside FullNativeHCFFNPrefixScratch). */
struct FastFFNSuffixScratch {
    router_native::RouterSelectionShared selection;
    float route_weights[router_native::kDeviceMoEMaxTopK];
    int route_experts[router_native::kDeviceMoEMaxTopK];
    float shared_gate_value;
    FastValidatedRuntimeBank bank;
    FastIQTablesLds iq_tables;
};

/** @brief Phase-exclusive static LDS union; strictly <= 3456 B (FullNativeHCFFNShared). */
union FastFullNativeHCFFNShared {
    FullNativeHCFFNPrefixScratch hc;
    FastFFNSuffixScratch ffn;
};
static_assert(sizeof(FastFullNativeHCFFNShared) <= 3456u,
              "FastFullNativeHCFFNShared must not exceed canonical FullNativeHCFFNShared capacity");

/**
 * @brief Wave-uniform scalar K-cache runtime-bank + histogram acquisition (zero printf, zero FLAT).
 *
 * Called by wave 15 (overlapped with hidden quantization on waves 0..9) or wave 0 on inactive rows.
 * Uses scalar uniform reads (L1K) so all 60 CTAs share the immutable runtime metadata without VMEM contention.
 */
__device__ __forceinline__ void fastAcquireValidatedBank(
    LdsPtr<FastValidatedRuntimeBank> dst,
    DeviceMoELayerRuntimeView* runtime,
    unsigned expected_experts,
    unsigned expected_top_k,
    bool live,
    const DeviceMoEIntermediateShardBinding& expected_shard,
    unsigned lane)
{
    auto* u_rt = waveUniform(runtime);
    if (!u_rt) __builtin_trap();
    const auto* rt = asGlobal(u_rt);
    const std::uint32_t active_bank = waveUniform(rt->active_bank);
    const std::uint32_t active_epoch = waveUniform(rt->active_epoch);
    const std::uint32_t rt_experts = waveUniform(rt->expert_count);
    const std::uint32_t rt_topk = waveUniform(rt->top_k);
    const auto shard = rt->intermediate_shard;
    const std::uint32_t llep_transient = waveUniform(rt->current_batch_llep_transient_bank_active);
    const std::int32_t* route_parts = waveUniform(rt->route_participant_ids);
    const std::uint32_t participant_id = waveUniform(rt->participant_id);
    const auto* ticket = waveUniform(rt->overlay_epoch_ticket);
    const auto* overlay_banks = waveUniform(rt->overlay_placement_banks);
    auto* hist_banks = waveUniform(rt->runtime_histogram_banks);
    const std::uint32_t* hist_active = waveUniform(rt->runtime_histogram_active_bank);

    const bool invalid = (active_epoch == 0u) |
                         (rt_experts != expected_experts) |
                         (rt_topk != expected_top_k) |
                         (!shard.enabled()) |
                         (llep_transient != 0u) |
                         (live && !shard.sameIdentity(expected_shard)) |
                         (route_parts == nullptr);
    if (__builtin_expect(invalid, 0)) __builtin_trap();

    std::uint32_t bank_idx = active_bank;
    const DeviceMoEPlacementBankView* banks_ptr = overlay_banks ? overlay_banks : u_rt->banks;
    std::uint64_t expected_bank_epoch = 0u;
    bool bank_invalid = false;
    if (ticket) {
        const auto* gt = asGlobal(ticket);
        const std::uint64_t t_epoch = gt->epoch;
        const std::uint64_t t_selector = gt->selector;
        const std::uint32_t t_bank = static_cast<std::uint32_t>(t_selector & 1u);
        const std::uint64_t t_gen = t_selector >> 1u;
        bank_invalid |= (t_epoch == 0u) | (t_gen == 0u) | (t_bank > 1u) | (banks_ptr == nullptr);
        bank_idx = t_bank;
        expected_bank_epoch = t_epoch;
    } else {
        bank_invalid |= (bank_idx > 1u);
    }
    if (__builtin_expect(bank_invalid, 0)) __builtin_trap();
    const auto* bank_g = asGlobal(banks_ptr + bank_idx);
    const std::uint32_t bank_epoch = waveUniform(bank_g->epoch);
    const std::uint32_t bank_experts = waveUniform(bank_g->expert_count);
    bool tail_invalid = (bank_experts != expected_experts) |
                        (expected_bank_epoch != 0u && std::uint64_t(bank_epoch) != expected_bank_epoch);

    std::uint64_t* h_sel = u_rt->decode_histogram;
    std::uint64_t* h_loc = u_rt->decode_local_histogram;
    if (hist_banks || hist_active) {
        if (__builtin_expect(!hist_banks || !hist_active, 0)) __builtin_trap();
        const std::uint32_t state = waveUniform(*asGlobal(hist_active));
        tail_invalid |= !llaminar2::moe_runtime_abi::validHistogramWriterState(state);
        if (!tail_invalid) {
            if ((state & llaminar2::moe_runtime_abi::kHistogramWriterQuarantineBit) != 0u) {
                h_sel = nullptr;
                h_loc = nullptr;
            } else {
                auto& writer = asGlobal(hist_banks)[state & llaminar2::moe_runtime_abi::kHistogramWriterBankMask];
                h_sel = waveUniform(writer.selected[0]);
                h_loc = waveUniform(writer.local[0]);
            }
        }
    }
    if (__builtin_expect(tail_invalid, 0)) __builtin_trap();
    if (lane == 0u) {
        dst->runtime = u_rt;
        dst->experts = banks_ptr[bank_idx].experts;
        dst->local_compute_mask = banks_ptr[bank_idx].local_compute_mask;
        dst->hist_selected = h_sel;
        dst->hist_local = h_loc;
        dst->expert_count = bank_experts;
        dst->participant_id = static_cast<int>(participant_id);
    }
}

/**
 * @brief Ultra-lean CTA-0 single-slot IntermediateShardLocal route + histogram publisher (zero FLAT).
 */
__device__ __forceinline__ void fastPublishLiveRouteSlot(
    LdsPtr<const FastValidatedRuntimeBank> bank,
    int slot,
    LdsPtr<const int> selected_ids,
    LdsPtr<const float> selected_weights,
    unsigned num_experts,
    float* legacy_indices,
    float* legacy_weights,
    bool write_legacy_outputs,
    bool observe_histogram = true)
{
    auto* runtime = asGlobal(bank->runtime);
    const int expert_id = selected_ids[slot];
    const float weight = selected_weights[slot];
    bool local_compute = false;
    int participant = -1;
    if (expert_id != -1) {
        if (__builtin_expect(expert_id < 0 || static_cast<unsigned>(expert_id) >= num_experts, 0))
            __builtin_trap();
        if (globalWeight<std::uint8_t>(bank->local_compute_mask + expert_id) != 0u) {
            local_compute = true;
            participant = bank->participant_id;
        }
    }
    asGlobal(runtime->topk_expert_ids)[slot] = local_compute ? expert_id : -1;
    asGlobal(runtime->topk_weights)[slot] = local_compute ? weight : 0.0f;
    asGlobal(runtime->route_participant_ids)[slot] = participant;
    if (write_legacy_outputs && legacy_indices && legacy_weights) {
        asGlobal(legacy_indices)[slot] = static_cast<float>(expert_id);
        asGlobal(legacy_weights)[slot] = weight;
    }
    std::uint64_t* hist_sel = observe_histogram ? bank->hist_selected : nullptr;
    if (hist_sel && expert_id >= 0) {
        __hip_atomic_fetch_add(
            asGlobal(reinterpret_cast<unsigned long long*>(hist_sel + expert_id)),
            1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (local_compute) {
            std::uint64_t* hist_loc = bank->hist_local;
            if (hist_loc)
                __hip_atomic_fetch_add(
                    asGlobal(reinterpret_cast<unsigned long long*>(hist_loc + expert_id)),
                    1ULL, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

/** @brief Word-packed block operand holding raw payload + metadata words for codebook CB. */
template<bool HasEmin> struct FastBlockEmin {};
template<> struct FastBlockEmin<true> { std::uint32_t extended; };

template<std::uint8_t CB>
struct FastBlockOperand : FastBlockEmin<CB == 10u> {
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    static constexpr unsigned kWordCount = (static_cast<unsigned>(Traits::payload_bytes) + 3u) / 4u;
    std::uint32_t words[kWordCount];
    std::uint32_t scale_pair;
    __device__ __forceinline__ std::uint16_t primary() const { return static_cast<std::uint16_t>(scale_pair); }
    __device__ __forceinline__ std::uint16_t secondary() const { return static_cast<std::uint16_t>(scale_pair >> 16); }
};

/**
 * @brief Fused single-flight Gate + Up scale/min metadata panel preload into swizzled AS3 LDS.
 *
 * Issues all four global_load_dwordx4 packets (Gate scales, Gate mins, Up scales, Up mins)
 * concurrently before any vmcnt(0) wait, reducing panel preload latency from 4 round-trips to 1.
 */
template<bool HasSecondary>
__device__ __forceinline__ void fastPublishMetaPanelPair(
    const std::uint16_t* g_scales,
    const std::uint16_t* g_mins,
    const std::uint16_t* u_scales,
    const std::uint16_t* u_mins,
    unsigned N,
    unsigned first_row,
    unsigned rows,
    unsigned tid,
    unsigned panel_blocks,
    LdsPtr<std::uint32_t> g_panel,
    LdsPtr<std::uint32_t> u_panel)
{
    constexpr unsigned kPacket = 8u;
    const unsigned packets = (rows + kPacket - 1u) >> 3;
    const unsigned total = panel_blocks * packets;
    if (rows == 0u || tid >= total) return;
    // Exact zero-division 16-bit reciprocal for packets in [1..8] (rows <= 64).
    const unsigned m16 = (packets == 1u) ? 65536u : (packets == 2u) ? 32768u : (packets == 3u) ? 21846u
                       : (packets == 4u) ? 16384u : (packets == 5u) ? 13108u : (packets == 6u) ? 10923u
                       : (packets == 7u) ? 9363u  : 8192u;
    auto g_panel4 = asLds<RawInt4>(g_panel);
    auto u_panel4 = asLds<RawInt4>(u_panel);
#pragma unroll 1
    for (unsigned task = tid; task < total; task += 1024u) {
        const unsigned block = (packets == 1u) ? task : ((task * m16) >> 16);
        const unsigned pkt = task - block * packets;
        const unsigned row = pkt * kPacket;
        const unsigned count = (rows - row < kPacket) ? (rows - row) : kPacket;
        const std::size_t source = std::size_t(block) * N + first_row + row;
        if (__builtin_expect(count == kPacket && (source & (kPacket - 1u)) == 0u, 1)) {
            const int4 gs = globalWeight<int4>(g_scales + source);
            const int4 us = globalWeight<int4>(u_scales + source);
            int4 gm{}, um{};
            if constexpr (HasSecondary) {
                gm = globalWeight<int4>(g_mins + source);
                um = globalWeight<int4>(u_mins + source);
            }
            const unsigned dst0 = fastMetaPanelIndex(row, block) >> 2;
            const unsigned dst1 = fastMetaPanelIndex(row + 4u, block) >> 2;
            const auto pack_pair = [] __device__ (int4 s, int4 m) -> std::pair<int4, int4> {
                if constexpr (HasSecondary) {
                    return {
                        make_int4(
                            int((unsigned(s.x) & 0xFFFFu) | (unsigned(m.x) << 16)),
                            int((unsigned(s.x) >> 16) | (unsigned(m.x) & 0xFFFF0000u)),
                            int((unsigned(s.y) & 0xFFFFu) | (unsigned(m.y) << 16)),
                            int((unsigned(s.y) >> 16) | (unsigned(m.y) & 0xFFFF0000u))),
                        make_int4(
                            int((unsigned(s.z) & 0xFFFFu) | (unsigned(m.z) << 16)),
                            int((unsigned(s.z) >> 16) | (unsigned(m.z) & 0xFFFF0000u)),
                            int((unsigned(s.w) & 0xFFFFu) | (unsigned(m.w) << 16)),
                            int((unsigned(s.w) >> 16) | (unsigned(m.w) & 0xFFFF0000u)))};
                } else {
                    (void)m;
                    return {
                        make_int4(
                            int(unsigned(s.x) & 0xFFFFu), int(unsigned(s.x) >> 16),
                            int(unsigned(s.y) & 0xFFFFu), int(unsigned(s.y) >> 16)),
                        make_int4(
                            int(unsigned(s.z) & 0xFFFFu), int(unsigned(s.z) >> 16),
                            int(unsigned(s.w) & 0xFFFFu), int(unsigned(s.w) >> 16))};
                }
            };
            const auto [ga, gb] = pack_pair(gs, gm);
            const auto [ua, ub] = pack_pair(us, um);
            ldsStoreInt4(g_panel4, dst0, ga);
            ldsStoreInt4(g_panel4, dst1, gb);
            ldsStoreInt4(u_panel4, dst0, ua);
            ldsStoreInt4(u_panel4, dst1, ub);
        } else {
#pragma unroll 1
            for (unsigned o = 0u; o < count; ++o) {
                const std::uint32_t gs = globalWeight<std::uint16_t>(g_scales + source + o);
                const std::uint32_t us = globalWeight<std::uint16_t>(u_scales + source + o);
                std::uint32_t gm = 0u, um = 0u;
                if constexpr (HasSecondary) {
                    gm = globalWeight<std::uint16_t>(g_mins + source + o);
                    um = globalWeight<std::uint16_t>(u_mins + source + o);
                }
                const unsigned idx = fastMetaPanelIndex(row + o, block);
                g_panel[idx] = gs | (gm << 16);
                u_panel[idx] = us | (um << 16);
            }
        }
    }
}

/** @brief Preload one word-packed K32 block from pre-stepped pointers (zero per-block multiply). */
template<std::uint8_t CB, bool PreloadedMeta = false>
__device__ __forceinline__ FastBlockOperand<CB> fastPreloadBlockPtr(
    const std::uint8_t* src,
    const std::uint16_t* sc_ptr,
    const std::uint16_t* mn_ptr,
    const std::uint32_t* em_ptr,
    std::uint32_t panel_scale_pair = 0u)
{
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    constexpr unsigned bytes = static_cast<unsigned>(Traits::payload_bytes);
    FastBlockOperand<CB> op{};
    constexpr unsigned n16 = bytes / 16u;
    constexpr unsigned rem16 = bytes % 16u;
    constexpr unsigned n8 = rem16 / 8u;
    constexpr unsigned rem8 = rem16 % 8u;
    constexpr unsigned n4 = rem8 / 4u;
    constexpr unsigned rem4 = rem8 % 4u;
    unsigned widx = 0u;
    if constexpr (n16 > 0u) {
#pragma unroll
        for (unsigned v = 0u; v < n16; ++v) {
            const int4 data = globalWeight<int4>(src + v * 16u);
            op.words[widx + 0u] = static_cast<std::uint32_t>(data.x);
            op.words[widx + 1u] = static_cast<std::uint32_t>(data.y);
            op.words[widx + 2u] = static_cast<std::uint32_t>(data.z);
            op.words[widx + 3u] = static_cast<std::uint32_t>(data.w);
            widx += 4u;
        }
    }
    if constexpr (n8 > 0u) {
        const uint2 data = globalWeight<uint2>(src + n16 * 16u);
        op.words[widx + 0u] = data.x;
        op.words[widx + 1u] = data.y;
        widx += 2u;
    }
    if constexpr (n4 > 0u) {
        op.words[widx] = globalWeight<std::uint32_t>(src + n16 * 16u + n8 * 8u);
        ++widx;
    }
    if constexpr (rem4 > 0u) {
        std::uint32_t tail = 0u;
#pragma unroll
        for (unsigned b = 0u; b < rem4; ++b)
            tail |= std::uint32_t(globalWeight<std::uint8_t>(src + n16 * 16u + n8 * 8u + n4 * 4u + b)) << (b * 8u);
        op.words[widx] = tail;
    }
    if constexpr (PreloadedMeta) {
        op.scale_pair = panel_scale_pair;
    } else {
        const std::uint32_t primary = globalWeight<std::uint16_t>(sc_ptr);
        std::uint32_t secondary = 0u;
        if constexpr (Traits::requires_secondary_metadata)
            secondary = globalWeight<std::uint16_t>(mn_ptr);
        op.scale_pair = primary | (secondary << 16);
    }
    if constexpr (CB == 10u)
        op.extended = globalWeight<std::uint32_t>(em_ptr);
    return op;
}

/** @brief Fast word-packed decode + 2x ds_read_b128 + 1x ds_read_b64 + sdot4 returning exact block contribution for ALL 18 codebooks. */
template<std::uint8_t CB>
__device__ __forceinline__ float fastBlockContribution(
    rocm_q8_residual::NativeDecoder& decoder,
    const FastBlockOperand<CB>& op,
    LdsPtr<const FastQ8Block48> q8_blk)
{
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    const auto* a128 = reinterpret_cast<LdsPtr<const RawInt4>>(q8_blk->codes);

    std::int32_t dot_lo = 0;
    std::int32_t dot_hi = 0;
    std::int32_t sum_a_lo = 0, sum_a_hi = 0;
    std::uint32_t emin_bits = 0u;
    std::uint16_t iq1m_qh = 0u;
    std::int32_t sg0 = 0, sg1 = 0, sg2 = 0, sg3 = 0;

    if constexpr (Traits::is_iq4_lut) {
        const auto* a32_lds = reinterpret_cast<LdsPtr<const std::int32_t>>(q8_blk->codes);
#pragma unroll 1
        for (unsigned g = 0u; g < 4u; ++g) {
            const std::uint32_t w = (g == 0u) ? op.words[0] : (g == 1u) ? op.words[1]
                                  : (g == 2u) ? op.words[2] : op.words[3];
            std::uint32_t lo = 0u, hi = 0u;
            decoder.iq4nl_decode_word(w, lo, hi);
            dot_lo = rocm_integer_contract::signedDot4(a32_lds[g], static_cast<std::int32_t>(lo), dot_lo);
            dot_lo = rocm_integer_contract::signedDot4(a32_lds[g + 4u], static_cast<std::int32_t>(hi), dot_lo);
        }
    } else if constexpr (CB == 9u) {
        const std::uint32_t hbits = op.words[2];
        const auto* a32_lds = reinterpret_cast<LdsPtr<const std::int32_t>>(q8_blk->codes);
#pragma unroll 1
        for (unsigned half = 0u; half < 2u; ++half) {
            const std::uint32_t raw = (half == 0u) ? op.words[0] : op.words[1];
            std::int32_t acc_half = 0;
#pragma unroll 1
            for (unsigned g = 0u; g < 4u; ++g) {
                const std::uint32_t hb4 = (hbits >> ((half * 4u + g) * 4u)) & 0xFu;
                const std::uint32_t packed = ((raw >> (g * 2u)) & 0x03030303u) | decoder.q3k_highbit_pack(hb4);
                acc_half = rocm_integer_contract::signedDot4(
                    a32_lds[half * 4u + g], static_cast<std::int32_t>(decoder.centered_sub_4(packed)), acc_half);
            }
            if (half == 0u) dot_lo = acc_half; else dot_hi = acc_half;
        }
    } else if constexpr (Traits::is_iq3_grid) {
        const auto* a32_lds = reinterpret_cast<LdsPtr<const std::int32_t>>(q8_blk->codes);
        const std::uint64_t idx_bytes =
            static_cast<std::uint64_t>(op.words[0]) |
            (static_cast<std::uint64_t>(op.words[1]) << 32);
        std::uint32_t signs32 = op.words[2];
        std::uint32_t qh_byte = 0u;
        if constexpr (CB == 11u) {
            qh_byte = op.words[2] & 0xFFu;
            signs32 = (op.words[2] >> 8) | (op.words[3] << 24);
        }
#pragma unroll 1
        for (unsigned g = 0u; g < 8u; ++g) {
            int idx = static_cast<int>((idx_bytes >> (g * 8u)) & 0xFFu);
            if constexpr (CB == 11u) idx |= static_cast<int>((qh_byte >> g) & 1u) << 8;
            const std::uint32_t grid4 = decoder.template iq3_grid_lookup<CB>(idx);
            const std::uint8_t nibble = static_cast<std::uint8_t>((signs32 >> (g * 4u)) & 0x0Fu);
            dot_lo = rocm_integer_contract::signedDot4(
                a32_lds[g], static_cast<std::int32_t>(decoder.iq_apply_signs_4(grid4, nibble)), dot_lo);
        }
    } else if constexpr (Traits::is_iq2_grid) {
        const auto* a32_lds = reinterpret_cast<LdsPtr<const std::int32_t>>(q8_blk->codes);
        std::uint32_t qh_byte = 0u;
        std::uint32_t signs32 = op.words[1];
        if constexpr (CB != 15u) {
            qh_byte = op.words[1] & 0xFFu;
            signs32 = (op.words[1] >> 8) | (op.words[2] << 24);
        }
#pragma unroll 1
        for (unsigned l = 0u; l < 4u; ++l) {
            int idx = static_cast<int>((op.words[0] >> (l * 8u)) & 0xFFu);
            if constexpr (CB == 13u) idx |= static_cast<int>((qh_byte >> (2u * l)) & 0x3u) << 8;
            else if constexpr (CB == 14u) idx |= static_cast<int>((qh_byte >> l) & 0x1u) << 8;
            const std::uint64_t grid8 = decoder.template iq2_grid_lookup<CB>(idx);
            const std::uint8_t signs = static_cast<std::uint8_t>((signs32 >> (l * 8u)) & 0xFFu);
            const std::int32_t w0 = static_cast<std::int32_t>(
                decoder.iq_apply_signs_4(static_cast<std::uint32_t>(grid8), static_cast<std::uint8_t>(signs & 0x0Fu)));
            const std::int32_t w1 = static_cast<std::int32_t>(
                decoder.iq_apply_signs_4(static_cast<std::uint32_t>(grid8 >> 32), static_cast<std::uint8_t>(signs >> 4)));
            if constexpr (Traits::is_dual_scale) {
                if (l < 2u) {
                    dot_lo = rocm_integer_contract::signedDot4(a32_lds[l * 2u], w0, dot_lo);
                    dot_lo = rocm_integer_contract::signedDot4(a32_lds[l * 2u + 1u], w1, dot_lo);
                } else {
                    dot_hi = rocm_integer_contract::signedDot4(a32_lds[l * 2u], w0, dot_hi);
                    dot_hi = rocm_integer_contract::signedDot4(a32_lds[l * 2u + 1u], w1, dot_hi);
                }
            } else {
                dot_lo = rocm_integer_contract::signedDot4(a32_lds[l * 2u], w0, dot_lo);
                dot_lo = rocm_integer_contract::signedDot4(a32_lds[l * 2u + 1u], w1, dot_lo);
            }
        }
    } else if constexpr (CB == 19u || CB == kNativeQuantisedExpandedInt8MinCodebook ||
                         CB == 5u || CB == 0u || CB == 6u || CB == 7u || CB == 8u) {
        const int4 a0 = ldsLoadInt4(a128, 0u);
        const int4 a1 = ldsLoadInt4(a128, 1u);
        const std::int32_t a32[8]{a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};

        if constexpr (CB == 19u || CB == kNativeQuantisedExpandedInt8MinCodebook) {
#pragma unroll
            for (unsigned g = 0u; g < 8u; ++g)
                dot_lo = rocm_integer_contract::signedDot4(a32[g], static_cast<std::int32_t>(op.words[g]), dot_lo);
        } else if constexpr (CB == 5u) {
#pragma unroll
            for (unsigned g = 0u; g < 4u; ++g) {
                const std::uint32_t raw = op.words[g];
                dot_lo = rocm_integer_contract::signedDot4(a32[g], static_cast<std::int32_t>(raw & 0x0F0F0F0Fu), dot_lo);
                dot_lo = rocm_integer_contract::signedDot4(a32[g + 4u], static_cast<std::int32_t>((raw >> 4) & 0x0F0F0F0Fu), dot_lo);
            }
        } else if constexpr (CB == 0u) {
#pragma unroll
            for (unsigned g = 0u; g < 4u; ++g) {
                const std::uint32_t raw = op.words[g];
                const std::uint32_t lo = decoder.__vsub4(raw & 0x0F0F0F0Fu, 0x08080808u);
                const std::uint32_t hi = decoder.__vsub4((raw >> 4) & 0x0F0F0F0Fu, 0x08080808u);
                dot_lo = rocm_integer_contract::signedDot4(a32[g], static_cast<std::int32_t>(lo), dot_lo);
                dot_lo = rocm_integer_contract::signedDot4(a32[g + 4u], static_cast<std::int32_t>(hi), dot_lo);
            }
        } else if constexpr (CB == 6u || CB == 7u) {
            const std::uint32_t qh_bits = op.words[4];
#pragma unroll
            for (unsigned g = 0u; g < 4u; ++g) {
                const std::uint32_t raw = op.words[g];
                const std::uint32_t hb_lo = native_quantised::q5HighBitsToPackedBytes((qh_bits >> (g * 4u)) & 0xFu);
                const std::uint32_t hb_hi = native_quantised::q5HighBitsToPackedBytes((qh_bits >> (g * 4u + 16u)) & 0xFu);
                std::uint32_t lo = (raw & 0x0F0F0F0Fu) | hb_lo;
                std::uint32_t hi = ((raw >> 4) & 0x0F0F0F0Fu) | hb_hi;
                if constexpr (CB == 6u) {
                    lo = decoder.centered_sub_16(lo);
                    hi = decoder.centered_sub_16(hi);
                }
                dot_lo = rocm_integer_contract::signedDot4(a32[g], static_cast<std::int32_t>(lo), dot_lo);
                dot_lo = rocm_integer_contract::signedDot4(a32[g + 4u], static_cast<std::int32_t>(hi), dot_lo);
            }
        } else if constexpr (CB == 8u) {
            const std::uint32_t qh0 = op.words[4], qh1 = op.words[5];
#pragma unroll
            for (unsigned g = 0u; g < 4u; ++g) {
                const std::uint32_t raw = op.words[g];
                std::uint32_t qlo = (qh0 >> (g * 8u)) & 0xFFu;
                qlo = ((qlo | (qlo << 12)) & 0x000F000Fu);
                qlo = ((qlo | (qlo << 6)) & 0x03030303u) << 4;
                std::uint32_t lo = (((raw & 0x0F0F0F0Fu) | qlo) ^ 0x80808080u) - 0x20202020u;
                lo ^= 0x80808080u;
                dot_lo = rocm_integer_contract::signedDot4(a32[g], static_cast<std::int32_t>(lo), dot_lo);

                std::uint32_t qhi = (qh1 >> (g * 8u)) & 0xFFu;
                qhi = ((qhi | (qhi << 12)) & 0x000F000Fu);
                qhi = ((qhi | (qhi << 6)) & 0x03030303u) << 4;
                std::uint32_t hi = ((((raw >> 4) & 0x0F0F0F0Fu) | qhi) ^ 0x80808080u) - 0x20202020u;
                hi ^= 0x80808080u;
                dot_hi = rocm_integer_contract::signedDot4(a32[g + 4u], static_cast<std::int32_t>(hi), dot_hi);
            }
        }
    } else {
        // Decode LUT-based / dual-scale codebooks BEFORE loading a0/a1 from LDS to save 8 VGPRs on gfx906.
        std::int32_t w[8];
        decoder.template decode_groups<CB>(reinterpret_cast<const std::uint8_t*>(op.words), w);
        const int4 a0 = ldsLoadInt4(a128, 0u);
        const int4 a1 = ldsLoadInt4(a128, 1u);
        const std::int32_t a32[8]{a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        if constexpr (Traits::is_dual_scale) {
#pragma unroll
            for (unsigned g = 0u; g < 4u; ++g)
                dot_lo = rocm_integer_contract::signedDot4(a32[g], w[g], dot_lo);
#pragma unroll
            for (unsigned g = 4u; g < 8u; ++g)
                dot_hi = rocm_integer_contract::signedDot4(a32[g], w[g], dot_hi);
        } else {
#pragma unroll
            for (unsigned g = 0u; g < 8u; ++g)
                dot_lo = rocm_integer_contract::signedDot4(a32[g], w[g], dot_lo);
        }

        if constexpr (Traits::is_dual_scale_asym) {
            constexpr std::int32_t ones = 0x01010101;
#pragma unroll
            for (unsigned g = 0u; g < 4u; ++g)
                sum_a_lo = rocm_integer_contract::signedDot4(a32[g], ones, sum_a_lo);
#pragma unroll
            for (unsigned g = 4u; g < 8u; ++g)
                sum_a_hi = rocm_integer_contract::signedDot4(a32[g], ones, sum_a_hi);
            emin_bits = op.extended;
        }

        if constexpr (Traits::is_iq1_m) {
            constexpr std::int32_t ones = 0x01010101;
            iq1m_qh = static_cast<std::uint16_t>(op.words[1] & 0xFFFFu);
            sg0 = rocm_integer_contract::signedDot4(a32[1], ones, rocm_integer_contract::signedDot4(a32[0], ones, 0));
            sg1 = rocm_integer_contract::signedDot4(a32[3], ones, rocm_integer_contract::signedDot4(a32[2], ones, 0));
            sg2 = rocm_integer_contract::signedDot4(a32[5], ones, rocm_integer_contract::signedDot4(a32[4], ones, 0));
            sg3 = rocm_integer_contract::signedDot4(a32[7], ones, rocm_integer_contract::signedDot4(a32[6], ones, 0));
        }
    }

    float act_scale = 0.0f;
    std::int32_t act_sum = 0;
    if constexpr (Traits::is_asymmetric) {
        const uint2 meta = ldsLoadUint2(reinterpret_cast<LdsPtr<const RawUint2>>(&q8_blk->scale));
        act_scale = __builtin_bit_cast(float, meta.x);
        act_sum = static_cast<std::int32_t>(meta.y);
    } else {
        act_scale = q8_blk->scale;
    }

    return decoder.template native_quantised_block_contribution_from_reduced_terms_rn<CB>(
        dot_lo, dot_hi, act_scale,
        op.primary(),
        op.secondary(),
        emin_bits, act_sum, sum_a_lo, sum_a_hi,
        iq1m_qh, sg0, sg1, sg2, sg3);
}

/** @brief Accumulate one decoded block contribution in canonical RN order. */
template<std::uint8_t CB>
__device__ __forceinline__ void fastConsumeBlock(
    rocm_q8_residual::NativeDecoder& decoder,
    const FastBlockOperand<CB>& op,
    LdsPtr<const FastQ8Block48> q8_blk,
    float& acc)
{
    acc = rocm_native_quantised_accumulate_rn(acc, fastBlockContribution<CB>(decoder, op, q8_blk));
}

/** @brief Load and compute one exact block contribution at block index `idx` for codebook CB. */
template<std::uint8_t CB>
__device__ __forceinline__ float fastStreamBlockContributionIdx(
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    std::size_t idx,
    LdsPtr<const FastQ8Block48> q8_blk)
{
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    constexpr std::size_t kPayBytes = static_cast<std::size_t>(Traits::payload_bytes);
    const auto cur = fastPreloadBlockPtr<CB, false>(
        payload + idx * kPayBytes, scales + idx,
        mins ? (mins + idx) : nullptr, emins ? (emins + idx) : nullptr, 0u);
    return fastBlockContribution<CB>(decoder, cur, q8_blk);
}

/** @brief Runtime all-18-codebook single-block contribution dispatcher. */
__device__ __forceinline__ float fastStreamBlockContributionIdxDynamic(
    std::uint8_t cb,
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    std::size_t idx,
    LdsPtr<const FastQ8Block48> q8_blk)
{
    switch (cb) {
    case 0u:  return fastStreamBlockContributionIdx<0u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 4u:  return fastStreamBlockContributionIdx<4u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 5u:  return fastStreamBlockContributionIdx<5u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 6u:  return fastStreamBlockContributionIdx<6u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 7u:  return fastStreamBlockContributionIdx<7u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 8u:  return fastStreamBlockContributionIdx<8u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 9u:  return fastStreamBlockContributionIdx<9u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 10u: return fastStreamBlockContributionIdx<10u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 11u: return fastStreamBlockContributionIdx<11u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 12u: return fastStreamBlockContributionIdx<12u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 13u: return fastStreamBlockContributionIdx<13u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 14u: return fastStreamBlockContributionIdx<14u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 15u: return fastStreamBlockContributionIdx<15u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 16u: return fastStreamBlockContributionIdx<16u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 17u: return fastStreamBlockContributionIdx<17u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case 19u: return fastStreamBlockContributionIdx<19u>(decoder, payload, scales, mins, emins, idx, q8_blk);
    case kNativeQuantisedExpandedInt8MinCodebook:
        return fastStreamBlockContributionIdx<kNativeQuantisedExpandedInt8MinCodebook>(
            decoder, payload, scales, mins, emins, idx, q8_blk);
    case kNativeQuantisedIQ4XSSourceScaleCodebook:
        return fastStreamBlockContributionIdx<kNativeQuantisedIQ4XSSourceScaleCodebook>(
            decoder, payload, scales, mins, emins, idx, q8_blk);
    default:
        __builtin_trap();
        return 0.0f;
    }
}

/** @brief Direct 1-block streaming consumption with running pointers (zero per-block multiply). */
template<std::uint8_t CB, bool PreloadedMeta = false>
__device__ __forceinline__ void fastStreamBlockPtr(
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* src,
    const std::uint16_t* sc_ptr,
    const std::uint16_t* mn_ptr,
    const std::uint32_t* em_ptr,
    LdsPtr<const FastQ8Block48> q8_blk,
    float& acc,
    std::uint32_t panel_scale_pair = 0u)
{
    const auto cur = fastPreloadBlockPtr<CB, PreloadedMeta>(src, sc_ptr, mn_ptr, em_ptr, panel_scale_pair);
    fastConsumeBlock<CB>(decoder, cur, q8_blk, acc);
}

/** @brief Compact single-matrix partition loop (L1I-resident, zero duplication). */
template<std::uint8_t CB>
__device__ __forceinline__ float fastPartition(
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    unsigned n, unsigned N,
    unsigned b_first, unsigned b_last,
    LdsPtr<const FastQ8Block48> q8_lds)
{
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    constexpr std::size_t kPayBytes = static_cast<std::size_t>(Traits::payload_bytes);
    if (b_first >= b_last) return 0.0f;
    unsigned lin = b_first * N + n;
    float acc = 0.0f;
#pragma unroll 1
    for (unsigned b = b_first; b < b_last; ++b) {
        const std::size_t idx = static_cast<std::size_t>(lin);
        fastStreamBlockPtr<CB, false>(
            decoder,
            payload + idx * kPayBytes,
            scales + idx,
            mins ? (mins + idx) : nullptr,
            emins ? (emins + idx) : nullptr,
            q8_lds + b, acc);
        lin += N;
    }
    return acc;
}

/**
 * @brief Compact paired Gate+Up loop with preloaded AS3 LDS scale/min panel (1 loop body, L1I-resident).
 */
template<std::uint8_t CB>
__device__ __forceinline__ void fastPairedGateUpPanelPartition(
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* g_pay, const std::uint32_t* g_em,
    const std::uint8_t* u_pay, const std::uint32_t* u_em,
    unsigned n, unsigned N, unsigned local_row,
    unsigned b_first, unsigned b_last,
    LdsPtr<const FastQ8Block48> q8_lds,
    LdsPtr<const std::uint32_t> g_panel,
    LdsPtr<const std::uint32_t> u_panel,
    float& gg, float& uu)
{
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    constexpr std::size_t kPayBytes = static_cast<std::size_t>(Traits::payload_bytes);
    if (b_first >= b_last) return;
    unsigned lin = b_first * N + n;

    if constexpr (CB == 5u) {
#pragma unroll 1
        for (unsigned b = b_first; b < b_last; ++b) {
            const std::size_t off = static_cast<std::size_t>(lin) * 16u;
            const int4 gw = globalWeight<int4>(g_pay + off);
            const int4 uw = globalWeight<int4>(u_pay + off);
            const auto* q8_blk = q8_lds + b;
            const auto* a128 = reinterpret_cast<LdsPtr<const RawInt4>>(q8_blk->codes);
            const int4 a0 = ldsLoadInt4(a128, 0u);
            const int4 a1 = ldsLoadInt4(a128, 1u);
            const unsigned pidx = fastMetaPanelIndex(local_row, b);
            const std::uint32_t g_sp = g_panel[pidx];
            const std::uint32_t u_sp = u_panel[pidx];
            const uint2 act_meta = ldsLoadUint2(reinterpret_cast<LdsPtr<const RawUint2>>(&q8_blk->scale));
            int gd = 0, ud = 0;
            q4PairedDotStep(a0.x, static_cast<std::uint32_t>(gw.x), static_cast<std::uint32_t>(uw.x), 0u, gd, ud);
            q4PairedDotStep(a0.y, static_cast<std::uint32_t>(gw.y), static_cast<std::uint32_t>(uw.y), 0u, gd, ud);
            q4PairedDotStep(a0.z, static_cast<std::uint32_t>(gw.z), static_cast<std::uint32_t>(uw.z), 0u, gd, ud);
            q4PairedDotStep(a0.w, static_cast<std::uint32_t>(gw.w), static_cast<std::uint32_t>(uw.w), 0u, gd, ud);
            q4PairedDotStep(a1.x, static_cast<std::uint32_t>(gw.x), static_cast<std::uint32_t>(uw.x), 4u, gd, ud);
            q4PairedDotStep(a1.y, static_cast<std::uint32_t>(gw.y), static_cast<std::uint32_t>(uw.y), 4u, gd, ud);
            q4PairedDotStep(a1.z, static_cast<std::uint32_t>(gw.z), static_cast<std::uint32_t>(uw.z), 4u, gd, ud);
            q4PairedDotStep(a1.w, static_cast<std::uint32_t>(gw.w), static_cast<std::uint32_t>(uw.w), 4u, gd, ud);
            const float act_scale = __builtin_bit_cast(float, act_meta.x);
            const std::int32_t act_sum = static_cast<std::int32_t>(act_meta.y);
            const float2 gm = __half22float2(__builtin_bit_cast(__half2, g_sp));
            const float2 um = __half22float2(__builtin_bit_cast(__half2, u_sp));
            gg = rocm_native_quantised_commit_corrected_single_scale_block(gg, gd, gm.x, act_sum, gm.y, act_scale);
            uu = rocm_native_quantised_commit_corrected_single_scale_block(uu, ud, um.x, act_sum, um.y, act_scale);
            lin += N;
        }
    } else {
#pragma unroll 1
        for (unsigned b = b_first; b < b_last; ++b) {
            const std::size_t idx = static_cast<std::size_t>(lin);
            const unsigned pidx = fastMetaPanelIndex(local_row, b);
            const auto* q8_blk = q8_lds + b;
#pragma unroll 1
            for (unsigned role = 0u; role < 2u; ++role) {
                const std::uint8_t* pay = (role == 0u) ? g_pay : u_pay;
                const std::uint32_t* em = (role == 0u) ? g_em : u_em;
                const std::uint32_t sp = (role == 0u) ? g_panel[pidx] : u_panel[pidx];
                float acc = (role == 0u) ? gg : uu;
                fastStreamBlockPtr<CB, true>(
                    decoder, pay + idx * kPayBytes, nullptr, nullptr, em ? (em + idx) : nullptr, q8_blk, acc, sp);
                if (role == 0u) gg = acc; else uu = acc;
            }
            lin += N;
        }
    }
}

/**
 * @brief Compact single-loop down row across ascending K32/P partitions (zero code duplication).
 */
template<std::uint8_t CB>
__device__ __forceinline__ float fastDownRow(
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    unsigned n, unsigned H, unsigned blocks, unsigned parts, unsigned width,
    LdsPtr<const FastQ8Block48> q8_lds)
{
    using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    constexpr std::size_t kPayBytes = static_cast<std::size_t>(Traits::payload_bytes);
    const unsigned u_blocks = waveUniform(blocks);
    const unsigned u_parts = waveUniform(parts);
    const unsigned u_width = waveUniform(width);
    unsigned lin = n;
    float raw = 0.0f;

    float part_acc = 0.0f;
    unsigned in_part = 0u;
    unsigned done_parts = 0u;
#pragma unroll 1
    for (unsigned b = 0u; b < u_blocks; ++b) {
        const std::size_t idx = static_cast<std::size_t>(lin);
        fastStreamBlockPtr<CB, false>(
            decoder, payload + idx * kPayBytes, scales + idx,
            mins ? (mins + idx) : nullptr, emins ? (emins + idx) : nullptr, q8_lds + b, part_acc);
        lin += H;
        ++in_part;
        if (__builtin_expect(in_part == u_width || b + 1u == u_blocks, 1)) {
            const float pval = device_fp32_contract::multiply(part_acc, 1.0f);
            raw = (u_parts == 1u) ? pval : rocm_native_quantised_accumulate_rn(raw, pval);
            part_acc = 0.0f;
            in_part = 0u;
            ++done_parts;
        }
    }
    if (__builtin_expect(done_parts < u_parts, 0)) {
#pragma unroll 1
        while (done_parts < u_parts) {
            const float z = device_fp32_contract::multiply(0.0f, 1.0f);
            raw = (u_parts == 1u) ? z : rocm_native_quantised_accumulate_rn(raw, z);
            ++done_parts;
        }
    }
    return raw;
}

/** @brief Materialize a 1-pointer NativeDecoder from LDS for codebook CB (zero unused IQ SGPRs!). */
template<std::uint8_t CB>
__device__ __forceinline__ rocm_q8_residual::NativeDecoder fastMakeSingleCodebookDecoder(
    LdsPtr<const FastIQTablesLds> iq_lds)
{
    NativeEmbeddingIQTables t{};
    if constexpr (CB == 11u) t.iq3s = waveUniform(iq_lds->iq3s);
    else if constexpr (CB == 12u) t.iq3xxs = waveUniform(iq_lds->iq3xxs);
    else if constexpr (CB == 13u) t.iq2s = waveUniform(iq_lds->iq2s);
    else if constexpr (CB == 14u) t.iq2xs = waveUniform(iq_lds->iq2xs);
    else if constexpr (CB == 15u) t.iq2xxs = waveUniform(iq_lds->iq2xxs);
    else if constexpr (CB == 16u || CB == 17u) t.iq1s = waveUniform(iq_lds->iq1s);
    return rocm_q8_residual::NativeDecoder(t);
}

/** @brief Runtime all-18-codebook partition dispatcher for arbitrary mixed GGUF quantization tuples. */
__device__ __forceinline__ float fastPartitionDynamic(
    std::uint8_t cb,
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    unsigned n, unsigned N,
    unsigned b_first, unsigned b_last,
    LdsPtr<const FastQ8Block48> q8_lds)
{
    switch (cb) {
    case 0u:  return fastPartition<0u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 4u:  return fastPartition<4u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 5u:  return fastPartition<5u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 6u:  return fastPartition<6u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 7u:  return fastPartition<7u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 8u:  return fastPartition<8u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 9u:  return fastPartition<9u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 10u: return fastPartition<10u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 11u: return fastPartition<11u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 12u: return fastPartition<12u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 13u: return fastPartition<13u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 14u: return fastPartition<14u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 15u: return fastPartition<15u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 16u: return fastPartition<16u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 17u: return fastPartition<17u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case 19u: return fastPartition<19u>(decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case kNativeQuantisedExpandedInt8MinCodebook:
        return fastPartition<kNativeQuantisedExpandedInt8MinCodebook>(
            decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    case kNativeQuantisedIQ4XSSourceScaleCodebook:
        return fastPartition<kNativeQuantisedIQ4XSSourceScaleCodebook>(
            decoder, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds);
    default:
        __builtin_trap();
        return 0.0f;
    }
}

/** @brief LDS-backed all-18-codebook partition dispatcher (loads <=1 IQ pointer per arm, 0 cross-phase SGPRs). */
__device__ __forceinline__ float fastPartitionDynamicLds(
    std::uint8_t cb,
    LdsPtr<const FastIQTablesLds> iq_lds,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    unsigned n, unsigned N,
    unsigned b_first, unsigned b_last,
    LdsPtr<const FastQ8Block48> q8_lds)
{
#define LLAMINAR_FAST_PART_LDS_CASE(CODEBOOK)                                                     \
    case CODEBOOK: {                                                                              \
        auto dec = fastMakeSingleCodebookDecoder<CODEBOOK>(iq_lds);                               \
        return fastPartition<CODEBOOK>(dec, payload, scales, mins, emins, n, N, b_first, b_last, q8_lds); \
    }
    switch (cb) {
    LLAMINAR_FAST_PART_LDS_CASE(0u)
    LLAMINAR_FAST_PART_LDS_CASE(4u)
    LLAMINAR_FAST_PART_LDS_CASE(5u)
    LLAMINAR_FAST_PART_LDS_CASE(6u)
    LLAMINAR_FAST_PART_LDS_CASE(7u)
    LLAMINAR_FAST_PART_LDS_CASE(8u)
    LLAMINAR_FAST_PART_LDS_CASE(9u)
    LLAMINAR_FAST_PART_LDS_CASE(10u)
    LLAMINAR_FAST_PART_LDS_CASE(11u)
    LLAMINAR_FAST_PART_LDS_CASE(12u)
    LLAMINAR_FAST_PART_LDS_CASE(13u)
    LLAMINAR_FAST_PART_LDS_CASE(14u)
    LLAMINAR_FAST_PART_LDS_CASE(15u)
    LLAMINAR_FAST_PART_LDS_CASE(16u)
    LLAMINAR_FAST_PART_LDS_CASE(17u)
    LLAMINAR_FAST_PART_LDS_CASE(19u)
    LLAMINAR_FAST_PART_LDS_CASE(kNativeQuantisedExpandedInt8MinCodebook)
    LLAMINAR_FAST_PART_LDS_CASE(kNativeQuantisedIQ4XSSourceScaleCodebook)
    default:
        __builtin_trap();
        return 0.0f;
    }
#undef LLAMINAR_FAST_PART_LDS_CASE
}

/** @brief Runtime all-18-codebook down-row dispatcher for arbitrary mixed GGUF quantization tuples. */
__device__ __forceinline__ float fastDownRowDynamic(
    std::uint8_t cb,
    rocm_q8_residual::NativeDecoder& decoder,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    unsigned n, unsigned H, unsigned blocks, unsigned parts, unsigned width,
    LdsPtr<const FastQ8Block48> q8_lds)
{
    switch (cb) {
    case 0u:  return fastDownRow<0u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 4u:  return fastDownRow<4u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 5u:  return fastDownRow<5u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 6u:  return fastDownRow<6u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 7u:  return fastDownRow<7u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 8u:  return fastDownRow<8u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 9u:  return fastDownRow<9u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 10u: return fastDownRow<10u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 11u: return fastDownRow<11u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 12u: return fastDownRow<12u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 13u: return fastDownRow<13u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 14u: return fastDownRow<14u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 15u: return fastDownRow<15u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 16u: return fastDownRow<16u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 17u: return fastDownRow<17u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case 19u: return fastDownRow<19u>(decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case kNativeQuantisedExpandedInt8MinCodebook:
        return fastDownRow<kNativeQuantisedExpandedInt8MinCodebook>(
            decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    case kNativeQuantisedIQ4XSSourceScaleCodebook:
        return fastDownRow<kNativeQuantisedIQ4XSSourceScaleCodebook>(
            decoder, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds);
    default:
        __builtin_trap();
        return 0.0f;
    }
}

/** @brief LDS-backed all-18-codebook down-row dispatcher (loads <=1 IQ pointer per arm, 0 cross-phase SGPRs). */
__device__ __forceinline__ float fastDownRowDynamicLds(
    std::uint8_t cb,
    LdsPtr<const FastIQTablesLds> iq_lds,
    const std::uint8_t* payload,
    const std::uint16_t* scales,
    const std::uint16_t* mins,
    const std::uint32_t* emins,
    unsigned n, unsigned H, unsigned blocks, unsigned parts, unsigned width,
    LdsPtr<const FastQ8Block48> q8_lds)
{
#define LLAMINAR_FAST_DOWN_LDS_CASE(CODEBOOK)                                                             \
    case CODEBOOK: {                                                                                      \
        auto dec = fastMakeSingleCodebookDecoder<CODEBOOK>(iq_lds);                                       \
        return fastDownRow<CODEBOOK>(dec, payload, scales, mins, emins, n, H, blocks, parts, width, q8_lds); \
    }
    switch (cb) {
    LLAMINAR_FAST_DOWN_LDS_CASE(0u)
    LLAMINAR_FAST_DOWN_LDS_CASE(4u)
    LLAMINAR_FAST_DOWN_LDS_CASE(5u)
    LLAMINAR_FAST_DOWN_LDS_CASE(6u)
    LLAMINAR_FAST_DOWN_LDS_CASE(7u)
    LLAMINAR_FAST_DOWN_LDS_CASE(8u)
    LLAMINAR_FAST_DOWN_LDS_CASE(9u)
    LLAMINAR_FAST_DOWN_LDS_CASE(10u)
    LLAMINAR_FAST_DOWN_LDS_CASE(11u)
    LLAMINAR_FAST_DOWN_LDS_CASE(12u)
    LLAMINAR_FAST_DOWN_LDS_CASE(13u)
    LLAMINAR_FAST_DOWN_LDS_CASE(14u)
    LLAMINAR_FAST_DOWN_LDS_CASE(15u)
    LLAMINAR_FAST_DOWN_LDS_CASE(16u)
    LLAMINAR_FAST_DOWN_LDS_CASE(17u)
    LLAMINAR_FAST_DOWN_LDS_CASE(19u)
    LLAMINAR_FAST_DOWN_LDS_CASE(kNativeQuantisedExpandedInt8MinCodebook)
    LLAMINAR_FAST_DOWN_LDS_CASE(kNativeQuantisedIQ4XSSourceScaleCodebook)
    default:
        __builtin_trap();
        return 0.0f;
    }
#undef LLAMINAR_FAST_DOWN_LDS_CASE
}

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
/** @brief 4-insn-per-component DPP 8-lane magnitude max on gfx906 (zero ds_bpermute, bit-exact). */
__device__ __forceinline__ float fastQuietMax32(float4 values)
{
    const auto qabs = [] __device__ (float x) {
        float out;
        asm("v_max_f32_e64 %0, |%1|, |%1|" : "=v"(out) : "v"(x));
        return out;
    };
    const auto compRoot = [] __device__ (float c) {
        float r;
        asm volatile("s_nop 1\n"
                     "v_max_f32_dpp %0, %1, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                     "v_max_f32_dpp %0, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                     "s_nop 1\n"
                     "v_max_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n"
                     "s_nop 1\n"
                     "v_max_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
                     : "=&v"(r) : "v"(c));
        return r;
    };
    const float x = compRoot(qabs(values.x)), y = compRoot(qabs(values.y));
    const float z = compRoot(qabs(values.z)), w = compRoot(qabs(values.w));
    return fmaxf(fmaxf(x, z), fmaxf(y, w));
}
#endif

/** @brief Fast HC-norm Q8 publisher using DPP quietMagnitudeMax32 on gfx906 (bit-exact with normQ8Four). */
__device__ __forceinline__ void fastNormQ8Four(
    float4 value, signed char* output, float* scale_output, unsigned lane)
{
    float maximum;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    maximum = fastQuietMax32(value);
#else
    float a = fabsf(value.x), b = fabsf(value.y), c = fabsf(value.z), d = fabsf(value.w);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        a = fmaxf(a, __shfl_down(a, offset, 8));
        b = fmaxf(b, __shfl_down(b, offset, 8));
        c = fmaxf(c, __shfl_down(c, offset, 8));
        d = fmaxf(d, __shfl_down(d, offset, 8));
    }
    maximum = __shfl(fmaxf(fmaxf(a, c), fmaxf(b, d)), 0, 8);
#endif
    const float scale = device_q8_activation_contract::publishPositiveFiniteHalfScale(
        device_q8_activation_contract::boundedScale(maximum));
    const float inverse = 1.0f / scale;
    if (lane == 0u) *scale_output = scale;
    const unsigned q0 = unsigned(decode_equivalent::quantizeQ8(value.x, inverse)) & 255u;
    const unsigned q1 = unsigned(decode_equivalent::quantizeQ8(value.y, inverse)) & 255u;
    const unsigned q2 = unsigned(decode_equivalent::quantizeQ8(value.z, inverse)) & 255u;
    const unsigned q3 = unsigned(decode_equivalent::quantizeQ8(value.w, inverse)) & 255u;
    const unsigned packed = q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
    __builtin_memcpy(output + 4u * lane, &packed, sizeof(packed));
}

/** @brief Preserve canonical five-Newton finite-half reciprocal. */
__device__ __forceinline__ float fastPreparedScaleReciprocal(float scale)
{
    namespace fp = device_fp32_contract;
    const auto bits = fp::bitsFromFloat(scale);
    const int exponent = int((bits >> 23) & 255u) - 127;
    const float normalized = fp::floatFromBits((127u << 23) | (bits & 0x007fffffu));
    const auto inverse = fp::bitsFromFloat(fp::reciprocalOneToTwo(normalized));
    return fp::floatFromBits((unsigned(int((inverse >> 23) & 255u) - exponent) << 23) | (inverse & 0x007fffffu));
}

/** @brief Quantize 4 adjacent FP32 values directly into bank-conflict-free AS3 FastQ8Block48. */
__device__ __forceinline__ void fastQuantizeFourLds(
    float4 value, unsigned quad, LdsPtr<FastQ8Block48> q8_lds)
{
    using namespace moe_wave_contract;
    float maximum;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx906__)
    maximum = fastQuietMax32(value);
#else
    float a = fabsf(value.x), b = fabsf(value.y), c = fabsf(value.z), d = fabsf(value.w);
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1) {
        a = fmaxf(a, quantizerXor32<QuantizerXorPolicy::TargetNative>(a, mask));
        b = fmaxf(b, quantizerXor32<QuantizerXorPolicy::TargetNative>(b, mask));
        c = fmaxf(c, quantizerXor32<QuantizerXorPolicy::TargetNative>(c, mask));
        d = fmaxf(d, quantizerXor32<QuantizerXorPolicy::TargetNative>(d, mask));
    }
    maximum = fmaxf(fmaxf(a, c), fmaxf(b, d));
#endif
    const float scale = device_q8_activation_contract::publishPositiveFiniteHalfScale(
        device_q8_activation_contract::boundedScale(maximum));
    const float inverse = fastPreparedScaleReciprocal(scale);
    const int q0 = device_q8_activation_contract::quantizePreparedInverse(value.x, inverse);
    const int q1 = device_q8_activation_contract::quantizePreparedInverse(value.y, inverse);
    const int q2 = device_q8_activation_contract::quantizePreparedInverse(value.z, inverse);
    const int q3 = device_q8_activation_contract::quantizePreparedInverse(value.w, inverse);
    const unsigned packed = unsigned(std::uint8_t(q0)) | (unsigned(std::uint8_t(q1)) << 8) |
                            (unsigned(std::uint8_t(q2)) << 16) | (unsigned(std::uint8_t(q3)) << 24);
    auto* blk = q8_lds + (quad >> 3);
    reinterpret_cast<LdsPtr<unsigned>>(blk->codes)[quad & 7u] = packed;
    int sum = (q0 + q2) + (q1 + q3);
#pragma unroll
    for (int mask = 4; mask > 0; mask >>= 1)
        sum += quantizerXor32<QuantizerXorPolicy::TargetNative>(sum, mask);
    if ((threadIdx.x & 7u) == 0u) {
        ldsStoreUint2(reinterpret_cast<LdsPtr<RawUint2>>(&blk->scale),
                      __builtin_bit_cast(std::uint32_t, scale),
                      static_cast<std::uint32_t>(sum));
    }
}

/** @brief 8+4-way load-ahead ascending FP32 fold from pure AS3 LDS (optimal for Per=20 & P=16). */
__device__ __forceinline__ float fastOrderedAppendLds(
    float sum, LdsPtr<const float> values, unsigned stride, unsigned column, unsigned count)
{
    unsigned i = 0u;
#pragma unroll 1
    for (; i + 8u <= count; i += 8u) {
        float a = values[(i + 0u) * stride + column], b = values[(i + 1u) * stride + column];
        float c = values[(i + 2u) * stride + column], d = values[(i + 3u) * stride + column];
        float e = values[(i + 4u) * stride + column], f = values[(i + 5u) * stride + column];
        float g = values[(i + 6u) * stride + column], h = values[(i + 7u) * stride + column];
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d), "+v"(e), "+v"(f), "+v"(g), "+v"(h));
#endif
        sum = device_fp32_contract::add(sum, a);
        sum = device_fp32_contract::add(sum, b);
        sum = device_fp32_contract::add(sum, c);
        sum = device_fp32_contract::add(sum, d);
        sum = device_fp32_contract::add(sum, e);
        sum = device_fp32_contract::add(sum, f);
        sum = device_fp32_contract::add(sum, g);
        sum = device_fp32_contract::add(sum, h);
    }
    if (i + 4u <= count) {
        float a = values[(i + 0u) * stride + column], b = values[(i + 1u) * stride + column];
        float c = values[(i + 2u) * stride + column], d = values[(i + 3u) * stride + column];
#if defined(__HIP_DEVICE_COMPILE__)
        asm volatile("" : "+v"(a), "+v"(b), "+v"(c), "+v"(d));
#endif
        sum = device_fp32_contract::add(sum, a);
        sum = device_fp32_contract::add(sum, b);
        sum = device_fp32_contract::add(sum, c);
        sum = device_fp32_contract::add(sum, d);
        i += 4u;
    }
    if (__builtin_expect(i < count, 0)) {
#pragma unroll 1
        for (; i < count; ++i)
            sum = device_fp32_contract::add(sum, values[i * stride + column]);
    }
    return sum;
}

/** @brief Consume one preloaded HC-Down Q8_0 packet using AS1 128-bit activation loads (zero flat_load). */
__device__ __forceinline__ float fastConsumeHCDownPacket(
    const full_hc_native::Packet& packet, const signed char* input, float scale)
{
    const int4 x0 = globalWeight<int4>(input);
    const int4 x1 = globalWeight<int4>(input + 16u);
    int dot = 0;
    dot = rocm_integer_contract::signedDot4(x0.x, packet.low.x, dot);
    dot = rocm_integer_contract::signedDot4(x0.y, packet.low.y, dot);
    dot = rocm_integer_contract::signedDot4(x0.z, packet.low.z, dot);
    dot = rocm_integer_contract::signedDot4(x0.w, packet.low.w, dot);
    dot = rocm_integer_contract::signedDot4(x1.x, packet.high.x, dot);
    dot = rocm_integer_contract::signedDot4(x1.y, packet.high.y, dot);
    dot = rocm_integer_contract::signedDot4(x1.z, packet.high.z, dot);
    dot = rocm_integer_contract::signedDot4(x1.w, packet.high.w, dot);
    const float ws = __half2float(__ushort_as_half(static_cast<unsigned short>(packet.scale_bits)));
    return device_native_quantised_contract::singleScaleBlock(dot, ws, scale);
}

/** @brief Preloaded Q8_0 HC-Up weight packet (8 code dwords + 1 FP16 scale). */
struct FastHCUpPacket {
    int4 lo, hi;
    std::uint16_t s_bits;
};

/** @brief Issue global loads for one HC-Up Q8_0 block. */
__device__ __forceinline__ FastHCUpPacket fastHCUpLoadPacket(
    const std::uint8_t* up_pay, const std::uint16_t* up_sc, unsigned up_n,
    unsigned column, unsigned block)
{
    const std::size_t linear = std::size_t(block) * up_n + column;
    const std::uint8_t* pay = up_pay + linear * 32u;
    return {globalWeight<int4>(pay), globalWeight<int4>(pay + 16u), globalWeight<std::uint16_t>(up_sc + linear)};
}

/** @brief Consume one preloaded HC-Up Q8_0 packet against AS3 LDS primary/residual codes. */
__device__ __forceinline__ void fastHCUpConsumePacket(
    const FastHCUpPacket& pkt, unsigned block,
    LdsPtr<const RawInt4> p_codes4, LdsPtr<const RawInt4> r_codes4,
    float ps, float rs, float& p_out, float& q_out)
{
    const int4 pa0 = ldsLoadInt4(p_codes4, block * 2u);
    const int4 ra0 = ldsLoadInt4(r_codes4, block * 2u);
    int dp = 0, dq = 0;
    dp = rocm_integer_contract::signedDot4(pa0.x, pkt.lo.x, dp);
    dq = rocm_integer_contract::signedDot4(ra0.x, pkt.lo.x, dq);
    dp = rocm_integer_contract::signedDot4(pa0.y, pkt.lo.y, dp);
    dq = rocm_integer_contract::signedDot4(ra0.y, pkt.lo.y, dq);
    dp = rocm_integer_contract::signedDot4(pa0.z, pkt.lo.z, dp);
    dq = rocm_integer_contract::signedDot4(ra0.z, pkt.lo.z, dq);
    dp = rocm_integer_contract::signedDot4(pa0.w, pkt.lo.w, dp);
    dq = rocm_integer_contract::signedDot4(ra0.w, pkt.lo.w, dq);
    const int4 pa1 = ldsLoadInt4(p_codes4, block * 2u + 1u);
    const int4 ra1 = ldsLoadInt4(r_codes4, block * 2u + 1u);
    dp = rocm_integer_contract::signedDot4(pa1.x, pkt.hi.x, dp);
    dq = rocm_integer_contract::signedDot4(ra1.x, pkt.hi.x, dq);
    dp = rocm_integer_contract::signedDot4(pa1.y, pkt.hi.y, dp);
    dq = rocm_integer_contract::signedDot4(ra1.y, pkt.hi.y, dq);
    dp = rocm_integer_contract::signedDot4(pa1.z, pkt.hi.z, dp);
    dq = rocm_integer_contract::signedDot4(ra1.z, pkt.hi.z, dq);
    dp = rocm_integer_contract::signedDot4(pa1.w, pkt.hi.w, dp);
    dq = rocm_integer_contract::signedDot4(ra1.w, pkt.hi.w, dq);
    const float ws = __half2float(__ushort_as_half(pkt.s_bits));
    p_out = device_native_quantised_contract::singleScaleBlock(dp, ws, ps);
    q_out = device_native_quantised_contract::singleScaleBlock(dq, ws, rs);
}

/** @brief Pure AS3 LDS branch-input adapter (eliminates flat_load inside BranchRMSNorm). */
struct FastLdsBranchInput {
    LdsPtr<const float> cached;
    __device__ __forceinline__ float retained(unsigned ch) const { return cached[ch]; }
    __device__ __forceinline__ float publish(unsigned ch) const { return cached[ch]; }
};

/** @brief Pure AS3 LDS ds_read_b128 norm+Q8 publisher (zero flat_load, zero aligned16 bloat). */
struct FastLdsRegisterNormQ8Output {
    LdsPtr<const RawFloat4> post_lds4;
    LdsPtr<const RawFloat4> gain_lds4;
    signed char* codes;
    float* scales;

    template<class Input, class Output>
    __device__ __forceinline__ void publish(
        const Input& /*input*/, const float* /*branch_gain*/, Output output,
        int hidden, float epsilon, std::uint64_t base, float /*logical_sum*/,
        const float* completed_total, unsigned lane, bool /*member*/) const
    {
        using namespace qwen4exp::branch_norm_native;
        auto* out_g = asGlobal(output);
        auto* codes_g = asGlobal(codes);
        auto* scales_g = asGlobal(scales);
        const float inverse = normalizationInverse(completed_total[0], hidden, epsilon);
        for (unsigned quad = lane; quad < unsigned(hidden / 4); quad += 1024u) {
            const float4 p4 = ldsLoadFloat4(post_lds4, quad);
            const float4 g4 = ldsLoadFloat4(gain_lds4, quad);
            float4 value;
            value.x = normalizedElement(p4.x, inverse, g4.x, full_hc_native::OriginalNormalizedProducts{});
            value.y = normalizedElement(p4.y, inverse, g4.y, full_hc_native::OriginalNormalizedProducts{});
            value.z = normalizedElement(p4.z, inverse, g4.z, full_hc_native::OriginalNormalizedProducts{});
            value.w = normalizedElement(p4.w, inverse, g4.w, full_hc_native::OriginalNormalizedProducts{});
            RawFloat4 out4;
            out4[0] = value.x;
            out4[1] = value.y;
            out4[2] = value.z;
            out4[3] = value.w;
            reinterpret_cast<GlobalPtr<RawFloat4>>(out_g + base)[quad] = out4;
            fastNormQ8Four(value, reinterpret_cast<signed char*>(codes_g + (quad >> 3) * 32u),
                           reinterpret_cast<float*>(scales_g + (quad >> 3)), quad & 7u);
        }
    }
};

/**
 * @brief Compact, L1I-resident HC prefix (arbitrary gridDim.x, H <= 3072, R <= 320).
 */
template<class Grid>
__device__ __forceinline__ void fastHCPrefixPhase(
    const FullNativeHCFFNPrefixArgs& prefix,
    Grid& grid,
    FullNativeHCFFNPrefixScratch& local,
    unsigned char* dynamic_storage,
    unsigned dynamic_bytes)
{
    using namespace full_hc_native;
    const auto& a = prefix.hc;
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned gdim = gridDim.x;
    const unsigned wave = waveUniform(tid >> 6);
    const unsigned lane = lane64();
    const unsigned H = waveUniform(static_cast<unsigned>(a.norm.hidden));
    const unsigned R = waveUniform(static_cast<unsigned>(a.down_weights.n));
    const unsigned words = waveUniform(dynamic_bytes >> 2);
    if (__builtin_expect(words < 256u + 2u * H, 0)) __builtin_trap();

    FullHCDownOwnership owner{0u, 0u};
    {
        const unsigned norm_owners = (gdim < 4u) ? gdim : 4u;
        if (!(gdim > norm_owners && bid < norm_owners)) {
            const unsigned owners = (gdim > norm_owners) ? (gdim - norm_owners) : gdim;
            const unsigned idx = (gdim > norm_owners) ? (bid - norm_owners) : bid;
            const unsigned f0 = (idx * R) / owners;
            const unsigned f1 = ((idx + 1u) * R) / owners;
            owner = {f0, f1 - f0};
        }
    }
    const unsigned total_parts = waveUniform(static_cast<unsigned>(a.partitions));
    const unsigned down_blocks = waveUniform(static_cast<unsigned>(a.down_weights.k) >> 5);
    const unsigned per_part = (down_blocks + total_parts - 1u) / total_parts;
    const unsigned per_col_words = total_parts * (per_part + 1u);
    const unsigned max_cols_per_chunk = (owner.columns > 0u && per_col_words <= words)
        ? static_cast<unsigned>(fullHCMax(1u, fullHCMin(owner.columns, words / per_col_words)))
        : 0u;
    const unsigned early_items = total_parts * per_part * owner.columns;
    const bool early = waveUniform((gdim > 4u) && (bid >= 4u) && (owner.columns > 0u) &&
                                   (owner.columns <= max_cols_per_chunk) &&
                                   (owner.columns * early_items < 65536u));

    Packet first_pkt{}, second_pkt{};
    float* arena_generic = reinterpret_cast<float*>(dynamic_storage);
    auto arena_lds = asLds<float>(dynamic_storage);
    const FastDiv16 div_early16 = FastDiv16::make(early ? owner.columns : 1u);

    // Issue idle-CTA HC-down prefetches IMMEDIATELY using exact 2-insn FastDiv16!
    if (early) {
        if (tid < early_items) {
            unsigned leaf = 0u, rem = 0u;
            div_early16.divmod(tid, leaf, rem);
            if (leaf < down_blocks)
                first_pkt = loadPacket(a.down_weights, owner.first + rem, leaf);
        }
        if (tid + 1024u < early_items) {
            unsigned leaf = 0u, rem = 0u;
            div_early16.divmod(tid + 1024u, leaf, rem);
            if (leaf < down_blocks)
                second_pkt = loadPacket(a.down_weights, owner.first + rem, leaf);
        }
    } else {
        const int norm_rows = waveUniform(a.norm.extent.liveRows ? *asGlobal(a.norm.extent.liveRows) : 1);
        if (__builtin_expect(norm_rows < 0 || norm_rows > 1, 0)) __builtin_trap();
        auto post_lds = arena_lds + 256u;
        auto gain_lds = post_lds + H;
        auto post_lds4 = asLds<RawFloat4>(post_lds);
        auto gain_lds4 = asLds<RawFloat4>(gain_lds);
        const bool has_res = waveUniform(prefix.write.residual != nullptr);
#pragma unroll 1
        for (unsigned task = bid; task < 4u; task += gdim) {
            signed char* codes = a.input_codes + std::size_t(task) * H;
            float* scales = a.input_scales + std::size_t(task) * (H >> 5);
            if (__builtin_expect(norm_rows != 0, 1)) {
                const auto base = std::size_t(task) * H;
                const float* gain_p = a.norm.gain + base;
                const float gate = has_res ? qwen4exp::branch_norm_native::residualGate(
                    globalWeight<float>(prefix.write.gateLogits + task)) : 0.0f;
                const float* src_p = (has_res ? prefix.write.residual : a.norm.input) + base;
                const float* blk_p = prefix.write.block;
                float* out_p = prefix.write.output + base;
                const unsigned quads = H >> 2;
#pragma unroll 1
                for (unsigned q = tid; q < quads; q += 1024u) {
                    const float4 g4 = globalWeight<float4>(gain_p + 4u * q);
                    float4 p4 = globalWeight<float4>(src_p + 4u * q);
                    if (has_res) {
                        const float4 b4 = globalWeight<float4>(blk_p + 4u * q);
                        p4.x = qwen4exp::branch_norm_native::residualValue(p4.x, gate, b4.x);
                        p4.y = qwen4exp::branch_norm_native::residualValue(p4.y, gate, b4.y);
                        p4.z = qwen4exp::branch_norm_native::residualValue(p4.z, gate, b4.z);
                        p4.w = qwen4exp::branch_norm_native::residualValue(p4.w, gate, b4.w);
                        RawFloat4 ro;
                        ro[0] = p4.x; ro[1] = p4.y; ro[2] = p4.z; ro[3] = p4.w;
                        reinterpret_cast<GlobalPtr<RawFloat4>>(asGlobal(out_p))[q] = ro;
                    }
                    ldsStoreFloat4(post_lds4, q, p4);
                    ldsStoreFloat4(gain_lds4, q, g4);
                }
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                qwen4exp::branch_norm_native::taskFromBranchInput<
                    FastLdsBranchInput, float*, FastLdsRegisterNormQ8Output,
                    qwen4exp::branch_norm_native::SquareSumPublication::WaveOwnsBothRoots>(
                    FastLdsBranchInput{post_lds}, gain_p, a.norm.output, H, a.norm.epsilon, task,
                    arena_generic, tid, tid < 256u,
                    FastLdsRegisterNormQ8Output{post_lds4, gain_lds4, codes, scales});
            } else {
                for (unsigned quad = tid; quad < (H >> 2); quad += 1024u) {
                    const float4 val = globalWeight<float4>(a.norm.output + std::size_t(task) * H + 4u * quad);
                    fastNormQ8Four(val, codes + (quad >> 3) * 32u, scales + (quad >> 3), quad & 7u);
                }
            }
            if (task + gdim < 4u)
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        }
    }
    grid.template sync<0>(); // Barrier 1: normalized + Q8 ready; idle CTAs' HC-down prefetches already landed!

    if (__builtin_expect(!early && a.injection.weights && a.injection.rows.activeRows(), 0)) {
        for (unsigned output = bid; output < 4u; output += gdim) {
            const float* w = a.injection.weights + std::size_t(output) * a.down_weights.k;
            const float value = floating_projection_native::row<8>(
                a.norm.output, w, a.down_weights.k, arena_generic, tid, tid < 256u);
            if (tid == 0u) asGlobal(a.injection.output)[output] = value;
            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        }
    }

    // HC-Down: compact L1I-resident consume using exact 2-insn FastDiv16 + AS1 int4 loads!
    if (owner.columns > 0u) {
        if (__builtin_expect(max_cols_per_chunk == 0u, 0)) __builtin_trap();
#pragma unroll 1
        for (unsigned begin_col = 0u; begin_col < owner.columns; begin_col += max_cols_per_chunk) {
            const unsigned cols = static_cast<unsigned>(fullHCMin(max_cols_per_chunk, owner.columns - begin_col));
            const unsigned items = total_parts * per_part * cols;
            const bool use_early = early && (begin_col == 0u);
            if (__builtin_expect(use_early, 1)) {
                if (tid < items) {
                    const unsigned leaf = div_early16.div(tid);
                    if (leaf < down_blocks)
                        arena_lds[tid] = fastConsumeHCDownPacket(first_pkt, a.input_codes + leaf * 32u, globalWeight<float>(a.input_scales + leaf));
                }
                if (tid + 1024u < items) {
                    const unsigned leaf = div_early16.div(tid + 1024u);
                    if (leaf < down_blocks)
                        arena_lds[tid + 1024u] = fastConsumeHCDownPacket(second_pkt, a.input_codes + leaf * 32u, globalWeight<float>(a.input_scales + leaf));
                }
            }
            if (__builtin_expect(!use_early || items > 2048u, 0)) {
                const FastDiv32 div_cols = FastDiv32::make(cols);
#pragma unroll 1
                for (unsigned item = use_early ? (tid + 2048u) : tid; item < items; item += 1024u) {
                    unsigned leaf = 0u, rem = 0u;
                    div_cols.divmod(item, leaf, rem);
                    if (leaf < down_blocks) {
                        const Packet pkt = loadPacket(a.down_weights, owner.first + begin_col + rem, leaf);
                        arena_lds[item] = fastConsumeHCDownPacket(pkt, a.input_codes + leaf * 32u, globalWeight<float>(a.input_scales + leaf));
                    }
                }
            }
            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
            auto part_lds = arena_lds + items;
            if (tid < total_parts * cols) {
                unsigned part = 0u, col = 0u;
                if (__builtin_expect(use_early, 1))
                    div_early16.divmod(tid, part, col);
                else
                    FastDiv32::make(cols).divmod(tid, part, col);
                const unsigned first_blk = part * per_part;
                const unsigned count = (first_blk < down_blocks)
                    ? static_cast<unsigned>(fullHCMin(per_part, down_blocks - first_blk)) : 0u;
                const float sum = fastOrderedAppendLds(0.0f, arena_lds + part * per_part * cols, cols, col, count);
                part_lds[tid] = __fmul_rn(sum, 1.0f);
            }
            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
            if (tid < cols) {
                const float raw = (total_parts == 1u)
                    ? part_lds[tid]
                    : fastOrderedAppendLds(0.0f, part_lds, cols, tid, total_parts);
                asGlobal(a.raw_down)[owner.first + begin_col + tid] = raw;
            }
            if (begin_col + max_cols_per_chunk < owner.columns)
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        }
    }
    grid.template sync<1>(); // Barrier 2: a.raw_down[0..R-1] ready

    const bool publish = waveUniform(a.publication.rows.activeRows() != 0);
    if (tid < R) {
        const float raw_val = globalWeight<float>(a.raw_down + tid);
        if (__builtin_expect(publish, 1)) {
            auto output = a.publication;
            output.activated = local.activated;
            output.primary = local.primary;
            output.residual = local.residual;
            output.primaryScales = local.primary_scales;
            output.residualScales = local.residual_scales;
            hc_down_publication::publish(raw_val, output, tid,
                hc_down_publication::NativePositiveHalfScalePublication{});
        } else {
            local.primary[tid] = a.up.primary[tid];
            local.residual[tid] = a.up.residual[tid];
        }
    }
    if (__builtin_expect(!publish && tid < (R >> 5), 0)) {
        local.primary_scales[tid] = a.up.primaryScales[tid];
        local.residual_scales[tid] = a.up.residualScales[tid];
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    if (__builtin_expect(publish && bid == 0u, 0)) {
        if (tid < R) {
            asGlobal(a.publication.primary)[tid] = local.primary[tid];
            asGlobal(a.publication.residual)[tid] = local.residual[tid];
        }
        if (tid < (R >> 5)) {
            asGlobal(a.publication.primaryScales)[tid] = local.primary_scales[tid];
            asGlobal(a.publication.residualScales)[tid] = local.residual_scales[tid];
        }
    }

    // HC-Up + Mix: single compact loop (zero 5x unroll bloat, L1I-resident) + 1-flight 4-branch Mix.
    const bool up_active = waveUniform(
        (a.up.rows.countOwner() == a.publication.rows.countOwner())
            ? publish
            : (a.up.rows.activeRows() != 0));
    if (__builtin_expect(up_active, 1)) {
        const unsigned up_groups = waveUniform(static_cast<unsigned>(a.up.weights.blocks_per_row));
        const std::uint8_t* up_pay = waveUniform(a.up.weights.payload);
        const std::uint16_t* up_sc = waveUniform(static_cast<const std::uint16_t*>(a.up.weights.scales));
        const unsigned up_n = waveUniform(static_cast<unsigned>(a.up.weights.n));
        const unsigned e0 = (bid * H) / gdim;
        const unsigned e1 = ((bid + 1u) * H) / gdim;
        const unsigned total_n = e1 - e0;
        if (total_n > 0u) {
            auto p_codes4 = asLds<const RawInt4>(local.primary);
            auto r_codes4 = asLds<const RawInt4>(local.residual);
            auto p_scales = asLds<const float>(local.primary_scales);
            auto r_scales = asLds<const float>(local.residual_scales);
            auto gates_lds = arena_lds;
            const unsigned max_panel_n = static_cast<unsigned>(fullHCMax(1u, fullHCMin(total_n, words >> 2)));
            const auto ordered_add = [] __device__ (float previous, float leaf) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
                float value;
                asm("v_add_f32 %0, %1, %2" : "=v"(value) : "v"(previous), "v"(leaf));
                return value;
#else
                return device_fp32_contract::add(previous, leaf);
#endif
            };
            const unsigned part = (lane >> 3) & 1u;
            const unsigned first_ci = wave * 32u + ((lane >> 4) << 3) + (lane & 7u);
            const unsigned chunk = (up_groups + 1u) >> 1;
            const unsigned segs = (up_groups + chunk - 1u) / chunk;

#pragma unroll 1
            for (unsigned begin = e0; begin < e1; begin += max_panel_n) {
                const unsigned n = static_cast<unsigned>(fullHCMin(max_panel_n, e1 - begin));
                const unsigned columns = 4u * n;
                const FastDiv32 div_n = FastDiv32::make(n);
#pragma unroll 1
                for (unsigned base = 0u; base < columns; base += 512u) {
                    const unsigned ci = base + first_ci;
                    const bool live_ci = ci < columns;
                    unsigned c = 0u, i = 0u;
                    if (live_ci) div_n.divmod(ci, c, i);
                    const unsigned column = c * H + begin + i;
                    float cp0 = 0.f, cp1 = 0.f, cp2 = 0.f, cp3 = 0.f, cp4 = 0.f;
                    float cq0 = 0.f, cq1 = 0.f, cq2 = 0.f, cq3 = 0.f, cq4 = 0.f;
                    if (live_ci) {
                        const unsigned blk0 = part * chunk;
                        const unsigned blk_end = min(up_groups, blk0 + chunk);
#pragma unroll 1
                        for (unsigned j = 0u; blk0 + j < blk_end; ++j) {
                            const unsigned blk = blk0 + j;
                            const FastHCUpPacket pkt = fastHCUpLoadPacket(up_pay, up_sc, up_n, column, blk);
                            float tp = 0.f, tq = 0.f;
                            fastHCUpConsumePacket(pkt, blk, p_codes4, r_codes4, p_scales[blk], r_scales[blk], tp, tq);
                            if (j == 0u) { cp0 = tp; cq0 = tq; }
                            else if (j == 1u) { cp1 = tp; cq1 = tq; }
                            else if (j == 2u) { cp2 = tp; cq2 = tq; }
                            else if (j == 3u) { cp3 = tp; cq3 = tq; }
                            else { cp4 = tp; cq4 = tq; }
                        }
                    }
                    float p = 0.0f, q = 0.0f;
                    const auto fold5 = [&](unsigned seg_base) {
                        if (0u < chunk && seg_base + 0u < up_groups) { p = ordered_add(p, cp0); q = ordered_add(q, cq0); }
                        if (1u < chunk && seg_base + 1u < up_groups) { p = ordered_add(p, cp1); q = ordered_add(q, cq1); }
                        if (2u < chunk && seg_base + 2u < up_groups) { p = ordered_add(p, cp2); q = ordered_add(q, cq2); }
                        if (3u < chunk && seg_base + 3u < up_groups) { p = ordered_add(p, cp3); q = ordered_add(q, cq3); }
                        if (4u < chunk && seg_base + 4u < up_groups) { p = ordered_add(p, cp4); q = ordered_add(q, cq4); }
                    };
                    if (part == 0u) fold5(0u);
                    if (segs > 1u) {
                        p = moe_wave_contract::xor32NativeBits<8>(p);
                        q = moe_wave_contract::xor32NativeBits<8>(q);
                        if (part == 1u) fold5(chunk);
                    }
                    if (live_ci && part == (segs - 1u))
                        gates_lds[c * n + i] = ordered_add(p, q);
                }
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                const float* norm_out = waveUniform(a.norm.output);
                float* mixed_out = waveUniform(a.mixed);
#pragma unroll 1
                for (unsigned ch = tid; ch < n; ch += 1024u) {
                    const float g0 = gates_lds[0u * n + ch], g1 = gates_lds[1u * n + ch];
                    const float g2 = gates_lds[2u * n + ch], g3 = gates_lds[3u * n + ch];
                    const float x0 = globalWeight<float>(norm_out + std::size_t(0u) * H + begin + ch);
                    const float x1 = globalWeight<float>(norm_out + std::size_t(1u) * H + begin + ch);
                    const float x2 = globalWeight<float>(norm_out + std::size_t(2u) * H + begin + ch);
                    const float x3 = globalWeight<float>(norm_out + std::size_t(3u) * H + begin + ch);
                    float mixed = 0.0f;
#pragma unroll 1
                    for (unsigned c = 0u; c < 4u; ++c) {
                        const float gc = (c == 0u) ? g0 : (c == 1u) ? g1 : (c == 2u) ? g2 : g3;
                        const float xc = (c == 0u) ? x0 : (c == 1u) ? x1 : (c == 2u) ? x2 : x3;
                        mixed = fmaf(1.0f / (1.0f + expf(-gc)), xc, mixed);
                    }
                    asGlobal(mixed_out)[begin + ch] = mixed * 0.25f;
                }
                if (begin + max_panel_n < e1)
                    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
            }
        }
    }
    grid.template sync<2>(); // Barrier 3: mixed/injection published (includes trailing workgroupJoin)
}

/**
 * @brief Compact 2-way lookahead canonical logical-128 router dot reading pre-rounded FP16 hidden from AS3 LDS.
 */
__device__ __forceinline__ float fastRouter128(
    LdsPtr<const std::uint32_t> h2_lds,
    const __half2* weights2,
    unsigned pair_count,
    unsigned lane)
{
    const unsigned u_pairs = waveUniform(pair_count);
    float first = 0.0f, second = 0.0f;
#pragma unroll 1
    for (unsigned bank = 0u; bank < 2u; ++bank) {
        float sum = 0.0f;
#pragma unroll 1
        for (unsigned i = lane + 64u * bank; i < u_pairs; i += 256u) {
            const __half2 w0 = globalWeight<__half2>(weights2 + i);
            const bool live1 = (i + 128u) < u_pairs;
            __half2 w1{};
            if (live1)
                w1 = globalWeight<__half2>(weights2 + i + 128u);
            const float2 hf0 = __half22float2(ldsLoadHalf2(h2_lds, i));
            const float2 wf0 = __half22float2(w0);
            sum += hf0.x * wf0.x + hf0.y * wf0.y;
            if (live1) {
                const float2 hf1 = __half22float2(ldsLoadHalf2(h2_lds, i + 128u));
                const float2 wf1 = __half22float2(w1);
                sum += hf1.x * wf1.x + hf1.y * wf1.y;
            }
        }
        sum = moe_wave_contract::rocm_moe_wave_reduce_sum_f32(sum);
        const float root = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, sum), 0));
        if (bank == 0u) first = root; else second = root;
    }
    const float seed = (lane == 0u) ? first : ((lane == 1u) ? second : 0.0f);
    return moe_wave_contract::rocm_moe_wave_reduce_sum_f32(seed);
}

/** @brief Compact canonical 256-lane FP32 shared-input-gate dot in 1 wave64 (L1I-resident). */
__device__ __forceinline__ float fastSharedGate256(
    const float* input, const float* weights, unsigned width, unsigned lane)
{
    float first = 0.f, second = 0.f, third = 0.f, fourth = 0.f;
    const auto* a = reinterpret_cast<const float4*>(input);
    const auto* b = reinterpret_cast<const float4*>(weights);
    const unsigned quads = width >> 2;
#pragma unroll 1
    for (unsigned bank = 0u; bank < 4u; ++bank) {
        float sum = 0.f;
#pragma unroll 1
        for (unsigned i = lane + bank * 64u; i < quads; i += 256u) {
            const float4 av = globalWeight<float4>(a + i);
            const float4 bv = globalWeight<float4>(b + i);
            sum += av.x * bv.x + av.y * bv.y + av.z * bv.z + av.w * bv.w;
        }
        sum = moe_wave_contract::rocm_moe_wave_reduce_sum_f32(sum);
        const float root = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, sum), 0));
        if (bank == 0u) first = root;
        else if (bank == 1u) second = root;
        else if (bank == 2u) third = root;
        else fourth = root;
    }
    const float seed = (lane == 0u) ? first : (lane == 1u) ? second : (lane == 2u) ? third : (lane == 3u) ? fourth : 0.f;
    return moe_wave_contract::rocm_moe_wave_reduce_sum_f32(seed);
}

/** @brief Aligned cardinal row grain for cohort balancing in routedGateUp. */
__device__ __forceinline__ constexpr unsigned fastRowGrain(unsigned rows, unsigned owners)
{
    const unsigned small = (rows + 7u) / 8u;
    const unsigned large = (rows + 15u) / 16u;
    return (rows % 16u == 0u) && (((large + owners - 1u) / owners) * 16u <= ((small + owners - 1u) / owners) * 8u)
               ? 16u : 8u;
}

/**
 * @brief Compact, L1I-resident Router + 2-col parallel Shared Gate/Up + Shared Input Gate.
 */
template<std::uint8_t SharedCB>
__device__ __forceinline__ void fastRouterAndSharedPhase(
    const NativeLayoutRouterFFNArgs& args,
    FastFFNSuffixScratch& scratch,
    rocm_q8_residual::NativeDecoder& decoder,
    LdsPtr<FastQ8Block48> q8_lds,
    LdsPtr<std::uint32_t> h2_lds)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned gdim = gridDim.x;
    const unsigned H = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(hidden_columns)));
    const unsigned S = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(shared_columns)));
    const unsigned F = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(routed_columns)));
    const unsigned routes = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(top_k)));
    const unsigned experts = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(experts)));

    const float* hidden_ptr = waveUniform(LLAMINAR_FAST_FFN_FIELD(hidden));
    const float4* hidden4 = reinterpret_cast<const float4*>(hidden_ptr);
    const unsigned wave = waveUniform(tid >> 6);
    const unsigned lane = lane64();

    for (unsigned quad = tid; quad < (H >> 2); quad += 1024u) {
        const float4 val = globalWeight<float4>(hidden4 + quad);
        fastQuantizeFourLds(val, quad, q8_lds);
        ldsStoreHalf2(h2_lds, quad * 2u, make_half2(__float2half_rn(val.x), __float2half_rn(val.y)));
        ldsStoreHalf2(h2_lds, quad * 2u + 1u, make_half2(__float2half_rn(val.z), __float2half_rn(val.w)));
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

    const auto router_rows = NativeLayoutDownGeometry::rows(experts, bid, gdim);
    const unsigned nl = waveUniform(router_rows.end - router_rows.begin);
    const auto shared_rows = NativeLayoutDownGeometry::rows(S, bid, gdim);
    const unsigned ns = waveUniform(shared_rows.end - shared_rows.begin);
    const unsigned shared_tasks = (ns + 1u) >> 1;
    const unsigned scalar = (bid == 0u) ? 1u : 0u;
    const unsigned total_tasks = nl + shared_tasks + scalar;

    const __half2* w_all = reinterpret_cast<const __half2*>(waveUniform(LLAMINAR_FAST_ROUTER_FIELD(router_weights)));
    float* router_logits_out = waveUniform(LLAMINAR_FAST_ROUTER_FIELD(router_logits));
    float* gate_partials_out = waveUniform(LLAMINAR_FAST_FFN_FIELD(gate_partials));
    const unsigned pair_count = H >> 1;
    const unsigned blocks = H >> 5;

    // Single descriptor load site for shared Gate/Up (zero duplication inside task loop!).
    FastPhaseProjection s_g_rec{}, s_u_rec{};
    bool shared_descs_loaded = false;

#pragma unroll 1
    for (unsigned task = wave; task < total_tasks; task += 16u) {
        if (task < nl) {
            const unsigned expert = router_rows.begin + task;
            const float val = fastRouter128(h2_lds, w_all + std::size_t(expert) * pair_count, pair_count, lane);
            if (lane == 0u) asGlobal(router_logits_out)[expert] = val;
        } else if (task < nl + shared_tasks) {
            if constexpr (SharedCB != kFastDynamicCodebook) {
                if (!shared_descs_loaded) {
                    const unsigned policy_parts = waveUniform(LLAMINAR_FAST_SUFFIX_POLICY_MAX(shared_gate_policy));
                    const std::uint32_t mask = waveUniform(LLAMINAR_FAST_SUFFIX_MASK(shared_gate));
                    s_g_rec = makeFastProjectionDesc(LLAMINAR_FAST_FFN_FIELD(shared_gate), policy_parts, mask, H);
                    s_u_rec = makeFastProjectionDesc(LLAMINAR_FAST_FFN_FIELD(shared_up), policy_parts, mask, H);
                    if (__builtin_expect((s_g_rec.ready && waveUniform(s_g_rec.codebook) != SharedCB) ||
                                         (s_u_rec.ready && waveUniform(s_u_rec.codebook) != SharedCB), 0))
                        __builtin_trap();
                    shared_descs_loaded = true;
                }
            }
            const unsigned local_first = (task - nl) * 2u;
            const bool paired = waveUniform((local_first + 1u) < ns);
            const unsigned col = lane & 1u;
            const unsigned role = (lane >> 1) & 1u;
            const unsigned part = lane >> 2;
            const unsigned local_row = local_first + col;
            if constexpr (SharedCB == kFastDynamicCodebook) {
                const unsigned policy_parts = waveUniform(LLAMINAR_FAST_SUFFIX_POLICY_MAX(shared_gate_policy));
                const std::uint32_t mask = waveUniform(LLAMINAR_FAST_SUFFIX_MASK(shared_gate));
                float gf = 0.0f, uf = 0.0f;
#pragma unroll 1
                for (unsigned r_idx = 0u; r_idx < 2u; ++r_idx) {
                    const FastPhaseProjection rec = makeFastProjectionDesc(
                        (r_idx == 0u) ? LLAMINAR_FAST_FFN_FIELD(shared_gate) : LLAMINAR_FAST_FFN_FIELD(shared_up),
                        policy_parts, mask, H);
                    const unsigned parts = waveUniform(rec.parts);
                    const unsigned width = waveUniform(rec.partition_blocks);
                    float partial = 0.0f;
                    if (local_row < ns && role == r_idx && waveUniform(rec.ready) && part < parts) {
                        const unsigned bf = part * width;
                        const unsigned bl = (bf < blocks) ? min(blocks, bf + width) : bf;
                        partial = fastPartitionDynamicLds(
                            waveUniform(rec.codebook),
                            asLds<const FastIQTablesLds>(&scratch.iq_tables),
                            waveUniform(rec.payload), waveUniform(rec.scales),
                            waveUniform(rec.mins), waveUniform(rec.emins),
                            shared_rows.begin + local_row, S, bf, bl, q8_lds);
                    }
                    const float folded = __builtin_expect(paired, 1)
                        ? fastFoldFourColumns(partial, parts, lane)
                        : fastFoldStrided(partial, parts, r_idx * 2u, 4u);
                    if (r_idx == 0u) gf = folded;
                    else uf = __builtin_expect(paired, 1) ? moe_wave_contract::rowShiftNativeBits<2>(folded) : folded;
                }
                if (lane < 2u && (local_first + lane) < ns) {
                    const unsigned col_n = shared_rows.begin + local_first + lane;
                    asGlobal(gate_partials_out)[std::size_t(routes) * F + col_n] =
                        decode_equivalent::swigluValue(gf, uf);
                }
            } else {
                const bool g_ready = waveUniform(s_g_rec.ready);
                const bool u_ready = waveUniform(s_u_rec.ready);
                const unsigned parts = waveUniform(s_g_rec.parts);
                const unsigned width = waveUniform(s_g_rec.partition_blocks);
                const bool is_up = (role != 0u);
                float partial = 0.0f;
                if (local_row < ns && (is_up ? u_ready : g_ready) && part < parts) {
                    const unsigned bf = part * width;
                    const unsigned bl = (bf < blocks) ? min(blocks, bf + width) : bf;
                    partial = fastPartition<SharedCB>(
                        decoder,
                        is_up ? s_u_rec.payload : s_g_rec.payload,
                        is_up ? s_u_rec.scales : s_g_rec.scales,
                        is_up ? s_u_rec.mins : s_g_rec.mins,
                        is_up ? s_u_rec.emins : s_g_rec.emins,
                        shared_rows.begin + local_row, S, bf, bl, q8_lds);
                }
                float gf, uf;
                if (__builtin_expect(paired, 1)) {
                    gf = fastFoldFourColumns(partial, parts, lane);
                    uf = moe_wave_contract::rowShiftNativeBits<2>(gf);
                } else {
                    gf = fastFoldStrided(partial, parts, 0u, 4u);
                    uf = fastFoldStrided(partial, parts, 2u, 4u);
                }
                if (lane < 2u && (local_first + lane) < ns) {
                    const unsigned col_n = shared_rows.begin + local_first + lane;
                    asGlobal(gate_partials_out)[std::size_t(routes) * F + col_n] =
                        decode_equivalent::swigluValue(gf, uf);
                }
            }
        } else {
            const float value = fastSharedGate256(
                hidden_ptr, LLAMINAR_FAST_FFN_FIELD(shared_input_gate), H, lane);
            if (lane == 0u) asGlobal(LLAMINAR_FAST_FFN_FIELD(down_partials))[0] = value;
        }
    }
    // Overlap runtime-bank validation on wave 15 completely in parallel with Router+Shared (~10us)!
    // Barrier 4's leading workgroupJoin publishes scratch.bank to all waves before Phase 4.
    if (wave == 15u) {
        auto* rt = reinterpret_cast<DeviceMoELayerRuntimeView*>(LLAMINAR_FAST_FFN_FIELD(runtime));
        fastAcquireValidatedBank(
            asLds<FastValidatedRuntimeBank>(&scratch.bank), rt, experts, routes, true, args.ffn.original, lane);
    }
}

/**
 * @brief Zero-barrier-setup 1-tile 4-col x 16-part peeled 2-block-pipelined paired Gate+Up.
 */
template<std::uint8_t GateCB, bool AllowExpanded>
__device__ __forceinline__ void fastRoutedGateUpPhase(
    FastFFNSuffixScratch& scratch,
    rocm_q8_residual::NativeDecoder& decoder,
    unsigned cohorts,
    unsigned cohort,
    LdsPtr<const FastQ8Block48> q8_lds,
    LdsPtr<unsigned char> panel_lds,
    unsigned panel_lds_bytes)
{
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned gdim = gridDim.x;
    const unsigned H = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(hidden_columns)));
    const unsigned F = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(routed_columns)));
    const unsigned routes = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(top_k)));
    const unsigned blocks = H >> 5;

    const unsigned begin = (cohort * gdim) / cohorts;
    const unsigned end = ((cohort + 1u) * gdim) / cohorts;
    const unsigned owners = end - begin;
    const unsigned rank = bid - begin;
    const unsigned grain = fastRowGrain(F, owners);
    const unsigned chunks = (F + grain - 1u) / grain;
    const unsigned first = ((rank * chunks) / owners) * grain;
    const unsigned last = min(F, (((rank + 1u) * chunks) / owners) * grain);
    if (first >= last) return;

    const unsigned wave = waveUniform(tid >> 6);
    const unsigned lane = lane64();
    const unsigned row_in_wave = lane & 3u;
    const unsigned part = lane >> 2;
    auto bank = asLds<const FastValidatedRuntimeBank>(&scratch.bank);
    const DeviceMoEExpertDescriptorView* experts_table = waveUniform(bank->experts);
    const std::uint8_t* local_mask = waveUniform(bank->local_compute_mask);
    const unsigned expert_count = waveUniform(bank->expert_count);
    const unsigned policy_parts = waveUniform(LLAMINAR_FAST_SUFFIX_POLICY_MAX(routed_gate_policy));
    const std::uint32_t exec_mask = waveUniform(LLAMINAR_FAST_SUFFIX_MASK(routed_gate));

    const unsigned slab_blocks = (blocks + 15u) & ~15u;
    const unsigned slab_stride_words = slab_blocks * 64u;
    if (__builtin_expect(panel_lds_bytes < 2u * slab_stride_words * sizeof(std::uint32_t), 0)) __builtin_trap();
    auto g_panel = asLds<std::uint32_t>(panel_lds);
    auto u_panel = g_panel + slab_stride_words;
    auto sel_ids = asLds<const int>(scratch.selection.selected_ids);

#pragma unroll 1
    for (unsigned route = cohort; route < routes; route += cohorts) {
        const int expert = waveUniform(sel_ids[route]);
        if (__builtin_expect(expert < 0 || static_cast<unsigned>(expert) >= expert_count, 0)) {
            float* gp = waveUniform(LLAMINAR_FAST_FFN_FIELD(gate_partials));
            for (unsigned o = first + tid; o < last; o += 1024u)
                asGlobal(gp)[std::size_t(route) * F + o] = 0.0f;
            continue;
        }
        constexpr bool kDynamicGate = (GateCB == kFastDynamicCodebook);
        if constexpr (kDynamicGate) {
            // Dynamic all-18-codebook path: load one projection descriptor per role so gate and up
            // descriptors never occupy 16 SGPRs simultaneously across the 18-codebook switch on gfx1100.
            const bool is_local = waveUniform(globalWeight<std::uint8_t>(local_mask + expert) != 0u);
            const auto* row_g = asGlobal(experts_table + expert);
            const bool is_native = is_local &&
                waveUniform(static_cast<unsigned>(row_g->weight_format) ==
                            static_cast<unsigned>(DeviceMoEWeightFormat::NativeQuantised));
            if (__builtin_expect(!is_native, 0)) {
                float* gp = waveUniform(LLAMINAR_FAST_FFN_FIELD(gate_partials));
                for (unsigned o = first + tid; o < last; o += 1024u)
                    asGlobal(gp)[std::size_t(route) * F + o] = 0.0f;
                continue;
            }
#pragma unroll 1
            for (unsigned tile = first; tile < last; tile += 64u) {
                const unsigned count = min(64u, last - tile);
                const unsigned wave_row = wave * 4u;
                if (wave_row < count) {
                    const unsigned local_row = wave_row + row_in_wave;
                    const unsigned n = tile + local_row;
                    float gg = 0.0f, uu = 0.0f;
#pragma unroll 1
                    for (unsigned role = 0u; role < 2u; ++role) {
                        const DeviceNativeQuantisedMatrixDesc* desc =
                            (role == 0u) ? &(experts_table + expert)->gate : &(experts_table + expert)->up;
                        const FastPhaseProjection rec = makeFastProjectionDesc(desc, policy_parts, exec_mask, H);
                        float partial = 0.0f;
                        if (local_row < count && waveUniform(rec.ready) && part < waveUniform(rec.parts)) {
                            const unsigned width = waveUniform(rec.partition_blocks);
                            const unsigned bf = part * width;
                            const unsigned bl = (bf < blocks) ? min(blocks, bf + width) : bf;
                            partial = fastPartitionDynamicLds(
                                waveUniform(rec.codebook),
                                asLds<const FastIQTablesLds>(&scratch.iq_tables),
                                waveUniform(rec.payload), waveUniform(rec.scales),
                                waveUniform(rec.mins), waveUniform(rec.emins),
                                n, F, bf, bl, q8_lds);
                        }
                        const float folded = fastFoldFourColumns(partial, waveUniform(rec.parts), lane);
                        if (role == 0u) gg = folded; else uu = folded;
                    }
                    if (lane < 4u && (wave_row + lane) < count) {
                        asGlobal(waveUniform(LLAMINAR_FAST_FFN_FIELD(gate_partials)))[std::size_t(route) * F + tile + wave_row + lane] =
                            decode_equivalent::swigluValue(gg, uu);
                    }
                }
            }
            continue;
        }
        // Single-flight scalar load of local_mask + format + gate/up descriptors (1 round-trip instead of 4!).
        const FastGateUpDescPair pair = makeFastGateUpDescPair<GateCB, AllowExpanded>(
            experts_table, local_mask, static_cast<unsigned>(expert), policy_parts, exec_mask, H);
        if (__builtin_expect(!pair.local_native || (!pair.gate.ready && !pair.up.ready), 0)) {
            float* gp = waveUniform(LLAMINAR_FAST_FFN_FIELD(gate_partials));
            for (unsigned o = first + tid; o < last; o += 1024u)
                asGlobal(gp)[std::size_t(route) * F + o] = 0.0f;
            continue;
        }
        const FastPhaseProjection& g_rec = pair.gate;
        const FastPhaseProjection& u_rec = pair.up;
        const bool g_ready = waveUniform(g_rec.ready);
        const bool u_ready = waveUniform(u_rec.ready);

        const unsigned g_parts = waveUniform(g_rec.parts);
        const unsigned u_parts = waveUniform(u_rec.parts);
        const unsigned g_width = waveUniform(g_rec.partition_blocks);
        const unsigned u_width = waveUniform(u_rec.partition_blocks);
        const std::uint8_t g_cb = waveUniform(g_rec.codebook);
        const std::uint8_t u_cb = waveUniform(u_rec.codebook);

        if constexpr (!AllowExpanded && !kDynamicGate) {
            if (__builtin_expect(!g_ready || !u_ready || g_cb != GateCB || u_cb != GateCB || g_parts != u_parts || g_width != u_width, 0))
                __builtin_trap();
        }
        constexpr bool kGateSec = kDynamicGate
            ? true
            : rocm_q8_residual::NativeDecoder::CodebookTraits<kDynamicGate ? 19u : GateCB>::requires_secondary_metadata;
        const bool paired_fast = !kDynamicGate && (!AllowExpanded ||
            waveUniform(g_ready && u_ready && (g_cb == u_cb) && (g_parts == u_parts) && (g_width == u_width)));

#pragma unroll 1
        for (unsigned tile = first; tile < last; tile += 64u) {
            const unsigned count = min(64u, last - tile);
            if constexpr (!kDynamicGate) {
                if (paired_fast) {
                    if constexpr (!AllowExpanded) {
                        fastPublishMetaPanelPair<kGateSec>(
                            g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                            F, tile, count, tid, blocks, g_panel, u_panel);
                    } else {
                        if (g_cb == GateCB)
                            fastPublishMetaPanelPair<kGateSec>(
                                g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                F, tile, count, tid, blocks, g_panel, u_panel);
                        else
                            fastPublishMetaPanelPair<true>(
                                g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                F, tile, count, tid, blocks, g_panel, u_panel);
                    }
                    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                }
            }
            const unsigned wave_row = wave * 4u;
            if (wave_row < count) {
                const unsigned local_row = wave_row + row_in_wave;
                const unsigned n = tile + local_row;
                float gg = 0.0f, uu = 0.0f;

                if constexpr (kDynamicGate) {
                    if (local_row < count) {
#pragma unroll 1
                        for (unsigned role = 0u; role < 2u; ++role) {
                            const auto& rec = (role == 0u) ? g_rec : u_rec;
                            if (rec.ready && part < rec.parts) {
                                const unsigned bf = part * rec.partition_blocks;
                                const unsigned bl = (bf < blocks) ? min(blocks, bf + rec.partition_blocks) : bf;
                                const float val = fastPartitionDynamicLds(
                                    waveUniform(rec.codebook),
                                    asLds<const FastIQTablesLds>(&scratch.iq_tables),
                                    rec.payload, rec.scales,
                                    rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                if (role == 0u) gg = val; else uu = val;
                            }
                        }
                    }
                } else if constexpr (!AllowExpanded) {
                    if (local_row < count && part < g_parts) {
                        const unsigned bf = part * g_width;
                        const unsigned bl = (bf < blocks) ? min(blocks, bf + g_width) : bf;
                        fastPairedGateUpPanelPartition<GateCB>(
                            decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                            n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                    }
                } else {
                    if (paired_fast) {
                        if (local_row < count && part < g_parts) {
                            const unsigned bf = part * g_width;
                            const unsigned bl = (bf < blocks) ? min(blocks, bf + g_width) : bf;
                            if (g_cb == GateCB) {
                                fastPairedGateUpPanelPartition<GateCB>(
                                    decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                    n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                            } else if (g_cb == kNativeQuantisedExpandedInt8MinCodebook) {
                                fastPairedGateUpPanelPartition<kNativeQuantisedExpandedInt8MinCodebook>(
                                    decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                    n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                            } else {
                                __builtin_trap();
                            }
                        }
                    } else if (local_row < count) {
#pragma unroll 1
                        for (unsigned role = 0u; role < 2u; ++role) {
                            const auto& rec = (role == 0u) ? g_rec : u_rec;
                            if (rec.ready && part < rec.parts) {
                                const unsigned bf = part * rec.partition_blocks;
                                const unsigned bl = (bf < blocks) ? min(blocks, bf + rec.partition_blocks) : bf;
                                float val = 0.0f;
                                if (rec.codebook == GateCB)
                                    val = fastPartition<GateCB>(
                                        decoder, rec.payload, rec.scales, rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                else if (rec.codebook == kNativeQuantisedExpandedInt8MinCodebook)
                                    val = fastPartition<kNativeQuantisedExpandedInt8MinCodebook>(
                                        decoder, rec.payload, rec.scales, rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                else __builtin_trap();
                                if (role == 0u) gg = val; else uu = val;
                            }
                        }
                    }
                }

#pragma unroll 1
                for (unsigned role = 0u; role < 2u; ++role) {
                    const float folded = fastFoldFourColumns(
                        (role == 0u) ? gg : uu, (role == 0u) ? g_parts : u_parts, lane);
                    if (role == 0u) gg = folded; else uu = folded;
                }
                if (lane < 4u && (wave_row + lane) < count) {
                    asGlobal(waveUniform(LLAMINAR_FAST_FFN_FIELD(gate_partials)))[std::size_t(route) * F + tile + wave_row + lane] =
                        decode_equivalent::swigluValue(gg, uu);
                }
            }
            if (paired_fast && (tile + 64u < last || route + cohorts < routes))
                workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        }
    }
}

/**
 * @brief Scalar-dispatched 1-wave-per-route 64-col coalesced Down (zero EXEC-mask spills).
 *
 * Wave-uniform `is_shared`, `ready`, and `cb` are dispatched OUTSIDE `lane < count` so
 * AMDGPU uses scalar `s_cbranch_scc` instead of spilling nested 64-bit EXEC masks.
 */
template<std::uint8_t DownCB, std::uint8_t SharedCB, bool AllowExpanded>
__device__ __forceinline__ void fastDownPhase(
    FastFFNSuffixScratch& scratch,
    rocm_q8_residual::NativeDecoder& decoder,
    LdsPtr<const FastQ8Block48> q8_lds,
    LdsPtr<float> lds_completed)
{
    const unsigned tid = threadIdx.x;
    const unsigned H = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(hidden_columns)));
    const unsigned F = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(routed_columns)));
    const unsigned S = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(shared_columns)));
    const unsigned routes = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(top_k)));

    const auto owned = NativeLayoutDownGeometry::rows(H, blockIdx.x, gridDim.x);
    if (owned.begin == owned.end) return;

    const unsigned wave = waveUniform(tid >> 6);
    const unsigned lane = lane64();
    const float shared_gate_val = waveUniform(scratch.shared_gate_value);
    auto route_experts = asLds<const int>(scratch.route_experts);
    auto route_weights = asLds<const float>(scratch.route_weights);
    auto bank = asLds<const FastValidatedRuntimeBank>(&scratch.bank);
    const DeviceMoEExpertDescriptorView* experts_table = waveUniform(bank->experts);
    const unsigned expert_count = waveUniform(bank->expert_count);

#pragma unroll 1
    for (unsigned first = owned.begin; first < owned.end; first += 64u) {
        const unsigned count = waveUniform(min(64u, owned.end - first));
        const unsigned n = first + lane;

#pragma unroll 1
        for (unsigned r = wave; r <= routes; r += 16u) {
            const bool is_shared = waveUniform(r == routes);
            if constexpr (DownCB == kFastDynamicCodebook && SharedCB == kFastDynamicCodebook) {
                FastPhaseProjection rec{};
                unsigned blk_count = 0u;
                LdsPtr<const FastQ8Block48> act_q8 = q8_lds;
                float rw = 1.0f;
                if (is_shared) {
                    rec = makeFastProjectionDesc(
                        waveUniform(LLAMINAR_FAST_FFN_FIELD(shared_down)),
                        waveUniform(LLAMINAR_FAST_SUFFIX_POLICY_MAX(shared_down_policy)),
                        waveUniform(LLAMINAR_FAST_SUFFIX_MASK(shared_down)), S);
                    blk_count = S >> 5;
                    act_q8 = q8_lds + ((std::size_t(routes) * F) >> 5);
                } else {
                    const int expert = waveUniform(route_experts[r]);
                    const DeviceNativeQuantisedMatrixDesc* down_desc =
                        (expert >= 0 && static_cast<unsigned>(expert) < expert_count)
                            ? &(experts_table + expert)->down : nullptr;
                    rec = makeFastProjectionDesc(
                        down_desc,
                        waveUniform(LLAMINAR_FAST_SUFFIX_POLICY_MAX(routed_down_policy)),
                        waveUniform(LLAMINAR_FAST_SUFFIX_MASK(routed_down)), F);
                    blk_count = F >> 5;
                    act_q8 = q8_lds + ((std::size_t(r) * F) >> 5);
                    rw = waveUniform(route_weights[r]);
                }
                if (__builtin_expect(!waveUniform(rec.ready), 0)) {
                    if (lane < count)
                        lds_completed[std::size_t(r) * 64u + lane] = 0.0f;
                } else if (lane < count) {
                    const float raw = fastDownRowDynamicLds(
                        waveUniform(rec.codebook),
                        asLds<const FastIQTablesLds>(&scratch.iq_tables),
                        waveUniform(rec.payload), waveUniform(rec.scales),
                        waveUniform(rec.mins), waveUniform(rec.emins),
                        n, H, blk_count, waveUniform(rec.parts), waveUniform(rec.partition_blocks), act_q8);
                    const float weighted = rocm_native_quantised_weight_route_rn(rw, raw);
                    lds_completed[std::size_t(r) * 64u + lane] =
                        is_shared ? rocm_native_quantised_accumulate_rn(0.f, weighted) : weighted;
                }
            } else if (is_shared) {
                const FastPhaseProjection rec = makeFastProjectionDesc(
                    waveUniform(LLAMINAR_FAST_FFN_FIELD(shared_down)),
                    waveUniform(LLAMINAR_FAST_SUFFIX_POLICY_MAX(shared_down_policy)),
                    waveUniform(LLAMINAR_FAST_SUFFIX_MASK(shared_down)), S);
                const bool ready = waveUniform(rec.ready);
                const unsigned parts = waveUniform(rec.parts);
                const unsigned width = waveUniform(rec.partition_blocks);
                if (__builtin_expect(!ready, 0)) {
                    if (lane < count)
                        lds_completed[std::size_t(r) * 64u + lane] = 0.0f;
                } else {
                    const std::uint8_t s_cb = waveUniform(rec.codebook);
                    if constexpr (SharedCB != kFastDynamicCodebook) {
                        if (__builtin_expect(s_cb != SharedCB, 0)) __builtin_trap();
                    }
                    const std::uint8_t* pay = waveUniform(rec.payload);
                    const std::uint16_t* sc = waveUniform(rec.scales);
                    const std::uint16_t* mn = waveUniform(rec.mins);
                    const std::uint32_t* em = waveUniform(rec.emins);
                    const unsigned s_blocks = S >> 5;
                    auto s_q8 = q8_lds + ((std::size_t(routes) * F) >> 5);
                    if (lane < count) {
                        const float raw = [&]() {
                            if constexpr (SharedCB == kFastDynamicCodebook)
                                return fastDownRowDynamic(s_cb, decoder, pay, sc, mn, em, n, H, s_blocks, parts, width, s_q8);
                            else
                                return fastDownRow<SharedCB>(decoder, pay, sc, mn, em, n, H, s_blocks, parts, width, s_q8);
                        }();
                        lds_completed[std::size_t(r) * 64u + lane] =
                            rocm_native_quantised_accumulate_rn(0.f, rocm_native_quantised_weight_route_rn(1.f, raw));
                    }
                }
            } else {
                const int expert = waveUniform(route_experts[r]);
                const DeviceNativeQuantisedMatrixDesc* down_desc =
                    (expert >= 0 && static_cast<unsigned>(expert) < expert_count)
                        ? &(experts_table + expert)->down : nullptr;
                const FastPhaseProjection rec = makeFastProjectionDesc(
                    down_desc,
                    waveUniform(LLAMINAR_FAST_SUFFIX_POLICY_MAX(routed_down_policy)),
                    waveUniform(LLAMINAR_FAST_SUFFIX_MASK(routed_down)), F);
                const bool ready = waveUniform(rec.ready);
                const unsigned parts = waveUniform(rec.parts);
                const unsigned width = waveUniform(rec.partition_blocks);
                if (__builtin_expect(!ready, 0)) {
                    if (lane < count)
                        lds_completed[std::size_t(r) * 64u + lane] = 0.0f;
                } else {
                    const std::uint8_t cb = waveUniform(rec.codebook);
                    const std::uint8_t* pay = waveUniform(rec.payload);
                    const std::uint16_t* sc = waveUniform(rec.scales);
                    const std::uint16_t* mn = waveUniform(rec.mins);
                    const std::uint32_t* em = waveUniform(rec.emins);
                    const unsigned f_blocks = F >> 5;
                    auto r_q8 = q8_lds + ((std::size_t(r) * F) >> 5);
                    const float rw = waveUniform(route_weights[r]);
                    if constexpr (DownCB == kFastDynamicCodebook) {
                        if (lane < count) {
                            const float raw = fastDownRowDynamic(
                                cb, decoder, pay, sc, mn, em, n, H, f_blocks, parts, width, r_q8);
                            lds_completed[std::size_t(r) * 64u + lane] = rocm_native_quantised_weight_route_rn(rw, raw);
                        }
                    } else if (__builtin_expect(cb == DownCB, 1)) {
                        if (lane < count) {
                            const float raw = fastDownRow<DownCB>(
                                decoder, pay, sc, mn, em, n, H, f_blocks, parts, width, r_q8);
                            lds_completed[std::size_t(r) * 64u + lane] = rocm_native_quantised_weight_route_rn(rw, raw);
                        }
                    } else if constexpr (AllowExpanded) {
                        if (cb == kNativeQuantisedExpandedInt8MinCodebook) {
                            if (lane < count) {
                                const float raw = fastDownRow<kNativeQuantisedExpandedInt8MinCodebook>(
                                    decoder, pay, sc, mn, em, n, H, f_blocks, parts, width, r_q8);
                                lds_completed[std::size_t(r) * 64u + lane] = rocm_native_quantised_weight_route_rn(rw, raw);
                            }
                        } else {
                            __builtin_trap();
                        }
                    } else {
                        __builtin_trap();
                    }
                }
            }
        }

        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        if (tid < count) {
            const unsigned col_n = first + tid;
            float routed = 0.0f;
            for (unsigned r = 0u; r < routes; ++r)
                routed = rocm_native_quantised_accumulate_rn(routed, lds_completed[std::size_t(r) * 64u + tid]);
            const float shared = device_fp32_contract::multiply(
                shared_gate_val, lds_completed[std::size_t(routes) * 64u + tid]);
            asGlobal(waveUniform(LLAMINAR_FAST_FFN_FIELD(combined_output)))[col_n] = device_fp32_contract::add(routed, shared);
        }
        if (owned.end - first > 64u)
            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
    }
}

/**
 * @brief Complete sharded L1-I-cache-resident HC + Router + MoE FFN megakernel.
 */
template<std::uint8_t GateCB, std::uint8_t DownCB, std::uint8_t SharedCB, bool AllowExpanded = false>
__global__ __launch_bounds__(1024, 1) void fastFullNativeHCFFNKernel(
    NativeLayoutRouterFFNArgs suffix,
    FullNativeHCFFNPrefixArgs prefix,
    unsigned dynamic_bytes,
    NativeEmbeddingIQTables iq_tables)
{
    __shared__ FastFullNativeHCFFNShared shared;
    extern __shared__ __align__(16) unsigned char dynamic_storage[];

    FastWeightedGridBarrier grid{prefix.barrier_state};
    grid.validateEntry();
    fastHCPrefixPhase(prefix, grid, shared.hc, dynamic_storage, dynamic_bytes);

    // After Barrier 3, ZERO prefix fields remain live (only `grid` = 2 SGPRs)!
    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned gdim = gridDim.x;
    const unsigned H = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(hidden_columns)));
    const unsigned F = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(routed_columns)));
    const unsigned S = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(shared_columns)));
    const unsigned routes = waveUniform(static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(top_k)));
    const bool live = waveUniform(suffix.ffn.rows.activeRows() != 0);

    if (__builtin_expect(!live, 0)) goto handle_inactive_row;

    {
        if constexpr (GateCB == kFastDynamicCodebook || DownCB == kFastDynamicCodebook || SharedCB == kFastDynamicCodebook) {
            if (tid == 0u) {
                shared.ffn.iq_tables.iq3s = iq_tables.iq3s;
                shared.ffn.iq_tables.iq3xxs = iq_tables.iq3xxs;
                shared.ffn.iq_tables.iq2s = iq_tables.iq2s;
                shared.ffn.iq_tables.iq2xs = iq_tables.iq2xs;
                shared.ffn.iq_tables.iq2xxs = iq_tables.iq2xxs;
                shared.ffn.iq_tables.iq1s = iq_tables.iq1s;
            }
        }
        rocm_q8_residual::NativeDecoder decoder(
            (GateCB == kFastDynamicCodebook && DownCB == kFastDynamicCodebook && SharedCB == kFastDynamicCodebook)
                ? NativeEmbeddingIQTables{}
                : iq_tables);

        // Phase 1: Hidden Q8 (48B bank-conflict-free) + 2-way Router + Shared Gate/Up.
        const unsigned hidden_bytes = (H >> 5) * static_cast<unsigned>(sizeof(FastQ8Block48));
        auto q8_lds = asLds<FastQ8Block48>(dynamic_storage);
        auto arena_lds = asLds<unsigned char>(dynamic_storage) + hidden_bytes;
        const unsigned avail_after_hidden = (dynamic_bytes > hidden_bytes) ? (dynamic_bytes - hidden_bytes) : 0u;

        fastRouterAndSharedPhase<SharedCB>(suffix, shared.ffn, decoder, q8_lds, asLds<std::uint32_t>(arena_lds));
        grid.sync<3>(); // Barrier 4: router logits + shared activations + shared gate + bank ready

        // Phase 2: Packed TopK Selection (OwnedSlotPrefix) + Zero-FLAT Publish + Paired Routed Gate/Up.
        const unsigned cohorts = min(routes, gdim);
        const unsigned cohort = ((bid + 1u) * cohorts - 1u) / gdim;
        const bool closed_owned_prefix = (gdim >= routes);
        const int selection_extent = (closed_owned_prefix && bid != 0u) ? static_cast<int>(cohort + 1u)
                                                                        : static_cast<int>(routes);
        const bool normalize_here = LLAMINAR_FAST_ROUTER_FIELD(normalize_router_weights) &&
                                    (!closed_owned_prefix || bid == 0u);

        if (tid == 0u)
            shared.ffn.shared_gate_value = 1.0f / (1.0f + expf(-globalWeight<float>(LLAMINAR_FAST_FFN_FIELD(down_partials))));
        auto& selection = shared.ffn.selection;
        const int num_experts = LLAMINAR_FAST_FFN_FIELD(experts);
        router_native::selectProbabilities<false, 256, false, WorkgroupJoinMemory::LDSOnly,
            router_native::TopKTransferPolicy::RowNativeRoot,
            router_native::SelectorIntermediateStorage::ThreadPair, true, true>(
            LLAMINAR_FAST_ROUTER_FIELD(router_logits), nullptr, num_experts, selection_extent, normalize_here, selection);

        if (__builtin_expect(bid == 0u && tid < routes, 0)) {
            fastPublishLiveRouteSlot(
                asLds<const FastValidatedRuntimeBank>(&shared.ffn.bank),
                static_cast<int>(tid),
                asLds<const int>(selection.selected_ids),
                asLds<const float>(selection.selected_weights),
                static_cast<unsigned>(num_experts),
                LLAMINAR_FAST_ROUTER_FIELD(router_indices),
                LLAMINAR_FAST_ROUTER_FIELD(router_weights_output),
                LLAMINAR_FAST_ROUTER_FIELD(write_router_outputs));
        }

        fastRoutedGateUpPhase<GateCB, AllowExpanded>(
            shared.ffn, decoder, cohorts, cohort,
            q8_lds, arena_lds, avail_after_hidden);
        grid.sync<4>(); // Barrier 5: routed SwiGLU activations + authoritative runtime topk ids/weights ready

        // Phase 3: Vectorized Activated Q8 Quantize + Overlapped Route Fetch + 1-wave-per-route Down.
        if (tid >= 960u && (tid - 960u) < routes) {
            const unsigned r = tid - 960u;
            const auto* v_rt = reinterpret_cast<const volatile DeviceMoELayerRuntimeView*>(
                LLAMINAR_FAST_FFN_FIELD(runtime));
            shared.ffn.route_experts[r] = v_rt->topk_expert_ids[r];
            shared.ffn.route_weights[r] = v_rt->topk_weights[r];
        }
        const unsigned values = routes * F + S;
        auto d_q8 = asLds<FastQ8Block48>(dynamic_storage);
        const float4* gate_partials4 = reinterpret_cast<const float4*>(waveUniform(LLAMINAR_FAST_FFN_FIELD(gate_partials)));
        for (unsigned quad = tid; quad < (values >> 2); quad += 1024u) {
            const float4 val = globalWeight<float4>(gate_partials4 + quad);
            fastQuantizeFourLds(val, quad, d_q8);
        }
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

        const unsigned down_q8_bytes = (values >> 5) * static_cast<unsigned>(sizeof(FastQ8Block48));
        auto lds_completed = asLds<float>(dynamic_storage + down_q8_bytes);
        fastDownPhase<DownCB, SharedCB, AllowExpanded>(shared.ffn, decoder, d_q8, lds_completed);
        return;
    }

handle_inactive_row:
    if (bid == 0u) {
        if (waveUniform(tid >> 6) == 0u) {
            auto* rt = reinterpret_cast<DeviceMoELayerRuntimeView*>(LLAMINAR_FAST_FFN_FIELD(runtime));
            fastAcquireValidatedBank(
                asLds<FastValidatedRuntimeBank>(&shared.ffn.bank), rt,
                static_cast<unsigned>(LLAMINAR_FAST_FFN_FIELD(experts)), routes, false, suffix.ffn.original, lane64());
        }
        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
        if (tid < routes) {
            auto* rt = asGlobal(shared.ffn.bank.runtime);
            asGlobal(rt->topk_expert_ids)[tid] = -1;
            asGlobal(rt->topk_weights)[tid] = 0.0f;
            asGlobal(rt->route_participant_ids)[tid] = -1;
            if (LLAMINAR_FAST_ROUTER_FIELD(write_router_outputs)) {
                float* li = LLAMINAR_FAST_ROUTER_FIELD(router_indices);
                float* lw = LLAMINAR_FAST_ROUTER_FIELD(router_weights_output);
                if (li && lw) {
                    asGlobal(li)[tid] = -1.0f;
                    asGlobal(lw)[tid] = 0.0f;
                }
            }
        }
    }
    {
        const auto owned = NativeLayoutDownGeometry::rows(H, bid, gdim);
        float* out = waveUniform(LLAMINAR_FAST_FFN_FIELD(combined_output));
        for (unsigned c = owned.begin + tid; c < owned.end; c += 1024u)
            asGlobal(out)[c] = 0.0f;
    }
}

/** @brief Helper to extract exact single codebook ID (0..31) plus optional EXPANDED_INT8_MIN flag. */
inline bool extractSingleCodebookMask(std::uint32_t mask, std::uint8_t& codebook, bool& has_expanded) noexcept
{
    constexpr std::uint32_t kExpBit = std::uint32_t{1} << kNativeQuantisedExpandedInt8MinCodebook;
    has_expanded = (mask & kExpBit) != 0u;
    const std::uint32_t base = mask & ~kExpBit;
    if (base == 0u) {
        if (!has_expanded) return false;
        codebook = kNativeQuantisedExpandedInt8MinCodebook;
        has_expanded = false;
        return true;
    }
    if ((base & (base - 1u)) != 0u) return false;
    codebook = static_cast<std::uint8_t>(__builtin_ctz(base));
    return true;
}

/** @brief Decode (GateCB, DownCB, SharedCB, allow_expanded) from execution masks. */
inline bool decodeFastKey(
    const ResidentMoEFFNExecutionMasks& execution,
    std::uint8_t& gate_cb,
    std::uint8_t& down_cb,
    std::uint8_t& shared_cb,
    bool& allow_expanded) noexcept
{
    bool g_exp = false, d_exp = false, sg_exp = false, sd_exp = false;
    std::uint8_t sg_cb = 0u, sd_cb = 0u;
    if (!extractSingleCodebookMask(execution.routed_gate, gate_cb, g_exp) ||
        !extractSingleCodebookMask(execution.routed_down, down_cb, d_exp) ||
        !extractSingleCodebookMask(execution.shared_gate, sg_cb, sg_exp) ||
        !extractSingleCodebookMask(execution.shared_down, sd_cb, sd_exp))
        return false;
    if (sg_exp || sd_exp || sg_cb != sd_cb)
        return false;
    shared_cb = sg_cb;
    allow_expanded = g_exp || d_exp;
    return true;
}

/** @brief Populate FastPhaseProjection directly from a by-value descriptor (safe for kernargs). */
__device__ __forceinline__ FastPhaseProjection makeFastProjectionDescFromValue(
    const DeviceNativeQuantisedMatrixDesc& d,
    unsigned policy_parts,
    std::uint32_t execution_mask,
    unsigned reduction)
{
    FastPhaseProjection r{};
    r.parts = 1u;
    r.partition_blocks = reduction >> 5;
    const auto* pay = waveUniform(d.payload);
    const auto* sc = waveUniform(static_cast<const std::uint16_t*>(d.scales));
    if (__builtin_expect(!pay || !sc, 0)) return r;
    const std::uint8_t cb = waveUniform(d.codebook_id);
    if (__builtin_expect(!residentMoEFFNDecoderAllows(execution_mask, cb), 0))
        __builtin_trap();
    r.parts = policy_parts;
    r.partition_blocks = ((reduction >> 5) + policy_parts - 1u) / policy_parts;
    r.payload = pay;
    r.scales = sc;
    r.mins = waveUniform(static_cast<const std::uint16_t*>(d.mins));
    r.emins = waveUniform(static_cast<const std::uint32_t*>(d.emins));
    r.codebook = cb;
    r.ready = true;
    return r;
}

/** @brief Compact static LDS for fastMappedFFNSuffixKernel (replaces unused TopK scratch). */
struct FastMappedSuffixScratch {
    FastValidatedRuntimeBank bank;
    int local_experts[router_native::kDeviceMoEMaxTopK];
    unsigned local_slots[router_native::kDeviceMoEMaxTopK];
    unsigned remote_slots[router_native::kDeviceMoEMaxTopK];
    unsigned local_count;
    unsigned remote_count;
};
static_assert(sizeof(FastMappedSuffixScratch) <= 3456u);

/**
 * @brief Mapped-overlay FFN Suffix megakernel: compact local routed + shared Gate/Up/SwiGLU + Down.
 *
 * Runs concurrently with outbound mapped dispatch/ticket publication after
 * `fastMappedRoutePrefixKernel` completes. Compacts active local routed slots
 * so 100% of admitted CTAs execute live GPU experts (skipping remote CPU slots),
 * writes raw unweighted local contributions (and exact 0.0f for remote slots)
 * to `route_contributions`, and writes un-gated partial Shared Down to `shared_output`.
 */
template<std::uint8_t GateCB, std::uint8_t DownCB, std::uint8_t SharedCB, bool AllowExpanded = false>
__global__ __launch_bounds__(1024, 1) void fastMappedFFNSuffixKernel(
    MappedFFNSuffixArgs args,
    unsigned dynamic_bytes,
    NativeEmbeddingIQTables iq_tables)
{
    __shared__ FastMappedSuffixScratch scratch;
    extern __shared__ __align__(16) unsigned char dynamic_storage[];

    FastWeightedGridBarrier grid{args.barrier_state};
    grid.validateEntry();

    const unsigned tid = threadIdx.x;
    const unsigned bid = blockIdx.x;
    const unsigned gdim = gridDim.x;
    const unsigned wave = waveUniform(tid >> 6);
    const unsigned lane = lane64();
    const unsigned H = waveUniform(static_cast<unsigned>(args.hidden_columns));
    const unsigned F = waveUniform(static_cast<unsigned>(args.routed_columns));
    const unsigned S = waveUniform(static_cast<unsigned>(args.shared_columns));
    const unsigned routes = waveUniform(static_cast<unsigned>(args.top_k));
    const unsigned experts = waveUniform(static_cast<unsigned>(args.experts));
    const bool live = waveUniform(args.rows.activeRows() != 0);

    if (__builtin_expect(!live, 0)) {
        const auto owned = NativeLayoutDownGeometry::rows(H, bid, gdim);
        auto* rc_out = asGlobal(waveUniform(args.route_contributions));
        auto* sh_out = asGlobal(waveUniform(args.shared_output));
        for (unsigned r = 0u; r < routes; ++r) {
            for (unsigned c = owned.begin + tid; c < owned.end; c += 1024u)
                rc_out[std::size_t(r) * H + c] = 0.0f;
        }
        for (unsigned c = owned.begin + tid; c < owned.end; c += 1024u)
            sh_out[c] = 0.0f;
        return;
    }

    rocm_q8_residual::NativeDecoder decoder(iq_tables);

    // Phase 0: Quantize mixed hidden into 48B bank-conflict-free AS3 LDS while
    // Wave 15 validates the acquired bank and compacts ONLY live local route slots.
    const unsigned hidden_bytes = (H >> 5) * static_cast<unsigned>(sizeof(FastQ8Block48));
    auto q8_lds = asLds<FastQ8Block48>(dynamic_storage);
    auto panel_lds = asLds<unsigned char>(dynamic_storage) + hidden_bytes;
    const unsigned avail_after_hidden = (dynamic_bytes > hidden_bytes) ? (dynamic_bytes - hidden_bytes) : 0u;

    const float4* hidden4 = reinterpret_cast<const float4*>(waveUniform(args.hidden));
    for (unsigned quad = tid; quad < (H >> 2); quad += 1024u) {
        const float4 val = globalWeight<float4>(hidden4 + quad);
        fastQuantizeFourLds(val, quad, q8_lds);
    }
    if (wave == 15u) {
        auto* rt = reinterpret_cast<DeviceMoELayerRuntimeView*>(waveUniform(args.runtime));
        fastAcquireValidatedBank(
            asLds<FastValidatedRuntimeBank>(&scratch.bank), rt, experts, routes, true, args.original, lane);
        if (lane == 0u) {
            const auto* v_rt = reinterpret_cast<const volatile DeviceMoELayerRuntimeView*>(rt);
            unsigned lc = 0u, rc = 0u;
            for (unsigned r = 0u; r < routes; ++r) {
                const int eid = v_rt->topk_expert_ids[r];
                if (eid >= 0) {
                    scratch.local_experts[lc] = eid;
                    scratch.local_slots[lc] = r;
                    ++lc;
                } else {
                    scratch.remote_slots[rc] = r;
                    ++rc;
                }
            }
            scratch.local_count = lc;
            scratch.remote_count = rc;
        }
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

    const unsigned local_count = waveUniform(scratch.local_count);
    const unsigned blocks = H >> 5;
    float* gate_partials_out = waveUniform(args.gate_partials);

    // Phase 1a: Shared Expert Gate + Up + SwiGLU across all CTAs (writes compact slot local_count*F).
    {
        const auto shared_rows = NativeLayoutDownGeometry::rows(S, bid, gdim);
        const unsigned ns = waveUniform(shared_rows.end - shared_rows.begin);
        const unsigned shared_tasks = (ns + 1u) >> 1;
        if (shared_tasks > 0u) {
            const unsigned policy_parts = waveUniform(args.shared_gate_policy.maximum);
            const std::uint32_t mask = waveUniform(args.execution.shared_gate);
            const FastPhaseProjection s_g_rec = makeFastProjectionDescFromValue(args.shared_gate, policy_parts, mask, H);
            const FastPhaseProjection s_u_rec = makeFastProjectionDescFromValue(args.shared_up, policy_parts, mask, H);
            if constexpr (SharedCB != kFastDynamicCodebook) {
                if (__builtin_expect((s_g_rec.ready && waveUniform(s_g_rec.codebook) != SharedCB) ||
                                     (s_u_rec.ready && waveUniform(s_u_rec.codebook) != SharedCB), 0))
                    __builtin_trap();
            }
            const bool g_ready = waveUniform(s_g_rec.ready);
            const bool u_ready = waveUniform(s_u_rec.ready);
            const unsigned parts = waveUniform(s_g_rec.parts);
            const unsigned width = waveUniform(s_g_rec.partition_blocks);
#pragma unroll 1
            for (unsigned task = wave; task < shared_tasks; task += 16u) {
                const unsigned local_first = task * 2u;
                const bool paired = waveUniform((local_first + 1u) < ns);
                const unsigned col = lane & 1u;
                const unsigned role = (lane >> 1) & 1u;
                const unsigned part = lane >> 2;
                const unsigned local_row = local_first + col;
                const bool is_up = (role != 0u);
                float partial = 0.0f;
                if (local_row < ns && (is_up ? u_ready : g_ready) && part < parts) {
                    const unsigned bf = part * width;
                    const unsigned bl = (bf < blocks) ? min(blocks, bf + width) : bf;
                    if constexpr (SharedCB == kFastDynamicCodebook) {
                        partial = fastPartitionDynamic(
                            is_up ? waveUniform(s_u_rec.codebook) : waveUniform(s_g_rec.codebook),
                            decoder,
                            is_up ? s_u_rec.payload : s_g_rec.payload,
                            is_up ? s_u_rec.scales : s_g_rec.scales,
                            is_up ? s_u_rec.mins : s_g_rec.mins,
                            is_up ? s_u_rec.emins : s_g_rec.emins,
                            shared_rows.begin + local_row, S, bf, bl, q8_lds);
                    } else {
                        partial = fastPartition<SharedCB>(
                            decoder,
                            is_up ? s_u_rec.payload : s_g_rec.payload,
                            is_up ? s_u_rec.scales : s_g_rec.scales,
                            is_up ? s_u_rec.mins : s_g_rec.mins,
                            is_up ? s_u_rec.emins : s_g_rec.emins,
                            shared_rows.begin + local_row, S, bf, bl, q8_lds);
                    }
                }
                float gf, uf;
                if (__builtin_expect(paired, 1)) {
                    gf = fastFoldFourColumns(partial, parts, lane);
                    uf = moe_wave_contract::rowShiftNativeBits<2>(gf);
                } else {
                    gf = fastFoldStrided(partial, parts, 0u, 4u);
                    uf = fastFoldStrided(partial, parts, 2u, 4u);
                }
                if (lane < 2u && (local_first + lane) < ns) {
                    const unsigned col_n = shared_rows.begin + local_first + lane;
                    asGlobal(gate_partials_out)[std::size_t(local_count) * F + col_n] =
                        decode_equivalent::swigluValue(gf, uf);
                }
            }
        }
    }

    // Phase 1b: Compact Local Routed Gate + Up + SwiGLU (ONLY live GPU experts across all CTAs!).
    if (local_count > 0u) {
        const unsigned cohorts = min(local_count, gdim);
        const unsigned cohort = ((bid + 1u) * cohorts - 1u) / gdim;
        const unsigned begin = (cohort * gdim) / cohorts;
        const unsigned end = ((cohort + 1u) * gdim) / cohorts;
        const unsigned owners = end - begin;
        const unsigned rank = bid - begin;
        const unsigned grain = fastRowGrain(F, owners);
        const unsigned chunks = (F + grain - 1u) / grain;
        const unsigned first = ((rank * chunks) / owners) * grain;
        const unsigned last = min(F, (((rank + 1u) * chunks) / owners) * grain);
        if (first < last) {
            const unsigned row_in_wave = lane & 3u;
            const unsigned part = lane >> 2;
            auto bank = asLds<const FastValidatedRuntimeBank>(&scratch.bank);
            const DeviceMoEExpertDescriptorView* experts_table = waveUniform(bank->experts);
            const std::uint8_t* local_mask = waveUniform(bank->local_compute_mask);
            const unsigned expert_count = waveUniform(bank->expert_count);
            const unsigned policy_parts = waveUniform(args.routed_gate_policy.maximum);
            const std::uint32_t exec_mask = waveUniform(args.execution.routed_gate);

            const unsigned slab_blocks = (blocks + 15u) & ~15u;
            const unsigned slab_stride_words = slab_blocks * 64u;
            if (__builtin_expect(avail_after_hidden < 2u * slab_stride_words * sizeof(std::uint32_t), 0))
                __builtin_trap();
            auto g_panel = asLds<std::uint32_t>(panel_lds);
            auto u_panel = g_panel + slab_stride_words;
            auto loc_exp = asLds<const int>(scratch.local_experts);

#pragma unroll 1
            for (unsigned k = cohort; k < local_count; k += cohorts) {
                const int expert = waveUniform(loc_exp[k]);
                if (__builtin_expect(expert < 0 || static_cast<unsigned>(expert) >= expert_count, 0))
                    __builtin_trap();
                const FastGateUpDescPair pair = makeFastGateUpDescPair<GateCB, AllowExpanded>(
                    experts_table, local_mask, static_cast<unsigned>(expert), policy_parts, exec_mask, H);
                if (__builtin_expect(!pair.local_native || (!pair.gate.ready && !pair.up.ready), 0)) {
                    for (unsigned o = first + tid; o < last; o += 1024u)
                        asGlobal(gate_partials_out)[std::size_t(k) * F + o] = 0.0f;
                    continue;
                }
                const FastPhaseProjection& g_rec = pair.gate;
                const FastPhaseProjection& u_rec = pair.up;
                const bool g_ready = waveUniform(g_rec.ready);
                const bool u_ready = waveUniform(u_rec.ready);
                const unsigned g_parts = waveUniform(g_rec.parts);
                const unsigned u_parts = waveUniform(u_rec.parts);
                const unsigned g_width = waveUniform(g_rec.partition_blocks);
                const unsigned u_width = waveUniform(u_rec.partition_blocks);
                const std::uint8_t g_cb = waveUniform(g_rec.codebook);
                const std::uint8_t u_cb = waveUniform(u_rec.codebook);
                constexpr bool kDynamicGate = (GateCB == kFastDynamicCodebook);
                if constexpr (!AllowExpanded && !kDynamicGate) {
                    if (__builtin_expect(!g_ready || !u_ready || g_cb != GateCB || u_cb != GateCB ||
                                         g_parts != u_parts || g_width != u_width, 0))
                        __builtin_trap();
                }
                constexpr bool kGateSec = kDynamicGate
                    ? true
                    : rocm_q8_residual::NativeDecoder::CodebookTraits<kDynamicGate ? 19u : GateCB>::requires_secondary_metadata;
                const bool paired_fast = !kDynamicGate && (!AllowExpanded ||
                    waveUniform(g_ready && u_ready && (g_cb == u_cb) && (g_parts == u_parts) && (g_width == u_width)));

#pragma unroll 1
                for (unsigned tile = first; tile < last; tile += 64u) {
                    const unsigned count = min(64u, last - tile);
                    if constexpr (!kDynamicGate) {
                        if (paired_fast) {
                            if constexpr (!AllowExpanded) {
                                fastPublishMetaPanelPair<kGateSec>(
                                    g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                    F, tile, count, tid, blocks, g_panel, u_panel);
                            } else {
                                if (g_cb == GateCB)
                                    fastPublishMetaPanelPair<kGateSec>(
                                        g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                        F, tile, count, tid, blocks, g_panel, u_panel);
                                else
                                    fastPublishMetaPanelPair<true>(
                                        g_rec.scales, g_rec.mins, u_rec.scales, u_rec.mins,
                                        F, tile, count, tid, blocks, g_panel, u_panel);
                            }
                            workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                        }
                    }
                    const unsigned wave_row = wave * 4u;
                    if (wave_row < count) {
                        const unsigned local_row = wave_row + row_in_wave;
                        const unsigned n = tile + local_row;
                        float gg = 0.0f, uu = 0.0f;
                        if constexpr (kDynamicGate) {
                            if (local_row < count) {
#pragma unroll 1
                                for (unsigned role = 0u; role < 2u; ++role) {
                                    const auto& rec = (role == 0u) ? g_rec : u_rec;
                                    if (rec.ready && part < rec.parts) {
                                        const unsigned bf = part * rec.partition_blocks;
                                        const unsigned bl = (bf < blocks) ? min(blocks, bf + rec.partition_blocks) : bf;
                                        const float val = fastPartitionDynamic(
                                            waveUniform(rec.codebook), decoder, rec.payload, rec.scales,
                                            rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                        if (role == 0u) gg = val; else uu = val;
                                    }
                                }
                            }
                        } else if constexpr (!AllowExpanded) {
                            if (local_row < count && part < g_parts) {
                                const unsigned bf = part * g_width;
                                const unsigned bl = (bf < blocks) ? min(blocks, bf + g_width) : bf;
                                fastPairedGateUpPanelPartition<GateCB>(
                                    decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                    n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                            }
                        } else {
                            if (paired_fast) {
                                if (local_row < count && part < g_parts) {
                                    const unsigned bf = part * g_width;
                                    const unsigned bl = (bf < blocks) ? min(blocks, bf + g_width) : bf;
                                    if (g_cb == GateCB) {
                                        fastPairedGateUpPanelPartition<GateCB>(
                                            decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                            n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                                    } else if (g_cb == kNativeQuantisedExpandedInt8MinCodebook) {
                                        fastPairedGateUpPanelPartition<kNativeQuantisedExpandedInt8MinCodebook>(
                                            decoder, g_rec.payload, g_rec.emins, u_rec.payload, u_rec.emins,
                                            n, F, local_row, bf, bl, q8_lds, g_panel, u_panel, gg, uu);
                                    } else {
                                        __builtin_trap();
                                    }
                                }
                            } else if (local_row < count) {
#pragma unroll 1
                                for (unsigned role = 0u; role < 2u; ++role) {
                                    const auto& rec = (role == 0u) ? g_rec : u_rec;
                                    if (rec.ready && part < rec.parts) {
                                        const unsigned bf = part * rec.partition_blocks;
                                        const unsigned bl = (bf < blocks) ? min(blocks, bf + rec.partition_blocks) : bf;
                                        float val = 0.0f;
                                        if (rec.codebook == GateCB)
                                            val = fastPartition<GateCB>(
                                                decoder, rec.payload, rec.scales, rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                        else if (rec.codebook == kNativeQuantisedExpandedInt8MinCodebook)
                                            val = fastPartition<kNativeQuantisedExpandedInt8MinCodebook>(
                                                decoder, rec.payload, rec.scales, rec.mins, rec.emins, n, F, bf, bl, q8_lds);
                                        else __builtin_trap();
                                        if (role == 0u) gg = val; else uu = val;
                                    }
                                }
                            }
                        }
#pragma unroll 1
                        for (unsigned role = 0u; role < 2u; ++role) {
                            const float folded = fastFoldFourColumns(
                                (role == 0u) ? gg : uu, (role == 0u) ? g_parts : u_parts, lane);
                            if (role == 0u) gg = folded; else uu = folded;
                        }
                        if (lane < 4u && (wave_row + lane) < count) {
                            asGlobal(gate_partials_out)[std::size_t(k) * F + tile + wave_row + lane] =
                                decode_equivalent::swigluValue(gg, uu);
                        }
                    }
                    if (paired_fast && (tile + 64u < last || k + cohorts < local_count))
                        workgroupJoin<WorkgroupJoinMemory::LDSOnly>();
                }
            }
        }
    }

    // Single grid barrier (Phase 4): all compact local routed + shared SwiGLU activations visible!
    grid.sync<4>();

    // Phase 2: Compact Activated Q8 quantization (ONLY local_count*F + S values!).
    const unsigned values = local_count * F + S;
    auto d_q8 = asLds<FastQ8Block48>(dynamic_storage);
    const float4* gate_partials4 = reinterpret_cast<const float4*>(gate_partials_out);
    for (unsigned quad = tid; quad < (values >> 2); quad += 1024u) {
        const float4 val = globalWeight<float4>(gate_partials4 + quad);
        fastQuantizeFourLds(val, quad, d_q8);
    }
    workgroupJoin<WorkgroupJoinMemory::LDSOnly>();

    // Phase 3: Direct coalesced 1-wave-per-slot Down (raw unweighted local routed, exact 0.0f remote,
    // un-gated partial Shared Down — zero trailing LDS fold needed!).
    const auto owned = NativeLayoutDownGeometry::rows(H, bid, gdim);
    if (owned.begin == owned.end) return;

    auto bank = asLds<const FastValidatedRuntimeBank>(&scratch.bank);
    const DeviceMoEExpertDescriptorView* experts_table = waveUniform(bank->experts);
    const unsigned expert_count = waveUniform(bank->expert_count);
    auto loc_exp = asLds<const int>(scratch.local_experts);
    auto loc_slots = asLds<const unsigned>(scratch.local_slots);
    auto rem_slots = asLds<const unsigned>(scratch.remote_slots);
    auto* rc_out = asGlobal(waveUniform(args.route_contributions));
    auto* sh_out = asGlobal(waveUniform(args.shared_output));

#pragma unroll 1
    for (unsigned first = owned.begin; first < owned.end; first += 64u) {
        const unsigned count = waveUniform(min(64u, owned.end - first));
        const unsigned n = first + lane;
#pragma unroll 1
        for (unsigned task = wave; task <= routes; task += 16u) {
            if (task < local_count) {
                const unsigned orig_slot = waveUniform(loc_slots[task]);
                const int expert = waveUniform(loc_exp[task]);
                if (__builtin_expect(expert < 0 || static_cast<unsigned>(expert) >= expert_count, 0))
                    __builtin_trap();
                const FastPhaseProjection rec = makeFastProjectionDesc(
                    &experts_table[static_cast<unsigned>(expert)].down,
                    waveUniform(args.routed_down_policy.maximum),
                    waveUniform(args.execution.routed_down), F);
                const bool ready = waveUniform(rec.ready);
                if (__builtin_expect(!ready, 0)) {
                    if (lane < count) rc_out[std::size_t(orig_slot) * H + n] = 0.0f;
                } else {
                    const std::uint8_t cb = waveUniform(rec.codebook);
                    const unsigned parts = waveUniform(rec.parts);
                    const unsigned width = waveUniform(rec.partition_blocks);
                    auto r_q8 = d_q8 + ((std::size_t(task) * F) >> 5);
                    if constexpr (DownCB == kFastDynamicCodebook) {
                        if (lane < count) {
                            rc_out[std::size_t(orig_slot) * H + n] = fastDownRowDynamic(
                                cb, decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                waveUniform(rec.mins), waveUniform(rec.emins),
                                n, H, F >> 5, parts, width, r_q8);
                        }
                    } else if (__builtin_expect(cb == DownCB, 1)) {
                        if (lane < count) {
                            rc_out[std::size_t(orig_slot) * H + n] = fastDownRow<DownCB>(
                                decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                waveUniform(rec.mins), waveUniform(rec.emins),
                                n, H, F >> 5, parts, width, r_q8);
                        }
                    } else if constexpr (AllowExpanded) {
                        if (cb == kNativeQuantisedExpandedInt8MinCodebook) {
                            if (lane < count) {
                                rc_out[std::size_t(orig_slot) * H + n] = fastDownRow<kNativeQuantisedExpandedInt8MinCodebook>(
                                    decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                    waveUniform(rec.mins), waveUniform(rec.emins),
                                    n, H, F >> 5, parts, width, r_q8);
                            }
                        } else {
                            __builtin_trap();
                        }
                    } else {
                        __builtin_trap();
                    }
                }
            } else if (task == local_count) {
                const FastPhaseProjection rec = makeFastProjectionDescFromValue(
                    args.shared_down,
                    waveUniform(args.shared_down_policy.maximum),
                    waveUniform(args.execution.shared_down), S);
                const bool ready = waveUniform(rec.ready);
                if (__builtin_expect(!ready, 0)) {
                    if (lane < count) sh_out[n] = 0.0f;
                } else {
                    const std::uint8_t s_cb = waveUniform(rec.codebook);
                    if constexpr (SharedCB != kFastDynamicCodebook) {
                        if (__builtin_expect(s_cb != SharedCB, 0))
                            __builtin_trap();
                    }
                    if (lane < count) {
                        if constexpr (SharedCB == kFastDynamicCodebook) {
                            sh_out[n] = fastDownRowDynamic(
                                s_cb, decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                waveUniform(rec.mins), waveUniform(rec.emins),
                                n, H, S >> 5, waveUniform(rec.parts), waveUniform(rec.partition_blocks),
                                d_q8 + ((std::size_t(local_count) * F) >> 5));
                        } else {
                            sh_out[n] = fastDownRow<SharedCB>(
                                decoder, waveUniform(rec.payload), waveUniform(rec.scales),
                                waveUniform(rec.mins), waveUniform(rec.emins),
                                n, H, S >> 5, waveUniform(rec.parts), waveUniform(rec.partition_blocks),
                                d_q8 + ((std::size_t(local_count) * F) >> 5));
                        }
                    }
                }
            } else {
                const unsigned remote_slot = waveUniform(rem_slots[task - local_count - 1u]);
                if (lane < count) rc_out[std::size_t(remote_slot) * H + n] = 0.0f;
            }
        }
    }
}

} // namespace llaminar2::rocm::fast_full_hc_ffn
