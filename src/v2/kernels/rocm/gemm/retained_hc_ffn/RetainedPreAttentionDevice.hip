/**
 * @file RetainedPreAttentionDevice.hip
 * @brief Native retained fixed-shape POST/HC followed by original ordered Q8 projection.
 *
 * The original graph owns the tensors, row range, descriptor bank and persistent
 * scratch. This translation unit owns only the exact resident device entry
 * and its byte layout. Castagna ancestry and MIT terms accompany the body.
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "CastagnaNativeWeightExport.h"
#include "kernels/rocm/gemm/RetainedPreAttentionDevice.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/rocm/moe/ROCmNativeRuntimeBankView.h"
#include "RuntimeNativeExpertTableView.hip.inc"
#include "CanonicalFFNQ8.h"
#include "kernels/common/DeviceNativeQuantisedContributionContract.h"
#include "kernels/common/MoEProjectionNumericalContract.h"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/rocm/common/ROCmQ4PairedDot.h"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"
#include "kernels/common/OriginalSingleQ8Block32.cuh"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/common/QuantizedActivationOperandPrecision.h"
#include "kernels/rocm/gemm/ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/rocm/gemm/ROCmQ8ResidualNativeDecode.cuh"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
#include "kernels/rocm/gemm/ROCmResidentGridBarrier.cuh"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFNPrefixBody.hip.inc"
#include <algorithm>
#include <bit>
#include <type_traits>

extern "C" bool rocmNativeEmbeddingIQTableView(int device_ordinal, llaminar2::NativeEmbeddingIQTables *out);

namespace qfn_retained_m1_hc_ffn_device {
constexpr int E = 2560;
constexpr int HC = 4;
constexpr int K = HC * E;
constexpr int LR = 320;
constexpr int NT = 1024;

template<class T>
__device__ __forceinline__ T native_global_weight_read(const void* pointer)
{
    const auto* source = reinterpret_cast<const unsigned char __attribute__((address_space(1)))*>(
        reinterpret_cast<std::uintptr_t>(pointer));
    T value;
    __builtin_memcpy(&value, source, sizeof(T));
    return value;
}

using ProbeProjection=llaminar2::rocm::pre_attention_probe::Projection;
using FullGDNChain=llaminar2::rocm::pre_attention_probe::FullGDNChain;
constexpr unsigned kMaxFullGDNVHeads = 48u;
constexpr unsigned kMaxFullGDNOutBlocks = kMaxFullGDNVHeads * 4u; // 48 * 128 / 32 = 192 blocks

/**
 * @brief 16-byte-aligned 48-byte Q8 activation block for bank-conflict-free ds_read_b128.
 *
 * Stride = 12 dwords eliminates 2-way LDS bank conflicts versus 40-byte CanonicalFFNQ8Block
 * and allows each half-wave to load 16 int8 activation codes in a single 128-bit LDS read.
 */
struct alignas(16) PreAttnQ8Block48
{
    union {
        std::int32_t qs[8];
        struct {
            int4 q0;
            int4 q1;
        };
    };
    float scale;
    std::int32_t pad[3];
};
static_assert(sizeof(PreAttnQ8Block48) == 48u && alignof(PreAttnQ8Block48) == 16u);

constexpr std::size_t kFullGDNBarrierBytes = (llaminar2::FullNativeHCFFNBarrierLayout::bytes + 15u) & ~std::size_t(15u);
constexpr std::size_t kFullGDNQ8Offset = kFullGDNBarrierBytes;
constexpr std::size_t kFullGDNDoneOffset =
    ((kFullGDNQ8Offset + kMaxFullGDNOutBlocks * sizeof(PreAttnQ8Block48)) + 15u) & ~std::size_t(15u);
constexpr std::size_t kFullGDNScratchBytes =
    ((kFullGDNDoneOffset + kMaxFullGDNVHeads * sizeof(std::uint32_t)) + 15u) & ~std::size_t(15u);

/** @brief Bare device pointers from the already authenticated half owner. */
struct HalfPointers
{
    const std::uint16_t* alpha;
    const std::uint16_t* beta;
    float* raw_alpha;
    float* raw_beta;
    unsigned heads;
};

/**
 * @brief Low-latency gfx906 scalar-memory resident grid barrier for arbitrary gridDim.x.
 *
 * All 16 waves drain pending vector/LDS memory (gfx906 L1 is write-through to L2),
 * wave 0 increments a per-phase scalar counter in L2 via s_atomic_add and spins on
 * s_load_dword glc until every CTA has arrived, then invalidates L1 via buffer_wbinvl1_vol.
 * Handles 32-bit wraparound via signed difference and preserves the existing barrier_state ABI.
 */
template<unsigned PhaseCount>
struct ScalarResidentGridBarrierFor {
    std::uint32_t* words;
    unsigned phase = 0u;
    __device__ __forceinline__ void sync()
    {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
        if (!words || phase >= PhaseCount) __builtin_trap();
        const unsigned grid = __builtin_amdgcn_readfirstlane(gridDim.x);
        std::uint32_t* counter = words + llaminar2::ResidentGridBarrierLayout<PhaseCount>::words_per_phase * phase;
        __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
        __asm__ volatile("s_barrier" ::: "memory");
        if (threadIdx.x < 64u) {
            unsigned old = 1u;
            __asm__ volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                             : "+s"(old) : "s"(counter) : "memory");
            const unsigned target = (old / grid + 1u) * grid;
            while (true) {
                unsigned cur = 0u;
                __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                                 : "=s"(cur) : "s"(counter) : "memory");
                if (static_cast<int>(cur - target) >= 0) break;
                __builtin_amdgcn_s_sleep(1);
            }
            __asm__ volatile("buffer_wbinvl1_vol" ::: "memory");
        }
        __asm__ volatile("s_barrier" ::: "memory");
        ++phase;
#else
        __builtin_trap();
#endif
    }
};

/**
 * @brief 1024-byte coalesced 32-column CTA tile Q8_0 projection (2-block pipelined, double-buffered LDS).
 *
 * All 16 waves own the 16 canonical K-partitions. Within each wave64, lanes 0..63 map to 32 contiguous
 * columns x 2 16-byte halves, issuing one 1024-byte coalesced HBM2 transaction per wave (~778 GB/s on MI50).
 * Reads 16-byte-aligned 48-byte Q8 activation blocks via bank-conflict-free 128-bit LDS loads.
 */
template<bool InlineConv = false>
__device__ __forceinline__ void completeProjectionTile32(
    const ProbeProjection& projection, unsigned col_base,
    const PreAttnQ8Block48* activated, unsigned width, float* lds_partials,
    bool do_conv = false,
    const float* conv_weight = nullptr, const float* conv_bias = nullptr,
    float* conv_state = nullptr, float* gdn_recurrence_in = nullptr)
{
    namespace numeric = llaminar2::device_fp32_contract;
    const unsigned tid = threadIdx.x;
    const unsigned part = tid >> 6;          // 0..15 == 16 canonical K-partitions
    const unsigned lane = tid & 63u;
    const unsigned col_in_wave = lane >> 1;  // 0..31 (32 contiguous columns = 1024B!)
    const unsigned half_part = lane & 1u;    // 0: low 16B, 1: high 16B
    const unsigned blocks = width >> 5;
    constexpr unsigned parts = 16u;
    const unsigned step = (blocks + parts - 1u) / parts;
    const unsigned b_begin = part * step;
    const unsigned b_end = (b_begin + step < blocks) ? (b_begin + step) : blocks;
    const unsigned n_cols = static_cast<unsigned>(projection.weights.n);
    const unsigned col = col_base + col_in_wave;
    const bool valid_col = col < n_cols;
    const uint8_t* payload = projection.weights.payload;
    const unsigned short* scales = static_cast<const unsigned short*>(projection.weights.scales);
    using RawLdsInt4 = int __attribute__((ext_vector_type(4)));
    const auto* act_lds = (const PreAttnQ8Block48 __attribute__((opencl_local))*)(
        (const unsigned char __attribute__((opencl_local))*)activated);
    auto* partials_lds = (float __attribute__((opencl_local))*)(
        (unsigned char __attribute__((opencl_local))*)lds_partials);

    float result = 0.0f;
    if (valid_col && b_begin < b_end) {
        std::size_t elem = static_cast<std::size_t>(b_begin) * n_cols + col;
        int4 w_cur = native_global_weight_read<int4>(payload + elem * 32u + half_part * 16u);
        unsigned short sc_cur = (half_part == 0u)
            ? native_global_weight_read<unsigned short>(scales + elem) : 0u;
#pragma unroll 1
        for (unsigned b = b_begin; b < b_end; ++b) {
            int4 w_next{};
            unsigned short sc_next = 0u;
            if (b + 1u < b_end) {
                const std::size_t next_elem = elem + n_cols;
                w_next = native_global_weight_read<int4>(payload + next_elem * 32u + half_part * 16u);
                if (half_part == 0u)
                    sc_next = native_global_weight_read<unsigned short>(scales + next_elem);
            }
            const RawLdsInt4 aq = reinterpret_cast<const RawLdsInt4 __attribute__((opencl_local))*>(
                act_lds[b].qs)[half_part];
            int d = 0;
            d = llaminar2::rocm_integer_contract::signedDot4(aq[0], w_cur.x, d);
            d = llaminar2::rocm_integer_contract::signedDot4(aq[1], w_cur.y, d);
            d = llaminar2::rocm_integer_contract::signedDot4(aq[2], w_cur.z, d);
            d = llaminar2::rocm_integer_contract::signedDot4(aq[3], w_cur.w, d);
            const int full = d + __shfl_xor(d, 1, 64);
            if (half_part == 0u) {
                const float ws = __half2float(__ushort_as_half(sc_cur));
                result = numeric::add(result,
                    llaminar2::device_native_quantised_contract::singleScaleBlock(full, ws, act_lds[b].scale));
            }
            w_cur = w_next;
            sc_cur = sc_next;
            elem += n_cols;
        }
    }
    if (half_part == 0u)
        partials_lds[part * 32u + col_in_wave] = result;
    __syncthreads();
    if (tid < 32u && (col_base + tid) < n_cols) {
        float total = 0.0f;
#pragma unroll
        for (unsigned p = 0u; p < parts; ++p)
            total = numeric::add(total, partials_lds[p * 32u + tid]);
        const unsigned column = col_base + tid;
        projection.output[column] = total;
        if constexpr (InlineConv) if (do_conv) {
            const std::size_t s_base = static_cast<std::size_t>(column) * 3u;
            const std::size_t w_base = static_cast<std::size_t>(column) * 4u;
            float sum = 0.0f;
            sum += conv_state[s_base + 0u] * conv_weight[w_base + 0u];
            const float c1 = conv_state[s_base + 1u];
            sum += c1 * conv_weight[w_base + 1u];
            conv_state[s_base + 0u] = c1;
            const float c2 = conv_state[s_base + 2u];
            sum += c2 * conv_weight[w_base + 2u];
            conv_state[s_base + 1u] = c2;
            sum += total * conv_weight[w_base + 3u];
            conv_state[s_base + 2u] = total;
            if (conv_bias) sum += conv_bias[column];
            gdn_recurrence_in[column] = sum / (1.0f + expf(-sum));
        }
    }
}

/**
 * @brief 512-byte coalesced 16-column CTA tile Q8_0 projection (4 lanes/col, 2-block pipelined).
 *
 * Automatically selected over 32-column tiles whenever 16-column tiling strictly reduces the
 * worst-CTA column count for the admitted gridDim.x (e.g. 2560 cols on 56 CUs: 48 vs 64 cols/CTA).
 */
template<bool InlineConv = false>
__device__ __forceinline__ void completeProjectionTile16(
    const ProbeProjection& projection, unsigned col_base,
    const PreAttnQ8Block48* activated, unsigned width, float* lds_partials,
    bool do_conv = false,
    const float* conv_weight = nullptr, const float* conv_bias = nullptr,
    float* conv_state = nullptr, float* gdn_recurrence_in = nullptr)
{
    namespace numeric = llaminar2::device_fp32_contract;
    const unsigned tid = threadIdx.x;
    const unsigned part = tid >> 6;           // 0..15 == 16 canonical K-partitions
    const unsigned lane = tid & 63u;
    const unsigned col_in_wave = lane >> 2;   // 0..15 (16 contiguous columns = 512B/wave!)
    const unsigned quarter = lane & 3u;       // 0..3 (8B per lane -> 32B/col)
    const unsigned blocks = width >> 5;
    constexpr unsigned parts = 16u;
    const unsigned step = (blocks + parts - 1u) / parts;
    const unsigned b_begin = part * step;
    const unsigned b_end = (b_begin + step < blocks) ? (b_begin + step) : blocks;
    const unsigned n_cols = static_cast<unsigned>(projection.weights.n);
    const unsigned col = col_base + col_in_wave;
    const bool valid_col = col < n_cols;
    const uint8_t* payload = projection.weights.payload;
    const unsigned short* scales = static_cast<const unsigned short*>(projection.weights.scales);

    float result = 0.0f;
    if (valid_col && b_begin < b_end) {
        std::size_t elem = static_cast<std::size_t>(b_begin) * n_cols + col;
        int2 w_cur = native_global_weight_read<int2>(payload + elem * 32u + quarter * 8u);
        unsigned short sc_cur = (quarter == 0u)
            ? native_global_weight_read<unsigned short>(scales + elem) : 0u;
#pragma unroll 1
        for (unsigned b = b_begin; b < b_end; ++b) {
            int2 w_next{};
            unsigned short sc_next = 0u;
            if (b + 1u < b_end) {
                const std::size_t next_elem = elem + n_cols;
                w_next = native_global_weight_read<int2>(payload + next_elem * 32u + quarter * 8u);
                if (quarter == 0u)
                    sc_next = native_global_weight_read<unsigned short>(scales + next_elem);
            }
            const int2 aq = reinterpret_cast<const int2*>(activated[b].qs)[quarter];
            int d = llaminar2::rocm_integer_contract::signedDot4(aq.x, w_cur.x, 0);
            d = llaminar2::rocm_integer_contract::signedDot4(aq.y, w_cur.y, d);
            d += __shfl_xor(d, 1, 64);
            d += __shfl_xor(d, 2, 64);
            if (quarter == 0u) {
                const float ws = __half2float(__ushort_as_half(sc_cur));
                result = numeric::add(result,
                    llaminar2::device_native_quantised_contract::singleScaleBlock(d, ws, activated[b].scale));
            }
            w_cur = w_next;
            sc_cur = sc_next;
            elem += n_cols;
        }
    }
    if (quarter == 0u)
        lds_partials[part * 16u + col_in_wave] = result;
    __syncthreads();
    if (tid < 16u && (col_base + tid) < n_cols) {
        float total = 0.0f;
#pragma unroll
        for (unsigned p = 0u; p < parts; ++p)
            total = numeric::add(total, lds_partials[p * 16u + tid]);
        const unsigned column = col_base + tid;
        projection.output[column] = total;
        if constexpr (InlineConv) if (do_conv) {
            const std::size_t s_base = static_cast<std::size_t>(column) * 3u;
            const std::size_t w_base = static_cast<std::size_t>(column) * 4u;
            float sum = 0.0f;
            sum += conv_state[s_base + 0u] * conv_weight[w_base + 0u];
            const float c1 = conv_state[s_base + 1u];
            sum += c1 * conv_weight[w_base + 1u];
            conv_state[s_base + 0u] = c1;
            const float c2 = conv_state[s_base + 2u];
            sum += c2 * conv_weight[w_base + 2u];
            conv_state[s_base + 1u] = c2;
            sum += total * conv_weight[w_base + 3u];
            conv_state[s_base + 2u] = total;
            if (conv_bias) sum += conv_bias[column];
            gdn_recurrence_in[column] = sum / (1.0f + expf(-sum));
        }
    }
}

/** @brief Exact ordered four-segment quarter dot matching rocm_gdn_value_major_scalar_kernel. */
__device__ __forceinline__ float gdnOrderedQuarterDot8Lds(
    const float (&state)[8], const float* lds_vec, unsigned key_begin, int segment, int quarter)
{
    const auto* vec_lds = (const float __attribute__((opencl_local))*)(
        (const unsigned char __attribute__((opencl_local))*)lds_vec);
    float accumulator = 0.0f;
#pragma unroll 1
    for (int owner = 0; owner < 4; ++owner) {
        if (segment == owner) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float k_or_q = vec_lds[key_begin + static_cast<unsigned>(i)];
                asm volatile("v_fma_f32 %0, %1, %2, %0" : "+v"(accumulator) : "v"(state[i]), "v"(k_or_q));
            }
        }
        if (owner < 3)
            accumulator = __shfl(accumulator, quarter * 4 + owner, 16);
    }
    return accumulator;
}

/** @brief Exact ascending 0..3 quarter fold matching rocm_gdn_value_major_scalar_kernel. */
__device__ __forceinline__ float gdnOrderedQuarterFold(float quarter_value)
{
    float total = 0.0f;
#pragma unroll
    for (int quarter = 0; quarter < 4; ++quarter)
        total += __shfl(quarter_value, quarter * 4 + 3, 16);
    return total;
}

/** @brief Widen with the original floating GEMM's NaN spelling and exact finite conversion. */
__device__ __forceinline__ float originalHalfWiden(std::uint16_t half)
{
    const auto native=__float_as_uint(__half2float(__ushort_as_half(half)));
    const auto nan=(std::uint32_t(half&0x8000u)<<16)|0x7f800000u|
        (std::uint32_t(half&0x03ffu)<<13);
    const auto mask=0u-std::uint32_t((half&0x7fffu)>0x7c00u);
    return __uint_as_float(native^((native^nan)&mask));
}

/** @brief One wave executes the original 256 logical partials using LDS-cached FP16 mixed + 2-step pipelined weights. */
__device__ __forceinline__ void completeHalfOutput(
    const std::uint16_t* mixed_fp16_lds, const std::uint16_t* weight, float* out)
{
    const unsigned lane = threadIdx.x & 63u;
    float partial[4]{};
    // Issue all 4 weight loads of each step in parallel (4 x 128B = 512B/wave)
    // while reading pre-rounded FP16 activations directly from LDS (zero global mixed reads!).
#pragma unroll 1
    for (unsigned step = 0u; step < E / 256u; ++step) {
        std::uint16_t w_cur[4];
#pragma unroll
        for (unsigned group = 0u; group < 4u; ++group)
            w_cur[group] = native_global_weight_read<std::uint16_t>(
                weight + step * 256u + group * 64u + lane);
#pragma unroll
        for (unsigned group = 0u; group < 4u; ++group) {
            const unsigned index = step * 256u + group * 64u + lane;
            partial[group] += originalHalfWiden(mixed_fp16_lds[index]) * originalHalfWiden(w_cur[group]);
        }
    }
    float value = (partial[0] + partial[2]) + (partial[1] + partial[3]);
    for (unsigned stride = 32u; stride > 0u; stride >>= 1)
        value += __shfl_down(value, stride, 64);
    if (lane == 0u) *out = value;
}

/** @brief Phase-exclusive static LDS shared between HC prefix, early Q8 bank, recurrence Q/K, and GDN output Q8 bank. */
union PreAttentionShared
{
    llaminar2::rocm::FullNativeHCFFNPrefixScratch hc;
    PreAttnQ8Block48 activation[E / 32];
    struct {
        float q_head[128];
        float k_head[128];
        float scalars[4]; // [0]=q_inv/w0, [1]=k_inv/w1, [2]=decay, [3]=beta_h
        unsigned finished_chunks;
    } rec;
    PreAttnQ8Block48 out_q8[kMaxFullGDNOutBlocks];
};

/** @brief Execute exact ValueMajor M1 GDN recurrence + last-CTA inline gated RMSNorm + Q8 pack with 0 VGPR spills. */
__device__ __forceinline__ void executeFullGDNRecurrenceAndNorm(
    const FullGDNChain& full,
    const ProbeProjection& z,
    const HalfPointers& half,
    PreAttnQ8Block48* gdn_out_q8_global,
    std::uint32_t* head_done,
    PreAttentionShared& shared)
{
    const unsigned n_v = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(full.n_v_heads));
    const unsigned n_k = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(full.n_k_heads));
    const unsigned chunks_per_head = __builtin_amdgcn_readfirstlane(
        (n_v * 4u <= gridDim.x) ? 4u : (n_v * 2u <= gridDim.x) ? 2u : 1u);
    const unsigned cols_per_chunk = __builtin_amdgcn_readfirstlane(128u / chunks_per_head);
    const unsigned total_chunks = __builtin_amdgcn_readfirstlane(n_v * chunks_per_head);
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 63u;
    const unsigned key_span = __builtin_amdgcn_readfirstlane(n_k * 128u);
    const float* raw = full.gdn_recurrence_in;

#pragma unroll 1
    for (unsigned chunk_id = __builtin_amdgcn_readfirstlane(blockIdx.x);
         chunk_id < total_chunks; chunk_id += __builtin_amdgcn_readfirstlane(gridDim.x)) {
        const unsigned h = __builtin_amdgcn_readfirstlane(chunk_id / chunks_per_head);
        const unsigned c = __builtin_amdgcn_readfirstlane(chunk_id % chunks_per_head);
        const unsigned vi_begin = __builtin_amdgcn_readfirstlane(c * cols_per_chunk);
        int source_head = static_cast<int>(h) + full.global_v_offset;
        source_head %= static_cast<int>(n_k);
        if (source_head < 0) source_head += static_cast<int>(n_k);
        source_head = __builtin_amdgcn_readfirstlane(source_head);

        // Stage Q/K into LDS once per chunk with exact two-wave64 L2 norm matching
        // rocm_gdn_scalar_prepared_qk_conv_kernel (zero per-thread Q/K VGPR array!).
        if (tid < 128u) {
            const unsigned src_off = static_cast<unsigned>(source_head) * 128u + tid;
            const float qv = raw[src_off];
            const float kv = raw[key_span + src_off];
            shared.rec.q_head[tid] = qv;
            shared.rec.k_head[tid] = kv;
            if (full.use_qk_l2norm) {
#pragma clang fp contract(off)
                float q_sq = qv * qv;
                float k_sq = kv * kv;
                for (int offset = 32; offset > 0; offset >>= 1) {
                    q_sq += __shfl_xor(q_sq, offset);
                    k_sq += __shfl_xor(k_sq, offset);
                }
                if (lane == 0u) {
                    shared.rec.scalars[tid >> 6] = q_sq;
                    shared.rec.scalars[2u + (tid >> 6)] = k_sq;
                }
            }
        }
        __syncthreads();
        if (tid == 0u) {
            float q_inv = rsqrtf(128.0f);
            float k_inv = 1.0f;
            if (full.use_qk_l2norm) {
#pragma clang fp contract(off)
                float q_total = 0.0f, k_total = 0.0f;
                q_total += shared.rec.scalars[0]; q_total += shared.rec.scalars[1];
                q_total += 0.0f; q_total += 0.0f;
                k_total += shared.rec.scalars[2]; k_total += shared.rec.scalars[3];
                k_total += 0.0f; k_total += 0.0f;
                q_inv = rsqrtf(128.0f) / sqrtf(q_total + 1e-6f);
                k_inv = 1.0f / sqrtf(k_total + 1e-6f);
            }
            const float x = half.raw_alpha[h] + full.dt_bias[h];
            const float sp = x > 20.0f ? x : log1pf(expf(x));
            shared.rec.scalars[0] = q_inv;
            shared.rec.scalars[1] = k_inv;
            shared.rec.scalars[2] = expf(full.A_log[h] * sp);
            shared.rec.scalars[3] = 1.0f / (1.0f + expf(-half.raw_beta[h]));
        }
        __syncthreads();
        if (tid < 128u) {
            shared.rec.q_head[tid] *= shared.rec.scalars[0];
            if (full.use_qk_l2norm)
                shared.rec.k_head[tid] *= shared.rec.scalars[1];
        }
        __syncthreads();

        // 1024 threads / 16 lanes = 64 value columns per pass. Only state[8] lives in VGPRs!
#pragma unroll 1
        for (unsigned local_vi = tid >> 4; local_vi < cols_per_chunk; local_vi += (NT >> 4)) {
            const unsigned group_lane = tid & 15u;
            const unsigned key_begin = group_lane * 8u;
            const unsigned vi = vi_begin + local_vi;
            float state[8];
            {
                const float* col_in = full.recurrence_state + ((h * 128u + vi) * 128u + key_begin);
                const float decay = shared.rec.scalars[2];
                const float4 s0 = *reinterpret_cast<const float4*>(col_in);
                const float4 s1 = *reinterpret_cast<const float4*>(col_in + 4u);
                state[0] = s0.x * decay; state[1] = s0.y * decay;
                state[2] = s0.z * decay; state[3] = s0.w * decay;
                state[4] = s1.x * decay; state[5] = s1.y * decay;
                state[6] = s1.z * decay; state[7] = s1.w * decay;
            }
            const float kv = gdnOrderedQuarterFold(
                gdnOrderedQuarterDot8Lds(state, shared.rec.k_head, key_begin,
                    static_cast<int>(group_lane & 3u), static_cast<int>(group_lane >> 2)));
            float value = 0.0f;
            if (group_lane == 0u)
                value = raw[2u * key_span + h * 128u + vi];
            value = __shfl(value, 0, 16);
            const float delta = (value - kv) * shared.rec.scalars[3];
#pragma unroll
            for (unsigned i = 0; i < 8u; ++i) {
                const float kval = shared.rec.k_head[key_begin + i];
                asm volatile("v_fma_f32 %0, %1, %2, %0" : "+v"(state[i]) : "v"(kval), "v"(delta));
            }
            const float out = gdnOrderedQuarterFold(
                gdnOrderedQuarterDot8Lds(state, shared.rec.q_head, key_begin,
                    static_cast<int>(group_lane & 3u), static_cast<int>(group_lane >> 2)));
            if (group_lane == 0u)
                full.attn_output[h * 128u + vi] = out;
            float* col_out = full.recurrence_state + ((h * 128u + vi) * 128u + key_begin);
            *reinterpret_cast<float4*>(col_out) = make_float4(state[0], state[1], state[2], state[3]);
            *reinterpret_cast<float4*>(col_out + 4u) = make_float4(state[4], state[5], state[6], state[7]);
        }

        // Release ALL waves' attn_output VMEM stores to L2 BEFORE tid==0 signals
        // head_done[h]; s_barrier alone does not wait for per-wave vmcnt(0) on gfx906.
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        __syncthreads();
        if (chunks_per_head == 1u) {
            if (tid == 0u) shared.rec.finished_chunks = 1u;
        } else {
            if (tid == 0u) {
                const unsigned done = __hip_atomic_fetch_add(
                    head_done + h, 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT) + 1u;
                if (done == chunks_per_head)
                    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
                shared.rec.finished_chunks = done;
            }
        }
        __syncthreads();

        if (shared.rec.finished_chunks == chunks_per_head) {
            const unsigned head_off = h * 128u;
            float val = 0.0f;
            if (tid < 128u) {
                val = full.attn_output[head_off + tid];
                float sq = llaminar2::device_fp32_contract::multiply(val, val);
#pragma unroll
                for (int off = 32; off > 0; off >>= 1)
                    sq = llaminar2::device_fp32_contract::add(sq, __shfl_xor(sq, off));
                if (lane == 0u) shared.rec.scalars[tid >> 6] = sq;
            }
            __syncthreads();
            if (tid < 128u) {
                const float total = shared.rec.scalars[0] + shared.rec.scalars[1];
                // Match rocm_gated_norm_single_q8_production_body's non-FMA
                // runtime division + eps add (no v_fma_f32 contraction).
                const float inv_rms = rsqrtf(
                    llaminar2::device_fp32_contract::add(total * (1.0f / 128.0f), full.ssm_norm_eps));
                const float gamma_eff = full.ssm_norm_subtract_one
                    ? (1.0f + full.ssm_norm_gamma[tid]) : full.ssm_norm_gamma[tid];
                const float norm_val = (val * inv_rms) * gamma_eff;
                const float gate_val = z.output[head_off + tid];
                float gate_act = gate_val;
                if (full.ssm_norm_activation != llaminar2::GatedNormActivation::Identity) {
                    const float denom = 1.0f + expf(-gate_val);
                    const float num = (full.ssm_norm_activation == llaminar2::GatedNormActivation::SiLU)
                        ? gate_val : 1.0f;
                    gate_act = num / denom;
                }
                const float out_val = norm_val * gate_act;
                full.attn_output[head_off + tid] = out_val;

                // Exact width-32 Q8 quantization matching rocm_gdn_gated_norm_single_q8_kernel.
                float maximum = fabsf(out_val);
#pragma unroll
                for (int shift = 16; shift > 0; shift >>= 1)
                    maximum = fmaxf(maximum, __shfl_down(maximum, shift, 32));
                maximum = __shfl(maximum, 0, 32);
                const float scale = llaminar2::rocm::decode_equivalent::q8Scale(maximum);
                const float inverse = 1.0f / scale;
                const int q = llaminar2::rocm::decode_equivalent::quantizeQ8(out_val, inverse);
                const unsigned sub = tid & 31u;
                const unsigned blk = tid >> 5;
                const int base_sub = static_cast<int>(sub & ~3u);
                std::uint32_t packed = static_cast<std::uint8_t>(q);
                packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(__shfl(q, base_sub + 1, 32))) << 8;
                packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(__shfl(q, base_sub + 2, 32))) << 16;
                packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(__shfl(q, base_sub + 3, 32))) << 24;
                auto& dst = gdn_out_q8_global[h * 4u + blk];
                if ((sub & 3u) == 0u) {
                    dst.qs[sub >> 2] = __builtin_bit_cast(std::int32_t, packed);
                }
                if (sub == 0u) {
                    dst.scale = scale;
                }
            }
            __syncthreads();
        }
    }
}

/** @brief Exact all-18-codebook K32 block dot contribution for general ROCm GDN projections. */
template<std::uint8_t CB>
__device__ __forceinline__ float generalGDNBlockContribution(
    const llaminar2::NativeEmbeddingIQTables& iq_tables,
    const llaminar2::DeviceNativeQuantisedMatrixDesc& desc,
    std::size_t linear,
    const PreAttnQ8Block48 __attribute__((opencl_local))* act)
{
    llaminar2::NativeEmbeddingIQTables single_iq{};
    if constexpr (CB == 11u) single_iq.iq3s = iq_tables.iq3s;
    else if constexpr (CB == 12u) single_iq.iq3xxs = iq_tables.iq3xxs;
    else if constexpr (CB == 13u) single_iq.iq2s = iq_tables.iq2s;
    else if constexpr (CB == 14u) single_iq.iq2xs = iq_tables.iq2xs;
    else if constexpr (CB == 15u) single_iq.iq2xxs = iq_tables.iq2xxs;
    else if constexpr (CB == 16u || CB == 17u) single_iq.iq1s = iq_tables.iq1s;
    llaminar2::rocm_q8_residual::NativeDecoder decoder(single_iq);
    using Traits = llaminar2::rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
    constexpr unsigned bytes = static_cast<unsigned>(Traits::payload_bytes);
    using RawLdsInt4 = int __attribute__((ext_vector_type(4)));
    const auto* a128 = reinterpret_cast<const RawLdsInt4 __attribute__((opencl_local))*>(act->qs);
    constexpr std::int32_t ones = 0x01010101;

    // Fast low-register direct path for 32-byte raw int8 codebooks (Q8_0 and ExpandedInt8Min)
    // avoids materializing both pay_buf[32] and w[8] simultaneously (saves 12+ VGPRs).
    if constexpr (CB == 19u || CB == llaminar2::kNativeQuantisedExpandedInt8MinCodebook) {
        const std::uint8_t* pay_src = desc.payload + linear * 32u;
        std::int32_t dot = 0, sum_a = 0;
        {
            const int4 w0 = native_global_weight_read<int4>(pay_src);
            const RawLdsInt4 a0 = a128[0];
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[0], w0.x, dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[1], w0.y, dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[2], w0.z, dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[3], w0.w, dot);
            if constexpr (Traits::is_asymmetric) {
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[0], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[1], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[2], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[3], ones, sum_a);
            }
        }
        {
            const int4 w1 = native_global_weight_read<int4>(pay_src + 16u);
            const RawLdsInt4 a1 = a128[1];
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[0], w1.x, dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[1], w1.y, dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[2], w1.z, dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[3], w1.w, dot);
            if constexpr (Traits::is_asymmetric) {
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[0], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[1], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[2], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[3], ones, sum_a);
            }
        }
        const std::uint16_t primary = native_global_weight_read<std::uint16_t>(
            static_cast<const std::uint16_t*>(desc.scales) + linear);
        const std::uint16_t secondary = Traits::requires_secondary_metadata
            ? native_global_weight_read<std::uint16_t>(static_cast<const std::uint16_t*>(desc.mins) + linear) : 0u;
        return decoder.template native_quantised_block_contribution_from_reduced_terms_rn<CB>(
            dot, 0, act->scale, primary, secondary, 0u, sum_a, 0, 0, 0u, 0, 0, 0, 0);
    }

    std::int32_t w[8];
    const std::uint8_t* pay_src = desc.payload + linear * bytes;
    decoder.template decode_groups<CB>(pay_src, w);
    const std::uint16_t iq1m_qh = Traits::is_iq1_m
        ? static_cast<std::uint16_t>(
              static_cast<std::uint16_t>(native_global_weight_read<std::uint8_t>(pay_src + 4u)) |
              (static_cast<std::uint16_t>(native_global_weight_read<std::uint8_t>(pay_src + 5u)) << 8u))
        : 0u;
    if constexpr (Traits::is_dual_scale) {
        std::int32_t dot_lo = 0, dot_hi = 0;
        std::int32_t sg0 = 0, sg1 = 0, sg2 = 0, sg3 = 0;
        {
            const RawLdsInt4 a0 = a128[0];
            dot_lo = llaminar2::rocm_integer_contract::signedDot4(a0[0], w[0], dot_lo);
            dot_lo = llaminar2::rocm_integer_contract::signedDot4(a0[1], w[1], dot_lo);
            dot_lo = llaminar2::rocm_integer_contract::signedDot4(a0[2], w[2], dot_lo);
            dot_lo = llaminar2::rocm_integer_contract::signedDot4(a0[3], w[3], dot_lo);
            sg0 = llaminar2::rocm_integer_contract::signedDot4(a0[1], ones,
                llaminar2::rocm_integer_contract::signedDot4(a0[0], ones, 0));
            sg1 = llaminar2::rocm_integer_contract::signedDot4(a0[3], ones,
                llaminar2::rocm_integer_contract::signedDot4(a0[2], ones, 0));
        }
        {
            const RawLdsInt4 a1 = a128[1];
            dot_hi = llaminar2::rocm_integer_contract::signedDot4(a1[0], w[4], dot_hi);
            dot_hi = llaminar2::rocm_integer_contract::signedDot4(a1[1], w[5], dot_hi);
            dot_hi = llaminar2::rocm_integer_contract::signedDot4(a1[2], w[6], dot_hi);
            dot_hi = llaminar2::rocm_integer_contract::signedDot4(a1[3], w[7], dot_hi);
            sg2 = llaminar2::rocm_integer_contract::signedDot4(a1[1], ones,
                llaminar2::rocm_integer_contract::signedDot4(a1[0], ones, 0));
            sg3 = llaminar2::rocm_integer_contract::signedDot4(a1[3], ones,
                llaminar2::rocm_integer_contract::signedDot4(a1[2], ones, 0));
        }
        const std::int32_t sum_lo = sg0 + sg1;
        const std::int32_t sum_hi = sg2 + sg3;
        const std::uint16_t primary = native_global_weight_read<std::uint16_t>(
            static_cast<const std::uint16_t*>(desc.scales) + linear);
        const std::uint16_t secondary = Traits::requires_secondary_metadata
            ? native_global_weight_read<std::uint16_t>(static_cast<const std::uint16_t*>(desc.mins) + linear) : 0u;
        const std::uint32_t emin = Traits::is_dual_scale_asym
            ? native_global_weight_read<std::uint32_t>(static_cast<const std::uint32_t*>(desc.emins) + linear) : 0u;
        return decoder.template native_quantised_block_contribution_from_reduced_terms_rn<CB>(
            dot_lo, dot_hi, act->scale, primary, secondary, emin,
            sum_lo + sum_hi, sum_lo, sum_hi, iq1m_qh, sg0, sg1, sg2, sg3);
    } else {
        std::int32_t dot = 0, sum_a = 0;
        {
            const RawLdsInt4 a0 = a128[0];
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[0], w[0], dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[1], w[1], dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[2], w[2], dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a0[3], w[3], dot);
            if constexpr (Traits::is_asymmetric) {
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[0], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[1], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[2], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a0[3], ones, sum_a);
            }
        }
        {
            const RawLdsInt4 a1 = a128[1];
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[0], w[4], dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[1], w[5], dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[2], w[6], dot);
            dot = llaminar2::rocm_integer_contract::signedDot4(a1[3], w[7], dot);
            if constexpr (Traits::is_asymmetric) {
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[0], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[1], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[2], ones, sum_a);
                sum_a = llaminar2::rocm_integer_contract::signedDot4(a1[3], ones, sum_a);
            }
        }
        const std::uint16_t primary = native_global_weight_read<std::uint16_t>(
            static_cast<const std::uint16_t*>(desc.scales) + linear);
        const std::uint16_t secondary = Traits::requires_secondary_metadata
            ? native_global_weight_read<std::uint16_t>(static_cast<const std::uint16_t*>(desc.mins) + linear) : 0u;
        return decoder.template native_quantised_block_contribution_from_reduced_terms_rn<CB>(
            dot, 0, act->scale, primary, secondary, 0u, sum_a, 0, 0, 0u, 0, 0, 0, 0);
    }
}

/** @brief Runtime-dispatched all-18-codebook 32-column projection tile for non-Q8_0 GDN weights. */
template<bool InlineConv = false>
__device__ __forceinline__ void completeGeneralProjectionTile32(
    const llaminar2::NativeEmbeddingIQTables& iq_tables,
    const ProbeProjection& projection, unsigned col_base,
    const PreAttnQ8Block48* activated, unsigned width, float* lds_partials,
    bool do_conv = false,
    const float* conv_weight = nullptr, const float* conv_bias = nullptr,
    float* conv_state = nullptr, float* gdn_recurrence_in = nullptr)
{
    namespace numeric = llaminar2::device_fp32_contract;
    const unsigned tid = threadIdx.x;
    const unsigned part = __builtin_amdgcn_readfirstlane(tid >> 6);
    const unsigned lane = tid & 63u;
    const unsigned blocks = __builtin_amdgcn_readfirstlane(width >> 5);
    constexpr unsigned parts = 16u;
    const unsigned step = __builtin_amdgcn_readfirstlane((blocks + parts - 1u) / parts);
    const unsigned b_begin = __builtin_amdgcn_readfirstlane(part * step);
    const unsigned b_end = __builtin_amdgcn_readfirstlane((b_begin + step < blocks) ? (b_begin + step) : blocks);
    const unsigned n_cols = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(projection.weights.n));
    const unsigned col = __builtin_amdgcn_readfirstlane(col_base) + lane;
    const std::uint8_t cb = static_cast<std::uint8_t>(__builtin_amdgcn_readfirstlane(projection.weights.codebook_id));
    const auto* act_lds = (const PreAttnQ8Block48 __attribute__((opencl_local))*)(
        (const unsigned char __attribute__((opencl_local))*)activated);

    float result = 0.0f;
    if (b_begin < b_end && lane < 32u && col < n_cols) {
#pragma unroll 1
        for (unsigned b = b_begin; b < b_end; ++b) {
            const std::size_t elem = static_cast<std::size_t>(b) * n_cols + col;
            float contrib = 0.0f;
            switch (cb) {
            case 0u:  contrib = generalGDNBlockContribution<0u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 4u:  contrib = generalGDNBlockContribution<4u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 5u:  contrib = generalGDNBlockContribution<5u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 6u:  contrib = generalGDNBlockContribution<6u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 7u:  contrib = generalGDNBlockContribution<7u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 8u:  contrib = generalGDNBlockContribution<8u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 9u:  contrib = generalGDNBlockContribution<9u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 10u: contrib = generalGDNBlockContribution<10u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 11u: contrib = generalGDNBlockContribution<11u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 12u: contrib = generalGDNBlockContribution<12u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 13u: contrib = generalGDNBlockContribution<13u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 14u: contrib = generalGDNBlockContribution<14u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 15u: contrib = generalGDNBlockContribution<15u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 16u: contrib = generalGDNBlockContribution<16u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 17u: contrib = generalGDNBlockContribution<17u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case 19u: contrib = generalGDNBlockContribution<19u>(iq_tables, projection.weights, elem, act_lds + b); break;
            case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
                contrib = generalGDNBlockContribution<llaminar2::kNativeQuantisedExpandedInt8MinCodebook>(
                    iq_tables, projection.weights, elem, act_lds + b);
                break;
            case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
                contrib = generalGDNBlockContribution<llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook>(
                    iq_tables, projection.weights, elem, act_lds + b);
                break;
            default: __builtin_trap();
            }
            result = numeric::add(result, contrib);
        }
    }
    if (lane < 32u)
        lds_partials[part * 32u + lane] = result;
    __syncthreads();
    if (tid < 32u && (col_base + tid) < n_cols) {
        float total = 0.0f;
#pragma unroll
        for (unsigned p = 0u; p < parts; ++p)
            total = numeric::add(total, lds_partials[p * 32u + tid]);
        const unsigned column = col_base + tid;
        projection.output[column] = total;
        if constexpr (InlineConv) if (do_conv) {
            const std::size_t s_base = static_cast<std::size_t>(column) * 3u;
            const std::size_t w_base = static_cast<std::size_t>(column) * 4u;
            float sum = 0.0f;
            sum += conv_state[s_base + 0u] * conv_weight[w_base + 0u];
            const float c1 = conv_state[s_base + 1u];
            sum += c1 * conv_weight[w_base + 1u];
            conv_state[s_base + 0u] = c1;
            const float c2 = conv_state[s_base + 2u];
            sum += c2 * conv_weight[w_base + 2u];
            conv_state[s_base + 1u] = c2;
            sum += total * conv_weight[w_base + 3u];
            conv_state[s_base + 2u] = total;
            if (conv_bias) sum += conv_bias[column];
            gdn_recurrence_in[column] = sum / (1.0f + expf(-sum));
        }
    }
}

/** @brief General CU-count resident POST/HC prefix and proven local Q8 + FP16 GDN megakernel. */
template<bool FusedHalf, bool FullGDN = false, bool GeneralCodebooks = false>
__global__ __launch_bounds__(NT, 1) void preAttentionKernel(
    llaminar2::rocm::FullNativeHCFFNPrefixArgs prefix,
    unsigned prefix_bytes,
    ProbeProjection qkv,
    ProbeProjection z,
    HalfPointers half,
    FullGDNChain full,
    llaminar2::NativeEmbeddingIQTables iq_tables)
{
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
    __shared__ PreAttentionShared shared;
    extern __shared__ __align__(16) unsigned char dynamic_storage[];
    ScalarResidentGridBarrierFor<llaminar2::FullNativeHCFFNBarrierLayout::prefix_phases> grid{
        prefix.barrier_state};
    llaminar2::rocm::fullNativeHCFFNPrefix(prefix, grid, shared.hc, dynamic_storage, prefix_bytes);

    ScalarResidentGridBarrierFor<2> gdn_grid{
        prefix.barrier_state + llaminar2::FullNativeHCFFNBarrierLayout::suffix_word_offset};

    const bool live = llaminar2::rocm::router_native::runtime_decode_row_active(
        prefix.hc.norm.extent.liveRows);
    if (!live) {
        if constexpr (FullGDN) {
            gdn_grid.sync();
            gdn_grid.sync();
        }
        return;
    }

    if constexpr (FullGDN) {
        auto* head_done = reinterpret_cast<std::uint32_t*>(
            reinterpret_cast<unsigned char*>(prefix.barrier_state) + kFullGDNDoneOffset);
        if (blockIdx.x == 0u && threadIdx.x < static_cast<unsigned>(full.n_v_heads))
            head_done[threadIdx.x] = 0u;
    }

    {
        const float* mixed = prefix.hc.mixed;
        float* tile_lds = reinterpret_cast<float*>(dynamic_storage);
        auto* mixed_fp16_lds = reinterpret_cast<std::uint16_t*>(tile_lds + 1024u); // +4 KB offset, 5120 B
        // Exact native v4 K32 quantization + pre-rounded FP16 mixed cache in LDS for Alpha/Beta.
        for (unsigned physical = threadIdx.x; physical < E; physical += NT) {
            const unsigned block = physical / 32u, lane = physical % 32u;
            const float mv = mixed[physical];
            if constexpr (FusedHalf)
                mixed_fp16_lds[physical] = __half_as_ushort(__float2half_rn(mv));
            auto& q = shared.activation[block];
            llaminar2::original_single_q8::publish(mv, true,
                reinterpret_cast<std::int8_t*>(q.qs), &q.scale, nullptr, lane);
        }
        __syncthreads();
        const unsigned qkv_tiles32 = (static_cast<unsigned>(qkv.weights.n) + 31u) >> 5;
        const unsigned z_tiles32 = (static_cast<unsigned>(z.weights.n) + 31u) >> 5;
        const unsigned total_tiles32 = qkv_tiles32 + z_tiles32;
        unsigned total_tiles = total_tiles32;
        unsigned buf_idx = 0u;
        if constexpr (!GeneralCodebooks) {
#pragma unroll 1
            for (unsigned tile = blockIdx.x; tile < total_tiles32; tile += gridDim.x) {
                const bool is_qkv = tile < qkv_tiles32;
                const unsigned col_base = (is_qkv ? tile : (tile - qkv_tiles32)) << 5;
                completeProjectionTile32<FullGDN>(
                    is_qkv ? qkv : z, col_base, shared.activation, E,
                    tile_lds + (buf_idx & 1u) * 512u, is_qkv,
                    full.conv_weight, full.conv_bias, full.conv_state, full.gdn_recurrence_in);
                ++buf_idx;
            }
        } else {
#pragma unroll 1
            for (unsigned tile = blockIdx.x; tile < total_tiles32; tile += gridDim.x) {
                const bool is_qkv = tile < qkv_tiles32;
                const unsigned col_base = (is_qkv ? tile : (tile - qkv_tiles32)) << 5;
                completeGeneralProjectionTile32<FullGDN>(
                    iq_tables, is_qkv ? qkv : z, col_base, shared.activation, E,
                    tile_lds + (buf_idx & 1u) * 512u, is_qkv,
                    full.conv_weight, full.conv_bias, full.conv_state, full.gdn_recurrence_in);
                ++buf_idx;
            }
        }
        if constexpr (FusedHalf) {
            // Schedule ALL Alpha/Beta FP16 rows exclusively onto the tail CTAs that received
            // 1 fewer QKV/Z tile (40 tail CTAs on nv=12, 24 on nv=24) so they run in the tail shadow!
            const unsigned half_tasks = 2u * half.heads;
            const unsigned tail_shift = total_tiles % gridDim.x;
            const unsigned shifted_bid = (blockIdx.x + gridDim.x - tail_shift) % gridDim.x;
            const unsigned tail_ctas = (tail_shift > 0u) ? (gridDim.x - tail_shift) : gridDim.x;
            if (shifted_bid < tail_ctas) {
                const unsigned h_begin = shifted_bid * half_tasks / tail_ctas;
                const unsigned h_end = (shifted_bid + 1u) * half_tasks / tail_ctas;
#pragma unroll 1
                for (unsigned h = h_begin + (threadIdx.x >> 6); h < h_end; h += NT / 64u) {
                    const bool beta = h >= half.heads;
                    const unsigned row = beta ? (h - half.heads) : h;
                    const auto* matrix = beta ? half.beta : half.alpha;
                    auto* output = beta ? half.raw_beta : half.raw_alpha;
                    completeHalfOutput(mixed_fp16_lds, matrix + static_cast<std::size_t>(row) * E, output + row);
                }
            }
        }
    }

    if constexpr (FullGDN) {
        auto* scratch_bytes = reinterpret_cast<unsigned char*>(prefix.barrier_state);
        auto* gdn_out_q8_global = reinterpret_cast<PreAttnQ8Block48*>(scratch_bytes + kFullGDNQ8Offset);
        auto* head_done = reinterpret_cast<std::uint32_t*>(scratch_bytes + kFullGDNDoneOffset);
        // Barrier A: all QKV+short_conv1d, Z, Alpha, Beta, and head_done resets visible.
        gdn_grid.sync();
        executeFullGDNRecurrenceAndNorm(
            full, z, half, gdn_out_q8_global, head_done, shared);
        // Barrier B: all n_v_heads*4 Q8 blocks of gated-normalized output visible.
        gdn_grid.sync();

        const unsigned value_dim = static_cast<unsigned>(full.n_v_heads) * 128u;
        const unsigned v_blocks = value_dim / 32u;
        const auto* src_vec4 = reinterpret_cast<const int4*>(gdn_out_q8_global);
        auto* dst_vec4 = reinterpret_cast<int4*>(shared.out_q8);
        const unsigned total_vec4 = v_blocks * (sizeof(PreAttnQ8Block48) / sizeof(int4));
        for (unsigned w = threadIdx.x; w < total_vec4; w += NT)
            dst_vec4[w] = src_vec4[w];
        __syncthreads();

        float* ssm_lds = reinterpret_cast<float*>(dynamic_storage);
        const unsigned ssm_tiles32 = (static_cast<unsigned>(full.ssm_out.weights.n) + 31u) >> 5;
        unsigned ssm_buf = 0u;
        if constexpr (!GeneralCodebooks) {
#pragma unroll 1
            for (unsigned tile = blockIdx.x; tile < ssm_tiles32; tile += gridDim.x) {
                completeProjectionTile32<false>(
                    full.ssm_out, tile << 5, shared.out_q8, value_dim,
                    ssm_lds + (ssm_buf & 1u) * 512u);
                ++ssm_buf;
            }
        } else {
#pragma unroll 1
            for (unsigned tile = blockIdx.x; tile < ssm_tiles32; tile += gridDim.x) {
                completeGeneralProjectionTile32<false>(
                    iq_tables, full.ssm_out, tile << 5, shared.out_q8, value_dim,
                    ssm_lds + (ssm_buf & 1u) * 512u);
                ++ssm_buf;
            }
        }
    }
#else
    __builtin_trap();
#endif
}
} // namespace qfn_retained_m1_hc_ffn_device

namespace llaminar2::rocm::pre_attention_probe {
using namespace qfn_retained_m1_hc_ffn_device;
std::size_t scratchBytes() noexcept
{
    return kFullGDNScratchBytes;
}

/** @brief Check whether codebook_id is one of the 18 compiled NativeQuantised decoders. */
static bool supportedNativeCodebook(std::uint8_t cb) noexcept
{
    switch (cb) {
    case 0u: case 4u: case 5u: case 6u: case 7u: case 8u: case 9u: case 10u:
    case 11u: case 12u: case 13u: case 14u: case 15u: case 16u: case 17u: case 19u:
    case llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
    case llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
        return true;
    default:
        return false;
    }
}

/** @brief Check the original native, block-major Q8 descriptor and immutable shape. */
static bool originalQ8(const DeviceNativeQuantisedMatrixDesc& weight,int n,int k) noexcept
{
    return weight.valid()&&weight.n==n&&weight.k==k&&weight.blocks_per_row==unsigned(k/32)&&
        weight.codebook_id==19&&weight.source_identity_present&&weight.source_codebook_id==19&&
        !weight.source_is_superblock&&!weight.mins&&!weight.emins;
}

/** @brief Check any valid NativeQuantised descriptor across all 18 compiled codebooks. */
static bool validNativeMatrix(const DeviceNativeQuantisedMatrixDesc& weight, int n, int k) noexcept
{
    return weight.valid() && weight.n == n && weight.k == k &&
        weight.blocks_per_row == unsigned(k / 32) &&
        weight.source_identity_present && supportedNativeCodebook(weight.codebook_id);
}

/** @brief Select one exact wave64 function from the validated immutable capture policy. */
static const void* selectFunction(HalfGDNPolicy policy, bool full_gdn, bool general_codebooks) noexcept
{
    if (full_gdn) {
        if (policy == HalfGDNPolicy::RetainedPreparedFP16RNE)
            return general_codebooks
                ? reinterpret_cast<const void*>(&preAttentionKernel<true, true, true>)
                : reinterpret_cast<const void*>(&preAttentionKernel<true, true, false>);
        return nullptr;
    }
    switch(policy) {
    case HalfGDNPolicy::SeparateOriginalSubgroup:
        return general_codebooks
            ? reinterpret_cast<const void*>(&preAttentionKernel<false, false, true>)
            : reinterpret_cast<const void*>(&preAttentionKernel<false, false, false>);
    case HalfGDNPolicy::RetainedPreparedFP16RNE:
        return general_codebooks
            ? reinterpret_cast<const void*>(&preAttentionKernel<true, false, true>)
            : reinterpret_cast<const void*>(&preAttentionKernel<true, false, false>);
    }
    return nullptr;
}

/** @brief Verify one complete prepared half matrix with the original GDN role extent. */
static bool validHalfMatrix(const PreparedHalfMatrix& descriptor, int heads) noexcept
{
    return heads > 0 && descriptor.words &&
        reinterpret_cast<std::uintptr_t>(descriptor.words)%alignof(std::uint16_t)==0 &&
        descriptor.bytes==std::size_t(heads)*E*sizeof(std::uint16_t);
}

hipError_t prepare(const Invocation& invocation,int device,Admission& admission)
{
    Invocation resolved = invocation;
    auto& prefix = resolved.prefix;
    auto& hc = prefix.hc;
    if(!resolved.scratch||resolved.scratch_bytes!=scratchBytes()||
       reinterpret_cast<std::uintptr_t>(resolved.scratch)%16||hc.norm.hidden!=E||
       !hc.norm.gain||!hc.norm.output||!hc.mixed||!hc.raw_down||!hc.down_partials||
       !hc.norm.input||(bool(hc.injection.weights)!=bool(hc.injection.output))||
       !originalQ8(hc.down_weights,LR,K)||!originalQ8(hc.up.weights,K,LR)||
       hc.norm.extent.rows!=1||hc.up.rows.physicalRows()!=1||hc.up.rows.capacity()!=1||
       hc.norm.extent.liveRows!=hc.up.rows.countOwner()||
       hc.up.quantizer!=Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal||
       prefix.output_demand.bits())
        return hipErrorInvalidValue;
    const int n_qkv=resolved.projections[0].weights.n;
    const int n_z=resolved.projections[1].weights.n;
    if(resolved.geometry!=EarlyQ8Geometry::GDN||
       n_qkv<=0||(n_qkv%4)!=0||n_z<=0||(n_z%4)!=0||
       !resolved.projections[0].output||!validNativeMatrix(resolved.projections[0].weights,n_qkv,E)||
       !resolved.projections[1].output||!validNativeMatrix(resolved.projections[1].weights,n_z,E)||
       resolved.projections[2].output||resolved.projections[2].weights.payload)
        return hipErrorInvalidValue;

    const auto& half=resolved.half;
    if(resolved.half_policy==HalfGDNPolicy::RetainedPreparedFP16RNE) {
        if(!validHalfMatrix(half.alpha,half.heads)||!validHalfMatrix(half.beta,half.heads)||
           !half.raw_alpha||!half.raw_beta||half.raw_alpha==half.raw_beta)
            return hipErrorInvalidValue;
    } else if(half.alpha.words||half.alpha.bytes||half.beta.words||half.beta.bytes||half.raw_alpha||half.raw_beta) {
        return hipErrorInvalidValue;
    }

    switch(resolved.trunk_source) {
    case TrunkSource::PreviousWrite:
        if(!prefix.write.residual||!prefix.write.block||!prefix.write.gateLogits||
           !prefix.write.output||hc.norm.input!=prefix.write.output)
            return hipErrorInvalidValue;
        break;
    case TrunkSource::PublishedTrunk:
        if(prefix.write.residual||prefix.write.block||prefix.write.gateLogits||prefix.write.output)
            return hipErrorInvalidValue;
        break;
    default:
        return hipErrorInvalidValue;
    }

    if (resolved.full_gdn.enabled) {
        const auto& fg = resolved.full_gdn;
        const int expected_qkv = 2 * fg.n_k_heads * 128 + fg.n_v_heads * 128;
        const int expected_z = fg.n_v_heads * 128;
        if (resolved.half_policy != HalfGDNPolicy::RetainedPreparedFP16RNE ||
            fg.n_k_heads <= 0 || fg.n_v_heads < 4 ||
            static_cast<unsigned>(fg.n_v_heads) > kMaxFullGDNVHeads ||
            fg.n_v_heads != half.heads ||
            n_qkv != expected_qkv || (n_qkv & 7) != 0 ||
            n_z != expected_z || (n_z & 7) != 0 ||
            !fg.conv_weight || !fg.conv_state || !fg.gdn_recurrence_in ||
            !fg.A_log || !fg.dt_bias || !fg.recurrence_state || !fg.attn_output ||
            !fg.ssm_norm_gamma || !std::isfinite(fg.ssm_norm_eps) || fg.ssm_norm_eps <= 0.0f ||
            !llaminar2::validGatedNormActivation(fg.ssm_norm_activation) ||
            !fg.ssm_out.output || fg.ssm_out.weights.n <= 0 || (fg.ssm_out.weights.n & 7) != 0 ||
            !validNativeMatrix(fg.ssm_out.weights, fg.ssm_out.weights.n, expected_z) ||
            reinterpret_cast<std::uintptr_t>(fg.recurrence_state) % 16 != 0 ||
            reinterpret_cast<std::uintptr_t>(fg.gdn_recurrence_in) % 16 != 0)
            return hipErrorInvalidValue;
    }

    const bool all_q8 = originalQ8(resolved.projections[0].weights, n_qkv, E) &&
                        originalQ8(resolved.projections[1].weights, n_z, E) &&
                        (!resolved.full_gdn.enabled ||
                         originalQ8(resolved.full_gdn.ssm_out.weights, resolved.full_gdn.ssm_out.weights.n, n_z));
    const bool general_codebooks = !all_q8;
    const void* function=selectFunction(resolved.half_policy, resolved.full_gdn.enabled, general_codebooks);
    if(!function)return hipErrorInvalidValue;
    NativeEmbeddingIQTables iq_tables{};
    if (general_codebooks && !rocmNativeEmbeddingIQTableView(device, &iq_tables))
        return hipErrorInvalidDevice;
    int current=-1;
    auto error=hipGetDevice(&current);
    if(error!=hipSuccess)return error;
    if(current!=device)return hipErrorInvalidDevice;
    const auto& hardware=rocmHardwareMetadata().at(device);
    if(hardware.compute_units<=0 || hardware.native_wave_width!=64 ||
       hardware.lds_per_block<=sizeof(PreAttentionShared))
        return hipErrorInvalidDevice;

    try {
        resolvePersistentHCReadPolicy(prefix.hc, prefix.output_demand);
    } catch (...) {
        return hipErrorInvalidValue;
    }
    prefix.barrier_state = reinterpret_cast<std::uint32_t*>(resolved.scratch);
    const std::size_t max_dynamic_bytes = fullNativeHCFFNPrefixDynamicBytes(prefix, 1);
    if(!max_dynamic_bytes || max_dynamic_bytes > 32768u || (max_dynamic_bytes & 15u) ||
       max_dynamic_bytes + sizeof(PreAttentionShared) > hardware.lds_per_block)
        return hipErrorInvalidConfiguration;

    // Declaratively reserve up to 4 CUs so co-resident overlay repack and
    // maintenance kernels never contend with or deadlock our ResidentGridBarrier.
    const unsigned cus = static_cast<unsigned>(hardware.compute_units);
    const auto co_residency = llaminar2::NativeCoResidencyReservation::reserveProcessors(
        cus > 4u ? 4u : 0u);
    std::optional<llaminar2::NativeCooperativeKernelAdmission> native;
    try {
        native.emplace(llaminar2::NativeCooperativeKernelBoundary<llaminar2::NativeCooperativeBackend::ROCm>::prepare(
            llaminar2::DeviceId::rocm(device), function, NT, max_dynamic_bytes, 1u,
            llaminar2::NativeResidentLaunchPolicy::Ordinary, co_residency));
    } catch (...) {
        return hipErrorInvalidConfiguration;
    }
    hipFuncAttributes attributes{};
    error = hipFuncGetAttributes(&attributes, function);
    if (error != hipSuccess) return error;

    const unsigned grid = std::max(1u, std::min(native->maximumGrid(), native->availableProcessors()));
    const std::size_t prefix_bytes = fullNativeHCFFNPrefixDynamicBytes(prefix, grid);
    if(!prefix_bytes || prefix_bytes > max_dynamic_bytes || (prefix_bytes & 15u))
        return hipErrorInvalidConfiguration;
    prefix.hc.resident_grid = grid;

    admission={resolved,function,attributes,device,grid,static_cast<unsigned>(prefix_bytes),general_codebooks,iq_tables};
    return hipSuccess;
}

hipError_t launch(const Admission& admission,hipStream_t stream) noexcept
{
    const auto& invocation=admission.invocation;
    if(reinterpret_cast<std::uintptr_t>(stream)<=2||!admission.function||
       !admission.grid||!admission.dynamic_shared_bytes)
        return hipErrorInvalidValue;
    auto prefix = invocation.prefix;
    unsigned prefix_bytes = admission.dynamic_shared_bytes;
    auto qkv=invocation.projections[0],z=invocation.projections[1];
    HalfPointers half{invocation.half.alpha.words,
        invocation.half.beta.words,
        invocation.half.raw_alpha,invocation.half.raw_beta,
        static_cast<unsigned>(invocation.half.heads)};
    auto full = invocation.full_gdn;
    auto iq_tables = admission.iq_tables;
    void* argv[]{&prefix,&prefix_bytes,&qkv,&z,&half,&full,&iq_tables};
    return hipLaunchKernel(admission.function,dim3(admission.grid),dim3(NT),argv,prefix_bytes,stream);
}
}
