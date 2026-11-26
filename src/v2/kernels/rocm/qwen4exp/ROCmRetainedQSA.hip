/**
 * @file ROCmRetainedQSA.hip
 * @brief Spill-free resident ROCm cooperative QSA Pre-Attention and Post-Attention megakernels.
 *
 * Replaces all 35 participant-local ROCm QSA attention kernels per layer with 4:
 *   1. rocmRetainedQSAPreAttentionKernel (HC prefix + replay + Wq(split)/Wk/Wv +
 *      qsa_key/query + QK RMSNorm + 64-dim RoPE + FP16 KV ring append + advance)
 *   2. Standalone QSA Indexer
 *   3. Standalone selectedAttentionFP16Tiles
 *   4. rocmRetainedQSAPostAttentionKernel (foldAttentionPrefix + commitPartial +
 *      sigmoid fa_gate + Q8 staging + Wo projection)
 *
 * Compiles with zero VGPR/SGPR spills across both gfx906 and gfx1100 and supports
 * all 18 NativeQuantised codebooks plus BF16/FP16/FP32 indexer weights across
 * arbitrary ROCm topologies (4xROCm TP4, 2xROCm+2xCPU TP2, singleton, etc.).
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "kernels/rocm/qwen4exp/ROCmRetainedQSA.h"
#include "backends/rocm/ROCmHardwareMetadata.h"
#include "kernels/common/DeviceNativeQuantisedContributionContract.h"
#include "kernels/common/FullNativeHCFFNBarrierLayout.h"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/common/NativeCooperativeKernel.cuh"
#include "kernels/common/NativeCounter64.h"
#include "kernels/common/OriginalSingleQ8Block32.cuh"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/rocm/gemm/ROCmFullNativeHCFFNPrefixBody.hip.inc"
#include "kernels/rocm/gemm/ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "kernels/rocm/gemm/ROCmQ8ResidualNativeDecode.cuh"
#include "kernels/rocm/moe/ROCmOriginalRouterPhase.hip.inc"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>

extern "C" bool rocmNativeEmbeddingIQTableView(int device_ordinal, llaminar2::NativeEmbeddingIQTables *out);

namespace llaminar2::rocm::retained_qsa
{
namespace
{
    constexpr unsigned kHidden = 2560u;
    constexpr unsigned kLowRank = 320u;
    constexpr unsigned kThreads = 1024u;
    constexpr unsigned kMaxKPartitions = 32u;
    constexpr unsigned kMaxOutBlocks = kMaxROCmRetainedQSAQHeads * 8u; // 192

    template <class T>
    __device__ __forceinline__ T qsaGlobalRead(const void *pointer)
    {
        const auto *source = reinterpret_cast<const unsigned char __attribute__((address_space(1))) *>(
            reinterpret_cast<std::uintptr_t>(pointer));
        T value;
        __builtin_memcpy(&value, source, sizeof(T));
        return value;
    }

    /** @brief 16-byte-aligned 48-byte Q8 block for bank-conflict-free ds_read_b128. */
    struct alignas(16) QSAQ8Block48
    {
        union
        {
            std::int32_t qs[8];
            struct
            {
                int4 q0;
                int4 q1;
            };
        };
        float scale;
        std::int32_t pad[3];
    };
    static_assert(sizeof(QSAQ8Block48) == 48u && alignof(QSAQ8Block48) == 16u);

    constexpr std::size_t kBarrierBytes =
        (FullNativeHCFFNBarrierLayout::bytes + 63u) & ~std::size_t(63u);
    constexpr std::size_t kPostBarrierOffset = kBarrierBytes;
    constexpr std::size_t kOutQ8Offset = kPostBarrierOffset + 64u;
    constexpr std::size_t kTotalScratchBytes =
        ((kOutQ8Offset + kMaxOutBlocks * sizeof(QSAQ8Block48)) + 15u) & ~std::size_t(15u);

    /** @brief Scalar-memory resident grid barrier for gfx906 across arbitrary gridDim.x. */
    template <unsigned PhaseCount>
    struct QSAScalarGridBarrier
    {
        std::uint32_t *words;
        unsigned phase = 0u;
        __device__ __forceinline__ void sync()
        {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX9__)
            if (!words || phase >= PhaseCount)
                __builtin_trap();
            const unsigned grid = __builtin_amdgcn_readfirstlane(gridDim.x);
            std::uint32_t *counter = words + ResidentGridBarrierLayout<PhaseCount>::words_per_phase * phase;
            __asm__ volatile("s_waitcnt vmcnt(0) lgkmcnt(0)" ::: "memory");
            __asm__ volatile("s_barrier" ::: "memory");
            if (threadIdx.x < 64u)
            {
                unsigned old = 1u;
                __asm__ volatile("s_atomic_add %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                                 : "+s"(old) : "s"(counter) : "memory");
                const unsigned target = (old / grid + 1u) * grid;
                while (true)
                {
                    unsigned cur = 0u;
                    __asm__ volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)"
                                     : "=s"(cur) : "s"(counter) : "memory");
                    if (static_cast<int>(cur - target) >= 0)
                        break;
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

    /** @brief Phase-exclusive static LDS for Pre-Attention. */
    union ROCmQSAPreShared
    {
        FullNativeHCFFNPrefixScratch hc;
        QSAQ8Block48 activation[kHidden / 32u];
    };

    /** @brief Phase-exclusive static LDS for Post-Attention. */
    union ROCmQSAPostShared
    {
        struct
        {
            float scales[33];
            float denominator;
            float cached_maximum[33];
            float cached_denominator[33];
            std::uint32_t cached_keys[33];
            std::uint32_t cached_values[33];
        } fold;
        QSAQ8Block48 wo_act[kMaxOutBlocks];
    };

    /** @brief Exact all-18-codebook K32 block dot contribution on ROCm. */
    template <std::uint8_t CB>
    __device__ __forceinline__ float qsaGeneralBlockContribution(
        rocm_q8_residual::NativeDecoder &decoder,
        const DeviceNativeQuantisedMatrixDesc &desc,
        std::size_t linear,
        const QSAQ8Block48 &act)
    {
        using Traits = rocm_q8_residual::NativeDecoder::CodebookTraits<CB>;
        constexpr unsigned bytes = static_cast<unsigned>(Traits::payload_bytes);
        const std::uint8_t *pay_src = desc.payload + linear * bytes;
        std::uint8_t pay_buf[32];
#pragma unroll
        for (unsigned i = 0u; i < bytes; ++i)
            pay_buf[i] = qsaGlobalRead<std::uint8_t>(pay_src + i);
        std::int32_t w[8];
        decoder.template decode_groups<CB>(pay_buf, w);
        const int4 a0 = act.q0;
        const int4 a1 = act.q1;
        const std::int32_t a[8]{a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
        const std::uint16_t primary = qsaGlobalRead<std::uint16_t>(
            static_cast<const std::uint16_t *>(desc.scales) + linear);
        const std::uint16_t secondary = Traits::requires_secondary_metadata
            ? qsaGlobalRead<std::uint16_t>(static_cast<const std::uint16_t *>(desc.mins) + linear) : 0u;
        const std::uint32_t emin = Traits::is_dual_scale_asym
            ? qsaGlobalRead<std::uint32_t>(static_cast<const std::uint32_t *>(desc.emins) + linear) : 0u;
        constexpr std::int32_t ones = 0x01010101;
        if constexpr (Traits::is_dual_scale)
        {
            std::int32_t dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
            for (unsigned g = 0u; g < 4u; ++g)
            {
                dot_lo = rocm_integer_contract::signedDot4(a[g], w[g], dot_lo);
                dot_hi = rocm_integer_contract::signedDot4(a[g + 4u], w[g + 4u], dot_hi);
                sum_lo = rocm_integer_contract::signedDot4(a[g], ones, sum_lo);
                sum_hi = rocm_integer_contract::signedDot4(a[g + 4u], ones, sum_hi);
            }
            std::uint16_t iq1m_qh = 0u;
            std::int32_t sg0 = 0, sg1 = 0, sg2 = 0, sg3 = 0;
            if constexpr (Traits::is_iq1_m)
            {
                iq1m_qh = static_cast<std::uint16_t>(pay_buf[4]) | (static_cast<std::uint16_t>(pay_buf[5]) << 8u);
                sg0 = rocm_integer_contract::signedDot4(a[1], ones, rocm_integer_contract::signedDot4(a[0], ones, 0));
                sg1 = rocm_integer_contract::signedDot4(a[3], ones, rocm_integer_contract::signedDot4(a[2], ones, 0));
                sg2 = rocm_integer_contract::signedDot4(a[5], ones, rocm_integer_contract::signedDot4(a[4], ones, 0));
                sg3 = rocm_integer_contract::signedDot4(a[7], ones, rocm_integer_contract::signedDot4(a[6], ones, 0));
            }
            return decoder.template native_quantised_block_contribution_from_reduced_terms_rn<CB>(
                dot_lo, dot_hi, act.scale, primary, secondary, emin,
                sum_lo + sum_hi, sum_lo, sum_hi, iq1m_qh, sg0, sg1, sg2, sg3);
        }
        else
        {
            std::int32_t dot = 0, sum_a = 0;
#pragma unroll
            for (unsigned g = 0u; g < 8u; ++g)
            {
                dot = rocm_integer_contract::signedDot4(a[g], w[g], dot);
                if constexpr (Traits::is_asymmetric)
                    sum_a = rocm_integer_contract::signedDot4(a[g], ones, sum_a);
            }
            return decoder.template native_quantised_block_contribution_from_reduced_terms_rn<CB>(
                dot, 0, act.scale, primary, secondary, 0u, sum_a, 0, 0, 0u, 0, 0, 0, 0);
        }
    }

    /**
     * @brief 1024B-coalesced 32-column Q8_0 (or all-18-codebook general) projection tile
     *        supporting arbitrary canonical ROCm M1 k_partitions in [1, 32].
     */
    template <bool GeneralCodebooks>
    __device__ __forceinline__ void qsaCompleteROCmTile32(
        const ROCmRetainedQSAProjection &proj,
        unsigned col_base,
        const QSAQ8Block48 *activated,
        unsigned width,
        float *lds_partials,
        const NativeEmbeddingIQTables &iq_tables,
        bool split_q_gate,
        float *dst_q,
        float *dst_gate)
    {
        namespace numeric = device_fp32_contract;
        const unsigned tid = threadIdx.x;
        const unsigned wave = tid >> 6; // 0..15
        const unsigned lane = tid & 63u;
        const unsigned blocks = width >> 5;
        const unsigned kb = __builtin_amdgcn_readfirstlane(static_cast<unsigned>(proj.k_partitions));
        const unsigned step = (blocks + kb - 1u) / kb;
        const unsigned n_cols = static_cast<unsigned>(proj.weights.n);

        float total = 0.0f;
#pragma unroll 1
        for (unsigned p_base = 0u; p_base < kb; p_base += 16u)
        {
            const unsigned chunk = (kb - p_base < 16u) ? (kb - p_base) : 16u;
            if constexpr (!GeneralCodebooks)
            {
                const unsigned col_in_wave = lane >> 1;
                const unsigned half_part = lane & 1u;
                const unsigned col = col_base + col_in_wave;
                const bool valid_col = col < n_cols;
                const std::uint8_t *payload = proj.weights.payload;
                const unsigned short *scales = static_cast<const unsigned short *>(proj.weights.scales);

                if (wave < chunk)
                {
                    const unsigned part = p_base + wave;
                    const unsigned b_begin = part * step;
                    const unsigned b_end = (b_begin + step < blocks) ? (b_begin + step) : blocks;
                    float result = 0.0f;
                    if (valid_col && b_begin < b_end)
                    {
                        std::size_t elem = static_cast<std::size_t>(b_begin) * n_cols + col;
                        int4 w_cur = qsaGlobalRead<int4>(payload + elem * 32u + half_part * 16u);
                        unsigned short sc_cur = (half_part == 0u)
                            ? qsaGlobalRead<unsigned short>(scales + elem) : 0u;
#pragma unroll 1
                        for (unsigned b = b_begin; b < b_end; ++b)
                        {
                            int4 w_next{};
                            unsigned short sc_next = 0u;
                            if (b + 1u < b_end)
                            {
                                const std::size_t next_elem = elem + n_cols;
                                w_next = qsaGlobalRead<int4>(payload + next_elem * 32u + half_part * 16u);
                                if (half_part == 0u)
                                    sc_next = qsaGlobalRead<unsigned short>(scales + next_elem);
                            }
                            const int4 aq = (half_part == 0u) ? activated[b].q0 : activated[b].q1;
                            int d = 0;
                            d = rocm_integer_contract::signedDot4(aq.x, w_cur.x, d);
                            d = rocm_integer_contract::signedDot4(aq.y, w_cur.y, d);
                            d = rocm_integer_contract::signedDot4(aq.z, w_cur.z, d);
                            d = rocm_integer_contract::signedDot4(aq.w, w_cur.w, d);
                            const int full = d + __shfl_xor(d, 1, 64);
                            if (half_part == 0u)
                            {
                                const float ws = __half2float(__ushort_as_half(sc_cur));
                                result = numeric::add(result,
                                    device_native_quantised_contract::singleScaleBlock(full, ws, activated[b].scale));
                            }
                            w_cur = w_next;
                            sc_cur = sc_next;
                            elem += n_cols;
                        }
                    }
                    if (half_part == 0u)
                        lds_partials[wave * 32u + col_in_wave] = result;
                }
            }
            else
            {
                rocm_q8_residual::NativeDecoder decoder(iq_tables);
                const unsigned col = col_base + lane;
                const std::uint8_t cb = static_cast<std::uint8_t>(__builtin_amdgcn_readfirstlane(proj.weights.codebook_id));
                if (wave < chunk)
                {
                    const unsigned part = p_base + wave;
                    const unsigned b_begin = part * step;
                    const unsigned b_end = (b_begin + step < blocks) ? (b_begin + step) : blocks;
                    float result = 0.0f;
                    if (lane < 32u && col < n_cols && b_begin < b_end)
                    {
#pragma unroll 1
                        for (unsigned b = b_begin; b < b_end; ++b)
                        {
                            const std::size_t elem = static_cast<std::size_t>(b) * n_cols + col;
                            float contrib = 0.0f;
                            switch (cb)
                            {
                            case 0u:  contrib = qsaGeneralBlockContribution<0u>(decoder, proj.weights, elem, activated[b]); break;
                            case 4u:  contrib = qsaGeneralBlockContribution<4u>(decoder, proj.weights, elem, activated[b]); break;
                            case 5u:  contrib = qsaGeneralBlockContribution<5u>(decoder, proj.weights, elem, activated[b]); break;
                            case 6u:  contrib = qsaGeneralBlockContribution<6u>(decoder, proj.weights, elem, activated[b]); break;
                            case 7u:  contrib = qsaGeneralBlockContribution<7u>(decoder, proj.weights, elem, activated[b]); break;
                            case 8u:  contrib = qsaGeneralBlockContribution<8u>(decoder, proj.weights, elem, activated[b]); break;
                            case 9u:  contrib = qsaGeneralBlockContribution<9u>(decoder, proj.weights, elem, activated[b]); break;
                            case 10u: contrib = qsaGeneralBlockContribution<10u>(decoder, proj.weights, elem, activated[b]); break;
                            case 11u: contrib = qsaGeneralBlockContribution<11u>(decoder, proj.weights, elem, activated[b]); break;
                            case 12u: contrib = qsaGeneralBlockContribution<12u>(decoder, proj.weights, elem, activated[b]); break;
                            case 13u: contrib = qsaGeneralBlockContribution<13u>(decoder, proj.weights, elem, activated[b]); break;
                            case 14u: contrib = qsaGeneralBlockContribution<14u>(decoder, proj.weights, elem, activated[b]); break;
                            case 15u: contrib = qsaGeneralBlockContribution<15u>(decoder, proj.weights, elem, activated[b]); break;
                            case 16u: contrib = qsaGeneralBlockContribution<16u>(decoder, proj.weights, elem, activated[b]); break;
                            case 17u: contrib = qsaGeneralBlockContribution<17u>(decoder, proj.weights, elem, activated[b]); break;
                            case 19u: contrib = qsaGeneralBlockContribution<19u>(decoder, proj.weights, elem, activated[b]); break;
                            case kNativeQuantisedExpandedInt8MinCodebook:
                                contrib = qsaGeneralBlockContribution<kNativeQuantisedExpandedInt8MinCodebook>(
                                    decoder, proj.weights, elem, activated[b]);
                                break;
                            case kNativeQuantisedIQ4XSSourceScaleCodebook:
                                contrib = qsaGeneralBlockContribution<kNativeQuantisedIQ4XSSourceScaleCodebook>(
                                    decoder, proj.weights, elem, activated[b]);
                                break;
                            default: __builtin_trap();
                            }
                            result = numeric::add(result, contrib);
                        }
                    }
                    if (lane < 32u)
                        lds_partials[wave * 32u + lane] = result;
                }
            }
            __syncthreads();

            if (tid < 32u && (col_base + tid) < n_cols)
            {
                for (unsigned dp = 0u; dp < chunk; ++dp)
                    total = numeric::add(total, lds_partials[dp * 32u + tid]);
            }
            __syncthreads();
        }

        if (tid < 32u && (col_base + tid) < n_cols)
        {
            const unsigned column = col_base + tid;
            if (split_q_gate)
            {
                const unsigned head = column >> 9;
                const unsigned within = column & 511u;
                if (within < 256u)
                    dst_q[head * 256u + within] = total;
                else
                    dst_gate[head * 256u + (within - 256u)] = total;
            }
            else
            {
                proj.output[column] = total;
            }
        }
        __syncthreads();
    }

    /** @brief Exact stride-256 wave64 floating projection row matching fp32x16_scoped_projection_columns_kernel. */
    __device__ __forceinline__ void qsaCompleteROCmFloatingWave64(
        const float *mixed,
        const void *weights,
        ROCmRetainedQSAIndexerFormat format,
        unsigned row,
        float *output)
    {
        const unsigned lane = threadIdx.x & 63u;
        float partial[4]{};
        const std::size_t row_base = static_cast<std::size_t>(row) * kHidden;
        if (format == ROCmRetainedQSAIndexerFormat::BF16)
        {
            const auto *w = static_cast<const std::uint16_t *>(weights) + row_base;
#pragma unroll 1
            for (unsigned step = 0u; step < kHidden / 256u; ++step)
            {
                std::uint16_t wv[4];
#pragma unroll
                for (unsigned g = 0u; g < 4u; ++g)
                    wv[g] = qsaGlobalRead<std::uint16_t>(w + step * 256u + g * 64u + lane);
#pragma unroll
                for (unsigned g = 0u; g < 4u; ++g)
                {
                    const unsigned idx = step * 256u + g * 64u + lane;
                    const float wf = __uint_as_float(static_cast<std::uint32_t>(wv[g]) << 16);
                    partial[g] += mixed[idx] * wf;
                }
            }
        }
        else if (format == ROCmRetainedQSAIndexerFormat::FP16)
        {
            const auto *w = static_cast<const std::uint16_t *>(weights) + row_base;
#pragma unroll 1
            for (unsigned step = 0u; step < kHidden / 256u; ++step)
            {
                std::uint16_t wv[4];
#pragma unroll
                for (unsigned g = 0u; g < 4u; ++g)
                    wv[g] = qsaGlobalRead<std::uint16_t>(w + step * 256u + g * 64u + lane);
#pragma unroll
                for (unsigned g = 0u; g < 4u; ++g)
                {
                    const unsigned idx = step * 256u + g * 64u + lane;
                    const float wf = __half2float(__ushort_as_half(wv[g]));
                    partial[g] += mixed[idx] * wf;
                }
            }
        }
        else
        {
            const auto *w = static_cast<const float *>(weights) + row_base;
#pragma unroll 1
            for (unsigned step = 0u; step < kHidden / 256u; ++step)
            {
                float wv[4];
#pragma unroll
                for (unsigned g = 0u; g < 4u; ++g)
                    wv[g] = qsaGlobalRead<float>(w + step * 256u + g * 64u + lane);
#pragma unroll
                for (unsigned g = 0u; g < 4u; ++g)
                {
                    const unsigned idx = step * 256u + g * 64u + lane;
                    partial[g] += mixed[idx] * wv[g];
                }
            }
        }
        // Exact 128/64 + 6-step wave64 reduction matching reduceFloating256.
        float value = (partial[0] + partial[2]) + (partial[1] + partial[3]);
        for (unsigned stride = 32u; stride > 0u; stride >>= 1)
            value += __shfl_down(value, stride, 64);
        if (lane == 0u)
            output[row] = value;
    }

    /** @brief Exact single-wave64 256-dim RMSNorm + optional 64-dim split-half RoPE (zero LDS). */
    __device__ __forceinline__ float4 qsaROCmWave64NormAndRoPE(
        float4 v,
        const float *gamma,
        float eps,
        bool do_rope,
        int pos,
        unsigned rotary_dim,
        float neg_log_theta,
        unsigned lane)
    {
        if (gamma)
        {
            float sum_sq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            sum_sq += __shfl_xor(sum_sq, 32);
            sum_sq += __shfl_xor(sum_sq, 16);
            sum_sq += __shfl_xor(sum_sq, 8);
            sum_sq += __shfl_xor(sum_sq, 4);
            sum_sq += __shfl_xor(sum_sq, 2);
            sum_sq += __shfl_xor(sum_sq, 1);
            const float inv_rms = rsqrtf(sum_sq * (1.0f / 256.0f) + eps);
            const float4 g = reinterpret_cast<const float4 *>(gamma)[lane];
            v.x = v.x * inv_rms * g.x;
            v.y = v.y * inv_rms * g.y;
            v.z = v.z * inv_rms * g.z;
            v.w = v.w * inv_rms * g.w;
        }
        if (do_rope)
        {
            const unsigned half_rotary = rotary_dim >> 1;
            const unsigned half_lanes = half_rotary >> 2; // 8 lanes for rotary_dim=64
            const unsigned partner_lane = (lane < half_lanes)
                ? (lane + half_lanes)
                : ((lane < 2u * half_lanes) ? (lane - half_lanes) : lane);
            const float px = __shfl(v.x, static_cast<int>(partner_lane), 64);
            const float py = __shfl(v.y, static_cast<int>(partner_lane), 64);
            const float pz = __shfl(v.z, static_cast<int>(partner_lane), 64);
            const float pw = __shfl(v.w, static_cast<int>(partner_lane), 64);
            if (lane < 2u * half_lanes)
            {
                const bool is_low = lane < half_lanes;
                const unsigned pair_base = (is_low ? lane : (lane - half_lanes)) * 4u;
                float vals[4]{v.x, v.y, v.z, v.w};
                const float pvals[4]{px, py, pz, pw};
                float out[4];
#pragma unroll
                for (unsigned c = 0u; c < 4u; ++c)
                {
                    const unsigned pair = pair_base + c;
                    const float exponent = (2.0f * static_cast<float>(pair)) / static_cast<float>(rotary_dim);
                    const float angle = static_cast<float>(pos) * expf(neg_log_theta * exponent);
                    const float cv = __cosf(angle);
                    const float sv = __sinf(angle);
                    out[c] = is_low ? (vals[c] * cv - pvals[c] * sv) : (pvals[c] * sv + vals[c] * cv);
                }
                v = make_float4(out[0], out[1], out[2], out[3]);
            }
        }
        return v;
    }

    __device__ __forceinline__ void qsaROCmRequire(bool cond)
    {
        if (!cond)
            __builtin_trap();
    }

    __device__ __forceinline__ llaminar2::qwen4exp::QwenSparseReplay qsaROCmLoadReplay(
        const llaminar2::qwen4exp::QwenSparseReplay *state, int rows, int tokens)
    {
        const auto r = *state;
        qsaROCmRequire(r.prefix_tokens >= 0 && r.live_rows >= 0 && r.live_rows <= rows &&
                       static_cast<long long>(r.prefix_tokens) + r.live_rows <= tokens &&
                       r.position_origin >= 0 &&
                       static_cast<long long>(r.position_origin) + r.prefix_tokens + r.live_rows <= 0x7fffffff);
        return r;
    }

    /** @brief Exact ROCm foldAttentionPrefix executed inside a 1024-thread CTA (0 spills, #pragma unroll 1). */
    template <int StorageParts>
    __device__ __forceinline__ float2 qsaFoldAttentionPrefixIn1024(
        const llaminar2::qwen4exp::QwenSparseAttention &p,
        const llaminar2::qwen4exp::QwenSparseAttentionPartial *parts,
        int fold_parts,
        ROCmQSAPostShared &shared,
        int row,
        int head,
        unsigned tid)
    {
        constexpr bool PerOutputDenominator = (StorageParts == 33);
        const int fp = __builtin_amdgcn_readfirstlane(fold_parts);
        if constexpr (StorageParts == 33)
        {
            if (tid < static_cast<unsigned>(fp))
            {
                shared.fold.cached_maximum[tid] = parts[tid].maximum;
                shared.fold.cached_denominator[tid] = parts[tid].denominator;
                if (p.observations)
                {
                    shared.fold.cached_keys[tid] = parts[tid].key_elements;
                    shared.fold.cached_values[tid] = parts[tid].value_elements;
                }
            }
            __syncthreads();
            if (tid < 32u)
            {
                float maximum = -INFINITY;
#pragma unroll 1
                for (int part = 0; part < fp; ++part)
                    maximum = fmaxf(maximum, shared.fold.cached_maximum[part]);
                if (tid < static_cast<unsigned>(fp))
                {
                    const float localDen = shared.fold.cached_denominator[tid];
                    shared.fold.scales[tid] = (localDen == 0.0f) ? 0.0f : expf(shared.fold.cached_maximum[tid] - maximum);
                }
                if (fp == 33)
                {
                    const float partDen = shared.fold.cached_denominator[32];
                    const float partScale = (partDen == 0.0f) ? 0.0f : expf(shared.fold.cached_maximum[32] - maximum);
                    if (tid == 0u)
                        shared.fold.scales[32] = partScale;
                }
            }
        }
        else
        {
            if (tid == 0u)
            {
                float maximum = -INFINITY;
#pragma unroll 1
                for (int part = 0; part < fp; ++part)
                    maximum = fmaxf(maximum, parts[part].maximum);
                float sum = 0.0f;
#pragma unroll 1
                for (int part = 0; part < fp; ++part)
                {
                    const float pd = parts[part].denominator;
                    const float pm = parts[part].maximum;
                    shared.fold.scales[part] = (pd == 0.0f) ? 0.0f : expf(pm - maximum);
                    sum = fmaf(pd, shared.fold.scales[part], sum);
                }
                qsaROCmRequire(isfinite(sum) && sum > 0.0f);
                shared.fold.denominator = sum;
                if (p.observations)
                {
                    llaminar2::qwen4exp::QwenSparseAttentionObservation obs{};
#pragma unroll 1
                    for (int part = 0; part < fp; ++part)
                    {
                        obs.key_elements += parts[part].key_elements;
                        obs.value_elements += parts[part].value_elements;
                    }
                    obs.selected_tokens = p.selected_count[row];
                    p.observations[row * p.query_heads + head] = obs;
                }
            }
        }
        __syncthreads();

        if (tid >= 256u)
            return make_float2(0.0f, 1.0f);

        float result = 0.0f;
        float localDenominator = 0.0f;
#pragma unroll 1
        for (int part = 0; part < fp; ++part)
        {
            const float pd = (StorageParts == 33) ? shared.fold.cached_denominator[part] : parts[part].denominator;
            const float weighted = (pd == 0.0f) ? 0.0f : parts[part].weighted[tid];
            const float ps = shared.fold.scales[part];
            result = fmaf(weighted, ps, result);
            if constexpr (PerOutputDenominator)
                localDenominator = fmaf(pd, ps, localDenominator);
        }
        if (fp < StorageParts)
            result = fmaf(0.0f, 0.0f, result);
        if (fp + 1 < StorageParts)
            result = fmaf(0.0f, 0.0f, result);
        if constexpr (PerOutputDenominator)
        {
            qsaROCmRequire(isfinite(localDenominator) && localDenominator > 0.0f);
            if (tid == 0u && p.observations)
            {
                llaminar2::qwen4exp::QwenSparseAttentionObservation obs{};
#pragma unroll 1
                for (int part = 0; part < fp; ++part)
                {
                    obs.key_elements += shared.fold.cached_keys[part];
                    obs.value_elements += shared.fold.cached_values[part];
                }
                obs.selected_tokens = p.selected_count[row];
                p.observations[row * p.query_heads + head] = obs;
            }
            return make_float2(result, localDenominator);
        }
        return make_float2(result, shared.fold.denominator);
    }

    /** @brief Resident ROCm cooperative QSA Pre-Attention megakernel. */
    template <bool GeneralCodebooks>
    __global__ __launch_bounds__(kThreads, 1)
    void rocmRetainedQSAPreAttentionKernel(
        ROCmRetainedQSAPreInvocation inv,
        unsigned prefix_bytes,
        NativeEmbeddingIQTables iq_tables)
    {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
        __shared__ ROCmQSAPreShared shared;
        extern __shared__ __align__(16) unsigned char dynamic_storage[];
        const unsigned tid = threadIdx.x;
        const unsigned bid = blockIdx.x;
        const unsigned num_ctas = gridDim.x;

        // Publish pre-append replay & row_parameters[0] before HC prefix grid barrier.
        if (bid == 0u && tid == 0u)
        {
            const int prefix = *inv.d_cached_tokens;
            const int cursor = *inv.d_ring_head;
            const int cap = inv.kv_token_capacity;
            const int pos = inv.position_ids_device[0];
            if (cursor < 0 || cursor >= cap || prefix < 0 || prefix >= cap || pos < prefix)
                __builtin_trap();
            const int origin = (cursor >= prefix) ? (cursor - prefix) : (cursor - prefix + cap);
            *inv.replay_out = {prefix, 1, pos - prefix};
            inv.row_params_out[0] = {prefix + 1, cap, pos, 0, origin, cap};
        }

        QSAScalarGridBarrier<FullNativeHCFFNBarrierLayout::prefix_phases> hc_grid{inv.prefix.barrier_state};
        fullNativeHCFFNPrefix(inv.prefix, hc_grid, shared.hc, dynamic_storage, prefix_bytes);

        QSAScalarGridBarrier<1> qsa_grid{
            inv.prefix.barrier_state + FullNativeHCFFNBarrierLayout::suffix_word_offset};

        // =====================================================================
        // Phase 1: Quantize mixed -> LDS Q8 + Wq(split)/Wk/Wv + qsa_key/query
        // =====================================================================
        const float *mixed = inv.prefix.hc.mixed;
        float *tile_lds = reinterpret_cast<float *>(dynamic_storage);
        for (unsigned physical = tid; physical < kHidden; physical += kThreads)
        {
            const unsigned block = physical >> 5;
            const unsigned lane = physical & 31u;
            const float mv = mixed[physical];
            auto &q = shared.activation[block];
            original_single_q8::publish(
                mv, true, reinterpret_cast<std::int8_t *>(q.qs), &q.scale, nullptr, lane);
        }
        __syncthreads();

        const int prefix_tokens = inv.replay_out->prefix_tokens;
        const bool run_qsa_query =
            (inv.qsa_query.format != ROCmRetainedQSAIndexerFormat::Absent) &&
            (!inv.qsa_query.ranking_only_guard ||
             prefix_tokens >= llaminar2::qwen4exp::QwenSparseFirstRankingLogicalRow);

        const unsigned wq_tiles = (static_cast<unsigned>(inv.wq.weights.n) + 31u) >> 5;
        const unsigned wk_tiles = (static_cast<unsigned>(inv.wk.weights.n) + 31u) >> 5;
        const unsigned wv_tiles = (static_cast<unsigned>(inv.wv.weights.n) + 31u) >> 5;
        const unsigned key_q_tiles = (inv.qsa_key.format == ROCmRetainedQSAIndexerFormat::NativeQuantised)
            ? ((static_cast<unsigned>(inv.qsa_key.n) + 31u) >> 5) : 0u;
        const unsigned qry_q_tiles = (run_qsa_query &&
            inv.qsa_query.format == ROCmRetainedQSAIndexerFormat::NativeQuantised)
            ? ((static_cast<unsigned>(inv.qsa_query.n) + 31u) >> 5) : 0u;
        const unsigned total_q_tiles = wq_tiles + wk_tiles + wv_tiles + key_q_tiles + qry_q_tiles;

#pragma unroll 1
        for (unsigned tile = bid; tile < total_q_tiles; tile += num_ctas)
        {
            const ROCmRetainedQSAProjection *proj = nullptr;
            unsigned col_base = 0u;
            bool split_q = false;
            if (tile < wq_tiles)
            {
                proj = &inv.wq;
                col_base = tile << 5;
                split_q = true;
            }
            else if (tile < wq_tiles + wk_tiles)
            {
                proj = &inv.wk;
                col_base = (tile - wq_tiles) << 5;
            }
            else if (tile < wq_tiles + wk_tiles + wv_tiles)
            {
                proj = &inv.wv;
                col_base = (tile - wq_tiles - wk_tiles) << 5;
            }
            else if (tile < wq_tiles + wk_tiles + wv_tiles + key_q_tiles)
            {
                proj = &inv.qsa_key.quantised;
                col_base = (tile - wq_tiles - wk_tiles - wv_tiles) << 5;
            }
            else
            {
                proj = &inv.qsa_query.quantised;
                col_base = (tile - wq_tiles - wk_tiles - wv_tiles - key_q_tiles) << 5;
            }
            qsaCompleteROCmTile32<GeneralCodebooks>(
                *proj, col_base, shared.activation, kHidden, tile_lds, iq_tables,
                split_q, inv.Q, inv.fa_gate);
        }

        // Tail-shadow schedule floating BF16/FP16/FP32 indexer rows (1 wave64 per row).
        const unsigned key_f_rows = (inv.qsa_key.format == ROCmRetainedQSAIndexerFormat::FP32 ||
                                     inv.qsa_key.format == ROCmRetainedQSAIndexerFormat::FP16 ||
                                     inv.qsa_key.format == ROCmRetainedQSAIndexerFormat::BF16)
            ? static_cast<unsigned>(inv.qsa_key.n) : 0u;
        const unsigned qry_f_rows = (run_qsa_query &&
                                     (inv.qsa_query.format == ROCmRetainedQSAIndexerFormat::FP32 ||
                                      inv.qsa_query.format == ROCmRetainedQSAIndexerFormat::FP16 ||
                                      inv.qsa_query.format == ROCmRetainedQSAIndexerFormat::BF16))
            ? static_cast<unsigned>(inv.qsa_query.n) : 0u;
        const unsigned total_f_rows = key_f_rows + qry_f_rows;
        if (total_f_rows > 0u)
        {
            const unsigned shift = total_q_tiles % num_ctas;
            const unsigned shifted_bid = (bid + num_ctas - shift) % num_ctas;
            const unsigned global_wave = shifted_bid * 16u + (tid >> 6);
            const unsigned wave_stride = num_ctas * 16u;
#pragma unroll 1
            for (unsigned r = global_wave; r < total_f_rows; r += wave_stride)
            {
                const bool is_key = r < key_f_rows;
                const void *fw = is_key ? inv.qsa_key.floating_weights : inv.qsa_query.floating_weights;
                const auto fmt = is_key ? inv.qsa_key.format : inv.qsa_query.format;
                const unsigned row_idx = is_key ? r : (r - key_f_rows);
                float *fout = is_key ? inv.qsa_key.floating_output : inv.qsa_query.floating_output;
                qsaCompleteROCmFloatingWave64(mixed, fw, fmt, row_idx, fout);
            }
        }

        qsa_grid.sync();

        // =====================================================================
        // Phase 2: 1 wave64 per head Q/K RMSNorm + 64-dim RoPE + FP16 KV append
        //          + canonical device ring_head/cached_tokens advance
        // =====================================================================
        const int pos = inv.position_ids_device[0];
        const int cap = inv.kv_token_capacity;
        const int old_head = (inv.row_params_out[0].ring_row_origin + prefix_tokens) % cap;
        const unsigned n_q = static_cast<unsigned>(inv.local_q_heads);
        const unsigned n_kv = static_cast<unsigned>(inv.local_kv_heads);
        const unsigned total_heads = n_q + n_kv;
        const unsigned lane = tid & 63u;
        const unsigned global_wave = bid * 16u + (tid >> 6);
        const unsigned wave_stride = num_ctas * 16u;

        if (bid == 0u && tid == 0u)
        {
            *inv.d_ring_head = (old_head + 1) % cap;
            const int next_count = prefix_tokens + 1;
            *inv.d_cached_tokens = (next_count > cap) ? cap : next_count;
        }

#pragma unroll 1
        for (unsigned h = global_wave; h < total_heads; h += wave_stride)
        {
            if (h < n_q)
            {
                auto *q4 = reinterpret_cast<float4 *>(inv.Q + static_cast<std::size_t>(h) * 256u);
                const float4 out = qsaROCmWave64NormAndRoPE(
                    q4[lane], inv.q_norm_gamma, inv.qk_norm_eps,
                    true, pos, static_cast<unsigned>(inv.rotary_dim), inv.rope_neg_log_theta, lane);
                q4[lane] = out;
            }
            else
            {
                const unsigned kv_h = h - n_q;
                auto *k4 = reinterpret_cast<float4 *>(inv.wk.output + static_cast<std::size_t>(kv_h) * 256u);
                const auto *v4 = reinterpret_cast<const float4 *>(inv.wv.output + static_cast<std::size_t>(kv_h) * 256u);
                const float4 kv = qsaROCmWave64NormAndRoPE(
                    k4[lane], inv.k_norm_gamma, inv.qk_norm_eps,
                    !inv.skip_k_rope, pos, static_cast<unsigned>(inv.rotary_dim), inv.rope_neg_log_theta, lane);
                const float4 vv = v4[lane];
                k4[lane] = kv;

                const std::size_t byte_off =
                    static_cast<std::size_t>(old_head) * inv.kv_token_stride_bytes +
                    static_cast<std::size_t>(kv_h) * inv.kv_head_stride_bytes;
                auto *dst_k = reinterpret_cast<_Float16 *>(
                    static_cast<unsigned char *>(inv.kv_keys_data) + byte_off) + lane * 4u;
                auto *dst_v = reinterpret_cast<_Float16 *>(
                    static_cast<unsigned char *>(inv.kv_values_data) + byte_off) + lane * 4u;
                dst_k[0] = static_cast<_Float16>(kv.x);
                dst_k[1] = static_cast<_Float16>(kv.y);
                dst_k[2] = static_cast<_Float16>(kv.z);
                dst_k[3] = static_cast<_Float16>(kv.w);
                dst_v[0] = static_cast<_Float16>(vv.x);
                dst_v[1] = static_cast<_Float16>(vv.y);
                dst_v[2] = static_cast<_Float16>(vv.z);
                dst_v[3] = static_cast<_Float16>(vv.w);
            }
        }
#else
        __builtin_trap();
#endif
    }

    /** @brief Resident ROCm cooperative QSA Post-Attention megakernel (fold + commit + gate + Wo). */
    template <int Parts, bool GeneralCodebooks>
    __global__ __launch_bounds__(kThreads, 1)
    void rocmRetainedQSAPostAttentionKernel(
        ROCmRetainedQSAPostInvocation inv,
        NativeEmbeddingIQTables iq_tables)
    {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX9__)
        static_assert(Parts == 8 || Parts == 33, "Closed ROCm FP16 QSA partition program");
        __shared__ ROCmQSAPostShared shared;
        extern __shared__ __align__(16) unsigned char dynamic_storage[];
        const unsigned tid = threadIdx.x;
        const unsigned bid = blockIdx.x;
        const unsigned num_ctas = gridDim.x;
        const unsigned n_q = static_cast<unsigned>(inv.attention.query_heads);

        auto *scratch_bytes = static_cast<unsigned char *>(inv.scratch);
        auto *wo_q8_global = reinterpret_cast<QSAQ8Block48 *>(scratch_bytes + kOutQ8Offset);
        QSAScalarGridBarrier<1> post_grid{
            reinterpret_cast<std::uint32_t *>(scratch_bytes + kPostBarrierOffset)};

        // =====================================================================
        // Phase 0: Per-head fold + sigmoid gate + Q8 pack, overlapped with commitPartial
        // =====================================================================
        const auto r = qsaROCmLoadReplay(
            inv.attention.replay, inv.attention.row_capacity, inv.attention.token_capacity);
        const int live_rows = r.live_rows;

#pragma unroll 1
        for (unsigned head = bid; head < n_q; head += num_ctas)
        {
            if (live_rows <= 0)
            {
                if (tid == 0u && inv.attention.observations)
                    inv.attention.observations[head] = {};
                continue;
            }
            const auto *parts = inv.attention.gpu_partials + static_cast<std::size_t>(head) * Parts;
            int fold_parts = 8;
            if constexpr (Parts == 33)
            {
                const int selected = inv.attention.selected_count[0];
                qsaROCmRequire(selected > 0 && selected <= llaminar2::qwen4exp::QwenSparseSelected);
                if (selected <= 8 * 64)
                    fold_parts = 8;
                else if (selected <= 12 * 64)
                    fold_parts = 12;
                else if (selected <= 16 * 64)
                    fold_parts = 16;
                else if (selected <= 32 * 64)
                    fold_parts = 32;
                else
                    fold_parts = 33;
            }
            const float2 folded = qsaFoldAttentionPrefixIn1024<Parts>(
                inv.attention, parts, fold_parts, shared, 0, static_cast<int>(head), tid);

            if (tid < 256u)
            {
                float result = folded.x / folded.y;
                const unsigned elem = head * 256u + tid;
                const float sig = 1.0f / (1.0f + expf(-inv.fa_gate[elem]));
                result = sig * result;
                inv.attention.output[elem] = result;

                // Exact width-32 Q8 quantization matching rocm_gdn_gated_norm_single_q8_kernel.
                float maximum = fabsf(result);
#pragma unroll
                for (int shift = 16; shift > 0; shift >>= 1)
                    maximum = fmaxf(maximum, __shfl_down(maximum, shift, 32));
                maximum = __shfl(maximum, 0, 32);
                const float scale = rocm::decode_equivalent::q8Scale(maximum);
                const float inverse = 1.0f / scale;
                const int q = rocm::decode_equivalent::quantizeQ8(result, inverse);
                const unsigned sub = tid & 31u;
                const unsigned blk = tid >> 5;
                const int base_sub = static_cast<int>(sub & ~3u);
                std::uint32_t packed = static_cast<std::uint8_t>(q);
                packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(__shfl(q, base_sub + 1, 32))) << 8;
                packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(__shfl(q, base_sub + 2, 32))) << 16;
                packed |= static_cast<std::uint32_t>(static_cast<std::uint8_t>(__shfl(q, base_sub + 3, 32))) << 24;
                auto &dst = wo_q8_global[head * 8u + blk];
                if ((sub & 3u) == 0u)
                    dst.qs[sub >> 2] = __builtin_bit_cast(std::int32_t, packed);
                if (sub == 0u)
                    dst.scale = scale;
            }
            __syncthreads();
        }

        if (inv.has_commit && bid == (n_q % num_ctas))
        {
            const auto &cp = inv.commit;
            const auto cr = qsaROCmLoadReplay(
                cp.replay, cp.acceptance.candidates().rowsPerRequest(),
                cp.transaction_prior ? cp.token_capacity : 0x7fffffff);
            qsaROCmRequire(!cp.transaction_prior ||
                           cp.transaction_prior->row_capacity == cp.acceptance.candidates().rowsPerRequest());
            const int accepted = cp.acceptance.resolveNative(0);
            qsaROCmRequire(accepted >= 0 && accepted <= cr.live_rows);
            const bool active = !(cr.live_rows == 0 || (accepted == 0 && !cp.transaction_prior));
            const int total = cr.prefix_tokens + accepted;
            const int begin = (total / 4) * 4;
            const int rem = total % 4;
            float values[3];
            if (active && tid < 128u)
            {
                for (int member = 0; member < rem; ++member)
                {
                    const int token = begin + member;
                    values[member] = token < cr.prefix_tokens
                        ? cp.prior_partial[static_cast<std::size_t>(token - (cr.prefix_tokens / 4) * 4) * 128u + tid]
                        : cp.raw_keys[static_cast<std::size_t>(token - cr.prefix_tokens) * 128u + tid];
                }
            }
            __syncthreads();
            if (active && tid < 128u)
            {
                for (int member = 0; member < rem; ++member)
                    cp.partial_keys[static_cast<std::size_t>(member) * 128u + tid] = values[member];
                if (tid == 0u && cp.work_counters.bound())
                    qsaROCmRequire(::llaminar2::addNativeCounter64(
                        cp.work_counters.metric(llaminar2::qwen4exp::QwenSparseWorkCounter::AcceptedPayload),
                        static_cast<std::uint64_t>(rem) * 128u * sizeof(float)));
            }
        }

        post_grid.sync();

        // =====================================================================
        // Phase 1: Load Q8 attn_output into LDS + coalesced 32-col Wo projection
        // =====================================================================
        const unsigned wo_k = n_q * 256u;
        const unsigned wo_blocks = wo_k >> 5;
        const auto *src_v4 = reinterpret_cast<const int4 *>(wo_q8_global);
        auto *dst_v4 = reinterpret_cast<int4 *>(shared.wo_act);
        const unsigned total_v4 = wo_blocks * (sizeof(QSAQ8Block48) / sizeof(int4));
        for (unsigned v = tid; v < total_v4; v += kThreads)
            dst_v4[v] = src_v4[v];
        __syncthreads();

        float *wo_lds = reinterpret_cast<float *>(dynamic_storage);
        const unsigned wo_tiles = (static_cast<unsigned>(inv.wo.weights.n) + 31u) >> 5;
#pragma unroll 1
        for (unsigned tile = bid; tile < wo_tiles; tile += num_ctas)
        {
            qsaCompleteROCmTile32<GeneralCodebooks>(
                inv.wo, tile << 5, shared.wo_act, wo_k, wo_lds, iq_tables, false, nullptr, nullptr);
        }
#else
        __builtin_trap();
#endif
    }

    bool supportedROCmNativeCodebook(std::uint8_t cb) noexcept
    {
        switch (cb)
        {
        case 0u: case 4u: case 5u: case 6u: case 7u: case 8u: case 9u: case 10u:
        case 11u: case 12u: case 13u: case 14u: case 15u: case 16u: case 17u: case 19u:
        case kNativeQuantisedExpandedInt8MinCodebook:
        case kNativeQuantisedIQ4XSSourceScaleCodebook:
            return true;
        default:
            return false;
        }
    }

    bool validROCmHCMatrix(const DeviceNativeQuantisedMatrixDesc &w, int n, int k) noexcept
    {
        return w.valid() && w.n == n && w.k == k && w.blocks_per_row == static_cast<unsigned>(k / 32) &&
               w.source_identity_present && supportedROCmNativeCodebook(w.codebook_id);
    }

    bool validROCmProj(const ROCmRetainedQSAProjection &p, int expected_n, int expected_k, bool require_out) noexcept
    {
        return (!require_out || p.output != nullptr) &&
               p.weights.valid() && p.weights.n == expected_n && p.weights.k == expected_k &&
               p.weights.blocks_per_row == static_cast<unsigned>(expected_k / 32) &&
               p.weights.source_identity_present &&
               supportedROCmNativeCodebook(p.weights.codebook_id) &&
               p.k_partitions > 0 && p.k_partitions <= static_cast<int>(p.weights.blocks_per_row);
    }

    bool isQ8ROCmProj(const ROCmRetainedQSAProjection &p) noexcept
    {
        return p.weights.codebook_id == 19u && !p.weights.mins && !p.weights.emins;
    }

    bool validROCmIndexerProj(const ROCmRetainedQSAIndexerProj &p, int expected_n, bool optional) noexcept
    {
        if (p.format == ROCmRetainedQSAIndexerFormat::Absent)
            return optional;
        if (p.n != expected_n)
            return false;
        if (p.format == ROCmRetainedQSAIndexerFormat::NativeQuantised)
            return validROCmProj(p.quantised, expected_n, static_cast<int>(kHidden), true);
        return p.floating_weights != nullptr && p.floating_output != nullptr;
    }

    bool isQ8OrFloatingROCmIndexer(const ROCmRetainedQSAIndexerProj &p) noexcept
    {
        return p.format != ROCmRetainedQSAIndexerFormat::NativeQuantised || isQ8ROCmProj(p.quantised);
    }
} // namespace

std::size_t retainedROCmQSAScratchBytes() noexcept
{
    return kTotalScratchBytes;
}

hipError_t prepareROCmRetainedQSAPreAttention(
    const ROCmRetainedQSAPreInvocation &invocation,
    int device,
    ROCmRetainedQSAPreAdmission &admission)
{
    ROCmRetainedQSAPreInvocation resolved = invocation;
    auto &prefix = resolved.prefix;
    auto &hc = prefix.hc;
    if (!resolved.scratch || resolved.scratch_bytes < retainedROCmQSAScratchBytes() ||
        (reinterpret_cast<std::uintptr_t>(resolved.scratch) & 15u) != 0u ||
        hc.norm.hidden != kHidden || !hc.norm.gain || !hc.norm.output ||
        !hc.mixed || !hc.raw_down || !hc.down_partials || !hc.norm.input ||
        (static_cast<bool>(hc.injection.weights) != static_cast<bool>(hc.injection.output)) ||
        !validROCmHCMatrix(hc.down_weights, static_cast<int>(kLowRank), static_cast<int>(4u * kHidden)) ||
        !validROCmHCMatrix(hc.up.weights, static_cast<int>(4u * kHidden), static_cast<int>(kLowRank)) ||
        hc.norm.extent.rows != 1 || hc.up.rows.physicalRows() != 1 || hc.up.rows.capacity() != 1 ||
        hc.norm.extent.liveRows != hc.up.rows.countOwner() ||
        hc.up.quantizer != Q8ResidualQuantizerPolicy::ROCmFiniteHalfNativeReciprocal ||
        prefix.output_demand.bits())
        return hipErrorInvalidValue;

    if (resolved.local_q_heads < 1 ||
        static_cast<unsigned>(resolved.local_q_heads) > kMaxROCmRetainedQSAQHeads ||
        resolved.local_kv_heads < 1 ||
        static_cast<unsigned>(resolved.local_kv_heads) > kMaxROCmRetainedQSAKVHeads ||
        resolved.rotary_dim <= 0 || resolved.rotary_dim > 256 || (resolved.rotary_dim & 7) != 0 ||
        !resolved.Q || !resolved.fa_gate ||
        !resolved.position_ids_device || !resolved.d_ring_head || !resolved.d_cached_tokens ||
        resolved.kv_token_capacity <= 0 ||
        !resolved.kv_keys_data || !resolved.kv_values_data ||
        resolved.kv_token_stride_bytes == 0 || resolved.kv_head_stride_bytes == 0 ||
        !resolved.replay_out || !resolved.row_params_out ||
        !validROCmProj(resolved.wq, resolved.local_q_heads * 512, static_cast<int>(kHidden), false) ||
        !validROCmProj(resolved.wk, resolved.local_kv_heads * 256, static_cast<int>(kHidden), true) ||
        !validROCmProj(resolved.wv, resolved.local_kv_heads * 256, static_cast<int>(kHidden), true) ||
        !validROCmIndexerProj(resolved.qsa_key, 128, false) ||
        !validROCmIndexerProj(resolved.qsa_query, 512, true))
        return hipErrorInvalidValue;

    const bool all_q8 = isQ8ROCmProj(resolved.wq) && isQ8ROCmProj(resolved.wk) && isQ8ROCmProj(resolved.wv) &&
                        isQ8OrFloatingROCmIndexer(resolved.qsa_key) &&
                        isQ8OrFloatingROCmIndexer(resolved.qsa_query);
    const void *fn = all_q8
        ? reinterpret_cast<const void *>(&rocmRetainedQSAPreAttentionKernel<false>)
        : reinterpret_cast<const void *>(&rocmRetainedQSAPreAttentionKernel<true>);

    NativeEmbeddingIQTables iq_tables{};
    if (!all_q8 && !rocmNativeEmbeddingIQTableView(device, &iq_tables))
        return hipErrorInvalidDevice;

    int current = -1;
    auto err = hipGetDevice(&current);
    if (err != hipSuccess)
        return err;
    if (current != device)
        return hipErrorInvalidDevice;
    const auto &hw = rocmHardwareMetadata().at(device);
    if (hw.compute_units <= 0 || hw.native_wave_width != 64 ||
        hw.lds_per_block <= sizeof(ROCmQSAPreShared))
        return hipErrorInvalidDevice;

    try
    {
        resolvePersistentHCReadPolicy(prefix.hc, prefix.output_demand);
    }
    catch (...)
    {
        return hipErrorInvalidValue;
    }
    prefix.barrier_state = reinterpret_cast<std::uint32_t *>(resolved.scratch);
    const std::size_t max_dyn = std::max<std::size_t>(
        fullNativeHCFFNPrefixDynamicBytes(prefix, 1),
        kMaxKPartitions * 32u * sizeof(float));
    if (!max_dyn || max_dyn > 32768u || (max_dyn & 15u) ||
        max_dyn + sizeof(ROCmQSAPreShared) > hw.lds_per_block)
        return hipErrorInvalidConfiguration;

    const unsigned cus = static_cast<unsigned>(hw.compute_units);
    const auto co_residency = NativeCoResidencyReservation::reserveProcessors(cus > 4u ? 4u : 0u);
    std::optional<NativeCooperativeKernelAdmission> native;
    try
    {
        native.emplace(NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(
            DeviceId::rocm(device), fn, kThreads, max_dyn, 1u,
            NativeResidentLaunchPolicy::Ordinary, co_residency));
    }
    catch (...)
    {
        return hipErrorInvalidConfiguration;
    }
    const unsigned grid = std::max(1u, std::min(native->maximumGrid(), native->availableProcessors()));
    const std::size_t prefix_bytes = std::max<std::size_t>(
        fullNativeHCFFNPrefixDynamicBytes(prefix, grid),
        kMaxKPartitions * 32u * sizeof(float));
    if (!prefix_bytes || prefix_bytes > max_dyn || (prefix_bytes & 15u))
        return hipErrorInvalidConfiguration;
    prefix.hc.resident_grid = grid;

    admission = {resolved, fn, device, grid, static_cast<unsigned>(prefix_bytes), iq_tables};
    return hipSuccess;
}

hipError_t launchROCmRetainedQSAPreAttention(
    const ROCmRetainedQSAPreAdmission &admission,
    hipStream_t stream) noexcept
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2u || !admission.function ||
        !admission.grid || !admission.dynamic_shared_bytes)
        return hipErrorInvalidValue;
    auto inv = admission.invocation;
    unsigned prefix_bytes = admission.dynamic_shared_bytes;
    auto iq_tables = admission.iq_tables;
    void *argv[] = {&inv, &prefix_bytes, &iq_tables};
    return hipLaunchKernel(admission.function, dim3(admission.grid), dim3(kThreads), argv, prefix_bytes, stream);
}

hipError_t prepareROCmRetainedQSAPostAttention(
    const ROCmRetainedQSAPostInvocation &invocation,
    int device,
    ROCmRetainedQSAPostAdmission &admission)
{
    const auto &a = invocation.attention;
    if (a.row_capacity != 1 || a.query_heads < 1 ||
        static_cast<unsigned>(a.query_heads) > kMaxROCmRetainedQSAQHeads ||
        (a.arithmetic != llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals &&
         a.arithmetic != llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33) ||
        !a.replay || !a.selected_count || !a.gpu_partials || !a.output ||
        !invocation.fa_gate ||
        !validROCmProj(invocation.wo, static_cast<int>(kHidden), a.query_heads * 256, true) ||
        !invocation.scratch || invocation.scratch_bytes < retainedROCmQSAScratchBytes() ||
        (reinterpret_cast<std::uintptr_t>(invocation.scratch) & 15u) != 0u)
        return hipErrorInvalidValue;

    const bool parts33 =
        a.arithmetic == llaminar2::qwen4exp::QwenSparseAttentionArithmetic::TileFP16WithFP32Totals33;
    const bool q8_wo = isQ8ROCmProj(invocation.wo);
    const void *fn = parts33
        ? (q8_wo ? reinterpret_cast<const void *>(&rocmRetainedQSAPostAttentionKernel<33, false>)
                 : reinterpret_cast<const void *>(&rocmRetainedQSAPostAttentionKernel<33, true>))
        : (q8_wo ? reinterpret_cast<const void *>(&rocmRetainedQSAPostAttentionKernel<8, false>)
                 : reinterpret_cast<const void *>(&rocmRetainedQSAPostAttentionKernel<8, true>));

    NativeEmbeddingIQTables iq_tables{};
    if (!q8_wo && !rocmNativeEmbeddingIQTableView(device, &iq_tables))
        return hipErrorInvalidDevice;

    int current = -1;
    auto err = hipGetDevice(&current);
    if (err != hipSuccess)
        return err;
    if (current != device)
        return hipErrorInvalidDevice;
    const auto &hw = rocmHardwareMetadata().at(device);
    const std::size_t dyn_bytes = kMaxKPartitions * 32u * sizeof(float); // 4096 B
    if (hw.compute_units <= 0 || hw.native_wave_width != 64 ||
        hw.lds_per_block <= sizeof(ROCmQSAPostShared) + dyn_bytes)
        return hipErrorInvalidDevice;

    const unsigned cus = static_cast<unsigned>(hw.compute_units);
    const auto co_residency = NativeCoResidencyReservation::reserveProcessors(cus > 4u ? 4u : 0u);
    std::optional<NativeCooperativeKernelAdmission> native;
    try
    {
        native.emplace(NativeCooperativeKernelBoundary<NativeCooperativeBackend::ROCm>::prepare(
            DeviceId::rocm(device), fn, kThreads, dyn_bytes, 1u,
            NativeResidentLaunchPolicy::Ordinary, co_residency));
    }
    catch (...)
    {
        return hipErrorInvalidConfiguration;
    }
    const unsigned grid = std::max(1u, std::min(native->maximumGrid(), native->availableProcessors()));
    admission = {invocation, fn, device, grid, static_cast<unsigned>(dyn_bytes), iq_tables};
    return hipSuccess;
}

hipError_t launchROCmRetainedQSAPostAttention(
    const ROCmRetainedQSAPostAdmission &admission,
    hipStream_t stream) noexcept
{
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2u || !admission.function ||
        !admission.grid || !admission.dynamic_shared_bytes)
        return hipErrorInvalidValue;
    auto inv = admission.invocation;
    auto iq_tables = admission.iq_tables;
    void *argv[] = {&inv, &iq_tables};
    return hipLaunchKernel(
        admission.function, dim3(admission.grid), dim3(kThreads), argv,
        admission.dynamic_shared_bytes, stream);
}
} // namespace llaminar2::rocm::retained_qsa
