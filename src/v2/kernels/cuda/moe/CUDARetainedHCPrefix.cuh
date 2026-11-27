/**
 * @file CUDARetainedHCPrefix.cuh
 * @brief Resident CUDA arithmetic from the rounded attention write to the mixed FFN input.
 *
 * The native grid joins the original publication edges. Lanes retain their
 * integer accumulation and the original serial FP32 folds. The entry is called
 * only for an admitted live row and uses the existing quantization banks.
 */
#pragma once

#include "CUDARetainedHCPrefix.h"
#include "kernels/common/DeviceNativeQuantisedContributionContract.h"
#include "kernels/common/GPUHCDownPublication.cuh"
#include "kernels/cuda/gemm/CUDANativeQuantisedDecodeCommon.cuh"
#include "kernels/qwen4exp/GPUBranchRMSNormBody.cuh"

#include <cooperative_groups.h>
#include <cuda_fp16.h>

namespace llaminar2::cuda::moe
{
    /**
     * @brief CTA-local norm, HC Down and HC Up tiles reused across sequential prefix phases.
     * @tparam Hidden Physical branch width authenticated by the resident program.
     */
    template<unsigned Hidden>
    struct RetainedHCPrefixShared
    {
        float norm[256]; ///< Original fixed 256-lane square reduction.
        union
        {
            alignas(16) float post[Hidden]; ///< Rounded POST reused for the four-channel wave layout.
            float down_blocks[32][32]; ///< Per-K32-block partials for one coalesced 32-column Down partition.
            float down_blocks16[32][16]; ///< Per-K32-block partials for one warp-split 16-column Down partition.
            struct
            {
                float primary[4][10][16];
                float residual[4][10][16];
                float logit[4][16];
            } up; ///< Coalesced 16-channel x 4-branch HC Up block partials.
        };
    };

    /**
     * @brief Publish four neighboring CUDA norm columns through the original Q8 tree.
     * @param value The four rounded normalized values for this lane.
     * @param codes The owned four original Q8 output bytes.
     * @param scale_word Original scale for this complete 32-column packet.
     * @param lane Physical lane in a complete width-eight subgroup.
     */
    __device__ __forceinline__ void retainedHCNormQ8Quad(float4 value, int8_t *codes,
        float *scale_word, unsigned lane)
    {
        float x = fabsf(value.x);
        float y = fabsf(value.y);
        float z = fabsf(value.z);
        float w = fabsf(value.w);
#pragma unroll
        for (int mask = 4; mask > 0; mask >>= 1)
        {
            x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, mask, 8));
            y = fmaxf(y, __shfl_xor_sync(0xffffffffu, y, mask, 8));
            z = fmaxf(z, __shfl_xor_sync(0xffffffffu, z, mask, 8));
            w = fmaxf(w, __shfl_xor_sync(0xffffffffu, w, mask, 8));
        }
        const float scale = original_single_q8::cudaScale(fmaxf(fmaxf(x, z), fmaxf(y, w)));
        const float inverse = 1.0f / scale;
        if (lane == 0)
            *scale_word = scale;
        const uint32_t packed =
            (static_cast<uint32_t>(original_single_q8::cudaCode(value.x, inverse)) & 255u) |
            ((static_cast<uint32_t>(original_single_q8::cudaCode(value.y, inverse)) & 255u) << 8) |
            ((static_cast<uint32_t>(original_single_q8::cudaCode(value.z, inverse)) & 255u) << 16) |
            ((static_cast<uint32_t>(original_single_q8::cudaCode(value.w, inverse)) & 255u) << 24);
        __builtin_memcpy(codes, &packed, sizeof(packed));
    }

    /**
     * @brief Compute one CUDA Q8_0 packet against both primary and residual Q8 activations with a single weight load.
     */
    __device__ __forceinline__ void retainedHCQ8DualContribution(
        const uint8_t* weights, const int8_t* primary_codes, const int8_t* residual_codes,
        float weight_scale, float primary_scale, float residual_scale,
        float& primary_out, float& residual_out)
    {
        const auto* w = reinterpret_cast<const int4*>(weights);
        const auto* p = reinterpret_cast<const int4*>(primary_codes);
        const auto* r = reinterpret_cast<const int4*>(residual_codes);
        const int4 wl = w[0], wh = w[1];
        const int4 pl = p[0], ph = p[1];
        const int4 rl = r[0], rh = r[1];
        int plo = __dp4a(wl.x, pl.x, 0);
        int phi = __dp4a(wh.x, ph.x, 0);
        int rlo = __dp4a(wl.x, rl.x, 0);
        int rhi = __dp4a(wh.x, rh.x, 0);
        plo = __dp4a(wl.y, pl.y, plo);
        phi = __dp4a(wh.y, ph.y, phi);
        rlo = __dp4a(wl.y, rl.y, rlo);
        rhi = __dp4a(wh.y, rh.y, rhi);
        plo = __dp4a(wl.z, pl.z, plo);
        phi = __dp4a(wh.z, ph.z, phi);
        rlo = __dp4a(wl.z, rl.z, rlo);
        rhi = __dp4a(wh.z, rh.z, rhi);
        plo = __dp4a(wl.w, pl.w, plo);
        phi = __dp4a(wh.w, ph.w, phi);
        rlo = __dp4a(wl.w, rl.w, rlo);
        rhi = __dp4a(wh.w, rh.w, rhi);
        primary_out = device_native_quantised_contract::singleScaleBlock(plo + phi, weight_scale, primary_scale);
        residual_out = device_native_quantised_contract::singleScaleBlock(rlo + rhi, weight_scale, residual_scale);
    }

    /**
     * @brief Compute a CUDA Q8_0 packet contribution using its original FP32 scale order.
     * @param weights Base of one aligned prepared 32-byte packet.
     * @param codes Base of one aligned prepared activation packet.
     * @param weight_scale Widened prepared FP16 scale.
     * @param activation_scale Original FP32 scale.
     * @return The original rounded contribution of this K32 packet.
     */
    __device__ __forceinline__ float retainedHCQ8Contribution(
        const uint8_t* weights, const int8_t* codes, float weight_scale, float activation_scale)
    {
        const auto* w = reinterpret_cast<const int4*>(weights);
        const auto* a = reinterpret_cast<const int4*>(codes);
        const int4 wl = w[0], wh = w[1];
        const int4 al = a[0], ah = a[1];
        int lo = __dp4a(wl.x, al.x, 0);
        int hi = __dp4a(wh.x, ah.x, 0);
        lo = __dp4a(wl.y, al.y, lo);
        hi = __dp4a(wh.y, ah.y, hi);
        lo = __dp4a(wl.z, al.z, lo);
        hi = __dp4a(wh.z, ah.z, hi);
        lo = __dp4a(wl.w, al.w, lo);
        hi = __dp4a(wh.w, ah.w, hi);
        return device_native_quantised_contract::singleScaleBlock(lo + hi, weight_scale, activation_scale);
    }

    /** @brief Exact all-18-codebook K32 block dot contribution for one CUDA NativeQuantised matrix. */
    template<uint8_t CodebookId>
    __device__ __forceinline__ float retainedNativeQuantisedBlockContribution(
        const DeviceNativeQuantisedMatrixDesc& desc,
        unsigned linear,
        const int8_t* codes,
        float act_scale)
    {
        using Traits = ::llaminar2::cuda_native_quantised::CodebookTraits<CodebookId>;
        const auto* a4 = reinterpret_cast<const int4*>(codes);
        const int4 lo = a4[0], hi = a4[1];
        const int32_t a[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        const uint8_t* payload = desc.payload + static_cast<std::size_t>(linear) * Traits::payload_bytes;
        int32_t w[8];
        ::llaminar2::cuda_native_quantised::decode_groups_vec<CodebookId>(payload, w);
        const uint16_t primary = static_cast<const uint16_t*>(desc.scales)[linear];
        const uint16_t secondary = Traits::requires_secondary_metadata
            ? static_cast<const uint16_t*>(desc.mins)[linear] : 0u;
        const uint32_t emin = Traits::is_dual_scale_asym
            ? static_cast<const uint32_t*>(desc.emins)[linear] : 0u;

        if constexpr (Traits::is_dual_scale)
        {
            int dot_lo = 0, dot_hi = 0, sum_lo = 0, sum_hi = 0;
#pragma unroll
            for (int g = 0; g < 4; ++g)
            {
                dot_lo = __dp4a(a[g], w[g], dot_lo);
                dot_hi = __dp4a(a[g + 4], w[g + 4], dot_hi);
                sum_lo = __dp4a(a[g], 0x01010101, sum_lo);
                sum_hi = __dp4a(a[g + 4], 0x01010101, sum_hi);
            }
            uint16_t iq1m_qh = 0u;
            int sg0 = 0, sg1 = 0, sg2 = 0, sg3 = 0;
            if constexpr (Traits::is_iq1_m)
            {
                iq1m_qh = static_cast<uint16_t>(payload[4]) | (static_cast<uint16_t>(payload[5]) << 8u);
                sg0 = __dp4a(a[0], 0x01010101, 0) + __dp4a(a[1], 0x01010101, 0);
                sg1 = __dp4a(a[2], 0x01010101, 0) + __dp4a(a[3], 0x01010101, 0);
                sg2 = __dp4a(a[4], 0x01010101, 0) + __dp4a(a[5], 0x01010101, 0);
                sg3 = __dp4a(a[6], 0x01010101, 0) + __dp4a(a[7], 0x01010101, 0);
            }
            return ::llaminar2::cuda_native_quantised::native_quantised_block_contribution_from_reduced_terms_rn<CodebookId>(
                dot_lo, dot_hi, act_scale, primary, secondary, emin,
                sum_lo + sum_hi, sum_lo, sum_hi, iq1m_qh, sg0, sg1, sg2, sg3);
        }
        else
        {
            int dot = 0, sum_a = 0;
#pragma unroll
            for (int g = 0; g < 8; ++g)
            {
                dot = __dp4a(a[g], w[g], dot);
                if constexpr (Traits::is_asymmetric)
                    sum_a = __dp4a(a[g], 0x01010101, sum_a);
            }
            return ::llaminar2::cuda_native_quantised::native_quantised_block_contribution_from_reduced_terms_rn<CodebookId>(
                dot, 0, act_scale, primary, secondary, 0u, sum_a, 0, 0, 0u, 0, 0, 0, 0);
        }
    }

    /** @brief Runtime-dispatched all-18-codebook K32 block dot contribution. */
    static __device__ __forceinline__ float retainedNativeQuantisedBlockDynamic(
        const DeviceNativeQuantisedMatrixDesc& desc,
        unsigned linear,
        const int8_t* codes,
        float act_scale)
    {
        switch (desc.codebook_id)
        {
        case 0:  return retainedNativeQuantisedBlockContribution<0>(desc, linear, codes, act_scale);
        case 4:  return retainedNativeQuantisedBlockContribution<4>(desc, linear, codes, act_scale);
        case 5:  return retainedNativeQuantisedBlockContribution<5>(desc, linear, codes, act_scale);
        case 6:  return retainedNativeQuantisedBlockContribution<6>(desc, linear, codes, act_scale);
        case 7:  return retainedNativeQuantisedBlockContribution<7>(desc, linear, codes, act_scale);
        case 8:  return retainedNativeQuantisedBlockContribution<8>(desc, linear, codes, act_scale);
        case 9:  return retainedNativeQuantisedBlockContribution<9>(desc, linear, codes, act_scale);
        case 10: return retainedNativeQuantisedBlockContribution<10>(desc, linear, codes, act_scale);
        case 11: return retainedNativeQuantisedBlockContribution<11>(desc, linear, codes, act_scale);
        case 12: return retainedNativeQuantisedBlockContribution<12>(desc, linear, codes, act_scale);
        case 13: return retainedNativeQuantisedBlockContribution<13>(desc, linear, codes, act_scale);
        case 14: return retainedNativeQuantisedBlockContribution<14>(desc, linear, codes, act_scale);
        case 15: return retainedNativeQuantisedBlockContribution<15>(desc, linear, codes, act_scale);
        case 16: return retainedNativeQuantisedBlockContribution<16>(desc, linear, codes, act_scale);
        case 17: return retainedNativeQuantisedBlockContribution<17>(desc, linear, codes, act_scale);
        case 19: return retainedNativeQuantisedBlockContribution<19>(desc, linear, codes, act_scale);
        case ::llaminar2::kNativeQuantisedExpandedInt8MinCodebook:
            return retainedNativeQuantisedBlockContribution<::llaminar2::kNativeQuantisedExpandedInt8MinCodebook>(
                desc, linear, codes, act_scale);
        case ::llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook:
            return retainedNativeQuantisedBlockContribution<::llaminar2::kNativeQuantisedIQ4XSSourceScaleCodebook>(
                desc, linear, codes, act_scale);
        default:
            __trap();
            return 0.0f;
        }
    }

    /** @brief General all-18-codebook HC Down partition fold. */
    __device__ __forceinline__ float retainedHCDownPartialGeneral(
        const RetainedHCPrefix& prefix, int column, int partition)
    {
        const auto& down = prefix.down;
        const int first = partition * prefix.down_partitions.blocks_per_partition;
        const int last = min(static_cast<int>(down.blocks_per_row),
            first + prefix.down_partitions.blocks_per_partition);
        float sum = 0.0f;
#pragma unroll 1
        for (int group = first; group < last; ++group)
        {
            const unsigned linear = static_cast<unsigned>(group * down.n + column);
            const float value = retainedNativeQuantisedBlockDynamic(
                down, linear, prefix.norm_codes + group * 32, prefix.norm_scales[group]);
            sum = __fadd_rn(sum, value);
        }
        return sum;
    }

    /** @brief General all-18-codebook HC Up dual-term column fold. */
    __device__ __forceinline__ float retainedHCUpColumnGeneral(
        const GPUQ8ResidualCorrectionArgs& up, int column)
    {
        const auto& weight = up.weights;
        float primary = 0.0f;
        float residual = 0.0f;
#pragma unroll 1
        for (unsigned group = 0; group < weight.blocks_per_row; ++group)
        {
            const unsigned linear = group * static_cast<unsigned>(weight.n) + static_cast<unsigned>(column);
            primary = __fadd_rn(primary, retainedNativeQuantisedBlockDynamic(
                weight, linear, up.primary + group * 32, up.primaryScales[group]));
            residual = __fadd_rn(residual, retainedNativeQuantisedBlockDynamic(
                weight, linear, up.residual + group * 32, up.residualScales[group]));
        }
        return __fadd_rn(primary, residual);
    }

    /**
     * @brief Fold one original serial Down partition for one output, without public scratch.
     * @param prefix Sealed original Down and normalized input operands.
     * @param column Original Down output column.
     * @param partition Original serial partition number.
     * @return The original rounded partial, including positive zero for an empty partition.
     */
    __device__ __forceinline__ float retainedHCDownPartial(const RetainedHCPrefix& prefix,
        int column, int partition)
    {
        const auto& down = prefix.down;
        const int first = partition * prefix.down_partitions.blocks_per_partition;
        const int last = min(static_cast<int>(down.blocks_per_row),
            first + prefix.down_partitions.blocks_per_partition);
        const auto* weight_scales = static_cast<const uint16_t*>(down.scales);
        float sum = 0.0f;
#pragma unroll 1
        for (int group = first; group < last; ++group)
        {
            const int linear = group * down.n + column;
            const float weight_scale = __half2float(__ushort_as_half(weight_scales[linear]));
            const float value = retainedHCQ8Contribution(down.payload + linear * 32,
                prefix.norm_codes + group * 32, weight_scale, prefix.norm_scales[group]);
            sum = __fadd_rn(sum, value);
        }
        return sum;
    }

    /**
     * @brief Fold both original Up terms in increasing K32 order for one branch/channel.
     * @param up Original prepared matrix and two Q8 quantizations produced by Down.
     * @param column Original branch-major output column.
     * @return One rounded Up value after the original add of the complete terms.
     */
    __device__ __forceinline__ float retainedHCUpColumn(const GPUQ8ResidualCorrectionArgs& up, int column)
    {
        const auto& weight = up.weights;
        const auto* weight_scales = static_cast<const uint16_t*>(weight.scales);
        float primary = 0.0f;
        float residual = 0.0f;
#pragma unroll 1
        for (unsigned group = 0; group < weight.blocks_per_row; ++group)
        {
            const unsigned linear = group * weight.n + column;
            const uint8_t* packet = weight.payload + linear * 32;
            const float scale = __half2float(__ushort_as_half(weight_scales[linear]));
            primary = __fadd_rn(primary, retainedHCQ8Contribution(packet, up.primary + group * 32,
                scale, up.primaryScales[group]));
            residual = __fadd_rn(residual, retainedHCQ8Contribution(packet, up.residual + group * 32,
                scale, up.residualScales[group]));
        }
        return __fadd_rn(primary, residual);
    }

    /**
     * @brief Compute one row of the original 4x(4H) FP32 HC injection projection with the exact 128/64/32 tree.
     * @tparam Hidden Physical branch width authenticated by the resident program.
     * @param prefix Sealed original HC prefix carrying normalized output and FP32 injection weights.
     * @param row Output injection row in [0, 4).
     * @param partials CTA-local 256-float scratch reused after the branch-norm phase.
     */
    template<unsigned Hidden>
    __device__ __forceinline__ void retainedHCInjectionRow(
        const RetainedHCPrefix& prefix, unsigned row, float* partials)
    {
        const unsigned tid = threadIdx.x;
        constexpr unsigned k = 4u * Hidden;
        const float* w_row = prefix.injection_weights + static_cast<std::size_t>(row) * k;
        float sum = 0.0f;
        for (unsigned j = tid; j < k; j += 256u)
            sum += prefix.norm.output[j] * w_row[j];
        partials[tid] = sum;
        __syncthreads();
        if (tid < 32u)
        {
            const float low = (partials[tid] + partials[tid + 128u]) +
                              (partials[tid + 64u] + partials[tid + 192u]);
            const float high = (partials[tid + 32u] + partials[tid + 160u]) +
                               (partials[tid + 96u] + partials[tid + 224u]);
            float result = low + high;
#pragma unroll
            for (int stride = 16; stride > 0; stride >>= 1)
                result += __shfl_down_sync(0xffffffffu, result, stride);
            if (tid == 0u)
                prefix.injection_output[row] = result;
        }
        __syncthreads();
    }

    /**
     * @brief Publish the original post-attention mixed row inside an enclosing resident program.
     * @tparam PublishActivated The graph reads the original rounded Down activation.
     * @tparam PublishUp The graph reads the original branch logits.
     * @tparam Hidden Physical branch width authenticated by the resident program.
     * @tparam SupportOptionalWriteAndInject Enable PublishedTrunk (Layer 0) and overlapped 4-CTA FP32 injection.
     * @param prefix Cold-prepared original HC operands and fold.
     * @param shared Borrowed CTA storage, reusable after the final grid join.
     * @param grid Resident native grid, with one row active for every participant.
     * @pre All CTAs enter using 256 threads and at least four native blocks.
     *      Both weight sources are original prepared Q8_0. The Down output has
     *      complete K32 quantizer groups, and the Up output width is 4H.
     */
    template<bool PublishActivated, bool PublishUp, unsigned Hidden,
             bool SupportOptionalWriteAndInject = false, bool GeneralCodebooks = false>
    __device__ __forceinline__ void publishRetainedHCPrefix(const RetainedHCPrefix& prefix,
        RetainedHCPrefixShared<Hidden>& shared, cooperative_groups::grid_group grid,
        uint64_t* profile_ns = nullptr, uint32_t* hc_down_done = nullptr)
    {
        auto mark = [&](int idx) {
            if (profile_ns && blockIdx.x == 0 && threadIdx.x == 0) {
                uint64_t t;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
                profile_ns[idx] = t;
            }
        };
        using namespace ::llaminar2::qwen4exp::branch_norm_native;
        const unsigned tid = threadIdx.x;
        const unsigned lane = tid & 31u;
        const unsigned warp = tid >> 5;

        if (blockIdx.x < 4)
        {
            const unsigned hidden = prefix.norm.hidden;
            const unsigned branch = blockIdx.x;
            const unsigned offset = branch * hidden;
            static_assert(Hidden % 128 == 0);
            float partial = 0.0f;
            if constexpr (SupportOptionalWriteAndInject)
            {
                if (prefix.has_previous_write)
                {
                    const ResidualBranchInput input{prefix.write.residual + offset, prefix.write.block,
                        prefix.write.output + offset, residualGate(prefix.write.gateLogits[branch])};
#pragma unroll 2
                    for (unsigned channel = tid; channel < Hidden; channel += 256)
                    {
                        const float value = input.publish(channel);
                        shared.post[channel] = value;
                        partial = fmaf(value, value, partial);
                    }
                }
                else
                {
                    const float* trunk = prefix.norm.input + offset;
#pragma unroll 2
                    for (unsigned channel = tid; channel < Hidden; channel += 256)
                    {
                        const float value = trunk[channel];
                        shared.post[channel] = value;
                        partial = fmaf(value, value, partial);
                    }
                }
            }
            else
            {
                const ResidualBranchInput input{prefix.write.residual + offset, prefix.write.block,
                    prefix.write.output + offset, residualGate(prefix.write.gateLogits[branch])};
#pragma unroll 2
                for (unsigned channel = tid; channel < Hidden; channel += 256)
                {
                    const float value = input.publish(channel);
                    shared.post[channel] = value;
                    partial = fmaf(value, value, partial);
                }
            }
            const float sum = squareSum(partial, shared.norm, tid, true);
            const float inverse = normalizationInverse(sum, prefix.norm.hidden, prefix.norm.epsilon);
            const auto *gain = reinterpret_cast<const float4 *>(prefix.norm.gain + offset);
            auto *normalized = reinterpret_cast<float4 *>(prefix.norm.output + offset);
            auto *codes = prefix.norm_codes + offset;
            auto *scales = prefix.norm_scales + offset / 32;
            const auto *post = reinterpret_cast<const float4 *>(shared.post);
            for (unsigned quad = tid; quad < Hidden / 4; quad += 256)
            {
                const unsigned first = 4 * quad;
                const float4 value = post[quad];
                const float4 g = gain[quad];
                const auto output = make_float4(
                    normalizedElement(value.x, inverse, g.x),
                    normalizedElement(value.y, inverse, g.y),
                    normalizedElement(value.z, inverse, g.z),
                    normalizedElement(value.w, inverse, g.w));
                normalized[quad] = output;
                retainedHCNormQ8Quad(output, codes + first, scales + first / 32, tid % 8);
            }
        }
        else if constexpr (!GeneralCodebooks)
        {
            // While CTAs 0..3 compute BranchRMSNorm (~8 us), idle CTAs 4..gridDim.x-1 warm
            // the immutable HC Down weight matrix (3.48 MB) into the GPU's L2 cache so the
            // immediately following HC Down phase reads directly from L2 at >2 TB/s with 0 spills.
            const unsigned idle_ctas = gridDim.x - 4u;
            const unsigned idle_tid = (blockIdx.x - 4u) * 256u + tid;
            const unsigned idle_stride = idle_ctas * 256u;
            const unsigned payload_lines =
                (static_cast<unsigned>(prefix.down.blocks_per_row) * static_cast<unsigned>(prefix.down.n) * 32u) >> 7;
            const uint8_t* pay = prefix.down.payload;
#pragma unroll 1
            for (unsigned line = idle_tid; line < payload_lines; line += idle_stride)
            {
                asm volatile("prefetch.global.L2 [%0];" :: "l"(pay + (static_cast<std::size_t>(line) << 7)));
            }
            const unsigned scale_lines =
                (static_cast<unsigned>(prefix.down.blocks_per_row) * static_cast<unsigned>(prefix.down.n) * 2u) >> 7;
            const uint8_t* sc = static_cast<const uint8_t*>(prefix.down.scales);
#pragma unroll 1
            for (unsigned line = idle_tid; line < scale_lines; line += idle_stride)
            {
                asm volatile("prefetch.global.L2 [%0];" :: "l"(sc + (static_cast<std::size_t>(line) << 7)));
            }
        }
        grid.sync();
        mark(1);

        {
        const unsigned partitions = prefix.down_partitions.count;
        const unsigned columns = prefix.down.n;
        const bool ordered = prefix.down_fold == HCDownFold::OrderedPartitions;
        if (ordered)
        {
            if constexpr (!GeneralCodebooks)
            {
                if (columns % 32 == 0 && prefix.down_partitions.blocks_per_partition <= 32)
                {
                    // Unified single-wave multi-partition HC Down + overlapped Injection:
                    // when floor((gridDim.x - inject) / col_tiles) slices fit inside
                    // shared.down_blocks[32][32], each 32-column tile splits across
                    // eff_slices CTAs so HC Down + Injection finish in ONE single wave
                    // with ~37% fewer atomic fences and zero idle warps; otherwise
                    // eff_slices == partitions degrades seamlessly to 1 partition/task.
                    const unsigned col_tiles = columns >> 5;
                    const unsigned bps = static_cast<unsigned>(prefix.down_partitions.blocks_per_partition);
                    const unsigned total_blocks = static_cast<unsigned>(prefix.down.blocks_per_row);
                    const bool has_inject = SupportOptionalWriteAndInject && (prefix.injection_weights != nullptr);
                    const unsigned inject_ctas = has_inject ? 4u : 0u;
                    const bool fuse_down_publish = (hc_down_done != nullptr && col_tiles <= 32u);
                    const unsigned half_lane = lane >> 1;
                    const unsigned half_part = lane & 1u;

                    // Compute single-wave slice schedule on tid==0 into shared.norm[0..5]
                    // so p_begin/p_end/eff_slices/col_tile never consume live registers
                    // across the peak-pressure int4 dp4a loop (keeps <=64 regs, 0 spills).
                    unsigned* s_meta = reinterpret_cast<unsigned*>(shared.norm);
                    if (tid == 0u)
                    {
                        const unsigned avail_down_ctas = (gridDim.x > inject_ctas) ? (gridDim.x - inject_ctas) : 0u;
                        const unsigned cand_slices = (avail_down_ctas >= col_tiles)
                            ? min(partitions, avail_down_ctas / col_tiles) : 0u;
                        const unsigned cand_max_parts = (cand_slices > 0u)
                            ? ((partitions + cand_slices - 1u) / cand_slices) : 0u;
                        const unsigned eff_slices =
                            (fuse_down_publish && cand_slices > 0u && cand_max_parts * bps <= 32u)
                                ? cand_slices : partitions;
                        const unsigned cta_tasks = eff_slices * col_tiles;
                        s_meta[5] = eff_slices;
                        s_meta[6] = cta_tasks;
                        s_meta[7] = cta_tasks + inject_ctas;
                    }
                    __syncthreads();
#pragma unroll 1
                    for (unsigned cta_task = blockIdx.x; cta_task < s_meta[7]; cta_task += gridDim.x)
                    {
                        if constexpr (SupportOptionalWriteAndInject)
                        {
                            if (cta_task >= s_meta[6])
                            {
                                retainedHCInjectionRow<Hidden>(prefix, cta_task - s_meta[6], shared.norm);
                                break;
                            }
                        }
                        if (tid == 0u)
                        {
                            const unsigned eff_slices = s_meta[5];
                            const unsigned slice = cta_task / col_tiles;
                            const unsigned col_tile = cta_task - slice * col_tiles;
                            const unsigned p_begin = (eff_slices == partitions) ? slice : ((slice * partitions) / eff_slices);
                            const unsigned p_end = (eff_slices == partitions) ? (slice + 1u) : (((slice + 1u) * partitions) / eff_slices);
                            const unsigned first = p_begin * bps;
                            const unsigned last = min(total_blocks, p_end * bps);
                            s_meta[0] = col_tile;
                            s_meta[1] = p_begin;
                            s_meta[2] = p_end;
                            s_meta[3] = first;
                            s_meta[4] = (last > first) ? (last - first) : 0u;
                        }
                        __syncthreads();
                        const unsigned col_base = s_meta[0] << 5;
                        const unsigned first = s_meta[3];
                        const unsigned count = s_meta[4];
#pragma unroll 1
                        for (unsigned b = warp; b < count; b += 8u)
                        {
                            const unsigned group = first + b;
                            const unsigned row_base = group * columns + col_base;
                            const int4 a4 = reinterpret_cast<const int4*>(
                                prefix.norm_codes + group * 32u)[half_part];
                            const int4 w0 = reinterpret_cast<const int4*>(
                                prefix.down.payload + row_base * 32u)[lane];
                            const int4 w1 = reinterpret_cast<const int4*>(
                                prefix.down.payload + (row_base + 16u) * 32u)[lane];
                            float act_scale = 0.0f;
                            uint16_t raw_ws0 = 0u, raw_ws1 = 0u;
                            if (half_part == 0u)
                            {
                                const auto* weight_scales = static_cast<const uint16_t*>(prefix.down.scales);
                                act_scale = prefix.norm_scales[group];
                                raw_ws0 = weight_scales[row_base + half_lane];
                                raw_ws1 = weight_scales[row_base + 16u + half_lane];
                            }
                            int d0 = __dp4a(w0.x, a4.x, 0);
                            int d1 = __dp4a(w1.x, a4.x, 0);
                            d0 = __dp4a(w0.y, a4.y, d0);
                            d1 = __dp4a(w1.y, a4.y, d1);
                            d0 = __dp4a(w0.z, a4.z, d0);
                            d1 = __dp4a(w1.z, a4.z, d1);
                            d0 = __dp4a(w0.w, a4.w, d0);
                            d1 = __dp4a(w1.w, a4.w, d1);
                            const int full0 = d0 + __shfl_xor_sync(0xffffffffu, d0, 1);
                            const int full1 = d1 + __shfl_xor_sync(0xffffffffu, d1, 1);
                            if (half_part == 0u)
                            {
                                const float ws0 = __half2float(__ushort_as_half(raw_ws0));
                                const float ws1 = __half2float(__ushort_as_half(raw_ws1));
                                shared.down_blocks[b][half_lane] =
                                    device_native_quantised_contract::singleScaleBlock(full0, ws0, act_scale);
                                shared.down_blocks[b][16u + half_lane] =
                                    device_native_quantised_contract::singleScaleBlock(full1, ws1, act_scale);
                            }
                        }
                        __syncthreads();
                        if (warp == 0u)
                        {
                            const unsigned col_tile = s_meta[0];
                            const unsigned p_begin = s_meta[1];
                            const unsigned p_end = s_meta[2];
                            const unsigned first_b = s_meta[3];
#pragma unroll 1
                            for (unsigned p = p_begin; p < p_end; ++p)
                            {
                                const unsigned pf = p * bps;
                                const unsigned pl = min(total_blocks, pf + bps);
                                const unsigned b_off = pf - first_b;
                                const unsigned b_cnt = (pl > pf) ? (pl - pf) : 0u;
                                float sum = 0.0f;
#pragma unroll 1
                                for (unsigned b = 0u; b < b_cnt; ++b)
                                    sum = __fadd_rn(sum, shared.down_blocks[b_off + b][lane]);
                                prefix.down_partials[p * columns + col_base + lane] = sum;
                            }
                            if (fuse_down_publish)
                            {
                                __threadfence();
                                unsigned finished = (lane == 0u) ? (atomicAdd(hc_down_done + col_tile, 1u) + 1u) : 0u;
                                finished = __shfl_sync(0xffffffffu, finished, 0);
                                if (finished == s_meta[5])
                                {
                                    const unsigned column = col_base + lane;
                                    const float* dp_col = prefix.down_partials + column;
                                    float total = 0.0f;
                                    unsigned part = 0u;
#pragma unroll 1
                                    for (; part + 4u <= partitions; part += 4u)
                                    {
                                        const float v0 = __ldcg(dp_col + (part + 0u) * columns);
                                        const float v1 = __ldcg(dp_col + (part + 1u) * columns);
                                        const float v2 = __ldcg(dp_col + (part + 2u) * columns);
                                        const float v3 = __ldcg(dp_col + (part + 3u) * columns);
                                        total = __fadd_rn(total, v0);
                                        total = __fadd_rn(total, v1);
                                        total = __fadd_rn(total, v2);
                                        total = __fadd_rn(total, v3);
                                    }
#pragma unroll 1
                                    for (; part < partitions; ++part)
                                        total = __fadd_rn(total, __ldcg(dp_col + part * columns));
                                    prefix.raw_down[column] = total;
                                    hc_down_publication::publish<hc_down_publication::OriginalScalePublication, PublishActivated>(
                                        total, prefix.down_publication, column);
                                }
                            }
                        }
                        if (cta_task + gridDim.x < s_meta[7])
                            __syncthreads();
                    }
                }
                else
                {
                    const unsigned global_thread = blockIdx.x * blockDim.x + tid;
                    const unsigned grid_threads = gridDim.x * blockDim.x;
                    for (unsigned task = global_thread; task < partitions * columns; task += grid_threads)
                    {
                        const unsigned column = task % columns;
                        const unsigned partition = task / columns;
                        prefix.down_partials[task] = retainedHCDownPartial(prefix, column, partition);
                    }
                    if constexpr (SupportOptionalWriteAndInject)
                    {
                        if (prefix.injection_weights != nullptr && blockIdx.x < 4u)
                            retainedHCInjectionRow<Hidden>(prefix, blockIdx.x, shared.norm);
                    }
                }
            }
            else
            {
                const unsigned global_thread = blockIdx.x * blockDim.x + tid;
                const unsigned grid_threads = gridDim.x * blockDim.x;
                for (unsigned task = global_thread; task < partitions * columns; task += grid_threads)
                {
                    const unsigned column = task % columns;
                    const unsigned partition = task / columns;
                    prefix.down_partials[task] = retainedHCDownPartialGeneral(prefix, column, partition);
                }
                if constexpr (SupportOptionalWriteAndInject)
                {
                    if (prefix.injection_weights != nullptr && blockIdx.x < 4u)
                        retainedHCInjectionRow<Hidden>(prefix, blockIdx.x, shared.norm);
                }
            }
            grid.sync();
        }
        else if constexpr (SupportOptionalWriteAndInject)
        {
            if (prefix.injection_weights != nullptr && blockIdx.x < 4u)
                retainedHCInjectionRow<Hidden>(prefix, blockIdx.x, shared.norm);
        }
        mark(2);

        const bool down_already_published =
            (!GeneralCodebooks) &&
            (prefix.down_fold == HCDownFold::OrderedPartitions) &&
            (prefix.down.n % 32 == 0) &&
            (prefix.down_partitions.blocks_per_partition <= 32) &&
            (hc_down_done != nullptr) &&
            (static_cast<unsigned>(prefix.down.n) <= 1024u);
        if (!down_already_published)
        {
            // Every publishing warp owns a complete quantizer packet. It folds
            // the original partition bank in order, or computes the single admitted
            // direct partition without materializing a partial or joining the grid.
            const unsigned global_thread = blockIdx.x * blockDim.x + tid;
            const unsigned grid_threads = gridDim.x * blockDim.x;
            for (unsigned first = (global_thread / 32) * 32; first < columns; first += grid_threads)
            {
                const unsigned column = first + lane;
                float sum = 0.0f;
                if (ordered)
                {
                    for (unsigned part = 0; part < partitions; ++part)
                        sum = __fadd_rn(sum, prefix.down_partials[part * columns + column]);
                }
                else
                {
                    if constexpr (GeneralCodebooks)
                        sum = retainedHCDownPartialGeneral(prefix, column, 0);
                    else
                        sum = retainedHCDownPartial(prefix, column, 0);
                }
                prefix.raw_down[column] = sum;
                hc_down_publication::publish<hc_down_publication::OriginalScalePublication, PublishActivated>(
                    sum, prefix.down_publication, column);
            }
            grid.sync();
        }
        }
        mark(3);

        const unsigned hidden = prefix.norm.hidden;
        if constexpr (!GeneralCodebooks)
        {
        if (prefix.up.weights.blocks_per_row == 10 && hidden % 16 == 0)
        {
            // 100% coalesced warp-split 160-CTA HC Up: each CTA owns 16 contiguous channels
            // across all 4 branches, issuing all 5 x 512-byte int4 warp loads in flight via
            // full unroll (~89% of GDDR6X roofline on SM86).
            const unsigned half_lane = lane >> 1;
            const unsigned half_part = lane & 1u;
            const unsigned channel_tiles = hidden >> 4;
#pragma unroll 1
            for (unsigned tile = blockIdx.x; tile < channel_tiles; tile += gridDim.x)
            {
                const unsigned first_channel = tile << 4;
#pragma unroll 2
                for (unsigned step = 0u; step < 5u; ++step)
                {
                    const unsigned task = step * 8u + warp;
                    const unsigned branch = (task * 205u) >> 11;
                    const unsigned group = task - branch * 10u;
                    const unsigned col_base = branch * hidden + first_channel;
                    const int4 w4 = reinterpret_cast<const int4*>(
                        prefix.up.weights.payload + (group * prefix.up.weights.n + col_base) * 32u)[lane];
                    const int4 p4 = reinterpret_cast<const int4*>(prefix.up.primary + group * 32u)[half_part];
                    const int4 r4 = reinterpret_cast<const int4*>(prefix.up.residual + group * 32u)[half_part];
                    int pd = __dp4a(w4.x, p4.x, 0);
                    int rd = __dp4a(w4.x, r4.x, 0);
                    pd = __dp4a(w4.y, p4.y, pd);
                    rd = __dp4a(w4.y, r4.y, rd);
                    pd = __dp4a(w4.z, p4.z, pd);
                    rd = __dp4a(w4.z, r4.z, rd);
                    pd = __dp4a(w4.w, p4.w, pd);
                    rd = __dp4a(w4.w, r4.w, rd);
                    const int p_full = pd + __shfl_xor_sync(0xffffffffu, pd, 1);
                    const int r_full = rd + __shfl_xor_sync(0xffffffffu, rd, 1);
                    if (half_part == 0u)
                    {
                        const unsigned linear = group * prefix.up.weights.n + col_base + half_lane;
                        const float ws = __half2float(__ushort_as_half(
                            static_cast<const uint16_t*>(prefix.up.weights.scales)[linear]));
                        shared.up.primary[branch][group][half_lane] =
                            device_native_quantised_contract::singleScaleBlock(p_full, ws, prefix.up.primaryScales[group]);
                        shared.up.residual[branch][group][half_lane] =
                            device_native_quantised_contract::singleScaleBlock(r_full, ws, prefix.up.residualScales[group]);
                    }
                }
                __syncthreads();
                if (tid < 64u)
                {
                    const unsigned branch = tid >> 4;
                    const unsigned ch = tid & 15u;
                    float primary = 0.0f, residual = 0.0f;
#pragma unroll
                    for (unsigned group = 0u; group < 10u; ++group)
                    {
                        primary = __fadd_rn(primary, shared.up.primary[branch][group][ch]);
                        residual = __fadd_rn(residual, shared.up.residual[branch][group][ch]);
                    }
                    const float logit = __fadd_rn(primary, residual);
                    if constexpr (PublishUp)
                        prefix.up.output[branch * hidden + first_channel + ch] = logit;
                    shared.up.logit[branch][ch] = logit;
                }
                __syncthreads();
                if (tid < 16u)
                {
                    float mixed = 0.0f;
#pragma unroll
                    for (unsigned branch = 0u; branch < 4u; ++branch)
                        mixed = appendReadBranch(mixed, shared.up.logit[branch][tid],
                            prefix.norm.output[branch * hidden + first_channel + tid]);
                    prefix.mixed[first_channel + tid] = mixed * 0.25f;
                }
                if (tile + gridDim.x < channel_tiles)
                    __syncthreads();
            }
            grid.sync();
            return;
        }
        }

        // General fallback geometry for non-10-block HC Up widths or arbitrary codebooks.
        constexpr unsigned branch_width = retained_hc_read_warp_channels;
        const unsigned branch = lane / branch_width;
        const unsigned read_warp = tid / 32 * gridDim.x + blockIdx.x;
        const unsigned grid_threads = gridDim.x * blockDim.x;
        for (unsigned first = read_warp * branch_width; first < hidden;
             first += grid_threads / 32 * branch_width)
        {
            const unsigned channel = first + lane % branch_width;
            const unsigned column = branch * hidden + channel;
            float logit;
            if constexpr (GeneralCodebooks)
                logit = retainedHCUpColumnGeneral(prefix.up, column);
            else
                logit = retainedHCUpColumn(prefix.up, column);
            if constexpr (PublishUp)
                prefix.up.output[column] = logit;
            float mixed = 0.0f;
#pragma unroll
            for (unsigned current = 0; current < 4; ++current)
            {
                const float term = __shfl_sync(0xffffffffu, logit, current * branch_width + lane % branch_width);
                if (branch == 0)
                    mixed = appendReadBranch(mixed, term, prefix.norm.output[current * hidden + channel]);
            }
            if (branch == 0)
                prefix.mixed[channel] = mixed * 0.25f;
        }
        grid.sync();
    }
}
