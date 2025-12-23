/**
 * @file WoProjection.cpp
 * @brief Reference and SIMD implementations of Wo output projection
 * @author David Sanftenberg
 */

#include "WoProjection.h"
#include "../../../../../tensors/FP16Utils.h"
#include "../../../../../tensors/SIMDHelpers.h"
#include "../../../../../utils/CPUFeatures.h"

#include <cstring>
#include <vector>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

namespace llaminar::v2::kernels::microkernels
{

    void wo_projection_fp32_ref(const WoProjectionParams &params)
    {
        const float *wo_fp32 = static_cast<const float *>(params.wo_weights);
        const int wo_row_stride = params.n_heads * params.head_dim; // Stride for one output row
        const int head_offset = params.head_idx * params.head_dim;  // Offset within row for this head

        for (int out_col = 0; out_col < params.d_model; ++out_col)
        {
            // Compute dot product: context · Wo[out_col, head_offset:head_offset+head_dim]
            float dot = 0.0f;
            const float *wo_row = wo_fp32 + out_col * wo_row_stride + head_offset;

            for (int d = 0; d < params.head_dim; ++d)
            {
                dot += params.context[d] * wo_row[d];
            }

            if (params.accumulate)
            {
                params.output[out_col] += dot;
            }
            else
            {
                params.output[out_col] = dot;
            }
        }
    }

    void wo_projection_fp16_ref(const WoProjectionParams &params)
    {
        const uint16_t *wo_fp16 = static_cast<const uint16_t *>(params.wo_weights);
        const int wo_row_stride = params.n_heads * params.head_dim;
        const int head_offset = params.head_idx * params.head_dim;

        for (int out_col = 0; out_col < params.d_model; ++out_col)
        {
            float dot = 0.0f;
            const uint16_t *wo_row = wo_fp16 + out_col * wo_row_stride + head_offset;

            for (int d = 0; d < params.head_dim; ++d)
            {
                // Dequantize FP16 on-the-fly
                float wo_val = llaminar2::fp16_to_fp32(wo_row[d]);
                dot += params.context[d] * wo_val;
            }

            if (params.accumulate)
            {
                params.output[out_col] += dot;
            }
            else
            {
                params.output[out_col] = dot;
            }
        }
    }

    void wo_projection_bf16_ref(const WoProjectionParams &params)
    {
        const uint16_t *wo_bf16 = static_cast<const uint16_t *>(params.wo_weights);
        const int wo_row_stride = params.n_heads * params.head_dim;
        const int head_offset = params.head_idx * params.head_dim;

        for (int out_col = 0; out_col < params.d_model; ++out_col)
        {
            float dot = 0.0f;
            const uint16_t *wo_row = wo_bf16 + out_col * wo_row_stride + head_offset;

            for (int d = 0; d < params.head_dim; ++d)
            {
                // Dequantize BF16 on-the-fly (simple shift, very fast)
                float wo_val = llaminar2::simd::bf16_to_fp32(wo_row[d]);
                dot += params.context[d] * wo_val;
            }

            if (params.accumulate)
            {
                params.output[out_col] += dot;
            }
            else
            {
                params.output[out_col] = dot;
            }
        }
    }

    void wo_projection_q8_ref(const WoProjectionParams &params)
    {
        // Q8_1 Wo layout: [d_model, (n_heads * head_dim) / 32 blocks]
        // Each row of Wo has (n_heads * head_dim) / 32 Q8_1 blocks

        const Q8_1Block *wo_q8 = static_cast<const Q8_1Block *>(params.wo_weights);
        const int blocks_per_row = (params.n_heads * params.head_dim) / 32;
        const int head_block_start = (params.head_idx * params.head_dim) / 32;
        const int blocks_for_head = params.head_dim / 32;
        const int elem_offset_in_block = (params.head_idx * params.head_dim) % 32;

        // Dequantize the relevant portion of Wo for this head into a buffer
        // This is suboptimal but correct - can optimize later
        std::vector<float> wo_slice(params.d_model * params.head_dim);

        for (int out_col = 0; out_col < params.d_model; ++out_col)
        {
            const Q8_1Block *row_blocks = wo_q8 + out_col * blocks_per_row + head_block_start;
            float *wo_row = wo_slice.data() + out_col * params.head_dim;

            int d = 0;
            for (int b = 0; b < blocks_for_head; ++b)
            {
                float scale = llaminar2::fp16_to_fp32(row_blocks[b].d);
                for (int i = 0; i < 32 && d < params.head_dim; ++i, ++d)
                {
                    wo_row[d] = static_cast<float>(row_blocks[b].qs[i]) * scale;
                }
            }
        }

        // Now do the projection with dequantized weights
        for (int out_col = 0; out_col < params.d_model; ++out_col)
        {
            float dot = 0.0f;
            const float *wo_row = wo_slice.data() + out_col * params.head_dim;

            for (int d = 0; d < params.head_dim; ++d)
            {
                dot += params.context[d] * wo_row[d];
            }

            if (params.accumulate)
            {
                params.output[out_col] += dot;
            }
            else
            {
                params.output[out_col] = dot;
            }
        }
    }

    void wo_projection_ref(const WoProjectionParams &params)
    {
        switch (params.wo_type)
        {
        case WoWeightType::FP32:
            wo_projection_fp32_ref(params);
            break;
        case WoWeightType::FP16:
            wo_projection_fp16_ref(params);
            break;
        case WoWeightType::BF16:
            wo_projection_bf16_ref(params);
            break;
        case WoWeightType::Q8_1:
            wo_projection_q8_ref(params);
            break;
        default:
            // Unsupported type - fall back to FP32
            wo_projection_fp32_ref(params);
            break;
        }
    }

#if defined(__AVX512F__)

    void wo_projection_fp32_avx512(const WoProjectionParams &params)
    {
        const float *wo_fp32 = static_cast<const float *>(params.wo_weights);
        const int wo_row_stride = params.n_heads * params.head_dim;
        const int head_offset = params.head_idx * params.head_dim;

        // Process multiple output columns in parallel for better cache utilization
        // For each output column, compute dot product of context with Wo row

        for (int out_col = 0; out_col < params.d_model; ++out_col)
        {
            const float *wo_row = wo_fp32 + out_col * wo_row_stride + head_offset;

            __m512 acc = _mm512_setzero_ps();

            int d = 0;
            // Vectorized loop (16 floats at a time)
            for (; d + 16 <= params.head_dim; d += 16)
            {
                __m512 ctx = _mm512_loadu_ps(params.context + d);
                __m512 wo = _mm512_loadu_ps(wo_row + d);
                acc = _mm512_fmadd_ps(ctx, wo, acc);
            }

            // Horizontal sum
            float dot = _mm512_reduce_add_ps(acc);

            // Scalar tail
            for (; d < params.head_dim; ++d)
            {
                dot += params.context[d] * wo_row[d];
            }

            if (params.accumulate)
            {
                params.output[out_col] += dot;
            }
            else
            {
                params.output[out_col] = dot;
            }
        }
    }

    void wo_projection_q8_avx512(const WoProjectionParams &params)
    {
        // For now, use reference implementation
        // TODO: Optimize with on-the-fly dequantization in SIMD
        wo_projection_q8_ref(params);
    }

#else

    void wo_projection_fp32_avx512(const WoProjectionParams &params)
    {
        wo_projection_fp32_ref(params);
    }

    void wo_projection_q8_avx512(const WoProjectionParams &params)
    {
        wo_projection_q8_ref(params);
    }

#endif

    void wo_projection(const WoProjectionParams &params)
    {
#if defined(__AVX512F__)
        if (llaminar2::cpu_supports_avx512())
        {
            switch (params.wo_type)
            {
            case WoWeightType::FP32:
                wo_projection_fp32_avx512(params);
                return;
            case WoWeightType::Q8_1:
                wo_projection_q8_avx512(params);
                return;
            case WoWeightType::FP16:
            case WoWeightType::BF16:
                // Use reference for now - can add SIMD versions later
                break;
            default:
                break;
            }
        }
#endif
        wo_projection_ref(params);
    }

    const void *get_wo_head_slice(
        const void *wo_weights,
        WoWeightType wo_type,
        int head_idx,
        int head_dim,
        int n_heads,
        int d_model)
    {
        switch (wo_type)
        {
        case WoWeightType::FP32:
        {
            const float *wo_fp32 = static_cast<const float *>(wo_weights);
            // Return pointer to start of this head's slice in first row
            // Caller should use stride to access subsequent rows
            return wo_fp32 + head_idx * head_dim;
        }
        case WoWeightType::FP16:
        case WoWeightType::BF16:
        {
            // FP16 and BF16 have same layout as FP32 but with uint16_t elements
            const uint16_t *wo_16 = static_cast<const uint16_t *>(wo_weights);
            return wo_16 + head_idx * head_dim;
        }
        case WoWeightType::Q8_1:
        {
            const Q8_1Block *wo_q8 = static_cast<const Q8_1Block *>(wo_weights);
            int blocks_per_row = (n_heads * head_dim) / 32;
            int head_block_start = (head_idx * head_dim) / 32;
            return wo_q8 + head_block_start;
        }
        default:
            return wo_weights;
        }
    }

} // namespace llaminar::v2::kernels::microkernels
