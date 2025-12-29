/**
 * @file WoProjectionVNNIRef.cpp
 * @brief Reference implementation of Wo projection using VPDPWSSD (INT16×INT16)
 *
 * ALL INTEGER DOMAIN - maximum precision by using INT16 context (not INT8).
 * Weights are sign-extended from INT8 to INT16 at runtime (lossless).
 * Output is Q16_1 format for direct native Q16_1 + Q16_1 residual addition.
 *
 * Pipeline: INT32 context → requant → INT16 → VPDPWSSD → INT32 → requant → Q16_1
 * Then: Q16_1(projection) + Q16_1(residual) → Q16_1 (via simd::q16_1_add_q16_1)
 */

#include "WoProjectionVNNIRef.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // FP16 Utilities (for scale storage only, not data conversion)
    // ============================================================================

    static inline uint16_t fp32_to_fp16(float f)
    {
        uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));

        uint32_t sign = (bits >> 16) & 0x8000;
        int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (bits >> 13) & 0x3FF;

        if (exp <= 0)
        {
            return static_cast<uint16_t>(sign);
        }
        else if (exp >= 31)
        {
            return static_cast<uint16_t>(sign | 0x7C00);
        }

        return static_cast<uint16_t>(sign | (exp << 10) | mant);
    }

    // ============================================================================
    // Integer Domain Utilities
    // ============================================================================

    int64_t int32_maxabs(const int32_t *data, int n)
    {
        int64_t maxabs = 0;
        for (int i = 0; i < n; ++i)
        {
            int64_t abs_val = std::abs(static_cast<int64_t>(data[i]));
            maxabs = std::max(maxabs, abs_val);
        }
        return maxabs;
    }

    void sign_extend_int8_to_int16(
        const int8_t *int8_weights,
        int n,
        int16_t *int16_weights)
    {
        // Simple sign extension: INT8 → INT16 (lossless)
        // In SIMD, this would be vpmovsxbw
        for (int i = 0; i < n; ++i)
        {
            int16_weights[i] = static_cast<int16_t>(int8_weights[i]);
        }
    }

    int32_t quantize_int32_to_int16(
        const int32_t *int32_input,
        int n,
        float scale,
        int16_t *int16_output)
    {
        // scale = maxabs / 32767, so inv_scale = 32767 / maxabs
        const float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;
        int64_t sum = 0;

        for (int i = 0; i < n; ++i)
        {
            // Convert INT32 to INT16 by dividing by scale
            float scaled = static_cast<float>(int32_input[i]) * inv_scale;
            int16_t q = static_cast<int16_t>(std::round(std::clamp(scaled, -32767.0f, 32767.0f)));
            int16_output[i] = q;
            sum += q;
        }

        return static_cast<int32_t>(std::clamp(sum, static_cast<int64_t>(INT32_MIN),
                                               static_cast<int64_t>(INT32_MAX)));
    }

    // ============================================================================
    // Context Requantization (INT32 → INT16 for VPDPWSSD)
    // ============================================================================

    void requantize_int32_to_int16_context(
        const int32_t *int32_input,
        int n,
        const int32_t *weight_sums,
        const float *v_scales,
        int head_dim,
        int16_t *int16_output,
        float *out_combined_scale)
    {
        // Calculate true float values to find global maxabs
        // true_val = int32_val * v_scale / weight_sum
        float global_maxabs = 0.0f;

        const int safe_head_dim = (head_dim > 0) ? head_dim : n;

        // Per-element scaling: element i belongs to head (i / head_dim).
        // Each element also has an effective V scale (typically per 32-wide Q16_1 block).
        for (int idx = 0; idx < n; ++idx)
        {
            const int head = idx / safe_head_dim;
            const int32_t ws = (weight_sums) ? weight_sums[head] : 1;
            const float inv_ws = (ws > 0) ? 1.0f / static_cast<float>(ws) : 0.0f;
            const float v_scale = (v_scales) ? v_scales[idx] : 1.0f;

            float val = static_cast<float>(int32_input[idx]) * inv_ws * v_scale;
            global_maxabs = std::max(global_maxabs, std::abs(val));
        }

        // Determine common scale to map maxabs to 32767
        // S_common = maxabs / 32767
        float common_scale = (global_maxabs > 0.0f) ? global_maxabs / 32767.0f : 1.0f;
        float inv_common_scale = (common_scale > 0.0f) ? 1.0f / common_scale : 0.0f;

        // Quantize to INT16 using per-element scaling
        // int16_val = round((int32_val / weight_sum(head) * v_scale(idx)) / common_scale)
        for (int idx = 0; idx < n; ++idx)
        {
            const int head = idx / safe_head_dim;
            const int32_t ws = (weight_sums) ? weight_sums[head] : 1;
            const float inv_ws = (ws > 0) ? 1.0f / static_cast<float>(ws) : 0.0f;
            const float v_scale = (v_scales) ? v_scales[idx] : 1.0f;

            float val = static_cast<float>(int32_input[idx]) * inv_ws * v_scale * inv_common_scale;
            int16_t q = static_cast<int16_t>(std::round(std::clamp(val, -32767.0f, 32767.0f)));
            int16_output[idx] = q;
        }

        if (out_combined_scale)
        {
            *out_combined_scale = common_scale;
        }
    }

    // ============================================================================
    // Output Requantization (FP32 → Q16_1)
    // ============================================================================

    void requantize_fp32_to_q16_1(
        const float *fp32_output,
        int n,
        const float *bias,
        Q16_1Block *output)
    {
        constexpr int BLOCK_SIZE = 32;
        const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const int start = b * BLOCK_SIZE;
            const int end = std::min(start + BLOCK_SIZE, n);
            const int count = end - start;

            // Find maxabs in this block (with bias added if present)
            float block_maxabs = 0.0f;
            for (int i = 0; i < count; ++i)
            {
                float val = fp32_output[start + i];
                if (bias)
                {
                    val += bias[start + i];
                }
                block_maxabs = std::max(block_maxabs, std::abs(val));
            }

            // Compute Q16_1 scale (value = scale * qs)
            float block_scale = (block_maxabs > 0.0f) ? block_maxabs / 32767.0f : 1.0f;
            float inv_scale = (block_scale > 0.0f) ? 1.0f / block_scale : 0.0f;

            // Quantize FP32 → INT16
            Q16_1Block &blk = output[b];
            blk.d = block_scale;
            int64_t sum_qs = 0;

            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                if (i < count)
                {
                    float val = fp32_output[start + i];
                    if (bias)
                    {
                        val += bias[start + i];
                    }
                    int16_t q = static_cast<int16_t>(std::round(val * inv_scale));
                    q = std::max(static_cast<int16_t>(-32767), std::min(static_cast<int16_t>(32767), q));
                    blk.qs[i] = q;
                    sum_qs += q;
                }
                else
                {
                    blk.qs[i] = 0;
                }
            }
            blk.sum_qs = static_cast<int32_t>(sum_qs);
        }
    }

    // ============================================================================
    // Output Requantization (INT32 → Q16_1)
    // ============================================================================

    void requantize_int32_to_q16_1(
        const int32_t *int32_output,
        int n,
        float combined_scale,
        const float *bias,
        Q16_1Block *output)
    {
        constexpr int BLOCK_SIZE = 32;
        const int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for (int b = 0; b < num_blocks; ++b)
        {
            const int start = b * BLOCK_SIZE;
            const int end = std::min(start + BLOCK_SIZE, n);
            const int count = end - start;

            // Work with a temporary buffer that includes bias
            alignas(32) int32_t adjusted[BLOCK_SIZE] = {0};
            for (int i = 0; i < count; ++i)
            {
                adjusted[i] = int32_output[start + i];
                // If bias exists, scale it to INT32 range and add
                if (bias && combined_scale > 0.0f)
                {
                    int32_t bias_int = static_cast<int32_t>(std::round(bias[start + i] / combined_scale));
                    adjusted[i] += bias_int;
                }
            }

            // Find maxabs in this block
            int64_t block_maxabs = 0;
            for (int i = 0; i < count; ++i)
            {
                int64_t abs_val = std::abs(static_cast<int64_t>(adjusted[i]));
                block_maxabs = std::max(block_maxabs, abs_val);
            }

            // Compute Q16_1 scale
            float block_scale = (block_maxabs > 0) ? static_cast<float>(block_maxabs) / 32767.0f : 1.0f;
            float q16_scale = block_scale * combined_scale;

            // Quantize INT32 → INT16
            int64_t sum_qs = 0;
            for (int i = 0; i < BLOCK_SIZE; ++i)
            {
                if (i < count)
                {
                    float scaled = static_cast<float>(adjusted[i]) / block_scale;
                    int16_t q = static_cast<int16_t>(std::round(std::clamp(scaled, -32767.0f, 32767.0f)));
                    output[b].qs[i] = q;
                    sum_qs += q;
                }
                else
                {
                    output[b].qs[i] = 0;
                }
            }

            output[b].d = q16_scale;
            output[b].sum_qs = static_cast<int32_t>(sum_qs);
        }
    }

    // ============================================================================
    // GEMV Variant (Decode) - VPDPWSSD (INT16×INT16) → Q16_1 Output
    // ============================================================================

    void wo_projection_vpdpwssd_to_q16_1_gemv(
        const WoProjectionVNNIParams &params,
        const IntegerContext &context,
        Q16_1Projection &output)
    {
        const int input_dim = params.input_dim;
        const int d_model = params.d_model;
        const auto *packed = params.Wo_packed;

        if (!packed || !context.int32_data || !output.blocks)
        {
            return;
        }

        // Step 1: Requantize INT32 context to INT16 (keeps 16 bits of precision!)
        std::vector<int16_t> context_int16(input_dim);
        float context_scale = 0.0f;

        requantize_int32_to_int16_context(
            context.int32_data,
            context.count,
            context.weight_sums,
            context.v_scales,
            context.head_dim,
            context_int16.data(),
            &context_scale);

        // Step 2: VPDPWSSD GEMV (INT16 × INT16 → INT32)
        // We sign-extend INT8 weights to INT16 on the fly
        const int K = packed->K;
        const int N = packed->N;
        const int K_blocks = (K + 31) / 32;
        const int N_padded = (N + 63) / 64 * 64;

        // Compute per-dimension FP32 output (INT32 * per-dim scale)
        // This is needed because each output dimension has its own weight scale
        std::vector<float> fp32_output(d_model);

        // For each output dimension
        for (int n = 0; n < d_model; ++n)
        {
            float fp32_acc = 0.0f;

            // For each K position
            for (int k = 0; k < input_dim; ++k)
            {
                // Get INT16 context value (already requantized)
                int16_t ctx_val = context_int16[k];

                // Extract weight from packed layout [N/64][K/4][64][4]
                // and sign-extend INT8 → INT16
                int n_blk = n / 64;
                int n_rem = n % 64;
                int k_group = k / 4;
                int k_rem = k % 4;
                size_t pack_idx = static_cast<size_t>(n_blk) * (K * 64) + static_cast<size_t>(k_group) * (64 * 4) + static_cast<size_t>(n_rem) * 4 + k_rem;

                int8_t w_int8 = packed->packed_data[pack_idx];
                int16_t w_int16 = static_cast<int16_t>(w_int8); // Sign-extend!

                // Get block scale
                int k_blk = k / 32;
                float blk_scale = packed->scales[k_blk * N_padded + n];

                // Accumulate with correct scale
                fp32_acc += static_cast<float>(ctx_val) * static_cast<float>(w_int16) * blk_scale;
            }

            // Apply context scale
            fp32_output[n] = fp32_acc * context_scale;
        }

        // Step 3: Requantize FP32 → Q16_1
        // Use the new FP32→Q16_1 requantization path
        requantize_fp32_to_q16_1(
            fp32_output.data(),
            d_model,
            params.bias,
            output.blocks);

        output.count = d_model;
    }

    // ============================================================================
    // GEMM Variant (Prefill) - VPDPWSSD → Q16_1 Output
    // ============================================================================

    void wo_projection_vpdpwssd_to_q16_1_gemm(
        const WoProjectionVNNIParams &params,
        const IntegerContext *contexts,
        int seq_len,
        Q16_1Projection *outputs)
    {
        // Process each row independently (reference implementation)
        for (int row = 0; row < seq_len; ++row)
        {
            wo_projection_vpdpwssd_to_q16_1_gemv(params, contexts[row], outputs[row]);
        }
    }

} // namespace llaminar2::kernels::q16_1::microkernels
