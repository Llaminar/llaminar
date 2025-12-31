/**
 * @file WoProjection.cpp
 * @brief Integer Wo projection microkernel implementation
 *
 * @see WoProjection.h for algorithm description
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#include "WoProjection.h"
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace llaminar2::kernels::q16_1::microkernels
{

    // ============================================================================
    // Normalization: INT32 → INT16
    // ============================================================================

    void q16_context_normalize_to_int16(
        const int32_t *context_int32,
        int16_t *context_int16,
        float &context_scale,
        int num_elements)
    {
        // Find max absolute value for scaling
        int32_t max_abs = 1; // Avoid division by zero
        for (int i = 0; i < num_elements; ++i)
        {
            int32_t abs_val = std::abs(context_int32[i]);
            if (abs_val > max_abs)
                max_abs = abs_val;
        }

        // Compute scale to fit in INT16 range
        // We use 32767 (not 32768) to have symmetric range
        constexpr int32_t INT16_MAX_VAL = 32767;
        context_scale = static_cast<float>(max_abs) / static_cast<float>(INT16_MAX_VAL);

        // Normalize to INT16
        float inv_scale = static_cast<float>(INT16_MAX_VAL) / static_cast<float>(max_abs);
        for (int i = 0; i < num_elements; ++i)
        {
            float scaled = context_int32[i] * inv_scale;
            // Clamp to INT16 range with rounding
            int32_t rounded = static_cast<int32_t>(std::round(scaled));
            context_int16[i] = static_cast<int16_t>(
                std::clamp(rounded, -32768, 32767));
        }
    }

    // ============================================================================
    // Single Row Wo Projection (GEMV) - Tiled Version
    // ============================================================================

    /**
     * @brief Tiled GEMV microkernel for Wo projection.
     *
     * Processes K_tile elements of the reduction at a time to improve cache locality.
     */
    template <typename BlockType>
    void q16_wo_row_gemv_tiled(
        const int16_t *context_int16,
        const BlockType *Wo,
        int32_t &output_int32,
        int input_dim,
        int blocks_per_input,
        int K_tile)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        output_int32 = 0;

        // Process in K_tile chunks for better cache locality
        for (int k_start = 0; k_start < input_dim; k_start += K_tile)
        {
            int k_end = std::min(k_start + K_tile, input_dim);

            // Determine which blocks this tile spans
            int b_start = k_start / BLOCK_SIZE;
            int b_end = (k_end + BLOCK_SIZE - 1) / BLOCK_SIZE;

            // Process blocks in this tile
            for (int b = b_start; b < b_end && b < blocks_per_input; ++b)
            {
                const int16_t *wo_data = Wo[b].qs;

                int block_start = b * BLOCK_SIZE;
                int block_end = std::min(block_start + BLOCK_SIZE, input_dim);

                // Clip to current tile
                int start = std::max(block_start, k_start);
                int end = std::min(block_end, k_end);
                int offset_in_block = start - block_start;

                // INT16 × INT16 → INT32 accumulation
                for (int i = start; i < end; ++i)
                {
                    output_int32 += static_cast<int32_t>(context_int16[i]) *
                                    static_cast<int32_t>(wo_data[i - block_start]);
                }
            }
        }
    }

    template <typename BlockType>
    void q16_wo_row_gemv(
        const int16_t *context_int16,
        const BlockType *Wo,
        int32_t &output_int32,
        int input_dim,
        int blocks_per_input)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        output_int32 = 0;

        // For each block in the input dimension
        for (int b = 0; b < blocks_per_input; ++b)
        {
            const int16_t *wo_data = Wo[b].qs;

            int start = b * BLOCK_SIZE;
            int end = std::min(start + BLOCK_SIZE, input_dim);
            int count = end - start;

            // Pure INT16 × INT16 → INT32 accumulation
            // This loop pattern maps to VPDPWSSD in JIT
            for (int i = 0; i < count; ++i)
            {
                output_int32 += static_cast<int32_t>(context_int16[start + i]) *
                                static_cast<int32_t>(wo_data[i]);
            }
        }
    }

    // ============================================================================
    // Output Quantization: INT32 → Q16_1
    // ============================================================================

    template <typename BlockType>
    void q16_quantize_to_q16_1(
        const int32_t *accumulators,
        BlockType *output,
        int num_values,
        float input_scale,
        int blocks_per_output)
    {
        constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;

        for (int b = 0; b < blocks_per_output; ++b)
        {
            int start = b * BLOCK_SIZE;
            int end = std::min(start + BLOCK_SIZE, num_values);
            int count = end - start;

            // Find max absolute value for this block's scale
            float max_abs = 0.0f;
            for (int i = 0; i < count; ++i)
            {
                float val = accumulators[start + i] * input_scale;
                float abs_val = std::abs(val);
                if (abs_val > max_abs)
                    max_abs = abs_val;
            }

            // Compute block scale (avoid division by zero)
            float block_scale = max_abs / 32767.0f;
            if (block_scale < 1e-10f)
                block_scale = 1e-10f;

            float inv_scale = 1.0f / block_scale;

            // Quantize values
            output[b].d = block_scale;
            int32_t sum_qs = 0;

            for (int i = 0; i < count; ++i)
            {
                float val = accumulators[start + i] * input_scale;
                int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale));
                quantized = std::clamp(quantized, -32768, 32767);
                output[b].qs[i] = static_cast<int16_t>(quantized);
                sum_qs += quantized;
            }

            // Zero-pad remaining elements if any
            for (int i = count; i < BLOCK_SIZE; ++i)
            {
                output[b].qs[i] = 0;
            }

            output[b].sum_qs = sum_qs;
        }
    }

    // ============================================================================
    // Full Wo Projection (GEMV for decode) - Cache-Aware Tiled
    // ============================================================================

    template <typename BlockType>
    void q16_wo_projection(
        const int32_t *context_int32,
        const BlockType *Wo,
        BlockType *output,
        int d_model,
        int input_dim,
        int blocks_per_input,
        int blocks_per_output)
    {
        // Compute cache-aware tile configuration
        const auto tile_cfg = compute_wo_tile_config(d_model, input_dim, 1);

        LOG_DEBUG("Wo Projection (decode): M_tile=" << tile_cfg.M_tile
                                                    << " K_tile=" << tile_cfg.K_tile
                                                    << " (L1 working set=" << tile_cfg.l1_working_set() << " bytes)");

        // Step 1: Normalize INT32 context to INT16
        std::vector<int16_t> context_int16(input_dim);
        float context_scale;
        q16_context_normalize_to_int16(
            context_int32, context_int16.data(), context_scale, input_dim);

        // Step 2: Compute projection with M-tiled GEMV
        std::vector<int32_t> accumulators(d_model, 0);

        const int M_tile = tile_cfg.M_tile;
        const int K_tile = tile_cfg.K_tile;

        // Process output in M_tile chunks
        for (int m_start = 0; m_start < d_model; m_start += M_tile)
        {
            int m_end = std::min(m_start + M_tile, d_model);

// Parallelize over output rows within this tile
#pragma omp parallel for schedule(static)
            for (int d = m_start; d < m_end; ++d)
            {
                // Wo layout: [d_model, blocks_per_input] row-major
                const BlockType *Wo_row = Wo + d * blocks_per_input;

                q16_wo_row_gemv_tiled<BlockType>(
                    context_int16.data(),
                    Wo_row,
                    accumulators[d],
                    input_dim,
                    blocks_per_input,
                    K_tile);
            }
        }

        // Step 3: Quantize INT32 accumulators to Q16_1 output
        // The scale needs to account for:
        // - context_scale from normalization
        // - Wo scales from each block (we approximate by using average)
        //
        // For now, compute effective scale from accumulator magnitudes
        // A more precise implementation would track per-block scales

        // Compute effective scale: accumulators are in INT16 × INT16 range
        // with context_scale factored out. We need to restore it.
        float wo_scale_approx = 1.0f; // Approximate average Wo scale
        for (int b = 0; b < blocks_per_input; ++b)
        {
            wo_scale_approx += Wo[b].d;
        }
        wo_scale_approx /= blocks_per_input;

        float effective_scale = context_scale * wo_scale_approx;

        q16_quantize_to_q16_1<BlockType>(
            accumulators.data(),
            output,
            d_model,
            effective_scale,
            blocks_per_output);
    }

    // ============================================================================
    // Batched Wo Projection (GEMM for prefill) - Cache-Aware Tiled
    // ============================================================================

    template <typename BlockType>
    void q16_wo_projection_batched(
        const int32_t *context_int32,
        const BlockType *Wo,
        BlockType *output,
        int batch_size,
        int d_model,
        int input_dim,
        int context_stride,
        int output_stride,
        int blocks_per_input,
        int blocks_per_output)
    {
        // Compute cache-aware tile configuration
        const auto tile_cfg = compute_wo_tile_config(d_model, input_dim, batch_size);

        LOG_DEBUG("Wo Projection (prefill): M_tile=" << tile_cfg.M_tile
                                                     << " K_tile=" << tile_cfg.K_tile
                                                     << " N_tile=" << tile_cfg.N_tile
                                                     << " batch=" << batch_size
                                                     << " (L1 working set=" << tile_cfg.l1_working_set() << " bytes)");

        const int M_tile = tile_cfg.M_tile;
        const int K_tile = tile_cfg.K_tile;
        const int N_tile = tile_cfg.N_tile;

        // Process queries in N_tile batches to amortize Wo loads
        for (int q_start = 0; q_start < batch_size; q_start += N_tile)
        {
            int q_end = std::min(q_start + N_tile, batch_size);
            int current_batch = q_end - q_start;

            // For each M_tile chunk of output dimensions
            for (int m_start = 0; m_start < d_model; m_start += M_tile)
            {
                int m_end = std::min(m_start + M_tile, d_model);

// Process queries in this batch (parallel over queries)
#pragma omp parallel for schedule(static)
                for (int q = q_start; q < q_end; ++q)
                {
                    const int32_t *query_context = context_int32 + q * context_stride;
                    BlockType *query_output = output + q * output_stride;

                    // Step 1: Normalize this query's context (if first M_tile)
                    // We cache the normalized context for reuse across M tiles
                    thread_local std::vector<int16_t> context_int16;
                    thread_local float context_scale;
                    thread_local int cached_query = -1;

                    if (cached_query != q || m_start == 0)
                    {
                        context_int16.resize(input_dim);
                        q16_context_normalize_to_int16(
                            query_context, context_int16.data(), context_scale, input_dim);
                        cached_query = q;
                    }

                    // Step 2: Compute accumulators for this M_tile
                    std::vector<int32_t> accumulators(m_end - m_start, 0);

                    for (int d_offset = 0; d_offset < m_end - m_start; ++d_offset)
                    {
                        int d = m_start + d_offset;
                        const BlockType *Wo_row = Wo + d * blocks_per_input;

                        q16_wo_row_gemv_tiled<BlockType>(
                            context_int16.data(),
                            Wo_row,
                            accumulators[d_offset],
                            input_dim,
                            blocks_per_input,
                            K_tile);
                    }

                    // Step 3: Quantize this tile's outputs
                    // Note: For partial tiles, we need to handle block boundaries
                    float wo_scale_approx = 1.0f;
                    for (int b = 0; b < blocks_per_input; ++b)
                    {
                        wo_scale_approx += Wo[b].d;
                    }
                    wo_scale_approx /= blocks_per_input;
                    float effective_scale = context_scale * wo_scale_approx;

                    // Determine output block range for this M_tile
                    constexpr int BLOCK_SIZE = BlockType::BLOCK_SIZE;
                    int b_start = m_start / BLOCK_SIZE;
                    int b_end = (m_end + BLOCK_SIZE - 1) / BLOCK_SIZE;

                    // Quantize directly into output blocks
                    for (int b = b_start; b < b_end && b < blocks_per_output; ++b)
                    {
                        int block_start = b * BLOCK_SIZE;
                        int block_end = std::min(block_start + BLOCK_SIZE, d_model);

                        // Clip to current M_tile
                        int start = std::max(block_start, m_start);
                        int end = std::min(block_end, m_end);

                        // Find max for this block's portion
                        float max_abs = 0.0f;
                        for (int i = start; i < end; ++i)
                        {
                            float val = accumulators[i - m_start] * effective_scale;
                            max_abs = std::max(max_abs, std::abs(val));
                        }

                        float block_scale = max_abs / 32767.0f;
                        if (block_scale < 1e-10f)
                            block_scale = 1e-10f;
                        float inv_scale = 1.0f / block_scale;

                        // Only update scale and values if this is the first tile touching this block
                        if (start == block_start)
                        {
                            query_output[b].d = block_scale;
                            query_output[b].sum_qs = 0;
                        }

                        int32_t partial_sum = 0;
                        for (int i = start; i < end; ++i)
                        {
                            float val = accumulators[i - m_start] * effective_scale;
                            int32_t quantized = static_cast<int32_t>(std::round(val * inv_scale));
                            quantized = std::clamp(quantized, -32768, 32767);
                            query_output[b].qs[i - block_start] = static_cast<int16_t>(quantized);
                            partial_sum += quantized;
                        }
                        query_output[b].sum_qs += partial_sum;
                    }
                }
            }
        }
    }

    // ============================================================================
    // Dispatch Functions
    // ============================================================================

    void q16_wo_projection_dispatch(
        const int32_t *context_int32,
        const void *Wo,
        void *output,
        int d_model,
        int input_dim,
        Q16BlockSize block_size)
    {
        int bs = static_cast<int>(block_size);
        int blocks_per_input = (input_dim + bs - 1) / bs;
        int blocks_per_output = (d_model + bs - 1) / bs;

        switch (block_size)
        {
        case Q16BlockSize::BLOCK_64:
            q16_wo_projection<Q16_1Block_64>(
                context_int32,
                reinterpret_cast<const Q16_1Block_64 *>(Wo),
                reinterpret_cast<Q16_1Block_64 *>(output),
                d_model, input_dim,
                blocks_per_input, blocks_per_output);
            break;

        case Q16BlockSize::BLOCK_128:
            q16_wo_projection<Q16_1Block_128>(
                context_int32,
                reinterpret_cast<const Q16_1Block_128 *>(Wo),
                reinterpret_cast<Q16_1Block_128 *>(output),
                d_model, input_dim,
                blocks_per_input, blocks_per_output);
            break;

        default:
            LOG_ERROR("WoProjection: Unsupported block size: "
                      << static_cast<int>(block_size));
            break;
        }
    }

    void q16_wo_projection_batched_dispatch(
        const int32_t *context_int32,
        const void *Wo,
        void *output,
        int batch_size,
        int d_model,
        int input_dim,
        int context_stride,
        int output_stride,
        Q16BlockSize block_size)
    {
        int bs = static_cast<int>(block_size);
        int blocks_per_input = (input_dim + bs - 1) / bs;
        int blocks_per_output = (d_model + bs - 1) / bs;

        switch (block_size)
        {
        case Q16BlockSize::BLOCK_64:
            q16_wo_projection_batched<Q16_1Block_64>(
                context_int32,
                reinterpret_cast<const Q16_1Block_64 *>(Wo),
                reinterpret_cast<Q16_1Block_64 *>(output),
                batch_size, d_model, input_dim,
                context_stride, output_stride,
                blocks_per_input, blocks_per_output);
            break;

        case Q16BlockSize::BLOCK_128:
            q16_wo_projection_batched<Q16_1Block_128>(
                context_int32,
                reinterpret_cast<const Q16_1Block_128 *>(Wo),
                reinterpret_cast<Q16_1Block_128 *>(output),
                batch_size, d_model, input_dim,
                context_stride, output_stride,
                blocks_per_input, blocks_per_output);
            break;

        default:
            LOG_ERROR("WoProjection batched: Unsupported block size: "
                      << static_cast<int>(block_size));
            break;
        }
    }

    // ============================================================================
    // Explicit Template Instantiations
    // ============================================================================

    template void q16_wo_row_gemv<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t &, int, int);
    template void q16_wo_row_gemv<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t &, int, int);

    template void q16_wo_row_gemv_tiled<Q16_1Block_64>(
        const int16_t *, const Q16_1Block_64 *, int32_t &, int, int, int);
    template void q16_wo_row_gemv_tiled<Q16_1Block_128>(
        const int16_t *, const Q16_1Block_128 *, int32_t &, int, int, int);

    template void q16_quantize_to_q16_1<Q16_1Block_64>(
        const int32_t *, Q16_1Block_64 *, int, float, int);
    template void q16_quantize_to_q16_1<Q16_1Block_128>(
        const int32_t *, Q16_1Block_128 *, int, float, int);

    template void q16_wo_projection<Q16_1Block_64>(
        const int32_t *, const Q16_1Block_64 *, Q16_1Block_64 *, int, int, int, int);
    template void q16_wo_projection<Q16_1Block_128>(
        const int32_t *, const Q16_1Block_128 *, Q16_1Block_128 *, int, int, int, int);

    template void q16_wo_projection_batched<Q16_1Block_64>(
        const int32_t *, const Q16_1Block_64 *, Q16_1Block_64 *, int, int, int, int, int, int, int);
    template void q16_wo_projection_batched<Q16_1Block_128>(
        const int32_t *, const Q16_1Block_128 *, Q16_1Block_128 *, int, int, int, int, int, int, int);

} // namespace llaminar2::kernels::q16_1::microkernels
