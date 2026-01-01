/**
 * @file WoProjection.cpp
 * @brief Integer Wo projection microkernel implementation
 *
 * @see WoProjection.h for algorithm description
 * @see docs/v2/PROJECT_Q16_INTEGER_ATTENTION_V2.md
 */

#include "WoProjection.h"
#include "kernels/cpu/gemm_v4/QuantisedGemmJit_M1.h" // For QuantisedPackedWeights
#include "utils/Logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

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
    // INT16 VNNI Wo Projection Implementation
    // ============================================================================
    //
    // Uses existing QuantisedPackedWeights (INT8) but performs INT16×INT16 compute.
    // The INT8 weights are sign-extended to INT16 at load time.
    //
    // Memory layout of QuantisedPackedWeights:
    //   packed_data: [K/4][N][4] INT8 (groups of 4 for VPDPBUSD)
    //   scales: [K/32][N] float
    //   compensation: [K/32][N] int32
    //
    // For VPDPWSSD, we load INT8 and sign-extend to INT16.
    // ============================================================================

    int32_t wo_gemv_row_vnni_int16(
        const int16_t *context_int16,
        const QuantisedPackedWeights *Wo_packed,
        int row,
        int input_dim)
    {
        const int N = Wo_packed->N;
        const int K = Wo_packed->K;
        const int K_padded = ((K + 31) / 32) * 32;

        // Packed layout: [K/4][N][4], but actually [N/64][K][64] after reorganization
        // Let's navigate the actual layout
        int n_blk = row / 64;
        int n_rem = row % 64;

        int32_t acc = 0;

#ifdef __AVX512F__
        // AVX-512 path using VPDPWSSD
        __m512i acc_vec = _mm512_setzero_si512();

        // Process 32 elements at a time (one ZMM of INT16)
        int k = 0;
        for (; k + 31 < input_dim; k += 32)
        {
            // Load 32 INT16 context values
            __m512i ctx = _mm512_loadu_si512(context_int16 + k);

            // Load 32 INT8 weight values and sign-extend to INT16
            // Packed layout: base + n_blk * (K * 64) + k * 64 + n_rem * ???
            // Actually the layout is [N/64][K/4][64][4]
            // For row 'row' and k-position 'k':
            //   offset = n_blk * (K_padded * 64) + (k/4) * (64 * 4) + n_rem * 4 + (k%4)
            // But we need 32 consecutive k values, so we load 8 groups of 4

            // Simpler approach: use scalar loads for reference, optimize later
            alignas(64) int16_t wo_int16[32];
            for (int i = 0; i < 32; ++i)
            {
                int ki = k + i;
                size_t offset = (size_t)n_blk * (K_padded * 64) + (ki / 4) * (64 * 4) + n_rem * 4 + (ki % 4);
                wo_int16[i] = static_cast<int16_t>(Wo_packed->packed_data[offset]);
            }

            __m512i wo = _mm512_load_si512(wo_int16);

            // VPDPWSSD: acc += ctx[2i]*wo[2i] + ctx[2i+1]*wo[2i+1] for each dword
            acc_vec = _mm512_dpwssd_epi32(acc_vec, ctx, wo);
        }

        // Horizontal sum of acc_vec (16 INT32 lanes)
        // Reduce to 256-bit
        __m256i acc_256 = _mm256_add_epi32(
            _mm512_extracti32x8_epi32(acc_vec, 0),
            _mm512_extracti32x8_epi32(acc_vec, 1));
        // Reduce to 128-bit
        __m128i acc_128 = _mm_add_epi32(
            _mm256_extracti128_si256(acc_256, 0),
            _mm256_extracti128_si256(acc_256, 1));
        // Reduce to scalar
        acc_128 = _mm_add_epi32(acc_128, _mm_shuffle_epi32(acc_128, 0x4E)); // 01 00 11 10
        acc_128 = _mm_add_epi32(acc_128, _mm_shuffle_epi32(acc_128, 0xB1)); // 10 11 00 01
        acc = _mm_cvtsi128_si32(acc_128);

        // Process remaining elements (tail)
        for (; k < input_dim; ++k)
        {
            size_t offset = (size_t)n_blk * (K_padded * 64) + (k / 4) * (64 * 4) + n_rem * 4 + (k % 4);
            int16_t wo_val = static_cast<int16_t>(Wo_packed->packed_data[offset]);
            acc += static_cast<int32_t>(context_int16[k]) * static_cast<int32_t>(wo_val);
        }
#else
        // Scalar fallback
        for (int k = 0; k < input_dim; ++k)
        {
            size_t offset = (size_t)n_blk * (K_padded * 64) + (k / 4) * (64 * 4) + n_rem * 4 + (k % 4);
            int16_t wo_val = static_cast<int16_t>(Wo_packed->packed_data[offset]);
            acc += static_cast<int32_t>(context_int16[k]) * static_cast<int32_t>(wo_val);
        }
#endif

        return acc;
    }

    void wo_projection_vnni_int16(
        const int32_t *context_int32,
        float context_scale,
        const QuantisedPackedWeights *Wo_packed,
        void *output,
        int d_model,
        int input_dim,
        Q16BlockSize block_size,
        float *wo_output_fp32)
    {
        if (!Wo_packed || !context_int32 || !output)
        {
            LOG_ERROR("wo_projection_vnni_int16: null pointer");
            return;
        }

        if (d_model != Wo_packed->N || input_dim > Wo_packed->K)
        {
            LOG_ERROR("wo_projection_vnni_int16: dimension mismatch. d_model=" << d_model
                                                                               << " Wo_packed->N=" << Wo_packed->N
                                                                               << " input_dim=" << input_dim << " Wo_packed->K=" << Wo_packed->K);
            return;
        }

        const int K_blocks = (input_dim + 31) / 32;
        const int bs = static_cast<int>(block_size);
        const int blocks_per_output = (d_model + bs - 1) / bs;

        LOG_DEBUG("wo_projection_vnni_int16: d_model=" << d_model << " input_dim=" << input_dim
                                                       << " K_blocks=" << K_blocks << " context_scale=" << context_scale);

        // Step 1: Normalize INT32 context to INT16 range
        std::vector<int16_t> context_int16(input_dim);
        float ctx_norm_scale;
        q16_context_normalize_to_int16(context_int32, context_int16.data(), ctx_norm_scale, input_dim);

        // Step 2: Compute projection for each output dimension
        std::vector<int32_t> accumulators(d_model);
        std::vector<float> wo_scales(d_model, 0.0f);

        // Compute combined scale per output row
        // Each output row uses scales from K_blocks input blocks
        // For simplicity, we'll compute a single combined scale per row
        for (int d = 0; d < d_model; ++d)
        {
            // Average the Wo scales for this row
            float scale_sum = 0.0f;
            int n_blk = d / 64;
            int n_rem = d % 64;
            for (int kb = 0; kb < K_blocks; ++kb)
            {
                // scales layout: [K/32][N_padded]
                int N_padded = ((Wo_packed->N + 63) / 64) * 64;
                size_t scale_idx = kb * N_padded + d;
                if (scale_idx < Wo_packed->scales.size())
                {
                    scale_sum += Wo_packed->scales[scale_idx];
                }
            }
            wo_scales[d] = (K_blocks > 0) ? (scale_sum / K_blocks) : 1.0f;
        }

#pragma omp parallel for schedule(static)
        for (int d = 0; d < d_model; ++d)
        {
            accumulators[d] = wo_gemv_row_vnni_int16(context_int16.data(), Wo_packed, d, input_dim);
        }

        // Step 3: Apply scales and optionally snapshot to FP32
        // Combined scale: ctx_norm_scale * context_scale * wo_scale
        // - ctx_norm_scale: from INT32→INT16 normalization
        // - context_scale: from attention softmax (passed in)
        // - wo_scale: per-row Wo weight scale
        if (wo_output_fp32)
        {
            for (int d = 0; d < d_model; ++d)
            {
                float combined_scale = ctx_norm_scale * context_scale * wo_scales[d];
                wo_output_fp32[d] = static_cast<float>(accumulators[d]) * combined_scale;
            }
        }

        // Step 4: Quantize to Q16_1 output
        // The input_scale for quantization is the combined scale
        float combined_scale = ctx_norm_scale * context_scale;
        // We need to incorporate wo_scales into the quantization
        // For simplicity, compute average wo_scale
        float avg_wo_scale = 0.0f;
        for (int d = 0; d < d_model; ++d)
        {
            avg_wo_scale += wo_scales[d];
        }
        avg_wo_scale /= d_model;
        float quant_scale = combined_scale * avg_wo_scale;

        switch (block_size)
        {
        case Q16BlockSize::BLOCK_32:
            q16_quantize_to_q16_1<Q16_1Block>(
                accumulators.data(),
                reinterpret_cast<Q16_1Block *>(output),
                d_model, quant_scale, blocks_per_output);
            break;
        case Q16BlockSize::BLOCK_64:
            q16_quantize_to_q16_1<Q16_1Block_64>(
                accumulators.data(),
                reinterpret_cast<Q16_1Block_64 *>(output),
                d_model, quant_scale, blocks_per_output);
            break;
        case Q16BlockSize::BLOCK_128:
            q16_quantize_to_q16_1<Q16_1Block_128>(
                accumulators.data(),
                reinterpret_cast<Q16_1Block_128 *>(output),
                d_model, quant_scale, blocks_per_output);
            break;
        default:
            LOG_ERROR("wo_projection_vnni_int16: unsupported block size");
            break;
        }

        LOG_DEBUG("wo_projection_vnni_int16: complete, quant_scale=" << quant_scale);
    }

    void wo_projection_vnni_int16_batched(
        const int32_t *context_int32,
        const float *context_scales,
        const QuantisedPackedWeights *Wo_packed,
        void *output,
        int batch_size,
        int d_model,
        int input_dim,
        int context_stride,
        int output_stride,
        Q16BlockSize block_size,
        float *wo_output_fp32)
    {
        // Process each batch element
        for (int b = 0; b < batch_size; ++b)
        {
            const int32_t *ctx = context_int32 + b * context_stride;
            float ctx_scale = context_scales ? context_scales[b] : 1.0f;

            // Output pointer depends on block size
            void *out;
            switch (block_size)
            {
            case Q16BlockSize::BLOCK_32:
                out = reinterpret_cast<Q16_1Block *>(output) + b * output_stride;
                break;
            case Q16BlockSize::BLOCK_64:
                out = reinterpret_cast<Q16_1Block_64 *>(output) + b * output_stride;
                break;
            case Q16BlockSize::BLOCK_128:
                out = reinterpret_cast<Q16_1Block_128 *>(output) + b * output_stride;
                break;
            default:
                LOG_ERROR("wo_projection_vnni_int16_batched: unsupported block size");
                return;
            }

            float *fp32_out = wo_output_fp32 ? (wo_output_fp32 + b * d_model) : nullptr;

            wo_projection_vnni_int16(ctx, ctx_scale, Wo_packed, out, d_model, input_dim, block_size, fp32_out);
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

    template void q16_quantize_to_q16_1<Q16_1Block>(
        const int32_t *, Q16_1Block *, int, float, int);
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
