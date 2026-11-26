/**
 * @file ROCmGemvKernel_INT8_VNNI_fused.hip
 * @brief ROCm INT8 VNNI fused quantize-GEMV, scatter, self-reducing scatter, and batched scatter kernels.
 */
#include "ROCmGemvKernel_INT8_VNNI_common.hip.inc"

// ============================================================================
// Fused FP32→INT8 GEMV kernel: Quantize + GEMV + Scale in one launch
//
// Eliminates the 3-kernel pipeline (quantize → GEMV → scale) by:
//   1. Loading FP32 activations to shared memory per K-block
//   2. Computing block-local absmax for INT8 quantization
//   3. Quantizing in-register and using v_dot4_i32_i8 for the dot product
//   4. Converting INT32 accumulator to FP32 with (block_scale * scale_B[n])
//   5. atomicAdd FP32 partial results across K-blocks
//   6. Optional fused bias addition
//
// Memory traffic: Same weight reads as INT8 GEMV, but no separate
// activation INT8 buffer, no INT32 output buffer, no scale_A buffer.
// Activations (14KB for K=3584) stay in L2 after first projection.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N/CPT)
// Same grid structure as grid_kpar, but FP32 in → FP32 out.
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_fp32_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_C_fp32,                // [N] FP32 output (zeroed before launch)
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int K, int kblocks)
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    // Process K in chunks of QUANT_TILE_K elements at a time for quantization.
    // Each chunk: load FP32 → find absmax → quantize → dot product.
    // 256 elements = 1KB FP32 in shared memory, fits easily.
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    // Each block processes its K-segment: [k_start, k_end)
    // Round k_per UP to a multiple of 4 so every k_start aligns with
    // VNNI K-groups (4 consecutive K values packed together).
    // Without this, k_start for intermediate blocks can land mid-group,
    // causing activation/weight misalignment in sdot4.
    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // After rounding k_per up, the last block(s) may have k_start >= K.
    // Those blocks have nothing to do — skip entirely.
    if (k_start >= K) return;

    // Shared memory: FP32 activations tile + reduction scratch
    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;        // Block-wide absmax for this tile
    __shared__ int8_t s_A_int8[QUANT_TILE_K]; // Quantized activations

    // FP32 accumulators for partial results (will be atomicAdd'd)
    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    // Process K-segment in QUANT_TILE_K-sized chunks
    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // --- Phase 1: Cooperative load of FP32 activations to shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        }
        __syncthreads();

        // --- Phase 2: Find absmax via parallel reduction ---
        // Each thread finds local max over its portion
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        // Warp reduction (wavefront = 64 threads on gfx906)
        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Broadcast absmax from lane 0 to all lanes via __shfl (no LDS, no sync).
        // (s_absmax LDS slot is reused later in Phase 6 as a completion flag.)
        const float absmax = __shfl(local_max, 0);
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // --- Phase 3: Cooperative quantize FP32 → INT8 in shared memory ---
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // --- Phase 4: INT8 dot product using v_dot4_i32_i8 ---
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, tile_acc[0]);
                    tile_acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, tile_acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, tile_acc[c]);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
            }
        }

        __syncthreads();  // Ensure shared memory is safe for next tile
    }

    // --- Phase 5: Apply weight scales, write to output ---
    if (active)
    {
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            float result = f_acc[c] * d_scales_B[n_base + c];
            if (kblocks == 1)
                d_C_fp32[n_base + c] = result;      // Plain store — no contention
            else
                atomicAdd(&d_C_fp32[n_base + c], result);
        }
    }

    // Bias: only apply once across all K-blocks (last block does it)
    // We use k_block == kblocks - 1 as the "last" block.
    // Note: this has a data race if atomicAdds from other blocks haven't
    // completed yet. Instead, bias is applied as a separate follow-up.
    // See dispatch function for bias handling.
}


// ============================================================================
// Fused FP32→INT8 GEMV scatter kernel: Quantize + GEMV → partial buffer
//
// Same inner loop as gemv_fused_fp32_int8_vnni_kernel_t but writes unscaled
// FP32 partials to d_partial[k_block * N + n], enabling grid-level K-splitting
// WITHOUT atomicAdd or memset. Paired with gemv_reduce_scale_kernel_t to
// produce the final FP32 output.
//
// This replaces the 3-kernel pipeline (quantize → GEMV → scale) with a
// 2-kernel pipeline (fused_scatter → reduce+scale) that eliminates:
//   - The separate quantize kernel launch
//   - hipMemsetAsync for atomicAdd zeroing
//   - All atomicAdd contention
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_scatter_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,           // [KB × N] FP32 partial buffer
    int N, int K, int kblocks)
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;
    __shared__ int8_t s_A_int8[QUANT_TILE_K];

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // Phase 1: Cooperative load of FP32 activations to shared memory
        for (int i = tid; i < tile_len; i += THREADS)
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        __syncthreads();

        // Phase 2: Find absmax via warp reduction
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Broadcast absmax from lane 0 to all lanes via __shfl (no LDS, no sync).
        const float absmax = __shfl(local_max, 0);
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // Phase 3: Cooperative quantize FP32 → INT8 in shared memory
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // Phase 4: INT8 dot product using v_dot4_i32_i8
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, tile_acc[0]);
                    tile_acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, tile_acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, tile_acc[c]);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
        }

        __syncthreads();
    }

    // Phase 5: Write unscaled FP32 partials to buffer (plain store, no atomicAdd)
    if (active)
    {
        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = f_acc[c];
    }
}


// ============================================================================
// Reduce + scale kernel: Sum KB FP32 partials, apply scale_B + bias
//
// Used after gemv_fused_scatter_int8_vnni_kernel_t to combine K-block
// partial results into the final FP32 output with scaling and bias.
//
// Grid: dim3((N + TILE_N - 1) / TILE_N), Block: dim3(TILE_N / CPT)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_reduce_scale_kernel_t(
    const float* __restrict__ d_partial_fp32,    // [KB × N] partial buffer
    float* __restrict__ d_C_fp32,                // [N] output
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int KB,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n >= N) return;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < KB; ++kb)
            sum += d_partial_fp32[static_cast<size_t>(kb) * N + n];

        float result = sum * d_scales_B[n] * alpha;
        if (beta != 0.0f && d_C_existing)
            result += beta * d_C_existing[n];
        if (d_bias)
            result += d_bias[n];

        d_C_fp32[n] = result;
    }
}


// ============================================================================
// Self-reducing scatter kernel: fused scatter + inline reduce (single kernel)
//
// Instead of writing partials to a buffer and launching a separate reduce
// kernel, the last k-block to complete each n-block sums all partials inline
// using a "last-to-arrive" atomic counter pattern.
//
// Benefits:
//   - Eliminates reduce kernel launch overhead (5-27μs savings per shape)
//   - Single kernel launch instead of two
//   - INT32 atomicAdd for counter (native HW on gfx906, ~1 cycle)
//
// Requires: d_counter array must be zero-initialized before launch.
//           d_counter has ceil(N/TILE_N) elements.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_fused_scatter_selfreduce_int8_vnni_kernel_t(
    const float* __restrict__ d_A_fp32,          // [K] FP32 activations
    const int8_t* __restrict__ d_B_int8_vnni,    // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,           // [KB × N] FP32 partial buffer
    float* __restrict__ d_C_fp32,                // [N] FP32 output
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int* __restrict__ d_counter,                 // [grid_n] atomic completion counters
    int N, int K, int kblocks, int valid_kb,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    constexpr int THREADS = TILE_N / CPT;  // 64 threads = 1 wavefront
    constexpr int QUANT_TILE_K = 256;

    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    __shared__ float s_A_fp32[QUANT_TILE_K];
    __shared__ float s_absmax;
    __shared__ int8_t s_A_int8[QUANT_TILE_K];

    float f_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) f_acc[c] = 0.0f;

    for (int k_tile_start = k_start; k_tile_start < k_end; k_tile_start += QUANT_TILE_K)
    {
        const int tile_end = min(k_tile_start + QUANT_TILE_K, k_end);
        const int tile_len = tile_end - k_tile_start;
        const int tile_kgroups = tile_len / 4;

        // Phase 1: Cooperative load of FP32 activations to shared memory
        for (int i = tid; i < tile_len; i += THREADS)
            s_A_fp32[i] = d_A_fp32[k_tile_start + i];
        __syncthreads();

        // Phase 2: Find absmax via warp reduction
        float local_max = 0.0f;
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float v = s_A_fp32[i];
            float a = (v >= 0.0f) ? v : -v;
            local_max = (a > local_max) ? a : local_max;
        }

        #pragma unroll
        for (int offset = 32; offset > 0; offset >>= 1)
        {
            float other = __shfl_down(local_max, offset);
            local_max = (other > local_max) ? other : local_max;
        }

        // Broadcast absmax from lane 0 to all lanes via __shfl (no LDS, no sync).
        // (s_absmax LDS slot is reused later in Phase 6 as a completion flag.)
        const float absmax = __shfl(local_max, 0);
        const float scale = (absmax > 0.0f) ? (absmax / 127.0f) : 1.0f;
        const float inv_scale = (absmax > 0.0f) ? (127.0f / absmax) : 0.0f;

        // Phase 3: Cooperative quantize FP32 → INT8 in shared memory
        for (int i = tid; i < tile_len; i += THREADS)
        {
            float val = s_A_fp32[i] * inv_scale;
            int q = static_cast<int>(rintf(val));
            q = (q < -127) ? -127 : ((q > 127) ? 127 : q);
            s_A_int8[i] = static_cast<int8_t>(q);
        }
        __syncthreads();

        // Phase 4: INT8 dot product using v_dot4_i32_i8
        if (active)
        {
            int32_t tile_acc[CPT];
            #pragma unroll
            for (int c = 0; c < CPT; ++c) tile_acc[c] = 0;

            const int32_t* __restrict__ b_row =
                reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
            const size_t b_stride = static_cast<size_t>(N);
            const int kg_offset = k_tile_start / 4;

            #pragma unroll 4
            for (int tkg = 0; tkg < tile_kgroups; ++tkg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(
                    s_A_int8 + tkg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + (kg_offset + tkg) * b_stride);
                    tile_acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, tile_acc[0]);
                    tile_acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, tile_acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + (kg_offset + tkg) * b_stride + c);
                        tile_acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, tile_acc[c]);
                    }
                }
            }

            // Convert tile INT32 → FP32 with tile-local scale
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                f_acc[c] += static_cast<float>(tile_acc[c]) * scale;
        }
    }

    // Phase 5: Write FP32 partials to buffer (plain store, no atomicAdd)
    if (active)
    {
        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = f_acc[c];
    }

    // Phase 6: Last-to-arrive reduction
    // Make partial writes globally visible, then atomically increment counter.
    // The last k-block to arrive for each n-block sums all partials and
    // writes the final output (eliminating the separate reduce kernel).
    __threadfence();

    if (tid == 0)
    {
        // Shared flag: did this block's lane 0 observe completion?
        // We use s_absmax (reusing LDS since scatter phase is done)
        unsigned int old = atomicAdd(&d_counter[blockIdx.x], 1);
        // Store 1.0 if last-to-arrive, 0.0 otherwise (broadcast via LDS)
        s_absmax = (old + 1 == static_cast<unsigned int>(valid_kb)) ? 1.0f : 0.0f;
    }
    __syncthreads();

    // Only the last k-block's wavefront performs the reduction
    if (s_absmax > 0.0f && active)
    {
        // Sum all valid_kb partials for this thread's columns
        float sum[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) sum[c] = 0.0f;

        for (int kb = 0; kb < valid_kb; ++kb)
        {
            const size_t off = static_cast<size_t>(kb) * N + n_base;
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                sum[c] += d_partial_fp32[off + c];
        }

        // Apply scale_B, alpha, beta, bias and write final output
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            float result = sum[c] * d_scales_B[n] * alpha;
            if (beta != 0.0f && d_C_existing)
                result += beta * d_C_existing[n];
            if (d_bias)
                result += d_bias[n];
            d_C_fp32[n] = result;
        }
    }
}


// ============================================================================
// INT8-input DIRECT kernel: pre-quantized GEMV → final FP32 output
//
// Single-kernel path for kb=1 (no K-parallelism). Instead of writing to a
// partial buffer and launching a separate reduce, this kernel writes the
// final scaled+biased FP32 output directly. Eliminates the reduce kernel
// entirely, saving ~2ms per call on large-N projections (e.g., LM_HEAD).
//
// Grid: dim3(grid_n), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_direct_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,        // [K] pre-quantized INT8 activations
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] VNNI weights
    float* __restrict__ d_C_fp32,               // [N] FP32 output (final)
    const float* __restrict__ d_scale_A,         // [1] activation scale or [K/32] blockwise scales
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int N, int K,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            #pragma unroll 1
            for (int act_block = 0; act_block < act_blocks; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int k_groups = K / 4;

            #pragma unroll 4
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            float result = fp_acc[c] * d_scales_B[n] * alpha;
            if (beta != 0.0f && d_C_existing)
                result += beta * d_C_existing[n];
            if (d_bias)
                result += d_bias[n];
            d_C_fp32[n] = result;
        }
    }
}


// ============================================================================
// INT8-input scatter kernel: pre-quantized GEMV → partial buffer
//
// Like gemv_fused_scatter_int8_vnni_kernel_t but takes pre-quantized INT8
// activations instead of FP32. This enables "quantize once, compute many"
// for multiply_fused_tensor (Q/K/V/Wo/Gate/Up/Down share one quantize).
//
// Key advantages over FP32 scatter:
//   1. No shared memory → 0 bytes LDS → 10 waves/SIMD (max occupancy)
//   2. No __syncthreads() → no serialization overhead
//   3. 1 byte/element activation reads vs 4 bytes → 4× less activation BW
//   4. No redundant per-projection quantization
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_scatter_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,        // [K] pre-quantized INT8 activations
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,          // [KB × N] FP32 partial buffer
    const float* __restrict__ d_scale_A,         // [1] activation scale or [K/32] blockwise scales
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int act_blocks_per_raw = (act_blocks + kblocks - 1) / kblocks;
            const int act_block_start = k_block * act_blocks_per_raw;
            const int act_block_end = min(act_block_start + act_blocks_per_raw, act_blocks);

            if (act_block_start >= act_blocks) return;

            #pragma unroll 1
            for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;

            #pragma unroll 4
            for (int kg = kg_start; kg < kg_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = fp_acc[c];
    }
}


// ============================================================================
// INT8-input self-reducing scatter kernel: pre-quantized GEMV with
// last-to-arrive inline reduction (single kernel, no reduce launch)
//
// Same inner loop as gemv_int8_scatter_vnni_kernel_t but the last k-block
// to complete each n-tile sums all partials and writes the final FP32 output.
//
// Grid: dim3(grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_scatter_selfreduce_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,        // [K] pre-quantized INT8 activations
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] VNNI weights
    float* __restrict__ d_partial_fp32,          // [KB × N] FP32 partial buffer
    float* __restrict__ d_C_fp32,               // [N] FP32 output
    const float* __restrict__ d_scale_A,         // [1] activation scale or [K/32] blockwise scales
    const float* __restrict__ d_scales_B,        // [N] per-column weight scales
    const float* __restrict__ d_bias,            // [N] optional bias (nullable)
    int* __restrict__ d_counter,                 // [grid_n] atomic completion counters
    int N, int K, int kblocks, int valid_kb,
    float alpha, float beta,
    const float* __restrict__ d_C_existing)      // [N] used when beta != 0
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;
    const int tid = threadIdx.x;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(d_B_int8_vnni) + n_base;
        const size_t b_stride = static_cast<size_t>(N);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int act_blocks_per_raw = (act_blocks + kblocks - 1) / kblocks;
            const int act_block_start = k_block * act_blocks_per_raw;
            const int act_block_end = min(act_block_start + act_blocks_per_raw, act_blocks);

            if (act_block_start >= act_blocks) return;

            #pragma unroll 1
            for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;

            #pragma unroll 4
            for (int kg = kg_start; kg < kg_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        const size_t partial_offset = static_cast<size_t>(k_block) * N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_base + c] = fp_acc[c];
    }

    // Phase 2: Last-to-arrive reduction via atomic counter
    __threadfence();

    __shared__ float s_last_flag;  // Reuse to broadcast last-to-arrive status

    if (tid == 0)
    {
        unsigned int old = atomicAdd(&d_counter[blockIdx.x], 1);
        s_last_flag = (old + 1 == static_cast<unsigned int>(valid_kb)) ? 1.0f : 0.0f;
    }
    __syncthreads();

    if (s_last_flag > 0.0f && active)
    {
        // Sum all valid_kb partials
        float sum[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) sum[c] = 0.0f;

        for (int kb = 0; kb < valid_kb; ++kb)
        {
            const size_t off = static_cast<size_t>(kb) * N + n_base;
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                sum[c] += d_partial_fp32[off + c];
        }

        // Apply scale_B, alpha, beta, bias — write final output
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            float result = sum[c] * d_scales_B[n] * alpha;
            if (beta != 0.0f && d_C_existing)
                result += beta * d_C_existing[n];
            if (d_bias)
                result += d_bias[n];
            d_C_fp32[n] = result;
        }
    }
}


// ============================================================================
// Batched INT8-input scatter kernel: multiple projections in one launch
//
// Processes multiple projections (e.g., Q/K/V or Gate/Up) that share the
// same pre-quantized INT8 activations and K dimension. Each thread block
// determines which projection it belongs to via blockIdx.x → batch lookup.
//
// Partial buffer layout: concatenated N across batches.
//   partial[kb * total_N + n_offset[batch] + local_n]
//
// Benefits over individual launches:
//   1. 3→1 kernel launches for QKV (2→1 for Gate/Up)
//   2. Better GPU occupancy (K/V alone = 4 n-blocks = can't fill 60 CUs;
//      batched QKV = 36 n-blocks = much better utilization)
//   3. Single input activation read amortized across all projections
//
// Grid: dim3(total_grid_n, KB), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT, bool UseBlockwise>
__global__ void gemv_int8_scatter_batched_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,    // [K] pre-quantized INT8 activations
    const float* __restrict__ d_scale_A,     // [1] activation scale or [K/32] blockwise scales
    float* __restrict__ d_partial_fp32,      // [KB × total_N] FP32 partial buffer
    GemvBatchArgs args,                      // Batch descriptors (by value)
    int K, int kblocks)
{
    // Phase 1: Determine which batch this block belongs to
    // Linear scan over prefix sums (max 8 entries, trivial cost)
    int batch = 0;
    #pragma unroll
    for (int b = 1; b < MAX_GEMV_BATCH; ++b)
    {
        if (b < args.num_batches && blockIdx.x >= args.grid_n_offset[b])
            batch = b;
    }

    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_local = local_block_x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_local < args.N[batch]);
    const int k_block = blockIdx.y;

    const int k_per_raw = (K + kblocks - 1) / kblocks;
    const int k_per = ((k_per_raw + 3) / 4) * 4;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    if (k_start >= K) return;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    float fp_acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) fp_acc[c] = 0.0f;

    if (active)
    {
        // Index into THIS batch's weight matrix using local n position
        const int32_t* __restrict__ b_row =
            reinterpret_cast<const int32_t*>(args.d_B[batch]) + n_local;
        const size_t b_stride = static_cast<size_t>(args.N[batch]);

        if constexpr (UseBlockwise)
        {
            const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
            const int act_blocks_per_raw = (act_blocks + kblocks - 1) / kblocks;
            const int act_block_start = k_block * act_blocks_per_raw;
            const int act_block_end = min(act_block_start + act_blocks_per_raw, act_blocks);

            if (act_block_start >= act_blocks) return;

            #pragma unroll 1
            for (int act_block = act_block_start; act_block < act_block_end; ++act_block)
            {
                #pragma unroll
                for (int c = 0; c < CPT; ++c) acc[c] = 0;

                gemv_int8_accumulate_act_block(acc, d_A_int8, b_row, b_stride,
                                               act_block * GEMV_INT8_VNNI_KGROUPS_PER_ACT_BLOCK);
                gemv_int8_flush_blockwise_acc(acc, fp_acc, d_scale_A, act_block);
            }
        }
        else
        {
            const int kg_start = k_start / 4;
            const int kg_end = k_end / 4;

            #pragma unroll 4
            for (int kg = kg_start; kg < kg_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(d_A_int8 + kg * 4);

                if constexpr (CPT == 2)
                {
                    const int2 b_pair = *reinterpret_cast<const int2*>(
                        b_row + kg * b_stride);
                    acc[0] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.x, acc[0]);
                    acc[1] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_pair.y, acc[1]);
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < CPT; ++c)
                    {
                        const int32_t b_packed = *(b_row + kg * b_stride + c);
                        acc[c] = ::llaminar2::rocm_integer_contract::signedDot4(a_packed, b_packed, acc[c]);
                    }
                }
            }

            const float sA = d_scale_A[0];
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
                fp_acc[c] = static_cast<float>(acc[c]) * sA;
        }

        const int n_global = args.n_offset[batch] + n_local;
        const size_t partial_offset = static_cast<size_t>(k_block) * args.total_N;
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
            d_partial_fp32[partial_offset + n_global + c] = fp_acc[c];
    }
}


// ============================================================================
// Batched reduce+scale kernel: reduces partials for multiple projections
//
// Reads from concatenated partial buffer and writes to per-projection outputs
// with per-projection scale_B, bias, alpha/beta.
//
// Grid: dim3(total_grid_n), Block: dim3(TILE_N / CPT) = dim3(64)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_reduce_scale_batched_kernel_t(
    const float* __restrict__ d_partial_fp32,  // [KB × total_N] partial buffer
    GemvBatchArgs args,                         // Batch descriptors
    int KB,
    float alpha, float beta)
{
    // Find which batch this block belongs to
    int batch = 0;
    #pragma unroll
    for (int b = 1; b < MAX_GEMV_BATCH; ++b)
    {
        if (b < args.num_batches && blockIdx.x >= args.grid_n_offset[b])
            batch = b;
    }

    const int local_block_x = blockIdx.x - args.grid_n_offset[batch];
    const int n_local = local_block_x * TILE_N + threadIdx.x * CPT;
    const int n_global = args.n_offset[batch] + n_local;

    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        if (n_local + c >= args.N[batch]) return;

        float sum = 0.0f;
        #pragma unroll 16
        for (int kb = 0; kb < KB; ++kb)
            sum += d_partial_fp32[static_cast<size_t>(kb) * args.total_N + n_global + c];

        float result = sum * args.d_scales_B[batch][n_local + c] * alpha;
        if (args.d_bias[batch])
            result += args.d_bias[batch][n_local + c];

        args.d_C[batch][n_local + c] = result;
    }
}



extern "C" {

// =============================================================================
// Fused FP32→FP32 GEMV: Quantize + GEMV + Scale in one kernel
// =============================================================================
//
// Replaces the 3-kernel decode pipeline:
//   OLD: quantizeActivations → rocmGemv_int8_int8_int32_vnni → applyScaling
//   NEW: rocmGemv_fused_fp32_int8_vnni (single launch, directly FP32→FP32)
//
// Eliminates:
//   - d_A_int8 buffer (activation quantization done in shared memory)
//   - d_scales_A buffer (scale folded into per-tile FP32 conversion)
//   - d_CK_int32 buffer (output is FP32 directly)
//   - 2 kernel launches per projection (quantize + scale)
//
bool rocmGemv_fused_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output (will be zeroed)
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    int N, int K,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    if ((N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] N must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // KB heuristic for FUSED kernel: very different from INT8 grid_kpar!
    //
    // The fused kernel uses FP32 atomicAdd for cross-block K-accumulation.
    // On gfx906, FP32 atomicAdd is emulated via CAS loop (not hardware native),
    // so ANY K-splitting causes massive contention overhead.
    //
    // Force kb=1: each block processes the entire K dimension sequentially
    // via QUANT_TILE_K=256 tiles in a loop, then does a single plain store.
    // No atomicAdd, no hipMemsetAsync needed.
    const int kb = 1;

    // Zero output only if using atomicAdd accumulation across K-blocks
    if (kb > 1)
        (void)hipMemsetAsync(d_C_fp32, 0, N * sizeof(float), static_cast<hipStream_t>(stream));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
        dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, d_bias, N, K, kb);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    // Apply bias separately if provided and using K-splitting.
    // With kb==1, bias could be fused but we keep the separate path for simplicity.
    if (d_bias)
    {
        // Reuse existing biasAdd from ROCmQuantisedGemmKernel_CK.hip
        extern bool rocmQuantGemm_biasAdd(
            float *d_output, const float *d_bias,
            int M, int N, int rocm_device_id, void *stream);

        if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
        {
            fprintf(stderr, "[rocmGemv_fused_fp32_int8_vnni] biasAdd failed (N=%d)\n", N);
            return false;
        }
    }

    return true;
}


// =============================================================================
// Fused Scatter FP32→FP32 GEMV: Quantize + GEMV → partials, then reduce+scale
// =============================================================================
//
// 2-kernel pipeline replacing the 3-kernel pipeline:
//   OLD: quantizeActivations → rocmGemv_int8_int8_int32_vnni → applyScaling
//   NEW: fused_scatter (quant+GEMV→partials) → reduce+scale
//
// Eliminates:
//   - Separate quant kernel launch (~10 μs)
//   - hipMemsetAsync (~3 μs)
//   - All atomicAdd contention
//   - One kernel launch overhead (~5-8 μs)
//
// When KB=1 (enough N-blocks for full occupancy), falls back to the
// single-kernel fused path for maximum launch elimination.
//
bool rocmGemv_fused_scatter_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    float* d_partial_buf,            // [KB_MAX × N] pre-allocated partial buffer
    int* d_counter,                  // [ceil(N / TN)] pre-allocated self-reduce counters
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,       // [N] used when beta != 0
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    if ((N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] N must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // KB heuristic: same occupancy target as grid_kpar (~8 waves/CU),
    // but min KB=1 since scatter has no atomicAdd penalty for KB=1.
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_KGROUPS_PER_WAVE = 16;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
    int kb_raw = max(1, (target_blocks + grid_n - 1) / grid_n);

    int kb_max = max(1, k_groups / MIN_KGROUPS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    // Round to nearest factor of k_groups (avoid uneven work distribution)
    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb <= 1)
    {
        // Single-kernel path: enough N-blocks for occupancy with KB=1.
        // Use the existing fused kernel directly (no partial buffer needed).
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, 1), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, nullptr, N, K, 1);

        // Apply bias + alpha/beta if needed (the single-kernel path doesn't handle these)
        if (d_bias || beta != 0.0f)
        {
            // The fused kernel wrote: d_C_fp32[n] = sum * scale_B[n]
            // We need: result = alpha * sum * scale_B + beta * C_existing + bias
            // Since alpha and beta are not applied by the fused kernel for kb=1 path,
            // handle them via a quick fixup if non-trivial
            if (alpha != 1.0f || beta != 0.0f)
            {
                fprintf(stderr, "[rocmGemv_fused_scatter] alpha/beta not yet supported in kb=1 path\n");
                return false;
            }
            if (d_bias)
            {
                extern bool rocmQuantGemm_biasAdd(
                    float *d_output, const float *d_bias,
                    int M, int N, int rocm_device_id, void *stream);
                if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
                    return false;
            }
        }
    }
    else
    {
        // Hybrid dispatch: use self-reducing scatter for small N (few n-blocks),
        // 2-kernel scatter+reduce for large N.
        //
        // Rationale: For small grid_n (< 16), the reduce kernel has too few
        // blocks to fill 60 CUs → catastrophically slow (~27μs for 4 blocks).
        // Self-reduce eliminates the reduce kernel using atomicAdd counters.
        // For large grid_n, the reduce kernel is fast (~4μs) and the atomic
        // overhead in self-reduce would cause a slight regression.
        constexpr int SELFREDUCE_GRID_N_THRESHOLD = 16;

        if (grid_n < SELFREDUCE_GRID_N_THRESHOLD)
        {
            if (!d_counter)
            {
                fprintf(stderr, "[rocmGemv_fused_scatter] self-reduce counter workspace is required (N=%d K=%d)\n",
                        N, K);
                return false;
            }

            int k_per_block = (k_groups + kb - 1) / kb;
            int valid_kb = (k_groups + k_per_block - 1) / k_per_block;

            (void)hipMemsetAsync(d_counter, 0, grid_n * sizeof(int),
                           static_cast<hipStream_t>(stream));

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_selfreduce_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, d_C_fp32, d_scales_B, d_bias,
                d_counter, N, K, kb, valid_kb, alpha, beta, d_C_existing);
        }
        else
        {
            // Two-kernel path: fused scatter → reduce+scale
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, N, K, kb);

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_reduce_scale_kernel_t<TN, CPT>),
                dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_partial_buf, d_C_fp32, d_scales_B, d_bias, N, kb,
                alpha, beta, d_C_existing);
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_fused_scatter] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    return true;
}


// =============================================================================
// Self-Reducing Scatter: single-kernel GEMV with last-to-arrive reduction
// =============================================================================
//
// Fuses scatter + reduce into a single kernel using atomicAdd counters.
// The last k-block to complete for each n-block performs the reduction
// inline, eliminating the reduce kernel launch entirely.
//
// Requires:
//   - d_counter: [grid_n] int array, zeroed before each launch
//   - d_partial_buf: [KB_MAX × N] float buffer for partials
//
bool rocmGemv_fused_scatter_selfreduce_fp32_int8_vnni(
    const float* d_A_fp32,           // [K] FP32 activations
    const int8_t* d_B_int8_vnni,     // [K/4 × N × 4] VNNI-packed weights
    float* d_C_fp32,                 // [N] FP32 output
    const float* d_scales_B,         // [N] per-column weight scales
    const float* d_bias,             // [N] optional bias (nullable)
    float* d_partial_buf,            // [KB_MAX × N] pre-allocated partial buffer
    int* d_counter,                  // [grid_n] atomic counters (zeroed before call)
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,       // [N] used when beta != 0
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0 || (N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_selfreduce] N and K must be multiples of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;  // MI50/MI60

    const int grid_n = (N + TN - 1) / TN;
    const int k_groups = K / 4;

    // Same KB heuristic as scatter dispatch
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_KGROUPS_PER_WAVE = 16;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
    int kb_raw = max(1, (target_blocks + grid_n - 1) / grid_n);

    int kb_max = max(1, k_groups / MIN_KGROUPS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    // Round to nearest factor of k_groups
    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb <= 1)
    {
        // Single-kernel path: no k-splitting needed, use existing fused kernel
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_fused_fp32_int8_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, 1), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_fp32, d_B_int8_vnni, d_C_fp32, d_scales_B, nullptr, N, K, 1);

        if (alpha != 1.0f || beta != 0.0f)
        {
            fprintf(stderr, "[rocmGemv_selfreduce] alpha/beta not supported in kb=1 path\n");
            return false;
        }
        if (d_bias)
        {
            extern bool rocmQuantGemm_biasAdd(
                float *d_output, const float *d_bias,
                int M, int N, int rocm_device_id, void *stream);
            if (!rocmQuantGemm_biasAdd(d_C_fp32, d_bias, 1, N, device_id, stream))
                return false;
        }
    }
    else
    {
        // Compute valid_kb: the actual number of k-blocks that will participate.
        // With uniform k-splitting, all kb blocks do work when k_groups is
        // exactly divisible. The heuristic above ensures this is usually the case,
        // but guard against edge cases.
        int k_per_block = (k_groups + kb - 1) / kb;
        int valid_kb = (k_groups + k_per_block - 1) / k_per_block;

        // Hybrid dispatch: use self-reducing scatter only when grid_n is small
        // (< 16 n-blocks, i.e. N < 2048). For small grid_n, the reduce kernel
        // has too few blocks to fill 60 CUs and becomes catastrophically slow
        // (~27μs for 4 blocks). For large grid_n, the reduce kernel is fast
        // (~4μs) and avoiding atomicAdd/threadfence overhead is better.
        constexpr int SELFREDUCE_GRID_N_THRESHOLD = 16;

        if (grid_n < SELFREDUCE_GRID_N_THRESHOLD && d_counter)
        {
            // Self-reducing path: single kernel, last-to-arrive reduction
            (void)hipMemsetAsync(d_counter, 0, grid_n * sizeof(int),
                           static_cast<hipStream_t>(stream));

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_selfreduce_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, d_C_fp32, d_scales_B, d_bias,
                d_counter, N, K, kb, valid_kb, alpha, beta, d_C_existing);
        }
        else
        {
            // Two-kernel path: scatter + reduce (better for large grid_n)
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_fused_scatter_int8_vnni_kernel_t<TN, CPT>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_fp32, d_B_int8_vnni, d_partial_buf, N, K, kb);

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_reduce_scale_kernel_t<TN, CPT>),
                dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_partial_buf, d_C_fp32, d_scales_B, d_bias, N, kb,
                alpha, beta, d_C_existing);
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_selfreduce] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    return true;
}


bool rocmGemv_int8_scatter_vnni_blockwise(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    float* d_C_fp32,
    const float* d_scales_A_blockwise,
    const float* d_scales_B,
    const float* d_bias,
    float* d_partial_buf,
    int* d_counter,
    int N, int K,
    float alpha, float beta,
    const float* d_C_existing,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if ((K % 4) != 0 || (N % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] N and K must be multiples of 4 (N=%d K=%d)\n", N, K);
        return false;
    }
    if ((K % GEMV_INT8_VNNI_ACT_BLOCK_K) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] K must be a multiple of %d for blockwise scales (K=%d)\n",
                GEMV_INT8_VNNI_ACT_BLOCK_K, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    const int grid_n = (N + TN - 1) / TN;
    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    const int kb = select_blockwise_split_count(grid_n, act_blocks, int8_kb_override);

    if (kb == 1)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_int8_direct_vnni_kernel_t<TN, CPT, true>),
            dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
            d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise,
            d_scales_B, d_bias, N, K,
            alpha, beta, d_C_existing);
    }
    else
    {
        constexpr int SELFREDUCE_GRID_N_THRESHOLD = 16;

        if (grid_n < SELFREDUCE_GRID_N_THRESHOLD)
        {
            if (!d_counter)
            {
                fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] self-reduce counter workspace is required (N=%d K=%d)\n",
                        N, K);
                return false;
            }

            int act_blocks_per_block = (act_blocks + kb - 1) / kb;
            int valid_kb = (act_blocks + act_blocks_per_block - 1) / act_blocks_per_block;

            (void)hipMemsetAsync(d_counter, 0, grid_n * sizeof(int),
                           static_cast<hipStream_t>(stream));

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_scatter_selfreduce_vnni_kernel_t<TN, CPT, true>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_partial_buf, d_C_fp32, d_scales_A_blockwise,
                d_scales_B, d_bias, d_counter, N, K, kb, valid_kb,
                alpha, beta, d_C_existing);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_int8_scatter_vnni_kernel_t<TN, CPT, true>),
                dim3(grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_A_int8, d_B_int8_vnni, d_partial_buf, d_scales_A_blockwise, N, K, kb);

            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(gemv_reduce_scale_kernel_t<TN, CPT>),
                dim3(grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
                d_partial_buf, d_C_fp32, d_scales_B, d_bias, N, kb,
                alpha, beta, d_C_existing);
        }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_blockwise] Kernel launch failed: %s (N=%d K=%d kb=%d)\n",
                hipGetErrorString(err), N, K, kb);
        return false;
    }

    return true;
}


// =============================================================================
// Batched INT8 scatter GEMV dispatch: multiple projections in 2 kernel launches
// =============================================================================
//
// Launches a single batched scatter kernel + a single batched reduce kernel
// for up to MAX_GEMV_BATCH projections that share the same K dimension and
// pre-quantized INT8 activations.
//
// Grid for scatter: dim3(total_grid_n, KB)
// Grid for reduce:  dim3(total_grid_n)
//
// All projections share a single partial buffer with concatenated N layout:
//   partial[kb * total_N + n_offset[batch] + local_n]
//
// Advantages:
//   - QKV: 6 launches → 2 (3 scatter + 3 reduce → 1 scatter + 1 reduce)
//   - Gate/Up: 4 launches → 2
//   - Better reduce occupancy (QKV reduce: 36 blocks vs K/V alone: 4 blocks)
//
bool rocmGemv_int8_scatter_batched_vnni(
    const int8_t* d_A_int8,          // [K] pre-quantized INT8 activations
    const float* d_scale_A,          // [1] activation scale (device pointer)
    float* d_partial_buf,            // [KB_MAX × max_N] pre-allocated partial buffer
    int num_projections,             // Number of projections to batch (1..MAX_GEMV_BATCH)
    const int8_t* const* d_B_ptrs,   // [num_proj] VNNI weight pointers
    float* const* d_C_ptrs,          // [num_proj] FP32 output pointers
    const float* const* d_scales_B_ptrs, // [num_proj] weight scale pointers
    const float* const* d_bias_ptrs, // [num_proj] bias pointers (nullable entries OK)
    const int* N_per_proj,           // [num_proj] output dimension per projection
    int K,
    float alpha, float beta,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (num_projections < 1 || num_projections > MAX_GEMV_BATCH)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched] Invalid num_projections=%d (max=%d)\n",
                num_projections, MAX_GEMV_BATCH);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;  // 128
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;    // 2
    constexpr int NUM_CUS = 60;

    // Build batch args
    GemvBatchArgs args = {};
    args.num_batches = num_projections;
    args.total_grid_n = 0;
    args.total_N = 0;

    for (int b = 0; b < num_projections; ++b)
    {
        if ((N_per_proj[b] % 4) != 0)
        {
            fprintf(stderr, "[rocmGemv_int8_scatter_batched] N[%d]=%d must be multiple of 4\n",
                    b, N_per_proj[b]);
            return false;
        }

        args.d_B[b] = d_B_ptrs[b];
        args.d_C[b] = d_C_ptrs[b];
        args.d_scales_B[b] = d_scales_B_ptrs[b];
        args.d_bias[b] = d_bias_ptrs[b];
        args.N[b] = N_per_proj[b];
        args.grid_n_offset[b] = args.total_grid_n;
        args.n_offset[b] = args.total_N;

        int grid_n_b = (N_per_proj[b] + TN - 1) / TN;
        args.total_grid_n += grid_n_b;
        args.total_N += N_per_proj[b];
    }

    // KB heuristic: same as individual scatter, but using total_grid_n
    const int k_groups = K / 4;
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_KGROUPS_PER_WAVE = 16;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;  // 480
    int kb_raw = max(1, (target_blocks + args.total_grid_n - 1) / args.total_grid_n);

    int kb_max = max(1, k_groups / MIN_KGROUPS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb < 1) kb = 1;

    // Launch batched scatter: dim3(total_grid_n, kb)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_int8_scatter_batched_vnni_kernel_t<TN, CPT, false>),
        dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_int8, d_scale_A, d_partial_buf, args, K, kb);

    // Launch batched reduce: dim3(total_grid_n)
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_reduce_scale_batched_kernel_t<TN, CPT>),
        dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_partial_buf, args, kb, alpha, beta);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched] Kernel launch failed: %s (K=%d kb=%d num_proj=%d)\n",
                hipGetErrorString(err), K, kb, num_projections);
        return false;
    }

    return true;
}


bool rocmGemv_int8_scatter_batched_vnni_blockwise(
    const int8_t* d_A_int8,
    const float* d_scales_A_blockwise,
    float* d_partial_buf,
    int num_projections,
    const int8_t* const* d_B_ptrs,
    float* const* d_C_ptrs,
    const float* const* d_scales_B_ptrs,
    const float* const* d_bias_ptrs,
    const int* N_per_proj,
    int K,
    float alpha, float beta,
    int device_id, void* stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_id);

    if (num_projections < 1 || num_projections > MAX_GEMV_BATCH)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] Invalid num_projections=%d (max=%d)\n",
                num_projections, MAX_GEMV_BATCH);
        return false;
    }

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }
    if ((K % GEMV_INT8_VNNI_ACT_BLOCK_K) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] K must be a multiple of %d for blockwise scales (K=%d)\n",
                GEMV_INT8_VNNI_ACT_BLOCK_K, K);
        return false;
    }

    constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
    constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
    constexpr int NUM_CUS = 60;
    const int int8_kb_override = g_int8_vnni_kb_override.load(std::memory_order_relaxed);

    GemvBatchArgs args = {};
    args.num_batches = num_projections;
    args.total_grid_n = 0;
    args.total_N = 0;

    for (int b = 0; b < num_projections; ++b)
    {
        if ((N_per_proj[b] % 4) != 0)
        {
            fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] N[%d]=%d must be multiple of 4\n",
                    b, N_per_proj[b]);
            return false;
        }

        args.d_B[b] = d_B_ptrs[b];
        args.d_C[b] = d_C_ptrs[b];
        args.d_scales_B[b] = d_scales_B_ptrs[b];
        args.d_bias[b] = d_bias_ptrs[b];
        args.N[b] = N_per_proj[b];
        args.grid_n_offset[b] = args.total_grid_n;
        args.n_offset[b] = args.total_N;

        int grid_n_b = (N_per_proj[b] + TN - 1) / TN;
        args.total_grid_n += grid_n_b;
        args.total_N += N_per_proj[b];
    }

    const int act_blocks = K / GEMV_INT8_VNNI_ACT_BLOCK_K;
    constexpr int TARGET_WAVES_PER_CU = 8;
    constexpr int MIN_ACT_BLOCKS_PER_WAVE = 2;

    int target_blocks = TARGET_WAVES_PER_CU * NUM_CUS;
    int kb_raw = max(1, (target_blocks + args.total_grid_n - 1) / args.total_grid_n);

    int kb_max = max(1, act_blocks / MIN_ACT_BLOCKS_PER_WAVE);
    kb_raw = min(kb_raw, kb_max);

    int kb = kb_raw;
    if (act_blocks % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 1 && act_blocks % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && act_blocks % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    if (kb < 1) kb = 1;

    if (int8_kb_override > 0)
    {
        kb = min(int8_kb_override, act_blocks);
    }

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_int8_scatter_batched_vnni_kernel_t<TN, CPT, true>),
        dim3(args.total_grid_n, kb), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_A_int8, d_scales_A_blockwise, d_partial_buf, args, K, kb);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(gemv_reduce_scale_batched_kernel_t<TN, CPT>),
        dim3(args.total_grid_n), dim3(TN / CPT), 0, static_cast<hipStream_t>(stream),
        d_partial_buf, args, kb, alpha, beta);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_scatter_batched_blockwise] Kernel launch failed: %s (K=%d kb=%d num_proj=%d)\n",
                hipGetErrorString(err), K, kb, num_projections);
        return false;
    }

    return true;
}



} // extern "C"
