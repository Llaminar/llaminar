/**
 * @file ROCmGemvKernel.hip
 * @brief Bandwidth-optimized INT8 GEMV kernel for decode (M=1) on MI50/MI60 (gfx906)
 *
 * ## Problem
 *
 * CK INT8 GEMM at M=1 achieves only 2.4 TFLOPS (~8% of MI50's 29.5 TOPS INT8),
 * because 32×32 tiles waste 31 of 32 rows and can't fill the GPU. Decode is
 * memory-bandwidth-bound (reading all 7B weights per token), not compute-bound.
 *
 * ## Solution: Direct FP32×INT8 GEMV
 *
 * Skip activation quantization entirely. Read INT8 weights, multiply with FP32
 * activations directly, scale by per-column weight scale only.
 *
 * For M=1:  output[n] = scale_B[n] * sum_k( A_fp32[k] * B_int8[k * N + n] )
 *
 * ## Memory Layout
 *
 * Weights are [K × N] RowMajor INT8 (same as CK's packed format).
 * For GEMV we want to stream through K (outer dim), accessing N elements per row.
 *
 * ## Kernel Design (gfx906-optimized)
 *
 * - Each workgroup handles a tile of output columns [n_start, n_start + TILE_N)
 * - Each thread accumulates one output element across all K
 * - K-reduction done in registers (no shared memory needed for M=1)
 * - Uses vectorized loads (int4 = 16 bytes = 16 INT8 values) for coalesced access
 * - MI50 has 64 CUs, 1000 GB/s HBM2 bandwidth
 *
 * ## Expected Performance
 *
 * Weights: 7 GB INT8. At 480 GB/s → ~14.6ms theoretical minimum.
 * With activation reads + scale reads + output writes → ~16-18ms realistic.
 * Current CK: ~113ms. Target: ~15-20ms (6-7× speedup).
 *
 * @author Llaminar V2
 * @date February 2026
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

// ============================================================================
// Kernel: gemv_int8_fp32
//
// Each thread computes one output element: output[n] = scale_B[n] * dot(A, B[:, n])
//
// Grid: (ceil(N / TILE_N), 1, 1)
// Block: (TILE_N, 1, 1)
//
// Uses K-reduction in registers with vectorized INT8 loads.
// ============================================================================

// Tile width: number of output columns per workgroup
// 256 threads per block — each thread does one output column
constexpr int GEMV_TILE_N = 256;

// VNNI kernel tile parameters
constexpr int GEMV_INT8_VNNI_WIDE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_WIDE2_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_SQUARE_TILE_N = 256;
constexpr int GEMV_INT8_VNNI_KPAR_TILE_K = 256;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_TILE_N = 128;
constexpr int GEMV_INT8_VNNI_GRID_KPAR_CPT = 4;

__global__ void gemv_int8_int8_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t packed = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + n) * 4);
        const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
        const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
        const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
        const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

        acc += a0 * static_cast<int32_t>(b0)
             + a1 * static_cast<int32_t>(b1)
             + a2 * static_cast<int32_t>(b2)
             + a3 * static_cast<int32_t>(b3);
    }

    d_C_int32[n] = acc;
}

// Vectorized variant: each thread handles 4 columns (N must be divisible by 4)
__global__ void gemv_int8_int8_wide_vec4_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int lane_n = threadIdx.x;
    const int n4 = blockIdx.x * GEMV_INT8_VNNI_WIDE_TILE_N + lane_n * 4;
    if (n4 >= N) return;

    int4 acc = {0, 0, 0, 0};
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 0)) * 4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 1)) * 4);
        const uint32_t p2 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 2)) * 4);
        const uint32_t p3 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n4 + 3)) * 4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        const int8_t b20 = static_cast<int8_t>(p2 & 0xFF);
        const int8_t b21 = static_cast<int8_t>((p2 >> 8) & 0xFF);
        const int8_t b22 = static_cast<int8_t>((p2 >> 16) & 0xFF);
        const int8_t b23 = static_cast<int8_t>((p2 >> 24) & 0xFF);

        const int8_t b30 = static_cast<int8_t>(p3 & 0xFF);
        const int8_t b31 = static_cast<int8_t>((p3 >> 8) & 0xFF);
        const int8_t b32 = static_cast<int8_t>((p3 >> 16) & 0xFF);
        const int8_t b33 = static_cast<int8_t>((p3 >> 24) & 0xFF);

        acc.x += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
               + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
        acc.y += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
               + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
        acc.z += a0 * static_cast<int32_t>(b20) + a1 * static_cast<int32_t>(b21)
               + a2 * static_cast<int32_t>(b22) + a3 * static_cast<int32_t>(b23);
        acc.w += a0 * static_cast<int32_t>(b30) + a1 * static_cast<int32_t>(b31)
               + a2 * static_cast<int32_t>(b32) + a3 * static_cast<int32_t>(b33);
    }

    d_C_int32[n4 + 0] = acc.x;
    d_C_int32[n4 + 1] = acc.y;
    d_C_int32[n4 + 2] = acc.z;
    d_C_int32[n4 + 3] = acc.w;
}

// Wide-N kernel: 2 columns per thread
__global__ void gemv_int8_int8_wide2_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n2 = blockIdx.x * GEMV_INT8_VNNI_WIDE2_TILE_N + threadIdx.x * 2;
    if (n2 >= N) return;

    int32_t acc0 = 0;
    int32_t acc1 = 0;
    const int k_groups = K / 4;
    for (int kg = 0; kg < k_groups; ++kg)
    {
        const int k_base = kg * 4;
        const int32_t a0 = static_cast<int32_t>(d_A_int8[k_base + 0]);
        const int32_t a1 = static_cast<int32_t>(d_A_int8[k_base + 1]);
        const int32_t a2 = static_cast<int32_t>(d_A_int8[k_base + 2]);
        const int32_t a3 = static_cast<int32_t>(d_A_int8[k_base + 3]);

        const uint32_t p0 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 0)) * 4);
        const uint32_t p1 = *reinterpret_cast<const uint32_t*>(
            d_B_int8_vnni + (static_cast<size_t>(kg) * N + (n2 + 1)) * 4);

        const int8_t b00 = static_cast<int8_t>(p0 & 0xFF);
        const int8_t b01 = static_cast<int8_t>((p0 >> 8) & 0xFF);
        const int8_t b02 = static_cast<int8_t>((p0 >> 16) & 0xFF);
        const int8_t b03 = static_cast<int8_t>((p0 >> 24) & 0xFF);

        const int8_t b10 = static_cast<int8_t>(p1 & 0xFF);
        const int8_t b11 = static_cast<int8_t>((p1 >> 8) & 0xFF);
        const int8_t b12 = static_cast<int8_t>((p1 >> 16) & 0xFF);
        const int8_t b13 = static_cast<int8_t>((p1 >> 24) & 0xFF);

        acc0 += a0 * static_cast<int32_t>(b00) + a1 * static_cast<int32_t>(b01)
              + a2 * static_cast<int32_t>(b02) + a3 * static_cast<int32_t>(b03);
        acc1 += a0 * static_cast<int32_t>(b10) + a1 * static_cast<int32_t>(b11)
              + a2 * static_cast<int32_t>(b12) + a3 * static_cast<int32_t>(b13);
    }

    d_C_int32[n2 + 0] = acc0;
    d_C_int32[n2 + 1] = acc1;
}

// K-parallel kernel: scalar threads, shared A tiles
__global__ void gemv_int8_int8_square_vnni_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int N,
    int K)
{
    const int n = blockIdx.x * GEMV_INT8_VNNI_SQUARE_TILE_N + threadIdx.x;
    if (n >= N) return;

    int32_t acc = 0;
    __shared__ int8_t sA[GEMV_INT8_VNNI_KPAR_TILE_K];

    for (int k_base = 0; k_base < K; k_base += GEMV_INT8_VNNI_KPAR_TILE_K)
    {
        const int tile_k = min(GEMV_INT8_VNNI_KPAR_TILE_K, K - k_base);
        for (int i = threadIdx.x; i < tile_k; i += blockDim.x)
        {
            sA[i] = d_A_int8[k_base + i];
        }
        __syncthreads();

        const int tile_groups = tile_k / 4;
        const int k_group_base = k_base / 4;
        for (int kg = 0; kg < tile_groups; ++kg)
        {
            const int k_local = kg * 4;
            const int32_t a0 = static_cast<int32_t>(sA[k_local + 0]);
            const int32_t a1 = static_cast<int32_t>(sA[k_local + 1]);
            const int32_t a2 = static_cast<int32_t>(sA[k_local + 2]);
            const int32_t a3 = static_cast<int32_t>(sA[k_local + 3]);

            const int global_kg = k_group_base + kg;
            const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                d_B_int8_vnni + (static_cast<size_t>(global_kg) * N + n) * 4);
            const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
            const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
            const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
            const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

            acc += a0 * static_cast<int32_t>(b0)
                 + a1 * static_cast<int32_t>(b1)
                 + a2 * static_cast<int32_t>(b2)
                 + a3 * static_cast<int32_t>(b3);
        }
        __syncthreads();
    }

    d_C_int32[n] = acc;
}

// ============================================================================
// Grid-level K-parallel VNNI kernel with atomicAdd reduction
//
// Each block processes a TILE_N/CPT-thread tile × one K-segment.
// Grid: (ceil(N/TILE_N), KB) where KB splits K across blockIdx.y.
// Output MUST be zeroed before launch (atomicAdd accumulation).
//
// Key insight: Grid-level K-splitting creates 28×48=1344 independent blocks
// (vs 56 with square_vnni), providing enough wavefronts to fully hide
// HBM2 latency on MI50 (60 CUs × 40 max concurrent waves = 2400).
//
// Benchmark results (MI50, gfx906):
//   Q/Wo (3584×3584):    22.0μs / 58.5% peak BW (9.6× vs square_vnni)
//   FFN Down (3584×18944): 95.7μs / 70.9% peak BW (2.0× vs smem kpar)
// ============================================================================
template <int TILE_N, int CPT>
__global__ void gemv_int8_int8_grid_kpar_vnni_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,   // [K/4][N][4] flat VNNI
    int32_t* __restrict__ d_C_int32,
    int N, int K, int kblocks)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const bool active = (n_base < N);
    const int k_block = blockIdx.y;

    // Each block processes its K-segment: [k_start, k_end)
    const int k_per = (K + kblocks - 1) / kblocks;
    const int k_start = k_block * k_per;
    const int k_end = min(k_start + k_per, K);

    // Accumulate partial dot product for CPT output columns
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0;

    if (active)
    {
        // Process VNNI k-groups within this block's K-segment
        const int kg_start = k_start / 4;
        const int kg_end = k_end / 4;

        for (int kg = kg_start; kg < kg_end; ++kg)
        {
            const int kb = kg * 4;
            const int32_t a0 = static_cast<int32_t>(d_A_int8[kb + 0]);
            const int32_t a1 = static_cast<int32_t>(d_A_int8[kb + 1]);
            const int32_t a2 = static_cast<int32_t>(d_A_int8[kb + 2]);
            const int32_t a3 = static_cast<int32_t>(d_A_int8[kb + 3]);

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const uint32_t packed = *reinterpret_cast<const uint32_t*>(
                    d_B_int8_vnni + (static_cast<size_t>(kg) * N + n_base + c) * 4);
                const int8_t b0 = static_cast<int8_t>(packed & 0xFF);
                const int8_t b1 = static_cast<int8_t>((packed >> 8) & 0xFF);
                const int8_t b2 = static_cast<int8_t>((packed >> 16) & 0xFF);
                const int8_t b3 = static_cast<int8_t>((packed >> 24) & 0xFF);

                acc[c] += a0 * static_cast<int32_t>(b0)
                        + a1 * static_cast<int32_t>(b1)
                        + a2 * static_cast<int32_t>(b2)
                        + a3 * static_cast<int32_t>(b3);
            }
        }

        // Accumulate partials to output via atomicAdd
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            atomicAdd(&d_C_int32[n_base + c], acc[c]);
        }
    }
}


// ============================================================================
// C API: Launch wrappers
// ============================================================================

extern "C" {

bool rocmGemv_int8_int8_int32_vnni(
    const int8_t* d_A_int8,
    const int8_t* d_B_int8_vnni,
    int32_t* d_C_int32,
    int N, int K,
    int device_id)
{
    hipSetDevice(device_id);

    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] K must be multiple of 4 (N=%d K=%d)\n", N, K);
        return false;
    }

    const bool use_wide = (N >= 3 * K);
    const bool use_vec4 = ((N % 4) == 0);
    const bool use_vec2 = ((N % 2) == 0);
    const bool use_wide_vec4 = use_wide && (N >= 8 * K) && use_vec4;
    const bool use_wide2 = use_wide && (N < 8 * K) && use_vec2 && (N >= GEMV_INT8_VNNI_WIDE2_TILE_N);
    // Grid-level K-parallel: replaces old ffn_down, kpar, and square paths for non-wide shapes
    // Uses blockIdx.y for K-splitting + atomicAdd → 1000+ wavefronts vs 56-224 old
    const bool use_grid_kpar = !use_wide && use_vec4 && (N >= GEMV_INT8_VNNI_GRID_KPAR_TILE_N);
    const bool use_square = (N >= GEMV_INT8_VNNI_SQUARE_TILE_N);

    if (use_wide2)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_WIDE2_TILE_N - 1) / GEMV_INT8_VNNI_WIDE2_TILE_N;
        dim3 block(GEMV_INT8_VNNI_WIDE2_TILE_N / 2, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_wide2_vnni_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else if (use_wide_vec4 && (N >= GEMV_INT8_VNNI_WIDE_TILE_N))
    {
        const int grid_n = (N + GEMV_INT8_VNNI_WIDE_TILE_N - 1) / GEMV_INT8_VNNI_WIDE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_WIDE_TILE_N / 4, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_wide_vec4_vnni_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else if (use_grid_kpar)
    {
        // Grid-level K-parallel with atomicAdd: KB independent blocks per output tile
        // Each block processes K/KB fraction; output must be zeroed before launch
        // KB heuristic: target ~1500 wavefronts to saturate MI50 (60 CUs × ~40 waves/CU)
        constexpr int TN = GEMV_INT8_VNNI_GRID_KPAR_TILE_N;
        constexpr int CPT = GEMV_INT8_VNNI_GRID_KPAR_CPT;
        const int grid_n = (N + TN - 1) / TN;
        const int kb = max(2, min(64, (K + 63) / 64));

        hipMemsetAsync(d_C_int32, 0, N * sizeof(int32_t), 0);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(gemv_int8_int8_grid_kpar_vnni_kernel_t<TN, CPT>),
            dim3(grid_n, kb), dim3(TN / CPT), 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K, kb);
    }
    else if (use_square)
    {
        const int grid_n = (N + GEMV_INT8_VNNI_SQUARE_TILE_N - 1) / GEMV_INT8_VNNI_SQUARE_TILE_N;
        dim3 block(GEMV_INT8_VNNI_SQUARE_TILE_N, 1, 1);
        hipLaunchKernelGGL(
            gemv_int8_int8_square_vnni_kernel,
            dim3(grid_n), block, 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }
    else
    {
        const int grid_n = (N + GEMV_TILE_N - 1) / GEMV_TILE_N;
        hipLaunchKernelGGL(
            gemv_int8_int8_vnni_kernel,
            dim3(grid_n), dim3(GEMV_TILE_N), 0, 0,
            d_A_int8, d_B_int8_vnni, d_C_int32, N, K);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_int8_int8_int32_vnni] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

// =============================================================================
// VNNI → Row-Major Repack Kernel
// =============================================================================
//
// Repacks weights from flat VNNI layout [K/4][N][4] to row-major [N×K]
// (equivalently [K×N] column-major for CK).
//
// VNNI layout:  vnni[(kg*N + n)*4 + i]  where kg = k/4, i = k%4
// Row-major:    rowmajor[n*K + k]       where k = kg*4 + i
//
// Mapping:  rowmajor[n*K + kg*4 + i] = vnni[(kg*N + n)*4 + i]
//
// Each thread handles one (n, kg) pair = 4 elements, packed as int32_t.
// Grid: ceil(N * K/4 / blockDim.x) blocks, blockDim.x = 256 threads.
//
// Access pattern:
//   - VNNI reads:  consecutive tids (consecutive n) → stride 4 → coalesced
//   - Row-major writes: consecutive tids (consecutive n) → stride K → scattered
//     but each write is a 4-byte int32_t, so individual writes are efficient.
//
// Cost: ~65-200μs for 18944×3584 (Gate/Up shape). Amortized over prefill.
//
__global__ void
repack_vnni_to_rowmajor_kernel(
    const int8_t* __restrict__ d_B_vnni,     // [K/4][N][4] input
    int8_t* __restrict__ d_B_rowmajor,       // [N×K] output
    int N, int K)
{
    const int k_groups = K / 4;
    const int total = N * k_groups;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= total)
        return;

    const int n  = tid % N;
    const int kg = tid / N;

    // Read 4 bytes from VNNI as a single int32_t (coalesced for consecutive n)
    const int32_t packed = *reinterpret_cast<const int32_t*>(
        &d_B_vnni[(static_cast<int64_t>(kg) * N + n) * 4]);

    // Write 4 bytes to row-major (scattered for consecutive n, but 4B atomic)
    *reinterpret_cast<int32_t*>(
        &d_B_rowmajor[static_cast<int64_t>(n) * K + kg * 4]) = packed;
}


bool rocmGemv_repackVNNI_to_rowmajor(
    const int8_t* d_B_vnni,
    int8_t* d_B_rowmajor,
    int N, int K,
    int device_id)
{
    if ((K % 4) != 0)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] K must be multiple of 4 (K=%d)\n", K);
        return false;
    }

    hipSetDevice(device_id);

    const int k_groups = K / 4;
    const int total = N * k_groups;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;

    hipLaunchKernelGGL(
        repack_vnni_to_rowmajor_kernel,
        dim3(grid), dim3(BLOCK), 0, 0,
        d_B_vnni, d_B_rowmajor, N, K);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGemv_repackVNNI_to_rowmajor] Kernel launch failed: %s (N=%d K=%d)\n",
                hipGetErrorString(err), N, K);
        return false;
    }

    return true;
}

} // extern "C"
