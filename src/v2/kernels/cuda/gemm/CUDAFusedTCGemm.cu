/**
 * @file CUDAFusedTCGemm.cu
 * @brief Fused tensor-core GEMM for blockwise-quantized INT8 inference.
 *
 * Custom kernel that processes the *entire* K dimension in a single launch,
 * accumulating scaled INT32 tensor-core products into FP32 registers.
 *
 * Replaces the decomposed CUTLASS approach (one CUTLASS launch per K=32
 * block + separate accumulation kernel) with a monolithic kernel:
 *   1. Loads A[BM×32] and B[BN×32] into padded shared memory per K-step
 *   2. Computes INT8×INT8→INT32 via WMMA tensor-core MMA (m16n16k16 × 2)
 *   3. Applies per-block activation scale in-register (no global partials)
 *   4. Accumulates in FP32 registers across all K blocks
 *   5. Applies weight scaling + bias and writes final FP32 output
 *
 * Eliminates: per-K-block launch overhead, INT32 partial plane traffic,
 *             separate accumulation kernel launches.
 *
 * Requires sm_75+ (Turing). Uses WMMA m16n16k16 INT8 (2 MMA ops per K=32
 * block — both halves share the same activation scale, so INT32 partial
 * sums are correct before the FP32 scale conversion).
 *
 * Per-shape specialisation — tile config selected by aspect ratio:
 *
 *   Family     BM   BN   Warps   Threads  When
 *   ─────────  ───  ───  ──────  ───────  ────────────────────────────
 *   SkinnyM    32   128  1×4     128      M ≤ 32
 *   NarrowN   128    64  4×2     256      N ≤ 2048  (more grid blocks)
 *   DeepK      64   128  2×4     256      K > 2·N   (extra M-blocks)
 *   WideN      64   256  2×8     512      N ≥ 8192  (wide N tile)
 *   Balanced  128   128  2×4     256      default
 *
 * Shared memory uses SMEM_STRIDE = 48 (32 data + 16 pad) to reduce
 * bank conflicts from 4-way to 2-way on WMMA fragment loads.
 *
 * Data layout contract:
 *   A:        [M × K]       row-major INT8 (blockwise quantized, block=32)
 *   B:        [K/32][N][32] tc-blocked INT8 (column-major per 32-element group)
 *   C:        [M × N]       row-major FP32
 *   scales_A: [M × K/32]    per-row per-block activation scales
 *   scales_B: [N]           per-column weight scales
 */

#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>

using namespace nvcuda;

namespace
{

    // ─── Shared memory stride with padding ───
    // 32 bytes data per row/col + 16 bytes padding = 48 bytes.
    // WMMA requires lda to be a multiple of 16 for INT8: 48/16 = 3 ✓
    // Bank analysis (4-byte banks, 32 banks):
    //   row n starts at bank (n × 12) mod 32 → period 8 → 2-way conflict
    //   (unpadded: period 4 → 4-way conflict → 2× improvement)
    constexpr int BK = 32;                     // quantisation block size (fixed)
    constexpr int SMEM_PAD = 16;               // padding bytes per row
    constexpr int SMEM_STRIDE = BK + SMEM_PAD; // = 48

    // ─── Fragment layout helpers for m16n16k16 accumulator ───
    //
    // Each warp (32 threads) holds 8 INT32 elements per thread.
    // Elements span exactly 2 rows: groupId and groupId+8.
    // Elements 0,1,4,5 → row = groupId; 2,3,6,7 → row = groupId + 8.

    __device__ __forceinline__ int frag_row(int lane_id, int elem_idx)
    {
        return (lane_id >> 2) + ((elem_idx & 2) ? 8 : 0);
    }

    __device__ __forceinline__ int frag_col(int lane_id, int elem_idx)
    {
        return (elem_idx >> 2) * 8 + (lane_id & 3) * 2 + (elem_idx & 1);
    }

    // ════════════════════════════════════════════════════════════════════════
    // Fused tensor-core GEMM kernel template
    //
    // Template params:
    //   BM, BN          — block tile dimensions (output rows × cols per CTA)
    //   WARPS_M, WARPS_N — warp grid layout within the CTA
    //
    // Derived per-warp:
    //   WARP_M = BM / WARPS_M,   WARP_N = BN / WARPS_N
    //   WM = WARP_M / 16,        WN = WARP_N / 16  (WMMA tiles per warp)
    //
    // K loop: 2 × m16n16k16 per K=32 block.  Double-buffered shared memory.
    // ════════════════════════════════════════════════════════════════════════

    template <int BM, int BN, int WARPS_M, int WARPS_N>
    __global__ void __launch_bounds__(WARPS_M * WARPS_N * 32,
                                      (WARPS_M * WARPS_N <= 4) ? 2 : 1)
        fusedTCGemmKernel(
            const int8_t *__restrict__ A,
            const int8_t *__restrict__ B_tc,
            float *__restrict__ C,
            const float *__restrict__ scales_A,
            const float *__restrict__ scales_B,
            const float *__restrict__ C_existing,
            const float *__restrict__ bias,
            int M, int N, int K,
            float alpha, float beta)
    {
        constexpr int NUM_WARPS = WARPS_M * WARPS_N;
        constexpr int BLOCK_SIZE = NUM_WARPS * 32;
        constexpr int WARP_M = BM / WARPS_M;
        constexpr int WARP_N = BN / WARPS_N;
        constexpr int WM = WARP_M / 16;
        constexpr int WN = WARP_N / 16;

        static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0);
        static_assert(WARP_M % 16 == 0 && WARP_N % 16 == 0);

        const int warp_id = threadIdx.x >> 5;
        const int lane_id = threadIdx.x & 31;
        const int wr = warp_id / WARPS_N;
        const int wc = warp_id % WARPS_N;

        const int block_m = blockIdx.x * BM;
        const int block_n = blockIdx.y * BN;
        const int num_k_blocks = K / BK;

        // ─── Double-buffered shared memory (padded) ───
        __shared__ int8_t smem_A[2][BM * SMEM_STRIDE];
        __shared__ int8_t smem_B[2][BN * SMEM_STRIDE];

        // ─── FP32 accumulators (persist across all K blocks) ───
        float acc[WM][WN][8];
#pragma unroll
        for (int i = 0; i < WM; i++)
#pragma unroll
            for (int j = 0; j < WN; j++)
#pragma unroll
                for (int e = 0; e < 8; e++)
                    acc[i][j][e] = 0.0f;

        // ─── Tile load helpers (global → padded shared memory) ───
        // A: [M × K] row-major.  We load BM rows × 32 bytes, storing into
        //    smem_A[buf][row * SMEM_STRIDE + col] (padded stride).
        // Uses 128-bit (int4 = 16-byte) vectorized loads for 4× bandwidth.
        // Each row is exactly 32 bytes = two int4 loads.
        constexpr int A_VEC_LOADS = BM * BK / 16; // int4-loads for A tile

        auto load_A_tile = [&](int buf, int kb) __attribute__((always_inline))
        {
#pragma unroll 4
            for (int idx = threadIdx.x; idx < A_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4; // byte offset in dense tile
                const int row = linear >> 5; // / 32
                const int col = linear & 31; // % 32
                const int grow = block_m + row;
                int4 val = make_int4(0, 0, 0, 0);
                if (grow < M)
                    val = *reinterpret_cast<const int4 *>(
                        &A[static_cast<size_t>(grow) * K + kb * BK + col]);
                *reinterpret_cast<int4 *>(
                    &smem_A[buf][row * SMEM_STRIDE + col]) = val;
            }
        };

        // B: [K/32][N][32] tc-blocked.  Contiguous BN×32 bytes per K-block.
        // Same 128-bit vectorised loads.
        constexpr int B_VEC_LOADS = BN * BK / 16;

        auto load_B_tile = [&](int buf, int kb) __attribute__((always_inline))
        {
            const int8_t *b_base = B_tc + static_cast<size_t>(kb) * N * BK + static_cast<size_t>(block_n) * BK;
#pragma unroll 4
            for (int idx = threadIdx.x; idx < B_VEC_LOADS; idx += BLOCK_SIZE)
            {
                const int linear = idx << 4;
                const int col = linear >> 5;
                const int elem = linear & 31;
                const int gcol = block_n + col;
                int4 val = make_int4(0, 0, 0, 0);
                if (gcol < N)
                    val = *reinterpret_cast<const int4 *>(&b_base[linear]);
                *reinterpret_cast<int4 *>(
                    &smem_B[buf][col * SMEM_STRIDE + elem]) = val;
            }
        };

        // ─── Load first K block into buffer 0 ───
        load_A_tile(0, 0);
        load_B_tile(0, 0);
        __syncthreads();

        // ─── Main K-block loop with double buffering ───
        for (int kb = 0; kb < num_k_blocks; kb++)
        {
            const int cur = kb & 1;
            const int nxt = 1 - cur;

            // Prefetch next tile into alternate buffer
            if (kb + 1 < num_k_blocks)
            {
                load_A_tile(nxt, kb + 1);
                load_B_tile(nxt, kb + 1);
            }

            // ─── WMMA compute: 2 × m16n16k16 per K=32 block ───
            wmma::fragment<wmma::accumulator, 16, 16, 16, int32_t>
                c_frag[WM][WN];

#pragma unroll
            for (int i = 0; i < WM; i++)
#pragma unroll
                for (int j = 0; j < WN; j++)
                    wmma::fill_fragment(c_frag[i][j], 0);

#pragma unroll
            for (int kh = 0; kh < 2; kh++)
            {
                const int k_off = kh * 16;

                wmma::fragment<wmma::matrix_a, 16, 16, 16,
                               int8_t, wmma::row_major>
                    a_frag[WM];
                wmma::fragment<wmma::matrix_b, 16, 16, 16,
                               int8_t, wmma::col_major>
                    b_frag[WN];

#pragma unroll
                for (int wi = 0; wi < WM; wi++)
                {
                    const int srow = wr * WARP_M + wi * 16;
                    wmma::load_matrix_sync(a_frag[wi],
                                           &smem_A[cur][srow * SMEM_STRIDE + k_off], SMEM_STRIDE);
                }

#pragma unroll
                for (int wj = 0; wj < WN; wj++)
                {
                    const int scol = wc * WARP_N + wj * 16;
                    wmma::load_matrix_sync(b_frag[wj],
                                           &smem_B[cur][scol * SMEM_STRIDE + k_off], SMEM_STRIDE);
                }

#pragma unroll
                for (int wi = 0; wi < WM; wi++)
#pragma unroll
                    for (int wj = 0; wj < WN; wj++)
                        wmma::mma_sync(c_frag[wi][wj], a_frag[wi],
                                       b_frag[wj], c_frag[wi][wj]);
            }

// ─── Scale INT32 → FP32 and accumulate ───
#pragma unroll
            for (int wi = 0; wi < WM; wi++)
            {
                const int tile_m = block_m + wr * WARP_M + wi * 16;
                const int row0 = tile_m + (lane_id >> 2);
                const int row1 = row0 + 8;
                const float sa0 = (row0 < M)
                                      ? scales_A[row0 * num_k_blocks + kb]
                                      : 0.0f;
                const float sa1 = (row1 < M)
                                      ? scales_A[row1 * num_k_blocks + kb]
                                      : 0.0f;

#pragma unroll
                for (int wj = 0; wj < WN; wj++)
                {
#pragma unroll
                    for (int e = 0; e < 8; e++)
                    {
                        const float s = (e & 2) ? sa1 : sa0;
                        acc[wi][wj][e] +=
                            static_cast<float>(c_frag[wi][wj].x[e]) * s;
                    }
                }
            }

            __syncthreads();
        }

// ─── Epilogue: scale_B · alpha + beta · C_existing + bias → C ───
// The fragment layout means groups of 4 threads write consecutive
// column pairs.  For interior tiles (fully within bounds) we skip
// per-element bounds checks to let the compiler vectorise stores.
#pragma unroll
        for (int wi = 0; wi < WM; wi++)
        {
            const int tile_m = block_m + wr * WARP_M + wi * 16;

#pragma unroll
            for (int wj = 0; wj < WN; wj++)
            {
                const int tile_n = block_n + wc * WARP_N + wj * 16;
                const bool interior = (tile_m + 15 < M) && (tile_n + 15 < N);

                if (interior)
                {
// Fast path: no bounds checks
#pragma unroll
                    for (int e = 0; e < 8; e++)
                    {
                        const int gr = tile_m + frag_row(lane_id, e);
                        const int gc = tile_n + frag_col(lane_id, e);
                        const int out_idx = gr * N + gc;
                        float val = alpha * acc[wi][wj][e] * scales_B[gc];

                        if (beta != 0.0f && C_existing)
                            val += beta * C_existing[out_idx];
                        if (bias)
                            val += bias[gc];

                        C[out_idx] = val;
                    }
                }
                else
                {
// Edge path: bounds-checked
#pragma unroll
                    for (int e = 0; e < 8; e++)
                    {
                        const int gr = tile_m + frag_row(lane_id, e);
                        const int gc = tile_n + frag_col(lane_id, e);

                        if (gr < M && gc < N)
                        {
                            const int out_idx = gr * N + gc;
                            float val = alpha * acc[wi][wj][e] * scales_B[gc];

                            if (beta != 0.0f && C_existing)
                                val += beta * C_existing[out_idx];
                            if (bias)
                                val += bias[gc];

                            C[out_idx] = val;
                        }
                    }
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // Shape classification and dispatch
    // ════════════════════════════════════════════════════════════════════════

    enum class ShapeFamily
    {
        SkinnyM,  // M ≤ 32           → BM=32,  BN=128 (1×4, 128 threads)
        NarrowN,  // N ≤ 2048         → BM=128, BN=64  (4×2, 256 threads)
        DeepK,    // K > 2·N          → BM=64,  BN=128 (2×4, 256 threads)
        WideN,    // N ≥ 8192         → BM=64,  BN=256 (2×8, 512 threads)
        Balanced, // default           → BM=128, BN=128 (2×4, 256 threads)
        Compact,  // grid fallback     → BM=32,  BN=64  (1×2,  64 threads)
    };

    static const char *shapeFamilyName(ShapeFamily f)
    {
        switch (f)
        {
        case ShapeFamily::SkinnyM:
            return "skinny_m";
        case ShapeFamily::NarrowN:
            return "narrow_n";
        case ShapeFamily::DeepK:
            return "deep_k";
        case ShapeFamily::WideN:
            return "wide_n";
        case ShapeFamily::Balanced:
            return "balanced";
        case ShapeFamily::Compact:
            return "compact";
        }
        return "unknown";
    }

    // Last-selected family for perf harness introspection (host-side only).
    static int g_last_family = 4; // Balanced

    // Minimum grid blocks before we fall back to smaller tiles.
    // 64 is a reasonable threshold for modern GPUs (RTX 3090 = 82 SMs).
    static constexpr int MIN_GRID_BLOCKS = 64;

    static ShapeFamily classifyShape(int M, int N, int K)
    {
        if (M <= 32)
            return ShapeFamily::SkinnyM;

        // Standard aspect-ratio classification
        ShapeFamily family;
        if (N <= 2048)
            family = ShapeFamily::NarrowN;
        else if (K > 2 * N)
            family = ShapeFamily::DeepK;
        else if (N >= 8192)
            family = ShapeFamily::WideN;
        else
            family = ShapeFamily::Balanced;

        // Deep-K shapes are bandwidth-bound on the K-loop; smaller tiles
        // reduce per-CTA bandwidth without meaningful parallelism gain.
        if (K > 2 * N)
            return family;

        // Grid-size fallback for non-deep-K shapes:
        // When the natural tile produces too few blocks, switch to SkinnyM
        // (BM=32, BN=128) for higher grid-level parallelism.
        int bm, bn;
        switch (family)
        {
        case ShapeFamily::NarrowN:
            bm = 128;
            bn = 64;
            break;
        case ShapeFamily::DeepK:
            bm = 64;
            bn = 128;
            break;
        case ShapeFamily::WideN:
            bm = 64;
            bn = 256;
            break;
        default:
            bm = 128;
            bn = 128;
            break;
        }
        int grid_blocks = ((M + bm - 1) / bm) * ((N + bn - 1) / bn);

        if (grid_blocks < MIN_GRID_BLOCKS)
            return ShapeFamily::SkinnyM;

        return family;
    }

    static bool isTuringPlus(int device_id)
    {
        static int cached_device = -1;
        static bool cached_result = false;
        if (cached_device == device_id)
            return cached_result;

        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess)
            return false;

        cached_result = (prop.major > 7 || (prop.major == 7 && prop.minor >= 5));
        cached_device = device_id;
        return cached_result;
    }

    // ─── Launch helper (reduces dispatch boilerplate) ───
    template <int BM, int BN, int WM, int WN>
    static bool launchKernel(
        const int8_t *A, const int8_t *B_tc,
        float *C, const float *scales_A, const float *scales_B,
        const float *C_existing, const float *bias,
        int M, int N, int K, float alpha, float beta,
        cudaStream_t stream)
    {
        const dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
        const dim3 block(WM * WN * 32);
        fusedTCGemmKernel<BM, BN, WM, WN><<<grid, block, 0, stream>>>(
            A, B_tc, C, scales_A, scales_B, C_existing, bias,
            M, N, K, alpha, beta);
        return cudaGetLastError() == cudaSuccess;
    }

} // anonymous namespace

// ════════════════════════════════════════════════════════════════════════
// Public C API
// ════════════════════════════════════════════════════════════════════════

extern "C"
{
    bool cudaFusedTCGemm_blockwiseGemm(
        const int8_t *d_A_int8,
        const int8_t *d_weights_int8_tc_blocked,
        int32_t * /* d_partial_int32 — unused, kept for API compat */,
        float *d_C_fp32,
        const float *d_scales_A_block,
        const float *d_scales_B,
        int M, int N, int K,
        float alpha, float beta,
        const float *d_C_existing,
        const float *d_bias,
        int cuda_device_id,
        void *stream)
    {
        if (!d_A_int8 || !d_weights_int8_tc_blocked || !d_C_fp32 || !d_scales_A_block || !d_scales_B)
            return false;
        if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0)
            return false;
        if (!isTuringPlus(cuda_device_id))
            return false;

        cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
        if (cudaSetDevice(cuda_device_id) != cudaSuccess)
            return false;

        const ShapeFamily family = classifyShape(M, N, K);
        g_last_family = static_cast<int>(family);

        switch (family)
        {
        case ShapeFamily::SkinnyM:
            // BM=32, BN=128: 1×4 warps (128 threads)
            return launchKernel<32, 128, 1, 4>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);

        case ShapeFamily::NarrowN:
            // BM=128, BN=64: 4×2 warps (256 threads)
            return launchKernel<128, 64, 4, 2>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);

        case ShapeFamily::DeepK:
            // BM=64, BN=128: 2×4 warps (256 threads)
            return launchKernel<64, 128, 2, 4>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);

        case ShapeFamily::WideN:
            // BM=64, BN=256: 2×8 warps (512 threads)
            return launchKernel<64, 256, 2, 8>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);

        case ShapeFamily::Balanced:
            // BM=128, BN=128: 2×4 warps (256 threads)
            return launchKernel<128, 128, 2, 4>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);

        case ShapeFamily::Compact:
            // BM=32, BN=64: 1×2 warps (64 threads) — max grid parallelism
            return launchKernel<32, 64, 1, 2>(
                d_A_int8, d_weights_int8_tc_blocked,
                d_C_fp32, d_scales_A_block, d_scales_B,
                d_C_existing, d_bias, M, N, K, alpha, beta, cuda_stream);
        }

        return false;
    }

    const char *cudaFusedTCGemm_lastSelectedFamily()
    {
        return shapeFamilyName(static_cast<ShapeFamily>(g_last_family));
    }
}
