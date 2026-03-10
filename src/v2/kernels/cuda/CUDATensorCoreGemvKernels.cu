/**
 * @file CUDATensorCoreGemvKernels.cu
 * @brief CUDA DP4A INT8 GEMV kernel family with per-shape specializations.
 *
 * Mirrors the ROCm INT8-VNNI GEMV family at a high level, but remains a CUDA
 * SIMT/DP4A decode path rather than a tensor-core implementation:
 *   - **Wide**:     N >> K  (LM_Head, FFN_Up shapes) — maximizes N-parallelism
 *   - **GridKPar**: N ≈ K or K > N (Attention, FFN_Down) — splits K across CTAs,
 *                   atomicAdd reduction
 *   - **Direct**:   Fallback for small N or odd shapes
 *
 * All kernels use blockwise activation quantization (block_size=32) with FP32 output.
 *
 * Template parameters mirror the ROCm family naming where practical:
 *   TILE_N — output columns per CTA (128, 256)
 *   CPT    — columns per thread (1, 2, 4)
 *   WK     — warps per K-tile for shared-memory reduction
 *
 * @author Llaminar automated kernel generator
 * @date March 2026
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <atomic>
#include <sstream>
#include <stdexcept>

// =========================================================================
// Compile-time constants
// =========================================================================
static constexpr int BLOCKWISE_K = 32;                 // Activation quantization block size

// Shape-family tile defaults (can be overridden at runtime)
static constexpr int GEMV_WIDE_TILE_N_DEFAULT = 256;   // Wide: broad N coverage
static constexpr int GEMV_WIDE_CPT_DEFAULT    = 4;     // Wide: 4 outputs per thread
static constexpr int GEMV_KPAR_TILE_N_DEFAULT = 128;   // KPar: narrower N tile
static constexpr int GEMV_KPAR_CPT_DEFAULT    = 2;     // KPar: 2 outputs per thread
static constexpr int GEMV_DIRECT_TILE_N       = 128;   // Direct fallback

// SM occupancy target for K-parallel splitting (matches ROCm NUM_CUS=60 strategy)
static constexpr int TARGET_WAVES_PER_SM      = 8;
static constexpr int MIN_KGROUPS_PER_BLOCK    = 16;    // Proven floor from ROCm sweep

// =========================================================================
// Runtime tuning overrides (atomic, settable from host)
// =========================================================================
namespace {
    std::atomic<int> g_cuda_gemv_tn_override{0};
    std::atomic<int> g_cuda_gemv_cpt_override{0};
    std::atomic<int> g_cuda_gemv_kb_override{0};
    std::atomic<int> g_cuda_gemv_wk_override{0};
}

#define CUDA_GEMV_CHECK(call)                                                 \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::ostringstream oss__;                                           \
            oss__ << "[CUDABlockwiseGemv] CUDA error: "                        \
                  << cudaGetErrorString(err__) << " at " << __FILE__           \
                  << ":" << __LINE__;                                          \
            throw std::runtime_error(oss__.str());                             \
        }                                                                      \
    } while (0)

// =========================================================================
// K-split heuristic (mirrors ROCm select_blockwise_split_count)
// =========================================================================
static int selectKSplitCount(int grid_n, int k_groups, int num_sms, int kb_override)
{
    if (kb_override > 0)
        return std::min(kb_override, k_groups);

    int target_blocks = TARGET_WAVES_PER_SM * num_sms;
    int kb_raw = std::max(2, (target_blocks + grid_n - 1) / grid_n);

    // Cap: ensure minimum inner loop iterations
    int kb_max = std::max(2, k_groups / MIN_KGROUPS_PER_BLOCK);
    kb_raw = std::min(kb_raw, kb_max);

    // Round to nearest factor of k_groups (prefer lower → fewer atomicAdds)
    int kb = kb_raw;
    if (k_groups % kb_raw != 0)
    {
        for (int d = 1; d < kb_raw; ++d)
        {
            if (kb_raw - d >= 2 && k_groups % (kb_raw - d) == 0) { kb = kb_raw - d; break; }
            if (kb_raw + d <= kb_max && k_groups % (kb_raw + d) == 0) { kb = kb_raw + d; break; }
        }
    }

    return std::max(1, kb);
}

static int getCudaSmCount()
{
    static int sm_count = 0;
    if (sm_count == 0)
    {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
        if (sm_count <= 0) sm_count = 80;  // Conservative default (A100)
    }
    return sm_count;
}

// =========================================================================
// Kernel: Zero output buffer (for K-parallel atomic-reduce path)
// =========================================================================
__global__ void cudaGemv_zeroOutput(float* __restrict__ C, int N)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = 0.0f;
}

// =========================================================================
// Kernel family 1: WIDE GEMV — optimized for N >> K
//
// Each CTA owns TILE_N output columns, each thread owns CPT columns.
// Entire K-dimension processed in a single CTA (no K-splitting).
// Uses __dp4a for INT8×INT8→INT32 dot products within each activation block,
// then apply blockwise scale and accumulate into FP32.
//
// Uses packed 32-bit loads plus dp4a accumulation when CPT=4.
// =========================================================================
template <int TILE_N, int CPT>
__global__ void cudaGemv_wide_blockwise_kernel(
    const int8_t*  __restrict__ d_A_int8,           // [K] quantized activations
    const int8_t*  __restrict__ d_B_int8,           // [K × N] col-major weights
    float*         __restrict__ d_C_fp32,           // [N] output
    const float*   __restrict__ d_scales_A_block,   // [K/32] blockwise act scales
    const float*   __restrict__ d_scales_B,         // [N] per-column weight scales
    int N, int K,
    float alpha,
    const float*   __restrict__ d_C_existing,       // [N] for beta accumulation (nullable)
    const float*   __restrict__ d_bias,             // [N] optional bias (nullable)
    float beta)
{
    const int n_base = blockIdx.x * TILE_N + threadIdx.x * CPT;
    if (n_base >= N) return;

    const int num_k_blocks = K / BLOCKWISE_K;
    const int dp4a_groups  = BLOCKWISE_K / 4;

    float acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0.0f;

    // Process each activation block
    for (int blk = 0; blk < num_k_blocks; ++blk)
    {
        const int k_start = blk * BLOCKWISE_K;
        const float scale_a = d_scales_A_block[blk];

        int32_t partial[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) partial[c] = 0;

        // dp4a dot products across the 32-element K-block
        for (int g = 0; g < dp4a_groups; ++g)
        {
            const int32_t a_pack = *reinterpret_cast<const int32_t*>(
                d_A_int8 + k_start + g * 4);

            if constexpr (CPT == 4)
            {
                // Packed 32-bit loads for 4 consecutive output columns.
                const int n0 = n_base;
                if (n0 + 3 < N)
                {
                    // 4 columns are at B[n*K + k_start + g*4] for n = n0..n0+3
                    // In col-major: each column is contiguous in K
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        const int32_t b_pack = *reinterpret_cast<const int32_t*>(
                            d_B_int8 + static_cast<int64_t>(n0 + c) * K + k_start + g * 4);
                        partial[c] = __dp4a(a_pack, b_pack, partial[c]);
                    }
                }
                else
                {
                    #pragma unroll
                    for (int c = 0; c < 4; ++c)
                    {
                        if (n0 + c < N)
                        {
                            const int32_t b_pack = *reinterpret_cast<const int32_t*>(
                                d_B_int8 + static_cast<int64_t>(n0 + c) * K + k_start + g * 4);
                            partial[c] = __dp4a(a_pack, b_pack, partial[c]);
                        }
                    }
                }
            }
            else if constexpr (CPT == 2)
            {
                #pragma unroll
                for (int c = 0; c < 2; ++c)
                {
                    if (n_base + c < N)
                    {
                        const int32_t b_pack = *reinterpret_cast<const int32_t*>(
                            d_B_int8 + static_cast<int64_t>(n_base + c) * K + k_start + g * 4);
                        partial[c] = __dp4a(a_pack, b_pack, partial[c]);
                    }
                }
            }
            else
            {
                if (n_base < N)
                {
                    const int32_t b_pack = *reinterpret_cast<const int32_t*>(
                        d_B_int8 + static_cast<int64_t>(n_base) * K + k_start + g * 4);
                    partial[0] = __dp4a(a_pack, b_pack, partial[0]);
                }
            }
        }

        // Apply blockwise activation scale and accumulate
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            acc[c] += static_cast<float>(partial[c]) * scale_a;
        }
    }

    // Apply weight scales, alpha/beta, bias, and write output
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            float result = alpha * acc[c] * d_scales_B[n];
            if (beta != 0.0f && d_C_existing)
                result += beta * d_C_existing[n];
            if (d_bias)
                result += d_bias[n];
            d_C_fp32[n] = result;
        }
    }
}


// =========================================================================
// Kernel family 2: GRID K-PARALLEL GEMV — optimized for N ≈ K or K >> N
//
// Splits K-dimension across gridDim.y blocks. Each CTA processes a K-slice,
// accumulates partial FP32 results with blockwise scaling, then atomicAdd
// into global output.
//
// Shared memory used to broadcast the activation K-block to all threads.
// =========================================================================
template <int TILE_N, int CPT>
__global__ void cudaGemv_gridKPar_blockwise_kernel(
    const int8_t*  __restrict__ d_A_int8,           // [K] quantized activations
    const int8_t*  __restrict__ d_B_int8,           // [K × N] col-major weights
    float*         __restrict__ d_C_fp32,           // [N] output (atomicAdd)
    const float*   __restrict__ d_scales_A_block,   // [K/32] blockwise act scales
    const float*   __restrict__ d_scales_B,         // [N] per-column weight scales
    int N, int K,
    int kb,               // Total K-splits
    float alpha)
{
    const int n_base    = blockIdx.x * TILE_N + threadIdx.x * CPT;
    const int split_idx = blockIdx.y;

    if (n_base >= N) return;

    const int num_k_blocks = K / BLOCKWISE_K;
    const int blocks_per_split = (num_k_blocks + kb - 1) / kb;
    const int blk_begin = split_idx * blocks_per_split;
    const int blk_end   = min(num_k_blocks, blk_begin + blocks_per_split);

    if (blk_begin >= num_k_blocks) return;

    const int dp4a_groups = BLOCKWISE_K / 4;

    // Shared memory for A block broadcast
    __shared__ int8_t smem_A[BLOCKWISE_K];

    float acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0.0f;

    for (int blk = blk_begin; blk < blk_end; ++blk)
    {
        const int k_start = blk * BLOCKWISE_K;

        // Cooperative load of A block into shared memory
        if (threadIdx.x < BLOCKWISE_K)
        {
            smem_A[threadIdx.x] = d_A_int8[k_start + threadIdx.x];
        }
        __syncthreads();

        const float scale_a = d_scales_A_block[blk];

        int32_t partial[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) partial[c] = 0;

        for (int g = 0; g < dp4a_groups; ++g)
        {
            const int32_t a_pack = *reinterpret_cast<const int32_t*>(&smem_A[g * 4]);

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const int32_t b_pack = *reinterpret_cast<const int32_t*>(
                        d_B_int8 + static_cast<int64_t>(n) * K + k_start + g * 4);
                    partial[c] = __dp4a(a_pack, b_pack, partial[c]);
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            acc[c] += static_cast<float>(partial[c]) * scale_a;
        }

        __syncthreads();
    }

    // Atomic accumulation into global output
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            atomicAdd(&d_C_fp32[n], alpha * acc[c] * d_scales_B[n]);
        }
    }
}


// =========================================================================
// Kernel family 2b: LDS K-REDUCE GEMV — K-parallel with shared-memory
// reduction instead of atomicAdd. Uses WK warps per K-tile for better
// occupancy on large-K shapes.
//
// Grid: (ceil(N/TILE_N), KB_OUTER)
// Block: (TILE_N/CPT * WK) — WK warps cooperatively process K-blocks
// =========================================================================
template <int TILE_N, int CPT, int WK>
__global__ void cudaGemv_ldsKReduce_blockwise_kernel(
    const int8_t*  __restrict__ d_A_int8,
    const int8_t*  __restrict__ d_B_int8,
    float*         __restrict__ d_C_fp32,
    const float*   __restrict__ d_scales_A_block,
    const float*   __restrict__ d_scales_B,
    int N, int K,
    int kb_outer,
    float alpha)
{
    constexpr int THREADS_PER_N = TILE_N / CPT;
    const int tid        = threadIdx.x;
    const int n_lane     = tid % THREADS_PER_N;
    const int wk_lane    = tid / THREADS_PER_N;

    const int n_base     = blockIdx.x * TILE_N + n_lane * CPT;
    const int outer_idx  = blockIdx.y;

    if (n_base >= N) return;

    const int num_k_blocks = K / BLOCKWISE_K;
    const int blocks_per_outer = (num_k_blocks + kb_outer - 1) / kb_outer;
    const int k_begin = outer_idx * blocks_per_outer;
    const int k_end   = min(num_k_blocks, k_begin + blocks_per_outer);

    if (k_begin >= num_k_blocks) return;

    const int dp4a_groups = BLOCKWISE_K / 4;

    // Shared memory for per-warp partial results
    extern __shared__ float smem_partials[];
    // Layout: [WK][THREADS_PER_N * CPT]

    float acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c) acc[c] = 0.0f;

    // Each warp processes every WK-th K-block
    for (int blk = k_begin + wk_lane; blk < k_end; blk += WK)
    {
        const int k_start = blk * BLOCKWISE_K;
        const float scale_a = d_scales_A_block[blk];

        int32_t partial[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) partial[c] = 0;

        for (int g = 0; g < dp4a_groups; ++g)
        {
            const int32_t a_pack = *reinterpret_cast<const int32_t*>(
                d_A_int8 + k_start + g * 4);

            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    const int32_t b_pack = *reinterpret_cast<const int32_t*>(
                        d_B_int8 + static_cast<int64_t>(n) * K + k_start + g * 4);
                    partial[c] = __dp4a(a_pack, b_pack, partial[c]);
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            acc[c] += static_cast<float>(partial[c]) * scale_a;
        }
    }

    // Store per-warp partials to shared memory
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        smem_partials[wk_lane * THREADS_PER_N * CPT + n_lane * CPT + c] = acc[c];
    }
    __syncthreads();

    // Reduce across WK warps (wk_lane 0 does final accumulation)
    if (wk_lane == 0)
    {
        float final_acc[CPT];
        #pragma unroll
        for (int c = 0; c < CPT; ++c) final_acc[c] = 0.0f;

        for (int w = 0; w < WK; ++w)
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                final_acc[c] += smem_partials[w * THREADS_PER_N * CPT + n_lane * CPT + c];
            }
        }

        // Atomic accumulation across KB_OUTER splits
        #pragma unroll
        for (int c = 0; c < CPT; ++c)
        {
            const int n = n_base + c;
            if (n < N)
            {
                atomicAdd(&d_C_fp32[n], alpha * final_acc[c] * d_scales_B[n]);
            }
        }
    }
}


// =========================================================================
// Kernel family 3: DIRECT GEMV — simple fallback for small N or odd shapes
//
// One thread per output column, no shared memory, no K-splitting.
// Good for tiny N (e.g., KV projections: N=128-512).
// =========================================================================
__global__ void cudaGemv_direct_blockwise_kernel(
    const int8_t*  __restrict__ d_A_int8,
    const int8_t*  __restrict__ d_B_int8,
    float*         __restrict__ d_C_fp32,
    const float*   __restrict__ d_scales_A_block,
    const float*   __restrict__ d_scales_B,
    int N, int K,
    float alpha, float beta,
    const float*   __restrict__ d_C_existing,
    const float*   __restrict__ d_bias)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    const int num_k_blocks = K / BLOCKWISE_K;
    const int dp4a_groups  = BLOCKWISE_K / 4;

    float acc = 0.0f;
    for (int blk = 0; blk < num_k_blocks; ++blk)
    {
        const int k_start = blk * BLOCKWISE_K;
        const float scale_a = d_scales_A_block[blk];

        int32_t partial = 0;
        for (int g = 0; g < dp4a_groups; ++g)
        {
            const int32_t a_pack = *reinterpret_cast<const int32_t*>(
                d_A_int8 + k_start + g * 4);
            const int32_t b_pack = *reinterpret_cast<const int32_t*>(
                d_B_int8 + static_cast<int64_t>(n) * K + k_start + g * 4);
            partial = __dp4a(a_pack, b_pack, partial);
        }

        acc += static_cast<float>(partial) * scale_a;
    }

    float result = alpha * acc * d_scales_B[n];
    if (beta != 0.0f && d_C_existing)
        result += beta * d_C_existing[n];
    if (d_bias)
        result += d_bias[n];
    d_C_fp32[n] = result;
}


// =========================================================================
// Kernel: Apply bias + beta to zeroed output (for K-par paths)
// =========================================================================
__global__ void cudaGemv_applyBiasBeta(
    float* __restrict__ C,
    const float* __restrict__ C_existing,
    const float* __restrict__ bias,
    int N,
    float beta)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float val = C[n];
    if (beta != 0.0f && C_existing)
        val += beta * C_existing[n];
    if (bias)
        val += bias[n];
    C[n] = val;
}


// =========================================================================
// Shape classifier: determines which kernel family to use
// =========================================================================
enum class GemvShapeClass
{
    WIDE,        // N >> K — LM_Head, FFN_Up, FFN_Gate
    GRID_KPAR,   // K ≥ N or balanced — Attention, FFN_Down
    DIRECT,      // Small N (KV projections, etc.)
};

static GemvShapeClass classifyGemvShape(int N, int K)
{
    // K/V projections or very small N
    if (N <= 512)
        return GemvShapeClass::DIRECT;

    // Wide: N is much larger than K
    // Threshold: N ≥ 4×K captures LM_Head and FFN_Up shapes
    if (N >= 4 * K)
        return GemvShapeClass::WIDE;

    // Wide: even moderate N advantage benefits from wide path
    if (N >= 2 * K && N >= 2048)
        return GemvShapeClass::WIDE;

    // All other shapes: K-parallel is better
    return GemvShapeClass::GRID_KPAR;
}


// =========================================================================
// Dispatch macros for template instantiation
// =========================================================================
#define LAUNCH_WIDE(TN, CPT_VAL, stream) \
    cudaGemv_wide_blockwise_kernel<TN, CPT_VAL><<<grid, block, 0, stream>>>( \
        d_A_int8, d_B_int8, d_C_fp32, d_scales_A_block, d_scales_B,          \
        N, K, alpha, d_C_existing, d_bias, beta)

#define LAUNCH_KPAR(TN, CPT_VAL, stream) \
    cudaGemv_gridKPar_blockwise_kernel<TN, CPT_VAL><<<grid, block, 0, stream>>>( \
        d_A_int8, d_B_int8, d_C_fp32, d_scales_A_block, d_scales_B,              \
        N, K, kb, alpha)

#define LAUNCH_LDS_KREDUCE(TN, CPT_VAL, WK_VAL, stream)                          \
    cudaGemv_ldsKReduce_blockwise_kernel<TN, CPT_VAL, WK_VAL>                    \
        <<<grid, block, smem_bytes, stream>>>(                                    \
        d_A_int8, d_B_int8, d_C_fp32, d_scales_A_block, d_scales_B,              \
        N, K, kb_outer, alpha)


// =========================================================================
// Host dispatch function — single entry point for the CUDA DP4A GEMV path
// =========================================================================
extern "C"
{

void cudaTCGemv_setTuningOverrides(int tn, int cpt, int kb, int wk)
{
    g_cuda_gemv_tn_override.store(tn, std::memory_order_relaxed);
    g_cuda_gemv_cpt_override.store(cpt, std::memory_order_relaxed);
    g_cuda_gemv_kb_override.store(kb, std::memory_order_relaxed);
    g_cuda_gemv_wk_override.store(wk, std::memory_order_relaxed);
}

bool cudaTCGemv_blockwiseGemv(
    const int8_t*  d_A_int8,           // [K] quantized activations
    const int8_t*  d_B_int8,           // [K × N] col-major weights
    float*         d_C_fp32,           // [N] output
    const float*   d_scales_A_block,   // [K/32] blockwise activation scales
    const float*   d_scales_B,         // [N] per-column weight scales
    int N, int K,
    float alpha, float beta,
    const float*   d_C_existing,       // [N] for beta accumulation (nullable)
    const float*   d_bias,             // [N] optional bias (nullable)
    int cuda_device_id,
    void* stream)
{
    if (!d_A_int8 || !d_B_int8 || !d_C_fp32 || !d_scales_A_block || !d_scales_B)
        return false;
    if (N <= 0 || K <= 0 || (K % BLOCKWISE_K) != 0)
        return false;

    CUDA_GEMV_CHECK(cudaSetDevice(cuda_device_id));
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);

    // Read runtime overrides
    const int tn_override  = g_cuda_gemv_tn_override.load(std::memory_order_relaxed);
    const int cpt_override = g_cuda_gemv_cpt_override.load(std::memory_order_relaxed);
    const int kb_override  = g_cuda_gemv_kb_override.load(std::memory_order_relaxed);
    const int wk_override  = g_cuda_gemv_wk_override.load(std::memory_order_relaxed);

    const int num_sms = getCudaSmCount();
    const GemvShapeClass shape = classifyGemvShape(N, K);

    switch (shape)
    {
    // =====================================================================
    // WIDE path: N >> K
    // =====================================================================
    case GemvShapeClass::WIDE:
    {
        // Select tile size
        int tn = GEMV_WIDE_TILE_N_DEFAULT;  // 256
        if (tn_override == 128 || tn_override == 256)
            tn = tn_override;
        // For very large N (LM_Head ≈ 152K), 256 tiles give better occupancy
        if (N > 50000 && tn_override == 0)
            tn = 256;

        // Select CPT
        int cpt = GEMV_WIDE_CPT_DEFAULT;  // 4
        if (cpt_override == 1 || cpt_override == 2 || cpt_override == 4)
            cpt = cpt_override;

        const int grid_n = (N + tn - 1) / tn;
        dim3 grid(grid_n);

        if (tn == 256 && cpt == 4)
        {
            dim3 block(256 / 4);  // 64 threads
            LAUNCH_WIDE(256, 4, cuda_stream);
        }
        else if (tn == 256 && cpt == 2)
        {
            dim3 block(256 / 2);  // 128 threads
            LAUNCH_WIDE(256, 2, cuda_stream);
        }
        else if (tn == 128 && cpt == 4)
        {
            dim3 block(128 / 4);  // 32 threads
            LAUNCH_WIDE(128, 4, cuda_stream);
        }
        else if (tn == 128 && cpt == 2)
        {
            dim3 block(128 / 2);  // 64 threads
            LAUNCH_WIDE(128, 2, cuda_stream);
        }
        else
        {
            // Default: TN=256, CPT=4
            dim3 block(256 / 4);
            LAUNCH_WIDE(256, 4, cuda_stream);
        }

        CUDA_GEMV_CHECK(cudaGetLastError());
        return true;
    }

    // =====================================================================
    // GRID K-PARALLEL path: K ≥ N or balanced shapes
    // =====================================================================
    case GemvShapeClass::GRID_KPAR:
    {
        int tn = GEMV_KPAR_TILE_N_DEFAULT;  // 128
        if (tn_override == 128 || tn_override == 256)
            tn = tn_override;

        int cpt = GEMV_KPAR_CPT_DEFAULT;  // 2
        if (cpt_override == 1 || cpt_override == 2 || cpt_override == 4)
            cpt = cpt_override;

        int wk = (wk_override > 0) ? wk_override : 4;
        // Clamp WK to valid range
        if (wk < 1) wk = 1;
        if (wk > 8) wk = 8;

        const int grid_n = (N + tn - 1) / tn;
        const int k_groups = K / 4;  // For int32 packing
        const int num_k_blocks = K / BLOCKWISE_K;

        // K-par: LDS reduce path for large K
        // Use LDS K-reduce for shapes with enough K to benefit from multi-warp
        if (num_k_blocks >= 8 && wk >= 2)
        {
            const int kb_outer = selectKSplitCount(grid_n, num_k_blocks, num_sms, kb_override);
            const int threads_per_n = tn / cpt;

            // Zero output before atomic accumulation
            {
                const int zero_threads = 256;
                const int zero_blocks = (N + zero_threads - 1) / zero_threads;
                cudaGemv_zeroOutput<<<zero_blocks, zero_threads, 0, cuda_stream>>>(d_C_fp32, N);
                CUDA_GEMV_CHECK(cudaGetLastError());
            }

            dim3 grid(grid_n, kb_outer);

            if (tn == 128 && cpt == 2)
            {
                const int smem_bytes = wk * (128 / 2) * 2 * sizeof(float);
                dim3 block((128 / 2) * wk);
                switch (wk)
                {
                case 2: LAUNCH_LDS_KREDUCE(128, 2, 2, cuda_stream); break;
                case 3: LAUNCH_LDS_KREDUCE(128, 2, 3, cuda_stream); break;
                case 4: LAUNCH_LDS_KREDUCE(128, 2, 4, cuda_stream); break;
                case 6: LAUNCH_LDS_KREDUCE(128, 2, 6, cuda_stream); break;
                case 8: LAUNCH_LDS_KREDUCE(128, 2, 8, cuda_stream); break;
                default: LAUNCH_LDS_KREDUCE(128, 2, 4, cuda_stream); break;
                }
            }
            else if (tn == 256 && cpt == 2)
            {
                const int smem_bytes = wk * (256 / 2) * 2 * sizeof(float);
                dim3 block((256 / 2) * wk);
                switch (wk)
                {
                case 2: LAUNCH_LDS_KREDUCE(256, 2, 2, cuda_stream); break;
                case 3: LAUNCH_LDS_KREDUCE(256, 2, 3, cuda_stream); break;
                case 4: LAUNCH_LDS_KREDUCE(256, 2, 4, cuda_stream); break;
                case 6: LAUNCH_LDS_KREDUCE(256, 2, 6, cuda_stream); break;
                case 8: LAUNCH_LDS_KREDUCE(256, 2, 8, cuda_stream); break;
                default: LAUNCH_LDS_KREDUCE(256, 2, 4, cuda_stream); break;
                }
            }
            else
            {
                // Fallback to simple grid kpar
                const int kb = selectKSplitCount(grid_n, k_groups, num_sms, kb_override);
                dim3 grid_kpar(grid_n, kb);
                dim3 block(tn / cpt);
                if (tn == 128)
                {
                    LAUNCH_KPAR(128, 2, cuda_stream);
                }
                else
                {
                    LAUNCH_KPAR(256, 2, cuda_stream);
                }
            }

            CUDA_GEMV_CHECK(cudaGetLastError());

            // Apply bias and beta after atomic accumulation
            if (beta != 0.0f || d_bias)
            {
                const int post_threads = 256;
                const int post_blocks = (N + post_threads - 1) / post_threads;
                cudaGemv_applyBiasBeta<<<post_blocks, post_threads, 0, cuda_stream>>>(
                    d_C_fp32, d_C_existing, d_bias, N, beta);
                CUDA_GEMV_CHECK(cudaGetLastError());
            }

            return true;
        }
        else
        {
            // Simple grid K-par for modest K
            const int kb = selectKSplitCount(grid_n, k_groups, num_sms, kb_override);

            // Zero output
            {
                const int zero_threads = 256;
                const int zero_blocks = (N + zero_threads - 1) / zero_threads;
                cudaGemv_zeroOutput<<<zero_blocks, zero_threads, 0, cuda_stream>>>(d_C_fp32, N);
                CUDA_GEMV_CHECK(cudaGetLastError());
            }

            dim3 grid(grid_n, kb);
            dim3 block(tn / cpt);
            if (tn == 128 && cpt == 2)
            {
                LAUNCH_KPAR(128, 2, cuda_stream);
            }
            else if (tn == 256 && cpt == 2)
            {
                LAUNCH_KPAR(256, 2, cuda_stream);
            }
            else
            {
                LAUNCH_KPAR(128, 2, cuda_stream);
            }

            CUDA_GEMV_CHECK(cudaGetLastError());

            if (beta != 0.0f || d_bias)
            {
                const int post_threads = 256;
                const int post_blocks = (N + post_threads - 1) / post_threads;
                cudaGemv_applyBiasBeta<<<post_blocks, post_threads, 0, cuda_stream>>>(
                    d_C_fp32, d_C_existing, d_bias, N, beta);
                CUDA_GEMV_CHECK(cudaGetLastError());
            }

            return true;
        }
    }

    // =====================================================================
    // DIRECT path: Small N or unusual shapes
    // =====================================================================
    case GemvShapeClass::DIRECT:
    {
        const int threads = 256;
        const int blocks = (N + threads - 1) / threads;
        cudaGemv_direct_blockwise_kernel<<<blocks, threads, 0, cuda_stream>>>(
            d_A_int8, d_B_int8, d_C_fp32, d_scales_A_block, d_scales_B,
            N, K, alpha, beta, d_C_existing, d_bias);

        CUDA_GEMV_CHECK(cudaGetLastError());
        return true;
    }
    }

    return false;
}

} // extern "C"

#undef LAUNCH_WIDE
#undef LAUNCH_KPAR
#undef LAUNCH_LDS_KREDUCE
