/**
 * @file ROCmArgmaxKernels.hip
 * @brief ROCm/HIP Argmax kernel for greedy sampling
 *
 * Provides a GPU-side argmax over FP32 logits, returning the index and value
 * of the maximum element. Used for greedy decode sampling to avoid D2H of
 * the entire logits tensor (~600 KB for 152K vocab).
 *
 * Mirrors the CUDA implementation in CUDASamplingKernels.cu for cross-backend
 * parity: a two-pass multi-block reduction (pass 1 spreads the vocab across
 * many blocks/CUs producing one partial per block; pass 2 reduces the partials
 * in a single small block). The caller supplies the per-block partial scratch
 * (owned by the orchestrator's BufferArena); there is no single-block fallback.
 *
 * Performance: ~5µs for 76K elements on MI50 vs ~170µs for CPU std::max_element
 */

#include <hip/hip_runtime.h>
#include <cfloat>
#include <cstdio>
#include "../../../backends/rocm/HipDeviceGuard.h"

// ── Argmax multi-block reduction tuning ─────────────────────────────────────
// Threads per block for both reduction passes. Must be a power of two because
// the in-block tree reduction halves the active range each step.
constexpr int ARGMAX_REDUCE_THREADS = 256;
constexpr int ARGMAX_FINALIZE_THREADS = 256;
// Target elements processed per thread in pass 1. Chosen so each thread reads a
// handful of elements (good memory coalescing) while still spreading the vocab
// across enough blocks to occupy most CUs. 8 → ~74 blocks for a 152K vocab.
constexpr int ARGMAX_ELEMS_PER_THREAD = 8;

// ============================================================================
// Argmax Pass 1 — Multi-block partial reduction
// ============================================================================
//
// Each block performs a grid-strided scan over its slice of the input and
// reduces it (in shared memory) to a single (value, index) partial, written to
// partial_vals[blockIdx.x] / partial_idxs[blockIdx.x]. Spreading the work over
// gridDim.x blocks lets the reduction use many CUs instead of a single one.
__global__ void rocm_argmax_partial_f32_kernel(
    const float *__restrict__ data,
    int n,
    float *__restrict__ partial_vals,
    int *__restrict__ partial_idxs)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * blockDim.x + tid;
    const int grid_stride = blockDim.x * gridDim.x;

    // Phase 1: Grid-strided scan — each thread reduces its share to a local max.
    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = gid; i < n; i += grid_stride)
    {
        float val = data[i];
        if (val > local_max)
        {
            local_max = val;
            local_idx = i;
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    // Phase 2: In-block tree reduction (blockDim.x is a power of two).
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (smax[tid + stride] > smax[tid])
            {
                smax[tid] = smax[tid + stride];
                sidx[tid] = sidx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Phase 3: Thread 0 emits this block's partial result.
    if (tid == 0)
    {
        partial_vals[blockIdx.x] = smax[0];
        partial_idxs[blockIdx.x] = sidx[0];
    }
}

// ============================================================================
// Argmax Pass 2 — Finalize over per-block partials (single block)
// ============================================================================
//
// Reduces the `num_partials` partial results from pass 1 into the final
// (value, index). Runs as a single small block; num_partials is bounded by the
// pass-1 grid size (a few hundred at most), so this is cheap.
__global__ void rocm_argmax_finalize_f32_kernel(
    const float *__restrict__ partial_vals,
    const int *__restrict__ partial_idxs,
    int num_partials,
    float *__restrict__ out_value,
    int *__restrict__ out_index)
{
    extern __shared__ char shared_mem[];
    float *smax = reinterpret_cast<float *>(shared_mem);
    int *sidx = reinterpret_cast<int *>(shared_mem + blockDim.x * sizeof(float));

    const int tid = threadIdx.x;

    // Phase 1: Strided scan of the partials into a per-thread local max.
    float local_max = -FLT_MAX;
    int local_idx = 0;
    for (int i = tid; i < num_partials; i += blockDim.x)
    {
        float val = partial_vals[i];
        if (val > local_max)
        {
            local_max = val;
            local_idx = partial_idxs[i];
        }
    }

    smax[tid] = local_max;
    sidx[tid] = local_idx;
    __syncthreads();

    // Phase 2: In-block tree reduction (blockDim.x is a power of two).
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            if (smax[tid + stride] > smax[tid])
            {
                smax[tid] = smax[tid + stride];
                sidx[tid] = sidx[tid + stride];
            }
        }
        __syncthreads();
    }

    // Phase 3: Thread 0 writes the global argmax.
    if (tid == 0)
    {
        *out_value = smax[0];
        *out_index = sidx[0];
    }
}

// ============================================================================
// Extern "C" Wrapper
// ============================================================================

extern "C"
{

/**
 * @brief Launch argmax kernel on device (two-pass multi-block reduction)
 *
 * Finds the index and value of the maximum element in a float array.
 *
 * @param data             Device pointer to FP32 data
 * @param n                Number of elements
 * @param out_value        Device pointer to store max value (1 float)
 * @param out_index        Device pointer to store max index (1 int)
 * @param partial_vals     Device scratch for pass-1 per-block partial values
 * @param partial_idxs     Device scratch for pass-1 per-block partial indices
 * @param partial_capacity Number of entries in the partial scratch (bounds grid)
 * @param device_idx       GPU device index
 * @param stream           HIP stream (nullptr for default)
 * @return true on success
 */
bool rocmOps_argmax_f32(
    const float* data,
    int n,
    float* out_value,
    int* out_index,
    float* partial_vals,
    int* partial_idxs,
    int partial_capacity,
    int device_idx,
    void* stream)
{
    if (n <= 0 || !data || !out_value || !out_index)
        return false;

    // The partial-reduction scratch is mandatory: every production caller
    // supplies arena-owned scratch (single-device orchestrator and the
    // multi-device sampler). There is no single-block fallback — a missing
    // or undersized scratch buffer is a wiring bug, so fail loud.
    if (!partial_vals || !partial_idxs || partial_capacity < 1)
    {
        fprintf(stderr,
                "ROCm Argmax FP32: missing partial-reduction scratch "
                "(partial_vals=%p partial_idxs=%p capacity=%d)\n",
                (void *)partial_vals, (void *)partial_idxs, partial_capacity);
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    hipStream_t s = static_cast<hipStream_t>(stream);

    // Two-pass multi-block reduction.
    // Size the pass-1 grid so each thread processes ~ARGMAX_ELEMS_PER_THREAD
    // elements, then clamp to the partial-buffer capacity. This spreads the
    // vocab across many blocks (and thus CUs) instead of a single block.
    const int threads = ARGMAX_REDUCE_THREADS;
    const long per_block = static_cast<long>(threads) * ARGMAX_ELEMS_PER_THREAD;
    long blocks = (static_cast<long>(n) + per_block - 1) / per_block;
    if (blocks < 1)
        blocks = 1;
    if (blocks > partial_capacity)
        blocks = partial_capacity;
    const int num_blocks = static_cast<int>(blocks);

    // Pass 1: each block reduces its slice to one partial.
    const size_t smem1 = threads * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(
        rocm_argmax_partial_f32_kernel,
        dim3(num_blocks), dim3(threads), smem1, s,
        data, n, partial_vals, partial_idxs);

    // Pass 2: single small block reduces the partials to the final result.
    const int fthreads = ARGMAX_FINALIZE_THREADS;
    const size_t smem2 = fthreads * (sizeof(float) + sizeof(int));
    hipLaunchKernelGGL(
        rocm_argmax_finalize_f32_kernel,
        dim3(1), dim3(fthreads), smem2, s,
        partial_vals, partial_idxs, num_blocks, out_value, out_index);

    (void)hipGetLastError();  // Clear stale errors
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "ROCm Argmax FP32 (multi-block) launch failed: %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
