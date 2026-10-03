/**
 * @file ROCmGatedDeltaNetKernels.hip
 * @brief HIP kernels for Gated Delta Net (GDN) recurrence and short convolution
 *
 * ROCm/HIP port of the CUDA GDN kernels. Implements:
 * - Delta rule recurrence (decode + prefill)
 * - Short causal conv1d + SiLU
 * - Gated RMSNorm
 * - Attention output gate (sigmoid)
 *
 * Long bounded prefill owns recurrent columns per wave and caches independent
 * inputs cooperatively. Every token still uses scalar decode's rounded state
 * updates and ordered four-part key reduction. No prefix-scan reassociation,
 * host replay, global scratch or dynamic allocation is required. Capture-time
 * snapshot policy separates terminal-only publication from bounded row output.
 *
 * Key HIP differences from CUDA:
 * - 64-thread wavefronts (vs 32 warps), shuffle width adjusted
 * - hip_runtime.h / hip_fp16.h headers
 * - __expf → expf (HIP math)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "../../../utils/DebugEnv.h"
#include "kernels/gdn/GDNDeinterleaveRows.h"

namespace {

const char *memory_type_name(hipMemoryType type)
{
    switch (type)
    {
    case hipMemoryTypeHost: return "host";
    case hipMemoryTypeDevice: return "device";
    case hipMemoryTypeArray: return "array";
    case hipMemoryTypeUnified: return "unified";
    default: return "unknown";
    }
}

void print_pointer_attrs(const char *label, const void *ptr)
{
    hipPointerAttribute_t attrs{};
    hipError_t err = hipPointerGetAttributes(&attrs, ptr);
    if (err != hipSuccess)
    {
        fprintf(stderr, "[rocmGDN_ptr] %s=%p attrs=%s\n", label, ptr, hipGetErrorString(err));
        return;
    }
    fprintf(stderr, "[rocmGDN_ptr] %s=%p type=%s device=%d devicePointer=%p hostPointer=%p\n",
            label, ptr, memory_type_name(attrs.type), attrs.device,
            attrs.devicePointer, attrs.hostPointer);
}

} // anonymous namespace (diagnostics)

#include "../../../backends/rocm/HipDeviceGuard.h"

namespace { // anonymous namespace to avoid symbol conflicts with CUDA kernels

constexpr int kGdnRecurrentThreads = 256;
constexpr int kGdnRecurrentRowSplit = 4;
constexpr int kGdnRecurrentColumnsPerBlock =
    kGdnRecurrentThreads / kGdnRecurrentRowSplit;

/// @brief Bounded independent-input tile for wave-owned recurrence columns.
constexpr int kGdnPrefillInputTileRows = 16;

/** @brief Capture-time state publication contract, independent of live row count. */
enum class GdnPrefillSnapshots
{
    TerminalOnly, ///< Publish only the final recurrent state, as ordinary prefill requires.
    BoundedRows,  ///< Also publish each live row admitted by the caller's snapshot bank.
};

/**
 * @brief Advance grouped recurrence rows while retaining state in VGPRs.
 *
 * Scalar decode and grouped MTP verification both enter this kernel. M=1 uses
 * one loop iteration; a runtime-sized verifier matrix keeps one block resident
 * and advances every row in causal order. Each lane owns one fixed partition
 * of a value column and loads that partition from global state exactly once.
 * The first twenty-two rows remain in VGPRs; any remaining rows occupy an explicit
 * coalesced LDS tile. This split prevents clang from silently spilling the
 * Qwen-width register array into private memory while preserving the same
 * ascending key-row arithmetic. Post-row snapshots remain complete, but the
 * speculative work bank is written only once after the final row rather than
 * being read twice and written once for every row.
 *
 * The four-way key-row split, wave64 normalization reduction, gate expressions,
 * K-part reduction order, and state update expressions are deliberately the
 * same as scalar HIP decode. Register retention changes storage traffic, not
 * arithmetic order. The all-M captured publication sweep proves every grouped
 * output, selected snapshot, and continuation byte against serial M=1 decode.
 *
 * Grid X flattens request/head identity and grid Y owns independent value
 * columns. Snapshot rows use the flattened request-row namespace, so the launch
 * publishes every possible accepted state without host replay.
 *
 * @tparam D_K Compile-time key width. Supported graph geometries select a
 *             specialization before capture so register extent and occupancy
 *             are fixed properties of the captured launch.
 */
template <int D_K>
__global__ __launch_bounds__(kGdnRecurrentThreads, 2)
void rocm_gdn_recurrent_step_kernel(
    const float *__restrict__ q,
    const float *__restrict__ k,
    const float *__restrict__ v,
    const float *__restrict__ alpha,
    const float *__restrict__ beta_raw,
    const float *__restrict__ A_log,
    const float *__restrict__ dt_bias,
    float *__restrict__ output,
    const float *initial_state,
    float *updated_state,
    int request_count, int request_seq_len,
    int n_heads, int d_v,
    bool use_qk_l2norm,
    const int *__restrict__ effective_seq_len_ptr,
    int effective_row_idx,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows)
{
    const int request_head = blockIdx.x;
    const int request = request_head / n_heads;
    const int h = request_head - request * n_heads;
    if (request >= request_count || h >= n_heads) return;

    static_assert(D_K % kGdnRecurrentRowSplit == 0);
    constexpr int kRowsPerSplit = D_K / kGdnRecurrentRowSplit;
    constexpr int kRegisterRowsPerSplit =
        kRowsPerSplit < 22 ? kRowsPerSplit : 22;
    constexpr int kLdsRowsPerSplit =
        kRowsPerSplit - kRegisterRowsPerSplit;

    const int tid = threadIdx.x;
    const int split_id = tid / kGdnRecurrentColumnsPerBlock;
    const int col_in_block = tid % kGdnRecurrentColumnsPerBlock;
    const int vi =
        blockIdx.y * kGdnRecurrentColumnsPerBlock + col_in_block;

    const size_t row_qk_stride =
        static_cast<size_t>(n_heads) * static_cast<size_t>(D_K);
    const size_t row_v_stride =
        static_cast<size_t>(n_heads) * static_cast<size_t>(d_v);
    const size_t request_state_stride =
        static_cast<size_t>(n_heads) * static_cast<size_t>(D_K) *
        static_cast<size_t>(d_v);
    const float *S_initial =
        initial_state + static_cast<size_t>(request) * request_state_stride +
        static_cast<size_t>(h) * D_K * d_v;
    float *S_updated =
        updated_state + static_cast<size_t>(request) * request_state_stride +
        static_cast<size_t>(h) * D_K * d_v;

    extern __shared__ float smem[];
    float *q_local = smem;
    float *k_local = smem + D_K;
    float *wave_sums = smem + 2 * D_K;
    float *reduce_kv = wave_sums + 4;
    float *reduce_out = reduce_kv + kGdnRecurrentThreads;
    float *state_tail = reduce_out + kGdnRecurrentThreads;

    const float scale = rsqrtf(static_cast<float>(D_K));
    const int warp_id = tid / 64;
    const int lane_id = tid % 64;
    constexpr int kWaveCount = kGdnRecurrentThreads / 64;

    /*
     * The bounded compile-time array is scalarized into VGPRs by clang. The
     * explicit LDS tail replaces what would otherwise become compiler-owned
     * private scratch at D_K=128. M=16 still performs one global state load and
     * one final global state store; LDS is reused wholly inside the launch.
     */
    float state_rows[kRegisterRowsPerSplit];
    const int row_begin = split_id * kRowsPerSplit;
    if (vi < d_v)
    {
#pragma unroll
        for (int local_row = 0;
             local_row < kRegisterRowsPerSplit;
             ++local_row)
        {
            const int state_row = row_begin + local_row;
            state_rows[local_row] = S_initial[state_row * d_v + vi];
        }
#pragma clang loop unroll(disable)
        for (int tail_row = 0; tail_row < kLdsRowsPerSplit; ++tail_row)
        {
            const int state_row =
                row_begin + kRegisterRowsPerSplit + tail_row;
            state_tail[tail_row * kGdnRecurrentThreads + tid] =
                S_initial[state_row * d_v + vi];
        }
    }
    else
    {
#pragma unroll
        for (int local_row = 0;
             local_row < kRegisterRowsPerSplit;
             ++local_row)
            state_rows[local_row] = 0.0f;
#pragma clang loop unroll(disable)
        for (int tail_row = 0; tail_row < kLdsRowsPerSplit; ++tail_row)
        {
            state_tail[tail_row * kGdnRecurrentThreads + tid] = 0.0f;
        }
    }
    __syncthreads();

    for (int request_row = 0; request_row < request_seq_len; ++request_row)
    {
        const int flat_row = request * request_seq_len + request_row;
        const float *q_head =
            q + static_cast<size_t>(flat_row) * row_qk_stride +
            static_cast<size_t>(h) * D_K;
        const float *k_head =
            k + static_cast<size_t>(flat_row) * row_qk_stride +
            static_cast<size_t>(h) * D_K;
        const float *v_head =
            v + static_cast<size_t>(flat_row) * row_v_stride +
            static_cast<size_t>(h) * d_v;
        float *o_head =
            output + static_cast<size_t>(flat_row) * row_v_stride +
            static_cast<size_t>(h) * d_v;

        const int logical_row = effective_row_idx + request_row;
        if (effective_seq_len_ptr &&
            logical_row >= effective_seq_len_ptr[request])
        {
            if (vi < d_v && split_id == 0)
                o_head[vi] = 0.0f;
            __syncthreads();
            continue;
        }

        // Preserve scalar decode's 256-thread Q/K normalization tree.
        for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
        {
            q_local[i] = q_head[i];
            k_local[i] = k_head[i];
        }
        __syncthreads();

        if (use_qk_l2norm)
        {
            float q_sum = 0.0f;
            for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
                q_sum += q_local[i] * q_local[i];
            for (int offset = 32; offset > 0; offset >>= 1)
                q_sum += __shfl_xor(q_sum, offset);
            if (lane_id == 0)
                wave_sums[warp_id] = q_sum;
            __syncthreads();
            if (tid == 0)
            {
                float total = 0.0f;
                for (int wave = 0; wave < kWaveCount; ++wave)
                    total += wave_sums[wave];
                wave_sums[0] = total;
            }
            __syncthreads();
            const float q_inv =
                scale / fmaxf(sqrtf(wave_sums[0]), 1e-6f);
            for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
                q_local[i] *= q_inv;
            __syncthreads();

            float k_sum = 0.0f;
            for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
                k_sum += k_local[i] * k_local[i];
            for (int offset = 32; offset > 0; offset >>= 1)
                k_sum += __shfl_xor(k_sum, offset);
            if (lane_id == 0)
                wave_sums[warp_id] = k_sum;
            __syncthreads();
            if (tid == 0)
            {
                float total = 0.0f;
                for (int wave = 0; wave < kWaveCount; ++wave)
                    total += wave_sums[wave];
                wave_sums[0] = total;
            }
            __syncthreads();
            const float k_inv =
                1.0f / fmaxf(sqrtf(wave_sums[0]), 1e-6f);
            for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
                k_local[i] *= k_inv;
            __syncthreads();
        }
        else
        {
            for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
                q_local[i] *= scale;
            __syncthreads();
        }

        if (tid == 0)
        {
            const size_t gate_index =
                static_cast<size_t>(flat_row) *
                    static_cast<size_t>(n_heads) +
                h;
            const float x = alpha[gate_index] + dt_bias[h];
            const float sp = (x > 20.0f) ? x : log1pf(expf(x));
            wave_sums[0] = expf(A_log[h] * sp);
            wave_sums[1] =
                1.0f / (1.0f + expf(-beta_raw[gate_index]));
        }
        __syncthreads();
        const float decay = wave_sums[0];
        const float beta_h = wave_sums[1];

        /*
         * Pass 1 installs the rounded decay into resident state and then uses
         * that exact value for K^T S. CPU recurrence and CUDA long prefill
         * already define this canonical five-step GDN arithmetic. Keeping the
         * rounded value avoids a second state*decay operation in pass two while
         * making M=1, grouped verification, and long prefill share one contract.
         */
        float partial_kv = 0.0f;
        if (vi < d_v)
        {
#pragma unroll
            for (int local_row = 0;
                 local_row < kRegisterRowsPerSplit;
                 ++local_row)
            {
                const int state_row = row_begin + local_row;
                state_rows[local_row] *= decay;
                partial_kv += state_rows[local_row] * k_local[state_row];
            }
#pragma clang loop unroll(disable)
            for (int tail_row = 0;
                 tail_row < kLdsRowsPerSplit;
                 ++tail_row)
            {
                const int state_row =
                    row_begin + kRegisterRowsPerSplit + tail_row;
                const int lds_index =
                    tail_row * kGdnRecurrentThreads + tid;
                const float s_decayed = state_tail[lds_index] * decay;
                state_tail[lds_index] = s_decayed;
                partial_kv += s_decayed * k_local[state_row];
            }
        }
        reduce_kv[tid] = partial_kv;
        __syncthreads();

        // The split reduction order is fixed and shared with M=1 decode.
        float delta = 0.0f;
        if (vi < d_v)
        {
            float kv = 0.0f;
            for (int split = 0; split < kGdnRecurrentRowSplit; ++split)
            {
                kv += reduce_kv[
                    col_in_block +
                    split * kGdnRecurrentColumnsPerBlock];
            }
            delta = (v_head[vi] - kv) * beta_h;
        }

        // Pass 2 updates VGPR state and writes the optional accepted-row image.
        float partial_out = 0.0f;
        float *snapshot =
            state_snapshots && flat_row < max_snapshot_rows
                ? state_snapshots +
                      static_cast<size_t>(flat_row) *
                          static_cast<size_t>(snapshot_stride_floats) +
                      static_cast<size_t>(h) * D_K * d_v
                : nullptr;
        if (vi < d_v)
        {
#pragma unroll
            for (int local_row = 0;
                 local_row < kRegisterRowsPerSplit;
                 ++local_row)
            {
                const int state_row = row_begin + local_row;
                const float s_new =
                    state_rows[local_row] +
                    k_local[state_row] * delta;
                state_rows[local_row] = s_new;
                if (snapshot)
                    snapshot[state_row * d_v + vi] = s_new;
                partial_out += s_new * q_local[state_row];
            }
#pragma clang loop unroll(disable)
            for (int tail_row = 0;
                 tail_row < kLdsRowsPerSplit;
                 ++tail_row)
            {
                const int state_row =
                    row_begin + kRegisterRowsPerSplit + tail_row;
                const int lds_index =
                    tail_row * kGdnRecurrentThreads + tid;
                const float s_new =
                    state_tail[lds_index] +
                    k_local[state_row] * delta;
                state_tail[lds_index] = s_new;
                if (snapshot)
                    snapshot[state_row * d_v + vi] = s_new;
                partial_out += s_new * q_local[state_row];
            }
        }
        reduce_out[tid] = partial_out;
        __syncthreads();

        if (vi < d_v && split_id == 0)
        {
            float out_vi = 0.0f;
            for (int split = 0; split < kGdnRecurrentRowSplit; ++split)
            {
                out_vi += reduce_out[
                    col_in_block +
                    split * kGdnRecurrentColumnsPerBlock];
            }
            o_head[vi] = out_vi;
        }

        // Shared rows and reduction slots belong to exactly one causal row.
        __syncthreads();
    }

    const bool terminal_state_is_snapshotted =
        state_snapshots != nullptr &&
        max_snapshot_rows >= request_count * request_seq_len;
    if (vi < d_v && !terminal_state_is_snapshotted)
    {
#pragma unroll
        for (int local_row = 0;
             local_row < kRegisterRowsPerSplit;
             ++local_row)
        {
            const int state_row = row_begin + local_row;
            S_updated[state_row * d_v + vi] = state_rows[local_row];
        }
#pragma clang loop unroll(disable)
        for (int tail_row = 0; tail_row < kLdsRowsPerSplit; ++tail_row)
        {
            const int state_row =
                row_begin + kRegisterRowsPerSplit + tail_row;
            S_updated[state_row * d_v + vi] =
                state_tail[tail_row * kGdnRecurrentThreads + tid];
        }
    }
}

/**
 * @brief Launch one fixed HIP recurrence specialization for graph capture.
 *
 * GDN key width is immutable model topology. Dispatching here makes register
 * ownership, resource usage, and launch geometry part of graph construction.
 * Unknown widths fail immediately; there is no eager or global-state alternate
 * route hidden behind the optimized grouped implementation.
 */
bool launch_rocm_gdn_recurrent_step(
    const char *caller,
    const float *q,
    const float *k,
    const float *v,
    const float *alpha,
    const float *beta_raw,
    const float *A_log,
    const float *dt_bias,
    float *output,
    const float *initial_state,
    float *updated_state,
    int request_count,
    int request_seq_len,
    int n_heads,
    int d_k,
    int d_v,
    bool use_qk_l2norm,
    const int *effective_seq_len_ptr,
    int effective_row_idx,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    hipStream_t stream)
{
    const dim3 grid(
        request_count * n_heads,
        (d_v + kGdnRecurrentColumnsPerBlock - 1) /
            kGdnRecurrentColumnsPerBlock);
    const int rows_per_split = d_k / kGdnRecurrentRowSplit;
    const int lds_rows_per_split = std::max(0, rows_per_split - 22);
    const size_t shared_bytes =
        static_cast<size_t>(
            2 * d_k + 4 + 2 * kGdnRecurrentThreads +
            lds_rows_per_split * kGdnRecurrentThreads) *
        sizeof(float);

#define LLAMINAR_LAUNCH_ROCM_GDN_RECURRENT(D_K)                               \
    hipLaunchKernelGGL(                                                       \
        HIP_KERNEL_NAME(rocm_gdn_recurrent_step_kernel<D_K>),                 \
        grid,                                                                 \
        dim3(kGdnRecurrentThreads),                                           \
        shared_bytes,                                                         \
        stream,                                                               \
        q, k, v, alpha, beta_raw, A_log, dt_bias, output,                    \
        initial_state, updated_state,                                        \
        request_count, request_seq_len, n_heads, d_v, use_qk_l2norm,          \
        effective_seq_len_ptr, effective_row_idx, state_snapshots,             \
        snapshot_stride_floats, max_snapshot_rows)

    switch (d_k)
    {
    case 64:
        LLAMINAR_LAUNCH_ROCM_GDN_RECURRENT(64);
        break;
    case 128:
        LLAMINAR_LAUNCH_ROCM_GDN_RECURRENT(128);
        break;
    default:
        fprintf(
            stderr,
            "[%s] unsupported ROCm GDN d_k=%d; capture supports d_k in "
            "{64,128}\n",
            caller,
            d_k);
        return false;
    }

#undef LLAMINAR_LAUNCH_ROCM_GDN_RECURRENT

    const hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr, "[%s] %s\n", caller, hipGetErrorString(err));
        return false;
    }
    return true;
}

// =========================================================================
// Prefill Recurrence Kernel (seq_len>1, register-cached state)
//
// Long-prefill optimizations:
//   1. D_K is specialized before graph capture. Each thread retains exactly
//      D_K/4 state values across every timestep: 16 for D_K=64 and 32 for
//      D_K=128. State is loaded from global once and written back once.
//   2. Single-pass per timestep: kv dot product → delta → state update →
//      output accumulation in ONE loop over the 32 cached rows.
//   3. The four-way row split is identical to grouped/scalar decode.
//   4. The in-loop-normalization D_K=128 specialization places its final five
//      rows in explicit LDS, avoiding private scratch at the two-wave VGPR
//      limit. Precomputed normalization shortens live ranges enough to retain
//      all 32 state values in VGPRs without that LDS tail.
//   5. Bounded prompt buckets evaluate Q/K normalization cooperatively before
//      loading state. Recurrence retains the serial decode sum tree but no
//      longer waits on three extra normalization barriers for each token.
//
// Release ISA metadata is a gate: every specialization must report zero
// private segment and zero VGPR/SGPR spills before it may remain dispatchable.
// =========================================================================

/**
 * @brief Keep one head's fixed-order recurrence resident across a prompt chunk.
 *
 * Independent workgroups partition value columns without changing the four-way
 * key reduction used by decode. Width/block specialization removes dynamic
 * indexing; head count remains a runtime stride so TP shards share this route.
 * The optional normalization cache retains per-row scale factors in LDS; its
 * capture-time size limit keeps larger buckets within the same hardware limit.
 * With cached inputs, each wave owns all four key partitions of its independent
 * value columns. Those waves advance concurrently through the input tile, with
 * no per-token workgroup barrier. Lane exchange preserves the ordered four-part
 * sum; it does not compose or reassociate recurrent transitions across tokens.
 * Caller-owned state, output, snapshots and shared-memory capacity must remain
 * valid until the explicit launch stream completes.
 */
template <int STATIC_D_K,
          int STATIC_D_V = 0,
          int STATIC_BLOCK_SIZE = 0,
          bool CACHE_QK_NORMALIZATION = false,
          GdnPrefillSnapshots SNAPSHOTS = GdnPrefillSnapshots::BoundedRows>
__global__ __launch_bounds__(256, 2)
void rocm_gdn_chunk_forward_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    const float *__restrict__ alpha,
    const float *__restrict__ beta_raw,
    const float *__restrict__ A_log,
    const float *__restrict__ dt_bias,
    float *__restrict__ output,
    const float *initial_state,
    float *updated_state,
    const int *__restrict__ effective_seq_len_ptr,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int request_count, int request_seq_len,
    int n_heads, int d_v,
    bool use_qk_l2norm)
{
    static_assert(
        STATIC_D_K == 64 || STATIC_D_K == 128,
        "ROCm long-prefill GDN requires a capture-time D_K specialization");
    static_assert(
        !CACHE_QK_NORMALIZATION ||
            (STATIC_D_K == 128 && STATIC_BLOCK_SIZE == 256),
        "Wave-owned prefill retains the 128-key, four-part serial decode tree");
    const int kernel_n_heads = n_heads;
    constexpr int kernel_d_k = STATIC_D_K;
    const int kernel_d_v = STATIC_D_V > 0 ? STATIC_D_V : d_v;
    const int kernel_block_size =
        STATIC_BLOCK_SIZE > 0 ? STATIC_BLOCK_SIZE : blockDim.x;

    const int request = blockIdx.z;
    const int h = blockIdx.x;
    if (request >= request_count || h >= kernel_n_heads) return;

    const int tid = threadIdx.x;

    constexpr int ROW_SPLIT = 4;
    constexpr int columns_per_wave = 64 / ROW_SPLIT;
    const int cols_per_block = kernel_block_size / ROW_SPLIT;
    const int column_in_block = CACHE_QK_NORMALIZATION
        ? (tid / 64) * columns_per_wave + tid % columns_per_wave
        : tid % cols_per_block;
    const int key_partition = CACHE_QK_NORMALIZATION
        ? (tid % 64) / columns_per_wave : tid / cols_per_block;
    const int vi =
        blockIdx.y * cols_per_block + column_in_block;

    const int qk_stride = kernel_n_heads * kernel_d_k;
    const int v_stride = kernel_n_heads * kernel_d_v;
    const float scale = rsqrtf(static_cast<float>(kernel_d_k));
    const int request_row_base = request * request_seq_len;
    const size_t request_state_stride =
        static_cast<size_t>(kernel_n_heads) * kernel_d_k * kernel_d_v;
    const float *S_initial =
        initial_state +
        static_cast<size_t>(request) * request_state_stride +
        static_cast<size_t>(h) * kernel_d_k * kernel_d_v;
    float *S_updated =
        updated_state +
        static_cast<size_t>(request) * request_state_stride +
        static_cast<size_t>(h) * kernel_d_k * kernel_d_v;

    extern __shared__ float smem[];
    constexpr int input_tile_rows =
        CACHE_QK_NORMALIZATION ? kGdnPrefillInputTileRows : 1;
    float *q_tile = smem;
    float *k_tile = q_tile + input_tile_rows * kernel_d_k;
    float *v_tile = k_tile + input_tile_rows * kernel_d_k;
    float *warp_sums = v_tile +
        (CACHE_QK_NORMALIZATION ? input_tile_rows * cols_per_block : 0);
    const int shared_reduction_size = CACHE_QK_NORMALIZATION ? 0 : kernel_block_size;
    float *reduce_kv = warp_sums + 16;
    float *reduce_out = reduce_kv + shared_reduction_size;
    constexpr int kExplicitLdsStateTailRows =
        STATIC_D_K == 128 && STATIC_BLOCK_SIZE == 256 &&
                !CACHE_QK_NORMALIZATION
            ? 5 : 0;
    constexpr bool kUsesExplicitLdsStateTail =
        kExplicitLdsStateTailRows > 0;
    float *state_tail = reduce_out + shared_reduction_size;
    float *gate_decay = state_tail +
        kExplicitLdsStateTailRows * kernel_block_size;
    float *gate_beta = gate_decay + request_seq_len;

    int num_warps = (kernel_block_size + 63) / 64;

    // Pre-load A_log and dt_bias (constant across timesteps)
    __shared__ float A_log_h;
    __shared__ float dt_bias_h;
    if (tid == 0)
    {
        A_log_h = A_log[h];
        dt_bias_h = dt_bias[h];
    }
    __syncthreads();

    int effective_seq_len = request_seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = effective_seq_len_ptr[request];
        effective_seq_len = raw_effective < 0
                                ? 0
                                : (raw_effective > request_seq_len
                                       ? request_seq_len
                                       : raw_effective);
    }

    /*
     * Gate evaluation has no recurrent dependency. The former implementation
     * evaluated softplus, decay, and sigmoid on lane zero inside every token
     * iteration, making 255 peers wait for one long transcendental chain. Each
     * workgroup now evaluates its complete token range cooperatively once and
     * retains the exact FP32 results in LDS. The recurrent loop below consumes
     * the same per-token values without changing state arithmetic or ordering.
     */
    for (int t = tid; t < effective_seq_len; t += kernel_block_size)
    {
        const int row = request_row_base + t;
        const float x =
            alpha[row * kernel_n_heads + h] + dt_bias_h;
        const float softplus =
            x > 20.0f ? x : log1pf(expf(x));
        gate_decay[t] = expf(A_log_h * softplus);
        gate_beta[t] =
            1.0f /
            (1.0f + expf(-beta_raw[row * kernel_n_heads + h]));
    }
    if constexpr (CACHE_QK_NORMALIZATION)
    {
        if (use_qk_l2norm)
        {
#pragma clang fp contract(off)
            /* Normalization has no dependency on recurrent state. One wave
             * prepares each row before that state occupies the register file.
             * Keep the two 64-key reductions separate: adding keys 0 and 64
             * in each lane first would change the serial-decode sum tree.
             * The final two zero waves are retained explicitly as well. */
            const int lane = tid % 64;
            const int wave = tid / 64;
            for (int t = wave; t < effective_seq_len; t += kernel_block_size / 64)
            {
                const int row = request_row_base + t;
                const int base = row * qk_stride + h * kernel_d_k;
                const float q0 = Q[base + lane];
                const float q1 = Q[base + lane + 64];
                const float k0 = K[base + lane];
                const float k1 = K[base + lane + 64];
                float q_sum0 = q0 * q0;
                float q_sum1 = q1 * q1;
                float k_sum0 = k0 * k0;
                float k_sum1 = k1 * k1;
                for (int offset = 32; offset > 0; offset >>= 1)
                {
                    // The square is rounded before the first shuffle add.
                    // HIP's round-to-nearest helpers alone do not prevent LLVM
                    // contracting that first add into an FMA: the exchanged
                    // lane is rounded but the local lane would not be. The
                    // lexical contract(off) applies to these direct expressions,
                    // not helper bodies compiled outside this scope. It keeps
                    // both boundaries without changing recurrence contraction.
                    q_sum0 += __shfl_xor(q_sum0, offset);
                    q_sum1 += __shfl_xor(q_sum1, offset);
                    k_sum0 += __shfl_xor(k_sum0, offset);
                    k_sum1 += __shfl_xor(k_sum1, offset);
                }
                if (lane == 0)
                {
                    float q_total = 0.0f;
                    float k_total = 0.0f;
                    q_total += q_sum0;
                    q_total += q_sum1;
                    q_total += 0.0f;
                    q_total += 0.0f;
                    k_total += k_sum0;
                    k_total += k_sum1;
                    k_total += 0.0f;
                    k_total += 0.0f;
                    gate_beta[request_seq_len + t] =
                        scale / fmaxf(sqrtf(q_total), 1e-6f);
                    gate_beta[2 * request_seq_len + t] =
                        1.0f / fmaxf(sqrtf(k_total), 1e-6f);
                }
            }
        }
    }
    __syncthreads();

    // =====================================================================
    // REGISTER-CACHED STATE: Load state into registers ONCE
    // Each thread owns rows_per_split (=d_k/ROW_SPLIT) state values for
    // its column vi. With d_k=128, ROW_SPLIT=4: 32 floats per thread.
    // =====================================================================
    constexpr int rows_per_split = kernel_d_k / ROW_SPLIT;
    // State addressing is an affine, nonnegative element offset, not signed
    // 32-bit arithmetic. In the signed form LLVM must preserve a separate
    // sign-extended offset for every snapshot row and keeps those 32 pointer
    // offsets live through the entire recurrence, even with no snapshots.
    // Widen the common base once so each row can use a constant displacement.
    // This changes address calculation only; the four key partitions and all
    // floating-point operations below retain their serial-decode order.
    const size_t j_start =
        static_cast<size_t>(key_partition) * rows_per_split;
    // Put the four key partitions in neighbouring LDS banks. A wave now owns
    // all four partitions; retaining key-major order would issue four distinct
    // addresses to the same bank for every input load.
    constexpr int input_key_step = CACHE_QK_NORMALIZATION ? ROW_SPLIT : 1;
    const size_t input_key_begin = CACHE_QK_NORMALIZATION ? key_partition : j_start;

    // Use individual variables in groups of 8 to help the compiler
    // keep them in VGPRs (AMD compiler more likely to register-allocate
    // fixed-size unrolled groups than a single large array).
    float sc0, sc1, sc2, sc3, sc4, sc5, sc6, sc7;
    float sc8, sc9, sc10, sc11, sc12, sc13, sc14, sc15;
    float sc16, sc17, sc18, sc19, sc20, sc21, sc22, sc23;
    float sc24, sc25, sc26, sc27, sc28, sc29, sc30, sc31;

    if (vi < kernel_d_v) {
        sc0  = S_initial[(j_start +  0) * kernel_d_v + vi];
        sc1  = S_initial[(j_start +  1) * kernel_d_v + vi];
        sc2  = S_initial[(j_start +  2) * kernel_d_v + vi];
        sc3  = S_initial[(j_start +  3) * kernel_d_v + vi];
        sc4  = S_initial[(j_start +  4) * kernel_d_v + vi];
        sc5  = S_initial[(j_start +  5) * kernel_d_v + vi];
        sc6  = S_initial[(j_start +  6) * kernel_d_v + vi];
        sc7  = S_initial[(j_start +  7) * kernel_d_v + vi];
        sc8  = S_initial[(j_start +  8) * kernel_d_v + vi];
        sc9  = S_initial[(j_start +  9) * kernel_d_v + vi];
        sc10 = S_initial[(j_start + 10) * kernel_d_v + vi];
        sc11 = S_initial[(j_start + 11) * kernel_d_v + vi];
        sc12 = S_initial[(j_start + 12) * kernel_d_v + vi];
        sc13 = S_initial[(j_start + 13) * kernel_d_v + vi];
        sc14 = S_initial[(j_start + 14) * kernel_d_v + vi];
        sc15 = S_initial[(j_start + 15) * kernel_d_v + vi];
        if constexpr (STATIC_D_K == 128)
        {
            sc16 = S_initial[(j_start + 16) * kernel_d_v + vi];
            sc17 = S_initial[(j_start + 17) * kernel_d_v + vi];
            sc18 = S_initial[(j_start + 18) * kernel_d_v + vi];
            sc19 = S_initial[(j_start + 19) * kernel_d_v + vi];
            sc20 = S_initial[(j_start + 20) * kernel_d_v + vi];
            sc21 = S_initial[(j_start + 21) * kernel_d_v + vi];
            sc22 = S_initial[(j_start + 22) * kernel_d_v + vi];
            sc23 = S_initial[(j_start + 23) * kernel_d_v + vi];
            sc24 = S_initial[(j_start + 24) * kernel_d_v + vi];
            sc25 = S_initial[(j_start + 25) * kernel_d_v + vi];
            sc26 = S_initial[(j_start + 26) * kernel_d_v + vi];
            if constexpr (kUsesExplicitLdsStateTail)
            {
                state_tail[0 * kernel_block_size + tid] =
                    S_initial[(j_start + 27) * kernel_d_v + vi];
                state_tail[1 * kernel_block_size + tid] =
                    S_initial[(j_start + 28) * kernel_d_v + vi];
                state_tail[2 * kernel_block_size + tid] =
                    S_initial[(j_start + 29) * kernel_d_v + vi];
                state_tail[3 * kernel_block_size + tid] =
                    S_initial[(j_start + 30) * kernel_d_v + vi];
                state_tail[4 * kernel_block_size + tid] =
                    S_initial[(j_start + 31) * kernel_d_v + vi];
            }
            else
            {
                sc27 = S_initial[(j_start + 27) * kernel_d_v + vi];
                sc28 = S_initial[(j_start + 28) * kernel_d_v + vi];
                sc29 = S_initial[(j_start + 29) * kernel_d_v + vi];
                sc30 = S_initial[(j_start + 30) * kernel_d_v + vi];
                sc31 = S_initial[(j_start + 31) * kernel_d_v + vi];
            }
        }
    } else {
        sc0=sc1=sc2=sc3=sc4=sc5=sc6=sc7=0.0f;
        sc8=sc9=sc10=sc11=sc12=sc13=sc14=sc15=0.0f;
        if constexpr (STATIC_D_K == 128)
        {
            sc16=sc17=sc18=sc19=sc20=sc21=sc22=sc23=0.0f;
            sc24=sc25=sc26=0.0f;
            if constexpr (kUsesExplicitLdsStateTail)
            {
                state_tail[0 * kernel_block_size + tid] = 0.0f;
                state_tail[1 * kernel_block_size + tid] = 0.0f;
                state_tail[2 * kernel_block_size + tid] = 0.0f;
                state_tail[3 * kernel_block_size + tid] = 0.0f;
                state_tail[4 * kernel_block_size + tid] = 0.0f;
            }
            else
            {
                sc27=sc28=sc29=sc30=sc31=0.0f;
            }
        }
    }

    for (int t = 0; t < request_seq_len; t++) {
        const int row = request_row_base + t;
        const float *q_src = Q + row * qk_stride + h * kernel_d_k;
        const float *k_src = K + row * qk_stride + h * kernel_d_k;
        const float *v_src = V + row * v_stride + h * kernel_d_v;
        float *o_dst = output + row * v_stride + h * kernel_d_v;

        if (t >= effective_seq_len) {
            if (vi < kernel_d_v && key_partition == 0)
                o_dst[vi] = 0.0f;
            continue;
        }

        float *q_local = q_tile + (t % input_tile_rows) * kernel_d_k;
        float *k_local = k_tile + (t % input_tile_rows) * kernel_d_k;
        if constexpr (CACHE_QK_NORMALIZATION)
        {
            // Q/K/V are independent input rows, unlike the recurrent state.
            // Prepare a bounded tile once so global input latency and its
            // publication barrier are paid once per tile, not once per row.
            // Waves advance independently inside the tile. Join before reuse
            // so no fast wave overwrites inputs still consumed by a slow wave.
            if (t % input_tile_rows == 0)
            {
                __syncthreads();
                for (int cell = tid; cell < input_tile_rows * kernel_d_k;
                     cell += kernel_block_size)
                {
                    const int tile_row = t + cell / kernel_d_k;
                    if (tile_row < effective_seq_len)
                    {
                        const int key = cell % kernel_d_k;
                        const int prepared_index =
                            (cell / kernel_d_k) * kernel_d_k +
                            (key % rows_per_split) * ROW_SPLIT + key / rows_per_split;
                        const int base = (request_row_base + tile_row) * qk_stride +
                            h * kernel_d_k + key;
                        q_tile[prepared_index] = Q[base] * (use_qk_l2norm
                            ? gate_beta[request_seq_len + tile_row] : scale);
                        k_tile[prepared_index] = use_qk_l2norm
                            ? K[base] * gate_beta[2 * request_seq_len + tile_row]
                            : K[base];
                    }
                }
                for (int cell = tid; cell < input_tile_rows * cols_per_block;
                     cell += kernel_block_size)
                {
                    const int tile_row = t + cell / cols_per_block;
                    const int col = blockIdx.y * cols_per_block + cell % cols_per_block;
                    if (tile_row < effective_seq_len && col < kernel_d_v)
                        v_tile[cell] = V[(request_row_base + tile_row) * v_stride +
                            h * kernel_d_v + col];
                }
                __syncthreads();
            }
        }
        else
        {
            for (int i = tid; i < kernel_d_k; i += kernel_block_size)
            {
                q_local[i] = q_src[i];
                k_local[i] = k_src[i];
            }
            __syncthreads();
        }

        if constexpr (!CACHE_QK_NORMALIZATION)
        {
            if (use_qk_l2norm) {
                /*
                 * Q and K normalization are independent, but both reductions use
                 * the same fixed wavefront and warp ordering.  Keep separate
                 * accumulators and shared-memory lanes so each result remains
                 * byte-identical to the former serial Q-then-K implementation,
                 * while publishing both reductions through one barrier sequence.
                 */
                float q_sum = 0.0f;
                float k_sum = 0.0f;
                for (int i = tid; i < kernel_d_k; i += kernel_block_size)
                {
                    q_sum += q_local[i] * q_local[i];
                    k_sum += k_local[i] * k_local[i];
                }
                for (int offset = 32; offset > 0; offset >>= 1)
                {
                    q_sum += __shfl_xor(q_sum, offset);
                    k_sum += __shfl_xor(k_sum, offset);
                }
                if (tid % 64 == 0)
                {
                    warp_sums[tid / 64] = q_sum;
                    warp_sums[8 + tid / 64] = k_sum;
                }
                __syncthreads();
                if (tid == 0)
                {
                    float q_total = 0.0f;
                    float k_total = 0.0f;
                    for (int w = 0; w < num_warps; ++w)
                    {
                        q_total += warp_sums[w];
                        k_total += warp_sums[8 + w];
                    }
                    warp_sums[0] = q_total;
                    warp_sums[8] = k_total;
                }
                __syncthreads();

                const float q_inv =
                    scale / fmaxf(sqrtf(warp_sums[0]), 1e-6f);
                const float k_inv =
                    1.0f / fmaxf(sqrtf(warp_sums[8]), 1e-6f);
                for (int i = tid; i < kernel_d_k; i += kernel_block_size)
                {
                    q_local[i] *= q_inv;
                    k_local[i] *= k_inv;
                }
                __syncthreads();
            } else {
                for (int i = tid; i < kernel_d_k; i += kernel_block_size)
                    q_local[i] *= scale;
                __syncthreads();
            }
        }

        const float decay = gate_decay[t];
        const float beta_h = gate_beta[t];

        // K is immutable for both recurrence passes. Retain this lane's key
        // partition across the delta-publication barrier instead of issuing a
        // second LDS read for every state update. The affine state addresses
        // above leave room for these values without private-memory spills.
        float k_registers[rows_per_split];
#pragma unroll
        for (int key = 0; key < rows_per_split; ++key)
            k_registers[key] = k_local[input_key_begin + input_key_step * key];

        // =================================================================
        // DECODE-EQUIVALENT register-cached state processing
        // Pass 1: round decay once into the resident state, then observe it.
        //
        // The canonical recurrence is `S = S * decay`, `kv = S^T k`, then
        // `S = S + k outer delta`. ROCm M=1/grouped decode, CUDA, and CPU use
        // this same rounded intermediate. Reusing it removes one dependency-
        // chained multiply per state cell without changing the GDN equation.
        // =================================================================
        float partial_kv = 0.0f;
        if (vi < kernel_d_v) {
#define LLAMINAR_ROCM_GDN_DECAY_OBSERVE(OFFSET, MEMBER)                       \
            MEMBER *= decay;                                                  \
            partial_kv += MEMBER * k_registers[OFFSET]
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(0, sc0);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(1, sc1);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(2, sc2);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(3, sc3);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(4, sc4);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(5, sc5);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(6, sc6);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(7, sc7);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(8, sc8);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(9, sc9);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(10, sc10);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(11, sc11);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(12, sc12);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(13, sc13);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(14, sc14);
            LLAMINAR_ROCM_GDN_DECAY_OBSERVE(15, sc15);
            if constexpr (STATIC_D_K == 128)
            {
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(16, sc16);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(17, sc17);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(18, sc18);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(19, sc19);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(20, sc20);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(21, sc21);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(22, sc22);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(23, sc23);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(24, sc24);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(25, sc25);
                LLAMINAR_ROCM_GDN_DECAY_OBSERVE(26, sc26);
                if constexpr (kUsesExplicitLdsStateTail)
                {
                    float tail_state =
                        state_tail[0 * kernel_block_size + tid] * decay;
                    state_tail[0 * kernel_block_size + tid] = tail_state;
                    partial_kv += tail_state * k_registers[27];

                    tail_state =
                        state_tail[1 * kernel_block_size + tid] * decay;
                    state_tail[1 * kernel_block_size + tid] = tail_state;
                    partial_kv += tail_state * k_registers[28];

                    tail_state =
                        state_tail[2 * kernel_block_size + tid] * decay;
                    state_tail[2 * kernel_block_size + tid] = tail_state;
                    partial_kv += tail_state * k_registers[29];

                    tail_state =
                        state_tail[3 * kernel_block_size + tid] * decay;
                    state_tail[3 * kernel_block_size + tid] = tail_state;
                    partial_kv += tail_state * k_registers[30];

                    tail_state =
                        state_tail[4 * kernel_block_size + tid] * decay;
                    state_tail[4 * kernel_block_size + tid] = tail_state;
                    partial_kv += tail_state * k_registers[31];
                }
                else
                {
                    sc27 *= decay; partial_kv += sc27 * k_registers[27];
                    sc28 *= decay; partial_kv += sc28 * k_registers[28];
                    sc29 *= decay; partial_kv += sc29 * k_registers[29];
                    sc30 *= decay; partial_kv += sc30 * k_registers[30];
                    sc31 *= decay; partial_kv += sc31 * k_registers[31];
                }
            }
#undef LLAMINAR_ROCM_GDN_DECAY_OBSERVE
        }
        if constexpr (!CACHE_QK_NORMALIZATION)
        {
            reduce_kv[tid] = partial_kv;
            __syncthreads();
        }

        // Reduce kv across splits and compute delta
        float delta = 0.0f;
        if (vi < kernel_d_v) {
            int reduction_col = 0;
            if constexpr (CACHE_QK_NORMALIZATION)
            {
                reduction_col = column_in_block;
            }
            else if constexpr (STATIC_BLOCK_SIZE == 256)
            {
                /*
                 * Keep this invariant address out of the 425-row live range.
                 * Clang otherwise hoists `tid % 64` before the loop, exhausts
                 * the 128-VGPR two-wave budget, and spills the address through
                 * private memory. The volatile instruction creates the same
                 * integer column locally and lets it die after this reduction.
                 */
                asm volatile(
                    "v_and_b32_e32 %0, 63, %1"
                    : "=v"(reduction_col)
                    : "v"(tid));
            }
            else
            {
                reduction_col = tid % cols_per_block;
            }
            float kv = 0.0f;
            for (int s = 0; s < ROW_SPLIT; s++)
            {
                if constexpr (CACHE_QK_NORMALIZATION)
                    kv += __shfl(partial_kv,
                        tid % columns_per_wave + s * columns_per_wave, 64);
                else
                    kv += reduce_kv[reduction_col + s * cols_per_block];
            }
            const float value = CACHE_QK_NORMALIZATION
                ? v_tile[(t % input_tile_rows) * cols_per_block + reduction_col]
                : v_src[vi];
            delta = (value - kv) * beta_h;
        }

        // =================================================================
        // Pass 2: Update state + accumulate output (all in registers)
        // =================================================================
        // Stage Q independently of the dependent output FMA chain. K and Q
        // lifetimes overlap here deliberately; compiler spill certification
        // proves this remains a register-only operation for the selected ISA.
        float q_registers[rows_per_split];
        if constexpr (CACHE_QK_NORMALIZATION)
        {
#pragma unroll
            for (int key = 0; key < rows_per_split; ++key)
                q_registers[key] = q_local[input_key_begin + input_key_step * key];
        }
        float partial_out = 0.0f;
        if (vi < kernel_d_v) {
#define LLAMINAR_ROCM_GDN_UPDATE_PROJECT(OFFSET, MEMBER)                      \
            MEMBER += k_registers[OFFSET] * delta;                            \
            partial_out += MEMBER * (CACHE_QK_NORMALIZATION ? q_registers[OFFSET] \
                : q_local[input_key_begin + input_key_step * OFFSET])
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(0, sc0);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(1, sc1);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(2, sc2);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(3, sc3);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(4, sc4);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(5, sc5);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(6, sc6);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(7, sc7);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(8, sc8);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(9, sc9);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(10, sc10);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(11, sc11);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(12, sc12);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(13, sc13);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(14, sc14);
            LLAMINAR_ROCM_GDN_UPDATE_PROJECT(15, sc15);
            if constexpr (STATIC_D_K == 128)
            {
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(16, sc16);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(17, sc17);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(18, sc18);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(19, sc19);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(20, sc20);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(21, sc21);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(22, sc22);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(23, sc23);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(24, sc24);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(25, sc25);
                LLAMINAR_ROCM_GDN_UPDATE_PROJECT(26, sc26);
                if constexpr (kUsesExplicitLdsStateTail)
                {
                    float tail_state =
                        state_tail[0 * kernel_block_size + tid] +
                        k_registers[27] * delta;
                    state_tail[0 * kernel_block_size + tid] = tail_state;
                    partial_out += tail_state * q_local[j_start + 27];

                    tail_state =
                        state_tail[1 * kernel_block_size + tid] +
                        k_registers[28] * delta;
                    state_tail[1 * kernel_block_size + tid] = tail_state;
                    partial_out += tail_state * q_local[j_start + 28];

                    tail_state =
                        state_tail[2 * kernel_block_size + tid] +
                        k_registers[29] * delta;
                    state_tail[2 * kernel_block_size + tid] = tail_state;
                    partial_out += tail_state * q_local[j_start + 29];

                    tail_state =
                        state_tail[3 * kernel_block_size + tid] +
                        k_registers[30] * delta;
                    state_tail[3 * kernel_block_size + tid] = tail_state;
                    partial_out += tail_state * q_local[j_start + 30];

                    tail_state =
                        state_tail[4 * kernel_block_size + tid] +
                        k_registers[31] * delta;
                    state_tail[4 * kernel_block_size + tid] = tail_state;
                    partial_out += tail_state * q_local[j_start + 31];
                }
                else
                {
                    LLAMINAR_ROCM_GDN_UPDATE_PROJECT(27, sc27);
                    LLAMINAR_ROCM_GDN_UPDATE_PROJECT(28, sc28);
                    LLAMINAR_ROCM_GDN_UPDATE_PROJECT(29, sc29);
                    LLAMINAR_ROCM_GDN_UPDATE_PROJECT(30, sc30);
                    LLAMINAR_ROCM_GDN_UPDATE_PROJECT(31, sc31);
                }
            }
#undef LLAMINAR_ROCM_GDN_UPDATE_PROJECT
        }
        if (SNAPSHOTS == GdnPrefillSnapshots::BoundedRows &&
            state_snapshots && row < max_snapshot_rows && vi < kernel_d_v) {
            float *snapshot =
                state_snapshots +
                (size_t)row * snapshot_stride_floats +
                (size_t)h * kernel_d_k * kernel_d_v;
            snapshot[(j_start +  0) * kernel_d_v + vi] = sc0;
            snapshot[(j_start +  1) * kernel_d_v + vi] = sc1;
            snapshot[(j_start +  2) * kernel_d_v + vi] = sc2;
            snapshot[(j_start +  3) * kernel_d_v + vi] = sc3;
            snapshot[(j_start +  4) * kernel_d_v + vi] = sc4;
            snapshot[(j_start +  5) * kernel_d_v + vi] = sc5;
            snapshot[(j_start +  6) * kernel_d_v + vi] = sc6;
            snapshot[(j_start +  7) * kernel_d_v + vi] = sc7;
            snapshot[(j_start +  8) * kernel_d_v + vi] = sc8;
            snapshot[(j_start +  9) * kernel_d_v + vi] = sc9;
            snapshot[(j_start + 10) * kernel_d_v + vi] = sc10;
            snapshot[(j_start + 11) * kernel_d_v + vi] = sc11;
            snapshot[(j_start + 12) * kernel_d_v + vi] = sc12;
            snapshot[(j_start + 13) * kernel_d_v + vi] = sc13;
            snapshot[(j_start + 14) * kernel_d_v + vi] = sc14;
            snapshot[(j_start + 15) * kernel_d_v + vi] = sc15;
            if constexpr (STATIC_D_K == 128)
            {
                snapshot[(j_start + 16) * kernel_d_v + vi] = sc16;
                snapshot[(j_start + 17) * kernel_d_v + vi] = sc17;
                snapshot[(j_start + 18) * kernel_d_v + vi] = sc18;
                snapshot[(j_start + 19) * kernel_d_v + vi] = sc19;
                snapshot[(j_start + 20) * kernel_d_v + vi] = sc20;
                snapshot[(j_start + 21) * kernel_d_v + vi] = sc21;
                snapshot[(j_start + 22) * kernel_d_v + vi] = sc22;
                snapshot[(j_start + 23) * kernel_d_v + vi] = sc23;
                snapshot[(j_start + 24) * kernel_d_v + vi] = sc24;
                snapshot[(j_start + 25) * kernel_d_v + vi] = sc25;
                snapshot[(j_start + 26) * kernel_d_v + vi] = sc26;
                if constexpr (kUsesExplicitLdsStateTail)
                {
                    snapshot[(j_start + 27) * kernel_d_v + vi] =
                        state_tail[0 * kernel_block_size + tid];
                    snapshot[(j_start + 28) * kernel_d_v + vi] =
                        state_tail[1 * kernel_block_size + tid];
                    snapshot[(j_start + 29) * kernel_d_v + vi] =
                        state_tail[2 * kernel_block_size + tid];
                    snapshot[(j_start + 30) * kernel_d_v + vi] =
                        state_tail[3 * kernel_block_size + tid];
                    snapshot[(j_start + 31) * kernel_d_v + vi] =
                        state_tail[4 * kernel_block_size + tid];
                }
                else
                {
                    snapshot[(j_start + 27) * kernel_d_v + vi] = sc27;
                    snapshot[(j_start + 28) * kernel_d_v + vi] = sc28;
                    snapshot[(j_start + 29) * kernel_d_v + vi] = sc29;
                    snapshot[(j_start + 30) * kernel_d_v + vi] = sc30;
                    snapshot[(j_start + 31) * kernel_d_v + vi] = sc31;
                }
            }
        }
        if constexpr (!CACHE_QK_NORMALIZATION)
        {
            reduce_out[tid] = partial_out;
            __syncthreads();
        }

        // Every source lane must participate in a wave shuffle. Exchanging
        // values inside the writer-only branch would read inactive partitions.
        float wave_output = 0.0f;
        if constexpr (CACHE_QK_NORMALIZATION)
        {
            for (int s = 0; s < ROW_SPLIT; ++s)
                wave_output += __shfl(partial_out,
                    tid % columns_per_wave + s * columns_per_wave, 64);
        }
        // Reduce output and write (only split_id=0 writes).
        if (vi < kernel_d_v && key_partition == 0) {
            int reduction_col = 0;
            if constexpr (!CACHE_QK_NORMALIZATION && STATIC_BLOCK_SIZE == 256)
            {
                asm volatile(
                    "v_and_b32_e32 %0, 63, %1"
                    : "=v"(reduction_col)
                    : "v"(tid));
            }
            else
            {
                reduction_col = tid % cols_per_block;
            }
            float out_vi = wave_output;
            if constexpr (!CACHE_QK_NORMALIZATION)
            {
                for (int s = 0; s < ROW_SPLIT; s++)
                    out_vi += reduce_out[reduction_col + s * cols_per_block];
            }
            o_dst[vi] = out_vi;
        }
    }

    // =====================================================================
    // Write state back to global memory ONCE at the end
    // =====================================================================
    if (vi < kernel_d_v) {
        S_updated[(j_start +  0) * kernel_d_v + vi] = sc0;
        S_updated[(j_start +  1) * kernel_d_v + vi] = sc1;
        S_updated[(j_start +  2) * kernel_d_v + vi] = sc2;
        S_updated[(j_start +  3) * kernel_d_v + vi] = sc3;
        S_updated[(j_start +  4) * kernel_d_v + vi] = sc4;
        S_updated[(j_start +  5) * kernel_d_v + vi] = sc5;
        S_updated[(j_start +  6) * kernel_d_v + vi] = sc6;
        S_updated[(j_start +  7) * kernel_d_v + vi] = sc7;
        S_updated[(j_start +  8) * kernel_d_v + vi] = sc8;
        S_updated[(j_start +  9) * kernel_d_v + vi] = sc9;
        S_updated[(j_start + 10) * kernel_d_v + vi] = sc10;
        S_updated[(j_start + 11) * kernel_d_v + vi] = sc11;
        S_updated[(j_start + 12) * kernel_d_v + vi] = sc12;
        S_updated[(j_start + 13) * kernel_d_v + vi] = sc13;
        S_updated[(j_start + 14) * kernel_d_v + vi] = sc14;
        S_updated[(j_start + 15) * kernel_d_v + vi] = sc15;
        if constexpr (STATIC_D_K == 128)
        {
            S_updated[(j_start + 16) * kernel_d_v + vi] = sc16;
            S_updated[(j_start + 17) * kernel_d_v + vi] = sc17;
            S_updated[(j_start + 18) * kernel_d_v + vi] = sc18;
            S_updated[(j_start + 19) * kernel_d_v + vi] = sc19;
            S_updated[(j_start + 20) * kernel_d_v + vi] = sc20;
            S_updated[(j_start + 21) * kernel_d_v + vi] = sc21;
            S_updated[(j_start + 22) * kernel_d_v + vi] = sc22;
            S_updated[(j_start + 23) * kernel_d_v + vi] = sc23;
            S_updated[(j_start + 24) * kernel_d_v + vi] = sc24;
            S_updated[(j_start + 25) * kernel_d_v + vi] = sc25;
            S_updated[(j_start + 26) * kernel_d_v + vi] = sc26;
            if constexpr (kUsesExplicitLdsStateTail)
            {
                S_updated[(j_start + 27) * kernel_d_v + vi] =
                    state_tail[0 * kernel_block_size + tid];
                S_updated[(j_start + 28) * kernel_d_v + vi] =
                    state_tail[1 * kernel_block_size + tid];
                S_updated[(j_start + 29) * kernel_d_v + vi] =
                    state_tail[2 * kernel_block_size + tid];
                S_updated[(j_start + 30) * kernel_d_v + vi] =
                    state_tail[3 * kernel_block_size + tid];
                S_updated[(j_start + 31) * kernel_d_v + vi] =
                    state_tail[4 * kernel_block_size + tid];
            }
            else
            {
                S_updated[(j_start + 27) * kernel_d_v + vi] = sc27;
                S_updated[(j_start + 28) * kernel_d_v + vi] = sc28;
                S_updated[(j_start + 29) * kernel_d_v + vi] = sc29;
                S_updated[(j_start + 30) * kernel_d_v + vi] = sc30;
                S_updated[(j_start + 31) * kernel_d_v + vi] = sc31;
            }
        }
    }
}

// =========================================================================
// Short Conv1d Kernels
// =========================================================================

__global__ void rocm_short_conv1d_decode_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    const float *initial_conv_state,
    float *updated_conv_state,
    int channels, int kernel_size,
    bool apply_silu)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels) return;

    const int ks_minus1 = kernel_size - 1;

    // Compute convolution FIRST using OLD state, then shift
    // (must compute before shift to avoid overwriting state values)
    float sum = 0.0f;
    for (int k = 0; k < ks_minus1; k++)
        sum += initial_conv_state[ch * ks_minus1 + k] * weight[ch * kernel_size + k];
    sum += input[ch] * weight[ch * kernel_size + ks_minus1];

    if (bias) sum += bias[ch];
    if (apply_silu)
        sum = sum / (1.0f + expf(-sum));
    output[ch] = sum;

    // Now shift conv_state left by 1, insert new input at the end
    for (int k = 0; k < ks_minus1 - 1; k++)
        updated_conv_state[ch * ks_minus1 + k] =
            initial_conv_state[ch * ks_minus1 + k + 1];
    updated_conv_state[ch * ks_minus1 + ks_minus1 - 1] = input[ch];
}

__global__ void rocm_short_conv1d_prefill_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    const float *__restrict__ initial_conv_state,
    const int *__restrict__ effective_seq_len_ptr,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int seq_len, int channels, int kernel_size,
    bool apply_silu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int ks_minus1 = kernel_size - 1;
    int total = seq_len * channels;
    if (idx >= total) return;

    int effective_seq_len = seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = *effective_seq_len_ptr;
        effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
    }

    int t = idx / channels;
    int ch = idx % channels;

    float sum = 0.0f;
    if (t < effective_seq_len) {
        for (int k = 0; k < kernel_size; k++) {
            int src_t = t - ks_minus1 + k;
            float val = 0.0f;
            if (src_t >= 0)
                val = input[src_t * channels + ch];
            else if (initial_conv_state)
                val = initial_conv_state[ch * ks_minus1 + ks_minus1 + src_t];
            sum += val * weight[ch * kernel_size + k];
        }

        if (bias) sum += bias[ch];
        if (apply_silu)
            sum = sum / (1.0f + expf(-sum));
    }
    output[t * channels + ch] = sum;

    if (state_snapshots && t < effective_seq_len && t < max_snapshot_rows) {
        float *snapshot =
            state_snapshots +
            (size_t)t * snapshot_stride_floats +
            (size_t)ch * ks_minus1;
        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx) {
            int src_t = t - ks_minus1 + 1 + state_idx;
            snapshot[state_idx] =
                (src_t >= 0) ? input[src_t * channels + ch]
                             : (initial_conv_state ? initial_conv_state[ch * ks_minus1 + ks_minus1 + src_t] : 0.0f);
        }
    }
}

/**
 * @brief Decode-equivalent grouped short-conv kernel for MTP verifier rows.
 *
 * The ordinary prefill route needs two launches so the final live state cannot
 * race with threads that still read the old state. A verifier already needs
 * every post-row history, so this kernel assigns one lane to each channel,
 * walks the runtime-sized matrix in causal order, publishes each snapshot, and
 * commits speculative state once all reads from the old channel history finish.
 * There is one launch for the complete matrix and no row replay.
 */
__global__ void rocm_short_conv1d_small_m_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    const float *initial_conv_state,
    float *updated_conv_state,
    const int *__restrict__ effective_seq_len_ptr,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int seq_len, int channels, int kernel_size,
    bool apply_silu)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels)
        return;

    const int ks_minus1 = kernel_size - 1;
    if (ks_minus1 <= 0)
        return;

    int effective_seq_len = seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = *effective_seq_len_ptr;
        effective_seq_len =
            raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
    }

    const float *channel_weight = weight + static_cast<size_t>(ch) * kernel_size;
    const float *channel_initial_state =
        initial_conv_state
            ? initial_conv_state + static_cast<size_t>(ch) * ks_minus1
            : nullptr;

    for (int t = 0; t < seq_len; ++t)
    {
        float sum = 0.0f;
        if (t < effective_seq_len)
        {
            for (int k = 0; k < kernel_size; ++k)
            {
                const int src_t = t - ks_minus1 + k;
                const float val =
                    (src_t >= 0)
                        ? input[static_cast<size_t>(src_t) * channels + ch]
                        : (channel_initial_state
                               ? channel_initial_state[ks_minus1 + src_t]
                               : 0.0f);
                sum += val * channel_weight[k];
            }

            if (bias)
                sum += bias[ch];
            if (apply_silu)
                sum = sum / (1.0f + expf(-sum));
        }
        output[static_cast<size_t>(t) * channels + ch] = sum;

        if (state_snapshots && t < effective_seq_len && t < max_snapshot_rows)
        {
            float *snapshot =
                state_snapshots +
                static_cast<size_t>(t) * snapshot_stride_floats +
                static_cast<size_t>(ch) * ks_minus1;
            for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
            {
                const int src_t = t - ks_minus1 + 1 + state_idx;
                snapshot[state_idx] =
                    (src_t >= 0)
                        ? input[static_cast<size_t>(src_t) * channels + ch]
                        : (channel_initial_state
                               ? channel_initial_state[ks_minus1 + src_t]
                               : 0.0f);
            }
        }
    }

    const bool terminal_state_is_snapshotted =
        state_snapshots != nullptr && max_snapshot_rows >= seq_len;
    if (updated_conv_state && !terminal_state_is_snapshotted)
    {
        float *state =
            updated_conv_state + static_cast<size_t>(ch) * ks_minus1;
        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
        {
            const int src_t = effective_seq_len - ks_minus1 + state_idx;
            state[state_idx] =
                (src_t >= 0 && src_t < effective_seq_len)
                    ? input[static_cast<size_t>(src_t) * channels + ch]
                    : channel_initial_state[ks_minus1 + src_t];
        }
    }
}

__global__ void rocm_short_conv1d_state_update_kernel(
    const float *__restrict__ input,
    const float *initial_conv_state,
    float *updated_conv_state,
    const int *__restrict__ effective_seq_len_ptr,
    int seq_len, int channels, int kernel_size)
{
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= channels || !initial_conv_state || !updated_conv_state) return;

    int ks_minus1 = kernel_size - 1;
    if (ks_minus1 <= 0) return;

    int effective_seq_len = seq_len;
    if (effective_seq_len_ptr)
    {
        const int raw_effective = *effective_seq_len_ptr;
        effective_seq_len = raw_effective < 1 ? 1 : (raw_effective > seq_len ? seq_len : raw_effective);
    }

    const float *initial_state = initial_conv_state + ch * ks_minus1;
    float *updated_state = updated_conv_state + ch * ks_minus1;
    for (int state_idx = 0; state_idx < ks_minus1; ++state_idx) {
        int src_t = effective_seq_len - ks_minus1 + state_idx;
        updated_state[state_idx] =
            (src_t >= 0 && src_t < effective_seq_len) ? input[src_t * channels + ch]
                                                       : initial_state[ks_minus1 + src_t];
    }
}

/**
 * @brief Grouped short-conv for tiny request-local row groups.
 *
 * A lane owns one `(request, channel)` pair for the complete row group. This
 * preserves serial decode's causal accumulation order, zeroes padding, writes
 * request-indexed snapshots, and commits one request-owned live state in a
 * single HIP launch.
 */
__global__ void rocm_short_conv1d_batched_small_m_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    const float *initial_request_states,
    float *updated_request_states,
    const int *__restrict__ request_seq_lens,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int request_count,
    int request_seq_len,
    int channels,
    int kernel_size,
    bool apply_silu)
{
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    const int request = blockIdx.y;
    if (ch >= channels || request >= request_count)
        return;

    const int ks_minus1 = kernel_size - 1;
    if (ks_minus1 <= 0)
        return;

    const int raw_real_len = request_seq_lens[request];
    const int real_len =
        raw_real_len < 0 ? 0 :
        (raw_real_len > request_seq_len ? request_seq_len : raw_real_len);
    const size_t row_base =
        static_cast<size_t>(request) *
        static_cast<size_t>(request_seq_len);
    const size_t element_base = row_base * static_cast<size_t>(channels);
    const size_t state_base =
        static_cast<size_t>(request) *
        static_cast<size_t>(channels) *
        static_cast<size_t>(ks_minus1);
    const float *channel_initial_state =
        initial_request_states + state_base +
        static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
    float *channel_updated_state =
        updated_request_states + state_base +
        static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
    const float *channel_weight =
        weight + static_cast<size_t>(ch) * static_cast<size_t>(kernel_size);

    for (int t = 0; t < request_seq_len; ++t)
    {
        float sum = 0.0f;
        if (t < real_len)
        {
            for (int k = 0; k < kernel_size; ++k)
            {
                const int src_t = t - ks_minus1 + k;
                const float value =
                    src_t >= 0
                        ? input[element_base +
                                static_cast<size_t>(src_t) * channels + ch]
                        : channel_initial_state[ks_minus1 + src_t];
                sum += value * channel_weight[k];
            }
            if (bias)
                sum += bias[ch];
            if (apply_silu)
                sum = sum / (1.0f + expf(-sum));
        }
        output[element_base + static_cast<size_t>(t) * channels + ch] = sum;

        const int snapshot_row = request * request_seq_len + t;
        if (state_snapshots &&
            t < real_len &&
            snapshot_row < max_snapshot_rows)
        {
            float *snapshot =
                state_snapshots +
                static_cast<size_t>(snapshot_row) *
                    static_cast<size_t>(snapshot_stride_floats) +
                static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
            for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
            {
                const int src_t = t - ks_minus1 + 1 + state_idx;
                snapshot[state_idx] =
                    src_t >= 0
                        ? input[element_base +
                                static_cast<size_t>(src_t) * channels + ch]
                        : channel_initial_state[ks_minus1 + src_t];
            }
        }
    }

    const bool terminal_state_is_snapshotted =
        state_snapshots != nullptr &&
        max_snapshot_rows >= request_count * request_seq_len;
    if (!terminal_state_is_snapshotted)
    {
        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
        {
            const int src_t = real_len - ks_minus1 + state_idx;
            channel_updated_state[state_idx] =
                src_t >= 0 && src_t < real_len
                    ? input[element_base +
                            static_cast<size_t>(src_t) * channels + ch]
                    : channel_initial_state[ks_minus1 + src_t];
        }
    }
}

/**
 * @brief Grouped long-prefill convolution over flattened request rows.
 *
 * Threads cover `(request, row, channel)` directly. Causal indexing is always
 * request-relative, and the companion commit kernel publishes terminal state
 * only after every output thread has finished reading the initial state bank.
 */
__global__ void rocm_short_conv1d_batched_prefill_kernel(
    const float *__restrict__ input,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ output,
    const float *__restrict__ request_states,
    const int *__restrict__ request_seq_lens,
    float *__restrict__ state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int request_count,
    int request_seq_len,
    int channels,
    int kernel_size,
    bool apply_silu)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int request_elements = request_seq_len * channels;
    const int total = request_count * request_elements;
    if (idx >= total)
        return;

    const int request = idx / request_elements;
    const int request_element = idx - request * request_elements;
    const int t = request_element / channels;
    const int ch = request_element - t * channels;
    const int ks_minus1 = kernel_size - 1;
    const int raw_real_len = request_seq_lens[request];
    const int real_len =
        raw_real_len < 0 ? 0 :
        (raw_real_len > request_seq_len ? request_seq_len : raw_real_len);
    const size_t row_base =
        static_cast<size_t>(request) *
        static_cast<size_t>(request_seq_len);
    const size_t element_base = row_base * static_cast<size_t>(channels);
    const float *channel_state =
        request_states +
        static_cast<size_t>(request) * channels * ks_minus1 +
        static_cast<size_t>(ch) * ks_minus1;

    float sum = 0.0f;
    if (t < real_len)
    {
        for (int k = 0; k < kernel_size; ++k)
        {
            const int src_t = t - ks_minus1 + k;
            const float value =
                src_t >= 0
                    ? input[element_base +
                            static_cast<size_t>(src_t) * channels + ch]
                    : channel_state[ks_minus1 + src_t];
            sum += value * weight[ch * kernel_size + k];
        }
        if (bias)
            sum += bias[ch];
        if (apply_silu)
            sum = sum / (1.0f + expf(-sum));
    }
    output[element_base + static_cast<size_t>(t) * channels + ch] = sum;

    const int snapshot_row = request * request_seq_len + t;
    if (state_snapshots &&
        t < real_len &&
        snapshot_row < max_snapshot_rows)
    {
        float *snapshot =
            state_snapshots +
            static_cast<size_t>(snapshot_row) *
                static_cast<size_t>(snapshot_stride_floats) +
            static_cast<size_t>(ch) * static_cast<size_t>(ks_minus1);
        for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
        {
            const int src_t = t - ks_minus1 + 1 + state_idx;
            snapshot[state_idx] =
                src_t >= 0
                    ? input[element_base +
                            static_cast<size_t>(src_t) * channels + ch]
                    : channel_state[ks_minus1 + src_t];
        }
    }
}

/** @brief Commit each request's terminal real-row convolution state. */
__global__ void rocm_short_conv1d_batched_state_update_kernel(
    const float *__restrict__ input,
    const float *initial_request_states,
    float *updated_request_states,
    const int *__restrict__ request_seq_lens,
    int request_count,
    int request_seq_len,
    int channels,
    int kernel_size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = request_count * channels;
    if (idx >= total)
        return;

    const int request = idx / channels;
    const int ch = idx - request * channels;
    const int ks_minus1 = kernel_size - 1;
    if (ks_minus1 <= 0)
        return;
    const int raw_real_len = request_seq_lens[request];
    const int real_len =
        raw_real_len < 0 ? 0 :
        (raw_real_len > request_seq_len ? request_seq_len : raw_real_len);
    const size_t element_base =
        static_cast<size_t>(request) *
        static_cast<size_t>(request_seq_len) *
        static_cast<size_t>(channels);
    const float *channel_initial_state =
        initial_request_states +
        static_cast<size_t>(request) * channels * ks_minus1 +
        static_cast<size_t>(ch) * ks_minus1;
    float *channel_updated_state =
        updated_request_states +
        static_cast<size_t>(request) * channels * ks_minus1 +
        static_cast<size_t>(ch) * ks_minus1;

    for (int state_idx = 0; state_idx < ks_minus1; ++state_idx)
    {
        const int src_t = real_len - ks_minus1 + state_idx;
        channel_updated_state[state_idx] =
            src_t >= 0 && src_t < real_len
                ? input[element_base +
                        static_cast<size_t>(src_t) * channels + ch]
                : channel_initial_state[ks_minus1 + src_t];
    }
}

// =========================================================================
// GatedRMSNorm Kernel — Cooperative (one block per normalization group)
//
// Old kernel: 1 thread per group → for decode (seq_len=1), only 20 threads
//   across the entire GPU. Each thread sequentially loops over norm_dim.
// New kernel: 1 block per group → all threads cooperate on the reduction.
//   With norm_dim=128 and 128 threads/block, each thread handles 1 element.
//   Uses warp shuffle reduction (no shared memory for the sum).
//   Register-caches input values to avoid re-reading from global memory.
// =========================================================================

__global__ void rocm_gated_rmsnorm_kernel(
    const float *__restrict__ input,
    const float *__restrict__ gate,
    const float *__restrict__ gamma,
    float *__restrict__ output,
    int total_work,
    int n_groups,
    int norm_dim,
    int d_model,
    int gamma_period,
    float eps,
    bool subtract_one,
    bool gate_silu)
{
    // One block per (seq_position, group) pair
    const int work_idx = blockIdx.x;
    if (work_idx >= total_work) return;

    const int t = work_idx / n_groups;
    const int g = work_idx % n_groups;
    const int offset = t * d_model + g * norm_dim;
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;

    // Pass 1: Each thread handles its slice of norm_dim, accumulate sum_sq
    constexpr int kMaxPerThread = 8; // handles norm_dim up to 8*256=2048
    float local_vals[kMaxPerThread];
    int local_count = 0;

    float sum_sq = 0.0f;
    for (int j = tid; j < norm_dim; j += block_size) {
        float v = input[offset + j];
        local_vals[local_count++] = v;
        sum_sq += v * v;
    }

    // Warp shuffle reduction (64-wide wavefronts on GCN/CDNA)
    for (int offset_s = 32; offset_s > 0; offset_s >>= 1)
        sum_sq += __shfl_xor(sum_sq, offset_s);

    // Cross-wavefront reduction via shared memory
    const int lane = tid & 63;
    const int warp_id = tid >> 6;
    const int num_warps = (block_size + 63) >> 6;

    __shared__ float warp_sums[4]; // max 256 threads = 4 wavefronts
    if (lane == 0) warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++) total += warp_sums[w];
        warp_sums[0] = total;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / (float)norm_dim + eps);

    // Pass 2: Normalize with register-cached values
    int idx = 0;
    for (int j = tid; j < norm_dim; j += block_size) {
        float normalized = local_vals[idx++] * inv_rms;
        int gamma_idx = j % gamma_period;
        float gamma_eff = subtract_one ? (1.0f + gamma[gamma_idx]) : gamma[gamma_idx];
        float gate_val = gate[offset + j];
        float gate_act;
        if (gate_silu)
            gate_act = gate_val / (1.0f + expf(-gate_val));
        else
            gate_act = gate_val;
        output[offset + j] = normalized * gamma_eff * gate_act;
    }
}

// =========================================================================
// AttentionOutputGate Kernel
// =========================================================================

__global__ void rocm_attention_output_gate_kernel(
    const float *__restrict__ input,
    const float *__restrict__ gate,
    float *__restrict__ output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sig = 1.0f / (1.0f + expf(-gate[idx]));
        output[idx] = sig * input[idx];
    }
}

} // anonymous namespace (kernel functions)

// =========================================================================
// QKV Deinterleave Kernel (for merged QKV buffer on device)
// =========================================================================

namespace {

/** @brief Captured request-local copy; the shared mapping never touches padding. */
__global__ void rocm_gdn_deinterleave_qkv_kernel(
    const float *__restrict__ merged,
    float *__restrict__ out_q,
    float *__restrict__ out_k,
    float *__restrict__ out_v,
    llaminar2::DeviceRequestRowRanges rows, int n_k_heads, int n_v_heads,
    int d_k, int d_v, int global_v_offset)
{
    llaminar2::gdnDeinterleaveLiveRows(merged, out_q, out_k, out_v,
        rows, n_k_heads, n_v_heads, d_k, d_v, global_v_offset);
}

/**
 * @brief Publish one captured verifier-state row selected by a device scalar.
 *
 * The accepted-row selector is produced by GPU stochastic verifier metadata.
 * Reading it in this kernel keeps recurrent and short-convolution state
 * publication on the verifier stream and avoids a host synchronization merely
 * to discover the accepted row.  Arena-backed recurrent rows are 16-byte
 * aligned, so the common path moves four FP32 values per memory instruction.
 * A grid-stride loop keeps the launch at a bounded number of workgroups instead
 * of scheduling one thousand mostly single-iteration blocks for a 2 MiB state.
 *
 * The alignment predicate is workgroup-uniform.  Non-vector tails and unusual
 * externally supplied buffers retain exact scalar copy semantics, while every
 * production arena row takes the vectorized path.  No value is converted or
 * recomputed: publication remains byte-for-byte state selection.
 *
 * @param live_state Scalar live recurrent or short-convolution state.
 * @param request_zero_state Request-bank slot representing the same sequence.
 * @param capture Post-row verifier state matrix.
 * @param row_index Device-owned accepted-row selector.
 * @param rows Number of rows available in @p capture.
 * @param state_size Number of FP32 values in one state row.
 */
__global__ __launch_bounds__(256)
void rocm_gdn_publish_capture_row_from_device_index_kernel(
    float *live_state,
    float *request_zero_state,
    const float *__restrict__ capture,
    const int *__restrict__ row_index,
    int rows,
    int state_size)
{
    if (!live_state || !request_zero_state || !capture || !row_index ||
        rows <= 0 || state_size <= 0)
        return;
    const int row = *row_index;
    if (row < 0 || row >= rows)
        return;

    const float *src =
        capture + static_cast<size_t>(row) * static_cast<size_t>(state_size);
    const int thread =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
        static_cast<int>(threadIdx.x);
    const int stride =
        static_cast<int>(blockDim.x) * static_cast<int>(gridDim.x);
    const bool vector_aligned =
        ((reinterpret_cast<uintptr_t>(live_state) |
          reinterpret_cast<uintptr_t>(request_zero_state) |
          reinterpret_cast<uintptr_t>(src)) & 0x0fu) == 0;
    const bool shared_request_zero_owner =
        live_state == request_zero_state;

    if (vector_aligned)
    {
        float4 *live4 = reinterpret_cast<float4 *>(live_state);
        float4 *request4 = reinterpret_cast<float4 *>(request_zero_state);
        const float4 *src4 = reinterpret_cast<const float4 *>(src);
        const int vector_count = state_size / 4;
        for (int vector_index = thread;
             vector_index < vector_count;
             vector_index += stride)
        {
            const float4 value = src4[vector_index];
            live4[vector_index] = value;
            if (!shared_request_zero_owner)
                request4[vector_index] = value;
        }

        const int tail_begin = vector_count * 4;
        for (int i = tail_begin + thread; i < state_size; i += stride)
        {
            const float value = src[i];
            live_state[i] = value;
            if (!shared_request_zero_owner)
                request_zero_state[i] = value;
        }
        return;
    }

    for (int i = thread; i < state_size; i += stride)
    {
        const float value = src[i];
        live_state[i] = value;
        if (!shared_request_zero_owner)
            request_zero_state[i] = value;
    }
}

/**
 * @brief Publish selected rows into request state and its scalar request-zero owner.
 *
 * A negative row index leaves that request untouched. This lets the compact
 * device-side MTP metadata publish only accepted requests without a host sync.
 * Request zero is also written to the scalar live allocation by the same lane,
 * making coherence a property of this publication transaction rather than a
 * second memcpy that can be omitted, reordered, or captured independently.
 */
__global__ void rocm_gdn_publish_capture_rows_from_device_indices_kernel(
    float *request_states,
    float *request_zero_live_state,
    const float *__restrict__ capture,
    const int *__restrict__ row_indices,
    int request_count,
    int row_index_stride,
    int request_row_width,
    int rows,
    int state_size)
{
    if (!request_states || !request_zero_live_state || !capture || !row_indices ||
        request_count <= 0 || row_index_stride <= 0 ||
        rows <= 0 || state_size <= 0)
    {
        return;
    }

    const int total = request_count * state_size;
    const int stride = blockDim.x * gridDim.x;
    for (int linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total;
         linear += stride)
    {
        const int request = linear / state_size;
        const int state_offset = linear - request * state_size;
        const int metadata_value = row_indices[request * row_index_stride];
        const int row =
            request_row_width > 0
                ? request * request_row_width + metadata_value - 1
                : metadata_value;
        if (row < 0 || row >= rows)
            continue;
        const float value =
            capture[static_cast<size_t>(row) * static_cast<size_t>(state_size) +
                    static_cast<size_t>(state_offset)];
        request_states[linear] = value;
        if (request == 0 && request_zero_live_state != request_states)
            request_zero_live_state[state_offset] = value;
    }
}

/**
 * @brief Convert rank-major allgather output to modulo-linked group order.
 */
__global__ void rocm_gdn_reassemble_modulo_linked_state_kernel(
    const float *__restrict__ gathered,
    float *__restrict__ full,
    int degree,
    int global_key_heads,
    int repeat_factor,
    int key_elements_per_head,
    int value_elements_per_head,
    int prefix_group_count)
{
    if (!gathered || !full ||
        degree <= 0 || global_key_heads <= 0 ||
        global_key_heads % degree != 0 || repeat_factor <= 0 ||
        key_elements_per_head < 0 || value_elements_per_head <= 0 ||
        prefix_group_count < 0)
    {
        return;
    }

    const int local_key_heads = global_key_heads / degree;
    const int local_key_group = local_key_heads * key_elements_per_head;
    const int full_key_group = global_key_heads * key_elements_per_head;
    const int local_value_group = local_key_heads * value_elements_per_head;
    const int full_value_group = global_key_heads * value_elements_per_head;
    const int local_prefix = prefix_group_count * local_key_group;
    const int full_prefix = prefix_group_count * full_key_group;
    const int local_state_size =
        local_prefix + repeat_factor * local_value_group;
    const int full_state_size =
        full_prefix + repeat_factor * full_value_group;
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < full_state_size;
         idx += stride)
    {
        int participant = 0;
        int participant_offset = 0;
        if (idx < full_prefix)
        {
            const int group = idx / full_key_group;
            const int within_group = idx - group * full_key_group;
            participant = within_group / local_key_group;
            participant_offset =
                group * local_key_group + within_group % local_key_group;
        }
        else
        {
            const int value_offset = idx - full_prefix;
            const int repeat = value_offset / full_value_group;
            const int within_group = value_offset - repeat * full_value_group;
            participant = within_group / local_value_group;
            participant_offset =
                local_prefix + repeat * local_value_group +
                within_group % local_value_group;
        }
        full[idx] = gathered[
            static_cast<size_t>(participant) *
                static_cast<size_t>(local_state_size) +
            static_cast<size_t>(participant_offset)];
    }
}

} // anonymous namespace (deinterleave kernel)

// =========================================================================
// GPU transfer helpers (called from headers via extern "C")
// =========================================================================

extern "C" {

void rocmGDN_gpu_memset_zero(float *ptr, size_t count)
{
    if (ptr && count > 0) (void)hipMemset(ptr, 0, count * sizeof(float));
}

void rocmGDN_gpu_memset_zero_async(float *ptr, size_t count, void *stream)
{
    if (ptr && count > 0) (void)hipMemsetAsync(ptr, 0, count * sizeof(float), (hipStream_t)stream);
}

void rocmGDN_gpu_set_device(int ordinal)
{
    (void)hipSetDevice(ordinal);
}

void rocmGDN_gpu_memcpy(float *dst, const float *src, size_t count)
{
    (void)hipMemcpy(dst, src, count * sizeof(float), hipMemcpyDefault);
}

void rocmGDN_gpu_memcpy_async(float *dst, const float *src, size_t count, void *stream)
{
    (void)hipMemcpyAsync(dst, src, count * sizeof(float), hipMemcpyDefault, (hipStream_t)stream);
}

void rocmGDN_gpu_memcpy_d2h(float *host_dst, const float *device_src, size_t count)
{
    (void)hipMemcpy(host_dst, device_src, count * sizeof(float), hipMemcpyDeviceToHost);
}

void rocmGDN_gpu_memcpy_d2h_async(float *host_dst, const float *device_src, size_t count, void *stream)
{
    (void)hipMemcpyAsync(host_dst, device_src, count * sizeof(float), hipMemcpyDeviceToHost, (hipStream_t)stream);
}

void rocmGDN_stream_synchronize(void *stream)
{
    (void)hipStreamSynchronize((hipStream_t)stream);
}

bool rocmGDN_gpu_publish_capture_row_from_device_index(
    float *live_state,
    float *request_zero_state,
    const float *capture,
    const int *device_row_index,
    int rows,
    int state_size,
    int device_idx,
    void *stream)
{
    if (!live_state || !request_zero_state || !capture || !device_row_index ||
        rows <= 0 || state_size <= 0 || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int threads = 256;
    constexpr int max_blocks = 256;
    const int vector_work_items = (state_size + 3) / 4;
    int blocks = (vector_work_items + threads - 1) / threads;
    if (blocks < 1)
        blocks = 1;
    if (blocks > max_blocks)
        blocks = max_blocks;
    hipLaunchKernelGGL(
        rocm_gdn_publish_capture_row_from_device_index_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        (hipStream_t)stream,
        live_state,
        request_zero_state,
        capture,
        device_row_index,
        rows,
        state_size);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGDN_gpu_publish_capture_row_from_device_index] %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_gpu_publish_capture_rows_from_device_indices(
    float *request_states,
    float *request_zero_live_state,
    const float *capture,
    const int *device_row_indices,
    int request_count,
    int row_index_stride,
    int rows,
    int state_size,
    int device_idx,
    void *stream)
{
    if (!request_states || !request_zero_live_state || !capture || !device_row_indices ||
        request_count <= 0 || row_index_stride <= 0 ||
        rows <= 0 || state_size <= 0 || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int threads = 256;
    int blocks = ((request_count * state_size) + threads - 1) / threads;
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    hipLaunchKernelGGL(
        rocm_gdn_publish_capture_rows_from_device_indices_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        (hipStream_t)stream,
        request_states,
        request_zero_live_state,
        capture,
        device_row_indices,
        request_count,
        row_index_stride,
        /*request_row_width=*/0,
        rows,
        state_size);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGDN_gpu_publish_capture_rows_from_device_indices] %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_gpu_publish_capture_terminal_rows_from_device_lengths(
    float *request_states,
    float *request_zero_live_state,
    const float *capture,
    const int *device_request_seq_lens,
    int request_count,
    int request_row_width,
    int rows,
    int state_size,
    int device_idx,
    void *stream)
{
    if (!request_states || !request_zero_live_state || !capture || !device_request_seq_lens ||
        request_count <= 0 || request_row_width <= 0 ||
        rows < request_count * request_row_width ||
        state_size <= 0 || !stream)
    {
        return false;
    }

    llaminar2::HipDeviceGuard::setDevice(device_idx);
    constexpr int threads = 256;
    int blocks = ((request_count * state_size) + threads - 1) / threads;
    blocks = std::max(1, std::min(blocks, 1024));
    hipLaunchKernelGGL(
        rocm_gdn_publish_capture_rows_from_device_indices_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        static_cast<hipStream_t>(stream),
        request_states,
        request_zero_live_state,
        capture,
        device_request_seq_lens,
        request_count,
        /*row_index_stride=*/1,
        request_row_width,
        rows,
        state_size);
    const hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGDN_gpu_publish_capture_terminal_rows_from_device_lengths] %s\n",
                hipGetErrorString(error));
        return false;
    }
    return true;
}

bool rocmGDN_reassemble_modulo_linked_state(
    const float *gathered,
    float *full,
    int degree,
    int global_key_heads,
    int global_value_heads,
    int key_elements_per_head,
    int value_elements_per_head,
    int prefix_group_count,
    int device_idx,
    void *stream)
{
    if (!gathered || !full || !stream ||
        degree <= 0 || global_key_heads <= 0 ||
        global_value_heads <= 0 ||
        global_key_heads % degree != 0 ||
        global_value_heads % global_key_heads != 0 ||
        (prefix_group_count != 0 && prefix_group_count != 2) ||
        (prefix_group_count > 0 && key_elements_per_head <= 0) ||
        value_elements_per_head <= 0)
    {
        return false;
    }

    const long long full_state_size_wide =
        static_cast<long long>(prefix_group_count) * global_key_heads *
            key_elements_per_head +
        static_cast<long long>(global_value_heads) *
            value_elements_per_head;
    if (full_state_size_wide <= 0 ||
        full_state_size_wide > std::numeric_limits<int>::max())
    {
        return false;
    }

    hipError_t set_err = static_cast<hipError_t>(llaminar2::HipDeviceGuard::setDevice(device_idx));
    if (set_err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGDN_reassemble_modulo_linked_state] hipSetDevice(%d) failed: %s\n",
                device_idx,
                hipGetErrorString(set_err));
        return false;
    }
    (void)hipGetLastError();
    const int full_state_size = static_cast<int>(full_state_size_wide);
    constexpr int threads = 256;
    int blocks = (full_state_size + threads - 1) / threads;
    if (blocks < 1)
        blocks = 1;
    if (blocks > 1024)
        blocks = 1024;
    hipLaunchKernelGGL(
        rocm_gdn_reassemble_modulo_linked_state_kernel,
        dim3(blocks),
        dim3(threads),
        0,
        (hipStream_t)stream,
        gathered,
        full,
        degree,
        global_key_heads,
        global_value_heads / global_key_heads,
        key_elements_per_head,
        value_elements_per_head,
        prefix_group_count);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGDN_reassemble_modulo_linked_state] %s\n",
                hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" (memory helpers)

// =========================================================================
// QKV Deinterleave Wrapper
// =========================================================================

extern "C" {

/** @brief Validate immutable extent and enqueue live-row copying on a non-null stream. */
bool rocmGDN_deinterleave_qkv(
    const float *merged, float *out_q, float *out_k, float *out_v,
    llaminar2::DeviceRequestRowRanges rows, int n_k_heads, int n_v_heads,
    int d_k, int d_v, int global_v_offset,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("merged", merged);
        print_pointer_attrs("out_q", out_q);
        print_pointer_attrs("out_k", out_k);
        print_pointer_attrs("out_v", out_v);
    }

    const int seq_len = rows.physicalRows();
    const int total = llaminar2::gdnDeinterleaveElementsPerRequest(
        rows, n_k_heads, n_v_heads, d_k, d_v);
    if (!stream || !merged || !out_q || !out_k || !out_v || total <= 0)
    {
        fprintf(stderr,
                "[rocmGDN_deinterleave_qkv] invalid launch geometry: "
                "seq_len=%d n_k_heads=%d n_v_heads=%d d_k=%d d_v=%d "
                "global_v_offset=%d total=%d stream=%p\n",
                seq_len, n_k_heads, n_v_heads, d_k, d_v,
                global_v_offset, total, stream);
        return false;
    }

    int threads = 256;
    const int blocks = 1 + (total - 1) / threads;

    hipLaunchKernelGGL(rocm_gdn_deinterleave_qkv_kernel,
        dim3(blocks, rows.requests()), dim3(threads), 0, (hipStream_t)stream,
        merged, out_q, out_k, out_v,
        rows, n_k_heads, n_v_heads, d_k, d_v, global_v_offset);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        int current_device = -1;
        (void)hipGetDevice(&current_device);
        fprintf(stderr,
                "[rocmGDN_deinterleave_qkv] %s "
                "(seq_len=%d n_k_heads=%d n_v_heads=%d d_k=%d d_v=%d "
                "global_v_offset=%d total=%d blocks=%d requested_device=%d current_device=%d "
                "merged=%p out_q=%p out_k=%p out_v=%p stream=%p)\n",
                hipGetErrorString(err), seq_len, n_k_heads, n_v_heads,
                d_k, d_v, global_v_offset, total, blocks,
                device_idx, current_device,
                (const void *)merged, (void *)out_q, (void *)out_k, (void *)out_v, stream);
        return false;
    }
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_deinterleave_qkv_sync] %s\n", hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

} // extern "C" (deinterleave)

// =========================================================================
// Extern "C" Wrapper Functions
// =========================================================================

extern "C" {

bool rocmGDN_recurrent_step(
    const float *q, const float *k, const float *v,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, const float *initial_state, float *updated_state,
    int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("q", q);
        print_pointer_attrs("k", k);
        print_pointer_attrs("v", v);
        print_pointer_attrs("alpha", alpha);
        print_pointer_attrs("beta", beta_raw);
        print_pointer_attrs("A_log", A_log);
        print_pointer_attrs("dt_bias", dt_bias);
        print_pointer_attrs("output", output);
        print_pointer_attrs("initial_state", initial_state);
        print_pointer_attrs("updated_state", updated_state);
    }

    if (!launch_rocm_gdn_recurrent_step(
            "rocmGDN_recurrent_step",
            q, k, v, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            /*request_count=*/1,
            /*request_seq_len=*/1,
            n_heads, d_k, d_v, use_qk_l2norm,
            /*effective_seq_len_ptr=*/nullptr,
            /*effective_row_idx=*/-1,
            /*state_snapshots=*/nullptr,
            /*snapshot_stride_floats=*/0,
            /*max_snapshot_rows=*/0,
            static_cast<hipStream_t>(stream)))
    {
        return false;
    }
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        const hipError_t err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_recurrent_step_sync] %s\n", hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmGDN_recurrent_step_effective_row(
    const float *q, const float *k, const float *v,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, const float *initial_state, float *updated_state,
    int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    const int *device_effective_seq_len,
    int row_idx,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (!launch_rocm_gdn_recurrent_step(
            "rocmGDN_recurrent_step_effective_row",
            q, k, v, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            /*request_count=*/1,
            /*request_seq_len=*/1,
            n_heads, d_k, d_v, use_qk_l2norm,
            device_effective_seq_len,
            row_idx,
            /*state_snapshots=*/nullptr,
            /*snapshot_stride_floats=*/0,
            /*max_snapshot_rows=*/0,
            static_cast<hipStream_t>(stream)))
    {
        return false;
    }
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        const hipError_t err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_recurrent_step_effective_row_sync] %s\n", hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

/**
 * @brief Enqueue the shape-appropriate, serial-row-exact GDN recurrence.
 *
 * All pointers and the explicit stream remain caller-owned for the captured
 * graph lifetime. Complete speculative snapshots select the verifier route;
 * ordinary long prefill specializes key/value widths and block geometry,
 * independently of participant-local head count. TP sharding changes strides
 * and grid width, not the arithmetic or storage contract of each head.
 * @return False for invalid bindings/geometry or a failed HIP launch.
 */
bool rocmGDN_chunk_forward_batched_kernel_route(
    const float *Q, const float *K, const float *V,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, const float *initial_state, float *updated_state,
    int seq_len, int request_count, int request_seq_len,
    int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    const int *device_effective_seq_len,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    if (!Q || !K || !V || !alpha || !beta_raw || !A_log || !dt_bias ||
        !output || !initial_state || !updated_state || !stream ||
        seq_len <= 0 || request_count <= 0 || request_seq_len <= 0 ||
        seq_len != request_count * request_seq_len ||
        n_heads <= 0 || (d_k != 64 && d_k != 128) || d_v <= 0)
    {
        return false;
    }

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("Q", Q);
        print_pointer_attrs("K", K);
        print_pointer_attrs("V", V);
        print_pointer_attrs("alpha", alpha);
        print_pointer_attrs("beta", beta_raw);
        print_pointer_attrs("A_log", A_log);
        print_pointer_attrs("dt_bias", dt_bias);
        print_pointer_attrs("output", output);
        print_pointer_attrs("initial_state", initial_state);
        print_pointer_attrs("updated_state", updated_state);
    }

    const int required_snapshot_stride = n_heads * d_k * d_v;
    const bool complete_verifier_capture =
        state_snapshots != nullptr &&
        snapshot_stride_floats >= required_snapshot_stride &&
        max_snapshot_rows >= seq_len;

    /*
     * Widths through eight use the exact scalar recurrence for launch economy.
     * A complete verifier capture matrix selects it at every runtime M because
     * any post-row state may be published and must therefore preserve scalar
     * decode's wavefront reduction schedule. Long ordinary prefill supplies no
     * complete capture matrix and continues through the optimized path below.
     */
    if (request_seq_len <= 8 || complete_verifier_capture)
    {
        if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
        {
            fprintf(
                stderr,
                "[rocmGDN_recurrent_step_batched] requests=%d heads=%d d_k=%d d_v=%d initial_state=%p updated_state=%p output=%p lengths=%p\n",
                request_count,
                n_heads,
                d_k,
                d_v,
                static_cast<const void *>(initial_state),
                static_cast<void *>(updated_state),
                static_cast<void *>(output),
                static_cast<const void *>(device_effective_seq_len));
        }
        if (!launch_rocm_gdn_recurrent_step(
                "rocmGDN_recurrent_step_batched",
                Q, K, V, alpha, beta_raw, A_log, dt_bias,
                output, initial_state, updated_state,
                request_count, request_seq_len,
                n_heads, d_k, d_v, use_qk_l2norm,
                device_effective_seq_len,
                /*effective_row_idx=*/0,
                state_snapshots,
                snapshot_stride_floats,
                max_snapshot_rows,
                static_cast<hipStream_t>(stream)))
        {
            return false;
        }
        if (llaminar2::debugEnv().rocm.gdn_sync)
        {
            const hipError_t err =
                hipStreamSynchronize(static_cast<hipStream_t>(stream));
            if (err != hipSuccess)
            {
                fprintf(stderr,
                        "[rocmGDN_recurrent_step_batched_sync] %s\n",
                        hipGetErrorString(err));
                return false;
            }
        }
        return true;
    }

    // Four threads per column retain scalar decode's four fixed key partitions.
    // Bounded prefill places those threads in the same wave; other geometries
    // exchange them through LDS. Neither changes the partition or sum order.
    int col_threads = 256;
    int cols_per_block = col_threads / 4; // 64 columns per block
    if (d_v <= 64)
    {
        col_threads = 128;
        cols_per_block = col_threads / 4; // 32 columns per block
    }
    int num_col_blocks = (d_v + cols_per_block - 1) / cols_per_block;
    // Specialize the recurrence geometry, not a model's unsharded head count.
    // The head count only controls independent workgroups and row strides;
    // restricting this route to 32 made TP shards use slower dynamic indexing.
    const bool uses_fixed_width_kernel =
        d_k == 128 && d_v == 128 && col_threads == 256;
    // Two additional per-row scales fit in LDS through the 2048-row bucket.
    // Larger buckets retain the same on-row normalization arithmetic without
    // exceeding gfx906's shared-memory limit. This is capture-time geometry,
    // not a runtime capability switch or a new physical allocation.
    const bool cache_qk_normalization =
        uses_fixed_width_kernel && request_seq_len <= 2048;
    const int lds_state_tail_rows =
        uses_fixed_width_kernel && !cache_qk_normalization ? 5 : 0;
    // Bounded Q/K/V input tile + warp sums[16] + reduction arrays + state tail
    // + per-row decay/beta and, when selected, the two normalization scales.
    const int input_tile_floats = cache_qk_normalization
        ? kGdnPrefillInputTileRows * (2 * d_k + cols_per_block)
        : 2 * d_k;
    int smem_size =
        (input_tile_floats + 16 +
         ((cache_qk_normalization ? 0 : 2) + lds_state_tail_rows) * col_threads +
         (cache_qk_normalization ? 4 : 2) * request_seq_len) * sizeof(float);

    if (cache_qk_normalization && !state_snapshots)
    {
        // Snapshot presence is frozen into this captured launch. Ordinary
        // prefill must not carry a never-taken intermediate-publication branch
        // around every token back edge. Both policies share exact arithmetic.
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_gdn_chunk_forward_kernel<
                128, 128, 256, true, GdnPrefillSnapshots::TerminalOnly>),
            dim3(n_heads, num_col_blocks, request_count), dim3(col_threads),
            smem_size, static_cast<hipStream_t>(stream),
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state, device_effective_seq_len,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            request_count, request_seq_len, n_heads, d_v, use_qk_l2norm);
    }
    else if (cache_qk_normalization)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rocm_gdn_chunk_forward_kernel<128, 128, 256, true>),
            dim3(n_heads, num_col_blocks, request_count),
            dim3(col_threads),
            smem_size,
            (hipStream_t)stream,
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            request_count, request_seq_len,
            n_heads, d_v, use_qk_l2norm);
    }
    else if (uses_fixed_width_kernel)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rocm_gdn_chunk_forward_kernel<128, 128, 256>),
            dim3(n_heads, num_col_blocks, request_count),
            dim3(col_threads),
            smem_size,
            (hipStream_t)stream,
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            request_count, request_seq_len,
            n_heads, d_v, use_qk_l2norm);
    }
    else if (d_k == 64)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_gdn_chunk_forward_kernel<64>),
            dim3(n_heads, num_col_blocks, request_count),
            dim3(col_threads),
            smem_size,
            (hipStream_t)stream,
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            request_count, request_seq_len,
            n_heads, d_v, use_qk_l2norm);
    }
    else
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocm_gdn_chunk_forward_kernel<128>),
            dim3(n_heads, num_col_blocks, request_count),
            dim3(col_threads),
            smem_size,
            (hipStream_t)stream,
            Q, K, V, alpha, beta_raw, A_log, dt_bias,
            output, initial_state, updated_state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            request_count, request_seq_len,
            n_heads, d_v, use_qk_l2norm);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_chunk_forward] %s\n", hipGetErrorString(err));
        return false;
    }
    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_chunk_forward_sync] %s\n", hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

bool rocmGDN_chunk_forward_kernel_route(
    const float *Q, const float *K, const float *V,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, const float *initial_state, float *updated_state,
    int seq_len, int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    const int *device_effective_seq_len,
    int device_idx, void *stream)
{
    return rocmGDN_chunk_forward_batched_kernel_route(
        Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, initial_state, updated_state,
        seq_len, /*request_count=*/1, /*request_seq_len=*/seq_len,
        n_heads, d_k, d_v, use_qk_l2norm,
        state_snapshots,
        snapshot_stride_floats,
        max_snapshot_rows,
        device_effective_seq_len,
        device_idx, stream);
}

bool rocmGDN_chunk_forward(
    const float *Q, const float *K, const float *V,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, const float *initial_state, float *updated_state,
    int seq_len, int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    /*
     * MTP verifier chunks are tiny, but routing them through one kernel per row
     * loses the speedup we need.  The chunk kernel processes timesteps
     * sequentially inside one launch, keeps each thread's recurrence-state
     * slice resident in registers, and publishes the post-row state snapshots.
     * That is the desired grouped/concurrent contract: no hidden serial launch
     * loop, while preserving the same recurrence order as serial decode.
     */
    return rocmGDN_chunk_forward_kernel_route(
        Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, initial_state, updated_state,
        seq_len, n_heads, d_k, d_v, use_qk_l2norm,
        state_snapshots,
        snapshot_stride_floats,
        max_snapshot_rows,
        nullptr,
        device_idx, stream);
}

bool rocmGDN_chunk_forward_effective(
    const float *Q, const float *K, const float *V,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, const float *initial_state, float *updated_state,
    int seq_len, int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    const int *device_effective_seq_len,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    if (llaminar2::debugEnv().rocm.gdn_ptr_trace)
    {
        print_pointer_attrs("Q", Q);
        print_pointer_attrs("K", K);
        print_pointer_attrs("V", V);
        print_pointer_attrs("alpha", alpha);
        print_pointer_attrs("beta", beta_raw);
        print_pointer_attrs("A_log", A_log);
        print_pointer_attrs("dt_bias", dt_bias);
        print_pointer_attrs("output", output);
        print_pointer_attrs("initial_state", initial_state);
        print_pointer_attrs("updated_state", updated_state);
        print_pointer_attrs("effective_seq_len", device_effective_seq_len);
    }

    const bool ok = rocmGDN_chunk_forward_kernel_route(
        Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, initial_state, updated_state,
        seq_len, n_heads, d_k, d_v, use_qk_l2norm,
        state_snapshots,
        snapshot_stride_floats,
        max_snapshot_rows,
        device_effective_seq_len,
        device_idx, stream);
    if (!ok)
        return false;

    if (llaminar2::debugEnv().rocm.gdn_sync)
    {
        hipError_t err = hipStreamSynchronize((hipStream_t)stream);
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmGDN_chunk_forward_effective_sync] %s\n", hipGetErrorString(err));
            return false;
        }
    }
    return true;
}

/**
 * @brief Launch one request-grouped HIP recurrence over padded rows.
 *
 * Request identity is encoded in grid Z. Every request therefore advances its
 * own state bank inside the same launch while reading its real row count from
 * device memory; no host request loop participates in graph replay.
 */
bool rocmGDN_chunk_forward_batched_effective(
    const float *Q, const float *K, const float *V,
    const float *alpha, const float *beta_raw,
    const float *A_log, const float *dt_bias,
    float *output, const float *initial_state, float *updated_state,
    int seq_len, int request_count, int request_seq_len,
    int n_heads, int d_k, int d_v,
    bool use_qk_l2norm,
    const int *device_effective_seq_lens,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int device_idx, void *stream)
{
    if (!device_effective_seq_lens)
        return false;
    return rocmGDN_chunk_forward_batched_kernel_route(
        Q, K, V, alpha, beta_raw, A_log, dt_bias,
        output, initial_state, updated_state,
        seq_len, request_count, request_seq_len,
        n_heads, d_k, d_v, use_qk_l2norm,
        state_snapshots,
        snapshot_stride_floats,
        max_snapshot_rows,
        device_effective_seq_lens,
        device_idx, stream);
}

bool rocmGDN_short_conv1d(
    const float *input, const float *weight, const float *bias,
    float *output,
    const float *initial_conv_state,
    float *updated_conv_state,
    int seq_len, int channels, int kernel_size,
    bool apply_silu,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    const int required_snapshot_stride = channels * (kernel_size - 1);
    const bool complete_verifier_capture =
        state_snapshots != nullptr &&
        snapshot_stride_floats >= required_snapshot_stride &&
        max_snapshot_rows >= seq_len;

    /*
     * Verifier execution supplies one complete post-row state matrix. Use the
     * one-channel-owner kernel at every runtime M, including M=1, so every row
     * can be selected by device-side publication without changing arithmetic.
     */
    if (complete_verifier_capture) {
        int threads = 256;
        int blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_small_m_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output,
            initial_conv_state, updated_conv_state,
            nullptr,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
    } else if (seq_len == 1) {
        int threads = 256;
        int blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_decode_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output,
            initial_conv_state, updated_conv_state,
            channels, kernel_size, apply_silu);
    } else if (seq_len <= 4) {
        int threads = 256;
        int blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_small_m_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output,
            initial_conv_state, updated_conv_state,
            nullptr,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
    } else {
        int total = seq_len * channels;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_prefill_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output, initial_conv_state,
            nullptr,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
        int state_blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_state_update_kernel,
            dim3(state_blocks), dim3(threads), 0, (hipStream_t)stream,
            input, initial_conv_state, updated_conv_state, nullptr,
            seq_len, channels, kernel_size);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_short_conv1d] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_short_conv1d_effective(
    const float *input, const float *weight, const float *bias,
    float *output,
    const float *initial_conv_state,
    float *updated_conv_state,
    int seq_len, int channels, int kernel_size,
    bool apply_silu,
    const int *device_effective_seq_len,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    const int required_snapshot_stride = channels * (kernel_size - 1);
    const bool complete_verifier_capture =
        state_snapshots != nullptr &&
        snapshot_stride_floats >= required_snapshot_stride &&
        max_snapshot_rows >= seq_len;

    if (complete_verifier_capture) {
        int threads = 256;
        int blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_small_m_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output,
            initial_conv_state, updated_conv_state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
    } else if (seq_len == 1) {
        int threads = 256;
        int blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_decode_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output,
            initial_conv_state, updated_conv_state,
            channels, kernel_size, apply_silu);
    } else if (seq_len <= 4) {
        int threads = 256;
        int blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_small_m_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output,
            initial_conv_state, updated_conv_state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
    } else {
        int total = seq_len * channels;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_prefill_kernel,
            dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
            input, weight, bias, output, initial_conv_state,
            device_effective_seq_len,
            state_snapshots,
            snapshot_stride_floats,
            max_snapshot_rows,
            seq_len, channels, kernel_size, apply_silu);
        int state_blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(rocm_short_conv1d_state_update_kernel,
            dim3(state_blocks), dim3(threads), 0, (hipStream_t)stream,
            input, initial_conv_state, updated_conv_state,
            device_effective_seq_len,
            seq_len, channels, kernel_size);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_short_conv1d_effective] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

/**
 * @brief Launch native grouped request-batched short convolution on HIP.
 *
 * Tiny groups use one request-by-channel launch. Longer prefill uses one
 * flattened output launch and one grouped terminal-state commit, preserving
 * the production scalar arithmetic while avoiding a host request loop.
 */
bool rocmGDN_short_conv1d_batched(
    const float *input, const float *weight, const float *bias,
    float *output,
    const float *initial_request_states,
    float *updated_request_states,
    int request_count, int request_seq_len,
    int channels, int kernel_size,
    bool apply_silu,
    const int *device_request_seq_lens,
    float *state_snapshots,
    int snapshot_stride_floats,
    int max_snapshot_rows,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);
    if (!input || !weight || !output ||
        !initial_request_states || !updated_request_states ||
        !device_request_seq_lens || !stream ||
        request_count <= 0 || request_seq_len <= 0 ||
        channels <= 0 || kernel_size <= 1)
    {
        return false;
    }

    constexpr int threads = 256;
    const int required_snapshot_stride = channels * (kernel_size - 1);
    const bool complete_verifier_capture =
        state_snapshots != nullptr &&
        snapshot_stride_floats >= required_snapshot_stride &&
        max_snapshot_rows >= request_count * request_seq_len;
    if (request_seq_len <= 4 || complete_verifier_capture)
    {
        const int channel_blocks = (channels + threads - 1) / threads;
        hipLaunchKernelGGL(
            rocm_short_conv1d_batched_small_m_kernel,
            dim3(channel_blocks, request_count),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            input, weight, bias, output,
            initial_request_states, updated_request_states,
            device_request_seq_lens,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            request_count, request_seq_len,
            channels, kernel_size, apply_silu);
    }
    else
    {
        const int output_elements = request_count * request_seq_len * channels;
        const int output_blocks = (output_elements + threads - 1) / threads;
        hipLaunchKernelGGL(
            rocm_short_conv1d_batched_prefill_kernel,
            dim3(output_blocks),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            input, weight, bias, output, initial_request_states,
            device_request_seq_lens,
            state_snapshots, snapshot_stride_floats, max_snapshot_rows,
            request_count, request_seq_len,
            channels, kernel_size, apply_silu);

        const int state_elements = request_count * channels;
        const int state_blocks = (state_elements + threads - 1) / threads;
        hipLaunchKernelGGL(
            rocm_short_conv1d_batched_state_update_kernel,
            dim3(state_blocks),
            dim3(threads),
            0,
            static_cast<hipStream_t>(stream),
            input, initial_request_states, updated_request_states,
            device_request_seq_lens,
            request_count, request_seq_len, channels, kernel_size);
    }

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess)
    {
        fprintf(stderr,
                "[rocmGDN_short_conv1d_batched] %s\n",
                hipGetErrorString(error));
        return false;
    }
    return true;
}

bool rocmGDN_gated_rmsnorm(
    const float *input, const float *gate, const float *gamma,
    float *output,
    int seq_len, int d_model, int norm_dim, int gamma_period,
    float eps, bool subtract_one, bool gate_silu,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int n_groups = d_model / norm_dim;
    int total_work = seq_len * n_groups;

    // Cooperative launch: one block per (seq_pos, group) pair.
    // Threads within a block cooperate on the norm_dim reduction.
    // Use min(norm_dim, 256) threads — for norm_dim=128, use 128 threads.
    int threads = (norm_dim <= 64) ? 64 : ((norm_dim <= 128) ? 128 : 256);
    int blocks = total_work;

    hipLaunchKernelGGL(rocm_gated_rmsnorm_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        input, gate, gamma, output,
        total_work, n_groups, norm_dim, d_model, gamma_period,
        eps, subtract_one, gate_silu);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_gated_rmsnorm] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

bool rocmGDN_attention_output_gate(
    const float *input, const float *gate, float *output,
    int size,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_attention_output_gate_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        input, gate, output, size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_attention_output_gate] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C" — end of first block

// =========================================================================
// Q+Gate Split kernel
// =========================================================================
namespace {
__global__ void rocm_q_gate_split_kernel(
    const float *__restrict__ input,
    float *__restrict__ output_q,
    float *__restrict__ output_gate,
    int total_elements, int n_heads, int head_dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    int q_dim = n_heads * head_dim;
    int t = idx / q_dim;
    int offset = idx % q_dim;
    int h = offset / head_dim;
    int d = offset % head_dim;

    int input_dim = n_heads * head_dim * 2;
    int src_idx = t * input_dim + h * (head_dim * 2) + d;
    int gate_src_idx = t * input_dim + h * (head_dim * 2) + head_dim + d;

    output_q[idx] = input[src_idx];
    output_gate[idx] = input[gate_src_idx];
}
} // anonymous namespace (rocm_q_gate_split_kernel)

extern "C" {

bool rocmGDN_q_gate_split(
    const float *input, float *output_q, float *output_gate,
    int seq_len, int n_heads, int head_dim,
    int device_idx, void *stream)
{
    llaminar2::HipDeviceGuard::setDevice(device_idx);

    int total = seq_len * n_heads * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    hipLaunchKernelGGL(rocm_q_gate_split_kernel,
        dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
        input, output_q, output_gate, total, n_heads, head_dim);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[rocmGDN_q_gate_split] %s\n", hipGetErrorString(err));
        return false;
    }
    return true;
}

} // extern "C"
