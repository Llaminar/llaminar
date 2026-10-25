/**
 * @file ROCmGatedDeltaNetKernels.hip
 * @brief HIP kernels for Gated Delta Net (GDN) recurrence and short convolution
 *
 * ROCm/HIP port of the CUDA GDN kernels. Implements:
 * - Delta rule recurrence (decode + prefill), with Q/K epsilon inside squared norm
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
 * The bounded recurrence uses128 threads and32 value columns per CTA while
 * retaining four K-row splits. Its two physical wave64 groups preserve the
 * historical four-wave normalization fold through explicit zero-only leaves.
 * This is native geometry only: global state/snapshot extents do not change.
 *
 * Key HIP differences from CUDA:
 * - 64-thread wavefronts (vs 32 warps), shuffle width adjusted
 * - hip_runtime.h / hip_fp16.h headers
 * - __expf → expf (HIP math)
 */

#include "../../common/GatedNormActivation.h"
#include "kernels/rocm/gdn/ROCmGatedNormSingleQ8.h"
#include "../ops/ROCmSwiGLUDecodeEquivalentMath.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include "../../../utils/DebugEnv.h"
#include "kernels/gdn/GDNDeinterleaveRows.h"
#include "kernels/gdn/GDNMergedRecurrence.h"
#include "kernels/gdn/GDNPreparedQK.h"
#include "kernels/gdn/GDNPlanarRecurrence.h"

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

/**
 * @brief Dispatch the admitted value-major recurrence without converting state.
 * @param inputs Borrowed merged operands and immutable state-layout authority.
 * @param stream Exact non-default caller stream retained through graph retirement.
 * @return Whether validation and native submission succeed.
 */
bool enqueueGdnValueMajorMerged(
    const llaminar2::gdn::MergedRecurrenceInput& inputs, void* stream);

// Four K-row partitions each retain32 independent value columns. This doubles
// CTA coverage for small local head sets without repartitioning a column's K
// arithmetic or introducing cross-CTA reduction/state ownership.
constexpr int kGdnRecurrentThreads = 128;
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
 * Q and K normalization share publication edges, while their independent
 * square/shuffle/four-wave-fold trees remain unchanged. K's four wave totals
 * borrow the first reduce_kv words before that array's recurrent phase; the
 * scale-publication barrier retires those reads. This removes three CTA waits
 * per live row without adding LDS, workspace, a launch, or cross-CTA ordering.
 *
 * The four-way key-row split, wave64 normalization reduction, gate expressions,
 * K-part reduction order, and state update expressions are deliberately the
 * same as scalar HIP decode. Register retention changes storage traffic, not
 * arithmetic order. The all-M captured publication sweep proves every grouped
 * output, selected snapshot, and continuation byte against serial M=1 decode.
 *
 * Grid X flattens request/head identity and grid Y owns independent32-column
 * tiles. Each physical wave holds two K splits; recurrence reductions still
 * read split0,1,2,3 from LDS in the original order. Q/K normalization retains
 * original lanes0..127 and their wave64 XOR trees. The two absent historical
 * waves contained only +0 leaves at the admitted DK64/128, so their exact +0
 * totals are explicitly stored before the existing publication barrier and
 * the ascending four-wave scalar fold remains intact. No wait is added.
 *
 * Snapshot rows use the flattened request-row namespace, so the launch
 * publishes every possible accepted state without host replay.
 *
 * @tparam D_K Compile-time key width. Supported graph geometries select a
 *             specialization before capture so register extent and occupancy
 *             are fixed properties of the captured launch.
 */
/** @brief Compile-time physical input coordinates; arithmetic policy is independent. */
enum class GDNRecurrentInputLayout { ExpandedPlanar, MergedPostConvolution, PreparedCompactQK };
/**
 * @brief Compile-time canonical physical address; no arithmetic/reduction policy.
 * @tparam Layout One immutable state-bank format.
 * @param key Original logical key coordinate.
 * @param value Original logical value coordinate.
 * @param dk Original key width.
 * @param dv Original value width.
 * @return Exact original INT32 bounded per-head coordinate in selected format.
 */
template<llaminar2::GDNRecurrenceStateLayout Layout>
__device__ __forceinline__ int gdnRecurrenceStateOffset(int key,int value,int dk,int dv)
{
    static_assert(llaminar2::validGDNRecurrenceStateLayout(Layout));
    if constexpr(Layout==llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous)return key*dv+value;
    else return value*dk+key;
}

/**
 * @brief One original raw recurrence body over explicit planar or merged coordinates.
 * @tparam D_K Original64/128 key width.
 * @tparam Layout Explicit physical planar or merged input coordinates.
 * @tparam StateLayout Canonical immutable physical format; only addresses differ.
 * @param q,k,v Original planar inputs, or merged QKV in q for explicit merged layout.
 * @param alpha,beta_raw Original projected gates without preparation or changed rounding.
 * @param A_log,dt_bias Original local-head gate parameters.
 * @param output Original physical row-major output plane.
 * @param initial_state,updated_state Borrowed initial and exclusive terminal state banks.
 * @param request_count,request_seq_len Original request-major physical row geometry.
 * @param n_heads,d_v Original local head and value width dimensions.
 * @param use_qk_l2norm Original raw normalization policy.
 * @param effective_seq_len_ptr Borrowed per-request device live lengths, or fully active.
 * @param effective_row_idx Original row offset for a serial diagnostic slice.
 * @param state_snapshots Optional exclusive physical snapshot bank.
 * @param snapshot_stride_floats Original per-row snapshot stride.
 * @param max_snapshot_rows Admitted physical snapshot row capacity.
 * @param merged_key_heads Merged source head count; unused for planar layout.
 * @param global_value_offset Signed modular head coordinate; unused for planar layout.
 * @details Both physical layouts retain literal original raw gate/normalization,
 * state arithmetic and snapshot/publication ordering.
 */
template <int D_K,
          GDNRecurrentInputLayout Layout = GDNRecurrentInputLayout::ExpandedPlanar,
          llaminar2::GDNRecurrenceStateLayout StateLayout = llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous>
__device__ __forceinline__
void rocm_gdn_recurrent_step_body(
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
    int max_snapshot_rows,
    int merged_key_heads = 0, int global_value_offset = 0)
{
    const int request_head = blockIdx.x;
    const int request = request_head / n_heads;
    const int h = request_head - request * n_heads;
    if (request >= request_count || h >= n_heads) return;

    // The missing historical norm waves are zero only for these closed widths.
    static_assert(D_K == 64 || D_K == 128);
    static_assert(D_K % kGdnRecurrentRowSplit == 0);
    constexpr int kRowsPerSplit = D_K / kGdnRecurrentRowSplit;
    constexpr int kRegisterRowsPerSplit =
        kRowsPerSplit < 22 ? kRowsPerSplit : 22;
    constexpr int kLdsRowsPerSplit =
        kRowsPerSplit - kRegisterRowsPerSplit;

    const int tid = threadIdx.x;
    // Original128-thread column/split mapping is identical for both layouts.
    static_assert(kGdnRecurrentRowSplit == 4);
    static_assert(64 % kGdnRecurrentRowSplit == 0);
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
    constexpr int kPhysicalWaveCount = kGdnRecurrentThreads / 64;
    static_assert(kPhysicalWaveCount == 2);
    constexpr int kWaveCount = 4; // Arithmetic leaves, including two explicit zero waves.

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
            state_rows[local_row] = S_initial[gdnRecurrenceStateOffset<StateLayout>(state_row,vi,D_K,d_v)];
        }
#pragma clang loop unroll(disable)
        for (int tail_row = 0; tail_row < kLdsRowsPerSplit; ++tail_row)
        {
            const int state_row =
                row_begin + kRegisterRowsPerSplit + tail_row;
            state_tail[tail_row * kGdnRecurrentThreads + tid] =
                S_initial[gdnRecurrenceStateOffset<StateLayout>(state_row,vi,D_K,d_v)];
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

    // Merged coordinates are invariant across this request's live rows.
    int merged_source_head = 0;
    std::size_t merged_key_span = 0, merged_row_stride = 0;
    if constexpr (Layout == GDNRecurrentInputLayout::MergedPostConvolution)
    {
        merged_source_head = static_cast<int>((std::int64_t{h} + global_value_offset) % merged_key_heads);
        if (merged_source_head < 0) merged_source_head += merged_key_heads;
        merged_key_span = std::size_t(merged_key_heads) * D_K;
        merged_row_stride = 2 * merged_key_span + row_v_stride;
    }
    for (int request_row = 0; request_row < request_seq_len; ++request_row)
    {
        const int flat_row = request * request_seq_len + request_row;
        const float *q_head, *k_head, *v_head;
        if constexpr (Layout == GDNRecurrentInputLayout::MergedPostConvolution)
        {

            // Match gdnDeinterleaveLiveRows positive modulo using64-bit addition.
            // Admission reuses its exact INT32 physical extent proof. Values
            // retain original request/row/head coordinates, including holes.
            const auto source_head = merged_source_head;
            const auto key_span = merged_key_span;
            const auto merged_stride = merged_row_stride;
            const float *source = q + std::size_t(flat_row) * merged_stride;
            q_head = source + std::size_t(source_head) * D_K;
            k_head = source + key_span + std::size_t(source_head) * D_K;
            v_head = source + 2 * key_span + std::size_t(h) * d_v;
        }
        else
        {
        q_head =
            q + static_cast<size_t>(flat_row) * row_qk_stride +
            static_cast<size_t>(h) * D_K;
        k_head =
            k + static_cast<size_t>(flat_row) * row_qk_stride +
            static_cast<size_t>(h) * D_K;
        v_head =
            v + static_cast<size_t>(flat_row) * row_v_stride +
            static_cast<size_t>(h) * d_v;
        }
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


        // Original norm lanes0..127 retain their input and physical wave64 rank.
        // The two absent historical waves are published as exact +0 below.
        for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
        {
            q_local[i] = q_head[i];
            k_local[i] = k_head[i];
        }

        __syncthreads();


        if (use_qk_l2norm)
        {
            // Q/K are independent. Keep each original wave64 sum and ascending
            // four-wave fold, but publish them together. reduce_kv is idle until
            // the later decay/K^T S pass, so its first four words can own K sums.
            // The final scale-publication barrier retires every K-sum reader
            // before recurrence may overwrite that same storage.
            float *k_wave_sums = reduce_kv;
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
            {
                q_sum += q_local[i] * q_local[i];
                k_sum += k_local[i] * k_local[i];
            }
            for (int offset = 32; offset > 0; offset >>= 1)
            {
                q_sum += __shfl_xor(q_sum, offset);
                k_sum += __shfl_xor(k_sum, offset);
            }
            if (lane_id == 0)
            {
                wave_sums[warp_id] = q_sum;
                k_wave_sums[warp_id] = k_sum;
                // Historical waves2/3 began at +0 and processed no DK64/128
                // elements. Every XOR add therefore returned +0. These words
                // must be refreshed for every live row before the four-leaf
                // fold; they may contain prior gate/reduction data on replay.
                wave_sums[warp_id + kPhysicalWaveCount] = 0.0f;
                k_wave_sums[warp_id + kPhysicalWaveCount] = 0.0f;
            }
            __syncthreads();
            if (tid == 0)
            {
                float q_total = 0.0f;
                float k_total = 0.0f;
                for (int wave = 0; wave < kWaveCount; ++wave)
                {
                    q_total += wave_sums[wave];
                    k_total += k_wave_sums[wave];
                }
                wave_sums[0] = q_total;
                k_wave_sums[0] = k_total;
            }
            __syncthreads();
            // Preserve the old quotient/sqrt expressions and each operand's
            // multiplication boundary. Do not combine scale with the square
            // root or replace either expression with reciprocal sqrt.
            const float q_inv = scale / sqrtf(wave_sums[0] + 1e-6f);
            const float k_inv = 1.0f / sqrtf(k_wave_sums[0] + 1e-6f);
            for (int i = tid; i < D_K; i += kGdnRecurrentThreads)
            {
                q_local[i] *= q_inv;
                k_local[i] *= k_inv;
            }
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
                const int qk_row = state_row;
                state_rows[local_row] *= decay;
                partial_kv += state_rows[local_row] * k_local[qk_row];
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
                const int qk_row = state_row;
                const float s_decayed = state_tail[lds_index] * decay;
                state_tail[lds_index] = s_decayed;
                partial_kv += s_decayed * k_local[qk_row];
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
                const int qk_row = state_row;
                const float s_new =
                    state_rows[local_row] +
                    k_local[qk_row] * delta;
                state_rows[local_row] = s_new;
                if (snapshot)
                    snapshot[gdnRecurrenceStateOffset<StateLayout>(state_row,vi,D_K,d_v)] = s_new;
                partial_out += s_new * q_local[qk_row];
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
                const int qk_row = state_row;
                const float s_new =
                    state_tail[lds_index] +
                    k_local[qk_row] * delta;
                state_tail[lds_index] = s_new;
                if (snapshot)
                    snapshot[gdnRecurrenceStateOffset<StateLayout>(state_row,vi,D_K,d_v)] = s_new;
                partial_out += s_new * q_local[qk_row];
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
        // Prepared waves may finish at different times: retain this CTA edge
        // before the next row overwrites Q/K and gate words still being read.
        // State-tail words remain exclusively lane-owned for the entire CTA.
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
            S_updated[gdnRecurrenceStateOffset<StateLayout>(state_row,vi,D_K,d_v)] = state_rows[local_row];
        }
#pragma clang loop unroll(disable)
        for (int tail_row = 0; tail_row < kLdsRowsPerSplit; ++tail_row)
        {
            const int state_row =
                row_begin + kRegisterRowsPerSplit + tail_row;
            S_updated[gdnRecurrenceStateOffset<StateLayout>(state_row,vi,D_K,d_v)] =
                state_tail[tail_row * kGdnRecurrentThreads + tid];
        }
    }
}
/** @brief Enqueue-equivalent kernel boundary over the single literal recurrence body.
 * @tparam D_K Admitted64/128 key width. The original raw symbol remains unchanged.
 * @details All pointer/geometry arguments retain the body's documented lifetime;
 * merged entry changes only physical input addresses, never arithmetic.
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
    rocm_gdn_recurrent_step_body<D_K>(q, k, v, alpha, beta_raw, A_log, dt_bias, output,
        initial_state, updated_state, request_count, request_seq_len,
        n_heads, d_v, use_qk_l2norm, effective_seq_len_ptr, effective_row_idx,
        state_snapshots, snapshot_stride_floats, max_snapshot_rows);
}

/**
 * @brief Explicit merged-post-convolution entry, with original raw arithmetic.
 * @tparam D_K Admitted key width64/128.
 * @details First pointer names merged rows; unused K/V slots are null. Remaining
 * original arguments retain raw recurrence order, lifetime and state semantics.
 * Final scalars name source key-head count and signed global modular offset.
 */
template <int D_K>
__global__ __launch_bounds__(kGdnRecurrentThreads, 2)
void rocm_gdn_merged_recurrent_step_kernel(
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
    int max_snapshot_rows,
    int merged_key_heads, int global_value_offset)
{
    rocm_gdn_recurrent_step_body<D_K, GDNRecurrentInputLayout::MergedPostConvolution>(q, k, v, alpha, beta_raw, A_log, dt_bias, output,
        initial_state, updated_state, request_count, request_seq_len,
        n_heads, d_v, use_qk_l2norm, effective_seq_len_ptr, effective_row_idx,
        state_snapshots, snapshot_stride_floats, max_snapshot_rows, merged_key_heads, global_value_offset);
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
                        scale / sqrtf(q_total + 1e-6f);
                    gate_beta[2 * request_seq_len + t] =
                        1.0f / sqrtf(k_total + 1e-6f);
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
                    scale / sqrtf(warp_sums[0] + 1e-6f);
                const float k_inv =
                    1.0f / sqrtf(warp_sums[8] + 1e-6f);
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

/**
 * @brief Publish original four-tap convolution and exact compact normalized Q/K.
 * @tparam D_K Original key width64/128; every block retains128 normalization lanes.
 * @details One channel tile owns its original state and raw output. Normalization
 * uses the old two wave64 trees and four ascending leaves, including explicit
 * zeros. The scalar production caller admits exactly one row; retaining the
 * historical terminal/snapshot stores preserves accepted state publication.
 * The compact arena root is disjoint from every original raw/state owner.
 */
template <int D_K>
__global__ __launch_bounds__(128, 2)
void rocm_gdn_scalar_prepared_qk_conv_kernel(
    const float* input, const float* weight, const float* bias,
    const float* initial_state, float* raw, float* updated_state,
    float* snapshots, float* prepared_q, float* prepared_k,
    int rows, int key_heads, int value_heads, int value_width,
    const int* live_rows, int snapshot_stride, int max_snapshot_rows,
    bool normalize_qk)
{
    const int tid = threadIdx.x;
    const int channel = blockIdx.x * D_K + tid;
    const int key_span = key_heads * D_K;
    const int channels = 2 * key_span + value_heads * value_width;
    const bool valid = tid < D_K && channel < channels;
    const bool is_q = blockIdx.x < key_heads;
    const bool is_k = blockIdx.x >= key_heads && blockIdx.x < 2 * key_heads;
    const bool normalized_tile = is_q || is_k;
    const int original_head = is_q ? blockIdx.x : blockIdx.x - key_heads;
    __shared__ float partial[4];
    int effective_rows = rows;
    if (live_rows)
    {
        const int count = *live_rows;
        // Match the original scalar/small-M convolution contract, including
        // its minimum-one row. Recurrence independently owns its true live count.
        effective_rows = count < 1 ? 1 : (count > rows ? rows : count);
    }
    const bool terminal_is_snapshot = snapshots && max_snapshot_rows >= rows;
    const float scale = rsqrtf(static_cast<float>(D_K));
    for (int row = 0; row < rows; ++row)
    {
        float sum = 0.0f;
        if (valid && row < effective_rows)
        {
            // Preserve the old ascending four products from positive zero;
            // neither a reassociated four-term expression nor a fused tree.
            for (int tap = 0; tap < 4; ++tap)
            {
                const int source_row = row - 3 + tap;
                const float value = source_row >= 0
                    ? input[std::size_t(source_row) * channels + channel]
                    : initial_state[std::size_t(channel) * 3 + 3 + source_row];
                sum += value * weight[std::size_t(channel) * 4 + tap];
            }
            if (bias) sum += bias[channel];
            sum = sum / (1.0f + expf(-sum));
        }
        if (valid)
        {
            raw[std::size_t(row) * channels + channel] = sum;
            if (snapshots && row < effective_rows && row < max_snapshot_rows)
            {
                for (int index = 0; index < 3; ++index)
                {
                    const int source_row = row - 2 + index;
                    snapshots[std::size_t(row) * snapshot_stride + std::size_t(channel) * 3 + index] =
                        source_row >= 0
                            ? input[std::size_t(source_row) * channels + channel]
                            : initial_state[std::size_t(channel) * 3 + 3 + source_row];
                }
            }
        }
        // The branch is block-uniform; inactive tail lanes contribute +0 to
        // the historical norm tree and participate in every required barrier.
        if (normalized_tile)
        {
            float prepared = sum;
            if (normalize_qk)
            {
#pragma clang fp contract(off)
                float squared_sum = 0.0f;
                if (tid < D_K) squared_sum += sum * sum;
                for (int offset = 32; offset > 0; offset >>= 1)
                    squared_sum += __shfl_xor(squared_sum, offset);
                if (tid % 64 == 0)
                {
                    partial[tid / 64] = squared_sum;
                    partial[tid / 64 + 2] = 0.0f;
                }
                __syncthreads();
                if (tid == 0)
                {
                    float total = 0.0f;
                    for (int leaf = 0; leaf < 4; ++leaf) total += partial[leaf];
                    partial[0] = total;
                }
                __syncthreads();
                const float inverse = is_q
                    ? scale / sqrtf(partial[0] + 1e-6f)
                    : 1.0f / sqrtf(partial[0] + 1e-6f);
                prepared *= inverse;
            }
            else if (is_q)
                prepared *= scale;
            if (tid < D_K)
            {
                float* destination = is_q ? prepared_q : prepared_k;
                destination[std::size_t(row) * key_span + original_head * D_K + tid] = prepared;
            }
            // Retire all readers before the next row reuses norm partials.
            __syncthreads();
        }
    }
    if (valid && !terminal_is_snapshot)
    {
        for (int index = 0; index < 3; ++index)
        {
            const int source_row = effective_rows - 3 + index;
            updated_state[std::size_t(channel) * 3 + index] = source_row >= 0
                ? input[std::size_t(source_row) * channels + channel]
                : initial_state[std::size_t(channel) * 3 + 3 + source_row];
        }
    }
}

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

/** @brief Normalize one immutable group without bounded private input arrays.
 * @details Cache the first value in a scalar register, covering native128-wide
 * heads completely. Larger groups reread later coalesced elements after the
 * identical reduction, retaining arithmetic while removing a hidden width cap.
 */
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
    llaminar2::GatedNormActivation gate_activation)
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
    float first_value = 0.0f;

    float sum_sq = 0.0f;
    for (int j = tid; j < norm_dim; j += block_size) {
        float v = input[offset + j];
        if(j==tid) first_value=v;
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
    for (int j = tid; j < norm_dim; j += block_size) {
        float normalized = (j==tid?first_value:input[offset+j]) * inv_rms;
        int gamma_idx = j % gamma_period;
        float gamma_eff = subtract_one ? (1.0f + gamma[gamma_idx]) : gamma[gamma_idx];
        float gate_val = gate[offset + j];
        float gate_act;
        if (gate_activation == llaminar2::GatedNormActivation::SiLU)
            gate_act = gate_val / (1.0f + expf(-gate_val));
        else if (gate_activation == llaminar2::GatedNormActivation::Sigmoid)
            gate_act = 1.0f / (1.0f + expf(-gate_val));
        else
            gate_act = gate_val;
        output[offset + j] = normalized * gamma_eff * gate_act;
    }
}

/** @brief Internal borrowed destination; the prepared invocation retains its owner. */
struct GatedNormSingleQ8Destination { std::int8_t* bytes;float* scales; };
/**
 * @brief Retain the green aligned128 norm/Q8 numeric program without reassociation.
 * @details The function body is the retained rows/sums prototype verbatim apart
 * from its private ABI names. Only<true,false,false> is instantiated here:
 * ordinaryM1 owns full FP32/Q8/scales; sums remain untouched as in the original
 * scalar quantizer/consumer. Other row programs keep their existing kernels.
 */
template <bool Fixed128, bool Masked, bool WithSums>
__device__ __forceinline__ void
rocm_gated_norm_single_q8_production_body(const float *input,
                               const float *__restrict__ gate,
                               const float *__restrict__ gamma,
                               float *output,
                               int total_work,
                               int n_groups,
                               int runtime_norm_dim,
                               int d_model,
                               int runtime_gamma_period,
                               float eps,
                               bool subtract_one,
                               llaminar2::GatedNormActivation gate_activation,
                               GatedNormSingleQ8Destination destination,
                               bool quantize_row,
                               std::int32_t *sums)
{
    // One block per (seq_position, group) pair
    const int work_idx = blockIdx.x;
    if (work_idx >= total_work)
        return;

    const int norm_dim = Fixed128 ? 128 : runtime_norm_dim;
    const int gamma_period = Fixed128 ? 128 : runtime_gamma_period;
    int offset;
    if constexpr (Fixed128)
    {
        // For divisible rows, flattened group ownership equals the original
        // row/group mapping at every M. No query/context threshold is involved.
        offset = work_idx * 128;
    }
    else
    {
        const int t = work_idx / n_groups;
        const int g = work_idx % n_groups;
        offset = t * d_model + g * norm_dim;
    }
    const int tid = threadIdx.x;
    const int block_size = Fixed128 ? 128 : blockDim.x;

    // Pass 1: Each thread handles its slice of norm_dim, accumulate sum_sq
    float first_value = 0.0f;

    float sum_sq = 0.0f;
    for (int j = tid; j < norm_dim; j += block_size)
    {
        float v = input[offset + j];
        if (j == tid)
            first_value = v;
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
    if (lane == 0)
        warp_sums[warp_id] = sum_sq;
    __syncthreads();

    if (tid == 0)
    {
        float total = 0.0f;
        for (int w = 0; w < num_warps; w++)
            total += warp_sums[w];
        warp_sums[0] = total;
    }
    __syncthreads();

    float inv_rms = rsqrtf(warp_sums[0] / (float)runtime_norm_dim + eps);

    // Pass 2: Normalize with register-cached values
    for (int j = tid; j < norm_dim; j += block_size)
    {
        float normalized = (j == tid ? first_value : input[offset + j]) * inv_rms;
        int gamma_idx = j % gamma_period;
        float gamma_eff = subtract_one ? (1.0f + gamma[gamma_idx]) : gamma[gamma_idx];
        float gate_val = gate[offset + j];
        float gate_act;
        if (gate_activation == llaminar2::GatedNormActivation::SiLU)
            gate_act = gate_val / (1.0f + expf(-gate_val));
        else if (gate_activation == llaminar2::GatedNormActivation::Sigmoid)
            gate_act = 1.0f / (1.0f + expf(-gate_val));
        else
            gate_act = gate_val;
        // The original final FP32 value remains an observable full tensor store.
        // FP32 register arithmetic has the same final rounding point; byte gates
        // certify its use here rather than assuming expression reuse is proof.
        const float value = normalized * gamma_eff * gate_act;
        output[offset + j] = value;
        if constexpr (Masked)
        {
            // This predicate cannot skip physical normalization or its store.
            // Original request holes own only quantized publication extents.
            if (!quantize_row)
                continue;
        }
        float maximum = fabsf(value);
// Four independent32-value blocks per128-thread norm CTA. Width32
// prevents adjacent logical blocks sharing a physical wave64 maximum.
#pragma unroll
        for (int shift = 16; shift > 0; shift >>= 1)
            maximum = fmaxf(maximum, __shfl_down(maximum, shift, 32));
        maximum = __shfl(maximum, 0, 32);
        const float scale = llaminar2::rocm::decode_equivalent::q8Scale(maximum);
        const float inverse = 1.0f / scale;
        const int element = offset + j;
        if ((tid & 31) == 0)
            destination.scales[element / 32] = scale;
        if constexpr (WithSums)
        {
            const int q = llaminar2::rocm::decode_equivalent::quantizeQ8(value, inverse);
            destination.bytes[element] = static_cast<int8_t>(q);
            int sum = q;
#pragma unroll
            for (int shift = 16; shift > 0; shift >>= 1)
                sum += __shfl_down(sum, shift, 32);
            if ((tid & 31) == 0)
                sums[element / 32] = sum;
        }
        else
        {
            destination.bytes[element] =
                static_cast<int8_t>(llaminar2::rocm::decode_equivalent::quantizeQ8(value, inverse));
        }
    }
}
/** @brief One original head per CTA; persistent scratch belongs to the caller. */
__global__ void rocm_gdn_gated_norm_single_q8_kernel(llaminar2::rocm::GatedNormSingleQ8Input input)
{
    const auto& p=input.parameters;
    rocm_gated_norm_single_q8_production_body<true,false,false>(input.input,input.gate,
        input.gamma,input.output,input.columns/128,input.columns/128,p.norm_dim,
        input.columns,p.gamma_period,p.eps,p.subtract_one,p.activation,
        {input.quant,input.scales},true,nullptr);
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
    float eps, bool subtract_one, llaminar2::GatedNormActivation gate_activation,
    int device_idx, void *stream)
{
        if (!stream || !llaminar2::validGatedNormActivation(gate_activation))
            return false; // Exact-stream and immutable-policy contracts are mandatory.

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
        eps, subtract_one, gate_activation);

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




namespace llaminar2::gdn
{
/**
 * @brief Submit original raw recurrence directly over admitted merged QKV rows.
 * @param p Frozen original row/shape/offset and borrowed immutable native inputs.
 * @param stream Exact caller-owned stream retained through graph retirement.
 * @return False on invalid admission or failed native enqueue; no layout copy.
 * @details Existing deinterleave extent validation proves original physical
 * indexing. No plane, gate, workspace, transfer, or graph topology is allocated.
 */
bool launchROCmMergedRecurrence(const MergedRecurrenceInput& p, void* stream)
{
    if(!validMergedRecurrenceStorage(p))return false;
    if(!validGDNRecurrenceStateLayout(p.recurrence_layout))return false;
    if(p.recurrence_layout==GDNRecurrenceStateLayout::ValueMajorKeyContiguous)return enqueueGdnValueMajorMerged(p,stream);
    // Match the original raw route: storage may retain a larger verifier
    // envelope or padded snapshot stride than this M1 invocation consumes.
    // Physical rows select stores; admitted capacity is never the work extent.
    if (reinterpret_cast<std::uintptr_t>(stream) <= 2 || !p.sourceOutputDisjoint() || !p.merged || !p.alpha || !p.beta_raw ||
        !p.A_log || !p.dt_bias || !p.output || !p.initial_state || !p.updated_state ||
        (p.key_width != 64 && p.key_width != 128) ||
        p.value_width <= 0 || p.rows.rowsPerRequest() > 16 ||
        gdnDeinterleaveElementsPerRequest(p.rows,p.key_heads,p.value_heads,p.key_width,p.value_width) <= 0 ||
        (p.rows.rowsPerRequest() > 1 && !p.snapshots) ||
        (p.snapshots && (std::int64_t(p.snapshot_stride_floats) < std::int64_t(p.value_heads) * p.key_width * p.value_width ||
                        p.max_snapshot_rows < p.rows.physicalRows()))) return false;
    // The original recurrent grid multiplies request_count and local heads.
    // Guard its INT32 coordinate before forming either the grid or state stride.
    if (std::int64_t(p.rows.requests()) * p.value_heads > INT_MAX ||
        std::int64_t(p.value_heads) * p.key_width * p.value_width > INT_MAX) return false;
    const auto hs=static_cast<hipStream_t>(stream);
    const dim3 grid(p.rows.requests()*p.value_heads,
                    1+(p.value_width-1)/kGdnRecurrentColumnsPerBlock);
    const int tail=std::max(0,p.key_width/kGdnRecurrentRowSplit-22);
    const std::size_t shared=std::size_t(2*p.key_width+4+2*kGdnRecurrentThreads+tail*kGdnRecurrentThreads)*sizeof(float);
#define MERGED(DK) hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_merged_recurrent_step_kernel<DK>), \
    grid,dim3(kGdnRecurrentThreads),shared,hs,p.merged,nullptr,nullptr,p.alpha,p.beta_raw,p.A_log,p.dt_bias, \
    p.output,p.initial_state,p.updated_state,p.rows.requests(),p.rows.rowsPerRequest(),p.value_heads,p.value_width, \
    p.use_qk_l2norm,p.rows.lengthOwner(),0,p.snapshots,p.snapshot_stride_floats,p.max_snapshot_rows,p.key_heads,p.global_value_offset)
    if(p.key_width==64){MERGED(64);}else{MERGED(128);}
#undef MERGED
    return hipGetLastError()==hipSuccess;
}
}


namespace
{

/** @brief Immutable physical endpoint access, independent of recurrence math. */
enum class GdnRecurrenceEndpointAccess { VectorAligned, ScalarAligned };
/**
 * @brief Load four logical state/source words with their admitted alignment.
 * @tparam Access Immutable source/state physical alignment program.
 * @param p Borrowed address of four valid consecutive FP32 words.
 * @return Original bits, with no floating arithmetic or implicit conversion.
 */
template<GdnRecurrenceEndpointAccess Access>
__device__ __forceinline__ float4 gdnRecurrenceLoadFour(const float* p)
{
    if constexpr(Access==GdnRecurrenceEndpointAccess::VectorAligned)return *reinterpret_cast<const float4*>(p);
    else return make_float4(p[0],p[1],p[2],p[3]);
}
/**
 * @brief Publish four original state words using admitted destination alignment.
 * @tparam Access Immutable state endpoint access program.
 * @param p Exclusive address of four valid consecutive FP32 words.
 * @param bits Original rounded resident state values.
 */
template<GdnRecurrenceEndpointAccess Access>
__device__ __forceinline__ void gdnRecurrenceStoreFour(float* p,float4 bits)
{
    if constexpr(Access==GdnRecurrenceEndpointAccess::VectorAligned)*reinterpret_cast<float4*>(p)=bits;
    else {p[0]=bits.x;p[1]=bits.y;p[2]=bits.z;p[3]=bits.w;}
}
/**
 * @brief Carry each original quarter's accumulator through consecutive key owners.
 * @tparam OwnedKeys Four/eight consecutive keys per lane at DK64/128.
 * @param state Original independently rounded resident state words.
 * @param operand Original normalized Q or K words for the same owned keys.
 * @param segment Lane's consecutive segment0..3 inside one original quarter.
 * @param quarter Original arithmetic quarter0..3 inside the16-lane column group.
 * @return Only segment3 carries the final original sequential quarter result.
 * @details Do not sum independently started eight-key partials. Each handoff
 * carries the previous exact accumulator before the next segment appends its
 * products, retaining original ascending key order and contraction expressions.
 */
template<int OwnedKeys>
__device__ __forceinline__ float gdnValueMajorOrderedQuarterDot(
    const float (&state)[OwnedKeys],const float (&operand)[OwnedKeys],int segment,int quarter)
{
    float accumulator=0.0f;
#pragma unroll
    for(int owner=0;owner<4;++owner)
    {
        if(segment==owner)
        {
#pragma unroll
            for(int i=0;i<OwnedKeys;++i)
                accumulator+=state[i]*operand[i];
        }
        if(owner<3)
            accumulator=__shfl(accumulator,quarter*4+owner,16);
    }
    return accumulator;
}
/** @brief Retain the original ascending0,1,2,3 quarter fold.
 * @param quarter_value Final accumulator in lane3 of each original quarter.
 * @return Exact fixed-order fold, replicated to every lane of one value column.
 */
__device__ __forceinline__ float gdnValueMajorOrderedQuarterFold(float quarter_value)
{
    float total=0.0f;
#pragma unroll
    for(int quarter=0;quarter<4;++quarter)
        total+=__shfl(quarter_value,quarter*4+3,16);
    return total;
}

/**
 * @brief Production value-major M1 recurrence with original fixed-order arithmetic.
 * @tparam D_K Original native key width64/128.
 * @tparam InputLayout Explicit expanded or merged original source coordinates.
 * @tparam Access Immutable vector/scalar endpoint alignment program.
 * @details Native22arg ABI matches merged recurrence. Initial, updated and
 * snapshots now have physical [request,head,value,key] state order. Sixteen
 * lanes own one column; each four-lane quartet is one original key quarter.
 * State, Q and K use contiguous float4 loads. The sequential handoff preserves
 * each original key dot instead of replacing it with a different shuffle tree.
 * Structural inspiration is Castagna-Veloce's value-major state and16-lane
 * column mapping (revision3483d462715d61f21ed5877836be35a724446359,
 * ggml/src/ggml-cuda/gated_delta_net.cu:410-563). Arithmetic here is independently
 * retained from Llaminar: its original four sequential key quarters and fixed
 * fold, not Castagna's accumulator/shuffle reduction or prepared-input policy.
 */
template<int D_K,GDNRecurrentInputLayout InputLayout,GdnRecurrenceEndpointAccess Access>
__global__ __launch_bounds__(512,2)
void rocm_gdn_value_major_scalar_kernel(
    const float *__restrict__ q,const float *__restrict__ k,const float *__restrict__ v,
    const float *__restrict__ alpha,const float *__restrict__ beta_raw,
    const float *__restrict__ A_log,const float *__restrict__ dt_bias,
    float *__restrict__ output,const float* initial_state,float* updated_state,
    int request_count,int request_seq_len,int n_heads,int d_v,bool use_qk_l2norm,
    const int *__restrict__ effective_seq_len_ptr,int effective_row_idx,
    float *__restrict__ state_snapshots,int snapshot_stride_floats,int max_snapshot_rows,
    int merged_key_heads,int global_value_offset)
{
    static_assert(D_K==64 || D_K==128);
    static_assert(InputLayout==GDNRecurrentInputLayout::ExpandedPlanar||
        InputLayout==GDNRecurrentInputLayout::MergedPostConvolution||InputLayout==GDNRecurrentInputLayout::PreparedCompactQK);
    constexpr int kOwnedKeys=D_K/16;
    const int tid=threadIdx.x,lane=tid%64,group_lane=tid%16;
    const int quarter=group_lane/4,segment=group_lane%4;
    const int request_head=blockIdx.x,request=request_head/n_heads;
    const int h=request_head-request*n_heads;
    if(request>=request_count || h>=n_heads)return;
    if(request_seq_len!=1){__builtin_trap();return;}
    const int vi=blockIdx.y*32+tid/16;
    // Only a complete physical wave may exit before the wave64 norm tree.
    // Later invalid16-lane column groups may exit after normalization converges.
    if(blockIdx.y*32+(tid/64)*4>=d_v)return;
    const auto values=std::size_t(n_heads)*d_v;
    const auto state_stride=values*D_K;
    const auto column_base=std::size_t(request)*state_stride+
        (std::size_t(h)*d_v+vi)*D_K;
    const int key_begin=group_lane*kOwnedKeys;
    const bool snapshotted=state_snapshots!=nullptr &&
        max_snapshot_rows>=request_count*request_seq_len;
    const bool live=!effective_seq_len_ptr ||
        effective_row_idx<effective_seq_len_ptr[request];
    if(!live)
    {
        if(vi<d_v)
        {
            if(group_lane==0)output[std::size_t(request)*values+std::size_t(h)*d_v+vi]=0.0f;
            if(!snapshotted)
            {
#pragma unroll
                for(int pack=0;pack<kOwnedKeys/4;++pack)
                {
                    const auto index=column_base+key_begin+pack*4;
                    const auto bits=gdnRecurrenceLoadFour<Access>(initial_state+index);
                    gdnRecurrenceStoreFour<Access>(updated_state+index,bits);
                }
            }
        }
        return;
    }
    int source_head=h;
    if constexpr(InputLayout==GDNRecurrentInputLayout::MergedPostConvolution||InputLayout==GDNRecurrentInputLayout::PreparedCompactQK)
    {
        source_head=static_cast<int>((std::int64_t{h}+global_value_offset)%merged_key_heads);
        if(source_head<0)source_head+=merged_key_heads;
    }
    const auto key_span=std::size_t(merged_key_heads)*D_K;
    const auto raw_stride=2*key_span+values;
    const float* raw=nullptr;
    if constexpr(InputLayout==GDNRecurrentInputLayout::MergedPostConvolution||InputLayout==GDNRecurrentInputLayout::PreparedCompactQK)raw=q+std::size_t(request)*raw_stride;
    const float* q_head=nullptr;const float* k_head=nullptr;
    if constexpr(InputLayout==GDNRecurrentInputLayout::PreparedCompactQK)
    {q_head=k+std::size_t(source_head)*D_K;k_head=v+std::size_t(source_head)*D_K;}
    else if constexpr(InputLayout==GDNRecurrentInputLayout::MergedPostConvolution)
    {q_head=raw+std::size_t(source_head)*D_K;k_head=raw+key_span+std::size_t(source_head)*D_K;}
    else
    {q_head=q+(std::size_t(request)*n_heads+h)*D_K;k_head=k+(std::size_t(request)*n_heads+h)*D_K;}
    float q_inv=1.0f,k_inv=1.0f;
    if constexpr(InputLayout!=GDNRecurrentInputLayout::PreparedCompactQK)
    {
    const float scale=rsqrtf(float(D_K));
    q_inv=scale;
    if(use_qk_l2norm)
    {
#pragma clang fp contract(off)
        // Existing CACHE_QK_NORMALIZATION arithmetic: distinct64-key trees,
        // rounded squares before XOR adds, then lane0's ascending four leaves.
        const float q0=q_head[lane],k0=k_head[lane];
        const float q1=D_K==128?q_head[lane+64]:0.0f;
        const float k1=D_K==128?k_head[lane+64]:0.0f;
        float q_sum0=q0*q0,q_sum1=q1*q1,k_sum0=k0*k0,k_sum1=k1*k1;
        for(int offset=32;offset>0;offset>>=1)
        {
            q_sum0+=__shfl_xor(q_sum0,offset);
            q_sum1+=__shfl_xor(q_sum1,offset);
            k_sum0+=__shfl_xor(k_sum0,offset);
            k_sum1+=__shfl_xor(k_sum1,offset);
        }
        if(lane==0)
        {
            float q_total=0.0f,k_total=0.0f;
            q_total+=q_sum0;q_total+=q_sum1;q_total+=0.0f;q_total+=0.0f;
            k_total+=k_sum0;k_total+=k_sum1;k_total+=0.0f;k_total+=0.0f;
            q_inv=scale/sqrtf(q_total+1e-6f);
            k_inv=1.0f/sqrtf(k_total+1e-6f);
        }
        q_inv=__shfl(q_inv,0,64);k_inv=__shfl(k_inv,0,64);
    }
    }
    float decay=0.0f,beta_h=0.0f;
    if(lane==0)
    {
        const auto gate_index=std::size_t(request)*n_heads+h;
        const float x=alpha[gate_index]+dt_bias[h];
        const float sp=x>20.0f?x:log1pf(expf(x));
        decay=expf(A_log[h]*sp);
        beta_h=1.0f/(1.0f+expf(-beta_raw[gate_index]));
    }
    decay=__shfl(decay,0,64);beta_h=__shfl(beta_h,0,64);
    if(vi>=d_v)return; // Complete16-lane group exits; its peers never shuffle across groups.
    float state[kOwnedKeys],query[kOwnedKeys],key[kOwnedKeys];
#pragma unroll
    for(int pack=0;pack<kOwnedKeys/4;++pack)
    {
        const auto offset=key_begin+pack*4;
        const auto sr=gdnRecurrenceLoadFour<Access>(initial_state+column_base+offset);
        const auto qr=gdnRecurrenceLoadFour<Access>(q_head+offset);
        const auto kr=gdnRecurrenceLoadFour<Access>(k_head+offset);
        state[pack*4+0]=sr.x;state[pack*4+1]=sr.y;
        state[pack*4+2]=sr.z;state[pack*4+3]=sr.w;
        if constexpr(InputLayout==GDNRecurrentInputLayout::PreparedCompactQK)
        {
            // These bits already include original normalization and query scale;
            // even a multiply by one would violate the explicit prepared contract.
            query[pack*4+0]=qr.x;query[pack*4+1]=qr.y;
            query[pack*4+2]=qr.z;query[pack*4+3]=qr.w;
            key[pack*4+0]=kr.x;key[pack*4+1]=kr.y;
            key[pack*4+2]=kr.z;key[pack*4+3]=kr.w;
        }
        else
        {
        query[pack*4+0]=qr.x*q_inv;query[pack*4+1]=qr.y*q_inv;
        query[pack*4+2]=qr.z*q_inv;query[pack*4+3]=qr.w*q_inv;
        key[pack*4+0]=use_qk_l2norm?kr.x*k_inv:kr.x;
        key[pack*4+1]=use_qk_l2norm?kr.y*k_inv:kr.y;
        key[pack*4+2]=use_qk_l2norm?kr.z*k_inv:kr.z;
        key[pack*4+3]=use_qk_l2norm?kr.w*k_inv:kr.w;
        }
    }
#pragma unroll
    for(int i=0;i<kOwnedKeys;++i)state[i]*=decay;
    const float part_kv=gdnValueMajorOrderedQuarterDot(state,key,segment,quarter);
    const float kv=gdnValueMajorOrderedQuarterFold(part_kv);
    float value=0.0f;
    if(group_lane==0)
    {
        if constexpr(InputLayout==GDNRecurrentInputLayout::MergedPostConvolution||InputLayout==GDNRecurrentInputLayout::PreparedCompactQK)value=raw[2*key_span+std::size_t(h)*d_v+vi];
        else value=v[std::size_t(request)*values+std::size_t(h)*d_v+vi];
    }
    value=__shfl(value,0,16);
    const float delta=(value-kv)*beta_h;
#pragma unroll
    for(int i=0;i<kOwnedKeys;++i)
    {
        state[i]=state[i]+key[i]*delta;
        if(state_snapshots && request<max_snapshot_rows)
            state_snapshots[std::size_t(request)*snapshot_stride_floats+
                (std::size_t(h)*d_v+vi)*D_K+key_begin+i]=state[i];
    }
    const float part_out=gdnValueMajorOrderedQuarterDot(state,query,segment,quarter);
    const float out=gdnValueMajorOrderedQuarterFold(part_out);
    if(group_lane==0)output[std::size_t(request)*values+std::size_t(h)*d_v+vi]=out;
    if(!snapshotted)
    {
#pragma unroll
        for(int pack=0;pack<kOwnedKeys/4;++pack)
        {
            const auto bits=make_float4(state[pack*4+0],state[pack*4+1],state[pack*4+2],state[pack*4+3]);
            gdnRecurrenceStoreFour<Access>(updated_state+column_base+key_begin+pack*4,bits);
        }
    }
}
/**
 * @brief Grouped value-major entry over the original fixed-order numeric body.
 * @tparam D_K Original64/128 key width.
 * @tparam InputLayout Expanded planar or compact merged raw coordinates.
 * @details Physical state addressing alone differs. Grouped rows never invoke
 * a scalar-row replay; one original request/head CTA advances the full prefix.
 */
template<int D_K,GDNRecurrentInputLayout InputLayout>
__global__ __launch_bounds__(kGdnRecurrentThreads,2)
void rocm_gdn_value_major_recurrent_step_kernel(
    const float* q,const float* k,const float* v,const float* alpha,const float* beta_raw,
    const float* A_log,const float* dt_bias,float* output,const float* initial_state,float* updated_state,
    int request_count,int request_seq_len,int n_heads,int d_v,bool use_qk_l2norm,
    const int* effective_seq_len_ptr,int effective_row_idx,float* state_snapshots,
    int snapshot_stride_floats,int max_snapshot_rows,int merged_key_heads,int global_value_offset)
{
    rocm_gdn_recurrent_step_body<D_K,InputLayout,
        llaminar2::GDNRecurrenceStateLayout::ValueMajorKeyContiguous>(
        q,k,v,alpha,beta_raw,A_log,dt_bias,output,initial_state,updated_state,
        request_count,request_seq_len,n_heads,d_v,use_qk_l2norm,effective_seq_len_ptr,effective_row_idx,
        state_snapshots,snapshot_stride_floats,max_snapshot_rows,merged_key_heads,global_value_offset);
}

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
          GdnPrefillSnapshots SNAPSHOTS = GdnPrefillSnapshots::BoundedRows,
          GdnRecurrenceEndpointAccess Access = GdnRecurrenceEndpointAccess::ScalarAligned>
__global__ __launch_bounds__(256, 2)
void rocm_gdn_value_major_chunk_forward_kernel(
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
                        scale / sqrtf(q_total + 1e-6f);
                    gate_beta[2 * request_seq_len + t] =
                        1.0f / sqrtf(k_total + 1e-6f);
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
        const auto state_pack0=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+0);
        sc0=state_pack0.x;
        sc1=state_pack0.y;
        sc2=state_pack0.z;
        sc3=state_pack0.w;
        const auto state_pack1=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+4);
        sc4=state_pack1.x;
        sc5=state_pack1.y;
        sc6=state_pack1.z;
        sc7=state_pack1.w;
        const auto state_pack2=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+8);
        sc8=state_pack2.x;
        sc9=state_pack2.y;
        sc10=state_pack2.z;
        sc11=state_pack2.w;
        const auto state_pack3=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+12);
        sc12=state_pack3.x;
        sc13=state_pack3.y;
        sc14=state_pack3.z;
        sc15=state_pack3.w;
        if constexpr(STATIC_D_K==128)
        {
        const auto state_pack4=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+16);
        sc16=state_pack4.x;
        sc17=state_pack4.y;
        sc18=state_pack4.z;
        sc19=state_pack4.w;
        const auto state_pack5=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+20);
        sc20=state_pack5.x;
        sc21=state_pack5.y;
        sc22=state_pack5.z;
        sc23=state_pack5.w;
        const auto state_pack6=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+24);
        sc24=state_pack6.x;sc25=state_pack6.y;sc26=state_pack6.z;
        if constexpr(kUsesExplicitLdsStateTail)state_tail[0*kernel_block_size+tid]=state_pack6.w;else sc27=state_pack6.w;
        const auto state_pack7=gdnRecurrenceLoadFour<Access>(S_initial+std::size_t(vi)*kernel_d_k+j_start+28);
        if constexpr(kUsesExplicitLdsStateTail)
        {
            state_tail[1*kernel_block_size+tid]=state_pack7.x;
            state_tail[2*kernel_block_size+tid]=state_pack7.y;
            state_tail[3*kernel_block_size+tid]=state_pack7.z;
            state_tail[4*kernel_block_size+tid]=state_pack7.w;
        }
        else {sc28=state_pack7.x;sc29=state_pack7.y;sc30=state_pack7.z;sc31=state_pack7.w;}
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
                    scale / sqrtf(warp_sums[0] + 1e-6f);
                const float k_inv =
                    1.0f / sqrtf(warp_sums[8] + 1e-6f);
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
            snapshot[std::size_t(vi)*kernel_d_k+j_start+0] = sc0;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+1] = sc1;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+2] = sc2;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+3] = sc3;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+4] = sc4;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+5] = sc5;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+6] = sc6;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+7] = sc7;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+8] = sc8;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+9] = sc9;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+10] = sc10;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+11] = sc11;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+12] = sc12;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+13] = sc13;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+14] = sc14;
            snapshot[std::size_t(vi)*kernel_d_k+j_start+15] = sc15;
            if constexpr (STATIC_D_K == 128)
            {
                snapshot[std::size_t(vi)*kernel_d_k+j_start+16] = sc16;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+17] = sc17;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+18] = sc18;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+19] = sc19;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+20] = sc20;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+21] = sc21;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+22] = sc22;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+23] = sc23;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+24] = sc24;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+25] = sc25;
                snapshot[std::size_t(vi)*kernel_d_k+j_start+26] = sc26;
                if constexpr (kUsesExplicitLdsStateTail)
                {
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+27] =
                        state_tail[0 * kernel_block_size + tid];
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+28] =
                        state_tail[1 * kernel_block_size + tid];
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+29] =
                        state_tail[2 * kernel_block_size + tid];
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+30] =
                        state_tail[3 * kernel_block_size + tid];
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+31] =
                        state_tail[4 * kernel_block_size + tid];
                }
                else
                {
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+27] = sc27;
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+28] = sc28;
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+29] = sc29;
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+30] = sc30;
                    snapshot[std::size_t(vi)*kernel_d_k+j_start+31] = sc31;
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
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+0,make_float4(sc0,sc1,sc2,sc3));
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+4,make_float4(sc4,sc5,sc6,sc7));
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+8,make_float4(sc8,sc9,sc10,sc11));
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+12,make_float4(sc12,sc13,sc14,sc15));
        if constexpr(STATIC_D_K==128)
        {
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+16,make_float4(sc16,sc17,sc18,sc19));
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+20,make_float4(sc20,sc21,sc22,sc23));
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+24,make_float4(sc24,sc25,sc26,(kUsesExplicitLdsStateTail?state_tail[0*kernel_block_size+tid]:sc27)));
        gdnRecurrenceStoreFour<Access>(S_updated+std::size_t(vi)*kernel_d_k+j_start+28,make_float4((kUsesExplicitLdsStateTail?state_tail[1*kernel_block_size+tid]:sc28),(kUsesExplicitLdsStateTail?state_tail[2*kernel_block_size+tid]:sc29),(kUsesExplicitLdsStateTail?state_tail[3*kernel_block_size+tid]:sc30),(kUsesExplicitLdsStateTail?state_tail[4*kernel_block_size+tid]:sc31)));
        }
    }

}

}

namespace
{
/**
 * @brief Enqueue one explicit raw value-major row program over borrowed original coordinates.
 * @tparam D_K Original64/128 width.
 * @tparam InputLayout Immutable original planar/merged source interpretation.
 * @param p Validated physical row/state/gate tuple; merged q occupies query.
 * @param merged_heads Original compact heads, ignored for expanded planar.
 * @param offset Original modular merged-head offset, ignored for planar.
 * @param hs Exact caller-owned stream.
 * @return Native enqueue status; no replay, format conversion or alternate math.
 */
template<int D_K,GDNRecurrentInputLayout InputLayout>
bool enqueueGdnValueMajorRows(const llaminar2::gdn::PlanarRecurrenceInput& p,
    int merged_heads,int offset,hipStream_t hs)
{
    const dim3 grid(p.rows.requests()*p.heads,1+(p.value_width-1)/32);
#define VALUE_MAJOR_ARGS p.query,p.key,p.value,p.alpha,p.beta_raw,p.A_log,p.dt_bias,p.output,p.initial_state,p.updated_state, \
    p.rows.requests(),p.rows.rowsPerRequest(),p.heads,p.value_width,p.normalize_qk,p.rows.lengthOwner(),0, \
    p.snapshots,p.snapshot_stride_floats,p.max_snapshot_rows,merged_heads,offset
    if(p.rows.rowsPerRequest()==1)
    {
        std::uintptr_t alignment=reinterpret_cast<std::uintptr_t>(p.query)|
            reinterpret_cast<std::uintptr_t>(p.initial_state)|reinterpret_cast<std::uintptr_t>(p.updated_state);
        if constexpr(InputLayout==GDNRecurrentInputLayout::ExpandedPlanar)alignment|=reinterpret_cast<std::uintptr_t>(p.key);
        const auto raw_stride=2*std::size_t(merged_heads)*D_K+std::size_t(p.heads)*p.value_width;
        const bool vector_endpoints=(alignment%16==0)&&
            (InputLayout==GDNRecurrentInputLayout::ExpandedPlanar||p.rows.requests()==1||raw_stride%4==0);
        if(vector_endpoints)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_scalar_kernel<D_K,InputLayout,
                GdnRecurrenceEndpointAccess::VectorAligned>),grid,dim3(512),0,hs,VALUE_MAJOR_ARGS);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_scalar_kernel<D_K,InputLayout,
                GdnRecurrenceEndpointAccess::ScalarAligned>),grid,dim3(512),0,hs,VALUE_MAJOR_ARGS);
    }
    else
    {
        constexpr int tail=D_K/4>22?D_K/4-22:0;
        constexpr std::size_t shared=(2*D_K+4+2*128+tail*128)*sizeof(float);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_recurrent_step_kernel<D_K,InputLayout>),
            grid,dim3(128),shared,hs,VALUE_MAJOR_ARGS);
    }
#undef VALUE_MAJOR_ARGS
    return hipGetLastError()==hipSuccess;
}

/**
 * @brief Enqueue every original optimized long-M geometry in one explicit physical layout.
 * @tparam Layout Canonical persistent state-bank format.
 * @tparam Access Immutable endpoint alignment program for value-major storage.
 * @param p Validated original planar operands and replay-owned counts.
 * @param hs Exact caller-owned stream.
 * @return Native enqueue status; shared limits retain native hard-error behavior.
 */
template<llaminar2::GDNRecurrenceStateLayout Layout,GdnRecurrenceEndpointAccess Access>
bool enqueueGdnPlanarLong(const llaminar2::gdn::PlanarRecurrenceInput& p,hipStream_t hs)
{
    const int threads=p.value_width<=64?128:256,columns=threads/4;
    const bool fixed=p.key_width==128&&p.value_width==128&&threads==256;
    const bool cached=fixed&&p.rows.rowsPerRequest()<=2048;
    const int tail=fixed&&!cached?5:0;
    const int input=cached?16*(2*p.key_width+columns):2*p.key_width;
    const std::size_t shared=(std::size_t(input)+16+((cached?0:2)+tail)*threads+
        std::size_t(cached?4:2)*p.rows.rowsPerRequest())*sizeof(float);
    const dim3 grid(p.heads,1+(p.value_width-1)/columns,p.rows.requests());
#define PLANAR_LONG_ARGS p.query,p.key,p.value,p.alpha,p.beta_raw,p.A_log,p.dt_bias,p.output,p.initial_state,p.updated_state, \
    p.rows.lengthOwner(),p.snapshots,p.snapshot_stride_floats,p.max_snapshot_rows,p.rows.requests(),p.rows.rowsPerRequest(), \
    p.heads,p.value_width,p.normalize_qk
#define PLANAR_LONG(DK,DV,BLOCK,CACHE,SNAP) \
    if constexpr(Layout==llaminar2::GDNRecurrenceStateLayout::KeyMajorValueContiguous) \
    {hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_chunk_forward_kernel<DK,DV,BLOCK,CACHE,SNAP>), \
        grid,dim3(threads),shared,hs,PLANAR_LONG_ARGS);} \
    else \
    {hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_chunk_forward_kernel<DK,DV,BLOCK,CACHE,SNAP,Access>), \
        grid,dim3(threads),shared,hs,PLANAR_LONG_ARGS);}
    if(cached&&!p.snapshots){PLANAR_LONG(128,128,256,true,GdnPrefillSnapshots::TerminalOnly);}
    else if(cached){PLANAR_LONG(128,128,256,true,GdnPrefillSnapshots::BoundedRows);}
    else if(fixed){PLANAR_LONG(128,128,256,false,GdnPrefillSnapshots::BoundedRows);}
    else if(p.key_width==64){PLANAR_LONG(64,0,0,false,GdnPrefillSnapshots::BoundedRows);}
    else{PLANAR_LONG(128,0,0,false,GdnPrefillSnapshots::BoundedRows);}
#undef PLANAR_LONG
#undef PLANAR_LONG_ARGS
    return hipGetLastError()==hipSuccess;
}
} // anonymous namespace

namespace llaminar2::gdn
{
/**
 * @brief Submit one complete original planar recurrence in its admitted physical state format.
 * @param p Original planar sources, device prefix authority and immutable state layout.
 * @param stream Exact non-default caller-owned stream through captured retirement.
 * @return Whether cold admission and the selected native enqueue succeed.
 * @details State-format and alignment selection are immutable capture metadata.
 * No device payload is read by the host, and no format conversion is launched.
 */
bool launchROCmPlanarRecurrence(const PlanarRecurrenceInput& p,void* stream)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2||!validPlanarRecurrenceStorage(p))return false;
    const auto hs=static_cast<hipStream_t>(stream);
    const auto major=GDNRecurrenceStateLayout::ValueMajorKeyContiguous;
    const bool complete=p.snapshots&&p.max_snapshot_rows>=p.rows.physicalRows();
    if(p.rows.rowsPerRequest()<=8||complete)
    {
        if(p.recurrence_layout==major)
        {
            if(p.key_width==64)return enqueueGdnValueMajorRows<64,GDNRecurrentInputLayout::ExpandedPlanar>(p,0,0,hs);
            return enqueueGdnValueMajorRows<128,GDNRecurrentInputLayout::ExpandedPlanar>(p,0,0,hs);
        }
        return launch_rocm_gdn_recurrent_step("typed-planar-key-major",p.query,p.key,p.value,p.alpha,p.beta_raw,p.A_log,p.dt_bias,
            p.output,p.initial_state,p.updated_state,p.rows.requests(),p.rows.rowsPerRequest(),p.heads,p.key_width,p.value_width,
            p.normalize_qk,p.rows.lengthOwner(),0,p.snapshots,p.snapshot_stride_floats,p.max_snapshot_rows,hs);
    }
    if(p.recurrence_layout==major)
    {
        const auto alignment=reinterpret_cast<std::uintptr_t>(p.initial_state)|reinterpret_cast<std::uintptr_t>(p.updated_state);
        if(alignment%16==0)return enqueueGdnPlanarLong<GDNRecurrenceStateLayout::ValueMajorKeyContiguous,GdnRecurrenceEndpointAccess::VectorAligned>(p,hs);
        return enqueueGdnPlanarLong<GDNRecurrenceStateLayout::ValueMajorKeyContiguous,GdnRecurrenceEndpointAccess::ScalarAligned>(p,hs);
    }
    return enqueueGdnPlanarLong<GDNRecurrenceStateLayout::KeyMajorValueContiguous,GdnRecurrenceEndpointAccess::ScalarAligned>(p,hs);
}
}

namespace
{
/**
 * @brief Enqueue the original compact merged topology over an admitted value-major bank.
 * @param p Original raw source/rows/gates and value-major state/publication authority.
 * @param stream Exact non-default native stream.
 * @return Cold admission/enqueue status, without conversion or scalar replay.
 */
bool enqueueGdnValueMajorMerged(const llaminar2::gdn::MergedRecurrenceInput& p,void* stream)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2||!llaminar2::gdn::validMergedRecurrenceStorage(p)||
        p.rows.rowsPerRequest()>16||(p.rows.rowsPerRequest()>1&&!p.snapshots)||
        (p.snapshots&&p.max_snapshot_rows<p.rows.physicalRows()))return false;
    const auto hs=static_cast<hipStream_t>(stream);
    const dim3 grid(p.rows.requests()*p.value_heads,1+(p.value_width-1)/32);
#define MERGED_VALUE_ARGS p.merged,nullptr,nullptr,p.alpha,p.beta_raw,p.A_log,p.dt_bias,p.output,p.initial_state,p.updated_state, \
    p.rows.requests(),p.rows.rowsPerRequest(),p.value_heads,p.value_width,p.use_qk_l2norm,p.rows.lengthOwner(),0, \
    p.snapshots,p.snapshot_stride_floats,p.max_snapshot_rows,p.key_heads,p.global_value_offset
#define MERGED_VALUE_MAJOR(DK,ACCESS) hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_scalar_kernel<DK, \
    GDNRecurrentInputLayout::MergedPostConvolution,ACCESS>),grid,dim3(512),0,hs,MERGED_VALUE_ARGS)
    if(p.rows.rowsPerRequest()==1)
    {
        const auto alignment=reinterpret_cast<std::uintptr_t>(p.merged)|reinterpret_cast<std::uintptr_t>(p.initial_state)|
            reinterpret_cast<std::uintptr_t>(p.updated_state);
        const auto stride=2*std::size_t(p.key_heads)*p.key_width+std::size_t(p.value_heads)*p.value_width;
        const bool vector_endpoints=alignment%16==0&&(p.rows.requests()==1||stride%4==0);
        if(p.key_width==64)
        {
            if(vector_endpoints){MERGED_VALUE_MAJOR(64,GdnRecurrenceEndpointAccess::VectorAligned);}
            else{MERGED_VALUE_MAJOR(64,GdnRecurrenceEndpointAccess::ScalarAligned);}
        }
        else
        {
            if(vector_endpoints){MERGED_VALUE_MAJOR(128,GdnRecurrenceEndpointAccess::VectorAligned);}
            else{MERGED_VALUE_MAJOR(128,GdnRecurrenceEndpointAccess::ScalarAligned);}
        }
    }
    else
    {
        const int tail=std::max(0,p.key_width/4-22);
        const auto shared=std::size_t(2*p.key_width+4+2*128+tail*128)*sizeof(float);
        if(p.key_width==64)
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_recurrent_step_kernel<64,GDNRecurrentInputLayout::MergedPostConvolution>),grid,dim3(128),shared,hs,MERGED_VALUE_ARGS);
        else
            hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_recurrent_step_kernel<128,GDNRecurrentInputLayout::MergedPostConvolution>),grid,dim3(128),shared,hs,MERGED_VALUE_ARGS);
    }
#undef MERGED_VALUE_MAJOR
#undef MERGED_VALUE_ARGS
    return hipGetLastError()==hipSuccess;
}
} // anonymous namespace


namespace llaminar2::gdn
{
bool launchROCmScalarPreparedQKConvolution(const ScalarPreparedQKConvolutionInput& p,void* stream)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2||!validScalarPreparedQKConvolutionStorage(p))return false;
    const auto& a=p.operands;
    const auto layout=ScalarPreparedQKLayout::fromHeads(a.key_heads,a.value_heads,a.key_width,a.value_width);
    const int channels=static_cast<int>(layout.elements+std::size_t(a.value_heads)*a.value_width);
    const dim3 grid((channels-1)/a.key_width+1);
    const auto hs=reinterpret_cast<hipStream_t>(stream);
#define SCALAR_PREPARED_CONV(DK) hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_scalar_prepared_qk_conv_kernel<DK>), \
    grid,dim3(128),0,hs,a.input,a.weight,a.bias,p.initial_state,a.raw_output,p.updated_state,p.snapshots, \
    a.prepared,a.prepared+layout.plane_elements,1,a.key_heads,a.value_heads,a.value_width,nullptr, \
    p.snapshot_stride_floats,p.max_snapshot_rows,a.normalize_qk)
    if(a.key_width==64){SCALAR_PREPARED_CONV(64);}
    else{SCALAR_PREPARED_CONV(128);}
#undef SCALAR_PREPARED_CONV
    return hipGetLastError()==hipSuccess;
}

bool launchROCmScalarPreparedQKRecurrence(const ScalarPreparedQKRecurrenceInput& inputs,void* stream)
{
    if(reinterpret_cast<std::uintptr_t>(stream)<=2||!validScalarPreparedQKRecurrenceStorage(inputs))return false;
    const auto& p=inputs.original;
    const auto layout=ScalarPreparedQKLayout::fromHeads(p.key_heads,p.value_heads,p.key_width,p.value_width);
    const auto hs=reinterpret_cast<hipStream_t>(stream);
    const dim3 grid(p.value_heads,(p.value_width-1)/32+1);
    // One immutable alignment specialization also accepts float-aligned native
    // test endpoints. Odd snapshot strides remain scalar stores in the kernel.
    const auto alignment=reinterpret_cast<std::uintptr_t>(p.initial_state)|
        reinterpret_cast<std::uintptr_t>(p.updated_state)|reinterpret_cast<std::uintptr_t>(inputs.prepared);
    const bool vector_endpoints=alignment%16==0;
#define SCALAR_PREPARED_REC(DK,ACCESS) hipLaunchKernelGGL(HIP_KERNEL_NAME(rocm_gdn_value_major_scalar_kernel<DK, \
    GDNRecurrentInputLayout::PreparedCompactQK,ACCESS>),grid,dim3(512),0,hs,p.merged,inputs.prepared, \
    inputs.prepared+layout.plane_elements,p.alpha,p.beta_raw,p.A_log,p.dt_bias,p.output,p.initial_state,p.updated_state, \
    1,1,p.value_heads,p.value_width,p.use_qk_l2norm,p.rows.lengthOwner(),0,p.snapshots, \
    p.snapshot_stride_floats,p.max_snapshot_rows,p.key_heads,p.global_value_offset)
    if(p.key_width==64)
    {
        if(vector_endpoints){SCALAR_PREPARED_REC(64,GdnRecurrenceEndpointAccess::VectorAligned);}
        else{SCALAR_PREPARED_REC(64,GdnRecurrenceEndpointAccess::ScalarAligned);}
    }
    else
    {
        if(vector_endpoints){SCALAR_PREPARED_REC(128,GdnRecurrenceEndpointAccess::VectorAligned);}
        else{SCALAR_PREPARED_REC(128,GdnRecurrenceEndpointAccess::ScalarAligned);}
    }
#undef SCALAR_PREPARED_REC
    return hipGetLastError()==hipSuccess;
}
} // namespace llaminar2::gdn

namespace llaminar2::rocm
{
/** @copydoc launchGatedNormSingleQ8 */
bool launchGatedNormSingleQ8(const GatedNormSingleQ8Input& input,int device,void* stream)
{
    if(device<0||reinterpret_cast<std::uintptr_t>(stream)<=2||!input.input||!input.gate||!input.gamma||
       !input.output||!input.quant||!input.scales||
       !gatedNormSingleQ8Geometry(1,input.columns,input.parameters))
        throw std::invalid_argument("Gated-norm SingleQ8 requires exact stream, owned pointers and aligned128 scalar geometry");
    if(HipDeviceGuard::setDevice(device)!=static_cast<int>(hipSuccess))return false;
    hipLaunchKernelGGL(rocm_gdn_gated_norm_single_q8_kernel,dim3(input.columns/128),dim3(128),
        0,static_cast<hipStream_t>(stream),input);
    return hipGetLastError()==hipSuccess;
}
} // namespace llaminar2::rocm
