/**
 * @file ROCmGDNHistoricalRecurrentOracle.hip
 * @brief Literal pre-pairing native recurrence oracle, linked only into focused tests.
 *
 * The full historical kernel is frozen independently of production helpers.
 * Root first calibrates it against unchanged production, then preserves this
 * file unchanged while testing normalization publication scheduling. State
 * updates, gates, scalar/grouped loops, tails and every old barrier remain here.
 */
#include <hip/hip_runtime.h>
#include "ROCmGDNHistoricalRecurrentOracle.h"
namespace gdn_historical_native
{
constexpr int kGdnRecurrentThreads=256;
constexpr int kGdnRecurrentRowSplit=4;
constexpr int kGdnRecurrentColumnsPerBlock=64;
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
                scale / sqrtf(wave_sums[0] + 1e-6f);
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
                1.0f / sqrtf(wave_sums[0] + 1e-6f);
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


bool launch(const Arguments &a)
{
    auto stream=static_cast<hipStream_t>(a.stream);
    if(!stream || stream==hipStreamLegacy || stream==hipStreamPerThread ||
       !a.q || !a.k || !a.v || !a.alpha || !a.beta || !a.a_log || !a.dt_bias ||
       !a.output || !a.initial || !a.updated || a.requests<=0 || a.rows<=0 ||
       a.heads<=0 || (a.key_width!=64 && a.key_width!=128) || a.value_width<=0)
        return false;
    const dim3 grid(a.requests*a.heads,(a.value_width+63)/64);
    const int tail=a.key_width/4>22 ? a.key_width/4-22 : 0;
    const auto shared=(2*a.key_width+4+512+tail*256)*sizeof(float);
#define GDN_HISTORICAL_LAUNCH(K) hipLaunchKernelGGL( \
    HIP_KERNEL_NAME(rocm_gdn_recurrent_step_kernel<K>),grid,dim3(256),shared,stream, \
    a.q,a.k,a.v,a.alpha,a.beta,a.a_log,a.dt_bias,a.output,a.initial,a.updated, \
    a.requests,a.rows,a.heads,a.value_width,a.normalize,a.lengths,a.row_offset, \
    a.snapshots,a.snapshot_stride,a.snapshot_rows)
    if(a.key_width==64){GDN_HISTORICAL_LAUNCH(64);}
    else{GDN_HISTORICAL_LAUNCH(128);}
#undef GDN_HISTORICAL_LAUNCH
    return hipPeekAtLastError()==hipSuccess;
}
} // namespace gdn_historical_native
