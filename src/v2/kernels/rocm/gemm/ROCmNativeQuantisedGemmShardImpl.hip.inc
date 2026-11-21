/**
 * @file ROCmNativeQuantisedGemmShardImpl.hip.inc
 * @brief NativeQuantised GEMM kernels for lossless sub-8-bit prefill on AMD MI50/MI60 (gfx906)
 *
 * @target AMD Instinct MI50 / MI60 (gfx906), 60 CUs, 4 SIMDs/CU
 * @isa    v_dot4_i32_i8 (VNNI: 4-way signed INT8 dot product, 1 cycle/op)
 * @op     C[M×N] (FP32) = scale_A[M] × Σ_b { block_d[b,N] × sdot4(A_int8[M,K], W_decoded[K,N]) }
 *
 * This file contains the production native-quantised GEMM kernels for the **prefill**
 * (prompt-processing) phase on AMD gfx906 GPUs.  Unlike the INT8 GEMM kernels
 * (ROCmQuantisedGemmKernel_INT8_VNNI.hip), these kernels:
 *
 *   1. Load COMPACT quantized payloads from HBM (4.5 bpw for Q4_0/IQ4_NL
 *      vs 8 bpw for INT8) — halving HBM bandwidth requirements
 *   2. Decode quantized weights to INT8 in-register during the load phase
 *      (Decode-in-Load architecture)
 *   3. Use FP32 accumulators with source-exact Q6/IQ2_S/IQ4_XS factors or prepared half scales
 *      for other formats (no separate epilogue)
 *   4. Apply activation scale inline → FP32 output directly
 *
 * The decode-in-load approach stores decoded INT8 weights in LDS (same format
 * as INT8 GEMM), making the compute phase identical to V3's inner loop.
 * The key benefit is that global memory reads are halved — the decode overhead
 * is hidden in the load-to-LDS pipeline.
 *
 * Tile dispatch mirrors INT8 V3/V7 strategy:
 *   K >= N (K-heavy) → N_TILE=64  (V3-derived, deeper K iteration)
 *   K <  N (N-heavy) → N_TILE=128 (V7-derived, wider N coverage, all threads load B)
 *
 * The physical shard family supports every source format admitted by the
 * canonical NativeQuantised registry and enumerated below. Format-specific payload
 * decoding remains compile-time, while all formats share one explicit FP32
 * contribution/publication contract with serial GEMV and CUDA.
 * Serial partition boundaries advance in monotonically increasing K order;
 * a uniform boundary cursor avoids repeated integer remainder instructions
 * without changing any floating-point contribution or partition fold.
 *
 * @see ROCmQuantisedGemmKernel_INT8_VNNI.hip for INT8 VNNI GEMM (V3/V7)
 * @see ROCmGemvKernel_native_quantised.hip for native-quantised GEMV (M=1 decode)
 * @see docs/v2/projects/2026-03/NATIVE_QUANTISED_GEMM_DESIGN.md for architecture design document
 * Native signed INT8 dot products share the exact unsaturated arithmetic
 * contract across gfx906 and gfx1100; lane ownership and reductions are unchanged.
 * Compact decoder source is shared with the separately compiled wave64 WMMA
 * family. These ordered dot4 kernels retain their wave64 compilation and call
 * the typed WMMA submission boundary only after native gfx1100 admission.
 */

#include "kernels/rocm/common/ROCmSignedDot4.h"
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "../../../backends/rocm/HipDeviceGuard.h"
#include "../../../execution/local_execution/graph/GraphCaptureGuard.h"
#include "../../../tensors/NativeQuantisedFormatInfo.h"
#include "../../../utils/DebugEnv.h"
#include "ROCmNativeQuantisedDecodeEquivalentMath.h"
#include "ROCmNativeQuantisedGemmShard.h"
#include "ROCmDenseProductionPrefillOverlay.h"
#include "ROCmNativeQuantisedArchitecture.h"
#include "ROCmNativeQuantisedWmmaLaunch.h"

#ifndef LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD
#error "ROCmNativeQuantisedGemmShardImpl.hip.inc requires a physical shard index"
#endif

#if LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD < 0 || LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD > 7
#error "ROCm NativeQuantised GEMM shard index must be in [0, 7]"
#endif

#define LLAMINAR_ROCM_NATIVE_QUANTISED_CONCAT_IMPL(A, B) A##B
#define LLAMINAR_ROCM_NATIVE_QUANTISED_CONCAT(A, B) \
    LLAMINAR_ROCM_NATIVE_QUANTISED_CONCAT_IMPL(A, B)
#define LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD_SYMBOL(PREFIX) \
    LLAMINAR_ROCM_NATIVE_QUANTISED_CONCAT(PREFIX, LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD)

extern "C" bool rocmGemv_native_quantised_query_serial_m1_config_with_policy(
    uint8_t arithmetic_policy_codebook_id,
    int N,
    int K,
    int* kb,
    int* target_waves_per_cu);

#include "ROCmNativeQuantisedGemmDecode.hip.inc"

// ============================================================================
// NativeQuantised GEMM kernel: Decode-in-Load with INT32 block accumulators
//
// Architecture: LDS double-buffered pipeline (V3-derived)
//   - A tile: INT8 activations → LDS (identical to INT8 V3)
//   - B tile: raw Q4 payload loaded from HBM, decoded to INT8, stored in LDS
//   - Scales: FP16 per-block scales loaded into small LDS array
//   - Compute: INT32 accumulation within each block → FP32 scale at block exit
//
// Tile geometry (default S64, BPT=2):
//   N_TILE=64, BPT=2 (2 blocks/tile = 16 k-groups = 64 K-elements)
//   M_TILE=64, M_PER_THREAD=4, THREADS_M=16, THREADS_N=16, N_PER_THREAD=4
//   Grid: (ceil(N/64), ceil(M/64))
//
// LDS budget:
//   a_lds[2][16×64]  =  8192 bytes (A: same as V3)
//   b_lds[2][16×64]  =  8192 bytes (B: decoded INT8)
//   scale_lds[2][2×64] = 512 bytes (FP16 per-block scales)
//   Total: 16896 bytes ≈ 16.5 KB → 2 WGs/CU
//
// Key optimisation: INT32 block accumulation.
// Within each quantized block (8 k-groups), dot products accumulate
// directly into int32 via sdot4(a, b, block_acc). Scale conversion
// happens once per block (not per k-group), reducing inner-loop VALU
// from 3 instructions/dot to 1 instruction/dot — a 2.4× reduction.
// The K-step loop uses `#pragma clang loop unroll(disable)` to
// contain register pressure (matching V7's approach).
//
// Output: FP32 with scale_A applied inline. No separate epilogue needed.
// ============================================================================

// ============================================================================
// Per-block accumulate helper — handles four accumulation patterns:
//
//   Pattern S (simple symmetric):     acc += block_acc * scale
//   Pattern A (asymmetric):           acc += block_acc * scale + sum_a * aux
//   Pattern D (dual-scale symmetric): acc += acc_lo * scale_lo + acc_hi * aux_hi
//   Pattern DA (dual-scale + asym):   acc += acc_lo * scale + acc_hi * aux
//                                          + sum_a_lo * emin_lo + sum_a_hi * emin_hi
//
// Called once per block (bp) per tile to avoid triplication of compute code
// in safe/boundary/final loops.
// ============================================================================

/**
 * @brief Commit one quantization block with bounded integer-dot lifetimes.
 * @tparam N_TILE Workgroup output-column extent.
 * @tparam BPT Number of quantization blocks in each shared-memory slot.
 * @tparam M_TILE Workgroup output-row extent.
 * @tparam FMT Physical codebook and its exact scale/correction arithmetic.
 * @tparam UNROLL_G Dot-group unrolling policy for simple symmetric formats.
 * @tparam BLOCK_SIZE_ Workgroup thread count.
 * @tparam M_PER_THREAD_ Output rows owned by each lane.
 * @tparam MINIMUM_WAVES Kernel launch bound constraining the register budget.
 * @param bp Quantization-block index within the current shared-memory slot.
 * @param a_buf Packed activation operands, indexed by K group then row.
 * @param b_buf Decoded weight operands, indexed by K group then column.
 * @param s_buf Q6/IQ4_XS original half d, otherwise first prepared half weight scale.
 * @param aux_buf Q6 signed factors, IQ4_XS raw encoded6, or format-specific half auxiliary.
 * @param iq1_m_qh_buf IQ1_M delta-sign bits; unused for other codebooks.
 * @param emin_lo_buf Lower-half asymmetric correction coefficients.
 * @param emin_hi_buf Upper-half asymmetric correction coefficients.
 * @param sum_a_buf Cooperatively computed half-block activation sums.
 * @param acc Current serial-K-partition FP32 output accumulators.
 * @param m_base First row owned by this lane within the shared slot.
 * @param t_n Lane's output-column group.
 * @param sa_blk Exact per-row activation scale for this quantization block.
 *
 * Independent columns may be committed earlier to release integer registers;
 * the order of contributions to any one output must never change. This keeps
 * ordinary tiles' operand reuse and narrows only register-constrained tiles.
 */
template <int N_TILE, int BPT, int M_TILE, NativeQuantisedGemmFormat FMT, int UNROLL_G = 4,
          int BLOCK_SIZE_ = 256, int M_PER_THREAD_ = 4, int MINIMUM_WAVES = 2>
__device__ __forceinline__ void accumulate_block_gemm(
    int bp,
    const int32_t* __restrict__ a_buf,
    const int32_t* __restrict__ b_buf,
    const uint16_t* __restrict__ s_buf,
    const uint16_t* __restrict__ aux_buf,
    const uint16_t* __restrict__ iq1_m_qh_buf,
    const uint16_t* __restrict__ emin_lo_buf,
    const uint16_t* __restrict__ emin_hi_buf,
    const int32_t* __restrict__ sum_a_buf,
    float (&acc)[M_PER_THREAD_][N_TILE / (BLOCK_SIZE_ / (M_TILE / M_PER_THREAD_))],
    int m_base, int t_n,
    const float (&sa_blk)[M_PER_THREAD_])
{
    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;

    constexpr int M_PER_THREAD = M_PER_THREAD_;
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;
    constexpr int THREADS_N = BLOCK_SIZE_ / THREADS_M;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;

    if constexpr (Traits::is_dual_scale_asym || Traits::is_dual_scale)
    {
        // Preserve the full MxN operand reuse wherever it fits. Only a tight
        // three-wave budget or a wide correction-bearing tile needs to retire
        // columns earlier. This is independent-output scheduling: every dot4
        // and FP32 commit retains the exact serial-M1 arithmetic tree.
        constexpr bool bounded_columns =
            (MINIMUM_WAVES >= 3 && N_PER_THREAD >= 2) ||
            (MINIMUM_WAVES >= 2 && N_PER_THREAD >= 4 &&
             (Traits::is_dual_scale_asym || FMT == NATIVE_QUANTISED_GEMM_IQ1_M));
        constexpr int columns = bounded_columns ? 1 : N_PER_THREAD;
        #pragma unroll
        for (int column_base = 0; column_base < N_PER_THREAD; column_base += columns)
        {
            int32_t low_dot[M_PER_THREAD][columns] = {};
            int32_t high_dot[M_PER_THREAD][columns] = {};
            #pragma clang loop unroll_count(2)
            for (int g = 0; g < 4; ++g)
            {
                const int kk = bp * 8 + g;
                int32_t a_reg[M_PER_THREAD];
                int32_t b_reg[columns];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                #pragma unroll
                for (int nn = 0; nn < columns; ++nn)
                    b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + column_base + nn];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < columns; ++nn)
                        low_dot[mm][nn] = ::llaminar2::rocm_integer_contract::signedDot4(
                            a_reg[mm], b_reg[nn], low_dot[mm][nn]);
            }
            #pragma clang loop unroll_count(2)
            for (int g = 4; g < 8; ++g)
            {
                const int kk = bp * 8 + g;
                int32_t a_reg[M_PER_THREAD];
                int32_t b_reg[columns];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                #pragma unroll
                for (int nn = 0; nn < columns; ++nn)
                    b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + column_base + nn];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < columns; ++nn)
                        high_dot[mm][nn] = ::llaminar2::rocm_integer_contract::signedDot4(
                            a_reg[mm], b_reg[nn], high_dot[mm][nn]);
            }

            #pragma unroll
            for (int nn = 0; nn < columns; ++nn)
            {
                const int column = column_base + nn;
                const int sn = bp * N_TILE + t_n * N_PER_THREAD + column;
                const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
                    s_buf[sn], aux_buf[sn]);
                const float scale_lo = weight_scales.low;
                const float scale_hi = weight_scales.high;
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                {
                    if constexpr (Traits::is_dual_scale_asym)
                    {
                        const int32_t low_sum = sum_a_buf[(bp * 2) * M_TILE + m_base + mm];
                        const int32_t high_sum = sum_a_buf[(bp * 2 + 1) * M_TILE + m_base + mm];
                        const float emin_lo = __half2float(
                            *reinterpret_cast<const __half*>(&emin_lo_buf[sn]));
                        const float emin_hi = __half2float(
                            *reinterpret_cast<const __half*>(&emin_hi_buf[sn]));
                        acc[mm][column] = rocm_native_quantised_commit_corrected_dual_scale_block(
                            acc[mm][column], low_dot[mm][nn], high_dot[mm][nn],
                            scale_lo, scale_hi, low_sum, high_sum,
                            emin_lo, emin_hi, sa_blk[mm]);
                    }
                    else if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    {
                        constexpr int32_t ones = 0x01010101;
                        constexpr float delta = 0.125f;
                        int32_t sums[4];
                        #pragma unroll
                        for (int group = 0; group < 4; ++group)
                        {
                            const int kg0 = bp * 8 + group * 2;
                            sums[group] = ::llaminar2::rocm_integer_contract::signedDot4(
                                a_buf[(kg0 + 1) * M_TILE + m_base + mm], ones,
                                ::llaminar2::rocm_integer_contract::signedDot4(
                                    a_buf[kg0 * M_TILE + m_base + mm], ones, 0));
                        }
                        const uint16_t qh = iq1_m_qh_buf[sn];
                        acc[mm][column] = rocm_native_quantised_commit_iq1_m_block(
                            acc[mm][column], low_dot[mm][nn], high_dot[mm][nn],
                            sums[0], sums[1], sums[2], sums[3],
                            (qh & 0x0008) ? -delta : delta,
                            (qh & 0x0080) ? -delta : delta,
                            (qh & 0x0800) ? -delta : delta,
                            (qh & 0x8000) ? -delta : delta,
                            scale_lo, scale_hi, sa_blk[mm]);
                    }
                    else
                        acc[mm][column] = rocm_native_quantised_commit_dual_scale_block(
                            acc[mm][column], low_dot[mm][nn], high_dot[mm][nn],
                            scale_lo, scale_hi, sa_blk[mm]);
                }
            }
        }
    }
    else if constexpr (Traits::is_asymmetric)
    {
        // Pattern A: asymmetric single-scale (Q4_1, Q5_1, IQ1_S)
        //
        // sum_a is pre-computed cooperatively during A-load and stored in
        // sum_a_buf (same layout as Pattern DA: per-half sums at bhi*M_TILE).
        // This eliminates M_PER_THREAD × 8 sdot4/block from the inner loop
        // and removes 4 VGPRs (sum_a[] array no longer live across groups).
        // Bound wide correction tiles to two independent columns at a time.
        // Both dots and their FP32 corrections retire before the next pair;
        // the output's ordered K contributions are unchanged. The remaining
        // A-operand reuse avoids turning a register-pressure repair into a
        // one-column serial schedule.
        constexpr int columns =
            MINIMUM_WAVES >= 2 && N_PER_THREAD >= 4 ? 2 : N_PER_THREAD;
        #pragma unroll
        for (int column_base = 0; column_base < N_PER_THREAD; column_base += columns)
        {
            int32_t block_acc[M_PER_THREAD][columns];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < columns; ++nn)
                    block_acc[mm][nn] = 0;
            #pragma clang loop unroll_count(4)
            for (int g = 0; g < 8; ++g)
            {
                const int kk = bp * 8 + g;
                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_buf[kk * M_TILE + m_base + mm];
                int32_t b_reg[columns];
                #pragma unroll
                for (int nn = 0; nn < columns; ++nn)
                    b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + column_base + nn];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < columns; ++nn)
                        block_acc[mm][nn] = ::llaminar2::rocm_integer_contract::signedDot4(
                            a_reg[mm], b_reg[nn], block_acc[mm][nn]);
            }
            // The canonical helper rounds the scaled dot and minimum correction
            // before one ordered accumulator publication. Read cooperative sum_a
            // from LDS (lo + hi halves = full block sum).
            int32_t sum_a[M_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                sum_a[mm] = sum_a_buf[(bp * 2) * M_TILE + m_base + mm]
                          + sum_a_buf[(bp * 2 + 1) * M_TILE + m_base + mm];
            #pragma unroll
            for (int nn = 0; nn < columns; ++nn)
            {
                const int column = column_base + nn;
                const int sn = bp * N_TILE + t_n * N_PER_THREAD + column;
                const float block_scale = __half2float(*reinterpret_cast<const __half*>(&s_buf[sn]));
                const float block_min   = __half2float(*reinterpret_cast<const __half*>(&aux_buf[sn]));
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                {
                    acc[mm][column] =
                        rocm_native_quantised_commit_corrected_single_scale_block(
                            acc[mm][column],
                            block_acc[mm][nn],
                            block_scale,
                            sum_a[mm],
                            block_min,
                            sa_blk[mm]);
                }
            }
        }
    }
    else
    {
        // Pattern S: simple symmetric (Q4_0, IQ4_NL, Q5_0, IQ3_S, IQ3_XXS, IQ2_XXS)
        int32_t block_acc[M_PER_THREAD][N_PER_THREAD];
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                block_acc[mm][nn] = 0;

        // UNROLL_G controls the inner group loop unroll for tuning.
        // Default (4) matches original behavior: #pragma clang loop unroll_count(4).
#define NATIVE_QUANTISED_PATTERN_S_G_BODY \
        { \
            const int kk = bp * 8 + g; \
            int32_t a_reg[M_PER_THREAD]; \
            _Pragma("unroll") \
            for (int mm = 0; mm < M_PER_THREAD; ++mm) \
                a_reg[mm] = a_buf[kk * M_TILE + m_base + mm]; \
            int32_t b_reg[N_PER_THREAD]; \
            _Pragma("unroll") \
            for (int nn = 0; nn < N_PER_THREAD; ++nn) \
                b_reg[nn] = b_buf[kk * N_TILE + t_n * N_PER_THREAD + nn]; \
            _Pragma("unroll") \
            for (int mm = 0; mm < M_PER_THREAD; ++mm) \
                _Pragma("unroll") \
                for (int nn = 0; nn < N_PER_THREAD; ++nn) \
                    block_acc[mm][nn] = ::llaminar2::rocm_integer_contract::signedDot4( \
                        a_reg[mm], b_reg[nn], block_acc[mm][nn]); \
        }
        if constexpr (UNROLL_G == 0) {
            for (int g = 0; g < 8; ++g) NATIVE_QUANTISED_PATTERN_S_G_BODY
        } else if constexpr (UNROLL_G == 2) {
            #pragma clang loop unroll_count(2)
            for (int g = 0; g < 8; ++g) NATIVE_QUANTISED_PATTERN_S_G_BODY
        } else if constexpr (UNROLL_G == 1) {
            #pragma unroll
            for (int g = 0; g < 8; ++g) NATIVE_QUANTISED_PATTERN_S_G_BODY
        } else { // UNROLL_G == 4 (default)
            #pragma clang loop unroll_count(4)
            for (int g = 0; g < 8; ++g) NATIVE_QUANTISED_PATTERN_S_G_BODY
        }
#undef NATIVE_QUANTISED_PATTERN_S_G_BODY
        // FMA: acc += block_acc * scale
        float block_scale[N_PER_THREAD];
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            const uint16_t s_bits = s_buf[bp * N_TILE + t_n * N_PER_THREAD + nn];
            block_scale[nn] = rocm_native_quantised_load_single_weight_scale<FMT>(s_bits,
                (FMT == NATIVE_QUANTISED_GEMM_IQ3_S || FMT == NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE) ?
                    aux_buf[bp * N_TILE + t_n * N_PER_THREAD + nn] : uint16_t{0});
        }
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                acc[mm][nn] = rocm_native_quantised_commit_single_scale_block(
                    acc[mm][nn],
                    block_acc[mm][nn],
                    block_scale[nn],
                    sa_blk[mm]);
    }
}

/**
 * @brief Commit one completed serial-M1 K partition into workgroup-local state.
 *
 * Serial decode gives each split-K worker a zero-initialized accumulator, then
 * the reducer adds those partials in ascending partition order. A prefill tile
 * already has enough M-by-N parallelism and does not benefit from launching
 * separate K workers, but it must retain the same FP32 expression tree. Each
 * output owner therefore stores its running partition in `acc`, commits it to
 * a private register accumulator at the exact serial boundary, and starts the
 * next partition from positive zero. Keeping both values in registers avoids
 * adding global traffic or repeated LDS round trips to the K loop.
 *
 * @tparam N_TILE Number of output columns owned by the workgroup.
 * @tparam M_TILE Number of output rows owned by the workgroup.
 * @tparam M_PER_THREAD Number of output rows owned by one thread.
 * @tparam N_PER_THREAD Number of output columns owned by one thread.
 * @param acc Current partition accumulators.
 * @param ordered_sum Per-thread ordered reduction accumulators.
 * @param global_block Zero-based 32-value weight block just accumulated.
 * @param blocks_per_row Total number of real weight blocks.
 * @param blocks_per_partition Serial-M1 partition span.
 * @param next_partition_end Next exclusive block boundary, advanced only on a commit.
 * @param row_origin First global activation row owned by the register tile.
 * @param live_rows Admitted live row count; inactive tail rows read no scale.
 * @param whole_row_scales Null for blockwise scaling, otherwise one scale per row.
 * @param t_n Thread column-group index.
 */
template <int N_TILE, int M_TILE, int M_PER_THREAD, int N_PER_THREAD>
__device__ __forceinline__ void commit_serial_m1_partition_if_complete(
    float (&acc)[M_PER_THREAD][N_PER_THREAD],
    float (&ordered_sum)[M_PER_THREAD][N_PER_THREAD],
    int global_block,
    int blocks_per_row,
    int blocks_per_partition,
    int &next_partition_end,
    int row_origin,
    int live_rows,
    const float* whole_row_scales,
    int)
{
    if (global_block >= blocks_per_row)
        return;

    const int completed_blocks = global_block + 1;
    const bool partition_complete =
        completed_blocks == blocks_per_row ||
        completed_blocks == next_partition_end;
    if (!partition_complete)
        return;

    // All lanes visit every real K block once in ascending order. Advancing
    // only at that boundary is equivalent to modulo, including a short final
    // partition. This is integer bookkeeping, not a different reduction tree.
    next_partition_end += blocks_per_partition;

    // Finish the optional scale phase before opening the reduction phase.
    // In-place publication prevents scale/mask temporaries from overlapping
    // the ordered-sum register tile; blockwise execution reads no row scale.
    if (whole_row_scales)
    {
        // Serial GEMV scales a completed partition before its ordered reducer.
        // Multiplying the final grouped sum instead changes rounding, even
        // though the two expressions are equal over real numbers. Only the
        // whole-row contract needs this late load; the production blockwise
        // lane retains its existing contributions and register lifetimes.
        // A partition needs these addresses only after its dot products have
        // retired. Prevent the compiler from keeping sign-extended row
        // addresses and tail masks across the register-heavy K body. These
        // identity operands emit no instruction and change no scale value.
        int scale_row_origin = row_origin;
        int scale_live_rows = live_rows;
        asm volatile("" : "+v"(scale_row_origin), "+s"(scale_live_rows));
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            const int global_row = scale_row_origin + mm;
            if (global_row >= scale_live_rows)
                continue;
            const float row_scale = whole_row_scales[global_row];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
                acc[mm][nn] = __fmul_rn(acc[mm][nn], row_scale);
        }
    }
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            // The serial reducer starts at positive zero, including its first
            // addition. Assignment is not interchangeable here: 0 + (-0) is
            // positive zero, while assigning the first partial retains -0.
            ordered_sum[mm][nn] = __fadd_rn(
                ordered_sum[mm][nn], acc[mm][nn]);
            acc[mm][nn] = 0.0f;
        }
    }
}

/**
 * @brief Reload the completed serial-M1 reduction tree for final publication.
 *
 * @tparam N_TILE Number of output columns owned by the workgroup.
 * @tparam M_PER_THREAD Number of output rows owned by one thread.
 * @tparam N_PER_THREAD Number of output columns owned by one thread.
 * @param acc Destination register tile used by the existing vectorized store.
 * @param ordered_sum Per-thread ordered reduction accumulators.
 * @param m_base Thread-local row origin within the workgroup tile.
 * @param t_n Thread column-group index.
 */
template <int N_TILE, int M_PER_THREAD, int N_PER_THREAD>
__device__ __forceinline__ void load_serial_m1_partition_sum(
    float (&acc)[M_PER_THREAD][N_PER_THREAD],
    const float (&ordered_sum)[M_PER_THREAD][N_PER_THREAD],
    int,
    int)
{
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            acc[mm][nn] = ordered_sum[mm][nn];
        }
    }
}

/**
 * @brief Evaluate one cooperative output tile with serial-row-exact arithmetic.
 *
 * Two LDS banks separate current readers from prefetched weight/activation
 * tiles. Every output owner retains its own partition and ordered reduction
 * registers. The serial-M1 policy defines K boundaries independently of tile
 * geometry; whole-row scales are consumed before that ordered fold, while
 * blockwise scales belong to each block contribution. No row is replayed and
 * no extra reduction workspace is allocated by the grouped producer.
 *
 * @tparam N_TILE Output columns owned by this workgroup.
 * @tparam BPT Number of 32-element K blocks in one staged tile.
 * @tparam M_TILE Activation rows owned by this workgroup.
 * @tparam FMT Physical packed codebook and contribution contract.
 * @tparam MIN_BLOCKS Required resident-block register budget.
 * @tparam UNROLL_G Integer-dot unroll choice; floating-point order is fixed.
 * @tparam FULL_TILES Admission proved every M/N/K tile fully live.
 * @param d_A_int8 Row-major prepared INT8 activations.
 * @param d_payload Compact weight blocks in K-block/column order.
 * @param d_block_scales Primary FP16 weight scale per block/column.
 * @param d_block_mins Optional auxiliary weight scale or minimum plane.
 * @param d_block_emins Optional packed FP16 minimum pairs for Q2_K.
 * @param d_output Live row-major FP32 output extent.
 * @param d_scales_A Whole-row activation scales when the block plane is absent.
 * @param M Live activation rows; inactive tails never read row scales.
 * @param N Live output columns.
 * @param K Admitted activation width, divisible by the physical block width.
 * @param d_scales_A_blockwise Optional per-row/per-block activation scales.
 * @param serial_m1_kblocks Exact partition count resolved by serial decode policy.
 */
template <int N_TILE,
          int BPT,
          int M_TILE,
          NativeQuantisedGemmFormat FMT,
          int MIN_BLOCKS = 2,
          int UNROLL_G = 4,
          bool FULL_TILES = false>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void native_quantised_gemm_kernel(
    const int8_t*    __restrict__ d_A_int8,       // [M × K] INT8 activations, row-major
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × PB] native-quantised payload
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N] FP16 per-block scales
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row × N] FP16 per-block min/aux
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row × N] packed FP16 emin pair (Q2_K only, else NULL)
    float*           __restrict__ d_output,       // [M × N] FP32 output, row-major
    const float*     __restrict__ d_scales_A,     // [M] per-row activation scales
    int M, int N, int K,
    const float*     __restrict__ d_scales_A_blockwise, // [M × blocks_per_row] per-block activation scales (nullable)
    int serial_m1_kblocks)
{
    // -------------------------------------------------------------------------
    // Tile and thread geometry
    // -------------------------------------------------------------------------
    constexpr int KT = BPT * 8;   // k-groups per tile (each k-group = 4 INT8 elements)
    constexpr int BLOCK_SIZE = 256;
    constexpr int M_PER_THREAD = 4;
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;

    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    constexpr int PB = Traits::payload_bytes;
    constexpr int PD = Traits::payload_dwords;

    static_assert(M_TILE == 16 || M_TILE == 32 || M_TILE == 64 || M_TILE == 128,
                  "M_TILE must be 16, 32, 64, or 128");
    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT >= 8, "Need at least 1 block per tile");

    // Workgroup tile position
    const int n_block = blockIdx.x * N_TILE;
    const int m_block = blockIdx.y * M_TILE;

    // Thread position within tile
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];       // A: same layout as INT8 V3
    __shared__ int32_t b_lds[2][KT * N_TILE];       // B: decoded INT8 (same layout as V3)
    __shared__ uint16_t scale_lds[2][BPT * N_TILE];  // Raw per-block16-bit metadata
    __shared__ uint16_t aux_lds[2][Traits::needs_aux_compute ? BPT * N_TILE : 1];  // Aux scale/min
    __shared__ uint16_t iq1_m_qh_lds[
        2][FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? BPT * N_TILE : 1];
    __shared__ uint16_t emin_lo_lds[2][Traits::is_dual_scale_asym ? BPT * N_TILE : 1]; // Embedded min lo
    __shared__ uint16_t emin_hi_lds[2][Traits::is_dual_scale_asym ? BPT * N_TILE : 1]; // Embedded min hi
    // Precomputed activation sums for cooperative sum_a (Pattern DA + Pattern A).
    // Computed cooperatively during A-load: each A-load thread covers exactly
    // one block-half for one M-row (4 sdot4 per thread vs 32 sdot4/block in
    // the old per-thread approach → 64× fewer sdot4 total + 4 fewer VGPRs).
    // Layout: [block_half_idx * M_TILE + mi], block_half_idx = bp*2 + {0=lo,1=hi}
    // Pattern DA reads halves separately (lo/hi emin correction).
    // Pattern A reads and sums both halves (full-block min correction).
    __shared__ int32_t sum_a_lds[2][Traits::needs_cooperative_sum_a ? BPT * 2 * M_TILE : 1];
    // FP32 accumulators — accumulate across all K-blocks
    float acc[M_PER_THREAD][N_PER_THREAD];
    float ordered_partition_sum[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
        {
            acc[mm][nn] = 0.0f;
            ordered_partition_sum[mm][nn] = 0.0f;
        }

    // K dimensions
    const int blocks_per_row = K / 32;
    const int total_k_groups = K / 4;
    const int num_block_tiles = (blocks_per_row + BPT - 1) / BPT;
    const int blocks_per_partition =
        (blocks_per_row + serial_m1_kblocks - 1) / serial_m1_kblocks;
    int next_partition_end = blocks_per_partition;

    if (num_block_tiles == 0) return;

    // =========================================================================
    // Safe-tile split (mirrors INT8 V7's branchless interior optimization)
    //
    // Pre-compute how many tiles have ALL loads guaranteed in-bounds:
    //   - a_m_safe: all M-rows in this workgroup tile are within [0, M)
    //   - b_n_safe: all N-cols in this workgroup tile are within [0, N)
    //   - safe_k:   number of tiles with full K coverage (no partial blocks)
    //
    // For tiles 0..num_safe-1, ALL global loads (A and B) are unconditional
    // → no branch overhead in the hot loop → better IPC and icache utilization.
    // For aligned shapes (common in transformer layers), num_safe == num_block_tiles
    // → the boundary loop never executes.
    // =========================================================================
    const bool a_m_safe = (m_block + M_TILE <= M);
    const bool b_n_safe = (n_block + N_TILE <= N);
    const int  safe_k   = blocks_per_row / BPT;  // tiles with full K coverage
    // FULL_TILES is selected only by a launch-site divisibility proof. Keeping
    // that fact in the kernel type lets LLVM erase the checked-load loop and
    // scalar output tail completely. Those paths otherwise retain thread and
    // pointer geometry across the whole K loop, which creates scratch traffic
    // even though production transformer dimensions never take either edge.
    const int num_safe = FULL_TILES
        ? num_block_tiles
        : ((a_m_safe && b_n_safe)
               ? (safe_k < num_block_tiles ? safe_k : num_block_tiles)
               : 0);

    // Reinterpret A as int32_t* for vectorized loads (4 packed INT8 per int32)
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int a_row_stride = K / 4;  // k-groups per A row

    // =========================================================================
    // A cooperative load geometry (identical to INT8 V3)
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry — Decode-in-Load
    //
    // First BPT×N_TILE threads load raw payload (PB bytes each) and scales.
    // Thread mapping: (block_local, n_local) via div/mod N_TILE.
    //
    // Each thread:
    //   1. Loads PB bytes of raw payload via load_payload<PB>()
    //   2. Loads 1 FP16 scale (from d_block_scales or embedded in payload)
    //   3. Optionally loads 1 FP16 aux (from d_block_mins or payload)
    //   4. Decodes payload → 8 packed INT32 groups
    //   5. Writes 8 int32 to b_lds + scales/aux/emin to LDS
    // =========================================================================
    constexpr int B_LOAD_THREADS = BPT * N_TILE;  // 128 for BPT=2, N_TILE=64

    const int b_load_block = threadIdx.x / N_TILE;  // which block in tile (0..BPT-1)
    const int b_load_n     = threadIdx.x % N_TILE;  // which N-column (tile-local)
    const int b_load_n_global = n_block + b_load_n;  // global N-column

    // Staging registers for pipeline
    uint4 a_staging[A_NUM_PASSES];
    uint32_t b_payload_dw[PD];  // variable-size payload dwords
    uint16_t b_scale_staging;   // Original Q6 d or derived-half scale
    uint16_t b_aux_staging;     // Q6 signed coefficients or format half auxiliary
    uint16_t b_emin_lo_staging; // embedded min lo (for dual_scale_asym)
    uint16_t b_emin_hi_staging; // embedded min hi (for dual_scale_asym)

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================
    {
        const bool tile0_safe = FULL_TILES || (num_safe > 0);

        // A loads
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;
            if (tile0_safe)
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(gm) * a_row_stride + a_kg);
            }
            else if (__builtin_expect(gm < M && a_kg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(gm) * a_row_stride + a_kg);
            }
            else
            {
                a_staging[p].x = (gm < M && a_kg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 0] : 0;
                a_staging[p].y = (gm < M && a_kg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 1] : 0;
                a_staging[p].z = (gm < M && a_kg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 2] : 0;
                a_staging[p].w = (gm < M && a_kg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg + 3] : 0;
            }
        }

        // B loads
        #pragma unroll
        for (int d = 0; d < PD; ++d) b_payload_dw[d] = 0;
        b_scale_staging = 0;
        if constexpr (Traits::needs_aux_compute) b_aux_staging = 0;
        if constexpr (Traits::is_dual_scale_asym) { b_emin_lo_staging = 0; b_emin_hi_staging = 0; }
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = b_load_block;
            if (tile0_safe || __builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                load_payload<PB>(d_payload + linear * PB, b_payload_dw);
                b_scale_staging = d_block_scales[linear];
                if constexpr (Traits::needs_aux_compute)
                    b_aux_staging = d_block_mins[linear];
                if constexpr (Traits::is_dual_scale_asym) {
                    if constexpr (Traits::needs_emin_array) {
                        const uint32_t emin_packed = d_block_emins[linear];
                        b_emin_lo_staging = static_cast<uint16_t>(emin_packed);
                        b_emin_hi_staging = static_cast<uint16_t>(emin_packed >> 16);
                    } else {
                        b_emin_lo_staging = 0;
                        b_emin_hi_staging = 0;
                    }
                }
            }
        }
    }

    // Write preload to a_lds[0] + b_lds[0]
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
    {
        const int mi = a_mi[p];
        if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
        a_lds[0][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
        a_lds[0][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
        a_lds[0][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
        a_lds[0][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        if constexpr (Traits::needs_cooperative_sum_a) {
            // Cooperative sum_a: this thread's 4 k-groups = 1 block-half
            const int bhi = a_kg / 4;
            int32_t sa = ::llaminar2::rocm_integer_contract::signedDot4(
                static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0);
            sa = ::llaminar2::rocm_integer_contract::signedDot4(
                static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa);
            sa = ::llaminar2::rocm_integer_contract::signedDot4(
                static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa);
            sa = ::llaminar2::rocm_integer_contract::signedDot4(
                static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa);
            sum_a_lds[0][bhi * M_TILE + mi] = sa;
        }
    }
    if (threadIdx.x < B_LOAD_THREADS)
    {
        int32_t packed_groups[8];
        decode_block<FMT>(b_payload_dw, packed_groups);

        #pragma unroll
        for (int g = 0; g < 8; ++g)
            b_lds[0][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

        scale_lds[0][b_load_block * N_TILE + b_load_n] = b_scale_staging;
        if constexpr (Traits::needs_aux_compute)
            aux_lds[0][b_load_block * N_TILE + b_load_n] = b_aux_staging;
        if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
            // The payload remains live until decode; derive its sign bits
            // here instead of retaining a duplicate register across compute.
            iq1_m_qh_lds[0][b_load_block * N_TILE + b_load_n] =
                static_cast<uint16_t>(b_payload_dw[1]);
        if constexpr (Traits::is_dual_scale_asym) {
            emin_lo_lds[0][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
            emin_hi_lds[0][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
        }
    }
    __syncthreads();  // buf[0] ready

    // =========================================================================
    // SAFE LOOP: tiles 1..num_safe-1  (unconditional loads — no bounds checks)
    //
    // Split into a separate loop from the boundary path so the compiler can
    // fully optimize the load phase: no dead branches, no speculative zero-
    // init, better instruction scheduling.  For aligned shapes (common in
    // transformer layers), num_safe == num_block_tiles → boundary loop is
    // never entered.
    //
    // Note: despite the code duplication of the compute + LDS-write phases,
    // benchmarks show this is ~10-20% faster than a single loop with a
    // runtime branch (the branch prevents the compiler from optimizing the
    // safe load path aggressively).
    // =========================================================================
    for (int tile = 1; tile < num_safe; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int tile_block_base = tile * BPT;
        const int kt_base = tile_block_base * 8;

        // --- 1. SAFE LOAD: all loads guaranteed in-bounds ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;
            const int gkg = kt_base + a_kg;
            a_staging[p] = *reinterpret_cast<const uint4*>(
                a_global + static_cast<int64_t>(gm) * a_row_stride + gkg);
        }

        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = tile_block_base + b_load_block;
            const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
            load_payload<PB>(d_payload + linear * PB, b_payload_dw);
            b_scale_staging = d_block_scales[linear];
            if constexpr (Traits::needs_aux_compute)
                b_aux_staging = d_block_mins[linear];
            if constexpr (Traits::is_dual_scale_asym) {
                if constexpr (Traits::needs_emin_array) {
                    const uint32_t emin_packed = d_block_emins[linear];
                    b_emin_lo_staging = static_cast<uint16_t>(emin_packed);
                    b_emin_hi_staging = static_cast<uint16_t>(emin_packed >> 16);
                } else {
                    b_emin_lo_staging = 0;
                    b_emin_hi_staging = 0;
                }
            }
        }

        // The previous iteration's final CTA edge released both shared
        // slots. IQ1_M's four correction sums need the register budget that
        // next-tile operands otherwise retain across compute: publish those
        // operands directly into the independent next slot first. One final
        // CTA edge both joins publication and retires current-slot readers.
        // Other formats retain their measured register-prefetch overlap.
        if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
        {
            #pragma unroll
            for (int p = 0; p < A_NUM_PASSES; ++p)
            {
                const int mi = a_mi[p];
                if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
                a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
                a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
                a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
                a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
                if constexpr (Traits::needs_cooperative_sum_a) {
                    const int bhi = a_kg / 4;
                    int32_t sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa);
                    sum_a_lds[nxt_buf][bhi * M_TILE + mi] = sa;
                }
            }

            if (threadIdx.x < B_LOAD_THREADS)
            {
                int32_t packed_groups[8];
                decode_block<FMT>(b_payload_dw, packed_groups);

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

                scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
                if constexpr (Traits::needs_aux_compute)
                    aux_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_aux_staging;
                if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    iq1_m_qh_lds[nxt_buf][b_load_block * N_TILE + b_load_n] =
                        static_cast<uint16_t>(b_payload_dw[1]);
                if constexpr (Traits::is_dual_scale_asym) {
                    emin_lo_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
                    emin_hi_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
                }
            }
        }

        // --- 2. Compute on CURRENT buffer (contains tile-1's data) ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
        {
            float sa_blk[M_PER_THREAD];
            if (d_scales_A_blockwise) {
                const int gblk = (tile - 1) * BPT + bp;
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm) {
                    const int gm = m_block + m_base + mm;
                    sa_blk[mm] = (FULL_TILES || gm < M)
                        ? d_scales_A_blockwise[
                              static_cast<int64_t>(gm) * blocks_per_row + gblk]
                        : 0.0f;
                }
            } else {
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    sa_blk[mm] = 1.0f;
            }
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT, UNROLL_G,
                                  BLOCK_SIZE, M_PER_THREAD, MIN_BLOCKS>(
                bp, a_lds[cur_buf], b_lds[cur_buf],
                scale_lds[cur_buf], aux_lds[cur_buf],
                iq1_m_qh_lds[cur_buf],
                emin_lo_lds[cur_buf], emin_hi_lds[cur_buf],
                sum_a_lds[cur_buf],
                acc, m_base, t_n, sa_blk);
            if (serial_m1_kblocks > 1)
            {
                const int global_block = (tile - 1) * BPT + bp;
                commit_serial_m1_partition_if_complete<
                    N_TILE, M_TILE, M_PER_THREAD, N_PER_THREAD>(
                        acc,
                        ordered_partition_sum,
                        global_block,
                        blocks_per_row,
                        blocks_per_partition,
                        next_partition_end,
                        m_block + m_base,
                        M,
                        d_scales_A_blockwise ? nullptr : d_scales_A,
                        t_n);
            }
        }

        if constexpr (FMT != NATIVE_QUANTISED_GEMM_IQ1_M)
        {
            __syncthreads(); // Retire current readers before staged publication.
            // Keep this schedule local: a reference-capturing publication
            // closure extends address lifetimes in other codebooks. IQ1_M
            // alone needs the early-publication path above.
            #pragma unroll
            for (int p = 0; p < A_NUM_PASSES; ++p)
            {
                const int mi = a_mi[p];
                if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
                a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
                a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
                a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
                a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
                if constexpr (Traits::needs_cooperative_sum_a) {
                    const int bhi = a_kg / 4;
                    int32_t sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa);
                    sum_a_lds[nxt_buf][bhi * M_TILE + mi] = sa;
                }
            }

            if (threadIdx.x < B_LOAD_THREADS)
            {
                int32_t packed_groups[8];
                decode_block<FMT>(b_payload_dw, packed_groups);

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

                scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
                if constexpr (Traits::needs_aux_compute)
                    aux_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_aux_staging;
                if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    iq1_m_qh_lds[nxt_buf][b_load_block * N_TILE + b_load_n] =
                        static_cast<uint16_t>(b_payload_dw[1]);
                if constexpr (Traits::is_dual_scale_asym) {
                    emin_lo_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
                    emin_hi_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
                }
            }
        }

        // --- 5. Barrier: LDS writes visible ---
        __syncthreads();
    }

    // =========================================================================
    // BOUNDARY LOOP: tiles max(1,num_safe)..num_block_tiles-1  (checked loads)
    //
    // Handles edge tiles where M, N, or K dimensions may be out of bounds.
    // Identical compute and LDS-write phases; only the load phase adds bounds
    // checks and zero-initialization for partial tiles.
    // =========================================================================
    for (int tile = (num_safe > 1 ? num_safe : 1); tile < num_block_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int tile_block_base = tile * BPT;
        const int kt_base = tile_block_base * 8;

        // --- 1. BOUNDARY LOAD: with bounds checks ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;
            const int gkg = kt_base + a_kg;
            if (__builtin_expect(gm < M && gkg + 3 < total_k_groups, 1))
            {
                a_staging[p] = *reinterpret_cast<const uint4*>(
                    a_global + static_cast<int64_t>(gm) * a_row_stride + gkg);
            }
            else
            {
                a_staging[p].x = (gm < M && gkg + 0 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 0] : 0;
                a_staging[p].y = (gm < M && gkg + 1 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 1] : 0;
                a_staging[p].z = (gm < M && gkg + 2 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 2] : 0;
                a_staging[p].w = (gm < M && gkg + 3 < total_k_groups)
                    ? a_global[static_cast<int64_t>(gm) * a_row_stride + gkg + 3] : 0;
            }
        }

        #pragma unroll
        for (int d = 0; d < PD; ++d) b_payload_dw[d] = 0;
        b_scale_staging = 0;
        if constexpr (Traits::needs_aux_compute) b_aux_staging = 0;
        if constexpr (Traits::is_dual_scale_asym) { b_emin_lo_staging = 0; b_emin_hi_staging = 0; }
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = tile_block_base + b_load_block;
            if (__builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                load_payload<PB>(d_payload + linear * PB, b_payload_dw);
                b_scale_staging = d_block_scales[linear];
                if constexpr (Traits::needs_aux_compute)
                    b_aux_staging = d_block_mins[linear];
                if constexpr (Traits::is_dual_scale_asym) {
                    if constexpr (Traits::needs_emin_array) {
                        const uint32_t emin_packed = d_block_emins[linear];
                        b_emin_lo_staging = static_cast<uint16_t>(emin_packed);
                        b_emin_hi_staging = static_cast<uint16_t>(emin_packed >> 16);
                    } else {
                        b_emin_lo_staging = 0;
                        b_emin_hi_staging = 0;
                    }
                }
            }
        }

        // The previous iteration's final CTA edge released both shared
        // slots. IQ1_M's four correction sums need the register budget that
        // next-tile operands otherwise retain across compute: publish those
        // operands directly into the independent next slot first. One final
        // CTA edge both joins publication and retires current-slot readers.
        // Other formats retain their measured register-prefetch overlap.
        if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
        {
            #pragma unroll
            for (int p = 0; p < A_NUM_PASSES; ++p)
            {
                const int mi = a_mi[p];
                if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
                a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
                a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
                a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
                a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
                if constexpr (Traits::needs_cooperative_sum_a) {
                    const int bhi = a_kg / 4;
                    int32_t sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa);
                    sum_a_lds[nxt_buf][bhi * M_TILE + mi] = sa;
                }
            }

            if (threadIdx.x < B_LOAD_THREADS)
            {
                int32_t packed_groups[8];
                decode_block<FMT>(b_payload_dw, packed_groups);

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

                scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
                if constexpr (Traits::needs_aux_compute)
                    aux_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_aux_staging;
                if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    iq1_m_qh_lds[nxt_buf][b_load_block * N_TILE + b_load_n] =
                        static_cast<uint16_t>(b_payload_dw[1]);
                if constexpr (Traits::is_dual_scale_asym) {
                    emin_lo_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
                    emin_hi_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
                }
            }
        }

        // --- 2. Compute on CURRENT buffer (contains tile-1's data) ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
        {
            float sa_blk[M_PER_THREAD];
            if (d_scales_A_blockwise) {
                const int gblk = (tile - 1) * BPT + bp;
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm) {
                    const int gm = m_block + m_base + mm;
                    sa_blk[mm] = (FULL_TILES || gm < M)
                        ? d_scales_A_blockwise[
                              static_cast<int64_t>(gm) * blocks_per_row + gblk]
                        : 0.0f;
                }
            } else {
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    sa_blk[mm] = 1.0f;
            }
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT, UNROLL_G,
                                  BLOCK_SIZE, M_PER_THREAD, MIN_BLOCKS>(
                bp, a_lds[cur_buf], b_lds[cur_buf],
                scale_lds[cur_buf], aux_lds[cur_buf],
                iq1_m_qh_lds[cur_buf],
                emin_lo_lds[cur_buf], emin_hi_lds[cur_buf],
                sum_a_lds[cur_buf],
                acc, m_base, t_n, sa_blk);
            if (serial_m1_kblocks > 1)
            {
                const int global_block = (tile - 1) * BPT + bp;
                commit_serial_m1_partition_if_complete<
                    N_TILE, M_TILE, M_PER_THREAD, N_PER_THREAD>(
                        acc,
                        ordered_partition_sum,
                        global_block,
                        blocks_per_row,
                        blocks_per_partition,
                        next_partition_end,
                        m_block + m_base,
                        M,
                        d_scales_A_blockwise ? nullptr : d_scales_A,
                        t_n);
            }
        }

        if constexpr (FMT != NATIVE_QUANTISED_GEMM_IQ1_M)
        {
            __syncthreads(); // Retire current readers before staged publication.
            // Keep this schedule local: a reference-capturing publication
            // closure extends address lifetimes in other codebooks. IQ1_M
            // alone needs the early-publication path above.
            #pragma unroll
            for (int p = 0; p < A_NUM_PASSES; ++p)
            {
                const int mi = a_mi[p];
                if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
                a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
                a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
                a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
                a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
                if constexpr (Traits::needs_cooperative_sum_a) {
                    const int bhi = a_kg / 4;
                    int32_t sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].x), static_cast<int32_t>(0x01010101), 0);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].y), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].z), static_cast<int32_t>(0x01010101), sa);
                    sa = ::llaminar2::rocm_integer_contract::signedDot4(
                        static_cast<int32_t>(a_staging[p].w), static_cast<int32_t>(0x01010101), sa);
                    sum_a_lds[nxt_buf][bhi * M_TILE + mi] = sa;
                }
            }

            if (threadIdx.x < B_LOAD_THREADS)
            {
                int32_t packed_groups[8];
                decode_block<FMT>(b_payload_dw, packed_groups);

                #pragma unroll
                for (int g = 0; g < 8; ++g)
                    b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

                scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
                if constexpr (Traits::needs_aux_compute)
                    aux_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_aux_staging;
                if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    iq1_m_qh_lds[nxt_buf][b_load_block * N_TILE + b_load_n] =
                        static_cast<uint16_t>(b_payload_dw[1]);
                if constexpr (Traits::is_dual_scale_asym) {
                    emin_lo_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_lo_staging;
                    emin_hi_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_emin_hi_staging;
                }
            }
        }

        // --- 5. Barrier: LDS writes visible ---
        __syncthreads();
    }

    // =========================================================================
    // FINAL TILE: Compute the last tile (already in LDS)
    // =========================================================================
    {
        const int last_buf = (num_block_tiles - 1) & 1;

        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
        {
            float sa_blk[M_PER_THREAD];
            if (d_scales_A_blockwise) {
                const int gblk = (num_block_tiles - 1) * BPT + bp;
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm) {
                    const int gm = m_block + m_base + mm;
                    sa_blk[mm] = (FULL_TILES || gm < M)
                        ? d_scales_A_blockwise[
                              static_cast<int64_t>(gm) * blocks_per_row + gblk]
                        : 0.0f;
                }
            } else {
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    sa_blk[mm] = 1.0f;
            }
            accumulate_block_gemm<N_TILE, BPT, M_TILE, FMT, UNROLL_G,
                                  BLOCK_SIZE, M_PER_THREAD, MIN_BLOCKS>(
                bp, a_lds[last_buf], b_lds[last_buf],
                scale_lds[last_buf], aux_lds[last_buf],
                iq1_m_qh_lds[last_buf],
                emin_lo_lds[last_buf], emin_hi_lds[last_buf],
                sum_a_lds[last_buf],
                acc, m_base, t_n, sa_blk);
            if (serial_m1_kblocks > 1)
            {
                const int global_block =
                    (num_block_tiles - 1) * BPT + bp;
                commit_serial_m1_partition_if_complete<
                    N_TILE, M_TILE, M_PER_THREAD, N_PER_THREAD>(
                        acc,
                        ordered_partition_sum,
                        global_block,
                        blocks_per_row,
                        blocks_per_partition,
                        next_partition_end,
                        m_block + m_base,
                        M,
                        d_scales_A_blockwise ? nullptr : d_scales_A,
                        t_n);
            }
        }
    }

    if (serial_m1_kblocks > 1)
    {
        load_serial_m1_partition_sum<
            N_TILE, M_PER_THREAD, N_PER_THREAD>(
                acc,
                ordered_partition_sum,
                m_base,
                t_n);
    }

    // =========================================================================
    // STORE RESULTS: Apply scale_A and write FP32 to global memory
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;
        if constexpr (!FULL_TILES)
        {
            if (m_idx >= M)
                continue;
        }

        // Keep row scales out of the K-loop live range. The former prologue
        // array survived until this epilogue and forced three values into
        // scratch in the production Q6_K N128/M32 binary. Blockwise-scaled
        // rows already consumed their scale inside each ordered K block.
        // Partitioned whole-row execution already applied the scale at every
        // canonical boundary. A single-partition launch applies it here once.
        const float sa = d_scales_A_blockwise || serial_m1_kblocks > 1
            ? 1.0f : d_scales_A[m_idx];

        if constexpr (N_PER_THREAD >= 4)
        {
            // Vectorized store path
            if constexpr (FULL_TILES)
            {
                const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                #pragma unroll
                for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                {
                    float4 out;
                    out.x = acc[mm][chunk + 0] * sa;
                    out.y = acc[mm][chunk + 1] * sa;
                    out.z = acc[mm][chunk + 2] * sa;
                    out.w = acc[mm][chunk + 3] * sa;
                    *reinterpret_cast<float4*>(d_output + row_off + chunk) = out;
                }
            }
            else
            {
                if (n_base + N_PER_THREAD - 1 < N)
                {
                    const int64_t row_off =
                        static_cast<int64_t>(m_idx) * N + n_base;
                    #pragma unroll
                    for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                    {
                        float4 out;
                        out.x = acc[mm][chunk + 0] * sa;
                        out.y = acc[mm][chunk + 1] * sa;
                        out.z = acc[mm][chunk + 2] * sa;
                        out.w = acc[mm][chunk + 3] * sa;
                        *reinterpret_cast<float4*>(d_output + row_off + chunk) = out;
                    }
                }
                else
                {
                    // Scalar publication is reserved for a genuine N edge.
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    {
                        const int n_idx = n_base + nn;
                        if (n_idx < N)
                            d_output[
                                static_cast<int64_t>(m_idx) * N + n_idx] =
                                acc[mm][nn] * sa;
                    }
                }
            }
        }
        else
        {
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const int n_idx = n_base + nn;
                if (n_idx < N)
                    d_output[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn] * sa;
            }
        }
    }
}


// ============================================================================
// Streaming LM_Head kernel: register streaming with optional setup-only LDS
//
// Architecture: Per-thread independent streaming through K dimension.
// Each thread computes M_TILE M-rows × 1 N-column with zero synchronization.
//
// Why this is faster for LM_Head (N=151936, K=896-2048):
//   1. LM_Head is 100% COMPUTE-BOUND (BW util 5-25% at 1000 GB/s)
//   2. The cooperative LDS kernel spends ~50% of time in barriers
//      (28-64 barriers per WG at ~150 cyc each vs ~5000-10000 VALU)
//   3. Decode VALU cannot overlap with compute VALU (same pipe)
//   4. BW savings from 4.5 bpw are stranded (not BW-limited)
//
// The symmetric fast path eliminates all barriers and LDS usage:
//   - B: loaded per-thread from global memory → decode in registers
//   - A: loaded per-thread, all 256 threads read same addresses → L2 broadcast
//   - Zero __syncthreads → VALU pipeline runs uninterrupted
//   - ~50 VGPRs → 3-wave occupancy (vs 2-wave for LDS kernel)
//
// Formats with row-only correction terms use one cooperative setup phase and
// one barrier before the K loop. The K loop itself remains synchronization
// free, and no correction sum is redundantly recomputed per N column.
//
// Thread geometry (M_TILE=16):
//   Each of 256 lanes owns all sixteen rows for one output column.
//   A decoded B block is reused across those rows without a K-loop barrier.
//   Both row tiles retain the measured three-wave register budget; scalar
//   register-lane moves are legal, while actual memory spills fail the build.
//
// B data redundancy (across m-tiles): absorbed by L2 cache (4MB).
// A data broadcast: 256 threads read same address → L0/L1 cache broadcast.
// ============================================================================

/**
 * @brief Stream compact weight blocks while preserving serial-row arithmetic.
 * @tparam M_TILE Independent output rows owned by each output-column lane.
 * @tparam FMT Packed codebook and its scale/correction contract.
 * @tparam MIN_BLOCKS Minimum resident waves per SIMD used for register admission.
 * @param d_A_int8 Packed row-major activation bytes.
 * @param d_payload Compact block-major weight payload.
 * @param d_block_scales First weight scale for each block and output column.
 * @param d_block_mins Auxiliary weight scale/correction when the codebook needs it.
 * @param d_block_emins Packed asymmetric half-block corrections when required.
 * @param d_output Unique row-major FP32 destination for every output owner.
 * @param d_scales_A Whole-row activation scales used by the final publication.
 * @param d_scales_A_blockwise Optional per-block activation scale table.
 * @param M Live output-row count, including a possible final partial row tile.
 * @param N Live output-column count.
 * @param K Logical activation/weight width, divisible by the block width.
 * @param serial_m1_kblocks Canonical serial decode partition count.
 */
template <int M_TILE, NativeQuantisedGemmFormat FMT, int MIN_BLOCKS = 3>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void streaming_lm_head_kernel(
    const int8_t*    __restrict__ d_A_int8,       // [M × K] INT8 activations
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × PB] payload
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N] FP16 scales
    const uint16_t*  __restrict__ d_block_mins,   // [blocks_per_row × N] FP16 min/aux
    const uint32_t*  __restrict__ d_block_emins,  // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, else nullptr)
    float*           __restrict__ d_output,       // [M × N] FP32 output
    const float*     __restrict__ d_scales_A,     // [M] per-row activation scales
    const float*     __restrict__ d_scales_A_blockwise, // [M × blocks_per_row] per-block activation scales
    int M, int N, int K,
    int serial_m1_kblocks)
{
    static_assert(M_TILE == 8 || M_TILE == 16 || M_TILE == 32,
                  "Streaming LM_Head kernel supports M_TILE 8, 16, or 32");

    using Traits = NATIVE_QUANTISEDGemmTraits<FMT>;
    constexpr int PB = Traits::payload_bytes;

    // Thread position: 1 thread = M_TILE M-rows × 1 N-column
    const int n_col   = blockIdx.x * 256 + threadIdx.x;
    const int m_block = blockIdx.y * M_TILE;

    // Reinterpret A as int32 (4 packed INT8 per int32 k-group)
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int a_row_stride  = K / 4;   // k-groups per row
    const int blocks_per_row = K / 32;  // quantized blocks per row
    const int blocks_per_partition =
        (blocks_per_row + serial_m1_kblocks - 1) / serial_m1_kblocks;
    int next_partition_end = blocks_per_partition;

    // Bounds: if entire M_TILE is within [0, M), skip per-element checks
    const bool m_safe = (m_block + M_TILE <= M);

    // =========================================================================
    // Cooperative activation-correction sums pre-computed into LDS
    //
    // Instead of each thread computing sum_a independently (128 sdot4 per block
    // per thread — 99.6% redundant across the 256 threads in a WG), pre-compute
    // sum_a cooperatively: 256 threads share M_TILE × blocks_per_row work items.
    // This eliminates sum_a from the hot inner loop, reducing per-block VALU by
    // 45% and dropping VGPRs from ~70 (3-wave) to ~54 (4-wave).
    //
    // Pattern A LDS: M_TILE × blocks_per_row × 4 bytes.
    // IQ1_M LDS:    M_TILE × blocks_per_row × 4 groups × 4 bytes.
    // =========================================================================
    extern __shared__ int32_t sum_a_shared[];

    if constexpr (Traits::is_asymmetric)
    {
        const int total_items = M_TILE * blocks_per_row;
        for (int item = threadIdx.x; item < total_items; item += 256)
        {
            const int mm    = item / blocks_per_row;
            const int block = item % blocks_per_row;
            const int gm    = m_block + mm;
            int32_t s = 0;
            if (m_safe || gm < M) {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                    s = ::llaminar2::rocm_integer_contract::signedDot4(
                        a_global[static_cast<int64_t>(gm) * a_row_stride + block * 8 + kg],
                        static_cast<int32_t>(0x01010101), s);
            }
            sum_a_shared[item] = s;
        }
        __syncthreads();
    }
    else if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
    {
        /*
         * IQ1_M's signed one-eighth correction depends on four sums of eight
         * activation values for every row/block. Those sums are independent
         * of N, so computing them in every output-column thread would repeat
         * identical work 256 times. The workgroup computes each sum once and
         * publishes it through LDS before any thread enters the K loop.
         *
         * Layout: [row][block][group], where group selects one of the four
         * consecutive eight-value regions in the 32-value quantization block.
         */
        constexpr int kGroupsPerBlock = 4;
        constexpr int32_t kPackedOnes = 0x01010101;
        const int items_per_row = blocks_per_row * kGroupsPerBlock;
        const int total_items = M_TILE * items_per_row;
        for (int item = threadIdx.x; item < total_items; item += 256)
        {
            const int mm = item / items_per_row;
            const int row_item = item - mm * items_per_row;
            const int block = row_item / kGroupsPerBlock;
            const int group = row_item - block * kGroupsPerBlock;
            const int gm = m_block + mm;
            int32_t sum = 0;
            if (m_safe || gm < M)
            {
                const int kg0 = block * 8 + group * 2;
                const int kg1 = kg0 + 1;
                const int64_t row_base =
                    static_cast<int64_t>(gm) * a_row_stride;
                sum = ::llaminar2::rocm_integer_contract::signedDot4(
                    a_global[row_base + kg1],
                    kPackedOnes,
                    ::llaminar2::rocm_integer_contract::signedDot4(
                        a_global[row_base + kg0],
                        kPackedOnes,
                        0));
            }
            sum_a_shared[item] = sum;
        }
        __syncthreads();
    }

    // Now safe to exit early for out-of-bounds N-columns
    // (correction pre-compute must complete for ALL threads before barrier)
    if (n_col >= N) return;

    // FP32 accumulators — accumulated across all K-blocks
    float acc[M_TILE];
    float ordered_partition_sum[M_TILE];
    #pragma unroll
    for (int mm = 0; mm < M_TILE; ++mm)
    {
        acc[mm] = 0.0f;
        ordered_partition_sum[mm] = 0.0f;
    }

    // =========================================================================
    // Main loop: stream through K-blocks with zero barriers
    // =========================================================================
    for (int block = 0; block < blocks_per_row; ++block)
    {
        // Keep row-address and tail-mask lifetimes inside this block. LLVM can
        // otherwise hoist every unrolled row's 64-bit address and exec mask
        // across the entire K scan, overflowing scalar registers and then the
        // vector lanes used to save them. This instruction-free compiler edge
        // preserves the values, arithmetic, memory accesses and three-wave
        // schedule; it is neither a GPU barrier nor a memory clobber.
        int block_m_origin = m_block;
        int block_m_limit = M;
        asm volatile("" : "+s"(block_m_origin), "+s"(block_m_limit));
        const bool block_rows_complete = block_m_origin + M_TILE <= block_m_limit;

        // --- 0. Compute per-block activation scales ---
        float sa_blk[M_TILE];
        if (d_scales_A_blockwise) {
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm) {
                const int gm = block_m_origin + mm;
                sa_blk[mm] = (block_rows_complete || gm < block_m_limit) ? d_scales_A_blockwise[static_cast<int64_t>(gm) * blocks_per_row + block] : 0.0f;
            }
        } else {
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                sa_blk[mm] = 1.0f;
        }

        // --- 1. Load B payload + scale/aux for this thread's N-column ---
        const size_t linear = static_cast<size_t>(block) * N + n_col;
        uint32_t b_payload_dw[Traits::payload_dwords];
        load_payload<PB>(d_payload + linear * PB, b_payload_dw);

        uint16_t scale_bits = d_block_scales[linear];

        uint16_t aux_bits = 0;
        if constexpr (Traits::needs_aux_compute)
            aux_bits = d_block_mins[linear];

        uint16_t emin_lo_bits = 0, emin_hi_bits = 0;
        if constexpr (Traits::is_dual_scale_asym) {
            if constexpr (Traits::needs_emin_array) {
                const uint32_t emin_packed = d_block_emins[linear];
                emin_lo_bits = static_cast<uint16_t>(emin_packed);
                emin_hi_bits = static_cast<uint16_t>(emin_packed >> 16);
            }
        }

        // --- 2. Decode B → 8 INT32 packed groups ---
        int32_t packed[8];
        decode_block<FMT>(b_payload_dw, packed);

        // --- 3. Accumulate based on format pattern ---
        if constexpr (Traits::is_dual_scale_asym)
        {
            // Pattern DA: two-phase accumulation (Q2_K only)
            // Single-path with ternary bounds check — avoids m_safe code
            // duplication that would push VGPR past 64 (losing 4-wave occupancy).
            const int kg_base = block * 8;

            // Lo half
            int32_t temp_acc[M_TILE], temp_sum[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                temp_acc[mm] = 0; temp_sum[mm] = 0;
            }
            #pragma unroll
            for (int kg = 0; kg < 4; ++kg)
            {
                const int32_t b_val = packed[kg];
                const int a_kg_idx = kg_base + kg;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    const int gm = block_m_origin + mm;
                    const int32_t a_val = (block_rows_complete || gm < block_m_limit)
                        ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                    temp_acc[mm] = ::llaminar2::rocm_integer_contract::signedDot4(a_val, b_val, temp_acc[mm]);
                    temp_sum[mm] = ::llaminar2::rocm_integer_contract::signedDot4(
                        a_val, static_cast<int32_t>(0x01010101), temp_sum[mm]);
                }
            }
            const float slo = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            const float elo = __half2float(*reinterpret_cast<const __half*>(&emin_lo_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]) * sa_blk[mm], slo, acc[mm]);
                acc[mm] = __fmaf_rn(static_cast<float>(temp_sum[mm]) * sa_blk[mm], elo, acc[mm]);
            }

            // Hi half (reuse temp_acc, temp_sum)
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                temp_acc[mm] = 0; temp_sum[mm] = 0;
            }
            #pragma unroll
            for (int kg = 4; kg < 8; ++kg)
            {
                const int32_t b_val = packed[kg];
                const int a_kg_idx = kg_base + kg;
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    const int gm = block_m_origin + mm;
                    const int32_t a_val = (block_rows_complete || gm < block_m_limit)
                        ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                    temp_acc[mm] = ::llaminar2::rocm_integer_contract::signedDot4(a_val, b_val, temp_acc[mm]);
                    temp_sum[mm] = ::llaminar2::rocm_integer_contract::signedDot4(
                        a_val, static_cast<int32_t>(0x01010101), temp_sum[mm]);
                }
            }
            const float shi = __half2float(*reinterpret_cast<const __half*>(&aux_bits));
            const float ehi = __half2float(*reinterpret_cast<const __half*>(&emin_hi_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] = __fmaf_rn(static_cast<float>(temp_acc[mm]) * sa_blk[mm], shi, acc[mm]);
                acc[mm] = __fmaf_rn(static_cast<float>(temp_sum[mm]) * sa_blk[mm], ehi, acc[mm]);
            }
        }
        else if constexpr (Traits::is_dual_scale)
        {
            // Pattern D: retain both half-dots until one serial-order update.
            // Retire four independent rows before preparing the next group.
            // Weight operands stay shared by the complete row tile, but the
            // two integer dot arrays and correction inputs need not all be
            // live together. Every row still sees the same eight dot4 terms
            // and exact FP32 contribution tree in this one grouped kernel.
            constexpr int kRowsPerCommit = M_TILE < 4 ? M_TILE : 4;
            const int kg_base = block * 8;
            #pragma unroll
            for (int row_base = 0; row_base < M_TILE; row_base += kRowsPerCommit)
            {
                int32_t low_dot[kRowsPerCommit];
                int32_t high_dot[kRowsPerCommit];
                #pragma unroll
                for (int row = 0; row < kRowsPerCommit; ++row)
                {
                    low_dot[row] = 0;
                    high_dot[row] = 0;
                }
                if (block_rows_complete)
                {
                    #pragma unroll
                    for (int kg = 0; kg < 4; ++kg)
                    {
                        const int32_t b_val = packed[kg];
                        const int a_kg_idx = kg_base + kg;
                        #pragma unroll
                        for (int row = 0; row < kRowsPerCommit; ++row)
                        {
                            const int mm = row_base + row;
                            const int32_t a_val = a_global[
                                static_cast<int64_t>(block_m_origin + mm) * a_row_stride + a_kg_idx];
                            low_dot[row] = ::llaminar2::rocm_integer_contract::signedDot4(
                                a_val, b_val, low_dot[row]);
                        }
                    }
                }
                else
                {
                    #pragma unroll
                    for (int kg = 0; kg < 4; ++kg)
                    {
                        const int32_t b_val = packed[kg];
                        const int a_kg_idx = kg_base + kg;
                        #pragma unroll
                        for (int row = 0; row < kRowsPerCommit; ++row)
                        {
                            const int mm = row_base + row;
                            const int gm = block_m_origin + mm;
                            const int32_t a_val = (gm < block_m_limit)
                                ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                            low_dot[row] = ::llaminar2::rocm_integer_contract::signedDot4(
                                a_val, b_val, low_dot[row]);
                        }
                    }
                }

                if (block_rows_complete)
                {
                    #pragma unroll
                    for (int kg = 4; kg < 8; ++kg)
                    {
                        const int32_t b_val = packed[kg];
                        const int a_kg_idx = kg_base + kg;
                        #pragma unroll
                        for (int row = 0; row < kRowsPerCommit; ++row)
                        {
                            const int mm = row_base + row;
                            const int32_t a_val = a_global[
                                static_cast<int64_t>(block_m_origin + mm) * a_row_stride + a_kg_idx];
                            high_dot[row] = ::llaminar2::rocm_integer_contract::signedDot4(
                                a_val, b_val, high_dot[row]);
                        }
                    }
                }
                else
                {
                    #pragma unroll
                    for (int kg = 4; kg < 8; ++kg)
                    {
                        const int32_t b_val = packed[kg];
                        const int a_kg_idx = kg_base + kg;
                        #pragma unroll
                        for (int row = 0; row < kRowsPerCommit; ++row)
                        {
                            const int mm = row_base + row;
                            const int gm = block_m_origin + mm;
                            const int32_t a_val = (gm < block_m_limit)
                                ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                            high_dot[row] = ::llaminar2::rocm_integer_contract::signedDot4(
                                a_val, b_val, high_dot[row]);
                        }
                    }
                }

                const auto weight_scales = rocm_native_quantised_load_dual_weight_scales<FMT>(
                    scale_bits, aux_bits);
                const float slo = weight_scales.low;
                const float shi = weight_scales.high;
                uint16_t iq1_m_qh = 0;
                if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    iq1_m_qh = static_cast<uint16_t>(b_payload_dw[1]);
                #pragma unroll
                for (int row = 0; row < kRowsPerCommit; ++row)
                {
                    const int mm = row_base + row;
                    if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M)
                    {
                        constexpr int kGroupsPerBlock = 4;
                        constexpr float kDelta = 0.125f;
                        int sum_base =
                            (mm * blocks_per_row + block) * kGroupsPerBlock;
                        // Four LDS corrections belong to this row's commit, not
                        // to all unrolled rows simultaneously. Keep the uniform
                        // address local to this instruction-free compiler edge.
                        asm volatile("" : "+s"(sum_base));
                        const uint8_t qh0 = static_cast<uint8_t>(iq1_m_qh);
                        const uint8_t qh1 =
                            static_cast<uint8_t>(iq1_m_qh >> 8);
                        acc[mm] = rocm_native_quantised_commit_iq1_m_block(
                            acc[mm],
                            low_dot[row],
                            high_dot[row],
                            sum_a_shared[sum_base],
                            sum_a_shared[sum_base + 1],
                            sum_a_shared[sum_base + 2],
                            sum_a_shared[sum_base + 3],
                            (qh0 & 0x08) ? -kDelta : kDelta,
                            (qh0 & 0x80) ? -kDelta : kDelta,
                            (qh1 & 0x08) ? -kDelta : kDelta,
                            (qh1 & 0x80) ? -kDelta : kDelta,
                            slo,
                            shi,
                            sa_blk[mm]);
                    }
                    else
                    {
                        acc[mm] = rocm_native_quantised_commit_dual_scale_block(
                            acc[mm],
                            low_dot[row],
                            high_dot[row],
                            slo,
                            shi,
                            sa_blk[mm]);
                    }
                }
            }
        }
        else if constexpr (Traits::is_asymmetric)
        {
            // Pattern A: asymmetric single-scale (Q4_1, Q5_1, IQ1_S)
            //
            // sum_a is pre-computed in LDS (cooperative phase above).
            // This eliminates M_TILE × 8 redundant sdot4 per block per thread
            // and drops the VGPR count from ~70 (3-wave) to ~54 (4-wave).
            int32_t block_acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                block_acc[mm] = 0;
            const int kg_base = block * 8;
            if (block_rows_complete)
            {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int32_t a_val = a_global[
                            static_cast<int64_t>(block_m_origin + mm) * a_row_stride + a_kg_idx];
                        block_acc[mm] = ::llaminar2::rocm_integer_contract::signedDot4(a_val, b_val, block_acc[mm]);
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int gm = block_m_origin + mm;
                        const int32_t a_val = (gm < block_m_limit)
                            ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                        block_acc[mm] = ::llaminar2::rocm_integer_contract::signedDot4(a_val, b_val, block_acc[mm]);
                    }
                }
            }
            const float bs = __half2float(*reinterpret_cast<const __half*>(&scale_bits));
            const float bm = __half2float(*reinterpret_cast<const __half*>(&aux_bits));
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
            {
                acc[mm] =
                    rocm_native_quantised_commit_corrected_single_scale_block(
                        acc[mm],
                        block_acc[mm],
                        bs,
                        sum_a_shared[mm * blocks_per_row + block],
                        bm,
                        sa_blk[mm]);
            }
        }
        else
        {
            // Pattern S: simple symmetric (Q4_0, IQ4_NL, Q5_0, ...)
            int32_t block_acc[M_TILE];
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                block_acc[mm] = 0;
            const int kg_base = block * 8;
            if (block_rows_complete)
            {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int32_t a_val = a_global[
                            static_cast<int64_t>(block_m_origin + mm) * a_row_stride + a_kg_idx];
                        block_acc[mm] = ::llaminar2::rocm_integer_contract::signedDot4(
                            a_val, b_val, block_acc[mm]);
                    }
                }
            }
            else
            {
                #pragma unroll
                for (int kg = 0; kg < 8; ++kg)
                {
                    const int32_t b_val = packed[kg];
                    const int a_kg_idx = kg_base + kg;
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int gm = block_m_origin + mm;
                        const int32_t a_val = (gm < block_m_limit)
                            ? a_global[static_cast<int64_t>(gm) * a_row_stride + a_kg_idx] : 0;
                        block_acc[mm] = ::llaminar2::rocm_integer_contract::signedDot4(
                            a_val, b_val, block_acc[mm]);
                    }
                }
            }
            const float block_scale = rocm_native_quantised_load_single_weight_scale<FMT>(scale_bits, aux_bits);
            #pragma unroll
            for (int mm = 0; mm < M_TILE; ++mm)
                acc[mm] = rocm_native_quantised_commit_single_scale_block(
                    acc[mm],
                    block_acc[mm],
                    block_scale,
                    sa_blk[mm]);
        }

        /*
         * Reproduce serial M=1's split-K reducer inside each output owner.
         * A streaming thread owns one N column and all M_TILE rows, so both
         * the current partition and ordered sum remain private registers.
         */
        if (serial_m1_kblocks > 1)
        {
            const int completed_blocks = block + 1;
            const bool partition_complete =
                completed_blocks == blocks_per_row ||
                completed_blocks == next_partition_end;
            if (partition_complete)
            {
                // The streaming and cooperative families retain identical
                // ascending partition boundaries, including the final tail.
                next_partition_end += blocks_per_partition;
                if (!d_scales_A_blockwise)
                {
                    // Whole-row serial decode scales each completed partial,
                    // not the final reduction; retain its rounding boundary.
                    #pragma unroll
                    for (int mm = 0; mm < M_TILE; ++mm)
                    {
                        const int global_row = m_block + mm;
                        if (global_row < M)
                            acc[mm] = __fmul_rn(
                                acc[mm], d_scales_A[global_row]);
                    }
                }
                #pragma unroll
                for (int mm = 0; mm < M_TILE; ++mm)
                {
                    // Include the serial reducer's initial positive-zero
                    // addition, just as the cooperative family does.
                    ordered_partition_sum[mm] = __fadd_rn(
                        ordered_partition_sum[mm], acc[mm]);
                    acc[mm] = 0.0f;
                }
            }
        }
    }

    if (serial_m1_kblocks > 1)
    {
        #pragma unroll
        for (int mm = 0; mm < M_TILE; ++mm)
            acc[mm] = ordered_partition_sum[mm];
    }

    // =========================================================================
    // STORE: Apply scale_A and write FP32 output
    // =========================================================================
    if (m_safe)
    {
        #pragma unroll
        for (int mm = 0; mm < M_TILE; ++mm)
        {
            const int gm = m_block + mm;
            const float sa = d_scales_A_blockwise || serial_m1_kblocks > 1
                ? 1.0f : d_scales_A[gm];
            d_output[static_cast<int64_t>(gm) * N + n_col] =
                acc[mm] * sa;
        }
    }
    else
    {
        #pragma unroll
        for (int mm = 0; mm < M_TILE; ++mm)
        {
            const int gm = m_block + mm;
            if (gm < M) {
                const float sa = d_scales_A_blockwise || serial_m1_kblocks > 1
                    ? 1.0f : d_scales_A[gm];
                d_output[static_cast<int64_t>(gm) * N + n_col] =
                    acc[mm] * sa;
            }
        }
    }
}


// ============================================================================
// Tile dispatch: select N_TILE (64 or 128) and M_TILE for minimal waste
//
// N_TILE selection mirrors INT8 V3/V7 strategy:
//   K >= N (K-heavy: AttnOut, FFN_Dn) → N_TILE=64  (deeper K iteration)
//   K <  N (N-heavy: FFN_Up, LM_Head) → N_TILE=128 (wider N coverage)
//
// N_TILE=128 benefits for N-heavy shapes:
//   - ALL 256 threads participate in B cooperative loading (vs 128 at N64)
//   - Half the grid in N dimension → fewer kernel launches + barriers
//   - N_PER_THREAD doubles (4→8 at M64) → better compute density
//   - LDS budget: 25.6KB at M64 → 2 WGs/CU, occupancy 2 maintained
//
// M_TILE selection uses V7's minimum-waste strategy: pick the M_TILE
// that minimizes padding waste for the given M value.
// ============================================================================

template <NativeQuantisedGemmFormat FMT>
static bool launch_native_quantised_gemm(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const uint16_t* d_block_scales,
    const uint16_t* d_block_mins,
    const uint32_t* d_block_emins,
    float* d_output,
    const float* d_scales_A,
    const float* d_scales_A_blockwise,
    int M, int N, int K,
    int device_id, void* stream,
    uint8_t arithmetic_policy_codebook_id)
{
    if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ3_S || FMT == NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE)
        if (!d_block_mins) return false;
    constexpr int BPT = 2;
    constexpr int BLOCK_SIZE = 256;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    const bool stream_capture_active = [&]() {
        if (llaminar2::isGraphCaptureActive())
            return true;
        if (!hip_stream)
            return false;
        hipStreamCaptureStatus status = hipStreamCaptureStatusNone;
        const hipError_t err = hipStreamIsCapturing(hip_stream, &status);
        return err == hipSuccess && status == hipStreamCaptureStatusActive;
    }();

    /*
     * Batch invariance is an arithmetic launch property, not a debug mode.
     * Resolve the exact split count used by serial M=1 before choosing a tile
     * family. Every cooperative family below replays those K boundaries
     * locally, so changing M or N_TILE cannot change a published row.
     */
    const auto& rocm_env = llaminar2::debugEnv().rocm;
    int serial_m1_kblocks = rocm_env.native_quantised_gemv_kb;
    if (serial_m1_kblocks <= 0)
    {
        int ignored_target_waves = 0;
        if (!rocmGemv_native_quantised_query_serial_m1_config_with_policy(
                arithmetic_policy_codebook_id,
                N,
                K,
                &serial_m1_kblocks,
                &ignored_target_waves))
        {
            fprintf(
                stderr,
                "[rocmGemm_native_quantised_fp32] no serial M=1 split-K policy "
                "for batch-invariant prefill (M=%d N=%d K=%d "
                "codebook=%u arithmetic_policy_codebook=%u)\n",
                M,
                N,
                K,
                static_cast<unsigned>(FMT),
                static_cast<unsigned>(arithmetic_policy_codebook_id));
            return false;
        }
    }
    serial_m1_kblocks =
        min(max(serial_m1_kblocks, 1), K / 32);

    auto report_launch_status = [&](const char* route,
                                    int n_tile,
                                    int m_tile,
                                    int min_blocks,
                                    int unroll,
                                    hipError_t err) -> bool
    {
        if (err != hipSuccess)
        {
            fprintf(stderr,
                    "[rocmGemm_native_quantised_fp32] launch failed: %s "
                    "route=%s M=%d N=%d K=%d codebook=%u N_TILE=%d M_TILE=%d "
                    "min_blocks=%d unroll=%d device=%d stream=%p\n",
                    hipGetErrorString(err),
                    route ? route : "unknown",
                    M, N, K,
                    static_cast<unsigned>(FMT),
                    n_tile,
                    m_tile,
                    min_blocks,
                    unroll,
                    device_id,
                    stream);
        }
        return err == hipSuccess;
    };

    auto clear_launch_error = [&]() {
        if (!stream_capture_active)
            (void)hipGetLastError();
    };

    auto complete_launch = [&](llaminar2::rocm::VNNIPrefillProducer producer,
                               const char* route,
                               int n_tile,
                               int m_tile,
                               int min_blocks,
                               int unroll,
                               bool full_tiles,
                               const void* function,
                               size_t dynamic_shared_bytes = 0,
                               int block_threads = BLOCK_SIZE) -> bool
    {
        llaminar2::rocm::publishVNNIPrefillLaunch({
            .producer = producer,
            .execution_codebook = static_cast<uint8_t>(FMT),
            .n_tile = n_tile,
            .m_tile = m_tile,
            .min_blocks = min_blocks,
            .unroll = unroll,
            .full_tiles = full_tiles,
            .function = function,
            .block_threads = block_threads,
            .dynamic_shared_memory_bytes = dynamic_shared_bytes,
        });
        if (stream_capture_active)
            return true;
        return report_launch_status(
            route,
            n_tile,
            m_tile,
            min_blocks,
            unroll,
            hipGetLastError());
    };

    // Source policy stays here; the dedicated TU owns its native wave64 layout.
    auto launch_wmma = [&](llaminar2::rocm::NativeQuantisedWmmaTile tile) -> bool
    {
        const auto hardware = llaminar2::rocm::nativeQuantisedHardwareForDevice(device_id);
        if (hardware.architecture != llaminar2::native_quantised::NativeQuantisedArchitecture::rocm(11, 0, 0))
        {
            fprintf(stderr, "NativeQuantised INT8 WMMA requires actual gfx1100 hardware\n");
            return false;
        }
        clear_launch_error();
        const auto submission = llaminar2::rocm::submitNativeQuantisedWmma(tile, {
            .activations = d_A_int8, .payload = d_payload,
            .scales = d_block_scales, .auxiliary = d_block_mins,
            .minima_pairs = d_block_emins, .output = d_output,
            .row_scales = d_scales_A, .block_scales = d_scales_A_blockwise,
            .m = M, .n = N, .k = K, .serial_partitions = serial_m1_kblocks,
            .codebook = static_cast<uint8_t>(FMT)}, hip_stream);
        return complete_launch(submission.producer, "native_wmma", submission.n_tile,
            submission.m_tile, submission.min_blocks, submission.unroll,
            submission.full_tiles, submission.function,
            submission.dynamic_shared_memory_bytes, submission.block_threads);
    };
    if (const auto wmma = llaminar2::rocm::nativeQuantisedWmmaOverride())
        return launch_wmma(*wmma);

    auto launch = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr bool Q6_CHECKED_MB2_SPILLS =
            FMT == NATIVE_QUANTISED_GEMM_Q6_K &&
            ((NT == 64 && MT == 64) || (NT == 128 && MT == 32));
        constexpr int MB = Q6_CHECKED_MB2_SPILLS ? 1 : 2;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
            native_quantised_gemm_kernel<NT, BPT, MT, FMT, MB, 4, false>));

        clear_launch_error();
        native_quantised_gemm_kernel<NT, BPT, MT, FMT, MB, 4, false>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                d_output, d_scales_A, M, N, K, d_scales_A_blockwise,
                serial_m1_kblocks);

        return complete_launch(
            llaminar2::rocm::VNNIPrefillProducer::NativeCooperative,
            Q6_CHECKED_MB2_SPILLS ? "q6_checked_mb1" : "default",
            NT, MT, MB, 4, /*full_tiles=*/false, function);
    };

    // Small-grid occupancy policy. IQ1_M's four correction sums cannot fit
    // the 84-VGPR three-wave ceiling: request two resident waves explicitly
    // instead of asking the compiler to manufacture occupancy with scratch.
    auto launch3 = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr int kMinimumWaves = FMT == NATIVE_QUANTISED_GEMM_IQ1_M ? 2 : 3;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);
        const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
            native_quantised_gemm_kernel<NT, BPT, MT, FMT, kMinimumWaves>));

        clear_launch_error();
        native_quantised_gemm_kernel<NT, BPT, MT, FMT, kMinimumWaves>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                d_output, d_scales_A, M, N, K, d_scales_A_blockwise,
                serial_m1_kblocks);

        return complete_launch(
            llaminar2::rocm::VNNIPrefillProducer::NativeCooperative,
            "small_grid", NT, MT, kMinimumWaves, 4,
            /*full_tiles=*/false, function);
    };

    // Streaming LM_Head variant: synchronization-free K loop and three-wave
    // register budget. Correction formats use setup-only LDS.
    // (Pattern A uses small LDS for sum_a precompute → 4-wave achievable)
    auto launch_streaming = [&](auto m_tile_tag) -> bool
    {
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr int N_PER_WG = 256;  // one complete row tile per output-column lane
        // HIP's second launch bound is waves per SIMD, not CTAs per CU.
        // Preserve the measured three-wave schedule for reachable streaming
        // formats; register-bank moves do not justify reducing occupancy.
        constexpr int kMinimumWaves = 3;
        dim3 grid((N + N_PER_WG - 1) / N_PER_WG, (M + MT - 1) / MT, 1);
        dim3 block(256, 1, 1);

        // Row-only correction sums are computed once per workgroup. Pattern A
        // stores one full-block sum; IQ1_M stores four eight-value group sums.
        const int bpr = K / 32;
        const size_t sum_a_bytes =
            NATIVE_QUANTISEDGemmTraits<FMT>::is_asymmetric
                ? (MT * bpr * sizeof(int32_t))
                : FMT == NATIVE_QUANTISED_GEMM_IQ1_M
                    ? (MT * bpr * 4 * sizeof(int32_t))
                : 0;
        const size_t shared_bytes = sum_a_bytes;
        const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
            streaming_lm_head_kernel<MT, FMT, kMinimumWaves>));

        clear_launch_error();
        streaming_lm_head_kernel<MT, FMT, kMinimumWaves>
            <<<grid, block, shared_bytes, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                d_output, d_scales_A, d_scales_A_blockwise, M, N, K,
                serial_m1_kblocks);

        return complete_launch(
            llaminar2::rocm::VNNIPrefillProducer::NativeStreaming,
            "streaming_lm_head",
            N_PER_WG,
            MT,
            kMinimumWaves,
            0,
            /*full_tiles=*/false,
            function,
            shared_bytes);
    };

    auto launch_exact_config = [&](auto n_tile_tag,
                                   auto m_tile_tag,
                                   auto min_blocks_tag,
                                   auto unroll_tag,
                                   auto full_tiles_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        constexpr int MB = decltype(min_blocks_tag)::value;
        constexpr int UG = decltype(unroll_tag)::value;
        constexpr bool FULL_TILES = decltype(full_tiles_tag)::value;
        constexpr bool Q6_CHECKED_SPILLS =
            FMT == NATIVE_QUANTISED_GEMM_Q6_K && !FULL_TILES && MB == 2 &&
            ((NT == 64 && MT == 64) || (NT == 128 && MT == 32));
        if constexpr (Q6_CHECKED_SPILLS)
        {
            // This source-level rejection prevents the known spilling Q6_K
            // specialization from becoming a launchable override or even a
            // compiled dispatch target. The trainer inventory mirrors it.
            return false;
        }
        else
        {
            const dim3 grid(
                FULL_TILES ? N / NT : (N + NT - 1) / NT,
                FULL_TILES ? M / MT : (M + MT - 1) / MT,
                1);
            const dim3 block(BLOCK_SIZE, 1, 1);
            const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                native_quantised_gemm_kernel<
                    NT, BPT, MT, FMT, MB, UG, FULL_TILES>));
            clear_launch_error();
            native_quantised_gemm_kernel<NT, BPT, MT, FMT, MB, UG, FULL_TILES>
                <<<grid, block, 0, hip_stream>>>(
                    d_A_int8,
                    d_payload,
                    d_block_scales,
                    d_block_mins,
                    d_block_emins,
                    d_output,
                    d_scales_A,
                    M,
                    N,
                    K,
                    d_scales_A_blockwise,
                    serial_m1_kblocks);
            return complete_launch(
                llaminar2::rocm::VNNIPrefillProducer::NativeCooperative,
                FULL_TILES ? "configured_full_tiles" : "configured_checked",
                NT,
                MT,
                MB,
                UG,
                FULL_TILES,
                function);
        }
    };

    auto launch_config = [&](int n_tile,
                             int m_tile_cfg,
                             int min_blocks,
                             int unroll,
                             bool full_tiles) -> bool
    {
        if ((min_blocks != 1 && min_blocks != 2) ||
            (unroll != 0 && unroll != 1 && unroll != 2 && unroll != 4))
        {
            return false;
        }
        if (full_tiles &&
            ((M % m_tile_cfg) != 0 || (N % n_tile) != 0 ||
             (K % (BPT * 32)) != 0))
        {
            return false;
        }

        auto dispatch_unroll = [&](auto nt, auto mt, auto mb, auto full)
        {
            switch (unroll)
            {
            case 0:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 0>{}, full);
            case 1:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 1>{}, full);
            case 2:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 2>{}, full);
            default:
                return launch_exact_config(
                    nt, mt, mb, std::integral_constant<int, 4>{}, full);
            }
        };

        auto dispatch_min_blocks = [&](auto nt, auto mt, auto full)
        {
            if (min_blocks == 1)
                return dispatch_unroll(
                    nt, mt, std::integral_constant<int, 1>{}, full);
            return dispatch_unroll(
                nt, mt, std::integral_constant<int, 2>{}, full);
        };

        auto dispatch_m_tile = [&](auto nt, auto full)
        {
            if (m_tile_cfg == 16)
                return dispatch_min_blocks(
                    nt, std::integral_constant<int, 16>{}, full);
            if (m_tile_cfg == 32)
                return dispatch_min_blocks(
                    nt, std::integral_constant<int, 32>{}, full);
            // N128/M64 is not a supported launch geometry. A runtime test of
            // n_tile still instantiated that dead specialization, including
            // its overflowing register tile. Make the template inventory match
            // the admitted runtime shapes instead of compiling unreachable code.
            if constexpr (decltype(nt)::value == 64 && !decltype(full)::value)
            {
                if (m_tile_cfg == 64 && !full_tiles)
                    return dispatch_min_blocks(
                        nt, std::integral_constant<int, 64>{}, full);
            }
            return false;
        };

        auto dispatch_n_tile = [&](auto full)
        {
            if (n_tile == 64)
                return dispatch_m_tile(
                    std::integral_constant<int, 64>{}, full);
            if (n_tile == 128 && m_tile_cfg != 64)
                return dispatch_m_tile(
                    std::integral_constant<int, 128>{}, full);
            return false;
        };

        if (full_tiles)
        {
            // The extra proof-carrying template surface is currently compiled
            // only for Q6_K, the production hotspot that justified it.
            if constexpr (FMT == NATIVE_QUANTISED_GEMM_Q6_K)
                return dispatch_n_tile(std::true_type{});
            return false;
        }
        return dispatch_n_tile(std::false_type{});
    };

    // M_TILE selection: minimize row waste (same as INT8 V7)
    auto select_m_tile = [](int rows) -> int {
        int best_mt = 64;
        int best_waste = ((rows + 63) / 64) * 64 - rows;
        for (int mt : {32, 16}) {
            int waste = ((rows + mt - 1) / mt) * mt - rows;
            if (waste < best_waste) {
                best_waste = waste;
                best_mt = mt;
            }
        }
        return best_mt;
    };

    const int m_tile = (M <= 64) ? select_m_tile(M) : 64;

    // N_TILE selection: N-heavy → N128 (wider coverage), K-heavy → N64 (deeper K)
    //
    // N128 dispatch uses a HYBRID tile strategy:
    //   - M16 tiles with 3-wave occupancy (84 VGPRs) when GPU is undersaturated
    //   - M32 tiles with 2-wave occupancy (128 VGPRs) when GPU is saturated
    //
    // M16/3-wave halves accumulators (16→8/thread) for 50% more waves/SIMD.
    // This wins when total WG count is small (GPU can't fill all SIMDs at
    // 2-wave). For large grids, M32/2-wave has 2× better A-tile reuse per
    // WG and dominates once all SIMDs are already occupied.
    //
    // Threshold: total M16 WGs ≤ 256 (~1 WG/SIMD on MI50/60: 60 CUs × 4).
    // Below this, 3-wave occupancy provides the biggest speedup over INT8.
    // Above this, M32 tile efficiency wins despite lower occupancy.
    const bool use_n128 = (N > K);

    // =========================================================================
    // ENV VAR OVERRIDE DISPATCH (for tuning sweeps)
    //
    // When any LLAMINAR_ROCM_NATIVE_QUANTISED_* override is set, bypass the default
    // heuristic and launch the exact kernel variant specified. Full-tile
    // specializations are compiled only for Q6_K; checked candidates retain
    // the existing all-format template surface.
    // =========================================================================
    const auto& env = llaminar2::debugEnv().rocm;
    const bool has_overrides = (env.native_quantised_mt > 0 || env.native_quantised_unroll >= 0 ||
                                env.native_quantised_min_blocks > 0 ||
                                env.native_quantised_force_n64 || env.native_quantised_force_n128 ||
                                env.native_quantised_full_tiles >= 0);
    if (has_overrides) {
        int eff_mt = (env.native_quantised_mt > 0) ? env.native_quantised_mt : m_tile;
        const int eff_unroll = (env.native_quantised_unroll >= 0) ? env.native_quantised_unroll : 4;
        const int eff_min_blocks = (env.native_quantised_min_blocks > 0) ? env.native_quantised_min_blocks : 2;
        const bool eff_n128 = env.native_quantised_force_n128 ? true :
                              env.native_quantised_force_n64 ? false : use_n128;
        if (eff_n128 && eff_mt > 32)
            eff_mt = 32;
        return launch_config(
            eff_n128 ? 128 : 64,
            eff_mt,
            eff_min_blocks,
            eff_unroll,
            env.native_quantised_full_tiles == 1);
    }

    // Exact entries come only from the common, complete all-format tournament.
    // Diagnostic overrides above remain authoritative. Every unseen geometry
    // continues through the same total generic dispatch below; an admitted
    // measured producer that cannot launch is an error, never a second choice.
    const auto actual_hardware = llaminar2::rocm::nativeQuantisedHardwareForDevice(device_id);
    const auto measured = [&]() {
        auto exact = llaminar2::rocm::selectDensePrefillExactConfig(
            static_cast<uint8_t>(FMT), M, N, K,
            llaminar2::rocm::DensePrefillWeightFamily::NativeQuantised, &actual_hardware);
        if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE)
        {
            // Historical source4 rows predate physical24. Preserve those
            // unqualified records only; new physical4 measurements cannot
            // qualify a different decoder or its source-factor metadata.
            if (!exact) exact = llaminar2::rocm::selectDensePrefillExactConfig(
                uint8_t{4}, M, N, K, llaminar2::rocm::DensePrefillWeightFamily::NativeQuantised);
        }
        return exact;
    }();
    if (const auto exact = measured)
    {
        switch (exact->producer)
        {
        case llaminar2::rocm::VNNIPrefillProducer::NativeWmma:
            return launch_wmma(exact->n_tile == 128 ? llaminar2::rocm::NativeQuantisedWmmaTile::M64N128 :
                exact->n_tile == 32 ? llaminar2::rocm::NativeQuantisedWmmaTile::M32N32 :
                exact->m_tile == 32 ? llaminar2::rocm::NativeQuantisedWmmaTile::M32N64 :
                llaminar2::rocm::NativeQuantisedWmmaTile::M64N64);
        case llaminar2::rocm::VNNIPrefillProducer::NativeCooperative:
            return launch_config(exact->n_tile, exact->m_tile,
                                 exact->min_blocks, exact->unroll,
                                 exact->full_tiles);
        case llaminar2::rocm::VNNIPrefillProducer::NativeStreaming:
            // Compile exactly the existing spill-qualified streaming surface.
            // A runtime branch must not instantiate forbidden wide row tiles.
            if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::prefer_streaming)
            {
                if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale)
                    return exact->m_tile == 8
                        ? launch_streaming(std::integral_constant<int, 8>{}) : false;
                else
                    return exact->m_tile == 16
                        ? launch_streaming(std::integral_constant<int, 16>{}) : false;
            }
            else if constexpr (FMT == NATIVE_QUANTISED_GEMM_Q4_1 || FMT == NATIVE_QUANTISED_GEMM_IQ1_S)
                return exact->m_tile == 16
                    ? launch_streaming(std::integral_constant<int, 16>{}) : false;
            return false;
        default:
            return false;
        }
    }

    // =========================================================================
    // PATTERN S OPTIMIZED DISPATCH (Q4_0, IQ4_NL)
    //
    // Sweep results (gfx906, MI50/60) show N128/MT32/MB1/UG2 wins 41/45
    // shape×M combinations across Qwen2.5-0.5B/3B/7B.  MIN_BLOCKS=1
    // eliminates VGPR spills (same insight as INT8-VNNI), and UNROLL_G=2
    // reduces register pressure vs the default UG=4.
    //
    // Exceptions at small N (≤1024) / low M:
    //   - 0.5B_AttnOut (N=896):  N64/MT16/MB2 wins at M=128 (+19%)
    //   - 0.5B_FFN_Dn (N=896):  N64/MT32/MB1 wins at M=128 (+24%)
    // These shapes have too few N128-tiles to saturate the GPU.
    //
    // Cooperative N128/MT32/MB1/UG2 also beats streaming for LM_Head shapes
    // (verified: 7B_LM_Head +13% vs streaming Auto).
    // =========================================================================
    if constexpr (!NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale &&
                  !NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale_asym &&
                  !NATIVE_QUANTISEDGemmTraits<FMT>::is_asymmetric) {
        // Pattern S: simple symmetric (Q4_0, IQ4_NL, Q5_0, IQ3_S, IQ3_XXS, IQ2_XXS)
        // Use MB1/UG2 always for these formats.
        auto launch_mb1_ug2 = [&](auto n_tile_tag, auto m_tile_tag) -> bool
        {
            constexpr int NT = decltype(n_tile_tag)::value;
            constexpr int MT = decltype(m_tile_tag)::value;
            dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
            dim3 block(BLOCK_SIZE, 1, 1);
            const void* function = reinterpret_cast<const void*>(HIP_KERNEL_NAME(
                native_quantised_gemm_kernel<NT, BPT, MT, FMT, 1, 2>));

            clear_launch_error();
            native_quantised_gemm_kernel<NT, BPT, MT, FMT, 1, 2>
                <<<grid, block, 0, hip_stream>>>(
                    d_A_int8, d_payload, d_block_scales, d_block_mins, d_block_emins,
                    d_output, d_scales_A, M, N, K, d_scales_A_blockwise,
                    serial_m1_kblocks);

            return complete_launch(
                llaminar2::rocm::VNNIPrefillProducer::NativeCooperative,
                "simple_symmetric_mb1_ug2",
                NT,
                MT,
                1,
                2,
                /*full_tiles=*/false,
                function);
        };

        using N64_  = std::integral_constant<int, 64>;
        using N128_ = std::integral_constant<int, 128>;
        using M16_  = std::integral_constant<int, 16>;
        using M32_  = std::integral_constant<int, 32>;

        // Small-N exception: N≤1024 with few rows → N64 tiles better
        if (N <= 1024 && M <= 128) {
            // At small N, N128 tiles produce too few WGs.
            // N64/MT16 or MT32 depending on waste
            const int n64_mt = (m_tile <= 16) ? 16 : 32;
            if (n64_mt <= 16) {
                return launch_mb1_ug2(N64_{}, M16_{});
            } else {
                return launch_mb1_ug2(N64_{}, M32_{});
            }
        }

        // Default: N128/MT32/MB1/UG2
        return launch_mb1_ug2(N128_{}, M32_{});
    }
    else
    {
    // These families are disjoint at compile time. A return above does not
    // prevent HIP from emitting kernels mentioned in later source, so retain
    // only dispatches that a format can actually reach. In particular, simple
    // symmetric weights never use the spilling legacy M16 streaming body.

    // Dispatch tags for (N_TILE, M_TILE) combinations
    using N64  = std::integral_constant<int, 64>;
    using N128 = std::integral_constant<int, 128>;
    using M16  = std::integral_constant<int, 16>;
    using M32  = std::integral_constant<int, 32>;
    using M64  = std::integral_constant<int, 64>;
    using M8   = std::integral_constant<int, 8>;

    // Q6_K is the dominant Qwen 3.6 dense-prefill codebook. A complete sweep of
    // every physical graph bucket and all active 35B MoE geometries proves that
    // N128/M32 full tiles are byte-identical and spill-free. M=600 is the sole
    // non-M32-aligned capture; its common winner is checked N128/M16/MB1. Both
    // routes retain serial-M1 K partition boundaries and arithmetic order.
    if constexpr (FMT == NATIVE_QUANTISED_GEMM_Q6_K)
    {
        if (M >= 32)
        {
            const bool full_tiles =
                (M % 32) == 0 &&
                (N % 128) == 0 &&
                (K % (BPT * 32)) == 0;
            if (full_tiles)
                return launch_config(128, 32, 2, 4, true);
            return launch_config(128, 16, 1, 4, false);
        }
    }

    // =========================================================================
    // STREAMING DISPATCH: LM_Head-like shapes (massive N, moderate K)
    //
    // For shapes where N >> K and N is very large, the streaming kernel
    // eliminates ALL barriers and LDS, achieving 3-4 wave occupancy.
    //
    // However, this only benefits Pattern S and D formats (light decode).
    // Pattern A and DA formats have heavy decode that causes register
    // spilling in the streaming kernel — cooperative is 25-130% faster.
    //
    // N threshold is format-dependent:
    //   Pattern D (is_dual_scale): N >= 8192.  The cooperative kernel suffers
    //     72-132 bytes scratch at N128/M32 due to temp_acc[4][4] + scale arrays
    //     + LDS pipeline staging.  The streaming kernel uses temp_acc[M_TILE]
    //     (1D array) → 4-wave occupancy, zero scratch.  Tested: streaming wins
    //     by 15-30% at N=11008 (3B_FFN_Up) for Q3_K, Q6_K, IQ2_S, IQ2_XS.
    //   Pattern S: N >= 16384.  Cooperative is scratch-free at N128/M32 so only
    //     massive-N shapes (LM_Head) benefit from streaming's barrier-free design.
    //
    // K <= 4096 is a safety check (streaming loads A per-thread, not via LDS).
    // =========================================================================
    if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::prefer_streaming) {
        constexpr int streaming_n_threshold =
            NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale ? 8192 : 16384;
        if (N >= streaming_n_threshold && K <= 4096) {
            // IQ1_M uses the same M8/three-wave tile. An M16/two-wave variant
            // was measured across M=64..256 and regressed by 25-38%; the lost
            // latency hiding outweighed reduced decode and activation loads.
            if constexpr (FMT == NATIVE_QUANTISED_GEMM_IQ1_M) {
                return launch_streaming(M8{});
            // Other Pattern D formats use M8: temp_acc[8] = 8 VGPRs.
            // A-data per thread = 8×K/4×4 = 2K bytes ≤ 16KB.
            } else if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale) {
                return launch_streaming(M8{});
            } else {
                return launch_streaming(M16{});
            }
        }
    }

    // NOTE: Pattern DA (Q2_K) must NOT use streaming — tested at N=11008
    // and it was 2-3× slower (Q2_K: 1390μs vs 565μs cooperative).  The Pattern DA
    // decode + accumulation (temp_acc + temp_sum + emin per half) is
    // too register-heavy for streaming's per-thread A-loading model.
    // IQ1_M was moved from DA to D (emins always zero) and is now streaming-eligible.

    // Pattern A streaming with cooperative sum_a precompute:
    // Only for formats where streaming is proven faster than cooperative.
    // - IQ1_S: +18% (heavy grid decode benefits hugely from 4-wave occupancy)
    // - Q4_1:  +2.5% (light decode, small but consistent improvement)
    // NOT Q5_1: -16% regression (heavy 5-bit qh decode + 20B payload → VGPRs
    //           push past 4-wave threshold, and extra HBM for larger payload)
    if constexpr (FMT == NATIVE_QUANTISED_GEMM_Q4_1 || FMT == NATIVE_QUANTISED_GEMM_IQ1_S) {
        if (N >= 8192 && K <= 4096) {
            return launch_streaming(M16{});
        }
    }

    if (use_n128) {
        // Dual-scale-asymmetric formats (Pattern DA: Q2_K only):
        // N128/M16/2-wave is the best tile for Pattern DA.
        // IQ1_M moved to Pattern D (emins always zero) — uses hybrid dispatch below.
        //
        // Tested alternatives for Q2_K (all regressed):
        //   - N128/M32/2w: +200 bytes scratch from temp_acc+temp_sum+emin staging
        //   - N64/M16/3w:  +55% slower! N_PER_THREAD 2→1 kills ILP in the heavy
        //                  DA accumulation (sdot4 interleaving across N-columns)
        //   - Streaming:   +95% slower! Per-thread A-loading + DA register pressure
        //   - Split-emin:  Went to 1-wave! 4 separate loops = too much code for RA
        //   - launch3 (3w hint): compiler still achieves 2w only, no perf change
        if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale_asym) {
            return launch(N128{}, M16{});
        }

        // Pattern A (asymmetric: Q4_1, Q5_1, IQ1_S): force M32/2-wave on N128.
        // N128/M16/3w spills 40-72 bytes scratch to achieve 3-wave occupancy.
        // N128/M32/2w is scratch-free (126-128 VGPRs) with better A-data reuse.
        // Tested hybrid M16/M32: no improvement at M=32 (142→143μs for Q4_1),
        // minor regression at M=32 for Q5_1 (147→151μs).  Scratch cost ≈ 3-wave gain.
        if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::is_asymmetric) {
            return launch(N128{}, M32{});
        }

        // Hybrid N128 dispatch: M16/3-wave for small grids, M32/2-wave for large
        //
        // Guard with if constexpr: only dual_scale (non-asym) formats reach here
        // at runtime (Q6_K, Q3_K, IQ2_S, IQ2_XS, IQ1_M), but without the guard
        // the compiler still instantiates launch3 for all formats — causing
        // occupancy failures for Q2_K (drops to 1-wave at N128/M16/3w).
        //
        // NOTE: M_TILE > 32 is NOT viable on N128 due to A-LDS bank conflicts.
        // gfx906 has 32 LDS banks at 4-byte granularity.  With M_PER_THREAD=4,
        // each thread reads 4 consecutive int32 from banks (t_m*4)%32 through
        // (t_m*4+3)%32.  Bank groups wrap every 8 threads:
        //   THREADS_M=8  (M32):  8 unique bank groups → zero conflicts
        //   THREADS_M=16 (M64):  2-way conflicts → ~1.7× penalty
        //   THREADS_M=32 (M128): 4-way conflicts → ~2.5× penalty
        // The A-data reuse gain from N_PER_THREAD 4→8 is completely eaten by
        // the bank conflict stalls.  Empirically: M64/N128 and M128/N64 both
        // regressed LM_Head from 0.88× to 0.30-0.37× (2.5× slower).
        if constexpr (!NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale_asym &&
                      !NATIVE_QUANTISEDGemmTraits<FMT>::is_asymmetric) {
        const int n128_blocks = (N + 127) / 128;
        const int m16_blocks  = (M + 15) / 16;
        const int total_wgs_m16 = n128_blocks * m16_blocks;

        if (total_wgs_m16 <= 256) {
            // GPU undersaturated: M16/3-wave → more waves hide latency
            return launch3(N128{}, M16{});
        } else {
            // GPU saturated: M32/2-wave → better tile efficiency
            const int m_tile_n128 = (m_tile <= 16) ? 16 : 32;
            if (m_tile_n128 <= 16)
                return launch3(N128{}, M16{});
            else
                return launch(N128{}, M32{});
        }
        }
    } else {
        // Dual-scale-asymmetric formats (Pattern DA: Q2_K only): force M16 on N64.
        // N64/M16/3w: VGPR=84, 84B scratch (compiler spills to hit 3-wave).
        // N64/M16/2w: VGPR=106, 0 scratch.  Use 2-wave to avoid scratch.
        // IQ1_M now uses Pattern D and falls through to the hybrid dispatch below.
        if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale_asym) {
            return launch(N64{}, M16{});
        }

        // N64 (K-heavy) dispatch with hybrid M16/3-wave for small-M shapes.
        //
        // Mirrors the N128 hybrid strategy: when the total M16 workgroup count
        // is ≤256 (~1 WG/SIMD on MI50/60), the M16/3-wave variant provides
        // better GPU coverage (more CUs active) despite lower per-tile compute
        // density.  This is critical for K-heavy shapes (FFN_Dn, AttnOut) at
        // small M where few N64 workgroups are launched.
        //
        // With M32/2-wave, e.g. 7B_AttnOut M=32 has only 56 WGs → 28 CUs (47%).
        // M16/3-wave doubles WGs to 112 → 37-56 CUs (62-93%) active.
        //
        // For M≥128, the waste-minimizing M64/M32 tiles already achieve good
        // GPU coverage (many WGs) and their superior A-data reuse dominates.
        //
        // Threshold is 128 WGs (vs 256 for N128) because N64 tiles have half
        // the column-output of N128 tiles, so per-WG compute density is lower.
        // Above 128 WGs the GPU has enough parallelism for M32/M64 tiles to
        // win via better A-data reuse.  Verified: keeps 0.5B/3B M=64 wins
        // (+0.02..+0.42) while avoiding 7B M=64 regressions (224 WGs > 128).
        if (M <= 64) {
            const int n64_blocks = (N + 63) / 64;
            const int m16_blocks = (M + 15) / 16;
            const int total_wgs_m16 = n64_blocks * m16_blocks;

            if (total_wgs_m16 <= 128) {
                return launch3(N64{}, M16{});
            }
        }

        // Cap M_TILE at 32 for Pattern A and Pattern D on N64 path.
        //
        // Pattern A (asymmetric: Q4_1, Q5_1, IQ1_S):
        //   N64/M64 causes scratch spill (8-16B) and saturates VGPRs at 127-128.
        //   N64/M32 is clean: 108-111 VGPRs, zero scratch, 2-wave.
        //
        // Pattern D (dual-scale: Q6_K, Q3_K, IQ2_S, IQ2_XS):
        //   N64/M64 causes 124 bytes scratch spill for Pattern D (M_PER_THREAD=4
        //   with two-phase still needs too many registers at THREADS_M=16).
        int effective_m_tile = m_tile;
        if constexpr (NATIVE_QUANTISEDGemmTraits<FMT>::is_dual_scale ||
                       NATIVE_QUANTISEDGemmTraits<FMT>::is_asymmetric) {
            effective_m_tile = (effective_m_tile > 32) ? 32 : effective_m_tile;
        }

        switch (effective_m_tile) {
            case 16:  return launch(N64{}, M16{});
            case 32:  return launch(N64{}, M32{});
            default:  return launch(N64{}, M64{});
        }
    }
    } // non-simple-symmetric dispatch families
}


// ============================================================================
// C API: Format-dispatched native-quantised GEMM
//
// Mirrors rocmGemv_native_quantised_fp32() but for M>1 prefill.
// Output is FP32 with scale_A applied inline — no separate epilogue.
// ============================================================================

extern "C"
bool LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD_SYMBOL(
    rocmGemm_native_quantised_fp32_shard_)(
    const int8_t*  d_A_int8,       // [M × K] INT8 activations
    const uint8_t* d_payload,      // [blocks_per_row × N × PB] native-quantised payload
    const void*    d_block_scales, // [blocks_per_row × N] __half per-block scales
    const void*    d_block_mins,   // [blocks_per_row × N] __half per-block mins/aux (nullable)
    const void*    d_block_emins,  // [blocks_per_row × N] packed {lo,hi} emins (Q2_K only, nullable)
    float*         d_output,       // [M × N] FP32 output
    const float*   d_scales_A,     // [M] per-row activation scales
    const float*   d_scales_A_blockwise, // [M × blocks_per_row] per-block activation scales
    int M, int N, int K,
    uint8_t codebook_id,
    uint8_t arithmetic_policy_codebook_id,
    int device_id, void* stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || K % 32 != 0)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const auto* scales = reinterpret_cast<const uint16_t*>(d_block_scales);
    const auto* mins   = reinterpret_cast<const uint16_t*>(d_block_mins);
    const auto* emins  = reinterpret_cast<const uint32_t*>(d_block_emins);

    #define LAUNCH_GEMM(FMT) \
        return launch_native_quantised_gemm<FMT>( \
            d_A_int8, d_payload, scales, mins, emins, d_output, d_scales_A, \
            d_scales_A_blockwise, M, N, K, device_id, stream, \
            arithmetic_policy_codebook_id)

    switch (codebook_id)
    {
#if LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 0
    case NATIVE_QUANTISED_GEMM_Q4_0:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q4_0);
    case NATIVE_QUANTISED_GEMM_IQ4_NL:   LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ4_NL);
    case NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE: LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ4_XS_SOURCE);
#elif LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 1
    case NATIVE_QUANTISED_GEMM_Q4_1:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q4_1);
    case NATIVE_QUANTISED_GEMM_Q5_0:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q5_0);
#elif LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 2
    case NATIVE_QUANTISED_GEMM_Q5_1:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q5_1);
    case NATIVE_QUANTISED_GEMM_Q6_K:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q6_K);
#elif LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 3
    case NATIVE_QUANTISED_GEMM_Q3_K:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q3_K);
    case NATIVE_QUANTISED_GEMM_Q2_K:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q2_K);
#elif LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 4
    case NATIVE_QUANTISED_GEMM_IQ3_S:    LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ3_S);
    case NATIVE_QUANTISED_GEMM_IQ3_XXS:  LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ3_XXS);
#elif LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 5
    case NATIVE_QUANTISED_GEMM_IQ2_S:    LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ2_S);
    case NATIVE_QUANTISED_GEMM_IQ2_XS:   LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ2_XS);
#elif LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 6
    case NATIVE_QUANTISED_GEMM_IQ2_XXS:  LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ2_XXS);
    case NATIVE_QUANTISED_GEMM_IQ1_S:    LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ1_S);
#elif LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD == 7
    case NATIVE_QUANTISED_GEMM_IQ1_M:    LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_IQ1_M);
    case NATIVE_QUANTISED_GEMM_Q8_0:     LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_Q8_0);
    case NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN:
        LAUNCH_GEMM(NATIVE_QUANTISED_GEMM_EXPANDED_INT8_MIN);
#endif
    default:
        fprintf(
            stderr,
            "[rocmGemm_native_quantised_fp32_shard_%d] codebook_id=%d does not "
            "belong to this physical shard\n",
            LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD,
            codebook_id);
        return false;
    }

    #undef LAUNCH_GEMM
}


// ============================================================================
// IQ grid table initialization for GEMM translation unit.
// Each .hip file has its own __constant__ symbols; must be initialized
// separately from the GEMV init (rocmInitIQGridTables).
// ============================================================================

extern "C"
bool LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD_SYMBOL(
    rocmInitIQGridTables_gemm_shard_)(
    int device_id,
    const void* h_iq3s_grid,    // [512]  uint32_t
    const void* h_iq3xxs_grid,  // [256]  uint32_t
    const void* h_iq2s_grid,    // [1024] uint64_t
    const void* h_iq2xs_grid,   // [512]  uint64_t
    const void* h_iq2xxs_grid,  // [256]  uint64_t
    const void* h_iq1s_grid)    // [2048] uint64_t (IQ1_S/IQ1_M shared)
{
    return initialize_native_quantised_gemm_grids(device_id, h_iq3s_grid,
        h_iq3xxs_grid, h_iq2s_grid, h_iq2xs_grid, h_iq2xxs_grid, h_iq1s_grid);
}

#undef LLAMINAR_ROCM_NATIVE_QUANTISED_SHARD_SYMBOL
#undef LLAMINAR_ROCM_NATIVE_QUANTISED_CONCAT
#undef LLAMINAR_ROCM_NATIVE_QUANTISED_CONCAT_IMPL
