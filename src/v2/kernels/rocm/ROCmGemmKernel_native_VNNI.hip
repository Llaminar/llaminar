/**
 * @file ROCmGemmKernel_native_VNNI.hip
 * @brief Native-VNNI GEMM kernels for lossless sub-8-bit prefill on AMD MI50/MI60 (gfx906)
 *
 * @target AMD Instinct MI50 / MI60 (gfx906), 60 CUs, 4 SIMDs/CU
 * @isa    v_dot4_i32_i8 (VNNI: 4-way signed INT8 dot product, 1 cycle/op)
 * @op     C[M×N] (FP32) = scale_A[M] × Σ_b { block_d[b,N] × sdot4(A_int8[M,K], W_decoded[K,N]) }
 *
 * This file contains the production native-VNNI GEMM kernels for the **prefill**
 * (prompt-processing) phase on AMD gfx906 GPUs.  Unlike the INT8 GEMM kernels
 * (ROCmQuantisedGemmKernel_INT8_VNNI.hip), these kernels:
 *
 *   1. Load COMPACT quantized payloads from HBM (4.5 bpw for Q4_0/IQ4_NL
 *      vs 8 bpw for INT8) — halving HBM bandwidth requirements
 *   2. Decode quantized weights to INT8 in-register during the load phase
 *      (Decode-in-Load architecture)
 *   3. Use FP32 accumulators with per-block FP16 scales (no separate epilogue)
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
 * Currently supports:
 *   Q4_0   (codebook_id=0): 4-bit symmetric, linear decode (q - 8)
 *   IQ4_NL (codebook_id=4): 4-bit symmetric, LUT decode via v_perm_b32
 *
 * @see ROCmQuantisedGemmKernel_INT8_VNNI.hip for INT8 VNNI GEMM (V3/V7)
 * @see ROCmGemvKernel_native_VNNI.hip for native-VNNI GEMV (M=1 decode)
 * @see docs/v2/NATIVE_VNNI_GEMM_DESIGN.md for architecture design document
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

#include "../../backends/rocm/HipDeviceGuard.h"

// ============================================================================
// Format tags (subset of NativeVNNIFormat from ROCmGemvKernel_native_VNNI.hip)
// ============================================================================

enum NativeVNNIGemmFormat : uint8_t {
    NVNNI_GEMM_Q4_0   = 0,   // 4-bit symmetric, linear decode (q - 8)
    NVNNI_GEMM_IQ4_NL = 4,   // 4-bit symmetric, LUT decode via v_perm_b32
};

// ============================================================================
// Device decode functions: raw Q4 payload → 8 packed INT32 groups
//
// Each 32-element quantized block has 16 bytes of nibble payload.
// 4 dwords (uint32_t raw[4]) contain interleaved nibbles:
//   byte j = lo_nibble(element j) | hi_nibble(element j+16)
//
// Decode produces 8 int32_t "packed groups", each containing 4 signed INT8:
//   groups[0..3]: elements [0..15]  from low nibbles
//   groups[4..7]: elements [16..31] from high nibbles
// ============================================================================

/**
 * Q4_0 decode: nibble → signed INT8 via carry-free SWAR centering.
 * values in [-8, +7], packed as 4 signed bytes per int32_t.
 *
 * Uses add+XOR SWAR method (same as GEMV kernel) to avoid inter-byte
 * borrow propagation: nibble + 0x78 places result in [0x78, 0x87] with
 * no byte overflow, then XOR 0x80 flips the sign bit to produce [-8, +7].
 */
__device__ __forceinline__ void decode_q4_0_block(
    const uint32_t raw[4], int32_t packed_groups[8])
{
    constexpr uint32_t SWAR_ADD = 0x78787878u;
    constexpr uint32_t SWAR_XOR = 0x80808080u;
    constexpr uint32_t NIBBLE_MASK = 0x0F0F0F0Fu;

    #pragma unroll
    for (int g = 0; g < 4; ++g)
    {
        const uint32_t lo_nibs = raw[g] & NIBBLE_MASK;
        const uint32_t hi_nibs = (raw[g] >> 4) & NIBBLE_MASK;
        packed_groups[g]     = static_cast<int32_t>((lo_nibs + SWAR_ADD) ^ SWAR_XOR);
        packed_groups[g + 4] = static_cast<int32_t>((hi_nibs + SWAR_ADD) ^ SWAR_XOR);
    }
}

/**
 * IQ4_NL decode: nibble → signed INT8 via 16-entry codebook LUT.
 * Uses v_perm_b32 for register-based byte permutation (3 v_perm per 4 values).
 * Codebook: {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}
 */
__device__ __forceinline__ void decode_iq4_nl_block(
    const uint32_t raw[4], int32_t packed_groups[8])
{
    // Codebook packed as 4 registers for v_perm_b32 lookup:
    //   CB_LO_A/B: entries 0-7 (src0/src1 for low-nibble half)
    //   CB_HI_A/B: entries 8-15 (src0/src1 for high-nibble half)
    constexpr uint32_t CB_LO_A = 0xBFAD9881u;  // cb[3..0] = {-65,-83,-104,-127}
    constexpr uint32_t CB_LO_B = 0xF6EADDCFu;  // cb[7..4] = {-10,-22,-35,-49}
    constexpr uint32_t CB_HI_A = 0x26190D01u;   // cb[11..8] = {38,25,13,1}
    constexpr uint32_t CB_HI_B = 0x71594535u;   // cb[15..12] = {113,89,69,53}
    constexpr uint32_t MUX_IDENTITY  = 0x03020100u;  // byte identity permutation
    constexpr uint32_t MUX_BIT3_MASK = 0x04040404u;  // bit3 → +4 offset per byte

    // NOTE: __builtin_amdgcn_perm(a, b, sel) on HIP/ROCm maps sel 0-3 → b (2nd arg),
    // sel 4-7 → a (1st arg). This is reversed from the ISA manual's src0/src1 convention.
    // We pass (B, A) so that sel 0-3 picks from A (cb[0..3]) and sel 4-7 from B (cb[4..7]).
    // Similarly, the mux perm uses (r_hi, r_lo) so bit3=0 → r_lo, bit3=1 → r_hi.
    #pragma unroll
    for (int g = 0; g < 4; ++g)
    {
        // --- Low nibbles (elements [4g..4g+3]) ---
        const uint32_t lo_nibs = raw[g] & 0x0F0F0F0Fu;
        {
            const uint32_t sel = lo_nibs & 0x07070707u;
            const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, sel);
            const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, sel);
            const uint32_t mux = MUX_IDENTITY | ((lo_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g] = static_cast<int32_t>(
                __builtin_amdgcn_perm(r_hi, r_lo, mux));
        }

        // --- High nibbles (elements [4g+16..4g+19]) ---
        const uint32_t hi_nibs = (raw[g] >> 4) & 0x0F0F0F0Fu;
        {
            const uint32_t sel = hi_nibs & 0x07070707u;
            const uint32_t r_lo = __builtin_amdgcn_perm(CB_LO_B, CB_LO_A, sel);
            const uint32_t r_hi = __builtin_amdgcn_perm(CB_HI_B, CB_HI_A, sel);
            const uint32_t mux = MUX_IDENTITY | ((hi_nibs >> 1) & MUX_BIT3_MASK);
            packed_groups[g + 4] = static_cast<int32_t>(
                __builtin_amdgcn_perm(r_hi, r_lo, mux));
        }
    }
}

/**
 * Format-dispatched decode: compile-time selection of Q4_0 or IQ4_NL path.
 */
template <NativeVNNIGemmFormat FMT>
__device__ __forceinline__ void decode_block(
    const uint32_t raw[4], int32_t packed_groups[8])
{
    if constexpr (FMT == NVNNI_GEMM_Q4_0)
        decode_q4_0_block(raw, packed_groups);
    else if constexpr (FMT == NVNNI_GEMM_IQ4_NL)
        decode_iq4_nl_block(raw, packed_groups);
}


// ============================================================================
// Native-VNNI GEMM kernel: Decode-in-Load with INT32 block accumulators
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

template <int N_TILE, int BPT, int M_TILE, NativeVNNIGemmFormat FMT, int MIN_BLOCKS = 2>
__global__ __launch_bounds__(256, MIN_BLOCKS)
void native_vnni_gemm_kernel(
    const int8_t*    __restrict__ d_A_int8,       // [M × K] INT8 activations, row-major
    const uint8_t*   __restrict__ d_payload,      // [blocks_per_row × N × 16] native-VNNI payload
    const uint16_t*  __restrict__ d_block_scales, // [blocks_per_row × N] FP16 per-block scales
    float*           __restrict__ d_output,       // [M × N] FP32 output, row-major
    const float*     __restrict__ d_scales_A,     // [M] per-row activation scales
    int M, int N, int K)
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
    constexpr int PAYLOAD_BYTES = 16;  // Q4_0 and IQ4_NL: 16 bytes per 32-element block

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
    __shared__ uint16_t scale_lds[2][BPT * N_TILE];  // Per-block FP16 scales

    // FP32 accumulators — accumulate across all K-blocks
    float acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0.0f;

    // Pre-load per-row activation scales (one per M-row this thread owns)
    float scale_a[M_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int gm = m_block + m_base + mm;
        scale_a[mm] = (gm < M) ? d_scales_A[gm] : 0.0f;
    }

    // K dimensions
    const int blocks_per_row = K / 32;
    const int total_k_groups = K / 4;
    const int num_block_tiles = (blocks_per_row + BPT - 1) / BPT;

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
    const int  num_safe = (a_m_safe && b_n_safe)
                        ? (safe_k < num_block_tiles ? safe_k : num_block_tiles)
                        : 0;

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
    // First BPT×N_TILE threads load raw payload (16 bytes each) and scales.
    // Thread mapping: (block_local, n_local) via div/mod N_TILE.
    //
    // Each thread:
    //   1. Loads 16 bytes of raw payload via uint4 (1 global read)
    //   2. Loads 1 FP16 scale (1 global read)
    //   3. Decodes payload → 8 packed INT32 groups
    //   4. Writes 8 int32 to b_lds + 1 uint16 to scale_lds
    // =========================================================================
    constexpr int B_LOAD_THREADS = BPT * N_TILE;  // 128 for BPT=2, N_TILE=64

    const int b_load_block = threadIdx.x / N_TILE;  // which block in tile (0..BPT-1)
    const int b_load_n     = threadIdx.x % N_TILE;  // which N-column (tile-local)
    const int b_load_n_global = n_block + b_load_n;  // global N-column

    // Staging registers for pipeline
    uint4 a_staging[A_NUM_PASSES];
    uint4 b_payload_staging;  // 16 bytes of raw payload
    uint16_t b_scale_staging; // 1 FP16 scale

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    // =========================================================================
    {
        const bool tile0_safe = (num_safe > 0);

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
        b_payload_staging = make_uint4(0, 0, 0, 0);
        b_scale_staging = 0;
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = b_load_block;
            if (tile0_safe || __builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                b_payload_staging = *reinterpret_cast<const uint4*>(
                    d_payload + linear * PAYLOAD_BYTES);
                b_scale_staging = d_block_scales[linear];
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
    }
    if (threadIdx.x < B_LOAD_THREADS)
    {
        uint32_t raw[4] = {b_payload_staging.x, b_payload_staging.y,
                           b_payload_staging.z, b_payload_staging.w};
        int32_t packed_groups[8];
        decode_block<FMT>(raw, packed_groups);

        #pragma unroll
        for (int g = 0; g < 8; ++g)
            b_lds[0][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

        scale_lds[0][b_load_block * N_TILE + b_load_n] = b_scale_staging;
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
            b_payload_staging = *reinterpret_cast<const uint4*>(
                d_payload + linear * PAYLOAD_BYTES);
            b_scale_staging = d_block_scales[linear];
        }

        // --- 2. Compute on CURRENT buffer ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
        {
            int32_t block_acc[M_PER_THREAD][N_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    block_acc[mm][nn] = 0;

            #pragma clang loop unroll_count(4)
            for (int g = 0; g < 8; ++g)
            {
                const int kk = bp * 8 + g;

                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_lds[cur_buf][kk * M_TILE + m_base + mm];

                int32_t b_reg[N_PER_THREAD];
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    b_reg[nn] = b_lds[cur_buf][kk * N_TILE + t_n * N_PER_THREAD + nn];

                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                        block_acc[mm][nn] = __builtin_amdgcn_sdot4(
                            a_reg[mm], b_reg[nn], block_acc[mm][nn], false);
            }

            float block_scale[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const uint16_t s_bits = scale_lds[cur_buf][bp * N_TILE + t_n * N_PER_THREAD + nn];
                block_scale[nn] = __half2float(*reinterpret_cast<const __half*>(&s_bits));
            }

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __fmaf_rn(
                        static_cast<float>(block_acc[mm][nn]), block_scale[nn], acc[mm][nn]);
        }

        // --- 3. Barrier: all reads from cur_buf done ---
        __syncthreads();

        // --- 4. Decode + write staging → next buffer ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }

        if (threadIdx.x < B_LOAD_THREADS)
        {
            uint32_t raw[4] = {b_payload_staging.x, b_payload_staging.y,
                               b_payload_staging.z, b_payload_staging.w};
            int32_t packed_groups[8];
            decode_block<FMT>(raw, packed_groups);

            #pragma unroll
            for (int g = 0; g < 8; ++g)
                b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

            scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
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

        b_payload_staging = make_uint4(0, 0, 0, 0);
        b_scale_staging = 0;
        if (threadIdx.x < B_LOAD_THREADS)
        {
            const int block_global = tile_block_base + b_load_block;
            if (__builtin_expect(b_load_n_global < N && block_global < blocks_per_row, 1))
            {
                const size_t linear = static_cast<size_t>(block_global) * N + b_load_n_global;
                b_payload_staging = *reinterpret_cast<const uint4*>(
                    d_payload + linear * PAYLOAD_BYTES);
                b_scale_staging = d_block_scales[linear];
            }
        }

        // --- 2. Compute on CURRENT buffer ---
        #pragma unroll
        for (int bp = 0; bp < BPT; ++bp)
        {
            int32_t block_acc[M_PER_THREAD][N_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    block_acc[mm][nn] = 0;

            #pragma clang loop unroll_count(4)
            for (int g = 0; g < 8; ++g)
            {
                const int kk = bp * 8 + g;

                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_lds[cur_buf][kk * M_TILE + m_base + mm];

                int32_t b_reg[N_PER_THREAD];
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    b_reg[nn] = b_lds[cur_buf][kk * N_TILE + t_n * N_PER_THREAD + nn];

                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                        block_acc[mm][nn] = __builtin_amdgcn_sdot4(
                            a_reg[mm], b_reg[nn], block_acc[mm][nn], false);
            }

            float block_scale[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const uint16_t s_bits = scale_lds[cur_buf][bp * N_TILE + t_n * N_PER_THREAD + nn];
                block_scale[nn] = __half2float(*reinterpret_cast<const __half*>(&s_bits));
            }

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __fmaf_rn(
                        static_cast<float>(block_acc[mm][nn]), block_scale[nn], acc[mm][nn]);
        }

        // --- 3. Barrier: all reads from cur_buf done ---
        __syncthreads();

        // --- 4. Decode + write staging → next buffer ---
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            a_lds[nxt_buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[nxt_buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[nxt_buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[nxt_buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }

        if (threadIdx.x < B_LOAD_THREADS)
        {
            uint32_t raw[4] = {b_payload_staging.x, b_payload_staging.y,
                               b_payload_staging.z, b_payload_staging.w};
            int32_t packed_groups[8];
            decode_block<FMT>(raw, packed_groups);

            #pragma unroll
            for (int g = 0; g < 8; ++g)
                b_lds[nxt_buf][(b_load_block * 8 + g) * N_TILE + b_load_n] = packed_groups[g];

            scale_lds[nxt_buf][b_load_block * N_TILE + b_load_n] = b_scale_staging;
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
            int32_t block_acc[M_PER_THREAD][N_PER_THREAD];
            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    block_acc[mm][nn] = 0;

            #pragma clang loop unroll_count(4)
            for (int g = 0; g < 8; ++g)
            {
                const int kk = bp * 8 + g;

                int32_t a_reg[M_PER_THREAD];
                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    a_reg[mm] = a_lds[last_buf][kk * M_TILE + m_base + mm];

                int32_t b_reg[N_PER_THREAD];
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    b_reg[nn] = b_lds[last_buf][kk * N_TILE + t_n * N_PER_THREAD + nn];

                #pragma unroll
                for (int mm = 0; mm < M_PER_THREAD; ++mm)
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                        block_acc[mm][nn] = __builtin_amdgcn_sdot4(
                            a_reg[mm], b_reg[nn], block_acc[mm][nn], false);
            }

            float block_scale[N_PER_THREAD];
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                const uint16_t s_bits = scale_lds[last_buf][bp * N_TILE + t_n * N_PER_THREAD + nn];
                block_scale[nn] = __half2float(*reinterpret_cast<const __half*>(&s_bits));
            }

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __fmaf_rn(
                        static_cast<float>(block_acc[mm][nn]), block_scale[nn], acc[mm][nn]);
        }
    }

    // =========================================================================
    // STORE RESULTS: Apply scale_A and write FP32 to global memory
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;
        if (m_idx >= M) continue;

        const float sa = scale_a[mm];

        if constexpr (N_PER_THREAD >= 4)
        {
            // Vectorized store path
            if (n_base + N_PER_THREAD - 1 < N)
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
                // Scalar fallback for boundary columns
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                        d_output[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn] * sa;
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

template <NativeVNNIGemmFormat FMT>
static bool launch_native_vnni_gemm(
    const int8_t* d_A_int8,
    const uint8_t* d_payload,
    const uint16_t* d_block_scales,
    float* d_output,
    const float* d_scales_A,
    int M, int N, int K,
    int device_id, void* stream)
{
    constexpr int BPT = 2;
    constexpr int BLOCK_SIZE = 256;

    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // Templated launch helper (default 2-wave)
    auto launch = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);

        native_vnni_gemm_kernel<NT, BPT, MT, FMT>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales,
                d_output, d_scales_A, M, N, K);

        return hipGetLastError() == hipSuccess;
    };

    // 3-wave variant: MIN_BLOCKS=3 tells compiler to target ≤84 VGPRs
    auto launch3 = [&](auto n_tile_tag, auto m_tile_tag) -> bool
    {
        constexpr int NT = decltype(n_tile_tag)::value;
        constexpr int MT = decltype(m_tile_tag)::value;
        dim3 grid((N + NT - 1) / NT, (M + MT - 1) / MT, 1);
        dim3 block(BLOCK_SIZE, 1, 1);

        native_vnni_gemm_kernel<NT, BPT, MT, FMT, 3>
            <<<grid, block, 0, hip_stream>>>(
                d_A_int8, d_payload, d_block_scales,
                d_output, d_scales_A, M, N, K);

        return hipGetLastError() == hipSuccess;
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

    // Dispatch tags for (N_TILE, M_TILE) combinations
    using N64  = std::integral_constant<int, 64>;
    using N128 = std::integral_constant<int, 128>;
    using M16  = std::integral_constant<int, 16>;
    using M32  = std::integral_constant<int, 32>;
    using M64  = std::integral_constant<int, 64>;

    if (use_n128) {
        // Hybrid N128 dispatch: M16/3-wave for small grids, M32/2-wave for large
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
    } else {
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

        switch (m_tile) {
            case 16:  return launch(N64{}, M16{});
            case 32:  return launch(N64{}, M32{});
            default:  return launch(N64{}, M64{});
        }
    }
}


// ============================================================================
// C API: Format-dispatched native-VNNI GEMM
//
// Mirrors rocmGemv_native_vnni_fp32() but for M>1 prefill.
// Output is FP32 with scale_A applied inline — no separate epilogue.
// ============================================================================

extern "C"
bool rocmGemm_native_vnni_fp32(
    const int8_t*  d_A_int8,       // [M × K] INT8 activations
    const uint8_t* d_payload,      // [blocks_per_row × N × 16] native-VNNI payload
    const void*    d_block_scales, // [blocks_per_row × N] __half per-block scales
    float*         d_output,       // [M × N] FP32 output
    const float*   d_scales_A,     // [M] per-row activation scales
    int M, int N, int K,
    uint8_t codebook_id,
    int device_id, void* stream)
{
    if (M <= 0 || N <= 0 || K <= 0 || K % 32 != 0)
        return false;

    llaminar2::HipDeviceGuard::setDevice(device_id);

    const auto* scales = reinterpret_cast<const uint16_t*>(d_block_scales);

    switch (codebook_id)
    {
    case NVNNI_GEMM_Q4_0:
        return launch_native_vnni_gemm<NVNNI_GEMM_Q4_0>(
            d_A_int8, d_payload, scales, d_output, d_scales_A,
            M, N, K, device_id, stream);

    case NVNNI_GEMM_IQ4_NL:
        return launch_native_vnni_gemm<NVNNI_GEMM_IQ4_NL>(
            d_A_int8, d_payload, scales, d_output, d_scales_A,
            M, N, K, device_id, stream);

    default:
        fprintf(stderr, "[rocmGemm_native_vnni_fp32] Unsupported codebook_id=%d\n",
                codebook_id);
        return false;
    }
}
