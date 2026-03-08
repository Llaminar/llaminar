/**
 * @file ROCmQuantisedGemmKernel_INT8_VNNI.hip
 * @brief INT8 VNNI wide-tile GEMM kernels for AMD MI50/MI60 (gfx906) prefill
 *
 * @target AMD Instinct MI50 / MI60 (gfx906), 60 CUs, 4 SIMDs/CU
 * @isa    v_dot4_i32_i8 (VNNI: 4-way signed INT8 dot product, 1 cycle/op)
 * @op     C[M×N] (INT32) = A[M×K] (INT8) × B[K×N] (INT8, VNNI-packed)
 *
 * This file contains the production INT8×INT8→INT32 GEMM kernels used for the
 * **prefill** (prompt-processing) phase on AMD gfx906 GPUs.  All kernels use
 * the `v_dot4_i32_i8` instruction to compute 4 multiply-accumulates per cycle.
 *
 * Scaling from INT32 to FP32 is handled *outside* this file, in
 * ROCmQuantisedGemmKernel.hip (the common utilities file).
 *
 * ============================================================================
 * INT8 VNNI MEMORY LAYOUT
 * ============================================================================
 *
 * **A matrix (activations)** — standard row-major INT8:
 *
 *   A[M × K] — each row is K contiguous INT8 values.
 *   Accessed as int32_t* → each load fetches 4 packed INT8 elements.
 *
 *     Row m:  [ a(m,0) a(m,1) a(m,2) a(m,3) | a(m,4) … a(m,K-1) ]
 *               ↑ 1 int32_t load = 4 INT8s ↑
 *
 * **B matrix (weights)** — VNNI-packed layout [K/4 × N × 4]:
 *
 *   Standard B would be [K × N] row-major, where B[k][n] is one INT8.
 *   VNNI packing groups 4 consecutive K elements for each N column:
 *
 *     B_vnni[kg][n] = { B[4*kg+0][n], B[4*kg+1][n], B[4*kg+2][n], B[4*kg+3][n] }
 *
 *   This packs 4 INT8s into one int32_t, so that a single `v_dot4_i32_i8`
 *   instruction can compute:
 *
 *     acc += A(m, 4*kg+0) * B(4*kg+0, n)
 *          + A(m, 4*kg+1) * B(4*kg+1, n)
 *          + A(m, 4*kg+2) * B(4*kg+2, n)
 *          + A(m, 4*kg+3) * B(4*kg+3, n)
 *
 *   Memory layout visualization (K=8, N=4):
 *
 *     kg=0: [ {b(0,0)..b(3,0)} {b(0,1)..b(3,1)} {b(0,2)..b(3,2)} {b(0,3)..b(3,3)} ]
 *     kg=1: [ {b(4,0)..b(7,0)} {b(4,1)..b(7,1)} {b(4,2)..b(7,2)} {b(4,3)..b(7,3)} ]
 *             ↑ 1 int32_t = 4 INT8s from consecutive K rows for column n ↑
 *
 *   Stride: kg-groups are N int32_t's apart (one int32_t per column).
 *   This layout is created by the weight upload/repacking step in the
 *   common file (ROCmQuantisedGemmKernel.hip → uploadWeights).
 *
 * **Why VNNI packing?**
 *   The `v_dot4_i32_i8` instruction requires both operands as packed int32_t.
 *   Without VNNI packing, B elements from consecutive K rows would be
 *   scattered across memory (stride = N bytes apart), requiring 4 separate
 *   byte loads.  VNNI packing makes them contiguous → 1 dword load feeds
 *   the dot product directly.
 *
 * ============================================================================
 * KERNEL VARIANTS (Production Auto-Dispatch)
 * ============================================================================
 *
 * Two wide-tile variants handle all production prefill GEMM shapes:
 *
 *   V3 (N_TILE=64, LDS double-buffered pipeline)
 *     - Best for K-heavy shapes: Attention (K≈N), FFN_Down (K>>N)
 *     - LDS double-buffering overlaps global→LDS loads with compute
 *     - KT=8 default (auto), supports KT=8 or KT=16
 *     - ~66 VGPRs, 3 waves/SIMD (KT=8), 0 spills
 *
 *   V7 (N_TILE=128, safe-tile split with branchless interior)
 *     - Best for N-heavy shapes: FFN_Up/Gate (N>>K), LM_Head (N>>>K)
 *     - Splits outer loop into safe (unconditional vectorized loads) and
 *       boundary (checked loads) loops; for aligned shapes the boundary
 *       loop never executes → tighter ISA, better icache utilization
 *     - KT=16 default (auto), supports KT=8 or KT=16
 *     - ~117 VGPRs, 2 waves/SIMD, 0 spills
 *     - Matches or beats Composable Kernel (CK) on all N-heavy shapes
 *
 * Auto-dispatch logic (in ROCmQuantisedGemmKernel.cpp):
 *   K >= N  → V3/KT8   (K-heavy: long K-loop benefits from LDS pipelining)
 *   K <  N  → V7/KT16  (N-heavy: safe-tile split eliminates branch overhead)
 *
 * Additionally, two fallback kernel families exist for non-wide-tile cases:
 *
 *   Baseline prefill  — simple per-thread GEMM with configurable tile geometry
 *   Grid split-K      — K-dimension parallelism via atomicAdd accumulation
 *
 * Env var overrides for benchmarking:
 *   LLAMINAR_ROCM_WIDE_TILE_V3=1     → Force V3 for all wide-tile shapes
 *   LLAMINAR_ROCM_WIDE_TILE_V7=1     → Force V7 for all wide-tile shapes
 *   LLAMINAR_ROCM_WIDE_TILE_KT=8|16  → Override KT parameter
 *
 * For the full development history (V1→V7 evolution, ISA analysis, profiling
 * methodology), see: kernels/rocm/README.vnni-gemm-tuning.md
 *
 * ============================================================================
 * WIDE-TILE GEOMETRY
 * ============================================================================
 *
 * Both wide-tile kernels use 256 threads.  The M-dimension tile size is
 * parameterized via the M_TILE template parameter (16, 32, 64, or 128):
 *
 *   | M_TILE | THREADS_M | THREADS_N | V3 N/thr | V7 N/thr | V3 Acc | V7 Acc |
 *   |--------|-----------|-----------|----------|----------|--------|--------|
 *   |  16    |     4     |    64     |    1     |    2     |   4    |   8    |
 *   |  32    |     8     |    32     |    2     |    4     |   8    |  16    |
 *   |  64    |    16     |    16     |    4     |    8     |  16    |  32    |
 *   | 128    |    32     |     8     |    8     |   16     |  32    |  64    |
 *
 * Dispatch selects M_TILE based on M: small M uses a smaller tile to
 * minimize wasted compute; large M uses M_TILE=128 for best B-data reuse.
 *
 * Grid: (ceil(N / N_TILE), ceil(M / M_TILE), 1) — 2D tiling over M and N.
 * Each workgroup covers up to M_TILE M-rows × N_TILE N-cols.  Boundary
 * handling for partial M-tiles (last workgroup) is via per-element
 * bounds checks in loads and stores.
 */

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// WIDE-TILE V3: LDS Double-Buffered Pipeline
// ============================================================================

/**
 * @brief Wide-tile V3 INT8 GEMM kernel: LDS double-buffered pipeline, N_TILE=64.
 *
 * @tparam N_TILE  Number of N-columns per workgroup tile (always 64 for V3).
 * @tparam KT      K-tile depth in k-groups (each k-group = 4 INT8 elements).
 *                  KT=8 → 32 K-elements/tile, KT=16 → 64 K-elements/tile.
 * @tparam M_TILE  M-dimension tile height (16, 32, 64, or 128; default 128).
 *                  Smaller M_TILE reduces waste for small M (tail handling).
 *
 * @param d_A_int8      [M × K] INT8 activations, row-major.
 * @param d_B_int8_vnni [K/4 × N × 4] INT8 weights, VNNI-packed (see file header).
 * @param d_C_int32     [M × N] INT32 output accumulator, row-major.
 * @param M  Number of rows (tokens). Tiled in chunks of M_TILE via grid.y.
 * @param N  Number of columns (output features).
 * @param K  Reduction dimension. Must be multiple of 4.
 *
 * **Strategy**: Software-pipelined LDS double-buffering.
 *
 * The K dimension is tiled into chunks of KT k-groups.  Two LDS buffers
 * (ping/pong) allow the kernel to overlap global memory loads for tile T+1
 * with v_dot4 compute on tile T.  This hides global memory latency, which
 * is the dominant bottleneck for K-heavy shapes where the K-loop is long.
 *
 * **Structure**:
 *   1. PRELOAD — Load first K-tile (tile 0) into LDS buf[0]
 *   2. MAIN LOOP — For tiles 1..num_tiles-1:
 *        a. Issue global loads for tile T into staging registers (async)
 *        b. Compute tile T-1 from LDS buf[(T-1)%2] (overlaps with a.)
 *        c. __syncthreads()  — all reads from current LDS buffer done
 *        d. Write staging registers into LDS buf[T%2]
 *        e. __syncthreads()  — LDS writes visible to all threads
 *   3. FINAL TILE — Compute the last tile from LDS
 *   4. STORE — Write accumulators to global memory
 *
 * **Cooperative loads**: 256 threads cooperatively load the A and B tiles
 * into LDS.  With KT=16, A requires 2 passes (256 threads / 4 k-chunks
 * = 64 rows/pass, need 128 rows total).  B fits in a single pass when
 * B_ACTIVE_THREADS ≤ 256.
 *
 * **Key improvements over V1** (based on CK ISA analysis — see
 * changelog/2025-07-24-ck-vs-v1-isa-analysis.md):
 *   1. LDS double-buffering: overlap loads with compute (~5-8% gain)
 *   2. Multi-pass A load for KT=16: halves loop iterations (~2-3% gain)
 *   3. Software-pipelined main loop: issues loads before compute
 *
 * **LDS budget** (for default M_TILE=128; smaller M_TILE reduces proportionally):
 *   KT=8:  a_lds[2][8×128] + b_lds[2][8×64] = 12 KB → 5 workgroups/CU
 *   KT=16: a_lds[2][16×128] + b_lds[2][16×64] = 24 KB → 2 workgroups/CU
 *
 * **Register budget** (~66 VGPRs for KT=8):
 *   32 accumulators + 8 a_staging + 4 b_staging + 12 operands + ~10 misc
 *   → 3 waves/SIMD, 0 register spills
 */

template <int N_TILE, int KT, int M_TILE = 128>
__global__ __launch_bounds__(256, 0)
void qgemm_int8_vnni_wide_tile_v3_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    // -------------------------------------------------------------------------
    // Thread layout: 256 threads arranged as THREADS_M × THREADS_N where
    // THREADS_M = M_TILE / 4, THREADS_N = 256 / THREADS_M.
    // Each thread computes a 4 × N_PER_THREAD sub-tile of the output.
    // One workgroup covers an M_TILE × N_TILE output tile.
    //
    //   M_TILE=128: 32×8,  N/thr=8,  32 acc  (default, best B-data reuse)
    //   M_TILE=64:  16×16, N/thr=4,  16 acc
    //   M_TILE=32:  8×32,  N/thr=2,   8 acc
    //   M_TILE=16:  4×64,  N/thr=1,   4 acc  (minimum waste for small M)
    // -------------------------------------------------------------------------
    constexpr int BLOCK_SIZE = 256;
    constexpr int M_PER_THREAD = 4;                    // Each thread handles 4 M-rows
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;   // M_TILE/4 threads spanning M
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M;   // Remaining threads span N
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;    // N-cols per thread

    static_assert(M_TILE == 16 || M_TILE == 32 || M_TILE == 64 || M_TILE == 128,
                  "M_TILE must be 16, 32, 64, or 128");
    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT % 4 == 0, "KT must be multiple of 4 for vectorized A loads");
    static_assert(N_TILE % 4 == 0, "N_TILE must be multiple of 4 for vectorized B loads");

    // Each workgroup handles one (M-tile, N-tile) pair.
    // Grid is 2D: (ceil(N / N_TILE), ceil(M / M_TILE)) — blockIdx.x selects
    // the N-tile, blockIdx.y selects the M-tile.
    const int n_block = blockIdx.x * N_TILE;
    const int m_block = blockIdx.y * M_TILE;  // Global M-row offset for this tile

    // Map flat threadIdx.x to 2D (m, n) position within the tile.
    // Thread (t_m=5, t_n=2) computes output rows [m_block+20..m_block+23]
    // × cols [n_block+16..n_block+23].
    const int t_m = threadIdx.x % THREADS_M; // M-thread index: 0..31
    const int t_n = threadIdx.x / THREADS_M; // N-thread index: 0..7

    const int m_base = t_m * M_PER_THREAD;            // First M-row this thread owns (tile-local)
    const int n_base = n_block + t_n * N_PER_THREAD;  // First N-col this thread owns (global)

    // =========================================================================
    // LDS: Double-buffered (ping-pong between buf[0] and buf[1])
    //
    // Two copies of each tile in LDS allow overlapping global loads into
    // buf[next] while compute reads from buf[current].  Layout:
    //   a_lds[buf][kk * M_TILE + mi] — A tile transposed into LDS so that
    //     consecutive threads read consecutive M-rows (coalesced LDS reads).
    //   b_lds[buf][kk * N_TILE + ni] — B tile in natural layout.
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // Accumulators: M_PER_THREAD × N_PER_THREAD = 4 × 8 = 32 INT32 values.
    // These stay in registers across all K-tile iterations and accumulate
    // the v_dot4 results.  Final values are the complete dot products.
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    // K is measured in INT8 elements; k-groups are packs of 4 (one int32_t).
    const int total_k_groups = K / 4;

    // Reinterpret INT8 pointers as int32_t* for vectorized 4-byte loads.
    // Each int32_t contains 4 packed INT8 values — one k-group.
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4; // Number of k-groups per A row

    // =========================================================================
    // A cooperative load geometry — multi-pass for KT ≥ 16
    //
    // All 256 threads cooperate to load the A tile [KT × M_TILE] into LDS.
    // Each thread loads one uint4 (4 consecutive k-groups) for one M-row.
    //
    // A_K_CHUNKS = KT/4: how many uint4 loads span the KT k-groups.
    //   KT=8  → A_K_CHUNKS=2 → 256/2 = 128 rows per pass → 1 pass
    //   KT=16 → A_K_CHUNKS=4 → 256/4 = 64 rows per pass  → 2 passes
    //
    // Thread assignment: threadIdx.x is split into (k-chunk, M-row).
    //   a_kg: which group of 4 k-groups this thread loads (k offset in LDS).
    //   a_thread_mi_base: which M-row this thread loads in pass 0.
    // =========================================================================
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    // Per-thread A load coordinates (computed once, reused every K-tile)
    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;  // K-offset in k-groups
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;  // M-row in pass 0

    // Pre-compute M-row index for each pass.  Pass p loads row
    // a_thread_mi_base + p * A_ROWS_PER_PASS.  Out-of-bounds rows
    // (mi >= M) are handled with zero-fill at load time.
    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // =========================================================================
    // B cooperative load geometry
    //
    // B tile [KT × N_TILE] is loaded in a single pass.  Each active thread
    // loads one uint4 (4 consecutive N-columns for one k-row).
    //
    // B_N_CHUNKS = N_TILE/4: number of uint4 loads to cover N_TILE columns.
    // B_ACTIVE_THREADS = KT × B_N_CHUNKS: total loads needed for the B tile.
    //   KT=8,  N_TILE=64 → 8×16 = 128 active threads (first half of block)
    //   KT=16, N_TILE=64 → 16×16 = 256 active threads (all threads used)
    //
    // Thread assignment: first B_ACTIVE_THREADS threads participate.
    //   b_kk_local: k-row within the tile (0..KT-1)
    //   b_nl: N-column offset within the tile (multiple of 4)
    //   b_gn: global N-column index
    //   b_lds_off: linear offset into b_lds[buf]
    // =========================================================================
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_ACTIVE_THREADS = KT * B_N_CHUNKS;
    static_assert(B_ACTIVE_THREADS <= BLOCK_SIZE,
                  "B tile too large for single-pass cooperative load");

    const int b_kk_local = threadIdx.x / B_N_CHUNKS;   // K-row within tile
    const int b_nl = (threadIdx.x % B_N_CHUNKS) * 4;   // N-col offset (×4)
    const int b_gn = n_block + b_nl;                    // Global N-column
    const int b_lds_off = b_kk_local * N_TILE + b_nl;   // LDS write offset

    // Number of K-tiles.  Each tile covers KT k-groups (= KT*4 INT8 elements).
    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    // Staging registers: temporaries between global load and LDS write.
    // Global loads are slow (~300 cycles); by storing results in staging
    // registers, we can overlap compute on the current tile while waiting
    // for the next tile's loads to complete.
    uint4 a_staging[A_NUM_PASSES]; // One uint4 per pass (4 k-groups at once)
    uint4 b_staging;               // Single uint4 for this thread's B load

    // =========================================================================
    // PRELOAD: Load first tile (tile 0) into buf[0]
    //
    // This kicks off the pipeline: tile 0 goes into LDS so that the main
    // loop can begin computing tile 0 while loading tile 1.
    // =========================================================================

    // A loads — multi-pass cooperative loading into staging registers.
    // Fast path: all 4 k-groups are in-bounds → single vectorized uint4 load.
    // Slow path: boundary tile → scalar loads with per-element bounds checks.
    // Note: a_mi[p] is tile-local (0..127); global row = m_block + a_mi[p].
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
    {
        const int mi = a_mi[p];
        if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
        const int gm = m_block + mi;  // Global M-row index
        const int gkg = a_kg; // kt_base = 0
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

    // B load — single pass, only first B_ACTIVE_THREADS threads participate.
    // B is stored in VNNI layout: [k_group × N], so consecutive N-columns
    // at the same k-row are contiguous → uint4 loads are naturally coalesced.
    b_staging = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < B_ACTIVE_THREADS)
    {
        const int gkk = b_kk_local; // kt_base = 0
        if (__builtin_expect(b_gn + 3 < N && gkk < total_k_groups, 1))
        {
            b_staging = *reinterpret_cast<const uint4*>(
                b_global + static_cast<int64_t>(gkk) * N + b_gn);
        }
        else
        {
            b_staging.x = (b_gn + 0 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 0] : 0;
            b_staging.y = (b_gn + 1 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 1] : 0;
            b_staging.z = (b_gn + 2 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 2] : 0;
            b_staging.w = (b_gn + 3 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 3] : 0;
        }
    }

    // Write staged data to buf[0].  Note the TRANSPOSE for A:
    // Global A is [M × K/4] row-major, but LDS stores as [K × M] so that
    // consecutive threads (adjacent t_m) read adjacent LDS addresses.
    // This converts scattered global reads into coalesced LDS reads.
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
    if (threadIdx.x < B_ACTIVE_THREADS)
    {
        *reinterpret_cast<uint4*>(&b_lds[0][b_lds_off]) = b_staging;
    }

    __syncthreads(); // buf[0] ready for compute

    // =========================================================================
    // MAIN LOOP: Double-buffered pipeline
    //
    // For each tile t (1 .. num_tiles-1):
    //   1. Issue global loads for tile t      → staging regs  (async)
    //   2. Compute tile (t-1) from buf[(t-1)%2]               (overlaps!)
    //   3. __syncthreads()                    (all reads from cur_buf done)
    //   4. Write staging → buf[t%2]           (vmcnt(0) ensures loads done)
    //   5. __syncthreads()                    (LDS writes visible)
    // =========================================================================
    for (int tile = 1; tile < num_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1; // buffer to compute from
        const int nxt_buf = tile & 1;       // buffer to load into
        const int kt_base = tile * KT;

        // --- 1. Issue ALL global loads for next tile (async) ---

        // A loads — multi-pass (global row = m_block + a_mi[p])
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;  // Global M-row index
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

        // B load
        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            const int gkk = kt_base + b_kk_local;
            if (__builtin_expect(b_gn + 3 < N && gkk < total_k_groups, 1))
            {
                b_staging = *reinterpret_cast<const uint4*>(
                    b_global + static_cast<int64_t>(gkk) * N + b_gn);
            }
            else
            {
                b_staging = make_uint4(0, 0, 0, 0);
                b_staging.x = (b_gn + 0 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 0] : 0;
                b_staging.y = (b_gn + 1 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 1] : 0;
                b_staging.z = (b_gn + 2 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 2] : 0;
                b_staging.w = (b_gn + 3 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 3] : 0;
            }
        }

        // --- 2. Compute on CURRENT buffer (overlaps with global loads!) ---
        // Each thread reads its M_PER_THREAD×1 column from A and 1×N_PER_THREAD
        // row from B, then does M_PER_THREAD × N_PER_THREAD = 32 v_dot4 ops.
        // v_dot4 (sdot4): acc += Σ(a_byte[i] * b_byte[i]) for i=0..3
        // This is the VNNI dot product — 4 INT8 multiply-adds per cycle.
        #pragma unroll
        for (int kk = 0; kk < KT; ++kk)
        {
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
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }

        // --- 3. Barrier: all threads done reading cur_buf ---
        __syncthreads();

        // --- 4. Write staged data to NEXT buffer ---
        // (compiler inserts s_waitcnt vmcnt(0) here for the global loads)
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
        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            *reinterpret_cast<uint4*>(&b_lds[nxt_buf][b_lds_off]) = b_staging;
        }

        // --- 5. Barrier: LDS writes visible before next iteration ---
        __syncthreads();
    }

    // =========================================================================
    // FINAL TILE: Compute the last tile (already loaded into LDS)
    //
    // After the main loop ends, the last tile's data is in LDS but hasn't
    // been consumed yet (the loop loads tile T, computes tile T-1).  We
    // compute it here without any loads — just pure ALU work.
    // =========================================================================
    {
        const int last_buf = (num_tiles - 1) & 1;

        #pragma unroll
        for (int kk = 0; kk < KT; ++kk)
        {
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
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }
    }

    // =========================================================================
    // STORE RESULTS
    //
    // Each thread writes its M_PER_THREAD × N_PER_THREAD INT32 accumulators
    // to global memory.  When N_PER_THREAD >= 4, a vectorized uint4 store
    // path is used for in-bounds columns.  For smaller N_PER_THREAD (M_TILE
    // 16 or 32), scalar stores are used unconditionally.
    // Note: m_idx is the GLOBAL M-row index (m_block + tile-local offset).
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;  // Global M-row
        if (m_idx < M)
        {
            if constexpr (N_PER_THREAD >= 4)
            {
                if (n_base + N_PER_THREAD - 1 < N)
                {
                    // All N_PER_THREAD columns valid — vectorized uint4 stores
                    const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                    #pragma unroll
                    for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                    {
                        *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                            make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                       static_cast<uint32_t>(acc[mm][chunk + 1]),
                                       static_cast<uint32_t>(acc[mm][chunk + 2]),
                                       static_cast<uint32_t>(acc[mm][chunk + 3]));
                    }
                }
                else
                {
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    {
                        const int n_idx = n_base + nn;
                        if (n_idx < N)
                            d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
            else
            {
                // N_PER_THREAD < 4: scalar stores (M_TILE 16 or 32)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                }
            }
        }
    }
}

/**
 * @brief Wide-tile V7 GEMM kernel: safe-tile split with branchless interior loop.
 *
 * @tparam N_TILE  N-dimension tile width (128 for V7).
 * @tparam KT      K-tile depth in k-groups of 4 INT8 elements (8 or 16).
 * @tparam M_TILE  M-dimension tile height (16, 32, 64, or 128; default 128).
 *                  Smaller M_TILE reduces waste for small M (tail handling).
 *
 * @param d_A_int8      [M × K] row-major INT8 activations.
 * @param d_B_int8_vnni [K/4 × N × 4] VNNI-packed INT8 weights.
 * @param d_C_int32     [M × N] INT32 accumulator output (row-major).
 * @param M  Number of rows in A / rows in C. Tiled in chunks of M_TILE via grid.y.
 * @param N  Number of columns in B / columns in C.
 * @param K  Shared (contraction) dimension in INT8 elements.
 *
 * **Purpose & Specialization**:
 * V7 addresses outer-pipeline overhead identified by rocprof ISA analysis.
 * Profiling showed V5's single loop body contained both fast (vectorized)
 * and slow (boundary-checked) code paths, totalling ~99 branches and ~800
 * extra instructions that pollute the 16 KB/CU L1I$ on gfx906.
 *
 * **Strategy — Two-Loop Split**:
 *   - **Safe loop** (tiles 0..num_safe-1): ALL loads are unconditional
 *     vectorized (`global_load_dwordx4`).  Zero boundary checks, zero
 *     branches in the load path.  Compiler generates maximally tight ISA.
 *   - **Boundary loop** (tiles num_safe..num_tiles-1): Full per-element
 *     bounds checking.  Runs for the last 0-2 tiles (or never for aligned
 *     shapes).
 *
 * **Safe Tile Conditions** (all must hold):
 *   1. `m_block + M_TILE <= M` — all A row indices for this M-tile in-bounds
 *   2. `n_block + N_TILE <= N` — all B column indices in-bounds
 *   3. `(tile+1) * KT <= total_k_groups` — all K-group indices in-bounds
 *
 * **Structure**:
 *   1. Thread layout + LDS allocation (same as V3 but N_TILE=128)
 *   2. Pre-compute `num_safe` tile count
 *   3. Define safe/boundary load lambdas (always_inline)
 *   4. Preload tile 0 (safe or boundary depending on num_safe)
 *   5. Safe loop: tiles 1..num_safe-1 with branchless loads
 *   6. Boundary loop: remaining tiles with per-element checks
 *   7. Final tile compute + vectorized store
 *
 * **rocprof Motivation** (7B FFN_Up, M=128 N=18944 K=3584):
 *   - V5 kernel: 0.771ms, CK kernel: 0.723ms (6.6% gap)
 *   - V5 ISA: 15 fast-path + 84 boundary-path = 99 loop branches
 *   - V7 safe loop: 15 branches only (boundary code eliminated)
 *   - Result: V7 matches or beats CK on N-heavy shapes
 *
 * **Register Budget**: 64 acc + 20 staging + 20 operand + ~13 = 117 VGPRs
 *   → 0 spills, 2 waves/SIMD (same as V5).
 *
 * **LDS Budget**: 2 × (KT × M_TILE + KT × N_TILE) × 4 bytes
 *   For default M_TILE=128:
 *   - KT=8:  2 × (8×128 + 8×128) × 4 = 16,384 bytes
 *   - KT=16: 2 × (16×128 + 16×128) × 4 = 32,768 bytes (LDS limit)
 *   Smaller M_TILE reduces LDS proportionally (e.g., M_TILE=32/KT=16 → 20 KB).
 */
template <int N_TILE, int KT, int M_TILE = 128>
__global__ __launch_bounds__(256, 2)
void qgemm_int8_vnni_wide_tile_v7_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    // -------------------------------------------------------------------------
    // Thread layout: 256 threads arranged as THREADS_M × THREADS_N where
    // THREADS_M = M_TILE / 4, THREADS_N = 256 / THREADS_M.
    // Each thread computes a 4 × N_PER_THREAD sub-tile of the output.
    // One workgroup covers an M_TILE × N_TILE output tile.
    //
    //   M_TILE=128: 32×8,  N/thr=16, 64 acc  (default, best B-data reuse)
    //   M_TILE=64:  16×16, N/thr=8,  32 acc
    //   M_TILE=32:  8×32,  N/thr=4,  16 acc
    //   M_TILE=16:  4×64,  N/thr=2,   8 acc  (minimum waste for small M)
    // -------------------------------------------------------------------------
    constexpr int BLOCK_SIZE = 256;
    constexpr int M_PER_THREAD = 4;                    // Each thread handles 4 M-rows
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;   // M_TILE/4 threads spanning M
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M;   // Remaining threads span N
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;    // N-cols per thread

    static_assert(M_TILE == 16 || M_TILE == 32 || M_TILE == 64 || M_TILE == 128,
                  "M_TILE must be 16, 32, 64, or 128");
    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(KT % 4 == 0);
    static_assert(N_TILE % 4 == 0);
    static_assert(M_PER_THREAD == 4, "Vectorized A load requires M_PER_THREAD == 4");

    const int n_block = blockIdx.x * N_TILE;
    const int m_block = blockIdx.y * M_TILE;  // Global M-row offset for this tile
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;            // Tile-local M offset
    const int n_base = n_block + t_n * N_PER_THREAD;

    // =========================================================================
    // LDS: Double-buffered (same layout as V3)
    // =========================================================================
    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // 64 accumulators per thread (4 M-rows × 16 N-cols).
    // These stay in registers across all K-tile iterations.
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    const int total_k_groups = K / 4;
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    // =========================================================================
    // A cooperative load geometry
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
    // B cooperative load geometry — multi-pass for N_TILE=128
    //
    // Unlike V3 (single-pass B load with 128 active threads), V7's larger
    // N_TILE=128 requires more loads: B_TOTAL_GROUPS = KT × (N_TILE/4).
    //   KT=8,  N_TILE=128 → 8×32 = 256 loads = 1 pass (all 256 threads)
    //   KT=16, N_TILE=128 → 16×32 = 512 loads = 2 passes
    //
    // Each pass assigns a flat index to each thread, which maps to a
    // (k-row, N-chunk) coordinate via div/mod.  The BLoadCoord struct
    // caches these per-pass so they're computed once and reused.
    // =========================================================================
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_TOTAL_GROUPS = KT * B_N_CHUNKS;
    constexpr int B_NUM_PASSES = (B_TOTAL_GROUPS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    struct BLoadCoord {
        int kk_local;
        int nl;
        int gn;
        int lds_off;
        bool active;
    };
    BLoadCoord b_coords[B_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < B_NUM_PASSES; ++p)
    {
        const int flat = p * BLOCK_SIZE + threadIdx.x;
        if (flat < B_TOTAL_GROUPS)
        {
            b_coords[p].kk_local = flat / B_N_CHUNKS;
            b_coords[p].nl = (flat % B_N_CHUNKS) * 4;
            b_coords[p].gn = n_block + b_coords[p].nl;
            b_coords[p].lds_off = b_coords[p].kk_local * N_TILE + b_coords[p].nl;
            b_coords[p].active = true;
        }
        else
        {
            b_coords[p] = {0, 0, 0, 0, false};
        }
    }

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging[B_NUM_PASSES];

    // =========================================================================
    // V7 CHANGE: Pre-compute safe tile count
    //
    // Safe tiles are those where ALL loads (A and B) are guaranteed in-bounds
    // for ALL threads. This allows the safe loop to use unconditional
    // vectorized loads with zero boundary checks.
    //
    // Safe conditions (all must hold):
    //   1. m_block + M_TILE <= M (all A row indices for this M-tile in-bounds)
    //   2. n_block + N_TILE <= N (all B column accesses within bounds)
    //   3. (tile+1)*KT <= total_k_groups (all K-group accesses within bounds)
    //
    // num_safe_tiles = number of tiles satisfying condition 3 (0-indexed).
    // If conditions 1 or 2 fail, num_safe_tiles = 0 (all tiles boundary-checked).
    // =========================================================================
    const bool a_m_safe  = (m_block + M_TILE <= M);
    const bool b_n_safe  = (n_block + N_TILE <= N);
    const int  safe_k    = total_k_groups / KT;  // floor: tiles with full K coverage
    const int  num_safe  = (a_m_safe && b_n_safe) ? (safe_k < num_tiles ? safe_k : num_tiles) : 0;

    // =========================================================================
    // SAFE load lambdas: unconditional vectorized loads (no boundary checks)
    // Note: global A row = m_block + a_mi[p] (tile-local → global).
    // =========================================================================
    auto safe_issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (a_mi[p] >= M_TILE) continue; }
            a_staging[p] = *reinterpret_cast<const uint4*>(
                a_global + static_cast<int64_t>(m_block + a_mi[p]) * a_row_stride + kt_base + a_kg);
        }
    };

    auto safe_issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                b_staging[p] = *reinterpret_cast<const uint4*>(
                    b_global + static_cast<int64_t>(kt_base + b_coords[p].kk_local) * N + b_coords[p].gn);
            }
        }
    };

    // =========================================================================
    // BOUNDARY load lambdas: with boundary checks
    // Note: global A row = m_block + mi (tile-local → global).
    // =========================================================================
    auto boundary_issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            const int gm = m_block + mi;  // Global M-row
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
    };

    auto boundary_issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                const int gkk = kt_base + b_coords[p].kk_local;
                const int gn = b_coords[p].gn;
                if (__builtin_expect(gn + 3 < N && gkk < total_k_groups, 1))
                {
                    b_staging[p] = *reinterpret_cast<const uint4*>(
                        b_global + static_cast<int64_t>(gkk) * N + gn);
                }
                else
                {
                    b_staging[p] = make_uint4(0, 0, 0, 0);
                    b_staging[p].x = (gn + 0 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 0] : 0;
                    b_staging[p].y = (gn + 1 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 1] : 0;
                    b_staging[p].z = (gn + 2 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 2] : 0;
                    b_staging[p].w = (gn + 3 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 3] : 0;
                }
            }
        }
    };

    // =========================================================================
    // LDS write lambdas (same as V5)
    // =========================================================================
    auto write_a_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            a_lds[buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }
    };

    auto write_b_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                *reinterpret_cast<uint4*>(&b_lds[buf][b_coords[p].lds_off]) = b_staging[p];
            }
        }
    };

    // =========================================================================
    // Compute tile: reads from LDS, computes v_dot4 for all K-steps.
    //
    // Unlike V3's fully-unrolled inner loop, V7 uses `#pragma clang loop
    // unroll(disable)` on the K-step loop.  With 64 accumulators (vs V3's
    // 32), full unrolling would double register pressure and risk spills.
    // The un-unrolled loop lets the compiler schedule instructions across
    // iterations instead.
    //
    // A loads use uint4 vectorized LDS reads (4 M-rows at once).
    // B loads use uint4 vectorized LDS reads (4 N-cols at once).
    // Inner compute: 4×16 = 64 v_dot4 ops per K-step.
    // =========================================================================
    auto compute_tile = [&](int buf) __attribute__((always_inline))
    {
        const int32_t* __restrict__ a_ptr = &a_lds[buf][m_base];
        const int32_t* __restrict__ b_ptr = &b_lds[buf][t_n * N_PER_THREAD];

        #pragma clang loop unroll(disable)
        for (int kk = 0; kk < KT; ++kk)
        {
            int32_t a_reg[M_PER_THREAD];
            int32_t b_reg[N_PER_THREAD];

            *reinterpret_cast<uint4*>(&a_reg[0]) =
                *reinterpret_cast<const uint4*>(&a_ptr[kk * M_TILE]);

            // B load: vectorized for N_PER_THREAD >= 4, scalar for smaller.
            if constexpr (N_PER_THREAD >= 4)
            {
                #pragma unroll
                for (int c = 0; c < N_PER_THREAD; c += 4)
                    *reinterpret_cast<uint4*>(&b_reg[c]) =
                        *reinterpret_cast<const uint4*>(&b_ptr[kk * N_TILE + c]);
            }
            else if constexpr (N_PER_THREAD == 2)
            {
                *reinterpret_cast<uint2*>(&b_reg[0]) =
                    *reinterpret_cast<const uint2*>(&b_ptr[kk * N_TILE]);
            }
            else
            {
                b_reg[0] = b_ptr[kk * N_TILE];
            }

            #pragma unroll
            for (int mm = 0; mm < M_PER_THREAD; ++mm)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    acc[mm][nn] = __builtin_amdgcn_sdot4(
                        a_reg[mm], b_reg[nn], acc[mm][nn], false);
        }
    };

    // =========================================================================
    // PRELOAD: tile 0 (safe or boundary)
    // =========================================================================
    if (num_safe > 0)
    {
        safe_issue_a_loads(0);
        safe_issue_b_loads(0);
    }
    else
    {
        boundary_issue_a_loads(0);
        boundary_issue_b_loads(0);
    }
    write_a_to_lds(0);
    write_b_to_lds(0);
    __syncthreads();

    // =========================================================================
    // SAFE LOOP: tiles 1..num_safe-1 (zero boundary checks)
    //
    // This is the hot path for aligned shapes. The compiler generates a
    // tight loop body without any boundary-checking branches or fallback
    // scalar load code, resulting in smaller ISA and better icache utilization.
    // =========================================================================
    for (int tile = 1; tile < num_safe; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int kt_base = tile * KT;

        safe_issue_a_loads(kt_base);
        safe_issue_b_loads(kt_base);

        compute_tile(cur_buf);

        __syncthreads();

        write_a_to_lds(nxt_buf);
        write_b_to_lds(nxt_buf);

        __syncthreads();
    }

    // =========================================================================
    // BOUNDARY LOOP: tiles max(1, num_safe)..num_tiles-1 (with boundary checks)
    //
    // Handles the last 0-2 tiles where K or N boundary checks are needed.
    // For aligned shapes (M=128, N%128==0, K%32==0), this loop never executes.
    // =========================================================================
    {
        const int boundary_start = (num_safe > 0) ? num_safe : 1;
        for (int tile = boundary_start; tile < num_tiles; ++tile)
        {
            const int cur_buf = (tile - 1) & 1;
            const int nxt_buf = tile & 1;
            const int kt_base = tile * KT;

            boundary_issue_a_loads(kt_base);
            boundary_issue_b_loads(kt_base);

            compute_tile(cur_buf);

            __syncthreads();

            write_a_to_lds(nxt_buf);
            write_b_to_lds(nxt_buf);

            __syncthreads();
        }
    }

    // =========================================================================
    // FINAL TILE
    // =========================================================================
    compute_tile((num_tiles - 1) & 1);

    // =========================================================================
    // STORE RESULTS
    // Vectorized uint4 stores when N_PER_THREAD >= 4, scalar otherwise.
    // Note: m_idx is the GLOBAL M-row index (m_block + tile-local offset).
    // =========================================================================
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;  // Global M-row
        if (m_idx < M)
        {
            if constexpr (N_PER_THREAD >= 4)
            {
                if (n_base + N_PER_THREAD - 1 < N)
                {
                    const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                    #pragma unroll
                    for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                    {
                        *reinterpret_cast<uint4*>(d_C_int32 + row_off + chunk) =
                            make_uint4(static_cast<uint32_t>(acc[mm][chunk + 0]),
                                       static_cast<uint32_t>(acc[mm][chunk + 1]),
                                       static_cast<uint32_t>(acc[mm][chunk + 2]),
                                       static_cast<uint32_t>(acc[mm][chunk + 3]));
                    }
                }
                else
                {
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    {
                        const int n_idx = n_base + nn;
                        if (n_idx < N)
                            d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                    }
                }
            }
            else
            {
                // N_PER_THREAD < 4: scalar stores (M_TILE 16 or 32)
                #pragma unroll
                for (int nn = 0; nn < N_PER_THREAD; ++nn)
                {
                    const int n_idx = n_base + nn;
                    if (n_idx < N)
                        d_C_int32[static_cast<int64_t>(m_idx) * N + n_idx] = acc[mm][nn];
                }
            }
        }
    }
}

// ============================================================================
// BASELINE PREFILL KERNELS (Standard + Grid Split-K)
//
// These are the original (non-wide-tile) INT8 VNNI prefill kernels used as
// fallback paths when wide-tile V3/V7 aren't selected.  Dispatch priority:
//
//   Wide-tile (V3/V7)  →  Grid-kpar  →  Baseline prefill
//
// ============================================================================

/**
 * @brief Tile shape variants for the baseline prefill kernel.
 *
 * Each variant defines a (BLOCK_X × BLOCK_Y) thread grid with CPT
 * (columns per thread) = 4.  The tile shapes trade off M-parallelism
 * vs N-parallelism and are selected based on the M:N aspect ratio.
 *
 * | Variant    | BLOCK_X | BLOCK_Y | Output tile | Best when          |
 * |------------|---------|---------|-------------|--------------------|
 * | 16×16      | 16      | 16      | 64 × 16    | Balanced shapes    |
 * | 32×8       | 32      | 8       | 128 × 8    | N >> M (wide)      |
 * | 8×32       | 8       | 32      | 32 × 32    | M >> N (tall)      |
 * | 8×8        | 8       | 8       | 32 × 8     | Very small shapes  |
 */
enum PrefillInt8TileVariant
{
    PREFILL_TILE_16x16 = 0, ///< Balanced: 16 threads × 4 cols = 64 N-cols
    PREFILL_TILE_32x8 = 1,  ///< Wide N:   32 threads × 4 cols = 128 N-cols
    PREFILL_TILE_8x32 = 2,  ///< Tall M:   8 threads × 4 cols = 32 N-cols, 32 M-rows
    PREFILL_TILE_8x8 = 3,   ///< Small:    8 threads × 4 cols = 32 N-cols, 8 M-rows
};

/**
 * @brief Selects tile variant based on M:N aspect ratio.
 *
 * Uses **aspect ratio** (N/M) as the primary dispatch signal, not hardcoded
 * dimension thresholds, so the heuristic scales to arbitrary model shapes.
 *
 * **Dispatch logic** (evaluated in priority order):
 *
 *   1. **Very small M** (M ≤ 16, near-balanced aspect): use 8×8 tile to
 *      avoid wasting threads on empty M-rows.  The threshold M=16 is
 *      tile-relative (2× the smallest BLOCK_Y=8), not model-specific.
 *
 *   2. **Strongly N-heavy** (N/M ≥ 4): use 32×8 tile for maximum N
 *      coverage.  The 4:1 threshold means N is at least 4× wider than M.
 *
 *   3. **Strongly M-heavy** (M/N ≥ 4): use 8×32 tile for maximum M
 *      coverage.  The inverse of case 2.
 *
 *   4. **Moderately N-heavy with grid split-K** (N/M ≥ 2): use 32×8 tile.
 *      Split-K reduces each thread's K-work, so N-coverage matters more.
 *
 *   5. **Default** (balanced shapes): use 16×16 tile.
 *
 * @param M         Number of output rows (sequence length for prefill).
 * @param N         Number of output columns (hidden dimension).
 * @param K         Contraction dimension (unused, reserved for future use).
 * @param grid_kpar Whether grid-level K-parallelism (split-K) is active.
 * @return PrefillInt8TileVariant enum value.
 */
static int select_prefill_int8_tile_variant_auto(int M, int N, int K, bool grid_kpar)
{
    (void)K;

    // Compute aspect ratio: values > 1 mean N-heavy, < 1 mean M-heavy.
    // Guard against M=0 to avoid division by zero.
    const float aspect = static_cast<float>(N) / static_cast<float>(M > 0 ? M : 1);

    // 1. Very small M: use minimal 8×8 tile to avoid wasted M-threads.
    //    Threshold M ≤ 16 is tile-relative (2× smallest BLOCK_Y=8).
    //    Only applies when aspect is near-balanced — if N >> M, we'd
    //    rather use the 32×8 wide tile even with small M.
    if (M <= 16 && aspect < 4.0f)
    {
        return PREFILL_TILE_8x8;
    }

    // 2. Strongly N-heavy (N/M ≥ 4): 32×8 tile maximizes N-coverage.
    //    Example: M=128, N=4096 → aspect=32 → 32×8.
    if (aspect >= 4.0f)
    {
        return PREFILL_TILE_32x8;
    }

    // 3. Strongly M-heavy (M/N ≥ 4, i.e. aspect ≤ 0.25): 8×32 tile
    //    maximizes M-coverage.
    //    Example: M=512, N=64 → aspect=0.125 → 8×32.
    if (aspect <= 0.25f)
    {
        return PREFILL_TILE_8x32;
    }

    // 4. Moderately N-heavy with grid split-K (N/M ≥ 2): 32×8 tile.
    //    Split-K reduces per-thread K-work, making N-parallelism more
    //    important for saturating memory bandwidth.
    if (grid_kpar && aspect >= 2.0f)
    {
        return PREFILL_TILE_32x8;
    }

    // 5. Default: balanced 16×16 tile for shapes where neither
    //    dimension strongly dominates (0.25 < N/M < 4).
    return PREFILL_TILE_16x16;
}

/**
 * @brief Resolves tile variant: uses forced override if valid, else auto-selects.
 *
 * @param forced_variant  Environment-forced variant (-1 for auto).
 * @param M, N, K         Matrix dimensions.
 * @param grid_kpar       Whether grid-level K-parallelism is active.
 * @return Resolved PrefillInt8TileVariant enum value.
 */
static int resolve_prefill_int8_tile_variant(int forced_variant, int M, int N, int K, bool grid_kpar)
{
    if (forced_variant >= PREFILL_TILE_16x16 && forced_variant <= PREFILL_TILE_8x8)
    {
        return forced_variant;
    }
    return select_prefill_int8_tile_variant_auto(M, N, K, grid_kpar);
}

/**
 * @brief Baseline INT8 VNNI prefill GEMM kernel — one thread per output element.
 *
 * @tparam BLOCK_X  Threads per block in the N-dimension (e.g. 16, 32, 8).
 * @tparam BLOCK_Y  Threads per block in the M-dimension (e.g. 16, 8, 32).
 * @tparam CPT      Columns per thread — each thread computes CPT adjacent
 *                  N-columns for one M-row.  Typically 4, 2, or 1.
 *
 * @param d_A_int8      [M × K] row-major INT8 activations.
 * @param d_B_int8_vnni [K/4 × N × 4] VNNI-packed INT8 weights.
 * @param d_C_int32     [M × N] INT32 accumulator output (row-major).
 * @param M  Number of rows in A / C.
 * @param N  Number of columns in B / C.
 * @param K  Contraction dimension in INT8 elements (must be multiple of 4).
 *
 * **Purpose**:
 * Simple, non-LDS GEMM kernel where each thread computes CPT complete dot
 * products (one full K-reduction per output element).  No shared memory or
 * inter-thread cooperation — straightforward but memory-bandwidth limited.
 *
 * **Strategy**:
 *   - Grid: (ceil(N / (BLOCK_X*CPT)), ceil(M / BLOCK_Y)) workgroups.
 *   - Each thread walks the full K-dimension, loading one int32 (4 bytes)
 *     from A and CPT int32s from B per k-group, computing CPT v_dot4 ops.
 *   - Fast path: when all CPT columns are in-bounds, uses vectorized uint4
 *     or uint2 loads from B (coalesced across threads).
 *   - Slow path: when near the N-boundary, uses per-column scalar loads.
 *
 * **Structure**:
 *   1. Map thread to (m, n_base) output coordinates
 *   2. Initialize CPT accumulators to zero
 *   3. Compile-time dispatch on CPT for the K-loop:
 *      - CPT=4: vectorized uint4 B loads (fast), per-column scalar (boundary)
 *      - CPT=2: vectorized uint2 B loads (fast), per-column scalar (boundary)
 *      - CPT=1: single scalar B load per k-group
 *   4. Store: vectorized when all CPT columns in-bounds, scalar otherwise
 *
 * **When Used**: Fallback when wide-tile V3/V7 and grid-split-K don't apply.
 */
template <int BLOCK_X, int BLOCK_Y, int CPT>
__global__ void qgemm_int8_int8_vnni_prefill_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K)
{
    // Map thread to output coordinates.
    // n_base: first N-column this thread handles (threads cover CPT columns).
    // m: which M-row this thread handles (one row per thread in Y-dim).
    const int n_base = blockIdx.x * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = blockIdx.y * BLOCK_Y + threadIdx.y;

    if (m >= M || n_base >= N)
    {
        return;
    }

    // Number of k-groups (each group = 4 INT8 elements packed into one int32).
    const int k_groups = K / 4;

    // Check if all CPT output columns are within bounds.
    // This allows the fast path to use vectorized loads without per-element checks.
    const bool full_tile = (n_base + CPT - 1) < N;

    // B row stride in BYTES: B is laid out as [k_group × N × 4_bytes], so
    // stepping one k-group means advancing by N * 4 bytes.
    const int64_t b_row_stride_bytes = static_cast<int64_t>(N) * 4;

    // Pointers to this thread's A row and B column(s).
    // A: one row starting at d_A_int8[m * K]
    // B: VNNI-packed, so d_B_int8_vnni[n_base * 4] is the first k-group's
    //    data for columns [n_base..n_base+CPT-1].
    const int8_t* a_ptr0 = d_A_int8 + static_cast<int64_t>(m) * K;
    const int8_t* b_ptr0 = d_B_int8_vnni + static_cast<int64_t>(n_base) * 4;

    // CPT accumulators — one INT32 per output column.  These accumulate
    // the v_dot4 results across all k-groups.
    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    // =====================================================================
    // K-LOOP: Compile-time dispatch on CPT for specialised load patterns.
    //
    // CPT=4 (most common): loads 4 packed B values per k-group as uint4,
    //   giving 4 v_dot4 ops per iteration.  B's VNNI layout guarantees
    //   that consecutive columns' 4-byte packs are contiguous in memory,
    //   so the uint4 load reads columns [n_base..n_base+3] atomically.
    //
    // CPT=2: similar but uses uint2 for 2-column loads.
    //
    // CPT=1: single scalar B load per k-group.
    // =====================================================================
    if constexpr (CPT == 4)
    {
        if (full_tile)
        {
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b_ptr = b_ptr0;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                const uint4 b_pack4 = *reinterpret_cast<const uint4*>(b_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.y), acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.z), acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack4.w), acc[3], false);
                a_ptr += 4;
                b_ptr += b_row_stride_bytes;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;
            const int8_t* b2 = b_ptr0 + 8;
            const int8_t* b3 = b_ptr0 + 12;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (valid > 0)
                {
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b0), acc[0], false);
                }
                if (valid > 1)
                {
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b1), acc[1], false);
                }
                if (valid > 2)
                {
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b2), acc[2], false);
                }
                if (valid > 3)
                {
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b3), acc[3], false);
                }
                a_ptr += 4;
                b0 += b_row_stride_bytes;
                b1 += b_row_stride_bytes;
                b2 += b_row_stride_bytes;
                b3 += b_row_stride_bytes;
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if (full_tile)
        {
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b_ptr = b_ptr0;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                const uint2 b_pack2 = *reinterpret_cast<const uint2*>(b_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.x), acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, static_cast<int32_t>(b_pack2.y), acc[1], false);
                a_ptr += 4;
                b_ptr += b_row_stride_bytes;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t* a_ptr = a_ptr0;
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;
            for (int kg = 0; kg < k_groups; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (valid > 0)
                {
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b0), acc[0], false);
                }
                if (valid > 1)
                {
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *reinterpret_cast<const int32_t*>(b1), acc[1], false);
                }
                a_ptr += 4;
                b0 += b_row_stride_bytes;
                b1 += b_row_stride_bytes;
            }
        }
    }
    else
    {
        const int8_t* a_ptr = a_ptr0;
        const int8_t* b_ptr = b_ptr0;
        for (int kg = 0; kg < k_groups; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(b_ptr);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
            a_ptr += 4;
            b_ptr += b_row_stride_bytes;
        }
    }

    if constexpr (CPT == 4)
    {
        if ((n_base + 3) < N)
        {
            *reinterpret_cast<uint4*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint4(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]), static_cast<uint32_t>(acc[2]), static_cast<uint32_t>(acc[3]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if ((n_base + 1) < N)
        {
            *reinterpret_cast<uint2*>(d_C_int32 + static_cast<int64_t>(m) * N + n_base) =
                make_uint2(static_cast<uint32_t>(acc[0]), static_cast<uint32_t>(acc[1]));
        }
        else
        {
            #pragma unroll
            for (int c = 0; c < CPT; ++c)
            {
                const int n = n_base + c;
                if (n < N)
                {
                    d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
                }
            }
        }
    }
    else
    {
        d_C_int32[static_cast<int64_t>(m) * N + n_base] = acc[0];
    }
}

/**
 * @brief Grid-level K-parallel (split-K) INT8 VNNI prefill GEMM kernel.
 *
 * @tparam BLOCK_X              Threads per block in the N-dimension.
 * @tparam BLOCK_Y              Threads per block in the M-dimension.
 * @tparam CPT                  Columns per thread (4, 2, or 1).
 * @tparam STAGE_B              Use LDS staging for B tile loads (reduces
 *                              global memory traffic by reusing B across
 *                              BLOCK_Y M-rows).
 * @tparam ATOMIC_ACCUM         If true, use atomicAdd for cross-slice
 *                              accumulation (default).  If false, direct write
 *                              (only valid for single-slice or pre-zeroed output).
 * @tparam SOFTWARE_PIPELINED   Overlap next iteration's loads with current
 *                              iteration's compute (manual software pipelining).
 * @tparam LDS_B_TILE_CACHE     Use a KT\u00d7128 LDS tile to cache B data across
 *                              multiple k-groups (amortizes global load latency).
 *
 * @param d_A_int8           [M \u00d7 K] row-major INT8 activations.
 * @param d_B_int8_vnni      [K/4 \u00d7 N \u00d7 4] VNNI-packed INT8 weights.
 * @param d_C_int32          [M \u00d7 N] INT32 accumulator output (row-major).
 * @param M                  Number of rows in A / C.
 * @param N                  Number of columns in B / C.
 * @param K                  Contraction dimension in INT8 elements.
 * @param k_groups_per_slice Number of k-groups each Z-slice processes.
 * @param slice_index_base   Starting slice index (for multi-launch accumulation).
 * @param grid_swizzle_variant  Grid dimension swizzle (0: X=N/Y=M, 1: X=M/Y=N).
 *
 * **Purpose & Specialization**:
 * Splits the K-dimension across `blockIdx.z` slices so that multiple
 * workgroups cooperate on the same (M, N) output element.  Each slice
 * computes a partial sum over its k-group range, then accumulates into
 * the output via `atomicAdd`.  This increases parallelism for K-heavy
 * shapes where M\u00d7N alone doesn't saturate the GPU.
 *
 * **Strategy**:
 *   - Grid: (ceil(N/(BLOCK_X*CPT)), ceil(M/BLOCK_Y), num_k_slices).
 *   - Each Z-slice processes k-groups [k_begin..k_end) where:
 *     - k_begin = (split_idx + slice_index_base) * k_groups_per_slice
 *     - k_end = min(total_k_groups, k_begin + k_groups_per_slice)
 *   - Partial sums are accumulated via atomicAdd (INT32 atomics).
 *   - Grid swizzle variant swaps X/Y block indices for cache locality.
 *
 * **Compute Variants** (compile-time selection via template bools):
 *   1. **LDS_B_TILE_CACHE**: Cooperative B loading into KT\u00d7128 LDS tile,
 *      with k-group tiling.  Best for shapes with high B reuse.
 *   2. **STAGE_B**: One M-row loads B into LDS, all BLOCK_Y M-rows reuse.
 *      Reduces redundant global B loads by BLOCK_Y\u00d7.
 *   3. **SOFTWARE_PIPELINED**: Manual load/compute overlap using
 *      staging registers (load iteration i+1 while computing iteration i).
 *   4. **Default**: Simple direct-load path (same structure as baseline).
 *
 * **Structure**:
 *   1. Map thread to (m, n_base) + determine K-slice range
 *   2. Initialize CPT accumulators to zero
 *   3. K-loop (k_begin..k_end) with variant-specific compute
 *   4. atomicAdd partial sums to output
 */
template <int BLOCK_X, int BLOCK_Y, int CPT, bool STAGE_B = false, bool ATOMIC_ACCUM = true, bool SOFTWARE_PIPELINED = false, bool LDS_B_TILE_CACHE = false>
__global__ void qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    int32_t* __restrict__ d_C_int32,
    int M,
    int N,
    int K,
    int k_groups_per_slice,
    int slice_index_base,
    int grid_swizzle_variant)
{
    // Grid dimension mapping: swizzle variant 0 uses X=N, Y=M (standard);
    // variant 1 swaps to X=M, Y=N for improved L2 cache hit rate when
    // N-tiles are much larger than M-tiles.
    const int n_tile_idx = (grid_swizzle_variant != 0) ? static_cast<int>(blockIdx.y) : static_cast<int>(blockIdx.x);
    const int m_tile_idx = (grid_swizzle_variant != 0) ? static_cast<int>(blockIdx.x) : static_cast<int>(blockIdx.y);
    const int n_base = n_tile_idx * (BLOCK_X * CPT) + threadIdx.x * CPT;
    const int m = m_tile_idx * BLOCK_Y + threadIdx.y;
    const int split_idx = blockIdx.z; // Which K-slice this workgroup handles

    if (m >= M || n_base >= N)
    {
        return;
    }

    // Compute this slice's K-range in k-groups (each k-group = 4 INT8 elements).
    // slice_index_base allows multi-launch accumulation: first launch handles
    // slices 0..S-1, second launch handles slices S..2S-1, etc.
    const int total_k_groups = K / 4;
    const int k_begin = (split_idx + slice_index_base) * k_groups_per_slice;
    if (k_begin >= total_k_groups)
    {
        return; // This slice has no work (over-provisioned Z-dim)
    }

    const int k_end = min(total_k_groups, k_begin + k_groups_per_slice);
    const bool full_tile = (n_base + CPT - 1) < N;
    const int64_t b_row_stride_bytes = static_cast<int64_t>(N) * 4;

    // Offset A and B pointers to this slice's starting k-group.
    // A: skip k_begin k-groups (each 4 bytes) from the row start.
    // B (VNNI): skip k_begin complete k-rows of N columns.
    const int8_t* a_ptr0 = d_A_int8 + static_cast<int64_t>(m) * K + static_cast<int64_t>(k_begin) * 4;
    const int8_t* b_ptr0 = d_B_int8_vnni + (static_cast<int64_t>(k_begin) * N + n_base) * 4;

    int32_t acc[CPT];
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        acc[c] = 0;
    }

    // =====================================================================
    // VARIANT: LDS B Tile Cache (LDS_B_TILE_CACHE = true)
    //
    // Cooperatively loads a KT×128 tile of B into LDS, then all BLOCK_Y
    // M-rows compute against the same cached B data.  This amortizes
    // the global load cost across M-rows and across KT k-groups.
    //
    // LDS layout: b_lds[kg_local * 128 + n_local] where kg_local=0..KT-1
    // and n_local=0..127.  All 256 threads participate in loading
    // (4 passes of 256 threads to cover KT×128 = 1024 entries).
    // =====================================================================
    if constexpr (CPT == 4 && LDS_B_TILE_CACHE)
    {
        constexpr int KT = 8;
        __shared__ int32_t b_lds[KT * 128];

        const int tid = threadIdx.y * BLOCK_X + threadIdx.x;
        const int n_block_base = n_tile_idx * (BLOCK_X * CPT);
        const int n_local = threadIdx.x * CPT;

        const int32_t* a_row = reinterpret_cast<const int32_t*>(
            d_A_int8 + static_cast<int64_t>(m) * K);
        const int32_t* b_global_base = reinterpret_cast<const int32_t*>(
            d_B_int8_vnni + static_cast<int64_t>(n_block_base) * 4);

        for (int kt = k_begin; kt < k_end; kt += KT)
        {
            const int kg_limit = min(KT, k_end - kt);

            #pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                const int flat = i * 256 + tid;
                const int kg_l = flat >> 7;
                const int n_l = flat & 127;
                int32_t b_val = 0;
                if (kg_l < kg_limit)
                {
                    b_val = b_global_base[static_cast<int64_t>(kt + kg_l) * N + n_l];
                }
                b_lds[kg_l * 128 + n_l] = b_val;
            }
            __syncthreads();

            if (kg_limit > 0)
            {
                int32_t a_packed = a_row[kt];
                #pragma unroll
                for (int kg_l = 0; kg_l < KT - 1; ++kg_l)
                {
                    if (kg_l >= kg_limit - 1)
                    {
                        break;
                    }

                    const int32_t a_next = a_row[kt + kg_l + 1];
                    const int off = kg_l * 128 + n_local;
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 0], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 1], acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 2], acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, b_lds[off + 3], acc[3], false);
                    a_packed = a_next;
                }

                const int off_last = (kg_limit - 1) * 128 + n_local;
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 0], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 1], acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 2], acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, b_lds[off_last + 3], acc[3], false);
            }

            __syncthreads();
        }
    }
    // =====================================================================
    // VARIANT: STAGE_B (LDS staging for B reuse across M-rows)
    //
    // Only threadIdx.y==0 loads B into s_b_local[CPT][BLOCK_X], then
    // all BLOCK_Y M-rows in the block read from LDS.  Saves BLOCK_Y×
    // redundant global B loads at the cost of two __syncthreads per
    // k-group (one after load, one after compute).
    // =====================================================================
    else if constexpr (STAGE_B)
    {
        __shared__ int32_t s_b_local[CPT][BLOCK_X];
        const int valid = max(0, N - n_base);
        const int lane_x = threadIdx.x;
        const int8_t* a_ptr = a_ptr0;

        if constexpr (CPT == 4)
        {
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;
            const int8_t* b2 = b_ptr0 + 8;
            const int8_t* b3 = b_ptr0 + 12;

            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (threadIdx.y == 0)
                {
                    s_b_local[0][lane_x] = (valid > 0) ? *reinterpret_cast<const int32_t*>(b0) : 0;
                    s_b_local[1][lane_x] = (valid > 1) ? *reinterpret_cast<const int32_t*>(b1) : 0;
                    s_b_local[2][lane_x] = (valid > 2) ? *reinterpret_cast<const int32_t*>(b2) : 0;
                    s_b_local[3][lane_x] = (valid > 3) ? *reinterpret_cast<const int32_t*>(b3) : 0;
                    b0 += b_row_stride_bytes;
                    b1 += b_row_stride_bytes;
                    b2 += b_row_stride_bytes;
                    b3 += b_row_stride_bytes;
                }
                __syncthreads();

                acc[0] = __builtin_amdgcn_sdot4(a_packed, s_b_local[0][lane_x], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, s_b_local[1][lane_x], acc[1], false);
                acc[2] = __builtin_amdgcn_sdot4(a_packed, s_b_local[2][lane_x], acc[2], false);
                acc[3] = __builtin_amdgcn_sdot4(a_packed, s_b_local[3][lane_x], acc[3], false);

                __syncthreads();
                a_ptr += 4;
            }
        }
        else if constexpr (CPT == 2)
        {
            const int8_t* b0 = b_ptr0;
            const int8_t* b1 = b_ptr0 + 4;

            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (threadIdx.y == 0)
                {
                    s_b_local[0][lane_x] = (valid > 0) ? *reinterpret_cast<const int32_t*>(b0) : 0;
                    s_b_local[1][lane_x] = (valid > 1) ? *reinterpret_cast<const int32_t*>(b1) : 0;
                    b0 += b_row_stride_bytes;
                    b1 += b_row_stride_bytes;
                }
                __syncthreads();

                acc[0] = __builtin_amdgcn_sdot4(a_packed, s_b_local[0][lane_x], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, s_b_local[1][lane_x], acc[1], false);

                __syncthreads();
                a_ptr += 4;
            }
        }
        else
        {
            const int8_t* b0 = b_ptr0;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (threadIdx.y == 0)
                {
                    s_b_local[0][lane_x] = (valid > 0) ? *reinterpret_cast<const int32_t*>(b0) : 0;
                    b0 += b_row_stride_bytes;
                }
                __syncthreads();

                acc[0] = __builtin_amdgcn_sdot4(a_packed, s_b_local[0][lane_x], acc[0], false);

                __syncthreads();
                a_ptr += 4;
            }
        }
    }
    else if constexpr (CPT == 4)
    {
        if constexpr (SOFTWARE_PIPELINED)
        {
            if (full_tile)
            {
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b_ptr = reinterpret_cast<const int32_t*>(b_ptr0);
                const int b_row_stride_i32 = N;
                const int k_len = k_end - k_begin;

                if (k_len > 0)
                {
                    int32_t a_cur = *reinterpret_cast<const int32_t*>(a_ptr);
                    int32_t b0 = b_ptr[0];
                    int32_t b1 = b_ptr[1];
                    int32_t b2 = b_ptr[2];
                    int32_t b3 = b_ptr[3];
                    a_ptr += 4;
                    b_ptr += b_row_stride_i32;

                    for (int i = 1; i < k_len; ++i)
                    {
                        const int32_t a_nxt = *reinterpret_cast<const int32_t*>(a_ptr);
                        const int32_t bn0 = b_ptr[0];
                        const int32_t bn1 = b_ptr[1];
                        const int32_t bn2 = b_ptr[2];
                        const int32_t bn3 = b_ptr[3];

                        acc[0] = __builtin_amdgcn_sdot4(a_cur, b0, acc[0], false);
                        acc[1] = __builtin_amdgcn_sdot4(a_cur, b1, acc[1], false);
                        acc[2] = __builtin_amdgcn_sdot4(a_cur, b2, acc[2], false);
                        acc[3] = __builtin_amdgcn_sdot4(a_cur, b3, acc[3], false);

                        a_cur = a_nxt;
                        b0 = bn0;
                        b1 = bn1;
                        b2 = bn2;
                        b3 = bn3;
                        a_ptr += 4;
                        b_ptr += b_row_stride_i32;
                    }

                    acc[0] = __builtin_amdgcn_sdot4(a_cur, b0, acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_cur, b1, acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_cur, b2, acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_cur, b3, acc[3], false);
                }
            }
            else
            {
                const int valid = max(0, N - n_base);
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b0 = reinterpret_cast<const int32_t*>(b_ptr0);
                const int32_t* b1 = b0 + 1;
                const int32_t* b2 = b0 + 2;
                const int32_t* b3 = b0 + 3;
                const int b_row_stride_i32 = N;
                for (int kg = k_begin; kg < k_end; ++kg)
                {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                    if (valid > 0)
                    {
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                    }
                    if (valid > 1)
                    {
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                    }
                    if (valid > 2)
                    {
                        acc[2] = __builtin_amdgcn_sdot4(a_packed, *b2, acc[2], false);
                    }
                    if (valid > 3)
                    {
                        acc[3] = __builtin_amdgcn_sdot4(a_packed, *b3, acc[3], false);
                    }
                    a_ptr += 4;
                    b0 += b_row_stride_i32;
                    b1 += b_row_stride_i32;
                    b2 += b_row_stride_i32;
                    b3 += b_row_stride_i32;
                }
            }
        }
        else
        {
            if (full_tile)
            {
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b_ptr = reinterpret_cast<const int32_t*>(b_ptr0);
                const int b_row_stride_i32 = N;
                for (int kg = k_begin; kg < k_end; ++kg)
                {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, b_ptr[0], acc[0], false);
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, b_ptr[1], acc[1], false);
                    acc[2] = __builtin_amdgcn_sdot4(a_packed, b_ptr[2], acc[2], false);
                    acc[3] = __builtin_amdgcn_sdot4(a_packed, b_ptr[3], acc[3], false);
                    a_ptr += 4;
                    b_ptr += b_row_stride_i32;
                }
            }
            else
            {
                const int valid = max(0, N - n_base);
                const int8_t* a_ptr = a_ptr0;
                const int32_t* b0 = reinterpret_cast<const int32_t*>(b_ptr0);
                const int32_t* b1 = b0 + 1;
                const int32_t* b2 = b0 + 2;
                const int32_t* b3 = b0 + 3;
                const int b_row_stride_i32 = N;
                for (int kg = k_begin; kg < k_end; ++kg)
                {
                    const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                    if (valid > 0)
                    {
                        acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                    }
                    if (valid > 1)
                    {
                        acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                    }
                    if (valid > 2)
                    {
                        acc[2] = __builtin_amdgcn_sdot4(a_packed, *b2, acc[2], false);
                    }
                    if (valid > 3)
                    {
                        acc[3] = __builtin_amdgcn_sdot4(a_packed, *b3, acc[3], false);
                    }
                    a_ptr += 4;
                    b0 += b_row_stride_i32;
                    b1 += b_row_stride_i32;
                    b2 += b_row_stride_i32;
                    b3 += b_row_stride_i32;
                }
            }
        }
    }
    else if constexpr (CPT == 2)
    {
        if (full_tile)
        {
            const int8_t* a_ptr = a_ptr0;
            const int32_t* b_ptr = reinterpret_cast<const int32_t*>(b_ptr0);
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                acc[0] = __builtin_amdgcn_sdot4(a_packed, b_ptr[0], acc[0], false);
                acc[1] = __builtin_amdgcn_sdot4(a_packed, b_ptr[1], acc[1], false);
                a_ptr += 4;
                b_ptr += b_row_stride_i32;
            }
        }
        else
        {
            const int valid = max(0, N - n_base);
            const int8_t* a_ptr = a_ptr0;
            const int32_t* b0 = reinterpret_cast<const int32_t*>(b_ptr0);
            const int32_t* b1 = b0 + 1;
            const int b_row_stride_i32 = N;
            for (int kg = k_begin; kg < k_end; ++kg)
            {
                const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
                if (valid > 0)
                {
                    acc[0] = __builtin_amdgcn_sdot4(a_packed, *b0, acc[0], false);
                }
                if (valid > 1)
                {
                    acc[1] = __builtin_amdgcn_sdot4(a_packed, *b1, acc[1], false);
                }
                a_ptr += 4;
                b0 += b_row_stride_i32;
                b1 += b_row_stride_i32;
            }
        }
    }
    else
    {
        const int8_t* a_ptr = a_ptr0;
        const int8_t* b_ptr = b_ptr0;
        for (int kg = k_begin; kg < k_end; ++kg)
        {
            const int32_t a_packed = *reinterpret_cast<const int32_t*>(a_ptr);
            const int32_t b_packed = *reinterpret_cast<const int32_t*>(b_ptr);
            acc[0] = __builtin_amdgcn_sdot4(a_packed, b_packed, acc[0], false);
            a_ptr += 4;
            b_ptr += b_row_stride_bytes;
        }
    }

    // =====================================================================
    // STORE: atomicAdd partial sums from this K-slice into the output.
    //
    // Multiple K-slices write to the same output element, so atomic
    // accumulation is needed.  The caller must zero d_C_int32 before
    // the first launch when using ATOMIC_ACCUM.
    // =====================================================================
    #pragma unroll
    for (int c = 0; c < CPT; ++c)
    {
        const int n = n_base + c;
        if (n < N)
        {
            if constexpr (ATOMIC_ACCUM)
            {
                atomicAdd(&d_C_int32[static_cast<int64_t>(m) * N + n], acc[c]);
            }
            else
            {
                d_C_int32[static_cast<int64_t>(m) * N + n] = acc[c];
            }
        }
    }
}

// ============================================================================
// DISPATCH FUNCTIONS
// ============================================================================

extern "C"
{
    bool rocmQuantGemm_setDevice(int rocm_device_id);

    /**
     * @brief Dispatch function for wide-tile V3 kernel (N64 LDS double-buffered).
     *
     * Launches `qgemm_int8_vnni_wide_tile_v3_kernel<64, KT>` with KT=8 or KT=16.
     * Grid is 2D: (ceil(N/64), ceil(M/128)) — one workgroup per (M-tile, N-tile).
     * Each workgroup covers up to 128 M-rows × 64 N-cols.
     *
     * @param d_A_int8      [M × K] INT8 activations (device memory).
     * @param d_B_int8_vnni [K/4 × N × 4] VNNI-packed INT8 weights (device memory).
     * @param d_C_int32     [M × N] INT32 output (device memory).
     * @param M, N, K       Matrix dimensions.
     * @param kt_select     K-tile depth: 8 → KT=8 (12 KB LDS, 3 waves/SIMD),
     *                      16 → KT=16 (24 KB LDS, fewer K-iterations).
     * @param rocm_device_id  HIP device ID.
     * @param stream        HIP stream (cast from hipStream_t).
     * @return true on success, false on validation failure or kernel error.
     */
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v3(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v3] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v3] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        // V3 uses N_TILE=64.  Grid is 2D: (ceil(N/N_TILE), ceil(M/M_TILE)).
        //
        // M_TILE dispatch strategy:
        //   M <= 128: waste-minimizing M_TILE (picks the tile that minimizes
        //     ceil(M/mt)*mt - M, preferring larger mt as tiebreaker).
        //   M > 128:  always M_TILE=128 (best B-data reuse; boundary checks
        //     in the store section handle the partial last tile cheaply).
        //
        // A main+tail split (two launches) was tested but the extra kernel
        // launch overhead (~5-10 tok/s regression at M=596) outweighs the
        // waste savings from smaller tail tiles.
        constexpr int N_TILE = 64;
        constexpr int BLOCK_SIZE = 256;
        const auto s = static_cast<hipStream_t>(stream);

        // Select M_TILE that minimizes wasted rows (ceil-up waste).
        // Prefers larger M_TILE as tiebreaker (fewer tiles, better B-reuse).
        auto select_m_tile = [](int rows) -> int {
            int best_mt = 128;
            int best_waste = ((rows + 127) / 128) * 128 - rows;
            for (int mt : {64, 32, 16}) {
                int waste = ((rows + mt - 1) / mt) * mt - rows;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_mt = mt;
                }
            }
            return best_mt;
        };

        const int m_tile = (M <= 128) ? select_m_tile(M) : 128;
        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const unsigned m_tiles = static_cast<unsigned>((M + m_tile - 1) / m_tile);
        const dim3 grid(n_tiles, m_tiles, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

#define LAUNCH_V3(MT, KT_VAL) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v3_kernel<N_TILE, KT_VAL, MT>), \
                           grid, block, 0, s, d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K)

        if (kt_select == 16) {
            switch (m_tile) {
                case 16:  LAUNCH_V3(16, 16);  break;
                case 32:  LAUNCH_V3(32, 16);  break;
                case 64:  LAUNCH_V3(64, 16);  break;
                default:  LAUNCH_V3(128, 16); break;
            }
        } else if (kt_select == 4) {
            switch (m_tile) {
                case 16:  LAUNCH_V3(16, 4);  break;
                case 32:  LAUNCH_V3(32, 4);  break;
                case 64:  LAUNCH_V3(64, 4);  break;
                default:  LAUNCH_V3(128, 4); break;
            }
        } else {
            switch (m_tile) {
                case 16:  LAUNCH_V3(16, 8);  break;
                case 32:  LAUNCH_V3(32, 8);  break;
                case 64:  LAUNCH_V3(64, 8);  break;
                default:  LAUNCH_V3(128, 8); break;
            }
        }
#undef LAUNCH_V3

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v3] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    /**
     * @brief Dispatch function for wide-tile V7 kernel (N128 safe-tile split).
     *
     * Launches `qgemm_int8_vnni_wide_tile_v7_kernel<128, KT>` with KT=8 or KT=16.
     * Grid is 2D: (ceil(N/128), ceil(M/128)) — one workgroup per (M-tile, N-tile).
     * V7 splits the K-loop into safe (branchless) and boundary (checked) loops.
     *
     * @param d_A_int8      [M × K] INT8 activations (device memory).
     * @param d_B_int8_vnni [K/4 × N × 4] VNNI-packed INT8 weights (device memory).
     * @param d_C_int32     [M × N] INT32 output (device memory).
     * @param M, N, K       Matrix dimensions.
     * @param kt_select     K-tile depth: 8 → KT=8 (16 KB LDS, 4 workgroups/CU),
     *                      16 → KT=16 (32 KB LDS, 2 workgroups/CU).
     * @param rocm_device_id  HIP device ID.
     * @param stream        HIP stream (cast from hipStream_t).
     * @return true on success, false on validation failure or kernel error.
     */
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_wide_tile_v7(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int kt_select,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v7] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v7] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        // V7 uses N_TILE=128.  Same dispatch strategy as V3.
        constexpr int N_TILE = 128;
        constexpr int BLOCK_SIZE = 256;
        const auto s = static_cast<hipStream_t>(stream);

        auto select_m_tile = [](int rows) -> int {
            int best_mt = 128;
            int best_waste = ((rows + 127) / 128) * 128 - rows;
            for (int mt : {64, 32, 16}) {
                int waste = ((rows + mt - 1) / mt) * mt - rows;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_mt = mt;
                }
            }
            return best_mt;
        };

        const int m_tile = (M <= 128) ? select_m_tile(M) : 128;
        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const unsigned m_tiles = static_cast<unsigned>((M + m_tile - 1) / m_tile);
        const dim3 grid(n_tiles, m_tiles, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

#define LAUNCH_V7(MT, KT_VAL) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v7_kernel<N_TILE, KT_VAL, MT>), \
                           grid, block, 0, s, d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K)

        if (kt_select == 16) {
            switch (m_tile) {
                case 16:  LAUNCH_V7(16, 16);  break;
                case 32:  LAUNCH_V7(32, 16);  break;
                case 64:  LAUNCH_V7(64, 16);  break;
                default:  LAUNCH_V7(128, 16); break;
            }
        } else if (kt_select == 4) {
            switch (m_tile) {
                case 16:  LAUNCH_V7(16, 4);  break;
                case 32:  LAUNCH_V7(32, 4);  break;
                case 64:  LAUNCH_V7(64, 4);  break;
                default:  LAUNCH_V7(128, 4); break;
            }
        } else {
            switch (m_tile) {
                case 16:  LAUNCH_V7(16, 8);  break;
                case 32:  LAUNCH_V7(32, 8);  break;
                case 64:  LAUNCH_V7(64, 8);  break;
                default:  LAUNCH_V7(128, 8); break;
            }
        }
#undef LAUNCH_V7

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_vnni_prefill_wide_tile_v7] Kernel launch failed: %s (M=%d N=%d K=%d kt=%d)\n",
                    hipGetErrorString(err), M, N, K, kt_select);
            return false;
        }

        return true;
    }

    /**
     * @brief Dispatch function for baseline prefill kernel (standard tile variants).
     *
     * Launches `qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, CPT>` with tile
     * geometry selected by `tile_variant` (auto or forced) and CPT (1, 2, or 4).
     * Used as fallback when wide-tile V3/V7 are not applicable.
     *
     * @param d_A_int8      [M × K] INT8 activations (device memory).
     * @param d_B_int8_vnni [K/4 × N × 4] VNNI-packed INT8 weights (device memory).
     * @param d_C_int32     [M × N] INT32 output (device memory).
     * @param M, N, K       Matrix dimensions.
     * @param tile_variant   Tile geometry override (-1 for auto via aspect ratio).
     *                       See @ref PrefillInt8TileVariant.
     * @param cpt            Columns per thread (1, 2, or 4).
     * @param rocm_device_id  HIP device ID.
     * @param stream        HIP stream (cast from hipStream_t).
     * @return true on success, false on validation failure or kernel error.
     */
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int tile_variant,
        int cpt,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        // Resolve tile variant (auto-select based on aspect ratio or forced override)
        // and validate CPT.  cpt_v ensures we only instantiate templates for 1, 2, or 4.
        const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, false);
        const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;

        // Each variant block configures (BX, BY) and launches the kernel.
        // Grid: (ceil(N / (BX*CPT)), ceil(M / BY), 1).
        if (variant == PREFILL_TILE_32x8)
        {
            constexpr int BX = 32;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }
        else if (variant == PREFILL_TILE_8x32)
        {
            constexpr int BX = 8;
            constexpr int BY = 32;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }
        else if (variant == PREFILL_TILE_8x8)
        {
            constexpr int BX = 8;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }
        else
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 4>),
                                   dim3(static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else if (cpt_v == 2)
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 2>),
                                   dim3(static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2)), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
            else
            {
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_kernel_t<BX, BY, 1>),
                                   dim3(static_cast<unsigned>((N + BX - 1) / BX), static_cast<unsigned>((M + BY - 1) / BY), 1),
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K);
            }
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                    hipGetErrorString(err), M, N, K);
            return false;
        }

        return true;
    }

    /**
     * @brief Dispatch function for grid-level K-parallel (split-K) prefill.
     *
     * Launches `qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t` with the
     * K-dimension split across `split_k_slices` Z-slices.  The output is
     * zeroed via `hipMemsetAsync` before launch since slices use `atomicAdd`.
     *
     * @param d_A_int8              [M × K] INT8 activations (device memory).
     * @param d_B_int8_vnni         [K/4 × N × 4] VNNI-packed INT8 weights.
     * @param d_C_int32             [M × N] INT32 output (zeroed, then accumulated).
     * @param M, N, K               Matrix dimensions.
     * @param split_k_slices        Number of K-slices (grid.z dimension).
     * @param tile_variant          Tile geometry override (-1 for auto).
     * @param cpt                   Columns per thread (1, 2, or 4).
     * @param kernel_body_variant   Compute variant: 0=default, 1=software-pipelined,
     *                              2=LDS B tile cache.
     * @param grid_swizzle_variant  Grid dim swap: 0=X:N/Y:M, 1=X:M/Y:N.
     * @param rocm_device_id        HIP device ID.
     * @param stream                HIP stream (cast from hipStream_t).
     * @return true on success, false on validation failure or kernel error.
     */
    bool rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        int32_t* d_C_int32,
        int M, int N, int K,
        int split_k_slices,
        int tile_variant,
        int cpt,
        int kernel_body_variant,
        int grid_swizzle_variant,
        int rocm_device_id, void* stream)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_int32 || M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 4) != 0)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] K must be multiple of 4 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        // Compute K-slice parameters: clamp slices to total k-groups,
        // then compute equal-sized slices (last slice may be smaller).
        const int total_k_groups = K / 4;
        const int slices = max(1, min(split_k_slices, total_k_groups));
        const int k_groups_per_slice = (total_k_groups + slices - 1) / slices;

        if (!rocmQuantGemm_setDevice(rocm_device_id))
        {
            return false;
        }

        // Resolve tile variant and kernel body options.
        // kernel_body_variant: 0=default, 1=software-pipelined, 2=LDS B tile cache.
        // grid_swizzle_variant: swaps X/Y grid dims for L2 cache locality.
        const int variant = resolve_prefill_int8_tile_variant(tile_variant, M, N, K, true);
        const int cpt_v = (cpt == 2 || cpt == 4) ? cpt : 1;
        const bool use_software_pipelined = (kernel_body_variant == 1 || kernel_body_variant == 2);
        const bool request_lds_b_tile = (kernel_body_variant == 2);
        const bool use_grid_swizzled = (grid_swizzle_variant == 1);

        // Zero the output buffer: atomicAdd accumulates partial sums from
        // multiple K-slices, so the buffer must start at zero.
        {
            const size_t c_bytes = static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t);
            hipError_t memset_err = hipMemsetAsync(d_C_int32, 0, c_bytes, static_cast<hipStream_t>(stream));
            if (memset_err != hipSuccess)
            {
                fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] hipMemsetAsync failed: %s\n", hipGetErrorString(memset_err));
                return false;
            }
        }

        if (variant == PREFILL_TILE_32x8)
        {
            constexpr int BX = 32;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }
        else if (variant == PREFILL_TILE_8x32)
        {
            constexpr int BX = 8;
            constexpr int BY = 32;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }
        else if (variant == PREFILL_TILE_8x8)
        {
            constexpr int BX = 8;
            constexpr int BY = 8;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }
        else
        {
            constexpr int BX = 16;
            constexpr int BY = 16;
            const dim3 block(BX, BY, 1);
            if (cpt_v == 4)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 4 - 1) / (BX * 4));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                const bool can_use_lds_b_tile = request_lds_b_tile && ((M % BY) == 0) && ((N % (BX * 4)) == 0);
                if (can_use_lds_b_tile)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, true>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else if (use_software_pipelined)
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, true, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
                else
                {
                    hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 4, false, true, false, false>),
                                       grid,
                                       block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
                }
            }
            else if (cpt_v == 2)
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX * 2 - 1) / (BX * 2));
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 2, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
            else
            {
                const unsigned n_tiles = static_cast<unsigned>((N + BX - 1) / BX);
                const unsigned m_tiles = static_cast<unsigned>((M + BY - 1) / BY);
                const dim3 grid(use_grid_swizzled ? m_tiles : n_tiles, use_grid_swizzled ? n_tiles : m_tiles, static_cast<unsigned>(slices));
                hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_int8_vnni_prefill_grid_kpar_kernel_t<BX, BY, 1, false, true, false, false>),
                                   grid,
                                   block, 0, static_cast<hipStream_t>(stream), d_A_int8, d_B_int8_vnni, d_C_int32, M, N, K, k_groups_per_slice, 0, use_grid_swizzled ? 1 : 0);
            }
        }

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[rocmQuantGemm_int8_int8_int32_vnni_prefill_grid_kpar] Kernel launch failed: %s (M=%d N=%d K=%d slices=%d)\n",
                    hipGetErrorString(err), M, N, K, slices);
            return false;
        }

        return true;
    }

} // extern "C" (close before blockwise kernel templates - they need C++ linkage)

// ============================================================================
// BLOCKWISE V3: LDS Double-Buffered Pipeline with Per-Block Activation Scales
// ============================================================================
//
// Variant of V3 GEMM kernel that applies per-K-block activation scales during
// accumulation instead of a single per-row scale in the epilogue.
//
// Key differences from V3:
// - Forces KT=8 (8 k-groups = 32 elements = 1 activation block)
// - Uses FP32 running accumulators (fp32_acc) alongside INT32 per-tile accs
// - After each K-tile compute: cast INT32→FP32, multiply by blockwise scale,
//   accumulate to fp32_acc, reset INT32 acc
// - Outputs FP32 (with activation scales applied)
// - Epilogue only needs to apply weight scale_B[n]
//
// Register budget (ISA-verified gfx906):
//   M_TILE=16: 101 VGPRs → 2 waves/SIMD  M_TILE=32: 115 VGPRs → 2 waves/SIMD
//   M_TILE=64: 147 VGPRs → 1 wave (avoid)  M_TILE=128: 213 VGPRs → 1 wave (avoid)
// Launcher caps at M_TILE=32 for 2 waves/SIMD with 0 spills.
//
template <int N_TILE, int M_TILE = 128, int UNROLL_KK = 1>
__global__ __launch_bounds__(256)
void qgemm_int8_vnni_wide_tile_v3_blockwise_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    int M, int N, int K)
{
    constexpr int KT = 8; // forced: 1 block per tile
    constexpr int BLOCK_SIZE = 256;
    constexpr int M_PER_THREAD = 4;
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;

    static_assert(M_TILE == 16 || M_TILE == 32 || M_TILE == 64 || M_TILE == 128);
    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);

    const int n_block = blockIdx.x * N_TILE;
    const int m_block = blockIdx.y * M_TILE;
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    // INT32 per-tile accumulators (reset after each tile)
    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    // FP32 running accumulators (accumulate scaled partial sums across tiles)
    float fp32_acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            fp32_acc[mm][nn] = 0.0f;

    const int total_k_groups = K / 4;
    const int blocks_per_row = total_k_groups / KT; // K/32 blocks per row
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    // A cooperative load geometry
    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    // B cooperative load geometry
    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_ACTIVE_THREADS = KT * B_N_CHUNKS;
    static_assert(B_ACTIVE_THREADS <= BLOCK_SIZE);

    const int b_kk_local = threadIdx.x / B_N_CHUNKS;
    const int b_nl = (threadIdx.x % B_N_CHUNKS) * 4;
    const int b_gn = n_block + b_nl;
    const int b_lds_off = b_kk_local * N_TILE + b_nl;

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging;

    // Blockwise scale+accumulate+reset helper (called after each tile compute)
    auto blockwise_accumulate = [&](int block_idx) __attribute__((always_inline))
    {
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            const int gm = m_block + m_base + mm;
            const float s = (gm < M)
                ? d_scales_A_blockwise[static_cast<int64_t>(gm) * blocks_per_row + block_idx]
                : 0.0f;
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                fp32_acc[mm][nn] += static_cast<float>(acc[mm][nn]) * s;
                acc[mm][nn] = 0;
            }
        }
    };

    // PRELOAD tile 0 → buf[0]
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
    {
        const int mi = a_mi[p];
        if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
        const int gm = m_block + mi;
        const int gkg = a_kg;
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

    b_staging = make_uint4(0, 0, 0, 0);
    if (threadIdx.x < B_ACTIVE_THREADS)
    {
        const int gkk = b_kk_local;
        if (__builtin_expect(b_gn + 3 < N && gkk < total_k_groups, 1))
        {
            b_staging = *reinterpret_cast<const uint4*>(
                b_global + static_cast<int64_t>(gkk) * N + b_gn);
        }
        else
        {
            b_staging.x = (b_gn + 0 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 0] : 0;
            b_staging.y = (b_gn + 1 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 1] : 0;
            b_staging.z = (b_gn + 2 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 2] : 0;
            b_staging.w = (b_gn + 3 < N && gkk < total_k_groups)
                ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 3] : 0;
        }
    }

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
    if (threadIdx.x < B_ACTIVE_THREADS)
    {
        *reinterpret_cast<uint4*>(&b_lds[0][b_lds_off]) = b_staging;
    }

    __syncthreads();

    // MAIN LOOP: double-buffered pipeline
    for (int tile = 1; tile < num_tiles; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int kt_base = tile * KT;

        // 1. Issue global loads for next tile
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

        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            const int gkk = kt_base + b_kk_local;
            if (__builtin_expect(b_gn + 3 < N && gkk < total_k_groups, 1))
            {
                b_staging = *reinterpret_cast<const uint4*>(
                    b_global + static_cast<int64_t>(gkk) * N + b_gn);
            }
            else
            {
                b_staging = make_uint4(0, 0, 0, 0);
                b_staging.x = (b_gn + 0 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 0] : 0;
                b_staging.y = (b_gn + 1 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 1] : 0;
                b_staging.z = (b_gn + 2 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 2] : 0;
                b_staging.w = (b_gn + 3 < N && gkk < total_k_groups)
                    ? b_global[static_cast<int64_t>(gkk) * N + b_gn + 3] : 0;
            }
        }

        // 2. Compute on current buffer
        // UNROLL_KK: 0=disable, 1=full, 2=unroll_count(2), 4=unroll_count(4)

#define V3_BW_KK_BODY(BUF) \
                int32_t a_reg[M_PER_THREAD]; int32_t b_reg[N_PER_THREAD]; \
                _Pragma("unroll") \
                for (int mm = 0; mm < M_PER_THREAD; ++mm) a_reg[mm] = a_lds[BUF][kk * M_TILE + m_base + mm]; \
                _Pragma("unroll") \
                for (int nn = 0; nn < N_PER_THREAD; ++nn) b_reg[nn] = b_lds[BUF][kk * N_TILE + t_n * N_PER_THREAD + nn]; \
                _Pragma("unroll") \
                for (int mm = 0; mm < M_PER_THREAD; ++mm) \
                    _Pragma("unroll") \
                    for (int nn = 0; nn < N_PER_THREAD; ++nn) \
                        acc[mm][nn] = __builtin_amdgcn_sdot4(a_reg[mm], b_reg[nn], acc[mm][nn], false);

        if constexpr (UNROLL_KK == 0) {
            #pragma clang loop unroll(disable)
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(cur_buf) }
        } else if constexpr (UNROLL_KK == 2) {
            _Pragma("clang loop unroll_count(2)")
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(cur_buf) }
        } else if constexpr (UNROLL_KK == 4) {
            _Pragma("clang loop unroll_count(4)")
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(cur_buf) }
        } else { // UNROLL_KK == 1: full unroll
            #pragma unroll
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(cur_buf) }
        }

        // 2b. BLOCKWISE: Scale partial sums and accumulate to FP32
        blockwise_accumulate(tile - 1);

        // 3. Barrier
        __syncthreads();

        // 4. Write staged data to next buffer
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
        if (threadIdx.x < B_ACTIVE_THREADS)
        {
            *reinterpret_cast<uint4*>(&b_lds[nxt_buf][b_lds_off]) = b_staging;
        }

        // 5. Barrier
        __syncthreads();
    }

    // FINAL TILE
    {
        const int last_buf = (num_tiles - 1) & 1;

        if constexpr (UNROLL_KK == 0) {
            #pragma clang loop unroll(disable)
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(last_buf) }
        } else if constexpr (UNROLL_KK == 2) {
            _Pragma("clang loop unroll_count(2)")
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(last_buf) }
        } else if constexpr (UNROLL_KK == 4) {
            _Pragma("clang loop unroll_count(4)")
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(last_buf) }
        } else { // UNROLL_KK == 1: full unroll
            #pragma unroll
            for (int kk = 0; kk < KT; ++kk) { V3_BW_KK_BODY(last_buf) }
        }

        // BLOCKWISE: Scale partial sums for the final tile
        blockwise_accumulate(num_tiles - 1);
    }

#undef V3_BW_KK_BODY

    // STORE FP32 RESULTS (activation scales already applied)
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;
        if (m_idx < M)
        {
            if constexpr (N_PER_THREAD >= 4)
            {
                if (n_base + N_PER_THREAD - 1 < N)
                {
                    const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                    #pragma unroll
                    for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                    {
                        *reinterpret_cast<float4*>(d_C_fp32 + row_off + chunk) =
                            make_float4(fp32_acc[mm][chunk + 0],
                                        fp32_acc[mm][chunk + 1],
                                        fp32_acc[mm][chunk + 2],
                                        fp32_acc[mm][chunk + 3]);
                    }
                }
                else
                {
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    {
                        const int n_idx = n_base + nn;
                        if (n_idx < N)
                            d_C_fp32[static_cast<int64_t>(m_idx) * N + n_idx] = fp32_acc[mm][nn];
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
                        d_C_fp32[static_cast<int64_t>(m_idx) * N + n_idx] = fp32_acc[mm][nn];
                }
            }
        }
    }
}

// ============================================================================
// BLOCKWISE V7: Safe-Tile Split with Per-Block Activation Scales
// ============================================================================
//
// Variant of V7 GEMM kernel with blockwise activation scale accumulation.
// Same structure as V7 (safe loop + boundary loop + final tile) but with
// FP32 running accumulators and per-tile blockwise scale application.
//
// KT parameter: kk-groups per tile. KT=8 = 1 activation block per tile.
// KT=16 support exists but M_TILE=64/KT=8 wins benchmarks (772 tok/s) because
// KT=16 either causes excessive VGPR spills (M_TILE=64→81 spills) or requires
// smaller M_TILE (M_TILE=32→757 tok/s) with worse weight reuse.
//
template <int N_TILE, int M_TILE = 128, int KT = 8, int UNROLL_KK = 0>
__global__ __launch_bounds__(256)
void qgemm_int8_vnni_wide_tile_v7_blockwise_kernel(
    const int8_t* __restrict__ d_A_int8,
    const int8_t* __restrict__ d_B_int8_vnni,
    float* __restrict__ d_C_fp32,
    const float* __restrict__ d_scales_A_blockwise,
    int M, int N, int K)
{
    constexpr int BLOCK_SIZE = 256;
    constexpr int M_PER_THREAD = 4;
    constexpr int THREADS_M = M_TILE / M_PER_THREAD;
    constexpr int THREADS_N = BLOCK_SIZE / THREADS_M;
    constexpr int N_PER_THREAD = N_TILE / THREADS_N;
    constexpr int KK_PER_BLOCK = 8;  // kk-groups per activation block (32 elements / 4)
    constexpr int BLOCKS_PER_TILE = KT / KK_PER_BLOCK;  // 1 for KT=8, 2 for KT=16

    static_assert(M_TILE == 16 || M_TILE == 32 || M_TILE == 64 || M_TILE == 128);
    static_assert(KT == 8 || KT == 16);
    static_assert(KT % KK_PER_BLOCK == 0);
    static_assert(THREADS_M * THREADS_N == BLOCK_SIZE);
    static_assert(N_TILE % THREADS_N == 0);
    static_assert(M_PER_THREAD == 4);

    const int n_block = blockIdx.x * N_TILE;
    const int m_block = blockIdx.y * M_TILE;
    const int t_m = threadIdx.x % THREADS_M;
    const int t_n = threadIdx.x / THREADS_M;
    const int m_base = t_m * M_PER_THREAD;
    const int n_base = n_block + t_n * N_PER_THREAD;

    __shared__ int32_t a_lds[2][KT * M_TILE];
    __shared__ int32_t b_lds[2][KT * N_TILE];

    int32_t acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            acc[mm][nn] = 0;

    float fp32_acc[M_PER_THREAD][N_PER_THREAD];
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
        #pragma unroll
        for (int nn = 0; nn < N_PER_THREAD; ++nn)
            fp32_acc[mm][nn] = 0.0f;

    const int total_k_groups = K / 4;
    const int blocks_per_row = total_k_groups / KK_PER_BLOCK;  // K/32: activation blocks per row
    const int32_t* a_global = reinterpret_cast<const int32_t*>(d_A_int8);
    const int32_t* b_global = reinterpret_cast<const int32_t*>(d_B_int8_vnni);
    const int a_row_stride = K / 4;

    constexpr int A_K_CHUNKS = KT / 4;
    constexpr int A_ROWS_PER_PASS = BLOCK_SIZE / A_K_CHUNKS;
    constexpr int A_NUM_PASSES = (M_TILE + A_ROWS_PER_PASS - 1) / A_ROWS_PER_PASS;

    const int a_kg = (threadIdx.x % A_K_CHUNKS) * 4;
    const int a_thread_mi_base = threadIdx.x / A_K_CHUNKS;

    int a_mi[A_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < A_NUM_PASSES; ++p)
        a_mi[p] = a_thread_mi_base + p * A_ROWS_PER_PASS;

    constexpr int B_N_CHUNKS = N_TILE / 4;
    constexpr int B_TOTAL_GROUPS = KT * B_N_CHUNKS;
    constexpr int B_NUM_PASSES = (B_TOTAL_GROUPS + BLOCK_SIZE - 1) / BLOCK_SIZE;

    struct BLoadCoord {
        int kk_local;
        int nl;
        int gn;
        int lds_off;
        bool active;
    };
    BLoadCoord b_coords[B_NUM_PASSES];
    #pragma unroll
    for (int p = 0; p < B_NUM_PASSES; ++p)
    {
        const int flat = p * BLOCK_SIZE + threadIdx.x;
        if (flat < B_TOTAL_GROUPS)
        {
            b_coords[p].kk_local = flat / B_N_CHUNKS;
            b_coords[p].nl = (flat % B_N_CHUNKS) * 4;
            b_coords[p].gn = n_block + b_coords[p].nl;
            b_coords[p].lds_off = b_coords[p].kk_local * N_TILE + b_coords[p].nl;
            b_coords[p].active = true;
        }
        else
        {
            b_coords[p] = {0, 0, 0, 0, false};
        }
    }

    const int num_tiles = (total_k_groups + KT - 1) / KT;
    if (num_tiles == 0) return;

    uint4 a_staging[A_NUM_PASSES];
    uint4 b_staging[B_NUM_PASSES];

    // Pre-compute safe tile count
    const bool a_m_safe = (m_block + M_TILE <= M);
    const bool b_n_safe = (n_block + N_TILE <= N);
    const int safe_k = total_k_groups / KT;
    const int num_safe = (a_m_safe && b_n_safe) ? (safe_k < num_tiles ? safe_k : num_tiles) : 0;

    // Blockwise scale+accumulate helper
    auto blockwise_accumulate = [&](int block_idx) __attribute__((always_inline))
    {
        #pragma unroll
        for (int mm = 0; mm < M_PER_THREAD; ++mm)
        {
            const int gm = m_block + m_base + mm;
            const float s = (gm < M)
                ? d_scales_A_blockwise[static_cast<int64_t>(gm) * blocks_per_row + block_idx]
                : 0.0f;
            #pragma unroll
            for (int nn = 0; nn < N_PER_THREAD; ++nn)
            {
                fp32_acc[mm][nn] += static_cast<float>(acc[mm][nn]) * s;
                acc[mm][nn] = 0;
            }
        }
    };

    // Safe load lambdas
    auto safe_issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (a_mi[p] >= M_TILE) continue; }
            a_staging[p] = *reinterpret_cast<const uint4*>(
                a_global + static_cast<int64_t>(m_block + a_mi[p]) * a_row_stride + kt_base + a_kg);
        }
    };

    auto safe_issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                b_staging[p] = *reinterpret_cast<const uint4*>(
                    b_global + static_cast<int64_t>(kt_base + b_coords[p].kk_local) * N + b_coords[p].gn);
            }
        }
    };

    // Boundary load lambdas
    auto boundary_issue_a_loads = [&](int kt_base) __attribute__((always_inline))
    {
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
    };

    auto boundary_issue_b_loads = [&](int kt_base) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                const int gkk = kt_base + b_coords[p].kk_local;
                const int gn = b_coords[p].gn;
                if (__builtin_expect(gn + 3 < N && gkk < total_k_groups, 1))
                {
                    b_staging[p] = *reinterpret_cast<const uint4*>(
                        b_global + static_cast<int64_t>(gkk) * N + gn);
                }
                else
                {
                    b_staging[p] = make_uint4(0, 0, 0, 0);
                    b_staging[p].x = (gn + 0 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 0] : 0;
                    b_staging[p].y = (gn + 1 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 1] : 0;
                    b_staging[p].z = (gn + 2 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 2] : 0;
                    b_staging[p].w = (gn + 3 < N && gkk < total_k_groups)
                        ? b_global[static_cast<int64_t>(gkk) * N + gn + 3] : 0;
                }
            }
        }
    };

    auto write_a_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < A_NUM_PASSES; ++p)
        {
            const int mi = a_mi[p];
            if constexpr (A_ROWS_PER_PASS > M_TILE) { if (mi >= M_TILE) continue; }
            a_lds[buf][(a_kg + 0) * M_TILE + mi] = a_staging[p].x;
            a_lds[buf][(a_kg + 1) * M_TILE + mi] = a_staging[p].y;
            a_lds[buf][(a_kg + 2) * M_TILE + mi] = a_staging[p].z;
            a_lds[buf][(a_kg + 3) * M_TILE + mi] = a_staging[p].w;
        }
    };

    auto write_b_to_lds = [&](int buf) __attribute__((always_inline))
    {
        #pragma unroll
        for (int p = 0; p < B_NUM_PASSES; ++p)
        {
            if (b_coords[p].active)
            {
                *reinterpret_cast<uint4*>(&b_lds[buf][b_coords[p].lds_off]) = b_staging[p];
            }
        }
    };

    // Fused compute + blockwise accumulate for one LDS tile.
    // For KT=8 (BLOCKS_PER_TILE=1): single 8-iteration phase + 1 scale application.
    // For KT=16 (BLOCKS_PER_TILE=2): two 8-iteration phases, each with its own scale.
    auto compute_and_accumulate_tile = [&](int buf, int tile_idx) __attribute__((always_inline))
    {
        const int32_t* __restrict__ a_ptr = &a_lds[buf][m_base];
        const int32_t* __restrict__ b_ptr = &b_lds[buf][t_n * N_PER_THREAD];

        #pragma unroll
        for (int phase = 0; phase < BLOCKS_PER_TILE; ++phase)
        {
            const int kk_start = phase * KK_PER_BLOCK;
            const int kk_end = kk_start + KK_PER_BLOCK;

#define V7_BW_KK_BODY \
                int32_t a_reg[M_PER_THREAD]; \
                int32_t b_reg[N_PER_THREAD]; \
                *reinterpret_cast<uint4*>(&a_reg[0]) = \
                    *reinterpret_cast<const uint4*>(&a_ptr[kk * M_TILE]); \
                if constexpr (N_PER_THREAD >= 4) { \
                    _Pragma("unroll") \
                    for (int c = 0; c < N_PER_THREAD; c += 4) \
                        *reinterpret_cast<uint4*>(&b_reg[c]) = \
                            *reinterpret_cast<const uint4*>(&b_ptr[kk * N_TILE + c]); \
                } else if constexpr (N_PER_THREAD == 2) { \
                    *reinterpret_cast<uint2*>(&b_reg[0]) = \
                        *reinterpret_cast<const uint2*>(&b_ptr[kk * N_TILE]); \
                } else { \
                    b_reg[0] = b_ptr[kk * N_TILE]; \
                } \
                _Pragma("unroll") \
                for (int mm = 0; mm < M_PER_THREAD; ++mm) \
                    _Pragma("unroll") \
                    for (int nn = 0; nn < N_PER_THREAD; ++nn) \
                        acc[mm][nn] = __builtin_amdgcn_sdot4( \
                            a_reg[mm], b_reg[nn], acc[mm][nn], false);

            if constexpr (UNROLL_KK == 0) {
                #pragma clang loop unroll(disable)
                for (int kk = kk_start; kk < kk_end; ++kk) { V7_BW_KK_BODY }
            } else if constexpr (UNROLL_KK == 2) {
                _Pragma("clang loop unroll_count(2)")
                for (int kk = kk_start; kk < kk_end; ++kk) { V7_BW_KK_BODY }
            } else if constexpr (UNROLL_KK == 4) {
                _Pragma("clang loop unroll_count(4)")
                for (int kk = kk_start; kk < kk_end; ++kk) { V7_BW_KK_BODY }
            } else { // UNROLL_KK == 1: full unroll
                #pragma unroll
                for (int kk = kk_start; kk < kk_end; ++kk) { V7_BW_KK_BODY }
            }

#undef V7_BW_KK_BODY

            blockwise_accumulate(tile_idx * BLOCKS_PER_TILE + phase);
        }
    };

    // PRELOAD tile 0
    if (num_safe > 0)
    {
        safe_issue_a_loads(0);
        safe_issue_b_loads(0);
    }
    else
    {
        boundary_issue_a_loads(0);
        boundary_issue_b_loads(0);
    }
    write_a_to_lds(0);
    write_b_to_lds(0);
    __syncthreads();

    // SAFE LOOP
    for (int tile = 1; tile < num_safe; ++tile)
    {
        const int cur_buf = (tile - 1) & 1;
        const int nxt_buf = tile & 1;
        const int kt_base = tile * KT;

        safe_issue_a_loads(kt_base);
        safe_issue_b_loads(kt_base);

        compute_and_accumulate_tile(cur_buf, tile - 1);

        __syncthreads();

        write_a_to_lds(nxt_buf);
        write_b_to_lds(nxt_buf);

        __syncthreads();
    }

    // BOUNDARY LOOP
    {
        const int boundary_start = (num_safe > 0) ? num_safe : 1;
        for (int tile = boundary_start; tile < num_tiles; ++tile)
        {
            const int cur_buf = (tile - 1) & 1;
            const int nxt_buf = tile & 1;
            const int kt_base = tile * KT;

            boundary_issue_a_loads(kt_base);
            boundary_issue_b_loads(kt_base);

            compute_and_accumulate_tile(cur_buf, tile - 1);

            __syncthreads();

            write_a_to_lds(nxt_buf);
            write_b_to_lds(nxt_buf);

            __syncthreads();
        }
    }

    // FINAL TILE
    compute_and_accumulate_tile((num_tiles - 1) & 1, num_tiles - 1);

    // STORE FP32 RESULTS
    #pragma unroll
    for (int mm = 0; mm < M_PER_THREAD; ++mm)
    {
        const int m_idx = m_block + m_base + mm;
        if (m_idx < M)
        {
            if constexpr (N_PER_THREAD >= 4)
            {
                if (n_base + N_PER_THREAD - 1 < N)
                {
                    const int64_t row_off = static_cast<int64_t>(m_idx) * N + n_base;
                    #pragma unroll
                    for (int chunk = 0; chunk < N_PER_THREAD; chunk += 4)
                    {
                        *reinterpret_cast<float4*>(d_C_fp32 + row_off + chunk) =
                            make_float4(fp32_acc[mm][chunk + 0],
                                        fp32_acc[mm][chunk + 1],
                                        fp32_acc[mm][chunk + 2],
                                        fp32_acc[mm][chunk + 3]);
                    }
                }
                else
                {
                    #pragma unroll
                    for (int nn = 0; nn < N_PER_THREAD; ++nn)
                    {
                        const int n_idx = n_base + nn;
                        if (n_idx < N)
                            d_C_fp32[static_cast<int64_t>(m_idx) * N + n_idx] = fp32_acc[mm][nn];
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
                        d_C_fp32[static_cast<int64_t>(m_idx) * N + n_idx] = fp32_acc[mm][nn];
                }
            }
        }
    }
}

extern "C"
{

// ============================================================================
// BLOCKWISE LAUNCHER FUNCTIONS
// ============================================================================

    /**
     * @brief Dispatch blockwise V3 INT8 GEMM (LDS double-buffered, KT=8 forced).
     *
     * Output is FP32 with per-block activation scales already applied.
     * Caller must apply weight scales (scale_B) in a separate epilogue.
     */
    bool rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v3_blockwise(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        float* d_C_fp32,
        const float* d_scales_A_blockwise,
        int M, int N, int K,
        int rocm_device_id, void* stream,
        int m_tile_override,
        int unroll_kk)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_fp32 || !d_scales_A_blockwise ||
            M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[v3_blockwise] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 32) != 0)
        {
            fprintf(stderr, "[v3_blockwise] K must be multiple of 32 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        constexpr int N_TILE = 64;
        constexpr int BLOCK_SIZE = 256;
        // Cap M_TILE at 32 for blockwise: ISA shows M_TILE=64 → 147 VGPRs (1 wave),
        // M_TILE=128 → 213 VGPRs (1 wave). M_TILE=32 → 115 VGPRs, 2 waves, 0 spills.
        constexpr int MAX_M_TILE = 32;
        const auto s = static_cast<hipStream_t>(stream);

        auto select_m_tile = [](int rows) -> int {
            int best_mt = MAX_M_TILE;
            int best_waste = ((rows + MAX_M_TILE - 1) / MAX_M_TILE) * MAX_M_TILE - rows;
            for (int mt : {16}) {
                int waste = ((rows + mt - 1) / mt) * mt - rows;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_mt = mt;
                }
            }
            return best_mt;
        };

        // Use override if provided, otherwise auto-select
        int m_tile;
        if (m_tile_override > 0) {
            m_tile = m_tile_override;
        } else {
            m_tile = (M <= MAX_M_TILE) ? select_m_tile(M) : MAX_M_TILE;
        }
        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const unsigned m_tiles = static_cast<unsigned>((M + m_tile - 1) / m_tile);
        const dim3 grid(n_tiles, m_tiles, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

#define LAUNCH_V3B(MT, UK) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v3_blockwise_kernel<N_TILE, MT, UK>), \
                           grid, block, 0, s, d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, M, N, K)

#define LAUNCH_V3B_UNROLL(MT) \
        switch (unroll_kk) { \
            case 0:  LAUNCH_V3B(MT, 0); break; \
            case 1:  LAUNCH_V3B(MT, 1); break; \
            case 4:  LAUNCH_V3B(MT, 4); break; \
            default: LAUNCH_V3B(MT, 2); break; \
        }

        switch (m_tile) {
            case 16:  LAUNCH_V3B_UNROLL(16);  break;
            case 64:  LAUNCH_V3B_UNROLL(64);  break;
            default:  LAUNCH_V3B_UNROLL(32);  break;
        }
#undef LAUNCH_V3B_UNROLL
#undef LAUNCH_V3B

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[v3_blockwise] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                    hipGetErrorString(err), M, N, K);
            return false;
        }

        return true;
    }

    /**
     * @brief Dispatch blockwise V7 INT8 GEMM (safe-tile split).
     *
     * Output is FP32 with per-block activation scales already applied.
     * Caller must apply weight scales (scale_B) in a separate epilogue.
     *
     * Dispatches KT=16 when K is a multiple of 64 (2 activation blocks per tile),
     * otherwise falls back to KT=8 (1 block per tile). KT=16 halves the tile count,
     * reducing barrier and LDS load overhead.
     */
    bool rocmQuantGemm_int8_int8_fp32_vnni_prefill_wide_tile_v7_blockwise(
        const int8_t* d_A_int8,
        const int8_t* d_B_int8_vnni,
        float* d_C_fp32,
        const float* d_scales_A_blockwise,
        int M, int N, int K,
        int rocm_device_id, void* stream,
        int m_tile_override,
        int unroll_kk)
    {
        if (!d_A_int8 || !d_B_int8_vnni || !d_C_fp32 || !d_scales_A_blockwise ||
            M <= 0 || N <= 0 || K <= 0)
        {
            fprintf(stderr, "[v7_blockwise] Invalid arguments (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if ((K % 32) != 0)
        {
            fprintf(stderr, "[v7_blockwise] K must be multiple of 32 (M=%d N=%d K=%d)\n", M, N, K);
            return false;
        }

        if (!rocmQuantGemm_setDevice(rocm_device_id))
            return false;

        constexpr int N_TILE = 128;
        constexpr int BLOCK_SIZE = 256;
        // Cap M_TILE at 64 for blockwise: ISA shows M_TILE=128 → 128 VGPRs but 167 spills!
        // M_TILE=64 → 123 VGPRs, 2 waves, 0 spills. Best weight reuse at 2 waves.
        // Measured: M_TILE=64→772 tok/s, M_TILE=32→727, M_TILE=16→691 (more tiles = more overhead).
        constexpr int MAX_M_TILE = 64;
        const auto s = static_cast<hipStream_t>(stream);

        auto select_m_tile = [](int rows) -> int {
            int best_mt = MAX_M_TILE;
            int best_waste = ((rows + MAX_M_TILE - 1) / MAX_M_TILE) * MAX_M_TILE - rows;
            for (int mt : {32, 16}) {
                int waste = ((rows + mt - 1) / mt) * mt - rows;
                if (waste < best_waste) {
                    best_waste = waste;
                    best_mt = mt;
                }
            }
            return best_mt;
        };

        // Use override if provided, otherwise auto-select
        int m_tile;
        if (m_tile_override > 0) {
            m_tile = m_tile_override;
        } else {
            m_tile = (M <= MAX_M_TILE) ? select_m_tile(M) : MAX_M_TILE;
        }
        const unsigned n_tiles = static_cast<unsigned>((N + N_TILE - 1) / N_TILE);
        const unsigned m_tiles = static_cast<unsigned>((M + m_tile - 1) / m_tile);
        const dim3 grid(n_tiles, m_tiles, 1);
        const dim3 block(BLOCK_SIZE, 1, 1);

#define LAUNCH_V7B(MT, UK) \
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_int8_vnni_wide_tile_v7_blockwise_kernel<N_TILE, MT, 8, UK>), \
                           grid, block, 0, s, d_A_int8, d_B_int8_vnni, d_C_fp32, d_scales_A_blockwise, M, N, K)

#define LAUNCH_V7B_UNROLL(MT) \
        switch (unroll_kk) { \
            case 0:  LAUNCH_V7B(MT, 0); break; \
            case 1:  LAUNCH_V7B(MT, 1); break; \
            case 4:  LAUNCH_V7B(MT, 4); break; \
            default: LAUNCH_V7B(MT, 2); break; \
        }

        switch (m_tile) {
            case 16:  LAUNCH_V7B_UNROLL(16);  break;
            case 32:  LAUNCH_V7B_UNROLL(32);  break;
            default:  LAUNCH_V7B_UNROLL(64);  break;
        }
#undef LAUNCH_V7B_UNROLL
#undef LAUNCH_V7B

        const hipError_t err = hipGetLastError();
        if (err != hipSuccess)
        {
            fprintf(stderr, "[v7_blockwise] Kernel launch failed: %s (M=%d N=%d K=%d)\n",
                    hipGetErrorString(err), M, N, K);
            return false;
        }

        return true;
    }

} // extern "C"
