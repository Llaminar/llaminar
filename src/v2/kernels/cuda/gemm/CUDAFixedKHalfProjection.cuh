/**
 * @file CUDAFixedKHalfProjection.cuh
 * @brief Tensor-core half-RNE prefill projections with the fixed-K partition contract.
 *
 * Every schedule evaluates `kernels/common/FixedKProjectionContract.h` with
 * m16n8k16 MMA units: eight K partitions of 64-element blocks, each an
 * ascending MMA chain from +0, folded left to right in FP32. One cp.async
 * multi-stage kernel body serves both modes:
 *
 *  - Tiled: a CTA walks all of K for its output tile, keeps a partition
 *    accumulator `q` and folded result `r`, and flushes q into r at each
 *    partition boundary. Wide groups have enough output tiles to fill the GPU.
 *  - Partial: a CTA walks one partition (grid.z) and publishes its chain to
 *    workspace; foldPartialsKernel then folds P0..P7 left to right. Narrow
 *    groups, such as GDN alpha/beta, need this K parallelism.
 *
 * Columns are tiled in the group's compact column space, so a CTA may span
 * ragged projections and reads each input tile once for all of them. Lane
 * operands come from contiguous 32-byte spans via 128-bit shared loads
 * (the contract's slot assignment). Nothing depends on M, row position or
 * launch geometry, so a prompt row has identical bytes in every prefill
 * chunk and bucket, and every schedule is byte-identical to every other.
 *
 * Kernels borrow a validated FixedHalfProjectionGroup by value: one published
 * half input [M,K], up to eight persistent half weights [N_p,K] and disjoint
 * FP32 outputs [M,N_p]. They allocate nothing; only the partial mode needs
 * caller-owned workspace. Matrix bases are 16-byte aligned (arena and
 * prepared-weight owners); K that is a multiple of eight uses cp.async, any
 * other positive K uses an element-gather staging variant.
 */
#pragma once

#include "kernels/common/FixedHalfProjectionGroup.h"
#include "kernels/common/FixedKProjectionContract.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace llaminar2::cuda::fixed_k_half
{
using Group = FixedHalfProjectionGroup;
namespace contract = ::llaminar2::fixed_k_projection;

/** @brief Elements in one reduction block; also every schedule's staged K width. */
inline constexpr int kBlock = contract::kHalfMMABlockWidth;
/** @brief Row tiles per tiled-mode rasterization band (see fixedKProjectionKernel). */
inline constexpr int kRowTileBand = 16;

/**
 * @brief One m16n8k16 tensor-core unit: c += A(16x16 half) * B(16x8 half), FP32 accumulate.
 * @param a Row-major A fragment words (rows g / g+8, slot pairs lo / hi).
 * @param b0 B fragment word for the low slot pair. @param b1 B fragment word for the high pair.
 * @param c FP32 accumulator fragment, updated in place.
 */
__device__ __forceinline__ void mmaUnit(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                        uint32_t b0, uint32_t b1, float (&c)[4])
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

/** @brief 16-byte cp.async to a 32-bit shared-window address; zero-fills when invalid. */
__device__ __forceinline__ void copyAsync16(unsigned address, const void *global, bool valid)
{
    const int bytes = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(address), "l"(global), "r"(bytes));
}
/** @brief Close the current cp.async group. */
__device__ __forceinline__ void commitAsync() { asm volatile("cp.async.commit_group;\n"); }
/** @brief Wait until at most @p Pending cp.async groups remain in flight. */
template<int Pending>
__device__ __forceinline__ void waitAsync() { asm volatile("cp.async.wait_group %0;\n" ::"n"(Pending)); }
/** @brief 128-bit shared load from a 32-bit shared-window address. */
__device__ __forceinline__ uint4 loadShared16(unsigned address)
{
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(address));
    return v;
}

/**
 * @brief Unaligned-K staging: gather one 8-element chunk and store it to shared memory.
 * @param dst Shared chunk address. @param src First element of the chunk.
 * @param origin K index of the chunk's first element. @param k Reduction width.
 * @param row_ok Whether the row exists; absent rows and elements past K stage zero.
 * @details The target slot was last read before the pipeline barrier that
 * precedes this stage call, so the synchronous store cannot race a reader.
 */
__device__ __forceinline__ void gatherChunk(unsigned dst, const uint16_t *src, int origin, int k, bool row_ok)
{
    uint32_t words[4];
#pragma unroll
    for (int w = 0; w < 4; ++w)
    {
        const int e0 = origin + 2 * w;
        const uint32_t lo = row_ok && e0 < k ? src[2 * w] : 0u;
        const uint32_t hi = row_ok && e0 + 1 < k ? src[2 * w + 1] : 0u;
        words[w] = lo | (hi << 16);
    }
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "r"(words[0]), "r"(words[1]),
                 "r"(words[2]), "r"(words[3]));
}

/**
 * @brief Byte offset of a 16-byte chunk in a staged 64-wide (128-byte) row.
 * @param row Tile row. @param chunk Logical chunk 0..7.
 * @return XOR-swizzled offset; eight consecutive rows reading one logical chunk
 *         pair hit eight distinct bank groups for 128-bit shared loads.
 */
__device__ __forceinline__ unsigned swizzledChunk(int row, int chunk)
{
    return static_cast<unsigned>(row * kBlock * 2 + ((chunk ^ (row & 7)) * 16));
}

/**
 * @brief Apply two MMA units (one half of a block) from 16-byte half spans.
 * @param a Words for rows g / g+8 of every m16 tile: units (2h, 2h+1) of the block.
 * @param b Words for column g of every n8 tile. @param acc Partition accumulators.
 */
template<int TM, int TN>
__device__ __forceinline__ void halfBlockUnits(const uint4 (&a)[TM][2], const uint4 (&b)[TN],
                                               float (&acc)[TM][TN][4])
{
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            mmaUnit(a[i][0].x, a[i][1].x, a[i][0].y, a[i][1].y, b[j].x, b[j].y, acc[i][j]);
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            mmaUnit(a[i][0].z, a[i][1].z, a[i][0].w, a[i][1].w, b[j].z, b[j].w, acc[i][j]);
}

/**
 * @brief Resolve a compact group column to its projection and local column.
 * @param g Validated group. @param compact Column in [0, totalColumns()).
 * @param local Out: column within the owning projection. @return Projection index.
 */
__device__ __forceinline__ int projectionOf(const Group &g, int compact, int &local)
{
    int p = 0;
    while (compact >= g.columnEnd(p)) ++p;
    local = compact - (p ? g.columnEnd(p - 1) : 0);
    return p;
}

/** @brief Whether a CTA evaluates all partitions (Tiled) or one published partition (Partial). */
enum class FixedKMode { Tiled, Partial };

/**
 * @brief Fixed-K half projection body for both schedule modes.
 * @tparam BM CTA rows. @tparam BN CTA compact columns. @tparam WM Warps along M.
 * @tparam WN Warps along N. @tparam STAGES Shared-memory pipeline depth (blocks).
 * @tparam Mode Tiled register fold or Partial publication (partition = grid.z).
 * @tparam Vector K is a multiple of eight, so every 16-byte chunk is aligned
 *         and copied with cp.async; otherwise chunks are gathered element by
 *         element (zero past K) and stored synchronously. Arithmetic is identical.
 * @param g Borrowed group. @param m Physical rows. @param k Reduction width (multiple of 8).
 * @param partials Partial mode only: [kPartitions][m][totalColumns] chains.
 *
 * Each pipeline stage holds one 64-element block of the CTA's input rows and
 * compact weight rows; every staged weight row resolves its projection once.
 * grid.x walks M tiles so CTAs that share a weight tile run together and
 * re-read it from L2. Fragments are spans of the contract's slot assignment:
 * a lane's first 16 bytes feed units 0/1, its second 16 bytes units 2/3.
 */
template<int BM, int BN, int WM, int WN, int STAGES, FixedKMode Mode, bool Vector = true>
__global__ __launch_bounds__(WM * WN * 32, 1) void fixedKProjectionKernel(
    const __grid_constant__ Group g, int m, int k, float *__restrict__ partials)
{
    constexpr int THREADS = WM * WN * 32;
    constexpr int TM = BM / WM / 16, TN = BN / WN / 8;
    constexpr int CHUNKS = kBlock / 8;
    static_assert(TM >= 1 && TN >= 1 && STAGES >= 2);
    static_assert((BM * CHUNKS) % THREADS == 0 && (BN * CHUNKS) % THREADS == 0,
                  "Copy slots must divide evenly across the CTA");
    constexpr int A_SLOTS = BM * CHUNKS / THREADS, B_SLOTS = BN * CHUNKS / THREADS;
    constexpr unsigned A_STAGE_BYTES = BM * kBlock * 2, B_STAGE_BYTES = BN * kBlock * 2;
    extern __shared__ __align__(128) uint16_t staged[];

    const int total = g.totalColumns();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int group = lane / 4, quad = lane % 4;
    const int wm = warp / WN, wn = warp % WN;
    // Tiled mode rasterizes a 1-D grid in bands of up to kRowTileBand row
    // tiles: each band walks every column tile while its input rows stay in
    // L2, so large prompts stream the weights once per band rather than the
    // input once per column tile. Partial mode keeps its (row, column, part) grid.
    int row_tile = static_cast<int>(blockIdx.x), col_tile = static_cast<int>(blockIdx.y);
    if constexpr (Mode == FixedKMode::Tiled)
    {
        const int row_tiles = (m + BM - 1) / BM, col_tiles = (total + BN - 1) / BN;
        const int linear = static_cast<int>(blockIdx.x);
        const int band = linear / (kRowTileBand * col_tiles);
        const int band_first = band * kRowTileBand;
        const int band_rows = min(row_tiles - band_first, kRowTileBand);
        const int within = linear - band * kRowTileBand * col_tiles;
        row_tile = band_first + within % band_rows;
        col_tile = within / band_rows;
    }
    const int m0 = row_tile * BM, c0 = col_tile * BN;
    const int blocks = contract::halfMMABlocks(k);
    const int first = Mode == FixedKMode::Partial ? contract::partitionBegin(blockIdx.z, blocks) : 0;
    const int walk = (Mode == FixedKMode::Partial ? contract::partitionBegin(blockIdx.z + 1, blocks) : blocks) - first;
    const uint16_t *A = g.input();
    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(staged));
    const unsigned sb = sa + STAGES * A_STAGE_BYTES;

    // Each thread owns fixed (row, chunk) copy slots for the whole K walk.
    const uint16_t *a_src[A_SLOTS]; unsigned a_dst[A_SLOTS]; int a_k[A_SLOTS]; bool a_ok[A_SLOTS];
    const uint16_t *b_src[B_SLOTS]; unsigned b_dst[B_SLOTS]; int b_k[B_SLOTS]; bool b_ok[B_SLOTS];
#pragma unroll
    for (int i = 0; i < A_SLOTS; ++i)
    {
        const int c = threadIdx.x + i * THREADS, row = c / CHUNKS, chunk = c % CHUNKS;
        a_ok[i] = m0 + row < m;
        a_k[i] = chunk * 8;
        a_src[i] = A + static_cast<size_t>(a_ok[i] ? m0 + row : 0) * k + chunk * 8;
        a_dst[i] = sa + swizzledChunk(row, chunk);
    }
#pragma unroll
    for (int i = 0; i < B_SLOTS; ++i)
    {
        const int c = threadIdx.x + i * THREADS, row = c / CHUNKS, chunk = c % CHUNKS;
        b_ok[i] = c0 + row < total;
        int local = 0;
        const int p = b_ok[i] ? projectionOf(g, c0 + row, local) : 0;
        b_k[i] = chunk * 8;
        b_src[i] = g.weight(p) + static_cast<size_t>(local) * k + chunk * 8;
        b_dst[i] = sb + swizzledChunk(row, chunk);
    }
    auto stage = [&](int slot, int block) {
        const int k0 = block * kBlock;
#pragma unroll
        for (int i = 0; i < A_SLOTS; ++i)
        {
            if constexpr (Vector)
            {
                const bool valid = a_ok[i] && k0 + a_k[i] < k;
                copyAsync16(a_dst[i] + slot * A_STAGE_BYTES, valid ? a_src[i] + k0 : A, valid);
            }
            else
                gatherChunk(a_dst[i] + slot * A_STAGE_BYTES, a_src[i] + k0, k0 + a_k[i], k, a_ok[i]);
        }
#pragma unroll
        for (int i = 0; i < B_SLOTS; ++i)
        {
            if constexpr (Vector)
            {
                const bool valid = b_ok[i] && k0 + b_k[i] < k;
                copyAsync16(b_dst[i] + slot * B_STAGE_BYTES, valid ? b_src[i] + k0 : A, valid);
            }
            else
                gatherChunk(b_dst[i] + slot * B_STAGE_BYTES, b_src[i] + k0, k0 + b_k[i], k, b_ok[i]);
        }
    };

    float q[TM][TN][4] = {};
    // Tiled mode only: folded result and the next partition boundary. Partial
    // mode never reads them, so the compiler removes them from that variant.
    float r[TM][TN][4] = {};
    int partition = 0;
    int boundary = contract::partitionBegin(1, blocks);
    // Fold every partition whose last block is `done`; empty partitions add +0.
    auto flush = [&](int done) {
        while (partition < contract::kPartitions && boundary == done)
        {
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                    {
                        r[i][j][e] = partition ? __fadd_rn(r[i][j][e], q[i][j][e]) : q[i][j][e];
                        q[i][j][e] = 0.0f;
                    }
            ++partition;
            boundary = contract::partitionBegin(partition + 1, blocks);
        }
    };
    if constexpr (Mode == FixedKMode::Tiled) flush(0);

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < walk) stage(s, first + s);
        commitAsync();
    }
    for (int step = 0; step < walk; ++step)
    {
        waitAsync<STAGES - 2>();
        __syncthreads();
        {
            const int next = step + STAGES - 1;
            if (next < walk) stage(next % STAGES, first + next);
            commitAsync();
        }
        const unsigned ta = sa + (step % STAGES) * A_STAGE_BYTES;
        const unsigned tb = sb + (step % STAGES) * B_STAGE_BYTES;
#pragma unroll
        for (int half = 0; half < 2; ++half)
        {
            uint4 a[TM][2], b[TN];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = loadShared16(tb + swizzledChunk(wn * (BN / WN) + j * 8 + group, 2 * quad + half));
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int h = 0; h < 2; ++h)
                    a[i][h] = loadShared16(ta + swizzledChunk(wm * (BM / WM) + i * 16 + group + h * 8,
                                                              2 * quad + half));
            halfBlockUnits<TM, TN>(a, b, q);
        }
        if constexpr (Mode == FixedKMode::Tiled) flush(first + step + 1);
    }
    waitAsync<0>();

    // Partial mode publishes q in compact order; tiled mode stores r per projection.
    // Each lane owns column pairs (col, col+1); the pair's projection is resolved
    // once and a pair that straddles projections or ends a row falls back to
    // scalar stores, so stores never cross an owner boundary.
    float *out = Mode == FixedKMode::Partial ? partials + static_cast<size_t>(blockIdx.z) * m * total : nullptr;
#pragma unroll
    for (int j = 0; j < TN; ++j)
    {
        const int col = c0 + wn * (BN / WN) + j * 8 + quad * 2;
        if (col >= total) continue;
        int local = 0;
        const int p = Mode == FixedKMode::Tiled ? projectionOf(g, col, local) : 0;
        const int width = Mode == FixedKMode::Tiled ? g.columns(p) : total;
        const int base_col = Mode == FixedKMode::Tiled ? local : col;
        float *dst_base = Mode == FixedKMode::Tiled ? g.output(p) : out;
        const bool pair = base_col + 1 < width && ((width | base_col) & 1) == 0;
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int row = m0 + wm * (BM / WM) + i * 16 + group + h * 8;
                if (row >= m) continue;
                const float v0 = Mode == FixedKMode::Tiled ? r[i][j][2 * h] : q[i][j][2 * h];
                const float v1 = Mode == FixedKMode::Tiled ? r[i][j][2 * h + 1] : q[i][j][2 * h + 1];
                float *dst = dst_base + static_cast<size_t>(row) * width + base_col;
                if (pair)
                    *reinterpret_cast<float2 *>(dst) = make_float2(v0, v1);
                else
                {
                    dst[0] = v0;
                    if (col + 1 < total)
                    {
                        // The second column may belong to the next projection.
                        int next_local = 0;
                        const int next = Mode == FixedKMode::Tiled ? projectionOf(g, col + 1, next_local) : 0;
                        if (Mode == FixedKMode::Tiled)
                            g.output(next)[static_cast<size_t>(row) * g.columns(next) + next_local] = v1;
                        else
                            dst[1] = v1;
                    }
                }
            }
    }
}

/**
 * @brief Fold published partition partials left to right into the group outputs.
 * @param g Borrowed group (compact column order). @param m Physical rows.
 * @param partials [kPartitions][m][totalColumns] chains from the partial mode.
 *
 * One thread per output: R = P0, then R = R + Pp for p = 1..7, exactly the
 * tiled mode's register fold. Reads are coalesced along compact columns.
 */
__global__ __launch_bounds__(256) void foldPartialsKernel(
    const __grid_constant__ Group g, int m, const float *__restrict__ partials)
{
    const int total = g.totalColumns();
    const size_t outputs = static_cast<size_t>(m) * total;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= outputs) return;
    float result = partials[index];
#pragma unroll
    for (int part = 1; part < contract::kPartitions; ++part)
        result = __fadd_rn(result, partials[part * outputs + index]);
    const int row = static_cast<int>(index / total);
    int local = 0;
    const int p = projectionOf(g, static_cast<int>(index % total), local);
    g.output(p)[static_cast<size_t>(row) * g.columns(p) + local] = result;
}

/** @brief Dynamic shared-memory bytes for one instantiation. */
template<int BM, int BN, int STAGES>
inline constexpr int fixedKSharedBytes = STAGES * (BM + BN) * kBlock * 2;
} // namespace llaminar2::cuda::fixed_k_half
