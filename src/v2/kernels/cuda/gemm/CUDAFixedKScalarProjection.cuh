/**
 * @file CUDAFixedKScalarProjection.cuh
 * @brief SGEMM-class CUDA kernel for strict-FP32 fixed-K prefill projections.
 *
 * Evaluates the scalar form of `kernels/common/FixedKProjectionContract.h`:
 * eight K partitions fixed by K, each one IEEE FMA chain from +0 over its
 * elements in ascending order, folded left to right. It produces exactly the
 * bytes of `kernels/common/FixedKScalarProjectionBody.cuh` (narrow and HIP
 * schedules) and of a host FMA reference.
 *
 * Wide groups use a register-blocked tile: each thread owns TM x TN outputs at
 * rows ty + i*(BM/TM) and columns tx + j*(BN/TN). Input and weight tiles stay
 * row-major in shared memory, copied by a multi-stage cp.async pipeline, and
 * each thread reads four consecutive K values per row with one 128-bit shared
 * load; the strided row/column ownership keeps those loads bank-conflict free.
 * Each output keeps a partition chain `q` and a folded result `r`; when the
 * staged tile ends a partition the fold r = q (first) or r += q runs once.
 */
#pragma once

#include "kernels/common/FixedHalfProjectionGroup.h"
#include "kernels/common/FixedKProjectionContract.h"

#include <cuda_runtime.h>
#include <cstdint>

namespace llaminar2::cuda::fixed_k_scalar_cuda
{
namespace contract = ::llaminar2::fixed_k_projection;
using Group = FixedStrictFP32ProjectionGroup;

/** @brief 16-byte cp.async into a 32-bit shared address; zero-fills when invalid. */
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

/**
 * @brief Register-blocked fixed-K FP32 projection.
 * @tparam BM CTA rows. @tparam BN CTA compact columns. @tparam TM/TN Outputs per thread.
 * @tparam BK Staged K elements per pipeline stage (multiple of 4). @tparam STAGES Pipeline depth.
 * @tparam Aligned K % 4 == 0 and every partition boundary (multiples of K/8) ends a staged
 *         tile, so 16-byte copies are aligned and the fold runs once per tile; otherwise
 *         elements are gathered and the fold is checked per element.
 * @tparam SPLIT Raster tiles are (BM*SPLIT) x BN; this CTA computes row slice
 *         blockIdx.x % SPLIT of raster tile tile_first + blockIdx.x / SPLIT. The
 *         host launches full waves with SPLIT=1 and the final partial wave with
 *         thinner slices so its work spreads over every SM. Per-output arithmetic
 *         never depends on the tiling.
 * @param g Borrowed group. @param m Physical rows. @param k Reduction width.
 * @param tile_first First raster tile of this launch.
 */
template<int BM, int BN, int TM, int TN, int BK, int STAGES, bool Aligned, int SPLIT = 1>
__global__ __launch_bounds__((BM / TM) * (BN / TN), 1) void sgemmProjectionKernel(
    const __grid_constant__ Group g, int m, int k, int tile_first)
{
    constexpr int TY = BM / TM, TX = BN / TN, THREADS = TY * TX;
    constexpr int PITCH = BK + 4;                         // floats; +4 rotates 16-byte bank groups
    constexpr int CHUNKS = BK / 4;
    static_assert(BK % 4 == 0 && BM % TM == 0 && BN % TN == 0);
    // Copy slots per thread; a final partial round leaves some threads idle so
    // column tiles need not be powers of two (wave-fitting widths).
    constexpr int A_SLOTS = (BM * CHUNKS + THREADS - 1) / THREADS, B_SLOTS = (BN * CHUNKS + THREADS - 1) / THREADS;
    constexpr unsigned A_STAGE = BM * PITCH * 4, B_STAGE = BN * PITCH * 4;
    extern __shared__ __align__(16) float staged[];

    const int total = g.totalColumns();
    const int tid = static_cast<int>(threadIdx.x), tx = tid % TX, ty = tid / TX;
    // Banded raster: up to 16 row tiles walk every column tile while their rows stay in L2.
    constexpr int RBM = BM * SPLIT;
    const int row_tiles = (m + RBM - 1) / RBM, col_tiles = (total + BN - 1) / BN;
    const int linear = tile_first + static_cast<int>(blockIdx.x) / SPLIT;
    const int band = linear / (16 * col_tiles), band_rows = min(row_tiles - band * 16, 16);
    const int within = linear - band * 16 * col_tiles;
    const int m0 = (band * 16 + within % band_rows) * RBM + (static_cast<int>(blockIdx.x) % SPLIT) * BM;
    const int c0 = (within / band_rows) * BN;

    const unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(staged));
    const unsigned sb = sa + STAGES * A_STAGE;
    const float *a_src[A_SLOTS]; unsigned a_dst[A_SLOTS]; int a_k[A_SLOTS]; bool a_ok[A_SLOTS], a_slot[A_SLOTS];
    const float *b_src[B_SLOTS]; unsigned b_dst[B_SLOTS]; int b_k[B_SLOTS]; bool b_ok[B_SLOTS], b_slot[B_SLOTS];
#pragma unroll
    for (int i = 0; i < A_SLOTS; ++i)
    {
        const int c = tid + i * THREADS, row = c / CHUNKS, chunk = c % CHUNKS;
        a_slot[i] = c < BM * CHUNKS;
        a_ok[i] = a_slot[i] && m0 + row < m;
        a_k[i] = chunk * 4;
        a_src[i] = g.input() + static_cast<size_t>(a_ok[i] ? m0 + row : 0) * k + chunk * 4;
        a_dst[i] = sa + static_cast<unsigned>(((a_slot[i] ? row : 0) * PITCH + chunk * 4) * 4);
    }
#pragma unroll
    for (int i = 0; i < B_SLOTS; ++i)
    {
        const int c = tid + i * THREADS, row = c / CHUNKS, chunk = c % CHUNKS;
        b_slot[i] = c < BN * CHUNKS;
        b_ok[i] = b_slot[i] && c0 + row < total;
        int local = 0;
        const int p = b_ok[i] ? projectionOf(g, c0 + row, local) : 0;
        b_k[i] = chunk * 4;
        b_src[i] = g.weight(p) + static_cast<size_t>(local) * k + chunk * 4;
        b_dst[i] = sb + static_cast<unsigned>(((b_slot[i] ? row : 0) * PITCH + chunk * 4) * 4);
    }
    auto stage = [&](int slot, int k0) {
#pragma unroll
        for (int i = 0; i < A_SLOTS; ++i)
        {
            if (!a_slot[i]) continue;
            const unsigned dst = a_dst[i] + slot * A_STAGE;
            if constexpr (Aligned)
            {
                const bool valid = a_ok[i] && k0 + a_k[i] < k;
                copyAsync16(dst, valid ? a_src[i] + k0 : g.input(), valid);
            }
            else
            {
                // Unaligned K: gather; the slot was released by the preceding barrier.
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    v[e] = a_ok[i] && k0 + a_k[i] + e < k ? a_src[i][k0 + e] : 0.0f;
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
            }
        }
#pragma unroll
        for (int i = 0; i < B_SLOTS; ++i)
        {
            if (!b_slot[i]) continue;
            const unsigned dst = b_dst[i] + slot * B_STAGE;
            if constexpr (Aligned)
            {
                const bool valid = b_ok[i] && k0 + b_k[i] < k;
                copyAsync16(dst, valid ? b_src[i] + k0 : g.input(), valid);
            }
            else
            {
                float v[4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    v[e] = b_ok[i] && k0 + b_k[i] + e < k ? b_src[i][k0 + e] : 0.0f;
                asm volatile("st.shared.v4.f32 [%0], {%1,%2,%3,%4};\n" ::"r"(dst), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]));
            }
        }
    };

    float q[TM][TN] = {}, r[TM][TN] = {};
    int partition = 0;
    int boundary = contract::partitionBegin(1, k);
    const auto flush = [&](int done) {
        while (partition < contract::kPartitions && boundary == done)
        {
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    r[i][j] = partition ? __fadd_rn(r[i][j], q[i][j]) : q[i][j];
                    q[i][j] = 0.0f;
                }
            ++partition;
            boundary = contract::partitionBegin(partition + 1, k);
        }
    };
    flush(0);

    const int tiles = (k + BK - 1) / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < tiles) stage(s, s * BK);
        commitAsync();
    }
    for (int t = 0; t < tiles; ++t)
    {
        waitAsync<STAGES - 2>();
        __syncthreads();
        {
            const int next = t + STAGES - 1;
            if (next < tiles) stage(next % STAGES, next * BK);
            commitAsync();
        }
        const float *ta = staged + (t % STAGES) * (A_STAGE / 4);
        const float *tb = staged + STAGES * (A_STAGE / 4) + (t % STAGES) * (B_STAGE / 4);
#pragma unroll
        for (int kq = 0; kq < BK; kq += 4)
        {
            float4 a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = *reinterpret_cast<const float4 *>(ta + (ty + i * TY) * PITCH + kq);
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = *reinterpret_cast<const float4 *>(tb + (tx + j * TX) * PITCH + kq);
            // Four ascending K steps; the unaligned variant checks the fold per element.
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
#pragma unroll
                for (int i = 0; i < TM; ++i)
                {
                    const float av = e == 0 ? a[i].x : e == 1 ? a[i].y : e == 2 ? a[i].z : a[i].w;
#pragma unroll
                    for (int j = 0; j < TN; ++j)
                    {
                        const float bv = e == 0 ? b[j].x : e == 1 ? b[j].y : e == 2 ? b[j].z : b[j].w;
                        q[i][j] = __fmaf_rn(av, bv, q[i][j]);
                    }
                }
                if constexpr (!Aligned) flush(t * BK + kq + e + 1);
            }
        }
        if constexpr (Aligned) flush((t + 1) * BK);
    }
    waitAsync<0>();

#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        const int row = m0 + ty + i * TY;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            const int col = c0 + tx + j * TX;
            if (col >= total) continue;
            int local = 0;
            const int p = projectionOf(g, col, local);
            g.output(p)[static_cast<size_t>(row) * g.columns(p) + local] = r[i][j];
        }
    }
}

/** @brief Whether the aligned variant applies: K % 4 == 0 and (K/8) % BK == 0. */
template<int BK>
inline constexpr bool sgemmAligned(int k) noexcept
{
    return k % 4 == 0 && k % contract::kPartitions == 0 && (k / contract::kPartitions) % BK == 0;
}

/**
 * @brief Slice factor for the final partial wave, in raster-tile time units.
 * @param remainder Raster tiles left after the full waves. @param slots Resident CTAs per wave.
 * @return 4 or 2 when every slice of the tail still runs in one round on its own
 *         SM (thinner slices are less efficient, so fuller tails stay whole), else 1.
 * @details Measured on GB10: splitting a tail that needs a second round of
 * slices is slower than one round of whole tiles.
 */
inline int tailSplit(int remainder, int slots) noexcept
{
    if (remainder <= 0) return 1;
    if (remainder * 4 <= slots) return 4;
    if (remainder * 2 <= slots) return 2;
    return 1;
}

/** @brief Dynamic shared bytes of one instantiation. */
template<int BM, int BN, int BK, int STAGES>
inline constexpr int sgemmSharedBytes = STAGES * (BM + BN) * (BK + 4) * 4;
} // namespace llaminar2::cuda::fixed_k_scalar_cuda
