/**
 * @file FixedKScalarProjectionBody.cuh
 * @brief Scalar-FMA fixed-K prefill projection body shared by CUDA and HIP.
 *
 * Evaluates `kernels/common/FixedKProjectionContract.h` with single-element
 * reduction blocks: eight K partitions fixed by K alone, each one IEEE fused
 * multiply-add chain from +0 over its elements in ascending order, folded left
 * to right with round-to-nearest adds. It serves strict FP32 operands on every
 * GPU and half-RNE operands on GPUs without matrix instructions (gfx906); half
 * values widen exactly, so each product is exact and the chain is identical to
 * a scalar FP32 FMA chain over the widened operands.
 *
 * One register-tiled body serves both modes, exactly as the half MMA family:
 *  - Tiled: a CTA walks all of K for its output tile and folds partitions in
 *    registers at each boundary (wide groups).
 *  - Partial: a CTA walks one partition (grid.z) and publishes its chain; the
 *    fold kernel applies P0 + P1 + ... + P7 (narrow groups).
 * Staged elements outside the walked range are zero; a zero product never
 * changes a chain that started from +0, so staging alignment never changes bytes.
 *
 * The body is plain CUDA/HIP C++: no tensor-core, warp-size or cp.async
 * assumption, so wave64 and warp32 devices evaluate identical arithmetic.
 */
#pragma once

#include "kernels/common/FixedHalfProjectionGroup.h"
#include "kernels/common/FixedKProjectionContract.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#else
#include <cuda_fp16.h>
#endif

namespace llaminar2::fixed_k_scalar
{
namespace contract = ::llaminar2::fixed_k_projection;

/** @brief Whether a CTA evaluates all partitions (Tiled) or one published partition (Partial). */
enum class Mode { Tiled, Partial };

/** @brief Exact FP32 value of one operand element (half words widen exactly). */
template<class Operand>
__device__ __forceinline__ float widenOperand(Operand value)
{
    if constexpr (std::is_same_v<Operand, float>)
        return value;
    else
        return __half2float(__ushort_as_half(static_cast<unsigned short>(value)));
}

/**
 * @brief Read four consecutive operands starting at a multiple-of-four element, widened to FP32.
 * @tparam Operand float (strict FP32) or uint16_t (half-RNE words).
 * @param base Row start. @param index First element (multiple of four).
 * @param k Row width; elements at or past K read as zero.
 * @return Exact FP32 values. When K is a multiple of four every row base and
 *         index is vector aligned; otherwise elements are read individually.
 */
template<class Operand>
__device__ __forceinline__ float4 loadFour(const Operand *base, int index, int k)
{
    if ((k & 3) == 0)
    {
        if constexpr (std::is_same_v<Operand, float>)
            return *reinterpret_cast<const float4 *>(base + index);
        else
        {
            const uint2 words = *reinterpret_cast<const uint2 *>(base + index);
            return make_float4(widenOperand<uint16_t>(words.x & 0xffffu), widenOperand<uint16_t>(words.x >> 16),
                               widenOperand<uint16_t>(words.y & 0xffffu), widenOperand<uint16_t>(words.y >> 16));
        }
    }
    float v[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) v[e] = index + e < k ? widenOperand(base[index + e]) : 0.0f;
    return make_float4(v[0], v[1], v[2], v[3]);
}

/**
 * @brief Resolve a compact group column to its projection and local column.
 * @param g Validated group. @param compact Column in [0, totalColumns()).
 * @param local Out: column within the owning projection. @return Projection index.
 */
template<class Group>
__device__ __forceinline__ int projectionOf(const Group &g, int compact, int &local)
{
    int p = 0;
    while (compact >= g.columnEnd(p)) ++p;
    local = compact - (p ? g.columnEnd(p - 1) : 0);
    return p;
}

/**
 * @brief Register-tiled scalar fixed-K projection.
 * @tparam Operand float or uint16_t half words. @tparam BM CTA rows. @tparam BN CTA compact columns.
 * @tparam TM Rows per thread. @tparam TN Columns per thread. @tparam BK Staged K elements (multiple of 4).
 * @tparam M_ Tiled register fold or Partial publication (partition = grid.z).
 * @tparam Aligned Host-selected when every partition boundary (multiples of K/8)
 *         is a staged-tile boundary and the walk origin is tile aligned; the
 *         partition fold then runs once per tile instead of once per element.
 * @param g Borrowed group. @param m Physical rows. @param k Reduction width (multiple of 8).
 * @param partials Partial mode only: [kPartitions][m][totalColumns] chains.
 *
 * Shared tiles are stored K-major (transposed) so each thread reads its row
 * and column fragments with broadcast-friendly loads. Two shared buffers and a
 * register prefetch overlap the next tile's global loads with this tile's FMAs.
 */
template<class Operand, int BM, int BN, int TM, int TN, int BK, Mode M_, bool Aligned = false>
__global__ __launch_bounds__((BM / TM) * (BN / TN)) void scalarProjectionKernel(
    const FixedRaggedProjectionGroup<Operand> g, int m, int k, float *__restrict__ partials)
{
    constexpr int THREADS = (BM / TM) * (BN / TN);
    constexpr int QUADS = BK / 4;
    static_assert(BK % 4 == 0 && BM % TM == 0 && BN % TN == 0);
    // Staging slots per thread; a final partial round leaves some threads idle.
    constexpr int A_LOADS = (BM * QUADS + THREADS - 1) / THREADS, B_LOADS = (BN * QUADS + THREADS - 1) / THREADS;
    __shared__ float sa[2][BK][BM];
    __shared__ float sb[2][BK][BN];

    const int total = g.totalColumns();
    const int tid = static_cast<int>(threadIdx.x);
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int m0 = static_cast<int>(blockIdx.x) * BM, c0 = static_cast<int>(blockIdx.y) * BN;
    const int first = M_ == Mode::Partial ? contract::partitionBegin(blockIdx.z, k) : 0;
    const int last = M_ == Mode::Partial ? contract::partitionBegin(blockIdx.z + 1, k) : k;
    const int walk_first = first - first % 4;  // 4-aligned staging origin

    const Operand *a_row[A_LOADS]; int a_slot_row[A_LOADS], a_quad[A_LOADS]; bool a_ok[A_LOADS];
    const Operand *b_row[B_LOADS]; int b_slot_row[B_LOADS], b_quad[B_LOADS]; bool b_ok[B_LOADS];
#pragma unroll
    for (int i = 0; i < A_LOADS; ++i)
    {
        const int c = tid + i * THREADS;
        const bool slot = c < BM * QUADS;
        a_slot_row[i] = slot ? c / QUADS : -1; a_quad[i] = c % QUADS;
        a_ok[i] = slot && m0 + a_slot_row[i] < m;
        a_row[i] = g.input() + static_cast<size_t>(a_ok[i] ? m0 + a_slot_row[i] : 0) * k;
    }
#pragma unroll
    for (int i = 0; i < B_LOADS; ++i)
    {
        const int c = tid + i * THREADS;
        const bool slot = c < BN * QUADS;
        b_slot_row[i] = slot ? c / QUADS : -1; b_quad[i] = c % QUADS;
        b_ok[i] = slot && c0 + b_slot_row[i] < total;
        int local = 0;
        const int p = b_ok[i] ? projectionOf(g, c0 + b_slot_row[i], local) : 0;
        b_row[i] = g.weight(p) + static_cast<size_t>(local) * k;
    }
    float4 a_stage[A_LOADS], b_stage[B_LOADS];
    // Elements outside [first, last) stage as zero: they never alter a chain.
    const auto masked = [&](float4 v, int index) {
        if (index < first || index >= last) v.x = 0.0f;
        if (index + 1 < first || index + 1 >= last) v.y = 0.0f;
        if (index + 2 < first || index + 2 >= last) v.z = 0.0f;
        if (index + 3 < first || index + 3 >= last) v.w = 0.0f;
        return v;
    };
    const auto fetch = [&](int k0) {
#pragma unroll
        for (int i = 0; i < A_LOADS; ++i)
        {
            const int index = k0 + a_quad[i] * 4;
            a_stage[i] = a_ok[i] && index < last ? masked(loadFour(a_row[i], index, k), index) : make_float4(0, 0, 0, 0);
        }
#pragma unroll
        for (int i = 0; i < B_LOADS; ++i)
        {
            const int index = k0 + b_quad[i] * 4;
            b_stage[i] = b_ok[i] && index < last ? masked(loadFour(b_row[i], index, k), index) : make_float4(0, 0, 0, 0);
        }
    };
    const auto publish = [&](int buffer) {
#pragma unroll
        for (int i = 0; i < A_LOADS; ++i)
        {
            if (a_slot_row[i] < 0) continue;
            sa[buffer][a_quad[i] * 4 + 0][a_slot_row[i]] = a_stage[i].x;
            sa[buffer][a_quad[i] * 4 + 1][a_slot_row[i]] = a_stage[i].y;
            sa[buffer][a_quad[i] * 4 + 2][a_slot_row[i]] = a_stage[i].z;
            sa[buffer][a_quad[i] * 4 + 3][a_slot_row[i]] = a_stage[i].w;
        }
#pragma unroll
        for (int i = 0; i < B_LOADS; ++i)
        {
            if (b_slot_row[i] < 0) continue;
            sb[buffer][b_quad[i] * 4 + 0][b_slot_row[i]] = b_stage[i].x;
            sb[buffer][b_quad[i] * 4 + 1][b_slot_row[i]] = b_stage[i].y;
            sb[buffer][b_quad[i] * 4 + 2][b_slot_row[i]] = b_stage[i].z;
            sb[buffer][b_quad[i] * 4 + 3][b_slot_row[i]] = b_stage[i].w;
        }
    };

    float q[TM][TN] = {}, r[TM][TN] = {};
    int partition = 0;
    int boundary = contract::partitionBegin(1, k);
    // Tiled mode: fold every partition ending at element `done` (empty ones add +0).
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
    if constexpr (M_ == Mode::Tiled) flush(0);

    int buffer = 0;
    fetch(walk_first);
    publish(0);
    __syncthreads();
    for (int k0 = walk_first; k0 < last; k0 += BK)
    {
        const bool more = k0 + BK < last;
        if (more) fetch(k0 + BK);
#pragma unroll
        for (int kk = 0; kk < BK; ++kk)
        {
            float a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = sa[buffer][kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = sb[buffer][kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    q[i][j] = __fmaf_rn(a[i], b[j], q[i][j]);
            if constexpr (M_ == Mode::Tiled && !Aligned) flush(k0 + kk + 1);
        }
        if constexpr (M_ == Mode::Tiled && Aligned) flush(k0 + BK);
        if (more)
        {
            publish(buffer ^ 1);
            __syncthreads();
            buffer ^= 1;
        }
    }

    float *out = M_ == Mode::Partial ? partials + static_cast<size_t>(blockIdx.z) * m * total : nullptr;
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        const int row = m0 + ty * TM + i;
        if (row >= m) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            const int col = c0 + tx * TN + j;
            if (col >= total) continue;
            if constexpr (M_ == Mode::Partial)
                out[static_cast<size_t>(row) * total + col] = q[i][j];
            else
            {
                int local = 0;
                const int p = projectionOf(g, col, local);
                g.output(p)[static_cast<size_t>(row) * g.columns(p) + local] = r[i][j];
            }
        }
    }
}


/**
 * @brief Whether the Aligned variant is exact for this K and staged width.
 * @tparam BK Staged K elements. @param k Reduction width (multiple of 8).
 * @return True when K/8 is a multiple of BK (all partition boundaries are tile ends).
 */
template<int BK>
inline constexpr bool scalarPartitionsAligned(int k) noexcept
{
    return k % contract::kPartitions == 0 && (k / contract::kPartitions) % BK == 0;
}

/**
 * @brief Fold published partition chains left to right into the group outputs.
 * @param g Borrowed group (compact column order). @param m Physical rows.
 * @param partials [kPartitions][m][totalColumns] chains from the partial mode.
 */
template<class Operand>
__global__ __launch_bounds__(256) void foldScalarPartialsKernel(
    const FixedRaggedProjectionGroup<Operand> g, int m, const float *__restrict__ partials)
{
    const int total = g.totalColumns();
    const size_t outputs = static_cast<size_t>(m) * total;
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= outputs) return;
    float result = partials[index];
#pragma unroll
    for (int part = 1; part < contract::kPartitions; ++part)
        result = __fadd_rn(result, partials[part * outputs + index]);
    int local = 0;
    const int p = projectionOf(g, static_cast<int>(index % total), local);
    g.output(p)[(index / total) * g.columns(p) + local] = result;
}
} // namespace llaminar2::fixed_k_scalar
