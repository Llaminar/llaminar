/**
 * @file ROCmQ8ResidualGrouped.cuh
 * @brief gfx906 native grouped two-term dots with cooperative operand reuse.
 *
 * The installed NativeVNNI group maps four rows and adjacent columns to each
 * worker. The retained Q8 fast specialization stages both original activation banks
 * and unchanged native weight words once per32-row tile. Independent integer
 * block dots feed the canonical K32 FP32 contribution/fold and one final sum.
 * No private arrays spill to global memory or output-sized partials are needed.
 * Native signed INT8 dot products share the exact unsaturated arithmetic
 * contract across gfx906 and gfx1100; lane ownership and reductions are unchanged.
 */
#pragma once
#include "kernels/rocm/common/ROCmSignedDot4.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/common/DeviceNativeVNNIContributionContract.h"
namespace llaminar2::rocm_q8_residual {
/**
 * @brief Compute one32×64 native-dot tile over the actual live rows only.
 * @param a Borrowed prepared source, device count and four persistent Q8 banks.
 *
 * K-word-major LDS permits broadcast A and B reads across the source-owned
 * grouped thread mapping. A producer join and reader-retirement join protect
 * each shared slot; no host synchronization or live-count readback is involved.
 */
__global__ __launch_bounds__(256,2) void projectGrouped(GPUQ8ResidualCorrectionArgs a) {
 constexpr int Rows=32,Cols=64,Words=8;
 const auto columnTiles=(static_cast<unsigned>(a.weights.n)+Cols-1)/Cols;
 const auto rowBase=static_cast<std::uint64_t>(blockIdx.x/columnTiles)*Rows;
 const unsigned columnBase=(blockIdx.x%columnTiles)*Cols;
 const int live=a.rows.activeRows();if(rowBase>=static_cast<unsigned>(live))return;
 __shared__ int primary[Words][Rows],residual[Words][Rows],weights[Words][Cols];
 __shared__ float primaryScale[Rows],residualScale[Rows],weightScale[Cols];
 const int localRow=(threadIdx.x%8)*4,localColumn=(threadIdx.x/8)*2;
 float sum0[4][2]{},sum1[4][2]{};
 for(unsigned group=0;group<a.weights.blocks_per_row;++group) {
  if(threadIdx.x<Rows) {
   const auto row=rowBase+threadIdx.x;
   const auto block=row*a.weights.blocks_per_row+group;
#pragma unroll
   for(int word=0;word<Words;++word) {
    primary[word][threadIdx.x]=row<static_cast<unsigned>(live)?reinterpret_cast<const int*>(a.primary+block*32)[word]:0;
    residual[word][threadIdx.x]=row<static_cast<unsigned>(live)?reinterpret_cast<const int*>(a.residual+block*32)[word]:0;
   }
   primaryScale[threadIdx.x]=row<static_cast<unsigned>(live)?a.primaryScales[block]:0.f;
   residualScale[threadIdx.x]=row<static_cast<unsigned>(live)?a.residualScales[block]:0.f;
  }
  if(threadIdx.x<Cols) {
   const unsigned column=columnBase+threadIdx.x;
   const auto block=static_cast<std::uint64_t>(group)*a.weights.n+column;
#pragma unroll
   for(int word=0;word<Words;++word)
    weights[word][threadIdx.x]=column<static_cast<unsigned>(a.weights.n)?reinterpret_cast<const int*>(a.weights.payload+block*32)[word]:0;
   weightScale[threadIdx.x]=column<static_cast<unsigned>(a.weights.n)?
    __half2float(__ushort_as_half(static_cast<const unsigned short*>(a.weights.scales)[block])):0.f;
  }
  __syncthreads();
  // Outer products reuse one packed word across all independent outputs.
  // The earlier row/column-first loop made LDS reread both activations for
  // every column and the same weight for every row. Exact integer dots may
  // interleave independently; only each output's K32 FP32 fold is ordered.
  int dot0[4][2]{},dot1[4][2]{};
#pragma clang loop unroll_count(2)
  for(int word=0;word<Words;++word) {
   int p[4],r[4],w[2];
#pragma unroll
   for(int row=0;row<4;++row) {
    p[row]=primary[word][localRow+row];
    r[row]=residual[word][localRow+row];
   }
#pragma unroll
   for(int column=0;column<2;++column)
    w[column]=weights[word][localColumn+column];
#pragma unroll
   for(int row=0;row<4;++row) {
#pragma unroll
    for(int column=0;column<2;++column) {
     dot0[row][column]=::llaminar2::rocm_integer_contract::signedDot4(p[row],w[column],dot0[row][column]);
     dot1[row][column]=::llaminar2::rocm_integer_contract::signedDot4(r[row],w[column],dot1[row][column]);
    }
   }
  }
  float pScale[4],rScale[4],wScale[2];
#pragma unroll
  for(int row=0;row<4;++row) {
   pScale[row]=primaryScale[localRow+row];
   rScale[row]=residualScale[localRow+row];
  }
#pragma unroll
  for(int column=0;column<2;++column)wScale[column]=weightScale[localColumn+column];
#pragma unroll
  for(int row=0;row<4;++row) {
#pragma unroll
   for(int column=0;column<2;++column) {
    sum0[row][column]=device_fp32_contract::add(sum0[row][column],
     device_native_vnni_contract::singleScaleBlock(dot0[row][column],wScale[column],pScale[row]));
    sum1[row][column]=device_fp32_contract::add(sum1[row][column],
     device_native_vnni_contract::singleScaleBlock(dot1[row][column],wScale[column],rScale[row]));
   }
  }
  __syncthreads();
 }
#pragma unroll
 for(int row=0;row<4;++row) {
#pragma unroll
  for(int column=0;column<2;++column) {
   const auto mr=rowBase+localRow+row;const unsigned nc=columnBase+localColumn+column;
   if(mr<static_cast<unsigned>(live)&&nc<static_cast<unsigned>(a.weights.n))
    a.output[mr*a.weights.n+nc]=device_fp32_contract::add(sum0[row][column],sum1[row][column]);
  }
 }
}
/**
 * @brief Cooperative native-codebook tile, sharing original weights and both activation terms.
 * @tparam C Exact native physical codebook, including raw source factors and IQ signs.
 * @param a Original prepared planes and device-owned live row range.
 * @param tables Cold original-generation IQ view; no publication occurs here.
 *
 * Every K32 dot is exact integer arithmetic. Half dots and subgroup sums retain
 * the canonical asymmetric/IQ contribution tree; each term has its own increasing
 * K fold and the two complete terms are added only once. Inactive rows are never
 * read or written globally. LDS-only zero lanes are arithmetic tile padding.
 */
template <std::uint8_t C>
__global__ __launch_bounds__(256, 2) void projectNativeGrouped(GPUQ8ResidualCorrectionArgs a,
                                                               NativeEmbeddingIQTables tables)
{
    using Traits = NativeDecoder::CodebookTraits<C>;
    constexpr int Rows = 32, Cols = 64, Words = 8;
    const auto columnTiles = (static_cast<unsigned>(a.weights.n) + Cols - 1) / Cols;
    const auto rowBase = static_cast<std::uint64_t>(blockIdx.x / columnTiles) * Rows;
    const unsigned columnBase = (blockIdx.x % columnTiles) * Cols;
    const int live = a.rows.activeRows();
    if (rowBase >= static_cast<unsigned>(live))
        return;
    NativeDecoder decoder(tables);
    __shared__ float primaryScale[Rows], residualScale[Rows];
    __shared__ int primarySums[4][Rows], residualSums[4][Rows];
    __shared__ std::uint16_t scales[Cols], secondary[Cols], signs[Cols];
    __shared__ std::uint32_t emins[Cols];
    __shared__ int primary[Words][Rows], residual[Words][Rows], weights[Words][Cols];
    const int localRow = (threadIdx.x % 8) * 4, localColumn = (threadIdx.x / 8) * 2;
    float sum0[4][2]{}, sum1[4][2]{};
    for (unsigned group = 0; group < a.weights.blocks_per_row; ++group)
    {
        if (threadIdx.x < Rows)
        {
            const auto row = rowBase + threadIdx.x, index = row * a.weights.blocks_per_row + group;
            int sp[4]{}, sq[4]{};
#pragma unroll
            for (int word = 0; word < Words; ++word)
            {
                const int p = row < static_cast<unsigned>(live)
                                  ? reinterpret_cast<const int *>(a.primary + index * 32)[word]
                                  : 0;
                const int q = row < static_cast<unsigned>(live)
                                  ? reinterpret_cast<const int *>(a.residual + index * 32)[word]
                                  : 0;
                primary[word][threadIdx.x] = p;
                residual[word][threadIdx.x] = q;
                if constexpr (Traits::is_asymmetric || Traits::is_dual_scale_asym || Traits::is_iq1_m)
                {
                    sp[word / 2] = ::llaminar2::rocm_integer_contract::signedDot4(p, 0x01010101, sp[word / 2]);
                    sq[word / 2] = ::llaminar2::rocm_integer_contract::signedDot4(q, 0x01010101, sq[word / 2]);
                }
            }
            if constexpr (Traits::is_asymmetric || Traits::is_dual_scale_asym || Traits::is_iq1_m)
            {
#pragma unroll
                for (int part = 0; part < 4; ++part)
                {
                    primarySums[part][threadIdx.x] = sp[part];
                    residualSums[part][threadIdx.x] = sq[part];
                }
            }
            primaryScale[threadIdx.x] = row < static_cast<unsigned>(live) ? a.primaryScales[index] : 0.f;
            residualScale[threadIdx.x] = row < static_cast<unsigned>(live) ? a.residualScales[index] : 0.f;
        }
        if (threadIdx.x < Cols)
        {
            const unsigned column = columnBase + threadIdx.x;
            int w[8]{};
            NativeBlock<C> meta;
            if (column < static_cast<unsigned>(a.weights.n))
            {
                const auto block = static_cast<std::uint64_t>(group) * a.weights.n + column;
                const auto *payload = a.weights.payload + block * Traits::payload_bytes;
                decoder.decode_groups<C>(payload, w);
                meta = NativeBlock<C>(a.weights, block, payload);
            }
#pragma unroll
            for (int word = 0; word < Words; ++word)
            {
                weights[word][threadIdx.x] = w[word];
            }
            scales[threadIdx.x] = meta.scale;
            secondary[threadIdx.x] = meta.secondary;
            emins[threadIdx.x] = meta.emin;
            signs[threadIdx.x] = meta.qh;
        }
        __syncthreads();
        int dot0[2][4][2]{}, dot1[2][4][2]{};
#pragma clang loop unroll_count(2)
        for (int word = 0; word < Words; ++word)
        {
            const int half = Traits::is_dual_scale ? word / 4 : 0;
            int p[4], q[4], w[2];
#pragma unroll
            for (int row = 0; row < 4; ++row)
            {
                p[row] = primary[word][localRow + row];
                q[row] = residual[word][localRow + row];
            }
#pragma unroll
            for (int column = 0; column < 2; ++column)
                w[column] = weights[word][localColumn + column];
#pragma unroll
            for (int row = 0; row < 4; ++row)
            {
#pragma unroll
                for (int column = 0; column < 2; ++column)
                {
                    dot0[half][row][column] =
                        ::llaminar2::rocm_integer_contract::signedDot4(p[row], w[column], dot0[half][row][column]);
                    dot1[half][row][column] =
                        ::llaminar2::rocm_integer_contract::signedDot4(q[row], w[column], dot1[half][row][column]);
                }
            }
        }
#pragma unroll
        for (int row = 0; row < 4; ++row)
        {
            int sp[4]{}, sq[4]{};
            if constexpr (Traits::is_asymmetric || Traits::is_dual_scale_asym || Traits::is_iq1_m)
            {
#pragma unroll
                for (int part = 0; part < 4; ++part)
                {
                    sp[part] = primarySums[part][localRow + row];
                    sq[part] = residualSums[part][localRow + row];
                }
            }
#pragma unroll
            for (int column = 0; column < 2; ++column)
            {
                const int c = localColumn + column;
                NativeBlock<C> meta;
                meta.scale = scales[c];
                meta.secondary = secondary[c];
                meta.emin = emins[c];
                meta.qh = signs[c];
                sum0[row][column] = device_fp32_contract::add(
                    sum0[row][column],
                    nativeContribution<C>(decoder, meta, dot0[0][row][column], dot0[1][row][column],
                                          primaryScale[localRow + row], sp));
                sum1[row][column] = device_fp32_contract::add(
                    sum1[row][column],
                    nativeContribution<C>(decoder, meta, dot1[0][row][column], dot1[1][row][column],
                                          residualScale[localRow + row], sq));
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int row = 0; row < 4; ++row)
    {
#pragma unroll
        for (int column = 0; column < 2; ++column)
        {
            const auto r = rowBase + localRow + row;
            const unsigned c = columnBase + localColumn + column;
            if (r < static_cast<unsigned>(live) && c < static_cast<unsigned>(a.weights.n))
                a.output[r * a.weights.n + c] =
                    device_fp32_contract::add(sum0[row][column], sum1[row][column]);
        }
    }
}

}
