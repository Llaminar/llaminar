/**
 * @file CUDAQ8ResidualGrouped.cuh
 * @brief Cooperative original-native matrix tiles with two exact integer MMA terms.
 *
 * This borrows the installed NativeVNNI ldmatrix/MMA mapping and float contribution
 * authority. One CTA stages unchanged B once for64 real rows and both terms.
 * Each K32 group completes independently, then joins two separate ascending
 * FP32 folds. Local zero fill covers only shared-memory arithmetic padding;
 * inactive global input, scratch and output bytes are never accessed.
 */
#pragma once
#include "kernels/cuda/gemm/CUDANativeVNNIPrefillDevice.cuh"
#include "kernels/common/Q8ResidualCorrection.h"
#include "kernels/common/DeviceNativeVNNIContributionContract.h"
#include "kernels/common/Q8ResidualGroupedGeometry.h"
namespace llaminar2::cuda_q8_residual {
/**
 * @brief Apply one64×64 tile through the canonical signed-INT8 MMA primitive.
 * @param a Borrowed prepared source, exact count authority and four admitted banks.
 *
 * Warp rows16 and columns32 bound each thread to sixteen outputs per term.
 * Two CTA barriers protect shared producer publication and old-reader retirement.
 * There are no global partials, atomics, partition sums or output-sized scratch.
 */
__global__ __launch_bounds__(256,2) void projectGrouped(GPUQ8ResidualCorrectionArgs a) {
 constexpr int Rows=64,Cols=64,Stride=12;
 const auto columnTiles=(static_cast<unsigned>(a.weights.n)+Cols-1)/Cols;
 const auto rowBase=static_cast<std::uint64_t>(blockIdx.x/columnTiles)*Rows;
 const unsigned columnBase=(blockIdx.x%columnTiles)*Cols;
 const int live=a.rows.activeRows();
 if(rowBase>=static_cast<unsigned>(live))return;
 __shared__ __align__(16) int primary[Rows][Stride],residual[Rows][Stride],weights[Cols][Stride];
 __shared__ float primaryScale[Rows],residualScale[Rows],weightScale[Cols];
 const int lane=threadIdx.x%32,warp=threadIdx.x/32;
 const int warpRow=(warp/2)*16,warpColumn=(warp%2)*32;
 float sum0[4][4]{},sum1[4][4]{};
 for(unsigned group=0;group<a.weights.blocks_per_row;++group) {
  if(threadIdx.x<Rows*2) {
   const unsigned local=threadIdx.x/2,half=threadIdx.x%2;
   const auto row=rowBase+local;
   const auto activation=(row*a.weights.blocks_per_row+group)*32;
   const int4 p=row<static_cast<unsigned>(live)?reinterpret_cast<const int4*>(a.primary+activation)[half]:make_int4(0,0,0,0);
   *reinterpret_cast<int4*>(&primary[local][half*4])=p;
   const int4 r=row<static_cast<unsigned>(live)?reinterpret_cast<const int4*>(a.residual+activation)[half]:make_int4(0,0,0,0);
   *reinterpret_cast<int4*>(&residual[local][half*4])=r;
   const unsigned column=columnBase+local;
   const auto block=(static_cast<std::uint64_t>(group)*a.weights.n+column)*32;
   const int4 w=column<static_cast<unsigned>(a.weights.n)?reinterpret_cast<const int4*>(a.weights.payload+block)[half]:make_int4(0,0,0,0);
   *reinterpret_cast<int4*>(&weights[local][half*4])=w;
  }
  if(threadIdx.x<Rows) {
   const auto row=rowBase+threadIdx.x;
   const auto block=row*a.weights.blocks_per_row+group;
   primaryScale[threadIdx.x]=row<static_cast<unsigned>(live)?a.primaryScales[block]:0.f;
   residualScale[threadIdx.x]=row<static_cast<unsigned>(live)?a.residualScales[block]:0.f;
   const unsigned column=columnBase+threadIdx.x;
   weightScale[threadIdx.x]=column<static_cast<unsigned>(a.weights.n)?
    __half2float(__ushort_as_half(static_cast<const unsigned short*>(a.weights.scales)[static_cast<std::uint64_t>(group)*a.weights.n+column])):0.f;
  }
  __syncthreads();
  std::uint32_t af0[4],af1[4];
  cuda::prefill::load_ldmatrix_a_m16n8k32(af0,&primary[warpRow][0],Stride,lane);
  cuda::prefill::load_ldmatrix_a_m16n8k32(af1,&residual[warpRow][0],Stride,lane);
#pragma unroll
  for(int fragment=0;fragment<4;++fragment) {
   const int firstColumn=warpColumn+fragment*8;
   std::uint32_t bf[2];cuda::prefill::load_ldmatrix_b_m16n8k32(bf,&weights[firstColumn][0],Stride,lane);
   std::int32_t dot0[4]{},dot1[4]{};
   cuda::prefill::mma_m16n8k32_s8(dot0,af0,bf);
   cuda::prefill::mma_m16n8k32_s8(dot1,af1,bf);
#pragma unroll
   for(int element=0;element<4;++element) {
    const int row=warpRow+cuda::prefill::frag_row(lane,element);
    const int column=firstColumn+cuda::prefill::frag_col(lane,element);
    sum0[fragment][element]=device_fp32_contract::add(sum0[fragment][element],
     device_native_vnni_contract::singleScaleBlock(dot0[element],weightScale[column],primaryScale[row]));
    sum1[fragment][element]=device_fp32_contract::add(sum1[fragment][element],
     device_native_vnni_contract::singleScaleBlock(dot1[element],weightScale[column],residualScale[row]));
   }
  }
  __syncthreads();
 }
#pragma unroll
 for(int fragment=0;fragment<4;++fragment) {
#pragma unroll
  for(int element=0;element<4;++element) {
   const auto row=rowBase+warpRow+cuda::prefill::frag_row(lane,element);
   const unsigned column=columnBase+warpColumn+fragment*8+cuda::prefill::frag_col(lane,element);
   if(row<static_cast<unsigned>(live)&&column<static_cast<unsigned>(a.weights.n))
    a.output[row*a.weights.n+column]=device_fp32_contract::add(sum0[fragment][element],sum1[fragment][element]);
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
    constexpr int Rows = 64, Words = 8;
    constexpr int Cols = q8ResidualCooperativeColumns(Q8ResidualGroupedBackend::CUDA, C);
    constexpr int Fragments = Cols / 16;
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
    constexpr int Stride = 12, Planes = Traits::is_dual_scale ? 2 : 1;
    __shared__ __align__(16) int primary[Rows][Stride], residual[Rows][Stride], weights[Planes][Cols][Stride];
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int warpRow = (warp / 2) * 16, warpColumn = (warp % 2) * (Cols / 2);
    float sum0[Fragments][4]{}, sum1[Fragments][4]{};
    for (unsigned group = 0; group < a.weights.blocks_per_row; ++group)
    {
        if (threadIdx.x < Rows * 2)
        {
            const unsigned local = threadIdx.x / 2, half = threadIdx.x % 2;
            const auto row = rowBase + local, index = row * a.weights.blocks_per_row + group;
            const int4 p = row < static_cast<unsigned>(live)
                               ? reinterpret_cast<const int4 *>(a.primary + index * 32)[half]
                               : make_int4(0, 0, 0, 0);
            const int4 q = row < static_cast<unsigned>(live)
                               ? reinterpret_cast<const int4 *>(a.residual + index * 32)[half]
                               : make_int4(0, 0, 0, 0);
            *reinterpret_cast<int4 *>(&primary[local][half * 4]) = p;
            *reinterpret_cast<int4 *>(&residual[local][half * 4]) = q;
            if constexpr (Traits::is_asymmetric || Traits::is_dual_scale_asym || Traits::is_iq1_m)
            {
                primarySums[half * 2][local] = __dp4a(p.y, 0x01010101, __dp4a(p.x, 0x01010101, 0));
                primarySums[half * 2 + 1][local] = __dp4a(p.w, 0x01010101, __dp4a(p.z, 0x01010101, 0));
                residualSums[half * 2][local] = __dp4a(q.y, 0x01010101, __dp4a(q.x, 0x01010101, 0));
                residualSums[half * 2 + 1][local] = __dp4a(q.w, 0x01010101, __dp4a(q.z, 0x01010101, 0));
            }
            if (half == 0)
            {
                primaryScale[local] = row < static_cast<unsigned>(live) ? a.primaryScales[index] : 0.f;
                residualScale[local] = row < static_cast<unsigned>(live) ? a.residualScales[index] : 0.f;
            }
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
                if constexpr (Traits::is_dual_scale)
                {
                    weights[0][threadIdx.x][word] = word < 4 ? w[word] : 0;
                    weights[1][threadIdx.x][word] = word >= 4 ? w[word] : 0;
                }
                else
                    weights[0][threadIdx.x][word] = w[word];
            }
            scales[threadIdx.x] = meta.scale;
            secondary[threadIdx.x] = meta.secondary;
            emins[threadIdx.x] = meta.emin;
            signs[threadIdx.x] = meta.qh;
        }
        __syncthreads();
        // Consume the two terms in disjoint register lifetimes. The B fragment
        // is shared, while only one A fragment, half-dot pair and subgroup-sum
        // set is live. IQ1_M needs all four correction sums; retaining both
        // terms' temporaries together exceeded the two-CTA register envelope.
#pragma unroll
        for (int fragment = 0; fragment < Fragments; ++fragment)
        {
            const int firstColumn = warpColumn + fragment * 8;
            std::uint32_t lowWeights[2], highWeights[2];
            cuda::prefill::load_ldmatrix_b_m16n8k32(
                lowWeights, &weights[0][firstColumn][0], Stride, lane);
            if constexpr (Traits::is_dual_scale)
                cuda::prefill::load_ldmatrix_b_m16n8k32(
                    highWeights, &weights[1][firstColumn][0], Stride, lane);
            {
                std::uint32_t activations[4];
                int lowDot[4]{}, highDot[4]{};
                cuda::prefill::load_ldmatrix_a_m16n8k32(
                    activations, &primary[warpRow][0], Stride, lane);
                cuda::prefill::mma_m16n8k32_s8(lowDot, activations, lowWeights);
                if constexpr (Traits::is_dual_scale)
                    cuda::prefill::mma_m16n8k32_s8(highDot, activations, highWeights);
#pragma unroll
                for (int element = 0; element < 4; ++element)
                {
                    const int row = warpRow + cuda::prefill::frag_row(lane, element);
                    const int column = firstColumn + cuda::prefill::frag_col(lane, element);
                    NativeBlock<C> meta;
                    meta.scale = scales[column];
                    meta.secondary = secondary[column];
                    meta.emin = emins[column];
                    meta.qh = signs[column];
                    int subgroupSums[4]{};
                    if constexpr (Traits::is_asymmetric || Traits::is_dual_scale_asym || Traits::is_iq1_m)
                    {
#pragma unroll
                        for (int part = 0; part < 4; ++part)
                            subgroupSums[part] = primarySums[part][row];
                    }
                    sum0[fragment][element] = device_fp32_contract::add(
                        sum0[fragment][element], nativeContribution<C>(decoder, meta, lowDot[element],
                            highDot[element], primaryScale[row], subgroupSums));
                }
            }
            {
                std::uint32_t activations[4];
                int lowDot[4]{}, highDot[4]{};
                cuda::prefill::load_ldmatrix_a_m16n8k32(
                    activations, &residual[warpRow][0], Stride, lane);
                cuda::prefill::mma_m16n8k32_s8(lowDot, activations, lowWeights);
                if constexpr (Traits::is_dual_scale)
                    cuda::prefill::mma_m16n8k32_s8(highDot, activations, highWeights);
#pragma unroll
                for (int element = 0; element < 4; ++element)
                {
                    const int row = warpRow + cuda::prefill::frag_row(lane, element);
                    const int column = firstColumn + cuda::prefill::frag_col(lane, element);
                    NativeBlock<C> meta;
                    meta.scale = scales[column];
                    meta.secondary = secondary[column];
                    meta.emin = emins[column];
                    meta.qh = signs[column];
                    int subgroupSums[4]{};
                    if constexpr (Traits::is_asymmetric || Traits::is_dual_scale_asym || Traits::is_iq1_m)
                    {
#pragma unroll
                        for (int part = 0; part < 4; ++part)
                            subgroupSums[part] = residualSums[part][row];
                    }
                    sum1[fragment][element] = device_fp32_contract::add(
                        sum1[fragment][element], nativeContribution<C>(decoder, meta, lowDot[element],
                            highDot[element], residualScale[row], subgroupSums));
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int fragment = 0; fragment < Fragments; ++fragment)
    {
#pragma unroll
        for (int element = 0; element < 4; ++element)
        {
            const auto row = rowBase + warpRow + cuda::prefill::frag_row(lane, element);
            const unsigned column =
                columnBase + warpColumn + fragment * 8 + cuda::prefill::frag_col(lane, element);
            if (row < static_cast<unsigned>(live) && column < static_cast<unsigned>(a.weights.n))
                a.output[row * a.weights.n + column] =
                    device_fp32_contract::add(sum0[fragment][element], sum1[fragment][element]);
        }
    }
}

}
