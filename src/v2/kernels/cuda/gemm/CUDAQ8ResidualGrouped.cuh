/**
 * @file CUDAQ8ResidualGrouped.cuh
 * @brief Cooperative native-Q8 matrix tiles with two exact integer MMA terms.
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
}
