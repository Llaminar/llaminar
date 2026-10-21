/**
 * @file ROCmQ8ResidualGrouped.cuh
 * @brief gfx906 native grouped two-term dots with cooperative operand reuse.
 *
 * The installed NativeVNNI group maps four rows and adjacent columns to each
 * worker. This Q8-only specialization stages both original activation banks
 * and unchanged native weight words once per32-row tile. Independent integer
 * block dots feed the canonical K32 FP32 contribution/fold and one final sum.
 * No private arrays spill to global memory or output-sized partials are needed.
 */
#pragma once
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
     dot0[row][column]=__builtin_amdgcn_sdot4(p[row],w[column],dot0[row][column],false);
     dot1[row][column]=__builtin_amdgcn_sdot4(r[row],w[column],dot1[row][column],false);
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
}
