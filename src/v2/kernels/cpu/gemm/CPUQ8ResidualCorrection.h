/**
 * @file CPUQ8ResidualCorrection.h
 * @brief Native prepared-weight two-term Q8 publication and SIMD projection.
 *
 * Existing VNNI microkernels retain their canonical block contributions and
 * independent register accumulators; a typed store policy folds each real
 * row's two final terms once. Two real rows share immutable weight loads.
 * Output/scratch holes are untouched. The CPU authority validates the complete
 * live prefix before starting its workshare. The operation owns no hot heap
 * allocation; standalone OpenMP team runtime overhead is shared with SingleQ8.
 */
#pragma once
#include "CPUNativeVNNIGemv.h"
#include "kernels/common/Q8ResidualCorrection.h"
#include <span>
namespace llaminar2::cpu::native_vnni {
/** @brief Form one quantized residual against the first block's stored scale.
 * @param input Thirty-two original FP32 values.
 * @param primary Already published unchanged native primary Q8 block.
 * @param residual Exact secondary Q8 destination.
 * @param policy Existing prepared source's activation/reduction contract.
 */
inline void quantizeResidualBlock(const float*input,const Q8_1Block&primary,Q8_1Block&residual,
 CPUProjectionNumericalPolicy policy) {
 alignas(64)float values[32];const auto scale=_mm256_set1_ps(nativeVNNIFP16ScaleToFP32(primary.d));
 for(int i=0;i<32;i+=8) {
  const auto bytes=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(primary.qs+i));
  const auto decoded=persistRoundedAVX2(_mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes)),scale));
  _mm256_store_ps(values+i,_mm256_sub_ps(_mm256_loadu_ps(input+i),decoded));
 }
 quantizeActivationBlock(values,residual,32,policy);
}
/** @brief Validate and borrow actual immutable native source storage.
 * @param packed Exact prepared engine export; no repack or duplicate source exists.
 * @param m Positive physical row capacity.
 * @param n Exact output width.
 * @param k Exact reduction width.
 * @throws std::invalid_argument For an unsupported source tuple or incomplete shape.
 */
inline void validateCPUQ8ResidualSource(const CPUNativeVNNIPackedWeights&packed,int m,int n,int k) {
 validateQ8ResidualCorrectionGeometry(m,n,k);
 if(packed.codebook_id!=native_vnni_formats::Q8_0.codebook_id||!packed.usesExpandedInt8()||packed.is_asymmetric||packed.is_superblock||
    packed.N!=n||packed.K!=k||!packed.hasInterleavedData()||packed.numerical_policy!=CPUProjectionNumericalPolicy::BackendNative)
  throw std::invalid_argument("Two-term Q8 requires the exact prepared Q8_0 source, complete native reduction blocks and geometry");
}
/** @brief Execute the explicit paired-Q8 operation over one borrowed prepared source.
 * @param packed Actual prepared native source with fixed arithmetic policy.
 * @param input Distinct host-valid FP32 rows.
 * @param output Host-owned destination; only live rows are authored.
 * @param rows Immutable capacity and optional CPU-owned authoritative count.
 * @param primary Participant-owned first Q8 block bank.
 * @param residual Independently owned second Q8 block bank.
 */
inline void cpuQ8ResidualCorrectedProjection(const CPUNativeVNNIPackedWeights&packed,const float*input,
 float*output,DeviceRowRange rows,std::span<Q8_1Block>primary,std::span<Q8_1Block>residual) {
 const int physical=rows.physicalRows(),n=packed.N,k=packed.K,bpr=packed.blocks_per_row;
 validateCPUQ8ResidualSource(packed,physical,n,k);
 const auto needed=cpuProjectionQ8Blocks(physical,k);
 if(!input||!output||input==output||primary.size()<needed||residual.size()<needed||primary.data()==residual.data())
  throw std::invalid_argument("Two-term CPU Q8 requires distinct prepared operands and complete admitted primary/residual banks");
 const std::array writes{
  Q8ResidualStorageSpan{output,static_cast<std::size_t>(physical)*n*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{primary.data(),needed*sizeof(Q8_1Block),alignof(Q8_1Block)},
  Q8ResidualStorageSpan{residual.data(),needed*sizeof(Q8_1Block),alignof(Q8_1Block)}};
 const std::array reads{
  Q8ResidualStorageSpan{input,static_cast<std::size_t>(physical)*k*sizeof(float),alignof(float)},
  Q8ResidualStorageSpan{packed.native_interleaved.data(),packed.native_interleaved.size(),1},
  Q8ResidualStorageSpan{rows.countOwner(),rows.countOwner()?sizeof(std::int32_t):0,alignof(std::int32_t)}};
 validateQ8ResidualStorage(writes,reads);
 const int live=rows.activeRowsFor(rows.countOwner()?*rows.countOwner():rows.capacity());
 if(live<0)throw std::invalid_argument("Two-term CPU Q8 live count exceeds retained capacity");
 if(live==0)return;
 const bool avx512=activeISALevel()>=ISALevel::AVX512;
 const std::int64_t tiles=(static_cast<std::int64_t>(live)+1)/2,chunks=(static_cast<std::int64_t>(n)+63)/64;
 const auto finitePublication=[&]() {
  for(std::size_t block=0;block<static_cast<std::size_t>(live)*bpr;++block)
   if((primary[block].d&0x7c00u)==0x7c00u)return false;
  return true;
 };
 auto work=[&]() {
#pragma omp for schedule(static)
  for(int row=0;row<live;++row)for(int block=0;block<bpr;++block) {
   const auto offset=static_cast<std::size_t>(row)*bpr+block;
   const auto*source=input+static_cast<std::size_t>(row)*k+block*32;
   bool finite=true;for(int element=0;element<32;++element)finite=finite&&std::isfinite(source[element]);
   if(!finite){primary[offset].d=0x7c00u;continue;}
   quantizeActivationBlock(source,primary[offset],32,packed.numerical_policy);
   if((primary[offset].d&0x7c00u)!=0x7c00u)
    quantizeResidualBlock(source,primary[offset],residual[offset],packed.numerical_policy);
  }
  // The first workshare's implicit barrier authenticates every active primary
  // scale. All workers observe the same bank and skip output on any invalid
  // block, without a second flag authority or a throw inside this workshare.
  if(finitePublication()) {
#pragma omp for schedule(static)
  for(std::int64_t task=0;task<tiles*chunks;++task) {
   const auto row=task%tiles*2,chunk=task/tiles;
   const int valid=std::min<std::int64_t>(64,n-chunk*64);
   alignas(64) float tail0[64],tail1[64];
   float *destination0=valid==64?output+row*n+chunk*64:tail0;
   float *destination1=row+1<live?(valid==64?output+(row+1)*n+chunk*64:tail1):nullptr;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
   if(avx512&&row+1<live) {
    gemm_4row_non_nibble_2z_chunk<true>(packed,primary.data()+row*bpr,residual.data()+row*bpr,
      primary.data()+(row+1)*bpr,residual.data()+(row+1)*bpr,
      destination0,nullptr,destination1,nullptr,chunk,0,bpr,false);
    if(valid<64){std::copy_n(tail0,valid,output+row*n+chunk*64);std::copy_n(tail1,valid,output+(row+1)*n+chunk*64);}
    continue;
   }
#endif
   for(auto current=row;current<std::min<std::int64_t>(row+2,live);++current) {
    float*destination=current==row?destination0:destination1;
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VNNI__)
    if(avx512)gemm_2row_non_nibble_chunk<true>(packed,primary.data()+current*bpr,residual.data()+current*bpr,
      destination,nullptr,chunk,0,bpr,false);
    else
#endif
      gemm_2row_non_nibble_chunk_avx2<true>(packed,primary.data()+current*bpr,residual.data()+current*bpr,
        destination,nullptr,chunk,0,bpr,false);
    if(valid<64)std::copy_n(destination,valid,output+current*n+chunk*64);
   }
  }
  }
 };
 OMP_WORKSHARE_REGION(work);
 if(!finitePublication())throw std::invalid_argument("Two-term CPU Q8 activation is nonfinite or its native primary half scale overflows");
}
}
